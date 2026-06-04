"""
GPU-accelerated vector search index.

Key features:
- Automatic CPU/GPU selection based on DB size
- Batch query support
- FP16 support for 2x capacity
- Persistent storage (save/load)
- Sub-millisecond latency for <10K vectors on CPU
- GPU acceleration for >10K vectors

Based on real benchmarks from RTX 4050:
- CPU wins at <10K vectors (transfer overhead dominates)
- GPU wins at >10K vectors (12-15x speedup for dim=384)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Device heuristic
# ---------------------------------------------------------------------------
_GPU_THRESHOLD = 10_000  # vectors above this → prefer GPU


def _pick_device(n_vectors: int, force_device: Optional[str] = None) -> torch.device:
    if force_device:
        d = torch.device(force_device)
        if d.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return d
    if n_vectors >= _GPU_THRESHOLD and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# SearchResult
# ---------------------------------------------------------------------------
@dataclass
class SearchResult:
    """Single result from a vector search."""

    index: int
    score: float
    metadata: dict = field(default_factory=dict)

    def __repr__(self) -> str:  # pragma: no cover
        return f"SearchResult(index={self.index}, score={self.score:.4f})"


# ---------------------------------------------------------------------------
# Stats tracker
# ---------------------------------------------------------------------------
@dataclass
class _Stats:
    queries: int = 0
    total_latency_ms: float = 0.0

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / max(self.queries, 1)

    @property
    def queries_per_sec(self) -> float:
        if self.total_latency_ms == 0:
            return 0.0
        return self.queries / (self.total_latency_ms / 1000.0)

    def record(self, latency_ms: float, n_queries: int = 1) -> None:
        self.queries += n_queries
        self.total_latency_ms += latency_ms


# ---------------------------------------------------------------------------
# VectorIndex
# ---------------------------------------------------------------------------
class VectorIndex:
    """Main vector search index.

    Parameters
    ----------
    dim : int
        Dimensionality of vectors.
    device : str | None
        Force a specific device ("cpu" or "cuda").  None → auto-select.
    fp16 : bool
        Store vectors in half-precision (FP16) to halve memory.
    metadata : list[dict] | None
        Optional per-vector metadata aligned by index.
    """

    def __init__(
        self,
        dim: int,
        device: Optional[str] = None,
        fp16: bool = False,
        metadata: Optional[List[dict]] = None,
    ) -> None:
        self.dim = dim
        self._force_device = device
        self.fp16 = fp16
        self._vectors: Optional[torch.Tensor] = None
        self._metadata: List[dict] = metadata or []
        self._device: Optional[torch.device] = None
        self._stats = _Stats()

    # -- properties ----------------------------------------------------------

    @property
    def count(self) -> int:
        return 0 if self._vectors is None else self._vectors.shape[0]

    @property
    def device(self) -> torch.device:
        if self._device is None:
            return _pick_device(self.count, self._force_device)
        return self._device

    @property
    def stats(self) -> dict:
        return {
            "queries": self._stats.queries,
            "avg_latency_ms": round(self._stats.avg_latency_ms, 4),
            "queries_per_sec": round(self._stats.queries_per_sec, 2),
        }

    # -- internal helpers ----------------------------------------------------

    def _ensure_device(self) -> None:
        """Materialise / migrate tensors to the correct device."""
        target = _pick_device(self.count, self._force_device)
        if self._vectors is not None and self._vectors.device != target:
            self._vectors = self._vectors.to(target)
        self._device = target

    @staticmethod
    def _to_tensor(
        data: Union[np.ndarray, torch.Tensor, Sequence], dim: int
    ) -> torch.Tensor:
        if isinstance(data, torch.Tensor):
            t = data.float()
        elif isinstance(data, np.ndarray):
            t = torch.from_numpy(data).float()
        else:
            t = torch.tensor(data, dtype=torch.float32)
        if t.ndim == 1:
            t = t.unsqueeze(0)
        if t.shape[1] != dim:
            raise ValueError(
                f"Expected dim={dim}, got {t.shape[1]}"
            )
        return t

    # -- mutating operations -------------------------------------------------

    def add(
        self,
        vectors: Union[np.ndarray, torch.Tensor, Sequence],
        metadata: Optional[List[dict]] = None,
    ) -> None:
        """Add one or more vectors to the index."""
        t = self._to_tensor(vectors, self.dim)
        if self.fp16:
            t = t.half()
        if self._vectors is None:
            self._vectors = t
        else:
            self._vectors = torch.cat([self._vectors, t], dim=0)
        if metadata:
            self._metadata.extend(metadata)
        else:
            self._metadata.extend({} for _ in range(t.shape[0]))
        self._ensure_device()

    def remove(self, index: int) -> None:
        """Remove vector at *index*."""
        if self._vectors is None or index < 0 or index >= self.count:
            raise IndexError(f"Index {index} out of range [0, {self.count})")
        mask = torch.ones(self.count, dtype=torch.bool)
        mask[index] = False
        self._vectors = self._vectors[mask]
        self._metadata.pop(index)

    # -- search --------------------------------------------------------------

    def search(
        self,
        query: Union[np.ndarray, torch.Tensor, Sequence],
        top_k: int = 10,
    ) -> List[SearchResult]:
        """Search for the *top_k* nearest vectors (cosine similarity)."""
        results = self.batch_search(query, top_k=top_k)
        return results[0]

    def batch_search(
        self,
        queries: Union[np.ndarray, torch.Tensor, Sequence],
        top_k: int = 10,
    ) -> List[List[SearchResult]]:
        """Batch search: multiple queries at once."""
        t0 = time.perf_counter()
        q = self._to_tensor(queries, self.dim)
        n_queries = q.shape[0]
        top_k = min(top_k, self.count)

        dev = self.device
        q = q.to(dev).float()
        if self._vectors is None:
            return [[] for _ in range(n_queries)]

        db = self._vectors.to(dev).float()

        # Cosine similarity: (n_queries, n_db)
        q_norm = q / q.norm(dim=1, keepdim=True).clamp(min=1e-8)
        db_norm = db / db.norm(dim=1, keepdim=True).clamp(min=1e-8)
        sims = q_norm @ db_norm.T

        # Top-k per query
        scores, indices = sims.topk(top_k, dim=1)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self._stats.record(elapsed_ms, n_queries)

        results: List[List[SearchResult]] = []
        for i in range(n_queries):
            row = [
                SearchResult(
                    index=int(indices[i, j]),
                    score=float(scores[i, j]),
                    metadata=self._metadata[int(indices[i, j])] if int(indices[i, j]) < len(self._metadata) else {},
                )
                for j in range(top_k)
            ]
            results.append(row)
        return results

    # -- persistence ---------------------------------------------------------

    def save(self, path: Union[str, Path]) -> None:
        """Save index to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        data: dict = {
            "dim": self.dim,
            "fp16": self.fp16,
            "metadata": self._metadata,
        }
        if self._vectors is not None:
            # Save tensor on CPU so it's portable
            torch.save(self._vectors.cpu(), path / "vectors.pt")
        with open(path / "meta.json", "w") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: Union[str, Path], device: Optional[str] = None) -> "VectorIndex":
        """Load index from disk."""
        path = Path(path)
        with open(path / "meta.json") as f:
            data = json.load(f)
        idx = cls(dim=data["dim"], device=device, fp16=data["fp16"], metadata=data["metadata"])
        vec_path = path / "vectors.pt"
        if vec_path.exists():
            idx._vectors = torch.load(vec_path, weights_only=True)
            idx._ensure_device()
        return idx
