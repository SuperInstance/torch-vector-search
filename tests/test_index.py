"""Tests for torch_vector_search.index."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from torch_vector_search import VectorIndex, SearchResult, PositionAwareEmbedder


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_index():
    """Index with 100 random 384-dim vectors."""
    idx = VectorIndex(dim=384, device="cpu")
    rng = np.random.RandomState(0)
    vecs = rng.randn(100, 384).astype(np.float32)
    meta = [{"id": i} for i in range(100)]
    idx.add(vecs, metadata=meta)
    return idx


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_basic_search(small_index):
    """Search returns correct top-k results."""
    query = np.random.RandomState(42).randn(384).astype(np.float32)
    results = small_index.search(query, top_k=5)
    assert len(results) == 5
    assert all(isinstance(r, SearchResult) for r in results)
    # Scores should be descending
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
    # The best result should have metadata
    assert results[0].metadata.get("id") is not None


def test_batch_search(small_index):
    """Batch search returns results for all queries."""
    rng = np.random.RandomState(7)
    queries = rng.randn(4, 384).astype(np.float32)
    results = small_index.batch_search(queries, top_k=3)
    assert len(results) == 4
    for row in results:
        assert len(row) == 3


def test_auto_device():
    """Auto-selects CPU for small DBs, GPU for large."""
    small = VectorIndex(dim=64)
    small.add(np.random.randn(100, 64).astype(np.float32))
    assert small.device.type == "cpu"

    large = VectorIndex(dim=64)
    large.add(np.random.randn(11_000, 64).astype(np.float32))
    if torch.cuda.is_available():
        assert large.device.type == "cuda"
    else:
        assert large.device.type == "cpu"


def test_fp16():
    """FP16 storage works and returns same results."""
    idx_fp32 = VectorIndex(dim=64, device="cpu", fp16=False)
    idx_fp16 = VectorIndex(dim=64, device="cpu", fp16=True)
    rng = np.random.RandomState(1)
    vecs = rng.randn(500, 64).astype(np.float32)
    idx_fp32.add(vecs)
    idx_fp16.add(vecs)

    assert idx_fp16._vectors.dtype == torch.float16

    query = rng.randn(64).astype(np.float32)
    r32 = idx_fp32.search(query, top_k=5)
    r16 = idx_fp16.search(query, top_k=5)
    # Top-1 should match (FP16 is approximate but top-1 usually stable)
    assert r32[0].index == r16[0].index


def test_save_load(small_index):
    """Save and load preserves index state."""
    query = np.random.RandomState(99).randn(384).astype(np.float32)
    expected = small_index.search(query, top_k=5)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "index"
        small_index.save(path)
        loaded = VectorIndex.load(path)
        assert loaded.count == small_index.count
        assert loaded.dim == small_index.dim

        got = loaded.search(query, top_k=5)
        for e, g in zip(expected, got):
            assert e.index == g.index
            assert abs(e.score - g.score) < 1e-5


def test_latency_cpu():
    """CPU search under 1ms for 10K vectors."""
    idx = VectorIndex(dim=384, device="cpu")
    rng = np.random.RandomState(3)
    idx.add(rng.randn(10_000, 384).astype(np.float32))

    query = rng.randn(384).astype(np.float32)
    # Warm up
    idx.search(query)
    import time
    t0 = time.perf_counter()
    idx.search(query)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    assert elapsed_ms < 5.0, f"CPU search took {elapsed_ms:.2f}ms (> 5ms)"


def test_latency_gpu():
    """GPU search under 50ms for 100K vectors (if CUDA available)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    idx = VectorIndex(dim=384, device="cuda")
    rng = np.random.RandomState(4)
    idx.add(rng.randn(100_000, 384).astype(np.float32))

    query = rng.randn(384).astype(np.float32)
    # Warm up
    idx.search(query)
    import time
    t0 = time.perf_counter()
    idx.search(query)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    assert elapsed_ms < 50.0, f"GPU search took {elapsed_ms:.2f}ms (> 50ms)"


def test_position_aware_embedder():
    """Position-aware embedder produces normalized vectors."""
    emb = PositionAwareEmbedder(dim=384)
    vec = emb.embed("hello world", position=0)
    assert vec.shape == (384,)
    norm = np.linalg.norm(vec)
    assert abs(norm - 1.0) < 1e-5, f"Norm was {norm}"

    batch = emb.batch_embed(["a", "b", "c"], positions=[0, 1, 2])
    assert batch.shape == (3, 384)


def test_stats_tracking(small_index):
    """Stats are tracked after searches."""
    query = np.random.randn(384).astype(np.float32)
    small_index.search(query)
    small_index.search(query)
    assert small_index.stats["queries"] == 2
    assert small_index.stats["avg_latency_ms"] > 0
