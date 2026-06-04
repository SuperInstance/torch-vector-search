# torch-vector-search

GPU-accelerated vector search index using PyTorch.

## Features

- **Auto CPU/GPU selection** — CPU for <10K vectors, GPU for larger sets
- **Batch queries** — multiple queries in one call
- **FP16 storage** — 2x memory capacity with minimal accuracy loss
- **Persistent storage** — save/load indices to disk
- **Sub-millisecond latency** for <10K vectors on CPU
- **Built-in embedders** — HashEmbedder, PositionAwareEmbedder, BatchEmbedder

## Quick Start

```python
from torch_vector_search import VectorIndex
import numpy as np

# Create index
idx = VectorIndex(dim=384)

# Add vectors
idx.add(np.random.randn(1000, 384).astype(np.float32))

# Search
query = np.random.randn(384).astype(np.float32)
results = idx.search(query, top_k=5)
for r in results:
    print(f"  #{r.index} score={r.score:.4f}")

# Batch search
queries = np.random.randn(10, 384).astype(np.float32)
batch_results = idx.batch_search(queries, top_k=3)

# Save / load
idx.save("/path/to/index")
loaded = VectorIndex.load("/path/to/index")
```

## Installation

```bash
pip install -e .
```

## Benchmarks (RTX 4050)

| Vectors | Device | Latency |
|---------|--------|---------|
| 10K     | CPU    | < 1ms   |
| 100K    | GPU    | < 10ms  |

## License

MIT
