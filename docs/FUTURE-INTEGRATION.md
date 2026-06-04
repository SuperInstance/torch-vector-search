# Future Integration: torch-vector-search

## Current State
GPU-accelerated vector search index using PyTorch. Auto CPU/GPU selection, FP16 storage, persistent indices, sub-millisecond latency for <10K vectors. Includes HashEmbedder and PositionAwareEmbedder.

## Integration Opportunities

### With open-vectors/weaviate GPU acceleration
torch-vector-search provides GPU-accelerated vector search for the fleet's Weaviate instance. When Weaviate needs to search millions of skill/room/strategy vectors, torch-vector-search handles the GPU acceleration. FP16 storage doubles capacity; batch queries handle fleet-scale throughput.

### With position-aware-embed at scale
The PositionAwareEmbedder in torch-vector-search generates the same embeddings as position-aware-embed (standalone Rust crate), but on GPU and at scale. For fleet-wide embedding generation (690+ repos, thousands of skills), GPU acceleration is essential.

### With oracle1-index semantic search
The current keyword-based index (674KB) is augmented with torch-vector-search for semantic queries. Repo descriptions, README content, and capability declarations are embedded and indexed. "Find repos about conservation laws in ternary systems" returns semantically relevant results, not just keyword matches.

## Dormant Ideas Now Unlockable
The GPU vector search was standalone. Now the fleet has a concrete need: searching across 690+ repos, thousands of skills, and hundreds of rooms. GPU acceleration makes fleet-scale search practical.

## Potential in Mature Systems
torch-vector-search is the fleet's search engine. Every skill, every room, every strategy is embedded and indexed. GPU-accelerated search finds the right resource in milliseconds, even across millions of vectors.

## Cross-Pollination Ideas
- **open-vectors/weaviate**: GPU-accelerated backend for Weaviate
- **position-aware-embed**: Same embedding algorithm, GPU-accelerated
- **oracle1-index**: Semantic search for fleet catalog

## Dependencies for Next Steps
- Fleet-wide embedding generation pipeline
- Integration with Weaviate's query API
- Persistent index management for fleet scale
