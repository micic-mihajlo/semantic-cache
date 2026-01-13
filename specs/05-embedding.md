# Embedding & Similarity Approach

## Overview

The system uses local embeddings via sentence-transformers to convert queries into vector representations for semantic similarity comparison.

## Embedding Model: all-MiniLM-L6-v2

### Why This Model

| Criteria | Value | Benefit |
|----------|-------|---------|
| Speed | 14,000 sentences/sec | Fast enough for real-time |
| Dimensions | 384 | Compact, efficient storage |
| API Cost | $0 | Runs locally, no external calls |
| Quality | Good | Sufficient for semantic similarity |

### Alternatives Considered

| Model | Speed | Dims | Cost | Quality |
|-------|-------|------|------|---------|
| all-MiniLM-L6-v2 | 14k/s | 384 | Free | Good |
| OpenAI text-embedding-3-small | ~1k/s | 1536 | $0.02/1M tokens | Better |
| all-mpnet-base-v2 | 2.8k/s | 768 | Free | Better |

Decision: all-MiniLM-L6-v2 provides the best balance of speed and quality for this use case. No API cost means no rate limits or additional latency.

## Implementation

```python
from sentence_transformers import SentenceTransformer
import numpy as np

class EmbeddingService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.model = SentenceTransformer('all-MiniLM-L6-v2')
        return cls._instance

    def embed(self, text: str) -> np.ndarray:
        return self.model.encode(text, normalize_embeddings=True)
```

### Key Implementation Details

1. **Singleton Pattern**: Only one model instance to avoid loading multiple times
2. **Normalized Embeddings**: `normalize_embeddings=True` for cosine similarity
3. **NumPy Array Output**: Compatible with Redis vector storage

## Vector Similarity

### Distance Metric: Cosine

Cosine distance is used because:
- Works well with normalized embeddings
- Scale-invariant (query length doesn't matter)
- Industry standard for text similarity

### Distance Formula

```
cosine_distance = 1 - cosine_similarity
```

Where:
- 0.0 = identical
- 1.0 = completely different

### Threshold Interpretation

| Distance | Interpretation |
|----------|----------------|
| 0.0 - 0.1 | Nearly identical |
| 0.1 - 0.2 | Very similar |
| 0.2 - 0.3 | Similar |
| 0.3 - 0.5 | Somewhat related |
| 0.5+ | Different topics |

## Redis Vector Search

### Index Schema

```python
schema = [
    TextField("query"),
    TextField("response"),
    TextField("query_type"),
    NumericField("created_at"),
    VectorField("embedding", "FLAT", {
        "TYPE": "FLOAT32",
        "DIM": 384,
        "DISTANCE_METRIC": "COSINE",
    }),
]
```

### Search Query

```python
query = (
    Query(f"*=>[KNN 1 @embedding $vec AS distance]")
    .return_fields("query", "response", "query_type", "distance")
    .sort_by("distance")
    .dialect(2)
)
```

This finds the 1 nearest neighbor and returns its distance for threshold comparison.

## Acceptance Criteria

- [ ] Embedding model loads successfully on startup
- [ ] Embeddings are 384-dimensional float32 arrays
- [ ] Embeddings are normalized (unit length)
- [ ] Similar queries produce similar embeddings (distance < 0.3)
- [ ] Unrelated queries produce different embeddings (distance > 0.5)
- [ ] Model is cached in Docker volume for fast restarts

## Performance Considerations

- Model loading takes ~2-3 seconds on first use
- Embedding generation takes ~1-2ms per query
- Pre-download model in Dockerfile to avoid cold start
- Singleton ensures model loaded only once per process
