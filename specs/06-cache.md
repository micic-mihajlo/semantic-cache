# Cache Service Specification

## Overview

Redis with RediSearch provides vector similarity search with automatic TTL-based expiration. This enables semantic cache lookups and time-based invalidation.

## Why Redis

| Feature | Benefit |
|---------|---------|
| Native TTL | Automatic expiration for time-sensitive queries |
| RediSearch | Built-in vector similarity search |
| Persistence | Cache survives restarts |
| Performance | In-memory, sub-millisecond lookups |
| Scaling Path | Cluster mode for production |

## Data Model

### Key Format
```
cache:{md5_hash_of_query}
```

### Hash Fields
| Field | Type | Description |
|-------|------|-------------|
| query | string | Original query text |
| response | string | LLM response |
| query_type | string | "time_sensitive" or "evergreen" |
| created_at | int | Unix timestamp |
| embedding | bytes | Float32 array as bytes |

## Index Configuration

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

index_definition = IndexDefinition(
    prefix=["cache:"],
    index_type=IndexType.HASH
)
```

### Index Type: FLAT vs HNSW

- **FLAT**: Exact nearest neighbor, best for small datasets (<100k vectors)
- **HNSW**: Approximate, faster for large datasets

Decision: Use FLAT for accuracy. Can switch to HNSW if scale requires.

## Operations

### Search (Cache Lookup)

```python
def search(self, embedding: np.ndarray, threshold: float) -> dict | None:
    query_vector = embedding.astype(np.float32).tobytes()
    q = (
        Query(f"*=>[KNN 1 @embedding $vec AS distance]")
        .return_fields("query", "response", "query_type", "distance")
        .sort_by("distance")
        .dialect(2)
    )
    results = self.redis.ft(self.index_name).search(q, {"vec": query_vector})

    if results.docs:
        doc = results.docs[0]
        distance = float(doc.distance)
        if distance <= threshold:
            return {
                "query": doc.query,
                "response": doc.response,
                "distance": distance,
            }
    return None
```

### Store (Cache Write)

```python
def store(self, query: str, response: str, embedding: np.ndarray,
          query_type: str, ttl: int):
    key = f"cache:{hashlib.md5(query.encode()).hexdigest()}"
    self.redis.hset(key, mapping={
        "query": query,
        "response": response,
        "query_type": query_type,
        "created_at": int(time.time()),
        "embedding": embedding.astype(np.float32).tobytes(),
    })
    self.redis.expire(key, ttl)
```

## TTL Strategy

| Query Type | TTL | Rationale |
|------------|-----|-----------|
| time_sensitive | 5 minutes | Data becomes stale quickly |
| evergreen | 7 days | Facts don't change often |

TTL is set per-key using `EXPIRE` command after storing.

## Acceptance Criteria

- [ ] Index is created on service startup if not exists
- [ ] Vector search returns nearest neighbor with distance
- [ ] Results are filtered by distance threshold
- [ ] Cache entries expire automatically based on TTL
- [ ] Duplicate queries (exact match) overwrite existing entry
- [ ] Cache persists across container restarts (Redis volume)

## Error Handling

- If Redis is unavailable, the system should fall back to LLM-only mode
- Log connection errors but don't crash the service
- Index creation is idempotent (check if exists first)

## Eviction Policy

Redis handles eviction via:
1. **TTL expiration**: Primary mechanism for cache invalidation
2. **maxmemory-policy**: Set to `volatile-ttl` to evict keys with shortest TTL when memory is full

## Implementation Notes

- Use `redis.from_url()` for connection
- Ensure index exists before any search operations
- Convert embeddings to float32 bytes for storage
- Hash query for deterministic key generation
