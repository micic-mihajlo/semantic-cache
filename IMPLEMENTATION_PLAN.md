# Implementation Plan

## Overview

Build a production-ready semantic caching system to reduce LLM API costs by identifying semantically similar queries and returning cached responses. The system differentiates time-sensitive vs evergreen queries with distinct matching thresholds and TTLs.

**Ultimate Goal**: `docker-compose up` starts the system at localhost:3000, handles query classification, embedding generation, vector search, and LLM fallback seamlessly.

---

## Status: COMPLETE

All phases implemented and tested. The semantic caching system is production-ready.

### Recent Updates

**v0.0.6** - Fixed HTTP status codes and Redis eviction policy:
- **HTTP status code differentiation**: APIError now returns HTTP 502 (Bad Gateway), RateLimitError returns HTTP 429 (Too Many Requests)
- Added custom exception classes `LLMServiceUnavailableError` and `LLMRateLimitError` in `app/services/llm.py`
- Updated `app/api/routes.py` to catch and return appropriate HTTP status codes per spec
- **Redis eviction policy**: Configured `volatile-ttl` eviction policy in docker-compose.yml and cache.py
- Added `_configure_eviction_policy()` method to CacheService for runtime configuration
- **Comprehensive cache service tests**: Added 17 new tests in `TestCacheService` class
- Tests cover: initialization, search, store, close, error handling, eviction policy configuration
- **Test coverage improved from 67% to 90%**
- Total tests: 55 (up from 36)

**v0.0.5** - Expanded test coverage:
- Added `tests/test_services.py` with 20 unit tests for embedding, classifier, and LLM services
- Added edge case tests for special characters, Unicode, newlines, long queries, and numbers
- Added whitespace-only query validation in `QueryRequest` schema
- Embedding tests verify: 384-dim output, normalized vectors, semantic similarity, singleton pattern
- Classifier tests verify: time-sensitive and evergreen detection, caching parameters
- LLM tests verify: error handling (APIError, RateLimitError), empty content handling
- Total tests: 36 (up from 10)

**v0.0.4** - Fixed mypy type errors:
- Added proper type annotations to `EmbeddingService` singleton pattern in `app/services/embedding.py`
- Added null check guard to `CacheService._ensure_index()` in `app/services/cache.py`
- All 13 source files now pass mypy with `--ignore-missing-imports`

---

## Completed

### Phase 1: Docker & Infrastructure
- [x] `docker-compose.yml` - api + redis services, health checks, volumes
- [x] `Dockerfile` - Python 3.11-slim, pre-downloads embedding model
- [x] `requirements.txt` - all dependencies including dev tools (pytest-cov, ruff, mypy)
- [x] `.env.example` - OPENAI_API_KEY, REDIS_URL
- [x] `.gitignore` - Python project ignores

### Phase 2: Configuration & Core Setup
- [x] `app/config.py` - Pydantic Settings class using SettingsConfigDict
- [x] `app/main.py` - FastAPI app with lifespan handler for service initialization

### Phase 3: Services
- [x] `app/services/embedding.py` - SentenceTransformer singleton wrapper (all-MiniLM-L6-v2)
  - Singleton pattern to avoid loading model multiple times
  - `embed(text: str) -> np.ndarray` returning normalized 384-dim vectors
- [x] `app/services/classifier.py` - Time-sensitive vs evergreen detection
  - Regex-based pattern matching for temporal keywords and domain-specific patterns
  - Returns (query_type, confidence) tuple
  - time_sensitive: 0.15 threshold, 300s TTL
  - evergreen: 0.30 threshold, 604800s TTL
- [x] `app/services/cache.py` - Redis vector search with TTL
  - Create RediSearch index on startup (FLAT, 384 dims, COSINE)
  - Configure eviction policy to `volatile-ttl` on connect
  - `search(embedding, threshold)` - KNN 1 vector search
  - `store(query, response, embedding, query_type, ttl)` - Hash storage with TTL
- [x] `app/services/llm.py` - OpenAI AsyncClient wrapper
  - Async generation with gpt-4o-mini, temperature=0
  - Error handling for API and rate limit errors

### Phase 4: Core Orchestrator
- [x] `app/core/semantic_cache.py` - SemanticCacheManager class
  - Orchestrates: classify -> embed -> search cache -> (hit: return) / (miss: call LLM -> store -> return)
  - Handles forceRefresh bypass
  - Returns response with source metadata

### Phase 5: API Layer
- [x] `app/api/schemas.py` - Pydantic models
  - QueryRequest: query (str, required), forceRefresh (bool, default=False)
  - QueryResponse: response (str), metadata.source ("cache" | "llm")
  - ErrorResponse: error (str)
- [x] `app/api/routes.py` - API router
  - POST /api/query - main endpoint
  - GET /health - health check returning {"status": "ok"}
  - Proper error handling (400 for invalid input, 429 for rate limit, 500 for internal errors, 502 for LLM unavailable)

### Phase 6: Testing
- [x] `tests/conftest.py` - Pytest fixtures
  - AsyncClient fixture for testing FastAPI app
  - Mock OpenAI responses to avoid real API calls
- [x] `tests/test_api.py` - Integration tests (18 tests)
  - test_health_endpoint - health check works
  - test_cache_miss_calls_llm - LLM called on miss
  - test_cache_hit_returns_cached - cache hit returns cached
  - test_semantic_similarity - rephrased query returns cache
  - test_force_refresh_bypasses_cache - bypasses cache
  - test_unrelated_queries_miss_cache - different topics miss cache
  - test_empty_query_error - 422 error for empty query
  - test_missing_query_error - 422 error for missing query
  - test_time_sensitive_classification - time-sensitive queries handled
  - test_time_sensitive_strict_matching - stricter threshold for time-sensitive
  - test_special_characters_in_query - handles C++, C#, & characters
  - test_unicode_characters_in_query - handles Unicode (e.g., café)
  - test_long_query_handling - handles queries > 500 characters
  - test_query_with_newlines - handles multiline queries
  - test_query_with_numbers - handles numeric expressions
  - test_whitespace_only_query_error - rejects whitespace-only queries
  - test_llm_api_error_returns_502 - 502 for LLM API errors
  - test_llm_rate_limit_returns_429 - 429 for rate limit errors
- [x] `tests/test_services.py` - Unit tests (37 tests)
  - TestEmbeddingService: dimensions, normalization, type, similarity, singleton
  - TestClassifier: time-sensitive detection, evergreen detection, caching params
  - TestLLMService: initialization, error handling, successful generation
  - TestCacheService: initialization, search, store, close, index, eviction policy
- [x] `pytest.ini` - asyncio_mode = auto

### Module Markers
- [x] `app/__init__.py`
- [x] `app/api/__init__.py`
- [x] `app/services/__init__.py`
- [x] `app/core/__init__.py`
- [x] `tests/__init__.py`

### Phase 7: Documentation
- [x] `README.md` with:
  - Project overview and architecture diagram
  - Quick start instructions (`docker-compose up`)
  - API documentation (POST /api/query, GET /health)
  - Configuration options (OPENAI_API_KEY, REDIS_URL)
  - Design decisions (embedding model choice, thresholds, TTLs)
  - Why semantic caching matters (cost, speed, smart matching)

---

## Technical Reference

### Embedding Model: all-MiniLM-L6-v2
- Dimensions: 384
- Speed: 14k sentences/sec
- Cost: $0 (local)
- Normalization: Required for cosine similarity

### Caching Parameters
| Query Type | Distance Threshold | TTL |
|------------|-------------------|-----|
| time_sensitive | 0.15 (strict) | 5 min (300s) |
| evergreen | 0.30 (relaxed) | 7 days (604800s) |

### Time-Sensitive Patterns
- Temporal: today, now, current, latest, recent, yesterday, tomorrow
- Domain: weather, forecast, news, headlines, stock, price, market, bitcoin, score, game

### Evergreen Patterns
- "who was the first...", "what year did...", "definition of...", "what is a...", "how do you...", "history of..."

### Redis Schema
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

---

## Success Criteria (from specs/01-overview.md)

- [x] `docker-compose up` starts the system with a single command
- [x] Server exposes API at localhost:3000
- [x] Semantically similar queries return cached responses
- [x] Time-sensitive queries use stricter matching and shorter TTL
- [x] Cache hit/miss is reported in response metadata
- [x] System handles various query patterns correctly

---

## Development Notes

### Local Development (Python 3.12 required)
```bash
# Use Python 3.12 for local development (3.14 has compatibility issues with pydantic-core)
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=app
```

### pytest-asyncio Version
- Updated from 0.23.0 to 0.23.8 to fix collection errors with pytest 8.0.0
