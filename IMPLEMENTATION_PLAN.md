# Implementation Plan

## Priority Tasks

### Phase 1: Docker & Infrastructure
- [ ] Create `docker-compose.yml` with api and redis services
- [ ] Create `Dockerfile` for Python 3.11 with FastAPI
- [ ] Create `requirements.txt` with all dependencies
- [ ] Create `.env.example` with OPENAI_API_KEY and REDIS_URL
- [ ] Create `.gitignore` for Python project

### Phase 2: Configuration & Core Setup
- [ ] Create `app/config.py` with pydantic-settings for env vars
- [ ] Create `app/__init__.py`
- [ ] Create `app/main.py` with FastAPI app and lifespan handler

### Phase 3: Services
- [ ] Create `app/services/__init__.py`
- [ ] Create `app/services/embedding.py` - SentenceTransformer wrapper (all-MiniLM-L6-v2)
- [ ] Create `app/services/classifier.py` - Time-sensitive vs evergreen detection
- [ ] Create `app/services/cache.py` - Redis vector search with TTL
- [ ] Create `app/services/llm.py` - OpenAI AsyncClient wrapper

### Phase 4: Core Orchestrator
- [ ] Create `app/core/__init__.py`
- [ ] Create `app/core/semantic_cache.py` - SemanticCacheManager orchestrating the flow

### Phase 5: API Layer
- [ ] Create `app/api/__init__.py`
- [ ] Create `app/api/schemas.py` - Pydantic models for request/response
- [ ] Create `app/api/routes.py` - POST /api/query endpoint

### Phase 6: Testing
- [ ] Create `tests/__init__.py`
- [ ] Create `tests/test_api.py` - Integration tests for cache hit/miss scenarios
- [ ] Create `tests/conftest.py` - Pytest fixtures

### Phase 7: Documentation
- [ ] Create `README.md` with setup instructions, architecture, design decisions

## Completed

<!-- Move completed items here -->

## Notes & Learnings

- Embedding model: all-MiniLM-L6-v2 (384 dims, 14k sent/sec, no API cost)
- Time-sensitive threshold: 0.15 (strict), TTL: 5 min
- Evergreen threshold: 0.30 (relaxed), TTL: 7 days
- Redis with RediSearch for vector similarity + native TTL
- OpenAI gpt-4o-mini for LLM fallback
