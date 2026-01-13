# Semantic Caching System Overview

## Purpose

Build an AI-powered semantic caching system that reduces LLM API costs by identifying when incoming queries are semantically similar to previously seen queries and returning cached responses instead of making redundant LLM calls.

## Jobs to Be Done (JTBD)

When a user submits a query that is semantically similar to a previous query, I want to return the cached response, so I can reduce LLM API costs while maintaining response quality.

When a user submits a time-sensitive query, I want to use stricter matching and shorter cache TTL, so I can avoid returning stale information.

When no suitable cache match exists, I want to call the LLM and cache the response, so future similar queries can be served from cache.

## System Flow

```
POST /api/query
       │
       ▼
┌─────────────────┐
│ Classify Query  │ → time_sensitive (5min TTL, strict threshold)
│                 │ → evergreen (7day TTL, relaxed threshold)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Generate        │
│ Embedding       │ → 384-dim vector via sentence-transformers
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌─────────────────┐
│ Check Cache     │────▶│ HIT: Return     │
│ (Redis vector   │     │ cached response │
│  search)        │     └─────────────────┘
└────────┬────────┘
         │ MISS
         ▼
┌─────────────────┐
│ Call LLM        │
│ (gpt-4o-mini)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Store in cache  │
│ with TTL        │
└────────┬────────┘
         │
         ▼
    Return response
```

## Tech Stack

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Framework | FastAPI | Async, auto-docs, fast development |
| Embeddings | all-MiniLM-L6-v2 | 14k sent/sec, 384 dims, no API cost |
| Vector Store | Redis + RediSearch | TTL built-in, vector search ready |
| LLM | OpenAI gpt-4o-mini | Cheap, fast, reliable |
| Container | Docker Compose | Single command startup |

## Business Context

- LLM API calls cost $0.01-$0.10 per call
- Users often ask semantically similar questions with different wording
- Some queries are time-sensitive (weather, news, stocks) while others are evergreen (facts, definitions)
- Response time is critical for user experience
- System must scale efficiently as query volume increases

## Project Structure

```
boardy-semantic-cache/
├── docker-compose.yml
├── Dockerfile
├── .env.example
├── README.md
├── requirements.txt
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI app
│   ├── config.py            # Settings from env
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes.py        # POST /api/query
│   │   └── schemas.py       # Request/Response models
│   ├── services/
│   │   ├── __init__.py
│   │   ├── embedding.py     # Sentence transformer
│   │   ├── cache.py         # Redis operations
│   │   ├── llm.py           # OpenAI wrapper
│   │   └── classifier.py    # Time-sensitive detection
│   └── core/
│       ├── __init__.py
│       └── semantic_cache.py # Main orchestrator
└── tests/
    └── test_api.py
```

## Success Criteria

- [ ] `docker-compose up` starts the system with a single command
- [ ] Server exposes API at localhost:3000
- [ ] Semantically similar queries return cached responses
- [ ] Time-sensitive queries use stricter matching and shorter TTL
- [ ] Cache hit/miss is reported in response metadata
- [ ] System handles various query patterns correctly
