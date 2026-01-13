# Semantic Caching System

A production-ready semantic caching system that reduces LLM API costs by identifying semantically similar queries and returning cached responses instead of making redundant API calls.

## Why Semantic Caching?

- **Cost Reduction**: LLM API calls cost $0.01-$0.10 per call. Caching eliminates redundant calls.
- **Faster Response**: Cache hits return instantly vs waiting for LLM generation.
- **Smart Matching**: Semantically similar questions (different wording, same meaning) hit the cache.
- **Time-Aware**: Time-sensitive queries (weather, stocks) use stricter matching and shorter TTL.

## Architecture

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

## Quick Start

### Prerequisites

- Docker and Docker Compose
- OpenAI API key

### Run with Docker (Recommended)

```bash
# 1. Clone and navigate to the project
cd boardy-semantic-cache

# 2. Create .env file with your OpenAI API key
echo "OPENAI_API_KEY=sk-your-key-here" > .env

# 3. Start the system
docker-compose up --build

# The API is now available at http://localhost:3000
```

### Verify It's Working

```bash
# Health check
curl http://localhost:3000/health

# Submit a query
curl -X POST http://localhost:3000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the capital of France?"}'
```

## API Documentation

### POST /api/query

Submit a query for processing. Returns cached response if semantically similar query exists, otherwise calls LLM.

**Request:**
```json
{
  "query": "What is the capital of France?",
  "forceRefresh": false
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| query | string | Yes | The user's query text |
| forceRefresh | boolean | No | Bypass cache and force LLM call (default: false) |

**Response:**
```json
{
  "response": "The capital of France is Paris.",
  "metadata": {
    "source": "cache"
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| response | string | The answer to the query |
| metadata.source | string | `"cache"` or `"llm"` indicating response origin |

**Status Codes:**
- `200`: Success
- `400`: Invalid request (missing query)
- `429`: Rate limit exceeded (OpenAI rate limit)
- `500`: Internal server error
- `502`: LLM service unavailable (OpenAI API error)

### GET /health

Health check endpoint.

**Response:**
```json
{
  "status": "ok"
}
```

## Configuration

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | Yes | - | OpenAI API key for LLM calls |
| `REDIS_URL` | No | `redis://redis:6379` | Redis connection string |

### .env.example

```bash
OPENAI_API_KEY=sk-your-key-here
REDIS_URL=redis://redis:6379
```

## Design Decisions

### Embedding Model: all-MiniLM-L6-v2

| Criteria | Value | Benefit |
|----------|-------|---------|
| Speed | 14,000 sentences/sec | Real-time performance |
| Dimensions | 384 | Compact, efficient storage |
| API Cost | $0 | Runs locally, no external calls |
| Quality | Good | Sufficient for semantic similarity |

**Why not OpenAI embeddings?** Local embeddings eliminate API latency, rate limits, and costs. The quality difference is negligible for cache matching purposes.

### Time-Sensitive vs Evergreen Classification

Queries are classified to determine appropriate caching behavior:

| Query Type | Distance Threshold | TTL | Use Case |
|------------|-------------------|-----|----------|
| time_sensitive | 0.15 (strict) | 5 minutes | Weather, stocks, news, scores |
| evergreen | 0.30 (relaxed) | 7 days | Facts, definitions, how-to |

**Why different thresholds?**
- **Strict (0.15)**: For time-sensitive data, even small differences matter. "Weather in NYC" should not match "Weather in LA".
- **Relaxed (0.30)**: For evergreen facts, semantic equivalence is acceptable. "Capital of France" matches "France's capital".

**Why different TTLs?**
- **5 minutes**: Weather, stock prices, and news change frequently.
- **7 days**: Historical facts and definitions rarely change.

### Time-Sensitive Detection Patterns

**Temporal keywords:** today, now, current, latest, recent, yesterday, tomorrow

**Domain-specific:** weather, forecast, news, headlines, stock, price, market, bitcoin, score, game

### Distance Metric: Cosine

| Distance | Interpretation |
|----------|----------------|
| 0.0 - 0.1 | Nearly identical |
| 0.1 - 0.2 | Very similar |
| 0.2 - 0.3 | Similar |
| 0.3 - 0.5 | Somewhat related |
| 0.5+ | Different topics |

### Redis with RediSearch

| Feature | Benefit |
|---------|---------|
| Native TTL | Automatic cache expiration |
| Vector Search | Semantic similarity via KNN |
| Persistence | Cache survives restarts |
| Performance | Sub-millisecond lookups |

### LLM: gpt-4o-mini

| Criteria | Value |
|----------|-------|
| Cost | ~$0.15/1M input, ~$0.60/1M output tokens |
| Speed | Fast response times |
| Temperature | 0 (deterministic for consistent caching) |

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
│   ├── main.py              # FastAPI app with lifespan
│   ├── config.py            # Pydantic Settings
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes.py        # POST /api/query, GET /health
│   │   └── schemas.py       # Request/Response models
│   ├── services/
│   │   ├── __init__.py
│   │   ├── embedding.py     # SentenceTransformer singleton
│   │   ├── cache.py         # Redis vector search
│   │   ├── llm.py           # OpenAI async wrapper
│   │   └── classifier.py    # Time-sensitive detection
│   └── core/
│       ├── __init__.py
│       └── semantic_cache.py # Main orchestrator
└── tests/
    ├── __init__.py
    ├── conftest.py          # Pytest fixtures
    ├── test_api.py          # Integration tests
    └── test_services.py     # Service unit tests
```

## Local Development

```bash
# Python 3.12 required (3.14 has pydantic-core compatibility issues)
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=app

# Type checking
mypy app/

# Linting
ruff check app/

# Run locally (requires Redis running)
uvicorn app.main:app --host 0.0.0.0 --port 3000 --reload
```

## Docker Commands

```bash
# Build and start
docker-compose up --build

# Run detached
docker-compose up -d --build

# View logs
docker-compose logs -f api

# Rebuild after changes
docker-compose down && docker-compose up --build

# Shell into container
docker-compose exec api bash

# Check Redis
docker-compose exec redis redis-cli
```

## How Semantic Matching Works

1. **Query Classification**: Regex patterns identify time-sensitive queries
2. **Embedding Generation**: Query converted to 384-dimensional vector
3. **Vector Search**: Redis KNN finds nearest cached query
4. **Threshold Check**: If distance < threshold, return cache; else call LLM
5. **Cache Storage**: New responses stored with appropriate TTL

### Example: Cache Hit

```
Query 1: "What is the capital of France?"
  → Embedding generated
  → Cache miss (empty cache)
  → LLM called: "The capital of France is Paris."
  → Stored in cache with 7-day TTL

Query 2: "What's France's capital?"
  → Embedding generated
  → Vector search finds Query 1 (distance: 0.18)
  → 0.18 < 0.30 threshold → Cache hit!
  → Returns cached response instantly
```

### Example: Time-Sensitive Strict Matching

```
Query 1: "What's the weather in NYC today?"
  → Classified as time_sensitive
  → Stored with 5-minute TTL, threshold 0.15

Query 2: "What's the weather in LA today?"
  → Classified as time_sensitive
  → Vector search finds Query 1 (distance: 0.22)
  → 0.22 > 0.15 threshold → Cache miss!
  → LLM called (different city = different weather)
```

## License

MIT
