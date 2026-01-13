# AGENTS.md — Operational Reference

> Keep this file brief and operational. Status updates belong in IMPLEMENTATION_PLAN.md.

## Build & Run

```bash
# Install dependencies (local dev - Python 3.12 required)
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run the application (local dev)
uvicorn app.main:app --host 0.0.0.0 --port 3000 --reload

# Run with Docker (production)
docker-compose up --build

# Run detached
docker-compose up -d --build
```

## Validation

```bash
# Run tests
pytest tests/ -v

# Run tests with coverage
pytest tests/ -v --cov=app

# Type checking
mypy app/

# Linting
ruff check app/
```

## Docker Commands

```bash
# Rebuild and restart
docker-compose down && docker-compose up --build

# View logs
docker-compose logs -f api

# Shell into container
docker-compose exec api bash

# Check Redis
docker-compose exec redis redis-cli
```

## API Testing

```bash
# Health check
curl http://localhost:3000/health

# Query endpoint
curl -X POST http://localhost:3000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the capital of France?"}'

# Force refresh (bypass cache)
curl -X POST http://localhost:3000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the capital of France?", "forceRefresh": true}'
```

## Environment

Required env vars (see .env.example):
- `OPENAI_API_KEY` - OpenAI API key for LLM calls
- `REDIS_URL` - Redis connection string (default: redis://redis:6379)

## Codebase Patterns

- FastAPI app entry: `app/main.py`
- API routes: `app/api/routes.py`
- Services: `app/services/` (embedding, cache, llm, classifier)
- Core logic: `app/core/semantic_cache.py`
- Tests: `tests/`
