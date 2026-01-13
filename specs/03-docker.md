# Docker & Containerization Specification

## Overview

The solution must be containerized and startable with a single `docker-compose up` command, exposing the API at localhost:3000.

## docker-compose.yml

```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "3000:3000"
    environment:
      - REDIS_URL=redis://redis:6379
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    depends_on:
      redis:
        condition: service_healthy
    volumes:
      - model_cache:/root/.cache/huggingface

  redis:
    image: redis/redis-stack:latest
    ports:
      - "6379:6379"
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      timeout: 3s
      retries: 5
    volumes:
      - redis_data:/data

volumes:
  redis_data:
  model_cache:
```

## Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download embedding model during build
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY . .

EXPOSE 3000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "3000"]
```

## requirements.txt

```
fastapi==0.109.0
uvicorn==0.27.0
redis==5.0.1
sentence-transformers==2.3.1
openai==1.12.0
pydantic==2.6.0
pydantic-settings==2.1.0
python-dotenv==1.0.1
pytest==8.0.0
pytest-asyncio==0.23.0
httpx==0.26.0
```

## .env.example

```bash
OPENAI_API_KEY=sk-your-key-here
REDIS_URL=redis://redis:6379
```

## Acceptance Criteria

- [ ] `docker-compose up --build` starts all services
- [ ] API is accessible at http://localhost:3000
- [ ] Redis is healthy before API starts (depends_on condition)
- [ ] Embedding model is pre-downloaded in Docker image
- [ ] Environment variables are loaded from .env file
- [ ] Cache data persists across container restarts (volumes)
- [ ] No hardcoded credentials in code

## Implementation Notes

- Use redis/redis-stack for RediSearch support (vector search)
- Volume mount for HuggingFace model cache speeds up container rebuilds
- Health check ensures Redis is ready before API starts
- Python 3.11-slim for smaller image size
