# Testing Specification

## Overview

Tests validate the semantic caching system handles various query patterns correctly, including exact duplicates, semantic similarity, time-sensitive queries, and edge cases.

## Test Categories

### 1. Cache Hit/Miss Tests

```python
@pytest.mark.asyncio
async def test_cache_hit(client):
    # First call - miss
    r1 = await client.post("/api/query", json={"query": "What is the capital of France?"})
    assert r1.json()["metadata"]["source"] == "llm"

    # Second call - hit
    r2 = await client.post("/api/query", json={"query": "What is the capital of France?"})
    assert r2.json()["metadata"]["source"] == "cache"
```

### 2. Semantic Similarity Tests

```python
@pytest.mark.asyncio
async def test_semantic_similarity(client):
    await client.post("/api/query", json={"query": "What is the capital of France?"})
    r = await client.post("/api/query", json={"query": "What's France's capital?"})
    assert r.json()["metadata"]["source"] == "cache"
```

### 3. Force Refresh Tests

```python
@pytest.mark.asyncio
async def test_force_refresh(client):
    await client.post("/api/query", json={"query": "What is the capital of France?"})
    r = await client.post("/api/query", json={
        "query": "What is the capital of France?",
        "forceRefresh": True
    })
    assert r.json()["metadata"]["source"] == "llm"
```

### 4. Unrelated Query Tests

```python
@pytest.mark.asyncio
async def test_unrelated_queries(client):
    await client.post("/api/query", json={"query": "What is the capital of France?"})
    r = await client.post("/api/query", json={"query": "How do I make pasta?"})
    assert r.json()["metadata"]["source"] == "llm"
```

### 5. Time-Sensitive Query Tests

```python
@pytest.mark.asyncio
async def test_time_sensitive_strict_matching(client):
    # Time-sensitive queries should use stricter threshold
    await client.post("/api/query", json={"query": "What's the weather in NYC today?"})
    r = await client.post("/api/query", json={"query": "What's the weather in LA today?"})
    # Different cities should NOT match due to strict threshold
    assert r.json()["metadata"]["source"] == "llm"
```

## Test Patterns to Cover

### Exact Duplicates
- Same query returns cache hit

### Semantically Similar
- Rephrased questions match
- Different word order matches
- Synonyms match

### Completely Unrelated
- Different topics return cache miss

### Time-Sensitive vs Evergreen
- Weather queries use strict matching
- Historical facts use relaxed matching

### Complexity and Length
- Short queries work
- Long queries work
- Complex queries work

### Edge Cases
- Empty query returns error
- Special characters handled
- Unicode/international characters handled

## Test Infrastructure

### conftest.py

```python
import pytest
from httpx import AsyncClient
from app.main import app

@pytest.fixture
async def client():
    async with AsyncClient(app=app, base_url="http://test") as c:
        yield c
```

### pytest.ini

```ini
[pytest]
asyncio_mode = auto
testpaths = tests
```

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=app --cov-report=html

# Run specific test file
pytest tests/test_api.py -v

# Run specific test
pytest tests/test_api.py::test_cache_hit -v
```

## Acceptance Criteria

- [ ] All test categories pass
- [ ] Tests can run against live Redis instance
- [ ] Tests are isolated (don't affect each other)
- [ ] Tests complete in reasonable time (<30s)
- [ ] Coverage > 80% for core modules

## Mocking Considerations

For unit tests, mock:
- OpenAI API calls (avoid real API costs)
- Redis operations (for isolated testing)

For integration tests:
- Use real Redis (via Docker)
- Mock only OpenAI to control responses
