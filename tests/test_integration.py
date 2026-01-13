"""Integration tests that hit real services (Redis, OpenAI).

Run with: pytest tests/test_integration.py -v

Prerequisites:
- Redis running (docker compose up redis)
- OPENAI_API_KEY set in environment or .env file
"""

import os

import pytest
from httpx import ASGITransport, AsyncClient

from app.config import settings
from app.services.cache import CacheService
from app.services.classifier import classify, get_caching_params
from app.services.embedding import embedding_service
from app.services.llm import LLMService


# Skip all tests in this file if OPENAI_API_KEY is not set
pytestmark = pytest.mark.integration


def has_openai_key() -> bool:
    """Check if OpenAI API key is available."""
    return bool(settings.openai_api_key)


def get_redis_url() -> str:
    """Get Redis URL, preferring localhost for local testing."""
    # Try localhost first (for running tests outside Docker)
    # Fall back to settings.redis_url (for Docker-to-Docker)
    return "redis://localhost:6379"


def has_redis() -> bool:
    """Check if Redis is reachable."""
    try:
        import redis
        r = redis.from_url(get_redis_url(), socket_connect_timeout=1)
        r.ping()
        r.close()
        return True
    except Exception:
        return False


skip_no_openai = pytest.mark.skipif(
    not has_openai_key(),
    reason="OPENAI_API_KEY not set"
)

skip_no_redis = pytest.mark.skipif(
    not has_redis(),
    reason="Redis not reachable"
)


class TestEmbeddingIntegration:
    """Test embedding service with real model."""

    def test_real_embedding_generation(self):
        """Test that real embeddings are generated correctly."""
        embedding = embedding_service.embed("What is the capital of France?")

        assert embedding is not None
        assert embedding.shape == (384,)
        assert embedding.dtype.name == "float32"

    def test_real_semantic_similarity(self):
        """Test that semantically similar queries have low distance."""
        import numpy as np

        emb1 = embedding_service.embed("What is the capital of France?")
        emb2 = embedding_service.embed("What's France's capital city?")
        emb3 = embedding_service.embed("How do I cook pasta?")

        # Cosine distance (since embeddings are normalized)
        dist_similar = 1 - np.dot(emb1, emb2)
        dist_different = 1 - np.dot(emb1, emb3)

        # Similar queries should have low distance
        assert dist_similar < 0.3, f"Similar queries too distant: {dist_similar}"
        # Different queries should have higher distance
        assert dist_different > 0.5, f"Different queries too close: {dist_different}"


class TestClassifierIntegration:
    """Test classifier with various query types."""

    def test_time_sensitive_queries(self):
        """Test classification of time-sensitive queries."""
        time_sensitive_queries = [
            "What's the weather in NYC today?",
            "What's the current bitcoin price?",
            "Latest news about AI",
            "What's the stock price of Apple now?",
        ]

        for query in time_sensitive_queries:
            query_type, confidence = classify(query)
            params = get_caching_params(query_type)
            assert query_type == "time_sensitive", f"Failed: {query}"
            assert params["threshold"] == 0.15
            assert params["ttl"] == 300

    def test_evergreen_queries(self):
        """Test classification of evergreen queries."""
        evergreen_queries = [
            "Who was the first president of the United States?",
            "What is the definition of photosynthesis?",
            "How do you calculate the area of a circle?",
            "What is the capital of France?",
        ]

        for query in evergreen_queries:
            query_type, confidence = classify(query)
            params = get_caching_params(query_type)
            assert query_type == "evergreen", f"Failed: {query}"
            assert params["threshold"] == 0.30
            assert params["ttl"] == 604800


@skip_no_redis
class TestCacheIntegration:
    """Test cache service with real Redis."""

    @pytest.fixture
    def cache_service(self):
        """Create a real cache service connected to Redis."""
        service = CacheService(redis_url=get_redis_url())
        service.connect()
        yield service
        service.close()

    def test_store_and_retrieve(self, cache_service):
        """Test storing and retrieving from real Redis."""
        import numpy as np

        query = "Integration test query"
        response = "Integration test response"
        embedding = embedding_service.embed(query)

        # Store
        cache_service.store(
            query=query,
            response=response,
            embedding=embedding,
            query_type="evergreen",
            ttl=60,  # Short TTL for test cleanup
        )

        # Retrieve
        result = cache_service.search(
            embedding=embedding,
            threshold=0.3,
        )

        assert result is not None
        assert result["response"] == response
        assert result["distance"] < 0.01  # Should be nearly identical

    def test_semantic_cache_hit(self, cache_service):
        """Test that semantically similar queries hit the cache."""
        import numpy as np

        # Store original query
        original_query = "What is the capital of Germany?"
        response = "The capital of Germany is Berlin."
        original_embedding = embedding_service.embed(original_query)

        cache_service.store(
            query=original_query,
            response=response,
            embedding=original_embedding,
            query_type="evergreen",
            ttl=60,
        )

        # Search with similar query
        similar_query = "What's Germany's capital?"
        similar_embedding = embedding_service.embed(similar_query)

        result = cache_service.search(
            embedding=similar_embedding,
            threshold=0.3,
        )

        assert result is not None
        assert result["response"] == response

    def test_cache_miss_for_different_queries(self, cache_service):
        """Test that unrelated queries don't hit the cache."""
        import numpy as np

        # Store a query
        cache_service.store(
            query="What is machine learning?",
            response="Machine learning is...",
            embedding=embedding_service.embed("What is machine learning?"),
            query_type="evergreen",
            ttl=60,
        )

        # Search with completely different query
        different_embedding = embedding_service.embed("How do I bake a cake?")

        result = cache_service.search(
            embedding=different_embedding,
            threshold=0.3,
        )

        assert result is None


@skip_no_openai
class TestLLMIntegration:
    """Test LLM service with real OpenAI API."""

    @pytest.fixture
    def llm_service(self):
        """Create a real LLM service."""
        service = LLMService(api_key=settings.openai_api_key)
        service.initialize()
        return service

    @pytest.mark.asyncio
    async def test_real_llm_generation(self, llm_service):
        """Test actual LLM response generation."""
        response = await llm_service.generate("What is 2 + 2?")

        assert response is not None
        assert len(response) > 0
        assert "4" in response

    @pytest.mark.asyncio
    async def test_llm_handles_complex_query(self, llm_service):
        """Test LLM with a more complex query."""
        response = await llm_service.generate(
            "Explain what a CPU is in one sentence."
        )

        assert response is not None
        assert len(response) > 10


@skip_no_openai
@skip_no_redis
class TestEndToEndIntegration:
    """Full end-to-end integration tests."""

    @pytest.fixture
    async def client(self):
        """Create a real HTTP client without mocks."""
        from unittest.mock import patch

        from app.main import app
        from app.services.cache import CacheService
        from app.services.llm import LLMService

        # Create services with correct URLs for local testing
        test_cache_service = CacheService(redis_url=get_redis_url())
        test_cache_service.connect()

        test_llm_service = LLMService(api_key=settings.openai_api_key)
        test_llm_service.initialize()

        # Patch the global services used by the app
        with patch("app.core.semantic_cache.cache_service", test_cache_service), \
             patch("app.core.semantic_cache.llm_service", test_llm_service), \
             patch("app.main.cache_service", test_cache_service), \
             patch("app.main.llm_service", test_llm_service):

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as c:
                yield c

        test_cache_service.close()

    @pytest.mark.asyncio
    async def test_full_query_flow_cache_miss(self, client):
        """Test a query that results in cache miss and LLM call."""
        import uuid

        # Use unique query to ensure cache miss
        unique_query = f"What is {uuid.uuid4().hex[:8]} in simple terms?"

        response = await client.post(
            "/api/query",
            json={"query": unique_query},
        )

        assert response.status_code == 200
        data = response.json()
        assert "response" in data
        assert data["metadata"]["source"] == "llm"

    @pytest.mark.asyncio
    async def test_full_query_flow_cache_hit(self, client):
        """Test that repeated queries hit the cache."""
        query = "What is the speed of light?"

        # First request - should hit LLM
        response1 = await client.post(
            "/api/query",
            json={"query": query},
        )
        assert response1.status_code == 200
        data1 = response1.json()

        # Second request - should hit cache
        response2 = await client.post(
            "/api/query",
            json={"query": query},
        )
        assert response2.status_code == 200
        data2 = response2.json()

        assert data2["metadata"]["source"] == "cache"
        assert data2["response"] == data1["response"]

    @pytest.mark.asyncio
    async def test_force_refresh_bypasses_cache(self, client):
        """Test that forceRefresh bypasses the cache."""
        query = "What is the boiling point of water?"

        # First request to populate cache
        await client.post("/api/query", json={"query": query})

        # Second request with forceRefresh
        response = await client.post(
            "/api/query",
            json={"query": query, "forceRefresh": True},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["metadata"]["source"] == "llm"

    @pytest.mark.asyncio
    async def test_semantic_similarity_cache_hit(self, client):
        """Test that semantically similar queries hit the cache."""
        # First query
        response1 = await client.post(
            "/api/query",
            json={"query": "What is the capital of Japan?"},
        )
        assert response1.status_code == 200
        original_response = response1.json()["response"]

        # Similar query - should hit cache
        response2 = await client.post(
            "/api/query",
            json={"query": "What's Japan's capital city?"},
        )
        assert response2.status_code == 200
        data2 = response2.json()

        # Should be cache hit with same response
        assert data2["metadata"]["source"] == "cache"
        assert data2["response"] == original_response
