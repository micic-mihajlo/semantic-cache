"""Pytest fixtures for semantic cache tests."""

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient


@pytest.fixture
def mock_redis():
    """Mock Redis client for testing without a live Redis instance."""
    mock = MagicMock()
    mock.ft.return_value.info.side_effect = Exception("Index not found")
    mock.ft.return_value.create_index.return_value = None
    mock.ft.return_value.search.return_value = MagicMock(docs=[])
    mock.hset.return_value = None
    mock.expire.return_value = None
    mock.close.return_value = None
    return mock


@pytest.fixture
def mock_openai_response():
    """Mock OpenAI API response."""
    return "Paris is the capital of France."


@pytest.fixture
def mock_llm_service(mock_openai_response):
    """Mock LLM service to avoid real API calls."""
    with patch("app.services.llm.llm_service") as mock:
        mock.client = MagicMock()
        mock.generate = AsyncMock(return_value=mock_openai_response)
        mock.initialize = MagicMock()
        yield mock


@pytest.fixture
def mock_cache_service(mock_redis):
    """Mock cache service for isolated testing."""
    with patch("app.services.cache.cache_service") as mock:
        mock.redis_client = mock_redis
        mock.connect = MagicMock()
        mock.close = MagicMock()
        # Default to cache miss
        mock.search = MagicMock(return_value=None)
        mock.store = MagicMock()
        yield mock


@pytest.fixture
def mock_embedding_service():
    """Mock embedding service."""
    with patch("app.services.embedding.embedding_service") as mock:
        # Return consistent embeddings for testing
        mock.embed = MagicMock(
            return_value=np.random.rand(384).astype(np.float32)
        )
        yield mock


@pytest.fixture
async def client(mock_llm_service, mock_cache_service, mock_embedding_service):
    """Async HTTP client for testing FastAPI endpoints."""
    # Need to patch at the module level where they're used
    with patch("app.services.semantic_cache.llm_service", mock_llm_service), \
         patch("app.services.semantic_cache.cache_service", mock_cache_service), \
         patch("app.services.semantic_cache.embedding_service", mock_embedding_service), \
         patch("app.main.cache_service", mock_cache_service), \
         patch("app.main.llm_service", mock_llm_service):
        from app.main import app
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            yield c


@pytest.fixture
def cached_response():
    """Sample cached response for cache hit tests."""
    return {
        "query": "What is the capital of France?",
        "response": "Paris is the capital of France.",
        "distance": 0.05,
    }
