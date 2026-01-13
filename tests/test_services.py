"""Unit tests for individual services."""

import numpy as np
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestEmbeddingService:
    """Tests for embedding service functionality."""

    def test_embedding_dimensions(self):
        """Test that embeddings are 384-dimensional."""
        from app.services.embedding import EmbeddingService

        # Create a fresh instance for testing
        service = EmbeddingService()
        embedding = service.embed("What is the capital of France?")

        assert embedding.shape == (384,), f"Expected 384 dimensions, got {embedding.shape}"

    def test_embedding_normalized(self):
        """Test that embeddings are normalized to unit length."""
        from app.services.embedding import EmbeddingService

        service = EmbeddingService()
        embedding = service.embed("What is the capital of France?")

        # Normalized vectors have magnitude ~1.0
        magnitude = np.linalg.norm(embedding)
        assert abs(magnitude - 1.0) < 0.001, f"Expected magnitude ~1.0, got {magnitude}"

    def test_embedding_type(self):
        """Test that embeddings are numpy float32 arrays."""
        from app.services.embedding import EmbeddingService

        service = EmbeddingService()
        embedding = service.embed("Test query")

        assert isinstance(embedding, np.ndarray), f"Expected numpy array, got {type(embedding)}"
        assert embedding.dtype == np.float32, f"Expected float32, got {embedding.dtype}"

    def test_similar_queries_low_distance(self):
        """Test that semantically similar queries have low cosine distance."""
        from app.services.embedding import EmbeddingService

        service = EmbeddingService()

        emb1 = service.embed("What is the capital of France?")
        emb2 = service.embed("What's France's capital city?")

        # Cosine distance = 1 - cosine_similarity
        # For normalized vectors: cosine_similarity = dot product
        similarity = np.dot(emb1, emb2)
        distance = 1 - similarity

        # Similar queries should have distance < 0.3 (evergreen threshold)
        assert distance < 0.3, f"Similar queries should have distance < 0.3, got {distance}"

    def test_unrelated_queries_high_distance(self):
        """Test that unrelated queries have high cosine distance."""
        from app.services.embedding import EmbeddingService

        service = EmbeddingService()

        emb1 = service.embed("What is the capital of France?")
        emb2 = service.embed("How to make chocolate cake?")

        similarity = np.dot(emb1, emb2)
        distance = 1 - similarity

        # Unrelated queries should have distance > 0.3
        assert distance > 0.3, f"Unrelated queries should have distance > 0.3, got {distance}"

    def test_multiple_instances_work(self):
        """Test that multiple EmbeddingService instances work independently."""
        from app.services.embedding import EmbeddingService

        service1 = EmbeddingService()
        service2 = EmbeddingService()

        # Both should produce valid embeddings
        emb1 = service1.embed("Test query")
        emb2 = service2.embed("Test query")

        assert emb1.shape == (384,)
        assert emb2.shape == (384,)


class TestClassifier:
    """Tests for query classification."""

    def test_time_sensitive_weather_query(self):
        """Test that weather queries are classified as time-sensitive."""
        from app.services.classifier import classify

        assert classify("What's the weather in NYC today?") == "time_sensitive"

    def test_time_sensitive_news_query(self):
        """Test that news queries are classified as time-sensitive."""
        from app.services.classifier import classify

        assert classify("What are the latest news headlines?") == "time_sensitive"

    def test_time_sensitive_stock_query(self):
        """Test that stock queries are classified as time-sensitive."""
        from app.services.classifier import classify

        assert classify("What is the current bitcoin price?") == "time_sensitive"

    def test_evergreen_historical_query(self):
        """Test that historical queries are classified as evergreen."""
        from app.services.classifier import classify

        assert classify("Who was the first president of the USA?") == "evergreen"

    def test_evergreen_definition_query(self):
        """Test that definition queries are classified as evergreen."""
        from app.services.classifier import classify

        assert classify("What is the definition of democracy?") == "evergreen"

    def test_evergreen_howto_query(self):
        """Test that how-to queries are classified as evergreen."""
        from app.services.classifier import classify

        assert classify("How do you tie a tie?") == "evergreen"

    def test_default_evergreen_classification(self):
        """Test that queries without patterns default to evergreen."""
        from app.services.classifier import classify

        assert classify("What is the capital of France?") == "evergreen"

    def test_caching_params_time_sensitive(self):
        """Test caching parameters for time-sensitive queries."""
        from app.services.classifier import get_caching_params

        params = get_caching_params("time_sensitive")

        assert params["threshold"] == 0.15  # Strict matching
        assert params["ttl"] == 300  # 5 minutes

    def test_caching_params_evergreen(self):
        """Test caching parameters for evergreen queries."""
        from app.services.classifier import get_caching_params

        params = get_caching_params("evergreen")

        assert params["threshold"] == 0.30  # Relaxed matching
        assert params["ttl"] == 604800  # 7 days


class TestLLMService:
    """Tests for LLM service error handling."""

    @pytest.mark.asyncio
    async def test_llm_uninitialized_error(self):
        """Test that uninitialized LLM raises RuntimeError."""
        from app.services.llm import LLMService

        service = LLMService(api_key=None)
        # Don't call initialize - client remains None

        with pytest.raises(RuntimeError, match="LLM client not initialized"):
            await service.generate("Test query")

    @pytest.mark.asyncio
    async def test_llm_api_error_handling(self):
        """Test that API errors are converted to LLMServiceUnavailableError."""
        import httpx
        import openai
        from app.services.llm import LLMService, LLMServiceUnavailableError

        service = LLMService(api_key="test-key")
        service.initialize()

        # Mock the client to raise APIError
        mock_request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
        service.client.chat.completions.create = AsyncMock(
            side_effect=openai.APIError(
                message="API error",
                request=mock_request,
                body=None
            )
        )

        with pytest.raises(LLMServiceUnavailableError, match="LLM service unavailable"):
            await service.generate("Test query")

    @pytest.mark.asyncio
    async def test_llm_rate_limit_error_handling(self):
        """Test that rate limit errors are converted to LLMRateLimitError."""
        import openai
        from app.services.llm import LLMService, LLMRateLimitError

        service = LLMService(api_key="test-key")
        service.initialize()

        # Mock the client to raise RateLimitError
        mock_response = MagicMock()
        mock_response.status_code = 429
        service.client.chat.completions.create = AsyncMock(
            side_effect=openai.RateLimitError(
                message="Rate limit exceeded",
                response=mock_response,
                body=None
            )
        )

        with pytest.raises(LLMRateLimitError, match="Rate limit exceeded"):
            await service.generate("Test query")

    @pytest.mark.asyncio
    async def test_llm_successful_generation(self):
        """Test successful LLM response generation."""
        from app.services.llm import LLMService

        service = LLMService(api_key="test-key")
        service.initialize()

        # Mock successful response
        mock_message = MagicMock()
        mock_message.content = "Paris is the capital of France."
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        service.client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await service.generate("What is the capital of France?")

        assert result == "Paris is the capital of France."

    @pytest.mark.asyncio
    async def test_llm_empty_content_handling(self):
        """Test that empty content returns empty string."""
        from app.services.llm import LLMService

        service = LLMService(api_key="test-key")
        service.initialize()

        # Mock response with None content
        mock_message = MagicMock()
        mock_message.content = None
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        service.client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await service.generate("Test query")

        assert result == ""


class TestCacheService:
    """Tests for cache service functionality."""

    def test_cache_service_initialization(self):
        """Test cache service initializes with default or provided URL."""
        from app.services.cache import CacheService

        service = CacheService()
        assert service.redis_url is not None

        custom_service = CacheService(redis_url="redis://custom:6379")
        assert custom_service.redis_url == "redis://custom:6379"

    def test_search_returns_none_when_not_connected(self):
        """Test search returns None when Redis not connected."""
        from app.services.cache import CacheService

        service = CacheService()
        # Don't connect - redis_client is None

        result = service.search(np.random.rand(384).astype(np.float32), 0.3)

        assert result is None

    def test_store_logs_warning_when_not_connected(self):
        """Test store handles missing connection gracefully."""
        from app.services.cache import CacheService

        service = CacheService()
        # Don't connect - redis_client is None

        # Should not raise, just log warning
        service.store(
            query="test",
            response="response",
            embedding=np.random.rand(384).astype(np.float32),
            query_type="evergreen",
            ttl=300
        )

    def test_close_handles_none_client(self):
        """Test close handles None client gracefully."""
        from app.services.cache import CacheService

        service = CacheService()
        # redis_client is None

        # Should not raise
        service.close()
        assert service.redis_client is None

    def test_search_with_mocked_redis(self):
        """Test search with mocked Redis client returning cache hit."""
        from app.services.cache import CacheService

        service = CacheService()
        mock_redis = MagicMock()

        # Mock search result
        mock_doc = MagicMock()
        mock_doc.distance = "0.05"
        mock_doc.query = b"What is the capital of France?"
        mock_doc.response = b"Paris is the capital of France."
        mock_result = MagicMock()
        mock_result.docs = [mock_doc]

        mock_redis.ft.return_value.search.return_value = mock_result
        service.redis_client = mock_redis

        embedding = np.random.rand(384).astype(np.float32)
        result = service.search(embedding, threshold=0.3)

        assert result is not None
        assert result["distance"] == 0.05
        assert result["response"] == "Paris is the capital of France."

    def test_search_returns_none_above_threshold(self):
        """Test search returns None when distance exceeds threshold."""
        from app.services.cache import CacheService

        service = CacheService()
        mock_redis = MagicMock()

        # Mock search result with high distance
        mock_doc = MagicMock()
        mock_doc.distance = "0.5"  # Above threshold
        mock_doc.query = b"test"
        mock_doc.response = b"response"
        mock_result = MagicMock()
        mock_result.docs = [mock_doc]

        mock_redis.ft.return_value.search.return_value = mock_result
        service.redis_client = mock_redis

        embedding = np.random.rand(384).astype(np.float32)
        result = service.search(embedding, threshold=0.3)

        assert result is None

    def test_search_returns_none_on_empty_results(self):
        """Test search returns None when no documents found."""
        from app.services.cache import CacheService

        service = CacheService()
        mock_redis = MagicMock()

        # Mock empty search result
        mock_result = MagicMock()
        mock_result.docs = []

        mock_redis.ft.return_value.search.return_value = mock_result
        service.redis_client = mock_redis

        embedding = np.random.rand(384).astype(np.float32)
        result = service.search(embedding, threshold=0.3)

        assert result is None

    def test_search_handles_redis_error(self):
        """Test search handles Redis errors gracefully."""
        import redis
        from app.services.cache import CacheService

        service = CacheService()
        mock_redis = MagicMock()
        mock_redis.ft.return_value.search.side_effect = redis.ResponseError("Search error")
        service.redis_client = mock_redis

        embedding = np.random.rand(384).astype(np.float32)
        result = service.search(embedding, threshold=0.3)

        assert result is None

    def test_store_with_mocked_redis(self):
        """Test store correctly stores data in Redis."""
        from app.services.cache import CacheService

        service = CacheService()
        mock_redis = MagicMock()
        service.redis_client = mock_redis

        embedding = np.random.rand(384).astype(np.float32)
        service.store(
            query="What is the capital of France?",
            response="Paris",
            embedding=embedding,
            query_type="evergreen",
            ttl=604800
        )

        mock_redis.hset.assert_called_once()
        mock_redis.expire.assert_called_once()

    def test_store_handles_redis_error(self):
        """Test store handles Redis errors gracefully."""
        import redis
        from app.services.cache import CacheService

        service = CacheService()
        mock_redis = MagicMock()
        mock_redis.hset.side_effect = redis.RedisError("Store error")
        service.redis_client = mock_redis

        embedding = np.random.rand(384).astype(np.float32)
        # Should not raise
        service.store(
            query="test",
            response="response",
            embedding=embedding,
            query_type="evergreen",
            ttl=300
        )

    def test_close_closes_connection(self):
        """Test close properly closes Redis connection."""
        from app.services.cache import CacheService

        service = CacheService()
        mock_redis = MagicMock()
        service.redis_client = mock_redis

        service.close()

        mock_redis.close.assert_called_once()
        assert service.redis_client is None

    def test_ensure_index_creates_index(self):
        """Test _ensure_index creates index when not exists."""
        import redis
        from app.services.cache import CacheService

        service = CacheService()
        mock_redis = MagicMock()
        mock_redis.ft.return_value.info.side_effect = redis.ResponseError("Index not found")
        service.redis_client = mock_redis

        service._ensure_index()

        mock_redis.ft.return_value.create_index.assert_called_once()

    def test_ensure_index_skips_existing(self):
        """Test _ensure_index skips creation when index exists."""
        from app.services.cache import CacheService

        service = CacheService()
        mock_redis = MagicMock()
        mock_redis.ft.return_value.info.return_value = {"index_name": "cache_index"}
        service.redis_client = mock_redis

        service._ensure_index()

        mock_redis.ft.return_value.create_index.assert_not_called()

    def test_ensure_index_handles_none_client(self):
        """Test _ensure_index handles None client gracefully."""
        from app.services.cache import CacheService

        service = CacheService()
        # redis_client is None

        # Should not raise
        service._ensure_index()

    def test_configure_eviction_policy(self):
        """Test _configure_eviction_policy sets volatile-ttl."""
        from app.services.cache import CacheService

        service = CacheService()
        mock_redis = MagicMock()
        service.redis_client = mock_redis

        service._configure_eviction_policy()

        mock_redis.config_set.assert_called_once_with("maxmemory-policy", "volatile-ttl")

    def test_configure_eviction_policy_handles_error(self):
        """Test _configure_eviction_policy handles error gracefully."""
        import redis
        from app.services.cache import CacheService

        service = CacheService()
        mock_redis = MagicMock()
        mock_redis.config_set.side_effect = redis.ResponseError("Config not allowed")
        service.redis_client = mock_redis

        # Should not raise
        service._configure_eviction_policy()

    def test_configure_eviction_policy_handles_none_client(self):
        """Test _configure_eviction_policy handles None client gracefully."""
        from app.services.cache import CacheService

        service = CacheService()
        # redis_client is None

        # Should not raise
        service._configure_eviction_policy()


class TestMetrics:
    """Tests for metrics service functionality."""

    def test_metrics_initialization(self):
        """Test metrics initializes with zero values."""
        from app.services.metrics import Metrics

        m = Metrics()
        stats = m.get_stats()

        assert stats["total_queries"] == 0
        assert stats["cache_hits"] == 0
        assert stats["cache_misses"] == 0
        assert stats["hit_rate_percent"] == 0.0

    def test_record_cache_hit(self):
        """Test recording cache hits."""
        from app.services.metrics import Metrics

        m = Metrics()
        m.record_cache_hit(50.0)
        m.record_cache_hit(100.0)

        stats = m.get_stats()
        assert stats["total_queries"] == 2
        assert stats["cache_hits"] == 2
        assert stats["cache_misses"] == 0
        assert stats["hit_rate_percent"] == 100.0
        assert stats["latency"]["avg_cache_ms"] == 75.0

    def test_record_cache_miss(self):
        """Test recording cache misses."""
        from app.services.metrics import Metrics

        m = Metrics()
        m.record_cache_miss(1000.0)

        stats = m.get_stats()
        assert stats["total_queries"] == 1
        assert stats["cache_hits"] == 0
        assert stats["cache_misses"] == 1
        assert stats["llm_calls"] == 1
        assert stats["hit_rate_percent"] == 0.0
        assert stats["latency"]["avg_llm_ms"] == 1000.0

    def test_record_query_type(self):
        """Test recording query types."""
        from app.services.metrics import Metrics

        m = Metrics()
        m.record_query_type("time_sensitive")
        m.record_query_type("time_sensitive")
        m.record_query_type("evergreen")

        stats = m.get_stats()
        assert stats["query_types"]["time_sensitive"] == 2
        assert stats["query_types"]["evergreen"] == 1

    def test_record_error(self):
        """Test recording errors."""
        from app.services.metrics import Metrics

        m = Metrics()
        m.record_error()
        m.record_error()

        stats = m.get_stats()
        assert stats["errors"] == 2

    def test_reset(self):
        """Test resetting metrics."""
        from app.services.metrics import Metrics

        m = Metrics()
        m.record_cache_hit(100.0)
        m.record_cache_miss(500.0)
        m.record_query_type("evergreen")
        m.record_error()

        m.reset()
        stats = m.get_stats()

        assert stats["total_queries"] == 0
        assert stats["cache_hits"] == 0
        assert stats["cache_misses"] == 0
        assert stats["errors"] == 0

    def test_hit_rate_calculation(self):
        """Test hit rate percentage calculation."""
        from app.services.metrics import Metrics

        m = Metrics()
        m.record_cache_hit(10.0)
        m.record_cache_hit(10.0)
        m.record_cache_miss(100.0)

        stats = m.get_stats()
        assert stats["hit_rate_percent"] == 66.67  # 2/3 = 66.67%


class TestCircuitBreaker:
    """Tests for circuit breaker functionality."""

    def test_circuit_starts_closed(self):
        """Test circuit starts in closed state."""
        from app.services.circuit_breaker import CircuitBreaker, CircuitState

        cb = CircuitBreaker(name="test")
        assert cb.state == CircuitState.CLOSED
        assert cb.is_available() is True

    def test_circuit_opens_after_failures(self):
        """Test circuit opens after threshold failures."""
        from app.services.circuit_breaker import CircuitBreaker, CircuitState

        cb = CircuitBreaker(name="test", failure_threshold=3)

        cb.record_failure()
        assert cb.state == CircuitState.CLOSED

        cb.record_failure()
        assert cb.state == CircuitState.CLOSED

        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        assert cb.is_available() is False

    def test_circuit_resets_on_success(self):
        """Test circuit resets failure count on success."""
        from app.services.circuit_breaker import CircuitBreaker, CircuitState

        cb = CircuitBreaker(name="test", failure_threshold=3)

        cb.record_failure()
        cb.record_failure()
        cb.record_success()

        # Failure count should reset
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.CLOSED  # Still closed

    def test_circuit_half_open_after_timeout(self):
        """Test circuit enters half-open after recovery timeout."""
        from app.services.circuit_breaker import CircuitBreaker, CircuitState

        cb = CircuitBreaker(name="test", failure_threshold=1, recovery_timeout=0.1)

        cb.record_failure()  # Opens circuit
        assert cb.state == CircuitState.OPEN

        import time
        time.sleep(0.15)

        assert cb.state == CircuitState.HALF_OPEN
        assert cb.is_available() is True

    def test_circuit_closes_on_half_open_success(self):
        """Test circuit closes after successful call in half-open state."""
        from app.services.circuit_breaker import CircuitBreaker, CircuitState

        cb = CircuitBreaker(name="test", failure_threshold=1, recovery_timeout=0.1)

        cb.record_failure()

        import time
        time.sleep(0.15)

        cb.is_available()  # Triggers transition to half-open
        cb.record_success()

        assert cb.state == CircuitState.CLOSED

    def test_get_status(self):
        """Test get_status returns correct info."""
        from app.services.circuit_breaker import CircuitBreaker

        cb = CircuitBreaker(name="test_service", failure_threshold=5, recovery_timeout=30.0)
        cb.record_failure()

        status = cb.get_status()
        assert status["name"] == "test_service"
        assert status["state"] == "closed"
        assert status["failure_count"] == 1
        assert status["failure_threshold"] == 5
