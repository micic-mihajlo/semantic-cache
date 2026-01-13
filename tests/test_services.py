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

    def test_singleton_pattern(self):
        """Test that EmbeddingService uses singleton pattern."""
        from app.services.embedding import EmbeddingService

        service1 = EmbeddingService()
        service2 = EmbeddingService()

        assert service1 is service2, "Singleton pattern should return same instance"


class TestClassifier:
    """Tests for query classification."""

    def test_time_sensitive_weather_query(self):
        """Test that weather queries are classified as time-sensitive."""
        from app.services.classifier import classify

        query_type, confidence = classify("What's the weather in NYC today?")

        assert query_type == "time_sensitive"
        assert confidence >= 0.7

    def test_time_sensitive_news_query(self):
        """Test that news queries are classified as time-sensitive."""
        from app.services.classifier import classify

        query_type, confidence = classify("What are the latest news headlines?")

        assert query_type == "time_sensitive"
        assert confidence >= 0.7

    def test_time_sensitive_stock_query(self):
        """Test that stock queries are classified as time-sensitive."""
        from app.services.classifier import classify

        query_type, confidence = classify("What is the current bitcoin price?")

        assert query_type == "time_sensitive"
        assert confidence >= 0.7

    def test_evergreen_historical_query(self):
        """Test that historical queries are classified as evergreen."""
        from app.services.classifier import classify

        query_type, confidence = classify("Who was the first president of the USA?")

        assert query_type == "evergreen"
        assert confidence >= 0.7

    def test_evergreen_definition_query(self):
        """Test that definition queries are classified as evergreen."""
        from app.services.classifier import classify

        query_type, confidence = classify("What is the definition of democracy?")

        assert query_type == "evergreen"
        assert confidence >= 0.7

    def test_evergreen_howto_query(self):
        """Test that how-to queries are classified as evergreen."""
        from app.services.classifier import classify

        query_type, confidence = classify("How do you tie a tie?")

        assert query_type == "evergreen"
        assert confidence >= 0.7

    def test_default_evergreen_classification(self):
        """Test that queries without patterns default to evergreen."""
        from app.services.classifier import classify

        query_type, confidence = classify("What is the capital of France?")

        assert query_type == "evergreen"
        assert confidence == 0.6  # Default confidence

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
        """Test that API errors are converted to RuntimeError."""
        import httpx
        import openai
        from app.services.llm import LLMService

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

        with pytest.raises(RuntimeError, match="LLM service unavailable"):
            await service.generate("Test query")

    @pytest.mark.asyncio
    async def test_llm_rate_limit_error_handling(self):
        """Test that rate limit errors are converted to RuntimeError."""
        import openai
        from app.services.llm import LLMService

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

        with pytest.raises(RuntimeError, match="Rate limit exceeded"):
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
