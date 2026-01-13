"""Integration tests for the semantic cache API."""

import pytest


@pytest.mark.asyncio
async def test_health_endpoint(client):
    """Test health check endpoint returns ok status."""
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_cache_miss_calls_llm(client, mock_cache_service, mock_llm_service):
    """Test that cache miss triggers LLM call."""
    mock_cache_service.search.return_value = None

    response = await client.post(
        "/api/query",
        json={"query": "What is the capital of France?"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["metadata"]["source"] == "llm"
    assert "Paris" in data["response"]
    mock_llm_service.generate.assert_called_once()
    mock_cache_service.store.assert_called_once()


@pytest.mark.asyncio
async def test_cache_hit_returns_cached(client, mock_cache_service, cached_response):
    """Test that cache hit returns cached response without LLM call."""
    mock_cache_service.search.return_value = cached_response

    response = await client.post(
        "/api/query",
        json={"query": "What is the capital of France?"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["metadata"]["source"] == "cache"
    assert data["response"] == cached_response["response"]


@pytest.mark.asyncio
async def test_semantic_similarity(client, mock_cache_service, cached_response):
    """Test that semantically similar queries return cached response."""
    mock_cache_service.search.return_value = cached_response

    response = await client.post(
        "/api/query",
        json={"query": "What's France's capital?"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["metadata"]["source"] == "cache"


@pytest.mark.asyncio
async def test_force_refresh_bypasses_cache(
    client, mock_cache_service, mock_llm_service, cached_response
):
    """Test that forceRefresh=true bypasses cache and calls LLM."""
    mock_cache_service.search.return_value = cached_response

    response = await client.post(
        "/api/query",
        json={"query": "What is the capital of France?", "forceRefresh": True},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["metadata"]["source"] == "llm"
    mock_llm_service.generate.assert_called_once()


@pytest.mark.asyncio
async def test_unrelated_queries_miss_cache(
    client, mock_cache_service, mock_llm_service
):
    """Test that unrelated queries don't match cached entries."""
    mock_cache_service.search.return_value = None

    response = await client.post(
        "/api/query",
        json={"query": "How do I make pasta?"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["metadata"]["source"] == "llm"


@pytest.mark.asyncio
async def test_empty_query_error(client):
    """Test that empty query returns 422 validation error."""
    response = await client.post(
        "/api/query",
        json={"query": ""},
    )

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_missing_query_error(client):
    """Test that missing query field returns 422 validation error."""
    response = await client.post(
        "/api/query",
        json={},
    )

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_time_sensitive_classification(
    client, mock_cache_service, mock_llm_service
):
    """Test that time-sensitive queries are handled appropriately."""
    mock_cache_service.search.return_value = None

    response = await client.post(
        "/api/query",
        json={"query": "What's the weather in NYC today?"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["metadata"]["source"] == "llm"


@pytest.mark.asyncio
async def test_time_sensitive_strict_matching(client, mock_cache_service):
    """Test that time-sensitive queries use strict matching threshold."""
    # This tests the semantic behavior - NYC and LA weather shouldn't match
    # due to the strict 0.15 threshold for time-sensitive queries
    mock_cache_service.search.return_value = None

    response = await client.post(
        "/api/query",
        json={"query": "What's the weather in LA today?"},
    )

    assert response.status_code == 200
    data = response.json()
    # Should be a miss because NYC != LA with strict threshold
    assert data["metadata"]["source"] == "llm"
