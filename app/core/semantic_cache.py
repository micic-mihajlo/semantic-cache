"""Semantic cache manager - orchestrates classification, embedding, caching, and LLM calls."""

import logging

from app.services.cache import cache_service
from app.services.classifier import classify, get_caching_params
from app.services.embedding import embedding_service
from app.services.llm import llm_service

logger = logging.getLogger(__name__)


class SemanticCacheManager:
    """
    Main orchestrator for the semantic caching system.

    Flow:
    1. Classify query (time_sensitive vs evergreen)
    2. Generate embedding
    3. Search cache with appropriate threshold
    4. On hit: return cached response
    5. On miss: call LLM, cache response, return
    """

    async def process_query(
        self,
        query: str,
        force_refresh: bool = False,
    ) -> dict:
        """
        Process a query through the semantic cache system.

        Args:
            query: User query text
            force_refresh: If True, bypass cache and force LLM call

        Returns:
            Dict with 'response' and 'metadata' containing 'source'
        """
        # Classify the query to determine caching parameters
        query_type, confidence = classify(query)
        params = get_caching_params(query_type)
        threshold = params["threshold"]
        ttl = params["ttl"]

        logger.debug(
            f"Query classified as {query_type} (confidence: {confidence:.2f}), "
            f"threshold: {threshold}, ttl: {ttl}s"
        )

        # Generate embedding for the query
        embedding = embedding_service.embed(query)

        # Check cache unless force refresh is requested
        if not force_refresh:
            cached = cache_service.search(embedding, threshold)
            if cached:
                logger.info(
                    f"Cache hit (distance: {cached['distance']:.4f}): {query[:50]}..."
                )
                return {
                    "response": cached["response"],
                    "metadata": {"source": "cache"},
                }

        # Cache miss or force refresh - call LLM
        logger.info(f"Cache miss, calling LLM: {query[:50]}...")
        response = await llm_service.generate(query)

        # Store in cache
        cache_service.store(query, response, embedding, query_type, ttl)

        return {
            "response": response,
            "metadata": {"source": "llm"},
        }


# Global instance
semantic_cache_manager = SemanticCacheManager()
