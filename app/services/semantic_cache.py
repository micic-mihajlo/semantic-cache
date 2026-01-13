"""Semantic cache manager - orchestrates classification, embedding, caching, and LLM calls."""

import logging
import time

from app.services.cache import cache_service
from app.services.classifier import classify, get_caching_params
from app.services.embedding import embedding_service
from app.services.llm import llm_service
from app.services.metrics import metrics

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
        start_time = time.time()

        # Classify the query to determine caching parameters
        query_type = classify(query)
        params = get_caching_params(query_type)
        threshold = params["threshold"]
        ttl = params["ttl"]

        # Record query type
        metrics.record_query_type(query_type)

        logger.debug(f"Query classified as {query_type}, threshold: {threshold}, ttl: {ttl}s")

        # Generate embedding for the query
        embedding = embedding_service.embed(query)

        # Check cache unless force refresh is requested
        if not force_refresh:
            cached = cache_service.search(embedding, threshold)
            if cached:
                latency_ms = (time.time() - start_time) * 1000
                metrics.record_cache_hit(latency_ms)
                # Convert distance to confidence (0 distance = 1.0 confidence)
                confidence = round(1.0 - cached["distance"], 4)
                logger.info(
                    f"Cache hit (distance: {cached['distance']:.4f}, confidence: {confidence}): {query[:50]}..."
                )
                return {
                    "response": cached["response"],
                    "metadata": {"source": "cache", "confidence": confidence},
                }

        # Cache miss or force refresh - call LLM
        logger.info(f"Cache miss, calling LLM: {query[:50]}...")
        response = await llm_service.generate(query)

        # Store in cache
        cache_service.store(query, response, embedding, query_type, ttl)

        latency_ms = (time.time() - start_time) * 1000
        metrics.record_cache_miss(latency_ms)

        return {
            "response": response,
            "metadata": {"source": "llm"},
        }


# Global instance
semantic_cache_manager = SemanticCacheManager()
