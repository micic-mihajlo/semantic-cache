"""Semantic cache manager - orchestrates the caching pipeline."""

import logging
import time

from app.services.cache import cache_service
from app.services.classifier import classify_full
from app.services.embedding import embedding_service
from app.services.llm import llm_service
from app.services.metrics import metrics

logger = logging.getLogger(__name__)


class SemanticCacheManager:
    """Main orchestrator for the semantic caching system."""

    async def process_query(self, query: str, force_refresh: bool = False) -> dict:
        """Process a query through the semantic cache system."""
        start_time = time.time()

        classification = classify_full(query)
        query_type = classification.query_type
        topic = classification.topic
        threshold = classification.threshold
        ttl = classification.ttl

        metrics.record_query_type(query_type)
        metrics.record_topic(topic)

        logger.debug(
            f"Query classified as {query_type}, topic={topic}, "
            f"threshold: {threshold}, ttl: {ttl}s"
        )

        embedding = embedding_service.embed(query)

        if not force_refresh:
            cached = cache_service.search(embedding, threshold, topic)
            if cached:
                latency_ms = (time.time() - start_time) * 1000
                metrics.record_cache_hit(latency_ms)
                confidence = round(1.0 - cached["distance"], 4)
                cached_topic = cached.get("topic", "unknown")
                logger.info(
                    f"Cache hit (distance: {cached['distance']:.4f}, "
                    f"confidence: {confidence}, topic: {cached_topic}): {query[:50]}..."
                )
                return {
                    "response": cached["response"],
                    "metadata": {"source": "cache", "confidence": confidence, "topic": cached_topic},
                }

        logger.info(f"Cache miss (topic={topic}), calling LLM: {query[:50]}...")
        response = await llm_service.generate(query)

        cache_service.store(query, response, embedding, query_type, ttl, topic)

        latency_ms = (time.time() - start_time) * 1000
        metrics.record_cache_miss(latency_ms)

        return {
            "response": response,
            "metadata": {"source": "llm", "topic": topic},
        }


semantic_cache_manager = SemanticCacheManager()
