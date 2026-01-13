"""Redis cache service with vector similarity search."""

import hashlib
import logging
import time

import numpy as np
import redis
from redis.commands.search.field import NumericField, TextField, VectorField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query

from app.config import settings
from app.services.circuit_breaker import redis_circuit

logger = logging.getLogger(__name__)


class CacheService:
    """Redis cache with vector similarity search capabilities."""

    INDEX_NAME = "cache_index"
    KEY_PREFIX = "cache:"

    def __init__(self, redis_url: str | None = None):
        self.redis_url = redis_url or settings.redis_url
        self.redis_client: redis.Redis | None = None

    def connect(self) -> None:
        """Connect to Redis and ensure index exists."""
        self.redis_client = redis.from_url(self.redis_url, decode_responses=False)
        self._configure_eviction_policy()
        self._ensure_index()

    def _configure_eviction_policy(self) -> None:
        """Configure Redis eviction policy to volatile-ttl."""
        if self.redis_client is None:
            return

        try:
            # Set eviction policy to volatile-ttl (evicts keys with shortest TTL when memory full)
            self.redis_client.config_set("maxmemory-policy", "volatile-ttl")
            logger.info("Redis eviction policy set to volatile-ttl")
        except redis.ResponseError as e:
            # Some Redis configurations may not allow runtime config changes
            logger.warning(f"Could not set eviction policy: {e}")

    def _ensure_index(self) -> None:
        """Create RediSearch index if it doesn't exist."""
        if self.redis_client is None:
            logger.warning("Redis client not connected, cannot ensure index")
            return

        try:
            self.redis_client.ft(self.INDEX_NAME).info()
            logger.info("Redis index already exists")
        except redis.ResponseError:
            logger.info("Creating Redis index")
            schema = [
                TextField("query"),
                TextField("response"),
                TextField("query_type"),
                NumericField("created_at"),
                VectorField(
                    "embedding",
                    "FLAT",
                    {
                        "TYPE": "FLOAT32",
                        "DIM": 384,
                        "DISTANCE_METRIC": "COSINE",
                    },
                ),
            ]
            definition = IndexDefinition(
                prefix=[self.KEY_PREFIX],
                index_type=IndexType.HASH,
            )
            self.redis_client.ft(self.INDEX_NAME).create_index(
                schema,
                definition=definition,
            )
            logger.info("Redis index created successfully")

    def search(
        self,
        embedding: np.ndarray,
        threshold: float,
    ) -> dict | None:
        """
        Search for a semantically similar cached query.

        Args:
            embedding: Query embedding vector
            threshold: Maximum distance threshold for a match

        Returns:
            Dict with query, response, distance if match found, else None
        """
        if self.redis_client is None:
            logger.warning("Redis client not connected")
            return None

        # Check circuit breaker
        if not redis_circuit.is_available():
            logger.warning("Redis circuit breaker is OPEN, skipping cache search")
            return None

        query_vector = embedding.astype(np.float32).tobytes()
        q = (
            Query("*=>[KNN 1 @embedding $vec AS distance]")
            .return_fields("query", "response", "query_type", "distance")
            .sort_by("distance")
            .dialect(2)
        )

        try:
            results = self.redis_client.ft(self.INDEX_NAME).search(
                q,
                {"vec": query_vector},
            )
            redis_circuit.record_success()

            if results.docs:
                doc = results.docs[0]
                distance = float(doc.distance)
                if distance <= threshold:
                    # Decode bytes to strings
                    query_text = doc.query
                    response_text = doc.response
                    if isinstance(query_text, bytes):
                        query_text = query_text.decode("utf-8")
                    if isinstance(response_text, bytes):
                        response_text = response_text.decode("utf-8")
                    return {
                        "query": query_text,
                        "response": response_text,
                        "distance": distance,
                    }
        except redis.ResponseError as e:
            redis_circuit.record_failure()
            logger.error(f"Redis search error: {e}")

        return None

    def store(
        self,
        query: str,
        response: str,
        embedding: np.ndarray,
        query_type: str,
        ttl: int,
    ) -> None:
        """
        Store a query-response pair in the cache.

        Args:
            query: Original query text
            response: LLM response
            embedding: Query embedding vector
            query_type: "time_sensitive" or "evergreen"
            ttl: Time to live in seconds
        """
        if self.redis_client is None:
            logger.warning("Redis client not connected")
            return

        # Check circuit breaker
        if not redis_circuit.is_available():
            logger.warning("Redis circuit breaker is OPEN, skipping cache store")
            return

        key = f"{self.KEY_PREFIX}{hashlib.md5(query.encode()).hexdigest()}"
        mapping = {
            "query": query.encode("utf-8"),
            "response": response.encode("utf-8"),
            "query_type": query_type.encode("utf-8"),
            "created_at": int(time.time()),
            "embedding": embedding.astype(np.float32).tobytes(),
        }

        try:
            self.redis_client.hset(key, mapping=mapping)
            self.redis_client.expire(key, ttl)
            redis_circuit.record_success()
            logger.debug(f"Cached query with TTL {ttl}s: {query[:50]}...")
        except redis.RedisError as e:
            redis_circuit.record_failure()
            logger.error(f"Redis store error: {e}")

    def close(self) -> None:
        """Close Redis connection."""
        if self.redis_client:
            self.redis_client.close()
            self.redis_client = None


# Global instance
cache_service = CacheService()
