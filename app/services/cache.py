"""Redis cache service with vector similarity search."""

import hashlib
import logging
import time

import numpy as np
import redis
from redis.commands.search.field import NumericField, TagField, TextField, VectorField
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
            self.redis_client.config_set("maxmemory-policy", "volatile-ttl")
            logger.info("Redis eviction policy set to volatile-ttl")
        except redis.ResponseError as e:
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
                TagField("topic"),
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
        topic: str | None = None,
    ) -> dict | None:
        """Search for a semantically similar cached query."""
        if self.redis_client is None:
            logger.warning("Redis client not connected")
            return None

        if not redis_circuit.is_available():
            logger.warning("Redis circuit breaker is OPEN, skipping cache search")
            return None

        if topic and topic != "general":
            result = self._search_with_filter(embedding, threshold, topic)
            if result:
                logger.debug(f"Cache hit in topic partition: {topic}")
                return result
            logger.debug(f"No match in topic '{topic}', falling back to global search")

        return self._search_with_filter(embedding, threshold, None)

    def _search_with_filter(
        self,
        embedding: np.ndarray,
        threshold: float,
        topic: str | None,
    ) -> dict | None:
        """Execute a search with optional topic filter."""
        query_vector = embedding.astype(np.float32).tobytes()

        if topic:
            query_str = f"@topic:{{{topic}}}=>[KNN 1 @embedding $vec AS distance]"
        else:
            query_str = "*=>[KNN 1 @embedding $vec AS distance]"

        q = (
            Query(query_str)
            .return_fields("query", "response", "query_type", "topic", "distance")
            .sort_by("distance")
            .dialect(2)
        )

        try:
            results = self.redis_client.ft(self.INDEX_NAME).search(  # type: ignore[union-attr]
                q,
                {"vec": query_vector},
            )
            redis_circuit.record_success()

            if results.docs:
                doc = results.docs[0]
                distance = float(doc.distance)
                if distance <= threshold:
                    query_text = doc.query
                    response_text = doc.response
                    topic_text = getattr(doc, "topic", b"general")
                    if isinstance(query_text, bytes):
                        query_text = query_text.decode("utf-8")
                    if isinstance(response_text, bytes):
                        response_text = response_text.decode("utf-8")
                    if isinstance(topic_text, bytes):
                        topic_text = topic_text.decode("utf-8")
                    return {
                        "query": query_text,
                        "response": response_text,
                        "distance": distance,
                        "topic": topic_text,
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
        topic: str = "general",
    ) -> None:
        """Store a query-response pair in the cache."""
        if self.redis_client is None:
            logger.warning("Redis client not connected")
            return

        if not redis_circuit.is_available():
            logger.warning("Redis circuit breaker is OPEN, skipping cache store")
            return

        key = f"{self.KEY_PREFIX}{hashlib.md5(query.encode()).hexdigest()}"
        mapping = {
            "query": query.encode("utf-8"),
            "response": response.encode("utf-8"),
            "query_type": query_type.encode("utf-8"),
            "topic": topic.encode("utf-8"),
            "created_at": int(time.time()),
            "embedding": embedding.astype(np.float32).tobytes(),
        }

        try:
            self.redis_client.hset(key, mapping=mapping)
            self.redis_client.expire(key, ttl)
            redis_circuit.record_success()
            logger.debug(f"Cached query (topic={topic}) with TTL {ttl}s: {query[:50]}...")
        except redis.RedisError as e:
            redis_circuit.record_failure()
            logger.error(f"Redis store error: {e}")

    def close(self) -> None:
        """Close Redis connection."""
        if self.redis_client:
            self.redis_client.close()
            self.redis_client = None


cache_service = CacheService()
