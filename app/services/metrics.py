"""Metrics service for tracking cache performance."""

import time
from dataclasses import dataclass, field
from threading import Lock


@dataclass
class Metrics:
    """Thread-safe metrics collector for cache performance."""

    _lock: Lock = field(default_factory=Lock, repr=False)

    # Counters
    total_queries: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    llm_calls: int = 0
    errors: int = 0

    # Latency tracking (in milliseconds)
    total_latency_ms: float = 0.0
    cache_latency_ms: float = 0.0
    llm_latency_ms: float = 0.0

    # Query type counters
    time_sensitive_queries: int = 0
    evergreen_queries: int = 0

    def record_cache_hit(self, latency_ms: float) -> None:
        """Record a cache hit with latency."""
        with self._lock:
            self.total_queries += 1
            self.cache_hits += 1
            self.total_latency_ms += latency_ms
            self.cache_latency_ms += latency_ms

    def record_cache_miss(self, latency_ms: float) -> None:
        """Record a cache miss (LLM call) with latency."""
        with self._lock:
            self.total_queries += 1
            self.cache_misses += 1
            self.llm_calls += 1
            self.total_latency_ms += latency_ms
            self.llm_latency_ms += latency_ms

    def record_query_type(self, query_type: str) -> None:
        """Record query classification."""
        with self._lock:
            if query_type == "time_sensitive":
                self.time_sensitive_queries += 1
            else:
                self.evergreen_queries += 1

    def record_error(self) -> None:
        """Record an error."""
        with self._lock:
            self.errors += 1

    def get_stats(self) -> dict:
        """Get current statistics."""
        with self._lock:
            hit_rate = (
                (self.cache_hits / self.total_queries * 100)
                if self.total_queries > 0
                else 0.0
            )
            avg_latency = (
                (self.total_latency_ms / self.total_queries)
                if self.total_queries > 0
                else 0.0
            )
            avg_cache_latency = (
                (self.cache_latency_ms / self.cache_hits)
                if self.cache_hits > 0
                else 0.0
            )
            avg_llm_latency = (
                (self.llm_latency_ms / self.llm_calls)
                if self.llm_calls > 0
                else 0.0
            )

            return {
                "total_queries": self.total_queries,
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "hit_rate_percent": round(hit_rate, 2),
                "llm_calls": self.llm_calls,
                "errors": self.errors,
                "latency": {
                    "avg_total_ms": round(avg_latency, 2),
                    "avg_cache_ms": round(avg_cache_latency, 2),
                    "avg_llm_ms": round(avg_llm_latency, 2),
                },
                "query_types": {
                    "time_sensitive": self.time_sensitive_queries,
                    "evergreen": self.evergreen_queries,
                },
            }

    def reset(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self.total_queries = 0
            self.cache_hits = 0
            self.cache_misses = 0
            self.llm_calls = 0
            self.errors = 0
            self.total_latency_ms = 0.0
            self.cache_latency_ms = 0.0
            self.llm_latency_ms = 0.0
            self.time_sensitive_queries = 0
            self.evergreen_queries = 0


# Global metrics instance
metrics = Metrics()
