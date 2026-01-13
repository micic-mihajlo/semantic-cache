"""Load testing script for semantic cache API using Locust.

Run with: locust -f loadtest/locustfile.py --host=http://localhost:3000

Or headless mode:
    locust -f loadtest/locustfile.py --host=http://localhost:3000 \
           --headless -u 10 -r 2 -t 60s
"""

import random

from locust import HttpUser, between, task

# Sample queries for testing
EVERGREEN_QUERIES = [
    "What is the capital of France?",
    "What is Python programming language?",
    "Who was the first president of the United States?",
    "What is the definition of democracy?",
    "How do you calculate the area of a circle?",
    "What is machine learning?",
    "Explain what a CPU is",
    "What is the speed of light?",
    "What is the boiling point of water?",
    "What is photosynthesis?",
]

TIME_SENSITIVE_QUERIES = [
    "What is the weather in New York today?",
    "What are the latest news headlines?",
    "What is the current bitcoin price?",
    "What is the stock price of Apple now?",
    "What is the weather in Los Angeles today?",
    "What are the latest sports scores?",
]

# Variations for semantic similarity testing
QUERY_VARIATIONS = [
    ("What is Python?", "What is the Python language?", "Explain Python programming"),
    ("Capital of France", "What is France's capital?", "What city is the capital of France?"),
    ("What is AI?", "What is artificial intelligence?", "Explain AI to me"),
]


class SemanticCacheUser(HttpUser):
    """Simulated user for load testing the semantic cache API."""

    wait_time = between(0.5, 2.0)  # Wait 0.5-2 seconds between requests

    @task(10)
    def query_evergreen(self):
        """Query with evergreen (cacheable) content."""
        query = random.choice(EVERGREEN_QUERIES)
        self.client.post(
            "/api/query",
            json={"query": query},
            name="/api/query (evergreen)",
        )

    @task(5)
    def query_time_sensitive(self):
        """Query with time-sensitive content."""
        query = random.choice(TIME_SENSITIVE_QUERIES)
        self.client.post(
            "/api/query",
            json={"query": query},
            name="/api/query (time_sensitive)",
        )

    @task(8)
    def query_variations(self):
        """Query with semantic variations to test cache hits."""
        variation_group = random.choice(QUERY_VARIATIONS)
        query = random.choice(variation_group)
        self.client.post(
            "/api/query",
            json={"query": query},
            name="/api/query (variation)",
        )

    @task(2)
    def query_force_refresh(self):
        """Query with force refresh to bypass cache."""
        query = random.choice(EVERGREEN_QUERIES)
        self.client.post(
            "/api/query",
            json={"query": query, "forceRefresh": True},
            name="/api/query (force_refresh)",
        )

    @task(1)
    def check_stats(self):
        """Check cache statistics."""
        self.client.get("/api/stats", name="/api/stats")

    @task(1)
    def check_health(self):
        """Check health endpoint."""
        self.client.get("/health", name="/health")

    @task(1)
    def check_circuits(self):
        """Check circuit breaker status."""
        self.client.get("/api/circuits", name="/api/circuits")


class HeavyLoadUser(HttpUser):
    """User that generates heavy load with rapid requests."""

    wait_time = between(0.1, 0.5)  # Very short wait times

    @task
    def rapid_queries(self):
        """Rapid-fire queries to stress test."""
        queries = EVERGREEN_QUERIES + TIME_SENSITIVE_QUERIES
        query = random.choice(queries)
        self.client.post(
            "/api/query",
            json={"query": query},
            name="/api/query (rapid)",
        )
