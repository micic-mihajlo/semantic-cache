"""Pydantic models for API request/response validation."""

from pydantic import BaseModel, Field, field_validator


class QueryRequest(BaseModel):
    """Request body for the query endpoint."""

    query: str = Field(..., min_length=1, description="The user's query text")
    forceRefresh: bool = Field(
        default=False,
        description="If true, bypass cache and force LLM call",
    )

    @field_validator("query")
    @classmethod
    def query_must_not_be_whitespace_only(cls, v: str) -> str:
        """Validate that query is not whitespace-only."""
        if not v.strip():
            raise ValueError("Query cannot be empty or whitespace-only")
        return v


class QueryMetadata(BaseModel):
    """Metadata about query response."""

    source: str = Field(..., description='Either "cache" or "llm"')
    confidence: float | None = Field(
        default=None,
        description="Confidence score (0-1) for cache hits based on similarity. Higher is better."
    )


class QueryResponse(BaseModel):
    """Response body for the query endpoint."""

    response: str = Field(..., description="The answer to the query")
    metadata: QueryMetadata


class ErrorResponse(BaseModel):
    """Error response body."""

    error: str = Field(..., description="Error message")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(default="ok")


class LatencyStats(BaseModel):
    """Latency statistics."""

    avg_total_ms: float = Field(..., description="Average total latency in ms")
    avg_cache_ms: float = Field(..., description="Average cache hit latency in ms")
    avg_llm_ms: float = Field(..., description="Average LLM call latency in ms")


class QueryTypeStats(BaseModel):
    """Query type statistics."""

    time_sensitive: int = Field(..., description="Number of time-sensitive queries")
    evergreen: int = Field(..., description="Number of evergreen queries")


class StatsResponse(BaseModel):
    """Cache statistics response."""

    total_queries: int = Field(..., description="Total number of queries processed")
    cache_hits: int = Field(..., description="Number of cache hits")
    cache_misses: int = Field(..., description="Number of cache misses")
    hit_rate_percent: float = Field(..., description="Cache hit rate percentage")
    llm_calls: int = Field(..., description="Number of LLM API calls")
    errors: int = Field(..., description="Number of errors")
    latency: LatencyStats
    query_types: QueryTypeStats
