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
    confidence: float | None = Field(default=None)
    topic: str | None = Field(default=None)


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

    avg_total_ms: float
    avg_cache_ms: float
    avg_llm_ms: float


class QueryTypeStats(BaseModel):
    """Query type statistics."""

    time_sensitive: int
    evergreen: int


class StatsResponse(BaseModel):
    """Cache statistics response."""

    total_queries: int
    cache_hits: int
    cache_misses: int
    hit_rate_percent: float
    llm_calls: int
    errors: int
    latency: LatencyStats
    query_types: QueryTypeStats
    topics: dict[str, int] = Field(default_factory=dict)
