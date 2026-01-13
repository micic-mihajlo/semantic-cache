"""Pydantic models for API request/response validation."""

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Request body for the query endpoint."""

    query: str = Field(..., min_length=1, description="The user's query text")
    forceRefresh: bool = Field(
        default=False,
        description="If true, bypass cache and force LLM call",
    )


class QueryMetadata(BaseModel):
    """Metadata about query response."""

    source: str = Field(..., description='Either "cache" or "llm"')


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
