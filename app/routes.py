"""API routes for the semantic cache service."""

import logging

from fastapi import APIRouter, HTTPException

from app.schemas import (
    ErrorResponse,
    HealthResponse,
    QueryRequest,
    QueryResponse,
)
from app.services.semantic_cache import semantic_cache_manager
from app.services.llm import LLMRateLimitError, LLMServiceUnavailableError

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/api/query",
    response_model=QueryResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
        502: {"model": ErrorResponse, "description": "LLM service unavailable"},
    },
)
async def query(request: QueryRequest) -> QueryResponse:
    """
    Submit a query for processing.

    Returns a response from cache if a semantically similar query exists,
    otherwise calls the LLM and caches the result.
    """
    try:
        result = await semantic_cache_manager.process_query(
            query=request.query,
            force_refresh=request.forceRefresh,
        )
        return QueryResponse(
            response=result["response"],
            metadata=result["metadata"],
        )
    except LLMRateLimitError as e:
        logger.error(f"Rate limit exceeded: {e}")
        raise HTTPException(status_code=429, detail=str(e))
    except LLMServiceUnavailableError as e:
        logger.error(f"LLM service unavailable: {e}")
        raise HTTPException(status_code=502, detail=str(e))
    except RuntimeError as e:
        logger.error(f"Query processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.exception(f"Unexpected error processing query: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(status="ok")
