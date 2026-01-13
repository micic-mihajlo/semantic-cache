"""FastAPI application entry point with lifespan management."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router
from app.services.cache import cache_service
from app.services.llm import llm_service

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown."""
    # Startup
    logger.info("Starting semantic cache service...")

    # Initialize LLM service
    llm_service.initialize()
    logger.info("LLM service initialized")

    # Connect to Redis
    try:
        cache_service.connect()
        logger.info("Connected to Redis")
    except Exception as e:
        logger.error(f"Failed to connect to Redis: {e}")
        raise

    logger.info("Semantic cache service started successfully")

    yield

    # Shutdown
    logger.info("Shutting down semantic cache service...")
    cache_service.close()
    logger.info("Semantic cache service stopped")


app = FastAPI(
    title="Semantic Cache API",
    description="AI-powered semantic caching system for LLM queries",
    version="1.0.0",
    lifespan=lifespan,
)

# Include API routes
app.include_router(router)
