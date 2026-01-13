"""LLM service wrapper for OpenAI API."""

import logging

import openai
from openai import AsyncOpenAI

from app.config import settings

logger = logging.getLogger(__name__)


class LLMService:
    """Async OpenAI client wrapper for LLM generation."""

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or settings.openai_api_key
        self.client: AsyncOpenAI | None = None

    def initialize(self) -> None:
        """Initialize the OpenAI client."""
        if self.api_key:
            self.client = AsyncOpenAI(api_key=self.api_key)
        else:
            logger.warning("No OpenAI API key configured")

    async def generate(self, query: str) -> str:
        """
        Generate a response for a query using the LLM.

        Args:
            query: User query text

        Returns:
            Generated response text

        Raises:
            RuntimeError: If LLM client not initialized or API error occurs
        """
        if self.client is None:
            raise RuntimeError("LLM client not initialized. Check OPENAI_API_KEY.")

        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": query}],
                temperature=0,
            )
            content = response.choices[0].message.content
            return content if content else ""
        except openai.APIError as e:
            logger.error(f"OpenAI API error: {e}")
            raise RuntimeError(f"LLM service unavailable: {e}")
        except openai.RateLimitError as e:
            logger.error(f"OpenAI rate limit error: {e}")
            raise RuntimeError(f"Rate limit exceeded: {e}")


# Global instance
llm_service = LLMService()
