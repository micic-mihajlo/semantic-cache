# LLM Service Specification

## Overview

When no cache match exists, the system calls OpenAI's API to generate a response. The response is then cached for future similar queries.

## Model Selection: gpt-4o-mini

| Criteria | Value |
|----------|-------|
| Cost | ~$0.15 per 1M input tokens, ~$0.60 per 1M output tokens |
| Speed | Fast response times |
| Quality | Good for general queries |
| Availability | High uptime |

### Alternatives Considered

| Model | Cost | Speed | Quality |
|-------|------|-------|---------|
| gpt-4o-mini | Low | Fast | Good |
| gpt-4o | Medium | Medium | Better |
| gpt-4-turbo | High | Slow | Best |

Decision: gpt-4o-mini provides sufficient quality for cached responses at minimal cost.

## Implementation

```python
from openai import AsyncOpenAI
from app.config import settings

class LLMService:
    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)

    async def generate(self, query: str) -> str:
        response = await self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": query}],
            temperature=0,
        )
        return response.choices[0].message.content
```

### Key Implementation Details

1. **AsyncOpenAI**: Non-blocking for FastAPI async handlers
2. **temperature=0**: Deterministic outputs for consistent caching
3. **Simple message format**: Single user message, no system prompt

## Configuration

Required environment variable:
```
OPENAI_API_KEY=sk-your-key-here
```

## Error Handling

```python
async def generate(self, query: str) -> str:
    try:
        response = await self.client.chat.completions.create(...)
        return response.choices[0].message.content
    except openai.APIError as e:
        # Log error
        raise HTTPException(status_code=502, detail="LLM service unavailable")
    except openai.RateLimitError as e:
        # Log and potentially retry
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
```

## Acceptance Criteria

- [ ] LLM service initializes with API key from environment
- [ ] Async generation works with FastAPI
- [ ] Temperature is 0 for deterministic outputs
- [ ] Errors are handled gracefully with appropriate HTTP status codes
- [ ] Response content is extracted correctly from API response

## Rate Limiting Considerations

OpenAI has rate limits based on:
- Requests per minute (RPM)
- Tokens per minute (TPM)

For this application:
- Semantic caching reduces API calls significantly
- Most requests should be cache hits after warmup
- Consider exponential backoff for rate limit errors

## Future Optimizations

1. **Streaming**: Stream responses for perceived faster UX
2. **Retry logic**: Automatic retries with exponential backoff
3. **Fallback models**: Try cheaper models first, escalate if needed
4. **Batch requests**: Batch multiple queries when possible
