# REST API Specification

## Overview

The API server accepts query requests, returns responses (from cache or LLM), and provides metadata on cache hits/misses.

## Endpoints

### POST /api/query

Submit a query for processing.

**Request Body:**
```json
{
  "query": "What's the weather like in New York today?",
  "forceRefresh": false
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| query | string | Yes | The user's query text |
| forceRefresh | boolean | No | If true, bypass cache and force LLM call (default: false) |

**Response Body:**
```json
{
  "response": "The weather in New York today is sunny with a high of 75°F.",
  "metadata": {
    "source": "cache"
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| response | string | The answer to the query |
| metadata.source | string | Either "cache" or "llm" indicating where the response came from |

**Status Codes:**
- 200: Success
- 400: Invalid request (missing query)
- 500: Internal server error

### GET /health

Health check endpoint.

**Response:**
```json
{
  "status": "ok"
}
```

## Acceptance Criteria

- [ ] POST /api/query accepts JSON body with query field
- [ ] Response includes metadata.source indicating "cache" or "llm"
- [ ] forceRefresh=true bypasses cache and calls LLM
- [ ] Empty or missing query returns 400 error
- [ ] GET /health returns status ok when service is healthy

## Error Handling

All errors return JSON with error field:
```json
{
  "error": "Query is required"
}
```

## Implementation Notes

- Use Pydantic models for request/response validation
- FastAPI router for endpoint definitions
- Async handlers for non-blocking operation
