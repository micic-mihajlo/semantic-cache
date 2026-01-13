# Time-Sensitive vs Evergreen Query Classification

## Overview

Queries must be classified as either "time_sensitive" or "evergreen" to determine appropriate caching behavior. Time-sensitive queries need stricter matching and shorter TTL to avoid returning stale information.

## Classification Rules

### Time-Sensitive Patterns

Queries matching these patterns are time-sensitive:

**Temporal keywords:**
- today, now, current, currently, latest, recent
- yesterday, tomorrow, this week, tonight

**Domain-specific:**
- weather, forecast, temperature
- news, headlines, breaking
- stock, price, market, trading, bitcoin
- score, game, match, won, lost

### Evergreen Patterns

Queries matching these patterns are evergreen:

- "who was the first..."
- "what year did..."
- "definition of..."
- "what is a..."
- "how do you..."
- "history of..."

## Classification Logic

```python
def classify(query: str) -> tuple[str, float]:
    """Returns (query_type, confidence)."""
    q = query.lower()

    # Check evergreen patterns first
    for pattern in EVERGREEN_PATTERNS:
        if re.search(pattern, q):
            return ("evergreen", 0.9)

    # Count time-sensitive matches
    time_matches = sum(1 for p in TIME_SENSITIVE_PATTERNS if re.search(p, q))

    if time_matches >= 2:
        return ("time_sensitive", 0.95)
    elif time_matches == 1:
        return ("time_sensitive", 0.7)

    # Default to evergreen with lower confidence
    return ("evergreen", 0.6)
```

## Caching Parameters by Type

| Query Type | Distance Threshold | TTL |
|------------|-------------------|-----|
| time_sensitive | 0.15 (strict) | 300 seconds (5 min) |
| evergreen | 0.30 (relaxed) | 604800 seconds (7 days) |

### Distance Threshold Explanation

- **0.15 (strict)**: Only very similar queries match. "What's the weather in NYC?" won't match "What's the weather in LA?" because even small differences matter for time-sensitive data.
- **0.30 (relaxed)**: Broader matching allowed. "What is the capital of France?" can match "What's France's capital?" because the answer doesn't change.

## Acceptance Criteria

- [ ] "What's the weather today?" classified as time_sensitive
- [ ] "What's the stock price of AAPL?" classified as time_sensitive
- [ ] "Who was the first president?" classified as evergreen
- [ ] "What is the capital of France?" classified as evergreen
- [ ] Default classification is evergreen for ambiguous queries
- [ ] Time-sensitive queries use 5 min TTL
- [ ] Evergreen queries use 7 day TTL
- [ ] Time-sensitive queries use stricter distance threshold (0.15)
- [ ] Evergreen queries use relaxed distance threshold (0.30)

## Edge Cases

- Query with mixed signals: "What's the current definition of inflation?" - temporal keyword but asking for definition. Default to time_sensitive due to "current".
- Query in different languages: Fall back to evergreen with low confidence.
- Query with typos: Embedding handles this; classification may miss patterns.

## Implementation Notes

- Use regex for pattern matching (simple, fast, interpretable)
- Consider case-insensitive matching
- Return confidence score for potential future use (e.g., logging, adaptive thresholds)
