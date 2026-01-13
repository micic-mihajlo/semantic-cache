"""Query classifier for time-sensitive vs evergreen detection."""

import re

# Temporal keywords indicating time-sensitive queries
TIME_SENSITIVE_PATTERNS = [
    r"\btoday\b",
    r"\bnow\b",
    r"\bcurrent(ly)?\b",
    r"\blatest\b",
    r"\brecent(ly)?\b",
    r"\byesterday\b",
    r"\btomorrow\b",
    r"\bthis week\b",
    r"\btonight\b",
    r"\bweather\b",
    r"\bforecast\b",
    r"\btemperature\b",
    r"\bnews\b",
    r"\bheadlines?\b",
    r"\bbreaking\b",
    r"\bstock\b",
    r"\bprice\b",
    r"\bmarket\b",
    r"\btrading\b",
    r"\bbitcoin\b",
    r"\bscore\b",
    r"\bgame\b",
    r"\bmatch\b",
    r"\bwon\b",
    r"\blost\b",
]

# Patterns indicating evergreen queries
EVERGREEN_PATTERNS = [
    r"who was the first",
    r"what year did",
    r"definition of",
    r"what is a\b",
    r"how do you",
    r"history of",
]

# Caching parameters by query type
CACHING_PARAMS = {
    "time_sensitive": {
        "threshold": 0.15,  # Strict matching
        "ttl": 300,  # 5 minutes
    },
    "evergreen": {
        "threshold": 0.30,  # Relaxed matching
        "ttl": 604800,  # 7 days
    },
}


def classify(query: str) -> tuple[str, float]:
    """
    Classify a query as time-sensitive or evergreen.

    Returns:
        tuple of (query_type, confidence) where query_type is
        "time_sensitive" or "evergreen" and confidence is 0.0-1.0
    """
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


def get_caching_params(query_type: str) -> dict:
    """Get threshold and TTL for a query type."""
    return CACHING_PARAMS.get(query_type, CACHING_PARAMS["evergreen"])
