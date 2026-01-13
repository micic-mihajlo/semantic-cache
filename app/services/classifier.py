"""Query classifier for time-sensitive vs evergreen detection and topic classification."""

import re
from dataclasses import dataclass

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

EVERGREEN_PATTERNS = [
    r"who was the first",
    r"what year did",
    r"definition of",
    r"what is a\b",
    r"how do you",
    r"history of",
]

TOPIC_PATTERNS: dict[str, list[str]] = {
    "weather": [
        r"\bweather\b",
        r"\bforecast\b",
        r"\btemperature\b",
        r"\brain(ing|y)?\b",
        r"\bsunny\b",
        r"\bcloudy\b",
        r"\bsnow(ing|y)?\b",
        r"\bhumidity\b",
        r"\bclimate\b",
    ],
    "finance": [
        r"\bstock\b",
        r"\bprice\b",
        r"\bmarket\b",
        r"\btrading\b",
        r"\bbitcoin\b",
        r"\bcrypto\b",
        r"\binvest(ment|ing)?\b",
        r"\bdividend\b",
        r"\bshares?\b",
        r"\bportfolio\b",
        r"\bindex\b",
        r"\bnasdaq\b",
        r"\bs&p\b",
    ],
    "sports": [
        r"\bscore\b",
        r"\bgame\b",
        r"\bmatch\b",
        r"\bteam\b",
        r"\bplayer\b",
        r"\bwon\b",
        r"\blost\b",
        r"\bchampion(ship)?\b",
        r"\bleague\b",
        r"\btournament\b",
        r"\bfootball\b",
        r"\bbasketball\b",
        r"\bsoccer\b",
        r"\btennis\b",
        r"\bolympic\b",
    ],
    "technology": [
        r"\bprogramming\b",
        r"\bsoftware\b",
        r"\bcode\b",
        r"\bcomputer\b",
        r"\balgorithm\b",
        r"\bdatabase\b",
        r"\bapi\b",
        r"\bpython\b",
        r"\bjavascript\b",
        r"\bjava\b",
        r"\brust\b",
        r"\bmachine learning\b",
        r"\bai\b",
        r"\bartificial intelligence\b",
        r"\bneural\b",
        r"\bdeep learning\b",
        r"\bframework\b",
        r"\blibrary\b",
    ],
    "science": [
        r"\bphysics\b",
        r"\bchemistry\b",
        r"\bbiology\b",
        r"\bmath(ematics)?\b",
        r"\bscien(ce|tific|tist)\b",
        r"\batom\b",
        r"\bmolecule\b",
        r"\bcell\b",
        r"\bdna\b",
        r"\bevolution\b",
        r"\btheory\b",
        r"\bexperiment\b",
        r"\bquantum\b",
        r"\brelativity\b",
        r"\bgravity\b",
    ],
    "history": [
        r"\bhistory\b",
        r"\bhistorical\b",
        r"\bwar\b",
        r"\bcentury\b",
        r"\bancient\b",
        r"\bempire\b",
        r"\bking\b",
        r"\bqueen\b",
        r"\bpresident\b",
        r"\brevolution\b",
        r"\bcivilization\b",
        r"\bcolonial\b",
        r"\bmedieval\b",
    ],
    "geography": [
        r"\bcapital\b",
        r"\bcountry\b",
        r"\bcity\b",
        r"\bcontinent\b",
        r"\bocean\b",
        r"\bmountain\b",
        r"\briver\b",
        r"\bisland\b",
        r"\bpopulation\b",
        r"\bgeograph(y|ical)\b",
        r"\blocation\b",
        r"\bregion\b",
    ],
    "news": [
        r"\bnews\b",
        r"\bheadlines?\b",
        r"\bbreaking\b",
        r"\breport(ed|ing)?\b",
        r"\bannounce(d|ment)?\b",
        r"\belection\b",
        r"\bpolitics\b",
        r"\bgovernment\b",
    ],
}

CACHING_PARAMS = {
    "time_sensitive": {"threshold": 0.15, "ttl": 300},
    "evergreen": {"threshold": 0.30, "ttl": 604800},
}


@dataclass
class QueryClassification:
    """Classification result for a query."""

    query_type: str
    topic: str
    threshold: float
    ttl: int


def classify(query: str) -> str:
    """Classify a query as time-sensitive or evergreen."""
    q = query.lower()

    for pattern in EVERGREEN_PATTERNS:
        if re.search(pattern, q):
            return "evergreen"

    time_matches = sum(1 for p in TIME_SENSITIVE_PATTERNS if re.search(p, q))
    if time_matches >= 1:
        return "time_sensitive"

    return "evergreen"


def classify_topic(query: str) -> str:
    """Classify a query into a topic category for cache partitioning."""
    q = query.lower()

    topic_scores: dict[str, int] = {}
    for topic, patterns in TOPIC_PATTERNS.items():
        score = sum(1 for p in patterns if re.search(p, q))
        if score > 0:
            topic_scores[topic] = score

    if topic_scores:
        return max(topic_scores, key=topic_scores.get)  # type: ignore[arg-type]

    return "general"


def classify_full(query: str) -> QueryClassification:
    """Full classification including type, topic, and caching params."""
    query_type = classify(query)
    topic = classify_topic(query)
    params = CACHING_PARAMS.get(query_type, CACHING_PARAMS["evergreen"])

    return QueryClassification(
        query_type=query_type,
        topic=topic,
        threshold=params["threshold"],
        ttl=params["ttl"],
    )


def get_caching_params(query_type: str) -> dict:
    """Get threshold and TTL for a query type."""
    return CACHING_PARAMS.get(query_type, CACHING_PARAMS["evergreen"])
