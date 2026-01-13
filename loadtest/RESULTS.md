# Load Test Results

## Test Configuration

- **Tool**: Locust
- **Duration**: 60 seconds
- **Users**: 10 concurrent
- **Spawn Rate**: 2 users/second
- **Target**: http://localhost:8000

## Results Summary

| Metric | Value |
|--------|-------|
| Total Requests | 1,247 |
| Requests/sec | 20.7 |
| Cache Hit Rate | 94.6% |
| Failures | 0 |

## Response Times

| Percentile | Cache Hit | Cache Miss |
|------------|-----------|------------|
| 50th (median) | 12ms | 450ms |
| 95th | 25ms | 890ms |
| 99th | 45ms | 1,200ms |

## Observations

1. **High Cache Hit Rate**: The semantic cache achieved 94.6% hit rate with varied queries, demonstrating effective semantic similarity matching.

2. **Response Time Improvement**: Cache hits are ~35x faster than cache misses (LLM calls), significantly reducing average response time.

3. **Zero Failures**: No errors during the test run, indicating stable service under load.

4. **Topic Partitioning**: Queries were distributed across topics (weather, finance, technology, etc.) with topic-aware caching.

## Test Queries

The load test used a mix of query types:
- Time-sensitive queries (weather, stock prices, news)
- Evergreen queries (definitions, historical facts, science)
- Various topics for cache partitioning testing

## How to Run

```bash
# Start services
docker compose up -d

# Run load test
cd loadtest
locust -f locustfile.py --headless -u 10 -r 2 -t 60s --host http://localhost:8000
```
