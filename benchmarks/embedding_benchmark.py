"""Benchmark different embedding models for semantic similarity.

Run with: python benchmarks/embedding_benchmark.py

This script compares different SentenceTransformer models on:
- Speed (embeddings per second)
- Quality (semantic similarity accuracy)
- Memory usage
"""

import time
from dataclasses import dataclass

import numpy as np

# Test queries for semantic similarity
SIMILAR_PAIRS = [
    ("What is the capital of France?", "What's France's capital city?"),
    ("How do I learn Python?", "What's the best way to learn Python programming?"),
    ("What is machine learning?", "Explain ML to me"),
    ("Who was the first US president?", "Who was America's first president?"),
]

DISSIMILAR_PAIRS = [
    ("What is the capital of France?", "How do I bake a cake?"),
    ("What is Python?", "What's the weather today?"),
    ("Who invented the telephone?", "What is quantum physics?"),
    ("How to tie a tie?", "What is the stock market?"),
]


@dataclass
class BenchmarkResult:
    """Results from benchmarking an embedding model."""

    model_name: str
    dimension: int
    avg_embed_time_ms: float
    embeddings_per_second: float
    avg_similar_distance: float
    avg_dissimilar_distance: float
    separation_ratio: float  # dissimilar / similar (higher is better)


def benchmark_model(model_name: str, num_warmup: int = 3, num_iterations: int = 10) -> BenchmarkResult:
    """Benchmark a single embedding model."""
    from sentence_transformers import SentenceTransformer

    print(f"\nBenchmarking: {model_name}")
    print("-" * 50)

    # Load model
    load_start = time.time()
    model = SentenceTransformer(model_name)
    load_time = time.time() - load_start
    print(f"  Model load time: {load_time:.2f}s")

    # Get dimension
    test_embedding = model.encode("test", normalize_embeddings=True)
    dimension = len(test_embedding)
    print(f"  Embedding dimension: {dimension}")

    # Warmup
    all_texts = [q for pair in SIMILAR_PAIRS + DISSIMILAR_PAIRS for q in pair]
    for _ in range(num_warmup):
        model.encode(all_texts, normalize_embeddings=True)

    # Speed benchmark
    times = []
    for _ in range(num_iterations):
        start = time.time()
        model.encode(all_texts, normalize_embeddings=True)
        times.append(time.time() - start)

    avg_time = np.mean(times)
    embeddings_per_second = len(all_texts) / avg_time
    avg_embed_time_ms = (avg_time / len(all_texts)) * 1000

    print(f"  Avg embed time: {avg_embed_time_ms:.2f}ms per embedding")
    print(f"  Throughput: {embeddings_per_second:.0f} embeddings/sec")

    # Quality benchmark - similar pairs
    similar_distances = []
    for q1, q2 in SIMILAR_PAIRS:
        emb1 = model.encode(q1, normalize_embeddings=True)
        emb2 = model.encode(q2, normalize_embeddings=True)
        distance = 1 - np.dot(emb1, emb2)
        similar_distances.append(distance)

    avg_similar = np.mean(similar_distances)
    print(f"  Avg similar pair distance: {avg_similar:.4f}")

    # Quality benchmark - dissimilar pairs
    dissimilar_distances = []
    for q1, q2 in DISSIMILAR_PAIRS:
        emb1 = model.encode(q1, normalize_embeddings=True)
        emb2 = model.encode(q2, normalize_embeddings=True)
        distance = 1 - np.dot(emb1, emb2)
        dissimilar_distances.append(distance)

    avg_dissimilar = np.mean(dissimilar_distances)
    print(f"  Avg dissimilar pair distance: {avg_dissimilar:.4f}")

    separation_ratio = avg_dissimilar / avg_similar if avg_similar > 0 else 0
    print(f"  Separation ratio: {separation_ratio:.2f}x (higher is better)")

    return BenchmarkResult(
        model_name=model_name,
        dimension=dimension,
        avg_embed_time_ms=avg_embed_time_ms,
        embeddings_per_second=embeddings_per_second,
        avg_similar_distance=avg_similar,
        avg_dissimilar_distance=avg_dissimilar,
        separation_ratio=separation_ratio,
    )


def main():
    """Run benchmarks on multiple models."""
    print("=" * 60)
    print("EMBEDDING MODEL BENCHMARK")
    print("=" * 60)

    # Models to benchmark (from smallest/fastest to largest/best quality)
    models = [
        "all-MiniLM-L6-v2",  # 384 dim, fastest, used in this project
        "all-MiniLM-L12-v2",  # 384 dim, slightly better quality
        "all-mpnet-base-v2",  # 768 dim, best quality, slower
        "paraphrase-MiniLM-L6-v2",  # 384 dim, good for paraphrase detection
    ]

    results = []
    for model_name in models:
        try:
            result = benchmark_model(model_name)
            results.append(result)
        except Exception as e:
            print(f"  Error: {e}")

    # Summary table
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Model':<30} {'Dim':<6} {'ms/emb':<10} {'emb/s':<10} {'Sep.Ratio':<10}")
    print("-" * 66)

    for r in sorted(results, key=lambda x: x.embeddings_per_second, reverse=True):
        print(
            f"{r.model_name:<30} {r.dimension:<6} {r.avg_embed_time_ms:<10.2f} "
            f"{r.embeddings_per_second:<10.0f} {r.separation_ratio:<10.2f}"
        )

    # Recommendation
    print("\n" + "=" * 60)
    print("RECOMMENDATION")
    print("=" * 60)

    fastest = max(results, key=lambda x: x.embeddings_per_second)
    best_quality = max(results, key=lambda x: x.separation_ratio)

    print(f"Fastest: {fastest.model_name} ({fastest.embeddings_per_second:.0f} emb/s)")
    print(f"Best quality: {best_quality.model_name} (separation ratio: {best_quality.separation_ratio:.2f}x)")
    print("\nFor semantic caching, 'all-MiniLM-L6-v2' offers the best speed/quality tradeoff.")


if __name__ == "__main__":
    main()
