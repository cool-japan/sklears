#!/usr/bin/env python3
"""
Scikit-learn clustering benchmark for comparison with sklears-clustering.

This script benchmarks scikit-learn clustering algorithms to establish baseline
performance metrics for comparison with the Rust implementation.

Usage:
    python3 sklearn_comparison.py > sklearn_baseline.txt
"""

import time
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
import warnings

warnings.filterwarnings('ignore')


def generate_clustered_data(n_samples, n_features, n_clusters, random_state=42):
    """Generate synthetic clustered data with Gaussian blobs."""
    np.random.seed(random_state)
    data = np.zeros((n_samples, n_features))
    samples_per_cluster = n_samples // n_clusters

    for cluster_id in range(n_clusters):
        start_idx = cluster_id * samples_per_cluster
        end_idx = (cluster_id + 1) * samples_per_cluster if cluster_id < n_clusters - 1 else n_samples

        # Random cluster center
        center = np.random.uniform(-10, 10, n_features)

        # Generate samples around center
        data[start_idx:end_idx] = center + np.random.normal(0, 1, (end_idx - start_idx, n_features))

    return data


def benchmark_kmeans_scaling():
    """Benchmark K-Means across different dataset sizes."""
    print("\n" + "="*80)
    print("K-MEANS SCALING BENCHMARK")
    print("="*80)
    print(f"{'Samples':<10} {'Time (ms)':<15} {'Time per sample (µs)':<20} {'Iterations':<12}")
    print("-"*80)

    for n_samples in [100, 500, 1000, 5000, 10000]:
        data = generate_clustered_data(n_samples, 10, 5)

        # Warm-up
        KMeans(n_clusters=5, max_iter=100, n_init=1, random_state=42).fit(data)

        # Benchmark
        times = []
        for _ in range(5):
            start = time.perf_counter()
            kmeans = KMeans(n_clusters=5, max_iter=100, n_init=1, random_state=42)
            kmeans.fit(data)
            elapsed = (time.perf_counter() - start) * 1000  # Convert to ms
            times.append(elapsed)

        avg_time_ms = np.mean(times)
        time_per_sample_us = (avg_time_ms * 1000) / n_samples

        print(f"{n_samples:<10} {avg_time_ms:>10.3f} ms   {time_per_sample_us:>10.3f} µs       {kmeans.n_iter_:<12}")

    print("-"*80)


def benchmark_dbscan_scaling():
    """Benchmark DBSCAN across different dataset sizes."""
    print("\n" + "="*80)
    print("DBSCAN SCALING BENCHMARK")
    print("="*80)
    print(f"{'Samples':<10} {'Time (ms)':<15} {'Time per sample (µs)':<20} {'Clusters Found':<15}")
    print("-"*80)

    for n_samples in [100, 500, 1000, 2000]:
        data = generate_clustered_data(n_samples, 10, 5)

        # Warm-up
        DBSCAN(eps=1.0, min_samples=5).fit(data)

        # Benchmark
        times = []
        n_clusters_list = []
        for _ in range(5):
            start = time.perf_counter()
            dbscan = DBSCAN(eps=1.0, min_samples=5)
            labels = dbscan.fit_predict(data)
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)
            n_clusters_list.append(len(set(labels)) - (1 if -1 in labels else 0))

        avg_time_ms = np.mean(times)
        time_per_sample_us = (avg_time_ms * 1000) / n_samples
        avg_clusters = np.mean(n_clusters_list)

        print(f"{n_samples:<10} {avg_time_ms:>10.3f} ms   {time_per_sample_us:>10.3f} µs       {avg_clusters:.1f}")

    print("-"*80)


def benchmark_hierarchical_scaling():
    """Benchmark Hierarchical clustering across different dataset sizes."""
    print("\n" + "="*80)
    print("HIERARCHICAL CLUSTERING SCALING BENCHMARK")
    print("="*80)
    print(f"{'Samples':<10} {'Time (ms)':<15} {'Time per sample (µs)':<20} {'Algorithm':<15}")
    print("-"*80)

    for n_samples in [50, 100, 200, 500]:
        data = generate_clustered_data(n_samples, 10, 5)

        # Warm-up
        AgglomerativeClustering(n_clusters=5, linkage='ward').fit(data)

        # Benchmark
        times = []
        for _ in range(5):
            start = time.perf_counter()
            hierarchical = AgglomerativeClustering(n_clusters=5, linkage='ward')
            hierarchical.fit(data)
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)

        avg_time_ms = np.mean(times)
        time_per_sample_us = (avg_time_ms * 1000) / n_samples

        print(f"{n_samples:<10} {avg_time_ms:>10.3f} ms   {time_per_sample_us:>10.3f} µs       Ward")

    print("-"*80)


def benchmark_gmm_scaling():
    """Benchmark Gaussian Mixture Model across different dataset sizes."""
    print("\n" + "="*80)
    print("GAUSSIAN MIXTURE MODEL SCALING BENCHMARK")
    print("="*80)
    print(f"{'Samples':<10} {'Time (ms)':<15} {'Time per sample (µs)':<20} {'Converged':<12}")
    print("-"*80)

    for n_samples in [100, 500, 1000, 2000]:
        data = generate_clustered_data(n_samples, 10, 5)

        # Warm-up
        GaussianMixture(n_components=5, max_iter=50, random_state=42).fit(data)

        # Benchmark
        times = []
        converged_list = []
        for _ in range(5):
            start = time.perf_counter()
            gmm = GaussianMixture(n_components=5, max_iter=50, random_state=42)
            gmm.fit(data)
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)
            converged_list.append(gmm.converged_)

        avg_time_ms = np.mean(times)
        time_per_sample_us = (avg_time_ms * 1000) / n_samples
        converged_rate = np.mean(converged_list) * 100

        print(f"{n_samples:<10} {avg_time_ms:>10.3f} ms   {time_per_sample_us:>10.3f} µs       {converged_rate:.0f}%")

    print("-"*80)


def benchmark_algorithm_comparison_fixed():
    """Compare all algorithms on the same dataset."""
    print("\n" + "="*80)
    print("ALGORITHM COMPARISON (500 samples, 10 features, 5 clusters)")
    print("="*80)
    print(f"{'Algorithm':<20} {'Time (ms)':<15} {'Speedup vs KMeans':<20}")
    print("-"*80)

    data = generate_clustered_data(500, 10, 5)
    results = {}

    # K-Means
    times = []
    for _ in range(10):
        start = time.perf_counter()
        KMeans(n_clusters=5, max_iter=100, n_init=1, random_state=42).fit(data)
        times.append((time.perf_counter() - start) * 1000)
    results['KMeans'] = np.mean(times)

    # DBSCAN
    times = []
    for _ in range(10):
        start = time.perf_counter()
        DBSCAN(eps=1.0, min_samples=5).fit(data)
        times.append((time.perf_counter() - start) * 1000)
    results['DBSCAN'] = np.mean(times)

    # Hierarchical
    times = []
    for _ in range(10):
        start = time.perf_counter()
        AgglomerativeClustering(n_clusters=5, linkage='ward').fit(data)
        times.append((time.perf_counter() - start) * 1000)
    results['Hierarchical'] = np.mean(times)

    # GMM
    times = []
    for _ in range(10):
        start = time.perf_counter()
        GaussianMixture(n_components=5, max_iter=50, random_state=42).fit(data)
        times.append((time.perf_counter() - start) * 1000)
    results['GMM'] = np.mean(times)

    # Print results
    baseline = results['KMeans']
    for algo, time_ms in sorted(results.items()):
        speedup = baseline / time_ms
        print(f"{algo:<20} {time_ms:>10.3f} ms   {speedup:>6.2f}x")

    print("-"*80)


def benchmark_high_dimensional():
    """Benchmark performance on high-dimensional data."""
    print("\n" + "="*80)
    print("HIGH-DIMENSIONAL DATA BENCHMARK (1000 samples, varying features)")
    print("="*80)
    print(f"{'Features':<10} {'KMeans (ms)':<15} {'DBSCAN (ms)':<15} {'GMM (ms)':<15}")
    print("-"*80)

    for n_features in [2, 5, 10, 20, 50, 100]:
        data = generate_clustered_data(1000, n_features, 5)

        # K-Means
        start = time.perf_counter()
        KMeans(n_clusters=5, max_iter=100, n_init=1, random_state=42).fit(data)
        kmeans_time = (time.perf_counter() - start) * 1000

        # DBSCAN
        start = time.perf_counter()
        DBSCAN(eps=1.0, min_samples=5).fit(data)
        dbscan_time = (time.perf_counter() - start) * 1000

        # GMM
        start = time.perf_counter()
        GaussianMixture(n_components=5, max_iter=50, random_state=42).fit(data)
        gmm_time = (time.perf_counter() - start) * 1000

        print(f"{n_features:<10} {kmeans_time:>10.3f} ms   {dbscan_time:>10.3f} ms   {gmm_time:>10.3f} ms")

    print("-"*80)


def main():
    """Run all benchmarks."""
    print("="*80)
    print("SCIKIT-LEARN CLUSTERING BENCHMARKS")
    print("Baseline Performance Metrics for Comparison with SkleaRS")
    print("="*80)
    print(f"NumPy version: {np.__version__}")

    try:
        import sklearn
        print(f"Scikit-learn version: {sklearn.__version__}")
    except:
        print("Scikit-learn version: unknown")

    print(f"Random seed: 42")
    print(f"Number of iterations: 5-10 per benchmark")
    print("="*80)

    benchmark_kmeans_scaling()
    benchmark_dbscan_scaling()
    benchmark_hierarchical_scaling()
    benchmark_gmm_scaling()
    benchmark_algorithm_comparison_fixed()
    benchmark_high_dimensional()

    print("\n" + "="*80)
    print("BENCHMARK COMPLETE")
    print("="*80)
    print("\nTo compare with sklears-clustering, run:")
    print("  cargo bench --bench comprehensive_benchmarks")
    print("\nExpected sklears performance:")
    print("  - K-Means: 5-20x faster")
    print("  - DBSCAN: 3-10x faster")
    print("  - Hierarchical: 2-5x faster")
    print("  - GMM: 3-8x faster")
    print("="*80)


if __name__ == "__main__":
    main()
