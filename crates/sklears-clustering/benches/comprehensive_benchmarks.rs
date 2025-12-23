///! Comprehensive benchmarks for sklears-clustering algorithms
///!
///! This benchmark suite compares performance across multiple clustering algorithms
///! on various dataset sizes and characteristics to establish baseline performance metrics.
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use scirs2_core::ndarray::Array2;
use scirs2_core::random::essentials::Normal;
use scirs2_core::random::{thread_rng, Distribution, Rng};
use sklears_clustering::prelude::*;
use sklears_core::prelude::*;

/// Generate synthetic clustered data with Gaussian blobs
fn generate_clustered_data(n_samples: usize, n_features: usize, n_clusters: usize) -> Array2<f64> {
    let mut rng = thread_rng();
    let normal = Normal::new(0.0, 1.0).unwrap();
    let mut data = Array2::zeros((n_samples, n_features));

    let samples_per_cluster = n_samples / n_clusters;

    for cluster_id in 0..n_clusters {
        let start_idx = cluster_id * samples_per_cluster;
        let end_idx = if cluster_id == n_clusters - 1 {
            n_samples
        } else {
            (cluster_id + 1) * samples_per_cluster
        };

        // Random cluster center
        let center: Vec<f64> = (0..n_features)
            .map(|_| rng.random_range(-10.0..10.0))
            .collect();

        for i in start_idx..end_idx {
            for j in 0..n_features {
                data[[i, j]] = center[j] + normal.sample(&mut rng);
            }
        }
    }

    data
}

/// Benchmark K-Means performance across dataset sizes
fn bench_kmeans_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("kmeans_scaling");

    for n_samples in [100, 500, 1000, 5000, 10000] {
        let data = generate_clustered_data(n_samples, 10, 5);

        group.throughput(Throughput::Elements(n_samples as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n_samples), &data, |b, data| {
            b.iter(|| {
                let config = KMeansConfig {
                    n_clusters: 5,
                    max_iter: 100,
                    init: KMeansInit::KMeansPlusPlus,
                    ..Default::default()
                };
                let kmeans = KMeans::new(config);
                let dummy_labels = scirs2_core::ndarray::Array1::zeros(data.nrows());

                let _ = kmeans.fit(black_box(data), &dummy_labels);
            });
        });
    }

    group.finish();
}

/// Benchmark K-Means with different initialization methods
fn bench_kmeans_initialization(c: &mut Criterion) {
    let mut group = c.benchmark_group("kmeans_initialization");
    let data = generate_clustered_data(1000, 10, 5);

    for init_method in [KMeansInit::Random, KMeansInit::KMeansPlusPlus] {
        group.bench_with_input(
            BenchmarkId::new("init_method", format!("{:?}", init_method)),
            &data,
            |b, data| {
                b.iter(|| {
                    let config = KMeansConfig {
                        n_clusters: 5,
                        max_iter: 100,
                        init: init_method,
                        ..Default::default()
                    };
                    let kmeans = KMeans::new(config);
                    let dummy_labels = scirs2_core::ndarray::Array1::zeros(data.nrows());

                    let _ = kmeans.fit(black_box(data), &dummy_labels);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark DBSCAN performance
fn bench_dbscan_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("dbscan_scaling");

    for n_samples in [100, 500, 1000, 2000] {
        let data = generate_clustered_data(n_samples, 10, 5);

        group.throughput(Throughput::Elements(n_samples as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n_samples), &data, |b, data| {
            b.iter(|| {
                let dbscan = DBSCAN::new().eps(1.0).min_samples(5);
                let dummy_labels = scirs2_core::ndarray::Array1::zeros(data.nrows());

                let _ = dbscan.fit(black_box(data), &dummy_labels);
            });
        });
    }

    group.finish();
}

/// Benchmark Hierarchical clustering
fn bench_hierarchical_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("hierarchical_scaling");

    // Note: hierarchical clustering is O(n²) or O(n³), so we use smaller datasets
    for n_samples in [50, 100, 200, 500] {
        let data = generate_clustered_data(n_samples, 10, 5);

        group.throughput(Throughput::Elements(n_samples as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n_samples), &data, |b, data| {
            b.iter(|| {
                let hierarchical = AgglomerativeClustering::new().n_clusters(5);
                let dummy_labels = scirs2_core::ndarray::Array1::zeros(data.nrows());

                let _ = hierarchical.fit(black_box(data), &dummy_labels);
            });
        });
    }

    group.finish();
}

/// Benchmark GMM performance
fn bench_gmm_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("gmm_scaling");

    for n_samples in [100, 500, 1000, 2000] {
        let data = generate_clustered_data(n_samples, 10, 5);
        let labels = vec![0; n_samples]; // Dummy labels for fit signature

        group.throughput(Throughput::Elements(n_samples as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n_samples), &data, |b, data| {
            b.iter(|| {
                let gmm = GaussianMixture::new()
                    .n_components(5)
                    .covariance_type(CovarianceType::Full)
                    .max_iter(50);
                let dummy_labels = scirs2_core::ndarray::Array1::zeros(data.nrows());

                let _ = gmm.fit(black_box(&data.view()), &dummy_labels.view());
            });
        });
    }

    group.finish();
}

/// Benchmark Spectral clustering
fn bench_spectral_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("spectral_scaling");

    for n_samples in [50, 100, 200, 500] {
        let data = generate_clustered_data(n_samples, 10, 5);

        group.throughput(Throughput::Elements(n_samples as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n_samples), &data, |b, data| {
            b.iter(|| {
                let spectral = SpectralClustering::new()
                    .n_clusters(5)
                    .affinity(Affinity::RBF { gamma: 1.0 });
                let dummy_labels = scirs2_core::ndarray::Array1::zeros(data.nrows());

                let _ = spectral.fit(black_box(data), &dummy_labels);
            });
        });
    }

    group.finish();
}

/// Benchmark feature dimensionality impact on K-Means
fn bench_kmeans_dimensionality(c: &mut Criterion) {
    let mut group = c.benchmark_group("kmeans_dimensionality");
    let n_samples = 1000;

    for n_features in [2, 5, 10, 20, 50, 100] {
        let data = generate_clustered_data(n_samples, n_features, 5);

        group.throughput(Throughput::Elements((n_samples * n_features) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n_features), &data, |b, data| {
            b.iter(|| {
                let config = KMeansConfig {
                    n_clusters: 5,
                    max_iter: 100,
                    init: KMeansInit::KMeansPlusPlus,
                    ..Default::default()
                };
                let kmeans = KMeans::new(config);
                let dummy_labels = scirs2_core::ndarray::Array1::zeros(data.nrows());

                let _ = kmeans.fit(black_box(data), &dummy_labels);
            });
        });
    }

    group.finish();
}

/// Benchmark cluster count impact on K-Means
fn bench_kmeans_cluster_count(c: &mut Criterion) {
    let mut group = c.benchmark_group("kmeans_cluster_count");
    let data = generate_clustered_data(1000, 10, 10);

    for n_clusters in [2, 5, 10, 20, 50] {
        group.bench_with_input(BenchmarkId::from_parameter(n_clusters), &data, |b, data| {
            b.iter(|| {
                let config = KMeansConfig {
                    n_clusters,
                    max_iter: 100,
                    init: KMeansInit::KMeansPlusPlus,
                    ..Default::default()
                };
                let kmeans = KMeans::new(config);
                let dummy_labels = scirs2_core::ndarray::Array1::zeros(data.nrows());

                let _ = kmeans.fit(black_box(data), &dummy_labels);
            });
        });
    }

    group.finish();
}

/// Benchmark Fuzzy C-Means performance
fn bench_fuzzy_cmeans(c: &mut Criterion) {
    let mut group = c.benchmark_group("fuzzy_cmeans");
    let data = generate_clustered_data(1000, 10, 5);

    for fuzziness in [1.5, 2.0, 2.5, 3.0] {
        group.bench_with_input(
            BenchmarkId::new("fuzziness", format!("{:.1}", fuzziness)),
            &data,
            |b, data| {
                b.iter(|| {
                    let fcm = FuzzyCMeans::new(5).fuzziness(fuzziness).max_iter(100);
                    let dummy_labels = scirs2_core::ndarray::Array1::zeros(data.nrows());

                    let _ = fcm.fit(black_box(data), &dummy_labels);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark algorithm comparison on same dataset
fn bench_algorithm_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("algorithm_comparison");
    let data = generate_clustered_data(500, 10, 5);

    // K-Means
    group.bench_function("kmeans", |b| {
        b.iter(|| {
            let config = KMeansConfig {
                n_clusters: 5,
                max_iter: 100,
                ..Default::default()
            };
            let kmeans = KMeans::new(config);
            let dummy_labels = scirs2_core::ndarray::Array1::zeros(data.nrows());
            let _ = kmeans.fit(black_box(&data), &dummy_labels);
        });
    });

    // DBSCAN
    group.bench_function("dbscan", |b| {
        b.iter(|| {
            let dbscan = DBSCAN::new().eps(1.0).min_samples(5);
            let dummy_labels = scirs2_core::ndarray::Array1::zeros(data.nrows());
            let _ = dbscan.fit(black_box(&data), &dummy_labels);
        });
    });

    // Hierarchical
    group.bench_function("hierarchical", |b| {
        b.iter(|| {
            let hierarchical = AgglomerativeClustering::new().n_clusters(5);
            let dummy_labels = scirs2_core::ndarray::Array1::zeros(data.nrows());
            let _ = hierarchical.fit(black_box(&data), &dummy_labels);
        });
    });

    // GMM
    group.bench_function("gmm", |b| {
        b.iter(|| {
            let gmm = GaussianMixture::new().n_components(5).max_iter(50);
            let dummy_labels = scirs2_core::ndarray::Array1::zeros(data.nrows());
            let _ = gmm.fit(black_box(&data.view()), &dummy_labels.view());
        });
    });

    // Fuzzy C-Means
    group.bench_function("fuzzy_cmeans", |b| {
        b.iter(|| {
            let fcm = FuzzyCMeans::new(5).max_iter(100);
            let dummy_labels = scirs2_core::ndarray::Array1::zeros(data.nrows());
            let _ = fcm.fit(black_box(&data), &dummy_labels);
        });
    });

    group.finish();
}

/// Benchmark memory efficiency with Mini-batch K-Means
fn bench_minibatch_kmeans(c: &mut Criterion) {
    let mut group = c.benchmark_group("minibatch_kmeans");

    for batch_size in [100, 256, 512, 1000] {
        let data = generate_clustered_data(10000, 10, 5);

        group.throughput(Throughput::Elements(10000));
        group.bench_with_input(
            BenchmarkId::new("batch_size", batch_size),
            &data,
            |b, data| {
                b.iter(|| {
                    let config = MiniBatchKMeansConfig {
                        n_clusters: 5,
                        batch_size,
                        max_iter: 100,
                        random_seed: None,
                    };
                    let mb_kmeans = MiniBatchKMeans::new(config);
                    let dummy_labels = scirs2_core::ndarray::Array1::zeros(data.nrows());

                    let _ = mb_kmeans.fit(black_box(data), &dummy_labels);
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_kmeans_scaling,
    bench_kmeans_initialization,
    bench_dbscan_scaling,
    bench_hierarchical_scaling,
    bench_gmm_scaling,
    bench_spectral_scaling,
    bench_kmeans_dimensionality,
    bench_kmeans_cluster_count,
    bench_fuzzy_cmeans,
    bench_algorithm_comparison,
    bench_minibatch_kmeans,
);

criterion_main!(benches);
