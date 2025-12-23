//! Comprehensive benchmarks for sklears-covariance estimators
//!
//! This benchmark suite measures the performance of various covariance
//! estimators across different data sizes and configurations.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use scirs2_core::ndarray::Array2;
use scirs2_core::random::essentials::Normal;
use scirs2_core::random::Distribution;
use sklears_core::traits::Fit;
use sklears_covariance::{
    AdaptiveLassoCovariance, ChenSteinCovariance, ElasticNetCovariance, EmpiricalCovariance,
    GraphicalLasso, HuberCovariance, LedoitWolf, MinCovDet, RaoBlackwellLedoitWolf,
    RidgeCovariance, ShrunkCovariance, OAS,
};

/// Generate random test data
fn generate_test_data(n_samples: usize, n_features: usize, seed: u64) -> Array2<f64> {
    use scirs2_core::random::SeedableRng;
    let mut rng = scirs2_core::random::StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0, 1.0).unwrap();

    Array2::from_shape_fn((n_samples, n_features), |_| normal.sample(&mut rng))
}

/// Benchmark EmpiricalCovariance estimator
fn bench_empirical_covariance(c: &mut Criterion) {
    let mut group = c.benchmark_group("EmpiricalCovariance");

    for size in [50, 100, 200, 500].iter() {
        let n_samples = size * 2;
        let n_features = *size;
        let data = generate_test_data(n_samples, n_features, 42);

        group.throughput(Throughput::Elements((n_samples * n_features) as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}x{}", n_samples, n_features)),
            &data,
            |b, data| {
                b.iter(|| {
                    let estimator = EmpiricalCovariance::new();
                    let _ = estimator.fit(&data.view(), &());
                });
            },
        );
    }

    group.finish();
}

/// Benchmark LedoitWolf estimator
fn bench_ledoit_wolf(c: &mut Criterion) {
    let mut group = c.benchmark_group("LedoitWolf");

    for size in [50, 100, 200, 500].iter() {
        let n_samples = size * 2;
        let n_features = *size;
        let data = generate_test_data(n_samples, n_features, 42);

        group.throughput(Throughput::Elements((n_samples * n_features) as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}x{}", n_samples, n_features)),
            &data,
            |b, data| {
                b.iter(|| {
                    let estimator = LedoitWolf::new();
                    let _ = estimator.fit(&data.view(), &());
                });
            },
        );
    }

    group.finish();
}

/// Benchmark OAS estimator
fn bench_oas(c: &mut Criterion) {
    let mut group = c.benchmark_group("OAS");

    for size in [50, 100, 200, 500].iter() {
        let n_samples = size * 2;
        let n_features = *size;
        let data = generate_test_data(n_samples, n_features, 42);

        group.throughput(Throughput::Elements((n_samples * n_features) as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}x{}", n_samples, n_features)),
            &data,
            |b, data| {
                b.iter(|| {
                    let estimator = OAS::new();
                    let _ = estimator.fit(&data.view(), &());
                });
            },
        );
    }

    group.finish();
}

/// Benchmark RidgeCovariance estimator
fn bench_ridge_covariance(c: &mut Criterion) {
    let mut group = c.benchmark_group("RidgeCovariance");

    for size in [50, 100, 200, 500].iter() {
        let n_samples = size * 2;
        let n_features = *size;
        let data = generate_test_data(n_samples, n_features, 42);

        group.throughput(Throughput::Elements((n_samples * n_features) as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}x{}", n_samples, n_features)),
            &data,
            |b, data| {
                b.iter(|| {
                    let estimator = RidgeCovariance::new().alpha(0.1);
                    let _ = estimator.fit(&data.view(), &());
                });
            },
        );
    }

    group.finish();
}

/// Benchmark GraphicalLasso estimator (smaller sizes due to complexity)
fn bench_graphical_lasso(c: &mut Criterion) {
    let mut group = c.benchmark_group("GraphicalLasso");
    group.sample_size(10); // Reduce sample size for slower estimators

    for size in [10, 20, 50].iter() {
        let n_samples = size * 3;
        let n_features = *size;
        let data = generate_test_data(n_samples, n_features, 42);

        group.throughput(Throughput::Elements((n_samples * n_features) as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}x{}", n_samples, n_features)),
            &data,
            |b, data| {
                b.iter(|| {
                    let estimator = GraphicalLasso::new().alpha(0.1).max_iter(50);
                    let _ = estimator.fit(&data.view(), &());
                });
            },
        );
    }

    group.finish();
}

/// Benchmark MinCovDet estimator (smaller sizes due to complexity)
fn bench_min_cov_det(c: &mut Criterion) {
    let mut group = c.benchmark_group("MinCovDet");
    group.sample_size(10); // Reduce sample size for slower estimators

    for size in [10, 20, 50].iter() {
        let n_samples = size * 3;
        let n_features = *size;
        let data = generate_test_data(n_samples, n_features, 42);

        group.throughput(Throughput::Elements((n_samples * n_features) as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}x{}", n_samples, n_features)),
            &data,
            |b, data| {
                b.iter(|| {
                    let estimator = MinCovDet::new().support_fraction(0.8);
                    let _ = estimator.fit(&data.view(), &());
                });
            },
        );
    }

    group.finish();
}

/// Benchmark comparison across estimators for fixed size
fn bench_estimator_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("EstimatorComparison");
    let n_samples = 200;
    let n_features = 100;
    let data = generate_test_data(n_samples, n_features, 42);

    group.throughput(Throughput::Elements((n_samples * n_features) as u64));

    group.bench_function("EmpiricalCovariance", |b| {
        b.iter(|| {
            let estimator = EmpiricalCovariance::new();
            black_box(estimator.fit(&data.view(), &()))
        });
    });

    group.bench_function("LedoitWolf", |b| {
        b.iter(|| {
            let estimator = LedoitWolf::new();
            black_box(estimator.fit(&data.view(), &()))
        });
    });

    group.bench_function("OAS", |b| {
        b.iter(|| {
            let estimator = OAS::new();
            black_box(estimator.fit(&data.view(), &()))
        });
    });

    group.bench_function("RidgeCovariance", |b| {
        b.iter(|| {
            let estimator = RidgeCovariance::new().alpha(0.1);
            black_box(estimator.fit(&data.view(), &()))
        });
    });

    group.finish();
}

/// Benchmark ShrunkCovariance estimator
fn bench_shrunk_covariance(c: &mut Criterion) {
    let mut group = c.benchmark_group("ShrunkCovariance");

    for size in [50, 100, 200, 500].iter() {
        let n_samples = size * 2;
        let n_features = *size;
        let data = generate_test_data(n_samples, n_features, 42);

        group.throughput(Throughput::Elements((n_samples * n_features) as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}x{}", n_samples, n_features)),
            &data,
            |b, data| {
                b.iter(|| {
                    let estimator = ShrunkCovariance::new().shrinkage(0.1);
                    let _ = estimator.fit(&data.view(), &());
                });
            },
        );
    }

    group.finish();
}

/// Benchmark RaoBlackwellLedoitWolf estimator
fn bench_rao_blackwell_lw(c: &mut Criterion) {
    let mut group = c.benchmark_group("RaoBlackwellLedoitWolf");

    for size in [50, 100, 200, 500].iter() {
        let n_samples = size * 2;
        let n_features = *size;
        let data = generate_test_data(n_samples, n_features, 42);

        group.throughput(Throughput::Elements((n_samples * n_features) as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}x{}", n_samples, n_features)),
            &data,
            |b, data| {
                b.iter(|| {
                    let estimator = RaoBlackwellLedoitWolf::new();
                    let _ = estimator.fit(&data.view(), &());
                });
            },
        );
    }

    group.finish();
}

/// Benchmark ChenSteinCovariance estimator
fn bench_chen_stein(c: &mut Criterion) {
    let mut group = c.benchmark_group("ChenSteinCovariance");

    for size in [50, 100, 200, 500].iter() {
        let n_samples = size * 2;
        let n_features = *size;
        let data = generate_test_data(n_samples, n_features, 42);

        group.throughput(Throughput::Elements((n_samples * n_features) as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}x{}", n_samples, n_features)),
            &data,
            |b, data| {
                b.iter(|| {
                    let estimator = ChenSteinCovariance::new();
                    let _ = estimator.fit(&data.view(), &());
                });
            },
        );
    }

    group.finish();
}

/// Benchmark HuberCovariance estimator
fn bench_huber_covariance(c: &mut Criterion) {
    let mut group = c.benchmark_group("HuberCovariance");
    group.sample_size(10); // Reduce sample size for iterative estimators

    for size in [20, 50, 100].iter() {
        let n_samples = size * 3;
        let n_features = *size;
        let data = generate_test_data(n_samples, n_features, 42);

        group.throughput(Throughput::Elements((n_samples * n_features) as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}x{}", n_samples, n_features)),
            &data,
            |b, data| {
                b.iter(|| {
                    let estimator = HuberCovariance::new().max_iter(50);
                    let _ = estimator.fit(&data.view(), &());
                });
            },
        );
    }

    group.finish();
}

/// Benchmark ElasticNetCovariance estimator
fn bench_elastic_net(c: &mut Criterion) {
    let mut group = c.benchmark_group("ElasticNetCovariance");
    group.sample_size(10); // Reduce sample size for iterative estimators

    for size in [20, 50, 100].iter() {
        let n_samples = size * 3;
        let n_features = *size;
        let data = generate_test_data(n_samples, n_features, 42);

        group.throughput(Throughput::Elements((n_samples * n_features) as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}x{}", n_samples, n_features)),
            &data,
            |b, data| {
                b.iter(|| {
                    let estimator = ElasticNetCovariance::new()
                        .alpha(0.1)
                        .l1_ratio(0.5)
                        .max_iter(50);
                    let _ = estimator.fit(&data.view(), &());
                });
            },
        );
    }

    group.finish();
}

/// Benchmark AdaptiveLassoCovariance estimator
fn bench_adaptive_lasso(c: &mut Criterion) {
    let mut group = c.benchmark_group("AdaptiveLassoCovariance");
    group.sample_size(10); // Reduce sample size for iterative estimators

    for size in [20, 50, 100].iter() {
        let n_samples = size * 3;
        let n_features = *size;
        let data = generate_test_data(n_samples, n_features, 42);

        group.throughput(Throughput::Elements((n_samples * n_features) as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}x{}", n_samples, n_features)),
            &data,
            |b, data| {
                b.iter(|| {
                    let estimator = AdaptiveLassoCovariance::new().alpha(0.1).max_iter(50);
                    let _ = estimator.fit(&data.view(), &());
                });
            },
        );
    }

    group.finish();
}

/// Benchmark shrinkage estimators comparison
fn bench_shrinkage_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("ShrinkageEstimatorComparison");
    let n_samples = 200;
    let n_features = 100;
    let data = generate_test_data(n_samples, n_features, 42);

    group.throughput(Throughput::Elements((n_samples * n_features) as u64));

    group.bench_function("LedoitWolf", |b| {
        b.iter(|| {
            let estimator = LedoitWolf::new();
            black_box(estimator.fit(&data.view(), &()))
        });
    });

    group.bench_function("OAS", |b| {
        b.iter(|| {
            let estimator = OAS::new();
            black_box(estimator.fit(&data.view(), &()))
        });
    });

    group.bench_function("RaoBlackwellLedoitWolf", |b| {
        b.iter(|| {
            let estimator = RaoBlackwellLedoitWolf::new();
            black_box(estimator.fit(&data.view(), &()))
        });
    });

    group.bench_function("ChenStein", |b| {
        b.iter(|| {
            let estimator = ChenSteinCovariance::new();
            black_box(estimator.fit(&data.view(), &()))
        });
    });

    group.finish();
}

/// Benchmark regularized estimators comparison
fn bench_regularized_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("RegularizedEstimatorComparison");
    let n_samples = 100;
    let n_features = 50;
    let data = generate_test_data(n_samples, n_features, 42);

    group.throughput(Throughput::Elements((n_samples * n_features) as u64));
    group.sample_size(10);

    group.bench_function("Ridge", |b| {
        b.iter(|| {
            let estimator = RidgeCovariance::new().alpha(0.1);
            black_box(estimator.fit(&data.view(), &()))
        });
    });

    group.bench_function("ElasticNet", |b| {
        b.iter(|| {
            let estimator = ElasticNetCovariance::new()
                .alpha(0.1)
                .l1_ratio(0.5)
                .max_iter(50);
            black_box(estimator.fit(&data.view(), &()))
        });
    });

    group.bench_function("AdaptiveLasso", |b| {
        b.iter(|| {
            let estimator = AdaptiveLassoCovariance::new().alpha(0.1).max_iter(50);
            black_box(estimator.fit(&data.view(), &()))
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_empirical_covariance,
    bench_ledoit_wolf,
    bench_oas,
    bench_ridge_covariance,
    bench_graphical_lasso,
    bench_min_cov_det,
    bench_estimator_comparison,
    bench_shrunk_covariance,
    bench_rao_blackwell_lw,
    bench_chen_stein,
    bench_huber_covariance,
    bench_elastic_net,
    bench_adaptive_lasso,
    bench_shrinkage_comparison,
    bench_regularized_comparison,
);

criterion_main!(benches);
