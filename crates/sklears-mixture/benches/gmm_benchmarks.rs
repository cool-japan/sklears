//! Performance benchmarks for sklears-mixture
//!
//! These benchmarks establish performance baselines and track regression
//! across releases using the Criterion framework.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::essentials::Normal;
use scirs2_core::random::{seeded_rng, Distribution};
use sklears_core::traits::{Fit, Predict};
use sklears_mixture::{CovarianceType, GaussianMixture};
use std::hint::black_box;

/// Number of mixture components used to fit/evaluate in every benchmark below.
const N_CLUSTERS: usize = 3;

/// Generate `N_CLUSTERS` well-separated Gaussian blobs (deterministic, seeded).
fn generate_blob_data(n_samples: usize, n_features: usize, seed: u64) -> Array2<f64> {
    let mut rng = seeded_rng(seed);
    let normal = Normal::new(0.0, 1.0).expect("operation should succeed");

    let mut data = Array2::<f64>::zeros((n_samples, n_features));
    for i in 0..n_samples {
        let cluster = i % N_CLUSTERS;
        for j in 0..n_features {
            let offset = if j == 0 { cluster as f64 * 12.0 } else { 0.0 };
            data[[i, j]] = normal.sample(&mut rng) + offset;
        }
    }
    data
}

/// Benchmark GaussianMixture::fit (EM algorithm, diagonal covariance, bounded iterations).
fn benchmark_gmm_fit(c: &mut Criterion) {
    let mut group = c.benchmark_group("gmm_fit");
    group.sample_size(10);

    for size in [100, 1000, 5000, 10000].iter() {
        let x = generate_blob_data(*size, 4, 321);

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                let gmm = GaussianMixture::new()
                    .n_components(N_CLUSTERS)
                    .covariance_type(CovarianceType::Diagonal)
                    .max_iter(30);
                black_box(gmm.fit(&x.view(), &()))
            });
        });
    }

    group.finish();
}

/// Benchmark GaussianMixture::predict (hard cluster assignment) on a model
/// that is fitted once outside the timed loop.
fn benchmark_gmm_predict(c: &mut Criterion) {
    let mut group = c.benchmark_group("gmm_predict");
    group.sample_size(10);

    let x_train = generate_blob_data(2000, 4, 111);
    let gmm = GaussianMixture::new()
        .n_components(N_CLUSTERS)
        .covariance_type(CovarianceType::Diagonal)
        .max_iter(30);
    let fitted = gmm
        .fit(&x_train.view(), &())
        .expect("GMM fitting should succeed with valid data");

    for size in [100, 1000, 5000, 10000].iter() {
        let x_test = generate_blob_data(*size, 4, 654);

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                if let Ok(labels) = fitted.predict(&x_test.view()) {
                    black_box(labels)
                } else {
                    black_box(Array1::<i32>::zeros(x_test.nrows()))
                }
            });
        });
    }

    group.finish();
}

/// Benchmark GaussianMixture::score (total log-likelihood) on a model that is
/// fitted once outside the timed loop.
fn benchmark_gmm_score(c: &mut Criterion) {
    let mut group = c.benchmark_group("gmm_score");
    group.sample_size(10);

    let x_train = generate_blob_data(2000, 4, 222);
    let gmm = GaussianMixture::new()
        .n_components(N_CLUSTERS)
        .covariance_type(CovarianceType::Diagonal)
        .max_iter(30);
    let fitted = gmm
        .fit(&x_train.view(), &())
        .expect("GMM fitting should succeed with valid data");

    for size in [100, 1000, 5000, 10000].iter() {
        let x_test = generate_blob_data(*size, 4, 987);

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                if let Ok(score) = fitted.score(&x_test.view()) {
                    black_box(score)
                } else {
                    black_box(0.0_f64)
                }
            });
        });
    }

    group.finish();
}

criterion_group!(fitting, benchmark_gmm_fit,);
criterion_group!(prediction, benchmark_gmm_predict, benchmark_gmm_score,);

criterion_main!(fitting, prediction,);
