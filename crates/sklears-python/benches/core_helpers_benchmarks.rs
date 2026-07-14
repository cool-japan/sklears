//! Benchmarks for the PyO3-free `*_core` helpers behind sklears-python's
//! clustering and model-selection bindings.
//!
//! `sklears-python` builds with pyo3's `extension-module` feature enabled
//! for real wheel builds (required so the compiled `cdylib` can be
//! imported from Python), which means `Python::with_gil` -- and therefore
//! any `#[pymethods]` -- cannot be exercised from a standalone `cargo
//! bench` binary (there is no live interpreter attached). This is exactly
//! why `PyKMeans`/`PyDBSCAN`/`PyKFold`/`train_test_split` had their logic
//! split into private `*_core` helpers operating on plain
//! `ndarray`/`Vec`/`f64` types, with no PyO3 dependency, mirroring the
//! `#[cfg(test)]` coverage already in `src/clustering.rs` and
//! `src/model_selection.rs`. Those helpers (plus the `new` constructors
//! needed to obtain an instance to call them on) were made `pub` so this
//! file -- which compiles as a separate crate -- can reach them; see the
//! doc comments at each call site in `src/` for details.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::essentials::Normal;
use scirs2_core::random::{seeded_rng, Distribution};
use sklears_python::{train_test_split_core, PyDBSCAN, PyKFold, PyKMeans};
use std::hint::black_box;

/// Generate `n_clusters` well-separated Gaussian blobs (`n_samples` points
/// total, `n_features` dimensions each), scaling up the toy `two_blob_data`
/// fixture used by `src/clustering.rs`'s own `#[cfg(test)]` module to
/// arbitrary benchmark sizes. Cluster centers are spaced `20.0` apart so
/// unit-variance noise never causes two clusters to overlap, regardless of
/// size.
fn generate_blob_data(
    n_samples: usize,
    n_features: usize,
    n_clusters: usize,
    seed: u64,
) -> Array2<f64> {
    let mut rng = seeded_rng(seed);
    let normal = Normal::new(0.0, 1.0).expect("operation should succeed");
    let mut data = Array2::<f64>::zeros((n_samples, n_features));

    let samples_per_cluster = n_samples / n_clusters;
    for cluster_id in 0..n_clusters {
        let start = cluster_id * samples_per_cluster;
        let end = if cluster_id == n_clusters - 1 {
            n_samples
        } else {
            (cluster_id + 1) * samples_per_cluster
        };
        let center = cluster_id as f64 * 20.0;

        for i in start..end {
            for j in 0..n_features {
                data[[i, j]] = center + normal.sample(&mut rng);
            }
        }
    }

    data
}

/// Generate a plain `(features, targets)` dataset, matching the shape and
/// intent of `make_indexed_dataset` in `src/model_selection.rs`'s tests --
/// only the shape matters for these benchmarks, not the values.
fn generate_indexed_dataset(
    n_samples: usize,
    n_features: usize,
    seed: u64,
) -> (Array2<f64>, Array1<f64>) {
    let mut rng = seeded_rng(seed);
    let normal = Normal::new(0.0, 1.0).expect("operation should succeed");
    let x_data: Vec<f64> = (0..n_samples * n_features)
        .map(|_| normal.sample(&mut rng))
        .collect();
    let x =
        Array2::from_shape_vec((n_samples, n_features), x_data).expect("shape matches data length");
    let y = Array1::from_vec((0..n_samples).map(|i| i as f64).collect());
    (x, y)
}

/// Benchmark `PyKMeans::fit_core` followed by `predict_core` -- the core
/// logic behind the Python `KMeans.fit`/`.predict` methods -- across
/// dataset sizes.
fn benchmark_kmeans_fit_predict(c: &mut Criterion) {
    let mut group = c.benchmark_group("kmeans_fit_predict");
    group.sample_size(10); // K-means iterates to convergence; keep this affordable.

    for size in [100, 1000, 5000].iter() {
        let x = generate_blob_data(*size, 4, 4, 42);

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                let mut model = PyKMeans::new(4, 300, 1e-4, Some(42));
                model.fit_core(black_box(&x)).expect("fit should succeed");
                black_box(model.predict_core(&x).expect("predict should succeed"))
            });
        });
    }

    group.finish();
}

/// Benchmark `PyDBSCAN::fit_predict_core` -- the core logic behind the
/// Python `DBSCAN.fit_predict` method -- across dataset sizes.
fn benchmark_dbscan_fit_predict(c: &mut Criterion) {
    let mut group = c.benchmark_group("dbscan_fit_predict");
    group.sample_size(10); // Neighbor search is quadratic in the sample count.

    for size in [100, 1000, 5000].iter() {
        let x = generate_blob_data(*size, 4, 4, 7);

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                let mut model = PyDBSCAN::new(3.0, 5);
                black_box(
                    model
                        .fit_predict_core(black_box(&x))
                        .expect("fit_predict should succeed"),
                )
            });
        });
    }

    group.finish();
}

/// Benchmark `train_test_split_core` -- the core logic behind the Python
/// `train_test_split` function -- across dataset sizes.
fn benchmark_train_test_split(c: &mut Criterion) {
    let mut group = c.benchmark_group("train_test_split");

    for size in [100, 1000, 5000].iter() {
        let (x, y) = generate_indexed_dataset(*size, 10, 99);

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                black_box(
                    train_test_split_core(black_box(&x), &y, Some(0.25), Some(42))
                        .expect("split should succeed"),
                )
            });
        });
    }

    group.finish();
}

/// Benchmark `PyKFold::split_core` -- the core logic behind the Python
/// `KFold.split` method -- across dataset sizes.
fn benchmark_kfold_split(c: &mut Criterion) {
    let mut group = c.benchmark_group("kfold_split");

    for size in [100, 1000, 5000].iter() {
        let kfold = PyKFold::new(5, false, None).expect("n_splits=5 is valid");

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                black_box(
                    kfold
                        .split_core(black_box(*size))
                        .expect("split should succeed"),
                )
            });
        });
    }

    group.finish();
}

criterion_group!(
    core_helpers,
    benchmark_kmeans_fit_predict,
    benchmark_dbscan_fit_predict,
    benchmark_train_test_split,
    benchmark_kfold_split,
);
criterion_main!(core_helpers);
