//! Performance benchmarks for sklears-inspection
//!
//! These benchmarks measure the scaling behavior of this crate's core
//! model-inspection routines, `permutation_importance` and
//! `partial_dependence`, across the axes that drive their cost: number of
//! samples, number of repeats, number of features, and (for partial
//! dependence) grid resolution.
//!
//! Both functions are model-agnostic: they accept a `predict_fn` closure
//! standing in for an already-fitted model's prediction function. These
//! benchmarks therefore use a trivial, deterministic additive "model"
//! (row-wise sum of features) as the stand-in fitted model — the exact
//! pattern already used throughout this crate's own doctests and unit
//! tests — so no dependency on another sklears crate is required to obtain
//! a "real" fitted model.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use scirs2_core::ndarray::{Array1, Array2, ArrayView2};
use scirs2_core::random::essentials::Normal;
use scirs2_core::random::{seeded_rng, Distribution};
use sklears_inspection::{
    partial_dependence, permutation_importance, PartialDependenceKind, PartialDependenceResult,
    PermutationImportanceResult, ScoreFunction,
};
use std::hint::black_box;

/// Generate deterministic pseudo-random test data for benchmarks.
fn generate_data(nrows: usize, ncols: usize, seed: u64) -> Array2<f64> {
    let mut rng = seeded_rng(seed);
    let normal = Normal::new(0.0, 1.0).expect("normal distribution parameters should be valid");

    let data: Vec<f64> = (0..nrows * ncols)
        .map(|_| normal.sample(&mut rng))
        .collect();

    Array2::from_shape_vec((nrows, ncols), data).expect("shape and data length should match")
}

/// Deterministic stand-in for an already-fitted model's prediction function.
///
/// Mirrors the pattern used throughout this crate's own doctests and unit
/// tests: a trivial additive model whose prediction is the row-wise sum of
/// features. Both `permutation_importance` and `partial_dependence` are
/// free functions written against exactly this kind of closure, so this is
/// a faithful, zero-extra-dependency stand-in for "a simple fitted model".
fn predict_fn(x: &ArrayView2<f64>) -> Vec<f64> {
    x.rows().into_iter().map(|row| row.iter().sum()).collect()
}

/// Benchmark `permutation_importance` scaling with the number of samples.
///
/// Cost scales roughly linearly with `n_samples`: each of the
/// `n_features * n_repeats` permutation rounds re-runs `predict_fn` (and the
/// scoring function) over every sample.
fn benchmark_permutation_importance_by_samples(c: &mut Criterion) {
    let mut group = c.benchmark_group("permutation_importance_by_samples");
    group.sample_size(10);

    let n_features = 10;
    let n_repeats = 5;

    for n_samples in [100, 1000, 5000, 10000].iter() {
        let x = generate_data(*n_samples, n_features, 4001);
        let y = Array1::from_vec(predict_fn(&x.view()));

        group.throughput(Throughput::Elements(*n_samples as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n_samples), n_samples, |b, _| {
            b.iter(|| {
                if let Ok(result) = permutation_importance(
                    &predict_fn,
                    &x.view(),
                    &y.view(),
                    ScoreFunction::R2,
                    n_repeats,
                    Some(7),
                ) {
                    black_box(result)
                } else {
                    black_box(PermutationImportanceResult {
                        importances: vec![],
                        importances_mean: Array1::zeros(0),
                        importances_std: Array1::zeros(0),
                    })
                }
            });
        });
    }

    group.finish();
}

/// Benchmark `permutation_importance` scaling with `n_repeats`.
///
/// Mirrors the precedent's `benchmark_feature_dimensions` pattern of
/// sweeping a second axis at a fixed data size: cost is expected to scale
/// linearly with the number of repeats, since each repeat re-shuffles every
/// feature and re-scores the model from scratch.
fn benchmark_permutation_importance_by_repeats(c: &mut Criterion) {
    let mut group = c.benchmark_group("permutation_importance_by_repeats");
    group.sample_size(10);

    let n_samples = 1000;
    let n_features = 10;
    let x = generate_data(n_samples, n_features, 4002);
    let y = Array1::from_vec(predict_fn(&x.view()));

    for n_repeats in [5, 10, 20, 50].iter() {
        group.throughput(Throughput::Elements(*n_repeats as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n_repeats), n_repeats, |b, _| {
            b.iter(|| {
                if let Ok(result) = permutation_importance(
                    &predict_fn,
                    &x.view(),
                    &y.view(),
                    ScoreFunction::R2,
                    *n_repeats,
                    Some(7),
                ) {
                    black_box(result)
                } else {
                    black_box(PermutationImportanceResult {
                        importances: vec![],
                        importances_mean: Array1::zeros(0),
                        importances_std: Array1::zeros(0),
                    })
                }
            });
        });
    }

    group.finish();
}

/// Benchmark `permutation_importance` scaling with the number of features.
///
/// Importance cost is roughly linear in `n_features` since each feature is
/// permuted, and the model re-scored, independently of the others.
fn benchmark_permutation_importance_by_features(c: &mut Criterion) {
    let mut group = c.benchmark_group("permutation_importance_by_features");
    group.sample_size(10);

    let n_samples = 1000;
    let n_repeats = 5;

    for n_features in [5, 10, 50, 100].iter() {
        let x = generate_data(n_samples, *n_features, 4003);
        let y = Array1::from_vec(predict_fn(&x.view()));

        group.throughput(Throughput::Elements((n_samples * n_features) as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(n_features),
            n_features,
            |b, _| {
                b.iter(|| {
                    if let Ok(result) = permutation_importance(
                        &predict_fn,
                        &x.view(),
                        &y.view(),
                        ScoreFunction::R2,
                        n_repeats,
                        Some(7),
                    ) {
                        black_box(result)
                    } else {
                        black_box(PermutationImportanceResult {
                            importances: vec![],
                            importances_mean: Array1::zeros(0),
                            importances_std: Array1::zeros(0),
                        })
                    }
                });
            },
        );
    }

    group.finish();
}

/// Benchmark `partial_dependence` scaling with grid resolution.
///
/// Cost scales linearly with the number of grid points: each grid point
/// requires one full `predict_fn` pass over all `n_samples` rows.
fn benchmark_partial_dependence_by_grid_size(c: &mut Criterion) {
    let mut group = c.benchmark_group("partial_dependence_by_grid_size");
    group.sample_size(10);

    let n_samples = 1000;
    let n_features = 10;
    let x = generate_data(n_samples, n_features, 4004);
    let features = vec![0usize];

    for grid_size in [10, 50, 100, 500].iter() {
        let grid = vec![(0..*grid_size).map(|i| i as f64).collect::<Vec<f64>>()];

        group.throughput(Throughput::Elements(*grid_size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(grid_size), grid_size, |b, _| {
            b.iter(|| {
                if let Ok(result) = partial_dependence(
                    &predict_fn,
                    &x.view(),
                    &features,
                    &grid,
                    PartialDependenceKind::Average,
                ) {
                    black_box(result)
                } else {
                    black_box(PartialDependenceResult {
                        values: vec![],
                        individual_values: vec![],
                        grid: vec![],
                    })
                }
            });
        });
    }

    group.finish();
}

criterion_group!(
    permutation_importance_benches,
    benchmark_permutation_importance_by_samples,
    benchmark_permutation_importance_by_repeats,
    benchmark_permutation_importance_by_features,
);

criterion_group!(
    partial_dependence_benches,
    benchmark_partial_dependence_by_grid_size,
);

criterion_main!(permutation_importance_benches, partial_dependence_benches);
