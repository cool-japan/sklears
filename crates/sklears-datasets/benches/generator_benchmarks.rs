//! Performance benchmarks for sklears-datasets synthetic data generators
//!
//! These benchmarks establish performance baselines and track regression
//! across releases for the crate's core synthetic dataset generators, using
//! the Criterion framework. Unlike a fit/transform benchmark, the generator
//! call itself is the operation under test, so it is invoked directly inside
//! the timed loop rather than being run once as setup.

#![allow(non_snake_case)]

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use scirs2_core::ndarray::{Array1, Array2};
use sklears_datasets::{make_blobs, make_classification, make_moons, make_regression};
use std::hint::black_box;

/// Benchmark `make_blobs` across a range of requested sample counts
fn benchmark_make_blobs(c: &mut Criterion) {
    let mut group = c.benchmark_group("make_blobs");

    for size in [100, 1000, 5000, 10000].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                if let Ok((x, y)) = make_blobs(*size, 10, 3, 1.0, Some(42)) {
                    black_box((x, y))
                } else {
                    black_box((Array2::<f64>::zeros((0, 0)), Array1::<i32>::zeros(0)))
                }
            });
        });
    }

    group.finish();
}

/// Benchmark `make_classification` across a range of requested sample counts
fn benchmark_make_classification(c: &mut Criterion) {
    let mut group = c.benchmark_group("make_classification");

    for size in [100, 1000, 5000, 10000].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                if let Ok((x, y)) = make_classification(*size, 20, 15, 5, 3, Some(42)) {
                    black_box((x, y))
                } else {
                    black_box((Array2::<f64>::zeros((0, 0)), Array1::<i32>::zeros(0)))
                }
            });
        });
    }

    group.finish();
}

/// Benchmark `make_regression` across a range of requested sample counts
fn benchmark_make_regression(c: &mut Criterion) {
    let mut group = c.benchmark_group("make_regression");

    for size in [100, 1000, 5000, 10000].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                if let Ok((x, y)) = make_regression(*size, 20, 10, 0.1, Some(42)) {
                    black_box((x, y))
                } else {
                    black_box((Array2::<f64>::zeros((0, 0)), Array1::<f64>::zeros(0)))
                }
            });
        });
    }

    group.finish();
}

/// Benchmark `make_moons` (non-linear, fixed-feature-count shape generator)
/// across a range of requested sample counts
fn benchmark_make_moons(c: &mut Criterion) {
    let mut group = c.benchmark_group("make_moons");

    for size in [100, 1000, 5000, 10000].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                if let Ok((x, y)) = make_moons(*size, Some(0.1), Some(42)) {
                    black_box((x, y))
                } else {
                    black_box((Array2::<f64>::zeros((0, 0)), Array1::<i32>::zeros(0)))
                }
            });
        });
    }

    group.finish();
}

criterion_group!(
    classification_generators,
    benchmark_make_blobs,
    benchmark_make_classification,
);

criterion_group!(
    regression_and_shape_generators,
    benchmark_make_regression,
    benchmark_make_moons,
);

criterion_main!(classification_generators, regression_and_shape_generators);
