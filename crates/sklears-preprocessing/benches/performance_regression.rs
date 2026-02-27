//! Performance regression benchmarks for sklears-preprocessing
//!
//! These benchmarks establish performance baselines and track regression
//! across releases using the Criterion framework.

#![allow(non_snake_case)]

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use scirs2_core::ndarray::Array2;
use scirs2_core::random::essentials::Normal;
use scirs2_core::random::{seeded_rng, Distribution};
use sklears_core::traits::{Fit, Transform};
use sklears_preprocessing::*;
use std::hint::black_box;

/// Generate test data for benchmarks
fn generate_data(nrows: usize, ncols: usize, seed: u64) -> Array2<f64> {
    let mut rng = seeded_rng(seed);
    let normal = Normal::new(0.0, 1.0).unwrap();

    let data: Vec<f64> = (0..nrows * ncols)
        .map(|_| normal.sample(&mut rng))
        .collect();

    Array2::from_shape_vec((nrows, ncols), data).unwrap()
}

/// Benchmark PolynomialFeatures (computational intensive)
fn benchmark_polynomial_features(c: &mut Criterion) {
    let mut group = c.benchmark_group("polynomial_features");
    group.sample_size(10); // Reduce sample size for slow operations

    for size in [100, 1000, 5000].iter() {
        let x = generate_data(*size, 5, 321);

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}x5", size)),
            size,
            |b, _| {
                b.iter(|| {
                    let poly = PolynomialFeatures::new();
                    if let Ok(fitted) = poly.fit(&x, &()) {
                        black_box(fitted.transform(&x))
                    } else {
                        Ok(x.clone())
                    }
                });
            },
        );
    }

    group.finish();
}

/// Benchmark QuantileTransformer
fn benchmark_quantile_transformer(c: &mut Criterion) {
    let mut group = c.benchmark_group("quantile_transformer");
    group.sample_size(10);

    for size in [100, 1000, 10000].iter() {
        let x = generate_data(*size, 10, 987);

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                let transformer = QuantileTransformer::new();
                if let Ok(fitted) = transformer.fit(&x, &()) {
                    black_box(fitted.transform(&x))
                } else {
                    Ok(x.clone())
                }
            });
        });
    }

    group.finish();
}

/// Benchmark PowerTransformer
fn benchmark_power_transformer(c: &mut Criterion) {
    let mut group = c.benchmark_group("power_transformer");
    group.sample_size(10);

    for size in [100, 1000, 10000].iter() {
        // PowerTransformer with YeoJohnson works with any sign
        let x = generate_data(*size, 10, 111);

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                let transformer = PowerTransformer::new();
                if let Ok(fitted) = transformer.fit(&x, &()) {
                    black_box(fitted.transform(&x))
                } else {
                    Ok(x.clone())
                }
            });
        });
    }

    group.finish();
}

/// Benchmark Normalizer transform operations
fn benchmark_normalizer(c: &mut Criterion) {
    let mut group = c.benchmark_group("normalizer");

    for size in [100, 1000, 10000, 100000].iter() {
        let x = generate_data(*size, 10, 789);

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                let normalizer = Normalizer::new();
                black_box(normalizer.transform(&x))
            });
        });
    }

    group.finish();
}

/// Benchmark varying feature dimensions
fn benchmark_feature_dimensions(c: &mut Criterion) {
    let mut group = c.benchmark_group("feature_dimensions");

    for ncols in [5, 10, 50, 100, 500].iter() {
        let x = generate_data(10000, *ncols, 222);

        group.throughput(Throughput::Elements((10000 * ncols) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(ncols), ncols, |b, _| {
            b.iter(|| {
                // Benchmark Normalizer as a representative transformer
                let normalizer = Normalizer::new().norm(NormType::L1);
                black_box(normalizer.transform(&x))
            });
        });
    }

    group.finish();
}

criterion_group!(scalers, benchmark_normalizer,);

criterion_group!(feature_engineering, benchmark_polynomial_features,);

criterion_group!(
    transformers,
    benchmark_quantile_transformer,
    benchmark_power_transformer,
);

criterion_group!(dimensions, benchmark_feature_dimensions,);

criterion_main!(scalers, feature_engineering, transformers, dimensions,);
