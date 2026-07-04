//! Performance regression benchmarks for sklears-cross-decomposition
//!
//! These benchmarks establish performance baselines and track regression
//! across releases using the Criterion framework, covering `PLSRegression`
//! and `CCA` fit/predict/transform across a range of input sizes.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use scirs2_core::ndarray::Array2;
use scirs2_core::random::essentials::Normal;
use scirs2_core::random::{seeded_rng, Distribution};
use sklears_core::traits::{Fit, Predict, Transform};
use sklears_cross_decomposition::{PLSRegression, CCA};
use std::hint::black_box;

/// Generate test data for benchmarks
fn generate_data(nrows: usize, ncols: usize, seed: u64) -> Array2<f64> {
    let mut rng = seeded_rng(seed);
    let normal = Normal::new(0.0, 1.0).expect("operation should succeed");

    let data: Vec<f64> = (0..nrows * ncols)
        .map(|_| normal.sample(&mut rng))
        .collect();

    Array2::from_shape_vec((nrows, ncols), data).expect("shape and data length should match")
}

/// Benchmark PLSRegression fit + predict across input sizes
fn benchmark_pls_fit_predict(c: &mut Criterion) {
    let mut group = c.benchmark_group("pls_fit_predict");
    group.sample_size(10); // NIPALS is iterative; reduce sample size for slow operations

    for size in [100, 500, 1000, 2000].iter() {
        let x = generate_data(*size, 10, 1001);
        let y = generate_data(*size, 2, 2002);

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                let pls = PLSRegression::new(2);
                if let Ok(fitted) = pls.fit(&x, &y) {
                    black_box(fitted.predict(&x))
                } else {
                    Ok(x.clone())
                }
            });
        });
    }

    group.finish();
}

/// Benchmark PLSRegression fit + transform across input sizes
fn benchmark_pls_transform(c: &mut Criterion) {
    let mut group = c.benchmark_group("pls_transform");
    group.sample_size(10);

    for size in [100, 500, 1000, 2000].iter() {
        let x = generate_data(*size, 10, 3003);
        let y = generate_data(*size, 2, 4004);

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                let pls = PLSRegression::new(2);
                if let Ok(fitted) = pls.fit(&x, &y) {
                    black_box(fitted.transform(&x))
                } else {
                    Ok(x.clone())
                }
            });
        });
    }

    group.finish();
}

/// Benchmark CCA fit + transform across input sizes
fn benchmark_cca_fit_transform(c: &mut Criterion) {
    let mut group = c.benchmark_group("cca_fit_transform");
    group.sample_size(10);

    for size in [100, 500, 1000, 2000].iter() {
        let x = generate_data(*size, 8, 5005);
        let y = generate_data(*size, 3, 6006);

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                let cca = CCA::new(2);
                if let Ok(fitted) = cca.fit(&x, &y) {
                    black_box(fitted.transform(&x))
                } else {
                    Ok(x.clone())
                }
            });
        });
    }

    group.finish();
}

/// Benchmark PLSRegression fit + predict while varying `n_components` at a fixed sample size
fn benchmark_pls_components(c: &mut Criterion) {
    let mut group = c.benchmark_group("pls_components");
    group.sample_size(10);

    let n_samples = 1000;
    let x = generate_data(n_samples, 10, 7007);
    let y = generate_data(n_samples, 5, 8008);

    for n_components in [1, 2, 3, 4, 5].iter() {
        group.throughput(Throughput::Elements((n_samples * n_components) as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(n_components),
            n_components,
            |b, _| {
                b.iter(|| {
                    let pls = PLSRegression::new(*n_components);
                    if let Ok(fitted) = pls.fit(&x, &y) {
                        black_box(fitted.predict(&x))
                    } else {
                        Ok(x.clone())
                    }
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    pls_benchmarks,
    benchmark_pls_fit_predict,
    benchmark_pls_transform,
);

criterion_group!(cca_benchmarks, benchmark_cca_fit_transform,);

criterion_group!(dimensions, benchmark_pls_components,);

criterion_main!(pls_benchmarks, cca_benchmarks, dimensions);
