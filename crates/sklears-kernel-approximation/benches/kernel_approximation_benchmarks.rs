//! Performance regression benchmarks for sklears-kernel-approximation
//!
//! These benchmarks establish performance baselines and track regression
//! across releases using the Criterion framework. They exercise a
//! representative cross-section of this crate's kernel approximation
//! feature maps: `RBFSampler` and `ArcCosineSampler` (random-feature
//! methods), `Nystroem` (data-dependent low-rank approximation), and
//! `PolynomialSampler` (explicit polynomial expansion). Usage mirrors the
//! crate's own integration test in `src/lib.rs`.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use scirs2_core::ndarray::Array2;
use scirs2_core::random::essentials::Normal;
use scirs2_core::random::{seeded_rng, Distribution};
use sklears_core::traits::{Fit, Transform};
use sklears_kernel_approximation::{ArcCosineSampler, Kernel, Nystroem, PolynomialSampler, RBFSampler};
use std::hint::black_box;

/// Generate test data for benchmarks
fn generate_data(nrows: usize, ncols: usize, seed: u64) -> Array2<f64> {
    let mut rng = seeded_rng(seed);
    let normal = Normal::new(0.0, 1.0).expect("distribution parameters should be valid");

    let data: Vec<f64> = (0..nrows * ncols)
        .map(|_| normal.sample(&mut rng))
        .collect();

    Array2::from_shape_vec((nrows, ncols), data).expect("shape and data length should match")
}

/// Benchmark `RBFSampler` (Random Fourier Features) fit + transform.
fn benchmark_rbf_sampler(c: &mut Criterion) {
    let mut group = c.benchmark_group("rbf_sampler");
    group.sample_size(10);

    for size in [100, 1000, 5000, 10000].iter() {
        let x = generate_data(*size, 20, 11);

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                let rbf = RBFSampler::new(100);
                if let Ok(fitted) = rbf.fit(&x, &()) {
                    black_box(fitted.transform(&x))
                } else {
                    black_box(Ok(x.clone()))
                }
            });
        });
    }

    group.finish();
}

/// Benchmark `ArcCosineSampler` fit + transform.
fn benchmark_arc_cosine_sampler(c: &mut Criterion) {
    let mut group = c.benchmark_group("arc_cosine_sampler");
    group.sample_size(10);

    for size in [100, 1000, 5000].iter() {
        let x = generate_data(*size, 10, 44);

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                let arc_cosine = ArcCosineSampler::new(50).degree(1);
                if let Ok(fitted) = arc_cosine.fit(&x, &()) {
                    black_box(fitted.transform(&x))
                } else {
                    black_box(Ok(x.clone()))
                }
            });
        });
    }

    group.finish();
}

/// Benchmark `Nystroem` (data-dependent low-rank kernel approximation)
/// fit + transform. `n_components` is kept below the smallest sample count
/// since Nystroem samples landmark points from the training data.
fn benchmark_nystroem(c: &mut Criterion) {
    let mut group = c.benchmark_group("nystroem");
    group.sample_size(10);

    for size in [100, 1000, 5000].iter() {
        let x = generate_data(*size, 20, 22);

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                let nystroem = Nystroem::new(Kernel::Rbf { gamma: 0.1 }, 50);
                if let Ok(fitted) = nystroem.fit(&x, &()) {
                    black_box(fitted.transform(&x))
                } else {
                    black_box(Ok(x.clone()))
                }
            });
        });
    }

    group.finish();
}

/// Benchmark `PolynomialSampler` (explicit polynomial kernel expansion)
/// fit + transform.
fn benchmark_polynomial_sampler(c: &mut Criterion) {
    let mut group = c.benchmark_group("polynomial_sampler");
    group.sample_size(10);

    for size in [100, 1000, 5000].iter() {
        let x = generate_data(*size, 10, 33);

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                let poly = PolynomialSampler::new(50).degree(3).gamma(1.0).coef0(1.0);
                if let Ok(fitted) = poly.fit(&x, &()) {
                    black_box(fitted.transform(&x))
                } else {
                    black_box(Ok(x.clone()))
                }
            });
        });
    }

    group.finish();
}

/// Benchmark how `RBFSampler` fit + transform scales with the number of
/// random features (`n_components`) at a fixed sample count.
fn benchmark_n_components_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("rbf_sampler_component_scaling");
    group.sample_size(10);

    for n_components in [50, 100, 200, 500].iter() {
        let x = generate_data(2000, 20, 55);

        group.throughput(Throughput::Elements(*n_components as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(n_components),
            n_components,
            |b, _| {
                b.iter(|| {
                    let rbf = RBFSampler::new(*n_components);
                    if let Ok(fitted) = rbf.fit(&x, &()) {
                        black_box(fitted.transform(&x))
                    } else {
                        black_box(Ok(x.clone()))
                    }
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    random_features,
    benchmark_rbf_sampler,
    benchmark_arc_cosine_sampler,
);

criterion_group!(low_rank_approximation, benchmark_nystroem,);

criterion_group!(explicit_expansion, benchmark_polynomial_sampler,);

criterion_group!(scaling, benchmark_n_components_scaling,);

criterion_main!(
    random_features,
    low_rank_approximation,
    explicit_expansion,
    scaling,
);
