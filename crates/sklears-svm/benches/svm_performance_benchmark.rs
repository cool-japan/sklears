use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::essentials::Uniform;
use scirs2_core::random::seeded_rng;
use scirs2_core::Distribution;
use sklears_core::prelude::*;
use sklears_core::types::Float;
use sklears_svm::SVC;
use std::hint::black_box;

/// Generate synthetic linearly separable dataset
fn generate_linearly_separable_data(
    n_samples: usize,
    n_features: usize,
    seed: u64,
) -> (Array2<Float>, Array1<Float>) {
    let mut rng = seeded_rng(seed);
    let uniform = Uniform::new(-10.0, 10.0).unwrap();

    // Generate two clusters with clear separation
    let mut x_data = Vec::with_capacity(n_samples * n_features);
    let mut y_data = Vec::with_capacity(n_samples);

    let half = n_samples / 2;

    // Cluster 1: centered around (-5, -5, ...)
    for _ in 0..half {
        for _ in 0..n_features {
            x_data.push(uniform.sample(&mut rng) - 5.0);
        }
        y_data.push(-1.0);
    }

    // Cluster 2: centered around (5, 5, ...)
    for _ in 0..(n_samples - half) {
        for _ in 0..n_features {
            x_data.push(uniform.sample(&mut rng) + 5.0);
        }
        y_data.push(1.0);
    }

    let x = Array2::from_shape_vec((n_samples, n_features), x_data).unwrap();
    let y = Array1::from_vec(y_data);

    (x, y)
}

/// Generate synthetic nonlinearly separable dataset (XOR-like pattern)
fn generate_nonlinear_data(n_samples: usize, seed: u64) -> (Array2<Float>, Array1<Float>) {
    let mut rng = seeded_rng(seed);
    let uniform = Uniform::new(-1.0, 1.0).unwrap();

    let mut x_data = Vec::with_capacity(n_samples * 2);
    let mut y_data = Vec::with_capacity(n_samples);

    for _ in 0..n_samples {
        let x1 = uniform.sample(&mut rng) * 2.0;
        let x2 = uniform.sample(&mut rng) * 2.0;

        // XOR pattern
        let label = if (x1 > 0.0) == (x2 > 0.0) { 1.0 } else { -1.0 };

        x_data.push(x1);
        x_data.push(x2);
        y_data.push(label);
    }

    let x = Array2::from_shape_vec((n_samples, 2), x_data).unwrap();
    let y = Array1::from_vec(y_data);

    (x, y)
}

fn bench_svc_linear_small(c: &mut Criterion) {
    let mut group = c.benchmark_group("SVC_Linear_Small");

    for n_samples in [6, 10, 20, 30, 50].iter() {
        let (x, y) = generate_linearly_separable_data(*n_samples, 2, 42);

        group.bench_with_input(
            BenchmarkId::new("fit", n_samples),
            &(*n_samples, &x, &y),
            |b, (_n, x, y)| {
                b.iter(|| {
                    let model = SVC::new();
                    black_box(model.fit(x, y).unwrap())
                })
            },
        );
    }

    group.finish();
}

fn bench_svc_linear_medium(c: &mut Criterion) {
    let mut group = c.benchmark_group("SVC_Linear_Medium");

    for n_samples in [100, 200].iter() {
        let (x, y) = generate_linearly_separable_data(*n_samples, 2, 42);

        group.bench_with_input(
            BenchmarkId::new("fit", n_samples),
            &(*n_samples, &x, &y),
            |b, (_n, x, y)| {
                b.iter(|| {
                    let model = SVC::new();
                    black_box(model.fit(x, y).unwrap())
                })
            },
        );
    }

    group.finish();
}

fn bench_svc_rbf_small(c: &mut Criterion) {
    let mut group = c.benchmark_group("SVC_RBF_Small");

    for n_samples in [6, 10, 20, 30].iter() {
        let (x, y) = generate_nonlinear_data(*n_samples, 42);

        group.bench_with_input(
            BenchmarkId::new("fit", n_samples),
            &(*n_samples, &x, &y),
            |b, (_n, x, y)| {
                b.iter(|| {
                    let model = SVC::new().rbf(Some(0.5));
                    black_box(model.fit(x, y).unwrap())
                })
            },
        );
    }

    group.finish();
}

// LinearSVC removed - requires integer labels

fn bench_svc_predict(c: &mut Criterion) {
    let mut group = c.benchmark_group("SVC_Predict");

    let (x_train, y_train) = generate_linearly_separable_data(50, 2, 42);
    let (x_test, _) = generate_linearly_separable_data(100, 2, 43);

    let model = SVC::new();
    let trained_model = model.fit(&x_train, &y_train).unwrap();

    group.bench_function("predict_100_samples", |b| {
        b.iter(|| black_box(trained_model.predict(&x_test).unwrap()))
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_svc_linear_small,
    bench_svc_linear_medium,
    bench_svc_rbf_small,
    bench_svc_predict
);
criterion_main!(benches);
