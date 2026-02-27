//! Benchmarks for semi-supervised learning methods
//!
//! These benchmarks compare the performance of sklears semi-supervised methods
//! against their scikit-learn equivalents where applicable.

#![allow(non_snake_case)]

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use scirs2_core::ndarray_ext::{Array1, Array2};
use scirs2_core::random::Random;
use sklears_core::traits::{Fit, Predict};
use sklears_semi_supervised::{
    GraphBuilder, GraphPipeline, KNNGraphBuilder, LabelPropagation, LabelSpreading,
    NormalizeTransform, SelfTrainingClassifier, SymmetrizeTransform,
};

/// Generate synthetic semi-supervised dataset
fn generate_dataset(
    n_samples: usize,
    n_features: usize,
    n_labeled: usize,
) -> (Array2<f64>, Array1<i32>) {
    let mut rng = Random::seed(42);

    let mut X = Array2::<f64>::zeros((n_samples, n_features));
    let mut y = Array1::<i32>::from_elem(n_samples, -1);

    // Generate data
    for i in 0..n_samples {
        for j in 0..n_features {
            X[[i, j]] = rng.random_range(-1.0..1.0);
        }
    }

    // Label first n_labeled samples
    for i in 0..n_labeled {
        y[i] = if X.row(i).sum() > 0.0 { 1 } else { 0 };
    }

    (X, y)
}

/// Benchmark label propagation
fn bench_label_propagation(c: &mut Criterion) {
    let mut group = c.benchmark_group("label_propagation");

    for n_samples in [100, 500, 1000].iter() {
        let (X, y) = generate_dataset(*n_samples, 10, n_samples / 10);

        group.bench_with_input(BenchmarkId::new("fit", n_samples), n_samples, |b, _| {
            b.iter(|| {
                let lp = LabelPropagation::new().n_neighbors(5).max_iter(10);
                let _ = lp.fit(&X.view(), &y.view());
            });
        });

        // Benchmark prediction
        let lp = LabelPropagation::new().n_neighbors(5).max_iter(10);
        let fitted = lp.fit(&X.view(), &y.view()).unwrap();

        group.bench_with_input(BenchmarkId::new("predict", n_samples), n_samples, |b, _| {
            b.iter(|| {
                let _ = fitted.predict(&X.view());
            });
        });
    }

    group.finish();
}

/// Benchmark label spreading
fn bench_label_spreading(c: &mut Criterion) {
    let mut group = c.benchmark_group("label_spreading");

    for n_samples in [100, 500, 1000].iter() {
        let (X, y) = generate_dataset(*n_samples, 10, n_samples / 10);

        group.bench_with_input(BenchmarkId::new("fit", n_samples), n_samples, |b, _| {
            b.iter(|| {
                let ls = LabelSpreading::new().n_neighbors(5).alpha(0.2).max_iter(10);
                let _ = ls.fit(&X.view(), &y.view());
            });
        });
    }

    group.finish();
}

/// Benchmark self-training
fn bench_self_training(c: &mut Criterion) {
    let mut group = c.benchmark_group("self_training");

    for n_samples in [100, 500].iter() {
        let (X, y) = generate_dataset(*n_samples, 10, n_samples / 10);

        group.bench_with_input(BenchmarkId::new("fit", n_samples), n_samples, |b, _| {
            b.iter(|| {
                let st = SelfTrainingClassifier::new().threshold(0.75).max_iter(5);
                let _ = st.fit(&X.view(), &y.view());
            });
        });
    }

    group.finish();
}

/// Benchmark composable graph construction
fn bench_composable_graph(c: &mut Criterion) {
    let mut group = c.benchmark_group("composable_graph");

    for n_samples in [100, 500, 1000].iter() {
        let (X, _) = generate_dataset(*n_samples, 10, n_samples / 10);

        group.bench_with_input(
            BenchmarkId::new("knn_graph", n_samples),
            n_samples,
            |b, _| {
                b.iter(|| {
                    let builder = KNNGraphBuilder::new(5);
                    let _ = builder.build(&X.view());
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("pipeline", n_samples),
            n_samples,
            |b, _| {
                b.iter(|| {
                    let pipeline = GraphPipeline::new(KNNGraphBuilder::new(5))
                        .add_transform(SymmetrizeTransform::new("average".to_string()))
                        .add_transform(NormalizeTransform::new("row".to_string()));
                    let _ = pipeline.build(&X.view());
                });
            },
        );
    }

    group.finish();
}

/// Benchmark graph-based vs non-graph methods
fn bench_method_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("method_comparison");
    let (X, y) = generate_dataset(500, 10, 50);

    group.bench_function("label_propagation", |b| {
        b.iter(|| {
            let lp = LabelPropagation::new().n_neighbors(5).max_iter(10);
            let fitted = lp.fit(&X.view(), &y.view()).unwrap();
            let _ = fitted.predict(&X.view());
        });
    });

    group.bench_function("label_spreading", |b| {
        b.iter(|| {
            let ls = LabelSpreading::new().n_neighbors(5).alpha(0.2).max_iter(10);
            let fitted = ls.fit(&X.view(), &y.view()).unwrap();
            let _ = fitted.predict(&X.view());
        });
    });

    group.bench_function("self_training", |b| {
        b.iter(|| {
            let st = SelfTrainingClassifier::new().threshold(0.75).max_iter(5);
            let fitted = st.fit(&X.view(), &y.view()).unwrap();
            let _ = fitted.predict(&X.view());
        });
    });

    group.finish();
}

/// Benchmark scalability with different dataset sizes
fn bench_scalability(c: &mut Criterion) {
    let mut group = c.benchmark_group("scalability");

    for n_samples in [50, 100, 200, 500, 1000].iter() {
        let (X, y) = generate_dataset(*n_samples, 10, n_samples / 10);

        group.bench_with_input(
            BenchmarkId::new("label_propagation", n_samples),
            n_samples,
            |b, _| {
                b.iter(|| {
                    let lp = LabelPropagation::new().n_neighbors(5).max_iter(5);
                    let fitted = lp.fit(&X.view(), &y.view()).unwrap();
                    let _ = fitted.predict(&X.view());
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_label_propagation,
    bench_label_spreading,
    bench_self_training,
    bench_composable_graph,
    bench_method_comparison,
    bench_scalability
);
criterion_main!(benches);
