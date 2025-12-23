//! Benchmarks for KNN algorithms

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::traits::{Fit, Predict, PredictProba};
use sklears_neighbors::{KNeighborsClassifier, KNeighborsRegressor};

fn generate_classification_data(n_samples: usize, n_features: usize) -> (Array2<f64>, Array1<i32>) {
    // Generate simple synthetic data for benchmarking
    use scirs2_core::random::*;
    let mut rng = StdRng::seed_from_u64(42);

    let mut data = Vec::with_capacity(n_samples * n_features);
    let mut labels = Vec::with_capacity(n_samples);

    for i in 0..n_samples {
        let class = i % 3; // 3 classes
        labels.push(class as i32);

        for j in 0..n_features {
            let base_value = class as f64 * 2.0 + j as f64 * 0.1;
            let noise = rng.gen() * 0.5;
            data.push(base_value + noise);
        }
    }

    (
        Array2::from_shape_vec((n_samples, n_features), data).unwrap(),
        Array1::from(labels),
    )
}

fn generate_regression_data(n_samples: usize, n_features: usize) -> (Array2<f64>, Array1<f64>) {
    use scirs2_core::random::*;
    let mut rng = StdRng::seed_from_u64(42);

    let mut data = Vec::with_capacity(n_samples * n_features);
    let mut targets = Vec::with_capacity(n_samples);

    for _ in 0..n_samples {
        let mut feature_sum = 0.0;
        for _ in 0..n_features {
            let value = rng.gen() * 10.0 - 5.0;
            data.push(value);
            feature_sum += value;
        }
        targets.push(feature_sum + rng.gen() * 0.1);
    }

    (
        Array2::from_shape_vec((n_samples, n_features), data).unwrap(),
        Array1::from(targets),
    )
}

fn bench_knn_classification_fit(c: &mut Criterion) {
    let mut group = c.benchmark_group("knn_classification_fit");

    for &n_samples in &[100, 500, 1000] {
        for &n_features in &[5, 10, 20] {
            let (X, y) = generate_classification_data(n_samples, n_features);

            group.throughput(Throughput::Elements(n_samples as u64));
            group.bench_with_input(
                BenchmarkId::new("fit", format!("{}x{}", n_samples, n_features)),
                &(&X, &y),
                |b, (X, y)| {
                    b.iter(|| {
                        let classifier = KNeighborsClassifier::new(5);
                        black_box(classifier.fit(X, y).unwrap())
                    })
                },
            );
        }
    }
    group.finish();
}

fn bench_knn_classification_predict(c: &mut Criterion) {
    let mut group = c.benchmark_group("knn_classification_predict");

    for &n_samples in &[100, 500, 1000] {
        for &n_features in &[5, 10, 20] {
            let (X, y) = generate_classification_data(n_samples, n_features);
            let classifier = KNeighborsClassifier::new(5);
            let fitted_classifier = classifier.fit(&X, &y).unwrap();

            group.throughput(Throughput::Elements(n_samples as u64));
            group.bench_with_input(
                BenchmarkId::new("predict", format!("{}x{}", n_samples, n_features)),
                &(&X, &fitted_classifier),
                |b, (X, classifier)| b.iter(|| black_box(classifier.predict(X).unwrap())),
            );
        }
    }
    group.finish();
}

fn bench_knn_regression_fit(c: &mut Criterion) {
    let mut group = c.benchmark_group("knn_regression_fit");

    for &n_samples in &[100, 500, 1000] {
        for &n_features in &[5, 10, 20] {
            let (X, y) = generate_regression_data(n_samples, n_features);

            group.throughput(Throughput::Elements(n_samples as u64));
            group.bench_with_input(
                BenchmarkId::new("fit", format!("{}x{}", n_samples, n_features)),
                &(&X, &y),
                |b, (X, y)| {
                    b.iter(|| {
                        let regressor = KNeighborsRegressor::new(5);
                        black_box(regressor.fit(X, y).unwrap())
                    })
                },
            );
        }
    }
    group.finish();
}

fn bench_knn_predict_proba(c: &mut Criterion) {
    let mut group = c.benchmark_group("knn_predict_proba");

    for &n_samples in &[100, 500, 1000] {
        let (X, y) = generate_classification_data(n_samples, 10);
        let classifier = KNeighborsClassifier::new(5);
        let fitted_classifier = classifier.fit(&X, &y).unwrap();

        group.throughput(Throughput::Elements(n_samples as u64));
        group.bench_with_input(
            BenchmarkId::new("predict_proba", n_samples),
            &(&X, &fitted_classifier),
            |b, (X, classifier)| b.iter(|| black_box(classifier.predict_proba(X).unwrap())),
        );
    }
    group.finish();
}

fn bench_different_k_values(c: &mut Criterion) {
    let mut group = c.benchmark_group("knn_different_k");
    let (X, y) = generate_classification_data(500, 10);

    for &k in &[1, 3, 5, 10, 20] {
        group.bench_with_input(BenchmarkId::new("predict", k), &k, |b, &k| {
            let classifier = KNeighborsClassifier::new(k);
            let fitted_classifier = classifier.fit(&X, &y).unwrap();
            b.iter(|| black_box(fitted_classifier.predict(&X).unwrap()))
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_knn_classification_fit,
    bench_knn_classification_predict,
    bench_knn_regression_fit,
    bench_knn_predict_proba,
    bench_different_k_values
);

criterion_main!(benches);
