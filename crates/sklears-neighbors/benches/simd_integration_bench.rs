//! Benchmark SIMD integration performance improvements for KNN algorithms

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::thread_rng;
use sklears_core::traits::{Fit, Predict};
use sklears_neighbors::{Distance, KNeighborsClassifier};

fn generate_classification_data(n_samples: usize, n_features: usize) -> (Array2<f64>, Array1<i32>) {
    let mut rng = thread_rng();

    let mut data = Array2::zeros((n_samples, n_features));
    let mut targets = Array1::zeros(n_samples);

    for i in 0..n_samples {
        for j in 0..n_features {
            data[[i, j]] = rng.gen_range(-1.0..1.0);
        }
        // Simple binary classification based on sum of features
        targets[i] = if data.row(i).sum() > 0.0 { 1 } else { 0 };
    }

    (data, targets)
}

fn benchmark_knn_euclidean(c: &mut Criterion) {
    let mut group = c.benchmark_group("KNN_Euclidean_SIMD");

    let sizes = vec![100, 500, 1000, 2000];
    let n_features = 20;
    let k = 5;

    for &n_samples in &sizes {
        let (x_train, y_train) = generate_classification_data(n_samples, n_features);
        let (x_test, _) = generate_classification_data(n_samples / 10, n_features);

        let classifier = KNeighborsClassifier::new(k).with_metric(Distance::Euclidean);
        let fitted = classifier
            .fit(&x_train, &y_train)
            .expect("Failed to fit classifier");

        group.bench_with_input(BenchmarkId::new("fit", n_samples), &n_samples, |b, _| {
            let classifier = KNeighborsClassifier::new(k).with_metric(Distance::Euclidean);
            b.iter(|| {
                black_box(
                    classifier
                        .clone()
                        .fit(black_box(&x_train), black_box(&y_train))
                        .unwrap(),
                )
            });
        });

        group.bench_with_input(
            BenchmarkId::new("predict", n_samples),
            &n_samples,
            |b, _| {
                b.iter(|| black_box(fitted.predict(black_box(&x_test)).unwrap()));
            },
        );
    }

    group.finish();
}

fn benchmark_knn_manhattan(c: &mut Criterion) {
    let mut group = c.benchmark_group("KNN_Manhattan_SIMD");

    let sizes = vec![100, 500, 1000, 2000];
    let n_features = 20;
    let k = 5;

    for &n_samples in &sizes {
        let (x_train, y_train) = generate_classification_data(n_samples, n_features);
        let (x_test, _) = generate_classification_data(n_samples / 10, n_features);

        let classifier = KNeighborsClassifier::new(k).with_metric(Distance::Manhattan);
        let fitted = classifier
            .fit(&x_train, &y_train)
            .expect("Failed to fit classifier");

        group.bench_with_input(
            BenchmarkId::new("predict", n_samples),
            &n_samples,
            |b, _| {
                b.iter(|| black_box(fitted.predict(black_box(&x_test)).unwrap()));
            },
        );
    }

    group.finish();
}

fn benchmark_knn_cosine(c: &mut Criterion) {
    let mut group = c.benchmark_group("KNN_Cosine_SIMD");

    let sizes = vec![100, 500, 1000, 2000];
    let n_features = 20;
    let k = 5;

    for &n_samples in &sizes {
        let (x_train, y_train) = generate_classification_data(n_samples, n_features);
        let (x_test, _) = generate_classification_data(n_samples / 10, n_features);

        let classifier = KNeighborsClassifier::new(k).with_metric(Distance::Cosine);
        let fitted = classifier
            .fit(&x_train, &y_train)
            .expect("Failed to fit classifier");

        group.bench_with_input(
            BenchmarkId::new("predict", n_samples),
            &n_samples,
            |b, _| {
                b.iter(|| black_box(fitted.predict(black_box(&x_test)).unwrap()));
            },
        );
    }

    group.finish();
}

fn benchmark_distance_computation_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("Distance_Computation_SIMD");

    let vector_sizes = vec![10, 50, 100, 500, 1000];

    for &size in &vector_sizes {
        let mut rng = thread_rng();
        let a: Array1<f64> =
            Array1::from_vec((0..size).map(|_| rng.gen_range(-1.0..1.0)).collect());
        let b: Array1<f64> =
            Array1::from_vec((0..size).map(|_| rng.gen_range(-1.0..1.0)).collect());

        group.bench_with_input(BenchmarkId::new("euclidean", size), &size, |bench, _| {
            bench.iter(|| {
                black_box(sklears_neighbors::distance::euclidean_distance(
                    black_box(&a.view()),
                    black_box(&b.view()),
                ))
            });
        });

        group.bench_with_input(BenchmarkId::new("manhattan", size), &size, |bench, _| {
            bench.iter(|| {
                black_box(sklears_neighbors::distance::manhattan_distance(
                    black_box(&a.view()),
                    black_box(&b.view()),
                ))
            });
        });

        group.bench_with_input(BenchmarkId::new("cosine", size), &size, |bench, _| {
            bench.iter(|| {
                black_box(sklears_neighbors::distance::cosine_distance(
                    black_box(&a.view()),
                    black_box(&b.view()),
                ))
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    benchmark_knn_euclidean,
    benchmark_knn_manhattan,
    benchmark_knn_cosine,
    benchmark_distance_computation_comparison
);
criterion_main!(benches);
