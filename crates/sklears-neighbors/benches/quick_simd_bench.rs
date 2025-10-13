//! Quick benchmark to verify SIMD integration performance

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::thread_rng;
use sklears_core::traits::{Fit, Predict};
use sklears_neighbors::{Distance, KNeighborsClassifier};

fn generate_test_data() -> (Array2<f64>, Array1<i32>) {
    let mut rng = thread_rng();
    let n_samples = 1000;
    let n_features = 10;

    let mut data = Array2::zeros((n_samples, n_features));
    let mut targets = Array1::zeros(n_samples);

    for i in 0..n_samples {
        for j in 0..n_features {
            data[[i, j]] = rng.gen_range(-1.0..1.0);
        }
        targets[i] = if data.row(i).sum() > 0.0 { 1 } else { 0 };
    }

    (data, targets)
}

fn bench_knn_simd_integration(c: &mut Criterion) {
    let (x_train, y_train) = generate_test_data();
    let (x_test, _) = generate_test_data();

    // Fit the classifier once
    let classifier = KNeighborsClassifier::new(5).with_metric(Distance::Euclidean);
    let fitted = classifier.fit(&x_train, &y_train).expect("Failed to fit");

    c.bench_function("knn_predict_euclidean_simd", |b| {
        b.iter(|| black_box(fitted.predict(black_box(&x_test)).unwrap()))
    });

    let classifier = KNeighborsClassifier::new(5).with_metric(Distance::Manhattan);
    let fitted = classifier.fit(&x_train, &y_train).expect("Failed to fit");

    c.bench_function("knn_predict_manhattan_simd", |b| {
        b.iter(|| black_box(fitted.predict(black_box(&x_test)).unwrap()))
    });
}

fn bench_distance_functions(c: &mut Criterion) {
    let mut rng = thread_rng();
    let size = 100;
    let a: Array1<f64> = Array1::from_vec((0..size).map(|_| rng.gen_range(-1.0..1.0)).collect());
    let b: Array1<f64> = Array1::from_vec((0..size).map(|_| rng.gen_range(-1.0..1.0)).collect());

    c.bench_function("euclidean_distance_simd", |bench| {
        bench.iter(|| {
            black_box(sklears_neighbors::distance::euclidean_distance(
                black_box(&a.view()),
                black_box(&b.view()),
            ))
        })
    });

    c.bench_function("manhattan_distance_simd", |bench| {
        bench.iter(|| {
            black_box(sklears_neighbors::distance::manhattan_distance(
                black_box(&a.view()),
                black_box(&b.view()),
            ))
        })
    });
}

criterion_group!(
    benches,
    bench_knn_simd_integration,
    bench_distance_functions
);
criterion_main!(benches);
