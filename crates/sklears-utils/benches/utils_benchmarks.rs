//! Performance benchmarks for utility functions
//!
//! These benchmarks measure the performance of various utility functions
//! including data generation, validation, and random sampling.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use scirs2_core::ndarray::{Array1, Array2};
use sklears_utils::{
    array_utils::{label_counts, unique_labels},
    data_generation::{make_blobs, make_classification, make_regression},
    metrics::{euclidean_distance, manhattan_distance},
    random::{
        bootstrap_indices, k_fold_indices, stratified_split_indices, train_test_split_indices,
    },
    validation::{check_array_2d, check_consistent_length, check_finite},
};

fn bench_data_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("Data Generation");

    let data_sizes = vec![(100, 10), (500, 20), (1000, 50), (2000, 100)];

    for &(n_samples, n_features) in &data_sizes {
        group.bench_with_input(
            BenchmarkId::new(
                "make_classification",
                format!("{}x{}", n_samples, n_features),
            ),
            &(n_samples, n_features),
            |b, &(n_samples, n_features)| {
                b.iter(|| {
                    black_box(
                        make_classification(
                            n_samples,
                            n_features,
                            3,
                            None,
                            None,
                            0.0,
                            1.0,
                            Some(42),
                        )
                        .unwrap(),
                    )
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("make_regression", format!("{}x{}", n_samples, n_features)),
            &(n_samples, n_features),
            |b, &(n_samples, n_features)| {
                b.iter(|| {
                    black_box(
                        make_regression(
                            n_samples,
                            n_features,
                            Some(n_features),
                            0.1,
                            0.0,
                            Some(42),
                        )
                        .unwrap(),
                    )
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("make_blobs", format!("{}x{}", n_samples, n_features)),
            &(n_samples, n_features),
            |b, &(n_samples, n_features)| {
                b.iter(|| {
                    black_box(
                        make_blobs(n_samples, n_features, Some(3), 1.0, (-5.0, 5.0), Some(42))
                            .unwrap(),
                    )
                })
            },
        );
    }

    group.finish();
}

fn bench_random_sampling(c: &mut Criterion) {
    let mut group = c.benchmark_group("Random Sampling");

    let sample_sizes = vec![100, 500, 1000, 5000, 10000];

    for &n_samples in &sample_sizes {
        group.bench_with_input(
            BenchmarkId::new("train_test_split", n_samples),
            &n_samples,
            |b, &n_samples| {
                b.iter(|| {
                    black_box(train_test_split_indices(n_samples, 0.2, true, Some(42)).unwrap())
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("bootstrap_indices", n_samples),
            &n_samples,
            |b, &n_samples| b.iter(|| black_box(bootstrap_indices(n_samples, Some(42)))),
        );

        // K-fold with reasonable number of splits
        let n_splits = if n_samples >= 10 { 5 } else { 2 };
        group.bench_with_input(
            BenchmarkId::new("k_fold_indices", n_samples),
            &(n_samples, n_splits),
            |b, &(n_samples, n_splits)| {
                b.iter(|| black_box(k_fold_indices(n_samples, n_splits, true, Some(42)).unwrap()))
            },
        );

        // Stratified split
        if n_samples >= 20 {
            let labels: Vec<i32> = (0..n_samples).map(|i| (i % 3) as i32).collect();
            group.bench_with_input(
                BenchmarkId::new("stratified_split", n_samples),
                &labels,
                |b, labels| {
                    b.iter(|| black_box(stratified_split_indices(labels, 0.3, Some(42)).unwrap()))
                },
            );
        }
    }

    group.finish();
}

fn bench_validation_functions(c: &mut Criterion) {
    let mut group = c.benchmark_group("Validation Functions");

    let data_sizes = vec![(100, 10), (500, 50), (1000, 100), (2000, 200)];

    for &(n_samples, n_features) in &data_sizes {
        let x = Array2::<f64>::ones((n_samples, n_features));
        let y = Array1::<i32>::zeros(n_samples);

        group.bench_with_input(
            BenchmarkId::new("check_array_2d", format!("{}x{}", n_samples, n_features)),
            &x,
            |b, x| b.iter(|| black_box(check_array_2d(x).unwrap())),
        );

        let y_float = y.mapv(|v| v as f64);
        group.bench_with_input(
            BenchmarkId::new(
                "check_consistent_length",
                format!("{}x{}", n_samples, n_features),
            ),
            &(&x, &y_float),
            |b, (x, y)| {
                let x_slice = x.row(0).to_owned();
                b.iter(|| black_box(check_consistent_length(&[&x_slice, y]).unwrap()))
            },
        );

        group.bench_with_input(
            BenchmarkId::new("check_finite", format!("{}x{}", n_samples, n_features)),
            &x,
            |b, x| b.iter(|| black_box(check_finite(x).unwrap())),
        );
    }

    group.finish();
}

fn bench_array_utilities(c: &mut Criterion) {
    let mut group = c.benchmark_group("Array Utilities");

    let label_sizes = vec![100, 500, 1000, 5000];
    let n_classes = 10;

    for &n_labels in &label_sizes {
        let labels: Array1<i32> =
            Array1::from_vec((0..n_labels).map(|i| (i % n_classes) as i32).collect());

        group.bench_with_input(
            BenchmarkId::new("unique_labels", n_labels),
            &labels,
            |b, labels| b.iter(|| black_box(unique_labels(labels))),
        );

        group.bench_with_input(
            BenchmarkId::new("label_counts", n_labels),
            &labels,
            |b, labels| b.iter(|| black_box(label_counts(labels))),
        );
    }

    group.finish();
}

fn bench_distance_functions(c: &mut Criterion) {
    let mut group = c.benchmark_group("Distance Functions");

    let dimensions = vec![2, 5, 10, 50, 100, 500];

    for &dim in &dimensions {
        let point1 = Array1::<f64>::zeros(dim);
        let point2 = Array1::<f64>::ones(dim);

        group.bench_with_input(
            BenchmarkId::new("euclidean_distance", dim),
            &(&point1, &point2),
            |b, (p1, p2)| b.iter(|| black_box(euclidean_distance(p1, p2))),
        );

        group.bench_with_input(
            BenchmarkId::new("manhattan_distance", dim),
            &(&point1, &point2),
            |b, (p1, p2)| b.iter(|| black_box(manhattan_distance(p1, p2))),
        );
    }

    group.finish();
}

fn bench_cross_validation_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("Cross Validation Scaling");

    let sample_sizes = vec![100, 500, 1000, 5000];
    let split_sizes = vec![3, 5, 10];

    for &n_samples in &sample_sizes {
        for &n_splits in &split_sizes {
            if n_samples >= n_splits * 2 {
                group.bench_with_input(
                    BenchmarkId::new("k_fold", format!("{}s_{}splits", n_samples, n_splits)),
                    &(n_samples, n_splits),
                    |b, &(n_samples, n_splits)| {
                        b.iter(|| {
                            black_box(k_fold_indices(n_samples, n_splits, true, Some(42)).unwrap())
                        })
                    },
                );
            }
        }
    }

    group.finish();
}

fn bench_data_generation_parameters(c: &mut Criterion) {
    let mut group = c.benchmark_group("Data Generation Parameters");

    let n_samples = 1000;
    let n_features = 20;

    // Benchmark different number of classes
    let class_counts = vec![2, 5, 10, 20];
    for &n_classes in &class_counts {
        if n_classes <= n_features {
            group.bench_with_input(
                BenchmarkId::new("classification_classes", n_classes),
                &n_classes,
                |b, &n_classes| {
                    b.iter(|| {
                        black_box(
                            make_classification(
                                n_samples,
                                n_features,
                                n_classes,
                                None,
                                None,
                                0.0,
                                1.0,
                                Some(42),
                            )
                            .unwrap(),
                        )
                    })
                },
            );
        }
    }

    // Benchmark different noise levels
    let noise_levels = vec![0.0, 0.1, 0.5, 1.0];
    for &noise in &noise_levels {
        group.bench_with_input(
            BenchmarkId::new("regression_noise", format!("{:.1}", noise)),
            &noise,
            |b, &noise| {
                b.iter(|| {
                    black_box(
                        make_regression(
                            n_samples,
                            n_features,
                            Some(n_features),
                            noise,
                            0.0,
                            Some(42),
                        )
                        .unwrap(),
                    )
                })
            },
        );
    }

    // Benchmark different cluster standards
    let cluster_stds = vec![0.5, 1.0, 2.0, 5.0];
    for &cluster_std in &cluster_stds {
        group.bench_with_input(
            BenchmarkId::new("blobs_std", format!("{:.1}", cluster_std)),
            &cluster_std,
            |b, &cluster_std| {
                b.iter(|| {
                    black_box(
                        make_blobs(
                            n_samples,
                            n_features,
                            Some(3),
                            cluster_std,
                            (-5.0, 5.0),
                            Some(42),
                        )
                        .unwrap(),
                    )
                })
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_data_generation,
    bench_random_sampling,
    bench_validation_functions,
    bench_array_utilities,
    bench_distance_functions,
    bench_cross_validation_scaling,
    bench_data_generation_parameters,
);

criterion_main!(benches);
