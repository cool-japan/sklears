#![cfg(feature = "incomplete-benchmarks")]

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use ndarray::{Array1, Array2};
use ndarray_rand::RandomExt;
use rand_chacha::ChaCha8Rng;
use sklears_model_selection::{
    train_test_split, GroupKFold, KFold, ShuffleSplit, StratifiedKFold, StratifiedShuffleSplit,
    TimeSeriesSplit,
};

fn generate_test_data(n_samples: usize, n_features: usize) -> (Array2<f64>, Array1<i32>) {
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let x = Array2::random_using(
        (n_samples, n_features),
        rand_distr::StandardNormal,
        &mut rng,
    );
    let y = Array1::random_using(n_samples, rand_distr::Uniform::new(0, 3), &mut rng);
    (x, y)
}

fn generate_regression_data(n_samples: usize, n_features: usize) -> (Array2<f64>, Array1<f64>) {
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let x = Array2::random_using(
        (n_samples, n_features),
        rand_distr::StandardNormal,
        &mut rng,
    );
    let y = Array1::random_using(n_samples, rand_distr::StandardNormal, &mut rng);
    (x, y)
}

fn generate_groups(n_samples: usize, n_groups: usize) -> Array1<i32> {
    let mut groups = Array1::zeros(n_samples);
    for (i, group) in groups.iter_mut().enumerate() {
        *group = (i % n_groups) as i32;
    }
    groups
}

fn bench_kfold_cv(c: &mut Criterion) {
    let mut group = c.benchmark_group("KFold Cross-Validation");

    for &n_samples in &[100, 500, 1000, 5000] {
        let (x, y) = generate_test_data(n_samples, 10);

        group.bench_with_input(
            BenchmarkId::new("KFold-5", n_samples),
            &n_samples,
            |b, _| {
                b.iter(|| {
                    let cv = KFold::new(5).unwrap();
                    let splits: Vec<_> =
                        cv.split(black_box(&x), Some(black_box(&y)), None).collect();
                    black_box(splits)
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("KFold-10", n_samples),
            &n_samples,
            |b, _| {
                b.iter(|| {
                    let cv = KFold::new(10).unwrap();
                    let splits: Vec<_> =
                        cv.split(black_box(&x), Some(black_box(&y)), None).collect();
                    black_box(splits)
                })
            },
        );
    }

    group.finish();
}

fn bench_stratified_kfold_cv(c: &mut Criterion) {
    let mut group = c.benchmark_group("Stratified KFold Cross-Validation");

    for &n_samples in &[100, 500, 1000, 5000] {
        let (x, y) = generate_test_data(n_samples, 10);

        group.bench_with_input(
            BenchmarkId::new("StratifiedKFold-5", n_samples),
            &n_samples,
            |b, _| {
                b.iter(|| {
                    let cv = StratifiedKFold::new(5).unwrap();
                    let splits: Vec<_> =
                        cv.split(black_box(&x), Some(black_box(&y)), None).collect();
                    black_box(splits)
                })
            },
        );
    }

    group.finish();
}

fn bench_shuffle_split_cv(c: &mut Criterion) {
    let mut group = c.benchmark_group("Shuffle Split Cross-Validation");

    for &n_samples in &[100, 500, 1000, 5000] {
        let (x, y) = generate_test_data(n_samples, 10);

        group.bench_with_input(
            BenchmarkId::new("ShuffleSplit", n_samples),
            &n_samples,
            |b, _| {
                b.iter(|| {
                    let cv = ShuffleSplit::new(10, Some(0.8), None, Some(42)).unwrap();
                    let splits: Vec<_> =
                        cv.split(black_box(&x), Some(black_box(&y)), None).collect();
                    black_box(splits)
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("StratifiedShuffleSplit", n_samples),
            &n_samples,
            |b, _| {
                b.iter(|| {
                    let cv = StratifiedShuffleSplit::new(10, Some(0.8), None, Some(42)).unwrap();
                    let splits: Vec<_> =
                        cv.split(black_box(&x), Some(black_box(&y)), None).collect();
                    black_box(splits)
                })
            },
        );
    }

    group.finish();
}

fn bench_time_series_split(c: &mut Criterion) {
    let mut group = c.benchmark_group("Time Series Split");

    for &n_samples in &[100, 500, 1000, 5000] {
        let (x, y) = generate_regression_data(n_samples, 10);

        group.bench_with_input(
            BenchmarkId::new("TimeSeriesSplit", n_samples),
            &n_samples,
            |b, _| {
                b.iter(|| {
                    let cv = TimeSeriesSplit::new(5, None, None).unwrap();
                    let splits: Vec<_> =
                        cv.split(black_box(&x), Some(black_box(&y)), None).collect();
                    black_box(splits)
                })
            },
        );
    }

    group.finish();
}

fn bench_group_kfold_cv(c: &mut Criterion) {
    let mut group = c.benchmark_group("Group KFold Cross-Validation");

    for &n_samples in &[100, 500, 1000, 5000] {
        let (x, y) = generate_test_data(n_samples, 10);
        let groups = generate_groups(n_samples, n_samples / 10); // 10% unique groups

        group.bench_with_input(
            BenchmarkId::new("GroupKFold", n_samples),
            &n_samples,
            |b, _| {
                b.iter(|| {
                    let cv = GroupKFold::new(5).unwrap();
                    let splits: Vec<_> = cv
                        .split(black_box(&x), Some(black_box(&y)), Some(black_box(&groups)))
                        .collect();
                    black_box(splits)
                })
            },
        );
    }

    group.finish();
}

fn bench_train_test_split_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("Train-Test Split");

    for &n_samples in &[100, 500, 1000, 5000, 10000] {
        let (x, y) = generate_test_data(n_samples, 10);

        group.bench_with_input(
            BenchmarkId::new("TrainTestSplit", n_samples),
            &n_samples,
            |b, _| {
                b.iter(|| {
                    let result = train_test_split(
                        black_box(&x),
                        black_box(&y),
                        Some(0.8),
                        Some(42),
                        true,
                        None,
                    );
                    black_box(result)
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("StratifiedTrainTestSplit", n_samples),
            &n_samples,
            |b, _| {
                b.iter(|| {
                    let result = train_test_split(
                        black_box(&x),
                        black_box(&y),
                        Some(0.8),
                        Some(42),
                        true,
                        Some(black_box(&y)),
                    );
                    black_box(result)
                })
            },
        );
    }

    group.finish();
}

fn bench_cv_memory_efficiency(c: &mut Criterion) {
    let mut group = c.benchmark_group("CV Memory Efficiency");

    // Test memory efficiency with larger datasets
    for &n_features in &[10, 50, 100, 500] {
        let (x, y) = generate_test_data(1000, n_features);

        group.bench_with_input(
            BenchmarkId::new("KFold-HighDim", n_features),
            &n_features,
            |b, _| {
                b.iter(|| {
                    let cv = KFold::new(5).unwrap();
                    let splits: Vec<_> =
                        cv.split(black_box(&x), Some(black_box(&y)), None).collect();
                    // Force evaluation of all splits
                    for (train_idx, test_idx) in splits {
                        black_box((train_idx.len(), test_idx.len()));
                    }
                })
            },
        );
    }

    group.finish();
}

fn bench_cv_scalability(c: &mut Criterion) {
    let mut group = c.benchmark_group("CV Scalability");

    // Test scalability with different fold counts
    let (x, y) = generate_test_data(1000, 10);

    for &n_folds in &[3, 5, 10, 20] {
        group.bench_with_input(
            BenchmarkId::new("KFold-Scalability", n_folds),
            &n_folds,
            |b, _| {
                b.iter(|| {
                    let cv = KFold::new(n_folds).unwrap();
                    let splits: Vec<_> =
                        cv.split(black_box(&x), Some(black_box(&y)), None).collect();
                    black_box(splits.len())
                })
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_kfold_cv,
    bench_stratified_kfold_cv,
    bench_shuffle_split_cv,
    bench_time_series_split,
    bench_group_kfold_cv,
    bench_train_test_split_performance,
    bench_cv_memory_efficiency,
    bench_cv_scalability
);

criterion_main!(benches);
