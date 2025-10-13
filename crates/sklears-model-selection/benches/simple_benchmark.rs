#![cfg(feature = "incomplete-benchmarks")]

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use ndarray::{Array1, Array2};
use ndarray_rand::RandomExt;
use rand_chacha::ChaCha8Rng;
use sklears_model_selection::{
    train_test_split, CrossValidator, KFold, ShuffleSplit, StratifiedKFold, TimeSeriesSplit,
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

fn bench_kfold_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("KFold Performance");

    for &n_samples in &[100, 500, 1000, 5000] {
        let (x, y) = generate_test_data(n_samples, 10);

        group.bench_with_input(
            BenchmarkId::new("KFold-5", n_samples),
            &n_samples,
            |b, _| {
                b.iter(|| {
                    let cv = KFold::new(5);
                    let splits = cv.split(black_box(x.nrows()), Some(black_box(&y)));
                    black_box(splits.len())
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("KFold-10", n_samples),
            &n_samples,
            |b, _| {
                b.iter(|| {
                    let cv = KFold::new(10);
                    let splits = cv.split(black_box(x.nrows()), Some(black_box(&y)));
                    black_box(splits.len())
                })
            },
        );
    }

    group.finish();
}

fn bench_stratified_kfold_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("Stratified KFold Performance");

    for &n_samples in &[100, 500, 1000, 5000] {
        let (x, y) = generate_test_data(n_samples, 10);

        group.bench_with_input(
            BenchmarkId::new("StratifiedKFold", n_samples),
            &n_samples,
            |b, _| {
                b.iter(|| {
                    let cv = StratifiedKFold::new(5);
                    let splits = cv.split(black_box(x.nrows()), Some(black_box(&y)));
                    black_box(splits.len())
                })
            },
        );
    }

    group.finish();
}

fn bench_shuffle_split_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("Shuffle Split Performance");

    for &n_samples in &[100, 500, 1000, 5000] {
        let (x, y) = generate_test_data(n_samples, 10);

        group.bench_with_input(
            BenchmarkId::new("ShuffleSplit", n_samples),
            &n_samples,
            |b, _| {
                b.iter(|| {
                    let cv = ShuffleSplit::new(10);
                    let splits = cv.split(black_box(x.nrows()), Some(black_box(&y)));
                    black_box(splits.len())
                })
            },
        );
    }

    group.finish();
}

fn bench_time_series_split_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("Time Series Split Performance");

    for &n_samples in &[100, 500, 1000, 5000] {
        let (x, y) = generate_test_data(n_samples, 10);

        group.bench_with_input(
            BenchmarkId::new("TimeSeriesSplit", n_samples),
            &n_samples,
            |b, _| {
                b.iter(|| {
                    let cv = TimeSeriesSplit::new(5);
                    let splits = cv.split(black_box(x.nrows()), Some(black_box(&y)));
                    black_box(splits.len())
                })
            },
        );
    }

    group.finish();
}

fn bench_train_test_split_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("Train-Test Split Performance");

    for &n_samples in &[100, 500, 1000, 5000, 10000] {
        let (x, y) = generate_test_data(n_samples, 10);

        group.bench_with_input(
            BenchmarkId::new("TrainTestSplit", n_samples),
            &n_samples,
            |b, _| {
                b.iter(|| {
                    let result = train_test_split(black_box(&x), black_box(&y), 0.8, Some(42));
                    black_box(result.is_ok())
                })
            },
        );
    }

    group.finish();
}

fn bench_cv_scalability(c: &mut Criterion) {
    let mut group = c.benchmark_group("CV Scalability");

    let (x, y) = generate_test_data(1000, 20);

    for &n_folds in &[3, 5, 10, 20] {
        group.bench_with_input(
            BenchmarkId::new("KFold-Scalability", n_folds),
            &n_folds,
            |b, _| {
                b.iter(|| {
                    let cv = KFold::new(n_folds);
                    let splits = cv.split(black_box(x.nrows()), Some(black_box(&y)));
                    let mut fold_count = 0;
                    for (train_idx, test_idx) in splits {
                        fold_count += 1;
                        black_box(train_idx.len() + test_idx.len());
                    }
                    black_box(fold_count)
                })
            },
        );
    }

    group.finish();
}

fn bench_memory_efficiency(c: &mut Criterion) {
    let mut group = c.benchmark_group("Memory Efficiency");

    for &n_features in &[10, 50, 100, 500] {
        let (x, y) = generate_test_data(1000, n_features);

        group.bench_with_input(
            BenchmarkId::new("HighDim-KFold", n_features),
            &n_features,
            |b, _| {
                b.iter(|| {
                    let cv = KFold::new(5);
                    let splits = cv.split(black_box(x.nrows()), Some(black_box(&y)));
                    let mut total_samples = 0;

                    for (train_idx, test_idx) in splits {
                        // Only access indices, not actual data to test memory efficiency
                        total_samples += train_idx.len() + test_idx.len();
                    }

                    black_box(total_samples)
                })
            },
        );
    }

    group.finish();
}

fn bench_parallel_vs_sequential(c: &mut Criterion) {
    let mut group = c.benchmark_group("Parallel vs Sequential");
    let (x, y) = generate_test_data(2000, 30);

    group.bench_function("Sequential-CV", |b| {
        b.iter(|| {
            let cv = KFold::new(10);
            let splits = cv.split(x.nrows(), Some(&y));
            let mut results = Vec::new();

            for (train_idx, test_idx) in splits {
                // Simulate some computation
                let computation_result = train_idx.len() * test_idx.len();
                results.push(computation_result);
            }

            black_box(results)
        })
    });

    group.bench_function("Parallel-CV", |b| {
        b.iter(|| {
            let cv = KFold::new(10);
            let splits = cv.split(x.nrows(), Some(&y));

            // Simulate parallel processing
            use rayon::prelude::*;
            let results: Vec<usize> = splits
                .par_iter()
                .map(|(train_idx, test_idx)| {
                    // Simulate some computation
                    train_idx.len() * test_idx.len()
                })
                .collect();

            black_box(results)
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_kfold_performance,
    bench_stratified_kfold_performance,
    bench_shuffle_split_performance,
    bench_time_series_split_performance,
    bench_train_test_split_performance,
    bench_cv_scalability,
    bench_memory_efficiency,
    bench_parallel_vs_sequential
);

criterion_main!(benches);
