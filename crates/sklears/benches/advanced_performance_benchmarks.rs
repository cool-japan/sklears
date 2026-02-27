//! Advanced performance benchmarks for sklears
//!
//! These benchmarks focus on scaling behavior, memory efficiency,
//! and algorithmic complexity analysis.

#![allow(non_snake_case)]

use criterion::{
    criterion_group, criterion_main, AxisScale, BenchmarkId, Criterion, PlotConfiguration,
    Throughput,
};
use scirs2_core::ndarray::{Array1, Array2};
use sklears_utils::data_generation::make_classification;
use std::hint::black_box;

// Scaling analysis benchmarks
fn benchmark_scaling_analysis(c: &mut Criterion) {
    let mut group = c.benchmark_group("scaling_analysis");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    // Test how different algorithms scale with data size
    let base_features = 10;
    let sample_sizes = vec![100, 200, 500, 1000, 2000, 5000];

    for &n_samples in &sample_sizes {
        let (X, _y) =
            make_classification(n_samples, base_features, 3, None, None, 0.0, 1.0, Some(42))
                .unwrap();

        group.throughput(Throughput::Elements(n_samples as u64));

        // Benchmark data generation scaling
        group.bench_with_input(
            BenchmarkId::new("data_generation", n_samples),
            &n_samples,
            |b, &n_samples| {
                b.iter(|| {
                    black_box(
                        make_classification(
                            n_samples,
                            base_features,
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

        // Benchmark matrix operations scaling
        group.bench_with_input(
            BenchmarkId::new("matrix_transpose", n_samples),
            &X,
            |b, X| b.iter(|| black_box(X.t().to_owned())),
        );

        // Benchmark matrix multiplication scaling
        if n_samples >= base_features {
            group.bench_with_input(
                BenchmarkId::new("matrix_multiply", n_samples),
                &X,
                |b, X| {
                    b.iter(|| {
                        let xt = X.t();
                        black_box(xt.dot(X))
                    })
                },
            );
        }
    }
    group.finish();
}

// Feature scaling benchmarks
fn benchmark_feature_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("feature_scaling");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    let base_samples = 1000;
    let feature_sizes = vec![5, 10, 20, 50, 100, 200];

    for &n_features in &feature_sizes {
        let (X, _y) =
            make_classification(base_samples, n_features, 3, None, None, 0.0, 1.0, Some(42))
                .unwrap();

        group.throughput(Throughput::Elements((base_samples * n_features) as u64));

        // Test how algorithms scale with feature count
        group.bench_with_input(
            BenchmarkId::new("feature_generation", n_features),
            &n_features,
            |b, &n_features| {
                b.iter(|| {
                    black_box(
                        make_classification(
                            base_samples,
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

        // Test feature-wise operations
        group.bench_with_input(BenchmarkId::new("column_means", n_features), &X, |b, X| {
            b.iter(|| {
                let mut means = Vec::with_capacity(X.shape()[1]);
                for j in 0..X.shape()[1] {
                    let col_mean = X.column(j).mean().unwrap();
                    means.push(col_mean);
                }
                black_box(means)
            })
        });

        // Test feature-wise standard deviation
        group.bench_with_input(BenchmarkId::new("column_std", n_features), &X, |b, X| {
            b.iter(|| {
                let mut stds = Vec::with_capacity(X.shape()[1]);
                for j in 0..X.shape()[1] {
                    let col = X.column(j);
                    let mean = col.mean().unwrap();
                    let variance =
                        col.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / col.len() as f64;
                    stds.push(variance.sqrt());
                }
                black_box(stds)
            })
        });
    }
    group.finish();
}

// Memory efficiency benchmarks
fn benchmark_memory_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_patterns");

    let sizes = vec![1000, 5000, 10000];

    for &n_samples in &sizes {
        let (X, y) = make_classification(n_samples, 10, 3, None, None, 0.0, 1.0, Some(42)).unwrap();

        group.throughput(Throughput::Elements(n_samples as u64));

        // Test repeated allocations
        group.bench_with_input(
            BenchmarkId::new("repeated_allocation", n_samples),
            &(&X, &y),
            |b, (X, y)| {
                b.iter(|| {
                    for _ in 0..10 {
                        let copied_x = (*X).to_owned();
                        let copied_y = (*y).to_owned();
                        black_box((copied_x, copied_y));
                    }
                })
            },
        );

        // Test in-place operations vs allocating operations
        group.bench_with_input(
            BenchmarkId::new("inplace_operations", n_samples),
            &X,
            |b, X| {
                b.iter(|| {
                    let mut result = X.clone();
                    result.mapv_inplace(|x| x * 2.0);
                    black_box(result)
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("allocating_operations", n_samples),
            &X,
            |b, X| {
                b.iter(|| {
                    let result = X.mapv(|x| x * 2.0);
                    black_box(result)
                })
            },
        );

        // Test memory access patterns
        group.bench_with_input(BenchmarkId::new("row_access", n_samples), &X, |b, X| {
            b.iter(|| {
                let mut sum = 0.0;
                for i in 0..X.shape()[0] {
                    for j in 0..X.shape()[1] {
                        sum += X[[i, j]];
                    }
                }
                black_box(sum)
            })
        });

        group.bench_with_input(BenchmarkId::new("column_access", n_samples), &X, |b, X| {
            b.iter(|| {
                let mut sum = 0.0;
                for j in 0..X.shape()[1] {
                    for i in 0..X.shape()[0] {
                        sum += X[[i, j]];
                    }
                }
                black_box(sum)
            })
        });
    }
    group.finish();
}

// Algorithmic complexity benchmarks
fn benchmark_algorithmic_complexity(c: &mut Criterion) {
    let mut group = c.benchmark_group("algorithmic_complexity");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    // Test O(n) algorithms
    let sizes = vec![100, 200, 500, 1000, 2000, 5000];

    for &n in &sizes {
        let data = Array1::from_vec((0..n).map(|i| i as f64).collect());

        group.throughput(Throughput::Elements(n as u64));

        // O(n) - linear scan
        group.bench_with_input(BenchmarkId::new("linear_scan", n), &data, |b, data| {
            b.iter(|| {
                let mut sum = 0.0;
                for &x in data.iter() {
                    sum += x;
                }
                black_box(sum)
            })
        });

        // O(n log n) - sorting
        group.bench_with_input(BenchmarkId::new("sorting", n), &data, |b, data| {
            b.iter(|| {
                let mut sorted_data = data.to_vec();
                sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());
                black_box(sorted_data)
            })
        });

        // O(n²) - nested loops
        if n <= 1000 {
            // Limit for quadratic algorithms
            group.bench_with_input(BenchmarkId::new("quadratic", n), &data, |b, data| {
                b.iter(|| {
                    let mut sum = 0.0;
                    for &x in data.iter() {
                        for &y in data.iter() {
                            sum += x * y;
                        }
                    }
                    black_box(sum)
                })
            });
        }
    }
    group.finish();
}

// Cache efficiency benchmarks
fn benchmark_cache_efficiency(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_efficiency");

    // Test different array sizes to see cache effects
    let sizes = vec![
        (100, 100),   // Should fit in L1 cache
        (500, 500),   // Should fit in L2 cache
        (1000, 1000), // Should fit in L3 cache
        (2000, 2000), // Larger than typical L3 cache
    ];

    for &(rows, cols) in &sizes {
        let X = Array2::<f64>::zeros((rows, cols));

        group.throughput(Throughput::Elements((rows * cols) as u64));

        // Test sequential access (cache-friendly)
        group.bench_with_input(
            BenchmarkId::new("sequential_access", format!("{}x{}", rows, cols)),
            &X,
            |b, X| {
                b.iter(|| {
                    let mut sum = 0.0;
                    for row in X.rows() {
                        for &val in row.iter() {
                            sum += val;
                        }
                    }
                    black_box(sum)
                })
            },
        );

        // Test strided access (cache-unfriendly)
        group.bench_with_input(
            BenchmarkId::new("strided_access", format!("{}x{}", rows, cols)),
            &X,
            |b, X| {
                b.iter(|| {
                    let mut sum = 0.0;
                    let stride = 7; // Prime number for irregular access
                    for i in (0..X.len()).step_by(stride) {
                        let row = i / X.shape()[1];
                        let col = i % X.shape()[1];
                        if row < X.shape()[0] && col < X.shape()[1] {
                            sum += X[[row, col]];
                        }
                    }
                    black_box(sum)
                })
            },
        );

        // Test block access (cache-optimized)
        group.bench_with_input(
            BenchmarkId::new("block_access", format!("{}x{}", rows, cols)),
            &X,
            |b, X| {
                b.iter(|| {
                    let mut sum = 0.0;
                    let block_size = 64; // Typical cache line size

                    for block_row in (0..X.shape()[0]).step_by(block_size) {
                        for block_col in (0..X.shape()[1]).step_by(block_size) {
                            for i in block_row..std::cmp::min(block_row + block_size, X.shape()[0])
                            {
                                for j in
                                    block_col..std::cmp::min(block_col + block_size, X.shape()[1])
                                {
                                    sum += X[[i, j]];
                                }
                            }
                        }
                    }
                    black_box(sum)
                })
            },
        );
    }
    group.finish();
}

// Numerical precision benchmarks
fn benchmark_numerical_precision(c: &mut Criterion) {
    let mut group = c.benchmark_group("numerical_precision");

    let precisions = vec![("f32", 1e-6), ("f64", 1e-15)];

    for &(precision_name, epsilon) in &precisions {
        let n = 1000;
        let data = Array1::from_vec((0..n).map(|i| (i as f64) * epsilon).collect());

        group.throughput(Throughput::Elements(n as u64));

        // Test accumulation precision
        group.bench_with_input(
            BenchmarkId::new("accumulation", precision_name),
            &data,
            |b, data| {
                b.iter(|| {
                    let mut sum = 0.0;
                    for &x in data.iter() {
                        sum += x;
                    }
                    black_box(sum)
                })
            },
        );

        // Test Kahan summation (compensated summation)
        group.bench_with_input(
            BenchmarkId::new("kahan_summation", precision_name),
            &data,
            |b, data| {
                b.iter(|| {
                    let mut sum = 0.0;
                    let mut c = 0.0; // Compensation term

                    for &x in data.iter() {
                        let y = x - c;
                        let t = sum + y;
                        c = (t - sum) - y;
                        sum = t;
                    }
                    black_box(sum)
                })
            },
        );

        // Test numerical stability in matrix operations
        let small_matrix =
            Array2::from_shape_fn((50, 50), |(i, j)| epsilon * (i as f64 + j as f64));

        group.bench_with_input(
            BenchmarkId::new("matrix_operations", precision_name),
            &small_matrix,
            |b, matrix| {
                b.iter(|| {
                    let transpose = matrix.t();
                    let result = transpose.dot(matrix);
                    black_box(result)
                })
            },
        );
    }
    group.finish();
}

criterion_group!(
    advanced_benches,
    benchmark_scaling_analysis,
    benchmark_feature_scaling,
    benchmark_memory_patterns,
    benchmark_algorithmic_complexity,
    benchmark_cache_efficiency,
    benchmark_numerical_precision
);

criterion_main!(advanced_benches);
