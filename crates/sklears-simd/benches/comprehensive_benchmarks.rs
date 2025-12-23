use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use scirs2_core::random::thread_rng;
use sklears_simd::activation::{relu, sigmoid, tanh_activation};
use sklears_simd::benchmark_framework::{
    BenchmarkSuite, Duration, OptimizationAdvisor, RegressionDetector,
};
use sklears_simd::distance::{cosine_distance, euclidean_distance, manhattan_distance};
use sklears_simd::kernels::{linear_kernel, polynomial_kernel, rbf_kernel};
use sklears_simd::matrix::{matrix_multiply_f32_simd, transpose_simd};
// use sklears_simd::optimization::{gradient_descent_step, momentum_update, soft_threshold};
// use sklears_simd::reduction::{parallel_max, parallel_min, parallel_sum};
use sklears_simd::vector::{dot_product, mean, norm_l2};

fn generate_random_vector(size: usize) -> Vec<f32> {
    let mut rng = thread_rng();
    (0..size).map(|_| rng.random_range(-1.0, 1.0)).collect()
}

fn generate_random_matrix(rows: usize, cols: usize) -> Vec<Vec<f32>> {
    (0..rows).map(|_| generate_random_vector(cols)).collect()
}

/// Comprehensive cross-platform benchmarks using the new framework
fn bench_cross_platform_vector_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("cross_platform_vector_ops");
    let mut suite = BenchmarkSuite::new();
    let mut advisor = OptimizationAdvisor::new();

    for size in [1000, 10000, 100000].iter() {
        let a = generate_random_vector(*size);
        let b = generate_random_vector(*size);

        // Cross-platform dot product comparison
        let dot_result = suite.cross_platform_benchmark("dot_product", *size, |data| {
            if data.len() >= 2 {
                data[0] * data[1]
            } else {
                0.0
            }
        });
        advisor.add_results(dot_result);

        // Benchmark individual operations
        group.bench_with_input(
            BenchmarkId::new("dot_product_comprehensive", size),
            size,
            |bench, _| {
                bench.iter(|| dot_product(black_box(&a), black_box(&b)));
            },
        );

        group.bench_with_input(
            BenchmarkId::new("norm_l2_comprehensive", size),
            size,
            |bench, _| {
                bench.iter(|| norm_l2(black_box(&a)));
            },
        );

        group.bench_with_input(
            BenchmarkId::new("mean_comprehensive", size),
            size,
            |bench, _| {
                bench.iter(|| mean(black_box(&a)));
            },
        );
    }

    // Generate and print optimization recommendations
    let recommendations = advisor.generate_recommendations();
    if !recommendations.is_empty() {
        println!("\n=== Optimization Recommendations ===");
        for rec in recommendations {
            println!("- {}: {}", rec.operation, rec.description);
        }
    }

    group.finish();
}

/// Benchmark different distance metrics across platforms
fn bench_cross_platform_distances(c: &mut Criterion) {
    let mut group = c.benchmark_group("cross_platform_distances");
    let mut suite = BenchmarkSuite::new();

    for size in [100, 1000, 10000].iter() {
        let a = generate_random_vector(*size);
        let b = generate_random_vector(*size);

        // Test euclidean distance across platforms
        let _euclidean_result =
            suite.cross_platform_benchmark("euclidean_distance", *size, |data| {
                if data.len() >= 2 {
                    (data[0] - data[1]).abs()
                } else {
                    0.0
                }
            });

        group.bench_with_input(
            BenchmarkId::new("euclidean_distance_advanced", size),
            size,
            |bench, _| {
                bench.iter(|| euclidean_distance(black_box(&a), black_box(&b)));
            },
        );

        group.bench_with_input(
            BenchmarkId::new("manhattan_distance_advanced", size),
            size,
            |bench, _| {
                bench.iter(|| manhattan_distance(black_box(&a), black_box(&b)));
            },
        );

        group.bench_with_input(
            BenchmarkId::new("cosine_distance_advanced", size),
            size,
            |bench, _| {
                bench.iter(|| cosine_distance(black_box(&a), black_box(&b)));
            },
        );
    }

    group.finish();
}

/// Benchmark activation functions with performance analysis
fn bench_activation_performance_analysis(c: &mut Criterion) {
    let mut group = c.benchmark_group("activation_performance_analysis");
    let mut suite = BenchmarkSuite::new();

    for size in [1000, 10000, 50000].iter() {
        let input = generate_random_vector(*size);
        let mut output = vec![0.0f32; *size];

        // Benchmark different activation functions
        let relu_result = suite.benchmark("relu_performance", 1000, || relu(&input, &mut output));

        let sigmoid_result =
            suite.benchmark("sigmoid_performance", 1000, || sigmoid(&input, &mut output));

        let tanh_result = suite.benchmark("tanh_performance", 1000, || {
            tanh_activation(&input, &mut output)
        });

        // Compare performance
        group.bench_with_input(BenchmarkId::new("relu_analyzed", size), size, |bench, _| {
            bench.iter(|| relu(black_box(&input), black_box(&mut output)));
        });

        group.bench_with_input(
            BenchmarkId::new("sigmoid_analyzed", size),
            size,
            |bench, _| {
                bench.iter(|| sigmoid(black_box(&input), black_box(&mut output)));
            },
        );

        group.bench_with_input(BenchmarkId::new("tanh_analyzed", size), size, |bench, _| {
            bench.iter(|| tanh_activation(black_box(&input), black_box(&mut output)));
        });

        // Print performance comparison for this size
        if *size == 10000 {
            println!("\n=== Activation Function Performance (size: {}) ===", size);
            println!(
                "ReLU: {:?} ({} iter/s)",
                relu_result.duration,
                relu_result.throughput.unwrap_or(0.0)
            );
            println!(
                "Sigmoid: {:?} ({} iter/s)",
                sigmoid_result.duration,
                sigmoid_result.throughput.unwrap_or(0.0)
            );
            println!(
                "Tanh: {:?} ({} iter/s)",
                tanh_result.duration,
                tanh_result.throughput.unwrap_or(0.0)
            );
        }
    }

    group.finish();
}

/// Benchmark kernel functions with cross-platform analysis
fn bench_kernel_cross_platform(c: &mut Criterion) {
    let mut group = c.benchmark_group("kernel_cross_platform");

    for size in [100, 500, 1000].iter() {
        let a = generate_random_vector(*size);
        let b = generate_random_vector(*size);

        group.bench_with_input(
            BenchmarkId::new("rbf_kernel_cross", size),
            size,
            |bench, _| {
                bench.iter(|| rbf_kernel(black_box(&a), black_box(&b), black_box(1.0)));
            },
        );

        group.bench_with_input(
            BenchmarkId::new("polynomial_kernel_cross", size),
            size,
            |bench, _| {
                bench.iter(|| {
                    polynomial_kernel(
                        black_box(&a),
                        black_box(&b),
                        black_box(3.0),
                        black_box(1.0),
                        black_box(0.0),
                    )
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("linear_kernel_cross", size),
            size,
            |bench, _| {
                bench.iter(|| linear_kernel(black_box(&a), black_box(&b)));
            },
        );
    }

    group.finish();
}

/// Benchmark matrix operations with memory analysis
fn bench_matrix_memory_analysis(c: &mut Criterion) {
    let mut group = c.benchmark_group("matrix_memory_analysis");
    let mut suite = BenchmarkSuite::new();

    for size in [50, 100, 200].iter() {
        let a_flat: Vec<f32> = (0..*size * *size).map(|i| i as f32).collect();
        let b_flat: Vec<f32> = (0..*size * *size).map(|i| (i + 1) as f32).collect();

        // Convert to ndarray for compatibility
        let a = sklears_core::prelude::Array2::from_shape_vec((*size, *size), a_flat).unwrap();
        let b = sklears_core::prelude::Array2::from_shape_vec((*size, *size), b_flat).unwrap();

        // Benchmark matrix multiplication with memory analysis
        let mm_result = suite.benchmark(&format!("matrix_multiply_{}", size), 100, || {
            let _ = matrix_multiply_f32_simd(&a, &b);
        });

        group.bench_with_input(
            BenchmarkId::new("matrix_multiply_memory", size),
            size,
            |bench, _| {
                bench.iter(|| matrix_multiply_f32_simd(black_box(&a), black_box(&b)));
            },
        );

        group.bench_with_input(
            BenchmarkId::new("transpose_memory", size),
            size,
            |bench, _| {
                bench.iter(|| transpose_simd(black_box(&a)));
            },
        );

        // Print memory efficiency analysis
        if *size == 100 {
            let ops_per_sec = mm_result.throughput.unwrap_or(0.0);
            let flops = 2.0 * (*size as f64).powi(3); // 2n^3 operations for matrix multiplication
            let gflops = (flops * ops_per_sec) / 1e9;
            println!(
                "\n=== Matrix Multiplication Analysis ({}x{}) ===",
                size, size
            );
            println!("Operations per second: {:.2}", ops_per_sec);
            println!("GFLOPS: {:.2}", gflops);
        }
    }

    group.finish();
}

// TODO: Re-enable when clustering API is updated to match benchmarks
// /// Benchmark clustering operations
// fn bench_clustering_comprehensive(c: &mut Criterion) {
//     let mut group = c.benchmark_group("clustering_comprehensive");

//     for size in [1000, 5000, 10000].iter() {
//         let points = generate_random_matrix(*size, 2); // 2D points
//         let centroids = generate_random_matrix(5, 2); // 5 clusters

//         let points_flat: Vec<f32> = points.iter().flatten().cloned().collect();
//         let centroids_flat: Vec<f32> = centroids.iter().flatten().cloned().collect();

//         group.bench_with_input(
//             BenchmarkId::new("kmeans_distances_comprehensive", size),
//             size,
//             |bench, _| {
//                 bench.iter(|| {
//                     kmeans_distances(
//                         black_box(&points_flat),
//                         black_box(&centroids_flat),
//                         black_box(2),
//                         black_box(5),
//                     )
//                 });
//             },
//         );

//         group.bench_with_input(
//             BenchmarkId::new("update_centroids_comprehensive", size),
//             size,
//             |bench, _| {
//                 let assignments = vec![0, 1, 2, 3, 4]
//                     .iter()
//                     .cycle()
//                     .take(*size)
//                     .cloned()
//                     .collect::<Vec<_>>();
//                 bench.iter(|| {
//                     update_centroids(
//                         black_box(&points_flat),
//                         black_box(&assignments),
//                         black_box(2),
//                         black_box(5),
//                     )
//                 });
//             },
//         );
//     }

//     group.finish();
// }

// TODO: Re-enable when reduction module is implemented
// /// Benchmark reduction operations across platforms
// fn bench_reduction_cross_platform(c: &mut Criterion) {
//     let mut group = c.benchmark_group("reduction_cross_platform");

//     for size in [10000, 100000, 1000000].iter() {
//         let data = generate_random_vector(*size);

//         group.bench_with_input(
//             BenchmarkId::new("parallel_sum_cross", size),
//             size,
//             |bench, _| {
//                 bench.iter(|| parallel_sum(black_box(&data)));
//             },
//         );

//         group.bench_with_input(
//             BenchmarkId::new("parallel_max_cross", size),
//             size,
//             |bench, _| {
//                 bench.iter(|| parallel_max(black_box(&data)));
//             },
//         );

//         group.bench_with_input(
//             BenchmarkId::new("parallel_min_cross", size),
//             size,
//             |bench, _| {
//                 bench.iter(|| parallel_min(black_box(&data)));
//             },
//         );
//     }

//     group.finish();
// }

// TODO: Re-enable when optimization module is implemented
// /// Benchmark optimization operations
// fn bench_optimization_comprehensive(c: &mut Criterion) {
//     let mut group = c.benchmark_group("optimization_comprehensive");

//     for size in [1000, 10000, 100000].iter() {
//         let mut params = generate_random_vector(*size);
//         let gradient = generate_random_vector(*size);
//         let mut velocity = vec![0.0f32; *size];
//         let learning_rate = 0.01f32;
//         let momentum = 0.9f32;

//         group.bench_with_input(
//             BenchmarkId::new("gradient_descent_step_comprehensive", size),
//             size,
//             |bench, _| {
//                 bench.iter(|| {
//                     gradient_descent_step(
//                         black_box(&mut params),
//                         black_box(&gradient),
//                         black_box(learning_rate),
//                     )
//                 });
//             },
//         );

//         group.bench_with_input(
//             BenchmarkId::new("momentum_update_comprehensive", size),
//             size,
//             |bench, _| {
//                 bench.iter(|| {
//                     momentum_update(
//                         black_box(&mut velocity),
//                         black_box(&gradient),
//                         black_box(momentum),
//                     )
//                 });
//             },
//         );

//         // Reset for next iteration
//         params = generate_random_vector(*size);
//         velocity = vec![0.0f32; *size];
//     }

//     group.finish();
// }

/// Performance regression testing
fn bench_regression_testing(c: &mut Criterion) {
    let mut group = c.benchmark_group("regression_testing");
    let mut detector = RegressionDetector::new(5.0); // 5% threshold

    // Set baseline (this would normally be loaded from a file)
    let baseline = vec![sklears_simd::benchmark_framework::BenchmarkResult {
        name: "dot_product_baseline".to_string(),
        duration: Duration::from_millis(10),
        throughput: Some(1000.0),
        simd_width: 8,
        architecture: "AVX2".to_string(),
        iterations: 1000,
    }];
    detector.set_baseline(baseline);

    let data_a = generate_random_vector(10000);
    let data_b = generate_random_vector(10000);

    group.bench_function("dot_product_regression_test", |bench| {
        bench.iter(|| dot_product(black_box(&data_a), black_box(&data_b)));
    });

    // In a real scenario, you would check for regressions here
    // let current_results = vec![current_measurement];
    // let regressions = detector.check_regression(&current_results);

    group.finish();
}

criterion_group!(
    comprehensive_benches,
    bench_cross_platform_vector_ops,
    bench_cross_platform_distances,
    bench_activation_performance_analysis,
    bench_kernel_cross_platform,
    bench_matrix_memory_analysis,
    // bench_clustering_comprehensive,  // TODO: Re-enable when clustering API is updated
    // bench_reduction_cross_platform,  // TODO: Re-enable when reduction module is implemented
    // bench_optimization_comprehensive,  // TODO: Re-enable when optimization module is implemented
    bench_regression_testing
);

criterion_main!(comprehensive_benches);
