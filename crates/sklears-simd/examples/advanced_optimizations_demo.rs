//! Advanced SIMD Optimizations Demo
//!
//! This example showcases advanced SIMD optimization techniques including:
//! - Cache-aware matrix multiplication
//! - Vectorized operations with manual optimizations
//! - Memory-efficient algorithms
//! - Performance comparisons

use sklears_simd::advanced_optimizations::{AdvancedSimdOptimizer, CacheAwareSort, ReductionOp};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Advanced SIMD Optimizations Demo");
    println!("===================================\n");

    let optimizer = AdvancedSimdOptimizer::new();

    // Demo 1: Vectorized dot product
    println!("📊 Demo 1: Vectorized Dot Product");
    demo_vectorized_dot_product(&optimizer)?;

    // Demo 2: Advanced reductions
    println!("\n📊 Demo 2: Advanced Vectorized Reductions");
    demo_vectorized_reductions(&optimizer)?;

    // Demo 3: Cache-aware matrix multiplication
    println!("\n📊 Demo 3: Cache-Aware Matrix Multiplication");
    demo_cache_aware_matrix_multiply(&optimizer)?;

    // Demo 4: Cache-aware sorting
    println!("\n📊 Demo 4: Cache-Aware Sorting");
    demo_cache_aware_sorting()?;

    // Demo 5: Performance benchmarks
    println!("\n📊 Demo 5: Performance Benchmarks");
    demo_performance_benchmarks(&optimizer)?;

    println!("\n✨ All demos completed successfully!");
    println!("🎯 Key Benefits Demonstrated:");
    println!("   ✅ Vectorized operations with SIMD intrinsics");
    println!("   ✅ Cache-aware algorithms for better memory usage");
    println!("   ✅ Manual optimizations for maximum performance");
    println!("   ✅ Automatic fallback to scalar implementations");
    println!("   ✅ Platform-specific optimizations");

    Ok(())
}

fn demo_vectorized_dot_product(
    optimizer: &AdvancedSimdOptimizer,
) -> Result<(), Box<dyn std::error::Error>> {
    let size = 10000;
    let a: Vec<f32> = (0..size).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..size).map(|i| (i + 1) as f32).collect();

    println!("  Vector size: {}", size);

    // Vectorized dot product
    let start = Instant::now();
    let result = optimizer.vectorized_dot_product(&a, &b)?;
    let vectorized_time = start.elapsed();

    // Naive scalar implementation for comparison
    let start = Instant::now();
    let scalar_result: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let scalar_time = start.elapsed();

    println!("  Vectorized result: {:.2}", result);
    println!("  Scalar result: {:.2}", scalar_result);
    println!("  Vectorized time: {:.2?}", vectorized_time);
    println!("  Scalar time: {:.2?}", scalar_time);

    if scalar_time > vectorized_time {
        let speedup = scalar_time.as_secs_f64() / vectorized_time.as_secs_f64();
        println!("  🚀 Speedup: {:.2}x", speedup);
    }

    assert!(
        (result - scalar_result).abs() < 1e6,
        "Results should match within reasonable tolerance"
    );

    Ok(())
}

fn demo_vectorized_reductions(
    optimizer: &AdvancedSimdOptimizer,
) -> Result<(), Box<dyn std::error::Error>> {
    let data: Vec<f32> = (1..=1000).map(|i| i as f32).collect();

    println!("  Data size: {}", data.len());

    // Test all reduction operations
    let sum = optimizer.vectorized_reduction(&data, ReductionOp::Sum)?;
    let max = optimizer.vectorized_reduction(&data, ReductionOp::Max)?;
    let min = optimizer.vectorized_reduction(&data, ReductionOp::Min)?;
    let mean = optimizer.vectorized_reduction(&data, ReductionOp::Mean)?;

    println!("  Sum: {:.2}", sum);
    println!("  Max: {:.2}", max);
    println!("  Min: {:.2}", min);
    println!("  Mean: {:.2}", mean);

    // Verify against standard library
    let expected_sum: f32 = data.iter().sum();
    let expected_max = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let expected_min = data.iter().copied().fold(f32::INFINITY, f32::min);
    let expected_mean = expected_sum / data.len() as f32;

    assert!((sum - expected_sum).abs() < 1e-3);
    assert!((max - expected_max).abs() < 1e-3);
    assert!((min - expected_min).abs() < 1e-3);
    assert!((mean - expected_mean).abs() < 1e-3);

    println!("  ✅ All reductions verified!");

    Ok(())
}

fn demo_cache_aware_matrix_multiply(
    optimizer: &AdvancedSimdOptimizer,
) -> Result<(), Box<dyn std::error::Error>> {
    let m = 128;
    let n = 128;
    let k = 128;

    println!("  Matrix dimensions: {}x{} × {}x{}", m, k, k, n);

    // Generate test matrices
    let a: Vec<f32> = (0..m * k).map(|i| (i % 100) as f32).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (i % 100) as f32).collect();
    let mut c = vec![0.0f32; m * n];

    // Cache-aware matrix multiplication
    let start = Instant::now();
    optimizer.cache_aware_matrix_multiply(&a, &b, &mut c, m, n, k)?;
    let cache_aware_time = start.elapsed();

    // Naive matrix multiplication for comparison
    let mut c_naive = vec![0.0f32; m * n];
    let start = Instant::now();
    naive_matrix_multiply(&a, &b, &mut c_naive, m, n, k);
    let naive_time = start.elapsed();

    println!("  Cache-aware time: {:.2?}", cache_aware_time);
    println!("  Naive time: {:.2?}", naive_time);

    if naive_time > cache_aware_time {
        let speedup = naive_time.as_secs_f64() / cache_aware_time.as_secs_f64();
        println!("  🚀 Speedup: {:.2}x", speedup);
    }

    // Verify results match (allow small floating point differences)
    let max_diff = c
        .iter()
        .zip(c_naive.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);

    assert!(
        max_diff < 1e-3,
        "Matrix multiplication results should match"
    );
    println!("  ✅ Results verified (max diff: {:.2e})", max_diff);

    Ok(())
}

fn demo_cache_aware_sorting() -> Result<(), Box<dyn std::error::Error>> {
    let size = 10000;
    let mut data: Vec<f32> = (0..size).map(|i| (size - i) as f32).collect(); // Reverse sorted
    let mut data_std = data.clone();

    println!("  Data size: {}", size);
    println!("  Initial order: reverse sorted");

    // Cache-aware sort
    let start = Instant::now();
    CacheAwareSort::vectorized_merge_sort(&mut data);
    let cache_aware_time = start.elapsed();

    // Standard library sort
    let start = Instant::now();
    data_std.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let std_time = start.elapsed();

    println!("  Cache-aware sort time: {:.2?}", cache_aware_time);
    println!("  Standard sort time: {:.2?}", std_time);

    // Verify sorting correctness
    assert_eq!(data, data_std, "Sorted arrays should match");
    assert!(
        data.windows(2).all(|w| w[0] <= w[1]),
        "Array should be sorted"
    );

    println!("  ✅ Sorting verified!");

    Ok(())
}

fn demo_performance_benchmarks(
    optimizer: &AdvancedSimdOptimizer,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("  Running comprehensive performance benchmarks...");

    let sizes = vec![1000, 5000, 10000, 50000];

    for size in sizes {
        println!("\n  Size: {}", size);

        // Generate test data
        let a: Vec<f32> = (0..size).map(|i| (i % 1000) as f32).collect();
        let b: Vec<f32> = (0..size).map(|i| ((i + 1) % 1000) as f32).collect();

        // Benchmark dot product
        let start = Instant::now();
        let _ = optimizer.vectorized_dot_product(&a, &b)?;
        let dot_time = start.elapsed();

        // Benchmark reduction
        let start = Instant::now();
        let _ = optimizer.vectorized_reduction(&a, ReductionOp::Sum)?;
        let sum_time = start.elapsed();

        println!("    Dot product: {:.2?}", dot_time);
        println!("    Sum reduction: {:.2?}", sum_time);

        // Calculate throughput
        let dot_throughput = (size as f64 * 2.0) / dot_time.as_secs_f64() / 1e9; // GFLOPS
        let sum_throughput = size as f64 / sum_time.as_secs_f64() / 1e9; // GElements/s

        println!("    Dot product throughput: {:.2} GFLOPS", dot_throughput);
        println!("    Sum throughput: {:.2} GElements/s", sum_throughput);
    }

    Ok(())
}

fn naive_matrix_multiply(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for kk in 0..k {
                sum += a[i * k + kk] * b[kk * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}
