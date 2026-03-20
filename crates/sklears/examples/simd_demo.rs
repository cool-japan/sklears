//! Demo of SIMD-optimized distance functions

use scirs2_core::ndarray::Array1;
use sklears_utils::metrics::{euclidean_distance, euclidean_distance_f32, manhattan_distance_f32};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("SIMD Distance Functions Demo");
    println!("============================");

    // Test correctness first
    let a_f32 = Array1::from_vec(vec![1.0f32, 2.0f32, 3.0f32, 4.0f32]);
    let b_f32 = Array1::from_vec(vec![5.0f32, 6.0f32, 7.0f32, 8.0f32]);

    let simd_result = euclidean_distance_f32(&a_f32, &b_f32);
    println!("SIMD f32 Euclidean distance: {}", simd_result);

    let a_f64 = Array1::from_vec(vec![1.0f64, 2.0f64, 3.0f64, 4.0f64]);
    let b_f64 = Array1::from_vec(vec![5.0f64, 6.0f64, 7.0f64, 8.0f64]);

    let scalar_result = euclidean_distance(&a_f64, &b_f64);
    println!("Scalar f64 Euclidean distance: {}", scalar_result);
    println!("Expected: 8.0");
    println!();

    // Performance comparison
    let size = 10000;
    let a_large_f32: Array1<f32> = Array1::from_vec((0..size).map(|i| i as f32).collect());
    let b_large_f32: Array1<f32> = Array1::from_vec((0..size).map(|i| (i + 1) as f32).collect());

    let a_large_f64: Array1<f64> = Array1::from_vec((0..size).map(|i| i as f64).collect());
    let b_large_f64: Array1<f64> = Array1::from_vec((0..size).map(|i| (i + 1) as f64).collect());

    // Warm up
    for _ in 0..10 {
        euclidean_distance_f32(&a_large_f32, &b_large_f32);
        euclidean_distance(&a_large_f64, &b_large_f64);
    }

    // Benchmark SIMD f32
    let iterations = 1000;
    let start = Instant::now();
    for _ in 0..iterations {
        std::hint::black_box(euclidean_distance_f32(
            std::hint::black_box(&a_large_f32),
            std::hint::black_box(&b_large_f32),
        ));
    }
    let simd_time = start.elapsed();

    // Benchmark scalar f64
    let start = Instant::now();
    for _ in 0..iterations {
        std::hint::black_box(euclidean_distance(
            std::hint::black_box(&a_large_f64),
            std::hint::black_box(&b_large_f64),
        ));
    }
    let scalar_time = start.elapsed();

    println!(
        "Performance Comparison (size={}, iterations={}):",
        size, iterations
    );
    println!("SIMD f32 time: {:?}", simd_time);
    println!("Scalar f64 time: {:?}", scalar_time);
    if scalar_time > simd_time {
        println!(
            "SIMD speedup: {:.2}x",
            scalar_time.as_nanos() as f64 / simd_time.as_nanos() as f64
        );
    } else {
        println!(
            "No speedup observed (scalar is faster by {:.2}x)",
            simd_time.as_nanos() as f64 / scalar_time.as_nanos() as f64
        );
    }

    // Test Manhattan distance as well
    println!();
    let manhattan_result = manhattan_distance_f32(&a_f32, &b_f32);
    println!("SIMD f32 Manhattan distance: {}", manhattan_result);
    println!("Expected: 16.0");

    Ok(())
}
