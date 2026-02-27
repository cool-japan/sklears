//! Performance Comparison Tool
//!
//! This example provides a comprehensive performance comparison across
//! different calibration methods, measuring training time, throughput,
//! and scalability.

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::essentials::Normal;
use scirs2_core::random::{thread_rng, Distribution};
use sklears_calibration::{CalibratedClassifierCV, CalibrationMethod};
use sklears_core::traits::Fit;
use std::time::{Duration, Instant};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Calibration Performance Comparison Tool ===\n");

    // Test configurations
    let dataset_sizes = vec![100, 500, 1000, 2000];

    println!("Configuration:");
    println!("  Dataset sizes: {:?}", dataset_sizes);
    println!("  Runs per configuration: 3");
    println!("  CV folds: 3\n");

    // Methods to benchmark
    let methods = vec![
        ("Sigmoid", CalibrationMethod::Sigmoid),
        ("Isotonic", CalibrationMethod::Isotonic),
        ("Temperature", CalibrationMethod::Temperature),
        ("Beta", CalibrationMethod::Beta),
        (
            "Histogram",
            CalibrationMethod::HistogramBinning { n_bins: 10 },
        ),
        (
            "BBQ",
            CalibrationMethod::BBQ {
                min_bins: 5,
                max_bins: 15,
            },
        ),
    ];

    println!("=== Performance Benchmarks ===\n");

    // Results storage
    let mut all_results = Vec::new();

    for &n_samples in &dataset_sizes {
        println!("Dataset size: {} samples", n_samples);
        println!("{}", "-".repeat(70));

        for (name, method) in &methods {
            let mut durations = Vec::new();
            let mut successes = 0;

            // Run multiple times for stable measurements
            for _run in 0..3 {
                let (x, y) = generate_benchmark_data(n_samples);

                let calibrator = CalibratedClassifierCV::new().method(method.clone()).cv(3);

                let start = Instant::now();
                if calibrator.fit(&x, &y).is_ok() {
                    durations.push(start.elapsed());
                    successes += 1;
                }
            }

            if successes > 0 {
                let avg_duration = durations.iter().sum::<Duration>() / successes as u32;
                let throughput = (n_samples as f64) / avg_duration.as_secs_f64();

                println!(
                    "  {:<15} {:>10.2}ms  {:>12.0} samples/sec",
                    name,
                    avg_duration.as_secs_f64() * 1000.0,
                    throughput
                );

                all_results.push(BenchmarkResult {
                    method: name.to_string(),
                    n_samples,
                    avg_time_ms: avg_duration.as_secs_f64() * 1000.0,
                    throughput,
                });
            } else {
                println!("  {:<15} FAILED", name);
            }
        }
        println!();
    }

    // Scalability analysis
    println!("=== Scalability Analysis ===\n");
    println!("Time growth from 100 to 2000 samples:\n");

    for method_name in methods.iter().map(|(name, _)| name) {
        let method_results: Vec<_> = all_results
            .iter()
            .filter(|r| &r.method == method_name)
            .collect();

        if let (Some(smallest), Some(largest)) = (
            method_results.iter().find(|r| r.n_samples == 100),
            method_results.iter().find(|r| r.n_samples == 2000),
        ) {
            let growth_factor = largest.avg_time_ms / smallest.avg_time_ms;
            let theoretical_linear = 2000.0 / 100.0; // 20x

            println!(
                "{:<15} {:>6.2}x growth (linear would be {:.1}x)",
                method_name, growth_factor, theoretical_linear
            );
        }
    }

    // Find fastest method per dataset size
    println!("\n=== Fastest Method per Dataset Size ===\n");

    for &n_samples in &dataset_sizes {
        let mut size_results: Vec<_> = all_results
            .iter()
            .filter(|r| r.n_samples == n_samples)
            .collect();

        size_results.sort_by(|a, b| a.avg_time_ms.partial_cmp(&b.avg_time_ms).unwrap());

        if let Some(fastest) = size_results.first() {
            println!(
                "{:>5} samples: {} ({:.2}ms)",
                n_samples, fastest.method, fastest.avg_time_ms
            );
        }
    }

    // Recommendations
    println!("\n=== Performance Recommendations ===\n");
    println!("For small datasets (< 500 samples):");
    println!("  - Any method is fast enough");
    println!("  - Choose based on calibration quality needs\n");

    println!("For medium datasets (500-2000 samples):");
    println!("  - Sigmoid/Temperature: Fastest");
    println!("  - Isotonic: Good balance of speed and quality");
    println!("  - Beta/BBQ: Slower but potentially better calibration\n");

    println!("For large datasets (> 2000 samples):");
    println!("  - Consider computational budget");
    println!("  - Temperature scaling scales best");
    println!("  - Isotonic may become slow on very large datasets");

    println!("\n=== Benchmark Complete ===");
    Ok(())
}

#[derive(Debug)]
#[allow(dead_code)]
struct BenchmarkResult {
    method: String,
    n_samples: usize,
    avg_time_ms: f64,
    throughput: f64,
}

fn generate_benchmark_data(n_samples: usize) -> (Array2<f64>, Array1<i32>) {
    let mut rng = thread_rng();
    let normal = Normal::new(0.0, 1.0).unwrap();

    let x = Array2::from_shape_fn((n_samples, 3), |_| normal.sample(&mut rng));
    let y = Array1::from_shape_fn(n_samples, |i| if i < n_samples / 2 { 0 } else { 1 });

    (x, y)
}
