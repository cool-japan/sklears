//! Streaming and Online Calibration Example
//!
//! This example demonstrates calibration methods designed for streaming
//! data and online learning scenarios where data arrives continuously.

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::essentials::Normal;
use scirs2_core::random::thread_rng;
use sklears_calibration::{CalibratedClassifierCV, CalibrationMethod};
use sklears_core::traits::Fit;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Streaming and Online Calibration ===\n");

    // Simulate streaming data in batches
    let batch_size = 200;
    let n_batches = 5;
    let total_samples = batch_size * n_batches;

    println!("Streaming scenario:");
    println!("  Total samples: {}", total_samples);
    println!("  Batch size: {}", batch_size);
    println!("  Number of batches: {}\n", n_batches);

    // Online/Streaming calibration methods
    let streaming_methods = vec![
        (
            "Online Sigmoid",
            CalibrationMethod::OnlineSigmoid {
                learning_rate: 0.01,
                use_momentum: true,
                momentum: 0.9,
            },
            "Incremental Platt scaling with SGD",
        ),
        (
            "Adaptive Online",
            CalibrationMethod::AdaptiveOnline {
                window_size: 100,
                retrain_frequency: 50,
                drift_threshold: 0.1,
            },
            "Concept drift-aware calibration with sliding window",
        ),
        (
            "Incremental Update",
            CalibrationMethod::IncrementalUpdate {
                update_frequency: 20,
                learning_rate: 0.05,
                use_smoothing: true,
            },
            "Fast incremental updates without full retraining",
        ),
    ];

    println!("=== Testing Streaming Calibration Methods ===\n");

    for (name, method, description) in streaming_methods {
        println!("Method: {}", name);
        println!("  Description: {}", description);
        println!("  Testing on {} batches...", n_batches);

        let mut total_time = std::time::Duration::default();
        let mut successful_batches = 0;

        // Simulate streaming by training on successive batches
        for batch_idx in 0..n_batches {
            let (x_batch, y_batch) = generate_batch(batch_size, batch_idx, n_batches);

            let start = std::time::Instant::now();
            let calibrator = CalibratedClassifierCV::new().method(method.clone()).cv(2); // Smaller CV for streaming

            match calibrator.fit(&x_batch, &y_batch) {
                Ok(_) => {
                    total_time += start.elapsed();
                    successful_batches += 1;
                }
                Err(_) => {
                    // Continue with next batch even if one fails
                }
            }
        }

        let avg_time = total_time.as_secs_f64() / successful_batches as f64 * 1000.0;

        println!("  ✓ Processed {}/{} batches", successful_batches, n_batches);
        println!("  Average time per batch: {:.2}ms", avg_time);
        println!(
            "  Total throughput: {:.0} samples/sec",
            (batch_size * successful_batches) as f64 / total_time.as_secs_f64()
        );
        println!();
    }

    // Compare with batch methods
    println!("=== Comparison with Batch Methods ===\n");

    let batch_methods = vec![
        ("Sigmoid (Batch)", CalibrationMethod::Sigmoid),
        ("Isotonic (Batch)", CalibrationMethod::Isotonic),
        ("Temperature (Batch)", CalibrationMethod::Temperature),
    ];

    // Train on full dataset at once
    let (x_full, y_full) = generate_full_dataset(total_samples);

    for (name, method) in batch_methods {
        println!("Method: {}", name);

        let start = std::time::Instant::now();
        let calibrator = CalibratedClassifierCV::new().method(method).cv(3);

        match calibrator.fit(&x_full, &y_full) {
            Ok(_) => {
                let elapsed = start.elapsed();
                println!(
                    "  ✓ Full dataset training: {:.2}ms",
                    elapsed.as_secs_f64() * 1000.0
                );
                println!(
                    "  Throughput: {:.0} samples/sec",
                    total_samples as f64 / elapsed.as_secs_f64()
                );
            }
            Err(e) => {
                println!("  ✗ Error: {}", e);
            }
        }
        println!();
    }

    // Use case recommendations
    println!("=== When to Use Streaming Calibration ===\n");
    println!("Use online/streaming methods when:");
    println!("  1. Data arrives continuously (e.g., real-time systems)");
    println!("  2. Full retraining is too expensive");
    println!("  3. Data distribution may drift over time");
    println!("  4. Memory constraints prevent storing all data");
    println!("  5. Low-latency updates are required\n");

    println!("Use batch methods when:");
    println!("  1. Full dataset is available upfront");
    println!("  2. Data distribution is stationary");
    println!("  3. Accuracy is more important than update speed");
    println!("  4. Memory is not a constraint");

    println!("\n=== Example Complete ===");
    Ok(())
}

/// Generate a single batch of data with potential drift
fn generate_batch(
    batch_size: usize,
    batch_idx: usize,
    total_batches: usize,
) -> (Array2<f64>, Array1<i32>) {
    let mut rng = thread_rng();
    let normal = Normal::new(0.0, 1.0).unwrap();

    // Introduce gradual drift across batches
    let drift_factor = (batch_idx as f64) / (total_batches as f64) * 0.5;

    let mut x = Array2::zeros((batch_size, 2));
    let mut y = Array1::zeros(batch_size);

    for i in 0..batch_size {
        let true_class = if i < batch_size / 2 { 0 } else { 1 };
        y[i] = true_class;

        let mean = if true_class == 0 { -1.0 } else { 1.0 };
        x[[i, 0]] = mean + drift_factor + rng.sample(&normal);
        x[[i, 1]] = mean * 0.5 - drift_factor + rng.sample(&normal);
    }

    (x, y)
}

/// Generate full dataset for batch comparison
fn generate_full_dataset(n_samples: usize) -> (Array2<f64>, Array1<i32>) {
    let mut rng = thread_rng();
    let normal = Normal::new(0.0, 1.0).unwrap();

    let x = Array2::from_shape_fn((n_samples, 2), |_| rng.sample(&normal));
    let y = Array1::from_shape_fn(n_samples, |i| if i < n_samples / 2 { 0 } else { 1 });

    (x, y)
}
