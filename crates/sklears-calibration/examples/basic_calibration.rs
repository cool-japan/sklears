//! Basic Calibration Example
//!
//! This example demonstrates how to use various calibration methods
//! to improve probability estimates from a classifier.

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::essentials::Normal;
use scirs2_core::random::thread_rng;
use sklears_calibration::{CalibratedClassifierCV, CalibrationMethod};
use sklears_core::traits::Fit;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Basic Calibration Example ===\n");

    // Generate synthetic classification data
    println!("Generating synthetic classification data...");
    let n_samples = 1000;
    let (x, y) = generate_synthetic_data(n_samples);

    println!("Dataset: {} samples", n_samples);
    println!("Features shape: {:?}", x.dim());
    println!("Labels shape: {:?}\n", y.dim());

    // Note: In a real scenario, you would get probabilities from a trained classifier
    // For this example, we'll demonstrate the calibration API directly
    println!("Note: This example demonstrates the calibration API.");

    // Test different calibration methods
    let methods = vec![
        ("Sigmoid (Platt Scaling)", CalibrationMethod::Sigmoid),
        ("Isotonic Regression", CalibrationMethod::Isotonic),
        ("Temperature Scaling", CalibrationMethod::Temperature),
        (
            "Histogram Binning",
            CalibrationMethod::HistogramBinning { n_bins: 10 },
        ),
        (
            "Bayesian Binning Quantiles",
            CalibrationMethod::BBQ {
                min_bins: 5,
                max_bins: 15,
            },
        ),
    ];

    println!("\n=== Testing Calibration Methods ===\n");

    for (name, method) in methods {
        println!("Testing: {}", name);

        // Create calibrated classifier using builder pattern
        let calibrator = CalibratedClassifierCV::new().method(method).cv(3);

        // Fit calibrator on data (it will use internal probability estimates)
        let fitted_calibrator = calibrator.fit(&x, &y)?;

        println!("  Calibrator trained successfully");
        println!(
            "  Number of classes: {:?}\n",
            fitted_calibrator.classes().len()
        );
    }

    println!("\n=== Example Complete ===");
    Ok(())
}

/// Generate synthetic classification data
fn generate_synthetic_data(n_samples: usize) -> (Array2<f64>, Array1<i32>) {
    let mut rng = thread_rng();
    let normal = Normal::new(0.0, 1.0).unwrap();

    // Generate features
    let x = Array2::from_shape_fn((n_samples, 2), |_| rng.sample(&normal));

    // Generate true labels (binary classification)
    let y = Array1::from_shape_fn(n_samples, |i| {
        if (i as f64) < (n_samples as f64 * 0.6) {
            0
        } else {
            1
        }
    });

    (x, y)
}
