//! Quickstart Example for Calibration
//!
//! This example demonstrates the quickest way to get started with
//! probability calibration in sklears.

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::essentials::Normal;
use scirs2_core::random::{thread_rng, Distribution};
use sklears_calibration::{CalibratedClassifierCV, CalibrationMethod};
use sklears_core::traits::Fit;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Sklears Calibration Quickstart ===\n");

    // Generate simple binary classification dataset
    let (x, y) = generate_data(500);

    println!("Created dataset with {} samples", x.nrows());
    println!("Features: {} dimensions\n", x.ncols());

    // Create a calibrated classifier with sigmoid (Platt scaling)
    println!("Training calibrated classifier with Sigmoid (Platt) method...");
    let calibrator = CalibratedClassifierCV::new()
        .method(CalibrationMethod::Sigmoid)
        .cv(3); // 3-fold cross-validation

    // Fit the calibrator
    let trained = calibrator.fit(&x, &y)?;

    println!("✓ Calibrator trained successfully!");
    println!("  Number of classes: {}", trained.classes().len());
    println!("  Cross-validation folds: 3\n");

    // Try different calibration methods
    println!("Comparing different calibration methods:\n");

    let methods = vec![
        ("Sigmoid", CalibrationMethod::Sigmoid),
        ("Isotonic", CalibrationMethod::Isotonic),
        ("Temperature", CalibrationMethod::Temperature),
        ("Beta", CalibrationMethod::Beta),
    ];

    for (name, method) in methods {
        let cal = CalibratedClassifierCV::new().method(method);
        let result = cal.fit(&x, &y);

        match result {
            Ok(fitted) => println!("✓ {} calibration: {} classes", name, fitted.classes().len()),
            Err(e) => println!("✗ {} calibration failed: {}", name, e),
        }
    }

    println!("\n=== Quickstart Complete ===");
    println!("\nNext steps:");
    println!("  - See basic_calibration.rs for more detailed usage");
    println!("  - Check the documentation: cargo doc --open");

    Ok(())
}

fn generate_data(n: usize) -> (Array2<f64>, Array1<i32>) {
    let mut rng = thread_rng();
    let normal = Normal::new(0.0, 1.0).unwrap();

    let x = Array2::from_shape_fn((n, 2), |_| normal.sample(&mut rng));
    let y = Array1::from_shape_fn(n, |i| if i < n / 2 { 0 } else { 1 });

    (x, y)
}
