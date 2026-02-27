//! Multiclass Calibration Example
//!
//! This example demonstrates advanced calibration methods specifically
//! designed for multiclass classification problems (3+ classes).

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::essentials::Normal;
use scirs2_core::random::{thread_rng, Distribution};
use sklears_calibration::{CalibratedClassifierCV, CalibrationMethod};
use sklears_core::traits::Fit;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Multiclass Calibration Example ===\n");

    // Generate synthetic 4-class classification problem
    let n_samples = 1000;
    let n_classes = 4;
    let (x, y) = generate_multiclass_data(n_samples, n_classes);

    println!("Dataset created:");
    println!("  Samples: {}", n_samples);
    println!("  Classes: {}", n_classes);
    println!("  Features: {}\n", x.ncols());

    // Count samples per class
    println!("Class distribution:");
    for class in 0..n_classes {
        let count = y.iter().filter(|&&label| label == class as i32).count();
        println!(
            "  Class {}: {} samples ({:.1}%)",
            class,
            count,
            (count as f64 / n_samples as f64) * 100.0
        );
    }
    println!();

    // Test multiclass-specific calibration methods
    let methods = vec![
        (
            "Multiclass Temperature Scaling",
            CalibrationMethod::MulticlassTemperature,
            "Single temperature parameter for all classes",
        ),
        (
            "Matrix Scaling",
            CalibrationMethod::MatrixScaling,
            "Full matrix transformation of logits",
        ),
        (
            "Dirichlet Calibration",
            CalibrationMethod::Dirichlet { concentration: 1.0 },
            "Principled Bayesian multiclass calibration",
        ),
        (
            "Isotonic (One-vs-Rest)",
            CalibrationMethod::Isotonic,
            "Separate isotonic regression per class",
        ),
        (
            "Sigmoid (One-vs-Rest)",
            CalibrationMethod::Sigmoid,
            "Separate Platt scaling per class",
        ),
    ];

    println!("=== Testing Multiclass Calibration Methods ===\n");

    let mut results = Vec::new();

    for (name, method, description) in methods {
        println!("Method: {}", name);
        println!("  Description: {}", description);

        let start = std::time::Instant::now();

        // Create and train calibrator
        let calibrator = CalibratedClassifierCV::new()
            .method(method)
            .cv(3)
            .ensemble(true);

        match calibrator.fit(&x, &y) {
            Ok(trained) => {
                let elapsed = start.elapsed();
                let n_classes = trained.classes().len();

                println!("  ✓ Training successful");
                println!("    Classes detected: {}", n_classes);
                println!("    Training time: {:.2}ms", elapsed.as_secs_f64() * 1000.0);

                results.push((name, n_classes, elapsed));
            }
            Err(e) => {
                println!("  ✗ Training failed: {}", e);
            }
        }
        println!();
    }

    // Summary comparison
    println!("=== Performance Summary ===\n");
    println!("{:<35} {:>10} {:>15}", "Method", "Classes", "Time (ms)");
    println!("{}", "-".repeat(62));

    for (name, n_classes, elapsed) in results {
        println!(
            "{:<35} {:>10} {:>15.2}",
            name,
            n_classes,
            elapsed.as_secs_f64() * 1000.0
        );
    }

    println!("\n=== Multiclass Calibration Tips ===\n");
    println!("1. Temperature Scaling: Fast, works well for neural networks");
    println!("2. Matrix Scaling: More flexible than temperature, learns full transformation");
    println!("3. Dirichlet: Principled Bayesian approach, good for well-specified models");
    println!("4. One-vs-Rest methods: Work with any binary calibrator");
    println!();
    println!("Choose based on:");
    println!("  - Dataset size (larger = more complex methods possible)");
    println!("  - Model type (neural nets benefit from scaling methods)");
    println!("  - Calibration quality requirements");
    println!("  - Computational budget");

    println!("\n=== Example Complete ===");
    Ok(())
}

/// Generate synthetic multiclass classification data
fn generate_multiclass_data(n_samples: usize, n_classes: usize) -> (Array2<f64>, Array1<i32>) {
    let mut rng = thread_rng();
    let normal = Normal::new(0.0, 1.0).unwrap();

    // Generate features with some structure
    // Classes will have different mean centers
    let mut x = Array2::zeros((n_samples, 2));
    let mut y = Array1::zeros(n_samples);

    let samples_per_class = n_samples / n_classes;

    for class in 0..n_classes {
        let start_idx = class * samples_per_class;
        let end_idx = if class == n_classes - 1 {
            n_samples
        } else {
            (class + 1) * samples_per_class
        };

        // Each class has a different center
        let center_x = ((class as f64) - (n_classes as f64 / 2.0)) * 2.0;
        let center_y = ((class % 2) as f64) * 2.0 - 1.0;

        for i in start_idx..end_idx {
            x[[i, 0]] = center_x + normal.sample(&mut rng) * 1.5;
            x[[i, 1]] = center_y + normal.sample(&mut rng) * 1.5;
            y[i] = class as i32;
        }
    }

    (x, y)
}
