//! Bayesian Calibration and Uncertainty Quantification Example
//!
//! This example demonstrates advanced Bayesian calibration methods and
//! uncertainty quantification techniques for robust probability estimates.

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::essentials::Normal;
use scirs2_core::random::thread_rng;
use sklears_calibration::{CalibratedClassifierCV, CalibrationMethod};
use sklears_core::traits::Fit;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Bayesian Calibration & Uncertainty Quantification ===\n");

    // Generate dataset with inherent uncertainty
    let n_samples = 800;
    let (x, y) = generate_noisy_data(n_samples);

    println!("Dataset: {} samples with added noise", n_samples);
    println!("Features: {} dimensions", x.ncols());
    println!("Classes: 2 (binary classification)\n");

    // Bayesian calibration methods
    let bayesian_methods = vec![
        (
            "Bayesian Model Averaging",
            CalibrationMethod::BayesianModelAveraging { n_models: 5 },
            "Combines multiple calibration models with Bayesian weighting",
        ),
        (
            "Variational Inference",
            CalibrationMethod::VariationalInference {
                learning_rate: 0.01,
                n_samples: 100,
                max_iter: 50,
            },
            "Approximate Bayesian inference for calibration parameters",
        ),
        (
            "MCMC Calibration",
            CalibrationMethod::MCMC {
                n_samples: 200,
                burn_in: 50,
                step_size: 0.1,
            },
            "Full Bayesian posterior inference via MCMC sampling",
        ),
        (
            "Hierarchical Bayesian",
            CalibrationMethod::HierarchicalBayesian,
            "Multi-level Bayesian model for grouped data",
        ),
        (
            "Dirichlet Process",
            CalibrationMethod::DirichletProcess {
                concentration: 1.0,
                max_clusters: 10,
            },
            "Non-parametric Bayesian clustering for calibration",
        ),
    ];

    println!("=== Testing Bayesian Calibration Methods ===\n");

    let mut successful_methods = Vec::new();

    for (name, method, description) in bayesian_methods {
        println!("Method: {}", name);
        println!("  Description: {}", description);

        let start = std::time::Instant::now();
        let calibrator = CalibratedClassifierCV::new().method(method).cv(3);

        match calibrator.fit(&x, &y) {
            Ok(_trained) => {
                let elapsed = start.elapsed();
                println!(
                    "  ✓ Success - Time: {:.2}ms",
                    elapsed.as_secs_f64() * 1000.0
                );
                successful_methods.push(name);
            }
            Err(e) => {
                println!("  ⚠ Error: {}", e);
            }
        }
        println!();
    }

    // Advanced uncertainty quantification methods
    println!("=== Uncertainty Quantification Methods ===\n");

    let uq_methods = vec![
        (
            "Conformal Prediction (Split)",
            CalibrationMethod::ConformalSplit { alpha: 0.05 },
            "Distribution-free prediction intervals with coverage guarantees",
        ),
        (
            "Conformal Cross-Validation",
            CalibrationMethod::ConformalCross {
                alpha: 0.05,
                n_folds: 5,
            },
            "Cross-validated conformal prediction for better efficiency",
        ),
        (
            "Gaussian Process Calibration",
            CalibrationMethod::GaussianProcess,
            "Probabilistic calibration with uncertainty estimates",
        ),
        (
            "Variational GP",
            CalibrationMethod::VariationalGP { n_inducing: 50 },
            "Scalable GP calibration with inducing points",
        ),
    ];

    for (name, method, description) in uq_methods {
        println!("Method: {}", name);
        println!("  Description: {}", description);

        let start = std::time::Instant::now();
        let calibrator = CalibratedClassifierCV::new().method(method).cv(2);

        match calibrator.fit(&x, &y) {
            Ok(_trained) => {
                let elapsed = start.elapsed();
                println!(
                    "  ✓ Success - Time: {:.2}ms",
                    elapsed.as_secs_f64() * 1000.0
                );
            }
            Err(e) => {
                println!("  ⚠ Error: {}", e);
            }
        }
        println!();
    }

    // Method selection guidance
    println!("=== Method Selection Guide ===\n");
    println!("Bayesian Methods:");
    println!("  - Use when you need principled uncertainty quantification");
    println!("  - Bayesian Model Averaging: Good balance of accuracy and speed");
    println!("  - MCMC: Most accurate but slowest, use for critical applications");
    println!("  - Variational Inference: Faster approximation to full Bayesian");
    println!("  - Dirichlet Process: Non-parametric, adapts to data complexity\n");

    println!("Uncertainty Quantification:");
    println!("  - Conformal Prediction: Provides guaranteed coverage");
    println!("  - GP methods: Natural uncertainty estimates with smooth calibration");
    println!("  - Use conformal for finite-sample guarantees");
    println!("  - Use GP for smooth probability surfaces\n");

    println!(
        "Successfully trained methods: {}/{}",
        successful_methods.len(),
        9
    );

    println!("\n=== Example Complete ===");
    Ok(())
}

/// Generate noisy binary classification data with overlapping classes
fn generate_noisy_data(n_samples: usize) -> (Array2<f64>, Array1<i32>) {
    let mut rng = thread_rng();
    let normal = Normal::new(0.0, 1.0).unwrap();

    // Create overlapping classes with significant noise
    let mut x = Array2::zeros((n_samples, 3));
    let mut y = Array1::zeros(n_samples);

    for i in 0..n_samples {
        let true_class = if i < n_samples / 2 { 0 } else { 1 };
        y[i] = true_class;

        // Class-dependent features with high overlap
        let mean_shift = if true_class == 0 { -0.5 } else { 0.5 };

        x[[i, 0]] = mean_shift + rng.sample(&normal) * 1.5;
        x[[i, 1]] = mean_shift * 0.5 + rng.sample(&normal) * 1.5;
        x[[i, 2]] = rng.sample(&normal) * 2.0; // Pure noise feature
    }

    (x, y)
}
