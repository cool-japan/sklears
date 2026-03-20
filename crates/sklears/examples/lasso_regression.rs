//! Example: Lasso Regression with feature selection
//!
//! This example demonstrates using Lasso regression for sparse linear models
//! and automatic feature selection.

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::thread_rng;
use sklears::linear::LinearRegression;
use sklears::metrics::regression::r2_score;
use sklears::prelude::*;

fn main() -> Result<()> {
    println!("=== Lasso Regression Example ===\n");

    // Generate synthetic data with sparse ground truth
    let n_samples = 100;
    let n_features = 20;
    let n_informative = 5;

    // Create random feature matrix
    let mut rng = thread_rng();
    let mut x = Array2::zeros((n_samples, n_features));
    for i in 0..n_samples {
        for j in 0..n_features {
            x[[i, j]] = rng.random_range(-2.0..2.0);
        }
    }

    // Create sparse ground truth coefficients
    let mut true_coef = Array1::zeros(n_features);
    for i in 0..n_informative {
        true_coef[i] = rng.random_range(1.0..5.0) * if rng.gen_bool(0.5) { 1.0 } else { -1.0 };
    }

    println!("True coefficients (first 10):");
    for i in 0..10 {
        println!("  Feature {}: {:.3}", i, true_coef[i]);
    }
    println!();

    // Generate target with noise
    let y = x.dot(&true_coef) + Array1::from_shape_fn(n_samples, |_| rng.random_range(-0.5..0.5));

    // Fit models with different alpha values
    let alphas = [0.001, 0.01, 0.1, 0.5, 1.0];

    for &alpha in &alphas {
        println!("Alpha = {}", alpha);

        // Fit Lasso model
        let model = LinearRegression::lasso(alpha)
            .fit_intercept(true)
            .fit(&x, &y)?;

        let coef = model.coef();

        // Count non-zero coefficients
        let n_nonzero = coef.iter().filter(|&&c| c.abs() > 1e-6).count();
        println!("  Non-zero coefficients: {}/{}", n_nonzero, n_features);

        // Show first few coefficients
        println!("  Coefficients (first 10):");
        for i in 0..10 {
            if coef[i].abs() > 1e-6 {
                println!("    Feature {}: {:.3}", i, coef[i]);
            }
        }

        // Calculate R² score
        let y_pred = model.predict(&x)?;
        let score = r2_score(&y, &y_pred)?;
        println!("  R² score: {:.4}", score);
        println!();
    }

    // Compare with ElasticNet
    println!("=== ElasticNet Comparison ===\n");

    let alpha = 0.1;
    let l1_ratios = [0.1, 0.5, 0.9];

    for &l1_ratio in &l1_ratios {
        println!("ElasticNet: alpha={}, l1_ratio={}", alpha, l1_ratio);

        let model = LinearRegression::elastic_net(alpha, l1_ratio)
            .fit_intercept(true)
            .fit(&x, &y)?;

        let coef = model.coef();
        let n_nonzero = coef.iter().filter(|&&c| c.abs() > 1e-6).count();

        println!("  Non-zero coefficients: {}/{}", n_nonzero, n_features);

        let y_pred = model.predict(&x)?;
        let score = r2_score(&y, &y_pred)?;
        println!("  R² score: {:.4}", score);
        println!();
    }

    Ok(())
}
