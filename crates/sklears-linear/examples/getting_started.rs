//! Getting Started with Sklears Linear Models
//!
//! This example demonstrates the basic usage of linear models in sklears-linear
//! with simple synthetic data to show core functionality.

use scirs2_core::essentials::Normal;
use scirs2_core::ndarray::{s, Array1, Array2};
use scirs2_core::random::{seeded_rng, Distribution};
use sklears_core::traits::{Fit, Predict};
use sklears_linear::LinearRegression;
use std::time::Instant;

/// Generate simple synthetic regression data
#[allow(non_snake_case)]
fn generate_synthetic_data(n_samples: usize, n_features: usize) -> (Array2<f64>, Array1<f64>) {
    let mut rng = seeded_rng(42);
    let normal = Normal::new(0.0, 1.0).unwrap();

    // Generate random feature matrix
    let X = Array2::from_shape_fn((n_samples, n_features), |_| normal.sample(&mut rng));

    // Generate true coefficients (some zero for sparsity)
    let true_coefs: Array1<f64> = (0..n_features)
        .map(|i| {
            if i % 3 == 0 {
                rng.random_range(-2.0..2.0)
            } else {
                0.0
            }
        })
        .collect::<Vec<_>>()
        .into();

    // Generate targets with some noise
    let y_clean = X.dot(&true_coefs);
    let noise: Array1<f64> = Array1::from_shape_fn(n_samples, |_| normal.sample(&mut rng)) * 0.1;
    let y = y_clean + noise;

    (X, y)
}

/// Calculate mean squared error
fn mse(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> f64 {
    let diff = y_true - y_pred;
    diff.mapv(|x| x * x).mean().unwrap()
}

/// Calculate R² score
fn r2_score(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> f64 {
    let y_mean = y_true.mean().unwrap();
    let ss_tot: f64 = y_true.iter().map(|&y| (y - y_mean).powi(2)).sum();
    let ss_res: f64 = y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(&yt, &yp)| (yt - yp).powi(2))
        .sum();

    1.0 - (ss_res / ss_tot)
}

#[allow(non_snake_case)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Getting Started with Sklears Linear Models");
    println!("=============================================");
    println!();

    // Generate synthetic dataset
    let n_samples = 100;
    let n_features = 5;

    println!("📊 Generating synthetic dataset...");
    let (X, y) = generate_synthetic_data(n_samples, n_features);
    println!("Dataset: {} samples, {} features", n_samples, n_features);
    println!("Feature matrix shape: {:?}", X.dim());
    println!("Target vector shape: {:?}", y.dim());
    println!();

    // Split data into train/test (simple split)
    let split_idx = (n_samples as f64 * 0.8) as usize;
    let X_train = X.slice(s![..split_idx, ..]).to_owned();
    let X_test = X.slice(s![split_idx.., ..]).to_owned();
    let y_train = y.slice(s![..split_idx]).to_owned();
    let y_test = y.slice(s![split_idx..]).to_owned();

    println!("📈 Train set: {} samples", X_train.nrows());
    println!("🧪 Test set: {} samples", X_test.nrows());
    println!();

    // 1. Basic Linear Regression
    println!("🔵 Linear Regression Example");
    println!("===========================");

    let start = Instant::now();

    // Create and fit the model
    let model = LinearRegression::new();
    println!("✓ Created LinearRegression model");

    let fitted_model = model.fit(&X_train, &y_train)?;
    println!("✓ Fitted model to training data");

    // Make predictions
    let y_pred_train = fitted_model.predict(&X_train)?;
    let y_pred_test = fitted_model.predict(&X_test)?;
    println!("✓ Made predictions on train and test sets");

    let training_time = start.elapsed();

    // Evaluate performance
    let train_mse = mse(&y_train, &y_pred_train);
    let test_mse = mse(&y_test, &y_pred_test);
    let train_r2 = r2_score(&y_train, &y_pred_train);
    let test_r2 = r2_score(&y_test, &y_pred_test);

    println!();
    println!("📊 Results:");
    println!("Training time: {:?}", training_time);
    println!("Training MSE: {:.6}", train_mse);
    println!("Training R²:  {:.6}", train_r2);
    println!("Test MSE:     {:.6}", test_mse);
    println!("Test R²:      {:.6}", test_r2);

    // Show basic model information
    println!();
    println!("📋 Model successfully trained and evaluated!");

    println!();
    println!("✅ Example completed successfully!");
    println!();
    println!("💡 Next steps:");
    println!("   - Try Ridge regression for regularization: RidgeRegression::new().alpha(1.0)");
    println!("   - Try Lasso for feature selection: LassoRegression::new().alpha(0.1)");
    println!("   - Experiment with different solvers and parameters");
    println!("   - Use cross-validation: RidgeCV::new() or LassoCV::new()");

    Ok(())
}
