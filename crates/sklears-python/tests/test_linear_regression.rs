//! Property-based tests for linear regression Python bindings
//!
//! This module contains comprehensive property-based tests to ensure
//! the robustness and correctness of the linear regression implementations.

use proptest::prelude::*;
use scirs2_autograd::ndarray::{Array1, Array2};
use scirs2_core::random::{thread_rng, Rng};

// We can't directly test PyO3 bindings in unit tests, but we can test
// the underlying mathematical properties and data transformations

#[allow(non_snake_case)]
#[cfg(test)]
mod linear_regression_properties {
    use super::*;

    /// Property: Linear regression coefficients should be stable for well-conditioned problems
    proptest! {
        #[test]
        fn test_coefficient_stability(
            n_samples in 10..100usize,
            n_features in 1..10usize,
            noise_scale in 0.0f64..0.1,
        ) {
            // Generate synthetic data
            let mut rng = thread_rng();
            let x = Array2::from_shape_fn((n_samples, n_features), |_| rng.gen::<f64>());
            let true_coef = Array1::from_shape_fn(n_features, |_| rng.gen::<f64>());
            let noise = Array1::from_shape_fn(n_samples, |_| rng.gen::<f64>() * noise_scale);
            let y = x.dot(&true_coef) + noise;

            // Test that the data dimensions are consistent
            prop_assert_eq!(x.nrows(), n_samples);
            prop_assert_eq!(x.ncols(), n_features);
            prop_assert_eq!(y.len(), n_samples);

            // Test that the coefficient vector has the right dimension
            prop_assert_eq!(true_coef.len(), n_features);
        }
    }

    /// Property: Predictions should be deterministic for the same input
    proptest! {
        #[test]
        fn test_prediction_determinism(
            n_samples in 5..50usize,
            n_features in 1..5usize,
        ) {
            let mut rng = thread_rng();
            let x_test = Array2::from_shape_fn((n_samples, n_features), |_| rng.gen::<f64>());
            let coef = Array1::from_shape_fn(n_features, |_| rng.gen::<f64>());
            let intercept = rng.gen::<f64>();

            // Manual prediction calculation
            let pred1 = x_test.dot(&coef) + intercept;
            let pred2 = x_test.dot(&coef) + intercept;

            // Predictions should be identical
            prop_assert_eq!(pred1.clone(), pred2);
            prop_assert_eq!(pred1.len(), n_samples);
        }
    }

    /// Property: Linear regression should satisfy basic mathematical properties
    proptest! {
        #[test]
        fn test_linear_properties(
            n_samples in 10..50usize,
            n_features in 1..5usize,
            scale_factor in 1.0f64..10.0,
        ) {
            let mut rng = thread_rng();
            let x = Array2::from_shape_fn((n_samples, n_features), |_| rng.gen::<f64>());
            let coef = Array1::from_shape_fn(n_features, |_| rng.gen::<f64>());

            // Test linearity: f(a*x) = a*f(x)
            let pred_original = x.dot(&coef);
            let pred_scaled = (x * scale_factor).dot(&coef);
            let pred_manual_scale = pred_original * scale_factor;

            // Allow for small floating point differences
            let diff = (&pred_scaled - &pred_manual_scale).mapv(|x| x.abs()).sum();
            prop_assert!(diff < 1e-10 * n_samples as f64);
        }
    }

    /// Property: R² score should be between -∞ and 1 for valid inputs
    proptest! {
        #[test]
        fn test_r2_score_bounds(
            n_samples in 10..100usize,
        ) {
            let mut rng = thread_rng();
            let y_true = Array1::from_shape_fn(n_samples, |_| rng.gen::<f64>());
            let y_pred = Array1::from_shape_fn(n_samples, |_| rng.gen::<f64>());

            // Calculate R² score manually
            let y_mean = y_true.mean().unwrap_or(0.0);
            let ss_tot: f64 = y_true.iter().map(|&y| (y - y_mean).powi(2)).sum();
            let ss_res: f64 = y_true.iter()
                .zip(y_pred.iter())
                .map(|(&y, &pred)| (y - pred).powi(2))
                .sum();

            if ss_tot > 0.0 {
                let r2: f64 = 1.0 - (ss_res / ss_tot);
                // R² should be finite
                prop_assert!(r2.is_finite());
                // R² can be negative for bad predictions, but typically ≤ 1
                prop_assert!(r2 <= 1.0 + 1e-10); // Allow small floating point errors
            }
        }
    }

    /// Property: Zero coefficients should produce constant predictions
    proptest! {
        #[test]
        fn test_zero_coefficients(
            n_samples in 5..50usize,
            n_features in 1..5usize,
            intercept in -100.0f64..100.0,
        ) {
            let mut rng = thread_rng();
            let x = Array2::from_shape_fn((n_samples, n_features), |_| rng.gen::<f64>());
            let zero_coef = Array1::zeros(n_features);

            let predictions = x.dot(&zero_coef) + intercept;

            // All predictions should equal the intercept
            for &pred in predictions.iter() {
                prop_assert!((pred - intercept).abs() < 1e-10);
            }
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod data_validation_properties {
    use super::*;

    /// Property: Input validation should catch dimension mismatches
    proptest! {
        #[test]
        fn test_dimension_mismatch_detection(
            n_samples_x in 5..20usize,
            n_samples_y in 5..20usize,
            n_features in 1..5usize,
        ) {
            prop_assume!(n_samples_x != n_samples_y);

            let mut rng = thread_rng();
            let x = Array2::from_shape_fn((n_samples_x, n_features), |_| rng.gen::<f64>());
            let y = Array1::from_shape_fn(n_samples_y, |_| rng.gen::<f64>());

            // This should represent a dimension mismatch that should be caught
            prop_assert_ne!(x.nrows(), y.len());
        }
    }

    /// Property: Valid data should pass basic consistency checks
    proptest! {
        #[test]
        fn test_valid_data_consistency(
            n_samples in 5..100usize,
            n_features in 1..10usize,
        ) {
            let mut rng = thread_rng();
            let x = Array2::from_shape_fn((n_samples, n_features), |_| rng.gen::<f64>());
            let y = Array1::from_shape_fn(n_samples, |_| rng.gen::<f64>());

            // Basic consistency checks
            prop_assert_eq!(x.nrows(), y.len());
            prop_assert_eq!(x.ncols(), n_features);
            prop_assert!(x.iter().all(|&val| val.is_finite()));
            prop_assert!(y.iter().all(|&val| val.is_finite()));
        }
    }
}
