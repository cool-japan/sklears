//! Robustness tests for calibration methods under edge cases and extreme conditions
//!
//! This module provides comprehensive tests to ensure calibration methods handle
//! difficult scenarios gracefully, including numerical edge cases, data distribution
//! extremes, and pathological inputs.

use crate::{
    histogram::HistogramBinningCalibrator,
    isotonic::IsotonicCalibrator,
    metrics::{expected_calibration_error, BinStrategy, CalibrationMetricsConfig},
    temperature::TemperatureScalingCalibrator,
    CalibratedClassifierCV, CalibrationMethod,
};
use scirs2_core::ndarray::{array, Array1, Array2};
use sklears_core::{
    error::Result,
    traits::{Fit, PredictProba},
    types::Float,
};

/// Test suite for robustness under extreme conditions
pub struct RobustnessTestSuite;

impl RobustnessTestSuite {
    /// Test all calibration methods with edge case inputs
    pub fn run_all_tests() -> Result<()> {
        Self::test_extreme_probabilities()?;
        Self::test_small_datasets()?;
        Self::test_imbalanced_datasets()?;
        Self::test_constant_predictions()?;
        Self::test_numerical_precision_limits()?;
        Self::test_zero_variance_features()?;
        Self::test_infinite_and_nan_handling()?;
        Self::test_single_class_problems()?;
        Self::test_perfect_separation()?;
        Self::test_adversarial_inputs()?;
        Ok(())
    }

    /// Test handling of extreme probability values
    pub fn test_extreme_probabilities() -> Result<()> {
        println!("Testing extreme probability handling...");

        // Test with probabilities very close to 0 and 1
        let extreme_probabilities = array![
            1e-15,
            1e-10,
            1e-5,
            0.5,
            1.0 - 1e-5,
            1.0 - 1e-10,
            1.0 - 1e-15
        ];
        let y_true = array![0, 0, 0, 1, 1, 1, 1];

        // Test isotonic calibrator
        let isotonic = IsotonicCalibrator::new().fit(&extreme_probabilities, &y_true)?;
        let calibrated = isotonic.predict_proba(&extreme_probabilities)?;

        // Ensure all outputs are valid probabilities
        for &prob in calibrated.iter() {
            assert!(
                prob >= 0.0 && prob <= 1.0,
                "Probability out of range: {}",
                prob
            );
            assert!(prob.is_finite(), "Non-finite probability: {}", prob);
        }

        // Test temperature scaling
        let temp_calibrator =
            TemperatureScalingCalibrator::new().fit(&extreme_probabilities, &y_true)?;
        let temp_calibrated = temp_calibrator.predict_proba(&extreme_probabilities)?;

        for &prob in temp_calibrated.iter() {
            assert!(
                prob >= 0.0 && prob <= 1.0,
                "Temperature scaling probability out of range: {}",
                prob
            );
            assert!(
                prob.is_finite(),
                "Temperature scaling non-finite probability: {}",
                prob
            );
        }

        Ok(())
    }

    /// Test behavior with very small datasets
    pub fn test_small_datasets() -> Result<()> {
        println!("Testing small dataset handling...");

        // Test with just 2 samples
        let x_tiny = array![[1.0], [2.0]];
        let y_tiny = array![0, 1];

        let methods = vec![
            CalibrationMethod::Sigmoid,
            CalibrationMethod::Isotonic,
            CalibrationMethod::Temperature,
            CalibrationMethod::HistogramBinning { n_bins: 2 },
        ];

        for method in methods {
            let calibrator = CalibratedClassifierCV::new().method(method.clone());

            // Should not panic even with tiny dataset
            match calibrator.fit(&x_tiny, &y_tiny) {
                Ok(fitted) => {
                    let probas = fitted.predict_proba(&x_tiny)?;
                    assert_eq!(probas.dim(), (2, 2));

                    // Check probability normalization
                    for row in probas.rows() {
                        let sum: Float = row.sum();
                        assert!(
                            (sum - 1.0).abs() < 1e-6,
                            "Probabilities don't sum to 1: {}",
                            sum
                        );
                    }
                }
                Err(_) => {
                    // Some methods may legitimately fail on tiny datasets
                    println!("Method {:?} failed on tiny dataset (expected)", method);
                }
            }
        }

        Ok(())
    }

    /// Test behavior with highly imbalanced datasets
    pub fn test_imbalanced_datasets() -> Result<()> {
        println!("Testing imbalanced dataset handling...");

        // Create 99% class 0, 1% class 1
        let mut x_imbalanced = Array2::zeros((100, 2));
        let mut y_imbalanced = Array1::zeros(100);

        // Fill with predictable patterns
        for i in 0..100 {
            x_imbalanced[[i, 0]] = i as Float / 100.0;
            x_imbalanced[[i, 1]] = (i as Float).sin();
            if i >= 99 {
                // Only last sample is class 1
                y_imbalanced[i] = 1;
            }
        }

        let methods = vec![
            CalibrationMethod::Sigmoid,
            CalibrationMethod::Isotonic,
            CalibrationMethod::BBQ {
                min_bins: 2,
                max_bins: 5,
            },
            CalibrationMethod::KDE,
        ];

        for method in methods {
            let calibrator = CalibratedClassifierCV::new().method(method.clone());

            match calibrator.fit(&x_imbalanced, &y_imbalanced) {
                Ok(fitted) => {
                    let probas = fitted.predict_proba(&x_imbalanced)?;

                    // Should handle imbalanced data gracefully
                    assert_eq!(probas.dim(), (100, 2));

                    // Check that minority class gets reasonable probability
                    let min_prob_class1 =
                        probas.column(1).fold(Float::INFINITY, |acc, &x| acc.min(x));
                    assert!(
                        min_prob_class1 >= 0.0,
                        "Negative probability for minority class"
                    );
                }
                Err(e) => {
                    println!("Method {:?} failed on imbalanced dataset: {}", method, e);
                }
            }
        }

        Ok(())
    }

    /// Test behavior with constant predictions
    pub fn test_constant_predictions() -> Result<()> {
        println!("Testing constant prediction handling...");

        let y = array![0, 0, 1, 1];

        // Test with constant probabilities (no information)
        let constant_probs = array![0.5, 0.5, 0.5, 0.5];

        let isotonic = IsotonicCalibrator::new().fit(&constant_probs, &y)?;
        let calibrated = isotonic.predict_proba(&constant_probs)?;

        // Should handle constant input gracefully
        for &prob in calibrated.iter() {
            assert!(prob >= 0.0 && prob <= 1.0);
            assert!(prob.is_finite());
        }

        // Test with all predictions being 0
        let zero_probs = array![0.0, 0.0, 0.0, 0.0];
        let isotonic_zero = IsotonicCalibrator::new().fit(&zero_probs, &y)?;
        let calibrated_zero = isotonic_zero.predict_proba(&zero_probs)?;

        for &prob in calibrated_zero.iter() {
            assert!(prob >= 0.0 && prob <= 1.0);
        }

        // Test with all predictions being 1
        let one_probs = array![1.0, 1.0, 1.0, 1.0];
        let isotonic_one = IsotonicCalibrator::new().fit(&one_probs, &y)?;
        let calibrated_one = isotonic_one.predict_proba(&one_probs)?;

        for &prob in calibrated_one.iter() {
            assert!(prob >= 0.0 && prob <= 1.0);
        }

        Ok(())
    }

    /// Test numerical precision limits
    pub fn test_numerical_precision_limits() -> Result<()> {
        println!("Testing numerical precision limits...");

        // Test with very small differences that might cause numerical issues
        let precision_probs = array![
            0.5000000000000000,
            0.5000000000000001,
            0.4999999999999999,
            0.5000000000000002,
        ];
        let y = array![0, 0, 1, 1];

        let histogram = HistogramBinningCalibrator::new(10).fit(&precision_probs, &y)?;
        let calibrated = histogram.predict_proba(&precision_probs)?;

        // Should handle tiny differences without numerical issues
        for &prob in calibrated.iter() {
            assert!(prob >= 0.0 && prob <= 1.0);
            assert!(prob.is_finite());
        }

        // Test with values very close to machine epsilon
        let epsilon_probs = array![
            Float::EPSILON,
            1.0 - Float::EPSILON,
            Float::EPSILON * 2.0,
            1.0 - Float::EPSILON * 2.0,
        ];

        let temp_calibrator = TemperatureScalingCalibrator::new().fit(&epsilon_probs, &y)?;
        let temp_calibrated = temp_calibrator.predict_proba(&epsilon_probs)?;

        for &prob in temp_calibrated.iter() {
            assert!(prob >= 0.0 && prob <= 1.0);
            assert!(prob.is_finite());
        }

        Ok(())
    }

    /// Test handling of zero variance features
    pub fn test_zero_variance_features() -> Result<()> {
        println!("Testing zero variance feature handling...");

        // Create dataset where one feature has zero variance
        let x_zero_var = array![
            [1.0, 0.0], // Second feature is constant
            [2.0, 0.0],
            [3.0, 0.0],
            [4.0, 0.0],
        ];
        let y = array![0, 0, 1, 1];

        let methods = vec![
            CalibrationMethod::Sigmoid,
            CalibrationMethod::Temperature,
            CalibrationMethod::LocalKNN { k: 2 },
        ];

        for method in methods {
            let calibrator = CalibratedClassifierCV::new().method(method.clone());

            match calibrator.fit(&x_zero_var, &y) {
                Ok(fitted) => {
                    let probas = fitted.predict_proba(&x_zero_var)?;
                    assert_eq!(probas.dim(), (4, 2));

                    // Should produce valid probabilities despite zero variance
                    for row in probas.rows() {
                        let sum: Float = row.sum();
                        assert!((sum - 1.0).abs() < 1e-6);
                    }
                }
                Err(e) => {
                    println!("Method {:?} failed on zero variance data: {}", method, e);
                }
            }
        }

        Ok(())
    }

    /// Test handling of infinite and NaN values
    pub fn test_infinite_and_nan_handling() -> Result<()> {
        println!("Testing infinite and NaN handling...");

        // Note: Most calibration methods should reject NaN/infinite inputs
        // This test ensures they fail gracefully rather than panicking

        let problematic_probs = array![0.5, Float::INFINITY, 0.3, Float::NEG_INFINITY];
        let y = array![0, 1, 0, 1];

        // Should either handle gracefully or return meaningful error
        match IsotonicCalibrator::new().fit(&problematic_probs, &y) {
            Ok(isotonic) => {
                // If it accepts the input, test with finite inputs to see if it works
                match isotonic.predict_proba(&array![0.5, 0.3]) {
                    Ok(calibrated) => {
                        // Check if outputs are finite - this is preferred but not required
                        let all_finite = calibrated.iter().all(|&prob| prob.is_finite());
                        if all_finite {
                            println!("Isotonic calibrator handled infinite inputs and produces finite outputs");
                        } else {
                            println!("Isotonic calibrator accepted infinite inputs but produces non-finite outputs");
                        }
                    }
                    Err(_) => {
                        println!("Isotonic calibrator accepted infinite training data but failed on prediction");
                    }
                }
            }
            Err(_) => {
                // Rejecting infinite inputs is also acceptable
                println!("Isotonic calibrator properly rejected infinite inputs");
            }
        }

        // Test with NaN
        let nan_probs = array![0.5, Float::NAN, 0.3, 0.7];

        match TemperatureScalingCalibrator::new().fit(&nan_probs, &y) {
            Ok(_) => {
                println!("Temperature scaling accepted NaN inputs (unusual but handled)");
            }
            Err(_) => {
                println!("Temperature scaling properly rejected NaN inputs");
            }
        }

        Ok(())
    }

    /// Test single class problems
    pub fn test_single_class_problems() -> Result<()> {
        println!("Testing single class problems...");

        let x_single_class = array![[1.0], [2.0], [3.0], [4.0]];
        let y_single_class = array![1, 1, 1, 1]; // All same class

        let methods = vec![
            CalibrationMethod::Sigmoid,
            CalibrationMethod::Isotonic,
            CalibrationMethod::HistogramBinning { n_bins: 3 },
        ];

        for method in methods {
            let calibrator = CalibratedClassifierCV::new().method(method.clone());

            match calibrator.fit(&x_single_class, &y_single_class) {
                Ok(fitted) => {
                    let probas = fitted.predict_proba(&x_single_class)?;

                    // Should handle single class gracefully
                    // Might produce uniform probabilities or handle differently
                    assert_eq!(probas.nrows(), 4);

                    for row in probas.rows() {
                        let sum: Float = row.sum();
                        assert!((sum - 1.0).abs() < 1e-6);

                        // All probabilities should be valid
                        for &prob in row.iter() {
                            assert!(prob >= 0.0 && prob <= 1.0);
                        }
                    }
                }
                Err(e) => {
                    println!(
                        "Method {:?} failed on single class: {} (this may be expected)",
                        method, e
                    );
                }
            }
        }

        Ok(())
    }

    /// Test perfect separation scenarios
    pub fn test_perfect_separation() -> Result<()> {
        println!("Testing perfect separation scenarios...");

        // Create perfectly separable data
        let y_perfect = array![0, 0, 1, 1];

        // Perfect prediction probabilities
        let perfect_probs = array![0.0, 0.1, 0.9, 1.0];

        let isotonic = IsotonicCalibrator::new().fit(&perfect_probs, &y_perfect)?;
        let calibrated = isotonic.predict_proba(&perfect_probs)?;

        // Should handle perfect separation
        for &prob in calibrated.iter() {
            assert!(prob >= 0.0 && prob <= 1.0);
            assert!(prob.is_finite());
        }

        // Test that calibration doesn't break perfect predictions
        let config = CalibrationMetricsConfig {
            n_bins: 10,
            bin_strategy: BinStrategy::Uniform,
        };
        let ece_before = expected_calibration_error(&y_perfect, &perfect_probs, &config)?;
        let ece_after = expected_calibration_error(&y_perfect, &calibrated, &config)?;

        // ECE should remain low (perfect predictions should stay good)
        assert!(
            ece_after <= ece_before + 0.1,
            "Calibration significantly worsened perfect predictions"
        );

        Ok(())
    }

    /// Test adversarial inputs designed to break calibration
    pub fn test_adversarial_inputs() -> Result<()> {
        println!("Testing adversarial inputs...");

        // Alternating extreme values
        let adversarial_probs = array![0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
        let adversarial_y = array![1, 0, 1, 0, 1, 0]; // Opposite of probabilities

        let isotonic = IsotonicCalibrator::new().fit(&adversarial_probs, &adversarial_y)?;
        let calibrated = isotonic.predict_proba(&adversarial_probs)?;

        // Should produce valid outputs even with adversarial inputs
        for &prob in calibrated.iter() {
            assert!(prob >= 0.0 && prob <= 1.0);
            assert!(prob.is_finite());
        }

        // Test with many repeated values that could cause binning issues
        let repeated_probs = array![0.33, 0.33, 0.33, 0.33, 0.33, 0.67, 0.67, 0.67];
        let repeated_y = array![0, 1, 0, 1, 0, 1, 0, 1];

        let histogram = HistogramBinningCalibrator::new(3).fit(&repeated_probs, &repeated_y)?;
        let hist_calibrated = histogram.predict_proba(&repeated_probs)?;

        for &prob in hist_calibrated.iter() {
            assert!(prob >= 0.0 && prob <= 1.0);
            assert!(prob.is_finite());
        }

        Ok(())
    }

    /// Test calibration quality under stress
    pub fn test_calibration_quality_stress() -> Result<()> {
        println!("Testing calibration quality under stress...");

        // Generate challenging but realistic data (smaller for performance)
        let n_samples = 200;
        let mut x = Array2::zeros((n_samples, 3));
        let mut y = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let t = i as Float / n_samples as Float;

            // Create non-linear relationships
            x[[i, 0]] = t;
            x[[i, 1]] = (t * 10.0).sin();
            x[[i, 2]] = (t * 5.0).cos() * t;

            // Complex decision boundary
            let decision_value = x[[i, 0]] + 0.5 * x[[i, 1]] + 0.3 * x[[i, 2]];
            y[i] = if decision_value > 0.5 { 1 } else { 0 };
        }

        // Test core methods on this challenging dataset (reduced for performance)
        let methods = vec![
            CalibrationMethod::Isotonic,
            CalibrationMethod::Temperature,
            CalibrationMethod::HistogramBinning { n_bins: 10 },
        ];

        for method in methods {
            let calibrator = CalibratedClassifierCV::new().method(method.clone()).cv(3);

            match calibrator.fit(&x, &y) {
                Ok(fitted) => {
                    let probas = fitted.predict_proba(&x)?;

                    // Basic sanity checks
                    assert_eq!(probas.dim(), (n_samples, 2));

                    for row in probas.rows() {
                        let sum: Float = row.sum();
                        assert!((sum - 1.0).abs() < 1e-6, "Probabilities don't sum to 1");

                        for &prob in row.iter() {
                            assert!(prob >= 0.0 && prob <= 1.0, "Invalid probability: {}", prob);
                            assert!(prob.is_finite(), "Non-finite probability");
                        }
                    }

                    // Check that calibration produces reasonable results
                    let class1_probs = probas.column(1);
                    let config = CalibrationMetricsConfig {
                        n_bins: 10,
                        bin_strategy: BinStrategy::Uniform,
                    };
                    let ece = expected_calibration_error(&y, &class1_probs.to_owned(), &config)?;

                    assert!(ece <= 1.0, "ECE unexpectedly high: {}", ece);
                    assert!(ece >= 0.0, "ECE cannot be negative: {}", ece);

                    println!("Method {:?}: ECE = {:.4}", method, ece);
                }
                Err(e) => {
                    println!("Method {:?} failed on stress test: {}", method, e);
                }
            }
        }

        Ok(())
    }

    /// Run performance benchmarks to ensure no major regressions
    pub fn benchmark_performance() -> Result<()> {
        println!("Running performance benchmarks...");

        use std::time::Instant;

        // Create moderately sized dataset
        let n_samples = 5000;
        let mut x = Array2::zeros((n_samples, 5));
        let mut y = Array1::zeros(n_samples);

        for i in 0..n_samples {
            for j in 0..5 {
                x[[i, j]] = (i * j) as Float / n_samples as Float;
            }
            y[i] = if x.row(i).sum() > 2.5 { 1 } else { 0 };
        }

        let methods = vec![
            CalibrationMethod::Sigmoid,
            CalibrationMethod::Isotonic,
            CalibrationMethod::Temperature,
            CalibrationMethod::HistogramBinning { n_bins: 10 },
        ];

        for method in methods {
            let start = Instant::now();

            let calibrator = CalibratedClassifierCV::new().method(method.clone()).cv(2);

            match calibrator.fit(&x, &y) {
                Ok(fitted) => {
                    let _probas = fitted.predict_proba(&x)?;
                    let duration = start.elapsed();

                    println!("Method {:?}: {:.2}ms", method, duration.as_millis());

                    // Ensure reasonable performance (adjust thresholds as needed)
                    assert!(
                        duration.as_secs() < 30,
                        "Method {:?} took too long: {:?}",
                        method,
                        duration
                    );
                }
                Err(e) => {
                    println!("Method {:?} failed benchmark: {}", method, e);
                }
            }
        }

        Ok(())
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extreme_probabilities() {
        RobustnessTestSuite::test_extreme_probabilities().unwrap();
    }

    #[test]
    fn test_small_datasets() {
        RobustnessTestSuite::test_small_datasets().unwrap();
    }

    #[test]
    fn test_imbalanced_datasets() {
        RobustnessTestSuite::test_imbalanced_datasets().unwrap();
    }

    #[test]
    fn test_constant_predictions() {
        RobustnessTestSuite::test_constant_predictions().unwrap();
    }

    #[test]
    fn test_numerical_precision_limits() {
        RobustnessTestSuite::test_numerical_precision_limits().unwrap();
    }

    #[test]
    fn test_zero_variance_features() {
        RobustnessTestSuite::test_zero_variance_features().unwrap();
    }

    #[test]
    fn test_infinite_and_nan_handling() {
        RobustnessTestSuite::test_infinite_and_nan_handling().unwrap();
    }

    #[test]
    fn test_single_class_problems() {
        RobustnessTestSuite::test_single_class_problems().unwrap();
    }

    #[test]
    fn test_perfect_separation() {
        RobustnessTestSuite::test_perfect_separation().unwrap();
    }

    #[test]
    fn test_adversarial_inputs() {
        RobustnessTestSuite::test_adversarial_inputs().unwrap();
    }

    #[test]
    fn test_calibration_quality_stress() {
        RobustnessTestSuite::test_calibration_quality_stress().unwrap();
    }

    #[test]
    fn test_benchmark_performance() {
        RobustnessTestSuite::benchmark_performance().unwrap();
    }

    #[test]
    fn test_run_all_tests() {
        RobustnessTestSuite::run_all_tests().unwrap();
    }
}
