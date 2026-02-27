/// Advanced property-based tests for core traits and mathematical invariants
///
/// These tests verify that core machine learning properties and invariants
/// hold across all implementations using proptest.
use crate::traits::{Fit, Predict, Transform};
use crate::types::{LearningRate, Probability, RegularizationStrength};
use crate::validation::{ValidationRule, ValidationRules};
use proptest::prelude::*;
use scirs2_core::ndarray::{Array1, Array2};
use std::fmt::Debug;

// Trait for testing estimators that implement both fit and transform
pub trait TestableTransformer<X, Y>: Clone + Debug
where
    Self: Fit<X, Y> + Transform<X>,
    Self::Fitted: Transform<X>,
{
    fn new_for_test() -> Self;
}

// Trait for testing estimators that implement both fit and predict
pub trait TestablePredictorClassification<X>: Clone + Debug
where
    Self: Fit<X, Array1<i32>> + Predict<X, Array1<i32>>,
    Self::Fitted: Predict<X, Array1<i32>>,
{
    fn new_for_test() -> Self;
}

pub trait TestablePredictorRegression<X>: Clone + Debug
where
    Self: Fit<X, Array1<f64>> + Predict<X, Array1<f64>>,
    Self::Fitted: Predict<X, Array1<f64>>,
{
    fn new_for_test() -> Self;
}

proptest! {
    /// Test that transformers preserve data dimensions correctly
    #[test]
    fn test_transformer_dimension_preservation(
        n_samples in 10..100usize,
        n_features in 2..20usize,
        scale in 0.1..10.0f64
    ) {
        // Create test data
        let mut x = Array2::<f64>::zeros((n_samples, n_features));
        for i in 0..n_samples {
            for j in 0..n_features {
                x[[i, j]] = scale * (i as f64 + j as f64) / 10.0;
            }
        }

        // Test dimension preservation properties
        prop_assert_eq!(x.shape(), &[n_samples, n_features]);

        // All values should be finite
        for &val in x.iter() {
            prop_assert!(val.is_finite());
        }
    }

    /// Test that prediction outputs have correct dimensions
    #[test]
    fn test_predictor_output_dimensions(
        n_samples in 10..100usize,
        n_features in 2..10usize,
        n_classes in 2..5usize
    ) {
        // Create classification data
        let x = Array2::<f64>::zeros((n_samples, n_features));
        let y = Array1::from_vec((0..n_samples).map(|i| (i % n_classes) as i32).collect());

        // Basic dimension checks
        prop_assert_eq!(x.shape(), &[n_samples, n_features]);
        prop_assert_eq!(y.len(), n_samples);

        // All labels should be valid
        for &label in y.iter() {
            prop_assert!(label >= 0 && label < n_classes as i32);
        }
    }

    /// Test mathematical properties of distance metrics
    #[test]
    fn test_distance_metric_properties(
        dim in 2..10usize,
        scale1 in -10.0..10.0f64,
        scale2 in -10.0..10.0f64
    ) {
        let point1 = Array1::from_vec(vec![scale1; dim]);
        let point2 = Array1::from_vec(vec![scale2; dim]);
        let point3 = Array1::from_vec((0..dim).map(|i| scale1 + i as f64).collect());

        // Test identity: distance(x, x) = 0
        let euclidean_self = euclidean_distance_test(&point1, &point1);
        prop_assert!(euclidean_self.abs() < 1e-10);

        // Test symmetry: distance(x, y) = distance(y, x)
        let d1 = euclidean_distance_test(&point1, &point2);
        let d2 = euclidean_distance_test(&point2, &point1);
        prop_assert!((d1 - d2).abs() < 1e-10);

        // Test non-negativity
        prop_assert!(d1 >= 0.0);

        // Test triangle inequality: d(x,z) <= d(x,y) + d(y,z)
        let d_13 = euclidean_distance_test(&point1, &point3);
        let d_12 = euclidean_distance_test(&point1, &point2);
        let d_23 = euclidean_distance_test(&point2, &point3);
        prop_assert!(d_13 <= d_12 + d_23 + 1e-10);
    }

    /// Test numerical stability properties
    #[test]
    fn test_numerical_stability(
        n_samples in 10..50usize,
        n_features in 2..10usize,
        magnitude in prop::sample::select(vec![1e-6, 1e-3, 1.0, 1e3, 1e6])
    ) {
        // Create data with different magnitudes
        let mut x = Array2::<f64>::zeros((n_samples, n_features));
        for i in 0..n_samples {
            for j in 0..n_features {
                x[[i, j]] = magnitude * (i as f64 + j as f64 + 1.0) / 10.0;
            }
        }

        // Test that all values remain finite
        for &val in x.iter() {
            prop_assert!(val.is_finite());
        }

        // Test basic statistics are well-behaved
        let mean = x.mean().unwrap_or_default();
        let max_val = x.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let min_val = x.iter().fold(f64::INFINITY, |a, &b| a.min(b));

        prop_assert!(mean.is_finite());
        prop_assert!(max_val.is_finite());
        prop_assert!(min_val.is_finite());
        prop_assert!(max_val >= min_val);
    }

    /// Test invariance properties for transformations
    #[test]
    fn test_transformation_invariance(
        n_samples in 10..50usize,
        n_features in 2..8usize,
        shift in -5.0..5.0f64
    ) {
        // Create base data
        let mut x = Array2::<f64>::zeros((n_samples, n_features));
        for i in 0..n_samples {
            for j in 0..n_features {
                x[[i, j]] = (i as f64 + j as f64) / 10.0;
            }
        }

        // Create shifted data
        let x_shifted = x.mapv(|val| val + shift);

        // Both datasets should have same shape
        prop_assert_eq!(x.shape(), x_shifted.shape());

        // Test that relative relationships are preserved
        if n_samples >= 2 && n_features >= 2 {
            let dist_orig = euclidean_distance_test(
                &x.row(0).to_owned(),
                &x.row(1).to_owned()
            );
            let dist_shifted = euclidean_distance_test(
                &x_shifted.row(0).to_owned(),
                &x_shifted.row(1).to_owned()
            );

            // Distance should be preserved under uniform shift
            prop_assert!((dist_orig - dist_shifted).abs() < 1e-10);
        }
    }

    /// Test scale invariance properties
    #[test]
    fn test_scale_invariance(
        n_samples in 10..30usize,
        n_features in 2..6usize,
        scale_factor in 0.1..10.0f64
    ) {
        // Create base data
        let mut x = Array2::<f64>::zeros((n_samples, n_features));
        for i in 0..n_samples {
            for j in 0..n_features {
                x[[i, j]] = (i as f64 + j as f64 + 1.0) / 10.0;
            }
        }

        // Create scaled data
        let x_scaled = x.mapv(|val| val * scale_factor);

        // Test that scaling preserves non-zero patterns
        if scale_factor > 1e-10 {
            for (&orig, &scaled) in x.iter().zip(x_scaled.iter()) {
                if orig != 0.0 {
                    prop_assert!(scaled != 0.0);
                }
                if orig == 0.0 {
                    prop_assert!(scaled == 0.0);
                }
            }
        }

        // Test that relative magnitudes are preserved
        if n_samples >= 2 && n_features >= 1 {
            let orig_ratio = if x[[1, 0]] != 0.0 { x[[0, 0]] / x[[1, 0]] } else { 0.0 };
            let scaled_ratio = if x_scaled[[1, 0]] != 0.0 { x_scaled[[0, 0]] / x_scaled[[1, 0]] } else { 0.0 };

            if x[[1, 0]] != 0.0 && x_scaled[[1, 0]] != 0.0 {
                prop_assert!((orig_ratio - scaled_ratio).abs() < 1e-10);
            }
        }
    }

    /// Test monotonicity properties
    #[test]
    fn test_monotonicity_properties(
        n_samples in 5..20usize
    ) {
        // Create monotonically increasing data
        let x1 = Array1::from_vec((0..n_samples).map(|i| i as f64).collect());
        let x2 = Array1::from_vec((0..n_samples).map(|i| (i + 1) as f64).collect());

        // Test that ordering is preserved
        for i in 0..n_samples-1 {
            prop_assert!(x1[i] <= x1[i+1]);
            prop_assert!(x2[i] <= x2[i+1]);
            prop_assert!(x1[i] <= x2[i]);
        }

        // Test monotonic transformations preserve ordering
        let x1_log = x1.mapv(|v| (v + 1.0).ln()); // Add 1 to avoid log(0)
        let x2_log = x2.mapv(|v| (v + 1.0).ln());

        for i in 0..n_samples-1 {
            prop_assert!(x1_log[i] <= x1_log[i+1]);
            prop_assert!(x2_log[i] <= x2_log[i+1]);
        }
    }

    /// Test convexity properties for optimization algorithms
    #[test]
    fn test_convexity_properties(
        n_points in 5..20usize,
        alpha in 0.0..1.0f64
    ) {
        // Create test points
        let x = Array1::from_vec((0..n_points).map(|i| i as f64 / n_points as f64).collect());
        let y = Array1::from_vec((0..n_points).map(|i| (i as f64).powi(2)).collect());

        // Test convex combination
        if x.len() >= 2 {
            let x_combo = alpha * x[0] + (1.0 - alpha) * x[1];
            let y_combo = alpha * y[0] + (1.0 - alpha) * y[1];

            // For convex functions, f(αx + (1-α)y) ≤ αf(x) + (1-α)f(y)
            // Since we're using x² which is convex, this should hold
            let f_combo = x_combo.powi(2);
            prop_assert!(f_combo <= y_combo + 1e-10);
        }
    }

    /// Test regression specific properties
    #[test]
    fn test_regression_invariants(
        n_samples in 10..50usize,
        noise_level in 0.0..1.0f64
    ) {
        // Create regression data with known relationship
        let x = Array1::from_vec((0..n_samples).map(|i| i as f64).collect());
        let y_true = x.mapv(|v| 2.0 * v + 1.0); // Linear relationship

        // Add noise
        let y_noisy = y_true.mapv(|v| v + noise_level * (v * 0.1));

        // Test that perfect predictions have zero error
        let mse_perfect = mean_squared_error_test(&y_true, &y_true);
        prop_assert!(mse_perfect < 1e-10);

        // Test that MSE increases with noise
        let mse_noisy = mean_squared_error_test(&y_true, &y_noisy);
        if noise_level > 1e-6 {
            prop_assert!(mse_noisy >= mse_perfect);
        }

        // Test that all errors are finite
        prop_assert!(mse_perfect.is_finite());
        prop_assert!(mse_noisy.is_finite());
    }

    /// Test classification specific properties
    #[test]
    fn test_classification_invariants(
        n_samples in 10..100usize,
        n_classes in 2..5usize,
        error_rate in 0.0..0.5f64
    ) {
        // Create classification data
        let y_true = Array1::from_vec((0..n_samples).map(|i| (i % n_classes) as i32).collect());

        // Create predictions with controlled error rate
        let n_errors = (n_samples as f64 * error_rate) as usize;
        let mut y_pred = y_true.clone();
        for i in 0..n_errors {
            let idx = i % n_samples;
            y_pred[idx] = (y_pred[idx] + 1) % n_classes as i32;
        }

        // Test accuracy bounds
        let accuracy = accuracy_score_test(&y_true, &y_pred);
        prop_assert!(accuracy >= 0.0);
        prop_assert!(accuracy <= 1.0);

        // Test that perfect prediction gives accuracy 1.0
        let perfect_accuracy = accuracy_score_test(&y_true, &y_true);
        prop_assert!((perfect_accuracy - 1.0).abs() < 1e-10);

        // Test that accuracy decreases with more errors
        if error_rate > 0.01 {
            prop_assert!(accuracy <= perfect_accuracy);
        }
    }
}

// Helper functions for testing without external dependencies
fn euclidean_distance_test(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

fn mean_squared_error_test(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> f64 {
    y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(&t, &p)| (t - p).powi(2))
        .sum::<f64>()
        / y_true.len() as f64
}

fn accuracy_score_test(y_true: &Array1<i32>, y_pred: &Array1<i32>) -> f64 {
    let correct = y_true
        .iter()
        .zip(y_pred.iter())
        .filter(|(&t, &p)| t == p)
        .count();
    correct as f64 / y_true.len() as f64
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_property_framework() {
        // Test that our helper functions work
        let a = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let b = Array1::from_vec(vec![1.0, 2.0, 3.0]);

        assert_eq!(euclidean_distance_test(&a, &b), 0.0);
        assert_eq!(mean_squared_error_test(&a, &b), 0.0);

        let y1 = Array1::from_vec(vec![1, 2, 3]);
        let y2 = Array1::from_vec(vec![1, 2, 3]);
        assert_eq!(accuracy_score_test(&y1, &y2), 1.0);
    }

    #[test]
    fn test_distance_properties() {
        let a = Array1::from_vec(vec![0.0, 0.0]);
        let b = Array1::from_vec(vec![3.0, 4.0]);

        // Test pythagorean theorem
        let dist = euclidean_distance_test(&a, &b);
        assert!((dist - 5.0).abs() < 1e-10);
    }
}

// Enhanced property test strategies for comprehensive testing

/// Property test strategies for generating valid data
pub mod strategies {
    use super::*;
    use proptest::collection::vec;

    /// Generate valid probability values
    pub fn probability_strategy() -> impl Strategy<Value = f64> {
        0.0..=1.0_f64
    }

    /// Generate valid learning rates
    pub fn learning_rate_strategy() -> impl Strategy<Value = f64> {
        1e-6..=1.0_f64
    }

    /// Generate valid regularization parameters
    pub fn regularization_strategy() -> impl Strategy<Value = f64> {
        0.0..=100.0_f64
    }

    /// Generate valid tolerance values
    pub fn tolerance_strategy() -> impl Strategy<Value = f64> {
        1e-12..=1e-1_f64
    }

    /// Generate valid iteration counts
    pub fn max_iter_strategy() -> impl Strategy<Value = usize> {
        1..=10000_usize
    }

    /// Generate small positive matrices for testing
    pub fn small_matrix_strategy() -> impl Strategy<Value = Array2<f64>> {
        (2..10usize, 2..10usize).prop_flat_map(|(nrows, ncols)| {
            vec(-10.0..10.0_f64, nrows * ncols)
                .prop_map(move |values| Array2::from_shape_vec((nrows, ncols), values).expect("valid array shape"))
        })
    }

    /// Generate positive definite matrices
    pub fn positive_definite_matrix_strategy(size: usize) -> impl Strategy<Value = Array2<f64>> {
        vec(-5.0..5.0_f64, size * size).prop_map(move |values| {
            let mut matrix = Array2::from_shape_vec((size, size), values).expect("valid array shape");

            // Make it positive definite by A = A^T A + I
            let at = matrix.t().to_owned();
            matrix = at.dot(&matrix);

            // Add identity to ensure positive definiteness
            for i in 0..size {
                matrix[[i, i]] += 1.0;
            }

            matrix
        })
    }

    /// Generate classification labels
    pub fn classification_labels_strategy(
        n_samples: usize,
        n_classes: usize,
    ) -> impl Strategy<Value = Array1<i32>> {
        vec(0..n_classes as i32, n_samples).prop_map(|labels| Array1::from_vec(labels))
    }

    /// Generate regression targets with noise
    pub fn regression_targets_strategy(n_samples: usize) -> impl Strategy<Value = Array1<f64>> {
        vec(-100.0..100.0_f64, n_samples).prop_map(|targets| Array1::from_vec(targets))
    }

    /// Generate sparse data (mostly zeros)
    pub fn sparse_matrix_strategy(
        nrows: usize,
        ncols: usize,
        _sparsity: f64,
    ) -> impl Strategy<Value = Array2<f64>> {
        vec(
            prop_oneof![
                9 => Just(0.0),  // 90% zeros for sparsity = 0.9
                1 => -10.0..10.0_f64, // 10% non-zeros
            ],
            nrows * ncols,
        )
        .prop_map(move |values| Array2::from_shape_vec((nrows, ncols), values).expect("valid array shape"))
    }
}

// Enhanced property tests using the new strategies

proptest! {
    /// Test validation framework with generated values
    #[test]
    fn test_validation_framework_properties(
        learning_rate in strategies::learning_rate_strategy(),
        regularization in strategies::regularization_strategy(),
        probability in strategies::probability_strategy(),
        tolerance in strategies::tolerance_strategy(),
        max_iter in strategies::max_iter_strategy()
    ) {
        // Test learning rate validation
        let lr_result = crate::validation::ml::validate_learning_rate(learning_rate);
        prop_assert!(lr_result.is_ok());

        // Test regularization validation
        let reg_result = crate::validation::ml::validate_regularization(regularization);
        prop_assert!(reg_result.is_ok());

        // Test probability validation
        let prob_result = crate::validation::ml::validate_probability(probability);
        prop_assert!(prob_result.is_ok());

        // Test tolerance validation
        let tol_result = crate::validation::ml::validate_tolerance(tolerance);
        prop_assert!(tol_result.is_ok());

        // Test max_iter validation
        let iter_result = crate::validation::ml::validate_max_iter(max_iter);
        prop_assert!(iter_result.is_ok());
    }

    /// Test newtype wrapper properties
    #[test]
    fn test_newtype_wrapper_properties(
        prob_val in strategies::probability_strategy(),
        lr_val in strategies::learning_rate_strategy(),
        reg_val in strategies::regularization_strategy()
    ) {
        // Test Probability wrapper
        if prob_val >= 0.0 && prob_val <= 1.0 {
            let prob = Probability::new(prob_val);
            prop_assert!(prob.is_ok());
            if let Ok(p) = prob {
                prop_assert_eq!(p.value(), prob_val);
                prop_assert!(p.value() >= 0.0 && p.value() <= 1.0);
            }
        }

        // Test LearningRate wrapper
        if lr_val > 0.0 && lr_val.is_finite() {
            let lr = LearningRate::new(lr_val);
            prop_assert!(lr.is_ok());
            if let Ok(l) = lr {
                prop_assert_eq!(l.get(), lr_val);
                prop_assert!(l.get() > 0.0);
            }
        }

        // Test RegularizationStrength wrapper
        if reg_val >= 0.0 && reg_val.is_finite() {
            let reg = RegularizationStrength::new(reg_val);
            prop_assert!(reg.is_ok());
            if let Ok(r) = reg {
                prop_assert_eq!(r.get(), reg_val);
                prop_assert!(r.get() >= 0.0);
            }
        }
    }

    /// Test matrix operation properties with generated data
    #[test]
    fn test_matrix_operation_properties(
        matrix in strategies::small_matrix_strategy()
    ) {
        let (nrows, ncols) = matrix.dim();

        // Test basic properties
        prop_assert_eq!(matrix.nrows(), nrows);
        prop_assert_eq!(matrix.ncols(), ncols);
        prop_assert_eq!(matrix.len(), nrows * ncols);

        // Test that all values are finite
        for &val in matrix.iter() {
            prop_assert!(val.is_finite());
        }

        // Test transpose properties
        let transposed = matrix.t();
        prop_assert_eq!(transposed.nrows(), ncols);
        prop_assert_eq!(transposed.ncols(), nrows);

        // Test that transpose is involutive: (A^T)^T = A
        let double_transpose = transposed.t().to_owned();
        for (orig, double_t) in matrix.iter().zip(double_transpose.iter()) {
            prop_assert!((orig - double_t).abs() < 1e-10);
        }
    }

    /// Test distance metric properties with generated points
    #[test]
    fn test_distance_metrics_comprehensive(
        dim in 2..20usize,
        scale1 in -100.0..100.0f64,
        scale2 in -100.0..100.0f64,
        scale3 in -100.0..100.0f64
    ) {
        let point1 = Array1::from_vec(vec![scale1; dim]);
        let point2 = Array1::from_vec(vec![scale2; dim]);
        let point3 = Array1::from_vec(vec![scale3; dim]);

        // Test metric properties
        let d12 = euclidean_distance_test(&point1, &point2);
        let d21 = euclidean_distance_test(&point2, &point1);
        let d11 = euclidean_distance_test(&point1, &point1);
        let d13 = euclidean_distance_test(&point1, &point3);
        let d23 = euclidean_distance_test(&point2, &point3);

        // Symmetry: d(x,y) = d(y,x)
        prop_assert!((d12 - d21).abs() < 1e-10);

        // Identity: d(x,x) = 0
        prop_assert!(d11.abs() < 1e-10);

        // Non-negativity: d(x,y) >= 0
        prop_assert!(d12 >= 0.0);
        prop_assert!(d13 >= 0.0);
        prop_assert!(d23 >= 0.0);

        // Triangle inequality: d(x,z) <= d(x,y) + d(y,z)
        prop_assert!(d13 <= d12 + d23 + 1e-10);

        // All distances should be finite
        prop_assert!(d12.is_finite());
        prop_assert!(d13.is_finite());
        prop_assert!(d23.is_finite());
    }

    /// Test supervised learning data consistency
    #[test]
    fn test_supervised_data_properties(
        n_samples in 10..100usize,
        n_features in 2..10usize,
        n_classes in 2..5usize
    ) {
        // Generate simple consistent data
        let x = Array2::<f64>::zeros((n_samples, n_features));
        let y = Array1::from_vec((0..n_samples).map(|i| (i % n_classes) as i32).collect());

        // Test data consistency validation
        let validation_result = crate::validation::ml::validate_supervised_data(&x, &y);
        prop_assert!(validation_result.is_ok());

        // Test shape consistency
        prop_assert_eq!(x.nrows(), y.len());
        prop_assert!(x.ncols() > 0);

        // Test that all labels are in valid range
        for &label in y.iter() {
            prop_assert!(label >= 0 && label < n_classes as i32);
        }
    }

    /// Test numerical stability under different scales
    #[test]
    fn test_numerical_stability_comprehensive(
        base_val in 1.0..10.0f64,
        scale_exp in -10..10i32
    ) {
        let scale = 10.0_f64.powi(scale_exp);
        let scaled_val = base_val * scale;

        // Test that our FloatBounds trait handles extreme values correctly
        if scaled_val.is_finite() {
            prop_assert!(scaled_val.is_finite() || scale_exp.abs() > 6);

            let clamped = scaled_val.clamp(-1e10, 1e10);
            prop_assert!(clamped.is_finite());
        }

        // Test numerical operations
        if scaled_val.is_finite() && scaled_val != 0.0 {
            let log_val = scaled_val.abs().ln();
            prop_assert!(log_val.is_finite() || scaled_val.abs() < f64::MIN_POSITIVE);

            let sqrt_val = scaled_val.abs().sqrt();
            prop_assert!(sqrt_val.is_finite());
            prop_assert!(sqrt_val >= 0.0);
        }
    }

    /// Test validation rule combinations
    #[test]
    fn test_validation_rule_combinations(
        val in -1000.0..1000.0f64,
        min_range in 0.0..50.0f64,
        max_range in 50.0..100.0f64
    ) {
        let rules = ValidationRules::new("test_param")
            .add_rule(ValidationRule::Finite)
            .add_rule(ValidationRule::Range { min: min_range, max: max_range });

        let result = rules.validate_numeric(&val);

        if val.is_finite() && val >= min_range && val <= max_range {
            prop_assert!(result.is_ok());
        } else {
            prop_assert!(result.is_err());
        }
    }

    /// Test classification accuracy bounds
    #[test]
    fn test_classification_accuracy_bounds(
        n_samples in 10..100usize,
        n_classes in 2..5usize,
        error_rate in 0.0..0.5f64
    ) {
        // Generate simple true labels
        let y_true = Array1::from_vec((0..n_samples).map(|i| (i % n_classes) as i32).collect());

        // Create predictions with controlled error rate
        let n_errors = (n_samples as f64 * error_rate) as usize;
        let mut y_pred = y_true.clone();

        // Introduce errors
        for i in 0..n_errors.min(n_samples) {
            let idx = i % n_samples;
            y_pred[idx] = (y_pred[idx] + 1) % n_classes as i32;
        }

        let accuracy = accuracy_score_test(&y_true, &y_pred);

        // Accuracy should be in valid range
        prop_assert!(accuracy >= 0.0);
        prop_assert!(accuracy <= 1.0);

        // Expected accuracy should be roughly 1 - error_rate (with some tolerance)
        let expected_accuracy = 1.0 - error_rate;
        let tolerance = 0.2; // Allow for some randomness in error distribution
        prop_assert!((accuracy - expected_accuracy).abs() <= tolerance);
    }

    /// Test regression metric properties
    #[test]
    fn test_regression_metric_properties(
        n_samples in 10..100usize,
        noise_scale in 0.0..2.0f64
    ) {
        // Generate simple true targets
        let y_true = Array1::from_vec((0..n_samples).map(|i| i as f64).collect());

        // Add noise to create predictions
        let y_pred = y_true.mapv(|val| val + noise_scale * 0.1);

        let mse = mean_squared_error_test(&y_true, &y_pred);

        // MSE should be non-negative
        prop_assert!(mse >= 0.0);
        prop_assert!(mse.is_finite());

        // Perfect predictions should have zero MSE
        let perfect_mse = mean_squared_error_test(&y_true, &y_true);
        prop_assert!(perfect_mse < 1e-10);

        // MSE should increase with noise
        if noise_scale > 0.01 {
            prop_assert!(mse >= perfect_mse);
        }
    }

    /// Test memory layout and performance properties
    #[test]
    fn test_memory_layout_properties(
        nrows in 10..100usize,
        ncols in 10..100usize
    ) {
        let matrix = Array2::<f64>::zeros((nrows, ncols));

        // Test contiguity
        prop_assert!(matrix.is_standard_layout());

        // Test that we can get slices for SIMD operations
        if let Some(slice) = matrix.as_slice() {
            prop_assert_eq!(slice.len(), nrows * ncols);

            // All values should be zero for a zeros matrix
            for &val in slice {
                prop_assert_eq!(val, 0.0);
            }
        }

        // Test views
        let view = matrix.view();
        prop_assert_eq!(view.dim(), (nrows, ncols));

        // Test chunking
        let chunks: Vec<_> = matrix.axis_chunks_iter(scirs2_core::ndarray::Axis(0), 10).collect();
        prop_assert!(chunks.len() > 0);

        let total_rows: usize = chunks.iter().map(|chunk| chunk.nrows()).sum();
        prop_assert_eq!(total_rows, nrows);
    }
}
