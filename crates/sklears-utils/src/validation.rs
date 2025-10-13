//! Input validation utilities

use crate::{UtilsError, UtilsResult};
use scirs2_core::ndarray::{Array1, Array2, ArrayBase, Dimension, OwnedRepr};
use sklears_core::types::{Float, Int};

/// Check that all arrays have consistent first dimension (number of samples)
pub fn check_consistent_length<T>(arrays: &[&Array1<T>]) -> UtilsResult<()> {
    if arrays.is_empty() {
        return Ok(());
    }

    let first_length = arrays[0].len();
    for (_i, array) in arrays.iter().enumerate().skip(1) {
        if array.len() != first_length {
            return Err(UtilsError::ShapeMismatch {
                expected: vec![first_length],
                actual: vec![array.len()],
            });
        }
    }
    Ok(())
}

/// Check that X and y have consistent number of samples (generic version)
pub fn check_consistent_length_xy<T, U>(x: &Array2<T>, y: &Array1<U>) -> UtilsResult<()> {
    if x.nrows() != y.len() {
        return Err(UtilsError::ShapeMismatch {
            expected: vec![x.nrows()],
            actual: vec![y.len()],
        });
    }
    Ok(())
}

/// Check that a 2D array has valid shape and properties
pub fn check_array_2d<T>(array: &Array2<T>) -> UtilsResult<()> {
    check_non_empty(array)?;

    if array.ncols() == 0 {
        return Err(UtilsError::InvalidParameter(
            "Array must have at least one column".to_string(),
        ));
    }

    Ok(())
}

/// Check that X and y have consistent number of samples
pub fn check_x_y(x: &Array2<Float>, y: &Array1<Int>) -> UtilsResult<()> {
    if x.nrows() != y.len() {
        return Err(UtilsError::ShapeMismatch {
            expected: vec![x.nrows()],
            actual: vec![y.len()],
        });
    }

    if x.is_empty() || y.is_empty() {
        return Err(UtilsError::EmptyInput);
    }

    Ok(())
}

/// Check that X and y have consistent number of samples (regression version)
pub fn check_x_y_regression(x: &Array2<Float>, y: &Array1<Float>) -> UtilsResult<()> {
    if x.nrows() != y.len() {
        return Err(UtilsError::ShapeMismatch {
            expected: vec![x.nrows()],
            actual: vec![y.len()],
        });
    }

    if x.is_empty() || y.is_empty() {
        return Err(UtilsError::EmptyInput);
    }

    Ok(())
}

/// Check that an array is not empty
pub fn check_non_empty<T, D: Dimension>(array: &ArrayBase<OwnedRepr<T>, D>) -> UtilsResult<()> {
    if array.is_empty() {
        return Err(UtilsError::EmptyInput);
    }
    Ok(())
}

/// Check that a parameter is positive
pub fn check_positive(value: Float, name: &str) -> UtilsResult<()> {
    if value <= 0.0 {
        return Err(UtilsError::InvalidParameter(format!(
            "{name} must be positive, got {value}"
        )));
    }
    Ok(())
}

/// Check that a parameter is non-negative
pub fn check_non_negative(value: Float, name: &str) -> UtilsResult<()> {
    if value < 0.0 {
        return Err(UtilsError::InvalidParameter(format!(
            "{name} must be non-negative, got {value}"
        )));
    }
    Ok(())
}

/// Check that a parameter is in a valid range
pub fn check_range(value: Float, min: Float, max: Float, name: &str) -> UtilsResult<()> {
    if value < min || value > max {
        return Err(UtilsError::InvalidParameter(format!(
            "{name} must be in range [{min}, {max}], got {value}"
        )));
    }
    Ok(())
}

/// Check that integer parameter is positive
pub fn check_positive_int(value: usize, name: &str) -> UtilsResult<()> {
    if value == 0 {
        return Err(UtilsError::InvalidParameter(format!(
            "{name} must be positive, got {value}"
        )));
    }
    Ok(())
}

/// Check that we have enough samples for the operation
pub fn check_min_samples(n_samples: usize, min_samples: usize) -> UtilsResult<()> {
    if n_samples < min_samples {
        return Err(UtilsError::InsufficientData {
            min: min_samples,
            actual: n_samples,
        });
    }
    Ok(())
}

/// Check that array contains only finite values (no NaN or infinity)
pub fn check_finite(array: &Array2<Float>) -> UtilsResult<()> {
    for &value in array.iter() {
        if !value.is_finite() {
            return Err(UtilsError::InvalidParameter(
                "Array contains non-finite values (NaN or infinity)".to_string(),
            ));
        }
    }
    Ok(())
}

/// Check that array contains only finite values (1D version)
pub fn check_finite_1d(array: &Array1<Float>) -> UtilsResult<()> {
    for &value in array.iter() {
        if !value.is_finite() {
            return Err(UtilsError::InvalidParameter(
                "Array contains non-finite values (NaN or infinity)".to_string(),
            ));
        }
    }
    Ok(())
}

/// Validate feature matrix shape and contents
pub fn validate_features(x: &Array2<Float>) -> UtilsResult<()> {
    check_non_empty(x)?;
    check_finite(x)?;

    if x.ncols() == 0 {
        return Err(UtilsError::InvalidParameter(
            "Feature matrix must have at least one feature".to_string(),
        ));
    }

    Ok(())
}

/// Validate target array
pub fn validate_target(y: &Array1<Int>) -> UtilsResult<()> {
    check_non_empty(y)?;
    Ok(())
}

/// Validate target array (regression version)
pub fn validate_target_regression(y: &Array1<Float>) -> UtilsResult<()> {
    check_non_empty(y)?;
    check_finite_1d(y)?;
    Ok(())
}

/// Check that class labels are valid (non-negative integers)
pub fn validate_class_labels(y: &Array1<Int>) -> UtilsResult<Vec<Int>> {
    validate_target(y)?;

    let mut classes: Vec<Int> = y.iter().copied().collect();
    classes.sort_unstable();
    classes.dedup();

    for &class in &classes {
        if class < 0 {
            return Err(UtilsError::InvalidParameter(format!(
                "Class labels must be non-negative, found {class}"
            )));
        }
    }

    Ok(classes)
}

/// Check that we have at least min_classes distinct classes
pub fn check_min_classes(classes: &[Int], min_classes: usize) -> UtilsResult<()> {
    if classes.len() < min_classes {
        return Err(UtilsError::InvalidParameter(format!(
            "Need at least {min_classes} classes, found {}",
            classes.len()
        )));
    }
    Ok(())
}

/// Validate sample weights
pub fn validate_sample_weights(sample_weight: &Array1<Float>, n_samples: usize) -> UtilsResult<()> {
    if sample_weight.len() != n_samples {
        return Err(UtilsError::ShapeMismatch {
            expected: vec![n_samples],
            actual: vec![sample_weight.len()],
        });
    }

    check_finite_1d(sample_weight)?;

    for &weight in sample_weight.iter() {
        if weight < 0.0 {
            return Err(UtilsError::InvalidParameter(
                "Sample weights must be non-negative".to_string(),
            ));
        }
    }

    if sample_weight.sum() <= 0.0 {
        return Err(UtilsError::InvalidParameter(
            "Sum of sample weights must be positive".to_string(),
        ));
    }

    Ok(())
}

/// Check that matrices have compatible shapes for matrix multiplication
pub fn check_matmul_shapes(a: &Array2<Float>, b: &Array2<Float>) -> UtilsResult<()> {
    if a.ncols() != b.nrows() {
        return Err(UtilsError::ShapeMismatch {
            expected: vec![a.nrows(), a.ncols(), b.ncols()],
            actual: vec![a.nrows(), a.ncols(), b.nrows(), b.ncols()],
        });
    }
    Ok(())
}

/// Validate learning rate parameter
pub fn validate_learning_rate(learning_rate: Float) -> UtilsResult<()> {
    check_positive(learning_rate, "learning_rate")?;
    check_range(learning_rate, 0.0, 1.0, "learning_rate")?;
    Ok(())
}

/// Validate regularization parameter
pub fn validate_regularization(alpha: Float) -> UtilsResult<()> {
    check_non_negative(alpha, "alpha")?;
    Ok(())
}

/// Validate tolerance parameter
pub fn validate_tolerance(tol: Float) -> UtilsResult<()> {
    check_positive(tol, "tol")?;
    Ok(())
}

/// Validate maximum iterations parameter
pub fn validate_max_iter(max_iter: usize) -> UtilsResult<()> {
    check_positive_int(max_iter, "max_iter")?;
    Ok(())
}

/// Validate cross-validation fold indices
pub fn validate_cv_folds(folds: &Array1<i32>, n_samples: usize, n_folds: usize) -> UtilsResult<()> {
    if folds.len() != n_samples {
        return Err(UtilsError::ShapeMismatch {
            expected: vec![n_samples],
            actual: vec![folds.len()],
        });
    }

    // Check that fold indices are in valid range
    for &fold_idx in folds.iter() {
        if fold_idx < 0 || fold_idx >= n_folds as i32 {
            return Err(UtilsError::InvalidParameter(format!(
                "Fold index {fold_idx} is out of range [0, {n_folds})"
            )));
        }
    }

    // Check that all folds are represented
    let mut fold_counts = vec![0; n_folds];
    for &fold_idx in folds.iter() {
        fold_counts[fold_idx as usize] += 1;
    }

    for (i, &count) in fold_counts.iter().enumerate() {
        if count == 0 {
            return Err(UtilsError::InvalidParameter(format!(
                "Fold {i} has no samples assigned"
            )));
        }
    }

    Ok(())
}

/// Validate feature importance values
pub fn validate_feature_importance(
    importance: &Array1<Float>,
    n_features: usize,
) -> UtilsResult<()> {
    if importance.len() != n_features {
        return Err(UtilsError::ShapeMismatch {
            expected: vec![n_features],
            actual: vec![importance.len()],
        });
    }

    // Check for non-negative values
    for (i, &value) in importance.iter().enumerate() {
        if value < 0.0 {
            return Err(UtilsError::InvalidParameter(format!(
                "Feature importance at index {i} is negative: {value}"
            )));
        }
        if !value.is_finite() {
            return Err(UtilsError::InvalidParameter(format!(
                "Feature importance at index {i} is not finite: {value}"
            )));
        }
    }

    // Check if all importance values are zero (usually indicates an error)
    if importance.iter().all(|&x| x == 0.0) {
        return Err(UtilsError::InvalidParameter(
            "All feature importance values are zero".to_string(),
        ));
    }

    Ok(())
}

/// Validate model prediction format for classification
pub fn validate_classification_predictions(
    predictions: &Array1<i32>,
    n_samples: usize,
    valid_classes: &[i32],
) -> UtilsResult<()> {
    if predictions.len() != n_samples {
        return Err(UtilsError::ShapeMismatch {
            expected: vec![n_samples],
            actual: vec![predictions.len()],
        });
    }

    // Check that all predictions are valid class labels
    for (i, &pred) in predictions.iter().enumerate() {
        if !valid_classes.contains(&pred) {
            return Err(UtilsError::InvalidParameter(format!(
                "Prediction at index {i} ({pred}) is not a valid class label"
            )));
        }
    }

    Ok(())
}

/// Validate model prediction format for regression
pub fn validate_regression_predictions(
    predictions: &Array1<Float>,
    n_samples: usize,
) -> UtilsResult<()> {
    if predictions.len() != n_samples {
        return Err(UtilsError::ShapeMismatch {
            expected: vec![n_samples],
            actual: vec![predictions.len()],
        });
    }

    // Check for finite values
    for (i, &value) in predictions.iter().enumerate() {
        if !value.is_finite() {
            return Err(UtilsError::InvalidParameter(format!(
                "Prediction at index {i} is not finite: {value}"
            )));
        }
    }

    Ok(())
}

/// Validate sparse matrix properties
pub fn validate_sparse_matrix(
    data: &Array1<Float>,
    indices: &Array1<usize>,
    indptr: &Array1<usize>,
    n_rows: usize,
    n_cols: usize,
) -> UtilsResult<()> {
    // Check basic consistency
    if data.len() != indices.len() {
        return Err(UtilsError::ShapeMismatch {
            expected: vec![data.len()],
            actual: vec![indices.len()],
        });
    }

    if indptr.len() != n_rows + 1 {
        return Err(UtilsError::ShapeMismatch {
            expected: vec![n_rows + 1],
            actual: vec![indptr.len()],
        });
    }

    // Check indptr is non-decreasing and starts at 0
    if indptr[0] != 0 {
        return Err(UtilsError::InvalidParameter(
            "indptr must start with 0".to_string(),
        ));
    }

    for i in 1..indptr.len() {
        if indptr[i] < indptr[i - 1] {
            return Err(UtilsError::InvalidParameter(
                "indptr must be non-decreasing".to_string(),
            ));
        }
    }

    // Check that last indptr value matches data length
    if indptr[indptr.len() - 1] != data.len() {
        return Err(UtilsError::InvalidParameter(
            "Last indptr value must equal data length".to_string(),
        ));
    }

    // Check column indices are valid
    for &col_idx in indices.iter() {
        if col_idx >= n_cols {
            return Err(UtilsError::InvalidParameter(format!(
                "Column index {col_idx} is out of bounds for matrix with {n_cols} columns"
            )));
        }
    }

    // Check for finite data values
    for (i, &value) in data.iter().enumerate() {
        if !value.is_finite() {
            return Err(UtilsError::InvalidParameter(format!(
                "Data value at index {i} is not finite: {value}"
            )));
        }
    }

    Ok(())
}

/// Validate time series data for temporal consistency
pub fn validate_time_series(
    data: &Array2<Float>,
    timestamps: &Array1<Float>,
    min_samples: usize,
) -> UtilsResult<()> {
    // Check consistent dimensions
    if data.nrows() != timestamps.len() {
        return Err(UtilsError::ShapeMismatch {
            expected: vec![data.nrows()],
            actual: vec![timestamps.len()],
        });
    }

    // Check minimum number of samples
    if data.nrows() < min_samples {
        return Err(UtilsError::InsufficientData {
            min: min_samples,
            actual: data.nrows(),
        });
    }

    // Check that timestamps are strictly increasing
    for i in 1..timestamps.len() {
        if timestamps[i] <= timestamps[i - 1] {
            return Err(UtilsError::InvalidParameter(format!(
                "Timestamps must be strictly increasing. Found {} <= {} at index {}",
                timestamps[i],
                timestamps[i - 1],
                i
            )));
        }
    }

    // Check for finite values in both data and timestamps
    for (i, &ts) in timestamps.iter().enumerate() {
        if !ts.is_finite() {
            return Err(UtilsError::InvalidParameter(format!(
                "Timestamp at index {i} is not finite: {ts}"
            )));
        }
    }

    for ((i, j), &value) in data.indexed_iter() {
        if !value.is_finite() {
            return Err(UtilsError::InvalidParameter(format!(
                "Data value at index ({i}, {j}) is not finite: {value}"
            )));
        }
    }

    Ok(())
}

/// Validate probability distribution (must sum to 1, all non-negative)
pub fn validate_probability_distribution(
    probabilities: &Array1<Float>,
    tolerance: Float,
) -> UtilsResult<()> {
    // Check for non-negative values
    for (i, &prob) in probabilities.iter().enumerate() {
        if prob < 0.0 {
            return Err(UtilsError::InvalidParameter(format!(
                "Probability at index {i} is negative: {prob}"
            )));
        }
        if !prob.is_finite() {
            return Err(UtilsError::InvalidParameter(format!(
                "Probability at index {i} is not finite: {prob}"
            )));
        }
    }

    // Check that probabilities sum to 1
    let sum: Float = probabilities.sum();
    if (sum - 1.0).abs() > tolerance {
        return Err(UtilsError::InvalidParameter(format!(
            "Probabilities must sum to 1.0 (±{tolerance}), got {sum}"
        )));
    }

    Ok(())
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{array, Array2};

    #[test]
    fn test_check_consistent_length() {
        let a = array![1, 2, 3];
        let b = array![4, 5, 6];
        let c = array![7, 8];

        assert!(check_consistent_length(&[&a, &b]).is_ok());
        assert!(check_consistent_length(&[&a, &c]).is_err());
    }

    #[test]
    fn test_check_x_y() {
        let x = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let y_good = array![0, 1, 0];
        let y_bad = array![0, 1];

        assert!(check_x_y(&x, &y_good).is_ok());
        assert!(check_x_y(&x, &y_bad).is_err());
    }

    #[test]
    fn test_check_positive() {
        assert!(check_positive(1.0, "test").is_ok());
        assert!(check_positive(0.0, "test").is_err());
        assert!(check_positive(-1.0, "test").is_err());
    }

    #[test]
    fn test_check_range() {
        assert!(check_range(0.5, 0.0, 1.0, "test").is_ok());
        assert!(check_range(-0.1, 0.0, 1.0, "test").is_err());
        assert!(check_range(1.1, 0.0, 1.0, "test").is_err());
    }

    #[test]
    fn test_validate_class_labels() {
        let y_good = array![0, 1, 2, 1, 0];
        let y_bad = array![0, 1, -1, 1, 0];

        let classes = validate_class_labels(&y_good).unwrap();
        assert_eq!(classes, vec![0, 1, 2]);

        assert!(validate_class_labels(&y_bad).is_err());
    }

    #[test]
    fn test_check_finite() {
        let good = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let bad = Array2::from_shape_vec((2, 2), vec![1.0, Float::NAN, 3.0, 4.0]).unwrap();

        assert!(check_finite(&good).is_ok());
        assert!(check_finite(&bad).is_err());
    }

    #[test]
    fn test_validate_sample_weights() {
        let good_weights = array![1.0, 2.0, 1.5];
        let bad_weights = array![1.0, -1.0, 1.5];
        let zero_sum_weights = array![0.0, 0.0, 0.0];

        assert!(validate_sample_weights(&good_weights, 3).is_ok());
        assert!(validate_sample_weights(&bad_weights, 3).is_err());
        assert!(validate_sample_weights(&zero_sum_weights, 3).is_err());
    }

    #[test]
    fn test_validate_cv_folds() {
        let good_folds = array![0, 1, 2, 0, 1, 2];
        let bad_folds_range = array![0, 1, 3, 0, 1, 2]; // 3 is out of range for 3 folds
        let bad_folds_missing = array![0, 0, 1, 1, 1, 1]; // missing fold 2

        assert!(validate_cv_folds(&good_folds, 6, 3).is_ok());
        assert!(validate_cv_folds(&bad_folds_range, 6, 3).is_err());
        assert!(validate_cv_folds(&bad_folds_missing, 6, 3).is_err());
    }

    #[test]
    fn test_validate_feature_importance() {
        let good_importance = array![0.5, 0.3, 0.2];
        let bad_importance_negative = array![0.5, -0.1, 0.2];
        let bad_importance_all_zero = array![0.0, 0.0, 0.0];

        assert!(validate_feature_importance(&good_importance, 3).is_ok());
        assert!(validate_feature_importance(&bad_importance_negative, 3).is_err());
        assert!(validate_feature_importance(&bad_importance_all_zero, 3).is_err());
    }

    #[test]
    fn test_validate_classification_predictions() {
        let good_predictions = array![0, 1, 2, 1, 0];
        let bad_predictions = array![0, 1, 3, 1, 0]; // 3 is not a valid class
        let valid_classes = vec![0, 1, 2];

        assert!(validate_classification_predictions(&good_predictions, 5, &valid_classes).is_ok());
        assert!(validate_classification_predictions(&bad_predictions, 5, &valid_classes).is_err());
    }

    #[test]
    fn test_validate_regression_predictions() {
        let good_predictions = array![1.5, 2.3, -0.5, 10.0];
        let bad_predictions = array![1.5, Float::NAN, -0.5, 10.0];

        assert!(validate_regression_predictions(&good_predictions, 4).is_ok());
        assert!(validate_regression_predictions(&bad_predictions, 4).is_err());
    }

    #[test]
    fn test_validate_sparse_matrix() {
        // Valid CSR matrix: [[1, 0, 2], [0, 0, 3], [4, 5, 6]]
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let indices = array![0, 2, 2, 0, 1, 2];
        let indptr = array![0, 2, 3, 6];

        assert!(validate_sparse_matrix(&data, &indices, &indptr, 3, 3).is_ok());

        // Invalid: indptr doesn't start with 0
        let bad_indptr = array![1, 2, 3, 6];
        assert!(validate_sparse_matrix(&data, &indices, &bad_indptr, 3, 3).is_err());
    }

    #[test]
    fn test_validate_time_series() {
        let data = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let good_timestamps = array![1.0, 2.0, 3.0];
        let bad_timestamps = array![1.0, 1.5, 1.2]; // not increasing

        assert!(validate_time_series(&data, &good_timestamps, 2).is_ok());
        assert!(validate_time_series(&data, &bad_timestamps, 2).is_err());
    }

    #[test]
    fn test_validate_probability_distribution() {
        let good_probs = array![0.3, 0.5, 0.2];
        let bad_probs_negative = array![0.3, -0.1, 0.8];
        let bad_probs_sum = array![0.3, 0.5, 0.3]; // sums to 1.1

        assert!(validate_probability_distribution(&good_probs, 1e-6).is_ok());
        assert!(validate_probability_distribution(&bad_probs_negative, 1e-6).is_err());
        assert!(validate_probability_distribution(&bad_probs_sum, 1e-6).is_err());
    }
}
