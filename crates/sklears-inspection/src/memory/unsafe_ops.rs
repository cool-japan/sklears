//! High-performance unsafe operations for critical performance paths
//!
//! This module contains unsafe functions optimized for maximum performance
//! in computation-intensive explanation algorithms. All functions include
//! comprehensive safety documentation and should be used with extreme care.

use super::layout_manager::{ExplanationDataLayout, MemoryLayoutManager};
use crate::types::*;
use crate::SklResult;
// ✅ SciRS2 Policy Compliant Import
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use scirs2_core::random::SeedableRng;

/// Cache-friendly feature importance computation with unsafe optimizations
pub fn compute_feature_importance_unsafe<F>(
    model_fn: &F,
    X: &ArrayView2<Float>,
    y: &ArrayView1<Float>,
    n_repeats: usize,
    random_state: Option<u64>,
) -> SklResult<Array1<Float>>
where
    F: Fn(&ArrayView2<Float>) -> Array1<Float>,
{
    let n_features = X.ncols();
    let n_samples = X.nrows();

    // Allocate aligned memory for better cache performance
    let layout = ExplanationDataLayout {
        feature_major: true,
        block_size: 64,
        alignment: 64,
    };
    let memory_manager = MemoryLayoutManager::new(layout);

    // Get baseline score
    let baseline_predictions = model_fn(X);
    let baseline_score =
        unsafe { compute_r2_score_unsafe(y.as_ptr(), baseline_predictions.as_ptr(), n_samples) };

    // Allocate memory for importance scores
    let mut importance_scores = memory_manager.allocate_aligned(n_features);

    // Compute importance for each feature
    for feature_idx in 0..n_features {
        let mut feature_importance = 0.0;

        for _ in 0..n_repeats {
            // Create permuted version of X
            let mut X_permuted = X.to_owned();

            // Permute the feature column using unsafe for better performance
            unsafe {
                permute_column_unsafe(&mut X_permuted, feature_idx, random_state);
            }

            // Get predictions on permuted data
            let permuted_predictions = model_fn(&X_permuted.view());

            // Compute score on permuted data
            let permuted_score = unsafe {
                compute_r2_score_unsafe(y.as_ptr(), permuted_predictions.as_ptr(), n_samples)
            };

            // Add to importance score
            feature_importance += baseline_score - permuted_score;
        }

        // Average over repeats
        importance_scores[feature_idx] = feature_importance / n_repeats as Float;
    }

    // Convert to ndarray
    let importance_array = Array1::from_vec(importance_scores);

    Ok(importance_array)
}

/// Unsafe R2 score computation for maximum performance
///
/// # Safety
///
/// This function is safe when:
/// - `y_true` and `y_pred` are valid pointers
/// - Both arrays have at least `n_samples` elements
unsafe fn compute_r2_score_unsafe(
    y_true: *const Float,
    y_pred: *const Float,
    n_samples: usize,
) -> Float {
    // Compute mean of true values
    let mut y_mean = 0.0;
    for i in 0..n_samples {
        y_mean += *y_true.add(i);
    }
    y_mean /= n_samples as Float;

    // Compute total sum of squares and residual sum of squares
    let mut ss_tot = 0.0;
    let mut ss_res = 0.0;

    for i in 0..n_samples {
        let y_i = *y_true.add(i);
        let pred_i = *y_pred.add(i);

        ss_tot += (y_i - y_mean) * (y_i - y_mean);
        ss_res += (y_i - pred_i) * (y_i - pred_i);
    }

    if ss_tot == 0.0 {
        return 0.0;
    }

    1.0 - (ss_res / ss_tot)
}

/// Unsafe column permutation for maximum performance
///
/// # Safety
///
/// This function is safe when:
/// - `X` is a valid mutable array
/// - `feature_idx` is a valid column index
unsafe fn permute_column_unsafe(
    X: &mut Array2<Float>,
    feature_idx: usize,
    random_state: Option<u64>,
) {
    let n_samples = X.nrows();
    let mut rng = match random_state {
        Some(seed) => scirs2_core::random::rngs::StdRng::seed_from_u64(seed),
        None => scirs2_core::random::rngs::StdRng::seed_from_u64(42),
    };

    // Get mutable pointer to the column data
    let column_ptr = X.column_mut(feature_idx).as_mut_ptr();

    // Fisher-Yates shuffle using unsafe for better performance
    for i in (1..n_samples).rev() {
        let j = scirs2_core::random::Rng::gen_range(&mut rng, 0..=i);
        if i != j {
            let temp = *column_ptr.add(i);
            *column_ptr.add(i) = *column_ptr.add(j);
            *column_ptr.add(j) = temp;
        }
    }
}

/// Unsafe vectorized array operations for maximum performance
pub struct UnsafeArrayOps;

impl UnsafeArrayOps {
    /// Unsafe fast element-wise multiplication
    ///
    /// # Safety
    ///
    /// This function is safe when:
    /// - `a`, `b`, and `result` are valid pointers
    /// - All arrays have at least `len` elements
    /// - Arrays do not overlap unless `result` is the same as `a` or `b`
    pub unsafe fn fast_multiply(a: *const Float, b: *const Float, result: *mut Float, len: usize) {
        for i in 0..len {
            *result.add(i) = *a.add(i) * *b.add(i);
        }
    }

    /// Unsafe fast array sum
    ///
    /// # Safety
    ///
    /// This function is safe when:
    /// - `array` is a valid pointer
    /// - Array has at least `len` elements
    pub unsafe fn fast_sum(array: *const Float, len: usize) -> Float {
        let mut sum = 0.0;
        for i in 0..len {
            sum += *array.add(i);
        }
        sum
    }

    /// Unsafe fast array scaling (multiply by scalar)
    ///
    /// # Safety
    ///
    /// This function is safe when:
    /// - `array` and `result` are valid pointers
    /// - Both arrays have at least `len` elements
    /// - Arrays do not overlap unless they are the same
    pub unsafe fn fast_scale(array: *const Float, scalar: Float, result: *mut Float, len: usize) {
        for i in 0..len {
            *result.add(i) = *array.add(i) * scalar;
        }
    }

    /// Unsafe fast array difference
    ///
    /// # Safety
    ///
    /// This function is safe when:
    /// - `a`, `b`, and `result` are valid pointers
    /// - All arrays have at least `len` elements
    /// - Arrays do not overlap unless `result` is the same as `a` or `b`
    pub unsafe fn fast_subtract(a: *const Float, b: *const Float, result: *mut Float, len: usize) {
        for i in 0..len {
            *result.add(i) = *a.add(i) - *b.add(i);
        }
    }

    /// Unsafe fast mean computation
    ///
    /// # Safety
    ///
    /// This function is safe when:
    /// - `array` is a valid pointer
    /// - Array has at least `len` elements
    /// - `len` is greater than 0
    pub unsafe fn fast_mean(array: *const Float, len: usize) -> Float {
        if len == 0 {
            return 0.0;
        }
        Self::fast_sum(array, len) / len as Float
    }

    /// Unsafe fast variance computation
    ///
    /// # Safety
    ///
    /// This function is safe when:
    /// - `array` is a valid pointer
    /// - Array has at least `len` elements
    /// - `len` is greater than 0
    pub unsafe fn fast_variance(array: *const Float, len: usize) -> Float {
        if len == 0 {
            return 0.0;
        }

        let mean = Self::fast_mean(array, len);
        let mut variance = 0.0;

        for i in 0..len {
            let diff = *array.add(i) - mean;
            variance += diff * diff;
        }

        variance / len as Float
    }

    /// Unsafe fast standard deviation computation
    ///
    /// # Safety
    ///
    /// This function is safe when:
    /// - `array` is a valid pointer
    /// - Array has at least `len` elements
    /// - `len` is greater than 0
    pub unsafe fn fast_std(array: *const Float, len: usize) -> Float {
        Self::fast_variance(array, len).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // ✅ SciRS2 Policy Compliant Import
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_unsafe_r2_score() {
        let y_true = vec![1.0, 2.0, 3.0, 4.0];
        let y_pred = vec![1.1, 1.9, 3.1, 3.9];

        let r2 = unsafe { compute_r2_score_unsafe(y_true.as_ptr(), y_pred.as_ptr(), y_true.len()) };

        assert!(r2 > 0.9); // Should be high for good predictions
        assert!(r2 <= 1.0);
    }

    #[test]
    fn test_unsafe_column_permutation() {
        let mut X = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let original_col0 = X.column(0).to_owned();

        unsafe {
            permute_column_unsafe(&mut X, 0, Some(42)); // Fixed seed for reproducibility
        }

        // Column should be permuted (different order, but same values)
        let permuted_col0 = X.column(0);

        // Check that all original values are still present
        let mut original_sorted = original_col0.to_vec();
        let mut permuted_sorted = permuted_col0.to_vec();
        original_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        permuted_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        for (orig, perm) in original_sorted.iter().zip(permuted_sorted.iter()) {
            assert_abs_diff_eq!(*orig, *perm, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_unsafe_array_ops() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![2.0, 3.0, 4.0, 5.0];
        let mut result = vec![0.0; 4];

        unsafe {
            // Test multiplication
            UnsafeArrayOps::fast_multiply(a.as_ptr(), b.as_ptr(), result.as_mut_ptr(), 4);
            assert_abs_diff_eq!(result[0], 2.0, epsilon = 1e-10);
            assert_abs_diff_eq!(result[1], 6.0, epsilon = 1e-10);
            assert_abs_diff_eq!(result[2], 12.0, epsilon = 1e-10);
            assert_abs_diff_eq!(result[3], 20.0, epsilon = 1e-10);

            // Test sum
            let sum = UnsafeArrayOps::fast_sum(a.as_ptr(), 4);
            assert_abs_diff_eq!(sum, 10.0, epsilon = 1e-10);

            // Test scaling
            UnsafeArrayOps::fast_scale(a.as_ptr(), 2.0, result.as_mut_ptr(), 4);
            assert_abs_diff_eq!(result[0], 2.0, epsilon = 1e-10);
            assert_abs_diff_eq!(result[1], 4.0, epsilon = 1e-10);
            assert_abs_diff_eq!(result[2], 6.0, epsilon = 1e-10);
            assert_abs_diff_eq!(result[3], 8.0, epsilon = 1e-10);

            // Test subtraction
            UnsafeArrayOps::fast_subtract(b.as_ptr(), a.as_ptr(), result.as_mut_ptr(), 4);
            assert_abs_diff_eq!(result[0], 1.0, epsilon = 1e-10);
            assert_abs_diff_eq!(result[1], 1.0, epsilon = 1e-10);
            assert_abs_diff_eq!(result[2], 1.0, epsilon = 1e-10);
            assert_abs_diff_eq!(result[3], 1.0, epsilon = 1e-10);

            // Test mean
            let mean = UnsafeArrayOps::fast_mean(a.as_ptr(), 4);
            assert_abs_diff_eq!(mean, 2.5, epsilon = 1e-10);

            // Test variance
            let variance = UnsafeArrayOps::fast_variance(a.as_ptr(), 4);
            assert!(variance > 0.0);

            // Test standard deviation
            let std_dev = UnsafeArrayOps::fast_std(a.as_ptr(), 4);
            assert_abs_diff_eq!(std_dev, variance.sqrt(), epsilon = 1e-10);
        }
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_feature_importance_unsafe() {
        let X = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let y = array![3.0, 7.0, 11.0]; // y = x1 + x2

        let model = |x: &ArrayView2<Float>| -> Array1<Float> {
            // Simple model that sums the features
            x.sum_axis(scirs2_core::ndarray::Axis(1))
        };

        let importances = compute_feature_importance_unsafe(
            &model,
            &X.view(),
            &y.view(),
            3,        // n_repeats
            Some(42), // random_state
        )
        .unwrap();

        assert_eq!(importances.len(), 2);
        // Both features should have some importance since they both contribute
        assert!(importances.iter().any(|&x| x.abs() > 0.0));
    }

    #[test]
    fn test_edge_cases() {
        unsafe {
            // Test with empty arrays (should not crash)
            let empty_r2 = compute_r2_score_unsafe([].as_ptr(), [].as_ptr(), 0);
            assert!(empty_r2.is_finite());

            // Test with single element
            let single_true = [1.0];
            let single_pred = [1.0];
            let single_r2 = compute_r2_score_unsafe(single_true.as_ptr(), single_pred.as_ptr(), 1);
            assert!(single_r2.is_finite());
        }
    }
}
