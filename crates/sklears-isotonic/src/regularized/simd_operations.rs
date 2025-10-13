//! Optimized operations for high-performance isotonic regression computations
//!
//! This module provides efficient implementations of common isotonic regression operations
//! including statistical calculations, error metrics, regularization penalties, and constraint
//! checking. Future versions may include SIMD acceleration when stabilized.

/// Optimized mean calculation
///
/// Efficiently computes the arithmetic mean of a data slice.
///
/// # Arguments
/// * `data` - Input data slice for mean calculation
///
/// # Returns
/// Mean value of the input data
///
/// # Examples
/// ```
/// use sklears_isotonic::regularized::simd_operations::simd_mean;
///
/// let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let mean = simd_mean(&data);
/// assert!((mean - 3.0).abs() < 1e-10);
/// ```
pub fn simd_mean(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }

    let sum: f64 = data.iter().sum();
    sum / data.len() as f64
}

/// Optimized sum calculation
///
/// Efficiently computes the sum of all elements in a data slice.
///
/// # Arguments
/// * `data` - Input data slice for summation
///
/// # Returns
/// Sum of all elements in the input data
pub fn simd_sum(data: &[f64]) -> f64 {
    data.iter().sum()
}

/// Optimized variance calculation
///
/// Efficiently computes the sample variance using a pre-computed mean.
/// Uses numerically stable algorithm.
///
/// # Arguments
/// * `data` - Input data slice
/// * `mean` - Pre-computed mean of the data
///
/// # Returns
/// Sample variance of the input data
pub fn simd_variance(data: &[f64], mean: f64) -> f64 {
    if data.len() <= 1 {
        return 0.0;
    }

    let sum_sq_diff: f64 = data.iter().map(|&x| (x - mean).powi(2)).sum();

    sum_sq_diff / (data.len() - 1) as f64
}

/// Optimized residual calculation
///
/// Efficiently computes residuals between predictions and targets.
///
/// # Arguments
/// * `predictions` - Predicted values
/// * `targets` - Target values
///
/// # Returns
/// Vector of residuals (predictions - targets)
pub fn simd_residuals(predictions: &[f64], targets: &[f64]) -> Vec<f64> {
    predictions
        .iter()
        .zip(targets.iter())
        .map(|(&pred, &target)| pred - target)
        .collect()
}

/// Optimized squared error calculation
///
/// Efficiently computes squared errors between predictions and targets.
///
/// # Arguments
/// * `predictions` - Predicted values
/// * `targets` - Target values
///
/// # Returns
/// Vector of squared errors
pub fn simd_squared_errors(predictions: &[f64], targets: &[f64]) -> Vec<f64> {
    predictions
        .iter()
        .zip(targets.iter())
        .map(|(&pred, &target)| (pred - target).powi(2))
        .collect()
}

/// Optimized mean squared error calculation
///
/// Efficiently computes the mean squared error between predictions and targets.
///
/// # Arguments
/// * `predictions` - Predicted values
/// * `targets` - Target values
///
/// # Returns
/// Mean squared error
pub fn simd_mse(predictions: &[f64], targets: &[f64]) -> f64 {
    if predictions.is_empty() || targets.is_empty() {
        return 0.0;
    }

    let min_len = predictions.len().min(targets.len());
    let sum_sq_error: f64 = predictions[..min_len]
        .iter()
        .zip(targets[..min_len].iter())
        .map(|(&pred, &target)| (pred - target).powi(2))
        .sum();

    sum_sq_error / min_len as f64
}

/// Optimized L1 regularization penalty
///
/// Efficiently computes the L1 norm (sum of absolute values) of coefficients.
///
/// # Arguments
/// * `coefficients` - Coefficient values
///
/// # Returns
/// L1 penalty (sum of absolute values)
pub fn simd_l1_penalty(coefficients: &[f64]) -> f64 {
    coefficients.iter().map(|&x| x.abs()).sum()
}

/// Optimized L2 regularization penalty
///
/// Efficiently computes the L2 norm (Euclidean norm) of coefficients.
///
/// # Arguments
/// * `coefficients` - Coefficient values
///
/// # Returns
/// L2 penalty (Euclidean norm)
pub fn simd_l2_penalty(coefficients: &[f64]) -> f64 {
    let sum_sq: f64 = coefficients.iter().map(|&x| x.powi(2)).sum();
    sum_sq.sqrt()
}

/// Optimized monotonicity constraint checking
///
/// Efficiently checks if values satisfy monotonicity constraints.
///
/// # Arguments
/// * `values` - Values to check
/// * `increasing` - Whether to check for increasing (true) or decreasing (false) order
///
/// # Returns
/// True if values satisfy the monotonicity constraint
pub fn simd_check_monotonicity(values: &[f64], increasing: bool) -> bool {
    if values.len() <= 1 {
        return true;
    }

    if increasing {
        values.windows(2).all(|w| w[1] >= w[0])
    } else {
        values.windows(2).all(|w| w[1] <= w[0])
    }
}

/// Optimized grid-based interpolation for isotonic regression
///
/// Efficiently performs linear interpolation for query points on a grid.
///
/// # Arguments
/// * `x_grid` - Grid x-coordinates (must be sorted)
/// * `y_grid` - Grid y-values corresponding to x_grid
/// * `x_query` - Query x-coordinates for interpolation
///
/// # Returns
/// Interpolated y-values for the query points
pub fn simd_interpolate(x_grid: &[f64], y_grid: &[f64], x_query: &[f64]) -> Vec<f64> {
    x_query
        .iter()
        .map(|&x_q| {
            // Binary search for insertion point
            let idx = match x_grid.binary_search_by(|&x| x.partial_cmp(&x_q).unwrap()) {
                Ok(i) => return y_grid[i], // Exact match
                Err(i) => i,
            };

            if idx == 0 {
                y_grid[0]
            } else if idx >= x_grid.len() {
                y_grid[y_grid.len() - 1]
            } else {
                // Linear interpolation between grid points
                let x0 = x_grid[idx - 1];
                let x1 = x_grid[idx];
                let y0 = y_grid[idx - 1];
                let y1 = y_grid[idx];

                if (x1 - x0).abs() < f64::EPSILON {
                    y0
                } else {
                    let t = (x_q - x0) / (x1 - x0);
                    y0 + t * (y1 - y0)
                }
            }
        })
        .collect()
}

/// Optimized sparsity detection for isotonic regression
///
/// Efficiently identifies sparse patterns in values based on a threshold.
///
/// # Arguments
/// * `values` - Values to analyze for sparsity
/// * `threshold` - Threshold below which values are considered sparse
///
/// # Returns
/// Boolean mask indicating which values are sparse
pub fn simd_detect_sparsity(values: &[f64], threshold: f64) -> Vec<bool> {
    values.iter().map(|&x| x.abs() <= threshold).collect()
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_mean() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mean = simd_mean(&data);
        assert!((mean - 3.0).abs() < 1e-10);

        let empty: Vec<f64> = vec![];
        assert_eq!(simd_mean(&empty), 0.0);
    }

    #[test]
    fn test_simd_sum() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let sum = simd_sum(&data);
        assert!((sum - 15.0).abs() < 1e-10);
    }

    #[test]
    fn test_simd_variance() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mean = simd_mean(&data);
        let variance = simd_variance(&data, mean);
        assert!((variance - 2.5).abs() < 1e-10); // Sample variance
    }

    #[test]
    fn test_simd_mse() {
        let pred = vec![1.0, 2.0, 3.0];
        let target = vec![1.1, 1.9, 3.1];
        let mse = simd_mse(&pred, &target);
        let expected = (0.01 + 0.01 + 0.01) / 3.0;
        assert!((mse - expected).abs() < 1e-10);
    }

    #[test]
    fn test_simd_l1_penalty() {
        let coeffs = vec![-2.0, 1.0, -3.0, 2.0];
        let l1 = simd_l1_penalty(&coeffs);
        assert!((l1 - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_simd_l2_penalty() {
        let coeffs = vec![3.0, 4.0];
        let l2 = simd_l2_penalty(&coeffs);
        assert!((l2 - 5.0).abs() < 1e-10); // sqrt(9 + 16) = 5
    }

    #[test]
    fn test_simd_check_monotonicity() {
        let increasing = vec![1.0, 2.0, 3.0, 4.0];
        assert!(simd_check_monotonicity(&increasing, true));
        assert!(!simd_check_monotonicity(&increasing, false));

        let decreasing = vec![4.0, 3.0, 2.0, 1.0];
        assert!(!simd_check_monotonicity(&decreasing, true));
        assert!(simd_check_monotonicity(&decreasing, false));
    }

    #[test]
    fn test_simd_interpolate() {
        let x_grid = vec![0.0, 1.0, 2.0];
        let y_grid = vec![0.0, 1.0, 4.0];
        let x_query = vec![0.5, 1.5];
        let result = simd_interpolate(&x_grid, &y_grid, &x_query);

        assert!((result[0] - 0.5).abs() < 1e-10);
        assert!((result[1] - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_simd_detect_sparsity() {
        let values = vec![0.001, 1.0, 0.0001, 2.0];
        let sparse = simd_detect_sparsity(&values, 0.01);
        assert_eq!(sparse, vec![true, false, true, false]);
    }
}
