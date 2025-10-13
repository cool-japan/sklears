//! Pool Adjacent Violators Algorithm (PAVA) implementations
//!
//! This module provides efficient implementations of the Pool Adjacent Violators
//! algorithm for different loss functions including L2, L1, Huber, and quantile losses.

use scirs2_core::ndarray::Array1;
use sklears_core::{
    error::{Result, SklearsError},
    types::Float,
};

/// Pool Adjacent Violators Algorithm for L2 (squared) loss
///
/// This is the standard PAVA algorithm that minimizes the sum of squared deviations
/// subject to monotonicity constraints.
pub fn pool_adjacent_violators_l2(
    y: &Array1<Float>,
    weights: Option<&Array1<Float>>,
    increasing: bool,
) -> Result<Array1<Float>> {
    if y.is_empty() {
        return Ok(Array1::zeros(0));
    }

    let n = y.len();
    let mut result = y.clone();
    let w = if let Some(weights) = weights {
        weights.clone()
    } else {
        Array1::ones(n)
    };

    if !increasing {
        // For decreasing, we negate the values, run increasing PAVA, then negate back
        result.mapv_inplace(|x| -x);
    }

    // PAVA algorithm for increasing constraint
    let mut i = 0;
    while i < result.len() - 1 {
        // Find violating region
        if result[i] > result[i + 1] {
            let mut j = i;
            let mut sum_wy = w[i] * result[i];
            let mut sum_w = w[i];

            // Extend the violating region
            while j < result.len() - 1 && result[j] > result[j + 1] {
                j += 1;
                sum_wy += w[j] * result[j];
                sum_w += w[j];
            }

            // Check if we need to extend further back
            let avg = sum_wy / sum_w;
            while i > 0 && result[i - 1] > avg {
                i -= 1;
                sum_wy += w[i] * result[i];
                sum_w += w[i];
            }

            // Update the average
            let new_avg = sum_wy / sum_w;

            // Set all values in the violating region to the weighted average
            for k in i..=j {
                result[k] = new_avg;
            }

            // Continue from the beginning of the fixed region
            i = i.saturating_sub(1);
        } else {
            i += 1;
        }
    }

    if !increasing {
        // Negate back for decreasing constraint
        result.mapv_inplace(|x| -x);
    }

    Ok(result)
}

/// Pool Adjacent Violators Algorithm for L1 (absolute) loss
///
/// This implements PAVA for L1 loss by finding weighted medians.
pub fn pool_adjacent_violators_l1(
    y: &Array1<Float>,
    weights: Option<&Array1<Float>>,
    increasing: bool,
) -> Result<Array1<Float>> {
    if y.is_empty() {
        return Ok(Array1::zeros(0));
    }

    let n = y.len();
    let mut result = y.clone();
    let w = if let Some(weights) = weights {
        weights.clone()
    } else {
        Array1::ones(n)
    };

    if !increasing {
        result.mapv_inplace(|x| -x);
    }

    let mut i = 0;
    while i < result.len() - 1 {
        if result[i] > result[i + 1] {
            let mut j = i;

            // Find the violating region
            while j < result.len() - 1 && result[j] > result[j + 1] {
                j += 1;
            }

            // Collect values and weights for the violating region
            let mut values_weights: Vec<(Float, Float)> = Vec::new();
            for k in i..=j {
                values_weights.push((result[k], w[k]));
            }

            // Find weighted median
            let median = weighted_median(&values_weights);

            // Set all values in the violating region to the median
            for k in i..=j {
                result[k] = median;
            }

            // Check if we need to backtrack
            while i > 0 && result[i - 1] > result[i] {
                i -= 1;
                let mut values_weights: Vec<(Float, Float)> = Vec::new();
                for k in i..=j {
                    values_weights.push((result[k], w[k]));
                }
                let median = weighted_median(&values_weights);
                for k in i..=j {
                    result[k] = median;
                }
            }
        } else {
            i += 1;
        }
    }

    if !increasing {
        result.mapv_inplace(|x| -x);
    }

    Ok(result)
}

/// Pool Adjacent Violators Algorithm for Huber loss
///
/// This implements PAVA for Huber loss using iterative reweighting.
pub fn pool_adjacent_violators_huber(
    y: &Array1<Float>,
    weights: Option<&Array1<Float>>,
    delta: Float,
    increasing: bool,
) -> Result<Array1<Float>> {
    if y.is_empty() {
        return Ok(Array1::zeros(0));
    }

    if delta <= 0.0 {
        return Err(SklearsError::InvalidInput(
            "Huber delta parameter must be positive".to_string(),
        ));
    }

    let n = y.len();
    let base_weights = if let Some(weights) = weights {
        weights.clone()
    } else {
        Array1::ones(n)
    };

    // Start with L2 solution
    let mut result = pool_adjacent_violators_l2(y, Some(&base_weights), increasing)?;

    // Iterative reweighting for Huber loss
    for _iter in 0..10 {
        // Maximum 10 iterations
        let mut huber_weights = base_weights.clone();

        // Update weights based on residuals
        for i in 0..n {
            let residual = (y[i] - result[i]).abs();
            if residual > delta {
                huber_weights[i] = base_weights[i] * delta / residual;
            }
        }

        // Solve weighted L2 problem
        let new_result = pool_adjacent_violators_l2(y, Some(&huber_weights), increasing)?;

        // Check for convergence
        let mut converged = true;
        for i in 0..n {
            if (result[i] - new_result[i]).abs() > 1e-6 {
                converged = false;
                break;
            }
        }

        result = new_result;
        if converged {
            break;
        }
    }

    Ok(result)
}

/// Pool Adjacent Violators Algorithm for quantile loss
///
/// This implements PAVA for quantile regression using weighted quantiles.
pub fn pool_adjacent_violators_quantile(
    y: &Array1<Float>,
    weights: Option<&Array1<Float>>,
    quantile: Float,
    increasing: bool,
) -> Result<Array1<Float>> {
    if y.is_empty() {
        return Ok(Array1::zeros(0));
    }

    if quantile <= 0.0 || quantile >= 1.0 {
        return Err(SklearsError::InvalidInput(
            "Quantile must be between 0 and 1".to_string(),
        ));
    }

    let n = y.len();
    let mut result = y.clone();
    let w = if let Some(weights) = weights {
        weights.clone()
    } else {
        Array1::ones(n)
    };

    if !increasing {
        result.mapv_inplace(|x| -x);
    }

    let mut i = 0;
    while i < result.len() - 1 {
        if result[i] > result[i + 1] {
            let mut j = i;

            // Find the violating region
            while j < result.len() - 1 && result[j] > result[j + 1] {
                j += 1;
            }

            // Collect values and weights for the violating region
            let mut values_weights: Vec<(Float, Float)> = Vec::new();
            for k in i..=j {
                values_weights.push((result[k], w[k]));
            }

            // Find weighted quantile
            let q = weighted_quantile(&values_weights, quantile);

            // Set all values in the violating region to the quantile
            for k in i..=j {
                result[k] = q;
            }

            // Check if we need to backtrack
            while i > 0 && result[i - 1] > result[i] {
                i -= 1;
                let mut values_weights: Vec<(Float, Float)> = Vec::new();
                for k in i..=j {
                    values_weights.push((result[k], w[k]));
                }
                let q = weighted_quantile(&values_weights, quantile);
                for k in i..=j {
                    result[k] = q;
                }
            }
        } else {
            i += 1;
        }
    }

    if !increasing {
        result.mapv_inplace(|x| -x);
    }

    Ok(result)
}

/// Calculate weighted median of values
pub fn weighted_median(values_weights: &[(Float, Float)]) -> Float {
    if values_weights.is_empty() {
        return 0.0;
    }

    if values_weights.len() == 1 {
        return values_weights[0].0;
    }

    // Sort by value
    let mut sorted = values_weights.to_vec();
    sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    // Calculate cumulative weights
    let total_weight: Float = sorted.iter().map(|(_, w)| w).sum();
    let half_weight = total_weight / 2.0;

    let mut cum_weight = 0.0;
    for (value, weight) in &sorted {
        cum_weight += weight;
        if cum_weight >= half_weight {
            return *value;
        }
    }

    // Fallback
    sorted.last().unwrap().0
}

/// Calculate weighted quantile of values
pub fn weighted_quantile(values_weights: &[(Float, Float)], quantile: Float) -> Float {
    if values_weights.is_empty() {
        return 0.0;
    }

    if values_weights.len() == 1 {
        return values_weights[0].0;
    }

    // Sort by value
    let mut sorted = values_weights.to_vec();
    sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    // Calculate cumulative weights
    let total_weight: Float = sorted.iter().map(|(_, w)| w).sum();
    let target_weight = total_weight * quantile;

    let mut cum_weight = 0.0;
    for (value, weight) in &sorted {
        cum_weight += weight;
        if cum_weight >= target_weight {
            return *value;
        }
    }

    // Fallback
    sorted.last().unwrap().0
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_pav_l2_increasing() {
        let y = array![1.0, 3.0, 2.0, 4.0, 5.0];
        let result = pool_adjacent_violators_l2(&y, None, true).unwrap();

        // Check that result is increasing
        for i in 0..result.len() - 1 {
            assert!(result[i] <= result[i + 1]);
        }
    }

    #[test]
    fn test_pav_l2_decreasing() {
        let y = array![5.0, 3.0, 4.0, 2.0, 1.0];
        let result = pool_adjacent_violators_l2(&y, None, false).unwrap();

        // Check that result is decreasing
        for i in 0..result.len() - 1 {
            assert!(result[i] >= result[i + 1]);
        }
    }

    #[test]
    fn test_pav_l2_weighted() {
        let y = array![1.0, 3.0, 2.0, 4.0, 5.0];
        let weights = array![1.0, 10.0, 1.0, 1.0, 1.0]; // Heavy weight on second value
        let result = pool_adjacent_violators_l2(&y, Some(&weights), true).unwrap();

        // Check that result is increasing
        for i in 0..result.len() - 1 {
            assert!(result[i] <= result[i + 1]);
        }
    }

    #[test]
    fn test_weighted_median() {
        let values = vec![(1.0, 1.0), (2.0, 1.0), (3.0, 1.0)];
        let median = weighted_median(&values);
        assert_abs_diff_eq!(median, 2.0, epsilon = 1e-10);

        // Test with heavy weight on one value
        let values = vec![(1.0, 1.0), (2.0, 10.0), (3.0, 1.0)];
        let median = weighted_median(&values);
        assert_abs_diff_eq!(median, 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_weighted_quantile() {
        let values = vec![(1.0, 1.0), (2.0, 1.0), (3.0, 1.0), (4.0, 1.0), (5.0, 1.0)];
        let q25 = weighted_quantile(&values, 0.25);
        let q50 = weighted_quantile(&values, 0.5);
        let q75 = weighted_quantile(&values, 0.75);

        assert!(q25 <= q50);
        assert!(q50 <= q75);
    }

    #[test]
    fn test_pav_empty() {
        let y = Array1::zeros(0);
        let result = pool_adjacent_violators_l2(&y, None, true).unwrap();
        assert_eq!(result.len(), 0);
    }
}
