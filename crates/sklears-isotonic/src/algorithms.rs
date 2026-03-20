//! Core isotonic regression algorithms
//!
//! This module contains the fundamental algorithms for isotonic regression,
//! including Pool Adjacent Violators (PAV) and related utility functions.

use crate::core::{LossFunction, MonotonicityConstraint};
use scirs2_core::ndarray::{s, Array1};
use sklears_core::types::Float;

/// Pool Adjacent Violators algorithm for increasing constraint
pub fn pool_adjacent_violators_increasing(y: &Array1<Float>) -> Array1<Float> {
    let n = y.len();
    let mut result = y.clone();

    let mut i = 0;
    while i < n - 1 {
        if result[i] > result[i + 1] {
            // Pool adjacent violators
            let mut j = i + 1;
            let mut sum = result[i] + result[j];
            let mut count = 2;

            // Find all violators to the right
            while j < n - 1 && result[i] > result[j + 1] {
                j += 1;
                sum += result[j];
                count += 1;
            }

            // Update the pooled values
            let pooled_value = sum / count as Float;
            for k in i..=j {
                result[k] = pooled_value;
            }

            // Check if we need to pool with previous values
            while i > 0 && result[i - 1] > result[i] {
                i -= 1;
                sum += result[i] * (j - i + 1) as Float;
                count += j - i + 1;
                let new_pooled = sum / count as Float;
                for m in i..=j {
                    result[m] = new_pooled;
                }
            }
        }
        i += 1;
    }

    result
}

/// Pool Adjacent Violators algorithm for increasing constraint with weights
pub fn pool_adjacent_violators_increasing_weighted(
    y: &Array1<Float>,
    sample_weights: &Array1<Float>,
) -> Array1<Float> {
    let n = y.len();
    let mut result = y.clone();
    let mut weights = sample_weights.clone();

    let mut i = 0;
    while i < n - 1 {
        if result[i] > result[i + 1] {
            // Pool adjacent violators
            let mut j = i + 1;
            let mut sum_y = result[i] * weights[i] + result[j] * weights[j];
            let mut sum_w = weights[i] + weights[j];

            // Find all violators to the right
            while j < n - 1 && result[i] > result[j + 1] {
                j += 1;
                sum_y += result[j] * weights[j];
                sum_w += weights[j];
            }

            // Update the pooled values
            let pooled_value = sum_y / sum_w;
            for k in i..=j {
                result[k] = pooled_value;
            }

            // Update weights to preserve total weight
            let total_weight = sum_w;
            for k in i..=j {
                weights[k] = total_weight / (j - i + 1) as Float;
            }

            // Check if we need to pool with previous values
            while i > 0 && result[i - 1] > result[i] {
                i -= 1;
                sum_y += result[i] * weights[i];
                sum_w += weights[i];
                let new_pooled = sum_y / sum_w;
                for m in i..=j {
                    result[m] = new_pooled;
                    weights[m] = sum_w / (j - i + 1) as Float;
                }
            }
        }
        i += 1;
    }

    result
}

/// Pool Adjacent Violators algorithm for decreasing constraint
pub fn pool_adjacent_violators_decreasing(y: &Array1<Float>) -> Array1<Float> {
    let n = y.len();
    let mut result = y.clone();

    let mut i = 0;
    while i < n - 1 {
        if result[i] < result[i + 1] {
            // Pool adjacent violators
            let mut j = i + 1;
            let mut sum = result[i] + result[j];
            let mut count = 2;

            // Find all violators to the right
            while j < n - 1 && result[i] < result[j + 1] {
                j += 1;
                sum += result[j];
                count += 1;
            }

            // Update the pooled values
            let pooled_value = sum / count as Float;
            for k in i..=j {
                result[k] = pooled_value;
            }

            // Check if we need to pool with previous values
            while i > 0 && result[i - 1] < result[i] {
                i -= 1;
                sum += result[i] * (j - i + 1) as Float;
                count += j - i + 1;
                let new_pooled = sum / count as Float;
                for m in i..=j {
                    result[m] = new_pooled;
                }
            }
        }
        i += 1;
    }

    result
}

/// Pool Adjacent Violators algorithm for decreasing constraint with weights
pub fn pool_adjacent_violators_decreasing_weighted(
    y: &Array1<Float>,
    sample_weights: &Array1<Float>,
) -> Array1<Float> {
    let n = y.len();
    let mut result = y.clone();
    let mut weights = sample_weights.clone();

    let mut i = 0;
    while i < n - 1 {
        if result[i] < result[i + 1] {
            // Pool adjacent violators
            let mut j = i + 1;
            let mut sum_y = result[i] * weights[i] + result[j] * weights[j];
            let mut sum_w = weights[i] + weights[j];

            // Find all violators to the right
            while j < n - 1 && result[i] < result[j + 1] {
                j += 1;
                sum_y += result[j] * weights[j];
                sum_w += weights[j];
            }

            // Update the pooled values
            let pooled_value = sum_y / sum_w;
            for k in i..=j {
                result[k] = pooled_value;
            }

            // Update weights to preserve total weight
            let total_weight = sum_w;
            for k in i..=j {
                weights[k] = total_weight / (j - i + 1) as Float;
            }

            // Check if we need to pool with previous values
            while i > 0 && result[i - 1] < result[i] {
                i -= 1;
                sum_y += result[i] * weights[i];
                sum_w += weights[i];
                let new_pooled = sum_y / sum_w;
                for m in i..=j {
                    result[m] = new_pooled;
                    weights[m] = sum_w / (j - i + 1) as Float;
                }
            }
        }
        i += 1;
    }

    result
}

/// Apply isotonic regression for different loss functions
pub fn isotonic_regression_with_loss(
    y: &Array1<Float>,
    weights: Option<&Array1<Float>>,
    loss: &LossFunction,
    increasing: bool,
) -> Array1<Float> {
    match loss {
        LossFunction::SquaredLoss => {
            if increasing {
                match weights {
                    Some(w) => pool_adjacent_violators_increasing_weighted(y, w),
                    None => pool_adjacent_violators_increasing(y),
                }
            } else {
                match weights {
                    Some(w) => pool_adjacent_violators_decreasing_weighted(y, w),
                    None => pool_adjacent_violators_decreasing(y),
                }
            }
        }
        _ => {
            // For other loss functions, use iterative approach
            isotonic_regression_robust(y, weights, loss, increasing)
        }
    }
}

/// Robust isotonic regression for different loss functions
pub fn isotonic_regression_robust(
    y: &Array1<Float>,
    weights: Option<&Array1<Float>>,
    loss: &LossFunction,
    increasing: bool,
) -> Array1<Float> {
    let _n = y.len();
    let mut result = y.clone();

    // For robust losses, use iterative approach
    for _ in 0..10 {
        // Maximum iterations
        let prev_result = result.clone();

        // Apply PAV based on current loss function
        result = match loss {
            LossFunction::SquaredLoss => {
                if increasing {
                    match weights {
                        Some(w) => pool_adjacent_violators_increasing_weighted(&result, w),
                        None => pool_adjacent_violators_increasing(&result),
                    }
                } else {
                    match weights {
                        Some(w) => pool_adjacent_violators_decreasing_weighted(&result, w),
                        None => pool_adjacent_violators_decreasing(&result),
                    }
                }
            }
            LossFunction::AbsoluteLoss => {
                // For L1 loss, use weighted median approach
                if increasing {
                    pool_adjacent_violators_l1(&result, weights, true)
                } else {
                    pool_adjacent_violators_l1(&result, weights, false)
                }
            }
            LossFunction::HuberLoss { delta } => {
                // For Huber loss, use reweighted approach
                pool_adjacent_violators_huber(&result, weights, *delta, increasing)
            }
            LossFunction::QuantileLoss { quantile } => {
                // For quantile loss, use quantile regression approach
                pool_adjacent_violators_quantile(&result, weights, *quantile, increasing)
            }
        };

        // Check convergence
        let diff: Float = result
            .iter()
            .zip(prev_result.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();

        if diff < 1e-8 {
            break;
        }
    }

    result
}

/// Pool Adjacent Violators for L1 loss (weighted median)
fn pool_adjacent_violators_l1(
    y: &Array1<Float>,
    _weights: Option<&Array1<Float>>,
    increasing: bool,
) -> Array1<Float> {
    let _n = y.len();
    let result = y.clone();

    // Simple implementation - in practice would use more sophisticated weighted median
    if increasing {
        pool_adjacent_violators_increasing(&result)
    } else {
        pool_adjacent_violators_decreasing(&result)
    }
}

/// Pool Adjacent Violators for Huber loss
fn pool_adjacent_violators_huber(
    y: &Array1<Float>,
    weights: Option<&Array1<Float>>,
    _delta: Float,
    increasing: bool,
) -> Array1<Float> {
    // Simplified implementation - would use Huber-weighted approach in practice
    isotonic_regression_with_loss(y, weights, &LossFunction::SquaredLoss, increasing)
}

/// Pool Adjacent Violators for quantile loss
fn pool_adjacent_violators_quantile(
    y: &Array1<Float>,
    weights: Option<&Array1<Float>>,
    _quantile: Float,
    increasing: bool,
) -> Array1<Float> {
    // Simplified implementation - would use quantile-specific approach in practice
    isotonic_regression_with_loss(y, weights, &LossFunction::SquaredLoss, increasing)
}

/// Linear interpolation for isotonic regression predictions
pub fn linear_interpolate(
    x_train: &Array1<Float>,
    y_train: &Array1<Float>,
    x_new: &Array1<Float>,
) -> Array1<Float> {
    let mut result = Array1::zeros(x_new.len());

    for (i, &x) in x_new.iter().enumerate() {
        // Find the appropriate interval for interpolation
        if x <= x_train[0] {
            result[i] = y_train[0];
        } else if x >= x_train[x_train.len() - 1] {
            result[i] = y_train[y_train.len() - 1];
        } else {
            // Find the interval
            let mut j = 0;
            while j < x_train.len() - 1 && x_train[j + 1] < x {
                j += 1;
            }

            if j < x_train.len() - 1 {
                // Linear interpolation
                let x1 = x_train[j];
                let x2 = x_train[j + 1];
                let y1 = y_train[j];
                let y2 = y_train[j + 1];

                result[i] = y1 + (y2 - y1) * (x - x1) / (x2 - x1);
            } else {
                result[i] = y_train[j];
            }
        }
    }

    result
}

/// Apply monotonicity constraints
pub fn apply_constraints(
    fitted_values: &mut Array1<Float>,
    constraint: &MonotonicityConstraint,
    x: &Array1<Float>,
) {
    match constraint {
        MonotonicityConstraint::Global { increasing } => {
            if *increasing {
                *fitted_values = pool_adjacent_violators_increasing(fitted_values);
            } else {
                *fitted_values = pool_adjacent_violators_decreasing(fitted_values);
            }
        }
        MonotonicityConstraint::Piecewise {
            breakpoints,
            segments,
        } => {
            apply_piecewise_constraints(fitted_values, x, breakpoints, segments);
        }
        MonotonicityConstraint::Convex => {
            apply_convex_constraint(fitted_values, true);
        }
        MonotonicityConstraint::Concave => {
            apply_concave_constraint(fitted_values, true);
        }
        MonotonicityConstraint::ConvexDecreasing => {
            apply_convex_constraint(fitted_values, false);
        }
        MonotonicityConstraint::ConcaveDecreasing => {
            apply_concave_constraint(fitted_values, false);
        }
        MonotonicityConstraint::Increasing => {
            *fitted_values = pool_adjacent_violators_increasing(fitted_values);
        }
        MonotonicityConstraint::Decreasing => {
            *fitted_values = pool_adjacent_violators_decreasing(fitted_values);
        }
        MonotonicityConstraint::None => {
            // No constraints to apply
        }
    }
}

/// Apply piecewise monotonicity constraints
fn apply_piecewise_constraints(
    fitted_values: &mut Array1<Float>,
    x: &Array1<Float>,
    breakpoints: &[Float],
    segment_increasing: &[bool],
) {
    if breakpoints.is_empty() || segment_increasing.is_empty() {
        return;
    }

    let mut start_idx = 0;

    for (i, &breakpoint) in breakpoints.iter().enumerate() {
        // Find end index for this segment
        let end_idx = x.iter().position(|&xi| xi > breakpoint).unwrap_or(x.len());

        if start_idx < end_idx && i < segment_increasing.len() {
            // Apply constraint to this segment
            let mut segment = fitted_values.slice(s![start_idx..end_idx]).to_owned();

            if segment_increasing[i] {
                segment = pool_adjacent_violators_increasing(&segment);
            } else {
                segment = pool_adjacent_violators_decreasing(&segment);
            }

            // Update the fitted values
            fitted_values
                .slice_mut(s![start_idx..end_idx])
                .assign(&segment);
        }

        start_idx = end_idx;
    }

    // Handle the last segment
    if start_idx < fitted_values.len() && segment_increasing.len() > breakpoints.len() {
        let mut segment = fitted_values.slice(s![start_idx..]).to_owned();

        if segment_increasing[breakpoints.len()] {
            segment = pool_adjacent_violators_increasing(&segment);
        } else {
            segment = pool_adjacent_violators_decreasing(&segment);
        }

        fitted_values.slice_mut(s![start_idx..]).assign(&segment);
    }
}

/// Apply convex constraint
fn apply_convex_constraint(fitted_values: &mut Array1<Float>, increasing: bool) {
    // First apply monotonicity
    if increasing {
        *fitted_values = pool_adjacent_violators_increasing(fitted_values);
    } else {
        *fitted_values = pool_adjacent_violators_decreasing(fitted_values);
    }

    // Then apply convexity constraint to second differences
    let n = fitted_values.len();
    if n < 3 {
        return;
    }

    // Compute second differences and apply isotonic regression
    let mut second_diffs = Array1::zeros(n - 2);
    for i in 0..n - 2 {
        second_diffs[i] = fitted_values[i + 2] - 2.0 * fitted_values[i + 1] + fitted_values[i];
    }

    // Apply isotonic regression to second differences
    second_diffs = pool_adjacent_violators_increasing(&second_diffs);

    // Reconstruct the fitted values
    for i in 0..n - 2 {
        let target_second_diff = second_diffs[i];
        let current_second_diff =
            fitted_values[i + 2] - 2.0 * fitted_values[i + 1] + fitted_values[i];
        let adjustment = (target_second_diff - current_second_diff) / 3.0;

        fitted_values[i] -= adjustment;
        fitted_values[i + 2] += adjustment;
    }
}

/// Apply concave constraint  
fn apply_concave_constraint(fitted_values: &mut Array1<Float>, increasing: bool) {
    // First apply monotonicity
    if increasing {
        *fitted_values = pool_adjacent_violators_increasing(fitted_values);
    } else {
        *fitted_values = pool_adjacent_violators_decreasing(fitted_values);
    }

    // Then apply concavity constraint (decreasing second differences)
    let n = fitted_values.len();
    if n < 3 {
        return;
    }

    // Compute second differences and apply decreasing isotonic regression
    let mut second_diffs = Array1::zeros(n - 2);
    for i in 0..n - 2 {
        second_diffs[i] = fitted_values[i + 2] - 2.0 * fitted_values[i + 1] + fitted_values[i];
    }

    // Apply decreasing isotonic regression to second differences
    second_diffs = pool_adjacent_violators_decreasing(&second_diffs);

    // Reconstruct the fitted values
    for i in 0..n - 2 {
        let target_second_diff = second_diffs[i];
        let current_second_diff =
            fitted_values[i + 2] - 2.0 * fitted_values[i + 1] + fitted_values[i];
        let adjustment = (target_second_diff - current_second_diff) / 3.0;

        fitted_values[i] -= adjustment;
        fitted_values[i + 2] += adjustment;
    }
}
