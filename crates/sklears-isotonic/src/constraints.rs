//! Constraint handling and validation for isotonic regression
//!
//! This module provides utilities for applying and validating monotonicity
//! constraints, bounds, and other constraints used in isotonic regression.

use crate::{pool_adjacent_violators_l2, MonotonicityConstraint};
use scirs2_core::ndarray::Array1;
use sklears_core::{error::Result, types::Float};

/// Apply piecewise monotonicity constraints based on breakpoints
fn apply_piecewise_constraint(
    values: &Array1<Float>,
    breakpoints: &[Float],
    segments: &[bool],
    weights: Option<&Array1<Float>>,
) -> Result<Array1<Float>> {
    if breakpoints.is_empty() || segments.is_empty() {
        return Ok(values.clone());
    }

    let n = values.len();
    let mut result = values.clone();

    // Create index ranges for each segment
    // For simplicity, we treat indices as proxies for x-values
    // In a real implementation, we'd need the x-values to map breakpoints
    let segment_count = segments.len();
    let segment_size = (n as f64 / segment_count as f64).ceil() as usize;

    for (seg_idx, &is_increasing) in segments.iter().enumerate() {
        let start_idx = seg_idx * segment_size;
        let end_idx = ((seg_idx + 1) * segment_size).min(n);

        if start_idx >= n {
            break;
        }

        // Extract segment
        let segment_values = result
            .slice(scirs2_core::ndarray::s![start_idx..end_idx])
            .to_owned();
        let segment_weights = weights.map(|w| {
            w.slice(scirs2_core::ndarray::s![start_idx..end_idx])
                .to_owned()
        });

        // Apply monotonicity constraint to this segment
        let constrained_segment =
            pool_adjacent_violators_l2(&segment_values, segment_weights.as_ref(), is_increasing)?;

        // Update result with constrained segment
        for (i, &val) in constrained_segment.iter().enumerate() {
            if start_idx + i < n {
                result[start_idx + i] = val;
            }
        }
    }

    Ok(result)
}

/// Apply global monotonicity constraint to a sequence of values
pub fn apply_global_constraint(
    values: &Array1<Float>,
    constraint: MonotonicityConstraint,
    weights: Option<&Array1<Float>>,
) -> Result<Array1<Float>> {
    match constraint {
        MonotonicityConstraint::Increasing => pool_adjacent_violators_l2(values, weights, true),
        MonotonicityConstraint::Decreasing => pool_adjacent_violators_l2(values, weights, false),
        MonotonicityConstraint::Global { increasing } => {
            pool_adjacent_violators_l2(values, weights, increasing)
        }
        MonotonicityConstraint::None => Ok(values.clone()),
        MonotonicityConstraint::Piecewise {
            breakpoints,
            segments,
        } => {
            apply_piecewise_constraint(values, breakpoints.as_slice(), segments.as_slice(), weights)
        }
        MonotonicityConstraint::Convex => {
            // Apply both increasing constraint and convexity constraint
            let monotonic = pool_adjacent_violators_l2(values, weights, true)?;
            apply_convexity_constraint(&monotonic, weights)
        }
        MonotonicityConstraint::Concave => {
            // Apply both increasing constraint and concavity constraint
            let monotonic = pool_adjacent_violators_l2(values, weights, true)?;
            apply_concavity_constraint(&monotonic, weights)
        }
        MonotonicityConstraint::ConvexDecreasing => {
            // Apply both decreasing constraint and convexity constraint
            let monotonic = pool_adjacent_violators_l2(values, weights, false)?;
            apply_convexity_constraint(&monotonic, weights)
        }
        MonotonicityConstraint::ConcaveDecreasing => {
            // Apply both decreasing constraint and concavity constraint
            let monotonic = pool_adjacent_violators_l2(values, weights, false)?;
            apply_concavity_constraint(&monotonic, weights)
        }
    }
}

/// Validate that a sequence satisfies monotonicity constraints
pub fn validate_monotonicity(
    values: &Array1<Float>,
    constraint: MonotonicityConstraint,
    tolerance: Float,
) -> bool {
    if values.len() <= 1 {
        return true;
    }

    match constraint {
        MonotonicityConstraint::Increasing => {
            for i in 0..values.len() - 1 {
                if values[i + 1] < values[i] - tolerance {
                    return false;
                }
            }
            true
        }
        MonotonicityConstraint::Decreasing => {
            for i in 0..values.len() - 1 {
                if values[i + 1] > values[i] + tolerance {
                    return false;
                }
            }
            true
        }
        MonotonicityConstraint::Global { increasing } => {
            if increasing {
                for i in 0..values.len() - 1 {
                    if values[i + 1] < values[i] - tolerance {
                        return false;
                    }
                }
            } else {
                for i in 0..values.len() - 1 {
                    if values[i + 1] > values[i] + tolerance {
                        return false;
                    }
                }
            }
            true
        }
        MonotonicityConstraint::None => true,
        MonotonicityConstraint::Piecewise {
            segments,
            breakpoints: _,
        } => {
            // Validate each segment separately
            let n = values.len();
            if segments.is_empty() {
                return true;
            }

            let segment_count = segments.len();
            let segment_size = (n as f64 / segment_count as f64).ceil() as usize;

            for (seg_idx, &is_increasing) in segments.iter().enumerate() {
                let start_idx = seg_idx * segment_size;
                let end_idx = ((seg_idx + 1) * segment_size).min(n);

                if start_idx >= n {
                    break;
                }

                // Check monotonicity within this segment
                for i in start_idx..end_idx.saturating_sub(1) {
                    if is_increasing {
                        if values[i + 1] < values[i] - tolerance {
                            return false;
                        }
                    } else if values[i + 1] > values[i] + tolerance {
                        return false;
                    }
                }
            }
            true
        }
        MonotonicityConstraint::Convex => {
            // Check both increasing and convexity
            validate_monotonicity(values, MonotonicityConstraint::Increasing, tolerance)
                && validate_convexity(values, tolerance)
        }
        MonotonicityConstraint::Concave => {
            // Check both increasing and concavity
            validate_monotonicity(values, MonotonicityConstraint::Increasing, tolerance)
                && validate_concavity(values, tolerance)
        }
        MonotonicityConstraint::ConvexDecreasing => {
            // Check both decreasing and convexity
            validate_monotonicity(values, MonotonicityConstraint::Decreasing, tolerance)
                && validate_convexity(values, tolerance)
        }
        MonotonicityConstraint::ConcaveDecreasing => {
            // Check both decreasing and concavity
            validate_monotonicity(values, MonotonicityConstraint::Decreasing, tolerance)
                && validate_concavity(values, tolerance)
        }
    }
}

/// Apply bounds constraints to values
pub fn apply_bounds(values: &mut Array1<Float>, y_min: Option<Float>, y_max: Option<Float>) {
    if let Some(min_val) = y_min {
        values.mapv_inplace(|x| x.max(min_val));
    }
    if let Some(max_val) = y_max {
        values.mapv_inplace(|x| x.min(max_val));
    }
}

/// Validate bounds constraints
pub fn validate_bounds(
    values: &Array1<Float>,
    y_min: Option<Float>,
    y_max: Option<Float>,
    tolerance: Float,
) -> bool {
    for &val in values {
        if let Some(min_val) = y_min {
            if val < min_val - tolerance {
                return false;
            }
        }
        if let Some(max_val) = y_max {
            if val > max_val + tolerance {
                return false;
            }
        }
    }
    true
}

/// Check if values satisfy convexity constraint
pub fn validate_convexity(values: &Array1<Float>, tolerance: Float) -> bool {
    if values.len() <= 2 {
        return true;
    }

    for i in 1..values.len() - 1 {
        // Check if second derivative is non-negative (convex)
        let second_deriv = values[i + 1] - 2.0 * values[i] + values[i - 1];
        if second_deriv < -tolerance {
            return false;
        }
    }
    true
}

/// Check if values satisfy concavity constraint
pub fn validate_concavity(values: &Array1<Float>, tolerance: Float) -> bool {
    if values.len() <= 2 {
        return true;
    }

    for i in 1..values.len() - 1 {
        // Check if second derivative is non-positive (concave)
        let second_deriv = values[i + 1] - 2.0 * values[i] + values[i - 1];
        if second_deriv > tolerance {
            return false;
        }
    }
    true
}

/// Apply convexity constraint using iterative projection
pub fn apply_convexity_constraint(
    values: &Array1<Float>,
    _weights: Option<&Array1<Float>>,
) -> Result<Array1<Float>> {
    let mut result = values.clone();
    let n = result.len();

    if n <= 2 {
        return Ok(result);
    }

    // Iterative projection to enforce convexity
    let max_iter = 100;
    for _iter in 0..max_iter {
        let mut converged = true;

        // Check each triplet for convexity violation
        for i in 1..n - 1 {
            let left = result[i - 1];
            let center = result[i];
            let right = result[i + 1];

            // For convexity: center <= (left + right) / 2
            let expected_center = (left + right) / 2.0;
            if center > expected_center + 1e-10 {
                result[i] = expected_center;
                converged = false;
            }
        }

        if converged {
            break;
        }
    }

    Ok(result)
}

/// Apply concavity constraint using iterative projection
pub fn apply_concavity_constraint(
    values: &Array1<Float>,
    _weights: Option<&Array1<Float>>,
) -> Result<Array1<Float>> {
    let mut result = values.clone();
    let n = result.len();

    if n <= 2 {
        return Ok(result);
    }

    // Iterative projection to enforce concavity
    let max_iter = 100;
    for _iter in 0..max_iter {
        let mut converged = true;

        // Check each triplet for concavity violation
        for i in 1..n - 1 {
            let left = result[i - 1];
            let center = result[i];
            let right = result[i + 1];

            // For concavity: center >= (left + right) / 2
            let expected_center = (left + right) / 2.0;
            if center < expected_center - 1e-10 {
                result[i] = expected_center;
                converged = false;
            }
        }

        if converged {
            break;
        }
    }

    Ok(result)
}

/// Count convexity violations (second derivative should be non-negative)
fn count_convexity_violations(values: &Array1<Float>) -> usize {
    if values.len() <= 2 {
        return 0;
    }

    let mut violations = 0;
    for i in 1..values.len() - 1 {
        let second_deriv = values[i + 1] - 2.0 * values[i] + values[i - 1];
        if second_deriv < 0.0 {
            violations += 1;
        }
    }
    violations
}

/// Count concavity violations (second derivative should be non-positive)
fn count_concavity_violations(values: &Array1<Float>) -> usize {
    if values.len() <= 2 {
        return 0;
    }

    let mut violations = 0;
    for i in 1..values.len() - 1 {
        let second_deriv = values[i + 1] - 2.0 * values[i] + values[i - 1];
        if second_deriv > 0.0 {
            violations += 1;
        }
    }
    violations
}

/// Count monotonicity violations in a sequence
pub fn count_monotonicity_violations(
    values: &Array1<Float>,
    constraint: MonotonicityConstraint,
) -> usize {
    if values.len() <= 1 {
        return 0;
    }

    let mut violations = 0;

    match constraint {
        MonotonicityConstraint::Increasing => {
            for i in 0..values.len() - 1 {
                if values[i + 1] < values[i] {
                    violations += 1;
                }
            }
        }
        MonotonicityConstraint::Decreasing => {
            for i in 0..values.len() - 1 {
                if values[i + 1] > values[i] {
                    violations += 1;
                }
            }
        }
        MonotonicityConstraint::Global { increasing } => {
            if increasing {
                for i in 0..values.len() - 1 {
                    if values[i + 1] < values[i] {
                        violations += 1;
                    }
                }
            } else {
                for i in 0..values.len() - 1 {
                    if values[i + 1] > values[i] {
                        violations += 1;
                    }
                }
            }
        }
        MonotonicityConstraint::None => {}
        MonotonicityConstraint::Piecewise {
            segments,
            breakpoints: _,
        } => {
            // Count violations in each segment separately
            let n = values.len();
            if segments.is_empty() {
                return violations;
            }

            let segment_count = segments.len();
            let segment_size = (n as f64 / segment_count as f64).ceil() as usize;

            for (seg_idx, &is_increasing) in segments.iter().enumerate() {
                let start_idx = seg_idx * segment_size;
                let end_idx = ((seg_idx + 1) * segment_size).min(n);

                if start_idx >= n {
                    break;
                }

                // Count violations within this segment
                for i in start_idx..end_idx.saturating_sub(1) {
                    if is_increasing {
                        if values[i + 1] < values[i] {
                            violations += 1;
                        }
                    } else if values[i + 1] > values[i] {
                        violations += 1;
                    }
                }
            }
        }
        MonotonicityConstraint::Convex => {
            // Count both monotonicity and convexity violations
            violations += count_monotonicity_violations(values, MonotonicityConstraint::Increasing);
            violations += count_convexity_violations(values);
        }
        MonotonicityConstraint::Concave => {
            // Count both monotonicity and concavity violations
            violations += count_monotonicity_violations(values, MonotonicityConstraint::Increasing);
            violations += count_concavity_violations(values);
        }
        MonotonicityConstraint::ConvexDecreasing => {
            // Count both decreasing and convexity violations
            violations += count_monotonicity_violations(values, MonotonicityConstraint::Decreasing);
            violations += count_convexity_violations(values);
        }
        MonotonicityConstraint::ConcaveDecreasing => {
            // Count both decreasing and concavity violations
            violations += count_monotonicity_violations(values, MonotonicityConstraint::Decreasing);
            violations += count_concavity_violations(values);
        }
    }

    violations
}

/// Calculate the magnitude of convexity violations
fn calculate_convexity_violation_magnitude(values: &Array1<Float>) -> Float {
    if values.len() <= 2 {
        return 0.0;
    }

    let mut total_violation = 0.0;
    for i in 1..values.len() - 1 {
        let second_deriv = values[i + 1] - 2.0 * values[i] + values[i - 1];
        if second_deriv < 0.0 {
            total_violation += second_deriv.abs();
        }
    }
    total_violation
}

/// Calculate the magnitude of concavity violations
fn calculate_concavity_violation_magnitude(values: &Array1<Float>) -> Float {
    if values.len() <= 2 {
        return 0.0;
    }

    let mut total_violation = 0.0;
    for i in 1..values.len() - 1 {
        let second_deriv = values[i + 1] - 2.0 * values[i] + values[i - 1];
        if second_deriv > 0.0 {
            total_violation += second_deriv.abs();
        }
    }
    total_violation
}

/// Calculate the magnitude of constraint violations
pub fn constraint_violation_magnitude(
    values: &Array1<Float>,
    constraint: MonotonicityConstraint,
) -> Float {
    if values.len() <= 1 {
        return 0.0;
    }

    let mut total_violation = 0.0;

    match constraint {
        MonotonicityConstraint::Increasing => {
            for i in 0..values.len() - 1 {
                if values[i + 1] < values[i] {
                    total_violation += values[i] - values[i + 1];
                }
            }
        }
        MonotonicityConstraint::Decreasing => {
            for i in 0..values.len() - 1 {
                if values[i + 1] > values[i] {
                    total_violation += values[i + 1] - values[i];
                }
            }
        }
        MonotonicityConstraint::Global { increasing } => {
            if increasing {
                for i in 0..values.len() - 1 {
                    if values[i + 1] < values[i] {
                        total_violation += values[i] - values[i + 1];
                    }
                }
            } else {
                for i in 0..values.len() - 1 {
                    if values[i + 1] > values[i] {
                        total_violation += values[i + 1] - values[i];
                    }
                }
            }
        }
        MonotonicityConstraint::None => {}
        MonotonicityConstraint::Piecewise {
            segments,
            breakpoints: _,
        } => {
            // Calculate magnitude for each segment separately
            let n = values.len();
            if segments.is_empty() {
                return total_violation;
            }

            let segment_count = segments.len();
            let segment_size = (n as f64 / segment_count as f64).ceil() as usize;

            for (seg_idx, &is_increasing) in segments.iter().enumerate() {
                let start_idx = seg_idx * segment_size;
                let end_idx = ((seg_idx + 1) * segment_size).min(n);

                if start_idx >= n {
                    break;
                }

                // Calculate violations within this segment
                for i in start_idx..end_idx.saturating_sub(1) {
                    if is_increasing {
                        if values[i + 1] < values[i] {
                            total_violation += values[i] - values[i + 1];
                        }
                    } else if values[i + 1] > values[i] {
                        total_violation += values[i + 1] - values[i];
                    }
                }
            }
        }
        MonotonicityConstraint::Convex => {
            // Sum both monotonicity and convexity violation magnitudes
            total_violation +=
                constraint_violation_magnitude(values, MonotonicityConstraint::Increasing);
            total_violation += calculate_convexity_violation_magnitude(values);
        }
        MonotonicityConstraint::Concave => {
            // Sum both monotonicity and concavity violation magnitudes
            total_violation +=
                constraint_violation_magnitude(values, MonotonicityConstraint::Increasing);
            total_violation += calculate_concavity_violation_magnitude(values);
        }
        MonotonicityConstraint::ConvexDecreasing => {
            // Sum both decreasing and convexity violation magnitudes
            total_violation +=
                constraint_violation_magnitude(values, MonotonicityConstraint::Decreasing);
            total_violation += calculate_convexity_violation_magnitude(values);
        }
        MonotonicityConstraint::ConcaveDecreasing => {
            // Sum both decreasing and concavity violation magnitudes
            total_violation +=
                constraint_violation_magnitude(values, MonotonicityConstraint::Decreasing);
            total_violation += calculate_concavity_violation_magnitude(values);
        }
    }

    total_violation
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_apply_global_constraint_increasing() {
        let values = array![1.0, 3.0, 2.0, 4.0, 3.5];
        let result =
            apply_global_constraint(&values, MonotonicityConstraint::Increasing, None).unwrap();

        // Result should be increasing
        for i in 0..result.len() - 1 {
            assert!(result[i] <= result[i + 1]);
        }
    }

    #[test]
    fn test_apply_global_constraint_decreasing() {
        let values = array![4.0, 2.0, 3.0, 1.0, 1.5];
        let result =
            apply_global_constraint(&values, MonotonicityConstraint::Decreasing, None).unwrap();

        // Result should be decreasing
        for i in 0..result.len() - 1 {
            assert!(result[i] >= result[i + 1]);
        }
    }

    #[test]
    fn test_validate_monotonicity_increasing() {
        let increasing = array![1.0, 2.0, 3.0, 4.0];
        let not_increasing = array![1.0, 3.0, 2.0, 4.0];

        assert!(validate_monotonicity(
            &increasing,
            MonotonicityConstraint::Increasing,
            1e-10
        ));
        assert!(!validate_monotonicity(
            &not_increasing,
            MonotonicityConstraint::Increasing,
            1e-10
        ));
    }

    #[test]
    fn test_validate_monotonicity_decreasing() {
        let decreasing = array![4.0, 3.0, 2.0, 1.0];
        let not_decreasing = array![4.0, 2.0, 3.0, 1.0];

        assert!(validate_monotonicity(
            &decreasing,
            MonotonicityConstraint::Decreasing,
            1e-10
        ));
        assert!(!validate_monotonicity(
            &not_decreasing,
            MonotonicityConstraint::Decreasing,
            1e-10
        ));
    }

    #[test]
    fn test_apply_bounds() {
        let mut values = array![-1.0, 5.0, 10.0];
        apply_bounds(&mut values, Some(0.0), Some(8.0));

        assert_abs_diff_eq!(values[0], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(values[1], 5.0, epsilon = 1e-10);
        assert_abs_diff_eq!(values[2], 8.0, epsilon = 1e-10);
    }

    #[test]
    fn test_validate_bounds() {
        let values = array![1.0, 5.0, 8.0];

        assert!(validate_bounds(&values, Some(0.0), Some(10.0), 1e-10));
        assert!(!validate_bounds(&values, Some(2.0), Some(10.0), 1e-10)); // 1.0 < 2.0
        assert!(!validate_bounds(&values, Some(0.0), Some(7.0), 1e-10)); // 8.0 > 7.0
    }

    #[test]
    fn test_validate_convexity() {
        let convex = array![0.0, 1.0, 4.0, 9.0]; // x^2 is convex
        let not_convex = array![0.0, 4.0, 1.0, 9.0];

        assert!(validate_convexity(&convex, 1e-10));
        assert!(!validate_convexity(&not_convex, 1e-6));
    }

    #[test]
    fn test_validate_concavity() {
        let concave = array![0.0, 3.0, 4.0, 3.0]; // Inverted parabola-like
        let not_concave = array![0.0, 1.0, 4.0, 9.0]; // x^2 is not concave

        assert!(validate_concavity(&concave, 1e-10));
        assert!(!validate_concavity(&not_concave, 1e-6));
    }

    #[test]
    fn test_count_violations() {
        let values = array![1.0, 3.0, 2.0, 4.0]; // One violation: 3.0 > 2.0
        let violations = count_monotonicity_violations(&values, MonotonicityConstraint::Increasing);
        assert_eq!(violations, 1);
    }

    #[test]
    fn test_constraint_violation_magnitude() {
        let values = array![1.0, 5.0, 2.0, 4.0]; // Violation magnitude: 5.0 - 2.0 = 3.0
        let magnitude = constraint_violation_magnitude(&values, MonotonicityConstraint::Increasing);
        assert_abs_diff_eq!(magnitude, 3.0, epsilon = 1e-10);
    }
}
