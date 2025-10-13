//! Constraint handling and validation for isotonic regression
//!
//! This module provides utilities for applying and validating monotonicity
//! constraints, bounds, and other constraints used in isotonic regression.

use crate::{pool_adjacent_violators_l2, MonotonicityConstraint};
use scirs2_core::ndarray::Array1;
use sklears_core::{error::Result, types::Float};

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
        MonotonicityConstraint::Piecewise { .. } => {
            // TODO: Implement piecewise constraints
            pool_adjacent_violators_l2(values, weights, true)
        }
        MonotonicityConstraint::Convex => {
            // TODO: Implement convex constraints
            pool_adjacent_violators_l2(values, weights, true)
        }
        MonotonicityConstraint::Concave => {
            // TODO: Implement concave constraints
            pool_adjacent_violators_l2(values, weights, true)
        }
        MonotonicityConstraint::ConvexDecreasing => {
            // TODO: Implement convex decreasing constraints
            pool_adjacent_violators_l2(values, weights, false)
        }
        MonotonicityConstraint::ConcaveDecreasing => {
            // TODO: Implement concave decreasing constraints
            pool_adjacent_violators_l2(values, weights, false)
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
        MonotonicityConstraint::Piecewise { .. } => {
            // TODO: Implement piecewise validation
            true
        }
        MonotonicityConstraint::Convex => {
            // TODO: Implement convex validation
            validate_monotonicity(values, MonotonicityConstraint::Increasing, tolerance)
        }
        MonotonicityConstraint::Concave => {
            // TODO: Implement concave validation
            validate_monotonicity(values, MonotonicityConstraint::Increasing, tolerance)
        }
        MonotonicityConstraint::ConvexDecreasing => {
            // TODO: Implement convex decreasing validation
            validate_monotonicity(values, MonotonicityConstraint::Decreasing, tolerance)
        }
        MonotonicityConstraint::ConcaveDecreasing => {
            // TODO: Implement concave decreasing validation
            validate_monotonicity(values, MonotonicityConstraint::Decreasing, tolerance)
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
        MonotonicityConstraint::Piecewise { .. } => {
            // TODO: Implement piecewise violation counting
        }
        MonotonicityConstraint::Convex => {
            // TODO: Implement convex violation counting
        }
        MonotonicityConstraint::Concave => {
            // TODO: Implement concave violation counting
        }
        MonotonicityConstraint::ConvexDecreasing => {
            // TODO: Implement convex decreasing violation counting
        }
        MonotonicityConstraint::ConcaveDecreasing => {
            // TODO: Implement concave decreasing violation counting
        }
    }

    violations
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
        MonotonicityConstraint::Piecewise { .. } => {
            // TODO: Implement piecewise violation magnitude
        }
        MonotonicityConstraint::Convex => {
            // TODO: Implement convex violation magnitude
        }
        MonotonicityConstraint::Concave => {
            // TODO: Implement concave violation magnitude
        }
        MonotonicityConstraint::ConvexDecreasing => {
            // TODO: Implement convex decreasing violation magnitude
        }
        MonotonicityConstraint::ConcaveDecreasing => {
            // TODO: Implement concave decreasing violation magnitude
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
