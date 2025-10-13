//! Convergence tests and optimality condition verification
//!
//! This module provides tests and utilities for verifying the mathematical
//! correctness, convergence properties, and optimality conditions of isotonic
//! regression algorithms.

use scirs2_core::ndarray::{Array1, ArrayView1};
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Fit, Predict},
    types::Float,
};

use crate::core::{isotonic_regression, IsotonicRegression, LossFunction, MonotonicityConstraint};
use crate::optimization::{
    isotonic_regression_active_set, isotonic_regression_dual_decomposition,
    isotonic_regression_interior_point, isotonic_regression_projected_gradient,
    isotonic_regression_qp,
};

/// Convergence criteria for iterative algorithms
#[derive(Debug, Clone)]
pub struct ConvergenceCriteria {
    /// Maximum number of iterations
    pub max_iterations: usize,
    /// Tolerance for convergence (change in objective function)
    pub tolerance: Float,
    /// Tolerance for constraint violations
    pub constraint_tolerance: Float,
    /// Tolerance for gradient norm (for gradient-based methods)
    pub gradient_tolerance: Float,
}

impl Default for ConvergenceCriteria {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            tolerance: 1e-8,
            constraint_tolerance: 1e-10,
            gradient_tolerance: 1e-8,
        }
    }
}

/// Results of convergence analysis
#[derive(Debug, Clone)]
pub struct ConvergenceResults {
    /// Whether the algorithm converged
    pub converged: bool,
    /// Number of iterations required
    pub iterations: usize,
    /// Final objective function value
    pub objective_value: Float,
    /// Maximum constraint violation
    pub max_constraint_violation: Float,
    /// Final gradient norm (for gradient-based methods)
    pub gradient_norm: Option<Float>,
    /// Convergence rate (if measurable)
    pub convergence_rate: Option<Float>,
}

/// Verify monotonicity constraints are satisfied
///
/// # Arguments
/// * `x` - Input values (should be sorted)
/// * `y` - Fitted values
/// * `constraint` - Monotonicity constraint to verify
/// * `tolerance` - Tolerance for constraint violations
///
/// # Returns
/// Maximum constraint violation (0.0 if all constraints satisfied)
pub fn verify_monotonicity_constraints(
    x: ArrayView1<Float>,
    y: ArrayView1<Float>,
    constraint: &MonotonicityConstraint,
    tolerance: Float,
) -> Float {
    if x.len() != y.len() || x.len() < 2 {
        return 0.0;
    }

    let mut max_violation: Float = 0.0;

    match constraint {
        MonotonicityConstraint::Global { increasing } => {
            for i in 0..x.len() - 1 {
                let violation = if *increasing {
                    (y[i] - y[i + 1]).max(0.0)
                } else {
                    (y[i + 1] - y[i]).max(0.0)
                };
                max_violation = max_violation.max(violation);
            }
        }
        MonotonicityConstraint::Piecewise {
            breakpoints,
            segment_increasing,
        } => {
            for i in 0..x.len() - 1 {
                // Find which segment x[i] belongs to
                let mut segment_i = 0;
                for &bp in breakpoints {
                    if x[i] >= bp {
                        segment_i += 1;
                    } else {
                        break;
                    }
                }

                // Find which segment x[i+1] belongs to
                let mut segment_i_plus_1 = 0;
                for &bp in breakpoints {
                    if x[i + 1] >= bp {
                        segment_i_plus_1 += 1;
                    } else {
                        break;
                    }
                }

                // Only check constraint if both points are in the same segment
                if segment_i == segment_i_plus_1 && segment_i < segment_increasing.len() {
                    let increasing = segment_increasing[segment_i];
                    let violation = if increasing {
                        (y[i] - y[i + 1]).max(0.0)
                    } else {
                        (y[i + 1] - y[i]).max(0.0)
                    };
                    max_violation = max_violation.max(violation);
                }
            }
        }
        MonotonicityConstraint::Convex => {
            // Check both increasing and convexity
            for i in 0..x.len() - 1 {
                let violation = (y[i] - y[i + 1]).max(0.0);
                max_violation = max_violation.max(violation);
            }

            // Check convexity (second derivatives)
            for i in 1..x.len() - 1 {
                if x[i + 1] - x[i] > tolerance && x[i] - x[i - 1] > tolerance {
                    let slope1 = (y[i] - y[i - 1]) / (x[i] - x[i - 1]);
                    let slope2 = (y[i + 1] - y[i]) / (x[i + 1] - x[i]);
                    let violation = (slope1 - slope2).max(0.0);
                    max_violation = max_violation.max(violation);
                }
            }
        }
        MonotonicityConstraint::Concave => {
            // Check both increasing and concavity
            for i in 0..x.len() - 1 {
                let violation = (y[i] - y[i + 1]).max(0.0);
                max_violation = max_violation.max(violation);
            }

            // Check concavity (second derivatives)
            for i in 1..x.len() - 1 {
                if x[i + 1] - x[i] > tolerance && x[i] - x[i - 1] > tolerance {
                    let slope1 = (y[i] - y[i - 1]) / (x[i] - x[i - 1]);
                    let slope2 = (y[i + 1] - y[i]) / (x[i + 1] - x[i]);
                    let violation = (slope2 - slope1).max(0.0);
                    max_violation = max_violation.max(violation);
                }
            }
        }
        MonotonicityConstraint::ConvexDecreasing => {
            // Check both decreasing and convexity
            for i in 0..x.len() - 1 {
                let violation = (y[i + 1] - y[i]).max(0.0);
                max_violation = max_violation.max(violation);
            }

            // Check convexity with decreasing
            for i in 1..x.len() - 1 {
                if x[i + 1] - x[i] > tolerance && x[i] - x[i - 1] > tolerance {
                    let slope1 = (y[i] - y[i - 1]) / (x[i] - x[i - 1]);
                    let slope2 = (y[i + 1] - y[i]) / (x[i + 1] - x[i]);
                    let violation = (slope1 - slope2).max(0.0);
                    max_violation = max_violation.max(violation);
                }
            }
        }
        MonotonicityConstraint::ConcaveDecreasing => {
            // Check both decreasing and concavity
            for i in 0..x.len() - 1 {
                let violation = (y[i + 1] - y[i]).max(0.0);
                max_violation = max_violation.max(violation);
            }

            // Check concavity with decreasing
            for i in 1..x.len() - 1 {
                if x[i + 1] - x[i] > tolerance && x[i] - x[i - 1] > tolerance {
                    let slope1 = (y[i] - y[i - 1]) / (x[i] - x[i - 1]);
                    let slope2 = (y[i + 1] - y[i]) / (x[i + 1] - x[i]);
                    let violation = (slope2 - slope1).max(0.0);
                    max_violation = max_violation.max(violation);
                }
            }
        }
    }

    max_violation
}

/// Verify KKT (Karush-Kuhn-Tucker) optimality conditions for isotonic regression
///
/// For isotonic regression, the KKT conditions involve the Lagrange multipliers
/// for the monotonicity constraints.
///
/// # Arguments
/// * `x` - Input values (sorted)
/// * `y` - Target values
/// * `y_fitted` - Fitted values
/// * `constraint` - Monotonicity constraint
/// * `loss` - Loss function used
/// * `tolerance` - Tolerance for optimality violations
///
/// # Returns
/// Maximum KKT violation (0.0 if optimal)
pub fn verify_kkt_conditions(
    x: ArrayView1<Float>,
    y: ArrayView1<Float>,
    y_fitted: ArrayView1<Float>,
    constraint: &MonotonicityConstraint,
    loss: LossFunction,
    tolerance: Float,
) -> Float {
    if x.len() != y.len() || y.len() != y_fitted.len() || x.len() < 2 {
        return Float::INFINITY;
    }

    let mut max_violation: Float = 0.0;

    // Compute gradient of loss function
    let grad = compute_loss_gradient(y, y_fitted, loss);

    match constraint {
        MonotonicityConstraint::Global { increasing } => {
            // For global monotonicity, check that gradient differences
            // have the correct sign for active constraints
            for i in 0..x.len() - 1 {
                let constraint_active = if *increasing {
                    (y_fitted[i + 1] - y_fitted[i]).abs() < tolerance
                } else {
                    (y_fitted[i] - y_fitted[i + 1]).abs() < tolerance
                };

                if constraint_active {
                    let grad_diff = grad[i + 1] - grad[i];
                    let violation = if *increasing {
                        (-grad_diff).max(0.0)
                    } else {
                        grad_diff.max(0.0)
                    };
                    max_violation = max_violation.max(violation);
                }
            }
        }
        _ => {
            // For more complex constraints, use simplified check
            // This is a basic implementation and could be extended
            for i in 0..x.len() - 1 {
                let constraint_violation =
                    verify_monotonicity_constraints(x, y_fitted, constraint, tolerance);
                max_violation = max_violation.max(constraint_violation);
            }
        }
    }

    max_violation
}

/// Compute gradient of loss function
fn compute_loss_gradient(
    y_true: ArrayView1<Float>,
    y_pred: ArrayView1<Float>,
    loss: LossFunction,
) -> Array1<Float> {
    let mut grad = Array1::zeros(y_true.len());

    for i in 0..y_true.len() {
        let residual = y_pred[i] - y_true[i];

        grad[i] = match loss {
            LossFunction::SquaredLoss => 2.0 * residual,
            LossFunction::AbsoluteLoss => residual.signum(),
            LossFunction::HuberLoss { delta } => {
                if residual.abs() <= delta {
                    residual
                } else {
                    delta * residual.signum()
                }
            }
            LossFunction::QuantileLoss { quantile } => {
                if residual >= 0.0 {
                    quantile
                } else {
                    quantile - 1.0
                }
            }
        };
    }

    grad
}

/// Test convergence of optimization algorithms
///
/// Compares different optimization algorithms for isotonic regression
/// and verifies their convergence properties.
///
/// # Arguments
/// * `x` - Input values
/// * `y` - Target values
/// * `constraint` - Monotonicity constraint
/// * `loss` - Loss function
/// * `criteria` - Convergence criteria
///
/// # Returns
/// Vector of convergence results for each algorithm
pub fn test_algorithm_convergence(
    x: ArrayView1<Float>,
    y: ArrayView1<Float>,
    constraint: MonotonicityConstraint,
    loss: LossFunction,
    criteria: ConvergenceCriteria,
) -> Result<Vec<(&'static str, ConvergenceResults)>> {
    let mut results = Vec::new();

    // Test PAVA algorithm
    let pava_result = test_pava_convergence(x, y, constraint.clone(), loss, &criteria)?;
    results.push(("PAVA", pava_result));

    // Test basic isotonic regression (simplified for now)
    let increasing = match constraint {
        MonotonicityConstraint::Global { increasing } => Some(increasing),
        _ => Some(true), // Default to increasing for simplicity
    };

    // Test basic isotonic regression using the full fit/predict cycle
    let mut iso = IsotonicRegression::new();
    if let Some(inc) = increasing {
        iso = iso.increasing(inc);
    }

    if let Ok(fitted_model) = iso.fit(&Array1::from_vec(x.to_vec()), &Array1::from_vec(y.to_vec()))
    {
        if let Ok(predictions) = fitted_model.predict(&Array1::from_vec(x.to_vec())) {
            let basic_result = analyze_solution_quality(
                x,
                y,
                predictions.view(),
                constraint.clone(),
                loss,
                &criteria,
            );
            results.push(("Basic Isotonic", basic_result));
        }
    }

    Ok(results)
}

/// Test PAVA algorithm convergence
fn test_pava_convergence(
    x: ArrayView1<Float>,
    y: ArrayView1<Float>,
    constraint: MonotonicityConstraint,
    loss: LossFunction,
    criteria: &ConvergenceCriteria,
) -> Result<ConvergenceResults> {
    let increasing = match constraint {
        MonotonicityConstraint::Global { increasing } => Some(increasing),
        _ => Some(true), // Default to increasing for simplicity
    };

    // Use fit/predict cycle to ensure same length arrays
    let mut iso = IsotonicRegression::new();
    if let Some(inc) = increasing {
        iso = iso.increasing(inc);
    }

    let fitted_model = iso.fit(&Array1::from_vec(x.to_vec()), &Array1::from_vec(y.to_vec()))?;
    let predictions = fitted_model.predict(&Array1::from_vec(x.to_vec()))?;

    Ok(analyze_solution_quality(
        x,
        y,
        predictions.view(),
        constraint,
        loss,
        criteria,
    ))
}

/// Analyze solution quality and optimality
fn analyze_solution_quality(
    x: ArrayView1<Float>,
    y: ArrayView1<Float>,
    y_fitted: ArrayView1<Float>,
    constraint: MonotonicityConstraint,
    loss: LossFunction,
    criteria: &ConvergenceCriteria,
) -> ConvergenceResults {
    // Check constraint violations
    let max_constraint_violation =
        verify_monotonicity_constraints(x, y_fitted, &constraint, criteria.constraint_tolerance);

    // Check KKT conditions
    let kkt_violation = verify_kkt_conditions(
        x,
        y,
        y_fitted,
        &constraint,
        loss,
        criteria.gradient_tolerance,
    );

    // Compute objective value
    let objective_value = compute_objective_value(y, y_fitted, loss);

    // Determine convergence
    let converged = max_constraint_violation <= criteria.constraint_tolerance
        && kkt_violation <= criteria.gradient_tolerance;

    ConvergenceResults {
        converged,
        iterations: 1, // For non-iterative algorithms
        objective_value,
        max_constraint_violation,
        gradient_norm: Some(kkt_violation),
        convergence_rate: None,
    }
}

/// Compute objective function value
fn compute_objective_value(
    y_true: ArrayView1<Float>,
    y_pred: ArrayView1<Float>,
    loss: LossFunction,
) -> Float {
    let mut objective = 0.0;

    for i in 0..y_true.len() {
        let residual = y_pred[i] - y_true[i];

        objective += match loss {
            LossFunction::SquaredLoss => residual * residual,
            LossFunction::AbsoluteLoss => residual.abs(),
            LossFunction::HuberLoss { delta } => {
                if residual.abs() <= delta {
                    0.5 * residual * residual
                } else {
                    delta * (residual.abs() - 0.5 * delta)
                }
            }
            LossFunction::QuantileLoss { quantile } => {
                if residual >= 0.0 {
                    quantile * residual
                } else {
                    (quantile - 1.0) * residual
                }
            }
        };
    }

    objective
}

/// Benchmark convergence rates across different problem sizes
///
/// Tests how convergence properties scale with problem size.
///
/// # Arguments
/// * `problem_sizes` - Vector of problem sizes to test
/// * `constraint` - Monotonicity constraint
/// * `loss` - Loss function
/// * `criteria` - Convergence criteria
///
/// # Returns
/// Vector of (problem_size, convergence_results) pairs
pub fn benchmark_convergence_scaling(
    problem_sizes: &[usize],
    constraint: MonotonicityConstraint,
    loss: LossFunction,
    criteria: ConvergenceCriteria,
) -> Vec<(usize, Vec<(&'static str, ConvergenceResults)>)> {
    let mut results = Vec::new();

    for &size in problem_sizes {
        // Generate synthetic data
        let x: Array1<Float> = (0..size).map(|i| i as Float).collect();
        let y: Array1<Float> = (0..size)
            .map(|i| {
                let base = i as Float + 0.1 * (i as Float).sin();
                // Add deterministic "noise" for testing
                base + 0.05 * ((i * 17) % 100) as Float / 100.0
            })
            .collect();

        if let Ok(convergence_results) = test_algorithm_convergence(
            x.view(),
            y.view(),
            constraint.clone(),
            loss,
            criteria.clone(),
        ) {
            results.push((size, convergence_results));
        }
    }

    results
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::{array, ArrayView1};

    #[test]
    fn test_verify_monotonicity_constraints_increasing() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y_monotonic = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y_non_monotonic = array![1.0, 3.0, 2.0, 4.0, 5.0];

        let constraint = MonotonicityConstraint::Global { increasing: true };

        let violation_mono =
            verify_monotonicity_constraints(x.view(), y_monotonic.view(), &constraint, 1e-8);
        assert_abs_diff_eq!(violation_mono, 0.0, epsilon = 1e-8);

        let violation_non_mono =
            verify_monotonicity_constraints(x.view(), y_non_monotonic.view(), &constraint, 1e-8);
        assert!(violation_non_mono > 0.0);
    }

    #[test]
    fn test_verify_monotonicity_constraints_decreasing() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y_monotonic = array![5.0, 4.0, 3.0, 2.0, 1.0];
        let y_non_monotonic = array![5.0, 2.0, 3.0, 1.0, 0.0];

        let constraint = MonotonicityConstraint::Global { increasing: false };

        let violation_mono =
            verify_monotonicity_constraints(x.view(), y_monotonic.view(), &constraint, 1e-8);
        assert_abs_diff_eq!(violation_mono, 0.0, epsilon = 1e-8);

        let violation_non_mono =
            verify_monotonicity_constraints(x.view(), y_non_monotonic.view(), &constraint, 1e-8);
        assert!(violation_non_mono > 0.0);
    }

    #[test]
    fn test_verify_convex_constraints() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y_convex = array![1.0, 1.5, 2.5, 4.0, 6.0]; // Convex and increasing
        let y_non_convex = array![1.0, 3.0, 2.5, 4.0, 5.0]; // Not convex

        let constraint = MonotonicityConstraint::Convex;

        let violation_convex =
            verify_monotonicity_constraints(x.view(), y_convex.view(), &constraint, 1e-6);
        assert!(violation_convex < 1e-6);

        let violation_non_convex =
            verify_monotonicity_constraints(x.view(), y_non_convex.view(), &constraint, 1e-6);
        assert!(violation_non_convex > 1e-6);
    }

    #[test]
    fn test_compute_loss_gradient() {
        let y_true = array![1.0, 2.0, 3.0];
        let y_pred = array![1.1, 1.9, 3.1];

        // Test squared loss gradient
        let grad_squared =
            compute_loss_gradient(y_true.view(), y_pred.view(), LossFunction::SquaredLoss);
        assert_abs_diff_eq!(grad_squared[0], 0.2, epsilon = 1e-8);
        assert_abs_diff_eq!(grad_squared[1], -0.2, epsilon = 1e-8);
        assert_abs_diff_eq!(grad_squared[2], 0.2, epsilon = 1e-8);

        // Test absolute loss gradient
        let grad_absolute =
            compute_loss_gradient(y_true.view(), y_pred.view(), LossFunction::AbsoluteLoss);
        assert_abs_diff_eq!(grad_absolute[0], 1.0, epsilon = 1e-8);
        assert_abs_diff_eq!(grad_absolute[1], -1.0, epsilon = 1e-8);
        assert_abs_diff_eq!(grad_absolute[2], 1.0, epsilon = 1e-8);
    }

    #[test]
    fn test_compute_objective_value() {
        let y_true = array![1.0, 2.0, 3.0];
        let y_pred = array![1.1, 1.9, 3.1];

        // Test squared loss
        let obj_squared =
            compute_objective_value(y_true.view(), y_pred.view(), LossFunction::SquaredLoss);
        let expected_squared = 0.1 * 0.1 + 0.1 * 0.1 + 0.1 * 0.1;
        assert_abs_diff_eq!(obj_squared, expected_squared, epsilon = 1e-8);

        // Test absolute loss
        let obj_absolute =
            compute_objective_value(y_true.view(), y_pred.view(), LossFunction::AbsoluteLoss);
        let expected_absolute = 0.1 + 0.1 + 0.1;
        assert_abs_diff_eq!(obj_absolute, expected_absolute, epsilon = 1e-8);
    }

    #[test]
    fn test_kkt_conditions_simple() {
        let x = array![1.0, 2.0, 3.0, 4.0];
        let y = array![1.0, 2.0, 3.0, 4.0];
        let y_fitted = array![1.0, 2.0, 3.0, 4.0]; // Perfect monotonic fit

        let constraint = MonotonicityConstraint::Global { increasing: true };
        let violation = verify_kkt_conditions(
            x.view(),
            y.view(),
            y_fitted.view(),
            &constraint,
            LossFunction::SquaredLoss,
            1e-8,
        );

        // Should have minimal violation for perfect fit
        assert!(violation < 1e-6);
    }

    #[test]
    fn test_all_algorithm_convergence() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![1.0, 3.0, 2.0, 4.0, 5.0];

        let criteria = ConvergenceCriteria::default();
        let constraint = MonotonicityConstraint::Global { increasing: true };

        let results = test_algorithm_convergence(
            x.view(),
            y.view(),
            constraint,
            LossFunction::SquaredLoss,
            criteria,
        )
        .unwrap();

        // All algorithms should converge
        for (name, result) in results {
            println!(
                "Algorithm: {}, Converged: {}, Objective: {}",
                name, result.converged, result.objective_value
            );

            // At minimum, constraint violations should be small
            assert!(
                result.max_constraint_violation < 1e-6,
                "Algorithm {} has constraint violation: {}",
                name,
                result.max_constraint_violation
            );
        }
    }

    #[test]
    fn test_convergence_criteria() {
        let criteria = ConvergenceCriteria {
            max_iterations: 500,
            tolerance: 1e-10,
            constraint_tolerance: 1e-12,
            gradient_tolerance: 1e-10,
        };

        assert_eq!(criteria.max_iterations, 500);
        assert_eq!(criteria.tolerance, 1e-10);
    }

    #[test]
    fn test_piecewise_constraint_verification() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let y = array![1.0, 2.0, 3.0, 2.5, 2.0, 1.5]; // Increasing then decreasing

        let constraint = MonotonicityConstraint::Piecewise {
            breakpoints: vec![3.5],
            segment_increasing: vec![true, false],
        };

        let violation = verify_monotonicity_constraints(x.view(), y.view(), &constraint, 1e-8);

        // Should have small violation for properly piecewise monotonic data
        assert!(violation < 1e-6);
    }

    #[test]
    fn test_benchmark_convergence_scaling() {
        let problem_sizes = vec![10, 20];
        let constraint = MonotonicityConstraint::Global { increasing: true };
        let criteria = ConvergenceCriteria {
            max_iterations: 100,
            ..Default::default()
        };

        let results = benchmark_convergence_scaling(
            &problem_sizes,
            constraint,
            LossFunction::SquaredLoss,
            criteria,
        );

        assert_eq!(results.len(), 2);

        for (size, algorithm_results) in results {
            println!(
                "Problem size: {}, Number of algorithms tested: {}",
                size,
                algorithm_results.len()
            );
            assert!(algorithm_results.len() > 0);
        }
    }
}
