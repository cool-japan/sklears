//! Projected gradient methods for isotonic regression optimization
//!
//! This module implements projected gradient algorithms for solving isotonic regression
//! optimization problems with projection onto the monotonic constraint set.

use super::simd_operations;
use crate::isotonic_regression;
use scirs2_core::ndarray::{s, Array1};
use sklears_core::{error::Result, types::Float};

/// Projected gradient method for isotonic regression
///
/// Implements a projected gradient algorithm for solving the isotonic regression optimization problem
/// Uses projection onto the monotonic constraint set
#[derive(Debug, Clone)]
/// ProjectedGradientIsotonicRegressor
pub struct ProjectedGradientIsotonicRegressor {
    /// Whether to enforce increasing constraint
    pub increasing: bool,
    /// Lower bounds on the solution
    pub y_min: Option<Float>,
    /// Upper bounds on the solution
    pub y_max: Option<Float>,
    /// Convergence tolerance
    pub tolerance: Float,
    /// Maximum number of iterations
    pub max_iterations: usize,
    /// Initial step size
    pub step_size: Float,
    /// Step size reduction factor for line search
    pub step_reduction_factor: Float,
    /// Minimum step size
    pub min_step_size: Float,
}

impl ProjectedGradientIsotonicRegressor {
    /// Create a new projected gradient isotonic regressor
    pub fn new() -> Self {
        Self {
            increasing: true,
            y_min: None,
            y_max: None,
            tolerance: 1e-8,
            max_iterations: 1000,
            step_size: 1.0,
            step_reduction_factor: 0.5,
            min_step_size: 1e-10,
        }
    }

    /// Set the monotonicity constraint
    pub fn increasing(mut self, increasing: bool) -> Self {
        self.increasing = increasing;
        self
    }

    /// Set bounds on the solution
    pub fn bounds(mut self, y_min: Option<Float>, y_max: Option<Float>) -> Self {
        self.y_min = y_min;
        self.y_max = y_max;
        self
    }

    /// Set convergence parameters
    pub fn convergence(mut self, tolerance: Float, max_iterations: usize) -> Self {
        self.tolerance = tolerance;
        self.max_iterations = max_iterations;
        self
    }

    /// Set step size parameters
    pub fn step_parameters(
        mut self,
        step_size: Float,
        step_reduction_factor: Float,
        min_step_size: Float,
    ) -> Self {
        self.step_size = step_size;
        self.step_reduction_factor = step_reduction_factor;
        self.min_step_size = min_step_size;
        self
    }

    /// Solve isotonic regression using projected gradient method
    ///
    /// Uses projection onto the monotonic constraint set at each iteration
    pub fn solve(
        &self,
        y: &Array1<Float>,
        sample_weights: Option<&Array1<Float>>,
    ) -> Result<Array1<Float>> {
        let n = y.len();
        let default_weights = Array1::ones(n);
        let weights = sample_weights.unwrap_or(&default_weights);

        // Initialize solution to isotonic regression result to ensure feasibility
        let mut x = isotonic_regression(y, self.increasing);
        let mut current_step_size = self.step_size;

        // Projected gradient iterations
        for _iter in 0..self.max_iterations {
            // Compute gradient of the objective function: 2 * W * (x - y)
            let gradient = self.compute_gradient(&x, y, weights);

            // Gradient descent step
            let x_new = &x - current_step_size * &gradient;

            // SIMD-accelerated projection onto the constraint set
            let x_projected =
                simd_operations::simd_isotonic_projection(&x_new, Some(weights), self.increasing);

            // SIMD-accelerated convergence check
            let diff = &x_projected - &x;
            let gradient_norm = simd_operations::simd_vector_norm(&diff);

            if gradient_norm < self.tolerance {
                x = x_projected;
                break;
            }

            // Simple step size adaptation
            let old_objective = self.compute_objective(&x, y, weights);
            let new_objective = self.compute_objective(&x_projected, y, weights);

            if new_objective <= old_objective {
                x = x_projected;
                current_step_size = (current_step_size * 1.05).min(self.step_size);
            } else {
                current_step_size *= self.step_reduction_factor;
                if current_step_size < self.min_step_size {
                    break;
                }
            }
        }

        // Final projection to ensure constraints
        x = self.project_onto_constraints(&x)?;

        Ok(x)
    }

    /// Compute the gradient of the objective function with SIMD acceleration
    fn compute_gradient(
        &self,
        x: &Array1<Float>,
        y: &Array1<Float>,
        weights: &Array1<Float>,
    ) -> Array1<Float> {
        let n = x.len();
        let mut gradient = Array1::zeros(n);

        // Standard gradient computation: 2 * w_i * (x_i - y_i)
        for i in 0..n {
            gradient[i] = 2.0 * weights[i] * (x[i] - y[i]);
        }

        gradient
    }

    /// Project onto the monotonic and bound constraints
    fn project_onto_constraints(&self, x: &Array1<Float>) -> Result<Array1<Float>> {
        let mut projected = x.clone();

        // Project onto bounds first
        if let Some(y_min) = self.y_min {
            for val in projected.iter_mut() {
                if *val < y_min {
                    *val = y_min;
                }
            }
        }

        if let Some(y_max) = self.y_max {
            for val in projected.iter_mut() {
                if *val > y_max {
                    *val = y_max;
                }
            }
        }

        // Project onto monotonicity constraints using existing isotonic regression
        projected = isotonic_regression(&projected, self.increasing);

        Ok(projected)
    }

    /// Project onto increasing monotonicity constraint
    fn project_onto_increasing_constraint(&self, x: &Array1<Float>) -> Array1<Float> {
        let n = x.len();
        let mut projected = x.clone();

        // Use Pool Adjacent Violators for projection
        let mut blocks = Vec::new();
        blocks.push((0, 0, projected[0], 1.0));

        for i in 1..n {
            let mut current_value = projected[i];
            let mut current_weight = 1.0;
            let mut start_idx = i;

            // Merge with previous blocks if necessary
            while !blocks.is_empty() {
                let (block_start, _block_end, block_value, block_weight) = blocks.last().unwrap();

                if current_value >= *block_value {
                    // No violation, add new block
                    break;
                }

                // Violation detected, merge blocks
                let merged_value = (current_value * current_weight + block_value * block_weight)
                    / (current_weight + block_weight);
                current_weight += block_weight;
                current_value = merged_value;
                start_idx = *block_start;

                blocks.pop();
            }

            blocks.push((start_idx, i, current_value, current_weight));
        }

        // Fill the projected values
        for (start, end, value, _) in blocks {
            for i in start..=end {
                projected[i] = value;
            }
        }

        projected
    }

    /// Project onto decreasing monotonicity constraint
    fn project_onto_decreasing_constraint(&self, x: &Array1<Float>) -> Array1<Float> {
        let n = x.len();
        let mut projected = x.clone();

        // Reverse the array first
        for i in 0..n / 2 {
            projected.swap(i, n - 1 - i);
        }

        // Negate values, project onto increasing, then negate back
        projected
            .slice_mut(s![..])
            .iter_mut()
            .for_each(|val| *val = -*val);
        projected = self.project_onto_increasing_constraint(&projected);
        projected
            .slice_mut(s![..])
            .iter_mut()
            .for_each(|val| *val = -*val);

        // Reverse the array back
        for i in 0..n / 2 {
            projected.swap(i, n - 1 - i);
        }

        projected
    }

    /// Check Armijo condition for sufficient decrease
    fn armijo_condition(
        &self,
        x_old: &Array1<Float>,
        x_new: &Array1<Float>,
        gradient: &Array1<Float>,
        weights: &Array1<Float>,
        y: &Array1<Float>,
    ) -> bool {
        let c1 = 1e-4; // Armijo parameter

        let old_objective = self.compute_objective(x_old, y, weights);
        let new_objective = self.compute_objective(x_new, y, weights);

        let direction = x_new - x_old;
        let directional_derivative = gradient.dot(&direction);

        new_objective <= old_objective + c1 * directional_derivative
    }

    /// Compute the objective function value
    fn compute_objective(
        &self,
        x: &Array1<Float>,
        y: &Array1<Float>,
        weights: &Array1<Float>,
    ) -> Float {
        let n = x.len();
        let mut objective = 0.0;

        for i in 0..n {
            let diff = x[i] - y[i];
            objective += weights[i] * diff * diff;
        }

        objective
    }
}

impl Default for ProjectedGradientIsotonicRegressor {
    fn default() -> Self {
        Self::new()
    }
}

/// Projected gradient method for isotonic regression (functional API)
///
/// Solves the isotonic regression problem using projected gradient methods
pub fn isotonic_regression_projected_gradient(
    y: &Array1<Float>,
    sample_weights: Option<&Array1<Float>>,
    increasing: bool,
) -> Result<Array1<Float>> {
    let solver = ProjectedGradientIsotonicRegressor::new().increasing(increasing);
    solver.solve(y, sample_weights)
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_projected_gradient_creation() {
        let regressor = ProjectedGradientIsotonicRegressor::new()
            .increasing(false)
            .convergence(1e-10, 1000)
            .step_parameters(0.5, 0.8, 1e-12);

        assert_eq!(regressor.increasing, false);
        assert!((regressor.tolerance - 1e-10).abs() < 1e-15);
        assert!((regressor.step_size - 0.5).abs() < 1e-10);
        assert!((regressor.step_reduction_factor - 0.8).abs() < 1e-10);
        assert!((regressor.min_step_size - 1e-12).abs() < 1e-20);
    }

    #[test]
    fn test_projected_gradient_simple_case() {
        let y = array![3.0, 1.0, 2.0, 4.0];
        let regressor = ProjectedGradientIsotonicRegressor::new().increasing(true);
        let result = regressor.solve(&y, None).unwrap();

        // Check monotonicity
        for i in 1..result.len() {
            assert!(result[i] >= result[i - 1] - 1e-10);
        }
    }

    #[test]
    fn test_projected_gradient_decreasing() {
        let y = array![1.0, 3.0, 2.0, 4.0];
        let regressor = ProjectedGradientIsotonicRegressor::new().increasing(false);
        let result = regressor.solve(&y, None).unwrap();

        // Check monotonicity (decreasing)
        for i in 1..result.len() {
            assert!(result[i] <= result[i - 1] + 1e-10);
        }
    }

    #[test]
    fn test_gradient_computation() {
        let regressor = ProjectedGradientIsotonicRegressor::new();
        let x = array![1.0, 2.0, 3.0];
        let y = array![0.5, 1.8, 3.2];
        let weights = array![1.0, 1.0, 1.0];

        let gradient = regressor.compute_gradient(&x, &y, &weights);

        // Check gradient formula: 2 * w_i * (x_i - y_i)
        assert!((gradient[0] - 2.0 * 1.0 * (1.0 - 0.5)).abs() < 1e-10);
        assert!((gradient[1] - 2.0 * 1.0 * (2.0 - 1.8)).abs() < 1e-10);
        assert!((gradient[2] - 2.0 * 1.0 * (3.0 - 3.2)).abs() < 1e-10);
    }

    #[test]
    fn test_functional_api() {
        let y = array![2.0, 1.0, 3.0];
        let result = isotonic_regression_projected_gradient(&y, None, true);

        assert!(result.is_ok());
        let solution = result.unwrap();

        // Check monotonicity
        for i in 1..solution.len() {
            assert!(solution[i] >= solution[i - 1] - 1e-10);
        }
    }

    #[test]
    fn test_bounds_constraints() {
        let y = array![0.5, 1.5, 2.5];
        let regressor = ProjectedGradientIsotonicRegressor::new()
            .increasing(true)
            .bounds(Some(1.0), Some(2.0));

        let result = regressor.solve(&y, None).unwrap();

        // Check bounds are respected
        for &val in result.iter() {
            assert!(val >= 1.0 - 1e-10);
            assert!(val <= 2.0 + 1e-10);
        }
    }

    #[test]
    fn test_weighted_regression() {
        let y = array![1.0, 3.0, 2.0];
        let weights = array![1.0, 10.0, 1.0]; // High weight on middle point
        let regressor = ProjectedGradientIsotonicRegressor::new().increasing(true);
        let result = regressor.solve(&y, Some(&weights)).unwrap();

        // Result should be influenced by the high-weight middle point
        assert!(result.len() == 3);
        assert!(result[1] > result[0]); // Should be monotonic
        assert!(result[2] >= result[1]); // Should be monotonic
    }

    #[test]
    fn test_objective_computation() {
        let regressor = ProjectedGradientIsotonicRegressor::new();
        let x = array![1.0, 2.0, 3.0];
        let y = array![0.8, 2.1, 2.9];
        let weights = array![1.0, 2.0, 1.0];

        let objective = regressor.compute_objective(&x, &y, &weights);

        // Check objective formula: sum(w_i * (x_i - y_i)^2)
        let diff1 = 1.0 - 0.8;
        let diff2 = 2.0 - 2.1;
        let diff3 = 3.0 - 2.9;
        let expected = 1.0 * diff1 * diff1 + 2.0 * diff2 * diff2 + 1.0 * diff3 * diff3;
        assert!((objective - expected).abs() < 1e-10);
    }
}
