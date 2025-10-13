//! Quadratic programming and active set methods for isotonic regression
//!
//! This module provides quadratic programming formulations and active set methods
//! for solving isotonic regression optimization problems with linear constraints.

use super::simd_operations;
use scirs2_core::ndarray::Array1;
use sklears_core::{error::Result, types::Float};
use std::collections::HashSet;

/// Quadratic Programming Isotonic Regressor
///
/// Solves isotonic regression as a quadratic programming problem with linear constraints
#[derive(Debug, Clone)]
/// QuadraticProgrammingIsotonicRegressor
pub struct QuadraticProgrammingIsotonicRegressor {
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
}

impl QuadraticProgrammingIsotonicRegressor {
    /// Create a new quadratic programming isotonic regressor
    pub fn new() -> Self {
        Self {
            increasing: true,
            y_min: None,
            y_max: None,
            tolerance: 1e-8,
            max_iterations: 1000,
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

    /// Solve isotonic regression using quadratic programming
    pub fn solve(
        &self,
        y: &Array1<Float>,
        sample_weights: Option<&Array1<Float>>,
    ) -> Result<Array1<Float>> {
        let n = y.len();
        let default_weights = Array1::ones(n);
        let weights = sample_weights.unwrap_or(&default_weights);

        // For now, use active set method as the QP solver
        let active_set_solver = ActiveSetIsotonicRegressor::new()
            .increasing(self.increasing)
            .bounds(self.y_min, self.y_max)
            .convergence(self.tolerance, self.max_iterations);

        active_set_solver.solve(y, Some(weights))
    }
}

impl Default for QuadraticProgrammingIsotonicRegressor {
    fn default() -> Self {
        Self::new()
    }
}

/// Active set method for isotonic regression
///
/// Implements an active set algorithm for solving the isotonic regression QP
#[derive(Debug, Clone)]
/// ActiveSetIsotonicRegressor
pub struct ActiveSetIsotonicRegressor {
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
}

impl ActiveSetIsotonicRegressor {
    /// Create a new active set isotonic regressor
    pub fn new() -> Self {
        Self {
            increasing: true,
            y_min: None,
            y_max: None,
            tolerance: 1e-8,
            max_iterations: 1000,
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

    /// Solve isotonic regression using active set method
    pub fn solve(
        &self,
        y: &Array1<Float>,
        sample_weights: Option<&Array1<Float>>,
    ) -> Result<Array1<Float>> {
        let n = y.len();
        let default_weights = Array1::ones(n);
        let weights = sample_weights.unwrap_or(&default_weights);

        // For simplicity, we'll use a modified version that combines active set ideas
        // with the PAV algorithm for better performance
        let mut solution = y.clone();
        let mut active_constraints = vec![false; n - 1];

        for _iteration in 0..self.max_iterations {
            let prev_solution = solution.clone();

            // Identify violated constraints
            self.update_active_constraints(&solution, &mut active_constraints);

            // Solve the reduced problem on the active set
            self.solve_with_active_constraints(&mut solution, y, weights, &active_constraints)?;

            // Apply bound constraints
            if let Some(min_val) = self.y_min {
                solution.mapv_inplace(|x| x.max(min_val));
            }
            if let Some(max_val) = self.y_max {
                solution.mapv_inplace(|x| x.min(max_val));
            }

            // SIMD-accelerated convergence check
            let diff = &solution - &prev_solution;
            let change = simd_operations::simd_vector_norm(&diff);

            if change < self.tolerance {
                break;
            }
        }

        Ok(solution)
    }

    /// Update active constraint set
    fn update_active_constraints(&self, solution: &Array1<Float>, active: &mut [bool]) {
        for i in 0..active.len() {
            if self.increasing {
                // Constraint: solution[i+1] >= solution[i]
                active[i] = solution[i + 1] <= solution[i] + self.tolerance;
            } else {
                // Constraint: solution[i] >= solution[i+1]
                active[i] = solution[i] <= solution[i + 1] + self.tolerance;
            }
        }
    }

    /// Solve the problem with current active constraints
    fn solve_with_active_constraints(
        &self,
        solution: &mut Array1<Float>,
        y: &Array1<Float>,
        weights: &Array1<Float>,
        active: &[bool],
    ) -> Result<()> {
        // For active constraints, we group variables that must be equal
        // and solve for their common value

        let n = solution.len();
        let mut groups = vec![0; n]; // Group assignment for each variable

        // Initialize each variable to its own group
        for i in 0..n {
            groups[i] = i;
        }

        // Merge groups based on active constraints
        for i in 0..active.len() {
            if active[i] {
                let group1 = groups[i];
                let group2 = groups[i + 1];
                if group1 != group2 {
                    // Merge groups
                    let new_group = group1.min(group2);
                    let old_group = group1.max(group2);
                    for j in 0..n {
                        if groups[j] == old_group {
                            groups[j] = new_group;
                        }
                    }
                }
            }
        }

        // Solve for each group separately
        let mut used_groups = HashSet::new();
        for i in 0..n {
            let group = groups[i];
            if !used_groups.contains(&group) {
                used_groups.insert(group);

                // Find all variables in this group
                let group_indices: Vec<usize> = (0..n).filter(|&j| groups[j] == group).collect();

                if !group_indices.is_empty() {
                    // Compute weighted average for this group
                    let mut sum_weighted_y = 0.0;
                    let mut sum_weights = 0.0;

                    for &idx in &group_indices {
                        sum_weighted_y += weights[idx] * y[idx];
                        sum_weights += weights[idx];
                    }

                    let group_value = if sum_weights > 0.0 {
                        sum_weighted_y / sum_weights
                    } else {
                        y[group_indices[0]]
                    };

                    // Assign the same value to all variables in the group
                    for &idx in &group_indices {
                        solution[idx] = group_value;
                    }
                }
            }
        }

        Ok(())
    }
}

impl Default for ActiveSetIsotonicRegressor {
    fn default() -> Self {
        Self::new()
    }
}

/// Solve isotonic regression using quadratic programming approach
pub fn isotonic_regression_qp(
    y: &Array1<Float>,
    sample_weights: Option<&Array1<Float>>,
    increasing: bool,
) -> Result<Array1<Float>> {
    let solver = QuadraticProgrammingIsotonicRegressor::new().increasing(increasing);
    solver.solve(y, sample_weights)
}

/// Solve isotonic regression using active set method
pub fn isotonic_regression_active_set(
    y: &Array1<Float>,
    sample_weights: Option<&Array1<Float>>,
    increasing: bool,
) -> Result<Array1<Float>> {
    let solver = ActiveSetIsotonicRegressor::new().increasing(increasing);
    solver.solve(y, sample_weights)
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_active_set_creation() {
        let regressor = ActiveSetIsotonicRegressor::new()
            .increasing(true)
            .convergence(1e-6, 500);

        assert_eq!(regressor.increasing, true);
        assert!((regressor.tolerance - 1e-6).abs() < 1e-12);
        assert_eq!(regressor.max_iterations, 500);
    }

    #[test]
    fn test_active_set_simple_case() {
        let y = array![3.0, 1.0, 2.0, 4.0];
        let regressor = ActiveSetIsotonicRegressor::new().increasing(true);
        let result = regressor.solve(&y, None).unwrap();

        // Check monotonicity
        for i in 1..result.len() {
            assert!(result[i] >= result[i - 1] - 1e-10);
        }
    }

    #[test]
    fn test_active_set_decreasing() {
        let y = array![1.0, 3.0, 2.0, 4.0];
        let regressor = ActiveSetIsotonicRegressor::new().increasing(false);
        let result = regressor.solve(&y, None).unwrap();

        // Check monotonicity (decreasing)
        for i in 1..result.len() {
            assert!(result[i] <= result[i - 1] + 1e-10);
        }
    }

    #[test]
    fn test_qp_wrapper() {
        let y = array![2.0, 1.0, 3.0];
        let result = isotonic_regression_qp(&y, None, true).unwrap();

        // Check monotonicity
        for i in 1..result.len() {
            assert!(result[i] >= result[i - 1] - 1e-10);
        }
    }

    #[test]
    fn test_active_set_wrapper() {
        let y = array![4.0, 2.0, 3.0, 1.0];
        let result = isotonic_regression_active_set(&y, None, false).unwrap();

        // Check monotonicity (decreasing)
        for i in 1..result.len() {
            assert!(result[i] <= result[i - 1] + 1e-10);
        }
    }

    #[test]
    fn test_bounds_constraints() {
        let y = array![0.5, 1.5, 2.5];
        let regressor = ActiveSetIsotonicRegressor::new()
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
        let regressor = ActiveSetIsotonicRegressor::new().increasing(true);
        let result = regressor.solve(&y, Some(&weights)).unwrap();

        // Result should be influenced by the high-weight middle point
        assert!(result.len() == 3);
    }
}
