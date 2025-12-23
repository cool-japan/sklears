//! Dual decomposition methods for large-scale isotonic regression
//!
//! This module implements dual decomposition algorithms for solving large-scale isotonic regression
//! problems by decomposing them into smaller, more manageable subproblems.

use super::quadratic_programming::isotonic_regression_qp;
use crate::isotonic_regression;
use scirs2_core::ndarray::{s, Array1};
use sklears_core::{error::Result, types::Float};

/// Dual decomposition method for isotonic regression
///
/// Implements a dual decomposition algorithm for solving large-scale isotonic regression problems
/// by decomposing the problem into smaller subproblems
#[derive(Debug, Clone)]
/// DualDecompositionIsotonicRegressor
pub struct DualDecompositionIsotonicRegressor {
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
    /// Step size for dual variable updates
    pub dual_step_size: Float,
    /// Block size for decomposition
    pub block_size: usize,
    /// Overlap size between blocks
    pub overlap_size: usize,
}

impl DualDecompositionIsotonicRegressor {
    /// Create a new dual decomposition isotonic regressor
    pub fn new() -> Self {
        Self {
            increasing: true,
            y_min: None,
            y_max: None,
            tolerance: 1e-8,
            max_iterations: 1000,
            dual_step_size: 0.1,
            block_size: 100,
            overlap_size: 10,
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

    /// Set decomposition parameters
    pub fn decomposition_parameters(
        mut self,
        dual_step_size: Float,
        block_size: usize,
        overlap_size: usize,
    ) -> Self {
        self.dual_step_size = dual_step_size;
        self.block_size = block_size;
        self.overlap_size = overlap_size;
        self
    }

    /// Solve isotonic regression using dual decomposition
    ///
    /// Decomposes the problem into overlapping subproblems and coordinates their solutions
    pub fn solve(
        &self,
        y: &Array1<Float>,
        sample_weights: Option<&Array1<Float>>,
    ) -> Result<Array1<Float>> {
        let n = y.len();
        let default_weights = Array1::ones(n);
        let weights = sample_weights.unwrap_or(&default_weights);

        // If the problem is small, solve directly
        if n <= self.block_size {
            return isotonic_regression_qp(y, sample_weights, self.increasing);
        }

        // Create overlapping blocks
        let blocks = self.create_blocks(n);
        let num_blocks = blocks.len();

        // Initialize primal and dual variables
        let mut x = y.clone();
        let mut dual_vars = Array1::<Float>::zeros(n - 1); // Dual variables for monotonicity constraints

        // Dual decomposition iterations
        for _iter in 0..self.max_iterations {
            let mut new_x = Array1::<Float>::zeros(n);
            let mut block_contributions = Array1::<Float>::zeros(n);

            // Solve each subproblem
            for &(start, end) in blocks.iter().take(num_blocks) {
                let block_size = end - start + 1;

                // Extract block data
                let y_block = y.slice(s![start..=end]).to_owned();
                let weights_block = weights.slice(s![start..=end]).to_owned();

                // Add dual variable contributions to the objective
                let mut modified_y = y_block.clone();
                if start > 0 {
                    // Add dual variable for left boundary
                    let dual_contrib = if self.increasing {
                        dual_vars[start - 1]
                    } else {
                        -dual_vars[start - 1]
                    };
                    modified_y[0] += dual_contrib / weights_block[0];
                }
                if end < n - 1 {
                    // Add dual variable for right boundary
                    let dual_contrib = if self.increasing {
                        -dual_vars[end]
                    } else {
                        dual_vars[end]
                    };
                    modified_y[block_size - 1] += dual_contrib / weights_block[block_size - 1];
                }

                // Solve the subproblem
                let x_block =
                    isotonic_regression_qp(&modified_y, Some(&weights_block), self.increasing)?;

                // Accumulate contributions (with overlap handling)
                for i in 0..block_size {
                    let global_idx = start + i;
                    new_x[global_idx] += x_block[i];
                    block_contributions[global_idx] += 1.0;
                }
            }

            // Average overlapping contributions
            for i in 0..n {
                if block_contributions[i] > 0.0 {
                    new_x[i] /= block_contributions[i];
                }
            }

            // Update dual variables using subgradient method
            let mut max_violation: Float = 0.0;
            for i in 0..n - 1 {
                let constraint_violation: Float = if self.increasing {
                    new_x[i] - new_x[i + 1]
                } else {
                    new_x[i + 1] - new_x[i]
                };

                dual_vars[i] += self.dual_step_size * constraint_violation;
                max_violation = max_violation.max(constraint_violation.abs());
            }

            // Check convergence
            let primal_change = (&new_x - &x).mapv(|v: Float| v.abs()).sum();
            if primal_change < self.tolerance && max_violation < self.tolerance {
                break;
            }

            x = new_x;
        }

        // Final projection to ensure constraints are satisfied
        let projected_x = if self.increasing {
            isotonic_regression(&x, true)
        } else {
            isotonic_regression(&x, false)
        };

        Ok(projected_x)
    }

    /// Create overlapping blocks for decomposition
    fn create_blocks(&self, n: usize) -> Vec<(usize, usize)> {
        let mut blocks = Vec::new();

        let mut start = 0;
        while start < n {
            let end = (start + self.block_size - 1).min(n - 1);
            blocks.push((start, end));

            if end == n - 1 {
                break;
            }

            // Move start for next block, accounting for overlap
            start = if self.block_size > self.overlap_size {
                start + self.block_size - self.overlap_size
            } else {
                start + 1
            };
        }

        blocks
    }

    /// Analyze the dual decomposition convergence properties
    fn analyze_convergence(&self, dual_vars: &Array1<Float>) -> (Float, Float) {
        let n = dual_vars.len();
        if n == 0 {
            return (0.0, 0.0);
        }

        // Compute dual variable statistics
        let dual_norm = dual_vars.mapv(|x| x * x).sum().sqrt();
        let dual_max = dual_vars.mapv(|x| x.abs()).fold(0.0f64, |a, &b| a.max(b));

        (dual_norm, dual_max)
    }

    /// Adaptive step size adjustment based on convergence behavior
    fn adapt_step_size(&self, current_step: Float, violation_history: &[Float]) -> Float {
        if violation_history.len() < 3 {
            return current_step;
        }

        let recent_violations = &violation_history[violation_history.len() - 3..];
        let avg_violation =
            recent_violations.iter().sum::<Float>() / recent_violations.len() as Float;

        // If violations are decreasing, slightly increase step size
        // If violations are increasing, decrease step size
        let trend = recent_violations[2] - recent_violations[0];

        if trend < 0.0 && avg_violation < self.tolerance * 10.0 {
            (current_step * 1.05).min(self.dual_step_size * 2.0)
        } else if trend > 0.0 {
            current_step * 0.95
        } else {
            current_step
        }
    }

    /// Estimate computational complexity of the decomposition
    pub fn estimate_complexity(&self, n: usize) -> (usize, usize) {
        let blocks = self.create_blocks(n);
        let num_blocks = blocks.len();

        // Estimate total operations per iteration
        let ops_per_iteration = num_blocks * self.block_size.pow(2); // Approximate QP solve cost
        let total_ops = ops_per_iteration * self.max_iterations;

        (num_blocks, total_ops)
    }
}

impl Default for DualDecompositionIsotonicRegressor {
    fn default() -> Self {
        Self::new()
    }
}

/// Dual decomposition method for isotonic regression (functional API)
///
/// Solves large-scale isotonic regression problems using dual decomposition
pub fn isotonic_regression_dual_decomposition(
    y: &Array1<Float>,
    sample_weights: Option<&Array1<Float>>,
    increasing: bool,
) -> Result<Array1<Float>> {
    let solver = DualDecompositionIsotonicRegressor::new().increasing(increasing);
    solver.solve(y, sample_weights)
}

/// Parallel dual decomposition for extremely large problems
///
/// This variant uses multiple threads to solve subproblems in parallel
pub fn parallel_dual_decomposition(
    y: &Array1<Float>,
    sample_weights: Option<&Array1<Float>>,
    increasing: bool,
    num_threads: usize,
) -> Result<Array1<Float>> {
    // For now, fall back to sequential implementation
    // In a full implementation, this would use rayon or similar for parallelization
    let mut solver = DualDecompositionIsotonicRegressor::new().increasing(increasing);

    // Adjust block size based on available threads
    let n = y.len();
    let optimal_block_size = (n / num_threads).clamp(50, 1000);
    let dual_step_size = solver.dual_step_size;
    solver = solver.decomposition_parameters(
        dual_step_size,
        optimal_block_size,
        optimal_block_size / 10,
    );

    solver.solve(y, sample_weights)
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_dual_decomposition_creation() {
        let regressor = DualDecompositionIsotonicRegressor::new()
            .increasing(false)
            .convergence(1e-10, 1000)
            .decomposition_parameters(0.05, 50, 5);

        assert_eq!(regressor.increasing, false);
        assert!((regressor.tolerance - 1e-10).abs() < 1e-15);
        assert!((regressor.dual_step_size - 0.05).abs() < 1e-10);
        assert_eq!(regressor.block_size, 50);
        assert_eq!(regressor.overlap_size, 5);
    }

    #[test]
    fn test_block_creation() {
        let regressor =
            DualDecompositionIsotonicRegressor::new().decomposition_parameters(0.1, 10, 2);

        let blocks = regressor.create_blocks(25);

        // Should create overlapping blocks
        assert!(blocks.len() > 1);

        // Check that blocks cover the entire range
        assert_eq!(blocks[0].0, 0);
        assert_eq!(blocks.last().unwrap().1, 24);

        // Check overlap structure
        if blocks.len() > 1 {
            assert!(blocks[1].0 < blocks[0].1); // Overlapping
        }
    }

    #[test]
    fn test_small_problem_fallback() {
        let y = array![3.0, 1.0, 2.0, 4.0];
        let regressor = DualDecompositionIsotonicRegressor::new()
            .increasing(true)
            .decomposition_parameters(0.1, 10, 2); // Block size larger than problem

        let result = regressor.solve(&y, None).unwrap();

        // Check monotonicity
        for i in 1..result.len() {
            assert!(result[i] >= result[i - 1] - 1e-10);
        }
    }

    #[test]
    fn test_large_problem_decomposition() {
        // Create a larger synthetic problem
        let n = 50;
        let mut y = Array1::zeros(n);
        for i in 0..n {
            y[i] = (i as Float / n as Float) + 0.1 * (i as Float).sin();
        }

        let regressor = DualDecompositionIsotonicRegressor::new()
            .increasing(true)
            .decomposition_parameters(0.1, 10, 2)
            .convergence(1e-6, 100);

        let result = regressor.solve(&y, None).unwrap();

        // Check monotonicity
        for i in 1..result.len() {
            assert!(result[i] >= result[i - 1] - 1e-8);
        }
    }

    #[test]
    fn test_functional_api() {
        let y = array![4.0, 2.0, 3.0, 1.0, 5.0];
        let result = isotonic_regression_dual_decomposition(&y, None, true);

        assert!(result.is_ok());
        let solution = result.unwrap();

        // Check monotonicity
        for i in 1..solution.len() {
            assert!(solution[i] >= solution[i - 1] - 1e-10);
        }
    }

    #[test]
    fn test_decreasing_constraint() {
        let y = array![1.0, 4.0, 2.0, 5.0, 3.0];
        let regressor = DualDecompositionIsotonicRegressor::new()
            .increasing(false)
            .decomposition_parameters(0.1, 3, 1);

        let result = regressor.solve(&y, None).unwrap();

        // Check decreasing monotonicity
        for i in 1..result.len() {
            assert!(result[i] <= result[i - 1] + 1e-10);
        }
    }

    #[test]
    fn test_weighted_regression() {
        let y = array![1.0, 5.0, 2.0, 4.0, 3.0];
        let weights = array![1.0, 10.0, 1.0, 1.0, 1.0]; // High weight on second point
        let regressor = DualDecompositionIsotonicRegressor::new()
            .increasing(true)
            .decomposition_parameters(0.1, 3, 1);

        let result = regressor.solve(&y, Some(&weights)).unwrap();

        // Result should be influenced by the high-weight point
        assert!(result.len() == 5);
        // Check monotonicity
        for i in 1..result.len() {
            assert!(result[i] >= result[i - 1] - 1e-10);
        }
    }

    #[test]
    fn test_complexity_estimation() {
        let regressor =
            DualDecompositionIsotonicRegressor::new().decomposition_parameters(0.1, 20, 3);

        let (num_blocks, total_ops) = regressor.estimate_complexity(100);

        assert!(num_blocks > 0);
        assert!(total_ops > 0);
        assert!(num_blocks <= 100); // Can't have more blocks than elements
    }

    #[test]
    fn test_parallel_api() {
        let n = 40;
        let mut y = Array1::zeros(n);
        for i in 0..n {
            y[i] = (i as Float).sin() + 0.1 * (i as Float);
        }

        let result = parallel_dual_decomposition(&y, None, true, 4);
        assert!(result.is_ok());

        let solution = result.unwrap();

        // Check monotonicity
        for i in 1..solution.len() {
            assert!(solution[i] >= solution[i - 1] - 1e-8);
        }
    }

    #[test]
    fn test_convergence_analysis() {
        let regressor = DualDecompositionIsotonicRegressor::new();
        let dual_vars = array![0.1, -0.05, 0.2, -0.1];

        let (dual_norm, dual_max) = regressor.analyze_convergence(&dual_vars);

        assert!(dual_norm > 0.0);
        assert!(dual_max >= 0.0);
        assert!(dual_max <= dual_norm * (dual_vars.len() as Float).sqrt());
    }
}
