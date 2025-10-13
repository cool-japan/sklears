//! Dual Coordinate Ascent algorithm for large-scale SVM training
//!
//! This module implements the dual coordinate ascent algorithm which is particularly
//! effective for large-scale SVM problems. It directly optimizes the dual formulation
//! of the SVM problem by updating one coordinate at a time.

use crate::kernels::Kernel;
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::numeric::Float as NumFloat;
use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::SeedableRng;
use scirs2_core::SliceRandomExt;
use sklears_core::{error::Result, types::Float};

/// Configuration for dual coordinate ascent algorithm
#[derive(Debug, Clone)]
pub struct DualCoordinateAscentConfig {
    /// Regularization parameter
    pub c: Float,
    /// Tolerance for stopping criterion
    pub tol: Float,
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Random seed for coordinate selection
    pub random_state: Option<u64>,
    /// Use random vs cyclic coordinate selection
    pub random_selection: bool,
    /// Line search parameter for step size optimization
    pub line_search: bool,
    /// Shrinking factor for inactive coordinates
    pub shrinking_factor: Float,
}

impl Default for DualCoordinateAscentConfig {
    fn default() -> Self {
        Self {
            c: 1.0,
            tol: 1e-3,
            max_iter: 1000,
            random_state: None,
            random_selection: true,
            line_search: true,
            shrinking_factor: 0.1,
        }
    }
}

/// Dual coordinate ascent solver for SVM
pub struct DualCoordinateAscent {
    config: DualCoordinateAscentConfig,
    rng: StdRng,
}

impl DualCoordinateAscent {
    /// Create a new dual coordinate ascent solver
    pub fn new(config: DualCoordinateAscentConfig) -> Self {
        let rng = StdRng::seed_from_u64(42);

        Self { config, rng }
    }

    /// Solve SVM dual problem using coordinate ascent
    pub fn solve<K: Kernel>(
        &mut self,
        kernel: &K,
        x: &Array2<Float>,
        y: &Array1<Float>,
    ) -> Result<DualCoordinateAscentResult> {
        let n_samples = x.nrows();
        let mut alpha = Array1::zeros(n_samples);
        let mut gradient = -y.clone(); // Initialize gradient = -e (all ones with labels)

        // Precompute diagonal of kernel matrix for efficiency
        let mut kernel_diagonal = Array1::zeros(n_samples);
        for i in 0..n_samples {
            kernel_diagonal[i] = kernel.compute(x.row(i), x.row(i));
        }

        let mut active_set: Vec<usize> = (0..n_samples).collect();
        let mut iteration = 0;
        let mut convergence_history = Vec::new();

        while iteration < self.config.max_iter {
            let mut max_violation = 0.0;
            let mut updates_made = 0;

            // Randomly shuffle active set if using random selection
            if self.config.random_selection {
                active_set.shuffle(&mut self.rng);
            }

            for &i in &active_set {
                let old_alpha_i = alpha[i];
                let gradient_i = gradient[i];

                // Compute optimal step size
                let quad_coeff = kernel_diagonal[i];
                if quad_coeff <= 0.0 {
                    continue; // Skip degenerate cases
                }

                // Compute new alpha value
                let mut new_alpha_i: Float = old_alpha_i - gradient_i / quad_coeff;

                // Project onto feasible region [0, C]
                new_alpha_i = new_alpha_i.max(0.0).min(self.config.c);

                let delta_alpha: Float = new_alpha_i - old_alpha_i;

                // Check if update is significant
                if delta_alpha.abs() < 1e-12 {
                    continue;
                }

                // Update alpha
                alpha[i] = new_alpha_i;
                updates_made += 1;

                // Update gradient for all samples
                for j in 0..n_samples {
                    let kernel_ij = kernel.compute(x.row(i), x.row(j));
                    gradient[j] += y[i] * y[j] * delta_alpha * kernel_ij;
                }

                // Track maximum violation for convergence checking
                let violation = self.compute_violation(new_alpha_i, gradient_i, self.config.c);
                max_violation = max_violation.max(violation);
            }

            // Convergence check
            if max_violation < self.config.tol {
                convergence_history.push(max_violation);
                break;
            }

            // Shrinking: remove inactive coordinates
            if iteration % 10 == 0 {
                self.shrink_active_set(&mut active_set, &alpha, &gradient);
            }

            convergence_history.push(max_violation);
            iteration += 1;
        }

        // Compute bias term
        let bias = self.compute_bias(&alpha, &gradient, y, self.config.c)?;
        let n_support_vectors = alpha.iter().filter(|&&a| a > 1e-10).count();

        Ok(DualCoordinateAscentResult {
            alpha,
            bias,
            n_iterations: iteration,
            converged: iteration < self.config.max_iter,
            convergence_history,
            n_support_vectors,
        })
    }

    /// Compute KKT violation for convergence checking
    fn compute_violation(&self, alpha: Float, gradient: Float, c: Float) -> Float {
        if alpha < 1e-10 {
            // Lower bound: violation is max(0, -gradient)
            (-gradient).max(0.0)
        } else if alpha > c - 1e-10 {
            // Upper bound: violation is max(0, gradient)
            gradient.max(0.0)
        } else {
            // Free variable: violation is |gradient|
            gradient.abs()
        }
    }

    /// Shrink active set by removing coordinates with small gradient
    fn shrink_active_set(
        &self,
        active_set: &mut Vec<usize>,
        alpha: &Array1<Float>,
        gradient: &Array1<Float>,
    ) {
        let threshold = self.config.shrinking_factor * self.config.tol;

        active_set.retain(|&i| {
            let violation = self.compute_violation(alpha[i], gradient[i], self.config.c);
            violation > threshold
        });
    }

    /// Compute bias term from KKT conditions
    fn compute_bias(
        &self,
        alpha: &Array1<Float>,
        gradient: &Array1<Float>,
        y: &Array1<Float>,
        c: Float,
    ) -> Result<Float> {
        let mut bias_sum = 0.0;
        let mut bias_count = 0;

        for i in 0..alpha.len() {
            // Free support vectors (0 < alpha < C)
            if alpha[i] > 1e-10 && alpha[i] < c - 1e-10 {
                bias_sum += y[i] - gradient[i];
                bias_count += 1;
            }
        }

        if bias_count > 0 {
            Ok(bias_sum / bias_count as Float)
        } else {
            // Fallback: use bounded support vectors
            let mut lower_bound = Float::NEG_INFINITY;
            let mut upper_bound = Float::INFINITY;

            for i in 0..alpha.len() {
                let value = y[i] - gradient[i];
                if alpha[i] < 1e-10 && y[i] > 0.0 {
                    lower_bound = lower_bound.max(value);
                } else if alpha[i] > c - 1e-10 && y[i] < 0.0 {
                    upper_bound = upper_bound.min(value);
                }
            }

            Ok((lower_bound + upper_bound) / 2.0)
        }
    }

    /// Solve with warm start from previous solution
    pub fn solve_with_warm_start<K: Kernel>(
        &mut self,
        kernel: &K,
        x: &Array2<Float>,
        y: &Array1<Float>,
        initial_alpha: &Array1<Float>,
    ) -> Result<DualCoordinateAscentResult> {
        let n_samples = x.nrows();
        let alpha = initial_alpha.clone();

        // Recompute gradient from current alpha
        let mut gradient = -y.clone();
        for i in 0..n_samples {
            for j in 0..n_samples {
                if alpha[j] > 1e-12 {
                    let kernel_ij = kernel.compute(x.row(i), x.row(j));
                    gradient[i] += y[i] * y[j] * alpha[j] * kernel_ij;
                }
            }
        }

        // Continue with normal solving process
        self.solve_from_state(kernel, x, y, alpha, gradient)
    }

    /// Continue solving from given state
    fn solve_from_state<K: Kernel>(
        &mut self,
        kernel: &K,
        x: &Array2<Float>,
        y: &Array1<Float>,
        mut alpha: Array1<Float>,
        mut gradient: Array1<Float>,
    ) -> Result<DualCoordinateAscentResult> {
        let n_samples = x.nrows();
        let mut active_set: Vec<usize> = (0..n_samples).collect();
        let mut iteration = 0;
        let mut convergence_history = Vec::new();

        // Precompute diagonal
        let mut kernel_diagonal = Array1::zeros(n_samples);
        for i in 0..n_samples {
            kernel_diagonal[i] = kernel.compute(x.row(i), x.row(i));
        }

        while iteration < self.config.max_iter {
            let mut max_violation = 0.0;

            if self.config.random_selection {
                active_set.shuffle(&mut self.rng);
            }

            for &i in &active_set {
                let old_alpha_i = alpha[i];
                let gradient_i = gradient[i];

                let quad_coeff = kernel_diagonal[i];
                if quad_coeff <= 0.0 {
                    continue;
                }

                let mut new_alpha_i = old_alpha_i - gradient_i / quad_coeff;
                new_alpha_i = new_alpha_i.max(0.0).min(self.config.c);

                let delta_alpha: Float = new_alpha_i - old_alpha_i;

                if delta_alpha.abs() < 1e-12 {
                    continue;
                }

                alpha[i] = new_alpha_i;

                // Update gradient
                for j in 0..n_samples {
                    let kernel_ij = kernel.compute(x.row(i), x.row(j));
                    gradient[j] += y[i] * y[j] * delta_alpha * kernel_ij;
                }

                let violation = self.compute_violation(new_alpha_i, gradient_i, self.config.c);
                max_violation = max_violation.max(violation);
            }

            if max_violation < self.config.tol {
                convergence_history.push(max_violation);
                break;
            }

            if iteration % 10 == 0 {
                self.shrink_active_set(&mut active_set, &alpha, &gradient);
            }

            convergence_history.push(max_violation);
            iteration += 1;
        }

        let bias = self.compute_bias(&alpha, &gradient, y, self.config.c)?;
        let n_support_vectors = alpha.iter().filter(|&&a| a > 1e-10).count();

        Ok(DualCoordinateAscentResult {
            alpha,
            bias,
            n_iterations: iteration,
            converged: iteration < self.config.max_iter,
            convergence_history,
            n_support_vectors,
        })
    }
}

/// Result of dual coordinate ascent optimization
#[derive(Debug, Clone)]
pub struct DualCoordinateAscentResult {
    /// Dual variables (Lagrange multipliers)
    pub alpha: Array1<Float>,
    /// Bias term
    pub bias: Float,
    /// Number of iterations performed
    pub n_iterations: usize,
    /// Whether the algorithm converged
    pub converged: bool,
    /// History of convergence violations
    pub convergence_history: Vec<Float>,
    /// Number of support vectors
    pub n_support_vectors: usize,
}

impl DualCoordinateAscentResult {
    /// Get support vector indices
    pub fn support_vector_indices(&self) -> Vec<usize> {
        self.alpha
            .iter()
            .enumerate()
            .filter_map(|(i, &alpha)| if alpha > 1e-10 { Some(i) } else { None })
            .collect()
    }

    /// Get support vector coefficients
    pub fn support_vector_coefficients(&self) -> Array1<Float> {
        let indices = self.support_vector_indices();
        Array1::from_vec(indices.into_iter().map(|i| self.alpha[i]).collect())
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::RbfKernel;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_dual_coordinate_ascent_linear_separable() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 3.0], [2.0, 1.0], [3.0, 2.0]];
        let y = array![1.0, 1.0, 1.0, -1.0, -1.0];

        let kernel = RbfKernel::new(1.0);
        let config = DualCoordinateAscentConfig::default();
        let mut solver = DualCoordinateAscent::new(config);

        let result = solver.solve(&kernel, &x, &y).unwrap();

        assert!(result.converged);
        assert!(result.n_support_vectors > 0);
        assert!(result.alpha.sum() > 0.0);
    }

    #[test]
    fn test_dual_coordinate_ascent_warm_start() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 3.0], [2.0, 1.0]];
        let y = array![1.0, 1.0, -1.0, -1.0];

        let kernel = RbfKernel::new(1.0);
        let config = DualCoordinateAscentConfig::default();
        let mut solver = DualCoordinateAscent::new(config);

        // First solve
        let result1 = solver.solve(&kernel, &x, &y).unwrap();

        // Warm start with previous solution
        let result2 = solver
            .solve_with_warm_start(&kernel, &x, &y, &result1.alpha)
            .unwrap();

        assert!(result2.n_iterations <= result1.n_iterations);
    }

    #[test]
    fn test_violation_computation() {
        let config = DualCoordinateAscentConfig::default();
        let solver = DualCoordinateAscent::new(config);

        // Test different cases
        assert_abs_diff_eq!(
            solver.compute_violation(0.0, -0.5, 1.0),
            0.5,
            epsilon = 1e-10
        );
        assert_abs_diff_eq!(
            solver.compute_violation(1.0, 0.3, 1.0),
            0.3,
            epsilon = 1e-10
        );
        assert_abs_diff_eq!(
            solver.compute_violation(0.5, 0.2, 1.0),
            0.2,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_shrinking() {
        let x = array![[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]];
        let y = array![1.0, 1.0, -1.0, -1.0];

        let kernel = RbfKernel::new(1.0);
        let mut config = DualCoordinateAscentConfig::default();
        config.shrinking_factor = 0.5; // More aggressive shrinking
        let mut solver = DualCoordinateAscent::new(config);

        let result = solver.solve(&kernel, &x, &y).unwrap();
        assert!(result.converged);
    }
}
