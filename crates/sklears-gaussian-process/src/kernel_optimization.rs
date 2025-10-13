//! Kernel parameter optimization
//!
//! This module provides functionality for optimizing kernel hyperparameters
//! using gradient-based methods and other optimization strategies.

use crate::kernels::Kernel;
use crate::marginal_likelihood::log_marginal_likelihood;
// SciRS2 Policy - Use scirs2-autograd for ndarray types and operations
use scirs2_core::ndarray::{ArrayView1, ArrayView2};
use sklears_core::error::{Result as SklResult, SklearsError};

/// Kernel parameter optimizer
#[derive(Debug, Clone)]
pub struct KernelOptimizer {
    /// Maximum number of optimization iterations
    pub max_iter: usize,
    /// Learning rate for gradient descent
    pub learning_rate: f64,
    /// Tolerance for convergence
    pub tol: f64,
    /// Whether to use line search
    pub use_line_search: bool,
    /// Number of random restarts
    pub n_restarts: usize,
    /// Bounds for parameters (optional)
    pub bounds: Option<Vec<(f64, f64)>>,
    /// Random state for reproducible results
    pub random_state: Option<u64>,
}

impl Default for KernelOptimizer {
    fn default() -> Self {
        Self {
            max_iter: 100,
            learning_rate: 0.01,
            tol: 1e-6,
            use_line_search: true,
            n_restarts: 5,
            bounds: None,
            random_state: Some(42),
        }
    }
}

/// Result of kernel parameter optimization
#[derive(Debug, Clone)]
pub struct OptimizationResult {
    /// Optimized kernel parameters
    pub optimized_params: Vec<f64>,
    /// Final objective value (negative log marginal likelihood)
    pub final_objective: f64,
    /// Number of iterations used
    pub n_iterations: usize,
    /// Whether optimization converged
    pub converged: bool,
    /// Optimization history (objective values)
    pub history: Vec<f64>,
    /// Final gradient norm
    pub final_gradient_norm: f64,
}

impl KernelOptimizer {
    /// Create a new kernel optimizer
    pub fn new() -> Self {
        Self::default()
    }

    /// Set maximum number of iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set learning rate
    pub fn learning_rate(mut self, learning_rate: f64) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    /// Set convergence tolerance
    pub fn tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Set whether to use line search
    pub fn use_line_search(mut self, use_line_search: bool) -> Self {
        self.use_line_search = use_line_search;
        self
    }

    /// Set number of random restarts
    pub fn n_restarts(mut self, n_restarts: usize) -> Self {
        self.n_restarts = n_restarts;
        self
    }

    /// Set parameter bounds
    pub fn bounds(mut self, bounds: Vec<(f64, f64)>) -> Self {
        self.bounds = Some(bounds);
        self
    }

    /// Set random state
    pub fn random_state(mut self, random_state: Option<u64>) -> Self {
        self.random_state = random_state;
        self
    }

    /// Optimize kernel parameters
    pub fn optimize_kernel(
        &self,
        kernel: &mut Box<dyn Kernel>,
        X: ArrayView2<f64>,
        y: ArrayView1<f64>,
    ) -> SklResult<OptimizationResult> {
        let mut best_result = None;
        let mut best_objective = f64::INFINITY;

        // Try multiple random restarts
        for restart in 0..self.n_restarts {
            let seed = self.random_state.map(|s| s + restart as u64);

            // Initialize parameters (randomly for restarts > 0)
            let initial_params = if restart == 0 {
                kernel.get_params()
            } else {
                self.random_initialize_params(kernel, seed)?
            };

            // Optimize from this starting point
            let result = self.optimize_from_initial(kernel, &X, &y, initial_params)?;

            // Keep track of best result
            if result.final_objective < best_objective {
                best_objective = result.final_objective;
                best_result = Some(result);
            }
        }

        let final_result = best_result.ok_or_else(|| {
            SklearsError::InvalidOperation("No optimization result found".to_string())
        })?;

        // Set the kernel to the best parameters found
        kernel.set_params(&final_result.optimized_params)?;

        Ok(final_result)
    }

    /// Optimize from initial parameters
    fn optimize_from_initial(
        &self,
        kernel: &mut Box<dyn Kernel>,
        X: &ArrayView2<f64>,
        y: &ArrayView1<f64>,
        initial_params: Vec<f64>,
    ) -> SklResult<OptimizationResult> {
        let mut params = initial_params;
        let mut history = Vec::new();
        let mut converged = false;

        // Use Adam optimizer for better convergence
        let mut m = vec![0.0; params.len()]; // First moment
        let mut v = vec![0.0; params.len()]; // Second moment
        let beta1 = 0.9;
        let beta2 = 0.999;
        let epsilon = 1e-8;

        for iteration in 0..self.max_iter {
            // Set current parameters
            kernel.set_params(&params)?;

            // Compute objective (negative log marginal likelihood) and gradient
            let (objective, gradient) = self.compute_objective_and_gradient(kernel, X, y)?;
            history.push(objective);

            // Check convergence
            let gradient_norm = gradient.iter().map(|x| x * x).sum::<f64>().sqrt();
            if gradient_norm < self.tol {
                converged = true;
                break;
            }

            // Apply bounds if specified
            let clipped_gradient = self.apply_bounds_to_gradient(&params, &gradient);

            // Adam optimizer update
            for i in 0..params.len() {
                m[i] = beta1 * m[i] + (1.0 - beta1) * clipped_gradient[i];
                v[i] = beta2 * v[i] + (1.0 - beta2) * clipped_gradient[i] * clipped_gradient[i];

                let m_hat = m[i] / (1.0 - beta1.powi(iteration as i32 + 1));
                let v_hat = v[i] / (1.0 - beta2.powi(iteration as i32 + 1));

                let update = self.learning_rate * m_hat / (v_hat.sqrt() + epsilon);
                params[i] -= update;
            }

            // Apply bounds to parameters
            self.apply_bounds_to_params(&mut params);

            // Line search if enabled
            if self.use_line_search && iteration % 10 == 0 {
                params = self.line_search(kernel, X, y, &params, &clipped_gradient)?;
            }
        }

        let final_gradient = self.compute_gradient(kernel, X, y)?;
        let final_gradient_norm = final_gradient.iter().map(|x| x * x).sum::<f64>().sqrt();

        Ok(OptimizationResult {
            optimized_params: params,
            final_objective: history.last().copied().unwrap_or(f64::INFINITY),
            n_iterations: history.len(),
            converged,
            history,
            final_gradient_norm,
        })
    }

    /// Compute objective and gradient
    #[allow(non_snake_case)]
    fn compute_objective_and_gradient(
        &self,
        kernel: &Box<dyn Kernel>,
        X: &ArrayView2<f64>,
        y: &ArrayView1<f64>,
    ) -> SklResult<(f64, Vec<f64>)> {
        // Compute negative log marginal likelihood
        let X_owned = X.to_owned();
        let y_owned = y.to_owned();
        let neg_log_ml = -log_marginal_likelihood(&X_owned.view(), &y_owned.view(), kernel, 1e-6)?;

        // Compute gradient using finite differences
        let gradient = self.compute_gradient(kernel, X, y)?;

        Ok((neg_log_ml, gradient))
    }

    /// Compute gradient using finite differences
    #[allow(non_snake_case)]
    fn compute_gradient(
        &self,
        kernel: &Box<dyn Kernel>,
        X: &ArrayView2<f64>,
        y: &ArrayView1<f64>,
    ) -> SklResult<Vec<f64>> {
        let params = kernel.get_params();
        let mut gradient = vec![0.0; params.len()];
        let h = 1e-6; // Step size for finite differences

        let X_owned = X.to_owned();
        let y_owned = y.to_owned();

        // Central difference for each parameter
        for i in 0..params.len() {
            let mut params_plus = params.clone();
            let mut params_minus = params.clone();

            params_plus[i] += h;
            params_minus[i] -= h;

            // Create temporary kernels for evaluation
            let mut kernel_plus = kernel.clone_box();
            let mut kernel_minus = kernel.clone_box();

            kernel_plus.set_params(&params_plus)?;
            kernel_minus.set_params(&params_minus)?;

            let f_plus =
                -log_marginal_likelihood(&X_owned.view(), &y_owned.view(), &kernel_plus, 1e-6)?;
            let f_minus =
                -log_marginal_likelihood(&X_owned.view(), &y_owned.view(), &kernel_minus, 1e-6)?;

            gradient[i] = (f_plus - f_minus) / (2.0 * h);
        }

        Ok(gradient)
    }

    /// Apply bounds to gradient (set to zero if at boundary)
    fn apply_bounds_to_gradient(&self, params: &Vec<f64>, gradient: &Vec<f64>) -> Vec<f64> {
        if let Some(bounds) = &self.bounds {
            let mut clipped = gradient.clone();
            for i in 0..params.len() {
                if i < bounds.len() {
                    let (lower, upper) = bounds[i];
                    // If at lower bound and gradient is negative, set to zero
                    if params[i] <= lower && gradient[i] < 0.0 {
                        clipped[i] = 0.0;
                    }
                    // If at upper bound and gradient is positive, set to zero
                    if params[i] >= upper && gradient[i] > 0.0 {
                        clipped[i] = 0.0;
                    }
                }
            }
            clipped
        } else {
            gradient.clone()
        }
    }

    /// Apply bounds to parameters
    fn apply_bounds_to_params(&self, params: &mut Vec<f64>) {
        if let Some(bounds) = &self.bounds {
            for i in 0..params.len() {
                if i < bounds.len() {
                    let (lower, upper) = bounds[i];
                    params[i] = params[i].max(lower).min(upper);
                }
            }
        }
    }

    /// Simple line search (backtracking)
    #[allow(non_snake_case)]
    fn line_search(
        &self,
        kernel: &mut Box<dyn Kernel>,
        X: &ArrayView2<f64>,
        y: &ArrayView1<f64>,
        params: &Vec<f64>,
        direction: &Vec<f64>,
    ) -> SklResult<Vec<f64>> {
        let mut alpha = 1.0;
        let rho = 0.5;
        let c1 = 1e-4;

        // Current objective
        kernel.set_params(params)?;
        let X_owned = X.to_owned();
        let y_owned = y.to_owned();
        let f0 = -log_marginal_likelihood(&X_owned.view(), &y_owned.view(), kernel, 1e-6)?;

        // Directional derivative
        let grad_f = self.compute_gradient(kernel, X, y)?;
        let dir_deriv: f64 = grad_f
            .iter()
            .zip(direction.iter())
            .map(|(g, d)| g * d)
            .sum();

        for _ in 0..20 {
            // Maximum line search iterations
            let new_params: Vec<f64> = params
                .iter()
                .zip(direction.iter())
                .map(|(p, d)| p - alpha * d)
                .collect();

            // Apply bounds
            let mut bounded_params = new_params;
            self.apply_bounds_to_params(&mut bounded_params);

            kernel.set_params(&bounded_params)?;
            let f_new = -log_marginal_likelihood(&X_owned.view(), &y_owned.view(), kernel, 1e-6)?;

            // Armijo condition
            if f_new <= f0 + c1 * alpha * dir_deriv {
                return Ok(bounded_params);
            }

            alpha *= rho;
        }

        // If line search fails, return original parameters
        Ok(params.clone())
    }

    /// Random initialization of parameters for restarts
    fn random_initialize_params(
        &self,
        kernel: &Box<dyn Kernel>,
        seed: Option<u64>,
    ) -> SklResult<Vec<f64>> {
        let current_params = kernel.get_params();
        let mut params = current_params.clone();

        // Simple random perturbation (in practice, could use better strategies)
        let scale = 0.5; // Perturbation scale
        for i in 0..params.len() {
            let noise = if let Some(s) = seed {
                // Simple linear congruential generator for reproducibility
                let a = 1664525u64;
                let c = 1013904223u64;
                let m = 2u64.pow(32);
                let x = (a.wrapping_mul(s + i as u64).wrapping_add(c)) % m;
                (x as f64 / m as f64 - 0.5) * 2.0 // Range [-1, 1]
            } else {
                0.0 // No randomization if no seed
            };

            params[i] *= 1.0 + scale * noise;

            // Ensure parameters stay positive (common requirement for GP kernels)
            params[i] = params[i].abs().max(1e-6);
        }

        // Apply bounds if specified
        self.apply_bounds_to_params(&mut params);

        Ok(params)
    }
}

/// Optimize kernel parameters using gradient descent
pub fn optimize_kernel_parameters(
    kernel: &mut Box<dyn Kernel>,
    X: ArrayView2<f64>,
    y: ArrayView1<f64>,
    optimizer_config: Option<KernelOptimizer>,
) -> SklResult<OptimizationResult> {
    let optimizer = optimizer_config.unwrap_or_default();
    optimizer.optimize_kernel(kernel, X, y)
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::{Kernel, RBF};
    // SciRS2 Policy - Use scirs2-autograd for ndarray types and operations
    use scirs2_core::ndarray::{Array1, Array2};

    #[test]
    fn test_kernel_optimizer_creation() {
        let optimizer = KernelOptimizer::new();
        assert_eq!(optimizer.max_iter, 100);
        assert_eq!(optimizer.n_restarts, 5);
        assert!(optimizer.use_line_search);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_kernel_optimization() {
        let optimizer = KernelOptimizer::new()
            .max_iter(10)
            .n_restarts(1)
            .learning_rate(0.1);

        // Create simple test data
        let X = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let y = Array1::from_vec(vec![1.0, 4.0, 9.0, 16.0, 25.0]);

        let mut kernel: Box<dyn Kernel> = Box::new(RBF::new(1.0));

        let result = optimizer
            .optimize_kernel(&mut kernel, X.view(), y.view())
            .unwrap();

        assert!(result.final_objective.is_finite());
        assert_eq!(result.n_iterations, 10); // Should run full iterations for this test
        assert!(!result.history.is_empty());
    }

    #[test]
    fn test_bounds_application() {
        let bounds = vec![(0.1, 10.0)];
        let optimizer = KernelOptimizer::new().bounds(bounds);

        let mut params = vec![-1.0]; // Below lower bound
        optimizer.apply_bounds_to_params(&mut params);
        assert_eq!(params[0], 0.1);

        params[0] = 15.0; // Above upper bound
        optimizer.apply_bounds_to_params(&mut params);
        assert_eq!(params[0], 10.0);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_gradient_computation() {
        let optimizer = KernelOptimizer::new();
        let kernel: Box<dyn Kernel> = Box::new(RBF::new(1.0));

        let X = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = Array1::from_vec(vec![1.0, 2.0, 3.0]);

        let gradient = optimizer
            .compute_gradient(&kernel, &X.view(), &y.view())
            .unwrap();

        assert_eq!(gradient.len(), 1); // RBF has one parameter
        assert!(gradient[0].is_finite());
    }

    #[test]
    fn test_random_initialization() {
        let optimizer = KernelOptimizer::new();
        let kernel: Box<dyn Kernel> = Box::new(RBF::new(1.0));

        let params1 = optimizer
            .random_initialize_params(&kernel, Some(42))
            .unwrap();
        let params2 = optimizer
            .random_initialize_params(&kernel, Some(42))
            .unwrap();
        let params3 = optimizer
            .random_initialize_params(&kernel, Some(43))
            .unwrap();

        // Same seed should give same result
        assert_eq!(params1[0], params2[0]);
        // Different seed should give different result
        assert_ne!(params1[0], params3[0]);
        // All parameters should be positive
        assert!(params1[0] > 0.0);
        assert!(params3[0] > 0.0);
    }
}
