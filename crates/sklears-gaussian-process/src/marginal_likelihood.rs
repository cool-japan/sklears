//! Marginal likelihood optimization for Gaussian Process hyperparameters
//!
//! This module provides functions for optimizing the marginal likelihood
//! of Gaussian processes with respect to hyperparameters. It includes
//! gradient-based optimization methods and utilities for computing
//! gradients of the log marginal likelihood.

use crate::kernels::Kernel;
use crate::utils::{robust_cholesky, triangular_solve};
// SciRS2 Policy - Use scirs2-autograd for ndarray types and operations
use scirs2_core::ndarray::{s, Array1, Array2, ArrayView1, ArrayView2};
use sklears_core::error::{Result as SklResult, SklearsError};
use std::f64::consts::PI;

///
/// let X = array![[1.0], [2.0], [3.0], [4.0]];
/// let y = array![1.0, 4.0, 9.0, 16.0];
///
/// let mut kernel: Box<dyn sklears_gaussian_process::kernels::Kernel> = Box::new(RBF::new(1.0));
/// let optimizer = MarginalLikelihoodOptimizer::new()
///     .max_iter(100)
///     .tol(1e-6);
///
/// let result = optimizer.optimize(
///     &X.view(), &y.view(), &mut kernel, 0.1
/// ).unwrap();
///
/// let optimized_params = result.optimal_params;
/// let final_lml = result.optimal_log_marginal_likelihood;
/// ```
#[derive(Debug, Clone)]
pub struct MarginalLikelihoodOptimizer {
    max_iter: usize,
    tol: f64,
    learning_rate: f64,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
    line_search: bool,
    verbose: bool,
}

/// Result of marginal likelihood optimization
#[derive(Debug, Clone)]
pub struct OptimizationResult {
    /// optimal_params
    pub optimal_params: Array1<f64>,
    /// optimal_log_marginal_likelihood
    pub optimal_log_marginal_likelihood: f64,
    /// n_iterations
    pub n_iterations: usize,
    /// converged
    pub converged: bool,
    /// lml_history
    pub lml_history: Vec<f64>,
    /// gradient_norm_history
    pub gradient_norm_history: Vec<f64>,
}

impl MarginalLikelihoodOptimizer {
    /// Create a new marginal likelihood optimizer
    pub fn new() -> Self {
        Self {
            max_iter: 100,
            tol: 1e-6,
            learning_rate: 0.01,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            line_search: false,
            verbose: false,
        }
    }

    /// Set the maximum number of iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the convergence tolerance
    pub fn tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Set the learning rate
    pub fn learning_rate(mut self, learning_rate: f64) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    /// Set Adam optimizer parameters
    pub fn adam_params(mut self, beta1: f64, beta2: f64, epsilon: f64) -> Self {
        self.beta1 = beta1;
        self.beta2 = beta2;
        self.epsilon = epsilon;
        self
    }

    /// Enable or disable line search
    pub fn line_search(mut self, line_search: bool) -> Self {
        self.line_search = line_search;
        self
    }

    /// Set verbosity
    pub fn verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// Optimize the log marginal likelihood
    pub fn optimize(
        &self,
        X: &ArrayView2<f64>,
        y: &ArrayView1<f64>,
        kernel: &mut Box<dyn Kernel>,
        sigma_n: f64,
    ) -> SklResult<OptimizationResult> {
        let n_params = kernel.get_params().len() + 1; // +1 for noise parameter
        let mut params = Array1::<f64>::zeros(n_params);

        // Initialize parameters
        let kernel_params = kernel.get_params();
        for (i, &param) in kernel_params.iter().enumerate() {
            params[i] = param.ln(); // Work in log space
        }
        params[n_params - 1] = sigma_n.ln(); // Log noise parameter

        // Adam optimizer state
        let mut m = Array1::<f64>::zeros(n_params);
        let mut v = Array1::<f64>::zeros(n_params);

        let mut lml_history = Vec::new();
        let mut gradient_norm_history = Vec::new();
        let mut converged = false;

        for iter in 0..self.max_iter {
            // Update kernel parameters
            let exp_params = params.mapv(|x| x.exp());
            let kernel_params = exp_params.slice(s![..n_params - 1]).to_owned();
            let sigma_n_current = exp_params[n_params - 1];

            kernel.set_params(kernel_params.as_slice().unwrap())?;

            // Compute log marginal likelihood and gradients
            let (lml, grad) =
                self.compute_log_marginal_likelihood_and_gradients(X, y, kernel, sigma_n_current)?;

            lml_history.push(lml);
            let grad_norm = grad.dot(&grad).sqrt();
            gradient_norm_history.push(grad_norm);

            if self.verbose && iter % 10 == 0 {
                println!(
                    "Iteration {}: LML = {:.6}, ||grad|| = {:.6}",
                    iter, lml, grad_norm
                );
            }

            // Check convergence
            if grad_norm < self.tol {
                converged = true;
                if self.verbose {
                    println!("Converged at iteration {}", iter);
                }
                break;
            }

            // Adam update
            let t = (iter + 1) as f64;
            m = self.beta1 * &m + (1.0 - self.beta1) * &grad;
            v = self.beta2 * &v + (1.0 - self.beta2) * grad.mapv(|x| x * x);

            let m_hat = &m / (1.0 - self.beta1.powf(t));
            let v_hat = &v / (1.0 - self.beta2.powf(t));

            let update = &m_hat / (v_hat.mapv(|x| x.sqrt()) + self.epsilon);

            if self.line_search {
                // Simple line search
                let mut step_size = self.learning_rate;
                for _ in 0..5 {
                    let params_new = &params + step_size * &update;
                    let exp_params_new = params_new.mapv(|x| x.exp());
                    let kernel_params_new = exp_params_new.slice(s![..n_params - 1]).to_owned();
                    let sigma_n_new = exp_params_new[n_params - 1];

                    kernel.set_params(kernel_params_new.as_slice().unwrap())?;
                    if let Ok((lml_new, _)) = self.compute_log_marginal_likelihood_and_gradients(
                        X,
                        y,
                        kernel,
                        sigma_n_new,
                    ) {
                        if lml_new > lml {
                            params = params_new;
                            break;
                        }
                    }
                    step_size *= 0.5;
                }
            } else {
                params = &params + self.learning_rate * &update;
            }
        }

        // Set final parameters
        let exp_params = params.mapv(|x| x.exp());
        let kernel_params = exp_params.slice(s![..n_params - 1]).to_owned();
        kernel.set_params(kernel_params.as_slice().unwrap())?;

        let final_lml = lml_history.last().copied().unwrap_or(f64::NEG_INFINITY);

        Ok(OptimizationResult {
            optimal_params: exp_params,
            optimal_log_marginal_likelihood: final_lml,
            n_iterations: lml_history.len(),
            converged,
            lml_history,
            gradient_norm_history,
        })
    }

    /// Compute log marginal likelihood and its gradients
    #[allow(non_snake_case)]
    fn compute_log_marginal_likelihood_and_gradients(
        &self,
        X: &ArrayView2<f64>,
        y: &ArrayView1<f64>,
        kernel: &Box<dyn Kernel>,
        sigma_n: f64,
    ) -> SklResult<(f64, Array1<f64>)> {
        let n = X.nrows();

        // Compute kernel matrix
        let X_owned = X.to_owned();
        let mut K = kernel.compute_kernel_matrix(&X_owned, None)?;

        // Add noise to diagonal
        for i in 0..n {
            K[[i, i]] += sigma_n * sigma_n;
        }

        // Cholesky decomposition
        let L = robust_cholesky(&K)?;

        // Solve for alpha = K^{-1} * y
        let alpha = triangular_solve(&L, &y.to_owned())?;

        // Compute log marginal likelihood
        let log_det_K = 2.0 * L.diag().mapv(|x| x.ln()).sum();
        let quadratic_term = alpha.dot(y);
        let lml = -0.5 * quadratic_term - 0.5 * log_det_K - 0.5 * n as f64 * (2.0 * PI).ln();

        // Compute gradients
        let kernel_params = kernel.get_params();
        let n_params = kernel_params.len() + 1; // +1 for noise
        let mut gradients = Array1::<f64>::zeros(n_params);

        // Gradient w.r.t. kernel parameters
        for (i, _) in kernel_params.iter().enumerate() {
            let grad_K = self.compute_kernel_gradient(X, kernel, i)?;
            let grad_lml = self.compute_lml_gradient(&grad_K, &alpha, &L)?;
            gradients[i] = grad_lml * kernel_params[i]; // Chain rule for log parameters
        }

        // Gradient w.r.t. noise parameter
        let mut grad_K_noise = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            grad_K_noise[[i, i]] = 2.0 * sigma_n;
        }
        let grad_lml_noise = self.compute_lml_gradient(&grad_K_noise, &alpha, &L)?;
        gradients[n_params - 1] = grad_lml_noise * sigma_n; // Chain rule for log parameter

        Ok((lml, gradients))
    }

    /// Compute gradient of kernel matrix w.r.t. a parameter
    #[allow(non_snake_case)]
    fn compute_kernel_gradient(
        &self,
        X: &ArrayView2<f64>,
        kernel: &Box<dyn Kernel>,
        param_idx: usize,
    ) -> SklResult<Array2<f64>> {
        // Finite difference approximation
        let params = kernel.get_params();
        let h = 1e-8;

        // Forward difference
        let mut params_plus = params.clone();
        params_plus[param_idx] += h;

        let mut kernel_plus = kernel.clone_box();
        kernel_plus.set_params(&params_plus)?;

        let X_owned = X.to_owned();
        let K_plus = kernel_plus.compute_kernel_matrix(&X_owned, None)?;
        let K = kernel.compute_kernel_matrix(&X_owned, None)?;

        let grad_K = (K_plus - K) / h;
        Ok(grad_K)
    }

    /// Compute gradient of log marginal likelihood w.r.t. kernel matrix
    fn compute_lml_gradient(
        &self,
        grad_K: &Array2<f64>,
        alpha: &Array1<f64>,
        L: &Array2<f64>,
    ) -> SklResult<f64> {
        // Compute K^{-1} using the Cholesky decomposition
        let n = L.nrows();
        let mut K_inv = Array2::<f64>::zeros((n, n));

        // Solve L * L^T * K_inv = I
        for i in 0..n {
            let mut e = Array1::<f64>::zeros(n);
            e[i] = 1.0;
            let col = triangular_solve(L, &e)?;
            K_inv.column_mut(i).assign(&col);
        }

        // Complete the inverse: K_inv = L^{-T} * L^{-1}
        for i in 0..n {
            let col = K_inv.column(i).to_owned();
            let inv_col = triangular_solve(&L.t().to_owned(), &col)?;
            K_inv.column_mut(i).assign(&inv_col);
        }

        // Gradient: 0.5 * trace((α α^T - K^{-1}) * ∂K/∂θ)
        let alpha_outer = Array2::from_shape_fn((n, n), |(i, j)| alpha[i] * alpha[j]);
        let diff = alpha_outer - K_inv;
        let grad = 0.5 * (&diff * grad_K).sum();

        Ok(grad)
    }
}

impl Default for MarginalLikelihoodOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Optimize hyperparameters using maximum likelihood estimation
pub fn optimize_hyperparameters(
    X: &ArrayView2<f64>,
    y: &ArrayView1<f64>,
    kernel: &mut Box<dyn Kernel>,
    sigma_n: f64,
    max_iter: usize,
    tol: f64,
    verbose: bool,
) -> SklResult<OptimizationResult> {
    let optimizer = MarginalLikelihoodOptimizer::new()
        .max_iter(max_iter)
        .tol(tol)
        .verbose(verbose);

    optimizer.optimize(X, y, kernel, sigma_n)
}

/// Compute log marginal likelihood for given hyperparameters
#[allow(non_snake_case)]
pub fn log_marginal_likelihood(
    X: &ArrayView2<f64>,
    y: &ArrayView1<f64>,
    kernel: &Box<dyn Kernel>,
    sigma_n: f64,
) -> SklResult<f64> {
    let n = X.nrows();

    // Compute kernel matrix
    let X_owned = X.to_owned();
    let mut K = kernel.compute_kernel_matrix(&X_owned, None)?;

    // Add noise to diagonal
    for i in 0..n {
        K[[i, i]] += sigma_n * sigma_n;
    }

    // Cholesky decomposition
    let L = robust_cholesky(&K)?;

    // Solve for alpha = K^{-1} * y
    let alpha = triangular_solve(&L, &y.to_owned())?;

    // Compute log marginal likelihood
    let log_det_K = 2.0 * L.diag().mapv(|x| x.ln()).sum();
    let quadratic_term = alpha.dot(y);
    let lml = -0.5 * quadratic_term - 0.5 * log_det_K - 0.5 * n as f64 * (2.0 * PI).ln();

    Ok(lml)
}

/// Numerically stable log marginal likelihood computation
///
/// This function provides enhanced numerical stability for computing the log marginal
/// likelihood of a Gaussian process. It uses log-space computations and more robust
/// algorithms to handle extreme cases and prevent numerical underflow/overflow.
///
/// # Arguments
///
/// * `X` - Input features (n_samples, n_features)
/// * `y` - Target values (n_samples,)
/// * `kernel` - Kernel function
/// * `sigma_n` - Noise standard deviation
///
/// # Returns
///
/// The log marginal likelihood value with enhanced numerical stability
///
/// # Examples
///
/// ```
/// use sklears_gaussian_process::{log_marginal_likelihood_stable, kernels::RBF};
/// // SciRS2 Policy - Use scirs2-autograd for ndarray types and operations
/// use scirs2_core::ndarray::array;
///
/// let X = array![[1.0], [2.0], [3.0], [4.0]];
/// let y = array![1.0, 4.0, 9.0, 16.0];
/// let kernel: Box<dyn sklears_gaussian_process::kernels::Kernel> = Box::new(RBF::new(1.0));
///
/// let lml = log_marginal_likelihood_stable(&X.view(), &y.view(), &kernel, 0.1).unwrap();
/// ```
#[allow(non_snake_case)]
pub fn log_marginal_likelihood_stable(
    X: &ArrayView2<f64>,
    y: &ArrayView1<f64>,
    kernel: &Box<dyn Kernel>,
    sigma_n: f64,
) -> SklResult<f64> {
    let n = X.nrows();

    // Validate inputs
    if n == 0 {
        return Err(SklearsError::InvalidInput("Empty input array".to_string()));
    }
    if sigma_n <= 0.0 {
        return Err(SklearsError::InvalidInput(
            "Noise level must be positive".to_string(),
        ));
    }

    // Compute kernel matrix
    let X_owned = X.to_owned();
    let mut K = kernel.compute_kernel_matrix(&X_owned, None)?;

    // Add noise to diagonal with numerical stability check
    let sigma_n_sq = sigma_n * sigma_n;
    for i in 0..n {
        K[[i, i]] += sigma_n_sq;
        // Ensure diagonal is positive
        if K[[i, i]] <= 0.0 {
            return Err(SklearsError::NumericalError(
                "Kernel matrix is not positive definite".to_string(),
            ));
        }
    }

    // Robust Cholesky decomposition with automatic jitter
    let L = robust_cholesky(&K)?;

    // Solve for alpha = K^{-1} * y using more stable approach
    let alpha = triangular_solve(&L, &y.to_owned())?;

    // Compute log determinant in log space for numerical stability
    let log_det_K = {
        let log_diag_L: Array1<f64> = L.diag().mapv(|x| {
            if x <= 0.0 {
                return f64::NEG_INFINITY;
            }
            x.ln()
        });

        // Check for numerical issues
        if log_diag_L.iter().any(|&x| !x.is_finite()) {
            return Err(SklearsError::NumericalError(
                "Numerical instability in Cholesky decomposition".to_string(),
            ));
        }

        2.0 * log_diag_L.sum()
    };

    // Compute quadratic term with overflow protection
    let quadratic_term = alpha.dot(y);
    if !quadratic_term.is_finite() || quadratic_term < 0.0 {
        return Err(SklearsError::NumericalError(
            "Numerical instability in quadratic term".to_string(),
        ));
    }

    // Compute log marginal likelihood with numerical stability checks
    let log_2pi = (2.0 * PI).ln();
    let lml = -0.5 * (quadratic_term + log_det_K + n as f64 * log_2pi);

    // Final numerical validation
    if !lml.is_finite() {
        return Err(SklearsError::NumericalError(
            "Log marginal likelihood is not finite".to_string(),
        ));
    }

    Ok(lml)
}

/// Cross-validation for hyperparameter selection
pub fn cross_validate_hyperparameters(
    X: &ArrayView2<f64>,
    y: &ArrayView1<f64>,
    kernel: &mut Box<dyn Kernel>,
    sigma_n_values: &[f64],
    n_folds: usize,
    random_state: Option<u64>,
) -> SklResult<(f64, f64)> {
    let n_samples = X.nrows();
    let fold_size = n_samples / n_folds;

    let mut best_sigma_n = sigma_n_values[0];
    let mut best_score = f64::NEG_INFINITY;

    // Simple random shuffle for CV folds
    let mut indices: Vec<usize> = (0..n_samples).collect();
    if let Some(seed) = random_state {
        // Simple shuffle using LCG
        let mut rng = seed;
        for i in (1..indices.len()).rev() {
            rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
            let j = (rng as usize) % (i + 1);
            indices.swap(i, j);
        }
    }

    for &sigma_n in sigma_n_values {
        let mut fold_scores = Vec::new();

        for fold in 0..n_folds {
            let start_idx = fold * fold_size;
            let end_idx = if fold == n_folds - 1 {
                n_samples
            } else {
                (fold + 1) * fold_size
            };

            // Create train/validation split
            let mut train_indices = Vec::new();
            let mut val_indices = Vec::new();

            for (i, &idx) in indices.iter().enumerate() {
                if i >= start_idx && i < end_idx {
                    val_indices.push(idx);
                } else {
                    train_indices.push(idx);
                }
            }

            // This is a simplified CV - in practice would need proper data splitting
            // For now, just use the full dataset for validation
            let lml = log_marginal_likelihood(X, y, kernel, sigma_n)?;
            fold_scores.push(lml);
        }

        let avg_score = fold_scores.iter().sum::<f64>() / fold_scores.len() as f64;
        if avg_score > best_score {
            best_score = avg_score;
            best_sigma_n = sigma_n;
        }
    }

    Ok((best_sigma_n, best_score))
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::RBF;
    use approx::assert_abs_diff_eq;
    // SciRS2 Policy - Use scirs2_core::ndarray for array operations (RC.1+)
    // Note: array! macro is available in scirs2_core::ndarray as of v0.1.0-RC.1
    use scirs2_core::ndarray::array;

    #[test]
    #[allow(non_snake_case)]
    fn test_log_marginal_likelihood_stable_basic() {
        let X = array![[1.0], [2.0], [3.0], [4.0]];
        let y = array![1.0, 4.0, 9.0, 16.0];
        let kernel: Box<dyn Kernel> = Box::new(RBF::new(1.0));

        let lml = log_marginal_likelihood_stable(&X.view(), &y.view(), &kernel, 0.1).unwrap();
        assert!(lml.is_finite());
        assert!(lml < 0.0); // Log marginal likelihood is typically negative
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_log_marginal_likelihood_stable_vs_standard() {
        let X = array![[1.0], [2.0], [3.0], [4.0]];
        let y = array![1.0, 4.0, 9.0, 16.0];
        let kernel: Box<dyn Kernel> = Box::new(RBF::new(1.0));

        let lml_stable =
            log_marginal_likelihood_stable(&X.view(), &y.view(), &kernel, 0.1).unwrap();
        let lml_standard = log_marginal_likelihood(&X.view(), &y.view(), &kernel, 0.1).unwrap();

        // Both methods should give similar results for well-conditioned problems
        assert_abs_diff_eq!(lml_stable, lml_standard, epsilon = 1e-10);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_log_marginal_likelihood_stable_input_validation() {
        let X = array![[1.0], [2.0], [3.0], [4.0]];
        let y = array![1.0, 4.0, 9.0, 16.0];
        let kernel: Box<dyn Kernel> = Box::new(RBF::new(1.0));

        // Test with negative noise
        let result = log_marginal_likelihood_stable(&X.view(), &y.view(), &kernel, -0.1);
        assert!(result.is_err());

        // Test with zero noise
        let result = log_marginal_likelihood_stable(&X.view(), &y.view(), &kernel, 0.0);
        assert!(result.is_err());

        // Test with empty arrays
        let X_empty = Array2::<f64>::zeros((0, 1));
        let y_empty = Array1::<f64>::zeros(0);
        let result = log_marginal_likelihood_stable(&X_empty.view(), &y_empty.view(), &kernel, 0.1);
        assert!(result.is_err());
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_log_marginal_likelihood_stable_numerical_robustness() {
        // Test with challenging numerical conditions
        let X = array![[1e-10], [2e-10], [3e-10]];
        let y = array![1e-10, 2e-10, 3e-10];
        let kernel: Box<dyn Kernel> = Box::new(RBF::new(1e-20)); // Very small length scale

        let result = log_marginal_likelihood_stable(&X.view(), &y.view(), &kernel, 1e-12);
        // Should either succeed or fail gracefully with a clear error
        match result {
            Ok(lml) => {
                assert!(lml.is_finite());
            }
            Err(e) => {
                // Should be a numerical error, not a panic
                assert!(matches!(e, SklearsError::NumericalError(_)));
            }
        }
    }

    #[test]
    fn test_marginal_likelihood_optimizer_creation() {
        let optimizer = MarginalLikelihoodOptimizer::new();
        assert_eq!(optimizer.max_iter, 100);
        assert_eq!(optimizer.tol, 1e-6);
        assert_eq!(optimizer.learning_rate, 0.01);
    }

    #[test]
    fn test_marginal_likelihood_optimizer_builder() {
        let optimizer = MarginalLikelihoodOptimizer::new()
            .max_iter(200)
            .tol(1e-8)
            .learning_rate(0.001)
            .line_search(true)
            .verbose(true);

        assert_eq!(optimizer.max_iter, 200);
        assert_eq!(optimizer.tol, 1e-8);
        assert_eq!(optimizer.learning_rate, 0.001);
        assert!(optimizer.line_search);
        assert!(optimizer.verbose);
    }
}
