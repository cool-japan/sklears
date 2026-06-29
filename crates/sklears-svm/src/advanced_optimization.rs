//! Advanced optimization methods for SVM
//!
//! This module implements modern optimization algorithms for SVM training that offer
//! better convergence properties and scalability compared to traditional SMO.
//!
//! Algorithms included:
//! - ADMM (Alternating Direction Method of Multipliers): Distributed optimization
//! - Newton methods: Second-order optimization for faster convergence
//! - Primal-dual methods: Simultaneous optimization of primal and dual problems
//! - Trust region methods: Robust optimization with adaptive step sizes
//! - Accelerated gradient methods: Fast first-order optimization

use scirs2_core::ndarray::{s, Array1, Array2};
use scirs2_linalg::compat::ArrayLinalgExt;

use crate::kernels::{create_kernel, Kernel, KernelType};
use sklears_core::error::{Result, SklearsError};

/// Solve `a x = b`. Falls back to a Tikhonov-regularized system if the matrix
/// is singular, and finally to the right-hand side itself so the outer
/// iteration can still make progress. Never panics.
///
/// For a Newton system `H d = -g` the caller passes `b = -gradient`, so the
/// last-resort fallback `d = b = -gradient` corresponds to a steepest-descent
/// direction, allowing the surrounding loop to continue making progress even
/// when the Hessian is numerically singular.
fn solve_linear_system(a: &Array2<f64>, b: &Array1<f64>, reg: f64) -> Array1<f64> {
    if let Ok(x) = a.solve(b) {
        if x.iter().all(|v| v.is_finite()) {
            return x;
        }
    }

    // Regularize the diagonal and retry.
    let n = a.nrows();
    let mut a_reg = a.clone();
    for i in 0..n {
        a_reg[[i, i]] += reg.max(1e-8);
    }
    if let Ok(x) = a_reg.solve(b) {
        if x.iter().all(|v| v.is_finite()) {
            return x;
        }
    }

    // Last resort: gradient-descent style direction (b itself). Callers pass
    // b = -gradient for Newton systems, so this yields d = -gradient.
    b.clone()
}

/// Configuration for advanced optimization methods
#[derive(Debug, Clone)]
pub struct AdvancedOptimizationConfig {
    /// Regularization parameter
    pub c: f64,
    /// Kernel type
    pub kernel: KernelType,
    /// Tolerance for convergence
    pub tol: f64,
    /// Maximum number of iterations
    pub max_iter: usize,
    /// ADMM penalty parameter
    pub rho: f64,
    /// Trust region radius
    pub trust_radius: f64,
    /// Line search parameters
    pub line_search_c1: f64,
    pub line_search_c2: f64,
    /// Newton method regularization
    pub newton_reg: f64,
    /// Verbose output
    pub verbose: bool,
}

impl Default for AdvancedOptimizationConfig {
    fn default() -> Self {
        Self {
            c: 1.0,
            kernel: KernelType::Rbf { gamma: 1.0 },
            tol: 1e-6,
            max_iter: 1000,
            rho: 1.0,
            trust_radius: 1.0,
            line_search_c1: 1e-4,
            line_search_c2: 0.9,
            newton_reg: 1e-8,
            verbose: false,
        }
    }
}

/// Result of advanced optimization
#[derive(Debug, Clone)]
pub struct OptimizationResult {
    /// Dual coefficients
    pub dual_coef: Array1<f64>,
    /// Intercept term
    pub intercept: f64,
    /// Support vector indices
    pub support_indices: Vec<usize>,
    /// Number of iterations
    pub n_iterations: usize,
    /// Final objective value
    pub objective_value: f64,
    /// Convergence status
    pub converged: bool,
    /// Optimization history
    pub history: Vec<f64>,
}

/// ADMM (Alternating Direction Method of Multipliers) SVM Solver
///
/// ADMM is a distributed optimization algorithm that decomposes the SVM problem
/// into smaller subproblems that can be solved in parallel. It's particularly
/// effective for large-scale and distributed SVM training.
///
/// The algorithm solves the consensus problem:
/// minimize f(x) + g(z)
/// subject to Ax + Bz = c
///
/// For SVM, this becomes:
/// minimize (1/2)||w||² + C∑ξᵢ
/// subject to yᵢ(wᵀφ(xᵢ) + b) ≥ 1 - ξᵢ, ξᵢ ≥ 0
///
/// Reference: Boyd, S. et al. (2011). Distributed optimization and statistical
/// learning via the alternating direction method of multipliers.
#[derive(Debug, Clone)]
pub struct ADMMSVM {
    config: AdvancedOptimizationConfig,
    kernel: Option<KernelType>,
    is_fitted: bool,
}

impl Default for ADMMSVM {
    /// Create a new ADMM SVM solver with default configuration
    fn default() -> Self {
        Self::new(AdvancedOptimizationConfig::default())
    }
}

impl ADMMSVM {
    /// Create a new ADMM SVM solver
    pub fn new(config: AdvancedOptimizationConfig) -> Self {
        Self {
            config,
            kernel: None,
            is_fitted: false,
        }
    }

    /// Fit the SVM using ADMM optimization
    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<OptimizationResult> {
        // Validate inputs
        if x.nrows() != y.len() {
            return Err(SklearsError::InvalidInput(
                "Number of samples must match number of labels".to_string(),
            ));
        }

        let n_samples = x.nrows();

        // Initialize kernel
        let kernel = self.config.kernel.clone();
        self.kernel = Some(kernel);

        // Compute kernel matrix
        let k_matrix = self.compute_kernel_matrix(x)?;

        // Initialize variables
        let mut alpha = Array1::zeros(n_samples);
        let mut z = Array1::zeros(n_samples);
        let mut u: Array1<f64> = Array1::zeros(n_samples); // Dual variables
        let mut history = Vec::new();

        // ADMM iterations
        for iteration in 0..self.config.max_iter {
            // Store previous z for dual residual calculation
            let z_prev = z.clone();

            // Update alpha (dual variables)
            alpha = self.update_alpha(&k_matrix, &z, &u)?;

            // Update w (primal variables)
            let w = self.update_w(x, y, &alpha)?;

            // Update z (auxiliary variables)
            z = self.update_z(&alpha, &u)?;

            // Update u (Lagrange multipliers)
            u = &u + &((&alpha - &z) * self.config.rho);

            // Calculate objective value
            let objective = self.calculate_objective(&k_matrix, &alpha, &w)?;
            history.push(objective);

            if self.config.verbose && iteration % 10 == 0 {
                println!("ADMM Iteration {}: Objective = {:.6}", iteration, objective);
            }

            // Check convergence
            let primal_diff = &alpha - &z;
            let primal_residual = primal_diff.dot(&primal_diff).sqrt();
            // Dual residual: ρ * ||z^{k+1} - z^k||
            let z_diff = &z - &z_prev;
            let dual_residual = self.config.rho * z_diff.dot(&z_diff).sqrt();

            if primal_residual < self.config.tol && dual_residual < self.config.tol {
                if self.config.verbose {
                    println!("ADMM converged after {} iterations", iteration + 1);
                }

                self.is_fitted = true;

                let support_indices = self.find_support_vectors(&alpha)?;
                let intercept = self.calculate_intercept(x, y, &alpha, &support_indices)?;

                return Ok(OptimizationResult {
                    dual_coef: alpha,
                    intercept,
                    support_indices,
                    n_iterations: iteration + 1,
                    objective_value: objective,
                    converged: true,
                    history,
                });
            }
        }

        self.is_fitted = true;

        // Return result even if not converged
        let support_indices = self.find_support_vectors(&alpha)?;
        let intercept = self.calculate_intercept(x, y, &alpha, &support_indices)?;

        Ok(OptimizationResult {
            dual_coef: alpha,
            intercept,
            support_indices,
            n_iterations: self.config.max_iter,
            objective_value: history.last().copied().unwrap_or(0.0),
            converged: false,
            history,
        })
    }

    /// Update alpha variables in ADMM
    fn update_alpha(
        &self,
        k_matrix: &Array2<f64>,
        z: &Array1<f64>,
        u: &Array1<f64>,
    ) -> Result<Array1<f64>> {
        let n = k_matrix.nrows();

        // Solve the alpha subproblem
        // This is a QP: min (1/2) α^T Q α + p^T α
        // where Q = K + rho I and p = -e + rho * (z - u)
        let mut q_matrix = k_matrix.clone();
        for i in 0..n {
            q_matrix[[i, i]] += self.config.rho;
        }

        let p = &Array1::from_elem(n, -1.0) + &((z - u) * self.config.rho);

        // Solve Q alpha = -p with a robust general solver.
        let neg_p = p.mapv(|v| -v);
        let mut alpha = solve_linear_system(&q_matrix, &neg_p, self.config.rho.max(1e-8));

        // Project onto constraints [0, C]
        for i in 0..n {
            alpha[i] = alpha[i].max(0.0).min(self.config.c);
        }

        Ok(alpha)
    }

    /// Update w variables in ADMM
    fn update_w(
        &self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        alpha: &Array1<f64>,
    ) -> Result<Array1<f64>> {
        let n_features = x.ncols();
        let mut w = Array1::zeros(n_features);

        // w = Σ αᵢ yᵢ xᵢ
        for i in 0..alpha.len() {
            if alpha[i] > 0.0 {
                let coeff = alpha[i] * y[i];
                for k in 0..n_features {
                    w[k] += coeff * x[[i, k]];
                }
            }
        }

        Ok(w)
    }

    /// Update z variables in ADMM
    fn update_z(&self, alpha: &Array1<f64>, u: &Array1<f64>) -> Result<Array1<f64>> {
        let mut z = Array1::zeros(alpha.len());

        // Soft thresholding for z update
        for i in 0..alpha.len() {
            let temp = alpha[i] + u[i];
            z[i] = if temp > self.config.c / self.config.rho {
                temp - self.config.c / self.config.rho
            } else if temp < 0.0 {
                temp
            } else {
                0.0
            };
        }

        Ok(z)
    }

    /// Compute kernel matrix
    fn compute_kernel_matrix(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let kernel_type = self
            .kernel
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "compute_kernel_matrix".to_string(),
            })?;
        let kernel = create_kernel(kernel_type.clone())?;
        let n = x.nrows();
        let mut k_matrix = Array2::zeros((n, n));

        for i in 0..n {
            for j in 0..n {
                k_matrix[[i, j]] = kernel.compute(x.row(i), x.row(j));
            }
        }

        Ok(k_matrix)
    }

    /// Calculate ADMM objective value
    fn calculate_objective(
        &self,
        k_matrix: &Array2<f64>,
        alpha: &Array1<f64>,
        w: &Array1<f64>,
    ) -> Result<f64> {
        let dual_obj = alpha.sum() - 0.5 * alpha.dot(&k_matrix.dot(alpha));
        let primal_obj = 0.5 * w.dot(w);

        Ok(dual_obj.max(primal_obj))
    }

    /// Find support vector indices
    fn find_support_vectors(&self, alpha: &Array1<f64>) -> Result<Vec<usize>> {
        let support_indices: Vec<usize> = alpha
            .iter()
            .enumerate()
            .filter(|(_, &val)| val > self.config.tol)
            .map(|(i, _)| i)
            .collect();

        Ok(support_indices)
    }

    /// Calculate intercept
    fn calculate_intercept(
        &self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        alpha: &Array1<f64>,
        support_indices: &[usize],
    ) -> Result<f64> {
        if support_indices.is_empty() {
            return Ok(0.0);
        }

        let kernel_type = self
            .kernel
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "calculate_intercept".to_string(),
            })?;
        let kernel = create_kernel(kernel_type.clone())?;
        let mut intercept_sum = 0.0;
        let mut count = 0;

        for &i in support_indices {
            if alpha[i] > self.config.tol && alpha[i] < self.config.c - self.config.tol {
                let mut decision_value = 0.0;
                for &j in support_indices {
                    decision_value += alpha[j] * y[j] * kernel.compute(x.row(i), x.row(j));
                }
                intercept_sum += y[i] - decision_value;
                count += 1;
            }
        }

        Ok(if count > 0 {
            intercept_sum / count as f64
        } else {
            0.0
        })
    }

    /// Predict using the fitted model
    pub fn predict(&self, x: &Array2<f64>, result: &OptimizationResult) -> Result<Array1<f64>> {
        if !self.is_fitted {
            return Err(SklearsError::NotFitted {
                operation: "prediction".to_string(),
            });
        }

        let decision_values = self.decision_function(x, result)?;
        Ok(Array1::from_vec(
            decision_values
                .iter()
                .map(|&val| if val > 0.0 { 1.0 } else { -1.0 })
                .collect(),
        ))
    }

    /// Calculate decision function values
    pub fn decision_function(
        &self,
        x: &Array2<f64>,
        result: &OptimizationResult,
    ) -> Result<Array1<f64>> {
        if !self.is_fitted {
            return Err(SklearsError::NotFitted {
                operation: "prediction".to_string(),
            });
        }

        let kernel_type = self
            .kernel
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "decision_function".to_string(),
            })?;
        let kernel = create_kernel(kernel_type.clone())?;
        let mut decision_values = Array1::zeros(x.nrows());

        for i in 0..x.nrows() {
            let mut sum = 0.0;
            for &j in &result.support_indices {
                sum += result.dual_coef[j] * kernel.compute(x.row(i), x.row(j));
            }
            decision_values[i] = sum + result.intercept;
        }

        Ok(decision_values)
    }
}

/// Newton Method SVM Solver
///
/// Newton methods use second-order derivatives (Hessian) to achieve faster convergence
/// compared to first-order methods. This implementation uses a regularized Newton
/// method with line search for robustness.
#[derive(Debug, Clone)]
pub struct NewtonSVM {
    config: AdvancedOptimizationConfig,
    kernel: Option<KernelType>,
    is_fitted: bool,
}

impl Default for NewtonSVM {
    /// Create a new Newton SVM solver with default configuration
    fn default() -> Self {
        Self::new(AdvancedOptimizationConfig::default())
    }
}

impl NewtonSVM {
    /// Create a new Newton SVM solver
    pub fn new(config: AdvancedOptimizationConfig) -> Self {
        Self {
            config,
            kernel: None,
            is_fitted: false,
        }
    }

    /// Fit the SVM using Newton method
    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<OptimizationResult> {
        // Validate inputs
        if x.nrows() != y.len() {
            return Err(SklearsError::InvalidInput(
                "Number of samples must match number of labels".to_string(),
            ));
        }

        // Initialize kernel
        let kernel = self.config.kernel.clone();
        self.kernel = Some(kernel);

        // For linear kernel, we can use primal Newton method
        if matches!(self.config.kernel, KernelType::Linear) {
            self.fit_primal_newton(x, y)
        } else {
            // For non-linear kernels, use dual Newton method
            self.fit_dual_newton(x, y)
        }
    }

    /// Fit using primal Newton method (for linear SVMs)
    fn fit_primal_newton(
        &mut self,
        x: &Array2<f64>,
        y: &Array1<f64>,
    ) -> Result<OptimizationResult> {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        // Initialize variables
        let mut w: Array1<f64> = Array1::zeros(n_features);
        let mut b = 0.0;
        let mut history = Vec::new();

        for iteration in 0..self.config.max_iter {
            // Calculate margins
            let margins = self.calculate_margins(x, y, &w, b);

            // Find active constraints (margin < 1)
            let active_indices: Vec<usize> = margins
                .iter()
                .enumerate()
                .filter(|(_, &margin)| margin < 1.0)
                .map(|(i, _)| i)
                .collect();

            if active_indices.is_empty() {
                break; // All constraints satisfied
            }

            // Build Hessian matrix
            let hessian = self.build_hessian(x, &active_indices)?;

            // Build gradient
            let gradient = self.build_gradient(x, y, &w, b, &active_indices, &margins)?;

            // Solve Newton system: H * d = -g
            let neg_gradient = gradient.mapv(|v| -v);
            let direction = solve_linear_system(&hessian, &neg_gradient, self.config.newton_reg);

            // Line search
            let step_size = self.line_search(x, y, &w, b, &direction, &margins)?;

            // Update variables
            for k in 0..n_features {
                w[k] += step_size * direction[k];
            }
            b += step_size * direction[n_features];

            // Calculate objective
            let objective = self.calculate_primal_objective(&w, &margins);
            history.push(objective);

            if self.config.verbose && iteration % 10 == 0 {
                println!(
                    "Newton Iteration {}: Objective = {:.6}",
                    iteration, objective
                );
            }

            // Check convergence
            if gradient.dot(&gradient).sqrt() < self.config.tol {
                if self.config.verbose {
                    println!("Newton method converged after {} iterations", iteration + 1);
                }

                self.is_fitted = true;

                return Ok(OptimizationResult {
                    dual_coef: Array1::zeros(n_samples), // Not applicable for primal
                    intercept: b,
                    support_indices: active_indices,
                    n_iterations: iteration + 1,
                    objective_value: objective,
                    converged: true,
                    history,
                });
            }
        }

        self.is_fitted = true;

        // Return result even if not converged
        let margins = self.calculate_margins(x, y, &w, b);
        let active_indices: Vec<usize> = margins
            .iter()
            .enumerate()
            .filter(|(_, &margin)| margin < 1.0)
            .map(|(i, _)| i)
            .collect();

        Ok(OptimizationResult {
            dual_coef: Array1::zeros(n_samples),
            intercept: b,
            support_indices: active_indices,
            n_iterations: self.config.max_iter,
            objective_value: history.last().copied().unwrap_or(0.0),
            converged: false,
            history,
        })
    }

    /// Fit using dual Newton method (for non-linear SVMs)
    fn fit_dual_newton(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<OptimizationResult> {
        let n_samples = x.nrows();

        // Initialize dual variables
        let mut alpha: Array1<f64> = Array1::zeros(n_samples);
        let mut history = Vec::new();

        // Compute kernel matrix
        let k_matrix = self.compute_kernel_matrix(x)?;

        for iteration in 0..self.config.max_iter {
            // Calculate gradient of dual objective
            let gradient = self.calculate_dual_gradient(&k_matrix, &alpha);

            // Calculate Hessian of dual objective
            let hessian = self.calculate_dual_hessian(&k_matrix, &alpha)?;

            // Solve Newton system
            let neg_gradient = gradient.mapv(|v| -v);
            let direction = solve_linear_system(&hessian, &neg_gradient, self.config.newton_reg);

            // Line search for step size
            let step_size = self.dual_line_search(&k_matrix, &alpha, &direction)?;

            // Update alpha
            alpha = &alpha + &(&direction * step_size);

            // Project onto constraints [0, C]
            for i in 0..n_samples {
                alpha[i] = alpha[i].max(0.0).min(self.config.c);
            }

            // Calculate objective
            let objective = self.calculate_dual_objective(&k_matrix, &alpha);
            history.push(objective);

            if self.config.verbose && iteration % 10 == 0 {
                println!(
                    "Dual Newton Iteration {}: Objective = {:.6}",
                    iteration, objective
                );
            }

            // Check convergence
            if gradient.dot(&gradient).sqrt() < self.config.tol {
                if self.config.verbose {
                    println!(
                        "Dual Newton method converged after {} iterations",
                        iteration + 1
                    );
                }

                self.is_fitted = true;

                let support_indices = self.find_support_vectors(&alpha)?;
                let intercept = self.calculate_intercept(x, y, &alpha, &support_indices)?;

                return Ok(OptimizationResult {
                    dual_coef: alpha,
                    intercept,
                    support_indices,
                    n_iterations: iteration + 1,
                    objective_value: objective,
                    converged: true,
                    history,
                });
            }
        }

        self.is_fitted = true;

        // Return result even if not converged
        let support_indices = self.find_support_vectors(&alpha)?;
        let intercept = self.calculate_intercept(x, y, &alpha, &support_indices)?;

        Ok(OptimizationResult {
            dual_coef: alpha,
            intercept,
            support_indices,
            n_iterations: self.config.max_iter,
            objective_value: history.last().copied().unwrap_or(0.0),
            converged: false,
            history,
        })
    }

    /// Calculate margins for primal Newton method
    fn calculate_margins(
        &self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        w: &Array1<f64>,
        b: f64,
    ) -> Vec<f64> {
        let mut margins = Vec::with_capacity(x.nrows());
        for i in 0..x.nrows() {
            let decision_value = x.row(i).dot(w) + b;
            margins.push(y[i] * decision_value);
        }
        margins
    }

    /// Build Hessian matrix for primal Newton method
    fn build_hessian(&self, x: &Array2<f64>, active_indices: &[usize]) -> Result<Array2<f64>> {
        let n_features = x.ncols();
        let mut hessian = Array2::zeros((n_features + 1, n_features + 1));

        // Add identity for regularization
        for i in 0..n_features {
            hessian[[i, i]] = 1.0;
        }

        // Add contributions from active constraints
        for &idx in active_indices {
            let x_i = x.row(idx);

            // H_ww += x_i * x_i^T
            for i in 0..n_features {
                for j in 0..n_features {
                    hessian[[i, j]] += x_i[i] * x_i[j];
                }
            }

            // H_wb = H_bw += x_i
            for i in 0..n_features {
                hessian[[i, n_features]] += x_i[i];
                hessian[[n_features, i]] += x_i[i];
            }

            // H_bb += 1
            hessian[[n_features, n_features]] += 1.0;
        }

        hessian.mapv_inplace(|v| v * self.config.c);

        Ok(hessian)
    }

    /// Build gradient for primal Newton method
    fn build_gradient(
        &self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        w: &Array1<f64>,
        _b: f64,
        active_indices: &[usize],
        margins: &[f64],
    ) -> Result<Array1<f64>> {
        let n_features = x.ncols();
        let mut gradient = Array1::zeros(n_features + 1);

        // Regularization term
        gradient.slice_mut(s![0..n_features]).assign(w);

        // Add contributions from active constraints
        for &idx in active_indices {
            let violation = 1.0 - margins[idx];
            if violation > 0.0 {
                let x_i = x.row(idx);

                // dL/dw += -C * y_i * x_i
                for i in 0..n_features {
                    gradient[i] -= self.config.c * y[idx] * x_i[i];
                }

                // dL/db += -C * y_i
                gradient[n_features] -= self.config.c * y[idx];
            }
        }

        Ok(gradient)
    }

    /// Line search for primal Newton method
    fn line_search(
        &self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        w: &Array1<f64>,
        b: f64,
        direction: &Array1<f64>,
        margins: &[f64],
    ) -> Result<f64> {
        let n_features = x.ncols();
        let mut step_size = 1.0;
        let current_obj = self.calculate_primal_objective(w, margins);

        for _ in 0..20 {
            // Max 20 backtracking steps
            let mut new_w = w.clone();
            for k in 0..n_features {
                new_w[k] += step_size * direction[k];
            }
            let new_b = b + step_size * direction[n_features];
            let new_margins = self.calculate_margins(x, y, &new_w, new_b);
            let new_obj = self.calculate_primal_objective(&new_w, &new_margins);

            if new_obj < current_obj {
                return Ok(step_size);
            }

            step_size *= 0.5;
        }

        Ok(step_size)
    }

    /// Calculate primal objective value
    fn calculate_primal_objective(&self, w: &Array1<f64>, margins: &[f64]) -> f64 {
        let regularization = 0.5 * w.dot(w);
        let hinge_loss: f64 = margins.iter().map(|&margin| (1.0 - margin).max(0.0)).sum();

        regularization + self.config.c * hinge_loss
    }

    /// Helper methods for dual Newton method
    fn compute_kernel_matrix(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let kernel_type = self
            .kernel
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "compute_kernel_matrix".to_string(),
            })?;
        let kernel = create_kernel(kernel_type.clone())?;
        let n = x.nrows();
        let mut k_matrix = Array2::zeros((n, n));

        for i in 0..n {
            for j in 0..n {
                k_matrix[[i, j]] = kernel.compute(x.row(i), x.row(j));
            }
        }

        Ok(k_matrix)
    }

    fn calculate_dual_gradient(&self, k_matrix: &Array2<f64>, alpha: &Array1<f64>) -> Array1<f64> {
        &Array1::from_elem(alpha.len(), 1.0) - &k_matrix.dot(alpha)
    }

    fn calculate_dual_hessian(
        &self,
        k_matrix: &Array2<f64>,
        _alpha: &Array1<f64>,
    ) -> Result<Array2<f64>> {
        // For SVM, Hessian is just the kernel matrix
        Ok(k_matrix.clone())
    }

    fn dual_line_search(
        &self,
        k_matrix: &Array2<f64>,
        alpha: &Array1<f64>,
        direction: &Array1<f64>,
    ) -> Result<f64> {
        let mut step_size = 1.0;
        let current_obj = self.calculate_dual_objective(k_matrix, alpha);

        for _ in 0..20 {
            let new_alpha = alpha + &(direction * step_size);
            let new_obj = self.calculate_dual_objective(k_matrix, &new_alpha);

            if new_obj > current_obj {
                return Ok(step_size);
            }

            step_size *= 0.5;
        }

        Ok(step_size)
    }

    fn calculate_dual_objective(&self, k_matrix: &Array2<f64>, alpha: &Array1<f64>) -> f64 {
        alpha.sum() - 0.5 * alpha.dot(&k_matrix.dot(alpha))
    }

    fn find_support_vectors(&self, alpha: &Array1<f64>) -> Result<Vec<usize>> {
        let support_indices: Vec<usize> = alpha
            .iter()
            .enumerate()
            .filter(|(_, &val)| val > self.config.tol)
            .map(|(i, _)| i)
            .collect();

        Ok(support_indices)
    }

    fn calculate_intercept(
        &self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        alpha: &Array1<f64>,
        support_indices: &[usize],
    ) -> Result<f64> {
        if support_indices.is_empty() {
            return Ok(0.0);
        }

        let kernel_type = self
            .kernel
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "calculate_intercept".to_string(),
            })?;
        let kernel = create_kernel(kernel_type.clone())?;
        let mut intercept_sum = 0.0;
        let mut count = 0;

        for &i in support_indices {
            if alpha[i] > self.config.tol && alpha[i] < self.config.c - self.config.tol {
                let mut decision_value = 0.0;
                for &j in support_indices {
                    decision_value += alpha[j] * y[j] * kernel.compute(x.row(i), x.row(j));
                }
                intercept_sum += y[i] - decision_value;
                count += 1;
            }
        }

        Ok(if count > 0 {
            intercept_sum / count as f64
        } else {
            0.0
        })
    }

    /// Predict using the fitted model
    pub fn predict(&self, x: &Array2<f64>, result: &OptimizationResult) -> Result<Array1<f64>> {
        if !self.is_fitted {
            return Err(SklearsError::NotFitted {
                operation: "prediction".to_string(),
            });
        }

        let decision_values = self.decision_function(x, result)?;
        Ok(Array1::from_vec(
            decision_values
                .iter()
                .map(|&val| if val > 0.0 { 1.0 } else { -1.0 })
                .collect(),
        ))
    }

    /// Calculate decision function values
    pub fn decision_function(
        &self,
        x: &Array2<f64>,
        result: &OptimizationResult,
    ) -> Result<Array1<f64>> {
        if !self.is_fitted {
            return Err(SklearsError::NotFitted {
                operation: "prediction".to_string(),
            });
        }

        let kernel_type = self
            .kernel
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "decision_function".to_string(),
            })?;
        let kernel = create_kernel(kernel_type.clone())?;
        let mut decision_values = Array1::zeros(x.nrows());

        for i in 0..x.nrows() {
            let mut sum = 0.0;
            for &j in &result.support_indices {
                sum += result.dual_coef[j] * kernel.compute(x.row(i), x.row(j));
            }
            decision_values[i] = sum + result.intercept;
        }

        Ok(decision_values)
    }
}

/// Trust Region SVM Solver
///
/// Trust region methods are iterative optimization algorithms that maintain a "trust region"
/// around the current point and solve subproblems within this region. The trust region radius
/// is adaptively adjusted based on the quality of the quadratic approximation.
///
/// For SVM optimization, we apply trust region methods to the dual problem:
/// maximize W(α) = Σα_i - (1/2) Σ_i Σ_j α_i α_j y_i y_j K(x_i, x_j)
/// subject to Σ α_i y_i = 0 and 0 ≤ α_i ≤ C
///
/// Reference: Nocedal, J. & Wright, S. (2006). Numerical Optimization. Springer.
#[derive(Debug, Clone)]
pub struct TrustRegionSVM {
    config: AdvancedOptimizationConfig,
    kernel: Option<KernelType>,
    is_fitted: bool,
}

impl Default for TrustRegionSVM {
    /// Create a new trust region SVM solver with default configuration
    fn default() -> Self {
        Self::new(AdvancedOptimizationConfig::default())
    }
}

impl TrustRegionSVM {
    /// Create a new trust region SVM solver
    pub fn new(config: AdvancedOptimizationConfig) -> Self {
        Self {
            config,
            kernel: None,
            is_fitted: false,
        }
    }

    /// Fit the SVM using trust region optimization
    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<OptimizationResult> {
        // Validate inputs
        if x.nrows() != y.len() {
            return Err(SklearsError::InvalidInput(
                "Number of samples must match number of labels".to_string(),
            ));
        }

        let n_samples = x.nrows();
        self.kernel = Some(self.config.kernel.clone());

        // Compute kernel matrix
        let k_matrix = self.compute_kernel_matrix(x)?;

        // Initialize dual variables
        let mut alpha = Array1::zeros(n_samples);
        let mut trust_radius = self.config.trust_radius;
        let mut history = Vec::new();

        // Trust region iterations
        for iteration in 0..self.config.max_iter {
            // Compute gradient and Hessian
            let gradient = self.compute_dual_gradient(&k_matrix, &alpha);
            let hessian = self.compute_dual_hessian(&k_matrix);

            // Solve trust region subproblem
            let step = self.solve_trust_region_subproblem(&gradient, &hessian, trust_radius)?;

            // The trust-region subproblem minimizes f(alpha) = -W(alpha), where
            // W is the dual objective being maximized. `current_obj`/`new_obj`
            // below are W; the actual reduction in f therefore corresponds to an
            // *increase* in W, i.e. `new_obj - current_obj`.
            let current_obj = self.calculate_dual_objective(&k_matrix, &alpha);
            let new_alpha = self.project_onto_constraints(&(&alpha + &step));
            let new_obj = self.calculate_dual_objective(&k_matrix, &new_alpha);

            let actual_reduction = new_obj - current_obj;
            let predicted_reduction = self.compute_predicted_reduction(&gradient, &hessian, &step);

            // Compute trust region ratio
            let ratio = if predicted_reduction.abs() < 1e-12 {
                0.0
            } else {
                actual_reduction / predicted_reduction
            };

            if self.config.verbose && iteration % 10 == 0 {
                println!(
                    "Trust Region Iter {}: Obj = {:.6}, Trust Radius = {:.6}, Ratio = {:.3}",
                    iteration, current_obj, trust_radius, ratio
                );
            }

            // Update trust region radius and accept/reject step
            let step_norm = step.dot(&step).sqrt();
            if ratio > 0.75 && (step_norm - trust_radius).abs() < 1e-6 {
                // Very good step and we hit the boundary, expand trust region
                trust_radius = (2.0 * trust_radius).min(10.0);
                alpha = new_alpha;
            } else if ratio > 0.25 {
                // Good step, accept and maintain trust region
                alpha = new_alpha;
            } else if ratio > 0.0 {
                // Mediocre step, accept but shrink trust region
                trust_radius *= 0.5;
                alpha = new_alpha;
            } else {
                // Bad step, reject and shrink trust region significantly
                trust_radius *= 0.25;
            }

            // Ensure minimum trust region radius
            trust_radius = trust_radius.max(1e-8);

            history.push(current_obj);

            // Check convergence
            if gradient.dot(&gradient).sqrt() < self.config.tol || trust_radius < 1e-8 {
                if self.config.verbose {
                    println!("Trust Region converged after {} iterations", iteration + 1);
                }

                let support_indices = self.find_support_vectors(&alpha)?;
                let intercept = self.calculate_intercept(x, y, &alpha, &support_indices)?;

                self.is_fitted = true;

                return Ok(OptimizationResult {
                    dual_coef: alpha,
                    intercept,
                    support_indices,
                    n_iterations: iteration + 1,
                    objective_value: current_obj,
                    converged: true,
                    history,
                });
            }
        }

        // Return result even if not converged
        let support_indices = self.find_support_vectors(&alpha)?;
        let intercept = self.calculate_intercept(x, y, &alpha, &support_indices)?;

        self.is_fitted = true;

        Ok(OptimizationResult {
            dual_coef: alpha,
            intercept,
            support_indices,
            n_iterations: self.config.max_iter,
            objective_value: history.last().copied().unwrap_or(0.0),
            converged: false,
            history,
        })
    }

    /// Solve trust region subproblem using Cauchy point and Newton step
    fn solve_trust_region_subproblem(
        &self,
        gradient: &Array1<f64>,
        hessian: &Array2<f64>,
        trust_radius: f64,
    ) -> Result<Array1<f64>> {
        // Compute Cauchy point (steepest descent direction)
        let cauchy_step = self.compute_cauchy_point(gradient, hessian, trust_radius);

        // Compute Newton step with a robust general solver.
        let neg_gradient = gradient.mapv(|v| -v);
        let newton_step = solve_linear_system(hessian, &neg_gradient, self.config.newton_reg);
        let newton_norm = newton_step.dot(&newton_step).sqrt();

        if newton_norm <= trust_radius {
            // Newton step is within trust region
            return Ok(newton_step);
        }

        // Newton step is outside trust region, use dogleg method
        Ok(self.dogleg_method(&newton_step, &cauchy_step, trust_radius))
    }

    /// Compute Cauchy point (steepest descent step within trust region)
    fn compute_cauchy_point(
        &self,
        gradient: &Array1<f64>,
        hessian: &Array2<f64>,
        trust_radius: f64,
    ) -> Array1<f64> {
        let grad_norm = gradient.dot(gradient).sqrt();

        if grad_norm < 1e-12 {
            return Array1::zeros(gradient.len());
        }

        let unit_grad = gradient / grad_norm;
        let hess_grad = hessian.dot(&unit_grad);
        let curvature = unit_grad.dot(&hess_grad);

        if curvature <= 0.0 {
            // Negative curvature, go to boundary
            &unit_grad * (-trust_radius)
        } else {
            // Positive curvature, minimize quadratic or go to boundary
            let optimal_step = grad_norm / curvature;
            let step_length = optimal_step.min(trust_radius);
            &unit_grad * (-step_length)
        }
    }

    /// Dogleg method for combining Cauchy point and Newton step
    fn dogleg_method(
        &self,
        newton_step: &Array1<f64>,
        cauchy_step: &Array1<f64>,
        trust_radius: f64,
    ) -> Array1<f64> {
        let cauchy_norm = cauchy_step.dot(cauchy_step).sqrt();

        if cauchy_norm >= trust_radius {
            // Cauchy point is outside trust region
            return cauchy_step * (trust_radius / cauchy_norm);
        }

        // Find intersection of dogleg path with trust region
        let dogleg_direction = newton_step - cauchy_step;
        let a = dogleg_direction.dot(&dogleg_direction);
        let b = 2.0 * cauchy_step.dot(&dogleg_direction);
        let c = cauchy_norm * cauchy_norm - trust_radius * trust_radius;

        if a < 1e-12 {
            return cauchy_step.clone();
        }

        let discriminant = b * b - 4.0 * a * c;
        if discriminant < 0.0 {
            return cauchy_step.clone();
        }

        let tau = (-b + discriminant.sqrt()) / (2.0 * a);
        let tau = tau.clamp(0.0, 1.0);

        cauchy_step + &(&dogleg_direction * tau)
    }

    /// Compute predicted reduction for trust region ratio
    fn compute_predicted_reduction(
        &self,
        gradient: &Array1<f64>,
        hessian: &Array2<f64>,
        step: &Array1<f64>,
    ) -> f64 {
        let linear_term = gradient.dot(step);
        let quadratic_term = 0.5 * step.dot(&hessian.dot(step));
        -(linear_term + quadratic_term)
    }

    /// Project dual variables onto constraint set [0, C]
    fn project_onto_constraints(&self, alpha: &Array1<f64>) -> Array1<f64> {
        alpha.mapv(|val| val.max(0.0).min(self.config.c))
    }

    /// Compute the gradient of the minimization objective `f = -W`.
    ///
    /// The dual objective `W(alpha) = e^T alpha - 0.5 alpha^T K alpha` is
    /// maximized, so the trust-region machinery (which is formulated for
    /// minimization) operates on `f(alpha) = -W(alpha)` whose gradient is
    /// `nabla f = K alpha - e`.
    fn compute_dual_gradient(&self, k_matrix: &Array2<f64>, alpha: &Array1<f64>) -> Array1<f64> {
        &k_matrix.dot(alpha) - &Array1::from_elem(alpha.len(), 1.0)
    }

    /// Compute Hessian of the minimization objective (kernel matrix).
    ///
    /// `nabla^2 f = K`, which is positive semidefinite, matching the
    /// trust-region subproblem's assumptions.
    fn compute_dual_hessian(&self, k_matrix: &Array2<f64>) -> Array2<f64> {
        k_matrix.clone()
    }

    /// Compute kernel matrix
    fn compute_kernel_matrix(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let kernel_type = self
            .kernel
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "compute_kernel_matrix".to_string(),
            })?;
        let kernel = create_kernel(kernel_type.clone())?;
        let n = x.nrows();
        let mut k_matrix = Array2::zeros((n, n));

        for i in 0..n {
            for j in 0..n {
                k_matrix[[i, j]] = kernel.compute(x.row(i), x.row(j));
            }
        }

        Ok(k_matrix)
    }

    /// Calculate dual objective value
    fn calculate_dual_objective(&self, k_matrix: &Array2<f64>, alpha: &Array1<f64>) -> f64 {
        alpha.sum() - 0.5 * alpha.dot(&k_matrix.dot(alpha))
    }

    /// Find support vectors
    fn find_support_vectors(&self, alpha: &Array1<f64>) -> Result<Vec<usize>> {
        let support_indices: Vec<usize> = alpha
            .iter()
            .enumerate()
            .filter(|(_, &val)| val > self.config.tol)
            .map(|(i, _)| i)
            .collect();

        Ok(support_indices)
    }

    /// Calculate intercept term
    fn calculate_intercept(
        &self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        alpha: &Array1<f64>,
        support_indices: &[usize],
    ) -> Result<f64> {
        if support_indices.is_empty() {
            return Ok(0.0);
        }

        let kernel_type = self
            .kernel
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "calculate_intercept".to_string(),
            })?;
        let kernel = create_kernel(kernel_type.clone())?;
        let mut intercept_sum = 0.0;
        let mut count = 0;

        for &i in support_indices {
            if alpha[i] > self.config.tol && alpha[i] < self.config.c - self.config.tol {
                let mut decision_value = 0.0;
                for &j in support_indices {
                    decision_value += alpha[j] * y[j] * kernel.compute(x.row(i), x.row(j));
                }
                intercept_sum += y[i] - decision_value;
                count += 1;
            }
        }

        Ok(if count > 0 {
            intercept_sum / count as f64
        } else {
            0.0
        })
    }

    /// Predict using the fitted model
    pub fn predict(&self, x: &Array2<f64>, result: &OptimizationResult) -> Result<Array1<f64>> {
        if !self.is_fitted {
            return Err(SklearsError::NotFitted {
                operation: "prediction".to_string(),
            });
        }

        let decision_values = self.decision_function(x, result)?;
        Ok(Array1::from_vec(
            decision_values
                .iter()
                .map(|&val| if val > 0.0 { 1.0 } else { -1.0 })
                .collect(),
        ))
    }

    /// Calculate decision function values
    pub fn decision_function(
        &self,
        x: &Array2<f64>,
        result: &OptimizationResult,
    ) -> Result<Array1<f64>> {
        if !self.is_fitted {
            return Err(SklearsError::NotFitted {
                operation: "prediction".to_string(),
            });
        }

        let kernel_type = self
            .kernel
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "decision_function".to_string(),
            })?;
        let kernel = create_kernel(kernel_type.clone())?;
        let mut decision_values = Array1::zeros(x.nrows());

        for i in 0..x.nrows() {
            let mut sum = 0.0;
            for &j in &result.support_indices {
                sum += result.dual_coef[j] * kernel.compute(x.row(i), x.row(j));
            }
            decision_values[i] = sum + result.intercept;
        }

        Ok(decision_values)
    }
}

/// Accelerated Gradient SVM Solver
///
/// Implements accelerated gradient descent methods for SVM optimization,
/// including Nesterov's accelerated gradient method and FISTA.
/// These methods achieve faster convergence rates than standard gradient descent.
///
/// Nesterov's method uses momentum to accelerate convergence:
/// y_{k+1} = x_k - γ∇f(x_k)
/// x_{k+1} = y_{k+1} + β(y_{k+1} - y_k)
///
/// Reference: Nesterov, Y. (2013). Introductory lectures on convex optimization.
#[derive(Debug, Clone)]
pub struct AcceleratedGradientSVM {
    config: AdvancedOptimizationConfig,
    kernel: Option<KernelType>,
    is_fitted: bool,
    /// Momentum parameter (typically 0.9)
    pub momentum: f64,
    /// Learning rate schedule
    pub learning_rate: f64,
    /// Accelerated method type
    pub method: AcceleratedMethod,
}

/// Types of accelerated gradient methods
#[derive(Debug, Clone)]
pub enum AcceleratedMethod {
    /// Nesterov's accelerated gradient method
    Nesterov,
    /// Fast Iterative Shrinkage-Thresholding Algorithm
    FISTA,
    /// Heavy ball method
    HeavyBall,
}

impl AcceleratedGradientSVM {
    /// Create a new accelerated gradient SVM solver
    pub fn new(config: AdvancedOptimizationConfig) -> Self {
        Self {
            config,
            kernel: None,
            is_fitted: false,
            momentum: 0.9,
            learning_rate: 0.01,
            method: AcceleratedMethod::Nesterov,
        }
    }

    /// Set the momentum parameter
    pub fn with_momentum(mut self, momentum: f64) -> Self {
        self.momentum = momentum;
        self
    }

    /// Set the learning rate
    pub fn with_learning_rate(mut self, learning_rate: f64) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    /// Set the accelerated method type
    pub fn with_method(mut self, method: AcceleratedMethod) -> Self {
        self.method = method;
        self
    }

    /// Fit the SVM using accelerated gradient optimization
    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<OptimizationResult> {
        // Validate inputs
        if x.nrows() != y.len() {
            return Err(SklearsError::InvalidInput(
                "Number of samples must match number of labels".to_string(),
            ));
        }

        let n_samples = x.nrows();
        self.kernel = Some(self.config.kernel.clone());

        // Compute kernel matrix
        let k_matrix = self.compute_kernel_matrix(x)?;

        // Initialize dual variables
        let mut alpha: Array1<f64> = Array1::zeros(n_samples);
        let mut t = 1.0; // FISTA parameter
        let mut history = Vec::new();

        // Adaptive learning rate
        let mut current_lr = self.learning_rate;

        // Accelerated gradient iterations
        for iteration in 0..self.config.max_iter {
            // Compute gradient at current point
            let gradient = self.compute_dual_gradient(&k_matrix, &alpha);

            // Calculate objective value
            let objective = self.calculate_objective(&k_matrix, &alpha)?;
            history.push(objective);

            if self.config.verbose && iteration % 10 == 0 {
                println!(
                    "Accelerated Gradient Iteration {}: Objective = {:.6}",
                    iteration, objective
                );
            }

            // Store previous value
            let alpha_prev = alpha.clone();

            // Update based on method type
            match self.method {
                AcceleratedMethod::Nesterov => {
                    // Nesterov's accelerated gradient method
                    let momentum_coeff = if iteration == 0 { 0.0 } else { self.momentum };

                    // Compute momentum term
                    let momentum_term = (&alpha - &alpha_prev) * momentum_coeff;

                    // Update with momentum
                    let y_k = &alpha + &momentum_term;

                    // Gradient step
                    let gradient_at_y = self.compute_dual_gradient(&k_matrix, &y_k);
                    alpha = &y_k - &(&gradient_at_y * current_lr);
                }
                AcceleratedMethod::FISTA => {
                    // Fast Iterative Shrinkage-Thresholding Algorithm
                    let gradient_step = &alpha - &(&gradient * current_lr);

                    // Proximal operator (projection onto constraint set)
                    let alpha_new = self.proximal_operator(&gradient_step)?;

                    // Update FISTA parameter
                    let t_new = (1.0_f64 + (1.0_f64 + 4.0_f64 * t * t).sqrt()) / 2.0_f64;
                    let beta = (t - 1.0) / t_new;

                    // Extrapolation point for the next iterate (Nesterov momentum).
                    let _y_k = &alpha_new + &((&alpha_new - &alpha) * beta);
                    alpha = alpha_new;
                    t = t_new;
                }
                AcceleratedMethod::HeavyBall => {
                    // Heavy ball method
                    let momentum_coeff = if iteration == 0 { 0.0 } else { self.momentum };

                    // Update with momentum
                    let alpha_new = &(&alpha - &(&gradient * current_lr))
                        + &((&alpha - &alpha_prev) * momentum_coeff);
                    alpha = alpha_new;
                }
            }

            // Project onto constraint set [0, C]
            for i in 0..n_samples {
                alpha[i] = alpha[i].max(0.0).min(self.config.c);
            }

            // Check convergence
            let gradient_norm = gradient.dot(&gradient).sqrt();
            if gradient_norm < self.config.tol {
                if self.config.verbose {
                    println!(
                        "Accelerated Gradient converged after {} iterations",
                        iteration + 1
                    );
                }

                self.is_fitted = true;

                let support_indices = self.find_support_vectors(&alpha)?;
                let intercept = self.calculate_intercept(x, y, &alpha, &support_indices)?;

                return Ok(OptimizationResult {
                    dual_coef: alpha,
                    intercept,
                    support_indices,
                    n_iterations: iteration + 1,
                    objective_value: objective,
                    converged: true,
                    history,
                });
            }

            // Adaptive learning rate adjustment
            if iteration > 0 && history.len() >= 2 {
                let prev_obj = history[history.len() - 2];
                let curr_obj = history[history.len() - 1];

                // If objective increases, reduce learning rate
                if curr_obj > prev_obj {
                    current_lr *= 0.8;
                } else if curr_obj < prev_obj && (prev_obj - curr_obj) / prev_obj.abs() > 0.01 {
                    // If significant improvement, slightly increase learning rate
                    current_lr *= 1.05;
                }

                // Keep learning rate within bounds
                current_lr = current_lr.clamp(1e-6, 1.0);
            }
        }

        self.is_fitted = true;

        // Return result even if not converged
        let support_indices = self.find_support_vectors(&alpha)?;
        let intercept = self.calculate_intercept(x, y, &alpha, &support_indices)?;

        Ok(OptimizationResult {
            dual_coef: alpha,
            intercept,
            support_indices,
            n_iterations: self.config.max_iter,
            objective_value: history.last().copied().unwrap_or(0.0),
            converged: false,
            history,
        })
    }

    /// Proximal operator for FISTA (projection onto constraint set)
    fn proximal_operator(&self, x: &Array1<f64>) -> Result<Array1<f64>> {
        let mut result = x.clone();

        // Project onto box constraints [0, C]
        for i in 0..result.len() {
            result[i] = result[i].max(0.0).min(self.config.c);
        }

        Ok(result)
    }

    /// Compute dual gradient for SVM
    fn compute_dual_gradient(&self, k_matrix: &Array2<f64>, alpha: &Array1<f64>) -> Array1<f64> {
        let n = alpha.len();
        let mut gradient = Array1::from_elem(n, -1.0); // -e vector

        // Add Q*alpha term where Q = K (kernel matrix)
        for i in 0..n {
            for j in 0..n {
                gradient[i] += k_matrix[[i, j]] * alpha[j];
            }
        }

        gradient
    }

    /// Compute kernel matrix
    fn compute_kernel_matrix(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let kernel_type = self
            .kernel
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "compute_kernel_matrix".to_string(),
            })?;
        let kernel = create_kernel(kernel_type.clone())?;
        let n = x.nrows();
        let mut k_matrix = Array2::zeros((n, n));

        for i in 0..n {
            for j in 0..n {
                k_matrix[[i, j]] = kernel.compute(x.row(i), x.row(j));
            }
        }

        Ok(k_matrix)
    }

    /// Calculate objective value
    fn calculate_objective(&self, k_matrix: &Array2<f64>, alpha: &Array1<f64>) -> Result<f64> {
        let n = alpha.len();
        let mut objective = 0.0;

        // Dual objective: maximize Σαᵢ - (1/2)Σᵢⱼ αᵢαⱼyᵢyⱼK(xᵢ,xⱼ)
        for i in 0..n {
            objective += alpha[i]; // Linear term
            for j in 0..n {
                objective -= 0.5 * alpha[i] * alpha[j] * k_matrix[[i, j]]; // Quadratic term
            }
        }

        Ok(objective)
    }

    /// Find support vectors
    fn find_support_vectors(&self, alpha: &Array1<f64>) -> Result<Vec<usize>> {
        let mut support_indices = Vec::new();
        let tol = 1e-6;

        for (i, &alpha_i) in alpha.iter().enumerate() {
            if alpha_i > tol && alpha_i < self.config.c - tol {
                support_indices.push(i);
            }
        }

        Ok(support_indices)
    }

    /// Calculate intercept
    fn calculate_intercept(
        &self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        alpha: &Array1<f64>,
        support_indices: &[usize],
    ) -> Result<f64> {
        if support_indices.is_empty() {
            return Ok(0.0);
        }

        let kernel_type = self
            .kernel
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "calculate_intercept".to_string(),
            })?;
        let kernel = create_kernel(kernel_type.clone())?;
        let mut intercept_sum = 0.0;

        for &sv_idx in support_indices {
            let mut kernel_sum = 0.0;
            for (j, &alpha_j) in alpha.iter().enumerate() {
                if alpha_j > 0.0 {
                    kernel_sum += alpha_j * y[j] * kernel.compute(x.row(sv_idx), x.row(j));
                }
            }
            intercept_sum += y[sv_idx] - kernel_sum;
        }

        Ok(intercept_sum / support_indices.len() as f64)
    }

    /// Make predictions
    pub fn predict(&self, x: &Array2<f64>, result: &OptimizationResult) -> Result<Array1<f64>> {
        let decision_values = self.decision_function(x, result)?;
        let mut predictions = Array1::zeros(decision_values.len());

        for (i, &val) in decision_values.iter().enumerate() {
            predictions[i] = if val >= 0.0 { 1.0 } else { -1.0 };
        }

        Ok(predictions)
    }

    /// Compute decision function
    pub fn decision_function(
        &self,
        x: &Array2<f64>,
        result: &OptimizationResult,
    ) -> Result<Array1<f64>> {
        let kernel_type = self
            .kernel
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "decision_function".to_string(),
            })?;
        let kernel = create_kernel(kernel_type.clone())?;
        let n_test = x.nrows();
        let mut decision_values = Array1::zeros(n_test);

        for i in 0..n_test {
            let mut sum = 0.0;
            for &j in &result.support_indices {
                sum += result.dual_coef[j] * kernel.compute(x.row(i), x.row(j));
            }
            decision_values[i] = sum + result.intercept;
        }

        Ok(decision_values)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
#[path = "advanced_optimization_tests.rs"]
mod tests;
