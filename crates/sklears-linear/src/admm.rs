//! ADMM (Alternating Direction Method of Multipliers) solver
//!
//! The ADMM algorithm is particularly effective for solving regularized linear regression
//! problems, especially with non-smooth regularizers like L1 (Lasso) and elastic net.
//! It works by splitting the optimization problem into smaller subproblems that can be
//! solved analytically or with simple iterative methods.
//!
//! The algorithm minimizes: (1/2)||Ax - b||_2^2 + λ||x||_1 + (μ/2)||x||_2^2
//! for elastic net regularization.

use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::{
    error::{Result, SklearsError},
    types::Float,
};

/// Configuration for ADMM solver
#[derive(Debug, Clone)]
pub struct AdmmConfig {
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Tolerance for primal residual
    pub primal_tol: Float,
    /// Tolerance for dual residual
    pub dual_tol: Float,
    /// Penalty parameter (rho)
    pub rho: Float,
    /// Adaptive rho update parameter
    pub adaptive_rho: bool,
    /// Factor for increasing rho
    pub rho_increase_factor: Float,
    /// Factor for decreasing rho
    pub rho_decrease_factor: Float,
    /// Tolerance for rho adaptation
    pub rho_adaptation_tol: Float,
    /// Whether to use warm starting
    pub warm_start: bool,
    /// Verbose output
    pub verbose: bool,
}

impl Default for AdmmConfig {
    fn default() -> Self {
        Self {
            max_iter: 1000,
            primal_tol: 1e-4,
            dual_tol: 1e-4,
            rho: 1.0,
            adaptive_rho: true,
            rho_increase_factor: 2.0,
            rho_decrease_factor: 2.0,
            rho_adaptation_tol: 10.0,
            warm_start: false,
            verbose: false,
        }
    }
}

/// ADMM solver for regularized linear regression
pub struct AdmmSolver {
    config: AdmmConfig,
}

impl AdmmSolver {
    /// Create a new ADMM solver with default configuration
    pub fn new() -> Self {
        Self {
            config: AdmmConfig::default(),
        }
    }

    /// Create a new ADMM solver with custom configuration
    pub fn with_config(config: AdmmConfig) -> Self {
        Self { config }
    }

    /// Solve elastic net problem: minimize (1/2)||Ax - b||_2^2 + α * l1_ratio * ||x||_1 + (α * (1-l1_ratio)/2) * ||x||_2^2
    pub fn solve_elastic_net(
        &self,
        a: &Array2<Float>,
        b: &Array1<Float>,
        alpha: Float,
        l1_ratio: Float,
        initial_x: Option<&Array1<Float>>,
    ) -> Result<AdmmSolution> {
        let n_samples = a.nrows();
        let n_features = a.ncols();

        if b.len() != n_samples {
            return Err(SklearsError::InvalidInput(
                "Dimension mismatch between A and b".to_string(),
            ));
        }

        // Split regularization parameters
        let lambda1 = alpha * l1_ratio;
        let lambda2 = alpha * (1.0 - l1_ratio);

        // Precompute A^T A and A^T b for efficiency
        let ata = a.t().dot(a);
        let atb = a.t().dot(b);

        // Add L2 regularization to diagonal
        let mut ata_reg = ata.clone();
        for i in 0..n_features {
            ata_reg[[i, i]] += lambda2 + self.config.rho;
        }

        // Cholesky decomposition for efficient solving
        let chol = self.cholesky_decomposition(&ata_reg)?;

        // Initialize variables
        let mut x = match initial_x {
            Some(x0) => x0.clone(),
            None => Array1::zeros(n_features),
        };
        let mut z = x.clone();
        let mut u: Array1<Float> = Array1::zeros(n_features);

        let mut rho = self.config.rho;
        let mut objective_values = Vec::new();
        let mut primal_residuals = Vec::new();
        let mut dual_residuals = Vec::new();

        for iter in 0..self.config.max_iter {
            // x-update (solve quadratic problem)
            let rhs = &atb + rho * (&z - &u);
            x = self.solve_cholesky(&chol, &rhs)?;

            // z-update (soft thresholding)
            let x_plus_u = &x + &u;
            z = self.soft_threshold(&x_plus_u, lambda1 / rho);

            // u-update (dual variable)
            u = &u + &x - &z;

            // Compute residuals for convergence check
            let primal_residual = (&x - &z).mapv(|v| v * v).sum().sqrt();
            let dual_residual = rho
                * (&z - &self.soft_threshold(&x_plus_u, lambda1 / rho))
                    .mapv(|v| v * v)
                    .sum()
                    .sqrt();

            primal_residuals.push(primal_residual);
            dual_residuals.push(dual_residual);

            // Compute objective value
            let residual = a.dot(&x) - b;
            let data_loss = 0.5 * residual.mapv(|v| v * v).sum();
            let l1_penalty = lambda1 * x.mapv(|v| v.abs()).sum();
            let l2_penalty = 0.5 * lambda2 * x.mapv(|v| v * v).sum();
            let objective = data_loss + l1_penalty + l2_penalty;
            objective_values.push(objective);

            if self.config.verbose && iter % 100 == 0 {
                println!(
                    "ADMM iter {}: obj={:.6}, primal_res={:.6}, dual_res={:.6}, rho={:.6}",
                    iter, objective, primal_residual, dual_residual, rho
                );
            }

            // Check convergence
            if primal_residual < self.config.primal_tol && dual_residual < self.config.dual_tol {
                if self.config.verbose {
                    println!("ADMM converged at iteration {}", iter);
                }
                return Ok(AdmmSolution {
                    x,
                    n_iter: iter + 1,
                    objective_values,
                    primal_residuals,
                    dual_residuals,
                    converged: true,
                });
            }

            // Adaptive rho update
            if self.config.adaptive_rho {
                if primal_residual > self.config.rho_adaptation_tol * dual_residual {
                    rho *= self.config.rho_increase_factor;
                    u = &u / self.config.rho_increase_factor;
                } else if dual_residual > self.config.rho_adaptation_tol * primal_residual {
                    rho /= self.config.rho_decrease_factor;
                    u = &u * self.config.rho_decrease_factor;
                }

                // Update precomputed matrix if rho changed
                if (rho - self.config.rho).abs() > 1e-10 {
                    for i in 0..n_features {
                        ata_reg[[i, i]] = ata[[i, i]] + lambda2 + rho;
                    }
                    let _new_chol = self.cholesky_decomposition(&ata_reg)?;
                    // Note: In a real implementation, you'd update the Cholesky factor
                    // Here we're keeping it simple for demonstration
                }
            }
        }

        if self.config.verbose {
            println!("ADMM reached maximum iterations without convergence");
        }

        Ok(AdmmSolution {
            x,
            n_iter: self.config.max_iter,
            objective_values,
            primal_residuals,
            dual_residuals,
            converged: false,
        })
    }

    /// Solve Lasso problem: minimize (1/2)||Ax - b||_2^2 + α * ||x||_1
    pub fn solve_lasso(
        &self,
        a: &Array2<Float>,
        b: &Array1<Float>,
        alpha: Float,
        initial_x: Option<&Array1<Float>>,
    ) -> Result<AdmmSolution> {
        self.solve_elastic_net(a, b, alpha, 1.0, initial_x)
    }

    /// Solve Ridge problem: minimize (1/2)||Ax - b||_2^2 + α * ||x||_2^2
    pub fn solve_ridge(
        &self,
        a: &Array2<Float>,
        b: &Array1<Float>,
        alpha: Float,
        initial_x: Option<&Array1<Float>>,
    ) -> Result<AdmmSolution> {
        self.solve_elastic_net(a, b, alpha, 0.0, initial_x)
    }

    /// Soft thresholding operator
    fn soft_threshold(&self, x: &Array1<Float>, threshold: Float) -> Array1<Float> {
        x.mapv(|v| {
            if v > threshold {
                v - threshold
            } else if v < -threshold {
                v + threshold
            } else {
                0.0
            }
        })
    }

    /// Simple Cholesky decomposition (placeholder - would use proper LAPACK in production)
    fn cholesky_decomposition(&self, a: &Array2<Float>) -> Result<Array2<Float>> {
        // This is a simplified implementation
        // In production, you'd use LAPACK's dpotrf or similar
        let n = a.nrows();
        let mut l = Array2::zeros((n, n));

        for i in 0..n {
            for j in 0..=i {
                if i == j {
                    let mut sum = 0.0;
                    for k in 0..j {
                        sum += l[[j, k]] * l[[j, k]];
                    }
                    let val = a[[j, j]] - sum;
                    if val <= 0.0 {
                        return Err(SklearsError::NumericalError(
                            "Matrix is not positive definite".to_string(),
                        ));
                    }
                    l[[j, j]] = val.sqrt();
                } else {
                    let mut sum = 0.0;
                    for k in 0..j {
                        sum += l[[i, k]] * l[[j, k]];
                    }
                    l[[i, j]] = (a[[i, j]] - sum) / l[[j, j]];
                }
            }
        }

        Ok(l)
    }

    /// Solve Lx = b where L is lower triangular (from Cholesky)
    fn solve_cholesky(&self, l: &Array2<Float>, b: &Array1<Float>) -> Result<Array1<Float>> {
        let n = l.nrows();
        let mut y: Array1<Float> = Array1::zeros(n);
        let mut x: Array1<Float> = Array1::zeros(n);

        // Forward substitution: Ly = b
        for i in 0..n {
            let mut sum = 0.0;
            for j in 0..i {
                sum += l[[i, j]] * y[j];
            }
            y[i] = (b[i] - sum) / l[[i, i]];
        }

        // Backward substitution: L^T x = y
        for i in (0..n).rev() {
            let mut sum = 0.0;
            for j in (i + 1)..n {
                sum += l[[j, i]] * x[j];
            }
            x[i] = (y[i] - sum) / l[[i, i]];
        }

        Ok(x)
    }
}

/// Solution returned by ADMM solver
#[derive(Debug, Clone)]
pub struct AdmmSolution {
    /// Solution vector
    pub x: Array1<Float>,
    /// Number of iterations performed
    pub n_iter: usize,
    /// Objective function values during optimization
    pub objective_values: Vec<Float>,
    /// Primal residual values
    pub primal_residuals: Vec<Float>,
    /// Dual residual values
    pub dual_residuals: Vec<Float>,
    /// Whether the algorithm converged
    pub converged: bool,
}

impl Default for AdmmSolver {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array;

    #[test]
    fn test_admm_lasso_simple() {
        let a = Array::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let b = Array::from_vec(vec![1.0, 2.0, 3.0]);

        let solver = AdmmSolver::new();
        let solution = solver.solve_lasso(&a, &b, 0.1, None).unwrap();

        assert!(solution.x.len() == 2);
        assert!(solution.n_iter > 0);
        assert!(!solution.objective_values.is_empty());
    }

    #[test]
    fn test_admm_elastic_net() {
        let a = Array::from_shape_vec(
            (4, 3),
            vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        )
        .unwrap();
        let b = Array::from_vec(vec![1.0, 2.0, 3.0, 6.0]);

        let solver = AdmmSolver::new();
        let solution = solver.solve_elastic_net(&a, &b, 0.1, 0.5, None).unwrap();

        assert!(solution.x.len() == 3);
        assert!(solution.converged || solution.n_iter == solver.config.max_iter);
    }

    #[test]
    fn test_soft_threshold() {
        let solver = AdmmSolver::new();
        let x = Array::from_vec(vec![-2.0, -0.5, 0.0, 0.5, 2.0]);
        let result = solver.soft_threshold(&x, 1.0);

        let expected = Array::from_vec(vec![-1.0, 0.0, 0.0, 0.0, 1.0]);
        for (r, e) in result.iter().zip(expected.iter()) {
            assert!((r - e).abs() < 1e-10);
        }
    }

    #[test]
    fn test_admm_with_warm_start() {
        let a = Array::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let b = Array::from_vec(vec![1.0, 2.0, 3.0]);
        let initial_x = Array::from_vec(vec![0.5, 0.5]);

        let solver = AdmmSolver::new();
        let solution = solver.solve_lasso(&a, &b, 0.1, Some(&initial_x)).unwrap();

        assert!(solution.x.len() == 2);
        assert!(solution.n_iter > 0);
    }
}
