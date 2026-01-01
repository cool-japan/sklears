//! CLIME (Constrained L1 Minimization) for sparse precision matrix estimation
//!
//! This module implements the CLIME algorithm for estimating sparse precision matrices
//! by solving a series of L1-constrained regression problems, providing robust
//! sparse estimation even in high-dimensional settings.

use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use scirs2_linalg::compat::ArrayLinalgExt;
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Untrained},
    types::Float,
};

/// CLIME (Constrained L1 Minimization) estimator
///
/// Estimates sparse precision matrices by solving L1-constrained optimization
/// problems for each column of the precision matrix independently.
#[derive(Debug, Clone)]
pub struct CLIME<S = Untrained> {
    state: S,
    /// L1 constraint parameter (lambda)
    lambda: f64,
    /// Maximum number of iterations for optimization
    max_iter: usize,
    /// Convergence tolerance
    tol: f64,
    /// Whether to assume centered data
    assume_centered: bool,
    /// Method for solving the L1-constrained subproblems
    solver: CLIMESolver,
}

/// Solver methods for CLIME optimization
#[derive(Debug, Clone)]
pub enum CLIMESolver {
    /// Coordinate descent algorithm
    CoordinateDescent,
    /// Proximal gradient method
    ProximalGradient,
}

impl Default for CLIME<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl CLIME<Untrained> {
    /// Create a new CLIME estimator
    pub fn new() -> Self {
        Self {
            state: Untrained,
            lambda: 0.1,
            max_iter: 1000,
            tol: 1e-6,
            assume_centered: false,
            solver: CLIMESolver::CoordinateDescent,
        }
    }

    /// Set the L1 constraint parameter
    pub fn lambda(mut self, lambda: f64) -> Self {
        self.lambda = lambda;
        self
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

    /// Set whether to assume centered data
    pub fn assume_centered(mut self, assume_centered: bool) -> Self {
        self.assume_centered = assume_centered;
        self
    }

    /// Set the optimization solver
    pub fn solver(mut self, solver: CLIMESolver) -> Self {
        self.solver = solver;
        self
    }
}

impl Estimator for CLIME<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, ()> for CLIME<Untrained> {
    type Fitted = CLIME<CLIMETrained>;

    fn fit(self, x: &ArrayView2<'_, Float>, _y: &()) -> SklResult<Self::Fitted> {
        let x = *x;
        let (n_samples, n_features) = x.dim();

        if n_samples < 2 {
            return Err(SklearsError::InvalidInput(
                "Need at least 2 samples for covariance estimation".to_string(),
            ));
        }

        if n_features < 2 {
            return Err(SklearsError::InvalidInput(
                "Need at least 2 features for covariance estimation".to_string(),
            ));
        }

        // Center the data if needed
        let x_centered = if self.assume_centered {
            x.to_owned()
        } else {
            let mean = x.mean_axis(Axis(0)).unwrap();
            x.to_owned() - &mean.insert_axis(Axis(0))
        };

        // Compute sample covariance matrix
        let sample_cov = (&x_centered.t().dot(&x_centered)) / (n_samples as f64 - 1.0);

        // Estimate precision matrix using CLIME
        let precision = self.estimate_precision_matrix(&sample_cov)?;

        // Compute covariance matrix from precision
        let covariance = self.compute_covariance_from_precision(&precision)?;

        Ok(CLIME {
            state: CLIMETrained {
                covariance,
                precision,
                lambda: self.lambda,
                n_iter: self.max_iter,
                assume_centered: self.assume_centered,
            },
            lambda: self.lambda,
            max_iter: self.max_iter,
            tol: self.tol,
            assume_centered: self.assume_centered,
            solver: self.solver.clone(),
        })
    }
}

/// Trained CLIME estimator
#[derive(Debug, Clone)]
pub struct CLIMETrained {
    covariance: Array2<f64>,
    precision: Array2<f64>,
    lambda: f64,
    n_iter: usize,
    assume_centered: bool,
}

impl CLIME<CLIMETrained> {
    /// Get the estimated covariance matrix
    pub fn get_covariance(&self) -> &Array2<f64> {
        &self.state.covariance
    }

    /// Get the estimated precision matrix
    pub fn get_precision(&self) -> &Array2<f64> {
        &self.state.precision
    }

    /// Get the lambda parameter used
    pub fn get_lambda(&self) -> f64 {
        self.state.lambda
    }

    /// Get the number of iterations performed
    pub fn get_n_iter(&self) -> usize {
        self.state.n_iter
    }

    /// Check if data was assumed to be centered
    pub fn is_assume_centered(&self) -> bool {
        self.state.assume_centered
    }

    /// Compute sparsity level of the precision matrix
    pub fn get_sparsity(&self) -> f64 {
        let total_elements = (self.state.precision.nrows() * self.state.precision.ncols()) as f64;
        let nonzero_elements = self
            .state
            .precision
            .iter()
            .filter(|&&x| x.abs() > 1e-10)
            .count() as f64;

        1.0 - (nonzero_elements / total_elements)
    }
}

impl CLIME<Untrained> {
    /// Estimate precision matrix using CLIME algorithm
    fn estimate_precision_matrix(&self, sample_cov: &Array2<f64>) -> SklResult<Array2<f64>> {
        let n_features = sample_cov.nrows();
        let mut precision = Array2::zeros((n_features, n_features));

        // Solve L1-constrained problem for each column
        for j in 0..n_features {
            let column_j = self.solve_clime_column(sample_cov, j)?;
            precision.column_mut(j).assign(&column_j);
        }

        // Symmetrize the precision matrix
        self.symmetrize_matrix(&mut precision);

        Ok(precision)
    }

    /// Solve CLIME optimization for a single column
    fn solve_clime_column(&self, sample_cov: &Array2<f64>, j: usize) -> SklResult<Array1<f64>> {
        let n_features = sample_cov.nrows();
        let mut beta = Array1::zeros(n_features);

        // Create the regression problem: minimize ||beta||_1 subject to ||S * beta - e_j||_inf <= lambda
        let ej = {
            let mut e = Array1::zeros(n_features);
            e[j] = 1.0;
            e
        };

        match self.solver {
            CLIMESolver::CoordinateDescent => {
                self.solve_coordinate_descent(sample_cov, &ej.view(), &mut beta)?;
            }
            CLIMESolver::ProximalGradient => {
                self.solve_proximal_gradient(sample_cov, &ej.view(), &mut beta)?;
            }
        }

        Ok(beta)
    }

    /// Solve using coordinate descent
    fn solve_coordinate_descent(
        &self,
        sample_cov: &Array2<f64>,
        target: &ArrayView1<f64>,
        beta: &mut Array1<f64>,
    ) -> SklResult<()> {
        let n_features = sample_cov.nrows();

        for _iter in 0..self.max_iter {
            let beta_old = beta.clone();

            for k in 0..n_features {
                // Compute residual without k-th component
                let mut residual = sample_cov.dot(beta) - target;
                residual = residual - &(sample_cov.column(k).to_owned() * beta[k]);

                // Update k-th component using soft thresholding
                let sk = sample_cov[[k, k]];
                // Use more conservative threshold to avoid division by very small numbers
                if sk.abs() > 1e-8 {
                    let zk = -residual.dot(&sample_cov.column(k)) / sk;
                    // Clip zk to prevent extreme values
                    let zk_clipped = zk.max(-1e10).min(1e10);
                    let threshold = (self.lambda / sk).max(-1e10).min(1e10);
                    beta[k] = self.soft_threshold(zk_clipped, threshold);

                    // Check for non-finite values
                    if !beta[k].is_finite() {
                        beta[k] = 0.0;
                    }
                }
            }

            // Check convergence
            let diff = &*beta - &beta_old;
            let norm_diff = diff.dot(&diff).sqrt();
            if !norm_diff.is_finite() {
                return Err(SklearsError::NumericalError(
                    "CLIME coordinate descent produced non-finite values".to_string(),
                ));
            }
            if norm_diff < self.tol {
                break;
            }
        }

        Ok(())
    }

    /// Solve using proximal gradient method
    fn solve_proximal_gradient(
        &self,
        sample_cov: &Array2<f64>,
        target: &ArrayView1<f64>,
        beta: &mut Array1<f64>,
    ) -> SklResult<()> {
        let step_size = 0.01; // Fixed step size for simplicity

        for _iter in 0..self.max_iter {
            let beta_old = beta.clone();

            // Gradient step
            let residual = sample_cov.dot(beta) - target;
            let gradient = sample_cov.t().dot(&residual);
            *beta -= &(gradient * step_size);

            // Proximal operator (soft thresholding)
            for i in 0..beta.len() {
                beta[i] = self.soft_threshold(beta[i], self.lambda * step_size);
            }

            // Check convergence
            let diff = &*beta - &beta_old;
            if diff.dot(&diff).sqrt() < self.tol {
                break;
            }
        }

        Ok(())
    }

    /// Soft thresholding operator
    fn soft_threshold(&self, x: f64, threshold: f64) -> f64 {
        if x > threshold {
            x - threshold
        } else if x < -threshold {
            x + threshold
        } else {
            0.0
        }
    }

    /// Symmetrize the precision matrix
    fn symmetrize_matrix(&self, matrix: &mut Array2<f64>) {
        let n = matrix.nrows();
        for i in 0..n {
            for j in (i + 1)..n {
                let avg = (matrix[[i, j]] + matrix[[j, i]]) / 2.0;
                matrix[[i, j]] = avg;
                matrix[[j, i]] = avg;
            }
        }
    }

    /// Compute covariance matrix from precision matrix
    fn compute_covariance_from_precision(&self, precision: &Array2<f64>) -> SklResult<Array2<f64>> {
        // Check for non-finite values in precision matrix
        for &val in precision.iter() {
            if !val.is_finite() {
                return Err(SklearsError::NumericalError(
                    "Precision matrix contains non-finite values".to_string(),
                ));
            }
        }

        // Add small regularization to ensure invertibility
        let n = precision.nrows();
        let mut regularized_precision = precision.clone();
        for i in 0..n {
            regularized_precision[[i, i]] += 1e-6;
        }

        // Try direct inversion first for better accuracy when possible
        if let Ok(covariance) = regularized_precision.inv() {
            return Ok(covariance);
        }

        // Fall back to SVD-based pseudo-inverse for numerical stability
        use scirs2_linalg::compat::svd;
        let (u, s, vt) = svd(&regularized_precision.view(), true).map_err(|e| {
            SklearsError::NumericalError(format!(
                "Matrix inversion failed - both direct and SVD methods failed: {}",
                e
            ))
        })?;

        // Compute reciprocals of singular values with threshold
        let threshold = 1e-10 * s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let s_inv = s.mapv(|x| if x > threshold { 1.0 / x } else { 0.0 });

        // Reconstruct covariance matrix: V * S^-1 * U^T
        use scirs2_core::ndarray::Array2 as NdArray2;
        let s_inv_diag = NdArray2::from_diag(&s_inv);
        let temp = vt.t().dot(&s_inv_diag);
        Ok(temp.dot(&u.t()))
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_clime_basic() {
        let x = array![
            [1.0, 0.5, 0.1],
            [2.0, 1.5, 0.2],
            [3.0, 2.8, 0.3],
            [4.0, 3.9, 0.4],
            [5.0, 4.1, 0.5],
            [1.5, 0.8, 0.15],
            [2.5, 1.9, 0.25],
            [3.5, 3.1, 0.35]
        ];

        let estimator = CLIME::new().lambda(0.1);
        let fitted = estimator.fit(&x.view(), &()).unwrap();

        assert_eq!(fitted.get_covariance().dim(), (3, 3));
        assert_eq!(fitted.get_precision().dim(), (3, 3));
        assert_eq!(fitted.get_lambda(), 0.1);
    }

    #[test]
    fn test_clime_sparsity() {
        let x = array![
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [5.0, 0.0, 0.0]
        ];

        let estimator = CLIME::new().lambda(0.5);
        let fitted = estimator.fit(&x.view(), &()).unwrap();

        let sparsity = fitted.get_sparsity();
        assert!(sparsity > 0.0); // Should produce some sparsity
    }

    #[test]
    fn test_clime_proximal_gradient() {
        let x = array![[1.0, 0.5], [2.0, 1.5], [3.0, 2.8], [4.0, 3.9], [5.0, 4.1]];

        let estimator = CLIME::new()
            .lambda(0.1)
            .solver(CLIMESolver::ProximalGradient);
        let fitted = estimator.fit(&x.view(), &()).unwrap();

        assert_eq!(fitted.get_covariance().dim(), (2, 2));
        assert_eq!(fitted.get_precision().dim(), (2, 2));
    }

    #[test]
    fn test_clime_assume_centered() {
        let x = array![[0.0, -0.5], [1.0, 0.5], [2.0, 1.8], [3.0, 2.9], [4.0, 4.1]];

        let estimator = CLIME::new().assume_centered(true);
        let fitted = estimator.fit(&x.view(), &()).unwrap();

        assert!(fitted.is_assume_centered());
        assert_eq!(fitted.get_covariance().dim(), (2, 2));
    }

    #[test]
    fn test_soft_threshold() {
        let clime = CLIME::new();

        assert_eq!(clime.soft_threshold(2.0, 1.0), 1.0);
        assert_eq!(clime.soft_threshold(-2.0, 1.0), -1.0);
        assert_eq!(clime.soft_threshold(0.5, 1.0), 0.0);
        assert_eq!(clime.soft_threshold(-0.5, 1.0), 0.0);
    }

    #[test]
    fn test_clime_parameters() {
        let estimator = CLIME::new().lambda(0.2).max_iter(500).tol(1e-8);

        let x = array![[1.0, 0.5], [2.0, 1.5], [3.0, 2.8]];

        let fitted = estimator.fit(&x.view(), &()).unwrap();
        assert_eq!(fitted.get_lambda(), 0.2);
    }
}
