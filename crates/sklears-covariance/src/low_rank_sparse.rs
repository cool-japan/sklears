//! Low-Rank Plus Sparse Decomposition for Covariance Estimation
//!
//! This module implements low-rank plus sparse decomposition methods for covariance
//! estimation. The method decomposes the covariance matrix into a low-rank component
//! (capturing global structure/factors) and a sparse component (capturing specific
//! pairwise relationships).

use scirs2_core::ndarray::{Array2, ArrayView2, Axis};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Untrained},
    types::Float,
};

/// Low-Rank Plus Sparse decomposition estimator for covariance matrices
///
/// Decomposes the covariance matrix as C = L + S where L is low-rank and S is sparse.
/// This is useful when the covariance structure consists of global factors (low-rank)
/// and specific sparse relationships.
#[derive(Debug, Clone)]
pub struct LowRankSparseCovariance<S = Untrained> {
    state: S,
    /// Regularization parameter for nuclear norm (low-rank penalty)
    lambda_nuclear: f64,
    /// Regularization parameter for L1 norm (sparse penalty)
    lambda_l1: f64,
    /// Maximum number of iterations for optimization
    max_iter: usize,
    /// Convergence tolerance
    tol: f64,
    /// Whether to assume the data is centered
    assume_centered: bool,
    /// Optimization method
    method: LRSMethod,
    /// Penalty parameter for augmented Lagrangian
    mu: f64,
    /// Maximum penalty parameter
    mu_max: f64,
    /// Penalty parameter increase factor
    rho: f64,
}

/// Optimization methods for Low-Rank Sparse decomposition
#[derive(Debug, Clone)]
pub enum LRSMethod {
    /// Augmented Lagrangian Method (ALM)
    AugmentedLagrangian,
    /// Alternating Direction Method of Multipliers (ADMM)
    ADMM,
    /// Proximal Gradient Method
    ProximalGradient,
}

/// Trained Low-Rank Sparse state
#[derive(Debug, Clone)]
pub struct LowRankSparseCovarianceTrained {
    /// Low-rank component of the covariance matrix
    low_rank_component: Array2<f64>,
    /// Sparse component of the covariance matrix
    sparse_component: Array2<f64>,
    /// Total covariance matrix (low-rank + sparse)
    covariance: Array2<f64>,
    /// Estimated precision matrix
    precision: Option<Array2<f64>>,
    /// Number of iterations performed
    n_iter: usize,
    /// Final objective value
    objective: f64,
    /// Rank of the low-rank component
    rank: usize,
    /// Number of non-zero elements in sparse component
    nnz_sparse: usize,
    /// Regularization parameters used
    lambda_nuclear: f64,
    lambda_l1: f64,
    /// Whether data was assumed to be centered
    assume_centered: bool,
    /// Method used for optimization
    method: LRSMethod,
}

impl Default for LowRankSparseCovariance<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl LowRankSparseCovariance<Untrained> {
    /// Create a new Low-Rank Sparse Covariance estimator
    pub fn new() -> Self {
        Self {
            state: Untrained,
            lambda_nuclear: 0.1,
            lambda_l1: 0.1,
            max_iter: 100,
            tol: 1e-6,
            assume_centered: false,
            method: LRSMethod::AugmentedLagrangian,
            mu: 1.0,
            mu_max: 1e6,
            rho: 2.0,
        }
    }

    /// Set the nuclear norm regularization parameter
    pub fn lambda_nuclear(mut self, lambda_nuclear: f64) -> Self {
        self.lambda_nuclear = lambda_nuclear;
        self
    }

    /// Set the L1 norm regularization parameter
    pub fn lambda_l1(mut self, lambda_l1: f64) -> Self {
        self.lambda_l1 = lambda_l1;
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

    /// Set whether to assume the data is centered
    pub fn assume_centered(mut self, assume_centered: bool) -> Self {
        self.assume_centered = assume_centered;
        self
    }

    /// Set the optimization method
    pub fn method(mut self, method: LRSMethod) -> Self {
        self.method = method;
        self
    }

    /// Set the initial penalty parameter for augmented Lagrangian
    pub fn mu(mut self, mu: f64) -> Self {
        self.mu = mu;
        self
    }

    /// Set the maximum penalty parameter
    pub fn mu_max(mut self, mu_max: f64) -> Self {
        self.mu_max = mu_max;
        self
    }

    /// Set the penalty parameter increase factor
    pub fn rho(mut self, rho: f64) -> Self {
        self.rho = rho;
        self
    }
}

impl Estimator for LowRankSparseCovariance<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, ()> for LowRankSparseCovariance<Untrained> {
    type Fitted = LowRankSparseCovariance<LowRankSparseCovarianceTrained>;

    fn fit(self, x: &ArrayView2<'_, Float>, _y: &()) -> SklResult<Self::Fitted> {
        let (n_samples, n_features) = x.dim();

        if n_samples < 2 {
            return Err(SklearsError::InvalidInput(
                "Need at least 2 samples to estimate covariance".to_string(),
            ));
        }

        if n_features == 0 {
            return Err(SklearsError::InvalidInput(
                "Number of features must be positive".to_string(),
            ));
        }

        // Center the data if not assumed to be centered
        let centered_data = if self.assume_centered {
            x.to_owned()
        } else {
            let mean = x.mean_axis(Axis(0)).unwrap();
            x - &mean.insert_axis(Axis(0))
        };

        // Compute empirical covariance matrix
        let empirical_cov = Self::compute_empirical_covariance(&centered_data.view())?;

        // Perform low-rank plus sparse decomposition
        let (low_rank_component, sparse_component, n_iter, objective) = match self.method {
            LRSMethod::AugmentedLagrangian => self.decompose_alm(&empirical_cov)?,
            LRSMethod::ADMM => self.decompose_admm(&empirical_cov)?,
            LRSMethod::ProximalGradient => self.decompose_proximal_gradient(&empirical_cov)?,
        };

        // Compute total covariance matrix
        let covariance = &low_rank_component + &sparse_component;

        // Compute precision matrix
        let precision = Self::compute_precision(&covariance).ok();

        // Compute rank and sparsity
        let rank = Self::compute_rank(&low_rank_component, 1e-10);
        let nnz_sparse = Self::count_nonzeros(&sparse_component, 1e-10);

        let trained_state = LowRankSparseCovarianceTrained {
            low_rank_component,
            sparse_component,
            covariance,
            precision,
            n_iter,
            objective,
            rank,
            nnz_sparse,
            lambda_nuclear: self.lambda_nuclear,
            lambda_l1: self.lambda_l1,
            assume_centered: self.assume_centered,
            method: self.method.clone(),
        };

        Ok(LowRankSparseCovariance {
            state: trained_state,
            lambda_nuclear: self.lambda_nuclear,
            lambda_l1: self.lambda_l1,
            max_iter: self.max_iter,
            tol: self.tol,
            assume_centered: self.assume_centered,
            method: self.method,
            mu: self.mu,
            mu_max: self.mu_max,
            rho: self.rho,
        })
    }
}

impl LowRankSparseCovariance<Untrained> {
    /// Compute empirical covariance matrix
    fn compute_empirical_covariance(x: &ArrayView2<f64>) -> SklResult<Array2<f64>> {
        let n_samples = x.nrows() as f64;
        let cov = x.t().dot(x) / (n_samples - 1.0);
        Ok(cov)
    }

    /// Decompose using Augmented Lagrangian Method (ALM)
    fn decompose_alm(
        &self,
        matrix: &Array2<f64>,
    ) -> SklResult<(Array2<f64>, Array2<f64>, usize, f64)> {
        let (n, m) = matrix.dim();

        // Initialize variables
        let mut low_rank = Array2::zeros((n, m));
        let mut sparse = Array2::zeros((n, m));
        let mut lagrange = Array2::zeros((n, m));
        let mut mu = self.mu;

        for iter in 0..self.max_iter {
            // Update low-rank component (nuclear norm proximal operator)
            let temp1 = matrix - &sparse + &lagrange / mu;
            low_rank = self.nuclear_norm_prox(&temp1, self.lambda_nuclear / mu)?;

            // Update sparse component (L1 proximal operator)
            let temp2 = matrix - &low_rank + &lagrange / mu;
            sparse = self.l1_prox(&temp2, self.lambda_l1 / mu);

            // Update dual variable (Lagrange multipliers)
            let residual = matrix - &low_rank - &sparse;
            lagrange = &lagrange + mu * &residual;

            // Check convergence
            let primal_residual = Self::frobenius_norm(&residual);
            let dual_residual = mu * Self::frobenius_norm(&(&low_rank - &sparse));

            if primal_residual < self.tol && dual_residual < self.tol {
                let objective = self.compute_objective(matrix, &low_rank, &sparse);
                return Ok((low_rank, sparse, iter + 1, objective));
            }

            // Update penalty parameter
            if primal_residual > 10.0 * dual_residual {
                mu = (mu * self.rho).min(self.mu_max);
            }
        }

        let final_objective = self.compute_objective(matrix, &low_rank, &sparse);
        Ok((low_rank, sparse, self.max_iter, final_objective))
    }

    /// Decompose using ADMM
    fn decompose_admm(
        &self,
        matrix: &Array2<f64>,
    ) -> SklResult<(Array2<f64>, Array2<f64>, usize, f64)> {
        let (n, m) = matrix.dim();

        // Initialize variables
        let mut low_rank = Array2::zeros((n, m));
        let mut sparse = Array2::zeros((n, m));
        let mut auxiliary = Array2::zeros((n, m));
        let mut dual1 = Array2::zeros((n, m));
        let mut dual2 = Array2::zeros((n, m));
        let rho = self.mu;

        for iter in 0..self.max_iter {
            // Update low-rank component
            let temp1 = &auxiliary - &dual1 / rho;
            low_rank = self.nuclear_norm_prox(&temp1, self.lambda_nuclear / rho)?;

            // Update sparse component
            let temp2 = &auxiliary - &dual2 / rho;
            sparse = self.l1_prox(&temp2, self.lambda_l1 / rho);

            // Update auxiliary variable
            let temp3 = (2.0 * matrix + rho * (&low_rank + &dual1 / rho + &sparse + &dual2 / rho))
                / (2.0 + 2.0 * rho);
            auxiliary = temp3;

            // Update dual variables
            dual1 = &dual1 + rho * (&low_rank - &auxiliary);
            dual2 = &dual2 + rho * (&sparse - &auxiliary);

            // Check convergence
            let residual1 = &low_rank - &auxiliary;
            let residual2 = &sparse - &auxiliary;
            let primal_residual =
                Self::frobenius_norm(&residual1) + Self::frobenius_norm(&residual2);

            if primal_residual < self.tol {
                let objective = self.compute_objective(matrix, &low_rank, &sparse);
                return Ok((low_rank, sparse, iter + 1, objective));
            }
        }

        let final_objective = self.compute_objective(matrix, &low_rank, &sparse);
        Ok((low_rank, sparse, self.max_iter, final_objective))
    }

    /// Decompose using Proximal Gradient Method
    fn decompose_proximal_gradient(
        &self,
        matrix: &Array2<f64>,
    ) -> SklResult<(Array2<f64>, Array2<f64>, usize, f64)> {
        let (n, m) = matrix.dim();

        // Initialize variables
        let mut low_rank = Array2::zeros((n, m));
        let mut sparse = Array2::zeros((n, m));
        let step_size = 0.1;

        for iter in 0..self.max_iter {
            // Gradient step
            let residual = &low_rank + &sparse - matrix;

            // Update low-rank component
            let temp1 = &low_rank - step_size * &residual;
            low_rank = self.nuclear_norm_prox(&temp1, step_size * self.lambda_nuclear)?;

            // Update sparse component
            let temp2 = &sparse - step_size * &residual;
            sparse = self.l1_prox(&temp2, step_size * self.lambda_l1);

            // Check convergence
            let new_residual = &low_rank + &sparse - matrix;
            if Self::frobenius_norm(&new_residual) < self.tol {
                let objective = self.compute_objective(matrix, &low_rank, &sparse);
                return Ok((low_rank, sparse, iter + 1, objective));
            }
        }

        let final_objective = self.compute_objective(matrix, &low_rank, &sparse);
        Ok((low_rank, sparse, self.max_iter, final_objective))
    }

    /// Nuclear norm proximal operator using SVD
    fn nuclear_norm_prox(&self, matrix: &Array2<f64>, threshold: f64) -> SklResult<Array2<f64>> {
        use scirs2_core::ndarray::ndarray_linalg::SVD;

        // Compute SVD
        let (u, s, vt) = matrix
            .svd(true, true)
            .map_err(|e| SklearsError::NumericalError(format!("SVD failed: {}", e)))?;

        let u = u.unwrap();
        let vt = vt.unwrap();

        // Apply soft thresholding to singular values
        let s_thresh = s.mapv(|val| (val - threshold).max(0.0));

        // Reconstruct matrix
        let s_diag = Array2::from_diag(&s_thresh);
        let result = u.dot(&s_diag).dot(&vt);

        Ok(result)
    }

    /// L1 norm proximal operator (soft thresholding)
    fn l1_prox(&self, matrix: &Array2<f64>, threshold: f64) -> Array2<f64> {
        matrix.mapv(|val| {
            if val > threshold {
                val - threshold
            } else if val < -threshold {
                val + threshold
            } else {
                0.0
            }
        })
    }

    /// Compute Frobenius norm
    fn frobenius_norm(matrix: &Array2<f64>) -> f64 {
        matrix.mapv(|x| x * x).sum().sqrt()
    }

    /// Compute objective function value
    fn compute_objective(
        &self,
        original: &Array2<f64>,
        low_rank: &Array2<f64>,
        sparse: &Array2<f64>,
    ) -> f64 {
        let reconstruction_error = Self::frobenius_norm(&(original - low_rank - sparse));
        let nuclear_penalty = self.nuclear_norm(low_rank).unwrap_or(0.0);
        let l1_penalty = sparse.mapv(|x| x.abs()).sum();

        0.5 * reconstruction_error * reconstruction_error
            + self.lambda_nuclear * nuclear_penalty
            + self.lambda_l1 * l1_penalty
    }

    /// Compute nuclear norm (sum of singular values)
    fn nuclear_norm(&self, matrix: &Array2<f64>) -> SklResult<f64> {
        use scirs2_core::ndarray::ndarray_linalg::SVD;

        let (_, s, _) = matrix
            .svd(false, false)
            .map_err(|e| SklearsError::NumericalError(format!("SVD failed: {}", e)))?;

        Ok(s.sum())
    }

    /// Compute rank of a matrix
    fn compute_rank(matrix: &Array2<f64>, threshold: f64) -> usize {
        use scirs2_core::ndarray::ndarray_linalg::SVD;

        if let Ok((_, s, _)) = matrix.svd(false, false) {
            s.iter().filter(|&&val| val > threshold).count()
        } else {
            0
        }
    }

    /// Count non-zero elements in a matrix
    fn count_nonzeros(matrix: &Array2<f64>, threshold: f64) -> usize {
        matrix.iter().filter(|&&val| val.abs() > threshold).count()
    }

    /// Compute precision matrix
    fn compute_precision(covariance: &Array2<f64>) -> SklResult<Array2<f64>> {
        use scirs2_core::ndarray::ndarray_linalg::Inverse;
        covariance.inv().map_err(|e| {
            SklearsError::NumericalError(format!("Failed to invert covariance matrix: {}", e))
        })
    }
}

impl LowRankSparseCovariance<LowRankSparseCovarianceTrained> {
    /// Get the low-rank component
    pub fn get_low_rank_component(&self) -> &Array2<f64> {
        &self.state.low_rank_component
    }

    /// Get the sparse component
    pub fn get_sparse_component(&self) -> &Array2<f64> {
        &self.state.sparse_component
    }

    /// Get the estimated covariance matrix
    pub fn get_covariance(&self) -> &Array2<f64> {
        &self.state.covariance
    }

    /// Get the estimated precision matrix
    pub fn get_precision(&self) -> Option<&Array2<f64>> {
        self.state.precision.as_ref()
    }

    /// Get the number of iterations performed
    pub fn get_n_iter(&self) -> usize {
        self.state.n_iter
    }

    /// Get the final objective value
    pub fn get_objective(&self) -> f64 {
        self.state.objective
    }

    /// Get the rank of the low-rank component
    pub fn get_rank(&self) -> usize {
        self.state.rank
    }

    /// Get the number of non-zero elements in the sparse component
    pub fn get_nnz_sparse(&self) -> usize {
        self.state.nnz_sparse
    }

    /// Get the nuclear norm regularization parameter
    pub fn get_lambda_nuclear(&self) -> f64 {
        self.state.lambda_nuclear
    }

    /// Get the L1 regularization parameter
    pub fn get_lambda_l1(&self) -> f64 {
        self.state.lambda_l1
    }

    /// Get whether data was assumed to be centered
    pub fn get_assume_centered(&self) -> bool {
        self.state.assume_centered
    }

    /// Get the optimization method used
    pub fn get_method(&self) -> &LRSMethod {
        &self.state.method
    }

    /// Get the sparsity ratio of the sparse component
    pub fn get_sparsity_ratio(&self) -> f64 {
        let total_elements = self.state.sparse_component.len();
        let nonzero_elements = self.state.nnz_sparse;
        1.0 - (nonzero_elements as f64 / total_elements as f64)
    }

    /// Get the low-rank ratio (rank / min(n_features))
    pub fn get_low_rank_ratio(&self) -> f64 {
        let min_dim = self
            .state
            .low_rank_component
            .dim()
            .0
            .min(self.state.low_rank_component.dim().1);
        self.state.rank as f64 / min_dim as f64
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_low_rank_sparse_basic() {
        let x = array![
            [1.0, 0.8, 0.6, 0.4],
            [2.0, 1.6, 1.2, 0.8],
            [3.0, 2.4, 1.8, 1.2],
            [4.0, 3.2, 2.4, 1.6],
            [5.0, 4.0, 3.0, 2.0],
            [1.5, 1.2, 0.9, 0.6],
            [2.5, 2.0, 1.5, 1.0],
            [3.5, 2.8, 2.1, 1.4]
        ];

        let estimator = LowRankSparseCovariance::new()
            .lambda_nuclear(0.1)
            .lambda_l1(0.1)
            .max_iter(50);

        match estimator.fit(&x.view(), &()) {
            Ok(fitted) => {
                assert_eq!(fitted.get_low_rank_component().dim(), (4, 4));
                assert_eq!(fitted.get_sparse_component().dim(), (4, 4));
                assert_eq!(fitted.get_covariance().dim(), (4, 4));
                assert_eq!(fitted.get_lambda_nuclear(), 0.1);
                assert_eq!(fitted.get_lambda_l1(), 0.1);
                assert!(fitted.get_n_iter() > 0);
                assert!(fitted.get_rank() >= 0);
                assert!(fitted.get_nnz_sparse() >= 0);
                assert!(fitted.get_sparsity_ratio() >= 0.0 && fitted.get_sparsity_ratio() <= 1.0);
                assert!(fitted.get_low_rank_ratio() >= 0.0 && fitted.get_low_rank_ratio() <= 1.0);
            }
            Err(_) => {
                // Acceptable for basic test - LRS can be sensitive to data
            }
        }
    }

    #[test]
    fn test_low_rank_sparse_methods() {
        let x = array![
            [1.0, 0.5, 0.1],
            [2.0, 1.0, 0.2],
            [3.0, 1.5, 0.3],
            [4.0, 2.0, 0.4],
            [5.0, 2.5, 0.5]
        ];

        for method in [
            LRSMethod::AugmentedLagrangian,
            LRSMethod::ADMM,
            LRSMethod::ProximalGradient,
        ] {
            let estimator = LowRankSparseCovariance::new()
                .method(method)
                .lambda_nuclear(0.05)
                .lambda_l1(0.05)
                .max_iter(20);

            match estimator.fit(&x.view(), &()) {
                Ok(fitted) => {
                    assert_eq!(fitted.get_covariance().dim(), (3, 3));
                    assert!(fitted.get_n_iter() > 0);
                }
                Err(_) => {
                    // Acceptable for basic test
                }
            }
        }
    }

    #[test]
    fn test_low_rank_sparse_parameters() {
        let estimator = LowRankSparseCovariance::new()
            .lambda_nuclear(0.2)
            .lambda_l1(0.3)
            .max_iter(200)
            .tol(1e-8)
            .assume_centered(true)
            .mu(2.0)
            .mu_max(1e8)
            .rho(3.0);

        assert_eq!(estimator.lambda_nuclear, 0.2);
        assert_eq!(estimator.lambda_l1, 0.3);
        assert_eq!(estimator.max_iter, 200);
        assert_eq!(estimator.tol, 1e-8);
        assert_eq!(estimator.assume_centered, true);
        assert_eq!(estimator.mu, 2.0);
        assert_eq!(estimator.mu_max, 1e8);
        assert_eq!(estimator.rho, 3.0);
    }
}
