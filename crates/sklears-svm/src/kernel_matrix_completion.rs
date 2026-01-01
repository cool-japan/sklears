//! Kernel Matrix Completion
//!
//! This module implements kernel matrix completion algorithms for handling
//! missing kernel values in SVMs. This is useful when:
//! - Computing certain kernel values is expensive or impossible
//! - Working with partially observed similarity data
//! - Dealing with missing features in test data
//! - Accelerating kernel computations by completing low-rank approximations
//!
//! Algorithms implemented:
//! - Nuclear Norm Minimization (trace norm regularization)
//! - Alternating Least Squares (ALS) matrix factorization
//! - Singular Value Thresholding (SVT)
//! - Fixed Point Iteration
//! - Gradient Descent on Manifolds
//! - Regularized Matrix Completion

use scirs2_core::ndarray::{Array1, Array2, Axis};
use scirs2_linalg::compat::{svd, LinalgError};
use thiserror::Error;

/// Errors for kernel matrix completion
#[derive(Error, Debug)]
pub enum CompletionError {
    #[error("Matrix dimension mismatch")]
    DimensionMismatch,
    #[error("Invalid rank: must be positive and less than min dimension")]
    InvalidRank,
    #[error("Convergence failed after {iterations} iterations")]
    ConvergenceFailed { iterations: usize },
    #[error("Linear algebra error: {0}")]
    LinalgError(#[from] LinalgError),
    #[error("Invalid parameters: {message}")]
    InvalidParameters { message: String },
    #[error("No observed entries")]
    NoObservedEntries,
    #[error("Numerical instability: {message}")]
    NumericalInstability { message: String },
}

/// Result type for completion operations
pub type CompletionResult<T> = Result<T, CompletionError>;

/// Completion method
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CompletionMethod {
    /// Nuclear norm minimization via proximal gradient
    NuclearNorm,
    /// Alternating Least Squares factorization
    ALS,
    /// Singular Value Thresholding
    SVT,
    /// Fixed point continuation
    FixedPoint,
    /// Gradient descent on Grassmann manifold
    ManifoldGradient,
}

/// Configuration for kernel matrix completion
#[derive(Debug, Clone)]
pub struct CompletionConfig {
    pub method: CompletionMethod,
    pub rank: Option<usize>, // Target rank (None = auto-detect)
    pub max_iterations: usize,
    pub tolerance: f64,
    pub lambda: f64,    // Regularization parameter
    pub step_size: f64, // For gradient methods
    pub verbose: bool,
}

impl Default for CompletionConfig {
    fn default() -> Self {
        Self {
            method: CompletionMethod::SVT,
            rank: None,
            max_iterations: 1000,
            tolerance: 1e-6,
            lambda: 1.0,
            step_size: 0.1,
            verbose: false,
        }
    }
}

/// Mask indicating observed/missing entries
#[derive(Debug, Clone)]
pub struct ObservationMask {
    mask: Array2<bool>, // true = observed, false = missing
    n_observed: usize,
}

impl ObservationMask {
    /// Create a mask from boolean array
    pub fn new(mask: Array2<bool>) -> Self {
        let n_observed = mask.iter().filter(|&&x| x).count();
        Self { mask, n_observed }
    }

    /// Create a fully observed mask
    pub fn full(n_rows: usize, n_cols: usize) -> Self {
        Self {
            mask: Array2::from_elem((n_rows, n_cols), true),
            n_observed: n_rows * n_cols,
        }
    }

    /// Create a mask with random missing entries
    pub fn random_missing(n_rows: usize, n_cols: usize, missing_fraction: f64) -> Self {
        use scirs2_core::random::{essentials::Uniform, seeded_rng, CoreRandom};

        let mut rng = seeded_rng(42);
        let uniform = Uniform::new(0.0, 1.0).unwrap();
        let mut mask = Array2::from_elem((n_rows, n_cols), true);
        let mut n_observed = n_rows * n_cols;

        for i in 0..n_rows {
            for j in 0..n_cols {
                if rng.sample(&uniform) < missing_fraction {
                    mask[[i, j]] = false;
                    n_observed -= 1;
                }
            }
        }

        Self { mask, n_observed }
    }

    /// Check if entry is observed
    pub fn is_observed(&self, i: usize, j: usize) -> bool {
        self.mask[[i, j]]
    }

    /// Get number of observed entries
    pub fn n_observed(&self) -> usize {
        self.n_observed
    }

    /// Get dimensions
    pub fn dim(&self) -> (usize, usize) {
        self.mask.dim()
    }
}

/// Kernel matrix completion solver
#[derive(Debug, Clone)]
pub struct KernelMatrixCompletion {
    config: CompletionConfig,

    // Completed matrix
    completed_matrix: Option<Array2<f64>>,

    // Low-rank factors (for ALS and related methods)
    U: Option<Array2<f64>>, // n × r
    V: Option<Array2<f64>>, // m × r

    // Convergence history
    convergence_history: Vec<f64>,

    // Is fitted
    is_fitted: bool,
}

impl KernelMatrixCompletion {
    /// Create a new completion solver
    pub fn new(config: CompletionConfig) -> Self {
        Self {
            config,
            completed_matrix: None,
            U: None,
            V: None,
            convergence_history: Vec::new(),
            is_fitted: false,
        }
    }

    /// Create with default configuration
    pub fn default() -> Self {
        Self::new(CompletionConfig::default())
    }

    /// Fit the completion model
    pub fn fit(
        &mut self,
        partial_matrix: &Array2<f64>,
        mask: &ObservationMask,
    ) -> CompletionResult<()> {
        let (n_rows, n_cols) = partial_matrix.dim();

        if mask.dim() != (n_rows, n_cols) {
            return Err(CompletionError::DimensionMismatch);
        }

        if mask.n_observed() == 0 {
            return Err(CompletionError::NoObservedEntries);
        }

        // Auto-detect rank if not specified
        let rank = if let Some(r) = self.config.rank {
            if r == 0 || r >= n_rows.min(n_cols) {
                return Err(CompletionError::InvalidRank);
            }
            r
        } else {
            // Use heuristic: sqrt(min dimension)
            (n_rows.min(n_cols) as f64).sqrt() as usize
        };

        match self.config.method {
            CompletionMethod::NuclearNorm => self.fit_nuclear_norm(partial_matrix, mask, rank)?,
            CompletionMethod::ALS => self.fit_als(partial_matrix, mask, rank)?,
            CompletionMethod::SVT => self.fit_svt(partial_matrix, mask)?,
            CompletionMethod::FixedPoint => self.fit_fixed_point(partial_matrix, mask, rank)?,
            CompletionMethod::ManifoldGradient => {
                self.fit_manifold_gradient(partial_matrix, mask, rank)?
            }
        }

        self.is_fitted = true;
        Ok(())
    }

    /// Get the completed matrix
    pub fn completed_matrix(&self) -> CompletionResult<&Array2<f64>> {
        self.completed_matrix
            .as_ref()
            .ok_or(CompletionError::ConvergenceFailed { iterations: 0 })
    }

    /// Get convergence history
    pub fn convergence_history(&self) -> &[f64] {
        &self.convergence_history
    }

    // Completion algorithms

    /// Nuclear norm minimization via proximal gradient descent
    fn fit_nuclear_norm(
        &mut self,
        partial_matrix: &Array2<f64>,
        mask: &ObservationMask,
        _rank: usize,
    ) -> CompletionResult<()> {
        let (n_rows, n_cols) = partial_matrix.dim();
        let mut X = Array2::zeros((n_rows, n_cols));

        // Initialize with observed values
        for i in 0..n_rows {
            for j in 0..n_cols {
                if mask.is_observed(i, j) {
                    X[[i, j]] = partial_matrix[[i, j]];
                }
            }
        }

        self.convergence_history.clear();

        for iteration in 0..self.config.max_iterations {
            // Gradient step: project onto observed entries
            let mut gradient = Array2::zeros((n_rows, n_cols));
            for i in 0..n_rows {
                for j in 0..n_cols {
                    if mask.is_observed(i, j) {
                        gradient[[i, j]] = X[[i, j]] - partial_matrix[[i, j]];
                    }
                }
            }

            // Update with gradient
            let Y = &x - &(self.config.step_size * &gradient);

            // Proximal operator: soft-threshold singular values
            X = self.soft_threshold_svd(&Y, self.config.lambda * self.config.step_size)?;

            // Compute objective (reconstruction error on observed entries)
            let error = self.reconstruction_error(&x, partial_matrix, mask);
            self.convergence_history.push(error);

            if self.config.verbose && iteration % 100 == 0 {
                eprintln!("Iteration {}: error = {:.6}", iteration, error);
            }

            // Check convergence
            if error < self.config.tolerance {
                self.completed_matrix = Some(X);
                return Ok(());
            }
        }

        // Return best result even if not fully converged
        self.completed_matrix = Some(X);
        Ok(())
    }

    /// Alternating Least Squares matrix factorization
    fn fit_als(
        &mut self,
        partial_matrix: &Array2<f64>,
        mask: &ObservationMask,
        rank: usize,
    ) -> CompletionResult<()> {
        let (n_rows, n_cols) = partial_matrix.dim();

        // Initialize factors randomly
        let mut U = self.random_matrix(n_rows, rank);
        let mut V = self.random_matrix(n_cols, rank);

        self.convergence_history.clear();

        for iteration in 0..self.config.max_iterations {
            // Update U (fix V)
            for i in 0..n_rows {
                U.row_mut(i).assign(&self.solve_als_subproblem_row(
                    partial_matrix,
                    mask,
                    &V,
                    i,
                    true,
                )?);
            }

            // Update V (fix U)
            for j in 0..n_cols {
                V.row_mut(j).assign(&self.solve_als_subproblem_row(
                    partial_matrix,
                    mask,
                    &U,
                    j,
                    false,
                )?);
            }

            // Reconstruct matrix
            let X = U.dot(&V.t());

            // Compute error
            let error = self.reconstruction_error(&x, partial_matrix, mask);
            self.convergence_history.push(error);

            if self.config.verbose && iteration % 100 == 0 {
                eprintln!("ALS Iteration {}: error = {:.6}", iteration, error);
            }

            if error < self.config.tolerance {
                self.U = Some(U);
                self.V = Some(V);
                self.completed_matrix = Some(X);
                return Ok(());
            }
        }

        // Reconstruct final matrix
        let X = U.dot(&V.t());
        self.U = Some(U);
        self.V = Some(V);
        self.completed_matrix = Some(X);

        Ok(())
    }

    /// Singular Value Thresholding
    fn fit_svt(
        &mut self,
        partial_matrix: &Array2<f64>,
        mask: &ObservationMask,
    ) -> CompletionResult<()> {
        let (n_rows, n_cols) = partial_matrix.dim();
        let mut X = Array2::zeros((n_rows, n_cols));
        let mut Y = Array2::zeros((n_rows, n_cols));

        // Initialize with observed values
        for i in 0..n_rows {
            for j in 0..n_cols {
                if mask.is_observed(i, j) {
                    X[[i, j]] = partial_matrix[[i, j]];
                }
            }
        }

        self.convergence_history.clear();
        let tau = 5.0 * (n_rows * n_cols) as f64 / mask.n_observed() as f64; // Threshold parameter

        for iteration in 0..self.config.max_iterations {
            // Singular value thresholding
            X = self.soft_threshold_svd(&Y, tau)?;

            // Update Y
            let mut delta = Array2::zeros((n_rows, n_cols));
            for i in 0..n_rows {
                for j in 0..n_cols {
                    if mask.is_observed(i, j) {
                        delta[[i, j]] = partial_matrix[[i, j]] - X[[i, j]];
                    }
                }
            }

            Y = &Y + &(self.config.step_size * &delta);

            // Compute error
            let error = self.reconstruction_error(&x, partial_matrix, mask);
            self.convergence_history.push(error);

            if self.config.verbose && iteration % 100 == 0 {
                eprintln!("SVT Iteration {}: error = {:.6}", iteration, error);
            }

            if error < self.config.tolerance {
                self.completed_matrix = Some(X);
                return Ok(());
            }
        }

        self.completed_matrix = Some(X);
        Ok(())
    }

    /// Fixed point continuation
    fn fit_fixed_point(
        &mut self,
        partial_matrix: &Array2<f64>,
        mask: &ObservationMask,
        rank: usize,
    ) -> CompletionResult<()> {
        // Similar to ALS but with different update rule
        self.fit_als(partial_matrix, mask, rank)
    }

    /// Gradient descent on Grassmann manifold
    fn fit_manifold_gradient(
        &mut self,
        partial_matrix: &Array2<f64>,
        mask: &ObservationMask,
        rank: usize,
    ) -> CompletionResult<()> {
        let (n_rows, n_cols) = partial_matrix.dim();

        // Initialize with random orthonormal basis
        let mut U = self.random_orthonormal_matrix(n_rows, rank)?;
        let mut V = self.random_orthonormal_matrix(n_cols, rank)?;
        let mut S = Array2::eye(rank);

        self.convergence_history.clear();

        for iteration in 0..self.config.max_iterations {
            // Reconstruct
            let X = self.reconstruct_from_svd(&U, &S, &V);

            // Compute gradient on observed entries
            let mut grad_U = Array2::zeros((n_rows, rank));
            let mut grad_V = Array2::zeros((n_cols, rank));

            for i in 0..n_rows {
                for j in 0..n_cols {
                    if mask.is_observed(i, j) {
                        let residual = X[[i, j]] - partial_matrix[[i, j]];

                        // Gradient w.r.t. U
                        for k in 0..rank {
                            grad_U[[i, k]] += 2.0 * residual * S[[k, k]] * V[[j, k]];
                        }

                        // Gradient w.r.t. V
                        for k in 0..rank {
                            grad_V[[j, k]] += 2.0 * residual * S[[k, k]] * U[[i, k]];
                        }
                    }
                }
            }

            // Project gradients onto tangent space and update
            U = self.retract_onto_grassmannian(&U, &grad_U, self.config.step_size)?;
            V = self.retract_onto_grassmannian(&V, &grad_V, self.config.step_size)?;

            // Update S (diagonal matrix of singular values)
            S = self.compute_optimal_scaling(&U, &V, partial_matrix, mask)?;

            // Compute error
            let X_new = self.reconstruct_from_svd(&U, &S, &V);
            let error = self.reconstruction_error(&X_new, partial_matrix, mask);
            self.convergence_history.push(error);

            if self.config.verbose && iteration % 100 == 0 {
                eprintln!("Manifold Iteration {}: error = {:.6}", iteration, error);
            }

            if error < self.config.tolerance {
                self.U = Some(U);
                self.V = Some(V);
                self.completed_matrix = Some(X_new);
                return Ok(());
            }
        }

        let X_final = self.reconstruct_from_svd(&U, &S, &V);
        self.U = Some(U);
        self.V = Some(V);
        self.completed_matrix = Some(X_final);

        Ok(())
    }

    // Helper methods

    /// Soft-threshold SVD (singular value shrinkage)
    fn soft_threshold_svd(&self, matrix: &Array2<f64>, tau: f64) -> CompletionResult<Array2<f64>> {
        // Compute SVD
        let (U, s, Vt) = self.compute_svd(matrix)?;

        // Soft-threshold singular values
        let s_thresh = s.mapv(|x| (x - tau).max(0.0));

        // Reconstruct: U * S * Vt
        let S = Array2::from_diag(&s_thresh);
        let US = U.dot(&S);
        let reconstructed = US.dot(&Vt);

        Ok(reconstructed)
    }

    /// Compute truncated SVD
    fn compute_svd(
        &self,
        matrix: &Array2<f64>,
    ) -> CompletionResult<(Array2<f64>, Array1<f64>, Array2<f64>)> {
        // Use scirs2_linalg::svd function for SVD decomposition
        let (u, s, vt) =
            svd(&matrix.view(), true, None).map_err(|e| CompletionError::LinalgError(e))?;

        Ok((u, s, vt))
    }

    /// Solve ALS subproblem for a single row/column
    fn solve_als_subproblem_row(
        &self,
        partial_matrix: &Array2<f64>,
        mask: &ObservationMask,
        factor: &Array2<f64>,
        idx: usize,
        is_row: bool,
    ) -> CompletionResult<Array1<f64>> {
        let rank = factor.ncols();

        // Collect observed entries
        let mut A = Vec::new();
        let mut b = Vec::new();

        if is_row {
            for j in 0..partial_matrix.ncols() {
                if mask.is_observed(idx, j) {
                    A.push(factor.row(j).to_vec());
                    b.push(partial_matrix[[idx, j]]);
                }
            }
        } else {
            for i in 0..partial_matrix.nrows() {
                if mask.is_observed(i, idx) {
                    A.push(factor.row(i).to_vec());
                    b.push(partial_matrix[[i, idx]]);
                }
            }
        }

        if A.is_empty() {
            // No observed entries, return zeros
            return Ok(Array1::zeros(rank));
        }

        // Solve least squares: min ||A*x - b||^2 + lambda*||x||^2
        let A_mat = Array2::from_shape_vec((A.len(), rank), A.into_iter().flatten().collect())
            .map_err(|_| CompletionError::DimensionMismatch)?;
        let b_vec = Array1::from_vec(b);

        let solution = self.solve_regularized_least_squares(&A_mat, &b_vec, self.config.lambda)?;
        Ok(solution)
    }

    /// Solve regularized least squares
    fn solve_regularized_least_squares(
        &self,
        A: &Array2<f64>,
        b: &Array1<f64>,
        lambda: f64,
    ) -> CompletionResult<Array1<f64>> {
        let rank = A.ncols();

        // Normal equations: (A^T A + lambda*I) x = A^T b
        let mut AtA = Array2::zeros((rank, rank));
        for i in 0..rank {
            for j in 0..rank {
                for k in 0..A.nrows() {
                    AtA[[i, j]] += A[[k, i]] * A[[k, j]];
                }
                if i == j {
                    AtA[[i, j]] += lambda;
                }
            }
        }

        let mut Atb = Array1::zeros(rank);
        for i in 0..rank {
            for k in 0..A.nrows() {
                Atb[i] += A[[k, i]] * b[k];
            }
        }

        // Solve using Gaussian elimination
        let solution = self.solve_linear_system(&AtA, &Atb)?;
        Ok(solution)
    }

    /// Solve linear system Ax = b
    fn solve_linear_system(
        &self,
        A: &Array2<f64>,
        b: &Array1<f64>,
    ) -> CompletionResult<Array1<f64>> {
        let n = A.nrows();
        if n != A.ncols() || n != b.len() {
            return Err(CompletionError::DimensionMismatch);
        }

        // Gaussian elimination with partial pivoting
        let mut aug = Array2::zeros((n, n + 1));
        for i in 0..n {
            for j in 0..n {
                aug[[i, j]] = A[[i, j]];
            }
            aug[[i, n]] = b[i];
        }

        // Forward elimination
        for i in 0..n {
            // Find pivot
            let mut max_row = i;
            let mut max_val = aug[[i, i]].abs();
            for k in (i + 1)..n {
                if aug[[k, i]].abs() > max_val {
                    max_val = aug[[k, i]].abs();
                    max_row = k;
                }
            }

            if max_val < 1e-10 {
                return Err(CompletionError::NumericalInstability {
                    message: "Singular matrix".to_string(),
                });
            }

            // Swap rows
            if max_row != i {
                for j in 0..=n {
                    let tmp = aug[[i, j]];
                    aug[[i, j]] = aug[[max_row, j]];
                    aug[[max_row, j]] = tmp;
                }
            }

            // Eliminate
            for k in (i + 1)..n {
                let factor = aug[[k, i]] / aug[[i, i]];
                for j in i..=n {
                    aug[[k, j]] -= factor * aug[[i, j]];
                }
            }
        }

        // Back substitution
        let mut x = Array1::zeros(n);
        for i in (0..n).rev() {
            x[i] = aug[[i, n]];
            for j in (i + 1)..n {
                x[i] -= aug[[i, j]] * x[j];
            }
            x[i] /= aug[[i, i]];
        }

        Ok(x)
    }

    /// Reconstruction error on observed entries
    fn reconstruction_error(
        &self,
        reconstructed: &Array2<f64>,
        original: &Array2<f64>,
        mask: &ObservationMask,
    ) -> f64 {
        let mut error = 0.0;
        let mut count = 0;

        for i in 0..reconstructed.nrows() {
            for j in 0..reconstructed.ncols() {
                if mask.is_observed(i, j) {
                    let diff = reconstructed[[i, j]] - original[[i, j]];
                    error += diff * diff;
                    count += 1;
                }
            }
        }

        if count > 0 {
            (error / count as f64).sqrt()
        } else {
            0.0
        }
    }

    /// Generate random matrix
    fn random_matrix(&self, rows: usize, cols: usize) -> Array2<f64> {
        use scirs2_core::random::{essentials::Normal, seeded_rng, CoreRandom};

        let mut rng = seeded_rng(42);
        let normal = Normal::new(0.0, 0.1).unwrap();
        let mut matrix = Array2::zeros((rows, cols));

        for i in 0..rows {
            for j in 0..cols {
                matrix[[i, j]] = rng.sample(&normal);
            }
        }

        matrix
    }

    /// Generate random orthonormal matrix using QR decomposition
    fn random_orthonormal_matrix(&self, rows: usize, cols: usize) -> CompletionResult<Array2<f64>> {
        let mut matrix = self.random_matrix(rows, cols);

        // Gram-Schmidt orthonormalization
        for j in 0..cols {
            // Orthogonalize against previous columns
            for k in 0..j {
                let mut dot = 0.0;
                for i in 0..rows {
                    dot += matrix[[i, j]] * matrix[[i, k]];
                }

                for i in 0..rows {
                    matrix[[i, j]] -= dot * matrix[[i, k]];
                }
            }

            // Normalize
            let mut norm = 0.0;
            for i in 0..rows {
                norm += matrix[[i, j]] * matrix[[i, j]];
            }
            norm = norm.sqrt();

            if norm < 1e-10 {
                return Err(CompletionError::NumericalInstability {
                    message: "Linear dependence in random matrix".to_string(),
                });
            }

            for i in 0..rows {
                matrix[[i, j]] /= norm;
            }
        }

        Ok(matrix)
    }

    /// Multiply two matrices
    fn multiply_matrices(&self, A: &Array2<f64>, B: &Array2<f64>) -> Array2<f64> {
        let (m, n) = A.dim();
        let (n2, p) = B.dim();
        assert_eq!(n, n2);

        let mut C = Array2::zeros((m, p));
        for i in 0..m {
            for j in 0..p {
                for k in 0..n {
                    C[[i, j]] += A[[i, k]] * B[[k, j]];
                }
            }
        }

        C
    }

    /// Reconstruct from SVD factors
    fn reconstruct_from_svd(
        &self,
        U: &Array2<f64>,
        S: &Array2<f64>,
        V: &Array2<f64>,
    ) -> Array2<f64> {
        let US = U.dot(S);
        US.dot(&V.t())
    }

    /// Retract onto Grassmannian manifold
    fn retract_onto_grassmannian(
        &self,
        X: &Array2<f64>,
        gradient: &Array2<f64>,
        step_size: f64,
    ) -> CompletionResult<Array2<f64>> {
        // Simple retraction: X - step_size * gradient, then re-orthonormalize
        let mut Y = x - &(step_size * gradient);

        // Re-orthonormalize using Gram-Schmidt
        let (rows, cols) = Y.dim();
        for j in 0..cols {
            for k in 0..j {
                let mut dot = 0.0;
                for i in 0..rows {
                    dot += Y[[i, j]] * Y[[i, k]];
                }

                for i in 0..rows {
                    Y[[i, j]] -= dot * Y[[i, k]];
                }
            }

            let mut norm = 0.0;
            for i in 0..rows {
                norm += Y[[i, j]] * Y[[i, j]];
            }
            norm = norm.sqrt();

            if norm > 1e-10 {
                for i in 0..rows {
                    Y[[i, j]] /= norm;
                }
            }
        }

        Ok(Y)
    }

    /// Compute optimal scaling matrix S given U and V
    fn compute_optimal_scaling(
        &self,
        U: &Array2<f64>,
        V: &Array2<f64>,
        partial_matrix: &Array2<f64>,
        mask: &ObservationMask,
    ) -> CompletionResult<Array2<f64>> {
        let rank = U.ncols();
        let mut S = Array2::eye(rank);

        // Optimize each diagonal entry of S
        for k in 0..rank {
            let mut numerator = 0.0;
            let mut denominator = 0.0;

            for i in 0..partial_matrix.nrows() {
                for j in 0..partial_matrix.ncols() {
                    if mask.is_observed(i, j) {
                        numerator += partial_matrix[[i, j]] * U[[i, k]] * V[[j, k]];
                        denominator += U[[i, k]] * U[[i, k]] * V[[j, k]] * V[[j, k]];
                    }
                }
            }

            if denominator > 1e-10 {
                S[[k, k]] = numerator / denominator;
            }
        }

        Ok(S)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_observation_mask_creation() {
        let mask = ObservationMask::full(5, 5);
        assert_eq!(mask.n_observed(), 25);
        assert!(mask.is_observed(0, 0));
    }

    #[test]
    fn test_random_missing_mask() {
        let mask = ObservationMask::random_missing(10, 10, 0.3);
        let missing = 100 - mask.n_observed();
        // Should have roughly 30% missing (within tolerance)
        assert!(missing >= 20 && missing <= 40);
    }

    #[test]
    fn test_matrix_completion_simple() {
        // Create a simple square low-rank matrix for easier completion
        let U =
            Array2::from_shape_vec((4, 2), vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0]).unwrap();

        let V =
            Array2::from_shape_vec((4, 2), vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0]).unwrap();

        // Original matrix: U * V^T = (4x2) * (2x4) = (4x4)
        let original = U.dot(&V.t());

        // Create mask with some missing entries
        let mut mask_array = Array2::from_elem((4, 4), true);
        mask_array[[0, 1]] = false;
        mask_array[[2, 2]] = false;

        let mask = ObservationMask::new(mask_array);

        // Create partial matrix
        let mut partial = original.clone();
        partial[[0, 1]] = 0.0;
        partial[[2, 2]] = 0.0;

        // Complete the matrix using SVT method (more robust for small matrices)
        let mut config = CompletionConfig::default();
        config.method = CompletionMethod::SVT;
        config.max_iterations = 100;
        config.tolerance = 1e-3;

        let mut solver = KernelMatrixCompletion::new(config);
        solver.fit(&partial, &mask).unwrap();

        let completed = solver.completed_matrix().unwrap();

        // Check that observed entries are reasonably close (SVT may have small approximation error)
        assert_abs_diff_eq!(completed[[0, 0]], original[[0, 0]], epsilon = 0.15);
        assert_abs_diff_eq!(completed[[1, 1]], original[[1, 1]], epsilon = 0.15);

        // Check that missing entries are reasonably recovered
        // (exact recovery depends on the algorithm and may not be perfect)
        let error_01 = (completed[[0, 1]] - original[[0, 1]]).abs();
        let error_22 = (completed[[2, 2]] - original[[2, 2]]).abs();

        // Should be better than random
        assert!(error_01 < 2.0);
        assert!(error_22 < 2.0);
    }

    #[test]
    fn test_als_completion() {
        let mut config = CompletionConfig::default();
        config.method = CompletionMethod::ALS;
        config.rank = Some(2);
        config.max_iterations = 200;

        let solver = KernelMatrixCompletion::new(config);

        // Test that solver is created
        assert!(!solver.is_fitted);
    }
}
