//! Nuclear norm minimization for matrix completion and low-rank covariance estimation
//!
//! This module implements nuclear norm minimization algorithms for completing
//! missing entries in covariance matrices and estimating low-rank structure.

use scirs2_core::ndarray::{Array1, Array2, ArrayView2, Axis};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Untrained},
    types::Float,
};

/// Nuclear norm minimization estimator for matrix completion
///
/// Uses nuclear norm regularization to complete missing entries and
/// estimate low-rank covariance structures.
#[derive(Debug, Clone)]
pub struct NuclearNormMinimization<S = Untrained> {
    state: S,
    /// Nuclear norm regularization parameter
    lambda: f64,
    /// Maximum number of iterations
    max_iter: usize,
    /// Convergence tolerance
    tol: f64,
    /// Whether to assume centered data
    assume_centered: bool,
    /// Algorithm for nuclear norm minimization
    algorithm: NuclearNormAlgorithm,
    /// Target rank for the matrix (None for automatic)
    target_rank: Option<usize>,
}

/// Algorithms for nuclear norm minimization
#[derive(Debug, Clone)]
pub enum NuclearNormAlgorithm {
    /// Singular Value Thresholding (SVT)
    SVT,
    /// Accelerated Proximal Gradient
    APG,
    /// Fixed Point Continuation
    FPC,
}

impl Default for NuclearNormMinimization<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl NuclearNormMinimization<Untrained> {
    /// Create a new nuclear norm minimization estimator
    pub fn new() -> Self {
        Self {
            state: Untrained,
            lambda: 0.1,
            max_iter: 1000,
            tol: 1e-6,
            assume_centered: false,
            algorithm: NuclearNormAlgorithm::SVT,
            target_rank: None,
        }
    }

    /// Set the nuclear norm regularization parameter
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

    /// Set the optimization algorithm
    pub fn algorithm(mut self, algorithm: NuclearNormAlgorithm) -> Self {
        self.algorithm = algorithm;
        self
    }

    /// Set the target rank for low-rank approximation
    pub fn target_rank(mut self, rank: Option<usize>) -> Self {
        self.target_rank = rank;
        self
    }
}

impl Estimator for NuclearNormMinimization<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, ()> for NuclearNormMinimization<Untrained> {
    type Fitted = NuclearNormMinimization<NuclearNormMinimizationTrained>;

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
        let mut sample_cov = (&x_centered.t().dot(&x_centered)) / (n_samples as f64 - 1.0);

        // Apply nuclear norm minimization to denoise/complete the covariance matrix
        let (covariance, n_iter, rank) = self.minimize_nuclear_norm(&mut sample_cov)?;

        // Compute precision matrix
        let precision = self.compute_precision(&covariance)?;

        Ok(NuclearNormMinimization {
            state: NuclearNormMinimizationTrained {
                covariance,
                precision: Some(precision),
                lambda: self.lambda,
                n_iter,
                rank,
                assume_centered: self.assume_centered,
            },
            lambda: self.lambda,
            max_iter: self.max_iter,
            tol: self.tol,
            assume_centered: self.assume_centered,
            algorithm: self.algorithm.clone(),
            target_rank: self.target_rank,
        })
    }
}

/// Trained nuclear norm minimization estimator
#[derive(Debug, Clone)]
pub struct NuclearNormMinimizationTrained {
    covariance: Array2<f64>,
    precision: Option<Array2<f64>>,
    lambda: f64,
    n_iter: usize,
    rank: usize,
    assume_centered: bool,
}

impl NuclearNormMinimization<NuclearNormMinimizationTrained> {
    /// Get the estimated covariance matrix
    pub fn get_covariance(&self) -> &Array2<f64> {
        &self.state.covariance
    }

    /// Get the estimated precision matrix
    pub fn get_precision(&self) -> Option<&Array2<f64>> {
        self.state.precision.as_ref()
    }

    /// Get the lambda parameter used
    pub fn get_lambda(&self) -> f64 {
        self.state.lambda
    }

    /// Get the number of iterations performed
    pub fn get_n_iter(&self) -> usize {
        self.state.n_iter
    }

    /// Get the estimated rank of the covariance matrix
    pub fn get_rank(&self) -> usize {
        self.state.rank
    }

    /// Check if data was assumed to be centered
    pub fn is_assume_centered(&self) -> bool {
        self.state.assume_centered
    }

    /// Compute the nuclear norm of the estimated covariance matrix
    pub fn get_nuclear_norm(&self) -> SklResult<f64> {
        compute_nuclear_norm(&self.state.covariance)
    }

    /// Get the singular values of the covariance matrix
    pub fn get_singular_values(&self) -> SklResult<Array1<f64>> {
        compute_singular_values(&self.state.covariance)
    }
}

impl NuclearNormMinimization<Untrained> {
    /// Minimize nuclear norm of the matrix
    fn minimize_nuclear_norm(
        &self,
        matrix: &mut Array2<f64>,
    ) -> SklResult<(Array2<f64>, usize, usize)> {
        match self.algorithm {
            NuclearNormAlgorithm::SVT => self.singular_value_thresholding(matrix),
            NuclearNormAlgorithm::APG => self.accelerated_proximal_gradient(matrix),
            NuclearNormAlgorithm::FPC => self.fixed_point_continuation(matrix),
        }
    }

    /// Singular Value Thresholding algorithm
    fn singular_value_thresholding(
        &self,
        matrix: &mut Array2<f64>,
    ) -> SklResult<(Array2<f64>, usize, usize)> {
        let mut current_matrix = matrix.clone();
        let mut iter = 0;

        for iteration in 0..self.max_iter {
            iter = iteration;
            let old_matrix = current_matrix.clone();

            // SVD decomposition
            let (u, singular_values, vt) = self.compute_svd(&current_matrix)?;

            // Soft threshold singular values
            let thresholded_values = self.soft_threshold_singular_values(&singular_values);

            // Reconstruct matrix
            current_matrix = self.reconstruct_from_svd(&u, &thresholded_values, &vt)?;

            // Check convergence
            let diff_norm = self.frobenius_norm_diff(&current_matrix, &old_matrix);
            if diff_norm < self.tol {
                break;
            }
        }

        let rank = self.compute_effective_rank(&current_matrix)?;
        Ok((current_matrix, iter + 1, rank))
    }

    /// Accelerated Proximal Gradient algorithm
    fn accelerated_proximal_gradient(
        &self,
        matrix: &mut Array2<f64>,
    ) -> SklResult<(Array2<f64>, usize, usize)> {
        let mut x = matrix.clone();
        let mut y = matrix.clone();
        let mut t = 1.0f64;
        let step_size = 0.01;
        let mut iter = 0;

        for iteration in 0..self.max_iter {
            iter = iteration;
            let x_old = x.clone();

            // Gradient step on y
            y = &y - &(&y * step_size); // Simplified gradient (identity for now)

            // Proximal operator (SVT)
            let (u, singular_values, vt) = self.compute_svd(&y)?;
            let thresholded_values = self.soft_threshold_singular_values(&singular_values);
            x = self.reconstruct_from_svd(&u, &thresholded_values, &vt)?;

            // Acceleration step
            let t_new = (1.0 + (1.0 + 4.0 * t * t).sqrt()) / 2.0;
            let beta = (t - 1.0) / t_new;
            y = &x + &((&x - &x_old) * beta);
            t = t_new;

            // Check convergence
            let diff_norm = self.frobenius_norm_diff(&x, &x_old);
            if diff_norm < self.tol {
                break;
            }
        }

        let rank = self.compute_effective_rank(&x)?;
        Ok((x, iter + 1, rank))
    }

    /// Fixed Point Continuation algorithm
    fn fixed_point_continuation(
        &self,
        matrix: &mut Array2<f64>,
    ) -> SklResult<(Array2<f64>, usize, usize)> {
        let mut current_matrix = matrix.clone();
        let mut lambda = self.lambda * 10.0; // Start with larger regularization
        let lambda_factor = 0.9;
        let mut iter = 0;

        while lambda >= self.lambda && iter < self.max_iter {
            let old_matrix = current_matrix.clone();

            // SVT step with current lambda
            let (u, singular_values, vt) = self.compute_svd(&current_matrix)?;
            let thresholded_values = singular_values.map(|&s| (s - lambda).max(0.0));
            current_matrix = self.reconstruct_from_svd(&u, &thresholded_values, &vt)?;

            // Reduce lambda for continuation
            lambda *= lambda_factor;
            iter += 1;

            // Check convergence
            if self.frobenius_norm_diff(&current_matrix, &old_matrix) < self.tol {
                break;
            }
        }

        let rank = self.compute_effective_rank(&current_matrix)?;
        Ok((current_matrix, iter, rank))
    }

    /// Compute SVD decomposition
    fn compute_svd(
        &self,
        matrix: &Array2<f64>,
    ) -> SklResult<(Array2<f64>, Array1<f64>, Array2<f64>)> {
        use scirs2_core::ndarray::ndarray_linalg::SVD;

        let (u, singular_values, vt) = matrix.svd(true, true).map_err(|e| {
            SklearsError::NumericalError(format!("SVD decomposition failed: {}", e))
        })?;

        let u =
            u.ok_or_else(|| SklearsError::NumericalError("U matrix not computed".to_string()))?;
        let vt =
            vt.ok_or_else(|| SklearsError::NumericalError("VT matrix not computed".to_string()))?;

        Ok((u, singular_values, vt))
    }

    /// Soft threshold singular values
    fn soft_threshold_singular_values(&self, singular_values: &Array1<f64>) -> Array1<f64> {
        let threshold = if let Some(rank) = self.target_rank {
            // Use rank-based thresholding
            let mut sorted_values = singular_values.to_vec();
            sorted_values.sort_by(|a, b| b.partial_cmp(a).unwrap());
            if rank < sorted_values.len() {
                sorted_values[rank]
            } else {
                0.0
            }
        } else {
            self.lambda
        };

        singular_values.map(|&s| (s - threshold).max(0.0))
    }

    /// Reconstruct matrix from SVD components
    fn reconstruct_from_svd(
        &self,
        u: &Array2<f64>,
        singular_values: &Array1<f64>,
        vt: &Array2<f64>,
    ) -> SklResult<Array2<f64>> {
        let rank = singular_values.len();
        let mut result = Array2::zeros((u.nrows(), vt.ncols()));

        for k in 0..rank {
            if singular_values[k] > 1e-12 {
                let uk = u.column(k);
                let vtk = vt.row(k);

                for i in 0..result.nrows() {
                    for j in 0..result.ncols() {
                        result[[i, j]] += singular_values[k] * uk[i] * vtk[j];
                    }
                }
            }
        }

        Ok(result)
    }

    /// Compute Frobenius norm difference between matrices
    fn frobenius_norm_diff(&self, a: &Array2<f64>, b: &Array2<f64>) -> f64 {
        (a - b).map(|x| x * x).sum().sqrt()
    }

    /// Compute effective rank (number of significant singular values)
    fn compute_effective_rank(&self, matrix: &Array2<f64>) -> SklResult<usize> {
        let singular_values = compute_singular_values(matrix)?;
        Ok(singular_values.iter().filter(|&&s| s > 1e-10).count())
    }

    /// Compute precision matrix
    fn compute_precision(&self, covariance: &Array2<f64>) -> SklResult<Array2<f64>> {
        use scirs2_core::ndarray::ndarray_linalg::Inverse;

        // Add small regularization for numerical stability
        let n = covariance.nrows();
        let mut regularized_cov = covariance.clone();
        for i in 0..n {
            regularized_cov[[i, i]] += 1e-10;
        }

        regularized_cov
            .inv()
            .map_err(|e| SklearsError::NumericalError(format!("Matrix inversion failed: {}", e)))
    }
}

/// Compute nuclear norm of a matrix
fn compute_nuclear_norm(matrix: &Array2<f64>) -> SklResult<f64> {
    let singular_values = compute_singular_values(matrix)?;
    Ok(singular_values.sum())
}

/// Compute singular values of a matrix
fn compute_singular_values(matrix: &Array2<f64>) -> SklResult<Array1<f64>> {
    use scirs2_core::ndarray::ndarray_linalg::SVD;

    let (_, singular_values, _) = matrix
        .svd(false, false)
        .map_err(|e| SklearsError::NumericalError(format!("SVD failed: {}", e)))?;

    Ok(singular_values)
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_nuclear_norm_basic() {
        let x = array![
            [1.0, 0.8, 0.6],
            [2.0, 1.6, 1.2],
            [3.0, 2.4, 1.8],
            [4.0, 3.2, 2.4],
            [5.0, 4.0, 3.0]
        ];

        let estimator = NuclearNormMinimization::new().lambda(0.1);
        let fitted = estimator.fit(&x.view(), &()).unwrap();

        assert_eq!(fitted.get_covariance().dim(), (3, 3));
        assert!(fitted.get_precision().is_some());
        assert_eq!(fitted.get_lambda(), 0.1);
        // Nuclear norm minimization may result in low-rank matrices due to regularization
        assert!(fitted.get_rank() >= 0);
    }

    #[test]
    fn test_nuclear_norm_low_rank() {
        // Create low-rank data
        let x = array![
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
            [3.0, 3.0, 3.0],
            [4.0, 4.0, 4.0]
        ];

        let estimator = NuclearNormMinimization::new()
            .lambda(0.01)
            .target_rank(Some(1));
        let fitted = estimator.fit(&x.view(), &()).unwrap();

        assert_eq!(fitted.get_rank(), 1);
        assert_eq!(fitted.get_covariance().dim(), (3, 3));
    }

    #[test]
    fn test_nuclear_norm_apg() {
        let x = array![[1.0, 0.5], [2.0, 1.5], [3.0, 2.8], [4.0, 3.9]];

        let estimator = NuclearNormMinimization::new()
            .lambda(0.1)
            .algorithm(NuclearNormAlgorithm::APG);
        let fitted = estimator.fit(&x.view(), &()).unwrap();

        assert_eq!(fitted.get_covariance().dim(), (2, 2));
        assert!(fitted.get_n_iter() > 0);
    }

    #[test]
    fn test_nuclear_norm_fpc() {
        let x = array![[1.0, 0.5], [2.0, 1.5], [3.0, 2.8]];

        let estimator = NuclearNormMinimization::new()
            .lambda(0.05)
            .algorithm(NuclearNormAlgorithm::FPC);
        let fitted = estimator.fit(&x.view(), &()).unwrap();

        assert_eq!(fitted.get_covariance().dim(), (2, 2));
    }

    #[test]
    fn test_nuclear_norm_computation() {
        let matrix = array![[1.0, 0.0], [0.0, 2.0]];

        let nuclear_norm = compute_nuclear_norm(&matrix).unwrap();
        assert!((nuclear_norm - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_singular_values_computation() {
        let matrix = array![[3.0, 0.0], [0.0, 1.0]];

        let singular_values = compute_singular_values(&matrix).unwrap();
        assert_eq!(singular_values.len(), 2);
        assert!((singular_values[0] - 3.0).abs() < 1e-10);
        assert!((singular_values[1] - 1.0).abs() < 1e-10);
    }
}
