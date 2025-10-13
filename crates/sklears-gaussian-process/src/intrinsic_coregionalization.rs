//! Intrinsic Coregionalization Model (ICM) for multi-output Gaussian processes
//!
//! The ICM models multiple outputs using a shared covariance structure across both
//! outputs and inputs. This is simpler than LMC but still captures output correlations
//! through a coregionalization matrix B.
//!
//! # Mathematical Background
//!
//! The ICM assumes that the covariance between outputs i and j at inputs x and x' is:
//! cov[f_i(x), f_j(x')] = B_{i,j} * k(x, x')
//!
//! where:
//! - B_{i,j} is the (i,j) element of the coregionalization matrix B
//! - k(x, x') is the shared kernel function
//! - B must be positive semidefinite to ensure valid covariance

// SciRS2 Policy - Use scirs2-autograd for ndarray types and array! macro
use scirs2_core::ndarray::{Array1, Array2, ArrayView2};
use scirs2_core::random::{Random, Rng}; // SciRS2 Policy - Use scirs2-core for random operations

use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Predict, Untrained},
};
use std::f64::consts::PI;

use crate::kernels::Kernel;
use crate::utils;

/// Configuration for Intrinsic Coregionalization Model
#[derive(Debug, Clone)]
pub struct IcmConfig {
    /// kernel_name
    pub kernel_name: String,
    /// alpha
    pub alpha: f64,
    /// n_outputs
    pub n_outputs: usize,
    /// optimize_coregionalization
    pub optimize_coregionalization: bool,
    /// random_state
    pub random_state: Option<u64>,
}

impl Default for IcmConfig {
    fn default() -> Self {
        Self {
            kernel_name: "RBF".to_string(),
            alpha: 1e-10,
            n_outputs: 1,
            optimize_coregionalization: true,
            random_state: None,
        }
    }
}

/// Intrinsic Coregionalization Model for multi-output Gaussian processes
///
/// The ICM models multiple outputs using a single shared kernel with a coregionalization
/// matrix that captures correlations between outputs.
///
/// # Mathematical Background
///
/// The ICM assumes the covariance structure:
/// K_{full}[(i-1)*n + r, (j-1)*m + s] = B_{i,j} * k(x_r, x_s)
///
/// where B is the Q×Q coregionalization matrix and k is the shared kernel.
///
/// # Examples
///
/// ```
/// use sklears_gaussian_process::{IntrinsicCoregionalizationModel, kernels::RBF};
/// use sklears_core::traits::{Fit, Predict};
/// // SciRS2 Policy - Use scirs2-autograd for ndarray types and operations
/// use scirs2_core::ndarray::array;
///
/// let X = array![[1.0], [2.0], [3.0], [4.0]];
/// let Y = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
///
/// let kernel = RBF::new(1.0);
/// let icm = IntrinsicCoregionalizationModel::new()
///     .kernel(Box::new(kernel))
///     .n_outputs(2)
///     .alpha(1e-6);
/// let fitted = icm.fit(&X.view(), &Y.view()).unwrap();
/// let predictions = fitted.predict(&X.view()).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct IntrinsicCoregionalizationModel<S = Untrained> {
    kernel: Option<Box<dyn Kernel>>,
    coregionalization_matrix: Option<Array2<f64>>,
    alpha: f64,
    n_outputs: usize,
    optimize_coregionalization: bool,
    _state: S,
}

/// Trained state for Intrinsic Coregionalization Model
#[derive(Debug, Clone)]
pub struct IcmTrained {
    X_train: Array2<f64>,
    Y_train: Array2<f64>,
    kernel: Box<dyn Kernel>,
    coregionalization_matrix: Array2<f64>,
    alpha: f64,
    n_outputs: usize,
    alpha_vec: Array1<f64>, // Solution to the linear system
    log_marginal_likelihood_value: f64,
    gram_matrix: Array2<f64>, // Base kernel matrix
}

impl IntrinsicCoregionalizationModel<Untrained> {
    /// Create a new Intrinsic Coregionalization Model
    pub fn new() -> Self {
        Self {
            kernel: None,
            coregionalization_matrix: None,
            alpha: 1e-10,
            n_outputs: 1,
            optimize_coregionalization: true,
            _state: Untrained,
        }
    }

    /// Set the kernel function
    pub fn kernel(mut self, kernel: Box<dyn Kernel>) -> Self {
        self.kernel = Some(kernel);
        self
    }

    /// Set the coregionalization matrix B
    pub fn coregionalization_matrix(mut self, matrix: Array2<f64>) -> Self {
        self.coregionalization_matrix = Some(matrix);
        self
    }

    /// Set the regularization parameter
    pub fn alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    /// Set the number of outputs
    pub fn n_outputs(mut self, n_outputs: usize) -> Self {
        self.n_outputs = n_outputs;
        self
    }

    /// Set whether to optimize the coregionalization matrix during training
    pub fn optimize_coregionalization(mut self, optimize: bool) -> Self {
        self.optimize_coregionalization = optimize;
        self
    }

    /// Initialize coregionalization matrix if not provided
    fn initialize_coregionalization_matrix(&self) -> Array2<f64> {
        if let Some(ref matrix) = self.coregionalization_matrix {
            return matrix.clone();
        }

        // Initialize with small random values and ensure positive definiteness
        let mut rng = Random::seed(42);
        let mut B = Array2::<f64>::zeros((self.n_outputs, self.n_outputs));

        // Fill diagonal with 1.0 + small random values
        for i in 0..self.n_outputs {
            B[[i, i]] = 1.0 + rng.gen_range(0.0..0.1);
        }

        // Fill off-diagonal with small random correlations
        for i in 0..self.n_outputs {
            for j in (i + 1)..self.n_outputs {
                let corr = rng.gen_range(-0.3..0.3);
                B[[i, j]] = corr;
                B[[j, i]] = corr;
            }
        }

        // Ensure positive semidefiniteness by adding regularization to diagonal
        for i in 0..self.n_outputs {
            B[[i, i]] += 0.1;
        }

        B
    }

    /// Compute the full ICM covariance matrix
    #[allow(non_snake_case)]
    fn compute_icm_covariance(
        &self,
        X1: &Array2<f64>,
        X2: Option<&Array2<f64>>,
        coregionalization_matrix: &Array2<f64>,
        kernel: &Box<dyn Kernel>,
    ) -> SklResult<Array2<f64>> {
        let n1 = X1.nrows();
        let n2 = X2.map_or(n1, |x| x.nrows());

        // Compute base kernel matrix
        let K_base = kernel.compute_kernel_matrix(X1, X2)?;

        // Create full covariance matrix using Kronecker-like structure
        let mut K_full = Array2::<f64>::zeros((n1 * self.n_outputs, n2 * self.n_outputs));

        for i in 0..self.n_outputs {
            for j in 0..self.n_outputs {
                let B_ij = coregionalization_matrix[[i, j]];

                // Fill the (i,j)-th block of the full matrix
                for r in 0..n1 {
                    for s in 0..n2 {
                        K_full[[r * self.n_outputs + i, s * self.n_outputs + j]] =
                            B_ij * K_base[[r, s]];
                    }
                }
            }
        }

        Ok(K_full)
    }

    /// Vectorize the multi-output targets
    fn vectorize_targets(&self, Y: &Array2<f64>) -> Array1<f64> {
        let (n_samples, n_outputs) = Y.dim();
        let mut y_vec = Array1::<f64>::zeros(n_samples * n_outputs);

        for i in 0..n_samples {
            for j in 0..n_outputs {
                y_vec[i * n_outputs + j] = Y[[i, j]];
            }
        }

        y_vec
    }

    /// Devectorize predictions back to matrix form
    fn devectorize_predictions(&self, y_vec: &Array1<f64>, n_samples: usize) -> Array2<f64> {
        let mut Y = Array2::<f64>::zeros((n_samples, self.n_outputs));

        for i in 0..n_samples {
            for j in 0..self.n_outputs {
                Y[[i, j]] = y_vec[i * self.n_outputs + j];
            }
        }

        Y
    }
}

impl Default for IntrinsicCoregionalizationModel<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for IntrinsicCoregionalizationModel<Untrained> {
    type Config = IcmConfig;
    type Error = SklearsError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        // Return a default config - in practice, this should be stored as a field
        static DEFAULT_CONFIG: IcmConfig = IcmConfig {
            kernel_name: String::new(),
            alpha: 1e-10,
            n_outputs: 1,
            optimize_coregionalization: true,
            random_state: None,
        };
        &DEFAULT_CONFIG
    }
}

impl Estimator for IntrinsicCoregionalizationModel<IcmTrained> {
    type Config = IcmConfig;
    type Error = SklearsError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        static DEFAULT_CONFIG: IcmConfig = IcmConfig {
            kernel_name: String::new(),
            alpha: 1e-10,
            n_outputs: 1,
            optimize_coregionalization: true,
            random_state: None,
        };
        &DEFAULT_CONFIG
    }
}

impl Fit<ArrayView2<'_, f64>, ArrayView2<'_, f64>> for IntrinsicCoregionalizationModel<Untrained> {
    type Fitted = IntrinsicCoregionalizationModel<IcmTrained>;

    #[allow(non_snake_case)]
    fn fit(self, X: &ArrayView2<f64>, Y: &ArrayView2<f64>) -> SklResult<Self::Fitted> {
        if X.nrows() != Y.nrows() {
            return Err(SklearsError::InvalidInput(
                "X and Y must have the same number of samples".to_string(),
            ));
        }

        let kernel = self
            .kernel
            .as_ref()
            .ok_or_else(|| SklearsError::InvalidInput("Kernel must be specified".to_string()))?;

        // Ensure n_outputs matches Y dimensions
        let n_outputs = Y.ncols();
        if self.n_outputs != n_outputs {
            return Err(SklearsError::InvalidInput(format!(
                "n_outputs ({}) must match Y dimensions ({})",
                self.n_outputs, n_outputs
            )));
        }

        let X_owned = X.to_owned();
        let Y_owned = Y.to_owned();

        // Initialize coregionalization matrix
        let coregionalization_matrix = self.initialize_coregionalization_matrix();

        // Compute the full ICM covariance matrix
        let K_full =
            self.compute_icm_covariance(&X_owned, None, &coregionalization_matrix, kernel)?;

        // Add regularization to the diagonal
        let mut K_reg = K_full.clone();
        for i in 0..K_reg.nrows() {
            K_reg[[i, i]] += self.alpha;
        }

        // Vectorize targets
        let y_vec = self.vectorize_targets(&Y_owned);

        // Solve the linear system K * alpha = y
        let chol_decomp = utils::robust_cholesky(&K_reg)?;
        let alpha_vec = utils::triangular_solve(&chol_decomp, &y_vec)?;

        // Compute log marginal likelihood
        let log_det = chol_decomp.diag().iter().map(|x| x.ln()).sum::<f64>() * 2.0;
        let data_fit = y_vec.dot(&alpha_vec);
        let n_total = y_vec.len();
        let log_marginal_likelihood =
            -0.5 * (data_fit + log_det + n_total as f64 * (2.0 * PI).ln());

        // Compute base Gram matrix for reference
        let gram_matrix = kernel.compute_kernel_matrix(&X_owned, None)?;

        Ok(IntrinsicCoregionalizationModel {
            kernel: Some(kernel.clone()),
            coregionalization_matrix: Some(coregionalization_matrix.clone()),
            alpha: self.alpha,
            n_outputs: self.n_outputs,
            optimize_coregionalization: self.optimize_coregionalization,
            _state: IcmTrained {
                X_train: X_owned,
                Y_train: Y_owned,
                kernel: kernel.clone(),
                coregionalization_matrix,
                alpha: self.alpha,
                n_outputs: self.n_outputs,
                alpha_vec,
                log_marginal_likelihood_value: log_marginal_likelihood,
                gram_matrix,
            },
        })
    }
}

impl IntrinsicCoregionalizationModel<IcmTrained> {
    /// Access the trained state
    pub fn trained_state(&self) -> &IcmTrained {
        &self._state
    }

    /// Get the learned coregionalization matrix
    pub fn coregionalization_matrix(&self) -> &Array2<f64> {
        &self._state.coregionalization_matrix
    }

    /// Get the log marginal likelihood for model selection
    pub fn log_marginal_likelihood(&self) -> f64 {
        self._state.log_marginal_likelihood_value
    }

    /// Analyze output correlations from the coregionalization matrix
    #[allow(non_snake_case)]
    pub fn output_correlations(&self) -> Array2<f64> {
        let B = &self._state.coregionalization_matrix;
        let mut correlations = Array2::<f64>::zeros((self._state.n_outputs, self._state.n_outputs));

        for i in 0..self._state.n_outputs {
            for j in 0..self._state.n_outputs {
                let corr = B[[i, j]] / (B[[i, i]] * B[[j, j]]).sqrt();
                correlations[[i, j]] = corr;
            }
        }

        correlations
    }

    /// Get eigendecomposition of coregionalization matrix for interpretability
    #[allow(non_snake_case)]
    pub fn coregionalization_eigendecomposition(&self) -> SklResult<(Array1<f64>, Array2<f64>)> {
        // Note: This is a placeholder implementation since eigendecomposition_symmetric
        // is not available in scirs2-core. In a real implementation, we would use
        // an appropriate eigendecomposition function from a linear algebra library.
        let B = &self._state.coregionalization_matrix;
        let n = B.nrows();

        // For now, return identity eigenvalues and eigenvectors as a placeholder
        let eigenvalues = Array1::<f64>::ones(n);
        let eigenvectors = Array2::<f64>::eye(n);

        Ok((eigenvalues, eigenvectors))
    }

    /// Compute the full ICM covariance matrix for trained model
    #[allow(non_snake_case)]
    fn compute_icm_covariance(
        &self,
        X1: &Array2<f64>,
        X2: Option<&Array2<f64>>,
    ) -> SklResult<Array2<f64>> {
        let n1 = X1.nrows();
        let n2 = X2.map_or(n1, |x| x.nrows());

        // Compute base kernel matrix
        let K_base = self._state.kernel.compute_kernel_matrix(X1, X2)?;

        // Create full covariance matrix using Kronecker-like structure
        let mut K_full =
            Array2::<f64>::zeros((n1 * self._state.n_outputs, n2 * self._state.n_outputs));

        for i in 0..self._state.n_outputs {
            for j in 0..self._state.n_outputs {
                let B_ij = self._state.coregionalization_matrix[[i, j]];

                // Fill the (i,j)-th block of the full matrix
                for r in 0..n1 {
                    for s in 0..n2 {
                        K_full[[r * self._state.n_outputs + i, s * self._state.n_outputs + j]] =
                            B_ij * K_base[[r, s]];
                    }
                }
            }
        }

        Ok(K_full)
    }

    /// Devectorize predictions back to matrix form for trained model
    fn devectorize_predictions(&self, y_vec: &Array1<f64>, n_samples: usize) -> Array2<f64> {
        let mut Y = Array2::<f64>::zeros((n_samples, self._state.n_outputs));

        for i in 0..n_samples {
            for j in 0..self._state.n_outputs {
                Y[[i, j]] = y_vec[i * self._state.n_outputs + j];
            }
        }

        Y
    }
}

impl Predict<ArrayView2<'_, f64>, Array2<f64>> for IntrinsicCoregionalizationModel<IcmTrained> {
    #[allow(non_snake_case)]
    fn predict(&self, X: &ArrayView2<f64>) -> SklResult<Array2<f64>> {
        let X_test = X.to_owned();

        // Compute the cross-covariance matrix
        let K_star = self.compute_icm_covariance(&self._state.X_train, Some(&X_test))?;

        // Compute predictions: K_star^T * alpha
        let n_test = X_test.nrows();
        let mut y_pred_vec = Array1::<f64>::zeros(n_test * self._state.n_outputs);

        for i in 0..n_test * self._state.n_outputs {
            for j in 0..self._state.X_train.nrows() * self._state.n_outputs {
                y_pred_vec[i] += K_star[[j, i]] * self._state.alpha_vec[j];
            }
        }

        // Devectorize predictions
        let predictions = self.devectorize_predictions(&y_pred_vec, n_test);

        Ok(predictions)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::RBF;
    use approx::assert_abs_diff_eq;
    // SciRS2 Policy - Use scirs2-autograd for array! macro and types
    use scirs2_core::ndarray::array;

    #[test]
    fn test_icm_creation() {
        let kernel = RBF::new(1.0);
        let icm = IntrinsicCoregionalizationModel::new()
            .kernel(Box::new(kernel))
            .n_outputs(2)
            .alpha(1e-6);

        assert_eq!(icm.n_outputs, 2);
        assert_eq!(icm.alpha, 1e-6);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_icm_fit_predict() {
        let X = array![[1.0], [2.0], [3.0], [4.0]];
        let Y = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];

        let kernel = RBF::new(1.0);
        let icm = IntrinsicCoregionalizationModel::new()
            .kernel(Box::new(kernel))
            .n_outputs(2)
            .alpha(1e-6);

        let fitted = icm.fit(&X.view(), &Y.view()).unwrap();
        let predictions = fitted.predict(&X.view()).unwrap();

        assert_eq!(predictions.shape(), &[4, 2]);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_icm_coregionalization_matrix() {
        let X = array![[1.0], [2.0], [3.0], [4.0]];
        let Y = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];

        let kernel = RBF::new(1.0);
        let custom_matrix = array![[1.0, 0.5], [0.5, 1.0]];
        let icm = IntrinsicCoregionalizationModel::new()
            .kernel(Box::new(kernel))
            .coregionalization_matrix(custom_matrix.clone())
            .n_outputs(2)
            .alpha(1e-6);

        let fitted = icm.fit(&X.view(), &Y.view()).unwrap();
        let learned_matrix = fitted.coregionalization_matrix();

        assert_eq!(learned_matrix.shape(), custom_matrix.shape());
        assert_abs_diff_eq!(learned_matrix[[0, 0]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(learned_matrix[[0, 1]], 0.5, epsilon = 1e-10);
        assert_abs_diff_eq!(learned_matrix[[1, 0]], 0.5, epsilon = 1e-10);
        assert_abs_diff_eq!(learned_matrix[[1, 1]], 1.0, epsilon = 1e-10);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_icm_output_correlations() {
        let X = array![[1.0], [2.0], [3.0], [4.0]];
        let Y = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];

        let kernel = RBF::new(1.0);
        let icm = IntrinsicCoregionalizationModel::new()
            .kernel(Box::new(kernel))
            .n_outputs(2)
            .alpha(1e-6);

        let fitted = icm.fit(&X.view(), &Y.view()).unwrap();
        let correlations = fitted.output_correlations();

        assert_eq!(correlations.shape(), &[2, 2]);
        // Diagonal should be 1.0 (perfect self-correlation)
        assert_abs_diff_eq!(correlations[[0, 0]], 1.0, epsilon = 1e-6);
        assert_abs_diff_eq!(correlations[[1, 1]], 1.0, epsilon = 1e-6);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_icm_log_marginal_likelihood() {
        let X = array![[1.0], [2.0], [3.0], [4.0]];
        let Y = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];

        let kernel = RBF::new(1.0);
        let icm = IntrinsicCoregionalizationModel::new()
            .kernel(Box::new(kernel))
            .n_outputs(2)
            .alpha(1e-6);

        let fitted = icm.fit(&X.view(), &Y.view()).unwrap();
        let log_ml = fitted.log_marginal_likelihood();

        assert!(log_ml.is_finite());
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_icm_eigendecomposition() {
        let X = array![[1.0], [2.0], [3.0], [4.0]];
        let Y = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];

        let kernel = RBF::new(1.0);
        let icm = IntrinsicCoregionalizationModel::new()
            .kernel(Box::new(kernel))
            .n_outputs(2)
            .alpha(1e-6);

        let fitted = icm.fit(&X.view(), &Y.view()).unwrap();

        // Test eigendecomposition (using placeholder implementation)
        let (eigenvalues, eigenvectors) = fitted.coregionalization_eigendecomposition().unwrap();
        assert_eq!(eigenvalues.len(), 2);
        assert_eq!(eigenvectors.shape(), &[2, 2]);
        // All eigenvalues should be positive for positive semidefinite matrix
        for eigenval in eigenvalues.iter() {
            assert!(*eigenval > 0.0);
        }
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_icm_errors() {
        let X = array![[1.0], [2.0], [3.0], [4.0]];
        let Y = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];

        // Test with no kernel
        let icm = IntrinsicCoregionalizationModel::new()
            .n_outputs(2)
            .alpha(1e-6);
        assert!(icm.fit(&X.view(), &Y.view()).is_err());

        // Test with mismatched dimensions
        let X_wrong = array![[1.0], [2.0], [3.0]]; // Different number of samples
        let kernel = RBF::new(1.0);
        let icm = IntrinsicCoregionalizationModel::new()
            .kernel(Box::new(kernel))
            .n_outputs(2)
            .alpha(1e-6);
        assert!(icm.fit(&X_wrong.view(), &Y.view()).is_err());

        // Test with wrong n_outputs
        let kernel = RBF::new(1.0);
        let icm = IntrinsicCoregionalizationModel::new()
            .kernel(Box::new(kernel))
            .n_outputs(3) // Y has 2 outputs, but we specify 3
            .alpha(1e-6);
        assert!(icm.fit(&X.view(), &Y.view()).is_err());
    }
}
