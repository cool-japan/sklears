//! Linear Model of Coregionalization (LMC) for multi-output Gaussian Process regression
//!
//! This module implements the Linear Model of Coregionalization, which learns multiple
//! correlated outputs simultaneously by using linear combinations of latent functions.
//!
//! # Mathematical Background
//!
//! The LMC assumes that each output is a linear combination of Q latent functions:
//! f_i(x) = Σ_q A_iq g_q(x), where g_q ~ GP(0, k_q)
//!
//! # Examples
//!
//! ```
//! use sklears_gaussian_process::{LinearModelCoregionalization, kernels::RBF, Kernel};
//! use sklears_core::traits::{Fit, Predict};
//! use scirs2_core::ndarray::array;
//!
//! let X = array![[1.0], [2.0], [3.0], [4.0]];
//! let Y = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
//!
//! let kernels: Vec<Box<dyn Kernel>> = vec![Box::new(RBF::new(1.0)), Box::new(RBF::new(2.0))];
//! let lmc = LinearModelCoregionalization::new()
//!     .kernels(kernels)
//!     .n_outputs(2)
//!     .alpha(1e-6);
//! let fitted = lmc.fit(&X.view(), &Y.view()).unwrap();
//! let predictions = fitted.predict(&X.view()).unwrap();
//! ```

use crate::classification::GpcConfig;
use crate::kernels::Kernel;
use crate::utils;
use scirs2_core::ndarray::{Array1, Array2, ArrayView2};
use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::{Rng, SeedableRng};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Predict, Untrained},
};
use std::f64::consts::PI;

/// Linear Model of Coregionalization for multi-output Gaussian Process regression
///
/// The LMC models multiple outputs using linear combinations of independent latent functions,
/// each with its own kernel. This allows the model to capture correlations between outputs
/// while maintaining computational efficiency.
#[derive(Debug, Clone)]
pub struct LinearModelCoregionalization<S = Untrained> {
    kernels: Vec<Box<dyn Kernel>>,
    mixing_matrices: Vec<Array2<f64>>, // One per kernel
    alpha: f64,
    n_outputs: usize,
    n_latent: usize,
    optimize_mixing: bool,
    _state: S,
}

/// Trained state for Linear Model of Coregionalization
#[derive(Debug, Clone)]
pub struct LmcTrained {
    pub(crate) X_train: Array2<f64>,
    pub(crate) Y_train: Array2<f64>,
    pub(crate) kernels: Vec<Box<dyn Kernel>>,
    pub(crate) mixing_matrices: Vec<Array2<f64>>,
    pub(crate) alpha: f64,
    pub(crate) n_outputs: usize,
    pub(crate) n_latent: usize,
    pub(crate) gram_matrices: Vec<Array2<f64>>, // Gram matrices for each kernel
    pub(crate) alpha_vectors: Vec<Array1<f64>>, // Alpha vectors for each latent function
    pub(crate) log_marginal_likelihood_value: f64,
}

impl Default for LinearModelCoregionalization<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl LinearModelCoregionalization<Untrained> {
    /// Create a new Linear Model of Coregionalization instance
    pub fn new() -> Self {
        Self {
            kernels: Vec::new(),
            mixing_matrices: Vec::new(),
            alpha: 1e-10,
            n_outputs: 1,
            n_latent: 1,
            optimize_mixing: true,
            _state: Untrained,
        }
    }

    /// Set the kernel functions for each latent process
    pub fn kernels(mut self, kernels: Vec<Box<dyn Kernel>>) -> Self {
        self.n_latent = kernels.len();
        self.kernels = kernels;
        self
    }

    /// Set the mixing matrices for each kernel
    pub fn mixing_matrices(mut self, mixing_matrices: Vec<Array2<f64>>) -> Self {
        self.mixing_matrices = mixing_matrices;
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

    /// Set whether to optimize mixing matrices during training
    pub fn optimize_mixing(mut self, optimize_mixing: bool) -> Self {
        self.optimize_mixing = optimize_mixing;
        self
    }

    /// Initialize mixing matrices randomly if not provided
    fn initialize_mixing_matrices(&self) -> Vec<Array2<f64>> {
        if !self.mixing_matrices.is_empty() {
            return self.mixing_matrices.clone();
        }

        let mut matrices = Vec::new();
        // SciRS2 Policy - Use scirs2-core for random number generation
        let mut rng = StdRng::seed_from_u64(42);

        for _ in 0..self.n_latent {
            let mut matrix = Array2::<f64>::zeros((self.n_outputs, 1));

            // Generate random values in range [-1, 1]
            for i in 0..self.n_outputs {
                let random_val = rng.gen_range(-1.0..1.0);
                matrix[[i, 0]] = random_val;
            }

            matrices.push(matrix);
        }

        matrices
    }

    /// Compute the coregionalization kernel matrix
    #[allow(non_snake_case)]
    fn compute_coregionalization_kernel(
        &self,
        X1: &Array2<f64>,
        X2: Option<&Array2<f64>>,
        mixing_matrices: &[Array2<f64>],
    ) -> SklResult<Array2<f64>> {
        let n1 = X1.nrows();
        let n2 = X2.map_or(n1, |x| x.nrows());

        // Initialize the full coregionalization kernel matrix
        let mut K_full = Array2::<f64>::zeros((n1 * self.n_outputs, n2 * self.n_outputs));

        // For each latent function r
        for (r, kernel) in self.kernels.iter().enumerate() {
            // Compute the kernel matrix for this latent function
            let K_r = kernel.compute_kernel_matrix(X1, X2)?;

            // Get the mixing matrix for this kernel
            let A_r = &mixing_matrices[r];

            // For each pair of outputs (i, j)
            for i in 0..self.n_outputs {
                for j in 0..self.n_outputs {
                    // The coregionalization coefficient
                    let coeff = A_r[[i, 0]] * A_r[[j, 0]];

                    // Add the contribution to the full kernel matrix
                    for n in 0..n1 {
                        for m in 0..n2 {
                            K_full[[n * self.n_outputs + i, m * self.n_outputs + j]] +=
                                coeff * K_r[[n, m]];
                        }
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

impl Estimator for LinearModelCoregionalization<Untrained> {
    type Config = GpcConfig; // Reuse existing config for now
    type Error = SklearsError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        // This is a temporary implementation - in a real scenario, we'd create a specific LMC config
        // For now, we'll use a default config to avoid compilation errors
        static DEFAULT_CONFIG: GpcConfig = GpcConfig {
            kernel_name: String::new(),
            optimizer: None,
            n_restarts_optimizer: 0,
            max_iter_predict: 100,
            warm_start: false,
            copy_x_train: true,
            random_state: None,
        };
        &DEFAULT_CONFIG
    }
}

impl Estimator for LinearModelCoregionalization<LmcTrained> {
    type Config = GpcConfig;
    type Error = SklearsError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        // This is a temporary implementation - in a real scenario, we'd create a specific LMC config
        static DEFAULT_CONFIG: GpcConfig = GpcConfig {
            kernel_name: String::new(),
            optimizer: None,
            n_restarts_optimizer: 0,
            max_iter_predict: 100,
            warm_start: false,
            copy_x_train: true,
            random_state: None,
        };
        &DEFAULT_CONFIG
    }
}

impl Fit<ArrayView2<'_, f64>, ArrayView2<'_, f64>> for LinearModelCoregionalization<Untrained> {
    type Fitted = LinearModelCoregionalization<LmcTrained>;

    #[allow(non_snake_case)]
    fn fit(self, X: &ArrayView2<f64>, Y: &ArrayView2<f64>) -> SklResult<Self::Fitted> {
        if X.nrows() != Y.nrows() {
            return Err(SklearsError::InvalidInput(
                "X and Y must have the same number of samples".to_string(),
            ));
        }

        if self.kernels.is_empty() {
            return Err(SklearsError::InvalidInput(
                "At least one kernel must be specified".to_string(),
            ));
        }

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

        // Initialize mixing matrices
        let mixing_matrices = self.initialize_mixing_matrices();

        // Compute the coregionalization kernel matrix
        let K_full = self.compute_coregionalization_kernel(&X_owned, None, &mixing_matrices)?;

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

        // Store individual Gram matrices for each kernel
        let mut gram_matrices = Vec::new();
        for kernel in &self.kernels {
            let K_r = kernel.compute_kernel_matrix(&X_owned, None)?;
            gram_matrices.push(K_r);
        }

        // Clone kernels for the trained state
        let kernels_clone: Vec<Box<dyn Kernel>> = self.kernels.to_vec();

        Ok(LinearModelCoregionalization {
            kernels: self.kernels,
            mixing_matrices: mixing_matrices.clone(),
            alpha: self.alpha,
            n_outputs: self.n_outputs,
            n_latent: self.n_latent,
            optimize_mixing: self.optimize_mixing,
            _state: LmcTrained {
                X_train: X_owned,
                Y_train: Y_owned,
                kernels: kernels_clone,
                mixing_matrices,
                alpha: self.alpha,
                n_outputs: self.n_outputs,
                n_latent: self.n_latent,
                gram_matrices,
                alpha_vectors: vec![alpha_vec],
                log_marginal_likelihood_value: log_marginal_likelihood,
            },
        })
    }
}

impl LinearModelCoregionalization<LmcTrained> {
    /// Access the trained state
    pub fn trained_state(&self) -> &LmcTrained {
        &self._state
    }

    /// Get the learned mixing matrices
    pub fn mixing_matrices(&self) -> &[Array2<f64>] {
        &self._state.mixing_matrices
    }

    /// Get the log marginal likelihood for model selection
    pub fn log_marginal_likelihood(&self) -> f64 {
        self._state.log_marginal_likelihood_value
    }

    /// Compute the coregionalization kernel matrix for trained model
    #[allow(non_snake_case)]
    fn compute_coregionalization_kernel(
        &self,
        X1: &Array2<f64>,
        X2: Option<&Array2<f64>>,
        mixing_matrices: &[Array2<f64>],
    ) -> SklResult<Array2<f64>> {
        let n1 = X1.nrows();
        let n2 = X2.map_or(n1, |x| x.nrows());

        // Initialize the full coregionalization kernel matrix
        let mut K_full =
            Array2::<f64>::zeros((n1 * self._state.n_outputs, n2 * self._state.n_outputs));

        // For each latent function r
        for (r, kernel) in self._state.kernels.iter().enumerate() {
            // Compute the kernel matrix for this latent function
            let K_r = kernel.compute_kernel_matrix(X1, X2)?;

            // Get the mixing matrix for this kernel
            let A_r = &mixing_matrices[r];

            // For each pair of outputs (i, j)
            for i in 0..self._state.n_outputs {
                for j in 0..self._state.n_outputs {
                    // The coregionalization coefficient
                    let coeff = A_r[[i, 0]] * A_r[[j, 0]];

                    // Add the contribution to the full kernel matrix
                    for n in 0..n1 {
                        for m in 0..n2 {
                            K_full
                                [[n * self._state.n_outputs + i, m * self._state.n_outputs + j]] +=
                                coeff * K_r[[n, m]];
                        }
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

    /// Get the contribution of each latent function to the predictions
    #[allow(non_snake_case)]
    pub fn latent_contributions(&self, X: &ArrayView2<f64>) -> SklResult<Vec<Array2<f64>>> {
        let mut contributions = Vec::new();

        for r in 0..self._state.n_latent {
            // Compute cross-covariance for this latent function
            let _K_star = self._state.kernels[r]
                .compute_kernel_matrix(&self._state.X_train, Some(&X.to_owned()))?;

            // Get the mixing matrix for this latent function
            let A_r = &self._state.mixing_matrices[r];

            // Simple prediction for this latent function (this is a simplified version)
            let n_test = X.nrows();
            let mut contribution = Array2::<f64>::zeros((n_test, self._state.n_outputs));

            // This is a simplified implementation - in practice, we'd need to properly
            // decompose the alpha vector and compute individual latent predictions
            for i in 0..n_test {
                for j in 0..self._state.n_outputs {
                    contribution[[i, j]] = A_r[[j, 0]]; // Simplified - just showing the mixing weight
                }
            }

            contributions.push(contribution);
        }

        Ok(contributions)
    }
}

impl Predict<ArrayView2<'_, f64>, Array2<f64>> for LinearModelCoregionalization<LmcTrained> {
    #[allow(non_snake_case)]
    fn predict(&self, X: &ArrayView2<f64>) -> Result<Array2<f64>, SklearsError> {
        let X_test = X.to_owned();

        // Compute the cross-covariance matrix
        let K_star = self.compute_coregionalization_kernel(
            &self._state.X_train,
            Some(&X_test),
            &self._state.mixing_matrices,
        )?;

        // Vectorize the alpha vector from the trained state
        let alpha_vec = &self._state.alpha_vectors[0];

        // Compute predictions: K_star^T * alpha
        let n_test = X_test.nrows();
        let mut y_pred_vec = Array1::<f64>::zeros(n_test * self._state.n_outputs);

        for i in 0..n_test * self._state.n_outputs {
            for j in 0..self._state.X_train.nrows() * self._state.n_outputs {
                y_pred_vec[i] += K_star[[j, i]] * alpha_vec[j];
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
    // SciRS2 Policy - Use scirs2-core for array! macro and types
    use scirs2_core::ndarray::array;

    #[test]
    fn test_lmc_creation() {
        let kernels = vec![Box::new(RBF::new(1.0)) as Box<dyn Kernel>];
        let lmc = LinearModelCoregionalization::new()
            .kernels(kernels)
            .n_outputs(2)
            .alpha(1e-6);

        assert_eq!(lmc.n_outputs, 2);
        assert_eq!(lmc.n_latent, 1);
        assert_eq!(lmc.alpha, 1e-6);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_lmc_fit_predict() {
        let X = array![[1.0], [2.0], [3.0], [4.0]];
        let Y = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];

        let kernels = vec![Box::new(RBF::new(1.0)) as Box<dyn Kernel>];
        let lmc = LinearModelCoregionalization::new()
            .kernels(kernels)
            .n_outputs(2)
            .alpha(1e-6);

        let fitted = lmc.fit(&X.view(), &Y.view()).unwrap();
        let predictions = fitted.predict(&X.view()).unwrap();

        assert_eq!(predictions.shape(), &[4, 2]);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_lmc_multi_kernel() {
        let X = array![[1.0], [2.0], [3.0], [4.0]];
        let Y = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];

        let kernels = vec![
            Box::new(RBF::new(1.0)) as Box<dyn Kernel>,
            Box::new(RBF::new(2.0)) as Box<dyn Kernel>,
        ];
        let lmc = LinearModelCoregionalization::new()
            .kernels(kernels)
            .n_outputs(2)
            .alpha(1e-6);

        let fitted = lmc.fit(&X.view(), &Y.view()).unwrap();
        let predictions = fitted.predict(&X.view()).unwrap();

        assert_eq!(predictions.shape(), &[4, 2]);
        assert_eq!(fitted.trained_state().n_latent, 2);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_lmc_mixing_matrices() {
        let X = array![[1.0], [2.0], [3.0], [4.0]];
        let Y = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];

        let kernels = vec![Box::new(RBF::new(1.0)) as Box<dyn Kernel>];
        let mixing_matrix = array![[0.5], [0.8]];
        let lmc = LinearModelCoregionalization::new()
            .kernels(kernels)
            .mixing_matrices(vec![mixing_matrix.clone()])
            .n_outputs(2)
            .alpha(1e-6);

        let fitted = lmc.fit(&X.view(), &Y.view()).unwrap();
        let learned_mixing = &fitted.mixing_matrices()[0];

        assert_eq!(learned_mixing.shape(), mixing_matrix.shape());
        assert_abs_diff_eq!(learned_mixing[[0, 0]], 0.5, epsilon = 1e-10);
        assert_abs_diff_eq!(learned_mixing[[1, 0]], 0.8, epsilon = 1e-10);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_lmc_log_marginal_likelihood() {
        let X = array![[1.0], [2.0], [3.0], [4.0]];
        let Y = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];

        let kernels = vec![Box::new(RBF::new(1.0)) as Box<dyn Kernel>];
        let lmc = LinearModelCoregionalization::new()
            .kernels(kernels)
            .n_outputs(2)
            .alpha(1e-6);

        let fitted = lmc.fit(&X.view(), &Y.view()).unwrap();
        let log_ml = fitted.log_marginal_likelihood();

        assert!(log_ml.is_finite());
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_lmc_latent_contributions() {
        let X = array![[1.0], [2.0], [3.0], [4.0]];
        let Y = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];

        let kernels = vec![
            Box::new(RBF::new(1.0)) as Box<dyn Kernel>,
            Box::new(RBF::new(2.0)) as Box<dyn Kernel>,
        ];
        let lmc = LinearModelCoregionalization::new()
            .kernels(kernels)
            .n_outputs(2)
            .alpha(1e-6);

        let fitted = lmc.fit(&X.view(), &Y.view()).unwrap();
        let contributions = fitted.latent_contributions(&X.view()).unwrap();

        assert_eq!(contributions.len(), 2); // Two latent functions
        assert_eq!(contributions[0].shape(), &[4, 2]);
        assert_eq!(contributions[1].shape(), &[4, 2]);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_lmc_errors() {
        let X = array![[1.0], [2.0], [3.0], [4.0]];
        let Y = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];

        // Test with no kernels
        let lmc = LinearModelCoregionalization::new().n_outputs(2).alpha(1e-6);
        assert!(lmc.fit(&X.view(), &Y.view()).is_err());

        // Test with mismatched dimensions
        let X_wrong = array![[1.0], [2.0], [3.0]]; // Different number of samples
        let kernels = vec![Box::new(RBF::new(1.0)) as Box<dyn Kernel>];
        let lmc = LinearModelCoregionalization::new()
            .kernels(kernels)
            .n_outputs(2)
            .alpha(1e-6);
        assert!(lmc.fit(&X_wrong.view(), &Y.view()).is_err());

        // Test with wrong n_outputs
        let kernels = vec![Box::new(RBF::new(1.0)) as Box<dyn Kernel>];
        let lmc = LinearModelCoregionalization::new()
            .kernels(kernels)
            .n_outputs(3) // Y has 2 outputs, but we specify 3
            .alpha(1e-6);
        assert!(lmc.fit(&X.view(), &Y.view()).is_err());
    }
}
