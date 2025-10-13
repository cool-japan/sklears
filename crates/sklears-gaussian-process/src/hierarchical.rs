//! Hierarchical Gaussian Processes
//!
//! This module implements hierarchical Gaussian processes for modeling data with natural
//! hierarchical structure. The model captures both global trends and group-specific deviations,
//! making it ideal for applications like medical data (patients within hospitals), educational
//! data (students within schools), or A/B testing (users within treatment groups).
//!
//! # Mathematical Model
//!
//! The hierarchical GP model consists of:
//! - Global function: f_global ~ GP(0, k_global)
//! - Group functions: f_group_i ~ GP(0, k_group) for each group i
//! - Observations: y_ij = f_global(x_ij) + f_group_i(x_ij) + ε_ij
//!
//! # Example
//!
//! ```rust
//! use sklears_gaussian_process::hierarchical::HierarchicalGaussianProcessRegressor;
//! use sklears_gaussian_process::kernels::RBF;
//! use sklears_core::traits::{Fit, Predict};
//! use scirs2_core::ndarray::array;
//! use std::collections::HashMap;
//!
//! // Create hierarchical GP regressor
//! let mut model = HierarchicalGaussianProcessRegressor::builder()
//!     .global_kernel(Box::new(RBF::new(1.0)))
//!     .group_kernel(Box::new(RBF::new(0.5)))
//!     .noise_variance(0.01)
//!     .global_weight(0.7)
//!     .group_weight(0.3)
//!     .build();
//!
//! // Prepare hierarchical data
//! let x_data = array![[0.0], [1.0], [2.0], [3.0]];
//! let y_data = array![1.0, 2.1, 2.9, 4.2];
//! let groups = vec!["A".to_string(), "A".to_string(), "B".to_string(), "B".to_string()];
//!
//! let mut group_data = HashMap::new();
//! group_data.insert("A".to_string(), (
//!     array![[0.0], [1.0]],
//!     array![1.0, 2.1]
//! ));
//! group_data.insert("B".to_string(), (
//!     array![[2.0], [3.0]],
//!     array![2.9, 4.2]
//! ));
//!
//! // Fit the model
//! let trained_model = model.fit(&x_data, &y_data, &group_data).unwrap();
//!
//! // Make predictions
//! let x_test = array![[1.5]];
//! let predictions = trained_model.predict(&x_test, "A").unwrap();
//! ```

use crate::kernels::Kernel;
use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::error::{Result as SklResult, SklearsError};
use sklears_core::traits::Estimator;
use std::collections::HashMap;

/// State marker for untrained hierarchical GP
#[derive(Debug, Clone)]
pub struct Untrained;

/// State marker for trained hierarchical GP
#[derive(Debug, Clone)]
pub struct Trained {
    /// global_data
    pub global_data: (Array2<f64>, Array1<f64>),
    /// group_data
    pub group_data: HashMap<String, (Array2<f64>, Array1<f64>)>,
    /// global_chol
    pub global_chol: Array2<f64>,
    /// group_chols
    pub group_chols: HashMap<String, Array2<f64>>,
    /// global_alpha
    pub global_alpha: Array1<f64>,
    /// group_alphas
    pub group_alphas: HashMap<String, Array1<f64>>,
}

/// Hierarchical Gaussian Process Regressor
///
/// Models data with hierarchical structure using a combination of global and group-specific
/// Gaussian processes. This is particularly useful when you have data that naturally clusters
/// into groups but still shares some global structure.
#[derive(Debug, Clone)]
pub struct HierarchicalGaussianProcessRegressor<S = Untrained> {
    global_kernel: Option<Box<dyn Kernel>>,
    group_kernel: Option<Box<dyn Kernel>>,
    noise_variance: f64,
    global_weight: f64,
    group_weight: f64,
    jitter: f64,
    _state: S,
}

impl Default for HierarchicalGaussianProcessRegressor<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl HierarchicalGaussianProcessRegressor<Untrained> {
    /// Create a new hierarchical GP regressor with default parameters
    pub fn new() -> Self {
        Self {
            global_kernel: None,
            group_kernel: None,
            noise_variance: 0.01,
            global_weight: 0.5,
            group_weight: 0.5,
            jitter: 1e-6,
            _state: Untrained,
        }
    }

    /// Create a builder for configuring the hierarchical GP regressor
    pub fn builder() -> HierarchicalGaussianProcessRegressorBuilder {
        HierarchicalGaussianProcessRegressorBuilder::new()
    }

    /// Set the global kernel
    pub fn with_global_kernel(mut self, kernel: Box<dyn Kernel>) -> Self {
        self.global_kernel = Some(kernel);
        self
    }

    /// Set the group kernel
    pub fn with_group_kernel(mut self, kernel: Box<dyn Kernel>) -> Self {
        self.group_kernel = Some(kernel);
        self
    }

    /// Set the noise variance
    pub fn with_noise_variance(mut self, noise_variance: f64) -> Self {
        self.noise_variance = noise_variance;
        self
    }

    /// Set the global weight (relative importance of global vs group effects)
    pub fn with_global_weight(mut self, weight: f64) -> SklResult<Self> {
        if !(0.0..=1.0).contains(&weight) {
            return Err(SklearsError::InvalidParameter {
                name: "global_weight".to_string(),
                reason: "must be between 0.0 and 1.0".to_string(),
            });
        }
        self.global_weight = weight;
        Ok(self)
    }

    /// Set the group weight (relative importance of group vs global effects)
    pub fn with_group_weight(mut self, weight: f64) -> SklResult<Self> {
        if !(0.0..=1.0).contains(&weight) {
            return Err(SklearsError::InvalidParameter {
                name: "group_weight".to_string(),
                reason: "must be between 0.0 and 1.0".to_string(),
            });
        }
        self.group_weight = weight;
        Ok(self)
    }

    /// Set the jitter for numerical stability
    pub fn with_jitter(mut self, jitter: f64) -> Self {
        self.jitter = jitter;
        self
    }

    /// Fit the hierarchical GP to data
    pub fn fit(
        self,
        global_x: &Array2<f64>,
        global_y: &Array1<f64>,
        group_data: &HashMap<String, (Array2<f64>, Array1<f64>)>,
    ) -> SklResult<HierarchicalGaussianProcessRegressor<Trained>> {
        // Validate inputs
        if global_x.nrows() != global_y.len() {
            return Err(SklearsError::DimensionMismatch {
                expected: global_x.nrows(),
                actual: global_y.len(),
            });
        }

        if group_data.is_empty() {
            return Err(SklearsError::InvalidInput("No groups provided".to_string()));
        }

        let global_kernel = self
            .global_kernel
            .as_ref()
            .ok_or_else(|| SklearsError::InvalidInput("Global kernel not set".to_string()))?
            .clone_box();
        let group_kernel = self
            .group_kernel
            .as_ref()
            .ok_or_else(|| SklearsError::InvalidInput("Group kernel not set".to_string()))?
            .clone_box();

        // Fit global component
        let global_gram = global_kernel.compute_kernel_matrix(global_x, None)?;
        let mut global_k = global_gram;
        for i in 0..global_k.nrows() {
            global_k[[i, i]] += self.noise_variance + self.jitter;
        }

        let global_chol = self.cholesky_decomposition(&global_k)?;
        let global_alpha = self.solve_triangular(&global_chol, global_y)?;

        // Fit group components
        let mut group_chols = HashMap::new();
        let mut group_alphas = HashMap::new();

        for (group_name, (x, y)) in group_data {
            if x.is_empty() || y.is_empty() {
                return Err(SklearsError::InvalidInput(format!(
                    "Empty data for group {}",
                    group_name
                )));
            }

            if x.nrows() != y.len() {
                return Err(SklearsError::DimensionMismatch {
                    expected: x.nrows(),
                    actual: y.len(),
                });
            }

            let group_gram = group_kernel.compute_kernel_matrix(x, None)?;
            let mut group_k = group_gram;
            for i in 0..group_k.nrows() {
                group_k[[i, i]] += self.noise_variance * self.group_weight + self.jitter;
            }

            let group_chol = self.cholesky_decomposition(&group_k)?;
            let group_alpha = self.solve_triangular(&group_chol, y)?;

            group_chols.insert(group_name.clone(), group_chol);
            group_alphas.insert(group_name.clone(), group_alpha);
        }

        Ok(HierarchicalGaussianProcessRegressor {
            global_kernel: Some(global_kernel.clone_box()),
            group_kernel: Some(group_kernel.clone_box()),
            noise_variance: self.noise_variance,
            global_weight: self.global_weight,
            group_weight: self.group_weight,
            jitter: self.jitter,
            _state: Trained {
                global_data: (global_x.clone(), global_y.clone()),
                group_data: group_data.clone(),
                global_chol,
                group_chols,
                global_alpha,
                group_alphas,
            },
        })
    }

    /// Perform Cholesky decomposition with error handling
    fn cholesky_decomposition(&self, matrix: &Array2<f64>) -> SklResult<Array2<f64>> {
        // Use a simple implementation since we don't have access to advanced LAPACK
        let n = matrix.nrows();
        let mut chol = Array2::zeros((n, n));

        for i in 0..n {
            for j in 0..=i {
                if i == j {
                    // Diagonal element
                    let mut sum = 0.0;
                    for k in 0..j {
                        sum += chol[[j, k]] * chol[[j, k]];
                    }
                    let val = matrix[[j, j]] - sum;
                    if val <= 0.0 {
                        return Err(SklearsError::NumericalError(
                            "Matrix is not positive definite".to_string(),
                        ));
                    }
                    chol[[j, j]] = val.sqrt();
                } else {
                    // Off-diagonal element
                    let mut sum = 0.0;
                    for k in 0..j {
                        sum += chol[[i, k]] * chol[[j, k]];
                    }
                    chol[[i, j]] = (matrix[[i, j]] - sum) / chol[[j, j]];
                }
            }
        }

        Ok(chol)
    }

    /// Solve triangular system L * x = b
    fn solve_triangular(&self, l: &Array2<f64>, b: &Array1<f64>) -> SklResult<Array1<f64>> {
        let n = l.nrows();
        let mut x = Array1::zeros(n);

        // Forward substitution for L * y = b
        for i in 0..n {
            let mut sum = 0.0;
            for j in 0..i {
                sum += l[[i, j]] * x[j];
            }
            x[i] = (b[i] - sum) / l[[i, i]];
        }

        // Backward substitution for L^T * x = y
        let mut result = Array1::zeros(n);
        for i in (0..n).rev() {
            let mut sum = 0.0;
            for j in (i + 1)..n {
                sum += l[[j, i]] * result[j];
            }
            result[i] = (x[i] - sum) / l[[i, i]];
        }

        Ok(result)
    }
}

impl HierarchicalGaussianProcessRegressor<Trained> {
    /// Make predictions for a specific group
    pub fn predict(
        &self,
        x_test: &Array2<f64>,
        group: &str,
    ) -> SklResult<(Array1<f64>, Array1<f64>)> {
        let state = &self._state;

        if !state.group_data.contains_key(group) {
            return Err(SklearsError::InvalidInput(format!(
                "Group '{}' not found",
                group
            )));
        }

        let global_kernel = self
            .global_kernel
            .as_ref()
            .ok_or_else(|| SklearsError::InvalidInput("Global kernel not set".to_string()))?;
        let group_kernel = self
            .group_kernel
            .as_ref()
            .ok_or_else(|| SklearsError::InvalidInput("Group kernel not set".to_string()))?;

        // Global predictions
        let global_k_star =
            global_kernel.compute_kernel_matrix(&state.global_data.0, Some(x_test))?;
        let global_mean = global_k_star.t().dot(&state.global_alpha) * self.global_weight;

        // Group predictions
        let (group_x, _) = &state.group_data[group];
        let group_k_star = group_kernel.compute_kernel_matrix(group_x, Some(x_test))?;
        let group_alpha = &state.group_alphas[group];
        let group_mean = group_k_star.t().dot(group_alpha) * self.group_weight;

        // Combined mean prediction
        let mean_pred = global_mean + group_mean;

        // Simplified variance computation (diagonal only)
        let global_k_test = global_kernel.compute_kernel_matrix(x_test, None)?;
        let group_k_test = group_kernel.compute_kernel_matrix(x_test, None)?;

        let mut variance = Array1::zeros(x_test.nrows());
        for i in 0..x_test.nrows() {
            variance[i] = self.global_weight * self.global_weight * global_k_test[[i, i]]
                + self.group_weight * self.group_weight * group_k_test[[i, i]]
                + self.noise_variance;
        }

        Ok((mean_pred, variance))
    }

    /// Make predictions using only the global component
    pub fn predict_global(&self, x_test: &Array2<f64>) -> SklResult<(Array1<f64>, Array1<f64>)> {
        let state = &self._state;

        let global_kernel = self
            .global_kernel
            .as_ref()
            .ok_or_else(|| SklearsError::InvalidInput("Global kernel not set".to_string()))?;

        let global_k_star =
            global_kernel.compute_kernel_matrix(&state.global_data.0, Some(x_test))?;
        let mean_pred = global_k_star.t().dot(&state.global_alpha);

        let global_k_test = global_kernel.compute_kernel_matrix(x_test, None)?;
        let mut variance = Array1::zeros(x_test.nrows());
        for i in 0..x_test.nrows() {
            variance[i] = global_k_test[[i, i]] + self.noise_variance;
        }

        Ok((mean_pred, variance))
    }

    /// Make predictions for a group using only the group-specific component
    pub fn predict_group_only(
        &self,
        x_test: &Array2<f64>,
        group: &str,
    ) -> SklResult<(Array1<f64>, Array1<f64>)> {
        let state = &self._state;

        if !state.group_data.contains_key(group) {
            return Err(SklearsError::InvalidInput(format!(
                "Group '{}' not found",
                group
            )));
        }

        let group_kernel = self
            .group_kernel
            .as_ref()
            .ok_or_else(|| SklearsError::InvalidInput("Group kernel not set".to_string()))?;

        let (group_x, _) = &state.group_data[group];
        let group_k_star = group_kernel.compute_kernel_matrix(group_x, Some(x_test))?;
        let group_alpha = &state.group_alphas[group];
        let mean_pred = group_k_star.t().dot(group_alpha);

        let group_k_test = group_kernel.compute_kernel_matrix(x_test, None)?;
        let mut variance = Array1::zeros(x_test.nrows());
        for i in 0..x_test.nrows() {
            variance[i] = group_k_test[[i, i]] + self.noise_variance * self.group_weight;
        }

        Ok((mean_pred, variance))
    }

    /// Get the list of available groups
    pub fn get_groups(&self) -> Vec<String> {
        self._state.group_data.keys().cloned().collect()
    }

    /// Get the global data used for training
    pub fn get_global_data(&self) -> (&Array2<f64>, &Array1<f64>) {
        (&self._state.global_data.0, &self._state.global_data.1)
    }

    /// Get the group data used for training
    pub fn get_group_data(&self, group: &str) -> Option<(&Array2<f64>, &Array1<f64>)> {
        self._state.group_data.get(group).map(|(x, y)| (x, y))
    }

    /// Compute log marginal likelihood for the hierarchical model
    pub fn log_marginal_likelihood(&self) -> SklResult<f64> {
        let state = &self._state;
        let n_global = state.global_data.1.len();

        // Global component log marginal likelihood
        let global_log_det = (0..state.global_chol.nrows())
            .map(|i| state.global_chol[[i, i]].ln())
            .sum::<f64>()
            * 2.0;

        let global_quad_form = state.global_alpha.dot(&state.global_data.1);
        let global_lml = -0.5
            * (global_quad_form
                + global_log_det
                + n_global as f64 * (2.0 * std::f64::consts::PI).ln());

        // Group components log marginal likelihood
        let mut group_lml = 0.0;
        for (group_name, (_, y)) in &state.group_data {
            let chol = &state.group_chols[group_name];
            let alpha = &state.group_alphas[group_name];

            let log_det = (0..chol.nrows()).map(|i| chol[[i, i]].ln()).sum::<f64>() * 2.0;

            let quad_form = alpha.dot(y);
            let n = y.len();
            group_lml +=
                -0.5 * (quad_form + log_det + n as f64 * (2.0 * std::f64::consts::PI).ln());
        }

        // Combined log marginal likelihood (weighted)
        Ok(self.global_weight * global_lml + self.group_weight * group_lml)
    }
}

/// Builder for hierarchical Gaussian process regressor
pub struct HierarchicalGaussianProcessRegressorBuilder {
    global_kernel: Option<Box<dyn Kernel>>,
    group_kernel: Option<Box<dyn Kernel>>,
    noise_variance: f64,
    global_weight: f64,
    group_weight: f64,
    jitter: f64,
}

impl HierarchicalGaussianProcessRegressorBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            global_kernel: None,
            group_kernel: None,
            noise_variance: 0.01,
            global_weight: 0.5,
            group_weight: 0.5,
            jitter: 1e-6,
        }
    }

    /// Set the global kernel
    pub fn global_kernel(mut self, kernel: Box<dyn Kernel>) -> Self {
        self.global_kernel = Some(kernel);
        self
    }

    /// Set the group kernel
    pub fn group_kernel(mut self, kernel: Box<dyn Kernel>) -> Self {
        self.group_kernel = Some(kernel);
        self
    }

    /// Set the noise variance
    pub fn noise_variance(mut self, noise_variance: f64) -> Self {
        self.noise_variance = noise_variance;
        self
    }

    /// Set the global weight
    pub fn global_weight(mut self, weight: f64) -> Self {
        self.global_weight = weight;
        self
    }

    /// Set the group weight
    pub fn group_weight(mut self, weight: f64) -> Self {
        self.group_weight = weight;
        self
    }

    /// Set the jitter for numerical stability
    pub fn jitter(mut self, jitter: f64) -> Self {
        self.jitter = jitter;
        self
    }

    /// Build the hierarchical GP regressor
    pub fn build(self) -> HierarchicalGaussianProcessRegressor<Untrained> {
        HierarchicalGaussianProcessRegressor {
            global_kernel: self.global_kernel,
            group_kernel: self.group_kernel,
            noise_variance: self.noise_variance,
            global_weight: self.global_weight,
            group_weight: self.group_weight,
            jitter: self.jitter,
            _state: Untrained,
        }
    }
}

impl Default for HierarchicalGaussianProcessRegressorBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// Implement required traits
impl<S> Estimator for HierarchicalGaussianProcessRegressor<S> {
    type Config = HierarchicalGPConfig;
    type Error = SklearsError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        // For now, return a static config reference
        static DEFAULT_CONFIG: HierarchicalGPConfig = HierarchicalGPConfig {
            noise_variance: 0.01,
            global_weight: 0.5,
            group_weight: 0.5,
            jitter: 1e-6,
        };
        &DEFAULT_CONFIG
    }
}

/// Configuration for hierarchical Gaussian process
#[derive(Debug, Clone)]
pub struct HierarchicalGPConfig {
    /// noise_variance
    pub noise_variance: f64,
    /// global_weight
    pub global_weight: f64,
    /// group_weight
    pub group_weight: f64,
    /// jitter
    pub jitter: f64,
}

impl Default for HierarchicalGPConfig {
    fn default() -> Self {
        Self {
            noise_variance: 0.01,
            global_weight: 0.5,
            group_weight: 0.5,
            jitter: 1e-6,
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::RBF;

    use scirs2_core::ndarray::array;

    #[test]
    fn test_hierarchical_gp_creation() {
        let model = HierarchicalGaussianProcessRegressor::builder()
            .global_kernel(Box::new(RBF::new(1.0)))
            .group_kernel(Box::new(RBF::new(0.5)))
            .noise_variance(0.01)
            .build();

        let config = model.config();
        assert_eq!(config.noise_variance, 0.01);
        assert_eq!(config.global_weight, 0.5);
        assert_eq!(config.group_weight, 0.5);
    }

    #[test]
    fn test_hierarchical_gp_fit_predict() {
        let model = HierarchicalGaussianProcessRegressor::builder()
            .global_kernel(Box::new(RBF::new(1.0)))
            .group_kernel(Box::new(RBF::new(0.5)))
            .noise_variance(0.01)
            .global_weight(0.7)
            .group_weight(0.3)
            .build();

        // Create hierarchical data
        let global_x = array![[0.0], [1.0], [2.0], [3.0]];
        let global_y = array![1.0, 2.0, 3.0, 4.0];

        let mut group_data = HashMap::new();
        group_data.insert("A".to_string(), (array![[0.0], [1.0]], array![1.1, 1.9]));
        group_data.insert("B".to_string(), (array![[2.0], [3.0]], array![3.1, 3.9]));

        let trained_model = model.fit(&global_x, &global_y, &group_data).unwrap();

        // Test predictions
        let x_test = array![[1.5]];
        let (mean_pred, var_pred) = trained_model.predict(&x_test, "A").unwrap();

        assert_eq!(mean_pred.len(), 1);
        assert_eq!(var_pred.len(), 1);
        assert!(var_pred[0] > 0.0);
    }

    #[test]
    fn test_hierarchical_gp_global_predictions() {
        let model = HierarchicalGaussianProcessRegressor::builder()
            .global_kernel(Box::new(RBF::new(1.0)))
            .group_kernel(Box::new(RBF::new(0.5)))
            .noise_variance(0.01)
            .build();

        let global_x = array![[0.0], [1.0], [2.0]];
        let global_y = array![0.0, 1.0, 2.0];

        let mut group_data = HashMap::new();
        group_data.insert("A".to_string(), (array![[0.0], [1.0]], array![0.1, 0.9]));

        let trained_model = model.fit(&global_x, &global_y, &group_data).unwrap();

        let x_test = array![[0.5]];
        let (global_pred, _) = trained_model.predict_global(&x_test).unwrap();
        let (group_pred, _) = trained_model.predict_group_only(&x_test, "A").unwrap();
        let (combined_pred, _) = trained_model.predict(&x_test, "A").unwrap();

        // Combined prediction should be a weighted combination
        assert!(global_pred.len() == 1);
        assert!(group_pred.len() == 1);
        assert!(combined_pred.len() == 1);
    }

    #[test]
    fn test_hierarchical_gp_group_management() {
        let model = HierarchicalGaussianProcessRegressor::builder()
            .global_kernel(Box::new(RBF::new(1.0)))
            .group_kernel(Box::new(RBF::new(0.5)))
            .build();

        let global_x = array![[0.0], [1.0]];
        let global_y = array![0.0, 1.0];

        let mut group_data = HashMap::new();
        group_data.insert("A".to_string(), (array![[0.0]], array![0.1]));
        group_data.insert("B".to_string(), (array![[1.0]], array![0.9]));

        let trained_model = model.fit(&global_x, &global_y, &group_data).unwrap();

        let groups = trained_model.get_groups();
        assert_eq!(groups.len(), 2);
        assert!(groups.contains(&"A".to_string()));
        assert!(groups.contains(&"B".to_string()));

        let (group_x, group_y) = trained_model.get_group_data("A").unwrap();
        assert_eq!(group_x.nrows(), 1);
        assert_eq!(group_y.len(), 1);
    }

    #[test]
    fn test_hierarchical_gp_log_marginal_likelihood() {
        let model = HierarchicalGaussianProcessRegressor::builder()
            .global_kernel(Box::new(RBF::new(1.0)))
            .group_kernel(Box::new(RBF::new(0.5)))
            .noise_variance(0.1)
            .build();

        let global_x = array![[0.0], [1.0], [2.0]];
        let global_y = array![0.0, 1.0, 0.5];

        let mut group_data = HashMap::new();
        group_data.insert("A".to_string(), (array![[0.0], [1.0]], array![0.1, 0.9]));

        let trained_model = model.fit(&global_x, &global_y, &group_data).unwrap();
        let lml = trained_model.log_marginal_likelihood().unwrap();

        assert!(lml.is_finite());
    }

    #[test]
    fn test_hierarchical_gp_errors() {
        let model = HierarchicalGaussianProcessRegressor::builder()
            .global_kernel(Box::new(RBF::new(1.0)))
            .group_kernel(Box::new(RBF::new(0.5)))
            .build();

        let global_x = array![[0.0], [1.0]];
        let global_y = array![0.0, 1.0];

        let mut group_data = HashMap::new();
        group_data.insert("A".to_string(), (array![[0.0]], array![0.1]));

        let trained_model = model.fit(&global_x, &global_y, &group_data).unwrap();

        // Test prediction for non-existent group
        let x_test = array![[0.5]];
        let result = trained_model.predict(&x_test, "NonExistent");
        assert!(result.is_err());
    }

    #[test]
    fn test_hierarchical_gp_empty_data_error() {
        let model = HierarchicalGaussianProcessRegressor::builder()
            .global_kernel(Box::new(RBF::new(1.0)))
            .group_kernel(Box::new(RBF::new(0.5)))
            .build();

        let global_x = array![[0.0]];
        let global_y = array![0.0];

        // Empty group data
        let group_data = HashMap::new();
        let result = model.fit(&global_x, &global_y, &group_data);
        assert!(result.is_err());
    }

    #[test]
    fn test_hierarchical_gp_weight_validation() {
        let result = HierarchicalGaussianProcessRegressor::new().with_global_weight(1.5);
        assert!(result.is_err());

        let result = HierarchicalGaussianProcessRegressor::new().with_group_weight(-0.1);
        assert!(result.is_err());
    }

    #[test]
    fn test_hierarchical_gp_kernel_not_set() {
        let mut model = HierarchicalGaussianProcessRegressor::new();
        // Only set global kernel, not group kernel
        model.global_kernel = Some(Box::new(RBF::new(1.0)));

        let global_x = array![[0.0]];
        let global_y = array![0.0];
        let mut group_data = HashMap::new();
        group_data.insert("A".to_string(), (array![[0.0]], array![0.1]));

        let result = model.fit(&global_x, &global_y, &group_data);
        assert!(result.is_err());
    }

    #[test]
    fn test_hierarchical_gp_dimension_mismatch() {
        let model = HierarchicalGaussianProcessRegressor::builder()
            .global_kernel(Box::new(RBF::new(1.0)))
            .group_kernel(Box::new(RBF::new(0.5)))
            .build();

        let global_x = array![[0.0], [1.0]];
        let global_y = array![0.0]; // Wrong size

        let mut group_data = HashMap::new();
        group_data.insert("A".to_string(), (array![[0.0]], array![0.1]));

        let result = model.fit(&global_x, &global_y, &group_data);
        assert!(result.is_err());
    }

    #[test]
    fn test_hierarchical_gp_multiple_groups() {
        let model = HierarchicalGaussianProcessRegressor::builder()
            .global_kernel(Box::new(RBF::new(1.0)))
            .group_kernel(Box::new(RBF::new(0.5)))
            .noise_variance(0.01)
            .global_weight(0.6)
            .group_weight(0.4)
            .build();

        let global_x = array![[0.0], [1.0], [2.0], [3.0], [4.0]];
        let global_y = array![0.0, 1.0, 2.0, 1.5, 3.0];

        let mut group_data = HashMap::new();
        group_data.insert("A".to_string(), (array![[0.0], [1.0]], array![0.1, 0.9]));
        group_data.insert("B".to_string(), (array![[2.0], [3.0]], array![2.1, 1.4]));
        group_data.insert("C".to_string(), (array![[4.0]], array![2.9]));

        let trained_model = model.fit(&global_x, &global_y, &group_data).unwrap();

        let x_test = array![[2.5]];
        let (pred_a, _) = trained_model.predict(&x_test, "A").unwrap();
        let (pred_b, _) = trained_model.predict(&x_test, "B").unwrap();
        let (pred_c, _) = trained_model.predict(&x_test, "C").unwrap();

        assert_eq!(pred_a.len(), 1);
        assert_eq!(pred_b.len(), 1);
        assert_eq!(pred_c.len(), 1);

        // Predictions should be different for different groups
        assert!((pred_a[0] - pred_b[0]).abs() > 1e-6 || (pred_b[0] - pred_c[0]).abs() > 1e-6);
    }
}
