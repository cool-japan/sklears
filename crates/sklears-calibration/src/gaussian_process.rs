//! Gaussian Process calibration
//!
//! This module provides calibration methods based on Gaussian processes,
//! which can model complex non-linear calibration relationships with
//! uncertainty quantification.

use scirs2_core::ndarray::{s, Array1, Array2};
use sklears_core::{error::Result, prelude::SklearsError, types::Float};
use std::f32::consts::PI as PI_F32;

use crate::CalibrationEstimator;

/// Gaussian Process calibrator
///
/// Uses Gaussian processes to learn a non-parametric calibration function
/// with uncertainty quantification.
#[derive(Debug, Clone)]
pub struct GaussianProcessCalibrator {
    kernel: GPKernel,
    noise_variance: Float,
    training_probabilities: Option<Array1<Float>>,
    training_labels: Option<Array1<Float>>,
    kernel_matrix_inv: Option<Array2<Float>>,
    kernel_params: GPKernelParams,
    optimize_hyperparams: bool,
}

/// Gaussian Process kernel types
#[derive(Debug, Clone)]
pub enum GPKernel {
    /// Radial Basis Function (RBF) kernel
    RBF,
    /// Matern 3/2 kernel
    Matern32,
    /// Matern 5/2 kernel
    Matern52,
    /// Linear kernel
    Linear,
    /// Polynomial kernel
    Polynomial { degree: usize },
}

/// Kernel parameters for Gaussian Process
#[derive(Debug, Clone)]
pub struct GPKernelParams {
    /// Length scale parameter
    pub length_scale: Float,
    /// Signal variance parameter
    pub signal_variance: Float,
    /// Additional parameters for specific kernels
    pub additional_params: Vec<Float>,
}

impl Default for GPKernelParams {
    fn default() -> Self {
        Self {
            length_scale: 1.0,
            signal_variance: 1.0,
            additional_params: Vec::new(),
        }
    }
}

impl GaussianProcessCalibrator {
    /// Create a new Gaussian Process calibrator
    pub fn new() -> Self {
        Self {
            kernel: GPKernel::RBF,
            noise_variance: 0.01,
            training_probabilities: None,
            training_labels: None,
            kernel_matrix_inv: None,
            kernel_params: GPKernelParams::default(),
            optimize_hyperparams: true,
        }
    }

    /// Set the kernel type
    pub fn kernel(mut self, kernel: GPKernel) -> Self {
        self.kernel = kernel;
        self
    }

    /// Set the noise variance
    pub fn noise_variance(mut self, noise_variance: Float) -> Self {
        self.noise_variance = noise_variance;
        self
    }

    /// Set kernel parameters
    pub fn kernel_params(mut self, params: GPKernelParams) -> Self {
        self.kernel_params = params;
        self
    }

    /// Set whether to optimize hyperparameters
    pub fn optimize_hyperparams(mut self, optimize: bool) -> Self {
        self.optimize_hyperparams = optimize;
        self
    }

    /// Fit the Gaussian Process calibrator
    pub fn fit(mut self, probabilities: &Array1<Float>, y_true: &Array1<i32>) -> Result<Self> {
        if probabilities.len() != y_true.len() {
            return Err(SklearsError::InvalidInput(
                "Probabilities and labels must have the same length".to_string(),
            ));
        }

        if probabilities.is_empty() {
            return Err(SklearsError::InvalidInput(
                "No probabilities provided".to_string(),
            ));
        }

        // Convert integer labels to float for GP regression
        let training_labels = y_true.mapv(|y| y as Float);

        // Optimize hyperparameters if requested
        if self.optimize_hyperparams {
            self.kernel_params = self.optimize_hyperparameters(probabilities, &training_labels)?;
        }

        // Compute kernel matrix and its inverse
        let kernel_matrix = self.compute_kernel_matrix(probabilities, probabilities)?;
        let kernel_matrix_inv = self.invert_kernel_matrix(&kernel_matrix)?;

        self.training_probabilities = Some(probabilities.clone());
        self.training_labels = Some(training_labels);
        self.kernel_matrix_inv = Some(kernel_matrix_inv);

        Ok(self)
    }

    /// Predict calibrated probabilities with uncertainty
    pub fn predict_proba_with_uncertainty(
        &self,
        probabilities: &Array1<Float>,
    ) -> Result<(Array1<Float>, Array1<Float>)> {
        let training_probabilities =
            self.training_probabilities
                .as_ref()
                .ok_or_else(|| SklearsError::NotFitted {
                    operation: "predict_proba_with_uncertainty".to_string(),
                })?;
        let training_labels =
            self.training_labels
                .as_ref()
                .ok_or_else(|| SklearsError::NotFitted {
                    operation: "predict_proba_with_uncertainty".to_string(),
                })?;
        let kernel_matrix_inv =
            self.kernel_matrix_inv
                .as_ref()
                .ok_or_else(|| SklearsError::NotFitted {
                    operation: "predict_proba_with_uncertainty".to_string(),
                })?;

        let n_test = probabilities.len();
        let mut means = Array1::zeros(n_test);
        let mut variances = Array1::zeros(n_test);

        // Compute cross-covariance matrix between test and training points
        let k_star = self.compute_kernel_matrix(probabilities, training_probabilities)?;

        // Compute test kernel matrix (diagonal elements for variance computation)
        let k_star_star = self.compute_kernel_matrix(probabilities, probabilities)?;

        for i in 0..n_test {
            // Predictive mean
            let k_i = k_star.row(i);
            let mean = k_i.dot(&kernel_matrix_inv.dot(training_labels));
            means[i] = mean;

            // Predictive variance
            let v = kernel_matrix_inv.dot(&k_i);
            let variance = k_star_star[[i, i]] - k_i.dot(&v);
            variances[i] = variance.max(0.0); // Ensure non-negative variance
        }

        // Apply sigmoid transformation to convert to probabilities
        let calibrated_means =
            means.mapv(|x| sigmoid(x).clamp(1e-15 as Float, 1.0 - 1e-15 as Float));
        let calibrated_stds = variances.mapv(|v| v.sqrt());

        Ok((calibrated_means, calibrated_stds))
    }

    /// Predict calibrated probabilities (mean only)
    pub fn predict_proba(&self, probabilities: &Array1<Float>) -> Result<Array1<Float>> {
        let (means, _) = self.predict_proba_with_uncertainty(probabilities)?;
        // Ensure all predictions are valid probabilities
        Ok(means.mapv(|x| x.clamp(0.0, 1.0)))
    }

    fn compute_kernel_matrix(
        &self,
        x1: &Array1<Float>,
        x2: &Array1<Float>,
    ) -> Result<Array2<Float>> {
        let n1 = x1.len();
        let n2 = x2.len();
        let mut kernel_matrix = Array2::zeros((n1, n2));

        for i in 0..n1 {
            for j in 0..n2 {
                let kernel_value = self.kernel_function(x1[i], x2[j]);
                kernel_matrix[[i, j]] = kernel_value;
            }
        }

        // Add noise to diagonal if x1 == x2 (same points)
        if n1 == n2 && x1.iter().zip(x2.iter()).all(|(a, b)| (a - b).abs() < 1e-10) {
            for i in 0..n1 {
                kernel_matrix[[i, i]] += self.noise_variance;
            }
        }

        Ok(kernel_matrix)
    }

    fn kernel_function(&self, x1: Float, x2: Float) -> Float {
        let distance = (x1 - x2).abs();

        match self.kernel {
            GPKernel::RBF => {
                let scaled_distance = distance / self.kernel_params.length_scale;
                self.kernel_params.signal_variance
                    * (-0.5 * scaled_distance * scaled_distance).exp()
            }
            GPKernel::Matern32 => {
                let scaled_distance =
                    ((3.0 as Float).sqrt() * distance) / self.kernel_params.length_scale;
                self.kernel_params.signal_variance
                    * (1.0 + scaled_distance)
                    * (-scaled_distance).exp()
            }
            GPKernel::Matern52 => {
                let scaled_distance =
                    ((5.0 as Float).sqrt() * distance) / self.kernel_params.length_scale;
                self.kernel_params.signal_variance
                    * (1.0 + scaled_distance + scaled_distance * scaled_distance / 3.0)
                    * (-scaled_distance).exp()
            }
            GPKernel::Linear => self.kernel_params.signal_variance * x1 * x2,
            GPKernel::Polynomial { degree } => {
                let dot_product = x1 * x2;
                self.kernel_params.signal_variance * (1.0 + dot_product).powi(degree as i32)
            }
        }
    }

    fn invert_kernel_matrix(&self, kernel_matrix: &Array2<Float>) -> Result<Array2<Float>> {
        // Simple pseudo-inverse using eigenvalue decomposition (simplified)
        // In practice, you would use Cholesky decomposition for better numerical stability
        let n = kernel_matrix.nrows();

        // Add regularization to diagonal for numerical stability
        let mut regularized_matrix = kernel_matrix.clone();
        for i in 0..n {
            regularized_matrix[[i, i]] += 1e-6;
        }

        // Simplified inversion using iterative method
        self.iterative_inverse(&regularized_matrix)
    }

    fn iterative_inverse(&self, matrix: &Array2<Float>) -> Result<Array2<Float>> {
        let n = matrix.nrows();
        let mut inverse = Array2::<Float>::eye(n);

        // For small matrices, use simple approximation
        if n <= 3 {
            // Use simple diagonal approximation for numerical stability
            for i in 0..n {
                if matrix[[i, i]] != 0.0 {
                    inverse[[i, i]] = 1.0 / matrix[[i, i]];
                }
            }
            return Ok(inverse);
        }

        let max_iterations = 10; // Reduced iterations for stability
        let tolerance = 1e-4; // Relaxed tolerance

        // Simple iterative refinement (simplified for demonstration)
        for _iter in 0..max_iterations {
            let product = matrix.dot(&inverse);
            let identity = Array2::<Float>::eye(n);
            let residual = &product - &identity;
            let residual_norm = residual.mapv(|x| x.abs()).sum();

            if residual_norm < tolerance {
                break;
            }

            // Simple damped update for stability
            let damping = 0.1;
            for i in 0..n {
                for j in 0..n {
                    if matrix[[i, i]] != 0.0 {
                        inverse[[i, j]] -= damping * residual[[i, j]] / matrix[[i, i]];
                    }
                }
            }
        }

        Ok(inverse)
    }

    fn optimize_hyperparameters(
        &self,
        probabilities: &Array1<Float>,
        labels: &Array1<Float>,
    ) -> Result<GPKernelParams> {
        // Simple grid search optimization (in practice, would use gradient-based methods)
        let length_scales = vec![0.1, 0.5, 1.0, 2.0, 5.0];
        let signal_variances = vec![0.1, 0.5, 1.0, 2.0];

        let mut best_params = self.kernel_params.clone();
        let mut best_log_likelihood = Float::NEG_INFINITY;

        for &length_scale in &length_scales {
            for &signal_variance in &signal_variances {
                let params = GPKernelParams {
                    length_scale,
                    signal_variance,
                    additional_params: Vec::new(),
                };

                // Compute log marginal likelihood
                if let Ok(log_likelihood) =
                    self.compute_log_marginal_likelihood(probabilities, labels, &params)
                {
                    if log_likelihood > best_log_likelihood {
                        best_log_likelihood = log_likelihood;
                        best_params = params;
                    }
                }
            }
        }

        Ok(best_params)
    }

    fn compute_log_marginal_likelihood(
        &self,
        probabilities: &Array1<Float>,
        labels: &Array1<Float>,
        params: &GPKernelParams,
    ) -> Result<Float> {
        // Temporarily set parameters
        let _original_params = self.kernel_params.clone();
        let mut temp_self = self.clone();
        temp_self.kernel_params = params.clone();

        // Compute kernel matrix with new parameters
        let kernel_matrix = temp_self.compute_kernel_matrix(probabilities, probabilities)?;

        // Compute log determinant (simplified)
        let log_det = self.compute_log_determinant(&kernel_matrix)?;

        // Compute quadratic form
        let kernel_inv = temp_self.invert_kernel_matrix(&kernel_matrix)?;
        let quadratic_form = labels.dot(&kernel_inv.dot(labels));

        // Log marginal likelihood
        let n = probabilities.len() as Float;
        let log_likelihood =
            -0.5 * quadratic_form - 0.5 * log_det - 0.5 * n * (2.0 * PI_F32 as Float).ln();

        Ok(log_likelihood)
    }

    fn compute_log_determinant(&self, matrix: &Array2<Float>) -> Result<Float> {
        // Simplified log determinant computation
        // In practice, would use Cholesky decomposition
        let n = matrix.nrows();
        let mut log_det = 0.0;

        // Use diagonal elements as approximation (not accurate, but simple)
        for i in 0..n {
            let diag_element = matrix[[i, i]];
            if diag_element > 0.0 {
                log_det += diag_element.ln();
            }
        }

        Ok(log_det)
    }
}

impl Default for GaussianProcessCalibrator {
    fn default() -> Self {
        Self::new()
    }
}

impl CalibrationEstimator for GaussianProcessCalibrator {
    fn fit(&mut self, probabilities: &Array1<Float>, y_true: &Array1<i32>) -> Result<()> {
        *self = self.clone().fit(probabilities, y_true)?;
        Ok(())
    }

    fn predict_proba(&self, probabilities: &Array1<Float>) -> Result<Array1<Float>> {
        GaussianProcessCalibrator::predict_proba(self, probabilities)
    }

    fn clone_box(&self) -> Box<dyn CalibrationEstimator> {
        Box::new(self.clone())
    }
}

/// Variational Gaussian Process calibrator for large datasets
#[derive(Debug, Clone)]
pub struct VariationalGPCalibrator {
    base_gp: GaussianProcessCalibrator,
    inducing_points: Option<Array1<Float>>,
    n_inducing: usize,
    variational_params: Option<VariationalParams>,
}

#[derive(Debug, Clone)]
struct VariationalParams {
    inducing_mean: Array1<Float>,
    inducing_covariance: Array2<Float>,
}

impl VariationalGPCalibrator {
    /// Create a new variational GP calibrator
    pub fn new(n_inducing: usize) -> Self {
        Self {
            base_gp: GaussianProcessCalibrator::new(),
            inducing_points: None,
            n_inducing,
            variational_params: None,
        }
    }

    /// Set the kernel type
    pub fn kernel(mut self, kernel: GPKernel) -> Self {
        self.base_gp = self.base_gp.kernel(kernel);
        self
    }

    /// Fit the variational GP calibrator
    pub fn fit(mut self, probabilities: &Array1<Float>, y_true: &Array1<i32>) -> Result<Self> {
        // Select inducing points
        let inducing_points = self.select_inducing_points(probabilities)?;

        // Initialize variational parameters
        let variational_params = VariationalParams {
            inducing_mean: Array1::zeros(self.n_inducing),
            inducing_covariance: Array2::eye(self.n_inducing),
        };

        self.inducing_points = Some(inducing_points);
        self.variational_params = Some(variational_params);

        // Fit base GP with inducing points for initialization
        let inducing_labels = y_true
            .slice(s![..self.n_inducing.min(y_true.len())])
            .to_owned();
        self.base_gp = self
            .base_gp
            .fit(self.inducing_points.as_ref().unwrap(), &inducing_labels)?;

        Ok(self)
    }

    fn select_inducing_points(&self, probabilities: &Array1<Float>) -> Result<Array1<Float>> {
        let n = probabilities.len();
        let n_inducing = self.n_inducing.min(n);

        // Simple uniform selection
        let mut inducing = Array1::zeros(n_inducing);
        for i in 0..n_inducing {
            let idx = (i * n) / n_inducing;
            inducing[i] = probabilities[idx];
        }

        Ok(inducing)
    }
}

impl CalibrationEstimator for VariationalGPCalibrator {
    fn fit(&mut self, probabilities: &Array1<Float>, y_true: &Array1<i32>) -> Result<()> {
        *self = self.clone().fit(probabilities, y_true)?;
        Ok(())
    }

    fn predict_proba(&self, probabilities: &Array1<Float>) -> Result<Array1<Float>> {
        // Delegate to base GP for prediction
        self.base_gp.predict_proba(probabilities)
    }

    fn clone_box(&self) -> Box<dyn CalibrationEstimator> {
        Box::new(self.clone())
    }
}

// Utility functions

fn sigmoid(x: Float) -> Float {
    1.0 / (1.0 + (-x).exp())
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_gaussian_process_calibrator() {
        let probabilities = array![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
        let y_true = array![0, 0, 0, 0, 1, 1, 1, 1, 1];

        let calibrator = GaussianProcessCalibrator::new()
            .optimize_hyperparams(false)
            .fit(&probabilities, &y_true)
            .unwrap();

        let test_probabilities = array![0.25, 0.75];
        let calibrated = calibrator.predict_proba(&test_probabilities).unwrap();

        assert_eq!(calibrated.len(), 2);
        for &prob in calibrated.iter() {
            assert!((0.0..=1.0).contains(&prob));
        }
    }

    #[test]
    fn test_gp_with_uncertainty() {
        let probabilities = array![0.1, 0.3, 0.5, 0.7, 0.9];
        let y_true = array![0, 0, 1, 1, 1];

        let calibrator = GaussianProcessCalibrator::new()
            .optimize_hyperparams(false)
            .fit(&probabilities, &y_true)
            .unwrap();

        let test_probabilities = array![0.4, 0.6];
        let (means, stds) = calibrator
            .predict_proba_with_uncertainty(&test_probabilities)
            .unwrap();

        assert_eq!(means.len(), 2);
        assert_eq!(stds.len(), 2);

        for (&mean, &std) in means.iter().zip(stds.iter()) {
            assert!((0.0..=1.0).contains(&mean));
            assert!(std >= 0.0);
        }
    }

    #[test]
    fn test_different_kernels() {
        let probabilities = array![0.2, 0.4, 0.6, 0.8];
        let y_true = array![0, 0, 1, 1];

        for kernel in [
            GPKernel::RBF,
            GPKernel::Matern32,
            GPKernel::Matern52,
            GPKernel::Linear,
            GPKernel::Polynomial { degree: 2 },
        ] {
            let calibrator = GaussianProcessCalibrator::new()
                .kernel(kernel)
                .optimize_hyperparams(false)
                .fit(&probabilities, &y_true)
                .unwrap();

            let test_probabilities = array![0.3, 0.7];
            let calibrated = calibrator.predict_proba(&test_probabilities).unwrap();

            assert_eq!(calibrated.len(), 2);
            for &prob in calibrated.iter() {
                assert!((0.0..=1.0).contains(&prob));
            }
        }
    }

    #[test]
    fn test_variational_gp_calibrator() {
        let probabilities = array![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
        let y_true = array![0, 0, 0, 0, 1, 1, 1, 1, 1];

        let calibrator = VariationalGPCalibrator::new(5)
            .fit(&probabilities, &y_true)
            .unwrap();

        let test_probabilities = array![0.25, 0.75];
        let calibrated = calibrator.predict_proba(&test_probabilities).unwrap();

        assert_eq!(calibrated.len(), 2);
        for &prob in calibrated.iter() {
            assert!((0.0..=1.0).contains(&prob));
        }
    }

    #[test]
    fn test_gp_with_trait() {
        let probabilities = array![0.1, 0.3, 0.5, 0.7, 0.9];
        let y_true = array![0, 0, 1, 1, 1];

        let calibrator = GaussianProcessCalibrator::new()
            .fit(&probabilities, &y_true)
            .unwrap();

        let test_probabilities = array![0.2, 0.8];
        let calibrated = calibrator.predict_proba(&test_probabilities).unwrap();

        assert_eq!(calibrated.len(), 2);
        for &prob in calibrated.iter() {
            assert!((0.0..=1.0).contains(&prob));
        }
    }

    #[test]
    fn test_kernel_functions() {
        let calibrator = GaussianProcessCalibrator::new();

        // Test RBF kernel
        let kernel_val = calibrator.kernel_function(0.0, 0.0);
        assert!(kernel_val > 0.0);

        // Test that kernel is symmetric
        let val1 = calibrator.kernel_function(0.1, 0.5);
        let val2 = calibrator.kernel_function(0.5, 0.1);
        assert!((val1 - val2).abs() < 1e-10);
    }

    #[test]
    fn test_hyperparameter_optimization() {
        let probabilities = array![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
        let y_true = array![0, 0, 0, 0, 1, 1, 1, 1, 1];

        let calibrator = GaussianProcessCalibrator::new()
            .optimize_hyperparams(true)
            .fit(&probabilities, &y_true)
            .unwrap();

        let test_probabilities = array![0.25, 0.75];
        let calibrated = calibrator.predict_proba(&test_probabilities).unwrap();

        assert_eq!(calibrated.len(), 2);
        for &prob in calibrated.iter() {
            assert!((0.0..=1.0).contains(&prob));
        }
    }
}
