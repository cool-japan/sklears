//! Core kernel trait for Gaussian Process models
//!
//! This module defines the fundamental Kernel trait that all kernel implementations must follow.
//! All implementations comply with SciRS2 Policy (never use rand/ndarray directly).

use scirs2_core::ndarray::{Array2, ArrayView1};
use sklears_core::error::Result as SklResult;

/// Kernel trait for Gaussian Process models
///
/// All kernel functions implement this trait to provide consistent interface
/// for computing kernel matrices, gradients, and parameter management.
pub trait Kernel: std::fmt::Debug + Send + Sync {
    /// Compute the kernel matrix between X1 and X2
    fn compute_kernel_matrix(
        &self,
        X1: &Array2<f64>,
        X2: Option<&Array2<f64>>,
    ) -> SklResult<Array2<f64>>;

    /// Compute the kernel between two points
    fn kernel(&self, x1: &ArrayView1<f64>, x2: &ArrayView1<f64>) -> f64;

    /// Get hyperparameters
    fn get_params(&self) -> Vec<f64>;

    /// Set hyperparameters
    fn set_params(&mut self, params: &[f64]) -> SklResult<()>;

    /// Clone the kernel
    fn clone_box(&self) -> Box<dyn Kernel>;

    /// Compute the gradient of the kernel matrix with respect to hyperparameters
    /// Returns a vector of gradient matrices, one for each hyperparameter
    fn compute_kernel_gradient(
        &self,
        X1: &Array2<f64>,
        X2: Option<&Array2<f64>>,
    ) -> SklResult<Vec<Array2<f64>>> {
        // Default implementation using finite differences
        let X2 = X2.unwrap_or(X1);
        let params = self.get_params();
        let n_params = params.len();
        let mut gradients = Vec::with_capacity(n_params);

        let epsilon = 1e-8;
        let base_matrix = self.compute_kernel_matrix(X1, Some(X2))?;

        for i in 0..n_params {
            let mut kernel_copy = self.clone_box();
            let mut params_plus = params.clone();
            params_plus[i] += epsilon;
            kernel_copy.set_params(&params_plus)?;

            let matrix_plus = kernel_copy.compute_kernel_matrix(X1, Some(X2))?;
            let gradient = (&matrix_plus - &base_matrix) / epsilon;
            gradients.push(gradient);
        }

        Ok(gradients)
    }

    /// Compute the gradient of the kernel function with respect to hyperparameters
    /// for a single pair of points
    fn kernel_gradient(&self, x1: &ArrayView1<f64>, x2: &ArrayView1<f64>) -> SklResult<Vec<f64>> {
        // Default implementation using finite differences
        let params = self.get_params();
        let n_params = params.len();
        let mut gradients = Vec::with_capacity(n_params);

        let epsilon = 1e-8;
        let base_value = self.kernel(x1, x2);

        for i in 0..n_params {
            let mut kernel_copy = self.clone_box();
            let mut params_plus = params.clone();
            params_plus[i] += epsilon;
            kernel_copy.set_params(&params_plus)?;

            let value_plus = kernel_copy.kernel(x1, x2);
            let gradient = (value_plus - base_value) / epsilon;
            gradients.push(gradient);
        }

        Ok(gradients)
    }
}

impl Clone for Box<dyn Kernel> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}
