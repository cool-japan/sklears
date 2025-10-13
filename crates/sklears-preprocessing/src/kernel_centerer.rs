//! Kernel Centerer for centering kernel matrices
//!
//! This module provides the KernelCenterer transformer which centers a kernel matrix
//! in the feature space defined by the kernel.

use scirs2_core::ndarray::{Array1, Array2, Axis};
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Fit, Trained, Transform, Untrained},
    types::Float,
};
use std::marker::PhantomData;

/// KernelCenterer centers a kernel matrix
///
/// Let K(x, z) be a kernel matrix computed from samples x and z.
/// KernelCenterer computes the centered kernel matrix K_c as:
///
/// K_c(x, z) = K(x, z) - K_mean_x - K_mean_z + K_mean_all
///
/// where:
/// - K_mean_x is the mean of K along samples x
/// - K_mean_z is the mean of K along samples z  
/// - K_mean_all is the overall mean of K
#[derive(Debug, Clone)]
pub struct KernelCenterer<State = Untrained> {
    state: PhantomData<State>,
    // Fitted parameters
    k_train_mean_: Option<Array1<Float>>,
    k_train_mean_all_: Option<Float>,
}

impl KernelCenterer<Untrained> {
    /// Create a new KernelCenterer
    pub fn new() -> Self {
        Self {
            state: PhantomData,
            k_train_mean_: None,
            k_train_mean_all_: None,
        }
    }
}

impl Default for KernelCenterer<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Fit<Array2<Float>, (), Untrained> for KernelCenterer<Untrained> {
    type Fitted = KernelCenterer<Trained>;

    fn fit(self, k: &Array2<Float>, _y: &()) -> Result<Self::Fitted> {
        let n_samples = k.nrows();

        if n_samples == 0 {
            return Err(SklearsError::InvalidInput(
                "Cannot fit KernelCenterer on empty kernel matrix".to_string(),
            ));
        }

        if k.nrows() != k.ncols() {
            return Err(SklearsError::InvalidInput(format!(
                "Kernel matrix must be square, got shape ({}, {})",
                k.nrows(),
                k.ncols()
            )));
        }

        // Compute mean of each row (mean over training samples)
        let k_train_mean = k
            .mean_axis(Axis(1))
            .ok_or_else(|| SklearsError::InvalidInput("Failed to compute row means".to_string()))?;

        // Compute overall mean
        let k_train_mean_all = k_train_mean.mean().ok_or_else(|| {
            SklearsError::InvalidInput("Failed to compute overall mean".to_string())
        })?;

        Ok(KernelCenterer {
            state: PhantomData,
            k_train_mean_: Some(k_train_mean),
            k_train_mean_all_: Some(k_train_mean_all),
        })
    }
}

impl Transform<Array2<Float>> for KernelCenterer<Trained> {
    fn transform(&self, k: &Array2<Float>) -> Result<Array2<Float>> {
        let k_train_mean = self
            .k_train_mean_
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "transform".to_string(),
            })?;
        let k_train_mean_all = self
            .k_train_mean_all_
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "transform".to_string(),
            })?;

        let n_samples_train = k_train_mean.len();
        let n_samples_test = k.nrows();

        if k.ncols() != n_samples_train {
            return Err(SklearsError::InvalidInput(format!(
                "Kernel matrix has wrong number of columns. Expected {}, got {}",
                n_samples_train,
                k.ncols()
            )));
        }

        // Center the kernel matrix
        let mut k_centered = k.clone();

        // Subtract row means (mean over training samples for each test sample)
        let k_test_mean = k
            .mean_axis(Axis(1))
            .ok_or_else(|| SklearsError::InvalidInput("Failed to compute row means".to_string()))?;

        // Apply centering formula: K_c(x, z) = K(x, z) - K_mean_x - K_mean_z + K_mean_all
        for i in 0..n_samples_test {
            for j in 0..n_samples_train {
                k_centered[[i, j]] =
                    k[[i, j]] - k_test_mean[i] - k_train_mean[j] + k_train_mean_all;
            }
        }

        Ok(k_centered)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::arr2;

    #[test]
    fn test_kernel_centerer_fit_transform() {
        // Create a simple kernel matrix (e.g., linear kernel)
        let k_train = arr2(&[[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [3.0, 6.0, 9.0]]);

        let centerer = KernelCenterer::new();
        let fitted = centerer.fit(&k_train, &()).unwrap();

        // Transform the training kernel itself
        let k_centered = fitted.transform(&k_train).unwrap();

        // Check that the centered kernel has zero mean
        let mean = k_centered.mean().unwrap();
        assert_abs_diff_eq!(mean, 0.0, epsilon = 1e-10);

        // Check row and column means are zero
        for i in 0..k_centered.nrows() {
            let row_mean = k_centered.row(i).mean().unwrap();
            assert_abs_diff_eq!(row_mean, 0.0, epsilon = 1e-10);

            let col_mean = k_centered.column(i).mean().unwrap();
            assert_abs_diff_eq!(col_mean, 0.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_kernel_centerer_transform_new() {
        // Train kernel
        let k_train = arr2(&[[1.0, 2.0], [2.0, 4.0]]);

        // Test kernel (new samples vs training samples)
        let k_test = arr2(&[[1.5, 3.0], [2.5, 5.0], [3.5, 7.0]]);

        let centerer = KernelCenterer::new();
        let fitted = centerer.fit(&k_train, &()).unwrap();
        let k_test_centered = fitted.transform(&k_test).unwrap();

        // Verify shape
        assert_eq!(k_test_centered.shape(), &[3, 2]);
    }

    #[test]
    fn test_kernel_centerer_errors() {
        // Non-square kernel matrix
        let k_invalid = arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);

        let centerer = KernelCenterer::new();
        assert!(centerer.fit(&k_invalid, &()).is_err());

        // Empty kernel matrix
        let k_empty = Array2::<Float>::zeros((0, 0));
        let centerer = KernelCenterer::new();
        assert!(centerer.fit(&k_empty, &()).is_err());
    }
}
