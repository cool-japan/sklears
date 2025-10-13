//! Kernel Principal Component Analysis (Kernel PCA) for non-linear dimensionality reduction
//!
//! Kernel PCA extends classical PCA to non-linear data by first mapping the data
//! to a higher-dimensional feature space using kernel functions, then performing
//! PCA in that space. This is particularly useful as a preprocessing step for SVMs.

// TODO: Replace with scirs2-linalg
// use nalgebra::DMatrix;
use scirs2_core::ndarray::{s, Array1, Array2, Axis};
use sklears_core::{error::SklearsError, types::Float};

use crate::kernels::{create_kernel, Kernel, KernelType};

/// Kernel Principal Component Analysis for non-linear dimensionality reduction
#[derive(Debug)]
pub struct KernelPCA {
    /// Number of components to keep
    n_components: usize,
    /// Kernel function to use
    kernel: KernelType,
    /// Tolerance for eigenvalue computation
    tol: Float,
    /// Maximum number of iterations for iterative solvers
    max_iter: usize,
    /// Whether to center the kernel matrix
    fit_inverse_transform: bool,
    /// Eigenvalues (computed during fit)
    eigenvalues_: Option<Array1<Float>>,
    /// Eigenvectors (computed during fit)
    eigenvectors_: Option<Array2<Float>>,
    /// Centered kernel matrix eigenvectors for inverse transform
    alphas_: Option<Array2<Float>>,
    /// Training data for inverse transform
    x_fit_: Option<Array2<Float>>,
    /// Mean of kernel matrix columns (for centering)
    k_fit_cols_: Option<Array1<Float>>,
    /// Mean of all kernel matrix elements (for centering)
    k_fit_all_: Option<Float>,
}

impl Default for KernelPCA {
    fn default() -> Self {
        Self::new()
    }
}

impl KernelPCA {
    /// Create a new KernelPCA instance with default parameters
    pub fn new() -> Self {
        Self {
            n_components: 2,
            kernel: KernelType::Rbf { gamma: 1.0 },
            tol: 1e-10,
            max_iter: 1000,
            fit_inverse_transform: false,
            eigenvalues_: None,
            eigenvectors_: None,
            alphas_: None,
            x_fit_: None,
            k_fit_cols_: None,
            k_fit_all_: None,
        }
    }

    /// Set the number of components
    pub fn with_n_components(mut self, n_components: usize) -> Self {
        self.n_components = n_components;
        self
    }

    /// Set the kernel function
    pub fn with_kernel(mut self, kernel: KernelType) -> Self {
        self.kernel = kernel;
        self
    }

    /// Set the tolerance for eigenvalue computation
    pub fn with_tol(mut self, tol: Float) -> Self {
        self.tol = tol;
        self
    }

    /// Set the maximum number of iterations
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Enable fitting of inverse transform
    pub fn with_fit_inverse_transform(mut self, fit_inverse_transform: bool) -> Self {
        self.fit_inverse_transform = fit_inverse_transform;
        self
    }

    /// Center the kernel matrix
    fn center_kernel_matrix(
        &self,
        k: &Array2<Float>,
        k_train: Option<&Array2<Float>>,
    ) -> Array2<Float> {
        let n_test = k.nrows();
        let n_train = k.ncols();

        if let (Some(k_fit_cols), Some(k_fit_all)) = (&self.k_fit_cols_, &self.k_fit_all_) {
            // For transform: center using training statistics
            let mut k_centered = k.clone();

            // Subtract column means from training data
            for i in 0..n_test {
                for j in 0..n_train {
                    k_centered[[i, j]] -= k_fit_cols[j];
                }
            }

            // Subtract column means (compute for test data)
            let test_row_means = k.mean_axis(Axis(1)).unwrap();
            for i in 0..n_test {
                for j in 0..n_train {
                    k_centered[[i, j]] -= test_row_means[i];
                }
            }

            // Add back grand mean
            k_centered + *k_fit_all
        } else if let Some(k_train) = k_train {
            // For fit: compute centering statistics
            let n = k_train.nrows();

            // Compute row means
            let row_means = k_train.mean_axis(Axis(1)).unwrap();

            // Compute column means
            let col_means = k_train.mean_axis(Axis(0)).unwrap();

            // Compute grand mean
            let grand_mean = k_train.mean().unwrap();

            // Center the matrix
            let mut k_centered = Array2::<Float>::zeros((n, n));
            for i in 0..n {
                for j in 0..n {
                    k_centered[[i, j]] = k_train[[i, j]] - row_means[i] - col_means[j] + grand_mean;
                }
            }

            k_centered
        } else {
            // Fallback: no centering
            k.clone()
        }
    }

    /// Fit the kernel PCA model to the data
    pub fn fit(&mut self, x: &Array2<Float>) -> Result<&mut Self, SklearsError> {
        let n_samples = x.nrows();

        if self.n_components > n_samples {
            return Err(SklearsError::InvalidInput(format!(
                "n_components ({}) cannot be larger than n_samples ({})",
                self.n_components, n_samples
            )));
        }

        // Create kernel instance
        let kernel = create_kernel(self.kernel.clone());

        // Compute kernel matrix
        let k = kernel.compute_matrix(x, x);

        // Store training statistics for centering
        self.k_fit_cols_ = Some(k.mean_axis(Axis(0)).unwrap()); // Column means
        self.k_fit_all_ = Some(k.mean().unwrap());

        // Center the kernel matrix
        let k_centered = self.center_kernel_matrix(&k, Some(&k));

        // Convert to nalgebra for eigenvalue decomposition
        let k_nalgebra = DMatrix::from_iterator(
            k_centered.nrows(),
            k_centered.ncols(),
            k_centered.iter().cloned(),
        );

        // Compute eigenvalue decomposition
        let eigen = k_nalgebra.symmetric_eigen();

        // Extract eigenvalues and eigenvectors
        let eigenvalues = Array1::from_vec(eigen.eigenvalues.as_slice().to_vec());
        let eigenvectors_matrix = eigen.eigenvectors;

        // Convert eigenvectors back to ndarray
        let mut eigenvectors = Array2::zeros((n_samples, n_samples));
        for i in 0..n_samples {
            for j in 0..n_samples {
                eigenvectors[[i, j]] = eigenvectors_matrix[(i, j)];
            }
        }

        // Sort eigenvalues and eigenvectors in descending order
        let mut indices: Vec<usize> = (0..n_samples).collect();
        indices.sort_by(|&i, &j| eigenvalues[j].partial_cmp(&eigenvalues[i]).unwrap());

        // Reorder based on sorted indices
        let sorted_eigenvalues = Array1::from_shape_fn(n_samples, |i| eigenvalues[indices[i]]);
        let mut sorted_eigenvectors = Array2::zeros((n_samples, n_samples));
        for i in 0..n_samples {
            for j in 0..n_samples {
                sorted_eigenvectors[[j, i]] = eigenvectors[[j, indices[i]]];
            }
        }

        // Keep only positive eigenvalues above tolerance
        let mut n_valid = 0;
        for i in 0..n_samples {
            if sorted_eigenvalues[i] > self.tol {
                n_valid += 1;
            } else {
                break;
            }
        }

        // Limit to requested number of components
        let n_keep = std::cmp::min(self.n_components, n_valid);

        // Store results
        self.eigenvalues_ = Some(sorted_eigenvalues.slice(s![..n_keep]).to_owned());
        self.eigenvectors_ = Some(sorted_eigenvectors.slice(s![.., ..n_keep]).to_owned());

        // Normalize eigenvectors by square root of eigenvalues
        let mut alphas = sorted_eigenvectors.slice(s![.., ..n_keep]).to_owned();
        for i in 0..n_keep {
            let norm = sorted_eigenvalues[i].sqrt();
            if norm > self.tol {
                alphas.column_mut(i).mapv_inplace(|x| x / norm);
            }
        }
        self.alphas_ = Some(alphas);

        // Store training data if inverse transform is needed
        if self.fit_inverse_transform {
            self.x_fit_ = Some(x.clone());
        }

        Ok(self)
    }

    /// Transform data to the kernel PCA space
    pub fn transform(&self, x: &Array2<Float>) -> Result<Array2<Float>, SklearsError> {
        let alphas = self
            .alphas_
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "transform".to_string(),
            })?;

        let x_fit = self
            .x_fit_
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "transform with stored training data".to_string(),
            })?;

        let n_test = x.nrows();
        let n_train = x_fit.nrows();
        let n_components = alphas.ncols();

        // Create kernel instance
        let kernel = create_kernel(self.kernel.clone());

        // Compute kernel matrix between test and training data (shape: n_test x n_train)
        let k = kernel.compute_matrix(x, x_fit);

        // Center the kernel matrix using training statistics
        let k_fit_cols = self
            .k_fit_cols_
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "transform with centering statistics".to_string(),
            })?;
        let k_fit_all = self
            .k_fit_all_
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "transform with centering statistics".to_string(),
            })?;

        // Center the kernel matrix
        let mut k_centered = Array2::zeros((n_test, n_train));
        for i in 0..n_test {
            // Compute row mean for test sample i
            let test_row_mean = k.row(i).mean().unwrap();

            for j in 0..n_train {
                // Apply centering: K_ij - mean_j(train) - mean_i(test) + grand_mean(train)
                k_centered[[i, j]] = k[[i, j]] - k_fit_cols[j] - test_row_mean + *k_fit_all;
            }
        }

        // Project onto principal components
        let mut x_transformed = Array2::zeros((n_test, n_components));
        for i in 0..n_test {
            for j in 0..n_components {
                x_transformed[[i, j]] = k_centered.row(i).dot(&alphas.column(j));
            }
        }

        Ok(x_transformed)
    }

    /// Fit the model and transform the data
    pub fn fit_transform(&mut self, x: &Array2<Float>) -> Result<Array2<Float>, SklearsError> {
        self.fit(x)?;

        let alphas = self.alphas_.as_ref().unwrap();
        let n_samples = x.nrows();
        let n_components = alphas.ncols();

        // Create kernel instance
        let kernel = create_kernel(self.kernel.clone());

        // Compute kernel matrix
        let k = kernel.compute_matrix(x, x);

        // Center the kernel matrix
        let k_centered = self.center_kernel_matrix(&k, Some(&k));

        // Project onto principal components
        let mut x_transformed = Array2::zeros((n_samples, n_components));
        for i in 0..n_samples {
            for j in 0..n_components {
                x_transformed[[i, j]] = k_centered.row(i).dot(&alphas.column(j));
            }
        }

        Ok(x_transformed)
    }

    /// Get the eigenvalues
    pub fn eigenvalues(&self) -> Option<&Array1<Float>> {
        self.eigenvalues_.as_ref()
    }

    /// Get the eigenvectors
    pub fn eigenvectors(&self) -> Option<&Array2<Float>> {
        self.eigenvectors_.as_ref()
    }

    /// Get explained variance ratio
    pub fn explained_variance_ratio(&self) -> Option<Array1<Float>> {
        self.eigenvalues_.as_ref().map(|eigenvals| {
            let total_var = eigenvals.sum();
            eigenvals.mapv(|x| x / total_var)
        })
    }
}

/// Builder pattern for KernelPCA
pub struct KernelPCABuilder {
    kernel_pca: KernelPCA,
}

impl KernelPCABuilder {
    pub fn new() -> Self {
        Self {
            kernel_pca: KernelPCA::new(),
        }
    }

    pub fn n_components(mut self, n_components: usize) -> Self {
        self.kernel_pca = self.kernel_pca.with_n_components(n_components);
        self
    }

    pub fn kernel(mut self, kernel: KernelType) -> Self {
        self.kernel_pca = self.kernel_pca.with_kernel(kernel);
        self
    }

    pub fn tol(mut self, tol: Float) -> Self {
        self.kernel_pca = self.kernel_pca.with_tol(tol);
        self
    }

    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.kernel_pca = self.kernel_pca.with_max_iter(max_iter);
        self
    }

    pub fn fit_inverse_transform(mut self, fit_inverse_transform: bool) -> Self {
        self.kernel_pca = self
            .kernel_pca
            .with_fit_inverse_transform(fit_inverse_transform);
        self
    }

    pub fn build(self) -> KernelPCA {
        self.kernel_pca
    }
}

impl Default for KernelPCABuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::Array;

    #[test]
    fn test_kernel_pca_basic() {
        // Create simple 2D data that should be separable with kernel PCA
        let x = Array::from_shape_vec(
            (6, 2),
            vec![
                0.0, 0.0, 1.0, 1.0, 2.0, 2.0, -1.0, -1.0, -2.0, -2.0, 0.5, -0.5,
            ],
        )
        .unwrap();

        let mut kpca = KernelPCA::new()
            .with_n_components(2)
            .with_kernel(KernelType::Rbf { gamma: 1.0 });

        let result = kpca.fit_transform(&x);
        assert!(result.is_ok());

        let transformed = result.unwrap();
        assert_eq!(transformed.shape(), &[6, 2]);

        // Check that eigenvalues are available
        assert!(kpca.eigenvalues().is_some());
        let eigenvals = kpca.eigenvalues().unwrap();
        assert_eq!(eigenvals.len(), 2);

        // Eigenvalues should be positive and in descending order
        for i in 0..eigenvals.len() {
            assert!(eigenvals[i] > 0.0);
            if i > 0 {
                assert!(eigenvals[i - 1] >= eigenvals[i]);
            }
        }
    }

    #[test]
    fn test_kernel_pca_polynomial_kernel() {
        let x = Array::from_shape_vec((4, 2), vec![1.0, 1.0, 2.0, 2.0, -1.0, -1.0, -2.0, -2.0])
            .unwrap();

        let mut kpca = KernelPCA::new()
            .with_n_components(2)
            .with_kernel(KernelType::Polynomial {
                gamma: 1.0,
                degree: 2.0,
                coef0: 1.0,
            });

        let result = kpca.fit_transform(&x);
        assert!(result.is_ok());

        let transformed = result.unwrap();
        assert_eq!(transformed.shape(), &[4, 2]);
    }

    #[test]
    fn test_kernel_pca_transform_new_data() {
        let x_train =
            Array::from_shape_vec((4, 2), vec![0.0, 0.0, 1.0, 1.0, -1.0, -1.0, 0.0, 1.0]).unwrap();

        let x_test = Array::from_shape_vec((2, 2), vec![0.5, 0.5, -0.5, -0.5]).unwrap();

        let mut kpca = KernelPCA::new()
            .with_n_components(2)
            .with_kernel(KernelType::Rbf { gamma: 1.0 })
            .with_fit_inverse_transform(true);

        // Fit on training data
        kpca.fit(&x_train).unwrap();

        // Transform test data
        let result = kpca.transform(&x_test);
        assert!(result.is_ok());

        let transformed = result.unwrap();
        assert_eq!(transformed.shape(), &[2, 2]);
    }

    #[test]
    fn test_kernel_pca_builder() {
        let kpca = KernelPCABuilder::new()
            .n_components(3)
            .kernel(KernelType::Linear)
            .tol(1e-12)
            .max_iter(500)
            .fit_inverse_transform(true)
            .build();

        assert_eq!(kpca.n_components, 3);
        assert_eq!(kpca.tol, 1e-12);
        assert_eq!(kpca.max_iter, 500);
        assert_eq!(kpca.fit_inverse_transform, true);
    }

    #[test]
    fn test_kernel_pca_explained_variance() {
        let x = Array::from_shape_vec(
            (5, 2),
            vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0, -1.0, -1.0, -2.0, -2.0],
        )
        .unwrap();

        let mut kpca = KernelPCA::new()
            .with_n_components(2)
            .with_kernel(KernelType::Rbf { gamma: 1.0 });

        kpca.fit_transform(&x).unwrap();

        let explained_var_ratio = kpca.explained_variance_ratio();
        assert!(explained_var_ratio.is_some());

        let ratios = explained_var_ratio.unwrap();
        let sum: Float = ratios.sum();
        assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-10);

        // First component should explain more variance than the second
        assert!(ratios[0] >= ratios[1]);
    }

    #[test]
    fn test_kernel_pca_invalid_n_components() {
        let x = Array::from_shape_vec((3, 2), vec![0.0, 0.0, 1.0, 1.0, -1.0, -1.0]).unwrap();

        let mut kpca = KernelPCA::new().with_n_components(5); // More components than samples

        let result = kpca.fit(&x);
        assert!(result.is_err());
    }
}
