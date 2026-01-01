//! Principal Component Analysis Integration for Covariance Estimation
//!
//! This module provides PCA-based covariance estimation methods, including standard PCA,
//! incremental PCA, kernel PCA, robust PCA, and sparse PCA variants for covariance estimation.
//! Unlike the factor model's PCA method, this focuses specifically on PCA-based covariance
//! with advanced features and optimizations.

use scirs2_core::ndarray::{s, Array1, Array2, ArrayView1, ArrayView2, Axis};
use scirs2_core::Distribution;
use scirs2_core::StandardNormal;
use scirs2_linalg::compat::ArrayLinalgExt;
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Transform, Untrained},
};

/// PCA-based covariance estimator
///
/// Provides various PCA methods for covariance estimation, including standard PCA,
/// incremental PCA for large datasets, kernel PCA for non-linear relationships,
/// and robust/sparse variants for handling outliers and high-dimensional data.
#[derive(Debug, Clone)]
pub struct PCACovariance<S = Untrained> {
    state: S,
    /// Number of principal components to use
    n_components: Option<usize>,
    /// Method for PCA computation
    method: PCAMethod,
    /// Whether to center the data
    center: bool,
    /// Whether to scale the data
    scale: bool,
    /// Tolerance for eigenvalue decomposition
    tol: f64,
    /// Random state for reproducible results
    random_state: Option<u64>,
    /// Batch size for incremental PCA
    batch_size: Option<usize>,
    /// Kernel function for kernel PCA
    kernel: Option<KernelFunction>,
    /// Regularization parameter for robust variants
    regularization: f64,
    /// Sparsity parameter for sparse PCA
    sparsity: f64,
}

/// Methods for PCA computation
#[derive(Debug, Clone)]
pub enum PCAMethod {
    Standard,
    Incremental,
    Kernel,
    Robust,
    Sparse,
    Probabilistic,
}

/// Kernel functions for kernel PCA
#[derive(Debug, Clone)]
pub enum KernelFunction {
    /// Linear kernel (equivalent to standard PCA)
    Linear,
    /// Polynomial kernel with degree
    Polynomial { degree: usize, coef: f64 },
    /// RBF (Gaussian) kernel with gamma parameter
    RBF { gamma: f64 },
    /// Sigmoid kernel with parameters
    Sigmoid { gamma: f64, coef: f64 },
}

/// Trained PCA Covariance state
#[derive(Debug, Clone)]
pub struct PCACovarianceTrained {
    /// Estimated covariance matrix
    covariance: Array2<f64>,
    /// Precision matrix (inverse covariance)
    precision: Option<Array2<f64>>,
    /// Principal components (loadings)
    components: Array2<f64>,
    /// Explained variance for each component
    explained_variance: Array1<f64>,
    /// Explained variance ratio for each component
    explained_variance_ratio: Array1<f64>,
    /// Singular values from SVD
    singular_values: Array1<f64>,
    /// Mean of the training data
    mean: Array1<f64>,
    /// Scaling factors (if scaling was applied)
    scale_factors: Option<Array1<f64>>,
    /// Number of components used
    n_components: usize,
    /// Method used for PCA
    method: PCAMethod,
    /// Number of observations used for training
    n_samples: usize,
    /// Number of features
    n_features: usize,
    /// Cumulative explained variance ratio
    cumulative_variance_ratio: Array1<f64>,
    /// Noise variance (for probabilistic PCA)
    noise_variance: Option<f64>,
    /// Kernel matrix (for kernel PCA)
    kernel_matrix: Option<Array2<f64>>,
    /// Support vectors (for sparse methods)
    support_vectors: Option<Array2<f64>>,
}

impl Default for PCACovariance {
    fn default() -> Self {
        Self::new()
    }
}

impl PCACovariance {
    /// Creates a new PCA covariance estimator
    pub fn new() -> Self {
        Self {
            state: Untrained,
            n_components: None,
            method: PCAMethod::Standard,
            center: true,
            scale: false,
            tol: 1e-8,
            random_state: None,
            batch_size: None,
            kernel: None,
            regularization: 0.01,
            sparsity: 0.1,
        }
    }

    /// Sets the number of components
    pub fn n_components(mut self, n_components: usize) -> Self {
        self.n_components = Some(n_components);
        self
    }

    /// Sets the PCA method
    pub fn method(mut self, method: PCAMethod) -> Self {
        self.method = method;
        self
    }

    /// Sets whether to center the data
    pub fn center(mut self, center: bool) -> Self {
        self.center = center;
        self
    }

    /// Sets whether to scale the data
    pub fn scale(mut self, scale: bool) -> Self {
        self.scale = scale;
        self
    }

    /// Sets the tolerance for eigenvalue decomposition
    pub fn tolerance(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Sets the random state
    pub fn random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }

    /// Sets the batch size for incremental PCA
    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = Some(batch_size);
        self
    }

    /// Sets the kernel function for kernel PCA
    pub fn kernel(mut self, kernel: KernelFunction) -> Self {
        self.kernel = Some(kernel);
        self
    }

    /// Sets the regularization parameter
    pub fn regularization(mut self, regularization: f64) -> Self {
        self.regularization = regularization;
        self
    }

    /// Sets the sparsity parameter
    pub fn sparsity(mut self, sparsity: f64) -> Self {
        self.sparsity = sparsity;
        self
    }
}

#[derive(Debug, Clone)]
pub struct PCAConfig {
    pub method: PCAMethod,
    pub n_components: usize,
    pub whiten: bool,
    pub regularization: f64,
    pub sparsity: f64,
    pub random_state: Option<u64>,
    pub kernel: Option<KernelFunction>,
}

impl Estimator for PCACovariance {
    type Config = PCAConfig;
    type Error = SklearsError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        static CONFIG: std::sync::OnceLock<PCAConfig> = std::sync::OnceLock::new();
        CONFIG.get_or_init(|| PCAConfig {
            method: PCAMethod::Standard,
            n_components: 2,
            whiten: false,
            regularization: 1e-8,
            sparsity: 0.1,
            random_state: None,
            kernel: None,
        })
    }
}

impl Fit<ArrayView2<'_, f64>, ()> for PCACovariance {
    type Fitted = PCACovariance<PCACovarianceTrained>;

    fn fit(self, x: &ArrayView2<'_, f64>, _y: &()) -> SklResult<Self::Fitted> {
        let (n_samples, n_features) = x.dim();

        if n_samples < 2 {
            return Err(SklearsError::InvalidInput(
                "PCA requires at least 2 samples".to_string(),
            ));
        }

        let n_components = self.n_components.unwrap_or(n_features.min(n_samples));

        if n_components > n_features.min(n_samples) {
            return Err(SklearsError::InvalidInput(
                "Number of components cannot exceed min(n_samples, n_features)".to_string(),
            ));
        }

        // Center and optionally scale the data
        let mean = if self.center {
            x.mean_axis(Axis(0)).unwrap()
        } else {
            Array1::zeros(n_features)
        };

        let mut x_centered = x.to_owned();
        if self.center {
            for mut row in x_centered.axis_iter_mut(Axis(0)) {
                row -= &mean;
            }
        }

        let scale_factors = if self.scale {
            let std_dev = x_centered
                .mapv(|x| x * x)
                .mean_axis(Axis(0))
                .unwrap()
                .mapv(|x| x.sqrt());
            for mut row in x_centered.axis_iter_mut(Axis(0)) {
                for (i, val) in row.iter_mut().enumerate() {
                    if std_dev[i] > 1e-10 {
                        *val /= std_dev[i];
                    }
                }
            }
            Some(std_dev)
        } else {
            None
        };

        // Perform PCA based on the selected method
        let (
            components,
            explained_variance,
            singular_values,
            precision,
            noise_variance,
            kernel_matrix,
        ) = match &self.method {
            PCAMethod::Standard => self.standard_pca(&x_centered, n_components)?,
            PCAMethod::Incremental => self.incremental_pca(&x_centered, n_components)?,
            PCAMethod::Kernel => self.kernel_pca(&x_centered, n_components)?,
            PCAMethod::Robust => self.robust_pca(&x_centered, n_components)?,
            PCAMethod::Sparse => self.sparse_pca(&x_centered, n_components)?,
            PCAMethod::Probabilistic => self.probabilistic_pca(&x_centered, n_components)?,
        };

        // Compute covariance matrix from components
        let covariance = if components.ncols() == n_features {
            // Full rank case
            components.t().dot(&components)
        } else {
            // Reduced rank case: C = W * W^T + sigma^2 * I (for probabilistic PCA)
            let ww_t = components.t().dot(&components);
            if let Some(sigma2) = noise_variance {
                ww_t + Array2::<f64>::eye(n_features) * sigma2
            } else {
                ww_t
            }
        };

        // Compute explained variance ratio
        let total_variance = explained_variance.sum();
        let explained_variance_ratio = explained_variance.mapv(|x| x / total_variance);

        // Compute cumulative explained variance ratio
        let mut cumulative_variance_ratio = Array1::zeros(n_components);
        let mut cumsum = 0.0;
        for (i, &var_ratio) in explained_variance_ratio.iter().enumerate() {
            cumsum += var_ratio;
            cumulative_variance_ratio[i] = cumsum;
        }

        let trained_state = PCACovarianceTrained {
            covariance: covariance.clone(),
            precision,
            components,
            explained_variance,
            explained_variance_ratio,
            singular_values,
            mean,
            scale_factors,
            n_components,
            method: self.method.clone(),
            n_samples,
            n_features,
            cumulative_variance_ratio,
            noise_variance,
            kernel_matrix,
            support_vectors: None, // Will be set by sparse methods if needed
        };

        Ok(PCACovariance {
            state: trained_state,
            n_components: self.n_components,
            method: self.method,
            center: self.center,
            scale: self.scale,
            tol: self.tol,
            random_state: self.random_state,
            batch_size: self.batch_size,
            kernel: self.kernel,
            regularization: self.regularization,
            sparsity: self.sparsity,
        })
    }
}

impl PCACovariance {
    /// Standard PCA using SVD
    fn standard_pca(
        &self,
        x: &Array2<f64>,
        n_components: usize,
    ) -> SklResult<(
        Array2<f64>,
        Array1<f64>,
        Array1<f64>,
        Option<Array2<f64>>,
        Option<f64>,
        Option<Array2<f64>>,
    )> {
        let (u, s, vt) = x
            .svd(true)
            .map_err(|e| SklearsError::NumericalError(format!("SVD failed: {}", e)))?;

        // Take first n_components
        let components = vt.slice(s![..n_components, ..]).to_owned();
        let singular_values = s.slice(s![..n_components]).to_owned();

        // Explained variance = (singular_values^2) / (n_samples - 1)
        let n_samples = x.nrows();
        let explained_variance = singular_values.mapv(|x| x * x / (n_samples - 1) as f64);

        // Compute precision matrix if possible
        let precision = if n_components == x.ncols() {
            match components.dot(&components.t()).inv() {
                Ok(inv) => Some(inv),
                Err(_) => None,
            }
        } else {
            None
        };

        Ok((
            components,
            explained_variance,
            singular_values,
            precision,
            None,
            None,
        ))
    }

    /// Incremental PCA for large datasets
    fn incremental_pca(
        &self,
        x: &Array2<f64>,
        n_components: usize,
    ) -> SklResult<(
        Array2<f64>,
        Array1<f64>,
        Array1<f64>,
        Option<Array2<f64>>,
        Option<f64>,
        Option<Array2<f64>>,
    )> {
        let batch_size = self.batch_size.unwrap_or(100.min(x.nrows()));
        let (n_samples, n_features) = x.dim();

        // Initialize running statistics
        let mut mean = Array1::zeros(n_features);
        let mut components = Array2::zeros((n_components, n_features));
        let mut explained_variance = Array1::zeros(n_components);
        let mut n_samples_seen = 0;

        // Process data in batches
        for chunk in x.axis_chunks_iter(Axis(0), batch_size) {
            let batch_size = chunk.nrows();
            let batch_mean = chunk.mean_axis(Axis(0)).unwrap();

            // Update running mean
            let new_n_samples = n_samples_seen + batch_size;
            mean = (mean * n_samples_seen as f64 + batch_mean * batch_size as f64)
                / new_n_samples as f64;

            // Center current batch
            let mut chunk_centered = chunk.to_owned();
            for mut row in chunk_centered.axis_iter_mut(Axis(0)) {
                row -= &mean;
            }

            // Update components using incremental SVD
            let (_, s, vt) = chunk_centered
                .svd(false)
                .map_err(|e| SklearsError::NumericalError(format!("SVD failed: {}", e)))?;
            let new_components = vt.slice(s![..n_components.min(vt.nrows()), ..]).to_owned();

            // Merge with existing components (simplified incremental update)
            if n_samples_seen == 0 {
                components = new_components;
                explained_variance = s
                    .slice(s![..n_components])
                    .mapv(|x| x * x / (batch_size as f64 - 1.0));
            } else {
                // Weighted average of components (simplified)
                let weight_old = n_samples_seen as f64 / new_n_samples as f64;
                let weight_new = batch_size as f64 / new_n_samples as f64;

                components = &components * weight_old + &new_components * weight_new;
                let new_explained_variance = s
                    .slice(s![..n_components])
                    .mapv(|x| x * x / (batch_size as f64 - 1.0));
                explained_variance =
                    &explained_variance * weight_old + &new_explained_variance * weight_new;
            }

            n_samples_seen = new_n_samples;
        }

        let singular_values = explained_variance.mapv(|x| (x * (n_samples as f64 - 1.0)).sqrt());

        Ok((
            components,
            explained_variance,
            singular_values,
            None,
            None,
            None,
        ))
    }

    /// Kernel PCA for non-linear relationships
    fn kernel_pca(
        &self,
        x: &Array2<f64>,
        n_components: usize,
    ) -> SklResult<(
        Array2<f64>,
        Array1<f64>,
        Array1<f64>,
        Option<Array2<f64>>,
        Option<f64>,
        Option<Array2<f64>>,
    )> {
        let kernel_func = self.kernel.as_ref().unwrap_or(&KernelFunction::Linear);

        // Compute kernel matrix
        let n_samples = x.nrows();
        let mut kernel_matrix = Array2::zeros((n_samples, n_samples));

        for i in 0..n_samples {
            for j in i..n_samples {
                let k_val = self.compute_kernel(&x.row(i), &x.row(j), kernel_func);
                kernel_matrix[[i, j]] = k_val;
                kernel_matrix[[j, i]] = k_val;
            }
        }

        // Center the kernel matrix
        let row_means = kernel_matrix.mean_axis(Axis(1)).unwrap();
        let total_mean = kernel_matrix.mean().unwrap();

        for i in 0..n_samples {
            for j in 0..n_samples {
                kernel_matrix[[i, j]] =
                    kernel_matrix[[i, j]] - row_means[i] - row_means[j] + total_mean;
            }
        }

        // Eigenvalue decomposition of centered kernel matrix
        let (eigenvalues, eigenvectors) = kernel_matrix.eig().map_err(|e| {
            SklearsError::NumericalError(format!("Eigenvalue decomposition failed: {}", e))
        })?;

        // Sort eigenvalues and eigenvectors in descending order
        // Extract real parts for symmetric matrices (eigenvalues should be real)
        let mut eigen_pairs: Vec<(f64, Array1<f64>)> = eigenvalues
            .iter()
            .zip(eigenvectors.axis_iter(Axis(1)))
            .map(|(&val, vec)| (val.re, vec.mapv(|x| x.re)))
            .collect();

        eigen_pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

        // Take first n_components
        let eigenvalues: Array1<f64> = eigen_pairs
            .iter()
            .take(n_components)
            .map(|(val, _)| *val)
            .collect();

        let eigenvectors: Array2<f64> = Array2::from_shape_vec(
            (n_samples, n_components),
            eigen_pairs
                .iter()
                .take(n_components)
                .flat_map(|(_, vec)| vec.iter())
                .cloned()
                .collect(),
        )
        .map_err(|_| {
            SklearsError::NumericalError("Failed to create eigenvector matrix".to_string())
        })?;

        // For kernel PCA, components are the eigenvectors scaled by sqrt(eigenvalues)
        let mut components = Array2::zeros((n_components, x.ncols()));
        for i in 0..n_components {
            if eigenvalues[i] > 0.0 {
                let alpha = eigenvectors.column(i).to_owned() / eigenvalues[i].sqrt();
                // Project back to feature space (approximation for finite kernel methods)
                for j in 0..x.ncols() {
                    components[[i, j]] = alpha
                        .iter()
                        .zip(x.axis_iter(Axis(0)))
                        .map(|(&a, x_row)| a * x_row[j])
                        .sum();
                }
            }
        }

        let explained_variance = eigenvalues.mapv(|x| x.max(0.0));
        let singular_values = explained_variance.mapv(|x| x.sqrt());

        Ok((
            components,
            explained_variance,
            singular_values,
            None,
            None,
            Some(kernel_matrix),
        ))
    }

    /// Robust PCA using iterative reweighting
    fn robust_pca(
        &self,
        x: &Array2<f64>,
        n_components: usize,
    ) -> SklResult<(
        Array2<f64>,
        Array1<f64>,
        Array1<f64>,
        Option<Array2<f64>>,
        Option<f64>,
        Option<Array2<f64>>,
    )> {
        let max_iter = 50;
        let tolerance = self.tol;
        let (n_samples, n_features) = x.dim();

        // Initialize weights
        let mut weights = Array1::ones(n_samples);
        let mut prev_components = Array2::zeros((n_components, n_features));

        for iter in 0..max_iter {
            // Weighted covariance computation
            let mut weighted_x = x.to_owned();
            for (i, mut row) in weighted_x.axis_iter_mut(Axis(0)).enumerate() {
                row *= weights[i];
            }

            // Perform standard PCA on weighted data
            let (u, s, vt) = weighted_x
                .svd(true)
                .map_err(|e| SklearsError::NumericalError(format!("SVD failed: {}", e)))?;

            let components = vt.slice(s![..n_components, ..]).to_owned();

            // Update weights based on residuals
            for i in 0..n_samples {
                let x_row = x.row(i);
                let projected = components.dot(&x_row);
                let reconstructed = components.t().dot(&projected);
                let residual = (&x_row - &reconstructed).mapv(|x| x * x).sum().sqrt();

                // Huber-type reweighting
                let threshold = self.regularization;
                weights[i] = if residual <= threshold {
                    1.0
                } else {
                    threshold / residual
                };
            }

            // Check convergence
            let component_diff = (&components - &prev_components)
                .mapv(|x| x * x)
                .sum()
                .sqrt();
            if component_diff < tolerance {
                break;
            }

            prev_components = components.clone();
        }

        // Final SVD with converged weights
        let mut weighted_x = x.to_owned();
        for (i, mut row) in weighted_x.axis_iter_mut(Axis(0)).enumerate() {
            row *= weights[i];
        }

        let (_, s, vt) = weighted_x
            .svd(false)
            .map_err(|e| SklearsError::NumericalError(format!("SVD failed: {}", e)))?;

        let components = vt.slice(s![..n_components, ..]).to_owned();
        let singular_values = s.slice(s![..n_components]).to_owned();
        let explained_variance = singular_values.mapv(|x| x * x / (n_samples as f64 - 1.0));

        Ok((
            components,
            explained_variance,
            singular_values,
            None,
            None,
            None,
        ))
    }

    /// Sparse PCA using L1 regularization
    fn sparse_pca(
        &self,
        x: &Array2<f64>,
        n_components: usize,
    ) -> SklResult<(
        Array2<f64>,
        Array1<f64>,
        Array1<f64>,
        Option<Array2<f64>>,
        Option<f64>,
        Option<Array2<f64>>,
    )> {
        let max_iter = 100;
        let tolerance = self.tol;
        let (n_samples, n_features) = x.dim();
        let sparsity = self.sparsity;

        // Initialize components randomly
        let mut components = Array2::zeros((n_components, n_features));
        for i in 0..n_components {
            for j in 0..n_features {
                components[[i, j]] = (i + j) as f64 * 0.01; // Simple initialization
            }
        }

        // Iterative sparse PCA
        for _iter in 0..max_iter {
            let prev_components = components.clone();

            // Update each component
            for k in 0..n_components {
                // Deflation: remove influence of previous components
                let mut residual = x.to_owned();
                for j in 0..k {
                    let scores = x.dot(&components.row(j));
                    for i in 0..n_samples {
                        let contrib = components.row(j).to_owned() * scores[i];
                        for (res, &c) in residual.row_mut(i).iter_mut().zip(contrib.iter()) {
                            *res -= c;
                        }
                    }
                }

                // Find sparse principal component using soft thresholding
                let covariance = residual.t().dot(&residual) / n_samples as f64;
                let (eigenvalues, eigenvectors) = covariance.eig().map_err(|e| {
                    SklearsError::NumericalError(format!("Eigenvalue decomposition failed: {}", e))
                })?;

                // Find largest eigenvalue and corresponding eigenvector
                // Extract real parts for comparison (eigenvalues should be real for symmetric matrices)
                let max_idx = eigenvalues
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.re.partial_cmp(&b.re).unwrap())
                    .map(|(idx, _)| idx)
                    .unwrap();

                let mut component = eigenvectors.column(max_idx).mapv(|x| x.re);

                // Apply soft thresholding for sparsity
                for val in component.iter_mut() {
                    *val = if val.abs() > sparsity {
                        val.signum() * (val.abs() - sparsity)
                    } else {
                        0.0
                    };
                }

                // Normalize
                let norm = component.mapv(|x| x * x).sum().sqrt();
                if norm > 1e-10 {
                    component /= norm;
                }

                components.row_mut(k).assign(&component);
            }

            // Check convergence
            let change = (&components - &prev_components)
                .mapv(|x| x * x)
                .sum()
                .sqrt();
            if change < tolerance {
                break;
            }
        }

        // Compute explained variance for sparse components
        let mut explained_variance = Array1::zeros(n_components);
        for i in 0..n_components {
            let scores = x.dot(&components.row(i));
            explained_variance[i] = scores.var(0.0);
        }

        let singular_values =
            explained_variance.mapv(|x| x.sqrt() * (n_samples as f64 - 1.0).sqrt());

        Ok((
            components,
            explained_variance,
            singular_values,
            None,
            None,
            None,
        ))
    }

    /// Probabilistic PCA using EM algorithm
    fn probabilistic_pca(
        &self,
        x: &Array2<f64>,
        n_components: usize,
    ) -> SklResult<(
        Array2<f64>,
        Array1<f64>,
        Array1<f64>,
        Option<Array2<f64>>,
        Option<f64>,
        Option<Array2<f64>>,
    )> {
        let max_iter = 100;
        let tolerance = self.tol;
        let (n_samples, n_features) = x.dim();

        // Initialize parameters
        let mut rng = scirs2_core::random::thread_rng();
        let mut w =
            Array2::from_shape_fn((n_features, n_components), |_| rng.sample(StandardNormal));
        let mut sigma2 = 1.0;

        for _iter in 0..max_iter {
            let prev_w = w.clone();
            let prev_sigma2 = sigma2;

            // E-step: compute posterior covariance and mean
            let m = w.t().dot(&w) + Array2::<f64>::eye(n_components) * sigma2;
            let m_inv = m
                .inv()
                .map_err(|_| SklearsError::NumericalError("Matrix inversion failed".to_string()))?;

            // M-step: update parameters
            let mut sum_ezz = Array2::zeros((n_components, n_components));
            let mut sum_xz = Array2::zeros((n_features, n_components));

            for i in 0..n_samples {
                let xi = x.row(i);
                let ez = m_inv.dot(&w.t()).dot(&xi); // E[z|x]
                let ez_col = ez.clone().insert_axis(Axis(1));
                let ez_row = ez.clone().insert_axis(Axis(0));
                let ezz = &m_inv + ez_col.dot(&ez_row); // E[zz^T|x]

                sum_ezz += &ezz;
                let xi_col = xi.insert_axis(Axis(1));
                let ez_row_2 = ez.insert_axis(Axis(0));
                sum_xz += &xi_col.dot(&ez_row_2);
            }

            // Update W
            w = sum_xz.dot(&sum_ezz.inv().map_err(|_| {
                SklearsError::NumericalError("Matrix inversion failed".to_string())
            })?);

            // Update sigma^2
            let mut sum_residual = 0.0;
            for i in 0..n_samples {
                let xi = x.row(i);
                let ez = m_inv.dot(&w.t()).dot(&xi);
                let residual = &xi - &w.dot(&ez);
                sum_residual += residual.mapv(|x| x * x).sum();
            }
            sigma2 = sum_residual / (n_samples * n_features) as f64;

            // Check convergence
            let w_change = (&w - &prev_w).mapv(|x| x * x).sum().sqrt();
            let sigma_change = (sigma2 - prev_sigma2).abs();

            if w_change < tolerance && sigma_change < tolerance {
                break;
            }
        }

        // Convert to PCA format
        let components = w.t().to_owned();
        let eigenvalues = components.mapv(|x| x * x).sum_axis(Axis(1));
        let explained_variance = eigenvalues.clone();
        let singular_values = eigenvalues.mapv(|x| x.sqrt());

        Ok((
            components,
            explained_variance,
            singular_values,
            None,
            Some(sigma2),
            None,
        ))
    }

    /// Compute kernel value between two samples
    fn compute_kernel(
        &self,
        x1: &ArrayView1<f64>,
        x2: &ArrayView1<f64>,
        kernel_func: &KernelFunction,
    ) -> f64 {
        match kernel_func {
            KernelFunction::Linear => x1.dot(x2),
            KernelFunction::Polynomial { degree, coef } => (x1.dot(x2) + coef).powi(*degree as i32),
            KernelFunction::RBF { gamma } => {
                let diff = x1 - x2;
                let sq_dist = diff.mapv(|x| x * x).sum();
                (-gamma * sq_dist).exp()
            }
            KernelFunction::Sigmoid { gamma, coef } => (gamma * x1.dot(x2) + coef).tanh(),
        }
    }
}

impl Transform<ArrayView2<'_, f64>, Array2<f64>> for PCACovariance<PCACovarianceTrained> {
    fn transform(&self, x: &ArrayView2<'_, f64>) -> SklResult<Array2<f64>> {
        let mut x_transformed = x.to_owned();

        // Apply centering
        if self.center {
            for mut row in x_transformed.axis_iter_mut(Axis(0)) {
                row -= &self.state.mean;
            }
        }

        // Apply scaling
        if let Some(ref scale_factors) = self.state.scale_factors {
            for mut row in x_transformed.axis_iter_mut(Axis(0)) {
                for (i, val) in row.iter_mut().enumerate() {
                    if scale_factors[i] > 1e-10 {
                        *val /= scale_factors[i];
                    }
                }
            }
        }

        // Project onto principal components
        Ok(x_transformed.dot(&self.state.components.t()))
    }
}

impl PCACovariance<PCACovarianceTrained> {
    /// Get the estimated covariance matrix
    pub fn get_covariance(&self) -> &Array2<f64> {
        &self.state.covariance
    }

    /// Get the precision matrix
    pub fn get_precision(&self) -> Option<&Array2<f64>> {
        self.state.precision.as_ref()
    }

    /// Get the principal components
    pub fn get_components(&self) -> &Array2<f64> {
        &self.state.components
    }

    /// Get the explained variance
    pub fn get_explained_variance(&self) -> &Array1<f64> {
        &self.state.explained_variance
    }

    /// Get the explained variance ratio
    pub fn get_explained_variance_ratio(&self) -> &Array1<f64> {
        &self.state.explained_variance_ratio
    }

    /// Get the singular values
    pub fn get_singular_values(&self) -> &Array1<f64> {
        &self.state.singular_values
    }

    /// Get the mean of training data
    pub fn get_mean(&self) -> &Array1<f64> {
        &self.state.mean
    }

    /// Get the number of components
    pub fn get_n_components(&self) -> usize {
        self.state.n_components
    }

    /// Get the method used
    pub fn get_method(&self) -> &PCAMethod {
        &self.state.method
    }

    /// Get cumulative explained variance ratio
    pub fn get_cumulative_variance_ratio(&self) -> &Array1<f64> {
        &self.state.cumulative_variance_ratio
    }

    /// Get noise variance (for probabilistic PCA)
    pub fn get_noise_variance(&self) -> Option<f64> {
        self.state.noise_variance
    }

    /// Compute the number of components needed to explain a certain variance ratio
    pub fn components_for_variance_ratio(&self, target_ratio: f64) -> usize {
        self.state
            .cumulative_variance_ratio
            .iter()
            .position(|&ratio| ratio >= target_ratio)
            .map(|pos| pos + 1)
            .unwrap_or(self.state.n_components)
    }

    /// Inverse transform from PCA space back to original space
    pub fn inverse_transform(&self, x_pca: &ArrayView2<f64>) -> SklResult<Array2<f64>> {
        // Project back from PCA space
        let mut x_reconstructed = x_pca.dot(&self.state.components);

        // Reverse scaling
        if let Some(ref scale_factors) = self.state.scale_factors {
            for mut row in x_reconstructed.axis_iter_mut(Axis(0)) {
                for (i, val) in row.iter_mut().enumerate() {
                    *val *= scale_factors[i];
                }
            }
        }

        // Reverse centering
        if self.center {
            for mut row in x_reconstructed.axis_iter_mut(Axis(0)) {
                row += &self.state.mean;
            }
        }

        Ok(x_reconstructed)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_standard_pca_basic() {
        let x = array![
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0]
        ];

        let estimator = PCACovariance::new()
            .n_components(2)
            .method(PCAMethod::Standard);

        let fitted = estimator.fit(&x.view(), &()).unwrap();

        assert_eq!(fitted.get_covariance().dim(), (3, 3));
        assert_eq!(fitted.get_components().dim(), (2, 3));
        assert_eq!(fitted.get_explained_variance().len(), 2);
        assert_eq!(fitted.get_n_components(), 2);

        // Test transform
        let transformed = fitted.transform(&x.view()).unwrap();
        assert_eq!(transformed.dim(), (5, 2));

        // Test inverse transform
        let reconstructed = fitted.inverse_transform(&transformed.view()).unwrap();
        assert_eq!(reconstructed.dim(), (5, 3));
    }

    #[test]
    fn test_kernel_pca_rbf() {
        let x = array![[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]];

        let estimator = PCACovariance::new()
            .n_components(2)
            .method(PCAMethod::Kernel)
            .kernel(KernelFunction::RBF { gamma: 1.0 });

        let fitted = estimator.fit(&x.view(), &()).unwrap();

        assert_eq!(fitted.get_components().dim(), (2, 2));
        assert_eq!(fitted.get_explained_variance().len(), 2);

        let transformed = fitted.transform(&x.view()).unwrap();
        assert_eq!(transformed.dim(), (4, 2));
    }

    #[test]
    fn test_probabilistic_pca() {
        let x = array![
            [1.0, 2.0, 3.0],
            [2.0, 4.0, 6.0],
            [3.0, 6.0, 9.0],
            [1.5, 3.0, 4.5],
            [2.5, 5.0, 7.5]
        ];

        let estimator = PCACovariance::new()
            .n_components(2)
            .method(PCAMethod::Probabilistic);

        let fitted = estimator.fit(&x.view(), &()).unwrap();

        assert_eq!(fitted.get_components().dim(), (2, 3));
        assert!(fitted.get_noise_variance().is_some());
        assert!(fitted.get_noise_variance().unwrap() > 0.0);
    }

    #[test]
    fn test_components_for_variance_ratio() {
        let x = array![
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0]
        ];

        let estimator = PCACovariance::new()
            .n_components(3)
            .method(PCAMethod::Standard);

        let fitted = estimator.fit(&x.view(), &()).unwrap();

        let n_comp_90 = fitted.components_for_variance_ratio(0.9);
        let n_comp_95 = fitted.components_for_variance_ratio(0.95);

        assert!(n_comp_90 <= n_comp_95);
        assert!(n_comp_95 <= 3);
    }

    #[test]
    fn test_sparse_pca_sparsity() {
        let x = array![
            [1.0, 0.1, 0.01],
            [2.0, 0.2, 0.02],
            [3.0, 0.3, 0.03],
            [1.5, 0.15, 0.015],
            [2.5, 0.25, 0.025]
        ];

        let estimator = PCACovariance::new()
            .n_components(2)
            .method(PCAMethod::Sparse)
            .sparsity(0.1);

        let fitted = estimator.fit(&x.view(), &()).unwrap();

        // Check that some components have been sparsified (set to zero)
        let components = fitted.get_components();
        let zero_count = components.iter().filter(|&&x| x.abs() < 1e-10).count();
        assert!(zero_count > 0);
    }
}
