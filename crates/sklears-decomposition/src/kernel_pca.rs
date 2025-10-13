//! Kernel Principal Component Analysis (Kernel PCA) implementation
//!
//! Kernel PCA is a non-linear dimensionality reduction technique that uses
//! kernel methods to perform PCA in a higher-dimensional feature space.

// TODO: Replace with scirs2-linalg
// use nalgebra::{DMatrix, DVector};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, Axis};
use scirs2_core::rand_prelude::SliceRandom;
use scirs2_core::random::{thread_rng, Random, Rng};
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Fit, Trained, Transform, Untrained},
    types::Float,
};
use std::marker::PhantomData;

/// Kernel functions for Kernel PCA
#[derive(Debug, Clone, Copy)]
pub enum KernelFunction {
    /// Linear kernel: K(x, y) = <x, y>
    Linear,
    /// RBF (Gaussian) kernel: K(x, y) = exp(-gamma * ||x - y||^2)
    Rbf { gamma: Float },
    /// Polynomial kernel: K(x, y) = (gamma * <x, y> + coef0)^degree
    Polynomial {
        degree: i32,
        gamma: Float,
        coef0: Float,
    },
    /// Sigmoid kernel: K(x, y) = tanh(gamma * <x, y> + coef0)
    Sigmoid { gamma: Float, coef0: Float },
    /// Laplacian kernel: K(x, y) = exp(-gamma * ||x - y||_1)
    Laplacian { gamma: Float },
    /// Chi-squared kernel: K(x, y) = exp(-gamma * sum((x_i - y_i)^2 / (x_i + y_i)))
    ChiSquared { gamma: Float },
}

impl Default for KernelFunction {
    fn default() -> Self {
        KernelFunction::Rbf { gamma: 1.0 }
    }
}

impl KernelFunction {
    /// Compute kernel value between two vectors
    pub fn compute(&self, x: &Array1<Float>, y: &Array1<Float>) -> Float {
        match self {
            KernelFunction::Linear => x.dot(y),
            KernelFunction::Rbf { gamma } => {
                let diff = x - y;
                let dist_sq = diff.dot(&diff);
                (-gamma * dist_sq).exp()
            }
            KernelFunction::Polynomial {
                degree,
                gamma,
                coef0,
            } => {
                let dot_product = x.dot(y);
                (gamma * dot_product + coef0).powi(*degree)
            }
            KernelFunction::Sigmoid { gamma, coef0 } => {
                let dot_product = x.dot(y);
                (gamma * dot_product + coef0).tanh()
            }
            KernelFunction::Laplacian { gamma } => {
                let l1_dist = (x - y).mapv(|x| x.abs()).sum();
                (-gamma * l1_dist).exp()
            }
            KernelFunction::ChiSquared { gamma } => {
                let mut chi_sq_dist = 0.0;
                for i in 0..x.len() {
                    let sum = x[i] + y[i];
                    if sum > 1e-12 {
                        let diff = x[i] - y[i];
                        chi_sq_dist += diff * diff / sum;
                    }
                }
                (-gamma * chi_sq_dist).exp()
            }
        }
    }

    /// Compute kernel matrix between two sets of vectors
    pub fn compute_matrix(&self, x: &Array2<Float>, y: &Array2<Float>) -> Array2<Float> {
        let n_x = x.nrows();
        let n_y = y.nrows();
        let mut kernel_matrix = Array2::zeros((n_x, n_y));

        for i in 0..n_x {
            for j in 0..n_y {
                let x_i = x.row(i).to_owned();
                let y_j = y.row(j).to_owned();
                kernel_matrix[[i, j]] = self.compute(&x_i, &y_j);
            }
        }

        kernel_matrix
    }
}

/// Kernel matrix approximation methods
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum KernelApproximation {
    /// Full kernel matrix computation (no approximation)
    #[default]
    Full,
    /// Nyström method approximation
    Nystrom { n_components: usize },
    /// Random sampling approximation
    RandomSampling { n_samples: usize },
}

/// Configuration for Kernel PCA
#[derive(Debug, Clone)]
pub struct KernelPcaConfig {
    /// Number of components to keep
    pub n_components: Option<usize>,
    /// Kernel function to use
    pub kernel: KernelFunction,
    /// Tolerance for eigenvalue computation
    pub tol: Float,
    /// Maximum number of iterations for eigenvalue computation
    pub max_iter: usize,
    /// Whether to center the kernel matrix
    pub center: bool,
    /// Whether to copy the input data
    pub copy: bool,
    /// Kernel matrix approximation method
    pub approximation: KernelApproximation,
    /// Random state for reproducible approximations
    pub random_state: Option<u64>,
}

impl Default for KernelPcaConfig {
    fn default() -> Self {
        Self {
            n_components: None,
            kernel: KernelFunction::default(),
            tol: 1e-8,
            max_iter: 300,
            center: true,
            copy: true,
            approximation: KernelApproximation::default(),
            random_state: None,
        }
    }
}

/// Kernel Principal Component Analysis (Kernel PCA)
///
/// Non-linear dimensionality reduction through the use of kernels.
/// It uses the kernel trick to perform PCA in a potentially infinite-dimensional
/// feature space without explicitly computing the features.
///
/// # Examples
///
/// ```rust,ignore
/// use sklears_decomposition::{KernelPCA, KernelFunction};
/// use sklears_core::traits::{Transform, Fit};
/// use scirs2_core::ndarray::array;
///
/// let x = array![
///     [1.0, 2.0],
///     [3.0, 4.0],
///     [5.0, 6.0],
///     [7.0, 8.0],
/// ];
///
/// let kpca = KernelPCA::new()
///     .n_components(2)
///     .kernel(KernelFunction::Rbf { gamma: 0.1 })
///     .fit(&x, &())?;
///
/// let x_transformed = kpca.transform(&x)?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone)]
pub struct KernelPCA<State = Untrained> {
    config: KernelPcaConfig,
    state: PhantomData<State>,
    // Fitted parameters
    x_fit_: Option<Array2<Float>>,
    lambdas_: Option<Array1<Float>>,
    alphas_: Option<Array2<Float>>,
    n_components_: Option<usize>,
    n_features_in_: Option<usize>,
    n_samples_: Option<usize>,
}

impl KernelPCA<Untrained> {
    /// Create a new Kernel PCA
    pub fn new() -> Self {
        Self {
            config: KernelPcaConfig::default(),
            state: PhantomData,
            x_fit_: None,
            lambdas_: None,
            alphas_: None,
            n_components_: None,
            n_features_in_: None,
            n_samples_: None,
        }
    }

    /// Set the number of components to keep
    pub fn n_components(mut self, n_components: usize) -> Self {
        self.config.n_components = Some(n_components);
        self
    }

    /// Set the kernel function
    pub fn kernel(mut self, kernel: KernelFunction) -> Self {
        self.config.kernel = kernel;
        self
    }

    /// Set the tolerance for eigenvalue computation
    pub fn tol(mut self, tol: Float) -> Self {
        self.config.tol = tol;
        self
    }

    /// Set the maximum number of iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.config.max_iter = max_iter;
        self
    }

    /// Set whether to center the kernel matrix
    pub fn center(mut self, center: bool) -> Self {
        self.config.center = center;
        self
    }

    /// Set whether to copy the input data
    pub fn copy(mut self, copy: bool) -> Self {
        self.config.copy = copy;
        self
    }

    /// Set the kernel matrix approximation method
    pub fn approximation(mut self, approximation: KernelApproximation) -> Self {
        self.config.approximation = approximation;
        self
    }

    /// Set the random state for reproducible approximations
    pub fn random_state(mut self, random_state: u64) -> Self {
        self.config.random_state = Some(random_state);
        self
    }
}

impl Default for KernelPCA<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Fit<Array2<Float>, ()> for KernelPCA<Untrained> {
    type Fitted = KernelPCA<Trained>;

    fn fit(self, x: &Array2<Float>, _y: &()) -> Result<Self::Fitted> {
        let (n_samples, n_features) = x.dim();

        if n_samples == 0 {
            return Err(SklearsError::InvalidInput("Empty dataset".to_string()));
        }

        if n_features == 0 {
            return Err(SklearsError::InvalidInput(
                "Dataset has no features".to_string(),
            ));
        }

        let n_components = self
            .config
            .n_components
            .unwrap_or(n_samples.min(n_features))
            .min(n_samples);

        if n_components == 0 {
            return Err(SklearsError::InvalidInput(
                "Number of components must be positive".to_string(),
            ));
        }

        // Store the training data
        let x_fit = if self.config.copy {
            x.clone()
        } else {
            x.to_owned()
        };

        // Compute kernel matrix or approximation based on configuration
        let (lambdas, alphas) = match self.config.approximation {
            KernelApproximation::Full => {
                // Compute full kernel matrix
                let mut k = self.config.kernel.compute_matrix(&x_fit, &x_fit);

                // Center the kernel matrix if requested
                if self.config.center {
                    self.center_kernel_matrix(&mut k)?;
                }

                // Solve eigenvalue problem: K * alpha = lambda * alpha
                self.solve_eigenvalue_problem(&k, n_components)?
            }
            KernelApproximation::Nystrom {
                n_components: nystrom_components,
            } => {
                // Use Nyström method for large-scale approximation
                self.nystrom_approximation(&x_fit, n_components, nystrom_components)?
            }
            KernelApproximation::RandomSampling { n_samples } => {
                // Use random sampling approximation
                self.random_sampling_approximation(&x_fit, n_components, n_samples)?
            }
        };

        Ok(KernelPCA {
            config: self.config,
            state: PhantomData,
            x_fit_: Some(x_fit),
            lambdas_: Some(lambdas),
            alphas_: Some(alphas),
            n_components_: Some(n_components),
            n_features_in_: Some(n_features),
            n_samples_: Some(n_samples),
        })
    }
}

impl KernelPCA<Untrained> {
    /// Center the kernel matrix
    fn center_kernel_matrix(&self, k: &mut Array2<Float>) -> Result<()> {
        let n = k.nrows();
        if n != k.ncols() {
            return Err(SklearsError::InvalidInput(
                "Kernel matrix must be square".to_string(),
            ));
        }

        // Compute row means
        let row_means = k.mean_axis(Axis(1)).unwrap();
        // Compute overall mean
        let overall_mean = k.mean().unwrap();

        // Center the kernel matrix: K_centered = K - K_row_means - K_col_means + K_overall_mean
        for i in 0..n {
            for j in 0..n {
                k[[i, j]] = k[[i, j]] - row_means[i] - row_means[j] + overall_mean;
            }
        }

        Ok(())
    }

    /// Solve eigenvalue problem using proper eigendecomposition
    ///
    /// This performs symmetric eigendecomposition of the centered kernel matrix
    /// to find the principal components in the feature space.
    fn solve_eigenvalue_problem(
        &self,
        k: &Array2<Float>,
        n_components: usize,
    ) -> Result<(Array1<Float>, Array2<Float>)> {
        let n = k.nrows();
        let n_comp = n_components.min(n);

        // Convert ndarray to nalgebra matrix for eigendecomposition
        let mut k_nalgebra = nalgebra::DMatrix::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                k_nalgebra[(i, j)] = k[[i, j]];
            }
        }

        // Perform symmetric eigendecomposition
        // Since kernel matrices should be positive semi-definite, we use symmetric solver
        let eigendecomp = k_nalgebra.symmetric_eigen();

        let eigenvalues = eigendecomp.eigenvalues;
        let eigenvectors = eigendecomp.eigenvectors;

        // Collect eigenvalue-eigenvector pairs
        let mut eigen_pairs: Vec<(Float, Vec<Float>)> = Vec::new();
        for i in 0..n {
            let eigenval = eigenvalues[i];
            let eigenvec: Vec<Float> = (0..n).map(|j| eigenvectors[(j, i)]).collect();
            eigen_pairs.push((eigenval, eigenvec));
        }

        // Sort eigenvalue-eigenvector pairs by eigenvalue in descending order
        eigen_pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        // Extract the top n_comp eigenvalues and eigenvectors
        let mut lambdas = Array1::zeros(n_comp);
        let mut alphas = Array2::zeros((n, n_comp));

        for comp in 0..n_comp {
            if comp < eigen_pairs.len() {
                // Extract eigenvalue (ensure non-negative for numerical stability)
                lambdas[comp] = eigen_pairs[comp].0.max(0.0);

                // Extract eigenvector
                let eigenvec = &eigen_pairs[comp].1;
                for i in 0..n {
                    alphas[[i, comp]] = eigenvec[i];
                }

                // Normalize the eigenvector (should already be normalized, but ensure it)
                let mut norm = 0.0;
                for i in 0..n {
                    norm += alphas[[i, comp]] * alphas[[i, comp]];
                }
                norm = norm.sqrt();

                if norm > 1e-12 {
                    for i in 0..n {
                        alphas[[i, comp]] /= norm;
                    }
                }
            }
        }

        Ok((lambdas, alphas))
    }

    /// Nyström method for efficient kernel matrix approximation
    ///
    /// The Nyström method approximates a large kernel matrix using a subset of columns/rows.
    /// This allows efficient computation of eigendecomposition for large datasets.
    ///
    /// # Arguments
    /// * `x` - Training data
    /// * `n_components` - Number of components to extract
    /// * `nystrom_components` - Number of landmark points for Nyström approximation
    fn nystrom_approximation(
        &self,
        x: &Array2<Float>,
        n_components: usize,
        nystrom_components: usize,
    ) -> Result<(Array1<Float>, Array2<Float>)> {
        let (n_samples, _) = x.dim();
        let m = nystrom_components.min(n_samples);

        // Initialize random number generator
        // TODO: Support seeding for reproducibility
        let mut rng = thread_rng();

        // Randomly sample m landmark points
        let mut landmark_indices: Vec<usize> = (0..n_samples).collect();
        landmark_indices.shuffle(&mut rng);
        landmark_indices.truncate(m);

        // Extract landmark points
        let mut landmarks = Array2::zeros((m, x.ncols()));
        for (i, &idx) in landmark_indices.iter().enumerate() {
            landmarks.row_mut(i).assign(&x.row(idx));
        }

        // Compute kernel submatrices
        // W: kernel matrix between landmarks (m x m)
        let mut w = self.config.kernel.compute_matrix(&landmarks, &landmarks);

        // C: kernel matrix between all points and landmarks (n x m)
        let c = self.config.kernel.compute_matrix(x, &landmarks);

        // Center the matrices if requested
        if self.config.center {
            self.center_kernel_matrix(&mut w)?;
            // Note: C centering is more complex and approximated here
        }

        // Eigendecomposition of W
        let (w_eigenvals, w_eigenvecs) = self.solve_eigenvalue_problem(&w, m)?;

        // Filter out near-zero eigenvalues for numerical stability
        let mut valid_components = Vec::new();
        for i in 0..m {
            if w_eigenvals[i] > 1e-10 {
                valid_components.push(i);
            }
        }
        let k = valid_components.len().min(n_components);

        // Compute Nyström approximation eigenvalues and eigenvectors
        let mut nystrom_eigenvals = Array1::zeros(k);
        let mut nystrom_eigenvecs = Array2::zeros((n_samples, k));

        for (comp_idx, &w_idx) in valid_components.iter().take(k).enumerate() {
            // Eigenvalue: scaled by n_samples/m
            nystrom_eigenvals[comp_idx] = w_eigenvals[w_idx] * (n_samples as Float) / (m as Float);

            // Eigenvector: C * w_eigenvec / sqrt(m * eigenval)
            let w_eigenvec = w_eigenvecs.column(w_idx);
            let scale = 1.0 / (m as Float * w_eigenvals[w_idx]).sqrt();

            for i in 0..n_samples {
                let mut eigenvec_val = 0.0;
                for j in 0..m {
                    eigenvec_val += c[[i, j]] * w_eigenvec[j];
                }
                nystrom_eigenvecs[[i, comp_idx]] = eigenvec_val * scale;
            }
        }

        Ok((nystrom_eigenvals, nystrom_eigenvecs))
    }

    /// Random sampling approximation for kernel matrix
    ///
    /// This method uses random sampling to create a smaller representative dataset
    /// and performs full kernel PCA on this subset.
    ///
    /// # Arguments
    /// * `x` - Training data
    /// * `n_components` - Number of components to extract
    /// * `n_samples_approx` - Number of samples to use for approximation
    fn random_sampling_approximation(
        &self,
        x: &Array2<Float>,
        n_components: usize,
        n_samples_approx: usize,
    ) -> Result<(Array1<Float>, Array2<Float>)> {
        let (n_samples, n_features) = x.dim();
        let m = n_samples_approx.min(n_samples);

        // Initialize random number generator
        // TODO: Support seeding for reproducibility
        let mut rng = thread_rng();

        // Randomly sample points
        let mut sample_indices: Vec<usize> = (0..n_samples).collect();
        sample_indices.shuffle(&mut rng);
        sample_indices.truncate(m);

        // Extract sampled data
        let mut x_sampled = Array2::zeros((m, n_features));
        for (i, &idx) in sample_indices.iter().enumerate() {
            x_sampled.row_mut(i).assign(&x.row(idx));
        }

        // Compute kernel matrix on sampled data
        let mut k_sampled = self.config.kernel.compute_matrix(&x_sampled, &x_sampled);

        // Center the kernel matrix if requested
        if self.config.center {
            self.center_kernel_matrix(&mut k_sampled)?;
        }

        // Solve eigenvalue problem on sampled data
        let (eigenvals_sampled, eigenvecs_sampled) =
            self.solve_eigenvalue_problem(&k_sampled, n_components.min(m))?;

        // Project eigenvectors back to full space
        // This is an approximation - in practice, we'd need to store the sampled data
        // and use it during transform phase
        let mut full_eigenvecs = Array2::zeros((n_samples, eigenvals_sampled.len()));

        // For each eigenvector, interpolate to full space using kernel evaluations
        for comp in 0..eigenvals_sampled.len() {
            for i in 0..n_samples {
                let mut projection = 0.0;
                for (j, &sample_idx) in sample_indices.iter().enumerate() {
                    let x_i = x.row(i).to_owned();
                    let x_sample_j = x.row(sample_idx).to_owned();
                    let k_val = self.config.kernel.compute(&x_i, &x_sample_j);
                    projection += k_val * eigenvecs_sampled[[j, comp]];
                }

                // Normalize by eigenvalue
                if eigenvals_sampled[comp] > 1e-10 {
                    projection /= eigenvals_sampled[comp].sqrt();
                }

                full_eigenvecs[[i, comp]] = projection;
            }
        }

        Ok((eigenvals_sampled, full_eigenvecs))
    }
}

impl Transform<Array2<Float>, Array2<Float>> for KernelPCA<Trained> {
    fn transform(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let (n_samples, n_features) = x.dim();

        if n_features != self.n_features_in() {
            return Err(SklearsError::FeatureMismatch {
                expected: self.n_features_in(),
                actual: n_features,
            });
        }

        let x_fit = self.x_fit_.as_ref().unwrap();
        let alphas = self.alphas_.as_ref().unwrap();
        let lambdas = self.lambdas_.as_ref().unwrap();
        let n_components = self.n_components();

        // Compute kernel matrix between x and training data
        let mut k_test = self.config.kernel.compute_matrix(x, x_fit);

        // Center the kernel matrix if needed
        if self.config.center {
            // For centering test data, we need to apply the same transformation
            // as was applied to the training kernel matrix
            let n_train = x_fit.nrows();
            let row_means_train: Array1<Float> = Array1::zeros(n_train); // Simplified - should store from training
            let overall_mean_train: Float = 0.0; // Simplified - should store from training

            for i in 0..n_samples {
                for j in 0..n_train {
                    k_test[[i, j]] = k_test[[i, j]] - row_means_train[j] - overall_mean_train;
                }
            }
        }

        // Project onto the principal components
        let mut x_transformed = Array2::zeros((n_samples, n_components));

        for i in 0..n_samples {
            for comp in 0..n_components {
                let mut projection = 0.0;
                for j in 0..x_fit.nrows() {
                    projection += k_test[[i, j]] * alphas[[j, comp]];
                }

                // Scale by sqrt(eigenvalue)
                if lambdas[comp] > 1e-10 {
                    projection /= lambdas[comp].sqrt();
                }

                x_transformed[[i, comp]] = projection;
            }
        }

        Ok(x_transformed)
    }
}

impl KernelPCA<Trained> {
    /// Get the eigenvalues
    pub fn eigenvalues(&self) -> &Array1<Float> {
        self.lambdas_.as_ref().unwrap()
    }

    /// Get the eigenvectors (alphas)
    pub fn eigenvectors(&self) -> &Array2<Float> {
        self.alphas_.as_ref().unwrap()
    }

    /// Get the number of components
    pub fn n_components(&self) -> usize {
        self.n_components_.unwrap()
    }

    /// Get the number of features in the input
    pub fn n_features_in(&self) -> usize {
        self.n_features_in_.unwrap()
    }

    /// Get the number of samples in the training data
    pub fn n_samples(&self) -> usize {
        self.n_samples_.unwrap()
    }

    /// Get the training data
    pub fn x_fit(&self) -> &Array2<Float> {
        self.x_fit_.as_ref().unwrap()
    }

    /// Pre-image reconstruction using fixed-point iteration
    ///
    /// Reconstructs the original space representation from the transformed features.
    /// This is useful for understanding what the transformed features represent
    /// in the original input space.
    ///
    /// # Arguments
    /// * `x_transformed` - The transformed data to reconstruct (n_samples, n_components)
    /// * `max_iter` - Maximum number of iterations for fixed-point iteration
    /// * `tol` - Tolerance for convergence
    ///
    /// # Returns
    /// Reconstructed data in original space (n_samples, n_features)
    pub fn inverse_transform(
        &self,
        x_transformed: &Array2<Float>,
        max_iter: usize,
        tol: Float,
    ) -> Result<Array2<Float>> {
        let (_n_samples, n_components_in) = x_transformed.dim();
        let n_components = self.n_components();
        let _n_features = self.n_features_in();
        let _x_fit = self.x_fit();
        let _alphas = self.eigenvectors();
        let _lambdas = self.eigenvalues();

        if n_components_in != n_components {
            return Err(SklearsError::FeatureMismatch {
                expected: n_components,
                actual: n_components_in,
            });
        }

        // Handle different kernel types
        match &self.config.kernel {
            KernelFunction::Linear => self.linear_preimage(x_transformed),
            KernelFunction::Rbf { gamma: _ } => {
                self.nonlinear_preimage_fixed_point(x_transformed, max_iter, tol)
            }
            KernelFunction::Polynomial { .. } => {
                self.nonlinear_preimage_fixed_point(x_transformed, max_iter, tol)
            }
            KernelFunction::Sigmoid { .. } => {
                self.nonlinear_preimage_fixed_point(x_transformed, max_iter, tol)
            }
            KernelFunction::Laplacian { .. } => {
                self.nonlinear_preimage_fixed_point(x_transformed, max_iter, tol)
            }
            KernelFunction::ChiSquared { .. } => {
                self.nonlinear_preimage_fixed_point(x_transformed, max_iter, tol)
            }
        }
    }

    /// Linear pre-image reconstruction (exact for linear kernels)
    fn linear_preimage(&self, x_transformed: &Array2<Float>) -> Result<Array2<Float>> {
        let x_fit = self.x_fit();
        let alphas = self.eigenvectors();
        let lambdas = self.eigenvalues();
        let (n_samples, n_components) = x_transformed.dim();
        let n_features = self.n_features_in();

        // For linear kernels: x_reconstructed = sum_i (alpha_i * lambda_i * x_fit_i)
        let mut x_reconstructed = Array2::zeros((n_samples, n_features));

        for i in 0..n_samples {
            for k in 0..n_features {
                let mut value = 0.0;
                for comp in 0..n_components {
                    if lambdas[comp] > 1e-10 {
                        let component_contrib = x_transformed[[i, comp]] * lambdas[comp].sqrt();
                        for j in 0..x_fit.nrows() {
                            value += alphas[[j, comp]] * component_contrib * x_fit[[j, k]];
                        }
                    }
                }
                x_reconstructed[[i, k]] = value;
            }
        }

        Ok(x_reconstructed)
    }

    /// Non-linear pre-image reconstruction using fixed-point iteration
    fn nonlinear_preimage_fixed_point(
        &self,
        x_transformed: &Array2<Float>,
        max_iter: usize,
        tol: Float,
    ) -> Result<Array2<Float>> {
        let (n_samples, _n_components) = x_transformed.dim();
        let n_features = self.n_features_in();
        let x_fit = self.x_fit();
        let alphas = self.eigenvectors();
        let lambdas = self.eigenvalues();

        let mut x_reconstructed = Array2::zeros((n_samples, n_features));

        for i in 0..n_samples {
            // Initialize with mean of training data
            let mut x_current = x_fit.mean_axis(Axis(0)).unwrap();
            let target_transformed = x_transformed.slice(scirs2_core::ndarray::s![i, ..]);

            for _iter in 0..max_iter {
                let x_old = x_current.clone();

                // Fixed-point iteration step
                x_current =
                    self.fixed_point_step(&x_old, &target_transformed, x_fit, alphas, lambdas)?;

                // Check convergence
                let diff = (&x_current - &x_old)
                    .mapv(|x| x.abs())
                    .fold(0.0f64, |acc, &x| acc.max(x));
                if diff < tol {
                    break;
                }
            }

            // Store result
            for j in 0..n_features {
                x_reconstructed[[i, j]] = x_current[j];
            }
        }

        Ok(x_reconstructed)
    }

    /// Fixed-point iteration step for pre-image reconstruction
    fn fixed_point_step(
        &self,
        x_current: &Array1<Float>,
        target_transformed: &scirs2_core::ndarray::ArrayView1<Float>,
        x_fit: &Array2<Float>,
        alphas: &Array2<Float>,
        lambdas: &Array1<Float>,
    ) -> Result<Array1<Float>> {
        let n_features = x_current.len();
        let n_train = x_fit.nrows();
        let n_components = target_transformed.len();

        // Compute kernel derivatives and weights
        let mut numerator = Array1::<Float>::zeros(n_features);
        let mut denominator = Array1::<Float>::zeros(n_features);

        for j in 0..n_train {
            let x_train_j = x_fit.slice(scirs2_core::ndarray::s![j, ..]).to_owned();
            let _k_val = self.config.kernel.compute(x_current, &x_train_j);
            let k_deriv = self.kernel_derivative(x_current, &x_train_j);

            // Weight for this training point
            let mut weight = 0.0;
            for comp in 0..n_components {
                if lambdas[comp] > 1e-10 {
                    weight += target_transformed[comp] * alphas[[j, comp]] / lambdas[comp].sqrt();
                }
            }

            for k in 0..n_features {
                numerator[k] += weight * k_deriv[k] * x_train_j[k];
                denominator[k] += weight * k_deriv[k];
            }
        }

        // Update step
        let mut x_new = Array1::zeros(n_features);
        for k in 0..n_features {
            if denominator[k].abs() > 1e-12 {
                x_new[k] = numerator[k] / denominator[k];
            } else {
                x_new[k] = x_current[k]; // Keep current value if denominator is too small
            }
        }

        Ok(x_new)
    }

    /// Compute kernel derivative with respect to the first argument
    fn kernel_derivative(&self, x: &Array1<Float>, y: &Array1<Float>) -> Array1<Float> {
        match &self.config.kernel {
            KernelFunction::Linear => {
                // d/dx K(x,y) = y for linear kernel
                y.clone()
            }
            KernelFunction::Rbf { gamma } => {
                // d/dx K(x,y) = -2*gamma*(x-y)*K(x,y) for RBF kernel
                let k_val = self.config.kernel.compute(x, y);
                let diff = x - y;
                diff.mapv(|d| -2.0 * gamma * d * k_val)
            }
            KernelFunction::Polynomial {
                degree,
                gamma,
                coef0,
            } => {
                // d/dx K(x,y) = degree * gamma * y * (gamma*<x,y> + coef0)^(degree-1)
                let dot_product = x.dot(y);
                let base = gamma * dot_product + coef0;
                if *degree == 1 || base.abs() < 1e-12 {
                    y.mapv(|yi| *gamma * yi)
                } else {
                    let factor = (*degree as Float) * gamma * base.powf(*degree as Float - 1.0);
                    y.mapv(|yi| factor * yi)
                }
            }
            KernelFunction::Sigmoid { gamma, coef0 } => {
                // d/dx K(x,y) = gamma * y * (1 - tanh²(gamma*<x,y> + coef0))
                let dot_product = x.dot(y);
                let tanh_val = (gamma * dot_product + coef0).tanh();
                let factor = gamma * (1.0 - tanh_val * tanh_val);
                y.mapv(|yi| factor * yi)
            }
            KernelFunction::Laplacian { gamma } => {
                // d/dx K(x,y) = -gamma * sign(x-y) * K(x,y) for Laplacian kernel
                let k_val = self.config.kernel.compute(x, y);
                let diff = x - y;
                diff.mapv(|d| -gamma * d.signum() * k_val)
            }
            KernelFunction::ChiSquared { gamma } => {
                // Approximate derivative for Chi-squared kernel (complex exact form)
                let k_val = self.config.kernel.compute(x, y);
                let mut derivative = Array1::zeros(x.len());
                for i in 0..x.len() {
                    let sum = x[i] + y[i];
                    if sum > 1e-12 {
                        let diff = x[i] - y[i];
                        derivative[i] = -2.0 * gamma * k_val * diff / sum;
                    }
                }
                derivative
            }
        }
    }

    /// Multi-dimensional scaling (MDS) based pre-image approximation
    ///
    /// An alternative pre-image reconstruction method using MDS to preserve
    /// distances in the original space.
    pub fn mds_preimage(&self, x_transformed: &Array2<Float>) -> Result<Array2<Float>> {
        let (n_samples, _n_components) = x_transformed.dim();
        let n_features = self.n_features_in();
        let x_fit = self.x_fit();

        // Compute distance matrix in transformed space
        let mut dist_transformed = Array2::zeros((n_samples, n_samples));
        for i in 0..n_samples {
            for j in 0..n_samples {
                let diff = &x_transformed.slice(scirs2_core::ndarray::s![i, ..])
                    - &x_transformed.slice(scirs2_core::ndarray::s![j, ..]);
                dist_transformed[[i, j]] = diff.mapv(|x| x * x).sum().sqrt();
            }
        }

        // Find closest training samples for each transformed point
        let mut x_reconstructed = Array2::zeros((n_samples, n_features));

        for i in 0..n_samples {
            // Find k nearest neighbors in the training set (using feature space distance)
            let k = 5.min(x_fit.nrows()); // Use top 5 neighbors
            let mut distances = Vec::new();

            for j in 0..x_fit.nrows() {
                // Compute transformed distance to training sample j
                let x_train_j_transformed =
                    self.transform_single_sample(&x_fit.slice(scirs2_core::ndarray::s![j, ..]))?;
                let diff =
                    &x_transformed.slice(scirs2_core::ndarray::s![i, ..]) - &x_train_j_transformed;
                let dist = diff.mapv(|x| x * x).sum().sqrt();
                distances.push((dist, j));
            }

            // Sort by distance and take k nearest
            distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

            // Weighted average of k nearest neighbors
            let mut weight_sum = 0.0;
            for feat in 0..n_features {
                let mut weighted_value = 0.0;
                for &(dist, idx) in distances.iter().take(k) {
                    let weight = if dist > 1e-12 {
                        1.0 / (dist + 1e-6)
                    } else {
                        1e6
                    };
                    weighted_value += weight * x_fit[[idx, feat]];
                    weight_sum += weight;
                }
                x_reconstructed[[i, feat]] = weighted_value / weight_sum;
            }
        }

        Ok(x_reconstructed)
    }

    /// Transform a single sample (helper method)
    fn transform_single_sample(
        &self,
        x_sample: &scirs2_core::ndarray::ArrayView1<Float>,
    ) -> Result<Array1<Float>> {
        let x_fit = self.x_fit();
        let alphas = self.eigenvectors();
        let lambdas = self.eigenvalues();
        let n_components = self.n_components();

        // Compute kernel values with training data
        let mut k_test = Array1::zeros(x_fit.nrows());
        for j in 0..x_fit.nrows() {
            let x_train_j = x_fit.slice(scirs2_core::ndarray::s![j, ..]).to_owned();
            k_test[j] = self.config.kernel.compute(&x_sample.to_owned(), &x_train_j);
        }

        // Center the kernel values if the model was trained with centering
        if self.config.center {
            // This is approximate centering for a single sample
            let mean_k = k_test.mean().unwrap();
            for val in k_test.iter_mut() {
                *val -= mean_k;
            }
        }

        // Project onto the principal components
        let mut x_transformed = Array1::zeros(n_components);
        for comp in 0..n_components {
            let mut projection = 0.0;
            for j in 0..x_fit.nrows() {
                projection += k_test[j] * alphas[[j, comp]];
            }

            // Scale by sqrt(eigenvalue)
            if lambdas[comp] > 1e-10 {
                projection /= lambdas[comp].sqrt();
            }

            x_transformed[comp] = projection;
        }

        Ok(x_transformed)
    }
}

/// Kernel selection and validation utilities
impl KernelFunction {
    /// Validate kernel parameters for numerical stability
    pub fn validate(&self) -> Result<()> {
        match self {
            KernelFunction::Linear => Ok(()),
            KernelFunction::Rbf { gamma } => {
                if *gamma <= 0.0 {
                    return Err(SklearsError::InvalidParameter {
                        name: "gamma".to_string(),
                        reason: "must be positive for RBF kernel".to_string(),
                    });
                }
                if *gamma > 1e6 {
                    return Err(SklearsError::InvalidParameter {
                        name: "gamma".to_string(),
                        reason: "too large, may cause numerical instability".to_string(),
                    });
                }
                Ok(())
            }
            KernelFunction::Polynomial {
                degree,
                gamma,
                coef0: _,
            } => {
                if *degree <= 0 {
                    return Err(SklearsError::InvalidParameter {
                        name: "degree".to_string(),
                        reason: "must be positive for polynomial kernel".to_string(),
                    });
                }
                if *degree > 10 {
                    return Err(SklearsError::InvalidParameter {
                        name: "degree".to_string(),
                        reason: "too large, may cause numerical overflow".to_string(),
                    });
                }
                if *gamma <= 0.0 {
                    return Err(SklearsError::InvalidParameter {
                        name: "gamma".to_string(),
                        reason: "must be positive for polynomial kernel".to_string(),
                    });
                }
                Ok(())
            }
            KernelFunction::Sigmoid { gamma, coef0: _ } => {
                if *gamma <= 0.0 {
                    return Err(SklearsError::InvalidParameter {
                        name: "gamma".to_string(),
                        reason: "must be positive for sigmoid kernel".to_string(),
                    });
                }
                Ok(())
            }
            KernelFunction::Laplacian { gamma } => {
                if *gamma <= 0.0 {
                    return Err(SklearsError::InvalidParameter {
                        name: "gamma".to_string(),
                        reason: "must be positive for Laplacian kernel".to_string(),
                    });
                }
                if *gamma > 1e6 {
                    return Err(SklearsError::InvalidParameter {
                        name: "gamma".to_string(),
                        reason: "too large, may cause numerical instability".to_string(),
                    });
                }
                Ok(())
            }
            KernelFunction::ChiSquared { gamma } => {
                if *gamma <= 0.0 {
                    return Err(SklearsError::InvalidParameter {
                        name: "gamma".to_string(),
                        reason: "must be positive for Chi-squared kernel".to_string(),
                    });
                }
                Ok(())
            }
        }
    }

    /// Evaluate kernel performance on given data using cross-validation
    pub fn evaluate_performance(
        &self,
        x: &Array2<Float>,
        n_components: usize,
        cv_folds: usize,
    ) -> Result<Float> {
        let (n_samples, _) = x.dim();
        if n_samples < cv_folds {
            return Err(SklearsError::InvalidParameter {
                name: "cv_folds".to_string(),
                reason: "Number of samples must be >= cv_folds".to_string(),
            });
        }

        self.validate()?;

        let fold_size = n_samples / cv_folds;
        let mut reconstruction_errors = Vec::new();

        for fold in 0..cv_folds {
            let start_idx = fold * fold_size;
            let end_idx = if fold == cv_folds - 1 {
                n_samples
            } else {
                (fold + 1) * fold_size
            };

            // Create training and validation sets
            let mut train_indices = Vec::new();
            let mut val_indices = Vec::new();

            for i in 0..n_samples {
                if i >= start_idx && i < end_idx {
                    val_indices.push(i);
                } else {
                    train_indices.push(i);
                }
            }

            if train_indices.is_empty() || val_indices.is_empty() {
                continue;
            }

            // Extract training and validation data
            let x_train = x.select(scirs2_core::ndarray::Axis(0), &train_indices);
            let x_val = x.select(scirs2_core::ndarray::Axis(0), &val_indices);

            // Fit Kernel PCA on training data
            let kpca = KernelPCA::new()
                .n_components(n_components)
                .kernel(*self)
                .fit(&x_train, &())?;

            // Transform validation data
            let x_val_transformed = kpca.transform(&x_val)?;

            // Compute reconstruction error (simplified)
            let error = self.compute_reconstruction_error(&x_val, &x_val_transformed, &kpca)?;
            reconstruction_errors.push(error);
        }

        if reconstruction_errors.is_empty() {
            return Err(SklearsError::InvalidParameter {
                name: "cv_folds".to_string(),
                reason: "No valid cross-validation folds".to_string(),
            });
        }

        // Return mean reconstruction error
        let mean_error =
            reconstruction_errors.iter().sum::<Float>() / reconstruction_errors.len() as Float;
        Ok(mean_error)
    }

    /// Compute reconstruction error for validation
    fn compute_reconstruction_error(
        &self,
        x_original: &Array2<Float>,
        x_transformed: &Array2<Float>,
        _kpca: &KernelPCA<sklears_core::traits::Trained>,
    ) -> Result<Float> {
        // For now, we use a simplified metric based on variance preservation
        let original_var = self.compute_total_variance(x_original);
        let transformed_var = self.compute_total_variance(x_transformed);

        // Higher preserved variance means lower reconstruction error
        let error = (original_var - transformed_var).abs() / original_var.max(1e-10);
        Ok(error)
    }

    /// Compute total variance of data matrix
    fn compute_total_variance(&self, x: &Array2<Float>) -> Float {
        let (n_samples, n_features) = x.dim();
        if n_samples == 0 || n_features == 0 {
            return 0.0;
        }

        let mut total_var = 0.0;
        for j in 0..n_features {
            let col = x.column(j);
            let mean = col.mean().unwrap_or(0.0);
            let var = col.mapv(|x| (x - mean).powi(2)).mean().unwrap_or(0.0);
            total_var += var;
        }
        total_var
    }

    /// Select best kernel from a list of candidates using cross-validation
    pub fn select_best_kernel(
        kernels: &[KernelFunction],
        x: &Array2<Float>,
        n_components: usize,
        cv_folds: usize,
    ) -> Result<(KernelFunction, Float)> {
        if kernels.is_empty() {
            return Err(SklearsError::InvalidParameter {
                name: "kernels".to_string(),
                reason: "Must provide at least one kernel".to_string(),
            });
        }

        let mut best_kernel = kernels[0];
        let mut best_score = Float::INFINITY;

        for &kernel in kernels {
            match kernel.evaluate_performance(x, n_components, cv_folds) {
                Ok(score) => {
                    if score < best_score {
                        best_score = score;
                        best_kernel = kernel;
                    }
                }
                Err(_) => {
                    // Skip invalid kernels
                    continue;
                }
            }
        }

        if best_score == Float::INFINITY {
            return Err(SklearsError::InvalidParameter {
                name: "kernels".to_string(),
                reason: "All kernels failed validation".to_string(),
            });
        }

        Ok((best_kernel, best_score))
    }

    /// Generate a grid of kernel candidates for hyperparameter tuning
    pub fn generate_kernel_grid() -> Vec<KernelFunction> {
        let mut kernels = Vec::new();

        // Linear kernel
        kernels.push(KernelFunction::Linear);

        // RBF kernels with different gamma values
        for &gamma in &[0.001, 0.01, 0.1, 1.0, 10.0, 100.0] {
            kernels.push(KernelFunction::Rbf { gamma });
        }

        // Polynomial kernels
        for &degree in &[2, 3, 4] {
            for &gamma in &[0.1, 1.0] {
                for &coef0 in &[0.0, 1.0] {
                    kernels.push(KernelFunction::Polynomial {
                        degree,
                        gamma,
                        coef0,
                    });
                }
            }
        }

        // Sigmoid kernels
        for &gamma in &[0.001, 0.01, 0.1] {
            for &coef0 in &[0.0, 1.0] {
                kernels.push(KernelFunction::Sigmoid { gamma, coef0 });
            }
        }

        // Laplacian kernels
        for &gamma in &[0.01, 0.1, 1.0, 10.0] {
            kernels.push(KernelFunction::Laplacian { gamma });
        }

        // Chi-squared kernels
        for &gamma in &[0.1, 1.0, 10.0] {
            kernels.push(KernelFunction::ChiSquared { gamma });
        }

        kernels
    }
}

/// Kernel PCA with automatic kernel selection
impl KernelPCA<Untrained> {
    /// Fit Kernel PCA with automatic kernel selection
    pub fn fit_with_kernel_selection(
        mut self,
        x: &Array2<Float>,
        kernel_candidates: Option<&[KernelFunction]>,
        cv_folds: usize,
    ) -> Result<KernelPCA<sklears_core::traits::Trained>> {
        let kernels = kernel_candidates
            .map(|k| k.to_vec())
            .unwrap_or_else(KernelFunction::generate_kernel_grid);

        let n_components = self.config.n_components.unwrap_or(x.ncols().min(x.nrows()));

        let (best_kernel, _score) =
            KernelFunction::select_best_kernel(&kernels, x, n_components, cv_folds)?;

        self.config.kernel = best_kernel;
        self.fit(x, &())
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_kernel_functions() {
        let x = array![1.0, 2.0];
        let y = array![3.0, 4.0];

        // Linear kernel
        let linear = KernelFunction::Linear;
        let linear_result = linear.compute(&x, &y);
        assert_abs_diff_eq!(linear_result, 11.0, epsilon = 1e-10); // 1*3 + 2*4 = 11

        // RBF kernel
        let rbf = KernelFunction::Rbf { gamma: 1.0 };
        let rbf_result = rbf.compute(&x, &y);
        // exp(-1.0 * ((1-3)^2 + (2-4)^2)) = exp(-8) ≈ 0.000335
        assert!(rbf_result > 0.0 && rbf_result < 1.0);

        // Polynomial kernel
        let poly = KernelFunction::Polynomial {
            degree: 2,
            gamma: 1.0,
            coef0: 1.0,
        };
        let poly_result = poly.compute(&x, &y);
        // (1.0 * 11 + 1.0)^2 = 12^2 = 144
        assert_abs_diff_eq!(poly_result, 144.0, epsilon = 1e-10);

        // Sigmoid kernel
        let sigmoid = KernelFunction::Sigmoid {
            gamma: 1.0,
            coef0: 0.0,
        };
        let sigmoid_result = sigmoid.compute(&x, &y);
        // tanh(1.0 * 11 + 0.0) = tanh(11) ≈ 1.0
        assert!(sigmoid_result > 0.9 && sigmoid_result <= 1.0);
    }

    #[test]
    fn test_kernel_matrix() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![[5.0, 6.0], [7.0, 8.0]];

        let kernel = KernelFunction::Linear;
        let k = kernel.compute_matrix(&x, &y);

        assert_eq!(k.dim(), (2, 2));
        // K[0,0] = [1,2] · [5,6] = 1*5 + 2*6 = 17
        assert_abs_diff_eq!(k[[0, 0]], 17.0, epsilon = 1e-10);
        // K[0,1] = [1,2] · [7,8] = 1*7 + 2*8 = 23
        assert_abs_diff_eq!(k[[0, 1]], 23.0, epsilon = 1e-10);
        // K[1,0] = [3,4] · [5,6] = 3*5 + 4*6 = 39
        assert_abs_diff_eq!(k[[1, 0]], 39.0, epsilon = 1e-10);
        // K[1,1] = [3,4] · [7,8] = 3*7 + 4*8 = 53
        assert_abs_diff_eq!(k[[1, 1]], 53.0, epsilon = 1e-10);
    }

    #[test]
    fn test_kernel_pca_creation() {
        let kpca = KernelPCA::new()
            .n_components(2)
            .kernel(KernelFunction::Rbf { gamma: 0.1 })
            .tol(1e-6)
            .center(false);

        assert_eq!(kpca.config.n_components, Some(2));
        assert_eq!(kpca.config.tol, 1e-6);
        assert_eq!(kpca.config.center, false);
    }

    #[test]
    fn test_kernel_pca_fit_transform() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0],];

        let kpca = KernelPCA::new()
            .n_components(2)
            .kernel(KernelFunction::Linear)
            .fit(&x, &())
            .unwrap();

        assert_eq!(kpca.n_components(), 2);
        assert_eq!(kpca.n_features_in(), 2);
        assert_eq!(kpca.n_samples(), 4);

        let x_transformed = kpca.transform(&x).unwrap();
        assert_eq!(x_transformed.dim(), (4, 2));

        // Transformed data should be finite
        for &val in x_transformed.iter() {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_kernel_pca_rbf() {
        let x = array![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0],];

        let kpca = KernelPCA::new()
            .n_components(2)
            .kernel(KernelFunction::Rbf { gamma: 0.5 })
            .fit(&x, &())
            .unwrap();

        let x_transformed = kpca.transform(&x).unwrap();
        assert_eq!(x_transformed.dim(), (4, 2));

        // Check that eigenvalues are non-negative
        let eigenvalues = kpca.eigenvalues();
        for &val in eigenvalues.iter() {
            assert!(val >= 0.0, "Eigenvalue should be non-negative: {}", val);
        }
    }

    #[test]
    fn test_kernel_pca_polynomial() {
        let x = array![[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0],];

        let kpca = KernelPCA::new()
            .n_components(3)
            .kernel(KernelFunction::Polynomial {
                degree: 2,
                gamma: 1.0,
                coef0: 1.0,
            })
            .fit(&x, &())
            .unwrap();

        let x_transformed = kpca.transform(&x).unwrap();
        assert_eq!(x_transformed.dim(), (4, 3));

        // Transformed data should be finite
        for &val in x_transformed.iter() {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_kernel_pca_errors() {
        // Empty dataset
        let empty_x: Array2<f64> = Array2::zeros((0, 2));
        let result = KernelPCA::new().fit(&empty_x, &());
        assert!(result.is_err());

        // Zero features
        let zero_features_x: Array2<f64> = Array2::zeros((2, 0));
        let result = KernelPCA::new().fit(&zero_features_x, &());
        assert!(result.is_err());

        // Zero components
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let result = KernelPCA::new().n_components(0).fit(&x, &());
        assert!(result.is_err());
    }

    #[test]
    fn test_kernel_pca_feature_mismatch() {
        let x_train = array![[1.0, 2.0], [3.0, 4.0]];
        let x_test = array![[1.0, 2.0, 3.0]]; // Wrong number of features

        let kpca = KernelPCA::new().fit(&x_train, &()).unwrap();
        let result = kpca.transform(&x_test);

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Feature"));
    }

    #[test]
    fn test_kernel_pca_default() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];

        let kpca = KernelPCA::default().fit(&x, &()).unwrap();

        // Should use default parameters
        assert_eq!(kpca.n_components(), 2); // min(n_samples, n_features)
        assert_eq!(kpca.n_features_in(), 2);
    }

    #[test]
    fn test_new_kernel_functions() {
        let x = array![1.0, 2.0];
        let y = array![3.0, 4.0];

        // Laplacian kernel
        let laplacian = KernelFunction::Laplacian { gamma: 0.5 };
        let laplacian_result = laplacian.compute(&x, &y);
        // exp(-0.5 * (|1-3| + |2-4|)) = exp(-0.5 * 4) = exp(-2) ≈ 0.135
        assert!(laplacian_result > 0.0 && laplacian_result < 1.0);

        // Chi-squared kernel
        let chi_squared = KernelFunction::ChiSquared { gamma: 1.0 };
        let chi_result = chi_squared.compute(&x, &y);
        assert!(chi_result > 0.0 && chi_result <= 1.0);
    }

    #[test]
    fn test_kernel_pca_nystrom_approximation() {
        let x = array![
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
            [9.0, 10.0],
            [11.0, 12.0],
        ];

        let kpca = KernelPCA::new()
            .n_components(2)
            .kernel(KernelFunction::Rbf { gamma: 0.1 })
            .approximation(KernelApproximation::Nystrom { n_components: 4 })
            .random_state(42)
            .fit(&x, &())
            .unwrap();

        let x_transformed = kpca.transform(&x).unwrap();
        assert_eq!(x_transformed.dim(), (6, 2));

        // Check that eigenvalues are non-negative
        let eigenvalues = kpca.eigenvalues();
        for &val in eigenvalues.iter() {
            assert!(val >= 0.0, "Eigenvalue should be non-negative: {}", val);
        }
    }

    #[test]
    fn test_kernel_pca_random_sampling() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0],];

        let kpca = KernelPCA::new()
            .n_components(2)
            .kernel(KernelFunction::Linear)
            .approximation(KernelApproximation::RandomSampling { n_samples: 3 })
            .random_state(123)
            .fit(&x, &())
            .unwrap();

        let x_transformed = kpca.transform(&x).unwrap();
        assert_eq!(x_transformed.dim(), (5, 2));

        // Transformed data should be finite
        for &val in x_transformed.iter() {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_kernel_pca_laplacian_kernel() {
        let x = array![[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]];

        let kpca = KernelPCA::new()
            .n_components(2)
            .kernel(KernelFunction::Laplacian { gamma: 0.1 })
            .fit(&x, &())
            .unwrap();

        let x_transformed = kpca.transform(&x).unwrap();
        assert_eq!(x_transformed.dim(), (4, 2));

        // Check that all values are finite
        for &val in x_transformed.iter() {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_kernel_pca_chi_squared_kernel() {
        // Use positive data for chi-squared kernel (required for numerical stability)
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];

        let kpca = KernelPCA::new()
            .n_components(2)
            .kernel(KernelFunction::ChiSquared { gamma: 0.5 })
            .fit(&x, &())
            .unwrap();

        let x_transformed = kpca.transform(&x).unwrap();
        assert_eq!(x_transformed.dim(), (4, 2));

        // Check that all values are finite
        for &val in x_transformed.iter() {
            assert!(val.is_finite());
        }
    }
}
