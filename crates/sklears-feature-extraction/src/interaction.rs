//! Interaction and kernel-based feature generation
//!
//! This module provides transformers for generating interaction features
//! and kernel-based approximations for non-linear feature transformation.

use crate::*;
use rayon::prelude::*;
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use sklears_core::prelude::{Fit, SklearsError, Transform};

/// Radial Basis Functions Generator
///
/// Generate radial basis functions (RBFs) for non-linear feature transformation.
/// Supports various RBF types including Gaussian, Multiquadric, and Inverse Multiquadric.
///
/// # Parameters
///
/// * `n_centers` - Number of RBF centers
/// * `rbf_type` - Type of radial basis function
/// * `centers` - RBF center positions (if None, centers are determined from data)
/// * `shape_parameter` - Shape parameter for RBFs (affects function width)
/// * `normalize` - Whether to normalize the output features
///
/// # Examples
///
/// ```
/// use sklears_feature_extraction::interaction::{RadialBasisFunctions, RBFType};
/// use scirs2_core::ndarray::array;
///
/// let X = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
/// let rbf = RadialBasisFunctions::new()
///     .n_centers(3)
///     .rbf_type(RBFType::Gaussian)
///     .shape_parameter(1.0);
///
/// let features = rbf.transform(&X.view()).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct RadialBasisFunctions {
    n_centers: usize,
    rbf_type: RBFType,
    centers: Option<Array2<Float>>,
    shape_parameter: Float,
    normalize: bool,
}

#[derive(Debug, Clone, Copy)]
pub enum RBFType {
    /// Gaussian RBF: exp(-epsilon * r^2)
    Gaussian,
    /// Multiquadric RBF: sqrt(1 + epsilon * r^2)
    Multiquadric,
    /// Inverse Multiquadric RBF: 1 / sqrt(1 + epsilon * r^2)
    InverseMultiquadric,
    /// Thin Plate Spline RBF: r^2 * log(r)
    ThinPlateSpline,
    /// Linear RBF: r
    Linear,
    /// Cubic RBF: r^3
    Cubic,
    /// Quintic RBF: r^5
    Quintic,
}

impl RadialBasisFunctions {
    /// Create a new RadialBasisFunctions transformer
    pub fn new() -> Self {
        Self {
            n_centers: 10,
            rbf_type: RBFType::Gaussian,
            centers: None,
            shape_parameter: 1.0,
            normalize: false,
        }
    }

    /// Set the number of RBF centers
    pub fn n_centers(mut self, n_centers: usize) -> Self {
        self.n_centers = n_centers;
        self
    }

    /// Set the type of radial basis function
    pub fn rbf_type(mut self, rbf_type: RBFType) -> Self {
        self.rbf_type = rbf_type;
        self
    }

    /// Set custom RBF centers
    pub fn centers(mut self, centers: Array2<Float>) -> Self {
        self.centers = Some(centers);
        self
    }

    /// Set the shape parameter (epsilon)
    pub fn shape_parameter(mut self, shape_parameter: Float) -> Self {
        self.shape_parameter = shape_parameter;
        self
    }

    /// Set whether to normalize output features
    pub fn normalize(mut self, normalize: bool) -> Self {
        self.normalize = normalize;
        self
    }

    /// Transform input data using radial basis functions
    pub fn transform(&self, X: &ArrayView2<Float>) -> SklResult<Array2<Float>> {
        let (n_samples, n_features) = X.dim();

        // Generate centers if not provided
        let centers = match &self.centers {
            Some(centers) => centers.clone(),
            None => self.generate_centers(X)?,
        };

        let n_centers = centers.nrows();
        let mut rbf_features = Array2::zeros((n_samples, n_centers));

        // Compute RBF features for each sample
        for (sample_idx, sample) in X.axis_iter(Axis(0)).enumerate() {
            for (center_idx, center) in centers.axis_iter(Axis(0)).enumerate() {
                let distance = self.compute_distance(&sample, &center);
                let rbf_value = self.evaluate_rbf(distance);
                rbf_features[[sample_idx, center_idx]] = rbf_value;
            }
        }

        // Apply normalization if requested
        if self.normalize {
            self.normalize_features(&mut rbf_features);
        }

        Ok(rbf_features)
    }

    fn generate_centers(&self, X: &ArrayView2<Float>) -> SklResult<Array2<Float>> {
        let (n_samples, n_features) = X.dim();

        if self.n_centers > n_samples {
            return Err(SklearsError::InvalidInput(
                "Number of centers cannot exceed number of samples".to_string(),
            ));
        }

        // Use k-means style center selection (simplified)
        let mut centers = Array2::zeros((self.n_centers, n_features));

        // Initialize first center randomly (use first sample)
        if n_samples > 0 {
            for j in 0..n_features {
                centers[[0, j]] = X[[0, j]];
            }
        }

        // Select remaining centers using farthest-first traversal
        for center_idx in 1..self.n_centers {
            let mut best_sample_idx = 0;
            let mut max_min_distance = 0.0;

            for sample_idx in 0..n_samples {
                let sample = X.row(sample_idx);
                let mut min_distance = Float::INFINITY;

                // Find minimum distance to existing centers
                for existing_center_idx in 0..center_idx {
                    let center = centers.row(existing_center_idx);
                    let distance = self.compute_distance(&sample, &center);
                    min_distance = min_distance.min(distance);
                }

                // Select sample with maximum minimum distance
                if min_distance > max_min_distance {
                    max_min_distance = min_distance;
                    best_sample_idx = sample_idx;
                }
            }

            // Set the new center
            for j in 0..n_features {
                centers[[center_idx, j]] = X[[best_sample_idx, j]];
            }
        }

        Ok(centers)
    }

    fn compute_distance(&self, point1: &ArrayView1<Float>, point2: &ArrayView1<Float>) -> Float {
        // Euclidean distance
        let mut sum_squared = 0.0;
        for (x1, x2) in point1.iter().zip(point2.iter()) {
            sum_squared += (x1 - x2).powi(2);
        }
        sum_squared.sqrt()
    }

    fn evaluate_rbf(&self, distance: Float) -> Float {
        let r = distance;
        let epsilon = self.shape_parameter;

        match self.rbf_type {
            RBFType::Gaussian => (-epsilon * r * r).exp(),
            RBFType::Multiquadric => (1.0 + epsilon * r * r).sqrt(),
            RBFType::InverseMultiquadric => 1.0 / (1.0 + epsilon * r * r).sqrt(),
            RBFType::ThinPlateSpline => {
                if r < 1e-10 {
                    0.0
                } else {
                    r * r * r.ln()
                }
            }
            RBFType::Linear => r,
            RBFType::Cubic => r.powi(3),
            RBFType::Quintic => r.powi(5),
        }
    }

    fn normalize_features(&self, features: &mut Array2<Float>) {
        let (n_samples, n_features) = features.dim();

        for j in 0..n_features {
            let mut column_sum = 0.0;
            for i in 0..n_samples {
                column_sum += features[[i, j]];
            }

            if column_sum > 1e-10 {
                for i in 0..n_samples {
                    features[[i, j]] /= column_sum;
                }
            }
        }
    }
}

impl Default for RadialBasisFunctions {
    fn default() -> Self {
        Self::new()
    }
}

impl Transform<ArrayView2<'_, Float>, Array2<Float>> for RadialBasisFunctions {
    fn transform(&self, X: &ArrayView2<'_, Float>) -> SklResult<Array2<Float>> {
        self.transform(X)
    }
}

/// RBF Sampler for Random Fourier Features
///
/// Approximates the feature map of an RBF kernel by Monte Carlo approximation
/// of its Fourier transform. This allows the use of linear algorithms with
/// kernel-like performance.
///
/// # Parameters
///
/// * `gamma` - Parameter of the RBF kernel
/// * `n_components` - Number of Monte Carlo samples (feature dimension)
/// * `random_state` - Random seed for reproducibility
///
/// # Mathematical Background
///
/// For an RBF kernel K(x,y) = exp(-gamma||x-y||²), the Fourier transform is:
/// φ(x) = sqrt(2/n_components) * cos(wx + b)
/// where w is drawn from N(0, 2*gamma*I) and b is drawn from Uniform(0, 2π)
///
/// # Examples
///
/// ```
/// use sklears_feature_extraction::interaction::RBFSampler;
/// use scirs2_core::ndarray::array;
///
/// let X = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
/// let mut rbf_sampler = RBFSampler::new()
///     .gamma(1.0)
///     .n_components(100)
///     .random_state(Some(42));
///
/// let features = rbf_sampler.fit_transform(&X.view()).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct RBFSampler {
    gamma: Float,
    n_components: usize,
    random_state: Option<u64>,
    random_weights: Option<Array2<Float>>,
    random_offset: Option<Array1<Float>>,
}

impl RBFSampler {
    /// Create a new RBFSampler
    pub fn new() -> Self {
        Self {
            gamma: 1.0,
            n_components: 100,
            random_state: None,
            random_weights: None,
            random_offset: None,
        }
    }

    /// Set the gamma parameter
    pub fn gamma(mut self, gamma: Float) -> Self {
        self.gamma = gamma;
        self
    }

    /// Set the number of components
    pub fn n_components(mut self, n_components: usize) -> Self {
        self.n_components = n_components;
        self
    }

    /// Set the random state
    pub fn random_state(mut self, random_state: Option<u64>) -> Self {
        self.random_state = random_state;
        self
    }

    /// Fit the sampler to the data (generates random weights)
    pub fn fit(&mut self, X: &ArrayView2<Float>) -> SklResult<&mut Self> {
        let (_, n_features) = X.dim();

        // Generate random weights from N(0, 2*gamma*I)
        let variance = 2.0 * self.gamma;
        let mut random_weights = Array2::zeros((self.n_components, n_features));
        let mut random_offset = Array1::zeros(self.n_components);

        // Simple pseudo-random number generation
        let mut seed = self.random_state.unwrap_or(42);

        for i in 0..self.n_components {
            for j in 0..n_features {
                // Box-Muller transform for normal distribution
                let u1 = (seed % 1000) as Float / 1000.0;
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                let u2 = (seed % 1000) as Float / 1000.0;
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);

                let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                random_weights[[i, j]] = z * variance.sqrt();
            }

            // Random offset from Uniform(0, 2π)
            let u = (seed % 1000) as Float / 1000.0;
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            random_offset[i] = 2.0 * std::f64::consts::PI * u;
        }

        self.random_weights = Some(random_weights);
        self.random_offset = Some(random_offset);

        Ok(self)
    }

    /// Transform the data using the fitted random features
    pub fn transform(&self, X: &ArrayView2<Float>) -> SklResult<Array2<Float>> {
        let weights = self.random_weights.as_ref().ok_or_else(|| {
            SklearsError::InvalidInput("RBFSampler must be fitted before transform".to_string())
        })?;
        let offset = self.random_offset.as_ref().ok_or_else(|| {
            SklearsError::InvalidInput("RBFSampler must be fitted before transform".to_string())
        })?;

        let (n_samples, n_features) = X.dim();
        if weights.ncols() != n_features {
            return Err(SklearsError::InvalidInput(
                "Feature dimension mismatch".to_string(),
            ));
        }

        let mut features = Array2::zeros((n_samples, self.n_components));
        let normalization = (2.0 / self.n_components as Float).sqrt();

        for i in 0..n_samples {
            let sample = X.row(i);
            for j in 0..self.n_components {
                let weight_row = weights.row(j);
                let projection = sample.dot(&weight_row) + offset[j];
                features[[i, j]] = normalization * projection.cos();
            }
        }

        Ok(features)
    }

    /// Fit and transform in one step
    pub fn fit_transform(&mut self, X: &ArrayView2<Float>) -> SklResult<Array2<Float>> {
        self.fit(X)?;
        self.transform(X)
    }
}

impl Default for RBFSampler {
    fn default() -> Self {
        Self::new()
    }
}

/// Nyström Method for Kernel Approximation
///
/// Approximates kernel maps using a subset of training data as landmarks.
/// The Nyström method provides a low-rank approximation to the kernel matrix.
///
/// # Parameters
///
/// * `kernel` - Type of kernel to approximate
/// * `n_components` - Number of components (rank of approximation)
/// * `gamma` - Parameter for RBF kernel
/// * `degree` - Degree for polynomial kernel
/// * `coef0` - Independent term for polynomial kernel
///
/// # Mathematical Background
///
/// Given a kernel matrix K, the Nyström method approximates it as:
/// K ≈ C W^+ C^T
/// where C is the kernel matrix between all points and landmarks,
/// and W is the kernel matrix between landmarks only.
///
/// # Examples
///
/// ```
/// use sklears_feature_extraction::interaction::{Nystroem, NystromKernel};
/// use scirs2_core::ndarray::array;
///
/// let X = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
/// let nystroem = Nystroem::new()
///     .kernel(NystromKernel::RBF)
///     .gamma(1.0)
///     .n_components(50);
///
/// let mut fitted = nystroem;
/// let features = fitted.fit_transform(&X.view()).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct Nystroem {
    kernel: NystromKernel,
    n_components: usize,
    gamma: Float,
    degree: usize,
    coef0: Float,
    random_state: Option<u64>,
    landmarks: Option<Array2<Float>>,
    normalization_matrix: Option<Array2<Float>>,
}

#[derive(Debug, Clone, Copy)]
pub enum NystromKernel {
    /// RBF kernel: exp(-gamma * ||x - y||²)
    RBF,
    /// Polynomial kernel: (gamma * <x, y> + coef0)^degree
    Polynomial,
    /// Linear kernel: <x, y>
    Linear,
    /// Sigmoid kernel: tanh(gamma * <x, y> + coef0)
    Sigmoid,
}

impl Nystroem {
    /// Create a new Nyström approximator
    pub fn new() -> Self {
        Self {
            kernel: NystromKernel::RBF,
            n_components: 100,
            gamma: 1.0,
            degree: 3,
            coef0: 1.0,
            random_state: None,
            landmarks: None,
            normalization_matrix: None,
        }
    }

    /// Set the kernel type
    pub fn kernel(mut self, kernel: NystromKernel) -> Self {
        self.kernel = kernel;
        self
    }

    /// Set the number of components
    pub fn n_components(mut self, n_components: usize) -> Self {
        self.n_components = n_components;
        self
    }

    /// Set the gamma parameter
    pub fn gamma(mut self, gamma: Float) -> Self {
        self.gamma = gamma;
        self
    }

    /// Set the degree for polynomial kernel
    pub fn degree(mut self, degree: usize) -> Self {
        self.degree = degree;
        self
    }

    /// Set the coef0 parameter
    pub fn coef0(mut self, coef0: Float) -> Self {
        self.coef0 = coef0;
        self
    }

    /// Set the random state
    pub fn random_state(mut self, random_state: Option<u64>) -> Self {
        self.random_state = random_state;
        self
    }

    /// Fit the Nyström approximation
    #[allow(non_snake_case)]
    pub fn fit(&mut self, X: &ArrayView2<Float>) -> SklResult<&mut Self> {
        let (n_samples, n_features) = X.dim();
        let n_landmarks = self.n_components.min(n_samples);

        // Select landmark points (random subset)
        let landmark_indices = self.select_landmarks(n_samples, n_landmarks)?;
        let mut landmarks = Array2::zeros((n_landmarks, n_features));

        for (i, &idx) in landmark_indices.iter().enumerate() {
            landmarks.row_mut(i).assign(&X.row(idx));
        }

        // Compute W matrix (kernel between landmarks)
        let W = self.compute_kernel_matrix(&landmarks.view(), &landmarks.view())?;

        // Compute pseudo-inverse of W (simplified using regularization)
        let W_inv = self.compute_regularized_inverse(&W)?;

        self.landmarks = Some(landmarks);
        self.normalization_matrix = Some(W_inv);

        Ok(self)
    }

    /// Transform data using the fitted Nyström approximation
    pub fn transform(&self, X: &ArrayView2<Float>) -> SklResult<Array2<Float>> {
        let landmarks = self.landmarks.as_ref().ok_or_else(|| {
            SklearsError::InvalidInput("Nystroem must be fitted before transform".to_string())
        })?;
        let W_inv = self.normalization_matrix.as_ref().ok_or_else(|| {
            SklearsError::InvalidInput("Nystroem must be fitted before transform".to_string())
        })?;

        // Compute C matrix (kernel between X and landmarks)
        let C = self.compute_kernel_matrix(X, &landmarks.view())?;

        // Apply Nyström approximation: features = C * W_inv^(1/2)
        let features = C.dot(W_inv);

        Ok(features)
    }

    /// Fit and transform in one step
    pub fn fit_transform(&mut self, X: &ArrayView2<Float>) -> SklResult<Array2<Float>> {
        self.fit(X)?;
        self.transform(X)
    }

    fn select_landmarks(&self, n_samples: usize, n_landmarks: usize) -> SklResult<Vec<usize>> {
        let mut landmarks = Vec::new();
        let mut seed = self.random_state.unwrap_or(42);

        // Simple random selection
        for _ in 0..n_landmarks {
            let idx = (seed % n_samples as u64) as usize;
            landmarks.push(idx);
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        }

        Ok(landmarks)
    }

    fn compute_kernel_matrix(
        &self,
        X: &ArrayView2<Float>,
        Y: &ArrayView2<Float>,
    ) -> SklResult<Array2<Float>> {
        let (n_x, _) = X.dim();
        let (n_y, _) = Y.dim();
        let mut K = Array2::zeros((n_x, n_y));

        for i in 0..n_x {
            for j in 0..n_y {
                let x_i = X.row(i);
                let y_j = Y.row(j);
                K[[i, j]] = self.kernel_function(&x_i, &y_j);
            }
        }

        Ok(K)
    }

    fn kernel_function(&self, x: &ArrayView1<Float>, y: &ArrayView1<Float>) -> Float {
        match self.kernel {
            NystromKernel::RBF => {
                let mut squared_distance = 0.0;
                for (xi, yi) in x.iter().zip(y.iter()) {
                    squared_distance += (xi - yi).powi(2);
                }
                (-self.gamma * squared_distance).exp()
            }
            NystromKernel::Linear => x.dot(y),
            NystromKernel::Polynomial => {
                let dot_product = x.dot(y);
                (self.gamma * dot_product + self.coef0).powf(self.degree as Float)
            }
            NystromKernel::Sigmoid => {
                let dot_product = x.dot(y);
                (self.gamma * dot_product + self.coef0).tanh()
            }
        }
    }

    fn compute_regularized_inverse(&self, matrix: &Array2<Float>) -> SklResult<Array2<Float>> {
        let n = matrix.nrows();
        if n != matrix.ncols() {
            return Err(SklearsError::InvalidInput(
                "Matrix must be square for inversion".to_string(),
            ));
        }

        // Add regularization to diagonal
        let regularization = 1e-12;
        let mut A = matrix.clone();
        for i in 0..n {
            A[[i, i]] += regularization;
        }

        // Simplified matrix inversion using LU decomposition approximation
        // In practice, you would use a proper linear algebra library
        self.approximate_matrix_sqrt(&A)
    }

    fn approximate_matrix_sqrt(&self, matrix: &Array2<Float>) -> SklResult<Array2<Float>> {
        let n = matrix.nrows();

        // Simplified eigendecomposition for matrix square root
        // Using power iteration to find dominant eigenvalues/eigenvectors
        let max_components = n.min(self.n_components);
        let mut result = Array2::zeros((n, max_components));

        for comp in 0..max_components {
            let mut v = Array1::ones(n);

            // Normalize
            let norm = v.iter().map(|x| x * x).sum::<Float>().sqrt();
            if norm > 0.0 {
                v /= norm;
            }

            // Power iteration
            for _iter in 0..20 {
                let mut Av = Array1::zeros(n);
                for i in 0..n {
                    for j in 0..n {
                        Av[i] += matrix[[i, j]] * v[j];
                    }
                }

                // Orthogonalize against previous components
                for prev_comp in 0..comp {
                    let prev_vec = result.column(prev_comp);
                    let projection = Av.dot(&prev_vec);
                    for k in 0..n {
                        Av[k] -= projection * prev_vec[k];
                    }
                }

                let norm = Av.iter().map(|x| x * x).sum::<Float>().sqrt();
                if norm < 1e-10 {
                    break;
                }
                v = Av / norm;
            }

            // Compute eigenvalue
            let mut Av = Array1::zeros(n);
            for i in 0..n {
                for j in 0..n {
                    Av[i] += matrix[[i, j]] * v[j];
                }
            }
            let eigenvalue = v.dot(&Av).max(1e-10);

            // Store sqrt(eigenvalue) * eigenvector
            let sqrt_eigenvalue = eigenvalue.sqrt();
            for i in 0..n {
                result[[i, comp]] = sqrt_eigenvalue * v[i];
            }
        }

        Ok(result)
    }
}

impl Default for Nystroem {
    fn default() -> Self {
        Self::new()
    }
}

/// Additive Chi-Squared Sampler
///
/// Approximate feature map for additive chi-squared kernel using random sampling.
/// The additive chi-squared kernel is defined as:
/// k(x, y) = exp(-gamma * sum_i (x_i - y_i)² / (x_i + y_i))
///
/// # Parameters
///
/// * `sample_steps` - Number of sampling steps
/// * `sample_interval` - Interval for sampling
/// * `n_components` - Number of components to generate
///
/// # Examples
///
/// ```
/// use sklears_feature_extraction::interaction::AdditiveChi2Sampler;
/// use scirs2_core::ndarray::array;
///
/// let X = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
/// let sampler = AdditiveChi2Sampler::new()
///     .sample_steps(2)
///     .sample_interval(0.5);
///
/// let features = sampler.transform(&X.view()).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct AdditiveChi2Sampler {
    sample_steps: usize,
    sample_interval: Float,
}

impl AdditiveChi2Sampler {
    /// Create a new AdditiveChi2Sampler
    pub fn new() -> Self {
        Self {
            sample_steps: 2,
            sample_interval: 1.0,
        }
    }

    /// Set the number of sampling steps
    pub fn sample_steps(mut self, sample_steps: usize) -> Self {
        self.sample_steps = sample_steps;
        self
    }

    /// Set the sampling interval
    pub fn sample_interval(mut self, sample_interval: Float) -> Self {
        self.sample_interval = sample_interval;
        self
    }

    /// Transform data using additive chi-squared features
    pub fn transform(&self, X: &ArrayView2<Float>) -> SklResult<Array2<Float>> {
        let (n_samples, n_features) = X.dim();
        let n_components = 2 * n_features * self.sample_steps;
        let mut features = Array2::zeros((n_samples, n_components));

        for (sample_idx, sample) in X.axis_iter(Axis(0)).enumerate() {
            let mut feature_idx = 0;

            for &x_i in sample.iter() {
                for step in 0..self.sample_steps {
                    let omega = (step + 1) as Float * self.sample_interval;

                    if x_i > 0.0 {
                        let sqrt_x = x_i.sqrt();
                        let cos_val = (2.0 * omega * sqrt_x).cos();
                        let sin_val = (2.0 * omega * sqrt_x).sin();

                        features[[sample_idx, feature_idx]] =
                            (2.0 / std::f64::consts::PI).sqrt() * cos_val;
                        features[[sample_idx, feature_idx + 1]] =
                            (2.0 / std::f64::consts::PI).sqrt() * sin_val;
                    }

                    feature_idx += 2;
                }
            }
        }

        Ok(features)
    }

    /// Get the number of output features
    pub fn get_n_output_features(&self, n_input_features: usize) -> usize {
        2 * n_input_features * self.sample_steps
    }
}

impl Default for AdditiveChi2Sampler {
    fn default() -> Self {
        Self::new()
    }
}

impl Transform<ArrayView2<'_, Float>, Array2<Float>> for AdditiveChi2Sampler {
    fn transform(&self, X: &ArrayView2<'_, Float>) -> SklResult<Array2<Float>> {
        self.transform(X)
    }
}
