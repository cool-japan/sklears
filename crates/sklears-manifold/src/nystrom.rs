//! Nyström Approximation for Kernel Methods
//! This module provides efficient approximation methods for kernel matrices using
//! the Nyström method, which enables scalable kernel-based manifold learning
//! algorithms by approximating large kernel matrices with smaller submatrices.

use scirs2_core::ndarray::ndarray_linalg::{Eigh, UPLO};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, Axis};
use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::thread_rng;
use scirs2_core::random::Rng;
use scirs2_core::random::{seq::SliceRandom, SeedableRng};
use scirs2_core::SliceRandomExt;
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Fit, Transform},
    types::Float,
};

/// Nyström Approximation for Kernel Methods
///
/// The Nyström method approximates a large kernel matrix by sampling a subset of
/// columns and using matrix factorization to reconstruct the full matrix. This
/// enables scalable kernel-based manifold learning for large datasets.
///
/// The method works by:
/// 1. Sampling a subset of landmark points
/// 2. Computing the kernel matrix between landmarks and all points
/// 3. Using eigendecomposition to approximate the full kernel matrix
///
/// # Parameters
///
/// * `n_components` - Number of components to use in the approximation
/// * `n_landmarks` - Number of landmark points to sample
/// * `kernel` - Kernel function to use ('rbf', 'polynomial', 'linear')
/// * `gamma` - Kernel coefficient for RBF kernel
/// * `degree` - Degree for polynomial kernel
/// * `coef0` - Independent term for polynomial kernel
/// * `random_state` - Random state for reproducibility
/// * `selection_method` - Method for selecting landmarks ('random', 'kmeans', 'uniform')
///
/// # Examples
///
/// ```rust,ignore
/// use sklears_manifold::nystrom::NystromApproximation;
/// use sklears_core::traits::{Transform, Fit};
/// use scirs2_core::ndarray::{ Array1, ArrayView1};
///
/// let data = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
///
/// let nystrom = NystromApproximation::new(2, 3);
/// let fitted = nystrom.fit(&data, &()).unwrap();
/// let approximation = fitted.transform(&data).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct NystromApproximation {
    n_components: usize,
    n_landmarks: usize,
    kernel: String,
    gamma: Option<Float>,
    degree: usize,
    coef0: Float,
    random_state: Option<u64>,
    selection_method: String,
}

impl NystromApproximation {
    /// Create a new Nyström approximation instance
    pub fn new(n_components: usize, n_landmarks: usize) -> Self {
        Self {
            n_components,
            n_landmarks,
            kernel: "rbf".to_string(),
            gamma: None,
            degree: 3,
            coef0: 1.0,
            random_state: None,
            selection_method: "random".to_string(),
        }
    }

    /// Set the kernel function
    pub fn kernel(mut self, kernel: &str) -> Self {
        self.kernel = kernel.to_string();
        self
    }

    /// Set the gamma parameter for RBF kernel
    pub fn gamma(mut self, gamma: Float) -> Self {
        self.gamma = Some(gamma);
        self
    }

    /// Set the degree for polynomial kernel
    pub fn degree(mut self, degree: usize) -> Self {
        self.degree = degree;
        self
    }

    /// Set the coef0 parameter for polynomial kernel
    pub fn coef0(mut self, coef0: Float) -> Self {
        self.coef0 = coef0;
        self
    }

    /// Set random state for reproducibility
    pub fn random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }

    /// Set the landmark selection method
    pub fn selection_method(mut self, method: &str) -> Self {
        self.selection_method = method.to_string();
        self
    }
}

/// Fitted Nyström Approximation model
#[derive(Debug, Clone)]
pub struct FittedNystromApproximation {
    n_components: usize,
    landmarks: Array2<Float>,
    landmark_kernel: Array2<Float>,
    normalization_matrix: Array2<Float>,
    eigenvalues: Array1<Float>,
    eigenvectors: Array2<Float>,
    kernel: String,
    gamma: Float,
    degree: usize,
    coef0: Float,
}

impl Fit<Array2<Float>, ()> for NystromApproximation {
    type Fitted = FittedNystromApproximation;

    fn fit(self, data: &Array2<Float>, _y: &()) -> SklResult<Self::Fitted> {
        let x = data;
        let n_samples = x.nrows();
        let n_features = x.ncols();

        if self.n_landmarks > n_samples {
            return Err(SklearsError::InvalidInput(format!(
                "n_landmarks ({}) cannot be greater than n_samples ({})",
                self.n_landmarks, n_samples
            )));
        }

        if self.n_components > self.n_landmarks {
            return Err(SklearsError::InvalidInput(format!(
                "n_components ({}) cannot be greater than n_landmarks ({})",
                self.n_components, self.n_landmarks
            )));
        }

        // Select landmarks
        let landmark_indices = self.select_landmarks(x)?;
        let landmarks = select_rows_by_indices(x, &landmark_indices);

        // Compute gamma for RBF kernel if not provided
        let gamma = if let Some(g) = self.gamma {
            g
        } else {
            // Use the median heuristic: gamma = 1 / (2 * median_distance^2)
            let median_distance = compute_median_distance(x)?;
            1.0 / (2.0 * median_distance * median_distance)
        };

        // Compute kernel matrix between landmarks
        let landmark_kernel = compute_kernel_matrix(
            &landmarks,
            &landmarks,
            &self.kernel,
            gamma,
            self.degree,
            self.coef0,
        )?;

        // Compute eigendecomposition of landmark kernel matrix
        let (eigenvalues, eigenvectors) = landmark_kernel.eigh(UPLO::Upper).map_err(|e| {
            SklearsError::NumericalError(format!("Eigendecomposition failed: {}", e))
        })?;

        // Sort eigenvalues and eigenvectors in descending order
        let mut eigen_pairs: Vec<(Float, Array1<Float>)> = eigenvalues
            .iter()
            .zip(eigenvectors.axis_iter(Axis(1)))
            .map(|(&val, vec)| (val, vec.to_owned()))
            .collect();

        eigen_pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

        // Take top n_components with positive eigenvalues
        let mut selected_eigenvalues = Vec::new();
        let mut selected_eigenvectors = Vec::new();

        for (val, vec) in eigen_pairs.into_iter().take(self.n_components) {
            if val > 1e-10 {
                selected_eigenvalues.push(val);
                selected_eigenvectors.push(vec);
            }
        }

        if selected_eigenvalues.is_empty() {
            return Err(SklearsError::NumericalError(
                "No positive eigenvalues found in landmark kernel matrix".to_string(),
            ));
        }

        let eigenvalues = Array1::from_vec(selected_eigenvalues);
        let eigenvectors =
            Array2::from_shape_fn((self.n_landmarks, eigenvalues.len()), |(i, j)| {
                selected_eigenvectors[j][i]
            });

        // Compute normalization matrix for the approximation
        let sqrt_eigenvalues = eigenvalues.mapv(|x| x.sqrt());
        let normalization_matrix = &eigenvectors / &sqrt_eigenvalues.insert_axis(Axis(0));

        Ok(FittedNystromApproximation {
            n_components: eigenvalues.len(),
            landmarks,
            landmark_kernel,
            normalization_matrix,
            eigenvalues,
            eigenvectors,
            kernel: self.kernel.clone(),
            gamma,
            degree: self.degree,
            coef0: self.coef0,
        })
    }
}

impl Transform<Array2<Float>, Array2<Float>> for FittedNystromApproximation {
    fn transform(&self, data: &Array2<Float>) -> SklResult<Array2<Float>> {
        let x = data;
        let n_samples = x.nrows();

        // Compute kernel matrix between data points and landmarks
        let kernel_matrix = compute_kernel_matrix(
            x,
            &self.landmarks,
            &self.kernel,
            self.gamma,
            self.degree,
            self.coef0,
        )?;

        // Apply Nyström approximation
        let approximation = kernel_matrix.dot(&self.normalization_matrix);

        Ok(approximation)
    }
}

impl FittedNystromApproximation {
    /// Get the selected landmarks
    pub fn landmarks(&self) -> &Array2<Float> {
        &self.landmarks
    }

    /// Get the landmark kernel matrix
    pub fn landmark_kernel(&self) -> &Array2<Float> {
        &self.landmark_kernel
    }

    /// Get the eigenvalues of the landmark kernel matrix
    pub fn eigenvalues(&self) -> &Array1<Float> {
        &self.eigenvalues
    }

    /// Get the eigenvectors of the landmark kernel matrix
    pub fn eigenvectors(&self) -> &Array2<Float> {
        &self.eigenvectors
    }

    /// Reconstruct the full kernel matrix approximation
    pub fn reconstruct_kernel(&self, data: &Array2<Float>) -> SklResult<Array2<Float>> {
        let kernel_to_landmarks = compute_kernel_matrix(
            data,
            &self.landmarks,
            &self.kernel,
            self.gamma,
            self.degree,
            self.coef0,
        )?;

        // K_approx = K_nl @ K_ll^(-1) @ K_ln
        // Using eigendecomposition: K_ll^(-1) = U @ Λ^(-1) @ U^T
        // So K_approx = K_nl @ U @ Λ^(-1) @ U^T @ K_nl^T

        // Compute the pseudoinverse of the landmark kernel matrix using eigendecomposition
        let lambda_inv = self
            .eigenvalues
            .mapv(|x| if x > 1e-10 { 1.0 / x } else { 0.0 });

        // K_ll^(-1) = U @ Λ^(-1) @ U^T
        let temp1 = &self.eigenvectors * &lambda_inv.insert_axis(Axis(0));
        let k_ll_inv = temp1.dot(&self.eigenvectors.t());

        // K_approx = K_nl @ K_ll^(-1) @ K_nl^T
        let temp2 = kernel_to_landmarks.dot(&k_ll_inv);
        let reconstructed = temp2.dot(&kernel_to_landmarks.t());

        Ok(reconstructed)
    }
}

impl NystromApproximation {
    fn select_landmarks(&self, data: &Array2<Float>) -> SklResult<Vec<usize>> {
        let n_samples = data.nrows();

        let mut rng = if let Some(seed) = self.random_state {
            StdRng::seed_from_u64(seed)
        } else {
            StdRng::seed_from_u64(thread_rng().gen())
        };

        match self.selection_method.as_str() {
            "random" => {
                let mut indices: Vec<usize> = (0..n_samples).collect();
                indices.shuffle(&mut rng);
                Ok(indices.into_iter().take(self.n_landmarks).collect())
            }
            "uniform" => {
                let step = n_samples as Float / self.n_landmarks as Float;
                let indices: Vec<usize> = (0..self.n_landmarks)
                    .map(|i| ((i as Float * step) as usize).min(n_samples - 1))
                    .collect();
                Ok(indices)
            }
            "kmeans" => {
                // Simple k-means++ initialization for landmark selection
                self.select_landmarks_kmeans(data, &mut rng)
            }
            _ => Err(SklearsError::InvalidInput(format!(
                "Unknown selection method: {}",
                self.selection_method
            ))),
        }
    }

    fn select_landmarks_kmeans(
        &self,
        data: &Array2<Float>,
        rng: &mut StdRng,
    ) -> SklResult<Vec<usize>> {
        let n_samples = data.nrows();
        let mut selected_indices = Vec::new();

        // Select first landmark randomly
        let first_index = rng.gen_range(0..n_samples);
        selected_indices.push(first_index);

        // Select remaining landmarks using k-means++ strategy
        for _ in 1..self.n_landmarks {
            let mut distances = Vec::new();
            let mut total_distance = 0.0;

            for i in 0..n_samples {
                if selected_indices.contains(&i) {
                    distances.push(0.0);
                    continue;
                }

                // Find distance to nearest selected landmark
                let mut min_distance = Float::INFINITY;
                for &selected_idx in &selected_indices {
                    let dist =
                        compute_euclidean_distance_squared(&data.row(i), &data.row(selected_idx));
                    min_distance = min_distance.min(dist);
                }

                distances.push(min_distance);
                total_distance += min_distance;
            }

            // Select next landmark with probability proportional to squared distance
            let threshold = rng.random::<Float>() * total_distance;
            let mut cumulative = 0.0;
            let mut selected_idx = 0;

            for (i, &dist) in distances.iter().enumerate() {
                cumulative += dist;
                if cumulative >= threshold {
                    selected_idx = i;
                    break;
                }
            }

            selected_indices.push(selected_idx);
        }

        Ok(selected_indices)
    }
}

/// Incremental Nyström Approximation
///
/// This variant allows for incremental updates to the Nyström approximation
/// as new data becomes available, making it suitable for streaming applications.
///
/// # Parameters
///
/// * `n_components` - Number of components to use in the approximation
/// * `n_landmarks` - Number of landmark points to maintain
/// * `kernel` - Kernel function to use
/// * `gamma` - Kernel coefficient for RBF kernel
/// * `update_rate` - Rate at which to update the approximation (0.0 to 1.0)
/// * `random_state` - Random state for reproducibility
#[derive(Debug, Clone)]
pub struct IncrementalNystromApproximation {
    n_components: usize,
    n_landmarks: usize,
    kernel: String,
    gamma: Option<Float>,
    degree: usize,
    coef0: Float,
    update_rate: Float,
    random_state: Option<u64>,
}

impl IncrementalNystromApproximation {
    /// Create a new incremental Nyström approximation instance
    pub fn new(n_components: usize, n_landmarks: usize) -> Self {
        Self {
            n_components,
            n_landmarks,
            kernel: "rbf".to_string(),
            gamma: None,
            degree: 3,
            coef0: 1.0,
            update_rate: 0.1,
            random_state: None,
        }
    }

    /// Set the update rate for incremental updates
    pub fn update_rate(mut self, rate: Float) -> Self {
        self.update_rate = rate;
        self
    }

    /// Set the kernel function
    pub fn kernel(mut self, kernel: &str) -> Self {
        self.kernel = kernel.to_string();
        self
    }

    /// Set the gamma parameter for RBF kernel
    pub fn gamma(mut self, gamma: Float) -> Self {
        self.gamma = Some(gamma);
        self
    }

    /// Set random state for reproducibility
    pub fn random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }
}

/// Fitted Incremental Nyström Approximation model
#[derive(Debug, Clone)]
pub struct FittedIncrementalNystromApproximation {
    n_components: usize,
    landmarks: Array2<Float>,
    landmark_kernel: Array2<Float>,
    normalization_matrix: Array2<Float>,
    eigenvalues: Array1<Float>,
    eigenvectors: Array2<Float>,
    kernel: String,
    gamma: Float,
    degree: usize,
    coef0: Float,
    update_rate: Float,
    n_updates: usize,
}

impl Fit<Array2<Float>, ()> for IncrementalNystromApproximation {
    type Fitted = FittedIncrementalNystromApproximation;

    fn fit(self, data: &Array2<Float>, _y: &()) -> SklResult<Self::Fitted> {
        // Initial fit using regular Nyström approximation
        let mut nystrom = NystromApproximation::new(self.n_components, self.n_landmarks)
            .kernel(&self.kernel)
            .degree(self.degree)
            .coef0(self.coef0)
            .random_state(self.random_state.unwrap_or(42));

        if let Some(gamma) = self.gamma {
            nystrom = nystrom.gamma(gamma);
        }

        let fitted = nystrom.fit(data, &())?;

        Ok(FittedIncrementalNystromApproximation {
            n_components: fitted.n_components,
            landmarks: fitted.landmarks,
            landmark_kernel: fitted.landmark_kernel,
            normalization_matrix: fitted.normalization_matrix,
            eigenvalues: fitted.eigenvalues,
            eigenvectors: fitted.eigenvectors,
            kernel: fitted.kernel,
            gamma: fitted.gamma,
            degree: fitted.degree,
            coef0: fitted.coef0,
            update_rate: self.update_rate,
            n_updates: 0,
        })
    }
}

impl Transform<Array2<Float>, Array2<Float>> for FittedIncrementalNystromApproximation {
    fn transform(&self, data: &Array2<Float>) -> SklResult<Array2<Float>> {
        let x = data;
        let n_samples = x.nrows();

        // Compute kernel matrix between data points and landmarks
        let kernel_matrix = compute_kernel_matrix(
            x,
            &self.landmarks,
            &self.kernel,
            self.gamma,
            self.degree,
            self.coef0,
        )?;

        // Apply Nyström approximation
        let approximation = kernel_matrix.dot(&self.normalization_matrix);

        Ok(approximation)
    }
}

impl FittedIncrementalNystromApproximation {
    /// Update the approximation with new data
    pub fn partial_fit(&mut self, data: &Array2<Float>) -> SklResult<()> {
        let n_new_samples = data.nrows();

        // Determine how many landmarks to replace
        let n_landmarks = self.landmarks.nrows();
        let n_replace = ((n_landmarks as Float * self.update_rate) as usize).max(1);

        // Select random landmarks to replace
        let mut rng = StdRng::seed_from_u64(thread_rng().gen());
        let mut replace_indices: Vec<usize> = (0..n_landmarks).collect();
        replace_indices.shuffle(&mut rng);
        replace_indices.truncate(n_replace);

        // Select new landmarks from new data
        let mut new_landmark_indices: Vec<usize> = (0..n_new_samples).collect();
        new_landmark_indices.shuffle(&mut rng);
        new_landmark_indices.truncate(n_replace);

        // Update landmarks
        for (i, &replace_idx) in replace_indices.iter().enumerate() {
            if i < new_landmark_indices.len() {
                let new_landmark_idx = new_landmark_indices[i];
                self.landmarks
                    .row_mut(replace_idx)
                    .assign(&data.row(new_landmark_idx));
            }
        }

        // Recompute kernel matrix and decomposition
        self.landmark_kernel = compute_kernel_matrix(
            &self.landmarks,
            &self.landmarks,
            &self.kernel,
            self.gamma,
            self.degree,
            self.coef0,
        )?;

        let (eigenvalues, eigenvectors) = self.landmark_kernel.eigh(UPLO::Upper).map_err(|e| {
            SklearsError::NumericalError(format!("Eigendecomposition failed: {}", e))
        })?;

        // Sort eigenvalues and eigenvectors in descending order
        let mut eigen_pairs: Vec<(Float, Array1<Float>)> = eigenvalues
            .iter()
            .zip(eigenvectors.axis_iter(Axis(1)))
            .map(|(&val, vec)| (val, vec.to_owned()))
            .collect();

        eigen_pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

        // Take top n_components with positive eigenvalues
        let mut selected_eigenvalues = Vec::new();
        let mut selected_eigenvectors = Vec::new();

        for (val, vec) in eigen_pairs.into_iter().take(self.n_components) {
            if val > 1e-10 {
                selected_eigenvalues.push(val);
                selected_eigenvectors.push(vec);
            }
        }

        if !selected_eigenvalues.is_empty() {
            self.eigenvalues = Array1::from_vec(selected_eigenvalues);
            self.eigenvectors =
                Array2::from_shape_fn((n_landmarks, self.eigenvalues.len()), |(i, j)| {
                    selected_eigenvectors[j][i]
                });

            // Update normalization matrix
            let sqrt_eigenvalues = self.eigenvalues.mapv(|x| x.sqrt());
            self.normalization_matrix = &self.eigenvectors / &sqrt_eigenvalues.insert_axis(Axis(0));
        }

        self.n_updates += 1;

        Ok(())
    }

    /// Get the number of updates performed
    pub fn n_updates(&self) -> usize {
        self.n_updates
    }
}

// Helper functions

fn select_rows_by_indices(data: &Array2<Float>, indices: &[usize]) -> Array2<Float> {
    let n_features = data.ncols();
    let mut result = Array2::zeros((indices.len(), n_features));

    for (i, &idx) in indices.iter().enumerate() {
        result.row_mut(i).assign(&data.row(idx));
    }

    result
}

fn compute_kernel_matrix(
    x: &Array2<Float>,
    y: &Array2<Float>,
    kernel: &str,
    gamma: Float,
    degree: usize,
    coef0: Float,
) -> SklResult<Array2<Float>> {
    let n_x = x.nrows();
    let n_y = y.nrows();
    let mut kernel_matrix = Array2::zeros((n_x, n_y));

    match kernel {
        "rbf" => {
            for i in 0..n_x {
                for j in 0..n_y {
                    let dist_sq = compute_euclidean_distance_squared(&x.row(i), &y.row(j));
                    kernel_matrix[[i, j]] = (-gamma * dist_sq).exp();
                }
            }
        }
        "polynomial" => {
            for i in 0..n_x {
                for j in 0..n_y {
                    let dot_product = x.row(i).dot(&y.row(j));
                    kernel_matrix[[i, j]] = (gamma * dot_product + coef0).powf(degree as Float);
                }
            }
        }
        "linear" => {
            kernel_matrix = x.dot(&y.t());
        }
        _ => {
            return Err(SklearsError::InvalidInput(format!(
                "Unknown kernel: {}",
                kernel
            )));
        }
    }

    Ok(kernel_matrix)
}

fn compute_euclidean_distance_squared(a: &ArrayView1<Float>, b: &ArrayView1<Float>) -> Float {
    (a - b).mapv(|x| x * x).sum()
}

fn compute_median_distance(data: &Array2<Float>) -> SklResult<Float> {
    let n_samples = data.nrows();

    if n_samples < 2 {
        return Err(SklearsError::NumericalError(
            "Need at least 2 samples to compute distances".to_string(),
        ));
    }

    let mut distances = Vec::new();

    // For small datasets, compute all pairwise distances
    if n_samples <= 100 {
        for i in 0..n_samples {
            for j in (i + 1)..n_samples {
                let dist = compute_euclidean_distance_squared(&data.row(i), &data.row(j)).sqrt();
                distances.push(dist);
            }
        }
    } else {
        // For larger datasets, sample a subset of distances
        let sample_size = (n_samples * n_samples / 4).min(10000);
        let mut rng = StdRng::seed_from_u64(thread_rng().gen());

        while distances.len() < sample_size {
            let i = rng.gen_range(0..n_samples);
            let j = rng.gen_range(0..n_samples);
            if i != j {
                let dist = compute_euclidean_distance_squared(&data.row(i), &data.row(j)).sqrt();
                distances.push(dist);
            }
        }
    }

    if distances.is_empty() {
        return Err(SklearsError::NumericalError(
            "No distances computed".to_string(),
        ));
    }

    distances.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = distances[distances.len() / 2];

    Ok(median)
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_nystrom_approximation_basic() {
        let data = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]];

        let nystrom = NystromApproximation::new(2, 3);
        let fitted = nystrom.fit(&data, &()).unwrap();
        let approximation = fitted.transform(&data).unwrap();

        assert_eq!(approximation.nrows(), 5);
        assert!(approximation.ncols() <= 2);
        assert!(approximation.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_nystrom_approximation_rbf_kernel() {
        let data = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];

        let nystrom = NystromApproximation::new(2, 3).kernel("rbf").gamma(1.0);
        let fitted = nystrom.fit(&data, &()).unwrap();
        let approximation = fitted.transform(&data).unwrap();

        assert_eq!(approximation.nrows(), 4);
        assert!(approximation.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_nystrom_approximation_polynomial_kernel() {
        let data = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];

        let nystrom = NystromApproximation::new(2, 3)
            .kernel("polynomial")
            .gamma(1.0)
            .degree(2)
            .coef0(1.0);
        let fitted = nystrom.fit(&data, &()).unwrap();
        let approximation = fitted.transform(&data).unwrap();

        assert_eq!(approximation.nrows(), 4);
        assert!(approximation.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_nystrom_approximation_linear_kernel() {
        let data = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];

        let nystrom = NystromApproximation::new(2, 3).kernel("linear");
        let fitted = nystrom.fit(&data, &()).unwrap();
        let approximation = fitted.transform(&data).unwrap();

        assert_eq!(approximation.nrows(), 4);
        assert!(approximation.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_nystrom_landmark_selection_methods() {
        let data = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]];

        // Test random selection
        let nystrom_random = NystromApproximation::new(2, 3)
            .selection_method("random")
            .random_state(42);
        let fitted_random = nystrom_random.fit(&data, &()).unwrap();
        assert_eq!(fitted_random.landmarks().nrows(), 3);

        // Test uniform selection
        let nystrom_uniform = NystromApproximation::new(2, 3).selection_method("uniform");
        let fitted_uniform = nystrom_uniform.fit(&data, &()).unwrap();
        assert_eq!(fitted_uniform.landmarks().nrows(), 3);

        // Test k-means selection
        let nystrom_kmeans = NystromApproximation::new(2, 3)
            .selection_method("kmeans")
            .random_state(42);
        let fitted_kmeans = nystrom_kmeans.fit(&data, &()).unwrap();
        assert_eq!(fitted_kmeans.landmarks().nrows(), 3);
    }

    #[test]
    fn test_incremental_nystrom_approximation() {
        let data1 = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];

        let data2 = array![[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]];

        let incremental = IncrementalNystromApproximation::new(2, 3)
            .update_rate(0.5)
            .random_state(42);
        let mut fitted = incremental.fit(&data1, &()).unwrap();

        // Initial transform
        let approximation1 = fitted.transform(&data1).unwrap();
        assert_eq!(approximation1.nrows(), 3);
        assert!(approximation1.iter().all(|&x| x.is_finite()));

        // Update with new data
        fitted.partial_fit(&data2).unwrap();
        assert_eq!(fitted.n_updates(), 1);

        // Transform with new data
        let approximation2 = fitted.transform(&data2).unwrap();
        assert_eq!(approximation2.nrows(), 3);
        assert!(approximation2.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_kernel_matrix_reconstruction() {
        let data = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];

        // Test with full rank approximation (should be nearly perfect)
        let nystrom = NystromApproximation::new(3, 3).kernel("rbf").gamma(1.0);
        let fitted = nystrom.fit(&data, &()).unwrap();
        let reconstructed = fitted.reconstruct_kernel(&data).unwrap();

        assert_eq!(reconstructed.shape(), &[3, 3]);
        assert!(reconstructed.iter().all(|&x| x.is_finite()));

        // Check that the matrix is symmetric
        for i in 0..3 {
            for j in 0..3 {
                assert_abs_diff_eq!(
                    reconstructed[[i, j]],
                    reconstructed[[j, i]],
                    epsilon = 1e-10
                );
            }
        }

        // With full rank, diagonal elements should be very close to 1 (for RBF kernel)
        for i in 0..3 {
            assert!(reconstructed[[i, i]] > 0.99); // Should be very close to 1
        }
    }

    #[test]
    fn test_kernel_matrix_reconstruction_low_rank() {
        let data = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];

        // Test with low-rank approximation (should still be reasonable)
        let nystrom = NystromApproximation::new(2, 3).kernel("rbf").gamma(1.0);
        let fitted = nystrom.fit(&data, &()).unwrap();
        let reconstructed = fitted.reconstruct_kernel(&data).unwrap();

        assert_eq!(reconstructed.shape(), &[3, 3]);
        assert!(reconstructed.iter().all(|&x| x.is_finite()));

        // Check that the matrix is symmetric
        for i in 0..3 {
            for j in 0..3 {
                assert_abs_diff_eq!(
                    reconstructed[[i, j]],
                    reconstructed[[j, i]],
                    epsilon = 1e-10
                );
            }
        }

        // With low-rank approximation, diagonal elements should be positive but may be < 1
        for i in 0..3 {
            assert!(reconstructed[[i, i]] > 0.0); // Should be positive
        }
    }

    #[test]
    fn test_nystrom_invalid_parameters() {
        let data = array![[1.0, 2.0], [3.0, 4.0]];

        // Test n_landmarks > n_samples
        let nystrom = NystromApproximation::new(2, 5);
        assert!(nystrom.fit(&data, &()).is_err());

        // Test n_components > n_landmarks
        let nystrom = NystromApproximation::new(5, 2);
        assert!(nystrom.fit(&data, &()).is_err());

        // Test unknown kernel
        let nystrom = NystromApproximation::new(2, 2).kernel("unknown");
        assert!(nystrom.fit(&data, &()).is_err());
    }
}
