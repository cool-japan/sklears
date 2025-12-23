//! Nyström method for kernel approximation
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::rngs::StdRng as RealStdRng;
use scirs2_core::random::seq::SliceRandom;
use scirs2_core::random::Rng;
use scirs2_core::random::{thread_rng, SeedableRng};
use sklears_core::{
    error::{Result, SklearsError},
    prelude::{Fit, Transform},
    traits::{Estimator, Trained, Untrained},
    types::Float,
};
use std::marker::PhantomData;

/// Sampling strategy for Nyström approximation
#[derive(Debug, Clone)]
/// SamplingStrategy
pub enum SamplingStrategy {
    /// Random uniform sampling
    Random,
    /// K-means clustering based sampling
    KMeans,
    /// Leverage score based sampling
    LeverageScore,
    /// Column norm based sampling
    ColumnNorm,
}

/// Kernel type for Nystroem approximation
#[derive(Debug, Clone)]
/// Kernel
pub enum Kernel {
    /// Linear kernel: K(x,y) = x^T y
    Linear,
    /// RBF kernel: K(x,y) = exp(-gamma * ||x-y||²)
    Rbf { gamma: Float },
    /// Polynomial kernel: K(x,y) = (gamma * x^T y + coef0)^degree
    Polynomial {
        gamma: Float,

        coef0: Float,

        degree: u32,
    },
}

impl Kernel {
    /// Compute kernel matrix between X and Y
    pub fn compute_kernel(&self, x: &Array2<Float>, y: &Array2<Float>) -> Array2<Float> {
        let (n_x, _) = x.dim();
        let (n_y, _) = y.dim();
        let mut kernel_matrix = Array2::zeros((n_x, n_y));

        match self {
            Kernel::Linear => {
                kernel_matrix = x.dot(&y.t());
            }
            Kernel::Rbf { gamma } => {
                for i in 0..n_x {
                    for j in 0..n_y {
                        let diff = &x.row(i) - &y.row(j);
                        let dist_sq = diff.dot(&diff);
                        kernel_matrix[[i, j]] = (-gamma * dist_sq).exp();
                    }
                }
            }
            Kernel::Polynomial {
                gamma,
                coef0,
                degree,
            } => {
                for i in 0..n_x {
                    for j in 0..n_y {
                        let dot_prod = x.row(i).dot(&y.row(j));
                        kernel_matrix[[i, j]] = (gamma * dot_prod + coef0).powf(*degree as Float);
                    }
                }
            }
        }

        kernel_matrix
    }
}

/// Nyström method for kernel approximation
///
/// General method for kernel approximation using eigendecomposition on a subset
/// of training data. Works with any kernel function and supports multiple
/// sampling strategies for improved approximation quality.
///
/// # Parameters
///
/// * `kernel` - Kernel function to approximate
/// * `n_components` - Number of samples to use for approximation (default: 100)
/// * `sampling_strategy` - Strategy for selecting landmark points
/// * `random_state` - Random seed for reproducibility
///
/// # Examples
///
/// ```rust,ignore
/// use sklears_kernel_approximation::nystroem::{Nystroem, Kernel, SamplingStrategy};
/// use sklears_core::traits::{Transform, Fit, Untrained}
/// use scirs2_core::ndarray::array;
///
/// let X = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
///
/// let nystroem = Nystroem::new(Kernel::Rbf { gamma: 1.0 }, 3)
///     .sampling_strategy(SamplingStrategy::LeverageScore);
/// let fitted_nystroem = nystroem.fit(&X, &()).unwrap();
/// let X_transformed = fitted_nystroem.transform(&X).unwrap();
/// assert_eq!(X_transformed.shape(), &[3, 3]);
/// ```
#[derive(Debug, Clone)]
/// Nystroem
pub struct Nystroem<State = Untrained> {
    /// Kernel function
    pub kernel: Kernel,
    /// Number of components for approximation
    pub n_components: usize,
    /// Sampling strategy for landmark selection
    pub sampling_strategy: SamplingStrategy,
    /// Random seed
    pub random_state: Option<u64>,

    // Fitted attributes
    components_: Option<Array2<Float>>,
    normalization_: Option<Array2<Float>>,
    component_indices_: Option<Vec<usize>>,

    _state: PhantomData<State>,
}

impl Nystroem<Untrained> {
    /// Create a new Nystroem approximator
    pub fn new(kernel: Kernel, n_components: usize) -> Self {
        Self {
            kernel,
            n_components,
            sampling_strategy: SamplingStrategy::Random,
            random_state: None,
            components_: None,
            normalization_: None,
            component_indices_: None,
            _state: PhantomData,
        }
    }

    /// Set the sampling strategy
    pub fn sampling_strategy(mut self, strategy: SamplingStrategy) -> Self {
        self.sampling_strategy = strategy;
        self
    }

    /// Set random state for reproducibility
    pub fn random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }
}

impl Estimator for Nystroem<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Nystroem<Untrained> {
    /// Select component indices based on sampling strategy
    fn select_components(
        &self,
        x: &Array2<Float>,
        n_components: usize,
        rng: &mut RealStdRng,
    ) -> Result<Vec<usize>> {
        let (n_samples, _) = x.dim();

        match &self.sampling_strategy {
            SamplingStrategy::Random => {
                let mut indices: Vec<usize> = (0..n_samples).collect();
                indices.shuffle(rng);
                Ok(indices[..n_components].to_vec())
            }
            SamplingStrategy::KMeans => {
                // Simple k-means based sampling
                self.kmeans_sampling(x, n_components, rng)
            }
            SamplingStrategy::LeverageScore => {
                // Leverage score based sampling
                self.leverage_score_sampling(x, n_components, rng)
            }
            SamplingStrategy::ColumnNorm => {
                // Column norm based sampling
                self.column_norm_sampling(x, n_components, rng)
            }
        }
    }

    /// Simple k-means based sampling
    fn kmeans_sampling(
        &self,
        x: &Array2<Float>,
        n_components: usize,
        rng: &mut RealStdRng,
    ) -> Result<Vec<usize>> {
        let (n_samples, n_features) = x.dim();
        let mut centers = Array2::zeros((n_components, n_features));

        // Initialize centers randomly
        let mut indices: Vec<usize> = (0..n_samples).collect();
        indices.shuffle(rng);
        for (i, &idx) in indices[..n_components].iter().enumerate() {
            centers.row_mut(i).assign(&x.row(idx));
        }

        // Run a few iterations of k-means
        for _iter in 0..5 {
            let mut assignments = vec![0; n_samples];

            // Assign points to nearest centers
            for i in 0..n_samples {
                let mut min_dist = Float::INFINITY;
                let mut best_center = 0;

                for j in 0..n_components {
                    let diff = &x.row(i) - &centers.row(j);
                    let dist = diff.dot(&diff);
                    if dist < min_dist {
                        min_dist = dist;
                        best_center = j;
                    }
                }
                assignments[i] = best_center;
            }

            // Update centers
            for j in 0..n_components {
                let cluster_points: Vec<usize> = assignments
                    .iter()
                    .enumerate()
                    .filter(|(_, &assignment)| assignment == j)
                    .map(|(i, _)| i)
                    .collect();

                if !cluster_points.is_empty() {
                    let mut new_center = Array1::zeros(n_features);
                    for &point_idx in &cluster_points {
                        new_center = new_center + x.row(point_idx);
                    }
                    new_center /= cluster_points.len() as Float;
                    centers.row_mut(j).assign(&new_center);
                }
            }
        }

        // Find closest points to final centers
        let mut selected_indices = Vec::new();
        for j in 0..n_components {
            let mut min_dist = Float::INFINITY;
            let mut best_point = 0;

            for i in 0..n_samples {
                let diff = &x.row(i) - &centers.row(j);
                let dist = diff.dot(&diff);
                if dist < min_dist {
                    min_dist = dist;
                    best_point = i;
                }
            }
            selected_indices.push(best_point);
        }

        selected_indices.sort_unstable();
        selected_indices.dedup();

        // Fill remaining slots randomly if needed
        while selected_indices.len() < n_components {
            let random_idx = rng.gen_range(0..n_samples);
            if !selected_indices.contains(&random_idx) {
                selected_indices.push(random_idx);
            }
        }

        Ok(selected_indices[..n_components].to_vec())
    }

    /// Leverage score based sampling
    fn leverage_score_sampling(
        &self,
        x: &Array2<Float>,
        n_components: usize,
        _rng: &mut RealStdRng,
    ) -> Result<Vec<usize>> {
        let (n_samples, _) = x.dim();

        // Compute leverage scores (diagonal of hat matrix)
        // For simplicity, we approximate using row norms as proxy
        let mut scores = Vec::new();
        for i in 0..n_samples {
            let row_norm = x.row(i).dot(&x.row(i)).sqrt();
            scores.push(row_norm + 1e-10); // Add small epsilon for numerical stability
        }

        // Sample based on scores using cumulative distribution
        let total_score: Float = scores.iter().sum();
        if total_score <= 0.0 {
            return Err(SklearsError::InvalidInput(
                "All scores are zero or negative".to_string(),
            ));
        }

        // Create cumulative distribution
        let mut cumulative = Vec::with_capacity(scores.len());
        let mut sum = 0.0;
        for &score in &scores {
            sum += score / total_score;
            cumulative.push(sum);
        }

        let mut selected_indices = Vec::new();
        for _ in 0..n_components {
            let r = thread_rng().gen::<Float>();
            // Find index where cumulative probability >= r
            let mut idx = cumulative
                .iter()
                .position(|&cum| cum >= r)
                .unwrap_or(scores.len() - 1);

            // Ensure no duplicates
            while selected_indices.contains(&idx) {
                let r = thread_rng().gen::<Float>();
                idx = cumulative
                    .iter()
                    .position(|&cum| cum >= r)
                    .unwrap_or(scores.len() - 1);
            }
            selected_indices.push(idx);
        }

        Ok(selected_indices)
    }

    /// Column norm based sampling
    fn column_norm_sampling(
        &self,
        x: &Array2<Float>,
        n_components: usize,
        rng: &mut RealStdRng,
    ) -> Result<Vec<usize>> {
        let (n_samples, _) = x.dim();

        // Compute row norms
        let mut norms = Vec::new();
        for i in 0..n_samples {
            let norm = x.row(i).dot(&x.row(i)).sqrt();
            norms.push(norm + 1e-10);
        }

        // Sort by norm and take diverse selection
        let mut indices_with_norms: Vec<(usize, Float)> = norms
            .iter()
            .enumerate()
            .map(|(i, &norm)| (i, norm))
            .collect();
        indices_with_norms.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let mut selected_indices = Vec::new();
        let step = n_samples.max(1) / n_components.max(1);

        for i in 0..n_components {
            let idx = (i * step).min(n_samples - 1);
            selected_indices.push(indices_with_norms[idx].0);
        }

        // Fill remaining with random if needed
        while selected_indices.len() < n_components {
            let random_idx = rng.gen_range(0..n_samples);
            if !selected_indices.contains(&random_idx) {
                selected_indices.push(random_idx);
            }
        }

        Ok(selected_indices)
    }

    /// Compute eigendecomposition using power iteration method
    /// Returns (eigenvalues, eigenvectors) for symmetric matrix
    fn compute_eigendecomposition(
        &self,
        matrix: &Array2<Float>,
        rng: &mut RealStdRng,
    ) -> Result<(Array1<Float>, Array2<Float>)> {
        let n = matrix.nrows();

        if n != matrix.ncols() {
            return Err(SklearsError::InvalidInput(
                "Matrix must be square for eigendecomposition".to_string(),
            ));
        }

        let mut eigenvals = Array1::zeros(n);
        let mut eigenvecs = Array2::zeros((n, n));

        // Use deflation method to find multiple eigenvalues
        let mut deflated_matrix = matrix.clone();

        for k in 0..n {
            // Power iteration for k-th eigenvalue/eigenvector
            let (eigenval, eigenvec) = self.power_iteration(&deflated_matrix, 100, 1e-8, rng)?;

            eigenvals[k] = eigenval;
            eigenvecs.column_mut(k).assign(&eigenvec);

            // Deflate matrix: A_new = A - λ * v * v^T
            for i in 0..n {
                for j in 0..n {
                    deflated_matrix[[i, j]] -= eigenval * eigenvec[i] * eigenvec[j];
                }
            }
        }

        // Sort eigenvalues and eigenvectors in descending order
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&i, &j| eigenvals[j].partial_cmp(&eigenvals[i]).unwrap());

        let mut sorted_eigenvals = Array1::zeros(n);
        let mut sorted_eigenvecs = Array2::zeros((n, n));

        for (new_idx, &old_idx) in indices.iter().enumerate() {
            sorted_eigenvals[new_idx] = eigenvals[old_idx];
            sorted_eigenvecs
                .column_mut(new_idx)
                .assign(&eigenvecs.column(old_idx));
        }

        Ok((sorted_eigenvals, sorted_eigenvecs))
    }

    /// Power iteration method to find dominant eigenvalue and eigenvector
    fn power_iteration(
        &self,
        matrix: &Array2<Float>,
        max_iter: usize,
        tol: Float,
        rng: &mut RealStdRng,
    ) -> Result<(Float, Array1<Float>)> {
        let n = matrix.nrows();

        // Initialize random vector
        let mut v = Array1::from_shape_fn(n, |_| rng.gen::<Float>() - 0.5);

        // Normalize
        let norm = v.dot(&v).sqrt();
        if norm < 1e-10 {
            return Err(SklearsError::InvalidInput(
                "Initial vector has zero norm".to_string(),
            ));
        }
        v /= norm;

        let mut eigenval = 0.0;

        for _iter in 0..max_iter {
            // Apply matrix
            let w = matrix.dot(&v);

            // Compute Rayleigh quotient
            let new_eigenval = v.dot(&w);

            // Normalize
            let w_norm = w.dot(&w).sqrt();
            if w_norm < 1e-10 {
                break;
            }
            let new_v = w / w_norm;

            // Check convergence
            let eigenval_change = (new_eigenval - eigenval).abs();
            let vector_change = (&new_v - &v).mapv(|x| x.abs()).sum();

            if eigenval_change < tol && vector_change < tol {
                return Ok((new_eigenval, new_v));
            }

            eigenval = new_eigenval;
            v = new_v;
        }

        Ok((eigenval, v))
    }
}

impl Fit<Array2<Float>, ()> for Nystroem<Untrained> {
    type Fitted = Nystroem<Trained>;

    fn fit(self, x: &Array2<Float>, _y: &()) -> Result<Self::Fitted> {
        let (n_samples, _) = x.dim();

        if self.n_components > n_samples {
            eprintln!(
                "Warning: n_components ({}) > n_samples ({})",
                self.n_components, n_samples
            );
        }

        let n_components_actual = self.n_components.min(n_samples);

        let mut rng = if let Some(seed) = self.random_state {
            RealStdRng::seed_from_u64(seed)
        } else {
            RealStdRng::from_seed(thread_rng().gen())
        };

        // Select component indices using specified strategy
        let component_indices = self.select_components(x, n_components_actual, &mut rng)?;

        // Extract component samples
        let mut components = Array2::zeros((n_components_actual, x.ncols()));
        for (i, &idx) in component_indices.iter().enumerate() {
            components.row_mut(i).assign(&x.row(idx));
        }

        // Compute kernel matrix K_11 on sampled points
        let k11: Array2<f64> = self.kernel.compute_kernel(&components, &components);

        // Proper Nyström approximation using eigendecomposition
        // K ≈ K₁₂ K₁₁⁻¹ K₁₂ᵀ where K₁₁⁻¹ is the pseudo-inverse of landmark kernel matrix
        let eps = 1e-12;

        // Add small regularization to diagonal for numerical stability
        let mut k11_reg = k11.clone();
        for i in 0..n_components_actual {
            k11_reg[[i, i]] += eps;
        }

        // Compute pseudo-inverse using eigendecomposition
        // For symmetric positive definite matrices, we can use power iteration for eigendecomposition
        let (eigenvals, eigenvecs) = self.compute_eigendecomposition(&k11_reg, &mut rng)?;

        // Filter out small eigenvalues for numerical stability
        let threshold = 1e-8;
        let valid_indices: Vec<usize> = eigenvals
            .iter()
            .enumerate()
            .filter(|(_, &val)| val > threshold)
            .map(|(i, _)| i)
            .collect();

        if valid_indices.is_empty() {
            return Err(SklearsError::InvalidInput(
                "No valid eigenvalues found in kernel matrix".to_string(),
            ));
        }

        // Construct pseudo-inverse: V * Λ⁻¹ * V^T
        let _n_valid = valid_indices.len();
        let mut pseudo_inverse = Array2::zeros((n_components_actual, n_components_actual));

        for i in 0..n_components_actual {
            for j in 0..n_components_actual {
                let mut sum = 0.0;
                for &k in &valid_indices {
                    sum += eigenvecs[[i, k]] * eigenvecs[[j, k]] / eigenvals[k];
                }
                pseudo_inverse[[i, j]] = sum;
            }
        }

        let normalization = pseudo_inverse;

        Ok(Nystroem {
            kernel: self.kernel,
            n_components: self.n_components,
            sampling_strategy: self.sampling_strategy,
            random_state: self.random_state,
            components_: Some(components),
            normalization_: Some(normalization),
            component_indices_: Some(component_indices),
            _state: PhantomData,
        })
    }
}

impl Transform<Array2<Float>, Array2<Float>> for Nystroem<Trained> {
    fn transform(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let components = self.components_.as_ref().unwrap();
        let normalization = self.normalization_.as_ref().unwrap();

        if x.ncols() != components.ncols() {
            return Err(SklearsError::InvalidInput(format!(
                "X has {} features, but Nystroem was fitted with {} features",
                x.ncols(),
                components.ncols()
            )));
        }

        // Compute kernel matrix K(X, components)
        let k_x_components = self.kernel.compute_kernel(x, components);

        // Apply normalization: K(X, components) @ normalization
        let result = k_x_components.dot(normalization);

        Ok(result)
    }
}

impl Nystroem<Trained> {
    /// Get the selected component samples
    pub fn components(&self) -> &Array2<Float> {
        self.components_.as_ref().unwrap()
    }

    /// Get the component indices
    pub fn component_indices(&self) -> &[usize] {
        self.component_indices_.as_ref().unwrap()
    }

    /// Get the normalization matrix
    pub fn normalization(&self) -> &Array2<Float> {
        self.normalization_.as_ref().unwrap()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_nystroem_linear_kernel() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0],];

        let nystroem = Nystroem::new(Kernel::Linear, 3);
        let fitted = nystroem.fit(&x, &()).unwrap();
        let x_transformed = fitted.transform(&x).unwrap();

        assert_eq!(x_transformed.nrows(), 4);
        assert!(x_transformed.ncols() <= 3); // May be less due to eigenvalue filtering
    }

    #[test]
    fn test_nystroem_rbf_kernel() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0],];

        let nystroem = Nystroem::new(Kernel::Rbf { gamma: 0.1 }, 2);
        let fitted = nystroem.fit(&x, &()).unwrap();
        let x_transformed = fitted.transform(&x).unwrap();

        assert_eq!(x_transformed.nrows(), 3);
        assert!(x_transformed.ncols() <= 2);
    }

    #[test]
    fn test_nystroem_polynomial_kernel() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0],];

        let kernel = Kernel::Polynomial {
            gamma: 1.0,
            coef0: 1.0,
            degree: 2,
        };
        let nystroem = Nystroem::new(kernel, 2);
        let fitted = nystroem.fit(&x, &()).unwrap();
        let x_transformed = fitted.transform(&x).unwrap();

        assert_eq!(x_transformed.nrows(), 3);
        assert!(x_transformed.ncols() <= 2);
    }

    #[test]
    fn test_nystroem_reproducibility() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0],];

        let nystroem1 = Nystroem::new(Kernel::Linear, 3).random_state(42);
        let fitted1 = nystroem1.fit(&x, &()).unwrap();
        let result1 = fitted1.transform(&x).unwrap();

        let nystroem2 = Nystroem::new(Kernel::Linear, 3).random_state(42);
        let fitted2 = nystroem2.fit(&x, &()).unwrap();
        let result2 = fitted2.transform(&x).unwrap();

        // Results should be very similar with same random state (allowing for numerical precision)
        assert_eq!(result1.shape(), result2.shape());
        for (a, b) in result1.iter().zip(result2.iter()) {
            assert!(
                (a - b).abs() < 1e-6,
                "Values differ too much: {} vs {}",
                a,
                b
            );
        }
    }

    #[test]
    fn test_nystroem_feature_mismatch() {
        let x_train = array![[1.0, 2.0], [3.0, 4.0],];

        let x_test = array![
            [1.0, 2.0, 3.0], // Wrong number of features
        ];

        let nystroem = Nystroem::new(Kernel::Linear, 2);
        let fitted = nystroem.fit(&x_train, &()).unwrap();
        let result = fitted.transform(&x_test);

        assert!(result.is_err());
    }

    #[test]
    fn test_nystroem_sampling_strategies() {
        let x = array![
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
            [2.0, 1.0],
            [4.0, 3.0],
            [6.0, 5.0],
            [8.0, 7.0]
        ];

        // Test Random sampling
        let nystroem_random = Nystroem::new(Kernel::Linear, 4)
            .sampling_strategy(SamplingStrategy::Random)
            .random_state(42);
        let fitted_random = nystroem_random.fit(&x, &()).unwrap();
        let result_random = fitted_random.transform(&x).unwrap();
        assert_eq!(result_random.nrows(), 8);

        // Test K-means sampling
        let nystroem_kmeans = Nystroem::new(Kernel::Linear, 4)
            .sampling_strategy(SamplingStrategy::KMeans)
            .random_state(42);
        let fitted_kmeans = nystroem_kmeans.fit(&x, &()).unwrap();
        let result_kmeans = fitted_kmeans.transform(&x).unwrap();
        assert_eq!(result_kmeans.nrows(), 8);

        // Test Leverage score sampling
        let nystroem_leverage = Nystroem::new(Kernel::Linear, 4)
            .sampling_strategy(SamplingStrategy::LeverageScore)
            .random_state(42);
        let fitted_leverage = nystroem_leverage.fit(&x, &()).unwrap();
        let result_leverage = fitted_leverage.transform(&x).unwrap();
        assert_eq!(result_leverage.nrows(), 8);

        // Test Column norm sampling
        let nystroem_norm = Nystroem::new(Kernel::Linear, 4)
            .sampling_strategy(SamplingStrategy::ColumnNorm)
            .random_state(42);
        let fitted_norm = nystroem_norm.fit(&x, &()).unwrap();
        let result_norm = fitted_norm.transform(&x).unwrap();
        assert_eq!(result_norm.nrows(), 8);
    }

    #[test]
    fn test_nystroem_rbf_with_different_sampling() {
        let x = array![
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
            [2.0, 1.0],
            [4.0, 3.0],
            [6.0, 5.0],
            [8.0, 7.0]
        ];

        let kernel = Kernel::Rbf { gamma: 0.1 };

        // Test with leverage score sampling
        let nystroem = Nystroem::new(kernel, 4)
            .sampling_strategy(SamplingStrategy::LeverageScore)
            .random_state(42);
        let fitted = nystroem.fit(&x, &()).unwrap();
        let result = fitted.transform(&x).unwrap();

        assert_eq!(result.shape(), &[8, 4]);

        // Check that all values are finite
        for val in result.iter() {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_nystroem_improved_eigendecomposition() {
        let x = array![[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]];

        let nystroem = Nystroem::new(Kernel::Linear, 3)
            .sampling_strategy(SamplingStrategy::Random)
            .random_state(42);
        let fitted = nystroem.fit(&x, &()).unwrap();
        let result = fitted.transform(&x).unwrap();

        assert_eq!(result.nrows(), 4);
        assert!(result.ncols() <= 3);

        // Check numerical stability
        for val in result.iter() {
            assert!(val.is_finite());
        }
    }
}
