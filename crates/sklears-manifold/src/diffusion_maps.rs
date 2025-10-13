//! Diffusion Maps implementation
//!
//! This module provides Diffusion Maps for non-linear dimensionality reduction through diffusion processes.

use scirs2_core::ndarray::ndarray_linalg::{Eigh, UPLO};
use scirs2_core::ndarray::{Array2, ArrayView2};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Transform, Untrained},
    types::Float,
};

/// Diffusion Maps
///
/// Diffusion Maps is a nonlinear dimensionality reduction technique that
/// captures the intrinsic geometry of data through the eigenfunctions of
/// a Markov chain. It constructs a diffusion process on the data and uses
/// the eigenvectors of the diffusion operator to embed the data.
///
/// # Parameters
///
/// * `n_components` - Number of coordinates for the manifold
/// * `diffusion_time` - Diffusion time parameter (t)
/// * `epsilon` - Bandwidth parameter for Gaussian kernel
/// * `alpha` - Normalization parameter (0 for symmetric, 1 for row-stochastic)
/// * `eigen_solver` - The eigensolver to use
/// * `random_state` - Random state for reproducibility
/// * `n_jobs` - Number of parallel jobs
///
/// # Examples
///
/// ```
/// use sklears_manifold::DiffusionMaps;
/// use sklears_core::traits::{Transform, Fit};
/// use scirs2_core::ndarray::array;
///
/// let x = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]];
///
/// let dm = DiffusionMaps::new()
///     .n_components(2)
///     .epsilon(1.0);
/// let fitted = dm.fit(&x.view(), &()).unwrap();
/// let embedded = fitted.transform(&x.view()).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct DiffusionMaps<S = Untrained> {
    state: S,
    n_components: usize,
    diffusion_time: usize,
    epsilon: Option<f64>,
    alpha: f64,
    eigen_solver: String,
    random_state: Option<u64>,
    n_jobs: Option<i32>,
}

/// Trained state for Diffusion Maps
#[derive(Debug, Clone)]
pub struct DiffusionMapsTrained {
    /// The low-dimensional embedding of the training data
    pub embedding: Array2<f64>,
    /// Eigenvalues of the diffusion operator
    pub eigenvalues: Array2<f64>,
    /// Eigenvectors of the diffusion operator
    pub eigenvectors: Array2<f64>,
    /// Affinity matrix used for diffusion
    pub affinity_matrix: Array2<f64>,
    /// Bandwidth parameter used
    pub epsilon: f64,
}

impl DiffusionMaps<Untrained> {
    /// Create a new DiffusionMaps instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            n_components: 2,
            diffusion_time: 1,
            epsilon: None,
            alpha: 0.5,
            eigen_solver: "auto".to_string(),
            random_state: None,
            n_jobs: None,
        }
    }

    /// Set the number of components
    pub fn n_components(mut self, n_components: usize) -> Self {
        self.n_components = n_components;
        self
    }

    /// Set the diffusion time
    pub fn diffusion_time(mut self, diffusion_time: usize) -> Self {
        self.diffusion_time = diffusion_time;
        self
    }

    /// Set the epsilon parameter
    pub fn epsilon(mut self, epsilon: f64) -> Self {
        self.epsilon = Some(epsilon);
        self
    }

    /// Set the alpha parameter
    pub fn alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    /// Set the eigen solver
    pub fn eigen_solver(mut self, eigen_solver: &str) -> Self {
        self.eigen_solver = eigen_solver.to_string();
        self
    }

    /// Set the random state
    pub fn random_state(mut self, random_state: Option<u64>) -> Self {
        self.random_state = random_state;
        self
    }

    /// Set the number of jobs
    pub fn n_jobs(mut self, n_jobs: Option<i32>) -> Self {
        self.n_jobs = n_jobs;
        self
    }
}

impl Default for DiffusionMaps<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for DiffusionMaps<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, ()> for DiffusionMaps<Untrained> {
    type Fitted = DiffusionMaps<DiffusionMapsTrained>;

    fn fit(self, x: &ArrayView2<'_, Float>, _y: &()) -> SklResult<Self::Fitted> {
        let x = x.mapv(|x| x);
        let (n_samples, _) = x.dim();

        if n_samples <= self.n_components {
            return Err(SklearsError::InvalidInput(
                "Number of samples must be greater than n_components".to_string(),
            ));
        }

        // Step 1: Compute affinity matrix
        let epsilon = self.epsilon.unwrap_or_else(|| self.estimate_epsilon(&x));
        let affinity = self.compute_affinity_matrix(&x, epsilon)?;

        // Step 2: Normalize the affinity matrix
        let transition_matrix = self.normalize_affinity_matrix(&affinity)?;

        // Step 3: Eigendecomposition
        let (eigenvalues, eigenvectors) = self.compute_eigenvectors(&transition_matrix)?;

        // Step 4: Create embedding
        let embedding = self.create_embedding(&eigenvalues, &eigenvectors)?;

        Ok(DiffusionMaps {
            state: DiffusionMapsTrained {
                embedding,
                eigenvalues,
                eigenvectors,
                affinity_matrix: affinity,
                epsilon,
            },
            n_components: self.n_components,
            diffusion_time: self.diffusion_time,
            epsilon: Some(epsilon),
            alpha: self.alpha,
            eigen_solver: self.eigen_solver,
            random_state: self.random_state,
            n_jobs: self.n_jobs,
        })
    }
}

impl DiffusionMaps<Untrained> {
    fn estimate_epsilon(&self, x: &Array2<f64>) -> f64 {
        let n_samples = x.nrows();
        let mut distances = Vec::new();

        // Compute pairwise distances
        for i in 0..n_samples {
            for j in i + 1..n_samples {
                let diff = &x.row(i) - &x.row(j);
                let dist = diff.mapv(|x| x * x).sum().sqrt();
                distances.push(dist);
            }
        }

        // Use median distance as epsilon estimate
        distances.sort_by(|a, b| a.partial_cmp(b).unwrap());
        distances[distances.len() / 2]
    }

    fn compute_affinity_matrix(&self, x: &Array2<f64>, epsilon: f64) -> SklResult<Array2<f64>> {
        let n_samples = x.nrows();
        let mut affinity = Array2::zeros((n_samples, n_samples));

        // Gaussian kernel: K(x,y) = exp(-||x-y||²/(2ε²))
        for i in 0..n_samples {
            for j in 0..n_samples {
                if i != j {
                    let diff = &x.row(i) - &x.row(j);
                    let dist_sq = diff.mapv(|x| x * x).sum();
                    let weight = (-dist_sq / (2.0 * epsilon * epsilon)).exp();
                    affinity[[i, j]] = weight;
                } else {
                    affinity[[i, j]] = 1.0;
                }
            }
        }

        Ok(affinity)
    }

    fn normalize_affinity_matrix(&self, affinity: &Array2<f64>) -> SklResult<Array2<f64>> {
        let n_samples = affinity.nrows();
        let mut normalized = affinity.clone();

        // Compute degree matrix
        let mut degrees = Array2::zeros((n_samples, n_samples));
        for i in 0..n_samples {
            let degree_sum: f64 = affinity.row(i).sum();
            degrees[[i, i]] = degree_sum;
        }

        // Normalize based on alpha parameter
        // P = D^(-α) * K * D^(-α)
        for i in 0..n_samples {
            for j in 0..n_samples {
                if degrees[[i, i]] > 1e-10 && degrees[[j, j]] > 1e-10 {
                    let degree_factor_i = degrees[[i, i]].powf(-self.alpha);
                    let degree_factor_j = degrees[[j, j]].powf(-self.alpha);
                    normalized[[i, j]] = degree_factor_i * affinity[[i, j]] * degree_factor_j;
                }
            }
        }

        // Row normalization for transition matrix
        for i in 0..n_samples {
            let row_sum: f64 = normalized.row(i).sum();
            if row_sum > 1e-10 {
                for j in 0..n_samples {
                    normalized[[i, j]] /= row_sum;
                }
            }
        }

        Ok(normalized)
    }

    fn compute_eigenvectors(
        &self,
        transition_matrix: &Array2<f64>,
    ) -> SklResult<(Array2<f64>, Array2<f64>)> {
        // Compute eigendecomposition
        let (eigenvals, eigenvecs) = transition_matrix
            .eigh(UPLO::Lower)
            .map_err(|e| SklearsError::InvalidInput(format!("Eigendecomposition failed: {e}")))?;

        // Sort eigenvalues and eigenvectors in descending order
        let mut eigen_pairs: Vec<(f64, usize)> = eigenvals
            .iter()
            .enumerate()
            .map(|(i, &val)| (val, i))
            .collect();
        eigen_pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

        // Create sorted eigenvalue and eigenvector matrices
        let n = eigenvals.len();
        let mut sorted_eigenvals = Array2::zeros((n, 1));
        let mut sorted_eigenvecs = Array2::zeros((n, n));

        for (new_idx, &(eigenval, old_idx)) in eigen_pairs.iter().enumerate() {
            sorted_eigenvals[[new_idx, 0]] = eigenval;
            for i in 0..n {
                sorted_eigenvecs[[i, new_idx]] = eigenvecs[[i, old_idx]];
            }
        }

        Ok((sorted_eigenvals, sorted_eigenvecs))
    }

    fn create_embedding(
        &self,
        eigenvalues: &Array2<f64>,
        eigenvectors: &Array2<f64>,
    ) -> SklResult<Array2<f64>> {
        let n_samples = eigenvectors.nrows();
        let mut embedding = Array2::zeros((n_samples, self.n_components));

        // Skip the first eigenvector (constant vector) and take the next n_components
        for comp_idx in 0..self.n_components {
            let eigen_idx = comp_idx + 1; // Skip first eigenvector
            if eigen_idx < eigenvalues.nrows() {
                let eigenval = eigenvalues[[eigen_idx, 0]];

                // Apply diffusion time: λ^t
                let diffusion_weight = eigenval.powf(self.diffusion_time as f64);

                for i in 0..n_samples {
                    embedding[[i, comp_idx]] = eigenvectors[[i, eigen_idx]] * diffusion_weight;
                }
            }
        }

        Ok(embedding)
    }
}

impl Transform<ArrayView2<'_, Float>, Array2<f64>> for DiffusionMaps<DiffusionMapsTrained> {
    fn transform(&self, _x: &ArrayView2<'_, Float>) -> SklResult<Array2<f64>> {
        // Diffusion Maps doesn't support transforming new data in this implementation
        Ok(self.state.embedding.clone())
    }
}

impl DiffusionMaps<DiffusionMapsTrained> {
    /// Get the embedding
    pub fn embedding(&self) -> &Array2<f64> {
        &self.state.embedding
    }

    /// Get the eigenvalues
    pub fn eigenvalues(&self) -> &Array2<f64> {
        &self.state.eigenvalues
    }

    /// Get the eigenvectors
    pub fn eigenvectors(&self) -> &Array2<f64> {
        &self.state.eigenvectors
    }

    /// Get the affinity matrix
    pub fn affinity_matrix(&self) -> &Array2<f64> {
        &self.state.affinity_matrix
    }

    /// Get the epsilon parameter used
    pub fn epsilon(&self) -> f64 {
        self.state.epsilon
    }
}
