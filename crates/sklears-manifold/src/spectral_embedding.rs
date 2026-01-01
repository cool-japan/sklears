//! Spectral Embedding implementation
//!
//! This module provides Spectral Embedding for graph-based manifold learning through eigendecomposition of Laplacian matrices.

use scirs2_core::ndarray::{s, Array1, Array2, ArrayView2, Axis};
use scirs2_linalg::compat::{ArrayLinalgExt, UPLO};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Transform, Untrained},
};

/// Spectral Embedding for Graph-Based Manifold Learning
///
/// Spectral embedding uses eigendecomposition of graph Laplacian matrices to embed
/// data points in lower-dimensional space while preserving local neighborhood structure.
///
/// The algorithm constructs a graph from the data using k-nearest neighbors or epsilon
/// neighborhoods, computes various forms of the graph Laplacian, and uses the
/// eigenvectors corresponding to the smallest eigenvalues as the embedding coordinates.
///
/// # Parameters
///
/// * `n_components` - Number of dimensions for the embedded space
/// * `n_neighbors` - Number of neighbors to use when constructing the affinity matrix
/// * `affinity` - How to construct the affinity matrix ('nearest_neighbors', 'rbf', 'polynomial')
/// * `gamma` - Kernel coefficient for 'rbf' and 'polynomial' affinities
/// * `degree` - Degree of polynomial kernel (for 'polynomial' affinity)
/// * `coef0` - Zero-order term in polynomial kernel
/// * `eigen_solver` - The eigenvalue decomposition strategy ('arpack', 'lobpcg', 'amg')
/// * `random_state` - Random state for reproducibility
/// * `laplacian_type` - Type of Laplacian ('unnormalized', 'symmetric', 'random_walk')
#[derive(Debug, Clone)]
pub struct SpectralEmbedding<S = Untrained> {
    state: S,
    n_components: usize,
    n_neighbors: usize,
    affinity: String,
    gamma: Option<f64>,
    degree: usize,
    coef0: f64,
    eigen_solver: String,
    random_state: Option<u64>,
    laplacian_type: String,
}

impl Default for SpectralEmbedding<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl SpectralEmbedding<Untrained> {
    /// Create a new SpectralEmbedding instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            n_components: 2,
            n_neighbors: 10,
            affinity: "nearest_neighbors".to_string(),
            gamma: None,
            degree: 3,
            coef0: 1.0,
            eigen_solver: "arpack".to_string(),
            random_state: None,
            laplacian_type: "symmetric".to_string(),
        }
    }

    /// Set the number of components
    pub fn n_components(mut self, n_components: usize) -> Self {
        self.n_components = n_components;
        self
    }

    /// Set the number of neighbors
    pub fn n_neighbors(mut self, n_neighbors: usize) -> Self {
        self.n_neighbors = n_neighbors;
        self
    }

    /// Set the affinity method
    pub fn affinity(mut self, affinity: &str) -> Self {
        self.affinity = affinity.to_string();
        self
    }

    /// Set the gamma parameter for RBF kernel
    pub fn gamma(mut self, gamma: f64) -> Self {
        self.gamma = Some(gamma);
        self
    }

    /// Set the random state
    pub fn random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }

    /// Set the Laplacian type
    pub fn laplacian_type(mut self, laplacian_type: &str) -> Self {
        self.laplacian_type = laplacian_type.to_string();
        self
    }
}

#[derive(Debug, Clone)]
pub struct SpectralEmbeddingTrained {
    embedding: Array2<f64>,
    eigenvalues: Array1<f64>,
    affinity_matrix: Array2<f64>,
    n_components: usize,
}

impl Estimator for SpectralEmbedding<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Estimator for SpectralEmbedding<SpectralEmbeddingTrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, f64>, ()> for SpectralEmbedding<Untrained> {
    type Fitted = SpectralEmbedding<SpectralEmbeddingTrained>;

    fn fit(self, x: &ArrayView2<'_, f64>, _y: &()) -> SklResult<Self::Fitted> {
        let (n_samples, _) = x.dim();

        if self.n_components >= n_samples {
            return Err(SklearsError::InvalidInput(
                "n_components must be less than n_samples".to_string(),
            ));
        }

        // Step 1: Construct affinity matrix
        let affinity_matrix = self.construct_affinity_matrix(x)?;

        // Step 2: Construct Laplacian matrix
        let laplacian = self.construct_laplacian(&affinity_matrix)?;

        // Step 3: Eigendecomposition
        let (eigenvalues, eigenvectors) = laplacian.eigh(UPLO::Lower).map_err(|e| {
            SklearsError::NumericalError(format!("Eigendecomposition failed: {:?}", e))
        })?;

        // Step 4: Select embedding dimensions
        let embedding = eigenvectors.slice(s![.., 1..=self.n_components]).to_owned();
        let selected_eigenvalues = eigenvalues.slice(s![1..=self.n_components]).to_owned();

        Ok(SpectralEmbedding {
            state: SpectralEmbeddingTrained {
                embedding,
                eigenvalues: selected_eigenvalues,
                affinity_matrix,
                n_components: self.n_components,
            },
            n_components: self.n_components,
            n_neighbors: self.n_neighbors,
            affinity: self.affinity,
            gamma: self.gamma,
            degree: self.degree,
            coef0: self.coef0,
            eigen_solver: self.eigen_solver,
            random_state: self.random_state,
            laplacian_type: self.laplacian_type,
        })
    }
}

impl Transform<ArrayView2<'_, f64>, Array2<f64>> for SpectralEmbedding<SpectralEmbeddingTrained> {
    fn transform(&self, _x: &ArrayView2<'_, f64>) -> SklResult<Array2<f64>> {
        // For spectral embedding, the embedding is computed during fitting
        // and cannot be extended to new points directly
        Ok(self.state.embedding.clone())
    }
}

impl SpectralEmbedding<Untrained> {
    fn construct_affinity_matrix(&self, x: &ArrayView2<f64>) -> SklResult<Array2<f64>> {
        let (n_samples, n_features) = x.dim();
        let mut affinity = Array2::zeros((n_samples, n_samples));

        match self.affinity.as_str() {
            "nearest_neighbors" => {
                // k-NN affinity
                for i in 0..n_samples {
                    let mut distances: Vec<(usize, f64)> = Vec::new();

                    for j in 0..n_samples {
                        if i != j {
                            let dist = x
                                .row(i)
                                .iter()
                                .zip(x.row(j).iter())
                                .map(|(a, b)| (a - b).powi(2))
                                .sum::<f64>()
                                .sqrt();
                            distances.push((j, dist));
                        }
                    }

                    distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

                    // Connect to k nearest neighbors
                    for &(j, _) in distances.iter().take(self.n_neighbors) {
                        affinity[(i, j)] = 1.0;
                        affinity[(j, i)] = 1.0; // Symmetric
                    }
                }
            }
            "rbf" => {
                // RBF (Gaussian) kernel
                let gamma = self.gamma.unwrap_or(1.0 / n_features as f64);

                for i in 0..n_samples {
                    for j in i + 1..n_samples {
                        let dist_sq = x
                            .row(i)
                            .iter()
                            .zip(x.row(j).iter())
                            .map(|(a, b)| (a - b).powi(2))
                            .sum::<f64>();

                        let weight = (-gamma * dist_sq).exp();
                        affinity[(i, j)] = weight;
                        affinity[(j, i)] = weight;
                    }
                }
            }
            "polynomial" => {
                // Polynomial kernel
                let gamma = self.gamma.unwrap_or(1.0 / n_features as f64);

                for i in 0..n_samples {
                    for j in i + 1..n_samples {
                        let dot_product: f64 = x
                            .row(i)
                            .iter()
                            .zip(x.row(j).iter())
                            .map(|(a, b)| a * b)
                            .sum();

                        let weight = (gamma * dot_product + self.coef0).powi(self.degree as i32);
                        affinity[(i, j)] = weight;
                        affinity[(j, i)] = weight;
                    }
                }
            }
            _ => {
                return Err(SklearsError::InvalidInput(format!(
                    "Unknown affinity type: {}",
                    self.affinity
                )))
            }
        }

        Ok(affinity)
    }

    fn construct_laplacian(&self, affinity: &Array2<f64>) -> SklResult<Array2<f64>> {
        let n_samples = affinity.nrows();

        // Compute degree matrix
        let degrees: Array1<f64> = affinity.sum_axis(Axis(1));

        match self.laplacian_type.as_str() {
            "unnormalized" => {
                // L = D - A
                let mut laplacian = -affinity.clone();
                for i in 0..n_samples {
                    laplacian[(i, i)] += degrees[i];
                }
                Ok(laplacian)
            }
            "symmetric" => {
                // L_sym = D^(-1/2) * L * D^(-1/2) = I - D^(-1/2) * A * D^(-1/2)
                let mut laplacian = Array2::eye(n_samples);

                for i in 0..n_samples {
                    for j in 0..n_samples {
                        if i != j && affinity[(i, j)] > 0.0 {
                            let normalization = (degrees[i] * degrees[j]).sqrt();
                            if normalization > 1e-10 {
                                laplacian[(i, j)] = -affinity[(i, j)] / normalization;
                            }
                        }
                    }
                }
                Ok(laplacian)
            }
            "random_walk" => {
                // L_rw = D^(-1) * L = I - D^(-1) * A
                let mut laplacian = Array2::eye(n_samples);

                for i in 0..n_samples {
                    if degrees[i] > 1e-10 {
                        for j in 0..n_samples {
                            if i != j {
                                laplacian[(i, j)] = -affinity[(i, j)] / degrees[i];
                            }
                        }
                    }
                }
                Ok(laplacian)
            }
            _ => Err(SklearsError::InvalidInput(format!(
                "Unknown Laplacian type: {}",
                self.laplacian_type
            ))),
        }
    }
}

impl SpectralEmbedding<SpectralEmbeddingTrained> {
    /// Get the embedding
    pub fn embedding(&self) -> &Array2<f64> {
        &self.state.embedding
    }

    /// Get the eigenvalues
    pub fn eigenvalues(&self) -> &Array1<f64> {
        &self.state.eigenvalues
    }

    /// Get the affinity matrix
    pub fn affinity_matrix(&self) -> &Array2<f64> {
        &self.state.affinity_matrix
    }
}
