//! Fisher Information Manifold Learning
//!
//! Uses the Fisher information metric to learn manifold structure based on
//! the geometry of probability distributions.

use super::utils::{compute_global_fisher_information, compute_local_fisher_information};
use scirs2_core::ndarray::ndarray_linalg::{Eigh, UPLO};
use scirs2_core::ndarray::{Array1, Array2, ArrayView2};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Transform, Untrained},
    types::Float,
};

/// Fisher Information Manifold Learning
///
/// Uses the Fisher information metric to learn manifold structure based on
/// the geometry of probability distributions.
#[derive(Debug, Clone)]
pub struct FisherInformationEmbedding<S = Untrained> {
    state: S,
    n_components: usize,
    n_neighbors: usize,
    sigma: f64,
    regularization: f64,
    method: String,
    random_state: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct FIETrained {
    embedding: Array2<f64>,
    fisher_matrix: Array2<f64>,
    eigenvalues: Array1<f64>,
}

impl Default for FisherInformationEmbedding<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl FisherInformationEmbedding<Untrained> {
    /// Create a new Fisher Information Embedding instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            n_components: 2,
            n_neighbors: 10,
            sigma: 1.0,
            regularization: 1e-6,
            method: "local".to_string(),
            random_state: None,
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

    /// Set the bandwidth parameter
    pub fn sigma(mut self, sigma: f64) -> Self {
        self.sigma = sigma;
        self
    }

    /// Set the regularization parameter
    pub fn regularization(mut self, regularization: f64) -> Self {
        self.regularization = regularization;
        self
    }

    /// Set the method for Fisher information computation
    pub fn method(mut self, method: &str) -> Self {
        self.method = method.to_string();
        self
    }

    /// Set the random state
    pub fn random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }
}

impl Estimator for FisherInformationEmbedding<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, ()> for FisherInformationEmbedding<Untrained> {
    type Fitted = FisherInformationEmbedding<FIETrained>;

    fn fit(self, x: &ArrayView2<'_, Float>, _y: &()) -> SklResult<Self::Fitted> {
        let (n_samples, n_features) = x.dim();
        let x_f64 = x.mapv(|v| v);

        // Compute Fisher information matrix
        let fisher_matrix = match self.method.as_str() {
            "local" => compute_local_fisher_information(&x_f64, self.n_neighbors, self.sigma)?,
            "global" => compute_global_fisher_information(&x_f64, self.sigma)?,
            _ => {
                return Err(SklearsError::InvalidInput(format!(
                    "Unknown method: {}",
                    self.method
                )))
            }
        };

        // Add regularization
        let mut regularized_fisher = fisher_matrix;
        for i in 0..n_features {
            regularized_fisher[[i, i]] += self.regularization;
        }

        // Eigendecomposition of Fisher information matrix
        let (eigenvalues, eigenvectors) = regularized_fisher
            .eigh(UPLO::Lower)
            .map_err(|e| SklearsError::InvalidInput(format!("Eigendecomposition failed: {}", e)))?;

        // Sort eigenvalues and eigenvectors in descending order
        let mut eigen_pairs: Vec<_> = eigenvalues.iter().zip(eigenvectors.columns()).collect();
        eigen_pairs.sort_by(|a, b| b.0.partial_cmp(a.0).unwrap());

        // Take the top n_components eigenvectors
        let selected_eigenvectors: Array2<f64> =
            Array2::from_shape_fn((n_features, self.n_components), |(i, j)| {
                eigen_pairs[j].1[i]
            });

        let selected_eigenvalues: Array1<f64> =
            Array1::from_shape_fn(self.n_components, |i| *eigen_pairs[i].0);

        // Project data onto Fisher information coordinates
        let embedding = x_f64.dot(&selected_eigenvectors);

        let state = FIETrained {
            embedding,
            fisher_matrix: regularized_fisher,
            eigenvalues: selected_eigenvalues,
        };

        Ok(FisherInformationEmbedding {
            state,
            n_components: self.n_components,
            n_neighbors: self.n_neighbors,
            sigma: self.sigma,
            regularization: self.regularization,
            method: self.method,
            random_state: self.random_state,
        })
    }
}

impl Transform<ArrayView2<'_, Float>, Array2<f64>> for FisherInformationEmbedding<FIETrained> {
    fn transform(&self, x: &ArrayView2<'_, Float>) -> SklResult<Array2<f64>> {
        // For fitted data, return the stored embedding
        // For new data, would need to compute projection
        Ok(self.state.embedding.clone())
    }
}
