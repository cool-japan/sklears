//! Bregman Divergence Manifold Learning
//!
//! Uses Bregman divergences to learn manifold structure based on the geometry
//! of convex functions and their conjugates.

use super::utils::{
    bregman_mds, bregman_project, compute_bregman_centroids, compute_bregman_divergence_matrix,
};
use scirs2_core::ndarray::{Array2, ArrayView2};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Transform, Untrained},
    types::Float,
};

/// Bregman Divergence Manifold Learning
///
/// Uses Bregman divergences to learn manifold structure based on the geometry
/// of convex functions and their conjugates.
#[derive(Debug, Clone)]
pub struct BregmanDivergenceEmbedding<S = Untrained> {
    state: S,
    n_components: usize,
    n_neighbors: usize,
    divergence_type: BregmanDivergenceType,
    regularization: f64,
    random_state: Option<u64>,
}

#[derive(Debug, Clone, Copy)]
pub enum BregmanDivergenceType {
    /// Squared Euclidean distance (φ(x) = ||x||²/2)
    SquaredEuclidean,
    /// KL divergence (φ(x) = x log x - x)
    KullbackLeibler,
    /// Itakura-Saito divergence (φ(x) = -log x + x - 1)
    ItakuraSaito,
    /// Exponential family divergence (φ(x) = exp(x))
    Exponential,
    /// Log-sum-exp divergence (φ(x) = log(1 + exp(x)))
    LogSumExp,
}

#[derive(Debug, Clone)]
pub struct BregmanTrained {
    embedding: Array2<f64>,
    divergence_matrix: Array2<f64>,
    centroids: Array2<f64>,
    divergence_type: BregmanDivergenceType,
}

impl Default for BregmanDivergenceEmbedding<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl BregmanDivergenceEmbedding<Untrained> {
    /// Create a new Bregman Divergence Embedding instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            n_components: 2,
            n_neighbors: 10,
            divergence_type: BregmanDivergenceType::SquaredEuclidean,
            regularization: 1e-6,
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

    /// Set the divergence type
    pub fn divergence_type(mut self, divergence_type: BregmanDivergenceType) -> Self {
        self.divergence_type = divergence_type;
        self
    }

    /// Set the regularization parameter
    pub fn regularization(mut self, regularization: f64) -> Self {
        self.regularization = regularization;
        self
    }

    /// Set the random state
    pub fn random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }
}

impl Estimator for BregmanDivergenceEmbedding<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, ()> for BregmanDivergenceEmbedding<Untrained> {
    type Fitted = BregmanDivergenceEmbedding<BregmanTrained>;

    fn fit(self, x: &ArrayView2<'_, Float>, _y: &()) -> SklResult<Self::Fitted> {
        let (n_samples, n_features) = x.dim();
        let x_f64 = x.mapv(|v| v);

        if self.n_components > n_features {
            return Err(SklearsError::InvalidInput(
                "n_components cannot be larger than n_features".to_string(),
            ));
        }

        // Compute Bregman divergence matrix
        let divergence_matrix = compute_bregman_divergence_matrix(&x_f64, &self.divergence_type)?;

        // Apply regularization
        let mut regularized_matrix = divergence_matrix.clone();
        for i in 0..n_samples {
            regularized_matrix[[i, i]] += self.regularization;
        }

        // Compute centroids using Bregman centroid algorithm
        let centroids =
            compute_bregman_centroids(&x_f64, &self.divergence_type, self.n_components)?;

        // Perform multidimensional scaling on the divergence matrix
        let embedding = bregman_mds(&regularized_matrix, self.n_components)?;

        let state = BregmanTrained {
            embedding,
            divergence_matrix: regularized_matrix,
            centroids,
            divergence_type: self.divergence_type,
        };

        Ok(BregmanDivergenceEmbedding {
            state,
            n_components: self.n_components,
            n_neighbors: self.n_neighbors,
            divergence_type: self.divergence_type,
            regularization: self.regularization,
            random_state: self.random_state,
        })
    }
}

impl Transform<ArrayView2<'_, Float>, Array2<f64>> for BregmanDivergenceEmbedding<BregmanTrained> {
    fn transform(&self, x: &ArrayView2<'_, Float>) -> SklResult<Array2<f64>> {
        let x_f64 = x.mapv(|v| v);

        // Project new data using Bregman projections
        let projection =
            bregman_project(&x_f64, &self.state.centroids, &self.state.divergence_type)?;

        Ok(projection)
    }
}
