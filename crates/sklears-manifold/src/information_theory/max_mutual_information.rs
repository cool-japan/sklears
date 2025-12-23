//! Maximum Mutual Information Embedding
//!
//! Finds a low-dimensional embedding that maximizes mutual information
//! between the original data and the embedding.

use super::utils::{
    compute_mi_gradient, compute_mutual_information_kde, compute_mutual_information_knn,
    estimate_bandwidth,
};
use scirs2_core::ndarray::{Array2, ArrayView2, Axis};
use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::thread_rng;
use scirs2_core::random::Rng;
use scirs2_core::random::SeedableRng;
use scirs2_core::Distribution;
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Transform, Untrained},
    types::Float,
};

/// Maximum Mutual Information Embedding
///
/// Finds a low-dimensional embedding that maximizes mutual information
/// between the original data and the embedding.
#[derive(Debug, Clone)]
pub struct MaxMutualInformation<S = Untrained> {
    state: S,
    n_components: usize,
    n_neighbors: usize,
    bandwidth: Option<f64>,
    method: String,
    n_iter: usize,
    random_state: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct MMITrained {
    embedding: Array2<f64>,
    mutual_information: f64,
    bandwidth: f64,
}

impl Default for MaxMutualInformation<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl MaxMutualInformation<Untrained> {
    /// Create a new Maximum Mutual Information instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            n_components: 2,
            n_neighbors: 10,
            bandwidth: None,
            method: "knn".to_string(),
            n_iter: 100,
            random_state: None,
        }
    }

    /// Set the number of components
    pub fn n_components(mut self, n_components: usize) -> Self {
        self.n_components = n_components;
        self
    }

    /// Set the number of neighbors for entropy estimation
    pub fn n_neighbors(mut self, n_neighbors: usize) -> Self {
        self.n_neighbors = n_neighbors;
        self
    }

    /// Set the bandwidth for kernel density estimation
    pub fn bandwidth(mut self, bandwidth: f64) -> Self {
        self.bandwidth = Some(bandwidth);
        self
    }

    /// Set the method for mutual information estimation
    pub fn method(mut self, method: &str) -> Self {
        self.method = method.to_string();
        self
    }

    /// Set the number of iterations
    pub fn n_iter(mut self, n_iter: usize) -> Self {
        self.n_iter = n_iter;
        self
    }

    /// Set the random state
    pub fn random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }
}

impl Estimator for MaxMutualInformation<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, ()> for MaxMutualInformation<Untrained> {
    type Fitted = MaxMutualInformation<MMITrained>;

    fn fit(self, x: &ArrayView2<'_, Float>, _y: &()) -> SklResult<Self::Fitted> {
        let (n_samples, n_features) = x.dim();

        if self.n_components > n_features {
            return Err(SklearsError::InvalidInput(
                "n_components cannot be larger than n_features".to_string(),
            ));
        }

        let x_f64 = x.mapv(|v| v);

        // Initialize embedding randomly
        let mut rng = if let Some(seed) = self.random_state {
            StdRng::seed_from_u64(seed)
        } else {
            StdRng::seed_from_u64(thread_rng().random::<u64>())
        };

        let mut embedding = Array2::from_shape_fn((n_samples, self.n_components), |_| {
            scirs2_core::StandardNormal.sample(&mut rng)
        });

        // Estimate bandwidth if not provided
        let bandwidth = self
            .bandwidth
            .unwrap_or_else(|| estimate_bandwidth(&x_f64, self.n_neighbors));

        // Iterative optimization to maximize mutual information
        let mut best_mi = f64::NEG_INFINITY;
        let mut best_embedding = embedding.clone();

        for _iter in 0..self.n_iter {
            // Compute mutual information between original data and embedding
            let mi = match self.method.as_str() {
                "knn" => compute_mutual_information_knn(&x_f64, &embedding, self.n_neighbors)?,
                "kde" => compute_mutual_information_kde(&x_f64, &embedding, bandwidth)?,
                _ => {
                    return Err(SklearsError::InvalidInput(format!(
                        "Unknown method: {}",
                        self.method
                    )))
                }
            };

            if mi > best_mi {
                best_mi = mi;
                best_embedding = embedding.clone();
            }

            // Update embedding using gradient ascent
            let grad = compute_mi_gradient(&x_f64, &embedding, bandwidth)?;
            let learning_rate = 0.01;
            embedding = embedding + learning_rate * grad;

            // Normalize embedding to prevent explosion
            let norm = embedding
                .mapv(|x| x * x)
                .sum_axis(Axis(1))
                .mapv(|x| x.sqrt());
            for (mut row, norm_val) in embedding.rows_mut().into_iter().zip(norm.iter()) {
                if *norm_val > 0.0 {
                    row /= *norm_val;
                }
            }
        }

        let state = MMITrained {
            embedding: best_embedding,
            mutual_information: best_mi,
            bandwidth,
        };

        Ok(MaxMutualInformation {
            state,
            n_components: self.n_components,
            n_neighbors: self.n_neighbors,
            bandwidth: Some(bandwidth),
            method: self.method,
            n_iter: self.n_iter,
            random_state: self.random_state,
        })
    }
}

impl Transform<ArrayView2<'_, Float>, Array2<f64>> for MaxMutualInformation<MMITrained> {
    fn transform(&self, x: &ArrayView2<'_, Float>) -> SklResult<Array2<f64>> {
        // For fitted data, return the stored embedding
        // For new data, this would require out-of-sample extension
        Ok(self.state.embedding.clone())
    }
}
