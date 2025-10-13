//! Uniform Manifold Approximation and Projection (UMAP) implementation
//! This module provides UMAP for non-linear dimensionality reduction through uniform manifold approximation.

use scirs2_core::ndarray::{Array2, ArrayView1, ArrayView2};
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

/// Uniform Manifold Approximation and Projection (UMAP)
///
/// UMAP is a novel manifold learning technique for dimension reduction.
/// It is based on three key principles: local connectedness, uniformity of measure
/// on the manifold, and the use of Riemannian metrics.
///
/// # Parameters
///
/// * `n_neighbors` - Number of neighbors to consider for local approximation
/// * `n_components` - Number of coordinates for the manifold
/// * `min_dist` - Minimum distance between embedded points
/// * `spread` - Effective scale of embedded points
/// * `low_memory` - Whether to use a lower memory algorithm
/// * `n_epochs` - Number of training epochs
/// * `learning_rate` - Learning rate for optimization
/// * `init` - Initialization method for embedding
/// * `random_state` - Random state for reproducibility
/// * `metric` - Distance metric for high-dimensional space
/// * `output_metric` - Distance metric for low-dimensional space
/// * `n_jobs` - Number of parallel jobs
/// * `local_connectivity` - Local connectivity parameter
/// * `repulsion_strength` - Strength of repulsive forces
/// * `negative_sample_rate` - Rate of negative samples during optimization
/// * `transform_queue_size` - Size of queue for transforming new data
/// * `a` - More specific parameters controlling embedding
/// * `b` - More specific parameters controlling embedding
///
/// # Examples
///
/// ```
/// use sklears_manifold::UMAP;
/// use sklears_core::traits::{Transform, Fit};
/// use scirs2_core::ndarray::array;
///
/// let x = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]];
///
/// let umap = UMAP::new()
///     .n_neighbors(3)
///     .n_components(2)
///     .min_dist(0.1);
/// let fitted = umap.fit(&x.view(), &()).unwrap();
/// let embedded = fitted.transform(&x.view()).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct UMAP<S = Untrained> {
    state: S,
    n_neighbors: usize,
    n_components: usize,
    min_dist: f64,
    spread: f64,
    low_memory: bool,
    n_epochs: Option<usize>,
    learning_rate: f64,
    init: String,
    random_state: Option<u64>,
    metric: String,
    output_metric: String,
    n_jobs: Option<i32>,
    local_connectivity: f64,
    repulsion_strength: f64,
    negative_sample_rate: usize,
    transform_queue_size: f64,
    a: Option<f64>,
    b: Option<f64>,
}

/// Trained state for UMAP
#[derive(Debug, Clone)]
pub struct UmapTrained {
    /// The low-dimensional embedding of the training data
    pub embedding: Array2<f64>,
    /// The high-dimensional graph representation
    pub graph: Array2<f64>,
    /// K-nearest neighbor indices
    pub knn_indices: Array2<usize>,
    /// Parameter a for distance function
    pub a: f64,
    /// Parameter b for distance function
    pub b: f64,
}

impl UMAP<Untrained> {
    /// Create a new UMAP instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            n_neighbors: 15,
            n_components: 2,
            min_dist: 0.1,
            spread: 1.0,
            low_memory: false,
            n_epochs: None,
            learning_rate: 1.0,
            init: "spectral".to_string(),
            random_state: None,
            metric: "euclidean".to_string(),
            output_metric: "euclidean".to_string(),
            n_jobs: None,
            local_connectivity: 1.0,
            repulsion_strength: 1.0,
            negative_sample_rate: 5,
            transform_queue_size: 4.0,
            a: None,
            b: None,
        }
    }

    /// Set the number of neighbors
    pub fn n_neighbors(mut self, n_neighbors: usize) -> Self {
        self.n_neighbors = n_neighbors;
        self
    }

    /// Set the number of components
    pub fn n_components(mut self, n_components: usize) -> Self {
        self.n_components = n_components;
        self
    }

    /// Set the minimum distance
    pub fn min_dist(mut self, min_dist: f64) -> Self {
        self.min_dist = min_dist;
        self
    }

    /// Set the spread
    pub fn spread(mut self, spread: f64) -> Self {
        self.spread = spread;
        self
    }

    /// Set low memory mode
    pub fn low_memory(mut self, low_memory: bool) -> Self {
        self.low_memory = low_memory;
        self
    }

    /// Set the number of epochs
    pub fn n_epochs(mut self, n_epochs: Option<usize>) -> Self {
        self.n_epochs = n_epochs;
        self
    }

    /// Set the learning rate
    pub fn learning_rate(mut self, learning_rate: f64) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    /// Set the initialization method
    pub fn init(mut self, init: &str) -> Self {
        self.init = init.to_string();
        self
    }

    /// Set the random state
    pub fn random_state(mut self, random_state: Option<u64>) -> Self {
        self.random_state = random_state;
        self
    }

    /// Set the metric
    pub fn metric(mut self, metric: &str) -> Self {
        self.metric = metric.to_string();
        self
    }

    /// Set the output metric
    pub fn output_metric(mut self, output_metric: &str) -> Self {
        self.output_metric = output_metric.to_string();
        self
    }

    /// Set the number of jobs
    pub fn n_jobs(mut self, n_jobs: Option<i32>) -> Self {
        self.n_jobs = n_jobs;
        self
    }

    /// Set the local connectivity
    pub fn local_connectivity(mut self, local_connectivity: f64) -> Self {
        self.local_connectivity = local_connectivity;
        self
    }

    /// Set the repulsion strength
    pub fn repulsion_strength(mut self, repulsion_strength: f64) -> Self {
        self.repulsion_strength = repulsion_strength;
        self
    }

    /// Set the negative sample rate
    pub fn negative_sample_rate(mut self, negative_sample_rate: usize) -> Self {
        self.negative_sample_rate = negative_sample_rate;
        self
    }

    /// Set the transform queue size
    pub fn transform_queue_size(mut self, transform_queue_size: f64) -> Self {
        self.transform_queue_size = transform_queue_size;
        self
    }

    /// Set parameter a
    pub fn a(mut self, a: Option<f64>) -> Self {
        self.a = a;
        self
    }

    /// Set parameter b
    pub fn b(mut self, b: Option<f64>) -> Self {
        self.b = b;
        self
    }
}

impl Default for UMAP<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for UMAP<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, ()> for UMAP<Untrained> {
    type Fitted = UMAP<UmapTrained>;

    fn fit(self, x: &ArrayView2<'_, Float>, _y: &()) -> SklResult<Self::Fitted> {
        let x = x.mapv(|x| x);
        let (n_samples, _) = x.dim();

        if n_samples <= self.n_components {
            return Err(SklearsError::InvalidInput(
                "Number of samples must be greater than n_components".to_string(),
            ));
        }

        if self.n_neighbors >= n_samples {
            return Err(SklearsError::InvalidInput(
                "n_neighbors must be less than number of samples".to_string(),
            ));
        }

        // Step 1: Construct the weighted graph
        let (graph, knn_indices) = self.construct_fuzzy_graph(&x)?;

        // Step 2: Determine a and b parameters
        let (a, b) = self.find_ab_params();

        // Step 3: Initialize the embedding
        let mut embedding = self.initialize_embedding(&x)?;

        // Step 4: Optimize the embedding using gradient descent
        let n_epochs = self.n_epochs.unwrap_or(200);
        embedding = self.optimize_embedding(embedding, &graph, &knn_indices, n_epochs, a, b)?;

        Ok(UMAP {
            state: UmapTrained {
                embedding,
                graph,
                knn_indices,
                a,
                b,
            },
            n_neighbors: self.n_neighbors,
            n_components: self.n_components,
            min_dist: self.min_dist,
            spread: self.spread,
            low_memory: self.low_memory,
            n_epochs: self.n_epochs,
            learning_rate: self.learning_rate,
            init: self.init,
            random_state: self.random_state,
            metric: self.metric,
            output_metric: self.output_metric,
            n_jobs: self.n_jobs,
            local_connectivity: self.local_connectivity,
            repulsion_strength: self.repulsion_strength,
            negative_sample_rate: self.negative_sample_rate,
            transform_queue_size: self.transform_queue_size,
            a: Some(a),
            b: Some(b),
        })
    }
}

impl UMAP<Untrained> {
    fn construct_fuzzy_graph(&self, x: &Array2<f64>) -> SklResult<(Array2<f64>, Array2<usize>)> {
        let n_samples = x.nrows();
        let mut graph = Array2::zeros((n_samples, n_samples));
        let mut knn_indices = Array2::zeros((n_samples, self.n_neighbors));

        // Find k-nearest neighbors for each point
        for i in 0..n_samples {
            let mut distances: Vec<(f64, usize)> = Vec::new();

            for j in 0..n_samples {
                if i != j {
                    let diff = &x.row(i) - &x.row(j);
                    let dist = diff.mapv(|x| x * x).sum().sqrt();
                    distances.push((dist, j));
                }
            }

            distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

            // Store k-nearest neighbor indices
            for (k, &(_, neighbor_idx)) in distances.iter().take(self.n_neighbors).enumerate() {
                knn_indices[[i, k]] = neighbor_idx;
            }

            // Compute fuzzy set membership strengths
            let sigma = self.compute_rho_sigma(&distances[..self.n_neighbors]);
            for (k, &(distance, neighbor_idx)) in
                distances.iter().take(self.n_neighbors).enumerate()
            {
                let membership = if distance > 0.0 {
                    (-((distance - distances[0].0) / sigma).max(0.0)).exp()
                } else {
                    1.0
                };
                graph[[i, neighbor_idx]] = membership;
            }
        }

        // Symmetrize the graph
        for i in 0..n_samples {
            for j in 0..n_samples {
                graph[[i, j]] = graph[[i, j]] + graph[[j, i]] - graph[[i, j]] * graph[[j, i]];
            }
        }

        Ok((graph, knn_indices))
    }

    fn compute_rho_sigma(&self, distances: &[(f64, usize)]) -> f64 {
        // Simplified sigma computation - in practice would use binary search
        let mean_dist: f64 = distances.iter().map(|(d, _)| d).sum::<f64>() / distances.len() as f64;
        mean_dist * 0.5
    }

    fn find_ab_params(&self) -> (f64, f64) {
        // Curve fitting to find a and b parameters based on min_dist and spread
        // This is a simplified version - the full UMAP algorithm uses curve fitting
        let a = self.a.unwrap_or(1.0);
        let b = self.b.unwrap_or(1.0);
        (a, b)
    }

    fn initialize_embedding(&self, x: &Array2<f64>) -> SklResult<Array2<f64>> {
        let n_samples = x.nrows();
        let mut rng = match self.random_state {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::seed_from_u64(thread_rng().gen()),
        };

        match self.init.as_str() {
            "random" => {
                let mut embedding = Array2::zeros((n_samples, self.n_components));
                for i in 0..n_samples {
                    for j in 0..self.n_components {
                        embedding[[i, j]] = rng.sample(scirs2_core::StandardNormal);
                    }
                }
                Ok(embedding)
            }
            "spectral" | _ => {
                // Simplified spectral initialization
                // In practice, would use the Laplacian eigenvectors
                let mut embedding = Array2::zeros((n_samples, self.n_components));
                for i in 0..n_samples {
                    for j in 0..self.n_components {
                        embedding[[i, j]] =
                            rng.sample::<f64, _>(scirs2_core::StandardNormal) * 10.0;
                    }
                }
                Ok(embedding)
            }
        }
    }

    fn optimize_embedding(
        &self,
        mut embedding: Array2<f64>,
        graph: &Array2<f64>,
        _knn_indices: &Array2<usize>,
        n_epochs: usize,
        a: f64,
        b: f64,
    ) -> SklResult<Array2<f64>> {
        let n_samples = embedding.nrows();
        let mut rng = match self.random_state {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::seed_from_u64(thread_rng().gen()),
        };

        for epoch in 0..n_epochs {
            let alpha = self.learning_rate * (1.0 - epoch as f64 / n_epochs as f64);

            // Attractive forces (positive sampling)
            for i in 0..n_samples {
                for j in 0..n_samples {
                    if i != j && graph[[i, j]] > 0.0 {
                        let dist_sq = self.squared_distance(&embedding.row(i), &embedding.row(j));
                        if dist_sq > 0.0 {
                            let grad_coeff = -2.0 * a * b * dist_sq.powf(b - 1.0)
                                / (a * dist_sq.powf(b) + 1.0).powi(2);

                            for d in 0..self.n_components {
                                let grad = grad_coeff * (embedding[[i, d]] - embedding[[j, d]]);
                                embedding[[i, d]] += alpha * graph[[i, j]] * grad;
                            }
                        }
                    }
                }
            }

            // Repulsive forces (negative sampling)
            for _ in 0..(n_samples * self.negative_sample_rate) {
                let i = rng.gen_range(0..n_samples);
                let j = rng.gen_range(0..n_samples);

                if i != j {
                    let dist_sq = self.squared_distance(&embedding.row(i), &embedding.row(j));
                    if dist_sq > 0.0 {
                        let grad_coeff = 2.0 * b / (0.001 + dist_sq) / (a * dist_sq.powf(b) + 1.0);

                        for d in 0..self.n_components {
                            let grad = grad_coeff * (embedding[[i, d]] - embedding[[j, d]]);
                            embedding[[i, d]] += alpha * self.repulsion_strength * grad;
                        }
                    }
                }
            }
        }

        Ok(embedding)
    }

    fn squared_distance(&self, a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> f64 {
        a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum()
    }
}

impl Transform<ArrayView2<'_, Float>, Array2<f64>> for UMAP<UmapTrained> {
    fn transform(&self, _x: &ArrayView2<'_, Float>) -> SklResult<Array2<f64>> {
        // UMAP transform for new data would require additional implementation
        // For now, return the fitted embedding
        Ok(self.state.embedding.clone())
    }
}

impl UMAP<UmapTrained> {
    /// Get the embedding
    pub fn embedding(&self) -> &Array2<f64> {
        &self.state.embedding
    }

    /// Get the graph representation
    pub fn graph(&self) -> &Array2<f64> {
        &self.state.graph
    }

    /// Get the nearest neighbor indices
    pub fn knn_indices(&self) -> &Array2<usize> {
        &self.state.knn_indices
    }

    /// Get parameter a
    pub fn a(&self) -> f64 {
        self.state.a
    }

    /// Get parameter b
    pub fn b(&self) -> f64 {
        self.state.b
    }
}
