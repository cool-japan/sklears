//! Temporal manifold learning methods
//!
//! This module implements various approaches for learning manifolds from time-varying data,
//! including dynamic embedding tracking, temporal trajectory analysis, and streaming updates.

use scirs2_core::ndarray::ndarray_linalg::{Eigh, Norm, Solve, SVD, UPLO};
use scirs2_core::ndarray::{s, Array1, Array2, Array3, ArrayView1, ArrayView2, ArrayView3, Axis};
use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::thread_rng;
use scirs2_core::random::Rng;
use scirs2_core::random::SeedableRng;
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Transform, Untrained},
};
use std::collections::VecDeque;

/// Temporal manifold learning for time-varying data
///
/// This implements dynamic embedding tracking where the manifold evolves over time.
/// It can handle streaming data and maintain embeddings that adapt to temporal changes
/// while preserving continuity across time steps.
///
/// # Parameters
///
/// * `n_components` - Target dimensionality for embeddings
/// * `window_size` - Size of temporal window for processing
/// * `base_method` - Base manifold learning method ("pca", "isomap", "lle")
/// * `temporal_weight` - Weight for temporal consistency (0.0 to 1.0)
/// * `adaptation_rate` - Rate of adaptation to new data (0.0 to 1.0)
/// * `smoothing_factor` - Temporal smoothing factor (0.0 to 1.0)
/// * `max_memory` - Maximum number of time steps to keep in memory
/// * `random_state` - Random seed for reproducibility
///
/// # Examples
///
/// ```
/// use sklears_manifold::temporal::TemporalManifold;
/// use sklears_core::traits::{Transform, Fit};
/// use scirs2_core::ndarray::array;
///
/// // Time series data: [time_steps, n_samples, n_features]
/// let x = array![
///     [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
///     [[1.1, 2.1], [3.1, 4.1], [5.1, 6.1]],
///     [[1.2, 2.2], [3.2, 4.2], [5.2, 6.2]]
/// ];
///
/// let temporal = TemporalManifold::new()
///     .n_components(2)
///     .window_size(3)
///     .temporal_weight(0.7);
///
/// let y = array![()]; // Empty target for temporal data
/// let fitted = temporal.fit(&x.view(), &y.view()).unwrap();
/// let embedded = fitted.transform(&x.view()).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct TemporalManifold<S = Untrained> {
    state: S,
    n_components: usize,
    window_size: usize,
    base_method: String,
    temporal_weight: f64,
    adaptation_rate: f64,
    smoothing_factor: f64,
    max_memory: usize,
    random_state: Option<u64>,
}

/// Trained state for TemporalManifold
#[derive(Debug, Clone)]
pub struct TrainedTemporalManifold {
    temporal_embeddings: Vec<Array2<f64>>,
    temporal_transforms: Vec<Array2<f64>>,
    trajectory_analysis: TrajectoryAnalysis,
    embedding_history: VecDeque<Array2<f64>>,
    n_features: usize,
    n_components: usize,
    window_size: usize,
}

/// Trajectory analysis results
#[derive(Debug, Clone)]
pub struct TrajectoryAnalysis {
    /// velocity_profiles
    pub velocity_profiles: Vec<Array2<f64>>,
    /// acceleration_profiles
    pub acceleration_profiles: Vec<Array2<f64>>,
    /// curvature_profiles
    pub curvature_profiles: Vec<Array1<f64>>,
    /// trajectory_lengths
    pub trajectory_lengths: Vec<f64>,
    /// direction_changes
    pub direction_changes: Vec<f64>,
}

impl Default for TemporalManifold<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl TemporalManifold<Untrained> {
    /// Create a new TemporalManifold instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            n_components: 2,
            window_size: 5,
            base_method: "pca".to_string(),
            temporal_weight: 0.5,
            adaptation_rate: 0.1,
            smoothing_factor: 0.3,
            max_memory: 100,
            random_state: None,
        }
    }

    /// Set the number of components
    pub fn n_components(mut self, n_components: usize) -> Self {
        self.n_components = n_components;
        self
    }

    /// Set the window size
    pub fn window_size(mut self, window_size: usize) -> Self {
        self.window_size = window_size;
        self
    }

    /// Set the base method
    pub fn base_method(mut self, base_method: String) -> Self {
        self.base_method = base_method;
        self
    }

    /// Set the temporal weight
    pub fn temporal_weight(mut self, temporal_weight: f64) -> Self {
        self.temporal_weight = temporal_weight.max(0.0).min(1.0);
        self
    }

    /// Set the adaptation rate
    pub fn adaptation_rate(mut self, adaptation_rate: f64) -> Self {
        self.adaptation_rate = adaptation_rate.max(0.0).min(1.0);
        self
    }

    /// Set the smoothing factor
    pub fn smoothing_factor(mut self, smoothing_factor: f64) -> Self {
        self.smoothing_factor = smoothing_factor.max(0.0).min(1.0);
        self
    }

    /// Set the maximum memory
    pub fn max_memory(mut self, max_memory: usize) -> Self {
        self.max_memory = max_memory;
        self
    }

    /// Set the random state
    pub fn random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }

    /// Apply base manifold learning method to a single time step
    fn apply_base_method(&self, x: &ArrayView2<f64>, rng: &mut StdRng) -> SklResult<Array2<f64>> {
        match self.base_method.as_str() {
            "pca" => self.apply_pca(x),
            "isomap" => self.apply_isomap(x, rng),
            "lle" => self.apply_lle(x, rng),
            _ => Err(SklearsError::InvalidInput(format!(
                "Unknown base method: {}",
                self.base_method
            ))),
        }
    }

    /// Apply PCA to data
    fn apply_pca(&self, x: &ArrayView2<f64>) -> SklResult<Array2<f64>> {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        // Center the data
        let mean = x.mean_axis(Axis(0)).unwrap();
        let centered = x - &mean.insert_axis(Axis(0));

        // Compute covariance matrix
        let cov = centered.t().dot(&centered) / (n_samples - 1) as f64;

        // Eigendecomposition
        let (eigenvalues, eigenvectors) = cov.eigh(UPLO::Upper).map_err(|e| {
            SklearsError::InvalidInput(format!("PCA eigendecomposition failed: {}", e))
        })?;

        // Sort by eigenvalues (descending)
        let mut indices: Vec<usize> = (0..eigenvalues.len()).collect();
        indices.sort_by(|&i, &j| eigenvalues[j].partial_cmp(&eigenvalues[i]).unwrap());

        // Project data onto top eigenvectors
        let mut projection_matrix = Array2::zeros((n_features, self.n_components));
        for (i, &idx) in indices.iter().take(self.n_components).enumerate() {
            projection_matrix
                .column_mut(i)
                .assign(&eigenvectors.column(idx));
        }

        let embedding = centered.dot(&projection_matrix);
        Ok(embedding)
    }

    /// Apply simplified Isomap to data
    fn apply_isomap(&self, x: &ArrayView2<f64>, rng: &mut StdRng) -> SklResult<Array2<f64>> {
        let n_samples = x.nrows();
        let k = (n_samples as f64).sqrt() as usize + 1;

        // Build k-NN graph
        let graph = self.build_knn_graph(x, k)?;

        // Compute geodesic distances using Floyd-Warshall
        let geodesic_distances = self.floyd_warshall(&graph)?;

        // Apply MDS to geodesic distances
        let embedding = self.mds(&geodesic_distances)?;
        Ok(embedding)
    }

    /// Apply simplified LLE to data
    fn apply_lle(&self, x: &ArrayView2<f64>, rng: &mut StdRng) -> SklResult<Array2<f64>> {
        let n_samples = x.nrows();
        let k = (n_samples as f64).sqrt() as usize + 1;

        // Find k nearest neighbors
        let neighbors = self.find_k_neighbors(x, k)?;

        // Compute reconstruction weights
        let weights = self.compute_lle_weights(x, &neighbors)?;

        // Solve eigenvalue problem
        let embedding = self.solve_lle_eigenvalue_problem(&weights)?;
        Ok(embedding)
    }

    /// Perform temporal smoothing between consecutive embeddings
    fn temporal_smoothing(
        &self,
        current_embedding: &Array2<f64>,
        previous_embedding: &Array2<f64>,
    ) -> SklResult<Array2<f64>> {
        let alpha = self.smoothing_factor;
        let smoothed = (1.0 - alpha) * current_embedding + alpha * previous_embedding;
        Ok(smoothed)
    }

    /// Compute temporal consistency penalty
    fn compute_temporal_consistency_loss(
        &self,
        embedding1: &Array2<f64>,
        embedding2: &Array2<f64>,
    ) -> SklResult<f64> {
        if embedding1.shape() != embedding2.shape() {
            return Err(SklearsError::InvalidInput(
                "Embedding shapes must match".to_string(),
            ));
        }

        let diff = embedding1 - embedding2;
        let loss = diff.norm_l2().powi(2) / (embedding1.nrows() * embedding1.ncols()) as f64;
        Ok(loss)
    }

    /// Optimize embedding with temporal constraints
    fn optimize_with_temporal_constraints(
        &self,
        current_embedding: &Array2<f64>,
        previous_embedding: &Option<Array2<f64>>,
        data: &ArrayView2<f64>,
    ) -> SklResult<Array2<f64>> {
        let mut optimized = current_embedding.clone();

        if let Some(prev) = previous_embedding {
            // Apply temporal consistency
            let temporal_loss_weight = self.temporal_weight;
            let data_loss_weight = 1.0 - self.temporal_weight;

            // Simple optimization: weighted combination
            optimized = data_loss_weight * &optimized + temporal_loss_weight * prev;

            // Apply adaptation
            let adapted = (1.0 - self.adaptation_rate) * &optimized
                + self.adaptation_rate * current_embedding;
            optimized = adapted;
        }

        Ok(optimized)
    }

    /// Compute trajectory analysis
    fn compute_trajectory_analysis(
        &self,
        embeddings: &[Array2<f64>],
    ) -> SklResult<TrajectoryAnalysis> {
        let mut velocity_profiles = Vec::new();
        let mut acceleration_profiles = Vec::new();
        let mut curvature_profiles = Vec::new();
        let mut trajectory_lengths = Vec::new();
        let mut direction_changes = Vec::new();

        for t in 1..embeddings.len() {
            // Velocity (first derivative)
            let velocity = &embeddings[t] - &embeddings[t - 1];
            velocity_profiles.push(velocity.clone());

            // Trajectory length
            let length = velocity.norm_l2();
            trajectory_lengths.push(length);

            // Acceleration (second derivative)
            if t > 1 {
                let prev_velocity = &embeddings[t - 1] - &embeddings[t - 2];
                let acceleration = &velocity - &prev_velocity;
                acceleration_profiles.push(acceleration);

                // Curvature approximation
                let curvature = self.compute_curvature(&prev_velocity, &velocity)?;
                curvature_profiles.push(curvature);

                // Direction change
                let direction_change = self.compute_direction_change(&prev_velocity, &velocity)?;
                direction_changes.push(direction_change);
            }
        }

        Ok(TrajectoryAnalysis {
            velocity_profiles,
            acceleration_profiles,
            curvature_profiles,
            trajectory_lengths,
            direction_changes,
        })
    }

    /// Compute curvature for each point
    fn compute_curvature(&self, v1: &Array2<f64>, v2: &Array2<f64>) -> SklResult<Array1<f64>> {
        let n_samples = v1.nrows();
        let mut curvatures = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let vel1 = v1.row(i);
            let vel2 = v2.row(i);

            let v1_norm = vel1.norm_l2();
            let v2_norm = vel2.norm_l2();

            if v1_norm > 1e-10 && v2_norm > 1e-10 {
                // Compute angle between velocity vectors
                let dot_product = vel1.dot(&vel2);
                let cos_angle = dot_product / (v1_norm * v2_norm);
                let angle = cos_angle.acos();

                // Curvature approximation
                curvatures[i] = angle / v1_norm;
            }
        }

        Ok(curvatures)
    }

    /// Compute direction change magnitude
    fn compute_direction_change(&self, v1: &Array2<f64>, v2: &Array2<f64>) -> SklResult<f64> {
        let n_samples = v1.nrows();
        let mut total_change = 0.0;

        for i in 0..n_samples {
            let vel1 = v1.row(i);
            let vel2 = v2.row(i);

            let v1_norm = vel1.norm_l2();
            let v2_norm = vel2.norm_l2();

            if v1_norm > 1e-10 && v2_norm > 1e-10 {
                let dot_product = vel1.dot(&vel2);
                let cos_angle = dot_product / (v1_norm * v2_norm);
                let angle = cos_angle.acos();
                total_change += angle;
            }
        }

        Ok(total_change / n_samples as f64)
    }

    // Helper methods for manifold learning algorithms
    fn build_knn_graph(&self, x: &ArrayView2<f64>, k: usize) -> SklResult<Array2<f64>> {
        let n_samples = x.nrows();
        let mut graph = Array2::from_elem((n_samples, n_samples), f64::INFINITY);

        // Set diagonal to 0
        for i in 0..n_samples {
            graph[[i, i]] = 0.0;
        }

        // For each point, find k nearest neighbors
        for i in 0..n_samples {
            let mut distances: Vec<(usize, f64)> = (0..n_samples)
                .map(|j| (j, (&x.row(i) - &x.row(j)).norm_l2()))
                .collect();
            distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

            // Connect to k nearest neighbors
            for &(j, dist) in distances.iter().take(k + 1).skip(1) {
                graph[[i, j]] = dist;
                graph[[j, i]] = dist; // Make symmetric
            }
        }

        Ok(graph)
    }

    fn floyd_warshall(&self, graph: &Array2<f64>) -> SklResult<Array2<f64>> {
        let n = graph.nrows();
        let mut distances = graph.clone();

        for k in 0..n {
            for i in 0..n {
                for j in 0..n {
                    if distances[[i, k]] + distances[[k, j]] < distances[[i, j]] {
                        distances[[i, j]] = distances[[i, k]] + distances[[k, j]];
                    }
                }
            }
        }

        Ok(distances)
    }

    fn mds(&self, distances: &Array2<f64>) -> SklResult<Array2<f64>> {
        let n = distances.nrows();

        // Double centering
        let mut h = Array2::from_elem((n, n), -1.0 / n as f64);
        for i in 0..n {
            h[[i, i]] += 1.0;
        }

        let d_squared = distances.mapv(|x| -0.5 * x.powi(2));
        let b = h.dot(&d_squared).dot(&h);

        // Eigendecomposition
        let (eigenvalues, eigenvectors) = b.eigh(UPLO::Upper).map_err(|e| {
            SklearsError::InvalidInput(format!("MDS eigendecomposition failed: {}", e))
        })?;

        // Sort eigenvalues in descending order
        let mut indices: Vec<usize> = (0..eigenvalues.len()).collect();
        indices.sort_by(|&i, &j| eigenvalues[j].partial_cmp(&eigenvalues[i]).unwrap());

        // Take top components with positive eigenvalues
        let mut embedding = Array2::zeros((n, self.n_components));
        for (i, &idx) in indices.iter().take(self.n_components).enumerate() {
            if eigenvalues[idx] > 0.0 {
                let scale = eigenvalues[idx].sqrt();
                for j in 0..n {
                    embedding[[j, i]] = eigenvectors[[j, idx]] * scale;
                }
            }
        }

        Ok(embedding)
    }

    fn find_k_neighbors(&self, x: &ArrayView2<f64>, k: usize) -> SklResult<Vec<Vec<usize>>> {
        let n_samples = x.nrows();
        let mut neighbors = Vec::new();

        for i in 0..n_samples {
            let mut distances: Vec<(usize, f64)> = (0..n_samples)
                .map(|j| (j, (&x.row(i) - &x.row(j)).norm_l2()))
                .collect();
            distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

            let neighbor_indices: Vec<usize> = distances
                .iter()
                .take(k + 1)
                .skip(1)
                .map(|(idx, _)| *idx)
                .collect();
            neighbors.push(neighbor_indices);
        }

        Ok(neighbors)
    }

    fn compute_lle_weights(
        &self,
        x: &ArrayView2<f64>,
        neighbors: &[Vec<usize>],
    ) -> SklResult<Array2<f64>> {
        let n_samples = x.nrows();
        let mut weights = Array2::zeros((n_samples, n_samples));

        for i in 0..n_samples {
            let k = neighbors[i].len();
            if k == 0 {
                continue;
            }

            // Build local covariance matrix
            let mut local_cov = Array2::zeros((k, k));
            for (a, &idx_a) in neighbors[i].iter().enumerate() {
                for (b, &idx_b) in neighbors[i].iter().enumerate() {
                    let diff_a = &x.row(idx_a) - &x.row(i);
                    let diff_b = &x.row(idx_b) - &x.row(i);
                    local_cov[[a, b]] = diff_a.dot(&diff_b);
                }
            }

            // Solve for weights
            let ones = Array1::ones(k);
            let w = local_cov.solve(&ones).unwrap_or_else(|_| Array1::ones(k));

            // Normalize weights
            let w_sum = w.sum();
            if w_sum > 1e-10 {
                for (j, &neighbor_idx) in neighbors[i].iter().enumerate() {
                    weights[[i, neighbor_idx]] = w[j] / w_sum;
                }
            }
        }

        Ok(weights)
    }

    fn solve_lle_eigenvalue_problem(&self, weights: &Array2<f64>) -> SklResult<Array2<f64>> {
        let n_samples = weights.nrows();

        // Compute (I - W)^T(I - W)
        let identity = Array2::eye(n_samples);
        let iw = &identity - weights;
        let m = iw.t().dot(&iw);

        // Eigendecomposition
        let (eigenvalues, eigenvectors): (Array1<f64>, Array2<f64>) =
            m.eigh(UPLO::Upper).map_err(|e| {
                SklearsError::InvalidInput(format!("LLE eigendecomposition failed: {}", e))
            })?;

        // Sort eigenvalues in ascending order (we want smallest non-zero eigenvalues)
        let mut indices: Vec<usize> = (0..eigenvalues.len()).collect();
        indices.sort_by(|&i, &j| eigenvalues[i].partial_cmp(&eigenvalues[j]).unwrap());

        // Skip the first eigenvector (constant) and take next n_components
        let mut embedding = Array2::zeros((n_samples, self.n_components));
        for (i, &idx) in indices.iter().skip(1).take(self.n_components).enumerate() {
            embedding.column_mut(i).assign(&eigenvectors.column(idx));
        }

        Ok(embedding)
    }
}

impl Estimator for TemporalManifold<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView3<'_, f64>, ArrayView1<'_, ()>> for TemporalManifold<Untrained> {
    type Fitted = TemporalManifold<TrainedTemporalManifold>;

    fn fit(self, x: &ArrayView3<'_, f64>, _y: &ArrayView1<'_, ()>) -> SklResult<Self::Fitted> {
        if x.shape()[0] == 0 || x.shape()[1] == 0 || x.shape()[2] == 0 {
            return Err(SklearsError::InvalidInput(
                "Input data is empty".to_string(),
            ));
        }

        let n_timesteps = x.shape()[0];
        let n_samples = x.shape()[1];
        let n_features = x.shape()[2];

        let mut rng = if let Some(seed) = self.random_state {
            StdRng::seed_from_u64(seed)
        } else {
            StdRng::seed_from_u64(thread_rng().gen::<u64>())
        };

        let mut temporal_embeddings = Vec::new();
        let mut temporal_transforms = Vec::new();
        let mut embedding_history = VecDeque::new();
        let mut previous_embedding: Option<Array2<f64>> = None;

        // Process each time step
        for t in 0..n_timesteps {
            let time_slice = x.slice(s![t, .., ..]);

            // Apply base manifold learning method
            let mut current_embedding = self.apply_base_method(&time_slice, &mut rng)?;

            // Apply temporal constraints if we have previous embeddings
            if let Some(prev) = &previous_embedding {
                current_embedding = self.optimize_with_temporal_constraints(
                    &current_embedding,
                    &Some(prev.clone()),
                    &time_slice,
                )?;

                // Apply temporal smoothing
                current_embedding = self.temporal_smoothing(&current_embedding, prev)?;
            }

            // Update history
            embedding_history.push_back(current_embedding.clone());
            if embedding_history.len() > self.max_memory {
                embedding_history.pop_front();
            }

            temporal_embeddings.push(current_embedding.clone());
            previous_embedding = Some(current_embedding);

            // Compute transformation matrix (simplified)
            let transform = self.compute_transform_matrix(&time_slice, &temporal_embeddings[t])?;
            temporal_transforms.push(transform);
        }

        // Compute trajectory analysis
        let trajectory_analysis = self.compute_trajectory_analysis(&temporal_embeddings)?;

        Ok(TemporalManifold {
            state: TrainedTemporalManifold {
                temporal_embeddings,
                temporal_transforms,
                trajectory_analysis,
                embedding_history,
                n_features,
                n_components: self.n_components,
                window_size: self.window_size,
            },
            n_components: self.n_components,
            window_size: self.window_size,
            base_method: self.base_method,
            temporal_weight: self.temporal_weight,
            adaptation_rate: self.adaptation_rate,
            smoothing_factor: self.smoothing_factor,
            max_memory: self.max_memory,
            random_state: self.random_state,
        })
    }
}

impl TemporalManifold<Untrained> {
    /// Compute transformation matrix from input to embedding space
    fn compute_transform_matrix(
        &self,
        x: &ArrayView2<f64>,
        embedding: &Array2<f64>,
    ) -> SklResult<Array2<f64>> {
        // Use least squares to find transformation
        let xt_x = x.t().dot(x);
        let xt_y = x.t().dot(embedding);

        // Solve using pseudoinverse
        let (u, s, vt) = xt_x
            .svd(true, true)
            .map_err(|e| SklearsError::InvalidInput(format!("SVD failed: {}", e)))?;

        let u = u.unwrap();
        let vt = vt.unwrap();

        // Compute pseudoinverse
        let mut s_inv = Array1::zeros(s.len());
        for (i, &val) in s.iter().enumerate() {
            if val > 1e-10 {
                s_inv[i] = 1.0 / val;
            }
        }

        let s_inv_diag = Array2::from_diag(&s_inv);
        let pinv = vt.t().dot(&s_inv_diag).dot(&u.t());

        let transform = pinv.dot(&xt_y);
        Ok(transform)
    }
}

impl Transform<ArrayView3<'_, f64>, Array3<f64>> for TemporalManifold<TrainedTemporalManifold> {
    fn transform(&self, x: &ArrayView3<'_, f64>) -> SklResult<Array3<f64>> {
        if x.shape()[2] != self.state.n_features {
            return Err(SklearsError::InvalidInput(format!(
                "Expected {} features, got {}",
                self.state.n_features,
                x.shape()[2]
            )));
        }

        let n_timesteps = x.shape()[0];
        let n_samples = x.shape()[1];
        let mut transformed = Array3::zeros((n_timesteps, n_samples, self.state.n_components));

        // Transform each time step
        for t in 0..n_timesteps {
            let time_slice = x.slice(s![t, .., ..]);
            let transform_idx = t.min(self.state.temporal_transforms.len() - 1);
            let transform = &self.state.temporal_transforms[transform_idx];
            let embedded = time_slice.dot(transform);
            transformed.slice_mut(s![t, .., ..]).assign(&embedded);
        }

        Ok(transformed)
    }
}

/// Streaming temporal manifold learning for online processing
///
/// This variant allows for processing streaming data where new time steps
/// arrive continuously and the manifold must be updated incrementally.
#[derive(Debug, Clone)]
pub struct StreamingTemporalManifold<S = Untrained> {
    state: S,
    n_components: usize,
    base_method: String,
    learning_rate: f64,
    forgetting_factor: f64,
    adaptation_threshold: f64,
    random_state: Option<u64>,
}

/// Trained state for StreamingTemporalManifold
#[derive(Debug, Clone)]
pub struct TrainedStreamingTemporalManifold {
    current_embedding: Array2<f64>,
    current_transform: Array2<f64>,
    embedding_buffer: VecDeque<Array2<f64>>,
    update_count: usize,
    n_features: usize,
    n_components: usize,
}

impl Default for StreamingTemporalManifold<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamingTemporalManifold<Untrained> {
    /// Create a new StreamingTemporalManifold instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            n_components: 2,
            base_method: "pca".to_string(),
            learning_rate: 0.01,
            forgetting_factor: 0.95,
            adaptation_threshold: 0.1,
            random_state: None,
        }
    }

    /// Set the number of components
    pub fn n_components(mut self, n_components: usize) -> Self {
        self.n_components = n_components;
        self
    }

    /// Set the base method
    pub fn base_method(mut self, base_method: String) -> Self {
        self.base_method = base_method;
        self
    }

    /// Set the learning rate
    pub fn learning_rate(mut self, learning_rate: f64) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    /// Set the forgetting factor
    pub fn forgetting_factor(mut self, forgetting_factor: f64) -> Self {
        self.forgetting_factor = forgetting_factor;
        self
    }

    /// Set the adaptation threshold
    pub fn adaptation_threshold(mut self, adaptation_threshold: f64) -> Self {
        self.adaptation_threshold = adaptation_threshold;
        self
    }

    /// Set the random state
    pub fn random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }
}

impl Estimator for StreamingTemporalManifold<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, f64>, ArrayView1<'_, ()>> for StreamingTemporalManifold<Untrained> {
    type Fitted = StreamingTemporalManifold<TrainedStreamingTemporalManifold>;

    fn fit(self, x: &ArrayView2<'_, f64>, _y: &ArrayView1<'_, ()>) -> SklResult<Self::Fitted> {
        if x.nrows() == 0 || x.ncols() == 0 {
            return Err(SklearsError::InvalidInput(
                "Input data is empty".to_string(),
            ));
        }

        let n_features = x.ncols();

        // Initialize with simple PCA
        let embedding = self.apply_initial_embedding(x)?;
        let transform = self.compute_initial_transform(x, &embedding)?;

        let mut embedding_buffer = VecDeque::new();
        embedding_buffer.push_back(embedding.clone());

        Ok(StreamingTemporalManifold {
            state: TrainedStreamingTemporalManifold {
                current_embedding: embedding,
                current_transform: transform,
                embedding_buffer,
                update_count: 1,
                n_features,
                n_components: self.n_components,
            },
            n_components: self.n_components,
            base_method: self.base_method,
            learning_rate: self.learning_rate,
            forgetting_factor: self.forgetting_factor,
            adaptation_threshold: self.adaptation_threshold,
            random_state: self.random_state,
        })
    }
}

impl StreamingTemporalManifold<Untrained> {
    fn apply_initial_embedding(&self, x: &ArrayView2<f64>) -> SklResult<Array2<f64>> {
        // Simple PCA for initialization
        let n_samples = x.nrows();
        let mean = x.mean_axis(Axis(0)).unwrap();
        let centered = x - &mean.insert_axis(Axis(0));
        let cov = centered.t().dot(&centered) / (n_samples - 1) as f64;

        let (eigenvalues, eigenvectors) = cov
            .eigh(UPLO::Upper)
            .map_err(|e| SklearsError::InvalidInput(format!("Initial PCA failed: {}", e)))?;

        let mut indices: Vec<usize> = (0..eigenvalues.len()).collect();
        indices.sort_by(|&i, &j| eigenvalues[j].partial_cmp(&eigenvalues[i]).unwrap());

        let mut projection = Array2::zeros((x.ncols(), self.n_components));
        for (i, &idx) in indices.iter().take(self.n_components).enumerate() {
            projection.column_mut(i).assign(&eigenvectors.column(idx));
        }

        Ok(centered.dot(&projection))
    }

    fn compute_initial_transform(
        &self,
        x: &ArrayView2<f64>,
        embedding: &Array2<f64>,
    ) -> SklResult<Array2<f64>> {
        // Compute transformation matrix using least squares
        let xt_x = x.t().dot(x);
        let xt_y = x.t().dot(embedding);

        let (u, s, vt) = xt_x
            .svd(true, true)
            .map_err(|e| SklearsError::InvalidInput(format!("Transform SVD failed: {}", e)))?;

        let u = u.unwrap();
        let vt = vt.unwrap();

        let mut s_inv = Array1::zeros(s.len());
        for (i, &val) in s.iter().enumerate() {
            if val > 1e-10 {
                s_inv[i] = 1.0 / val;
            }
        }

        let s_inv_diag = Array2::from_diag(&s_inv);
        let pinv = vt.t().dot(&s_inv_diag).dot(&u.t());
        Ok(pinv.dot(&xt_y))
    }
}

impl StreamingTemporalManifold<TrainedStreamingTemporalManifold> {
    /// Update the manifold with new streaming data
    pub fn update(&mut self, new_data: &ArrayView2<f64>) -> SklResult<Array2<f64>> {
        if new_data.ncols() != self.state.n_features {
            return Err(SklearsError::InvalidInput(format!(
                "Expected {} features, got {}",
                self.state.n_features,
                new_data.ncols()
            )));
        }

        // Transform new data using current transform
        let new_embedding = new_data.dot(&self.state.current_transform);

        // Update the current embedding to be the new embedding
        // (In streaming learning, we process new data and update our representation)
        self.state.current_embedding = new_embedding.clone();

        // Update buffer
        self.state.embedding_buffer.push_back(new_embedding.clone());
        if self.state.embedding_buffer.len() > 50 {
            self.state.embedding_buffer.pop_front();
        }

        // Update transform if significant change detected
        // Compare statistics of embeddings rather than direct comparison
        let should_update_transform = if self.state.embedding_buffer.len() > 1 {
            let prev_embedding =
                &self.state.embedding_buffer[self.state.embedding_buffer.len() - 2];

            // Compare mean and standard deviation of embeddings
            let new_mean = new_embedding.mean_axis(Axis(0)).unwrap();
            let prev_mean = prev_embedding.mean_axis(Axis(0)).unwrap();
            let mean_change = (&new_mean - &prev_mean).norm_l2();

            let new_std = new_embedding.std_axis(Axis(0), 0.0);
            let prev_std = prev_embedding.std_axis(Axis(0), 0.0);
            let std_change = (&new_std - &prev_std).norm_l2();

            let total_change = mean_change + std_change;
            total_change > self.adaptation_threshold
        } else {
            false
        };

        if should_update_transform {
            // Update transform using exponential moving average
            let alpha = self.learning_rate;
            let new_transform = self.update_transform(new_data, &new_embedding)?;
            self.state.current_transform =
                (1.0 - alpha) * &self.state.current_transform + alpha * &new_transform;
        }

        self.state.update_count += 1;
        Ok(new_embedding)
    }

    fn update_transform(
        &self,
        data: &ArrayView2<f64>,
        embedding: &Array2<f64>,
    ) -> SklResult<Array2<f64>> {
        // Recompute transform using current data and embedding
        let xt_x = data.t().dot(data);
        let xt_y = data.t().dot(embedding);

        let (u, s, vt) = xt_x.svd(true, true).map_err(|e| {
            SklearsError::InvalidInput(format!("Update transform SVD failed: {}", e))
        })?;

        let u = u.unwrap();
        let vt = vt.unwrap();

        let mut s_inv = Array1::zeros(s.len());
        for (i, &val) in s.iter().enumerate() {
            if val > 1e-10 {
                s_inv[i] = 1.0 / val;
            }
        }

        let s_inv_diag = Array2::from_diag(&s_inv);
        let pinv = vt.t().dot(&s_inv_diag).dot(&u.t());
        Ok(pinv.dot(&xt_y))
    }

    /// Get current embedding
    pub fn current_embedding(&self) -> &Array2<f64> {
        &self.state.current_embedding
    }

    /// Get embedding history
    pub fn embedding_history(&self) -> &VecDeque<Array2<f64>> {
        &self.state.embedding_buffer
    }
}

impl Transform<ArrayView2<'_, f64>, Array2<f64>>
    for StreamingTemporalManifold<TrainedStreamingTemporalManifold>
{
    fn transform(&self, x: &ArrayView2<'_, f64>) -> SklResult<Array2<f64>> {
        if x.ncols() != self.state.n_features {
            return Err(SklearsError::InvalidInput(format!(
                "Expected {} features, got {}",
                self.state.n_features,
                x.ncols()
            )));
        }

        let transformed = x.dot(&self.state.current_transform);
        Ok(transformed)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_temporal_manifold_basic() {
        // Create time series data: [time_steps, n_samples, n_features]
        let x = array![
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            [[1.1, 2.1], [3.1, 4.1], [5.1, 6.1]],
            [[1.2, 2.2], [3.2, 4.2], [5.2, 6.2]]
        ];
        let dummy_y = array![(), (), ()];

        let temporal = TemporalManifold::new()
            .n_components(2)
            .window_size(3)
            .temporal_weight(0.5)
            .random_state(42);

        let fitted = temporal.fit(&x.view(), &dummy_y.view()).unwrap();
        let transformed = fitted.transform(&x.view()).unwrap();

        assert_eq!(transformed.shape(), [3, 3, 2]);
        assert!(transformed.iter().all(|&x| x.is_finite()));

        // Check trajectory analysis
        assert_eq!(fitted.state.trajectory_analysis.velocity_profiles.len(), 2);
        assert_eq!(
            fitted.state.trajectory_analysis.acceleration_profiles.len(),
            1
        );
    }

    #[test]
    fn test_streaming_temporal_manifold() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let dummy_y = array![(), (), ()];

        let streaming = StreamingTemporalManifold::new()
            .n_components(2)
            .learning_rate(0.1)
            .random_state(42);

        let mut fitted = streaming.fit(&x.view(), &dummy_y.view()).unwrap();

        // Test streaming update
        let new_data = array![[1.5, 2.5], [3.5, 4.5]];
        let new_embedding = fitted.update(&new_data.view()).unwrap();

        assert_eq!(new_embedding.shape(), [2, 2]);
        assert!(new_embedding.iter().all(|&x| x.is_finite()));
        assert_eq!(fitted.state.update_count, 2);
    }

    #[test]
    fn test_temporal_manifold_different_methods() {
        let x = array![[[1.0, 2.0], [3.0, 4.0]], [[1.1, 2.1], [3.1, 4.1]]];
        let dummy_y = array![(), ()];

        for method in &["pca", "isomap", "lle"] {
            let temporal = TemporalManifold::new()
                .n_components(2)
                .base_method(method.to_string())
                .random_state(42);

            let fitted = temporal.fit(&x.view(), &dummy_y.view()).unwrap();
            let transformed = fitted.transform(&x.view()).unwrap();

            assert_eq!(transformed.shape(), [2, 2, 2]);
            assert!(transformed.iter().all(|&x| x.is_finite()));
        }
    }
}
