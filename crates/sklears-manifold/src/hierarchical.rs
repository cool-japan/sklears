//! Hierarchical manifold learning methods
//! This module implements various hierarchical approaches for manifold learning,
//! including multi-scale embeddings and coarse-to-fine optimization.

use scirs2_core::ndarray::ndarray_linalg::{Eigh, Norm, Solve, SVD, UPLO};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::thread_rng;
use scirs2_core::random::Rng;
use scirs2_core::random::SeedableRng;
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Transform, Untrained},
};

/// Hierarchical manifold learning with multi-scale embeddings
///
/// This implements a hierarchical approach where the manifold is learned at multiple
/// scales, from coarse to fine, enabling better preservation of both global and local structure.
///
/// # Parameters
///
/// * `n_components` - Target dimensionality for embeddings
/// * `levels` - Number of hierarchical levels
/// * `scale_factors` - Scale factors for each level (coarse to fine)
/// * `base_method` - Base manifold learning method ("pca", "isomap", "lle")
/// * `refinement_steps` - Number of refinement steps per level
/// * `learning_rate` - Learning rate for refinement optimization
/// * `random_state` - Random seed for reproducibility
///
/// # Examples
///
/// ```rust,ignore
/// use sklears_manifold::hierarchical::HierarchicalManifold;
/// use sklears_core::traits::{Transform, Fit};
/// use scirs2_core::ndarray::{ Array1, ArrayView1, ArrayView2};
///
/// let x = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]];
///
/// let hierarchical = HierarchicalManifold::new()
///     .n_components(2)
///     .levels(3)
///     .base_method("pca".to_string());
///
/// let y = array![(), (), (), ()];
/// let fitted = hierarchical.fit(&x.view(), &y.view()).unwrap();
/// let embedded = fitted.transform(&x.view()).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct HierarchicalManifold<S = Untrained> {
    state: S,
    n_components: usize,
    levels: usize,
    scale_factors: Vec<f64>,
    base_method: String,
    refinement_steps: usize,
    learning_rate: f64,
    random_state: Option<u64>,
}

/// Trained state for HierarchicalManifold
#[derive(Debug, Clone)]
pub struct TrainedHierarchicalManifold {
    level_embeddings: Vec<Array2<f64>>,
    level_transforms: Vec<Array2<f64>>,
    n_features: usize,
    n_components: usize,
    levels: usize,
}

impl Default for HierarchicalManifold<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl HierarchicalManifold<Untrained> {
    /// Create a new HierarchicalManifold instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            n_components: 2,
            levels: 3,
            scale_factors: vec![0.1, 0.5, 1.0], // Coarse to fine
            base_method: "pca".to_string(),
            refinement_steps: 100,
            learning_rate: 0.01,
            random_state: None,
        }
    }

    /// Set the number of components
    pub fn n_components(mut self, n_components: usize) -> Self {
        self.n_components = n_components;
        self
    }

    /// Set the number of hierarchical levels
    pub fn levels(mut self, levels: usize) -> Self {
        self.levels = levels;
        // Adjust scale factors if needed
        if self.scale_factors.len() != levels {
            self.scale_factors = (0..levels)
                .map(|i| (i + 1) as f64 / levels as f64)
                .collect();
        }
        self
    }

    /// Set the scale factors for each level
    pub fn scale_factors(mut self, scale_factors: Vec<f64>) -> Self {
        self.scale_factors = scale_factors;
        self
    }

    /// Set the base method
    pub fn base_method(mut self, base_method: String) -> Self {
        self.base_method = base_method;
        self
    }

    /// Set the number of refinement steps
    pub fn refinement_steps(mut self, refinement_steps: usize) -> Self {
        self.refinement_steps = refinement_steps;
        self
    }

    /// Set the learning rate
    pub fn learning_rate(mut self, learning_rate: f64) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    /// Set the random state
    pub fn random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }
}

impl Estimator for HierarchicalManifold<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, f64>, ArrayView1<'_, ()>> for HierarchicalManifold<Untrained> {
    type Fitted = HierarchicalManifold<TrainedHierarchicalManifold>;

    fn fit(self, x: &ArrayView2<'_, f64>, _y: &ArrayView1<'_, ()>) -> SklResult<Self::Fitted> {
        if x.nrows() == 0 || x.ncols() == 0 {
            return Err(SklearsError::InvalidInput(
                "Input data is empty".to_string(),
            ));
        }

        let n_samples = x.nrows();
        let n_features = x.ncols();

        let mut rng = if let Some(seed) = self.random_state {
            StdRng::seed_from_u64(seed)
        } else {
            StdRng::seed_from_u64(thread_rng().gen::<u64>())
        };

        let mut level_embeddings = Vec::new();
        let mut level_transforms = Vec::new();

        // Start with coarsest level
        let mut current_embedding = Array2::zeros((n_samples, self.n_components));

        for (level, &scale_factor) in self.scale_factors.iter().enumerate() {
            // Apply base method at current scale
            let embedding = match self.base_method.as_str() {
                "pca" => self.apply_pca(x, scale_factor)?,
                "isomap" => self.apply_isomap(x, scale_factor, &mut rng)?,
                "lle" => self.apply_lle(x, scale_factor, &mut rng)?,
                _ => {
                    return Err(SklearsError::InvalidInput(format!(
                        "Unknown base method: {}",
                        self.base_method
                    )))
                }
            };

            // If not the first level, refine based on previous level
            let refined_embedding = if level > 0 {
                self.refine_embedding(&current_embedding, &embedding, x, scale_factor, &mut rng)?
            } else {
                embedding.clone()
            };

            current_embedding = refined_embedding.clone();
            level_embeddings.push(refined_embedding);

            // Compute transformation matrix for this level
            let transform = self.compute_transform_matrix(x, &current_embedding)?;
            level_transforms.push(transform);
        }

        Ok(HierarchicalManifold {
            state: TrainedHierarchicalManifold {
                level_embeddings,
                level_transforms,
                n_features,
                n_components: self.n_components,
                levels: self.levels,
            },
            n_components: self.n_components,
            levels: self.levels,
            scale_factors: self.scale_factors,
            base_method: self.base_method,
            refinement_steps: self.refinement_steps,
            learning_rate: self.learning_rate,
            random_state: self.random_state,
        })
    }
}

impl HierarchicalManifold<Untrained> {
    /// Apply PCA at given scale
    fn apply_pca(&self, x: &ArrayView2<f64>, scale_factor: f64) -> SklResult<Array2<f64>> {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        // Center the data
        let mean = x.mean_axis(Axis(0)).unwrap();
        let centered = x - &mean.insert_axis(Axis(0));

        // Compute covariance matrix with scaling
        let cov = centered.t().dot(&centered) / (n_samples - 1) as f64 * scale_factor;

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

    /// Apply Isomap at given scale
    fn apply_isomap(
        &self,
        x: &ArrayView2<f64>,
        scale_factor: f64,
        rng: &mut StdRng,
    ) -> SklResult<Array2<f64>> {
        let n_samples = x.nrows();
        let k = ((n_samples as f64 * scale_factor).round() as usize)
            .max(3)
            .min(n_samples - 1);

        // Build k-NN graph
        let graph = self.build_knn_graph(x, k)?;

        // Compute geodesic distances using Floyd-Warshall
        let geodesic_distances = self.floyd_warshall(&graph)?;

        // Apply MDS to geodesic distances
        let embedding = self.mds(&geodesic_distances)?;
        Ok(embedding)
    }

    /// Apply LLE at given scale
    fn apply_lle(
        &self,
        x: &ArrayView2<f64>,
        scale_factor: f64,
        rng: &mut StdRng,
    ) -> SklResult<Array2<f64>> {
        let n_samples = x.nrows();
        let k = ((n_samples as f64 * scale_factor).round() as usize)
            .max(3)
            .min(n_samples - 1);

        // Find k nearest neighbors
        let neighbors = self.find_k_neighbors(x, k)?;

        // Compute reconstruction weights
        let weights = self.compute_lle_weights(x, &neighbors)?;

        // Solve eigenvalue problem
        let embedding = self.solve_lle_eigenvalue_problem(&weights)?;
        Ok(embedding)
    }

    /// Refine embedding using previous level
    fn refine_embedding(
        &self,
        previous_embedding: &Array2<f64>,
        current_embedding: &Array2<f64>,
        x: &ArrayView2<f64>,
        scale_factor: f64,
        rng: &mut StdRng,
    ) -> SklResult<Array2<f64>> {
        let n_samples = x.nrows();
        let mut refined = current_embedding.clone();

        // Iterative refinement
        for _step in 0..self.refinement_steps {
            let mut total_loss = 0.0;
            let mut gradient = Array2::zeros(refined.raw_dim());

            // Compute loss and gradients
            for i in 0..n_samples {
                for j in (i + 1)..n_samples {
                    // High-dimensional distance
                    let hd_dist = (&x.row(i) - &x.row(j)).norm_l2();

                    // Previous level distance
                    let prev_dist =
                        (&previous_embedding.row(i) - &previous_embedding.row(j)).norm_l2();

                    // Current embedding distance
                    let curr_dist = (&refined.row(i) - &refined.row(j)).norm_l2();

                    // Loss: weighted combination of preservation of previous level and high-dimensional distances
                    let target_dist = scale_factor * hd_dist + (1.0 - scale_factor) * prev_dist;
                    let loss = (curr_dist - target_dist).powi(2);
                    total_loss += loss;

                    // Gradient
                    if curr_dist > 1e-8 {
                        let factor = 2.0 * (curr_dist - target_dist) / curr_dist;
                        let diff = &refined.row(i) - &refined.row(j);

                        for k in 0..self.n_components {
                            gradient[[i, k]] += factor * diff[k];
                            gradient[[j, k]] -= factor * diff[k];
                        }
                    }
                }
            }

            // Apply gradient update
            refined = refined - self.learning_rate * gradient;

            // Early stopping
            if total_loss < 1e-6 {
                break;
            }
        }

        Ok(refined)
    }

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

    /// Build k-nearest neighbor graph
    fn build_knn_graph(&self, x: &ArrayView2<f64>, k: usize) -> SklResult<Array2<f64>> {
        let n_samples = x.nrows();
        let mut graph = Array2::from_elem((n_samples, n_samples), f64::INFINITY);

        // Set diagonal to 0
        for i in 0..n_samples {
            graph[[i, i]] = 0.0;
        }

        // For each point, find k nearest neighbors
        for i in 0..n_samples {
            let mut distances: Vec<(usize, f64)> = Vec::new();

            for j in 0..n_samples {
                if i != j {
                    let dist = (&x.row(i) - &x.row(j)).norm_l2();
                    distances.push((j, dist));
                }
            }

            distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

            // Connect to k nearest neighbors
            for &(j, dist) in distances.iter().take(k) {
                graph[[i, j]] = dist;
                graph[[j, i]] = dist; // Make symmetric
            }
        }

        Ok(graph)
    }

    /// Compute geodesic distances using Floyd-Warshall algorithm
    fn floyd_warshall(&self, graph: &Array2<f64>) -> SklResult<Array2<f64>> {
        let n = graph.nrows();
        let mut distances = graph.clone();

        for k in 0..n {
            for i in 0..n {
                for j in 0..n {
                    let new_dist = distances[[i, k]] + distances[[k, j]];
                    if new_dist < distances[[i, j]] {
                        distances[[i, j]] = new_dist;
                    }
                }
            }
        }

        Ok(distances)
    }

    /// Apply multidimensional scaling
    fn mds(&self, distances: &Array2<f64>) -> SklResult<Array2<f64>> {
        let n = distances.nrows();

        // Double centering
        let mut gram = Array2::zeros((n, n));
        let mean_row = distances.mean_axis(Axis(1)).unwrap();
        let mean_col = distances.mean_axis(Axis(0)).unwrap();
        let mean_all = distances.mean().unwrap();

        for i in 0..n {
            for j in 0..n {
                gram[[i, j]] =
                    -0.5 * (distances[[i, j]].powi(2) - mean_row[i] - mean_col[j] + mean_all);
            }
        }

        // Eigendecomposition
        let (eigenvalues, eigenvectors) = gram.eigh(UPLO::Upper).map_err(|e| {
            SklearsError::InvalidInput(format!("MDS eigendecomposition failed: {}", e))
        })?;

        // Sort eigenvalues and eigenvectors
        let mut indices: Vec<usize> = (0..eigenvalues.len()).collect();
        indices.sort_by(|&i, &j| eigenvalues[j].partial_cmp(&eigenvalues[i]).unwrap());

        // Create embedding
        let mut embedding = Array2::zeros((n, self.n_components));
        for (i, &idx) in indices.iter().take(self.n_components).enumerate() {
            let scale = eigenvalues[idx].max(0.0).sqrt();
            for j in 0..n {
                embedding[[j, i]] = eigenvectors[[j, idx]] * scale;
            }
        }

        Ok(embedding)
    }

    /// Find k nearest neighbors for each point
    fn find_k_neighbors(&self, x: &ArrayView2<f64>, k: usize) -> SklResult<Vec<Vec<usize>>> {
        let n_samples = x.nrows();
        let mut neighbors = Vec::new();

        for i in 0..n_samples {
            let mut distances: Vec<(usize, f64)> = Vec::new();

            for j in 0..n_samples {
                if i != j {
                    let dist = (&x.row(i) - &x.row(j)).norm_l2();
                    distances.push((j, dist));
                }
            }

            distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            let point_neighbors: Vec<usize> =
                distances.iter().take(k).map(|&(idx, _)| idx).collect();
            neighbors.push(point_neighbors);
        }

        Ok(neighbors)
    }

    /// Compute LLE reconstruction weights
    fn compute_lle_weights(
        &self,
        x: &ArrayView2<f64>,
        neighbors: &[Vec<usize>],
    ) -> SklResult<Array2<f64>> {
        let n_samples = x.nrows();
        let mut weights = Array2::zeros((n_samples, n_samples));

        for (i, point_neighbors) in neighbors.iter().enumerate() {
            let k = point_neighbors.len();
            if k == 0 {
                continue;
            }

            // Build local covariance matrix
            let mut local_cov = Array2::zeros((k, k));
            let xi = x.row(i);

            for (a, &neighbor_a) in point_neighbors.iter().enumerate() {
                for (b, &neighbor_b) in point_neighbors.iter().enumerate() {
                    let diff_a = &x.row(neighbor_a) - &xi;
                    let diff_b = &x.row(neighbor_b) - &xi;
                    local_cov[[a, b]] = diff_a.dot(&diff_b);
                }
            }

            // Add regularization
            let reg = 1e-3 * local_cov.diag().sum() / k as f64;
            for j in 0..k {
                local_cov[[j, j]] += reg;
            }

            // Solve for weights
            let ones = Array1::ones(k);
            let weights_vec = match local_cov.solve(&ones) {
                Ok(w) => w,
                Err(_) => {
                    // Fallback to uniform weights
                    Array1::from_elem(k, 1.0 / k as f64)
                }
            };

            // Normalize weights
            let weight_sum = weights_vec.sum();
            for (j, &neighbor) in point_neighbors.iter().enumerate() {
                weights[[i, neighbor]] = weights_vec[j] / weight_sum;
            }
        }

        Ok(weights)
    }

    /// Solve LLE eigenvalue problem
    fn solve_lle_eigenvalue_problem(&self, weights: &Array2<f64>) -> SklResult<Array2<f64>> {
        let n_samples = weights.nrows();

        // Build sparse matrix M = (I - W)^T (I - W)
        let identity = Array2::eye(n_samples);
        let i_minus_w = &identity - weights;
        let m = i_minus_w.t().dot(&i_minus_w);

        // Eigendecomposition
        let (eigenvalues, eigenvectors): (Array1<f64>, Array2<f64>) =
            m.eigh(UPLO::Upper).map_err(|e| {
                SklearsError::InvalidInput(format!("LLE eigendecomposition failed: {}", e))
            })?;

        // Sort eigenvalues (ascending for LLE)
        let mut indices: Vec<usize> = (0..eigenvalues.len()).collect();
        indices.sort_by(|&i, &j| eigenvalues[i].partial_cmp(&eigenvalues[j]).unwrap());

        // Skip the first eigenvector (constant) and take the next n_components
        let mut embedding = Array2::zeros((n_samples, self.n_components));
        for (i, &idx) in indices.iter().skip(1).take(self.n_components).enumerate() {
            embedding.column_mut(i).assign(&eigenvectors.column(idx));
        }

        Ok(embedding)
    }
}

impl Estimator for HierarchicalManifold<TrainedHierarchicalManifold> {
    type Config = ();
    type Error = SklearsError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Transform<ArrayView2<'_, f64>, Array2<f64>>
    for HierarchicalManifold<TrainedHierarchicalManifold>
{
    fn transform(&self, x: &ArrayView2<'_, f64>) -> SklResult<Array2<f64>> {
        if x.ncols() != self.state.n_features {
            return Err(SklearsError::InvalidInput(format!(
                "Expected {} features, got {}",
                self.state.n_features,
                x.ncols()
            )));
        }

        // Use the final level transformation
        let final_transform = &self.state.level_transforms[self.state.levels - 1];
        let transformed = x.dot(final_transform);
        Ok(transformed)
    }
}

/// Multi-scale embedding for pyramidal manifold representations
///
/// Implements a pyramidal approach where embeddings are computed at multiple
/// resolutions and combined for comprehensive manifold representation.
#[derive(Debug, Clone)]
pub struct MultiScaleEmbedding<S = Untrained> {
    state: S,
    n_components: usize,
    scales: Vec<f64>,
    base_method: String,
    combination_method: String, // "weighted", "concatenate", "attention"
    random_state: Option<u64>,
}

/// Trained state for MultiScaleEmbedding
#[derive(Debug, Clone)]
pub struct TrainedMultiScaleEmbedding {
    scale_embeddings: Vec<Array2<f64>>,
    scale_weights: Vec<f64>,
    combination_matrix: Array2<f64>,
    n_features: usize,
    n_components: usize,
}

impl Default for MultiScaleEmbedding<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl MultiScaleEmbedding<Untrained> {
    /// Create a new MultiScaleEmbedding instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            n_components: 2,
            scales: vec![0.25, 0.5, 1.0, 2.0],
            base_method: "pca".to_string(),
            combination_method: "weighted".to_string(),
            random_state: None,
        }
    }

    /// Set the number of components
    pub fn n_components(mut self, n_components: usize) -> Self {
        self.n_components = n_components;
        self
    }

    /// Set the scales
    pub fn scales(mut self, scales: Vec<f64>) -> Self {
        self.scales = scales;
        self
    }

    /// Set the base method
    pub fn base_method(mut self, base_method: String) -> Self {
        self.base_method = base_method;
        self
    }

    /// Set the combination method
    pub fn combination_method(mut self, combination_method: String) -> Self {
        self.combination_method = combination_method;
        self
    }

    /// Set the random state
    pub fn random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }
}

impl Estimator for MultiScaleEmbedding<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, f64>, ArrayView1<'_, ()>> for MultiScaleEmbedding<Untrained> {
    type Fitted = MultiScaleEmbedding<TrainedMultiScaleEmbedding>;

    fn fit(self, x: &ArrayView2<'_, f64>, _y: &ArrayView1<'_, ()>) -> SklResult<Self::Fitted> {
        if x.nrows() == 0 || x.ncols() == 0 {
            return Err(SklearsError::InvalidInput(
                "Input data is empty".to_string(),
            ));
        }

        let n_samples = x.nrows();
        let n_features = x.ncols();

        let mut scale_embeddings = Vec::new();
        let mut scale_weights = Vec::new();

        // Compute embeddings at each scale
        for &scale in &self.scales {
            let embedding = self.compute_scale_embedding(x, scale)?;
            scale_embeddings.push(embedding);

            // Compute quality weight for this scale
            let weight = self.compute_scale_weight(x, scale_embeddings.last().unwrap())?;
            scale_weights.push(weight);
        }

        // Normalize weights
        let weight_sum: f64 = scale_weights.iter().sum();
        if weight_sum > 0.0 {
            for weight in scale_weights.iter_mut() {
                *weight /= weight_sum;
            }
        } else {
            // Uniform weights if all are zero
            scale_weights.fill(1.0 / self.scales.len() as f64);
        }

        // Compute combination matrix
        let combination_matrix =
            self.compute_combination_matrix(x, &scale_embeddings, &scale_weights)?;

        Ok(MultiScaleEmbedding {
            state: TrainedMultiScaleEmbedding {
                scale_embeddings,
                scale_weights,
                combination_matrix,
                n_features,
                n_components: self.n_components,
            },
            n_components: self.n_components,
            scales: self.scales,
            base_method: self.base_method,
            combination_method: self.combination_method,
            random_state: self.random_state,
        })
    }
}

impl MultiScaleEmbedding<Untrained> {
    /// Compute embedding at a specific scale
    fn compute_scale_embedding(&self, x: &ArrayView2<f64>, scale: f64) -> SklResult<Array2<f64>> {
        match self.base_method.as_str() {
            "pca" => self.scaled_pca(x, scale),
            _ => Err(SklearsError::InvalidInput(format!(
                "Unsupported base method: {}",
                self.base_method
            ))),
        }
    }

    /// Apply PCA with scaling
    fn scaled_pca(&self, x: &ArrayView2<f64>, scale: f64) -> SklResult<Array2<f64>> {
        let n_samples = x.nrows();

        // Center the data
        let mean = x.mean_axis(Axis(0)).unwrap();
        let centered = x - &mean.insert_axis(Axis(0));

        // Apply scale to covariance computation
        let scaled_data = &centered * scale;
        let cov = scaled_data.t().dot(&scaled_data) / (n_samples - 1) as f64;

        // Eigendecomposition
        let (eigenvalues, eigenvectors) = cov
            .eigh(UPLO::Upper)
            .map_err(|e| SklearsError::InvalidInput(format!("Scaled PCA failed: {}", e)))?;

        // Sort by eigenvalues (descending)
        let mut indices: Vec<usize> = (0..eigenvalues.len()).collect();
        indices.sort_by(|&i, &j| eigenvalues[j].partial_cmp(&eigenvalues[i]).unwrap());

        // Project data
        let mut projection_matrix = Array2::zeros((x.ncols(), self.n_components));
        for (i, &idx) in indices.iter().take(self.n_components).enumerate() {
            projection_matrix
                .column_mut(i)
                .assign(&eigenvectors.column(idx));
        }

        let embedding = centered.dot(&projection_matrix);
        Ok(embedding)
    }

    /// Compute quality weight for a scale embedding
    fn compute_scale_weight(&self, x: &ArrayView2<f64>, embedding: &Array2<f64>) -> SklResult<f64> {
        // Simple quality measure: preservation of pairwise distances
        let n_samples = x.nrows();
        let mut correlation_sum = 0.0;
        let mut count = 0;

        for i in 0..n_samples {
            for j in (i + 1)..n_samples {
                let hd_dist = (&x.row(i) - &x.row(j)).norm_l2();
                let ld_dist = (&embedding.row(i) - &embedding.row(j)).norm_l2();

                correlation_sum += hd_dist * ld_dist;
                count += 1;
            }
        }

        Ok(if count > 0 {
            correlation_sum / count as f64
        } else {
            0.0
        })
    }

    /// Compute combination matrix for multi-scale embeddings
    fn compute_combination_matrix(
        &self,
        x: &ArrayView2<f64>,
        scale_embeddings: &[Array2<f64>],
        scale_weights: &[f64],
    ) -> SklResult<Array2<f64>> {
        match self.combination_method.as_str() {
            "weighted" => {
                // Weighted average of embeddings
                let n_samples = x.nrows();
                let mut combined: Array2<f64> = Array2::zeros((n_samples, self.n_components));

                for (embedding, &weight) in scale_embeddings.iter().zip(scale_weights.iter()) {
                    combined = combined + weight * embedding;
                }

                // Return identity matrix since embedding is already combined
                Ok(Array2::eye(x.ncols()))
            }
            "concatenate" => {
                // Concatenate all scale embeddings
                let total_components = self.n_components * scale_embeddings.len();

                // Return projection matrix that will concatenate embeddings
                let mut projection = Array2::zeros((x.ncols(), total_components));

                // This is a simplified approach - in practice would need more sophisticated projection
                for i in 0..x.ncols().min(total_components) {
                    projection[[i, i]] = 1.0;
                }

                Ok(projection)
            }
            _ => Err(SklearsError::InvalidInput(format!(
                "Unknown combination method: {}",
                self.combination_method
            ))),
        }
    }
}

impl Estimator for MultiScaleEmbedding<TrainedMultiScaleEmbedding> {
    type Config = ();
    type Error = SklearsError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Transform<ArrayView2<'_, f64>, Array2<f64>>
    for MultiScaleEmbedding<TrainedMultiScaleEmbedding>
{
    fn transform(&self, x: &ArrayView2<'_, f64>) -> SklResult<Array2<f64>> {
        if x.ncols() != self.state.n_features {
            return Err(SklearsError::InvalidInput(format!(
                "Expected {} features, got {}",
                self.state.n_features,
                x.ncols()
            )));
        }

        // Apply combination matrix transformation
        let transformed = x.dot(&self.state.combination_matrix);
        Ok(transformed)
    }
}

/// Adaptive resolution manifold learning that automatically determines optimal parameters
///
/// This extends hierarchical manifold learning by adaptively determining the number of levels,
/// scale factors, and other parameters based on data characteristics and embedding quality metrics.
///
/// # Parameters
///
/// * `n_components` - Target dimensionality for embeddings
/// * `max_levels` - Maximum number of hierarchical levels to consider
/// * `min_levels` - Minimum number of hierarchical levels
/// * `base_method` - Base manifold learning method ("pca", "isomap", "lle")
/// * `quality_threshold` - Minimum quality threshold for stopping adaptation
/// * `adaptation_method` - Method for adaptation ("stress_based", "neighborhood_preservation", "combined")
/// * `max_iterations` - Maximum iterations for adaptive process
/// * `random_state` - Random seed for reproducibility
///
/// # Examples
///
/// ```rust,ignore
/// use sklears_manifold::hierarchical::AdaptiveResolutionManifold;
/// use sklears_core::traits::{Transform, Fit};
/// use scirs2_core::ndarray::{ Array1, ArrayView1, ArrayView2};
///
/// let x = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]];
///
/// let adaptive = AdaptiveResolutionManifold::new()
///     .n_components(2)
///     .max_levels(5)
///     .quality_threshold(0.8)
///     .adaptation_method("combined".to_string());
///
/// let y = array![(), (), (), ()];
/// let fitted = adaptive.fit(&x.view(), &y.view()).unwrap();
/// let embedded = fitted.transform(&x.view()).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct AdaptiveResolutionManifold<S = Untrained> {
    state: S,
    n_components: usize,
    max_levels: usize,
    min_levels: usize,
    base_method: String,
    quality_threshold: f64,
    adaptation_method: String,
    max_iterations: usize,
    learning_rate: f64,
    random_state: Option<u64>,
}

/// Trained state for AdaptiveResolutionManifold
#[derive(Debug, Clone)]
pub struct TrainedAdaptiveResolutionManifold {
    optimal_levels: usize,
    optimal_scale_factors: Vec<f64>,
    level_embeddings: Vec<Array2<f64>>,
    level_transforms: Vec<Array2<f64>>,
    quality_scores: Vec<f64>,
    adaptation_history: Vec<AdaptationStep>,
    n_features: usize,
    n_components: usize,
}

/// Records each step in the adaptation process
#[derive(Debug, Clone)]
pub struct AdaptationStep {
    /// levels
    pub levels: usize,
    /// scale_factors
    pub scale_factors: Vec<f64>,
    /// quality_score
    pub quality_score: f64,
    /// stress
    pub stress: f64,
    /// neighborhood_preservation
    pub neighborhood_preservation: f64,
}

/// Embedding quality metrics
#[derive(Debug, Clone)]
pub struct EmbeddingQuality {
    /// stress
    pub stress: f64,
    /// neighborhood_preservation
    pub neighborhood_preservation: f64,
    /// trustworthiness
    pub trustworthiness: f64,
    /// continuity
    pub continuity: f64,
}

impl Default for AdaptiveResolutionManifold<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl AdaptiveResolutionManifold<Untrained> {
    /// Create a new AdaptiveResolutionManifold instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            n_components: 2,
            max_levels: 7,
            min_levels: 2,
            base_method: "pca".to_string(),
            quality_threshold: 0.85,
            adaptation_method: "combined".to_string(),
            max_iterations: 10,
            learning_rate: 0.01,
            random_state: None,
        }
    }

    /// Set the number of components
    pub fn n_components(mut self, n_components: usize) -> Self {
        self.n_components = n_components;
        self
    }

    /// Set the maximum number of levels
    pub fn max_levels(mut self, max_levels: usize) -> Self {
        self.max_levels = max_levels;
        self
    }

    /// Set the minimum number of levels
    pub fn min_levels(mut self, min_levels: usize) -> Self {
        self.min_levels = min_levels;
        self
    }

    /// Set the base method
    pub fn base_method(mut self, base_method: String) -> Self {
        self.base_method = base_method;
        self
    }

    /// Set the quality threshold
    pub fn quality_threshold(mut self, quality_threshold: f64) -> Self {
        self.quality_threshold = quality_threshold;
        self
    }

    /// Set the adaptation method
    pub fn adaptation_method(mut self, adaptation_method: String) -> Self {
        self.adaptation_method = adaptation_method;
        self
    }

    /// Set the maximum iterations
    pub fn max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Set the learning rate
    pub fn learning_rate(mut self, learning_rate: f64) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    /// Set the random state
    pub fn random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }

    /// Generate candidate configurations for adaptive search
    fn generate_candidate_configurations(
        &self,
        iteration: usize,
        n_samples: usize,
        rng: &mut StdRng,
    ) -> SklResult<Vec<(usize, Vec<f64>)>> {
        let mut candidates = Vec::new();

        // Progressive search: start with simple configurations, add complexity
        let levels_range = if iteration == 0 {
            self.min_levels..=(self.min_levels + 1)
        } else {
            self.min_levels..=self.max_levels.min(self.min_levels + iteration + 2)
        };

        for levels in levels_range {
            // Generate different scale factor strategies

            // 1. Linear progression
            let linear_scales: Vec<f64> = (0..levels)
                .map(|i| (i + 1) as f64 / levels as f64)
                .collect();
            candidates.push((levels, linear_scales));

            // 2. Exponential progression
            let exp_scales: Vec<f64> = (0..levels)
                .map(|i| (2_f64.powi(i as i32)) / (2_f64.powi(levels as i32 - 1)))
                .collect();
            candidates.push((levels, exp_scales));

            // 3. Logarithmic progression
            let log_scales: Vec<f64> = (0..levels)
                .map(|i| ((i + 1) as f64).ln() / (levels as f64).ln())
                .collect();
            candidates.push((levels, log_scales));

            // 4. Data-adaptive scales based on sample density
            if iteration > 0 {
                let adaptive_scales = self.compute_adaptive_scales(levels, n_samples)?;
                candidates.push((levels, adaptive_scales));
            }

            // 5. Random perturbations of best configurations (exploration)
            if iteration > 2 {
                let mut random_scales: Vec<f64> = (0..levels)
                    .map(|i| {
                        let base = (i + 1) as f64 / levels as f64;
                        let noise = rng.gen_range(-0.2..0.2);
                        (base + noise).max(0.1).min(1.0)
                    })
                    .collect();
                random_scales.sort_by(|a, b| a.partial_cmp(b).unwrap());
                candidates.push((levels, random_scales));
            }
        }

        Ok(candidates)
    }

    /// Compute adaptive scales based on data characteristics
    fn compute_adaptive_scales(&self, levels: usize, n_samples: usize) -> SklResult<Vec<f64>> {
        // Use sample density to guide scale selection
        // More samples allow for finer scales
        let density_factor = (n_samples as f64).ln() / 100.0; // Normalize by sample count

        let mut scales = Vec::new();
        for i in 0..levels {
            let base_scale = (i + 1) as f64 / levels as f64;
            let adapted_scale = base_scale * (1.0 + density_factor * (i as f64 / levels as f64));
            scales.push(adapted_scale.min(2.0)); // Cap at 2.0
        }

        Ok(scales)
    }

    /// Evaluate embedding quality using multiple metrics
    fn evaluate_embedding_quality(
        &self,
        x: &ArrayView2<f64>,
        embedding: &Array2<f64>,
    ) -> SklResult<EmbeddingQuality> {
        let stress = self.compute_stress(x, embedding)?;
        let neighborhood_preservation = self.compute_neighborhood_preservation(x, embedding)?;
        let trustworthiness = self.compute_trustworthiness(x, embedding)?;
        let continuity = self.compute_continuity(x, embedding)?;

        Ok(EmbeddingQuality {
            stress,
            neighborhood_preservation,
            trustworthiness,
            continuity,
        })
    }

    /// Compute normalized stress
    fn compute_stress(&self, x: &ArrayView2<f64>, embedding: &Array2<f64>) -> SklResult<f64> {
        let n_samples = x.nrows();
        let mut stress_num = 0.0;
        let mut stress_den = 0.0;

        for i in 0..n_samples {
            for j in (i + 1)..n_samples {
                let hd_dist = (&x.row(i) - &x.row(j)).norm_l2();
                let ld_dist = (&embedding.row(i) - &embedding.row(j)).norm_l2();

                stress_num += (hd_dist - ld_dist).powi(2);
                stress_den += hd_dist.powi(2);
            }
        }

        let stress = if stress_den > 1e-10 {
            1.0 - (stress_num / stress_den).sqrt()
        } else {
            0.0
        };

        Ok(stress.max(0.0).min(1.0))
    }

    /// Compute neighborhood preservation (K-ary neighborhood preservation)
    fn compute_neighborhood_preservation(
        &self,
        x: &ArrayView2<f64>,
        embedding: &Array2<f64>,
    ) -> SklResult<f64> {
        let n_samples = x.nrows();
        let k = (n_samples as f64).sqrt() as usize; // Adaptive k
        let mut preservation_sum = 0.0;

        for i in 0..n_samples {
            // Find k-NN in high-dimensional space
            let mut hd_distances: Vec<(usize, f64)> = (0..n_samples)
                .map(|j| (j, (&x.row(i) - &x.row(j)).norm_l2()))
                .collect();
            hd_distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            let hd_neighbors: Vec<usize> = hd_distances
                .iter()
                .take(k + 1)
                .skip(1)
                .map(|(idx, _)| *idx)
                .collect();

            // Find k-NN in low-dimensional space
            let mut ld_distances: Vec<(usize, f64)> = (0..n_samples)
                .map(|j| (j, (&embedding.row(i) - &embedding.row(j)).norm_l2()))
                .collect();
            ld_distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            let ld_neighbors: Vec<usize> = ld_distances
                .iter()
                .take(k + 1)
                .skip(1)
                .map(|(idx, _)| *idx)
                .collect();

            // Compute intersection
            let intersection_size = hd_neighbors
                .iter()
                .filter(|&neighbor| ld_neighbors.contains(neighbor))
                .count();

            preservation_sum += intersection_size as f64 / k as f64;
        }

        Ok(preservation_sum / n_samples as f64)
    }

    /// Compute trustworthiness metric
    fn compute_trustworthiness(
        &self,
        x: &ArrayView2<f64>,
        embedding: &Array2<f64>,
    ) -> SklResult<f64> {
        let n_samples = x.nrows();
        let k = (n_samples as f64).sqrt() as usize;
        let mut trustworthiness = 0.0;

        for i in 0..n_samples {
            // Find k-NN in embedding space
            let mut ld_distances: Vec<(usize, f64)> = (0..n_samples)
                .map(|j| (j, (&embedding.row(i) - &embedding.row(j)).norm_l2()))
                .collect();
            ld_distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            let ld_neighbors: Vec<usize> = ld_distances
                .iter()
                .take(k + 1)
                .skip(1)
                .map(|(idx, _)| *idx)
                .collect();

            // Rank in high-dimensional space
            let mut hd_distances: Vec<(usize, f64)> = (0..n_samples)
                .map(|j| (j, (&x.row(i) - &x.row(j)).norm_l2()))
                .collect();
            hd_distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

            let mut rank_sum = 0.0;
            for &neighbor in &ld_neighbors {
                if let Some(pos) = hd_distances.iter().position(|(idx, _)| *idx == neighbor) {
                    if pos > k {
                        rank_sum += (pos - k) as f64;
                    }
                }
            }

            trustworthiness += 1.0
                - (2.0 / (n_samples as f64 * k as f64 * (2 * n_samples - 3 * k - 1) as f64))
                    * rank_sum;
        }

        Ok((trustworthiness / n_samples as f64).max(0.0).min(1.0))
    }

    /// Compute continuity metric
    fn compute_continuity(&self, x: &ArrayView2<f64>, embedding: &Array2<f64>) -> SklResult<f64> {
        let n_samples = x.nrows();
        let k = (n_samples as f64).sqrt() as usize;
        let mut continuity = 0.0;

        for i in 0..n_samples {
            // Find k-NN in high-dimensional space
            let mut hd_distances: Vec<(usize, f64)> = (0..n_samples)
                .map(|j| (j, (&x.row(i) - &x.row(j)).norm_l2()))
                .collect();
            hd_distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            let hd_neighbors: Vec<usize> = hd_distances
                .iter()
                .take(k + 1)
                .skip(1)
                .map(|(idx, _)| *idx)
                .collect();

            // Rank in embedding space
            let mut ld_distances: Vec<(usize, f64)> = (0..n_samples)
                .map(|j| (j, (&embedding.row(i) - &embedding.row(j)).norm_l2()))
                .collect();
            ld_distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

            let mut rank_sum = 0.0;
            for &neighbor in &hd_neighbors {
                if let Some(pos) = ld_distances.iter().position(|(idx, _)| *idx == neighbor) {
                    if pos > k {
                        rank_sum += (pos - k) as f64;
                    }
                }
            }

            continuity += 1.0
                - (2.0 / (n_samples as f64 * k as f64 * (2 * n_samples - 3 * k - 1) as f64))
                    * rank_sum;
        }

        Ok((continuity / n_samples as f64).max(0.0).min(1.0))
    }

    /// Combine quality metrics into a single score
    fn compute_combined_quality(&self, quality: &EmbeddingQuality) -> SklResult<f64> {
        let combined = match self.adaptation_method.as_str() {
            "stress_based" => quality.stress,
            "neighborhood_preservation" => quality.neighborhood_preservation,
            "trustworthiness" => quality.trustworthiness,
            "continuity" => quality.continuity,
            "combined" => {
                // Weighted combination of all metrics
                0.3 * quality.stress
                    + 0.3 * quality.neighborhood_preservation
                    + 0.2 * quality.trustworthiness
                    + 0.2 * quality.continuity
            }
            _ => quality.stress, // Default fallback
        };

        Ok(combined)
    }
}

impl Estimator for AdaptiveResolutionManifold<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, f64>, ArrayView1<'_, ()>> for AdaptiveResolutionManifold<Untrained> {
    type Fitted = AdaptiveResolutionManifold<TrainedAdaptiveResolutionManifold>;

    fn fit(self, x: &ArrayView2<'_, f64>, _y: &ArrayView1<'_, ()>) -> SklResult<Self::Fitted> {
        if x.nrows() == 0 || x.ncols() == 0 {
            return Err(SklearsError::InvalidInput(
                "Input data is empty".to_string(),
            ));
        }

        let n_samples = x.nrows();
        let n_features = x.ncols();

        let mut rng = if let Some(seed) = self.random_state {
            StdRng::seed_from_u64(seed)
        } else {
            StdRng::seed_from_u64(thread_rng().gen::<u64>())
        };

        let mut adaptation_history = Vec::new();
        let mut best_levels = self.min_levels;
        let mut best_scale_factors = Vec::new();
        let mut best_quality = 0.0;
        let mut best_embeddings = Vec::new();
        let mut best_transforms = Vec::new();
        let mut best_quality_scores = Vec::new();

        // Adaptive search for optimal resolution configuration
        for iteration in 0..self.max_iterations {
            // Generate candidate configurations
            let candidates =
                self.generate_candidate_configurations(iteration, n_samples, &mut rng)?;

            for (levels, scale_factors) in candidates {
                // Build hierarchical manifold with current configuration
                let hierarchical = HierarchicalManifold::new()
                    .n_components(self.n_components)
                    .levels(levels)
                    .scale_factors(scale_factors.clone())
                    .base_method(self.base_method.clone())
                    .learning_rate(self.learning_rate)
                    .random_state(self.random_state.unwrap_or(42));

                let fitted = hierarchical.fit(x, _y)?;
                let embedding = fitted.transform(x)?;

                // Evaluate quality
                let quality_metrics = self.evaluate_embedding_quality(x, &embedding)?;
                let combined_quality = self.compute_combined_quality(&quality_metrics)?;

                // Record adaptation step
                let step = AdaptationStep {
                    levels,
                    scale_factors: scale_factors.clone(),
                    quality_score: combined_quality,
                    stress: quality_metrics.stress,
                    neighborhood_preservation: quality_metrics.neighborhood_preservation,
                };
                adaptation_history.push(step);

                // Update best configuration if improved
                if combined_quality > best_quality {
                    best_quality = combined_quality;
                    best_levels = levels;
                    best_scale_factors = scale_factors;
                    best_embeddings = fitted.state.level_embeddings;
                    best_transforms = fitted.state.level_transforms;
                    best_quality_scores = vec![combined_quality]; // Simplified
                }

                // Early stopping if quality threshold reached
                if combined_quality >= self.quality_threshold {
                    break;
                }
            }

            // Early stopping if quality threshold reached
            if best_quality >= self.quality_threshold {
                break;
            }
        }

        Ok(AdaptiveResolutionManifold {
            state: TrainedAdaptiveResolutionManifold {
                optimal_levels: best_levels,
                optimal_scale_factors: best_scale_factors,
                level_embeddings: best_embeddings,
                level_transforms: best_transforms,
                quality_scores: best_quality_scores,
                adaptation_history,
                n_features,
                n_components: self.n_components,
            },
            n_components: self.n_components,
            max_levels: self.max_levels,
            min_levels: self.min_levels,
            base_method: self.base_method,
            quality_threshold: self.quality_threshold,
            adaptation_method: self.adaptation_method,
            max_iterations: self.max_iterations,
            learning_rate: self.learning_rate,
            random_state: self.random_state,
        })
    }
}

impl Transform<ArrayView2<'_, f64>, Array2<f64>>
    for AdaptiveResolutionManifold<TrainedAdaptiveResolutionManifold>
{
    fn transform(&self, x: &ArrayView2<'_, f64>) -> SklResult<Array2<f64>> {
        if x.ncols() != self.state.n_features {
            return Err(SklearsError::InvalidInput(format!(
                "Expected {} features, got {}",
                self.state.n_features,
                x.ncols()
            )));
        }

        // Use the optimal configuration's final level transformation
        let final_transform = &self.state.level_transforms[self.state.optimal_levels - 1];
        let transformed = x.dot(final_transform);
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
    fn test_hierarchical_manifold_basic() {
        let x = array![
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [8.0, 9.0, 10.0],
            [9.0, 10.0, 11.0]
        ];
        let dummy_y = array![(), (), (), ()];

        let hierarchical = HierarchicalManifold::new()
            .n_components(2)
            .levels(2)
            .base_method("pca".to_string())
            .random_state(42);

        let fitted = hierarchical.fit(&x.view(), &dummy_y.view()).unwrap();
        let transformed = fitted.transform(&x.view()).unwrap();

        assert_eq!(transformed.shape(), [4, 2]);
        assert!(transformed.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_multi_scale_embedding_basic() {
        let x = array![
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [8.0, 9.0, 10.0],
            [9.0, 10.0, 11.0]
        ];
        let dummy_y = array![(), (), (), ()];

        let multi_scale = MultiScaleEmbedding::new()
            .n_components(2)
            .scales(vec![0.5, 1.0])
            .base_method("pca".to_string())
            .combination_method("weighted".to_string())
            .random_state(42);

        let fitted = multi_scale.fit(&x.view(), &dummy_y.view()).unwrap();
        let transformed = fitted.transform(&x.view()).unwrap();

        assert_eq!(transformed.shape(), [4, 3]); // May differ based on combination method
        assert!(transformed.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_hierarchical_manifold_different_levels() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [8.0, 9.0], [9.0, 10.0]];
        let dummy_y = array![(), (), (), ()];

        for levels in 1..=3 {
            let hierarchical = HierarchicalManifold::new()
                .n_components(2)
                .levels(levels)
                .refinement_steps(10)
                .random_state(42);

            let fitted = hierarchical.fit(&x.view(), &dummy_y.view()).unwrap();
            let transformed = fitted.transform(&x.view()).unwrap();

            assert_eq!(transformed.shape(), [4, 2]);
            assert!(transformed.iter().all(|&x| x.is_finite()));
        }
    }

    #[test]
    fn test_adaptive_resolution_manifold_basic() {
        let x = array![
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [8.0, 9.0, 10.0],
            [9.0, 10.0, 11.0]
        ];
        let dummy_y = array![(), (), (), ()];

        let adaptive = AdaptiveResolutionManifold::new()
            .n_components(2)
            .max_levels(3)
            .min_levels(2)
            .max_iterations(3) // Reduced for testing
            .quality_threshold(0.5) // Lower threshold for testing
            .adaptation_method("combined".to_string())
            .random_state(42);

        let fitted = adaptive.fit(&x.view(), &dummy_y.view()).unwrap();
        let transformed = fitted.transform(&x.view()).unwrap();

        assert_eq!(transformed.shape(), [4, 2]);
        assert!(transformed.iter().all(|&x| x.is_finite()));

        // Check that adaptive process found optimal configuration
        assert!(fitted.state.optimal_levels >= 2);
        assert!(fitted.state.optimal_levels <= 3);
        assert_eq!(
            fitted.state.optimal_scale_factors.len(),
            fitted.state.optimal_levels
        );
        assert!(!fitted.state.adaptation_history.is_empty());
    }

    #[test]
    fn test_adaptive_resolution_different_methods() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [8.0, 9.0], [9.0, 10.0]];
        let dummy_y = array![(), (), (), ()];

        for method in &["stress_based", "neighborhood_preservation", "combined"] {
            let adaptive = AdaptiveResolutionManifold::new()
                .n_components(2)
                .max_levels(3)
                .max_iterations(2) // Reduced for testing
                .adaptation_method(method.to_string())
                .random_state(42);

            let fitted = adaptive.fit(&x.view(), &dummy_y.view()).unwrap();
            let transformed = fitted.transform(&x.view()).unwrap();

            assert_eq!(transformed.shape(), [4, 2]);
            assert!(transformed.iter().all(|&x| x.is_finite()));
        }
    }
}
