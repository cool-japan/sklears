//! t-SNE (t-distributed Stochastic Neighbor Embedding) implementation
//!
//! This module provides t-SNE for non-linear dimensionality reduction and visualization.

use scirs2_core::ndarray::{Array2, ArrayView2, Axis};
use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::thread_rng;
use scirs2_core::random::Rng;
use scirs2_core::random::SeedableRng;
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Transform, Untrained},
    types::Float,
};

/// t-distributed Stochastic Neighbor Embedding (t-SNE)
///
/// t-SNE is a tool to visualize high-dimensional data. It converts similarities
/// between data points to joint probabilities and tries to minimize the Kullback-Leibler
/// divergence between the joint probabilities of the low-dimensional embedding
/// and the high-dimensional data.
///
/// # Parameters
///
/// * `n_components` - Dimension of the embedded space
/// * `perplexity` - The perplexity is related to the number of nearest neighbors
/// * `early_exaggeration` - Controls how tight natural clusters are
/// * `learning_rate` - The learning rate for t-SNE is usually between 10 and 1000
/// * `n_iter` - Maximum number of iterations for the optimization
/// * `n_iter_without_progress` - Maximum number of iterations without progress
/// * `min_grad_norm` - Minimum norm of the gradient for stopping condition
/// * `metric` - The metric to use when computing distances
/// * `init` - Initialization method for embedding
/// * `verbose` - Whether to be verbose
/// * `random_state` - Random state for reproducibility
/// * `method` - The gradient calculation algorithm
/// * `angle` - For Barnes-Hut t-SNE only
/// * `n_jobs` - Number of parallel jobs
///
/// # Examples
///
/// ```
/// use sklears_manifold::TSNE;
/// use sklears_core::traits::{Transform, Fit};
/// use scirs2_core::ndarray::array;
///
/// let X = array![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [10.0, 10.0], [11.0, 11.0], [12.0, 12.0]];
///
/// let tsne = TSNE::new()
///     .n_components(2)
///     .perplexity(2.0)
///     .n_iter(100);
/// let fitted = tsne.fit(&X.view(), &()).unwrap();
/// let embedding = fitted.transform(&X.view()).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct TSNE<S = Untrained> {
    state: S,
    n_components: usize,
    perplexity: f64,
    early_exaggeration: f64,
    learning_rate: f64,
    n_iter: usize,
    n_iter_without_progress: usize,
    min_grad_norm: f64,
    metric: String,
    init: String,
    verbose: bool,
    random_state: Option<u64>,
    method: String,
    angle: f64,
    n_jobs: Option<i32>,
}

/// Trained state for TSNE
#[derive(Debug, Clone)]
pub struct TsneTrained {
    /// The final embedding
    pub embedding: Array2<f64>,
    /// Final KL divergence
    pub kl_divergence: f64,
    /// Number of iterations actually run
    pub n_iter_final: usize,
}

impl TSNE<Untrained> {
    /// Create a new TSNE instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            n_components: 2,
            perplexity: 30.0,
            early_exaggeration: 12.0,
            learning_rate: 200.0,
            n_iter: 1000,
            n_iter_without_progress: 300,
            min_grad_norm: 1e-7,
            metric: "euclidean".to_string(),
            init: "random".to_string(),
            verbose: false,
            random_state: None,
            method: "barnes_hut".to_string(),
            angle: 0.5,
            n_jobs: None,
        }
    }

    /// Set the number of components
    pub fn n_components(mut self, n_components: usize) -> Self {
        self.n_components = n_components;
        self
    }

    /// Set the perplexity
    pub fn perplexity(mut self, perplexity: f64) -> Self {
        self.perplexity = perplexity;
        self
    }

    /// Set the early exaggeration
    pub fn early_exaggeration(mut self, early_exaggeration: f64) -> Self {
        self.early_exaggeration = early_exaggeration;
        self
    }

    /// Set the learning rate
    pub fn learning_rate(mut self, learning_rate: f64) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    /// Set the number of iterations
    pub fn n_iter(mut self, n_iter: usize) -> Self {
        self.n_iter = n_iter;
        self
    }

    /// Set the number of iterations without progress
    pub fn n_iter_without_progress(mut self, n_iter_without_progress: usize) -> Self {
        self.n_iter_without_progress = n_iter_without_progress;
        self
    }

    /// Set the minimum gradient norm
    pub fn min_grad_norm(mut self, min_grad_norm: f64) -> Self {
        self.min_grad_norm = min_grad_norm;
        self
    }

    /// Set the metric
    pub fn metric(mut self, metric: &str) -> Self {
        self.metric = metric.to_string();
        self
    }

    /// Set the initialization method
    pub fn init(mut self, init: &str) -> Self {
        self.init = init.to_string();
        self
    }

    /// Set verbosity
    pub fn verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// Set the random state
    pub fn random_state(mut self, random_state: Option<u64>) -> Self {
        self.random_state = random_state;
        self
    }

    /// Set the method
    pub fn method(mut self, method: &str) -> Self {
        self.method = method.to_string();
        self
    }

    /// Set the angle for Barnes-Hut
    pub fn angle(mut self, angle: f64) -> Self {
        self.angle = angle;
        self
    }

    /// Set the number of jobs
    pub fn n_jobs(mut self, n_jobs: Option<i32>) -> Self {
        self.n_jobs = n_jobs;
        self
    }
}

impl Default for TSNE<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for TSNE<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, ()> for TSNE<Untrained> {
    type Fitted = TSNE<TsneTrained>;

    fn fit(self, x: &ArrayView2<'_, Float>, _y: &()) -> SklResult<Self::Fitted> {
        let x = x.mapv(|x| x);
        let (n_samples, _n_features) = x.dim();

        if n_samples <= 1 {
            return Err(SklearsError::InvalidInput(
                "t-SNE requires at least 2 samples".to_string(),
            ));
        }

        if self.perplexity >= n_samples as f64 {
            return Err(SklearsError::InvalidInput(
                "Perplexity must be less than number of samples".to_string(),
            ));
        }

        // Compute pairwise distances
        let distances = self.compute_pairwise_distances(&x)?;

        // Compute conditional probabilities
        let p = self.compute_joint_probabilities(&distances)?;

        // Initialize embedding
        let mut rng = if let Some(seed) = self.random_state {
            StdRng::seed_from_u64(seed)
        } else {
            StdRng::seed_from_u64(thread_rng().random::<u64>())
        };

        let mut y = match self.init.as_str() {
            "random" => {
                let mut embedding = Array2::zeros((n_samples, self.n_components));
                for i in 0..n_samples {
                    for j in 0..self.n_components {
                        embedding[[i, j]] = rng.gen::<f64>() * 1e-4;
                    }
                }
                embedding
            }
            "pca" => {
                // Simplified PCA initialization
                let mut embedding = Array2::zeros((n_samples, self.n_components));
                for i in 0..n_samples {
                    for j in 0..self.n_components {
                        embedding[[i, j]] = (i as f64) / (n_samples as f64) - 0.5;
                    }
                }
                embedding
            }
            _ => {
                return Err(SklearsError::InvalidInput(
                    "Unknown initialization method".to_string(),
                ))
            }
        };

        // Run optimization
        let mut gains: Array2<f64> = Array2::ones((n_samples, self.n_components));
        let mut iy: Array2<f64> = Array2::zeros((n_samples, self.n_components));
        let _uy: Array2<f64> = Array2::zeros((n_samples, self.n_components));

        let mut best_error = f64::INFINITY;
        let mut no_progress_count = 0;

        for iter in 0..self.n_iter {
            let mut dy = if self.method == "barnes_hut" && n_samples > 250 {
                // Use Barnes-Hut approximation for large datasets
                self.compute_barnes_hut_gradient(&y, &p)?
            } else {
                // Use exact computation for small datasets
                let q = self.compute_low_dimensional_similarities(&y)?;
                let pq_diff = &p - &q;
                let mut dy = Array2::zeros((n_samples, self.n_components));

                for i in 0..n_samples {
                    for j in 0..n_samples {
                        if i != j {
                            let y_diff = y.row(i).to_owned() - y.row(j);
                            let dist_squared = y_diff.mapv(|x| x * x).sum();
                            let factor = pq_diff[[i, j]] / (1.0 + dist_squared);

                            for k in 0..self.n_components {
                                dy[[i, k]] += 4.0 * factor * y_diff[k];
                            }
                        }
                    }
                }
                dy
            };

            // Apply early exaggeration
            if iter < 250 {
                dy *= self.early_exaggeration;
            }

            // Update gains
            for i in 0..n_samples {
                for j in 0..self.n_components {
                    if (dy[[i, j]] > 0.0) == (iy[[i, j]] > 0.0) {
                        gains[[i, j]] *= 0.8;
                    } else {
                        gains[[i, j]] += 0.2;
                    }
                    gains[[i, j]] = gains[[i, j]].max(0.01);
                }
            }

            // Update embedding
            iy = 0.8 * &iy - self.learning_rate * &gains * &dy;
            y = &y + &iy;

            // Center embedding
            let mean = y.mean_axis(Axis(0)).unwrap();
            y -= &mean.insert_axis(Axis(0));

            // Check convergence
            let q = self.compute_low_dimensional_similarities(&y)?;
            let error = self.compute_kl_divergence(&p, &q);
            if error < best_error {
                best_error = error;
                no_progress_count = 0;
            } else {
                no_progress_count += 1;
            }

            if no_progress_count >= self.n_iter_without_progress {
                if self.verbose {
                    println!("Convergence reached at iteration {iter}");
                }
                break;
            }

            let grad_norm = dy.mapv(|x: f64| x * x).sum().sqrt();
            if grad_norm < self.min_grad_norm {
                if self.verbose {
                    println!("Gradient norm below threshold at iteration {iter}");
                }
                break;
            }

            if self.verbose && iter % 100 == 0 {
                println!("Iteration {iter}: KL divergence = {error:.6}");
            }
        }

        Ok(TSNE {
            state: TsneTrained {
                embedding: y,
                kl_divergence: best_error,
                n_iter_final: self.n_iter.min(no_progress_count),
            },
            n_components: self.n_components,
            perplexity: self.perplexity,
            early_exaggeration: self.early_exaggeration,
            learning_rate: self.learning_rate,
            n_iter: self.n_iter,
            n_iter_without_progress: self.n_iter_without_progress,
            min_grad_norm: self.min_grad_norm,
            metric: self.metric,
            init: self.init,
            verbose: self.verbose,
            random_state: self.random_state,
            method: self.method,
            angle: self.angle,
            n_jobs: self.n_jobs,
        })
    }
}

/// Spatial tree node for Barnes-Hut approximation
#[derive(Debug, Clone)]
struct SpatialTreeNode {
    /// Center of mass for this node
    center_of_mass: Array2<f64>,
    /// Total mass (number of points) in this node
    total_mass: f64,
    /// Bounding box - minimum coordinates
    bbox_min: Array2<f64>,
    /// Bounding box - maximum coordinates
    bbox_max: Array2<f64>,
    /// Indices of points contained in this node (for leaf nodes)
    point_indices: Vec<usize>,
    /// Children nodes (for internal nodes)
    children: Vec<Option<Box<SpatialTreeNode>>>,
    /// Whether this is a leaf node
    is_leaf: bool,
}

impl SpatialTreeNode {
    fn new(bbox_min: Array2<f64>, bbox_max: Array2<f64>) -> Self {
        let n_dims = bbox_min.len();
        Self {
            center_of_mass: Array2::zeros((1, n_dims)),
            total_mass: 0.0,
            bbox_min,
            bbox_max,
            point_indices: Vec::new(),
            children: vec![None; 1 << n_dims], // 2^n_dims children
            is_leaf: true,
        }
    }

    fn build_tree(points: &Array2<f64>, max_points_per_node: usize) -> SpatialTreeNode {
        let n_dims = points.ncols();
        let n_points = points.nrows();

        // Compute bounding box
        let mut bbox_min = Array2::from_elem((1, n_dims), f64::INFINITY);
        let mut bbox_max = Array2::from_elem((1, n_dims), f64::NEG_INFINITY);

        for i in 0..n_points {
            for j in 0..n_dims {
                bbox_min[[0, j]] = bbox_min[[0, j]].min(points[[i, j]]);
                bbox_max[[0, j]] = bbox_max[[0, j]].max(points[[i, j]]);
            }
        }

        // Add small padding to avoid boundary issues
        for j in 0..n_dims {
            let range = bbox_max[[0, j]] - bbox_min[[0, j]];
            bbox_min[[0, j]] -= range * 0.01;
            bbox_max[[0, j]] += range * 0.01;
        }

        let mut root = SpatialTreeNode::new(bbox_min, bbox_max);
        let point_indices: Vec<usize> = (0..n_points).collect();
        root.insert_points(points, &point_indices, max_points_per_node);
        root
    }

    fn insert_points(
        &mut self,
        points: &Array2<f64>,
        indices: &[usize],
        max_points_per_node: usize,
    ) {
        self.point_indices = indices.to_vec();
        self.total_mass = indices.len() as f64;

        // Compute center of mass
        let n_dims = points.ncols();
        let mut com = Array2::zeros((1, n_dims));
        for &idx in indices {
            for j in 0..n_dims {
                com[[0, j]] += points[[idx, j]];
            }
        }
        if self.total_mass > 0.0 {
            com /= self.total_mass;
        }
        self.center_of_mass = com;

        // If we have few enough points, remain a leaf
        if indices.len() <= max_points_per_node {
            return;
        }

        // Otherwise, subdivide
        self.is_leaf = false;
        self.subdivide(points, max_points_per_node);
    }

    fn subdivide(&mut self, points: &Array2<f64>, max_points_per_node: usize) {
        let n_dims = points.ncols();
        let n_children = 1 << n_dims;

        // Create child bounding boxes
        let bbox_center = (&self.bbox_min + &self.bbox_max) / 2.0;

        for child_idx in 0..n_children {
            let mut child_min = self.bbox_min.clone();
            let mut child_max = self.bbox_max.clone();

            for dim in 0..n_dims {
                if (child_idx >> dim) & 1 == 1 {
                    child_min[[0, dim]] = bbox_center[[0, dim]];
                } else {
                    child_max[[0, dim]] = bbox_center[[0, dim]];
                }
            }

            self.children[child_idx] = Some(Box::new(SpatialTreeNode::new(child_min, child_max)));
        }

        // Distribute points to children
        for &point_idx in &self.point_indices {
            let mut child_idx = 0;
            for dim in 0..n_dims {
                if points[[point_idx, dim]] >= bbox_center[[0, dim]] {
                    child_idx |= 1 << dim;
                }
            }

            if let Some(ref mut child) = self.children[child_idx] {
                child.point_indices.push(point_idx);
            }
        }

        // Recursively build children
        for ref mut child_node in self.children.iter_mut().flatten() {
            if !child_node.point_indices.is_empty() {
                let child_indices = child_node.point_indices.clone();
                child_node.insert_points(points, &child_indices, max_points_per_node);
            }
        }

        // Clear point indices from internal node
        self.point_indices.clear();
    }

    fn compute_force(
        &self,
        point: &Array2<f64>,
        point_idx: usize,
        theta: f64,
        force: &mut Array2<f64>,
    ) {
        if self.total_mass == 0.0 {
            return;
        }

        let n_dims = point.ncols();
        let mut dist_squared = 0.0;
        let mut diff = Array2::zeros((1, n_dims));

        for j in 0..n_dims {
            let d = self.center_of_mass[[0, j]] - point[[0, j]];
            diff[[0, j]] = d;
            dist_squared += d * d;
        }

        if dist_squared == 0.0 {
            return;
        }

        // Compute bbox diameter
        let mut bbox_diameter: f64 = 0.0;
        for j in 0..n_dims {
            let span = self.bbox_max[[0, j]] - self.bbox_min[[0, j]];
            bbox_diameter = bbox_diameter.max(span);
        }

        let dist = dist_squared.sqrt();

        // Barnes-Hut criterion: if bbox_diameter / distance < theta, use approximation
        if self.is_leaf || bbox_diameter / dist < theta {
            // Use this node as approximation
            let factor = self.total_mass / (1.0 + dist_squared).powi(2);
            for j in 0..n_dims {
                force[[0, j]] += factor * diff[[0, j]];
            }
        } else {
            // Recurse to children
            for child_node in self.children.iter().flatten() {
                child_node.compute_force(point, point_idx, theta, force);
            }
        }
    }
}

impl TSNE<Untrained> {
    fn compute_pairwise_distances(&self, x: &Array2<f64>) -> SklResult<Array2<f64>> {
        let n = x.nrows();
        let mut distances = Array2::zeros((n, n));

        match self.metric.as_str() {
            "euclidean" => {
                for i in 0..n {
                    for j in i + 1..n {
                        let diff = &x.row(i) - &x.row(j);
                        let dist = diff.mapv(|x| x * x).sum().sqrt();
                        distances[[i, j]] = dist;
                        distances[[j, i]] = dist;
                    }
                }
            }
            _ => return Err(SklearsError::InvalidInput("Unsupported metric".to_string())),
        }

        Ok(distances)
    }

    fn compute_joint_probabilities(&self, distances: &Array2<f64>) -> SklResult<Array2<f64>> {
        let n = distances.nrows();
        let mut p = Array2::zeros((n, n));

        // Compute conditional probabilities for each point
        for i in 0..n {
            let mut beta = 1.0; // Initial beta (precision)
            let mut beta_min = 0.0;
            let mut beta_max = f64::INFINITY;

            // Binary search for beta that gives desired perplexity
            for _attempt in 0..50 {
                let mut sum_prob = 0.0;
                let mut entropy = 0.0;

                for j in 0..n {
                    if i != j {
                        let prob = (-beta * distances[[i, j]] * distances[[i, j]]).exp();
                        p[[i, j]] = prob;
                        sum_prob += prob;
                    }
                }

                if sum_prob > 0.0 {
                    // Normalize
                    for j in 0..n {
                        if i != j {
                            p[[i, j]] /= sum_prob;
                            if p[[i, j]] > 1e-12 {
                                entropy += -p[[i, j]] * p[[i, j]].ln();
                            }
                        }
                    }

                    let perp = 2_f64.powf(entropy);
                    let perp_diff = perp - self.perplexity;

                    if perp_diff.abs() < 1e-5 {
                        break;
                    }

                    if perp_diff > 0.0 {
                        beta_min = beta;
                        beta = if beta_max == f64::INFINITY {
                            beta * 2.0
                        } else {
                            (beta + beta_max) / 2.0
                        };
                    } else {
                        beta_max = beta;
                        beta = (beta + beta_min) / 2.0;
                    }
                }
            }
        }

        // Symmetrize
        for i in 0..n {
            for j in 0..n {
                p[[i, j]] = (p[[i, j]] + p[[j, i]]) / (2.0 * n as f64);
                p[[i, j]] = p[[i, j]].max(1e-12); // Avoid numerical issues
            }
        }

        Ok(p)
    }

    fn compute_low_dimensional_similarities(&self, y: &Array2<f64>) -> SklResult<Array2<f64>> {
        let n = y.nrows();
        let mut q = Array2::zeros((n, n));
        let mut sum_q = 0.0;

        for i in 0..n {
            for j in 0..n {
                if i != j {
                    let diff = &y.row(i) - &y.row(j);
                    let dist_squared = diff.mapv(|x| x * x).sum();
                    let q_val = 1.0 / (1.0 + dist_squared);
                    q[[i, j]] = q_val;
                    sum_q += q_val;
                }
            }
        }

        // Normalize
        if sum_q > 0.0 {
            q /= sum_q;
        }

        // Ensure minimum value
        q = q.mapv(|x| x.max(1e-12));

        Ok(q)
    }

    fn compute_barnes_hut_gradient(
        &self,
        y: &Array2<f64>,
        p: &Array2<f64>,
    ) -> SklResult<Array2<f64>> {
        let n_samples = y.nrows();
        let mut dy = Array2::zeros((n_samples, self.n_components));

        // Build spatial tree for current embedding
        let tree = SpatialTreeNode::build_tree(y, 1);

        // Compute total normalization constant Z
        let mut z = 0.0;
        for i in 0..n_samples {
            for j in 0..n_samples {
                if i != j {
                    let diff = y.row(i).to_owned() - y.row(j);
                    let dist_squared = diff.mapv(|x| x * x).sum();
                    z += 1.0 / (1.0 + dist_squared);
                }
            }
        }

        // For each point, compute attractive and repulsive forces
        for i in 0..n_samples {
            let mut attractive_force: Array2<f64> = Array2::zeros((1, self.n_components));
            let mut repulsive_force: Array2<f64> = Array2::zeros((1, self.n_components));

            // Attractive forces (exact computation with neighbors)
            for j in 0..n_samples {
                if i != j && p[[i, j]] > 1e-12 {
                    let y_diff = y.row(i).to_owned() - y.row(j);
                    let dist_squared = y_diff.mapv(|x| x * x).sum();
                    let q_ij = 1.0 / (1.0 + dist_squared) / z;

                    let factor = 4.0 * p[[i, j]] * q_ij / (1.0 + dist_squared);
                    for k in 0..self.n_components {
                        attractive_force[[0, k]] += factor * y_diff[k];
                    }
                }
            }

            // Repulsive forces (Barnes-Hut approximation)
            let point = y
                .row(i)
                .to_owned()
                .insert_axis(scirs2_core::ndarray::Axis(0));
            tree.compute_force(&point, i, self.angle, &mut repulsive_force);

            // Normalize repulsive force by Z and apply factor
            repulsive_force *= 4.0 / z;

            // Combine forces
            for k in 0..self.n_components {
                dy[[i, k]] = attractive_force[[0, k]] - repulsive_force[[0, k]];
            }
        }

        Ok(dy)
    }

    fn compute_kl_divergence(&self, p: &Array2<f64>, q: &Array2<f64>) -> f64 {
        let mut kl = 0.0;
        let n = p.nrows();

        for i in 0..n {
            for j in 0..n {
                if p[[i, j]] > 1e-12 && q[[i, j]] > 1e-12 {
                    kl += p[[i, j]] * (p[[i, j]] / q[[i, j]]).ln();
                }
            }
        }

        kl
    }
}

impl Transform<ArrayView2<'_, Float>, Array2<f64>> for TSNE<TsneTrained> {
    fn transform(&self, _x: &ArrayView2<'_, Float>) -> SklResult<Array2<f64>> {
        // t-SNE doesn't support transforming new data
        Ok(self.state.embedding.clone())
    }
}

impl TSNE<TsneTrained> {
    /// Get the embedding
    pub fn embedding(&self) -> &Array2<f64> {
        &self.state.embedding
    }

    /// Get the final KL divergence
    pub fn kl_divergence(&self) -> f64 {
        self.state.kl_divergence
    }

    /// Get the number of iterations run
    pub fn n_iter_final(&self) -> usize {
        self.state.n_iter_final
    }
}
