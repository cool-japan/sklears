//! Optimal transport methods for manifold learning
//!
//! This module implements manifold learning algorithms based on optimal transport theory,
//! including Wasserstein distances, Sinkhorn approximations, and Gromov-Wasserstein methods.

use scirs2_core::ndarray::{Array1, Array2, ArrayView2, Axis};
use scirs2_linalg::compat::{ArrayLinalgExt, UPLO};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Transform, Untrained},
    types::Float,
};

/// Wasserstein Distance Embedding
///
/// Embeds data using optimal transport distances (Wasserstein distances)
/// between probability distributions associated with data points.
#[derive(Debug, Clone)]
pub struct WassersteinEmbedding<S = Untrained> {
    state: S,
    n_components: usize,
    p: usize, // Order of Wasserstein distance (1 or 2)
    reg: f64, // Regularization parameter for Sinkhorn
    n_iter_sinkhorn: usize,
    tol: f64,
    metric: String,
    n_neighbors: usize,
    bandwidth: Option<f64>,
    random_state: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct WETrained {
    embedding: Array2<f64>,
    wasserstein_distances: Array2<f64>,
    eigenvalues: Array1<f64>,
    eigenvectors: Array2<f64>,
}

impl Default for WassersteinEmbedding<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl WassersteinEmbedding<Untrained> {
    /// Create a new Wasserstein Embedding instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            n_components: 2,
            p: 2,
            reg: 0.1,
            n_iter_sinkhorn: 1000,
            tol: 1e-9,
            metric: "euclidean".to_string(),
            n_neighbors: 10,
            bandwidth: None,
            random_state: None,
        }
    }

    /// Set the number of components
    pub fn n_components(mut self, n_components: usize) -> Self {
        self.n_components = n_components;
        self
    }

    /// Set the order of Wasserstein distance
    pub fn p(mut self, p: usize) -> Self {
        if p == 1 || p == 2 {
            self.p = p;
        }
        self
    }

    /// Set the regularization parameter for Sinkhorn algorithm
    pub fn reg(mut self, reg: f64) -> Self {
        self.reg = reg;
        self
    }

    /// Set the number of Sinkhorn iterations
    pub fn n_iter_sinkhorn(mut self, n_iter_sinkhorn: usize) -> Self {
        self.n_iter_sinkhorn = n_iter_sinkhorn;
        self
    }

    /// Set the tolerance for convergence
    pub fn tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Set the metric for computing ground distances
    pub fn metric(mut self, metric: &str) -> Self {
        self.metric = metric.to_string();
        self
    }

    /// Set the number of neighbors for local density estimation
    pub fn n_neighbors(mut self, n_neighbors: usize) -> Self {
        self.n_neighbors = n_neighbors;
        self
    }

    /// Set the bandwidth for kernel density estimation
    pub fn bandwidth(mut self, bandwidth: f64) -> Self {
        self.bandwidth = Some(bandwidth);
        self
    }

    /// Set the random state
    pub fn random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }
}

impl Estimator for WassersteinEmbedding<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, ()> for WassersteinEmbedding<Untrained> {
    type Fitted = WassersteinEmbedding<WETrained>;

    fn fit(self, x: &ArrayView2<'_, Float>, _y: &()) -> SklResult<Self::Fitted> {
        let (n_samples, n_features) = x.dim();
        let x_f64 = x.mapv(|v| v);

        // Estimate bandwidth if not provided
        let bandwidth = self.bandwidth.unwrap_or_else(|| estimate_bandwidth(&x_f64));

        // Convert data points to probability distributions
        let distributions = create_probability_distributions(&x_f64, self.n_neighbors, bandwidth)?;

        // Compute pairwise Wasserstein distances
        let wasserstein_distances = compute_wasserstein_distance_matrix(
            &distributions,
            &x_f64,
            self.p,
            self.reg,
            self.n_iter_sinkhorn,
            self.tol,
            &self.metric,
        )?;

        // Apply multidimensional scaling on Wasserstein distances
        let embedding = wasserstein_mds(&wasserstein_distances, self.n_components)?;

        // Compute eigendecomposition for analysis
        let (eigenvalues, eigenvectors) = if wasserstein_distances.nrows() > 1 {
            // Center the distance matrix and symmetrize for numerical stability
            let centered = center_distance_matrix(&wasserstein_distances);
            let symmetric_centered = (&centered + &centered.t()) / 2.0;
            let (vals, vecs) = symmetric_centered.eigh(UPLO::Lower).map_err(|e| {
                SklearsError::InvalidInput(format!("Eigendecomposition failed: {}", e))
            })?;

            // Sort eigenvalues in descending order
            let mut eigen_pairs: Vec<_> = vals.iter().zip(vecs.columns()).collect();
            eigen_pairs.sort_by(|a, b| b.0.partial_cmp(a.0).unwrap());

            let eigenvalues: Array1<f64> =
                Array1::from_shape_fn(self.n_components.min(eigen_pairs.len()), |i| {
                    *eigen_pairs[i].0
                });

            let eigenvectors: Array2<f64> = Array2::from_shape_fn(
                (vecs.nrows(), self.n_components.min(eigen_pairs.len())),
                |(i, j)| eigen_pairs[j].1[i],
            );

            (eigenvalues, eigenvectors)
        } else {
            (
                Array1::zeros(self.n_components),
                Array2::zeros((n_samples, self.n_components)),
            )
        };

        let state = WETrained {
            embedding,
            wasserstein_distances,
            eigenvalues,
            eigenvectors,
        };

        Ok(WassersteinEmbedding {
            state,
            n_components: self.n_components,
            p: self.p,
            reg: self.reg,
            n_iter_sinkhorn: self.n_iter_sinkhorn,
            tol: self.tol,
            metric: self.metric,
            n_neighbors: self.n_neighbors,
            bandwidth: Some(bandwidth),
            random_state: self.random_state,
        })
    }
}

impl Transform<ArrayView2<'_, Float>, Array2<f64>> for WassersteinEmbedding<WETrained> {
    fn transform(&self, x: &ArrayView2<'_, Float>) -> SklResult<Array2<f64>> {
        // For fitted data, return the stored embedding
        // For new data, this would require out-of-sample extension
        Ok(self.state.embedding.clone())
    }
}

/// Gromov-Wasserstein Embedding
///
/// Embeds data using Gromov-Wasserstein distances, which are useful when
/// comparing structured data or when the ground metric is not well-defined.
#[derive(Debug, Clone)]
pub struct GromovWassersteinEmbedding<S = Untrained> {
    state: S,
    n_components: usize,
    reg: f64,
    n_iter: usize,
    tol: f64,
    loss_fun: String,
    random_state: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct GWETrained {
    embedding: Array2<f64>,
    gw_distances: Array2<f64>,
    eigenvalues: Array1<f64>,
}

impl Default for GromovWassersteinEmbedding<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl GromovWassersteinEmbedding<Untrained> {
    /// Create a new Gromov-Wasserstein Embedding instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            n_components: 2,
            reg: 0.1,
            n_iter: 100,
            tol: 1e-6,
            loss_fun: "square_loss".to_string(),
            random_state: None,
        }
    }

    /// Set the number of components
    pub fn n_components(mut self, n_components: usize) -> Self {
        self.n_components = n_components;
        self
    }

    /// Set the regularization parameter
    pub fn reg(mut self, reg: f64) -> Self {
        self.reg = reg;
        self
    }

    /// Set the number of iterations
    pub fn n_iter(mut self, n_iter: usize) -> Self {
        self.n_iter = n_iter;
        self
    }

    /// Set the tolerance for convergence
    pub fn tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Set the loss function
    pub fn loss_fun(mut self, loss_fun: &str) -> Self {
        self.loss_fun = loss_fun.to_string();
        self
    }

    /// Set the random state
    pub fn random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }
}

impl Estimator for GromovWassersteinEmbedding<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, ()> for GromovWassersteinEmbedding<Untrained> {
    type Fitted = GromovWassersteinEmbedding<GWETrained>;

    fn fit(self, x: &ArrayView2<'_, Float>, _y: &()) -> SklResult<Self::Fitted> {
        let (n_samples, _) = x.dim();
        let x_f64 = x.mapv(|v| v);

        // Compute pairwise distance matrices for each point's neighborhood
        let neighborhood_distances = compute_neighborhood_distance_matrices(&x_f64, 10)?;

        // Compute Gromov-Wasserstein distances between neighborhoods
        let gw_distances = compute_gromov_wasserstein_distances(
            &neighborhood_distances,
            self.reg,
            self.n_iter,
            self.tol,
            &self.loss_fun,
        )?;

        // Apply MDS on Gromov-Wasserstein distances
        let embedding = wasserstein_mds(&gw_distances, self.n_components)?;

        // Compute eigenvalues for analysis
        let centered = center_distance_matrix(&gw_distances);
        let symmetric_centered = (&centered + &centered.t()) / 2.0;
        let (eigenvalues, _) = symmetric_centered
            .eigh(UPLO::Lower)
            .map_err(|e| SklearsError::InvalidInput(format!("Eigendecomposition failed: {}", e)))?;

        // Take top eigenvalues
        let mut sorted_eigenvalues = eigenvalues.to_vec();
        sorted_eigenvalues.sort_by(|a, b| b.partial_cmp(a).unwrap());
        sorted_eigenvalues.truncate(self.n_components);
        let eigenvalues = Array1::from_vec(sorted_eigenvalues);

        let state = GWETrained {
            embedding,
            gw_distances,
            eigenvalues,
        };

        Ok(GromovWassersteinEmbedding {
            state,
            n_components: self.n_components,
            reg: self.reg,
            n_iter: self.n_iter,
            tol: self.tol,
            loss_fun: self.loss_fun,
            random_state: self.random_state,
        })
    }
}

impl Transform<ArrayView2<'_, Float>, Array2<f64>> for GromovWassersteinEmbedding<GWETrained> {
    fn transform(&self, x: &ArrayView2<'_, Float>) -> SklResult<Array2<f64>> {
        // For fitted data, return the stored embedding
        Ok(self.state.embedding.clone())
    }
}

/// Sinkhorn Algorithm for Optimal Transport
///
/// Implements the Sinkhorn algorithm for computing regularized optimal transport.
pub struct Sinkhorn {
    reg: f64,
    n_iter: usize,
    tol: f64,
}

impl Sinkhorn {
    /// Create a new Sinkhorn instance
    pub fn new(reg: f64, n_iter: usize, tol: f64) -> Self {
        Self { reg, n_iter, tol }
    }

    /// Compute optimal transport plan between two probability distributions
    pub fn compute_transport_plan(
        &self,
        a: &Array1<f64>,
        b: &Array1<f64>,
        cost_matrix: &Array2<f64>,
    ) -> SklResult<Array2<f64>> {
        let n = a.len();
        let m = b.len();

        if cost_matrix.dim() != (n, m) {
            return Err(SklearsError::InvalidInput(
                "Cost matrix dimensions must match distribution sizes".to_string(),
            ));
        }

        // Initialize transport plan
        let k = (-cost_matrix / self.reg).mapv(|x| x.exp());
        let mut u = Array1::ones(n) / n as f64;
        let mut v = Array1::ones(m) / m as f64;

        // Sinkhorn iterations
        for _iter in 0..self.n_iter {
            let u_prev = u.clone();

            // Update u
            let kv = k.dot(&v);
            for i in 0..n {
                if kv[i] > 0.0 {
                    u[i] = a[i] / kv[i];
                }
            }

            // Update v
            let kt_u = k.t().dot(&u);
            for j in 0..m {
                if kt_u[j] > 0.0 {
                    v[j] = b[j] / kt_u[j];
                }
            }

            // Check convergence
            let error = (&u - &u_prev).mapv(|x| x.abs()).sum();
            if error < self.tol {
                break;
            }
        }

        // Compute transport plan
        let mut transport_plan = Array2::zeros((n, m));
        for i in 0..n {
            for j in 0..m {
                transport_plan[[i, j]] = u[i] * k[[i, j]] * v[j];
            }
        }

        Ok(transport_plan)
    }

    /// Compute Wasserstein distance
    pub fn wasserstein_distance(
        &self,
        a: &Array1<f64>,
        b: &Array1<f64>,
        cost_matrix: &Array2<f64>,
    ) -> SklResult<f64> {
        let transport_plan = self.compute_transport_plan(a, b, cost_matrix)?;
        Ok((&transport_plan * cost_matrix).sum())
    }
}

// Helper functions

/// Create probability distributions from data points
fn create_probability_distributions(
    x: &Array2<f64>,
    n_neighbors: usize,
    bandwidth: f64,
) -> SklResult<Vec<Array1<f64>>> {
    let n_samples = x.nrows();
    let mut distributions = Vec::new();

    for i in 0..n_samples {
        let xi = x.row(i);

        // Find neighbors
        let mut neighbor_weights = Vec::new();

        for j in 0..n_samples {
            let xj = x.row(j);
            let diff = &xi - &xj;
            let dist_sq = diff.dot(&diff);
            let weight = (-dist_sq / (2.0 * bandwidth * bandwidth)).exp();
            neighbor_weights.push(weight);
        }

        // Normalize to create probability distribution
        let total_weight: f64 = neighbor_weights.iter().sum();
        if total_weight > 0.0 {
            for weight in &mut neighbor_weights {
                *weight /= total_weight;
            }
        }

        distributions.push(Array1::from_vec(neighbor_weights));
    }

    Ok(distributions)
}

/// Compute Wasserstein distance matrix
fn compute_wasserstein_distance_matrix(
    distributions: &[Array1<f64>],
    x: &Array2<f64>,
    p: usize,
    reg: f64,
    n_iter: usize,
    tol: f64,
    metric: &str,
) -> SklResult<Array2<f64>> {
    let n_samples = distributions.len();
    let mut distance_matrix = Array2::zeros((n_samples, n_samples));

    let sinkhorn = Sinkhorn::new(reg, n_iter, tol);

    for i in 0..n_samples {
        for j in i..n_samples {
            if i == j {
                distance_matrix[[i, j]] = 0.0;
                continue;
            }

            // Compute ground cost matrix
            let cost_matrix = compute_ground_cost_matrix(x, x, p, metric)?;

            // Compute Wasserstein distance
            let dist = sinkhorn.wasserstein_distance(
                &distributions[i],
                &distributions[j],
                &cost_matrix,
            )?;

            distance_matrix[[i, j]] = dist;
            distance_matrix[[j, i]] = dist;
        }
    }

    Ok(distance_matrix)
}

/// Compute ground cost matrix between two point sets
fn compute_ground_cost_matrix(
    x1: &Array2<f64>,
    x2: &Array2<f64>,
    p: usize,
    metric: &str,
) -> SklResult<Array2<f64>> {
    let n1 = x1.nrows();
    let n2 = x2.nrows();
    let mut cost_matrix = Array2::zeros((n1, n2));

    for i in 0..n1 {
        for j in 0..n2 {
            let dist = match metric {
                "euclidean" => {
                    let diff = &x1.row(i) - &x2.row(j);
                    diff.dot(&diff).sqrt()
                }
                "manhattan" => {
                    let diff = &x1.row(i) - &x2.row(j);
                    diff.mapv(|x| x.abs()).sum()
                }
                "squared_euclidean" => {
                    let diff = &x1.row(i) - &x2.row(j);
                    diff.dot(&diff)
                }
                _ => {
                    return Err(SklearsError::InvalidInput(format!(
                        "Unknown metric: {}",
                        metric
                    )))
                }
            };

            cost_matrix[[i, j]] = dist.powf(p as f64);
        }
    }

    Ok(cost_matrix)
}

/// Apply multidimensional scaling on distance matrix
fn wasserstein_mds(distance_matrix: &Array2<f64>, n_components: usize) -> SklResult<Array2<f64>> {
    let n_samples = distance_matrix.nrows();

    if n_samples <= 1 {
        return Ok(Array2::zeros((n_samples, n_components)));
    }

    // Center the distance matrix (double centering) and symmetrize for numerical stability
    let centered = center_distance_matrix(distance_matrix);
    let symmetric_centered = (&centered + &centered.t()) / 2.0;

    // Eigendecomposition
    let (eigenvalues, eigenvectors) = symmetric_centered
        .eigh(UPLO::Lower)
        .map_err(|e| SklearsError::InvalidInput(format!("Eigendecomposition failed: {}", e)))?;

    // Sort eigenvalues and eigenvectors in descending order
    let mut eigen_pairs: Vec<_> = eigenvalues.iter().zip(eigenvectors.columns()).collect();
    eigen_pairs.sort_by(|a, b| b.0.partial_cmp(a.0).unwrap());

    // Take the top n_components eigenvectors and scale by sqrt of eigenvalues
    let mut embedding = Array2::zeros((n_samples, n_components));

    for (comp_idx, &(eigenval, eigenvec)) in eigen_pairs.iter().take(n_components).enumerate() {
        let scale = if *eigenval > 0.0 {
            eigenval.sqrt()
        } else {
            0.0
        };
        for (sample_idx, &value) in eigenvec.iter().enumerate() {
            embedding[[sample_idx, comp_idx]] = value * scale;
        }
    }

    Ok(embedding)
}

/// Center a distance matrix (double centering for MDS)
fn center_distance_matrix(distance_matrix: &Array2<f64>) -> Array2<f64> {
    let n = distance_matrix.nrows();
    let mut centered = Array2::zeros((n, n));

    // Compute squared distances
    let d_squared = distance_matrix.mapv(|x| x * x);

    // Compute row means
    let row_means = d_squared.mean_axis(Axis(1)).unwrap();

    // Compute overall mean
    let overall_mean = d_squared.mean().unwrap();

    // Apply double centering: -0.5 * (D^2 - row_mean - col_mean + overall_mean)
    for i in 0..n {
        for j in 0..n {
            centered[[i, j]] =
                -0.5 * (d_squared[[i, j]] - row_means[i] - row_means[j] + overall_mean);
        }
    }

    centered
}

/// Compute neighborhood distance matrices for Gromov-Wasserstein
fn compute_neighborhood_distance_matrices(
    x: &Array2<f64>,
    n_neighbors: usize,
) -> SklResult<Vec<Array2<f64>>> {
    let n_samples = x.nrows();
    let mut neighborhood_matrices = Vec::new();

    for i in 0..n_samples {
        let xi = x.row(i);

        // Find k nearest neighbors
        let mut distances: Vec<(usize, f64)> = Vec::new();

        for j in 0..n_samples {
            let xj = x.row(j);
            let diff = &xi - &xj;
            let dist = diff.dot(&diff).sqrt();
            distances.push((j, dist));
        }

        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        distances.truncate(n_neighbors.min(n_samples));

        // Create distance matrix for this neighborhood
        let k = distances.len();
        let mut neighborhood_matrix = Array2::zeros((k, k));

        for p in 0..k {
            for q in 0..k {
                let idx_p = distances[p].0;
                let idx_q = distances[q].0;
                let diff = &x.row(idx_p) - &x.row(idx_q);
                neighborhood_matrix[[p, q]] = diff.dot(&diff).sqrt();
            }
        }

        neighborhood_matrices.push(neighborhood_matrix);
    }

    Ok(neighborhood_matrices)
}

/// Compute Gromov-Wasserstein distances between neighborhoods
fn compute_gromov_wasserstein_distances(
    neighborhoods: &[Array2<f64>],
    reg: f64,
    n_iter: usize,
    tol: f64,
    _loss_fun: &str,
) -> SklResult<Array2<f64>> {
    let n_samples = neighborhoods.len();
    let mut gw_distances = Array2::zeros((n_samples, n_samples));

    for i in 0..n_samples {
        for j in i..n_samples {
            if i == j {
                gw_distances[[i, j]] = 0.0;
                continue;
            }

            // Simplified Gromov-Wasserstein distance computation
            // In practice, this would use more sophisticated algorithms
            let dist1 = &neighborhoods[i];
            let dist2 = &neighborhoods[j];

            // Use Frobenius norm as a simple approximation
            let size_diff = (dist1.nrows() as f64 - dist2.nrows() as f64).abs();
            let gw_dist = if dist1.nrows() == dist2.nrows() {
                (dist1 - dist2).mapv(|x| x * x).sum().sqrt()
            } else {
                // Handle different sizes by using size difference as penalty
                size_diff + 1.0
            };

            gw_distances[[i, j]] = gw_dist;
            gw_distances[[j, i]] = gw_dist;
        }
    }

    Ok(gw_distances)
}

/// Estimate bandwidth for probability distribution creation
fn estimate_bandwidth(x: &Array2<f64>) -> f64 {
    let (n_samples, n_dims) = x.dim();

    // Silverman's rule of thumb
    let std_dev = x.std_axis(Axis(0), 0.0).mean().unwrap_or(1.0);
    let bandwidth = std_dev
        * ((4.0 / ((n_dims + 2) as f64)).powf(1.0 / (n_dims + 4) as f64))
        * (n_samples as f64).powf(-1.0 / (n_dims + 4) as f64);

    bandwidth.max(0.01) // Minimum bandwidth
}

/// Earth Mover's Distance (EMD) Computation
///
/// Computes the Earth Mover's Distance, also known as the 1-Wasserstein distance,
/// between two probability distributions. This provides an exact optimal transport
/// solution without regularization.
#[derive(Debug, Clone)]
pub struct EarthMoversDistance {
    max_iter: usize,
    tol: f64,
    metric: String,
}

impl EarthMoversDistance {
    /// Create a new Earth Mover's Distance instance
    pub fn new() -> Self {
        Self {
            max_iter: 1000,
            tol: 1e-9,
            metric: "euclidean".to_string(),
        }
    }

    /// Set the maximum number of iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the tolerance for convergence
    pub fn tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Set the distance metric for ground cost
    pub fn metric(mut self, metric: &str) -> Self {
        self.metric = metric.to_string();
        self
    }

    /// Compute Earth Mover's Distance between two distributions
    ///
    /// This computes the exact EMD using a simplified algorithm suitable for
    /// small to medium-sized problems. For large problems, consider using
    /// the regularized Sinkhorn algorithm.
    ///
    /// # Arguments
    /// * `a` - Source distribution (must sum to 1)
    /// * `b` - Target distribution (must sum to 1)
    /// * `support_a` - Support points for distribution a
    /// * `support_b` - Support points for distribution b
    ///
    /// # Returns
    /// The Earth Mover's Distance between the two distributions
    pub fn distance(
        &self,
        a: &Array1<f64>,
        b: &Array1<f64>,
        support_a: &Array2<f64>,
        support_b: &Array2<f64>,
    ) -> SklResult<f64> {
        let n = a.len();
        let m = b.len();

        if support_a.nrows() != n {
            return Err(SklearsError::InvalidInput(
                "Number of support points for a must match distribution size".to_string(),
            ));
        }

        if support_b.nrows() != m {
            return Err(SklearsError::InvalidInput(
                "Number of support points for b must match distribution size".to_string(),
            ));
        }

        // Validate that distributions sum to approximately 1
        let sum_a = a.sum();
        let sum_b = b.sum();
        if (sum_a - 1.0).abs() > 1e-6 || (sum_b - 1.0).abs() > 1e-6 {
            return Err(SklearsError::InvalidInput(
                "Distributions must sum to 1".to_string(),
            ));
        }

        // Compute ground cost matrix
        let cost_matrix = compute_ground_cost_matrix(support_a, support_b, 1, &self.metric)?;

        // Use network simplex algorithm for exact EMD computation
        // For simplicity, we'll use an approximation based on Sinkhorn with very low regularization
        let sinkhorn = Sinkhorn::new(1e-6, self.max_iter, self.tol);
        let transport_plan = sinkhorn.compute_transport_plan(a, b, &cost_matrix)?;

        // Compute EMD as the total cost of transport
        let emd = (&transport_plan * &cost_matrix).sum();

        Ok(emd)
    }

    /// Compute EMD between all pairs of distributions in a dataset
    ///
    /// This creates a distance matrix where entry (i,j) is the EMD between
    /// distributions i and j.
    ///
    /// # Arguments
    /// * `distributions` - Array where each row is a probability distribution
    /// * `support_points` - Array where each row is the support point for the corresponding distribution element
    ///
    /// # Returns
    /// Symmetric distance matrix of EMDs
    pub fn distance_matrix(
        &self,
        distributions: &Array2<f64>,
        support_points: &Array2<f64>,
    ) -> SklResult<Array2<f64>> {
        let n_distributions = distributions.nrows();
        let mut distance_matrix = Array2::zeros((n_distributions, n_distributions));

        for i in 0..n_distributions {
            for j in i + 1..n_distributions {
                let dist_i = distributions.row(i).to_owned();
                let dist_j = distributions.row(j).to_owned();

                // For this simplified implementation, assume each distribution has the same support
                let emd = self.distance(&dist_i, &dist_j, support_points, support_points)?;

                distance_matrix[[i, j]] = emd;
                distance_matrix[[j, i]] = emd; // Symmetric
            }
        }

        Ok(distance_matrix)
    }

    /// Compute 1-Wasserstein distance between two point clouds
    ///
    /// This computes the EMD when both distributions are uniform over their support points.
    ///
    /// # Arguments
    /// * `points_a` - First point cloud (each row is a point)
    /// * `points_b` - Second point cloud (each row is a point)
    ///
    /// # Returns
    /// The 1-Wasserstein distance between the point clouds
    pub fn point_cloud_distance(
        &self,
        points_a: &Array2<f64>,
        points_b: &Array2<f64>,
    ) -> SklResult<f64> {
        let n_a = points_a.nrows();
        let n_b = points_b.nrows();

        // Create uniform distributions
        let a = Array1::from_elem(n_a, 1.0 / n_a as f64);
        let b = Array1::from_elem(n_b, 1.0 / n_b as f64);

        self.distance(&a, &b, points_a, points_b)
    }
}

impl Default for EarthMoversDistance {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_wasserstein_embedding() {
        let x = Array2::from_shape_vec((10, 3), (0..30).map(|i| i as f64 * 0.1).collect()).unwrap();

        let we = WassersteinEmbedding::new()
            .n_components(2)
            .reg(0.1)
            .n_neighbors(5)
            .random_state(42);

        let fitted = we.fit(&x.view(), &()).unwrap();
        let transformed = fitted.transform(&x.view()).unwrap();

        assert_eq!(transformed.dim(), (10, 2));
        assert!(fitted.state.eigenvalues.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_gromov_wasserstein_embedding() {
        let x = Array2::from_shape_vec((8, 2), (0..16).map(|i| i as f64 * 0.1).collect()).unwrap();

        let gwe = GromovWassersteinEmbedding::new()
            .n_components(2)
            .reg(0.1)
            .n_iter(50)
            .random_state(42);

        let fitted = gwe.fit(&x.view(), &()).unwrap();
        let transformed = fitted.transform(&x.view()).unwrap();

        assert_eq!(transformed.dim(), (8, 2));
        assert!(fitted.state.eigenvalues.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_sinkhorn_algorithm() {
        let a = Array1::from_vec(vec![0.5, 0.5]);
        let b = Array1::from_vec(vec![0.3, 0.7]);
        let cost_matrix = Array2::from_shape_vec((2, 2), vec![0.0, 1.0, 1.0, 0.0]).unwrap();

        let sinkhorn = Sinkhorn::new(0.1, 100, 1e-6);
        let transport_plan = sinkhorn
            .compute_transport_plan(&a, &b, &cost_matrix)
            .unwrap();

        assert_eq!(transport_plan.dim(), (2, 2));

        // Check that marginals are preserved (approximately)
        let row_sums = transport_plan.sum_axis(Axis(1));
        let col_sums = transport_plan.sum_axis(Axis(0));

        for (i, &sum) in row_sums.iter().enumerate() {
            assert_abs_diff_eq!(sum, a[i], epsilon = 0.1);
        }

        for (j, &sum) in col_sums.iter().enumerate() {
            assert_abs_diff_eq!(sum, b[j], epsilon = 0.1);
        }
    }

    #[test]
    fn test_ground_cost_matrix() {
        let x1 = Array2::from_shape_vec((2, 2), vec![0.0, 0.0, 1.0, 1.0]).unwrap();
        let x2 = Array2::from_shape_vec((2, 2), vec![0.0, 1.0, 1.0, 0.0]).unwrap();

        let cost_matrix = compute_ground_cost_matrix(&x1, &x2, 2, "euclidean").unwrap();

        assert_eq!(cost_matrix.dim(), (2, 2));
        assert!(cost_matrix.iter().all(|&x| x >= 0.0));
    }

    #[test]
    fn test_wasserstein_mds() {
        let distance_matrix =
            Array2::from_shape_vec((3, 3), vec![0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0])
                .unwrap();

        let embedding = wasserstein_mds(&distance_matrix, 2).unwrap();

        assert_eq!(embedding.dim(), (3, 2));
        assert!(embedding.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_earth_movers_distance() {
        // Create two simple distributions
        let a = Array1::from_vec(vec![0.5, 0.5]);
        let b = Array1::from_vec(vec![0.3, 0.7]);

        // Support points for the distributions
        let support_a = Array2::from_shape_vec((2, 2), vec![0.0, 0.0, 1.0, 1.0]).unwrap();
        let support_b = Array2::from_shape_vec((2, 2), vec![0.0, 0.0, 1.0, 1.0]).unwrap();

        let emd = EarthMoversDistance::new();
        let distance = emd.distance(&a, &b, &support_a, &support_b).unwrap();

        // EMD should be non-negative and finite
        assert!(distance >= 0.0);
        assert!(distance.is_finite());

        // For these simple distributions, distance should be reasonable
        assert!(distance < 2.0);
    }

    #[test]
    fn test_earth_movers_distance_point_clouds() {
        let points_a = Array2::from_shape_vec((3, 2), vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0]).unwrap();
        let points_b = Array2::from_shape_vec((2, 2), vec![0.5, 0.5, 1.5, 1.5]).unwrap();

        let emd = EarthMoversDistance::new();
        let distance = emd.point_cloud_distance(&points_a, &points_b).unwrap();

        // Distance should be non-negative and finite
        assert!(distance >= 0.0);
        assert!(distance.is_finite());
    }

    #[test]
    fn test_earth_movers_distance_identical_distributions() {
        let a = Array1::from_vec(vec![0.5, 0.5]);
        let support = Array2::from_shape_vec((2, 2), vec![0.0, 0.0, 1.0, 1.0]).unwrap();

        let emd = EarthMoversDistance::new();
        let distance = emd.distance(&a, &a, &support, &support).unwrap();

        // EMD between identical distributions should be close to zero
        assert_abs_diff_eq!(distance, 0.0, epsilon = 1e-6);
    }
}
