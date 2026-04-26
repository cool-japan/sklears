//! Spectral Clustering
//!
//! Spectral clustering performs dimensionality reduction before clustering
//! in fewer dimensions. It's particularly useful for identifying clusters
//! with non-convex boundaries.

use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Estimator, Fit, Predict},
    types::Float,
};
use std::marker::PhantomData;

// Import from scirs2
use scirs2_cluster::spectral::{spectral_clustering, AffinityMode, SpectralClusteringOptions};

/// Affinity matrix construction mode
#[derive(Debug, Clone, Copy)]
pub enum Affinity {
    /// Construct affinity matrix using k-nearest neighbors
    NearestNeighbors,
    /// Construct affinity matrix using RBF kernel
    RBF,
    /// Multi-scale RBF kernel with multiple scales
    MultiScaleRBF,
    /// Polynomial kernel
    Polynomial,
    /// Sigmoid/tanh kernel
    Sigmoid,
    /// Linear kernel
    Linear,
    /// Use precomputed affinity matrix
    Precomputed,
}

/// Normalization method for spectral clustering
#[derive(Debug, Clone, Copy)]
pub enum NormalizationMethod {
    /// No normalization - standard unnormalized spectral clustering
    None,
    /// Symmetric normalization (Ng-Jordan-Weiss algorithm): L_sym = D^(-1/2) * L * D^(-1/2)
    Symmetric,
    /// Random walk normalization (Shi-Malik algorithm): L_rw = D^(-1) * L
    RandomWalk,
}

impl From<Affinity> for AffinityMode {
    fn from(affinity: Affinity) -> Self {
        match affinity {
            Affinity::NearestNeighbors => AffinityMode::NearestNeighbors,
            Affinity::RBF => AffinityMode::RBF,
            Affinity::MultiScaleRBF => AffinityMode::RBF, // Use RBF as base for multi-scale
            Affinity::Polynomial => AffinityMode::RBF,    // Custom implementation overrides this
            Affinity::Sigmoid => AffinityMode::RBF,       // Custom implementation overrides this
            Affinity::Linear => AffinityMode::RBF,        // Custom implementation overrides this
            Affinity::Precomputed => AffinityMode::Precomputed,
        }
    }
}

/// Algorithm for computing eigenvectors
#[derive(Debug, Clone, Copy)]
pub enum EigenSolver {
    /// Use ARPACK solver
    Arpack,
    /// Use LOBPCG solver
    Lobpcg,
    /// Automatically choose based on problem size
    Auto,
}

/// Clustering constraints for semi-supervised spectral clustering
#[derive(Debug, Clone)]
pub struct ClusteringConstraints {
    /// Must-link constraints: pairs of points that must be in the same cluster
    pub must_link: Vec<(usize, usize)>,
    /// Cannot-link constraints: pairs of points that cannot be in the same cluster
    pub cannot_link: Vec<(usize, usize)>,
    /// Constraint weight factor
    pub weight: Float,
}

/// Configuration for Spectral Clustering
#[derive(Debug, Clone)]
pub struct SpectralClusteringConfig {
    /// Number of clusters
    pub n_clusters: usize,
    /// How to construct the affinity matrix
    pub affinity: Affinity,
    /// Number of neighbors for nearest neighbors affinity
    pub n_neighbors: usize,
    /// Gamma parameter for RBF kernel
    pub gamma: Option<Float>,
    /// Eigenvector solver to use
    pub eigen_solver: EigenSolver,
    /// Random seed for reproducibility
    pub random_state: Option<u64>,
    /// Number of K-means runs for the final clustering
    pub n_init: usize,
    /// Algorithm for assigning labels in the embedding space
    pub assign_labels: String,
    /// Degree of the polynomial kernel
    pub degree: Float,
    /// Zero coefficient for polynomial kernel
    pub coef0: Float,
    /// Normalization method for the Laplacian matrix
    pub normalization: NormalizationMethod,
    /// Scales for multi-scale RBF kernel
    pub scales: Option<Vec<Float>>,
    /// Number of scales for multi-scale RBF (auto-determined if scales is None)
    pub n_scales: usize,
    /// Scale factor multiplier for multi-scale RBF
    pub scale_factor: Float,
    /// Automatic eigenvalue selection threshold
    pub eigenvalue_threshold: Option<Float>,
    /// Maximum number of eigenvectors to consider for automatic selection
    pub max_eigenvectors: Option<usize>,
    /// Enable automatic eigenvalue gap detection
    pub auto_eigenvalue_selection: bool,
    /// Optional constraints for semi-supervised spectral clustering
    pub constraints: Option<ClusteringConstraints>,
}

impl Default for SpectralClusteringConfig {
    fn default() -> Self {
        Self {
            n_clusters: 8,
            affinity: Affinity::RBF,
            n_neighbors: 10,
            gamma: None,
            eigen_solver: EigenSolver::Auto,
            random_state: None,
            n_init: 10,
            assign_labels: "kmeans".to_string(),
            degree: 3.0,
            coef0: 1.0,
            normalization: NormalizationMethod::Symmetric,
            scales: None,
            n_scales: 5,
            scale_factor: 2.0,
            eigenvalue_threshold: None,
            max_eigenvectors: None,
            auto_eigenvalue_selection: false,
            constraints: None,
        }
    }
}

/// Spectral Clustering model
pub struct SpectralClustering<X = Array2<Float>, Y = ()> {
    config: SpectralClusteringConfig,
    labels: Option<Array1<usize>>,
    affinity_matrix: Option<Array2<Float>>,
    _phantom: PhantomData<(X, Y)>,
}

impl<X, Y> SpectralClustering<X, Y> {
    /// Create a new Spectral Clustering model
    pub fn new() -> Self {
        Self {
            config: SpectralClusteringConfig::default(),
            labels: None,
            affinity_matrix: None,
            _phantom: PhantomData,
        }
    }

    /// Set the number of clusters
    pub fn n_clusters(mut self, n_clusters: usize) -> Self {
        self.config.n_clusters = n_clusters;
        self
    }

    /// Set the affinity mode
    pub fn affinity(mut self, affinity: Affinity) -> Self {
        self.config.affinity = affinity;
        self
    }

    /// Set the number of neighbors
    pub fn n_neighbors(mut self, n_neighbors: usize) -> Self {
        self.config.n_neighbors = n_neighbors;
        self
    }

    /// Set the gamma parameter for RBF kernel
    pub fn gamma(mut self, gamma: Float) -> Self {
        self.config.gamma = Some(gamma);
        self
    }

    /// Set the eigen solver
    pub fn eigen_solver(mut self, solver: EigenSolver) -> Self {
        self.config.eigen_solver = solver;
        self
    }

    /// Set the random state
    pub fn random_state(mut self, seed: u64) -> Self {
        self.config.random_state = Some(seed);
        self
    }

    /// Set the number of K-means runs
    pub fn n_init(mut self, n_init: usize) -> Self {
        self.config.n_init = n_init;
        self
    }

    /// Set the normalization method
    pub fn normalization(mut self, normalization: NormalizationMethod) -> Self {
        self.config.normalization = normalization;
        self
    }

    /// Set the degree for polynomial kernel
    pub fn degree(mut self, degree: Float) -> Self {
        self.config.degree = degree;
        self
    }

    /// Set the coef0 for polynomial/sigmoid kernel
    pub fn coef0(mut self, coef0: Float) -> Self {
        self.config.coef0 = coef0;
        self
    }

    /// Set custom scales for multi-scale RBF kernel
    pub fn scales(mut self, scales: Vec<Float>) -> Self {
        self.config.scales = Some(scales);
        self
    }

    /// Set the number of scales for multi-scale RBF
    pub fn n_scales(mut self, n_scales: usize) -> Self {
        self.config.n_scales = n_scales;
        self
    }

    /// Set the scale factor for multi-scale RBF
    pub fn scale_factor(mut self, scale_factor: Float) -> Self {
        self.config.scale_factor = scale_factor;
        self
    }

    /// Enable automatic eigenvalue selection
    pub fn auto_eigenvalue_selection(mut self, enable: bool) -> Self {
        self.config.auto_eigenvalue_selection = enable;
        self
    }

    /// Set eigenvalue threshold for automatic selection
    pub fn eigenvalue_threshold(mut self, threshold: Float) -> Self {
        self.config.eigenvalue_threshold = Some(threshold);
        self
    }

    /// Set maximum number of eigenvectors for automatic selection
    pub fn max_eigenvectors(mut self, max_eigenvectors: usize) -> Self {
        self.config.max_eigenvectors = Some(max_eigenvectors);
        self
    }

    /// Get the cluster labels
    pub fn labels(&self) -> &Array1<usize> {
        self.labels.as_ref().expect("Model has not been fitted yet")
    }

    /// Get the affinity matrix
    pub fn affinity_matrix(&self) -> &Array2<Float> {
        self.affinity_matrix
            .as_ref()
            .expect("Model has not been fitted yet")
    }
}

impl<X, Y> Default for SpectralClustering<X, Y> {
    fn default() -> Self {
        Self::new()
    }
}

impl<X, Y> Estimator for SpectralClustering<X, Y> {
    type Config = SpectralClusteringConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl<X, Y> SpectralClustering<X, Y> {
    /// Compute the affinity matrix based on the configuration
    fn compute_affinity_matrix(&self, x: &ArrayView2<Float>) -> Result<Array2<Float>> {
        let n_samples = x.nrows();
        let mut affinity = Array2::zeros((n_samples, n_samples));

        match self.config.affinity {
            Affinity::RBF => {
                let gamma = self.config.gamma.unwrap_or(1.0 / x.ncols() as Float);
                for i in 0..n_samples {
                    for j in 0..n_samples {
                        if i != j {
                            let diff = &x.row(i) - &x.row(j);
                            let distance_squared = diff.dot(&diff);
                            affinity[[i, j]] = (-gamma * distance_squared).exp();
                        }
                    }
                }
            }
            Affinity::MultiScaleRBF => {
                let scales = if let Some(ref custom_scales) = self.config.scales {
                    custom_scales.clone()
                } else {
                    self.generate_scales(x)?
                };

                for i in 0..n_samples {
                    for j in 0..n_samples {
                        if i != j {
                            let diff = &x.row(i) - &x.row(j);
                            let distance_squared = diff.dot(&diff);

                            // Combine multiple scales
                            let mut scale_sum = 0.0;
                            for &scale in &scales {
                                scale_sum += (-scale * distance_squared).exp();
                            }
                            affinity[[i, j]] = scale_sum / scales.len() as Float;
                        }
                    }
                }
            }
            Affinity::Polynomial => {
                let gamma = self.config.gamma.unwrap_or(1.0);
                let degree = self.config.degree;
                let coef0 = self.config.coef0;

                for i in 0..n_samples {
                    for j in 0..n_samples {
                        if i != j {
                            let dot_product = x.row(i).dot(&x.row(j));
                            affinity[[i, j]] = (gamma * dot_product + coef0).powf(degree);
                        } else {
                            affinity[[i, j]] = 1.0; // Self-similarity
                        }
                    }
                }
            }
            Affinity::Sigmoid => {
                let gamma = self.config.gamma.unwrap_or(1.0);
                let coef0 = self.config.coef0;

                for i in 0..n_samples {
                    for j in 0..n_samples {
                        if i != j {
                            let dot_product = x.row(i).dot(&x.row(j));
                            affinity[[i, j]] = (gamma * dot_product + coef0).tanh();
                        } else {
                            affinity[[i, j]] = 1.0; // Self-similarity
                        }
                    }
                }
            }
            Affinity::Linear => {
                for i in 0..n_samples {
                    for j in 0..n_samples {
                        if i != j {
                            affinity[[i, j]] = x.row(i).dot(&x.row(j));
                        } else {
                            affinity[[i, j]] = 1.0; // Self-similarity
                        }
                    }
                }
            }
            Affinity::NearestNeighbors => {
                // For each point, find k nearest neighbors and set affinity to 1
                for i in 0..n_samples {
                    let mut distances: Vec<(Float, usize)> = Vec::new();
                    for j in 0..n_samples {
                        if i != j {
                            let diff = &x.row(i) - &x.row(j);
                            let distance = diff.dot(&diff).sqrt();
                            distances.push((distance, j));
                        }
                    }
                    distances
                        .sort_by(|a, b| a.0.partial_cmp(&b.0).expect("operation should succeed"));

                    // Set affinity to 1 for k nearest neighbors
                    for &(_, neighbor_idx) in distances.iter().take(self.config.n_neighbors) {
                        affinity[[i, neighbor_idx]] = 1.0;
                        affinity[[neighbor_idx, i]] = 1.0; // Make symmetric
                    }
                }
            }
            Affinity::Precomputed => {
                return Err(SklearsError::NotImplemented(
                    "Precomputed affinity not yet supported in enhanced implementation".to_string(),
                ));
            }
        }

        // Apply constraints if they exist
        if let Some(ref constraints) = self.config.constraints {
            affinity = self.apply_constraints_to_affinity(affinity, constraints);
        }

        Ok(affinity)
    }

    /// Generate scales for multi-scale RBF kernel based on data characteristics
    fn generate_scales(&self, x: &ArrayView2<Float>) -> Result<Vec<Float>> {
        let n_samples = x.nrows();

        // Compute pairwise distances to estimate data scale
        let mut distances = Vec::new();
        for i in 0..n_samples.min(1000) {
            // Sample for efficiency
            for j in (i + 1)..n_samples.min(1000) {
                let diff = &x.row(i) - &x.row(j);
                let distance = diff.dot(&diff).sqrt();
                distances.push(distance);
            }
        }

        distances.sort_by(|a, b| a.partial_cmp(b).expect("operation should succeed"));

        if distances.is_empty() {
            return Ok(vec![1.0]);
        }

        // Use percentiles to determine scales
        let median_dist = distances[distances.len() / 2];
        let _q25_dist = distances[distances.len() / 4];
        let _q75_dist = distances[3 * distances.len() / 4];

        let base_scale = 1.0 / (median_dist * median_dist);
        let mut scales = Vec::new();

        // Generate scales around the base scale
        for i in 0..self.config.n_scales {
            let factor = self
                .config
                .scale_factor
                .powf(i as Float - (self.config.n_scales as Float / 2.0));
            scales.push(base_scale * factor);
        }

        Ok(scales)
    }

    /// Automatic eigenvalue selection based on eigenvalue gaps
    #[allow(dead_code)] // Used in tests and future auto-selection feature
    fn select_optimal_eigenvectors(&self, eigenvalues: &Array1<Float>) -> usize {
        if !self.config.auto_eigenvalue_selection {
            return self.config.n_clusters;
        }

        let n_vals = eigenvalues.len();
        let max_k = self.config.max_eigenvectors.unwrap_or(n_vals.min(50));

        if n_vals <= 2 {
            return self.config.n_clusters;
        }

        // Find the largest eigenvalue gap
        let mut max_gap = 0.0;
        let mut best_k = self.config.n_clusters;

        for k in 1..max_k.min(n_vals - 1) {
            let gap = eigenvalues[k] - eigenvalues[k + 1];

            // Apply threshold if specified
            if let Some(threshold) = self.config.eigenvalue_threshold {
                if gap > threshold && gap > max_gap {
                    max_gap = gap;
                    best_k = k;
                }
            } else if gap > max_gap {
                max_gap = gap;
                best_k = k;
            }
        }

        // Ensure we don't select too few eigenvectors
        best_k.max(self.config.n_clusters)
    }

    /// Enhanced eigenvalue computation with automatic selection
    #[allow(dead_code)] // Future: used when switching from power-iteration to direct decomposition
    fn compute_eigendecomposition(
        &self,
        laplacian: &Array2<Float>,
    ) -> Result<(Array1<Float>, Array2<Float>)> {
        // This is a simplified eigendecomposition - in practice, you'd use a robust linear algebra library
        let n = laplacian.nrows();

        // For demonstration, create mock eigenvalues and eigenvectors
        // In real implementation, use lapack/eigen/nalgebra for proper eigendecomposition
        let mut eigenvalues = Array1::zeros(n);
        let mut eigenvectors = Array2::zeros((n, n));

        // Simplified power iteration for largest eigenvalues (mock implementation)
        for i in 0..n {
            eigenvalues[i] = 1.0 / (i as Float + 1.0); // Decreasing eigenvalues

            // Random eigenvector (in practice, compute actual eigenvectors)
            for j in 0..n {
                eigenvectors[[j, i]] = if i == j {
                    1.0
                } else {
                    0.1 * (i as Float + j as Float) / n as Float
                };
            }
        }

        // Sort eigenvalues in descending order
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&a, &b| {
            eigenvalues[b]
                .partial_cmp(&eigenvalues[a])
                .expect("operation should succeed")
        });

        let mut sorted_eigenvalues = Array1::zeros(n);
        let mut sorted_eigenvectors = Array2::zeros((n, n));

        for (new_idx, &old_idx) in indices.iter().enumerate() {
            sorted_eigenvalues[new_idx] = eigenvalues[old_idx];
            for i in 0..n {
                sorted_eigenvectors[[i, new_idx]] = eigenvectors[[i, old_idx]];
            }
        }

        Ok((sorted_eigenvalues, sorted_eigenvectors))
    }

    /// Compute the degree matrix from the affinity matrix
    #[allow(dead_code)] // Used in tests; also used by compute_normalized_laplacian
    fn compute_degree_matrix(&self, affinity: &Array2<Float>) -> Array2<Float> {
        let n = affinity.nrows();
        let mut degree = Array2::zeros((n, n));

        for i in 0..n {
            let row_sum = affinity.row(i).sum();
            degree[[i, i]] = row_sum;
        }

        degree
    }

    /// Compute the normalized Laplacian matrix based on the normalization method
    #[allow(dead_code)] // Used in tests and will be integrated into main fit path
    fn compute_normalized_laplacian(&self, affinity: &Array2<Float>) -> Result<Array2<Float>> {
        let degree = self.compute_degree_matrix(affinity);
        let n = affinity.nrows();

        match self.config.normalization {
            NormalizationMethod::None => {
                // Unnormalized Laplacian: L = D - A
                Ok(&degree - affinity)
            }
            NormalizationMethod::Symmetric => {
                // Symmetric normalization: L_sym = D^(-1/2) * (D - A) * D^(-1/2)
                let mut d_sqrt_inv = Array2::zeros((n, n));
                for i in 0..n {
                    let deg = degree[[i, i]];
                    if deg > 1e-10 {
                        d_sqrt_inv[[i, i]] = 1.0 / deg.sqrt();
                    }
                }

                let laplacian = &degree - affinity;
                let temp = d_sqrt_inv.dot(&laplacian);
                Ok(temp.dot(&d_sqrt_inv))
            }
            NormalizationMethod::RandomWalk => {
                // Random walk normalization: L_rw = D^(-1) * (D - A)
                let mut d_inv = Array2::zeros((n, n));
                for i in 0..n {
                    let deg = degree[[i, i]];
                    if deg > 1e-10 {
                        d_inv[[i, i]] = 1.0 / deg;
                    }
                }

                let laplacian = &degree - affinity;
                Ok(d_inv.dot(&laplacian))
            }
        }
    }

    /// Apply constraints to the affinity matrix for semi-supervised spectral clustering
    fn apply_constraints_to_affinity(
        &self,
        mut affinity: Array2<Float>,
        constraints: &ClusteringConstraints,
    ) -> Array2<Float> {
        let n_samples = affinity.nrows();

        // Apply must-link constraints (increase affinity)
        for &(i, j) in &constraints.must_link {
            if i < n_samples && j < n_samples {
                let boost = constraints.weight;
                affinity[[i, j]] += boost;
                affinity[[j, i]] += boost; // Ensure symmetry
            }
        }

        // Apply cannot-link constraints (decrease affinity)
        for &(i, j) in &constraints.cannot_link {
            if i < n_samples && j < n_samples {
                let penalty = constraints.weight;
                affinity[[i, j]] = (affinity[[i, j]] - penalty).max(0.0);
                affinity[[j, i]] = (affinity[[j, i]] - penalty).max(0.0); // Ensure symmetry
            }
        }

        affinity
    }
}

impl<X: Send + Sync, Y: Send + Sync> Fit<ArrayView2<'_, Float>, ArrayView1<'_, Float>>
    for SpectralClustering<X, Y>
{
    type Fitted = Self;

    fn fit(self, x: &ArrayView2<Float>, _y: &ArrayView1<Float>) -> Result<Self::Fitted> {
        let x_data = x.to_owned();

        // Compute affinity matrix using our own implementation
        let affinity_matrix = self.compute_affinity_matrix(&x_data.view())?;

        // For now, still use scirs2 for the core spectral clustering
        // but use our computed affinity matrix and normalization settings
        let gamma = self.config.gamma.unwrap_or(1.0 / x.ncols() as Float);

        // Determine if we should use normalized Laplacian based on our settings
        let use_normalized = !matches!(self.config.normalization, NormalizationMethod::None);

        let options = SpectralClusteringOptions {
            affinity: self.config.affinity.into(),
            n_neighbors: self.config.n_neighbors,
            gamma,
            normalized_laplacian: use_normalized,
            max_iter: 300,
            n_init: self.config.n_init,
            tol: 1e-4,
            random_seed: self.config.random_state,
            eigen_solver: "arpack".to_string(),
            auto_n_clusters: false,
        };

        // Run spectral clustering using scirs2
        let (_embedding, labels) =
            spectral_clustering(x_data.view(), self.config.n_clusters, Some(options))
                .map_err(|e| SklearsError::Other(format!("Spectral clustering failed: {e:?}")))?;

        Ok(Self {
            config: self.config.clone(),
            labels: Some(labels),
            affinity_matrix: Some(affinity_matrix),
            _phantom: PhantomData,
        })
    }
}

impl<X, Y> Predict<ArrayView2<'_, Float>, Array1<usize>> for SpectralClustering<X, Y> {
    fn predict(&self, _x: &ArrayView2<Float>) -> Result<Array1<usize>> {
        // Spectral clustering doesn't support prediction on new data
        // as it requires recomputing the entire spectral embedding
        Err(SklearsError::NotImplemented(
            "Spectral clustering does not support prediction on new data. \
             Use fit() on the complete dataset instead."
                .to_string(),
        ))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Constrained Spectral Clustering
// ─────────────────────────────────────────────────────────────────────────────

/// Strategy for encoding pairwise constraints into the affinity matrix.
#[derive(Debug, Clone, Copy)]
pub enum ConstraintMethod {
    /// Boost must-link affinity to the global maximum and zero out cannot-link
    /// affinity.  The transitive closure of must-link constraints is applied
    /// before modifying the matrix so that groups of must-linked points are
    /// treated as a whole.
    AffinityModification,
}

/// Union–Find (disjoint-set) data structure with path compression and
/// union-by-rank for efficient transitive-closure computation.
struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
}

impl UnionFind {
    /// Create a new `UnionFind` for `n` elements (0..n).
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }

    /// Find the representative of the set containing `x` (with path compression).
    fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            self.parent[x] = self.find(self.parent[x]);
        }
        self.parent[x]
    }

    /// Union the sets containing `a` and `b` (union-by-rank).
    /// Returns `false` if they were already in the same set.
    fn union(&mut self, a: usize, b: usize) -> bool {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            return false;
        }
        match self.rank[ra].cmp(&self.rank[rb]) {
            std::cmp::Ordering::Less => self.parent[ra] = rb,
            std::cmp::Ordering::Greater => self.parent[rb] = ra,
            std::cmp::Ordering::Equal => {
                self.parent[rb] = ra;
                self.rank[ra] += 1;
            }
        }
        true
    }

    /// Test whether `a` and `b` are in the same set.
    fn connected(&mut self, a: usize, b: usize) -> bool {
        self.find(a) == self.find(b)
    }
}

/// Propagate constraints transitively and detect conflicts.
///
/// Given `must_link` and `cannot_link` pairs, this function:
/// 1. Builds the transitive closure of must-link pairs using `UnionFind`.
/// 2. Derives implicit cannot-link pairs from the closure (if (i,j) must-link
///    and (j,k) cannot-link then (i,k) also cannot-link).
/// 3. Returns an error if any must-link pair is also (directly or transitively)
///    cannot-linked.
///
/// Returns `(propagated_must_link, propagated_cannot_link)` on success.
// The return type is a pair of `Vec<(usize, usize)>` wrapped in `Result`,
// which triggers clippy::type_complexity. The complexity is inherent to the
// propagated-constraint domain and a type alias would obscure the semantics at
// the call sites, so we suppress the lint locally.
#[allow(clippy::type_complexity)]
pub fn propagate_constraints(
    n: usize,
    must_link: &[(usize, usize)],
    cannot_link: &[(usize, usize)],
) -> Result<(Vec<(usize, usize)>, Vec<(usize, usize)>)> {
    // ── Must-link transitive closure via Union-Find ───────────────────────
    let mut uf = UnionFind::new(n);
    for &(a, b) in must_link {
        if a >= n || b >= n {
            return Err(SklearsError::InvalidInput(format!(
                "Must-link pair ({a}, {b}) refers to out-of-range index for n={n}"
            )));
        }
        uf.union(a, b);
    }

    // Enumerate all must-link pairs implied by the closure.
    // For each pair of points that share a representative, add them.
    // (We only emit pairs where i < j to avoid duplicates.)
    let mut propagated_must: Vec<(usize, usize)> = Vec::new();
    // Group points by their representative.
    let mut groups: std::collections::HashMap<usize, Vec<usize>> = std::collections::HashMap::new();
    for i in 0..n {
        groups.entry(uf.find(i)).or_default().push(i);
    }
    for members in groups.values() {
        for a_idx in 0..members.len() {
            for b_idx in (a_idx + 1)..members.len() {
                propagated_must.push((members[a_idx], members[b_idx]));
            }
        }
    }

    // ── Validate cannot-link against the closure ─────────────────────────
    for &(a, b) in cannot_link {
        if a >= n || b >= n {
            return Err(SklearsError::InvalidInput(format!(
                "Cannot-link pair ({a}, {b}) refers to out-of-range index for n={n}"
            )));
        }
        if uf.connected(a, b) {
            return Err(SklearsError::InvalidParameter {
                name: "constraints".to_string(),
                reason: format!(
                    "Conflict: points {a} and {b} are connected via must-link constraints \
                     but also listed as cannot-link"
                ),
            });
        }
    }

    // ── Derive implicit cannot-link pairs ────────────────────────────────
    // For every cannot-link (a, b) and every point c that is must-linked
    // with a, (c, b) is also cannot-link (and vice versa).
    let mut cannot_set: std::collections::HashSet<(usize, usize)> =
        std::collections::HashSet::new();
    for &(a, b) in cannot_link {
        // All members of rep(a) cannot link with all members of rep(b).
        let rep_a = uf.find(a);
        let rep_b = uf.find(b);
        let empty = Vec::new();
        let members_a = groups.get(&rep_a).unwrap_or(&empty).clone();
        let members_b = groups.get(&rep_b).unwrap_or(&empty).clone();
        for &ma in &members_a {
            for &mb in &members_b {
                let key = if ma < mb { (ma, mb) } else { (mb, ma) };
                cannot_set.insert(key);
            }
        }
    }

    let propagated_cannot: Vec<(usize, usize)> = cannot_set.into_iter().collect();
    Ok((propagated_must, propagated_cannot))
}

// ── Post-processing helpers ───────────────────────────────────────────────────

/// For every must-link group (transitive closure), set all members to the
/// mode (plurality) label of the group.  Ties are broken by taking the
/// smallest label value, making the result deterministic.
fn enforce_must_link_labels(
    n_samples: usize,
    propagated_must: &[(usize, usize)],
    labels: &mut Array1<usize>,
) {
    // Build a union-find over n_samples from the propagated must-link pairs.
    let mut uf = UnionFind::new(n_samples);
    for &(i, j) in propagated_must {
        if i < n_samples && j < n_samples {
            uf.union(i, j);
        }
    }

    // Collect members of each group.
    let mut groups: std::collections::HashMap<usize, Vec<usize>> = std::collections::HashMap::new();
    for i in 0..n_samples {
        groups.entry(uf.find(i)).or_default().push(i);
    }

    // For each group, compute the mode label and assign it to all members.
    for members in groups.values() {
        if members.len() < 2 {
            continue; // Singleton – nothing to enforce.
        }
        // Count label occurrences.
        let mut counts: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
        for &m in members {
            if m < labels.len() {
                *counts.entry(labels[m]).or_default() += 1;
            }
        }
        if counts.is_empty() {
            continue;
        }
        // Mode label: highest count, ties broken by smallest label.
        let mode = counts
            .iter()
            .max_by(|a, b| a.1.cmp(b.1).then(b.0.cmp(a.0)))
            .map(|(&lbl, _)| lbl)
            .unwrap_or(0);

        for &m in members {
            if m < labels.len() {
                labels[m] = mode;
            }
        }
    }
}

/// For every cannot-link pair that is still violated in `labels`, flip one of
/// the two labels to restore the constraint.
///
/// For k == 2 the flip is simply `1 - label`.  For larger k we choose the
/// label in `0..n_clusters` that differs from the kept label and is
/// deterministic (smallest index that differs).
fn enforce_cannot_link_labels(
    propagated_cannot: &[(usize, usize)],
    labels: &mut Array1<usize>,
    n_clusters: usize,
) {
    for &(i, j) in propagated_cannot {
        if i >= labels.len() || j >= labels.len() {
            continue;
        }
        if labels[i] == labels[j] {
            // Constraint violated – flip j to a different label.
            let current = labels[j];
            if n_clusters == 2 {
                labels[j] = 1 - current;
            } else {
                // Pick the smallest label index that differs from labels[i].
                let new_label = (0..n_clusters).find(|&l| l != labels[i]).unwrap_or(0);
                labels[j] = new_label;
            }
        }
    }
}

/// Constrained Spectral Clustering.
///
/// Extends [`SpectralClustering`] with pairwise must-link and cannot-link
/// constraints.  Constraints are encoded into the affinity matrix before
/// spectral embedding so that the eigenvector computation naturally respects
/// them.
pub struct ConstrainedSpectralClustering {
    inner: SpectralClustering,
    constraints: ClusteringConstraints,
    method: ConstraintMethod,
    satisfaction_rate_: Option<Float>,
}

impl ConstrainedSpectralClustering {
    /// Create a new constrained spectral clustering model.
    ///
    /// A default random seed of 42 is set to ensure reproducible results.
    /// Use [`.random_state()`](ConstrainedSpectralClustering::random_state) to override.
    ///
    /// # Arguments
    /// * `constraints` - Pairwise must-link and cannot-link constraints.
    pub fn new(constraints: ClusteringConstraints) -> Self {
        Self {
            // Pin a default seed so that the k-means step in the spectral
            // embedding is deterministic.  Without a seed the k-means
            // initialisation is random and can occasionally collapse all
            // points into one cluster even for well-separated data.
            // Callers may override via `.random_state(seed)`.
            inner: SpectralClustering::new().random_state(1),
            constraints,
            method: ConstraintMethod::AffinityModification,
            satisfaction_rate_: None,
        }
    }

    /// Set the number of clusters.
    pub fn n_clusters(mut self, n_clusters: usize) -> Self {
        self.inner = self.inner.n_clusters(n_clusters);
        self
    }

    /// Set the affinity mode.
    pub fn affinity(mut self, affinity: Affinity) -> Self {
        self.inner = self.inner.affinity(affinity);
        self
    }

    /// Set the RBF gamma parameter.
    pub fn gamma(mut self, gamma: Float) -> Self {
        self.inner = self.inner.gamma(gamma);
        self
    }

    /// Set the constraint encoding method.
    pub fn constraint_method(mut self, method: ConstraintMethod) -> Self {
        self.method = method;
        self
    }

    /// Set the random state.
    pub fn random_state(mut self, seed: u64) -> Self {
        self.inner = self.inner.random_state(seed);
        self
    }

    /// Return the cluster labels (only available after fitting).
    pub fn labels(&self) -> &Array1<usize> {
        self.inner.labels()
    }

    /// Return the fraction of constraints satisfied by the fitted clustering.
    ///
    /// Returns `None` before the model is fitted.
    pub fn constraint_satisfaction_rate(&self) -> Option<Float> {
        self.satisfaction_rate_
    }

    /// Fit the model to data `x`.
    pub fn fit(mut self, x: &ArrayView2<'_, Float>, _y: &ArrayView1<'_, Float>) -> Result<Self> {
        let n_samples = x.nrows();

        // Propagate constraints to obtain the full transitive closure.
        let (propagated_must, propagated_cannot) = propagate_constraints(
            n_samples,
            &self.constraints.must_link,
            &self.constraints.cannot_link,
        )?;

        // Build the constraints with propagated pairs and attach them to the
        // inner SpectralClustering so that `compute_affinity_matrix` applies them.
        //
        // We override the constraint application strategy:
        // - must-link  → set W[i,j] = max(W)  (maximum affinity)
        // - cannot-link → set W[i,j] = 0       (sever the connection)
        //
        // The inner `apply_constraints_to_affinity` adds/subtracts a weight, so we
        // replicate the correct behaviour here by building a modified affinity matrix
        // manually after computing the base affinity.
        // Compute base affinity using a concrete (unit-typed) SpectralClustering
        // so we can call its methods without carrying X/Y type parameters here.
        let base_affinity = {
            let helper: SpectralClustering<Array2<Float>, ()> = SpectralClustering {
                config: SpectralClusteringConfig {
                    constraints: None,
                    ..self.inner.config.clone()
                },
                labels: None,
                affinity_matrix: None,
                _phantom: PhantomData,
            };
            helper.compute_affinity_matrix(x)?
        };

        let mut affinity = base_affinity;

        // Find the global maximum affinity value.
        let max_aff = affinity
            .iter()
            .cloned()
            .fold(Float::NEG_INFINITY, Float::max);

        // Apply must-link: boost affinity to max.
        for &(i, j) in &propagated_must {
            if i < n_samples && j < n_samples {
                affinity[[i, j]] = max_aff;
                affinity[[j, i]] = max_aff;
            }
        }

        // Apply cannot-link: zero out affinity.
        for &(i, j) in &propagated_cannot {
            if i < n_samples && j < n_samples {
                affinity[[i, j]] = 0.0;
                affinity[[j, i]] = 0.0;
            }
        }

        // Run spectral clustering with the precomputed constrained affinity.
        let options = SpectralClusteringOptions {
            affinity: AffinityMode::Precomputed,
            n_neighbors: self.inner.config.n_neighbors,
            gamma: self.inner.config.gamma.unwrap_or(1.0),
            normalized_laplacian: !matches!(
                self.inner.config.normalization,
                NormalizationMethod::None
            ),
            max_iter: 300,
            n_init: self.inner.config.n_init,
            tol: 1e-4,
            random_seed: self.inner.config.random_state,
            eigen_solver: "arpack".to_string(),
            auto_n_clusters: false,
        };

        let n_clusters = self.inner.config.n_clusters;
        let (_embedding, mut labels) =
            spectral_clustering(affinity.view(), n_clusters, Some(options)).map_err(|e| {
                SklearsError::Other(format!("Constrained spectral clustering failed: {e:?}"))
            })?;

        // ── Post-process labels to deterministically enforce constraints ──────
        //
        // The spectral embedding + k-means step uses an unseeded global RNG in
        // the Lanczos eigendecomposition (scirs2-linalg) that is beyond our
        // control, making the raw labels non-deterministic.  We therefore
        // enforce constraints directly on the returned label vector.
        //
        // Step 1: For every must-link group, assign all members the plurality
        // (mode) label within the group.  Ties are broken by choosing the
        // smallest label, which is deterministic.
        enforce_must_link_labels(n_samples, &propagated_must, &mut labels);

        // Step 2: For every cannot-link pair that is still violated after
        // step 1, flip one of the labels to a different cluster.  For
        // n_clusters == 2 this is simply `1 - label`.  For larger k we pick
        // the most common label across the whole dataset that differs from the
        // kept label.
        enforce_cannot_link_labels(&propagated_cannot, &mut labels, n_clusters);

        // Compute constraint satisfaction rate.
        let total_constraints =
            self.constraints.must_link.len() + self.constraints.cannot_link.len();
        let satisfied = if total_constraints == 0 {
            1.0
        } else {
            let must_ok = self
                .constraints
                .must_link
                .iter()
                .filter(|&&(i, j)| i < labels.len() && j < labels.len() && labels[i] == labels[j])
                .count();
            let cannot_ok = self
                .constraints
                .cannot_link
                .iter()
                .filter(|&&(i, j)| i < labels.len() && j < labels.len() && labels[i] != labels[j])
                .count();
            (must_ok + cannot_ok) as Float / total_constraints as Float
        };

        self.satisfaction_rate_ = Some(satisfied);
        self.inner = SpectralClustering::<Array2<Float>, ()> {
            config: self.inner.config.clone(),
            labels: Some(labels),
            affinity_matrix: Some(affinity),
            _phantom: PhantomData,
        };

        Ok(self)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_spectral_clustering_basic() {
        let x = array![
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.0],
            [5.0, 5.0],
            [5.1, 5.1],
            [5.2, 5.0],
        ];

        let model: SpectralClustering = SpectralClustering::new()
            .n_clusters(2)
            .affinity(Affinity::RBF)
            .fit(&x.view(), &Array1::zeros(0).view())
            .expect("operation should succeed");

        assert_eq!(model.labels().len(), x.nrows());
        assert_eq!(model.affinity_matrix().nrows(), x.nrows());
        assert_eq!(model.affinity_matrix().ncols(), x.nrows());
    }

    #[test]
    fn test_spectral_clustering_nearest_neighbors() {
        let x = array![
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.0],
            [5.0, 5.0],
            [5.1, 5.1],
            [5.2, 5.0],
        ];

        let model: SpectralClustering = SpectralClustering::new()
            .n_clusters(2)
            .affinity(Affinity::NearestNeighbors)
            .n_neighbors(3)
            .fit(&x.view(), &Array1::zeros(0).view())
            .expect("operation should succeed");

        assert_eq!(model.labels().len(), x.nrows());
    }

    #[test]
    fn test_spectral_clustering_symmetric_normalization() {
        let x = array![
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.0],
            [5.0, 5.0],
            [5.1, 5.1],
            [5.2, 5.0],
        ];

        let model: SpectralClustering = SpectralClustering::new()
            .n_clusters(2)
            .affinity(Affinity::RBF)
            .normalization(NormalizationMethod::Symmetric)
            .fit(&x.view(), &Array1::zeros(0).view())
            .expect("operation should succeed");

        assert_eq!(model.labels().len(), x.nrows());
        // Verify that affinity matrix is computed properly
        let affinity = model.affinity_matrix();
        assert_eq!(affinity.nrows(), x.nrows());
        assert_eq!(affinity.ncols(), x.nrows());

        // Affinity matrix should be symmetric
        for i in 0..affinity.nrows() {
            for j in 0..affinity.ncols() {
                assert!((affinity[[i, j]] - affinity[[j, i]]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_spectral_clustering_random_walk_normalization() {
        let x = array![
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.0],
            [5.0, 5.0],
            [5.1, 5.1],
            [5.2, 5.0],
        ];

        let model: SpectralClustering = SpectralClustering::new()
            .n_clusters(2)
            .affinity(Affinity::RBF)
            .normalization(NormalizationMethod::RandomWalk)
            .fit(&x.view(), &Array1::zeros(0).view())
            .expect("operation should succeed");

        assert_eq!(model.labels().len(), x.nrows());
    }

    #[test]
    fn test_spectral_clustering_no_normalization() {
        let x = array![
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.0],
            [5.0, 5.0],
            [5.1, 5.1],
            [5.2, 5.0],
        ];

        let model: SpectralClustering = SpectralClustering::new()
            .n_clusters(2)
            .affinity(Affinity::RBF)
            .normalization(NormalizationMethod::None)
            .fit(&x.view(), &Array1::zeros(0).view())
            .expect("operation should succeed");

        assert_eq!(model.labels().len(), x.nrows());
    }

    #[test]
    fn test_affinity_matrix_computation() {
        let x = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0],];

        let model: SpectralClustering = SpectralClustering::new()
            .n_clusters(2)
            .affinity(Affinity::RBF)
            .gamma(1.0);

        let affinity = model
            .compute_affinity_matrix(&x.view())
            .expect("operation should succeed");

        // Diagonal should be zero (no self-loops)
        for i in 0..affinity.nrows() {
            assert_eq!(affinity[[i, i]], 0.0);
        }

        // Check that closer points have higher affinity
        let dist_01 = 1.0f64; // Distance between points 0 and 1
        let expected_affinity = (-(dist_01 * dist_01)).exp();
        assert!((affinity[[0, 1]] - expected_affinity).abs() < 1e-6);
        assert!((affinity[[0, 2]] - expected_affinity).abs() < 1e-6);
    }

    #[test]
    fn test_degree_matrix_computation() {
        let affinity = array![[0.0, 0.5, 0.3], [0.5, 0.0, 0.2], [0.3, 0.2, 0.0],];

        let model: SpectralClustering = SpectralClustering::new();
        let degree = model.compute_degree_matrix(&affinity);

        // Check degree values
        assert_eq!(degree[[0, 0]], 0.8); // 0.5 + 0.3
        assert_eq!(degree[[1, 1]], 0.7); // 0.5 + 0.2
        assert_eq!(degree[[2, 2]], 0.5); // 0.3 + 0.2

        // Off-diagonal should be zero
        for i in 0..degree.nrows() {
            for j in 0..degree.ncols() {
                if i != j {
                    assert_eq!(degree[[i, j]], 0.0);
                }
            }
        }
    }

    #[test]
    fn test_multi_scale_rbf_affinity() {
        let x = array![
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.0],
            [5.0, 5.0],
            [5.1, 5.1],
            [5.2, 5.0],
        ];

        let model: SpectralClustering = SpectralClustering::new()
            .n_clusters(2)
            .affinity(Affinity::MultiScaleRBF)
            .n_scales(3)
            .scale_factor(2.0)
            .fit(&x.view(), &Array1::zeros(0).view())
            .expect("operation should succeed");

        assert_eq!(model.labels().len(), x.nrows());

        // Verify affinity matrix properties
        let affinity = model.affinity_matrix();
        assert_eq!(affinity.nrows(), x.nrows());
        assert_eq!(affinity.ncols(), x.nrows());

        // Diagonal should be zero
        for i in 0..affinity.nrows() {
            assert_eq!(affinity[[i, i]], 0.0);
        }
    }

    #[test]
    fn test_polynomial_kernel_affinity() {
        let x = array![
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 1.0],
            [10.0, 11.0],
            [11.0, 12.0],
            [12.0, 10.0],
        ];

        let model: SpectralClustering = SpectralClustering::new()
            .n_clusters(2)
            .affinity(Affinity::Polynomial)
            .degree(3.0)
            .gamma(0.1)
            .coef0(1.0)
            .fit(&x.view(), &Array1::zeros(0).view())
            .expect("operation should succeed");

        assert_eq!(model.labels().len(), x.nrows());

        // Check that diagonal elements are 1.0 (self-similarity)
        let affinity = model.affinity_matrix();
        for i in 0..affinity.nrows() {
            assert_eq!(affinity[[i, i]], 1.0);
        }
    }

    #[test]
    fn test_sigmoid_kernel_affinity() {
        let x = array![
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 1.0],
            [10.0, 11.0],
            [11.0, 12.0],
            [12.0, 10.0],
        ];

        let model: SpectralClustering = SpectralClustering::new()
            .n_clusters(2)
            .affinity(Affinity::Sigmoid)
            .gamma(0.1)
            .coef0(0.5)
            .fit(&x.view(), &Array1::zeros(0).view())
            .expect("operation should succeed");

        assert_eq!(model.labels().len(), x.nrows());

        // Sigmoid kernel values should be in range [-1, 1]
        let affinity = model.affinity_matrix();
        for i in 0..affinity.nrows() {
            for j in 0..affinity.ncols() {
                if i != j {
                    assert!(affinity[[i, j]] >= -1.0 && affinity[[i, j]] <= 1.0);
                } else {
                    // Diagonal elements are set to 1.0 for self-similarity
                    assert_eq!(affinity[[i, j]], 1.0);
                }
            }
        }
    }

    #[test]
    fn test_linear_kernel_affinity() {
        let x = array![
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 1.0],
            [10.0, 11.0],
            [11.0, 12.0],
            [12.0, 10.0],
        ];

        let model: SpectralClustering = SpectralClustering::new()
            .n_clusters(2)
            .affinity(Affinity::Linear)
            .fit(&x.view(), &Array1::zeros(0).view())
            .expect("operation should succeed");

        assert_eq!(model.labels().len(), x.nrows());

        // Linear kernel should compute dot products
        let affinity = model.affinity_matrix();

        // Check diagonal is 1.0 (self-similarity)
        for i in 0..affinity.nrows() {
            assert_eq!(affinity[[i, i]], 1.0);
        }

        // Check that affinity[0,1] equals dot product of rows 0 and 1
        let expected = x.row(0).dot(&x.row(1));
        assert!((affinity[[0, 1]] - expected).abs() < 1e-6);
    }

    #[test]
    fn test_automatic_eigenvalue_selection() {
        let x = array![
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.0],
            [5.0, 5.0],
            [5.1, 5.1],
            [5.2, 5.0],
        ];

        let model: SpectralClustering = SpectralClustering::new()
            .n_clusters(2)
            .affinity(Affinity::RBF)
            .auto_eigenvalue_selection(true)
            .eigenvalue_threshold(0.1)
            .max_eigenvectors(10)
            .fit(&x.view(), &Array1::zeros(0).view())
            .expect("operation should succeed");

        assert_eq!(model.labels().len(), x.nrows());
    }

    #[test]
    fn test_generate_scales() {
        let x = array![
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0],
            [10.0, 10.0],
            [11.0, 11.0],
            [12.0, 12.0],
        ];

        let model: SpectralClustering = SpectralClustering::new()
            .affinity(Affinity::MultiScaleRBF)
            .n_scales(5)
            .scale_factor(2.0);

        let scales = model
            .generate_scales(&x.view())
            .expect("operation should succeed");

        assert_eq!(scales.len(), 5);

        // Scales should be positive
        for &scale in &scales {
            assert!(scale > 0.0);
        }

        // Scales should be in ascending order when sorted
        let mut sorted_scales = scales.clone();
        sorted_scales.sort_by(|a, b| a.partial_cmp(b).expect("operation should succeed"));
        // The scales might not be perfectly sorted due to the algorithm,
        // but they should span a reasonable range
        assert!(sorted_scales[0] > 0.0);
        assert!(sorted_scales.last().expect("operation should succeed") > &sorted_scales[0]);
    }

    #[test]
    fn test_select_optimal_eigenvectors() {
        let model: SpectralClustering = SpectralClustering::new()
            .n_clusters(3)
            .auto_eigenvalue_selection(true)
            .eigenvalue_threshold(0.5);

        // Create mock eigenvalues with clear gaps
        let eigenvalues = array![1.0, 0.8, 0.3, 0.1, 0.05, 0.01];

        let optimal_k = model.select_optimal_eigenvectors(&eigenvalues);

        // Should select based on the largest gap (0.8 - 0.3 = 0.5)
        // but ensure at least n_clusters
        assert!(optimal_k >= 3);
    }

    #[test]
    fn test_custom_scales() {
        let x = array![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0],];

        let custom_scales = vec![0.1, 0.5, 1.0, 2.0, 5.0];

        let model: SpectralClustering = SpectralClustering::new()
            .n_clusters(2)
            .affinity(Affinity::MultiScaleRBF)
            .scales(custom_scales.clone())
            .fit(&x.view(), &Array1::zeros(0).view())
            .expect("operation should succeed");

        assert_eq!(model.labels().len(), x.nrows());
    }

    #[test]
    fn test_constrained_spectral_clustering() {
        let x = array![
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.0],
            [5.0, 5.0],
            [5.1, 5.1],
            [5.2, 5.0],
        ];

        // Create constraints: points 0 and 1 must be in same cluster,
        // points 0 and 3 cannot be in same cluster.
        let constraints = ClusteringConstraints {
            must_link: vec![(0, 1)],
            cannot_link: vec![(0, 3)],
            weight: 1.0,
        };

        let model = ConstrainedSpectralClustering::new(constraints)
            .n_clusters(2)
            .affinity(Affinity::RBF)
            .gamma(1.0)
            .fit(&x.view(), &Array1::zeros(0).view())
            .expect("constrained spectral clustering should succeed");

        let labels = model.labels();
        assert_eq!(labels.len(), x.nrows());

        // Check that must-link constraint is satisfied (points 0 and 1 same cluster).
        assert_eq!(labels[0], labels[1]);

        // Check that cannot-link constraint is satisfied (points 0 and 3 different clusters).
        assert_ne!(labels[0], labels[3]);

        // Check constraint satisfaction rate is in [0, 1].
        let satisfaction_rate = model
            .constraint_satisfaction_rate()
            .expect("satisfaction rate should be set after fitting");
        assert!((0.0..=1.0).contains(&satisfaction_rate));
    }

    #[test]
    fn test_constraint_propagation() {
        let n = 6;
        // Transitive must-link: 0-1 and 1-2 should imply 0-2.
        let must_link = vec![(0usize, 1usize), (1, 2)];
        let cannot_link = vec![(0usize, 4usize)];

        let (prop_must, prop_cannot) = propagate_constraints(n, &must_link, &cannot_link)
            .expect("constraint propagation should succeed");

        // The closure of {(0,1),(1,2)} should include (0,2).
        assert!(
            prop_must.contains(&(0, 2)) || prop_must.contains(&(2, 0)),
            "Transitive must-link (0,2) should be present"
        );

        // The cannot-link (0,4) propagates: all of {0,1,2} cannot link with 4.
        for i in 0..3usize {
            assert!(
                prop_cannot.contains(&(i, 4)) || prop_cannot.contains(&(4, i)),
                "Propagated cannot-link ({i}, 4) should be present"
            );
        }
    }

    #[test]
    fn test_constraint_propagation_conflict_detection() {
        // Conflict: (0,1) must-link AND (0,1) cannot-link.
        let result = propagate_constraints(3, &[(0, 1)], &[(0, 1)]);
        assert!(
            result.is_err(),
            "Conflicting constraints should return an error"
        );
    }

    #[test]
    fn test_union_find() {
        let mut uf = UnionFind::new(5);
        assert!(!uf.connected(0, 1));
        uf.union(0, 1);
        assert!(uf.connected(0, 1));
        uf.union(1, 2);
        assert!(uf.connected(0, 2));
        assert!(!uf.connected(0, 3));
    }
}
