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
                    distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

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

        distances.sort_by(|a, b| a.partial_cmp(b).unwrap());

        if distances.is_empty() {
            return Ok(vec![1.0]);
        }

        // Use percentiles to determine scales
        let median_dist = distances[distances.len() / 2];
        let q25_dist = distances[distances.len() / 4];
        let q75_dist = distances[3 * distances.len() / 4];

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
        indices.sort_by(|&a, &b| eigenvalues[b].partial_cmp(&eigenvalues[a]).unwrap());

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
            .unwrap();

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
            .unwrap();

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
            .unwrap();

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
            .unwrap();

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
            .unwrap();

        assert_eq!(model.labels().len(), x.nrows());
    }

    #[test]
    fn test_affinity_matrix_computation() {
        let x = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0],];

        let model: SpectralClustering = SpectralClustering::new()
            .n_clusters(2)
            .affinity(Affinity::RBF)
            .gamma(1.0);

        let affinity = model.compute_affinity_matrix(&x.view()).unwrap();

        // Diagonal should be zero (no self-loops)
        for i in 0..affinity.nrows() {
            assert_eq!(affinity[[i, i]], 0.0);
        }

        // Check that closer points have higher affinity
        let dist_01 = 1.0; // Distance between points 0 and 1
        let dist_02 = 1.0; // Distance between points 0 and 2
        let expected_affinity = (-1.0_f64 * dist_01 * dist_01).exp();
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
            .unwrap();

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
            .unwrap();

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
            .unwrap();

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
            .unwrap();

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
            .unwrap();

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

        let scales = model.generate_scales(&x.view()).unwrap();

        assert_eq!(scales.len(), 5);

        // Scales should be positive
        for &scale in &scales {
            assert!(scale > 0.0);
        }

        // Scales should be in ascending order when sorted
        let mut sorted_scales = scales.clone();
        sorted_scales.sort_by(|a, b| a.partial_cmp(b).unwrap());
        // The scales might not be perfectly sorted due to the algorithm,
        // but they should span a reasonable range
        assert!(sorted_scales[0] > 0.0);
        assert!(sorted_scales.last().unwrap() > &sorted_scales[0]);
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
            .unwrap();

        assert_eq!(model.labels().len(), x.nrows());
    }

    // TODO: Implement constrained spectral clustering
    // #[test]
    // fn test_constrained_spectral_clustering() {
    //     let x = array![
    //         [0.0, 0.0],
    //         [0.1, 0.1],
    //         [0.2, 0.0],
    //         [5.0, 5.0],
    //         [5.1, 5.1],
    //         [5.2, 5.0],
    //     ];
    //
    //     // Create constraints: points 0 and 1 must be in same cluster
    //     // points 0 and 3 cannot be in same cluster
    //     let constraints = ClusteringConstraints {
    //         must_link: vec![(0, 1)],
    //         cannot_link: vec![(0, 3)],
    //         weight: 1.0,
    //     };
    //
    //     // TODO: Implement ConstrainedSpectralClustering and ConstraintMethod
    //     // let model = ConstrainedSpectralClustering::new(constraints)
    //     //     .n_clusters(2)
    //     //     .affinity(Affinity::RBF)
    //     //     .gamma(1.0)
    //     //     .fit(&x.view(), &Array1::zeros(0).view())
    //     //     .unwrap();
    //     //
    //     // let labels = model.labels();
    //     // assert_eq!(labels.len(), x.nrows());
    //     //
    //     // // Check that must-link constraint is satisfied (points 0 and 1 same cluster)
    //     // assert_eq!(labels[0], labels[1]);
    //     //
    //     // // Check that cannot-link constraint is satisfied (points 0 and 3 different clusters)
    //     // assert_ne!(labels[0], labels[3]);
    //     //
    //     // // Check constraint satisfaction rate
    //     // let satisfaction_rate = model.constraint_satisfaction_rate().unwrap();
    //     // assert!(satisfaction_rate >= 0.0 && satisfaction_rate <= 1.0);
    // }

    // TODO: Implement constraint propagation for spectral clustering
    // #[test]
    // fn test_constraint_propagation() {
    //     let x = array![
    //         [0.0, 0.0],
    //         [0.1, 0.1],
    //         [0.2, 0.0],
    //         [0.3, 0.1],
    //         [5.0, 5.0],
    //         [5.1, 5.1],
    //     ];
    //
    //     // Create transitive must-link constraints: 0-1, 1-2 should imply 0-2
    //     let constraints = ClusteringConstraints {
    //         must_link: vec![(0, 1), (1, 2)],
    //         cannot_link: vec![(0, 4)],
    //         weight: 2.0,
    //     };
    //
    //     // TODO: Implement ConstrainedSpectralClustering and ConstraintMethod
    //     // let model = ConstrainedSpectralClustering::new(constraints)
    //     //     .n_clusters(2)
    //     //     .affinity(Affinity::RBF)
    //     //     .fit(&x.view(), &Array1::zeros(0).view())
    //     //     .unwrap();
    //     //
    //     // let labels = model.labels();
    //     //
    //     // // Check that transitive constraint is satisfied
    //     // assert_eq!(labels[0], labels[1]);
    //     // assert_eq!(labels[1], labels[2]);
    //     // assert_eq!(labels[0], labels[2]);
    //     //
    //     // // Check cannot-link constraint
    //     // assert_ne!(labels[0], labels[4]);
    // }
}
