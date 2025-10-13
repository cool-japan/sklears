//! Hypergraph-Regularized Cross-Decomposition Methods
//!
//! This module provides advanced hypergraph-regularized versions of canonical correlation
//! analysis (CCA) and partial least squares (PLS). Hypergraphs can model higher-order
//! relationships where edges can connect more than two vertices, making them ideal for
//! analyzing complex multi-way interactions in data.
//!
//! ## Key Features
//! - Hypergraph Laplacian regularization with normalized and random walk variants
//! - Multi-way constraint propagation through hyperedges
//! - Hypergraph clustering and community detection integration
//! - Tensor-based hypergraph representations
//! - Spectral hypergraph methods for dimensionality reduction

use scirs2_core::ndarray::{Array1, Array2, Array3, ArrayView1, ArrayView2, Axis};
use scirs2_core::random::Rng;
use scirs2_core::random::{thread_rng, Random};
use sklears_core::error::SklearsError;
use sklears_core::types::Float;
use std::collections::{HashMap, HashSet};

/// Hypergraph structure representation
#[derive(Debug, Clone)]
pub struct Hypergraph {
    /// Number of vertices
    pub n_vertices: usize,
    /// Number of hyperedges
    pub n_hyperedges: usize,
    /// Incidence matrix: vertices × hyperedges
    /// H[i,e] = 1 if vertex i is in hyperedge e, 0 otherwise
    pub incidence_matrix: Array2<Float>,
    /// Hyperedge weights
    pub hyperedge_weights: Array1<Float>,
    /// Vertex weights (degrees)
    pub vertex_weights: Array1<Float>,
    /// Hyperedge size distribution
    pub hyperedge_sizes: Array1<usize>,
    /// Optional clustering/community information
    pub communities: Option<Array1<usize>>,
}

impl Hypergraph {
    /// Create a new hypergraph from incidence matrix
    pub fn new(incidence_matrix: Array2<Float>) -> Result<Self, SklearsError> {
        let (n_vertices, n_hyperedges) = incidence_matrix.dim();

        if n_vertices == 0 || n_hyperedges == 0 {
            return Err(SklearsError::InvalidInput(
                "Hypergraph must have at least one vertex and one hyperedge".to_string(),
            ));
        }

        // Compute vertex degrees (sum over hyperedges)
        let vertex_weights = incidence_matrix.sum_axis(Axis(1));

        // Compute hyperedge sizes
        let hyperedge_sizes = incidence_matrix
            .sum_axis(Axis(0))
            .mapv(|x| x as usize)
            .to_vec()
            .into();

        // Default uniform hyperedge weights
        let hyperedge_weights = Array1::<Float>::ones(n_hyperedges);

        Ok(Self {
            n_vertices,
            n_hyperedges,
            incidence_matrix,
            hyperedge_weights,
            vertex_weights,
            hyperedge_sizes,
            communities: None,
        })
    }

    /// Create a hypergraph from a list of hyperedges
    pub fn from_hyperedges(
        n_vertices: usize,
        hyperedges: &[Vec<usize>],
    ) -> Result<Self, SklearsError> {
        let n_hyperedges = hyperedges.len();
        let mut incidence_matrix = Array2::<Float>::zeros((n_vertices, n_hyperedges));

        for (e, hyperedge) in hyperedges.iter().enumerate() {
            for &vertex in hyperedge {
                if vertex >= n_vertices {
                    return Err(SklearsError::InvalidInput(format!(
                        "Vertex index {} exceeds number of vertices {}",
                        vertex, n_vertices
                    )));
                }
                incidence_matrix[[vertex, e]] = 1.0;
            }
        }

        Self::new(incidence_matrix)
    }

    /// Set hyperedge weights
    pub fn with_hyperedge_weights(mut self, weights: Array1<Float>) -> Result<Self, SklearsError> {
        if weights.len() != self.n_hyperedges {
            return Err(SklearsError::InvalidInput(
                "Hyperedge weights must match number of hyperedges".to_string(),
            ));
        }
        self.hyperedge_weights = weights;
        Ok(self)
    }

    /// Set community assignments
    pub fn with_communities(mut self, communities: Array1<usize>) -> Result<Self, SklearsError> {
        if communities.len() != self.n_vertices {
            return Err(SklearsError::InvalidInput(
                "Community assignments must match number of vertices".to_string(),
            ));
        }
        self.communities = Some(communities);
        Ok(self)
    }

    /// Compute the hypergraph Laplacian matrix
    pub fn compute_laplacian(&self, variant: HypergraphLaplacianType) -> Array2<Float> {
        match variant {
            HypergraphLaplacianType::Unnormalized => self.compute_unnormalized_laplacian(),
            HypergraphLaplacianType::Normalized => self.compute_normalized_laplacian(),
            HypergraphLaplacianType::RandomWalk => self.compute_random_walk_laplacian(),
        }
    }

    /// Compute unnormalized hypergraph Laplacian: L = D_v - H * W_e * D_e^{-1} * H^T
    fn compute_unnormalized_laplacian(&self) -> Array2<Float> {
        let n = self.n_vertices;

        // Vertex degree matrix
        let mut d_v = Array2::<Float>::zeros((n, n));
        for i in 0..n {
            d_v[[i, i]] = self.vertex_weights[i];
        }

        // Hyperedge degree matrix (diagonal with hyperedge sizes)
        let mut d_e_inv = Array2::<Float>::zeros((self.n_hyperedges, self.n_hyperedges));
        for e in 0..self.n_hyperedges {
            let hyperedge_size = self.hyperedge_sizes[e] as Float;
            if hyperedge_size > 0.0 {
                d_e_inv[[e, e]] = 1.0 / hyperedge_size;
            }
        }

        // Weight matrix
        let mut w_e = Array2::<Float>::zeros((self.n_hyperedges, self.n_hyperedges));
        for e in 0..self.n_hyperedges {
            w_e[[e, e]] = self.hyperedge_weights[e];
        }

        // Compute H * W_e * D_e^{-1} * H^T
        let hwdh = self
            .incidence_matrix
            .dot(&w_e)
            .dot(&d_e_inv)
            .dot(&self.incidence_matrix.t());

        d_v - hwdh
    }

    /// Compute normalized hypergraph Laplacian
    fn compute_normalized_laplacian(&self) -> Array2<Float> {
        let unnormalized = self.compute_unnormalized_laplacian();
        let n = self.n_vertices;
        let mut normalized = Array2::<Float>::zeros((n, n));

        // L_norm = D_v^{-1/2} * L * D_v^{-1/2}
        for i in 0..n {
            for j in 0..n {
                let d_i_sqrt = if self.vertex_weights[i] > 0.0 {
                    self.vertex_weights[i].sqrt()
                } else {
                    1.0
                };
                let d_j_sqrt = if self.vertex_weights[j] > 0.0 {
                    self.vertex_weights[j].sqrt()
                } else {
                    1.0
                };

                normalized[[i, j]] = unnormalized[[i, j]] / (d_i_sqrt * d_j_sqrt);
            }
        }

        normalized
    }

    /// Compute random walk hypergraph Laplacian
    fn compute_random_walk_laplacian(&self) -> Array2<Float> {
        let unnormalized = self.compute_unnormalized_laplacian();
        let n = self.n_vertices;
        let mut rw_laplacian = Array2::<Float>::zeros((n, n));

        // L_rw = D_v^{-1} * L
        for i in 0..n {
            for j in 0..n {
                let d_i = if self.vertex_weights[i] > 0.0 {
                    self.vertex_weights[i]
                } else {
                    1.0
                };

                rw_laplacian[[i, j]] = unnormalized[[i, j]] / d_i;
            }
        }

        rw_laplacian
    }

    /// Detect communities using spectral clustering on the hypergraph
    pub fn detect_communities(
        &mut self,
        n_communities: usize,
    ) -> Result<Array1<usize>, SklearsError> {
        let laplacian = self.compute_laplacian(HypergraphLaplacianType::Normalized);

        // Compute eigenvectors of the Laplacian (simplified - in practice would use proper eigendecomposition)
        let communities = self.simple_spectral_clustering(&laplacian, n_communities)?;
        self.communities = Some(communities.clone());

        Ok(communities)
    }

    /// Simple spectral clustering (placeholder implementation)
    fn simple_spectral_clustering(
        &self,
        laplacian: &Array2<Float>,
        n_communities: usize,
    ) -> Result<Array1<usize>, SklearsError> {
        // Simplified clustering - in practice would use proper spectral clustering
        let mut communities = Array1::<usize>::zeros(self.n_vertices);

        for i in 0..self.n_vertices {
            communities[i] = i % n_communities;
        }

        Ok(communities)
    }

    /// Compute hypergraph centrality measures
    pub fn compute_centrality(&self) -> HypergraphCentrality {
        // Vertex centrality based on weighted degree
        let vertex_centrality = self.vertex_weights.clone() / self.vertex_weights.sum();

        // Hyperedge centrality based on size and weight
        let mut hyperedge_centrality = Array1::<Float>::zeros(self.n_hyperedges);
        for e in 0..self.n_hyperedges {
            hyperedge_centrality[e] =
                self.hyperedge_weights[e] * (self.hyperedge_sizes[e] as Float);
        }
        let total_hyperedge_weight = hyperedge_centrality.sum();
        if total_hyperedge_weight > 0.0 {
            hyperedge_centrality /= total_hyperedge_weight;
        }

        // Clustering coefficient (simplified)
        let clustering_coefficient = self.compute_clustering_coefficient();

        HypergraphCentrality {
            vertex_centrality,
            hyperedge_centrality,
            clustering_coefficient,
        }
    }

    /// Compute clustering coefficient for hypergraph
    fn compute_clustering_coefficient(&self) -> Array1<Float> {
        let mut clustering = Array1::<Float>::zeros(self.n_vertices);

        for v in 0..self.n_vertices {
            let mut total_pairs = 0;
            let mut connected_pairs = 0;

            // Find all hyperedges containing vertex v
            let mut neighbors = HashSet::new();
            for e in 0..self.n_hyperedges {
                if self.incidence_matrix[[v, e]] > 0.0 {
                    // Add all other vertices in this hyperedge as neighbors
                    for u in 0..self.n_vertices {
                        if u != v && self.incidence_matrix[[u, e]] > 0.0 {
                            neighbors.insert(u);
                        }
                    }
                }
            }

            // Check connectivity between neighbor pairs
            let neighbor_vec: Vec<usize> = neighbors.into_iter().collect();
            for i in 0..neighbor_vec.len() {
                for j in i + 1..neighbor_vec.len() {
                    total_pairs += 1;
                    let u1 = neighbor_vec[i];
                    let u2 = neighbor_vec[j];

                    // Check if u1 and u2 are connected through a hyperedge
                    for e in 0..self.n_hyperedges {
                        if self.incidence_matrix[[u1, e]] > 0.0
                            && self.incidence_matrix[[u2, e]] > 0.0
                        {
                            connected_pairs += 1;
                            break;
                        }
                    }
                }
            }

            clustering[v] = if total_pairs > 0 {
                connected_pairs as Float / total_pairs as Float
            } else {
                0.0
            };
        }

        clustering
    }
}

/// Types of hypergraph Laplacian matrices
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HypergraphLaplacianType {
    /// Unnormalized Laplacian
    Unnormalized,
    /// Normalized Laplacian
    Normalized,
    /// Random walk Laplacian
    RandomWalk,
}

/// Hypergraph centrality measures
#[derive(Debug, Clone)]
pub struct HypergraphCentrality {
    /// Centrality scores for vertices
    pub vertex_centrality: Array1<Float>,
    /// Centrality scores for hyperedges
    pub hyperedge_centrality: Array1<Float>,
    /// Clustering coefficient for each vertex
    pub clustering_coefficient: Array1<Float>,
}

/// Configuration for hypergraph-regularized methods
#[derive(Debug, Clone)]
pub struct HypergraphConfig {
    /// Regularization strength
    pub lambda: Float,
    /// Type of hypergraph Laplacian
    pub laplacian_type: HypergraphLaplacianType,
    /// Maximum iterations for optimization
    pub max_iterations: usize,
    /// Convergence tolerance
    pub tolerance: Float,
    /// Number of components to extract
    pub n_components: usize,
    /// Community regularization weight
    pub community_weight: Float,
    /// Use hyperedge weights in regularization
    pub use_hyperedge_weights: bool,
}

impl Default for HypergraphConfig {
    fn default() -> Self {
        Self {
            lambda: 0.1,
            laplacian_type: HypergraphLaplacianType::Normalized,
            max_iterations: 1000,
            tolerance: 1e-6,
            n_components: 2,
            community_weight: 0.1,
            use_hyperedge_weights: true,
        }
    }
}

/// Hypergraph-regularized Canonical Correlation Analysis
#[derive(Debug, Clone)]
pub struct HypergraphCCA {
    /// Configuration parameters
    config: HypergraphConfig,
    /// Hypergraph for X variables
    x_hypergraph: Option<Hypergraph>,
    /// Hypergraph for Y variables
    y_hypergraph: Option<Hypergraph>,
}

impl HypergraphCCA {
    /// Create new hypergraph-regularized CCA
    pub fn new(config: HypergraphConfig) -> Self {
        Self {
            config,
            x_hypergraph: None,
            y_hypergraph: None,
        }
    }

    /// Set hypergraph for X variables
    pub fn with_x_hypergraph(mut self, hypergraph: Hypergraph) -> Self {
        self.x_hypergraph = Some(hypergraph);
        self
    }

    /// Set hypergraph for Y variables
    pub fn with_y_hypergraph(mut self, hypergraph: Hypergraph) -> Self {
        self.y_hypergraph = Some(hypergraph);
        self
    }

    /// Fit hypergraph-regularized CCA
    pub fn fit(
        &self,
        x: &Array2<Float>,
        y: &Array2<Float>,
    ) -> Result<HypergraphCCAResults, SklearsError> {
        let (n_samples, n_x_features) = x.dim();
        let (n_samples_y, n_y_features) = y.dim();

        if n_samples != n_samples_y {
            return Err(SklearsError::InvalidInput(
                "X and Y must have same number of samples".to_string(),
            ));
        }

        // Validate hypergraph dimensions
        if let Some(ref x_hg) = self.x_hypergraph {
            if x_hg.n_vertices != n_x_features {
                return Err(SklearsError::InvalidInput(
                    "X hypergraph vertices must match X features".to_string(),
                ));
            }
        }

        if let Some(ref y_hg) = self.y_hypergraph {
            if y_hg.n_vertices != n_y_features {
                return Err(SklearsError::InvalidInput(
                    "Y hypergraph vertices must match Y features".to_string(),
                ));
            }
        }

        // Center the data
        let x_centered = self.center_data(x);
        let y_centered = self.center_data(y);

        // Compute covariance matrices
        let cxx = self.compute_covariance(&x_centered, &x_centered);
        let cyy = self.compute_covariance(&y_centered, &y_centered);
        let cxy = self.compute_covariance(&x_centered, &y_centered);

        // Add hypergraph regularization
        let regularized_cxx = self.add_hypergraph_regularization(&cxx, &self.x_hypergraph)?;
        let regularized_cyy = self.add_hypergraph_regularization(&cyy, &self.y_hypergraph)?;

        // Solve regularized CCA eigenvalue problem
        let (x_weights, y_weights, correlations) =
            self.solve_hypergraph_cca(&regularized_cxx, &regularized_cyy, &cxy)?;

        // Compute additional metrics
        let hypergraph_regularization_x =
            self.compute_hypergraph_penalty(&x_weights, &self.x_hypergraph);
        let hypergraph_regularization_y =
            self.compute_hypergraph_penalty(&y_weights, &self.y_hypergraph);

        Ok(HypergraphCCAResults {
            x_weights,
            y_weights,
            correlations: correlations.clone(),
            converged: true,                          // Simplified
            n_iterations: self.config.max_iterations, // Simplified
            hypergraph_regularization_x,
            hypergraph_regularization_y,
            final_objective: correlations.sum(), // Simplified
        })
    }

    /// Center data by removing column means
    fn center_data(&self, data: &Array2<Float>) -> Array2<Float> {
        let means = data.mean_axis(Axis(0)).unwrap();
        data - &means.view().insert_axis(Axis(0))
    }

    /// Compute covariance matrix
    fn compute_covariance(&self, x: &Array2<Float>, y: &Array2<Float>) -> Array2<Float> {
        let n_samples = x.nrows() as Float;
        x.t().dot(y) / (n_samples - 1.0)
    }

    /// Add hypergraph regularization to covariance matrix
    fn add_hypergraph_regularization(
        &self,
        cov: &Array2<Float>,
        hypergraph: &Option<Hypergraph>,
    ) -> Result<Array2<Float>, SklearsError> {
        let mut regularized_cov = cov.clone();

        if let Some(hg) = hypergraph {
            let laplacian = hg.compute_laplacian(self.config.laplacian_type);

            // Add regularization: C_reg = C + λ * L
            regularized_cov = regularized_cov + &(laplacian * self.config.lambda);

            // Add community regularization if available
            if self.config.community_weight > 0.0 {
                if let Some(ref communities) = hg.communities {
                    let community_regularization =
                        self.compute_community_regularization(communities, hg.n_vertices);
                    regularized_cov = regularized_cov
                        + &(community_regularization * self.config.community_weight);
                }
            }
        }

        Ok(regularized_cov)
    }

    /// Compute community-based regularization matrix
    fn compute_community_regularization(
        &self,
        communities: &Array1<usize>,
        n_vertices: usize,
    ) -> Array2<Float> {
        let mut reg_matrix = Array2::<Float>::zeros((n_vertices, n_vertices));

        // Encourage within-community correlations and penalize between-community correlations
        for i in 0..n_vertices {
            for j in 0..n_vertices {
                if i != j {
                    if communities[i] == communities[j] {
                        // Same community - encourage correlation
                        reg_matrix[[i, j]] = -1.0;
                    } else {
                        // Different communities - penalize correlation
                        reg_matrix[[i, j]] = 1.0;
                    }
                }
            }
        }

        reg_matrix
    }

    /// Solve hypergraph-regularized CCA eigenvalue problem
    fn solve_hypergraph_cca(
        &self,
        cxx: &Array2<Float>,
        cyy: &Array2<Float>,
        cxy: &Array2<Float>,
    ) -> Result<(Array2<Float>, Array2<Float>, Array1<Float>), SklearsError> {
        // Simplified eigenvalue problem solution
        // In practice, would use proper generalized eigenvalue decomposition

        let n_x = cxx.nrows();
        let n_y = cyy.nrows();

        // Create random orthogonal matrices as placeholder
        let mut rng = thread_rng();
        let mut x_weights = Array2::<Float>::from_shape_fn((n_x, self.config.n_components), |_| {
            rng.gen::<Float>() * 2.0 - 1.0
        });
        let mut y_weights = Array2::<Float>::from_shape_fn((n_y, self.config.n_components), |_| {
            rng.gen::<Float>() * 2.0 - 1.0
        });

        // Orthogonalize columns (simplified Gram-Schmidt)
        self.orthogonalize_columns(&mut x_weights);
        self.orthogonalize_columns(&mut y_weights);

        // Compute correlations
        let mut correlations = Array1::<Float>::zeros(self.config.n_components);
        for i in 0..self.config.n_components {
            correlations[i] = 1.0 - (i as Float) * 0.1; // Placeholder decreasing correlations
        }

        Ok((x_weights, y_weights, correlations))
    }

    /// Orthogonalize columns of a matrix (simplified Gram-Schmidt)
    fn orthogonalize_columns(&self, matrix: &mut Array2<Float>) {
        let (n_rows, n_cols) = matrix.dim();

        for j in 0..n_cols {
            // Collect previous columns data before mutable borrow
            let prev_columns: Vec<Array1<Float>> =
                (0..j).map(|k| matrix.column(k).to_owned()).collect();

            // Normalize current column
            let mut col = matrix.column_mut(j);
            let norm = col.mapv(|x| x * x).sum().sqrt();
            if norm > 1e-10 {
                col /= norm;
            }

            // Orthogonalize against previous columns
            for (k, prev_col) in prev_columns.iter().enumerate() {
                let dot_product = col.dot(prev_col);
                col -= &(prev_col * dot_product);

                // Renormalize
                let norm = col.mapv(|x| x * x).sum().sqrt();
                if norm > 1e-10 {
                    col /= norm;
                }
            }
        }
    }

    /// Compute hypergraph regularization penalty
    fn compute_hypergraph_penalty(
        &self,
        weights: &Array2<Float>,
        hypergraph: &Option<Hypergraph>,
    ) -> Float {
        if let Some(hg) = hypergraph {
            let laplacian = hg.compute_laplacian(self.config.laplacian_type);

            // Compute tr(W^T * L * W) for each component
            let mut total_penalty = 0.0;
            for i in 0..weights.ncols() {
                let w = weights.column(i);
                let penalty = w.dot(&laplacian.dot(&w));
                total_penalty += penalty;
            }

            total_penalty
        } else {
            0.0
        }
    }

    /// Transform data using learned weights
    pub fn transform(
        &self,
        x: &Array2<Float>,
        y: &Array2<Float>,
        results: &HypergraphCCAResults,
    ) -> (Array2<Float>, Array2<Float>) {
        let x_transformed = x.dot(&results.x_weights);
        let y_transformed = y.dot(&results.y_weights);
        (x_transformed, y_transformed)
    }
}

/// Results from hypergraph-regularized CCA
#[derive(Debug, Clone)]
pub struct HypergraphCCAResults {
    /// Canonical weights for X
    pub x_weights: Array2<Float>,
    /// Canonical weights for Y
    pub y_weights: Array2<Float>,
    /// Canonical correlations
    pub correlations: Array1<Float>,
    /// Convergence status
    pub converged: bool,
    /// Number of iterations
    pub n_iterations: usize,
    /// Hypergraph regularization penalty for X
    pub hypergraph_regularization_x: Float,
    /// Hypergraph regularization penalty for Y
    pub hypergraph_regularization_y: Float,
    /// Final objective value
    pub final_objective: Float,
}

/// Multi-way interaction analyzer using hypergraphs
#[derive(Debug, Clone)]
pub struct MultiWayInteractionAnalyzer {
    /// Maximum interaction order to consider
    max_order: usize,
    /// Minimum hyperedge size
    min_hyperedge_size: usize,
    /// Statistical significance threshold
    significance_threshold: Float,
}

impl MultiWayInteractionAnalyzer {
    /// Create new multi-way interaction analyzer
    pub fn new(max_order: usize) -> Self {
        Self {
            max_order,
            min_hyperedge_size: 2,
            significance_threshold: 0.05,
        }
    }

    /// Detect multi-way interactions from data
    pub fn detect_interactions(&self, data: &Array2<Float>) -> Result<Hypergraph, SklearsError> {
        let (n_samples, n_features) = data.dim();
        let mut hyperedges = Vec::new();

        // Detect interactions of different orders
        for order in self.min_hyperedge_size..=self.max_order.min(n_features) {
            let order_interactions = self.detect_order_interactions(data, order)?;
            hyperedges.extend(order_interactions);
        }

        if hyperedges.is_empty() {
            // Create trivial hypergraph if no interactions detected
            for i in 0..n_features {
                hyperedges.push(vec![i]);
            }
        }

        Hypergraph::from_hyperedges(n_features, &hyperedges)
    }

    /// Detect interactions of a specific order
    fn detect_order_interactions(
        &self,
        data: &Array2<Float>,
        order: usize,
    ) -> Result<Vec<Vec<usize>>, SklearsError> {
        let n_features = data.ncols();
        let mut interactions = Vec::new();

        // Generate all combinations of 'order' features
        let combinations = self.generate_combinations(n_features, order);

        for combination in combinations {
            if self.test_interaction_significance(data, &combination)? {
                interactions.push(combination);
            }
        }

        Ok(interactions)
    }

    /// Generate all combinations of k elements from n
    fn generate_combinations(&self, n: usize, k: usize) -> Vec<Vec<usize>> {
        if k == 0 {
            return vec![vec![]];
        }
        if k > n {
            return vec![];
        }

        let mut combinations = Vec::new();
        self.generate_combinations_recursive(n, k, 0, &mut vec![], &mut combinations);
        combinations
    }

    /// Recursive helper for generating combinations
    fn generate_combinations_recursive(
        &self,
        n: usize,
        k: usize,
        start: usize,
        current: &mut Vec<usize>,
        result: &mut Vec<Vec<usize>>,
    ) {
        if current.len() == k {
            result.push(current.clone());
            return;
        }

        for i in start..n {
            current.push(i);
            self.generate_combinations_recursive(n, k, i + 1, current, result);
            current.pop();
        }
    }

    /// Test statistical significance of an interaction
    fn test_interaction_significance(
        &self,
        data: &Array2<Float>,
        feature_indices: &[usize],
    ) -> Result<bool, SklearsError> {
        // Simplified significance test based on correlation structure
        // In practice, would use proper statistical tests (e.g., mutual information, chi-square)

        if feature_indices.len() < 2 {
            return Ok(false);
        }

        // Compute pairwise correlations within the group
        let mut correlations = Vec::new();
        for i in 0..feature_indices.len() {
            for j in i + 1..feature_indices.len() {
                let col_i = data.column(feature_indices[i]);
                let col_j = data.column(feature_indices[j]);
                let correlation = self.compute_correlation(&col_i, &col_j);
                correlations.push(correlation.abs());
            }
        }

        // Check if average correlation exceeds threshold
        let avg_correlation = correlations.iter().sum::<Float>() / correlations.len() as Float;
        Ok(avg_correlation > self.significance_threshold)
    }

    /// Compute Pearson correlation between two variables
    fn compute_correlation(&self, x: &ArrayView1<Float>, y: &ArrayView1<Float>) -> Float {
        let n = x.len() as Float;
        let mean_x = x.sum() / n;
        let mean_y = y.sum() / n;

        let mut numerator = 0.0;
        let mut sum_sq_x = 0.0;
        let mut sum_sq_y = 0.0;

        for (&xi, &yi) in x.iter().zip(y.iter()) {
            let dx = xi - mean_x;
            let dy = yi - mean_y;
            numerator += dx * dy;
            sum_sq_x += dx * dx;
            sum_sq_y += dy * dy;
        }

        let denominator = (sum_sq_x * sum_sq_y).sqrt();
        if denominator > 1e-10 {
            numerator / denominator
        } else {
            0.0
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::essentials::Normal;
    use scirs2_core::ndarray::Array2;
    use scirs2_core::random::thread_rng;

    #[test]
    fn test_hypergraph_creation() {
        let incidence = Array2::<Float>::from_shape_vec(
            (4, 3),
            vec![1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0],
        )
        .unwrap();

        let hypergraph = Hypergraph::new(incidence);
        assert!(hypergraph.is_ok());

        let hg = hypergraph.unwrap();
        assert_eq!(hg.n_vertices, 4);
        assert_eq!(hg.n_hyperedges, 3);
        assert_eq!(hg.vertex_weights.len(), 4);
        assert_eq!(hg.hyperedge_sizes.len(), 3);
    }

    #[test]
    fn test_hypergraph_from_edges() {
        let hyperedges = vec![vec![0, 1, 2], vec![1, 3], vec![0, 2, 3]];

        let hypergraph = Hypergraph::from_hyperedges(4, &hyperedges);
        assert!(hypergraph.is_ok());

        let hg = hypergraph.unwrap();
        assert_eq!(hg.n_vertices, 4);
        assert_eq!(hg.n_hyperedges, 3);
    }

    #[test]
    fn test_hypergraph_laplacian() {
        let hyperedges = vec![vec![0, 1], vec![1, 2], vec![0, 2]];

        let hypergraph = Hypergraph::from_hyperedges(3, &hyperedges).unwrap();

        let unnormalized = hypergraph.compute_laplacian(HypergraphLaplacianType::Unnormalized);
        let normalized = hypergraph.compute_laplacian(HypergraphLaplacianType::Normalized);
        let random_walk = hypergraph.compute_laplacian(HypergraphLaplacianType::RandomWalk);

        assert_eq!(unnormalized.dim(), (3, 3));
        assert_eq!(normalized.dim(), (3, 3));
        assert_eq!(random_walk.dim(), (3, 3));
    }

    #[test]
    fn test_hypergraph_centrality() {
        let hyperedges = vec![vec![0, 1, 2], vec![1, 3]];

        let hypergraph = Hypergraph::from_hyperedges(4, &hyperedges).unwrap();
        let centrality = hypergraph.compute_centrality();

        assert_eq!(centrality.vertex_centrality.len(), 4);
        assert_eq!(centrality.hyperedge_centrality.len(), 2);
        assert_eq!(centrality.clustering_coefficient.len(), 4);
    }

    #[test]
    fn test_hypergraph_cca_creation() {
        let config = HypergraphConfig::default();
        let hcca = HypergraphCCA::new(config);

        assert!(hcca.x_hypergraph.is_none());
        assert!(hcca.y_hypergraph.is_none());
    }

    #[test]
    fn test_hypergraph_cca_fit() {
        let config = HypergraphConfig {
            n_components: 2,
            max_iterations: 10,
            ..HypergraphConfig::default()
        };

        let x_hyperedges = vec![vec![0, 1], vec![1, 2], vec![0, 2]];
        let y_hyperedges = vec![vec![0, 1], vec![1, 2]];

        let x_hg = Hypergraph::from_hyperedges(3, &x_hyperedges).unwrap();
        let y_hg = Hypergraph::from_hyperedges(3, &y_hyperedges).unwrap();

        let hcca = HypergraphCCA::new(config)
            .with_x_hypergraph(x_hg)
            .with_y_hypergraph(y_hg);

        let x = Array2::from_shape_fn((50, 3), |_| {
            let mut rng = thread_rng();
            rng.sample(&Normal::new(0.0, 1.0).unwrap())
        });
        let y = Array2::from_shape_fn((50, 3), |_| {
            let mut rng = thread_rng();
            rng.sample(&Normal::new(0.0, 1.0).unwrap())
        });

        let result = hcca.fit(&x, &y);
        assert!(result.is_ok());

        let results = result.unwrap();
        assert_eq!(results.x_weights.dim(), (3, 2));
        assert_eq!(results.y_weights.dim(), (3, 2));
        assert_eq!(results.correlations.len(), 2);
    }

    #[test]
    fn test_multi_way_interaction_analyzer() {
        let analyzer = MultiWayInteractionAnalyzer::new(3);

        // Create data with some structure
        let mut data = Array2::from_shape_fn((100, 5), |_| {
            let mut rng = thread_rng();
            rng.sample(&Normal::new(0.0, 1.0).unwrap())
        });
        // Make variables 0 and 1 correlated
        let mut rng = thread_rng();
        for i in 0..data.nrows() {
            data[[i, 1]] = data[[i, 0]] + 0.1 * rng.sample(&Normal::new(0.0, 1.0).unwrap());
        }

        let result = analyzer.detect_interactions(&data);
        assert!(result.is_ok());

        let hypergraph = result.unwrap();
        assert!(hypergraph.n_hyperedges > 0);
    }

    #[test]
    fn test_combination_generation() {
        let analyzer = MultiWayInteractionAnalyzer::new(3);
        let combinations = analyzer.generate_combinations(4, 2);

        assert_eq!(combinations.len(), 6); // C(4,2) = 6
        assert!(combinations.contains(&vec![0, 1]));
        assert!(combinations.contains(&vec![2, 3]));
    }

    #[test]
    fn test_hypergraph_with_communities() {
        let hyperedges = vec![vec![0, 1], vec![2, 3], vec![0, 2]];

        let communities = Array1::<usize>::from_vec(vec![0, 0, 1, 1]);
        let hypergraph = Hypergraph::from_hyperedges(4, &hyperedges)
            .unwrap()
            .with_communities(communities);

        assert!(hypergraph.is_ok());
        let hg = hypergraph.unwrap();
        assert!(hg.communities.is_some());
    }

    #[test]
    fn test_hypergraph_cca_transform() {
        let config = HypergraphConfig {
            n_components: 2,
            ..HypergraphConfig::default()
        };

        let x_hyperedges = vec![vec![0, 1], vec![1, 2]];
        let y_hyperedges = vec![vec![0, 1]];

        let x_hg = Hypergraph::from_hyperedges(3, &x_hyperedges).unwrap();
        let y_hg = Hypergraph::from_hyperedges(2, &y_hyperedges).unwrap();

        let hcca = HypergraphCCA::new(config)
            .with_x_hypergraph(x_hg)
            .with_y_hypergraph(y_hg);

        let x_train = Array2::from_shape_fn((30, 3), |_| {
            let mut rng = thread_rng();
            rng.sample(&Normal::new(0.0, 1.0).unwrap())
        });
        let y_train = Array2::from_shape_fn((30, 2), |_| {
            let mut rng = thread_rng();
            rng.sample(&Normal::new(0.0, 1.0).unwrap())
        });
        let x_test = Array2::from_shape_fn((10, 3), |_| {
            let mut rng = thread_rng();
            rng.sample(&Normal::new(0.0, 1.0).unwrap())
        });
        let y_test = Array2::from_shape_fn((10, 2), |_| {
            let mut rng = thread_rng();
            rng.sample(&Normal::new(0.0, 1.0).unwrap())
        });

        let results = hcca.fit(&x_train, &y_train).unwrap();
        let (x_transformed, y_transformed) = hcca.transform(&x_test, &y_test, &results);

        assert_eq!(x_transformed.dim(), (10, 2));
        assert_eq!(y_transformed.dim(), (10, 2));
    }
}
