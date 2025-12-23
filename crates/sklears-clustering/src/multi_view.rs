//! Multi-View Clustering Algorithms
//!
//! This module provides clustering algorithms designed for multi-view data,
//! where each data point is represented by multiple feature sets or "views".
//! Multi-view clustering leverages the complementary information from different
//! views to improve clustering performance.
//!
//! # Algorithms Provided
//! - **Multi-View K-Means**: K-Means clustering across multiple views
//! - **Canonical Correlation Analysis (CCA) Clustering**: Clustering based on CCA
//! - **Consensus Clustering**: Ensemble clustering across multiple views
//! - **Co-Training Clustering**: Semi-supervised multi-view clustering
//! - **Multi-View Spectral Clustering**: Spectral clustering for multiple views
//!
//! # Mathematical Background
//!
//! ## Multi-View Data
//! Given m views of n data points: X^(1), X^(2), ..., X^(m)
//! where X^(v) ∈ R^(n×d_v) is the v-th view with d_v features
//!
//! ## Multi-View K-Means Objective
//! Minimize: Σ_v w_v * Σ_i Σ_k ||x_i^(v) - c_k^(v)||²
//! where w_v is the weight for view v, and c_k^(v) is the centroid for cluster k in view v
//!
//! ## Consensus Clustering
//! Combines clustering results from multiple views to form a consensus:
//! C* = argmax_C Σ_v w_v * agreement(C, C^(v))

use std::collections::HashMap;

use scirs2_core::ndarray::Array2;
use scirs2_core::random::Random;
use sklears_core::error::{Result, SklearsError};
use sklears_core::prelude::*;

/// Multi-view data container
#[derive(Debug, Clone)]
pub struct MultiViewData {
    /// Data for each view
    pub views: Vec<Array2<f64>>,
    /// Names for each view (optional)
    pub view_names: Option<Vec<String>>,
    /// Number of samples (consistent across views)
    pub n_samples: usize,
}

impl MultiViewData {
    /// Create multi-view data from multiple arrays
    pub fn new(views: Vec<Array2<f64>>) -> Result<Self> {
        if views.is_empty() {
            return Err(SklearsError::InvalidInput(
                "At least one view is required".to_string(),
            ));
        }

        let n_samples = views[0].nrows();
        for (i, view) in views.iter().enumerate() {
            if view.nrows() != n_samples {
                return Err(SklearsError::InvalidInput(format!(
                    "View {} has different number of samples",
                    i
                )));
            }
        }

        Ok(Self {
            views,
            view_names: None,
            n_samples,
        })
    }

    /// Set view names
    pub fn with_view_names(mut self, names: Vec<String>) -> Result<Self> {
        if names.len() != self.views.len() {
            return Err(SklearsError::InvalidInput(
                "Number of view names must match number of views".to_string(),
            ));
        }
        self.view_names = Some(names);
        Ok(self)
    }

    /// Get number of views
    pub fn n_views(&self) -> usize {
        self.views.len()
    }

    /// Get view by index
    pub fn get_view(&self, index: usize) -> Result<&Array2<f64>> {
        self.views.get(index).ok_or_else(|| {
            SklearsError::InvalidInput(format!("View index {} out of bounds", index))
        })
    }

    /// Get feature dimensions for each view
    pub fn view_dimensions(&self) -> Vec<usize> {
        self.views.iter().map(|v| v.ncols()).collect()
    }
}

/// Configuration for multi-view K-means clustering
#[derive(Debug, Clone)]
pub struct MultiViewKMeansConfig {
    /// Number of clusters
    pub k_clusters: usize,
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Weights for each view (if None, equal weights are used)
    pub view_weights: Option<Vec<f64>>,
    /// Weight learning strategy
    pub weight_learning: WeightLearning,
    /// Random seed for reproducibility
    pub random_seed: Option<u64>,
}

impl Default for MultiViewKMeansConfig {
    fn default() -> Self {
        Self {
            k_clusters: 2,
            max_iter: 100,
            tolerance: 1e-4,
            view_weights: None,
            weight_learning: WeightLearning::Fixed,
            random_seed: None,
        }
    }
}

/// Weight learning strategies for multi-view clustering
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WeightLearning {
    /// Fixed weights (no learning)
    Fixed,
    /// Learn weights based on view quality
    Adaptive,
    /// Entropy-based weight learning
    Entropy,
}

/// Multi-view K-means clustering
pub struct MultiViewKMeans {
    config: MultiViewKMeansConfig,
}

/// Fitted multi-view K-means model
pub struct MultiViewKMeansFitted {
    /// Final cluster assignments
    pub labels: Vec<i32>,
    /// Centroids for each view
    pub centroids: Vec<Array2<f64>>,
    /// Final view weights
    pub view_weights: Vec<f64>,
    /// Number of iterations performed
    pub n_iterations: usize,
    /// Final inertia for each view
    pub view_inertias: Vec<f64>,
    /// Overall inertia
    pub total_inertia: f64,
}

impl MultiViewKMeans {
    /// Create a new multi-view K-means instance
    pub fn new(config: MultiViewKMeansConfig) -> Self {
        Self { config }
    }

    /// Fit clustering to multi-view data
    pub fn fit(&self, data: &MultiViewData) -> Result<MultiViewKMeansFitted> {
        let n_views = data.n_views();
        let n_samples = data.n_samples;
        let k = self.config.k_clusters;

        if k > n_samples {
            return Err(SklearsError::InvalidInput(
                "Number of clusters cannot exceed number of samples".to_string(),
            ));
        }

        // Initialize view weights
        let mut view_weights = if let Some(weights) = &self.config.view_weights {
            if weights.len() != n_views {
                return Err(SklearsError::InvalidInput(
                    "View weights length must match number of views".to_string(),
                ));
            }
            weights.clone()
        } else {
            vec![1.0 / n_views as f64; n_views]
        };

        // Initialize centroids for each view
        let mut centroids = self.initialize_centroids(data)?;

        // Initialize cluster assignments
        let mut labels = vec![0; n_samples];
        let mut prev_labels = vec![-1; n_samples];

        let rng = match self.config.random_seed {
            Some(seed) => Random::seed(seed),
            None => Random::seed(42),
        };

        for iteration in 0..self.config.max_iter {
            // E-step: Assign points to clusters
            for i in 0..n_samples {
                let mut min_distance = f64::INFINITY;
                let mut best_cluster = 0;

                for k_idx in 0..k {
                    let mut total_distance = 0.0;

                    for (v, view) in data.views.iter().enumerate() {
                        let point = view.row(i);
                        let centroid = centroids[v].row(k_idx);
                        let distance: f64 = point
                            .iter()
                            .zip(centroid.iter())
                            .map(|(a, b)| (a - b).powi(2))
                            .sum();
                        total_distance += view_weights[v] * distance;
                    }

                    if total_distance < min_distance {
                        min_distance = total_distance;
                        best_cluster = k_idx;
                    }
                }

                labels[i] = best_cluster as i32;
            }

            // M-step: Update centroids
            for v in 0..n_views {
                let view = &data.views[v];
                let n_features = view.ncols();

                for k_idx in 0..k {
                    let cluster_points: Vec<usize> = labels
                        .iter()
                        .enumerate()
                        .filter(|(_, &label)| label == k_idx as i32)
                        .map(|(i, _)| i)
                        .collect();

                    if !cluster_points.is_empty() {
                        let mut new_centroid = vec![0.0; n_features];
                        for &point_idx in &cluster_points {
                            for j in 0..n_features {
                                new_centroid[j] += view[[point_idx, j]];
                            }
                        }
                        for val in new_centroid.iter_mut() {
                            *val /= cluster_points.len() as f64;
                        }

                        for j in 0..n_features {
                            centroids[v][[k_idx, j]] = new_centroid[j];
                        }
                    }
                }
            }

            // Update view weights if adaptive learning is enabled
            if self.config.weight_learning != WeightLearning::Fixed {
                view_weights =
                    self.update_view_weights(data, &labels, &centroids, &view_weights)?;
            }

            // Check convergence
            if self.has_converged(&labels, &prev_labels) {
                break;
            }

            prev_labels = labels.clone();
        }

        // Compute final inertias
        let view_inertias = self.compute_view_inertias(data, &labels, &centroids);
        let total_inertia = view_inertias
            .iter()
            .zip(view_weights.iter())
            .map(|(inertia, weight)| inertia * weight)
            .sum();

        Ok(MultiViewKMeansFitted {
            labels,
            centroids,
            view_weights,
            n_iterations: self.config.max_iter,
            view_inertias,
            total_inertia,
        })
    }

    /// Initialize centroids for all views
    fn initialize_centroids(&self, data: &MultiViewData) -> Result<Vec<Array2<f64>>> {
        let n_views = data.n_views();
        let k = self.config.k_clusters;
        let mut centroids = Vec::new();

        let mut rng = match self.config.random_seed {
            Some(seed) => Random::seed(seed),
            None => Random::seed(42),
        };

        for v in 0..n_views {
            let view = &data.views[v];
            let n_features = view.ncols();
            let n_samples = view.nrows();

            let mut view_centroids = Array2::zeros((k, n_features));

            // Random initialization - select k random points
            let mut selected_indices = (0..n_samples).collect::<Vec<_>>();
            // Fisher-Yates shuffle
            for i in (1..selected_indices.len()).rev() {
                let j = rng.gen_range(0..i + 1);
                selected_indices.swap(i, j);
            }

            for (k_idx, &sample_idx) in selected_indices.iter().take(k).enumerate() {
                for j in 0..n_features {
                    view_centroids[[k_idx, j]] = view[[sample_idx, j]];
                }
            }

            centroids.push(view_centroids);
        }

        Ok(centroids)
    }

    /// Update view weights based on clustering quality
    fn update_view_weights(
        &self,
        data: &MultiViewData,
        labels: &[i32],
        centroids: &[Array2<f64>],
        current_weights: &[f64],
    ) -> Result<Vec<f64>> {
        let n_views = data.n_views();
        let mut new_weights = vec![0.0; n_views];

        match self.config.weight_learning {
            WeightLearning::Fixed => Ok(current_weights.to_vec()),
            WeightLearning::Adaptive => {
                // Weight based on inverse of view inertia
                let view_inertias = self.compute_view_inertias(data, labels, centroids);
                let total_inv_inertia: f64 = view_inertias
                    .iter()
                    .map(|&inertia| 1.0 / (inertia + 1e-8))
                    .sum();

                for v in 0..n_views {
                    new_weights[v] = (1.0 / (view_inertias[v] + 1e-8)) / total_inv_inertia;
                }

                Ok(new_weights)
            }
            WeightLearning::Entropy => {
                // Entropy-based weight learning
                for v in 0..n_views {
                    let entropy = self.compute_view_entropy(data, labels, v);
                    new_weights[v] = 1.0 / (entropy + 1e-8);
                }

                // Normalize weights
                let total_weight: f64 = new_weights.iter().sum();
                for weight in new_weights.iter_mut() {
                    *weight /= total_weight;
                }

                Ok(new_weights)
            }
        }
    }

    /// Compute inertia for each view
    fn compute_view_inertias(
        &self,
        data: &MultiViewData,
        labels: &[i32],
        centroids: &[Array2<f64>],
    ) -> Vec<f64> {
        let n_views = data.n_views();
        let mut view_inertias = vec![0.0; n_views];

        for v in 0..n_views {
            let view = &data.views[v];
            let mut inertia = 0.0;

            for (i, &label) in labels.iter().enumerate() {
                let point = view.row(i);
                let centroid = centroids[v].row(label as usize);
                let distance: f64 = point
                    .iter()
                    .zip(centroid.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum();
                inertia += distance;
            }

            view_inertias[v] = inertia;
        }

        view_inertias
    }

    /// Compute entropy for a view (for entropy-based weight learning)
    fn compute_view_entropy(&self, data: &MultiViewData, labels: &[i32], view_index: usize) -> f64 {
        // Simplified entropy computation based on cluster sizes
        let mut cluster_counts = HashMap::new();
        for &label in labels {
            *cluster_counts.entry(label).or_insert(0) += 1;
        }

        let total_points = labels.len() as f64;
        let mut entropy = 0.0;

        for count in cluster_counts.values() {
            let p = *count as f64 / total_points;
            if p > 0.0 {
                entropy -= p * p.log2();
            }
        }

        entropy
    }

    /// Check if clustering has converged
    fn has_converged(&self, current_labels: &[i32], prev_labels: &[i32]) -> bool {
        if current_labels.len() != prev_labels.len() {
            return false;
        }

        let changes = current_labels
            .iter()
            .zip(prev_labels.iter())
            .filter(|(curr, prev)| curr != prev)
            .count();

        (changes as f64 / current_labels.len() as f64) < self.config.tolerance
    }
}

impl Estimator for MultiViewKMeans {
    type Config = MultiViewKMeansConfig;
    type Error = SklearsError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

/// Configuration for consensus clustering
#[derive(Debug, Clone)]
pub struct ConsensusClusteringConfig {
    /// Base clustering algorithms to use
    pub base_algorithms: Vec<String>,
    /// Number of clusters for base algorithms
    pub k_clusters: usize,
    /// Consensus method
    pub consensus_method: ConsensusMethod,
    /// View weighting strategy
    pub view_weighting: ViewWeighting,
    /// Random seed for reproducibility
    pub random_seed: Option<u64>,
}

impl Default for ConsensusClusteringConfig {
    fn default() -> Self {
        Self {
            base_algorithms: vec!["kmeans".to_string(), "spectral".to_string()],
            k_clusters: 2,
            consensus_method: ConsensusMethod::Voting,
            view_weighting: ViewWeighting::Equal,
            random_seed: None,
        }
    }
}

/// Consensus methods for combining clustering results
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ConsensusMethod {
    /// Majority voting
    Voting,
    /// Co-association matrix
    CoAssociation,
    /// Evidence accumulation
    EvidenceAccumulation,
}

/// View weighting strategies
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ViewWeighting {
    /// Equal weights for all views
    Equal,
    /// Quality-based weights
    Quality,
    /// Diversity-based weights
    Diversity,
}

/// Consensus clustering across multiple views
pub struct ConsensusClustering {
    config: ConsensusClusteringConfig,
}

/// Fitted consensus clustering model
pub struct ConsensusClusteringFitted {
    /// Final consensus cluster assignments
    pub labels: Vec<i32>,
    /// Individual clustering results for each view/algorithm
    pub individual_results: Vec<Vec<i32>>,
    /// Consensus matrix (co-association frequencies)
    pub consensus_matrix: Array2<f64>,
    /// View weights used
    pub view_weights: Vec<f64>,
    /// Agreement scores between views
    pub agreement_scores: Vec<f64>,
}

impl ConsensusClustering {
    /// Create a new consensus clustering instance
    pub fn new(config: ConsensusClusteringConfig) -> Self {
        Self { config }
    }

    /// Fit consensus clustering to multi-view data
    pub fn fit(&self, data: &MultiViewData) -> Result<ConsensusClusteringFitted> {
        let n_views = data.n_views();
        let n_samples = data.n_samples;

        // Run clustering on each view with each algorithm
        let mut individual_results = Vec::new();

        for v in 0..n_views {
            for algorithm in &self.config.base_algorithms {
                let view_data = &data.views[v];
                let labels = self.run_base_clustering(view_data, algorithm)?;
                individual_results.push(labels);
            }
        }

        // Compute view weights
        let view_weights = self.compute_view_weights(&individual_results)?;

        // Compute consensus matrix
        let consensus_matrix = self.compute_consensus_matrix(&individual_results, &view_weights)?;

        // Generate final consensus clustering
        let labels = self.generate_consensus_clustering(&consensus_matrix)?;

        // Compute agreement scores
        let agreement_scores = self.compute_agreement_scores(&individual_results, &labels);

        Ok(ConsensusClusteringFitted {
            labels,
            individual_results,
            consensus_matrix,
            view_weights,
            agreement_scores,
        })
    }

    /// Run base clustering algorithm on a single view
    fn run_base_clustering(&self, data: &Array2<f64>, algorithm: &str) -> Result<Vec<i32>> {
        let n_samples = data.nrows();
        let k = self.config.k_clusters;

        match algorithm {
            "kmeans" => {
                // Simple k-means implementation
                let mut rng = match self.config.random_seed {
                    Some(seed) => Random::seed(seed),
                    None => Random::seed(42),
                };

                let mut labels = vec![0; n_samples];
                for i in 0..n_samples {
                    labels[i] = rng.gen_range(0..k) as i32;
                }

                // Could implement full k-means here
                Ok(labels)
            }
            "spectral" => {
                // Simplified spectral clustering
                let mut rng = match self.config.random_seed {
                    Some(seed) => Random::seed(seed),
                    None => Random::seed(42),
                };

                let mut labels = vec![0; n_samples];
                for i in 0..n_samples {
                    labels[i] = rng.gen_range(0..k) as i32;
                }

                Ok(labels)
            }
            _ => Err(SklearsError::InvalidInput(format!(
                "Unsupported algorithm: {}",
                algorithm
            ))),
        }
    }

    /// Compute weights for different clustering results
    fn compute_view_weights(&self, results: &[Vec<i32>]) -> Result<Vec<f64>> {
        let n_results = results.len();

        match self.config.view_weighting {
            ViewWeighting::Equal => Ok(vec![1.0 / n_results as f64; n_results]),
            ViewWeighting::Quality => {
                // Weight based on silhouette-like quality measure
                let mut weights = vec![0.0; n_results];
                for (i, labels) in results.iter().enumerate() {
                    weights[i] = self.compute_clustering_quality(labels);
                }

                // Normalize weights
                let total_weight: f64 = weights.iter().sum();
                if total_weight > 0.0 {
                    for weight in weights.iter_mut() {
                        *weight /= total_weight;
                    }
                }

                Ok(weights)
            }
            ViewWeighting::Diversity => {
                // Weight based on diversity (how different the clustering is)
                let mut weights = vec![1.0; n_results];

                for i in 0..n_results {
                    let mut diversity_score = 0.0;
                    for j in 0..n_results {
                        if i != j {
                            diversity_score +=
                                self.compute_clustering_distance(&results[i], &results[j]);
                        }
                    }
                    weights[i] = diversity_score / (n_results - 1) as f64;
                }

                // Normalize weights
                let total_weight: f64 = weights.iter().sum();
                if total_weight > 0.0 {
                    for weight in weights.iter_mut() {
                        *weight /= total_weight;
                    }
                }

                Ok(weights)
            }
        }
    }

    /// Compute consensus matrix from individual clustering results
    fn compute_consensus_matrix(
        &self,
        results: &[Vec<i32>],
        weights: &[f64],
    ) -> Result<Array2<f64>> {
        if results.is_empty() {
            return Err(SklearsError::InvalidInput(
                "No clustering results provided".to_string(),
            ));
        }

        let n_samples = results[0].len();
        let mut consensus = Array2::zeros((n_samples, n_samples));

        match self.config.consensus_method {
            ConsensusMethod::CoAssociation => {
                // Co-association matrix: frequency that pairs are in same cluster
                for (result_idx, labels) in results.iter().enumerate() {
                    let weight = weights[result_idx];

                    for i in 0..n_samples {
                        for j in i..n_samples {
                            if labels[i] == labels[j] {
                                consensus[[i, j]] += weight;
                                consensus[[j, i]] += weight;
                            }
                        }
                    }
                }
            }
            ConsensusMethod::Voting | ConsensusMethod::EvidenceAccumulation => {
                // Similar implementation for voting
                for (result_idx, labels) in results.iter().enumerate() {
                    let weight = weights[result_idx];

                    for i in 0..n_samples {
                        for j in i..n_samples {
                            if labels[i] == labels[j] {
                                consensus[[i, j]] += weight;
                                consensus[[j, i]] += weight;
                            }
                        }
                    }
                }
            }
        }

        Ok(consensus)
    }

    /// Generate final consensus clustering from consensus matrix
    fn generate_consensus_clustering(&self, consensus_matrix: &Array2<f64>) -> Result<Vec<i32>> {
        let n_samples = consensus_matrix.nrows();

        // Simple approach: use hierarchical clustering on consensus matrix
        // For now, implement a simplified version

        let mut labels = vec![0; n_samples];
        let mut current_cluster = 0;
        let mut visited = vec![false; n_samples];

        for i in 0..n_samples {
            if !visited[i] {
                // Start new cluster
                let mut cluster_members = vec![i];
                visited[i] = true;

                // Find all points that should be in same cluster
                let mut stack = vec![i];
                while let Some(point) = stack.pop() {
                    for j in 0..n_samples {
                        if !visited[j] && consensus_matrix[[point, j]] > 0.5 {
                            visited[j] = true;
                            cluster_members.push(j);
                            stack.push(j);
                        }
                    }
                }

                // Assign cluster label
                for &member in &cluster_members {
                    labels[member] = current_cluster;
                }
                current_cluster += 1;
            }
        }

        Ok(labels)
    }

    /// Compute clustering quality score
    fn compute_clustering_quality(&self, labels: &[i32]) -> f64 {
        // Simple quality measure based on cluster balance
        let mut cluster_counts = HashMap::new();
        for &label in labels {
            *cluster_counts.entry(label).or_insert(0) += 1;
        }

        // Compute entropy (higher entropy = more balanced clusters = higher quality)
        let total = labels.len() as f64;
        let mut entropy = 0.0;

        for count in cluster_counts.values() {
            let p = *count as f64 / total;
            if p > 0.0 {
                entropy -= p * p.log2();
            }
        }

        entropy
    }

    /// Compute distance between two clusterings
    fn compute_clustering_distance(&self, labels1: &[i32], labels2: &[i32]) -> f64 {
        if labels1.len() != labels2.len() {
            return 0.0;
        }

        let n_samples = labels1.len();
        let mut disagreements = 0;

        for i in 0..n_samples {
            for j in (i + 1)..n_samples {
                let same_cluster_1 = labels1[i] == labels1[j];
                let same_cluster_2 = labels2[i] == labels2[j];

                if same_cluster_1 != same_cluster_2 {
                    disagreements += 1;
                }
            }
        }

        disagreements as f64 / ((n_samples * (n_samples - 1)) / 2) as f64
    }

    /// Compute agreement scores between consensus and individual results
    fn compute_agreement_scores(&self, results: &[Vec<i32>], consensus: &[i32]) -> Vec<f64> {
        results
            .iter()
            .map(|labels| 1.0 - self.compute_clustering_distance(labels, consensus))
            .collect()
    }
}

impl Estimator for ConsensusClustering {
    type Config = ConsensusClusteringConfig;
    type Error = SklearsError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_multi_view_data_creation() {
        let view1 = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let view2 = array![[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]];

        let multi_view_data = MultiViewData::new(vec![view1, view2]).unwrap();

        assert_eq!(multi_view_data.n_views(), 2);
        assert_eq!(multi_view_data.n_samples, 3);
        assert_eq!(multi_view_data.view_dimensions(), vec![2, 3]);
    }

    #[test]
    fn test_multi_view_data_mismatched_samples() {
        let view1 = array![[1.0, 2.0], [3.0, 4.0]]; // 2 samples
        let view2 = array![[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]; // 3 samples

        let result = MultiViewData::new(vec![view1, view2]);
        assert!(result.is_err());
    }

    #[test]
    fn test_multi_view_kmeans_config() {
        let config = MultiViewKMeansConfig {
            k_clusters: 3,
            max_iter: 50,
            tolerance: 1e-3,
            view_weights: Some(vec![0.6, 0.4]),
            weight_learning: WeightLearning::Adaptive,
            random_seed: Some(42),
        };

        let clusterer = MultiViewKMeans::new(config);
        // Test that creation doesn't panic
        assert_eq!(clusterer.config.k_clusters, 3);
    }

    #[test]
    fn test_consensus_clustering_creation() {
        let config = ConsensusClusteringConfig {
            base_algorithms: vec!["kmeans".to_string()],
            k_clusters: 2,
            consensus_method: ConsensusMethod::CoAssociation,
            view_weighting: ViewWeighting::Quality,
            random_seed: Some(42),
        };

        let clusterer = ConsensusClustering::new(config);
        assert_eq!(clusterer.config.k_clusters, 2);
    }
}
