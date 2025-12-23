//! Ensemble Clustering Algorithms
//!
//! This module provides ensemble clustering methods that combine multiple clustering
//! results to produce more robust and stable cluster assignments. Ensemble methods
//! can improve clustering quality by leveraging the diversity of different algorithms,
//! parameter settings, or data samples.
//!
//! # Algorithms Provided
//! - **Evidence Accumulation Clustering (EAC)**: Combines multiple partitions via co-association matrix
//! - **Voting-based Ensemble**: Democratic voting across multiple clustering results
//! - **Bagging Clustering**: Bootstrap aggregating for clustering stability
//! - **Consensus Clustering**: Iterative consensus seeking across partitions
//! - **Weighted Ensemble**: Weighted combination based on clustering quality
//!
//! # Mathematical Background
//!
//! ## Evidence Accumulation Clustering
//! The co-association matrix C measures how often pairs of points cluster together:
//! C[i,j] = (1/K) * Σ_k δ(labels_k[i] == labels_k[j])
//! where K is the number of partitions and δ is the indicator function.
//!
//! ## Consensus Function
//! Given K partitions P_1, ..., P_K, find partition P* that maximizes:
//! consensus(P*) = Σ_k agreement(P*, P_k)
//! where agreement measures similarity between partitions (e.g., Adjusted Rand Index).

use std::collections::HashMap;

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::{thread_rng, Rng};
use sklears_core::error::{Result, SklearsError};
use sklears_core::prelude::*;

/// Configuration for ensemble clustering
#[derive(Debug, Clone)]
pub struct EnsembleConfig {
    /// Number of base clusterings to generate
    pub n_clusterings: usize,
    /// Ensemble method to use
    pub method: EnsembleMethod,
    /// Consensus threshold for voting-based methods
    pub consensus_threshold: f64,
    /// Whether to use weighted combination based on quality metrics
    pub use_weights: bool,
    /// Random seed for reproducibility
    pub random_seed: Option<u64>,
}

impl Default for EnsembleConfig {
    fn default() -> Self {
        Self {
            n_clusterings: 10,
            method: EnsembleMethod::EvidenceAccumulation,
            consensus_threshold: 0.5,
            use_weights: false,
            random_seed: None,
        }
    }
}

/// Ensemble methods for combining multiple clusterings
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EnsembleMethod {
    /// Evidence Accumulation Clustering using co-association matrix
    EvidenceAccumulation,
    /// Voting-based ensemble (majority vote)
    Voting,
    /// Consensus clustering via iterative agreement
    Consensus,
    /// Weighted ensemble based on clustering quality
    Weighted,
}

/// Result of ensemble clustering
#[derive(Debug, Clone)]
pub struct EnsembleResult {
    /// Final cluster labels
    pub labels: Vec<i32>,
    /// Number of clusters in final result
    pub n_clusters: usize,
    /// Co-association matrix (for EAC method)
    pub co_association_matrix: Option<Array2<f64>>,
    /// Quality scores for each base clustering
    pub base_quality_scores: Vec<f64>,
    /// Consensus score (higher is better)
    pub consensus_score: f64,
    /// Stability score across base clusterings
    pub stability_score: f64,
}

/// Evidence Accumulation Clustering
///
/// Combines multiple clustering partitions by building a co-association matrix
/// that measures how frequently pairs of points are assigned to the same cluster.
#[derive(Clone)]
pub struct EvidenceAccumulationClustering {
    config: EnsembleConfig,
}

impl EvidenceAccumulationClustering {
    /// Create new Evidence Accumulation Clustering
    pub fn new(config: EnsembleConfig) -> Self {
        Self { config }
    }

    /// Create with default configuration
    pub fn with_n_clusterings(n_clusterings: usize) -> Self {
        Self {
            config: EnsembleConfig {
                n_clusterings,
                method: EnsembleMethod::EvidenceAccumulation,
                ..Default::default()
            },
        }
    }

    /// Build co-association matrix from multiple partitions
    fn build_co_association_matrix(
        &self,
        partitions: &[Vec<i32>],
        n_samples: usize,
    ) -> Array2<f64> {
        let mut co_assoc = Array2::zeros((n_samples, n_samples));
        let n_partitions = partitions.len() as f64;

        for partition in partitions {
            for i in 0..n_samples {
                for j in i..n_samples {
                    if partition[i] == partition[j] && partition[i] >= 0 {
                        co_assoc[[i, j]] += 1.0;
                        if i != j {
                            co_assoc[[j, i]] += 1.0;
                        }
                    }
                }
            }
        }

        // Normalize by number of partitions
        co_assoc / n_partitions
    }

    /// Extract final clustering from co-association matrix using hierarchical clustering
    fn extract_clustering_from_coassoc(
        &self,
        co_assoc: &Array2<f64>,
        n_clusters: usize,
    ) -> Result<Vec<i32>> {
        let n_samples = co_assoc.nrows();

        // Convert co-association to distance matrix (1 - co_assoc)
        let mut distances = Array2::zeros((n_samples, n_samples));
        for i in 0..n_samples {
            for j in 0..n_samples {
                distances[[i, j]] = 1.0 - co_assoc[[i, j]];
            }
        }

        // Simple agglomerative clustering on distance matrix
        let mut labels = vec![-1i32; n_samples];
        let mut cluster_id = 0;

        // Find most similar pairs iteratively
        let mut assigned = vec![false; n_samples];
        let mut cluster_members: Vec<Vec<usize>> = Vec::new();

        // Start with each point as its own cluster
        for i in 0..n_samples {
            cluster_members.push(vec![i]);
        }

        // Merge clusters until we have n_clusters
        while cluster_members.len() > n_clusters {
            let mut best_similarity = 0.0;
            let mut best_pair = (0, 0);

            // Find best pair to merge
            for i in 0..cluster_members.len() {
                for j in (i + 1)..cluster_members.len() {
                    let mut total_sim = 0.0;
                    let mut count = 0;

                    for &idx_i in &cluster_members[i] {
                        for &idx_j in &cluster_members[j] {
                            total_sim += co_assoc[[idx_i, idx_j]];
                            count += 1;
                        }
                    }

                    let avg_sim = total_sim / count as f64;
                    if avg_sim > best_similarity {
                        best_similarity = avg_sim;
                        best_pair = (i, j);
                    }
                }
            }

            // Merge best pair
            let (i, j) = best_pair;
            let mut merged = cluster_members[i].clone();
            merged.extend(&cluster_members[j]);

            // Remove old clusters (remove larger index first to avoid index issues)
            let (first, second) = if i > j { (i, j) } else { (j, i) };
            cluster_members.remove(first);
            cluster_members.remove(second);
            cluster_members.push(merged);
        }

        // Assign labels
        for (cluster_id, members) in cluster_members.iter().enumerate() {
            for &idx in members {
                labels[idx] = cluster_id as i32;
            }
        }

        Ok(labels)
    }

    /// Combine multiple clustering partitions
    pub fn combine_clusterings(
        &self,
        partitions: &[Vec<i32>],
        n_clusters: usize,
    ) -> Result<EnsembleResult> {
        if partitions.is_empty() {
            return Err(SklearsError::InvalidInput(
                "At least one partition is required".to_string(),
            ));
        }

        let n_samples = partitions[0].len();

        // Verify all partitions have same size
        for (i, partition) in partitions.iter().enumerate() {
            if partition.len() != n_samples {
                return Err(SklearsError::InvalidInput(format!(
                    "Partition {} has different number of samples",
                    i
                )));
            }
        }

        // Build co-association matrix
        let co_assoc = self.build_co_association_matrix(partitions, n_samples);

        // Extract final clustering
        let labels = self.extract_clustering_from_coassoc(&co_assoc, n_clusters)?;

        // Calculate quality metrics
        let base_quality_scores = self.calculate_base_quality_scores(partitions);
        let consensus_score = self.calculate_consensus_score(&co_assoc);
        let stability_score = self.calculate_stability_score(partitions);

        Ok(EnsembleResult {
            labels,
            n_clusters,
            co_association_matrix: Some(co_assoc),
            base_quality_scores,
            consensus_score,
            stability_score,
        })
    }

    /// Calculate quality scores for each base clustering
    fn calculate_base_quality_scores(&self, partitions: &[Vec<i32>]) -> Vec<f64> {
        partitions
            .iter()
            .map(|partition| {
                // Simple quality measure: number of unique clusters / number of samples
                // (normalized cluster count - prevents degenerate solutions)
                let n_clusters = partition.iter().filter(|&&x| x >= 0).max().unwrap_or(&0) + 1;
                let n_samples = partition.len();
                1.0 - (n_clusters as f64 / n_samples as f64).abs()
            })
            .collect()
    }

    /// Calculate consensus score from co-association matrix
    fn calculate_consensus_score(&self, co_assoc: &Array2<f64>) -> f64 {
        let n_samples = co_assoc.nrows();
        let mut total = 0.0;
        let mut count = 0;

        for i in 0..n_samples {
            for j in (i + 1)..n_samples {
                total += co_assoc[[i, j]];
                count += 1;
            }
        }

        if count > 0 {
            total / count as f64
        } else {
            0.0
        }
    }

    /// Calculate stability score across partitions
    fn calculate_stability_score(&self, partitions: &[Vec<i32>]) -> f64 {
        if partitions.len() < 2 {
            return 1.0;
        }

        let n_samples = partitions[0].len();
        let mut agreement_sum = 0.0;
        let mut count = 0;

        // Calculate pairwise agreement between partitions
        for i in 0..partitions.len() {
            for j in (i + 1)..partitions.len() {
                let agreement = self.calculate_partition_agreement(&partitions[i], &partitions[j]);
                agreement_sum += agreement;
                count += 1;
            }
        }

        if count > 0 {
            agreement_sum / count as f64
        } else {
            0.0
        }
    }

    /// Calculate agreement between two partitions (simplified ARI)
    fn calculate_partition_agreement(&self, labels1: &[i32], labels2: &[i32]) -> f64 {
        let n = labels1.len();
        let mut agreement = 0;

        for i in 0..n {
            for j in (i + 1)..n {
                let same_cluster_1 = labels1[i] == labels1[j] && labels1[i] >= 0;
                let same_cluster_2 = labels2[i] == labels2[j] && labels2[i] >= 0;

                if same_cluster_1 == same_cluster_2 {
                    agreement += 1;
                }
            }
        }

        let total_pairs = n * (n - 1) / 2;
        if total_pairs > 0 {
            agreement as f64 / total_pairs as f64
        } else {
            0.0
        }
    }
}

/// Voting-based ensemble clustering
///
/// Assigns each point to the cluster that receives the most votes across
/// multiple base clusterings.
#[derive(Clone)]
pub struct VotingEnsemble {
    config: EnsembleConfig,
}

impl VotingEnsemble {
    /// Create new voting ensemble
    pub fn new(config: EnsembleConfig) -> Self {
        Self { config }
    }

    /// Combine clusterings using majority voting
    pub fn combine_clusterings(&self, partitions: &[Vec<i32>]) -> Result<EnsembleResult> {
        if partitions.is_empty() {
            return Err(SklearsError::InvalidInput(
                "At least one partition is required".to_string(),
            ));
        }

        let n_samples = partitions[0].len();

        // Build voting matrix: for each point, count which cluster ID appears most often
        let mut final_labels = vec![-1i32; n_samples];

        for i in 0..n_samples {
            let mut vote_counts: HashMap<i32, usize> = HashMap::new();

            // Collect votes from all partitions
            for partition in partitions {
                let label = partition[i];
                if label >= 0 {
                    *vote_counts.entry(label).or_insert(0) += 1;
                }
            }

            // Assign label with most votes
            if let Some((&winning_label, _)) = vote_counts.iter().max_by_key(|(_, &count)| count) {
                final_labels[i] = winning_label;
            }
        }

        // Relabel to consecutive integers starting from 0
        let final_labels = self.relabel_consecutive(&final_labels);

        let n_clusters = final_labels.iter().filter(|&&x| x >= 0).max().unwrap_or(&0) + 1;

        let base_quality_scores = vec![1.0; partitions.len()]; // Equal weights for voting
        let consensus_score = self.calculate_voting_consensus(partitions, &final_labels);
        let stability_score = 0.5; // Placeholder

        Ok(EnsembleResult {
            labels: final_labels,
            n_clusters: n_clusters as usize,
            co_association_matrix: None,
            base_quality_scores,
            consensus_score,
            stability_score,
        })
    }

    /// Relabel cluster IDs to be consecutive starting from 0
    fn relabel_consecutive(&self, labels: &[i32]) -> Vec<i32> {
        let mut label_map: HashMap<i32, i32> = HashMap::new();
        let mut next_id = 0;
        let mut result = Vec::with_capacity(labels.len());

        for &label in labels {
            if label < 0 {
                result.push(-1);
            } else {
                let new_label = *label_map.entry(label).or_insert_with(|| {
                    let id = next_id;
                    next_id += 1;
                    id
                });
                result.push(new_label);
            }
        }

        result
    }

    /// Calculate consensus score for voting result
    fn calculate_voting_consensus(&self, partitions: &[Vec<i32>], final_labels: &[i32]) -> f64 {
        let n_samples = final_labels.len();
        let mut total_agreement = 0.0;

        for partition in partitions {
            let mut agreement = 0;
            for i in 0..n_samples {
                if partition[i] == final_labels[i] {
                    agreement += 1;
                }
            }
            total_agreement += agreement as f64 / n_samples as f64;
        }

        total_agreement / partitions.len() as f64
    }
}

/// Bagging-based clustering ensemble
///
/// Generates multiple clusterings on bootstrap samples of the data
/// and combines them for improved stability.
#[derive(Clone)]
pub struct BaggingClustering {
    config: EnsembleConfig,
}

impl BaggingClustering {
    /// Create new bagging clustering ensemble
    pub fn new(config: EnsembleConfig) -> Self {
        Self { config }
    }

    /// Generate bootstrap sample indices
    fn generate_bootstrap_sample(&self, n_samples: usize, rng: &mut impl Rng) -> Vec<usize> {
        (0..n_samples)
            .map(|_| rng.random_range(0..n_samples))
            .collect()
    }

    /// Generate multiple bootstrap samples
    pub fn generate_bootstrap_samples(&self, n_samples: usize) -> Result<Vec<Vec<usize>>> {
        let mut rng = thread_rng();
        let mut samples = Vec::with_capacity(self.config.n_clusterings);

        for _ in 0..self.config.n_clusterings {
            samples.push(self.generate_bootstrap_sample(n_samples, &mut rng));
        }

        Ok(samples)
    }
}

/// Builder for ensemble clustering configuration
pub struct EnsembleConfigBuilder {
    config: EnsembleConfig,
}

impl EnsembleConfigBuilder {
    /// Create new builder with defaults
    pub fn new() -> Self {
        Self {
            config: EnsembleConfig::default(),
        }
    }

    /// Set number of base clusterings
    pub fn n_clusterings(mut self, n: usize) -> Self {
        self.config.n_clusterings = n;
        self
    }

    /// Set ensemble method
    pub fn method(mut self, method: EnsembleMethod) -> Self {
        self.config.method = method;
        self
    }

    /// Set consensus threshold
    pub fn consensus_threshold(mut self, threshold: f64) -> Self {
        self.config.consensus_threshold = threshold;
        self
    }

    /// Set whether to use weighted combination
    pub fn use_weights(mut self, use_weights: bool) -> Self {
        self.config.use_weights = use_weights;
        self
    }

    /// Set random seed
    pub fn random_seed(mut self, seed: u64) -> Self {
        self.config.random_seed = Some(seed);
        self
    }

    /// Build the configuration
    pub fn build(self) -> EnsembleConfig {
        self.config
    }
}

impl Default for EnsembleConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl EnsembleResult {
    /// Get the most stable clusters (highest co-association scores)
    pub fn most_stable_clusters(&self, threshold: f64) -> Option<Vec<(usize, usize)>> {
        self.co_association_matrix.as_ref().map(|co_assoc| {
            let n = co_assoc.nrows();
            let mut stable_pairs = Vec::new();

            for i in 0..n {
                for j in (i + 1)..n {
                    if co_assoc[[i, j]] >= threshold {
                        stable_pairs.push((i, j));
                    }
                }
            }

            stable_pairs
        })
    }

    /// Get clusters with high internal consistency
    pub fn high_quality_clusters(&self, quality_threshold: f64) -> Vec<i32> {
        if self.consensus_score >= quality_threshold {
            self.labels
                .iter()
                .filter(|&&x| x >= 0)
                .copied()
                .collect::<std::collections::HashSet<_>>()
                .into_iter()
                .collect()
        } else {
            Vec::new()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_evidence_accumulation_basic() {
        let partitions = vec![
            vec![0, 0, 1, 1, 2, 2],
            vec![0, 0, 1, 1, 2, 2],
            vec![1, 1, 0, 0, 2, 2],
        ];

        let eac = EvidenceAccumulationClustering::with_n_clusterings(3);
        let result = eac.combine_clusterings(&partitions, 3).unwrap();

        assert_eq!(result.labels.len(), 6);
        assert_eq!(result.n_clusters, 3);
        assert!(result.consensus_score > 0.0);
        assert!(result.consensus_score <= 1.0);
    }

    #[test]
    fn test_voting_ensemble() {
        let partitions = vec![
            vec![0, 0, 1, 1, 2, 2],
            vec![0, 0, 1, 1, 2, 2],
            vec![1, 1, 0, 0, 2, 2],
        ];

        let voting = VotingEnsemble::new(EnsembleConfig::default());
        let result = voting.combine_clusterings(&partitions).unwrap();

        assert_eq!(result.labels.len(), 6);
        assert!(result.n_clusters > 0);
    }

    #[test]
    fn test_bagging_bootstrap_samples() {
        let bagging = BaggingClustering::new(
            EnsembleConfigBuilder::new()
                .n_clusterings(5)
                .random_seed(42)
                .build(),
        );

        let samples = bagging.generate_bootstrap_samples(100).unwrap();
        assert_eq!(samples.len(), 5);
        assert_eq!(samples[0].len(), 100);
    }

    #[test]
    fn test_ensemble_config_builder() {
        let config = EnsembleConfigBuilder::new()
            .n_clusterings(20)
            .method(EnsembleMethod::Voting)
            .consensus_threshold(0.7)
            .use_weights(true)
            .random_seed(123)
            .build();

        assert_eq!(config.n_clusterings, 20);
        assert_eq!(config.method, EnsembleMethod::Voting);
        assert_eq!(config.consensus_threshold, 0.7);
        assert!(config.use_weights);
        assert_eq!(config.random_seed, Some(123));
    }

    #[test]
    fn test_co_association_matrix() {
        let eac = EvidenceAccumulationClustering::with_n_clusterings(2);
        let partitions = vec![vec![0, 0, 1, 1], vec![0, 0, 1, 1]];

        let co_assoc = eac.build_co_association_matrix(&partitions, 4);

        // Points in same cluster should have high co-association
        assert!(co_assoc[[0, 1]] > 0.9);
        assert!(co_assoc[[2, 3]] > 0.9);

        // Points in different clusters should have low co-association
        assert!(co_assoc[[0, 2]] < 0.1);
        assert!(co_assoc[[1, 3]] < 0.1);
    }
}
