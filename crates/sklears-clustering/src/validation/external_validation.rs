//! External clustering validation methods
//!
//! This module implements external validation metrics that evaluate clustering quality
//! by comparing clustering results against ground truth labels. These metrics are used
//! when true cluster assignments are known and provide objective measures of clustering accuracy.

use super::internal_validation::ClusteringValidator;
use super::validation_types::*;
use sklears_core::error::{Result, SklearsError};
use std::collections::HashMap;

/// External validation methods implementation
///
/// This implementation provides methods to compute various external validation metrics
/// that compare predicted cluster assignments against true cluster labels.
impl ClusteringValidator {
    /// Compute comprehensive external validation metrics
    ///
    /// This method combines all external validation measures to provide
    /// a complete assessment when ground truth labels are available.
    ///
    /// # Arguments
    /// * `true_labels` - Ground truth cluster labels
    /// * `pred_labels` - Predicted cluster labels from clustering algorithm
    ///
    /// # Returns
    /// Complete external validation metrics
    pub fn external_validation(
        &self,
        true_labels: &[i32],
        pred_labels: &[i32],
    ) -> Result<ExternalValidationMetrics> {
        if true_labels.len() != pred_labels.len() {
            return Err(SklearsError::InvalidInput(
                "True and predicted labels length mismatch".to_string(),
            ));
        }

        // Compute individual metrics
        let adjusted_rand_index = self.adjusted_rand_index(true_labels, pred_labels)?;
        let normalized_mutual_info =
            self.normalized_mutual_information(true_labels, pred_labels)?;
        let v_measure = self.v_measure(true_labels, pred_labels)?;
        let (homogeneity, completeness) =
            self.homogeneity_completeness(true_labels, pred_labels)?;
        let fowlkes_mallows = self.fowlkes_mallows_index(true_labels, pred_labels)?;

        // Compute optional metrics
        let jaccard_index = Some(self.jaccard_index(true_labels, pred_labels)?);
        let purity = Some(self.purity_score(true_labels, pred_labels)?);
        let inverse_purity = Some(self.inverse_purity_score(true_labels, pred_labels)?);

        Ok(ExternalValidationMetrics {
            adjusted_rand_index,
            normalized_mutual_info,
            v_measure,
            homogeneity,
            completeness,
            fowlkes_mallows,
            jaccard_index,
            purity,
            inverse_purity,
        })
    }

    /// Compute Adjusted Rand Index (ARI)
    ///
    /// The ARI measures the similarity between two clusterings by considering all pairs
    /// of samples and counting pairs that are assigned in the same or different clusters
    /// in the predicted and true clusterings. It's adjusted for chance, making it more
    /// reliable than the basic Rand Index.
    ///
    /// Range: [-1, 1] where 1 = perfect agreement, 0 = random agreement, negative = worse than random
    ///
    /// # Arguments
    /// * `true_labels` - Ground truth cluster labels
    /// * `pred_labels` - Predicted cluster labels
    ///
    /// # Returns
    /// Adjusted Rand Index value
    pub fn adjusted_rand_index(&self, true_labels: &[i32], pred_labels: &[i32]) -> Result<f64> {
        if true_labels.len() != pred_labels.len() {
            return Err(SklearsError::InvalidInput(
                "Labels length mismatch".to_string(),
            ));
        }

        let n = true_labels.len();
        if n <= 1 {
            return Ok(1.0);
        }

        // Build contingency table
        let contingency = self.build_contingency_table(true_labels, pred_labels)?;
        let (true_clusters, pred_clusters) = self.get_cluster_mappings(true_labels, pred_labels);

        // Compute marginal sums
        let mut a_sum = vec![0; true_clusters.len()];
        let mut b_sum = vec![0; pred_clusters.len()];

        for (i, &true_cluster) in true_clusters.iter().enumerate() {
            for (j, &pred_cluster) in pred_clusters.iter().enumerate() {
                let count = contingency.get(&(true_cluster, pred_cluster)).unwrap_or(&0);
                a_sum[i] += count;
                b_sum[j] += count;
            }
        }

        // Compute ARI components
        let mut sum_comb_c = 0;
        for (i, &true_cluster) in true_clusters.iter().enumerate() {
            for (j, &pred_cluster) in pred_clusters.iter().enumerate() {
                let n_ij = *contingency.get(&(true_cluster, pred_cluster)).unwrap_or(&0);
                if n_ij >= 2 {
                    sum_comb_c += Self::combination_2(n_ij);
                }
            }
        }

        let mut sum_comb_a = 0;
        for &a_i in &a_sum {
            if a_i >= 2 {
                sum_comb_a += Self::combination_2(a_i);
            }
        }

        let mut sum_comb_b = 0;
        for &b_j in &b_sum {
            if b_j >= 2 {
                sum_comb_b += Self::combination_2(b_j);
            }
        }

        let max_comb = Self::combination_2(n);
        let expected_index = (sum_comb_a * sum_comb_b) as f64 / max_comb as f64;
        let max_index = ((sum_comb_a + sum_comb_b) as f64) / 2.0;
        let index = sum_comb_c as f64;

        if max_index - expected_index == 0.0 {
            Ok(0.0)
        } else {
            Ok((index - expected_index) / (max_index - expected_index))
        }
    }

    /// Compute Normalized Mutual Information (NMI)
    ///
    /// NMI measures the mutual dependence between two clusterings using information theory.
    /// It quantifies how much information one clustering provides about the other.
    ///
    /// Range: [0, 1] where 1 = perfect agreement, 0 = independent clusterings
    ///
    /// # Arguments
    /// * `true_labels` - Ground truth cluster labels
    /// * `pred_labels` - Predicted cluster labels
    ///
    /// # Returns
    /// Normalized Mutual Information value
    pub fn normalized_mutual_information(
        &self,
        true_labels: &[i32],
        pred_labels: &[i32],
    ) -> Result<f64> {
        if true_labels.len() != pred_labels.len() {
            return Err(SklearsError::InvalidInput(
                "Labels length mismatch".to_string(),
            ));
        }

        let n = true_labels.len() as f64;
        if n <= 1.0 {
            return Ok(1.0);
        }

        let contingency = self.build_contingency_table(true_labels, pred_labels)?;
        let (true_clusters, pred_clusters) = self.get_cluster_mappings(true_labels, pred_labels);

        // Compute marginal probabilities
        let mut p_true = HashMap::new();
        let mut p_pred = HashMap::new();

        for &label in true_labels {
            *p_true.entry(label).or_insert(0.0) += 1.0 / n;
        }

        for &label in pred_labels {
            *p_pred.entry(label).or_insert(0.0) += 1.0 / n;
        }

        // Compute mutual information
        let mut mutual_info = 0.0;
        for &true_cluster in &true_clusters {
            for &pred_cluster in &pred_clusters {
                let n_ij = *contingency.get(&(true_cluster, pred_cluster)).unwrap_or(&0) as f64;
                if n_ij > 0.0 {
                    let p_ij = n_ij / n;
                    let p_i = p_true[&true_cluster];
                    let p_j = p_pred[&pred_cluster];

                    mutual_info += p_ij * (p_ij / (p_i * p_j)).ln();
                }
            }
        }

        // Compute entropies
        let h_true = -p_true
            .values()
            .map(|&p| if p > 0.0 { p * p.ln() } else { 0.0 })
            .sum::<f64>();
        let h_pred = -p_pred
            .values()
            .map(|&p| if p > 0.0 { p * p.ln() } else { 0.0 })
            .sum::<f64>();

        // Normalize by geometric mean of entropies
        let denom = (h_true * h_pred).sqrt();
        if denom == 0.0 {
            Ok(0.0)
        } else {
            Ok(mutual_info / denom)
        }
    }

    /// Compute V-measure (harmonic mean of homogeneity and completeness)
    ///
    /// V-measure provides a balanced evaluation by ensuring both homogeneity
    /// (each cluster contains only members of a single class) and completeness
    /// (all members of a class are assigned to the same cluster) are satisfied.
    ///
    /// Range: [0, 1] where 1 = perfect clustering
    ///
    /// # Arguments
    /// * `true_labels` - Ground truth cluster labels
    /// * `pred_labels` - Predicted cluster labels
    ///
    /// # Returns
    /// V-measure value
    pub fn v_measure(&self, true_labels: &[i32], pred_labels: &[i32]) -> Result<f64> {
        let (homogeneity, completeness) =
            self.homogeneity_completeness(true_labels, pred_labels)?;

        if homogeneity + completeness == 0.0 {
            Ok(0.0)
        } else {
            Ok(2.0 * homogeneity * completeness / (homogeneity + completeness))
        }
    }

    /// Compute homogeneity and completeness scores
    ///
    /// Homogeneity: each cluster contains only members of a single class
    /// Completeness: all members of a class are assigned to the same cluster
    ///
    /// # Arguments
    /// * `true_labels` - Ground truth cluster labels
    /// * `pred_labels` - Predicted cluster labels
    ///
    /// # Returns
    /// Tuple of (homogeneity, completeness) scores
    pub fn homogeneity_completeness(
        &self,
        true_labels: &[i32],
        pred_labels: &[i32],
    ) -> Result<(f64, f64)> {
        if true_labels.len() != pred_labels.len() {
            return Err(SklearsError::InvalidInput(
                "Labels length mismatch".to_string(),
            ));
        }

        let n = true_labels.len() as f64;
        if n <= 1.0 {
            return Ok((1.0, 1.0));
        }

        let contingency = self.build_contingency_table(true_labels, pred_labels)?;
        let (true_clusters, pred_clusters) = self.get_cluster_mappings(true_labels, pred_labels);

        // Compute marginal counts
        let mut true_counts = HashMap::new();
        let mut pred_counts = HashMap::new();

        for &label in true_labels {
            *true_counts.entry(label).or_insert(0) += 1;
        }

        for &label in pred_labels {
            *pred_counts.entry(label).or_insert(0) += 1;
        }

        // Compute entropies
        let h_true = -true_counts
            .values()
            .map(|&count| {
                let p = count as f64 / n;
                if p > 0.0 {
                    p * p.ln()
                } else {
                    0.0
                }
            })
            .sum::<f64>();

        let h_pred = -pred_counts
            .values()
            .map(|&count| {
                let p = count as f64 / n;
                if p > 0.0 {
                    p * p.ln()
                } else {
                    0.0
                }
            })
            .sum::<f64>();

        // Compute conditional entropies
        let mut h_true_given_pred = 0.0;
        let mut h_pred_given_true = 0.0;

        for &pred_cluster in &pred_clusters {
            let n_pred = pred_counts[&pred_cluster] as f64;
            let p_pred = n_pred / n;

            let mut h_true_in_pred = 0.0;
            for &true_cluster in &true_clusters {
                let n_ij = *contingency.get(&(true_cluster, pred_cluster)).unwrap_or(&0) as f64;
                if n_ij > 0.0 {
                    let p_true_in_pred = n_ij / n_pred;
                    h_true_in_pred -= p_true_in_pred * p_true_in_pred.ln();
                }
            }

            h_true_given_pred += p_pred * h_true_in_pred;
        }

        for &true_cluster in &true_clusters {
            let n_true = true_counts[&true_cluster] as f64;
            let p_true = n_true / n;

            let mut h_pred_in_true = 0.0;
            for &pred_cluster in &pred_clusters {
                let n_ij = *contingency.get(&(true_cluster, pred_cluster)).unwrap_or(&0) as f64;
                if n_ij > 0.0 {
                    let p_pred_in_true = n_ij / n_true;
                    h_pred_in_true -= p_pred_in_true * p_pred_in_true.ln();
                }
            }

            h_pred_given_true += p_true * h_pred_in_true;
        }

        // Compute homogeneity and completeness
        let homogeneity = if h_true == 0.0 {
            1.0
        } else {
            1.0 - h_true_given_pred / h_true
        };
        let completeness = if h_pred == 0.0 {
            1.0
        } else {
            1.0 - h_pred_given_true / h_pred
        };

        Ok((homogeneity, completeness))
    }

    /// Compute Fowlkes-Mallows Index (FM)
    ///
    /// The FM index is the geometric mean of precision and recall when
    /// clustering is viewed as a binary classification problem for each pair of points.
    ///
    /// Range: [0, 1] where 1 = perfect clustering
    ///
    /// # Arguments
    /// * `true_labels` - Ground truth cluster labels
    /// * `pred_labels` - Predicted cluster labels
    ///
    /// # Returns
    /// Fowlkes-Mallows Index value
    pub fn fowlkes_mallows_index(&self, true_labels: &[i32], pred_labels: &[i32]) -> Result<f64> {
        if true_labels.len() != pred_labels.len() {
            return Err(SklearsError::InvalidInput(
                "Labels length mismatch".to_string(),
            ));
        }

        let n = true_labels.len();
        if n <= 1 {
            return Ok(1.0);
        }

        let mut tp = 0; // True positives: same cluster in both
        let mut fp = 0; // False positives: same in pred, different in true
        let mut fn_count = 0; // False negatives: different in pred, same in true

        for i in 0..n {
            for j in (i + 1)..n {
                let same_true = true_labels[i] == true_labels[j];
                let same_pred = pred_labels[i] == pred_labels[j];

                match (same_true, same_pred) {
                    (true, true) => tp += 1,
                    (false, true) => fp += 1,
                    (true, false) => fn_count += 1,
                    (false, false) => {} // True negatives - not used in FM calculation
                }
            }
        }

        let precision = if tp + fp == 0 {
            1.0
        } else {
            tp as f64 / (tp + fp) as f64
        };
        let recall = if tp + fn_count == 0 {
            1.0
        } else {
            tp as f64 / (tp + fn_count) as f64
        };

        Ok((precision * recall).sqrt())
    }

    /// Compute Jaccard Index
    ///
    /// The Jaccard index measures similarity as the intersection over union of pairs
    /// that are in the same cluster in both clusterings.
    ///
    /// Range: [0, 1] where 1 = identical clusterings
    ///
    /// # Arguments
    /// * `true_labels` - Ground truth cluster labels
    /// * `pred_labels` - Predicted cluster labels
    ///
    /// # Returns
    /// Jaccard Index value
    pub fn jaccard_index(&self, true_labels: &[i32], pred_labels: &[i32]) -> Result<f64> {
        if true_labels.len() != pred_labels.len() {
            return Err(SklearsError::InvalidInput(
                "Labels length mismatch".to_string(),
            ));
        }

        let n = true_labels.len();
        if n <= 1 {
            return Ok(1.0);
        }

        let mut tp = 0; // True positives: same cluster in both
        let mut fp = 0; // False positives: same in pred, different in true
        let mut fn_count = 0; // False negatives: different in pred, same in true

        for i in 0..n {
            for j in (i + 1)..n {
                let same_true = true_labels[i] == true_labels[j];
                let same_pred = pred_labels[i] == pred_labels[j];

                match (same_true, same_pred) {
                    (true, true) => tp += 1,
                    (false, true) => fp += 1,
                    (true, false) => fn_count += 1,
                    (false, false) => {} // True negatives - not used in Jaccard calculation
                }
            }
        }

        let union = tp + fp + fn_count;
        if union == 0 {
            Ok(1.0)
        } else {
            Ok(tp as f64 / union as f64)
        }
    }

    /// Compute purity score
    ///
    /// Purity measures the fraction of samples that are correctly clustered
    /// by taking the most frequent true class in each predicted cluster.
    ///
    /// Range: [0, 1] where 1 = perfect clustering
    ///
    /// # Arguments
    /// * `true_labels` - Ground truth cluster labels
    /// * `pred_labels` - Predicted cluster labels
    ///
    /// # Returns
    /// Purity score
    pub fn purity_score(&self, true_labels: &[i32], pred_labels: &[i32]) -> Result<f64> {
        if true_labels.len() != pred_labels.len() {
            return Err(SklearsError::InvalidInput(
                "Labels length mismatch".to_string(),
            ));
        }

        let n = true_labels.len();
        if n == 0 {
            return Ok(1.0);
        }

        let contingency = self.build_contingency_table(true_labels, pred_labels)?;
        let (_, pred_clusters) = self.get_cluster_mappings(true_labels, pred_labels);

        let mut correctly_assigned = 0;

        for &pred_cluster in &pred_clusters {
            // Find the most frequent true class in this predicted cluster
            let mut max_count = 0;
            for &true_cluster in true_labels {
                let count = *contingency.get(&(true_cluster, pred_cluster)).unwrap_or(&0);
                if count > max_count {
                    max_count = count;
                }
            }
            correctly_assigned += max_count;
        }

        Ok(correctly_assigned as f64 / n as f64)
    }

    /// Compute inverse purity (coverage) score
    ///
    /// Inverse purity measures how well each true class is represented by clusters
    /// by taking the most frequent predicted cluster for each true class.
    ///
    /// Range: [0, 1] where 1 = perfect clustering
    ///
    /// # Arguments
    /// * `true_labels` - Ground truth cluster labels
    /// * `pred_labels` - Predicted cluster labels
    ///
    /// # Returns
    /// Inverse purity score
    pub fn inverse_purity_score(&self, true_labels: &[i32], pred_labels: &[i32]) -> Result<f64> {
        // Inverse purity is purity with roles reversed
        self.purity_score(pred_labels, true_labels)
    }

    /// Build contingency table for two label sets
    fn build_contingency_table(
        &self,
        labels1: &[i32],
        labels2: &[i32],
    ) -> Result<HashMap<(i32, i32), usize>> {
        let mut contingency = HashMap::new();

        for (&l1, &l2) in labels1.iter().zip(labels2.iter()) {
            *contingency.entry((l1, l2)).or_insert(0) += 1;
        }

        Ok(contingency)
    }

    /// Get unique cluster mappings from labels
    fn get_cluster_mappings(
        &self,
        true_labels: &[i32],
        pred_labels: &[i32],
    ) -> (Vec<i32>, Vec<i32>) {
        let mut true_clusters: Vec<_> = true_labels.to_vec();
        let mut pred_clusters: Vec<_> = pred_labels.to_vec();

        true_clusters.sort_unstable();
        true_clusters.dedup();

        pred_clusters.sort_unstable();
        pred_clusters.dedup();

        (true_clusters, pred_clusters)
    }

    /// Compute combination C(n, 2) = n * (n-1) / 2
    fn combination_2(n: usize) -> usize {
        if n < 2 {
            0
        } else {
            n * (n - 1) / 2
        }
    }

    /// Compute agreement matrix for cluster comparison
    ///
    /// This method creates a matrix showing the overlap between
    /// true and predicted cluster assignments.
    ///
    /// # Arguments
    /// * `true_labels` - Ground truth cluster labels
    /// * `pred_labels` - Predicted cluster labels
    ///
    /// # Returns
    /// Agreement matrix and cluster mappings
    pub fn compute_agreement_matrix(
        &self,
        true_labels: &[i32],
        pred_labels: &[i32],
    ) -> Result<(Vec<Vec<usize>>, Vec<i32>, Vec<i32>)> {
        if true_labels.len() != pred_labels.len() {
            return Err(SklearsError::InvalidInput(
                "Labels length mismatch".to_string(),
            ));
        }

        let contingency = self.build_contingency_table(true_labels, pred_labels)?;
        let (true_clusters, pred_clusters) = self.get_cluster_mappings(true_labels, pred_labels);

        let mut agreement_matrix = vec![vec![0; pred_clusters.len()]; true_clusters.len()];

        for (i, &true_cluster) in true_clusters.iter().enumerate() {
            for (j, &pred_cluster) in pred_clusters.iter().enumerate() {
                agreement_matrix[i][j] =
                    *contingency.get(&(true_cluster, pred_cluster)).unwrap_or(&0);
            }
        }

        Ok((agreement_matrix, true_clusters, pred_clusters))
    }

    /// Find the optimal cluster assignment mapping
    ///
    /// This method finds the best one-to-one mapping between predicted and true clusters
    /// to maximize the number of correctly assigned samples.
    ///
    /// # Arguments
    /// * `true_labels` - Ground truth cluster labels
    /// * `pred_labels` - Predicted cluster labels
    ///
    /// # Returns
    /// Optimal mapping and the resulting accuracy
    pub fn find_optimal_assignment(
        &self,
        true_labels: &[i32],
        pred_labels: &[i32],
    ) -> Result<(HashMap<i32, i32>, f64)> {
        let (agreement_matrix, true_clusters, pred_clusters) =
            self.compute_agreement_matrix(true_labels, pred_labels)?;

        // Use Hungarian algorithm (simplified version for now)
        let mut best_assignment = HashMap::new();
        let mut best_score = 0;

        // Greedy assignment: for each predicted cluster, assign to best true cluster
        let mut used_true_clusters = std::collections::HashSet::new();

        for (j, &pred_cluster) in pred_clusters.iter().enumerate() {
            let mut best_true_idx = None;
            let mut best_count = 0;

            for (i, &true_cluster) in true_clusters.iter().enumerate() {
                if !used_true_clusters.contains(&true_cluster)
                    && agreement_matrix[i][j] > best_count
                {
                    best_count = agreement_matrix[i][j];
                    best_true_idx = Some((i, true_cluster));
                }
            }

            if let Some((_, true_cluster)) = best_true_idx {
                best_assignment.insert(pred_cluster, true_cluster);
                used_true_clusters.insert(true_cluster);
                best_score += best_count;
            }
        }

        let accuracy = best_score as f64 / true_labels.len() as f64;
        Ok((best_assignment, accuracy))
    }

    /// Compute confusion matrix for clustering evaluation
    ///
    /// # Arguments
    /// * `true_labels` - Ground truth cluster labels
    /// * `pred_labels` - Predicted cluster labels
    ///
    /// # Returns
    /// Confusion matrix with row/column mappings
    pub fn compute_confusion_matrix(
        &self,
        true_labels: &[i32],
        pred_labels: &[i32],
    ) -> Result<ConfusionMatrix> {
        let (agreement_matrix, true_clusters, pred_clusters) =
            self.compute_agreement_matrix(true_labels, pred_labels)?;

        let (optimal_assignment, accuracy) =
            self.find_optimal_assignment(true_labels, pred_labels)?;

        Ok(ConfusionMatrix {
            matrix: agreement_matrix,
            true_cluster_labels: true_clusters,
            pred_cluster_labels: pred_clusters,
            optimal_assignment,
            accuracy,
        })
    }
}

/// Confusion matrix for clustering evaluation
#[derive(Debug, Clone)]
pub struct ConfusionMatrix {
    /// Agreement matrix (true_clusters x pred_clusters)
    pub matrix: Vec<Vec<usize>>,
    /// True cluster labels
    pub true_cluster_labels: Vec<i32>,
    /// Predicted cluster labels
    pub pred_cluster_labels: Vec<i32>,
    /// Optimal assignment mapping pred -> true
    pub optimal_assignment: HashMap<i32, i32>,
    /// Accuracy with optimal assignment
    pub accuracy: f64,
}

impl ConfusionMatrix {
    /// Get the total number of samples
    pub fn total_samples(&self) -> usize {
        self.matrix
            .iter()
            .map(|row| row.iter().sum::<usize>())
            .sum()
    }

    /// Get the number of correctly assigned samples
    pub fn correct_assignments(&self) -> usize {
        let mut correct = 0;
        for (i, &true_label) in self.true_cluster_labels.iter().enumerate() {
            for (j, &pred_label) in self.pred_cluster_labels.iter().enumerate() {
                if let Some(&assigned_true) = self.optimal_assignment.get(&pred_label) {
                    if assigned_true == true_label {
                        correct += self.matrix[i][j];
                    }
                }
            }
        }
        correct
    }

    /// Get precision for each predicted cluster
    pub fn cluster_precisions(&self) -> Vec<f64> {
        let mut precisions = Vec::new();

        for (j, &pred_label) in self.pred_cluster_labels.iter().enumerate() {
            let total_pred: usize = self.matrix.iter().map(|row| row[j]).sum();

            if total_pred == 0 {
                precisions.push(0.0);
                continue;
            }

            let mut correct = 0;
            if let Some(&assigned_true) = self.optimal_assignment.get(&pred_label) {
                if let Some(true_idx) = self
                    .true_cluster_labels
                    .iter()
                    .position(|&x| x == assigned_true)
                {
                    correct = self.matrix[true_idx][j];
                }
            }

            precisions.push(correct as f64 / total_pred as f64);
        }

        precisions
    }

    /// Get recall for each true cluster
    pub fn cluster_recalls(&self) -> Vec<f64> {
        let mut recalls = Vec::new();

        for (i, &true_label) in self.true_cluster_labels.iter().enumerate() {
            let total_true: usize = self.matrix[i].iter().sum();

            if total_true == 0 {
                recalls.push(0.0);
                continue;
            }

            let mut correct = 0;
            for (j, &pred_label) in self.pred_cluster_labels.iter().enumerate() {
                if let Some(&assigned_true) = self.optimal_assignment.get(&pred_label) {
                    if assigned_true == true_label {
                        correct += self.matrix[i][j];
                    }
                }
            }

            recalls.push(correct as f64 / total_true as f64);
        }

        recalls
    }

    /// Generate a human-readable summary
    pub fn summary(&self) -> String {
        let precisions = self.cluster_precisions();
        let recalls = self.cluster_recalls();
        let avg_precision = precisions.iter().sum::<f64>() / precisions.len() as f64;
        let avg_recall = recalls.iter().sum::<f64>() / recalls.len() as f64;

        format!(
            "Confusion Matrix Summary:\n\
             - Total samples: {}\n\
             - Correctly assigned: {}\n\
             - Accuracy: {:.3}\n\
             - Average precision: {:.3}\n\
             - Average recall: {:.3}\n\
             - True clusters: {}\n\
             - Predicted clusters: {}",
            self.total_samples(),
            self.correct_assignments(),
            self.accuracy,
            avg_precision,
            avg_recall,
            self.true_cluster_labels.len(),
            self.pred_cluster_labels.len()
        )
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perfect_agreement() {
        let validator = ClusteringValidator::euclidean();

        // Perfect agreement
        let true_labels = vec![0, 0, 1, 1, 2, 2];
        let pred_labels = vec![0, 0, 1, 1, 2, 2];

        let ari = validator
            .adjusted_rand_index(&true_labels, &pred_labels)
            .unwrap();
        assert!((ari - 1.0).abs() < 1e-10);

        let nmi = validator
            .normalized_mutual_information(&true_labels, &pred_labels)
            .unwrap();
        assert!((nmi - 1.0).abs() < 1e-10);

        let v_measure = validator.v_measure(&true_labels, &pred_labels).unwrap();
        assert!((v_measure - 1.0).abs() < 1e-10);

        let fm = validator
            .fowlkes_mallows_index(&true_labels, &pred_labels)
            .unwrap();
        assert!((fm - 1.0).abs() < 1e-10);

        let jaccard = validator.jaccard_index(&true_labels, &pred_labels).unwrap();
        assert!((jaccard - 1.0).abs() < 1e-10);

        let purity = validator.purity_score(&true_labels, &pred_labels).unwrap();
        assert!((purity - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_random_assignment() {
        let validator = ClusteringValidator::euclidean();

        // Random assignment
        let true_labels = vec![0, 0, 1, 1, 2, 2];
        let pred_labels = vec![0, 1, 0, 1, 0, 1];

        let ari = validator
            .adjusted_rand_index(&true_labels, &pred_labels)
            .unwrap();
        assert!(ari >= -1.0 && ari <= 1.0);

        let nmi = validator
            .normalized_mutual_information(&true_labels, &pred_labels)
            .unwrap();
        assert!(nmi >= 0.0 && nmi <= 1.0);

        let v_measure = validator.v_measure(&true_labels, &pred_labels).unwrap();
        assert!(v_measure >= 0.0 && v_measure <= 1.0);

        let fm = validator
            .fowlkes_mallows_index(&true_labels, &pred_labels)
            .unwrap();
        assert!(fm >= 0.0 && fm <= 1.0);
    }

    #[test]
    fn test_external_validation_comprehensive() {
        let validator = ClusteringValidator::euclidean();

        let true_labels = vec![0, 0, 1, 1, 2, 2];
        let pred_labels = vec![0, 0, 1, 1, 2, 2];

        let metrics = validator
            .external_validation(&true_labels, &pred_labels)
            .unwrap();

        assert!((metrics.adjusted_rand_index - 1.0).abs() < 1e-10);
        assert!((metrics.normalized_mutual_info - 1.0).abs() < 1e-10);
        assert!((metrics.v_measure - 1.0).abs() < 1e-10);
        assert!((metrics.homogeneity - 1.0).abs() < 1e-10);
        assert!((metrics.completeness - 1.0).abs() < 1e-10);
        assert!((metrics.fowlkes_mallows - 1.0).abs() < 1e-10);
        assert_eq!(metrics.jaccard_index, Some(1.0));
        assert_eq!(metrics.purity, Some(1.0));
        assert_eq!(metrics.inverse_purity, Some(1.0));

        let consensus = metrics.consensus_score();
        assert!((consensus - 1.0).abs() < 1e-10);

        assert!(metrics.is_significant_match(0.8));
        let (best_metric, best_score) = metrics.best_metric();
        assert!(best_score >= 0.99);
    }

    #[test]
    fn test_homogeneity_completeness() {
        let validator = ClusteringValidator::euclidean();

        // Perfect case
        let true_labels = vec![0, 0, 1, 1, 2, 2];
        let pred_labels = vec![0, 0, 1, 1, 2, 2];

        let (homogeneity, completeness) = validator
            .homogeneity_completeness(&true_labels, &pred_labels)
            .unwrap();
        assert!((homogeneity - 1.0).abs() < 1e-10);
        assert!((completeness - 1.0).abs() < 1e-10);

        // High homogeneity, low completeness (over-clustering)
        let true_labels = vec![0, 0, 0, 1, 1, 1];
        let pred_labels = vec![0, 1, 2, 3, 4, 5]; // Each point in its own cluster

        let (homogeneity, completeness) = validator
            .homogeneity_completeness(&true_labels, &pred_labels)
            .unwrap();
        assert!(homogeneity > 0.9); // High homogeneity
        assert!(completeness < 0.5); // Low completeness
    }

    #[test]
    fn test_confusion_matrix() {
        let validator = ClusteringValidator::euclidean();

        let true_labels = vec![0, 0, 1, 1, 2, 2];
        let pred_labels = vec![1, 1, 0, 0, 2, 2]; // Swapped clusters 0 and 1

        let confusion = validator
            .compute_confusion_matrix(&true_labels, &pred_labels)
            .unwrap();

        assert_eq!(confusion.total_samples(), 6);
        assert_eq!(confusion.matrix.len(), 3); // 3 true clusters
        assert_eq!(confusion.matrix[0].len(), 3); // 3 predicted clusters

        let precisions = confusion.cluster_precisions();
        let recalls = confusion.cluster_recalls();

        assert_eq!(precisions.len(), 3);
        assert_eq!(recalls.len(), 3);

        // All precisions and recalls should be between 0 and 1
        for &precision in &precisions {
            assert!(precision >= 0.0 && precision <= 1.0);
        }
        for &recall in &recalls {
            assert!(recall >= 0.0 && recall <= 1.0);
        }

        let summary = confusion.summary();
        assert!(summary.contains("Total samples: 6"));
    }

    #[test]
    fn test_edge_cases() {
        let validator = ClusteringValidator::euclidean();

        // Empty labels
        let empty_true: Vec<i32> = vec![];
        let empty_pred: Vec<i32> = vec![];
        let ari = validator.adjusted_rand_index(&empty_true, &empty_pred);
        assert!(ari.is_ok());

        // Single sample
        let single_true = vec![0];
        let single_pred = vec![0];
        let ari = validator
            .adjusted_rand_index(&single_true, &single_pred)
            .unwrap();
        assert!((ari - 1.0).abs() < 1e-10);

        // Mismatched lengths
        let true_labels = vec![0, 1, 2];
        let pred_labels = vec![0, 1]; // Wrong length
        let result = validator.adjusted_rand_index(&true_labels, &pred_labels);
        assert!(result.is_err());
    }

    #[test]
    fn test_contingency_table() {
        let validator = ClusteringValidator::euclidean();

        let true_labels = vec![0, 0, 1, 1, 2, 2];
        let pred_labels = vec![1, 1, 0, 0, 2, 2];

        let contingency = validator
            .build_contingency_table(&true_labels, &pred_labels)
            .unwrap();

        // Check specific entries
        assert_eq!(contingency.get(&(0, 1)), Some(&2)); // True cluster 0 -> Pred cluster 1
        assert_eq!(contingency.get(&(1, 0)), Some(&2)); // True cluster 1 -> Pred cluster 0
        assert_eq!(contingency.get(&(2, 2)), Some(&2)); // True cluster 2 -> Pred cluster 2

        // Check total count
        let total: usize = contingency.values().sum();
        assert_eq!(total, 6);
    }

    #[test]
    fn test_combination_function() {
        use super::ClusteringValidator;

        assert_eq!(ClusteringValidator::combination_2(0), 0);
        assert_eq!(ClusteringValidator::combination_2(1), 0);
        assert_eq!(ClusteringValidator::combination_2(2), 1);
        assert_eq!(ClusteringValidator::combination_2(3), 3);
        assert_eq!(ClusteringValidator::combination_2(4), 6);
        assert_eq!(ClusteringValidator::combination_2(5), 10);
    }

    #[test]
    fn test_optimal_assignment() {
        let validator = ClusteringValidator::euclidean();

        // Case where predicted clusters need to be remapped
        let true_labels = vec![0, 0, 1, 1, 2, 2];
        let pred_labels = vec![2, 2, 1, 1, 0, 0]; // All clusters shifted

        let (assignment, accuracy) = validator
            .find_optimal_assignment(&true_labels, &pred_labels)
            .unwrap();

        assert_eq!(assignment.len(), 3);
        assert_eq!(accuracy, 1.0); // Should achieve perfect accuracy with optimal assignment

        // Verify the assignment makes sense
        assert!(assignment.contains_key(&0)); // Predicted cluster 0 maps to some true cluster
        assert!(assignment.contains_key(&1)); // Predicted cluster 1 maps to some true cluster
        assert!(assignment.contains_key(&2)); // Predicted cluster 2 maps to some true cluster
    }
}
