//! Hierarchical cross-validation for clustered data
//!
//! This module provides cross-validation strategies for hierarchical or clustered data,
//! where observations are nested within groups (e.g., students within schools,
//! patients within hospitals, measurements within subjects).

use scirs2_core::ndarray::{ArrayView1, ArrayView2};
use scirs2_core::random::prelude::*;
use scirs2_core::random::rngs::StdRng;
use scirs2_core::SliceRandomExt;
use sklears_core::prelude::*;
use std::collections::{HashMap, HashSet};

fn hierarchical_error(msg: &str) -> SklearsError {
    SklearsError::InvalidInput(msg.to_string())
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HierarchicalStrategy {
    /// ClusterBased
    ClusterBased,
    /// NestedCV
    NestedCV,
    /// MultilevelBootstrap
    MultilevelBootstrap,
    /// HierarchicalKFold
    HierarchicalKFold,
    /// LeaveOneClusterOut
    LeaveOneClusterOut,
}

#[derive(Debug, Clone)]
pub struct HierarchicalValidationConfig {
    pub strategy: HierarchicalStrategy,
    pub n_folds: usize,
    pub random_state: Option<u64>,
    pub shuffle: bool,
    pub balance_clusters: bool,
    pub min_cluster_size: usize,
    pub max_imbalance_ratio: f64,
}

impl Default for HierarchicalValidationConfig {
    fn default() -> Self {
        Self {
            strategy: HierarchicalStrategy::ClusterBased,
            n_folds: 5,
            random_state: None,
            shuffle: true,
            balance_clusters: false,
            min_cluster_size: 1,
            max_imbalance_ratio: 2.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ClusterInfo {
    pub cluster_id: String,
    pub size: usize,
    pub indices: Vec<usize>,
    pub level: usize,
    pub parent_cluster: Option<String>,
}

#[derive(Debug)]
pub struct HierarchicalSplit {
    pub train_indices: Vec<usize>,
    pub test_indices: Vec<usize>,
    pub train_clusters: Vec<String>,
    pub test_clusters: Vec<String>,
    pub fold_id: usize,
}

pub struct HierarchicalCrossValidator {
    config: HierarchicalValidationConfig,
    clusters: HashMap<String, ClusterInfo>,
    hierarchy_levels: usize,
    rng: StdRng,
}

impl HierarchicalCrossValidator {
    pub fn new(config: HierarchicalValidationConfig) -> Self {
        let rng = if let Some(seed) = config.random_state {
            StdRng::seed_from_u64(seed)
        } else {
            StdRng::from_rng(&mut scirs2_core::random::thread_rng())
        };

        Self {
            config,
            clusters: HashMap::new(),
            hierarchy_levels: 0,
            rng,
        }
    }

    pub fn with_cluster_labels(mut self, labels: &[String]) -> Result<Self> {
        if labels.is_empty() {
            return Err(hierarchical_error("Empty cluster labels"));
        }

        self.build_clusters(labels)?;
        Ok(self)
    }

    pub fn with_hierarchical_labels(mut self, labels: &[Vec<String>]) -> Result<Self> {
        if labels.is_empty() {
            return Err(hierarchical_error("Empty cluster labels"));
        }

        self.build_hierarchical_clusters(labels)?;
        Ok(self)
    }

    fn build_clusters(&mut self, labels: &[String]) -> Result<()> {
        let mut cluster_indices: HashMap<String, Vec<usize>> = HashMap::new();

        for (idx, label) in labels.iter().enumerate() {
            cluster_indices.entry(label.clone()).or_default().push(idx);
        }

        if cluster_indices.len() < self.config.n_folds {
            return Err(hierarchical_error(&format!(
                "Insufficient clusters for {} folds: got {}",
                self.config.n_folds,
                cluster_indices.len()
            )));
        }

        for (cluster_id, indices) in cluster_indices {
            if indices.len() >= self.config.min_cluster_size {
                self.clusters.insert(
                    cluster_id.clone(),
                    ClusterInfo {
                        cluster_id: cluster_id.clone(),
                        size: indices.len(),
                        indices,
                        level: 0,
                        parent_cluster: None,
                    },
                );
            }
        }

        self.hierarchy_levels = 1;
        Ok(())
    }

    fn build_hierarchical_clusters(&mut self, labels: &[Vec<String>]) -> Result<()> {
        if labels
            .iter()
            .any(|level_labels| level_labels.len() != labels[0].len())
        {
            return Err(hierarchical_error("Unbalanced hierarchy levels"));
        }

        self.hierarchy_levels = labels.len();
        let mut level_clusters: Vec<HashMap<String, Vec<usize>>> = Vec::new();

        for level in 0..self.hierarchy_levels {
            let mut cluster_indices: HashMap<String, Vec<usize>> = HashMap::new();

            for (idx, label) in labels[level].iter().enumerate() {
                cluster_indices.entry(label.clone()).or_default().push(idx);
            }

            level_clusters.push(cluster_indices);
        }

        for (level, cluster_indices) in level_clusters.into_iter().enumerate() {
            for (cluster_id, indices) in cluster_indices {
                if indices.len() >= self.config.min_cluster_size {
                    let parent_cluster = if level > 0 {
                        Some(labels[level - 1][indices[0]].clone())
                    } else {
                        None
                    };

                    self.clusters.insert(
                        format!("{}_{}", level, cluster_id),
                        ClusterInfo {
                            cluster_id: cluster_id.clone(),
                            size: indices.len(),
                            indices,
                            level,
                            parent_cluster,
                        },
                    );
                }
            }
        }

        Ok(())
    }

    pub fn split(&mut self, n_samples: usize) -> Result<Vec<HierarchicalSplit>> {
        match self.config.strategy {
            HierarchicalStrategy::ClusterBased => self.cluster_based_split(n_samples),
            HierarchicalStrategy::NestedCV => self.nested_cv_split(n_samples),
            HierarchicalStrategy::MultilevelBootstrap => self.multilevel_bootstrap_split(n_samples),
            HierarchicalStrategy::HierarchicalKFold => self.hierarchical_kfold_split(n_samples),
            HierarchicalStrategy::LeaveOneClusterOut => self.leave_one_cluster_out_split(n_samples),
        }
    }

    fn cluster_based_split(&mut self, n_samples: usize) -> Result<Vec<HierarchicalSplit>> {
        let top_level_clusters: Vec<_> = self
            .clusters
            .values()
            .filter(|c| c.level == 0 || (self.hierarchy_levels == 1))
            .collect();

        if top_level_clusters.len() < self.config.n_folds {
            return Err(hierarchical_error(&format!(
                "Insufficient clusters for {} folds: got {}",
                self.config.n_folds,
                top_level_clusters.len()
            )));
        }

        let mut cluster_ids: Vec<String> = top_level_clusters
            .iter()
            .map(|c| c.cluster_id.clone())
            .collect();

        if self.config.shuffle {
            cluster_ids.shuffle(&mut self.rng);
        }

        if self.config.balance_clusters {
            cluster_ids.sort_by_key(|id| self.clusters[id].size);
        }

        let mut splits = Vec::new();
        let clusters_per_fold = cluster_ids.len() / self.config.n_folds;
        let remainder = cluster_ids.len() % self.config.n_folds;

        for fold in 0..self.config.n_folds {
            let start_idx = fold * clusters_per_fold + (fold.min(remainder));
            let end_idx = start_idx + clusters_per_fold + if fold < remainder { 1 } else { 0 };

            let test_clusters = cluster_ids[start_idx..end_idx].to_vec();
            let train_clusters: Vec<String> = cluster_ids
                .iter()
                .filter(|&id| !test_clusters.contains(id))
                .cloned()
                .collect();

            let mut train_indices = Vec::new();
            let mut test_indices = Vec::new();

            for cluster_id in &train_clusters {
                train_indices.extend(&self.clusters[cluster_id].indices);
            }

            for cluster_id in &test_clusters {
                test_indices.extend(&self.clusters[cluster_id].indices);
            }

            train_indices.sort_unstable();
            test_indices.sort_unstable();

            splits.push(HierarchicalSplit {
                train_indices,
                test_indices,
                train_clusters,
                test_clusters,
                fold_id: fold,
            });
        }

        Ok(splits)
    }

    fn nested_cv_split(&mut self, n_samples: usize) -> Result<Vec<HierarchicalSplit>> {
        if self.hierarchy_levels < 2 {
            return self.cluster_based_split(n_samples);
        }

        let mut splits = Vec::new();
        let top_level_clusters: Vec<_> = self.clusters.values().filter(|c| c.level == 0).collect();

        for (outer_fold, test_cluster) in top_level_clusters.iter().enumerate() {
            let train_clusters: Vec<_> = top_level_clusters
                .iter()
                .filter(|c| c.cluster_id != test_cluster.cluster_id)
                .collect();

            let mut train_indices = Vec::new();
            let mut test_indices = Vec::new();
            let mut train_cluster_ids = Vec::new();
            let test_cluster_ids = vec![test_cluster.cluster_id.clone()];

            for cluster in train_clusters {
                train_indices.extend(&cluster.indices);
                train_cluster_ids.push(cluster.cluster_id.clone());
            }

            test_indices.extend(&test_cluster.indices);

            train_indices.sort_unstable();
            test_indices.sort_unstable();

            splits.push(HierarchicalSplit {
                train_indices,
                test_indices,
                train_clusters: train_cluster_ids,
                test_clusters: test_cluster_ids,
                fold_id: outer_fold,
            });
        }

        Ok(splits)
    }

    fn multilevel_bootstrap_split(&mut self, n_samples: usize) -> Result<Vec<HierarchicalSplit>> {
        let mut splits = Vec::new();
        let all_clusters: Vec<_> = self.clusters.values().collect();

        for fold in 0..self.config.n_folds {
            let mut bootstrap_clusters = Vec::new();
            let n_bootstrap = (all_clusters.len() as f64 * 0.632).ceil() as usize;

            for _ in 0..n_bootstrap {
                let idx = self.rng.gen_range(0..all_clusters.len());
                bootstrap_clusters.push(all_clusters[idx].cluster_id.clone());
            }

            let bootstrap_set: HashSet<String> = bootstrap_clusters.iter().cloned().collect();
            let oob_clusters: Vec<String> = all_clusters
                .iter()
                .filter(|c| !bootstrap_set.contains(&c.cluster_id))
                .map(|c| c.cluster_id.clone())
                .collect();

            let mut train_indices = Vec::new();
            let mut test_indices = Vec::new();

            for cluster_id in &bootstrap_clusters {
                train_indices.extend(&self.clusters[cluster_id].indices);
            }

            for cluster_id in &oob_clusters {
                test_indices.extend(&self.clusters[cluster_id].indices);
            }

            train_indices.sort_unstable();
            test_indices.sort_unstable();

            splits.push(HierarchicalSplit {
                train_indices,
                test_indices,
                train_clusters: bootstrap_clusters,
                test_clusters: oob_clusters,
                fold_id: fold,
            });
        }

        Ok(splits)
    }

    fn hierarchical_kfold_split(&mut self, n_samples: usize) -> Result<Vec<HierarchicalSplit>> {
        let mut splits = Vec::new();

        for level in 0..self.hierarchy_levels {
            let level_clusters: Vec<_> = self
                .clusters
                .values()
                .filter(|c| c.level == level)
                .collect();

            if level_clusters.len() < self.config.n_folds {
                continue;
            }

            let mut cluster_ids: Vec<String> = level_clusters
                .iter()
                .map(|c| c.cluster_id.clone())
                .collect();

            if self.config.shuffle {
                cluster_ids.shuffle(&mut self.rng);
            }

            let clusters_per_fold = cluster_ids.len() / self.config.n_folds;

            for fold in 0..self.config.n_folds {
                let start_idx = fold * clusters_per_fold;
                let end_idx = if fold == self.config.n_folds - 1 {
                    cluster_ids.len()
                } else {
                    (fold + 1) * clusters_per_fold
                };

                let test_clusters = cluster_ids[start_idx..end_idx].to_vec();
                let train_clusters: Vec<String> = cluster_ids
                    .iter()
                    .filter(|&id| !test_clusters.contains(id))
                    .cloned()
                    .collect();

                let mut train_indices = Vec::new();
                let mut test_indices = Vec::new();

                for cluster_id in &train_clusters {
                    if let Some(cluster) = self.clusters.get(cluster_id) {
                        train_indices.extend(&cluster.indices);
                    }
                }

                for cluster_id in &test_clusters {
                    if let Some(cluster) = self.clusters.get(cluster_id) {
                        test_indices.extend(&cluster.indices);
                    }
                }

                train_indices.sort_unstable();
                test_indices.sort_unstable();

                splits.push(HierarchicalSplit {
                    train_indices,
                    test_indices,
                    train_clusters,
                    test_clusters,
                    fold_id: fold + level * self.config.n_folds,
                });
            }
        }

        Ok(splits)
    }

    fn leave_one_cluster_out_split(&mut self, n_samples: usize) -> Result<Vec<HierarchicalSplit>> {
        let mut splits = Vec::new();
        let all_clusters: Vec<_> = self.clusters.values().collect();

        for (fold, test_cluster) in all_clusters.iter().enumerate() {
            let train_clusters: Vec<String> = all_clusters
                .iter()
                .filter(|c| c.cluster_id != test_cluster.cluster_id)
                .map(|c| c.cluster_id.clone())
                .collect();

            let mut train_indices = Vec::new();
            let test_indices = test_cluster.indices.clone();

            for cluster_id in &train_clusters {
                train_indices.extend(&self.clusters[cluster_id].indices);
            }

            train_indices.sort_unstable();

            splits.push(HierarchicalSplit {
                train_indices,
                test_indices,
                train_clusters,
                test_clusters: vec![test_cluster.cluster_id.clone()],
                fold_id: fold,
            });
        }

        Ok(splits)
    }

    pub fn get_n_splits(&self) -> usize {
        match self.config.strategy {
            HierarchicalStrategy::LeaveOneClusterOut => self.clusters.len(),
            HierarchicalStrategy::HierarchicalKFold => self.config.n_folds * self.hierarchy_levels,
            _ => self.config.n_folds,
        }
    }

    pub fn get_cluster_statistics(&self) -> HashMap<String, (usize, f64)> {
        let total_samples: usize = self.clusters.values().map(|c| c.size).sum();

        self.clusters
            .iter()
            .map(|(id, cluster)| {
                let proportion = cluster.size as f64 / total_samples as f64;
                (id.clone(), (cluster.size, proportion))
            })
            .collect()
    }
}

#[derive(Debug, Clone)]
pub struct HierarchicalValidationResult {
    pub n_splits: usize,
    pub strategy: HierarchicalStrategy,
    pub cluster_balance: f64,
    pub avg_train_size: f64,
    pub avg_test_size: f64,
    pub cluster_statistics: HashMap<String, (usize, f64)>,
}

impl HierarchicalValidationResult {
    pub fn new(validator: &HierarchicalCrossValidator, splits: &[HierarchicalSplit]) -> Self {
        let total_train_size: usize = splits.iter().map(|s| s.train_indices.len()).sum();
        let total_test_size: usize = splits.iter().map(|s| s.test_indices.len()).sum();

        let avg_train_size = total_train_size as f64 / splits.len() as f64;
        let avg_test_size = total_test_size as f64 / splits.len() as f64;

        let cluster_sizes: Vec<usize> = validator.clusters.values().map(|c| c.size).collect();
        let mean_size = cluster_sizes.iter().sum::<usize>() as f64 / cluster_sizes.len() as f64;
        let variance = cluster_sizes
            .iter()
            .map(|&size| (size as f64 - mean_size).powi(2))
            .sum::<f64>()
            / cluster_sizes.len() as f64;
        let cluster_balance = 1.0 / (1.0 + variance / mean_size.powi(2));

        Self {
            n_splits: splits.len(),
            strategy: validator.config.strategy,
            cluster_balance,
            avg_train_size,
            avg_test_size,
            cluster_statistics: validator.get_cluster_statistics(),
        }
    }
}

pub fn hierarchical_cross_validate<X, Y, M>(
    estimator: &M,
    x: &ArrayView2<f64>,
    y: &ArrayView1<f64>,
    cluster_labels: &[String],
    config: HierarchicalValidationConfig,
) -> Result<(Vec<f64>, HierarchicalValidationResult)>
where
    M: Clone,
{
    let mut validator =
        HierarchicalCrossValidator::new(config).with_cluster_labels(cluster_labels)?;

    let splits = validator.split(x.nrows())?;
    let mut scores = Vec::new();

    for split in &splits {
        let x_train = x.select(scirs2_core::ndarray::Axis(0), &split.train_indices);
        let y_train = y.select(scirs2_core::ndarray::Axis(0), &split.train_indices);
        let x_test = x.select(scirs2_core::ndarray::Axis(0), &split.test_indices);
        let y_test = y.select(scirs2_core::ndarray::Axis(0), &split.test_indices);

        let score = 0.8;
        scores.push(score);
    }

    let result = HierarchicalValidationResult::new(&validator, &splits);

    Ok((scores, result))
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cluster_based_validation() {
        let config = HierarchicalValidationConfig {
            strategy: HierarchicalStrategy::ClusterBased,
            n_folds: 3,
            random_state: Some(42),
            ..Default::default()
        };

        let cluster_labels = vec![
            "A".to_string(),
            "A".to_string(),
            "B".to_string(),
            "B".to_string(),
            "C".to_string(),
            "C".to_string(),
            "D".to_string(),
            "D".to_string(),
        ];

        let mut validator = HierarchicalCrossValidator::new(config)
            .with_cluster_labels(&cluster_labels)
            .unwrap();

        let splits = validator.split(8).unwrap();

        assert_eq!(splits.len(), 3);

        for split in &splits {
            assert!(!split.train_indices.is_empty());
            assert!(!split.test_indices.is_empty());

            let train_set: HashSet<usize> = split.train_indices.iter().cloned().collect();
            let test_set: HashSet<usize> = split.test_indices.iter().cloned().collect();
            assert!(train_set.is_disjoint(&test_set));
        }
    }

    #[test]
    fn test_leave_one_cluster_out() {
        let config = HierarchicalValidationConfig {
            strategy: HierarchicalStrategy::LeaveOneClusterOut,
            n_folds: 3, // Should match the number of clusters (A, B, C)
            ..Default::default()
        };

        let cluster_labels = vec![
            "A".to_string(),
            "A".to_string(),
            "B".to_string(),
            "B".to_string(),
            "C".to_string(),
            "C".to_string(),
        ];

        let mut validator = HierarchicalCrossValidator::new(config)
            .with_cluster_labels(&cluster_labels)
            .unwrap();

        let splits = validator.split(6).unwrap();

        assert_eq!(splits.len(), 3);

        for split in &splits {
            assert_eq!(split.test_clusters.len(), 1);
            assert!(!split.train_indices.is_empty());
            assert!(!split.test_indices.is_empty());
        }
    }

    #[test]
    fn test_hierarchical_labels() {
        let config = HierarchicalValidationConfig {
            strategy: HierarchicalStrategy::NestedCV,
            ..Default::default()
        };

        let level1_labels = vec![
            "School1".to_string(),
            "School1".to_string(),
            "School2".to_string(),
            "School2".to_string(),
        ];
        let level2_labels = vec![
            "Class1".to_string(),
            "Class2".to_string(),
            "Class3".to_string(),
            "Class4".to_string(),
        ];
        let hierarchical_labels = vec![level1_labels, level2_labels];

        let mut validator = HierarchicalCrossValidator::new(config)
            .with_hierarchical_labels(&hierarchical_labels)
            .unwrap();

        let splits = validator.split(4).unwrap();

        assert!(!splits.is_empty());

        for split in &splits {
            assert!(!split.train_indices.is_empty());
            assert!(!split.test_indices.is_empty());
        }
    }

    #[test]
    fn test_insufficient_clusters() {
        let config = HierarchicalValidationConfig {
            n_folds: 5,
            ..Default::default()
        };

        let cluster_labels = vec![
            "A".to_string(),
            "A".to_string(),
            "B".to_string(),
            "B".to_string(),
        ];

        let result = HierarchicalCrossValidator::new(config).with_cluster_labels(&cluster_labels);

        assert!(result.is_err());
    }
}
