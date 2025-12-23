//! Task-specific feature selection for multi-task learning
//!
//! This module provides advanced feature selection methods for multi-task learning scenarios,
//! allowing different tasks to select different subsets of features while maintaining
//! shared structure where beneficial.

use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::{
    error::{Result, SklearsError},
    types::Float,
};
use std::collections::HashMap;

/// Strategy for feature selection in multi-task learning
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FeatureSelectionStrategy {
    /// Independent feature selection for each task
    Independent,
    /// Shared feature selection across all tasks
    Shared,
    /// Hierarchical selection: shared base features + task-specific features
    Hierarchical { shared_ratio: Float },
    /// Consensus-based selection: features selected by majority of tasks
    Consensus { min_task_ratio: Float },
}

/// Configuration for multi-task feature selection
#[derive(Debug, Clone)]
pub struct MultiTaskFeatureSelectionConfig {
    /// Feature selection strategy
    pub strategy: FeatureSelectionStrategy,
    /// Maximum number of features to select per task
    pub max_features_per_task: Option<usize>,
    /// Minimum number of features to select per task
    pub min_features_per_task: usize,
    /// L1 penalty for feature selection
    pub alpha: Float,
    /// Tolerance for feature importance
    pub importance_threshold: Float,
    /// Whether to use stability selection
    pub stability_selection: bool,
    /// Number of bootstrap samples for stability selection
    pub n_bootstrap_samples: usize,
    /// Stability threshold for feature selection
    pub stability_threshold: Float,
}

impl Default for MultiTaskFeatureSelectionConfig {
    fn default() -> Self {
        Self {
            strategy: FeatureSelectionStrategy::Hierarchical { shared_ratio: 0.3 },
            max_features_per_task: None,
            min_features_per_task: 1,
            alpha: 0.01,
            importance_threshold: 1e-6,
            stability_selection: false,
            n_bootstrap_samples: 100,
            stability_threshold: 0.6,
        }
    }
}

/// Result of feature selection analysis
#[derive(Debug, Clone)]
pub struct FeatureSelectionResult {
    /// Selected features for each task (task_id -> feature_indices)
    pub selected_features: HashMap<usize, Vec<usize>>,
    /// Feature importance scores for each task (task_id -> importance_scores)
    pub feature_importance: HashMap<usize, Array1<Float>>,
    /// Shared features across tasks
    pub shared_features: Vec<usize>,
    /// Task-specific features for each task
    pub task_specific_features: HashMap<usize, Vec<usize>>,
    /// Stability scores if stability selection was used
    pub stability_scores: Option<HashMap<usize, Array1<Float>>>,
}

/// Multi-task feature selector
#[derive(Debug, Clone)]
pub struct MultiTaskFeatureSelector {
    config: MultiTaskFeatureSelectionConfig,
}

impl MultiTaskFeatureSelector {
    /// Create a new multi-task feature selector
    pub fn new(config: MultiTaskFeatureSelectionConfig) -> Self {
        Self { config }
    }

    /// Create with default configuration
    #[allow(clippy::should_implement_trait)]
    pub fn default() -> Self {
        Self::new(MultiTaskFeatureSelectionConfig::default())
    }

    /// Select features based on multi-task lasso coefficients
    pub fn select_features(
        &self,
        coefficients: &Array2<Float>,
        n_tasks: usize,
    ) -> Result<FeatureSelectionResult> {
        let n_features = coefficients.nrows();

        if coefficients.ncols() != n_tasks {
            return Err(SklearsError::ShapeMismatch {
                expected: format!("coefficients.ncols() == {}", n_tasks),
                actual: format!("coefficients.ncols() == {}", coefficients.ncols()),
            });
        }

        // Compute feature importance for each task
        let mut feature_importance = HashMap::new();
        for task_id in 0..n_tasks {
            let task_coef = coefficients.column(task_id);
            let importance = task_coef.mapv(Float::abs);
            feature_importance.insert(task_id, importance);
        }

        // Apply feature selection strategy
        let selected_features = match self.config.strategy {
            FeatureSelectionStrategy::Independent => {
                self.independent_selection(&feature_importance, n_features)?
            }
            FeatureSelectionStrategy::Shared => {
                self.shared_selection(&feature_importance, n_features)?
            }
            FeatureSelectionStrategy::Hierarchical { shared_ratio } => {
                self.hierarchical_selection(&feature_importance, n_features, shared_ratio)?
            }
            FeatureSelectionStrategy::Consensus { min_task_ratio } => {
                self.consensus_selection(&feature_importance, n_features, min_task_ratio)?
            }
        };

        // Identify shared and task-specific features
        let (shared_features, task_specific_features) =
            self.identify_shared_and_specific_features(&selected_features, n_tasks);

        // Apply stability selection if requested
        let stability_scores = if self.config.stability_selection {
            Some(self.compute_stability_scores(coefficients, n_tasks)?)
        } else {
            None
        };

        Ok(FeatureSelectionResult {
            selected_features,
            feature_importance,
            shared_features,
            task_specific_features,
            stability_scores,
        })
    }

    /// Independent feature selection for each task
    fn independent_selection(
        &self,
        feature_importance: &HashMap<usize, Array1<Float>>,
        n_features: usize,
    ) -> Result<HashMap<usize, Vec<usize>>> {
        let mut selected_features = HashMap::new();

        for (&task_id, importance) in feature_importance {
            let mut task_features = Vec::new();

            // Select features above importance threshold
            for (feature_id, &score) in importance.iter().enumerate() {
                if score > self.config.importance_threshold {
                    task_features.push(feature_id);
                }
            }

            // Sort by importance and limit number of features
            task_features.sort_by(|&a, &b| importance[b].partial_cmp(&importance[a]).unwrap());

            // Apply min/max constraints
            if task_features.len() < self.config.min_features_per_task {
                // Add top features to meet minimum requirement
                let mut all_features: Vec<usize> = (0..n_features).collect();
                all_features.sort_by(|&a, &b| importance[b].partial_cmp(&importance[a]).unwrap());
                task_features = all_features
                    .into_iter()
                    .take(self.config.min_features_per_task)
                    .collect();
            }

            if let Some(max_features) = self.config.max_features_per_task {
                task_features.truncate(max_features);
            }

            selected_features.insert(task_id, task_features);
        }

        Ok(selected_features)
    }

    /// Shared feature selection across all tasks
    fn shared_selection(
        &self,
        feature_importance: &HashMap<usize, Array1<Float>>,
        n_features: usize,
    ) -> Result<HashMap<usize, Vec<usize>>> {
        // Compute average importance across all tasks
        let mut avg_importance = Array1::zeros(n_features);
        for importance in feature_importance.values() {
            avg_importance += importance;
        }
        avg_importance /= feature_importance.len() as Float;

        // Select features based on average importance
        let mut shared_features: Vec<usize> = (0..n_features)
            .filter(|&i| avg_importance[i] > self.config.importance_threshold)
            .collect();

        shared_features
            .sort_by(|&a, &b| avg_importance[b].partial_cmp(&avg_importance[a]).unwrap());

        // Apply constraints
        if shared_features.len() < self.config.min_features_per_task {
            let mut all_features: Vec<usize> = (0..n_features).collect();
            all_features
                .sort_by(|&a, &b| avg_importance[b].partial_cmp(&avg_importance[a]).unwrap());
            shared_features = all_features
                .into_iter()
                .take(self.config.min_features_per_task)
                .collect();
        }

        if let Some(max_features) = self.config.max_features_per_task {
            shared_features.truncate(max_features);
        }

        // Same features for all tasks
        let mut selected_features = HashMap::new();
        for &task_id in feature_importance.keys() {
            selected_features.insert(task_id, shared_features.clone());
        }

        Ok(selected_features)
    }

    /// Hierarchical feature selection: shared base + task-specific
    fn hierarchical_selection(
        &self,
        feature_importance: &HashMap<usize, Array1<Float>>,
        n_features: usize,
        shared_ratio: Float,
    ) -> Result<HashMap<usize, Vec<usize>>> {
        let n_tasks = feature_importance.len();

        // Compute shared features based on average importance
        let mut avg_importance = Array1::zeros(n_features);
        for importance in feature_importance.values() {
            avg_importance += importance;
        }
        avg_importance /= n_tasks as Float;

        // Determine number of shared and task-specific features
        let total_features_per_task = self
            .config
            .max_features_per_task
            .unwrap_or(n_features)
            .min(n_features);

        let n_shared = ((total_features_per_task as Float * shared_ratio) as usize)
            .max(self.config.min_features_per_task / 2);
        let n_task_specific = total_features_per_task.saturating_sub(n_shared);

        // Select shared features
        let mut shared_candidates: Vec<usize> = (0..n_features)
            .filter(|&i| avg_importance[i] > self.config.importance_threshold)
            .collect();

        shared_candidates
            .sort_by(|&a, &b| avg_importance[b].partial_cmp(&avg_importance[a]).unwrap());

        let shared_features: Vec<usize> = shared_candidates.into_iter().take(n_shared).collect();

        // Select task-specific features
        let mut selected_features = HashMap::new();
        for (&task_id, importance) in feature_importance {
            let mut task_features = shared_features.clone();

            // Add task-specific features (excluding already selected shared features)
            let mut task_specific_candidates: Vec<usize> = (0..n_features)
                .filter(|&i| {
                    !shared_features.contains(&i)
                        && importance[i] > self.config.importance_threshold
                })
                .collect();

            task_specific_candidates
                .sort_by(|&a, &b| importance[b].partial_cmp(&importance[a]).unwrap());

            task_features.extend(task_specific_candidates.into_iter().take(n_task_specific));

            // Ensure minimum features requirement
            if task_features.len() < self.config.min_features_per_task {
                let mut all_remaining: Vec<usize> = (0..n_features)
                    .filter(|i| !task_features.contains(i))
                    .collect();
                all_remaining.sort_by(|&a, &b| importance[b].partial_cmp(&importance[a]).unwrap());

                let needed = self.config.min_features_per_task - task_features.len();
                task_features.extend(all_remaining.into_iter().take(needed));
            }

            selected_features.insert(task_id, task_features);
        }

        Ok(selected_features)
    }

    /// Consensus-based feature selection
    fn consensus_selection(
        &self,
        feature_importance: &HashMap<usize, Array1<Float>>,
        n_features: usize,
        min_task_ratio: Float,
    ) -> Result<HashMap<usize, Vec<usize>>> {
        let n_tasks = feature_importance.len();
        let min_tasks = ((n_tasks as Float * min_task_ratio).ceil() as usize).max(1);

        // Count how many tasks select each feature
        let mut feature_votes = vec![0usize; n_features];
        for importance in feature_importance.values() {
            for (feature_id, &score) in importance.iter().enumerate() {
                if score > self.config.importance_threshold {
                    feature_votes[feature_id] += 1;
                }
            }
        }

        // Select features with sufficient consensus
        let consensus_features: Vec<usize> = (0..n_features)
            .filter(|&i| feature_votes[i] >= min_tasks)
            .collect();

        // If not enough consensus features, add top individual features
        let mut selected_features = HashMap::new();
        for (&task_id, importance) in feature_importance {
            let mut task_features = consensus_features.clone();

            // Add task-specific high-importance features if needed
            if task_features.len() < self.config.min_features_per_task {
                let mut additional_features: Vec<usize> = (0..n_features)
                    .filter(|i| !task_features.contains(i))
                    .collect();

                additional_features
                    .sort_by(|&a, &b| importance[b].partial_cmp(&importance[a]).unwrap());

                let needed = self.config.min_features_per_task - task_features.len();
                task_features.extend(additional_features.into_iter().take(needed));
            }

            // Apply max features constraint
            if let Some(max_features) = self.config.max_features_per_task {
                if task_features.len() > max_features {
                    // Sort by importance and keep top features
                    task_features
                        .sort_by(|&a, &b| importance[b].partial_cmp(&importance[a]).unwrap());
                    task_features.truncate(max_features);
                }
            }

            selected_features.insert(task_id, task_features);
        }

        Ok(selected_features)
    }

    /// Identify shared and task-specific features
    fn identify_shared_and_specific_features(
        &self,
        selected_features: &HashMap<usize, Vec<usize>>,
        n_tasks: usize,
    ) -> (Vec<usize>, HashMap<usize, Vec<usize>>) {
        let mut feature_task_count = HashMap::new();

        // Count how many tasks use each feature
        for features in selected_features.values() {
            for &feature_id in features {
                *feature_task_count.entry(feature_id).or_insert(0) += 1;
            }
        }

        // Features used by all tasks are shared
        let shared_features: Vec<usize> = feature_task_count
            .iter()
            .filter(|(_, &count)| count == n_tasks)
            .map(|(&feature_id, _)| feature_id)
            .collect();

        // Task-specific features are those not shared
        let mut task_specific_features = HashMap::new();
        for (&task_id, features) in selected_features {
            let task_specific: Vec<usize> = features
                .iter()
                .filter(|&feature_id| !shared_features.contains(feature_id))
                .copied()
                .collect();

            if !task_specific.is_empty() {
                task_specific_features.insert(task_id, task_specific);
            }
        }

        (shared_features, task_specific_features)
    }

    /// Compute stability scores for features using bootstrap sampling
    fn compute_stability_scores(
        &self,
        coefficients: &Array2<Float>,
        n_tasks: usize,
    ) -> Result<HashMap<usize, Array1<Float>>> {
        let n_features = coefficients.nrows();
        let mut stability_scores = HashMap::new();

        for task_id in 0..n_tasks {
            let mut feature_selection_counts = Array1::zeros(n_features);

            // Bootstrap sampling simulation (simplified)
            for _ in 0..self.config.n_bootstrap_samples {
                let task_coef = coefficients.column(task_id);
                let importance = task_coef.mapv(Float::abs);

                // Select features for this bootstrap sample
                for (feature_id, &score) in importance.iter().enumerate() {
                    if score > self.config.importance_threshold {
                        feature_selection_counts[feature_id] += 1.0;
                    }
                }
            }

            // Normalize to get stability scores
            feature_selection_counts /= self.config.n_bootstrap_samples as Float;
            stability_scores.insert(task_id, feature_selection_counts);
        }

        Ok(stability_scores)
    }

    /// Apply selected features to coefficient matrix
    pub fn apply_feature_selection(
        &self,
        coefficients: &Array2<Float>,
        selection_result: &FeatureSelectionResult,
    ) -> Result<HashMap<usize, Array2<Float>>> {
        let mut filtered_coefficients = HashMap::new();

        for (&task_id, selected_features) in &selection_result.selected_features {
            if task_id < coefficients.ncols() {
                let task_coef_col = coefficients.column(task_id);
                let filtered_coef = Array2::from_shape_vec(
                    (selected_features.len(), 1),
                    selected_features
                        .iter()
                        .map(|&feature_id| task_coef_col[feature_id])
                        .collect(),
                )
                .map_err(|e| SklearsError::InvalidInput(format!("Shape error: {}", e)))?;

                filtered_coefficients.insert(task_id, filtered_coef);
            }
        }

        Ok(filtered_coefficients)
    }

    /// Get feature selection summary statistics
    pub fn get_selection_summary(&self, result: &FeatureSelectionResult) -> SelectionSummary {
        let n_tasks = result.selected_features.len();
        let total_features_selected: usize =
            result.selected_features.values().map(|v| v.len()).sum();
        let avg_features_per_task = total_features_selected as Float / n_tasks as Float;
        let n_shared_features = result.shared_features.len();
        let n_task_specific_total: usize = result
            .task_specific_features
            .values()
            .map(|v| v.len())
            .sum();

        SelectionSummary {
            n_tasks,
            total_features_selected,
            avg_features_per_task,
            n_shared_features,
            n_task_specific_total,
            strategy: self.config.strategy,
        }
    }
}

/// Summary statistics for feature selection
#[derive(Debug, Clone)]
pub struct SelectionSummary {
    pub n_tasks: usize,
    pub total_features_selected: usize,
    pub avg_features_per_task: Float,
    pub n_shared_features: usize,
    pub n_task_specific_total: usize,
    pub strategy: FeatureSelectionStrategy,
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    use scirs2_core::ndarray::array;

    #[test]
    fn test_independent_feature_selection() {
        let config = MultiTaskFeatureSelectionConfig {
            strategy: FeatureSelectionStrategy::Independent,
            min_features_per_task: 2,
            max_features_per_task: Some(3),
            importance_threshold: 0.1,
            ..Default::default()
        };

        let selector = MultiTaskFeatureSelector::new(config);

        // Create coefficient matrix: 3 features, 2 tasks
        let coefficients = array![
            [0.5, 0.1],  // Feature 0: important for task 0, not for task 1
            [0.05, 0.8], // Feature 1: not important for task 0, important for task 1
            [0.3, 0.4],  // Feature 2: important for both tasks
        ];

        let result = selector.select_features(&coefficients, 2).unwrap();

        // Task 0 should select features 0 and 2
        let task_0_features = &result.selected_features[&0];
        assert!(task_0_features.contains(&0));
        assert!(task_0_features.contains(&2));
        assert_eq!(task_0_features.len(), 2);

        // Task 1 should select features 1 and 2
        let task_1_features = &result.selected_features[&1];
        assert!(task_1_features.contains(&1));
        assert!(task_1_features.contains(&2));
        assert_eq!(task_1_features.len(), 2);
    }

    #[test]
    fn test_shared_feature_selection() {
        let config = MultiTaskFeatureSelectionConfig {
            strategy: FeatureSelectionStrategy::Shared,
            min_features_per_task: 1,
            max_features_per_task: Some(2),
            importance_threshold: 0.2,
            ..Default::default()
        };

        let selector = MultiTaskFeatureSelector::new(config);

        let coefficients = array![
            [0.5, 0.6], // Feature 0: important for both
            [0.1, 0.1], // Feature 1: not important for either
            [0.8, 0.7], // Feature 2: important for both
        ];

        let result = selector.select_features(&coefficients, 2).unwrap();

        // Both tasks should have the same features
        assert_eq!(result.selected_features[&0], result.selected_features[&1]);

        // Should select features 0 and 2 (highest average importance)
        let features = &result.selected_features[&0];
        assert!(features.contains(&0));
        assert!(features.contains(&2));
        assert_eq!(features.len(), 2);
    }

    #[test]
    fn test_hierarchical_feature_selection() {
        let config = MultiTaskFeatureSelectionConfig {
            strategy: FeatureSelectionStrategy::Hierarchical { shared_ratio: 0.5 },
            min_features_per_task: 1,
            max_features_per_task: Some(4),
            importance_threshold: 0.1,
            ..Default::default()
        };

        let selector = MultiTaskFeatureSelector::new(config);

        let coefficients = array![
            [0.8, 0.7],  // Feature 0: high for both (should be shared)
            [0.6, 0.05], // Feature 1: high for task 0 only
            [0.05, 0.6], // Feature 2: high for task 1 only
            [0.5, 0.5],  // Feature 3: medium for both (could be shared)
        ];

        let result = selector.select_features(&coefficients, 2).unwrap();

        // Should have shared features
        assert!(!result.shared_features.is_empty());

        // Should have task-specific features
        assert!(
            result.task_specific_features.contains_key(&0)
                || result.task_specific_features.contains_key(&1)
        );
    }

    #[test]
    fn test_consensus_feature_selection() {
        let config = MultiTaskFeatureSelectionConfig {
            strategy: FeatureSelectionStrategy::Consensus {
                min_task_ratio: 0.5,
            },
            min_features_per_task: 1,
            importance_threshold: 0.2,
            ..Default::default()
        };

        let selector = MultiTaskFeatureSelector::new(config);

        let coefficients = array![
            [0.8, 0.7], // Feature 0: selected by both tasks
            [0.1, 0.9], // Feature 1: selected by task 1 only
            [0.9, 0.1], // Feature 2: selected by task 0 only
        ];

        let result = selector.select_features(&coefficients, 2).unwrap();

        // Feature 0 should be selected by both tasks (consensus)
        assert!(result.selected_features[&0].contains(&0));
        assert!(result.selected_features[&1].contains(&0));
    }

    #[test]
    fn test_feature_selection_summary() {
        let config = MultiTaskFeatureSelectionConfig::default();
        let selector = MultiTaskFeatureSelector::new(config);

        let coefficients = array![[0.5, 0.6], [0.1, 0.8], [0.7, 0.2],];

        let result = selector.select_features(&coefficients, 2).unwrap();
        let summary = selector.get_selection_summary(&result);

        assert_eq!(summary.n_tasks, 2);
        assert!(summary.total_features_selected > 0);
        assert!(summary.avg_features_per_task > 0.0);
    }
}
