//! Multi-Level Decomposition Classifiers
//!
//! This module provides multi-level decomposition strategies for multiclass classification.
//! Multi-level decomposition performs classification at multiple hierarchical levels 
//! simultaneously, combining predictions from multiple levels to improve overall accuracy
//! and robustness.

use std::marker::PhantomData;

use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::{
    error::{SklResult},
    estimator::{Estimator, Trained, Untrained},
    Float,
};
use sklears_core::traits::Fit;

/// Multi-level decomposition classifier that performs classification at multiple
/// hierarchical levels simultaneously. Unlike traditional hierarchical classifiers
/// that classify at a single level, this approach combines predictions from
/// multiple levels to improve overall accuracy and robustness.
///
/// # Type Parameters
///
/// * `C` - The base classifier type that implements Fit and Predict
/// * `S` - The state type (Untrained or Trained)
///
/// # Examples
///
/// ```rust
/// use sklears_multiclass::{MultiLevelDecompositionClassifier, DecompositionLevel};
/// use scirs2_autograd::ndarray::array;
///
/// // Define multiple decomposition levels
/// let levels = vec![
///     DecompositionLevel::new(1, vec![vec![0, 1], vec![2, 3]]), // Level 1: binary groups
///     DecompositionLevel::new(2, vec![vec![0], vec![1], vec![2], vec![3]]), // Level 2: individual classes
/// ];
///
/// // Example with a hypothetical base classifier
/// // let base_classifier = SomeClassifier::new();
/// // let classifier = MultiLevelDecompositionClassifier::new(base_classifier, levels);
/// ```
#[derive(Debug)]
pub struct MultiLevelDecompositionClassifier<C, S = Untrained> {
    base_estimator: C,
    levels: Vec<DecompositionLevel>,
    config: MultiLevelConfig,
    state: PhantomData<S>,
}

/// Configuration for multi-level decomposition
#[derive(Debug, Clone)]
pub struct MultiLevelConfig {
    /// Strategy for combining level predictions
    pub combination_strategy: LevelCombinationStrategy,
    /// Weights for each level (None for equal weighting)
    pub level_weights: Option<Vec<f64>>,
    /// Confidence threshold for predictions
    pub confidence_threshold: f64,
    /// Whether to use early stopping at higher levels
    pub early_stopping: bool,
    /// Number of parallel jobs (-1 for all cores)
    pub n_jobs: Option<i32>,
    /// Random state for reproducible results
    pub random_state: Option<u64>,
}

impl Default for MultiLevelConfig {
    fn default() -> Self {
        Self {
            combination_strategy: LevelCombinationStrategy::WeightedAverage,
            level_weights: None,
            confidence_threshold: 0.5,
            early_stopping: false,
            n_jobs: None,
            random_state: None,
        }
    }
}

/// Strategies for combining predictions from multiple levels
#[derive(Debug, Clone, PartialEq)]
pub enum LevelCombinationStrategy {
    /// Simple average of all level predictions
    Average,
    /// Weighted average with learned or specified weights
    WeightedAverage,
    /// Maximum confidence prediction
    MaxConfidence,
    /// Majority voting across levels
    MajorityVoting,
    /// Hierarchical combination (higher levels override lower)
    Hierarchical,
    /// Ensemble-based combination with meta-learning
    Ensemble,
}

/// A decomposition level defining class groupings
#[derive(Debug, Clone)]
pub struct DecompositionLevel {
    /// Level identifier (typically depth in hierarchy)
    pub level_id: usize,
    /// Class groupings at this level
    pub class_groups: Vec<Vec<i32>>,
    /// Level-specific configuration
    pub config: LevelConfig,
}

/// Configuration for individual decomposition levels
#[derive(Debug, Clone)]
pub struct LevelConfig {
    /// Strategy for this level (OneVsRest, OneVsOne, etc.)
    pub strategy: LevelStrategy,
    /// Weight of this level in final prediction
    pub weight: f64,
    /// Whether this level can provide early stopping
    pub can_early_stop: bool,
    /// Minimum confidence for early stopping
    pub early_stop_confidence: f64,
}

impl Default for LevelConfig {
    fn default() -> Self {
        Self {
            strategy: LevelStrategy::OneVsRest,
            weight: 1.0,
            can_early_stop: false,
            early_stop_confidence: 0.9,
        }
    }
}

/// Strategies for individual decomposition levels
#[derive(Debug, Clone, PartialEq)]
pub enum LevelStrategy {
    /// One-vs-Rest at this level
    OneVsRest,
    /// One-vs-One at this level
    OneVsOne,
    /// Error-correcting codes at this level
    ECOC,
    /// Custom grouping strategy
    Custom,
}

impl DecompositionLevel {
    /// Create a new decomposition level
    pub fn new(level_id: usize, class_groups: Vec<Vec<i32>>) -> Self {
        Self {
            level_id,
            class_groups,
            config: LevelConfig::default(),
        }
    }

    /// Set the strategy for this level
    pub fn with_strategy(mut self, strategy: LevelStrategy) -> Self {
        self.config.strategy = strategy;
        self
    }

    /// Set the weight for this level
    pub fn with_weight(mut self, weight: f64) -> Self {
        self.config.weight = weight;
        self
    }

    /// Enable early stopping for this level
    pub fn with_early_stopping(mut self, confidence_threshold: f64) -> Self {
        self.config.can_early_stop = true;
        self.config.early_stop_confidence = confidence_threshold;
        self
    }

    /// Validate the level structure
    pub fn validate(&self, n_classes: usize) -> SklResult<()> {
        // Check that all classes are covered
        let mut covered_classes = std::collections::HashSet::new();

        for group in &self.class_groups {
            if group.is_empty() {
                return Err(SklearsError::InvalidInput(format!(
                    "Empty class group found in level {}",
                    self.level_id
                )));
            }

            for &class in group {
                if class < 0 || class >= n_classes as i32 {
                    return Err(SklearsError::InvalidInput(format!(
                        "Invalid class {} in level {}",
                        class, self.level_id
                    )));
                }
                covered_classes.insert(class);
            }
        }

        // Check that all classes are covered (for complete decomposition)
        if covered_classes.len() != n_classes {
            return Err(SklearsError::InvalidInput(format!(
                "Not all classes are covered in level {}",
                self.level_id
            )));
        }

        Ok(())
    }

    /// Get the group index for a given class
    pub fn get_group_for_class(&self, class: i32) -> Option<usize> {
        for (group_idx, group) in self.class_groups.iter().enumerate() {
            if group.contains(&class) {
                return Some(group_idx);
            }
        }
        None
    }

    /// Get the number of groups at this level
    pub fn n_groups(&self) -> usize {
        self.class_groups.len()
    }

    /// Check if this level provides binary decomposition
    pub fn is_binary_level(&self) -> bool {
        self.class_groups.len() == 2
    }
}

impl<C> MultiLevelDecompositionClassifier<C, Untrained> {
    /// Create a new multi-level decomposition classifier
    pub fn new(base_estimator: C, levels: Vec<DecompositionLevel>) -> SklResult<Self> {
        if levels.is_empty() {
            return Err(SklearsError::InvalidInput(
                "At least one decomposition level must be provided".to_string(),
            ));
        }

        // Basic validation - detailed validation happens during fit
        for level in &levels {
            if level.class_groups.is_empty() {
                return Err(SklearsError::InvalidInput(format!(
                    "Level {} has no class groups",
                    level.level_id
                )));
            }
        }

        Ok(Self {
            base_estimator,
            levels,
            config: MultiLevelConfig::default(),
            state: PhantomData,
        })
    }

    /// Create a builder for MultiLevelDecompositionClassifier
    pub fn builder(
        base_estimator: C,
        levels: Vec<DecompositionLevel>,
    ) -> SklResult<MultiLevelBuilder<C>> {
        // Validate basic structure
        if levels.is_empty() {
            return Err(SklearsError::InvalidInput(
                "At least one decomposition level must be provided".to_string(),
            ));
        }

        Ok(MultiLevelBuilder::new(base_estimator, levels))
    }

    /// Set the combination strategy
    pub fn combination_strategy(mut self, strategy: LevelCombinationStrategy) -> Self {
        self.config.combination_strategy = strategy;
        self
    }

    /// Set level weights
    pub fn level_weights(mut self, weights: Vec<f64>) -> Self {
        self.config.level_weights = Some(weights);
        self
    }

    /// Set confidence threshold
    pub fn confidence_threshold(mut self, threshold: f64) -> Self {
        self.config.confidence_threshold = threshold;
        self
    }

    /// Enable or disable early stopping
    pub fn early_stopping(mut self, early_stopping: bool) -> Self {
        self.config.early_stopping = early_stopping;
        self
    }

    /// Set number of parallel jobs
    pub fn n_jobs(mut self, n_jobs: Option<i32>) -> Self {
        self.config.n_jobs = n_jobs;
        self
    }

    /// Set random state
    pub fn random_state(mut self, random_state: u64) -> Self {
        self.config.random_state = Some(random_state);
        self
    }

    /// Get a reference to the base estimator
    pub fn base_estimator(&self) -> &C {
        &self.base_estimator
    }

    /// Get the decomposition levels
    pub fn levels(&self) -> &[DecompositionLevel] {
        &self.levels
    }

    /// Get the number of levels
    pub fn n_levels(&self) -> usize {
        self.levels.len()
    }
}

/// Builder for MultiLevelDecompositionClassifier
#[derive(Debug)]
pub struct MultiLevelBuilder<C> {
    base_estimator: C,
    levels: Vec<DecompositionLevel>,
    config: MultiLevelConfig,
}

impl<C> MultiLevelBuilder<C> {
    /// Create a new builder
    fn new(base_estimator: C, levels: Vec<DecompositionLevel>) -> Self {
        Self {
            base_estimator,
            levels,
            config: MultiLevelConfig::default(),
        }
    }

    /// Set the combination strategy
    pub fn combination_strategy(mut self, strategy: LevelCombinationStrategy) -> Self {
        self.config.combination_strategy = strategy;
        self
    }

    /// Set level weights
    pub fn level_weights(mut self, weights: Vec<f64>) -> Self {
        self.config.level_weights = Some(weights);
        self
    }

    /// Set confidence threshold
    pub fn confidence_threshold(mut self, threshold: f64) -> Self {
        self.config.confidence_threshold = threshold;
        self
    }

    /// Enable or disable early stopping
    pub fn early_stopping(mut self, early_stopping: bool) -> Self {
        self.config.early_stopping = early_stopping;
        self
    }

    /// Set number of parallel jobs
    pub fn n_jobs(mut self, n_jobs: Option<i32>) -> Self {
        self.config.n_jobs = n_jobs;
        self
    }

    /// Set random state
    pub fn random_state(mut self, random_state: u64) -> Self {
        self.config.random_state = Some(random_state);
        self
    }

    /// Build the MultiLevelDecompositionClassifier
    pub fn build(self) -> MultiLevelDecompositionClassifier<C, Untrained> {
        MultiLevelDecompositionClassifier {
            base_estimator: self.base_estimator,
            levels: self.levels,
            config: self.config,
            state: PhantomData,
        }
    }
}

impl<C: Clone> Clone for MultiLevelDecompositionClassifier<C, Untrained> {
    fn clone(&self) -> Self {
        Self {
            base_estimator: self.base_estimator.clone(),
            levels: self.levels.clone(),
            config: self.config.clone(),
            state: PhantomData,
        }
    }
}

impl<C> Estimator for MultiLevelDecompositionClassifier<C, Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

/// Trained multi-level decomposition classifier state
#[derive(Debug)]
pub struct TrainedMultiLevel<C> {
    /// Classifiers for each level and group
    level_classifiers: HashMap<(usize, usize), C>, // (level_id, group_id) -> classifier
    /// Decomposition levels
    levels: Vec<DecompositionLevel>,
    /// Configuration
    config: MultiLevelConfig,
    /// Classes seen during training
    classes: Array1<i32>,
    /// Number of features seen during training
    n_features: usize,
    /// Level combination weights (learned or specified)
    combination_weights: Vec<f64>,
}

/// Type alias for trained multi-level decomposition classifier
pub type TrainedMultiLevelDecompositionClassifier<C> =
    MultiLevelDecompositionClassifier<TrainedMultiLevel<C>, Trained>;

impl<C> TrainedMultiLevel<C> {
    /// Get the decomposition levels
    pub fn levels(&self) -> &[DecompositionLevel] {
        &self.levels
    }

    /// Get the number of classes
    pub fn n_classes(&self) -> usize {
        self.classes.len()
    }

    /// Get the classes
    pub fn classes(&self) -> &Array1<i32> {
        &self.classes
    }

    /// Get classifier for a specific level and group
    pub fn get_level_classifier(&self, level_id: usize, group_id: usize) -> Option<&C> {
        self.level_classifiers.get(&(level_id, group_id))
    }

    /// Get the number of levels
    pub fn n_levels(&self) -> usize {
        self.levels.len()
    }

    /// Get combination weights
    pub fn combination_weights(&self) -> &[f64] {
        &self.combination_weights
    }

    /// Get the level configuration
    pub fn get_level_config(&self, level_id: usize) -> Option<&LevelConfig> {
        self.levels
            .iter()
            .find(|level| level.level_id == level_id)
            .map(|level| &level.config)
    }
}

/// Utilities for creating common multi-level decomposition patterns
pub struct MultiLevelDecompositionUtils;

impl MultiLevelDecompositionUtils {
    /// Create a binary tree decomposition with multiple levels
    pub fn binary_tree_decomposition(n_classes: usize) -> SklResult<Vec<DecompositionLevel>> {
        if n_classes < 2 {
            return Err(SklearsError::InvalidInput(
                "Need at least 2 classes for binary tree decomposition".to_string(),
            ));
        }

        let mut levels = Vec::new();
        let mut current_groups = vec![(0..n_classes as i32).collect::<Vec<_>>()];
        let mut level_id = 0;

        while current_groups.iter().any(|group| group.len() > 1) {
            let mut next_groups = Vec::new();

            for group in current_groups {
                if group.len() > 1 {
                    let mid = group.len() / 2;
                    let left = group[..mid].to_vec();
                    let right = group[mid..].to_vec();
                    next_groups.push(left);
                    next_groups.push(right);
                } else {
                    next_groups.push(group);
                }
            }

            levels.push(DecompositionLevel::new(level_id, next_groups.clone()));
            current_groups = next_groups;
            level_id += 1;
        }

        Ok(levels)
    }

    /// Create a multi-level OvR decomposition
    pub fn multi_level_ovr(
        n_classes: usize,
        n_levels: usize,
    ) -> SklResult<Vec<DecompositionLevel>> {
        if n_classes < 2 {
            return Err(SklearsError::InvalidInput(
                "Need at least 2 classes for multi-level OvR".to_string(),
            ));
        }

        let mut levels = Vec::new();

        for level_id in 0..n_levels {
            let group_size = (n_classes + level_id) / (level_id + 1);
            let mut groups = Vec::new();

            for i in (0..n_classes).step_by(group_size.max(1)) {
                let end = (i + group_size).min(n_classes);
                if i < end {
                    groups.push((i as i32..end as i32).collect());
                }
            }

            levels.push(
                DecompositionLevel::new(level_id, groups).with_strategy(LevelStrategy::OneVsRest),
            );
        }

        Ok(levels)
    }

    /// Create a hierarchical multi-level decomposition
    pub fn hierarchical_decomposition(
        class_hierarchy: &[(i32, Vec<i32>)],
    ) -> SklResult<Vec<DecompositionLevel>> {
        // class_hierarchy is (parent_class, child_classes) pairs
        let mut levels = Vec::new();
        let mut level_map: std::collections::HashMap<i32, usize> = std::collections::HashMap::new();

        // Determine level for each class based on hierarchy depth
        for (parent, children) in class_hierarchy {
            let parent_level = *level_map.get(parent).unwrap_or(&0);
            for &child in children {
                level_map.insert(child, parent_level + 1);
            }
        }

        // Group classes by level
        let max_level = level_map.values().max().unwrap_or(&0);

        for level_id in 0..=*max_level {
            let mut groups = Vec::new();

            // Find all parent-children relationships at this level
            for (parent, children) in class_hierarchy {
                if level_map.get(parent).unwrap_or(&0) == &level_id {
                    if !children.is_empty() {
                        groups.push(children.clone());
                    } else {
                        groups.push(vec![*parent]); // Leaf node
                    }
                }
            }

            if !groups.is_empty() {
                levels.push(DecompositionLevel::new(level_id, groups));
            }
        }

        Ok(levels)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::MockNativeClassifier;
    use scirs2_autograd::ndarray::array;

    fn create_simple_levels() -> Vec<DecompositionLevel> {
        vec![
            DecompositionLevel::new(0, vec![vec![0, 1], vec![2, 3]]), // Level 0: binary split
            DecompositionLevel::new(1, vec![vec![0], vec![1], vec![2], vec![3]]), // Level 1: individual classes
        ]
    }

    #[test]
    fn test_decomposition_level_creation() {
        let level = DecompositionLevel::new(0, vec![vec![0, 1], vec![2, 3]]);
        assert_eq!(level.level_id, 0);
        assert_eq!(level.n_groups(), 2);
        assert!(level.is_binary_level());

        assert_eq!(level.get_group_for_class(0), Some(0));
        assert_eq!(level.get_group_for_class(1), Some(0));
        assert_eq!(level.get_group_for_class(2), Some(1));
        assert_eq!(level.get_group_for_class(3), Some(1));
    }

    #[test]
    fn test_decomposition_level_validation() {
        let level = DecompositionLevel::new(0, vec![vec![0, 1], vec![2, 3]]);
        assert!(level.validate(4).is_ok());

        // Test with invalid class
        let invalid_level = DecompositionLevel::new(0, vec![vec![0, 1], vec![5, 6]]);
        assert!(invalid_level.validate(4).is_err());

        // Test with empty group
        let empty_group_level = DecompositionLevel::new(0, vec![vec![], vec![0, 1]]);
        assert!(empty_group_level.validate(4).is_err());
    }

    #[test]
    fn test_multi_level_classifier_creation() {
        let levels = create_simple_levels();
        let base_classifier = MockNativeClassifier::new();

        let classifier = MultiLevelDecompositionClassifier::new(base_classifier, levels);
        assert!(classifier.is_ok());

        let classifier = classifier.unwrap();
        assert_eq!(classifier.n_levels(), 2);
        assert_eq!(classifier.levels()[0].n_groups(), 2);
        assert_eq!(classifier.levels()[1].n_groups(), 4);
    }

    #[test]
    fn test_multi_level_builder() {
        let levels = create_simple_levels();
        let base_classifier = MockNativeClassifier::new();

        let classifier = MultiLevelDecompositionClassifier::builder(base_classifier, levels)
            .unwrap()
            .combination_strategy(LevelCombinationStrategy::WeightedAverage)
            .level_weights(vec![0.6, 0.4])
            .confidence_threshold(0.7)
            .early_stopping(true)
            .build();

        assert!(matches!(
            classifier.config.combination_strategy,
            LevelCombinationStrategy::WeightedAverage
        ));
        assert_eq!(classifier.config.level_weights, Some(vec![0.6, 0.4]));
        assert_eq!(classifier.config.confidence_threshold, 0.7);
        assert!(classifier.config.early_stopping);
    }

    #[test]
    fn test_binary_tree_decomposition() {
        let levels = MultiLevelDecompositionUtils::binary_tree_decomposition(4).unwrap();
        assert!(!levels.is_empty());

        // First level should have binary split
        assert!(levels[0].is_binary_level());

        // Test with single class (should fail)
        assert!(MultiLevelDecompositionUtils::binary_tree_decomposition(1).is_err());
    }

    #[test]
    fn test_multi_level_ovr() {
        let levels = MultiLevelDecompositionUtils::multi_level_ovr(6, 3).unwrap();
        assert_eq!(levels.len(), 3);

        for level in &levels {
            assert!(matches!(level.config.strategy, LevelStrategy::OneVsRest));
        }
    }

    #[test]
    fn test_hierarchical_decomposition() {
        let hierarchy = vec![
            (0, vec![1, 2]), // Root splits into 1 and 2
            (1, vec![3, 4]), // 1 splits into 3 and 4
            (2, vec![5, 6]), // 2 splits into 5 and 6
        ];

        let levels = MultiLevelDecompositionUtils::hierarchical_decomposition(&hierarchy).unwrap();
        assert!(!levels.is_empty());

        // Should have multiple levels based on hierarchy depth
        assert!(levels.len() >= 2);
    }

    #[test]
    fn test_invalid_multi_level_creation() {
        let base_classifier = MockNativeClassifier::new();

        // Test with empty levels
        let base_classifier2 = MockNativeClassifier::new();
        let result = MultiLevelDecompositionClassifier::new(base_classifier2, vec![]);
        assert!(result.is_err());

        // Test with level containing empty groups
        let invalid_levels = vec![DecompositionLevel::new(0, vec![])];
        let result = MultiLevelDecompositionClassifier::new(base_classifier, invalid_levels);
        assert!(result.is_err());
    }

    #[test]
    fn test_level_configuration() {
        let level = DecompositionLevel::new(0, vec![vec![0, 1], vec![2, 3]])
            .with_strategy(LevelStrategy::OneVsOne)
            .with_weight(0.8)
            .with_early_stopping(0.9);

        assert!(matches!(level.config.strategy, LevelStrategy::OneVsOne));
        assert_eq!(level.config.weight, 0.8);
        assert!(level.config.can_early_stop);
        assert_eq!(level.config.early_stop_confidence, 0.9);
    }
}