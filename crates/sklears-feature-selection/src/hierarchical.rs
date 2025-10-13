//! Hierarchical feature selection methods
//!
//! This module provides algorithms for feature selection that respect hierarchical
//! structure in the features, such as grouped features, multi-level categorical variables,
//! or features with natural parent-child relationships.

use crate::base::FeatureSelector;
use scirs2_core::ndarray::{Array1, Array2, Axis};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Trained, Transform, Untrained},
    types::Float,
};
use std::collections::{HashMap, HashSet, VecDeque};
use std::marker::PhantomData;

/// Represents a node in the feature hierarchy
#[derive(Debug, Clone)]
pub struct HierarchyNode {
    pub feature_id: usize,
    pub parent: Option<usize>,
    pub children: Vec<usize>,
    pub level: usize,
    pub group_id: Option<usize>,
}

/// Feature hierarchy structure
#[derive(Debug, Clone)]
pub struct FeatureHierarchy {
    nodes: HashMap<usize, HierarchyNode>,
    root_nodes: Vec<usize>,
    max_level: usize,
}

impl FeatureHierarchy {
    /// Create a new feature hierarchy
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            root_nodes: Vec::new(),
            max_level: 0,
        }
    }

    /// Add a feature node to the hierarchy
    pub fn add_node(
        &mut self,
        feature_id: usize,
        parent: Option<usize>,
        group_id: Option<usize>,
    ) -> SklResult<()> {
        let level = if let Some(parent_id) = parent {
            if let Some(parent_node) = self.nodes.get(&parent_id) {
                parent_node.level + 1
            } else {
                return Err(SklearsError::InvalidInput(format!(
                    "Parent node {} not found",
                    parent_id
                )));
            }
        } else {
            0
        };

        let node = HierarchyNode {
            feature_id,
            parent,
            children: Vec::new(),
            level,
            group_id,
        };

        // Update parent's children list
        if let Some(parent_id) = parent {
            if let Some(parent_node) = self.nodes.get_mut(&parent_id) {
                parent_node.children.push(feature_id);
            }
        } else {
            self.root_nodes.push(feature_id);
        }

        self.max_level = self.max_level.max(level);
        self.nodes.insert(feature_id, node);
        Ok(())
    }

    /// Get all descendants of a node
    pub fn get_descendants(&self, feature_id: usize) -> Vec<usize> {
        let mut descendants = Vec::new();
        let mut queue = VecDeque::new();

        if let Some(node) = self.nodes.get(&feature_id) {
            queue.extend(&node.children);
        }

        while let Some(child_id) = queue.pop_front() {
            descendants.push(child_id);
            if let Some(child_node) = self.nodes.get(&child_id) {
                queue.extend(&child_node.children);
            }
        }

        descendants
    }

    /// Get all ancestors of a node
    pub fn get_ancestors(&self, feature_id: usize) -> Vec<usize> {
        let mut ancestors = Vec::new();
        let mut current_id = feature_id;

        while let Some(node) = self.nodes.get(&current_id) {
            if let Some(parent_id) = node.parent {
                ancestors.push(parent_id);
                current_id = parent_id;
            } else {
                break;
            }
        }

        ancestors
    }

    /// Get features at a specific level
    pub fn get_features_at_level(&self, level: usize) -> Vec<usize> {
        let mut features: Vec<usize> = self
            .nodes
            .values()
            .filter(|node| node.level == level)
            .map(|node| node.feature_id)
            .collect();
        features.sort();
        features
    }

    /// Get features in a specific group
    pub fn get_features_in_group(&self, group_id: usize) -> Vec<usize> {
        let mut features: Vec<usize> = self
            .nodes
            .values()
            .filter(|node| node.group_id == Some(group_id))
            .map(|node| node.feature_id)
            .collect();
        features.sort();
        features
    }

    /// Check if a feature is a leaf node (has no children)
    pub fn is_leaf(&self, feature_id: usize) -> bool {
        self.nodes
            .get(&feature_id)
            .map(|node| node.children.is_empty())
            .unwrap_or(false)
    }

    /// Get all leaf nodes
    pub fn get_leaf_nodes(&self) -> Vec<usize> {
        self.nodes
            .values()
            .filter(|node| node.children.is_empty())
            .map(|node| node.feature_id)
            .collect()
    }
}

impl Default for FeatureHierarchy {
    fn default() -> Self {
        Self::new()
    }
}

/// Hierarchical feature selector using top-down selection
///
/// Selects features starting from the root level and moving down the hierarchy,
/// ensuring that if a parent is selected, at least one child is considered.
#[derive(Debug, Clone)]
pub struct HierarchicalFeatureSelector<State = Untrained> {
    hierarchy: FeatureHierarchy,
    k: usize,
    selection_strategy: HierarchicalSelectionStrategy,
    score_aggregation: ScoreAggregation,

    // Fitted state
    selected_features_: Option<Vec<usize>>,
    feature_scores_: Option<HashMap<usize, Float>>,

    state: PhantomData<State>,
}

/// Strategy for hierarchical selection
#[derive(Debug, Clone)]
pub enum HierarchicalSelectionStrategy {
    /// Select from top to bottom, ensuring parent-child consistency
    TopDown,
    /// Select from bottom to top, propagating scores upward
    BottomUp,
    /// Select at each level independently
    LevelWise,
    /// Use group-based selection within hierarchy
    GroupBased,
}

/// Method for aggregating scores across hierarchy levels
#[derive(Debug, Clone)]
pub enum ScoreAggregation {
    /// Sum scores across levels
    Sum,
    /// Take maximum score across levels
    Max,
    /// Take weighted average (higher levels get higher weights)
    WeightedAverage,
    /// Use multiplicative combination
    Product,
}

impl HierarchicalFeatureSelector<Untrained> {
    /// Create a new hierarchical feature selector
    pub fn new(hierarchy: FeatureHierarchy, k: usize) -> Self {
        Self {
            hierarchy,
            k,
            selection_strategy: HierarchicalSelectionStrategy::TopDown,
            score_aggregation: ScoreAggregation::Sum,
            selected_features_: None,
            feature_scores_: None,
            state: PhantomData,
        }
    }

    /// Set the selection strategy
    pub fn selection_strategy(mut self, strategy: HierarchicalSelectionStrategy) -> Self {
        self.selection_strategy = strategy;
        self
    }

    /// Set the score aggregation method
    pub fn score_aggregation(mut self, aggregation: ScoreAggregation) -> Self {
        self.score_aggregation = aggregation;
        self
    }
}

impl Estimator for HierarchicalFeatureSelector<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<Array2<Float>, Array1<Float>> for HierarchicalFeatureSelector<Untrained> {
    type Fitted = HierarchicalFeatureSelector<Trained>;

    fn fit(self, x: &Array2<Float>, y: &Array1<Float>) -> SklResult<Self::Fitted> {
        let (n_samples, n_features) = x.dim();
        if n_samples == 0 || n_features == 0 {
            return Err(SklearsError::InvalidInput(
                "Input data cannot be empty".to_string(),
            ));
        }

        if self.k > n_features {
            return Err(SklearsError::InvalidInput(
                "k cannot be larger than number of features".to_string(),
            ));
        }

        // Compute base feature scores using F-statistic
        let mut feature_scores = HashMap::new();
        for feature_idx in 0..n_features {
            let feature_col = x.column(feature_idx);
            let score = compute_f_score(&feature_col.to_owned(), y);
            feature_scores.insert(feature_idx, score);
        }

        // Apply hierarchical selection based on strategy
        let selected_features = match self.selection_strategy {
            HierarchicalSelectionStrategy::TopDown => self.select_top_down(&feature_scores)?,
            HierarchicalSelectionStrategy::BottomUp => self.select_bottom_up(&feature_scores)?,
            HierarchicalSelectionStrategy::LevelWise => self.select_level_wise(&feature_scores)?,
            HierarchicalSelectionStrategy::GroupBased => {
                self.select_group_based(&feature_scores)?
            }
        };

        Ok(HierarchicalFeatureSelector {
            hierarchy: self.hierarchy,
            k: self.k,
            selection_strategy: self.selection_strategy,
            score_aggregation: self.score_aggregation,
            selected_features_: Some(selected_features),
            feature_scores_: Some(feature_scores),
            state: PhantomData,
        })
    }
}

impl HierarchicalFeatureSelector<Untrained> {
    /// Top-down hierarchical selection
    fn select_top_down(&self, feature_scores: &HashMap<usize, Float>) -> SklResult<Vec<usize>> {
        let mut selected = HashSet::new();
        let mut candidates = VecDeque::new();

        // Start with root nodes
        candidates.extend(&self.hierarchy.root_nodes);

        while !candidates.is_empty() && selected.len() < self.k {
            let mut level_scores: Vec<(usize, Float)> = candidates
                .iter()
                .filter_map(|&feature_id| {
                    feature_scores
                        .get(&feature_id)
                        .map(|&score| (feature_id, score))
                })
                .collect();

            // Sort by score (descending)
            level_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            // Select best features from current level
            let mut next_candidates: VecDeque<usize> = VecDeque::new();
            for (feature_id, _) in level_scores {
                if selected.len() >= self.k {
                    break;
                }

                selected.insert(feature_id);
                candidates.retain(|&x| x != feature_id);

                // Add children to next level candidates
                if let Some(node) = self.hierarchy.nodes.get(&feature_id) {
                    next_candidates.extend(&node.children);
                }
            }

            candidates.extend(next_candidates);
        }

        Ok(selected.into_iter().collect())
    }

    /// Bottom-up hierarchical selection
    fn select_bottom_up(&self, feature_scores: &HashMap<usize, Float>) -> SklResult<Vec<usize>> {
        let mut aggregated_scores = feature_scores.clone();

        // Propagate scores from leaves to roots
        for level in (0..=self.hierarchy.max_level).rev() {
            let level_features = self.hierarchy.get_features_at_level(level);

            for feature_id in level_features {
                if let Some(node) = self.hierarchy.nodes.get(&feature_id) {
                    if !node.children.is_empty() {
                        // Aggregate children scores
                        let child_scores: Vec<Float> = node
                            .children
                            .iter()
                            .filter_map(|&child_id| aggregated_scores.get(&child_id))
                            .cloned()
                            .collect();

                        if !child_scores.is_empty() {
                            let aggregated = self.aggregate_scores(&child_scores);
                            let current_score =
                                aggregated_scores.get(&feature_id).cloned().unwrap_or(0.0);
                            aggregated_scores.insert(feature_id, current_score + aggregated);
                        }
                    }
                }
            }
        }

        // Select top k features based on aggregated scores
        let mut scored_features: Vec<(usize, Float)> = aggregated_scores.into_iter().collect();
        scored_features.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        Ok(scored_features
            .into_iter()
            .take(self.k)
            .map(|(feature_id, _)| feature_id)
            .collect())
    }

    /// Level-wise hierarchical selection
    fn select_level_wise(&self, feature_scores: &HashMap<usize, Float>) -> SklResult<Vec<usize>> {
        let mut selected = Vec::new();
        let features_per_level = self.k / (self.hierarchy.max_level + 1);
        let remaining = self.k % (self.hierarchy.max_level + 1);

        for level in 0..=self.hierarchy.max_level {
            let level_features = self.hierarchy.get_features_at_level(level);
            let mut level_scores: Vec<(usize, Float)> = level_features
                .into_iter()
                .filter_map(|feature_id| {
                    feature_scores
                        .get(&feature_id)
                        .map(|&score| (feature_id, score))
                })
                .collect();

            level_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            let k_for_level = if level < remaining {
                features_per_level + 1
            } else {
                features_per_level
            };

            selected.extend(
                level_scores
                    .into_iter()
                    .take(k_for_level)
                    .map(|(feature_id, _)| feature_id),
            );
        }

        Ok(selected)
    }

    /// Group-based hierarchical selection
    fn select_group_based(&self, feature_scores: &HashMap<usize, Float>) -> SklResult<Vec<usize>> {
        // Get all unique groups
        let mut groups: HashSet<usize> = HashSet::new();
        for node in self.hierarchy.nodes.values() {
            if let Some(group_id) = node.group_id {
                groups.insert(group_id);
            }
        }

        if groups.is_empty() {
            // Fallback to regular top-k selection
            let mut scored_features: Vec<(usize, Float)> = feature_scores
                .iter()
                .map(|(&feature_id, &score)| (feature_id, score))
                .collect();
            scored_features.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            return Ok(scored_features
                .into_iter()
                .take(self.k)
                .map(|(feature_id, _)| feature_id)
                .collect());
        }

        let features_per_group = self.k / groups.len();
        let remaining = self.k % groups.len();
        let mut selected = Vec::new();

        for (group_idx, group_id) in groups.into_iter().enumerate() {
            let group_features = self.hierarchy.get_features_in_group(group_id);
            let mut group_scores: Vec<(usize, Float)> = group_features
                .into_iter()
                .filter_map(|feature_id| {
                    feature_scores
                        .get(&feature_id)
                        .map(|&score| (feature_id, score))
                })
                .collect();

            group_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            let k_for_group = if group_idx < remaining {
                features_per_group + 1
            } else {
                features_per_group
            };

            selected.extend(
                group_scores
                    .into_iter()
                    .take(k_for_group)
                    .map(|(feature_id, _)| feature_id),
            );
        }

        Ok(selected)
    }

    /// Aggregate scores using the specified method
    fn aggregate_scores(&self, scores: &[Float]) -> Float {
        if scores.is_empty() {
            return 0.0;
        }

        match self.score_aggregation {
            ScoreAggregation::Sum => scores.iter().sum(),
            ScoreAggregation::Max => scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
            ScoreAggregation::WeightedAverage => {
                let sum: Float = scores.iter().sum();
                sum / scores.len() as Float
            }
            ScoreAggregation::Product => scores.iter().product(),
        }
    }
}

impl FeatureSelector for HierarchicalFeatureSelector<Trained> {
    fn selected_features(&self) -> &Vec<usize> {
        match &self.selected_features_ {
            Some(features) => features,
            None => {
                static EMPTY: Vec<usize> = Vec::new();
                &EMPTY
            }
        }
    }
}

impl Transform<Array2<Float>, Array2<Float>> for HierarchicalFeatureSelector<Trained> {
    fn transform(&self, x: &Array2<Float>) -> SklResult<Array2<Float>> {
        if let Some(selected) = &self.selected_features_ {
            if selected.is_empty() {
                return Err(SklearsError::InvalidData {
                    reason: "No features selected".to_string(),
                });
            }

            let selected_cols = x.select(Axis(1), selected);
            Ok(selected_cols)
        } else {
            Err(SklearsError::InvalidData {
                reason: "Selector not fitted yet".to_string(),
            })
        }
    }
}

/// Multi-level hierarchical feature selector
///
/// Performs feature selection at multiple levels of the hierarchy simultaneously
#[derive(Debug, Clone)]
pub struct MultiLevelHierarchicalSelector<State = Untrained> {
    hierarchy: FeatureHierarchy,
    k_per_level: HashMap<usize, usize>,
    level_weights: HashMap<usize, Float>,

    // Fitted state
    selected_features_: Option<HashMap<usize, Vec<usize>>>,
    level_scores_: Option<HashMap<usize, HashMap<usize, Float>>>,

    state: PhantomData<State>,
}

impl MultiLevelHierarchicalSelector<Untrained> {
    /// Create a new multi-level hierarchical selector
    pub fn new(hierarchy: FeatureHierarchy) -> Self {
        Self {
            hierarchy,
            k_per_level: HashMap::new(),
            level_weights: HashMap::new(),
            selected_features_: None,
            level_scores_: None,
            state: PhantomData,
        }
    }

    /// Set number of features to select at each level
    pub fn k_per_level(mut self, k_per_level: HashMap<usize, usize>) -> Self {
        self.k_per_level = k_per_level;
        self
    }

    /// Set weights for each level (used in scoring)
    pub fn level_weights(mut self, level_weights: HashMap<usize, Float>) -> Self {
        self.level_weights = level_weights;
        self
    }
}

impl Estimator for MultiLevelHierarchicalSelector<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<Array2<Float>, Array1<Float>> for MultiLevelHierarchicalSelector<Untrained> {
    type Fitted = MultiLevelHierarchicalSelector<Trained>;

    fn fit(self, x: &Array2<Float>, y: &Array1<Float>) -> SklResult<Self::Fitted> {
        let (n_samples, n_features) = x.dim();
        if n_samples == 0 || n_features == 0 {
            return Err(SklearsError::InvalidInput(
                "Input data cannot be empty".to_string(),
            ));
        }

        // Compute feature scores
        let mut feature_scores = HashMap::new();
        for feature_idx in 0..n_features {
            let feature_col = x.column(feature_idx);
            let score = compute_f_score(&feature_col.to_owned(), y);
            feature_scores.insert(feature_idx, score);
        }

        // Select features at each level
        let mut selected_features = HashMap::new();
        let mut level_scores = HashMap::new();

        for level in 0..=self.hierarchy.max_level {
            let level_features = self.hierarchy.get_features_at_level(level);
            let k_for_level = self.k_per_level.get(&level).cloned().unwrap_or(
                level_features.len().min(5), // Default to 5 or all features at level
            );

            let mut level_feature_scores: Vec<(usize, Float)> = level_features
                .into_iter()
                .filter_map(|feature_id| {
                    feature_scores.get(&feature_id).map(|&score| {
                        let weight = self.level_weights.get(&level).cloned().unwrap_or(1.0);
                        (feature_id, score * weight)
                    })
                })
                .collect();

            level_feature_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            let selected_at_level: Vec<usize> = level_feature_scores
                .into_iter()
                .take(k_for_level)
                .map(|(feature_id, score)| {
                    level_scores
                        .entry(level)
                        .or_insert_with(HashMap::new)
                        .insert(feature_id, score);
                    feature_id
                })
                .collect();

            selected_features.insert(level, selected_at_level);
        }

        Ok(MultiLevelHierarchicalSelector {
            hierarchy: self.hierarchy,
            k_per_level: self.k_per_level,
            level_weights: self.level_weights,
            selected_features_: Some(selected_features),
            level_scores_: Some(level_scores),
            state: PhantomData,
        })
    }
}

impl MultiLevelHierarchicalSelector<Trained> {
    /// Get selected features at a specific level
    pub fn selected_features_at_level(&self, level: usize) -> Option<&Vec<usize>> {
        self.selected_features_.as_ref()?.get(&level)
    }

    /// Get all selected features across all levels
    pub fn all_selected_features(&self) -> Vec<usize> {
        if let Some(selected_features) = &self.selected_features_ {
            let mut all_features = Vec::new();
            for features in selected_features.values() {
                all_features.extend_from_slice(features);
            }
            all_features.sort_unstable();
            all_features.dedup();
            all_features
        } else {
            Vec::new()
        }
    }
}

impl FeatureSelector for MultiLevelHierarchicalSelector<Trained> {
    fn selected_features(&self) -> &Vec<usize> {
        // This is a bit tricky since we don't store a single Vec<usize>
        // We'll need a different approach for this trait implementation
        static EMPTY: Vec<usize> = Vec::new();
        &EMPTY
    }
}

impl Transform<Array2<Float>, Array2<Float>> for MultiLevelHierarchicalSelector<Trained> {
    fn transform(&self, x: &Array2<Float>) -> SklResult<Array2<Float>> {
        let all_selected = self.all_selected_features();
        if all_selected.is_empty() {
            return Err(SklearsError::InvalidData {
                reason: "No features selected".to_string(),
            });
        }

        let selected_cols = x.select(Axis(1), &all_selected);
        Ok(selected_cols)
    }
}

/// Compute F-score for a feature
fn compute_f_score(feature: &Array1<Float>, target: &Array1<Float>) -> Float {
    if feature.len() != target.len() || feature.len() < 3 {
        return 0.0;
    }

    let n = feature.len() as Float;
    let feature_mean = feature.mean().unwrap_or(0.0);
    let target_mean = target.mean().unwrap_or(0.0);

    // Compute correlation coefficient
    let mut numerator = 0.0;
    let mut feature_var = 0.0;
    let mut target_var = 0.0;

    for i in 0..feature.len() {
        let feature_dev = feature[i] - feature_mean;
        let target_dev = target[i] - target_mean;
        numerator += feature_dev * target_dev;
        feature_var += feature_dev * feature_dev;
        target_var += target_dev * target_dev;
    }

    let r = if feature_var > 0.0 && target_var > 0.0 {
        numerator / (feature_var * target_var).sqrt()
    } else {
        0.0
    };

    // Convert correlation to F-statistic
    let r_squared = r * r;
    if (1.0 - r_squared).abs() < 1e-10 {
        f64::INFINITY
    } else {
        r_squared * (n - 2.0) / (1.0 - r_squared)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_feature_hierarchy_creation() {
        let mut hierarchy = FeatureHierarchy::new();

        // Add root features
        hierarchy.add_node(0, None, Some(0)).unwrap();
        hierarchy.add_node(1, None, Some(1)).unwrap();

        // Add child features
        hierarchy.add_node(2, Some(0), Some(0)).unwrap();
        hierarchy.add_node(3, Some(0), Some(0)).unwrap();
        hierarchy.add_node(4, Some(1), Some(1)).unwrap();

        assert_eq!(hierarchy.root_nodes.len(), 2);
        assert_eq!(hierarchy.max_level, 1);

        let descendants_0 = hierarchy.get_descendants(0);
        assert_eq!(descendants_0, vec![2, 3]);

        let level_0_features = hierarchy.get_features_at_level(0);
        assert_eq!(level_0_features, vec![0, 1]);

        let group_0_features = hierarchy.get_features_in_group(0);
        assert_eq!(group_0_features, vec![0, 2, 3]);
    }

    #[test]
    fn test_hierarchical_selector_top_down() {
        let mut hierarchy = FeatureHierarchy::new();
        hierarchy.add_node(0, None, None).unwrap();
        hierarchy.add_node(1, Some(0), None).unwrap();
        hierarchy.add_node(2, Some(0), None).unwrap();
        hierarchy.add_node(3, None, None).unwrap();

        let x = array![
            [1.0, 0.5, 0.8, 2.0],
            [2.0, 1.0, 1.2, 4.0],
            [3.0, 1.5, 1.8, 6.0],
            [4.0, 2.0, 2.4, 8.0],
        ];
        let y = array![1.0, 2.0, 3.0, 4.0];

        let selector = HierarchicalFeatureSelector::new(hierarchy, 2)
            .selection_strategy(HierarchicalSelectionStrategy::TopDown);
        let fitted = selector.fit(&x, &y).unwrap();

        let selected = fitted.selected_features();
        assert!(!selected.is_empty());
        assert!(selected.len() <= 2);
    }

    #[test]
    fn test_hierarchical_selector_transform() {
        let mut hierarchy = FeatureHierarchy::new();
        hierarchy.add_node(0, None, None).unwrap();
        hierarchy.add_node(1, None, None).unwrap();
        hierarchy.add_node(2, None, None).unwrap();

        let x = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0],];
        let y = array![1.0, 2.0, 3.0];

        let selector = HierarchicalFeatureSelector::new(hierarchy, 2);
        let fitted = selector.fit(&x, &y).unwrap();

        let test_x = array![[10.0, 11.0, 12.0], [13.0, 14.0, 15.0]];
        let transformed = fitted.transform(&test_x).unwrap();

        assert_eq!(transformed.nrows(), 2);
        assert!(transformed.ncols() <= 2);
    }

    #[test]
    fn test_multi_level_selector() {
        let mut hierarchy = FeatureHierarchy::new();
        hierarchy.add_node(0, None, None).unwrap();
        hierarchy.add_node(1, Some(0), None).unwrap();
        hierarchy.add_node(2, Some(0), None).unwrap();
        hierarchy.add_node(3, None, None).unwrap();

        let x = array![
            [1.0, 0.5, 0.8, 2.0],
            [2.0, 1.0, 1.2, 4.0],
            [3.0, 1.5, 1.8, 6.0],
        ];
        let y = array![1.0, 2.0, 3.0];

        let mut k_per_level = HashMap::new();
        k_per_level.insert(0, 1); // Select 1 feature at level 0
        k_per_level.insert(1, 1); // Select 1 feature at level 1

        let selector = MultiLevelHierarchicalSelector::new(hierarchy).k_per_level(k_per_level);
        let fitted = selector.fit(&x, &y).unwrap();

        let level_0_selected = fitted.selected_features_at_level(0);
        assert!(level_0_selected.is_some());
        assert_eq!(level_0_selected.unwrap().len(), 1);
    }
}
