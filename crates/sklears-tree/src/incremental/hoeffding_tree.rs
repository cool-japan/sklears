//! Hoeffding Tree (Very Fast Decision Tree) for streaming data
//!
//! The Hoeffding Tree uses the Hoeffding bound to determine when enough data has been seen
//! to make confident split decisions without waiting for all possible data. This makes it
//! ideal for streaming scenarios where data arrives continuously and memory is limited.
//!
//! ## Key Features
//!
//! - **Statistical Confidence**: Uses Hoeffding bound for statistically sound split decisions
//! - **Memory Efficient**: Maintains only sufficient statistics rather than storing all data
//! - **Adaptive**: Automatically determines when to split based on statistical significance
//! - **Robust**: Handles both classification and regression tasks
//! - **Online Learning**: Updates incrementally with each new sample

use super::core_tree_structures::StreamingTreeModel;
use crate::SplitCriterion;
use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::error::{Result, SklearsError};
use std::collections::HashMap;

/// Hoeffding Tree (Very Fast Decision Tree) for streaming data
///
/// The Hoeffding Tree uses the Hoeffding bound to determine when enough
/// data has been seen to make confident split decisions without waiting
/// for all possible data.
#[derive(Debug, Clone)]
pub struct HoeffdingTree {
    /// Tree nodes with sufficient statistics
    nodes: Vec<HoeffdingNode>,
    /// Configuration
    config: HoeffdingTreeConfig,
    /// Root node ID
    root_id: usize,
    /// Next available node ID
    next_node_id: usize,
    /// Feature importance scores
    feature_importances: HashMap<usize, f64>,
}

/// Configuration for Hoeffding Tree
#[derive(Debug, Clone)]
pub struct HoeffdingTreeConfig {
    /// Confidence level for Hoeffding bound (typically 0.95 to 0.99)
    pub confidence: f64,
    /// Tie threshold - difference below this is considered a tie
    pub tie_threshold: f64,
    /// Minimum number of samples before considering a split
    pub min_samples_split: usize,
    /// Maximum tree depth
    pub max_depth: Option<usize>,
    /// Split criterion
    pub split_criterion: SplitCriterion,
    /// Grace period - minimum samples before first split attempt
    pub grace_period: usize,
    /// Enable pre-pruning based on statistical tests
    pub enable_prepruning: bool,
    /// Memory limit (approximate number of attribute-value pairs to store)
    pub memory_limit: Option<usize>,
}

impl Default for HoeffdingTreeConfig {
    fn default() -> Self {
        Self {
            confidence: 0.95,
            tie_threshold: 0.05,
            min_samples_split: 30,
            max_depth: None,
            split_criterion: SplitCriterion::Gini,
            grace_period: 200,
            enable_prepruning: true,
            memory_limit: Some(100000),
        }
    }
}

/// Hoeffding tree node with sufficient statistics
#[derive(Debug, Clone)]
pub struct HoeffdingNode {
    /// Node ID
    id: usize,
    /// Node depth
    depth: usize,
    /// Is this a leaf node?
    is_leaf: bool,
    /// Split feature index (for internal nodes)
    split_feature: Option<usize>,
    /// Split threshold (for internal nodes)
    split_threshold: Option<f64>,
    /// Left child ID
    left_child: Option<usize>,
    /// Right child ID
    right_child: Option<usize>,
    /// Sufficient statistics for each feature
    feature_stats: HashMap<usize, FeatureSufficientStats>,
    /// Class counts (for classification)
    class_counts: HashMap<i32, usize>,
    /// Total sample count
    sample_count: usize,
    /// Sum of target values (for regression)
    sum_y: f64,
    /// Sum of squared target values (for regression)
    sum_y_squared: f64,
    /// Last split evaluation sample count
    last_split_evaluation: usize,
}

/// Sufficient statistics for a feature in Hoeffding tree
#[derive(Debug, Clone)]
pub struct FeatureSufficientStats {
    /// For numerical features: histogram bins
    numeric_bins: Option<HashMap<usize, BinStats>>,
    /// For categorical features: category counts
    categorical_counts: Option<HashMap<String, ClassCounts>>,
    /// Feature type
    feature_type: FeatureType,
}

/// Statistics for a histogram bin
#[derive(Debug, Clone)]
pub struct BinStats {
    /// Bin boundaries (min, max)
    boundaries: (f64, f64),
    /// Class counts in this bin
    class_counts: HashMap<i32, usize>,
    /// Sum of target values (for regression)
    sum_y: f64,
    /// Sample count in bin
    count: usize,
}

/// Class counts for categorical values
#[derive(Debug, Clone)]
pub struct ClassCounts {
    /// Class distribution
    counts: HashMap<i32, usize>,
    /// Total count
    total: usize,
}

/// Feature type for Hoeffding tree
#[derive(Debug, Clone, Copy)]
pub enum FeatureType {
    Numeric,
    Categorical,
}

impl HoeffdingNode {
    /// Create a new leaf node
    pub fn new_leaf(id: usize, depth: usize, n_features: usize) -> Self {
        let mut feature_stats = HashMap::new();

        // Initialize feature statistics
        for i in 0..n_features {
            feature_stats.insert(
                i,
                FeatureSufficientStats {
                    numeric_bins: Some(HashMap::new()),
                    categorical_counts: None,
                    feature_type: FeatureType::Numeric, // Default to numeric
                },
            );
        }

        Self {
            id,
            depth,
            is_leaf: true,
            split_feature: None,
            split_threshold: None,
            left_child: None,
            right_child: None,
            feature_stats,
            class_counts: HashMap::new(),
            sample_count: 0,
            sum_y: 0.0,
            sum_y_squared: 0.0,
            last_split_evaluation: 0,
        }
    }

    /// Update node with a new sample
    pub fn update(&mut self, x: &[f64], y: f64, class_label: Option<i32>) {
        self.sample_count += 1;
        self.sum_y += y;
        self.sum_y_squared += y * y;

        // Update class counts if classification
        if let Some(class) = class_label {
            *self.class_counts.entry(class).or_insert(0) += 1;
        }

        // Update feature statistics
        for (feature_idx, &feature_value) in x.iter().enumerate() {
            self.update_feature_stats_at_index(feature_idx, feature_value, y, class_label);
        }
    }

    /// Update sufficient statistics for a feature at a specific index
    fn update_feature_stats_at_index(
        &mut self,
        feature_idx: usize,
        value: f64,
        y: f64,
        class_label: Option<i32>,
    ) {
        if let Some(stats) = self.feature_stats.get_mut(&feature_idx) {
            match stats.feature_type {
                FeatureType::Numeric => {
                    if let Some(ref mut bins) = stats.numeric_bins {
                        // Simple binning strategy - create bins dynamically
                        let bin_id = Self::get_bin_id_static(value);

                        let bin = bins.entry(bin_id).or_insert_with(|| BinStats {
                            boundaries: (value - 0.5, value + 0.5),
                            class_counts: HashMap::new(),
                            sum_y: 0.0,
                            count: 0,
                        });

                        bin.sum_y += y;
                        bin.count += 1;

                        if let Some(class) = class_label {
                            *bin.class_counts.entry(class).or_insert(0) += 1;
                        }
                    }
                }
                FeatureType::Categorical => {
                    // Handle categorical features
                    if let Some(ref mut cat_counts) = stats.categorical_counts {
                        let value_str = value.to_string();
                        let counts = cat_counts.entry(value_str).or_insert_with(|| ClassCounts {
                            counts: HashMap::new(),
                            total: 0,
                        });

                        counts.total += 1;
                        if let Some(class) = class_label {
                            *counts.counts.entry(class).or_insert(0) += 1;
                        }
                    }
                }
            }
        }
    }

    /// Simple binning strategy (static version)
    fn get_bin_id_static(value: f64) -> usize {
        // Simple equal-width binning
        let bin_width = 1.0;
        ((value / bin_width).floor() as i32).max(0) as usize
    }

    /// Calculate Hoeffding bound for split evaluation
    pub fn calculate_hoeffding_bound(&self, confidence: f64) -> f64 {
        if self.sample_count == 0 {
            return f64::INFINITY;
        }

        // Hoeffding bound: sqrt(R^2 * ln(1/δ) / (2n))
        // where R is the range of the random variable (e.g., 1 for information gain)
        // δ = 1 - confidence
        let delta = 1.0 - confidence;
        let r_squared = 1.0; // For information gain, range is [0, 1]

        (r_squared * delta.ln().abs() / (2.0 * self.sample_count as f64)).sqrt()
    }

    /// Evaluate potential splits and determine if we should split
    pub fn should_split(&mut self, config: &HoeffdingTreeConfig) -> Option<(usize, f64)> {
        if self.sample_count < config.min_samples_split
            || self.sample_count - self.last_split_evaluation < config.grace_period
        {
            return None;
        }

        if let Some(max_depth) = config.max_depth {
            if self.depth >= max_depth {
                return None;
            }
        }

        self.last_split_evaluation = self.sample_count;

        let hoeffding_bound = self.calculate_hoeffding_bound(config.confidence);
        let mut best_split: Option<(usize, f64, f64)> = None; // (feature, threshold, gain)
        let mut second_best_gain = 0.0;

        // Evaluate all possible splits
        for (&feature_idx, feature_stats) in &self.feature_stats {
            if let Some(splits) = self.get_candidate_splits(feature_stats) {
                for (threshold, gain) in splits {
                    if best_split.is_none() || gain > best_split.as_ref().unwrap().2 {
                        if let Some((_, _, prev_best_gain)) = best_split {
                            second_best_gain = prev_best_gain;
                        }
                        best_split = Some((feature_idx, threshold, gain));
                    } else if gain > second_best_gain {
                        second_best_gain = gain;
                    }
                }
            }
        }

        if let Some((feature_idx, threshold, best_gain)) = best_split {
            let gain_difference = best_gain - second_best_gain;

            // Check Hoeffding bound condition
            if gain_difference > hoeffding_bound || hoeffding_bound < config.tie_threshold {
                return Some((feature_idx, threshold));
            }
        }

        None
    }

    /// Get candidate splits for a feature
    fn get_candidate_splits(
        &self,
        feature_stats: &FeatureSufficientStats,
    ) -> Option<Vec<(f64, f64)>> {
        match feature_stats.feature_type {
            FeatureType::Numeric => {
                if let Some(ref bins) = feature_stats.numeric_bins {
                    let mut splits = Vec::new();

                    // Create splits between bins
                    let mut sorted_bins: Vec<_> = bins.iter().collect();
                    sorted_bins.sort_by_key(|(bin_id, _)| *bin_id);

                    for i in 0..sorted_bins.len().saturating_sub(1) {
                        let (_, bin1) = sorted_bins[i];
                        let (_, bin2) = sorted_bins[i + 1];

                        let threshold = (bin1.boundaries.1 + bin2.boundaries.0) / 2.0;
                        let gain = self.calculate_information_gain(bins, threshold);
                        splits.push((threshold, gain));
                    }

                    Some(splits)
                } else {
                    None
                }
            }
            FeatureType::Categorical => {
                // For categorical features, each unique value becomes a potential split
                None // Simplified for now
            }
        }
    }

    /// Calculate information gain for a numeric split
    fn calculate_information_gain(&self, bins: &HashMap<usize, BinStats>, threshold: f64) -> f64 {
        let total_samples = self.sample_count as f64;
        if total_samples == 0.0 {
            return 0.0;
        }

        // Calculate weighted impurity after split
        let mut left_samples = 0.0;
        let mut right_samples = 0.0;
        let mut left_class_counts: HashMap<i32, usize> = HashMap::new();
        let mut right_class_counts: HashMap<i32, usize> = HashMap::new();

        for (_, bin) in bins {
            let bin_center = (bin.boundaries.0 + bin.boundaries.1) / 2.0;

            if bin_center <= threshold {
                left_samples += bin.count as f64;
                for (&class, &count) in &bin.class_counts {
                    *left_class_counts.entry(class).or_insert(0) += count;
                }
            } else {
                right_samples += bin.count as f64;
                for (&class, &count) in &bin.class_counts {
                    *right_class_counts.entry(class).or_insert(0) += count;
                }
            }
        }

        if left_samples == 0.0 || right_samples == 0.0 {
            return 0.0;
        }

        // Calculate impurities
        let left_impurity = self.calculate_gini_impurity(&left_class_counts, left_samples);
        let right_impurity = self.calculate_gini_impurity(&right_class_counts, right_samples);
        let current_impurity = self.calculate_gini_impurity(&self.class_counts, total_samples);

        // Information gain
        current_impurity
            - (left_samples / total_samples) * left_impurity
            - (right_samples / total_samples) * right_impurity
    }

    /// Calculate Gini impurity
    fn calculate_gini_impurity(
        &self,
        class_counts: &HashMap<i32, usize>,
        total_samples: f64,
    ) -> f64 {
        if total_samples == 0.0 {
            return 0.0;
        }

        let mut gini = 1.0;
        for &count in class_counts.values() {
            let probability = count as f64 / total_samples;
            gini -= probability * probability;
        }
        gini
    }

    /// Get the majority class prediction
    pub fn get_prediction(&self) -> f64 {
        if self.class_counts.is_empty() {
            // Regression case
            if self.sample_count > 0 {
                self.sum_y / self.sample_count as f64
            } else {
                0.0
            }
        } else {
            // Classification case - return majority class
            self.class_counts
                .iter()
                .max_by_key(|(_, &count)| count)
                .map(|(&class, _)| class as f64)
                .unwrap_or(0.0)
        }
    }
}

impl HoeffdingTree {
    /// Create a new Hoeffding Tree
    pub fn new(config: HoeffdingTreeConfig, n_features: usize) -> Self {
        let root_node = HoeffdingNode::new_leaf(0, 0, n_features);

        Self {
            nodes: vec![root_node],
            config,
            root_id: 0,
            next_node_id: 1,
            feature_importances: HashMap::new(),
        }
    }

    /// Update tree with a new sample
    pub fn update(&mut self, x: &[f64], y: f64, class_label: Option<i32>) -> Result<()> {
        // Find the leaf node for this sample
        let leaf_id = self.find_leaf(x, self.root_id)?;

        // Update the leaf node
        if let Some(leaf) = self.nodes.iter_mut().find(|n| n.id == leaf_id) {
            leaf.update(x, y, class_label);

            // Check if we should split this leaf
            if let Some((split_feature, split_threshold)) = leaf.should_split(&self.config) {
                self.split_leaf(leaf_id, split_feature, split_threshold, x.len())?;
            }
        }

        Ok(())
    }

    /// Find the leaf node for a given sample
    fn find_leaf(&self, x: &[f64], node_id: usize) -> Result<usize> {
        if let Some(node) = self.nodes.iter().find(|n| n.id == node_id) {
            if node.is_leaf {
                Ok(node_id)
            } else if let (Some(feature_idx), Some(threshold)) =
                (node.split_feature, node.split_threshold)
            {
                if feature_idx < x.len() {
                    let next_node = if x[feature_idx] <= threshold {
                        node.left_child
                    } else {
                        node.right_child
                    };

                    if let Some(next_id) = next_node {
                        self.find_leaf(x, next_id)
                    } else {
                        Ok(node_id)
                    }
                } else {
                    Err(SklearsError::InvalidInput(
                        "Feature index out of bounds".to_string(),
                    ))
                }
            } else {
                Ok(node_id)
            }
        } else {
            Err(SklearsError::InvalidInput("Node not found".to_string()))
        }
    }

    /// Split a leaf node
    fn split_leaf(
        &mut self,
        leaf_id: usize,
        split_feature: usize,
        split_threshold: f64,
        n_features: usize,
    ) -> Result<()> {
        // Create left and right child nodes
        let left_id = self.next_node_id;
        let right_id = self.next_node_id + 1;
        self.next_node_id += 2;

        if let Some(leaf_idx) = self.nodes.iter().position(|n| n.id == leaf_id) {
            let leaf_depth = self.nodes[leaf_idx].depth;

            let left_child = HoeffdingNode::new_leaf(left_id, leaf_depth + 1, n_features);
            let right_child = HoeffdingNode::new_leaf(right_id, leaf_depth + 1, n_features);

            // Convert leaf to internal node
            self.nodes[leaf_idx].is_leaf = false;
            self.nodes[leaf_idx].split_feature = Some(split_feature);
            self.nodes[leaf_idx].split_threshold = Some(split_threshold);
            self.nodes[leaf_idx].left_child = Some(left_id);
            self.nodes[leaf_idx].right_child = Some(right_id);

            // Add child nodes
            self.nodes.push(left_child);
            self.nodes.push(right_child);

            // Update feature importance
            *self.feature_importances.entry(split_feature).or_insert(0.0) += 1.0;
        }

        Ok(())
    }

    /// Predict on a single sample
    pub fn predict_single(&self, x: &[f64]) -> Result<f64> {
        let leaf_id = self.find_leaf(x, self.root_id)?;

        if let Some(leaf) = self.nodes.iter().find(|n| n.id == leaf_id) {
            Ok(leaf.get_prediction())
        } else {
            Err(SklearsError::PredictError(
                "Leaf node not found".to_string(),
            ))
        }
    }

    /// Get tree statistics
    pub fn get_stats(&self) -> HoeffdingTreeStats {
        HoeffdingTreeStats {
            n_nodes: self.nodes.len(),
            n_leaves: self.nodes.iter().filter(|n| n.is_leaf).count(),
            total_samples: self.nodes.iter().map(|n| n.sample_count).sum(),
            max_depth: self.nodes.iter().map(|n| n.depth).max().unwrap_or(0),
            memory_usage: self.estimate_memory_usage(),
        }
    }

    /// Estimate memory usage
    fn estimate_memory_usage(&self) -> usize {
        // Rough estimate based on number of nodes and their statistics
        self.nodes.len() * 1000 // Simplified estimation
    }
}

/// Statistics for Hoeffding Tree
#[derive(Debug, Clone)]
pub struct HoeffdingTreeStats {
    /// Number of nodes in the tree
    pub n_nodes: usize,
    /// Number of leaf nodes
    pub n_leaves: usize,
    /// Total samples processed
    pub total_samples: usize,
    /// Maximum depth of the tree
    pub max_depth: usize,
    /// Estimated memory usage (bytes)
    pub memory_usage: usize,
}

impl StreamingTreeModel for HoeffdingTree {
    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        let mut predictions = Array1::zeros(x.nrows());

        for (i, sample) in x.rows().into_iter().enumerate() {
            let sample_vec: Vec<f64> = sample.to_vec();
            predictions[i] = self.predict_single(&sample_vec)?;
        }

        Ok(predictions)
    }

    fn update(&mut self, x: &Array2<f64>, y: &Array1<f64>, _weights: &Array1<f64>) -> Result<()> {
        for (i, sample) in x.rows().into_iter().enumerate() {
            let sample_vec: Vec<f64> = sample.to_vec();
            let target = y[i];

            // Determine if this is classification (integer targets) or regression
            let class_label = if target.fract() == 0.0 {
                Some(target as i32)
            } else {
                None
            };

            self.update(&sample_vec, target, class_label)?;
        }

        Ok(())
    }

    fn get_accuracy(&self, x: &Array2<f64>, y: &Array1<f64>) -> Result<f64> {
        let predictions = self.predict(x)?;

        // Calculate accuracy based on problem type
        if y.iter().all(|&val| val.fract() == 0.0) {
            // Classification accuracy
            let correct = predictions
                .iter()
                .zip(y.iter())
                .map(|(&pred, &actual)| {
                    if (pred.round() - actual).abs() < 1e-6 {
                        1.0
                    } else {
                        0.0
                    }
                })
                .sum::<f64>();
            Ok(correct / predictions.len() as f64)
        } else {
            // Regression R²
            let y_mean = y.mean().unwrap_or(0.0);
            let ss_res = predictions
                .iter()
                .zip(y.iter())
                .map(|(&pred, &actual)| (actual - pred).powi(2))
                .sum::<f64>();
            let ss_tot = y
                .iter()
                .map(|&actual| (actual - y_mean).powi(2))
                .sum::<f64>();

            if ss_tot == 0.0 {
                Ok(1.0)
            } else {
                Ok((1.0 - ss_res / ss_tot).max(0.0))
            }
        }
    }

    fn rebuild(&mut self, x: &Array2<f64>, y: &Array1<f64>, _weights: &Array1<f64>) -> Result<()> {
        // Reset tree to single root node
        let n_features = x.ncols();
        *self = HoeffdingTree::new(self.config.clone(), n_features);

        // Re-train on all data using the StreamingTreeModel update method
        for (i, sample) in x.rows().into_iter().enumerate() {
            let sample_vec: Vec<f64> = sample.to_vec();
            let target = y[i];

            // Determine if this is classification (integer targets) or regression
            let class_label = if target.fract() == 0.0 {
                Some(target as i32)
            } else {
                None
            };

            self.update(&sample_vec, target, class_label)?;
        }

        Ok(())
    }
}
