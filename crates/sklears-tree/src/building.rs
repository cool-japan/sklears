//! Tree building algorithms and impurity calculations
//!
//! This module contains the core algorithms for constructing decision trees,
//! including various splitting strategies and impurity measures.

use std::collections::BinaryHeap;

use scirs2_core::ndarray::{Array1, Array2, ArrayView1};
use scirs2_core::random::{Random, rng};
use sklears_core::error::{Result, SklearsError};

use crate::criteria::{DecisionTreeConfig, SplitCriterion, TreeGrowingStrategy, FeatureType, MaxFeatures, SplitType, MultiwaySplit};

/// Tree node for building decision trees
#[derive(Debug, Clone)]
pub struct TreeNode {
    /// Node ID
    pub id: usize,
    /// Depth of this node
    pub depth: usize,
    /// Samples in this node
    pub sample_indices: Vec<usize>,
    /// Impurity of this node
    pub impurity: f64,
    /// Predicted value/class for this node
    pub prediction: f64,
    /// Potential impurity decrease if this node is split
    pub potential_decrease: f64,
    /// Best split for this node (if any)
    pub best_split: Option<CustomSplit>,
    /// Parent node ID
    pub parent_id: Option<usize>,
    /// Whether this is a leaf node
    pub is_leaf: bool,
}

/// Custom split information
#[derive(Debug, Clone)]
pub struct CustomSplit {
    pub feature_idx: usize,
    pub threshold: f64,
    pub impurity_decrease: f64,
    pub left_count: usize,
    pub right_count: usize,
}

/// Best-first tree builder
#[derive(Debug)]
pub struct BestFirstTreeBuilder {
    /// Nodes in the tree
    pub nodes: Vec<TreeNode>,
    /// Priority queue of nodes to expand (ordered by potential decrease)
    pub node_queue: BinaryHeap<NodePriority>,
    /// Next node ID
    pub next_node_id: usize,
    /// Current number of leaves
    pub n_leaves: usize,
}

/// Priority wrapper for nodes in the queue
#[derive(Debug, Clone)]
pub struct NodePriority {
    pub node_id: usize,
    pub priority: f64, // Negative of impurity decrease for max-heap behavior
}

impl PartialEq for NodePriority {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority
    }
}

impl Eq for NodePriority {}

impl PartialOrd for NodePriority {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        // Reverse order for max-heap (highest priority first)
        other.priority.partial_cmp(&self.priority)
    }
}

impl Ord for NodePriority {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap_or(std::cmp::Ordering::Equal)
    }
}

impl BestFirstTreeBuilder {
    /// Create a new best-first tree builder
    pub fn new(
        x: &Array2<f64>,
        y: &Array1<i32>,
        config: &DecisionTreeConfig,
        n_classes: usize,
    ) -> Self {
        let n_samples = x.nrows();
        let sample_indices: Vec<usize> = (0..n_samples).collect();

        // Calculate root impurity and prediction
        let mut class_counts = vec![0; n_classes];
        for &sample_idx in &sample_indices {
            let class = y[sample_idx] as usize;
            if class < n_classes {
                class_counts[class] += 1;
            }
        }

        let class_counts_i32: Vec<i32> = class_counts.iter().map(|&x| x as i32).collect();
        let impurity = gini_impurity(&class_counts_i32, n_samples as i32);
        let prediction = class_counts
            .iter()
            .enumerate()
            .max_by_key(|(_, &count)| count)
            .map(|(class, _)| class as f64)
            .unwrap_or(0.0);

        // Find best split for root
        let best_split = find_best_split_for_node(x, y, &sample_indices, config, n_classes);
        let potential_decrease = best_split
            .as_ref()
            .map(|s| s.impurity_decrease)
            .unwrap_or(0.0);

        let root_node = TreeNode {
            id: 0,
            depth: 0,
            sample_indices,
            impurity,
            prediction,
            potential_decrease,
            best_split,
            parent_id: None,
            is_leaf: false,
        };

        let mut node_queue = BinaryHeap::new();
        if potential_decrease > 0.0 {
            node_queue.push(NodePriority {
                node_id: 0,
                priority: -potential_decrease, // Negative for max-heap
            });
        }

        Self {
            nodes: vec![root_node],
            node_queue,
            next_node_id: 1,
            n_leaves: 1,
        }
    }

    /// Build the tree using best-first strategy
    pub fn build_tree(
        &mut self,
        x: &Array2<f64>,
        y: &Array1<i32>,
        config: &DecisionTreeConfig,
        n_classes: usize,
    ) -> Result<()> {
        let max_leaves = match config.growing_strategy {
            TreeGrowingStrategy::BestFirst { max_leaves } => max_leaves,
            _ => None,
        };

        while let Some(node_priority) = self.node_queue.pop() {
            let node_id = node_priority.node_id;

            // Check stopping criteria
            if let Some(max_leaves) = max_leaves {
                if self.n_leaves >= max_leaves {
                    break;
                }
            }

            if let Some(max_depth) = config.max_depth {
                if self.nodes[node_id].depth >= max_depth {
                    continue;
                }
            }

            // Check if we can split this node
            if self.nodes[node_id].sample_indices.len() < config.min_samples_split {
                continue;
            }

            // Split the node
            if self.split_node(node_id, x, y, config, n_classes).is_err() {
                continue;
            }
        }

        Ok(())
    }

    /// Split a node and add children to the queue
    fn split_node(
        &mut self,
        node_id: usize,
        x: &Array2<f64>,
        y: &Array1<i32>,
        config: &DecisionTreeConfig,
        n_classes: usize,
    ) -> Result<()> {
        let node = self.nodes[node_id].clone();
        let best_split = match &node.best_split {
            Some(split) => split.clone(),
            None => {
                return Err(SklearsError::InvalidInput(
                    "No valid split found".to_string(),
                ))
            }
        };

        // Split samples
        let (left_indices, right_indices) = split_samples_by_threshold(
            x,
            &node.sample_indices,
            best_split.feature_idx,
            best_split.threshold,
        );

        if left_indices.len() < config.min_samples_leaf
            || right_indices.len() < config.min_samples_leaf
        {
            return Err(SklearsError::InvalidInput(
                "Split would create leaves with too few samples".to_string(),
            ));
        }

        // Create child nodes
        let left_node_id = self.create_child_node(&left_indices, x, y, config, n_classes, Some(node_id))?;
        let right_node_id = self.create_child_node(&right_indices, x, y, config, n_classes, Some(node_id))?;

        // Mark current node as not a leaf
        self.nodes[node_id].is_leaf = false;

        // Update leaf count
        self.n_leaves += 1; // Added two leaves, removed one

        Ok(())
    }

    /// Create a child node
    fn create_child_node(
        &mut self,
        sample_indices: &[usize],
        x: &Array2<f64>,
        y: &Array1<i32>,
        config: &DecisionTreeConfig,
        n_classes: usize,
        parent_id: Option<usize>,
    ) -> Result<usize> {
        let node_id = self.next_node_id;
        self.next_node_id += 1;

        // Calculate class counts and prediction
        let mut class_counts = vec![0; n_classes];
        for &sample_idx in sample_indices {
            let class = y[sample_idx] as usize;
            if class < n_classes {
                class_counts[class] += 1;
            }
        }

        let class_counts_i32: Vec<i32> = class_counts.iter().map(|&x| x as i32).collect();
        let impurity = gini_impurity(&class_counts_i32, sample_indices.len() as i32);
        let prediction = class_counts
            .iter()
            .enumerate()
            .max_by_key(|(_, &count)| count)
            .map(|(class, _)| class as f64)
            .unwrap_or(0.0);

        let depth = parent_id.map(|pid| self.nodes[pid].depth + 1).unwrap_or(0);

        // Find best split
        let best_split = find_best_split_for_node(x, y, sample_indices, config, n_classes);
        let potential_decrease = best_split
            .as_ref()
            .map(|s| s.impurity_decrease)
            .unwrap_or(0.0);

        let node = TreeNode {
            id: node_id,
            depth,
            sample_indices: sample_indices.to_vec(),
            impurity,
            prediction,
            potential_decrease,
            best_split,
            parent_id,
            is_leaf: true,
        };

        // Add to priority queue if it can be split
        if potential_decrease > 0.0 {
            self.node_queue.push(NodePriority {
                node_id,
                priority: -potential_decrease,
            });
        }

        self.nodes.push(node);
        Ok(node_id)
    }
}

/// Calculate Gini impurity
pub fn gini_impurity(class_counts: &[i32], total_samples: i32) -> f64 {
    if total_samples == 0 {
        return 0.0;
    }

    let mut gini = 1.0;
    for &count in class_counts {
        let prob = count as f64 / total_samples as f64;
        gini -= prob * prob;
    }
    gini
}

/// Calculate Mean Absolute Error (MAE) impurity for regression
pub fn mae_impurity(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }

    let median = {
        let mut sorted_values = values.to_vec();
        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let len = sorted_values.len();
        if len % 2 == 0 {
            (sorted_values[len / 2 - 1] + sorted_values[len / 2]) / 2.0
        } else {
            sorted_values[len / 2]
        }
    };

    values.iter().map(|v| (v - median).abs()).sum::<f64>() / values.len() as f64
}

/// Calculate Twoing criterion impurity for binary classification
pub fn twoing_impurity(left_counts: &[usize], right_counts: &[usize]) -> f64 {
    let left_total: usize = left_counts.iter().sum();
    let right_total: usize = right_counts.iter().sum();
    let total = left_total + right_total;

    if total == 0 || left_total == 0 || right_total == 0 {
        return 0.0;
    }

    let mut twoing_value = 0.0;
    for i in 0..left_counts.len() {
        let left_prob = left_counts[i] as f64 / left_total as f64;
        let right_prob = right_counts[i] as f64 / right_total as f64;
        twoing_value += (left_prob - right_prob).abs();
    }

    // Twoing criterion: 0.25 * p_left * p_right * (sum|p_left_i - p_right_i|)^2
    let p_left = left_total as f64 / total as f64;
    let p_right = right_total as f64 / total as f64;
    0.25 * p_left * p_right * twoing_value.powi(2)
}

/// Calculate Log-loss impurity for probability-based classification
pub fn log_loss_impurity(class_counts: &[usize]) -> f64 {
    let total: usize = class_counts.iter().sum();
    if total == 0 {
        return 0.0;
    }

    class_counts
        .iter()
        .filter(|&&count| count > 0)
        .map(|&count| {
            let prob = count as f64 / total as f64;
            -prob * prob.ln()
        })
        .sum()
}

/// Find best split using MAE criterion for regression
pub fn find_best_mae_split(
    x: &Array2<f64>,
    y: &Array1<f64>,
    feature_indices: &[usize],
) -> Option<CustomSplit> {
    let mut best_split: Option<CustomSplit> = None;
    let mut best_impurity_decrease = f64::NEG_INFINITY;

    // Calculate initial impurity
    let y_values: Vec<f64> = y.iter().cloned().collect();
    let initial_impurity = mae_impurity(&y_values);

    for &feature_idx in feature_indices {
        let feature_values = x.column(feature_idx);

        // Create (value, target) pairs and sort by feature value
        let mut pairs: Vec<(f64, f64)> = feature_values
            .iter()
            .zip(y.iter())
            .map(|(&x_val, &y_val)| (x_val, y_val))
            .collect();

        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        // Try each potential split point
        for i in 1..pairs.len() {
            if pairs[i - 1].0 >= pairs[i].0 {
                continue; // Skip identical values
            }

            let threshold = (pairs[i - 1].0 + pairs[i].0) / 2.0;

            let left_values: Vec<f64> = pairs[..i].iter().map(|(_, y)| *y).collect();
            let right_values: Vec<f64> = pairs[i..].iter().map(|(_, y)| *y).collect();

            if left_values.is_empty() || right_values.is_empty() {
                continue;
            }

            let left_impurity = mae_impurity(&left_values);
            let right_impurity = mae_impurity(&right_values);

            let left_weight = left_values.len() as f64 / pairs.len() as f64;
            let right_weight = right_values.len() as f64 / pairs.len() as f64;
            let weighted_impurity = left_weight * left_impurity + right_weight * right_impurity;

            let impurity_decrease = initial_impurity - weighted_impurity;

            if impurity_decrease > best_impurity_decrease {
                best_impurity_decrease = impurity_decrease;
                best_split = Some(CustomSplit {
                    feature_idx,
                    threshold,
                    impurity_decrease,
                    left_count: left_values.len(),
                    right_count: right_values.len(),
                });
            }
        }
    }

    best_split
}

/// Find best split for a node
pub fn find_best_split_for_node(
    x: &Array2<f64>,
    y: &Array1<i32>,
    sample_indices: &[usize],
    config: &DecisionTreeConfig,
    n_classes: usize,
) -> Option<CustomSplit> {
    if sample_indices.len() < config.min_samples_split {
        return None;
    }

    let n_features = x.ncols();

    // Determine which features to consider
    let feature_indices: Vec<usize> = match &config.max_features {
        MaxFeatures::All => (0..n_features).collect(),
        MaxFeatures::Sqrt => {
            let n_features_to_use = (n_features as f64).sqrt() as usize;
            let mut rng = thread_rng();
            let mut indices: Vec<usize> = (0..n_features).collect();
            indices.shuffle(&mut rng);
            indices.into_iter().take(n_features_to_use).collect()
        }
        MaxFeatures::Log2 => {
            let n_features_to_use = (n_features as f64).log2() as usize;
            let mut rng = thread_rng();
            let mut indices: Vec<usize> = (0..n_features).collect();
            indices.shuffle(&mut rng);
            indices.into_iter().take(n_features_to_use).collect()
        }
        MaxFeatures::Number(n) => {
            let n_features_to_use = (*n).min(n_features);
            let mut rng = thread_rng();
            let mut indices: Vec<usize> = (0..n_features).collect();
            indices.shuffle(&mut rng);
            indices.into_iter().take(n_features_to_use).collect()
        }
        MaxFeatures::Fraction(f) => {
            let n_features_to_use = (n_features as f64 * f) as usize;
            let mut rng = thread_rng();
            let mut indices: Vec<usize> = (0..n_features).collect();
            indices.shuffle(&mut rng);
            indices.into_iter().take(n_features_to_use).collect()
        }
    };

    let mut best_split: Option<CustomSplit> = None;
    let mut best_impurity_decrease = f64::NEG_INFINITY;

    for &feature_idx in &feature_indices {
        // Get feature values for samples
        let feature_values: Vec<f64> = sample_indices
            .iter()
            .map(|&idx| x[[idx, feature_idx]])
            .collect();

        let target_values: Vec<i32> = sample_indices
            .iter()
            .map(|&idx| y[idx])
            .collect();

        // Create (value, target) pairs and sort by feature value
        let mut pairs: Vec<(f64, i32)> = feature_values
            .iter()
            .zip(target_values.iter())
            .map(|(&x_val, &y_val)| (x_val, y_val))
            .collect();

        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        // Calculate initial impurity
        let mut class_counts = vec![0; n_classes];
        for &(_, class) in &pairs {
            if class >= 0 && (class as usize) < n_classes {
                class_counts[class as usize] += 1;
            }
        }
        let class_counts_i32: Vec<i32> = class_counts.iter().map(|&x| x as i32).collect();
        let initial_impurity = gini_impurity(&class_counts_i32, pairs.len() as i32);

        // Try each potential split point
        for i in 1..pairs.len() {
            if pairs[i - 1].0 >= pairs[i].0 {
                continue; // Skip identical values
            }

            let threshold = (pairs[i - 1].0 + pairs[i].0) / 2.0;

            // Count classes on left and right
            let mut left_counts = vec![0; n_classes];
            let mut right_counts = vec![0; n_classes];

            for j in 0..pairs.len() {
                let class = pairs[j].1;
                if class >= 0 && (class as usize) < n_classes {
                    if j < i {
                        left_counts[class as usize] += 1;
                    } else {
                        right_counts[class as usize] += 1;
                    }
                }
            }

            let left_total: i32 = left_counts.iter().sum();
            let right_total: i32 = right_counts.iter().sum();

            if left_total < config.min_samples_leaf as i32 || right_total < config.min_samples_leaf as i32 {
                continue;
            }

            let left_counts_i32: Vec<i32> = left_counts.iter().map(|&x| x as i32).collect();
            let right_counts_i32: Vec<i32> = right_counts.iter().map(|&x| x as i32).collect();

            let left_impurity = gini_impurity(&left_counts_i32, left_total);
            let right_impurity = gini_impurity(&right_counts_i32, right_total);

            let total_samples = pairs.len() as f64;
            let left_weight = left_total as f64 / total_samples;
            let right_weight = right_total as f64 / total_samples;
            let weighted_impurity = left_weight * left_impurity + right_weight * right_impurity;

            let impurity_decrease = initial_impurity - weighted_impurity;

            if impurity_decrease > best_impurity_decrease && impurity_decrease > config.min_impurity_decrease {
                best_impurity_decrease = impurity_decrease;
                best_split = Some(CustomSplit {
                    feature_idx,
                    threshold,
                    impurity_decrease,
                    left_count: left_total as usize,
                    right_count: right_total as usize,
                });
            }
        }
    }

    best_split
}

/// Split samples by threshold
pub fn split_samples_by_threshold(
    x: &Array2<f64>,
    sample_indices: &[usize],
    feature_idx: usize,
    threshold: f64,
) -> (Vec<usize>, Vec<usize>) {
    let mut left_indices = Vec::new();
    let mut right_indices = Vec::new();

    for &sample_idx in sample_indices {
        let feature_value = x[[sample_idx, feature_idx]];
        if feature_value <= threshold {
            left_indices.push(sample_idx);
        } else {
            right_indices.push(sample_idx);
        }
    }

    (left_indices, right_indices)
}

/// Apply multiway split to samples
pub fn apply_multiway_split(
    feature_values: &[String],
    multiway_split: &MultiwaySplit,
) -> Vec<Vec<usize>> {
    let mut branches: Vec<Vec<usize>> = vec![Vec::new(); multiway_split.category_branches.len()];

    for (i, feature_value) in feature_values.iter().enumerate() {
        for (branch_idx, branch_categories) in multiway_split.category_branches.iter().enumerate() {
            if branch_categories.contains(feature_value) {
                branches[branch_idx].push(i);
                break;
            }
        }
    }

    branches
}

/// Apply surrogate splits for missing values
pub fn apply_surrogate_splits<T: Clone>(
    x: &Array2<f64>,
    sample_indices: &[usize],
    primary_feature: usize,
    primary_threshold: f64,
    surrogate_splits: &[(usize, f64)],
) -> (Vec<usize>, Vec<usize>) {
    let mut left_indices = Vec::new();
    let mut right_indices = Vec::new();

    for &sample_idx in sample_indices {
        let primary_value = x[[sample_idx, primary_feature]];

        // Check if primary feature has missing value (assuming NaN represents missing)
        if primary_value.is_nan() {
            // Use surrogate splits
            let mut assigned = false;
            for &(surrogate_feature, surrogate_threshold) in surrogate_splits {
                let surrogate_value = x[[sample_idx, surrogate_feature]];
                if !surrogate_value.is_nan() {
                    if surrogate_value <= surrogate_threshold {
                        left_indices.push(sample_idx);
                    } else {
                        right_indices.push(sample_idx);
                    }
                    assigned = true;
                    break;
                }
            }

            // If no surrogate could be used, assign to majority branch
            if !assigned {
                left_indices.push(sample_idx); // Default to left
            }
        } else {
            // Use primary split
            if primary_value <= primary_threshold {
                left_indices.push(sample_idx);
            } else {
                right_indices.push(sample_idx);
            }
        }
    }

    (left_indices, right_indices)
}

/// Find surrogate splits for handling missing values
pub fn find_surrogate_splits(
    x: &Array2<f64>,
    sample_indices: &[usize],
    primary_split: &CustomSplit,
    n_surrogates: usize,
) -> Vec<(usize, f64)> {
    let mut surrogate_splits = Vec::new();
    let n_features = x.ncols();

    // Get primary split assignment
    let (primary_left, primary_right) = split_samples_by_threshold(
        x,
        sample_indices,
        primary_split.feature_idx,
        primary_split.threshold,
    );

    for feature_idx in 0..n_features {
        if feature_idx == primary_split.feature_idx {
            continue;
        }

        // Find best threshold for this feature that mimics primary split
        let feature_values: Vec<f64> = sample_indices
            .iter()
            .map(|&idx| x[[idx, feature_idx]])
            .collect();

        let mut pairs: Vec<(f64, bool)> = feature_values
            .iter()
            .enumerate()
            .map(|(i, &val)| {
                let sample_idx = sample_indices[i];
                let goes_left = primary_left.contains(&sample_idx);
                (val, goes_left)
            })
            .collect();

        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        let mut best_threshold = 0.0;
        let mut best_agreement = 0;

        for i in 1..pairs.len() {
            if pairs[i - 1].0 >= pairs[i].0 {
                continue;
            }

            let threshold = (pairs[i - 1].0 + pairs[i].0) / 2.0;
            let mut agreement = 0;

            for j in 0..pairs.len() {
                let predicted_left = pairs[j].0 <= threshold;
                let actual_left = pairs[j].1;
                if predicted_left == actual_left {
                    agreement += 1;
                }
            }

            if agreement > best_agreement {
                best_agreement = agreement;
                best_threshold = threshold;
            }
        }

        if best_agreement > 0 {
            surrogate_splits.push((feature_idx, best_threshold));
        }

        if surrogate_splits.len() >= n_surrogates {
            break;
        }
    }

    surrogate_splits
}