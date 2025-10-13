//! Tree building algorithms and utilities
//!
//! This module contains algorithms for building decision trees, including
//! split finding, impurity calculations, and feature grouping utilities.

use crate::config::*;
use crate::node::*;
use crate::SplitCriterion;
use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::{
    error::{Result, SklearsError},
    types::Float,
};
use std::collections::BinaryHeap;

/// Handle missing values in the data based on the specified strategy
pub fn handle_missing_values<T: Clone>(
    x: &Array2<f64>,
    y: &Array1<T>,
    strategy: MissingValueStrategy,
) -> Result<(Array2<f64>, Array1<T>)> {
    // Check for missing values (NaN)
    let mut has_missing = false;
    for value in x.iter() {
        if value.is_nan() {
            has_missing = true;
            break;
        }
    }
    if !has_missing {
        // No missing values, return original data
        return Ok((x.clone(), y.clone()));
    }
    match strategy {
        MissingValueStrategy::Skip => {
            // Remove rows with any missing values
            let mut valid_indices = Vec::new();
            for (row_idx, row) in x.outer_iter().enumerate() {
                let mut row_valid = true;
                for &value in row.iter() {
                    if value.is_nan() {
                        row_valid = false;
                        break;
                    }
                }
                if row_valid {
                    valid_indices.push(row_idx);
                }
            }
            if valid_indices.is_empty() {
                return Err(SklearsError::InvalidData {
                    reason: "All samples contain missing values".to_string(),
                });
            }
            // Create new arrays with only valid rows
            let n_valid = valid_indices.len();
            let n_features = x.ncols();
            let mut x_clean = Array2::zeros((n_valid, n_features));
            let mut y_clean = Vec::with_capacity(n_valid);
            for (new_idx, &orig_idx) in valid_indices.iter().enumerate() {
                x_clean.row_mut(new_idx).assign(&x.row(orig_idx));
                y_clean.push(y[orig_idx].clone());
            }
            Ok((x_clean, Array1::from_vec(y_clean)))
        }
        MissingValueStrategy::Majority => {
            // Replace missing values with column means (for continuous) or mode (for discrete)
            let mut x_imputed = x.clone();
            for col_idx in 0..x.ncols() {
                let column = x.column(col_idx);
                // Calculate mean of non-missing values
                let mut sum = 0.0;
                let mut count = 0;
                for &value in column.iter() {
                    if !value.is_nan() {
                        sum += value;
                        count += 1;
                    }
                }
                if count > 0 {
                    let mean = sum / count as f64;
                    // Replace missing values with mean
                    for row_idx in 0..x.nrows() {
                        if x_imputed[[row_idx, col_idx]].is_nan() {
                            x_imputed[[row_idx, col_idx]] = mean;
                        }
                    }
                } else {
                    // All values in this column are missing, use 0.0
                    for row_idx in 0..x.nrows() {
                        x_imputed[[row_idx, col_idx]] = 0.0;
                    }
                }
            }
            Ok((x_imputed, y.clone()))
        }
        MissingValueStrategy::Surrogate => {
            // TODO: Implement proper surrogate splits for missing value handling
            // For now, fall back to mean imputation
            let mut x_imputed = x.clone();
            for col_idx in 0..x.ncols() {
                let mut sum = 0.0;
                let mut count = 0;
                // Calculate mean of non-missing values
                for row_idx in 0..x.nrows() {
                    let value = x[[row_idx, col_idx]];
                    if !value.is_nan() {
                        sum += value;
                        count += 1;
                    }
                }
                if count > 0 {
                    let mean = sum / count as f64;
                    // Replace missing values with mean
                    for row_idx in 0..x.nrows() {
                        if x_imputed[[row_idx, col_idx]].is_nan() {
                            x_imputed[[row_idx, col_idx]] = mean;
                        }
                    }
                } else {
                    // All values in this column are missing, use 0.0
                    for row_idx in 0..x.nrows() {
                        x_imputed[[row_idx, col_idx]] = 0.0;
                    }
                }
            }
            Ok((x_imputed, y.clone()))
        }
    }
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

        let impurity = gini_impurity(&class_counts, n_samples as i32);
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
        let node = &self.nodes[node_id].clone();
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
                "Split would create undersized leaves".to_string(),
            ));
        }

        // Create left child
        let left_node_id = self.next_node_id;
        self.next_node_id += 1;

        let left_node = self.create_child_node(
            left_node_id,
            node.id,
            node.depth + 1,
            left_indices,
            x,
            y,
            config,
            n_classes,
        );

        // Create right child
        let right_node_id = self.next_node_id;
        self.next_node_id += 1;

        let right_node = self.create_child_node(
            right_node_id,
            node.id,
            node.depth + 1,
            right_indices,
            x,
            y,
            config,
            n_classes,
        );

        // Add children to queue if they can be split
        if left_node.potential_decrease > config.min_impurity_decrease {
            self.node_queue.push(NodePriority {
                node_id: left_node_id,
                priority: -left_node.potential_decrease,
            });
        }

        if right_node.potential_decrease > config.min_impurity_decrease {
            self.node_queue.push(NodePriority {
                node_id: right_node_id,
                priority: -right_node.potential_decrease,
            });
        }

        self.nodes.push(left_node);
        self.nodes.push(right_node);

        // Mark parent as non-leaf and increment leaf count
        self.nodes[node_id].is_leaf = false;
        self.n_leaves += 1; // +2 children -1 parent = +1 net leaves

        Ok(())
    }

    /// Create a child node
    #[allow(clippy::too_many_arguments)]
    fn create_child_node(
        &self,
        node_id: usize,
        parent_id: usize,
        depth: usize,
        sample_indices: Vec<usize>,
        x: &Array2<f64>,
        y: &Array1<i32>,
        config: &DecisionTreeConfig,
        n_classes: usize,
    ) -> TreeNode {
        // Calculate impurity and prediction
        let mut class_counts = vec![0; n_classes];
        for &sample_idx in &sample_indices {
            let class = y[sample_idx] as usize;
            if class < n_classes {
                class_counts[class] += 1;
            }
        }

        let impurity = gini_impurity(&class_counts, sample_indices.len() as i32);
        let prediction = class_counts
            .iter()
            .enumerate()
            .max_by_key(|(_, &count)| count)
            .map(|(class, _)| class as f64)
            .unwrap_or(0.0);

        // Find best split for this node
        let best_split = find_best_split_for_node(x, y, &sample_indices, config, n_classes);
        let potential_decrease = best_split
            .as_ref()
            .map(|s| s.impurity_decrease)
            .unwrap_or(0.0);

        TreeNode {
            id: node_id,
            depth,
            sample_indices,
            impurity,
            prediction,
            potential_decrease,
            best_split,
            parent_id: Some(parent_id),
            is_leaf: true,
        }
    }
}

/// Find best split for a node given sample indices
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

    // Create subset of data for this node
    let n_samples = sample_indices.len();
    let n_features = x.ncols();

    let mut node_x = Array2::zeros((n_samples, n_features));
    let mut node_y = Array1::zeros(n_samples);

    for (new_idx, &orig_idx) in sample_indices.iter().enumerate() {
        for j in 0..n_features {
            node_x[[new_idx, j]] = x[[orig_idx, j]];
        }
        node_y[new_idx] = y[orig_idx];
    }

    // Find best split using existing logic
    let feature_indices: Vec<usize> = (0..n_features).collect();

    match config.criterion {
        SplitCriterion::Gini | SplitCriterion::Entropy => {
            find_best_twoing_split(&node_x, &node_y, &feature_indices, n_classes)
        }
        SplitCriterion::LogLoss => {
            find_best_logloss_split(&node_x, &node_y, &feature_indices, n_classes)
        }
        _ => None, // For regression criteria, would need separate implementation
    }
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
        if x[[sample_idx, feature_idx]] <= threshold {
            left_indices.push(sample_idx);
        } else {
            right_indices.push(sample_idx);
        }
    }

    (left_indices, right_indices)
}

/// Find best split using MAE criterion for regression
pub fn find_best_mae_split(
    x: &Array2<f64>,
    y: &Array1<f64>,
    feature_indices: &[usize],
) -> Option<CustomSplit> {
    let n_samples = x.nrows();
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

            // Split data
            let left_values: Vec<f64> = pairs[..i].iter().map(|(_, y)| *y).collect();
            let right_values: Vec<f64> = pairs[i..].iter().map(|(_, y)| *y).collect();

            if left_values.is_empty() || right_values.is_empty() {
                continue;
            }

            // Calculate weighted impurity
            let left_impurity = mae_impurity(&left_values);
            let right_impurity = mae_impurity(&right_values);
            let left_weight = left_values.len() as f64 / n_samples as f64;
            let right_weight = right_values.len() as f64 / n_samples as f64;
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

/// Find best split using Twoing criterion for classification
pub fn find_best_twoing_split(
    x: &Array2<f64>,
    y: &Array1<i32>,
    feature_indices: &[usize],
    n_classes: usize,
) -> Option<CustomSplit> {
    let _n_samples = x.nrows();
    let mut best_split: Option<CustomSplit> = None;
    let mut best_impurity_decrease = f64::NEG_INFINITY;

    for &feature_idx in feature_indices {
        let feature_values = x.column(feature_idx);

        // Create (value, class) pairs and sort by feature value
        let mut pairs: Vec<(f64, i32)> = feature_values
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

            // Count classes in left and right splits
            let mut left_counts = vec![0; n_classes];
            let mut right_counts = vec![0; n_classes];

            for (j, (_, class)) in pairs.iter().enumerate() {
                let class_idx = *class as usize;
                if j < i {
                    left_counts[class_idx] += 1;
                } else {
                    right_counts[class_idx] += 1;
                }
            }

            let left_total: usize = left_counts.iter().sum();
            let right_total: usize = right_counts.iter().sum();

            if left_total == 0 || right_total == 0 {
                continue;
            }

            let impurity_decrease = twoing_impurity(&left_counts, &right_counts);

            if impurity_decrease > best_impurity_decrease {
                best_impurity_decrease = impurity_decrease;
                best_split = Some(CustomSplit {
                    feature_idx,
                    threshold,
                    impurity_decrease,
                    left_count: left_total,
                    right_count: right_total,
                });
            }
        }
    }

    best_split
}

/// Find best split using Log-loss criterion for classification
pub fn find_best_logloss_split(
    x: &Array2<f64>,
    y: &Array1<i32>,
    feature_indices: &[usize],
    n_classes: usize,
) -> Option<CustomSplit> {
    let n_samples = x.nrows();
    let mut best_split: Option<CustomSplit> = None;
    let mut best_impurity_decrease = f64::NEG_INFINITY;

    // Calculate initial impurity
    let mut initial_counts = vec![0; n_classes];
    for &class in y.iter() {
        initial_counts[class as usize] += 1;
    }
    let initial_impurity = log_loss_impurity(&initial_counts);

    for &feature_idx in feature_indices {
        let feature_values = x.column(feature_idx);

        // Create (value, class) pairs and sort by feature value
        let mut pairs: Vec<(f64, i32)> = feature_values
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

            // Count classes in left and right splits
            let mut left_counts = vec![0; n_classes];
            let mut right_counts = vec![0; n_classes];

            for (j, (_, class)) in pairs.iter().enumerate() {
                let class_idx = *class as usize;
                if j < i {
                    left_counts[class_idx] += 1;
                } else {
                    right_counts[class_idx] += 1;
                }
            }

            let left_total: usize = left_counts.iter().sum();
            let right_total: usize = right_counts.iter().sum();

            if left_total == 0 || right_total == 0 {
                continue;
            }

            // Calculate weighted impurity
            let left_impurity = log_loss_impurity(&left_counts);
            let right_impurity = log_loss_impurity(&right_counts);
            let left_weight = left_total as f64 / n_samples as f64;
            let right_weight = right_total as f64 / n_samples as f64;
            let weighted_impurity = left_weight * left_impurity + right_weight * right_impurity;

            let impurity_decrease = initial_impurity - weighted_impurity;

            if impurity_decrease > best_impurity_decrease {
                best_impurity_decrease = impurity_decrease;
                best_split = Some(CustomSplit {
                    feature_idx,
                    threshold,
                    impurity_decrease,
                    left_count: left_total,
                    right_count: right_total,
                });
            }
        }
    }

    best_split
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

/// Calculate gini impurity for multiway splits
pub fn gini_impurity(class_counts: &[i32], total_samples: i32) -> f64 {
    if total_samples == 0 {
        return 0.0;
    }

    let mut impurity = 1.0;
    for &count in class_counts {
        let probability = count as f64 / total_samples as f64;
        impurity -= probability * probability;
    }
    impurity
}

/// Apply feature grouping to training data
pub fn apply_feature_grouping(
    grouping: &FeatureGrouping,
    x: &Array2<Float>,
    y: &Array1<Float>,
) -> Result<(Array2<Float>, FeatureGroupInfo)> {
    match grouping {
        FeatureGrouping::None => {
            // No grouping - return original data
            let n_features = x.ncols();
            let info = FeatureGroupInfo {
                groups: (0..n_features).map(|i| vec![i]).collect(),
                representatives: (0..n_features).collect(),
                correlation_matrix: None,
                group_correlations: vec![1.0; n_features],
            };
            Ok((x.clone(), info))
        }
        FeatureGrouping::AutoCorrelation {
            threshold,
            selection_method,
        } => apply_auto_correlation_grouping(x, y, *threshold, *selection_method),
        FeatureGrouping::Manual {
            groups,
            selection_method,
        } => apply_manual_grouping(x, y, groups, *selection_method),
        FeatureGrouping::Hierarchical {
            n_clusters,
            linkage,
            selection_method,
        } => apply_hierarchical_grouping(x, y, *n_clusters, *linkage, *selection_method),
    }
}

/// Apply automatic correlation-based feature grouping
pub fn apply_auto_correlation_grouping(
    x: &Array2<Float>,
    y: &Array1<Float>,
    threshold: Float,
    selection_method: GroupSelectionMethod,
) -> Result<(Array2<Float>, FeatureGroupInfo)> {
    let n_features = x.ncols();

    if n_features == 0 {
        return Err(SklearsError::InvalidInput(
            "Cannot apply feature grouping to empty feature set".to_string(),
        ));
    }

    // Calculate feature correlation matrix
    let correlation_matrix = calculate_correlation_matrix(x)?;

    // Find groups of correlated features
    let mut groups = Vec::new();
    let mut assigned = vec![false; n_features];

    for i in 0..n_features {
        if assigned[i] {
            continue;
        }

        let mut group = vec![i];
        assigned[i] = true;

        // Find features correlated with feature i above threshold
        for j in (i + 1)..n_features {
            if !assigned[j] && correlation_matrix[[i, j]].abs() >= threshold {
                group.push(j);
                assigned[j] = true;
            }
        }

        groups.push(group);
    }

    // Select representative features from each group
    let mut representatives = Vec::new();
    let mut group_correlations = Vec::new();

    for group in &groups {
        let (representative, avg_correlation) =
            select_group_representative(x, y, group, selection_method)?;
        representatives.push(representative);
        group_correlations.push(avg_correlation);
    }

    // Create reduced feature matrix with only representatives
    let reduced_x = create_reduced_feature_matrix(x, &representatives)?;

    let info = FeatureGroupInfo {
        groups,
        representatives,
        correlation_matrix: Some(correlation_matrix),
        group_correlations,
    };

    Ok((reduced_x, info))
}

/// Apply manual feature grouping specified by user
pub fn apply_manual_grouping(
    x: &Array2<Float>,
    y: &Array1<Float>,
    groups: &[Vec<usize>],
    selection_method: GroupSelectionMethod,
) -> Result<(Array2<Float>, FeatureGroupInfo)> {
    let n_features = x.ncols();

    // Validate that groups don't overlap and cover all features
    let mut assigned = vec![false; n_features];
    for group in groups {
        for &feature_idx in group {
            if feature_idx >= n_features {
                return Err(SklearsError::InvalidInput(format!(
                    "Feature index {} out of bounds",
                    feature_idx
                )));
            }
            if assigned[feature_idx] {
                return Err(SklearsError::InvalidInput(format!(
                    "Feature {} appears in multiple groups",
                    feature_idx
                )));
            }
            assigned[feature_idx] = true;
        }
    }

    // Add ungrouped features as singleton groups
    let mut complete_groups = groups.to_vec();
    for (i, &is_assigned) in assigned.iter().enumerate() {
        if !is_assigned {
            complete_groups.push(vec![i]);
        }
    }

    // Select representatives for each group
    let mut representatives = Vec::new();
    let mut group_correlations = Vec::new();

    for group in &complete_groups {
        let (representative, avg_correlation) =
            select_group_representative(x, y, group, selection_method)?;
        representatives.push(representative);
        group_correlations.push(avg_correlation);
    }

    // Create reduced feature matrix
    let reduced_x = create_reduced_feature_matrix(x, &representatives)?;

    let info = FeatureGroupInfo {
        groups: complete_groups,
        representatives,
        correlation_matrix: None,
        group_correlations,
    };

    Ok((reduced_x, info))
}

/// Apply hierarchical clustering-based feature grouping
pub fn apply_hierarchical_grouping(
    x: &Array2<Float>,
    y: &Array1<Float>,
    n_clusters: usize,
    linkage: LinkageMethod,
    selection_method: GroupSelectionMethod,
) -> Result<(Array2<Float>, FeatureGroupInfo)> {
    let n_features = x.ncols();

    if n_clusters == 0 || n_clusters > n_features {
        return Err(SklearsError::InvalidInput(format!(
            "n_clusters must be between 1 and {} (number of features)",
            n_features
        )));
    }

    // Calculate distance matrix (1 - |correlation|)
    let correlation_matrix = calculate_correlation_matrix(x)?;
    let mut distance_matrix = Array2::<Float>::zeros((n_features, n_features));

    for i in 0..n_features {
        for j in 0..n_features {
            distance_matrix[[i, j]] = 1.0 - correlation_matrix[[i, j]].abs();
        }
    }

    // Perform hierarchical clustering (simplified implementation)
    let groups = hierarchical_clustering(&distance_matrix, n_clusters, linkage)?;

    // Select representatives for each group
    let mut representatives = Vec::new();
    let mut group_correlations = Vec::new();

    for group in &groups {
        let (representative, avg_correlation) =
            select_group_representative(x, y, group, selection_method)?;
        representatives.push(representative);
        group_correlations.push(avg_correlation);
    }

    // Create reduced feature matrix
    let reduced_x = create_reduced_feature_matrix(x, &representatives)?;

    let info = FeatureGroupInfo {
        groups,
        representatives,
        correlation_matrix: Some(correlation_matrix),
        group_correlations,
    };

    Ok((reduced_x, info))
}

/// Calculate correlation matrix for features
pub fn calculate_correlation_matrix(x: &Array2<Float>) -> Result<Array2<Float>> {
    let n_features = x.ncols();
    let n_samples = x.nrows();

    if n_samples < 2 {
        return Err(SklearsError::InvalidInput(
            "Need at least 2 samples to calculate correlations".to_string(),
        ));
    }

    let mut correlation_matrix = Array2::<Float>::zeros((n_features, n_features));

    // Calculate means
    let means: Vec<Float> = (0..n_features)
        .map(|j| x.column(j).mean().unwrap_or(0.0))
        .collect();

    // Calculate correlation for each pair of features
    for i in 0..n_features {
        for j in i..n_features {
            if i == j {
                correlation_matrix[[i, j]] = 1.0;
            } else {
                let corr = calculate_pearson_correlation(
                    &x.column(i).to_owned(),
                    &x.column(j).to_owned(),
                    means[i],
                    means[j],
                )?;
                correlation_matrix[[i, j]] = corr;
                correlation_matrix[[j, i]] = corr;
            }
        }
    }

    Ok(correlation_matrix)
}

/// Calculate Pearson correlation between two feature vectors
pub fn calculate_pearson_correlation(
    x: &Array1<Float>,
    y: &Array1<Float>,
    mean_x: Float,
    mean_y: Float,
) -> Result<Float> {
    let n = x.len();

    if n != y.len() {
        return Err(SklearsError::InvalidInput(format!(
            "Feature vectors must have same length: {} vs {}",
            n,
            y.len()
        )));
    }

    let mut sum_xy = 0.0;
    let mut sum_x2 = 0.0;
    let mut sum_y2 = 0.0;

    for i in 0..n {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        sum_xy += dx * dy;
        sum_x2 += dx * dx;
        sum_y2 += dy * dy;
    }

    let denominator = (sum_x2 * sum_y2).sqrt();

    if denominator.abs() < Float::EPSILON {
        Ok(0.0) // No correlation if one or both variables have no variance
    } else {
        Ok(sum_xy / denominator)
    }
}

/// Select representative feature from a group
pub fn select_group_representative(
    x: &Array2<Float>,
    y: &Array1<Float>,
    group: &[usize],
    method: GroupSelectionMethod,
) -> Result<(usize, Float)> {
    if group.is_empty() {
        return Err(SklearsError::InvalidInput(
            "Cannot select representative from empty group".to_string(),
        ));
    }

    if group.len() == 1 {
        return Ok((group[0], 1.0));
    }

    match method {
        GroupSelectionMethod::MaxVariance => {
            let mut max_variance = f64::NEG_INFINITY;
            let mut best_feature = group[0];

            for &feature_idx in group {
                let column = x.column(feature_idx);
                let mean = column.mean().unwrap_or(0.0);
                let variance =
                    column.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / column.len() as f64;

                if variance > max_variance {
                    max_variance = variance;
                    best_feature = feature_idx;
                }
            }

            Ok((best_feature, max_variance))
        }
        GroupSelectionMethod::MaxTargetCorrelation => {
            let mut max_correlation = f64::NEG_INFINITY;
            let mut best_feature = group[0];

            let y_mean = y.mean().unwrap_or(0.0);

            for &feature_idx in group {
                let x_col = x.column(feature_idx).to_owned();
                let x_mean = x_col.mean().unwrap_or(0.0);
                let correlation = calculate_pearson_correlation(&x_col, y, x_mean, y_mean)?;

                if correlation.abs() > max_correlation {
                    max_correlation = correlation.abs();
                    best_feature = feature_idx;
                }
            }

            Ok((best_feature, max_correlation))
        }
        GroupSelectionMethod::First => Ok((group[0], 1.0)),
        GroupSelectionMethod::Random => {
            use scirs2_core::random::thread_rng;
            let mut rng = thread_rng();
            let idx = rng.gen_range(0..group.len());
            Ok((group[idx], 1.0))
        }
        GroupSelectionMethod::WeightedAll => {
            // For now, just return the first feature
            // In a full implementation, this would modify the training to use all features
            Ok((group[0], 1.0))
        }
    }
}

/// Create reduced feature matrix with only representative features
pub fn create_reduced_feature_matrix(
    x: &Array2<Float>,
    representatives: &[usize],
) -> Result<Array2<Float>> {
    let n_samples = x.nrows();
    let n_representatives = representatives.len();

    let mut reduced_x = Array2::zeros((n_samples, n_representatives));

    for (new_col, &orig_col) in representatives.iter().enumerate() {
        if orig_col >= x.ncols() {
            return Err(SklearsError::InvalidInput(format!(
                "Representative feature index {} out of bounds",
                orig_col
            )));
        }

        reduced_x.column_mut(new_col).assign(&x.column(orig_col));
    }

    Ok(reduced_x)
}

/// Simple hierarchical clustering implementation
pub fn hierarchical_clustering(
    distance_matrix: &Array2<Float>,
    n_clusters: usize,
    linkage: LinkageMethod,
) -> Result<Vec<Vec<usize>>> {
    let n_features = distance_matrix.nrows();

    if n_features != distance_matrix.ncols() {
        return Err(SklearsError::InvalidInput(
            "Distance matrix must be square".to_string(),
        ));
    }

    // Start with each feature in its own cluster
    let mut clusters: Vec<Vec<usize>> = (0..n_features).map(|i| vec![i]).collect();

    // Merge clusters until we have the desired number
    while clusters.len() > n_clusters {
        // Find the two closest clusters
        let mut min_distance = Float::INFINITY;
        let mut merge_i = 0;
        let mut merge_j = 1;

        for i in 0..clusters.len() {
            for j in (i + 1)..clusters.len() {
                let distance =
                    cluster_distance(&clusters[i], &clusters[j], distance_matrix, linkage);
                if distance < min_distance {
                    min_distance = distance;
                    merge_i = i;
                    merge_j = j;
                }
            }
        }

        // Merge the closest clusters
        let cluster_j = clusters.remove(merge_j);
        clusters[merge_i].extend(cluster_j);
    }

    Ok(clusters)
}

/// Calculate distance between two clusters
fn cluster_distance(
    cluster1: &[usize],
    cluster2: &[usize],
    distance_matrix: &Array2<Float>,
    linkage: LinkageMethod,
) -> Float {
    match linkage {
        LinkageMethod::Single => {
            // Minimum distance between any two points
            let mut min_dist = Float::INFINITY;
            for &i in cluster1 {
                for &j in cluster2 {
                    let dist = distance_matrix[[i, j]];
                    if dist < min_dist {
                        min_dist = dist;
                    }
                }
            }
            min_dist
        }
        LinkageMethod::Complete => {
            // Maximum distance between any two points
            let mut max_dist = Float::NEG_INFINITY;
            for &i in cluster1 {
                for &j in cluster2 {
                    let dist = distance_matrix[[i, j]];
                    if dist > max_dist {
                        max_dist = dist;
                    }
                }
            }
            max_dist
        }
        LinkageMethod::Average => {
            // Average distance between all pairs of points
            let mut total_dist = 0.0;
            let mut count = 0;
            for &i in cluster1 {
                for &j in cluster2 {
                    total_dist += distance_matrix[[i, j]];
                    count += 1;
                }
            }
            if count > 0 {
                total_dist / count as Float
            } else {
                0.0
            }
        }
        LinkageMethod::Ward => {
            // For simplicity, use average linkage
            // A full Ward implementation would require centroid calculations
            let mut total_dist = 0.0;
            let mut count = 0;
            for &i in cluster1 {
                for &j in cluster2 {
                    total_dist += distance_matrix[[i, j]];
                    count += 1;
                }
            }
            if count > 0 {
                total_dist / count as Float
            } else {
                0.0
            }
        }
    }
}
