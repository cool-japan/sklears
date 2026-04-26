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

/// Internal helper: stores the winning surrogate candidate for a column
#[derive(Debug, Clone)]
struct SurrogateInfo {
    surrogate_col: usize,
    threshold: f64,
    /// Fraction of observed-on-both-columns rows that this surrogate agrees with
    /// the primary split direction (always in (0.5, 1.0] when actually chosen).
    agreement: f64,
    /// If true, values ≤ threshold map to the *left* side of the primary split;
    /// if false, they map to the *right* side.
    send_left: bool,
}

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
            // Surrogate-split imputation.
            //
            // For each column c that has missing values we:
            //   1. Find the non-missing rows and compute a binarised "primary split"
            //      on c using its median as threshold (left = ≤ median, right = > median).
            //   2. For every other column j (that has no missing values in those same rows)
            //      we find the threshold on j that best *mimics* the primary split
            //      direction (maximises agreement = fraction of rows assigned the same
            //      left/right side).
            //   3. The surrogate with the highest agreement wins.
            //   4. For rows where c is missing we use the winning surrogate to impute:
            //      we replace the NaN with the median of c-values that fell on the
            //      same side as the surrogate predicts.  If no surrogate beats 50 %
            //      agreement (or no surrogate is available) we fall back to the column
            //      mean of c.
            let n_rows = x.nrows();
            let n_cols = x.ncols();
            let mut x_imputed = x.clone();

            // Pre-compute per-column means for the fallback
            let col_means: Vec<f64> = (0..n_cols)
                .map(|c| {
                    let mut sum = 0.0f64;
                    let mut cnt = 0usize;
                    for r in 0..n_rows {
                        let v = x[[r, c]];
                        if !v.is_nan() {
                            sum += v;
                            cnt += 1;
                        }
                    }
                    if cnt > 0 {
                        sum / cnt as f64
                    } else {
                        0.0
                    }
                })
                .collect();

            for col_idx in 0..n_cols {
                // Rows where this column is missing
                let missing_rows: Vec<usize> =
                    (0..n_rows).filter(|&r| x[[r, col_idx]].is_nan()).collect();

                if missing_rows.is_empty() {
                    continue;
                }

                // Rows where this column is observed
                let obs_rows: Vec<usize> =
                    (0..n_rows).filter(|&r| !x[[r, col_idx]].is_nan()).collect();

                if obs_rows.is_empty() {
                    // All values missing — use 0.0
                    for &r in &missing_rows {
                        x_imputed[[r, col_idx]] = 0.0;
                    }
                    continue;
                }

                // Primary split on col_idx: use median as threshold
                let mut obs_vals: Vec<f64> = obs_rows.iter().map(|&r| x[[r, col_idx]]).collect();
                obs_vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let median = if obs_vals.len().is_multiple_of(2) {
                    (obs_vals[obs_vals.len() / 2 - 1] + obs_vals[obs_vals.len() / 2]) / 2.0
                } else {
                    obs_vals[obs_vals.len() / 2]
                };

                // Primary direction: 0 = left (≤ median), 1 = right (> median)
                let primary_dir: Vec<u8> = obs_rows
                    .iter()
                    .map(|&r| if x[[r, col_idx]] <= median { 0 } else { 1 })
                    .collect();

                // Find best surrogate among all other columns
                let mut best_surrogate: Option<SurrogateInfo> = None;

                for surr_col in 0..n_cols {
                    if surr_col == col_idx {
                        continue;
                    }

                    // Only use rows where *both* col_idx and surr_col are observed
                    let joint_rows: Vec<usize> = obs_rows
                        .iter()
                        .filter(|&&r| !x[[r, surr_col]].is_nan())
                        .cloned()
                        .collect();

                    if joint_rows.len() < 2 {
                        continue;
                    }

                    // Index joint_rows back into primary_dir
                    let joint_primary: Vec<u8> = joint_rows
                        .iter()
                        .map(|r| {
                            let pos = obs_rows.iter().position(|o| o == r).unwrap_or(0);
                            primary_dir[pos]
                        })
                        .collect();

                    // Try all midpoints of surr_col values to find the threshold that
                    // maximises agreement with the primary direction.
                    let mut surr_vals: Vec<(f64, u8)> = joint_rows
                        .iter()
                        .zip(joint_primary.iter())
                        .map(|(&r, &d)| (x[[r, surr_col]], d))
                        .collect();
                    surr_vals
                        .sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

                    let n_joint = joint_rows.len() as f64;
                    let mut best_agree = 0.0f64;
                    let mut best_thr = 0.0f64;
                    let mut best_send_left = true; // default: ≤ thr → matches left

                    for w in surr_vals.windows(2) {
                        if (w[0].0 - w[1].0).abs() < f64::EPSILON {
                            continue;
                        }
                        let thr = (w[0].0 + w[1].0) / 2.0;

                        // agreement assuming ≤ thr goes left
                        let agree_left = surr_vals
                            .iter()
                            .filter(|(v, d)| (*v <= thr && *d == 0) || (*v > thr && *d == 1))
                            .count() as f64
                            / n_joint;

                        // agreement assuming ≤ thr goes right (flipped)
                        let agree_right = 1.0 - agree_left;

                        let (agree, send_left) = if agree_left >= agree_right {
                            (agree_left, true)
                        } else {
                            (agree_right, false)
                        };

                        if agree > best_agree {
                            best_agree = agree;
                            best_thr = thr;
                            best_send_left = send_left;
                        }
                    }

                    if best_agree > best_surrogate.as_ref().map_or(0.0, |s| s.agreement) {
                        best_surrogate = Some(SurrogateInfo {
                            surrogate_col: surr_col,
                            threshold: best_thr,
                            agreement: best_agree,
                            send_left: best_send_left,
                        });
                    }
                }

                // Impute missing values using the best surrogate (or mean fallback)
                match best_surrogate.filter(|s| s.agreement > 0.5) {
                    Some(surr) => {
                        // Compute left-side and right-side means of col_idx from observed rows
                        let left_vals: Vec<f64> = obs_rows
                            .iter()
                            .filter(|&&r| x[[r, col_idx]] <= median)
                            .map(|&r| x[[r, col_idx]])
                            .collect();
                        let right_vals: Vec<f64> = obs_rows
                            .iter()
                            .filter(|&&r| x[[r, col_idx]] > median)
                            .map(|&r| x[[r, col_idx]])
                            .collect();

                        let left_mean = if left_vals.is_empty() {
                            col_means[col_idx]
                        } else {
                            left_vals.iter().sum::<f64>() / left_vals.len() as f64
                        };
                        let right_mean = if right_vals.is_empty() {
                            col_means[col_idx]
                        } else {
                            right_vals.iter().sum::<f64>() / right_vals.len() as f64
                        };

                        for &r in &missing_rows {
                            let surr_val = if !x[[r, surr.surrogate_col]].is_nan() {
                                x[[r, surr.surrogate_col]]
                            } else {
                                // Surrogate also missing — use global mean
                                x_imputed[[r, col_idx]] = col_means[col_idx];
                                continue;
                            };
                            let goes_left = (surr_val <= surr.threshold) == surr.send_left;
                            x_imputed[[r, col_idx]] =
                                if goes_left { left_mean } else { right_mean };
                        }
                    }
                    None => {
                        // No good surrogate — fall back to column mean
                        for &r in &missing_rows {
                            x_imputed[[r, col_idx]] = col_means[col_idx];
                        }
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
            surrogate_splits: Vec::new(),
            majority_direction: true,
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
            surrogate_splits: Vec::new(),
            majority_direction: true,
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
        if len.is_multiple_of(2) {
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
