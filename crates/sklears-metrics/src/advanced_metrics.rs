//! Advanced classification metrics
//!
//! This module contains specialized metrics for advanced classification scenarios
//! including cost-sensitive learning, hierarchical classification, and other
//! domain-specific evaluation metrics.

use crate::{MetricsError, MetricsResult};
use scirs2_core::ndarray::{Array1, Array2};
use std::collections::{HashMap, HashSet};

/// Cost-sensitive accuracy
///
/// Computes accuracy with custom cost matrix, where different types of
/// misclassifications have different costs. The cost matrix should have
/// shape (n_classes, n_classes) where element [i, j] represents the cost
/// of predicting class j when the true class is i.
///
/// # Arguments
/// * `y_true` - True class labels
/// * `y_pred` - Predicted class labels
/// * `cost_matrix` - Cost matrix (n_classes x n_classes)
/// * `labels` - Optional list of class labels (if None, inferred from data)
///
/// # Returns
/// Cost-sensitive accuracy score (higher is better)
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::{array, Array2};
/// use sklears_metrics::advanced_metrics::cost_sensitive_accuracy;
///
/// let y_true = array![0, 1, 0, 1];
/// let y_pred = array![0, 1, 1, 0];
/// // Cost matrix: no cost for correct, high cost for specific errors
/// let cost_matrix = Array2::from_shape_vec((2, 2), vec![0.0, 2.0, 1.0, 0.0]).unwrap();
/// let accuracy = cost_sensitive_accuracy(&y_true, &y_pred, &cost_matrix, None).unwrap();
/// println!("Cost-sensitive accuracy: {:.3}", accuracy);
/// ```
pub fn cost_sensitive_accuracy<T: PartialEq + Copy + std::hash::Hash + Eq + Ord>(
    y_true: &Array1<T>,
    y_pred: &Array1<T>,
    cost_matrix: &Array2<f64>,
    labels: Option<&[T]>,
) -> MetricsResult<f64> {
    if y_true.len() != y_pred.len() {
        return Err(MetricsError::ShapeMismatch {
            expected: vec![y_true.len()],
            actual: vec![y_pred.len()],
        });
    }

    if y_true.is_empty() {
        return Err(MetricsError::EmptyInput);
    }

    let unique_labels: Vec<T> = if let Some(lbls) = labels {
        lbls.to_vec()
    } else {
        let mut labels_set = HashSet::new();
        for &label in y_true.iter().chain(y_pred.iter()) {
            labels_set.insert(label);
        }
        let mut labels_vec: Vec<T> = labels_set.into_iter().collect();
        labels_vec.sort();
        labels_vec
    };

    let n_classes = unique_labels.len();
    if cost_matrix.shape() != [n_classes, n_classes] {
        return Err(MetricsError::InvalidParameter(format!(
            "Cost matrix shape {:?} does not match number of classes {}",
            cost_matrix.shape(),
            n_classes
        )));
    }

    let label_to_index: HashMap<T, usize> = unique_labels
        .iter()
        .enumerate()
        .map(|(i, &label)| (label, i))
        .collect();

    let total_cost: f64 = y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(&true_label, &pred_label)| {
            let true_idx = label_to_index[&true_label];
            let pred_idx = label_to_index[&pred_label];
            cost_matrix[[true_idx, pred_idx]]
        })
        .sum();

    // Calculate minimum possible cost (if all predictions were perfect)
    let min_cost: f64 = y_true
        .iter()
        .map(|&true_label| {
            let true_idx = label_to_index[&true_label];
            cost_matrix[[true_idx, true_idx]]
        })
        .sum();

    // Calculate maximum possible cost (worst possible predictions)
    let max_cost: f64 = y_true
        .iter()
        .map(|&true_label| {
            let true_idx = label_to_index[&true_label];
            cost_matrix
                .row(true_idx)
                .iter()
                .fold(f64::NEG_INFINITY, |a, &b| a.max(b))
        })
        .sum();

    if (max_cost - min_cost).abs() < f64::EPSILON {
        return Ok(1.0); // All costs are the same
    }

    Ok(1.0 - (total_cost - min_cost) / (max_cost - min_cost))
}

/// Cost-sensitive loss
///
/// Computes the total cost of predictions using a cost matrix. Unlike
/// cost_sensitive_accuracy, this returns the raw cost value.
///
/// # Arguments
/// * `y_true` - True class labels
/// * `y_pred` - Predicted class labels
/// * `cost_matrix` - Cost matrix (n_classes x n_classes)
/// * `labels` - Optional list of class labels
/// * `normalize` - Whether to normalize by number of samples
///
/// # Returns
/// Total cost of misclassifications (lower is better)
pub fn cost_sensitive_loss<T: PartialEq + Copy + std::hash::Hash + Eq + Ord>(
    y_true: &Array1<T>,
    y_pred: &Array1<T>,
    cost_matrix: &Array2<f64>,
    labels: Option<&[T]>,
    normalize: bool,
) -> MetricsResult<f64> {
    if y_true.len() != y_pred.len() {
        return Err(MetricsError::ShapeMismatch {
            expected: vec![y_true.len()],
            actual: vec![y_pred.len()],
        });
    }

    if y_true.is_empty() {
        return Err(MetricsError::EmptyInput);
    }

    let unique_labels: Vec<T> = if let Some(lbls) = labels {
        lbls.to_vec()
    } else {
        let mut labels_set = HashSet::new();
        for &label in y_true.iter().chain(y_pred.iter()) {
            labels_set.insert(label);
        }
        let mut labels_vec: Vec<T> = labels_set.into_iter().collect();
        labels_vec.sort();
        labels_vec
    };

    let n_classes = unique_labels.len();
    if cost_matrix.shape() != [n_classes, n_classes] {
        return Err(MetricsError::InvalidParameter(format!(
            "Cost matrix shape {:?} does not match number of classes {}",
            cost_matrix.shape(),
            n_classes
        )));
    }

    let label_to_index: HashMap<T, usize> = unique_labels
        .iter()
        .enumerate()
        .map(|(i, &label)| (label, i))
        .collect();

    let total_cost: f64 = y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(&true_label, &pred_label)| {
            let true_idx = label_to_index[&true_label];
            let pred_idx = label_to_index[&pred_label];
            cost_matrix[[true_idx, pred_idx]]
        })
        .sum();

    if normalize {
        Ok(total_cost / y_true.len() as f64)
    } else {
        Ok(total_cost)
    }
}

/// Expected cost calculation
///
/// Computes the expected cost of predictions based on predicted probabilities
/// and a cost matrix. This is useful for decision-making under uncertainty.
///
/// # Arguments
/// * `y_true` - True class labels
/// * `y_proba` - Predicted probabilities (n_samples x n_classes)
/// * `cost_matrix` - Cost matrix (n_classes x n_classes)
/// * `labels` - Optional list of class labels
/// * `normalize` - Whether to normalize by number of samples
///
/// # Returns
/// Expected cost based on probability predictions
pub fn expected_cost<T: PartialEq + Copy + std::hash::Hash + Eq + Ord>(
    y_true: &Array1<T>,
    y_proba: &Array2<f64>,
    cost_matrix: &Array2<f64>,
    labels: Option<&[T]>,
    normalize: bool,
) -> MetricsResult<f64> {
    if y_true.len() != y_proba.nrows() {
        return Err(MetricsError::ShapeMismatch {
            expected: vec![y_true.len()],
            actual: vec![y_proba.nrows()],
        });
    }

    if y_true.is_empty() {
        return Err(MetricsError::EmptyInput);
    }

    let unique_labels: Vec<T> = if let Some(lbls) = labels {
        lbls.to_vec()
    } else {
        let mut labels_set = HashSet::new();
        for &label in y_true.iter() {
            labels_set.insert(label);
        }
        let mut labels_vec: Vec<T> = labels_set.into_iter().collect();
        labels_vec.sort();
        labels_vec
    };

    let n_classes = unique_labels.len();
    if cost_matrix.shape() != [n_classes, n_classes] || y_proba.ncols() != n_classes {
        return Err(MetricsError::InvalidParameter(
            "Cost matrix and probability dimensions do not match".to_string(),
        ));
    }

    let label_to_index: HashMap<T, usize> = unique_labels
        .iter()
        .enumerate()
        .map(|(i, &label)| (label, i))
        .collect();

    let mut total_expected_cost = 0.0;

    for (i, &true_label) in y_true.iter().enumerate() {
        let true_idx = label_to_index[&true_label];
        let proba_row = y_proba.row(i);

        // Calculate expected cost for this sample
        let expected_cost_sample: f64 = (0..n_classes)
            .map(|pred_idx| proba_row[pred_idx] * cost_matrix[[true_idx, pred_idx]])
            .sum();

        total_expected_cost += expected_cost_sample;
    }

    if normalize {
        Ok(total_expected_cost / y_true.len() as f64)
    } else {
        Ok(total_expected_cost)
    }
}

/// Hierarchical classification metrics for tree-structured label spaces
/// Represents a node in the hierarchical label tree
#[derive(Debug, Clone, PartialEq)]
pub struct HierarchicalNode {
    pub label: String,
    pub children: Vec<HierarchicalNode>,
}

impl HierarchicalNode {
    /// Create a new hierarchical node
    pub fn new(label: String) -> Self {
        Self {
            label,
            children: Vec::new(),
        }
    }

    /// Add a child node to this node
    pub fn add_child(&mut self, child: HierarchicalNode) {
        self.children.push(child);
    }

    /// Get all ancestors of a given label in the hierarchy
    pub fn get_ancestors(&self, target_label: &str) -> Vec<String> {
        let mut ancestors = Vec::new();
        if self.find_ancestors_recursive(target_label, &mut ancestors) {
            ancestors.reverse(); // Return from root to target
        }
        ancestors
    }

    fn find_ancestors_recursive(&self, target_label: &str, ancestors: &mut Vec<String>) -> bool {
        if self.label == target_label {
            ancestors.push(self.label.clone());
            return true;
        }

        for child in &self.children {
            if child.find_ancestors_recursive(target_label, ancestors) {
                ancestors.push(self.label.clone());
                return true;
            }
        }
        false
    }

    /// Check if one label is an ancestor of another
    pub fn is_ancestor(&self, ancestor: &str, descendant: &str) -> bool {
        let ancestors = self.get_ancestors(descendant);
        ancestors.contains(&ancestor.to_string())
    }

    /// Calculate the distance between two labels in the hierarchy
    pub fn label_distance(&self, label1: &str, label2: &str) -> Option<usize> {
        let ancestors1 = self.get_ancestors(label1);
        let ancestors2 = self.get_ancestors(label2);

        if ancestors1.is_empty() || ancestors2.is_empty() {
            return None;
        }

        // Find lowest common ancestor
        let mut lca_depth = 0;
        for (a1, a2) in ancestors1.iter().zip(ancestors2.iter()) {
            if a1 == a2 {
                lca_depth += 1;
            } else {
                break;
            }
        }

        // Distance is sum of depths from LCA to each label
        let dist1 = ancestors1.len() - lca_depth;
        let dist2 = ancestors2.len() - lca_depth;
        Some(dist1 + dist2)
    }
}

/// Hierarchical Precision - considers partial credit for predictions at different levels
///
/// # Arguments
/// * `y_true` - True hierarchical labels
/// * `y_pred` - Predicted hierarchical labels
/// * `hierarchy` - Tree structure of the label hierarchy
///
/// # Returns
/// Hierarchical precision score (0.0 to 1.0)
pub fn hierarchical_precision(
    y_true: &Array1<String>,
    y_pred: &Array1<String>,
    hierarchy: &HierarchicalNode,
) -> MetricsResult<f64> {
    if y_true.len() != y_pred.len() {
        return Err(MetricsError::ShapeMismatch {
            expected: vec![y_true.len()],
            actual: vec![y_pred.len()],
        });
    }

    if y_true.is_empty() {
        return Err(MetricsError::EmptyInput);
    }

    let mut total_precision = 0.0;
    let mut count = 0;

    for (true_label, pred_label) in y_true.iter().zip(y_pred.iter()) {
        let true_ancestors = hierarchy.get_ancestors(true_label);
        let pred_ancestors = hierarchy.get_ancestors(pred_label);

        if !pred_ancestors.is_empty() {
            // Count how many predicted ancestors are correct
            let correct_predictions = pred_ancestors
                .iter()
                .filter(|&pred_anc| true_ancestors.contains(pred_anc))
                .count();

            total_precision += correct_predictions as f64 / pred_ancestors.len() as f64;
            count += 1;
        }
    }

    if count == 0 {
        return Err(MetricsError::DivisionByZero);
    }

    Ok(total_precision / count as f64)
}

/// Hierarchical Recall - considers partial credit for recall at different levels
///
/// # Arguments
/// * `y_true` - True hierarchical labels
/// * `y_pred` - Predicted hierarchical labels
/// * `hierarchy` - Tree structure of the label hierarchy
///
/// # Returns
/// Hierarchical recall score (0.0 to 1.0)
pub fn hierarchical_recall(
    y_true: &Array1<String>,
    y_pred: &Array1<String>,
    hierarchy: &HierarchicalNode,
) -> MetricsResult<f64> {
    if y_true.len() != y_pred.len() {
        return Err(MetricsError::ShapeMismatch {
            expected: vec![y_true.len()],
            actual: vec![y_pred.len()],
        });
    }

    if y_true.is_empty() {
        return Err(MetricsError::EmptyInput);
    }

    let mut total_recall = 0.0;
    let mut count = 0;

    for (true_label, pred_label) in y_true.iter().zip(y_pred.iter()) {
        let true_ancestors = hierarchy.get_ancestors(true_label);
        let pred_ancestors = hierarchy.get_ancestors(pred_label);

        if !true_ancestors.is_empty() {
            // Count how many true ancestors are predicted correctly
            let correct_predictions = true_ancestors
                .iter()
                .filter(|&true_anc| pred_ancestors.contains(true_anc))
                .count();

            total_recall += correct_predictions as f64 / true_ancestors.len() as f64;
            count += 1;
        }
    }

    if count == 0 {
        return Err(MetricsError::DivisionByZero);
    }

    Ok(total_recall / count as f64)
}

/// Hierarchical F1-Score - harmonic mean of hierarchical precision and recall
///
/// # Arguments
/// * `y_true` - True hierarchical labels
/// * `y_pred` - Predicted hierarchical labels
/// * `hierarchy` - Tree structure of the label hierarchy
///
/// # Returns
/// Hierarchical F1 score (0.0 to 1.0)
pub fn hierarchical_f1_score(
    y_true: &Array1<String>,
    y_pred: &Array1<String>,
    hierarchy: &HierarchicalNode,
) -> MetricsResult<f64> {
    let precision = hierarchical_precision(y_true, y_pred, hierarchy)?;
    let recall = hierarchical_recall(y_true, y_pred, hierarchy)?;

    if precision + recall == 0.0 {
        return Ok(0.0);
    }

    Ok(2.0 * precision * recall / (precision + recall))
}

/// Tree Distance Loss - penalizes misclassifications based on tree distance
///
/// # Arguments
/// * `y_true` - True hierarchical labels
/// * `y_pred` - Predicted hierarchical labels
/// * `hierarchy` - Tree structure of the label hierarchy
///
/// # Returns
/// Average tree distance loss
pub fn tree_distance_loss(
    y_true: &Array1<String>,
    y_pred: &Array1<String>,
    hierarchy: &HierarchicalNode,
) -> MetricsResult<f64> {
    if y_true.len() != y_pred.len() {
        return Err(MetricsError::ShapeMismatch {
            expected: vec![y_true.len()],
            actual: vec![y_pred.len()],
        });
    }

    if y_true.is_empty() {
        return Err(MetricsError::EmptyInput);
    }

    let total_distance: usize = y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(true_label, pred_label)| {
            hierarchy
                .label_distance(true_label, pred_label)
                .unwrap_or(0)
        })
        .sum();

    Ok(total_distance as f64 / y_true.len() as f64)
}

/// Hierarchical Classification Accuracy - exact match considering hierarchy
///
/// # Arguments
/// * `y_true` - True hierarchical labels
/// * `y_pred` - Predicted hierarchical labels
/// * `hierarchy` - Tree structure of the label hierarchy
/// * `partial_credit` - Whether to give partial credit for ancestor matches
///
/// # Returns
/// Hierarchical accuracy score (0.0 to 1.0)
pub fn hierarchical_accuracy(
    y_true: &Array1<String>,
    y_pred: &Array1<String>,
    hierarchy: &HierarchicalNode,
    partial_credit: bool,
) -> MetricsResult<f64> {
    if y_true.len() != y_pred.len() {
        return Err(MetricsError::ShapeMismatch {
            expected: vec![y_true.len()],
            actual: vec![y_pred.len()],
        });
    }

    if y_true.is_empty() {
        return Err(MetricsError::EmptyInput);
    }

    if !partial_credit {
        // Standard exact match accuracy
        let correct = y_true
            .iter()
            .zip(y_pred.iter())
            .filter(|(t, p)| t == p)
            .count();
        return Ok(correct as f64 / y_true.len() as f64);
    }

    // Partial credit based on shared ancestors
    let mut total_score = 0.0;
    for (true_label, pred_label) in y_true.iter().zip(y_pred.iter()) {
        if true_label == pred_label {
            total_score += 1.0;
        } else {
            let true_ancestors = hierarchy.get_ancestors(true_label);
            let pred_ancestors = hierarchy.get_ancestors(pred_label);

            if !true_ancestors.is_empty() && !pred_ancestors.is_empty() {
                // Count shared ancestors
                let shared_ancestors = true_ancestors
                    .iter()
                    .filter(|&anc| pred_ancestors.contains(anc))
                    .count();

                // Give partial credit based on fraction of shared ancestors
                let max_ancestors = true_ancestors.len().max(pred_ancestors.len());
                total_score += shared_ancestors as f64 / max_ancestors as f64;
            }
        }
    }

    Ok(total_score / y_true.len() as f64)
}
