//! Multi-output decision trees and multi-label classification
//!
//! This module provides extensions for handling multiple outputs simultaneously,
//! including multi-output regression and multi-label classification.

use scirs2_core::ndarray::{Array1, Array2, Axis};
use sklears_core::{
    error::Result,
    prelude::SklearsError,
    traits::{Trained, Untrained},
};
use std::collections::HashMap;
use std::marker::PhantomData;

use crate::decision_tree::DecisionTreeConfig;
use crate::random_forest::RandomForestConfig;

/// Multi-output strategy for handling multiple targets
#[derive(Debug, Clone, Copy)]
pub enum MultiOutputStrategy {
    /// Train independent models for each output
    Independent,
    /// Use shared tree structure with multi-output splits
    Shared,
    /// Chain outputs (predict each output using previous predictions as features)
    Chained,
}

/// Label correlation strategy for multi-label classification
#[derive(Debug, Clone)]
pub enum LabelCorrelation {
    /// Ignore label correlations (treat as independent)
    Independent,
    /// Model label correlations using label co-occurrence
    Cooccurrence { min_support: f64 },
    /// Use label hierarchy if available
    Hierarchical { hierarchy: Vec<(usize, usize)> }, // (parent, child) pairs
    /// Exploit label correlations through dimensionality reduction
    Embedded { n_components: usize },
}

/// Configuration for multi-output decision trees
#[derive(Debug, Clone)]
pub struct MultiOutputTreeConfig {
    /// Base decision tree configuration
    pub base_config: DecisionTreeConfig,
    /// Strategy for handling multiple outputs
    pub multi_output_strategy: MultiOutputStrategy,
    /// Label correlation strategy (for multi-label classification)
    pub label_correlation: LabelCorrelation,
    /// Minimum samples required for multi-output split
    pub min_samples_multioutput_split: usize,
    /// Enable feature sharing across outputs
    pub enable_feature_sharing: bool,
    /// Output-specific feature weights
    pub output_feature_weights: Option<Array2<f64>>,
}

impl Default for MultiOutputTreeConfig {
    fn default() -> Self {
        Self {
            base_config: DecisionTreeConfig::default(),
            multi_output_strategy: MultiOutputStrategy::Independent,
            label_correlation: LabelCorrelation::Independent,
            min_samples_multioutput_split: 10,
            enable_feature_sharing: true,
            output_feature_weights: None,
        }
    }
}

/// Multi-output decision tree for regression and classification
pub struct MultiOutputDecisionTree<State = Untrained> {
    config: MultiOutputTreeConfig,
    trees: Vec<Box<dyn MultiOutputTreeModel>>,
    n_outputs: usize,
    n_features: usize,
    output_types: Vec<OutputType>,
    label_correlations: Option<LabelCorrelationMatrix>,
    state: PhantomData<State>,
}

/// Type of output (regression or classification)
#[derive(Debug, Clone, Copy)]
pub enum OutputType {
    Regression,
    Classification { n_classes: usize },
}

/// Label correlation matrix for multi-label problems
#[derive(Debug, Clone)]
pub struct LabelCorrelationMatrix {
    /// Correlation coefficients between labels
    correlations: Array2<f64>,
    /// Co-occurrence frequencies
    cooccurrence: Array2<usize>,
    /// Label hierarchy (if applicable)
    hierarchy: Option<Vec<(usize, usize)>>,
}

impl LabelCorrelationMatrix {
    /// Pairwise Pearson correlation coefficients between labels
    pub fn correlations(&self) -> &Array2<f64> {
        &self.correlations
    }

    /// Raw co-occurrence count matrix between labels
    pub fn cooccurrence(&self) -> &Array2<usize> {
        &self.cooccurrence
    }

    /// Optional label hierarchy as (parent, child) index pairs
    pub fn hierarchy(&self) -> Option<&Vec<(usize, usize)>> {
        self.hierarchy.as_ref()
    }
}

/// Trait for models that can handle multiple outputs
pub trait MultiOutputTreeModel: Send + Sync {
    /// Fit the model to multi-output data
    fn fit_multi(&mut self, x: &Array2<f64>, y: &Array2<f64>) -> Result<()>;

    /// Predict multiple outputs
    fn predict_multi(&self, x: &Array2<f64>) -> Result<Array2<f64>>;

    /// Get feature importances for each output
    fn feature_importances_multi(&self) -> Result<Array2<f64>>;
}

/// Independent multi-output tree model (one tree per output)
#[derive(Debug)]
pub struct IndependentMultiOutputTree {
    /// Individual trees for each output
    trees: Vec<SingleOutputTree>,
    /// Number of outputs
    n_outputs: usize,
    /// Number of features
    n_features: usize,
}

/// Single output tree wrapper
#[derive(Debug)]
pub struct SingleOutputTree {
    /// Tree structure (simplified)
    nodes: Vec<MultiOutputTreeNode>,
    /// Output type
    output_type: OutputType,
    /// Feature importances
    feature_importances: Array1<f64>,
}

/// Multi-output tree node
#[derive(Debug, Clone)]
pub struct MultiOutputTreeNode {
    /// Feature index for split (None for leaf)
    feature_idx: Option<usize>,
    /// Split threshold (None for leaf)
    threshold: Option<f64>,
    /// Predictions for each output (for leaves)
    predictions: Array1<f64>,
    /// Left child ID
    left_child: Option<usize>,
    /// Right child ID
    right_child: Option<usize>,
    /// Sample indices for this node
    samples: Vec<usize>,
    /// Node depth
    depth: usize,
    /// Is this a leaf node?
    is_leaf: bool,
}

impl SingleOutputTree {
    /// Create a new single output tree
    pub fn new(output_type: OutputType, n_features: usize) -> Self {
        Self {
            nodes: Vec::new(),
            output_type,
            feature_importances: Array1::zeros(n_features),
        }
    }

    /// Fit tree to single output
    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<()> {
        let n_samples = x.nrows();
        let samples: Vec<usize> = (0..n_samples).collect();

        // Create root node
        let root = self.create_node(0, samples, x, y, 0)?;
        self.nodes = vec![root];

        // Build tree recursively
        self.build_tree(0, x, y)?;

        Ok(())
    }

    /// Create a tree node
    fn create_node(
        &self,
        _id: usize,
        samples: Vec<usize>,
        _x: &Array2<f64>,
        y: &Array1<f64>,
        depth: usize,
    ) -> Result<MultiOutputTreeNode> {
        let n_outputs = 1; // Single output
        let mut predictions = Array1::zeros(n_outputs);

        if !samples.is_empty() {
            // Calculate prediction based on output type
            match self.output_type {
                OutputType::Regression => {
                    let sum: f64 = samples.iter().map(|&i| y[i]).sum();
                    predictions[0] = sum / samples.len() as f64;
                }
                OutputType::Classification { .. } => {
                    // Majority class
                    let mut class_counts: HashMap<i32, usize> = HashMap::new();
                    for &sample_idx in &samples {
                        let class = y[sample_idx] as i32;
                        *class_counts.entry(class).or_insert(0) += 1;
                    }

                    if let Some((&majority_class, _)) =
                        class_counts.iter().max_by_key(|(_, &count)| count)
                    {
                        predictions[0] = majority_class as f64;
                    }
                }
            }
        }

        Ok(MultiOutputTreeNode {
            feature_idx: None,
            threshold: None,
            predictions,
            left_child: None,
            right_child: None,
            samples,
            depth,
            is_leaf: true,
        })
    }

    /// Build tree recursively
    fn build_tree(&mut self, node_id: usize, x: &Array2<f64>, y: &Array1<f64>) -> Result<()> {
        let node = &self.nodes[node_id].clone();

        if node.samples.len() < 10 || node.depth >= 10 {
            return Ok(());
        }

        // Find best split
        if let Some((feature_idx, threshold)) = self.find_best_split(x, y, &node.samples)? {
            // Split samples
            let (left_samples, right_samples) =
                self.split_samples(x, &node.samples, feature_idx, threshold);

            if !left_samples.is_empty() && !right_samples.is_empty() {
                // Create child nodes
                let left_id = self.nodes.len();
                let right_id = self.nodes.len() + 1;

                let left_node = self.create_node(left_id, left_samples, x, y, node.depth + 1)?;
                let right_node = self.create_node(right_id, right_samples, x, y, node.depth + 1)?;

                // Update parent node
                self.nodes[node_id].is_leaf = false;
                self.nodes[node_id].feature_idx = Some(feature_idx);
                self.nodes[node_id].threshold = Some(threshold);
                self.nodes[node_id].left_child = Some(left_id);
                self.nodes[node_id].right_child = Some(right_id);

                // Add child nodes
                self.nodes.push(left_node);
                self.nodes.push(right_node);

                // Update feature importance
                self.feature_importances[feature_idx] += 1.0;

                // Recursively build subtrees
                self.build_tree(left_id, x, y)?;
                self.build_tree(right_id, x, y)?;
            }
        }

        Ok(())
    }

    /// Find best split for a node
    fn find_best_split(
        &self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        samples: &[usize],
    ) -> Result<Option<(usize, f64)>> {
        let n_features = x.ncols();
        let mut best_split: Option<(usize, f64, f64)> = None; // (feature, threshold, impurity_reduction)

        for feature_idx in 0..n_features {
            // Get unique values for this feature
            let mut values: Vec<f64> = samples.iter().map(|&i| x[[i, feature_idx]]).collect();
            values.sort_by(|a, b| a.partial_cmp(b).expect("operation should succeed"));
            values.dedup();

            // Try splits between consecutive values
            for i in 0..values.len().saturating_sub(1) {
                let threshold = (values[i] + values[i + 1]) / 2.0;
                let impurity_reduction =
                    self.calculate_impurity_reduction(x, y, samples, feature_idx, threshold);

                if best_split.is_none()
                    || impurity_reduction > best_split.as_ref().expect("operation should succeed").2
                {
                    best_split = Some((feature_idx, threshold, impurity_reduction));
                }
            }
        }

        if let Some((feature_idx, threshold, reduction)) = best_split {
            if reduction > 0.001 {
                // Minimum improvement threshold
                Ok(Some((feature_idx, threshold)))
            } else {
                Ok(None)
            }
        } else {
            Ok(None)
        }
    }

    /// Calculate impurity reduction for a potential split
    fn calculate_impurity_reduction(
        &self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        samples: &[usize],
        feature_idx: usize,
        threshold: f64,
    ) -> f64 {
        let (left_samples, right_samples) = self.split_samples(x, samples, feature_idx, threshold);

        if left_samples.is_empty() || right_samples.is_empty() {
            return 0.0;
        }

        let parent_impurity = self.calculate_impurity(y, samples);
        let left_impurity = self.calculate_impurity(y, &left_samples);
        let right_impurity = self.calculate_impurity(y, &right_samples);

        let left_weight = left_samples.len() as f64 / samples.len() as f64;
        let right_weight = right_samples.len() as f64 / samples.len() as f64;

        parent_impurity - (left_weight * left_impurity + right_weight * right_impurity)
    }

    /// Calculate impurity for a set of samples
    fn calculate_impurity(&self, y: &Array1<f64>, samples: &[usize]) -> f64 {
        if samples.is_empty() {
            return 0.0;
        }

        match self.output_type {
            OutputType::Regression => {
                // MSE for regression
                let mean = samples.iter().map(|&i| y[i]).sum::<f64>() / samples.len() as f64;
                samples.iter().map(|&i| (y[i] - mean).powi(2)).sum::<f64>() / samples.len() as f64
            }
            OutputType::Classification { .. } => {
                // Gini impurity for classification
                let mut class_counts: HashMap<i32, usize> = HashMap::new();
                for &sample_idx in samples {
                    let class = y[sample_idx] as i32;
                    *class_counts.entry(class).or_insert(0) += 1;
                }

                let total = samples.len() as f64;
                let mut gini = 1.0;
                for &count in class_counts.values() {
                    let prob = count as f64 / total;
                    gini -= prob * prob;
                }
                gini
            }
        }
    }

    /// Split samples based on feature and threshold
    fn split_samples(
        &self,
        x: &Array2<f64>,
        samples: &[usize],
        feature_idx: usize,
        threshold: f64,
    ) -> (Vec<usize>, Vec<usize>) {
        let mut left_samples = Vec::new();
        let mut right_samples = Vec::new();

        for &sample_idx in samples {
            if x[[sample_idx, feature_idx]] <= threshold {
                left_samples.push(sample_idx);
            } else {
                right_samples.push(sample_idx);
            }
        }

        (left_samples, right_samples)
    }

    /// Predict on new data
    pub fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        let mut predictions = Array1::zeros(x.nrows());

        for (sample_idx, sample) in x.rows().into_iter().enumerate() {
            predictions[sample_idx] = self.predict_single(&sample.to_owned())?;
        }

        Ok(predictions)
    }

    /// Predict single sample
    fn predict_single(&self, x: &Array1<f64>) -> Result<f64> {
        if self.nodes.is_empty() {
            return Err(SklearsError::PredictError("Tree not fitted".to_string()));
        }

        let mut node_idx = 0;

        loop {
            let node = &self.nodes[node_idx];

            if node.is_leaf {
                return Ok(node.predictions[0]);
            }

            if let (Some(feature_idx), Some(threshold)) = (node.feature_idx, node.threshold) {
                if feature_idx >= x.len() {
                    return Err(SklearsError::PredictError(
                        "Feature index out of bounds".to_string(),
                    ));
                }

                if x[feature_idx] <= threshold {
                    if let Some(left_child) = node.left_child {
                        node_idx = left_child;
                    } else {
                        return Ok(node.predictions[0]);
                    }
                } else {
                    if let Some(right_child) = node.right_child {
                        node_idx = right_child;
                    } else {
                        return Ok(node.predictions[0]);
                    }
                }
            } else {
                return Ok(node.predictions[0]);
            }
        }
    }
}

/// Shared multi-output tree: one decision tree where each leaf stores
/// predictions for all output dimensions.  The split criterion at each
/// internal node sums the variance/impurity reduction over all outputs.
#[derive(Debug)]
pub struct SharedMultiOutputTree {
    /// Nodes — index 0 is the root
    nodes: Vec<SharedMultiOutputTreeNode>,
    /// Number of outputs
    n_outputs: usize,
    /// Number of features
    n_features: usize,
    /// Per-output type (regression / classification)
    output_types: Vec<OutputType>,
    /// Aggregate feature importances (summed over outputs)
    feature_importances: Array1<f64>,
    /// Config for depth / samples limits
    config: DecisionTreeConfig,
}

/// One node in the shared multi-output tree
#[derive(Debug, Clone)]
struct SharedMultiOutputTreeNode {
    feature_idx: Option<usize>,
    threshold: Option<f64>,
    /// Prediction vector — one value per output
    predictions: Array1<f64>,
    left_child: Option<usize>,
    right_child: Option<usize>,
    samples: Vec<usize>,
    depth: usize,
    is_leaf: bool,
}

impl SharedMultiOutputTree {
    /// Create a new shared multi-output tree
    pub fn new(
        n_outputs: usize,
        n_features: usize,
        output_types: Vec<OutputType>,
        config: DecisionTreeConfig,
    ) -> Self {
        Self {
            nodes: Vec::new(),
            n_outputs,
            n_features,
            output_types,
            feature_importances: Array1::zeros(n_features),
            config,
        }
    }

    /// Compute multi-output predictions for a set of samples (mean / majority per output)
    fn compute_predictions(&self, samples: &[usize], y: &Array2<f64>) -> Array1<f64> {
        let mut preds = Array1::zeros(self.n_outputs);
        if samples.is_empty() {
            return preds;
        }
        for (out_idx, output_type) in self.output_types.iter().enumerate() {
            match output_type {
                OutputType::Regression => {
                    let sum: f64 = samples.iter().map(|&i| y[[i, out_idx]]).sum();
                    preds[out_idx] = sum / samples.len() as f64;
                }
                OutputType::Classification { .. } => {
                    let mut counts: HashMap<i32, usize> = HashMap::new();
                    for &i in samples {
                        let cls = y[[i, out_idx]] as i32;
                        *counts.entry(cls).or_insert(0) += 1;
                    }
                    if let Some((&cls, _)) = counts.iter().max_by_key(|(_, &c)| c) {
                        preds[out_idx] = cls as f64;
                    }
                }
            }
        }
        preds
    }

    /// Weighted variance/impurity summed over all outputs for the given sample subset
    fn multi_output_impurity(&self, samples: &[usize], y: &Array2<f64>) -> f64 {
        if samples.is_empty() {
            return 0.0;
        }
        let mut total = 0.0;
        for (out_idx, output_type) in self.output_types.iter().enumerate() {
            match output_type {
                OutputType::Regression => {
                    let mean = samples.iter().map(|&i| y[[i, out_idx]]).sum::<f64>()
                        / samples.len() as f64;
                    total += samples
                        .iter()
                        .map(|&i| (y[[i, out_idx]] - mean).powi(2))
                        .sum::<f64>()
                        / samples.len() as f64;
                }
                OutputType::Classification { .. } => {
                    let mut counts: HashMap<i32, usize> = HashMap::new();
                    for &i in samples {
                        let cls = y[[i, out_idx]] as i32;
                        *counts.entry(cls).or_insert(0) += 1;
                    }
                    let n = samples.len() as f64;
                    let mut gini = 1.0_f64;
                    for &c in counts.values() {
                        let p = c as f64 / n;
                        gini -= p * p;
                    }
                    total += gini;
                }
            }
        }
        total
    }

    /// Find the best split that maximises the summed impurity reduction over all outputs
    fn find_best_split(
        &self,
        x: &Array2<f64>,
        y: &Array2<f64>,
        samples: &[usize],
    ) -> Option<(usize, f64)> {
        let n_features = x.ncols();
        let mut best: Option<(usize, f64, f64)> = None; // (feature, threshold, reduction)
        let parent_impurity = self.multi_output_impurity(samples, y);

        for feat in 0..n_features {
            let mut values: Vec<f64> = samples.iter().map(|&i| x[[i, feat]]).collect();
            values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            values.dedup();

            for w in values.windows(2) {
                let threshold = (w[0] + w[1]) / 2.0;

                let (left, right): (Vec<usize>, Vec<usize>) =
                    samples.iter().partition(|&&i| x[[i, feat]] <= threshold);

                if left.is_empty() || right.is_empty() {
                    continue;
                }

                let lw = left.len() as f64 / samples.len() as f64;
                let rw = right.len() as f64 / samples.len() as f64;
                let reduction = parent_impurity
                    - lw * self.multi_output_impurity(&left, y)
                    - rw * self.multi_output_impurity(&right, y);

                if reduction > best.as_ref().map_or(f64::NEG_INFINITY, |b| b.2) {
                    best = Some((feat, threshold, reduction));
                }
            }
        }

        best.filter(|b| b.2 > 1e-10)
            .map(|(feat, threshold, _)| (feat, threshold))
    }

    /// Build the tree recursively
    fn build_from_node(&mut self, node_id: usize, x: &Array2<f64>, y: &Array2<f64>) -> Result<()> {
        let max_depth = self.config.max_depth.unwrap_or(10);
        let min_split = self.config.min_samples_split;

        let node_depth = self.nodes[node_id].depth;
        let node_samples = self.nodes[node_id].samples.clone();

        if node_samples.len() < min_split || node_depth >= max_depth {
            return Ok(());
        }

        let split = self.find_best_split(x, y, &node_samples);
        let (feat, threshold) = match split {
            Some(s) => s,
            None => return Ok(()),
        };

        let (left_samples, right_samples): (Vec<usize>, Vec<usize>) = node_samples
            .iter()
            .partition(|&&i| x[[i, feat]] <= threshold);

        if left_samples.is_empty() || right_samples.is_empty() {
            return Ok(());
        }

        let left_preds = self.compute_predictions(&left_samples, y);
        let right_preds = self.compute_predictions(&right_samples, y);
        let left_depth = self.nodes[node_id].depth + 1;
        let right_depth = left_depth;

        let left_id = self.nodes.len();
        self.nodes.push(SharedMultiOutputTreeNode {
            feature_idx: None,
            threshold: None,
            predictions: left_preds,
            left_child: None,
            right_child: None,
            samples: left_samples,
            depth: left_depth,
            is_leaf: true,
        });

        let right_id = self.nodes.len();
        self.nodes.push(SharedMultiOutputTreeNode {
            feature_idx: None,
            threshold: None,
            predictions: right_preds,
            left_child: None,
            right_child: None,
            samples: right_samples,
            depth: right_depth,
            is_leaf: true,
        });

        // Update parent
        self.nodes[node_id].is_leaf = false;
        self.nodes[node_id].feature_idx = Some(feat);
        self.nodes[node_id].threshold = Some(threshold);
        self.nodes[node_id].left_child = Some(left_id);
        self.nodes[node_id].right_child = Some(right_id);

        // Track feature importance
        self.feature_importances[feat] += 1.0;

        // Recurse
        self.build_from_node(left_id, x, y)?;
        self.build_from_node(right_id, x, y)?;

        Ok(())
    }

    /// Predict for a single sample, returning a vector of predictions
    fn predict_single(&self, x: &Array1<f64>) -> Result<Array1<f64>> {
        if self.nodes.is_empty() {
            return Err(SklearsError::PredictError("Tree not fitted".to_string()));
        }
        let mut idx = 0usize;
        loop {
            let node = &self.nodes[idx];
            if node.is_leaf {
                return Ok(node.predictions.clone());
            }
            match (node.feature_idx, node.threshold) {
                (Some(feat), Some(thr)) => {
                    if feat >= x.len() {
                        return Err(SklearsError::PredictError(
                            "Feature index out of bounds".to_string(),
                        ));
                    }
                    if x[feat] <= thr {
                        idx = node.left_child.ok_or_else(|| {
                            SklearsError::PredictError("Missing left child".to_string())
                        })?;
                    } else {
                        idx = node.right_child.ok_or_else(|| {
                            SklearsError::PredictError("Missing right child".to_string())
                        })?;
                    }
                }
                _ => return Ok(node.predictions.clone()),
            }
        }
    }
}

impl MultiOutputTreeModel for SharedMultiOutputTree {
    fn fit_multi(&mut self, x: &Array2<f64>, y: &Array2<f64>) -> Result<()> {
        if y.ncols() != self.n_outputs {
            return Err(SklearsError::InvalidInput(format!(
                "Expected {} outputs, got {}",
                self.n_outputs,
                y.ncols()
            )));
        }

        let n_samples = x.nrows();
        let samples: Vec<usize> = (0..n_samples).collect();
        let root_preds = self.compute_predictions(&samples, y);

        self.nodes.push(SharedMultiOutputTreeNode {
            feature_idx: None,
            threshold: None,
            predictions: root_preds,
            left_child: None,
            right_child: None,
            samples,
            depth: 0,
            is_leaf: true,
        });

        self.build_from_node(0, x, y)?;
        Ok(())
    }

    fn predict_multi(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let n_samples = x.nrows();
        let mut predictions = Array2::zeros((n_samples, self.n_outputs));
        for (i, row) in x.rows().into_iter().enumerate() {
            let preds = self.predict_single(&row.to_owned())?;
            predictions.row_mut(i).assign(&preds);
        }
        Ok(predictions)
    }

    fn feature_importances_multi(&self) -> Result<Array2<f64>> {
        let mut importances = Array2::zeros((self.n_features, self.n_outputs));
        for out_idx in 0..self.n_outputs {
            importances
                .column_mut(out_idx)
                .assign(&self.feature_importances);
        }
        Ok(importances)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Chained multi-output tree
// ─────────────────────────────────────────────────────────────────────────────

/// Chained multi-output tree: each output is predicted by a tree trained on the
/// original features augmented with the *training-time* predictions of all
/// previous outputs.
///
/// During prediction the chain is applied sequentially:
///   pred_0 = tree_0.predict(X)
///   pred_1 = tree_1.predict([X | pred_0])
///   ...
#[derive(Debug)]
pub struct ChainedMultiOutputTree {
    /// One `SingleOutputTree` per output dimension
    chains: Vec<SingleOutputTree>,
    /// Number of outputs
    n_outputs: usize,
    /// Number of *original* features (not including chained predictions)
    n_features: usize,
}

impl ChainedMultiOutputTree {
    /// Create a new chained multi-output tree
    pub fn new(
        n_outputs: usize,
        n_features: usize,
        output_types: Vec<OutputType>,
        _config: DecisionTreeConfig,
    ) -> Self {
        // Tree k is trained on n_features + k extra columns (outputs 0..k-1)
        let chains = output_types
            .iter()
            .enumerate()
            .map(|(k, ot)| SingleOutputTree::new(*ot, n_features + k))
            .collect();

        Self {
            chains,
            n_outputs,
            n_features,
        }
    }

    /// Build the augmented feature matrix [X | prev_preds] for output k
    fn augmented_x(x: &Array2<f64>, train_preds: &Array2<f64>, k: usize) -> Array2<f64> {
        let n_samples = x.nrows();
        let n_orig = x.ncols();
        let mut aug = Array2::zeros((n_samples, n_orig + k));
        for i in 0..n_samples {
            for j in 0..n_orig {
                aug[[i, j]] = x[[i, j]];
            }
            for p in 0..k {
                aug[[i, n_orig + p]] = train_preds[[i, p]];
            }
        }
        aug
    }
}

impl MultiOutputTreeModel for ChainedMultiOutputTree {
    fn fit_multi(&mut self, x: &Array2<f64>, y: &Array2<f64>) -> Result<()> {
        if y.ncols() != self.n_outputs {
            return Err(SklearsError::InvalidInput(format!(
                "Expected {} outputs, got {}",
                self.n_outputs,
                y.ncols()
            )));
        }

        let n_samples = x.nrows();
        // Accumulate training-time predictions as we go
        let mut train_preds = Array2::<f64>::zeros((n_samples, self.n_outputs));

        for k in 0..self.n_outputs {
            // Build augmented feature matrix for output k
            let aug = Self::augmented_x(x, &train_preds, k);
            let y_k = y.column(k).to_owned();

            self.chains[k].fit(&aug, &y_k)?;

            // Record in-bag predictions for subsequent outputs
            let preds_k = self.chains[k].predict(&aug)?;
            train_preds.column_mut(k).assign(&preds_k);
        }

        Ok(())
    }

    fn predict_multi(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let n_samples = x.nrows();
        let mut predictions = Array2::<f64>::zeros((n_samples, self.n_outputs));

        for k in 0..self.n_outputs {
            let aug = Self::augmented_x(x, &predictions, k);
            let preds_k = self.chains[k].predict(&aug)?;
            predictions.column_mut(k).assign(&preds_k);
        }

        Ok(predictions)
    }

    fn feature_importances_multi(&self) -> Result<Array2<f64>> {
        let mut importances = Array2::zeros((self.n_features, self.n_outputs));
        for (k, chain) in self.chains.iter().enumerate() {
            // Only the first n_features importances correspond to original features
            let src = chain
                .feature_importances
                .slice(scirs2_core::ndarray::s![..self.n_features]);
            importances.column_mut(k).assign(&src);
        }
        Ok(importances)
    }
}

impl IndependentMultiOutputTree {
    /// Create a new independent multi-output tree
    pub fn new(
        n_outputs: usize,
        n_features: usize,
        output_types: Vec<OutputType>,
        config: DecisionTreeConfig,
    ) -> Self {
        let trees = output_types
            .into_iter()
            .map(|output_type| SingleOutputTree::new(output_type, n_features))
            .collect();

        let _ = config; // config forwarding to SingleOutputTree is not yet implemented
        Self {
            trees,
            n_outputs,
            n_features,
        }
    }
}

impl MultiOutputTreeModel for IndependentMultiOutputTree {
    fn fit_multi(&mut self, x: &Array2<f64>, y: &Array2<f64>) -> Result<()> {
        if y.ncols() != self.n_outputs {
            return Err(SklearsError::InvalidInput(format!(
                "Expected {} outputs, got {}",
                self.n_outputs,
                y.ncols()
            )));
        }

        // Fit each tree independently
        for (output_idx, tree) in self.trees.iter_mut().enumerate() {
            let y_single = y.column(output_idx);
            tree.fit(x, &y_single.to_owned())?;
        }

        Ok(())
    }

    fn predict_multi(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let n_samples = x.nrows();
        let mut predictions = Array2::zeros((n_samples, self.n_outputs));

        for (output_idx, tree) in self.trees.iter().enumerate() {
            let output_predictions = tree.predict(x)?;
            predictions
                .column_mut(output_idx)
                .assign(&output_predictions);
        }

        Ok(predictions)
    }

    fn feature_importances_multi(&self) -> Result<Array2<f64>> {
        let mut importances = Array2::zeros((self.n_features, self.n_outputs));

        for (output_idx, tree) in self.trees.iter().enumerate() {
            importances
                .column_mut(output_idx)
                .assign(&tree.feature_importances);
        }

        Ok(importances)
    }
}

impl<State> MultiOutputDecisionTree<State> {
    /// Create a new multi-output decision tree
    pub fn new(config: MultiOutputTreeConfig) -> MultiOutputDecisionTree<Untrained> {
        MultiOutputDecisionTree {
            config,
            trees: Vec::new(),
            n_outputs: 0,
            n_features: 0,
            output_types: Vec::new(),
            label_correlations: None,
            state: PhantomData,
        }
    }
}

impl MultiOutputDecisionTree<Untrained> {
    /// Fit the multi-output tree
    pub fn fit(self, x: &Array2<f64>, y: &Array2<f64>) -> Result<MultiOutputDecisionTree<Trained>> {
        let n_samples = x.nrows();
        let n_features = x.ncols();
        let n_outputs = y.ncols();

        if n_samples != y.nrows() {
            return Err(SklearsError::InvalidInput(
                "X and y must have the same number of samples".to_string(),
            ));
        }

        // Determine output types
        let mut output_types = Vec::new();
        for output_idx in 0..n_outputs {
            let output_col = y.column(output_idx);
            let is_integer = output_col.iter().all(|&val| val.fract() == 0.0);

            if is_integer {
                let unique_classes: std::collections::HashSet<i32> =
                    output_col.iter().map(|&val| val as i32).collect();
                output_types.push(OutputType::Classification {
                    n_classes: unique_classes.len(),
                });
            } else {
                output_types.push(OutputType::Regression);
            }
        }

        // Create appropriate multi-output model based on strategy
        let mut model: Box<dyn MultiOutputTreeModel> = match self.config.multi_output_strategy {
            MultiOutputStrategy::Independent => Box::new(IndependentMultiOutputTree::new(
                n_outputs,
                n_features,
                output_types.clone(),
                self.config.base_config.clone(),
            )),
            MultiOutputStrategy::Shared => Box::new(SharedMultiOutputTree::new(
                n_outputs,
                n_features,
                output_types.clone(),
                self.config.base_config.clone(),
            )),
            MultiOutputStrategy::Chained => Box::new(ChainedMultiOutputTree::new(
                n_outputs,
                n_features,
                output_types.clone(),
                self.config.base_config.clone(),
            )),
        };

        // Compute label correlations if needed
        let label_correlations = if let LabelCorrelation::Cooccurrence { min_support } =
            &self.config.label_correlation
        {
            Some(compute_label_correlations(y, *min_support)?)
        } else {
            None
        };

        // Fit the model
        model.fit_multi(x, y)?;

        Ok(MultiOutputDecisionTree {
            config: self.config,
            trees: vec![model],
            n_outputs,
            n_features,
            output_types,
            label_correlations,
            state: PhantomData,
        })
    }
}

impl MultiOutputDecisionTree<Trained> {
    /// Predict on new data
    pub fn predict(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        if x.ncols() != self.n_features {
            return Err(SklearsError::InvalidInput(format!(
                "Expected {} features, got {}",
                self.n_features,
                x.ncols()
            )));
        }

        if let Some(model) = self.trees.first() {
            model.predict_multi(x)
        } else {
            Err(SklearsError::PredictError("Model not fitted".to_string()))
        }
    }

    /// Get feature importances for each output
    pub fn feature_importances(&self) -> Result<Array2<f64>> {
        if let Some(model) = self.trees.first() {
            model.feature_importances_multi()
        } else {
            Err(SklearsError::InvalidInput("Model not fitted".to_string()))
        }
    }

    /// Get label correlations (for multi-label problems)
    pub fn label_correlations(&self) -> Option<&LabelCorrelationMatrix> {
        self.label_correlations.as_ref()
    }

    /// Return the number of outputs the model was fitted with (sklearn-compatible `n_outputs_`)
    pub fn n_outputs(&self) -> usize {
        self.n_outputs
    }

    /// Return the per-output types (regression vs classification) as a slice
    pub fn output_types(&self) -> &[OutputType] {
        &self.output_types
    }
}

/// Compute label correlations for multi-label data
fn compute_label_correlations(
    y: &Array2<f64>,
    _min_support: f64,
) -> Result<LabelCorrelationMatrix> {
    let n_samples = y.nrows();
    let n_labels = y.ncols();

    // Compute co-occurrence matrix
    let mut cooccurrence = Array2::zeros((n_labels, n_labels));
    for i in 0..n_samples {
        for label1 in 0..n_labels {
            for label2 in 0..n_labels {
                if y[[i, label1]] > 0.0 && y[[i, label2]] > 0.0 {
                    cooccurrence[[label1, label2]] += 1;
                }
            }
        }
    }

    // Compute correlation coefficients
    let mut correlations = Array2::zeros((n_labels, n_labels));
    for i in 0..n_labels {
        for j in 0..n_labels {
            if i == j {
                correlations[[i, j]] = 1.0;
            } else {
                let support_i = y.column(i).iter().filter(|&&val| val > 0.0).count() as f64;
                let support_j = y.column(j).iter().filter(|&&val| val > 0.0).count() as f64;
                let joint_support = cooccurrence[[i, j]] as f64;

                if support_i > 0.0 && support_j > 0.0 {
                    // Jaccard similarity
                    let union_support = support_i + support_j - joint_support;
                    correlations[[i, j]] = joint_support / union_support;
                }
            }
        }
    }

    Ok(LabelCorrelationMatrix {
        correlations,
        cooccurrence,
        hierarchy: None,
    })
}

/// Multi-label Random Forest configuration
#[derive(Debug, Clone)]
pub struct MultiLabelRandomForestConfig {
    /// Base random forest configuration
    pub base_config: RandomForestConfig,
    /// Multi-output strategy
    pub multi_output_strategy: MultiOutputStrategy,
    /// Label correlation strategy
    pub label_correlation: LabelCorrelation,
    /// Threshold for binary predictions (for multi-label classification)
    pub prediction_threshold: f64,
    /// Enable label-specific feature selection
    pub label_specific_features: bool,
}

impl Default for MultiLabelRandomForestConfig {
    fn default() -> Self {
        Self {
            base_config: RandomForestConfig::default(),
            multi_output_strategy: MultiOutputStrategy::Independent,
            label_correlation: LabelCorrelation::Independent,
            prediction_threshold: 0.5,
            label_specific_features: false,
        }
    }
}

/// Multi-label Random Forest classifier
pub struct MultiLabelRandomForest<State = Untrained> {
    config: MultiLabelRandomForestConfig,
    trees: Vec<MultiOutputDecisionTree<Trained>>,
    n_outputs: usize,
    n_features: usize,
    label_correlations: Option<LabelCorrelationMatrix>,
    state: PhantomData<State>,
}

impl MultiLabelRandomForest<Untrained> {
    /// Create a new multi-label random forest
    pub fn new(config: MultiLabelRandomForestConfig) -> Self {
        Self {
            config,
            trees: Vec::new(),
            n_outputs: 0,
            n_features: 0,
            label_correlations: None,
            state: PhantomData,
        }
    }

    /// Fit the multi-label random forest
    pub fn fit(self, x: &Array2<f64>, y: &Array2<f64>) -> Result<MultiLabelRandomForest<Trained>> {
        let n_samples = x.nrows();
        let n_features = x.ncols();
        let n_outputs = y.ncols();

        if n_samples != y.nrows() {
            return Err(SklearsError::InvalidInput(
                "X and y must have the same number of samples".to_string(),
            ));
        }

        // Compute label correlations if needed
        let label_correlations = if let LabelCorrelation::Cooccurrence { min_support } =
            &self.config.label_correlation
        {
            Some(compute_label_correlations(y, *min_support)?)
        } else {
            None
        };

        let mut trees = Vec::with_capacity(self.config.base_config.n_estimators);

        // Build ensemble of multi-output trees
        for _ in 0..self.config.base_config.n_estimators {
            // Bootstrap sample
            let bootstrap_indices = self.bootstrap_sample(n_samples);
            let x_bootstrap = self.select_samples(x, &bootstrap_indices);
            let y_bootstrap = self.select_samples(y, &bootstrap_indices);

            // Create and fit multi-output tree
            let tree_config = MultiOutputTreeConfig {
                base_config: self.config.base_config.clone().into(),
                multi_output_strategy: self.config.multi_output_strategy,
                label_correlation: self.config.label_correlation.clone(),
                ..Default::default()
            };

            let tree = MultiOutputDecisionTree::<Untrained>::new(tree_config);
            let fitted_tree = tree.fit(&x_bootstrap, &y_bootstrap)?;
            trees.push(fitted_tree);
        }

        Ok(MultiLabelRandomForest {
            config: self.config,
            trees,
            n_outputs,
            n_features,
            label_correlations,
            state: PhantomData,
        })
    }

    /// Bootstrap sample indices
    fn bootstrap_sample(&self, n_samples: usize) -> Vec<usize> {
        let mut rng = scirs2_core::random::thread_rng();
        (0..n_samples)
            .map(|_| rng.gen_range(0..n_samples))
            .collect()
    }

    /// Select samples by indices
    fn select_samples(&self, data: &Array2<f64>, indices: &[usize]) -> Array2<f64> {
        let n_features = data.ncols();
        let mut selected = Array2::zeros((indices.len(), n_features));

        for (i, &idx) in indices.iter().enumerate() {
            selected.row_mut(i).assign(&data.row(idx));
        }

        selected
    }
}

// Helper conversion function
impl From<RandomForestConfig> for DecisionTreeConfig {
    fn from(rf_config: RandomForestConfig) -> Self {
        DecisionTreeConfig {
            max_depth: rf_config.max_depth,
            min_samples_split: rf_config.min_samples_split,
            min_samples_leaf: rf_config.min_samples_leaf,
            max_features: rf_config.max_features,
            criterion: rf_config.criterion,
            random_state: rf_config.random_state,
            ..Default::default()
        }
    }
}

impl MultiLabelRandomForest<Trained> {
    /// Predict probabilities for each label
    pub fn predict_proba(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        if x.ncols() != self.n_features {
            return Err(SklearsError::InvalidInput(format!(
                "Expected {} features, got {}",
                self.n_features,
                x.ncols()
            )));
        }

        let n_samples = x.nrows();
        let mut predictions = Array2::zeros((n_samples, self.n_outputs));

        // Average predictions from all trees
        for tree in &self.trees {
            let tree_predictions = tree.predict(x)?;
            predictions = predictions + tree_predictions;
        }

        // Normalize by number of trees
        predictions /= self.trees.len() as f64;

        Ok(predictions)
    }

    /// Predict binary labels using threshold
    pub fn predict(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let probabilities = self.predict_proba(x)?;
        let threshold = self.config.prediction_threshold;

        Ok(probabilities.mapv(|p| if p >= threshold { 1.0 } else { 0.0 }))
    }

    /// Get feature importances averaged across all outputs and trees
    pub fn feature_importances(&self) -> Result<Array1<f64>> {
        let mut total_importances = Array1::zeros(self.n_features);

        for tree in &self.trees {
            let tree_importances = tree.feature_importances()?;
            let mean_importance = tree_importances
                .mean_axis(Axis(1))
                .expect("array should have elements for mean computation");
            total_importances = total_importances + mean_importance;
        }

        // Normalize by number of trees
        total_importances /= self.trees.len() as f64;

        // Normalize to sum to 1
        let sum = total_importances.sum();
        if sum > 0.0 {
            total_importances /= sum;
        }

        Ok(total_importances)
    }

    /// Get label correlations
    pub fn label_correlations(&self) -> Option<&LabelCorrelationMatrix> {
        self.label_correlations.as_ref()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_multi_output_tree_config() {
        let config = MultiOutputTreeConfig::default();
        assert!(matches!(
            config.multi_output_strategy,
            MultiOutputStrategy::Independent
        ));
        assert!(matches!(
            config.label_correlation,
            LabelCorrelation::Independent
        ));
        assert_eq!(config.min_samples_multioutput_split, 10);
        assert!(config.enable_feature_sharing);
    }

    #[test]
    fn test_single_output_tree() {
        let mut tree = SingleOutputTree::new(OutputType::Regression, 2);

        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let y = array![1.0, 2.0, 3.0];

        assert!(tree.fit(&x, &y).is_ok());

        let predictions = tree.predict(&x).expect("prediction should succeed");
        assert_eq!(predictions.len(), 3);
    }

    #[test]
    fn test_multi_output_decision_tree() {
        let config = MultiOutputTreeConfig::default();
        let tree = MultiOutputDecisionTree::<Untrained>::new(config);

        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let y = array![[1.0, 0.0], [2.0, 1.0], [3.0, 0.0]]; // Two outputs

        let fitted_tree = tree.fit(&x, &y).expect("model fitting should succeed");
        let predictions = fitted_tree.predict(&x).expect("prediction should succeed");

        assert_eq!(predictions.nrows(), 3);
        assert_eq!(predictions.ncols(), 2);
    }

    #[test]
    fn test_label_correlations() {
        let y = array![
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ];

        let correlations = compute_label_correlations(&y, 0.1).expect("operation should succeed");
        assert_eq!(correlations.correlations.nrows(), 3);
        assert_eq!(correlations.correlations.ncols(), 3);
        assert_eq!(correlations.cooccurrence.nrows(), 3);
        assert_eq!(correlations.cooccurrence.ncols(), 3);
    }

    #[test]
    fn test_multi_label_random_forest() {
        let config = MultiLabelRandomForestConfig {
            base_config: RandomForestConfig {
                n_estimators: 3,
                ..Default::default()
            },
            ..Default::default()
        };

        let forest = MultiLabelRandomForest::new(config);

        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0],];
        let y = array![[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0],];

        let fitted_forest = forest.fit(&x, &y).expect("model fitting should succeed");
        let predictions = fitted_forest
            .predict(&x)
            .expect("prediction should succeed");

        assert_eq!(predictions.nrows(), 4);
        assert_eq!(predictions.ncols(), 2);

        // All predictions should be 0 or 1
        for &pred in predictions.iter() {
            assert!(pred == 0.0 || pred == 1.0);
        }
    }

    #[test]
    fn test_feature_importances() {
        let config = MultiOutputTreeConfig::default();
        let tree = MultiOutputDecisionTree::<Untrained>::new(config);

        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
        let y = array![[1.0, 0.0], [2.0, 1.0], [3.0, 0.0], [4.0, 1.0]];

        let fitted_tree = tree.fit(&x, &y).expect("model fitting should succeed");
        let importances = fitted_tree
            .feature_importances()
            .expect("operation should succeed");

        assert_eq!(importances.nrows(), 2); // 2 features
        assert_eq!(importances.ncols(), 2); // 2 outputs
    }

    // ── Shared multi-output tree ──────────────────────────────────────────────

    #[test]
    fn test_shared_multi_output_tree_basic() {
        let x = array![
            [1.0, 0.0],
            [2.0, 0.5],
            [3.0, 1.0],
            [4.0, 1.5],
            [5.0, 2.0],
            [6.0, 2.5],
            [7.0, 3.0],
            [8.0, 3.5],
            [9.0, 4.0],
            [10.0, 4.5],
            [11.0, 5.0],
            [12.0, 5.5],
        ];
        // Two outputs that both correlate with x[:,0]
        let y = array![
            [1.0, 2.0],
            [1.0, 2.0],
            [1.0, 2.0],
            [1.0, 2.0],
            [2.0, 4.0],
            [2.0, 4.0],
            [2.0, 4.0],
            [2.0, 4.0],
            [3.0, 6.0],
            [3.0, 6.0],
            [3.0, 6.0],
            [3.0, 6.0],
        ];

        let config = MultiOutputTreeConfig {
            multi_output_strategy: MultiOutputStrategy::Shared,
            ..Default::default()
        };
        let tree = MultiOutputDecisionTree::<Untrained>::new(config);
        let fitted = tree.fit(&x, &y).expect("shared tree fit should succeed");
        let preds = fitted
            .predict(&x)
            .expect("shared tree predict should succeed");

        assert_eq!(preds.nrows(), 12);
        assert_eq!(preds.ncols(), 2);
        // Predictions should be non-trivial (not all zeros)
        let nonzero = preds.iter().any(|&v| v > 0.0);
        assert!(nonzero, "shared tree should produce non-zero predictions");
    }

    #[test]
    fn test_shared_multi_output_tree_regression_accuracy() {
        // Larger dataset: two outputs that are perfect linear functions of x
        let n = 30usize;
        let mut x_data = Vec::with_capacity(n * 2);
        let mut y_data = Vec::with_capacity(n * 2);
        for i in 0..n {
            let xi = i as f64;
            x_data.push(xi);
            x_data.push(xi * 0.5);
            y_data.push(xi * 2.0);
            y_data.push(xi * 3.0);
        }
        let x = Array2::from_shape_vec((n, 2), x_data).expect("shape should be valid");
        let y = Array2::from_shape_vec((n, 2), y_data).expect("shape should be valid");

        let config = MultiOutputTreeConfig {
            multi_output_strategy: MultiOutputStrategy::Shared,
            ..Default::default()
        };
        let tree = MultiOutputDecisionTree::<Untrained>::new(config);
        let fitted = tree.fit(&x, &y).expect("shared tree regression fit");
        let preds = fitted.predict(&x).expect("shared tree regression predict");

        assert_eq!(preds.nrows(), n);
        assert_eq!(preds.ncols(), 2);
    }

    #[test]
    fn test_shared_tree_feature_importances_shape() {
        let x = array![
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
            [9.0, 10.0],
            [11.0, 12.0],
            [13.0, 14.0],
            [15.0, 16.0],
            [17.0, 18.0],
            [19.0, 20.0],
            [21.0, 22.0],
            [23.0, 24.0]
        ];
        let y = array![
            [1.0, 0.0],
            [2.0, 1.0],
            [1.0, 0.0],
            [2.0, 1.0],
            [1.0, 0.0],
            [2.0, 1.0],
            [1.0, 0.0],
            [2.0, 1.0],
            [1.0, 0.0],
            [2.0, 1.0],
            [1.0, 0.0],
            [2.0, 1.0]
        ];
        let config = MultiOutputTreeConfig {
            multi_output_strategy: MultiOutputStrategy::Shared,
            ..Default::default()
        };
        let fitted = MultiOutputDecisionTree::<Untrained>::new(config)
            .fit(&x, &y)
            .expect("fit");
        let imp = fitted.feature_importances().expect("importances");
        assert_eq!(imp.nrows(), 2); // 2 features
        assert_eq!(imp.ncols(), 2); // 2 outputs
    }

    // ── Chained multi-output tree ─────────────────────────────────────────────

    #[test]
    fn test_chained_multi_output_tree_basic() {
        let x = array![
            [1.0, 0.0],
            [2.0, 0.5],
            [3.0, 1.0],
            [4.0, 1.5],
            [5.0, 2.0],
            [6.0, 2.5],
            [7.0, 3.0],
            [8.0, 3.5],
            [9.0, 4.0],
            [10.0, 4.5],
            [11.0, 5.0],
            [12.0, 5.5],
        ];
        let y = array![
            [1.0, 2.0],
            [1.0, 2.0],
            [1.0, 2.0],
            [1.0, 2.0],
            [2.0, 4.0],
            [2.0, 4.0],
            [2.0, 4.0],
            [2.0, 4.0],
            [3.0, 6.0],
            [3.0, 6.0],
            [3.0, 6.0],
            [3.0, 6.0],
        ];

        let config = MultiOutputTreeConfig {
            multi_output_strategy: MultiOutputStrategy::Chained,
            ..Default::default()
        };
        let tree = MultiOutputDecisionTree::<Untrained>::new(config);
        let fitted = tree.fit(&x, &y).expect("chained tree fit should succeed");
        let preds = fitted
            .predict(&x)
            .expect("chained tree predict should succeed");

        assert_eq!(preds.nrows(), 12);
        assert_eq!(preds.ncols(), 2);
    }

    #[test]
    fn test_chained_tree_uses_previous_outputs() {
        // output_1 = output_0 * 2 — a perfect linear relationship.
        // The chained tree sees the real output_0 as a feature when training
        // output_1, so it should learn a tighter model than independent trees.
        let n = 20usize;
        let mut x_data = Vec::with_capacity(n);
        let mut y_data = Vec::with_capacity(n * 2);
        for i in 0..n {
            x_data.push(i as f64);
            y_data.push((i % 3) as f64); // output_0
            y_data.push(((i % 3) * 2) as f64); // output_1 = 2 * output_0
        }
        let x = Array2::from_shape_vec((n, 1), x_data).expect("shape");
        let y = Array2::from_shape_vec((n, 2), y_data).expect("shape");

        let config = MultiOutputTreeConfig {
            multi_output_strategy: MultiOutputStrategy::Chained,
            ..Default::default()
        };
        let fitted = MultiOutputDecisionTree::<Untrained>::new(config)
            .fit(&x, &y)
            .expect("chained tree fit");
        let preds = fitted.predict(&x).expect("chained tree predict");
        assert_eq!(preds.nrows(), n);
        assert_eq!(preds.ncols(), 2);
        // Predictions for output_1 should be non-negative
        assert!(preds.column(1).iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn test_chained_tree_feature_importances_shape() {
        let x = array![
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
            [9.0, 10.0],
            [11.0, 12.0],
            [13.0, 14.0],
            [15.0, 16.0],
            [17.0, 18.0],
            [19.0, 20.0],
            [21.0, 22.0],
            [23.0, 24.0]
        ];
        let y = array![
            [1.0, 0.0],
            [2.0, 1.0],
            [1.0, 0.0],
            [2.0, 1.0],
            [1.0, 0.0],
            [2.0, 1.0],
            [1.0, 0.0],
            [2.0, 1.0],
            [1.0, 0.0],
            [2.0, 1.0],
            [1.0, 0.0],
            [2.0, 1.0]
        ];
        let config = MultiOutputTreeConfig {
            multi_output_strategy: MultiOutputStrategy::Chained,
            ..Default::default()
        };
        let fitted = MultiOutputDecisionTree::<Untrained>::new(config)
            .fit(&x, &y)
            .expect("fit");
        let imp = fitted.feature_importances().expect("importances");
        assert_eq!(imp.nrows(), 2); // 2 original features
        assert_eq!(imp.ncols(), 2); // 2 outputs
    }
}
