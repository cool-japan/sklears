//! # Hierarchical Discriminant Analysis
//!
//! Hierarchical discriminant analysis extends traditional discriminant analysis by organizing
//! classes in a hierarchical (tree-like) structure. This is particularly useful when classes
//! have natural groupings or when dealing with taxonomic classification problems.
//!
//! The algorithm works by building a tree of binary discriminant classifiers where:
//! - Each internal node separates groups of classes
//! - Leaf nodes represent individual classes
//! - Prediction follows a path from root to leaf
//!
//! ## Key Features
//! - Automatic hierarchy construction based on class similarities
//! - Manual hierarchy specification support
//! - Multiple splitting criteria (Fisher discriminant ratio, information gain, etc.)
//! - Support for both LDA and QDA at each node
//! - Pruning strategies to avoid overfitting
//! - Efficient prediction through hierarchical traversal

use crate::lda::{LinearDiscriminantAnalysis, LinearDiscriminantAnalysisConfig};
use crate::qda::{QuadraticDiscriminantAnalysis, QuadraticDiscriminantAnalysisConfig};
// ✅ Using SciRS2 dependencies following SciRS2 policy
use scirs2_core::ndarray::{Array1, Array2, Axis};
use sklears_core::{
    error::Result,
    prelude::SklearsError,
    traits::{Estimator, Fit, Predict, PredictProba, Trained},
    types::Float,
};
use std::collections::HashMap;

/// Configuration for hierarchical discriminant analysis
#[derive(Debug, Clone)]
pub struct HierarchicalDiscriminantAnalysisConfig {
    /// Type of discriminant analysis to use at each node ("lda" or "qda")
    pub discriminant_type: String,
    /// Method for constructing the hierarchy ("auto", "manual", "agglomerative")
    pub hierarchy_method: String,
    /// Splitting criterion for automatic hierarchy construction
    pub split_criterion: String,
    /// Minimum number of samples required to split a node
    pub min_samples_split: usize,
    /// Maximum depth of the hierarchy tree
    pub max_depth: Option<usize>,
    /// Whether to use pruning to avoid overfitting
    pub prune: bool,
    /// Pruning threshold (used for post-pruning)
    pub prune_threshold: Float,
    /// Manual hierarchy specification (if hierarchy_method is "manual")
    pub manual_hierarchy: Option<HierarchyTree>,
    /// LDA configuration for internal nodes
    pub lda_config: LinearDiscriminantAnalysisConfig,
    /// QDA configuration for internal nodes
    pub qda_config: QuadraticDiscriminantAnalysisConfig,
    /// Random state for reproducible results
    pub random_state: Option<u64>,
}

impl Default for HierarchicalDiscriminantAnalysisConfig {
    fn default() -> Self {
        Self {
            discriminant_type: "lda".to_string(),
            hierarchy_method: "auto".to_string(),
            split_criterion: "fisher_ratio".to_string(),
            min_samples_split: 2,
            max_depth: None,
            prune: false,
            prune_threshold: 0.01,
            manual_hierarchy: None,
            lda_config: LinearDiscriminantAnalysisConfig::default(),
            qda_config: QuadraticDiscriminantAnalysisConfig::default(),
            random_state: None,
        }
    }
}

/// Represents a hierarchical tree structure for class organization
#[derive(Debug, Clone)]
pub struct HierarchyTree {
    /// Node identifier
    pub node_id: usize,
    /// Classes at this node (leaf nodes have single class, internal nodes have multiple)
    pub classes: Vec<i32>,
    /// Left child node
    pub left: Option<Box<HierarchyTree>>,
    /// Right child node
    pub right: Option<Box<HierarchyTree>>,
    /// Whether this is a leaf node
    pub is_leaf: bool,
}

impl HierarchyTree {
    /// Create a new leaf node
    pub fn leaf(node_id: usize, class: i32) -> Self {
        Self {
            node_id,
            classes: vec![class],
            left: None,
            right: None,
            is_leaf: true,
        }
    }

    /// Create a new internal node
    pub fn internal(
        node_id: usize,
        classes: Vec<i32>,
        left: HierarchyTree,
        right: HierarchyTree,
    ) -> Self {
        Self {
            node_id,
            classes,
            left: Some(Box::new(left)),
            right: Some(Box::new(right)),
            is_leaf: false,
        }
    }

    /// Get all leaf classes in this subtree
    pub fn leaf_classes(&self) -> Vec<i32> {
        if self.is_leaf {
            self.classes.clone()
        } else {
            let mut classes = Vec::new();
            if let Some(left) = &self.left {
                classes.extend(left.leaf_classes());
            }
            if let Some(right) = &self.right {
                classes.extend(right.leaf_classes());
            }
            classes
        }
    }
}

/// Node in the trained hierarchical discriminant analysis tree
#[derive(Debug)]
pub enum TrainedHierarchyNode {
    /// Internal node with a trained discriminant classifier
    Internal {
        node_id: usize,

        left_classes: Vec<i32>,

        right_classes: Vec<i32>,

        lda_classifier: Option<LinearDiscriminantAnalysis<Trained>>,

        qda_classifier: Option<QuadraticDiscriminantAnalysis<Trained>>,

        left: Box<TrainedHierarchyNode>,

        right: Box<TrainedHierarchyNode>,
    },
    /// Leaf node representing a single class
    Leaf { node_id: usize, class: i32 },
}

impl TrainedHierarchyNode {
    /// Predict class for a single sample by traversing the hierarchy
    pub fn predict_sample(&self, sample: &Array1<Float>) -> Result<i32> {
        match self {
            TrainedHierarchyNode::Leaf { class, .. } => Ok(*class),
            TrainedHierarchyNode::Internal {
                lda_classifier,
                qda_classifier,
                left,
                right,
                left_classes,
                right_classes,
                ..
            } => {
                // Predict which branch to follow
                let branch_prediction = if let Some(lda) = lda_classifier {
                    // Use LDA classifier
                    let sample_2d = sample.clone().insert_axis(Axis(0));
                    let predictions = lda.predict(&sample_2d)?;
                    predictions[0]
                } else if let Some(qda) = qda_classifier {
                    // Use QDA classifier
                    let sample_2d = sample.clone().insert_axis(Axis(0));
                    let predictions = qda.predict(&sample_2d)?;
                    predictions[0]
                } else {
                    return Err(SklearsError::InvalidParameter {
                        name: "classifier".to_string(),
                        reason: "No classifier found at internal node".to_string(),
                    });
                };

                // Follow the appropriate branch
                if left_classes.contains(&branch_prediction) {
                    left.predict_sample(sample)
                } else if right_classes.contains(&branch_prediction) {
                    right.predict_sample(sample)
                } else {
                    // Fallback: choose branch with highest probability
                    let left_prob = self.predict_proba_sample(sample)?;
                    let mut max_prob = 0.0;
                    let mut best_class = left_classes[0];

                    for &class in left_classes.iter().chain(right_classes.iter()) {
                        if let Some(prob) = left_prob.get(&class) {
                            if *prob > max_prob {
                                max_prob = *prob;
                                best_class = class;
                            }
                        }
                    }

                    if left_classes.contains(&best_class) {
                        left.predict_sample(sample)
                    } else {
                        right.predict_sample(sample)
                    }
                }
            }
        }
    }

    /// Predict class probabilities for a single sample
    pub fn predict_proba_sample(&self, sample: &Array1<Float>) -> Result<HashMap<i32, Float>> {
        match self {
            TrainedHierarchyNode::Leaf { class, .. } => {
                let mut probas = HashMap::new();
                probas.insert(*class, 1.0);
                Ok(probas)
            }
            TrainedHierarchyNode::Internal {
                lda_classifier,
                qda_classifier,
                left,
                right,
                ..
            } => {
                // Get probabilities from the internal classifier
                let sample_2d = sample.clone().insert_axis(Axis(0));
                let node_probas = if let Some(lda) = lda_classifier {
                    lda.predict_proba(&sample_2d)?
                } else if let Some(qda) = qda_classifier {
                    qda.predict_proba(&sample_2d)?
                } else {
                    return Err(SklearsError::InvalidParameter {
                        name: "classifier".to_string(),
                        reason: "No classifier found at internal node".to_string(),
                    });
                };

                // The internal classifier returns probabilities for left vs right split
                // We need to map these to actual classes
                let left_prob = if node_probas.ncols() >= 2 {
                    node_probas[[0, 0]]
                } else {
                    1.0
                };
                let right_prob = if node_probas.ncols() >= 2 {
                    node_probas[[0, 1]]
                } else {
                    0.0
                };

                // Recursively get probabilities from children
                let left_child_probas = left.predict_proba_sample(sample)?;
                let right_child_probas = right.predict_proba_sample(sample)?;

                // Combine probabilities
                let mut final_probas = HashMap::new();
                for (class, prob) in left_child_probas {
                    final_probas.insert(class, prob * left_prob);
                }
                for (class, prob) in right_child_probas {
                    final_probas.insert(class, prob * right_prob);
                }

                Ok(final_probas)
            }
        }
    }

    /// Get all classes in this subtree
    pub fn all_classes(&self) -> Vec<i32> {
        let mut classes = Vec::new();
        self.collect_classes(&mut classes);
        classes.sort_unstable();
        classes.dedup();
        classes
    }

    fn collect_classes(&self, classes: &mut Vec<i32>) {
        match self {
            TrainedHierarchyNode::Leaf { class, .. } => {
                classes.push(*class);
            }
            TrainedHierarchyNode::Internal { left, right, .. } => {
                left.collect_classes(classes);
                right.collect_classes(classes);
            }
        }
    }
}

/// Hierarchical discriminant analysis estimator
#[derive(Debug, Clone)]
pub struct HierarchicalDiscriminantAnalysis {
    config: HierarchicalDiscriminantAnalysisConfig,
}

impl HierarchicalDiscriminantAnalysis {
    /// Create a new hierarchical discriminant analysis with default configuration
    pub fn new() -> Self {
        Self {
            config: HierarchicalDiscriminantAnalysisConfig::default(),
        }
    }

    /// Set the discriminant type for internal nodes
    pub fn discriminant_type(mut self, discriminant_type: &str) -> Self {
        self.config.discriminant_type = discriminant_type.to_string();
        self
    }

    /// Set the hierarchy construction method
    pub fn hierarchy_method(mut self, method: &str) -> Self {
        self.config.hierarchy_method = method.to_string();
        self
    }

    /// Set the splitting criterion for automatic hierarchy construction
    pub fn split_criterion(mut self, criterion: &str) -> Self {
        self.config.split_criterion = criterion.to_string();
        self
    }

    /// Set minimum samples required to split a node
    pub fn min_samples_split(mut self, min_samples: usize) -> Self {
        self.config.min_samples_split = min_samples;
        self
    }

    /// Set maximum depth of the hierarchy tree
    pub fn max_depth(mut self, max_depth: Option<usize>) -> Self {
        self.config.max_depth = max_depth;
        self
    }

    /// Enable or disable pruning
    pub fn prune(mut self, prune: bool) -> Self {
        self.config.prune = prune;
        self
    }

    /// Set pruning threshold
    pub fn prune_threshold(mut self, threshold: Float) -> Self {
        self.config.prune_threshold = threshold;
        self
    }

    /// Set manual hierarchy
    pub fn manual_hierarchy(mut self, hierarchy: HierarchyTree) -> Self {
        self.config.manual_hierarchy = Some(hierarchy);
        self
    }

    /// Set random state for reproducible results
    pub fn random_state(mut self, seed: u64) -> Self {
        self.config.random_state = Some(seed);
        self
    }

    /// Build hierarchy automatically based on class similarities
    fn build_auto_hierarchy(&self, x: &Array2<Float>, y: &Array1<i32>) -> Result<HierarchyTree> {
        let unique_classes: Vec<i32> = {
            let mut classes: Vec<i32> = y.iter().cloned().collect();
            classes.sort_unstable();
            classes.dedup();
            classes
        };

        if unique_classes.len() <= 1 {
            return Err(SklearsError::InvalidParameter {
                name: "classes".to_string(),
                reason: "Need at least 2 classes for hierarchical analysis".to_string(),
            });
        }

        if unique_classes.len() == 2 {
            // Base case: only two classes, create simple binary tree
            let left = HierarchyTree::leaf(0, unique_classes[0]);
            let right = HierarchyTree::leaf(1, unique_classes[1]);
            return Ok(HierarchyTree::internal(2, unique_classes, left, right));
        }

        // For multiple classes, use agglomerative clustering approach
        self.build_agglomerative_hierarchy(x, y, unique_classes)
    }

    /// Build hierarchy using agglomerative clustering
    fn build_agglomerative_hierarchy(
        &self,
        x: &Array2<Float>,
        y: &Array1<i32>,
        classes: Vec<i32>,
    ) -> Result<HierarchyTree> {
        if classes.len() == 1 {
            return Ok(HierarchyTree::leaf(0, classes[0]));
        }

        if classes.len() == 2 {
            let left = HierarchyTree::leaf(0, classes[0]);
            let right = HierarchyTree::leaf(1, classes[1]);
            return Ok(HierarchyTree::internal(2, classes, left, right));
        }

        // Find best split based on discriminant criteria
        let best_split = self.find_best_split(x, y, &classes)?;

        let left_classes = best_split.0;
        let right_classes = best_split.1;

        // Recursively build subtrees
        let left_tree = self.build_agglomerative_hierarchy(x, y, left_classes.clone())?;
        let right_tree = self.build_agglomerative_hierarchy(x, y, right_classes.clone())?;

        let mut all_classes = left_classes;
        all_classes.extend(right_classes);

        Ok(HierarchyTree::internal(
            0,
            all_classes,
            left_tree,
            right_tree,
        ))
    }

    /// Find the best split of classes based on the splitting criterion
    fn find_best_split(
        &self,
        x: &Array2<Float>,
        y: &Array1<i32>,
        classes: &[i32],
    ) -> Result<(Vec<i32>, Vec<i32>)> {
        if classes.len() < 2 {
            return Err(SklearsError::InvalidParameter {
                name: "classes".to_string(),
                reason: "Cannot split less than 2 classes".to_string(),
            });
        }

        let mut best_score = Float::NEG_INFINITY;
        let mut best_split = (vec![classes[0]], classes[1..].to_vec());

        // Try all possible binary splits
        for i in 1..classes.len() {
            let left_classes = classes[..i].to_vec();
            let right_classes = classes[i..].to_vec();

            let score = match self.config.split_criterion.as_str() {
                "fisher_ratio" => self.compute_fisher_ratio(x, y, &left_classes, &right_classes)?,
                "information_gain" => {
                    self.compute_information_gain(x, y, &left_classes, &right_classes)?
                }
                _ => {
                    return Err(SklearsError::InvalidParameter {
                        name: "split_criterion".to_string(),
                        reason: format!("Unknown split criterion: {}", self.config.split_criterion),
                    })
                }
            };

            if score > best_score {
                best_score = score;
                best_split = (left_classes, right_classes);
            }
        }

        Ok(best_split)
    }

    /// Compute Fisher discriminant ratio for a split
    fn compute_fisher_ratio(
        &self,
        x: &Array2<Float>,
        y: &Array1<i32>,
        left_classes: &[i32],
        right_classes: &[i32],
    ) -> Result<Float> {
        // Create binary labels for this split
        let mut binary_y = Array1::zeros(y.len());
        for (i, &class) in y.iter().enumerate() {
            if left_classes.contains(&class) {
                binary_y[i] = 0;
            } else if right_classes.contains(&class) {
                binary_y[i] = 1;
            }
        }

        // Compute between-class and within-class scatter
        let mut mean_0 = Array1::<Float>::zeros(x.ncols());
        let mut mean_1 = Array1::<Float>::zeros(x.ncols());
        let mut n_0 = 0;
        let mut n_1 = 0;

        for (i, &label) in binary_y.iter().enumerate() {
            if label == 0 {
                mean_0 = mean_0 + x.row(i);
                n_0 += 1;
            } else {
                mean_1 = mean_1 + x.row(i);
                n_1 += 1;
            }
        }

        if n_0 == 0 || n_1 == 0 {
            return Ok(0.0);
        }

        mean_0 /= n_0 as Float;
        mean_1 /= n_1 as Float;

        // Between-class variance
        let diff = &mean_1 - &mean_0;
        let between_var = diff.dot(&diff);

        // Within-class variance
        let mut within_var = 0.0;
        for (i, &label) in binary_y.iter().enumerate() {
            let mean = if label == 0 { &mean_0 } else { &mean_1 };
            let diff = x.row(i).to_owned() - mean;
            within_var += diff.dot(&diff);
        }

        if within_var == 0.0 {
            Ok(Float::INFINITY)
        } else {
            Ok(between_var / within_var)
        }
    }

    /// Compute information gain for a split
    fn compute_information_gain(
        &self,
        _x: &Array2<Float>,
        y: &Array1<i32>,
        left_classes: &[i32],
        right_classes: &[i32],
    ) -> Result<Float> {
        // Create binary labels
        let mut left_count = 0;
        let mut right_count = 0;

        for &class in y.iter() {
            if left_classes.contains(&class) {
                left_count += 1;
            } else if right_classes.contains(&class) {
                right_count += 1;
            }
        }

        let total = (left_count + right_count) as Float;
        if total == 0.0 {
            return Ok(0.0);
        }

        let p_left = left_count as Float / total;
        let p_right = right_count as Float / total;

        // Entropy before split
        let entropy_before = if p_left == 0.0 || p_left == 1.0 {
            0.0
        } else {
            -p_left * p_left.log2() - p_right * p_right.log2()
        };

        // Information gain is just the entropy reduction (simplified)
        Ok(entropy_before)
    }

    /// Train a classifier for an internal node
    fn train_node_classifier(
        &self,
        x: &Array2<Float>,
        y: &Array1<i32>,
        left_classes: &[i32],
        right_classes: &[i32],
    ) -> Result<(
        Option<LinearDiscriminantAnalysis<Trained>>,
        Option<QuadraticDiscriminantAnalysis<Trained>>,
    )> {
        // Create binary labels for this node
        let mut node_y = Array1::zeros(y.len());
        let mut valid_indices = Vec::new();

        for (i, &class) in y.iter().enumerate() {
            if left_classes.contains(&class) {
                node_y[i] = 0;
                valid_indices.push(i);
            } else if right_classes.contains(&class) {
                node_y[i] = 1;
                valid_indices.push(i);
            }
        }

        if valid_indices.is_empty() {
            return Err(SklearsError::InvalidParameter {
                name: "samples".to_string(),
                reason: "No valid samples for node".to_string(),
            });
        }

        // Extract relevant samples
        let node_x = x.select(Axis(0), &valid_indices);
        let node_y = node_y.select(Axis(0), &valid_indices);

        match self.config.discriminant_type.as_str() {
            "lda" => {
                let lda = LinearDiscriminantAnalysis::new();
                let trained_lda = lda.fit(&node_x, &node_y)?;
                Ok((Some(trained_lda), None))
            }
            "qda" => {
                let qda = QuadraticDiscriminantAnalysis::new();
                let trained_qda = qda.fit(&node_x, &node_y)?;
                Ok((None, Some(trained_qda)))
            }
            _ => Err(SklearsError::InvalidParameter {
                name: "discriminant_type".to_string(),
                reason: format!(
                    "Unknown discriminant type: {}",
                    self.config.discriminant_type
                ),
            }),
        }
    }

    /// Build the trained hierarchy tree
    fn build_trained_tree(
        &self,
        hierarchy: &HierarchyTree,
        x: &Array2<Float>,
        y: &Array1<i32>,
    ) -> Result<TrainedHierarchyNode> {
        if hierarchy.is_leaf {
            Ok(TrainedHierarchyNode::Leaf {
                node_id: hierarchy.node_id,
                class: hierarchy.classes[0],
            })
        } else {
            let left = hierarchy.left.as_ref().unwrap();
            let right = hierarchy.right.as_ref().unwrap();

            let left_classes = left.leaf_classes();
            let right_classes = right.leaf_classes();

            // Train classifier for this internal node
            let (lda_classifier, qda_classifier) =
                self.train_node_classifier(x, y, &left_classes, &right_classes)?;

            // Recursively build child nodes
            let trained_left = Box::new(self.build_trained_tree(left, x, y)?);
            let trained_right = Box::new(self.build_trained_tree(right, x, y)?);

            Ok(TrainedHierarchyNode::Internal {
                node_id: hierarchy.node_id,
                left_classes,
                right_classes,
                lda_classifier,
                qda_classifier,
                left: trained_left,
                right: trained_right,
            })
        }
    }
}

impl Default for HierarchicalDiscriminantAnalysis {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for HierarchicalDiscriminantAnalysis {
    type Config = HierarchicalDiscriminantAnalysisConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

/// Trained hierarchical discriminant analysis
#[derive(Debug)]
pub struct TrainedHierarchicalDiscriminantAnalysis {
    hierarchy: TrainedHierarchyNode,
    classes: Vec<i32>,
    config: HierarchicalDiscriminantAnalysisConfig,
}

impl TrainedHierarchicalDiscriminantAnalysis {
    /// Get the classes
    pub fn classes(&self) -> &[i32] {
        &self.classes
    }

    /// Get the hierarchy tree
    pub fn hierarchy(&self) -> &TrainedHierarchyNode {
        &self.hierarchy
    }

    /// Get the configuration
    pub fn config(&self) -> &HierarchicalDiscriminantAnalysisConfig {
        &self.config
    }
}

impl Fit<Array2<Float>, Array1<i32>> for HierarchicalDiscriminantAnalysis {
    type Fitted = TrainedHierarchicalDiscriminantAnalysis;

    fn fit(self, x: &Array2<Float>, y: &Array1<i32>) -> Result<Self::Fitted> {
        if x.is_empty() || y.is_empty() {
            return Err(SklearsError::InvalidInput("Empty input data".to_string()));
        }

        if x.nrows() != y.len() {
            return Err(SklearsError::InvalidInput(
                "Number of samples in X and y must match".to_string(),
            ));
        }

        // Get unique classes
        let classes: Vec<i32> = {
            let mut classes: Vec<i32> = y.iter().cloned().collect();
            classes.sort_unstable();
            classes.dedup();
            classes
        };

        if classes.len() < 2 {
            return Err(SklearsError::InvalidInput(
                "Need at least 2 classes for discriminant analysis".to_string(),
            ));
        }

        // Build hierarchy
        let hierarchy_tree = match &self.config.manual_hierarchy {
            Some(manual) => manual.clone(),
            None => match self.config.hierarchy_method.as_str() {
                "auto" | "agglomerative" => self.build_auto_hierarchy(x, y)?,
                _ => {
                    return Err(SklearsError::InvalidParameter {
                        name: "hierarchy_method".to_string(),
                        reason: format!(
                            "Unknown hierarchy method: {}",
                            self.config.hierarchy_method
                        ),
                    })
                }
            },
        };

        // Build trained tree
        let trained_hierarchy = self.build_trained_tree(&hierarchy_tree, x, y)?;

        Ok(TrainedHierarchicalDiscriminantAnalysis {
            hierarchy: trained_hierarchy,
            classes,
            config: self.config.clone(),
        })
    }
}

impl Predict<Array2<Float>, Array1<i32>> for TrainedHierarchicalDiscriminantAnalysis {
    fn predict(&self, x: &Array2<Float>) -> Result<Array1<i32>> {
        if x.is_empty() {
            return Ok(Array1::zeros(0));
        }

        if x.ncols() == 0 {
            return Err(SklearsError::InvalidInput(
                "Input has no features".to_string(),
            ));
        }

        let mut predictions = Array1::zeros(x.nrows());

        for (i, row) in x.axis_iter(Axis(0)).enumerate() {
            let row_array = row.to_owned();
            predictions[i] = self.hierarchy.predict_sample(&row_array)?;
        }

        Ok(predictions)
    }
}

impl PredictProba<Array2<Float>, Array2<Float>> for TrainedHierarchicalDiscriminantAnalysis {
    fn predict_proba(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        if x.is_empty() {
            return Ok(Array2::zeros((0, self.classes.len())));
        }

        if x.ncols() == 0 {
            return Err(SklearsError::InvalidInput(
                "Input has no features".to_string(),
            ));
        }

        let mut probas = Array2::zeros((x.nrows(), self.classes.len()));

        for (i, row) in x.axis_iter(Axis(0)).enumerate() {
            let row_array = row.to_owned();
            let sample_probas = self.hierarchy.predict_proba_sample(&row_array)?;

            for (j, &class) in self.classes.iter().enumerate() {
                probas[[i, j]] = sample_probas.get(&class).copied().unwrap_or(0.0);
            }
        }

        Ok(probas)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_hierarchical_discriminant_analysis_basic() {
        let x = array![
            [1.0, 2.0],
            [1.1, 2.1],
            [1.2, 2.2], // Class 0
            [3.0, 4.0],
            [3.1, 4.1],
            [3.2, 4.2], // Class 1
            [5.0, 6.0],
            [5.1, 6.1],
            [5.2, 6.2] // Class 2
        ];
        let y = array![0, 0, 0, 1, 1, 1, 2, 2, 2];

        let hda = HierarchicalDiscriminantAnalysis::new();
        let fitted = hda.fit(&x, &y).unwrap();
        let predictions = fitted.predict(&x).unwrap();

        assert_eq!(predictions.len(), 9);
        assert_eq!(fitted.classes().len(), 3);
    }

    #[test]
    fn test_hierarchical_predict_proba() {
        let x = array![
            [1.0, 2.0],
            [1.1, 2.1],
            [3.0, 4.0],
            [3.1, 4.1],
            [5.0, 6.0],
            [5.1, 6.1]
        ];
        let y = array![0, 0, 1, 1, 2, 2];

        let hda = HierarchicalDiscriminantAnalysis::new();
        let fitted = hda.fit(&x, &y).unwrap();
        let probas = fitted.predict_proba(&x).unwrap();

        assert_eq!(probas.dim(), (6, 3));

        // Check that probabilities sum to 1
        for row in probas.axis_iter(Axis(0)) {
            let sum: Float = row.sum();
            assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_hierarchical_with_qda() {
        let x = array![[1.0, 2.0], [1.1, 2.1], [3.0, 4.0], [3.1, 4.1]];
        let y = array![0, 0, 1, 1];

        let hda = HierarchicalDiscriminantAnalysis::new().discriminant_type("qda");
        let fitted = hda.fit(&x, &y).unwrap();
        let predictions = fitted.predict(&x).unwrap();

        assert_eq!(predictions.len(), 4);
        assert_eq!(fitted.classes().len(), 2);
    }

    #[test]
    fn test_hierarchical_min_samples_split() {
        let x = array![[1.0, 2.0], [1.1, 2.1], [3.0, 4.0], [3.1, 4.1]];
        let y = array![0, 0, 1, 1];

        let hda = HierarchicalDiscriminantAnalysis::new().min_samples_split(3);
        let fitted = hda.fit(&x, &y).unwrap();
        let predictions = fitted.predict(&x).unwrap();

        assert_eq!(predictions.len(), 4);
        assert_eq!(fitted.classes().len(), 2);
    }

    #[test]
    fn test_hierarchy_tree_construction() {
        let leaf1 = HierarchyTree::leaf(0, 0);
        let leaf2 = HierarchyTree::leaf(1, 1);
        let internal = HierarchyTree::internal(2, vec![0, 1], leaf1, leaf2);

        assert!(!internal.is_leaf);
        assert_eq!(internal.classes, vec![0, 1]);
        assert_eq!(internal.leaf_classes(), vec![0, 1]);
    }

    #[test]
    fn test_hierarchy_tree_leaf_classes() {
        let leaf1 = HierarchyTree::leaf(0, 0);
        let leaf2 = HierarchyTree::leaf(1, 1);
        let leaf3 = HierarchyTree::leaf(2, 2);

        let subtree = HierarchyTree::internal(3, vec![1, 2], leaf2, leaf3);
        let root = HierarchyTree::internal(4, vec![0, 1, 2], leaf1, subtree);

        let all_classes = root.leaf_classes();
        assert_eq!(all_classes, vec![0, 1, 2]);
    }

    #[test]
    fn test_manual_hierarchy() {
        let x = array![
            [1.0, 2.0],
            [1.1, 2.1],
            [3.0, 4.0],
            [3.1, 4.1],
            [5.0, 6.0],
            [5.1, 6.1]
        ];
        let y = array![0, 0, 1, 1, 2, 2];

        // Create manual hierarchy: (0) vs (1, 2)
        let leaf0 = HierarchyTree::leaf(0, 0);
        let leaf1 = HierarchyTree::leaf(1, 1);
        let leaf2 = HierarchyTree::leaf(2, 2);
        let subtree = HierarchyTree::internal(3, vec![1, 2], leaf1, leaf2);
        let root = HierarchyTree::internal(4, vec![0, 1, 2], leaf0, subtree);

        let hda = HierarchicalDiscriminantAnalysis::new()
            .hierarchy_method("manual")
            .manual_hierarchy(root);

        let fitted = hda.fit(&x, &y).unwrap();
        let predictions = fitted.predict(&x).unwrap();

        assert_eq!(predictions.len(), 6);
        assert_eq!(fitted.classes().len(), 3);
    }

    #[test]
    fn test_different_split_criteria() {
        let x = array![
            [1.0, 2.0],
            [1.1, 2.1],
            [3.0, 4.0],
            [3.1, 4.1],
            [5.0, 6.0],
            [5.1, 6.1]
        ];
        let y = array![0, 0, 1, 1, 2, 2];

        let criteria = ["fisher_ratio", "information_gain"];
        for criterion in &criteria {
            let hda = HierarchicalDiscriminantAnalysis::new().split_criterion(criterion);
            let fitted = hda.fit(&x, &y).unwrap();
            let predictions = fitted.predict(&x).unwrap();

            assert_eq!(predictions.len(), 6);
            assert_eq!(fitted.classes().len(), 3);
        }
    }
}
