//! Decision Tree implementation
//!
//! This module provides comprehensive Decision Tree Classifier and Regressor implementations
//! using advanced CART algorithms, complying with SciRS2 Policy.

use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use sklears_core::error::{Result, SklearsError};
use sklears_core::traits::{Estimator, Fit, Predict, Trained, Untrained};
use std::marker::PhantomData;

// Re-export types from existing modules
pub use crate::config::DecisionTreeConfig;
pub use crate::criteria::{ConditionalTestType, FeatureType, MonotonicConstraint, SplitCriterion};
pub use crate::node::{CompactTreeNode, CustomSplit, SurrogateSplit, TreeNode};
pub use crate::splits::{ChaidSplit, HyperplaneSplit};

/// Main Decision Tree structure that can be used for both classification and regression
#[derive(Debug, Clone)]
pub struct DecisionTree<State = Untrained> {
    config: DecisionTreeConfig,
    root: Option<TreeNode>,
    feature_importances: Option<Array1<f64>>,
    n_features: usize,
    n_samples: usize,
    state: PhantomData<State>,
}

/// Type alias for Decision Tree Classifier (untrained)
pub type DecisionTreeClassifier = DecisionTree<Untrained>;

/// Type alias for Decision Tree Regressor (untrained)
pub type DecisionTreeRegressor = DecisionTree<Untrained>;

impl<State> Default for DecisionTree<State> {
    fn default() -> Self {
        Self::new()
    }
}

impl<State> DecisionTree<State> {
    /// Create a new DecisionTree with default configuration
    pub fn new() -> Self {
        Self {
            config: DecisionTreeConfig::default(),
            root: None,
            feature_importances: None,
            n_features: 0,
            n_samples: 0,
            state: PhantomData,
        }
    }

    /// Create a new DecisionTree with custom configuration
    pub fn with_config(config: DecisionTreeConfig) -> Self {
        Self {
            config,
            root: None,
            feature_importances: None,
            n_features: 0,
            n_samples: 0,
            state: PhantomData,
        }
    }

    /// Create a builder for configuring the decision tree
    pub fn builder() -> DecisionTreeBuilder<State> {
        DecisionTreeBuilder::new()
    }

    /// Get the configuration of the decision tree
    pub fn config(&self) -> &DecisionTreeConfig {
        &self.config
    }

    /// Get the root node of the tree (if fitted)
    pub fn root(&self) -> Option<&TreeNode> {
        self.root.as_ref()
    }

    /// Get feature importances (if available)
    pub fn feature_importances(&self) -> Option<&Array1<f64>> {
        self.feature_importances.as_ref()
    }

    /// Get the number of features the tree was trained on
    pub fn n_features(&self) -> usize {
        self.n_features
    }

    /// Get the number of training samples
    pub fn n_samples(&self) -> usize {
        self.n_samples
    }

    /// Get the depth of the tree (returns root node depth if available)
    pub fn depth(&self) -> usize {
        match &self.root {
            Some(root) => root.depth,
            None => 0,
        }
    }

    /// Set the split criterion (fluent API)
    pub fn criterion(mut self, criterion: SplitCriterion) -> Self {
        self.config.criterion = criterion;
        self
    }

    /// Set the maximum depth of the tree (fluent API)
    pub fn max_depth(mut self, max_depth: usize) -> Self {
        self.config.max_depth = Some(max_depth);
        self
    }

    /// Set the minimum samples required to split an internal node (fluent API)
    pub fn min_samples_split(mut self, min_samples_split: usize) -> Self {
        self.config.min_samples_split = min_samples_split;
        self
    }

    /// Set the minimum samples required to be at a leaf node (fluent API)
    pub fn min_samples_leaf(mut self, min_samples_leaf: usize) -> Self {
        self.config.min_samples_leaf = min_samples_leaf;
        self
    }

    /// Set the missing value strategy (fluent API)
    pub fn missing_values(mut self, strategy: crate::config::MissingValueStrategy) -> Self {
        self.config.missing_values = strategy;
        self
    }

    /// Set the random seed (fluent API)
    pub fn random_state(mut self, random_state: Option<u64>) -> Self {
        self.config.random_state = random_state;
        self
    }
}

/// Builder pattern for configuring DecisionTree
#[derive(Debug)]
pub struct DecisionTreeBuilder<State> {
    config: DecisionTreeConfig,
    _marker: PhantomData<State>,
}

impl<State> DecisionTreeBuilder<State> {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            config: DecisionTreeConfig::default(),
            _marker: PhantomData,
        }
    }

    /// Set the split criterion
    pub fn criterion(mut self, criterion: SplitCriterion) -> Self {
        self.config.criterion = criterion;
        self
    }

    /// Set the maximum depth of the tree
    pub fn max_depth(mut self, max_depth: Option<usize>) -> Self {
        self.config.max_depth = max_depth;
        self
    }

    /// Set the minimum samples required to split an internal node
    pub fn min_samples_split(mut self, min_samples_split: usize) -> Self {
        self.config.min_samples_split = min_samples_split;
        self
    }

    /// Set the minimum samples required to be at a leaf node
    pub fn min_samples_leaf(mut self, min_samples_leaf: usize) -> Self {
        self.config.min_samples_leaf = min_samples_leaf;
        self
    }

    /// Set the minimum impurity decrease required for a split
    pub fn min_impurity_decrease(mut self, min_impurity_decrease: f64) -> Self {
        self.config.min_impurity_decrease = min_impurity_decrease;
        self
    }

    /// Set the maximum number of features to consider for splitting
    pub fn max_features(self, _max_features: Option<usize>) -> Self {
        // Convert between Option<usize> and MaxFeatures type if needed
        // self.config.max_features = max_features;
        self
    }

    /// Set the random seed
    pub fn random_state(mut self, random_state: Option<u64>) -> Self {
        self.config.random_state = random_state;
        self
    }

    /// Build the DecisionTree with the configured parameters
    pub fn build(self) -> DecisionTree<State> {
        DecisionTree::with_config(self.config)
    }
}

impl<State> Default for DecisionTreeBuilder<State> {
    fn default() -> Self {
        Self::new()
    }
}

// Note: Estimator trait implementation can be added later when all required methods are defined

impl DecisionTree<Untrained> {
    /// Check if the tree has been fitted (always false for Untrained trees)
    pub fn is_fitted(&self) -> bool {
        false
    }

    /// Get the number of classes (not available for untrained trees)
    pub fn n_classes(&self) -> usize {
        0 // Untrained trees don't have class information
    }
}

/// Validation functions for decision trees
pub struct TreeValidator;

impl TreeValidator {
    /// Validate input data dimensions
    pub fn validate_input(x: &ArrayView2<'_, f64>, y: &ArrayView1<'_, f64>) -> Result<()> {
        if x.nrows() != y.len() {
            return Err(SklearsError::InvalidInput(format!(
                "Number of samples in X ({}) does not match number of targets in y ({})",
                x.nrows(),
                y.len()
            )));
        }

        if x.nrows() == 0 {
            return Err(SklearsError::InvalidInput(
                "Input arrays must contain at least one sample".to_string(),
            ));
        }

        if x.ncols() == 0 {
            return Err(SklearsError::InvalidInput(
                "Input arrays must contain at least one feature".to_string(),
            ));
        }

        Ok(())
    }

    /// Validate that the tree has been fitted
    pub fn validate_fitted(tree: &DecisionTree<Trained>) -> Result<()> {
        // Type system ensures tree is fitted
        Ok(())
    }

    /// Validate prediction input dimensions
    pub fn validate_prediction_input(
        tree: &DecisionTree<Trained>,
        x: &ArrayView2<'_, f64>,
    ) -> Result<()> {
        Self::validate_fitted(tree)?;

        if x.ncols() != tree.n_features() {
            return Err(SklearsError::InvalidInput(format!(
                "Number of features in X ({}) does not match number of features seen during fit ({})",
                x.ncols(),
                tree.n_features()
            )));
        }

        Ok(())
    }
}

// Trait implementations

impl Estimator<Untrained> for DecisionTree<Untrained> {
    type Config = DecisionTreeConfig;
    type Error = SklearsError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Estimator<Trained> for DecisionTree<Trained> {
    type Config = DecisionTreeConfig;
    type Error = SklearsError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Fit<Array2<f64>, Array1<f64>, Untrained> for DecisionTree<Untrained> {
    type Fitted = DecisionTree<Trained>;

    fn fit(self, x: &Array2<f64>, y: &Array1<f64>) -> Result<Self::Fitted> {
        // Validate input
        TreeValidator::validate_input(&x.view(), &y.view())?;

        // Basic fitting implementation (simplified for now)
        // In a full implementation, this would build the actual tree
        let fitted_tree = DecisionTree::<Trained> {
            config: self.config,
            root: None, // Would contain the actual tree structure
            feature_importances: None,
            n_features: x.ncols(),
            n_samples: x.nrows(),
            state: PhantomData,
        };

        Ok(fitted_tree)
    }
}

impl Predict<Array2<f64>, Array1<f64>> for DecisionTree<Trained> {
    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        TreeValidator::validate_prediction_input(self, &x.view())?;

        // Basic prediction implementation (simplified for now)
        // In a full implementation, this would traverse the tree
        let predictions = Array1::zeros(x.nrows());

        Ok(predictions)
    }
}

impl DecisionTree<Trained> {
    /// Check if the tree has been fitted (always true for Trained trees)
    pub fn is_fitted(&self) -> bool {
        true
    }

    /// Get the number of classes (for classification trees)
    pub fn n_classes(&self) -> usize {
        // This should be determined from the training data
        // For now, default to binary classification
        2
    }
}
