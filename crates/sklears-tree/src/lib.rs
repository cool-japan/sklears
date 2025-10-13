#![allow(dead_code)]
#![allow(non_snake_case)]
#![allow(missing_docs)]
#![allow(deprecated)]
//! Tree-based algorithms for sklears
//!
//! This crate provides implementations of tree-based machine learning algorithms including:
//! - Decision Trees (CART algorithm)
//! - Random Forest
//! - Extra Trees
//!
//! This is a simplified version focusing on core functionality.

// Core modules
pub mod builder;
pub mod config;
pub mod criteria;
pub mod decision_tree;
pub mod node;
pub mod splits;

// Extended modules
// pub mod incremental; // Temporarily disabled due to advanced features
pub mod parallel;
pub mod random_forest;
// pub mod shap; // Temporarily disabled

// Essential re-exports for the main API
pub use config::{ndarray_to_dense_matrix, DecisionTreeConfig, MaxFeatures, MissingValueStrategy};
pub use criteria::{ConditionalTestType, FeatureType, MonotonicConstraint, SplitCriterion};
pub use decision_tree::{
    DecisionTree, DecisionTreeBuilder, DecisionTreeClassifier, DecisionTreeRegressor, TreeValidator,
};
pub use node::{CompactTreeNode, CustomSplit, SurrogateSplit, TreeNode};
pub use random_forest::RandomForestClassifier;
pub use sklears_core::traits::{Trained, Untrained};
pub use splits::HyperplaneSplit;

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::config::DecisionTreeConfig;
    pub use crate::criteria::SplitCriterion;
    pub use crate::decision_tree::{DecisionTree, DecisionTreeClassifier, DecisionTreeRegressor};
}
