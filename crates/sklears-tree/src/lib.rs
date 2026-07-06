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
pub mod classifier;
pub mod config;
pub mod criteria;
pub mod decision_tree;
pub mod node;
pub mod regressor;
pub mod splits;

// Extended modules
pub mod extra_trees_enhanced;
pub mod incremental;
pub mod isolation_forest;
pub mod model_tree;
pub mod multi_output;
pub mod parallel;
pub mod random_forest;
pub mod shap;

// Essential re-exports for the main API
pub use classifier::DecisionTreeClassifier;
pub use config::{ndarray_to_dense_matrix, DecisionTreeConfig, MaxFeatures, MissingValueStrategy};
pub use criteria::{ConditionalTestType, FeatureType, MonotonicConstraint, SplitCriterion};
pub use decision_tree::{DecisionTree, DecisionTreeBuilder, TreeValidator};
pub use extra_trees_enhanced::{
    BinningStrategy, FeatureBinning, RandomizationStrategy, SparseCompression, SparseConfig,
    SparseFeature,
};
pub use isolation_forest::{
    IsolationForest, IsolationForestConfig, MaxSamples, StreamingIsolationForest,
};
pub use model_tree::{LeafModelType, ModelTree, ModelTreeConfig, ModelTreeNode};
pub use node::{CompactTreeNode, CustomSplit, SurrogateSplit, TreeNode};
pub use random_forest::RandomForestClassifier;
pub use regressor::DecisionTreeRegressor;
pub use sklears_core::traits::{Trained, Untrained};
pub use splits::HyperplaneSplit;

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::classifier::DecisionTreeClassifier;
    pub use crate::config::DecisionTreeConfig;
    pub use crate::criteria::SplitCriterion;
    pub use crate::decision_tree::DecisionTree;
    pub use crate::extra_trees_enhanced::{BinningStrategy, FeatureBinning, RandomizationStrategy};
    pub use crate::isolation_forest::{
        IsolationForest, IsolationForestConfig, StreamingIsolationForest,
    };
    pub use crate::model_tree::{LeafModelType, ModelTree, ModelTreeConfig};
    pub use crate::regressor::DecisionTreeRegressor;
}
