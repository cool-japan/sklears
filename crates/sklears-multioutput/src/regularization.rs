//! Multi-Task Regularization Methods
//!
//! This module provides various regularization techniques specifically designed for multi-task
//! and multi-output learning scenarios. These methods help in learning shared structure
//! across tasks while preventing overfitting.
//!
//! The module has been refactored into smaller submodules to comply with the 2000-line limit:
//!
//! - [`simd_ops`] - SIMD-accelerated operations for high-performance regularization computations
//! - [`group_lasso`] - Group Lasso regularization for feature group selection
//! - [`nuclear_norm`] - Nuclear norm regularization for low-rank structure learning
//! - [`task_clustering`] - Task clustering regularization for similar task grouping
//! - [`task_relationship`] - Task relationship learning for explicit task relationships
//! - [`meta_learning`] - Meta-learning approach for quick adaptation to new tasks

// Use SciRS2-Core for arrays and random number generation (SciRS2 Policy)
use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::{traits::Untrained, types::Float};
use std::collections::HashMap;

// Submodules
#[path = "regularization/simd_ops.rs"]
pub mod simd_ops;

#[path = "regularization/group_lasso.rs"]
pub mod group_lasso;

#[path = "regularization/nuclear_norm.rs"]
pub mod nuclear_norm;

#[path = "regularization/task_clustering.rs"]
pub mod task_clustering;

#[path = "regularization/task_relationship.rs"]
pub mod task_relationship;

#[path = "regularization/meta_learning.rs"]
pub mod meta_learning;

// Re-export the main types from submodules
pub use group_lasso::{GroupLasso, GroupLassoTrained};
pub use meta_learning::{MetaLearningMultiTask, MetaLearningMultiTaskTrained};
pub use nuclear_norm::{NuclearNormRegression, NuclearNormRegressionTrained};
pub use task_clustering::{TaskClusteringRegressionTrained, TaskClusteringRegularization};
pub use task_relationship::{
    TaskRelationshipLearning, TaskRelationshipLearningTrained, TaskSimilarityMethod,
};

/// Multi-Task Elastic Net with Group Structure
///
/// Combines L1 and L2 regularization with group structure awareness.
/// Useful for scenarios where we want both feature selection and group selection.
///
/// Note: This is a placeholder struct - implementation is not yet complete.
#[derive(Debug, Clone)]
pub struct MultiTaskElasticNet<S = Untrained> {
    state: S,
    /// L1 regularization strength
    alpha: Float,
    /// L1 vs L2 balance (0 = Ridge, 1 = Lasso)
    l1_ratio: Float,
    /// Feature groups for group penalties
    feature_groups: Vec<Vec<usize>>,
    /// Group penalty strength
    group_alpha: Float,
    /// Maximum number of iterations
    max_iter: usize,
    /// Convergence tolerance
    tolerance: Float,
    /// Learning rate
    learning_rate: Float,
    /// Task configurations
    task_outputs: HashMap<String, usize>,
    /// Include intercept term
    fit_intercept: bool,
}

/// Trained state for MultiTaskElasticNet
///
/// Note: This is a placeholder struct - implementation is not yet complete.
#[derive(Debug, Clone)]
pub struct MultiTaskElasticNetTrained {
    /// Coefficients for each task
    coefficients: HashMap<String, Array2<Float>>,
    /// Intercepts for each task
    intercepts: HashMap<String, Array1<Float>>,
    /// Number of input features
    n_features: usize,
    /// Task configurations
    task_outputs: HashMap<String, usize>,
    /// Training parameters
    alpha: Float,
    l1_ratio: Float,
    group_alpha: Float,
    /// Training iterations performed
    n_iter: usize,
}

/// Regularization strategies for multi-task learning
#[derive(Debug, Clone, PartialEq, Default)]
pub enum RegularizationStrategy {
    /// No regularization
    #[default]
    None,
    /// L1 regularization (Lasso)
    L1(Float),
    /// L2 regularization (Ridge)
    L2(Float),
    /// Elastic Net (L1 + L2)
    ElasticNet { alpha: Float, l1_ratio: Float },
    /// Group Lasso
    GroupLasso { alpha: Float },
    /// Nuclear norm regularization
    NuclearNorm { alpha: Float },
    /// Task clustering regularization
    TaskClustering {
        n_clusters: usize,
        intra_cluster_alpha: Float,
        inter_cluster_alpha: Float,
    },
    /// Task relationship learning
    TaskRelationship {
        relationship_strength: Float,
        similarity_threshold: Float,
    },
    /// Meta-learning for multi-task
    MetaLearning {
        meta_learning_rate: Float,
        inner_learning_rate: Float,
        n_inner_steps: usize,
    },
}

// Keep the tests in the main module for backwards compatibility
#[allow(non_snake_case)]
#[cfg(test)]
mod regularization_tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    // Use SciRS2-Core for arrays and random number generation (SciRS2 Policy)
    use scirs2_core::ndarray::array;
    use sklears_core::traits::{Fit, Predict};
    use std::collections::HashMap;

    #[test]
    fn test_group_lasso_creation() {
        let group_lasso = GroupLasso::new()
            .alpha(0.1)
            .feature_groups(vec![vec![0, 1], vec![2, 3]])
            .max_iter(100)
            .tolerance(1e-6)
            .learning_rate(0.01);

        assert_eq!(group_lasso.alpha, 0.1);
        assert_eq!(group_lasso.feature_groups, vec![vec![0, 1], vec![2, 3]]);
        assert_eq!(group_lasso.max_iter, 100);
        assert_abs_diff_eq!(group_lasso.tolerance, 1e-6);
        assert_abs_diff_eq!(group_lasso.learning_rate, 0.01);
    }

    #[test]
    fn test_group_lasso_fit_predict() {
        let X = array![
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 3.0, 4.0, 5.0],
            [3.0, 1.0, 2.0, 3.0],
            [4.0, 2.0, 1.0, 2.0]
        ];

        let mut y_tasks = HashMap::new();
        y_tasks.insert("task1".to_string(), array![[1.0], [2.0], [1.5], [2.5]]);
        y_tasks.insert("task2".to_string(), array![[0.5], [1.0], [0.8], [1.2]]);

        let feature_groups = vec![vec![0, 1], vec![2, 3]];

        let group_lasso = GroupLasso::new()
            .alpha(0.01)
            .feature_groups(feature_groups)
            .task_outputs(&[("task1", 1), ("task2", 1)])
            .max_iter(50)
            .tolerance(1e-4)
            .learning_rate(0.01);

        let trained = group_lasso.fit(&X.view(), &y_tasks).unwrap();

        // Test predictions
        let predictions = trained.predict(&X.view()).unwrap();
        assert!(predictions.contains_key("task1"));
        assert!(predictions.contains_key("task2"));

        let task1_pred = &predictions["task1"];
        let task2_pred = &predictions["task2"];

        assert_eq!(task1_pred.shape(), &[4, 1]);
        assert_eq!(task2_pred.shape(), &[4, 1]);

        // Test group sparsity
        let sparsity = trained.group_sparsity();
        assert!(sparsity >= 0.0 && sparsity <= 1.0); // Should be a percentage

        // Test accessors
        assert!(trained.task_coefficients("task1").is_some());
        assert!(trained.task_intercepts("task1").is_some());
        assert!(trained.n_iter() <= 50);
    }

    #[test]
    fn test_nuclear_norm_regression_creation() {
        let nuclear_norm = NuclearNormRegression::new()
            .alpha(0.1)
            .max_iter(100)
            .tolerance(1e-6)
            .learning_rate(0.01)
            .target_rank(Some(5));

        assert_eq!(nuclear_norm.alpha, 0.1);
        assert_eq!(nuclear_norm.max_iter, 100);
        assert_abs_diff_eq!(nuclear_norm.tolerance, 1e-6);
        assert_abs_diff_eq!(nuclear_norm.learning_rate, 0.01);
        assert_eq!(nuclear_norm.target_rank, Some(5));
    }

    #[test]
    fn test_nuclear_norm_regression_fit_predict() {
        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 1.0], [4.0, 4.0]];

        let mut y_tasks = HashMap::new();
        y_tasks.insert("task1".to_string(), array![[1.0], [2.0], [1.5], [2.5]]);
        y_tasks.insert("task2".to_string(), array![[0.5], [1.0], [0.8], [1.2]]);

        let nuclear_norm = NuclearNormRegression::new()
            .alpha(0.01)
            .task_outputs(&[("task1", 1), ("task2", 1)])
            .max_iter(50)
            .tolerance(1e-4)
            .learning_rate(0.01);

        let trained = nuclear_norm.fit(&X.view(), &y_tasks).unwrap();

        // Test predictions
        let predictions = trained.predict(&X.view()).unwrap();
        assert!(predictions.contains_key("task1"));
        assert!(predictions.contains_key("task2"));

        let task1_pred = &predictions["task1"];
        let task2_pred = &predictions["task2"];

        assert_eq!(task1_pred.shape(), &[4, 1]);
        assert_eq!(task2_pred.shape(), &[4, 1]);

        // Test accessors
        assert!(trained.task_coefficient_matrix("task1").is_some());
        assert!(trained.effective_rank() >= 0);
        assert!(!trained.singular_values().is_empty());
        assert!(trained.n_iter() <= 50);
    }

    #[test]
    fn test_regularization_strategies() {
        let strategies = vec![
            RegularizationStrategy::None,
            RegularizationStrategy::L1(0.1),
            RegularizationStrategy::L2(0.1),
            RegularizationStrategy::ElasticNet {
                alpha: 0.1,
                l1_ratio: 0.5,
            },
            RegularizationStrategy::GroupLasso { alpha: 0.1 },
            RegularizationStrategy::NuclearNorm { alpha: 0.1 },
            RegularizationStrategy::TaskClustering {
                n_clusters: 2,
                intra_cluster_alpha: 0.1,
                inter_cluster_alpha: 0.01,
            },
            RegularizationStrategy::TaskRelationship {
                relationship_strength: 0.1,
                similarity_threshold: 0.5,
            },
            RegularizationStrategy::MetaLearning {
                meta_learning_rate: 0.01,
                inner_learning_rate: 0.1,
                n_inner_steps: 5,
            },
        ];

        assert_eq!(strategies.len(), 9);
        assert_eq!(strategies[0], RegularizationStrategy::None);
        assert_eq!(strategies[1], RegularizationStrategy::L1(0.1));
    }

    #[test]
    fn test_task_similarity_methods() {
        let methods = vec![
            TaskSimilarityMethod::Correlation,
            TaskSimilarityMethod::Cosine,
            TaskSimilarityMethod::Euclidean,
            TaskSimilarityMethod::MutualInformation,
        ];

        assert_eq!(methods.len(), 4);
        assert_eq!(methods[0], TaskSimilarityMethod::Correlation);
        assert_eq!(methods[1], TaskSimilarityMethod::Cosine);
    }
}
