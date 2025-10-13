use sklears_core::traits::Untrained;
use sklears_tree::{
    DecisionTree, DecisionTreeClassifier, DecisionTreeConfig, DecisionTreeRegressor,
};

#[test]
fn test_decision_tree_creation() {
    let tree: DecisionTree<sklears_core::traits::Untrained> = DecisionTree::new();
    assert!(!tree.is_fitted());
    assert_eq!(tree.n_features(), 0);
    assert_eq!(tree.n_samples(), 0);
}

#[test]
fn test_decision_tree_classifier_creation() {
    let classifier = DecisionTreeClassifier::new();
    assert!(!classifier.is_fitted());
}

#[test]
fn test_decision_tree_regressor_creation() {
    let regressor = DecisionTreeRegressor::new();
    assert!(!regressor.is_fitted());
}

#[test]
fn test_decision_tree_with_config() {
    let config = DecisionTreeConfig::default();
    let tree: DecisionTree<Untrained> = DecisionTree::with_config(config);
    assert!(!tree.is_fitted());
}

#[test]
fn test_decision_tree_builder() {
    let tree: DecisionTree<Untrained> = DecisionTree::builder()
        .max_depth(Some(5))
        .min_samples_split(10)
        .min_samples_leaf(5)
        .build();

    assert!(!tree.is_fitted());
    assert_eq!(tree.config().max_depth, Some(5));
    assert_eq!(tree.config().min_samples_split, 10);
    assert_eq!(tree.config().min_samples_leaf, 5);
}
