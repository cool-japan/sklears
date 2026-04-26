//! Hierarchical discriminant analysis tests

use super::super::*;
use approx::assert_abs_diff_eq;
use scirs2_core::ndarray::{array, Axis};
use sklears_core::traits::{Fit, Predict, PredictProba};
use sklears_core::types::Float;

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
    let fitted = hda.fit(&x, &y).expect("model fitting should succeed");
    let predictions = fitted.predict(&x).expect("prediction should succeed");

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
    let fitted = hda.fit(&x, &y).expect("model fitting should succeed");
    let probas = fitted
        .predict_proba(&x)
        .expect("probability prediction should succeed");

    assert_eq!(probas.dim(), (6, 3));

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
    let fitted = hda.fit(&x, &y).expect("model fitting should succeed");
    let predictions = fitted.predict(&x).expect("prediction should succeed");

    assert_eq!(predictions.len(), 4);
    assert_eq!(fitted.classes().len(), 2);
}

#[test]
fn test_hierarchical_min_samples_split() {
    let x = array![[1.0, 2.0], [1.1, 2.1], [3.0, 4.0], [3.1, 4.1]];
    let y = array![0, 0, 1, 1];

    let hda = HierarchicalDiscriminantAnalysis::new().min_samples_split(3);
    let fitted = hda.fit(&x, &y).expect("model fitting should succeed");
    let predictions = fitted.predict(&x).expect("prediction should succeed");

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

    let fitted = hda.fit(&x, &y).expect("model fitting should succeed");
    let predictions = fitted.predict(&x).expect("prediction should succeed");

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
        let fitted = hda.fit(&x, &y).expect("model fitting should succeed");
        let predictions = fitted.predict(&x).expect("prediction should succeed");

        assert_eq!(predictions.len(), 6);
        assert_eq!(fitted.classes().len(), 3);
    }
}
