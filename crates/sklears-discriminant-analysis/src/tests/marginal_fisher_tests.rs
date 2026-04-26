//! Marginal fisher analysis tests

use super::super::*;
use approx::assert_abs_diff_eq;
use scirs2_core::ndarray::{array, Axis};
use sklears_core::traits::{Fit, Predict, PredictProba, Transform};
use sklears_core::types::Float;

#[test]
fn test_marginal_fisher_analysis_basic() {
    let x = array![
        [1.0, 2.0],
        [1.1, 2.1],
        [1.2, 2.2],
        [3.0, 4.0],
        [3.1, 4.1],
        [3.2, 4.2]
    ];
    let y = array![0, 0, 0, 1, 1, 1];

    let mfa = MarginalFisherAnalysis::new().k1(2).k2(2);
    let fitted = mfa.fit(&x, &y).expect("model fitting should succeed");
    let predictions = fitted.predict(&x).expect("prediction should succeed");

    assert_eq!(predictions.len(), 6);
    assert_eq!(fitted.classes().len(), 2);
}

#[test]
fn test_marginal_fisher_transform() {
    let x = array![
        [1.0, 2.0, 3.0],
        [1.1, 2.1, 3.1],
        [1.2, 2.2, 3.2],
        [3.0, 4.0, 5.0],
        [3.1, 4.1, 5.1],
        [3.2, 4.2, 5.2]
    ];
    let y = array![0, 0, 0, 1, 1, 1];

    let mfa = MarginalFisherAnalysis::new()
        .k1(2)
        .k2(2)
        .n_components(Some(1));
    let fitted = mfa.fit(&x, &y).expect("model fitting should succeed");
    let transformed = fitted.transform(&x).expect("transform should succeed");

    assert_eq!(transformed.nrows(), 6);
    assert_eq!(transformed.ncols(), 1);
}

#[test]
fn test_marginal_fisher_predict_proba() {
    let x = array![[1.0, 2.0], [1.1, 2.1], [3.0, 4.0], [3.1, 4.1]];
    let y = array![0, 0, 1, 1];

    let mfa = MarginalFisherAnalysis::new().k1(2).k2(2);
    let fitted = mfa.fit(&x, &y).expect("model fitting should succeed");
    let probas = fitted
        .predict_proba(&x)
        .expect("probability prediction should succeed");

    assert_eq!(probas.dim(), (4, 2));

    for row in probas.axis_iter(Axis(0)) {
        let sum: Float = row.sum();
        assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-6);
    }
}

#[test]
fn test_marginal_fisher_multiclass() {
    let x = array![
        [1.0, 2.0],
        [1.1, 2.1],
        [3.0, 4.0],
        [3.1, 4.1],
        [5.0, 6.0],
        [5.1, 6.1]
    ];
    let y = array![0, 0, 1, 1, 2, 2];

    let mfa = MarginalFisherAnalysis::new().k1(2).k2(2);
    let fitted = mfa.fit(&x, &y).expect("model fitting should succeed");
    let predictions = fitted.predict(&x).expect("prediction should succeed");

    assert_eq!(predictions.len(), 6);
    assert_eq!(fitted.classes().len(), 3);
}

#[test]
fn test_marginal_fisher_with_regularization() {
    let x = array![[1.0, 2.0], [1.1, 2.1], [3.0, 4.0], [3.1, 4.1]];
    let y = array![0, 0, 1, 1];

    let mfa = MarginalFisherAnalysis::new().k1(2).k2(2).reg_param(0.1);
    let fitted = mfa.fit(&x, &y).expect("model fitting should succeed");
    let predictions = fitted.predict(&x).expect("prediction should succeed");

    assert_eq!(predictions.len(), 4);
    assert_eq!(fitted.classes().len(), 2);
}

#[test]
fn test_marginal_fisher_intra_inter_graphs() {
    let x = array![[1.0, 2.0], [1.1, 2.1], [3.0, 4.0], [3.1, 4.1]];
    let y = array![0, 0, 1, 1];

    let mfa = MarginalFisherAnalysis::new().k1(2).k2(2);
    let fitted = mfa.fit(&x, &y).expect("model fitting should succeed");

    let intra_graph = fitted.intra_class_graph();
    let inter_graph = fitted.inter_class_graph();

    assert_eq!(intra_graph.nrows(), 4);
    assert_eq!(intra_graph.ncols(), 4);
    assert_eq!(inter_graph.nrows(), 4);
    assert_eq!(inter_graph.ncols(), 4);
}

#[test]
fn test_marginal_fisher_components_and_eigenvalues() {
    let x = array![[1.0, 2.0], [1.1, 2.1], [3.0, 4.0], [3.1, 4.1]];
    let y = array![0, 0, 1, 1];

    let mfa = MarginalFisherAnalysis::new().k1(2).k2(2);
    let fitted = mfa.fit(&x, &y).expect("model fitting should succeed");

    let components = fitted.components();
    let eigenvalues = fitted.eigenvalues();

    assert!(components.nrows() > 0);
    assert_eq!(eigenvalues.len(), components.ncols());
}
