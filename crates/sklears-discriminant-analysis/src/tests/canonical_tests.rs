//! Canonical discriminant analysis tests

use super::super::*;
use approx::assert_abs_diff_eq;
use scirs2_core::ndarray::{array, Axis};
use sklears_core::traits::{Fit, Predict, PredictProba, Transform};
use sklears_core::types::Float;

#[test]
fn test_canonical_discriminant_analysis_basic() {
    let x = array![
        [1.0, 2.0, 3.0],
        [1.1, 2.1, 3.1],
        [1.2, 2.2, 3.2],
        [4.0, 5.0, 6.0],
        [4.1, 5.1, 6.1],
        [4.2, 5.2, 6.2]
    ];
    let y = array![0, 0, 0, 1, 1, 1];

    let cda = CanonicalDiscriminantAnalysis::new();
    let fitted = cda.fit(&x, &y).expect("model fitting should succeed");
    let predictions = fitted.predict(&x).expect("prediction should succeed");

    assert_eq!(predictions.len(), 6);
    assert_eq!(fitted.classes().len(), 2);
    assert_eq!(fitted.coefficients().nrows(), 3);
}

#[test]
fn test_canonical_discriminant_predict_proba() {
    let x = array![[1.0, 2.0], [1.1, 2.1], [3.0, 4.0], [3.1, 4.1]];
    let y = array![0, 0, 1, 1];

    let cda = CanonicalDiscriminantAnalysis::new();
    let fitted = cda.fit(&x, &y).expect("model fitting should succeed");
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
fn test_canonical_discriminant_transform() {
    let x = array![
        [1.0, 2.0, 3.0],
        [1.1, 2.1, 3.1],
        [4.0, 5.0, 6.0],
        [4.1, 5.1, 6.1]
    ];
    let y = array![0, 0, 1, 1];

    let cda = CanonicalDiscriminantAnalysis::new().n_components(Some(1));
    let fitted = cda.fit(&x, &y).expect("model fitting should succeed");
    let transformed = fitted.transform(&x).expect("transform should succeed");

    assert_eq!(transformed.dim(), (4, 1));
}

#[test]
fn test_canonical_discriminant_multiclass() {
    let x = array![
        [1.0, 2.0],
        [1.1, 2.1],
        [3.0, 4.0],
        [3.1, 4.1],
        [5.0, 6.0],
        [5.1, 6.1]
    ];
    let y = array![0, 0, 1, 1, 2, 2];

    let cda = CanonicalDiscriminantAnalysis::new();
    let fitted = cda.fit(&x, &y).expect("model fitting should succeed");
    let predictions = fitted.predict(&x).expect("prediction should succeed");
    let probas = fitted
        .predict_proba(&x)
        .expect("probability prediction should succeed");

    assert_eq!(predictions.len(), 6);
    assert_eq!(fitted.classes().len(), 3);
    assert_eq!(probas.dim(), (6, 3));

    for row in probas.axis_iter(Axis(0)) {
        let sum: Float = row.sum();
        assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-6);
    }
}

#[test]
fn test_canonical_discriminant_no_standardization() {
    let x = array![[1.0, 2.0], [1.1, 2.1], [3.0, 4.0], [3.1, 4.1]];
    let y = array![0, 0, 1, 1];

    let cda = CanonicalDiscriminantAnalysis::new().standardize(false);
    let fitted = cda.fit(&x, &y).expect("model fitting should succeed");
    let predictions = fitted.predict(&x).expect("prediction should succeed");

    assert_eq!(predictions.len(), 4);
    assert_eq!(fitted.classes().len(), 2);
}

#[test]
fn test_canonical_discriminant_with_regularization() {
    let x = array![[1.0, 2.0], [1.1, 2.1], [3.0, 4.0], [3.1, 4.1]];
    let y = array![0, 0, 1, 1];

    let cda = CanonicalDiscriminantAnalysis::new().reg_param(0.1);
    let fitted = cda.fit(&x, &y).expect("model fitting should succeed");
    let predictions = fitted.predict(&x).expect("prediction should succeed");

    assert_eq!(predictions.len(), 4);
    assert_eq!(fitted.classes().len(), 2);
}

#[test]
fn test_canonical_correlations() {
    let x = array![
        [1.0, 2.0, 3.0],
        [1.1, 2.1, 3.1],
        [4.0, 5.0, 6.0],
        [4.1, 5.1, 6.1]
    ];
    let y = array![0, 0, 1, 1];

    let cda = CanonicalDiscriminantAnalysis::new();
    let fitted = cda.fit(&x, &y).expect("model fitting should succeed");
    let correlations = fitted.canonical_correlations();

    assert_eq!(correlations.len(), fitted.eigenvalues().len());
    for &corr in correlations.iter() {
        assert!((0.0..=1.0).contains(&corr));
    }
}
