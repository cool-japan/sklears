//! Mixture discriminant analysis tests

use super::super::*;
use approx::assert_abs_diff_eq;
use scirs2_core::ndarray::{array, Axis};
use sklears_core::traits::{Fit, Predict, PredictProba};
use sklears_core::types::Float;

#[test]
fn test_mixture_discriminant_analysis_basic() {
    let x = array![
        [1.0, 2.0],
        [1.1, 2.1],
        [1.2, 2.2],
        [3.0, 4.0],
        [3.1, 4.1],
        [3.2, 4.2]
    ];
    let y = array![0, 0, 0, 1, 1, 1];

    let mda = MixtureDiscriminantAnalysis::new().n_components_per_class(1);
    let fitted = mda.fit(&x, &y).expect("model fitting should succeed");
    let predictions = fitted.predict(&x).expect("prediction should succeed");

    assert_eq!(predictions.len(), 6);
    assert_eq!(fitted.classes().len(), 2);
}

#[test]
fn test_mixture_discriminant_predict_proba() {
    let x = array![[1.0, 2.0], [1.1, 2.1], [3.0, 4.0], [3.1, 4.1]];
    let y = array![0, 0, 1, 1];

    let mda = MixtureDiscriminantAnalysis::new().n_components_per_class(1);
    let fitted = mda.fit(&x, &y).expect("model fitting should succeed");
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
fn test_mixture_discriminant_multiple_components() {
    let x = array![
        [1.0, 2.0],
        [1.1, 2.1],
        [1.5, 3.0],
        [1.6, 3.1],
        [3.0, 4.0],
        [3.1, 4.1],
        [3.5, 5.0],
        [3.6, 5.1]
    ];
    let y = array![0, 0, 0, 0, 1, 1, 1, 1];

    let mda = MixtureDiscriminantAnalysis::new().n_components_per_class(2);
    let fitted = mda.fit(&x, &y).expect("model fitting should succeed");
    let predictions = fitted.predict(&x).expect("prediction should succeed");

    assert_eq!(predictions.len(), 8);
    assert_eq!(fitted.classes().len(), 2);
}

#[test]
fn test_mixture_discriminant_multiclass() {
    let x = array![
        [1.0, 2.0],
        [1.1, 2.1],
        [3.0, 4.0],
        [3.1, 4.1],
        [5.0, 6.0],
        [5.1, 6.1]
    ];
    let y = array![0, 0, 1, 1, 2, 2];

    let mda = MixtureDiscriminantAnalysis::new().n_components_per_class(1);
    let fitted = mda.fit(&x, &y).expect("model fitting should succeed");
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
fn test_mixture_discriminant_with_diagonal_covariance() {
    let x = array![
        [1.0, 2.0],
        [1.1, 2.1],
        [1.2, 2.2],
        [3.0, 4.0],
        [3.1, 4.1],
        [3.2, 4.2]
    ];
    let y = array![0, 0, 0, 1, 1, 1];

    let mda = MixtureDiscriminantAnalysis::new()
        .n_components_per_class(1)
        .diagonal_covariance(true);
    let fitted = mda.fit(&x, &y).expect("model fitting should succeed");
    let predictions = fitted.predict(&x).expect("prediction should succeed");

    assert_eq!(predictions.len(), 6);
    assert_eq!(fitted.classes().len(), 2);
}

#[test]
fn test_mixture_discriminant_with_regularization() {
    let x = array![[1.0, 2.0], [1.1, 2.1], [3.0, 4.0], [3.1, 4.1]];
    let y = array![0, 0, 1, 1];

    let mda = MixtureDiscriminantAnalysis::new()
        .n_components_per_class(1)
        .reg_param(0.1);
    let fitted = mda.fit(&x, &y).expect("model fitting should succeed");
    let predictions = fitted.predict(&x).expect("prediction should succeed");

    assert_eq!(predictions.len(), 4);
    assert_eq!(fitted.classes().len(), 2);
}

#[test]
fn test_mixture_discriminant_weights_and_means() {
    let x = array![[1.0, 2.0], [1.1, 2.1], [3.0, 4.0], [3.1, 4.1]];
    let y = array![0, 0, 1, 1];

    let mda = MixtureDiscriminantAnalysis::new().n_components_per_class(1);
    let fitted = mda.fit(&x, &y).expect("model fitting should succeed");

    let weights = fitted.mixture_weights();
    let means = fitted.mixture_means();

    assert_eq!(weights.len(), 2); // Two classes
    assert_eq!(means.len(), 2); // Two classes
    for w in weights {
        let sum: Float = w.sum();
        assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-6);
    }
}
