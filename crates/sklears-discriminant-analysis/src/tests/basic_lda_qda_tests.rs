//! Basic Linear and Quadratic Discriminant Analysis tests

use super::test_utils::*;

#[test]
fn test_linear_discriminant_analysis() {
    let (x, y) = create_simple_2d_data();

    let lda = LinearDiscriminantAnalysis::new();
    let fitted = lda.fit(&x, &y).unwrap();
    let predictions = fitted.predict(&x).unwrap();

    assert_eq!(predictions.len(), 4);
    assert_eq!(fitted.classes().len(), 2);
}

#[test]
fn test_lda_predict_proba() {
    let (x, y) = create_simple_2d_data();

    let lda = LinearDiscriminantAnalysis::new();
    let fitted = lda.fit(&x, &y).unwrap();
    let probas = fitted.predict_proba(&x).unwrap();

    assert_eq!(probas.dim(), (4, 2));
    assert_probabilities_sum_to_one(&probas);
}

#[test]
fn test_quadratic_discriminant_analysis() {
    let (x, y) = create_simple_2d_data();

    let qda = QuadraticDiscriminantAnalysis::new();
    let fitted = qda.fit(&x, &y).unwrap();
    let predictions = fitted.predict(&x).unwrap();

    assert_eq!(predictions.len(), 4);
    assert_eq!(fitted.classes().len(), 2);
}

#[test]
fn test_qda_predict_proba() {
    let (x, y) = create_simple_2d_data();

    let qda = QuadraticDiscriminantAnalysis::new();
    let fitted = qda.fit(&x, &y).unwrap();
    let probas = fitted.predict_proba(&x).unwrap();

    assert_eq!(probas.dim(), (4, 2));
    assert_probabilities_sum_to_one(&probas);
}

#[test]
fn test_lda_transform() {
    let (x, y) = create_simple_3d_data();

    let lda = LinearDiscriminantAnalysis::new().n_components(Some(1));
    let fitted = lda.fit(&x, &y).unwrap();
    let transformed = fitted.transform(&x).unwrap();

    assert_eq!(transformed.dim(), (4, 1));
}

#[test]
fn test_diagonal_qda() {
    let (x, y) = create_simple_2d_data();

    let qda = QuadraticDiscriminantAnalysis::new().diagonal_covariance(true);
    let fitted = qda.fit(&x, &y).unwrap();
    let predictions = fitted.predict(&x).unwrap();

    assert_eq!(predictions.len(), 4);
    assert_eq!(fitted.classes().len(), 2);
}

#[test]
fn test_mixture_discriminant_analysis() {
    let (x, y) = create_simple_2d_data();

    let mda = MixtureDiscriminantAnalysis::new().n_components_per_class(1);
    let fitted = mda.fit(&x, &y).unwrap();
    let predictions = fitted.predict(&x).unwrap();

    assert_eq!(predictions.len(), 4);
    assert_eq!(fitted.classes().len(), 2);
}
