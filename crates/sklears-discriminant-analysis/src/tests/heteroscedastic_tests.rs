//! Heteroscedastic discriminant analysis tests

use super::test_utils::*;

#[test]
fn test_heteroscedastic_discriminant_analysis_full() {
    let (x, y) = create_simple_2d_data();

    let hda = HeteroscedasticDiscriminantAnalysis::new().covariance_type("full");

    let fitted = hda.fit(&x, &y).unwrap();
    let predictions = fitted.predict(&x).unwrap();

    assert_eq!(predictions.len(), 4);
    assert_eq!(fitted.classes().len(), 2);
}

#[test]
fn test_heteroscedastic_discriminant_analysis_tied() {
    let (x, y) = create_simple_2d_data();

    let hda = HeteroscedasticDiscriminantAnalysis::new().covariance_type("tied");

    let fitted = hda.fit(&x, &y).unwrap();
    let predictions = fitted.predict(&x).unwrap();

    assert_eq!(predictions.len(), 4);
    assert_eq!(fitted.classes().len(), 2);
}

#[test]
fn test_heteroscedastic_discriminant_analysis_diag() {
    let (x, y) = create_simple_2d_data();

    let hda = HeteroscedasticDiscriminantAnalysis::new().covariance_type("diag");

    let fitted = hda.fit(&x, &y).unwrap();
    let predictions = fitted.predict(&x).unwrap();

    assert_eq!(predictions.len(), 4);
    assert_eq!(fitted.classes().len(), 2);
}

#[test]
fn test_heteroscedastic_discriminant_analysis_spherical() {
    let (x, y) = create_simple_2d_data();

    let hda = HeteroscedasticDiscriminantAnalysis::new().covariance_type("spherical");

    let fitted = hda.fit(&x, &y).unwrap();
    let predictions = fitted.predict(&x).unwrap();

    assert_eq!(predictions.len(), 4);
    assert_eq!(fitted.classes().len(), 2);
}

#[test]
fn test_heteroscedastic_predict_proba() {
    let (x, y) = create_simple_2d_data();

    let hda = HeteroscedasticDiscriminantAnalysis::new()
        .covariance_type("full")
        .store_covariance(true);

    let fitted = hda.fit(&x, &y).unwrap();
    let probas = fitted.predict_proba(&x).unwrap();

    assert_eq!(probas.dim(), (4, 2));
    assert_probabilities_sum_to_one(&probas);

    // Test covariance matrices are stored
    let covariances = fitted.covariances().unwrap();
    assert_eq!(covariances.len(), 2); // One for each class
}

#[test]
fn test_heteroscedastic_with_adaptive_regularization() {
    let (x, y) = create_simple_2d_data();

    let hda = HeteroscedasticDiscriminantAnalysis::new()
        .covariance_type("full")
        .adaptive_regularization(true)
        .adaptive_method("ledoit_wolf");

    let fitted = hda.fit(&x, &y).unwrap();
    let predictions = fitted.predict(&x).unwrap();
    let probas = fitted.predict_proba(&x).unwrap();

    assert_eq!(predictions.len(), 4);
    assert_eq!(fitted.classes().len(), 2);
    assert_probabilities_sum_to_one(&probas);
}

#[test]
fn test_heteroscedastic_with_shrinkage() {
    let (x, y) = create_simple_2d_data();

    let hda = HeteroscedasticDiscriminantAnalysis::new()
        .covariance_type("tied")
        .shrinkage(Some(0.1))
        .store_covariance(true);

    let fitted = hda.fit(&x, &y).unwrap();
    let predictions = fitted.predict(&x).unwrap();

    assert_eq!(predictions.len(), 4);
    assert_eq!(fitted.classes().len(), 2);

    // Test that shared covariance is stored (since we used tied covariance)
    let shared_cov = fitted.shared_covariance().unwrap();
    assert_eq!(shared_cov.nrows(), 2); // 2D data
}
