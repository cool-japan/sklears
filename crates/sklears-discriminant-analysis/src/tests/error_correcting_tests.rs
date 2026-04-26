//! Error correcting output codes tests

use super::super::*;
use approx::assert_abs_diff_eq;
use scirs2_core::ndarray::{array, Axis};
use sklears_core::traits::{Fit, Predict, PredictProba};
use sklears_core::types::Float;

#[test]
fn test_error_correcting_output_codes_basic() {
    let x = array![
        [1.0, 2.0],
        [2.0, 3.0],
        [3.0, 4.0],
        [4.0, 5.0],
        [5.0, 6.0],
        [6.0, 7.0]
    ];
    let y = array![0, 0, 1, 1, 2, 2];

    let ecoc = ErrorCorrectingOutputCodes::new()
        .code_method("random")
        .n_codes(5);

    let fitted = ecoc.fit(&x, &y).expect("model fitting should succeed");
    let predictions = fitted.predict(&x).expect("prediction should succeed");

    assert_eq!(predictions.len(), 6);
    assert_eq!(fitted.classes().len(), 3);
    assert_eq!(fitted.n_classifiers(), 5);
}

#[test]
fn test_error_correcting_output_codes_predict_proba() {
    let x = array![
        [1.0, 2.0],
        [2.0, 3.0],
        [3.0, 4.0],
        [4.0, 5.0],
        [5.0, 6.0],
        [6.0, 7.0]
    ];
    let y = array![0, 0, 1, 1, 2, 2];

    let ecoc = ErrorCorrectingOutputCodes::new()
        .code_method("dense_random")
        .n_codes(4);

    let fitted = ecoc.fit(&x, &y).expect("model fitting should succeed");
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
fn test_different_code_methods() {
    let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
    let y = array![0, 0, 1, 1];

    let methods = ["random", "dense_random", "sparse_random"];

    for method in &methods {
        let ecoc = ErrorCorrectingOutputCodes::new()
            .code_method(method)
            .n_codes(3);

        let fitted = ecoc.fit(&x, &y).expect("model fitting should succeed");
        let predictions = fitted.predict(&x).expect("prediction should succeed");

        assert_eq!(predictions.len(), 4);
        assert_eq!(fitted.classes().len(), 2);
    }
}

#[test]
fn test_exhaustive_code_method() {
    let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
    let y = array![0, 0, 1, 1];

    let ecoc = ErrorCorrectingOutputCodes::new().code_method("exhaustive");

    let fitted = ecoc.fit(&x, &y).expect("model fitting should succeed");
    let predictions = fitted.predict(&x).expect("prediction should succeed");

    assert_eq!(predictions.len(), 4);
    assert_eq!(fitted.classes().len(), 2);
}

#[test]
fn test_code_matrix_properties() {
    let ecoc = ErrorCorrectingOutputCodes::new()
        .code_method("random")
        .n_codes(4)
        .random_state(42);

    let code_matrix = ecoc
        .generate_code_matrix(3)
        .expect("operation should succeed");

    assert_eq!(code_matrix.dim(), (3, 4));

    for j in 0..4 {
        let col = code_matrix.column(j);
        let has_pos = col.iter().any(|&x| x == 1);
        let has_neg = col.iter().any(|&x| x == -1);
        assert!(
            has_pos && has_neg,
            "Code {} should have both +1 and -1 values",
            j
        );
    }
}

#[test]
fn test_different_correction_methods() {
    let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
    let y = array![0, 0, 1, 1];

    let methods = ["hamming", "euclidean", "manhattan"];

    for method in &methods {
        let ecoc = ErrorCorrectingOutputCodes::new()
            .correction_method(method)
            .n_codes(3);

        let fitted = ecoc.fit(&x, &y).expect("model fitting should succeed");
        let predictions = fitted.predict(&x).expect("prediction should succeed");

        assert_eq!(predictions.len(), 4);
        assert_eq!(fitted.classes().len(), 2);
    }
}
