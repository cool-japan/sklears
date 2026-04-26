//! Integration tests for sklears-multiclass OvR, OvO, and AdaBoost classifiers

use scirs2_core::ndarray::Array2;
use sklears_core::{
    traits::{Fit, Predict},
    types::{Array1, Float},
};
use sklears_linear::LinearRegression;
use sklears_multiclass::{AdaBoostClassifier, OneVsOneClassifier, OneVsRestClassifier};

/// Build a small synthetic 3-class dataset.
///
/// Returns `(x, y)` where:
///  - `x` is shape `(n_samples, n_features)` with linearly separable values
///  - `y` is `Array1<i32>` with labels 0, 1, 2
fn make_three_class_data(n_per_class: usize, n_features: usize) -> (Array2<Float>, Array1<i32>) {
    let n_samples = n_per_class * 3;
    let x = Array2::from_shape_fn((n_samples, n_features), |(i, j)| {
        let class = (i / n_per_class) as Float;
        // Each class is centred at class * 5.0; offset by feature index
        class * 5.0 + j as Float * 0.1 + (i as Float * 0.01)
    });
    let y = Array1::from_shape_fn(n_samples, |i| (i / n_per_class) as i32);
    (x, y)
}

#[test]
fn test_one_vs_rest_basic() {
    let (x, y) = make_three_class_data(20, 4);

    let base = LinearRegression::new();
    let ovr = OneVsRestClassifier::new(base);

    let fitted = ovr.fit(&x, &y).expect("OvR fit should succeed");

    assert_eq!(fitted.n_classes(), 3, "should detect 3 classes");

    let preds = fitted.predict(&x).expect("OvR predict should succeed");
    assert_eq!(
        preds.len(),
        x.nrows(),
        "prediction length should match number of samples"
    );

    // Verify all predicted labels are in {0, 1, 2}
    for &p in preds.iter() {
        assert!(
            (0..=2).contains(&p),
            "predicted label {p} must be in {{0, 1, 2}}"
        );
    }
}

#[test]
fn test_one_vs_one_basic() {
    let (x, y) = make_three_class_data(20, 4);

    let base = LinearRegression::new();
    let ovo = OneVsOneClassifier::new(base);

    let fitted = ovo.fit(&x, &y).expect("OvO fit should succeed");

    assert_eq!(fitted.n_classes(), 3, "should detect 3 classes");

    // n_pairs = 3*(3-1)/2 = 3
    assert_eq!(
        fitted.estimators().len(),
        3,
        "OvO should train 3 pairwise classifiers"
    );

    let preds = fitted.predict(&x).expect("OvO predict should succeed");
    assert_eq!(
        preds.len(),
        x.nrows(),
        "prediction length should match number of samples"
    );

    // Verify all predicted labels are in {0, 1, 2}
    for &p in preds.iter() {
        assert!(
            (0..=2).contains(&p),
            "predicted label {p} must be in {{0, 1, 2}}"
        );
    }
}

/// Build a small synthetic 2-class dataset for AdaBoost testing.
///
/// Returns `(x, y)` where:
///  - `x` is shape `(n_samples, n_features)` with linearly separable values
///  - `y` is `Array1<i32>` with labels 0 and 1
fn make_two_class_data(n_per_class: usize, n_features: usize) -> (Array2<Float>, Array1<i32>) {
    let n_samples = n_per_class * 2;
    let x = Array2::from_shape_fn((n_samples, n_features), |(i, j)| {
        let class = (i / n_per_class) as Float;
        // Each class is centred at class * 5.0; offset by feature index
        class * 5.0 + j as Float * 0.1 + (i as Float * 0.01)
    });
    let y = Array1::from_shape_fn(n_samples, |i| (i / n_per_class) as i32);
    (x, y)
}

#[test]
fn test_adaboost_basic() {
    let (x, y) = make_two_class_data(25, 4); // 50 samples, 4 features, 2-class

    let base = LinearRegression::new();
    let adaboost = AdaBoostClassifier::new(base)
        .n_estimators(5)
        .random_state(Some(42));

    let fitted = adaboost.fit(&x, &y).expect("AdaBoost fit should succeed");

    let preds = fitted.predict(&x).expect("AdaBoost predict should succeed");
    assert_eq!(
        preds.len(),
        x.nrows(),
        "prediction length should match number of samples"
    );

    // Verify all predicted labels are in {0, 1}
    for &p in preds.iter() {
        assert!(
            (0..=1).contains(&p),
            "predicted label {p} must be in {{0, 1}}"
        );
    }
}
