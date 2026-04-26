//! Integration tests for sklears-multiclass re-enabled modules:
//! advanced (stacking), calibration, core (ECOC), and ensemble (rotation forest, gradient boosting)

use scirs2_core::ndarray::Array2;
use sklears_core::{
    traits::{Fit, Predict, PredictProba},
    types::{Array1, Float},
};
use sklears_linear::LinearRegression;
use sklears_multiclass::{
    CalibratedClassifier, CalibrationMethod, ECOCClassifier, ECOCStrategy,
    GradientBoostingClassifier, MulticlassStackingClassifier, RotationForestClassifier,
};

/// Build a small synthetic 2-class dataset.
fn make_two_class_data(n_per_class: usize, n_features: usize) -> (Array2<Float>, Array1<i32>) {
    let n_samples = n_per_class * 2;
    let x = Array2::from_shape_fn((n_samples, n_features), |(i, j)| {
        let class = (i / n_per_class) as Float;
        class * 5.0 + j as Float * 0.1 + (i as Float * 0.01)
    });
    let y = Array1::from_shape_fn(n_samples, |i| (i / n_per_class) as i32);
    (x, y)
}

/// Build a small synthetic 3-class dataset.
fn make_three_class_data(n_per_class: usize, n_features: usize) -> (Array2<Float>, Array1<i32>) {
    let n_samples = n_per_class * 3;
    let x = Array2::from_shape_fn((n_samples, n_features), |(i, j)| {
        let class = (i / n_per_class) as Float;
        class * 5.0 + j as Float * 0.1 + (i as Float * 0.01)
    });
    let y = Array1::from_shape_fn(n_samples, |i| (i / n_per_class) as i32);
    (x, y)
}

// ─── Module: advanced (stacking) ───────────────────────────────────────────

/// Minimal mock classifier for stacking integration test.
/// Satisfies: Fit<Array2<f64>, Array1<i32>> + Clone.
/// Fitted satisfies: Predict + PredictProba + Clone + Send + Sync.
#[derive(Debug, Clone)]
struct SimpleMockClassifier;

#[derive(Debug, Clone)]
struct SimpleMockTrained {
    classes: Vec<i32>,
    n_classes: usize,
}

impl sklears_core::traits::Estimator for SimpleMockClassifier {
    type Config = ();
    type Error = sklears_core::error::SklearsError;
    type Float = f64;
    fn config(&self) -> &() {
        &()
    }
}

impl Fit<Array2<f64>, Array1<i32>> for SimpleMockClassifier {
    type Fitted = SimpleMockTrained;
    fn fit(self, _x: &Array2<f64>, y: &Array1<i32>) -> sklears_core::error::Result<Self::Fitted> {
        let mut classes: Vec<i32> = y.iter().cloned().collect();
        classes.sort_unstable();
        classes.dedup();
        let n_classes = classes.len();
        Ok(SimpleMockTrained { classes, n_classes })
    }
}

impl Predict<Array2<f64>, Array1<i32>> for SimpleMockTrained {
    fn predict(&self, x: &Array2<f64>) -> sklears_core::error::Result<Array1<i32>> {
        let n = x.nrows();
        Ok(Array1::from_vec(
            (0..n)
                .map(|i| self.classes[i % self.n_classes.max(1)])
                .collect(),
        ))
    }
}

impl PredictProba<Array2<f64>, Array2<f64>> for SimpleMockTrained {
    fn predict_proba(&self, x: &Array2<f64>) -> sklears_core::error::Result<Array2<f64>> {
        let n = x.nrows();
        let nc = self.n_classes.max(1);
        let mut out = Array2::zeros((n, nc));
        for i in 0..n {
            for j in 0..nc {
                out[[i, j]] = 1.0 / nc as f64;
            }
        }
        Ok(out)
    }
}

#[test]
fn test_stacking_classifier_basic() {
    let (x, y) = make_three_class_data(15, 4);

    let base1 = SimpleMockClassifier;
    let base2 = SimpleMockClassifier;
    let meta = SimpleMockClassifier;

    let stacking =
        MulticlassStackingClassifier::new(vec![base1, base2], meta).random_state(Some(42));

    let fitted = stacking.fit(&x, &y).expect("Stacking fit should succeed");
    let preds = fitted.predict(&x).expect("Stacking predict should succeed");

    assert_eq!(
        preds.len(),
        x.nrows(),
        "prediction count should match samples"
    );
    for &p in preds.iter() {
        assert!(
            (0..=2).contains(&p),
            "predicted label {p} must be in {{0, 1, 2}}"
        );
    }
}

// ─── Module: calibration ───────────────────────────────────────────────────

#[test]
fn test_calibrated_classifier_platt_scaling() {
    let (x, y) = make_two_class_data(20, 4);

    // SimpleMockClassifier implements Fit<Array2<f64>, Array1<i32>>
    // SimpleMockTrained implements PredictProba<Array2<f64>, Array2<f64>> + Clone
    let base = SimpleMockClassifier;

    let calibrated = CalibratedClassifier::new(base)
        .method(CalibrationMethod::PlattScaling)
        .random_state(Some(42));

    let fitted = calibrated
        .fit(&x, &y)
        .expect("Calibrated fit should succeed");

    let proba = fitted
        .predict_proba(&x)
        .expect("Calibrated predict_proba should succeed");

    assert_eq!(proba.nrows(), x.nrows(), "proba rows should match samples");
    assert_eq!(proba.ncols(), 2, "proba cols should equal n_classes");

    // Probabilities should approximately sum to 1.0 per row
    for i in 0..x.nrows() {
        let row_sum: f64 = proba.row(i).sum();
        assert!(
            (row_sum - 1.0).abs() < 0.1,
            "row {i} probabilities should sum near 1.0, got {row_sum}"
        );
    }
}

// ─── Module: core (ECOC) ───────────────────────────────────────────────────

#[test]
fn test_ecoc_classifier_basic() {
    let (x, y) = make_three_class_data(15, 4);

    // ECOC uses binary classifiers (Fit<Array2<f64>, Array1<f64>>)
    let base = LinearRegression::new();
    let ecoc = ECOCClassifier::new(base)
        .strategy(ECOCStrategy::StdRng)
        .random_state(42);

    let fitted = ecoc.fit(&x, &y).expect("ECOC fit should succeed");
    let preds = fitted.predict(&x).expect("ECOC predict should succeed");

    assert_eq!(
        preds.len(),
        x.nrows(),
        "prediction count should match samples"
    );
    for &p in preds.iter() {
        assert!(
            (0..=2).contains(&p),
            "predicted label {p} must be in {{0, 1, 2}}"
        );
    }
}

// ─── Module: ensemble (RotationForest) ─────────────────────────────────────

#[test]
fn test_rotation_forest_classifier_basic() {
    let (x, y) = make_three_class_data(20, 4);

    // RotationForest requires Fit<Array2<f64>, Array1<i32>> + Clone on base/fitted
    // SimpleMockClassifier satisfies all constraints
    let base = SimpleMockClassifier;
    let rf = RotationForestClassifier::new(base)
        .n_estimators(3)
        .random_state(Some(42));

    let fitted = rf.fit(&x, &y).expect("RotationForest fit should succeed");
    let preds = fitted
        .predict(&x)
        .expect("RotationForest predict should succeed");

    assert_eq!(
        preds.len(),
        x.nrows(),
        "prediction count should match samples"
    );
    for &p in preds.iter() {
        assert!(
            (0..=2).contains(&p),
            "predicted label {p} must be in {{0, 1, 2}}"
        );
    }
}

// ─── Module: ensemble (GradientBoosting) ───────────────────────────────────

#[test]
fn test_gradient_boosting_classifier_basic() {
    let (x, y) = make_two_class_data(25, 4);

    // GradientBoosting requires Fit<Array2<f64>, Array1<f64>> for residuals
    let base = LinearRegression::new();
    let gb = GradientBoostingClassifier::new(base)
        .n_estimators(5)
        .random_state(Some(42));

    let fitted = gb.fit(&x, &y).expect("GradientBoosting fit should succeed");
    let preds = fitted
        .predict(&x)
        .expect("GradientBoosting predict should succeed");

    assert_eq!(
        preds.len(),
        x.nrows(),
        "prediction count should match samples"
    );
    for &p in preds.iter() {
        assert!(
            (0..=1).contains(&p),
            "predicted label {p} must be in {{0, 1}}"
        );
    }
}
