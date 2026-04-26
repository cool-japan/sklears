//! Stochastic discriminant analysis tests

use super::super::*;
use approx::assert_abs_diff_eq;
use scirs2_core::ndarray::{array, Axis};
use sklears_core::traits::{Fit, Predict, PredictProba};
use sklears_core::types::Float;

#[test]
fn test_stochastic_discriminant_analysis_basic() {
    let x = array![
        [1.0, 2.0],
        [1.1, 2.1],
        [1.2, 2.2],
        [3.0, 4.0],
        [3.1, 4.1],
        [3.2, 4.2]
    ];
    let y = array![0, 0, 0, 1, 1, 1];

    let sda = StochasticDiscriminantAnalysis::new()
        .max_epochs(50)
        .batch_size(2);
    let fitted = sda.fit(&x, &y).expect("model fitting should succeed");
    let predictions = fitted.predict(&x).expect("prediction should succeed");

    assert_eq!(predictions.len(), 6);
    assert_eq!(fitted.classes().len(), 2);
}

#[test]
fn test_stochastic_discriminant_predict_proba() {
    let x = array![[1.0, 2.0], [1.1, 2.1], [3.0, 4.0], [3.1, 4.1]];
    let y = array![0, 0, 1, 1];

    let sda = StochasticDiscriminantAnalysis::new()
        .max_epochs(30)
        .learning_rate(LearningRateSchedule::Constant { rate: 0.1 });
    let fitted = sda.fit(&x, &y).expect("model fitting should succeed");
    let probas = fitted
        .predict_proba(&x)
        .expect("probability prediction should succeed");

    assert_eq!(probas.dim(), (4, 2));

    for row in probas.axis_iter(Axis(0)) {
        let sum: Float = row.sum();
        assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-5);
    }
}

#[test]
fn test_stochastic_discriminant_with_momentum() {
    let x = array![[1.0, 2.0], [1.1, 2.1], [3.0, 4.0], [3.1, 4.1]];
    let y = array![0, 0, 1, 1];

    let sda = StochasticDiscriminantAnalysis::new()
        .optimizer(Optimizer::Momentum { momentum: 0.9 })
        .max_epochs(20);
    let fitted = sda.fit(&x, &y).expect("model fitting should succeed");
    let predictions = fitted.predict(&x).expect("prediction should succeed");

    assert_eq!(predictions.len(), 4);
    assert_eq!(fitted.classes().len(), 2);
}

#[test]
fn test_stochastic_discriminant_with_adam() {
    let x = array![[1.0, 2.0], [1.1, 2.1], [3.0, 4.0], [3.1, 4.1]];
    let y = array![0, 0, 1, 1];

    let sda = StochasticDiscriminantAnalysis::new()
        .optimizer(Optimizer::Adam {
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
        })
        .max_epochs(20);
    let fitted = sda.fit(&x, &y).expect("model fitting should succeed");
    let predictions = fitted.predict(&x).expect("prediction should succeed");

    assert_eq!(predictions.len(), 4);
    assert_eq!(fitted.classes().len(), 2);
}

#[test]
fn test_stochastic_discriminant_partial_fit() {
    let x1 = array![[1.0, 2.0], [3.0, 4.0]];
    let y1 = array![0, 1]; // Need at least 2 classes for initial training

    let sda = StochasticDiscriminantAnalysis::new();
    let mut fitted = sda.fit(&x1, &y1).expect("model fitting should succeed");

    // Add more data
    let x2 = array![[3.0, 4.0], [3.1, 4.1]];
    let y2 = array![1, 1];
    fitted
        .partial_fit(&x2, &y2)
        .expect("partial fit should succeed");

    assert_eq!(fitted.n_samples_seen(), 4);
}

#[test]
fn test_stochastic_discriminant_hinge_loss() {
    let x = array![[1.0, 2.0], [1.1, 2.1], [3.0, 4.0], [3.1, 4.1]];
    let y = array![0, 0, 1, 1];

    let sda = StochasticDiscriminantAnalysis::new()
        .loss(LossFunction::Hinge)
        .max_epochs(20);
    let fitted = sda.fit(&x, &y).expect("model fitting should succeed");
    let predictions = fitted.predict(&x).expect("prediction should succeed");

    assert_eq!(predictions.len(), 4);
    assert_eq!(fitted.classes().len(), 2);
}

#[test]
fn test_stochastic_discriminant_training_history() {
    let x = array![[1.0, 2.0], [1.1, 2.1], [3.0, 4.0], [3.1, 4.1]];
    let y = array![0, 0, 1, 1];

    let sda = StochasticDiscriminantAnalysis::new().max_epochs(10);
    let fitted = sda.fit(&x, &y).expect("model fitting should succeed");
    let history = fitted.training_history();

    assert!(!history.is_empty());
    assert!(history.len() <= 10);

    if history.len() > 1 {
        let first_loss = history[0];
        let last_loss = history[history.len() - 1];
        // Allow for some fluctuation in stochastic optimization
        assert!(last_loss <= first_loss + 0.5);
    }
}
