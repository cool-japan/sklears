//! Regularization tests for discriminant analysis

use super::test_utils::*;
use crate::adaptive_discriminant::{
    AdaptationStrategy, AdaptiveDiscriminantLearning, BaseDiscriminant,
};

#[test]
fn test_lda_with_shrinkage() {
    let (x, y) = create_simple_2d_data();

    let lda = LinearDiscriminantAnalysis::new().shrinkage(Some(0.1));
    let fitted = lda.fit(&x, &y).unwrap();
    let predictions = fitted.predict(&x).unwrap();

    assert_eq!(predictions.len(), 4);
    assert_eq!(fitted.classes().len(), 2);
}

#[test]
fn test_qda_with_regularization() {
    let (x, y) = create_simple_2d_data();

    let qda = QuadraticDiscriminantAnalysis::new().reg_param(0.1);
    let fitted = qda.fit(&x, &y).unwrap();
    let predictions = fitted.predict(&x).unwrap();

    assert_eq!(predictions.len(), 4);
    assert_eq!(fitted.classes().len(), 2);
}

#[test]
fn test_sparse_lda() {
    let (x, y) = create_simple_2d_data();

    let lda = LinearDiscriminantAnalysis::new().l1_reg(0.1).max_iter(50);
    let fitted = lda.fit(&x, &y).unwrap();
    let predictions = fitted.predict(&x).unwrap();

    assert_eq!(predictions.len(), 4);
    assert_eq!(fitted.classes().len(), 2);
}

#[test]
fn test_elastic_net_lda() {
    let (x, y) = create_simple_2d_data();

    let lda = LinearDiscriminantAnalysis::new()
        .l1_reg(0.05)
        .l2_reg(0.05)
        .elastic_net_ratio(0.5)
        .max_iter(50);

    let fitted = lda.fit(&x, &y).unwrap();
    let predictions = fitted.predict(&x).unwrap();

    assert_eq!(predictions.len(), 4);
    assert_eq!(fitted.classes().len(), 2);
}

#[test]
fn test_adaptive_regularization_ledoit_wolf() {
    let (x, y) = create_simple_2d_data();

    let ada = AdaptiveDiscriminantLearning::new()
        .adaptation_strategy(AdaptationStrategy::PerformanceBased {
            performance_threshold: 0.8,
            adaptation_rate: 0.01,
        })
        .initial_learning_rate(0.01);

    let fitted = ada.fit(&x, &y).unwrap();
    let predictions = fitted.predict(&x).unwrap();

    assert_eq!(predictions.len(), 4);
    assert_eq!(fitted.classes().len(), 2);
}

#[test]
fn test_adaptive_regularization_oas() {
    let (x, y) = create_simple_2d_data();

    let ada = AdaptiveDiscriminantLearning::new()
        .base_discriminant(BaseDiscriminant::RegularizedLDA { reg_param: 0.1 })
        .adaptation_strategy(AdaptationStrategy::ExponentialMovingAverage { decay_rate: 0.9 });

    let fitted = ada.fit(&x, &y).unwrap();
    let predictions = fitted.predict(&x).unwrap();

    assert_eq!(predictions.len(), 4);
    assert_eq!(fitted.classes().len(), 2);
}

#[test]
fn test_adaptive_regularization_mcd() {
    let (x, y) = create_simple_2d_data();

    let ada = AdaptiveDiscriminantLearning::new()
        .adaptation_strategy(AdaptationStrategy::ConceptDriftDetection {
            drift_threshold: 0.1,
            detection_window: 50,
        })
        .adaptation_frequency(10);

    let fitted = ada.fit(&x, &y).unwrap();
    let predictions = fitted.predict(&x).unwrap();

    assert_eq!(predictions.len(), 4);
    assert_eq!(fitted.classes().len(), 2);
}
