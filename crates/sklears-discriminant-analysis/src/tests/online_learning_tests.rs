//! Online learning discriminant analysis tests

use super::test_utils::*;
use crate::online_discriminant::UpdateStrategy;
use scirs2_core::ndarray::s;

#[test]
fn test_online_discriminant_analysis() {
    let (x, y) = create_simple_2d_data();

    let oda = OnlineDiscriminantAnalysis::new()
        .update_strategy(UpdateStrategy::ExponentialMovingAverage { decay_rate: 0.01 })
        .reg_param(0.01);

    let fitted = oda.fit(&x, &y).unwrap();
    let predictions = fitted.predict(&x).unwrap();

    assert_eq!(predictions.len(), 4);
    assert_eq!(fitted.classes().len(), 2);
}

#[test]
fn test_online_partial_fit() {
    let (x, y) = create_simple_2d_data();

    let oda = OnlineDiscriminantAnalysis::new()
        .update_strategy(UpdateStrategy::ExponentialMovingAverage { decay_rate: 0.1 })
        .batch_size(2);

    // Fit in batches - ensure both batches have multiple classes
    let x_batch1 = x.slice(s![0..3, ..]).to_owned(); // [0, 0, 1] - has both classes
    let y_batch1 = y.slice(s![0..3]).to_owned();
    let x_batch2 = x.slice(s![1..4, ..]).to_owned(); // [0, 1, 1] - has both classes
    let y_batch2 = y.slice(s![1..4]).to_owned();

    let mut fitted = oda.fit(&x_batch1, &y_batch1).unwrap();
    fitted.partial_fit(&x_batch2, &y_batch2).unwrap();

    let predictions = fitted.predict(&x).unwrap();
    assert_eq!(predictions.len(), 4);
}

#[test]
fn test_online_update_strategies() {
    let (x, y) = create_simple_2d_data();

    // Test different update strategies
    let strategies = vec![
        UpdateStrategy::ExponentialMovingAverage { decay_rate: 0.9 },
        UpdateStrategy::SlidingWindow { window_size: 10 },
        UpdateStrategy::AdaptiveWindow {
            drift_threshold: 0.05,
        },
        UpdateStrategy::Cumulative,
    ];

    for strategy in strategies {
        let oda = OnlineDiscriminantAnalysis::new()
            .update_strategy(strategy)
            .reg_param(0.01);

        let fitted = oda.fit(&x, &y).unwrap();
        let predictions = fitted.predict(&x).unwrap();

        assert_eq!(predictions.len(), 4);
        assert_eq!(fitted.classes().len(), 2);
    }
}

#[test]
fn test_online_new_class_addition() {
    let x1 = array![[1.0, 2.0], [2.0, 3.0], [2.5, 3.5]];
    let y1 = array![0, 0, 1];

    let oda = OnlineDiscriminantAnalysis::new()
        .update_strategy(UpdateStrategy::ExponentialMovingAverage { decay_rate: 0.1 })
        .drift_detection(true);

    // Initial fit
    let mut fitted = oda.fit(&x1, &y1).unwrap();
    assert_eq!(fitted.classes().len(), 2);

    // Add new class
    let x2 = array![[3.0, 4.0], [4.0, 5.0]];
    let y2 = array![2, 2];

    fitted.partial_fit(&x2, &y2).unwrap();
    assert_eq!(fitted.classes().len(), 3);

    // Test predictions on all data
    let x_all = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
    let predictions = fitted.predict(&x_all).unwrap();
    assert_eq!(predictions.len(), 4);
}
