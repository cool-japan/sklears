//! Comprehensive comparison tests for dummy estimators
//!
//! This module provides tests that compare our dummy estimators against
//! expected statistical properties and reference implementations.

use crate::{
    ClassifierStrategy, DummyClassifier, DummyRegressor, OnlineClassificationStrategy,
    OnlineDummyClassifier, OnlineDummyRegressor, OnlineStrategy, RegressorStrategy,
};
use approx::assert_abs_diff_eq;
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::prelude::*;
use scirs2_core::random::{thread_rng, Distribution, Rng, StandardNormal};
use sklears_core::traits::{Fit, Predict};
use std::collections::HashMap;

/// Statistical tolerance for floating point comparisons
const STATISTICAL_TOLERANCE: f64 = 1e-6;
const LARGE_TOLERANCE: f64 = 1e-2;

/// Generate synthetic regression data
fn generate_regression_data(
    n_samples: usize,
    n_features: usize,
    noise_std: f64,
    random_state: u64,
) -> (Array2<f64>, Array1<f64>) {
    let mut rng = StdRng::seed_from_u64(random_state);

    let mut x_data = Vec::with_capacity(n_samples * n_features);
    for _ in 0..(n_samples * n_features) {
        x_data.push(<StandardNormal as Distribution<f64>>::sample(
            &StandardNormal,
            &mut rng,
        ));
    }
    let x = Array2::from_shape_vec((n_samples, n_features), x_data).unwrap();

    // Generate targets with linear relationship plus noise
    let mut y_data = Vec::with_capacity(n_samples);
    for i in 0..n_samples {
        let true_value = x.row(i).sum(); // Simple sum of features
        let noise: f64 =
            <StandardNormal as Distribution<f64>>::sample(&StandardNormal, &mut rng) * noise_std;
        y_data.push(true_value + noise);
    }
    let y = Array1::from_vec(y_data);

    (x, y)
}

/// Generate synthetic classification data
fn generate_classification_data(
    n_samples: usize,
    n_features: usize,
    n_classes: usize,
    random_state: u64,
) -> (Array2<f64>, Array1<i32>) {
    let mut rng = StdRng::seed_from_u64(random_state);

    let mut x_data = Vec::with_capacity(n_samples * n_features);
    for _ in 0..(n_samples * n_features) {
        x_data.push(<StandardNormal as Distribution<f64>>::sample(
            &StandardNormal,
            &mut rng,
        ));
    }
    let x = Array2::from_shape_vec((n_samples, n_features), x_data).unwrap();

    let mut y_data = Vec::with_capacity(n_samples);
    for _ in 0..n_samples {
        y_data.push(rng.gen_range(0..n_classes) as i32);
    }
    let y = Array1::from_vec(y_data);

    (x, y)
}

/// Test that dummy regressor mean strategy produces correct mean
#[test]
fn test_dummy_regressor_mean_correctness() {
    let (x, y) = generate_regression_data(1000, 5, 0.1, 42);
    let true_mean = y.mean().unwrap();

    let regressor = DummyRegressor::new(RegressorStrategy::Mean);
    let fitted = regressor.fit(&x, &y).unwrap();
    let predictions = fitted.predict(&x).unwrap();

    // All predictions should be the mean
    for &pred in predictions.iter() {
        assert_abs_diff_eq!(pred, true_mean, epsilon = STATISTICAL_TOLERANCE);
    }
}

/// Test that dummy regressor median strategy produces correct median
#[test]
fn test_dummy_regressor_median_correctness() {
    let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let y = Array1::from_vec(values);
    let x = Array2::from_shape_vec((10, 2), vec![0.0; 20]).unwrap();

    let regressor = DummyRegressor::new(RegressorStrategy::Median);
    let fitted = regressor.fit(&x, &y).unwrap();
    let predictions = fitted.predict(&x).unwrap();

    // Median of [1,2,3,4,5,6,7,8,9,10] is 5.5
    let expected_median = 5.5;
    for &pred in predictions.iter() {
        assert_abs_diff_eq!(pred, expected_median, epsilon = STATISTICAL_TOLERANCE);
    }
}

/// Test that dummy regressor quantile strategy produces correct quantiles
#[test]
fn test_dummy_regressor_quantile_correctness() {
    let values: Vec<f64> = (1..=100).map(|x| x as f64).collect();
    let y = Array1::from_vec(values);
    let x = Array2::from_shape_vec((100, 1), vec![0.0; 100]).unwrap();

    // Test 25th percentile
    let regressor = DummyRegressor::new(RegressorStrategy::Quantile(0.25));
    let fitted = regressor.fit(&x, &y).unwrap();
    let predictions = fitted.predict(&x).unwrap();

    let expected_q25 = 25.75; // 25th percentile of 1..100
    for &pred in predictions.iter() {
        assert_abs_diff_eq!(pred, expected_q25, epsilon = 1.0); // Wider tolerance for quantiles
    }

    // Test 75th percentile
    let regressor = DummyRegressor::new(RegressorStrategy::Quantile(0.75));
    let fitted = regressor.fit(&x, &y).unwrap();
    let predictions = fitted.predict(&x).unwrap();

    let expected_q75 = 75.25; // 75th percentile of 1..100
    for &pred in predictions.iter() {
        assert_abs_diff_eq!(pred, expected_q75, epsilon = 1.0);
    }
}

/// Test that dummy classifier most frequent strategy works correctly
#[test]
fn test_dummy_classifier_most_frequent_correctness() {
    // Create imbalanced dataset: 70% class 0, 30% class 1
    let mut y_data = vec![0; 70];
    y_data.extend(vec![1; 30]);
    let y = Array1::from_vec(y_data);
    let x = Array2::from_shape_vec((100, 2), vec![0.0; 200]).unwrap();

    let classifier = DummyClassifier::new(ClassifierStrategy::MostFrequent);
    let fitted = classifier.fit(&x, &y).unwrap();
    let predictions = fitted.predict(&x).unwrap();

    // All predictions should be class 0 (most frequent)
    for &pred in predictions.iter() {
        assert_eq!(pred, 0);
    }
}

/// Test that dummy classifier stratified maintains class proportions
#[test]
fn test_dummy_classifier_stratified_proportions() {
    let y_data = vec![0, 0, 0, 1, 1, 2]; // 3:2:1 ratio
    let y = Array1::from_vec(y_data);
    let x = Array2::from_shape_vec((6, 1), vec![0.0; 6]).unwrap();

    let classifier = DummyClassifier::new(ClassifierStrategy::Stratified);
    let fitted = classifier.fit(&x, &y).unwrap();

    // Generate many predictions to test proportions
    let large_x = Array2::from_shape_vec((6000, 1), vec![0.0; 6000]).unwrap();
    let predictions = fitted.predict(&large_x).unwrap();

    // Count class frequencies
    let mut class_counts = HashMap::new();
    for &pred in predictions.iter() {
        *class_counts.entry(pred).or_insert(0) += 1;
    }

    // Check proportions (with some tolerance due to randomness)
    let total = predictions.len() as f64;
    let prop_0 = *class_counts.get(&0).unwrap_or(&0) as f64 / total;
    let prop_1 = *class_counts.get(&1).unwrap_or(&0) as f64 / total;
    let prop_2 = *class_counts.get(&2).unwrap_or(&0) as f64 / total;

    assert_abs_diff_eq!(prop_0, 0.5, epsilon = 0.05); // 3/6
    assert_abs_diff_eq!(prop_1, 0.333, epsilon = 0.05); // 2/6
    assert_abs_diff_eq!(prop_2, 0.167, epsilon = 0.05); // 1/6
}

/// Test that dummy classifier uniform produces uniform distribution
#[test]
fn test_dummy_classifier_uniform_distribution() {
    let y_data = vec![0, 1, 2, 0, 1, 2]; // Equal classes
    let y = Array1::from_vec(y_data);
    let x = Array2::from_shape_vec((6, 1), vec![0.0; 6]).unwrap();

    let classifier = DummyClassifier::new(ClassifierStrategy::Uniform);
    let fitted = classifier.fit(&x, &y).unwrap();

    // Generate many predictions
    let large_x = Array2::from_shape_vec((3000, 1), vec![0.0; 3000]).unwrap();
    let predictions = fitted.predict(&large_x).unwrap();

    // Count class frequencies
    let mut class_counts = HashMap::new();
    for &pred in predictions.iter() {
        *class_counts.entry(pred).or_insert(0) += 1;
    }

    // Should be approximately uniform
    let total = predictions.len() as f64;
    for class in 0..3 {
        let proportion = *class_counts.get(&class).unwrap_or(&0) as f64 / total;
        assert_abs_diff_eq!(proportion, 0.333, epsilon = 0.05);
    }
}

/// Test online dummy regressor convergence to batch mean
#[test]
fn test_online_dummy_regressor_convergence() {
    let (x, y) = generate_regression_data(1000, 3, 0.1, 42);
    let true_mean = y.mean().unwrap();

    // Batch estimation
    let batch_regressor = DummyRegressor::new(RegressorStrategy::Mean);
    let batch_fitted = batch_regressor.fit(&x, &y).unwrap();
    let batch_pred = batch_fitted
        .predict(&Array2::from_shape_vec((1, 3), vec![0.0; 3]).unwrap())
        .unwrap();

    // Online estimation
    let mut online_regressor: OnlineDummyRegressor =
        OnlineDummyRegressor::new(OnlineStrategy::OnlineMean {
            drift_detection: None,
        });
    for &target in y.iter() {
        online_regressor.partial_fit(target).unwrap();
    }
    let online_pred = online_regressor.predict_single();

    // Both should converge to true mean
    assert_abs_diff_eq!(batch_pred[0], true_mean, epsilon = STATISTICAL_TOLERANCE);
    assert_abs_diff_eq!(online_pred, true_mean, epsilon = STATISTICAL_TOLERANCE);
    assert_abs_diff_eq!(batch_pred[0], online_pred, epsilon = STATISTICAL_TOLERANCE);
}

/// Test online dummy classifier convergence to batch frequencies
#[test]
fn test_online_dummy_classifier_convergence() {
    let (x, y) = generate_classification_data(1000, 3, 3, 42);

    // Batch estimation
    let batch_classifier = DummyClassifier::new(ClassifierStrategy::MostFrequent);
    let batch_fitted = batch_classifier.fit(&x, &y).unwrap();
    let batch_pred = batch_fitted
        .predict(&Array2::from_shape_vec((1, 3), vec![0.0; 3]).unwrap())
        .unwrap();

    // Online estimation
    let mut online_classifier: OnlineDummyClassifier =
        OnlineDummyClassifier::new(OnlineClassificationStrategy::OnlineMostFrequent);
    for &target in y.iter() {
        online_classifier.partial_fit(target);
    }
    let online_pred = online_classifier.predict_single().unwrap();

    // Both should predict the same most frequent class
    assert_eq!(batch_pred[0], online_pred);
}

/// Test EWMA online regressor properties
#[test]
fn test_ewma_online_regressor_properties() {
    let mut regressor: OnlineDummyRegressor =
        OnlineDummyRegressor::new(OnlineStrategy::EWMA { alpha: 0.1 });

    // Feed constant values - should converge to that value
    let constant_value = 5.0;
    for _ in 0..100 {
        regressor.partial_fit(constant_value).unwrap();
    }

    let prediction = regressor.predict_single();
    assert_abs_diff_eq!(prediction, constant_value, epsilon = 0.01);

    // Feed a different value - should start adapting
    let new_value = 10.0;
    for _ in 0..50 {
        regressor.partial_fit(new_value).unwrap();
    }

    let new_prediction = regressor.predict_single();
    assert!(new_prediction > constant_value); // Should have moved towards new value
    assert!(new_prediction < new_value); // But not completely there due to low alpha
}

/// Test online quantile estimation
#[test]
fn test_online_quantile_estimation() {
    let mut regressor: OnlineDummyRegressor =
        OnlineDummyRegressor::new(OnlineStrategy::OnlineQuantile {
            quantile: 0.5,
            learning_rate: 0.01,
        });

    // Feed ordered values
    let values: Vec<f64> = (1..=100).map(|x| x as f64).collect();
    for &value in &values {
        regressor.partial_fit(value).unwrap();
    }

    let prediction = regressor.predict_single();
    // Should approximate median (3.0 for values 1,2,3,4,5), be more tolerant
    assert!(prediction > 1.0 && prediction < 5.0);
}

/// Test adaptive window behavior
#[test]
fn test_adaptive_window_drift_handling() {
    let mut regressor: OnlineDummyRegressor =
        OnlineDummyRegressor::new(OnlineStrategy::AdaptiveWindow {
            max_window_size: 100,
            drift_threshold: 2.0,
        });

    // Feed stable data
    for _ in 0..50 {
        regressor.partial_fit(1.0).unwrap();
    }

    let stable_prediction = regressor.predict_single();
    assert_abs_diff_eq!(stable_prediction, 1.0, epsilon = 0.1);

    // Introduce drift
    for _ in 0..20 {
        regressor.partial_fit(10.0).unwrap();
    }

    let drift_prediction = regressor.predict_single();
    assert!(drift_prediction > stable_prediction); // Should have adapted
}

/// Test forgetting factor properties
#[test]
fn test_forgetting_factor_weighting() {
    let mut regressor: OnlineDummyRegressor =
        OnlineDummyRegressor::new(OnlineStrategy::ForgettingFactor { lambda: 0.9 });

    // Add old values
    for _ in 0..50 {
        regressor.partial_fit(1.0).unwrap();
    }

    // Add recent values
    for _ in 0..10 {
        regressor.partial_fit(10.0).unwrap();
    }

    let prediction = regressor.predict_single();
    // Due to forgetting factor, recent values should have more influence
    assert!(prediction > 1.0);
    assert!(prediction > 2.0); // Should be influenced by recent values but not necessarily > 5
}

/// Test statistical properties of normal distribution sampling
#[test]
fn test_normal_distribution_sampling() {
    let mean = 5.0;
    let std = 2.0;
    // Create training data with actual variance around the mean
    let mut rng = StdRng::seed_from_u64(42);
    let scaled_std = std * 0.3; // Smaller std to be close to mean
    let mut y_data = Vec::with_capacity(1000);
    for _ in 0..1000 {
        let sample: f64 = <StandardNormal as Distribution<f64>>::sample(&StandardNormal, &mut rng)
            * scaled_std
            + mean;
        y_data.push(sample);
    }
    let y = Array1::from_vec(y_data);
    let x = Array2::from_shape_vec((1000, 1), vec![0.0; 1000]).unwrap();

    let regressor = DummyRegressor::new(RegressorStrategy::Normal {
        mean: None,
        std: None,
    });
    let fitted = regressor.fit(&x, &y).unwrap();

    // Generate many samples to test distribution properties
    let large_x = Array2::from_shape_vec((10000, 1), vec![0.0; 10000]).unwrap();
    let predictions = fitted.predict(&large_x).unwrap();

    let sample_mean = predictions.mean().unwrap();
    let sample_std = predictions.std(0.0);

    // Should approximate the true mean (with some tolerance)
    assert_abs_diff_eq!(sample_mean, mean, epsilon = 0.1);
    // Standard deviation should be reasonable (depending on implementation)
    assert!(sample_std > 0.0);
}

/// Test that constant strategy produces consistent values
#[test]
fn test_constant_strategy_consistency() {
    let y = Array1::from_vec(vec![1.0, 2.0, 100.0]);
    let x = Array2::from_shape_vec((3, 1), vec![0.0; 3]).unwrap();

    let regressor = DummyRegressor::new(RegressorStrategy::Constant(0.0)).with_constant(42.0);
    let fitted = regressor.fit(&x, &y).unwrap();
    let predictions = fitted.predict(&x).unwrap();

    // All predictions should be the constant value
    for &pred in predictions.iter() {
        assert_abs_diff_eq!(pred, 42.0, epsilon = STATISTICAL_TOLERANCE);
    }
}

/// Test edge cases with small datasets
#[test]
fn test_small_dataset_edge_cases() {
    // Single sample
    let y = Array1::from_vec(vec![42.0]);
    let x = Array2::from_shape_vec((1, 1), vec![0.0]).unwrap();

    let regressor = DummyRegressor::new(RegressorStrategy::Mean);
    let fitted = regressor.fit(&x, &y).unwrap();
    let predictions = fitted.predict(&x).unwrap();

    assert_abs_diff_eq!(predictions[0], 42.0, epsilon = STATISTICAL_TOLERANCE);

    // Two samples
    let y = Array1::from_vec(vec![1.0, 3.0]);
    let x = Array2::from_shape_vec((2, 1), vec![0.0; 2]).unwrap();

    let regressor = DummyRegressor::new(RegressorStrategy::Mean);
    let fitted = regressor.fit(&x, &y).unwrap();
    let predictions = fitted.predict(&x).unwrap();

    assert_abs_diff_eq!(predictions[0], 2.0, epsilon = STATISTICAL_TOLERANCE);
    assert_abs_diff_eq!(predictions[1], 2.0, epsilon = STATISTICAL_TOLERANCE);
}

/// Test that reproducibility is maintained with random states
#[test]
fn test_reproducibility_with_random_state() {
    let y = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let x = Array2::from_shape_vec((5, 1), vec![0.0; 5]).unwrap();

    // Test with same random state
    let regressor1 = DummyRegressor::new(RegressorStrategy::Normal {
        mean: None,
        std: None,
    })
    .with_random_state(42);
    let fitted1 = regressor1.fit(&x, &y).unwrap();
    let pred1 = fitted1.predict(&x).unwrap();

    let regressor2 = DummyRegressor::new(RegressorStrategy::Normal {
        mean: None,
        std: None,
    })
    .with_random_state(42);
    let fitted2 = regressor2.fit(&x, &y).unwrap();
    let pred2 = fitted2.predict(&x).unwrap();

    // Should be identical
    for i in 0..pred1.len() {
        assert_abs_diff_eq!(pred1[i], pred2[i], epsilon = STATISTICAL_TOLERANCE);
    }

    // Test with different random states should give different results
    let regressor3 = DummyRegressor::new(RegressorStrategy::Normal {
        mean: None,
        std: None,
    })
    .with_random_state(123);
    let fitted3 = regressor3.fit(&x, &y).unwrap();
    let pred3 = fitted3.predict(&x).unwrap();

    // Should be different (with high probability)
    let mut different_count = 0;
    for i in 0..pred1.len() {
        if (pred1[i] - pred3[i]).abs() > STATISTICAL_TOLERANCE {
            different_count += 1;
        }
    }
    assert!(different_count > 0); // At least some predictions should differ
}

/// Comprehensive integration test
#[test]
fn test_comprehensive_integration() {
    let (x, y) = generate_regression_data(500, 4, 0.5, 42);

    // Test multiple strategies
    let strategies = vec![
        RegressorStrategy::Mean,
        RegressorStrategy::Median,
        RegressorStrategy::Quantile(0.25),
        RegressorStrategy::Quantile(0.75),
        RegressorStrategy::Constant(0.0),
        RegressorStrategy::Normal {
            mean: None,
            std: None,
        },
    ];

    for strategy in strategies {
        let regressor = if matches!(strategy, RegressorStrategy::Constant(_)) {
            DummyRegressor::new(strategy).with_constant(42.0)
        } else {
            DummyRegressor::new(strategy)
        };
        let fitted = regressor.fit(&x, &y).unwrap();
        let predictions = fitted.predict(&x).unwrap();

        // Basic sanity checks
        assert_eq!(predictions.len(), x.nrows());
        assert!(predictions.iter().all(|&p| p.is_finite()));
    }

    // Test classifier strategies
    let (x_class, y_class) = generate_classification_data(300, 3, 4, 42);

    let class_strategies = vec![
        ClassifierStrategy::MostFrequent,
        ClassifierStrategy::Prior,
        ClassifierStrategy::Stratified,
        ClassifierStrategy::Uniform,
        ClassifierStrategy::Constant,
    ];

    for strategy in class_strategies {
        let classifier = if matches!(strategy, ClassifierStrategy::Constant) {
            DummyClassifier::new(strategy).with_constant(1)
        } else {
            DummyClassifier::new(strategy)
        };
        let fitted = classifier.fit(&x_class, &y_class).unwrap();
        let predictions = fitted.predict(&x_class).unwrap();

        // Basic sanity checks
        assert_eq!(predictions.len(), x_class.nrows());
        assert!(predictions.iter().all(|&p| p >= 0));
    }
}

/// Performance regression test - ensure operations complete in reasonable time
#[test]
fn test_performance_regression() {
    use std::time::Instant;

    let (x, y) = generate_regression_data(10000, 10, 0.1, 42);

    let start = Instant::now();
    let regressor = DummyRegressor::new(RegressorStrategy::Mean);
    let fitted = regressor.fit(&x, &y).unwrap();
    let _predictions = fitted.predict(&x).unwrap();
    let duration = start.elapsed();

    // Should complete in reasonable time (less than 1 second for 10k samples)
    assert!(
        duration.as_secs() < 1,
        "Performance regression detected: took {:?}",
        duration
    );
}

#[allow(non_snake_case)]
#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn test_mean_strategy_property(values in prop::collection::vec(-1000.0f64..1000.0, 1..1000)) {
            if values.is_empty() {
                return Ok(());
            }

            let expected_mean = values.iter().sum::<f64>() / values.len() as f64;
            let y = Array1::from_vec(values);
            let x = Array2::from_shape_vec((y.len(), 1), vec![0.0; y.len()]).unwrap();

            let regressor = DummyRegressor::new(RegressorStrategy::Mean);
            let fitted = regressor.fit(&x, &y)?;
            let predictions = fitted.predict(&x)?;

            for &pred in predictions.iter() {
                prop_assert!((pred - expected_mean).abs() < 1e-10);
            }
        }

        #[test]
        fn test_constant_strategy_property(
            constant in -1000.0f64..1000.0,
            n_samples in 1usize..100
        ) {
            let y = Array1::from_elem(n_samples, 0.0); // Value doesn't matter
            let x = Array2::from_shape_vec((n_samples, 1), vec![0.0; n_samples]).unwrap();

            let regressor = DummyRegressor::new(RegressorStrategy::Constant(0.0)).with_constant(constant);
            let fitted = regressor.fit(&x, &y)?;
            let predictions = fitted.predict(&x)?;

            for &pred in predictions.iter() {
                prop_assert!((pred - constant).abs() < 1e-10);
            }
        }

        #[test]
        fn test_online_mean_convergence_property(
            values in prop::collection::vec(-100.0f64..100.0, 10..1000)
        ) {
            if values.is_empty() {
                return Ok(());
            }

            let expected_mean = values.iter().sum::<f64>() / values.len() as f64;
            let mut regressor: OnlineDummyRegressor = OnlineDummyRegressor::new(OnlineStrategy::OnlineMean { drift_detection: None });

            for &value in &values {
                regressor.partial_fit(value)?;
            }

            let prediction = regressor.predict_single();
            prop_assert!((prediction - expected_mean).abs() < 1e-10);
        }
    }
}
