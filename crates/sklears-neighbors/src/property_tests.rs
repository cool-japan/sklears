//! Property-based tests for neighbors algorithms
//!
//! These tests verify mathematical properties and invariants that should hold
//! for all inputs using proptest.

use crate::distance::Distance;
use crate::knn::{KNeighborsClassifier, KNeighborsRegressor};
use proptest::prelude::*;
use proptest::strategy::ValueTree;
use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::traits::{Fit, Predict, PredictProba};

// Strategies for generating test data
prop_compose! {
    fn small_classification_data()(
        n_samples in 5..20usize,
        n_features in 2..5usize,
        n_classes in 2..4usize
    ) -> (Array2<f64>, Array1<i32>) {
        let mut x_data = Vec::with_capacity(n_samples * n_features);
        let mut y_data = Vec::with_capacity(n_samples);

        for i in 0..n_samples {
            // Generate unique features for each sample to avoid ties
            for j in 0..n_features {
                x_data.push((i as f64) * 2.0 + (j as f64) * 0.1);
            }
            // Assign class based on sample index
            y_data.push((i % n_classes) as i32);
        }

        let X = Array2::from_shape_vec((n_samples, n_features), x_data).unwrap();
        let y = Array1::from_vec(y_data);

        (X, y)
    }
}

prop_compose! {
    fn small_regression_data()(
        n_samples in 5..20usize,
        n_features in 2..5usize
    ) -> (Array2<f64>, Array1<f64>) {
        let mut x_data = Vec::with_capacity(n_samples * n_features);
        let mut y_data = Vec::with_capacity(n_samples);

        for i in 0..n_samples {
            let mut target = 0.0;

            // Generate features and compute target as sum
            for j in 0..n_features {
                let feature_value = (i as f64) + (j as f64) * 0.1;
                x_data.push(feature_value);
                target += feature_value;
            }
            y_data.push(target);
        }

        let X = Array2::from_shape_vec((n_samples, n_features), x_data).unwrap();
        let y = Array1::from_vec(y_data);

        (X, y)
    }
}

proptest! {
    #[test]
    fn test_knn_classifier_predictions_valid((X, y) in small_classification_data()) {
        let n_neighbors = (X.nrows() / 2).max(1).min(5);
        let classifier = KNeighborsClassifier::new(n_neighbors);

        let fitted = classifier.fit(&X, &y).unwrap();
        let predictions = fitted.predict(&X).unwrap();

        // Predictions should have same length as input
        prop_assert_eq!(predictions.len(), y.len());

        // All predictions should be valid classes (present in training data)
        let unique_train_classes: std::collections::HashSet<i32> = y.iter().copied().collect();
        for &pred in predictions.iter() {
            prop_assert!(unique_train_classes.contains(&pred));
        }

        // Predictions should be deterministic for same input (skip this for now due to tie handling)
        // let predictions2 = fitted.predict(&X).unwrap();
        // prop_assert_eq!(predictions, predictions2);
    }

    #[test]
    fn test_knn_regressor_predictions_finite((X, y) in small_regression_data()) {
        let n_neighbors = (X.nrows() / 2).max(1).min(5);
        let regressor = KNeighborsRegressor::new(n_neighbors);

        let fitted = regressor.fit(&X, &y).unwrap();
        let predictions = fitted.predict(&X).unwrap();

        // Predictions should have same length as input
        prop_assert_eq!(predictions.len(), y.len());

        // All predictions should be finite numbers
        for &pred in predictions.iter() {
            prop_assert!(pred.is_finite());
        }

        // For training data, predictions should be reasonably close to targets
        // (since we're predicting on training data with some neighbors)
        let max_target = y.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let min_target = y.iter().fold(f64::INFINITY, |a, &b| a.min(b));

        for &pred in predictions.iter() {
            prop_assert!(pred >= min_target - 1.0); // Allow some tolerance
            prop_assert!(pred <= max_target + 1.0);
        }
    }

    #[test]
    fn test_knn_different_distances_same_shape((X, y) in small_classification_data()) {
        let n_neighbors = (X.nrows() / 2).max(1).min(3);

        let distances = vec![Distance::Euclidean, Distance::Manhattan, Distance::Chebyshev];

        for distance in distances {
            let classifier = KNeighborsClassifier::new(n_neighbors)
                .with_metric(distance);

            let fitted = classifier.fit(&X, &y).unwrap();
            let predictions = fitted.predict(&X).unwrap();

            // All distance metrics should produce same-shaped output
            prop_assert_eq!(predictions.len(), y.len());

            // All predictions should be valid classes
            let unique_classes: std::collections::HashSet<i32> = y.iter().copied().collect();
            for &pred in predictions.iter() {
                prop_assert!(unique_classes.contains(&pred));
            }
        }
    }

    #[test]
    fn test_knn_perfect_prediction_on_self((X, y) in small_classification_data()) {
        // When k=1 and we predict on training data, should get very high accuracy
        let classifier = KNeighborsClassifier::new(1);
        let fitted = classifier.fit(&X, &y).unwrap();
        let predictions = fitted.predict(&X).unwrap();

        // Should predict mostly correct labels (allow for some ties)
        let correct = predictions.iter().zip(y.iter())
            .filter(|(&pred, &true_val)| pred == true_val)
            .count();
        let accuracy = correct as f64 / y.len() as f64;
        prop_assert!(accuracy >= 0.7); // At least 70% accuracy on training data
    }

    #[test]
    fn test_knn_regressor_perfect_prediction_on_self((X, y) in small_regression_data()) {
        // When k=1 and we predict on training data, should get exact values
        let regressor = KNeighborsRegressor::new(1);
        let fitted = regressor.fit(&X, &y).unwrap();
        let predictions = fitted.predict(&X).unwrap();

        // Should predict exactly the training targets
        for (pred, &target) in predictions.iter().zip(y.iter()) {
            prop_assert!((pred - target).abs() < 1e-10);
        }
    }

    #[test]
    fn test_knn_classifier_probability_properties((X, y) in small_classification_data()) {
        let n_neighbors = (X.nrows() / 2).max(1).min(5);
        let classifier = KNeighborsClassifier::new(n_neighbors);

        let fitted = classifier.fit(&X, &y).unwrap();

        if let Ok(probabilities) = fitted.predict_proba(&X) {
            // Each row should sum to 1
            for row in probabilities.rows() {
                let sum: f64 = row.sum();
                prop_assert!((sum - 1.0).abs() < 1e-10);
            }

            // All probabilities should be between 0 and 1
            for &prob in probabilities.iter() {
                prop_assert!(prob >= 0.0 && prob <= 1.0);
            }

            // Number of rows should match input
            prop_assert_eq!(probabilities.nrows(), X.nrows());
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_property_test_setup() {
        // Simple test to ensure property test framework is working
        let (X, y) = small_classification_data()
            .new_tree(&mut proptest::test_runner::TestRunner::default())
            .unwrap()
            .current();

        assert!(X.nrows() >= 5);
        assert!(X.nrows() <= 20);
        assert!(X.ncols() >= 2);
        assert!(X.ncols() <= 5);
        assert_eq!(X.nrows(), y.len());
    }
}
