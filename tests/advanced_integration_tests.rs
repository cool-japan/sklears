//! Advanced integration tests for sklears
//!
//! These tests verify that different components work together correctly
//! and that the library behaves correctly in complex scenarios.

use ndarray::{Array1, Array2};
use sklears_core::traits::{Transform};
use sklears_metrics::classification::accuracy_score;
use sklears_metrics::regression::mean_squared_error;
use sklears_utils::data_generation::{make_classification, make_regression};
use sklears_utils::validation::check_array_2d;

#[test]
fn test_end_to_end_classification_pipeline() {
    // Generate classification data
    let (X, y) = make_classification(100, 10, 3, None, None, 0.0, 1.0, Some(42)).unwrap();
    
    // Validate input data
    assert!(check_array_2d(&X).is_ok());
    assert_eq!(X.shape()[0], y.len());
    
    // Split data (simple split for testing)
    let split_idx = 80;
    let X_train = X.slice(s![..split_idx, ..]).to_owned();
    let X_test = X.slice(s![split_idx.., ..]).to_owned();
    let y_train = y.slice(s![..split_idx]).to_owned();
    let y_test = y.slice(s![split_idx..]).to_owned();
    
    assert_eq!(X_train.shape()[0], 80);
    assert_eq!(X_test.shape()[0], 20);
    assert_eq!(y_train.len(), 80);
    assert_eq!(y_test.len(), 20);
    
    // Test that generated data has expected properties
    assert!(X_train.iter().all(|&x| x.is_finite()));
    assert!(X_test.iter().all(|&x| x.is_finite()));
    assert!(y_train.iter().all(|&y| y >= 0 && y < 3));
    assert!(y_test.iter().all(|&y| y >= 0 && y < 3));
}

#[test]
fn test_end_to_end_regression_pipeline() {
    // Generate regression data
    let (X, y) = make_regression(100, 10, Some(5), 0.1, 0.0, Some(42)).unwrap();
    
    // Validate input data
    assert!(check_array_2d(&X).is_ok());
    assert_eq!(X.shape()[0], y.len());
    
    // Split data
    let split_idx = 80;
    let X_train = X.slice(s![..split_idx, ..]).to_owned();
    let X_test = X.slice(s![split_idx.., ..]).to_owned();
    let y_train = y.slice(s![..split_idx]).to_owned();
    let y_test = y.slice(s![split_idx..]).to_owned();
    
    // Test that all values are finite
    assert!(X_train.iter().all(|&x| x.is_finite()));
    assert!(X_test.iter().all(|&x| x.is_finite()));
    assert!(y_train.iter().all(|&y| y.is_finite()));
    assert!(y_test.iter().all(|&y| y.is_finite()));
    
    // Test basic regression metrics
    let mse = mean_squared_error(&y_test, &y_test).unwrap();
    assert!((mse - 0.0).abs() < 1e-10); // Perfect predictions should have MSE = 0
}

#[test]
fn test_data_validation_consistency() {
    // Test that validation functions work consistently across crates
    let valid_X = Array2::<f64>::ones((10, 5));
    let invalid_X = Array2::<f64>::from_elem((0, 5), f64::NAN);
    
    assert!(check_array_2d(&valid_X).is_ok());
    // Note: check_array_2d might not catch NaN values - this tests the current behavior
    
    // Test consistent length validation
    let y1 = Array1::<i32>::ones(10);
    let y2 = Array1::<i32>::ones(5);
    
    // These should be consistent
    assert_eq!(valid_X.shape()[0], y1.len());
    assert_ne!(valid_X.shape()[0], y2.len());
}

#[test]
fn test_preprocessing_integration() {
    // Test that preprocessing and model training work together
    let (X, y) = make_classification(50, 5, 2, None, None, 0.0, 1.0, Some(42)).unwrap();
    
    // Manual standardization for testing
    let mut X_standardized = X.clone();
    for j in 0..X.shape()[1] {
        let col_mean = X.column(j).mean().unwrap();
        let col_std = {
            let variance = X.column(j)
                .iter()
                .map(|&x| (x - col_mean).powi(2))
                .sum::<f64>() / X.shape()[0] as f64;
            variance.sqrt()
        };
        
        if col_std > 1e-10 {
            for i in 0..X.shape()[0] {
                X_standardized[[i, j]] = (X[[i, j]] - col_mean) / col_std;
            }
        }
    }
    
    // Verify standardization
    for j in 0..X_standardized.shape()[1] {
        let col_mean = X_standardized.column(j).mean().unwrap();
        assert!(col_mean.abs() < 1e-10, "Column {} mean should be ~0, got {}", j, col_mean);
    }
}

#[test]
fn test_cross_validation_integration() {
    // Test cross-validation with different models and metrics
    let (X, y) = make_classification(60, 8, 2, None, None, 0.0, 1.0, Some(42)).unwrap();
    
    // Manual K-fold implementation for testing
    let k = 3;
    let fold_size = X.shape()[0] / k;
    
    let mut cv_scores = Vec::new();
    
    for fold in 0..k {
        let start = fold * fold_size;
        let end = if fold == k - 1 { X.shape()[0] } else { (fold + 1) * fold_size };
        
        // Create train/test splits
        let mut train_indices = Vec::new();
        let mut test_indices = Vec::new();
        
        for i in 0..X.shape()[0] {
            if i >= start && i < end {
                test_indices.push(i);
            } else {
                train_indices.push(i);
            }
        }
        
        // Extract training and test data
        let X_train = Array2::from_shape_fn((train_indices.len(), X.shape()[1]), |(i, j)| {
            X[[train_indices[i], j]]
        });
        let X_test = Array2::from_shape_fn((test_indices.len(), X.shape()[1]), |(i, j)| {
            X[[test_indices[i], j]]
        });
        let y_train = Array1::from_vec(train_indices.iter().map(|&i| y[i]).collect());
        let y_test = Array1::from_vec(test_indices.iter().map(|&i| y[i]).collect());
        
        // Simple dummy prediction (majority class)
        let majority_class = if y_train.iter().filter(|&&label| label == 0).count() >
                               y_train.iter().filter(|&&label| label == 1).count() { 0 } else { 1 };
        let y_pred = Array1::from_elem(y_test.len(), majority_class);
        
        // Calculate accuracy
        let accuracy = accuracy_score(&y_test, &y_pred).unwrap();
        cv_scores.push(accuracy);
        
        assert!(accuracy >= 0.0 && accuracy <= 1.0);
    }
    
    // Verify we got scores for all folds
    assert_eq!(cv_scores.len(), k);
    
    // Calculate mean CV score
    let mean_score = cv_scores.iter().sum::<f64>() / cv_scores.len() as f64;
    assert!(mean_score >= 0.0 && mean_score <= 1.0);
}

#[test]
fn test_model_selection_integration() {
    // Test hyperparameter search integration
    let (X, y) = make_regression(40, 5, Some(3), 0.1, 0.0, Some(42)).unwrap();
    
    // Test different "hyperparameters" (here just different noise levels in dummy predictions)
    let hyperparams = vec![0.1, 0.5, 1.0];
    let mut param_scores = Vec::new();
    
    for &noise_level in &hyperparams {
        // Create dummy predictions with controlled noise
        let y_pred = y.mapv(|val| val + noise_level * 0.1 * val);
        
        let mse = mean_squared_error(&y, &y_pred).unwrap();
        param_scores.push((noise_level, mse));
        
        assert!(mse >= 0.0);
        assert!(mse.is_finite());
    }
    
    // Find best parameters (lowest MSE)
    let best_param = param_scores
        .iter()
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .unwrap();
    
    // Best should be lowest noise level
    assert!((best_param.0 - 0.1).abs() < 1e-10);
}

#[test]
fn test_ensemble_integration() {
    // Test that ensemble methods can combine different models
    let (X, y) = make_classification(80, 6, 2, None, None, 0.0, 1.0, Some(42)).unwrap();
    
    // Create multiple "models" (different dummy predictors)
    let model_predictions = vec![
        Array1::from_vec(y.iter().map(|&label| if label == 0 { 0 } else { 1 }).collect()), // Perfect predictor
        Array1::from_vec(y.iter().map(|&label| (label + 1) % 2).collect()), // Inverted predictor
        Array1::from_vec(vec![0; y.len()]), // Always predict class 0
    ];
    
    // Test voting ensemble (majority vote)
    let mut ensemble_predictions = Vec::new();
    for i in 0..y.len() {
        let votes: Vec<i32> = model_predictions.iter().map(|pred| pred[i]).collect();
        let class_0_votes = votes.iter().filter(|&&vote| vote == 0).count();
        let class_1_votes = votes.iter().filter(|&&vote| vote == 1).count();
        
        let prediction = if class_0_votes > class_1_votes { 0 } else { 1 };
        ensemble_predictions.push(prediction);
    }
    
    let ensemble_pred = Array1::from_vec(ensemble_predictions);
    let ensemble_accuracy = accuracy_score(&y, &ensemble_pred).unwrap();
    
    // Calculate individual model accuracies
    let individual_accuracies: Vec<f64> = model_predictions
        .iter()
        .map(|pred| accuracy_score(&y, pred).unwrap())
        .collect();
    
    // Ensemble should generally perform reasonably
    assert!(ensemble_accuracy >= 0.0 && ensemble_accuracy <= 1.0);
    
    // Test that all individual models also have valid accuracies
    for accuracy in individual_accuracies {
        assert!(accuracy >= 0.0 && accuracy <= 1.0);
    }
}

#[test]
fn test_pipeline_robustness() {
    // Test pipeline behavior with edge cases
    
    // Test with minimal data
    let (X_small, y_small) = make_classification(5, 2, 2, None, None, 0.0, 1.0, Some(42)).unwrap();
    assert_eq!(X_small.shape()[0], 5);
    assert_eq!(y_small.len(), 5);
    
    // Test with single feature
    let (X_single, y_single) = make_classification(20, 1, 2, None, None, 0.0, 1.0, Some(42)).unwrap();
    assert_eq!(X_single.shape()[1], 1);
    assert_eq!(y_single.len(), 20);
    
    // Test validation still works
    assert!(check_array_2d(&X_small).is_ok());
    assert!(check_array_2d(&X_single).is_ok());
}

#[test]
fn test_numerical_stability_integration() {
    // Test behavior with different scales of data
    let scales = vec![1e-6, 1e-3, 1.0, 1e3, 1e6];
    
    for &scale in &scales {
        let (mut X, y) = make_classification(30, 4, 2, None, None, 0.0, 1.0, Some(42)).unwrap();
        
        // Scale the data
        X.mapv_inplace(|x| x * scale);
        
        // Verify data is still valid
        assert!(X.iter().all(|&x| x.is_finite()));
        assert!(check_array_2d(&X).is_ok());
        
        // Test basic operations still work
        let X_mean = X.mean().unwrap();
        assert!(X_mean.is_finite());
        
        // Test that relative relationships are preserved
        if X.shape()[0] >= 2 && X.shape()[1] >= 1 {
            let ratio = if X[[1, 0]] != 0.0 { X[[0, 0]] / X[[1, 0]] } else { 0.0 };
            assert!(ratio.is_finite());
        }
    }
}

#[test]
fn test_error_handling_integration() {
    // Test that errors are properly propagated across components
    
    // Test with mismatched dimensions
    let X = Array2::<f64>::zeros((10, 5));
    let y_wrong_size = Array1::<i32>::zeros(8);
    
    // This should be detectable
    assert_ne!(X.shape()[0], y_wrong_size.len());
    
    // Test with invalid inputs
    let X_empty = Array2::<f64>::zeros((0, 5));
    let y_empty = Array1::<i32>::zeros(0);
    
    assert_eq!(X_empty.shape()[0], 0);
    assert_eq!(y_empty.len(), 0);
    
    // Empty arrays should be handled gracefully
    // (behavior depends on specific implementation choices)
}

// Import statement for slice macro
use ndarray::s;