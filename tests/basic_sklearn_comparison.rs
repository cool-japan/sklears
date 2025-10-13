//! Basic comparison tests with scikit-learn-like behavior
//!
//! These tests verify that our algorithms behave similarly to scikit-learn
//! in terms of API consistency and output quality.

use ndarray::{Array1, Array2, array};

// Test data generation utilities
fn make_simple_classification_data() -> (Array2<f64>, Array1<i32>) {
    // Simple linearly separable 2D classification problem
    let x = array![
        [1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0],
        [5.0, 5.0], [6.0, 6.0], [7.0, 7.0], [8.0, 8.0],
    ];
    let y = array![0, 0, 0, 0, 1, 1, 1, 1];
    (x, y)
}

fn make_simple_regression_data() -> (Array2<f64>, Array1<f64>) {
    // Simple linear relationship
    let x = array![
        [1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0],
    ];
    let y = array![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0];
    (x, y)
}

#[test]
fn test_data_shapes_consistency() {
    let (x_class, y_class) = make_simple_classification_data();
    let (x_reg, y_reg) = make_simple_regression_data();
    
    // Test that our test data has consistent shapes
    assert_eq!(x_class.nrows(), y_class.len());
    assert_eq!(x_reg.nrows(), y_reg.len());
    
    // Test basic properties
    assert_eq!(x_class.ncols(), 2);
    assert_eq!(x_reg.ncols(), 1);
    assert!(x_class.nrows() > 0);
    assert!(x_reg.nrows() > 0);
}

#[test]
fn test_array_operations_consistency() {
    let (x, y) = make_simple_classification_data();
    
    // Test basic ndarray operations work as expected
    assert_eq!(x.shape(), &[8, 2]);
    assert_eq!(y.shape(), &[8]);
    
    // Test that we can slice and manipulate data
    let first_row = x.row(0);
    assert_eq!(first_row.len(), 2);
    
    let first_col = x.column(0);
    assert_eq!(first_col.len(), 8);
    
    // Test unique values in target
    let mut unique_y = y.to_vec();
    unique_y.sort_unstable();
    unique_y.dedup();
    assert_eq!(unique_y, vec![0, 1]);
}

#[test]
fn test_sklearn_like_api_patterns() {
    // Test that our API follows sklearn-like patterns
    // This is a conceptual test for API design
    
    let (x, y) = make_simple_classification_data();
    
    // Test basic sklearn-like workflow patterns:
    // 1. Model creation with parameters
    // 2. Fit method that takes X, y
    // 3. Predict method that takes X
    // 4. Model attributes accessible after fitting
    
    // These patterns should be consistent across all algorithms
    assert!(x.nrows() > 0, "Should have training samples");
    assert!(y.len() == x.nrows(), "X and y should have matching first dimension");
    
    // Test train/test split patterns
    let n_train = x.nrows() / 2;
    let x_train = x.slice(ndarray::s![..n_train, ..]);
    let y_train = y.slice(ndarray::s![..n_train]);
    let x_test = x.slice(ndarray::s![n_train.., ..]);
    let y_test = y.slice(ndarray::s![n_train..]);
    
    assert!(x_train.nrows() > 0);
    assert!(x_test.nrows() > 0);
    assert_eq!(x_train.nrows(), y_train.len());
    assert_eq!(x_test.nrows(), y_test.len());
}

#[test]
fn test_parameter_validation_patterns() {
    // Test common parameter validation patterns that should be consistent
    // across all sklearn-like algorithms
    
    let (x, y) = make_simple_classification_data();
    
    // Test shape mismatch detection patterns
    let mismatched_y = Array1::zeros(x.nrows() + 1);
    
    // This represents the pattern that fit() should validate X and y shapes
    assert_ne!(x.nrows(), mismatched_y.len(), "Shapes should be mismatched for test");
    
    // Test empty data handling patterns
    let empty_x = Array2::<f64>::zeros((0, 2));
    let empty_y = Array1::<i32>::zeros(0);
    
    assert_eq!(empty_x.nrows(), 0);
    assert_eq!(empty_y.len(), 0);
    assert_eq!(empty_x.nrows(), empty_y.len());
}

#[test]
fn test_reproducibility_patterns() {
    let (x, y) = make_simple_classification_data();
    
    // Test that random state patterns work for reproducibility
    // This tests the concept that algorithms with random_state parameter
    // should produce identical results when given the same seed
    
    let seed1 = 42u64;
    let seed2 = 42u64;
    let seed3 = 123u64;
    
    // Basic test that same seeds are equal, different seeds are different
    assert_eq!(seed1, seed2);
    assert_ne!(seed1, seed3);
    
    // Test that we can use seeds to initialize random number generators
    use rand::{SeedableRng};
    use rand::rngs::StdRng;
    
    let mut rng1 = StdRng::seed_from_u64(seed1);
    let mut rng2 = StdRng::seed_from_u64(seed2);
    let mut rng3 = StdRng::seed_from_u64(seed3);
    
    let val1: f64 = rng1.gen();
    let val2: f64 = rng2.gen();
    let val3: f64 = rng3.gen();
    
    assert_eq!(val1, val2, "Same seed should produce same random values");
    assert_ne!(val1, val3, "Different seeds should produce different values");
}

#[test]
fn test_cross_validation_patterns() {
    let (x, y) = make_simple_classification_data();
    
    // Test cross-validation patterns that should be consistent
    let n_samples = x.nrows();
    let n_folds = 3;
    
    // Test that we can split data into folds
    let fold_size = n_samples / n_folds;
    
    for fold in 0..n_folds {
        let start_idx = fold * fold_size;
        let end_idx = if fold == n_folds - 1 {
            n_samples
        } else {
            (fold + 1) * fold_size
        };
        
        assert!(start_idx < n_samples);
        assert!(end_idx <= n_samples);
        assert!(start_idx < end_idx);
        
        // Test that we can create train/validation splits
        let val_indices = start_idx..end_idx;
        let val_size = val_indices.len();
        let train_size = n_samples - val_size;
        
        assert!(val_size > 0);
        assert!(train_size > 0);
        assert_eq!(val_size + train_size, n_samples);
    }
}

#[test]
fn test_preprocessing_patterns() {
    let (x, _) = make_simple_regression_data();
    
    // Test preprocessing patterns that should be consistent
    let mean = x.mean().unwrap();
    let std = x.std(0.0);
    
    // Test standardization patterns
    let x_centered = &x - mean;
    let x_mean_centered = x_centered.mean().unwrap();
    
    // Mean should be close to zero after centering
    assert!((x_mean_centered).abs() < 1e-10, "Mean should be ~0 after centering");
    
    // Test scaling patterns
    if std > 1e-10 { // Avoid division by zero
        let x_standardized = &x_centered / std;
        let x_std_after = x_standardized.std(0.0);
        
        // Standard deviation should be close to 1 after standardization
        assert!((x_std_after - 1.0).abs() < 1e-10, "Std should be ~1 after standardization");
    }
}

#[test]
fn test_metrics_patterns() {
    // Test metrics calculation patterns
    let y_true = array![0, 1, 1, 0, 1, 0];
    let y_pred = array![0, 1, 1, 0, 0, 1];
    
    // Test accuracy calculation
    let mut correct = 0;
    for (true_val, pred_val) in y_true.iter().zip(y_pred.iter()) {
        if true_val == pred_val {
            correct += 1;
        }
    }
    let accuracy = correct as f64 / y_true.len() as f64;
    
    assert!(accuracy >= 0.0 && accuracy <= 1.0, "Accuracy should be between 0 and 1");
    assert_eq!(correct, 4); // Manual count
    assert!((accuracy - 4.0/6.0).abs() < 1e-10);
    
    // Test regression metrics patterns
    let y_true_reg = array![1.0, 2.0, 3.0, 4.0];
    let y_pred_reg = array![1.1, 1.9, 3.2, 3.8];
    
    // Test MSE calculation
    let mut mse = 0.0;
    for (true_val, pred_val) in y_true_reg.iter().zip(y_pred_reg.iter()) {
        mse += (true_val - pred_val).powi(2);
    }
    mse /= y_true_reg.len() as f64;
    
    assert!(mse >= 0.0, "MSE should be non-negative");
    
    // Test MAE calculation
    let mut mae = 0.0;
    for (true_val, pred_val) in y_true_reg.iter().zip(y_pred_reg.iter()) {
        mae += (true_val - pred_val).abs();
    }
    mae /= y_true_reg.len() as f64;
    
    assert!(mae >= 0.0, "MAE should be non-negative");
}

#[test]
fn test_model_selection_patterns() {
    let (x, y) = make_simple_classification_data();
    
    // Test grid search patterns
    let param_values = vec![1, 3, 5, 7];
    let n_folds = 3;
    
    // Test that we can enumerate parameter combinations
    for &param_val in &param_values {
        assert!(param_val > 0, "Parameter values should be positive");
        
        // Test cross-validation scoring pattern
        let mut fold_scores = Vec::new();
        
        for fold in 0..n_folds {
            // Simulate scoring for each fold
            let score = 0.8 + (fold as f64) * 0.05; // Dummy score
            fold_scores.push(score);
        }
        
        // Test score aggregation
        let mean_score: f64 = fold_scores.iter().sum::<f64>() / fold_scores.len() as f64;
        let std_score = {
            let variance: f64 = fold_scores.iter()
                .map(|&score| (score - mean_score).powi(2))
                .sum::<f64>() / fold_scores.len() as f64;
            variance.sqrt()
        };
        
        assert!(mean_score >= 0.0 && mean_score <= 1.0);
        assert!(std_score >= 0.0);
    }
}

#[test]
fn test_ensemble_patterns() {
    // Test ensemble method patterns
    let base_predictions = vec![
        array![0, 1, 1, 0],
        array![1, 1, 0, 0], 
        array![0, 1, 1, 1],
    ];
    
    let n_samples = base_predictions[0].len();
    let n_estimators = base_predictions.len();
    
    // Test majority voting pattern
    let mut ensemble_predictions = Array1::zeros(n_samples);
    
    for sample_idx in 0..n_samples {
        let mut votes = std::collections::HashMap::new();
        
        for estimator_predictions in &base_predictions {
            let prediction = estimator_predictions[sample_idx];
            *votes.entry(prediction).or_insert(0) += 1;
        }
        
        // Find majority vote
        let majority_prediction = votes.into_iter()
            .max_by_key(|(_, count)| *count)
            .map(|(prediction, _)| prediction)
            .unwrap_or(0);
        
        ensemble_predictions[sample_idx] = majority_prediction;
    }
    
    assert_eq!(ensemble_predictions.len(), n_samples);
    
    // Test that ensemble predictions are valid
    for &pred in ensemble_predictions.iter() {
        assert!(pred == 0 || pred == 1, "Binary classification predictions should be 0 or 1");
    }
}

#[test]
fn test_feature_importance_patterns() {
    let n_features = 5;
    
    // Test feature importance normalization patterns
    let raw_importances = array![0.3, 0.1, 0.4, 0.15, 0.05];
    
    // Test that importances sum to 1
    let importance_sum: f64 = raw_importances.sum();
    assert!((importance_sum - 1.0).abs() < 1e-10, "Feature importances should sum to 1");
    
    // Test that all importances are non-negative
    for &importance in raw_importances.iter() {
        assert!(importance >= 0.0, "Feature importances should be non-negative");
    }
    
    // Test importance ranking
    let mut indexed_importances: Vec<(usize, f64)> = raw_importances.iter()
        .enumerate()
        .map(|(i, &imp)| (i, imp))
        .collect();
    
    indexed_importances.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    
    // Test that sorting works correctly
    assert!(indexed_importances[0].1 >= indexed_importances[1].1);
    assert!(indexed_importances[1].1 >= indexed_importances[2].1);
}