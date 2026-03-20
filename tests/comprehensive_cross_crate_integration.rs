//! Comprehensive integration tests across multiple sklears crates.
//!
//! This test suite validates that different crates work together seamlessly
//! to create complete machine learning pipelines.

use approx::assert_abs_diff_eq;
use ndarray::{array, Array1, Array2};
use sklears_core::traits::{Transform};
use sklears_neighbors::{KNNClassifier, KNNRegressor};
use sklears_preprocessing::{StandardScaler, OneHotEncoder, SimpleImputer};
use sklears_tree::{DecisionTreeClassifier, RandomForestRegressor};
use sklears_metrics::{accuracy_score, mean_squared_error, r2_score};
use sklears_utils::{train_test_split, make_classification, make_regression};

#[test]
fn test_complete_classification_pipeline() {
    // Generate synthetic classification data
    let (x, y) = make_classification(100, 4, 2, 0, 42).unwrap();
    
    // Split data
    let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.3, Some(42)).unwrap();
    
    // Create preprocessing pipeline
    let mut scaler = StandardScaler::new();
    let x_train_scaled = scaler.fit_transform(&x_train).unwrap();
    let x_test_scaled = scaler.transform(&x_test).unwrap();
    
    // Train multiple models
    let knn = KNNClassifier::new(5).fit(&x_train_scaled, &y_train).unwrap();
    let tree = DecisionTreeClassifier::new().fit(&x_train_scaled, &y_train).unwrap();
    
    // Make predictions
    let knn_pred = knn.predict(&x_test_scaled).unwrap();
    let tree_pred = tree.predict(&x_test_scaled).unwrap();
    
    // Evaluate
    let knn_acc = accuracy_score(&y_test, &knn_pred);
    let tree_acc = accuracy_score(&y_test, &tree_pred);
    
    // Basic sanity checks
    assert!(knn_acc >= 0.0 && knn_acc <= 1.0);
    assert!(tree_acc >= 0.0 && tree_acc <= 1.0);
    assert_eq!(knn_pred.len(), y_test.len());
    assert_eq!(tree_pred.len(), y_test.len());
    
    println!("Classification Pipeline Results:");
    println!("  KNN Accuracy: {:.3}", knn_acc);
    println!("  Decision Tree Accuracy: {:.3}", tree_acc);
}

#[test]
fn test_complete_regression_pipeline() {
    // Generate synthetic regression data
    let (x, y) = make_regression(80, 3, 1, 0.1, 42).unwrap();
    
    // Split data
    let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.25, Some(42)).unwrap();
    
    // Preprocessing
    let mut scaler = StandardScaler::new();
    let x_train_scaled = scaler.fit_transform(&x_train).unwrap();
    let x_test_scaled = scaler.transform(&x_test).unwrap();
    
    // Train models
    let knn = KNNRegressor::new(3).fit(&x_train_scaled, &y_train).unwrap();
    let rf = RandomForestRegressor::new()
        .n_estimators(10)
        .random_state(42)
        .fit(&x_train_scaled, &y_train).unwrap();
    
    // Predictions
    let knn_pred = knn.predict(&x_test_scaled).unwrap();
    let rf_pred = rf.predict(&x_test_scaled).unwrap();
    
    // Evaluate
    let knn_mse = mean_squared_error(&y_test, &knn_pred);
    let rf_mse = mean_squared_error(&y_test, &rf_pred);
    let knn_r2 = r2_score(&y_test, &knn_pred);
    let rf_r2 = r2_score(&y_test, &rf_pred);
    
    // Sanity checks
    assert!(knn_mse >= 0.0);
    assert!(rf_mse >= 0.0);
    assert!(knn_r2 <= 1.0);
    assert!(rf_r2 <= 1.0);
    assert_eq!(knn_pred.len(), y_test.len());
    assert_eq!(rf_pred.len(), y_test.len());
    
    println!("Regression Pipeline Results:");
    println!("  KNN MSE: {:.3}, R²: {:.3}", knn_mse, knn_r2);
    println!("  Random Forest MSE: {:.3}, R²: {:.3}", rf_mse, rf_r2);
}

#[test]
fn test_missing_data_pipeline() {
    // Create data with missing values
    let mut x = array![
        [1.0, 2.0, 3.0],
        [4.0, f64::NAN, 6.0],
        [7.0, 8.0, f64::NAN],
        [10.0, 11.0, 12.0],
        [f64::NAN, 14.0, 15.0],
    ];
    let y = vec![0, 1, 0, 1, 0];
    
    // Impute missing values
    let mut imputer = SimpleImputer::new().strategy("mean");
    let x_imputed = imputer.fit_transform(&x).unwrap();
    
    // Verify no NaN values remain
    for &val in x_imputed.iter() {
        assert!(!val.is_nan());
    }
    
    // Scale the data
    let mut scaler = StandardScaler::new();
    let x_processed = scaler.fit_transform(&x_imputed).unwrap();
    
    // Train classifier
    let classifier = KNNClassifier::new(3).fit(&x_processed, &y).unwrap();
    let predictions = classifier.predict(&x_processed).unwrap();
    
    // Basic checks
    assert_eq!(predictions.len(), y.len());
    for &pred in predictions.iter() {
        assert!(pred == 0 || pred == 1);
    }
    
    println!("Missing Data Pipeline: Successfully handled {} missing values", 
             x.iter().filter(|&&val| val.is_nan()).count());
}

#[test]
fn test_categorical_data_pipeline() {
    // Create mixed data (numerical + categorical)
    let x_numerical = array![
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
        [7.0, 8.0],
        [9.0, 10.0],
        [11.0, 12.0],
    ];
    
    let x_categorical = vec![
        vec![0, 1],  // Category A, SubCategory 1
        vec![1, 0],  // Category B, SubCategory 0
        vec![0, 1],  // Category A, SubCategory 1
        vec![2, 0],  // Category C, SubCategory 0
        vec![1, 1],  // Category B, SubCategory 1
        vec![2, 0],  // Category C, SubCategory 0
    ];
    
    let y = vec![0, 1, 0, 1, 1, 1];
    
    // Encode categorical variables
    let mut encoder = OneHotEncoder::new();
    let x_cat_encoded = encoder.fit_transform(&x_categorical).unwrap();
    
    // Combine numerical and encoded categorical features
    assert_eq!(x_numerical.nrows(), x_cat_encoded.nrows());
    let n_samples = x_numerical.nrows();
    let n_num_features = x_numerical.ncols();
    let n_cat_features = x_cat_encoded.ncols();
    
    let mut x_combined = Array2::zeros((n_samples, n_num_features + n_cat_features));
    
    // Copy numerical features
    for i in 0..n_samples {
        for j in 0..n_num_features {
            x_combined[[i, j]] = x_numerical[[i, j]];
        }
    }
    
    // Copy encoded categorical features
    for i in 0..n_samples {
        for j in 0..n_cat_features {
            x_combined[[i, n_num_features + j]] = x_cat_encoded[[i, j]];
        }
    }
    
    // Scale combined features
    let mut scaler = StandardScaler::new();
    let x_processed = scaler.fit_transform(&x_combined).unwrap();
    
    // Train classifier
    let classifier = DecisionTreeClassifier::new()
        .random_state(42)
        .fit(&x_processed, &y).unwrap();
    
    let predictions = classifier.predict(&x_processed).unwrap();
    let accuracy = accuracy_score(&y, &predictions);
    
    // Verify results
    assert_eq!(predictions.len(), y.len());
    assert!(accuracy >= 0.0 && accuracy <= 1.0);
    
    println!("Categorical Data Pipeline:");
    println!("  Original features: {} numerical + {} categorical", n_num_features, 2);
    println!("  Encoded features: {}", x_combined.ncols());
    println!("  Accuracy: {:.3}", accuracy);
}

#[test]
fn test_model_comparison_pipeline() {
    // Generate comparison dataset
    let (x, y) = make_classification(150, 6, 3, 0, 123).unwrap();
    let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.3, Some(123)).unwrap();
    
    // Preprocess
    let mut scaler = StandardScaler::new();
    let x_train_scaled = scaler.fit_transform(&x_train).unwrap();
    let x_test_scaled = scaler.transform(&x_test).unwrap();
    
    // Train multiple models with different configurations
    let models = vec![
        ("KNN-3", Box::new(KNNClassifier::new(3).fit(&x_train_scaled, &y_train).unwrap()) as Box<dyn Predict<Array2<f64>, Vec<usize>>>),
        ("KNN-5", Box::new(KNNClassifier::new(5).fit(&x_train_scaled, &y_train).unwrap())),
        ("KNN-7", Box::new(KNNClassifier::new(7).fit(&x_train_scaled, &y_train).unwrap())),
        ("DecisionTree", Box::new(DecisionTreeClassifier::new().random_state(123).fit(&x_train_scaled, &y_train).unwrap())),
    ];
    
    println!("Model Comparison Results:");
    let mut best_accuracy = 0.0;
    let mut best_model = "";
    
    for (name, model) in models.iter() {
        let predictions = model.predict(&x_test_scaled).unwrap();
        let accuracy = accuracy_score(&y_test, &predictions);
        
        if accuracy > best_accuracy {
            best_accuracy = accuracy;
            best_model = name;
        }
        
        println!("  {}: {:.3}", name, accuracy);
        
        // Sanity checks
        assert_eq!(predictions.len(), y_test.len());
        assert!(accuracy >= 0.0 && accuracy <= 1.0);
    }
    
    println!("  Best model: {} ({:.3})", best_model, best_accuracy);
}

#[test]
fn test_cross_validation_integration() {
    // Test integration between models and cross-validation
    let (x, y) = make_regression(60, 4, 1, 0.1, 456).unwrap();
    
    // Create model
    let knn = KNNRegressor::new(5);
    
    // Simple train-test evaluation (mock cross-validation)
    let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.3, Some(456)).unwrap();
    
    // Preprocessing
    let mut scaler = StandardScaler::new();
    let x_train_scaled = scaler.fit_transform(&x_train).unwrap();
    let x_test_scaled = scaler.transform(&x_test).unwrap();
    
    // Train and evaluate
    let trained_model = knn.fit(&x_train_scaled, &y_train).unwrap();
    let predictions = trained_model.predict(&x_test_scaled).unwrap();
    
    let mse = mean_squared_error(&y_test, &predictions);
    let r2 = r2_score(&y_test, &predictions);
    
    // Verification
    assert!(mse >= 0.0);
    assert!(r2 <= 1.0);
    assert_eq!(predictions.len(), y_test.len());
    
    println!("Cross-validation Integration:");
    println!("  MSE: {:.3}", mse);
    println!("  R²: {:.3}", r2);
}

#[test] 
fn test_error_handling_across_crates() {
    // Test that errors are properly propagated across crate boundaries
    
    // Test dimension mismatch in KNN
    let x_train = array![[1.0, 2.0], [3.0, 4.0]];
    let y_train = vec![0, 1];
    let x_test_wrong = array![[1.0]]; // Wrong dimensions
    
    let knn = KNNClassifier::new(1).fit(&x_train, &y_train).unwrap();
    let result = knn.predict(&x_test_wrong);
    assert!(result.is_err());
    
    // Test empty data in scaler
    let empty_x: Array2<f64> = Array2::zeros((0, 2));
    let mut scaler = StandardScaler::new();
    let result = scaler.fit(&empty_x);
    assert!(result.is_err());
    
    // Test invalid parameters
    let x = array![[1.0, 2.0], [3.0, 4.0]];
    let y = vec![0, 1];
    
    // KNN with k > n_samples should handle gracefully
    let result = KNNClassifier::new(10).fit(&x, &y);
    // This should either work (by adjusting k) or fail gracefully
    if let Ok(model) = result {
        let pred_result = model.predict(&x);
        assert!(pred_result.is_ok());
    }
    
    println!("Error Handling: All error cases handled appropriately");
}

#[test]
fn test_reproducibility_across_crates() {
    // Test that results are reproducible when using the same random seeds
    let (x, y) = make_classification(50, 3, 2, 0, 789).unwrap();
    
    // Split with same seed twice
    let (x1, _, y1, _) = train_test_split(&x, &y, 0.5, Some(789)).unwrap();
    let (x2, _, y2, _) = train_test_split(&x, &y, 0.5, Some(789)).unwrap();
    
    // Should be identical
    assert_eq!(x1.shape(), x2.shape());
    assert_eq!(y1.len(), y2.len());
    
    for i in 0..x1.nrows() {
        for j in 0..x1.ncols() {
            assert_abs_diff_eq!(x1[[i, j]], x2[[i, j]], epsilon = 1e-10);
        }
    }
    
    for i in 0..y1.len() {
        assert_eq!(y1[i], y2[i]);
    }
    
    // Train models with same seed
    let tree1 = DecisionTreeClassifier::new().random_state(456).fit(&x1, &y1).unwrap();
    let tree2 = DecisionTreeClassifier::new().random_state(456).fit(&x2, &y2).unwrap();
    
    let pred1 = tree1.predict(&x1).unwrap();
    let pred2 = tree2.predict(&x2).unwrap();
    
    // Predictions should be identical
    assert_eq!(pred1.len(), pred2.len());
    for i in 0..pred1.len() {
        assert_eq!(pred1[i], pred2[i]);
    }
    
    println!("Reproducibility: All operations deterministic with same seeds");
}