//! Comprehensive integration tests for sklears
//! 
//! This module tests end-to-end workflows combining multiple crates
//! to ensure proper integration and functionality across the library.

use ndarray::{Array1, Array2, array};
use sklears_core::traits::{Transform};
use sklears_metrics::{accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score};
use sklears_utils::data_generation::{make_classification, make_regression, make_blobs};
use sklears_utils::random::{train_test_split_indices, set_random_state};
use sklears_preprocessing::{
    StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder, TargetEncoder,
    SimpleImputer, KNNImputer, PolynomialFeatures
};
use sklears_model_selection::{KFold, StratifiedKFold, cross_val_score};
use sklears_tree::{DecisionTreeClassifier, DecisionTreeRegressor, RandomForestClassifier};
use sklears_ensemble::VotingClassifier;
use approx::assert_abs_diff_eq;

#[test]
fn test_complete_classification_pipeline() {
    set_random_state(42);
    
    // Generate synthetic classification data
    let (mut x, y) = make_classification(100, 4, 3, 2, 0.1, 42).unwrap();
    
    // Add some missing values to test imputation
    x[[5, 0]] = f64::NAN;
    x[[15, 1]] = f64::NAN;
    x[[25, 2]] = f64::NAN;
    
    // Split data
    let indices = train_test_split_indices(x.nrows(), 0.2, true, Some(42)).unwrap();
    let train_indices = &indices.0;
    let test_indices = &indices.1;
    
    let x_train = x.select(ndarray::Axis(0), train_indices);
    let y_train = y.select(ndarray::Axis(0), train_indices);
    let x_test = x.select(ndarray::Axis(0), test_indices);
    let y_test = y.select(ndarray::Axis(0), test_indices);
    
    // Step 1: Impute missing values
    let imputer = SimpleImputer::new()
        .strategy(sklears_preprocessing::ImputationStrategy::Mean)
        .fit(&x_train, &()).unwrap();
    let x_train_imputed = imputer.transform(&x_train).unwrap();
    let x_test_imputed = imputer.transform(&x_test).unwrap();
    
    // Step 2: Scale features
    let scaler = StandardScaler::new()
        .fit(&x_train_imputed, &()).unwrap();
    let x_train_scaled = scaler.transform(&x_train_imputed).unwrap();
    let x_test_scaled = scaler.transform(&x_test_imputed).unwrap();
    
    // Step 3: Add polynomial features
    let poly = PolynomialFeatures::new()
        .degree(2)
        .interaction_only(true)
        .fit(&x_train_scaled, &()).unwrap();
    let x_train_poly = poly.transform(&x_train_scaled).unwrap();
    let x_test_poly = poly.transform(&x_test_scaled).unwrap();
    
    // Step 4: Train multiple models
    let dt = DecisionTreeClassifier::new()
        .max_depth(Some(5))
        .fit(&x_train_poly, &y_train).unwrap();
    
    let rf = RandomForestClassifier::new()
        .n_estimators(10)
        .max_depth(Some(3))
        .fit(&x_train_poly, &y_train).unwrap();
    
    // Step 5: Ensemble with voting
    let voting_clf = VotingClassifier::new()
        .add_estimator("dt".to_string(), Box::new(dt))
        .add_estimator("rf".to_string(), Box::new(rf))
        .voting_method(sklears_ensemble::VotingMethod::Hard)
        .fit(&x_train_poly, &y_train).unwrap();
    
    // Step 6: Make predictions and evaluate
    let y_pred = voting_clf.predict(&x_test_poly).unwrap();
    
    let accuracy = accuracy_score(&y_test, &y_pred).unwrap();
    let precision = precision_score(&y_test, &y_pred, "macro").unwrap();
    let recall = recall_score(&y_test, &y_pred, "macro").unwrap();
    let f1 = f1_score(&y_test, &y_pred, "macro").unwrap();
    
    // Verify metrics are reasonable
    assert!(accuracy >= 0.0 && accuracy <= 1.0);
    assert!(precision >= 0.0 && precision <= 1.0);
    assert!(recall >= 0.0 && recall <= 1.0);
    assert!(f1 >= 0.0 && f1 <= 1.0);
    
    println!("Classification Pipeline Results:");
    println!("Accuracy: {:.3}", accuracy);
    println!("Precision: {:.3}", precision);
    println!("Recall: {:.3}", recall);
    println!("F1-Score: {:.3}", f1);
}

#[test]
fn test_complete_regression_pipeline() {
    set_random_state(42);
    
    // Generate synthetic regression data
    let (mut x, y) = make_regression(100, 3, 1, 0.1, 42).unwrap();
    
    // Add some missing values
    x[[10, 0]] = f64::NAN;
    x[[20, 1]] = f64::NAN;
    
    // Split data
    let indices = train_test_split_indices(x.nrows(), 0.2, true, Some(42)).unwrap();
    let train_indices = &indices.0;
    let test_indices = &indices.1;
    
    let x_train = x.select(ndarray::Axis(0), train_indices);
    let y_train = y.select(ndarray::Axis(0), train_indices);
    let x_test = x.select(ndarray::Axis(0), test_indices);
    let y_test = y.select(ndarray::Axis(0), test_indices);
    
    // Step 1: Impute missing values using KNN
    let imputer = KNNImputer::new()
        .n_neighbors(3)
        .fit(&x_train, &()).unwrap();
    let x_train_imputed = imputer.transform(&x_train).unwrap();
    let x_test_imputed = imputer.transform(&x_test).unwrap();
    
    // Step 2: Scale features with MinMax
    let scaler = MinMaxScaler::new()
        .fit(&x_train_imputed, &()).unwrap();
    let x_train_scaled = scaler.transform(&x_train_imputed).unwrap();
    let x_test_scaled = scaler.transform(&x_test_imputed).unwrap();
    
    // Step 3: Train decision tree regressor
    let dt_regressor = DecisionTreeRegressor::new()
        .criterion(sklears_tree::SplitCriterion::MSE)
        .max_depth(Some(4))
        .fit(&x_train_scaled, &y_train).unwrap();
    
    // Step 4: Make predictions and evaluate
    let y_pred = dt_regressor.predict(&x_test_scaled).unwrap();
    
    let mse = mean_squared_error(&y_test, &y_pred).unwrap();
    let r2 = r2_score(&y_test, &y_pred).unwrap();
    
    // Verify metrics are reasonable
    assert!(mse >= 0.0);
    assert!(r2 <= 1.0);
    
    println!("Regression Pipeline Results:");
    println!("MSE: {:.3}", mse);
    println!("R² Score: {:.3}", r2);
}

#[test]
fn test_cross_validation_integration() {
    set_random_state(42);
    
    // Generate data
    let (x, y) = make_blobs(60, 2, 3, 1.0, (-10.0, 10.0), 42).unwrap();
    
    // Test with decision tree
    let dt = DecisionTreeClassifier::new()
        .max_depth(Some(3));
    
    // Perform cross-validation
    let cv = KFold::new(5).shuffle(true).random_state(42);
    let scores = cross_val_score(dt, &x, &y, cv, "accuracy").unwrap();
    
    // Check that we got 5 scores
    assert_eq!(scores.len(), 5);
    
    // All scores should be between 0 and 1
    for &score in scores.iter() {
        assert!(score >= 0.0 && score <= 1.0);
    }
    
    let mean_score = scores.iter().sum::<f64>() / scores.len() as f64;
    println!("Cross-validation mean accuracy: {:.3}", mean_score);
}

#[test]
fn test_stratified_cross_validation() {
    set_random_state(42);
    
    // Create imbalanced dataset
    let x_class_0 = Array2::random((80, 2), rand_distr::Uniform::new(-1.0, 1.0));
    let y_class_0 = Array1::zeros(80);
    
    let x_class_1 = Array2::random((20, 2), rand_distr::Uniform::new(1.0, 3.0));
    let y_class_1 = Array1::ones(20);
    
    let x = ndarray::concatenate![ndarray::Axis(0), x_class_0, x_class_1];
    let y = ndarray::concatenate![ndarray::Axis(0), y_class_0, y_class_1];
    
    let dt = DecisionTreeClassifier::new();
    
    // Test stratified K-fold
    let cv = StratifiedKFold::new(3).shuffle(true).random_state(42);
    let scores = cross_val_score(dt, &x, &y, cv, "accuracy").unwrap();
    
    assert_eq!(scores.len(), 3);
    
    for &score in scores.iter() {
        assert!(score >= 0.0 && score <= 1.0);
    }
    
    let mean_score = scores.iter().sum::<f64>() / scores.len() as f64;
    println!("Stratified CV mean accuracy: {:.3}", mean_score);
}

#[test]
fn test_encoding_pipeline() {
    // Create mixed data with categorical and numerical features
    let categorical_data = array![
        ["red".to_string(), "small".to_string()],
        ["blue".to_string(), "large".to_string()],
        ["green".to_string(), "medium".to_string()],
        ["red".to_string(), "large".to_string()],
        ["blue".to_string(), "small".to_string()],
    ];
    
    let numerical_data = array![
        [1.0, 2.0],
        [2.0, 3.0],
        [3.0, 1.0],
        [1.5, 2.5],
        [2.5, 1.5],
    ];
    
    let target = array![10.0, 20.0, 15.0, 12.0, 18.0];
    
    // Test Label Encoder
    let label_encoder = LabelEncoder::new()
        .fit(&vec!["A".to_string(), "B".to_string(), "A".to_string()], &()).unwrap();
    let encoded_labels = label_encoder.transform(&vec!["A".to_string(), "B".to_string()]).unwrap();
    assert_eq!(encoded_labels, array![0, 1]);
    
    // Test One-Hot Encoder
    let one_hot = OneHotEncoder::new()
        .fit(&categorical_data, &()).unwrap();
    let encoded_categorical = one_hot.transform(&categorical_data).unwrap();
    assert_eq!(encoded_categorical.ncols(), 6); // 3 colors + 3 sizes
    
    // Test Target Encoder
    let target_encoder = TargetEncoder::new()
        .with_smoothing(1.0)
        .fit(&categorical_data, &target).unwrap();
    let target_encoded = target_encoder.transform(&categorical_data).unwrap();
    assert_eq!(target_encoded.shape(), categorical_data.shape());
    
    // Verify target encoded values are different from original
    assert_ne!(target_encoded[[0, 0]], 0.0); // Should not be zero
    
    println!("Encoding pipeline completed successfully");
}

#[test]
fn test_feature_engineering_pipeline() {
    set_random_state(42);
    
    // Generate base data
    let x = array![
        [1.0, 2.0],
        [2.0, 3.0],
        [3.0, 4.0],
        [4.0, 5.0],
    ];
    
    // Test polynomial features
    let poly = PolynomialFeatures::new()
        .degree(2)
        .include_bias(true)
        .fit(&x, &()).unwrap();
    let x_poly = poly.transform(&x).unwrap();
    
    // Should have: bias(1) + original(2) + interactions(1) + squares(2) = 6 features
    assert_eq!(x_poly.ncols(), 6);
    
    // Test scaling after polynomial features
    let scaler = StandardScaler::new()
        .fit(&x_poly, &()).unwrap();
    let x_scaled = scaler.transform(&x_poly).unwrap();
    
    // Verify scaling worked
    let means = x_scaled.mean_axis(ndarray::Axis(0)).unwrap();
    for &mean in means.iter() {
        assert_abs_diff_eq!(mean, 0.0, epsilon = 1e-10);
    }
    
    println!("Feature engineering pipeline completed successfully");
}

#[test]
fn test_imputation_strategies() {
    // Create data with systematic missing values
    let mut x = array![
        [1.0, 10.0, 100.0],
        [2.0, 20.0, 200.0],
        [3.0, 30.0, 300.0],
        [4.0, 40.0, 400.0],
        [5.0, 50.0, 500.0],
    ];
    
    // Add missing values
    x[[1, 0]] = f64::NAN; // Missing in row 1, col 0
    x[[3, 1]] = f64::NAN; // Missing in row 3, col 1
    x[[4, 2]] = f64::NAN; // Missing in row 4, col 2
    
    // Test Mean Imputation
    let mean_imputer = SimpleImputer::new()
        .strategy(sklears_preprocessing::ImputationStrategy::Mean)
        .fit(&x, &()).unwrap();
    let x_mean_imputed = mean_imputer.transform(&x).unwrap();
    
    // Verify no NaN values remain
    for value in x_mean_imputed.iter() {
        assert!(!value.is_nan());
    }
    
    // Test KNN Imputation
    let knn_imputer = KNNImputer::new()
        .n_neighbors(2)
        .fit(&x, &()).unwrap();
    let x_knn_imputed = knn_imputer.transform(&x).unwrap();
    
    // Verify no NaN values remain
    for value in x_knn_imputed.iter() {
        assert!(!value.is_nan());
    }
    
    println!("Imputation strategies tested successfully");
}

#[test]
fn test_multiclass_metrics() {
    // Create a simple multiclass prediction scenario
    let y_true = array![0, 1, 2, 0, 1, 2, 0, 1, 2];
    let y_pred = array![0, 1, 1, 0, 2, 2, 0, 1, 2]; // Some misclassifications
    
    // Test all averaging methods
    for average in ["micro", "macro", "weighted"] {
        let precision = precision_score(&y_true, &y_pred, average).unwrap();
        let recall = recall_score(&y_true, &y_pred, average).unwrap();
        let f1 = f1_score(&y_true, &y_pred, average).unwrap();
        
        assert!(precision >= 0.0 && precision <= 1.0);
        assert!(recall >= 0.0 && recall <= 1.0);
        assert!(f1 >= 0.0 && f1 <= 1.0);
        
        println!("{} - Precision: {:.3}, Recall: {:.3}, F1: {:.3}", 
                average, precision, recall, f1);
    }
}

#[test]
fn test_end_to_end_model_comparison() {
    set_random_state(42);
    
    // Generate comparison dataset
    let (x, y) = make_classification(150, 4, 3, 2, 0.1, 42).unwrap();
    
    // Split data
    let indices = train_test_split_indices(x.nrows(), 0.3, true, Some(42)).unwrap();
    let train_indices = &indices.0;
    let test_indices = &indices.1;
    
    let x_train = x.select(ndarray::Axis(0), train_indices);
    let y_train = y.select(ndarray::Axis(0), train_indices);
    let x_test = x.select(ndarray::Axis(0), test_indices);
    let y_test = y.select(ndarray::Axis(0), test_indices);
    
    // Scale data
    let scaler = StandardScaler::new().fit(&x_train, &()).unwrap();
    let x_train_scaled = scaler.transform(&x_train).unwrap();
    let x_test_scaled = scaler.transform(&x_test).unwrap();
    
    // Train multiple models
    let mut models = Vec::new();
    let mut model_names = Vec::new();
    
    // Decision Tree
    let dt = DecisionTreeClassifier::new()
        .max_depth(Some(5))
        .fit(&x_train_scaled, &y_train).unwrap();
    models.push(dt);
    model_names.push("Decision Tree");
    
    // Random Forest
    let rf = RandomForestClassifier::new()
        .n_estimators(10)
        .max_depth(Some(4))
        .fit(&x_train_scaled, &y_train).unwrap();
    models.push(rf);
    model_names.push("Random Forest");
    
    // Evaluate all models
    println!("Model Comparison Results:");
    for (i, model) in models.iter().enumerate() {
        let y_pred = model.predict(&x_test_scaled).unwrap();
        let accuracy = accuracy_score(&y_test, &y_pred).unwrap();
        let f1 = f1_score(&y_test, &y_pred, "macro").unwrap();
        
        println!("{}: Accuracy = {:.3}, F1 = {:.3}", model_names[i], accuracy, f1);
        
        // Basic sanity checks
        assert!(accuracy >= 0.0 && accuracy <= 1.0);
        assert!(f1 >= 0.0 && f1 <= 1.0);
    }
}

#[test]
fn test_pipeline_with_categorical_features() {
    // Create dataset with mixed categorical and numerical features
    let numerical_features = array![
        [1.0, 2.0],
        [2.0, 3.0],
        [3.0, 4.0],
        [4.0, 5.0],
        [5.0, 6.0],
        [6.0, 7.0],
    ];
    
    let categorical_features = array![
        ["A".to_string(), "X".to_string()],
        ["B".to_string(), "Y".to_string()],
        ["A".to_string(), "X".to_string()],
        ["C".to_string(), "Z".to_string()],
        ["B".to_string(), "Y".to_string()],
        ["A".to_string(), "Z".to_string()],
    ];
    
    let y = array![0, 1, 0, 1, 1, 0];
    
    // Encode categorical features
    let encoder = OneHotEncoder::new()
        .fit(&categorical_features, &()).unwrap();
    let categorical_encoded = encoder.transform(&categorical_features).unwrap();
    
    // Combine numerical and encoded categorical features
    let x_combined = ndarray::concatenate![
        ndarray::Axis(1), 
        numerical_features, 
        categorical_encoded
    ];
    
    // Scale the combined features
    let scaler = StandardScaler::new()
        .fit(&x_combined, &()).unwrap();
    let x_scaled = scaler.transform(&x_combined).unwrap();
    
    // Train a model
    let model = DecisionTreeClassifier::new()
        .fit(&x_scaled, &y).unwrap();
    
    // Test prediction
    let predictions = model.predict(&x_scaled).unwrap();
    assert_eq!(predictions.len(), y.len());
    
    // Calculate accuracy
    let accuracy = accuracy_score(&y, &predictions).unwrap();
    assert!(accuracy >= 0.0 && accuracy <= 1.0);
    
    println!("Mixed features pipeline accuracy: {:.3}", accuracy);
}