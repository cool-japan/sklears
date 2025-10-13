//! Simple integration tests for sklears
//! 
//! This module tests basic end-to-end workflows to ensure
//! different crates work together properly.

use ndarray::{Array1, Array2, array};
use sklears_core::traits::{Transform};
use sklears_metrics::{accuracy_score, mean_squared_error};
use sklears_utils::data_generation::{make_classification, make_regression};
use sklears_utils::random::{train_test_split_indices, set_random_state};
use sklears_preprocessing::{StandardScaler, SimpleImputer, PolynomialFeatures};
use sklears_model_selection::{KFold, cross_val_score};
use sklears_tree::{DecisionTreeClassifier, DecisionTreeRegressor};

#[test]
fn test_basic_classification_pipeline() {
    set_random_state(42);
    
    // Generate synthetic classification data
    let (x, y) = make_classification(100, 4, 3, 2, 0.1, 42).unwrap();
    
    // Split data
    let indices = train_test_split_indices(x.nrows(), 0.2, true, Some(42)).unwrap();
    let train_indices = &indices.0;
    let test_indices = &indices.1;
    
    let x_train = x.select(ndarray::Axis(0), train_indices);
    let y_train = y.select(ndarray::Axis(0), train_indices);
    let x_test = x.select(ndarray::Axis(0), test_indices);
    let y_test = y.select(ndarray::Axis(0), test_indices);
    
    // Scale features
    let scaler = StandardScaler::new()
        .fit(&x_train, &()).unwrap();
    let x_train_scaled = scaler.transform(&x_train).unwrap();
    let x_test_scaled = scaler.transform(&x_test).unwrap();
    
    // Train model
    let dt = DecisionTreeClassifier::new()
        .max_depth(Some(5))
        .fit(&x_train_scaled, &y_train).unwrap();
    
    // Make predictions and evaluate
    let y_pred = dt.predict(&x_test_scaled).unwrap();
    let accuracy = accuracy_score(&y_test, &y_pred).unwrap();
    
    // Verify metrics are reasonable
    assert!(accuracy >= 0.0 && accuracy <= 1.0);
    
    println!("Basic Classification Pipeline Accuracy: {:.3}", accuracy);
}

#[test]
fn test_basic_regression_pipeline() {
    set_random_state(42);
    
    // Generate synthetic regression data
    let (x, y) = make_regression(100, 3, 1, 0.1, 42).unwrap();
    
    // Split data
    let indices = train_test_split_indices(x.nrows(), 0.2, true, Some(42)).unwrap();
    let train_indices = &indices.0;
    let test_indices = &indices.1;
    
    let x_train = x.select(ndarray::Axis(0), train_indices);
    let y_train = y.select(ndarray::Axis(0), train_indices);
    let x_test = x.select(ndarray::Axis(0), test_indices);
    let y_test = y.select(ndarray::Axis(0), test_indices);
    
    // Scale features
    let scaler = StandardScaler::new().fit(&x_train, &()).unwrap();
    let x_train_scaled = scaler.transform(&x_train).unwrap();
    let x_test_scaled = scaler.transform(&x_test).unwrap();
    
    // Train decision tree regressor
    let dt_regressor = DecisionTreeRegressor::new()
        .criterion(sklears_tree::SplitCriterion::MSE)
        .max_depth(Some(4))
        .fit(&x_train_scaled, &y_train).unwrap();
    
    // Make predictions and evaluate
    let y_pred = dt_regressor.predict(&x_test_scaled).unwrap();
    let mse = mean_squared_error(&y_test, &y_pred).unwrap();
    
    // Verify metrics are reasonable
    assert!(mse >= 0.0);
    
    println!("Basic Regression Pipeline MSE: {:.3}", mse);
}

#[test]
fn test_preprocessing_pipeline() {
    set_random_state(42);
    
    // Generate data with missing values
    let (mut x, y) = make_classification(50, 3, 2, 1, 0.0, 42).unwrap();
    
    // Add some missing values
    x[[5, 0]] = f64::NAN;
    x[[15, 1]] = f64::NAN;
    x[[25, 2]] = f64::NAN;
    
    // Step 1: Impute missing values
    let imputer = SimpleImputer::new()
        .strategy(sklears_preprocessing::ImputationStrategy::Mean)
        .fit(&x, &()).unwrap();
    let x_imputed = imputer.transform(&x).unwrap();
    
    // Verify no NaN values remain
    for value in x_imputed.iter() {
        assert!(!value.is_nan());
    }
    
    // Step 2: Scale features
    let scaler = StandardScaler::new()
        .fit(&x_imputed, &()).unwrap();
    let x_scaled = scaler.transform(&x_imputed).unwrap();
    
    // Step 3: Add polynomial features
    let poly = PolynomialFeatures::new()
        .degree(2)
        .interaction_only(true)
        .fit(&x_scaled, &()).unwrap();
    let x_poly = poly.transform(&x_scaled).unwrap();
    
    // Check that we have more features after polynomial expansion
    assert!(x_poly.ncols() > x_scaled.ncols());
    
    // Step 4: Train a model on the processed data
    let dt = DecisionTreeClassifier::new()
        .fit(&x_poly, &y).unwrap();
    
    let predictions = dt.predict(&x_poly).unwrap();
    assert_eq!(predictions.len(), y.len());
    
    println!("Preprocessing pipeline completed successfully");
}

#[test]
fn test_cross_validation_integration() {
    set_random_state(42);
    
    // Generate data
    let (x, y) = make_classification(80, 4, 3, 2, 0.1, 42).unwrap();
    
    // Scale the data
    let scaler = StandardScaler::new().fit(&x, &()).unwrap();
    let x_scaled = scaler.transform(&x).unwrap();
    
    // Test with decision tree
    let dt = DecisionTreeClassifier::new()
        .max_depth(Some(3));
    
    // Perform cross-validation
    let cv = KFold::new(5).shuffle(true).random_state(42);
    let scores = cross_val_score(dt, &x_scaled, &y, cv, "accuracy").unwrap();
    
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
fn test_encoding_workflow() {
    // Create simple categorical data
    let categorical_data = array![
        ["red".to_string(), "small".to_string()],
        ["blue".to_string(), "large".to_string()],
        ["green".to_string(), "medium".to_string()],
        ["red".to_string(), "large".to_string()],
        ["blue".to_string(), "small".to_string()],
    ];
    
    let target = array![10.0, 20.0, 15.0, 12.0, 18.0];
    
    // Test One-Hot Encoder
    let one_hot = sklears_preprocessing::OneHotEncoder::new()
        .fit(&categorical_data, &()).unwrap();
    let encoded_categorical = one_hot.transform(&categorical_data).unwrap();
    assert_eq!(encoded_categorical.ncols(), 6); // 3 colors + 3 sizes
    
    // Test Target Encoder
    let target_encoder = sklears_preprocessing::TargetEncoder::new()
        .with_smoothing(1.0)
        .fit(&categorical_data, &target).unwrap();
    let target_encoded = target_encoder.transform(&categorical_data).unwrap();
    assert_eq!(target_encoded.shape(), categorical_data.shape());
    
    println!("Encoding workflow completed successfully");
}

#[test]
fn test_simple_model_comparison() {
    set_random_state(42);
    
    // Generate comparison dataset
    let (x, y) = make_classification(100, 4, 3, 2, 0.1, 42).unwrap();
    
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
    let rf = sklears_tree::RandomForestClassifier::new()
        .n_estimators(5)
        .max_depth(Some(4))
        .fit(&x_train_scaled, &y_train).unwrap();
    models.push(rf);
    model_names.push("Random Forest");
    
    // Evaluate all models
    println!("Simple Model Comparison Results:");
    for (i, model) in models.iter().enumerate() {
        let y_pred = model.predict(&x_test_scaled).unwrap();
        let accuracy = accuracy_score(&y_test, &y_pred).unwrap();
        
        println!("{}: Accuracy = {:.3}", model_names[i], accuracy);
        
        // Basic sanity checks
        assert!(accuracy >= 0.0 && accuracy <= 1.0);
    }
}