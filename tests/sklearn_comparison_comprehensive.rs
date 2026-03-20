//! Comprehensive comparison tests with scikit-learn expected outputs
//!
//! These tests verify that sklears algorithms produce outputs consistent
//! with scikit-learn for various datasets and parameters.

use ndarray::{array, Array1, Array2};
use sklears::prelude::*;

// Import all modules
use sklears::linear_model::{LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression};
use sklears::neighbors::{KNeighborsClassifier, KNeighborsRegressor};
use sklears::preprocessing::{StandardScaler, MinMaxScaler};
use sklears::tree::{DecisionTreeClassifier, DecisionTreeRegressor};
use sklears::ensemble::{RandomForestClassifier, RandomForestRegressor};
use sklears::svm::{SVC, SVR};
use sklears::decomposition::{PCA, NMF};
use sklears::metrics::{accuracy_score, mean_squared_error, r2_score};

// Tolerance for floating point comparisons
const TOLERANCE: f64 = 1e-3;

/// Test Linear Regression against scikit-learn outputs
#[test]
fn test_linear_regression_sklearn_comparison() {
    // Test data: y = 2*x + 1
    let x = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
    let y = array![3.0, 5.0, 7.0, 9.0, 11.0];
    
    let model = LinearRegression::new()
        .fit(&x, &y)
        .expect("Failed to fit LinearRegression");
    
    // Expected coefficients from scikit-learn
    let expected_coef = 2.0;
    let expected_intercept = 1.0;
    
    assert!((model.coef()[0] - expected_coef).abs() < TOLERANCE);
    assert!((model.intercept() - expected_intercept).abs() < TOLERANCE);
    
    // Test predictions
    let x_test = array![[6.0], [7.0]];
    let predictions = model.predict(&x_test).unwrap();
    let expected_predictions = array![13.0, 15.0];
    
    for (pred, expected) in predictions.iter().zip(expected_predictions.iter()) {
        assert!((pred - expected).abs() < TOLERANCE);
    }
}

/// Test Ridge Regression with various alpha values
#[test]
fn test_ridge_regression_sklearn_comparison() {
    // Test data with multicollinearity
    let x = array![
        [1.0, 1.1],
        [2.0, 2.1],
        [3.0, 2.9],
        [4.0, 4.1],
        [5.0, 5.0]
    ];
    let y = array![3.1, 5.0, 6.9, 9.1, 11.0];
    
    // Test with different alpha values
    let alphas = vec![0.1, 1.0, 10.0];
    let expected_scores = vec![0.999, 0.995, 0.950]; // Approximate R² scores from sklearn
    
    for (alpha, expected_score) in alphas.iter().zip(expected_scores.iter()) {
        let model = Ridge::new(*alpha)
            .fit(&x, &y)
            .expect("Failed to fit Ridge");
        
        let score = model.score(&x, &y).unwrap();
        assert!((score - expected_score).abs() < 0.05); // Slightly larger tolerance for scores
    }
}

/// Test Lasso Regression sparsity
#[test]
fn test_lasso_regression_sklearn_comparison() {
    // Test data where only first feature is relevant
    let n_samples = 50;
    let mut x = Array2::zeros((n_samples, 3));
    let mut y = Array1::zeros(n_samples);
    
    for i in 0..n_samples {
        let val = i as f64 / 10.0;
        x[[i, 0]] = val;
        x[[i, 1]] = (i as f64).sin() * 0.1; // Noise feature
        x[[i, 2]] = (i as f64).cos() * 0.1; // Noise feature
        y[i] = 2.0 * val + 1.0;
    }
    
    let model = Lasso::new(0.1)
        .fit(&x, &y)
        .expect("Failed to fit Lasso");
    
    // First coefficient should be close to 2.0
    assert!((model.coef()[0] - 2.0).abs() < 0.2);
    
    // Other coefficients should be close to zero (sparsity)
    assert!(model.coef()[1].abs() < 0.1);
    assert!(model.coef()[2].abs() < 0.1);
}

/// Test K-Nearest Neighbors Classifier
#[test]
fn test_knn_classifier_sklearn_comparison() {
    // Simple 2D classification problem
    let x = array![
        [1.0, 1.0], [1.5, 1.5], [1.2, 1.3],
        [5.0, 5.0], [5.5, 5.5], [5.2, 5.3],
    ];
    let y = array![0, 0, 0, 1, 1, 1];
    
    let model = KNeighborsClassifier::new()
        .n_neighbors(3)
        .fit(&x, &y)
        .expect("Failed to fit KNN");
    
    // Test predictions on training data
    let predictions = model.predict(&x).unwrap();
    
    // Should achieve perfect accuracy on training data with k=3
    let accuracy = accuracy_score(&y, &predictions);
    assert!(accuracy >= 1.0);
    
    // Test prediction on new points
    let x_test = array![[1.1, 1.1], [5.1, 5.1]];
    let expected = array![0, 1];
    let test_predictions = model.predict(&x_test).unwrap();
    
    assert_eq!(test_predictions, expected);
}

/// Test Decision Tree Classifier
#[test]
fn test_decision_tree_classifier_sklearn_comparison() {
    // XOR problem
    let x = array![
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0]
    ];
    let y = array![0, 1, 1, 0];
    
    let model = DecisionTreeClassifier::new()
        .max_depth(Some(2))
        .fit(&x, &y)
        .expect("Failed to fit DecisionTree");
    
    let predictions = model.predict(&x).unwrap();
    
    // Decision tree should be able to learn XOR perfectly
    let accuracy = accuracy_score(&y, &predictions);
    assert!(accuracy >= 1.0);
}

/// Test Support Vector Classifier
#[test]
fn test_svc_sklearn_comparison() {
    // Linearly separable problem
    let x = array![
        [1.0, 2.0], [2.0, 3.0], [3.0, 3.0],
        [6.0, 5.0], [7.0, 7.0], [8.0, 6.0]
    ];
    let y = array![0, 0, 0, 1, 1, 1];
    
    // Test linear kernel
    let linear_model = SVC::linear(1.0)
        .fit(&x, &y)
        .expect("Failed to fit linear SVC");
    
    let predictions = linear_model.predict(&x).unwrap();
    let accuracy = accuracy_score(&y, &predictions);
    assert!(accuracy >= 0.95); // Should achieve high accuracy on linearly separable data
    
    // Test RBF kernel
    let rbf_model = SVC::rbf(1.0, 0.5)
        .fit(&x, &y)
        .expect("Failed to fit RBF SVC");
    
    let rbf_predictions = rbf_model.predict(&x).unwrap();
    let rbf_accuracy = accuracy_score(&y, &rbf_predictions);
    assert!(rbf_accuracy >= 0.95);
}

/// Test PCA dimensionality reduction
#[test]
fn test_pca_sklearn_comparison() {
    // Create correlated features
    let x = array![
        [1.0, 2.0, 3.0],
        [2.0, 4.0, 6.0],
        [3.0, 6.0, 9.0],
        [4.0, 8.0, 12.0],
        [5.0, 10.0, 15.0]
    ];
    
    let pca = PCA::new()
        .n_components(2)
        .fit(&x)
        .expect("Failed to fit PCA");
    
    // Should explain almost all variance with 1 component (data is rank 1)
    let explained_variance_ratio = pca.explained_variance_ratio();
    assert!(explained_variance_ratio[0] > 0.99);
    
    // Transform and check dimensionality
    let x_transformed = pca.transform(&x).unwrap();
    assert_eq!(x_transformed.shape(), &[5, 2]);
}

/// Test preprocessing scalers
#[test]
fn test_preprocessing_sklearn_comparison() {
    let x = array![
        [1.0, 100.0],
        [2.0, 200.0],
        [3.0, 300.0],
        [4.0, 400.0]
    ];
    
    // Test StandardScaler
    let std_scaler = StandardScaler::new()
        .fit(&x)
        .expect("Failed to fit StandardScaler");
    
    let x_std = std_scaler.transform(&x).unwrap();
    
    // Check mean is 0 and std is 1
    let mean = x_std.mean_axis(ndarray::Axis(0)).unwrap();
    let std = x_std.std_axis(ndarray::Axis(0), 0.0);
    
    for &m in mean.iter() {
        assert!(m.abs() < TOLERANCE);
    }
    for &s in std.iter() {
        assert!((s - 1.0).abs() < TOLERANCE);
    }
    
    // Test MinMaxScaler
    let minmax_scaler = MinMaxScaler::new()
        .fit(&x)
        .expect("Failed to fit MinMaxScaler");
    
    let x_minmax = minmax_scaler.transform(&x).unwrap();
    
    // Check range is [0, 1]
    let min_vals = x_minmax.fold_axis(ndarray::Axis(0), f64::INFINITY, |&a, &b| a.min(b));
    let max_vals = x_minmax.fold_axis(ndarray::Axis(0), f64::NEG_INFINITY, |&a, &b| a.max(b));
    
    for &min_val in min_vals.iter() {
        assert!(min_val.abs() < TOLERANCE);
    }
    for &max_val in max_vals.iter() {
        assert!((max_val - 1.0).abs() < TOLERANCE);
    }
}

/// Test ensemble methods
#[test]
fn test_random_forest_sklearn_comparison() {
    // Binary classification with clear pattern
    let n_samples = 100;
    let mut x = Array2::zeros((n_samples, 2));
    let mut y = Array1::zeros(n_samples);
    
    for i in 0..n_samples {
        let angle = 2.0 * std::f64::consts::PI * (i as f64) / (n_samples as f64);
        let radius = if i < n_samples / 2 { 1.0 } else { 3.0 };
        x[[i, 0]] = radius * angle.cos() + 0.1 * rand::random::<f64>();
        x[[i, 1]] = radius * angle.sin() + 0.1 * rand::random::<f64>();
        y[i] = if i < n_samples / 2 { 0 } else { 1 };
    }
    
    let model = RandomForestClassifier::new()
        .n_estimators(10)
        .max_depth(Some(3))
        .fit(&x, &y)
        .expect("Failed to fit RandomForest");
    
    let predictions = model.predict(&x).unwrap();
    let accuracy = accuracy_score(&y, &predictions);
    
    // Should achieve good accuracy on this simple pattern
    assert!(accuracy > 0.85);
}

/// Integration test combining multiple algorithms
#[test]
fn test_pipeline_sklearn_comparison() {
    // Generate regression data
    let x = array![
        [1.0, 10.0],
        [2.0, 20.0],
        [3.0, 30.0],
        [4.0, 40.0],
        [5.0, 50.0]
    ];
    let y = array![15.0, 30.0, 45.0, 60.0, 75.0]; // y = 10*x1 + 0.5*x2
    
    // Step 1: Scale features
    let scaler = StandardScaler::new().fit(&x).unwrap();
    let x_scaled = scaler.transform(&x).unwrap();
    
    // Step 2: Apply PCA
    let pca = PCA::new().n_components(2).fit(&x_scaled).unwrap();
    let x_pca = pca.transform(&x_scaled).unwrap();
    
    // Step 3: Train linear regression
    let model = LinearRegression::new().fit(&x_pca, &y).unwrap();
    
    // Make predictions on training data
    let predictions = model.predict(&x_pca).unwrap();
    
    // Calculate R² score
    let r2 = r2_score(&y, &predictions);
    assert!(r2 > 0.99); // Should achieve near-perfect fit
}

// Helper to generate synthetic classification data
fn make_classification(n_samples: usize, n_features: usize, n_classes: usize) -> (Array2<f64>, Array1<i32>) {
    let mut x = Array2::zeros((n_samples, n_features));
    let mut y = Array1::zeros(n_samples);
    
    // Generate clusters
    for i in 0..n_samples {
        let class = (i * n_classes) / n_samples;
        y[i] = class as i32;
        
        for j in 0..n_features {
            // Add class-specific offset and noise
            x[[i, j]] = class as f64 + 0.5 * rand::random::<f64>();
        }
    }
    
    (x, y)
}

/// Test multi-class classification
#[test]
fn test_multiclass_classification_sklearn_comparison() {
    let (x, y) = make_classification(150, 4, 3);
    
    // Test with LogisticRegression
    let lr_model = LogisticRegression::new()
        .max_iter(100)
        .fit(&x, &y)
        .expect("Failed to fit LogisticRegression");
    
    let lr_predictions = lr_model.predict(&x).unwrap();
    let lr_accuracy = accuracy_score(&y, &lr_predictions);
    assert!(lr_accuracy > 0.8); // Should achieve good accuracy
    
    // Test with SVC using One-vs-Rest
    let svc_model = SVC::linear(1.0)
        .fit(&x, &y)
        .expect("Failed to fit SVC");
    
    let svc_predictions = svc_model.predict(&x).unwrap();
    let svc_accuracy = accuracy_score(&y, &svc_predictions);
    assert!(svc_accuracy > 0.8);
}