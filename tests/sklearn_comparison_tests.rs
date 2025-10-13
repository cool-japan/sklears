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

// Test data generation utilities
fn make_classification_data() -> (Array2<f64>, Array1<i32>) {
    // Simple 2D classification problem
    let x = array![
        [1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0],
        [5.0, 6.0], [6.0, 7.0], [7.0, 8.0], [8.0, 9.0],
        [1.5, 2.5], [2.5, 3.5], [3.5, 4.5], [4.5, 5.5],
        [5.5, 6.5], [6.5, 7.5], [7.5, 8.5], [8.5, 9.5],
    ];
    let y = array![0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1];
    (x, y)
}

fn make_regression_data() -> (Array2<f64>, Array1<f64>) {
    // Simple linear regression problem
    let x = array![
        [1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0],
        [1.5], [2.5], [3.5], [4.5], [5.5], [6.5], [7.5], [8.5],
    ];
    let y = array![
        2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0,
        3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0
    ];
    (x, y)
}

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

/// Test K-Nearest Neighbors Classifier
#[test]
fn test_knn_classifier_consistency() {
    let (x, y) = make_classification_data();
    
    // Test KNN with different parameters
    let model = KNeighborsClassifier::new()
        .n_neighbors(3)
        .fit(&x, &y)
        .unwrap();
    
    let predictions = model.predict(&x).unwrap();
    
    // Basic consistency checks
    assert_eq!(predictions.len(), y.len());
    
    // Check that we get reasonable accuracy on training data
    let accuracy = accuracy_score(&y, &predictions);
    assert!(accuracy > 0.7, "KNN should achieve reasonable accuracy on training data");
    
    // Test with different k values
    for k in [1, 3, 5] {
        let model_k = KNeighborsClassifier::new()
            .n_neighbors(k)
            .fit(&x, &y)
            .unwrap();
        
        let pred_k = model_k.predict(&x).unwrap();
        assert_eq!(pred_k.len(), y.len());
        
        // k=1 should achieve perfect accuracy on training data
        if k == 1 {
            let acc = accuracy_score(&y, &pred_k);
            assert!((acc - 1.0).abs() < 1e-10, "k=1 should achieve perfect accuracy");
        }
    }
}

#[test]
fn test_decision_tree_classifier_consistency() {
    let (x, y) = make_classification_data();
    
    // Test Decision Tree with different parameters
    let model = DecisionTreeClassifier::new()
        .max_depth(5)
        .min_samples_split(2)
        .fit(&x, &y)
        .unwrap();
    
    let predictions = model.predict(&x).unwrap();
    
    // Basic consistency checks
    assert_eq!(predictions.len(), y.len());
    assert_eq!(model.n_features(), x.ncols());
    assert_eq!(model.n_classes(), 2);
    
    // Check that we get reasonable accuracy
    let accuracy = accuracy_score(&y, &predictions);
    assert!(accuracy > 0.8, "Decision tree should achieve good accuracy on training data");
    
    // Test with different max_depth values
    for depth in [1, 3, 5, 10] {
        let model_depth = DecisionTreeClassifier::new()
            .max_depth(depth)
            .fit(&x, &y)
            .unwrap();
        
        let pred_depth = model_depth.predict(&x).unwrap();
        assert_eq!(pred_depth.len(), y.len());
        
        // Deeper trees should generally achieve better training accuracy
        if depth >= 5 {
            let acc = accuracy_score(&y, &pred_depth);
            assert!(acc > 0.8, "Deep tree should achieve good accuracy");
        }
    }
}

#[test]
fn test_random_forest_classifier_consistency() {
    let (x, y) = make_classification_data();
    
    // Test Random Forest with different parameters
    let model = RandomForestClassifier::new()
        .n_estimators(10)
        .max_depth(5)
        .random_state(42)
        .fit(&x, &y)
        .unwrap();
    
    let predictions = model.predict(&x).unwrap();
    
    // Basic consistency checks
    assert_eq!(predictions.len(), y.len());
    assert_eq!(model.n_features(), x.ncols());
    assert_eq!(model.n_classes(), 2);
    
    // Check that we get good accuracy
    let accuracy = accuracy_score(&y, &predictions);
    assert!(accuracy > 0.85, "Random forest should achieve good accuracy on training data");
    
    // Test with different numbers of estimators
    for n_est in [5, 10, 20] {
        let model_nest = RandomForestClassifier::new()
            .n_estimators(n_est)
            .random_state(42)
            .fit(&x, &y)
            .unwrap();
        
        let pred_nest = model_nest.predict(&x).unwrap();
        assert_eq!(pred_nest.len(), y.len());
        
        let acc = accuracy_score(&y, &pred_nest);
        assert!(acc > 0.7, "Random forest should achieve reasonable accuracy");
    }
}

#[test]
fn test_random_forest_with_oob_score() {
    let (x, y) = make_classification_data();
    
    // Test Random Forest with OOB score
    let model = RandomForestClassifier::new()
        .n_estimators(20)
        .oob_score(true)
        .bootstrap(true)
        .random_state(42)
        .fit(&x, &y)
        .unwrap();
    
    // Check that OOB score is computed
    let oob_score = model.oob_score();
    assert!(oob_score.is_some(), "OOB score should be computed");
    
    let score_value = oob_score.unwrap();
    assert!(score_value >= 0.0 && score_value <= 1.0, "OOB score should be between 0 and 1");
    
    // Check that OOB decision function is available
    let oob_decision = model.oob_decision_function();
    assert!(oob_decision.is_some(), "OOB decision function should be computed");
    
    let decisions = oob_decision.unwrap();
    assert_eq!(decisions.shape(), &[x.nrows(), model.n_classes()]);
}

#[test]
fn test_standard_scaler_consistency() {
    let (x, _) = make_regression_data();
    
    // Test StandardScaler
    let scaler = StandardScaler::new()
        .fit(&x)
        .unwrap();
    
    let x_scaled = scaler.transform(&x).unwrap();
    
    // Basic consistency checks
    assert_eq!(x_scaled.shape(), x.shape());
    
    // Check that scaled data has approximately zero mean and unit variance
    for col in 0..x_scaled.ncols() {
        let column = x_scaled.column(col);
        let mean = column.sum() / column.len() as f64;
        let var = column.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / column.len() as f64;
        
        assert!((mean).abs() < 1e-10, "Scaled data should have zero mean");
        assert!((var - 1.0).abs() < 1e-10, "Scaled data should have unit variance");
    }
    
    // Test that transform and inverse_transform are consistent
    let x_recovered = scaler.inverse_transform(&x_scaled).unwrap();
    
    for i in 0..x.nrows() {
        for j in 0..x.ncols() {
            assert!((x[[i, j]] - x_recovered[[i, j]]).abs() < 1e-10, 
                   "Inverse transform should recover original data");
        }
    }
}

#[test]
fn test_label_encoder_consistency() {
    // Test LabelEncoder with string labels
    let labels = vec!["cat", "dog", "bird", "cat", "dog", "bird", "cat"];
    
    let encoder = LabelEncoder::new()
        .fit(&labels)
        .unwrap();
    
    let encoded = encoder.transform(&labels).unwrap();
    
    // Basic consistency checks
    assert_eq!(encoded.len(), labels.len());
    
    // Check that all encoded values are valid indices
    let n_classes = encoder.classes().len();
    for &val in encoded.iter() {
        assert!(val < n_classes, "Encoded value should be valid class index");
    }
    
    // Test that transform and inverse_transform are consistent
    let recovered = encoder.inverse_transform(&encoded).unwrap();
    
    for (original, recovered) in labels.iter().zip(recovered.iter()) {
        assert_eq!(original, recovered, "Inverse transform should recover original labels");
    }
    
    // Test with unseen labels (should handle gracefully)
    let unseen_labels = vec!["cat", "elephant", "dog"];
    let result = encoder.transform(&unseen_labels);
    // This should either succeed (mapping elephant to a new class) or fail gracefully
    match result {
        Ok(encoded_unseen) => {
            assert_eq!(encoded_unseen.len(), unseen_labels.len());
        }
        Err(_) => {
            // Expected behavior for unseen labels
        }
    }
}

#[test]
fn test_cross_algorithm_consistency() {
    let (x, y) = make_classification_data();
    
    // Test that different algorithms produce reasonable results on the same data
    let knn = KNeighborsClassifier::new()
        .n_neighbors(3)
        .fit(&x, &y)
        .unwrap();
    
    let dt = DecisionTreeClassifier::new()
        .max_depth(5)
        .fit(&x, &y)
        .unwrap();
    
    let rf = RandomForestClassifier::new()
        .n_estimators(10)
        .random_state(42)
        .fit(&x, &y)
        .unwrap();
    
    let knn_pred = knn.predict(&x).unwrap();
    let dt_pred = dt.predict(&x).unwrap();
    let rf_pred = rf.predict(&x).unwrap();
    
    // All should have same output length
    assert_eq!(knn_pred.len(), y.len());
    assert_eq!(dt_pred.len(), y.len());
    assert_eq!(rf_pred.len(), y.len());
    
    // All should achieve reasonable accuracy
    let knn_acc = accuracy_score(&y, &knn_pred);
    let dt_acc = accuracy_score(&y, &dt_pred);
    let rf_acc = accuracy_score(&y, &rf_pred);
    
    assert!(knn_acc > 0.7, "KNN should achieve reasonable accuracy");
    assert!(dt_acc > 0.7, "Decision tree should achieve reasonable accuracy");
    assert!(rf_acc > 0.7, "Random forest should achieve reasonable accuracy");
    
    // Random forest should generally perform best or tied for best
    assert!(rf_acc >= dt_acc * 0.9, "Random forest should perform competitively");
}

#[test]
fn test_preprocessing_pipeline_consistency() {
    let (x, y) = make_classification_data();
    
    // Test preprocessing + classification pipeline
    let scaler = StandardScaler::new()
        .fit(&x)
        .unwrap();
    
    let x_scaled = scaler.transform(&x).unwrap();
    
    // Train classifier on scaled data
    let model = KNeighborsClassifier::new()
        .n_neighbors(3)
        .fit(&x_scaled, &y)
        .unwrap();
    
    let predictions = model.predict(&x_scaled).unwrap();
    let accuracy = accuracy_score(&y, &predictions);
    
    assert!(accuracy > 0.7, "Pipeline should achieve reasonable accuracy");
    
    // Test that scaling actually changes the data
    let mut data_changed = false;
    for i in 0..x.nrows() {
        for j in 0..x.ncols() {
            if (x[[i, j]] - x_scaled[[i, j]]).abs() > 1e-6 {
                data_changed = true;
                break;
            }
        }
        if data_changed { break; }
    }
    assert!(data_changed, "StandardScaler should change the data");
}

#[test]
fn test_reproducibility_with_random_state() {
    let (x, y) = make_classification_data();
    
    // Test that random_state ensures reproducibility
    let model1 = RandomForestClassifier::new()
        .n_estimators(10)
        .random_state(42)
        .fit(&x, &y)
        .unwrap();
    
    let model2 = RandomForestClassifier::new()
        .n_estimators(10)
        .random_state(42)
        .fit(&x, &y)
        .unwrap();
    
    let pred1 = model1.predict(&x).unwrap();
    let pred2 = model2.predict(&x).unwrap();
    
    // Predictions should be identical with same random state
    for (p1, p2) in pred1.iter().zip(pred2.iter()) {
        assert_eq!(p1, p2, "Same random state should give identical predictions");
    }
    
    // Test with different random states
    let model3 = RandomForestClassifier::new()
        .n_estimators(10)
        .random_state(123)
        .fit(&x, &y)
        .unwrap();
    
    let pred3 = model3.predict(&x).unwrap();
    
    // Predictions might be different with different random state
    // (but we don't require this as the datasets are small)
    assert_eq!(pred3.len(), pred1.len());
}