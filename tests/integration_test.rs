//! Integration tests for sklears
//!
//! These tests verify that different crates work together correctly.

use ndarray::{array, Array2};
use sklears_core::traits::{Transform};
use sklears_utils::data_generation::{make_classification, make_blobs};
use sklears_neighbors::KNeighborsClassifier;
use sklears_preprocessing::scaling::StandardScaler;
use sklears_metrics::classification::{accuracy_score, precision_score, recall_score, f1_score};

#[test]
fn test_end_to_end_classification_pipeline() {
    // Generate synthetic data
    let (x, y) = make_classification(100, 4, 3, None, None, 0.0, 1.0, Some(42)).unwrap();
    
    // Preprocess data
    let scaler = StandardScaler::new();
    let fitted_scaler = scaler.fit(&x).unwrap();
    let x_scaled = fitted_scaler.transform(&x).unwrap();
    
    // Train classifier
    let classifier = KNeighborsClassifier::new(3);
    let fitted_classifier = classifier.fit(&x_scaled, &y).unwrap();
    
    // Make predictions
    let predictions = fitted_classifier.predict(&x_scaled).unwrap();
    
    // Evaluate performance
    let accuracy = accuracy_score(&y, &predictions).unwrap();
    
    // Should have reasonable accuracy on training data
    assert!(accuracy > 0.7, "Accuracy should be > 0.7, got {}", accuracy);
    
    // Check that predictions have the right shape
    assert_eq!(predictions.len(), y.len());
    
    // Check that all predicted classes are valid
    for &pred in predictions.iter() {
        assert!(pred >= 0 && pred <= 2, "Predicted class {} should be in [0, 2]", pred);
    }
}

#[test]
fn test_clustering_and_classification() {
    // Generate blob data for clustering
    let (x, y_true) = make_blobs(60, 2, Some(3), 1.0, (-5.0, 5.0), Some(42)).unwrap();
    
    // Use KNN to classify based on cluster structure
    let classifier = KNeighborsClassifier::new(5);
    let fitted_classifier = classifier.fit(&x, &y_true).unwrap();
    
    // Predict on the same data (should be very accurate)
    let predictions = fitted_classifier.predict(&x).unwrap();
    
    let accuracy = accuracy_score(&y_true, &predictions).unwrap();
    assert!(accuracy > 0.9, "Accuracy on blob data should be > 0.9, got {}", accuracy);
}

#[test]
fn test_metrics_consistency() {
    // Create a simple binary classification case
    let y_true = array![0, 1, 1, 0, 1, 0, 1, 1, 0, 0];
    let y_pred = array![0, 1, 0, 0, 1, 1, 1, 1, 0, 1];
    
    // Calculate all metrics
    let accuracy = accuracy_score(&y_true, &y_pred).unwrap();
    let precision = precision_score(&y_true, &y_pred, Some(1)).unwrap();
    let recall = recall_score(&y_true, &y_pred, Some(1)).unwrap();
    let f1 = f1_score(&y_true, &y_pred, Some(1)).unwrap();
    
    // Basic sanity checks
    assert!(accuracy >= 0.0 && accuracy <= 1.0);
    assert!(precision >= 0.0 && precision <= 1.0);
    assert!(recall >= 0.0 && recall <= 1.0);
    assert!(f1 >= 0.0 && f1 <= 1.0);
    
    // F1 should be harmonic mean of precision and recall
    let expected_f1 = 2.0 * precision * recall / (precision + recall);
    assert!((f1 - expected_f1).abs() < 1e-10);
}

#[test]
fn test_preprocessing_pipeline() {
    // Create test data with different scales
    let x = Array2::from_shape_vec((4, 3), vec![
        1.0, 10.0, 100.0,
        2.0, 20.0, 200.0,
        3.0, 30.0, 300.0,
        4.0, 40.0, 400.0,
    ]).unwrap();
    
    // Apply standard scaling
    let scaler = StandardScaler::new();
    let fitted_scaler = scaler.fit(&x).unwrap();
    let x_scaled = fitted_scaler.transform(&x).unwrap();
    
    // Check that scaled data has approximately zero mean and unit variance
    let means = x_scaled.mean_axis(ndarray::Axis(0)).unwrap();
    let vars = x_scaled.var_axis(ndarray::Axis(0), 0.0);
    
    for &mean in means.iter() {
        assert!((mean.abs()) < 1e-10, "Mean should be ~0, got {}", mean);
    }
    
    for &var in vars.iter() {
        assert!((var - 1.0).abs() < 1e-10, "Variance should be ~1, got {}", var);
    }
}

#[test]
fn test_data_generation_consistency() {
    // Test that data generation functions produce consistent outputs
    let (x1, y1) = make_classification(50, 3, 2, None, None, 0.0, 1.0, Some(42)).unwrap();
    let (x2, y2) = make_classification(50, 3, 2, None, None, 0.0, 1.0, Some(42)).unwrap();
    
    // With same random seed, should produce identical results
    assert_eq!(x1, x2);
    assert_eq!(y1, y2);
    
    // Check shapes
    assert_eq!(x1.shape(), &[50, 3]);
    assert_eq!(y1.len(), 50);
    
    // Check that we have the expected number of classes
    let mut classes: Vec<i32> = y1.iter().copied().collect();
    classes.sort_unstable();
    classes.dedup();
    assert_eq!(classes.len(), 2);
}

#[test]
fn test_cross_crate_type_compatibility() {
    // Test that types from different crates work together seamlessly
    use sklears_utils::array_utils::{unique, one_hot_encode};
    
    let labels = array![0, 1, 2, 1, 0, 2];
    let unique_labels = unique(&labels);
    assert_eq!(unique_labels, vec![0, 1, 2]);
    
    let encoded = one_hot_encode(&labels).unwrap();
    assert_eq!(encoded.shape(), &[6, 3]);
    
    // Verify encoding is correct
    assert_eq!(encoded[[0, 0]], 1.0); // label 0 -> [1, 0, 0]
    assert_eq!(encoded[[1, 1]], 1.0); // label 1 -> [0, 1, 0]
    assert_eq!(encoded[[2, 2]], 1.0); // label 2 -> [0, 0, 1]
}