//! Advanced tests for label analysis, instance-based learning, SVM, and ranking

use super::*;
use crate::utilities::CLARE;
use scirs2_core::ndarray::{array, Array2};
use sklears_core::traits::{Fit, Predict};
use sklears_core::types::Float;

#[test]
fn test_label_analysis_basic() {
    let y = array![
        [1, 0, 1], // Cardinality 2
        [0, 1, 0], // Cardinality 1
        [1, 0, 1], // Cardinality 2 (duplicate)
        [0, 0, 0], // Cardinality 0
        [1, 1, 1], // Cardinality 3
    ];

    let results =
        label_analysis::analyze_combinations(&y.view()).expect("operation should succeed");

    assert_eq!(results.total_samples, 5);
    assert_eq!(results.combinations[0].combination.len(), 3); // Number of labels = 3
    assert_eq!(results.unique_combinations, 4); // [1,0,1], [0,1,0], [0,0,0], [1,1,1]

    // Check that [1,0,1] is most frequent (appears twice)
    assert_eq!(
        results
            .most_frequent
            .as_ref()
            .expect("operation should succeed")
            .combination,
        vec![1, 0, 1]
    );
    assert_eq!(
        results
            .most_frequent
            .as_ref()
            .expect("operation should succeed")
            .frequency,
        2
    );
    assert!(
        (results
            .most_frequent
            .as_ref()
            .expect("operation should succeed")
            .relative_frequency
            - 0.4)
            .abs()
            < 1e-10
    );
    assert_eq!(
        results
            .most_frequent
            .as_ref()
            .expect("operation should succeed")
            .cardinality,
        2
    );

    // Average cardinality should be (2+1+2+0+3)/5 = 1.6
    assert!((results.average_cardinality - 1.6).abs() < 1e-10);
}

#[test]
fn test_label_analysis_utility_functions() {
    let y = array![
        [1, 0],
        [1, 0],
        [1, 0], // Frequent: [1, 0] appears 3 times
        [0, 1],
        [0, 1], // Frequent: [0, 1] appears 2 times
        [1, 1]  // Rare: [1, 1] appears 1 time
    ];

    let results =
        label_analysis::analyze_combinations(&y.view()).expect("operation should succeed");

    // Test get_rare_combinations
    let rare = label_analysis::get_rare_combinations(&results, 2);
    assert_eq!(rare.len(), 2); // [0,1] freq=2 and [1,1] freq=1 are both <= threshold
                               // Find the combination with frequency 1
    let freq_1_combo = rare
        .iter()
        .find(|combo| combo.frequency == 1)
        .expect("operation should succeed");
    assert_eq!(freq_1_combo.combination, vec![1, 1]);

    // Test get_combinations_by_cardinality
    let cardinality_1 = label_analysis::get_combinations_by_cardinality(&results, 1);
    assert_eq!(cardinality_1.len(), 2); // [1, 0] and [0, 1]

    let cardinality_2 = label_analysis::get_combinations_by_cardinality(&results, 2);
    assert_eq!(cardinality_2.len(), 1); // [1, 1]
    assert_eq!(cardinality_2[0].combination, vec![1, 1]);
}

#[test]
fn test_label_cooccurrence_matrix() {
    let y = array![
        [1, 1, 0], // Labels 0 and 1 co-occur
        [1, 0, 1], // Labels 0 and 2 co-occur
        [0, 1, 1], // Labels 1 and 2 co-occur
        [1, 1, 1], // All labels co-occur
    ];

    let cooccurrence =
        label_analysis::label_cooccurrence_matrix(&y.view()).expect("operation should succeed");
    assert_eq!(cooccurrence.dim(), (3, 3));

    // Label 0 appears with itself in samples 0, 1, 3 = 3 times
    assert_eq!(cooccurrence[[0, 0]], 3);
    // Label 1 appears with itself in samples 0, 2, 3 = 3 times
    assert_eq!(cooccurrence[[1, 1]], 3);
    // Label 2 appears with itself in samples 1, 2, 3 = 3 times
    assert_eq!(cooccurrence[[2, 2]], 3);

    // Labels 0 and 1 co-occur in samples 0, 3 = 2 times
    assert_eq!(cooccurrence[[0, 1]], 2);
    assert_eq!(cooccurrence[[1, 0]], 2);

    // Labels 0 and 2 co-occur in samples 1, 3 = 2 times
    assert_eq!(cooccurrence[[0, 2]], 2);
    assert_eq!(cooccurrence[[2, 0]], 2);

    // Labels 1 and 2 co-occur in samples 2, 3 = 2 times
    assert_eq!(cooccurrence[[1, 2]], 2);
    assert_eq!(cooccurrence[[2, 1]], 2);
}

#[test]
fn test_label_correlation_matrix() {
    let y = array![[1, 1, 0], [1, 0, 1], [0, 1, 1], [0, 0, 0],];

    let correlation =
        label_analysis::label_correlation_matrix(&y.view()).expect("operation should succeed");
    assert_eq!(correlation.dim(), (3, 3));

    // Diagonal should be 1.0 (perfect self-correlation)
    assert!((correlation[[0, 0]] - 1.0).abs() < 1e-10);
    assert!((correlation[[1, 1]] - 1.0).abs() < 1e-10);
    assert!((correlation[[2, 2]] - 1.0).abs() < 1e-10);

    // Matrix should be symmetric
    for i in 0..3 {
        for j in 0..3 {
            assert!((correlation[[i, j]] - correlation[[j, i]]).abs() < 1e-10);
        }
    }

    // All correlations should be between -1 and 1
    for i in 0..3 {
        for j in 0..3 {
            assert!(correlation[[i, j]] >= -1.0 && correlation[[i, j]] <= 1.0);
        }
    }
}

#[test]
fn test_label_analysis_invalid_input() {
    // Test with non-binary labels
    let y_bad = array![[2, 1], [1, 0]]; // Contains non-binary value
    assert!(label_analysis::analyze_combinations(&y_bad.view()).is_err());

    // Test with empty array
    let y_empty = Array2::<i32>::zeros((0, 2));
    assert!(label_analysis::analyze_combinations(&y_empty.view()).is_err());

    let y_no_labels = Array2::<i32>::zeros((2, 0));
    assert!(label_analysis::analyze_combinations(&y_no_labels.view()).is_err());

    // Test cooccurrence matrix with empty data
    assert!(label_analysis::label_cooccurrence_matrix(&y_empty.view()).is_err());
    assert!(label_analysis::label_correlation_matrix(&y_empty.view()).is_err());
}

#[test]
fn test_label_analysis_edge_cases() {
    // Test with single sample
    let y_single = array![[1, 0, 1]];
    let results =
        label_analysis::analyze_combinations(&y_single.view()).expect("operation should succeed");

    assert_eq!(results.total_samples, 1);
    assert_eq!(results.unique_combinations, 1);
    assert_eq!(
        results
            .most_frequent
            .as_ref()
            .expect("operation should succeed")
            .combination,
        vec![1, 0, 1]
    );
    assert_eq!(
        results
            .least_frequent
            .as_ref()
            .expect("operation should succeed")
            .combination,
        vec![1, 0, 1]
    );
    assert_eq!(results.average_cardinality, 2.0);

    // Test with all zeros
    let y_zeros = array![[0, 0], [0, 0]];
    let results =
        label_analysis::analyze_combinations(&y_zeros.view()).expect("operation should succeed");

    assert_eq!(results.average_cardinality, 0.0);

    // Test with all ones
    let y_ones = array![[1, 1], [1, 1]];
    let results =
        label_analysis::analyze_combinations(&y_ones.view()).expect("operation should succeed");

    assert_eq!(results.average_cardinality, 2.0);
}

#[test]
fn test_iblr_basic_functionality() {
    let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 1.0], [4.0, 4.0]];
    let y = array![[1, 0], [0, 1], [1, 1], [0, 0]]; // Multi-label classification targets

    let iblr = IBLR::new().k_neighbors(2);
    let trained_iblr = iblr
        .fit(&X.view(), &y.view())
        .expect("model fitting should succeed");
    let predictions = trained_iblr
        .predict(&X.view())
        .expect("prediction should succeed");

    assert_eq!(predictions.dim(), (4, 2));

    // Check that predictions are binary (0 or 1)
    for i in 0..4 {
        for j in 0..2 {
            assert!(predictions[[i, j]] == 0 || predictions[[i, j]] == 1);
        }
    }
}

#[test]
fn test_iblr_configuration() {
    let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 1.0]];
    let y = array![[1, 0], [0, 1], [1, 1]]; // Multi-label classification targets

    // Test different k values
    let iblr1 = IBLR::new().k_neighbors(1);
    let iblr2 = IBLR::new().k_neighbors(2); // Must be < n_samples (3)

    let trained1 = iblr1
        .fit(&X.view(), &y.view())
        .expect("model fitting should succeed");
    let trained2 = iblr2
        .fit(&X.view(), &y.view())
        .expect("model fitting should succeed");

    let pred1 = trained1
        .predict(&X.view())
        .expect("prediction should succeed");
    let pred2 = trained2
        .predict(&X.view())
        .expect("prediction should succeed");

    assert_eq!(pred1.dim(), (3, 2));
    assert_eq!(pred2.dim(), (3, 2));

    // Test weight functions
    let iblr_uniform = IBLR::new().k_neighbors(2).weights(WeightFunction::Uniform);
    let iblr_distance = IBLR::new().k_neighbors(2).weights(WeightFunction::Distance);

    let trained_uniform = iblr_uniform
        .fit(&X.view(), &y.view())
        .expect("model fitting should succeed");
    let trained_distance = iblr_distance
        .fit(&X.view(), &y.view())
        .expect("model fitting should succeed");

    let pred_uniform = trained_uniform
        .predict(&X.view())
        .expect("prediction should succeed");
    let pred_distance = trained_distance
        .predict(&X.view())
        .expect("prediction should succeed");

    assert_eq!(pred_uniform.dim(), (3, 2));
    assert_eq!(pred_distance.dim(), (3, 2));
}

#[test]
fn test_iblr_error_handling() {
    let X = array![[1.0, 2.0], [2.0, 3.0]];
    let y = array![[1, 0], [0, 1], [1, 1]]; // Mismatched samples (3 vs 2)

    let iblr = IBLR::new();
    assert!(iblr.fit(&X.view(), &y.view()).is_err());

    // Test k_neighbors validation
    let y_valid = array![[1, 0], [0, 1]]; // Matching samples

    let iblr_zero_k = IBLR::new().k_neighbors(0);
    assert!(iblr_zero_k.fit(&X.view(), &y_valid.view()).is_err());

    let iblr_large_k = IBLR::new().k_neighbors(5); // More than samples
    assert!(iblr_large_k.fit(&X.view(), &y_valid.view()).is_err());

    // Test prediction with wrong feature dimensions
    let X_train = array![[1.0, 2.0], [2.0, 3.0]];
    let y_train = array![[1, 0], [0, 1]];
    let iblr_for_predict = IBLR::new().k_neighbors(1); // Must be < n_samples (2)
    let trained = iblr_for_predict
        .fit(&X_train.view(), &y_train.view())
        .expect("operation should succeed");

    let X_wrong_features = array![[1.0, 2.0, 3.0]]; // Extra feature
    assert!(trained.predict(&X_wrong_features.view()).is_err());

    // Test empty data
    let X_empty = Array2::<Float>::zeros((0, 2));
    let y_empty = Array2::<i32>::zeros((0, 2));
    assert!(IBLR::new().fit(&X_empty.view(), &y_empty.view()).is_err());
}

#[test]
fn test_iblr_weight_functions() {
    let X = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [2.0, 2.0]];
    let y = array![[1, 1], [0, 1], [1, 0], [0, 0]]; // Binary classification labels

    // Test uniform weighting
    let iblr_uniform = IBLR::new().k_neighbors(3).weights(WeightFunction::Uniform);
    let trained_uniform = iblr_uniform
        .fit(&X.view(), &y.view())
        .expect("model fitting should succeed");
    let pred_uniform = trained_uniform
        .predict(&X.view())
        .expect("prediction should succeed");

    // Test distance weighting
    let iblr_distance = IBLR::new().k_neighbors(3).weights(WeightFunction::Distance);
    let trained_distance = iblr_distance
        .fit(&X.view(), &y.view())
        .expect("model fitting should succeed");
    let pred_distance = trained_distance
        .predict(&X.view())
        .expect("prediction should succeed");

    // Predictions should be reasonable for both
    assert_eq!(pred_uniform.dim(), (4, 2));
    assert_eq!(pred_distance.dim(), (4, 2));

    // Check that all predictions are binary (0 or 1)
    for i in 0..4 {
        for j in 0..2 {
            assert!(pred_uniform[[i, j]] == 0 || pred_uniform[[i, j]] == 1);
            assert!(pred_distance[[i, j]] == 0 || pred_distance[[i, j]] == 1);
        }
    }
}

#[test]
fn test_iblr_single_neighbor() {
    let X = array![[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]];
    let y = array![[1, 0], [0, 1], [1, 1]]; // Binary classification labels

    let iblr = IBLR::new().k_neighbors(1);
    let trained = iblr
        .fit(&X.view(), &y.view())
        .expect("model fitting should succeed");

    // Test prediction on training data (should be exact for k=1)
    let predictions = trained
        .predict(&X.view())
        .expect("prediction should succeed");

    for i in 0..3 {
        for j in 0..2 {
            assert_eq!(predictions[[i, j]], y[[i, j]]);
        }
    }
}

#[test]
fn test_iblr_interpolation() {
    let X = array![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]];
    let y = array![[0, 0], [1, 1], [0, 1]]; // Binary classification labels

    let iblr = IBLR::new().k_neighbors(2);
    let trained = iblr
        .fit(&X.view(), &y.view())
        .expect("model fitting should succeed");

    // Test prediction at midpoint
    let X_test = array![[0.5, 0.5]];
    let prediction = trained
        .predict(&X_test.view())
        .expect("prediction should succeed");

    // Should predict binary values (0 or 1)
    assert!(prediction[[0, 0]] == 0 || prediction[[0, 0]] == 1);
    assert!(prediction[[0, 1]] == 0 || prediction[[0, 1]] == 1);
}

#[test]
fn test_clare_basic_functionality() {
    let X = array![[1.0, 1.0], [1.5, 1.5], [5.0, 5.0], [5.5, 5.5]];
    let y = array![[1, 0], [1, 0], [0, 1], [0, 1]]; // Two clear clusters with different label patterns

    let clare = CLARE::new().n_clusters(2).random_state(42);
    let trained_clare = clare
        .fit(&X.view(), &y)
        .expect("model fitting should succeed");
    let predictions = trained_clare
        .predict(&X.view())
        .expect("prediction should succeed");

    assert_eq!(predictions.dim(), (4, 2));

    // Verify cluster centers and assignments were learned
    assert_eq!(trained_clare.cluster_centers().dim(), (2, 2));
    assert_eq!(trained_clare.cluster_assignments().len(), 4);
}

#[test]
fn test_clare_configuration() {
    let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 1.0], [4.0, 4.0]];
    let y = array![[1, 0], [0, 1], [1, 1], [0, 0]];

    // Test different configurations
    let clare1 = CLARE::new().n_clusters(2).threshold(0.3);
    let clare2 = CLARE::new().n_clusters(3).max_iter(50);
    let clare3 = CLARE::new().random_state(123);

    let trained1 = clare1
        .fit(&X.view(), &y)
        .expect("model fitting should succeed");
    let trained2 = clare2
        .fit(&X.view(), &y)
        .expect("model fitting should succeed");
    let trained3 = clare3
        .fit(&X.view(), &y)
        .expect("model fitting should succeed");

    let pred1 = trained1
        .predict(&X.view())
        .expect("prediction should succeed");
    let pred2 = trained2
        .predict(&X.view())
        .expect("prediction should succeed");
    let pred3 = trained3
        .predict(&X.view())
        .expect("prediction should succeed");

    assert_eq!(pred1.dim(), (4, 2));
    assert_eq!(pred2.dim(), (4, 2));
    assert_eq!(pred3.dim(), (4, 2));

    // Test accessors
    assert_eq!(trained1.threshold(), 0.3);
    assert_eq!(trained1.cluster_centers().dim(), (2, 2));
    assert_eq!(trained2.cluster_centers().dim(), (3, 2));
}

#[test]
fn test_clare_error_handling() {
    let X = array![[1.0, 2.0], [2.0, 3.0]];
    let y = array![[1, 0], [0, 1], [1, 1]]; // Mismatched samples

    let clare = CLARE::new();
    assert!(clare.fit(&X.view(), &y).is_err());

    // Test n_clusters validation
    let y_valid = array![[1, 0], [0, 1]];

    let clare_zero_clusters = CLARE::new().n_clusters(0);
    assert!(clare_zero_clusters.fit(&X.view(), &y_valid).is_err());

    let clare_too_many_clusters = CLARE::new().n_clusters(5); // More than samples
    assert!(clare_too_many_clusters.fit(&X.view(), &y_valid).is_err());

    // Test non-binary labels
    let y_non_binary = array![[1, 2], [0, 1]]; // Contains 2
    assert!(CLARE::new().fit(&X.view(), &y_non_binary).is_err());

    // Test prediction with wrong feature dimensions
    let X_train = array![[1.0, 2.0], [2.0, 3.0]];
    let y_train = array![[1, 0], [0, 1]];
    let clare_for_predict = CLARE::new().n_clusters(2);
    let trained = clare_for_predict
        .fit(&X_train.view(), &y_train)
        .expect("model fitting should succeed");

    let X_wrong_features = array![[1.0, 2.0, 3.0]]; // Extra feature
    assert!(trained.predict(&X_wrong_features.view()).is_err());

    // Test empty data
    let X_empty = Array2::<Float>::zeros((0, 2));
    let y_empty = Array2::<i32>::zeros((0, 2));
    assert!(CLARE::new().fit(&X_empty.view(), &y_empty).is_err());
}

#[test]
fn test_clare_threshold_prediction() {
    let X = array![[1.0, 1.0], [1.2, 1.2], [5.0, 5.0], [5.2, 5.2]];
    let y = array![[1, 0], [1, 0], [0, 1], [0, 1]];

    let clare = CLARE::new().n_clusters(2).threshold(0.3).random_state(42);
    let trained_clare = clare
        .fit(&X.view(), &y)
        .expect("model fitting should succeed");

    // Test predictions are binary
    let predictions = trained_clare
        .predict(&X.view())
        .expect("prediction should succeed");
    assert_eq!(predictions.dim(), (4, 2));

    // All predictions should be 0 or 1
    for pred in predictions.iter() {
        assert!(*pred == 0 || *pred == 1);
    }

    // Verify threshold was set correctly
    assert_eq!(trained_clare.threshold(), 0.3);
}

#[test]
fn test_clare_clustering_consistency() {
    let X = array![
        [0.0, 0.0],
        [0.1, 0.1], // First cluster
        [5.0, 5.0],
        [5.1, 5.1] // Second cluster
    ];
    let y = array![
        [1, 0],
        [1, 0], // First cluster: always label 0 active
        [0, 1],
        [0, 1] // Second cluster: always label 1 active
    ];

    let clare = CLARE::new().n_clusters(2).threshold(0.5).random_state(42);
    let trained_clare = clare
        .fit(&X.view(), &y)
        .expect("model fitting should succeed");
    let predictions = trained_clare
        .predict(&X.view())
        .expect("prediction should succeed");

    // With clear clustering, predictions should match patterns
    assert_eq!(predictions.dim(), (4, 2));
    assert!(predictions.iter().all(|&x| x == 0 || x == 1));

    // Test threshold accessor
    assert!((trained_clare.threshold() - 0.5).abs() < 1e-10);
}

#[test]
fn test_clare_single_cluster() {
    let X = array![[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]];
    let y = array![[1, 0], [0, 1], [1, 1]];

    // Use only 1 cluster
    let clare = CLARE::new().n_clusters(1);
    let trained_clare = clare
        .fit(&X.view(), &y)
        .expect("model fitting should succeed");
    let predictions = trained_clare
        .predict(&X.view())
        .expect("prediction should succeed");

    assert_eq!(predictions.dim(), (3, 2));
    assert_eq!(trained_clare.cluster_centers().dim(), (1, 2));

    // With 1 cluster, all samples should get same prediction
    // (based on average label frequency)
    for i in 1..3 {
        for j in 0..2 {
            assert_eq!(predictions[[0, j]], predictions[[i, j]]);
        }
    }
}

#[test]
fn test_clare_reproducibility() {
    let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 1.0], [4.0, 4.0]];
    let y = array![[1, 0], [0, 1], [1, 1], [0, 0]];

    // Train two models with same random state
    let clare1 = CLARE::new().n_clusters(2).random_state(42);
    let trained1 = clare1
        .fit(&X.view(), &y)
        .expect("model fitting should succeed");

    let clare2 = CLARE::new().n_clusters(2).random_state(42);
    let trained2 = clare2
        .fit(&X.view(), &y)
        .expect("model fitting should succeed");

    // Should produce same cluster centers
    let centers1 = trained1.cluster_centers();
    let centers2 = trained2.cluster_centers();

    for i in 0..centers1.nrows() {
        for j in 0..centers1.ncols() {
            assert!((centers1[[i, j]] - centers2[[i, j]]).abs() < 1e-10);
        }
    }

    // Should produce same predictions
    let pred1 = trained1
        .predict(&X.view())
        .expect("prediction should succeed");
    let pred2 = trained2
        .predict(&X.view())
        .expect("prediction should succeed");

    for i in 0..pred1.nrows() {
        for j in 0..pred1.ncols() {
            assert_eq!(pred1[[i, j]], pred2[[i, j]]);
        }
    }
}

#[test]
fn test_mltsvm_basic_functionality() {
    let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 1.0], [4.0, 4.0]];
    let y = array![[1, 0], [0, 1], [1, 1], [0, 0]]; // Multi-label binary

    let mltsvm = MLTSVM::new().c1(1.0).c2(1.0);
    let trained_mltsvm = mltsvm
        .fit(&X.view(), &y)
        .expect("model fitting should succeed");
    let predictions = trained_mltsvm
        .predict(&X.view())
        .expect("prediction should succeed");

    assert_eq!(predictions.dim(), (4, 2));
    assert_eq!(trained_mltsvm.n_labels(), 2);

    // All predictions should be binary (0 or 1)
    for &pred in predictions.iter() {
        assert!(pred == 0 || pred == 1);
    }
}

#[test]
fn test_mltsvm_configuration() {
    let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 1.0], [4.0, 4.0]];
    let y = array![[1, 0], [0, 1], [1, 1], [0, 0]];

    // Test different configurations
    let mltsvm1 = MLTSVM::new().c1(0.5).c2(1.5);
    let mltsvm2 = MLTSVM::new().epsilon(1e-8).max_iter(500);

    let trained1 = mltsvm1
        .fit(&X.view(), &y)
        .expect("model fitting should succeed");
    let trained2 = mltsvm2
        .fit(&X.view(), &y)
        .expect("model fitting should succeed");

    let pred1 = trained1
        .predict(&X.view())
        .expect("prediction should succeed");
    let pred2 = trained2
        .predict(&X.view())
        .expect("prediction should succeed");

    assert_eq!(pred1.dim(), (4, 2));
    assert_eq!(pred2.dim(), (4, 2));
}

#[test]
fn test_mltsvm_error_handling() {
    let X = array![[1.0, 2.0], [2.0, 3.0]];
    let y = array![[1, 0], [0, 1], [1, 1]]; // Mismatched samples

    let mltsvm = MLTSVM::new();
    assert!(mltsvm.fit(&X.view(), &y).is_err());

    // Test non-binary labels
    let y_non_binary = array![[1, 2], [0, 1]]; // Contains 2
    assert!(MLTSVM::new().fit(&X.view(), &y_non_binary).is_err());

    // Test prediction with wrong feature dimensions
    let X_train = array![[1.0, 2.0], [2.0, 3.0], [3.0, 1.0], [4.0, 2.0]];
    let y_train = array![[1, 0], [0, 1], [1, 1], [0, 0]];
    let mltsvm_for_predict = MLTSVM::new();
    let trained = mltsvm_for_predict
        .fit(&X_train.view(), &y_train)
        .expect("model fitting should succeed");

    let X_wrong_features = array![[1.0, 2.0, 3.0]]; // Extra feature
    assert!(trained.predict(&X_wrong_features.view()).is_err());

    // Test empty data
    let X_empty = Array2::<Float>::zeros((0, 2));
    let y_empty = Array2::<i32>::zeros((0, 2));
    assert!(MLTSVM::new().fit(&X_empty.view(), &y_empty).is_err());
}

#[test]
fn test_mltsvm_decision_function() {
    let X = array![[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]];
    let y = array![[1, 0], [1, 0], [0, 1], [0, 1]];

    let mltsvm = MLTSVM::new();
    let trained_mltsvm = mltsvm
        .fit(&X.view(), &y)
        .expect("model fitting should succeed");

    // Test decision function
    let decision_values = trained_mltsvm
        .decision_function(&X.view())
        .expect("operation should succeed");
    assert_eq!(decision_values.dim(), (4, 2));

    // Decision values should be real numbers (no constraints on range)
    // Just check that we get reasonable outputs
    for &val in decision_values.iter() {
        assert!(val.is_finite());
    }
}

#[test]
fn test_mltsvm_separable_data() {
    let X = array![
        [0.0, 0.0],
        [0.5, 0.5], // Negative class cluster
        [3.0, 3.0],
        [3.5, 3.5] // Positive class cluster
    ];
    let y = array![
        [0, 1],
        [0, 1], // First label: negative, Second label: positive
        [1, 0],
        [1, 0] // First label: positive, Second label: negative
    ];

    let mltsvm = MLTSVM::new().c1(1.0).c2(1.0);
    let trained_mltsvm = mltsvm
        .fit(&X.view(), &y)
        .expect("model fitting should succeed");
    let predictions = trained_mltsvm
        .predict(&X.view())
        .expect("prediction should succeed");

    // With linearly separable data, MLTSVM should perform well
    let mut correct_predictions = 0;
    let total_predictions = predictions.len();

    for i in 0..predictions.nrows() {
        for j in 0..predictions.ncols() {
            if predictions[[i, j]] == y[[i, j]] {
                correct_predictions += 1;
            }
        }
    }

    let accuracy = correct_predictions as Float / total_predictions as Float;
    // Should get reasonably good accuracy on separable data
    assert!(accuracy >= 0.5); // At least better than random
}

#[test]
fn test_mltsvm_feature_scaling() {
    // Test with features of very different scales
    let X = array![
        [1000.0, 0.001],
        [2000.0, 0.002],
        [3000.0, 0.003],
        [4000.0, 0.004]
    ];
    let y = array![[1, 0], [0, 1], [1, 1], [0, 0]];

    let mltsvm = MLTSVM::new();
    let trained_mltsvm = mltsvm
        .fit(&X.view(), &y)
        .expect("model fitting should succeed");
    let predictions = trained_mltsvm
        .predict(&X.view())
        .expect("prediction should succeed");

    // Should handle feature scaling internally
    assert_eq!(predictions.dim(), (4, 2));

    // All predictions should be binary
    for &pred in predictions.iter() {
        assert!(pred == 0 || pred == 1);
    }
}

#[test]
fn test_mltsvm_consistency() {
    let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 1.0], [4.0, 4.0]];
    let y = array![[1, 0], [0, 1], [1, 1], [0, 0]];

    // Train the same model multiple times (deterministic should give same results)
    let mltsvm1 = MLTSVM::new().c1(1.0).c2(1.0);
    let trained1 = mltsvm1
        .fit(&X.view(), &y)
        .expect("model fitting should succeed");

    let mltsvm2 = MLTSVM::new().c1(1.0).c2(1.0);
    let trained2 = mltsvm2
        .fit(&X.view(), &y)
        .expect("model fitting should succeed");

    let pred1 = trained1
        .predict(&X.view())
        .expect("prediction should succeed");
    let pred2 = trained2
        .predict(&X.view())
        .expect("prediction should succeed");

    // Should be deterministic (same predictions)
    for i in 0..pred1.nrows() {
        for j in 0..pred1.ncols() {
            assert_eq!(pred1[[i, j]], pred2[[i, j]]);
        }
    }
}

#[test]
fn test_ranksvm_basic_functionality() {
    let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 1.0], [4.0, 4.0]];
    let y = array![[1, 0], [0, 1], [1, 1], [0, 0]]; // Multi-label binary

    let ranksvm = RankSVM::new().c(1.0);
    let trained_ranksvm = ranksvm
        .fit(&X.view(), &y)
        .expect("model fitting should succeed");
    let predictions = trained_ranksvm
        .predict(&X.view())
        .expect("prediction should succeed");

    assert_eq!(predictions.dim(), (4, 2));
    assert_eq!(trained_ranksvm.n_labels(), 2);

    // All predictions should be binary (0 or 1)
    for &pred in predictions.iter() {
        assert!(pred == 0 || pred == 1);
    }
}

#[test]
fn test_ranksvm_threshold_strategies() {
    let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 1.0], [4.0, 4.0]];
    let y = array![[1, 0], [0, 1], [1, 1], [0, 0]];

    // Test different threshold strategies
    let ranksvm1 = RankSVM::new().threshold_strategy(SVMThresholdStrategy::Fixed(0.5));
    let ranksvm2 = RankSVM::new().threshold_strategy(SVMThresholdStrategy::OptimizeF1);
    let ranksvm3 = RankSVM::new().threshold_strategy(SVMThresholdStrategy::TopK(2));
    let ranksvm4 = RankSVM::new().threshold_strategy(SVMThresholdStrategy::OptimizeF1);

    let trained1 = ranksvm1
        .fit(&X.view(), &y)
        .expect("model fitting should succeed");
    let trained2 = ranksvm2
        .fit(&X.view(), &y)
        .expect("model fitting should succeed");
    let trained3 = ranksvm3
        .fit(&X.view(), &y)
        .expect("model fitting should succeed");
    let trained4 = ranksvm4
        .fit(&X.view(), &y)
        .expect("model fitting should succeed");

    let pred1 = trained1
        .predict(&X.view())
        .expect("prediction should succeed");
    let pred2 = trained2
        .predict(&X.view())
        .expect("prediction should succeed");
    let pred3 = trained3
        .predict(&X.view())
        .expect("prediction should succeed");
    let pred4 = trained4
        .predict(&X.view())
        .expect("prediction should succeed");

    assert_eq!(pred1.dim(), (4, 2));
    assert_eq!(pred2.dim(), (4, 2));
    assert_eq!(pred3.dim(), (4, 2));
    assert_eq!(pred4.dim(), (4, 2));

    // Test threshold accessors
    assert_eq!(trained1.thresholds().len(), 2);
    assert_eq!(trained2.thresholds().len(), 2);
    assert_eq!(trained3.thresholds().len(), 2);
}

#[test]
fn test_ranksvm_decision_function() {
    let X = array![[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]];
    let y = array![[1, 0], [1, 0], [0, 1], [0, 1]];

    let ranksvm = RankSVM::new();
    let trained_ranksvm = ranksvm
        .fit(&X.view(), &y)
        .expect("model fitting should succeed");

    // Test decision function
    let scores = trained_ranksvm
        .decision_function(&X.view())
        .expect("operation should succeed");
    assert_eq!(scores.dim(), (4, 2));

    // Scores should be real numbers
    for &score in scores.iter() {
        assert!(score.is_finite());
    }
}

#[test]
fn test_ranksvm_predict_ranking() {
    let X = array![[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]];
    let y = array![[1, 0, 0], [0, 1, 0], [0, 0, 1]]; // Three labels, one active per sample

    let ranksvm = RankSVM::new();
    let trained_ranksvm = ranksvm
        .fit(&X.view(), &y)
        .expect("model fitting should succeed");

    // Test ranking prediction
    let rankings = trained_ranksvm
        .predict_ranking(&X.view())
        .expect("operation should succeed");

    assert_eq!(rankings.dim(), (3, 3)); // 3 samples, 3 labels
    for sample_idx in 0..3 {
        let mut ranking_vec = Vec::new();
        for label_idx in 0..3 {
            ranking_vec.push(rankings[[sample_idx, label_idx]]);
        }
        // Should contain all label indices
        let mut sorted_ranking = ranking_vec.clone();
        sorted_ranking.sort();
        assert_eq!(sorted_ranking, vec![0, 1, 2]);
    }
}

#[test]
fn test_ranksvm_error_handling() {
    let X = array![[1.0, 2.0], [2.0, 3.0]];
    let y = array![[1, 0], [0, 1], [1, 1]]; // Mismatched samples

    let ranksvm = RankSVM::new();
    assert!(ranksvm.fit(&X.view(), &y).is_err());

    // Test non-binary labels
    let y_non_binary = array![[1, 2], [0, 1]]; // Contains 2
    assert!(RankSVM::new().fit(&X.view(), &y_non_binary).is_err());

    // Test prediction with wrong feature dimensions
    let X_train = array![[1.0, 2.0], [2.0, 3.0], [3.0, 1.0], [4.0, 2.0]];
    let y_train = array![[1, 0], [0, 1], [1, 1], [0, 0]];
    let ranksvm_for_predict = RankSVM::new();
    let trained = ranksvm_for_predict
        .fit(&X_train.view(), &y_train)
        .expect("model fitting should succeed");

    let X_wrong_features = array![[1.0, 2.0, 3.0]]; // Extra feature
    assert!(trained.predict(&X_wrong_features.view()).is_err());
    assert!(trained.decision_function(&X_wrong_features.view()).is_err());
    assert!(trained.predict_ranking(&X_wrong_features.view()).is_err());

    // Test empty data
    let X_empty = Array2::<Float>::zeros((0, 2));
    let y_empty = Array2::<i32>::zeros((0, 2));
    assert!(RankSVM::new().fit(&X_empty.view(), &y_empty).is_err());
}

#[test]
fn test_ranksvm_configuration() {
    let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 1.0], [4.0, 4.0]];
    let y = array![[1, 0], [0, 1], [1, 1], [0, 0]];

    // Test different configurations
    let ranksvm1 = RankSVM::new().c(0.5).epsilon(1e-8);
    let ranksvm2 = RankSVM::new().max_iter(500);

    let trained1 = ranksvm1
        .fit(&X.view(), &y)
        .expect("model fitting should succeed");
    let trained2 = ranksvm2
        .fit(&X.view(), &y)
        .expect("model fitting should succeed");

    let pred1 = trained1
        .predict(&X.view())
        .expect("prediction should succeed");
    let pred2 = trained2
        .predict(&X.view())
        .expect("prediction should succeed");

    assert_eq!(pred1.dim(), (4, 2));
    assert_eq!(pred2.dim(), (4, 2));
}

#[test]
fn test_ranksvm_ranking_consistency() {
    let X = array![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]];
    let y = array![
        [0, 0, 1], // Last label should rank highest
        [0, 1, 0], // Middle label should rank highest
        [1, 0, 0]  // First label should rank highest
    ];

    let ranksvm = RankSVM::new().threshold_strategy(SVMThresholdStrategy::OptimizeF1);
    let trained_ranksvm = ranksvm
        .fit(&X.view(), &y)
        .expect("model fitting should succeed");

    let rankings = trained_ranksvm
        .predict_ranking(&X.view())
        .expect("operation should succeed");
    let scores = trained_ranksvm
        .decision_function(&X.view())
        .expect("operation should succeed");

    // Check that rankings are consistent with scores
    for i in 0..3 {
        // First ranked label should have highest score
        let top_label = rankings[[i, 0]];
        for j in 1..3 {
            let other_label = rankings[[i, j]];
            assert!(scores[[i, top_label]] >= scores[[i, other_label]]);
        }
    }
}

#[test]
fn test_ranksvm_single_class_handling() {
    let X = array![[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]];

    // Test with all positive for one label, mixed for other
    let y = array![[1, 0], [1, 1], [1, 0]]; // First label: all positive, second label: mixed

    let ranksvm = RankSVM::new();
    let trained = ranksvm
        .fit(&X.view(), &y)
        .expect("model fitting should succeed");
    let predictions = trained
        .predict(&X.view())
        .expect("prediction should succeed");
    let scores = trained
        .decision_function(&X.view())
        .expect("operation should succeed");

    assert_eq!(predictions.dim(), (3, 2));
    assert_eq!(scores.dim(), (3, 2));

    // All scores should be finite
    for &score in scores.iter() {
        assert!(score.is_finite());
    }
}

#[test]
fn test_ranksvm_reproducibility() {
    let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 1.0], [4.0, 4.0]];
    let y = array![[1, 0], [0, 1], [1, 1], [0, 0]];

    // Train two models with same configuration
    let ranksvm1 = RankSVM::new().c(1.0).epsilon(1e-6);
    let trained1 = ranksvm1
        .fit(&X.view(), &y)
        .expect("model fitting should succeed");

    let ranksvm2 = RankSVM::new().c(1.0).epsilon(1e-6);
    let trained2 = ranksvm2
        .fit(&X.view(), &y)
        .expect("model fitting should succeed");

    let pred1 = trained1
        .predict(&X.view())
        .expect("prediction should succeed");
    let pred2 = trained2
        .predict(&X.view())
        .expect("prediction should succeed");
    let scores1 = trained1
        .decision_function(&X.view())
        .expect("operation should succeed");
    let scores2 = trained2
        .decision_function(&X.view())
        .expect("operation should succeed");

    // Should be deterministic (same predictions and scores)
    for i in 0..pred1.nrows() {
        for j in 0..pred1.ncols() {
            assert_eq!(pred1[[i, j]], pred2[[i, j]]);
            assert!((scores1[[i, j]] - scores2[[i, j]]).abs() < 1e-10);
        }
    }
}
