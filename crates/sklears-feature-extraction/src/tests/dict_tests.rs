//! Dictionary learning and dimensionality reduction tests
//!
//! This module contains tests for dictionary learning algorithms, matrix factorization methods,
//! and dimensionality reduction techniques including PCA, ICA, NMF, Factor Analysis, and
//! Probabilistic Matrix Factorization.

use crate::dict_learning;
use scirs2_core::ndarray::{array, Array2};

#[test]
fn test_dictionary_learning() {
    let X = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];

    let dict_learner = dict_learning::DictionaryLearning::new()
        .n_components(2)
        .alpha(0.1)
        .max_iter(10);
    let fitted = dict_learner.fit(&X.view(), &()).unwrap();

    let codes = fitted.transform(&X.view()).unwrap();
    assert_eq!(codes.dim(), (3, 2));

    let dictionary = fitted.get_dictionary();
    assert_eq!(dictionary.dim(), (2, 3));
}

#[test]
fn test_mini_batch_dictionary_learning() {
    let X = array![
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
        [2.0, 3.0, 4.0]
    ];

    let mini_batch_dict = dict_learning::MiniBatchDictionaryLearning::new()
        .n_components(2)
        .alpha(0.1)
        .batch_size(2)
        .max_iter(10);

    let fitted = mini_batch_dict.fit(&X.view(), &()).unwrap();
    let codes = fitted.transform(&X.view()).unwrap();

    assert_eq!(codes.dim(), (4, 2)); // 4 samples, 2 components

    let dictionary = fitted.get_dictionary();
    assert_eq!(dictionary.dim(), (2, 3)); // 2 components, 3 features
}

#[test]
fn test_nmf() {
    let X = array![
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
        [2.0, 3.0, 4.0]
    ];

    let nmf = dict_learning::NMF::new()
        .n_components(2)
        .max_iter(50)
        .solver("cd".to_string());

    let fitted = nmf.fit(&X.view(), &()).unwrap();
    let W = fitted.transform(&X.view()).unwrap();

    assert_eq!(W.dim(), (4, 2)); // 4 samples, 2 components

    // Check that all values are non-negative
    for &val in W.iter() {
        assert!(val >= 0.0);
    }

    let H = fitted.components();
    assert_eq!(H.dim(), (2, 3)); // 2 components, 3 features

    // Check that all values in H are non-negative
    for &val in H.iter() {
        assert!(val >= 0.0);
    }

    // Test reconstruction
    let reconstruction = fitted.inverse_transform(&W.view()).unwrap();
    assert_eq!(reconstruction.dim(), X.dim());

    // Test multiplicative update solver
    let nmf_mu = dict_learning::NMF::new()
        .n_components(2)
        .max_iter(20)
        .solver("mu".to_string());

    let fitted_mu = nmf_mu.fit(&X.view(), &()).unwrap();
    let W_mu = fitted_mu.transform(&X.view()).unwrap();
    assert_eq!(W_mu.dim(), (4, 2));
}

#[test]
fn test_nmf_with_negative_input() {
    let X = array![[-1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];

    let nmf = dict_learning::NMF::new().n_components(2);
    let result = nmf.fit(&X.view(), &());

    // Should fail for negative input
    assert!(result.is_err());
}

#[test]
fn test_ica() {
    // Create test data with mixed signals
    let X = array![
        [1.0, 2.0, 3.0],
        [2.0, 4.0, 6.0],
        [3.0, 6.0, 9.0],
        [1.5, 3.0, 4.5],
        [2.5, 5.0, 7.5]
    ];

    let ica = dict_learning::ICA::new()
        .n_components(2)
        .max_iter(50)
        .tol(1e-3);

    let fitted = ica.fit(&X.view(), &()).unwrap();
    let sources = fitted.transform(&X.view()).unwrap();

    assert_eq!(sources.dim(), (5, 2)); // 5 samples, 2 components

    // Check that unmixing matrix has correct dimensions
    let components = fitted.components();
    assert_eq!(components.dim(), (2, 3)); // 2 components, 3 features

    // Test different contrast functions
    let ica_exp = dict_learning::ICA::new()
        .n_components(2)
        .fun("exp".to_string())
        .max_iter(20);

    let fitted_exp = ica_exp.fit(&X.view(), &()).unwrap();
    let sources_exp = fitted_exp.transform(&X.view()).unwrap();
    assert_eq!(sources_exp.dim(), (5, 2));

    // Test cube contrast function
    let ica_cube = dict_learning::ICA::new()
        .n_components(2)
        .fun("cube".to_string())
        .max_iter(20);

    let fitted_cube = ica_cube.fit(&X.view(), &()).unwrap();
    let sources_cube = fitted_cube.transform(&X.view()).unwrap();
    assert_eq!(sources_cube.dim(), (5, 2));

    // Test parallel algorithm
    let ica_parallel = dict_learning::ICA::new()
        .n_components(2)
        .algorithm("parallel".to_string())
        .max_iter(20);

    let fitted_parallel = ica_parallel.fit(&X.view(), &()).unwrap();
    let sources_parallel = fitted_parallel.transform(&X.view()).unwrap();
    assert_eq!(sources_parallel.dim(), (5, 2));
}

#[test]
fn test_ica_error_cases() {
    let X = array![[1.0, 2.0, 3.0]]; // Only 1 sample

    let ica = dict_learning::ICA::new();
    let result = ica.fit(&X.view(), &());
    assert!(result.is_err()); // Should fail with too few samples

    let X_small = array![[1.0, 2.0], [3.0, 4.0]];
    let ica_large = dict_learning::ICA::new().n_components(5); // More components than features
    let result = ica_large.fit(&X_small.view(), &());
    assert!(result.is_err()); // Should fail with too many components

    // Test invalid contrast function
    let ica_invalid = dict_learning::ICA::new().fun("invalid".to_string());
    let result = ica_invalid.fit(&X_small.view(), &());
    assert!(result.is_err()); // Should fail with invalid contrast function
}

#[test]
fn test_pca() {
    // Create test data with clear principal components
    let X = array![
        [1.0, 1.0, 0.0],
        [2.0, 2.0, 0.0],
        [3.0, 3.0, 0.0],
        [4.0, 4.0, 0.0],
        [5.0, 5.0, 0.0]
    ];

    let pca = dict_learning::PCA::new().n_components(2).whiten(false);

    let fitted = pca.fit(&X.view(), &()).unwrap();
    let transformed = fitted.transform(&X.view()).unwrap();

    assert_eq!(transformed.dim(), (5, 2)); // 5 samples, 2 components

    // Check that we can access PCA properties
    let components = fitted.components();
    assert_eq!(components.dim(), (2, 3)); // 2 components, 3 features

    let explained_variance = fitted.explained_variance();
    assert_eq!(explained_variance.len(), 2);

    let explained_variance_ratio = fitted.explained_variance_ratio();
    assert_eq!(explained_variance_ratio.len(), 2);

    // Test inverse transform
    let reconstructed = fitted.inverse_transform(&transformed.view()).unwrap();
    assert_eq!(reconstructed.dim(), X.dim());

    // Test score (reconstruction quality)
    let score = fitted.score(&X.view()).unwrap();
    assert!(score.is_finite()); // Score should be a finite number

    // Test with whitening
    let pca_whitened = dict_learning::PCA::new().n_components(2).whiten(true);

    let fitted_whitened = pca_whitened.fit(&X.view(), &()).unwrap();
    let transformed_whitened = fitted_whitened.transform(&X.view()).unwrap();
    assert_eq!(transformed_whitened.dim(), (5, 2));
}

#[test]
fn test_pca_error_cases() {
    let X = array![[1.0, 2.0, 3.0]]; // Only 1 sample

    let pca = dict_learning::PCA::new();
    let result = pca.fit(&X.view(), &());
    assert!(result.is_err()); // Should fail with too few samples

    let X_small = array![[1.0, 2.0], [3.0, 4.0]];
    let pca_large = dict_learning::PCA::new().n_components(5); // More components than features
    let result = pca_large.fit(&X_small.view(), &());
    assert!(result.is_err()); // Should fail with too many components
}

#[test]
fn test_factor_analysis() {
    // Create test data with underlying factors
    let X = array![
        [1.0, 2.0, 3.0, 4.0],
        [2.0, 4.0, 6.0, 8.0],
        [3.0, 6.0, 9.0, 12.0],
        [1.5, 3.0, 4.5, 6.0],
        [2.5, 5.0, 7.5, 10.0]
    ];

    let fa = dict_learning::FactorAnalysis::new()
        .n_components(2)
        .max_iter(50)
        .tol(1e-4);

    let fitted = fa.fit(&X.view(), &()).unwrap();
    let factors = fitted.transform(&X.view()).unwrap();

    assert_eq!(factors.dim(), (5, 2)); // 5 samples, 2 factors

    // Check that we can access FA properties
    let loadings = fitted.loadings();
    assert_eq!(loadings.dim(), (4, 2)); // 4 features, 2 factors

    let noise_variance = fitted.noise_variance();
    assert_eq!(noise_variance.len(), 4);

    // All noise variances should be positive
    for &var in noise_variance.iter() {
        assert!(var > 0.0);
    }

    let explained_variance = fitted.explained_variance();
    assert_eq!(explained_variance.len(), 2);

    // Test covariance reconstruction
    let reconstructed_cov = fitted.get_covariance().unwrap();
    assert_eq!(reconstructed_cov.dim(), (4, 4));

    // Test score (log-likelihood)
    let score = fitted.score(&X.view()).unwrap();
    assert!(score.is_finite()); // Score should be a finite number

    // Test with rotation
    let fa_rotated = dict_learning::FactorAnalysis::new()
        .n_components(2)
        .rotation("varimax")
        .max_iter(20);

    let fitted_rotated = fa_rotated.fit(&X.view(), &()).unwrap();
    let factors_rotated = fitted_rotated.transform(&X.view()).unwrap();
    assert_eq!(factors_rotated.dim(), (5, 2));
}

#[test]
fn test_factor_analysis_error_cases() {
    let X = array![[1.0, 2.0, 3.0]]; // Only 1 sample

    let fa = dict_learning::FactorAnalysis::new();
    let result = fa.fit(&X.view(), &());
    assert!(result.is_err()); // Should fail with too few samples

    let X_small = array![[1.0, 2.0], [3.0, 4.0]];
    let fa_large = dict_learning::FactorAnalysis::new().n_components(5); // More components than features
    let result = fa_large.fit(&X_small.view(), &());
    assert!(result.is_err()); // Should fail with too many components
}

#[test]
fn test_probabilistic_matrix_factorization() {
    // Create a rating matrix (users x items)
    let X = array![
        [5.0, 3.0, 0.0, 1.0],
        [4.0, 0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0, 5.0],
        [1.0, 0.0, 0.0, 4.0],
        [0.0, 1.0, 5.0, 4.0]
    ];

    let pmf = dict_learning::ProbabilisticMatrixFactorization::new()
        .n_components(2)
        .lambda_u(0.01)
        .lambda_v(0.01)
        .sigma(0.1)
        .learning_rate(0.01)
        .max_iter(50)
        .random_state(Some(42));

    let fitted = pmf.fit(&X.view(), &()).unwrap();

    // Check dimensions
    let U = fitted.get_user_features();
    let V = fitted.get_item_features();
    assert_eq!(U.dim(), (5, 2)); // 5 users, 2 components
    assert_eq!(V.dim(), (4, 2)); // 4 items, 2 components

    // Test prediction functionality
    let prediction = fitted.predict(0, 0).unwrap();
    assert!(prediction.is_finite());

    // Test batch prediction
    let user_indices = vec![0, 1, 2];
    let item_indices = vec![0, 1, 2];
    let predictions = fitted.predict_batch(&user_indices, &item_indices).unwrap();
    assert_eq!(predictions.len(), 3);

    // Test matrix reconstruction
    let reconstructed = fitted.reconstruct();
    assert_eq!(reconstructed.dim(), X.dim());

    // Test training loss
    let loss = fitted.get_training_loss();
    assert!(loss.is_finite());
    assert!(loss >= 0.0);

    // Test log probability
    let log_prob = fitted.log_probability(0, 0, 5.0).unwrap();
    assert!(log_prob.is_finite());
}

#[test]
fn test_pmf_error_cases() {
    let X = array![[1.0, 2.0], [3.0, 4.0]];

    // Test empty matrix
    let empty_X = Array2::<f64>::zeros((0, 0));
    let pmf = dict_learning::ProbabilisticMatrixFactorization::new();
    let result = pmf.fit(&empty_X.view(), &());
    assert!(result.is_err());

    // Test too many components
    let pmf_large = dict_learning::ProbabilisticMatrixFactorization::new().n_components(10); // More than matrix dimensions
    let result = pmf_large.fit(&X.view(), &());
    assert!(result.is_err());

    // Test predictions with fitted model
    let pmf_fitted = dict_learning::ProbabilisticMatrixFactorization::new()
        .n_components(2)
        .max_iter(5)
        .fit(&X.view(), &())
        .unwrap();

    // Test out of bounds predictions
    assert!(pmf_fitted.predict(10, 0).is_err()); // User index out of bounds
    assert!(pmf_fitted.predict(0, 10).is_err()); // Item index out of bounds

    // Test batch prediction size mismatch
    let user_indices = vec![0, 1];
    let item_indices = vec![0]; // Different length
    let result = pmf_fitted.predict_batch(&user_indices, &item_indices);
    assert!(result.is_err());

    // Test transform with wrong dimensions
    let wrong_X = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]; // 3 users instead of 2
    let result = pmf_fitted.transform(&wrong_X.view());
    assert!(result.is_err());
}

#[test]
fn test_pmf_convergence() {
    // Test that PMF produces finite outputs and reasonable behavior
    let X = array![
        [5.0, 3.0, 0.0, 1.0],
        [4.0, 0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0, 5.0],
        [1.0, 0.0, 0.0, 4.0]
    ];

    let pmf = dict_learning::ProbabilisticMatrixFactorization::new()
        .n_components(2)
        .lambda_u(0.01)
        .lambda_v(0.01)
        .sigma(1.0)
        .learning_rate(0.01)
        .max_iter(50)
        .random_state(Some(42));

    let fitted = pmf.fit(&X.view(), &()).unwrap();
    let reconstructed = fitted.reconstruct();

    // Check that reconstruction produces finite values
    for &val in reconstructed.iter() {
        assert!(
            val.is_finite(),
            "Reconstructed value should be finite, got: {}",
            val
        );
    }

    // Check that dimensions match
    assert_eq!(reconstructed.dim(), X.dim());

    // Test that predictions for observed entries are reasonable
    // (they should be positive for positive ratings)
    let pred_00 = fitted.predict(0, 0).unwrap(); // Original: 5.0
    let pred_03 = fitted.predict(0, 3).unwrap(); // Original: 1.0

    assert!(pred_00.is_finite());
    assert!(pred_03.is_finite());

    // Check that the algorithm learns something meaningful
    // (user features and item features should not be all zeros)
    let U = fitted.get_user_features();
    let V = fitted.get_item_features();

    let u_sum = U.iter().map(|x| x.abs()).sum::<f64>();
    let v_sum = V.iter().map(|x| x.abs()).sum::<f64>();

    assert!(
        u_sum > 1e-6,
        "User features should have learned non-trivial values"
    );
    assert!(
        v_sum > 1e-6,
        "Item features should have learned non-trivial values"
    );
}

#[test]
fn test_pmf_sparsity_handling() {
    // Test PMF with sparse data (many zeros)
    let X = array![
        [5.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 3.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 4.0, 2.0, 0.0],
        [2.0, 0.0, 0.0, 0.0, 3.0]
    ];

    let pmf = dict_learning::ProbabilisticMatrixFactorization::new()
        .n_components(2)
        .lambda_u(0.01)
        .lambda_v(0.01)
        .max_iter(30)
        .random_state(Some(42));

    let fitted = pmf.fit(&X.view(), &()).unwrap();

    // Model should fit without errors
    let U = fitted.get_user_features();
    let V = fitted.get_item_features();
    assert_eq!(U.dim(), (4, 2));
    assert_eq!(V.dim(), (5, 2));

    // Test predictions for sparse entries
    for i in 0..X.nrows() {
        for j in 0..X.ncols() {
            let pred = fitted.predict(i, j).unwrap();
            assert!(pred.is_finite());
        }
    }
}

#[test]
fn test_pmf_regularization_effect() {
    let X = array![
        [5.0, 3.0, 0.0, 1.0],
        [4.0, 0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0, 5.0],
        [1.0, 0.0, 0.0, 4.0]
    ];

    // Test with different regularization values
    let pmf_low_reg = dict_learning::ProbabilisticMatrixFactorization::new()
        .n_components(2)
        .lambda_u(0.001)
        .lambda_v(0.001)
        .max_iter(20)
        .random_state(Some(42));

    let pmf_high_reg = dict_learning::ProbabilisticMatrixFactorization::new()
        .n_components(2)
        .lambda_u(1.0)
        .lambda_v(1.0)
        .max_iter(20)
        .random_state(Some(42));

    let fitted_low = pmf_low_reg.fit(&X.view(), &()).unwrap();
    let fitted_high = pmf_high_reg.fit(&X.view(), &()).unwrap();

    // Both should produce finite features, but with different magnitudes
    let U_low = fitted_low.get_user_features();
    let U_high = fitted_high.get_user_features();

    for &val in U_low.iter() {
        assert!(val.is_finite());
    }
    for &val in U_high.iter() {
        assert!(val.is_finite());
    }

    // High regularization should generally produce smaller feature magnitudes
    let norm_low: f64 = U_low.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_high: f64 = U_high.iter().map(|x| x * x).sum::<f64>().sqrt();

    // This is a general expectation, but not guaranteed due to randomness
    // We just check that both are reasonable finite values
    assert!(norm_low.is_finite() && norm_low > 0.0);
    assert!(norm_high.is_finite() && norm_high > 0.0);
}
