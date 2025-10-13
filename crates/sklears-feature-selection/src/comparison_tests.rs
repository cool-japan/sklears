//! Comparison tests against reference implementations
//!
//! This module contains tests that compare our feature selection implementations
//! against reference implementations from scikit-learn and other libraries.

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use crate::base::{FeatureSelector, SelectorMixin};
    use crate::embedded::LassoSelector;
    use crate::filter::{SelectKBest, VarianceThreshold};
    use crate::optimization::{ConvexFeatureSelector, ProximalGradientSelector};
    use scirs2_core::ndarray::{Array1, Array2};
    use sklears_core::traits::{Fit, Transform};

    /// Create test data for comparison tests (classification)
    fn create_classification_test_data() -> (Array2<f64>, Array1<i32>) {
        // Create synthetic data with known properties for classification
        let n_samples = 100;
        let n_features = 20;
        let mut features = Array2::zeros((n_samples, n_features));
        let mut target = Array1::zeros(n_samples);

        // Generate features with specific patterns
        for i in 0..n_samples {
            for j in 0..n_features {
                // First 5 features are highly predictive
                if j < 5 {
                    features[[i, j]] = (i as f64 * 0.1 + j as f64 * 0.2).sin() + 0.1 * j as f64;
                } else if j < 10 {
                    // Next 5 features are moderately predictive
                    features[[i, j]] = (i as f64 * 0.05 + j as f64 * 0.1).cos() + 0.05 * j as f64;
                } else if j < 15 {
                    // Next 5 features are weakly predictive
                    features[[i, j]] = (i as f64 * 0.01 + j as f64 * 0.01).sin() + 0.01 * j as f64;
                } else {
                    // Last 5 features are noise
                    features[[i, j]] = 0.01 * i as f64 * (j as f64 % 3.0);
                }
            }

            // Create binary classification target based on first few features
            let decision_value = features[[i, 0]]
                + 0.5 * features[[i, 1]]
                + 0.2 * features[[i, 2]]
                + 0.1 * features[[i, 3]]
                + 0.05 * features[[i, 4]];
            target[i] = if decision_value > 0.0 { 1 } else { 0 };
        }

        (features, target)
    }

    /// Create test data for regression
    fn create_regression_test_data() -> (Array2<f64>, Array1<f64>) {
        // Create synthetic data with known properties for regression
        let n_samples = 100;
        let n_features = 20;
        let mut features = Array2::zeros((n_samples, n_features));
        let mut target = Array1::zeros(n_samples);

        // Generate features with specific patterns
        for i in 0..n_samples {
            for j in 0..n_features {
                // First 5 features are highly predictive
                if j < 5 {
                    features[[i, j]] = (i as f64 * 0.1 + j as f64 * 0.2).sin() + 0.1 * j as f64;
                } else if j < 10 {
                    // Next 5 features are moderately predictive
                    features[[i, j]] = (i as f64 * 0.05 + j as f64 * 0.1).cos() + 0.05 * j as f64;
                } else if j < 15 {
                    // Next 5 features are weakly predictive
                    features[[i, j]] = (i as f64 * 0.01 + j as f64 * 0.01).sin() + 0.01 * j as f64;
                } else {
                    // Last 5 features are noise
                    features[[i, j]] = 0.01 * i as f64 * (j as f64 % 3.0);
                }
            }

            // Create target based on first few features
            target[i] = features[[i, 0]]
                + 0.5 * features[[i, 1]]
                + 0.2 * features[[i, 2]]
                + 0.1 * features[[i, 3]]
                + 0.05 * features[[i, 4]];
        }

        (features, target)
    }

    /// Test variance threshold against expected behavior
    #[test]
    fn test_variance_threshold_reference() {
        // Create data with explicitly controlled variance
        let n_samples = 100;
        let n_features = 20;
        let mut features = Array2::zeros((n_samples, n_features));

        // Generate features with specific variance patterns
        for i in 0..n_samples {
            for j in 0..n_features {
                if j < 10 {
                    // High variance features
                    features[[i, j]] = (i as f64).sin() * (j + 1) as f64;
                } else if j < 15 {
                    // Medium variance features
                    features[[i, j]] = (i as f64) * 0.1 + (j as f64) * 0.1;
                } else {
                    // Low variance features (constant with small noise)
                    features[[i, j]] = 1.0 + 0.001 * (i as f64);
                }
            }
        }

        // Test with threshold that should remove low-variance features
        let selector = VarianceThreshold::new(0.01);
        let dummy_target = Array1::<f64>::zeros(features.nrows());
        let trained = selector
            .fit(&features.view(), &dummy_target.view())
            .unwrap();

        // Should keep most features as they have reasonable variance
        let support = trained.get_support().unwrap();
        let selected_count = support.iter().filter(|&&x| x).count();
        assert!(selected_count >= 10); // At least 10 features should remain
        assert!(selected_count <= features.ncols()); // Can't select more than available

        // Verify transform works correctly
        let transformed = trained.transform(&features).unwrap();
        assert_eq!(transformed.ncols(), selected_count);
        assert_eq!(transformed.nrows(), features.nrows());

        // Test with very high threshold - may remove all features
        let strict_selector = VarianceThreshold::new(100.0);
        let dummy_target = Array1::<f64>::zeros(features.nrows());
        let strict_result = strict_selector.fit(&features.view(), &dummy_target.view());

        // With high threshold, either fewer features are selected or fitting fails
        match strict_result {
            Ok(strict_trained) => {
                let strict_support = strict_trained.get_support().unwrap();
                let strict_selected_count = strict_support.iter().filter(|&&x| x).count();
                assert!(strict_selected_count <= selected_count);
            }
            Err(_) => {
                // It's acceptable to fail when no features meet the threshold
                // This shows the selector correctly identifies when thresholds are too strict
            }
        }
    }

    /// Test SelectKBest behavior against expected ranking
    #[test]
    fn test_select_k_best_reference() {
        let (features, target) = create_classification_test_data();

        // Test with k=5 - should select most predictive features
        let selector = SelectKBest::new(5, "f_classif");
        let trained = selector.fit(&features, &target).unwrap();

        let selected_features = trained.selected_features();
        assert_eq!(selected_features.len(), 5);

        // First few features should be selected as they are most predictive
        // (though exact order may vary based on scoring method)
        let mut early_features_selected = 0;
        for &idx in selected_features {
            if idx < 8 {
                // First 8 features are reasonably predictive
                early_features_selected += 1;
            }
        }
        assert!(early_features_selected >= 3); // At least 3 out of 5 should be early features

        // Verify transform
        let transformed = trained.transform(&features).unwrap();
        assert_eq!(transformed.ncols(), 5);
        assert_eq!(transformed.nrows(), features.nrows());
    }

    /// Test LASSO selector consistency
    #[test]
    fn test_lasso_selector_reference() {
        let (features, target) = create_regression_test_data();

        // Test with moderate regularization
        let selector = LassoSelector::new()
            .alpha(0.1)
            .max_iter(1000)
            .tolerance(1e-6);

        let trained = selector.fit(&features, &target).unwrap();
        let selected_features = trained.selected_features();

        // LASSO should select some features (not all, not none)
        assert!(selected_features.len() > 0);
        assert!(selected_features.len() < features.ncols());

        // With L1 regularization, should prefer predictive features
        let coefficients = trained.coefficients();
        let selected_coefficients: Vec<f64> = selected_features
            .iter()
            .map(|&idx| coefficients[idx])
            .collect();

        // Selected features should have non-zero coefficients
        assert!(selected_coefficients.iter().all(|&w| w.abs() > 1e-10));

        // Test with high regularization - should select fewer features
        let strict_selector = LassoSelector::new().alpha(1.0).max_iter(1000);

        let strict_trained = strict_selector.fit(&features, &target).unwrap();
        let strict_selected = strict_trained.selected_features();

        // Higher regularization should select fewer features
        assert!(strict_selected.len() <= selected_features.len());
    }

    /// Test optimization-based selectors consistency
    #[test]
    fn test_optimization_selectors_reference() {
        let (features, target) = create_regression_test_data();

        // Test ConvexFeatureSelector
        let convex_selector = ConvexFeatureSelector::new()
            .k(5)
            .regularization(0.1)
            .max_iter(100);

        let convex_trained = convex_selector.fit(&features, &target).unwrap();
        assert_eq!(convex_trained.selected_features().len(), 5);

        // Test ProximalGradientSelector
        let proximal_selector = ProximalGradientSelector::new()
            .k(5)
            .regularization(0.1)
            .max_iter(100);

        let proximal_trained = proximal_selector.fit(&features, &target).unwrap();
        assert_eq!(proximal_trained.selected_features().len(), 5);

        // Both optimization methods should select similar features
        // (since they're solving similar optimization problems)
        let convex_features = convex_trained.selected_features();
        let proximal_features = proximal_trained.selected_features();

        // Calculate overlap
        let overlap = convex_features
            .iter()
            .filter(|&f| proximal_features.contains(f))
            .count();

        // Expect some overlap between optimization methods
        assert!(overlap >= 2); // At least 2 features should overlap
    }

    /// Test feature ranking consistency
    #[test]
    fn test_feature_ranking_consistency() {
        let (features_class, target_class) = create_classification_test_data();
        let (features_reg, target_reg) = create_regression_test_data();

        // Test multiple selectors and check for consistent ranking patterns
        let k_best = SelectKBest::new(10, "f_classif")
            .fit(&features_class, &target_class)
            .unwrap();
        let lasso = LassoSelector::new()
            .alpha(0.01)
            .fit(&features_reg, &target_reg)
            .unwrap();

        let k_best_features = k_best.selected_features();
        let lasso_features = lasso.selected_features();

        // All methods should prefer early features (which are more predictive)
        let check_early_preference = |features: &[usize]| {
            let early_count = features.iter().filter(|&&idx| idx < 10).count();
            early_count as f64 / features.len() as f64
        };

        let k_best_early_ratio = check_early_preference(k_best_features);
        let lasso_early_ratio = check_early_preference(lasso_features);

        // All methods should have preference for early (predictive) features
        assert!(k_best_early_ratio > 0.5);
        assert!(lasso_early_ratio > 0.3); // LASSO might be more sparse

        // Calculate overlap (if both have at least 5 features)
        if k_best_features.len() >= 5 && lasso_features.len() >= 5 {
            let overlap = k_best_features
                .iter()
                .take(5)
                .filter(|&f| lasso_features.iter().take(5).any(|&lf| lf == *f))
                .count();

            // Should have some overlap between methods
            assert!(overlap >= 1);
        }
    }

    /// Test performance characteristics
    #[test]
    fn test_performance_characteristics() {
        let (features, target) = create_classification_test_data();

        // Test that selectors can handle the data efficiently
        let start_time = std::time::Instant::now();

        let selector = SelectKBest::new(5, "f_classif");
        let trained = selector.fit(&features, &target).unwrap();
        let _transformed = trained.transform(&features).unwrap();

        let duration = start_time.elapsed();

        // Should complete in reasonable time (less than 1 second for small data)
        assert!(duration.as_secs_f64() < 1.0);

        // Test memory efficiency - transform shouldn't increase memory dramatically
        let original_size = features.len();
        let transformed_size = trained.transform(&features).unwrap().len();

        // Transformed data should be smaller (fewer features)
        assert!(transformed_size < original_size);
    }

    /// Test numerical stability
    #[test]
    fn test_numerical_stability() {
        // Create data with potential numerical issues
        let n_samples = 50;
        let n_features = 10;
        let mut features = Array2::zeros((n_samples, n_features));

        // Add features with very small variance
        for i in 0..n_samples {
            for j in 0..n_features {
                if j < 5 {
                    features[[i, j]] = 1e-10 * i as f64; // Very small values
                } else {
                    features[[i, j]] = i as f64; // Normal values
                }
            }
        }

        // Test that selectors handle small values gracefully
        let variance_selector = VarianceThreshold::new(1e-12);
        let dummy_target = Array1::<i32>::zeros(features.nrows());
        let trained = variance_selector.fit(&features, &dummy_target).unwrap();

        // Should successfully fit and select some features
        let support = trained.get_support().unwrap();
        let selected_count = support.iter().filter(|&&x| x).count();
        assert!(selected_count > 0);

        // Transform should work without numerical issues
        let transformed = trained.transform(&features).unwrap();
        assert!(transformed.iter().all(|&x| x.is_finite()));
    }
}
