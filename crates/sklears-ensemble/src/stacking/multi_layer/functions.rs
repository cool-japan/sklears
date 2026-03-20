//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::super::config::{
    MetaFeatureStrategy, MetaLearningStrategy, MultiLayerStackingConfig, StackingLayerConfig,
};

use super::types::MultiLayerStackingClassifier;

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{array, Array1};
    use sklears_core::traits::Fit;
    use sklears_core::types::Float;
    #[test]
    fn test_multi_layer_stacking_creation() {
        let config = MultiLayerStackingConfig::new();
        let stacking = MultiLayerStackingClassifier::new(config);
        assert!(stacking.layers_.is_none());
        assert!(stacking.final_meta_weights_.is_none());
        assert!(stacking.classes_.is_none());
    }
    #[test]
    fn test_two_layer_creation() {
        let stacking = MultiLayerStackingClassifier::two_layer(3, 2);
        assert_eq!(stacking.config.layers.len(), 2);
        assert_eq!(stacking.config.layers[0].n_estimators, 3);
        assert_eq!(stacking.config.layers[1].n_estimators, 2);
        assert!(stacking.config.enable_pruning);
        assert!(stacking.config.confidence_weighting);
    }
    #[test]
    fn test_deep_stacking_creation() {
        let stacking = MultiLayerStackingClassifier::deep(3, 5);
        assert_eq!(stacking.config.layers.len(), 3);
        assert!(stacking
            .config
            .layers
            .iter()
            .all(|layer| layer.n_estimators == 5));
        assert!(stacking.config.enable_pruning);
        assert_eq!(stacking.config.diversity_threshold, 0.15);
    }
    #[test]
    fn test_advanced_meta_features() {
        let predictions = array![[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]];
        let config = MultiLayerStackingConfig::new();
        let stacking = MultiLayerStackingClassifier::new(config);
        let statistical = stacking
            .generate_statistical_features(&predictions)
            .expect("operation should succeed");
        assert_eq!(statistical.dim(), (3, 7));
        let interactions = stacking
            .generate_interaction_features(&predictions)
            .expect("operation should succeed");
        assert_eq!(interactions.dim(), (3, 6));
        let confidence = stacking
            .generate_confidence_features(&predictions)
            .expect("operation should succeed");
        assert_eq!(confidence.dim(), (3, 6));
    }
    #[test]
    fn test_diversity_calculation() {
        let predictions = array![[0.1, 0.9, 0.5], [0.2, 0.8, 0.4], [0.3, 0.7, 0.6]];
        let config = MultiLayerStackingConfig::new();
        let stacking = MultiLayerStackingClassifier::new(config);
        let diversity_scores = stacking.calculate_diversity_scores(&predictions);
        assert_eq!(diversity_scores.len(), 3);
        assert!(diversity_scores.iter().all(|&score| score >= 0.0));
    }
    #[test]
    fn test_ensemble_pruning() {
        let diversity_scores = array![0.1, 0.5, 0.3, 0.8, 0.2];
        let threshold = 0.25;
        let config = MultiLayerStackingConfig::new();
        let stacking = MultiLayerStackingClassifier::new(config);
        let pruned_indices = stacking.prune_ensemble(&diversity_scores, threshold);
        let expected = vec![1, 2, 3];
        assert_eq!(pruned_indices, expected);
    }
    #[test]
    fn test_spectral_features() {
        let predictions = array![
            [0.1, 0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7, 0.8],
            [0.9, 0.1, 0.2, 0.3]
        ];
        let config = MultiLayerStackingConfig::new();
        let stacking = MultiLayerStackingClassifier::new(config);
        let spectral = stacking
            .generate_spectral_features(&predictions)
            .expect("operation should succeed");
        assert_eq!(spectral.dim(), (3, 8));
    }
    #[test]
    fn test_information_theoretic_features() {
        let predictions = array![
            [0.1, 0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7, 0.8],
            [0.9, 0.1, 0.2, 0.3]
        ];
        let config = MultiLayerStackingConfig::new();
        let stacking = MultiLayerStackingClassifier::new(config);
        let info_theoretic = stacking
            .generate_information_theoretic_features(&predictions)
            .expect("operation should succeed");
        assert_eq!(info_theoretic.dim(), (3, 9));
    }
    #[test]
    fn test_neural_embedding_features() {
        let predictions = array![[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]];
        let config = MultiLayerStackingConfig::new();
        let stacking = MultiLayerStackingClassifier::new(config);
        let neural_embedding = stacking
            .generate_neural_embedding_features(&predictions)
            .expect("operation should succeed");
        assert_eq!(neural_embedding.dim(), (3, 11));
    }
    #[test]
    fn test_kernel_features() {
        let predictions = array![[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]];
        let config = MultiLayerStackingConfig::new();
        let stacking = MultiLayerStackingClassifier::new(config);
        let kernel_features = stacking
            .generate_kernel_features(&predictions)
            .expect("operation should succeed");
        assert_eq!(kernel_features.dim(), (3, 6));
    }
    #[test]
    fn test_basis_expansion_features() {
        let predictions = array![[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]];
        let config = MultiLayerStackingConfig::new();
        let stacking = MultiLayerStackingClassifier::new(config);
        let basis_expansion = stacking
            .generate_basis_expansion_features(&predictions)
            .expect("operation should succeed");
        assert_eq!(basis_expansion.dim(), (3, 9));
    }
    #[test]
    fn test_meta_learning_features() {
        let predictions = array![[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]];
        let config = MultiLayerStackingConfig::new();
        let stacking = MultiLayerStackingClassifier::new(config);
        let meta_learning = stacking
            .generate_meta_learning_features(&predictions)
            .expect("operation should succeed");
        assert_eq!(meta_learning.dim(), (3, 7));
    }
    #[test]
    fn test_comprehensive_features() {
        let predictions = array![[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]];
        let config = MultiLayerStackingConfig::new();
        let stacking = MultiLayerStackingClassifier::new(config);
        let comprehensive = stacking
            .generate_comprehensive_features(&predictions)
            .expect("operation should succeed");
        let expected_features = 3 + 4 + 3 + 3 + 2;
        assert_eq!(comprehensive.ncols(), expected_features);
    }
    #[test]
    fn test_regression_methods() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
        let y = array![1.0, 2.0, 3.0, 4.0];
        let config = MultiLayerStackingConfig::new();
        let stacking = MultiLayerStackingClassifier::new(config);
        let (weights, intercept): (Array1<Float>, Float) = stacking
            .train_linear_regression(&x, &y, 0.0)
            .expect("operation should succeed");
        assert_eq!(weights.len(), 2);
        assert!(intercept.is_finite());
        let (ridge_weights, ridge_intercept): (Array1<Float>, Float) = stacking
            .train_linear_regression(&x, &y, 0.1)
            .expect("operation should succeed");
        assert_eq!(ridge_weights.len(), 2);
        assert!(ridge_intercept.is_finite());
        let (lasso_weights, lasso_intercept): (Array1<Float>, Float) = stacking
            .train_lasso_regression(&x, &y, 0.1)
            .expect("operation should succeed");
        assert_eq!(lasso_weights.len(), 2);
        assert!(lasso_intercept.is_finite());
        let (en_weights, en_intercept): (Array1<Float>, Float) = stacking
            .train_elastic_net_regression(&x, &y, 0.1, 0.1)
            .expect("operation should succeed");
        assert_eq!(en_weights.len(), 2);
        assert!(en_intercept.is_finite());
    }
    #[test]
    fn test_insufficient_samples() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![0, 1];
        let stacking = MultiLayerStackingClassifier::two_layer(2, 1);
        let result = stacking.fit(&x, &y);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("at least 20 samples"));
    }
    #[test]
    fn test_insufficient_classes() {
        let x = array![
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
            [9.0, 10.0],
            [11.0, 12.0],
            [13.0, 14.0],
            [15.0, 16.0],
            [17.0, 18.0],
            [19.0, 20.0],
            [21.0, 22.0],
            [23.0, 24.0],
            [25.0, 26.0],
            [27.0, 28.0],
            [29.0, 30.0],
            [31.0, 32.0],
            [33.0, 34.0],
            [35.0, 36.0],
            [37.0, 38.0],
            [39.0, 40.0]
        ];
        let y = array![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let stacking = MultiLayerStackingClassifier::two_layer(2, 1);
        let result = stacking.fit(&x, &y);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("at least 2 classes"));
    }
    #[test]
    fn test_shape_mismatch() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![0];
        let stacking = MultiLayerStackingClassifier::two_layer(1, 1);
        let result = stacking.fit(&x, &y);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Shape mismatch"));
    }
    #[test]
    fn test_deep_stacking_configuration() {
        let config = MultiLayerStackingConfig::deep_stacking(3, 4);
        assert_eq!(config.layers.len(), 3);
        assert_eq!(
            config.final_meta_strategy,
            MetaLearningStrategy::BayesianAveraging
        );
        assert!(config.enable_pruning);
        assert_eq!(config.diversity_threshold, 0.15);
        assert!(config.confidence_weighting);
        assert!(!config.layers[0].use_probabilities);
        assert!(config.layers[1].use_probabilities);
        assert!(config.layers[2].use_probabilities);
        assert!(config.layers[0].passthrough);
        assert!(!config.layers[1].passthrough);
        assert!(!config.layers[2].passthrough);
    }
    #[test]
    fn test_random_state_setting() {
        let stacking = MultiLayerStackingClassifier::two_layer(2, 1).random_state(42);
        assert_eq!(stacking.config.random_state, Some(42));
    }
    #[test]
    fn test_utility_methods() {
        let config = MultiLayerStackingConfig::new();
        let stacking = MultiLayerStackingClassifier::new(config);
        let x = array![1.0, 2.0, 3.0, 4.0];
        let y = array![2.0, 4.0, 6.0, 8.0];
        let corr = stacking.calculate_correlation(&x, &y);
        assert!((corr - 1.0).abs() < 1e-10);
        let values = array![0.5, 0.3, 0.2];
        let entropy = stacking.calculate_shannon_entropy(&values);
        assert!(entropy > 0.0);
        let signal = array![1.0, 2.0, 1.0, 2.0, 1.0];
        let dominant_freq = stacking.find_dominant_frequency(&signal);
        assert!(dominant_freq >= 0.0);
        let centroid = stacking.calculate_spectral_centroid(&signal);
        assert!(centroid > 0.0);
    }
}
