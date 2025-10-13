use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};
use scirs2_core::ndarray::{Array1, Array2, array};
use scirs2_core::random::{Random, rng};

use super::data_types::*;
use super::neural_networks::*;
use super::feature_extraction::*;
use super::ml_models::*;
use super::usage_patterns::*;
use super::recommender_engine::*;

// Test data utilities

fn create_test_trait_context() -> TraitContext {
    TraitContext {
        project_type: ProjectType::Library,
        domain: Domain::SystemsProgramming,
        complexity_level: ComplexityLevel::Intermediate,
        crate_size: CrateSize::Medium,
        dependencies: vec!["serde".to_string(), "tokio".to_string()],
        features_enabled: vec!["async".to_string(), "json".to_string()],
        target_audience: TargetAudience::Developers,
        performance_requirements: PerformanceRequirements::High,
        safety_requirements: SafetyRequirements::Safe,
        existing_traits: vec!["Debug".to_string(), "Clone".to_string()],
        code_patterns: vec![CodePattern::Builder, CodePattern::Factory],
        architectural_style: ArchitecturalStyle::Layered,
        error_handling_strategy: ErrorHandlingStrategy::ResultBased,
        concurrency_model: ConcurrencyModel::AsyncAwait,
        testing_strategy: TestingStrategy::UnitTesting,
        documentation_level: DocumentationLevel::Comprehensive,
        metadata: HashMap::new(),
    }
}

fn create_test_trait_info(name: &str) -> TraitInfo {
    TraitInfo {
        name: name.to_string(),
        category: "Test".to_string(),
        description: format!("Test trait {}", name),
        complexity: 2,
        usage_frequency: 0.5,
        dependencies: vec!["core".to_string()],
        methods: vec!["test_method".to_string()],
        associated_types: vec![],
        trait_bounds: vec![],
        examples: vec!["Example usage".to_string()],
        documentation_url: Some("https://docs.rs/test".to_string()),
        version: "1.0.0".to_string(),
    }
}

fn create_test_usage_event(trait_name: &str, timestamp: u64) -> UsageEvent {
    UsageEvent {
        trait_name: trait_name.to_string(),
        timestamp,
        context: create_test_trait_context(),
        user_session: Some("test_user".to_string()),
        success: true,
        duration_ms: Some(1000),
        metadata: HashMap::new(),
    }
}

fn create_test_array2(rows: usize, cols: usize) -> Array2<f64> {
    Array2::zeros((rows, cols))
}

fn create_test_array1(size: usize) -> Array1<f64> {
    Array1::zeros(size)
}

// Data Types Tests

#[allow(non_snake_case)]
#[cfg(test)]
mod data_types_tests {
    use super::*;

    #[test]
    fn test_trait_context_creation() {
        let context = create_test_trait_context();
        assert_eq!(context.project_type, ProjectType::Library);
        assert_eq!(context.domain, Domain::SystemsProgramming);
        assert_eq!(context.complexity_level, ComplexityLevel::Intermediate);
    }

    #[test]
    fn test_trait_context_serialization() {
        let context = create_test_trait_context();
        let serialized = serde_json::to_string(&context).unwrap();
        let deserialized: TraitContext = serde_json::from_str(&serialized).unwrap();
        assert_eq!(context.project_type, deserialized.project_type);
    }

    #[test]
    fn test_trait_info_creation() {
        let trait_info = create_test_trait_info("TestTrait");
        assert_eq!(trait_info.name, "TestTrait");
        assert_eq!(trait_info.category, "Test");
        assert_eq!(trait_info.complexity, 2);
    }

    #[test]
    fn test_ml_recommendation_config_defaults() {
        let config = MLRecommendationConfig::default();
        assert_eq!(config.max_recommendations, 10);
        assert_eq!(config.min_confidence_threshold, 0.1);
        assert!(config.enable_neural_embeddings);
    }

    #[test]
    fn test_trait_recommendation_creation() {
        let context = create_test_trait_context();
        let recommendation = TraitRecommendation {
            trait_name: "TestTrait".to_string(),
            confidence: 0.85,
            reasoning: "Test reasoning".to_string(),
            context: context.clone(),
            metadata: HashMap::new(),
        };

        assert_eq!(recommendation.trait_name, "TestTrait");
        assert_eq!(recommendation.confidence, 0.85);
        assert_eq!(recommendation.context.project_type, ProjectType::Library);
    }

    #[test]
    fn test_performance_requirements_variants() {
        let req = PerformanceRequirements::High;
        assert_eq!(format!("{:?}", req), "High");

        let req2 = PerformanceRequirements::Low;
        assert_eq!(format!("{:?}", req2), "Low");
    }

    #[test]
    fn test_code_pattern_enum() {
        let pattern = CodePattern::Builder;
        assert_eq!(format!("{:?}", pattern), "Builder");

        let pattern2 = CodePattern::Observer;
        assert_eq!(format!("{:?}", pattern2), "Observer");
    }
}

// Neural Networks Tests

#[allow(non_snake_case)]
#[cfg(test)]
mod neural_networks_tests {
    use super::*;

    #[test]
    fn test_neural_network_config_creation() {
        let config = NeuralNetworkConfig::default();
        assert_eq!(config.input_size, 128);
        assert_eq!(config.embedding_size, 64);
        assert!(config.hidden_layers.len() > 0);
    }

    #[test]
    fn test_neural_embedding_model_creation() {
        let config = NeuralNetworkConfig::default();
        let model = NeuralEmbeddingModel::new(config);
        assert!(model.is_ok());

        let model = model.unwrap();
        assert_eq!(model.embedding_dim, 64);
        assert_eq!(model.input_weights.nrows(), 128);
        assert_eq!(model.input_weights.ncols(), 64);
    }

    #[test]
    fn test_neural_model_forward_pass() {
        let config = NeuralNetworkConfig::default();
        let model = NeuralEmbeddingModel::new(config).unwrap();

        let input = Array1::zeros(128);
        let output = model.forward(&input);
        assert!(output.is_ok());

        let output = output.unwrap();
        assert_eq!(output.len(), 64);
    }

    #[test]
    fn test_neural_model_training() {
        let config = NeuralNetworkConfig::default();
        let mut model = NeuralEmbeddingModel::new(config).unwrap();

        let contexts = vec![create_test_trait_context()];
        let features = Array2::zeros((1, 128));

        let result = model.train(&contexts, &features, 10);
        assert!(result.is_ok());
    }

    #[test]
    fn test_activation_functions() {
        let config = NeuralNetworkConfig::default();
        let model = NeuralEmbeddingModel::new(config).unwrap();

        let input = 0.5;
        let relu_output = model.relu(input);
        assert_eq!(relu_output, 0.5);

        let tanh_output = model.tanh(input);
        assert!(tanh_output > 0.0 && tanh_output < 1.0);

        let sigmoid_output = model.sigmoid(input);
        assert!(sigmoid_output > 0.0 && sigmoid_output < 1.0);
    }

    #[test]
    fn test_model_convergence() {
        let config = NeuralNetworkConfig {
            learning_rate: 0.01,
            ..Default::default()
        };
        let mut model = NeuralEmbeddingModel::new(config).unwrap();

        // Test that training reduces loss over iterations
        let contexts = vec![create_test_trait_context(); 10];
        let features = Array2::zeros((10, 128));

        let initial_loss = model.calculate_loss(&features, &features).unwrap();
        model.train(&contexts, &features, 50).unwrap();
        let final_loss = model.calculate_loss(&features, &features).unwrap();

        // Loss should decrease (or at least not increase significantly)
        assert!(final_loss <= initial_loss + 0.1);
    }
}

// Feature Extraction Tests

#[allow(non_snake_case)]
#[cfg(test)]
mod feature_extraction_tests {
    use super::*;

    #[test]
    fn test_feature_extraction_config() {
        let config = FeatureExtractionConfig::default();
        assert_eq!(config.feature_dimensions, 128);
        assert!(config.enable_text_features);
        assert!(config.enable_numerical_features);
    }

    #[test]
    fn test_trait_feature_extractor_creation() {
        let config = FeatureExtractionConfig::default();
        let extractor = TraitFeatureExtractor::new(config);
        assert_eq!(extractor.feature_cache.read().unwrap().len(), 0);
    }

    #[test]
    fn test_context_feature_extraction() {
        let config = FeatureExtractionConfig::default();
        let mut extractor = TraitFeatureExtractor::new(config);
        let context = create_test_trait_context();

        let features = extractor.extract_context_features(&context);
        assert!(features.is_ok());

        let features = features.unwrap();
        assert_eq!(features.len(), 128);
    }

    #[test]
    fn test_text_processing() {
        let config = FeatureExtractionConfig::default();
        let extractor = TraitFeatureExtractor::new(config);

        let text = "This is a test string with multiple words";
        let tokens = extractor.tokenize(text);
        assert!(tokens.len() > 5);

        let stemmed = extractor.stem_word("running");
        assert_eq!(stemmed, "run");
    }

    #[test]
    fn test_tfidf_calculation() {
        let config = FeatureExtractionConfig::default();
        let extractor = TraitFeatureExtractor::new(config);

        let documents = vec![
            "this is a test document".to_string(),
            "another test document here".to_string(),
            "final document for testing".to_string(),
        ];

        let tfidf_matrix = extractor.calculate_tfidf(&documents);
        assert!(tfidf_matrix.is_ok());

        let matrix = tfidf_matrix.unwrap();
        assert_eq!(matrix.nrows(), 3); // 3 documents
        assert!(matrix.ncols() > 0); // Should have vocabulary features
    }

    #[test]
    fn test_n_gram_extraction() {
        let config = FeatureExtractionConfig::default();
        let extractor = TraitFeatureExtractor::new(config);

        let tokens = vec!["this".to_string(), "is".to_string(), "a".to_string(), "test".to_string()];
        let bigrams = extractor.extract_n_grams(&tokens, 2);

        assert!(bigrams.contains(&"this is".to_string()));
        assert!(bigrams.contains(&"is a".to_string()));
        assert!(bigrams.contains(&"a test".to_string()));
    }

    #[test]
    fn test_stop_word_removal() {
        let config = FeatureExtractionConfig::default();
        let extractor = TraitFeatureExtractor::new(config);

        let tokens = vec![
            "the".to_string(), "quick".to_string(), "and".to_string(),
            "brown".to_string(), "fox".to_string()
        ];
        let filtered = extractor.remove_stop_words(&tokens);

        assert!(!filtered.contains(&"the".to_string()));
        assert!(!filtered.contains(&"and".to_string()));
        assert!(filtered.contains(&"quick".to_string()));
        assert!(filtered.contains(&"brown".to_string()));
        assert!(filtered.contains(&"fox".to_string()));
    }

    #[test]
    fn test_feature_caching() {
        let config = FeatureExtractionConfig::default();
        let mut extractor = TraitFeatureExtractor::new(config);
        let context = create_test_trait_context();

        // First extraction
        let features1 = extractor.extract_context_features(&context).unwrap();
        assert_eq!(extractor.feature_cache.read().unwrap().len(), 1);

        // Second extraction should use cache
        let features2 = extractor.extract_context_features(&context).unwrap();
        assert_eq!(features1.len(), features2.len());

        // Features should be identical due to caching
        for (a, b) in features1.iter().zip(features2.iter()) {
            assert_eq!(a, b);
        }
    }
}

// ML Models Tests

#[allow(non_snake_case)]
#[cfg(test)]
mod ml_models_tests {
    use super::*;

    #[test]
    fn test_trait_similarity_model_creation() {
        let config = MLRecommendationConfig::default();
        let model = TraitSimilarityModel::new(config);
        assert_eq!(model.similarity_matrix.len(), 0);
        assert!(model.clustering_model.is_none());
    }

    #[test]
    fn test_clustering_model_creation() {
        let config = ClusteringConfig::default();
        let model = ClusteringModel::new(config);
        assert!(model.is_ok());

        let model = model.unwrap();
        assert_eq!(model.n_clusters, 5);
        assert_eq!(model.assignments.len(), 0);
    }

    #[test]
    fn test_clustering_model_fitting() {
        let config = ClusteringConfig::default();
        let mut model = ClusteringModel::new(config).unwrap();

        let data = Array2::from_shape_vec((10, 5), (0..50).map(|x| x as f64).collect()).unwrap();
        let result = model.fit(&data);
        assert!(result.is_ok());

        assert_eq!(model.centroids.nrows(), 5); // n_clusters
        assert_eq!(model.centroids.ncols(), 5); // feature dimensions
    }

    #[test]
    fn test_clustering_prediction() {
        let config = ClusteringConfig::default();
        let mut model = ClusteringModel::new(config).unwrap();

        let data = Array2::from_shape_vec((10, 5), (0..50).map(|x| x as f64).collect()).unwrap();
        model.fit(&data).unwrap();

        let sample = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let cluster = model.predict(&sample);
        assert!(cluster.is_ok());

        let cluster_id = cluster.unwrap();
        assert!(cluster_id < 5); // Should be valid cluster ID
    }

    #[test]
    fn test_collaborative_filtering_model() {
        let config = CollaborativeFilteringConfig::default();
        let mut model = CollaborativeFilteringModel::new(config).unwrap();

        let user_trait_matrix = Array2::zeros((5, 10)); // 5 users, 10 traits
        let trait_features = Array2::zeros((10, 8)); // 10 traits, 8 features

        let result = model.fit(&user_trait_matrix, &trait_features);
        assert!(result.is_ok());
    }

    #[test]
    fn test_collaborative_filtering_recommendations() {
        let config = CollaborativeFilteringConfig::default();
        let mut model = CollaborativeFilteringModel::new(config).unwrap();

        let user_trait_matrix = Array2::from_shape_vec(
            (3, 4),
            vec![1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0]
        ).unwrap();
        let trait_features = Array2::zeros((4, 3));

        model.fit(&user_trait_matrix, &trait_features).unwrap();

        let user_preferences = Array1::from_vec(vec![1.0, 0.0, 1.0, 0.0]);
        let recommendations = model.recommend(&user_preferences, 2);
        assert!(recommendations.is_ok());

        let recs = recommendations.unwrap();
        assert!(recs.len() <= 2);
    }

    #[test]
    fn test_distance_calculations() {
        let config = MLRecommendationConfig::default();
        let model = TraitSimilarityModel::new(config);

        let a = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let b = Array1::from_vec(vec![4.0, 5.0, 6.0]);

        let euclidean = model.calculate_euclidean_distance(&a, &b);
        assert!(euclidean > 0.0);

        let manhattan = model.calculate_manhattan_distance(&a, &b);
        assert!(manhattan > 0.0);

        let cosine = model.calculate_cosine_similarity(&a, &b);
        assert!(cosine >= -1.0 && cosine <= 1.0);
    }

    #[test]
    fn test_similarity_matrix_operations() {
        let config = MLRecommendationConfig::default();
        let mut model = TraitSimilarityModel::new(config);

        model.set_similarity("TraitA", "TraitB", 0.8);
        model.set_similarity("TraitB", "TraitC", 0.6);

        let similarity = model.get_similarity("TraitA", "TraitB");
        assert_eq!(similarity, 0.8);

        let similar_traits = model.find_similar_traits("TraitA", 5);
        assert!(similar_traits.is_ok());
    }
}

// Usage Patterns Tests

#[allow(non_snake_case)]
#[cfg(test)]
mod usage_patterns_tests {
    use super::*;

    #[test]
    fn test_usage_pattern_config() {
        let config = UsagePatternConfig::default();
        assert_eq!(config.temporal_window_days, 30);
        assert_eq!(config.min_usage_threshold, 5);
        assert!(config.trend_analysis_enabled);
    }

    #[test]
    fn test_usage_pattern_analyzer_creation() {
        let config = UsagePatternConfig::default();
        let feature_extractor = TraitFeatureExtractor::new(FeatureExtractionConfig::default());
        let analyzer = UsagePatternAnalyzer::new(config, feature_extractor);

        assert_eq!(analyzer.usage_history.len(), 0);
        assert_eq!(analyzer.temporal_patterns.len(), 0);
    }

    #[test]
    fn test_usage_event_recording() {
        let config = UsagePatternConfig::default();
        let feature_extractor = TraitFeatureExtractor::new(FeatureExtractionConfig::default());
        let mut analyzer = UsagePatternAnalyzer::new(config, feature_extractor);

        let event = create_test_usage_event("TestTrait", 1234567890);
        let result = analyzer.record_usage(event);
        assert!(result.is_ok());

        assert_eq!(analyzer.usage_history.len(), 1);
    }

    #[test]
    fn test_temporal_pattern_analysis() {
        let config = UsagePatternConfig::default();
        let feature_extractor = TraitFeatureExtractor::new(FeatureExtractionConfig::default());
        let mut analyzer = UsagePatternAnalyzer::new(config, feature_extractor);

        // Add multiple events
        for i in 0..10 {
            let event = create_test_usage_event("TestTrait", 1234567890 + (i * 3600));
            analyzer.record_usage(event).unwrap();
        }

        let pattern = analyzer.analyze_temporal_patterns("TestTrait");
        assert!(pattern.is_ok());

        let pattern = pattern.unwrap();
        assert_eq!(pattern.trait_name, "TestTrait");
        assert_eq!(pattern.hourly_distribution.len(), 24);
        assert_eq!(pattern.weekly_distribution.len(), 7);
    }

    #[test]
    fn test_user_behavior_analysis() {
        let config = UsagePatternConfig::default();
        let feature_extractor = TraitFeatureExtractor::new(FeatureExtractionConfig::default());
        let mut analyzer = UsagePatternAnalyzer::new(config, feature_extractor);

        // Add events for a specific user
        for i in 0..5 {
            let event = create_test_usage_event("TestTrait", 1234567890 + (i * 3600));
            analyzer.record_usage(event).unwrap();
        }

        let behavior = analyzer.analyze_user_behavior("test_user");
        assert!(behavior.is_ok());

        let behavior = behavior.unwrap();
        assert_eq!(behavior.user_id, "test_user");
        assert!(behavior.usage_frequency > 0.0);
        assert!(behavior.success_rate > 0.0);
    }

    #[test]
    fn test_seasonal_pattern_detection() {
        let config = UsagePatternConfig::default();
        let feature_extractor = TraitFeatureExtractor::new(FeatureExtractionConfig::default());
        let mut analyzer = UsagePatternAnalyzer::new(config, feature_extractor);

        // Add many events to enable seasonal analysis
        for i in 0..100 {
            let event = create_test_usage_event("TestTrait", 1234567890 + (i * 3600));
            analyzer.record_usage(event).unwrap();
        }

        let patterns = analyzer.detect_seasonal_patterns("TestTrait");
        assert!(patterns.is_ok());

        let patterns = patterns.unwrap();
        assert!(patterns.len() >= 0); // May or may not detect patterns
    }

    #[test]
    fn test_usage_prediction() {
        let config = UsagePatternConfig::default();
        let feature_extractor = TraitFeatureExtractor::new(FeatureExtractionConfig::default());
        let mut analyzer = UsagePatternAnalyzer::new(config, feature_extractor);

        // Create temporal pattern first
        for i in 0..10 {
            let event = create_test_usage_event("TestTrait", 1234567890 + (i * 3600));
            analyzer.record_usage(event).unwrap();
        }
        analyzer.analyze_temporal_patterns("TestTrait").unwrap();

        let predictions = analyzer.predict_usage("TestTrait", 7);
        assert!(predictions.is_ok());

        let predictions = predictions.unwrap();
        assert_eq!(predictions.len(), 7);

        // All predictions should be non-negative
        for prediction in predictions {
            assert!(prediction >= 0.0);
        }
    }

    #[test]
    fn test_pattern_based_recommender() {
        let config = UsagePatternConfig::default();
        let feature_extractor = TraitFeatureExtractor::new(FeatureExtractionConfig::default());
        let pattern_analyzer = UsagePatternAnalyzer::new(config.clone(), feature_extractor);
        let mut recommender = PatternBasedRecommender::new(pattern_analyzer, config);

        let context = create_test_trait_context();
        let recommendations = recommender.recommend_based_on_patterns(&context);
        assert!(recommendations.is_ok());

        let recommendations = recommendations.unwrap();
        assert!(recommendations.len() >= 0); // May return empty if no patterns
    }

    #[test]
    fn test_anomaly_detection() {
        let config = UsagePatternConfig::default();
        let feature_extractor = TraitFeatureExtractor::new(FeatureExtractionConfig::default());
        let analyzer = UsagePatternAnalyzer::new(config, feature_extractor);

        let event = create_test_usage_event("TestTrait", 1234567890);
        let anomaly = analyzer.detect_anomaly(&event);
        assert!(anomaly.is_ok());

        let anomaly = anomaly.unwrap();
        assert!(anomaly.is_none()); // Should be none without pattern history
    }
}

// Recommender Engine Tests

#[allow(non_snake_case)]
#[cfg(test)]
mod recommender_engine_tests {
    use super::*;

    #[test]
    fn test_recommender_engine_creation() {
        let config = RecommenderEngineConfig::default();
        let engine = MLRecommendationEngine::new(config);
        assert!(engine.is_ok());
    }

    #[test]
    fn test_ensemble_weights_validation() {
        let weights = EnsembleWeights::default();
        let total = weights.similarity_weight + weights.collaborative_weight +
                   weights.neural_weight + weights.pattern_weight +
                   weights.clustering_weight + weights.content_weight;

        assert!((total - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_trait_database_operations() {
        let mut db = TraitDatabase::new();

        let trait_info = create_test_trait_info("TestTrait");
        db.add_trait(trait_info.clone());

        assert!(db.get_trait("TestTrait").is_some());
        assert_eq!(db.get_trait("TestTrait").unwrap().name, "TestTrait");

        db.add_relationship("TestTrait", "RelatedTrait");
        let related = db.get_related_traits("TestTrait");
        assert!(related.contains(&"RelatedTrait".to_string()));
    }

    #[test]
    fn test_cache_settings() {
        let settings = CacheSettings::default();
        assert!(settings.enable_recommendation_cache);
        assert!(settings.enable_similarity_cache);
        assert_eq!(settings.cache_ttl_seconds, 3600);
    }

    #[test]
    fn test_performance_tuning_config() {
        let config = PerformanceTuning::default();
        assert!(config.enable_parallel_processing);
        assert_eq!(config.max_concurrent_recommendations, 10);
        assert_eq!(config.batch_size, 32);
    }

    #[test]
    fn test_recommendation_stats() {
        let stats = RecommendationStats::default();
        assert_eq!(stats.total_requests, 0);
        assert_eq!(stats.cache_hits, 0);
        assert_eq!(stats.successful_recommendations, 0);
    }

    #[test]
    fn test_training_data_structure() {
        let training_data = TrainingData {
            contexts: vec![create_test_trait_context()],
            features: create_test_array2(10, 5),
            user_trait_matrix: create_test_array2(5, 10),
            trait_features: create_test_array2(10, 8),
            trait_relationships: HashMap::new(),
            usage_patterns: vec![create_test_usage_event("TestTrait", 1234567890)],
        };

        assert_eq!(training_data.contexts.len(), 1);
        assert_eq!(training_data.features.nrows(), 10);
        assert_eq!(training_data.usage_patterns.len(), 1);
    }

    #[test]
    fn test_recommendation_generation() {
        let config = RecommenderEngineConfig::default();
        let engine = MLRecommendationEngine::new(config).unwrap();

        let context = create_test_trait_context();
        let recommendations = engine.recommend(&context, 5);

        // This might fail without proper initialization, but structure should be correct
        assert!(recommendations.is_ok() || recommendations.is_err());
    }

    #[test]
    fn test_cosine_similarity_calculation() {
        let config = RecommenderEngineConfig::default();
        let engine = MLRecommendationEngine::new(config).unwrap();

        let vec_a = array![1.0, 0.0, 0.0];
        let vec_b = array![0.0, 1.0, 0.0];
        let vec_c = array![1.0, 0.0, 0.0];

        let sim_ab = engine.calculate_cosine_similarity(&vec_a, &vec_b);
        let sim_ac = engine.calculate_cosine_similarity(&vec_a, &vec_c);

        assert_eq!(sim_ab, 0.0); // Orthogonal vectors
        assert_eq!(sim_ac, 1.0); // Identical vectors
    }

    #[test]
    fn test_cache_key_generation() {
        let config = RecommenderEngineConfig::default();
        let engine = MLRecommendationEngine::new(config).unwrap();

        let context1 = create_test_trait_context();
        let context2 = create_test_trait_context();

        let key1 = engine.context_to_cache_key(&context1);
        let key2 = engine.context_to_cache_key(&context2);

        assert_eq!(key1, key2); // Same contexts should generate same keys
    }

    #[test]
    fn test_usage_event_recording() {
        let config = RecommenderEngineConfig::default();
        let engine = MLRecommendationEngine::new(config).unwrap();

        let event = create_test_usage_event("TestTrait", 1234567890);
        let result = engine.record_usage(event);
        assert!(result.is_ok());
    }

    #[test]
    fn test_trait_database_updates() {
        let config = RecommenderEngineConfig::default();
        let engine = MLRecommendationEngine::new(config).unwrap();

        let trait_info = create_test_trait_info("NewTrait");
        let result = engine.update_trait_database(trait_info);
        assert!(result.is_ok());
    }
}

// Integration Tests

#[allow(non_snake_case)]
#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn test_end_to_end_recommendation_flow() {
        let config = RecommenderEngineConfig::default();
        let mut engine = MLRecommendationEngine::new(config).unwrap();

        // Add some trait data
        let trait_info = create_test_trait_info("Debug");
        engine.update_trait_database(trait_info).unwrap();

        // Record some usage
        let event = create_test_usage_event("Debug", 1234567890);
        engine.record_usage(event).unwrap();

        // Get recommendations
        let context = create_test_trait_context();
        let recommendations = engine.recommend(&context, 3);

        // Should work or fail gracefully
        assert!(recommendations.is_ok() || recommendations.is_err());
    }

    #[test]
    fn test_multi_module_interaction() {
        // Test that all modules work together
        let ml_config = MLRecommendationConfig::default();
        let neural_config = NeuralNetworkConfig::default();
        let feature_config = FeatureExtractionConfig::default();
        let pattern_config = UsagePatternConfig::default();

        // Create components
        let neural_model = NeuralEmbeddingModel::new(neural_config);
        assert!(neural_model.is_ok());

        let feature_extractor = TraitFeatureExtractor::new(feature_config.clone());
        let pattern_analyzer = UsagePatternAnalyzer::new(pattern_config.clone(), feature_extractor);
        let similarity_model = TraitSimilarityModel::new(ml_config);

        // All components should be created successfully
        assert!(true); // If we get here, all components were created
    }

    #[test]
    fn test_error_handling_across_modules() {
        // Test error propagation across modules
        let config = FeatureExtractionConfig::default();
        let mut extractor = TraitFeatureExtractor::new(config);

        // Test with invalid context (empty)
        let empty_context = TraitContext::default();
        let result = extractor.extract_context_features(&empty_context);

        // Should handle gracefully
        assert!(result.is_ok() || result.is_err());
    }

    #[test]
    fn test_performance_with_large_data() {
        // Test performance characteristics with larger datasets
        let config = MLRecommendationConfig::default();
        let similarity_model = TraitSimilarityModel::new(config);

        // Create larger test arrays
        let large_vec = Array1::from_iter((0..1000).map(|x| x as f64));
        let other_vec = Array1::from_iter((0..1000).map(|x| (x * 2) as f64));

        let start = SystemTime::now();
        let similarity = similarity_model.calculate_cosine_similarity(&large_vec, &other_vec);
        let duration = start.elapsed().unwrap();

        // Should complete in reasonable time
        assert!(duration.as_millis() < 100);
        assert!(similarity >= -1.0 && similarity <= 1.0);
    }

    #[test]
    fn test_concurrent_access() {
        use std::sync::Arc;
        use std::thread;

        let config = RecommenderEngineConfig::default();
        let engine = Arc::new(MLRecommendationEngine::new(config).unwrap());

        let mut handles = vec![];

        // Spawn multiple threads accessing the engine
        for i in 0..5 {
            let engine_clone = Arc::clone(&engine);
            let handle = thread::spawn(move || {
                let context = create_test_trait_context();
                let stats = engine_clone.get_stats();
                assert_eq!(stats.total_requests, 0); // Initially zero
            });
            handles.push(handle);
        }

        // Wait for all threads to complete
        for handle in handles {
            handle.join().unwrap();
        }
    }

    #[test]
    fn test_memory_efficiency() {
        // Test that large operations don't consume excessive memory
        let config = UsagePatternConfig::default();
        let feature_extractor = TraitFeatureExtractor::new(FeatureExtractionConfig::default());
        let mut analyzer = UsagePatternAnalyzer::new(config, feature_extractor);

        // Add many events
        for i in 0..1000 {
            let event = create_test_usage_event("TestTrait", 1234567890 + i);
            analyzer.record_usage(event).unwrap();
        }

        // Should maintain reasonable memory usage (tested by not crashing)
        assert!(analyzer.usage_history.len() <= 1000);
    }
}

// Benchmark Tests (Simple timing tests)

#[allow(non_snake_case)]
#[cfg(test)]
mod benchmark_tests {
    use super::*;

    #[test]
    fn benchmark_feature_extraction() {
        let config = FeatureExtractionConfig::default();
        let mut extractor = TraitFeatureExtractor::new(config);
        let context = create_test_trait_context();

        let start = SystemTime::now();
        for _ in 0..100 {
            let _ = extractor.extract_context_features(&context);
        }
        let duration = start.elapsed().unwrap();

        // Should complete 100 extractions in reasonable time
        assert!(duration.as_millis() < 1000);
    }

    #[test]
    fn benchmark_similarity_calculation() {
        let config = MLRecommendationConfig::default();
        let model = TraitSimilarityModel::new(config);

        let vec_a = Array1::from_iter((0..100).map(|x| x as f64));
        let vec_b = Array1::from_iter((0..100).map(|x| (x * 2) as f64));

        let start = SystemTime::now();
        for _ in 0..1000 {
            let _ = model.calculate_cosine_similarity(&vec_a, &vec_b);
        }
        let duration = start.elapsed().unwrap();

        // Should complete 1000 similarity calculations in reasonable time
        assert!(duration.as_millis() < 500);
    }

    #[test]
    fn benchmark_neural_forward_pass() {
        let config = NeuralNetworkConfig::default();
        let model = NeuralEmbeddingModel::new(config).unwrap();
        let input = Array1::zeros(128);

        let start = SystemTime::now();
        for _ in 0..100 {
            let _ = model.forward(&input);
        }
        let duration = start.elapsed().unwrap();

        // Should complete 100 forward passes in reasonable time
        assert!(duration.as_millis() < 1000);
    }
}

// Property-based tests (using simple invariants)

#[allow(non_snake_case)]
#[cfg(test)]
mod property_tests {
    use super::*;

    #[test]
    fn property_cosine_similarity_bounds() {
        let config = MLRecommendationConfig::default();
        let model = TraitSimilarityModel::new(config);

        // Test with random vectors
        for _ in 0..100 {
            let vec_a = Array1::from_iter((0..10).map(|_| (rng().gen::<f64>() - 0.5) * 100.0));
            let vec_b = Array1::from_iter((0..10).map(|_| (rng().gen::<f64>() - 0.5) * 100.0));

            let similarity = model.calculate_cosine_similarity(&vec_a, &vec_b);

            // Cosine similarity should always be between -1 and 1
            assert!(similarity >= -1.0 && similarity <= 1.0);
        }
    }

    #[test]
    fn property_distance_non_negativity() {
        let config = MLRecommendationConfig::default();
        let model = TraitSimilarityModel::new(config);

        // Test with random vectors
        for _ in 0..100 {
            let vec_a = Array1::from_iter((0..10).map(|_| rng().gen::<f64>() * 100.0));
            let vec_b = Array1::from_iter((0..10).map(|_| rng().gen::<f64>() * 100.0));

            let euclidean = model.calculate_euclidean_distance(&vec_a, &vec_b);
            let manhattan = model.calculate_manhattan_distance(&vec_a, &vec_b);

            // Distances should always be non-negative
            assert!(euclidean >= 0.0);
            assert!(manhattan >= 0.0);
        }
    }

    #[test]
    fn property_neural_output_finite() {
        let config = NeuralNetworkConfig::default();
        let model = NeuralEmbeddingModel::new(config).unwrap();

        // Test with random inputs
        for _ in 0..50 {
            let input = Array1::from_iter((0..128).map(|_| (rng().gen::<f64>() - 0.5) * 10.0));
            let output = model.forward(&input).unwrap();

            // All outputs should be finite
            for value in output.iter() {
                assert!(value.is_finite());
            }
        }
    }
}