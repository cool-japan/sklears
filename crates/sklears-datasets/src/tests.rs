//! Comprehensive test suite for dataset generation and validation
//!
//! This module provides extensive test coverage for all dataset generation capabilities including
//! basic generators, advanced clustering, time series datasets, computer vision datasets,
//! survival analysis, concept drift detection, multimodal datasets, and validation frameworks.
//! All tests have been refactored into focused modules for better maintainability and comply
//! with SciRS2 Policy for comprehensive quality assurance.

// Core testing infrastructure and base utilities
mod test_core;
pub use test_core::{
    TestFramework, TestRunner, TestValidator, TestUtilities,
    TestConfiguration, TestMetrics, TestReporter, AssertionUtilities
};

// Basic dataset generator tests
mod basic_generator_tests;
pub use basic_generator_tests::{
    test_make_blobs, test_make_classification, test_make_regression,
    test_make_circles, test_make_moons, test_make_gaussian_quantiles,
    BasicGeneratorTestSuite, GeneratorValidationTests
};

// Advanced clustering dataset tests
mod clustering_tests;
pub use clustering_tests::{
    test_make_overlapping_blobs, test_make_hierarchical_clusters,
    test_make_density_based_clusters, test_make_anisotropic_clusters,
    ClusteringTestSuite, ClusterValidationTests, ClusterQualityTests
};

// Time series and temporal dataset tests
mod time_series_tests;
pub use time_series_tests::{
    test_make_time_series, test_make_concept_drift_dataset,
    test_make_seasonal_time_series, test_make_autoregressive_series,
    TimeSeriesTestSuite, TemporalValidationTests, ConceptDriftTests
};

// Computer vision and image dataset tests
mod vision_tests;
pub use vision_tests::{
    test_make_image_classification_dataset, test_make_texture_dataset,
    test_make_pattern_dataset, test_make_geometric_shapes,
    VisionTestSuite, ImageValidationTests, VisualPatternTests
};

// Survival analysis dataset tests
mod survival_tests;
pub use survival_tests::{
    test_make_censored_survival_data, test_survival_data_weibull,
    test_survival_data_informative_censoring, test_survival_data_invalid_input,
    SurvivalTestSuite, CensoringTests, HazardModelTests
};

// Distribution shift and concept drift tests
mod distribution_shift_tests;
pub use distribution_shift_tests::{
    test_make_distribution_shift_dataset, test_covariate_shift,
    test_prior_shift, test_concept_shift, test_mixed_shift,
    DistributionShiftTestSuite, ConceptDriftTestSuite, AdaptationTests
};

// Biclustering and subspace dataset tests
mod biclustering_tests;
pub use biclustering_tests::{
    test_make_biclusters, test_make_checkerboard, test_make_spectral_biclusters,
    test_make_plaid_biclusters, BiclusteringTestSuite, SubspaceTests
};

// Multimodal and cross-modal dataset tests
mod multimodal_tests;
pub use multimodal_tests::{
    test_make_multimodal_dataset, test_make_cross_modal_dataset,
    test_make_vision_language_dataset, test_make_audio_visual_dataset,
    MultimodalTestSuite, CrossModalValidationTests, ModalityAlignmentTests
};

// Anomaly detection dataset tests
mod anomaly_tests;
pub use anomaly_tests::{
    test_make_anomaly_detection_dataset, test_make_outlier_dataset,
    test_make_novelty_dataset, test_make_fraud_detection_dataset,
    AnomalyTestSuite, OutlierDetectionTests, NoveltyValidationTests
};

// Graph and network dataset tests
mod graph_tests;
pub use graph_tests::{
    test_make_graph_classification_dataset, test_make_network_dataset,
    test_make_social_network, test_make_knowledge_graph,
    GraphTestSuite, NetworkValidationTests, TopologyTests
};

// Natural language processing dataset tests
mod nlp_tests;
pub use nlp_tests::{
    test_make_text_classification_dataset, test_make_sentiment_dataset,
    test_make_topic_modeling_dataset, test_make_language_modeling_dataset,
    NLPTestSuite, TextValidationTests, LanguageModelTests
};

// Recommendation system dataset tests
mod recommendation_tests;
pub use recommendation_tests::{
    test_make_recommendation_dataset, test_make_collaborative_filtering_dataset,
    test_make_content_based_dataset, test_make_hybrid_recommendation_dataset,
    RecommendationTestSuite, CollaborativeFilteringTests, ContentBasedTests
};

// Streaming and online learning dataset tests
mod streaming_tests;
pub use streaming_tests::{
    test_make_streaming_dataset, test_make_data_stream,
    test_make_concept_drift_stream, test_make_evolving_dataset,
    StreamingTestSuite, OnlineLearningTests, DataStreamValidationTests
};

// Noise and corruption pattern tests
mod noise_tests;
pub use noise_tests::{
    test_noise_injection, test_missing_data_patterns, test_label_noise,
    test_feature_corruption, test_outlier_injection,
    NoiseTestSuite, CorruptionTests, DataQualityTests
};

// Feature engineering and transformation tests
mod feature_tests;
pub use feature_tests::{
    test_synthetic_feature_generation, test_correlated_features,
    test_redundant_features, test_nonlinear_features, test_interaction_features,
    FeatureTestSuite, FeatureValidationTests, TransformationTests
};

// Distribution and statistical pattern tests
mod distribution_tests;
pub use distribution_tests::{
    test_gaussian_mixture_generation, test_exponential_family_generation,
    test_heavy_tailed_distributions, test_multimodal_distributions,
    DistributionTestSuite, StatisticalValidationTests, FittingTests
};

// Performance and scalability tests
mod performance_tests;
pub use performance_tests::{
    test_large_dataset_generation, test_memory_efficiency, test_parallel_generation,
    test_computation_time, test_scalability_limits,
    PerformanceTestSuite, ScalabilityTests, EfficiencyTests
};

// Validation and quality assurance tests
mod validation_tests;
pub use validation_tests::{
    test_dataset_quality_assessment, test_generation_validation,
    test_statistical_properties, test_distribution_validation,
    ValidationTestSuite, QualityAssuranceTests, ConsistencyTests
};

// Integration and end-to-end tests
mod integration_tests;
pub use integration_tests::{
    test_pipeline_integration, test_cross_dataset_compatibility,
    test_workflow_validation, test_end_to_end_scenarios,
    IntegrationTestSuite, WorkflowTests, CompatibilityTests
};

// Property-based and fuzzing tests
mod property_tests;
pub use property_tests::{
    test_invariant_properties, test_boundary_conditions, test_edge_cases,
    test_random_parameter_combinations, test_property_preservation,
    PropertyTestSuite, FuzzingTests, InvariantTests
};

// Benchmarking and comparison tests
mod benchmark_tests;
pub use benchmark_tests::{
    test_generation_benchmarks, test_quality_comparisons, test_performance_baselines,
    test_accuracy_benchmarks, test_efficiency_comparisons,
    BenchmarkTestSuite, ComparisonTests, BaselineTests
};

// Regression and compatibility tests
mod regression_tests;
pub use regression_tests::{
    test_backward_compatibility, test_version_consistency, test_api_stability,
    test_output_reproducibility, test_parameter_handling,
    RegressionTestSuite, CompatibilityTests, StabilityTests
};

// Documentation and example tests
mod documentation_tests;
pub use documentation_tests::{
    test_code_examples, test_documentation_accuracy, test_tutorial_validity,
    test_api_documentation, test_usage_patterns,
    DocumentationTestSuite, ExampleValidationTests, TutorialTests
};

// Utilities and helper test functions
mod test_utilities;
pub use test_utilities::{
    TestUtilities, AssertionHelpers, ValidationHelpers, ComparisonUtilities,
    StatisticalTestHelpers, TestDataGenerators, MockUtilities, TestConfiguration
};