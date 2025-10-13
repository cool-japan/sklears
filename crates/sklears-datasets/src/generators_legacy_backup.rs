//! Synthetic data generators
//!
//! This module provides comprehensive synthetic dataset generation capabilities including
//! classification datasets, regression datasets, clustering datasets, biclustering datasets,
//! time series datasets, computer vision datasets, object detection datasets, streaming datasets,
//! graph datasets, text datasets, recommendation datasets, and specialized domain datasets.
//! All algorithms have been refactored into focused modules for better maintainability
//! and comply with SciRS2 Policy.

// Core dataset generation types and base structures
mod generation_core;
pub use generation_core::{
    DatasetGenerator, GeneratorConfig, SyntheticDataset, GenerationResult,
    GeneratorMetadata, GenerationValidator, DatasetQuality
};

// Basic classification dataset generators
mod classification_generators;
pub use classification_generators::{
    make_classification, make_blobs, make_moons, make_circles,
    make_gaussian_quantiles, make_hastie_10_2, ClassificationGenerator,
    SeparabilityGenerator, NonLinearGenerator, BoundaryGenerator
};

// Regression dataset generators
mod regression_generators;
pub use regression_generators::{
    make_regression, make_friedman1, make_friedman2, make_friedman3,
    make_sparse_uncorrelated, RegressionGenerator, NonLinearRegressionGenerator,
    SparseRegressionGenerator, CorrelatedRegressionGenerator
};

// Clustering and unsupervised dataset generators
mod clustering_generators;
pub use clustering_generators::{
    make_clustering_dataset, make_anisotropic_blobs, make_varied_blobs,
    make_density_blobs, ClusteringGenerator, AnisotropicGenerator,
    DensityVariationGenerator, ClusterShapeGenerator
};

// Biclustering dataset generators
mod biclustering_generators;
pub use biclustering_generators::{
    make_biclusters, make_checkerboard, make_spectral_biclusters,
    make_plaid_biclusters, BiclusteringGenerator, CheckerboardGenerator,
    SpectralBiclusterGenerator, PlaidBiclusterGenerator
};

// Time series and temporal dataset generators
mod time_series_generators;
pub use time_series_generators::{
    make_time_series, make_concept_drift_dataset, make_seasonal_time_series,
    make_trend_time_series, TimeSeriesGenerator, ConceptDriftGenerator,
    SeasonalGenerator, TrendGenerator, TemporalPatternGenerator
};

// Computer vision and image dataset generators
mod vision_generators;
pub use vision_generators::{
    make_image_classification_dataset, make_texture_dataset, make_pattern_dataset,
    make_geometric_dataset, VisionGenerator, TextureGenerator,
    PatternGenerator, GeometricShapeGenerator, ImageAugmentor
};

// Object detection dataset generators
mod object_detection_generators;
pub use object_detection_generators::{
    make_object_detection_dataset, make_segmentation_dataset, make_keypoint_dataset,
    make_instance_segmentation_dataset, ObjectDetectionGenerator, SegmentationGenerator,
    KeypointGenerator, InstanceGenerator, BoundingBoxGenerator
};

// Natural language processing dataset generators
mod nlp_generators;
pub use nlp_generators::{
    make_text_classification_dataset, make_text_regression_dataset,
    make_sentiment_dataset, make_topic_modeling_dataset, TextGenerator,
    SentimentGenerator, TopicGenerator, LanguageModelGenerator
};

// Graph and network dataset generators
mod graph_generators;
pub use graph_generators::{
    make_graph_classification_dataset, make_network_dataset, make_social_network,
    make_knowledge_graph, GraphGenerator, NetworkGenerator,
    SocialNetworkGenerator, KnowledgeGraphGenerator, GraphTopologyGenerator
};

// Recommendation system dataset generators
mod recommendation_generators;
pub use recommendation_generators::{
    make_recommendation_dataset, make_collaborative_filtering_dataset,
    make_content_based_dataset, make_hybrid_recommendation_dataset,
    RecommendationGenerator, CollaborativeFilteringGenerator,
    ContentBasedGenerator, HybridRecommendationGenerator
};

// Streaming and online learning dataset generators
mod streaming_generators;
pub use streaming_generators::{
    make_streaming_dataset, make_data_stream, make_concept_drift_stream,
    make_evolving_dataset, StreamingGenerator, DataStreamGenerator,
    ConceptDriftStreamGenerator, EvolvingDataGenerator
};

// Anomaly detection dataset generators
mod anomaly_generators;
pub use anomaly_generators::{
    make_anomaly_detection_dataset, make_outlier_dataset, make_novelty_dataset,
    make_fraud_detection_dataset, AnomalyGenerator, OutlierGenerator,
    NoveltyGenerator, FraudDetectionGenerator, AnomalyPatternGenerator
};

// Multimodal and cross-modal dataset generators
mod multimodal_generators;
pub use multimodal_generators::{
    make_multimodal_dataset, make_cross_modal_dataset, make_vision_language_dataset,
    make_audio_visual_dataset, MultimodalGenerator, CrossModalGenerator,
    VisionLanguageGenerator, AudioVisualGenerator
};

// Domain-specific dataset generators
mod domain_generators;
pub use domain_generators::{
    make_medical_dataset, make_financial_dataset, make_scientific_dataset,
    make_iot_dataset, MedicalDataGenerator, FinancialDataGenerator,
    ScientificDataGenerator, IoTDataGenerator, BioinformaticsGenerator
};

// Noise and corruption pattern generators
mod noise_generators;
pub use noise_generators::{
    NoiseGenerator, CorruptionGenerator, MissingDataGenerator,
    LabelNoiseGenerator, FeatureNoiseGenerator, OutlierInjector,
    NoisePatternGenerator, DataQualityController
};

// Feature engineering and transformation generators
mod feature_generators;
pub use feature_generators::{
    FeatureGenerator, SyntheticFeatureGenerator, CorrelatedFeatureGenerator,
    RedundantFeatureGenerator, NonLinearFeatureGenerator, InteractionFeatureGenerator,
    PolynomialFeatureGenerator, TrigonometricFeatureGenerator
};

// Distribution and statistical pattern generators
mod distribution_generators;
pub use distribution_generators::{
    DistributionGenerator, GaussianMixtureGenerator, ExponentialFamilyGenerator,
    HeavyTailedGenerator, MultimodalDistributionGenerator, SkewedDistributionGenerator,
    StatisticalPatternGenerator, ProbabilityDistributionGenerator
};

// Validation and quality assessment utilities
mod generation_validation;
pub use generation_validation::{
    GenerationValidator, DatasetQualityAssessor, SyntheticDataValidator,
    GenerationDiagnostics, QualityMetrics, ValidationFramework,
    GenerationAnalyzer, DatasetProfiler
};

// Performance optimization for dataset generation
mod generation_optimization;
pub use generation_optimization::{
    GenerationOptimizer, ParallelGenerator, MemoryEfficientGenerator,
    BatchGenerator, StreamingOptimizer, CacheOptimizer,
    GenerationProfiler, PerformanceAnalyzer
};

// Utilities and helper functions for generation
mod generation_utilities;
pub use generation_utilities::{
    GenerationUtilities, RandomStateManager, ParameterValidator,
    ConfigurationManager, GenerationLogger, UtilityFunctions,
    GenerationHelper, MathematicalUtilities
};