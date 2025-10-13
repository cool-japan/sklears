// Module declarations for the ML-powered trait recommendation system
// This module provides comprehensive ML-based trait recommendations using
// multiple approaches including neural networks, collaborative filtering,
// clustering, usage patterns, and similarity analysis.

pub mod data_types;
pub mod neural_networks;
pub mod feature_extraction;
pub mod ml_models;
pub mod usage_patterns;
pub mod recommender_engine;

#[allow(non_snake_case)]
#[cfg(test)]
pub mod tests;

// Re-export main types for easier access
pub use data_types::*;
pub use neural_networks::NeuralEmbeddingModel;
pub use feature_extraction::TraitFeatureExtractor;
pub use ml_models::{TraitSimilarityModel, ClusteringModel, CollaborativeFilteringModel};
pub use usage_patterns::{UsagePatternAnalyzer, PatternBasedRecommender, UsagePatternConfig, UsageEvent};
pub use recommender_engine::{MLRecommendationEngine, RecommenderEngineConfig, TrainingData, TraitDatabase};