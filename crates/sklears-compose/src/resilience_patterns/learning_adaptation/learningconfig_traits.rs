//! # LearningConfig - Trait Implementations
//!
//! This module contains trait implementations for `LearningConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::fault_core::*;

use super::types::{FeatureExtractionConfig, FeedbackProcessingConfig, KnowledgeBaseConfig, LearningConfig, ModelManagementConfig, OptimizationConfig, PatternLearningConfig, PredictionConfig, SchedulingConfig};

impl Default for LearningConfig {
    fn default() -> Self {
        Self {
            pattern_learning: PatternLearningConfig::default(),
            optimization: OptimizationConfig::default(),
            prediction: PredictionConfig::default(),
            feedback_processing: FeedbackProcessingConfig::default(),
            model_management: ModelManagementConfig::default(),
            feature_extraction: FeatureExtractionConfig::default(),
            scheduling: SchedulingConfig::default(),
            knowledge_base: KnowledgeBaseConfig::default(),
        }
    }
}

