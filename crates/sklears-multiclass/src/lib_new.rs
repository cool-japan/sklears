//! Multiclass classification strategies
//!
//! This module provides meta-estimators for multiclass classification problems.
//! It implements strategies like One-vs-Rest and One-vs-One for transforming
//! binary classifiers into multiclass ones.

#![warn(missing_docs)]

// Existing modules
pub mod advanced;
pub mod calibration;
pub mod core;
pub mod ensemble;
pub mod incremental;
pub mod uncertainty;
pub mod utils;

// New extracted modules
pub mod one_vs_rest;
pub mod one_vs_one;
pub mod ecoc;
pub mod boosting;
pub mod rotation_forest;
pub mod dynamic_ensemble;

// Re-export existing module types
pub use advanced::*;
pub use calibration::*;
pub use core::ecoc::*;
pub use ensemble::*;
pub use utils::*;

// Re-export extracted module types
pub use one_vs_rest::{
    OneVsRestClassifier, OneVsRestConfig, OneVsRestBuilder,
    OneVsRestTrainedData, TrainedOneVsRest
};

pub use one_vs_one::{
    OneVsOneClassifier, OneVsOneConfig, OneVsOneBuilder,
    OneVsOneTrainedData, TrainedOneVsOne, VotingStrategy,
    ConsensusConfig, ConsensusStrategy, ConsensusMethod, ConsensusResult
};

pub use ecoc::{
    ECOCClassifier, ECOCConfig, ECOCBuilder, ECOCStrategy,
    ECOCTrainedData, TrainedECOC
};

pub use boosting::{
    AdaBoostClassifier, AdaBoostConfig, AdaBoostBuilder, AdaBoostStrategy,
    AdaBoostTrainedData, TrainedAdaBoost,
    GradientBoostingClassifier, GradientBoostingConfig, GradientBoostingBuilder,
    GradientBoostingLoss, GradientBoostingTrainedData, TrainedGradientBoosting
};

pub use rotation_forest::{
    RotationForestClassifier, RotationForestConfig, RotationForestBuilder,
    FeatureSelectionStrategy, RotationInfo, RotationForestTrainedData, TrainedRotationForest
};

pub use dynamic_ensemble::{
    DynamicEnsembleSelectionClassifier, DynamicEnsembleSelectionConfig, DynamicEnsembleSelectionBuilder,
    CompetenceMeasure, SelectionStrategy, PoolGenerationStrategy, CompetenceRegion,
    DynamicEnsembleSelectionTrainedData, TrainedDynamicEnsemble
};