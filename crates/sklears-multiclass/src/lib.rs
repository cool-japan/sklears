#![allow(dead_code)]
#![allow(non_snake_case)]
#![allow(missing_docs)]
#![allow(deprecated)]
//! Multiclass classification strategies
//!
//! This module provides meta-estimators for multiclass classification problems.
//! It implements strategies like One-vs-Rest and One-vs-One for transforming
//! binary classifiers into multiclass ones.

// #![warn(missing_docs)]

// Existing modules
pub mod advanced;
pub mod calibration;
pub mod core;
pub mod ensemble;
pub mod incremental;
pub mod uncertainty;
pub mod utils;

// New extracted modules
pub mod boosting;
pub mod dynamic_ensemble;
pub mod ecoc;
pub mod one_vs_one;
pub mod one_vs_rest;
pub mod rotation_forest;

// Re-export existing module types
pub use advanced::*;
pub use calibration::*;
pub use core::ecoc::*;
pub use ensemble::*;
pub use utils::*;

// Re-export extracted module types
pub use one_vs_rest::{
    OneVsRestBuilder, OneVsRestClassifier, OneVsRestConfig, OneVsRestTrainedData, TrainedOneVsRest,
};

pub use one_vs_one::{
    ConsensusConfig, ConsensusMethod, ConsensusResult, ConsensusStrategy, OneVsOneBuilder,
    OneVsOneClassifier, OneVsOneConfig, OneVsOneTrainedData, TrainedOneVsOne, VotingStrategy,
};

pub use ecoc::{
    ECOCBuilder, ECOCClassifier, ECOCConfig, ECOCStrategy, ECOCTrainedData, TrainedECOC,
};

pub use boosting::{
    AdaBoostBuilder, AdaBoostClassifier, AdaBoostConfig, AdaBoostStrategy, AdaBoostTrainedData,
    GradientBoostingBuilder, GradientBoostingClassifier, GradientBoostingConfig,
    GradientBoostingLoss, GradientBoostingTrainedData, TrainedAdaBoost, TrainedGradientBoosting,
};

pub use rotation_forest::{
    FeatureSelectionStrategy, RotationForestBuilder, RotationForestClassifier,
    RotationForestConfig, RotationForestTrainedData, RotationInfo, TrainedRotationForest,
};

pub use dynamic_ensemble::{
    CompetenceMeasure, CompetenceRegion, DynamicEnsembleSelectionBuilder,
    DynamicEnsembleSelectionClassifier, DynamicEnsembleSelectionConfig,
    DynamicEnsembleSelectionTrainedData, PoolGenerationStrategy, SelectionStrategy,
    TrainedDynamicEnsemble,
};
