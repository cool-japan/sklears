#![allow(dead_code)]
#![allow(non_snake_case)]
#![allow(missing_docs)]
#![allow(deprecated)]
#![allow(clippy::all)]
#![allow(clippy::pedantic)]
#![allow(clippy::nursery)]
//! Multiclass classification strategies
//!
//! This module provides meta-estimators for multiclass classification problems.
//! It implements strategies like One-vs-Rest and One-vs-One for transforming
//! binary classifiers into multiclass ones.
//!
//! ## Known Limitations
//!
//! The following modules are disabled due to ndarray HRTB (Higher-Ranked Trait Bound)
//! lifetime constraints introduced in ndarray 0.17. Planned for re-enabling in v0.2.0:
//! - `advanced`, `calibration`, `core`, `ensemble` - Core multiclass strategies
//! - `boosting`, `dynamic_ensemble`, `ecoc` - Ensemble multiclass methods
//! - `one_vs_one`, `one_vs_rest`, `rotation_forest` - Classification strategies

// #![warn(missing_docs)]

// KNOWN ISSUE (v0.1.0): Modules disabled due to ndarray HRTB lifetime constraints. Planned for v0.2.0.
// pub mod advanced;
// pub mod calibration;
// pub mod core;
// pub mod ensemble;
pub mod export;
pub mod gpu;
pub mod incremental;
pub mod memory;
pub mod simd;
pub mod uncertainty;
pub mod utils;

// KNOWN ISSUE (v0.1.0): Modules disabled due to ndarray HRTB lifetime constraints. Planned for v0.2.0.
// pub mod boosting;
// pub mod dynamic_ensemble;
// pub mod ecoc;
// pub mod one_vs_one;
// pub mod one_vs_rest;
// pub mod rotation_forest;

// KNOWN ISSUE (v0.1.0): Modules disabled due to ndarray HRTB lifetime constraints. Planned for v0.2.0.
// pub use advanced::*;
// pub use calibration::*;
// pub use core::ecoc::*;
// pub use ensemble::*;
pub use utils::*;

// KNOWN ISSUE (v0.1.0): Modules disabled due to ndarray HRTB lifetime constraints. Planned for v0.2.0.
// pub use one_vs_rest::{
//     OneVsRestBuilder, OneVsRestClassifier, OneVsRestConfig, OneVsRestTrainedData, TrainedOneVsRest,
// };

// pub use one_vs_one::{
//     ConsensusConfig, ConsensusMethod, ConsensusResult, ConsensusStrategy, OneVsOneBuilder,
//     OneVsOneClassifier, OneVsOneConfig, OneVsOneTrainedData, TrainedOneVsOne, VotingStrategy,
// };

// pub use ecoc::{
//     ECOCBuilder, ECOCClassifier, ECOCConfig, ECOCStrategy, ECOCTrainedData, TrainedECOC,
// };

// pub use boosting::{
//     AdaBoostBuilder, AdaBoostClassifier, AdaBoostConfig, AdaBoostStrategy, AdaBoostTrainedData,
//     GradientBoostingBuilder, GradientBoostingClassifier, GradientBoostingConfig,
//     GradientBoostingLoss, GradientBoostingTrainedData, TrainedAdaBoost, TrainedGradientBoosting,
// };

// pub use rotation_forest::{
//     FeatureSelectionStrategy, RotationForestBuilder, RotationForestClassifier,
//     RotationForestConfig, RotationForestTrainedData, RotationInfo, TrainedRotationForest,
// };

// pub use dynamic_ensemble::{
//     CompetenceMeasure, CompetenceRegion, DynamicEnsembleSelectionBuilder,
//     DynamicEnsembleSelectionClassifier, DynamicEnsembleSelectionConfig,
//     DynamicEnsembleSelectionTrainedData, PoolGenerationStrategy, SelectionStrategy,
//     TrainedDynamicEnsemble,
// };
