//! Multiclass classification strategies
//!
//! This module provides meta-estimators for multiclass classification problems.
//! It implements strategies like One-vs-Rest and One-vs-One for transforming
//! binary classifiers into multiclass ones.
//!
//! All modules are enabled as of v0.1.1. The ndarray HRTB (Higher-Ranked Trait Bound)
//! lifetime constraint issue introduced in ndarray 0.17 was resolved by adding explicit
//! `Fitted = F` associated-type bounds to all `Fit` trait constraints.

// #![warn(missing_docs)]

pub mod advanced;
pub mod calibration;
pub mod core;
pub mod ensemble;
pub mod export;
pub mod gpu;
pub mod incremental;
pub mod memory;
pub mod simd;
pub mod uncertainty;
pub mod utils;

pub mod boosting;
pub mod one_vs_one;
pub mod one_vs_rest;

pub use advanced::*;
pub use calibration::*;
pub use core::ecoc::*;
pub use ensemble::*;
pub use utils::*;

pub use one_vs_rest::{
    OneVsRestBuilder, OneVsRestClassifier, OneVsRestConfig, OneVsRestTrainedData, TrainedOneVsRest,
};

pub use one_vs_one::{
    ConsensusConfig, ConsensusMethod, ConsensusResult, ConsensusStrategy, OneVsOneBuilder,
    OneVsOneClassifier, OneVsOneConfig, OneVsOneTrainedData, TrainedOneVsOne, VotingStrategy,
};

pub use boosting::{
    AdaBoostBuilder, AdaBoostClassifier, AdaBoostConfig, AdaBoostStrategy, AdaBoostTrainedData,
    TrainedAdaBoost,
};
