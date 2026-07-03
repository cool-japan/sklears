//! Ensemble methods module
//!
//! This module provides various ensemble methods including voting, stacking,
//! dynamic selection, model fusion, and hierarchical composition.

pub mod common;
pub mod dynamic_selection;
pub mod voting;

// Re-export commonly used items
pub use common::{simd_fallback, ActivationFunction, EnsembleStatistics};
pub use dynamic_selection::{
    CompetenceEstimation, DynamicEnsembleSelector, DynamicEnsembleSelectorBuilder,
    DynamicEnsembleSelectorTrained, SelectionStrategy,
};
pub use voting::{
    VotingClassifier, VotingClassifierBuilder, VotingClassifierTrained, VotingRegressor,
    VotingRegressorBuilder, VotingRegressorTrained,
};

// `model_fusion`, `hierarchical_composition` and `stacking` live at the crate
// root (`crate::model_fusion`, `crate::hierarchical_composition`,
// `crate::stacking`), not as submodules of `ensemble`, so they're re-exported
// here with an explicit `crate::` prefix rather than a bare relative path.
pub use crate::hierarchical_composition::*;
pub use crate::model_fusion::*;
pub use crate::stacking::*;
