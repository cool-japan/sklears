//! Ensemble methods module
//!
//! This module provides various ensemble methods including voting, stacking,
//! dynamic selection, model fusion, and hierarchical composition.

pub mod common;
pub mod dynamic_selection;
pub mod voting;
// TODO: Add these modules when created:
// pub mod model_fusion;
// pub mod hierarchical;
// pub mod stacking;

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

// TODO: Re-export when modules are created:
// pub use model_fusion::*;
// pub use hierarchical::*;
// pub use stacking::*;

// Temporary placeholder types to maintain compilation
// These should be replaced with proper implementations from dedicated modules

/// Placeholder for `FusionStrategy` (to be implemented in `model_fusion` module)
#[derive(Debug, Clone)]
pub enum FusionStrategy {
    /// Average
    Average,
    /// Weighted
    Weighted,
    /// Stacking
    Stacking,
}

/// Placeholder for `HierarchicalComposition` (to be implemented in hierarchical module)
pub struct HierarchicalComposition<S = sklears_core::traits::Untrained> {
    _phantom: std::marker::PhantomData<S>,
}

/// Placeholder for `HierarchicalCompositionTrained`
pub struct HierarchicalCompositionTrained;

/// Placeholder for `HierarchicalCompositionBuilder`
pub struct HierarchicalCompositionBuilder;

/// Placeholder for `HierarchicalNode`
pub struct HierarchicalNode;

/// Placeholder for `HierarchicalStrategy`
#[derive(Debug, Clone)]
pub enum HierarchicalStrategy {
    /// TopDown
    TopDown,
    /// BottomUp
    BottomUp,
}

/// Placeholder for `ModelFusion`
pub struct ModelFusion<S = sklears_core::traits::Untrained> {
    _phantom: std::marker::PhantomData<S>,
}

/// Placeholder for `ModelFusionTrained`
pub struct ModelFusionTrained;

/// Placeholder for `ModelFusionBuilder`
pub struct ModelFusionBuilder;
