//! Specialized isotonic regression algorithms
//!
//! This module provides comprehensive specialized isotonic regression implementations including
//! additive models, tensor methods, regularized approaches, feature selection, partial orders,
//! tree structures, lattice orders, and adaptive weighting. All algorithms have been refactored
//! into focused modules for better maintainability and comply with SciRS2 Policy.

// Additive isotonic regression
mod additive_isotonic;
pub use additive_isotonic::{
    AdditiveIsotonicRegression, TrainedAdditiveIsotonicRegression,
    AdditiveConfig, BackfittingAlgorithm
};

// Tensor isotonic regression
mod tensor_isotonic;
pub use tensor_isotonic::{
    TensorIsotonicRegression, TrainedTensorIsotonicRegression,
    TensorConfig, TensorOrder, TensorConstraints
};

// Regularized isotonic regression
mod regularized_isotonic;
pub use regularized_isotonic::{
    RegularizedIsotonicRegression, TrainedRegularizedIsotonicRegression,
    RegularizationType, RegularizationConfig
};

// Feature selection isotonic regression
mod feature_selection_isotonic;
pub use feature_selection_isotonic::{
    FeatureSelectionIsotonicRegression, TrainedFeatureSelectionIsotonicRegression,
    FeatureSelectionCriteria, SelectionAlgorithm
};

// Partial order isotonic regression
mod partial_order_isotonic;
pub use partial_order_isotonic::{
    PartialOrderIsotonicRegression, TrainedPartialOrderIsotonicRegression,
    PartialOrderConstraints, OrderGraph
};

// Tree order isotonic regression
mod tree_order_isotonic;
pub use tree_order_isotonic::{
    TreeOrderIsotonicRegression, TrainedTreeOrderIsotonicRegression,
    TreeStructure, TreeConstraints
};

// Lattice order isotonic regression
mod lattice_order_isotonic;
pub use lattice_order_isotonic::{
    LatticeOrderIsotonicRegression, TrainedLatticeOrderIsotonicRegression,
    LatticeStructure, LatticeConstraints
};

// Adaptive weighting isotonic regression
mod adaptive_weighting_isotonic;
pub use adaptive_weighting_isotonic::{
    AdaptiveWeightingIsotonicRegression, TrainedAdaptiveWeightingIsotonicRegression,
    AdaptiveWeightingConfig, WeightingStrategy
};

// Sparse isotonic regression utilities
mod sparse_isotonic_utils;
pub use sparse_isotonic_utils::{
    sparse_isotonic_regression, SparseIsotonicConfig,
    SparseIsotonicResult
};

// Breakdown point analysis
mod breakdown_analysis;
pub use breakdown_analysis::{
    BreakdownPointAnalysis, BreakdownResult,
    breakdown_point_analysis, RobustnessMetrics
};

// Influence diagnostics
mod influence_diagnostics;
pub use influence_diagnostics::{
    InfluenceDiagnostics, InfluenceResult,
    influence_diagnostics, DiagnosticMetrics
};

// Specialized isotonic utilities
mod specialized_utils;
pub use specialized_utils::{
    IsotonicOptimizer, ConstraintSolver, OrderValidator,
    SpecializedConfig, UtilityFunctions
};