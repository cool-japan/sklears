#![allow(dead_code)]
#![allow(non_snake_case)]
#![allow(missing_docs)]
#![allow(deprecated)]
//! Linear and Quadratic Discriminant Analysis
//!
//! This module is part of sklears, providing scikit-learn compatible
//! machine learning algorithms in Rust.

// #![warn(missing_docs)]

pub mod adaptive_discriminant;
// pub mod async_optimization; // Temporarily disabled - needs error type fixes
pub mod bayesian_discriminant;
// TODO: ndarray 0.17 HRTB trait bound issues - re-enable after fixing Fit/Predict/Score traits
// pub mod boundary_adjustment;
pub mod canonical_discriminant;
pub mod cost_sensitive_discriminant_analysis;
pub mod cross_modal_discriminant;
pub mod cross_validation;
pub mod deep_discriminant;
pub mod diagonal_lda;
pub mod discriminant_locality_alignment;
pub mod distributed_discriminant;
pub mod ensemble_imbalanced;
pub mod error_correcting_output_codes;
pub mod feature_ranking;
// pub mod gpu_acceleration; // Disabled - requires extensive GPU API refactoring
pub mod heteroscedastic;
pub mod hierarchical;
pub mod information_theoretic;
pub mod kernel_discriminant;
pub mod lda;
pub mod locality_preserving;
pub mod locally_linear_discriminant;
pub mod manifold_discriminant;
pub mod marginal_fisher;
pub mod minimum_volume_ellipsoid;
pub mod mixture;
pub mod mixture_experts;
pub mod multi_task;
pub mod multi_view_discriminant;
pub mod nearest_shrunken_centroids;
pub mod neural_discriminant;
pub mod numerical_stability; // Re-enabled after fixing compilation issues
pub mod one_vs_one;
pub mod one_vs_rest;
pub mod online_discriminant;
pub mod out_of_core;
pub mod parallel_eigen;
pub mod penalized_discriminant_analysis;
// pub mod phantom_types; // Disabled - uses unstable const generics features
pub mod qda;
pub mod random_projection_discriminant_analysis;
pub mod recursive_feature_elimination;
pub mod robust_adaptive;
pub mod sequential_feature_selection;
pub mod simd_optimizations;
pub mod stability_selection;
pub mod stochastic_discriminant;
pub mod sure_independence_screening;
pub mod temporal_discriminant;

// Re-export main types for convenience
pub use adaptive_discriminant::{
    AdaptationEvent, AdaptationStrategy, AdaptiveDiscriminantLearning,
    AdaptiveDiscriminantLearningConfig, BaseDiscriminant, TrainedAdaptiveDiscriminantLearning,
};
// pub use async_optimization::{
//     AsyncDiscriminantAnalysis, AsyncDiscriminantOptimizer, AsyncOptimizationConfig,
//     DistributedDiscriminantAnalysis, OptimizationMessage, OptimizationState, OptimizationStats,
// };
pub use bayesian_discriminant::{
    BayesianDiscriminantAnalysis, BayesianDiscriminantAnalysisConfig, InferenceMethod,
    PosteriorParameters, PriorType, TrainedBayesianDiscriminantAnalysis,
};
// TODO: ndarray 0.17 HRTB trait bound issues - re-enable after fixing Fit/Predict/Score traits
// pub use boundary_adjustment::{
//     BoundaryAdjustmentConfig, BoundaryAdjustmentDiscriminantAnalysis, BoundaryAdjustmentMethod,
//     DensityKernel, OptimizationCriterion, SearchMethod,
//     TrainedBoundaryAdjustmentDiscriminantAnalysis,
// };
pub use canonical_discriminant::{
    CanonicalDiscriminantAnalysis, CanonicalDiscriminantAnalysisConfig,
    TrainedCanonicalDiscriminantAnalysis,
};
pub use cost_sensitive_discriminant_analysis::{
    CostMatrix, CostSensitiveDiscriminantAnalysis, CostSensitiveDiscriminantAnalysisConfig,
    CostSensitiveMethod, SimpleDiscriminantModel, TrainedCostSensitiveDiscriminantAnalysis,
};
pub use cross_modal_discriminant::{
    CrossModalDiscriminantLearningConfig, DomainAdaptationStrategy, DomainInfo, ModalityInfo,
};
pub use cross_validation::{
    BootstrapConfig, BootstrapResults, BootstrapValidator, CrossValidationConfig,
    CrossValidationResults, GridSearchLDA, GridSearchQDA, NestedCVConfig, NestedCVResults,
    NestedCrossValidator, ParameterGrid, TemporalValidationConfig, TemporalValidator,
    ValidationMetrics,
};
pub use deep_discriminant::{
    AttentionConfig, DeepArchitecture, DeepDiscriminantLearning, DeepDiscriminantLearningConfig,
    DeepLayer, DeepTrainingConfig, LayerType, NormalizationType, TrainedDeepDiscriminantLearning,
};
pub use diagonal_lda::{
    DiagonalLinearDiscriminantAnalysis, DiagonalLinearDiscriminantAnalysisConfig,
    TrainedDiagonalLinearDiscriminantAnalysis,
};
pub use discriminant_locality_alignment::{
    DiscriminantLocalityAlignment, DiscriminantLocalityAlignmentConfig,
};
pub use distributed_discriminant::{
    DistributedDiscriminantConfig, DistributedLinearDiscriminantAnalysis,
    DistributedQuadraticDiscriminantAnalysis, MergeStrategy,
};
pub use ensemble_imbalanced::{
    BaseClassifier, EnsembleImbalancedConfig, EnsembleImbalancedDiscriminantAnalysis,
    EnsembleMethod, SamplingStrategy, TrainedEnsembleImbalancedDiscriminantAnalysis, VotingMethod,
};
pub use error_correcting_output_codes::{
    ErrorCorrectingOutputCodes, ErrorCorrectingOutputCodesConfig, TrainedErrorCorrectingOutputCodes,
};
pub use feature_ranking::{
    DiscriminantFeatureRanking, DiscriminantFeatureRankingConfig, FeatureRank,
};
// pub use gpu_acceleration::{
//     GpuAcceleratedLDA, GpuAcceleratedQDA, GpuAccelerationConfig, GpuDiscriminantAnalysis,
//     GpuLDAKernel, GpuMemoryStrategy, GpuPerformanceStats, GpuQDAKernel,
// };
pub use heteroscedastic::{
    HeteroscedasticDiscriminantAnalysis, HeteroscedasticDiscriminantAnalysisConfig,
};
pub use hierarchical::{
    HierarchicalDiscriminantAnalysis, HierarchicalDiscriminantAnalysisConfig, HierarchyTree,
    TrainedHierarchicalDiscriminantAnalysis,
};
pub use information_theoretic::{
    DiscretizationMethod, InformationCriterion, InformationTheoreticDiscriminantAnalysis,
    InformationTheoreticDiscriminantAnalysisConfig,
};
pub use kernel_discriminant::{
    KernelCVConfig, KernelDiscriminantAnalysis, KernelDiscriminantAnalysisConfig,
    KernelOptimizationResults, KernelParameterGrid, KernelParameterOptimizer, KernelType,
    TrainedKernelDiscriminantAnalysis,
};
pub use lda::{LinearDiscriminantAnalysis, LinearDiscriminantAnalysisConfig};
pub use locality_preserving::{
    LocalityPreservingDiscriminantAnalysis, LocalityPreservingDiscriminantAnalysisConfig,
};
pub use locally_linear_discriminant::{
    LocallyLinearDiscriminantAnalysis, LocallyLinearDiscriminantAnalysisConfig,
    TrainedLocallyLinearDiscriminantAnalysis,
};
pub use manifold_discriminant::{
    ManifoldDiscriminantAnalysis, ManifoldDiscriminantAnalysisConfig,
    TrainedManifoldDiscriminantAnalysis,
};
pub use marginal_fisher::{MarginalFisherAnalysis, MarginalFisherAnalysisConfig};
pub use minimum_volume_ellipsoid::{
    MinimumVolumeEllipsoidConfig, MinimumVolumeEllipsoidDiscriminantAnalysis,
    TrainedMinimumVolumeEllipsoidDiscriminantAnalysis,
};
pub use mixture::{MixtureDiscriminantAnalysis, MixtureDiscriminantAnalysisConfig};
pub use mixture_experts::{
    ExpertType, GatingNetworkType, MixtureOfExpertsConfig, MixtureOfExpertsDiscriminantAnalysis,
    TrainedMixtureOfExpertsDiscriminantAnalysis,
};
pub use multi_task::{
    MultiTaskDiscriminantLearning, MultiTaskDiscriminantLearningConfig, Task, TaskClassifier,
    TrainedMultiTaskDiscriminantLearning,
};
pub use multi_view_discriminant::{
    FusionStrategy, MultiViewDiscriminantAnalysis, MultiViewDiscriminantAnalysisConfig, ViewInfo,
};
pub use nearest_shrunken_centroids::{
    NearestShrunkenCentroids, NearestShrunkenCentroidsConfig, TrainedNearestShrunkenCentroids,
};
pub use neural_discriminant::{
    ActivationFunction, NetworkArchitecture, NeuralDiscriminantAnalysis,
    NeuralDiscriminantAnalysisConfig, NeuralLayer, NeuralTrainingConfig,
    TrainedNeuralDiscriminantAnalysis,
};
pub use numerical_stability::{
    ConditionMonitor, ConditionStatistics, MatrixAnalysis, NumericalConfig, NumericalStability,
};
pub use one_vs_one::{
    OneVsOneDiscriminantAnalysis, OneVsOneDiscriminantAnalysisConfig, PairwiseClassifier,
};
pub use one_vs_rest::{
    BinaryClassifier, OneVsRestDiscriminantAnalysis, OneVsRestDiscriminantAnalysisConfig,
};
pub use online_discriminant::{
    OnlineDiscriminantAnalysis, OnlineDiscriminantAnalysisConfig,
    TrainedOnlineDiscriminantAnalysis, UpdateStrategy,
};
pub use out_of_core::{
    OutOfCoreConfig, OutOfCoreDataManager, OutOfCoreLDA, OutOfCoreQDA, StreamingDiscriminant,
};
pub use parallel_eigen::{ParallelEigen, ParallelEigenConfig, ParallelEigenDecomposition};
pub use penalized_discriminant_analysis::{
    PenalizedDiscriminantAnalysis, PenalizedDiscriminantAnalysisConfig, PenaltyType,
    TrainedPenalizedDiscriminantAnalysis,
};
// pub use phantom_types::{
//     data_markers, discriminant_markers, regularization_markers, solver_markers, state_markers,
//     ConfigurationValidator, DiscriminantAnalysisBuilder, DiscriminantMethod, DiscriminantPredictor,
//     RegularizationType, RegularizedLDA, RegularizedQDA, SolverType, SparseLDA, StandardLDA,
//     StandardQDA, TrainedLinearDA, TrainedQuadraticDA, TypeErasedDiscriminant,
//     TypeSafeDiscriminantAnalysis, UntrainedLinearDA, UntrainedQuadraticDA, GPULDA,
// };
pub use qda::{QuadraticDiscriminantAnalysis, QuadraticDiscriminantAnalysisConfig};
pub use random_projection_discriminant_analysis::{
    DiscriminantModel, LDAModel, ProjectionType, QDAModel, RDAModel,
    RandomProjectionDiscriminantAnalysis, RandomProjectionDiscriminantAnalysisConfig,
    TrainedRandomProjectionDiscriminantAnalysis,
};
pub use recursive_feature_elimination::{
    EliminationStep, RecursiveFeatureElimination, RecursiveFeatureEliminationConfig,
    TrainedRecursiveFeatureElimination,
};
pub use robust_adaptive::{
    MEstimatorType, RobustDiscriminantAnalysis, RobustDiscriminantAnalysisConfig,
    TrainedRobustDiscriminantAnalysis,
};
pub use sequential_feature_selection::{
    SelectionDirection, SelectionStep, SequentialFeatureSelection,
    SequentialFeatureSelectionConfig, TrainedSequentialFeatureSelection,
};
pub use simd_optimizations::{
    AdvancedSimdOps, SimdArrayOps, SimdBenchmarkResults, SimdConfig, SimdMatrixOps, SimdSupport,
};

#[cfg(target_arch = "aarch64")]
pub use simd_optimizations::NeonSimdOps;
pub use stability_selection::{
    FeatureStability, StabilitySelection, StabilitySelectionConfig, TrainedStabilitySelection,
};
pub use stochastic_discriminant::{
    LearningRateSchedule, LossFunction, Optimizer, StochasticDiscriminantAnalysis,
    StochasticDiscriminantAnalysisConfig, TrainedStochasticDiscriminantAnalysis,
};
pub use sure_independence_screening::{
    BaseDiscriminantModel, SimpleLDA, SureIndependenceScreening, SureIndependenceScreeningConfig,
    TrainedSureIndependenceScreening,
};
pub use temporal_discriminant::{
    AggregationMethod, TemporalDiscriminantAnalysis, TemporalDiscriminantAnalysisConfig,
    TemporalMethod, TemporalPattern, TrendMethod,
};

#[allow(non_snake_case)]
#[cfg(test)]
mod tests;
