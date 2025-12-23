#![allow(dead_code)]
#![allow(non_snake_case)]
#![allow(missing_docs)]
#![allow(deprecated)]
//! Gaussian Process models for regression and classification
//!
//! This module is part of sklears, providing scikit-learn compatible
//! machine learning algorithms in Rust.

// #![warn(missing_docs)]

// Module declarations
pub mod automatic_kernel;
// pub mod batch_bayesian_optimization;
pub mod bayesian_optimization;
pub mod classification;
// pub mod constrained_bayesian_optimization;
pub mod convolution_processes;
pub mod deep_gp;
pub mod features;
pub mod fitc;
pub mod gpr;
pub mod heteroscedastic;
pub mod hierarchical;
pub mod intrinsic_coregionalization;
pub mod kernel_optimization;
pub mod kernel_selection;
pub mod kernel_structure_learning;
pub mod kernel_trait;
pub mod kernels;
pub mod linear_model_coregionalization;
pub mod marginal_likelihood;
// pub mod multi_objective;
pub mod multi_task;
pub mod noise_function_learning;
pub mod nystrom;
pub mod regression;
pub mod robust;
pub mod sparse_gpr;
pub mod sparse_spectrum;
pub mod spatial;
pub mod structured_kernel_interpolation;
// pub mod temporal;
pub mod utils;
pub mod variational;
pub mod variational_deep_gp;

// Re-exports for convenient access
pub use automatic_kernel::{
    AutomaticKernelConstructor, DataCharacteristics, KernelConstructionResult,
};
// pub use batch_bayesian_optimization::{
//     BatchAcquisition, BatchBayesianOptimizer, BatchBayesianOptimizerBuilder, BatchConfig,
//     BatchOptimizationResult, DistanceMetric,
// };
pub use bayesian_optimization::{AcquisitionFunction, BayesianOptimizer, BayesianOptimizerFitted};
pub use classification::{
    sigmoid, sigmoid_derivative, EpGpcTrained, ExpectationPropagationGaussianProcessClassifier,
    GaussianProcessClassifier, GpcConfig, GpcTrained, McGpcTrained,
    MultiClassGaussianProcessClassifier,
};
// pub use constrained_bayesian_optimization::{
//     ConstrainedAcquisition, ConstrainedBayesianOptimizer, ConstrainedBayesianOptimizerBuilder,
//     ConstraintApproximation, ConstraintConfig, ConstraintEvaluation, ConstraintFunction,
//     FeasibilityAnalysis,
// };
// TODO: Re-enable when modules are fully implemented
// pub use convolution_processes::{ConvolutionProcess, ConvolutionProcessTrained};
// pub use deep_gp::{DeepGPConfig, DeepGPLayer, DeepGaussianProcessRegressor};
pub use features::{RandomFourierFeatures, RandomFourierFeaturesGPR, RffGprTrained};
// TODO: Re-enable when FITC is fully implemented
// pub use fitc::{
//     FitcGaussianProcessRegressor, FitcGaussianProcessRegressorConfig, FitcGprTrained,
//     InducingPointInit as FitcInducingPointInit,
// };
pub use gpr::{GaussianProcessRegressor, GprTrained};
pub use heteroscedastic::{
    ConstantNoise, HeteroscedasticGaussianProcessRegressor, LearnableNoiseFunction, LinearNoise,
    NeuralNetworkNoise, NoiseFunction, PolynomialNoise, Trained as HeteroscedasticGprTrained,
};
pub use hierarchical::{HierarchicalGPConfig, HierarchicalGaussianProcessRegressor};
pub use intrinsic_coregionalization::{IcmTrained, IntrinsicCoregionalizationModel};
pub use kernel_optimization::OptimizationResult as KernelOptimizationResult;
pub use kernel_optimization::{optimize_kernel_parameters, KernelOptimizer};
pub use linear_model_coregionalization::{LinearModelCoregionalization, LmcTrained};
// TODO: Re-enable when kernel_selection is fully implemented
// pub use kernel_selection::{
//     select_best_kernel, select_kernel_aic, select_kernel_bic, select_kernel_cv,
//     KernelSelectionConfig, KernelSelectionResult, KernelSelector, SelectionCriterion,
// };
pub use kernel_structure_learning::{
    ConvergenceInfo, KernelGrammar, KernelStructureLearner, NonTerminalOperation, SearchStrategy,
    StructureLearningResult, TerminalKernel,
};
pub use marginal_likelihood::OptimizationResult as MarginalLikelihoodOptimizationResult;
pub use marginal_likelihood::{
    cross_validate_hyperparameters, log_marginal_likelihood, log_marginal_likelihood_stable,
    optimize_hyperparameters, MarginalLikelihoodOptimizer,
};
// pub use multi_objective::{
//     MultiObjectiveAcquisition, MultiObjectiveBayesianOptimizer, MultiObjectiveConfig,
//     ParetoFrontier, ScalarizationMethod, Trained as MultiObjectiveTrained,
// };
pub use multi_task::{MtgpTrained, MultiTaskGaussianProcessRegressor};
pub use noise_function_learning::{
    AdaptiveRegularization, AutomaticNoiseFunctionSelector, CombinationMethod,
    EnsembleNoiseFunction, InformationCriterion, NoiseFunctionEvaluation,
};
pub use nystrom::{LandmarkSelection, NystromGaussianProcessRegressor, NystromGprTrained};
pub use regression::{
    MogprTrained, MultiOutputGaussianProcessRegressor, VariationalOptimizer,
    VariationalSparseGaussianProcessRegressor, VsgprTrained,
};
pub use robust::{
    OutlierDetectionMethod, RobustGPConfig, RobustGaussianProcessRegressor, RobustLikelihood,
    RobustnessMetrics, Trained as RobustGprTrained,
};
pub use sparse_gpr::{InducingPointInit, SgprTrained, SparseGaussianProcessRegressor};
pub use sparse_spectrum::{
    SparseSpectrumGaussianProcessRegressor, SparseSpectrumGprTrained, SpectralApproximationInfo,
    SpectralSelectionMethod,
};
pub use spatial::{
    KrigingType, SpatialGPConfig, SpatialGaussianProcessRegressor, SpatialKernel,
    Trained as SpatialGprTrained, Variogram,
};
pub use structured_kernel_interpolation::{
    GridBoundsMethod, InterpolationMethod, SkiApproximationInfo, SkiGprTrained,
    StructuredKernelInterpolationGPR,
};
// pub use temporal::{
//     SeasonalDecomposition, StateSpaceModel, TemporalGPConfig, TemporalGaussianProcessRegressor,
//     TemporalKernel, Trained as TemporalGprTrained,
// };
// TODO: Re-enable when variational is fully implemented
// pub use variational::{
//     SparseGaussianProcessClassifier, SgpcTrained, VariationalGaussianProcessClassifier,
//     VariationalGpcConfig, VariationalGpcTrained,
// };
pub use variational_deep_gp::{
    VariationalDeepGPBuilder, VariationalDeepGPConfig, VariationalDeepGaussianProcess,
    VariationalLayerConfig, VariationalLayerParameters, VariationalLikelihood,
};

// Re-export common kernel types
pub use kernels::{Kernel, ARDRBF, RBF};
