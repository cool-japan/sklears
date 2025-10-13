//! Gaussian Process models for regression and classification
//!
//! This module is part of sklears, providing scikit-learn compatible
//! machine learning algorithms in Rust.

#![warn(missing_docs)]

// Module declarations
pub mod automatic_kernel;
pub mod bayesian_optimization;
pub mod fitc;
pub mod gpc;
pub mod gpr;
pub mod kernel_optimization;
pub mod kernel_selection;
pub mod kernel_structure_learning;
pub mod kernels;
pub mod marginal_likelihood;
pub mod multi_output;
pub mod nystrom;
pub mod random_fourier_features;
pub mod sparse_gpr;
pub mod sparse_spectrum;
pub mod structured_kernel_interpolation;
pub mod utils;
pub mod variational;

// Re-exports for convenient access

// Automatic kernel construction
pub use automatic_kernel::{
    AutomaticKernelConstructor, DataCharacteristics, KernelConstructionResult,
};

// Bayesian optimization
pub use bayesian_optimization::{AcquisitionFunction, BayesianOptimizer, BayesianOptimizerFitted};

// FITC approximation
pub use fitc::{
    FitcGaussianProcessRegressor, FitcGaussianProcessRegressorConfig, FitcGprTrained,
    InducingPointInit as FitcInducingPointInit,
};

// Gaussian Process Regression
pub use gpr::{GaussianProcessRegressor, GprTrained};

// Gaussian Process Classification
pub use gpc::{
    ExpectationPropagationGaussianProcessClassifier, EpGpcTrained,
    GaussianProcessClassifier, GpcConfig, GpcTrained,
    MultiClassGaussianProcessClassifier, McGpcTrained,
    SparseGaussianProcessClassifier, SgpcTrained,
    VariationalGaussianProcessClassifier, VariationalGpcConfig, VariationalGpcTrained,
};

// Kernel optimization
pub use kernel_optimization::OptimizationResult as KernelOptimizationResult;
pub use kernel_optimization::{optimize_kernel_parameters, KernelOptimizer};

// Kernel selection
pub use kernel_selection::{
    select_best_kernel, select_kernel_aic, select_kernel_bic, select_kernel_cv,
    KernelSelectionConfig, KernelSelectionResult, KernelSelector, SelectionCriterion,
};

// Kernel structure learning
pub use kernel_structure_learning::{
    ConvergenceInfo, KernelGrammar, KernelStructureLearner, NonTerminalOperation, SearchStrategy,
    StructureLearningResult, TerminalKernel,
};

// Marginal likelihood optimization
pub use marginal_likelihood::OptimizationResult as MarginalLikelihoodOptimizationResult;
pub use marginal_likelihood::{
    cross_validate_hyperparameters, log_marginal_likelihood, log_marginal_likelihood_stable,
    optimize_hyperparameters, MarginalLikelihoodOptimizer,
};

// Multi-output Gaussian Processes
pub use multi_output::{
    LinearModelCoregionalization, LmcTrained,
    MultiOutputGaussianProcessRegressor, MogprTrained,
};

// Nyström approximation
pub use nystrom::{LandmarkSelection, NystromGaussianProcessRegressor, NystromGprTrained};

// Random Fourier Features
pub use random_fourier_features::{
    RandomFourierFeatures, RffFitted,
    RandomFourierFeaturesGPR, RffGprTrained,
};

// Sparse Gaussian Process Regression
pub use sparse_gpr::{InducingPointInit, SgprTrained, SparseGaussianProcessRegressor};

// Sparse spectrum approximation
pub use sparse_spectrum::{
    SparseSpectrumGaussianProcessRegressor, SparseSpectrumGprTrained, SpectralApproximationInfo,
    SpectralSelectionMethod,
};

// Structured kernel interpolation
pub use structured_kernel_interpolation::{
    GridBoundsMethod, InterpolationMethod, SkiApproximationInfo, SkiGprTrained,
    StructuredKernelInterpolationGPR,
};

// Variational Gaussian Processes
pub use variational::{
    VariationalOptimizer, VariationalSparseGaussianProcessRegressor, VsgprTrained,
};

// Common kernel types
pub use kernels::{Kernel, RBF};

// Common utility functions
pub use utils::{robust_cholesky, triangular_solve, matrix_inverse};