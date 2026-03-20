//! Gaussian Process models for regression and classification
//!
//! This module provides comprehensive Gaussian Process implementations including basic GP regression
//! and classification, sparse GP methods with inducing points, expectation propagation, variational
//! approaches, multi-output models, Bayesian optimization, and advanced kernel functions. All algorithms
//! have been refactored into focused modules for better maintainability and comply with SciRS2 Policy.

// Core Gaussian Process types and base structures
mod gp_core;
pub use gp_core::{
    GaussianProcess, GaussianProcessConfig, GPModel, GPBuilder,
    GPValidator, GPEstimator, GPPredictor
};

// Basic Gaussian Process regression implementation
mod gp_regression;
pub use gp_regression::{
    GaussianProcessRegressor, GprTrained, GaussianProcessRegressorBuilder,
    RegressionGP, BasicGPRegression, GPRegressionTrainer, RegressionPredictor
};

// Basic Gaussian Process classification implementation
mod gp_classification;
pub use gp_classification::{
    GaussianProcessClassifier, GpcTrained, GaussianProcessClassifierBuilder,
    ClassificationGP, BasicGPClassification, GPClassificationTrainer, ClassificationPredictor
};

// Sparse Gaussian Process methods with inducing points
mod sparse_gp;
pub use sparse_gp::{
    SparseGaussianProcessRegressor, SparseGaussianProcessClassifier,
    SgprTrained, SgpcTrained, SparseGPBuilder, InducingPointsManager,
    SparseApproximation, FITCApproximation, VFEApproximation
};

// Expectation Propagation Gaussian Process implementation
mod expectation_propagation;
pub use expectation_propagation::{
    ExpectationPropagationGaussianProcessClassifier, EpgpcTrained,
    ExpectationPropagationBuilder, EPAlgorithm, EPInference,
    EPOptimizer, EPApproximation, EPCavityDistribution
};

// Variational Gaussian Process methods
mod variational_gp;
pub use variational_gp::{
    VariationalSparseGaussianProcessRegressor, VsgprTrained,
    VariationalGPBuilder, VariationalInference, VariationalOptimizer,
    VariationalApproximation, EvidenceLowerBound, VariationalParameters
};

// Multi-output Gaussian Process models
mod multi_output_gp;
pub use multi_output_gp::{
    MultiOutputGaussianProcessRegressor, MogprTrained,
    MultiOutputGPBuilder, MultiTaskGP, OutputCorrelationKernel,
    CoregionalizationKernel, IndependentOutputs, SharedParameters
};

// Multi-class Gaussian Process classification
mod multi_class_gp;
pub use multi_class_gp::{
    MultiClassGaussianProcessClassifier, McgpcTrained,
    MultiClassGPBuilder, OneVsRestGP, OneVsOneGP,
    MultiClassStrategy, ClassProbabilityEstimator, MultiClassPredictor
};

// Kernel functions and covariance structures
mod kernels;
pub use kernels::{
    Kernel, RBFKernel, MaternKernel, LinearKernel, PolynomialKernel,
    ExponentialKernel, RationalQuadraticKernel, PeriodicKernel,
    WhiteKernel, ConstantKernel, KernelComposition, CompoundKernel
};

// Advanced kernel operations and combinations
mod kernel_operations;
pub use kernel_operations::{
    KernelSum, KernelProduct, KernelScaling, KernelTransformation,
    KernelDerivatives, HyperparameterOptimization, KernelComposer,
    KernelValidator, AdaptiveKernels
};

// Bayesian optimization and acquisition functions
mod bayesian_optimization;
pub use bayesian_optimization::{
    BayesianOptimizer, AcquisitionFunction, ExpectedImprovement,
    ProbabilityOfImprovement, UpperConfidenceBound, EntropySearch,
    KnowledgeGradient, OptimizationStrategy, AcquisitionOptimizer
};

// Hyperparameter optimization and model selection
mod hyperparameter_optimization;
pub use hyperparameter_optimization::{
    HyperparameterOptimizer, MarginalLikelihoodOptimizer, GradientBasedOptimizer,
    BayesianHyperparameterOptimization, CrossValidationOptimizer,
    HyperparameterSearchSpace, OptimizationAlgorithm, ModelSelection
};

// Approximate inference methods
mod approximate_inference;
pub use approximate_inference::{
    ApproximateInference, LaplaceApproximation, VariationalBayes,
    ExpectationPropagationInference, MonteCarloInference,
    StructuredVariationalInference, PowerExpectationPropagation, NaturalGradients
};

// Online and incremental Gaussian Process learning
mod online_gp;
pub use online_gp::{
    OnlineGaussianProcess, IncrementalGP, StreamingGP,
    OnlineLearningStrategy, IncrementalInducing, AdaptiveGP,
    ConceptDriftDetection, OnlineHyperparameterOptimization
};

// Warped and transformed Gaussian Processes
mod warped_gp;
pub use warped_gp::{
    WarpedGaussianProcess, BoxCoxWarpedGP, NeuralNetworkWarpedGP,
    IdentityWarping, LogitWarping, PowerWarping,
    WarpingFunction, NonlinearTransformation, TransformationLearning
};

// Gaussian Process with different likelihoods
mod likelihood_models;
pub use likelihood_models::{
    GaussianLikelihood, BernoulliLikelihood, PoissonLikelihood,
    GammaLikelihood, BetaLikelihood, StudentTLikelihood,
    LikelihoodFunction, NonGaussianLikelihood, LikelihoodApproximation
};

// Deep Gaussian Processes and hierarchical models
mod deep_gp;
pub use deep_gp::{
    DeepGaussianProcess, HierarchicalGP, LayeredGP,
    DeepGPBuilder, DeepGPInference, DeepGPTrainer,
    HierarchicalInference, MultilevelGP, DeepVariationalGP
};

// Gaussian Process latent variable models
mod latent_variable_gp;
pub use latent_variable_gp::{
    GaussianProcessLatentVariableModel, GPLVM, BayesianGPLVM,
    DimensionalityReduction, LatentSpaceOptimization, NonlinearEmbedding,
    ManifoldLearning, LatentDynamics, SharedLatentVariables
};

// Stochastic processes and random fields
mod stochastic_processes;
pub use stochastic_processes::{
    StochasticProcess, GaussianRandomField, MaternRandomField,
    SpatialGP, TemporalGP, SpatioTemporalGP,
    RandomFieldSimulation, ProcessSimulation, FieldInterpolation
};

// Constrained and monotonic Gaussian Processes
mod constrained_gp;
pub use constrained_gp::{
    ConstrainedGaussianProcess, MonotonicGP, ConvexGP,
    InequalityConstraints, EqualityConstraints, ConstraintEnforcement,
    ConstrainedOptimization, FeasibilityConstraints, BoundedGP
};

// Multi-fidelity and hierarchical approximations
mod multi_fidelity_gp;
pub use multi_fidelity_gp::{
    MultiFidelityGaussianProcess, HierarchicalApproximation,
    LowFidelityModels, HighFidelityModels, FidelityCorrection,
    CostAwareOptimization, MultiFidelityAcquisition, FidelitySelection
};

// Gaussian Process utilities and numerical methods
mod gp_utilities;
pub use gp_utilities::{
    GPUtilities, NumericalStability, CholeskyDecomposition, MatrixInversion,
    LinearSolvers, EigenDecomposition, SVDSolver, IterativeSolvers,
    NumericalOptimization, GradientComputation
};

// Prediction and uncertainty quantification
mod prediction;
pub use prediction::{
    GPPredictor, UncertaintyQuantification, PredictiveDistribution,
    ConfidenceIntervals, PredictionIntervals, CalibratedUncertainty,
    BayesianPrediction, EnsemblePrediction, RobustPrediction
};

// Validation and model diagnostics
mod validation;
pub use validation::{
    GPValidator, ModelDiagnostics, CrossValidation, LeaveOneOutValidation,
    ModelComparison, GoodnessOfFit, ResidualAnalysis,
    DiagnosticPlots, ValidationMetrics, QualityAssessment
};

// Performance optimization and computational efficiency
mod performance_optimization;
pub use performance_optimization::{
    GPPerformanceOptimizer, ParallelComputation, MemoryOptimization,
    CacheOptimization, ComputationalComplexity, ScalabilityAnalysis,
    HardwareAcceleration, SIMDOptimization, GPUAcceleration
};

// Visualization and interpretation tools
mod visualization;
pub use visualization::{
    GPVisualizer, UncertaintyVisualization, KernelVisualization,
    PredictionPlots, AcquisitionVisualization, HyperparameterVisualization,
    ConvergencePlots, DiagnosticVisualization, InteractiveVisualization
};

// Testing and benchmarking framework
mod testing_framework;
pub use testing_framework::{
    GPTestSuite, BenchmarkSuite, PerformanceTests, AccuracyTests,
    RobustnessTests, ScalabilityTests, NumericalStabilityTests,
    ValidationTests, ComparisonTests, RegressionTests
};