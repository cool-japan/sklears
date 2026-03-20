#![allow(dead_code)]
#![allow(non_snake_case)]
#![allow(missing_docs)]
#![allow(deprecated)]
#![allow(clippy::all)]
#![allow(clippy::pedantic)]
#![allow(clippy::nursery)]
//! Gaussian mixture models
//!
//! This module provides Gaussian mixture models for clustering and density estimation,
//! including standard EM-based GMM and Bayesian variants that can automatically
//! determine the number of components.

// #![warn(missing_docs)]

// Core modules
pub mod adaptive_streaming;
pub mod advi;
pub mod approximation;
pub mod bayesian;
pub mod common;
pub mod empirical_bayes;
pub mod exponential_family;
pub mod gaussian;
pub mod large_scale;
pub mod mean_field_variational;
pub mod multi_modal;
pub mod nonparametric;
pub mod nuts;
pub mod online;
pub mod optimization_enhancements;
pub mod prior_elicitation;
pub mod prior_sensitivity;
pub mod regularization;
pub mod robust;
pub mod robust_methods;
pub mod spatial;
pub mod stochastic_variational;
pub mod structured_variational;
pub mod student_t;
pub mod time_series;
pub mod variational;
pub mod von_mises_fisher;

// Re-export main types from common
pub use common::{CovarianceMatrices, CovarianceType, InitMethod, ModelSelection};

// Re-export main types from gaussian
pub use gaussian::{GaussianMixture, GaussianMixtureTrained};

// Re-export main types from variational
pub use variational::{VariationalBayesianGMM, VariationalBayesianGMMTrained};

// Re-export main types from mean_field_variational
pub use mean_field_variational::{MeanFieldVariationalGMM, MeanFieldVariationalGMMTrained};

// Re-export main types from stochastic_variational
pub use stochastic_variational::{
    OptimizerType, StochasticVariationalGMM, StochasticVariationalGMMTrained,
};

// Re-export main types from structured_variational
pub use structured_variational::{
    StructuredFamily, StructuredVariationalGMM, StructuredVariationalGMMTrained,
};

// Re-export main types from advi
pub use advi::{ADBackend, ADVIGaussianMixture, ADVIGaussianMixtureTrained, ADVIOptimizer, Dual};

// Re-export main types from empirical_bayes
pub use empirical_bayes::{
    EmpiricalBayesGMM, EmpiricalBayesGMMTrained, EmpiricalBayesMethod, HyperparameterState,
};

// Re-export main types from prior_sensitivity
pub use prior_sensitivity::{
    GridSearchResult, InfluenceScore, ParameterVariances, PerturbationResult,
    PriorSensitivityAnalyzer, SensitivityAnalysisResult, SensitivitySummary,
};

// Re-export main types from prior_elicitation
pub use prior_elicitation::{
    ConstraintType, DomainConstraint, ElicitationAnswer, ElicitationMethod, ElicitationQuestion,
    ElicitationResult, PriorElicitationEngine, PriorQualityMetrics, PriorSpecification,
    QuestionType,
};

// Re-export main types from nuts
pub use nuts::{NUTSResult, NUTSSampler};

// Re-export main types from bayesian
pub use bayesian::{BayesianGaussianMixture, BayesianGaussianMixtureTrained};

// Re-export main types from robust
pub use robust::{RobustGaussianMixture, RobustGaussianMixtureTrained};

// Re-export main types from online
pub use online::{OnlineGaussianMixture, OnlineGaussianMixtureTrained};

// Re-export main types from student_t
pub use student_t::{StudentTMixture, StudentTMixtureTrained};

// Re-export main types from exponential_family
pub use exponential_family::{
    ExponentialFamilyMixture, ExponentialFamilyMixtureTrained, ExponentialFamilyType,
};

// Re-export main types from von_mises_fisher
pub use von_mises_fisher::{VonMisesFisher, VonMisesFisherMixture, VonMisesFisherMixtureFitted};

// Re-export main types from time_series
pub use time_series::{
    DynamicMixture, DynamicMixtureBuilder, DynamicMixtureTrained, HMMConfig, HMMError,
    HiddenMarkovModel, HiddenMarkovModelBuilder, HiddenMarkovModelTrained, ParameterEvolution,
    RSMConfig, RegimeParameters, RegimeSwitchingModel, RegimeSwitchingModelBuilder,
    RegimeSwitchingModelTrained, RegimeType, SSMConfig, SwitchingStateSpaceModel,
    SwitchingStateSpaceModelBuilder, SwitchingStateSpaceModelTrained, TemporalGaussianMixture,
    TemporalGaussianMixtureBuilder, TemporalGaussianMixtureTrained,
};

// Re-export main types from nonparametric
pub use nonparametric::{
    ChineseRestaurantProcess, ChineseRestaurantProcessTrained, DirichletProcessGaussianMixture,
    DirichletProcessGaussianMixtureTrained,
};

// Re-export main types from spatial
pub use spatial::{
    GearysC, GeographicMixture, GeographicMixtureBuilder, GeographicMixtureTrained,
    LocalIndicators, MarkovRandomFieldMixture, MarkovRandomFieldMixtureBuilder,
    MarkovRandomFieldMixtureTrained, MoransI, SpatialAutocorrelationAnalyzer,
    SpatialClusteringQuality, SpatialConstraint, SpatiallyConstrainedGMM,
    SpatiallyConstrainedGMMBuilder, SpatiallyConstrainedGMMTrained,
};

// Re-export main types from multi_modal
pub use multi_modal::{
    FusionStrategy, ModalitySpec, MultiModalConfig, MultiModalGaussianMixture,
    MultiModalGaussianMixtureBuilder, MultiModalGaussianMixtureTrained,
};

// Re-export main types from robust_methods
pub use robust_methods::{
    BreakdownAnalysis, InfluenceDiagnostics, MEstimatorGMM, MEstimatorGMMBuilder,
    MEstimatorGMMTrained, MEstimatorType, TrimmedLikelihoodConfig,
};

// Re-export main types from regularization
pub use regularization::{
    ElasticNetGMM, ElasticNetGMMBuilder, ElasticNetGMMTrained, GroupLassoGMM, GroupLassoGMMBuilder,
    GroupLassoGMMTrained, L1RegularizedGMM, L1RegularizedGMMBuilder, L1RegularizedGMMTrained,
    L2RegularizedGMM, L2RegularizedGMMBuilder, L2RegularizedGMMTrained, RegularizationType,
};

// Re-export main types from optimization_enhancements
pub use optimization_enhancements::{
    AcceleratedEM, AcceleratedEMBuilder, AcceleratedEMTrained, AccelerationType,
    NaturalGradientGMM, NaturalGradientGMMBuilder, NaturalGradientGMMTrained, QuasiNewtonGMM,
    QuasiNewtonGMMBuilder, QuasiNewtonGMMTrained, QuasiNewtonMethod,
};

// Re-export main types from adaptive_streaming
pub use adaptive_streaming::{
    AdaptiveStreamingConfig, AdaptiveStreamingGMM, AdaptiveStreamingGMMBuilder,
    AdaptiveStreamingGMMTrained, CreationCriterion, DeletionCriterion, DriftDetectionMethod,
};

// Re-export main types from large_scale
pub use large_scale::{
    BatchStrategy, MiniBatchGMM, MiniBatchGMMBuilder, MiniBatchGMMTrained, ParallelGMM,
    ParallelGMMBuilder, ParallelGMMTrained, ParallelStrategy,
};

// Re-export main types from approximation
pub use approximation::{
    ImportanceSamplingGMM, ImportanceSamplingGMMBuilder, ImportanceSamplingGMMTrained,
    ImportanceSamplingStrategy, LaplaceGMM, LaplaceGMMBuilder, LaplaceGMMTrained, MonteCarloGMM,
    MonteCarloGMMBuilder, MonteCarloGMMTrained, MonteCarloMethod,
};

// Module stubs for future implementation
// TODO: Extract these from the original lib.rs
// pub mod nonparametric;
// pub mod constrained;
// pub mod mcmc;
// pub mod analysis;
// pub mod algorithms;

// TODO: Re-export additional mixture types as they are extracted:
// pub use online::{OnlineGaussianMixture, OnlineGaussianMixtureTrained};
// And so on...
