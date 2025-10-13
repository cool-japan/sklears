#![allow(dead_code)]
#![allow(non_snake_case)]
#![allow(missing_docs)]
#![allow(deprecated)]
//! Linear models for sklears
//!
//! This crate provides implementations of linear models including:
//! - Linear Regression (OLS, Ridge, Lasso)
//! - Logistic Regression
//! - Generalized Linear Models
//!
//! These implementations leverage scirs2's linear algebra and optimization capabilities.

#[cfg(feature = "admm")]
pub mod admm;
#[cfg(feature = "gpu")]
pub mod advanced_gpu_acceleration;
#[cfg(feature = "bayesian")]
// TODO: Migrate to scirs2-linalg (uses ndarray_linalg::Solve)
//pub mod bayesian;
pub mod builder_enhancements;
#[cfg(feature = "feature-selection")]
pub mod categorical_encoding;
#[cfg(feature = "chunked-processing")]
pub mod chunked_processing;
#[cfg(feature = "diagnostics")]
pub mod classification_diagnostics;
#[cfg(feature = "constrained-optimization")]
// TODO: Migrate to scirs2-linalg (uses nalgebra types)
//pub mod constrained_optimization;
#[cfg(feature = "convergence-analysis")]
pub mod convergence_visualization;
#[cfg(feature = "coordinate-descent")]
pub mod coordinate_descent;
#[cfg(feature = "cross-validation")]
pub mod cross_validation;
#[cfg(feature = "early-stopping")]
pub mod early_stopping;
#[cfg(feature = "elastic-net")]
pub mod elastic_net_cv;
pub mod errors;
#[cfg(feature = "feature-selection")]
pub mod feature_scaling;
#[cfg(feature = "feature-selection")]
pub mod feature_selection;
#[cfg(feature = "glm")]
// TODO: Migrate to scirs2-linalg (uses ndarray_linalg::Solve)
//pub mod glm;
#[cfg(feature = "gpu")]
pub mod gpu_acceleration;
#[cfg(feature = "huber")]
// TODO: Migrate to scirs2-linalg (uses ndarray_linalg::Solve)
//pub mod huber;
#[cfg(feature = "huber")]
pub mod irls;
#[cfg(feature = "lasso")]
pub mod lars;
#[cfg(feature = "lasso")]
pub mod lasso_cv;
#[cfg(feature = "lasso")]
pub mod lasso_lars;
#[cfg(feature = "linear-regression")]
pub mod linear_regression;
#[cfg(feature = "logistic-regression")]
pub mod logistic_regression;
#[cfg(feature = "logistic-regression")]
pub mod logistic_regression_cv;
#[cfg(feature = "memory-mapping")]
pub mod memory_efficient_ops;
#[cfg(feature = "memory-mapping")]
pub mod mmap_arrays;
#[cfg(any(feature = "multi-task", feature = "all-algorithms"))]
// TODO: Migrate to scirs2-linalg (uses nalgebra types)
//pub mod multi_output_regression;
#[cfg(feature = "multi-task-elastic-net")]
pub mod multi_task_elastic_net;
#[cfg(feature = "multi-task-elastic-net")]
pub mod multi_task_elastic_net_cv;
#[cfg(feature = "multi-task")]
pub mod multi_task_feature_selection;
#[cfg(feature = "multi-task-lasso")]
pub mod multi_task_lasso;
#[cfg(feature = "multi-task-lasso")]
pub mod multi_task_lasso_cv;
#[cfg(feature = "multi-task")]
pub mod multi_task_shared_representation;
#[cfg(feature = "lasso")]
pub mod omp;
#[cfg(feature = "online-learning")]
pub mod online_learning;
pub mod optimizer;
#[cfg(feature = "sgd")]
pub mod passive_aggressive;
#[cfg(feature = "regularization-path")]
pub mod paths;
#[cfg(feature = "sgd")]
pub mod perceptron;
#[cfg(feature = "feature-selection")]
pub mod polynomial_features;
#[cfg(feature = "quantile-regression")]
// TODO: Migrate to scirs2-linalg (uses ndarray_linalg::Solve)
//pub mod quantile;
#[cfg(feature = "ransac")]
pub mod ransac;
#[cfg(feature = "feature-selection")]
pub mod recursive_feature_elimination;
#[cfg(feature = "residual-analysis")]
pub mod residual_analysis;
#[cfg(feature = "ridge")]
pub mod ridge_classifier;
#[cfg(feature = "ridge")]
pub mod ridge_cv;
#[cfg(feature = "serde")]
// TODO: Migrate to scirs2-linalg (uses nalgebra types)
//pub mod serialization;
#[cfg(feature = "sgd")]
pub mod sgd;
// #[cfg(feature = "simd")]
// TODO: Migrate to scirs2-linalg (uses nalgebra types)
//pub mod simd_optimizations;
pub mod solver;
#[cfg(feature = "sparse")]
pub mod sparse;
#[cfg(feature = "sparse")]
pub mod sparse_linear_regression;
#[cfg(feature = "sparse")]
pub mod sparse_regularized;
#[cfg(feature = "feature-selection")]
pub mod stability_selection;
#[cfg(feature = "streaming")]
pub mod streaming_algorithms;
pub mod utils;

// New modular framework modules
pub mod large_scale_variational_inference;
pub mod loss_functions;
pub mod modular_framework;
pub mod regularization_schemes;
pub mod solver_implementations;
pub mod type_safety;
pub mod uncertainty_quantification;

#[cfg(feature = "theil-sen")]
// TODO: Migrate to scirs2-linalg (uses ndarray_linalg::Solve)
//pub mod theil_sen;

//#[allow(non_snake_case)]
#[cfg(test)]
//pub mod advanced_property_tests;
#[cfg(feature = "admm")]
pub use admm::{AdmmConfig, AdmmSolution, AdmmSolver};

#[cfg(feature = "bayesian")]
// TODO: Migrate to scirs2-linalg (uses ndarray_linalg::Solve)
// pub use bayesian::{
//     ARDRegression, ARDRegressionConfig, BayesianRidge, BayesianRidgeConfig,
//     VariationalBayesianConfig, VariationalBayesianRegression,
// };
pub use builder_enhancements::{
    EnhancedLinearRegressionBuilder, ModelPreset, ModelValidation, ValidationConfig,
};

#[cfg(feature = "logistic-regression")]
pub use builder_enhancements::EnhancedLogisticRegressionBuilder;

#[cfg(feature = "feature-selection")]
pub use categorical_encoding::{
    CategoricalEncoder, CategoricalEncodingConfig, CategoricalEncodingResult,
    CategoricalEncodingStrategy, CategoricalFeatureInfo, UnknownHandling,
};

#[cfg(feature = "chunked-processing")]
pub use chunked_processing::{
    ChunkProcessingConfig, ChunkProcessingResult, ChunkedDataIterator, ChunkedLinearRegression,
    ChunkedMatrixProcessor, ChunkedProcessingUtils, ChunkedProcessor, MemoryStats,
    ParallelChunkedProcessor,
};

#[cfg(feature = "diagnostics")]
pub use classification_diagnostics::{
    CalibrationResult, ClassImbalanceResult, ClassificationDiagnostics,
    ClassificationDiagnosticsConfig, DecisionBoundaryResult, FeatureImportanceMethod,
    FeatureImportanceResult,
};

#[cfg(feature = "constrained-optimization")]
// TODO: Migrate to scirs2-linalg (uses nalgebra types)
// pub use constrained_optimization::{
//     ConstrainedLinearRegression, ConstrainedOptimizationBuilder, ConstrainedOptimizationConfig,
//     ConstrainedOptimizationProblem, ConstrainedOptimizationResult, ConstraintType,
//     InteriorPointSolver,
// };
#[cfg(feature = "convergence-analysis")]
pub use convergence_visualization::{
    ComparisonResult, ConvergenceAnalysis, ConvergenceConfig, ConvergenceCriteria,
    ConvergenceCriterion, ConvergenceMetric, ConvergenceReport, ConvergenceStatus,
    ConvergenceTracker, MetricHistory, MetricSummary, PlotData,
};

#[cfg(feature = "coordinate-descent")]
pub use coordinate_descent::{CoordinateDescentSolver, ValidationInfo};

#[cfg(feature = "cross-validation")]
pub use cross_validation::{
    cross_validate_with_early_stopping, CVStrategy, CrossValidationResult,
    CrossValidatorWithEarlyStopping, StratifiedKFold,
};

#[cfg(feature = "early-stopping")]
pub use early_stopping::{
    train_validation_split, EarlyStopping, EarlyStoppingCallback, EarlyStoppingConfig,
    StoppingCriterion,
};

#[cfg(feature = "elastic-net")]
pub use elastic_net_cv::{ElasticNetCV, ElasticNetCVConfig};
pub use errors::{
    ConfigurationError, ConfigurationErrorKind, ConvergenceInfo, CrossValidationError,
    CrossValidationErrorKind, DataError, DataErrorKind, ErrorBuilder, ErrorSeverity, FeatureError,
    FeatureErrorKind, FoldInfo, LinearModelError, MatrixError, MatrixErrorKind, MatrixInfo,
    NumericalError, NumericalErrorKind, OptimizationError, OptimizationErrorKind, ResourceError,
    ResourceErrorKind, ResourceInfo, StateError, StateErrorKind,
};
#[cfg(feature = "feature-selection")]
pub use feature_scaling::{
    FeatureScaler, FeatureScalerBuilder, FeatureScalingConfig, FeatureStats, PowerTransformMethod,
    ScalingMethod,
};
#[cfg(feature = "feature-selection")]
pub use feature_selection::{
    FeatureScore, FeatureSelectionConfig, FeatureSelector, ModelBasedEstimator, UnivariateScoreFunc,
};
#[cfg(feature = "glm")]
// TODO: Migrate to scirs2-linalg (uses ndarray_linalg::Solve)
// pub use glm::{Family, GLMConfig, GeneralizedLinearModel, Link};
#[cfg(feature = "huber")]
// TODO: Migrate to scirs2-linalg (uses ndarray_linalg::Solve)
// pub use huber::{HuberRegressor, HuberRegressorConfig};
#[cfg(feature = "huber")]
// TODO: Migrate to scirs2-linalg (uses ndarray_linalg::Solve)
// pub use irls::{IRLSConfig, IRLSEstimator, IRLSResult, ScaleEstimator, WeightFunction};
#[cfg(feature = "lasso")]
pub use lars::{Lars, LarsConfig};
#[cfg(feature = "lasso")]
pub use lasso_cv::{LassoCV, LassoCVConfig};
#[cfg(feature = "lasso")]
pub use lasso_lars::{LassoLars, LassoLarsConfig};
#[cfg(feature = "linear-regression")]
pub use linear_regression::{LinearRegression, LinearRegressionConfig};
#[cfg(feature = "logistic-regression")]
pub use logistic_regression::{LogisticRegression, LogisticRegressionConfig};
#[cfg(feature = "logistic-regression")]
pub use logistic_regression_cv::{LogisticRegressionCV, LogisticRegressionCVConfig};
#[cfg(feature = "memory-mapping")]
pub use memory_efficient_ops::{
    MemoryEfficiencyConfig, MemoryEfficientCoordinateDescent, MemoryEfficientOps, MemoryOperation,
    NormType,
};
#[cfg(feature = "memory-mapping")]
pub use mmap_arrays::{
    MmapAdvice, MmapConfig, MmapMatrix, MmapMatrixMut, MmapUtils, MmapVector, MmapVectorMut,
};
#[cfg(any(feature = "multi-task", feature = "all-algorithms"))]
// TODO: Migrate to scirs2-linalg (uses nalgebra types)
// pub use multi_output_regression::{
//     MultiOutputConfig, MultiOutputRegression, MultiOutputRegressionBuilder, MultiOutputResult,
//     MultiOutputStrategy,
// };
#[cfg(feature = "multi-task-elastic-net")]
pub use multi_task_elastic_net::{MultiTaskElasticNet, MultiTaskElasticNetConfig};
#[cfg(feature = "multi-task-elastic-net")]
pub use multi_task_elastic_net_cv::{MultiTaskElasticNetCV, MultiTaskElasticNetCVConfig};
#[cfg(feature = "multi-task")]
pub use multi_task_feature_selection::{
    FeatureSelectionResult, FeatureSelectionStrategy, MultiTaskFeatureSelectionConfig,
    MultiTaskFeatureSelector, SelectionSummary,
};
#[cfg(feature = "multi-task-lasso")]
pub use multi_task_lasso::{MultiTaskLasso, MultiTaskLassoConfig};
#[cfg(feature = "multi-task-lasso")]
pub use multi_task_lasso_cv::{MultiTaskLassoCV, MultiTaskLassoCVConfig};
#[cfg(feature = "multi-task")]
pub use multi_task_shared_representation::{
    MultiTaskSharedRepresentation, SharedRepresentationBuilder, SharedRepresentationConfig,
    SharedRepresentationStrategy,
};
#[cfg(feature = "lasso")]
pub use omp::{OrthogonalMatchingPursuit, OrthogonalMatchingPursuitConfig};
#[cfg(feature = "online-learning")]
pub use online_learning::{
    LearningRateSchedule, MiniBatchConfig, MiniBatchIterator, OnlineCoordinateDescent,
    OnlineLearningConfig, OnlineLinearRegression, OnlineLogisticRegression, SGDVariant,
};
pub use optimizer::{
    FistaOptimizer, LbfgsOptimizer, NesterovAcceleratedGradient, ProximalGradientOptimizer,
    SagOptimizer, SagaOptimizer,
};
#[cfg(feature = "sgd")]
pub use passive_aggressive::{
    PassiveAggressiveClassifier, PassiveAggressiveClassifierConfig, PassiveAggressiveLoss,
    PassiveAggressiveRegressor, PassiveAggressiveRegressorConfig,
};
#[cfg(feature = "sgd")]
pub use perceptron::{Perceptron, PerceptronConfig, PerceptronPenalty};
#[cfg(feature = "feature-selection")]
pub use polynomial_features::{
    FeatureInfo, PolynomialConfig, PolynomialFeatures, PolynomialFeaturesBuilder, PolynomialUtils,
};
#[cfg(feature = "quantile-regression")]
// TODO: Migrate to scirs2-linalg (uses ndarray_linalg::Solve)
// pub use quantile::{QuantileRegressor, QuantileRegressorConfig, QuantileSolver, SolverOptions};
#[cfg(feature = "ransac")]
pub use ransac::{RANSACLoss, RANSACRegressor, RANSACRegressorConfig};
#[cfg(feature = "feature-selection")]
pub use recursive_feature_elimination::{
    RFEConfig, RFEEstimator, RFEFeatureInfo, RFEResult, RecursiveFeatureElimination, ScoringMetric,
};
#[cfg(feature = "residual-analysis")]
pub use residual_analysis::{
    AssumptionResult, AssumptionTests, InfluenceMeasures, OutlierAnalysis, ResidualAnalysisConfig,
    ResidualAnalysisResult, ResidualAnalyzer, ResidualStats, StatisticalTests, TestResult,
};
#[cfg(feature = "ridge")]
pub use ridge_classifier::{RidgeClassifier, RidgeClassifierConfig};
#[cfg(feature = "ridge")]
pub use ridge_cv::{RidgeCV, RidgeCVConfig};
#[cfg(feature = "serde")]
// TODO: Migrate to scirs2-linalg (uses nalgebra types)
// pub use serialization::{
//     ModelMetadata, ModelRegistry, ModelSerializer, ModelVersioning, PerformanceMetrics,
//     SerializableConstrainedOptimization, SerializableLassoRegression, SerializableLinearRegression,
//     SerializableMatrix, SerializableModel, SerializableMultiOutputRegression,
//     SerializableRidgeRegression, SerializableVector, SerializationFormat, TrainingInfo,
// };
#[cfg(feature = "sgd")]
pub use sgd::{
    SGDClassifier, SGDClassifierConfig, SGDLoss, SGDPenalty, SGDRegressor, SGDRegressorConfig,
};
// #[cfg(feature = "simd")]
// TODO: Migrate to scirs2-linalg (uses nalgebra types)
// pub use simd_optimizations::{
//     SimdConfig, SimdCoordinateDescent, SimdFeatures, SimdLinearRegression, SimdOps,
// };
#[cfg(feature = "theil-sen")]
// TODO: Migrate to scirs2-linalg (uses ndarray_linalg::Solve)
// pub use theil_sen::{TheilSenRegressor, TheilSenRegressorConfig};

// Exports for new modular framework
pub use modular_framework::{
    create_modular_linear_regression, BayesianPredictionProvider, CompositeObjective,
    LinearPredictionProvider, ModularConfig, ModularFramework, ModularLinearModel, Objective,
    ObjectiveData, ObjectiveMetadata, OptimizationResult, OptimizationSolver, PredictionProvider,
    PredictionWithConfidence, PredictionWithUncertainty, ProbabilisticPredictionProvider,
    SolverInfo, SolverRecommendations,
};
pub use solver::Solver;
#[cfg(feature = "feature-selection")]
pub use stability_selection::{
    BaseSelector, BootstrapResult, StabilityPath, StabilitySelection, StabilitySelectionConfig,
    StabilitySelectionResult,
};
#[cfg(feature = "streaming")]
pub use streaming_algorithms::{
    DataStreamIterator, StreamingConfig, StreamingLasso, StreamingLinearRegression,
    StreamingLinearRegressionBuilder, StreamingUtils,
};

pub use loss_functions::{
    AbsoluteLoss, EpsilonInsensitiveLoss, HingeLoss, HuberLoss, LogisticLoss, LossFactory,
    QuantileLoss, SquaredHingeLoss, SquaredLoss,
};

pub use regularization_schemes::{
    CompositeRegularization, ElasticNetRegularization, GroupLassoRegularization, L1Regularization,
    L2Regularization, RegularizationFactory,
};

pub use modular_framework::Regularization;

pub use solver_implementations::{
    BacktrackingConfig, CoordinateDescentConfig, CoordinateDescentResult, GradientDescentConfig,
    GradientDescentResult, GradientDescentSolver, LineSearchConfig, ProximalGradientConfig,
    ProximalGradientResult, ProximalGradientSolver, SolverFactory,
};

pub use type_safety::{
    problem_type, solver_capability, ComputationalComplexity, ConfigurationHints,
    ConfigurationValidator, FeatureValidator, FixedSizeOps, L1Scheme, L2Scheme,
    LargeLinearRegression, MediumLinearRegression, MemoryRequirements, RegularizationConstraint,
    RegularizationScheme, SmallLinearRegression, SolverConstraint, Trained, TypeSafeConfig,
    TypeSafeFit, TypeSafeLinearModel, TypeSafeModelBuilder, TypeSafePredict,
    TypeSafeSolverSelector, Untrained,
};

pub use large_scale_variational_inference::{
    ARDConfiguration, LargeScaleVariationalConfig, LargeScaleVariationalRegression,
    LearningRateDecay, PriorConfiguration, VariationalPosterior,
};

pub use uncertainty_quantification::{
    CalibrationMetrics, UncertaintyCapable, UncertaintyConfig, UncertaintyMethod,
    UncertaintyQuantifier, UncertaintyResult,
};

// Re-export path functions
#[cfg(feature = "regularization-path")]
pub use paths::{
    enet_path, enet_path_enhanced, lars_path, lars_path_gram, lasso_path, ElasticNetPathConfig,
    ElasticNetPathResult,
};

// Re-export utility functions
pub use crate::utils::{
    accurate_condition_number, adaptive_least_squares, condition_number,
    diagnose_numerical_stability, enhanced_ridge_regression, orthogonal_mp, orthogonal_mp_gram,
    qr_ridge_regression, rank_revealing_qr, ridge_regression, solve_with_iterative_refinement,
    stable_normal_equations, stable_ridge_regression, svd_ridge_regression, NumericalDiagnostics,
};

// Re-export sparse matrix functionality
#[cfg(feature = "sparse")]
pub use sparse::{
    Either, SparseConfig, SparseCoordinateDescentSolver, SparseMatrix, SparseMatrixCSR,
    SparsityAnalysis,
};

#[cfg(feature = "sparse")]
pub use sparse_linear_regression::{SparseLinearRegression, SparseLinearRegressionConfig};

#[cfg(feature = "sparse")]
pub use sparse_regularized::{
    SparseElasticNet, SparseElasticNetConfig, SparseLasso, SparseLassoConfig,
};

// Disabled error functions are defined inline when sparse feature is not enabled
#[cfg(not(feature = "sparse"))]
pub fn sparse_feature_disabled_error() -> sklears_core::error::SklearsError {
    sklears_core::error::SklearsError::InvalidParameter {
        name: "sparse".to_string(),
        reason: "Sparse matrix support requires the 'sparse' feature".to_string(),
    }
}

#[cfg(not(feature = "sparse"))]
pub fn sparse_linear_regression_disabled_error() -> sklears_core::error::SklearsError {
    sklears_core::error::SklearsError::InvalidParameter {
        name: "sparse-linear-regression".to_string(),
        reason: "Sparse linear regression requires the 'sparse' feature".to_string(),
    }
}

#[cfg(not(feature = "sparse"))]
pub fn sparse_regularized_disabled_error() -> sklears_core::error::SklearsError {
    sklears_core::error::SklearsError::InvalidParameter {
        name: "sparse-regularized".to_string(),
        reason: "Sparse regularized models require the 'sparse' feature".to_string(),
    }
}

/// Penalty types for regularized models
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum Penalty {
    /// No regularization
    #[default]
    None,
    /// L1 regularization (Lasso)
    L1(f64),
    /// L2 regularization (Ridge)
    L2(f64),
    /// Elastic Net (L1 + L2)
    ElasticNet { l1_ratio: f64, alpha: f64 },
}
