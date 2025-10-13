//! Gradient Boosting and ensemble learning utilities
//!
//! This module provides comprehensive gradient boosting implementations including
//! XGBoost-style gradient boosting, LightGBM-inspired optimizations, CatBoost categorical handling,
//! learning-to-rank algorithms, histogram-based training, GOSS (Gradient-based One-Side Sampling),
//! EFB (Exclusive Feature Bundling), Bayesian optimization, regularization techniques,
//! loss function optimization, early stopping, feature importance analysis, and
//! high-performance ensemble learning pipelines. All algorithms have been refactored
//! into focused modules for better maintainability and comply with SciRS2 Policy.

// Core gradient boosting types and base structures
mod gradient_boosting_core;
pub use gradient_boosting_core::{
    GradientBoostingProcessor, GradientBoostingConfig, GradientBoostingValidator, GradientBoostingEstimator,
    GradientBoostingTransformer, GradientBoostingAnalyzer, EnsembleLearner, BoostingEngine
};

// Loss functions and objective optimization
mod loss_functions;
pub use loss_functions::{
    LossFunction, LossCalculator, LossValidator, LeastSquaresLoss,
    LogisticLoss, HuberLoss, QuantileLoss, FocalLoss, MAELoss,
    LogCoshLoss, MultiClassLogLoss, CustomLossFunction, LossOptimizer
};

// Gradient and Hessian computation
mod gradient_computation;
pub use gradient_computation::{
    GradientCalculator, HessianCalculator, GradientValidator, FirstOrderGradient,
    SecondOrderGradient, NumericalGradient, AnalyticalGradient, GradientOptimizer,
    AdaptiveGradient, GradientClipping, GradientAccumulator
};

// Tree ensembles and boosting algorithms
mod tree_ensembles;
pub use tree_ensembles::{
    TreeEnsemble, EnsembleBuilder, EnsembleValidator, WeakLearner,
    TreeBooster, EnsemblePredictor, VotingEnsemble, BaggingEnsemble,
    StackingEnsemble, EnsembleOptimizer, DiversityMeasures
};

// Regularization techniques and overfitting prevention
mod regularization;
pub use regularization::{
    RegularizationConfig, RegularizationValidator, L1Regularization, L2Regularization,
    DropoutConfig, EarlyStoppingConfig, PruningConfig, ShrinkageRegularization,
    RegularizationOptimizer, AdaptiveRegularization, BayesianRegularization
};

// Learning-to-rank algorithms and ranking optimization
mod ranking_algorithms;
pub use ranking_algorithms::{
    RankingData, RankingLossFunctions, RankingValidator, PairwiseRanking,
    ListwiseRanking, LambdaRank, LambdaMART, YetiRank, RankNet,
    ListNet, NDCGOptimizer, RankingEvaluator, RankingAnalyzer
};

// Histogram-based training and efficient algorithms
mod histogram_training;
pub use histogram_training::{
    HistogramBin, FeatureHistogram, HistogramSplit, HistogramValidator,
    HistogramBuilder, BinningStrategy, SparseHistogram, DenseHistogram,
    HistogramOptimizer, QuantileBasedBinning, EqualWidthBinning
};

// GOSS (Gradient-based One-Side Sampling) optimization
mod goss_sampling;
pub use goss_sampling::{
    GOSSConfig, GOSSSampler, GOSSValidator, GradientBasedSampling,
    SampleSelection, ImportanceSampling, AdaptiveSampling, GOSSOptimizer,
    SamplingStrategy, VarianceReduction, BiasCorrection
};

// EFB (Exclusive Feature Bundling) optimization
mod efb_bundling;
pub use efb_bundling::{
    EFBConfig, EFBBundler, EFBValidator, ExclusiveFeatureBundling,
    FeatureBundler, ConflictDetection, BundleOptimization, SparsityAnalyzer,
    FeatureCombination, BundlingStrategy, EFBOptimizer
};

// Categorical feature processing (CatBoost-style)
mod categorical_processing;
pub use categorical_processing::{
    CatBoostCategoricalProcessor, CategoricalValidator, OrderedTargetStatistics, TargetStatistic,
    CategoricalEncoder, MeanTargetEncoding, CountEncoding, FrequencyEncoding,
    CategoricalOptimizer, CategoryAnalyzer, CategoricalFeatureHandler
};

// Bayesian optimization and hyperparameter tuning
mod bayesian_optimization;
pub use bayesian_optimization::{
    BayesianRegularizationConfig, BayesianEnsembleState, BayesianValidator, BayesianOptimizer,
    HyperparameterOptimization, GaussianProcess, AcquisitionFunction, ExpectedImprovement,
    BayesianSearch, BayesianAnalyzer, OptimizationStrategy
};

// Feature importance and interpretability
mod feature_importance;
pub use feature_importance::{
    FeatureImportanceCalculator, ImportanceValidator, GainBasedImportance, SplitBasedImportance,
    PermutationImportance, SHAPValues, TreeExplainer, FeatureContribution,
    ImportanceAnalyzer, GlobalImportance, LocalImportance
};

// Early stopping and convergence monitoring
mod early_stopping;
pub use early_stopping::{
    EarlyStoppingConfig, EarlyStoppingValidator, ConvergenceMonitor, ValidationScoreTracker,
    OverfittingDetector, PatientStopping, AdaptiveStopping, StoppingCriteria,
    EarlyStoppingAnalyzer, PerformanceMonitor, ValidationStrategy
};

// Performance optimization and computational efficiency
mod performance_optimization;
pub use performance_optimization::{
    GradientBoostingPerformanceOptimizer, ComputationalEfficiency, MemoryOptimizer,
    AlgorithmicOptimizer, CacheOptimizer, ParallelGradientBoostingProcessor
};

// Utilities and helper functions
mod gradient_boosting_utilities;
pub use gradient_boosting_utilities::{
    GradientBoostingUtilities, EnsembleMathUtils, BoostingUtils, ValidationUtils,
    ComputationalUtils, HelperFunctions, GradientBoostingAnalysisUtils, UtilityValidator
};

// Re-export main gradient boosting classes for backwards compatibility
pub use tree_ensembles::{GradientBoostingRegressor, GradientBoostingClassifier};
pub use loss_functions::{LossFunction, LossCalculator};
pub use regularization::{DropoutConfig, RegularizationConfig};
pub use ranking_algorithms::{RankingData, RankingLossFunctions};
pub use histogram_training::{HistogramBin, FeatureHistogram};
pub use goss_sampling::{GOSSConfig, GOSSSampler};
pub use efb_bundling::{EFBConfig, EFBBundler};
pub use categorical_processing::CatBoostCategoricalProcessor;
pub use bayesian_optimization::{BayesianRegularizationConfig, BayesianEnsembleState};

// Re-export common configurations and types
pub use gradient_boosting_core::GradientBoostingConfig;
pub use loss_functions::LossFunction;
pub use regularization::{DropoutConfig, RegularizationConfig};
pub use ranking_algorithms::RankingData;
pub use histogram_training::HistogramBin;
pub use goss_sampling::GOSSConfig;
pub use efb_bundling::EFBConfig;
pub use categorical_processing::OrderedTargetStatistics;
pub use bayesian_optimization::BayesianRegularizationConfig;
pub use early_stopping::EarlyStoppingConfig;