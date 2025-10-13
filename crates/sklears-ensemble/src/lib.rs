#![allow(dead_code)]
#![allow(non_snake_case)]
#![allow(missing_docs)]
#![allow(deprecated)]
#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(unused_mut)]
#![allow(unused_assignments)]
#![allow(unused_doc_comments)]
#![allow(unused_parens)]
#![allow(unused_comparisons)]
//! Ensemble methods for sklears
//!
//! This crate provides implementations of ensemble machine learning algorithms including:
//! - Bagging (Bootstrap Aggregating)
//! - Gradient Boosting
//! - AdaBoost (Adaptive Boosting)
//! - Voting Classifiers/Regressors
//! - Stacking and Blending

pub mod adaboost;
pub mod adversarial;
pub mod analysis;
pub mod bagging;
pub mod compression;
pub mod cpu_optimization;
pub mod gpu_acceleration;
pub mod gradient_boosting;
pub mod imbalanced;
pub mod memory_efficient;
pub mod mixed_precision;
pub mod model_selection;
pub mod monitoring;
pub mod multi_label;
pub mod multi_task;
pub mod parallel;
pub mod regularized;
pub mod simd_ops;
pub mod simd_stacking;
pub mod stacking;
pub mod streaming;
pub mod tensor_ops;
pub mod time_series;
pub mod voting;

pub use adaboost::{AdaBoostAlgorithm, AdaBoostClassifier, AdaBoostConfig};
pub use adversarial::{
    AdversarialEnsembleClassifier, AdversarialEnsembleConfig, AdversarialPredictionResults,
    AdversarialStrategy, AttackMethod, DefensiveStrategy, InputPreprocessing, RobustnessMetrics,
};
pub use analysis::{
    CalibrationMetrics, ConfidenceMetrics, EnsembleAnalyzer, FeatureImportanceAnalysis,
    ImportanceAggregationMethod, ReliabilityDiagram, UncertaintyDecomposition,
    UncertaintyQuantification,
};
pub use bagging::{BaggingClassifier, BaggingConfig, BaggingRegressor};
pub use compression::{
    AcquisitionFunction, BayesianEnsembleOptimizer, CompressedEnsemble, CompressionConfig,
    CompressionMetadata, CompressionStats, CompressionStrategy, EnsembleCompressor, EnsemblePruner,
    KnowledgeDistillationTrainer, QuantizationParams, SparsityInfo,
};
pub use cpu_optimization::{
    CacheOptimizedMatrixOps, CpuOptimizationConfig, CpuOptimizer, LoopOptimizedAlgorithms,
    PerformanceCounters, VectorizedEnsembleOps,
};
pub use gpu_acceleration::{
    detect_available_backends, GpuBackend, GpuConfig, GpuContext, GpuDeviceInfo,
    GpuEnsembleTrainer, GpuTensorOps, ProfilingResults,
};
pub use gradient_boosting::{
    FeatureImportanceMetrics, GradientBoostingClassifier, GradientBoostingConfig,
    GradientBoostingRegressor, GradientBoostingTree, LossFunction,
};
pub use imbalanced::{
    CombinationStrategy, CostSensitiveConfig, ImbalancedEnsembleClassifier,
    ImbalancedEnsembleConfig, SMOTEConfig, SMOTESampler, SamplingQualityMetrics, SamplingResult,
    SamplingStrategy, ThresholdMovingStrategy,
};
pub use memory_efficient::{
    IncrementalLinearRegression, IncrementalModel, MemoryEfficientConfig, MemoryEfficientEnsemble,
};
pub use mixed_precision::{
    AMPContext, GradientScaler, Half, MixedPrecisionArray, MixedPrecisionConfig,
    MixedPrecisionGradientAccumulator, MixedPrecisionTrainer, ScalerState,
};
pub use model_selection::{
    BiasVarianceAnalyzer, BiasVarianceConfig, BiasVarianceDecomposition,
    BiasVarianceEnsembleSizeAnalysis, DiversityAnalyzer, DiversityMetrics, EnsembleCVResults,
    EnsembleCVStrategy, EnsembleConstructionConfig, EnsembleCrossValidator, InterraterReliability,
    ModelSelectionLossFunction, SampleBiasVariance, ScoringMetric,
};
pub use monitoring::{
    DegradationIndicators, DriftDetectionResult, DriftType, EnsembleMonitor, ModelHealth,
    MonitoringConfig, MonitoringResults, PerformanceDataPoint, PerformanceMetric, PerformanceTrend,
    RecommendedAction,
};
pub use multi_label::{
    LabelCorrelationMethod, LabelTransformationStrategy, MultiLabelAggregationMethod,
    MultiLabelEnsembleClassifier, MultiLabelEnsembleConfig, MultiLabelPredictionResults,
    MultiLabelTrainingResults,
};
pub use multi_task::{
    CrossTaskValidation, MultiTaskEnsembleClassifier, MultiTaskEnsembleConfig,
    MultiTaskEnsembleRegressor, MultiTaskFeatureSelector, MultiTaskTrainingResults, TaskData,
    TaskHierarchy, TaskMetrics, TaskSharingStrategy, TaskSimilarityMetric, TaskWeightingStrategy,
};
pub use parallel::{
    AsyncEnsembleCoordinator, DataPartition, FederatedEnsembleCoordinator, ParallelConfig,
    ParallelEnsembleTrainer, ParallelPerformanceMetrics, ParallelStrategy, ParallelTrainable,
};
pub use regularized::{
    DropoutEnsemble, OptimizerState, RegularizationStep, RegularizedEnsembleClassifier,
    RegularizedEnsembleConfig, RegularizedEnsembleRegressor, WeightOptimizer,
};
pub use simd_ops::SimdOps;
pub use simd_stacking::{
    simd_aggregate_predictions, simd_batch_linear_predictions, simd_compute_ensemble_diversity,
    simd_compute_gradients, simd_dot_product, simd_generate_meta_features, simd_linear_prediction,
    simd_train_stacking_ensemble, StackingEnsembleModel,
};
pub use stacking::{
    BaseEstimator, BlendingClassifier, MetaEstimator, MetaFeatureStrategy, MetaLearningStrategy,
    MultiLayerStackingClassifier, MultiLayerStackingConfig, SimpleStackingClassifier,
    StackingClassifier, StackingConfig, StackingLayerConfig,
};
pub use streaming::{
    AdaptiveStreamingEnsemble, ConceptDriftDetector, StreamingConfig, StreamingEnsemble,
};
pub use tensor_ops::{
    ActivationType, AggregationType, ComputationGraph, EnsembleTensorOps, GraphNode, MemoryLayout,
    ReductionType, Tensor, TensorConfig, TensorDevice, TensorOperation, TensorOpsContext,
    TensorShape,
};
pub use time_series::{
    AdwinDriftDetector, DriftAdaptationStrategy, DriftStatistics, SeasonalComponents,
    TemporalAggregationMethod, TimeSeriesCVStrategy, TimeSeriesEnsembleClassifier,
    TimeSeriesEnsembleConfig, TimeSeriesEnsembleRegressor,
};
pub use voting::{
    EnsembleMember, EnsembleSizeAnalysis, EnsembleSizeRecommendations, VotingClassifier,
    VotingClassifierConfig, VotingStrategy,
};

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::adaboost::{AdaBoostAlgorithm, AdaBoostClassifier, AdaBoostConfig};
    pub use crate::adversarial::{
        AdversarialEnsembleClassifier, AdversarialEnsembleConfig, AdversarialPredictionResults,
        AdversarialStrategy, AttackMethod, DefensiveStrategy, InputPreprocessing,
        RobustnessMetrics,
    };
    pub use crate::analysis::{
        CalibrationMetrics, ConfidenceMetrics, EnsembleAnalyzer, FeatureImportanceAnalysis,
        ImportanceAggregationMethod, ReliabilityDiagram, UncertaintyDecomposition,
        UncertaintyQuantification,
    };
    pub use crate::bagging::{BaggingClassifier, BaggingConfig, BaggingRegressor};
    pub use crate::compression::{
        AcquisitionFunction, BayesianEnsembleOptimizer, CompressedEnsemble, CompressionConfig,
        CompressionMetadata, CompressionStats, CompressionStrategy, EnsembleCompressor,
        EnsemblePruner, KnowledgeDistillationTrainer, QuantizationParams, SparsityInfo,
    };
    pub use crate::cpu_optimization::{
        CacheOptimizedMatrixOps, CpuOptimizationConfig, CpuOptimizer, LoopOptimizedAlgorithms,
        PerformanceCounters, VectorizedEnsembleOps,
    };
    pub use crate::gpu_acceleration::{
        detect_available_backends, GpuBackend, GpuConfig, GpuContext, GpuDeviceInfo,
        GpuEnsembleTrainer, GpuTensorOps, ProfilingResults,
    };
    pub use crate::gradient_boosting::{
        FeatureImportanceMetrics, GradientBoostingClassifier, GradientBoostingConfig,
        GradientBoostingRegressor, GradientBoostingTree, LossFunction,
    };
    pub use crate::imbalanced::{
        CombinationStrategy, CostSensitiveConfig, ImbalancedEnsembleClassifier,
        ImbalancedEnsembleConfig, SMOTEConfig, SMOTESampler, SamplingQualityMetrics,
        SamplingResult, SamplingStrategy, ThresholdMovingStrategy,
    };
    pub use crate::memory_efficient::{
        IncrementalLinearRegression, IncrementalModel, MemoryEfficientConfig,
        MemoryEfficientEnsemble,
    };
    pub use crate::mixed_precision::{
        AMPContext, GradientScaler, Half, MixedPrecisionArray, MixedPrecisionConfig,
        MixedPrecisionGradientAccumulator, MixedPrecisionTrainer, ScalerState,
    };
    pub use crate::model_selection::{
        BiasVarianceAnalyzer, BiasVarianceConfig, BiasVarianceDecomposition,
        BiasVarianceEnsembleSizeAnalysis, DiversityAnalyzer, DiversityMetrics, EnsembleCVResults,
        EnsembleCVStrategy, EnsembleConstructionConfig, EnsembleCrossValidator,
        InterraterReliability, ModelSelectionLossFunction, SampleBiasVariance, ScoringMetric,
    };
    pub use crate::monitoring::{
        DegradationIndicators, DriftDetectionResult, DriftType, EnsembleMonitor, ModelHealth,
        MonitoringConfig, MonitoringResults, PerformanceDataPoint, PerformanceMetric,
        PerformanceTrend, RecommendedAction,
    };
    pub use crate::multi_label::{
        LabelCorrelationMethod, LabelTransformationStrategy, MultiLabelAggregationMethod,
        MultiLabelEnsembleClassifier, MultiLabelEnsembleConfig, MultiLabelPredictionResults,
        MultiLabelTrainingResults,
    };
    pub use crate::multi_task::{
        CrossTaskValidation, MultiTaskEnsembleClassifier, MultiTaskEnsembleConfig,
        MultiTaskEnsembleRegressor, MultiTaskFeatureSelector, MultiTaskTrainingResults, TaskData,
        TaskHierarchy, TaskMetrics, TaskSharingStrategy, TaskSimilarityMetric,
        TaskWeightingStrategy,
    };
    pub use crate::parallel::{
        AsyncEnsembleCoordinator, DataPartition, FederatedEnsembleCoordinator, ParallelConfig,
        ParallelEnsembleTrainer, ParallelPerformanceMetrics, ParallelStrategy, ParallelTrainable,
    };
    pub use crate::regularized::{
        DropoutEnsemble, OptimizerState, RegularizationStep, RegularizedEnsembleClassifier,
        RegularizedEnsembleConfig, RegularizedEnsembleRegressor, WeightOptimizer,
    };
    pub use crate::simd_ops::SimdOps;
    pub use crate::simd_stacking::{
        simd_aggregate_predictions, simd_batch_linear_predictions, simd_compute_ensemble_diversity,
        simd_compute_gradients, simd_dot_product, simd_generate_meta_features,
        simd_linear_prediction, simd_train_stacking_ensemble, StackingEnsembleModel,
    };
    pub use crate::stacking::{
        BaseEstimator, BlendingClassifier, MetaEstimator, MetaFeatureStrategy,
        MetaLearningStrategy, MultiLayerStackingClassifier, MultiLayerStackingConfig,
        SimpleStackingClassifier, StackingClassifier, StackingConfig, StackingLayerConfig,
    };
    pub use crate::streaming::{
        AdaptiveStreamingEnsemble, ConceptDriftDetector, StreamingConfig, StreamingEnsemble,
    };
    pub use crate::tensor_ops::{
        ActivationType, AggregationType, ComputationGraph, EnsembleTensorOps, GraphNode,
        MemoryLayout, ReductionType, Tensor, TensorConfig, TensorDevice, TensorOperation,
        TensorOpsContext, TensorShape,
    };
    pub use crate::time_series::{
        AdwinDriftDetector, DriftAdaptationStrategy, DriftStatistics, SeasonalComponents,
        TemporalAggregationMethod, TimeSeriesCVStrategy, TimeSeriesEnsembleClassifier,
        TimeSeriesEnsembleConfig, TimeSeriesEnsembleRegressor,
    };
    pub use crate::voting::{
        EnsembleMember, EnsembleSizeAnalysis, EnsembleSizeRecommendations, VotingClassifier,
        VotingClassifierConfig, VotingStrategy,
    };
}
