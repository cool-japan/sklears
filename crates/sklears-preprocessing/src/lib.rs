#![allow(non_snake_case)]
#![allow(missing_docs)]
#![allow(deprecated)]
#![allow(clippy::all)]
#![allow(clippy::pedantic)]
#![allow(clippy::nursery)]
//! Preprocessing utilities for sklears
//!
//! This crate provides data preprocessing utilities including:
//! - Scaling (StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, Normalizer)
//! - Encoding (LabelEncoder, OneHotEncoder, OrdinalEncoder)
//! - Imputation (SimpleImputer, KNNImputer, IterativeImputer, GAINImputer)
//! - Feature engineering (PolynomialFeatures, SplineTransformer, PowerTransformer, FunctionTransformer)
//! - Text processing (TfIdfVectorizer, TextTokenizer, NgramGenerator, TextSimilarity, BagOfWordsEmbedding)
//! - Advanced pipelines (conditional steps, parallel branches, caching, dynamic construction)

#![allow(dead_code)]
#![allow(clippy::manual_clamp)]
#![allow(clippy::single_char_add_str)]
#![allow(clippy::let_and_return)]
#![allow(clippy::map_clone)]
#![allow(clippy::manual_slice_size_calculation)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::if_same_then_else)]
#![allow(clippy::arc_with_non_send_sync)]
#![allow(clippy::excessive_precision)]
#![allow(clippy::type_complexity)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::only_used_in_recursion)]
#![allow(clippy::new_without_default)]
#![allow(clippy::derivable_impls)]
#![allow(clippy::ptr_arg)]

pub mod adaptive;
pub mod automated_feature_engineering;
pub mod binarization;
pub mod column_transformer;
pub mod cross_validation;
pub mod data_quality;
pub mod dimensionality_reduction;
pub mod encoding;
pub mod feature_engineering;
pub mod feature_union;
pub mod functional;
pub mod geospatial;
// TODO: Depends on scirs2_core::memory::BufferPool which doesn't exist yet
// pub mod gpu_acceleration;
pub mod image_preprocessing;
pub mod imputation;
pub mod information_theory;
pub mod kernel_centerer;
pub mod label_binarization;
// TODO: Depends on scirs2_core::memory::BufferPool which doesn't exist yet
// pub mod lazy_evaluation;
// TODO: Depends on scirs2_core::memory::BufferPool which doesn't exist yet
// pub mod memory_management;
pub mod monitoring;
pub mod outlier_detection;
pub mod outlier_transformation;
pub mod pipeline;
pub mod pipeline_validation;
pub mod probabilistic_imputation;
pub mod quantile_transformer;
pub mod robust_preprocessing;
pub mod scaling;
pub mod simd_optimizations;
pub mod sparse_optimizations;
pub mod streaming;
pub mod temporal;
pub mod text;
pub mod type_safety;
pub mod winsorization;

pub use adaptive::{
    AdaptationStrategy, AdaptiveConfig, AdaptiveParameterSelector, DataCharacteristics,
    DistributionType, ImputationParameters, OutlierDetectionParameters, ParameterEvaluation,
    ParameterRecommendations, ScalingParameters,
    TransformationParameters as AdaptiveTransformationParameters,
};
pub use automated_feature_engineering::{
    AutoFeatureConfig, AutoFeatureEngineer, AutoFeatureEngineerFitted, Domain, GenerationStrategy,
    MathFunction, SelectionMethod, TransformationFunction, TransformationType,
};
pub use binarization::{
    Binarizer, BinarizerConfig, DiscretizationStrategy, DiscretizerEncoding, KBinsDiscretizer,
    KBinsDiscretizerConfig,
};
pub use column_transformer::{
    ColumnSelector, ColumnTransformer, ColumnTransformerConfig, DataType, RemainderStrategy,
    TransformerStep, TransformerWrapper,
};
pub use cross_validation::{
    CVScore, InformationPreservationMetric, KFold, ParameterDistribution, ParameterGrid,
    PreprocessingMetric, StratifiedKFold, VariancePreservationMetric,
};
pub use data_quality::{
    CorrelationWarning, DataQualityConfig, DataQualityReport, DataQualityValidator,
    DistributionStats, IssueCategory, IssueSeverity, MissingStats, OutlierMethod, OutlierStats,
    QualityIssue,
};
pub use dimensionality_reduction::{
    ICAConfig, ICAFitted, IcaAlgorithm, IcaFunction, LDAConfig, LDAFitted, LdaSolver, NMFConfig,
    NMFFitted, NmfInit, NmfSolver, PCAConfig, PCAFitted, PcaSolver, ICA, LDA, NMF, PCA,
};
pub use encoding::{
    BinaryEncoder, BinaryEncoderConfig, CategoricalEmbedding, CategoricalEmbeddingConfig,
    FrequencyEncoder, FrequencyEncoderConfig, HashEncoder, HashEncoderConfig, LabelEncoder,
    OneHotEncoder, OrdinalEncoder, RareStrategy, TargetEncoder,
};
pub use feature_engineering::{
    ExtrapolationStrategy, FeatureOrder, KnotStrategy, PolynomialFeatures, PowerMethod,
    PowerTransformer, PowerTransformerConfig, SplineTransformer, SplineTransformerConfig,
};
pub use feature_union::{FeatureUnion, FeatureUnionConfig, FeatureUnionStep};
pub use geospatial::{
    calculate_distance, haversine_distance, vincenty_distance, Coordinate, CoordinateSystem,
    CoordinateTransformer, CoordinateTransformerConfig, CoordinateTransformerFitted, Geohash,
    GeohashEncoder, GeohashEncoderConfig, GeohashEncoderFitted, ProximityFeatures,
    ProximityFeaturesConfig, ProximityFeaturesFitted, SpatialAutocorrelation,
    SpatialAutocorrelationConfig, SpatialAutocorrelationFitted, SpatialBinning,
    SpatialBinningConfig, SpatialBinningFitted, SpatialClustering, SpatialClusteringConfig,
    SpatialClusteringFitted, SpatialClusteringMethod, SpatialDistanceFeatures,
    SpatialDistanceFeaturesConfig, SpatialDistanceFeaturesFitted, SpatialDistanceMetric,
};
// TODO: Depends on scirs2_core::memory::BufferPool which doesn't exist yet
// pub use gpu_acceleration::{
//     GpuBackend, GpuConfig, GpuContextManager, GpuMinMaxScaler, GpuMinMaxScalerFitted,
//     GpuPerformanceStats, GpuStandardScaler, GpuStandardScalerFitted,
// };
pub use image_preprocessing::{
    ColorSpace, ColorSpaceTransformer, EdgeDetectionMethod, EdgeDetector, ImageAugmenter,
    ImageAugmenterConfig, ImageFeatureExtractor, ImageNormalizationStrategy, ImageNormalizer,
    ImageNormalizerConfig, ImageNormalizerFitted, ImageResizer,
    InterpolationMethod as ImageInterpolationMethod,
};
pub use imputation::{
    BaseImputationMethod, DistanceMetric, FeatureMissingStats, GAINImputer, GAINImputerConfig,
    ImputationStrategy, IterativeImputer, KNNImputer, MissingPattern, MissingValueAnalysis,
    MissingnessType, MultipleImputationResult, MultipleImputer, MultipleImputerConfig,
    OutlierAwareImputer, OutlierAwareImputerConfig, OutlierAwareStatistics, OutlierAwareStrategy,
    OverallMissingStats, SimpleImputer,
};
pub use information_theory::{
    approximate_entropy, conditional_entropy, joint_entropy, lempel_ziv_complexity,
    mutual_information, normalized_mutual_information, permutation_entropy, renyi_entropy,
    sample_entropy, shannon_entropy, transfer_entropy, InformationFeatureSelector,
    InformationFeatureSelectorConfig, InformationFeatureSelectorFitted, InformationMetric,
};
pub use kernel_centerer::KernelCenterer;
pub use label_binarization::{
    LabelBinarizer, LabelBinarizerConfig, MultiLabelBinarizer, MultiLabelBinarizerConfig,
};
// TODO: Depends on scirs2_core::memory::BufferPool which doesn't exist yet
// pub use lazy_evaluation::{LazyConfig, LazyGraph, LazyNode, LazyOp, LazyPreprocessor};
// TODO: Depends on scirs2_core::memory::BufferPool which doesn't exist yet
// pub use memory_management::{
//     AdvancedMemoryConfig, AdvancedMemoryPool, AdvancedMemoryStats, CacheAlignedAllocator,
//     CompressedData, CopyOnWriteArray, MemoryCompressor, MemoryMappedDataset, MemoryPool,
//     MemoryPoolConfig, MemoryStats, PrefetchPattern, StreamingMemoryTransformer,
// };
pub use monitoring::{
    LogLevel, MonitoringConfig, MonitoringSession, MonitoringSummary, TransformationMetrics,
};
pub use outlier_detection::{
    FeatureOutlierParams, OutlierDetectionMethod, OutlierDetectionResult, OutlierDetector,
    OutlierDetectorConfig, OutlierStatistics, OutlierSummary,
};
pub use outlier_transformation::{
    FeatureTransformationParams, GlobalTransformationParams, OutlierTransformationConfig,
    OutlierTransformationMethod, OutlierTransformer, TransformationParameters,
};
pub use pipeline::{
    AdvancedPipeline, AdvancedPipelineBuilder, AdvancedPipelineConfig, BranchCombinationStrategy,
    CacheConfig, CacheStats, ConditionalStep, ConditionalStepConfig, DynamicPipeline,
    ErrorHandlingStrategy, ParallelBranchConfig, ParallelBranches, PipelineStep,
    TransformationCache,
};
pub use pipeline_validation::{
    PerformanceRecommendation, PipelineValidator, PipelineValidatorConfig, RecommendationCategory,
    ValidationError, ValidationErrorType, ValidationResult, ValidationWarning, WarningSeverity,
};
pub use probabilistic_imputation::{
    BayesianImputer, BayesianImputerConfig, BayesianImputerFitted, EMImputer, EMImputerConfig,
    EMImputerFitted, GaussianProcessImputer, GaussianProcessImputerConfig,
    GaussianProcessImputerFitted, MonteCarloBaseMethod, MonteCarloImputer, MonteCarloImputerConfig,
    MonteCarloImputerFitted,
};
pub use quantile_transformer::{QuantileOutput, QuantileTransformer, QuantileTransformerConfig};
pub use robust_preprocessing::{
    MissingValueStats, RobustPreprocessingStats, RobustPreprocessor, RobustPreprocessorConfig,
    RobustStrategy, TransformationStats,
};
pub use scaling::{
    FeatureWiseScaler, FeatureWiseScalerConfig, MaxAbsScaler, MinMaxScaler, NormType, Normalizer,
    OutlierAwareScaler, OutlierAwareScalerConfig, OutlierAwareScalingStrategy, OutlierScalingStats,
    RobustScaler, RobustStatistic, ScalingMethod, StandardScaler, UnitVectorScaler,
    UnitVectorScalerConfig,
};
pub use simd_optimizations::{
    add_scalar_f64_simd, add_vectors_f64_simd, mean_f64_simd, min_max_f64_simd,
    mul_scalar_f64_simd, ndarray_ops, sub_vectors_f64_simd, variance_f64_simd, SimdConfig,
};
pub use sparse_optimizations::{
    sparse_matvec, SparseConfig, SparseFormat, SparseMatrix, SparseStandardScaler,
    SparseStandardScalerFitted,
};
pub use streaming::{
    AdaptiveConfig as StreamingAdaptiveConfig, AdaptiveParameterManager,
    AdaptiveStreamingMinMaxScaler, AdaptiveStreamingStandardScaler, IncrementalPCA,
    IncrementalPCAStats, MiniBatchConfig, MiniBatchIterator, MiniBatchPipeline, MiniBatchStats,
    MiniBatchStreamingTransformer, MiniBatchTransformer, MultiQuantileEstimator,
    OnlineMADEstimator, OnlineMADStats, OnlineQuantileEstimator, OnlineQuantileStats,
    ParameterUpdate, StreamCharacteristics, StreamingConfig, StreamingLabelEncoder,
    StreamingMinMaxScaler, StreamingPipeline, StreamingRobustScaler, StreamingRobustScalerStats,
    StreamingSimpleImputer, StreamingStandardScaler, StreamingStats, StreamingTransformer,
};
pub use temporal::{
    ChangePointDetector, ChangePointDetectorConfig, ChangePointMethod, DateComponents, DateTime,
    DecompositionMethod, FillMethod, FourierFeatureGenerator, FourierFeatureGeneratorConfig,
    InterpolationMethod, LagFeatureGenerator, LagFeatureGeneratorConfig,
    MultiVariateTimeSeriesAligner, ResamplingMethod, SeasonalDecomposer, SeasonalDecomposerConfig,
    StationarityMethod, StationarityTransformer, StationarityTransformerConfig,
    StationarityTransformerFitted, TemporalFeatureExtractor, TemporalFeatureExtractorConfig,
    TimeSeriesInterpolator, TimeSeriesResampler, TrendDetector, TrendDetectorConfig, TrendMethod,
};
pub use text::{
    BagOfWordsConfig, BagOfWordsEmbedding, NgramGenerator, NgramGeneratorConfig, NgramType,
    NormalizationStrategy, SimilarityMetric, TextSimilarity, TextSimilarityConfig, TextTokenizer,
    TextTokenizerConfig, TfIdfVectorizer, TfIdfVectorizerConfig, TokenizationStrategy,
};
pub use type_safety::{
    Dimension, Dynamic, Fitted, Known, TransformState, TypeSafeConfig, TypeSafePipeline,
    TypeSafeTransformer, Unfitted,
};
pub use winsorization::{NanStrategy, WinsorizationStats, Winsorizer, WinsorizerConfig};

// Re-export functional APIs (excluding complex transformations that are commented out)
pub use functional::{
    add_dummy_feature, binarize, label_binarize, maxabs_scale, minmax_scale, normalize,
    robust_scale, scale,
};

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::adaptive::{
        AdaptationStrategy, AdaptiveConfig, AdaptiveParameterSelector, DataCharacteristics,
        DistributionType, ImputationParameters, OutlierDetectionParameters, ParameterEvaluation,
        ParameterRecommendations, ScalingParameters,
        TransformationParameters as AdaptiveTransformationParameters,
    };
    pub use crate::automated_feature_engineering::{
        AutoFeatureConfig, AutoFeatureEngineer, AutoFeatureEngineerFitted, Domain,
        GenerationStrategy, MathFunction, SelectionMethod, TransformationFunction,
        TransformationType,
    };
    pub use crate::binarization::{
        Binarizer, BinarizerConfig, DiscretizationStrategy, DiscretizerEncoding, KBinsDiscretizer,
        KBinsDiscretizerConfig,
    };
    pub use crate::column_transformer::{
        ColumnSelector, ColumnTransformer, ColumnTransformerConfig, DataType, RemainderStrategy,
        TransformerStep, TransformerWrapper,
    };
    pub use crate::cross_validation::{
        CVScore, InformationPreservationMetric, KFold, ParameterDistribution, ParameterGrid,
        PreprocessingMetric, StratifiedKFold, VariancePreservationMetric,
    };
    pub use crate::data_quality::{
        CorrelationWarning, DataQualityConfig, DataQualityReport, DataQualityValidator,
        DistributionStats, IssueCategory, IssueSeverity, MissingStats, OutlierMethod, OutlierStats,
        QualityIssue,
    };
    pub use crate::dimensionality_reduction::{
        ICAConfig, ICAFitted, IcaAlgorithm, IcaFunction, LDAConfig, LDAFitted, LdaSolver,
        NMFConfig, NMFFitted, NmfInit, NmfSolver, PCAConfig, PCAFitted, PcaSolver, ICA, LDA, NMF,
        PCA,
    };
    pub use crate::encoding::{
        BinaryEncoder, BinaryEncoderConfig, CategoricalEmbedding, CategoricalEmbeddingConfig,
        FrequencyEncoder, FrequencyEncoderConfig, HashEncoder, HashEncoderConfig, LabelEncoder,
        OneHotEncoder, OrdinalEncoder, RareStrategy, TargetEncoder,
    };
    pub use crate::feature_engineering::{
        ExtrapolationStrategy, FeatureOrder, KnotStrategy, PolynomialFeatures, PowerMethod,
        PowerTransformer, PowerTransformerConfig, SplineTransformer, SplineTransformerConfig,
    };
    pub use crate::feature_union::{FeatureUnion, FeatureUnionConfig, FeatureUnionStep};
    pub use crate::geospatial::{
        calculate_distance, haversine_distance, vincenty_distance, Coordinate, CoordinateSystem,
        CoordinateTransformer, CoordinateTransformerConfig, CoordinateTransformerFitted, Geohash,
        GeohashEncoder, GeohashEncoderConfig, GeohashEncoderFitted, ProximityFeatures,
        ProximityFeaturesConfig, ProximityFeaturesFitted, SpatialAutocorrelation,
        SpatialAutocorrelationConfig, SpatialAutocorrelationFitted, SpatialBinning,
        SpatialBinningConfig, SpatialBinningFitted, SpatialClustering, SpatialClusteringConfig,
        SpatialClusteringFitted, SpatialClusteringMethod, SpatialDistanceFeatures,
        SpatialDistanceFeaturesConfig, SpatialDistanceFeaturesFitted, SpatialDistanceMetric,
    };
    // TODO: Depends on scirs2_core::memory::BufferPool which doesn't exist yet
    // pub use crate::gpu_acceleration::{
    //     GpuBackend, GpuConfig, GpuContextManager, GpuMinMaxScaler, GpuMinMaxScalerFitted,
    //     GpuPerformanceStats, GpuStandardScaler, GpuStandardScalerFitted,
    // };
    pub use crate::image_preprocessing::{
        ColorSpace, ColorSpaceTransformer, EdgeDetectionMethod, EdgeDetector, ImageAugmenter,
        ImageAugmenterConfig, ImageFeatureExtractor, ImageNormalizationStrategy, ImageNormalizer,
        ImageNormalizerConfig, ImageNormalizerFitted, ImageResizer,
        InterpolationMethod as ImageInterpolationMethod,
    };
    pub use crate::imputation::{
        BaseImputationMethod, DistanceMetric, FeatureMissingStats, GAINImputer, GAINImputerConfig,
        ImputationStrategy, IterativeImputer, KNNImputer, MissingPattern, MissingValueAnalysis,
        MissingnessType, MultipleImputationResult, MultipleImputer, MultipleImputerConfig,
        OutlierAwareImputer, OutlierAwareImputerConfig, OutlierAwareStatistics,
        OutlierAwareStrategy, OverallMissingStats, SimpleImputer,
    };
    pub use crate::information_theory::{
        approximate_entropy, conditional_entropy, joint_entropy, lempel_ziv_complexity,
        mutual_information, normalized_mutual_information, permutation_entropy, renyi_entropy,
        sample_entropy, shannon_entropy, transfer_entropy, InformationFeatureSelector,
        InformationFeatureSelectorConfig, InformationFeatureSelectorFitted, InformationMetric,
    };
    pub use crate::kernel_centerer::KernelCenterer;
    pub use crate::label_binarization::{
        LabelBinarizer, LabelBinarizerConfig, MultiLabelBinarizer, MultiLabelBinarizerConfig,
    };
    // TODO: Depends on scirs2_core::memory::BufferPool which doesn't exist yet
    // pub use crate::lazy_evaluation::{LazyConfig, LazyGraph, LazyNode, LazyOp, LazyPreprocessor};
    // TODO: Depends on scirs2_core::memory::BufferPool which doesn't exist yet
    // pub use crate::memory_management::{
    //     AdvancedMemoryConfig, AdvancedMemoryPool, AdvancedMemoryStats, CacheAlignedAllocator,
    //     CompressedData, CopyOnWriteArray, MemoryCompressor, MemoryMappedDataset, MemoryPool,
    //     MemoryPoolConfig, MemoryStats, PrefetchPattern, StreamingMemoryTransformer,
    // };
    pub use crate::monitoring::{
        LogLevel, MonitoringConfig, MonitoringSession, MonitoringSummary, TransformationMetrics,
    };
    pub use crate::outlier_detection::{
        FeatureOutlierParams, OutlierDetectionMethod, OutlierDetectionResult, OutlierDetector,
        OutlierDetectorConfig, OutlierStatistics, OutlierSummary,
    };
    pub use crate::outlier_transformation::{
        FeatureTransformationParams, GlobalTransformationParams, OutlierTransformationConfig,
        OutlierTransformationMethod, OutlierTransformer,
    };
    pub use crate::pipeline::{
        AdvancedPipeline, AdvancedPipelineBuilder, AdvancedPipelineConfig,
        BranchCombinationStrategy, CacheConfig, CacheStats, ConditionalStep, ConditionalStepConfig,
        DynamicPipeline, ErrorHandlingStrategy, ParallelBranchConfig, ParallelBranches,
        PipelineStep, TransformationCache,
    };
    pub use crate::pipeline_validation::{
        PerformanceRecommendation, PipelineValidator, PipelineValidatorConfig,
        RecommendationCategory, ValidationError, ValidationErrorType, ValidationResult,
        ValidationWarning, WarningSeverity,
    };
    pub use crate::probabilistic_imputation::{
        BayesianImputer, BayesianImputerConfig, BayesianImputerFitted, EMImputer, EMImputerConfig,
        EMImputerFitted, GaussianProcessImputer, GaussianProcessImputerConfig,
        GaussianProcessImputerFitted, MonteCarloBaseMethod, MonteCarloImputer,
        MonteCarloImputerConfig, MonteCarloImputerFitted,
    };
    pub use crate::quantile_transformer::{
        QuantileOutput, QuantileTransformer, QuantileTransformerConfig,
    };
    pub use crate::robust_preprocessing::{
        MissingValueStats, RobustPreprocessingStats, RobustPreprocessor, RobustPreprocessorConfig,
        RobustStrategy, TransformationStats,
    };
    pub use crate::scaling::{
        FeatureWiseScaler, FeatureWiseScalerConfig, MaxAbsScaler, MinMaxScaler, NormType,
        Normalizer, OutlierAwareScaler, OutlierAwareScalerConfig, OutlierAwareScalingStrategy,
        OutlierScalingStats, RobustScaler, RobustStatistic, ScalingMethod, StandardScaler,
        UnitVectorScaler, UnitVectorScalerConfig,
    };
    pub use crate::simd_optimizations::{
        add_scalar_f64_simd, add_vectors_f64_simd, mean_f64_simd, min_max_f64_simd,
        mul_scalar_f64_simd, ndarray_ops, sub_vectors_f64_simd, variance_f64_simd, SimdConfig,
    };
    pub use crate::sparse_optimizations::{
        sparse_matvec, SparseConfig, SparseFormat, SparseMatrix, SparseStandardScaler,
        SparseStandardScalerFitted,
    };
    pub use crate::streaming::{
        AdaptiveConfig as StreamingAdaptiveConfig, AdaptiveParameterManager,
        AdaptiveStreamingMinMaxScaler, AdaptiveStreamingStandardScaler, IncrementalPCA,
        IncrementalPCAStats, MiniBatchConfig, MiniBatchIterator, MiniBatchPipeline, MiniBatchStats,
        MiniBatchStreamingTransformer, MiniBatchTransformer, MultiQuantileEstimator,
        OnlineMADEstimator, OnlineMADStats, OnlineQuantileEstimator, OnlineQuantileStats,
        ParameterUpdate, StreamCharacteristics, StreamingConfig, StreamingLabelEncoder,
        StreamingMinMaxScaler, StreamingPipeline, StreamingRobustScaler,
        StreamingRobustScalerStats, StreamingSimpleImputer, StreamingStandardScaler,
        StreamingStats, StreamingTransformer,
    };
    pub use crate::temporal::{
        ChangePointDetector, ChangePointDetectorConfig, ChangePointMethod, DateComponents,
        DateTime, DecompositionMethod, FillMethod, FourierFeatureGenerator,
        FourierFeatureGeneratorConfig, InterpolationMethod, LagFeatureGenerator,
        LagFeatureGeneratorConfig, MultiVariateTimeSeriesAligner, ResamplingMethod,
        SeasonalDecomposer, SeasonalDecomposerConfig, StationarityMethod, StationarityTransformer,
        StationarityTransformerConfig, StationarityTransformerFitted, TemporalFeatureExtractor,
        TemporalFeatureExtractorConfig, TimeSeriesInterpolator, TimeSeriesResampler, TrendDetector,
        TrendDetectorConfig, TrendMethod,
    };
    pub use crate::text::{
        BagOfWordsConfig, BagOfWordsEmbedding, NgramGenerator, NgramGeneratorConfig, NgramType,
        NormalizationStrategy, SimilarityMetric, TextSimilarity, TextSimilarityConfig,
        TextTokenizer, TextTokenizerConfig, TfIdfVectorizer, TfIdfVectorizerConfig,
        TokenizationStrategy,
    };
    pub use crate::type_safety::{
        Dimension, Dynamic, Fitted, Known, TransformState, TypeSafeConfig, TypeSafePipeline,
        TypeSafeTransformer, Unfitted,
    };
    pub use crate::winsorization::{NanStrategy, WinsorizationStats, Winsorizer, WinsorizerConfig};

    // Re-export functional APIs (excluding complex transformations that are commented out)
    pub use crate::functional::{
        add_dummy_feature, binarize, label_binarize, maxabs_scale, minmax_scale, normalize,
        robust_scale, scale,
    };
}
