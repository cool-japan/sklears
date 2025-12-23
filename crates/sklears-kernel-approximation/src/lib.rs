#![allow(dead_code)]
#![allow(non_snake_case)]
#![allow(missing_docs)]
#![allow(deprecated)]
#![allow(clippy::type_complexity)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::only_used_in_recursion)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::manual_clamp)]
#![allow(clippy::ptr_arg)]
#![allow(clippy::single_match)]
#![allow(clippy::if_same_then_else)]
#![allow(clippy::incompatible_msrv)]
//! Kernel approximation methods
//!
//! This module is part of sklears, providing scikit-learn compatible
//! machine learning algorithms in Rust.

// #![warn(missing_docs)]

pub mod adaptive_bandwidth_rbf;
pub mod adaptive_dimension;
pub mod adaptive_nystroem;
pub mod advanced_testing;
pub mod anisotropic_rbf;
pub mod benchmarking;
pub mod bioinformatics_kernels;
pub mod budget_constrained;
pub mod cache_optimization;
pub mod causal_kernels;
pub mod chi2_samplers;
pub mod computer_vision_kernels;
pub mod cross_validation;
pub mod custom_kernel;
pub mod deep_learning_kernels;
pub mod distributed_kernel;
pub mod ensemble_nystroem;
pub mod error_bounded;
pub mod fastfood;
pub mod feature_generation;
pub mod finance_kernels;
pub mod gpu_acceleration;
pub mod gradient_kernel_learning;
pub mod graph_kernels;
pub mod homogeneous_polynomial;
pub mod incremental_nystroem;
pub mod information_theoretic;
pub mod kernel_framework;
pub mod kernel_ridge_regression;
pub mod memory_efficient;
pub mod meta_learning_kernels;
pub mod middleware;
pub mod multi_kernel_learning;
pub mod multi_scale_rbf;
pub mod nlp_kernels;
pub mod numerical_stability;
pub mod nystroem;
pub mod optimal_transport;
pub mod out_of_core;
pub mod parameter_learning;
pub mod plugin_architecture;
pub mod polynomial_count_sketch;
pub mod polynomial_features;
pub mod progressive;
pub mod quantum_kernel_methods;
pub mod quasi_random_features;
pub mod rbf_sampler;
pub mod robust_kernels;
pub mod scientific_computing_kernels;
// SIMD modules now use SciRS2-Core (stable Rust compatible)
pub mod simd_kernel;
// #[cfg(feature = "nightly-simd")]
// pub mod simd_optimizations;
pub mod simple_test;
pub mod sparse_gp;
pub mod sparse_polynomial;
pub mod streaming_kernel;
pub mod string_kernels;
pub mod structured_random_features;
pub mod tensor_polynomial;
pub mod time_series_kernels;
pub mod type_safe_kernels;
pub mod type_safety;
pub mod unsafe_optimizations;
pub mod validation;

pub use adaptive_bandwidth_rbf::{
    AdaptiveBandwidthRBFSampler, BandwidthSelectionStrategy, ObjectiveFunction,
};
pub use adaptive_dimension::{
    AdaptiveDimensionConfig, AdaptiveRBFSampler, DimensionSelectionResult,
    FittedAdaptiveRBFSampler, QualityMetric as AdaptiveQualityMetric, SelectionStrategy,
};
pub use adaptive_nystroem::{
    AdaptiveNystroem, ComponentSelectionStrategy, ErrorBoundMethod as AdaptiveErrorBoundMethod,
};
pub use advanced_testing::{
    ApproximationError, BaselineMethod, BoundType as TestingBoundType, BoundValidation,
    ConvergenceAnalyzer, ConvergenceResult, ErrorBoundsResult, ErrorBoundsValidator,
    QualityAssessment, QualityMetric as TestingQualityMetric, QualityResult, ReferenceMethod,
};
pub use anisotropic_rbf::{
    AnisotropicRBFSampler, FittedAnisotropicRBF, FittedMahalanobisRBF, FittedRobustAnisotropicRBF,
    MahalanobisRBFSampler, RobustAnisotropicRBFSampler, RobustEstimator,
};
pub use benchmarking::{
    BenchmarkConfig, BenchmarkDataset, BenchmarkResult, BenchmarkSummary,
    BenchmarkableKernelMethod, BenchmarkableTransformer, KernelApproximationBenchmark,
    PerformanceMetric, QualityMetric,
};
pub use bioinformatics_kernels::{
    GenomicKernel, MetabolicNetworkKernel, MultiOmicsKernel, OmicsIntegrationMethod,
    PhylogeneticKernel, ProteinKernel,
};
pub use budget_constrained::{
    BudgetConstrainedConfig, BudgetConstrainedNystroem, BudgetConstrainedRBFSampler,
    BudgetConstraint, BudgetOptimizationResult, BudgetUsage, FittedBudgetConstrainedNystroem,
    FittedBudgetConstrainedRBFSampler, OptimizationStrategy,
};
pub use cache_optimization::{
    AlignedBuffer, CacheAwareTransform, CacheConfig, CacheFriendlyMatrix, MemoryLayout,
};
pub use causal_kernels::{CausalKernel, CausalKernelConfig, CausalMethod, CounterfactualKernel};
pub use chi2_samplers::{AdditiveChi2Sampler, SkewedChi2Sampler};
pub use computer_vision_kernels::{
    ActivationFunction, ConvolutionalKernelFeatures, FittedConvolutionalKernelFeatures,
    FittedScaleInvariantFeatures, FittedSpatialPyramidFeatures, FittedTextureKernelApproximation,
    PoolingMethod, ScaleInvariantFeatures, SpatialPyramidFeatures, TextureKernelApproximation,
};
pub use cross_validation::{
    CVSplitter, CVStrategy, CrossValidationConfig, CrossValidationResult as CVResult,
    CrossValidator, KFoldSplitter, MonteCarloCVSplitter, ScoringMetric, TimeSeriesSplitter,
};
pub use custom_kernel::{
    CustomExponentialKernel, CustomKernelSampler, CustomLaplacianKernel, CustomPolynomialKernel,
    CustomRBFKernel, KernelFunction,
};
pub use deep_learning_kernels::{
    Activation as DeepLearningActivation, DKLConfig, DeepKernelLearning, InfiniteWidthKernel,
    NTKConfig, NeuralTangentKernel,
};
pub use distributed_kernel::{
    AggregationMethod, CommunicationPattern, DistributedConfig, DistributedNystroem,
    DistributedRBFSampler, PartitionStrategy, Worker,
};
pub use ensemble_nystroem::{
    EnsembleMethod, EnsembleNystroem, QualityMetric as EnsembleQualityMetric,
};
pub use error_bounded::{
    ErrorBound, ErrorBoundMethod, ErrorBoundedConfig, ErrorBoundedNystroem, ErrorBoundedRBFSampler,
    FittedErrorBoundedNystroem, FittedErrorBoundedRBFSampler,
};
pub use fastfood::{FastfoodKernel, FastfoodKernelParams, FastfoodTransform};
pub use feature_generation::{
    CompositeGenerator, FeatureGenerator, FeatureGeneratorBuilder, PolynomialGenerator,
    RandomFourierGenerator,
};
pub use finance_kernels::{
    EconometricKernel, FinancialKernel, PortfolioKernel, RiskKernel, VolatilityKernel,
    VolatilityModel,
};
pub use gpu_acceleration::{
    FittedGpuNystroem, FittedGpuRBFSampler, GpuBackend, GpuConfig, GpuContext, GpuDevice,
    GpuNystroem, GpuProfiler, GpuRBFSampler, MemoryStrategy, Precision,
};
pub use gradient_kernel_learning::{
    GradientConfig, GradientKernelLearner, GradientMultiKernelLearner, GradientOptimizer,
    GradientResult, KernelObjective,
};
pub use graph_kernels::{
    FittedRandomWalkKernel, FittedShortestPathKernel, FittedSubgraphKernel,
    FittedWeisfeilerLehmanKernel, Graph, RandomWalkKernel, ShortestPathKernel, SubgraphKernel,
    WeisfeilerLehmanKernel,
};
pub use homogeneous_polynomial::{
    CoefficientMethod, HomogeneousPolynomialFeatures, NormalizationMethod,
};
pub use incremental_nystroem::{IncrementalNystroem, UpdateStrategy};
pub use information_theoretic::{
    EntropyFeatureSelector, EntropySelectionMethod, FittedEntropyFeatureSelector,
    FittedInformationBottleneckExtractor, FittedKLDivergenceKernel, FittedMutualInformationKernel,
    InformationBottleneckExtractor, KLDivergenceKernel, KLReferenceDistribution,
    MutualInformationKernel,
};
pub use kernel_framework::{
    ApproximationQuality, BoundType as FrameworkBoundType,
    CombinationStrategy as FrameworkCombinationStrategy, Complexity, CompositeKernelMethod,
    ErrorBound as FrameworkErrorBound, FeatureMap, KMeansSampling, KernelAlignmentMetric,
    KernelMethod, KernelType as FrameworkKernelType, SamplingStrategy as SamplingStrategyTrait,
    UniformSampling,
};
pub use kernel_ridge_regression::{
    ApproximationMethod as KRRApproximationMethod, KernelRidgeRegression,
    MultiTaskKernelRidgeRegression, OnlineKernelRidgeRegression, RobustKernelRidgeRegression,
    RobustLoss, Solver, TaskRegularization,
};
pub use memory_efficient::{
    FittedMemoryEfficientNystroem, FittedMemoryEfficientRBFSampler, MemoryConfig,
    MemoryEfficientNystroem, MemoryEfficientRBFSampler, MemoryMonitor,
};
pub use meta_learning_kernels::{
    DatasetMetaFeatures, MetaKernelType, MetaLearningConfig, MetaLearningKernelSelector,
    MetaLearningStrategy, PerformanceMetric as MetaPerformanceMetric, TaskMetadata,
};
pub use middleware::{
    Hook, HookContext, LoggingHook, Middleware, NormalizationMiddleware, PerformanceHook, Pipeline,
    PipelineBuilder, PipelineStage, ValidationHook,
};
pub use multi_kernel_learning::{
    ApproximationMethod as MKLApproximationMethod, BaseKernel,
    CombinationStrategy as MKLCombinationStrategy, KernelStatistics, MultiKernelConfig,
    MultipleKernelLearning, WeightLearningAlgorithm,
};
pub use multi_scale_rbf::{BandwidthStrategy, CombinationStrategy, MultiScaleRBFSampler};
pub use nlp_kernels::{
    AggregationMethod as NLPAggregationMethod, DocumentKernelApproximation,
    FittedDocumentKernelApproximation, FittedSemanticKernelApproximation,
    FittedSyntacticKernelApproximation, FittedTextKernelApproximation, SemanticKernelApproximation,
    SimilarityMeasure, SyntacticKernelApproximation, TextKernelApproximation, TreeKernelType,
};
pub use numerical_stability::{
    stable_kernel_matrix, NumericalStabilityMonitor, StabilityConfig, StabilityMetrics,
    StabilityWarning,
};
pub use nystroem::{Kernel, Nystroem, SamplingStrategy};
pub use optimal_transport::{
    EMDKernelSampler, GWLossFunction, GromovWassersteinSampler, GroundMetric, TransportMethod,
    WassersteinKernelSampler,
};
pub use out_of_core::{
    OutOfCoreConfig, OutOfCoreKernelPipeline, OutOfCoreLoader, OutOfCoreNystroem,
    OutOfCoreRBFSampler, OutOfCoreStrategy,
};
pub use parameter_learning::{
    ObjectiveFunction as ParameterObjectiveFunction, OptimizationResult, ParameterBounds,
    ParameterLearner, ParameterLearningConfig, ParameterSet, SearchStrategy,
};
pub use plugin_architecture::{
    create_global_plugin_instance, list_global_plugins, register_global_plugin,
    FittedPluginWrapper, KernelApproximationInstance, KernelApproximationPlugin,
    LinearKernelInstance, LinearKernelPlugin, PluginConfig, PluginError, PluginFactory,
    PluginMetadata, PluginWrapper,
};
pub use polynomial_count_sketch::PolynomialCountSketch;
pub use polynomial_features::PolynomialFeatures;
pub use progressive::{
    FittedProgressiveNystroem, FittedProgressiveRBFSampler, ProgressiveConfig, ProgressiveNystroem,
    ProgressiveQualityMetric, ProgressiveRBFSampler, ProgressiveResult, ProgressiveStep,
    ProgressiveStrategy, StoppingCriterion,
};
pub use quantum_kernel_methods::{
    EntanglementPattern, QuantumFeatureMap, QuantumKernelApproximation, QuantumKernelConfig,
};
pub use quasi_random_features::{QuasiRandomRBFSampler, QuasiRandomSequence};
pub use rbf_sampler::{ArcCosineSampler, LaplacianSampler, PolynomialSampler, RBFSampler};
pub use robust_kernels::{
    BreakdownPointAnalysis, InfluenceFunctionDiagnostics, RobustEstimator as RobustKernelEstimator,
    RobustKernelConfig, RobustLoss as RobustKernelLoss, RobustNystroem, RobustRBFSampler,
};
pub use scientific_computing_kernels::{
    MultiscaleKernel, PhysicalSystem, PhysicsInformedConfig, PhysicsInformedKernel,
};
// #[cfg(feature = "nightly-simd")]
// pub use simd_optimizations::{
//     FittedSimdRBFSampler, SimdBenchmarks, SimdOptimizations, SimdRBFSampler,
// };
pub use sparse_gp::{
    simd_sparse_gp, FittedSKI, FittedSparseGP, FittedTensorSKI, InducingPointSelector,
    InducingPointStrategy, InterpolationMethod, KernelOps, LanczosMethod, MaternKernel,
    PreconditionedCG, PreconditionerType, RBFKernel as SparseRBFKernel, ScalableInference,
    ScalableInferenceMethod, SparseApproximation, SparseApproximationMethods,
    SparseGaussianProcess, SparseKernel, StochasticVariationalInference,
    StructuredKernelInterpolation, TensorSKI, VariationalFreeEnergy, VariationalParams,
};
pub use sparse_polynomial::{
    SparseFormat, SparseMatrix, SparsePolynomialFeatures, SparsityStrategy,
};
pub use streaming_kernel::{
    BufferStrategy, FeatureStatistics, ForgettingMechanism, StreamingConfig, StreamingNystroem,
    StreamingRBFSampler, StreamingSample, UpdateFrequency,
};
pub use string_kernels::{
    EditDistanceKernel, FittedEditDistanceKernel, FittedMismatchKernel, FittedNGramKernel,
    FittedSpectrumKernel, FittedSubsequenceKernel, MismatchKernel, NGramKernel, NGramMode,
    SpectrumKernel, SubsequenceKernel,
};
pub use structured_random_features::{
    FastWalshHadamardTransform, StructuredMatrix, StructuredRFFHadamard, StructuredRandomFeatures,
};
pub use tensor_polynomial::{ContractionMethod, TensorOrdering, TensorPolynomialFeatures};
pub use time_series_kernels::{
    AutoregressiveKernelApproximation, DTWConfig, DTWDistanceMetric, DTWKernelApproximation,
    DTWStepPattern, DTWWindowType, GlobalAlignmentKernelApproximation, SpectralKernelApproximation,
    TimeSeriesKernelConfig, TimeSeriesKernelType,
};
pub use type_safe_kernels::{
    ArcCosineKernel as TypeSafeArcCosineKernel, FastfoodMethod as TypeSafeFastfoodMethod,
    FittedTypeSafeKernelApproximation, FittedTypeSafeLaplacianRandomFourierFeatures,
    FittedTypeSafePolynomialNystrom, FittedTypeSafeRBFFastfood, FittedTypeSafeRBFNystrom,
    FittedTypeSafeRBFRandomFourierFeatures, KernelType as TypeSafeKernelType,
    LaplacianKernel as TypeSafeLaplacianKernel, NystromMethod as TypeSafeNystromMethod,
    PolynomialKernel as TypeSafePolynomialKernel, QualityMetrics as TypeSafeQualityMetrics,
    RBFKernel as TypeSafeRBFKernel, RandomFourierFeatures as TypeSafeRandomFourierFeatures,
    Trained as TypeSafeTrained, TypeSafeKernelApproximation,
    TypeSafeLaplacianRandomFourierFeatures, TypeSafePolynomialNystrom, TypeSafeRBFFastfood,
    TypeSafeRBFNystrom, TypeSafeRBFRandomFourierFeatures, Untrained as TypeSafeUntrained,
};
pub use type_safety::{
    ApproximationMethod,
    ApproximationParameters,
    ApproximationState,
    ArcCosineKernel,
    ComplexityClass,
    FastfoodMethod,
    FittableKernel,
    FittableMethod,
    FittedLaplacianRandomFourierFeatures,
    FittedRBFNystrom,
    FittedRBFRandomFourierFeatures,
    // New exports for enhanced features
    KernelPresets,
    KernelType,
    KernelTypeWithBandwidth,
    LaplacianKernel,
    LaplacianRandomFourierFeatures,
    NystromMethod,
    OptimizationLevel,
    PolynomialKernel,
    PolynomialKernelType,
    PolynomialNystrom,
    ProfileGuidedConfig,
    QualityMetrics,
    RBFFastfood,
    RBFKernel,
    RBFNystrom,
    RBFRandomFourierFeatures,
    RandomFourierFeatures,
    SerializableFittedParams,
    SerializableKernelApproximation,
    SerializableKernelConfig,
    TargetArchitecture,
    Trained,
    TransformationParameters,
    Untrained,
};
pub use unsafe_optimizations::{
    batch_rbf_kernel_fast, dot_product_unrolled, elementwise_op_fast, fast_cosine_features,
    matvec_multiply_fast, rbf_kernel_fast, safe_dot_product, safe_matvec_multiply,
};
pub use validation::{
    BoundFunction, BoundType, CrossValidationResult, DimensionDependencyAnalysis,
    KernelApproximationValidator, SampleComplexityAnalysis, StabilityAnalysis, TheoreticalBound,
    ValidatableKernelMethod, ValidatedFittedMethod, ValidationConfig, ValidationResult,
};

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;
    use sklears_core::traits::{Fit, Transform};

    #[test]
    fn test_kernel_approximation_integration() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0],];

        // Test RBF sampler
        let rbf = RBFSampler::new(100);
        let fitted_rbf = rbf.fit(&x, &()).unwrap();
        let x_rbf = fitted_rbf.transform(&x).unwrap();
        assert_eq!(x_rbf.shape(), &[4, 100]);

        // Test Laplacian sampler
        let laplacian = LaplacianSampler::new(50);
        let fitted_laplacian = laplacian.fit(&x, &()).unwrap();
        let x_laplacian = fitted_laplacian.transform(&x).unwrap();
        assert_eq!(x_laplacian.shape(), &[4, 50]);

        // Test Additive Chi2 sampler (stateless)
        let chi2 = AdditiveChi2Sampler::new(2);
        let x_chi2 = chi2.transform(&x.mapv(|v| v.abs())).unwrap();
        assert_eq!(x_chi2.shape(), &[4, 6]); // 2 features * (2*2-1) = 6

        // Test Polynomial sampler
        let poly = PolynomialSampler::new(30).degree(3).gamma(1.0).coef0(1.0);
        let fitted_poly = poly.fit(&x, &()).unwrap();
        let x_poly = fitted_poly.transform(&x).unwrap();
        assert_eq!(x_poly.shape(), &[4, 30]);

        // Test Arc-cosine sampler
        let arc_cosine = ArcCosineSampler::new(25).degree(1);
        let fitted_arc = arc_cosine.fit(&x, &()).unwrap();
        let x_arc = fitted_arc.transform(&x).unwrap();
        assert_eq!(x_arc.shape(), &[4, 25]);
    }
}
