//! Cross decomposition algorithms (PLS, CCA)
//!
//! This module is part of sklears, providing scikit-learn compatible
//! machine learning algorithms in Rust.

#![allow(missing_docs)]
#![allow(unused_imports)]
#![allow(deprecated)]
#![allow(non_snake_case)]
#![allow(unused_variables)]
#![allow(unused_assignments)]
#![allow(unused_mut)]
#![allow(dead_code)]
#![allow(clippy::clone_on_copy)]
#![allow(clippy::assign_op_pattern)]
#![allow(clippy::derivable_impls)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::field_reassign_with_default)]
#![allow(clippy::type_complexity)]
#![allow(clippy::collapsible_else_if)]
#![allow(clippy::op_ref)]
#![allow(clippy::redundant_closure)]
#![allow(clippy::manual_clamp)]
#![allow(clippy::useless_format)]
#![allow(clippy::unnecessary_cast)]
#![allow(clippy::ptr_arg)]
#![allow(unpredictable_function_pointer_comparisons)]
#![recursion_limit = "512"]

pub mod bayesian;
pub mod benchmarks;
pub mod cca;
pub mod consensus_pca;
// TODO: Disabled due to ndarray 0.17 HRTB trait bound issues
// pub mod cross_validation;
pub mod deep_cca;
pub mod deep_learning;
pub mod differential_geometry;
pub mod federated_learning;
pub mod finance;
pub mod generalized_cca;
pub mod genomics;
pub mod gpu_acceleration;
pub mod graph_regularization;
pub mod information_theory;
pub mod interactive_visualization;
pub mod jive;
pub mod kernel_cca;
pub mod manifold_learning;
pub mod multi_omics;
pub mod multiblock_pls;
pub mod multitask;
pub mod multiview_cca;
pub mod multiview_clustering;
pub mod neuroimaging;
pub mod opls;
pub mod out_of_core;
pub mod parallel;
// TODO: Disabled due to ndarray 0.17 HRTB trait bound issues
// pub mod permutation_tests;
pub mod pls;
pub mod pls_canonical;
pub mod pls_da;
pub mod pls_svd;
pub mod quantum_methods;
pub mod regularization;
pub mod riemannian_optimization;
pub mod robust_methods;
pub mod scalability;
pub mod simd_acceleration;
pub mod sparse_pls;
pub mod tensor_methods;
pub mod time_series;
pub mod type_safe_linalg;
pub mod validation_framework;

pub use bayesian::{
    BayesianCCA, BayesianCCAResults, HierarchicalBayesianCCA, HierarchicalBayesianCCAResults,
    VariationalPLS, VariationalPLSResults,
};
pub use benchmarks::{
    AccuracyResults, BenchmarkResults, BenchmarkSuite, DecompositionResult, MethodBenchmarkResults,
    ScalabilityResults, SpeedResult, SummaryStats,
};
pub use cca::{RidgeCCA, SparseCCA, CCA};
pub use consensus_pca::ConsensusPCA;
// TODO: Disabled due to ndarray 0.17 HRTB trait bound issues
// pub use cross_validation::{
//     CVResults, CVStrategy, CrossValidator, NestedCrossValidator, ScoringFunction,
// };
pub use deep_cca::{ActivationFunction, DeepCCA};
pub use deep_learning::{
    ActivationFunction as DeepActivationFunction, AttentionActivation, AttentionConfig,
    AttentionLayer, AttentionOutput, AttentionTensorDecomposition, AttentionType,
    CrossModalAttention, CrossModalAttentionOutput, CrossModalSimilarity, CrossModalVAE,
    MultiHeadAttention, NeuralActivation, NeuralParafacDecomposition, NeuralTensorConfig,
    NeuralTensorResults, NeuralTuckerDecomposition, TransformerDecoderBlock,
    TransformerEncoderBlock, VAEConfig, VAETrainingResults, VariationalTensorNetwork,
};
pub use federated_learning::{
    AggregationStrategy as FederatedAggregationStrategy, ClientId, CommunicationConfig,
    FederatedCCA, FederatedCCAResults, FederatedClient, FederatedError, FederatedPCA,
    FederatedPCAResults, FederatedServer, PrivacyBudget,
};
pub use finance::{
    FactorConstrainedOptimization, FactorRotation, FactorStatistics, FinanceError,
    FinancialFactorAnalysis, FittedFinancialFactorAnalysis, FittedMacroeconomicFactorAnalysis,
    ForecastingModel, MacroFactorStatistics, MacroeconomicFactorAnalysis, OptimizedPortfolio,
    RiskDecomposition,
};
pub use generalized_cca::GeneralizedCCA;
pub use genomics::{
    ConsensusMethod, EnhancedPathwayAnalysis, EnhancedPathwayResults, EnrichmentMethod,
    FittedGeneEnvironmentInteraction, FittedMultiOmicsIntegration, FittedSingleCellMultiModal,
    FittedTemporalGeneExpression, GeneEnvironmentInteraction, GenomicsError, MLScoringConfig,
    MissingDataStrategy, MultiModalConfig, MultiOmicsIntegration, MultipleTestingCorrection,
    NetworkAnalysisConfig, PathwayAnalysis, PathwayAnalysisConfig, PathwayDatabase,
    SingleCellMultiModal, TemporalAnalysisConfig, TemporalGeneExpression,
};
pub use gpu_acceleration::{
    GpuAcceleratedContext, GpuCCA, GpuCCAFitted, GpuMatrixOps, GpuMemoryInfo,
};
pub use graph_regularization::{
    CommunityAlgorithm, CommunityDetectionConfig, CommunityDetector, CommunityStructure,
    GraphBuilder, GraphRegularizationConfig, GraphRegularizationError, GraphRegularizedCCA,
    GraphStructure, GraphType, Hypergraph, HypergraphCCA, HypergraphCCAResults,
    HypergraphCentrality, HypergraphConfig, HypergraphLaplacianType, MotifType, MultiGraphCCA,
    MultiWayInteractionAnalyzer, NetworkConstrainedPLS, RegularizationType,
    TemporalAnalysisResults, TemporalMotif, TemporalNetwork, TemporalNetworkAnalyzer,
    TemporalNetworkConfig,
};
pub use information_theory::{
    ComponentInterpretation, ComponentInterpreter, ComponentSelection, ComponentSimilarityAnalysis,
    DistanceBasedConfig, DistanceBasedMetric, DistanceBasedResults, DistanceCCA,
    DistanceCovariance, EntropyComponentSelection, EntropyEstimator, FeatureContribution,
    FeatureImportanceAnalyzer, FeatureImportanceResults, FittedMutualInformationCCA,
    HigherOrderAnalyzer, HigherOrderConfig, HigherOrderResults, ImportanceMethod,
    InformationGeometry, InformationMeasure, InformationTheoreticRegularization,
    InformationTheoryError, KLDivergenceMethods, ManifoldStructure, MutualInformationCCA,
    NonGaussianComponentAnalysis, NonGaussianResults, PolyspectralCCA, PolyspectralResults,
    RegularizationMethod, RiemannianOptimizer, SelectionCriteria, SingleComponentInterpretation,
    VariableInterpretation, HSIC,
};
pub use interactive_visualization::{
    ColorScheme, InteractivePlot, InteractiveVisualizationConfig, InteractiveVisualizer, PlotData,
    PlotType, VisualizationError,
};
pub use jive::JIVE;
pub use kernel_cca::{KernelCCA, KernelType};
pub use manifold_learning::{
    AdvancedManifoldLearning, ConvergenceInfo, CrossModalAlignment, DistanceMetric, EigenSolver,
    FittedManifoldAwareCCA, FittedManifoldCCA as FittedAdvancedManifoldCCA, GeodesicMethod,
    ManifoldAwareCCA, ManifoldCCA as AdvancedManifoldCCA, ManifoldError, ManifoldLearning,
    ManifoldLearningResult, ManifoldMethod, ManifoldProperties, ManifoldRegularization,
    ManifoldResults, OptimizationParams, PathMethod,
};
pub use multi_omics::GenomicsError as MultiOmicsGenomicsError;
pub use multiblock_pls::{BlockScaling, MultiBlockPLS};
pub use multitask::{
    DomainAdaptationCCA, FewShotCCA, MultiTaskCCA, SharedComponentAnalysis, TransferLearningCCA,
};
pub use multiview_cca::MultiViewCCA;
pub use multiview_clustering::{
    DistanceMetric as MultiViewDistanceMetric, InitMethod, MultiViewClustering,
};
pub use neuroimaging::{
    BrainBehaviorCorrelation, BrainBehaviorResults, ConnectivityType, CorrelationMethod,
    FunctionalConnectivity, FunctionalConnectivityResults, NetworkMeasures,
};
pub use opls::OPLS;
pub use out_of_core::{
    OOCAlgorithm, OutOfCoreCCA, OutOfCoreCCAResults, OutOfCorePLS, OutOfCorePLSResults,
};
pub use parallel::{
    EigenMethod, OptimizedMatrixOps, ParallelEigenSolver, ParallelMatrixOps, ParallelSVD,
    SVDAlgorithm, WorkStealingThreadPool,
};
// TODO: Disabled due to ndarray 0.17 HRTB trait bound issues
// pub use permutation_tests::{
//     ComputeStatistic, PermutationTest, PermutationTestResults, StabilityResults,
//     StabilitySelection, TestStatistic,
// };
pub use pls::PLSRegression;
pub use pls_canonical::PLSCanonical;
pub use pls_da::PLSDA;
pub use pls_svd::PLSSVD;
pub use quantum_methods::{
    QuantumCCA, QuantumCCAResults, QuantumCircuit, QuantumError, QuantumFeatureSelection,
    QuantumGate, QuantumMethod, QuantumPCA, QuantumPCAResults, QuantumState,
};
pub use regularization::{AdaptiveLasso, ElasticNet, FusedLasso, GroupLasso, MCP, SCAD};
pub use riemannian_optimization::{
    CCAObjective, GrassmannManifold, LineSearchParams, ManifoldType, RiemannianAlgorithm,
    RiemannianConfig, RiemannianError, RiemannianManifold, RiemannianObjective,
    RiemannianOptimizer as RiemannianOptimizerAdvanced, RiemannianResults, SPDManifold,
    StiefelManifold, TrustRegionParams,
};
pub use robust_methods::{MEstimatorType, RobustCCA, RobustPLS};
pub use scalability::{
    AggregationStrategy, DistributedCCA, DistributedCCAResults, MemoryEfficientCCA,
};
pub use simd_acceleration::{
    AdvancedSimdConfig, AdvancedSimdOps, SimdBenchmarkResults, SimdCCA, SimdCCAFitted,
    SimdMatrixOps,
};
pub use sparse_pls::SparsePLS;
pub use tensor_methods::{
    BayesianParafac, ParafacDecomposition, ProbabilisticConfig, ProbabilisticTensorResults,
    ProbabilisticTucker, RobustProbabilisticTensor, SparseTensorDecomposition, TensorCCA,
    TensorCompletion, TensorInitMethod, TuckerDecomposition,
};
pub use time_series::{
    DynamicCCA, DynamicCCAResults, DynamicCCASummary, FittedRegimeSwitchingModel,
    FittedStateSpaceModel, FittedVAR, GrangerCausalityTest, GrangerTestResult,
    InformationCriterion, RegimeSwitchingModel, StateSpaceForecast, StateSpaceModel,
    StateSpaceModelDiagnostics, StreamingCCA, TrendType, VARMethod, VectorAutoregression,
};
pub use type_safe_linalg::{
    decomp, ops, Dim, MatrixDimension, SquareMatrix, TypeSafeMatrix, TypeSafeVector,
};
pub use validation_framework::{
    BenchmarkDataset, CaseStudy, ComputationalBenchmarks, CorrelationStructure, CriterionType,
    CrossValidationResult, CrossValidationSettings, DatasetCharacteristics,
    DatasetValidationResult, DistributionType, PerformanceMetric, PerformanceRange,
    PerformanceSummary, RobustnessAnalysis, ScalabilityAnalysis, SignificanceTest,
    StatisticalTestResult, ValidationError, ValidationFramework, ValidationResults,
};

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;
    use sklears_core::traits::{Fit, Predict};

    #[test]
    fn test_pls_integration() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0],];

        let y = array![[1.5], [2.5], [3.5], [4.5],];

        let pls = PLSRegression::new(1);
        let fitted = pls.fit(&x, &y).unwrap();
        let predictions = fitted.predict(&x).unwrap();

        assert_eq!(predictions.shape(), &[4, 1]);
    }
}
