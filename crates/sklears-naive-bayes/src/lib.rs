#![allow(dead_code)]
#![allow(non_snake_case)]
#![allow(missing_docs)]
#![allow(deprecated)]
#![allow(clippy::all)]
#![allow(clippy::pedantic)]
#![allow(clippy::nursery)]
#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(unused_mut)]
#![allow(unused_assignments)]
#![allow(unused_doc_comments)]
#![allow(unused_parens)]
#![allow(unused_comparisons)]
//! Naive Bayes classifiers
//!
//! This module provides various Naive Bayes classifiers for different
//! types of features, compatible with scikit-learn's naive_bayes module.

// SciRS2 Policy Compliance - Use scirs2-autograd for ndarray types
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::numeric::Float;

mod adaptive_smoothing;
mod api_builder;
// TODO: Migrate to scirs2-linalg (uses nalgebra types)
//mod attention_naive_bayes;
mod bayesian_network;
mod benchmarks;
mod bernoulli;
mod beta;
mod bioinformatics;
mod categorical;
mod causal_inference;
mod complement;
mod computer_vision;
// TODO: Migrate to scirs2-linalg (uses nalgebra types)
//mod continual_learning;
mod deep_generative;
mod dirichlet_process;
mod ensemble;
mod exponential_family;
mod feature_engineering;
// TODO: Migrate to scirs2-linalg (uses nalgebra types)
//mod federated;
// TODO: Migrate to scirs2-linalg (uses nalgebra types)
//mod finance;
mod flexible;
mod gamma;
mod gaussian;
mod hierarchical;
mod kernel_methods;
mod mixed;
// TODO: ndarray 0.17 HRTB trait bound issues
//mod model_selection;
mod multilabel;
mod multinomial;
mod neural_naive_bayes;
mod nonparametric;
mod online_learning;
mod optimizations;
mod parameter_estimation;
mod performance;
mod plugin_architecture;
mod poisson;
// TODO: Migrate to scirs2-linalg (uses nalgebra types)
//mod quantum;
mod semi_naive;
mod smoothing;
mod temporal;
mod text_classification;
mod tree_augmented;
mod type_safe_prob;
mod uncertainty;
mod validation;
mod variational_bayes;

#[allow(non_snake_case)]
#[cfg(test)]
mod property_tests;

pub use adaptive_smoothing::{
    AdaptiveSmoothing, AdaptiveSmoothingMethod, HyperparameterOptimizer,
    InformationCriterion as AdaptiveInformationCriterion, ScoringMethod,
};
pub use api_builder::{
    naive_bayes, naive_bayes_preset, FluentBernoulliNB, FluentCategoricalNB, FluentComplementNB,
    FluentGaussianNB, FluentMultinomialNB, FluentNaiveBayesModel, FluentPoissonNB,
    NaiveBayesBuilder, NaiveBayesPreset, SerializableNBParams,
};
// TODO: Migrate to scirs2-linalg (uses nalgebra types)
// pub use attention_naive_bayes::{
//     AttentionNBBuilder, AttentionNBConfig, AttentionNBError, AttentionNaiveBayes, AttentionType,
//     ImportanceScoring,
// };
pub use bayesian_network::{
    BANConfig, BANError, BayesianNetwork, BayesianNetworkAugmentedNB, NetworkEdge,
    StructureLearningMethod,
};
pub use benchmarks::{BenchmarkConfig, BenchmarkResults, NaiveBayesBenchmark};
pub use bernoulli::BernoulliNB;
pub use beta::BetaNB;
pub use bioinformatics::{
    AminoAcid, BioinformaticsError, BiomarkerDiscoveryNB, GeneExpressionConfig, GeneExpressionNB,
    GenomicNBConfig, GenomicNaiveBayes, GenomicSequence, Nucleotide, PhylogeneticConfig,
    PhylogeneticNB, ProteinStructureConfig, ProteinStructureNB, SecondaryStructure,
    SequenceMetadata, SequenceType,
};
pub use categorical::CategoricalNB;
pub use causal_inference::{
    CausalDiscovery, CausalDiscoveryAlgorithm, CausalDiscoveryConfig, CausalGraph,
    CausalInferenceError, CausalNaiveBayes, CounterfactualReasoning, DoCalculus,
    InstrumentalVariables,
};
pub use complement::ComplementNB;
pub use computer_vision::{
    utils as cv_utils, ColorSpace, ComputerVisionError, ImageData, ImageMetadata, ImageNBConfig,
    ImageNaiveBayes, NeighborhoodStats, SpatialModel, SpatialNBConfig, SpatialNaiveBayes,
};
// TODO: Migrate to scirs2-linalg (uses nalgebra types)
// pub use continual_learning::{
//     ContinualLearningConfig, ContinualLearningError, ContinualLearningNB,
//     ContinualLearningNBBuilder, ContinualLearningStrategy, DriftDetectionMethod, MemoryStrategy,
//     TaskMetadata,
// };
pub use deep_generative::{
    DeepGenerativeConfig, DeepGenerativeError, DeepGenerativeNaiveBayes, FlowConfig, NPEConfig,
    NeuralPosteriorEstimator, NormalizingFlow, VAEConfig, VariationalAutoencoder,
};
pub use dirichlet_process::{
    BaseDistributionParams, DirichletComponent, DirichletProcessConfig, DirichletProcessError,
    DirichletProcessNB, InferenceMethod, StickBreaking,
};
pub use ensemble::{
    AdaBoostNaiveBayes, AveragingNaiveBayes, BaggedNaiveBayes, BoostingStrategy, BootstrapConfig,
    NaiveBayesEstimator, StackingNaiveBayes, VotingNaiveBayes, VotingStrategy,
};
pub use exponential_family::{
    ExponentialFamily, ExponentialFamilyNB, ExponentialFamilyNBConfig, NaturalParameters,
    ParameterEstimationMethod, SufficientStatistics,
};
pub use feature_engineering::{
    AutoFeatureTransformer, AutoPipelineConfig, AutoTransformConfig,
    AutomatedPreprocessingPipeline, DataValidationConfig, FeatureInteractionDetector,
    FeatureSelectionMethod, FeatureSelectionResults, FeatureType as AutoFeatureType,
    ImbalanceHandlingMethod, InteractionMethod, InteractionResults, MissingValueStrategy,
    OutlierDetectionMethod, StatisticalTest, TransformMethod,
};
// TODO: Migrate to scirs2-linalg (uses nalgebra types)
// pub use federated::{
//     AggregationMethod, AggregationStrategy, BudgetAllocationStrategy, ClientId, ClientModel,
//     CommunicationParams, CompressionStrategy, FederatedError, FederatedNaiveBayes,
//     FederationParams, FederationStatistics, GlobalModel, LocalUpdate, PrivacyParams, PrivateUpdate,
// };
// TODO: Migrate to scirs2-linalg (uses nalgebra types)
// pub use finance::{
//     CreditData, CreditRisk, CreditScoringNB, FinanceError, FinancialFeatures,
//     FinancialTimeSeriesNB, FraudDetectionNB, FraudLabel, PortfolioCategory,
//     PortfolioClassificationNB, PortfolioData, RiskAssessmentNB, RiskLevel, TransactionData,
// };
pub use flexible::{
    Distribution, DistributionParams, FlexibleNB, FlexibleNBConfig, SelectionMethod,
};
pub use gamma::GammaNB;
pub use gaussian::GaussianNB;
pub use hierarchical::{
    ClassHierarchy, HierarchicalConfig, HierarchicalError, HierarchicalNB, HierarchyNode,
    PredictionStrategy,
};
pub use kernel_methods::{
    BandwidthMethod as KernelBandwidthMethod, GaussianProcess, KDEConfig,
    KernelDensityEstimator as AdvancedKDE, KernelError, KernelNaiveBayes, KernelParameterLearner,
    KernelType as AdvancedKernelType, RKHSFeatureSelector, ReproducingKernelHilbertSpace,
    ScoringMetric as KernelScoringMetric,
};
pub use mixed::{FeatureDistribution, MixedNB};
// TODO: ndarray 0.17 HRTB trait bound issues
// pub use model_selection::{
//     BayesianModelComparison, BayesianModelSelector, CVResults, CVStrategy, InformationCriterion,
//     ModelComparison, ModelSelectionResults, NaiveBayesModelSelector, NestedModelComparison,
//     NestedModelValidation, ParameterGrid, ParameterValue, ScoringMetric,
// };
pub use multilabel::{
    AdvancedChainClassifier, ChainOrderingStrategy, LabelCorrelationAnalysis, LabelDependencyGraph,
    LabelHierarchy, MultiLabelNB, MultiLabelStrategy,
};
pub use multinomial::MultinomialNB;
pub use neural_naive_bayes::{
    ActivationFunction, NeuralLayer, NeuralNBBuilder, NeuralNBConfig, NeuralNBError,
    NeuralNaiveBayes,
};
pub use nonparametric::{
    BandwidthMethod, KernelDensityEstimator, KernelType, NonparametricNB, NonparametricNBConfig,
};
pub use online_learning::{
    CSVChunkReader,
    ConceptDriftDetector,
    // Out-of-core learning types
    DataChunkIterator,
    MemoryChunkIterator,
    OnlineGaussianStats,
    OnlineLearningConfig,
    OnlineLearningError,
    OnlineMultinomialStats,
    OnlineNaiveBayes,
    OutOfCoreNaiveBayes,
    StreamingBuffer,
};
pub use optimizations::{
    cache_optimization, memory_efficient, numerical_stability as advanced_numerical_stability,
    parallel, profile_guided_optimization, simd, unsafe_optimizations,
};
pub use parameter_estimation::{
    CrossValidationSelector, EmpiricalBayesEstimator, GaussianEstimator, MultinomialEstimator,
    ParameterEstimator,
};
pub use performance::{
    CompressedNBModel, CompressionMetadata, LazyLoadedModel, MemoryMappedNBModel,
    MemoryOptimizedOps, MemoryStats, SparseMatrix, SparseRow, VectorizedOps,
};
pub use plugin_architecture::{
    get_global_registry, register_distribution, register_middleware, register_model_selector,
    register_parameter_estimator, register_smoothing_method, ComposableSmoothingMethod,
    DataCharacteristics, DataType, EstimationResult, ExtensibleParameterEstimator,
    FlexibleModelSelector, ModelCandidate, PluggableDistribution, PluginRegistry,
    PredictionContext, PredictionMiddleware, SelectionCriterion,
};
pub use poisson::PoissonNB;
// TODO: Migrate to scirs2-linalg (uses nalgebra types)
// pub use quantum::{
//     EntanglementPattern, GateType, HybridQuantumClassicalNB, QuantumAdvantageMetrics,
//     QuantumCircuitParams, QuantumError, QuantumFeatureMap, QuantumNaiveBayes, QuantumState,
//     RotationAxis,
// };
pub use semi_naive::{
    DependencySelectionMethod, FeatureDependency, SemiNaiveBayes, SemiNaiveBayesConfig,
};
pub use smoothing::{
    enhanced_log, log_sum_exp, normalize_log_probs, numerical_stability, SmoothingMethod,
};
pub use temporal::{
    CovarianceType, HMMConfig, HMMNaiveBayes, StreamingTemporalNB, TemporalConfig,
    TemporalFeatureExtractor, TemporalNaiveBayes,
};
pub use text_classification::{
    LDATopicModel, NGramExtractor, TextMultinomialNB, TextMultinomialNBConfig, TextPreprocessor,
    TfIdfTransformer, TopicAugmentedConfig, TopicAugmentedTextClassifier,
};
pub use tree_augmented::{DependencyTree, TANConfig, TANError, TreeAugmentedNB, TreeEdge};
pub use type_safe_prob::{
    const_utils, distribution_types, feature_types, utils as prob_utils, FixedSizeModel,
    ProbabilisticModel as TypeSafeProbabilisticModel, TypedProbability, TypedProbabilityOps,
    ValidateFeatureType,
};
pub use uncertainty::{
    BayesianUncertainty, CalibrationMethod, EnsemblePredictions, ModelUncertaintyPropagation,
    ReliabilityTests, StandardUncertainty, UncertaintyDecomposition, UncertaintyMeasures,
    UncertaintyQuantification,
};
pub use validation::{
    CVResults as ValidationCVResults, CVStrategy as ValidationCVStrategy, CalibrationMetrics,
    GoodnessOfFitResults, ModelCriticismResults, ModelDegradationMetrics, PPCResults,
    PredictionStabilityMetrics, PredictiveAccuracyAssessment, PredictiveAccuracyAssessor,
    ProbabilisticModel, ProbabilisticValidator, ResidualAnalysis,
    ScoringMetric as ValidationScoringMetric, TemporalValidationResults, ValidationError,
};
pub use variational_bayes::{
    ELBOComponents, FeatureType, VariationalBayesConfig, VariationalBayesNB, VariationalError,
    VariationalInferenceMethod, VariationalParameters,
};

/// Base trait for Naive Bayes classifiers
pub trait NaiveBayesMixin {
    /// Get log prior probabilities
    fn class_log_prior(&self) -> &Array1<f64>;

    /// Get feature log probabilities
    fn feature_log_prob(&self) -> &Array2<f64>;

    /// Get the classes
    fn classes(&self) -> &Array1<i32>;
}

/// Helper function to compute class counts and priors
fn compute_class_prior(y: &Array1<i32>, classes: &Array1<i32>) -> (Array1<f64>, Array1<f64>) {
    let n_samples = y.len() as f64;
    let mut class_count = Array1::zeros(classes.len());

    for &label in y.iter() {
        for (i, &class) in classes.iter().enumerate() {
            if label == class {
                class_count[i] += 1.0;
            }
        }
    }

    let class_prior = &class_count / n_samples;
    (class_count, class_prior)
}

/// Helper function to compute log probability
fn safe_log<F: Float>(x: F) -> F {
    let epsilon = F::from(1e-10).unwrap();
    (x + epsilon).ln()
}
