//! Data-model types for the validation framework.
//!
//! These are the public input/result structures consumed and produced by the
//! [`ValidationFramework`](super::ValidationFramework). They carry only data and
//! have no behavior of their own.

use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::types::Float;
use std::collections::HashMap;
use std::time::Duration;

/// Benchmark dataset for validation
#[derive(Debug, Clone)]
pub struct BenchmarkDataset {
    /// Dataset name
    pub name: String,
    /// X data (features)
    pub x_data: Array2<Float>,
    /// Y data (targets)
    pub y_data: Array2<Float>,
    /// True canonical correlations (if known)
    pub true_correlations: Option<Array1<Float>>,
    /// True components (if known)
    pub true_x_components: Option<Array2<Float>>,
    pub true_y_components: Option<Array2<Float>>,
    /// Dataset characteristics
    pub characteristics: DatasetCharacteristics,
    /// Expected performance ranges
    pub expected_performance: HashMap<String, PerformanceRange>,
}

/// Dataset characteristics for analysis
#[derive(Debug, Clone)]
pub struct DatasetCharacteristics {
    /// Number of samples
    pub n_samples: usize,
    /// Number of X features
    pub n_x_features: usize,
    /// Number of Y features
    pub n_y_features: usize,
    /// Signal-to-noise ratio
    pub signal_to_noise: Float,
    /// Data distribution type
    pub distribution_type: DistributionType,
    /// Correlation structure
    pub correlation_structure: CorrelationStructure,
    /// Missing data percentage
    pub missing_data_percent: Float,
}

/// Types of data distributions
#[derive(Debug, Clone)]
pub enum DistributionType {
    /// Multivariate normal
    Gaussian,
    /// Heavy-tailed distributions
    HeavyTailed,
    /// Skewed distributions
    Skewed,
    /// Mixed distributions
    Mixed,
    /// Real-world (unknown distribution)
    RealWorld,
}

/// Correlation structure types
#[derive(Debug, Clone)]
pub enum CorrelationStructure {
    /// Linear correlations
    Linear,
    /// Nonlinear correlations
    Nonlinear,
    /// Sparse correlations
    Sparse,
    /// Block correlations
    Block,
    /// Complex (multiple types)
    Complex,
}

/// Performance metrics for evaluation
#[derive(Debug, Clone)]
pub enum PerformanceMetric {
    /// Canonical correlation accuracy
    CanonicalCorrelationAccuracy,
    /// Component recovery (angle between true and estimated)
    ComponentRecovery,
    /// Prediction accuracy on test set
    PredictionAccuracy,
    /// Cross-validation stability
    CrossValidationStability,
    /// Computational time
    ComputationalTime,
    /// Memory usage
    MemoryUsage,
    /// Robustness to noise
    NoiseRobustness,
    /// Scalability with sample size
    SampleScalability,
    /// Scalability with feature dimensionality
    FeatureScalability,
}

/// Expected performance range
#[derive(Debug, Clone)]
pub struct PerformanceRange {
    pub min_value: Float,
    pub max_value: Float,
    pub target_value: Float,
}

/// Statistical significance tests
#[derive(Debug, Clone)]
pub enum SignificanceTest {
    /// Permutation test for canonical correlations
    PermutationTest,
    /// Bootstrap confidence intervals
    BootstrapConfidenceIntervals,
    /// Cross-validation significance
    CrossValidationSignificance,
    /// Comparative algorithm tests
    ComparativeTests,
}

/// Real-world case studies
#[derive(Debug, Clone)]
pub struct CaseStudy {
    /// Case study name
    pub name: String,
    /// Domain (genomics, neuroscience, etc.)
    pub domain: String,
    /// Data description
    pub description: String,
    /// Expected insights
    pub expected_insights: Vec<String>,
    /// Validation criteria
    pub validation_criteria: Vec<ValidationCriterion>,
}

/// Validation criteria for case studies
#[derive(Debug, Clone)]
pub struct ValidationCriterion {
    pub name: String,
    pub description: String,
    pub metric_type: CriterionType,
    pub threshold: Float,
}

/// Types of validation criteria
#[derive(Debug, Clone)]
pub enum CriterionType {
    /// Biological relevance
    BiologicalRelevance,
    /// Statistical significance
    StatisticalSignificance,
    /// Reproducibility
    Reproducibility,
    /// Interpretability
    Interpretability,
    /// Prediction performance
    PredictionPerformance,
}

/// Cross-validation settings
#[derive(Debug, Clone)]
pub struct CrossValidationSettings {
    /// Number of folds
    pub n_folds: usize,
    /// Number of repetitions
    pub n_repetitions: usize,
    /// Stratification strategy
    pub stratification: bool,
    /// Random seed for reproducibility
    pub random_seed: Option<u64>,
}

/// Validation results
#[derive(Debug, Clone)]
pub struct ValidationResults {
    /// Results for each dataset
    pub dataset_results: HashMap<String, DatasetValidationResult>,
    /// Overall performance summary
    pub performance_summary: PerformanceSummary,
    /// Statistical test results
    pub statistical_results: HashMap<String, StatisticalTestResult>,
    /// Case study results
    pub case_study_results: HashMap<String, CaseStudyResult>,
    /// Computational benchmarks
    pub computational_benchmarks: ComputationalBenchmarks,
}

/// Validation results for a single dataset
#[derive(Debug, Clone)]
pub struct DatasetValidationResult {
    /// Performance metrics
    pub metrics: HashMap<String, Float>,
    /// Cross-validation results
    pub cv_results: CrossValidationResult,
    /// Component recovery analysis
    pub component_analysis: ComponentAnalysis,
    /// Robustness analysis
    pub robustness_analysis: RobustnessAnalysis,
}

/// Cross-validation results
#[derive(Debug, Clone)]
pub struct CrossValidationResult {
    /// Mean performance across folds
    pub mean_performance: HashMap<String, Float>,
    /// Standard deviation across folds
    pub std_performance: HashMap<String, Float>,
    /// Individual fold results
    pub fold_results: Vec<HashMap<String, Float>>,
    /// Stability metrics
    pub stability_metrics: StabilityMetrics,
}

/// Component analysis results
#[derive(Debug, Clone)]
pub struct ComponentAnalysis {
    /// Principal angles between true and estimated components
    pub principal_angles: Array1<Float>,
    /// Component correlation with ground truth
    pub component_correlations: Array1<Float>,
    /// Subspace recovery accuracy
    pub subspace_recovery: Float,
}

/// Robustness analysis results
#[derive(Debug, Clone)]
pub struct RobustnessAnalysis {
    /// Performance under different noise levels
    pub noise_robustness: HashMap<String, Float>,
    /// Performance with missing data
    pub missing_data_robustness: HashMap<String, Float>,
    /// Performance with outliers
    pub outlier_robustness: HashMap<String, Float>,
}

/// Stability metrics
#[derive(Debug, Clone)]
pub struct StabilityMetrics {
    /// Jaccard stability index
    pub jaccard_index: Float,
    /// Rand index
    pub rand_index: Float,
    /// Silhouette coefficient
    pub silhouette_coefficient: Float,
}

/// Performance summary across all tests
#[derive(Debug, Clone)]
pub struct PerformanceSummary {
    /// Overall accuracy scores
    pub overall_accuracy: HashMap<String, Float>,
    /// Algorithm rankings
    pub algorithm_rankings: HashMap<String, usize>,
    /// Strengths and weaknesses analysis
    pub strengths_weaknesses: HashMap<String, AlgorithmAnalysis>,
}

/// Algorithm analysis
#[derive(Debug, Clone)]
pub struct AlgorithmAnalysis {
    /// Strengths
    pub strengths: Vec<String>,
    /// Weaknesses
    pub weaknesses: Vec<String>,
    /// Recommended use cases
    pub recommended_use_cases: Vec<String>,
}

/// Statistical test result
#[derive(Debug, Clone)]
pub struct StatisticalTestResult {
    pub test_statistic: Float,
    pub p_value: Float,
    pub confidence_interval: (Float, Float),
    pub effect_size: Float,
}

/// Case study validation result
#[derive(Debug, Clone)]
pub struct CaseStudyResult {
    /// Criteria evaluation results
    pub criteria_results: HashMap<String, Float>,
    /// Overall success rate
    pub success_rate: Float,
    /// Insights discovered
    pub insights: Vec<String>,
    /// Recommendations
    pub recommendations: Vec<String>,
}

/// Computational benchmarks
#[derive(Debug, Clone)]
pub struct ComputationalBenchmarks {
    /// Execution times for different algorithms
    pub execution_times: HashMap<String, Duration>,
    /// Memory usage statistics
    pub memory_usage: HashMap<String, usize>,
    /// Scalability analysis
    pub scalability_analysis: ScalabilityAnalysis,
}

/// Scalability analysis results
#[derive(Debug, Clone)]
pub struct ScalabilityAnalysis {
    /// Time complexity with sample size
    pub time_vs_samples: Vec<(usize, Duration)>,
    /// Time complexity with features
    pub time_vs_features: Vec<(usize, Duration)>,
    /// Memory complexity analysis
    pub memory_complexity: HashMap<String, Float>,
}
