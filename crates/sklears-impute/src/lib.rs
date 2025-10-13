#![allow(dead_code)]
#![allow(non_snake_case)]
#![allow(missing_docs)]
#![allow(deprecated)]
//! Missing value imputation strategies
//!
//! This module provides various strategies for handling missing values in datasets.
//! It includes simple imputation methods as well as more sophisticated approaches
//! like iterative imputation, KNN-based imputation, matrix factorization, and Bayesian methods.

// #![warn(missing_docs)]

// Re-export the main modules
pub mod advanced;
pub mod approximate;
pub mod bayesian;
pub mod benchmarks;
pub mod categorical;
pub mod core;
pub mod dimensionality;
pub mod distributed;
pub mod domain_specific;
pub mod ensemble;
pub mod fluent_api;
pub mod independence;
pub mod information_theoretic;
pub mod kernel;
pub mod memory_profiler;
pub mod mixed_type;
pub mod multivariate;
pub mod neural;
pub mod out_of_core;
pub mod parallel;
pub mod sampling;
pub mod simd_ops;
pub mod simple;
pub mod testing_pipeline;
pub mod timeseries;
pub mod type_safe;
pub mod validation;
pub mod visualization;

// Re-export commonly used types and functions for convenience
pub use advanced::{
    analyze_breakdown_point, BreakdownPointAnalysis, CopulaImputer, CopulaParameters, EmpiricalCDF,
    EmpiricalQuantile, FactorAnalysisImputer, KDEImputer, LocalLinearImputer, LowessImputer,
    MultivariateNormalImputer, RobustRegressionImputer, TrimmedMeanImputer,
};
pub use bayesian::{
    BayesianLinearImputer, BayesianLogisticImputer, BayesianModel, BayesianModelAveraging,
    BayesianModelAveragingResults, BayesianMultipleImputer, ConvergenceDiagnostics,
    HierarchicalBayesianImputer, HierarchicalBayesianSample, PooledResults,
    VariationalBayesImputer,
};
pub use benchmarks::{
    AccuracyMetrics, BenchmarkDatasetGenerator, BenchmarkSuite, ImputationBenchmark,
    ImputationComparison, MissingPattern, MissingPatternGenerator,
};
pub use categorical::{
    AssociationRule, AssociationRuleImputer, CategoricalClusteringImputer,
    CategoricalRandomForestImputer, HotDeckImputer, Item, Itemset,
};
pub use core::{
    utils, ConvergenceInfo, ImputationError, ImputationMetadata, ImputationOutputWithMetadata,
    ImputationResult, Imputer, ImputerConfig, MissingPatternHandler, QualityAssessment,
    StatisticalValidator, TrainableImputer, TransformableImputer,
};
pub use dimensionality::{
    CompressedSensingImputer, ICAImputer, ManifoldLearningImputer, PCAImputer, SparseImputer,
};
pub use domain_specific::{
    CreditScoringImputer, DemographicDataImputer, EconomicIndicatorImputer,
    FinancialTimeSeriesImputer, GenomicImputer, LongitudinalStudyImputer, MetabolomicsImputer,
    MissingResponseHandler, PhylogeneticImputer, PortfolioDataImputer, ProteinExpressionImputer,
    RiskFactorImputer, SingleCellRNASeqImputer, SocialNetworkImputer, SurveyDataImputer,
};
pub use ensemble::{ExtraTreesImputer, GradientBoostingImputer, RandomForestImputer};
pub use fluent_api::{
    pluggable::{
        ComposedPipeline, DataCharacteristics, DataType, ImputationInstance, ImputationMiddleware,
        ImputationModule, LogLevel, LoggingMiddleware, MissingPatternType, ModuleConfig,
        ModuleConfigSchema, ModuleRegistry, ParameterGroup, ParameterRange, ParameterSchema,
        ParameterType, PipelineComposer, PipelineStage, StageCondition, ValidationMiddleware,
    },
    quick, DeepLearningBuilder, EnsembleImputationBuilder, GaussianProcessBuilder,
    ImputationBuilder, ImputationMethod, ImputationPipeline, ImputationPreset,
    IterativeImputationBuilder, KNNImputationBuilder, PostprocessingConfig, PreprocessingConfig,
    SimpleImputationBuilder, ValidationConfig,
};
pub use independence::{
    chi_square_independence_test, cramers_v_association_test, fisher_exact_independence_test,
    kolmogorov_smirnov_independence_test, pattern_sensitivity_analysis,
    run_independence_test_suite, sensitivity_analysis, ChiSquareTestResult, CramersVTestResult,
    FisherExactTestResult, IndependenceTestSuite, KolmogorovSmirnovTestResult, MARSensitivityCase,
    MNARSensitivityCase, MissingDataAssessment, PatternSensitivityResult, RobustnessSummary,
    SensitivityAnalysisResult,
};
pub use information_theoretic::{
    EntropyImputer, InformationGainImputer, MDLImputer, MaxEntropyImputer, MutualInformationImputer,
};
pub use kernel::{
    GPPredictionResult, GaussianProcessImputer, KernelRidgeImputer, ReproducingKernelImputer,
    SVRImputer,
};
pub use memory_profiler::{
    ImputationMemoryBenchmark, MemoryProfiler, MemoryProfilingResult, MemoryStats,
};
pub use mixed_type::{
    HeterogeneousImputer, MixedTypeMICEImputer, MixedTypeMultipleImputationResults, OrdinalImputer,
    VariableMetadata, VariableParameters, VariableType,
};
pub use multivariate::CanonicalCorrelationImputer;
pub use neural::{
    AutoencoderImputer, DiffusionImputer, GANImputer, MLPImputer, NeuralODEImputer,
    NormalizingFlowImputer, VAEImputer,
};
pub use parallel::{
    AdaptiveStreamingImputer, MemoryEfficientImputer, MemoryMappedData, MemoryOptimizedImputer,
    MemoryStrategy, OnlineStatistics, ParallelConfig, ParallelIterativeImputer, ParallelKNNImputer,
    SharedDataRef, SparseMatrix, StreamingImputer,
};
pub use simd_ops::{
    SimdDistanceCalculator, SimdImputationOps, SimdKMeans, SimdMatrixOps, SimdStatistics,
};
pub use simple::{MissingIndicator, SimpleImputer};
pub use timeseries::{
    ARIMAImputer, KalmanFilterImputer, SeasonalDecompositionImputer, StateSpaceImputer,
};
pub use type_safe::{
    ClassifiedArray, Complete, CompleteArray, FixedSizeArray, FixedSizeValidation,
    ImputationQualityMetrics, MARArray, MCARArray, MNARArray, MissingMechanism,
    MissingPatternValidator, MissingValueDetector, NaNDetector, SentinelDetector,
    TypeSafeImputation, TypeSafeMeanImputer, TypeSafeMissingOps, TypedArray, UnknownMechanism,
    WithMissing, MAR, MCAR, MNAR,
};
pub use validation::{
    validate_with_holdout, CrossValidationResults, CrossValidationStrategy, HoldOutValidator,
    ImputationCrossValidator, ImputationMetrics, MissingDataPattern, SyntheticMissingValidator,
};
pub use visualization::{
    create_completeness_matrix, create_missing_correlation_heatmap,
    create_missing_distribution_plot, create_missing_pattern_plot, export_correlation_csv,
    export_missing_pattern_csv, generate_missing_summary_stats, CompletenessMatrix,
    MissingCorrelationHeatmap, MissingDistributionPlot, MissingPatternPlot,
};

// New modules - Advanced Algorithms
pub use approximate::{
    ApproximateConfig, ApproximateKNNImputer, ApproximateSimpleImputer, ApproximationStrategy,
    LocalityHashTable, SketchingImputer,
};
pub use distributed::{
    CommunicationStrategy, DistributedConfig, DistributedKNNImputer, DistributedSimpleImputer,
    DistributedWorker, ImputationCoordinator,
};
pub use out_of_core::{
    IndexType, MemoryManager, NeighborIndex, OutOfCoreConfig, OutOfCoreKNNImputer,
    OutOfCoreSimpleImputer, PrefetchStrategy,
};
pub use sampling::{
    AdaptiveSamplingImputer, ImportanceSamplingImputer, ParametricDistribution,
    ProposalDistribution, QuasiSequenceType, SampleDistribution, SamplingConfig,
    SamplingSimpleImputer, SamplingStrategy, StratifiedSamplingImputer, WeightFunction,
};
pub use testing_pipeline::{
    AutomatedTestPipeline, CompletedTestCase, PerformanceBenchmarks, QualityThresholds, TestCase,
    TestDataset, TestPipelineConfig, TestResults, TestRunner, TestStatus,
};

// ✅ SciRS2 Policy compliant imports
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Transform, Untrained},
    types::Float,
};
use std::collections::HashMap;

// Legacy implementations (to be moved to separate modules when fully refactored)

/// K-Nearest Neighbors Imputer
///
/// Imputation for completing missing values using k-Nearest Neighbors.
/// Each missing value is imputed using values from k nearest neighbors
/// found in the training set.
///
/// # Parameters
///
/// * `n_neighbors` - Number of neighboring samples to use for imputation
/// * `weights` - Weight function used in prediction ('uniform' or 'distance')
/// * `metric` - Distance metric for searching neighbors ('nan_euclidean')
/// * `missing_values` - The placeholder for missing values (NaN by default)
/// * `add_indicator` - Whether to add a missing value indicator
///
/// # Examples
///
/// ```
/// use sklears_impute::KNNImputer;
/// use sklears_core::traits::{Transform, Fit};
/// use scirs2_core::ndarray::array;
///
/// let X = array![[1.0, 2.0, 3.0], [4.0, f64::NAN, 6.0], [7.0, 8.0, 9.0]];
///
/// let imputer = KNNImputer::new()
///     .n_neighbors(2);
/// let fitted = imputer.fit(&X.view(), &()).unwrap();
/// let X_imputed = fitted.transform(&X.view()).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct KNNImputer<S = Untrained> {
    state: S,
    n_neighbors: usize,
    weights: String,
    metric: String,
    missing_values: f64,
    add_indicator: bool,
}

/// Trained state for KNNImputer
#[derive(Debug, Clone)]
pub struct KNNImputerTrained {
    X_train_: Array2<f64>,
    n_features_in_: usize,
}

impl KNNImputer<Untrained> {
    /// Create a new KNNImputer instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            n_neighbors: 5,
            weights: "uniform".to_string(),
            metric: "nan_euclidean".to_string(),
            missing_values: f64::NAN,
            add_indicator: false,
        }
    }

    /// Set the number of neighbors
    pub fn n_neighbors(mut self, n_neighbors: usize) -> Self {
        self.n_neighbors = n_neighbors;
        self
    }

    /// Set the weight function
    pub fn weights(mut self, weights: String) -> Self {
        self.weights = weights;
        self
    }

    /// Set the distance metric
    pub fn metric(mut self, metric: String) -> Self {
        self.metric = metric;
        self
    }

    /// Set the missing values placeholder
    pub fn missing_values(mut self, missing_values: f64) -> Self {
        self.missing_values = missing_values;
        self
    }

    /// Set whether to add missing indicator
    pub fn add_indicator(mut self, add_indicator: bool) -> Self {
        self.add_indicator = add_indicator;
        self
    }

    fn is_missing(&self, value: f64) -> bool {
        if self.missing_values.is_nan() {
            value.is_nan()
        } else {
            (value - self.missing_values).abs() < f64::EPSILON
        }
    }
}

impl Default for KNNImputer<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for KNNImputer<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, ()> for KNNImputer<Untrained> {
    type Fitted = KNNImputer<KNNImputerTrained>;

    #[allow(non_snake_case)]
    fn fit(self, X: &ArrayView2<'_, Float>, _y: &()) -> SklResult<Self::Fitted> {
        let X = X.mapv(|x| x as f64);
        let (_, n_features) = X.dim();

        Ok(KNNImputer {
            state: KNNImputerTrained {
                X_train_: X.clone(),
                n_features_in_: n_features,
            },
            n_neighbors: self.n_neighbors,
            weights: self.weights,
            metric: self.metric,
            missing_values: self.missing_values,
            add_indicator: self.add_indicator,
        })
    }
}

impl Transform<ArrayView2<'_, Float>, Array2<Float>> for KNNImputer<KNNImputerTrained> {
    #[allow(non_snake_case)]
    fn transform(&self, X: &ArrayView2<'_, Float>) -> SklResult<Array2<Float>> {
        let X = X.mapv(|x| x as f64);
        let (n_samples, n_features) = X.dim();

        if n_features != self.state.n_features_in_ {
            return Err(SklearsError::InvalidInput(format!(
                "Number of features {} does not match training features {}",
                n_features, self.state.n_features_in_
            )));
        }

        let mut X_imputed = X.clone();
        let X_train = &self.state.X_train_;

        for i in 0..n_samples {
            for j in 0..n_features {
                if self.is_missing(X_imputed[[i, j]]) {
                    // Find k nearest neighbors
                    let mut distances: Vec<(f64, usize)> = Vec::new();

                    for train_idx in 0..X_train.nrows() {
                        let distance =
                            self.nan_euclidean_distance(X_imputed.row(i), X_train.row(train_idx));
                        distances.push((distance, train_idx));
                    }

                    distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

                    // Take k nearest neighbors that have this feature observed
                    let mut neighbor_values = Vec::new();
                    let mut weights = Vec::new();

                    for &(distance, train_idx) in distances.iter().take(self.n_neighbors * 3) {
                        if !self.is_missing(X_train[[train_idx, j]]) {
                            neighbor_values.push(X_train[[train_idx, j]]);
                            let weight = match self.weights.as_str() {
                                "distance" => {
                                    if distance > 0.0 {
                                        1.0 / distance
                                    } else {
                                        1e6
                                    }
                                }
                                _ => 1.0,
                            };
                            weights.push(weight);

                            if neighbor_values.len() >= self.n_neighbors {
                                break;
                            }
                        }
                    }

                    if neighbor_values.is_empty() {
                        // Fallback to mean of training data for this feature
                        let column = X_train.column(j);
                        let valid_values: Vec<f64> = column
                            .iter()
                            .filter(|&&x| !self.is_missing(x))
                            .cloned()
                            .collect();

                        if !valid_values.is_empty() {
                            X_imputed[[i, j]] =
                                valid_values.iter().sum::<f64>() / valid_values.len() as f64;
                        }
                    } else {
                        // Weighted average of neighbor values
                        let total_weight: f64 = weights.iter().sum();
                        let weighted_sum: f64 = neighbor_values
                            .iter()
                            .zip(weights.iter())
                            .map(|(&value, &weight)| value * weight)
                            .sum();

                        X_imputed[[i, j]] = if total_weight > 0.0 {
                            weighted_sum / total_weight
                        } else {
                            neighbor_values.iter().sum::<f64>() / neighbor_values.len() as f64
                        };
                    }
                }
            }
        }

        Ok(X_imputed.mapv(|x| x as Float))
    }
}

impl KNNImputer<KNNImputerTrained> {
    fn is_missing(&self, value: f64) -> bool {
        if self.missing_values.is_nan() {
            value.is_nan()
        } else {
            (value - self.missing_values).abs() < f64::EPSILON
        }
    }

    fn nan_euclidean_distance(&self, row1: ArrayView1<f64>, row2: ArrayView1<f64>) -> f64 {
        let mut sum_sq = 0.0;
        let mut valid_count = 0;

        for (x1, x2) in row1.iter().zip(row2.iter()) {
            if !self.is_missing(*x1) && !self.is_missing(*x2) {
                sum_sq += (x1 - x2).powi(2);
                valid_count += 1;
            }
        }

        if valid_count > 0 {
            (sum_sq / valid_count as f64).sqrt()
        } else {
            f64::INFINITY
        }
    }
}

/// Analysis functions for missing data patterns
#[allow(non_snake_case)]
pub fn analyze_missing_patterns(
    X: &ArrayView2<'_, Float>,
    missing_values: f64,
) -> SklResult<HashMap<String, Vec<usize>>> {
    let X = X.mapv(|x| x as f64);
    let (n_samples, n_features) = X.dim();
    let mut patterns = HashMap::new();

    for i in 0..n_samples {
        let mut pattern = Vec::new();
        for j in 0..n_features {
            let is_missing = if missing_values.is_nan() {
                X[[i, j]].is_nan()
            } else {
                (X[[i, j]] - missing_values).abs() < f64::EPSILON
            };
            if is_missing {
                pattern.push(j);
            }
        }

        let pattern_key = format!("{:?}", pattern);
        patterns.entry(pattern_key).or_insert_with(Vec::new).push(i);
    }

    Ok(patterns)
}

/// Compute missing correlation matrix
#[allow(non_snake_case)]
pub fn missing_correlation_matrix(
    X: &ArrayView2<'_, Float>,
    missing_values: f64,
) -> SklResult<Array2<f64>> {
    let X = X.mapv(|x| x as f64);
    let (n_samples, n_features) = X.dim();

    // Create missing indicators
    let mut missing_indicators = Array2::zeros((n_samples, n_features));
    for i in 0..n_samples {
        for j in 0..n_features {
            let is_missing = if missing_values.is_nan() {
                X[[i, j]].is_nan()
            } else {
                (X[[i, j]] - missing_values).abs() < f64::EPSILON
            };
            missing_indicators[[i, j]] = if is_missing { 1.0 } else { 0.0 };
        }
    }

    // Compute correlation matrix
    let mut correlation_matrix = Array2::zeros((n_features, n_features));
    for i in 0..n_features {
        for j in 0..n_features {
            if i == j {
                correlation_matrix[[i, j]] = 1.0;
            } else {
                let col_i = missing_indicators.column(i);
                let col_j = missing_indicators.column(j);
                correlation_matrix[[i, j]] =
                    compute_correlation(&col_i.to_owned(), &col_j.to_owned());
            }
        }
    }

    Ok(correlation_matrix)
}

/// Compute missing completeness matrix
#[allow(non_snake_case)]
pub fn missing_completeness_matrix(
    X: &ArrayView2<'_, Float>,
    missing_values: f64,
) -> SklResult<Array2<f64>> {
    let X = X.mapv(|x| x as f64);
    let (n_samples, n_features) = X.dim();

    let mut completeness_matrix = Array2::zeros((n_features, n_features));

    for i in 0..n_features {
        for j in 0..n_features {
            let mut joint_observed = 0;

            for sample_idx in 0..n_samples {
                let i_observed = if missing_values.is_nan() {
                    !X[[sample_idx, i]].is_nan()
                } else {
                    (X[[sample_idx, i]] - missing_values).abs() >= f64::EPSILON
                };

                let j_observed = if missing_values.is_nan() {
                    !X[[sample_idx, j]].is_nan()
                } else {
                    (X[[sample_idx, j]] - missing_values).abs() >= f64::EPSILON
                };

                if i_observed && j_observed {
                    joint_observed += 1;
                }
            }

            completeness_matrix[[i, j]] = joint_observed as f64 / n_samples as f64;
        }
    }

    Ok(completeness_matrix)
}

/// Generate comprehensive missing data summary
pub fn missing_data_summary(X: &ArrayView2<'_, Float>, missing_values: f64) -> SklResult<String> {
    generate_missing_summary_stats(X, missing_values)
}

// Helper function for correlation computation
fn compute_correlation(x: &Array1<f64>, y: &Array1<f64>) -> f64 {
    let n = x.len() as f64;
    if n == 0.0 {
        return 0.0;
    }

    let mean_x = x.sum() / n;
    let mean_y = y.sum() / n;

    let mut numerator = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;

    for i in 0..x.len() {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;

        numerator += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    let denominator = (var_x * var_y).sqrt();
    if denominator == 0.0 {
        0.0
    } else {
        numerator / denominator
    }
}

// Test module declaration - tests are in separate file for better organization
#[allow(non_snake_case)]
#[cfg(test)]
mod tests;
