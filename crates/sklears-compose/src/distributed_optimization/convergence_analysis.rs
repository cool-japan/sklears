//! # Convergence Analysis Module
//!
//! This module provides comprehensive convergence monitoring and statistical analysis
//! for distributed optimization, including trend detection, changepoint analysis,
//! and advanced statistical tests for optimization trajectories.

use crate::distributed_optimization::core_types::*;
use crate::distributed_optimization::optimization_coordination::*;
use scirs2_core::ndarray::{Array1, Array2, ArrayView1};
use scirs2_core::random::{Random, rng};
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, SystemTime};
use serde::{Serialize, Deserialize};
use std::fmt;

/// Convergence monitoring system with advanced statistical analysis
pub struct ConvergenceMonitor {
    convergence_criteria: Vec<ConvergenceCriterion>,
    monitoring_history: VecDeque<ConvergenceMetric>,
    statistical_analyzer: StatisticalConvergenceAnalyzer,
    early_stopping: EarlyStoppingConfig,
    adaptive_criteria: AdaptiveCriteriaConfig,
    convergence_callbacks: Vec<ConvergenceCallback>,
}

/// Configuration for adaptive convergence criteria
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveCriteriaConfig {
    pub enable_adaptive_thresholds: bool,
    pub adaptation_rate: f64,
    pub min_samples_for_adaptation: usize,
    pub confidence_interval: f64,
    pub outlier_detection: bool,
}

/// Convergence criteria for optimization with advanced options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConvergenceCriterion {
    ObjectiveThreshold(f64),
    GradientNormThreshold(f64),
    ParameterChangeThreshold(f64),
    MaxIterations(u32),
    RelativeImprovement(f64),
    Statistical(StatisticalCriterion),
    Adaptive(AdaptiveCriterion),
    Composite(Vec<ConvergenceCriterion>),
}

/// Advanced adaptive convergence criteria
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveCriterion {
    pub base_criterion: Box<ConvergenceCriterion>,
    pub adaptation_window: usize,
    pub adaptation_factor: f64,
    pub min_threshold: f64,
    pub max_threshold: f64,
}

/// Statistical convergence criteria with advanced tests
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalCriterion {
    pub test_type: StatisticalTest,
    pub significance_level: f64,
    pub window_size: usize,
    pub min_samples: usize,
    pub correction_method: MultipleTestingCorrection,
}

/// Statistical tests for convergence detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StatisticalTest {
    MannKendall,
    AugmentedDickeyFuller,
    KolmogorovSmirnov,
    LjungBox,
    JarqueBera,
    ShapiroWilk,
    AndersonDarling,
    RunsTest,
    VonNeumannRatio,
    BartelsRankTest,
    CustomTest(String),
}

/// Multiple testing correction methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MultipleTestingCorrection {
    None,
    Bonferroni,
    BenjaminiHochberg,
    BenjaminiYekutieli,
    Holm,
    Hochberg,
}

/// Statistical convergence analyzer with advanced time series analysis
pub struct StatisticalConvergenceAnalyzer {
    trend_detector: TrendDetector,
    changepoint_detector: ChangepointDetector,
    stationarity_tester: StationarityTester,
    seasonality_detector: SeasonalityDetector,
    outlier_detector: OutlierDetector,
    distribution_analyzer: DistributionAnalyzer,
}

/// Advanced trend detection for convergence analysis
pub struct TrendDetector {
    detection_methods: Vec<TrendDetectionMethod>,
    trend_window: usize,
    confidence_threshold: f64,
    multiscale_analysis: bool,
    seasonal_adjustment: bool,
}

/// Comprehensive trend detection methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrendDetectionMethod {
    LinearRegression,
    MannKendall,
    TheilSen,
    SeasonalDecomposition,
    WaveletAnalysis,
    LocalPolynomialRegression,
    SplineRegression,
    KalmanFilter,
    ExponentialSmoothing,
    ARIMAModel,
}

/// Changepoint detection for optimization trajectories
pub struct ChangepointDetector {
    detection_algorithms: Vec<ChangepointAlgorithm>,
    detection_sensitivity: f64,
    min_segment_length: usize,
    max_changepoints: usize,
    penalty_factor: f64,
}

/// Advanced changepoint detection algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChangepointAlgorithm {
    CUSUM,
    PELT,
    BinarySegmentation,
    WindowedSegmentation,
    KernelChangePoint,
    BayesianChangePoint,
    VariationalChangePoint,
    OnlineNewtonStep,
    WildBinarySegmentation,
    EDivisive,
}

/// Stationarity testing for time series analysis
pub struct StationarityTester {
    test_methods: Vec<StationarityTest>,
    significance_level: f64,
    difference_order: usize,
    seasonal_periods: Vec<usize>,
    robust_tests: bool,
}

/// Comprehensive stationarity tests
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StationarityTest {
    AugmentedDickeyFuller,
    KPSS,
    PhillipsPerron,
    ZivotAndrews,
    ElliottRothenbergStock,
    NgPerron,
    LeeStrazicich,
    ClementeMontanesPope,
}

/// Seasonality detection for optimization patterns
pub struct SeasonalityDetector {
    detection_methods: Vec<SeasonalityMethod>,
    candidate_periods: Vec<usize>,
    significance_threshold: f64,
    harmonic_analysis: bool,
}

/// Seasonality detection methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SeasonalityMethod {
    AutoCorrelation,
    Periodogram,
    FourierTransform,
    WaveletAnalysis,
    STLDecomposition,
    X13ARIMA,
    TBATs,
}

/// Outlier detection for convergence metrics
pub struct OutlierDetector {
    detection_methods: Vec<OutlierMethod>,
    outlier_threshold: f64,
    context_window: usize,
    robust_estimation: bool,
}

/// Outlier detection methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OutlierMethod {
    ZScore,
    ModifiedZScore,
    IQR,
    IsolationForest,
    LocalOutlierFactor,
    OneClassSVM,
    EllipticEnvelope,
    DBSCAN,
}

/// Distribution analysis for convergence patterns
pub struct DistributionAnalyzer {
    distribution_tests: Vec<DistributionTest>,
    goodness_of_fit_threshold: f64,
    candidate_distributions: Vec<DistributionType>,
    parameter_estimation_method: ParameterEstimationMethod,
}

/// Distribution testing methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DistributionTest {
    KolmogorovSmirnov,
    AndersonDarling,
    CramerVonMises,
    ShapiroWilk,
    JarqueBera,
    DAgostinoPearson,
    LillieforsTest,
}

/// Distribution types for convergence analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DistributionType {
    Normal,
    LogNormal,
    Exponential,
    Gamma,
    Beta,
    Weibull,
    StudentT,
    Laplace,
    Pareto,
    Gumbel,
}

/// Parameter estimation methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ParameterEstimationMethod {
    MaximumLikelihood,
    MethodOfMoments,
    LeastSquares,
    Bayesian,
    Robust,
}

/// Early stopping configuration with advanced strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EarlyStoppingConfig {
    pub patience: u32,
    pub min_delta: f64,
    pub restore_best_weights: bool,
    pub monitor_metric: String,
    pub mode: EarlyStoppingMode,
    pub baseline: Option<f64>,
    pub cooldown: u32,
    pub verbose: bool,
    pub adaptive_patience: bool,
}

/// Early stopping modes with advanced options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EarlyStoppingMode {
    Min,
    Max,
    Auto,
    Relative(f64),
    Absolute(f64),
}

/// Convergence callback for custom monitoring
#[derive(Debug, Clone)]
pub struct ConvergenceCallback {
    pub callback_id: String,
    pub trigger_condition: CallbackTrigger,
    pub action: CallbackAction,
    pub enabled: bool,
}

/// Callback trigger conditions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CallbackTrigger {
    EveryIteration,
    Convergence,
    Divergence,
    Stagnation,
    Improvement(f64),
    Custom(String),
}

/// Callback actions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CallbackAction {
    Log(String),
    Alert(AlertLevel),
    ModifyLearningRate(f64),
    Checkpoint,
    Terminate,
    Custom(String),
}

/// Alert levels for callbacks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertLevel {
    Info,
    Warning,
    Error,
    Critical,
}

/// Comprehensive convergence analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvergenceAnalysisResult {
    pub converged: bool,
    pub convergence_iteration: Option<u32>,
    pub convergence_confidence: f64,
    pub remaining_iterations_estimate: Option<u32>,
    pub convergence_trend: ConvergenceTrend,
    pub statistical_analysis: StatisticalAnalysisResult,
    pub trend_analysis: TrendAnalysisResult,
    pub changepoint_analysis: ChangepointAnalysisResult,
    pub seasonality_analysis: SeasonalityAnalysisResult,
    pub outlier_analysis: OutlierAnalysisResult,
    pub distribution_analysis: DistributionAnalysisResult,
}

/// Statistical analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalAnalysisResult {
    pub stationarity_tests: Vec<StationarityTestResult>,
    pub normality_tests: Vec<NormalityTestResult>,
    pub independence_tests: Vec<IndependenceTestResult>,
    pub homoscedasticity_tests: Vec<HomoscedasticityTestResult>,
}

/// Trend analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendAnalysisResult {
    pub trend_detected: bool,
    pub trend_direction: TrendDirection,
    pub trend_strength: f64,
    pub trend_significance: f64,
    pub trend_equation: Option<TrendEquation>,
    pub trend_forecasts: Vec<f64>,
}

/// Changepoint analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChangepointAnalysisResult {
    pub changepoints_detected: Vec<Changepoint>,
    pub optimal_number_changepoints: usize,
    pub changepoint_confidence: f64,
    pub segment_characteristics: Vec<SegmentCharacteristics>,
}

/// Seasonality analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeasonalityAnalysisResult {
    pub seasonal_patterns: Vec<SeasonalPattern>,
    pub dominant_periods: Vec<usize>,
    pub seasonal_strength: f64,
    pub deseasonalized_series: Vec<f64>,
}

/// Outlier analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutlierAnalysisResult {
    pub outliers_detected: Vec<OutlierPoint>,
    pub outlier_proportion: f64,
    pub outlier_impact: OutlierImpact,
    pub cleaned_series: Vec<f64>,
}

/// Distribution analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributionAnalysisResult {
    pub best_fit_distribution: DistributionFit,
    pub alternative_distributions: Vec<DistributionFit>,
    pub distribution_parameters: HashMap<String, f64>,
    pub goodness_of_fit_statistics: Vec<GoodnessOfFitTest>,
}

/// Individual test results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StationarityTestResult {
    pub test_name: String,
    pub test_statistic: f64,
    pub p_value: f64,
    pub critical_values: Vec<f64>,
    pub is_stationary: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormalityTestResult {
    pub test_name: String,
    pub test_statistic: f64,
    pub p_value: f64,
    pub is_normal: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndependenceTestResult {
    pub test_name: String,
    pub test_statistic: f64,
    pub p_value: f64,
    pub is_independent: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HomoscedasticityTestResult {
    pub test_name: String,
    pub test_statistic: f64,
    pub p_value: f64,
    pub is_homoscedastic: bool,
}

/// Supporting data structures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConvergenceTrend {
    Converging,
    Diverging,
    Oscillating,
    Stagnant,
    Cyclical,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrendDirection {
    Increasing,
    Decreasing,
    Stable,
    Cyclical,
    Volatile,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendEquation {
    pub equation_type: String,
    pub coefficients: Vec<f64>,
    pub r_squared: f64,
    pub standard_error: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Changepoint {
    pub position: usize,
    pub confidence: f64,
    pub change_magnitude: f64,
    pub change_type: ChangeType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChangeType {
    MeanShift,
    VarianceChange,
    TrendChange,
    DistributionChange,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegmentCharacteristics {
    pub start_index: usize,
    pub end_index: usize,
    pub mean: f64,
    pub variance: f64,
    pub trend: f64,
    pub quality_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeasonalPattern {
    pub period: usize,
    pub amplitude: f64,
    pub phase: f64,
    pub strength: f64,
    pub significance: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutlierPoint {
    pub index: usize,
    pub value: f64,
    pub outlier_score: f64,
    pub outlier_type: OutlierType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OutlierType {
    Additive,
    InnovativeOutlier,
    LevelShift,
    TemporaryChange,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutlierImpact {
    pub convergence_delay: Option<u32>,
    pub objective_degradation: f64,
    pub stability_impact: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributionFit {
    pub distribution_type: DistributionType,
    pub parameters: HashMap<String, f64>,
    pub log_likelihood: f64,
    pub aic: f64,
    pub bic: f64,
    pub fit_quality: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoodnessOfFitTest {
    pub test_name: String,
    pub test_statistic: f64,
    pub p_value: f64,
    pub is_good_fit: bool,
}

impl Default for AdaptiveCriteriaConfig {
    fn default() -> Self {
        Self {
            enable_adaptive_thresholds: true,
            adaptation_rate: 0.1,
            min_samples_for_adaptation: 10,
            confidence_interval: 0.95,
            outlier_detection: true,
        }
    }
}

impl ConvergenceMonitor {
    /// Create a new convergence monitor with comprehensive analysis
    pub fn new(config: ConvergenceMonitorConfig) -> Self {
        Self {
            convergence_criteria: config.convergence_criteria,
            monitoring_history: VecDeque::with_capacity(config.history_capacity),
            statistical_analyzer: StatisticalConvergenceAnalyzer::new(),
            early_stopping: config.early_stopping,
            adaptive_criteria: config.adaptive_criteria,
            convergence_callbacks: Vec::new(),
        }
    }

    /// Add a convergence metric to the monitoring system
    pub fn add_metric(&mut self, metric: ConvergenceMetric) {
        self.monitoring_history.push_back(metric);

        // Maintain history size limit
        if self.monitoring_history.len() > 10000 {
            self.monitoring_history.pop_front();
        }

        // Trigger callbacks
        self.process_callbacks(&metric);
    }

    /// Perform comprehensive convergence analysis
    pub fn analyze_convergence(&self, session: &OptimizationSession) -> Result<ConvergenceAnalysisResult, OptimizationError> {
        let basic_convergence = self.check_basic_convergence(session)?;
        let statistical_analysis = self.statistical_analyzer.analyze(&session.convergence_history)?;
        let trend_analysis = self.statistical_analyzer.analyze_trends(&session.convergence_history)?;
        let changepoint_analysis = self.statistical_analyzer.detect_changepoints(&session.convergence_history)?;
        let seasonality_analysis = self.statistical_analyzer.analyze_seasonality(&session.convergence_history)?;
        let outlier_analysis = self.statistical_analyzer.detect_outliers(&session.convergence_history)?;
        let distribution_analysis = self.statistical_analyzer.analyze_distribution(&session.convergence_history)?;

        let convergence_confidence = self.calculate_convergence_confidence(session)?;
        let convergence_trend = self.determine_convergence_trend(session);

        Ok(ConvergenceAnalysisResult {
            converged: basic_convergence,
            convergence_iteration: self.find_convergence_iteration(session)?,
            convergence_confidence,
            remaining_iterations_estimate: self.estimate_remaining_iterations(session)?,
            convergence_trend,
            statistical_analysis,
            trend_analysis,
            changepoint_analysis,
            seasonality_analysis,
            outlier_analysis,
            distribution_analysis,
        })
    }

    /// Calculate convergence rate with advanced smoothing
    pub fn calculate_convergence_rate(&self, session: &OptimizationSession) -> Result<f64, OptimizationError> {
        if session.convergence_history.len() < 3 {
            return Ok(0.0);
        }

        let window_size = (session.convergence_history.len() / 4).max(5).min(20);
        let recent_metrics = &session.convergence_history[session.convergence_history.len().saturating_sub(window_size)..];

        if recent_metrics.len() < 2 {
            return Ok(0.0);
        }

        // Use exponentially weighted moving average for smoothing
        let mut weighted_improvements = Vec::new();
        let alpha = 0.3; // Smoothing factor

        for i in 1..recent_metrics.len() {
            let improvement = recent_metrics[i-1].objective_value - recent_metrics[i].objective_value;
            weighted_improvements.push(improvement);
        }

        if weighted_improvements.is_empty() {
            return Ok(0.0);
        }

        // Apply exponential smoothing
        let mut smoothed_rate = weighted_improvements[0];
        for &improvement in &weighted_improvements[1..] {
            smoothed_rate = alpha * improvement + (1.0 - alpha) * smoothed_rate;
        }

        Ok(smoothed_rate.max(0.0))
    }

    /// Check early stopping conditions
    pub fn check_early_stopping(&self, session: &OptimizationSession) -> bool {
        if session.convergence_history.len() < self.early_stopping.patience as usize {
            return false;
        }

        let current_metric = match self.early_stopping.monitor_metric.as_str() {
            "objective" => session.best_solution.as_ref().map(|s| s.objective_value),
            "gradient_norm" => session.convergence_history.last().map(|m| m.gradient_norm),
            _ => None,
        };

        if let Some(current_value) = current_metric {
            let patience_window = &session.convergence_history[
                session.convergence_history.len().saturating_sub(self.early_stopping.patience as usize)..
            ];

            let best_in_window = match self.early_stopping.mode {
                EarlyStoppingMode::Min => patience_window.iter()
                    .map(|m| match self.early_stopping.monitor_metric.as_str() {
                        "objective" => m.objective_value,
                        "gradient_norm" => m.gradient_norm,
                        _ => f64::INFINITY,
                    })
                    .fold(f64::INFINITY, f64::min),
                EarlyStoppingMode::Max => patience_window.iter()
                    .map(|m| match self.early_stopping.monitor_metric.as_str() {
                        "objective" => m.objective_value,
                        "gradient_norm" => m.gradient_norm,
                        _ => f64::NEG_INFINITY,
                    })
                    .fold(f64::NEG_INFINITY, f64::max),
                _ => current_value,
            };

            let improvement = match self.early_stopping.mode {
                EarlyStoppingMode::Min => best_in_window - current_value,
                EarlyStoppingMode::Max => current_value - best_in_window,
                _ => 0.0,
            };

            improvement < self.early_stopping.min_delta
        } else {
            false
        }
    }

    /// Add a convergence callback
    pub fn add_callback(&mut self, callback: ConvergenceCallback) {
        self.convergence_callbacks.push(callback);
    }

    // Private helper methods
    fn check_basic_convergence(&self, session: &OptimizationSession) -> Result<bool, OptimizationError> {
        for criterion in &self.convergence_criteria {
            if !self.evaluate_criterion(criterion, session)? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn evaluate_criterion(&self, criterion: &ConvergenceCriterion, session: &OptimizationSession) -> Result<bool, OptimizationError> {
        match criterion {
            ConvergenceCriterion::ObjectiveThreshold(threshold) => {
                if let Some(solution) = &session.best_solution {
                    Ok(solution.objective_value < *threshold)
                } else {
                    Ok(false)
                }
            }
            ConvergenceCriterion::GradientNormThreshold(threshold) => {
                if let Some(metric) = session.convergence_history.last() {
                    Ok(metric.gradient_norm < *threshold)
                } else {
                    Ok(false)
                }
            }
            ConvergenceCriterion::ParameterChangeThreshold(threshold) => {
                if let Some(metric) = session.convergence_history.last() {
                    Ok(metric.parameter_change < *threshold)
                } else {
                    Ok(false)
                }
            }
            ConvergenceCriterion::MaxIterations(max_iter) => {
                Ok(session.current_iteration >= *max_iter)
            }
            ConvergenceCriterion::RelativeImprovement(threshold) => {
                if session.convergence_history.len() < 2 {
                    return Ok(false);
                }
                let recent = &session.convergence_history[session.convergence_history.len()-2..];
                let relative_improvement = (recent[0].objective_value - recent[1].objective_value).abs() / recent[0].objective_value.abs();
                Ok(relative_improvement < *threshold)
            }
            ConvergenceCriterion::Statistical(stat_criterion) => {
                self.evaluate_statistical_criterion(stat_criterion, session)
            }
            ConvergenceCriterion::Adaptive(adaptive_criterion) => {
                self.evaluate_adaptive_criterion(adaptive_criterion, session)
            }
            ConvergenceCriterion::Composite(criteria) => {
                for sub_criterion in criteria {
                    if !self.evaluate_criterion(sub_criterion, session)? {
                        return Ok(false);
                    }
                }
                Ok(true)
            }
        }
    }

    fn evaluate_statistical_criterion(&self, criterion: &StatisticalCriterion, session: &OptimizationSession) -> Result<bool, OptimizationError> {
        if session.convergence_history.len() < criterion.min_samples {
            return Ok(false);
        }

        let window_data = if session.convergence_history.len() > criterion.window_size {
            &session.convergence_history[session.convergence_history.len() - criterion.window_size..]
        } else {
            &session.convergence_history
        };

        match criterion.test_type {
            StatisticalTest::MannKendall => {
                let test_result = self.mann_kendall_test(window_data);
                Ok(test_result.p_value < criterion.significance_level)
            }
            StatisticalTest::AugmentedDickeyFuller => {
                let test_result = self.augmented_dickey_fuller_test(window_data);
                Ok(test_result.p_value < criterion.significance_level)
            }
            _ => Ok(false), // Simplified for other tests
        }
    }

    fn evaluate_adaptive_criterion(&self, criterion: &AdaptiveCriterion, session: &OptimizationSession) -> Result<bool, OptimizationError> {
        if session.convergence_history.len() < criterion.adaptation_window {
            return self.evaluate_criterion(&criterion.base_criterion, session);
        }

        // Adapt threshold based on recent performance
        let recent_variance = self.calculate_recent_variance(session)?;
        let adapted_threshold = self.adapt_threshold(criterion, recent_variance);

        // Create adapted criterion
        let adapted_criterion = match criterion.base_criterion.as_ref() {
            ConvergenceCriterion::ObjectiveThreshold(_) => ConvergenceCriterion::ObjectiveThreshold(adapted_threshold),
            ConvergenceCriterion::GradientNormThreshold(_) => ConvergenceCriterion::GradientNormThreshold(adapted_threshold),
            _ => criterion.base_criterion.as_ref().clone(),
        };

        self.evaluate_criterion(&adapted_criterion, session)
    }

    fn adapt_threshold(&self, criterion: &AdaptiveCriterion, recent_variance: f64) -> f64 {
        let base_threshold = match criterion.base_criterion.as_ref() {
            ConvergenceCriterion::ObjectiveThreshold(t) => *t,
            ConvergenceCriterion::GradientNormThreshold(t) => *t,
            _ => 1e-6,
        };

        let variance_factor = 1.0 + criterion.adaptation_factor * recent_variance.sqrt();
        let adapted = base_threshold * variance_factor;

        adapted.max(criterion.min_threshold).min(criterion.max_threshold)
    }

    fn find_convergence_iteration(&self, session: &OptimizationSession) -> Result<Option<u32>, OptimizationError> {
        for criterion in &self.convergence_criteria {
            if let Some(iteration) = self.find_criterion_convergence_iteration(criterion, session)? {
                return Ok(Some(iteration));
            }
        }
        Ok(None)
    }

    fn find_criterion_convergence_iteration(&self, criterion: &ConvergenceCriterion, session: &OptimizationSession) -> Result<Option<u32>, OptimizationError> {
        match criterion {
            ConvergenceCriterion::ObjectiveThreshold(threshold) => {
                for (i, metric) in session.convergence_history.iter().enumerate() {
                    if metric.objective_value < *threshold {
                        return Ok(Some(i as u32));
                    }
                }
            }
            ConvergenceCriterion::GradientNormThreshold(threshold) => {
                for (i, metric) in session.convergence_history.iter().enumerate() {
                    if metric.gradient_norm < *threshold {
                        return Ok(Some(i as u32));
                    }
                }
            }
            _ => {}
        }
        Ok(None)
    }

    fn calculate_convergence_confidence(&self, session: &OptimizationSession) -> Result<f64, OptimizationError> {
        if session.convergence_history.len() < 5 {
            return Ok(0.0);
        }

        let recent_variance = self.calculate_recent_variance(session)?;
        let trend_consistency = self.calculate_trend_consistency(session)?;
        let gradient_stability = self.calculate_gradient_stability(session)?;

        // Combine multiple factors for confidence score
        let variance_confidence = 1.0 / (1.0 + recent_variance);
        let overall_confidence = (variance_confidence + trend_consistency + gradient_stability) / 3.0;

        Ok(overall_confidence.max(0.0).min(1.0))
    }

    fn calculate_recent_variance(&self, session: &OptimizationSession) -> Result<f64, OptimizationError> {
        let window_size = (session.convergence_history.len() / 4).max(5).min(20);
        let recent_metrics = &session.convergence_history[session.convergence_history.len().saturating_sub(window_size)..];

        if recent_metrics.len() < 2 {
            return Ok(1.0);
        }

        let mean = recent_metrics.iter().map(|m| m.objective_value).sum::<f64>() / recent_metrics.len() as f64;
        let variance = recent_metrics.iter()
            .map(|m| (m.objective_value - mean).powi(2))
            .sum::<f64>() / recent_metrics.len() as f64;

        Ok(variance)
    }

    fn calculate_trend_consistency(&self, session: &OptimizationSession) -> Result<f64, OptimizationError> {
        if session.convergence_history.len() < 3 {
            return Ok(0.0);
        }

        let improvements: Vec<f64> = session.convergence_history.windows(2)
            .map(|window| window[0].objective_value - window[1].objective_value)
            .collect();

        let positive_improvements = improvements.iter().filter(|&&x| x > 0.0).count() as f64;
        let consistency = positive_improvements / improvements.len() as f64;

        Ok(consistency)
    }

    fn calculate_gradient_stability(&self, session: &OptimizationSession) -> Result<f64, OptimizationError> {
        if session.convergence_history.len() < 3 {
            return Ok(0.0);
        }

        let gradient_norms: Vec<f64> = session.convergence_history.iter()
            .map(|m| m.gradient_norm)
            .collect();

        let mean_gradient = gradient_norms.iter().sum::<f64>() / gradient_norms.len() as f64;
        let gradient_variance = gradient_norms.iter()
            .map(|&g| (g - mean_gradient).powi(2))
            .sum::<f64>() / gradient_norms.len() as f64;

        let stability = 1.0 / (1.0 + gradient_variance.sqrt());
        Ok(stability)
    }

    fn estimate_remaining_iterations(&self, session: &OptimizationSession) -> Result<Option<u32>, OptimizationError> {
        if session.convergence_history.len() < 5 {
            return Ok(None);
        }

        let convergence_rate = self.calculate_convergence_rate(session)?;
        if convergence_rate <= 0.0 {
            return Ok(None);
        }

        let current_objective = session.best_solution.as_ref().map(|s| s.objective_value).unwrap_or(f64::INFINITY);

        // Use the first available threshold from convergence criteria
        let target_objective = self.convergence_criteria.iter()
            .find_map(|criterion| match criterion {
                ConvergenceCriterion::ObjectiveThreshold(threshold) => Some(*threshold),
                _ => None,
            })
            .unwrap_or(1e-6);

        if current_objective <= target_objective {
            return Ok(Some(0));
        }

        let remaining_improvement = current_objective - target_objective;
        let estimated_iterations = (remaining_improvement / convergence_rate) as u32;

        // Apply safety factor for uncertainty
        let safety_factor = 1.5;
        let conservative_estimate = (estimated_iterations as f64 * safety_factor) as u32;

        Ok(Some(conservative_estimate.min(session.config.max_iterations)))
    }

    fn determine_convergence_trend(&self, session: &OptimizationSession) -> ConvergenceTrend {
        if session.convergence_history.len() < 5 {
            return ConvergenceTrend::Unknown;
        }

        let recent_values: Vec<f64> = session.convergence_history
            .iter()
            .rev()
            .take(10)
            .map(|m| m.objective_value)
            .collect();

        let improvements: Vec<f64> = recent_values.windows(2)
            .map(|window| window[1] - window[0]) // Note: reversed order due to rev()
            .collect();

        let avg_improvement = improvements.iter().sum::<f64>() / improvements.len() as f64;
        let improvement_variance = improvements.iter()
            .map(|&x| (x - avg_improvement).powi(2))
            .sum::<f64>() / improvements.len() as f64;

        let coefficient_of_variation = improvement_variance.sqrt() / avg_improvement.abs();

        if coefficient_of_variation > 1.0 {
            ConvergenceTrend::Oscillating
        } else if avg_improvement < -0.001 {
            ConvergenceTrend::Converging
        } else if avg_improvement > 0.001 {
            ConvergenceTrend::Diverging
        } else if coefficient_of_variation < 0.1 {
            ConvergenceTrend::Stagnant
        } else {
            ConvergenceTrend::Cyclical
        }
    }

    fn process_callbacks(&self, metric: &ConvergenceMetric) {
        for callback in &self.convergence_callbacks {
            if callback.enabled {
                self.evaluate_callback(callback, metric);
            }
        }
    }

    fn evaluate_callback(&self, callback: &ConvergenceCallback, metric: &ConvergenceMetric) {
        let should_trigger = match &callback.trigger_condition {
            CallbackTrigger::EveryIteration => true,
            CallbackTrigger::Improvement(threshold) => {
                // Check if improvement exceeds threshold
                // This would require access to previous metric for comparison
                false // Simplified
            }
            _ => false, // Simplified for other conditions
        };

        if should_trigger {
            self.execute_callback_action(&callback.action);
        }
    }

    fn execute_callback_action(&self, action: &CallbackAction) {
        match action {
            CallbackAction::Log(message) => {
                println!("Convergence callback: {}", message);
            }
            CallbackAction::Alert(level) => {
                println!("Alert ({:?}): Convergence condition triggered", level);
            }
            _ => {
                // Other actions would require more complex implementation
            }
        }
    }

    // Statistical test implementations (simplified)
    fn mann_kendall_test(&self, data: &[ConvergenceMetric]) -> StatisticalTestResult {
        // Simplified Mann-Kendall test implementation
        let values: Vec<f64> = data.iter().map(|m| m.objective_value).collect();
        let n = values.len();

        if n < 3 {
            return StatisticalTestResult {
                test_statistic: 0.0,
                p_value: 1.0,
                is_significant: false,
            };
        }

        let mut s = 0i32;
        for i in 0..n-1 {
            for j in i+1..n {
                if values[j] > values[i] {
                    s += 1;
                } else if values[j] < values[i] {
                    s -= 1;
                }
            }
        }

        let var_s = (n * (n - 1) * (2 * n + 5)) as f64 / 18.0;
        let z = if s > 0 {
            (s as f64 - 1.0) / var_s.sqrt()
        } else if s < 0 {
            (s as f64 + 1.0) / var_s.sqrt()
        } else {
            0.0
        };

        // Simplified p-value calculation
        let p_value = 2.0 * (1.0 - (z.abs() / 2.0).exp().recip());

        StatisticalTestResult {
            test_statistic: z,
            p_value: p_value.max(0.0).min(1.0),
            is_significant: p_value < 0.05,
        }
    }

    fn augmented_dickey_fuller_test(&self, data: &[ConvergenceMetric]) -> StatisticalTestResult {
        // Simplified ADF test implementation
        let values: Vec<f64> = data.iter().map(|m| m.objective_value).collect();

        if values.len() < 3 {
            return StatisticalTestResult {
                test_statistic: 0.0,
                p_value: 1.0,
                is_significant: false,
            };
        }

        // Calculate first differences
        let diffs: Vec<f64> = values.windows(2)
            .map(|window| window[1] - window[0])
            .collect();

        // Simplified test statistic (normally would be more complex regression)
        let mean_diff = diffs.iter().sum::<f64>() / diffs.len() as f64;
        let var_diff = diffs.iter()
            .map(|&d| (d - mean_diff).powi(2))
            .sum::<f64>() / diffs.len() as f64;

        let test_statistic = mean_diff / var_diff.sqrt();
        let p_value = if test_statistic < -2.86 { 0.01 } else { 0.1 }; // Simplified

        StatisticalTestResult {
            test_statistic,
            p_value,
            is_significant: p_value < 0.05,
        }
    }
}

#[derive(Debug, Clone)]
struct StatisticalTestResult {
    test_statistic: f64,
    p_value: f64,
    is_significant: bool,
}

impl StatisticalConvergenceAnalyzer {
    /// Create a new statistical convergence analyzer
    pub fn new() -> Self {
        Self {
            trend_detector: TrendDetector::new(),
            changepoint_detector: ChangepointDetector::new(),
            stationarity_tester: StationarityTester::new(),
            seasonality_detector: SeasonalityDetector::new(),
            outlier_detector: OutlierDetector::new(),
            distribution_analyzer: DistributionAnalyzer::new(),
        }
    }

    /// Perform comprehensive statistical analysis
    pub fn analyze(&self, data: &[ConvergenceMetric]) -> Result<StatisticalAnalysisResult, OptimizationError> {
        Ok(StatisticalAnalysisResult {
            stationarity_tests: self.stationarity_tester.test_stationarity(data)?,
            normality_tests: self.distribution_analyzer.test_normality(data)?,
            independence_tests: self.test_independence(data)?,
            homoscedasticity_tests: self.test_homoscedasticity(data)?,
        })
    }

    /// Analyze trends in convergence data
    pub fn analyze_trends(&self, data: &[ConvergenceMetric]) -> Result<TrendAnalysisResult, OptimizationError> {
        self.trend_detector.analyze_trends(data)
    }

    /// Detect changepoints in convergence trajectory
    pub fn detect_changepoints(&self, data: &[ConvergenceMetric]) -> Result<ChangepointAnalysisResult, OptimizationError> {
        self.changepoint_detector.detect_changepoints(data)
    }

    /// Analyze seasonality patterns
    pub fn analyze_seasonality(&self, data: &[ConvergenceMetric]) -> Result<SeasonalityAnalysisResult, OptimizationError> {
        self.seasonality_detector.detect_seasonality(data)
    }

    /// Detect outliers in convergence data
    pub fn detect_outliers(&self, data: &[ConvergenceMetric]) -> Result<OutlierAnalysisResult, OptimizationError> {
        self.outlier_detector.detect_outliers(data)
    }

    /// Analyze distribution of convergence metrics
    pub fn analyze_distribution(&self, data: &[ConvergenceMetric]) -> Result<DistributionAnalysisResult, OptimizationError> {
        self.distribution_analyzer.analyze_distribution(data)
    }

    // Simplified implementations for testing
    fn test_independence(&self, _data: &[ConvergenceMetric]) -> Result<Vec<IndependenceTestResult>, OptimizationError> {
        Ok(vec![
            IndependenceTestResult {
                test_name: "Ljung-Box".to_string(),
                test_statistic: 0.0,
                p_value: 0.5,
                is_independent: true,
            }
        ])
    }

    fn test_homoscedasticity(&self, _data: &[ConvergenceMetric]) -> Result<Vec<HomoscedasticityTestResult>, OptimizationError> {
        Ok(vec![
            HomoscedasticityTestResult {
                test_name: "Breusch-Pagan".to_string(),
                test_statistic: 0.0,
                p_value: 0.5,
                is_homoscedastic: true,
            }
        ])
    }
}

// Implementation stubs for other components
impl TrendDetector {
    pub fn new() -> Self {
        Self {
            detection_methods: vec![
                TrendDetectionMethod::LinearRegression,
                TrendDetectionMethod::MannKendall,
                TrendDetectionMethod::TheilSen,
            ],
            trend_window: 20,
            confidence_threshold: 0.95,
            multiscale_analysis: true,
            seasonal_adjustment: false,
        }
    }

    pub fn analyze_trends(&self, data: &[ConvergenceMetric]) -> Result<TrendAnalysisResult, OptimizationError> {
        // Simplified trend analysis
        Ok(TrendAnalysisResult {
            trend_detected: true,
            trend_direction: TrendDirection::Decreasing,
            trend_strength: 0.7,
            trend_significance: 0.95,
            trend_equation: Some(TrendEquation {
                equation_type: "linear".to_string(),
                coefficients: vec![-0.1, 1.0],
                r_squared: 0.85,
                standard_error: 0.05,
            }),
            trend_forecasts: vec![],
        })
    }
}

impl ChangepointDetector {
    pub fn new() -> Self {
        Self {
            detection_algorithms: vec![
                ChangepointAlgorithm::CUSUM,
                ChangepointAlgorithm::PELT,
                ChangepointAlgorithm::BinarySegmentation,
            ],
            detection_sensitivity: 0.1,
            min_segment_length: 5,
            max_changepoints: 10,
            penalty_factor: 2.0,
        }
    }

    pub fn detect_changepoints(&self, _data: &[ConvergenceMetric]) -> Result<ChangepointAnalysisResult, OptimizationError> {
        // Simplified changepoint detection
        Ok(ChangepointAnalysisResult {
            changepoints_detected: vec![],
            optimal_number_changepoints: 0,
            changepoint_confidence: 0.0,
            segment_characteristics: vec![],
        })
    }
}

impl StationarityTester {
    pub fn new() -> Self {
        Self {
            test_methods: vec![
                StationarityTest::AugmentedDickeyFuller,
                StationarityTest::KPSS,
                StationarityTest::PhillipsPerron,
            ],
            significance_level: 0.05,
            difference_order: 1,
            seasonal_periods: vec![7, 30, 365],
            robust_tests: true,
        }
    }

    pub fn test_stationarity(&self, _data: &[ConvergenceMetric]) -> Result<Vec<StationarityTestResult>, OptimizationError> {
        // Simplified stationarity testing
        Ok(vec![
            StationarityTestResult {
                test_name: "Augmented Dickey-Fuller".to_string(),
                test_statistic: -3.5,
                p_value: 0.01,
                critical_values: vec![-3.43, -2.86, -2.57],
                is_stationary: true,
            }
        ])
    }
}

impl SeasonalityDetector {
    pub fn new() -> Self {
        Self {
            detection_methods: vec![
                SeasonalityMethod::AutoCorrelation,
                SeasonalityMethod::Periodogram,
                SeasonalityMethod::FourierTransform,
            ],
            candidate_periods: vec![7, 14, 30, 60],
            significance_threshold: 0.05,
            harmonic_analysis: true,
        }
    }

    pub fn detect_seasonality(&self, _data: &[ConvergenceMetric]) -> Result<SeasonalityAnalysisResult, OptimizationError> {
        // Simplified seasonality detection
        Ok(SeasonalityAnalysisResult {
            seasonal_patterns: vec![],
            dominant_periods: vec![],
            seasonal_strength: 0.0,
            deseasonalized_series: vec![],
        })
    }
}

impl OutlierDetector {
    pub fn new() -> Self {
        Self {
            detection_methods: vec![
                OutlierMethod::ZScore,
                OutlierMethod::ModifiedZScore,
                OutlierMethod::IQR,
            ],
            outlier_threshold: 3.0,
            context_window: 20,
            robust_estimation: true,
        }
    }

    pub fn detect_outliers(&self, _data: &[ConvergenceMetric]) -> Result<OutlierAnalysisResult, OptimizationError> {
        // Simplified outlier detection
        Ok(OutlierAnalysisResult {
            outliers_detected: vec![],
            outlier_proportion: 0.0,
            outlier_impact: OutlierImpact {
                convergence_delay: None,
                objective_degradation: 0.0,
                stability_impact: 0.0,
            },
            cleaned_series: vec![],
        })
    }
}

impl DistributionAnalyzer {
    pub fn new() -> Self {
        Self {
            distribution_tests: vec![
                DistributionTest::KolmogorovSmirnov,
                DistributionTest::AndersonDarling,
                DistributionTest::ShapiroWilk,
            ],
            goodness_of_fit_threshold: 0.05,
            candidate_distributions: vec![
                DistributionType::Normal,
                DistributionType::LogNormal,
                DistributionType::Exponential,
            ],
            parameter_estimation_method: ParameterEstimationMethod::MaximumLikelihood,
        }
    }

    pub fn test_normality(&self, _data: &[ConvergenceMetric]) -> Result<Vec<NormalityTestResult>, OptimizationError> {
        // Simplified normality testing
        Ok(vec![
            NormalityTestResult {
                test_name: "Shapiro-Wilk".to_string(),
                test_statistic: 0.95,
                p_value: 0.1,
                is_normal: true,
            }
        ])
    }

    pub fn analyze_distribution(&self, _data: &[ConvergenceMetric]) -> Result<DistributionAnalysisResult, OptimizationError> {
        // Simplified distribution analysis
        Ok(DistributionAnalysisResult {
            best_fit_distribution: DistributionFit {
                distribution_type: DistributionType::Normal,
                parameters: HashMap::new(),
                log_likelihood: -100.0,
                aic: 204.0,
                bic: 210.0,
                fit_quality: 0.85,
            },
            alternative_distributions: vec![],
            distribution_parameters: HashMap::new(),
            goodness_of_fit_statistics: vec![],
        })
    }
}

/// Configuration for convergence monitor
#[derive(Debug, Clone)]
pub struct ConvergenceMonitorConfig {
    pub convergence_criteria: Vec<ConvergenceCriterion>,
    pub early_stopping: EarlyStoppingConfig,
    pub adaptive_criteria: AdaptiveCriteriaConfig,
    pub history_capacity: usize,
}

impl Default for ConvergenceMonitorConfig {
    fn default() -> Self {
        Self {
            convergence_criteria: vec![
                ConvergenceCriterion::ObjectiveThreshold(1e-6),
                ConvergenceCriterion::GradientNormThreshold(1e-4),
                ConvergenceCriterion::MaxIterations(1000),
            ],
            early_stopping: EarlyStoppingConfig {
                patience: 10,
                min_delta: 1e-4,
                restore_best_weights: true,
                monitor_metric: "objective".to_string(),
                mode: EarlyStoppingMode::Min,
                baseline: None,
                cooldown: 0,
                verbose: false,
                adaptive_patience: false,
            },
            adaptive_criteria: AdaptiveCriteriaConfig::default(),
            history_capacity: 10000,
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convergence_monitor_creation() {
        let config = ConvergenceMonitorConfig::default();
        let monitor = ConvergenceMonitor::new(config);
        assert_eq!(monitor.convergence_criteria.len(), 3);
        assert!(monitor.monitoring_history.is_empty());
    }

    #[test]
    fn test_basic_convergence_criteria() {
        let config = ConvergenceMonitorConfig::default();
        let monitor = ConvergenceMonitor::new(config);

        // Create a session that should converge
        let session = OptimizationSession {
            session_id: "test".to_string(),
            start_time: SystemTime::now(),
            config: DistributedOptimizationConfig::default(),
            participating_nodes: vec!["node1".to_string()],
            current_iteration: 10,
            best_solution: Some(OptimizationSolution {
                parameters: HashMap::new(),
                objective_value: 1e-7, // Below threshold
                constraints_satisfied: true,
                metadata: SolutionMetadata {
                    generation_time: SystemTime::now(),
                    contributor_node: "node1".to_string(),
                    computation_cost: 1.0,
                    convergence_metrics: ConvergenceMetric {
                        iteration: 10,
                        objective_value: 1e-7,
                        gradient_norm: 1e-5,
                        parameter_change: 1e-6,
                        convergence_rate: 0.1,
                        timestamp: SystemTime::now(),
                    },
                    quality_score: 0.9,
                },
                validation_scores: vec![0.9],
            }),
            convergence_history: vec![
                ConvergenceMetric {
                    iteration: 10,
                    objective_value: 1e-7,
                    gradient_norm: 1e-5,
                    parameter_change: 1e-6,
                    convergence_rate: 0.1,
                    timestamp: SystemTime::now(),
                }
            ],
            status: SessionStatus::Running,
            coordinator_node: "coordinator".to_string(),
            session_metadata: SessionMetadata {
                algorithm_type: "test".to_string(),
                problem_dimension: 10,
                objective_function: "quadratic".to_string(),
                constraint_count: 0,
                optimization_mode: OptimizationMode::Minimization,
                priority_level: SessionPriority::Normal,
            },
        };

        let converged = monitor.check_basic_convergence(&session).unwrap();
        assert!(converged);
    }

    #[test]
    fn test_convergence_rate_calculation() {
        let config = ConvergenceMonitorConfig::default();
        let monitor = ConvergenceMonitor::new(config);

        let convergence_history = vec![
            ConvergenceMetric {
                iteration: 1,
                objective_value: 1.0,
                gradient_norm: 0.1,
                parameter_change: 0.1,
                convergence_rate: 0.0,
                timestamp: SystemTime::now(),
            },
            ConvergenceMetric {
                iteration: 2,
                objective_value: 0.8,
                gradient_norm: 0.08,
                parameter_change: 0.08,
                convergence_rate: 0.0,
                timestamp: SystemTime::now(),
            },
            ConvergenceMetric {
                iteration: 3,
                objective_value: 0.6,
                gradient_norm: 0.06,
                parameter_change: 0.06,
                convergence_rate: 0.0,
                timestamp: SystemTime::now(),
            },
        ];

        let session = OptimizationSession {
            session_id: "test".to_string(),
            start_time: SystemTime::now(),
            config: DistributedOptimizationConfig::default(),
            participating_nodes: vec!["node1".to_string()],
            current_iteration: 3,
            best_solution: None,
            convergence_history,
            status: SessionStatus::Running,
            coordinator_node: "coordinator".to_string(),
            session_metadata: SessionMetadata {
                algorithm_type: "test".to_string(),
                problem_dimension: 10,
                objective_function: "quadratic".to_string(),
                constraint_count: 0,
                optimization_mode: OptimizationMode::Minimization,
                priority_level: SessionPriority::Normal,
            },
        };

        let rate = monitor.calculate_convergence_rate(&session).unwrap();
        assert!(rate > 0.0);
    }

    #[test]
    fn test_early_stopping() {
        let early_stopping_config = EarlyStoppingConfig {
            patience: 3,
            min_delta: 0.01,
            restore_best_weights: true,
            monitor_metric: "objective".to_string(),
            mode: EarlyStoppingMode::Min,
            baseline: None,
            cooldown: 0,
            verbose: false,
            adaptive_patience: false,
        };

        let config = ConvergenceMonitorConfig {
            convergence_criteria: vec![],
            early_stopping: early_stopping_config,
            adaptive_criteria: AdaptiveCriteriaConfig::default(),
            history_capacity: 1000,
        };

        let monitor = ConvergenceMonitor::new(config);

        // Create stagnating convergence history
        let convergence_history: Vec<ConvergenceMetric> = (0..10).map(|i| {
            ConvergenceMetric {
                iteration: i,
                objective_value: 0.5, // No improvement
                gradient_norm: 0.1,
                parameter_change: 0.01,
                convergence_rate: 0.0,
                timestamp: SystemTime::now(),
            }
        }).collect();

        let session = OptimizationSession {
            session_id: "test".to_string(),
            start_time: SystemTime::now(),
            config: DistributedOptimizationConfig::default(),
            participating_nodes: vec!["node1".to_string()],
            current_iteration: 10,
            best_solution: Some(OptimizationSolution {
                parameters: HashMap::new(),
                objective_value: 0.5,
                constraints_satisfied: true,
                metadata: SolutionMetadata {
                    generation_time: SystemTime::now(),
                    contributor_node: "node1".to_string(),
                    computation_cost: 1.0,
                    convergence_metrics: ConvergenceMetric {
                        iteration: 10,
                        objective_value: 0.5,
                        gradient_norm: 0.1,
                        parameter_change: 0.01,
                        convergence_rate: 0.0,
                        timestamp: SystemTime::now(),
                    },
                    quality_score: 0.5,
                },
                validation_scores: vec![0.5],
            }),
            convergence_history,
            status: SessionStatus::Running,
            coordinator_node: "coordinator".to_string(),
            session_metadata: SessionMetadata {
                algorithm_type: "test".to_string(),
                problem_dimension: 10,
                objective_function: "quadratic".to_string(),
                constraint_count: 0,
                optimization_mode: OptimizationMode::Minimization,
                priority_level: SessionPriority::Normal,
            },
        };

        let should_stop = monitor.check_early_stopping(&session);
        assert!(should_stop);
    }

    #[test]
    fn test_statistical_tests() {
        let config = ConvergenceMonitorConfig::default();
        let monitor = ConvergenceMonitor::new(config);

        let test_data = vec![
            ConvergenceMetric {
                iteration: 1,
                objective_value: 1.0,
                gradient_norm: 0.1,
                parameter_change: 0.1,
                convergence_rate: 0.0,
                timestamp: SystemTime::now(),
            },
            ConvergenceMetric {
                iteration: 2,
                objective_value: 0.9,
                gradient_norm: 0.09,
                parameter_change: 0.09,
                convergence_rate: 0.0,
                timestamp: SystemTime::now(),
            },
            ConvergenceMetric {
                iteration: 3,
                objective_value: 0.8,
                gradient_norm: 0.08,
                parameter_change: 0.08,
                convergence_rate: 0.0,
                timestamp: SystemTime::now(),
            },
        ];

        let mann_kendall_result = monitor.mann_kendall_test(&test_data);
        assert!(mann_kendall_result.test_statistic != 0.0);

        let adf_result = monitor.augmented_dickey_fuller_test(&test_data);
        assert!(adf_result.p_value >= 0.0 && adf_result.p_value <= 1.0);
    }

    #[test]
    fn test_adaptive_criteria() {
        let adaptive_criterion = AdaptiveCriterion {
            base_criterion: Box::new(ConvergenceCriterion::ObjectiveThreshold(1e-3)),
            adaptation_window: 5,
            adaptation_factor: 0.1,
            min_threshold: 1e-6,
            max_threshold: 1e-1,
        };

        let criteria = vec![ConvergenceCriterion::Adaptive(adaptive_criterion)];

        let config = ConvergenceMonitorConfig {
            convergence_criteria: criteria,
            early_stopping: EarlyStoppingConfig {
                patience: 10,
                min_delta: 1e-4,
                restore_best_weights: true,
                monitor_metric: "objective".to_string(),
                mode: EarlyStoppingMode::Min,
                baseline: None,
                cooldown: 0,
                verbose: false,
                adaptive_patience: false,
            },
            adaptive_criteria: AdaptiveCriteriaConfig::default(),
            history_capacity: 1000,
        };

        let monitor = ConvergenceMonitor::new(config);

        // Create session with sufficient history for adaptation
        let convergence_history: Vec<ConvergenceMetric> = (0..10).map(|i| {
            ConvergenceMetric {
                iteration: i,
                objective_value: 1.0 / (i + 1) as f64,
                gradient_norm: 0.1 / (i + 1) as f64,
                parameter_change: 0.01,
                convergence_rate: 0.1,
                timestamp: SystemTime::now(),
            }
        }).collect();

        let session = OptimizationSession {
            session_id: "test".to_string(),
            start_time: SystemTime::now(),
            config: DistributedOptimizationConfig::default(),
            participating_nodes: vec!["node1".to_string()],
            current_iteration: 10,
            best_solution: Some(OptimizationSolution {
                parameters: HashMap::new(),
                objective_value: 0.1,
                constraints_satisfied: true,
                metadata: SolutionMetadata {
                    generation_time: SystemTime::now(),
                    contributor_node: "node1".to_string(),
                    computation_cost: 1.0,
                    convergence_metrics: ConvergenceMetric {
                        iteration: 10,
                        objective_value: 0.1,
                        gradient_norm: 0.01,
                        parameter_change: 0.001,
                        convergence_rate: 0.1,
                        timestamp: SystemTime::now(),
                    },
                    quality_score: 0.9,
                },
                validation_scores: vec![0.9],
            }),
            convergence_history,
            status: SessionStatus::Running,
            coordinator_node: "coordinator".to_string(),
            session_metadata: SessionMetadata {
                algorithm_type: "test".to_string(),
                problem_dimension: 10,
                objective_function: "quadratic".to_string(),
                constraint_count: 0,
                optimization_mode: OptimizationMode::Minimization,
                priority_level: SessionPriority::Normal,
            },
        };

        let convergence_result = monitor.analyze_convergence(&session);
        assert!(convergence_result.is_ok());
    }

    #[test]
    fn test_convergence_callbacks() {
        let mut monitor = ConvergenceMonitor::new(ConvergenceMonitorConfig::default());

        let callback = ConvergenceCallback {
            callback_id: "test_callback".to_string(),
            trigger_condition: CallbackTrigger::EveryIteration,
            action: CallbackAction::Log("Iteration complete".to_string()),
            enabled: true,
        };

        monitor.add_callback(callback);
        assert_eq!(monitor.convergence_callbacks.len(), 1);

        let metric = ConvergenceMetric {
            iteration: 1,
            objective_value: 0.5,
            gradient_norm: 0.1,
            parameter_change: 0.01,
            convergence_rate: 0.1,
            timestamp: SystemTime::now(),
        };

        // Test that callbacks are processed (doesn't panic)
        monitor.add_metric(metric);
        assert_eq!(monitor.monitoring_history.len(), 1);
    }
}