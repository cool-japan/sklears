//! Alert evaluation engine and strategies
//!
//! This module provides the alert evaluation engine and various evaluation strategies
//! including statistical methods, machine learning models, and composite evaluations.

use super::types_config::{EvaluationQoS, PerformanceSnapshot};
use super::rule_management::AlertRule;

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};

/// Alert evaluation configuration with advanced features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertEvaluation {
    /// Evaluation interval
    pub interval: Duration,
    /// Evaluation timeout
    pub timeout: Duration,
    /// Number of consecutive failures before alerting
    pub consecutive_failures: u32,
    /// Recovery condition
    pub recovery_condition: RecoveryCondition,
    /// Evaluation strategy
    pub strategy: EvaluationStrategy,
    /// Quality of service settings
    pub qos: EvaluationQoS,
    /// Custom evaluation parameters
    pub custom_parameters: HashMap<String, String>,
}

impl Default for AlertEvaluation {
    fn default() -> Self {
        Self {
            interval: Duration::from_secs(60),
            timeout: Duration::from_secs(30),
            consecutive_failures: 1,
            recovery_condition: RecoveryCondition::ValueNormal,
            strategy: EvaluationStrategy::Simple,
            qos: EvaluationQoS::default(),
            custom_parameters: HashMap::new(),
        }
    }
}

/// Recovery conditions for alerts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecoveryCondition {
    /// Value returns to normal range
    ValueNormal,
    /// Manual resolution required
    Manual,
    /// Time-based auto-recovery
    TimeBasedRecovery(Duration),
    /// Multiple consecutive normal evaluations
    ConsecutiveNormal(u32),
    /// Custom recovery logic
    Custom(String),
    /// Percentage improvement threshold
    PercentageImprovement(f64),
    /// Trend-based recovery
    TrendBasedRecovery {
        trend_window: Duration,
        improvement_threshold: f64,
    },
}

/// Evaluation strategies for alert rules
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EvaluationStrategy {
    /// Simple threshold comparison
    Simple,
    /// Statistical analysis
    Statistical(StatisticalConfig),
    /// Machine learning based
    MachineLearning(MLConfig),
    /// Composite of multiple strategies
    Composite(Vec<EvaluationStrategy>),
    /// Time series analysis
    TimeSeries(TimeSeriesConfig),
    /// Anomaly detection
    AnomalyDetection(AnomalyConfig),
}

/// Statistical evaluation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalConfig {
    /// Statistical method to use
    pub method: StatisticalMethod,
    /// Window size for analysis
    pub window_size: Duration,
    /// Confidence level
    pub confidence_level: f64,
    /// Minimum sample size
    pub min_samples: usize,
    /// Outlier detection threshold
    pub outlier_threshold: f64,
}

impl Default for StatisticalConfig {
    fn default() -> Self {
        Self {
            method: StatisticalMethod::ZScore,
            window_size: Duration::from_secs(3600),
            confidence_level: 0.95,
            min_samples: 10,
            outlier_threshold: 2.0,
        }
    }
}

/// Statistical methods for evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StatisticalMethod {
    ZScore,
    TTest,
    ChiSquare,
    ANOVA,
    KolmogorovSmirnov,
    WilcoxonRankSum,
    MovingAverage,
    ExponentialSmoothing,
}

/// Machine learning configuration for evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MLConfig {
    /// ML model type
    pub model_type: MLModelType,
    /// Training requirements
    pub training_requirements: TrainingRequirements,
    /// Model quality thresholds
    pub quality_thresholds: QualityThresholds,
    /// Prediction parameters
    pub prediction_parameters: PredictionParameters,
    /// Ensemble configuration
    pub ensemble_config: Option<EnsembleParameters>,
}

impl Default for MLConfig {
    fn default() -> Self {
        Self {
            model_type: MLModelType::IsolationForest,
            training_requirements: TrainingRequirements::default(),
            quality_thresholds: QualityThresholds::default(),
            prediction_parameters: PredictionParameters::default(),
            ensemble_config: None,
        }
    }
}

/// Machine learning model types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MLModelType {
    IsolationForest,
    OneClassSVM,
    LocalOutlierFactor,
    LSTM,
    AutoEncoder,
    GaussianMixture,
    Prophet,
    ARIMA,
}

/// Training requirements for ML models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingRequirements {
    /// Minimum training data size
    pub min_training_samples: usize,
    /// Maximum training time
    pub max_training_time: Duration,
    /// Feature engineering requirements
    pub feature_requirements: Vec<String>,
    /// Data quality requirements
    pub data_quality_requirements: DataQualityRequirements,
}

impl Default for TrainingRequirements {
    fn default() -> Self {
        Self {
            min_training_samples: 1000,
            max_training_time: Duration::from_secs(3600),
            feature_requirements: Vec::new(),
            data_quality_requirements: DataQualityRequirements::default(),
        }
    }
}

/// Data quality requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataQualityRequirements {
    /// Maximum allowed missing data percentage
    pub max_missing_percentage: f64,
    /// Minimum data freshness
    pub min_data_freshness: Duration,
    /// Required data completeness
    pub required_completeness: f64,
}

impl Default for DataQualityRequirements {
    fn default() -> Self {
        Self {
            max_missing_percentage: 5.0,
            min_data_freshness: Duration::from_secs(3600),
            required_completeness: 0.95,
        }
    }
}

/// Quality thresholds for ML models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityThresholds {
    /// Minimum accuracy
    pub min_accuracy: f64,
    /// Maximum false positive rate
    pub max_false_positive_rate: f64,
    /// Maximum false negative rate
    pub max_false_negative_rate: f64,
    /// Minimum precision
    pub min_precision: f64,
    /// Minimum recall
    pub min_recall: f64,
    /// Minimum F1 score
    pub min_f1_score: f64,
}

impl Default for QualityThresholds {
    fn default() -> Self {
        Self {
            min_accuracy: 0.85,
            max_false_positive_rate: 0.1,
            max_false_negative_rate: 0.05,
            min_precision: 0.8,
            min_recall: 0.8,
            min_f1_score: 0.8,
        }
    }
}

/// Prediction parameters for ML models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionParameters {
    /// Prediction confidence threshold
    pub confidence_threshold: f64,
    /// Prediction horizon
    pub prediction_horizon: Duration,
    /// Update frequency for models
    pub model_update_frequency: Duration,
    /// Drift detection settings
    pub drift_detection: DriftDetectionConfig,
}

impl Default for PredictionParameters {
    fn default() -> Self {
        Self {
            confidence_threshold: 0.8,
            prediction_horizon: Duration::from_secs(300),
            model_update_frequency: Duration::from_secs(86400), // Daily
            drift_detection: DriftDetectionConfig::default(),
        }
    }
}

/// Drift detection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriftDetectionConfig {
    /// Enable drift detection
    pub enabled: bool,
    /// Drift detection method
    pub method: DriftDetectionMethod,
    /// Detection threshold
    pub threshold: f64,
    /// Minimum samples for detection
    pub min_samples: usize,
}

impl Default for DriftDetectionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            method: DriftDetectionMethod::KolmogorovSmirnov,
            threshold: 0.05,
            min_samples: 100,
        }
    }
}

/// Drift detection methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DriftDetectionMethod {
    KolmogorovSmirnov,
    ChiSquare,
    PageHinkley,
    ADWIN,
    DDM,
    EDDM,
}

/// Ensemble parameters for ML models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnsembleParameters {
    /// Number of models in ensemble
    pub ensemble_size: usize,
    /// Voting strategy
    pub voting_strategy: VotingStrategy,
    /// Diversity requirements
    pub diversity_requirements: DiversityRequirements,
}

/// Voting strategies for ensembles
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VotingStrategy {
    Majority,
    Weighted,
    Confidence,
    Unanimous,
}

/// Diversity requirements for ensemble models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiversityRequirements {
    /// Minimum diversity score
    pub min_diversity_score: f64,
    /// Diversity metrics to use
    pub diversity_metrics: Vec<DiversityMetric>,
    /// Correlation thresholds
    pub correlation_thresholds: CorrelationThresholds,
}

/// Diversity metrics for model ensembles
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DiversityMetric {
    DisagreementMeasure,
    DoubleFailureMeasure,
    CorrelationCoefficient,
    QStatistic,
    EntropyMeasure,
}

/// Correlation thresholds for diversity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrelationThresholds {
    /// Maximum correlation between models
    pub max_correlation: f64,
    /// Minimum variance in predictions
    pub min_prediction_variance: f64,
    /// Maximum similarity in errors
    pub max_error_similarity: f64,
}

/// Time series analysis configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSeriesConfig {
    /// Seasonality detection
    pub seasonality_detection: bool,
    /// Trend analysis
    pub trend_analysis: bool,
    /// Forecast horizon
    pub forecast_horizon: Duration,
    /// Model type for time series
    pub model_type: TimeSeriesModelType,
}

/// Time series model types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TimeSeriesModelType {
    ARIMA,
    Prophet,
    ExponentialSmoothing,
    LSTM,
    GRU,
    Transformer,
}

/// Anomaly detection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyConfig {
    /// Anomaly detection algorithm
    pub algorithm: AnomalyAlgorithm,
    /// Sensitivity level
    pub sensitivity: f64,
    /// Training period
    pub training_period: Duration,
    /// Update frequency
    pub update_frequency: Duration,
}

/// Anomaly detection algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnomalyAlgorithm {
    IsolationForest,
    LocalOutlierFactor,
    OneClassSVM,
    EllipticEnvelope,
    DBSCAN,
    KMeans,
    AutoEncoder,
}

/// Alert evaluation engine
#[derive(Debug)]
pub struct AlertEvaluationEngine {
    /// Worker pool for parallel evaluation
    pub worker_pool: WorkerPool,
    /// Evaluation cache
    pub evaluation_cache: Arc<Mutex<HashMap<String, CachedEvaluation>>>,
    /// Performance monitor
    pub performance_monitor: Arc<Mutex<EvaluationPerformanceMonitor>>,
    /// Engine configuration
    pub config: EvaluationEngineConfig,
}

/// Worker pool for parallel evaluation
#[derive(Debug)]
pub struct WorkerPool {
    /// Number of worker threads
    pub worker_count: usize,
    /// Task queue
    pub task_queue: Arc<Mutex<VecDeque<EvaluationTask>>>,
    /// Worker statistics
    pub worker_stats: HashMap<usize, WorkerStats>,
}

/// Individual evaluation task
#[derive(Debug, Clone)]
pub struct EvaluationTask {
    /// Task identifier
    pub task_id: String,
    /// Rule to evaluate
    pub rule: AlertRule,
    /// Task priority
    pub priority: i32,
    /// Created timestamp
    pub created_at: SystemTime,
}

/// Worker statistics
#[derive(Debug, Clone)]
pub struct WorkerStats {
    /// Tasks completed
    pub tasks_completed: u64,
    /// Average task duration
    pub average_task_duration: Duration,
    /// Last task completion time
    pub last_completion_time: SystemTime,
    /// Error count
    pub error_count: u64,
}

/// Cached evaluation result
#[derive(Debug, Clone)]
pub struct CachedEvaluation {
    /// Evaluation result
    pub result: EvaluationResult,
    /// Cache timestamp
    pub cached_at: SystemTime,
    /// Cache TTL
    pub ttl: Duration,
}

/// Evaluation result
#[derive(Debug, Clone)]
pub struct EvaluationResult {
    /// Whether the condition is met
    pub condition_met: bool,
    /// Confidence score
    pub confidence: f64,
    /// Evaluation details
    pub details: EvaluationDetails,
    /// Evaluation timestamp
    pub evaluated_at: SystemTime,
}

/// Detailed evaluation information
#[derive(Debug, Clone)]
pub struct EvaluationDetails {
    /// Metric values used in evaluation
    pub metric_values: HashMap<String, f64>,
    /// Threshold information
    pub threshold_info: ThresholdInfo,
    /// Statistical information
    pub statistical_info: Option<StatisticalInfo>,
    /// ML model information
    pub ml_info: Option<MLInfo>,
}

/// Threshold information
#[derive(Debug, Clone)]
pub struct ThresholdInfo {
    /// Current value
    pub current_value: f64,
    /// Threshold value
    pub threshold_value: f64,
    /// Operator used
    pub operator: String,
    /// Margin from threshold
    pub margin: f64,
}

/// Statistical evaluation information
#[derive(Debug, Clone)]
pub struct StatisticalInfo {
    /// Statistical test result
    pub test_result: f64,
    /// P-value
    pub p_value: f64,
    /// Test statistic
    pub test_statistic: f64,
    /// Sample size
    pub sample_size: usize,
}

/// Machine learning evaluation information
#[derive(Debug, Clone)]
pub struct MLInfo {
    /// Model type used
    pub model_type: String,
    /// Prediction score
    pub prediction_score: f64,
    /// Feature importance
    pub feature_importance: HashMap<String, f64>,
    /// Model confidence
    pub model_confidence: f64,
}

/// Configuration for evaluation engine
#[derive(Debug, Clone)]
pub struct EvaluationEngineConfig {
    /// Maximum cache size
    pub max_cache_size: usize,
    /// Default cache TTL
    pub default_cache_ttl: Duration,
    /// Maximum worker threads
    pub max_workers: usize,
    /// Task timeout
    pub task_timeout: Duration,
}

impl Default for EvaluationEngineConfig {
    fn default() -> Self {
        Self {
            max_cache_size: 1000,
            default_cache_ttl: Duration::from_secs(300),
            max_workers: 8,
            task_timeout: Duration::from_secs(30),
        }
    }
}

/// Performance monitor for evaluation engine
#[derive(Debug)]
pub struct EvaluationPerformanceMonitor {
    /// Performance metrics
    pub metrics: HashMap<String, f64>,
    /// Performance history
    pub history: VecDeque<PerformanceSnapshot>,
    /// Monitor configuration
    pub config: PerformanceMonitorConfig,
    /// Alert thresholds
    pub alert_thresholds: HashMap<String, f64>,
}

/// Performance monitor configuration
#[derive(Debug, Clone)]
pub struct PerformanceMonitorConfig {
    /// Monitoring interval
    pub monitoring_interval: Duration,
    /// History retention
    pub history_retention: Duration,
    /// Metrics to track
    pub tracked_metrics: Vec<String>,
    /// Enable alerting
    pub enable_alerting: bool,
}

impl AlertEvaluationEngine {
    /// Create a new evaluation engine
    pub fn new() -> Self {
        Self {
            worker_pool: WorkerPool {
                worker_count: 4,
                task_queue: Arc::new(Mutex::new(VecDeque::new())),
                worker_stats: HashMap::new(),
            },
            evaluation_cache: Arc::new(Mutex::new(HashMap::new())),
            performance_monitor: Arc::new(Mutex::new(EvaluationPerformanceMonitor {
                metrics: HashMap::new(),
                history: VecDeque::new(),
                config: PerformanceMonitorConfig {
                    monitoring_interval: Duration::from_secs(60),
                    history_retention: Duration::from_secs(3600),
                    tracked_metrics: vec!["evaluation_time".to_string(), "cache_hit_rate".to_string()],
                    enable_alerting: true,
                },
                alert_thresholds: HashMap::new(),
            })),
            config: EvaluationEngineConfig::default(),
        }
    }

    /// Evaluate an alert rule
    pub fn evaluate_rule(&self, rule: &AlertRule) -> Result<EvaluationResult, String> {
        // Check cache first
        if let Ok(cache) = self.evaluation_cache.lock() {
            if let Some(cached) = cache.get(&rule.rule_id) {
                if SystemTime::now().duration_since(cached.cached_at).unwrap_or(Duration::MAX) < cached.ttl {
                    return Ok(cached.result.clone());
                }
            }
        }

        // Perform evaluation
        let result = self.perform_evaluation(rule)?;

        // Cache result
        if let Ok(mut cache) = self.evaluation_cache.lock() {
            cache.insert(rule.rule_id.clone(), CachedEvaluation {
                result: result.clone(),
                cached_at: SystemTime::now(),
                ttl: Duration::from_secs(300),
            });

            // Limit cache size
            if cache.len() > self.config.max_cache_size {
                // Remove oldest entries (simplified)
                let keys: Vec<String> = cache.keys().cloned().collect();
                if let Some(key) = keys.first() {
                    cache.remove(key);
                }
            }
        }

        Ok(result)
    }

    /// Perform the actual evaluation
    fn perform_evaluation(&self, rule: &AlertRule) -> Result<EvaluationResult, String> {
        match &rule.evaluation.strategy {
            EvaluationStrategy::Simple => {
                self.evaluate_simple_threshold(rule)
            }
            EvaluationStrategy::Statistical(config) => {
                self.evaluate_statistical(rule, config)
            }
            EvaluationStrategy::MachineLearning(config) => {
                self.evaluate_ml(rule, config)
            }
            EvaluationStrategy::Composite(strategies) => {
                self.evaluate_composite(rule, strategies)
            }
            EvaluationStrategy::TimeSeries(_) => {
                Ok(EvaluationResult {
                    condition_met: false,
                    confidence: 0.5,
                    details: EvaluationDetails {
                        metric_values: HashMap::new(),
                        threshold_info: ThresholdInfo {
                            current_value: 0.0,
                            threshold_value: 0.0,
                            operator: "time_series".to_string(),
                            margin: 0.0,
                        },
                        statistical_info: None,
                        ml_info: None,
                    },
                    evaluated_at: SystemTime::now(),
                })
            }
            EvaluationStrategy::AnomalyDetection(_) => {
                Ok(EvaluationResult {
                    condition_met: false,
                    confidence: 0.5,
                    details: EvaluationDetails {
                        metric_values: HashMap::new(),
                        threshold_info: ThresholdInfo {
                            current_value: 0.0,
                            threshold_value: 0.0,
                            operator: "anomaly".to_string(),
                            margin: 0.0,
                        },
                        statistical_info: None,
                        ml_info: None,
                    },
                    evaluated_at: SystemTime::now(),
                })
            }
        }
    }

    /// Evaluate simple threshold
    fn evaluate_simple_threshold(&self, _rule: &AlertRule) -> Result<EvaluationResult, String> {
        // Simplified implementation
        Ok(EvaluationResult {
            condition_met: false,
            confidence: 1.0,
            details: EvaluationDetails {
                metric_values: HashMap::new(),
                threshold_info: ThresholdInfo {
                    current_value: 50.0,
                    threshold_value: 80.0,
                    operator: "greater_than".to_string(),
                    margin: -30.0,
                },
                statistical_info: None,
                ml_info: None,
            },
            evaluated_at: SystemTime::now(),
        })
    }

    /// Evaluate using statistical methods
    fn evaluate_statistical(&self, _rule: &AlertRule, _config: &StatisticalConfig) -> Result<EvaluationResult, String> {
        // Simplified implementation
        Ok(EvaluationResult {
            condition_met: false,
            confidence: 0.95,
            details: EvaluationDetails {
                metric_values: HashMap::new(),
                threshold_info: ThresholdInfo {
                    current_value: 0.0,
                    threshold_value: 0.0,
                    operator: "statistical".to_string(),
                    margin: 0.0,
                },
                statistical_info: Some(StatisticalInfo {
                    test_result: 1.96,
                    p_value: 0.05,
                    test_statistic: 2.1,
                    sample_size: 100,
                }),
                ml_info: None,
            },
            evaluated_at: SystemTime::now(),
        })
    }

    /// Evaluate using machine learning
    fn evaluate_ml(&self, _rule: &AlertRule, _config: &MLConfig) -> Result<EvaluationResult, String> {
        // Simplified implementation
        Ok(EvaluationResult {
            condition_met: false,
            confidence: 0.87,
            details: EvaluationDetails {
                metric_values: HashMap::new(),
                threshold_info: ThresholdInfo {
                    current_value: 0.0,
                    threshold_value: 0.0,
                    operator: "ml_prediction".to_string(),
                    margin: 0.0,
                },
                statistical_info: None,
                ml_info: Some(MLInfo {
                    model_type: "isolation_forest".to_string(),
                    prediction_score: 0.2,
                    feature_importance: HashMap::new(),
                    model_confidence: 0.87,
                }),
            },
            evaluated_at: SystemTime::now(),
        })
    }

    /// Evaluate composite strategy
    fn evaluate_composite(&self, _rule: &AlertRule, _strategies: &[EvaluationStrategy]) -> Result<EvaluationResult, String> {
        // Simplified implementation - would combine results from all strategies
        Ok(EvaluationResult {
            condition_met: false,
            confidence: 0.75,
            details: EvaluationDetails {
                metric_values: HashMap::new(),
                threshold_info: ThresholdInfo {
                    current_value: 0.0,
                    threshold_value: 0.0,
                    operator: "composite".to_string(),
                    margin: 0.0,
                },
                statistical_info: None,
                ml_info: None,
            },
            evaluated_at: SystemTime::now(),
        })
    }
}

impl Default for AlertEvaluationEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_evaluation_engine_creation() {
        let engine = AlertEvaluationEngine::new();
        assert_eq!(engine.worker_pool.worker_count, 4);
    }

    #[test]
    fn test_alert_evaluation_default() {
        let eval = AlertEvaluation::default();
        assert_eq!(eval.interval, Duration::from_secs(60));
        assert_eq!(eval.consecutive_failures, 1);
    }

    #[test]
    fn test_statistical_config_default() {
        let config = StatisticalConfig::default();
        assert_eq!(config.confidence_level, 0.95);
        assert!(matches!(config.method, StatisticalMethod::ZScore));
    }

    #[test]
    fn test_ml_config_default() {
        let config = MLConfig::default();
        assert!(matches!(config.model_type, MLModelType::IsolationForest));
        assert_eq!(config.quality_thresholds.min_accuracy, 0.85);
    }

    #[test]
    fn test_evaluation_strategies() {
        let strategies = vec![
            EvaluationStrategy::Simple,
            EvaluationStrategy::Statistical(StatisticalConfig::default()),
            EvaluationStrategy::MachineLearning(MLConfig::default()),
        ];

        for strategy in strategies {
            match strategy {
                EvaluationStrategy::Simple => (),
                EvaluationStrategy::Statistical(_) => (),
                EvaluationStrategy::MachineLearning(_) => (),
                _ => (),
            }
        }
    }

    #[test]
    fn test_evaluation_result() {
        let result = EvaluationResult {
            condition_met: true,
            confidence: 0.95,
            details: EvaluationDetails {
                metric_values: HashMap::new(),
                threshold_info: ThresholdInfo {
                    current_value: 85.0,
                    threshold_value: 80.0,
                    operator: "greater_than".to_string(),
                    margin: 5.0,
                },
                statistical_info: None,
                ml_info: None,
            },
            evaluated_at: SystemTime::now(),
        };

        assert!(result.condition_met);
        assert_eq!(result.confidence, 0.95);
        assert_eq!(result.details.threshold_info.margin, 5.0);
    }
}