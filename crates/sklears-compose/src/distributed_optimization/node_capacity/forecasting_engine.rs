use super::capacity_history::ConfidenceInterval;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, SystemTime};

/// Capacity forecasting engine
pub struct CapacityForecastingEngine {
    pub forecasting_models: HashMap<String, ForecastingModel>,
    pub forecast_schedules: Vec<ForecastSchedule>,
    pub forecast_results: HashMap<String, ForecastResult>,
    pub model_performance: ModelPerformanceTracker,
    pub forecasting_config: ForecastingConfig,
}

/// Forecasting model
pub struct ForecastingModel {
    pub model_id: String,
    pub model_type: ForecastingModelType,
    pub target_metrics: Vec<String>,
    pub training_data: TrainingDataset,
    pub model_parameters: HashMap<String, f64>,
    pub model_state: ModelState,
    pub last_trained: SystemTime,
}

/// Forecasting model types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ForecastingModelType {
    ARIMA,
    Prophet,
    LSTM,
    GRU,
    Transformer,
    LinearRegression,
    ExponentialSmoothing,
    Custom(String),
}

/// Training dataset
pub struct TrainingDataset {
    pub dataset_id: String,
    pub features: Vec<String>,
    pub target_variable: String,
    pub data_points: usize,
    pub time_range: (SystemTime, SystemTime),
    pub data_quality_score: f64,
}

/// Model state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelState {
    Untrained,
    Training,
    Trained,
    Predicting,
    Error(String),
    Deprecated,
}

/// Forecast schedule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForecastSchedule {
    pub schedule_id: String,
    pub model_id: String,
    pub forecast_frequency: Duration,
    pub forecast_horizon: Duration,
    pub next_forecast_time: SystemTime,
    pub enabled: bool,
}

/// Forecast result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForecastResult {
    pub forecast_id: String,
    pub model_id: String,
    pub forecast_time: SystemTime,
    pub forecast_horizon: Duration,
    pub predictions: Vec<PredictionPoint>,
    pub accuracy_metrics: ForecastAccuracy,
    pub confidence_score: f64,
}

/// Prediction point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionPoint {
    pub timestamp: SystemTime,
    pub predicted_value: f64,
    pub confidence_interval: ConfidenceInterval,
    pub prediction_context: HashMap<String, String>,
}

/// Forecast accuracy metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForecastAccuracy {
    pub mae: f64,
    pub mse: f64,
    pub rmse: f64,
    pub mape: f64,
    pub smape: f64,
    pub directional_accuracy: f64,
}

/// Model performance tracker
pub struct ModelPerformanceTracker {
    pub performance_history: HashMap<String, Vec<PerformanceMetric>>,
    pub model_rankings: Vec<ModelRanking>,
    pub performance_thresholds: HashMap<String, f64>,
    pub auto_retraining: AutoRetrainingConfig,
}

/// Performance metric
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetric {
    pub metric_name: String,
    pub metric_value: f64,
    pub measurement_time: SystemTime,
    pub data_context: HashMap<String, String>,
}

/// Model ranking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRanking {
    pub model_id: String,
    pub overall_score: f64,
    pub accuracy_score: f64,
    pub stability_score: f64,
    pub efficiency_score: f64,
    pub ranking_time: SystemTime,
}

/// Auto-retraining configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoRetrainingConfig {
    pub enable_auto_retraining: bool,
    pub performance_threshold: f64,
    pub retraining_frequency: Duration,
    pub data_drift_threshold: f64,
    pub concept_drift_threshold: f64,
}

/// Forecasting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForecastingConfig {
    pub default_forecast_horizon: Duration,
    pub min_training_data_points: usize,
    pub max_models_per_metric: usize,
    pub model_selection_strategy: ModelSelectionStrategy,
    pub ensemble_config: Option<EnsembleConfig>,
}

/// Model selection strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelSelectionStrategy {
    BestPerformance,
    Ensemble,
    Diversity,
    Stability,
    Custom(String),
}

/// Ensemble configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnsembleConfig {
    pub ensemble_method: EnsembleMethod,
    pub max_models: usize,
    pub model_weights: HashMap<String, f64>,
    pub combination_strategy: CombinationStrategy,
}

/// Ensemble methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EnsembleMethod {
    Averaging,
    WeightedAveraging,
    Voting,
    Stacking,
    Blending,
    Custom(String),
}

/// Combination strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CombinationStrategy {
    Simple,
    Weighted,
    Dynamic,
    MetaLearning,
    Custom(String),
}

impl CapacityForecastingEngine {
    pub fn new() -> Self {
        Self {
            forecasting_models: HashMap::new(),
            forecast_schedules: Vec::new(),
            forecast_results: HashMap::new(),
            model_performance: ModelPerformanceTracker::new(),
            forecasting_config: ForecastingConfig::default(),
        }
    }
}

impl ModelPerformanceTracker {
    pub fn new() -> Self {
        Self {
            performance_history: HashMap::new(),
            model_rankings: Vec::new(),
            performance_thresholds: HashMap::new(),
            auto_retraining: AutoRetrainingConfig::default(),
        }
    }
}

impl Default for ForecastingConfig {
    fn default() -> Self {
        Self {
            default_forecast_horizon: Duration::from_secs(86400 * 30), // 30 days
            min_training_data_points: 100,
            max_models_per_metric: 3,
            model_selection_strategy: ModelSelectionStrategy::BestPerformance,
            ensemble_config: None,
        }
    }
}

impl Default for AutoRetrainingConfig {
    fn default() -> Self {
        Self {
            enable_auto_retraining: true,
            performance_threshold: 0.8,
            retraining_frequency: Duration::from_secs(86400 * 7), // 7 days
            data_drift_threshold: 0.2,
            concept_drift_threshold: 0.3,
        }
    }
}