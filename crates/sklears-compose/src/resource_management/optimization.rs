//! Resource optimization and prediction engines
//!
//! This module provides intelligent resource optimization algorithms and
//! predictive scaling capabilities for the resource management system.

use super::resource_types::{
    AllocatedResources, MemoryUsage, ResourceAllocation, ResourcePoolType, ResourceUsage,
};
use super::simd_operations;
use crate::task_definitions::TaskRequirements;
use sklears_core::error::{Result as SklResult, SklearsError};
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, SystemTime};

/// Resource optimizer for intelligent resource allocation and rebalancing
#[derive(Debug)]
pub struct ResourceOptimizer {
    /// Optimization configuration
    config: OptimizerConfig,
    /// Optimization strategies
    strategies: Vec<Box<dyn OptimizationStrategy>>,
    /// Optimization state
    state: OptimizerState,
    /// Performance history
    history: OptimizationHistory,
}

/// Optimizer configuration
#[derive(Debug, Clone)]
pub struct OptimizerConfig {
    /// Optimization strategy
    pub strategy: OptimizationStrategyType,
    /// Rebalancing interval
    pub rebalancing_interval: Duration,
    /// Enable predictive scaling
    pub enable_predictive_scaling: bool,
    /// Enable energy optimization
    pub enable_energy_optimization: bool,
    /// Enable thermal management
    pub enable_thermal_management: bool,
    /// Optimization aggressiveness (0.0 to 1.0)
    pub aggressiveness: f64,
    /// Performance weight in optimization
    pub performance_weight: f64,
    /// Energy weight in optimization
    pub energy_weight: f64,
    /// Fairness weight in optimization
    pub fairness_weight: f64,
}

/// Types of optimization strategies
#[derive(Debug, Clone)]
pub enum OptimizationStrategyType {
    /// Maximize resource utilization
    MaxUtilization,
    /// Minimize energy consumption
    MinEnergy,
    /// Balance performance and energy
    Balanced,
    /// Maximize throughput
    MaxThroughput,
    /// Minimize latency
    MinLatency,
    /// Fair resource sharing
    FairShare,
    /// Custom optimization
    Custom(String),
}

/// Optimization strategy trait
pub trait OptimizationStrategy: Send + Sync + std::fmt::Debug {
    /// Get strategy name
    fn name(&self) -> &str;

    /// Optimize resource allocations
    fn optimize(
        &self,
        current_allocations: &[ResourceAllocation],
        available_resources: &ResourceUsage,
        pending_requests: &[TaskRequirements],
    ) -> SklResult<OptimizationResult>;

    /// Get optimization score for current state
    fn score(&self, allocations: &[ResourceAllocation], usage: &ResourceUsage) -> f64;
}

/// Result of optimization process
#[derive(Debug, Clone)]
pub struct OptimizationResult {
    /// Recommended reallocation actions
    pub actions: Vec<OptimizationAction>,
    /// Expected improvement
    pub expected_improvement: f64,
    /// Confidence in the optimization
    pub confidence: f64,
    /// Optimization metadata
    pub metadata: HashMap<String, String>,
}

/// Optimization actions to take
#[derive(Debug, Clone)]
pub enum OptimizationAction {
    /// Reallocate resources for a task
    Reallocate {
        task_id: String,
        old_allocation: ResourceAllocation,
        new_allocation: ResourceAllocation,
    },
    /// Migrate task to different resources
    Migrate {
        task_id: String,
        from_resources: AllocatedResources,
        to_resources: AllocatedResources,
    },
    /// Scale resources up
    ScaleUp {
        resource_type: ResourcePoolType,
        amount: u64,
    },
    /// Scale resources down
    ScaleDown {
        resource_type: ResourcePoolType,
        amount: u64,
    },
    /// Consolidate resources
    Consolidate {
        task_ids: Vec<String>,
        consolidated_allocation: ResourceAllocation,
    },
    /// Adjust resource frequencies
    AdjustFrequency { resource_id: String, frequency: f64 },
}

/// Optimizer state tracking
#[derive(Debug, Clone)]
pub struct OptimizerState {
    /// Is optimizer running
    pub running: bool,
    /// Last optimization time
    pub last_optimization: SystemTime,
    /// Optimization cycle count
    pub cycle_count: u64,
    /// Current optimization score
    pub current_score: f64,
    /// Best achieved score
    pub best_score: f64,
    /// Optimizations performed
    pub optimizations_performed: u64,
    /// Failed optimizations
    pub failed_optimizations: u64,
}

/// Optimization history and analytics
#[derive(Debug)]
pub struct OptimizationHistory {
    /// Optimization events
    events: VecDeque<OptimizationEvent>,
    /// Performance trends
    performance_trends: VecDeque<PerformanceMeasurement>,
    /// Energy usage trends
    energy_trends: VecDeque<EnergyMeasurement>,
    /// Configuration
    config: HistoryConfig,
}

/// Optimization event record
#[derive(Debug, Clone)]
pub struct OptimizationEvent {
    /// Event timestamp
    pub timestamp: SystemTime,
    /// Event type
    pub event_type: OptimizationEventType,
    /// Actions taken
    pub actions: Vec<OptimizationAction>,
    /// Measured impact
    pub impact: OptimizationImpact,
    /// Duration of optimization
    pub duration: Duration,
}

/// Types of optimization events
#[derive(Debug, Clone)]
pub enum OptimizationEventType {
    /// ScheduledOptimization
    ScheduledOptimization,
    /// ReactiveOptimization
    ReactiveOptimization,
    /// PredictiveOptimization
    PredictiveOptimization,
    /// EmergencyOptimization
    EmergencyOptimization,
    /// UserTriggered
    UserTriggered,
}

/// Impact of optimization
#[derive(Debug, Clone)]
pub struct OptimizationImpact {
    /// Performance change
    pub performance_delta: f64,
    /// Energy consumption change
    pub energy_delta: f64,
    /// Resource utilization change
    pub utilization_delta: f64,
    /// Cost change
    pub cost_delta: f64,
}

/// Performance measurement
#[derive(Debug, Clone)]
pub struct PerformanceMeasurement {
    /// Timestamp
    pub timestamp: SystemTime,
    /// Overall system performance score
    pub performance_score: f64,
    /// Throughput (tasks/sec)
    pub throughput: f64,
    /// Average latency
    pub avg_latency: Duration,
    /// 95th percentile latency
    pub p95_latency: Duration,
    /// Resource efficiency
    pub resource_efficiency: f64,
}

/// Energy consumption measurement
#[derive(Debug, Clone)]
pub struct EnergyMeasurement {
    /// Timestamp
    pub timestamp: SystemTime,
    /// Total power consumption (watts)
    pub total_power: f64,
    /// CPU power consumption
    pub cpu_power: f64,
    /// GPU power consumption
    pub gpu_power: f64,
    /// Memory power consumption
    pub memory_power: f64,
    /// Network power consumption
    pub network_power: f64,
    /// Storage power consumption
    pub storage_power: f64,
    /// Power efficiency (performance/watt)
    pub power_efficiency: f64,
}

/// History configuration
#[derive(Debug, Clone)]
pub struct HistoryConfig {
    /// Maximum history size
    pub max_history_size: usize,
    /// Measurement interval
    pub measurement_interval: Duration,
    /// Enable detailed tracking
    pub detailed_tracking: bool,
}

/// Resource prediction engine for predictive scaling
#[derive(Debug)]
pub struct ResourcePredictionEngine {
    /// Prediction models
    models: HashMap<String, Box<dyn PredictionModel>>,
    /// Prediction configuration
    config: PredictionConfig,
    /// Historical data
    historical_data: PredictionHistory,
    /// Current predictions
    current_predictions: HashMap<String, ResourcePrediction>,
}

/// Prediction configuration
#[derive(Debug, Clone)]
pub struct PredictionConfig {
    /// Prediction horizon
    pub prediction_horizon: Duration,
    /// Model update interval
    pub model_update_interval: Duration,
    /// Prediction confidence threshold
    pub confidence_threshold: f64,
    /// Enable ensemble predictions
    pub enable_ensemble: bool,
    /// Historical data window
    pub historical_window: Duration,
}

/// Prediction model trait
pub trait PredictionModel: Send + Sync + std::fmt::Debug {
    /// Model name
    fn name(&self) -> &str;

    /// Train the model with historical data
    fn train(&mut self, data: &PredictionHistory) -> SklResult<()>;

    /// Make a prediction
    fn predict(&self, context: &PredictionContext) -> SklResult<ResourcePrediction>;

    /// Get model accuracy
    fn accuracy(&self) -> f64;

    /// Update model with new data
    fn update(&mut self, actual: &ResourceUsage, predicted: &ResourcePrediction) -> SklResult<()>;
}

/// Prediction context
#[derive(Debug, Clone)]
pub struct PredictionContext {
    /// Current resource usage
    pub current_usage: ResourceUsage,
    /// Time of day
    pub time_of_day: u32, // seconds since midnight
    /// Day of week (0-6)
    pub day_of_week: u8,
    /// Current workload
    pub workload_characteristics: WorkloadCharacteristics,
    /// External factors
    pub external_factors: HashMap<String, f64>,
}

/// Workload characteristics
#[derive(Debug, Clone)]
pub struct WorkloadCharacteristics {
    /// Number of active tasks
    pub active_tasks: u32,
    /// Average task complexity
    pub avg_complexity: f64,
    /// Workload type distribution
    pub workload_types: HashMap<String, f64>,
    /// Resource requirements pattern
    pub requirement_pattern: ResourceRequirementPattern,
}

/// Pattern of resource requirements
#[derive(Debug, Clone)]
pub struct ResourceRequirementPattern {
    /// CPU intensity
    pub cpu_intensity: f64,
    /// Memory intensity
    pub memory_intensity: f64,
    /// GPU intensity
    pub gpu_intensity: f64,
    /// Network intensity
    pub network_intensity: f64,
    /// Storage intensity
    pub storage_intensity: f64,
    /// Temporal pattern
    pub temporal_pattern: TemporalPattern,
}

/// Temporal usage patterns
#[derive(Debug, Clone)]
pub enum TemporalPattern {
    /// Constant
    Constant,
    /// Periodic
    Periodic { period: Duration },
    /// Trending
    Trending { trend: f64 },
    /// Burst
    Burst { burst_duration: Duration },
    /// Random
    Random,
}

/// Resource usage prediction
#[derive(Debug, Clone)]
pub struct ResourcePrediction {
    /// Predicted resource usage
    pub predicted_usage: ResourceUsage,
    /// Prediction confidence (0.0 to 1.0)
    pub confidence: f64,
    /// Prediction horizon
    pub horizon: Duration,
    /// Prediction timestamp
    pub timestamp: SystemTime,
    /// Model used for prediction
    pub model_name: String,
    /// Prediction intervals
    pub intervals: PredictionIntervals,
}

/// Prediction confidence intervals
#[derive(Debug, Clone)]
pub struct PredictionIntervals {
    /// Lower bound (5th percentile)
    pub lower: ResourceUsage,
    /// Upper bound (95th percentile)
    pub upper: ResourceUsage,
    /// Standard deviation
    pub std_dev: ResourceUsageVariance,
}

/// Resource usage variance
#[derive(Debug, Clone)]
pub struct ResourceUsageVariance {
    /// CPU usage variance
    pub cpu_variance: f64,
    /// Memory usage variance
    pub memory_variance: f64,
    /// GPU usage variance
    pub gpu_variance: Vec<f64>,
    /// Network usage variance
    pub network_variance: f64,
    /// Storage usage variance
    pub storage_variance: f64,
}

/// Historical data for predictions
#[derive(Debug)]
pub struct PredictionHistory {
    /// Usage samples
    usage_samples: VecDeque<UsageSample>,
    /// Workload samples
    workload_samples: VecDeque<WorkloadSample>,
    /// Performance samples
    performance_samples: VecDeque<PerformanceSample>,
}

/// Usage sample with context
#[derive(Debug, Clone)]
pub struct UsageSample {
    /// Timestamp
    pub timestamp: SystemTime,
    /// Resource usage
    pub usage: ResourceUsage,
    /// Context
    pub context: PredictionContext,
}

/// Workload sample
#[derive(Debug, Clone)]
pub struct WorkloadSample {
    /// Timestamp
    pub timestamp: SystemTime,
    /// Workload characteristics
    pub characteristics: WorkloadCharacteristics,
    /// Task queue length
    pub queue_length: u32,
}

/// Performance sample
#[derive(Debug, Clone)]
pub struct PerformanceSample {
    /// Timestamp
    pub timestamp: SystemTime,
    /// Performance metrics
    pub metrics: PerformanceMeasurement,
}

impl Default for ResourceOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

impl ResourceOptimizer {
    /// Create a new resource optimizer
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: OptimizerConfig::default(),
            strategies: Vec::new(),
            state: OptimizerState {
                running: false,
                last_optimization: SystemTime::now(),
                cycle_count: 0,
                current_score: 0.0,
                best_score: 0.0,
                optimizations_performed: 0,
                failed_optimizations: 0,
            },
            history: OptimizationHistory {
                events: VecDeque::new(),
                performance_trends: VecDeque::new(),
                energy_trends: VecDeque::new(),
                config: HistoryConfig {
                    max_history_size: 10000,
                    measurement_interval: Duration::from_secs(60),
                    detailed_tracking: true,
                },
            },
        }
    }

    /// Add an optimization strategy
    pub fn add_strategy(&mut self, strategy: Box<dyn OptimizationStrategy>) {
        self.strategies.push(strategy);
    }

    /// Start optimization loop
    pub fn start_optimization(&mut self) -> SklResult<()> {
        self.state.running = true;
        Ok(())
    }

    /// Stop optimization loop
    pub fn stop_optimization(&mut self) -> SklResult<()> {
        self.state.running = false;
        Ok(())
    }

    /// Perform optimization cycle
    pub fn optimize(
        &mut self,
        current_allocations: &[ResourceAllocation],
        available_resources: &ResourceUsage,
        pending_requests: &[TaskRequirements],
    ) -> SklResult<OptimizationResult> {
        let mut best_result = OptimizationResult {
            actions: Vec::new(),
            expected_improvement: 0.0,
            confidence: 0.0,
            metadata: HashMap::new(),
        };

        // Run all optimization strategies and select the best one
        for strategy in &self.strategies {
            let result =
                strategy.optimize(current_allocations, available_resources, pending_requests)?;
            if result.expected_improvement > best_result.expected_improvement {
                best_result = result;
            }
        }

        // Update state
        self.state.cycle_count += 1;
        self.state.last_optimization = SystemTime::now();

        Ok(best_result)
    }

    /// Calculate optimization score using SIMD acceleration
    #[must_use]
    pub fn calculate_optimization_score(
        &self,
        allocations: &[ResourceAllocation],
        usage: &ResourceUsage,
    ) -> f64 {
        // Extract utilization metrics for SIMD processing
        let cpu_utils = vec![usage.cpu_percent];
        let avg_cpu = simd_operations::simd_average_utilization(&cpu_utils);

        // Collect memory utilization
        let memory_util = usage.memory_usage.used as f64 / usage.memory_usage.total as f64 * 100.0;
        let memory_utils = [memory_util];

        // Collect GPU utilizations
        let gpu_utils: Vec<f64> = usage
            .gpu_usage
            .iter()
            .map(|gpu| gpu.utilization_percent)
            .collect();

        let avg_gpu = if gpu_utils.is_empty() {
            0.0
        } else {
            simd_operations::simd_average_utilization(&gpu_utils)
        };

        // Calculate efficiency score using SIMD
        let utilizations = vec![avg_cpu, memory_util, avg_gpu];
        let weights = vec![
            self.config.performance_weight,
            self.config.performance_weight * 0.8,
            self.config.performance_weight * 1.2,
        ];

        let efficiency = simd_operations::simd_efficiency_score(&utilizations, &weights);

        // Calculate load balance score
        let balance_score = simd_operations::simd_load_balance_score(&utilizations);

        // Combine scores with configuration weights
        efficiency * 0.7 + balance_score * 0.3
    }
}

impl Default for ResourcePredictionEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl ResourcePredictionEngine {
    /// Create a new resource prediction engine
    #[must_use]
    pub fn new() -> Self {
        Self {
            models: HashMap::new(),
            config: PredictionConfig {
                prediction_horizon: Duration::from_secs(60 * 60), // 1 hour
                model_update_interval: Duration::from_secs(5 * 60), // 5 minutes
                confidence_threshold: 0.7,
                enable_ensemble: true,
                historical_window: Duration::from_secs(7 * 24 * 60 * 60), // 7 days
            },
            historical_data: PredictionHistory {
                usage_samples: VecDeque::new(),
                workload_samples: VecDeque::new(),
                performance_samples: VecDeque::new(),
            },
            current_predictions: HashMap::new(),
        }
    }

    /// Add a prediction model
    pub fn add_model(&mut self, name: String, model: Box<dyn PredictionModel>) {
        self.models.insert(name, model);
    }

    /// Make resource usage predictions
    pub fn predict(&mut self, context: &PredictionContext) -> SklResult<ResourcePrediction> {
        let mut predictions = Vec::new();

        // Get predictions from all models
        for (name, model) in &self.models {
            if let Ok(prediction) = model.predict(context) {
                predictions.push((name.clone(), prediction));
            }
        }

        if predictions.is_empty() {
            return Err(SklearsError::ResourceAllocationError(
                "No models available for prediction".to_string(),
            ));
        }

        // If ensemble is enabled, combine predictions
        if self.config.enable_ensemble && predictions.len() > 1 {
            self.ensemble_predict(&predictions)
        } else {
            // Use the best model's prediction
            let best_prediction = predictions
                .into_iter()
                .max_by(|a, b| {
                    a.1.confidence
                        .partial_cmp(&b.1.confidence)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap();

            Ok(best_prediction.1)
        }
    }

    /// Combine predictions using ensemble method
    fn ensemble_predict(
        &self,
        predictions: &[(String, ResourcePrediction)],
    ) -> SklResult<ResourcePrediction> {
        // Use weighted average based on confidence
        let total_confidence: f64 = predictions.iter().map(|(_, p)| p.confidence).sum();

        if total_confidence == 0.0 {
            return Err(SklearsError::ResourceAllocationError(
                "All predictions have zero confidence".to_string(),
            ));
        }

        // Calculate weighted average CPU usage
        let weighted_cpu: f64 = predictions
            .iter()
            .map(|(_, p)| p.predicted_usage.cpu_percent * p.confidence)
            .sum::<f64>()
            / total_confidence;

        // Calculate weighted average memory usage
        let weighted_memory_used: f64 = predictions
            .iter()
            .map(|(_, p)| p.predicted_usage.memory_usage.used as f64 * p.confidence)
            .sum::<f64>()
            / total_confidence;

        let reference_prediction = &predictions[0].1;

        Ok(ResourcePrediction {
            predicted_usage: ResourceUsage {
                cpu_percent: weighted_cpu,
                memory_usage: MemoryUsage {
                    total: reference_prediction.predicted_usage.memory_usage.total,
                    used: weighted_memory_used as u64,
                    free: reference_prediction.predicted_usage.memory_usage.total
                        - weighted_memory_used as u64,
                    cached: reference_prediction.predicted_usage.memory_usage.cached,
                    swap_used: reference_prediction.predicted_usage.memory_usage.swap_used,
                },
                gpu_usage: reference_prediction.predicted_usage.gpu_usage.clone(),
                network_usage: reference_prediction.predicted_usage.network_usage.clone(),
                storage_usage: reference_prediction.predicted_usage.storage_usage.clone(),
            },
            confidence: total_confidence / predictions.len() as f64,
            horizon: reference_prediction.horizon,
            timestamp: SystemTime::now(),
            model_name: "Ensemble".to_string(),
            intervals: reference_prediction.intervals.clone(),
        })
    }

    /// Update models with actual usage data
    pub fn update_models(
        &mut self,
        actual: &ResourceUsage,
        predictions: &[ResourcePrediction],
    ) -> SklResult<()> {
        for prediction in predictions {
            if let Some(model) = self.models.get_mut(&prediction.model_name) {
                model.update(actual, prediction)?;
            }
        }
        Ok(())
    }
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self {
            strategy: OptimizationStrategyType::Balanced,
            rebalancing_interval: Duration::from_secs(30),
            enable_predictive_scaling: true,
            enable_energy_optimization: true,
            enable_thermal_management: true,
            aggressiveness: 0.5,
            performance_weight: 0.4,
            energy_weight: 0.3,
            fairness_weight: 0.3,
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimizer_creation() {
        let optimizer = ResourceOptimizer::new();
        assert!(!optimizer.state.running);
        assert_eq!(optimizer.state.cycle_count, 0);
    }

    #[test]
    fn test_prediction_engine_creation() {
        let engine = ResourcePredictionEngine::new();
        assert!(engine.models.is_empty());
        assert_eq!(
            engine.config.prediction_horizon,
            Duration::from_secs(60 * 60)
        ); // 1 hour
    }
}
