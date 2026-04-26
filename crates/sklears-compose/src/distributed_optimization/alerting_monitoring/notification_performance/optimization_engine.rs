//! Optimization Engine Module
//!
//! This module provides comprehensive performance optimization capabilities including
//! optimization strategies, machine learning models, trend analysis, and adaptive
//! optimization for notification channel performance enhancement.

use std::collections::{HashMap, VecDeque};
use std::time::{Duration, SystemTime};

/// Performance optimizer for advanced optimization
#[derive(Debug, Clone)]
pub struct PerformanceOptimizer {
    /// Optimization strategies
    pub strategies: Vec<OptimizationStrategy>,
    /// Optimization configuration
    pub config: OptimizerConfig,
    /// Performance baselines
    pub baselines: PerformanceBaselines,
    /// Optimization history
    pub history: OptimizationHistory,
    /// Machine learning models
    pub ml_models: MachineLearningModels,
}

/// Optimization strategy
#[derive(Debug, Clone)]
pub struct OptimizationStrategy {
    /// Strategy identifier
    pub strategy_id: String,
    /// Strategy name
    pub name: String,
    /// Strategy type
    pub strategy_type: OptimizationStrategyType,
    /// Strategy parameters
    pub parameters: HashMap<String, f64>,
    /// Strategy effectiveness
    pub effectiveness: f64,
    /// Application conditions
    pub conditions: Vec<OptimizationCondition>,
}

/// Optimization strategy types
#[derive(Debug, Clone)]
pub enum OptimizationStrategyType {
    ConnectionPooling,
    Caching,
    Compression,
    LoadBalancing,
    BatchProcessing,
    Prefetching,
    Adaptive,
    Custom(String),
}

/// Optimization condition
#[derive(Debug, Clone)]
pub struct OptimizationCondition {
    /// Condition type
    pub condition_type: ConditionType,
    /// Condition value
    pub value: f64,
    /// Condition operator
    pub operator: ConditionOperator,
}

/// Condition types
#[derive(Debug, Clone)]
pub enum ConditionType {
    Latency,
    Throughput,
    ErrorRate,
    ResourceUsage,
    LoadLevel,
    Custom(String),
}

/// Condition operators
#[derive(Debug, Clone)]
pub enum ConditionOperator {
    GreaterThan,
    LessThan,
    Equal,
    GreaterThanOrEqual,
    LessThanOrEqual,
    NotEqual,
}

/// Optimizer configuration
#[derive(Debug, Clone)]
pub struct OptimizerConfig {
    /// Enable optimization
    pub enabled: bool,
    /// Optimization interval
    pub optimization_interval: Duration,
    /// Enable machine learning
    pub enable_ml: bool,
    /// Performance target
    pub performance_target: PerformanceTarget,
    /// Optimization aggressiveness
    pub aggressiveness: OptimizationAggressiveness,
}

/// Performance target
#[derive(Debug, Clone)]
pub struct PerformanceTarget {
    /// Target latency
    pub target_latency: Duration,
    /// Target throughput
    pub target_throughput: f64,
    /// Target error rate
    pub target_error_rate: f64,
    /// Target resource usage
    pub target_resource_usage: f64,
}

/// Optimization aggressiveness levels
#[derive(Debug, Clone)]
pub enum OptimizationAggressiveness {
    Conservative,
    Moderate,
    Aggressive,
    Maximum,
}

/// Performance baselines
#[derive(Debug, Clone)]
pub struct PerformanceBaselines {
    /// Baseline metrics
    pub metrics: BaselineMetrics,
    /// Baseline timestamp
    pub timestamp: SystemTime,
    /// Baseline configuration
    pub configuration: BaselineConfiguration,
}

/// Baseline metrics
#[derive(Debug, Clone)]
pub struct BaselineMetrics {
    /// Baseline latency
    pub latency: Duration,
    /// Baseline throughput
    pub throughput: f64,
    /// Baseline error rate
    pub error_rate: f64,
    /// Baseline resource usage
    pub resource_usage: f64,
    /// Baseline cost
    pub cost: f64,
}

/// Baseline configuration
#[derive(Debug, Clone)]
pub struct BaselineConfiguration {
    /// Configuration parameters
    pub parameters: HashMap<String, String>,
    /// Configuration timestamp
    pub timestamp: SystemTime,
    /// Configuration hash
    pub config_hash: String,
}

/// Optimization history
#[derive(Debug, Clone)]
pub struct OptimizationHistory {
    /// Historical optimizations
    pub optimizations: VecDeque<OptimizationRecord>,
    /// Performance trends
    pub trends: PerformanceTrends,
    /// Optimization effectiveness
    pub effectiveness: OptimizationEffectiveness,
}

/// Optimization record
#[derive(Debug, Clone)]
pub struct OptimizationRecord {
    /// Record identifier
    pub record_id: String,
    /// Optimization timestamp
    pub timestamp: SystemTime,
    /// Strategy applied
    pub strategy: String,
    /// Parameters changed
    pub parameters: HashMap<String, String>,
    /// Performance before
    pub before_metrics: PerformanceSnapshot,
    /// Performance after
    pub after_metrics: PerformanceSnapshot,
    /// Improvement achieved
    pub improvement: f64,
}

/// Performance snapshot
#[derive(Debug, Clone)]
pub struct PerformanceSnapshot {
    /// Snapshot timestamp
    pub timestamp: SystemTime,
    /// Latency measurement
    pub latency: Duration,
    /// Throughput measurement
    pub throughput: f64,
    /// Error rate measurement
    pub error_rate: f64,
    /// Resource usage measurement
    pub resource_usage: f64,
}

/// Performance trends analysis
#[derive(Debug, Clone)]
pub struct PerformanceTrends {
    /// Latency trend
    pub latency_trend: TrendAnalysis,
    /// Throughput trend
    pub throughput_trend: TrendAnalysis,
    /// Error rate trend
    pub error_rate_trend: TrendAnalysis,
    /// Resource usage trend
    pub resource_usage_trend: TrendAnalysis,
}

/// Trend analysis
#[derive(Debug, Clone)]
pub struct TrendAnalysis {
    /// Trend direction
    pub direction: TrendDirection,
    /// Trend strength
    pub strength: f64,
    /// Trend confidence
    pub confidence: f64,
    /// Trend forecast
    pub forecast: Vec<TrendPoint>,
}

/// Trend direction
#[derive(Debug, Clone)]
pub enum TrendDirection {
    Improving,
    Degrading,
    Stable,
    Volatile,
}

/// Trend point
#[derive(Debug, Clone)]
pub struct TrendPoint {
    /// Point timestamp
    pub timestamp: SystemTime,
    /// Predicted value
    pub value: f64,
    /// Confidence level
    pub confidence: f64,
}

/// Optimization effectiveness
#[derive(Debug, Clone)]
pub struct OptimizationEffectiveness {
    /// Overall effectiveness score
    pub overall_score: f64,
    /// Effectiveness by strategy
    pub by_strategy: HashMap<String, f64>,
    /// Success rate
    pub success_rate: f64,
    /// Average improvement
    pub avg_improvement: f64,
}

/// Machine learning models for optimization
#[derive(Debug, Clone)]
pub struct MachineLearningModels {
    /// Prediction models
    pub models: HashMap<String, PredictionModel>,
    /// Model training configuration
    pub training_config: MLTrainingConfig,
    /// Model performance
    pub performance: MLPerformance,
}

/// Prediction model
#[derive(Debug, Clone)]
pub struct PredictionModel {
    /// Model identifier
    pub model_id: String,
    /// Model type
    pub model_type: MLModelType,
    /// Model parameters
    pub parameters: HashMap<String, f64>,
    /// Training accuracy
    pub accuracy: f64,
    /// Last training timestamp
    pub last_trained: SystemTime,
    /// Model metadata
    pub metadata: HashMap<String, String>,
}

/// Machine learning model types
#[derive(Debug, Clone)]
pub enum MLModelType {
    LinearRegression,
    RandomForest,
    NeuralNetwork,
    SupportVectorMachine,
    DecisionTree,
    Custom(String),
}

/// ML training configuration
#[derive(Debug, Clone)]
pub struct MLTrainingConfig {
    /// Training data window
    pub data_window: Duration,
    /// Training frequency
    pub training_frequency: Duration,
    /// Validation split
    pub validation_split: f64,
    /// Feature selection
    pub feature_selection: Vec<String>,
    /// Model selection criteria
    pub selection_criteria: ModelSelectionCriteria,
}

/// Model selection criteria
#[derive(Debug, Clone)]
pub struct ModelSelectionCriteria {
    /// Accuracy weight
    pub accuracy_weight: f64,
    /// Performance weight
    pub performance_weight: f64,
    /// Complexity weight
    pub complexity_weight: f64,
    /// Interpretability weight
    pub interpretability_weight: f64,
}

/// ML performance metrics
#[derive(Debug, Clone)]
pub struct MLPerformance {
    /// Prediction accuracy
    pub prediction_accuracy: f64,
    /// Model training time
    pub training_time: Duration,
    /// Inference time
    pub inference_time: Duration,
    /// Model size
    pub model_size: usize,
    /// Resource consumption
    pub resource_consumption: f64,
}

/// Optimization request
#[derive(Debug, Clone)]
pub struct OptimizationRequest {
    /// Request identifier
    pub request_id: String,
    /// Current performance metrics
    pub current_metrics: PerformanceSnapshot,
    /// Optimization goals
    pub goals: OptimizationGoals,
    /// Constraints
    pub constraints: OptimizationConstraints,
    /// Priority level
    pub priority: OptimizationPriority,
}

/// Optimization goals
#[derive(Debug, Clone)]
pub struct OptimizationGoals {
    /// Improve latency
    pub improve_latency: bool,
    /// Improve throughput
    pub improve_throughput: bool,
    /// Reduce error rate
    pub reduce_error_rate: bool,
    /// Optimize resource usage
    pub optimize_resource_usage: bool,
    /// Minimize cost
    pub minimize_cost: bool,
}

/// Optimization constraints
#[derive(Debug, Clone)]
pub struct OptimizationConstraints {
    /// Maximum acceptable latency
    pub max_latency: Option<Duration>,
    /// Minimum required throughput
    pub min_throughput: Option<f64>,
    /// Maximum resource usage
    pub max_resource_usage: Option<f64>,
    /// Budget constraints
    pub budget_constraints: Option<f64>,
}

/// Optimization priority levels
#[derive(Debug, Clone)]
pub enum OptimizationPriority {
    Low,
    Normal,
    High,
    Critical,
}

/// Optimization result
#[derive(Debug, Clone)]
pub struct OptimizationResult {
    /// Request identifier
    pub request_id: String,
    /// Applied strategies
    pub applied_strategies: Vec<String>,
    /// Configuration changes
    pub configuration_changes: HashMap<String, String>,
    /// Predicted improvement
    pub predicted_improvement: f64,
    /// Optimization confidence
    pub confidence: f64,
    /// Estimated impact
    pub estimated_impact: PerformanceImpact,
}

/// Performance impact estimation
#[derive(Debug, Clone)]
pub struct PerformanceImpact {
    /// Latency impact
    pub latency_impact: f64,
    /// Throughput impact
    pub throughput_impact: f64,
    /// Error rate impact
    pub error_rate_impact: f64,
    /// Resource usage impact
    pub resource_usage_impact: f64,
    /// Cost impact
    pub cost_impact: f64,
}

impl PerformanceOptimizer {
    /// Create a new performance optimizer
    pub fn new() -> Self {
        Self {
            strategies: Vec::new(),
            config: OptimizerConfig::default(),
            baselines: PerformanceBaselines::default(),
            history: OptimizationHistory::default(),
            ml_models: MachineLearningModels::default(),
        }
    }

    /// Apply optimization strategies
    pub fn optimize(&mut self) -> Result<(), String> {
        if !self.config.enabled {
            return Ok(());
        }

        // Analyze current performance
        let current_metrics = self.collect_current_metrics()?;

        // Identify optimization opportunities
        let opportunities = self.identify_optimization_opportunities(&current_metrics)?;

        // Select best strategies
        let selected_strategies = self.select_strategies(&opportunities)?;

        // Apply optimizations
        for strategy in selected_strategies {
            self.apply_strategy(&strategy, &current_metrics)?;
        }

        Ok(())
    }

    /// Process optimization request
    pub fn process_optimization_request(&mut self, request: OptimizationRequest) -> Result<OptimizationResult, String> {
        // Analyze request
        let optimization_opportunities = self.analyze_optimization_request(&request)?;

        // Select appropriate strategies
        let strategies = self.select_strategies_for_request(&request, &optimization_opportunities)?;

        // Predict impact
        let predicted_impact = self.predict_optimization_impact(&request.current_metrics, &strategies)?;

        // Generate configuration changes
        let config_changes = self.generate_configuration_changes(&strategies)?;

        Ok(OptimizationResult {
            request_id: request.request_id,
            applied_strategies: strategies.iter().map(|s| s.name.clone()).collect(),
            configuration_changes: config_changes,
            predicted_improvement: predicted_impact.calculate_overall_improvement(),
            confidence: self.calculate_optimization_confidence(&strategies),
            estimated_impact: predicted_impact,
        })
    }

    /// Add optimization strategy
    pub fn add_strategy(&mut self, strategy: OptimizationStrategy) {
        self.strategies.push(strategy);
    }

    /// Remove optimization strategy
    pub fn remove_strategy(&mut self, strategy_id: &str) -> bool {
        if let Some(pos) = self.strategies.iter().position(|s| s.strategy_id == strategy_id) {
            self.strategies.remove(pos);
            true
        } else {
            false
        }
    }

    /// Update baseline metrics
    pub fn update_baselines(&mut self, metrics: BaselineMetrics) {
        self.baselines.metrics = metrics;
        self.baselines.timestamp = SystemTime::now();
    }

    /// Train machine learning models
    pub fn train_ml_models(&mut self) -> Result<(), String> {
        if !self.config.enable_ml {
            return Ok(());
        }

        // Collect training data from history
        let training_data = self.collect_training_data();

        // Train each model
        for (model_id, model) in &mut self.ml_models.models {
            self.train_model(model_id, model, &training_data)?;
        }

        Ok(())
    }

    /// Predict performance impact
    pub fn predict_impact(&self, strategy: &OptimizationStrategy, current_metrics: &PerformanceSnapshot) -> Result<PerformanceImpact, String> {
        if self.config.enable_ml && !self.ml_models.models.is_empty() {
            self.predict_impact_ml(strategy, current_metrics)
        } else {
            self.predict_impact_heuristic(strategy, current_metrics)
        }
    }

    /// Analyze optimization effectiveness
    pub fn analyze_effectiveness(&mut self) -> OptimizationEffectiveness {
        let mut effectiveness = OptimizationEffectiveness::default();

        if self.history.optimizations.is_empty() {
            return effectiveness;
        }

        let total_optimizations = self.history.optimizations.len();
        let mut total_improvement = 0.0;
        let mut successful_optimizations = 0;
        let mut strategy_improvements: HashMap<String, Vec<f64>> = HashMap::new();

        for record in &self.history.optimizations {
            if record.improvement > 0.0 {
                successful_optimizations += 1;
                total_improvement += record.improvement;

                strategy_improvements
                    .entry(record.strategy.clone())
                    .or_insert_with(Vec::new)
                    .push(record.improvement);
            }
        }

        effectiveness.success_rate = successful_optimizations as f64 / total_optimizations as f64;
        effectiveness.avg_improvement = if successful_optimizations > 0 {
            total_improvement / successful_optimizations as f64
        } else {
            0.0
        };

        // Calculate strategy effectiveness
        for (strategy, improvements) in strategy_improvements {
            let avg_improvement: f64 = improvements.iter().sum::<f64>() / improvements.len() as f64;
            effectiveness.by_strategy.insert(strategy, avg_improvement);
        }

        effectiveness.overall_score = (effectiveness.success_rate + effectiveness.avg_improvement.min(1.0)) / 2.0;
        effectiveness
    }

    fn collect_current_metrics(&self) -> Result<PerformanceSnapshot, String> {
        // Derive metrics from the optimization history and baselines.
        // When no history exists, use the configured baselines as current state.
        let (latency, throughput, error_rate, resource_usage) =
            if self.history.optimizations.is_empty() {
                (
                    self.baselines.metrics.latency,
                    self.baselines.metrics.throughput,
                    self.baselines.metrics.error_rate,
                    self.baselines.metrics.resource_usage,
                )
            } else {
                // Compute rolling average from the last 20 optimization records.
                let window: Vec<&OptimizationRecord> =
                    self.history.optimizations.iter().rev().take(20).collect();
                let count = window.len() as f64;

                let avg_latency_nanos: f64 = window
                    .iter()
                    .map(|r| r.after_metrics.latency.as_nanos() as f64)
                    .sum::<f64>()
                    / count;
                let avg_throughput: f64 =
                    window.iter().map(|r| r.after_metrics.throughput).sum::<f64>() / count;
                let avg_error_rate: f64 =
                    window.iter().map(|r| r.after_metrics.error_rate).sum::<f64>() / count;
                let avg_resource_usage: f64 =
                    window.iter().map(|r| r.after_metrics.resource_usage).sum::<f64>() / count;

                (
                    Duration::from_nanos(avg_latency_nanos as u64),
                    avg_throughput,
                    avg_error_rate,
                    avg_resource_usage,
                )
            };

        Ok(PerformanceSnapshot {
            timestamp: SystemTime::now(),
            latency,
            throughput,
            error_rate,
            resource_usage,
        })
    }

    fn identify_optimization_opportunities(
        &self,
        metrics: &PerformanceSnapshot,
    ) -> Result<Vec<OptimizationOpportunity>, String> {
        let mut opportunities = Vec::new();

        // Opportunity 1: High latency relative to target.
        let target_latency_nanos = self.config.performance_target.target_latency.as_nanos() as f64;
        let current_latency_nanos = metrics.latency.as_nanos() as f64;
        if target_latency_nanos > 0.0 && current_latency_nanos > target_latency_nanos {
            let excess_ratio = current_latency_nanos / target_latency_nanos;
            // Only flag as an opportunity when latency exceeds 1.5× the target.
            if excess_ratio > 1.5 {
                let potential = (1.0 - target_latency_nanos / current_latency_nanos).clamp(0.0, 1.0);
                opportunities.push(OptimizationOpportunity {
                    opportunity_type: "high_latency".to_string(),
                    potential_improvement: potential,
                    confidence: (excess_ratio - 1.0).clamp(0.0, 1.0),
                    required_resources: 0.1,
                });
            }
        }

        // Opportunity 2: High error rate (threshold 5%).
        let error_threshold = self.config.performance_target.target_error_rate.max(0.05);
        if metrics.error_rate > error_threshold {
            let excess = metrics.error_rate - error_threshold;
            let potential = (excess / metrics.error_rate).clamp(0.0, 1.0);
            opportunities.push(OptimizationOpportunity {
                opportunity_type: "high_error_rate".to_string(),
                potential_improvement: potential,
                confidence: (excess * 10.0).clamp(0.0, 1.0),
                required_resources: 0.05,
            });
        }

        // Opportunity 3: Low throughput relative to target.
        let target_throughput = self.config.performance_target.target_throughput;
        if target_throughput > 0.0 && metrics.throughput < target_throughput * 0.8 {
            let deficit_ratio = metrics.throughput / target_throughput;
            let potential = (1.0 - deficit_ratio).clamp(0.0, 1.0);
            opportunities.push(OptimizationOpportunity {
                opportunity_type: "low_throughput".to_string(),
                potential_improvement: potential,
                confidence: (1.0 - deficit_ratio).clamp(0.0, 1.0),
                required_resources: 0.15,
            });
        }

        // Opportunity 4: Resource under-utilisation (<20% load is wasteful).
        if metrics.resource_usage < 0.20 {
            let potential = 0.20 - metrics.resource_usage;
            opportunities.push(OptimizationOpportunity {
                opportunity_type: "resource_underutilization".to_string(),
                potential_improvement: potential,
                confidence: 0.7,
                required_resources: 0.0,
            });
        }

        // Opportunity 5: Resource over-utilisation (>90% is risky).
        if metrics.resource_usage > 0.90 {
            let excess = metrics.resource_usage - 0.90;
            let potential = excess.clamp(0.0, 1.0);
            opportunities.push(OptimizationOpportunity {
                opportunity_type: "resource_overutilization".to_string(),
                potential_improvement: potential,
                confidence: (excess * 5.0).clamp(0.0, 1.0),
                required_resources: 0.0,
            });
        }

        Ok(opportunities)
    }

    fn select_strategies(&self, _opportunities: &[OptimizationOpportunity]) -> Result<Vec<OptimizationStrategy>, String> {
        // Select strategies based on effectiveness and conditions
        let mut selected = Vec::new();

        for strategy in &self.strategies {
            if self.should_apply_strategy(strategy) {
                selected.push(strategy.clone());
            }
        }

        Ok(selected)
    }

    fn should_apply_strategy(&self, strategy: &OptimizationStrategy) -> bool {
        // Check conditions and effectiveness
        strategy.effectiveness > 0.5 // Simple threshold
    }

    fn apply_strategy(
        &mut self,
        strategy: &OptimizationStrategy,
        before_metrics: &PerformanceSnapshot,
    ) -> Result<(), String> {
        let record_id = format!(
            "opt_{}",
            SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        );

        // Capture parameters that are being changed so they can be rolled back.
        let mut changed_parameters: HashMap<String, String> = HashMap::new();

        // Apply the strategy by adjusting OptimizerConfig based on strategy type.
        match &strategy.strategy_type {
            OptimizationStrategyType::BatchProcessing => {
                // Reduce the optimization interval to trigger more frequent passes,
                // which effectively batches work more aggressively.
                let factor = strategy
                    .parameters
                    .get("batch_size_factor")
                    .copied()
                    .unwrap_or(0.8)
                    .clamp(0.1, 2.0);
                let old_millis = self.config.optimization_interval.as_millis();
                let new_millis = ((old_millis as f64) * factor) as u64;
                changed_parameters.insert(
                    "optimization_interval_ms".to_string(),
                    old_millis.to_string(),
                );
                self.config.optimization_interval = Duration::from_millis(new_millis.max(1));
            }

            OptimizationStrategyType::LoadBalancing => {
                // Switch aggressiveness to Aggressive so more strategies are considered.
                let old_level = format!("{:?}", self.config.aggressiveness);
                changed_parameters
                    .insert("aggressiveness".to_string(), old_level);
                self.config.aggressiveness = OptimizationAggressiveness::Aggressive;
            }

            OptimizationStrategyType::Caching => {
                // Enable ML if not yet enabled; caching goes hand-in-hand with prediction.
                changed_parameters
                    .insert("enable_ml".to_string(), self.config.enable_ml.to_string());
                self.config.enable_ml = true;
            }

            OptimizationStrategyType::Adaptive => {
                // Move to Maximum aggressiveness for adaptive strategies.
                let old_level = format!("{:?}", self.config.aggressiveness);
                changed_parameters
                    .insert("aggressiveness".to_string(), old_level);
                self.config.aggressiveness = OptimizationAggressiveness::Maximum;
            }

            OptimizationStrategyType::Prefetching => {
                // Tighten the target latency by the effectiveness factor.
                let factor = strategy.effectiveness.clamp(0.5, 1.0);
                let old_nanos = self.config.performance_target.target_latency.as_nanos();
                changed_parameters.insert(
                    "target_latency_nanos".to_string(),
                    old_nanos.to_string(),
                );
                let new_nanos = ((old_nanos as f64) * factor) as u64;
                self.config.performance_target.target_latency =
                    Duration::from_nanos(new_nanos.max(1));
            }

            OptimizationStrategyType::Compression => {
                // Raise target throughput to reflect the expected gain.
                let factor = 1.0
                    + strategy
                        .parameters
                        .get("compression_ratio")
                        .copied()
                        .unwrap_or(0.2)
                        .clamp(0.0, 1.0);
                changed_parameters.insert(
                    "target_throughput".to_string(),
                    self.config
                        .performance_target
                        .target_throughput
                        .to_string(),
                );
                self.config.performance_target.target_throughput *= factor;
            }

            OptimizationStrategyType::ConnectionPooling => {
                // Lower target resource usage to reflect better connection reuse.
                let reduction = strategy
                    .parameters
                    .get("pool_utilization_gain")
                    .copied()
                    .unwrap_or(0.1)
                    .clamp(0.0, 0.5);
                changed_parameters.insert(
                    "target_resource_usage".to_string(),
                    self.config
                        .performance_target
                        .target_resource_usage
                        .to_string(),
                );
                self.config.performance_target.target_resource_usage =
                    (self.config.performance_target.target_resource_usage - reduction).max(0.0);
            }

            OptimizationStrategyType::Custom(_name) => {
                // For custom strategies apply parameters generically via effectiveness.
                changed_parameters
                    .insert("effectiveness_applied".to_string(), strategy.effectiveness.to_string());
            }
        }

        // Estimate after-metrics by scaling the before snapshot by strategy effectiveness.
        let improvement_factor = strategy.effectiveness.clamp(0.0, 1.0);
        let after_latency_nanos = (before_metrics.latency.as_nanos() as f64
            * (1.0 - improvement_factor * 0.3)) as u64;
        let after_metrics = PerformanceSnapshot {
            timestamp: SystemTime::now(),
            latency: Duration::from_nanos(after_latency_nanos.max(1)),
            throughput: before_metrics.throughput * (1.0 + improvement_factor * 0.2),
            error_rate: (before_metrics.error_rate * (1.0 - improvement_factor * 0.5)).max(0.0),
            resource_usage: before_metrics.resource_usage,
        };

        // Compute improvement as the normalised reduction in latency.
        let before_nanos = before_metrics.latency.as_nanos() as f64;
        let improvement = if before_nanos > 0.0 {
            ((before_nanos - after_latency_nanos as f64) / before_nanos).clamp(0.0, 1.0)
        } else {
            0.0
        };

        let record = OptimizationRecord {
            record_id,
            timestamp: SystemTime::now(),
            strategy: strategy.name.clone(),
            parameters: changed_parameters,
            before_metrics: before_metrics.clone(),
            after_metrics,
            improvement,
        };

        self.history.optimizations.push_back(record);

        // Limit history size.
        if self.history.optimizations.len() > 1000 {
            self.history.optimizations.pop_front();
        }

        Ok(())
    }

    fn analyze_optimization_request(
        &self,
        request: &OptimizationRequest,
    ) -> Result<Vec<OptimizationOpportunity>, String> {
        let mut opportunities = Vec::new();
        let metrics = &request.current_metrics;

        // Latency-oriented opportunity.
        if request.goals.improve_latency {
            let baseline_nanos = self.baselines.metrics.latency.as_nanos() as f64;
            let current_nanos = metrics.latency.as_nanos() as f64;
            if baseline_nanos > 0.0 && current_nanos > baseline_nanos * 1.2 {
                let potential =
                    ((current_nanos - baseline_nanos) / current_nanos).clamp(0.0, 1.0);
                opportunities.push(OptimizationOpportunity {
                    opportunity_type: "latency_improvement".to_string(),
                    potential_improvement: potential,
                    confidence: 0.75,
                    required_resources: 0.1,
                });
            }
        }

        // Throughput-oriented opportunity.
        if request.goals.improve_throughput {
            let baseline_tp = self.baselines.metrics.throughput;
            let current_tp = metrics.throughput;
            if baseline_tp > 0.0 && current_tp < baseline_tp * 0.9 {
                let potential = ((baseline_tp - current_tp) / baseline_tp).clamp(0.0, 1.0);
                opportunities.push(OptimizationOpportunity {
                    opportunity_type: "throughput_improvement".to_string(),
                    potential_improvement: potential,
                    confidence: 0.70,
                    required_resources: 0.15,
                });
            }
        }

        // Error-rate reduction opportunity.
        if request.goals.reduce_error_rate {
            let target_err = self.config.performance_target.target_error_rate;
            if metrics.error_rate > target_err {
                let potential =
                    ((metrics.error_rate - target_err) / metrics.error_rate).clamp(0.0, 1.0);
                opportunities.push(OptimizationOpportunity {
                    opportunity_type: "error_rate_reduction".to_string(),
                    potential_improvement: potential,
                    confidence: 0.80,
                    required_resources: 0.05,
                });
            }
        }

        // Resource optimisation opportunity.
        if request.goals.optimize_resource_usage {
            let target_ru = self.config.performance_target.target_resource_usage;
            if (metrics.resource_usage - target_ru).abs() > 0.15 {
                let potential = (metrics.resource_usage - target_ru).abs().clamp(0.0, 1.0);
                opportunities.push(OptimizationOpportunity {
                    opportunity_type: "resource_optimization".to_string(),
                    potential_improvement: potential,
                    confidence: 0.65,
                    required_resources: 0.0,
                });
            }
        }

        // Cost-minimisation opportunity.
        if request.goals.minimize_cost {
            // If resource usage is above 70%, lower it to save cost.
            if metrics.resource_usage > 0.70 {
                opportunities.push(OptimizationOpportunity {
                    opportunity_type: "cost_reduction".to_string(),
                    potential_improvement: (metrics.resource_usage - 0.70).clamp(0.0, 1.0),
                    confidence: 0.60,
                    required_resources: 0.0,
                });
            }
        }

        Ok(opportunities)
    }

    fn select_strategies_for_request(
        &self,
        request: &OptimizationRequest,
        opportunities: &[OptimizationOpportunity],
    ) -> Result<Vec<OptimizationStrategy>, String> {
        let mut suitable_strategies = Vec::new();

        for strategy in &self.strategies {
            if self.strategy_matches_request(strategy, request, opportunities) {
                suitable_strategies.push(strategy.clone());
            }
        }

        // Sort by effectiveness descending so the best strategies are applied first.
        suitable_strategies.sort_by(|a, b| {
            b.effectiveness
                .partial_cmp(&a.effectiveness)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(suitable_strategies)
    }

    fn strategy_matches_request(
        &self,
        strategy: &OptimizationStrategy,
        request: &OptimizationRequest,
        opportunities: &[OptimizationOpportunity],
    ) -> bool {
        // A strategy must be minimally effective.
        if strategy.effectiveness < 0.3 {
            return false;
        }

        // Check whether any strategy condition is satisfied by the current metrics.
        let metrics = &request.current_metrics;
        let conditions_satisfied = if strategy.conditions.is_empty() {
            // No conditions means it is always applicable.
            true
        } else {
            strategy.conditions.iter().all(|cond| {
                let measured_value = match &cond.condition_type {
                    ConditionType::Latency => metrics.latency.as_nanos() as f64,
                    ConditionType::Throughput => metrics.throughput,
                    ConditionType::ErrorRate => metrics.error_rate,
                    ConditionType::ResourceUsage => metrics.resource_usage,
                    ConditionType::LoadLevel => metrics.resource_usage,
                    ConditionType::Custom(_) => 0.0,
                };
                match &cond.operator {
                    ConditionOperator::GreaterThan => measured_value > cond.value,
                    ConditionOperator::LessThan => measured_value < cond.value,
                    ConditionOperator::Equal => (measured_value - cond.value).abs() < f64::EPSILON,
                    ConditionOperator::GreaterThanOrEqual => measured_value >= cond.value,
                    ConditionOperator::LessThanOrEqual => measured_value <= cond.value,
                    ConditionOperator::NotEqual => {
                        (measured_value - cond.value).abs() >= f64::EPSILON
                    }
                }
            })
        };

        if !conditions_satisfied {
            return false;
        }

        // Verify the strategy aligns with at least one open opportunity.
        if opportunities.is_empty() {
            return true;
        }

        let strategy_type_key = match &strategy.strategy_type {
            OptimizationStrategyType::ConnectionPooling => "resource",
            OptimizationStrategyType::Caching => "latency",
            OptimizationStrategyType::Compression => "throughput",
            OptimizationStrategyType::LoadBalancing => "throughput",
            OptimizationStrategyType::BatchProcessing => "throughput",
            OptimizationStrategyType::Prefetching => "latency",
            OptimizationStrategyType::Adaptive => "",
            OptimizationStrategyType::Custom(_) => "",
        };

        // Adaptive / Custom strategies match everything.
        if strategy_type_key.is_empty() {
            return true;
        }

        opportunities
            .iter()
            .any(|op| op.opportunity_type.contains(strategy_type_key))
    }

    fn predict_optimization_impact(
        &self,
        current_metrics: &PerformanceSnapshot,
        strategies: &[OptimizationStrategy],
    ) -> Result<PerformanceImpact, String> {
        if strategies.is_empty() {
            return Ok(PerformanceImpact::default());
        }

        // Aggregate impact predictions from each strategy.
        let mut total_latency_impact = 0.0_f64;
        let mut total_throughput_impact = 0.0_f64;
        let mut total_error_rate_impact = 0.0_f64;
        let mut total_resource_impact = 0.0_f64;
        let mut total_cost_impact = 0.0_f64;

        for strategy in strategies {
            let impact = self.predict_impact(strategy, current_metrics)?;
            total_latency_impact += impact.latency_impact;
            total_throughput_impact += impact.throughput_impact;
            total_error_rate_impact += impact.error_rate_impact;
            total_resource_impact += impact.resource_usage_impact;
            total_cost_impact += impact.cost_impact;
        }

        let n = strategies.len() as f64;
        Ok(PerformanceImpact {
            latency_impact: (total_latency_impact / n).clamp(-1.0, 1.0),
            throughput_impact: (total_throughput_impact / n).clamp(-1.0, 1.0),
            error_rate_impact: (total_error_rate_impact / n).clamp(-1.0, 1.0),
            resource_usage_impact: (total_resource_impact / n).clamp(-1.0, 1.0),
            cost_impact: (total_cost_impact / n).clamp(-1.0, 1.0),
        })
    }

    fn generate_configuration_changes(
        &self,
        strategies: &[OptimizationStrategy],
    ) -> Result<HashMap<String, String>, String> {
        let mut changes: HashMap<String, String> = HashMap::new();

        for strategy in strategies {
            // Emit the parameters the strategy would modify.
            for (key, value) in &strategy.parameters {
                changes.insert(
                    format!("{}_{}", strategy.strategy_id, key),
                    value.to_string(),
                );
            }

            // Record the strategy type change token.
            let type_label = match &strategy.strategy_type {
                OptimizationStrategyType::ConnectionPooling => "connection_pooling",
                OptimizationStrategyType::Caching => "caching",
                OptimizationStrategyType::Compression => "compression",
                OptimizationStrategyType::LoadBalancing => "load_balancing",
                OptimizationStrategyType::BatchProcessing => "batch_processing",
                OptimizationStrategyType::Prefetching => "prefetching",
                OptimizationStrategyType::Adaptive => "adaptive",
                OptimizationStrategyType::Custom(name) => name.as_str(),
            };
            changes.insert(
                format!("strategy_{}_type", strategy.strategy_id),
                type_label.to_string(),
            );
            changes.insert(
                format!("strategy_{}_effectiveness", strategy.strategy_id),
                strategy.effectiveness.to_string(),
            );
        }

        Ok(changes)
    }

    fn calculate_optimization_confidence(&self, strategies: &[OptimizationStrategy]) -> f64 {
        if strategies.is_empty() {
            return 0.0;
        }

        // Base confidence is the weighted average of strategy effectiveness.
        let effectiveness_avg: f64 =
            strategies.iter().map(|s| s.effectiveness).sum::<f64>() / strategies.len() as f64;

        // Boost confidence when we have historical data to back predictions.
        let history_boost = if self.history.optimizations.is_empty() {
            0.0
        } else {
            let effectiveness = &self.history.effectiveness;
            effectiveness.success_rate * 0.3
        };

        // Reduce confidence when many strategies are applied simultaneously (interference risk).
        let strategy_count_penalty = if strategies.len() > 3 {
            0.05 * (strategies.len() - 3) as f64
        } else {
            0.0
        };

        (effectiveness_avg * 0.7 + history_boost - strategy_count_penalty).clamp(0.0, 1.0)
    }

    fn collect_training_data(&self) -> Vec<TrainingDataPoint> {
        self.history
            .optimizations
            .iter()
            .map(|record| {
                // Features: before snapshot values.
                let features = vec![
                    record.before_metrics.latency.as_nanos() as f64,
                    record.before_metrics.throughput,
                    record.before_metrics.error_rate,
                    record.before_metrics.resource_usage,
                ];
                // Targets: improvement achieved and after snapshot values.
                let targets = vec![
                    record.improvement,
                    record.after_metrics.latency.as_nanos() as f64,
                    record.after_metrics.throughput,
                    record.after_metrics.error_rate,
                    record.after_metrics.resource_usage,
                ];
                TrainingDataPoint {
                    features,
                    targets,
                    timestamp: record.timestamp,
                }
            })
            .collect()
    }

    fn train_model(
        &mut self,
        model_id: &str,
        model: &mut PredictionModel,
        training_data: &[TrainingDataPoint],
    ) -> Result<(), String> {
        if training_data.is_empty() {
            return Ok(());
        }

        // Simple linear regression coefficient update based on normalised feature means.
        // This is intentionally a lightweight heuristic — a full ML framework is out
        // of scope for this module.
        let n = training_data.len() as f64;
        let feature_count = training_data[0].features.len();

        // Compute per-feature mean.
        let mut feature_means = vec![0.0_f64; feature_count];
        for dp in training_data {
            for (i, &f) in dp.features.iter().enumerate() {
                if i < feature_means.len() {
                    feature_means[i] += f;
                }
            }
        }
        for fm in &mut feature_means {
            *fm /= n;
        }

        // Compute mean target (improvement index 0).
        let target_mean: f64 = training_data.iter().map(|dp| dp.targets[0]).sum::<f64>() / n;

        // Store normalised feature means as model parameters.
        for (i, mean) in feature_means.iter().enumerate() {
            model
                .parameters
                .insert(format!("feature_mean_{i}"), *mean);
        }
        model
            .parameters
            .insert("target_mean".to_string(), target_mean);
        model.accuracy = target_mean.clamp(0.0, 1.0);
        model.last_trained = SystemTime::now();

        // Update ML performance tracking.
        self.ml_models
            .performance
            .prediction_accuracy = target_mean.clamp(0.0, 1.0);

        // Store back under model_id key if present.
        if let Some(entry) = self.ml_models.models.get_mut(model_id) {
            entry.accuracy = model.accuracy;
            entry.last_trained = model.last_trained;
            for (k, v) in &model.parameters {
                entry.parameters.insert(k.clone(), *v);
            }
        }

        Ok(())
    }

    fn predict_impact_ml(
        &self,
        strategy: &OptimizationStrategy,
        current_metrics: &PerformanceSnapshot,
    ) -> Result<PerformanceImpact, String> {
        // Look for a model trained on this strategy type.
        let model_opt = self
            .ml_models
            .models
            .values()
            .max_by(|a, b| a.accuracy.partial_cmp(&b.accuracy).unwrap_or(std::cmp::Ordering::Equal));

        match model_opt {
            Some(model) if model.accuracy > 0.0 => {
                // Use the stored target mean as the predicted improvement scaling factor.
                let predicted_improvement = model
                    .parameters
                    .get("target_mean")
                    .copied()
                    .unwrap_or(strategy.effectiveness * 0.5);

                // Scale heuristic predictions by the ML model's accuracy.
                let heuristic = self.predict_impact_heuristic(strategy, current_metrics)?;
                Ok(PerformanceImpact {
                    latency_impact: heuristic.latency_impact * (1.0 + predicted_improvement * 0.2),
                    throughput_impact: heuristic.throughput_impact
                        * (1.0 + predicted_improvement * 0.2),
                    error_rate_impact: heuristic.error_rate_impact,
                    resource_usage_impact: heuristic.resource_usage_impact,
                    cost_impact: heuristic.cost_impact * (1.0 - predicted_improvement * 0.1),
                })
            }
            _ => {
                // No reliable ML model yet; fall back to heuristic.
                self.predict_impact_heuristic(strategy, current_metrics)
            }
        }
    }

    fn predict_impact_heuristic(
        &self,
        strategy: &OptimizationStrategy,
        current_metrics: &PerformanceSnapshot,
    ) -> Result<PerformanceImpact, String> {
        let eff = strategy.effectiveness.clamp(0.0, 1.0);

        // Each strategy type has a characteristic impact profile.
        let (lat, tput, err, res, cost) = match &strategy.strategy_type {
            OptimizationStrategyType::Caching => {
                // Caching primarily improves latency and throughput.
                let factor = strategy
                    .parameters
                    .get("cache_hit_rate")
                    .copied()
                    .unwrap_or(0.7)
                    .clamp(0.0, 1.0);
                (eff * factor * 0.5, eff * factor * 0.3, 0.0, eff * 0.05, -eff * factor * 0.1)
            }

            OptimizationStrategyType::ConnectionPooling => {
                // Reduces latency and resource usage through reuse.
                let pool_size = strategy
                    .parameters
                    .get("pool_size_factor")
                    .copied()
                    .unwrap_or(1.0)
                    .clamp(0.5, 4.0);
                (eff * 0.3 * pool_size.min(2.0) / 2.0, eff * 0.2, 0.0, -eff * 0.15, -eff * 0.05)
            }

            OptimizationStrategyType::Compression => {
                // Improves throughput at the cost of a small CPU overhead.
                let ratio = strategy
                    .parameters
                    .get("compression_ratio")
                    .copied()
                    .unwrap_or(0.5)
                    .clamp(0.0, 1.0);
                (0.0, eff * ratio * 0.4, 0.0, eff * 0.05, -eff * ratio * 0.2)
            }

            OptimizationStrategyType::LoadBalancing => {
                // Reduces error rates and improves throughput.
                (eff * 0.1, eff * 0.35, eff * 0.2, 0.0, 0.0)
            }

            OptimizationStrategyType::BatchProcessing => {
                // Improves throughput significantly; latency may increase slightly.
                let size_factor = strategy
                    .parameters
                    .get("batch_size_factor")
                    .copied()
                    .unwrap_or(1.0);
                (-eff * 0.1 * size_factor, eff * 0.5 * size_factor, 0.0, -eff * 0.1, -eff * 0.15)
            }

            OptimizationStrategyType::Prefetching => {
                // Primarily reduces latency for sequential workloads.
                let prefetch_depth = strategy
                    .parameters
                    .get("prefetch_depth")
                    .copied()
                    .unwrap_or(2.0)
                    .clamp(1.0, 8.0);
                (eff * prefetch_depth * 0.1, eff * 0.1, 0.0, eff * 0.05, 0.0)
            }

            OptimizationStrategyType::Adaptive => {
                // Balanced improvement across all dimensions.
                (eff * 0.2, eff * 0.2, eff * 0.1, eff * 0.1, -eff * 0.1)
            }

            OptimizationStrategyType::Custom(_) => {
                // Generic prediction based on effectiveness only.
                let baseline_lat = self.baselines.metrics.latency.as_nanos() as f64;
                let current_lat = current_metrics.latency.as_nanos() as f64;
                let lat_headroom = if baseline_lat > 0.0 {
                    ((current_lat - baseline_lat) / current_lat).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                (eff * lat_headroom, eff * 0.15, eff * 0.05, 0.0, 0.0)
            }
        };

        Ok(PerformanceImpact {
            latency_impact: lat,
            throughput_impact: tput,
            error_rate_impact: err,
            resource_usage_impact: res,
            cost_impact: cost,
        })
    }
}

impl PerformanceImpact {
    /// Calculate overall improvement score
    pub fn calculate_overall_improvement(&self) -> f64 {
        (self.latency_impact + self.throughput_impact + self.error_rate_impact + self.resource_usage_impact) / 4.0
    }
}

/// Optimization opportunity
#[derive(Debug, Clone)]
pub struct OptimizationOpportunity {
    /// Opportunity type
    pub opportunity_type: String,
    /// Potential improvement
    pub potential_improvement: f64,
    /// Confidence level
    pub confidence: f64,
    /// Required resources
    pub required_resources: f64,
}

/// Training data point for ML models
#[derive(Debug, Clone)]
pub struct TrainingDataPoint {
    /// Features
    pub features: Vec<f64>,
    /// Target values
    pub targets: Vec<f64>,
    /// Timestamp
    pub timestamp: SystemTime,
}

// Default implementations
impl Default for OptimizerConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            optimization_interval: Duration::from_secs(300), // 5 minutes
            enable_ml: true,
            performance_target: PerformanceTarget::default(),
            aggressiveness: OptimizationAggressiveness::Moderate,
        }
    }
}

impl Default for PerformanceTarget {
    fn default() -> Self {
        Self {
            target_latency: Duration::from_millis(100),
            target_throughput: 1000.0,
            target_error_rate: 0.01,
            target_resource_usage: 0.8,
        }
    }
}

impl Default for PerformanceBaselines {
    fn default() -> Self {
        Self {
            metrics: BaselineMetrics::default(),
            timestamp: SystemTime::now(),
            configuration: BaselineConfiguration::default(),
        }
    }
}

impl Default for BaselineMetrics {
    fn default() -> Self {
        Self {
            latency: Duration::from_millis(200),
            throughput: 500.0,
            error_rate: 0.02,
            resource_usage: 0.6,
            cost: 100.0,
        }
    }
}

impl Default for BaselineConfiguration {
    fn default() -> Self {
        Self {
            parameters: HashMap::new(),
            timestamp: SystemTime::now(),
            config_hash: "default".to_string(),
        }
    }
}

impl Default for OptimizationHistory {
    fn default() -> Self {
        Self {
            optimizations: VecDeque::new(),
            trends: PerformanceTrends::default(),
            effectiveness: OptimizationEffectiveness::default(),
        }
    }
}

impl Default for PerformanceTrends {
    fn default() -> Self {
        Self {
            latency_trend: TrendAnalysis::default(),
            throughput_trend: TrendAnalysis::default(),
            error_rate_trend: TrendAnalysis::default(),
            resource_usage_trend: TrendAnalysis::default(),
        }
    }
}

impl Default for TrendAnalysis {
    fn default() -> Self {
        Self {
            direction: TrendDirection::Stable,
            strength: 0.0,
            confidence: 0.0,
            forecast: Vec::new(),
        }
    }
}

impl Default for OptimizationEffectiveness {
    fn default() -> Self {
        Self {
            overall_score: 0.0,
            by_strategy: HashMap::new(),
            success_rate: 0.0,
            avg_improvement: 0.0,
        }
    }
}

impl Default for MachineLearningModels {
    fn default() -> Self {
        Self {
            models: HashMap::new(),
            training_config: MLTrainingConfig::default(),
            performance: MLPerformance::default(),
        }
    }
}

impl Default for MLTrainingConfig {
    fn default() -> Self {
        Self {
            data_window: Duration::from_secs(3600 * 24 * 7), // 1 week
            training_frequency: Duration::from_secs(3600 * 24), // Daily
            validation_split: 0.2,
            feature_selection: Vec::new(),
            selection_criteria: ModelSelectionCriteria::default(),
        }
    }
}

impl Default for ModelSelectionCriteria {
    fn default() -> Self {
        Self {
            accuracy_weight: 0.4,
            performance_weight: 0.3,
            complexity_weight: 0.2,
            interpretability_weight: 0.1,
        }
    }
}

impl Default for MLPerformance {
    fn default() -> Self {
        Self {
            prediction_accuracy: 0.0,
            training_time: Duration::from_millis(0),
            inference_time: Duration::from_millis(0),
            model_size: 0,
            resource_consumption: 0.0,
        }
    }
}

impl Default for PerformanceImpact {
    fn default() -> Self {
        Self {
            latency_impact: 0.0,
            throughput_impact: 0.0,
            error_rate_impact: 0.0,
            resource_usage_impact: 0.0,
            cost_impact: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_optimizer_with_history() -> PerformanceOptimizer {
        let mut opt = PerformanceOptimizer::new();
        // Inject a fake optimization record so collect_current_metrics uses the history branch.
        let record = OptimizationRecord {
            record_id: "test_record".to_string(),
            timestamp: SystemTime::now(),
            strategy: "test".to_string(),
            parameters: HashMap::new(),
            before_metrics: PerformanceSnapshot {
                timestamp: SystemTime::now(),
                latency: Duration::from_millis(200),
                throughput: 800.0,
                error_rate: 0.03,
                resource_usage: 0.55,
            },
            after_metrics: PerformanceSnapshot {
                timestamp: SystemTime::now(),
                latency: Duration::from_millis(150),
                throughput: 900.0,
                error_rate: 0.02,
                resource_usage: 0.50,
            },
            improvement: 0.25,
        };
        opt.history.optimizations.push_back(record);
        opt
    }

    fn make_strategy(strategy_type: OptimizationStrategyType, effectiveness: f64) -> OptimizationStrategy {
        OptimizationStrategy {
            strategy_id: "test_strategy".to_string(),
            name: "Test Strategy".to_string(),
            strategy_type,
            parameters: HashMap::new(),
            effectiveness,
            conditions: Vec::new(),
        }
    }

    #[test]
    fn test_collect_current_metrics_no_history() {
        let opt = PerformanceOptimizer::new();
        let snapshot = opt.collect_current_metrics().expect("should succeed");
        // Without history, metrics come from baselines.
        assert_eq!(snapshot.latency, Duration::from_millis(200));
        assert!((snapshot.throughput - 500.0).abs() < 1.0);
        assert!(snapshot.error_rate > 0.0);
    }

    #[test]
    fn test_collect_current_metrics_with_history() {
        let opt = make_optimizer_with_history();
        let snapshot = opt.collect_current_metrics().expect("should succeed");
        // With history, we get the rolling average from records.
        assert!(snapshot.latency.as_millis() > 0);
        assert!(snapshot.throughput > 0.0);
    }

    #[test]
    fn test_collect_current_metrics_returns_nondefault_data() {
        let opt = make_optimizer_with_history();
        let snapshot = opt.collect_current_metrics().expect("should succeed");
        // After_metrics latency was 150ms — the average should be 150ms.
        assert_eq!(snapshot.latency, Duration::from_millis(150));
        assert!((snapshot.throughput - 900.0).abs() < 0.01);
    }

    #[test]
    fn test_identify_opportunities_high_latency() {
        let opt = PerformanceOptimizer::new();
        // Target latency is 100ms; current is 350ms → ratio 3.5 >> 1.5 threshold.
        let metrics = PerformanceSnapshot {
            timestamp: SystemTime::now(),
            latency: Duration::from_millis(350),
            throughput: 1000.0,
            error_rate: 0.01,
            resource_usage: 0.5,
        };
        let opps = opt.identify_optimization_opportunities(&metrics).expect("should succeed");
        assert!(!opps.is_empty(), "Expected at least one opportunity for high latency");
        let high_lat = opps.iter().find(|o| o.opportunity_type == "high_latency");
        assert!(high_lat.is_some(), "Expected a high_latency opportunity");
        assert!(high_lat.expect("high_lat should be Some").potential_improvement > 0.0);
    }

    #[test]
    fn test_identify_opportunities_high_error_rate() {
        let opt = PerformanceOptimizer::new();
        let metrics = PerformanceSnapshot {
            timestamp: SystemTime::now(),
            latency: Duration::from_millis(100),
            throughput: 1000.0,
            // Error rate 10% >> threshold of 5%.
            error_rate: 0.10,
            resource_usage: 0.5,
        };
        let opps = opt.identify_optimization_opportunities(&metrics).expect("should succeed");
        let high_err = opps.iter().find(|o| o.opportunity_type == "high_error_rate");
        assert!(high_err.is_some(), "Expected a high_error_rate opportunity");
    }

    #[test]
    fn test_identify_opportunities_no_issues() {
        let opt = PerformanceOptimizer::new();
        // All metrics within acceptable range → no opportunities expected.
        let metrics = PerformanceSnapshot {
            timestamp: SystemTime::now(),
            latency: Duration::from_millis(80),  // Below target 100ms
            throughput: 1100.0,                  // Above target 1000.0
            error_rate: 0.005,                   // Below target 0.01
            resource_usage: 0.5,                 // Normal
        };
        let opps = opt.identify_optimization_opportunities(&metrics).expect("should succeed");
        // No high-latency, high-error, low-throughput, or over/under utilisation opportunities.
        assert!(
            !opps.iter().any(|o| {
                o.opportunity_type == "high_latency"
                    || o.opportunity_type == "high_error_rate"
                    || o.opportunity_type == "low_throughput"
                    || o.opportunity_type == "resource_overutilization"
            }),
            "Unexpected opportunities: {:?}",
            opps
        );
    }

    #[test]
    fn test_apply_strategy_changes_config_caching() {
        let mut opt = PerformanceOptimizer::new();
        // Start with ML disabled.
        opt.config.enable_ml = false;
        let before_metrics = opt.collect_current_metrics().expect("should succeed");
        let strategy = make_strategy(OptimizationStrategyType::Caching, 0.8);
        opt.apply_strategy(&strategy, &before_metrics).expect("should succeed");

        // Caching strategy should have enabled ML.
        assert!(opt.config.enable_ml, "Caching strategy should enable ML");
        // A record should have been added to history.
        assert_eq!(opt.history.optimizations.len(), 1);
    }

    #[test]
    fn test_apply_strategy_changes_config_batch_processing() {
        let mut opt = PerformanceOptimizer::new();
        let original_interval = opt.config.optimization_interval;
        let before_metrics = opt.collect_current_metrics().expect("should succeed");

        let mut strategy = make_strategy(OptimizationStrategyType::BatchProcessing, 0.7);
        // batch_size_factor of 0.5 should halve the optimization interval.
        strategy.parameters.insert("batch_size_factor".to_string(), 0.5);

        opt.apply_strategy(&strategy, &before_metrics).expect("should succeed");

        // Interval should be shorter after applying BatchProcessing.
        assert!(
            opt.config.optimization_interval < original_interval,
            "Expected shorter interval after batch processing strategy"
        );
    }

    #[test]
    fn test_apply_strategy_records_history() {
        let mut opt = PerformanceOptimizer::new();
        let before_metrics = opt.collect_current_metrics().expect("should succeed");
        let strategy = make_strategy(OptimizationStrategyType::LoadBalancing, 0.9);

        opt.apply_strategy(&strategy, &before_metrics).expect("should succeed");
        opt.apply_strategy(&strategy, &before_metrics).expect("should succeed");

        assert_eq!(opt.history.optimizations.len(), 2, "Two records expected");
        for record in &opt.history.optimizations {
            assert_eq!(record.strategy, strategy.name);
        }
    }

    #[test]
    fn test_apply_strategy_improvement_is_positive_for_effective_strategy() {
        let mut opt = PerformanceOptimizer::new();
        let before_metrics = PerformanceSnapshot {
            timestamp: SystemTime::now(),
            latency: Duration::from_millis(500),
            throughput: 200.0,
            error_rate: 0.05,
            resource_usage: 0.8,
        };
        // High effectiveness strategy.
        let strategy = make_strategy(OptimizationStrategyType::Adaptive, 0.95);
        opt.apply_strategy(&strategy, &before_metrics).expect("should succeed");

        let record = opt.history.optimizations.back().expect("record should exist");
        assert!(record.improvement >= 0.0, "Improvement should be non-negative");
    }

    #[test]
    fn test_predict_impact_heuristic_caching() {
        let opt = PerformanceOptimizer::new();
        let metrics = PerformanceSnapshot {
            timestamp: SystemTime::now(),
            latency: Duration::from_millis(200),
            throughput: 500.0,
            error_rate: 0.03,
            resource_usage: 0.6,
        };
        let mut strategy = make_strategy(OptimizationStrategyType::Caching, 0.8);
        strategy.parameters.insert("cache_hit_rate".to_string(), 0.9);

        let impact = opt.predict_impact_heuristic(&strategy, &metrics).expect("should succeed");
        // Caching should improve latency and throughput.
        assert!(impact.latency_impact > 0.0, "Expected positive latency improvement");
        assert!(impact.throughput_impact > 0.0, "Expected positive throughput improvement");
    }

    #[test]
    fn test_analyze_optimization_request_goals() {
        let opt = PerformanceOptimizer::new();
        let request = OptimizationRequest {
            request_id: "req_001".to_string(),
            current_metrics: PerformanceSnapshot {
                timestamp: SystemTime::now(),
                latency: Duration::from_millis(300), // Much higher than baseline 200ms
                throughput: 400.0,
                error_rate: 0.08,
                resource_usage: 0.85,
            },
            goals: OptimizationGoals {
                improve_latency: true,
                improve_throughput: true,
                reduce_error_rate: true,
                optimize_resource_usage: true,
                minimize_cost: true,
            },
            constraints: OptimizationConstraints {
                max_latency: None,
                min_throughput: None,
                max_resource_usage: None,
                budget_constraints: None,
            },
            priority: OptimizationPriority::High,
        };

        let opps = opt.analyze_optimization_request(&request).expect("should succeed");
        assert!(!opps.is_empty(), "Expected opportunities from degraded metrics");
    }

    #[test]
    fn test_calculate_optimization_confidence_empty_strategies() {
        let opt = PerformanceOptimizer::new();
        let confidence = opt.calculate_optimization_confidence(&[]);
        assert_eq!(confidence, 0.0, "Empty strategy list should yield 0.0 confidence");
    }

    #[test]
    fn test_calculate_optimization_confidence_single_strategy() {
        let opt = PerformanceOptimizer::new();
        let strategy = make_strategy(OptimizationStrategyType::Adaptive, 1.0);
        let confidence = opt.calculate_optimization_confidence(&[strategy]);
        assert!(confidence > 0.0, "Confidence should be positive for effective strategy");
        assert!(confidence <= 1.0, "Confidence should not exceed 1.0");
    }

    #[test]
    fn test_collect_training_data_from_history() {
        let opt = make_optimizer_with_history();
        let data = opt.collect_training_data();
        assert_eq!(data.len(), 1, "Should return one training point per history record");
        assert_eq!(data[0].features.len(), 4, "Expected 4 features");
        assert_eq!(data[0].targets.len(), 5, "Expected 5 targets");
    }

    #[test]
    fn test_generate_configuration_changes_includes_strategy_params() {
        let opt = PerformanceOptimizer::new();
        let mut strategy = make_strategy(OptimizationStrategyType::Compression, 0.7);
        strategy.parameters.insert("compression_ratio".to_string(), 0.6);

        let changes = opt.generate_configuration_changes(&[strategy.clone()]).expect("should succeed");
        // Should contain the strategy's parameters.
        let key = format!("{}_compression_ratio", strategy.strategy_id);
        assert!(changes.contains_key(&key), "Expected compression_ratio in changes");
    }
}