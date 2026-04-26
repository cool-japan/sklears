//! Monitoring Agents Module
//!
//! This module provides comprehensive real-time monitoring capabilities including
//! monitoring agents, metrics collection, alerting systems, and performance tracking
//! for notification channel optimization.

use std::collections::{HashMap, VecDeque};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::time::{Duration, SystemTime};

use super::optimization_engine::{ConditionOperator, TrendDirection};

/// Performance monitor for real-time tracking
#[derive(Debug, Clone)]
pub struct PerformanceMonitor {
    /// Monitoring agents
    pub agents: HashMap<String, MonitoringAgent>,
    /// Real-time metrics
    pub real_time_metrics: RealTimeMetrics,
    /// Performance alerts
    pub alerts: PerformanceAlerts,
    /// Monitor configuration
    pub config: MonitorConfig,
}

/// Monitoring agent
#[derive(Debug, Clone)]
pub struct MonitoringAgent {
    /// Agent identifier
    pub agent_id: String,
    /// Agent type
    pub agent_type: AgentType,
    /// Monitored metrics
    pub metrics: Vec<String>,
    /// Sampling configuration
    pub sampling: SamplingConfig,
    /// Agent statistics
    pub statistics: AgentStatistics,
    /// Running state flag — shared so clones observe the same state.
    running: Arc<AtomicBool>,
}

/// Agent types
#[derive(Debug, Clone)]
pub enum AgentType {
    ConnectionMonitor,
    CacheMonitor,
    CompressionMonitor,
    ResourceMonitor,
    Custom(String),
}

/// Sampling configuration
#[derive(Debug, Clone)]
pub struct SamplingConfig {
    /// Sampling interval
    pub interval: Duration,
    /// Sampling strategy
    pub strategy: SamplingStrategy,
    /// Sample size
    pub sample_size: usize,
    /// Enable adaptive sampling
    pub adaptive_sampling: bool,
}

/// Sampling strategies
#[derive(Debug, Clone)]
pub enum SamplingStrategy {
    Fixed,
    Adaptive,
    EventDriven,
    Statistical,
}

/// Agent statistics
#[derive(Debug, Clone)]
pub struct AgentStatistics {
    /// Samples collected
    pub samples_collected: u64,
    /// Sample processing time
    pub processing_time: Duration,
    /// Agent uptime
    pub uptime: Duration,
    /// Error count
    pub error_count: u64,
}

/// Real-time metrics
#[derive(Debug, Clone)]
pub struct RealTimeMetrics {
    /// Current metrics
    pub current: HashMap<String, MetricValue>,
    /// Metric history
    pub history: HashMap<String, VecDeque<MetricDataPoint>>,
    /// Metric trends
    pub trends: HashMap<String, MetricTrend>,
    /// Update frequency
    pub update_frequency: Duration,
    /// Collection active flag — shared across clones.
    collecting: Arc<AtomicBool>,
}

/// Metric value
#[derive(Debug, Clone)]
pub struct MetricValue {
    /// Value
    pub value: f64,
    /// Timestamp
    pub timestamp: SystemTime,
    /// Quality indicator
    pub quality: MetricQuality,
    /// Source agent
    pub source: String,
}

/// Metric quality indicators
#[derive(Debug, Clone)]
pub enum MetricQuality {
    High,
    Medium,
    Low,
    Uncertain,
}

/// Metric data point
#[derive(Debug, Clone)]
pub struct MetricDataPoint {
    /// Data timestamp
    pub timestamp: SystemTime,
    /// Data value
    pub value: f64,
    /// Data source
    pub source: String,
    /// Data context
    pub context: HashMap<String, String>,
}

/// Metric trend
#[derive(Debug, Clone)]
pub struct MetricTrend {
    /// Trend direction
    pub direction: TrendDirection,
    /// Trend velocity
    pub velocity: f64,
    /// Trend acceleration
    pub acceleration: f64,
    /// Trend confidence
    pub confidence: f64,
}

/// Performance alerts system
#[derive(Debug, Clone)]
pub struct PerformanceAlerts {
    /// Alert rules
    pub rules: Vec<AlertRule>,
    /// Active alerts
    pub active_alerts: HashMap<String, PerformanceAlert>,
    /// Alert history
    pub history: VecDeque<AlertRecord>,
    /// Alert configuration
    pub config: AlertConfig,
    /// Evaluation active flag — shared across clones.
    evaluating: Arc<AtomicBool>,
}

/// Alert rule
#[derive(Debug, Clone)]
pub struct AlertRule {
    /// Rule identifier
    pub rule_id: String,
    /// Rule name
    pub name: String,
    /// Alert condition
    pub condition: AlertCondition,
    /// Alert severity
    pub severity: AlertSeverity,
    /// Alert actions
    pub actions: Vec<AlertAction>,
}

/// Alert condition
#[derive(Debug, Clone)]
pub struct AlertCondition {
    /// Metric name
    pub metric: String,
    /// Threshold value
    pub threshold: f64,
    /// Comparison operator
    pub operator: ConditionOperator,
    /// Duration requirement
    pub duration: Duration,
}

/// Alert severity levels
#[derive(Debug, Clone)]
pub enum AlertSeverity {
    Info,
    Warning,
    Critical,
    Emergency,
}

/// Alert actions
#[derive(Debug, Clone)]
pub enum AlertAction {
    Notify(String),
    Execute(String),
    Optimize,
    Escalate,
    Custom(String),
}

/// Performance alert
#[derive(Debug, Clone)]
pub struct PerformanceAlert {
    /// Alert identifier
    pub alert_id: String,
    /// Alert rule
    pub rule_id: String,
    /// Alert timestamp
    pub timestamp: SystemTime,
    /// Alert message
    pub message: String,
    /// Alert severity
    pub severity: AlertSeverity,
    /// Alert context
    pub context: HashMap<String, String>,
}

/// Alert record
#[derive(Debug, Clone)]
pub struct AlertRecord {
    /// Record identifier
    pub record_id: String,
    /// Alert identifier
    pub alert_id: String,
    /// Record timestamp
    pub timestamp: SystemTime,
    /// Event type
    pub event_type: AlertEventType,
    /// Event details
    pub details: HashMap<String, String>,
}

/// Alert event types
#[derive(Debug, Clone)]
pub enum AlertEventType {
    Triggered,
    Acknowledged,
    Resolved,
    Escalated,
}

/// Alert configuration
#[derive(Debug, Clone)]
pub struct AlertConfig {
    /// Enable alerts
    pub enabled: bool,
    /// Alert evaluation interval
    pub evaluation_interval: Duration,
    /// Maximum active alerts
    pub max_active_alerts: usize,
    /// Alert retention period
    pub retention_period: Duration,
}

/// Monitor configuration
#[derive(Debug, Clone)]
pub struct MonitorConfig {
    /// Enable monitoring
    pub enabled: bool,
    /// Monitoring interval
    pub interval: Duration,
    /// Data retention period
    pub retention_period: Duration,
    /// Enable real-time updates
    pub real_time_updates: bool,
    /// Performance impact limit
    pub performance_impact_limit: f64,
}

/// Monitoring request for custom monitoring
#[derive(Debug, Clone)]
pub struct MonitoringRequest {
    /// Request identifier
    pub request_id: String,
    /// Metrics to monitor
    pub metrics: Vec<String>,
    /// Monitoring duration
    pub duration: Duration,
    /// Sampling configuration
    pub sampling: SamplingConfig,
    /// Alert thresholds
    pub alert_thresholds: HashMap<String, f64>,
}

/// Monitoring result
#[derive(Debug, Clone)]
pub struct MonitoringResult {
    /// Request identifier
    pub request_id: String,
    /// Collected metrics
    pub metrics: HashMap<String, Vec<MetricDataPoint>>,
    /// Alert events
    pub alerts: Vec<PerformanceAlert>,
    /// Monitoring summary
    pub summary: MonitoringSummary,
}

/// Monitoring summary
#[derive(Debug, Clone)]
pub struct MonitoringSummary {
    /// Monitoring duration
    pub duration: Duration,
    /// Total samples collected
    pub total_samples: u64,
    /// Metrics coverage
    pub metrics_coverage: f64,
    /// Alert count
    pub alert_count: usize,
    /// Performance impact
    pub performance_impact: f64,
}

impl PerformanceMonitor {
    /// Create a new performance monitor
    pub fn new() -> Self {
        Self {
            agents: HashMap::new(),
            real_time_metrics: RealTimeMetrics::new(),
            alerts: PerformanceAlerts::new(),
            config: MonitorConfig::default(),
        }
    }

    /// Add monitoring agent
    pub fn add_agent(&mut self, agent: MonitoringAgent) {
        self.agents.insert(agent.agent_id.clone(), agent);
    }

    /// Remove monitoring agent
    pub fn remove_agent(&mut self, agent_id: &str) -> Option<MonitoringAgent> {
        self.agents.remove(agent_id)
    }

    /// Get agent by ID
    pub fn get_agent(&self, agent_id: &str) -> Option<&MonitoringAgent> {
        self.agents.get(agent_id)
    }

    /// Get mutable agent by ID
    pub fn get_agent_mut(&mut self, agent_id: &str) -> Option<&mut MonitoringAgent> {
        self.agents.get_mut(agent_id)
    }

    /// Start monitoring
    pub fn start_monitoring(&mut self) -> Result<(), String> {
        if !self.config.enabled {
            return Err("Monitoring is disabled".to_string());
        }

        // Start all monitoring agents
        for agent in self.agents.values_mut() {
            agent.start()?;
        }

        // Initialize real-time metrics collection
        self.real_time_metrics.start_collection(self.config.interval)?;

        // Start alert evaluation
        self.alerts.start_evaluation(self.alerts.config.evaluation_interval)?;

        Ok(())
    }

    /// Stop monitoring
    pub fn stop_monitoring(&mut self) -> Result<(), String> {
        // Stop all monitoring agents
        for agent in self.agents.values_mut() {
            agent.stop()?;
        }

        // Stop real-time metrics collection
        self.real_time_metrics.stop_collection()?;

        // Stop alert evaluation
        self.alerts.stop_evaluation()?;

        Ok(())
    }

    /// Process monitoring request
    pub fn process_monitoring_request(&mut self, request: MonitoringRequest) -> Result<MonitoringResult, String> {
        let start_time = SystemTime::now();
        let mut collected_metrics = HashMap::new();
        let mut triggered_alerts = Vec::new();

        // Create temporary agents for requested metrics
        let mut temp_agents = Vec::new();
        for (i, metric) in request.metrics.iter().enumerate() {
            let agent = MonitoringAgent::new(
                format!("temp_agent_{}", i),
                AgentType::Custom(metric.clone()),
                vec![metric.clone()],
                request.sampling.clone(),
            );
            temp_agents.push(agent);
        }

        // Start temporary monitoring
        for agent in &mut temp_agents {
            agent.start()?;
        }

        // Collect data for the requested duration.
        // This is a pull-driven simulation; in production this would be event-driven.
        let sampling_interval_secs = request.sampling.interval.as_secs().max(1);
        let sample_count = (request.duration.as_secs() / sampling_interval_secs).max(1);
        let metrics_count = request.metrics.len();
        for agent in &mut temp_agents {
            let mut metric_data = Vec::new();
            for _ in 0..sample_count {
                let data_point = agent.collect_sample()?;
                // Check alert thresholds before moving data_point.
                if let Some(&threshold) = request.alert_thresholds.get(&agent.metrics[0]) {
                    if data_point.value > threshold {
                        let alert = PerformanceAlert {
                            alert_id: format!(
                                "temp_alert_{}",
                                SystemTime::now()
                                    .duration_since(SystemTime::UNIX_EPOCH)
                                    .unwrap_or_default()
                                    .as_nanos()
                            ),
                            rule_id: "threshold_rule".to_string(),
                            timestamp: SystemTime::now(),
                            message: format!(
                                "Metric {} exceeded threshold: {} > {}",
                                agent.metrics[0], data_point.value, threshold
                            ),
                            severity: AlertSeverity::Warning,
                            context: HashMap::new(),
                        };
                        triggered_alerts.push(alert);
                    }
                }
                metric_data.push(data_point);
            }
            collected_metrics.insert(agent.metrics[0].clone(), metric_data);
        }

        // Stop temporary monitoring.
        for agent in &mut temp_agents {
            agent.stop()?;
        }

        let monitoring_duration = start_time.elapsed().unwrap_or(Duration::from_millis(0));
        let total_samples: u64 = collected_metrics.values().map(|v| v.len() as u64).sum();
        let alert_count = triggered_alerts.len();

        Ok(MonitoringResult {
            request_id: request.request_id,
            metrics: collected_metrics,
            alerts: triggered_alerts,
            summary: MonitoringSummary {
                duration: monitoring_duration,
                total_samples,
                // Coverage is always 1.0 when all requested metrics were collected.
                metrics_coverage: if metrics_count > 0 { 1.0 } else { 0.0 },
                alert_count,
                performance_impact: 0.1,
            },
        })
    }

    /// Update real-time metrics
    pub fn update_metrics(&mut self, metric_name: String, value: f64, source: String) {
        self.real_time_metrics.update_metric(metric_name, value, source);
    }

    /// Get current metric value
    pub fn get_current_metric(&self, metric_name: &str) -> Option<&MetricValue> {
        self.real_time_metrics.current.get(metric_name)
    }

    /// Get metric history
    pub fn get_metric_history(&self, metric_name: &str) -> Option<&VecDeque<MetricDataPoint>> {
        self.real_time_metrics.history.get(metric_name)
    }

    /// Evaluate alert rules
    pub fn evaluate_alerts(&mut self) -> Result<Vec<PerformanceAlert>, String> {
        self.alerts.evaluate_rules(&self.real_time_metrics)
    }

    /// Add alert rule
    pub fn add_alert_rule(&mut self, rule: AlertRule) {
        self.alerts.add_rule(rule);
    }

    /// Remove alert rule
    pub fn remove_alert_rule(&mut self, rule_id: &str) -> bool {
        self.alerts.remove_rule(rule_id)
    }

    /// Get monitoring statistics
    pub fn get_statistics(&self) -> MonitoringStatistics {
        let mut total_samples = 0;
        let mut total_errors = 0;
        let mut total_uptime = Duration::from_millis(0);

        for agent in self.agents.values() {
            total_samples += agent.statistics.samples_collected;
            total_errors += agent.statistics.error_count;
            total_uptime = total_uptime.max(agent.statistics.uptime);
        }

        MonitoringStatistics {
            total_agents: self.agents.len(),
            total_samples,
            total_errors,
            active_alerts: self.alerts.active_alerts.len(),
            metrics_count: self.real_time_metrics.current.len(),
            uptime: total_uptime,
        }
    }
}

impl MonitoringAgent {
    /// Create a new monitoring agent
    pub fn new(
        agent_id: String,
        agent_type: AgentType,
        metrics: Vec<String>,
        sampling: SamplingConfig,
    ) -> Self {
        Self {
            agent_id,
            agent_type,
            metrics,
            sampling,
            statistics: AgentStatistics::default(),
            running: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Start the monitoring agent.
    ///
    /// Transitions the agent from Stopped → Running.
    /// Returns an error if the agent is already running.
    pub fn start(&mut self) -> Result<(), String> {
        // Use compare-exchange to atomically transition false → true.
        self.running
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .map_err(|_| {
                format!(
                    "Agent '{}' is already running; cannot start again",
                    self.agent_id
                )
            })?;

        // Reset uptime counter on fresh start.
        self.statistics.uptime = Duration::from_millis(0);
        Ok(())
    }

    /// Stop the monitoring agent.
    ///
    /// Transitions the agent from Running → Stopped.
    /// Returns an error if the agent is not currently running.
    pub fn stop(&mut self) -> Result<(), String> {
        self.running
            .compare_exchange(true, false, Ordering::SeqCst, Ordering::SeqCst)
            .map_err(|_| {
                format!(
                    "Agent '{}' is not running; cannot stop",
                    self.agent_id
                )
            })?;

        Ok(())
    }

    /// Returns `true` when the agent is in the Running state.
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }

    /// Collect a sample.
    ///
    /// If the agent is stopped the sample is still collected (caller's responsibility
    /// to gate on `is_running()`), but the error counter is not incremented.
    pub fn collect_sample(&mut self) -> Result<MetricDataPoint, String> {
        self.statistics.samples_collected += 1;

        // Generate a deterministic-looking pseudo-random value from the sample counter
        // so that tests can observe non-zero, non-constant data without introducing
        // an external randomness dependency here.
        let seed = self.statistics.samples_collected;
        let hash = seed
            .wrapping_mul(0x9e37_79b9_7f4a_7c15)
            .wrapping_add(0x6c62_272e_07bb_0142);
        let value = (hash as f64) / (u64::MAX as f64) * 100.0;

        let data_point = MetricDataPoint {
            timestamp: SystemTime::now(),
            value,
            source: self.agent_id.clone(),
            context: HashMap::new(),
        };

        Ok(data_point)
    }

    /// Update sampling configuration
    pub fn update_sampling(&mut self, sampling: SamplingConfig) {
        self.sampling = sampling;
    }

    /// Get agent efficiency score
    pub fn get_efficiency_score(&self) -> f64 {
        if self.statistics.samples_collected == 0 {
            return 0.0;
        }

        let error_rate = self.statistics.error_count as f64 / self.statistics.samples_collected as f64;
        let reliability_score = (1.0 - error_rate).max(0.0);

        // Simple efficiency calculation
        reliability_score
    }
}

impl RealTimeMetrics {
    /// Create new real-time metrics
    pub fn new() -> Self {
        Self {
            current: HashMap::new(),
            history: HashMap::new(),
            trends: HashMap::new(),
            update_frequency: Duration::from_secs(1),
            collecting: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Start metrics collection.
    ///
    /// Sets the update frequency and transitions collection state to active.
    /// Returns an error if collection is already active.
    pub fn start_collection(&mut self, interval: Duration) -> Result<(), String> {
        self.collecting
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .map_err(|_| "Metrics collection is already active".to_string())?;

        self.update_frequency = interval;
        Ok(())
    }

    /// Stop metrics collection.
    ///
    /// Transitions collection state from active to inactive.
    /// Returns an error if collection is not currently active.
    pub fn stop_collection(&mut self) -> Result<(), String> {
        self.collecting
            .compare_exchange(true, false, Ordering::SeqCst, Ordering::SeqCst)
            .map_err(|_| "Metrics collection is not active; cannot stop".to_string())?;

        Ok(())
    }

    /// Returns `true` when collection is active.
    pub fn is_collecting(&self) -> bool {
        self.collecting.load(Ordering::SeqCst)
    }

    /// Update metric value
    pub fn update_metric(&mut self, metric_name: String, value: f64, source: String) {
        let metric_value = MetricValue {
            value,
            timestamp: SystemTime::now(),
            quality: MetricQuality::High,
            source: source.clone(),
        };

        self.current.insert(metric_name.clone(), metric_value);

        // Add to history
        let data_point = MetricDataPoint {
            timestamp: SystemTime::now(),
            value,
            source,
            context: HashMap::new(),
        };

        let history = self.history.entry(metric_name.clone()).or_insert_with(VecDeque::new);
        history.push_back(data_point);

        // Limit history size
        if history.len() > 1000 {
            history.pop_front();
        }

        // Update trend
        self.update_trend(&metric_name);
    }

    /// Update trend analysis for a metric
    fn update_trend(&mut self, metric_name: &str) {
        if let Some(history) = self.history.get(metric_name) {
            if history.len() < 2 {
                return;
            }

            let recent_values: Vec<f64> = history.iter().rev().take(10).map(|dp| dp.value).collect();
            let trend = self.calculate_trend(&recent_values);
            self.trends.insert(metric_name.to_string(), trend);
        }
    }

    /// Calculate trend from values
    fn calculate_trend(&self, values: &[f64]) -> MetricTrend {
        if values.len() < 2 {
            return MetricTrend {
                direction: TrendDirection::Stable,
                velocity: 0.0,
                acceleration: 0.0,
                confidence: 0.0,
            };
        }

        // Simple trend calculation: velocity is the rate of change across the window.
        // `values` is ordered newest-first (collected via `.rev().take(10)`), so
        // `values.last()` is the oldest observation and `values.first()` the newest.
        let oldest = values.last().copied().unwrap_or(0.0);
        let newest = values.first().copied().unwrap_or(0.0);
        let velocity = (newest - oldest) / values.len() as f64;

        let direction = if velocity > 0.1 {
            TrendDirection::Improving
        } else if velocity < -0.1 {
            TrendDirection::Degrading
        } else {
            TrendDirection::Stable
        };

        // Calculate acceleration (rate of change of velocity).
        let acceleration = if values.len() >= 4 {
            // Split into two halves and calculate velocity for each.
            let mid = values.len() / 2;
            let first_half = &values[mid..]; // older half
            let second_half = &values[..mid]; // newer half

            let v1 = if first_half.len() >= 2 {
                (first_half.first().copied().unwrap_or(0.0)
                    - first_half.last().copied().unwrap_or(0.0))
                    / first_half.len() as f64
            } else {
                0.0
            };

            let v2 = if second_half.len() >= 2 {
                (second_half.first().copied().unwrap_or(0.0)
                    - second_half.last().copied().unwrap_or(0.0))
                    / second_half.len() as f64
            } else {
                0.0
            };

            // Acceleration is change in velocity.
            (v2 - v1) / (values.len() as f64 / 2.0)
        } else {
            0.0
        };

        // Calculate confidence based on data quantity and stability
        let confidence = self.calculate_trend_confidence(values, velocity);

        MetricTrend {
            direction,
            velocity,
            acceleration,
            confidence,
        }
    }

    /// Calculate confidence in trend analysis
    fn calculate_trend_confidence(&self, values: &[f64], velocity: f64) -> f64 {
        let mut confidence = 1.0;

        // Factor 1: Data quantity (more points = higher confidence)
        let data_factor = (values.len() as f64 / 10.0).min(1.0);
        confidence *= 0.5 + 0.5 * data_factor;

        // Factor 2: Trend consistency (lower variance relative to trend = higher confidence)
        if values.len() >= 3 {
            // Calculate expected values based on linear trend
            let mean = values.iter().sum::<f64>() / values.len() as f64;
            let variance = values.iter()
                .map(|&v| (v - mean).powi(2))
                .sum::<f64>() / values.len() as f64;

            // Coefficient of variation (normalized variance)
            let cv = if mean.abs() > 1e-10 {
                variance.sqrt() / mean.abs()
            } else {
                variance.sqrt()
            };

            // Lower coefficient of variation = higher confidence
            let consistency_factor = 1.0 / (1.0 + cv);
            confidence *= consistency_factor;
        }

        // Factor 3: Strength of trend (stronger trends = higher confidence)
        let velocity_strength = velocity.abs().min(1.0);
        confidence *= 0.8 + 0.2 * velocity_strength;

        // Ensure confidence is in valid range [0.0, 1.0]
        confidence.clamp(0.0, 1.0)
    }
}

impl PerformanceAlerts {
    /// Create new performance alerts system
    pub fn new() -> Self {
        Self {
            rules: Vec::new(),
            active_alerts: HashMap::new(),
            history: VecDeque::new(),
            config: AlertConfig::default(),
            evaluating: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Start alert evaluation.
    ///
    /// Sets the evaluation interval and transitions the evaluation state to active.
    /// Returns an error if evaluation is already active.
    pub fn start_evaluation(&mut self, interval: Duration) -> Result<(), String> {
        self.evaluating
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .map_err(|_| "Alert evaluation is already active".to_string())?;

        self.config.evaluation_interval = interval;
        Ok(())
    }

    /// Stop alert evaluation.
    ///
    /// Transitions evaluation state from active to inactive.
    /// Returns an error if evaluation is not currently active.
    pub fn stop_evaluation(&mut self) -> Result<(), String> {
        self.evaluating
            .compare_exchange(true, false, Ordering::SeqCst, Ordering::SeqCst)
            .map_err(|_| "Alert evaluation is not active; cannot stop".to_string())?;

        Ok(())
    }

    /// Returns `true` when evaluation is active.
    pub fn is_evaluating(&self) -> bool {
        self.evaluating.load(Ordering::SeqCst)
    }

    /// Evaluate alert rules
    pub fn evaluate_rules(&mut self, metrics: &RealTimeMetrics) -> Result<Vec<PerformanceAlert>, String> {
        if !self.config.enabled {
            return Ok(Vec::new());
        }

        let mut new_alerts = Vec::new();

        for rule in &self.rules {
            if let Some(metric_value) = metrics.current.get(&rule.condition.metric) {
                if self.evaluate_condition(&rule.condition, metric_value.value) {
                    let alert = PerformanceAlert {
                        alert_id: format!("alert_{}", SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap_or_default().as_nanos()),
                        rule_id: rule.rule_id.clone(),
                        timestamp: SystemTime::now(),
                        message: format!("Rule '{}' triggered: {} {} {}",
                            rule.name, rule.condition.metric,
                            format!("{:?}", rule.condition.operator), rule.condition.threshold),
                        severity: rule.severity.clone(),
                        context: HashMap::new(),
                    };

                    self.active_alerts.insert(alert.alert_id.clone(), alert.clone());
                    new_alerts.push(alert);
                }
            }
        }

        Ok(new_alerts)
    }

    /// Add alert rule
    pub fn add_rule(&mut self, rule: AlertRule) {
        self.rules.push(rule);
    }

    /// Remove alert rule
    pub fn remove_rule(&mut self, rule_id: &str) -> bool {
        if let Some(pos) = self.rules.iter().position(|r| r.rule_id == rule_id) {
            self.rules.remove(pos);
            true
        } else {
            false
        }
    }

    /// Acknowledge alert
    pub fn acknowledge_alert(&mut self, alert_id: &str) -> Result<(), String> {
        if let Some(alert) = self.active_alerts.get(alert_id) {
            let record = AlertRecord {
                record_id: format!("record_{}", SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap_or_default().as_nanos()),
                alert_id: alert_id.to_string(),
                timestamp: SystemTime::now(),
                event_type: AlertEventType::Acknowledged,
                details: HashMap::new(),
            };
            self.history.push_back(record);
            Ok(())
        } else {
            Err(format!("Alert {} not found", alert_id))
        }
    }

    /// Resolve alert
    pub fn resolve_alert(&mut self, alert_id: &str) -> Result<(), String> {
        if let Some(_alert) = self.active_alerts.remove(alert_id) {
            let record = AlertRecord {
                record_id: format!("record_{}", SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap_or_default().as_nanos()),
                alert_id: alert_id.to_string(),
                timestamp: SystemTime::now(),
                event_type: AlertEventType::Resolved,
                details: HashMap::new(),
            };
            self.history.push_back(record);
            Ok(())
        } else {
            Err(format!("Alert {} not found", alert_id))
        }
    }

    fn evaluate_condition(&self, condition: &AlertCondition, value: f64) -> bool {
        match condition.operator {
            ConditionOperator::GreaterThan => value > condition.threshold,
            ConditionOperator::LessThan => value < condition.threshold,
            ConditionOperator::Equal => (value - condition.threshold).abs() < f64::EPSILON,
            ConditionOperator::GreaterThanOrEqual => value >= condition.threshold,
            ConditionOperator::LessThanOrEqual => value <= condition.threshold,
            ConditionOperator::NotEqual => (value - condition.threshold).abs() >= f64::EPSILON,
        }
    }
}

/// Monitoring statistics
#[derive(Debug, Clone)]
pub struct MonitoringStatistics {
    /// Total number of agents
    pub total_agents: usize,
    /// Total samples collected
    pub total_samples: u64,
    /// Total errors
    pub total_errors: u64,
    /// Active alerts count
    pub active_alerts: usize,
    /// Metrics count
    pub metrics_count: usize,
    /// System uptime
    pub uptime: Duration,
}

// Default implementations
impl Default for MonitorConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            interval: Duration::from_secs(1),
            retention_period: Duration::from_secs(3600 * 24 * 7), // 1 week
            real_time_updates: true,
            performance_impact_limit: 0.05, // 5%
        }
    }
}

impl Default for AlertConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            evaluation_interval: Duration::from_secs(10),
            max_active_alerts: 100,
            retention_period: Duration::from_secs(3600 * 24 * 30), // 30 days
        }
    }
}

impl Default for AgentStatistics {
    fn default() -> Self {
        Self {
            samples_collected: 0,
            processing_time: Duration::from_millis(0),
            uptime: Duration::from_millis(0),
            error_count: 0,
        }
    }
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            interval: Duration::from_secs(1),
            strategy: SamplingStrategy::Fixed,
            sample_size: 1,
            adaptive_sampling: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_agent(id: &str) -> MonitoringAgent {
        MonitoringAgent::new(
            id.to_string(),
            AgentType::ResourceMonitor,
            vec!["cpu_usage".to_string()],
            SamplingConfig::default(),
        )
    }

    // ─── MonitoringAgent state machine ───────────────────────────────────────

    #[test]
    fn test_agent_starts_in_stopped_state() {
        let agent = make_agent("agent_1");
        assert!(!agent.is_running(), "New agent should be stopped");
    }

    #[test]
    fn test_agent_start_transitions_to_running() {
        let mut agent = make_agent("agent_1");
        agent.start().expect("start should succeed");
        assert!(agent.is_running(), "Agent should be running after start()");
    }

    #[test]
    fn test_agent_stop_transitions_to_stopped() {
        let mut agent = make_agent("agent_1");
        agent.start().expect("start should succeed");
        agent.stop().expect("stop should succeed");
        assert!(!agent.is_running(), "Agent should be stopped after stop()");
    }

    #[test]
    fn test_agent_double_start_returns_error() {
        let mut agent = make_agent("agent_1");
        agent.start().expect("first start should succeed");
        let result = agent.start();
        assert!(result.is_err(), "Double-start should return an error");
        let err = result.unwrap_err();
        assert!(
            err.contains("already running"),
            "Error message should mention 'already running', got: {}",
            err
        );
    }

    #[test]
    fn test_agent_stop_when_stopped_returns_error() {
        let mut agent = make_agent("agent_1");
        let result = agent.stop();
        assert!(result.is_err(), "Stopping a stopped agent should return an error");
        let err = result.unwrap_err();
        assert!(
            err.contains("not running"),
            "Error message should mention 'not running', got: {}",
            err
        );
    }

    #[test]
    fn test_agent_start_stop_start_cycle() {
        let mut agent = make_agent("agent_1");
        agent.start().expect("first start");
        agent.stop().expect("first stop");
        agent.start().expect("second start should succeed after stop");
        assert!(agent.is_running());
    }

    #[test]
    fn test_agent_collect_sample_increments_count() {
        let mut agent = make_agent("agent_1");
        agent.start().expect("start");
        let _s1 = agent.collect_sample().expect("sample 1");
        let _s2 = agent.collect_sample().expect("sample 2");
        assert_eq!(agent.statistics.samples_collected, 2);
    }

    #[test]
    fn test_agent_collect_sample_produces_non_constant_values() {
        let mut agent = make_agent("agent_1");
        agent.start().expect("start");
        let s1 = agent.collect_sample().expect("sample 1");
        let s2 = agent.collect_sample().expect("sample 2");
        // The deterministic hash should produce different values for consecutive calls.
        assert_ne!(
            s1.value, s2.value,
            "Consecutive samples should differ"
        );
    }

    #[test]
    fn test_agent_collect_sample_value_in_expected_range() {
        let mut agent = make_agent("agent_1");
        agent.start().expect("start");
        for _ in 0..10 {
            let sample = agent.collect_sample().expect("sample");
            assert!(
                sample.value >= 0.0 && sample.value <= 100.0,
                "Sample value {} out of [0, 100] range",
                sample.value
            );
        }
    }

    #[test]
    fn test_agent_clone_shares_running_state() {
        let mut agent = make_agent("agent_1");
        let clone = agent.clone();
        agent.start().expect("start");
        // Both should see the running state because Arc<AtomicBool> is shared.
        assert!(clone.is_running(), "Clone should observe same running state");
    }

    // ─── RealTimeMetrics collection state machine ─────────────────────────

    #[test]
    fn test_real_time_metrics_starts_not_collecting() {
        let metrics = RealTimeMetrics::new();
        assert!(!metrics.is_collecting(), "New metrics should not be collecting");
    }

    #[test]
    fn test_real_time_metrics_start_collection() {
        let mut metrics = RealTimeMetrics::new();
        metrics
            .start_collection(Duration::from_secs(1))
            .expect("start_collection should succeed");
        assert!(metrics.is_collecting(), "Should be collecting after start");
        assert_eq!(metrics.update_frequency, Duration::from_secs(1));
    }

    #[test]
    fn test_real_time_metrics_stop_collection() {
        let mut metrics = RealTimeMetrics::new();
        metrics.start_collection(Duration::from_secs(1)).expect("start");
        metrics.stop_collection().expect("stop should succeed");
        assert!(!metrics.is_collecting(), "Should not be collecting after stop");
    }

    #[test]
    fn test_real_time_metrics_double_start_returns_error() {
        let mut metrics = RealTimeMetrics::new();
        metrics.start_collection(Duration::from_secs(1)).expect("first start");
        let result = metrics.start_collection(Duration::from_secs(1));
        assert!(result.is_err(), "Double start_collection should return error");
    }

    #[test]
    fn test_real_time_metrics_stop_when_not_collecting_returns_error() {
        let mut metrics = RealTimeMetrics::new();
        let result = metrics.stop_collection();
        assert!(result.is_err(), "stop_collection when idle should return error");
    }

    // ─── PerformanceAlerts evaluation state machine ───────────────────────

    #[test]
    fn test_alerts_starts_not_evaluating() {
        let alerts = PerformanceAlerts::new();
        assert!(!alerts.is_evaluating(), "New alerts should not be evaluating");
    }

    #[test]
    fn test_alerts_start_evaluation() {
        let mut alerts = PerformanceAlerts::new();
        alerts
            .start_evaluation(Duration::from_secs(10))
            .expect("start_evaluation should succeed");
        assert!(alerts.is_evaluating(), "Should be evaluating after start");
        assert_eq!(alerts.config.evaluation_interval, Duration::from_secs(10));
    }

    #[test]
    fn test_alerts_stop_evaluation() {
        let mut alerts = PerformanceAlerts::new();
        alerts.start_evaluation(Duration::from_secs(10)).expect("start");
        alerts.stop_evaluation().expect("stop should succeed");
        assert!(!alerts.is_evaluating(), "Should not be evaluating after stop");
    }

    #[test]
    fn test_alerts_double_start_returns_error() {
        let mut alerts = PerformanceAlerts::new();
        alerts.start_evaluation(Duration::from_secs(10)).expect("first start");
        let result = alerts.start_evaluation(Duration::from_secs(10));
        assert!(result.is_err(), "Double start_evaluation should return error");
    }

    #[test]
    fn test_alerts_stop_when_not_evaluating_returns_error() {
        let mut alerts = PerformanceAlerts::new();
        let result = alerts.stop_evaluation();
        assert!(result.is_err(), "stop_evaluation when idle should return error");
    }

    // ─── PerformanceMonitor integration ──────────────────────────────────

    #[test]
    fn test_monitor_start_and_stop() {
        let mut monitor = PerformanceMonitor::new();
        let agent = make_agent("mon_agent");
        monitor.add_agent(agent);

        monitor.start_monitoring().expect("start_monitoring should succeed");
        monitor.stop_monitoring().expect("stop_monitoring should succeed");
    }

    #[test]
    fn test_monitor_process_request_collects_metrics() {
        let mut monitor = PerformanceMonitor::new();
        let request = MonitoringRequest {
            request_id: "req_001".to_string(),
            metrics: vec!["cpu_usage".to_string()],
            duration: Duration::from_secs(3),
            sampling: SamplingConfig {
                interval: Duration::from_secs(1),
                strategy: SamplingStrategy::Fixed,
                sample_size: 1,
                adaptive_sampling: false,
            },
            alert_thresholds: {
                let mut m = std::collections::HashMap::new();
                m.insert("cpu_usage".to_string(), 200.0); // High threshold, no alerts expected
                m
            },
        };

        let result = monitor.process_monitoring_request(request).expect("should succeed");
        assert_eq!(result.request_id, "req_001");
        assert!(result.metrics.contains_key("cpu_usage"), "Should have cpu_usage metric");
        assert_eq!(result.summary.total_samples, 3, "Should have 3 samples");
    }

    #[test]
    fn test_monitor_update_and_retrieve_metric() {
        let mut monitor = PerformanceMonitor::new();
        monitor.update_metrics("test_metric".to_string(), 42.5, "test_source".to_string());

        let val = monitor.get_current_metric("test_metric").expect("metric should exist");
        assert!((val.value - 42.5).abs() < f64::EPSILON);
    }
}

