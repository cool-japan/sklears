use crate::distributed_optimization::core_types::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, SystemTime};

/// Real-time capacity metrics
pub struct RealTimeCapacityMetrics {
    pub current_metrics: HashMap<NodeId, InstantaneousMetrics>,
    pub streaming_processors: Vec<StreamingProcessor>,
    pub alert_triggers: Vec<AlertTrigger>,
    pub metric_aggregators: HashMap<String, MetricAggregator>,
}

/// Instantaneous metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstantaneousMetrics {
    pub timestamp: SystemTime,
    pub cpu_usage: f64,
    pub memory_usage: f64,
    pub storage_usage: f64,
    pub network_usage: f64,
    pub gpu_usage: Option<f64>,
    pub custom_metrics: HashMap<String, f64>,
}

/// Streaming processor for real-time data
pub struct StreamingProcessor {
    pub processor_id: String,
    pub input_stream: String,
    pub processing_functions: Vec<ProcessingFunction>,
    pub output_destinations: Vec<String>,
    pub buffer_size: usize,
    pub processing_latency: Duration,
}

/// Processing functions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProcessingFunction {
    Filter(FilterCondition),
    Transform(TransformFunction),
    Aggregate(AggregateFunction),
    Window(WindowFunction),
    Custom(String),
}

/// Filter condition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilterCondition {
    pub field: String,
    pub operator: ComparisonOperator,
    pub value: f64,
}

/// Comparison operators
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComparisonOperator {
    Equal,
    NotEqual,
    GreaterThan,
    LessThan,
    GreaterThanOrEqual,
    LessThanOrEqual,
}

/// Transform function
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TransformFunction {
    Normalize,
    Scale(f64),
    Log,
    Square,
    SquareRoot,
    Custom(String),
}

/// Aggregate function
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AggregateFunction {
    Sum,
    Average,
    Count,
    Min,
    Max,
    Median,
    StandardDeviation,
    Custom(String),
}

/// Window function
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WindowFunction {
    pub window_type: WindowType,
    pub window_size: Duration,
    pub slide_interval: Duration,
}

/// Window types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WindowType {
    Tumbling,
    Sliding,
    Session,
    Count,
    Custom(String),
}

/// Alert trigger
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertTrigger {
    pub trigger_id: String,
    pub condition: AlertCondition,
    pub severity: AlertSeverity,
    pub notification_channels: Vec<String>,
    pub cooldown_period: Duration,
    pub enabled: bool,
}

/// Alert condition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertCondition {
    pub metric_name: String,
    pub threshold: f64,
    pub comparison: ComparisonOperator,
    pub duration: Duration,
    pub node_filter: Option<NodeFilter>,
}

/// Node filter for alerts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeFilter {
    pub include_nodes: Vec<NodeId>,
    pub exclude_nodes: Vec<NodeId>,
    pub node_tags: Vec<String>,
    pub node_types: Vec<String>,
}

/// Alert severity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertSeverity {
    Info,
    Warning,
    Error,
    Critical,
    Emergency,
}

/// Metric aggregator
pub struct MetricAggregator {
    pub aggregator_id: String,
    pub input_metrics: Vec<String>,
    pub aggregation_strategy: AggregationStrategy,
    pub output_metric: String,
    pub update_frequency: Duration,
}

/// Aggregation strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AggregationStrategy {
    Sum,
    Average,
    WeightedAverage(HashMap<String, f64>),
    Maximum,
    Minimum,
    Composite(CompositeStrategy),
    Custom(String),
}

/// Composite aggregation strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompositeStrategy {
    pub strategy_steps: Vec<AggregationStep>,
    pub final_operation: AggregateFunction,
}

/// Aggregation step
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregationStep {
    pub operation: AggregateFunction,
    pub weight: f64,
    pub condition: Option<FilterCondition>,
}

impl RealTimeCapacityMetrics {
    pub fn new() -> Self {
        Self {
            current_metrics: HashMap::new(),
            streaming_processors: Vec::new(),
            alert_triggers: Vec::new(),
            metric_aggregators: HashMap::new(),
        }
    }
}