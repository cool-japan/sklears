//! Enhanced debugging utilities for pipeline inspection and development
//!
//! This module provides comprehensive debugging tools for ML pipelines including:
//! - Step-by-step execution tracking with detailed metadata
//! - Data flow inspection at each pipeline stage
//! - Performance monitoring and bottleneck identification
//! - Interactive debugging with breakpoints and watch expressions
//! - Execution state visualization and reporting
//! - Error context capture with actionable suggestions

use crate::error::{Result, SklearsComposeError};
use scirs2_core::ndarray::{Array2, ArrayView2};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Comprehensive debugging framework for ML pipeline inspection
#[derive(Debug)]
pub struct PipelineDebugger {
    /// Execution tracer for step-by-step tracking
    tracer: Arc<RwLock<ExecutionTracer>>,

    /// Performance profiler for bottleneck identification
    profiler: Arc<RwLock<PerformanceProfiler>>,

    /// Interactive debugger for breakpoints and inspection
    interactive_debugger: Arc<Mutex<InteractiveDebugger>>,

    /// Data inspector for examining transformations
    data_inspector: Arc<RwLock<DataInspector>>,

    /// Error context manager for detailed error reporting
    error_manager: Arc<RwLock<ErrorContextManager>>,

    /// Configuration for debugging behavior
    config: DebuggingConfig,
}

/// Configuration for debugging utilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DebuggingConfig {
    /// Enable step-by-step execution tracking
    pub enable_execution_tracing: bool,

    /// Enable performance profiling
    pub enable_performance_profiling: bool,

    /// Enable data flow inspection
    pub enable_data_inspection: bool,

    /// Enable interactive debugging features
    pub enable_interactive_debugging: bool,

    /// Maximum number of execution steps to store
    pub max_execution_history: usize,

    /// Sample rate for data inspection (0.0 to 1.0)
    pub data_inspection_sample_rate: f64,

    /// Enable verbose logging
    pub verbose_logging: bool,

    /// Export format for debugging reports
    pub export_format: ExportFormat,
}

/// Export formats for debugging reports
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExportFormat {
    /// Json
    Json,
    /// Html
    Html,
    /// Markdown
    Markdown,
    /// Csv
    Csv,
}

/// Step-by-step execution tracker
#[derive(Debug)]
pub struct ExecutionTracer {
    /// History of execution steps
    execution_history: VecDeque<ExecutionStep>,

    /// Current execution context
    current_context: Option<ExecutionContext>,

    /// Execution statistics
    statistics: ExecutionStatistics,

    /// Configuration
    config: TracingConfig,
}

/// Individual execution step information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionStep {
    /// Unique step identifier
    pub step_id: String,

    /// Step name/description
    pub step_name: String,

    /// Step type (Transform, Fit, Predict, etc.)
    pub step_type: StepType,

    /// Timestamp when step started
    pub start_time: SystemTime,

    /// Step execution duration
    pub duration: Option<Duration>,

    /// Input data shape and metadata
    pub input_metadata: DataMetadata,

    /// Output data shape and metadata
    pub output_metadata: Option<DataMetadata>,

    /// Performance metrics for this step
    pub performance_metrics: PerformanceMetrics,

    /// Any warnings or notes
    pub warnings: Vec<String>,

    /// Step status
    pub status: StepStatus,

    /// Error information if step failed
    pub error_info: Option<ErrorInfo>,
}

/// Types of pipeline steps
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StepType {
    /// Transform
    Transform,
    /// Fit
    Fit,
    /// Predict
    Predict,
    /// Validate
    Validate,
    /// Preprocess
    Preprocess,
    /// FeatureEngineering
    FeatureEngineering,
    /// ModelSelection
    ModelSelection,
    /// Ensemble
    Ensemble,
    /// Custom
    Custom(String),
}

/// Step execution status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StepStatus {
    /// Running
    Running,
    /// Completed
    Completed,
    /// Failed
    Failed,
    /// Skipped
    Skipped,
    /// Warning
    Warning,
}

/// Current execution context
#[derive(Debug, Clone)]
pub struct ExecutionContext {
    /// Current pipeline name
    pub pipeline_name: String,

    /// Current step being executed
    pub current_step: String,

    /// Execution start time
    pub execution_start: Instant,

    /// Context variables and metadata
    pub context_variables: HashMap<String, ContextValue>,

    /// Current execution state
    pub execution_state: ExecutionState,
}

/// Execution state information
#[derive(Debug, Clone)]
pub enum ExecutionState {
    /// Initializing
    Initializing,
    /// Running
    Running { current_step: String },
    /// Paused
    Paused { reason: String },
    /// Completed
    Completed,
    /// Failed
    Failed { error: String },
}

/// Context value types
#[derive(Debug, Clone)]
pub enum ContextValue {
    /// String
    String(String),
    /// Number
    Number(f64),
    /// Boolean
    Boolean(bool),
    /// Array
    Array(Vec<f64>),
    /// Metadata
    Metadata(HashMap<String, String>),
}

/// Performance profiler for identifying bottlenecks
#[derive(Debug)]
pub struct PerformanceProfiler {
    /// Performance measurements by step
    step_measurements: HashMap<String, Vec<PerformanceMetrics>>,

    /// Current measurements
    current_measurements: HashMap<String, MeasurementSession>,

    /// Profiling configuration
    config: ProfilingConfig,

    /// Bottleneck analysis results
    bottleneck_analysis: Option<BottleneckAnalysis>,
}

/// Performance metrics for a single step
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Execution time in milliseconds
    pub execution_time_ms: f64,

    /// Memory usage in bytes
    pub memory_usage_bytes: u64,

    /// CPU utilization percentage
    pub cpu_utilization: f64,

    /// Input data size
    pub input_size: usize,

    /// Output data size
    pub output_size: usize,

    /// Cache hits/misses
    pub cache_statistics: CacheStatistics,

    /// Custom metrics
    pub custom_metrics: HashMap<String, f64>,
}

/// Cache usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheStatistics {
    pub hits: u64,
    pub misses: u64,
    pub hit_ratio: f64,
}

/// Active measurement session
#[derive(Debug)]
pub struct MeasurementSession {
    start_time: Instant,
    initial_memory: u64,
    step_name: String,
}

/// Interactive debugger for breakpoints and inspection
#[derive(Debug)]
pub struct InteractiveDebugger {
    /// Breakpoints set by user
    breakpoints: HashMap<String, Breakpoint>,

    /// Watch expressions for monitoring variables
    watch_expressions: Vec<WatchExpression>,

    /// Current debugging session
    current_session: Option<DebugSession>,

    /// Debug commands queue
    command_queue: VecDeque<DebugCommand>,

    /// Configuration
    config: InteractiveConfig,
}

/// Breakpoint definition
#[derive(Debug, Clone)]
pub struct Breakpoint {
    /// Breakpoint ID
    pub id: String,

    /// Step name where breakpoint is set
    pub step_name: String,

    /// Condition for triggering breakpoint
    pub condition: Option<BreakpointCondition>,

    /// Whether breakpoint is enabled
    pub enabled: bool,

    /// Hit count
    pub hit_count: usize,
}

/// Breakpoint trigger conditions
#[derive(Debug, Clone)]
pub enum BreakpointCondition {
    /// Always
    Always,
    /// DataShape
    DataShape { expected_shape: Vec<usize> },
    /// PerformanceThreshold
    PerformanceThreshold { max_time_ms: f64 },
    /// ErrorOccurred
    ErrorOccurred,
    /// Custom
    Custom(String),
}

/// Watch expression for monitoring variables
#[derive(Debug, Clone)]
pub struct WatchExpression {
    /// Expression ID
    pub id: String,

    /// Expression to evaluate
    pub expression: String,

    /// Current value
    pub current_value: Option<String>,

    /// Value history
    pub value_history: Vec<(SystemTime, String)>,
}

/// Active debugging session
#[derive(Debug)]
pub struct DebugSession {
    /// Session ID
    session_id: String,

    /// Current step being debugged
    current_step: String,

    /// Session variables
    session_variables: HashMap<String, ContextValue>,

    /// Debug state
    debug_state: DebugState,
}

/// Debug session state
#[derive(Debug, Clone)]
pub enum DebugState {
    /// Running
    Running,
    /// Paused
    Paused { reason: String },
    /// StepOver
    StepOver,
    /// StepInto
    StepInto,
    /// Continue
    Continue,
}

/// Debug commands
#[derive(Debug, Clone)]
pub enum DebugCommand {
    /// Continue
    Continue,
    /// StepOver
    StepOver,
    /// StepInto
    StepInto,
    /// StepOut
    StepOut,
    /// SetBreakpoint
    SetBreakpoint {
        step_name: String,
        condition: Option<BreakpointCondition>,
    },
    /// RemoveBreakpoint
    RemoveBreakpoint { breakpoint_id: String },
    /// AddWatch
    AddWatch { expression: String },
    /// RemoveWatch
    RemoveWatch { watch_id: String },
    /// InspectVariable
    InspectVariable { variable_name: String },
    /// EvaluateExpression
    EvaluateExpression { expression: String },
}

/// Data inspector for examining transformations
#[derive(Debug)]
pub struct DataInspector {
    /// Data snapshots at each step
    data_snapshots: HashMap<String, DataSnapshot>,

    /// Data flow analysis
    data_flow_analysis: Option<DataFlowAnalysis>,

    /// Transformation summaries
    transformation_summaries: Vec<TransformationSummary>,

    /// Configuration
    config: InspectionConfig,
}

/// Data snapshot at a specific step
#[derive(Debug, Clone)]
pub struct DataSnapshot {
    /// Step name
    pub step_name: String,

    /// Data shape
    pub shape: Vec<usize>,

    /// Data type information
    pub dtype: String,

    /// Statistical summary
    pub statistics: DataStatistics,

    /// Sample data (if enabled)
    pub sample_data: Option<Vec<f64>>,

    /// Timestamp
    pub timestamp: SystemTime,
}

/// Statistical summary of data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataStatistics {
    /// Number of samples
    pub n_samples: usize,

    /// Number of features
    pub n_features: usize,

    /// Mean values per feature
    pub means: Vec<f64>,

    /// Standard deviations per feature
    pub stds: Vec<f64>,

    /// Minimum values per feature
    pub mins: Vec<f64>,

    /// Maximum values per feature
    pub maxs: Vec<f64>,

    /// Missing value count per feature
    pub missing_counts: Vec<usize>,

    /// Data quality score
    pub quality_score: f64,
}

/// Data flow analysis results
#[derive(Debug, Clone)]
pub struct DataFlowAnalysis {
    /// Data lineage information
    pub lineage: Vec<DataLineageNode>,

    /// Transformation graph
    pub transformation_graph: TransformationGraph,

    /// Quality metrics throughout pipeline
    pub quality_metrics: Vec<QualityMetric>,
}

/// Data lineage node
#[derive(Debug, Clone)]
pub struct DataLineageNode {
    /// Node ID
    pub node_id: String,

    /// Step name
    pub step_name: String,

    /// Input dependencies
    pub inputs: Vec<String>,

    /// Output products
    pub outputs: Vec<String>,

    /// Transformation description
    pub transformation: String,
}

/// Error context manager for detailed error reporting
#[derive(Debug)]
pub struct ErrorContextManager {
    /// Error history
    error_history: Vec<ErrorContext>,

    /// Error patterns and suggestions
    error_patterns: HashMap<String, Vec<ErrorSuggestion>>,

    /// Configuration
    config: ErrorConfig,
}

/// Detailed error context
#[derive(Debug, Clone)]
pub struct ErrorContext {
    /// Error ID
    pub error_id: String,

    /// Error message
    pub error_message: String,

    /// Error type
    pub error_type: String,

    /// Step where error occurred
    pub step_name: String,

    /// Input data context
    pub input_context: DataMetadata,

    /// Execution state when error occurred
    pub execution_state: String,

    /// Stack trace
    pub stack_trace: Vec<String>,

    /// Suggested fixes
    pub suggestions: Vec<ErrorSuggestion>,

    /// Timestamp
    pub timestamp: SystemTime,
}

/// Error suggestion with actionable advice
#[derive(Debug, Clone)]
pub struct ErrorSuggestion {
    /// Suggestion description
    pub description: String,

    /// Confidence level (0.0 to 1.0)
    pub confidence: f64,

    /// Suggested code fix
    pub code_fix: Option<String>,

    /// Documentation reference
    pub documentation_link: Option<String>,
}

/// Data metadata information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataMetadata {
    /// Data shape
    pub shape: Vec<usize>,

    /// Data type
    pub dtype: String,

    /// Memory usage in bytes
    pub memory_usage: u64,

    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Error information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorInfo {
    /// Error message
    pub message: String,

    /// Error type
    pub error_type: String,

    /// Stack trace
    pub stack_trace: Vec<String>,
}

/// Configuration structs
#[derive(Debug, Clone)]
pub struct TracingConfig {
    pub max_history_size: usize,
    pub enable_metadata_collection: bool,
    pub enable_performance_tracking: bool,
}

#[derive(Debug, Clone)]
pub struct ProfilingConfig {
    pub enable_memory_tracking: bool,
    pub enable_cpu_monitoring: bool,
    pub measurement_interval_ms: u64,
}

#[derive(Debug, Clone)]
pub struct InteractiveConfig {
    pub enable_auto_breakpoints: bool,
    pub max_watch_expressions: usize,
    pub command_timeout_ms: u64,
}

#[derive(Debug, Clone)]
pub struct InspectionConfig {
    pub sample_size: usize,
    pub enable_statistical_analysis: bool,
    pub enable_quality_metrics: bool,
}

#[derive(Debug, Clone)]
pub struct ErrorConfig {
    pub max_error_history: usize,
    pub enable_pattern_recognition: bool,
    pub enable_auto_suggestions: bool,
}

/// Additional supporting types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionStatistics {
    pub total_steps: usize,
    pub successful_steps: usize,
    pub failed_steps: usize,
    pub total_execution_time: Duration,
    pub average_step_time: Duration,
}

#[derive(Debug, Clone)]
pub struct BottleneckAnalysis {
    pub slowest_steps: Vec<(String, Duration)>,
    pub memory_intensive_steps: Vec<(String, u64)>,
    pub optimization_suggestions: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct TransformationSummary {
    pub step_name: String,
    pub input_shape: Vec<usize>,
    pub output_shape: Vec<usize>,
    pub transformation_type: String,
    pub quality_impact: f64,
}

#[derive(Debug, Clone)]
pub struct TransformationGraph {
    pub nodes: Vec<TransformationNode>,
    pub edges: Vec<TransformationEdge>,
}

#[derive(Debug, Clone)]
pub struct TransformationNode {
    pub id: String,
    pub step_name: String,
    pub node_type: String,
}

#[derive(Debug, Clone)]
pub struct TransformationEdge {
    pub from: String,
    pub to: String,
    pub data_flow: String,
}

#[derive(Debug, Clone)]
pub struct QualityMetric {
    pub step_name: String,
    pub metric_name: String,
    pub value: f64,
    pub threshold: Option<f64>,
}

impl Default for DebuggingConfig {
    fn default() -> Self {
        Self {
            enable_execution_tracing: true,
            enable_performance_profiling: true,
            enable_data_inspection: true,
            enable_interactive_debugging: false,
            max_execution_history: 1000,
            data_inspection_sample_rate: 0.1,
            verbose_logging: false,
            export_format: ExportFormat::Json,
        }
    }
}

impl Default for PipelineDebugger {
    fn default() -> Self {
        Self::new()
    }
}

impl PipelineDebugger {
    /// Create a new pipeline debugger with default configuration
    #[must_use]
    pub fn new() -> Self {
        Self::with_config(DebuggingConfig::default())
    }

    /// Create a new pipeline debugger with custom configuration
    #[must_use]
    pub fn with_config(config: DebuggingConfig) -> Self {
        let tracer_config = TracingConfig {
            max_history_size: config.max_execution_history,
            enable_metadata_collection: config.enable_data_inspection,
            enable_performance_tracking: config.enable_performance_profiling,
        };

        let profiler_config = ProfilingConfig {
            enable_memory_tracking: true,
            enable_cpu_monitoring: true,
            measurement_interval_ms: 100,
        };

        let interactive_config = InteractiveConfig {
            enable_auto_breakpoints: false,
            max_watch_expressions: 50,
            command_timeout_ms: 5000,
        };

        let inspection_config = InspectionConfig {
            sample_size: (config.data_inspection_sample_rate * 1000.0) as usize,
            enable_statistical_analysis: true,
            enable_quality_metrics: true,
        };

        let error_config = ErrorConfig {
            max_error_history: 100,
            enable_pattern_recognition: true,
            enable_auto_suggestions: true,
        };

        Self {
            tracer: Arc::new(RwLock::new(ExecutionTracer::new(tracer_config))),
            profiler: Arc::new(RwLock::new(PerformanceProfiler::new(profiler_config))),
            interactive_debugger: Arc::new(Mutex::new(InteractiveDebugger::new(
                interactive_config,
            ))),
            data_inspector: Arc::new(RwLock::new(DataInspector::new(inspection_config))),
            error_manager: Arc::new(RwLock::new(ErrorContextManager::new(error_config))),
            config,
        }
    }

    /// Start debugging a pipeline execution
    pub fn start_execution(&self, pipeline_name: &str) -> Result<String> {
        if self.config.enable_execution_tracing {
            let mut tracer = self.tracer.write().unwrap();
            let context = ExecutionContext {
                pipeline_name: pipeline_name.to_string(),
                current_step: "initialization".to_string(),
                execution_start: Instant::now(),
                context_variables: HashMap::new(),
                execution_state: ExecutionState::Initializing,
            };
            tracer.current_context = Some(context);
        }

        Ok(format!(
            "debug_session_{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs()
        ))
    }

    /// Record the start of a pipeline step
    pub fn step_start(
        &self,
        step_name: &str,
        step_type: StepType,
        input_data: ArrayView2<f64>,
    ) -> Result<String> {
        let step_id = format!(
            "{}_{}",
            step_name,
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_millis()
        );

        if self.config.enable_execution_tracing {
            let mut tracer = self.tracer.write().unwrap();
            let input_metadata = self.create_data_metadata(&input_data);

            let step = ExecutionStep {
                step_id: step_id.clone(),
                step_name: step_name.to_string(),
                step_type,
                start_time: SystemTime::now(),
                duration: None,
                input_metadata,
                output_metadata: None,
                performance_metrics: PerformanceMetrics::default(),
                warnings: Vec::new(),
                status: StepStatus::Running,
                error_info: None,
            };

            tracer.add_step(step);
        }

        if self.config.enable_performance_profiling {
            let mut profiler = self.profiler.write().unwrap();
            profiler.start_measurement(&step_id, step_name);
        }

        if self.config.enable_data_inspection {
            let mut inspector = self.data_inspector.write().unwrap();
            inspector.capture_data_snapshot(step_name, &input_data)?;
        }

        // Check for breakpoints
        if self.config.enable_interactive_debugging {
            let debugger = self.interactive_debugger.lock().unwrap();
            if debugger.should_break(step_name) {
                // Breakpoint hit - could trigger user interaction
                println!("Breakpoint hit at step: {step_name}");
            }
        }

        Ok(step_id)
    }

    /// Record the completion of a pipeline step
    pub fn step_complete(&self, step_id: &str, output_data: ArrayView2<f64>) -> Result<()> {
        if self.config.enable_execution_tracing {
            let mut tracer = self.tracer.write().unwrap();
            let output_metadata = self.create_data_metadata(&output_data);
            tracer.complete_step(step_id, output_metadata)?;
        }

        if self.config.enable_performance_profiling {
            let mut profiler = self.profiler.write().unwrap();
            if let Some(metrics) = profiler.end_measurement(step_id) {
                // Store performance metrics
                if let Ok(mut tracer) = self.tracer.write() {
                    tracer.update_step_metrics(step_id, metrics)?;
                }
            }
        }

        Ok(())
    }

    /// Record an error during pipeline execution
    pub fn step_error(&self, step_id: &str, error: &SklearsComposeError) -> Result<()> {
        if self.config.enable_execution_tracing {
            let mut tracer = self.tracer.write().unwrap();
            let error_info = ErrorInfo {
                message: error.to_string(),
                error_type: format!("{error:?}"),
                stack_trace: vec![], // Could be enhanced with actual stack trace
            };
            tracer.fail_step(step_id, error_info)?;
        }

        // Generate error context and suggestions
        let mut error_manager = self.error_manager.write().unwrap();
        error_manager.record_error(step_id, error)?;

        Ok(())
    }

    /// Generate a comprehensive debugging report
    pub fn generate_report(&self) -> Result<DebugReport> {
        let tracer = self.tracer.read().unwrap();
        let profiler = self.profiler.read().unwrap();
        let inspector = self.data_inspector.read().unwrap();
        let error_manager = self.error_manager.read().unwrap();

        let report = DebugReport {
            execution_summary: tracer.get_execution_summary(),
            performance_analysis: profiler.get_analysis(),
            data_flow_summary: inspector.get_data_flow_summary(),
            error_summary: error_manager.get_error_summary(),
            recommendations: self.generate_recommendations()?,
            timestamp: SystemTime::now(),
        };

        Ok(report)
    }

    /// Export debugging information to specified format
    pub fn export_debug_info(&self, format: ExportFormat) -> Result<String> {
        let report = self.generate_report()?;

        match format {
            ExportFormat::Json => Ok(serde_json::to_string_pretty(&report).unwrap()),
            ExportFormat::Html => Ok(self.generate_html_report(&report)),
            ExportFormat::Markdown => Ok(self.generate_markdown_report(&report)),
            ExportFormat::Csv => Ok(self.generate_csv_report(&report)),
        }
    }

    // Private helper methods
    fn create_data_metadata(&self, data: &ArrayView2<f64>) -> DataMetadata {
        /// DataMetadata
        DataMetadata {
            shape: data.shape().to_vec(),
            dtype: "f64".to_string(),
            memory_usage: (data.len() * std::mem::size_of::<f64>()) as u64,
            metadata: HashMap::new(),
        }
    }

    fn generate_recommendations(&self) -> Result<Vec<String>> {
        let mut recommendations = Vec::new();

        // Analyze performance bottlenecks
        if let Ok(profiler) = self.profiler.read() {
            if let Some(analysis) = &profiler.bottleneck_analysis {
                recommendations.extend(analysis.optimization_suggestions.clone());
            }
        }

        // Analyze error patterns
        if let Ok(error_manager) = self.error_manager.read() {
            // Add error-based recommendations
            recommendations
                .push("Consider adding input validation to prevent common errors".to_string());
        }

        Ok(recommendations)
    }

    fn generate_html_report(&self, report: &DebugReport) -> String {
        format!(
            r#"<!DOCTYPE html>
<html>
<head>
    <title>Pipeline Debug Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .section {{ margin-bottom: 30px; }}
        .metric {{ margin: 5px 0; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <h1>Pipeline Debug Report</h1>
    <div class="section">
        <h2>Execution Summary</h2>
        <p>Total Steps: {}</p>
        <p>Successful Steps: {}</p>
        <p>Failed Steps: {}</p>
    </div>
    <div class="section">
        <h2>Recommendations</h2>
        <ul>{}</ul>
    </div>
</body>
</html>"#,
            report.execution_summary.total_steps,
            report.execution_summary.successful_steps,
            report.execution_summary.failed_steps,
            report
                .recommendations
                .iter()
                .map(|r| format!("<li>{r}</li>"))
                .collect::<String>()
        )
    }

    fn generate_markdown_report(&self, report: &DebugReport) -> String {
        format!(
            r"# Pipeline Debug Report

## Execution Summary
- Total Steps: {}
- Successful Steps: {}
- Failed Steps: {}

## Performance Analysis
- Average Step Time: {:?}
- Total Execution Time: {:?}

## Recommendations
{}
",
            report.execution_summary.total_steps,
            report.execution_summary.successful_steps,
            report.execution_summary.failed_steps,
            report.execution_summary.average_step_time,
            report.execution_summary.total_execution_time,
            report
                .recommendations
                .iter()
                .map(|r| format!("- {r}"))
                .collect::<Vec<_>>()
                .join("\n")
        )
    }

    fn generate_csv_report(&self, _report: &DebugReport) -> String {
        // Simplified CSV export - could be enhanced with step-by-step data
        "step_name,duration_ms,status,memory_usage\n".to_string()
    }
}

/// Comprehensive debug report
#[derive(Debug, Serialize, Deserialize)]
pub struct DebugReport {
    pub execution_summary: ExecutionStatistics,
    pub performance_analysis: String, // Could be more structured
    pub data_flow_summary: String,    // Could be more structured
    pub error_summary: String,        // Could be more structured
    pub recommendations: Vec<String>,
    pub timestamp: SystemTime,
}

// Implementation blocks for supporting components

impl ExecutionTracer {
    fn new(config: TracingConfig) -> Self {
        Self {
            execution_history: VecDeque::with_capacity(config.max_history_size),
            current_context: None,
            statistics: ExecutionStatistics {
                total_steps: 0,
                successful_steps: 0,
                failed_steps: 0,
                total_execution_time: Duration::from_secs(0),
                average_step_time: Duration::from_secs(0),
            },
            config,
        }
    }

    fn add_step(&mut self, step: ExecutionStep) {
        if self.execution_history.len() >= self.config.max_history_size {
            self.execution_history.pop_front();
        }
        self.execution_history.push_back(step);
        self.statistics.total_steps += 1;
    }

    fn complete_step(&mut self, step_id: &str, output_metadata: DataMetadata) -> Result<()> {
        if let Some(step) = self
            .execution_history
            .iter_mut()
            .find(|s| s.step_id == step_id)
        {
            step.status = StepStatus::Completed;
            step.output_metadata = Some(output_metadata);
            step.duration = Some(step.start_time.elapsed().unwrap_or(Duration::from_secs(0)));
            self.statistics.successful_steps += 1;
        }
        Ok(())
    }

    fn fail_step(&mut self, step_id: &str, error_info: ErrorInfo) -> Result<()> {
        if let Some(step) = self
            .execution_history
            .iter_mut()
            .find(|s| s.step_id == step_id)
        {
            step.status = StepStatus::Failed;
            step.error_info = Some(error_info);
            step.duration = Some(step.start_time.elapsed().unwrap_or(Duration::from_secs(0)));
            self.statistics.failed_steps += 1;
        }
        Ok(())
    }

    fn update_step_metrics(&mut self, step_id: &str, metrics: PerformanceMetrics) -> Result<()> {
        if let Some(step) = self
            .execution_history
            .iter_mut()
            .find(|s| s.step_id == step_id)
        {
            step.performance_metrics = metrics;
        }
        Ok(())
    }

    fn get_execution_summary(&self) -> ExecutionStatistics {
        self.statistics.clone()
    }
}

impl PerformanceProfiler {
    fn new(config: ProfilingConfig) -> Self {
        Self {
            step_measurements: HashMap::new(),
            current_measurements: HashMap::new(),
            config,
            bottleneck_analysis: None,
        }
    }

    fn start_measurement(&mut self, step_id: &str, step_name: &str) {
        let session = MeasurementSession {
            start_time: Instant::now(),
            initial_memory: 0, // Could be implemented with actual memory tracking
            step_name: step_name.to_string(),
        };
        self.current_measurements
            .insert(step_id.to_string(), session);
    }

    fn end_measurement(&mut self, step_id: &str) -> Option<PerformanceMetrics> {
        if let Some(session) = self.current_measurements.remove(step_id) {
            let metrics = PerformanceMetrics {
                execution_time_ms: session.start_time.elapsed().as_millis() as f64,
                memory_usage_bytes: 0, // Could be implemented
                cpu_utilization: 0.0,  // Could be implemented
                input_size: 0,         // Would need to be passed in
                output_size: 0,        // Would need to be passed in
                cache_statistics: CacheStatistics {
                    hits: 0,
                    misses: 0,
                    hit_ratio: 0.0,
                },
                custom_metrics: HashMap::new(),
            };

            self.step_measurements
                .entry(session.step_name)
                .or_default()
                .push(metrics.clone());

            Some(metrics)
        } else {
            None
        }
    }

    fn get_analysis(&self) -> String {
        // Simplified analysis - could be much more comprehensive
        format!(
            "Performance analysis for {} unique steps",
            self.step_measurements.len()
        )
    }
}

impl InteractiveDebugger {
    fn new(config: InteractiveConfig) -> Self {
        Self {
            breakpoints: HashMap::new(),
            watch_expressions: Vec::new(),
            current_session: None,
            command_queue: VecDeque::new(),
            config,
        }
    }

    fn should_break(&self, step_name: &str) -> bool {
        self.breakpoints
            .values()
            .any(|bp| bp.step_name == step_name && bp.enabled)
    }
}

impl DataInspector {
    fn new(config: InspectionConfig) -> Self {
        Self {
            data_snapshots: HashMap::new(),
            data_flow_analysis: None,
            transformation_summaries: Vec::new(),
            config,
        }
    }

    fn capture_data_snapshot(&mut self, step_name: &str, data: &ArrayView2<f64>) -> Result<()> {
        let statistics = self.compute_statistics(data);

        let snapshot = DataSnapshot {
            step_name: step_name.to_string(),
            shape: data.shape().to_vec(),
            dtype: "f64".to_string(),
            statistics,
            sample_data: None, // Could sample data based on configuration
            timestamp: SystemTime::now(),
        };

        self.data_snapshots.insert(step_name.to_string(), snapshot);
        Ok(())
    }

    fn compute_statistics(&self, data: &ArrayView2<f64>) -> DataStatistics {
        let (n_samples, n_features) = (data.nrows(), data.ncols());

        // Compute basic statistics
        let mut means = Vec::with_capacity(n_features);
        let mut stds = Vec::with_capacity(n_features);
        let mut mins = Vec::with_capacity(n_features);
        let mut maxs = Vec::with_capacity(n_features);
        let mut missing_counts = Vec::with_capacity(n_features);

        for col in 0..n_features {
            let column_data: Vec<f64> = data.column(col).to_vec();

            let mean = column_data.iter().sum::<f64>() / n_samples as f64;
            let variance =
                column_data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n_samples as f64;
            let std = variance.sqrt();
            let min = column_data.iter().copied().fold(f64::INFINITY, f64::min);
            let max = column_data
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);
            let missing = column_data.iter().filter(|x| x.is_nan()).count();

            means.push(mean);
            stds.push(std);
            mins.push(min);
            maxs.push(max);
            missing_counts.push(missing);
        }

        // Compute quality score (simplified)
        let total_missing = missing_counts.iter().sum::<usize>();
        let quality_score = 1.0 - (total_missing as f64 / (n_samples * n_features) as f64);

        /// DataStatistics
        DataStatistics {
            n_samples,
            n_features,
            means,
            stds,
            mins,
            maxs,
            missing_counts,
            quality_score,
        }
    }

    fn get_data_flow_summary(&self) -> String {
        format!(
            "Data flow analysis for {} snapshots",
            self.data_snapshots.len()
        )
    }
}

impl ErrorContextManager {
    fn new(config: ErrorConfig) -> Self {
        Self {
            error_history: Vec::new(),
            error_patterns: HashMap::new(),
            config,
        }
    }

    fn record_error(&mut self, step_id: &str, error: &SklearsComposeError) -> Result<()> {
        let error_context = ErrorContext {
            error_id: format!(
                "err_{}_{}",
                step_id,
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_millis()
            ),
            error_message: error.to_string(),
            error_type: format!("{error:?}"),
            step_name: step_id.to_string(),
            input_context: DataMetadata {
                shape: vec![],
                dtype: "unknown".to_string(),
                memory_usage: 0,
                metadata: HashMap::new(),
            },
            execution_state: "error".to_string(),
            stack_trace: vec![],
            suggestions: self.generate_suggestions(error),
            timestamp: SystemTime::now(),
        };

        if self.error_history.len() >= self.config.max_error_history {
            self.error_history.remove(0);
        }
        self.error_history.push(error_context);

        Ok(())
    }

    fn generate_suggestions(&self, error: &SklearsComposeError) -> Vec<ErrorSuggestion> {
        let mut suggestions = Vec::new();

        // Pattern-based suggestions
        let error_str = error.to_string();
        if error_str.contains("shape") {
            suggestions.push(ErrorSuggestion {
                description: "Check input data shape compatibility with the model".to_string(),
                confidence: 0.8,
                code_fix: Some(
                    "Verify that input dimensions match expected model input".to_string(),
                ),
                documentation_link: Some(
                    "https://docs.rs/sklears-compose/pipeline-shapes".to_string(),
                ),
            });
        }

        if error_str.contains("memory") {
            suggestions.push(ErrorSuggestion {
                description: "Consider reducing batch size or using streaming processing"
                    .to_string(),
                confidence: 0.7,
                code_fix: Some(
                    "Use smaller batches or enable memory-efficient processing".to_string(),
                ),
                documentation_link: Some(
                    "https://docs.rs/sklears-compose/memory-optimization".to_string(),
                ),
            });
        }

        suggestions
    }

    fn get_error_summary(&self) -> String {
        format!(
            "Recorded {} errors with {} unique patterns",
            self.error_history.len(),
            self.error_patterns.len()
        )
    }
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            execution_time_ms: 0.0,
            memory_usage_bytes: 0,
            cpu_utilization: 0.0,
            input_size: 0,
            output_size: 0,
            cache_statistics: CacheStatistics {
                hits: 0,
                misses: 0,
                hit_ratio: 0.0,
            },
            custom_metrics: HashMap::new(),
        }
    }
}

impl fmt::Display for StepType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StepType::Transform => write!(f, "Transform"),
            StepType::Fit => write!(f, "Fit"),
            StepType::Predict => write!(f, "Predict"),
            StepType::Validate => write!(f, "Validate"),
            StepType::Preprocess => write!(f, "Preprocess"),
            StepType::FeatureEngineering => write!(f, "FeatureEngineering"),
            StepType::ModelSelection => write!(f, "ModelSelection"),
            StepType::Ensemble => write!(f, "Ensemble"),
            StepType::Custom(name) => write!(f, "Custom({name})"),
        }
    }
}

// Example usage and convenience methods
impl PipelineDebugger {
    /// Create a debugger optimized for development
    #[must_use]
    pub fn for_development() -> Self {
        let config = DebuggingConfig {
            enable_execution_tracing: true,
            enable_performance_profiling: true,
            enable_data_inspection: true,
            enable_interactive_debugging: true,
            max_execution_history: 500,
            data_inspection_sample_rate: 0.2,
            verbose_logging: true,
            export_format: ExportFormat::Html,
        };
        Self::with_config(config)
    }

    /// Create a debugger optimized for production monitoring
    #[must_use]
    pub fn for_production() -> Self {
        let config = DebuggingConfig {
            enable_execution_tracing: true,
            enable_performance_profiling: true,
            enable_data_inspection: false,
            enable_interactive_debugging: false,
            max_execution_history: 100,
            data_inspection_sample_rate: 0.01,
            verbose_logging: false,
            export_format: ExportFormat::Json,
        };
        Self::with_config(config)
    }

    /// Quick debug of a single transformation step
    pub fn debug_transform<F>(
        &self,
        step_name: &str,
        input: ArrayView2<f64>,
        transform_fn: F,
    ) -> Result<Array2<f64>>
    where
        F: FnOnce(ArrayView2<f64>) -> Result<Array2<f64>>,
    {
        let step_id = self.step_start(step_name, StepType::Transform, input)?;

        match transform_fn(input) {
            Ok(output) => {
                self.step_complete(&step_id, output.view())?;
                Ok(output)
            }
            Err(e) => {
                self.step_error(&step_id, &e)?;
                Err(e)
            }
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_debugger_creation() {
        let debugger = PipelineDebugger::new();
        assert!(debugger.config.enable_execution_tracing);
    }

    #[test]
    fn test_step_tracking() {
        let debugger = PipelineDebugger::new();
        let data = Array2::zeros((10, 5));

        let session = debugger.start_execution("test_pipeline").unwrap();
        let step_id = debugger
            .step_start("test_step", StepType::Transform, data.view())
            .unwrap();
        let result = debugger.step_complete(&step_id, data.view());

        assert!(result.is_ok());
        assert!(!session.is_empty());
    }

    #[test]
    fn test_development_debugger() {
        let debugger = PipelineDebugger::for_development();
        assert!(debugger.config.enable_interactive_debugging);
        assert!(debugger.config.verbose_logging);
    }

    #[test]
    fn test_production_debugger() {
        let debugger = PipelineDebugger::for_production();
        assert!(!debugger.config.enable_interactive_debugging);
        assert!(!debugger.config.verbose_logging);
    }

    #[test]
    fn test_debug_transform() {
        let debugger = PipelineDebugger::new();
        let input = Array2::ones((5, 3));

        let result =
            debugger.debug_transform("scale", input.view(), |data| Ok(data.to_owned() * 2.0));

        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output[[0, 0]], 2.0);
    }

    #[test]
    fn test_error_handling() {
        let debugger = PipelineDebugger::new();
        let input = Array2::ones((5, 3));

        let result = debugger.debug_transform("failing_step", input.view(), |_| {
            Err(SklearsComposeError::InvalidConfiguration(
                "Test error".to_string(),
            ))
        });

        assert!(result.is_err());
    }

    #[test]
    fn test_report_generation() {
        let debugger = PipelineDebugger::new();
        let report = debugger.generate_report();
        assert!(report.is_ok());
    }

    #[test]
    fn test_export_formats() {
        let debugger = PipelineDebugger::new();

        let json_export = debugger.export_debug_info(ExportFormat::Json);
        assert!(json_export.is_ok());

        let html_export = debugger.export_debug_info(ExportFormat::Html);
        assert!(html_export.is_ok());

        let markdown_export = debugger.export_debug_info(ExportFormat::Markdown);
        assert!(markdown_export.is_ok());
    }

    #[test]
    fn test_performance_metrics() {
        let metrics = PerformanceMetrics::default();
        assert_eq!(metrics.execution_time_ms, 0.0);
        assert_eq!(metrics.memory_usage_bytes, 0);
    }

    #[test]
    fn test_data_statistics() {
        let debugger = PipelineDebugger::new();
        let data = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

        let inspector = debugger.data_inspector.read().unwrap();
        let stats = inspector.compute_statistics(&data.view());

        assert_eq!(stats.n_samples, 3);
        assert_eq!(stats.n_features, 2);
        assert_eq!(stats.means.len(), 2);
    }

    #[test]
    fn test_step_types() {
        assert_eq!(StepType::Transform.to_string(), "Transform");
        assert_eq!(
            StepType::Custom("test".to_string()).to_string(),
            "Custom(test)"
        );
    }
}
