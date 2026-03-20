//! Diagnostic context for debugging, profiling, and observability
//!
//! This module provides comprehensive diagnostic capabilities including
//! performance monitoring, distributed tracing, error tracking, and
//! system observability.

use std::{
    collections::{HashMap, HashSet, VecDeque, BTreeMap},
    sync::{Arc, RwLock, Mutex, atomic::{AtomicU64, AtomicUsize, Ordering}},
    time::{Duration, SystemTime, Instant},
    fmt::{Debug, Display},
    thread,
};
use serde::{Serialize, Deserialize};
use uuid::Uuid;

use crate::context_core::{
    ExecutionContextTrait, ContextType, ContextState, ContextError, ContextResult,
    ContextMetadata, ContextEvent,
};

/// Diagnostic context for monitoring and observability
#[derive(Debug)]
pub struct DiagnosticContext {
    /// Context identifier
    pub id: String,
    /// Diagnostic state
    pub state: Arc<RwLock<DiagnosticState>>,
    /// Performance profiler
    pub profiler: Arc<Mutex<PerformanceProfiler>>,
    /// Distributed tracer
    pub tracer: Arc<RwLock<DistributedTracer>>,
    /// Metrics collector
    pub metrics_collector: Arc<Mutex<MetricsCollector>>,
    /// Log manager
    pub log_manager: Arc<RwLock<LogManager>>,
    /// Error tracker
    pub error_tracker: Arc<Mutex<ErrorTracker>>,
    /// Health monitor
    pub health_monitor: Arc<Mutex<HealthMonitor>>,
    /// Alerting system
    pub alerting_system: Arc<Mutex<AlertingSystem>>,
    /// Diagnostic tools
    pub diagnostic_tools: Arc<RwLock<DiagnosticTools>>,
    /// Configuration
    pub config: Arc<RwLock<DiagnosticConfig>>,
    /// Created timestamp
    pub created_at: SystemTime,
}

/// Diagnostic context states
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DiagnosticState {
    /// Diagnostic context is initializing
    Initializing,
    /// Diagnostic context is active
    Active,
    /// Diagnostic context is in debug mode
    Debug,
    /// Diagnostic context is profiling
    Profiling,
    /// Diagnostic context is disabled
    Disabled,
    /// Diagnostic context is in maintenance mode
    Maintenance,
}

impl Display for DiagnosticState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DiagnosticState::Initializing => write!(f, "initializing"),
            DiagnosticState::Active => write!(f, "active"),
            DiagnosticState::Debug => write!(f, "debug"),
            DiagnosticState::Profiling => write!(f, "profiling"),
            DiagnosticState::Disabled => write!(f, "disabled"),
            DiagnosticState::Maintenance => write!(f, "maintenance"),
        }
    }
}

/// Diagnostic configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiagnosticConfig {
    /// Enable performance profiling
    pub enable_profiling: bool,
    /// Enable distributed tracing
    pub enable_tracing: bool,
    /// Enable metrics collection
    pub enable_metrics: bool,
    /// Enable error tracking
    pub enable_error_tracking: bool,
    /// Enable health monitoring
    pub enable_health_monitoring: bool,
    /// Enable alerting
    pub enable_alerting: bool,
    /// Profiling settings
    pub profiling_settings: ProfilingSettings,
    /// Tracing settings
    pub tracing_settings: TracingSettings,
    /// Metrics settings
    pub metrics_settings: MetricsSettings,
    /// Log settings
    pub log_settings: LogSettings,
    /// Health check settings
    pub health_check_settings: HealthCheckSettings,
    /// Custom configuration
    pub custom: HashMap<String, serde_json::Value>,
}

impl Default for DiagnosticConfig {
    fn default() -> Self {
        Self {
            enable_profiling: true,
            enable_tracing: true,
            enable_metrics: true,
            enable_error_tracking: true,
            enable_health_monitoring: true,
            enable_alerting: true,
            profiling_settings: ProfilingSettings::default(),
            tracing_settings: TracingSettings::default(),
            metrics_settings: MetricsSettings::default(),
            log_settings: LogSettings::default(),
            health_check_settings: HealthCheckSettings::default(),
            custom: HashMap::new(),
        }
    }
}

/// Profiling settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfilingSettings {
    /// Sample rate (0.0 to 1.0)
    pub sample_rate: f32,
    /// Maximum profile duration
    pub max_duration: Duration,
    /// Profile buffer size
    pub buffer_size: usize,
    /// Enable CPU profiling
    pub enable_cpu_profiling: bool,
    /// Enable memory profiling
    pub enable_memory_profiling: bool,
    /// Enable I/O profiling
    pub enable_io_profiling: bool,
    /// Profile flush interval
    pub flush_interval: Duration,
}

impl Default for ProfilingSettings {
    fn default() -> Self {
        Self {
            sample_rate: 0.1,
            max_duration: Duration::from_secs(60),
            buffer_size: 10000,
            enable_cpu_profiling: true,
            enable_memory_profiling: true,
            enable_io_profiling: true,
            flush_interval: Duration::from_secs(30),
        }
    }
}

/// Tracing settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TracingSettings {
    /// Service name
    pub service_name: String,
    /// Service version
    pub service_version: String,
    /// Trace sample rate
    pub sample_rate: f32,
    /// Maximum span duration
    pub max_span_duration: Duration,
    /// Enable OpenTelemetry
    pub enable_opentelemetry: bool,
    /// Export interval
    pub export_interval: Duration,
    /// Maximum batch size
    pub max_batch_size: usize,
}

impl Default for TracingSettings {
    fn default() -> Self {
        Self {
            service_name: "execution-context".to_string(),
            service_version: "1.0.0".to_string(),
            sample_rate: 1.0,
            max_span_duration: Duration::from_secs(300),
            enable_opentelemetry: true,
            export_interval: Duration::from_secs(10),
            max_batch_size: 512,
        }
    }
}

/// Metrics settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsSettings {
    /// Collection interval
    pub collection_interval: Duration,
    /// Retention period
    pub retention_period: Duration,
    /// Enable system metrics
    pub enable_system_metrics: bool,
    /// Enable application metrics
    pub enable_application_metrics: bool,
    /// Enable custom metrics
    pub enable_custom_metrics: bool,
    /// Metrics aggregation window
    pub aggregation_window: Duration,
}

impl Default for MetricsSettings {
    fn default() -> Self {
        Self {
            collection_interval: Duration::from_secs(10),
            retention_period: Duration::from_secs(24 * 60 * 60), // 24 hours
            enable_system_metrics: true,
            enable_application_metrics: true,
            enable_custom_metrics: true,
            aggregation_window: Duration::from_secs(60),
        }
    }
}

/// Log settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogSettings {
    /// Log level
    pub log_level: LogLevel,
    /// Log format
    pub log_format: LogFormat,
    /// Enable structured logging
    pub structured_logging: bool,
    /// Log buffer size
    pub buffer_size: usize,
    /// Flush interval
    pub flush_interval: Duration,
    /// Retention period
    pub retention_period: Duration,
    /// Enable log compression
    pub compression: bool,
}

impl Default for LogSettings {
    fn default() -> Self {
        Self {
            log_level: LogLevel::Info,
            log_format: LogFormat::Json,
            structured_logging: true,
            buffer_size: 10000,
            flush_interval: Duration::from_secs(5),
            retention_period: Duration::from_secs(7 * 24 * 60 * 60), // 7 days
            compression: true,
        }
    }
}

/// Log levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum LogLevel {
    /// Trace level
    Trace = 0,
    /// Debug level
    Debug = 1,
    /// Info level
    Info = 2,
    /// Warn level
    Warn = 3,
    /// Error level
    Error = 4,
    /// Fatal level
    Fatal = 5,
}

impl Display for LogLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LogLevel::Trace => write!(f, "TRACE"),
            LogLevel::Debug => write!(f, "DEBUG"),
            LogLevel::Info => write!(f, "INFO"),
            LogLevel::Warn => write!(f, "WARN"),
            LogLevel::Error => write!(f, "ERROR"),
            LogLevel::Fatal => write!(f, "FATAL"),
        }
    }
}

/// Log formats
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum LogFormat {
    /// Plain text format
    Text,
    /// JSON format
    Json,
    /// Logfmt format
    Logfmt,
    /// Custom format
    Custom(String),
}

/// Health check settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheckSettings {
    /// Check interval
    pub check_interval: Duration,
    /// Check timeout
    pub check_timeout: Duration,
    /// Failure threshold
    pub failure_threshold: usize,
    /// Recovery threshold
    pub recovery_threshold: usize,
    /// Enable dependency checks
    pub enable_dependency_checks: bool,
}

impl Default for HealthCheckSettings {
    fn default() -> Self {
        Self {
            check_interval: Duration::from_secs(30),
            check_timeout: Duration::from_secs(10),
            failure_threshold: 3,
            recovery_threshold: 2,
            enable_dependency_checks: true,
        }
    }
}

/// Performance profiler
#[derive(Debug)]
pub struct PerformanceProfiler {
    /// Active profiles
    pub active_profiles: HashMap<String, ProfileSession>,
    /// Completed profiles
    pub completed_profiles: VecDeque<ProfileResult>,
    /// Profile samplers
    pub samplers: HashMap<ProfileType, Box<dyn ProfileSampler>>,
    /// Configuration
    pub config: ProfilingConfig,
}

/// Profile types
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ProfileType {
    /// CPU profiling
    Cpu,
    /// Memory profiling
    Memory,
    /// I/O profiling
    Io,
    /// Network profiling
    Network,
    /// Custom profiling
    Custom(String),
}

/// Profile session
#[derive(Debug, Clone)]
pub struct ProfileSession {
    /// Session ID
    pub id: String,
    /// Profile type
    pub profile_type: ProfileType,
    /// Start time
    pub started_at: Instant,
    /// Sample count
    pub sample_count: AtomicUsize,
    /// Session metadata
    pub metadata: HashMap<String, String>,
}

/// Profile result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileResult {
    /// Profile ID
    pub id: String,
    /// Profile type
    pub profile_type: ProfileType,
    /// Start time
    pub started_at: SystemTime,
    /// End time
    pub ended_at: SystemTime,
    /// Duration
    pub duration: Duration,
    /// Sample count
    pub sample_count: usize,
    /// Profile data
    pub data: ProfileData,
    /// Analysis results
    pub analysis: Option<ProfileAnalysis>,
}

/// Profile data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProfileData {
    /// CPU profile data
    Cpu {
        samples: Vec<CpuSample>,
        call_stacks: Vec<CallStack>,
    },
    /// Memory profile data
    Memory {
        allocations: Vec<MemoryAllocation>,
        heap_snapshots: Vec<HeapSnapshot>,
    },
    /// I/O profile data
    Io {
        operations: Vec<IoOperation>,
        bandwidth_samples: Vec<BandwidthSample>,
    },
    /// Custom profile data
    Custom {
        data: serde_json::Value,
    },
}

/// CPU sample
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuSample {
    /// Timestamp
    pub timestamp: SystemTime,
    /// CPU usage percentage
    pub cpu_percent: f32,
    /// Thread ID
    pub thread_id: u64,
    /// Call stack
    pub call_stack: Vec<StackFrame>,
}

/// Call stack
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CallStack {
    /// Stack frames
    pub frames: Vec<StackFrame>,
    /// Sample count
    pub sample_count: usize,
    /// Total time
    pub total_time: Duration,
}

/// Stack frame
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StackFrame {
    /// Function name
    pub function: String,
    /// File name
    pub file: Option<String>,
    /// Line number
    pub line: Option<u32>,
    /// Module name
    pub module: Option<String>,
}

/// Memory allocation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryAllocation {
    /// Timestamp
    pub timestamp: SystemTime,
    /// Allocation size
    pub size: usize,
    /// Allocation type
    pub allocation_type: AllocationType,
    /// Call stack
    pub call_stack: Vec<StackFrame>,
}

/// Allocation types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AllocationType {
    /// Heap allocation
    Heap,
    /// Stack allocation
    Stack,
    /// Global allocation
    Global,
    /// Custom allocation
    Custom(String),
}

/// Heap snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeapSnapshot {
    /// Snapshot timestamp
    pub timestamp: SystemTime,
    /// Total heap size
    pub total_size: usize,
    /// Used heap size
    pub used_size: usize,
    /// Free heap size
    pub free_size: usize,
    /// Object count
    pub object_count: usize,
    /// Fragmentation ratio
    pub fragmentation_ratio: f32,
}

/// I/O operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IoOperation {
    /// Operation timestamp
    pub timestamp: SystemTime,
    /// Operation type
    pub operation_type: IoOperationType,
    /// Operation size
    pub size: usize,
    /// Operation duration
    pub duration: Duration,
    /// File or resource path
    pub path: Option<String>,
}

/// I/O operation types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum IoOperationType {
    /// File read
    FileRead,
    /// File write
    FileWrite,
    /// Network read
    NetworkRead,
    /// Network write
    NetworkWrite,
    /// Database read
    DatabaseRead,
    /// Database write
    DatabaseWrite,
    /// Custom I/O
    Custom(String),
}

/// Bandwidth sample
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BandwidthSample {
    /// Sample timestamp
    pub timestamp: SystemTime,
    /// Bytes per second
    pub bytes_per_second: usize,
    /// Operation count
    pub operation_count: usize,
}

/// Profile analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileAnalysis {
    /// Hot spots
    pub hot_spots: Vec<HotSpot>,
    /// Performance issues
    pub issues: Vec<PerformanceIssue>,
    /// Recommendations
    pub recommendations: Vec<PerformanceRecommendation>,
    /// Summary statistics
    pub summary: ProfileSummary,
}

/// Hot spot detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HotSpot {
    /// Function or location
    pub location: String,
    /// Time percentage
    pub time_percentage: f32,
    /// Call count
    pub call_count: usize,
    /// Average time per call
    pub avg_time: Duration,
    /// Hot spot type
    pub hot_spot_type: HotSpotType,
}

/// Hot spot types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum HotSpotType {
    /// CPU hot spot
    Cpu,
    /// Memory hot spot
    Memory,
    /// I/O hot spot
    Io,
    /// Lock contention
    Lock,
    /// Custom hot spot
    Custom(String),
}

/// Performance issues
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceIssue {
    /// Issue type
    pub issue_type: PerformanceIssueType,
    /// Issue description
    pub description: String,
    /// Severity level
    pub severity: IssueSeverity,
    /// Location
    pub location: Option<String>,
    /// Impact assessment
    pub impact: IssueImpact,
}

/// Performance issue types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PerformanceIssueType {
    /// High CPU usage
    HighCpuUsage,
    /// Memory leak
    MemoryLeak,
    /// Slow I/O
    SlowIo,
    /// Lock contention
    LockContention,
    /// Excessive allocations
    ExcessiveAllocations,
    /// Custom issue
    Custom(String),
}

/// Issue severity
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum IssueSeverity {
    /// Low severity
    Low = 1,
    /// Medium severity
    Medium = 2,
    /// High severity
    High = 3,
    /// Critical severity
    Critical = 4,
}

/// Issue impact assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IssueImpact {
    /// Performance degradation percentage
    pub performance_impact: f32,
    /// Resource usage impact
    pub resource_impact: f32,
    /// User experience impact
    pub user_experience_impact: f32,
}

/// Performance recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceRecommendation {
    /// Recommendation type
    pub recommendation_type: RecommendationType,
    /// Recommendation description
    pub description: String,
    /// Priority
    pub priority: RecommendationPriority,
    /// Expected benefit
    pub expected_benefit: String,
    /// Implementation effort
    pub implementation_effort: ImplementationEffort,
}

/// Recommendation types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RecommendationType {
    /// Code optimization
    CodeOptimization,
    /// Configuration change
    ConfigurationChange,
    /// Resource scaling
    ResourceScaling,
    /// Architectural change
    ArchitecturalChange,
    /// Custom recommendation
    Custom(String),
}

/// Recommendation priority
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum RecommendationPriority {
    /// Low priority
    Low = 1,
    /// Medium priority
    Medium = 2,
    /// High priority
    High = 3,
    /// Critical priority
    Critical = 4,
}

/// Implementation effort
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ImplementationEffort {
    /// Low effort
    Low,
    /// Medium effort
    Medium,
    /// High effort
    High,
    /// Very high effort
    VeryHigh,
}

/// Profile summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileSummary {
    /// Total samples
    pub total_samples: usize,
    /// Total duration
    pub total_duration: Duration,
    /// Average CPU usage
    pub avg_cpu_usage: f32,
    /// Peak memory usage
    pub peak_memory_usage: usize,
    /// Total I/O operations
    pub total_io_operations: usize,
    /// Total I/O bytes
    pub total_io_bytes: usize,
}

/// Profile sampler trait
pub trait ProfileSampler: Send + Sync {
    /// Start sampling
    fn start_sampling(&mut self, session: &ProfileSession) -> ContextResult<()>;

    /// Stop sampling
    fn stop_sampling(&mut self, session_id: &str) -> ContextResult<ProfileData>;

    /// Get sampler name
    fn name(&self) -> &str;
}

/// Profiling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfilingConfig {
    /// Enabled profile types
    pub enabled_types: HashSet<ProfileType>,
    /// Sample rate
    pub sample_rate: f32,
    /// Buffer size
    pub buffer_size: usize,
    /// Analysis enabled
    pub enable_analysis: bool,
}

impl Default for ProfilingConfig {
    fn default() -> Self {
        let mut enabled_types = HashSet::new();
        enabled_types.insert(ProfileType::Cpu);
        enabled_types.insert(ProfileType::Memory);

        Self {
            enabled_types,
            sample_rate: 0.1,
            buffer_size: 10000,
            enable_analysis: true,
        }
    }
}

/// Distributed tracer
#[derive(Debug)]
pub struct DistributedTracer {
    /// Active spans
    pub active_spans: HashMap<TraceId, TraceSpan>,
    /// Completed traces
    pub completed_traces: VecDeque<Trace>,
    /// Span exporters
    pub exporters: Vec<Box<dyn SpanExporter>>,
    /// Configuration
    pub config: TracingConfig,
}

/// Trace identifier
pub type TraceId = Uuid;

/// Span identifier
pub type SpanId = Uuid;

/// Trace span
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceSpan {
    /// Span ID
    pub span_id: SpanId,
    /// Trace ID
    pub trace_id: TraceId,
    /// Parent span ID
    pub parent_span_id: Option<SpanId>,
    /// Span name
    pub name: String,
    /// Start time
    pub start_time: SystemTime,
    /// End time
    pub end_time: Option<SystemTime>,
    /// Span status
    pub status: SpanStatus,
    /// Span attributes
    pub attributes: HashMap<String, AttributeValue>,
    /// Span events
    pub events: Vec<SpanEvent>,
    /// Span links
    pub links: Vec<SpanLink>,
}

/// Complete trace
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trace {
    /// Trace ID
    pub trace_id: TraceId,
    /// Root span
    pub root_span: TraceSpan,
    /// All spans in trace
    pub spans: Vec<TraceSpan>,
    /// Trace duration
    pub duration: Duration,
    /// Service map
    pub service_map: HashMap<String, Vec<SpanId>>,
}

/// Span status
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SpanStatus {
    /// Span is unset
    Unset,
    /// Span completed successfully
    Ok,
    /// Span completed with error
    Error,
}

/// Attribute value
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AttributeValue {
    /// String value
    String(String),
    /// Integer value
    Int(i64),
    /// Float value
    Float(f64),
    /// Boolean value
    Bool(bool),
    /// Array of values
    Array(Vec<AttributeValue>),
}

/// Span event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpanEvent {
    /// Event name
    pub name: String,
    /// Event timestamp
    pub timestamp: SystemTime,
    /// Event attributes
    pub attributes: HashMap<String, AttributeValue>,
}

/// Span link
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpanLink {
    /// Linked trace ID
    pub trace_id: TraceId,
    /// Linked span ID
    pub span_id: SpanId,
    /// Link attributes
    pub attributes: HashMap<String, AttributeValue>,
}

/// Span exporter trait
pub trait SpanExporter: Send + Sync {
    /// Export spans
    fn export(&mut self, spans: &[TraceSpan]) -> ContextResult<()>;

    /// Flush pending exports
    fn flush(&mut self) -> ContextResult<()>;

    /// Get exporter name
    fn name(&self) -> &str;
}

/// Metrics collector
#[derive(Debug)]
pub struct MetricsCollector {
    /// Metrics registry
    pub registry: MetricsRegistry,
    /// Metric values
    pub values: HashMap<String, MetricValue>,
    /// Collection history
    pub history: VecDeque<MetricSnapshot>,
    /// Configuration
    pub config: MetricsCollectorConfig,
}

/// Metrics registry
#[derive(Debug, Clone)]
pub struct MetricsRegistry {
    /// Registered metrics
    pub metrics: HashMap<String, MetricDefinition>,
}

/// Metric definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricDefinition {
    /// Metric name
    pub name: String,
    /// Metric type
    pub metric_type: MetricType,
    /// Metric description
    pub description: String,
    /// Metric unit
    pub unit: Option<String>,
    /// Metric labels
    pub labels: Vec<String>,
}

/// Metric types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MetricType {
    /// Counter metric (monotonic)
    Counter,
    /// Gauge metric (can go up/down)
    Gauge,
    /// Histogram metric
    Histogram,
    /// Summary metric
    Summary,
    /// Custom metric type
    Custom(String),
}

/// Metric value
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MetricValue {
    /// Counter value
    Counter(u64),
    /// Gauge value
    Gauge(f64),
    /// Histogram value
    Histogram {
        buckets: BTreeMap<f64, u64>,
        sum: f64,
        count: u64,
    },
    /// Summary value
    Summary {
        quantiles: BTreeMap<f64, f64>,
        sum: f64,
        count: u64,
    },
    /// Custom value
    Custom(serde_json::Value),
}

/// Metric snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricSnapshot {
    /// Snapshot timestamp
    pub timestamp: SystemTime,
    /// Metric values at snapshot time
    pub values: HashMap<String, MetricValue>,
}

/// Metrics collector configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsCollectorConfig {
    /// Collection interval
    pub collection_interval: Duration,
    /// History size
    pub history_size: usize,
    /// Enable system metrics
    pub enable_system_metrics: bool,
}

impl Default for MetricsCollectorConfig {
    fn default() -> Self {
        Self {
            collection_interval: Duration::from_secs(10),
            history_size: 1000,
            enable_system_metrics: true,
        }
    }
}

/// Log manager
#[derive(Debug)]
pub struct LogManager {
    /// Log appenders
    pub appenders: HashMap<String, Box<dyn LogAppender>>,
    /// Log buffer
    pub log_buffer: VecDeque<LogEntry>,
    /// Log filters
    pub filters: Vec<Box<dyn LogFilter>>,
    /// Configuration
    pub config: LogManagerConfig,
}

/// Log appender trait
pub trait LogAppender: Send + Sync {
    /// Append log entry
    fn append(&mut self, entry: &LogEntry) -> ContextResult<()>;

    /// Flush pending logs
    fn flush(&mut self) -> ContextResult<()>;

    /// Get appender name
    fn name(&self) -> &str;
}

/// Log filter trait
pub trait LogFilter: Send + Sync {
    /// Check if log entry should be processed
    fn should_log(&self, entry: &LogEntry) -> bool;

    /// Get filter name
    fn name(&self) -> &str;
}

/// Log entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogEntry {
    /// Log ID
    pub id: Uuid,
    /// Log level
    pub level: LogLevel,
    /// Log message
    pub message: String,
    /// Timestamp
    pub timestamp: SystemTime,
    /// Logger name
    pub logger: String,
    /// Thread ID
    pub thread_id: Option<u64>,
    /// Trace ID
    pub trace_id: Option<TraceId>,
    /// Span ID
    pub span_id: Option<SpanId>,
    /// Log fields
    pub fields: HashMap<String, serde_json::Value>,
    /// Error information
    pub error: Option<ErrorInfo>,
}

/// Error information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorInfo {
    /// Error type
    pub error_type: String,
    /// Error message
    pub message: String,
    /// Stack trace
    pub stack_trace: Vec<StackFrame>,
    /// Error cause
    pub cause: Option<Box<ErrorInfo>>,
}

/// Log manager configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogManagerConfig {
    pub buffer_size: usize,
    pub flush_interval: Duration,
    pub async_logging: bool,
}

impl Default for LogManagerConfig {
    fn default() -> Self {
        Self {
            buffer_size: 10000,
            flush_interval: Duration::from_secs(5),
            async_logging: true,
        }
    }
}

/// Error tracker
#[derive(Debug)]
pub struct ErrorTracker {
    /// Error statistics
    pub error_stats: HashMap<String, ErrorStatistics>,
    /// Recent errors
    pub recent_errors: VecDeque<TrackedError>,
    /// Error patterns
    pub error_patterns: Vec<ErrorPattern>,
    /// Configuration
    pub config: ErrorTrackerConfig,
}

/// Error statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorStatistics {
    /// Error type
    pub error_type: String,
    /// Total count
    pub total_count: u64,
    /// Recent count (last hour)
    pub recent_count: u64,
    /// First occurrence
    pub first_seen: SystemTime,
    /// Last occurrence
    pub last_seen: SystemTime,
    /// Affected users
    pub affected_users: HashSet<String>,
}

/// Tracked error
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrackedError {
    /// Error ID
    pub id: Uuid,
    /// Error type
    pub error_type: String,
    /// Error message
    pub message: String,
    /// Timestamp
    pub timestamp: SystemTime,
    /// Context information
    pub context: HashMap<String, serde_json::Value>,
    /// User ID (if applicable)
    pub user_id: Option<String>,
    /// Session ID (if applicable)
    pub session_id: Option<String>,
    /// Stack trace
    pub stack_trace: Vec<StackFrame>,
}

/// Error pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorPattern {
    /// Pattern name
    pub name: String,
    /// Pattern regex
    pub pattern: String,
    /// Pattern severity
    pub severity: IssueSeverity,
    /// Pattern description
    pub description: String,
    /// Alert threshold
    pub alert_threshold: usize,
}

/// Error tracker configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorTrackerConfig {
    /// Enable error tracking
    pub enabled: bool,
    /// Error buffer size
    pub buffer_size: usize,
    /// Statistics retention period
    pub stats_retention: Duration,
    /// Enable pattern matching
    pub enable_pattern_matching: bool,
}

impl Default for ErrorTrackerConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            buffer_size: 1000,
            stats_retention: Duration::from_secs(30 * 24 * 60 * 60), // 30 days
            enable_pattern_matching: true,
        }
    }
}

/// Health monitor
#[derive(Debug)]
pub struct HealthMonitor {
    /// Health checks
    pub health_checks: Vec<Box<dyn HealthCheck>>,
    /// Current health status
    pub current_status: HealthStatus,
    /// Health history
    pub health_history: VecDeque<HealthCheckResult>,
    /// Configuration
    pub config: HealthMonitorConfig,
}

/// Health check trait
pub trait HealthCheck: Send + Sync {
    /// Perform health check
    fn check(&self) -> ContextResult<HealthCheckResult>;

    /// Get health check name
    fn name(&self) -> &str;

    /// Get check timeout
    fn timeout(&self) -> Duration;
}

/// Health status
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum HealthStatus {
    /// System is healthy
    Healthy,
    /// System is degraded
    Degraded,
    /// System is unhealthy
    Unhealthy,
    /// Health status unknown
    Unknown,
}

impl Display for HealthStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HealthStatus::Healthy => write!(f, "healthy"),
            HealthStatus::Degraded => write!(f, "degraded"),
            HealthStatus::Unhealthy => write!(f, "unhealthy"),
            HealthStatus::Unknown => write!(f, "unknown"),
        }
    }
}

/// Health check result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheckResult {
    /// Check name
    pub check_name: String,
    /// Check status
    pub status: HealthStatus,
    /// Check message
    pub message: String,
    /// Check timestamp
    pub timestamp: SystemTime,
    /// Check duration
    pub duration: Duration,
    /// Additional details
    pub details: HashMap<String, serde_json::Value>,
}

/// Health monitor configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthMonitorConfig {
    /// Check interval
    pub check_interval: Duration,
    /// History size
    pub history_size: usize,
    /// Enable automatic recovery
    pub enable_auto_recovery: bool,
}

impl Default for HealthMonitorConfig {
    fn default() -> Self {
        Self {
            check_interval: Duration::from_secs(30),
            history_size: 1000,
            enable_auto_recovery: false,
        }
    }
}

/// Alerting system
#[derive(Debug)]
pub struct AlertingSystem {
    /// Alert rules
    pub alert_rules: Vec<AlertRule>,
    /// Active alerts
    pub active_alerts: HashMap<String, Alert>,
    /// Alert channels
    pub channels: HashMap<String, Box<dyn AlertChannel>>,
    /// Configuration
    pub config: AlertingConfig,
}

/// Alert rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertRule {
    /// Rule ID
    pub id: String,
    /// Rule name
    pub name: String,
    /// Rule condition
    pub condition: AlertCondition,
    /// Rule severity
    pub severity: IssueSeverity,
    /// Alert channels
    pub channels: Vec<String>,
    /// Rule enabled
    pub enabled: bool,
}

/// Alert condition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertCondition {
    /// Metric name
    pub metric: String,
    /// Threshold value
    pub threshold: f64,
    /// Comparison operator
    pub operator: ComparisonOperator,
    /// Evaluation window
    pub window: Duration,
    /// Minimum occurrences
    pub min_occurrences: usize,
}

/// Comparison operators
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ComparisonOperator {
    /// Greater than
    GreaterThan,
    /// Greater than or equal
    GreaterThanOrEqual,
    /// Less than
    LessThan,
    /// Less than or equal
    LessThanOrEqual,
    /// Equal to
    Equal,
    /// Not equal to
    NotEqual,
}

/// Alert
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alert {
    /// Alert ID
    pub id: String,
    /// Alert name
    pub name: String,
    /// Alert message
    pub message: String,
    /// Alert severity
    pub severity: IssueSeverity,
    /// Alert state
    pub state: AlertState,
    /// First triggered time
    pub first_triggered: SystemTime,
    /// Last triggered time
    pub last_triggered: SystemTime,
    /// Trigger count
    pub trigger_count: usize,
    /// Alert metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Alert states
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AlertState {
    /// Alert is firing
    Firing,
    /// Alert is resolved
    Resolved,
    /// Alert is acknowledged
    Acknowledged,
    /// Alert is suppressed
    Suppressed,
}

/// Alert channel trait
pub trait AlertChannel: Send + Sync {
    /// Send alert
    fn send_alert(&mut self, alert: &Alert) -> ContextResult<()>;

    /// Get channel name
    fn name(&self) -> &str;

    /// Check if channel is available
    fn is_available(&self) -> bool;
}

/// Alerting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertingConfig {
    /// Enable alerting
    pub enabled: bool,
    /// Evaluation interval
    pub evaluation_interval: Duration,
    /// Alert resolution timeout
    pub resolution_timeout: Duration,
    /// Enable auto-acknowledgment
    pub enable_auto_ack: bool,
}

impl Default for AlertingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            evaluation_interval: Duration::from_secs(60),
            resolution_timeout: Duration::from_secs(5 * 60),
            enable_auto_ack: false,
        }
    }
}

/// Diagnostic tools collection
#[derive(Debug)]
pub struct DiagnosticTools {
    /// Available tools
    pub tools: HashMap<String, Box<dyn DiagnosticTool>>,
    /// Tool execution history
    pub execution_history: VecDeque<ToolExecution>,
}

/// Diagnostic tool trait
pub trait DiagnosticTool: Send + Sync {
    /// Execute diagnostic tool
    fn execute(&mut self, parameters: HashMap<String, String>) -> ContextResult<ToolResult>;

    /// Get tool name
    fn name(&self) -> &str;

    /// Get tool description
    fn description(&self) -> &str;

    /// Get tool parameters
    fn parameters(&self) -> Vec<ToolParameter>;
}

/// Tool parameter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolParameter {
    /// Parameter name
    pub name: String,
    /// Parameter type
    pub parameter_type: ParameterType,
    /// Parameter description
    pub description: String,
    /// Is required
    pub required: bool,
    /// Default value
    pub default_value: Option<String>,
}

/// Parameter types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ParameterType {
    /// String parameter
    String,
    /// Integer parameter
    Integer,
    /// Float parameter
    Float,
    /// Boolean parameter
    Boolean,
    /// File path parameter
    FilePath,
    /// Custom parameter type
    Custom(String),
}

/// Tool execution record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolExecution {
    /// Execution ID
    pub id: Uuid,
    /// Tool name
    pub tool_name: String,
    /// Start time
    pub started_at: SystemTime,
    /// End time
    pub ended_at: Option<SystemTime>,
    /// Execution status
    pub status: ExecutionStatus,
    /// Parameters used
    pub parameters: HashMap<String, String>,
    /// Execution result
    pub result: Option<ToolResult>,
}

/// Execution status
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExecutionStatus {
    /// Execution is running
    Running,
    /// Execution completed successfully
    Completed,
    /// Execution failed
    Failed,
    /// Execution was cancelled
    Cancelled,
}

/// Tool result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    /// Result data
    pub data: serde_json::Value,
    /// Result summary
    pub summary: String,
    /// Execution duration
    pub duration: Duration,
    /// Output files
    pub output_files: Vec<String>,
}

impl DiagnosticContext {
    /// Create a new diagnostic context
    pub fn new(id: String, config: DiagnosticConfig) -> Self {
        Self {
            id,
            state: Arc::new(RwLock::new(DiagnosticState::Initializing)),
            profiler: Arc::new(Mutex::new(PerformanceProfiler::new())),
            tracer: Arc::new(RwLock::new(DistributedTracer::new())),
            metrics_collector: Arc::new(Mutex::new(MetricsCollector::new())),
            log_manager: Arc::new(RwLock::new(LogManager::new())),
            error_tracker: Arc::new(Mutex::new(ErrorTracker::new())),
            health_monitor: Arc::new(Mutex::new(HealthMonitor::new())),
            alerting_system: Arc::new(Mutex::new(AlertingSystem::new())),
            diagnostic_tools: Arc::new(RwLock::new(DiagnosticTools::new())),
            config: Arc::new(RwLock::new(config)),
            created_at: SystemTime::now(),
        }
    }

    /// Initialize the diagnostic context
    pub fn initialize(&self) -> ContextResult<()> {
        let mut state = self.state.write().map_err(|e|
            ContextError::internal(format!("Failed to acquire state lock: {}", e)))?;

        if *state != DiagnosticState::Initializing {
            return Err(ContextError::custom("invalid_state",
                format!("Cannot initialize diagnostic context in state: {}", state)));
        }

        *state = DiagnosticState::Active;
        Ok(())
    }

    /// Start profiling
    pub fn start_profiling(&self, profile_type: ProfileType, metadata: HashMap<String, String>) -> ContextResult<String> {
        let mut profiler = self.profiler.lock().map_err(|e|
            ContextError::internal(format!("Failed to acquire profiler lock: {}", e)))?;

        profiler.start_profile(profile_type, metadata)
    }

    /// Stop profiling
    pub fn stop_profiling(&self, profile_id: &str) -> ContextResult<ProfileResult> {
        let mut profiler = self.profiler.lock().map_err(|e|
            ContextError::internal(format!("Failed to acquire profiler lock: {}", e)))?;

        profiler.stop_profile(profile_id)
    }

    /// Start trace span
    pub fn start_span(&self, name: String, parent_span_id: Option<SpanId>) -> ContextResult<SpanId> {
        let mut tracer = self.tracer.write().map_err(|e|
            ContextError::internal(format!("Failed to acquire tracer lock: {}", e)))?;

        tracer.start_span(name, parent_span_id)
    }

    /// End trace span
    pub fn end_span(&self, span_id: SpanId) -> ContextResult<()> {
        let mut tracer = self.tracer.write().map_err(|e|
            ContextError::internal(format!("Failed to acquire tracer lock: {}", e)))?;

        tracer.end_span(span_id)
    }

    /// Record metric
    pub fn record_metric(&self, name: String, value: MetricValue) -> ContextResult<()> {
        let mut metrics_collector = self.metrics_collector.lock().map_err(|e|
            ContextError::internal(format!("Failed to acquire metrics collector lock: {}", e)))?;

        metrics_collector.record_metric(name, value)
    }

    /// Log message
    pub fn log(&self, level: LogLevel, message: String, fields: HashMap<String, serde_json::Value>) -> ContextResult<()> {
        let mut log_manager = self.log_manager.write().map_err(|e|
            ContextError::internal(format!("Failed to acquire log manager lock: {}", e)))?;

        let entry = LogEntry {
            id: Uuid::new_v4(),
            level,
            message,
            timestamp: SystemTime::now(),
            logger: self.id.clone(),
            thread_id: None, // Would be populated in real implementation
            trace_id: None,
            span_id: None,
            fields,
            error: None,
        };

        log_manager.log_entry(entry)
    }

    /// Track error
    pub fn track_error(&self, error_type: String, message: String, context: HashMap<String, serde_json::Value>) -> ContextResult<()> {
        let mut error_tracker = self.error_tracker.lock().map_err(|e|
            ContextError::internal(format!("Failed to acquire error tracker lock: {}", e)))?;

        error_tracker.track_error(error_type, message, context)
    }

    /// Get health status
    pub fn get_health_status(&self) -> ContextResult<HealthStatus> {
        let health_monitor = self.health_monitor.lock().map_err(|e|
            ContextError::internal(format!("Failed to acquire health monitor lock: {}", e)))?;

        Ok(health_monitor.current_status.clone())
    }

    /// Get diagnostic state
    pub fn get_state(&self) -> ContextResult<DiagnosticState> {
        let state = self.state.read().map_err(|e|
            ContextError::internal(format!("Failed to acquire state lock: {}", e)))?;
        Ok(*state)
    }
}

// Placeholder implementations (would be implemented properly in real code)
impl PerformanceProfiler {
    fn new() -> Self {
        Self {
            active_profiles: HashMap::new(),
            completed_profiles: VecDeque::new(),
            samplers: HashMap::new(),
            config: ProfilingConfig::default(),
        }
    }

    fn start_profile(&mut self, profile_type: ProfileType, metadata: HashMap<String, String>) -> ContextResult<String> {
        let profile_id = Uuid::new_v4().to_string();
        let session = ProfileSession {
            id: profile_id.clone(),
            profile_type,
            started_at: Instant::now(),
            sample_count: AtomicUsize::new(0),
            metadata,
        };
        self.active_profiles.insert(profile_id.clone(), session);
        Ok(profile_id)
    }

    fn stop_profile(&mut self, profile_id: &str) -> ContextResult<ProfileResult> {
        let session = self.active_profiles.remove(profile_id)
            .ok_or_else(|| ContextError::not_found(format!("profile:{}", profile_id)))?;

        let result = ProfileResult {
            id: profile_id.to_string(),
            profile_type: session.profile_type.clone(),
            started_at: SystemTime::now() - session.started_at.elapsed(),
            ended_at: SystemTime::now(),
            duration: session.started_at.elapsed(),
            sample_count: session.sample_count.load(Ordering::Relaxed),
            data: ProfileData::Custom { data: serde_json::json!({}) },
            analysis: None,
        };

        self.completed_profiles.push_back(result.clone());
        Ok(result)
    }
}

impl DistributedTracer {
    fn new() -> Self {
        Self {
            active_spans: HashMap::new(),
            completed_traces: VecDeque::new(),
            exporters: Vec::new(),
            config: TracingSettings::default(),
        }
    }

    fn start_span(&mut self, name: String, parent_span_id: Option<SpanId>) -> ContextResult<SpanId> {
        let span_id = Uuid::new_v4();
        let trace_id = parent_span_id
            .and_then(|pid| self.active_spans.get(&pid).map(|span| span.trace_id))
            .unwrap_or_else(|| Uuid::new_v4());

        let span = TraceSpan {
            span_id,
            trace_id,
            parent_span_id,
            name,
            start_time: SystemTime::now(),
            end_time: None,
            status: SpanStatus::Unset,
            attributes: HashMap::new(),
            events: Vec::new(),
            links: Vec::new(),
        };

        self.active_spans.insert(trace_id, span);
        Ok(span_id)
    }

    fn end_span(&mut self, span_id: SpanId) -> ContextResult<()> {
        if let Some(mut span) = self.active_spans.remove(&span_id) {
            span.end_time = Some(SystemTime::now());
            span.status = SpanStatus::Ok;
        }
        Ok(())
    }
}

impl MetricsCollector {
    fn new() -> Self {
        Self {
            registry: MetricsRegistry::new(),
            values: HashMap::new(),
            history: VecDeque::new(),
            config: MetricsCollectorConfig::default(),
        }
    }

    fn record_metric(&mut self, name: String, value: MetricValue) -> ContextResult<()> {
        self.values.insert(name, value);
        Ok(())
    }
}

impl MetricsRegistry {
    fn new() -> Self {
        Self {
            metrics: HashMap::new(),
        }
    }
}

impl LogManager {
    fn new() -> Self {
        Self {
            appenders: HashMap::new(),
            log_buffer: VecDeque::new(),
            filters: Vec::new(),
            config: LogManagerConfig::default(),
        }
    }

    fn log_entry(&mut self, entry: LogEntry) -> ContextResult<()> {
        self.log_buffer.push_back(entry);
        Ok(())
    }
}

impl ErrorTracker {
    fn new() -> Self {
        Self {
            error_stats: HashMap::new(),
            recent_errors: VecDeque::new(),
            error_patterns: Vec::new(),
            config: ErrorTrackerConfig::default(),
        }
    }

    fn track_error(&mut self, error_type: String, message: String, context: HashMap<String, serde_json::Value>) -> ContextResult<()> {
        let error = TrackedError {
            id: Uuid::new_v4(),
            error_type: error_type.clone(),
            message,
            timestamp: SystemTime::now(),
            context,
            user_id: None,
            session_id: None,
            stack_trace: Vec::new(),
        };

        self.recent_errors.push_back(error);

        // Update statistics
        let stats = self.error_stats.entry(error_type.clone()).or_insert(ErrorStatistics {
            error_type: error_type.clone(),
            total_count: 0,
            recent_count: 0,
            first_seen: SystemTime::now(),
            last_seen: SystemTime::now(),
            affected_users: HashSet::new(),
        });

        stats.total_count += 1;
        stats.recent_count += 1;
        stats.last_seen = SystemTime::now();

        Ok(())
    }
}

impl HealthMonitor {
    fn new() -> Self {
        Self {
            health_checks: Vec::new(),
            current_status: HealthStatus::Unknown,
            health_history: VecDeque::new(),
            config: HealthMonitorConfig::default(),
        }
    }
}

impl AlertingSystem {
    fn new() -> Self {
        Self {
            alert_rules: Vec::new(),
            active_alerts: HashMap::new(),
            channels: HashMap::new(),
            config: AlertingConfig::default(),
        }
    }
}

impl DiagnosticTools {
    fn new() -> Self {
        Self {
            tools: HashMap::new(),
            execution_history: VecDeque::new(),
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_diagnostic_context_creation() {
        let config = DiagnosticConfig::default();
        let context = DiagnosticContext::new("test-diagnostic".to_string(), config);
        assert_eq!(context.id, "test-diagnostic");
    }

    #[test]
    fn test_diagnostic_states() {
        assert_eq!(DiagnosticState::Active.to_string(), "active");
        assert_eq!(DiagnosticState::Debug.to_string(), "debug");
    }

    #[test]
    fn test_log_levels() {
        assert!(LogLevel::Error > LogLevel::Warn);
        assert!(LogLevel::Warn > LogLevel::Info);
        assert_eq!(LogLevel::Info.to_string(), "INFO");
    }

    #[test]
    fn test_profile_types() {
        assert_eq!(ProfileType::Cpu, ProfileType::Cpu);
        assert_ne!(ProfileType::Cpu, ProfileType::Memory);
    }

    #[test]
    fn test_span_status() {
        assert_eq!(SpanStatus::Ok, SpanStatus::Ok);
        assert_ne!(SpanStatus::Ok, SpanStatus::Error);
    }

    #[test]
    fn test_health_status() {
        assert_eq!(HealthStatus::Healthy.to_string(), "healthy");
        assert_eq!(HealthStatus::Unhealthy.to_string(), "unhealthy");
    }

    #[test]
    fn test_metric_types() {
        assert_eq!(MetricType::Counter, MetricType::Counter);
        assert_ne!(MetricType::Counter, MetricType::Gauge);
    }

    #[test]
    fn test_issue_severity() {
        assert!(IssueSeverity::Critical > IssueSeverity::High);
        assert!(IssueSeverity::High > IssueSeverity::Medium);
    }

    #[test]
    fn test_alert_states() {
        assert_eq!(AlertState::Firing, AlertState::Firing);
        assert_ne!(AlertState::Firing, AlertState::Resolved);
    }

    #[test]
    fn test_execution_status() {
        assert_eq!(ExecutionStatus::Running, ExecutionStatus::Running);
        assert_ne!(ExecutionStatus::Running, ExecutionStatus::Completed);
    }
}