//! Reporting System for Execution Monitoring
//!
//! This module provides comprehensive reporting capabilities for the execution monitoring
//! framework, including report generation, scheduling, distribution, templating, and
//! interactive dashboard functionality.

use std::collections::{HashMap, HashSet, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock, Mutex};
use std::time::{Duration, SystemTime, Instant};
use std::fmt;
use std::fs;
use std::io::Write;
use serde::{Serialize, Deserialize};
use serde_json::{Value, Map};
use chrono::{DateTime, Utc, TimeZone, NaiveDateTime};
use uuid::Uuid;

use crate::metrics_collection::{PerformanceMetric, MetricsStorage};
use crate::event_tracking::{TaskExecutionEvent, EventBuffer};
use crate::alerting_system::{AlertManager, AlertRule};
use crate::health_monitoring::{HealthChecker, ComponentHealthState};
use crate::performance_analysis::{PerformanceAnalyzer, PerformanceInsights};

/// Comprehensive reporting system for execution monitoring
#[derive(Debug)]
pub struct ReportingSystem {
    /// Report generator engine
    generator: Arc<ReportGenerator>,
    /// Report scheduler for automated generation
    scheduler: Arc<ReportScheduler>,
    /// Report distribution manager
    distributor: Arc<ReportDistributor>,
    /// Report storage and archival
    storage: Arc<ReportStorage>,
    /// Template manager for report customization
    template_manager: Arc<TemplateManager>,
    /// Interactive dashboard engine
    dashboard: Arc<DashboardEngine>,
    /// Report security and access control
    security: Arc<ReportSecurity>,
    /// Report cache for performance optimization
    cache: Arc<Mutex<ReportCache>>,
    /// Report registry for tracking all reports
    registry: Arc<Mutex<ReportRegistry>>,
}

/// Report generator for creating various types of reports
#[derive(Debug)]
pub struct ReportGenerator {
    /// Data aggregators for different metrics
    aggregators: HashMap<String, Box<dyn DataAggregator + Send + Sync>>,
    /// Report builders for different formats
    builders: HashMap<String, Box<dyn ReportBuilder + Send + Sync>>,
    /// Chart generators for visualizations
    chart_generators: HashMap<String, Box<dyn ChartGenerator + Send + Sync>>,
    /// Export handlers for different output formats
    exporters: HashMap<String, Box<dyn ReportExporter + Send + Sync>>,
}

/// Report scheduler for automated report generation
#[derive(Debug)]
pub struct ReportScheduler {
    /// Scheduled report definitions
    scheduled_reports: Arc<RwLock<HashMap<String, ScheduledReport>>>,
    /// Cron expression parser and evaluator
    cron_engine: Arc<CronEngine>,
    /// Scheduler thread pool
    thread_pool: Arc<ThreadPool>,
    /// Report execution history
    execution_history: Arc<Mutex<VecDeque<ReportExecution>>>,
}

/// Report distribution manager for delivering reports
#[derive(Debug)]
pub struct ReportDistributor {
    /// Distribution channels (email, file, API, etc.)
    channels: HashMap<String, Box<dyn DistributionChannel + Send + Sync>>,
    /// Distribution rules and filters
    rules: Vec<DistributionRule>,
    /// Delivery tracking and confirmation
    delivery_tracker: Arc<Mutex<DeliveryTracker>>,
    /// Retry manager for failed deliveries
    retry_manager: Arc<RetryManager>,
}

/// Report storage and archival system
#[derive(Debug)]
pub struct ReportStorage {
    /// Primary storage backend
    primary_storage: Box<dyn StorageBackend + Send + Sync>,
    /// Archival storage for long-term retention
    archival_storage: Option<Box<dyn StorageBackend + Send + Sync>>,
    /// Storage metadata index
    metadata_index: Arc<RwLock<StorageIndex>>,
    /// Compression and optimization settings
    compression: CompressionSettings,
}

/// Template manager for report customization
#[derive(Debug)]
pub struct TemplateManager {
    /// Template definitions and schemas
    templates: Arc<RwLock<HashMap<String, ReportTemplate>>>,
    /// Template rendering engine
    renderer: Arc<TemplateRenderer>,
    /// Template validation and compilation
    validator: Arc<TemplateValidator>,
    /// Custom template functions and filters
    custom_functions: HashMap<String, Box<dyn TemplateFunction + Send + Sync>>,
}

/// Interactive dashboard engine for real-time reporting
#[derive(Debug)]
pub struct DashboardEngine {
    /// Dashboard definitions and layouts
    dashboards: Arc<RwLock<HashMap<String, Dashboard>>>,
    /// Widget registry for dashboard components
    widgets: HashMap<String, Box<dyn DashboardWidget + Send + Sync>>,
    /// Real-time data streaming
    data_streamer: Arc<DataStreamer>,
    /// Dashboard state management
    state_manager: Arc<DashboardStateManager>,
}

/// Report security and access control
#[derive(Debug)]
pub struct ReportSecurity {
    /// Access control policies
    access_policies: Vec<AccessPolicy>,
    /// Report audit trail
    audit_trail: Arc<Mutex<AuditTrail>>,
    /// Report encryption settings
    encryption: EncryptionSettings,
    /// Data privacy and anonymization
    privacy: PrivacySettings,
}

/// Report definition structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportDefinition {
    /// Report unique identifier
    pub id: String,
    /// Report name and title
    pub name: String,
    /// Report description
    pub description: String,
    /// Report type (summary, detailed, analytics, etc.)
    pub report_type: ReportType,
    /// Data sources for the report
    pub data_sources: Vec<DataSource>,
    /// Report parameters and filters
    pub parameters: ReportParameters,
    /// Report layout and formatting
    pub layout: ReportLayout,
    /// Report output configuration
    pub output: OutputConfiguration,
    /// Report metadata
    pub metadata: ReportMetadata,
}

/// Report type enumeration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ReportType {
    /// Executive summary report
    Executive,
    /// Detailed operational report
    Operational,
    /// Performance analytics report
    Analytics,
    /// Health status report
    Health,
    /// Incident and alert report
    Incident,
    /// Compliance and audit report
    Compliance,
    /// Custom report type
    Custom(String),
}

/// Data source configuration for reports
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataSource {
    /// Data source identifier
    pub id: String,
    /// Data source type (metrics, events, alerts, etc.)
    pub source_type: DataSourceType,
    /// Data query or filter criteria
    pub query: DataQuery,
    /// Data transformation rules
    pub transformations: Vec<DataTransformation>,
    /// Data aggregation settings
    pub aggregation: AggregationSettings,
}

/// Data source type enumeration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum DataSourceType {
    /// Performance metrics data
    Metrics,
    /// Event tracking data
    Events,
    /// Alert and notification data
    Alerts,
    /// Health monitoring data
    Health,
    /// System performance data
    System,
    /// Custom data source
    Custom(String),
}

/// Data query structure for filtering and selection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataQuery {
    /// Time range for data selection
    pub time_range: TimeRange,
    /// Filter conditions
    pub filters: Vec<QueryFilter>,
    /// Sort criteria
    pub sort: Option<SortCriteria>,
    /// Result limits
    pub limit: Option<usize>,
    /// Field selection
    pub fields: Option<Vec<String>>,
}

/// Time range specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeRange {
    /// Start time (absolute or relative)
    pub start: TimeSpecification,
    /// End time (absolute or relative)
    pub end: TimeSpecification,
    /// Time zone for interpretation
    pub timezone: Option<String>,
}

/// Time specification (absolute or relative)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TimeSpecification {
    /// Absolute timestamp
    Absolute(DateTime<Utc>),
    /// Relative time (e.g., "1 hour ago", "yesterday")
    Relative(RelativeTime),
    /// Dynamic time (e.g., "start of today", "end of month")
    Dynamic(DynamicTime),
}

/// Relative time specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelativeTime {
    /// Time unit (minutes, hours, days, etc.)
    pub unit: TimeUnit,
    /// Number of units
    pub count: i64,
    /// Direction (ago or from now)
    pub direction: TimeDirection,
}

/// Time unit enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TimeUnit {
    Seconds,
    Minutes,
    Hours,
    Days,
    Weeks,
    Months,
    Years,
}

/// Time direction enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TimeDirection {
    Ago,
    FromNow,
}

/// Dynamic time specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DynamicTime {
    StartOfDay,
    EndOfDay,
    StartOfWeek,
    EndOfWeek,
    StartOfMonth,
    EndOfMonth,
    StartOfYear,
    EndOfYear,
}

/// Query filter for data selection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryFilter {
    /// Field to filter on
    pub field: String,
    /// Filter operator
    pub operator: FilterOperator,
    /// Filter value
    pub value: Value,
    /// Logical connector with next filter
    pub connector: Option<LogicalConnector>,
}

/// Filter operator enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FilterOperator {
    Equals,
    NotEquals,
    GreaterThan,
    GreaterThanOrEqual,
    LessThan,
    LessThanOrEqual,
    Contains,
    NotContains,
    StartsWith,
    EndsWith,
    In,
    NotIn,
    IsNull,
    IsNotNull,
    Regex,
}

/// Logical connector for filter combinations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LogicalConnector {
    And,
    Or,
    Not,
}

/// Sort criteria for query results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SortCriteria {
    /// Field to sort by
    pub field: String,
    /// Sort direction
    pub direction: SortDirection,
}

/// Sort direction enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SortDirection {
    Ascending,
    Descending,
}

/// Data transformation rules
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataTransformation {
    /// Transformation type
    pub transformation_type: TransformationType,
    /// Transformation parameters
    pub parameters: Map<String, Value>,
    /// Source fields
    pub source_fields: Vec<String>,
    /// Target field
    pub target_field: String,
}

/// Data transformation type enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TransformationType {
    /// Mathematical calculations
    Calculate,
    /// Data format conversion
    Convert,
    /// Data aggregation
    Aggregate,
    /// Data enrichment
    Enrich,
    /// Data filtering
    Filter,
    /// Custom transformation
    Custom(String),
}

/// Data aggregation settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregationSettings {
    /// Aggregation functions to apply
    pub functions: Vec<AggregationFunction>,
    /// Group by fields
    pub group_by: Vec<String>,
    /// Time bucket size for time-based aggregation
    pub time_bucket: Option<Duration>,
    /// Having conditions for aggregated data
    pub having: Vec<QueryFilter>,
}

/// Aggregation function specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregationFunction {
    /// Function type (sum, avg, count, etc.)
    pub function: AggregationType,
    /// Source field
    pub field: String,
    /// Output field name
    pub alias: Option<String>,
}

/// Aggregation type enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AggregationType {
    Count,
    Sum,
    Average,
    Minimum,
    Maximum,
    StandardDeviation,
    Variance,
    Percentile(f64),
    Distinct,
    First,
    Last,
}

/// Report parameters and filters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportParameters {
    /// Static parameters
    pub static_params: Map<String, Value>,
    /// Dynamic parameters based on context
    pub dynamic_params: Vec<DynamicParameter>,
    /// User-configurable parameters
    pub user_params: Vec<UserParameter>,
    /// Environment-specific parameters
    pub env_params: Map<String, Value>,
}

/// Dynamic parameter definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DynamicParameter {
    /// Parameter name
    pub name: String,
    /// Parameter source (context, environment, etc.)
    pub source: ParameterSource,
    /// Default value if source is unavailable
    pub default: Option<Value>,
    /// Parameter transformation
    pub transformation: Option<String>,
}

/// Parameter source enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ParameterSource {
    /// From execution context
    Context(String),
    /// From environment variables
    Environment(String),
    /// From configuration
    Configuration(String),
    /// From user input
    UserInput,
    /// From system state
    System(String),
}

/// User-configurable parameter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserParameter {
    /// Parameter name
    pub name: String,
    /// Parameter type
    pub param_type: ParameterType,
    /// Parameter description
    pub description: String,
    /// Default value
    pub default: Option<Value>,
    /// Validation rules
    pub validation: Option<ParameterValidation>,
    /// Parameter options (for selection types)
    pub options: Option<Vec<ParameterOption>>,
}

/// Parameter type enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ParameterType {
    String,
    Integer,
    Float,
    Boolean,
    Date,
    DateTime,
    Duration,
    Selection,
    MultiSelection,
    Range,
}

/// Parameter validation rules
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterValidation {
    /// Required parameter
    pub required: bool,
    /// Minimum value (for numeric types)
    pub min: Option<Value>,
    /// Maximum value (for numeric types)
    pub max: Option<Value>,
    /// Regular expression pattern (for string types)
    pub pattern: Option<String>,
    /// Custom validation function
    pub custom: Option<String>,
}

/// Parameter option for selection types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterOption {
    /// Option label
    pub label: String,
    /// Option value
    pub value: Value,
    /// Option description
    pub description: Option<String>,
}

/// Report layout and formatting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportLayout {
    /// Report sections
    pub sections: Vec<ReportSection>,
    /// Page settings
    pub page: PageSettings,
    /// Style and theming
    pub style: StyleSettings,
    /// Header and footer configuration
    pub header_footer: HeaderFooterSettings,
}

/// Report section definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportSection {
    /// Section identifier
    pub id: String,
    /// Section title
    pub title: String,
    /// Section type
    pub section_type: SectionType,
    /// Section content
    pub content: SectionContent,
    /// Section layout properties
    pub layout: SectionLayout,
    /// Section visibility conditions
    pub visibility: Option<VisibilityCondition>,
}

/// Report section type enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SectionType {
    /// Text content section
    Text,
    /// Data table section
    Table,
    /// Chart and visualization section
    Chart,
    /// Image or media section
    Media,
    /// Summary statistics section
    Summary,
    /// List or enumeration section
    List,
    /// Custom section type
    Custom(String),
}

/// Section content specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SectionContent {
    /// Content source
    pub source: ContentSource,
    /// Content formatting
    pub formatting: ContentFormatting,
    /// Content transformations
    pub transformations: Vec<ContentTransformation>,
}

/// Content source enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ContentSource {
    /// Static content
    Static(String),
    /// Data query result
    DataQuery(String),
    /// Template rendering
    Template(String),
    /// External content source
    External(String),
}

/// Content formatting settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentFormatting {
    /// Number formatting
    pub number_format: Option<NumberFormat>,
    /// Date formatting
    pub date_format: Option<String>,
    /// Text alignment
    pub alignment: Option<TextAlignment>,
    /// Font settings
    pub font: Option<FontSettings>,
    /// Color settings
    pub colors: Option<ColorSettings>,
}

/// Number formatting specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NumberFormat {
    /// Decimal places
    pub decimals: Option<usize>,
    /// Thousands separator
    pub thousands_separator: Option<String>,
    /// Decimal separator
    pub decimal_separator: Option<String>,
    /// Number prefix (currency, etc.)
    pub prefix: Option<String>,
    /// Number suffix (%, etc.)
    pub suffix: Option<String>,
}

/// Text alignment enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TextAlignment {
    Left,
    Center,
    Right,
    Justify,
}

/// Font settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FontSettings {
    /// Font family
    pub family: Option<String>,
    /// Font size
    pub size: Option<f64>,
    /// Font weight
    pub weight: Option<FontWeight>,
    /// Font style
    pub style: Option<FontStyle>,
}

/// Font weight enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FontWeight {
    Light,
    Normal,
    Bold,
    ExtraBold,
}

/// Font style enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FontStyle {
    Normal,
    Italic,
    Oblique,
}

/// Color settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColorSettings {
    /// Text color
    pub text: Option<String>,
    /// Background color
    pub background: Option<String>,
    /// Border color
    pub border: Option<String>,
}

/// Content transformation specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentTransformation {
    /// Transformation type
    pub transform_type: ContentTransformationType,
    /// Transformation parameters
    pub parameters: Map<String, Value>,
}

/// Content transformation type enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ContentTransformationType {
    /// Data sorting
    Sort,
    /// Data filtering
    Filter,
    /// Data grouping
    Group,
    /// Data pivoting
    Pivot,
    /// Data aggregation
    Aggregate,
    /// Custom transformation
    Custom(String),
}

/// Section layout properties
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SectionLayout {
    /// Column span
    pub columns: Option<usize>,
    /// Row span
    pub rows: Option<usize>,
    /// Margin settings
    pub margin: Option<MarginSettings>,
    /// Padding settings
    pub padding: Option<PaddingSettings>,
    /// Border settings
    pub border: Option<BorderSettings>,
}

/// Margin settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarginSettings {
    pub top: Option<f64>,
    pub right: Option<f64>,
    pub bottom: Option<f64>,
    pub left: Option<f64>,
}

/// Padding settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaddingSettings {
    pub top: Option<f64>,
    pub right: Option<f64>,
    pub bottom: Option<f64>,
    pub left: Option<f64>,
}

/// Border settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BorderSettings {
    /// Border width
    pub width: Option<f64>,
    /// Border style
    pub style: Option<BorderStyle>,
    /// Border color
    pub color: Option<String>,
}

/// Border style enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BorderStyle {
    None,
    Solid,
    Dashed,
    Dotted,
    Double,
}

/// Section visibility condition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisibilityCondition {
    /// Condition expression
    pub condition: String,
    /// Condition parameters
    pub parameters: Map<String, Value>,
}

/// Page settings for report layout
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PageSettings {
    /// Page size
    pub size: PageSize,
    /// Page orientation
    pub orientation: PageOrientation,
    /// Page margins
    pub margins: MarginSettings,
}

/// Page size enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PageSize {
    A4,
    A3,
    Letter,
    Legal,
    Custom { width: f64, height: f64 },
}

/// Page orientation enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PageOrientation {
    Portrait,
    Landscape,
}

/// Style and theming settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StyleSettings {
    /// Theme name
    pub theme: Option<String>,
    /// Color palette
    pub colors: Option<ColorPalette>,
    /// Typography settings
    pub typography: Option<TypographySettings>,
}

/// Color palette definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColorPalette {
    /// Primary colors
    pub primary: Vec<String>,
    /// Secondary colors
    pub secondary: Vec<String>,
    /// Accent colors
    pub accent: Vec<String>,
    /// Neutral colors
    pub neutral: Vec<String>,
}

/// Typography settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypographySettings {
    /// Heading font
    pub heading_font: Option<FontSettings>,
    /// Body font
    pub body_font: Option<FontSettings>,
    /// Code font
    pub code_font: Option<FontSettings>,
    /// Line height
    pub line_height: Option<f64>,
}

/// Header and footer settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeaderFooterSettings {
    /// Header configuration
    pub header: Option<HeaderFooterContent>,
    /// Footer configuration
    pub footer: Option<HeaderFooterContent>,
}

/// Header/footer content specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeaderFooterContent {
    /// Left content
    pub left: Option<String>,
    /// Center content
    pub center: Option<String>,
    /// Right content
    pub right: Option<String>,
    /// Content height
    pub height: Option<f64>,
}

/// Report output configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputConfiguration {
    /// Output formats
    pub formats: Vec<OutputFormat>,
    /// File naming pattern
    pub filename_pattern: String,
    /// Compression settings
    pub compression: Option<CompressionType>,
    /// Quality settings
    pub quality: Option<QualitySettings>,
}

/// Output format enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OutputFormat {
    PDF,
    HTML,
    Excel,
    CSV,
    JSON,
    XML,
    PNG,
    JPEG,
    SVG,
}

/// Compression type enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompressionType {
    None,
    ZIP,
    GZIP,
    BZIP2,
}

/// Quality settings for output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualitySettings {
    /// Image quality (for image formats)
    pub image_quality: Option<f64>,
    /// DPI setting
    pub dpi: Option<f64>,
    /// Optimization level
    pub optimization: Option<OptimizationLevel>,
}

/// Optimization level enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationLevel {
    None,
    Low,
    Medium,
    High,
    Maximum,
}

/// Report metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportMetadata {
    /// Report version
    pub version: String,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    /// Last modification timestamp
    pub modified_at: DateTime<Utc>,
    /// Report author
    pub author: String,
    /// Report tags
    pub tags: Vec<String>,
    /// Report category
    pub category: Option<String>,
    /// Report priority
    pub priority: ReportPriority,
}

/// Report priority enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReportPriority {
    Low,
    Normal,
    High,
    Critical,
}

/// Generated report information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratedReport {
    /// Report unique identifier
    pub id: String,
    /// Report definition ID
    pub definition_id: String,
    /// Generation timestamp
    pub generated_at: DateTime<Utc>,
    /// Report status
    pub status: ReportStatus,
    /// Report files and outputs
    pub outputs: Vec<ReportOutput>,
    /// Generation parameters used
    pub parameters: Map<String, Value>,
    /// Generation metrics
    pub metrics: GenerationMetrics,
    /// Error information (if any)
    pub error: Option<ReportError>,
}

/// Report status enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReportStatus {
    Pending,
    Generating,
    Completed,
    Failed,
    Cancelled,
}

/// Report output file information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportOutput {
    /// Output format
    pub format: OutputFormat,
    /// File path or URL
    pub location: String,
    /// File size in bytes
    pub size: u64,
    /// File checksum
    pub checksum: String,
    /// MIME type
    pub mime_type: String,
}

/// Report generation metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationMetrics {
    /// Generation duration
    pub duration: Duration,
    /// Memory usage peak
    pub memory_usage: u64,
    /// Data rows processed
    pub rows_processed: u64,
    /// Cache hit rate
    pub cache_hit_rate: f64,
}

/// Report error information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportError {
    /// Error code
    pub code: String,
    /// Error message
    pub message: String,
    /// Error details
    pub details: Option<String>,
    /// Stack trace
    pub stack_trace: Option<String>,
}

/// Scheduled report definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScheduledReport {
    /// Schedule identifier
    pub id: String,
    /// Report definition to generate
    pub report_definition_id: String,
    /// Schedule expression (cron format)
    pub schedule: String,
    /// Schedule enabled
    pub enabled: bool,
    /// Schedule parameters
    pub parameters: Map<String, Value>,
    /// Distribution configuration
    pub distribution: Vec<String>,
    /// Next execution time
    pub next_execution: DateTime<Utc>,
    /// Last execution information
    pub last_execution: Option<ReportExecution>,
}

/// Report execution record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportExecution {
    /// Execution identifier
    pub id: String,
    /// Schedule identifier
    pub schedule_id: String,
    /// Execution timestamp
    pub executed_at: DateTime<Utc>,
    /// Execution status
    pub status: ExecutionStatus,
    /// Generated report ID
    pub report_id: Option<String>,
    /// Execution duration
    pub duration: Duration,
    /// Error information
    pub error: Option<String>,
}

/// Execution status enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExecutionStatus {
    Scheduled,
    Running,
    Completed,
    Failed,
    Skipped,
}

/// Report template for customization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportTemplate {
    /// Template identifier
    pub id: String,
    /// Template name
    pub name: String,
    /// Template content
    pub content: String,
    /// Template engine type
    pub engine: TemplateEngine,
    /// Template variables
    pub variables: Vec<TemplateVariable>,
    /// Template metadata
    pub metadata: TemplateMetadata,
}

/// Template engine enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TemplateEngine {
    Handlebars,
    Jinja2,
    Mustache,
    Custom(String),
}

/// Template variable definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemplateVariable {
    /// Variable name
    pub name: String,
    /// Variable type
    pub var_type: VariableType,
    /// Default value
    pub default: Option<Value>,
    /// Variable description
    pub description: String,
}

/// Template variable type enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VariableType {
    String,
    Number,
    Boolean,
    Array,
    Object,
    Date,
}

/// Template metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemplateMetadata {
    /// Template version
    pub version: String,
    /// Template author
    pub author: String,
    /// Template description
    pub description: String,
    /// Template tags
    pub tags: Vec<String>,
}

/// Dashboard definition for interactive reporting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dashboard {
    /// Dashboard identifier
    pub id: String,
    /// Dashboard name
    pub name: String,
    /// Dashboard layout
    pub layout: DashboardLayout,
    /// Dashboard widgets
    pub widgets: Vec<DashboardWidgetInstance>,
    /// Dashboard filters
    pub filters: Vec<DashboardFilter>,
    /// Dashboard metadata
    pub metadata: DashboardMetadata,
}

/// Dashboard layout configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardLayout {
    /// Layout type
    pub layout_type: DashboardLayoutType,
    /// Grid configuration
    pub grid: GridConfiguration,
    /// Responsive settings
    pub responsive: ResponsiveSettings,
}

/// Dashboard layout type enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DashboardLayoutType {
    Grid,
    Flow,
    Fixed,
    Responsive,
}

/// Grid configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GridConfiguration {
    /// Number of columns
    pub columns: usize,
    /// Row height
    pub row_height: f64,
    /// Grid spacing
    pub spacing: f64,
}

/// Responsive design settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponsiveSettings {
    /// Breakpoints for different screen sizes
    pub breakpoints: Map<String, f64>,
    /// Column configuration per breakpoint
    pub columns_per_breakpoint: Map<String, usize>,
}

/// Dashboard widget instance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardWidgetInstance {
    /// Widget instance identifier
    pub id: String,
    /// Widget type
    pub widget_type: String,
    /// Widget configuration
    pub config: Map<String, Value>,
    /// Widget position and size
    pub position: WidgetPosition,
    /// Widget data binding
    pub data_binding: Option<DataBinding>,
}

/// Widget position and size
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WidgetPosition {
    /// Grid column position
    pub x: usize,
    /// Grid row position
    pub y: usize,
    /// Widget width in grid units
    pub width: usize,
    /// Widget height in grid units
    pub height: usize,
}

/// Data binding configuration for widgets
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataBinding {
    /// Data source identifier
    pub source: String,
    /// Data query
    pub query: DataQuery,
    /// Refresh interval
    pub refresh_interval: Option<Duration>,
    /// Auto-refresh enabled
    pub auto_refresh: bool,
}

/// Dashboard filter configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardFilter {
    /// Filter identifier
    pub id: String,
    /// Filter name
    pub name: String,
    /// Filter type
    pub filter_type: DashboardFilterType,
    /// Filter configuration
    pub config: Map<String, Value>,
    /// Affected widgets
    pub target_widgets: Vec<String>,
}

/// Dashboard filter type enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DashboardFilterType {
    TimeRange,
    Category,
    Search,
    Range,
    MultiSelect,
    Custom(String),
}

/// Dashboard metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardMetadata {
    /// Dashboard version
    pub version: String,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    /// Last modification timestamp
    pub modified_at: DateTime<Utc>,
    /// Dashboard author
    pub author: String,
    /// Dashboard description
    pub description: String,
    /// Dashboard tags
    pub tags: Vec<String>,
}

// Trait definitions for extensibility

/// Data aggregator trait for processing monitoring data
pub trait DataAggregator: Send + Sync {
    /// Aggregate data based on query and settings
    fn aggregate(&self, data: &[Value], settings: &AggregationSettings) -> Result<Vec<Value>, ReportingError>;

    /// Get supported aggregation functions
    fn supported_functions(&self) -> Vec<AggregationType>;
}

/// Report builder trait for creating reports in different formats
pub trait ReportBuilder: Send + Sync {
    /// Build report from definition and data
    fn build(&self, definition: &ReportDefinition, data: &[Value]) -> Result<Vec<u8>, ReportingError>;

    /// Get supported output formats
    fn supported_formats(&self) -> Vec<OutputFormat>;
}

/// Chart generator trait for creating visualizations
pub trait ChartGenerator: Send + Sync {
    /// Generate chart from data and configuration
    fn generate(&self, data: &[Value], config: &Map<String, Value>) -> Result<Vec<u8>, ReportingError>;

    /// Get supported chart types
    fn supported_types(&self) -> Vec<String>;
}

/// Report exporter trait for different output formats
pub trait ReportExporter: Send + Sync {
    /// Export report to specified format
    fn export(&self, content: &[u8], format: &OutputFormat, config: &OutputConfiguration) -> Result<Vec<u8>, ReportingError>;

    /// Get supported export formats
    fn supported_formats(&self) -> Vec<OutputFormat>;
}

/// Distribution channel trait for delivering reports
pub trait DistributionChannel: Send + Sync {
    /// Deliver report through the channel
    fn deliver(&self, report: &GeneratedReport, config: &Map<String, Value>) -> Result<DeliveryReceipt, ReportingError>;

    /// Get channel capabilities
    fn capabilities(&self) -> ChannelCapabilities;
}

/// Storage backend trait for report storage
pub trait StorageBackend: Send + Sync {
    /// Store report data
    fn store(&self, report: &GeneratedReport, data: &[u8]) -> Result<String, ReportingError>;

    /// Retrieve report data
    fn retrieve(&self, report_id: &str) -> Result<Vec<u8>, ReportingError>;

    /// Delete report data
    fn delete(&self, report_id: &str) -> Result<(), ReportingError>;

    /// List stored reports
    fn list(&self, filter: &StorageFilter) -> Result<Vec<StorageEntry>, ReportingError>;
}

/// Dashboard widget trait for interactive components
pub trait DashboardWidget: Send + Sync {
    /// Render widget with data
    fn render(&self, data: &[Value], config: &Map<String, Value>) -> Result<WidgetOutput, ReportingError>;

    /// Get widget metadata
    fn metadata(&self) -> WidgetMetadata;
}

/// Template function trait for custom template functions
pub trait TemplateFunction: Send + Sync {
    /// Execute template function
    fn execute(&self, args: &[Value]) -> Result<Value, ReportingError>;

    /// Get function signature
    fn signature(&self) -> FunctionSignature;
}

// Supporting types and structures

/// Reporting error enumeration
#[derive(Debug, Clone)]
pub enum ReportingError {
    /// Data source error
    DataSourceError(String),
    /// Template rendering error
    TemplateError(String),
    /// Output generation error
    OutputError(String),
    /// Distribution error
    DistributionError(String),
    /// Storage error
    StorageError(String),
    /// Configuration error
    ConfigurationError(String),
    /// Validation error
    ValidationError(String),
    /// Security error
    SecurityError(String),
    /// Network error
    NetworkError(String),
    /// Generic error
    Other(String),
}

impl fmt::Display for ReportingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ReportingError::DataSourceError(msg) => write!(f, "Data source error: {}", msg),
            ReportingError::TemplateError(msg) => write!(f, "Template error: {}", msg),
            ReportingError::OutputError(msg) => write!(f, "Output error: {}", msg),
            ReportingError::DistributionError(msg) => write!(f, "Distribution error: {}", msg),
            ReportingError::StorageError(msg) => write!(f, "Storage error: {}", msg),
            ReportingError::ConfigurationError(msg) => write!(f, "Configuration error: {}", msg),
            ReportingError::ValidationError(msg) => write!(f, "Validation error: {}", msg),
            ReportingError::SecurityError(msg) => write!(f, "Security error: {}", msg),
            ReportingError::NetworkError(msg) => write!(f, "Network error: {}", msg),
            ReportingError::Other(msg) => write!(f, "Reporting error: {}", msg),
        }
    }
}

impl std::error::Error for ReportingError {}

/// Delivery receipt for distribution confirmation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeliveryReceipt {
    /// Delivery identifier
    pub id: String,
    /// Delivery timestamp
    pub delivered_at: DateTime<Utc>,
    /// Delivery status
    pub status: DeliveryStatus,
    /// Delivery details
    pub details: Map<String, Value>,
}

/// Delivery status enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DeliveryStatus {
    Pending,
    Delivered,
    Failed,
    Retrying,
}

/// Channel capabilities description
#[derive(Debug, Clone)]
pub struct ChannelCapabilities {
    /// Supported formats
    pub formats: Vec<OutputFormat>,
    /// Maximum file size
    pub max_file_size: Option<u64>,
    /// Delivery confirmation support
    pub delivery_confirmation: bool,
    /// Retry support
    pub retry_support: bool,
}

/// Storage filter for listing reports
#[derive(Debug, Clone)]
pub struct StorageFilter {
    /// Time range filter
    pub time_range: Option<TimeRange>,
    /// Report type filter
    pub report_type: Option<ReportType>,
    /// Author filter
    pub author: Option<String>,
    /// Tag filter
    pub tags: Vec<String>,
}

/// Storage entry for stored reports
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageEntry {
    /// Report identifier
    pub report_id: String,
    /// Storage path or key
    pub path: String,
    /// File size
    pub size: u64,
    /// Storage timestamp
    pub stored_at: DateTime<Utc>,
    /// Report metadata
    pub metadata: Map<String, Value>,
}

/// Widget output for dashboard rendering
#[derive(Debug, Clone)]
pub struct WidgetOutput {
    /// Rendered content
    pub content: String,
    /// Content type
    pub content_type: String,
    /// Widget state
    pub state: Map<String, Value>,
}

/// Widget metadata description
#[derive(Debug, Clone)]
pub struct WidgetMetadata {
    /// Widget type identifier
    pub type_id: String,
    /// Widget name
    pub name: String,
    /// Widget description
    pub description: String,
    /// Configuration schema
    pub config_schema: Value,
}

/// Function signature for template functions
#[derive(Debug, Clone)]
pub struct FunctionSignature {
    /// Function name
    pub name: String,
    /// Parameter types
    pub parameters: Vec<VariableType>,
    /// Return type
    pub return_type: VariableType,
    /// Function description
    pub description: String,
}

// Implementation placeholders for complex types
pub struct CronEngine;
pub struct ThreadPool;
pub struct DeliveryTracker;
pub struct RetryManager;
pub struct StorageIndex;
pub struct CompressionSettings;
pub struct TemplateRenderer;
pub struct TemplateValidator;
pub struct DataStreamer;
pub struct DashboardStateManager;
pub struct AccessPolicy;
pub struct AuditTrail;
pub struct EncryptionSettings;
pub struct PrivacySettings;
pub struct ReportCache;
pub struct ReportRegistry;
pub struct DistributionRule;

impl fmt::Debug for CronEngine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CronEngine")
    }
}

impl fmt::Debug for ThreadPool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ThreadPool")
    }
}

impl fmt::Debug for DeliveryTracker {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DeliveryTracker")
    }
}

impl fmt::Debug for RetryManager {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RetryManager")
    }
}

impl fmt::Debug for StorageIndex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "StorageIndex")
    }
}

impl fmt::Debug for CompressionSettings {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CompressionSettings")
    }
}

impl fmt::Debug for TemplateRenderer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TemplateRenderer")
    }
}

impl fmt::Debug for TemplateValidator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TemplateValidator")
    }
}

impl fmt::Debug for DataStreamer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DataStreamer")
    }
}

impl fmt::Debug for DashboardStateManager {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DashboardStateManager")
    }
}

impl fmt::Debug for AccessPolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "AccessPolicy")
    }
}

impl fmt::Debug for AuditTrail {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "AuditTrail")
    }
}

impl fmt::Debug for EncryptionSettings {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "EncryptionSettings")
    }
}

impl fmt::Debug for PrivacySettings {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PrivacySettings")
    }
}

impl fmt::Debug for ReportCache {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ReportCache")
    }
}

impl fmt::Debug for ReportRegistry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ReportRegistry")
    }
}

impl fmt::Debug for DistributionRule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DistributionRule")
    }
}

impl ReportingSystem {
    /// Create a new reporting system
    pub fn new() -> Self {
        Self {
            generator: Arc::new(ReportGenerator::new()),
            scheduler: Arc::new(ReportScheduler::new()),
            distributor: Arc::new(ReportDistributor::new()),
            storage: Arc::new(ReportStorage::new()),
            template_manager: Arc::new(TemplateManager::new()),
            dashboard: Arc::new(DashboardEngine::new()),
            security: Arc::new(ReportSecurity::new()),
            cache: Arc::new(Mutex::new(ReportCache::new())),
            registry: Arc::new(Mutex::new(ReportRegistry::new())),
        }
    }

    /// Generate a report based on definition
    pub fn generate_report(&self, definition: &ReportDefinition, parameters: &Map<String, Value>) -> Result<GeneratedReport, ReportingError> {
        // Report generation implementation would go here
        // This is a placeholder
        Ok(GeneratedReport {
            id: Uuid::new_v4().to_string(),
            definition_id: definition.id.clone(),
            generated_at: Utc::now(),
            status: ReportStatus::Completed,
            outputs: vec![],
            parameters: parameters.clone(),
            metrics: GenerationMetrics {
                duration: Duration::from_secs(1),
                memory_usage: 1024,
                rows_processed: 100,
                cache_hit_rate: 0.75,
            },
            error: None,
        })
    }

    /// Schedule a report for automated generation
    pub fn schedule_report(&self, schedule: ScheduledReport) -> Result<(), ReportingError> {
        self.scheduler.add_schedule(schedule)
    }

    /// Create or update a dashboard
    pub fn create_dashboard(&self, dashboard: Dashboard) -> Result<(), ReportingError> {
        self.dashboard.create_dashboard(dashboard)
    }

    /// Get generated report by ID
    pub fn get_report(&self, report_id: &str) -> Result<GeneratedReport, ReportingError> {
        // Implementation would retrieve from storage/registry
        Err(ReportingError::Other("Not implemented".to_string()))
    }

    /// List reports based on filter criteria
    pub fn list_reports(&self, filter: &StorageFilter) -> Result<Vec<GeneratedReport>, ReportingError> {
        // Implementation would query storage/registry
        Err(ReportingError::Other("Not implemented".to_string()))
    }
}

impl ReportGenerator {
    fn new() -> Self {
        Self {
            aggregators: HashMap::new(),
            builders: HashMap::new(),
            chart_generators: HashMap::new(),
            exporters: HashMap::new(),
        }
    }
}

impl ReportScheduler {
    fn new() -> Self {
        Self {
            scheduled_reports: Arc::new(RwLock::new(HashMap::new())),
            cron_engine: Arc::new(CronEngine),
            thread_pool: Arc::new(ThreadPool),
            execution_history: Arc::new(Mutex::new(VecDeque::new())),
        }
    }

    fn add_schedule(&self, schedule: ScheduledReport) -> Result<(), ReportingError> {
        let mut schedules = self.scheduled_reports.write().unwrap();
        schedules.insert(schedule.id.clone(), schedule);
        Ok(())
    }
}

impl ReportDistributor {
    fn new() -> Self {
        Self {
            channels: HashMap::new(),
            rules: Vec::new(),
            delivery_tracker: Arc::new(Mutex::new(DeliveryTracker)),
            retry_manager: Arc::new(RetryManager),
        }
    }
}

impl ReportStorage {
    fn new() -> Self {
        Self {
            primary_storage: Box::new(FileSystemStorage::new()),
            archival_storage: None,
            metadata_index: Arc::new(RwLock::new(StorageIndex)),
            compression: CompressionSettings,
        }
    }
}

impl TemplateManager {
    fn new() -> Self {
        Self {
            templates: Arc::new(RwLock::new(HashMap::new())),
            renderer: Arc::new(TemplateRenderer),
            validator: Arc::new(TemplateValidator),
            custom_functions: HashMap::new(),
        }
    }
}

impl DashboardEngine {
    fn new() -> Self {
        Self {
            dashboards: Arc::new(RwLock::new(HashMap::new())),
            widgets: HashMap::new(),
            data_streamer: Arc::new(DataStreamer),
            state_manager: Arc::new(DashboardStateManager),
        }
    }

    fn create_dashboard(&self, dashboard: Dashboard) -> Result<(), ReportingError> {
        let mut dashboards = self.dashboards.write().unwrap();
        dashboards.insert(dashboard.id.clone(), dashboard);
        Ok(())
    }
}

impl ReportSecurity {
    fn new() -> Self {
        Self {
            access_policies: Vec::new(),
            audit_trail: Arc::new(Mutex::new(AuditTrail)),
            encryption: EncryptionSettings,
            privacy: PrivacySettings,
        }
    }
}

impl ReportCache {
    fn new() -> Self {
        Self
    }
}

impl ReportRegistry {
    fn new() -> Self {
        Self
    }
}

// Placeholder storage backend implementation
struct FileSystemStorage;

impl FileSystemStorage {
    fn new() -> Self {
        Self
    }
}

impl StorageBackend for FileSystemStorage {
    fn store(&self, report: &GeneratedReport, data: &[u8]) -> Result<String, ReportingError> {
        // File system storage implementation
        Ok(format!("stored_{}", report.id))
    }

    fn retrieve(&self, report_id: &str) -> Result<Vec<u8>, ReportingError> {
        // File system retrieval implementation
        Ok(vec![])
    }

    fn delete(&self, report_id: &str) -> Result<(), ReportingError> {
        // File system deletion implementation
        Ok(())
    }

    fn list(&self, filter: &StorageFilter) -> Result<Vec<StorageEntry>, ReportingError> {
        // File system listing implementation
        Ok(vec![])
    }
}

impl Default for ReportingSystem {
    fn default() -> Self {
        Self::new()
    }
}

// Re-export common types for convenience
pub use self::{
    ReportingSystem,
    ReportDefinition,
    ReportType,
    GeneratedReport,
    ReportStatus,
    ScheduledReport,
    Dashboard,
    ReportTemplate,
    ReportingError,
};