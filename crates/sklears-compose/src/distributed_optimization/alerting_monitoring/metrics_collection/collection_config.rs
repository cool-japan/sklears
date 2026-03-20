//! Collection Configuration Management
//!
//! This module handles collection source configurations, authentication,
//! connection management, and collection orchestration settings.

use std::collections::HashMap;
use std::time::Duration;
use serde::{Deserialize, Serialize};
use super::metrics_core::CollectionMethod;

/// Collection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionConfiguration {
    pub collection_interval: Duration,
    pub collection_method: CollectionMethod,
    pub collection_sources: Vec<CollectionSource>,
    pub batch_size: Option<u32>,
    pub timeout: Duration,
    pub retry_config: RetryConfiguration,
    pub buffer_config: BufferConfiguration,
    pub deduplication: DeduplicationConfig,
}

/// Collection sources
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionSource {
    pub source_id: String,
    pub source_type: SourceType,
    pub connection_config: ConnectionConfig,
    pub query_config: QueryConfig,
    pub authentication: AuthenticationConfig,
    pub filters: Vec<SourceFilter>,
    pub weight: f64,
    pub enabled: bool,
}

/// Source types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SourceType {
    Database(DatabaseSourceType),
    API(ApiSourceType),
    File(FileSourceType),
    JMX(JmxSourceType),
    SNMP(SnmpSourceType),
    Prometheus(PrometheusSourceType),
    StatsD(StatsDSourceType),
    System(SystemSourceType),
    Application(ApplicationSourceType),
    Custom(String),
}

/// Database source types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DatabaseSourceType {
    MySQL,
    PostgreSQL,
    MongoDB,
    InfluxDB,
    Elasticsearch,
    Redis,
    Cassandra,
    Oracle,
    SQLServer,
    Custom(String),
}

/// API source types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ApiSourceType {
    REST,
    GraphQL,
    SOAP,
    gRPC,
    WebSocket,
    Custom(String),
}

/// File source types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FileSourceType {
    CSV,
    JSON,
    XML,
    Log,
    Binary,
    Custom(String),
}

/// JMX source types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JmxSourceType {
    pub mbean_pattern: String,
    pub attributes: Vec<String>,
    pub composite_attributes: Vec<CompositeAttribute>,
}

/// Composite attributes for JMX
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompositeAttribute {
    pub attribute_name: String,
    pub composite_keys: Vec<String>,
}

/// SNMP source types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnmpSourceType {
    pub version: SnmpVersion,
    pub community: Option<String>,
    pub oids: Vec<String>,
    pub walk_oids: Vec<String>,
}

/// SNMP versions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SnmpVersion {
    V1,
    V2c,
    V3,
}

/// Prometheus source types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrometheusSourceType {
    pub endpoint: String,
    pub metrics_path: String,
    pub query: Option<String>,
    pub labels: HashMap<String, String>,
}

/// StatsD source types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatsDSourceType {
    pub host: String,
    pub port: u16,
    pub protocol: StatsDProtocol,
    pub prefix: Option<String>,
    pub tags: HashMap<String, String>,
}

/// StatsD protocols
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StatsDProtocol {
    UDP,
    TCP,
    Unix,
}

/// System source types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SystemSourceType {
    CPU,
    Memory,
    Disk,
    Network,
    Process,
    Service,
    Performance,
    EventLog,
    Custom(String),
}

/// Application source types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ApplicationSourceType {
    Java(JavaSourceConfig),
    DotNet(DotNetSourceConfig),
    Python(PythonSourceConfig),
    NodeJS(NodeJSSourceConfig),
    Go(GoSourceConfig),
    Custom(String),
}

/// Java source configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JavaSourceConfig {
    pub jmx_enabled: bool,
    pub gc_metrics: bool,
    pub thread_metrics: bool,
    pub heap_metrics: bool,
    pub class_loader_metrics: bool,
}

/// .NET source configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DotNetSourceConfig {
    pub performance_counters: Vec<String>,
    pub clr_metrics: bool,
    pub gc_metrics: bool,
    pub thread_metrics: bool,
    pub memory_metrics: bool,
}

/// Python source configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PythonSourceConfig {
    pub psutil_enabled: bool,
    pub gc_metrics: bool,
    pub memory_profiling: bool,
    pub custom_metrics: Vec<String>,
}

/// Node.js source configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeJSSourceConfig {
    pub v8_metrics: bool,
    pub event_loop_metrics: bool,
    pub gc_metrics: bool,
    pub heap_metrics: bool,
}

/// Go source configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoSourceConfig {
    pub runtime_metrics: bool,
    pub gc_metrics: bool,
    pub goroutine_metrics: bool,
    pub memory_metrics: bool,
}

/// Connection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectionConfig {
    pub host: String,
    pub port: Option<u16>,
    pub username: Option<String>,
    pub password: Option<String>,
    pub ssl_enabled: bool,
    pub timeout: Duration,
    pub keep_alive: bool,
    pub connection_pool: ConnectionPoolConfig,
}

/// Connection pool configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectionPoolConfig {
    pub min_connections: u32,
    pub max_connections: u32,
    pub connection_timeout: Duration,
    pub idle_timeout: Duration,
    pub max_lifetime: Duration,
    pub test_on_borrow: bool,
}

/// Query configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryConfig {
    pub query: String,
    pub parameters: HashMap<String, String>,
    pub result_format: ResultFormat,
    pub pagination: Option<PaginationConfig>,
    pub caching: CacheConfig,
    pub transformation: Option<TransformationConfig>,
}

/// Result formats
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResultFormat {
    JSON,
    XML,
    CSV,
    Binary,
    Custom(String),
}

/// Pagination configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaginationConfig {
    pub page_size: u32,
    pub offset_field: String,
    pub limit_field: String,
    pub total_count_field: Option<String>,
}

/// Cache configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    pub enabled: bool,
    pub ttl: Duration,
    pub max_size: u64,
    pub eviction_policy: EvictionPolicy,
}

/// Eviction policies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EvictionPolicy {
    LRU,
    LFU,
    FIFO,
    Random,
    TTL,
}

/// Transformation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformationConfig {
    pub transformations: Vec<DataTransformation>,
    pub validation_enabled: bool,
    pub error_handling: ErrorHandling,
}

/// Data transformations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DataTransformation {
    Map(MappingTransformation),
    Filter(FilterTransformation),
    Aggregate(AggregateTransformation),
    Join(JoinTransformation),
    Normalize(NormalizationTransformation),
    Custom(String),
}

/// Mapping transformation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MappingTransformation {
    pub field_mappings: HashMap<String, String>,
    pub value_mappings: HashMap<String, HashMap<String, String>>,
    pub default_values: HashMap<String, String>,
}

/// Filter transformation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilterTransformation {
    pub conditions: Vec<FilterCondition>,
    pub operator: LogicalOperator,
}

/// Filter conditions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilterCondition {
    pub field: String,
    pub operator: ComparisonOperator,
    pub value: String,
    pub data_type: DataType,
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
    Contains,
    StartsWith,
    EndsWith,
    Matches,
    In,
    NotIn,
    IsNull,
    IsNotNull,
}

/// Logical operators
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LogicalOperator {
    AND,
    OR,
    NOT,
    XOR,
}

/// Data types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DataType {
    String,
    Integer,
    Float,
    Boolean,
    Date,
    DateTime,
    Duration,
    Bytes,
    Array,
    Object,
}

/// Aggregate transformation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregateTransformation {
    pub group_by: Vec<String>,
    pub aggregations: Vec<AggregationFunction>,
    pub window: Option<WindowConfig>,
}

/// Aggregation functions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregationFunction {
    pub function: AggregationType,
    pub field: String,
    pub alias: Option<String>,
    pub parameters: HashMap<String, String>,
}

/// Aggregation types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AggregationType {
    Count,
    Sum,
    Average,
    Min,
    Max,
    StdDev,
    Variance,
    Median,
    Percentile(f64),
    Mode,
    Range,
    First,
    Last,
    Distinct,
    Custom(String),
}

/// Window configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WindowConfig {
    pub window_type: WindowType,
    pub size: WindowSize,
    pub advance: WindowAdvance,
    pub alignment: WindowAlignment,
}

/// Window types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WindowType {
    Tumbling,
    Sliding,
    Session,
    Custom(String),
}

/// Window sizes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WindowSize {
    Time(Duration),
    Count(u64),
    Bytes(u64),
    Custom(String),
}

/// Window advance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WindowAdvance {
    Time(Duration),
    Count(u64),
    Bytes(u64),
    Custom(String),
}

/// Window alignment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WindowAlignment {
    Start,
    End,
    Center,
    Custom(Duration),
}

/// Join transformation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JoinTransformation {
    pub join_type: JoinType,
    pub left_key: String,
    pub right_key: String,
    pub right_source: String,
    pub projection: Vec<String>,
}

/// Join types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum JoinType {
    Inner,
    Left,
    Right,
    Full,
    Cross,
}

/// Normalization transformation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormalizationTransformation {
    pub normalization_type: NormalizationType,
    pub fields: Vec<String>,
    pub parameters: HashMap<String, f64>,
}

/// Normalization types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NormalizationType {
    MinMax,
    ZScore,
    Robust,
    Unit,
    Decimal,
    Custom(String),
}

/// Error handling strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ErrorHandling {
    Ignore,
    Skip,
    Default(String),
    Fail,
    Log,
    Retry(RetryConfig),
}

/// Retry configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryConfig {
    pub max_attempts: u32,
    pub delay: Duration,
    pub backoff: BackoffStrategy,
    pub conditions: Vec<RetryCondition>,
}

/// Backoff strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BackoffStrategy {
    Fixed,
    Linear,
    Exponential { base: f64, cap: Duration },
    Custom(String),
}

/// Retry conditions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RetryCondition {
    NetworkError,
    Timeout,
    ServerError,
    RateLimit,
    Custom(String),
}

/// Authentication configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthenticationConfig {
    pub auth_type: AuthType,
    pub credentials: CredentialsConfig,
    pub token_config: Option<TokenConfig>,
    pub certificate_config: Option<CertificateConfig>,
}

/// Authentication types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuthType {
    None,
    Basic,
    Bearer,
    ApiKey,
    OAuth2,
    JWT,
    Certificate,
    Custom(String),
}

/// Credentials configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CredentialsConfig {
    pub username: Option<String>,
    pub password: Option<String>,
    pub api_key: Option<String>,
    pub secret: Option<String>,
    pub token: Option<String>,
}

/// Token configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenConfig {
    pub token_url: String,
    pub refresh_url: Option<String>,
    pub scope: Vec<String>,
    pub expiration_buffer: Duration,
    pub auto_refresh: bool,
}

/// Certificate configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CertificateConfig {
    pub cert_path: String,
    pub key_path: String,
    pub ca_path: Option<String>,
    pub verify_ssl: bool,
}

/// Source filters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceFilter {
    pub filter_id: String,
    pub field: String,
    pub operator: ComparisonOperator,
    pub value: String,
    pub enabled: bool,
}

/// Retry configuration for collection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryConfiguration {
    pub enabled: bool,
    pub max_retries: u32,
    pub initial_delay: Duration,
    pub max_delay: Duration,
    pub backoff_multiplier: f64,
    pub retry_conditions: Vec<RetryCondition>,
}

/// Buffer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BufferConfiguration {
    pub buffer_size: u32,
    pub flush_interval: Duration,
    pub flush_threshold: u32,
    pub overflow_strategy: OverflowStrategy,
    pub compression: CompressionConfig,
}

/// Overflow strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OverflowStrategy {
    Drop,
    Block,
    Backpressure,
    Spill,
    Custom(String),
}

/// Compression configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionConfig {
    pub enabled: bool,
    pub algorithm: CompressionAlgorithm,
    pub level: u32,
    pub threshold: u32,
}

/// Compression algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompressionAlgorithm {
    None,
    GZIP,
    LZ4,
    Snappy,
    ZSTD,
    Custom(String),
}

/// Deduplication configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeduplicationConfig {
    pub enabled: bool,
    pub key_fields: Vec<String>,
    pub time_window: Duration,
    pub dedup_strategy: DeduplicationStrategy,
}

/// Deduplication strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DeduplicationStrategy {
    First,
    Last,
    Merge,
    Custom(String),
}