//! Data Management Configuration
//!
//! This module contains all configuration structures related to data management
//! including retention policies, export capabilities, and sampling strategies.
//! It provides comprehensive control over monitoring data lifecycle and export.

use std::collections::HashMap;
use std::time::Duration;

/// Data retention configuration
///
/// Controls how long monitoring data is retained at different granularities
/// and manages the lifecycle of monitoring data from collection to archival.
///
/// # Architecture
///
/// The data retention system operates on multiple time horizons:
///
/// ```text
/// Data Retention Lifecycle
/// ├── Raw Data (high frequency, short retention)
/// ├── Aggregated Data (medium frequency, medium retention)
/// ├── Summary Data (low frequency, long retention)
/// ├── Archive Data (compressed, very long retention)
/// └── Purged Data (deleted after retention period)
/// ```
///
/// # Retention Strategies
///
/// Different retention strategies are applied based on data type and importance:
/// - **Hot storage**: Recent, frequently accessed data
/// - **Warm storage**: Older data that may be accessed occasionally
/// - **Cold storage**: Archive data for compliance and historical analysis
/// - **Purged**: Data that has exceeded retention period and is deleted
///
/// # Usage Examples
///
/// ## Production Data Retention
/// ```rust
/// use sklears_compose::monitoring_config::DataRetentionConfig;
///
/// let config = DataRetentionConfig::production();
/// ```
///
/// ## Compliance-Focused Retention
/// ```rust
/// let config = DataRetentionConfig::compliance_focused();
/// ```
///
/// ## Development Retention
/// ```rust
/// let config = DataRetentionConfig::development();
/// ```
#[derive(Debug, Clone)]
pub struct DataRetentionConfig {
    /// Metrics data retention period
    ///
    /// How long to retain raw metrics data before archival or deletion.
    pub metrics_retention: Duration,

    /// Events data retention period
    ///
    /// How long to retain event data for audit trails and analysis.
    pub events_retention: Duration,

    /// Logs data retention period
    ///
    /// How long to retain log data for debugging and troubleshooting.
    pub logs_retention: Duration,

    /// Traces data retention period
    ///
    /// How long to retain distributed tracing data for performance analysis.
    pub traces_retention: Duration,

    /// Retention policies for different data types
    ///
    /// Specific retention rules for different categories of monitoring data.
    pub policies: Vec<RetentionPolicy>,

    /// Cleanup configuration
    ///
    /// Settings for automated cleanup of expired data.
    pub cleanup: CleanupConfig,

    /// Archive configuration
    ///
    /// Settings for long-term archival of monitoring data.
    pub archive: ArchiveConfig,
}

/// Retention policy for specific data types
///
/// Defines retention rules for specific categories of monitoring data
/// with different retention periods and cleanup strategies.
#[derive(Debug, Clone)]
pub struct RetentionPolicy {
    /// Policy name for identification
    pub name: String,

    /// Data type this policy applies to
    pub data_type: String,

    /// Retention duration for this data type
    pub retention_duration: Duration,

    /// Storage tier for this data type
    pub storage_tier: StorageTier,

    /// Conditions for applying this policy
    pub conditions: Vec<RetentionCondition>,

    /// Policy priority (higher numbers take precedence)
    pub priority: u32,

    /// Custom metadata for the policy
    pub metadata: HashMap<String, String>,
}

/// Storage tiers for different data retention phases
#[derive(Debug, Clone)]
pub enum StorageTier {
    /// Hot storage for frequently accessed recent data
    Hot,
    /// Warm storage for occasionally accessed older data
    Warm,
    /// Cold storage for archive data
    Cold,
    /// Custom storage tier
    Custom { tier_name: String },
}

/// Conditions for applying retention policies
#[derive(Debug, Clone)]
pub struct RetentionCondition {
    /// Field to evaluate
    pub field: String,

    /// Comparison operator
    pub operator: ConditionOperator,

    /// Value to compare against
    pub value: String,
}

/// Operators for retention conditions
#[derive(Debug, Clone)]
pub enum ConditionOperator {
    /// Exact equality
    Equals,
    /// Field contains the value
    Contains,
    /// Field matches regular expression
    Regex,
    /// Field is greater than value
    GreaterThan,
    /// Field is less than value
    LessThan,
}

/// Cleanup configuration for expired data
///
/// Controls automated cleanup processes that remove or archive
/// expired monitoring data according to retention policies.
#[derive(Debug, Clone)]
pub struct CleanupConfig {
    /// Enable automatic cleanup
    pub enabled: bool,

    /// Cleanup execution frequency
    pub frequency: Duration,

    /// Cleanup strategy to use
    pub strategy: CleanupStrategy,

    /// Cleanup batch size
    ///
    /// Number of records to process in each cleanup batch.
    pub batch_size: usize,

    /// Cleanup execution timeout
    pub timeout: Duration,

    /// Cleanup thresholds
    pub thresholds: CleanupThresholds,

    /// Cleanup verification settings
    pub verification: CleanupVerification,
}

/// Cleanup strategies for different scenarios
#[derive(Debug, Clone)]
pub enum CleanupStrategy {
    TimeBased,

    SizeBased { max_size: u64 },

    /// Count-based cleanup (keep only N most recent records)
    CountBased { max_records: u64 },

    /// Priority-based cleanup (delete low-priority data first)
    PriorityBased { priority_threshold: u32 },

    /// Custom cleanup strategy
    Custom {
        strategy_name: String,
        parameters: HashMap<String, String>,
    },
}

/// Thresholds that trigger cleanup operations
#[derive(Debug, Clone)]
pub struct CleanupThresholds {
    /// Maximum age before cleanup
    pub max_age: Duration,

    /// Maximum storage size before cleanup
    pub max_size: u64,

    /// Maximum record count before cleanup
    pub max_count: u64,

    /// Disk usage threshold for emergency cleanup
    pub disk_usage_threshold: f64,
}

/// Cleanup verification settings
#[derive(Debug, Clone)]
pub struct CleanupVerification {
    /// Enable cleanup verification
    pub enabled: bool,

    /// Verification sample rate (0.0 to 1.0)
    pub sample_rate: f64,

    /// Maximum verification failures before stopping
    pub max_failures: u32,

    /// Verification timeout
    pub timeout: Duration,
}

/// Archive configuration for long-term data storage
///
/// Controls how monitoring data is archived for long-term retention,
/// compliance, and historical analysis.
#[derive(Debug, Clone)]
pub struct ArchiveConfig {
    /// Enable archiving
    pub enabled: bool,

    /// Archive storage backend
    pub storage: ArchiveStorage,

    /// Archive frequency
    pub frequency: Duration,

    /// Archive compression settings
    pub compression: CompressionConfig,

    /// Archive encryption settings
    pub encryption: EncryptionConfig,

    /// Archive verification settings
    pub verification: ArchiveVerification,

    /// Archive metadata settings
    pub metadata: ArchiveMetadata,
}

/// Archive storage backends
#[derive(Debug, Clone)]
pub enum ArchiveStorage {
    /// Local file system storage
    LocalFilesystem {
        /// Base directory for archives
        base_path: String,
        /// Directory structure pattern
        path_pattern: String,
    },

    /// Cloud storage (S3, GCS, Azure Blob)
    CloudStorage {
        /// Cloud provider
        provider: String,
        /// Bucket or container name
        bucket: String,
        /// Access credentials
        credentials: CloudCredentials,
    },

    /// Database storage
    Database {
        /// Database connection string
        connection: String,
        /// Table or collection name
        table_name: String,
    },

    /// Network attached storage
    NetworkStorage {
        /// NFS/SMB mount point
        mount_point: String,
        /// Storage protocol
        protocol: String,
    },

    /// Custom storage backend
    Custom {
        /// Backend type identifier
        backend_type: String,
        /// Configuration parameters
        config: HashMap<String, String>,
    },
}

/// Cloud storage credentials
#[derive(Debug, Clone)]
pub struct CloudCredentials {
    /// Access key or client ID
    pub access_key: String,
    /// Secret key or client secret
    pub secret_key: String,
    /// Additional configuration (region, endpoint, etc.)
    pub config: HashMap<String, String>,
}

/// Compression configuration for archived data
#[derive(Debug, Clone)]
pub struct CompressionConfig {
    /// Enable compression
    pub enabled: bool,

    /// Compression algorithm
    pub algorithm: CompressionAlgorithm,

    /// Compression level (0-9, higher = more compression)
    pub level: u8,

    /// Compression block size
    pub block_size: usize,
}

/// Compression algorithms for data archival
#[derive(Debug, Clone)]
pub enum CompressionAlgorithm {
    /// GZIP compression (widely compatible)
    Gzip,
    /// LZ4 compression (fast compression/decompression)
    Lz4,
    /// Zstandard compression (balanced performance)
    Zstd,
    /// BZIP2 compression (high compression ratio)
    Bzip2,
    /// XZ compression (highest compression ratio)
    Xz,
    /// Custom compression algorithm
    Custom { algorithm_name: String },
}

/// Encryption configuration for archived data
#[derive(Debug, Clone)]
pub struct EncryptionConfig {
    /// Enable encryption
    pub enabled: bool,

    /// Encryption algorithm
    pub algorithm: EncryptionAlgorithm,

    /// Key management system
    pub key_management: KeyManagement,

    /// Encryption metadata
    pub metadata: EncryptionMetadata,
}

/// Encryption algorithms for data protection
#[derive(Debug, Clone)]
pub enum EncryptionAlgorithm {
    /// AES-256 with GCM mode
    Aes256Gcm,
    /// AES-256 with CBC mode
    Aes256Cbc,
    /// ChaCha20-Poly1305
    ChaCha20Poly1305,
    /// Custom encryption algorithm
    Custom { algorithm_name: String },
}

/// Key management systems for encryption
#[derive(Debug, Clone)]
pub enum KeyManagement {
    /// Static key (not recommended for production)
    StaticKey { key: String },
    /// Key derivation from passphrase
    PassphraseDerivation { passphrase: String, salt: String },
    /// External key management service
    ExternalKms { service: String, key_id: String },
    /// Hardware security module
    Hsm { device: String, key_slot: u32 },
}

/// Encryption metadata configuration
#[derive(Debug, Clone)]
pub struct EncryptionMetadata {
    /// Include encryption metadata in archives
    pub include_metadata: bool,
    /// Metadata fields to include
    pub metadata_fields: Vec<String>,
}

/// Archive verification configuration
#[derive(Debug, Clone)]
pub struct ArchiveVerification {
    /// Enable archive verification
    pub enabled: bool,

    /// Verification methods
    pub methods: Vec<VerificationMethod>,

    /// Verification frequency
    pub frequency: Duration,

    /// Verification sample rate
    pub sample_rate: f64,
}

/// Methods for verifying archive integrity
#[derive(Debug, Clone)]
pub enum VerificationMethod {
    /// Checksum verification (MD5, SHA256, etc.)
    Checksum { algorithm: String },
    /// Digital signature verification
    DigitalSignature { public_key: String },
    /// Content validation (structure, format)
    ContentValidation,
    /// Size verification
    SizeVerification,
    /// Custom verification method
    Custom { method_name: String },
}

/// Archive metadata configuration
#[derive(Debug, Clone)]
pub struct ArchiveMetadata {
    /// Include metadata with archives
    pub enabled: bool,

    /// Metadata format
    pub format: MetadataFormat,

    /// Metadata fields to include
    pub fields: Vec<String>,

    /// Custom metadata generators
    pub generators: Vec<MetadataGenerator>,
}

/// Formats for archive metadata
#[derive(Debug, Clone)]
pub enum MetadataFormat {
    /// JSON format
    Json,
    /// XML format
    Xml,
    /// YAML format
    Yaml,
    /// Custom format
    Custom { format_name: String },
}

/// Metadata generators for archives
#[derive(Debug, Clone)]
pub struct MetadataGenerator {
    /// Generator name
    pub name: String,
    /// Fields this generator produces
    pub fields: Vec<String>,
    /// Generator configuration
    pub config: HashMap<String, String>,
}

/// Export configuration for monitoring data
///
/// Controls how monitoring data is exported to external systems
/// for integration, analysis, and backup purposes.
///
/// # Export Formats
///
/// Multiple export formats are supported:
/// - **JSON**: Human-readable, widely compatible
/// - **CSV**: Tabular data for spreadsheet analysis
/// - **Parquet**: Columnar format for analytics
/// - **Avro**: Schema evolution support
/// - **Protocol Buffers**: Efficient binary format
///
/// # Export Destinations
///
/// Data can be exported to various destinations:
/// - File systems (local or network)
/// - Databases (SQL or NoSQL)
/// - Message queues (Kafka, RabbitMQ)
/// - Cloud storage (S3, GCS, Azure)
/// - APIs (REST, GraphQL)
///
/// # Usage Examples
///
/// ## Analytics Export Configuration
/// ```rust
/// use sklears_compose::monitoring_config::ExportConfig;
///
/// let config = ExportConfig::analytics_focused();
/// ```
///
/// ## Real-time Export Configuration
/// ```rust
/// let config = ExportConfig::real_time();
/// ```
#[derive(Debug, Clone)]
pub struct ExportConfig {
    /// Enable data export
    pub enabled: bool,

    /// Export formats to use
    pub formats: Vec<ExportFormat>,

    /// Export destinations
    pub destinations: Vec<ExportDestination>,

    /// Export scheduling configuration
    pub scheduling: ExportSchedulingConfig,

    /// Export filtering configuration
    pub filtering: ExportFilteringConfig,

    /// Export transformation configuration
    pub transformation: ExportTransformationConfig,

    /// Export monitoring and reliability
    pub monitoring: ExportMonitoringConfig,
}

/// Export formats for monitoring data
#[derive(Debug, Clone)]
pub enum ExportFormat {
    Json {
        pretty_print: bool,
        compression: Option<CompressionConfig>,
    },

    /// CSV format for tabular data
    Csv {
        /// Field delimiter
        delimiter: char,
        /// Include headers
        include_headers: bool,
        /// Quote character
        quote_char: char,
    },

    /// Apache Parquet columnar format
    Parquet {
        /// Compression codec
        compression: String,
        /// Row group size
        row_group_size: usize,
    },

    /// Apache Avro format with schema
    Avro {
        /// Schema definition
        schema: String,
        /// Compression codec
        compression: String,
    },

    /// Protocol Buffers binary format
    Protobuf {
        /// Message schema definition
        schema: String,
        /// Compression settings
        compression: Option<CompressionConfig>,
    },

    /// MessagePack binary format
    MessagePack {
        /// Compression settings
        compression: Option<CompressionConfig>,
    },

    /// Custom export format
    Custom {
        /// Format name
        format_name: String,
        /// Format configuration
        config: HashMap<String, String>,
    },
}

/// Export destinations for monitoring data
#[derive(Debug, Clone)]
pub enum ExportDestination {
    /// Local or network file system
    Filesystem {
        /// Base directory path
        base_path: String,
        /// File naming pattern
        file_pattern: String,
        /// File rotation settings
        rotation: FileRotationConfig,
    },

    /// Database destination
    Database {
        /// Database connection string
        connection: String,
        /// Target table or collection
        table: String,
        /// Insert strategy
        insert_strategy: DatabaseInsertStrategy,
    },

    /// Message queue destination
    MessageQueue {
        /// Queue connection string
        connection: String,
        /// Topic or queue name
        topic: String,
        /// Message partitioning
        partitioning: Option<PartitioningConfig>,
    },

    /// Cloud storage destination
    CloudStorage {
        /// Cloud provider configuration
        provider: CloudCredentials,
        /// Bucket or container
        bucket: String,
        /// Object key pattern
        key_pattern: String,
    },

    /// HTTP/REST API destination
    RestApi {
        /// API endpoint URL
        url: String,
        /// HTTP method
        method: String,
        /// Request headers
        headers: HashMap<String, String>,
        /// Authentication
        auth: Option<ApiAuthentication>,
    },

    /// Custom export destination
    Custom {
        /// Destination type
        destination_type: String,
        /// Configuration parameters
        config: HashMap<String, String>,
    },
}

/// File rotation configuration for filesystem exports
#[derive(Debug, Clone)]
pub struct FileRotationConfig {
    /// Enable file rotation
    pub enabled: bool,
    /// Maximum file size before rotation
    pub max_size: u64,
    /// Maximum file age before rotation
    pub max_age: Duration,
    /// Maximum number of files to keep
    pub max_files: u32,
    /// Compression for rotated files
    pub compress_rotated: bool,
}

/// Database insert strategies
#[derive(Debug, Clone)]
pub enum DatabaseInsertStrategy {
    /// Insert new records only
    Insert,
    /// Insert or update existing records
    Upsert,
    /// Batch insert for efficiency
    BatchInsert { batch_size: usize },
    /// Bulk copy/load
    BulkLoad,
}

/// Message partitioning configuration
#[derive(Debug, Clone)]
pub struct PartitioningConfig {
    /// Partitioning strategy
    pub strategy: PartitioningStrategy,
    /// Number of partitions
    pub partition_count: u32,
    /// Partitioning key field
    pub key_field: String,
}

/// Partitioning strategies for message queues
#[derive(Debug, Clone)]
pub enum PartitioningStrategy {
    /// Hash-based partitioning
    Hash,
    /// Round-robin partitioning
    RoundRobin,
    /// Key-based partitioning
    KeyBased,
    /// Custom partitioning
    Custom { strategy_name: String },
}

/// API authentication methods
#[derive(Debug, Clone)]
pub enum ApiAuthentication {
    /// Basic authentication
    Basic { username: String, password: String },
    /// Bearer token
    Bearer { token: String },
    /// API key authentication
    ApiKey { key: String, header: String },
    /// OAuth 2.0
    OAuth2 { client_id: String, client_secret: String, token_url: String },
}

/// Export scheduling configuration
#[derive(Debug, Clone)]
pub struct ExportSchedulingConfig {
    /// Enable scheduled exports
    pub enabled: bool,

    /// Export frequency
    pub frequency: ExportFrequency,

    /// Export time window
    pub time_window: Option<TimeWindow>,

    /// Maximum export size per batch
    pub max_size: Option<u64>,

    /// Export timeout
    pub timeout: Duration,

    /// Retry configuration
    pub retry: RetryConfig,
}

/// Export frequencies
#[derive(Debug, Clone)]
pub enum ExportFrequency {
    /// Fixed interval exports
    Interval(Duration),
    /// Cron-based scheduling
    Cron(String),
    /// Event-triggered exports
    EventTriggered { events: Vec<String> },
    /// Manual exports only
    Manual,
}

/// Time window for exports
#[derive(Debug, Clone)]
pub struct TimeWindow {
    /// Window start time (hour of day)
    pub start_hour: u8,
    /// Window end time (hour of day)
    pub end_hour: u8,
    /// Days of week (0=Sunday, 6=Saturday)
    pub days_of_week: Vec<u8>,
    /// Time zone
    pub timezone: String,
}

/// Retry configuration for failed exports
#[derive(Debug, Clone)]
pub struct RetryConfig {
    /// Enable retries
    pub enabled: bool,
    /// Maximum retry attempts
    pub max_attempts: u32,
    /// Retry delay strategy
    pub delay_strategy: RetryDelayStrategy,
    /// Maximum total retry time
    pub max_total_time: Duration,
}

/// Retry delay strategies
#[derive(Debug, Clone)]
pub enum RetryDelayStrategy {
    /// Fixed delay between retries
    Fixed(Duration),
    /// Exponential backoff
    ExponentialBackoff { base_delay: Duration, max_delay: Duration },
    /// Linear backoff
    LinearBackoff { base_delay: Duration, increment: Duration },
}

/// Export filtering configuration
#[derive(Debug, Clone)]
pub struct ExportFilteringConfig {
    /// Enable export filtering
    pub enabled: bool,

    /// Filtering rules
    pub filters: Vec<ExportFilter>,

    /// Include patterns (regex)
    pub include_patterns: Vec<String>,

    /// Exclude patterns (regex)
    pub exclude_patterns: Vec<String>,

    /// Data sampling configuration
    pub sampling: Option<ExportSamplingConfig>,
}

/// Export filter definition
#[derive(Debug, Clone)]
pub struct ExportFilter {
    /// Filter name
    pub name: String,
    /// Field to filter on
    pub field: String,
    /// Filter operator
    pub operator: FilterOperator,
    /// Filter value
    pub value: String,
    /// Filter action
    pub action: FilterAction,
}

/// Filter operators for export filtering
#[derive(Debug, Clone)]
pub enum FilterOperator {
    /// Exact equality
    Equals,
    /// Not equal
    NotEquals,
    /// Contains substring
    Contains,
    /// Matches regex pattern
    Regex,
    /// Greater than (numeric)
    GreaterThan,
    /// Less than (numeric)
    LessThan,
}

/// Filter actions for matching data
#[derive(Debug, Clone)]
pub enum FilterAction {
    /// Include in export
    Include,
    /// Exclude from export
    Exclude,
    /// Transform the data
    Transform { transformation: String },
}

/// Export sampling configuration
#[derive(Debug, Clone)]
pub struct ExportSamplingConfig {
    /// Sampling rate (0.0 to 1.0)
    pub rate: f64,
    /// Sampling strategy
    pub strategy: SamplingStrategy,
    /// Sampling seed for reproducibility
    pub seed: Option<u64>,
}

/// Export transformation configuration
#[derive(Debug, Clone)]
pub struct ExportTransformationConfig {
    /// Enable transformations
    pub enabled: bool,
    /// Transformation rules
    pub rules: Vec<TransformationRule>,
    /// Default transformations
    pub defaults: Vec<DefaultTransformation>,
}

/// Transformation rule for export data
#[derive(Debug, Clone)]
pub struct TransformationRule {
    /// Rule name
    pub name: String,
    /// Source field
    pub source_field: String,
    /// Target field
    pub target_field: String,
    /// Transformation function
    pub transformation: TransformationType,
    /// Conditions for applying transformation
    pub conditions: Vec<String>,
}

/// Types of data transformations
#[derive(Debug, Clone)]
pub enum TransformationType {
    /// Field renaming
    Rename { new_name: String },
    /// Data type conversion
    TypeConversion { target_type: String },
    /// Value mapping
    ValueMapping { mappings: HashMap<String, String> },
    /// Mathematical operation
    MathOperation { operation: String },
    /// String manipulation
    StringOperation { operation: String },
    /// Custom transformation
    Custom { function_name: String },
}

/// Default transformations applied to all exports
#[derive(Debug, Clone)]
pub struct DefaultTransformation {
    /// Transformation name
    pub name: String,
    /// Transformation configuration
    pub config: HashMap<String, String>,
}

/// Export monitoring configuration
#[derive(Debug, Clone)]
pub struct ExportMonitoringConfig {
    /// Enable export monitoring
    pub enabled: bool,
    /// Success/failure tracking
    pub track_results: bool,
    /// Performance metrics
    pub track_performance: bool,
    /// Error notifications
    pub error_notifications: Vec<String>,
    /// Monitoring retention
    pub monitoring_retention: Duration,
}

/// Sampling configuration for high-volume data
///
/// Controls sampling strategies to manage data volume while maintaining
/// statistical representativeness of the monitoring data.
///
/// # Sampling Strategies
///
/// Different sampling strategies serve different use cases:
/// - **Random**: Uniform random sampling across all data
/// - **Systematic**: Every Nth record for consistent intervals
/// - **Stratified**: Sampling within predefined groups
/// - **Reservoir**: Fixed-size sampling for streaming data
/// - **Adaptive**: Dynamic sampling based on system load
///
/// # Usage Examples
///
/// ## High-Volume Sampling
/// ```rust
/// use sklears_compose::monitoring_config::SamplingConfig;
///
/// let config = SamplingConfig::high_volume();
/// ```
///
/// ## Quality-Focused Sampling
/// ```rust
/// let config = SamplingConfig::quality_focused();
/// ```
#[derive(Debug, Clone)]
pub struct SamplingConfig {
    /// Enable sampling
    pub enabled: bool,

    /// Default sampling strategy
    pub strategy: SamplingStrategy,

    /// Default sampling rate (0.0 to 1.0)
    pub rate: f64,

    /// Adaptive sampling configuration
    pub adaptive: AdaptiveSamplingConfig,

    /// Stratified sampling configuration
    pub stratified: Option<StratifiedSamplingConfig>,

    /// Sampling quality controls
    pub quality: SamplingQualityConfig,
}

/// Sampling strategies for data collection
#[derive(Debug, Clone)]
pub enum SamplingStrategy {
    /// Random sampling with uniform distribution
    Random,

    /// Systematic sampling (every Nth record)
    Systematic { interval: u32 },

    /// Stratified sampling within groups
    Stratified { strata: Vec<String> },

    /// Reservoir sampling for streaming data
    Reservoir { reservoir_size: usize },

    /// Time-based sampling
    TimeBased { window: Duration },

    /// Custom sampling strategy
    Custom { strategy_name: String, config: HashMap<String, String> },
}

/// Adaptive sampling configuration
#[derive(Debug, Clone)]
pub struct AdaptiveSamplingConfig {
    /// Enable adaptive sampling
    pub enabled: bool,

    /// Adaptation algorithm
    pub algorithm: AdaptationAlgorithm,

    /// Target sampling rate
    pub target_rate: f64,

    /// Adaptation frequency
    pub adaptation_frequency: Duration,

    /// Load thresholds for adaptation
    pub load_thresholds: LoadThresholds,
}

/// Algorithms for adaptive sampling
#[derive(Debug, Clone)]
pub enum AdaptationAlgorithm {
    /// Load-based adaptation
    LoadBased,
    /// Error-based adaptation
    ErrorBased,
    /// Machine learning-based adaptation
    MachineLearning { model: String },
    /// Threshold-based adaptation
    ThresholdBased { thresholds: HashMap<String, f64> },
    /// Custom adaptation algorithm
    Custom { algorithm_name: String },
}

/// Load thresholds for adaptive sampling
#[derive(Debug, Clone)]
pub struct LoadThresholds {
    /// CPU utilization threshold
    pub cpu_threshold: f64,
    /// Memory utilization threshold
    pub memory_threshold: f64,
    /// I/O utilization threshold
    pub io_threshold: f64,
    /// Network utilization threshold
    pub network_threshold: f64,
}

/// Stratified sampling configuration
#[derive(Debug, Clone)]
pub struct StratifiedSamplingConfig {
    /// Stratification fields
    pub strata_fields: Vec<String>,
    /// Sampling rates per stratum
    pub strata_rates: HashMap<String, f64>,
    /// Minimum samples per stratum
    pub min_samples_per_stratum: u32,
}

/// Sampling quality configuration
#[derive(Debug, Clone)]
pub struct SamplingQualityConfig {
    /// Enable quality monitoring
    pub enabled: bool,
    /// Quality metrics to track
    pub metrics: Vec<QualityMetric>,
    /// Quality thresholds
    pub thresholds: QualityThresholds,
    /// Quality reporting frequency
    pub reporting_frequency: Duration,
}

/// Quality metrics for sampling
#[derive(Debug, Clone)]
pub enum QualityMetric {
    /// Sampling bias measurement
    Bias,
    /// Sampling variance
    Variance,
    /// Coverage (percentage of data sampled)
    Coverage,
    /// Representativeness score
    Representativeness,
    /// Custom quality metric
    Custom { metric_name: String },
}

/// Quality thresholds for sampling
#[derive(Debug, Clone)]
pub struct QualityThresholds {
    /// Maximum acceptable bias
    pub max_bias: f64,
    /// Maximum acceptable variance
    pub max_variance: f64,
    /// Minimum required coverage
    pub min_coverage: f64,
    /// Minimum representativeness score
    pub min_representativeness: f64,
}

impl DataRetentionConfig {
    /// Create configuration optimized for production environments
    pub fn production() -> Self {
        Self {
            metrics_retention: Duration::from_secs(86400 * 90), // 90 days
            events_retention: Duration::from_secs(86400 * 365), // 1 year
            logs_retention: Duration::from_secs(86400 * 30),    // 30 days
            traces_retention: Duration::from_secs(86400 * 7),   // 7 days
            policies: vec![
                RetentionPolicy {
                    name: "critical_events".to_string(),
                    data_type: "events".to_string(),
                    retention_duration: Duration::from_secs(86400 * 365 * 7), // 7 years
                    storage_tier: StorageTier::Cold,
                    conditions: vec![RetentionCondition {
                        field: "severity".to_string(),
                        operator: ConditionOperator::Equals,
                        value: "critical".to_string(),
                    }],
                    priority: 100,
                    metadata: HashMap::new(),
                },
            ],
            cleanup: CleanupConfig {
                enabled: true,
                frequency: Duration::from_secs(86400), // Daily
                strategy: CleanupStrategy::TimeBased,
                batch_size: 1000,
                timeout: Duration::from_secs(3600),
                thresholds: CleanupThresholds {
                    max_age: Duration::from_secs(86400 * 90),
                    max_size: 100_000_000_000, // 100GB
                    max_count: 10_000_000,
                    disk_usage_threshold: 0.85,
                },
                verification: CleanupVerification {
                    enabled: true,
                    sample_rate: 0.01,
                    max_failures: 10,
                    timeout: Duration::from_secs(300),
                },
            },
            archive: ArchiveConfig {
                enabled: true,
                storage: ArchiveStorage::CloudStorage {
                    provider: "s3".to_string(),
                    bucket: "monitoring-archives".to_string(),
                    credentials: CloudCredentials {
                        access_key: "ACCESS_KEY".to_string(),
                        secret_key: "SECRET_KEY".to_string(),
                        config: HashMap::new(),
                    },
                },
                frequency: Duration::from_secs(86400 * 7), // Weekly
                compression: CompressionConfig {
                    enabled: true,
                    algorithm: CompressionAlgorithm::Zstd,
                    level: 6,
                    block_size: 1024 * 1024, // 1MB
                },
                encryption: EncryptionConfig {
                    enabled: true,
                    algorithm: EncryptionAlgorithm::Aes256Gcm,
                    key_management: KeyManagement::ExternalKms {
                        service: "aws-kms".to_string(),
                        key_id: "archive-key".to_string(),
                    },
                    metadata: EncryptionMetadata {
                        include_metadata: true,
                        metadata_fields: vec!["algorithm".to_string(), "key_id".to_string()],
                    },
                },
                verification: ArchiveVerification {
                    enabled: true,
                    methods: vec![
                        VerificationMethod::Checksum { algorithm: "sha256".to_string() },
                        VerificationMethod::SizeVerification,
                    ],
                    frequency: Duration::from_secs(86400 * 30), // Monthly
                    sample_rate: 0.1,
                },
                metadata: ArchiveMetadata {
                    enabled: true,
                    format: MetadataFormat::Json,
                    fields: vec![
                        "timestamp".to_string(),
                        "data_type".to_string(),
                        "size".to_string(),
                        "checksum".to_string(),
                    ],
                    generators: Vec::new(),
                },
            },
        }
    }

    /// Create configuration optimized for development environments
    pub fn development() -> Self {
        Self {
            metrics_retention: Duration::from_secs(86400 * 7), // 7 days
            events_retention: Duration::from_secs(86400 * 7),  // 7 days
            logs_retention: Duration::from_secs(86400 * 3),    // 3 days
            traces_retention: Duration::from_secs(86400),      // 1 day
            policies: Vec::new(),
            cleanup: CleanupConfig {
                enabled: true,
                frequency: Duration::from_secs(86400), // Daily
                strategy: CleanupStrategy::TimeBased,
                batch_size: 100,
                timeout: Duration::from_secs(300),
                thresholds: CleanupThresholds {
                    max_age: Duration::from_secs(86400 * 7),
                    max_size: 1_000_000_000, // 1GB
                    max_count: 100_000,
                    disk_usage_threshold: 0.90,
                },
                verification: CleanupVerification {
                    enabled: false,
                    sample_rate: 0.0,
                    max_failures: 5,
                    timeout: Duration::from_secs(60),
                },
            },
            archive: ArchiveConfig {
                enabled: false,
                storage: ArchiveStorage::LocalFilesystem {
                    base_path: "./dev/archive".to_string(),
                    path_pattern: "%Y/%m/%d".to_string(),
                },
                frequency: Duration::from_secs(86400 * 7), // Weekly
                compression: CompressionConfig {
                    enabled: false,
                    algorithm: CompressionAlgorithm::Gzip,
                    level: 1,
                    block_size: 64 * 1024, // 64KB
                },
                encryption: EncryptionConfig {
                    enabled: false,
                    algorithm: EncryptionAlgorithm::Aes256Gcm,
                    key_management: KeyManagement::StaticKey {
                        key: "dev_key".to_string(),
                    },
                    metadata: EncryptionMetadata {
                        include_metadata: false,
                        metadata_fields: Vec::new(),
                    },
                },
                verification: ArchiveVerification {
                    enabled: false,
                    methods: Vec::new(),
                    frequency: Duration::from_secs(86400 * 7),
                    sample_rate: 0.0,
                },
                metadata: ArchiveMetadata {
                    enabled: false,
                    format: MetadataFormat::Json,
                    fields: Vec::new(),
                    generators: Vec::new(),
                },
            },
        }
    }
}

impl ExportConfig {
    /// Create configuration optimized for analytics
    pub fn analytics_focused() -> Self {
        Self {
            enabled: true,
            formats: vec![
                ExportFormat::Parquet {
                    compression: "snappy".to_string(),
                    row_group_size: 1_000_000,
                },
                ExportFormat::Json {
                    pretty_print: false,
                    compression: Some(CompressionConfig {
                        enabled: true,
                        algorithm: CompressionAlgorithm::Gzip,
                        level: 6,
                        block_size: 1024 * 1024,
                    }),
                },
            ],
            destinations: vec![ExportDestination::CloudStorage {
                provider: CloudCredentials {
                    access_key: "ANALYTICS_ACCESS_KEY".to_string(),
                    secret_key: "ANALYTICS_SECRET_KEY".to_string(),
                    config: HashMap::new(),
                },
                bucket: "analytics-data".to_string(),
                key_pattern: "year=%Y/month=%m/day=%d/hour=%H/data.parquet".to_string(),
            }],
            scheduling: ExportSchedulingConfig {
                enabled: true,
                frequency: ExportFrequency::Interval(Duration::from_secs(3600)), // Hourly
                time_window: None,
                max_size: Some(100_000_000), // 100MB
                timeout: Duration::from_secs(600),
                retry: RetryConfig {
                    enabled: true,
                    max_attempts: 3,
                    delay_strategy: RetryDelayStrategy::ExponentialBackoff {
                        base_delay: Duration::from_secs(10),
                        max_delay: Duration::from_secs(300),
                    },
                    max_total_time: Duration::from_secs(1800),
                },
            },
            filtering: ExportFilteringConfig {
                enabled: true,
                filters: Vec::new(),
                include_patterns: vec!["metrics.*".to_string(), "events.*".to_string()],
                exclude_patterns: vec!["debug.*".to_string()],
                sampling: Some(ExportSamplingConfig {
                    rate: 1.0, // No sampling for analytics
                    strategy: SamplingStrategy::Random,
                    seed: Some(12345),
                }),
            },
            transformation: ExportTransformationConfig {
                enabled: true,
                rules: vec![TransformationRule {
                    name: "timestamp_iso".to_string(),
                    source_field: "timestamp".to_string(),
                    target_field: "timestamp_iso".to_string(),
                    transformation: TransformationType::TypeConversion {
                        target_type: "iso8601".to_string(),
                    },
                    conditions: Vec::new(),
                }],
                defaults: Vec::new(),
            },
            monitoring: ExportMonitoringConfig {
                enabled: true,
                track_results: true,
                track_performance: true,
                error_notifications: vec!["analytics-team@company.com".to_string()],
                monitoring_retention: Duration::from_secs(86400 * 30),
            },
        }
    }

    /// Create configuration optimized for real-time exports
    pub fn real_time() -> Self {
        Self {
            enabled: true,
            formats: vec![ExportFormat::Json {
                pretty_print: false,
                compression: None, // No compression for speed
            }],
            destinations: vec![ExportDestination::MessageQueue {
                connection: "kafka://localhost:9092".to_string(),
                topic: "real-time-monitoring".to_string(),
                partitioning: Some(PartitioningConfig {
                    strategy: PartitioningStrategy::Hash,
                    partition_count: 10,
                    key_field: "source".to_string(),
                }),
            }],
            scheduling: ExportSchedulingConfig {
                enabled: true,
                frequency: ExportFrequency::Interval(Duration::from_secs(10)), // Every 10 seconds
                time_window: None,
                max_size: Some(1_000_000), // 1MB
                timeout: Duration::from_secs(30),
                retry: RetryConfig {
                    enabled: true,
                    max_attempts: 2,
                    delay_strategy: RetryDelayStrategy::Fixed(Duration::from_secs(1)),
                    max_total_time: Duration::from_secs(60),
                },
            },
            filtering: ExportFilteringConfig {
                enabled: true,
                filters: vec![ExportFilter {
                    name: "critical_only".to_string(),
                    field: "severity".to_string(),
                    operator: FilterOperator::Equals,
                    value: "critical".to_string(),
                    action: FilterAction::Include,
                }],
                include_patterns: vec!["alerts.*".to_string(), "incidents.*".to_string()],
                exclude_patterns: vec!["debug.*".to_string(), "trace.*".to_string()],
                sampling: Some(ExportSamplingConfig {
                    rate: 1.0, // No sampling for real-time
                    strategy: SamplingStrategy::Random,
                    seed: None,
                }),
            },
            transformation: ExportTransformationConfig {
                enabled: false, // Minimal transformation for speed
                rules: Vec::new(),
                defaults: Vec::new(),
            },
            monitoring: ExportMonitoringConfig {
                enabled: true,
                track_results: true,
                track_performance: true,
                error_notifications: vec!["ops-team@company.com".to_string()],
                monitoring_retention: Duration::from_secs(86400 * 7),
            },
        }
    }
}

impl SamplingConfig {
    /// Create configuration optimized for high-volume scenarios
    pub fn high_volume() -> Self {
        Self {
            enabled: true,
            strategy: SamplingStrategy::Random,
            rate: 0.01, // 1% sampling
            adaptive: AdaptiveSamplingConfig {
                enabled: true,
                algorithm: AdaptationAlgorithm::LoadBased,
                target_rate: 0.01,
                adaptation_frequency: Duration::from_secs(60),
                load_thresholds: LoadThresholds {
                    cpu_threshold: 0.8,
                    memory_threshold: 0.8,
                    io_threshold: 0.8,
                    network_threshold: 0.8,
                },
            },
            stratified: None,
            quality: SamplingQualityConfig {
                enabled: true,
                metrics: vec![QualityMetric::Coverage, QualityMetric::Bias],
                thresholds: QualityThresholds {
                    max_bias: 0.05,
                    max_variance: 0.1,
                    min_coverage: 0.95,
                    min_representativeness: 0.9,
                },
                reporting_frequency: Duration::from_secs(3600),
            },
        }
    }

    /// Create configuration focused on sampling quality
    pub fn quality_focused() -> Self {
        Self {
            enabled: true,
            strategy: SamplingStrategy::Stratified {
                strata: vec!["service".to_string(), "severity".to_string()],
            },
            rate: 0.1, // 10% sampling
            adaptive: AdaptiveSamplingConfig {
                enabled: false,
                algorithm: AdaptationAlgorithm::LoadBased,
                target_rate: 0.1,
                adaptation_frequency: Duration::from_secs(300),
                load_thresholds: LoadThresholds {
                    cpu_threshold: 0.7,
                    memory_threshold: 0.7,
                    io_threshold: 0.7,
                    network_threshold: 0.7,
                },
            },
            stratified: Some(StratifiedSamplingConfig {
                strata_fields: vec!["service".to_string(), "severity".to_string()],
                strata_rates: {
                    let mut rates = HashMap::new();
                    rates.insert("critical".to_string(), 1.0); // 100% for critical
                    rates.insert("warning".to_string(), 0.5);  // 50% for warnings
                    rates.insert("info".to_string(), 0.1);     // 10% for info
                    rates
                },
                min_samples_per_stratum: 100,
            }),
            quality: SamplingQualityConfig {
                enabled: true,
                metrics: vec![
                    QualityMetric::Coverage,
                    QualityMetric::Bias,
                    QualityMetric::Variance,
                    QualityMetric::Representativeness,
                ],
                thresholds: QualityThresholds {
                    max_bias: 0.02,
                    max_variance: 0.05,
                    min_coverage: 0.98,
                    min_representativeness: 0.95,
                },
                reporting_frequency: Duration::from_secs(1800),
            },
        }
    }
}

impl Default for DataRetentionConfig {
    fn default() -> Self {
        Self {
            metrics_retention: Duration::from_secs(86400 * 30), // 30 days
            events_retention: Duration::from_secs(86400 * 30),  // 30 days
            logs_retention: Duration::from_secs(86400 * 7),     // 7 days
            traces_retention: Duration::from_secs(86400 * 3),   // 3 days
            policies: Vec::new(),
            cleanup: CleanupConfig {
                enabled: true,
                frequency: Duration::from_secs(86400), // Daily
                strategy: CleanupStrategy::TimeBased,
                batch_size: 500,
                timeout: Duration::from_secs(1800),
                thresholds: CleanupThresholds {
                    max_age: Duration::from_secs(86400 * 30),
                    max_size: 10_000_000_000, // 10GB
                    max_count: 1_000_000,
                    disk_usage_threshold: 0.80,
                },
                verification: CleanupVerification {
                    enabled: false,
                    sample_rate: 0.01,
                    max_failures: 5,
                    timeout: Duration::from_secs(300),
                },
            },
            archive: ArchiveConfig {
                enabled: false,
                storage: ArchiveStorage::LocalFilesystem {
                    base_path: "./data/archive".to_string(),
                    path_pattern: "%Y/%m/%d".to_string(),
                },
                frequency: Duration::from_secs(86400 * 7),
                compression: CompressionConfig {
                    enabled: false,
                    algorithm: CompressionAlgorithm::Gzip,
                    level: 6,
                    block_size: 1024 * 1024,
                },
                encryption: EncryptionConfig {
                    enabled: false,
                    algorithm: EncryptionAlgorithm::Aes256Gcm,
                    key_management: KeyManagement::StaticKey {
                        key: "default_key".to_string(),
                    },
                    metadata: EncryptionMetadata {
                        include_metadata: false,
                        metadata_fields: Vec::new(),
                    },
                },
                verification: ArchiveVerification {
                    enabled: false,
                    methods: Vec::new(),
                    frequency: Duration::from_secs(86400 * 30),
                    sample_rate: 0.0,
                },
                metadata: ArchiveMetadata {
                    enabled: false,
                    format: MetadataFormat::Json,
                    fields: Vec::new(),
                    generators: Vec::new(),
                },
            },
        }
    }
}

impl Default for ExportConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            formats: vec![ExportFormat::Json {
                pretty_print: true,
                compression: None,
            }],
            destinations: Vec::new(),
            scheduling: ExportSchedulingConfig {
                enabled: false,
                frequency: ExportFrequency::Interval(Duration::from_secs(3600)),
                time_window: None,
                max_size: Some(100_000_000),
                timeout: Duration::from_secs(300),
                retry: RetryConfig {
                    enabled: true,
                    max_attempts: 3,
                    delay_strategy: RetryDelayStrategy::ExponentialBackoff {
                        base_delay: Duration::from_secs(5),
                        max_delay: Duration::from_secs(300),
                    },
                    max_total_time: Duration::from_secs(1800),
                },
            },
            filtering: ExportFilteringConfig {
                enabled: false,
                filters: Vec::new(),
                include_patterns: Vec::new(),
                exclude_patterns: Vec::new(),
                sampling: None,
            },
            transformation: ExportTransformationConfig {
                enabled: false,
                rules: Vec::new(),
                defaults: Vec::new(),
            },
            monitoring: ExportMonitoringConfig {
                enabled: false,
                track_results: false,
                track_performance: false,
                error_notifications: Vec::new(),
                monitoring_retention: Duration::from_secs(86400 * 7),
            },
        }
    }
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            strategy: SamplingStrategy::Random,
            rate: 1.0, // No sampling by default
            adaptive: AdaptiveSamplingConfig {
                enabled: false,
                algorithm: AdaptationAlgorithm::LoadBased,
                target_rate: 0.1,
                adaptation_frequency: Duration::from_secs(60),
                load_thresholds: LoadThresholds {
                    cpu_threshold: 0.8,
                    memory_threshold: 0.8,
                    io_threshold: 0.8,
                    network_threshold: 0.8,
                },
            },
            stratified: None,
            quality: SamplingQualityConfig {
                enabled: false,
                metrics: Vec::new(),
                thresholds: QualityThresholds {
                    max_bias: 0.05,
                    max_variance: 0.1,
                    min_coverage: 0.95,
                    min_representativeness: 0.9,
                },
                reporting_frequency: Duration::from_secs(3600),
            },
        }
    }
}