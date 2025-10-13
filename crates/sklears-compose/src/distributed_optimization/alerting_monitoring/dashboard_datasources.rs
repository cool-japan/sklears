use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

use super::dashboard_visualization::{FilterOperator, CompressionAlgorithm};

/// Data source configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataSourceConfiguration {
    pub source_id: String,
    pub source_name: String,
    pub source_type: DataSourceType,
    pub connection_config: ConnectionConfiguration,
    pub query_config: QueryConfiguration,
    pub caching_config: CachingConfiguration,
    pub refresh_config: RefreshConfiguration,
    pub security_config: SecurityConfiguration,
}

/// Data source types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DataSourceType {
    Database(DatabaseSourceConfig),
    API(ApiSourceConfig),
    File(FileSourceConfig),
    Stream(StreamSourceConfig),
    Custom(String),
}

/// Database types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DatabaseType {
    PostgreSQL,
    MySQL,
    SQLite,
    Oracle,
    SQLServer,
    MongoDB,
    Cassandra,
    Redis,
    InfluxDB,
    TimescaleDB,
    ClickHouse,
    BigQuery,
    Snowflake,
    Redshift,
    Custom(String),
}

/// Database source configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseSourceConfig {
    pub database_type: DatabaseType,
    pub host: String,
    pub port: u16,
    pub database_name: String,
    pub schema: Option<String>,
    pub connection_pool: ConnectionPoolConfig,
    pub ssl_config: Option<SslConfiguration>,
}

/// Connection pool configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectionPoolConfig {
    pub min_connections: u32,
    pub max_connections: u32,
    pub connection_timeout: Duration,
    pub idle_timeout: Duration,
    pub max_lifetime: Duration,
}

/// SSL configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SslConfiguration {
    pub enabled: bool,
    pub certificate_path: Option<String>,
    pub key_path: Option<String>,
    pub ca_path: Option<String>,
    pub verify_certificate: bool,
    pub verify_hostname: bool,
}

/// API source configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiSourceConfig {
    pub base_url: String,
    pub api_version: Option<String>,
    pub authentication: ApiAuthentication,
    pub rate_limiting: RateLimitingConfig,
    pub retry_config: RetryConfiguration,
    pub timeout_config: TimeoutConfiguration,
}

/// API authentication methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ApiAuthentication {
    None,
    ApiKey { key: String, header: String },
    Bearer { token: String },
    Basic { username: String, password: String },
    OAuth2(OAuth2Config),
    Custom(HashMap<String, String>),
}

/// OAuth2 configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OAuth2Config {
    pub client_id: String,
    pub client_secret: String,
    pub auth_url: String,
    pub token_url: String,
    pub scope: Vec<String>,
    pub redirect_uri: String,
}

/// Rate limiting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitingConfig {
    pub enabled: bool,
    pub requests_per_second: f64,
    pub burst_capacity: u32,
    pub retry_after_rate_limit: bool,
}

/// Retry configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryConfiguration {
    pub max_retries: u32,
    pub base_delay: Duration,
    pub max_delay: Duration,
    pub exponential_backoff: bool,
    pub jitter: bool,
    pub retry_conditions: Vec<RetryCondition>,
}

/// Retry conditions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RetryCondition {
    HttpStatus(u16),
    Timeout,
    NetworkError,
    ServerError,
    Custom(String),
}

/// Timeout configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeoutConfiguration {
    pub connection_timeout: Duration,
    pub request_timeout: Duration,
    pub read_timeout: Duration,
    pub write_timeout: Duration,
}

/// File source configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileSourceConfig {
    pub file_path: String,
    pub file_format: FileFormat,
    pub encoding: String,
    pub compression: Option<CompressionAlgorithm>,
    pub watch_for_changes: bool,
    pub backup_config: Option<BackupConfiguration>,
}

/// File formats
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FileFormat {
    CSV { delimiter: String, has_header: bool },
    JSON { array_path: Option<String> },
    XML { root_element: String },
    Parquet,
    Excel { sheet_name: Option<String> },
    Log { pattern: String },
    Custom(String),
}

/// Backup configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackupConfiguration {
    pub enabled: bool,
    pub backup_interval: Duration,
    pub backup_location: String,
    pub max_backups: u32,
    pub compression_enabled: bool,
}

/// Stream source configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamSourceConfig {
    pub stream_type: StreamType,
    pub topic: String,
    pub consumer_group: Option<String>,
    pub partition_config: PartitionConfiguration,
    pub offset_config: OffsetConfiguration,
}

/// Stream types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StreamType {
    Kafka(KafkaConfig),
    Kinesis(KinesisConfig),
    PubSub(PubSubConfig),
    Custom(String),
}

/// Kafka configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KafkaConfig {
    pub brokers: Vec<String>,
    pub security_protocol: SecurityProtocol,
    pub sasl_config: Option<SaslConfiguration>,
    pub ssl_config: Option<SslConfiguration>,
}

/// Security protocols
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SecurityProtocol {
    PLAINTEXT,
    SSL,
    SASL_PLAINTEXT,
    SASL_SSL,
}

/// SASL configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SaslConfiguration {
    pub mechanism: SaslMechanism,
    pub username: String,
    pub password: String,
}

/// SASL mechanisms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SaslMechanism {
    PLAIN,
    SCRAM_SHA_256,
    SCRAM_SHA_512,
    GSSAPI,
}

/// Kinesis configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KinesisConfig {
    pub region: String,
    pub access_key: String,
    pub secret_key: String,
    pub session_token: Option<String>,
}

/// Pub/Sub configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PubSubConfig {
    pub project_id: String,
    pub credentials_path: String,
    pub subscription_name: String,
}

/// Partition configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartitionConfiguration {
    pub partitions: Option<Vec<u32>>,
    pub assignment_strategy: PartitionAssignmentStrategy,
    pub rebalance_strategy: RebalanceStrategy,
}

/// Partition assignment strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PartitionAssignmentStrategy {
    RoundRobin,
    Range,
    Sticky,
    CooperativeSticky,
    Custom(String),
}

/// Rebalance strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RebalanceStrategy {
    Eager,
    Cooperative,
    Custom(String),
}

/// Offset configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OffsetConfiguration {
    pub auto_offset_reset: AutoOffsetReset,
    pub commit_strategy: CommitStrategy,
    pub commit_interval: Duration,
}

/// Auto offset reset options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AutoOffsetReset {
    Earliest,
    Latest,
    None,
    Custom(String),
}

/// Commit strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CommitStrategy {
    Auto,
    Manual,
    Batch,
    Sync,
    Async,
}

/// Connection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectionConfiguration {
    pub max_connections: u32,
    pub connection_timeout: Duration,
    pub read_timeout: Duration,
    pub write_timeout: Duration,
    pub keep_alive: bool,
    pub connection_pooling: bool,
    pub ssl_verification: bool,
}

/// Query configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryConfiguration {
    pub query: String,
    pub parameters: HashMap<String, String>,
    pub timeout: Duration,
    pub result_limit: Option<u32>,
    pub pagination: Option<PaginationConfig>,
    pub aggregation: Option<AggregationConfig>,
}

/// Pagination configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaginationConfig {
    pub page_size: u32,
    pub max_pages: Option<u32>,
    pub offset_field: String,
    pub limit_field: String,
}

/// Aggregation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregationConfig {
    pub group_by_fields: Vec<String>,
    pub aggregate_functions: Vec<AggregateFunction>,
    pub having_conditions: Vec<HavingCondition>,
    pub order_by: Vec<OrderByClause>,
}

/// Aggregate functions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregateFunction {
    pub function_type: AggregateFunctionType,
    pub field: String,
    pub alias: Option<String>,
}

/// Aggregate function types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AggregateFunctionType {
    Count,
    Sum,
    Average,
    Min,
    Max,
    StdDev,
    Variance,
    Median,
    Percentile(f64),
    Custom(String),
}

/// Having conditions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HavingCondition {
    pub field: String,
    pub operator: FilterOperator,
    pub value: String,
}

/// Order by clauses
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderByClause {
    pub field: String,
    pub direction: SortDirection,
    pub nulls_handling: NullsHandling,
}

/// Sort directions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SortDirection {
    Ascending,
    Descending,
}

/// Nulls handling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NullsHandling {
    First,
    Last,
    Default,
}

/// Caching configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachingConfiguration {
    pub enabled: bool,
    pub cache_type: CacheType,
    pub ttl: Duration,
    pub max_size: u64,
    pub eviction_policy: EvictionPolicy,
    pub cache_key_strategy: CacheKeyStrategy,
}

/// Cache types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CacheType {
    Memory,
    Redis(RedisConfig),
    Memcached(MemcachedConfig),
    Database,
    File,
    Custom(String),
}

/// Redis configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RedisConfig {
    pub host: String,
    pub port: u16,
    pub password: Option<String>,
    pub database: u32,
    pub cluster_mode: bool,
    pub ssl_enabled: bool,
}

/// Memcached configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemcachedConfig {
    pub servers: Vec<String>,
    pub consistent_hashing: bool,
    pub compression: bool,
}

/// Eviction policies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EvictionPolicy {
    LRU,
    LFU,
    FIFO,
    Random,
    TTL,
    Custom(String),
}

/// Cache key strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CacheKeyStrategy {
    QueryHash,
    ParameterHash,
    Custom(String),
    Composite(Vec<CacheKeyStrategy>),
}

/// Refresh configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RefreshConfiguration {
    pub auto_refresh: bool,
    pub refresh_interval: Duration,
    pub refresh_strategy: RefreshStrategy,
    pub retry_on_failure: bool,
    pub max_retries: u32,
    pub backoff_strategy: BackoffStrategy,
}

/// Refresh strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RefreshStrategy {
    Full,
    Incremental,
    Differential,
    Adaptive,
    Custom(String),
}

/// Backoff strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BackoffStrategy {
    Fixed(Duration),
    Linear(Duration),
    Exponential { base: Duration, multiplier: f64 },
    Custom(String),
}

/// Security configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConfiguration {
    pub authentication_required: bool,
    pub authorization_rules: Vec<AuthorizationRule>,
    pub encryption_config: EncryptionConfiguration,
    pub audit_config: AuditConfiguration,
    pub access_control: AccessControlConfiguration,
}

/// Authorization rules
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthorizationRule {
    pub rule_id: String,
    pub resource: String,
    pub action: String,
    pub subjects: Vec<String>,
    pub conditions: Vec<AuthorizationCondition>,
    pub effect: AuthorizationEffect,
}

/// Authorization conditions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthorizationCondition {
    pub attribute: String,
    pub operator: String,
    pub value: String,
}

/// Authorization effects
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuthorizationEffect {
    Allow,
    Deny,
}

/// Encryption configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptionConfiguration {
    pub data_at_rest: EncryptionSettings,
    pub data_in_transit: EncryptionSettings,
    pub key_management: KeyManagementConfig,
}

/// Encryption settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptionSettings {
    pub enabled: bool,
    pub algorithm: EncryptionAlgorithm,
    pub key_size: u32,
    pub mode: EncryptionMode,
}

/// Encryption algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EncryptionAlgorithm {
    AES,
    DES,
    RSA,
    ECC,
    ChaCha20,
    Custom(String),
}

/// Encryption modes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EncryptionMode {
    ECB,
    CBC,
    CFB,
    OFB,
    CTR,
    GCM,
    Custom(String),
}

/// Key management configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyManagementConfig {
    pub key_provider: KeyProvider,
    pub key_rotation: KeyRotationConfig,
    pub key_derivation: KeyDerivationConfig,
}

/// Key providers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum KeyProvider {
    Local,
    HSM(HsmConfig),
    Cloud(CloudKeyConfig),
    External(ExternalKeyConfig),
    Custom(String),
}

/// HSM configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HsmConfig {
    pub provider: String,
    pub slot_id: u32,
    pub pin: String,
    pub key_label: String,
}

/// Cloud key configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CloudKeyConfig {
    pub provider: CloudProvider,
    pub key_id: String,
    pub region: String,
    pub credentials: CloudCredentials,
}

/// Cloud providers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CloudProvider {
    AWS,
    Azure,
    GCP,
    Oracle,
    IBM,
    Custom(String),
}

/// Cloud credentials
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CloudCredentials {
    AccessKey { access_key: String, secret_key: String },
    ServiceAccount { credentials_path: String },
    AssumeRole { role_arn: String },
    Default,
    Custom(HashMap<String, String>),
}

/// External key configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExternalKeyConfig {
    pub endpoint: String,
    pub authentication: ApiAuthentication,
    pub key_identifier: String,
}

/// Key rotation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyRotationConfig {
    pub enabled: bool,
    pub rotation_interval: Duration,
    pub auto_rotation: bool,
    pub notification_enabled: bool,
}

/// Key derivation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyDerivationConfig {
    pub algorithm: KeyDerivationAlgorithm,
    pub salt_length: u32,
    pub iterations: u32,
}

/// Key derivation algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum KeyDerivationAlgorithm {
    PBKDF2,
    Scrypt,
    Argon2,
    Custom(String),
}

/// Audit configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditConfiguration {
    pub enabled: bool,
    pub audit_level: AuditLevel,
    pub log_destination: LogDestination,
    pub retention_period: Duration,
    pub sensitive_data_masking: bool,
}

/// Audit levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuditLevel {
    None,
    Basic,
    Detailed,
    Comprehensive,
}

/// Log destinations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LogDestination {
    File(String),
    Database(DatabaseConfig),
    Syslog(SyslogConfig),
    CloudLog(CloudLogConfig),
    Custom(String),
}

/// Database configuration for logging
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseConfig {
    pub database_type: DatabaseType,
    pub connection_string: String,
    pub table_name: String,
}

/// Syslog configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyslogConfig {
    pub host: String,
    pub port: u16,
    pub facility: SyslogFacility,
    pub severity: SyslogSeverity,
}

/// Syslog facilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SyslogFacility {
    Kernel,
    User,
    Mail,
    Daemon,
    Auth,
    Syslog,
    News,
    UUCP,
    Cron,
    AuthPriv,
    FTP,
    Local0,
    Local1,
    Local2,
    Local3,
    Local4,
    Local5,
    Local6,
    Local7,
}

/// Syslog severities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SyslogSeverity {
    Emergency,
    Alert,
    Critical,
    Error,
    Warning,
    Notice,
    Info,
    Debug,
}

/// Cloud log configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CloudLogConfig {
    pub provider: CloudProvider,
    pub log_group: String,
    pub log_stream: String,
    pub region: String,
    pub credentials: CloudCredentials,
}

/// Access control configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessControlConfiguration {
    pub rbac_enabled: bool,
    pub abac_enabled: bool,
    pub session_management: SessionManagementConfig,
    pub ip_filtering: IpFilteringConfig,
    pub rate_limiting: RateLimitingConfig,
}

/// Session management configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionManagementConfig {
    pub session_timeout: Duration,
    pub concurrent_sessions: u32,
    pub session_storage: SessionStorage,
    pub session_encryption: bool,
}

/// Session storage types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SessionStorage {
    Memory,
    Database,
    Redis,
    File,
    Custom(String),
}

/// IP filtering configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IpFilteringConfig {
    pub enabled: bool,
    pub whitelist: Vec<String>,
    pub blacklist: Vec<String>,
    pub geo_filtering: GeoFilteringConfig,
}

/// Geographic filtering configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeoFilteringConfig {
    pub enabled: bool,
    pub allowed_countries: Vec<String>,
    pub blocked_countries: Vec<String>,
    pub allowed_regions: Vec<String>,
    pub blocked_regions: Vec<String>,
}