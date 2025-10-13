//! Environment Context Module
//!
//! Provides comprehensive environment variable and system settings management
//! for execution contexts, including variable isolation, inheritance, and validation.

use std::{
    collections::{HashMap, HashSet},
    sync::{Arc, RwLock, Mutex},
    time::{Duration, SystemTime},
    env::{self, VarError},
    path::PathBuf,
    fmt::{Debug, Display},
};
use serde::{Serialize, Deserialize};
use uuid::Uuid;

use crate::context_core::{
    ExecutionContextTrait, ContextType, ContextState, ContextMetadata, ContextError, ContextResult,
    ContextEvent, IsolationLevel, ContextPriority
};

/// Environment context for environment variables and system settings
#[derive(Debug)]
pub struct EnvironmentContext {
    /// Context identifier
    context_id: String,
    /// Environment manager
    env_manager: Arc<EnvironmentManager>,
    /// Variable store
    var_store: Arc<RwLock<VariableStore>>,
    /// Environment policies
    policies: Arc<RwLock<EnvironmentPolicies>>,
    /// Change tracker
    change_tracker: Arc<Mutex<ChangeTracker>>,
    /// Context state
    state: Arc<RwLock<ContextState>>,
    /// Metadata
    metadata: Arc<RwLock<ContextMetadata>>,
    /// Environment metrics
    metrics: Arc<Mutex<EnvironmentMetrics>>,
}

/// Environment manager for comprehensive environment management
#[derive(Debug)]
pub struct EnvironmentManager {
    /// Environment configuration
    config: Arc<RwLock<EnvironmentConfig>>,
    /// Variable validators
    validators: Arc<RwLock<Vec<Box<dyn VariableValidator>>>>,
    /// Environment providers
    providers: Arc<RwLock<Vec<Box<dyn EnvironmentProvider>>>>,
    /// Isolation manager
    isolation: Arc<IsolationManager>,
}

/// Environment configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentConfig {
    /// Enable environment isolation
    pub isolation_enabled: bool,
    /// Isolation level
    pub isolation_level: EnvironmentIsolationLevel,
    /// Environment inheritance
    pub inheritance: EnvironmentInheritance,
    /// Variable validation settings
    pub validation: ValidationConfig,
    /// Security settings
    pub security: SecurityConfig,
    /// Synchronization settings
    pub synchronization: SynchronizationConfig,
}

impl Default for EnvironmentConfig {
    fn default() -> Self {
        Self {
            isolation_enabled: true,
            isolation_level: EnvironmentIsolationLevel::Process,
            inheritance: EnvironmentInheritance::default(),
            validation: ValidationConfig::default(),
            security: SecurityConfig::default(),
            synchronization: SynchronizationConfig::default(),
        }
    }
}

/// Environment isolation levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EnvironmentIsolationLevel {
    /// No isolation (shared environment)
    None,
    /// Thread-level isolation
    Thread,
    /// Process-level isolation
    Process,
    /// Container-level isolation
    Container,
    /// Namespace-level isolation
    Namespace,
    /// Virtual environment isolation
    Virtual,
}

/// Environment inheritance settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentInheritance {
    /// Inherit from parent environment
    pub inherit_parent: bool,
    /// Inherit from system environment
    pub inherit_system: bool,
    /// Inheritance strategy
    pub strategy: InheritanceStrategy,
    /// Variable override rules
    pub override_rules: Vec<OverrideRule>,
    /// Inheritance filters
    pub filters: InheritanceFilters,
}

impl Default for EnvironmentInheritance {
    fn default() -> Self {
        Self {
            inherit_parent: true,
            inherit_system: true,
            strategy: InheritanceStrategy::Merge,
            override_rules: Vec::new(),
            filters: InheritanceFilters::default(),
        }
    }
}

/// Inheritance strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InheritanceStrategy {
    /// Replace parent variables
    Replace,
    /// Merge with parent variables
    Merge,
    /// Append to parent variables
    Append,
    /// Custom inheritance logic
    Custom,
}

/// Variable override rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OverrideRule {
    /// Rule name
    pub name: String,
    /// Variable pattern
    pub pattern: String,
    /// Override action
    pub action: OverrideAction,
    /// Rule priority
    pub priority: i32,
    /// Rule conditions
    pub conditions: Vec<RuleCondition>,
}

/// Override actions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OverrideAction {
    /// Allow override
    Allow,
    /// Deny override
    Deny,
    /// Warn on override
    Warn,
    /// Transform value
    Transform,
}

/// Rule condition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuleCondition {
    /// Condition type
    pub condition_type: ConditionType,
    /// Condition expression
    pub expression: String,
    /// Expected value
    pub expected: Option<String>,
}

/// Condition types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConditionType {
    /// Variable exists
    Exists,
    /// Variable equals value
    Equals,
    /// Variable matches pattern
    Matches,
    /// Variable contains substring
    Contains,
    /// Custom condition
    Custom,
}

/// Inheritance filters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InheritanceFilters {
    /// Include patterns
    pub include_patterns: Vec<String>,
    /// Exclude patterns
    pub exclude_patterns: Vec<String>,
    /// Prefix filters
    pub prefix_filters: Vec<PrefixFilter>,
    /// Tag filters
    pub tag_filters: Vec<String>,
}

impl Default for InheritanceFilters {
    fn default() -> Self {
        Self {
            include_patterns: vec!["*".to_string()],
            exclude_patterns: vec!["SECRET_*".to_string(), "PRIVATE_*".to_string()],
            prefix_filters: Vec::new(),
            tag_filters: Vec::new(),
        }
    }
}

/// Prefix filter for variable inheritance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrefixFilter {
    /// Variable prefix
    pub prefix: String,
    /// Filter action
    pub action: FilterAction,
    /// Transformation rule
    pub transformation: Option<String>,
}

/// Filter actions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FilterAction {
    /// Include variable
    Include,
    /// Exclude variable
    Exclude,
    /// Transform variable
    Transform,
    /// Rename variable
    Rename,
}

/// Validation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationConfig {
    /// Enable variable validation
    pub enabled: bool,
    /// Strict validation mode
    pub strict_mode: bool,
    /// Validation rules
    pub rules: Vec<ValidationRule>,
    /// Custom validators
    pub custom_validators: Vec<String>,
    /// Validation on change
    pub validate_on_change: bool,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            strict_mode: false,
            rules: Vec::new(),
            custom_validators: Vec::new(),
            validate_on_change: true,
        }
    }
}

/// Variable validation rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationRule {
    /// Rule name
    pub name: String,
    /// Variable pattern
    pub variable_pattern: String,
    /// Value validation
    pub value_validation: ValueValidation,
    /// Rule severity
    pub severity: ValidationSeverity,
    /// Error message
    pub error_message: Option<String>,
}

/// Value validation types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValueValidation {
    /// Required (not empty)
    Required,
    /// Matches regex pattern
    Regex(String),
    /// Is numeric
    Numeric,
    /// Is boolean
    Boolean,
    /// Is valid path
    Path,
    /// Is valid URL
    Url,
    /// Is valid email
    Email,
    /// Custom validation function
    Custom(String),
    /// Length constraints
    Length { min: Option<usize>, max: Option<usize> },
    /// Allowed values
    AllowedValues(Vec<String>),
}

/// Validation severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ValidationSeverity {
    /// Info level
    Info = 1,
    /// Warning level
    Warning = 2,
    /// Error level
    Error = 3,
    /// Critical level
    Critical = 4,
}

/// Security configuration for environment variables
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConfig {
    /// Enable variable encryption
    pub encryption_enabled: bool,
    /// Sensitive variable patterns
    pub sensitive_patterns: Vec<String>,
    /// Encryption key management
    pub key_management: KeyManagementConfig,
    /// Access control
    pub access_control: AccessControlConfig,
    /// Audit logging
    pub audit_logging: AuditLoggingConfig,
}

impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            encryption_enabled: false,
            sensitive_patterns: vec![
                "*PASSWORD*".to_string(),
                "*SECRET*".to_string(),
                "*KEY*".to_string(),
                "*TOKEN*".to_string(),
            ],
            key_management: KeyManagementConfig::default(),
            access_control: AccessControlConfig::default(),
            audit_logging: AuditLoggingConfig::default(),
        }
    }
}

/// Key management configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyManagementConfig {
    /// Key derivation method
    pub key_derivation: KeyDerivationMethod,
    /// Key rotation interval
    pub rotation_interval: Duration,
    /// Key storage location
    pub storage_location: KeyStorageLocation,
    /// Key backup enabled
    pub backup_enabled: bool,
}

impl Default for KeyManagementConfig {
    fn default() -> Self {
        Self {
            key_derivation: KeyDerivationMethod::PBKDF2,
            rotation_interval: Duration::from_secs(30 * 24 * 60 * 60), // 30 days
            storage_location: KeyStorageLocation::Memory,
            backup_enabled: false,
        }
    }
}

/// Key derivation methods
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum KeyDerivationMethod {
    /// PBKDF2 key derivation
    PBKDF2,
    /// Argon2 key derivation
    Argon2,
    /// Scrypt key derivation
    Scrypt,
    /// Custom key derivation
    Custom,
}

/// Key storage locations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum KeyStorageLocation {
    /// In-memory storage
    Memory,
    /// File system storage
    FileSystem,
    /// Hardware security module
    HSM,
    /// Key management service
    KMS,
    /// Environment variable
    Environment,
}

/// Access control configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessControlConfig {
    /// Enable access control
    pub enabled: bool,
    /// Default access policy
    pub default_policy: AccessPolicy,
    /// Variable-specific policies
    pub variable_policies: HashMap<String, AccessPolicy>,
    /// Role-based access control
    pub rbac_enabled: bool,
}

impl Default for AccessControlConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            default_policy: AccessPolicy::ReadWrite,
            variable_policies: HashMap::new(),
            rbac_enabled: false,
        }
    }
}

/// Access policies for environment variables
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AccessPolicy {
    /// No access
    None,
    /// Read-only access
    ReadOnly,
    /// Write-only access
    WriteOnly,
    /// Read-write access
    ReadWrite,
    /// Admin access
    Admin,
}

/// Audit logging configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditLoggingConfig {
    /// Enable audit logging
    pub enabled: bool,
    /// Log variable reads
    pub log_reads: bool,
    /// Log variable writes
    pub log_writes: bool,
    /// Log variable deletions
    pub log_deletions: bool,
    /// Include variable values in logs
    pub include_values: bool,
    /// Log retention period
    pub retention_period: Duration,
}

impl Default for AuditLoggingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            log_reads: false,
            log_writes: true,
            log_deletions: true,
            include_values: false, // Don't log sensitive values
            retention_period: Duration::from_secs(365 * 24 * 60 * 60), // 1 year
        }
    }
}

/// Synchronization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynchronizationConfig {
    /// Sync with system environment
    pub sync_system: bool,
    /// Sync interval
    pub sync_interval: Duration,
    /// Sync strategy
    pub sync_strategy: SyncStrategy,
    /// Conflict resolution
    pub conflict_resolution: ConflictResolution,
    /// Auto-sync enabled
    pub auto_sync: bool,
}

impl Default for SynchronizationConfig {
    fn default() -> Self {
        Self {
            sync_system: false,
            sync_interval: Duration::from_secs(60),
            sync_strategy: SyncStrategy::OneWay,
            conflict_resolution: ConflictResolution::LocalWins,
            auto_sync: false,
        }
    }
}

/// Synchronization strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SyncStrategy {
    /// One-way sync (system to context)
    OneWay,
    /// Two-way sync
    TwoWay,
    /// Manual sync only
    Manual,
}

/// Conflict resolution strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConflictResolution {
    /// Local value wins
    LocalWins,
    /// Remote value wins
    RemoteWins,
    /// Merge values
    Merge,
    /// Manual resolution
    Manual,
}

/// Variable store for environment variables
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariableStore {
    /// Environment variables
    variables: HashMap<String, EnvironmentVariable>,
    /// Variable groups
    groups: HashMap<String, VariableGroup>,
    /// Store configuration
    config: VariableStoreConfig,
    /// Store metadata
    metadata: StoreMetadata,
}

/// Environment variable with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentVariable {
    /// Variable name
    pub name: String,
    /// Variable value
    pub value: VariableValue,
    /// Variable metadata
    pub metadata: VariableMetadata,
    /// Variable source
    pub source: VariableSource,
    /// Variable status
    pub status: VariableStatus,
    /// Access permissions
    pub permissions: VariablePermissions,
}

/// Variable value with type information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariableValue {
    /// Raw string value
    pub raw_value: String,
    /// Parsed typed value
    pub typed_value: Option<TypedValue>,
    /// Value encryption status
    pub encrypted: bool,
    /// Value checksum
    pub checksum: String,
}

/// Typed values for environment variables
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TypedValue {
    /// String value
    String(String),
    /// Integer value
    Integer(i64),
    /// Float value
    Float(f64),
    /// Boolean value
    Boolean(bool),
    /// Path value
    Path(PathBuf),
    /// List value
    List(Vec<String>),
    /// JSON value
    Json(serde_json::Value),
}

/// Variable metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariableMetadata {
    /// Variable description
    pub description: Option<String>,
    /// Variable type
    pub variable_type: VariableType,
    /// Variable tags
    pub tags: HashSet<String>,
    /// Creation timestamp
    pub created_at: SystemTime,
    /// Last modified timestamp
    pub modified_at: SystemTime,
    /// Variable owner
    pub owner: Option<String>,
    /// Variable scope
    pub scope: VariableScope,
    /// Default value
    pub default_value: Option<String>,
    /// Is required
    pub required: bool,
    /// Is sensitive
    pub sensitive: bool,
}

/// Variable types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VariableType {
    /// String type
    String,
    /// Integer type
    Integer,
    /// Float type
    Float,
    /// Boolean type
    Boolean,
    /// Path type
    Path,
    /// URL type
    Url,
    /// Email type
    Email,
    /// JSON type
    Json,
    /// List type
    List,
    /// Unknown type
    Unknown,
}

/// Variable scopes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VariableScope {
    /// Global scope
    Global,
    /// Process scope
    Process,
    /// Thread scope
    Thread,
    /// Session scope
    Session,
    /// Custom scope
    Custom,
}

/// Variable source information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariableSource {
    /// Source type
    pub source_type: SourceType,
    /// Source identifier
    pub source_id: String,
    /// Source location
    pub location: Option<String>,
    /// Source priority
    pub priority: i32,
    /// Source timestamp
    pub timestamp: SystemTime,
}

/// Variable source types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SourceType {
    /// System environment
    System,
    /// Configuration file
    File,
    /// Command line argument
    CommandLine,
    /// User input
    User,
    /// Default value
    Default,
    /// Inherited from parent
    Inherited,
    /// External provider
    External,
}

/// Variable status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VariableStatus {
    /// Active variable
    Active,
    /// Deprecated variable
    Deprecated,
    /// Disabled variable
    Disabled,
    /// Deleted variable
    Deleted,
}

/// Variable permissions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariablePermissions {
    /// Read permission
    pub read: bool,
    /// Write permission
    pub write: bool,
    /// Delete permission
    pub delete: bool,
    /// Export permission
    pub export: bool,
}

impl Default for VariablePermissions {
    fn default() -> Self {
        Self {
            read: true,
            write: true,
            delete: true,
            export: true,
        }
    }
}

/// Variable group for organizing related variables
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariableGroup {
    /// Group name
    pub name: String,
    /// Group description
    pub description: Option<String>,
    /// Variable names in group
    pub variables: Vec<String>,
    /// Group metadata
    pub metadata: HashMap<String, String>,
    /// Group permissions
    pub permissions: VariablePermissions,
}

/// Variable store configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariableStoreConfig {
    /// Case sensitivity
    pub case_sensitive: bool,
    /// Variable name validation
    pub name_validation: NameValidation,
    /// Value size limits
    pub value_limits: ValueLimits,
    /// Store persistence
    pub persistence: PersistenceConfig,
}

impl Default for VariableStoreConfig {
    fn default() -> Self {
        Self {
            case_sensitive: true,
            name_validation: NameValidation::default(),
            value_limits: ValueLimits::default(),
            persistence: PersistenceConfig::default(),
        }
    }
}

/// Name validation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NameValidation {
    /// Allow special characters
    pub allow_special_chars: bool,
    /// Minimum name length
    pub min_length: usize,
    /// Maximum name length
    pub max_length: usize,
    /// Valid name pattern
    pub pattern: Option<String>,
    /// Reserved names
    pub reserved_names: HashSet<String>,
}

impl Default for NameValidation {
    fn default() -> Self {
        Self {
            allow_special_chars: true,
            min_length: 1,
            max_length: 256,
            pattern: Some(r"^[A-Za-z_][A-Za-z0-9_]*$".to_string()),
            reserved_names: HashSet::new(),
        }
    }
}

/// Value size limits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValueLimits {
    /// Maximum value size in bytes
    pub max_size: usize,
    /// Maximum number of variables
    pub max_variables: Option<usize>,
    /// Compression for large values
    pub compress_large_values: bool,
    /// Compression threshold
    pub compression_threshold: usize,
}

impl Default for ValueLimits {
    fn default() -> Self {
        Self {
            max_size: 64 * 1024, // 64KB per variable
            max_variables: Some(10000),
            compress_large_values: false,
            compression_threshold: 1024, // 1KB
        }
    }
}

/// Persistence configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistenceConfig {
    /// Enable persistence
    pub enabled: bool,
    /// Persistence backend
    pub backend: PersistenceBackend,
    /// Persistence location
    pub location: Option<PathBuf>,
    /// Auto-save enabled
    pub auto_save: bool,
    /// Save interval
    pub save_interval: Duration,
}

impl Default for PersistenceConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            backend: PersistenceBackend::File,
            location: None,
            auto_save: false,
            save_interval: Duration::from_secs(300), // 5 minutes
        }
    }
}

/// Persistence backends
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PersistenceBackend {
    /// File-based persistence
    File,
    /// Database persistence
    Database,
    /// Key-value store persistence
    KeyValue,
    /// Custom persistence
    Custom,
}

/// Store metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoreMetadata {
    /// Store version
    pub version: String,
    /// Created timestamp
    pub created_at: SystemTime,
    /// Last updated timestamp
    pub updated_at: SystemTime,
    /// Store size in bytes
    pub size_bytes: usize,
    /// Variable count
    pub variable_count: usize,
}

impl Default for StoreMetadata {
    fn default() -> Self {
        let now = SystemTime::now();
        Self {
            version: "1.0.0".to_string(),
            created_at: now,
            updated_at: now,
            size_bytes: 0,
            variable_count: 0,
        }
    }
}

/// Variable validator trait
pub trait VariableValidator: Send + Sync + Debug {
    /// Validator name
    fn name(&self) -> &str;

    /// Validate variable
    fn validate(&self, variable: &EnvironmentVariable) -> ContextResult<ValidationResult>;

    /// Check if validator applies to variable
    fn applies_to(&self, variable_name: &str) -> bool;
}

/// Validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    /// Validation success
    pub valid: bool,
    /// Validation errors
    pub errors: Vec<ValidationError>,
    /// Validation warnings
    pub warnings: Vec<ValidationWarning>,
}

/// Validation error
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationError {
    /// Error code
    pub code: String,
    /// Error message
    pub message: String,
    /// Error severity
    pub severity: ValidationSeverity,
}

/// Validation warning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationWarning {
    /// Warning code
    pub code: String,
    /// Warning message
    pub message: String,
}

/// Environment provider trait
pub trait EnvironmentProvider: Send + Sync + Debug {
    /// Provider name
    fn name(&self) -> &str;

    /// Provider priority
    fn priority(&self) -> i32;

    /// Load variables from provider
    fn load_variables(&self) -> ContextResult<HashMap<String, EnvironmentVariable>>;

    /// Save variables to provider
    fn save_variables(&self, variables: &HashMap<String, EnvironmentVariable>) -> ContextResult<()>;

    /// Check if provider supports watching
    fn supports_watch(&self) -> bool;

    /// Start watching for changes
    fn start_watch(&self) -> ContextResult<()>;

    /// Stop watching for changes
    fn stop_watch(&self) -> ContextResult<()>;
}

/// Isolation manager for environment isolation
#[derive(Debug)]
pub struct IsolationManager {
    /// Isolation configuration
    config: Arc<RwLock<IsolationConfig>>,
    /// Active isolation contexts
    contexts: Arc<RwLock<HashMap<String, IsolationContext>>>,
    /// Namespace manager
    namespace_manager: Arc<NamespaceManager>,
}

/// Isolation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IsolationConfig {
    /// Default isolation level
    pub default_level: EnvironmentIsolationLevel,
    /// Namespace configuration
    pub namespace_config: NamespaceConfig,
    /// Sandbox configuration
    pub sandbox_config: SandboxConfig,
}

/// Namespace configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NamespaceConfig {
    /// Enable namespaces
    pub enabled: bool,
    /// Default namespace
    pub default_namespace: String,
    /// Namespace isolation
    pub isolation: bool,
    /// Cross-namespace access
    pub cross_namespace_access: bool,
}

impl Default for NamespaceConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            default_namespace: "default".to_string(),
            isolation: true,
            cross_namespace_access: false,
        }
    }
}

/// Sandbox configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SandboxConfig {
    /// Enable sandboxing
    pub enabled: bool,
    /// Sandbox type
    pub sandbox_type: SandboxType,
    /// Resource limits
    pub resource_limits: HashMap<String, u64>,
    /// Network isolation
    pub network_isolation: bool,
    /// File system isolation
    pub filesystem_isolation: bool,
}

impl Default for SandboxConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            sandbox_type: SandboxType::Process,
            resource_limits: HashMap::new(),
            network_isolation: false,
            filesystem_isolation: false,
        }
    }
}

/// Sandbox types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SandboxType {
    /// Process-level sandbox
    Process,
    /// Container-level sandbox
    Container,
    /// Virtual machine sandbox
    VirtualMachine,
    /// Custom sandbox
    Custom,
}

/// Isolation context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IsolationContext {
    /// Context ID
    pub context_id: String,
    /// Isolation level
    pub isolation_level: EnvironmentIsolationLevel,
    /// Namespace
    pub namespace: String,
    /// Parent context
    pub parent_context: Option<String>,
    /// Child contexts
    pub child_contexts: Vec<String>,
    /// Isolation metadata
    pub metadata: HashMap<String, String>,
}

/// Namespace manager
#[derive(Debug)]
pub struct NamespaceManager {
    /// Active namespaces
    namespaces: Arc<RwLock<HashMap<String, Namespace>>>,
    /// Namespace hierarchy
    hierarchy: Arc<RwLock<NamespaceHierarchy>>,
}

/// Namespace definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Namespace {
    /// Namespace name
    pub name: String,
    /// Namespace description
    pub description: Option<String>,
    /// Parent namespace
    pub parent: Option<String>,
    /// Child namespaces
    pub children: Vec<String>,
    /// Namespace variables
    pub variables: HashMap<String, EnvironmentVariable>,
    /// Namespace permissions
    pub permissions: NamespacePermissions,
    /// Created timestamp
    pub created_at: SystemTime,
}

/// Namespace permissions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NamespacePermissions {
    /// Read access
    pub read: bool,
    /// Write access
    pub write: bool,
    /// Create child namespaces
    pub create_child: bool,
    /// Delete namespace
    pub delete: bool,
}

impl Default for NamespacePermissions {
    fn default() -> Self {
        Self {
            read: true,
            write: true,
            create_child: true,
            delete: true,
        }
    }
}

/// Namespace hierarchy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NamespaceHierarchy {
    /// Root namespaces
    pub roots: Vec<String>,
    /// Namespace tree
    pub tree: HashMap<String, Vec<String>>,
    /// Hierarchy depth
    pub depth: HashMap<String, usize>,
}

impl Default for NamespaceHierarchy {
    fn default() -> Self {
        Self {
            roots: vec!["default".to_string()],
            tree: HashMap::new(),
            depth: HashMap::new(),
        }
    }
}

/// Environment policies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentPolicies {
    /// Variable naming policy
    pub naming_policy: NamingPolicy,
    /// Value policy
    pub value_policy: ValuePolicy,
    /// Security policy
    pub security_policy: SecurityPolicy,
    /// Compliance policy
    pub compliance_policy: CompliancePolicy,
}

impl Default for EnvironmentPolicies {
    fn default() -> Self {
        Self {
            naming_policy: NamingPolicy::default(),
            value_policy: ValuePolicy::default(),
            security_policy: SecurityPolicy::default(),
            compliance_policy: CompliancePolicy::default(),
        }
    }
}

/// Naming policy for variables
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NamingPolicy {
    /// Enforce naming convention
    pub enforce_convention: bool,
    /// Naming convention
    pub convention: NamingConvention,
    /// Allowed prefixes
    pub allowed_prefixes: Vec<String>,
    /// Forbidden names
    pub forbidden_names: HashSet<String>,
}

impl Default for NamingPolicy {
    fn default() -> Self {
        Self {
            enforce_convention: false,
            convention: NamingConvention::UpperCase,
            allowed_prefixes: Vec::new(),
            forbidden_names: HashSet::new(),
        }
    }
}

/// Naming conventions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NamingConvention {
    /// UPPER_CASE naming
    UpperCase,
    /// lower_case naming
    LowerCase,
    /// PascalCase naming
    PascalCase,
    /// camelCase naming
    CamelCase,
    /// kebab-case naming
    KebabCase,
    /// Custom convention
    Custom,
}

/// Value policy for variable values
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValuePolicy {
    /// Maximum value length
    pub max_length: Option<usize>,
    /// Allowed characters
    pub allowed_chars: Option<String>,
    /// Forbidden patterns
    pub forbidden_patterns: Vec<String>,
    /// Require encryption for sensitive values
    pub encrypt_sensitive: bool,
}

impl Default for ValuePolicy {
    fn default() -> Self {
        Self {
            max_length: Some(4096),
            allowed_chars: None,
            forbidden_patterns: Vec::new(),
            encrypt_sensitive: false,
        }
    }
}

/// Security policy for environment variables
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityPolicy {
    /// Classify sensitive variables
    pub classify_sensitive: bool,
    /// Sensitive patterns
    pub sensitive_patterns: Vec<String>,
    /// Require secure transport
    pub secure_transport: bool,
    /// Audit all access
    pub audit_access: bool,
}

impl Default for SecurityPolicy {
    fn default() -> Self {
        Self {
            classify_sensitive: true,
            sensitive_patterns: vec!["*PASSWORD*".to_string(), "*SECRET*".to_string()],
            secure_transport: false,
            audit_access: false,
        }
    }
}

/// Compliance policy for regulatory requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompliancePolicy {
    /// Enable compliance checking
    pub enabled: bool,
    /// Compliance standards
    pub standards: Vec<ComplianceStandard>,
    /// Data residency requirements
    pub data_residency: Option<String>,
    /// Retention requirements
    pub retention_requirements: Option<Duration>,
}

impl Default for CompliancePolicy {
    fn default() -> Self {
        Self {
            enabled: false,
            standards: Vec::new(),
            data_residency: None,
            retention_requirements: None,
        }
    }
}

/// Compliance standards
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ComplianceStandard {
    /// GDPR compliance
    GDPR,
    /// HIPAA compliance
    HIPAA,
    /// SOX compliance
    SOX,
    /// PCI-DSS compliance
    PCIDSS,
    /// Custom compliance
    Custom,
}

/// Change tracker for environment changes
#[derive(Debug, Clone, Default)]
pub struct ChangeTracker {
    /// Change history
    changes: Vec<EnvironmentChange>,
    /// Change listeners
    listeners: Vec<Box<dyn ChangeListener>>,
    /// Tracking configuration
    config: ChangeTrackingConfig,
}

/// Environment change event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentChange {
    /// Change ID
    pub change_id: String,
    /// Change type
    pub change_type: ChangeType,
    /// Variable name
    pub variable_name: String,
    /// Old value
    pub old_value: Option<String>,
    /// New value
    pub new_value: Option<String>,
    /// Change timestamp
    pub timestamp: SystemTime,
    /// Change source
    pub source: String,
    /// Change metadata
    pub metadata: HashMap<String, String>,
}

/// Change types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChangeType {
    /// Variable created
    Created,
    /// Variable updated
    Updated,
    /// Variable deleted
    Deleted,
    /// Variable renamed
    Renamed,
}

/// Change listener trait
pub trait ChangeListener: Send + Sync + Debug {
    /// Handle environment change
    fn on_change(&mut self, change: &EnvironmentChange) -> ContextResult<()>;

    /// Get listener ID
    fn listener_id(&self) -> &str;

    /// Check if listener is interested in variable
    fn is_interested_in(&self, variable_name: &str) -> bool;
}

/// Change tracking configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChangeTrackingConfig {
    /// Enable change tracking
    pub enabled: bool,
    /// Maximum changes to track
    pub max_changes: Option<usize>,
    /// Change retention period
    pub retention_period: Duration,
    /// Track value changes
    pub track_values: bool,
    /// Compress old changes
    pub compress_old: bool,
}

impl Default for ChangeTrackingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_changes: Some(10000),
            retention_period: Duration::from_secs(30 * 24 * 60 * 60), // 30 days
            track_values: false, // Don't track sensitive values
            compress_old: true,
        }
    }
}

/// Environment metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EnvironmentMetrics {
    /// Total variables
    pub total_variables: usize,
    /// Active variables
    pub active_variables: usize,
    /// Sensitive variables
    pub sensitive_variables: usize,
    /// Variable reads per second
    pub reads_per_second: f64,
    /// Variable writes per second
    pub writes_per_second: f64,
    /// Total changes tracked
    pub total_changes: u64,
    /// Validation errors
    pub validation_errors: u64,
    /// Memory usage in bytes
    pub memory_usage: usize,
    /// Cache hit ratio
    pub cache_hit_ratio: f64,
}

impl EnvironmentContext {
    /// Create a new environment context
    pub fn new(context_id: String) -> ContextResult<Self> {
        let context = Self {
            context_id: context_id.clone(),
            env_manager: Arc::new(EnvironmentManager::new()),
            var_store: Arc::new(RwLock::new(VariableStore::new())),
            policies: Arc::new(RwLock::new(EnvironmentPolicies::default())),
            change_tracker: Arc::new(Mutex::new(ChangeTracker::default())),
            state: Arc::new(RwLock::new(ContextState::Initializing)),
            metadata: Arc::new(RwLock::new(ContextMetadata::default())),
            metrics: Arc::new(Mutex::new(EnvironmentMetrics::default())),
        };

        // Update state to active
        *context.state.write().unwrap() = ContextState::Active;

        // Initialize with system environment
        context.load_system_environment()?;

        Ok(context)
    }

    /// Set environment variable
    pub fn set_var(&self, name: &str, value: &str) -> ContextResult<()> {
        let var = EnvironmentVariable {
            name: name.to_string(),
            value: VariableValue {
                raw_value: value.to_string(),
                typed_value: None,
                encrypted: false,
                checksum: "".to_string(),
            },
            metadata: VariableMetadata {
                description: None,
                variable_type: VariableType::String,
                tags: HashSet::new(),
                created_at: SystemTime::now(),
                modified_at: SystemTime::now(),
                owner: None,
                scope: VariableScope::Process,
                default_value: None,
                required: false,
                sensitive: false,
            },
            source: VariableSource {
                source_type: SourceType::User,
                source_id: "user".to_string(),
                location: None,
                priority: 100,
                timestamp: SystemTime::now(),
            },
            status: VariableStatus::Active,
            permissions: VariablePermissions::default(),
        };

        let mut store = self.var_store.write().unwrap();
        let old_value = store.variables.get(name).map(|v| v.value.raw_value.clone());
        store.variables.insert(name.to_string(), var);

        // Track change
        let change = EnvironmentChange {
            change_id: Uuid::new_v4().to_string(),
            change_type: if old_value.is_some() { ChangeType::Updated } else { ChangeType::Created },
            variable_name: name.to_string(),
            old_value,
            new_value: Some(value.to_string()),
            timestamp: SystemTime::now(),
            source: "user".to_string(),
            metadata: HashMap::new(),
        };

        drop(store);
        self.track_change(change)?;

        // Update metrics
        let mut metrics = self.metrics.lock().unwrap();
        let store = self.var_store.read().unwrap();
        metrics.total_variables = store.variables.len();
        metrics.active_variables = store.variables.values()
            .filter(|v| v.status == VariableStatus::Active)
            .count();

        Ok(())
    }

    /// Get environment variable
    pub fn get_var(&self, name: &str) -> ContextResult<Option<String>> {
        let store = self.var_store.read().unwrap();
        Ok(store.variables.get(name).map(|v| v.value.raw_value.clone()))
    }

    /// Remove environment variable
    pub fn remove_var(&self, name: &str) -> ContextResult<Option<String>> {
        let mut store = self.var_store.write().unwrap();
        let removed = store.variables.remove(name);

        if let Some(var) = &removed {
            // Track change
            let change = EnvironmentChange {
                change_id: Uuid::new_v4().to_string(),
                change_type: ChangeType::Deleted,
                variable_name: name.to_string(),
                old_value: Some(var.value.raw_value.clone()),
                new_value: None,
                timestamp: SystemTime::now(),
                source: "user".to_string(),
                metadata: HashMap::new(),
            };

            drop(store);
            self.track_change(change)?;

            return Ok(Some(var.value.raw_value));
        }

        Ok(None)
    }

    /// Get all environment variables
    pub fn get_all_vars(&self) -> ContextResult<HashMap<String, String>> {
        let store = self.var_store.read().unwrap();
        let vars = store.variables.iter()
            .filter(|(_, v)| v.status == VariableStatus::Active)
            .map(|(k, v)| (k.clone(), v.value.raw_value.clone()))
            .collect();
        Ok(vars)
    }

    /// Load system environment
    pub fn load_system_environment(&self) -> ContextResult<()> {
        let mut store = self.var_store.write().unwrap();

        for (name, value) in env::vars() {
            let var = EnvironmentVariable {
                name: name.clone(),
                value: VariableValue {
                    raw_value: value,
                    typed_value: None,
                    encrypted: false,
                    checksum: "".to_string(),
                },
                metadata: VariableMetadata {
                    description: None,
                    variable_type: VariableType::String,
                    tags: HashSet::new(),
                    created_at: SystemTime::now(),
                    modified_at: SystemTime::now(),
                    owner: None,
                    scope: VariableScope::Global,
                    default_value: None,
                    required: false,
                    sensitive: false,
                },
                source: VariableSource {
                    source_type: SourceType::System,
                    source_id: "system".to_string(),
                    location: None,
                    priority: 1,
                    timestamp: SystemTime::now(),
                },
                status: VariableStatus::Active,
                permissions: VariablePermissions::default(),
            };

            store.variables.insert(name, var);
        }

        Ok(())
    }

    /// Validate environment
    pub fn validate(&self) -> ContextResult<ValidationResult> {
        let store = self.var_store.read().unwrap();
        let mut result = ValidationResult {
            valid: true,
            errors: Vec::new(),
            warnings: Vec::new(),
        };

        // Validate each variable
        for (_, variable) in &store.variables {
            let validators = self.env_manager.validators.read().unwrap();
            for validator in validators.iter() {
                if validator.applies_to(&variable.name) {
                    match validator.validate(variable) {
                        Ok(var_result) => {
                            if !var_result.valid {
                                result.valid = false;
                                result.errors.extend(var_result.errors);
                                result.warnings.extend(var_result.warnings);
                            }
                        },
                        Err(e) => {
                            result.valid = false;
                            result.errors.push(ValidationError {
                                code: "validator_error".to_string(),
                                message: e.to_string(),
                                severity: ValidationSeverity::Error,
                            });
                        }
                    }
                }
            }
        }

        Ok(result)
    }

    /// Track environment change
    fn track_change(&self, change: EnvironmentChange) -> ContextResult<()> {
        let mut tracker = self.change_tracker.lock().unwrap();
        tracker.changes.push(change);

        // Limit change history size
        if let Some(max_changes) = tracker.config.max_changes {
            if tracker.changes.len() > max_changes {
                tracker.changes.drain(0..tracker.changes.len() - max_changes);
            }
        }

        Ok(())
    }

    /// Get environment metrics
    pub fn get_metrics(&self) -> ContextResult<EnvironmentMetrics> {
        let metrics = self.metrics.lock().unwrap();
        Ok(metrics.clone())
    }
}

impl VariableStore {
    /// Create a new variable store
    pub fn new() -> Self {
        Self {
            variables: HashMap::new(),
            groups: HashMap::new(),
            config: VariableStoreConfig::default(),
            metadata: StoreMetadata::default(),
        }
    }
}

impl EnvironmentManager {
    /// Create a new environment manager
    pub fn new() -> Self {
        Self {
            config: Arc::new(RwLock::new(EnvironmentConfig::default())),
            validators: Arc::new(RwLock::new(Vec::new())),
            providers: Arc::new(RwLock::new(Vec::new())),
            isolation: Arc::new(IsolationManager::new()),
        }
    }
}

impl IsolationManager {
    /// Create a new isolation manager
    pub fn new() -> Self {
        Self {
            config: Arc::new(RwLock::new(IsolationConfig {
                default_level: EnvironmentIsolationLevel::Process,
                namespace_config: NamespaceConfig::default(),
                sandbox_config: SandboxConfig::default(),
            })),
            contexts: Arc::new(RwLock::new(HashMap::new())),
            namespace_manager: Arc::new(NamespaceManager::new()),
        }
    }
}

impl NamespaceManager {
    /// Create a new namespace manager
    pub fn new() -> Self {
        Self {
            namespaces: Arc::new(RwLock::new(HashMap::new())),
            hierarchy: Arc::new(RwLock::new(NamespaceHierarchy::default())),
        }
    }
}

impl ExecutionContextTrait for EnvironmentContext {
    fn id(&self) -> &str {
        &self.context_id
    }

    fn context_type(&self) -> ContextType {
        ContextType::Extension("environment".to_string())
    }

    fn state(&self) -> ContextState {
        *self.state.read().unwrap()
    }

    fn is_active(&self) -> bool {
        matches!(self.state(), ContextState::Active)
    }

    fn metadata(&self) -> &ContextMetadata {
        // Simplified implementation
        unsafe { &*(self.metadata.read().unwrap().as_ref() as *const ContextMetadata) }
    }

    fn validate(&self) -> Result<(), ContextError> {
        let result = self.validate()?;
        if result.valid {
            Ok(())
        } else {
            Err(ContextError::validation("Environment validation failed"))
        }
    }

    fn clone_with_id(&self, new_id: String) -> Result<Box<dyn ExecutionContextTrait>, ContextError> {
        let new_context = EnvironmentContext::new(new_id)?;
        Ok(Box::new(new_context))
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

impl Default for VariableStore {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for EnvironmentManager {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for IsolationManager {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for NamespaceManager {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_environment_context_creation() {
        let context = EnvironmentContext::new("test-env".to_string()).unwrap();
        assert_eq!(context.id(), "test-env");
        assert_eq!(context.context_type(), ContextType::Extension("environment".to_string()));
        assert!(context.is_active());
    }

    #[test]
    fn test_environment_variable_operations() {
        let context = EnvironmentContext::new("test-vars".to_string()).unwrap();

        // Set variable
        context.set_var("TEST_VAR", "test_value").unwrap();

        // Get variable
        let value = context.get_var("TEST_VAR").unwrap();
        assert_eq!(value, Some("test_value".to_string()));

        // Update variable
        context.set_var("TEST_VAR", "updated_value").unwrap();
        let updated_value = context.get_var("TEST_VAR").unwrap();
        assert_eq!(updated_value, Some("updated_value".to_string()));

        // Remove variable
        let removed = context.remove_var("TEST_VAR").unwrap();
        assert_eq!(removed, Some("updated_value".to_string()));

        // Check variable is removed
        let after_removal = context.get_var("TEST_VAR").unwrap();
        assert_eq!(after_removal, None);
    }

    #[test]
    fn test_get_all_variables() {
        let context = EnvironmentContext::new("test-all".to_string()).unwrap();

        context.set_var("VAR1", "value1").unwrap();
        context.set_var("VAR2", "value2").unwrap();
        context.set_var("VAR3", "value3").unwrap();

        let all_vars = context.get_all_vars().unwrap();

        // Should include our test variables plus system environment variables
        assert!(all_vars.contains_key("VAR1"));
        assert!(all_vars.contains_key("VAR2"));
        assert!(all_vars.contains_key("VAR3"));
        assert_eq!(all_vars.get("VAR1"), Some(&"value1".to_string()));
        assert_eq!(all_vars.get("VAR2"), Some(&"value2".to_string()));
        assert_eq!(all_vars.get("VAR3"), Some(&"value3".to_string()));
    }

    #[test]
    fn test_variable_store() {
        let store = VariableStore::new();
        assert!(store.variables.is_empty());
        assert!(store.groups.is_empty());
        assert!(store.config.case_sensitive);
        assert_eq!(store.metadata.version, "1.0.0");
    }

    #[test]
    fn test_environment_config() {
        let config = EnvironmentConfig::default();
        assert!(config.isolation_enabled);
        assert_eq!(config.isolation_level, EnvironmentIsolationLevel::Process);
        assert!(config.inheritance.inherit_parent);
        assert!(config.inheritance.inherit_system);
        assert_eq!(config.inheritance.strategy, InheritanceStrategy::Merge);
    }

    #[test]
    fn test_validation_config() {
        let config = ValidationConfig::default();
        assert!(config.enabled);
        assert!(!config.strict_mode);
        assert!(config.validate_on_change);
        assert!(config.rules.is_empty());
        assert!(config.custom_validators.is_empty());
    }

    #[test]
    fn test_variable_metadata() {
        let now = SystemTime::now();
        let metadata = VariableMetadata {
            description: Some("Test variable".to_string()),
            variable_type: VariableType::String,
            tags: HashSet::new(),
            created_at: now,
            modified_at: now,
            owner: Some("test_user".to_string()),
            scope: VariableScope::Process,
            default_value: None,
            required: true,
            sensitive: false,
        };

        assert_eq!(metadata.description, Some("Test variable".to_string()));
        assert_eq!(metadata.variable_type, VariableType::String);
        assert_eq!(metadata.owner, Some("test_user".to_string()));
        assert_eq!(metadata.scope, VariableScope::Process);
        assert!(metadata.required);
        assert!(!metadata.sensitive);
    }

    #[test]
    fn test_variable_permissions() {
        let permissions = VariablePermissions::default();
        assert!(permissions.read);
        assert!(permissions.write);
        assert!(permissions.delete);
        assert!(permissions.export);
    }

    #[test]
    fn test_environment_change() {
        let change = EnvironmentChange {
            change_id: Uuid::new_v4().to_string(),
            change_type: ChangeType::Updated,
            variable_name: "TEST_VAR".to_string(),
            old_value: Some("old_value".to_string()),
            new_value: Some("new_value".to_string()),
            timestamp: SystemTime::now(),
            source: "user".to_string(),
            metadata: HashMap::new(),
        };

        assert_eq!(change.change_type, ChangeType::Updated);
        assert_eq!(change.variable_name, "TEST_VAR");
        assert_eq!(change.old_value, Some("old_value".to_string()));
        assert_eq!(change.new_value, Some("new_value".to_string()));
        assert_eq!(change.source, "user");
    }

    #[test]
    fn test_naming_convention() {
        assert_eq!(NamingConvention::UpperCase, NamingConvention::UpperCase);
        assert_ne!(NamingConvention::UpperCase, NamingConvention::LowerCase);
    }

    #[test]
    fn test_isolation_levels() {
        assert_eq!(EnvironmentIsolationLevel::None, EnvironmentIsolationLevel::None);
        assert_ne!(EnvironmentIsolationLevel::Process, EnvironmentIsolationLevel::Thread);
    }
}