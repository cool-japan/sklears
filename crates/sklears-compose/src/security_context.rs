//! Security Context Module
//!
//! Provides comprehensive security management for execution contexts, including
//! authentication, authorization, encryption, audit trails, and security policies.

use std::{
    collections::{HashMap, HashSet},
    sync::{Arc, RwLock, Mutex},
    time::{Duration, SystemTime, UNIX_EPOCH},
    fmt::{Debug, Display},
    hash::{Hash, Hasher, DefaultHasher},
};
use serde::{Serialize, Deserialize};
use uuid::Uuid;

use crate::context_core::{
    ExecutionContextTrait, ContextType, ContextState, ContextMetadata, ContextError, ContextResult,
    ContextEvent, IsolationLevel, ContextPriority
};

/// Security context for authentication, authorization, and security management
#[derive(Debug)]
pub struct SecurityContext {
    /// Context identifier
    context_id: String,
    /// Security configuration
    config: Arc<RwLock<SecurityConfig>>,
    /// Authentication manager
    auth_manager: Arc<AuthenticationManager>,
    /// Authorization manager
    authz_manager: Arc<AuthorizationManager>,
    /// Encryption manager
    encryption_manager: Arc<EncryptionManager>,
    /// Audit trail manager
    audit_manager: Arc<Mutex<AuditTrailManager>>,
    /// Security policies
    policies: Arc<RwLock<SecurityPolicies>>,
    /// Active sessions
    sessions: Arc<RwLock<HashMap<String, SecuritySession>>>,
    /// Context state
    state: Arc<RwLock<ContextState>>,
    /// Metadata
    metadata: Arc<RwLock<ContextMetadata>>,
    /// Security metrics
    metrics: Arc<Mutex<SecurityMetrics>>,
}

/// Security configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConfig {
    /// Enable authentication
    pub enable_authentication: bool,
    /// Enable authorization
    pub enable_authorization: bool,
    /// Enable encryption
    pub enable_encryption: bool,
    /// Enable audit trail
    pub enable_audit_trail: bool,
    /// Security level
    pub security_level: SecurityLevel,
    /// Authentication methods
    pub auth_methods: Vec<AuthenticationMethod>,
    /// Session configuration
    pub session_config: SessionConfig,
    /// Encryption configuration
    pub encryption_config: EncryptionConfig,
    /// Audit configuration
    pub audit_config: AuditConfig,
    /// Password policy
    pub password_policy: PasswordPolicy,
    /// Access control configuration
    pub access_control: AccessControlConfig,
}

impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            enable_authentication: true,
            enable_authorization: true,
            enable_encryption: false,
            enable_audit_trail: true,
            security_level: SecurityLevel::Standard,
            auth_methods: vec![AuthenticationMethod::Token],
            session_config: SessionConfig::default(),
            encryption_config: EncryptionConfig::default(),
            audit_config: AuditConfig::default(),
            password_policy: PasswordPolicy::default(),
            access_control: AccessControlConfig::default(),
        }
    }
}

/// Security levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum SecurityLevel {
    /// Basic security
    Basic = 1,
    /// Standard security
    Standard = 2,
    /// High security
    High = 3,
    /// Maximum security
    Maximum = 4,
    /// Government/Military grade security
    TopSecret = 5,
}

/// Authentication methods
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AuthenticationMethod {
    /// Username/password authentication
    Password,
    /// Token-based authentication
    Token,
    /// API key authentication
    ApiKey,
    /// Certificate-based authentication
    Certificate,
    /// Multi-factor authentication
    MultiFactor,
    /// Single Sign-On
    SSO,
    /// OAuth2
    OAuth2,
    /// LDAP authentication
    LDAP,
    /// Biometric authentication
    Biometric,
    /// Custom authentication
    Custom(String),
}

/// Session configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionConfig {
    /// Session timeout
    pub timeout: Duration,
    /// Maximum concurrent sessions per user
    pub max_concurrent_sessions: usize,
    /// Session renewal threshold
    pub renewal_threshold: Duration,
    /// Enable session persistence
    pub enable_persistence: bool,
    /// Session storage type
    pub storage_type: SessionStorageType,
    /// Session ID length
    pub session_id_length: usize,
    /// Session security flags
    pub security_flags: SessionSecurityFlags,
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(3600), // 1 hour
            max_concurrent_sessions: 10,
            renewal_threshold: Duration::from_secs(300), // 5 minutes
            enable_persistence: false,
            storage_type: SessionStorageType::Memory,
            session_id_length: 32,
            security_flags: SessionSecurityFlags::default(),
        }
    }
}

/// Session storage types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SessionStorageType {
    /// In-memory storage
    Memory,
    /// Database storage
    Database,
    /// Redis storage
    Redis,
    /// File-based storage
    File,
    /// Custom storage
    Custom(String),
}

/// Session security flags
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionSecurityFlags {
    /// HTTP-only flag
    pub http_only: bool,
    /// Secure flag (HTTPS only)
    pub secure: bool,
    /// Same-site policy
    pub same_site: SameSitePolicy,
    /// Enable CSRF protection
    pub csrf_protection: bool,
}

impl Default for SessionSecurityFlags {
    fn default() -> Self {
        Self {
            http_only: true,
            secure: true,
            same_site: SameSitePolicy::Strict,
            csrf_protection: true,
        }
    }
}

/// Same-site policies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SameSitePolicy {
    /// Strict same-site policy
    Strict,
    /// Lax same-site policy
    Lax,
    /// No same-site restriction
    None,
}

/// Encryption configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptionConfig {
    /// Default encryption algorithm
    pub default_algorithm: EncryptionAlgorithm,
    /// Key derivation function
    pub key_derivation: KeyDerivationFunction,
    /// Key size in bits
    pub key_size: u32,
    /// Initialization vector size
    pub iv_size: u32,
    /// Salt size for key derivation
    pub salt_size: u32,
    /// Number of iterations for key derivation
    pub iterations: u32,
    /// Enable data at rest encryption
    pub encrypt_at_rest: bool,
    /// Enable data in transit encryption
    pub encrypt_in_transit: bool,
    /// Key rotation interval
    pub key_rotation_interval: Duration,
}

impl Default for EncryptionConfig {
    fn default() -> Self {
        Self {
            default_algorithm: EncryptionAlgorithm::AES256GCM,
            key_derivation: KeyDerivationFunction::PBKDF2,
            key_size: 256,
            iv_size: 96,
            salt_size: 128,
            iterations: 100000,
            encrypt_at_rest: false,
            encrypt_in_transit: true,
            key_rotation_interval: Duration::from_secs(24 * 60 * 60 * 30), // 30 days
        }
    }
}

/// Encryption algorithms
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EncryptionAlgorithm {
    /// AES 128-bit in GCM mode
    AES128GCM,
    /// AES 256-bit in GCM mode
    AES256GCM,
    /// ChaCha20-Poly1305
    ChaCha20Poly1305,
    /// AES 128-bit in CBC mode (legacy)
    AES128CBC,
    /// AES 256-bit in CBC mode (legacy)
    AES256CBC,
}

/// Key derivation functions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum KeyDerivationFunction {
    /// PBKDF2
    PBKDF2,
    /// Argon2
    Argon2,
    /// Scrypt
    Scrypt,
    /// Bcrypt
    Bcrypt,
}

/// Audit configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditConfig {
    /// Enable audit logging
    pub enabled: bool,
    /// Log level for audit events
    pub log_level: AuditLogLevel,
    /// Include request/response data
    pub include_data: bool,
    /// Maximum log entry size
    pub max_entry_size: usize,
    /// Log retention period
    pub retention_period: Duration,
    /// Audit storage type
    pub storage_type: AuditStorageType,
    /// Enable real-time alerts
    pub enable_alerts: bool,
    /// Events to audit
    pub audit_events: HashSet<AuditEventType>,
}

impl Default for AuditConfig {
    fn default() -> Self {
        let mut audit_events = HashSet::new();
        audit_events.insert(AuditEventType::Authentication);
        audit_events.insert(AuditEventType::Authorization);
        audit_events.insert(AuditEventType::DataAccess);
        audit_events.insert(AuditEventType::SecurityViolation);

        Self {
            enabled: true,
            log_level: AuditLogLevel::Info,
            include_data: false,
            max_entry_size: 10 * 1024, // 10KB
            retention_period: Duration::from_secs(365 * 24 * 60 * 60), // 1 year
            storage_type: AuditStorageType::File,
            enable_alerts: true,
            audit_events,
        }
    }
}

/// Audit log levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum AuditLogLevel {
    /// Error level
    Error = 1,
    /// Warning level
    Warn = 2,
    /// Info level
    Info = 3,
    /// Debug level
    Debug = 4,
}

/// Audit storage types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AuditStorageType {
    /// File-based storage
    File,
    /// Database storage
    Database,
    /// Syslog
    Syslog,
    /// Cloud storage
    Cloud,
    /// Custom storage
    Custom(String),
}

/// Audit event types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AuditEventType {
    /// Authentication events
    Authentication,
    /// Authorization events
    Authorization,
    /// Data access events
    DataAccess,
    /// Configuration changes
    ConfigurationChange,
    /// Security violations
    SecurityViolation,
    /// System events
    SystemEvent,
    /// User actions
    UserAction,
    /// Administrative actions
    AdminAction,
}

/// Password policy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PasswordPolicy {
    /// Minimum password length
    pub min_length: usize,
    /// Maximum password length
    pub max_length: usize,
    /// Require uppercase letters
    pub require_uppercase: bool,
    /// Require lowercase letters
    pub require_lowercase: bool,
    /// Require digits
    pub require_digits: bool,
    /// Require special characters
    pub require_special_chars: bool,
    /// Forbidden character patterns
    pub forbidden_patterns: Vec<String>,
    /// Password history size
    pub history_size: usize,
    /// Password expiration period
    pub expiration_period: Option<Duration>,
    /// Account lockout threshold
    pub lockout_threshold: u32,
    /// Account lockout duration
    pub lockout_duration: Duration,
}

impl Default for PasswordPolicy {
    fn default() -> Self {
        Self {
            min_length: 8,
            max_length: 128,
            require_uppercase: true,
            require_lowercase: true,
            require_digits: true,
            require_special_chars: true,
            forbidden_patterns: vec![
                "password".to_string(),
                "123456".to_string(),
                "qwerty".to_string(),
            ],
            history_size: 5,
            expiration_period: Some(Duration::from_secs(90 * 24 * 60 * 60)), // 90 days
            lockout_threshold: 5,
            lockout_duration: Duration::from_secs(15 * 60), // 15 minutes
        }
    }
}

/// Access control configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessControlConfig {
    /// Default access control model
    pub default_model: AccessControlModel,
    /// Enable role-based access control
    pub enable_rbac: bool,
    /// Enable attribute-based access control
    pub enable_abac: bool,
    /// Default permissions
    pub default_permissions: HashSet<Permission>,
    /// Permission inheritance enabled
    pub enable_inheritance: bool,
    /// Access decision strategy
    pub decision_strategy: AccessDecisionStrategy,
}

impl Default for AccessControlConfig {
    fn default() -> Self {
        Self {
            default_model: AccessControlModel::RBAC,
            enable_rbac: true,
            enable_abac: false,
            default_permissions: HashSet::new(),
            enable_inheritance: true,
            decision_strategy: AccessDecisionStrategy::Unanimous,
        }
    }
}

/// Access control models
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AccessControlModel {
    /// Role-based access control
    RBAC,
    /// Attribute-based access control
    ABAC,
    /// Discretionary access control
    DAC,
    /// Mandatory access control
    MAC,
    /// Access control lists
    ACL,
}

/// Access decision strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AccessDecisionStrategy {
    /// All voters must approve
    Unanimous,
    /// Majority of voters must approve
    Consensus,
    /// At least one voter must approve
    Affirmative,
}

/// Authentication manager
#[derive(Debug)]
pub struct AuthenticationManager {
    /// Authentication providers
    providers: Arc<RwLock<HashMap<String, Box<dyn AuthenticationProvider>>>>,
    /// Active authentication attempts
    active_attempts: Arc<Mutex<HashMap<String, AuthenticationAttempt>>>,
    /// Authentication cache
    auth_cache: Arc<Mutex<HashMap<String, CachedAuthResult>>>,
    /// Authentication metrics
    metrics: Arc<Mutex<AuthenticationMetrics>>,
}

/// Authentication provider trait
pub trait AuthenticationProvider: Send + Sync + Debug {
    /// Provider name
    fn name(&self) -> &str;

    /// Authentication method
    fn method(&self) -> AuthenticationMethod;

    /// Authenticate user
    fn authenticate(&self, credentials: &AuthenticationCredentials) -> ContextResult<AuthenticationResult>;

    /// Validate authentication token
    fn validate_token(&self, token: &str) -> ContextResult<TokenValidationResult>;

    /// Refresh authentication token
    fn refresh_token(&self, refresh_token: &str) -> ContextResult<TokenRefreshResult>;
}

/// Authentication credentials
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthenticationCredentials {
    /// Credential type
    pub credential_type: AuthenticationMethod,
    /// Username
    pub username: Option<String>,
    /// Password
    pub password: Option<String>,
    /// Token
    pub token: Option<String>,
    /// API key
    pub api_key: Option<String>,
    /// Certificate data
    pub certificate: Option<Vec<u8>>,
    /// Additional parameters
    pub additional_params: HashMap<String, String>,
}

/// Authentication result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthenticationResult {
    /// Authentication success
    pub success: bool,
    /// User principal
    pub principal: Option<UserPrincipal>,
    /// Authentication token
    pub token: Option<String>,
    /// Token expiration time
    pub token_expires_at: Option<SystemTime>,
    /// Refresh token
    pub refresh_token: Option<String>,
    /// Error message if authentication failed
    pub error_message: Option<String>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// User principal
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserPrincipal {
    /// User identifier
    pub user_id: String,
    /// Username
    pub username: String,
    /// Display name
    pub display_name: Option<String>,
    /// Email address
    pub email: Option<String>,
    /// User roles
    pub roles: HashSet<String>,
    /// User attributes
    pub attributes: HashMap<String, String>,
    /// Account status
    pub account_status: AccountStatus,
    /// Last login timestamp
    pub last_login: Option<SystemTime>,
}

/// Account status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AccountStatus {
    /// Active account
    Active,
    /// Inactive account
    Inactive,
    /// Locked account
    Locked,
    /// Suspended account
    Suspended,
    /// Disabled account
    Disabled,
    /// Expired account
    Expired,
}

/// Authentication attempt
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthenticationAttempt {
    /// Attempt ID
    pub attempt_id: String,
    /// Username
    pub username: String,
    /// Authentication method
    pub method: AuthenticationMethod,
    /// Attempt timestamp
    pub timestamp: SystemTime,
    /// Client IP address
    pub client_ip: Option<String>,
    /// User agent
    pub user_agent: Option<String>,
    /// Attempt result
    pub result: Option<bool>,
}

/// Cached authentication result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedAuthResult {
    /// Authentication result
    pub result: AuthenticationResult,
    /// Cache expiration time
    pub expires_at: SystemTime,
    /// Cache hit count
    pub hit_count: u64,
}

/// Token validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenValidationResult {
    /// Token is valid
    pub valid: bool,
    /// User principal
    pub principal: Option<UserPrincipal>,
    /// Token expiration time
    pub expires_at: Option<SystemTime>,
    /// Error message if invalid
    pub error_message: Option<String>,
}

/// Token refresh result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenRefreshResult {
    /// Refresh success
    pub success: bool,
    /// New access token
    pub access_token: Option<String>,
    /// New refresh token
    pub refresh_token: Option<String>,
    /// Token expiration time
    pub expires_at: Option<SystemTime>,
    /// Error message if failed
    pub error_message: Option<String>,
}

/// Authentication metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AuthenticationMetrics {
    /// Total authentication attempts
    pub total_attempts: u64,
    /// Successful authentications
    pub successful_attempts: u64,
    /// Failed authentications
    pub failed_attempts: u64,
    /// Blocked attempts
    pub blocked_attempts: u64,
    /// Average authentication time
    pub avg_auth_time: Duration,
    /// Authentication by method
    pub by_method: HashMap<AuthenticationMethod, u64>,
}

/// Authorization manager
#[derive(Debug)]
pub struct AuthorizationManager {
    /// Role definitions
    roles: Arc<RwLock<HashMap<String, Role>>>,
    /// Permission definitions
    permissions: Arc<RwLock<HashMap<String, Permission>>>,
    /// User role assignments
    user_roles: Arc<RwLock<HashMap<String, HashSet<String>>>>,
    /// Resource permissions
    resource_permissions: Arc<RwLock<HashMap<String, HashSet<Permission>>>>,
    /// Authorization cache
    authz_cache: Arc<Mutex<HashMap<String, CachedAuthzResult>>>,
    /// Authorization metrics
    metrics: Arc<Mutex<AuthorizationMetrics>>,
}

/// Role definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Role {
    /// Role name
    pub name: String,
    /// Role description
    pub description: Option<String>,
    /// Role permissions
    pub permissions: HashSet<Permission>,
    /// Parent roles (for inheritance)
    pub parent_roles: HashSet<String>,
    /// Role metadata
    pub metadata: HashMap<String, String>,
}

/// Permission definition
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Permission {
    /// Permission name
    pub name: String,
    /// Resource type
    pub resource_type: String,
    /// Action
    pub action: String,
    /// Resource instance (optional)
    pub resource_instance: Option<String>,
    /// Conditions
    pub conditions: Vec<PermissionCondition>,
}

/// Permission condition
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PermissionCondition {
    /// Attribute name
    pub attribute: String,
    /// Condition operator
    pub operator: ConditionOperator,
    /// Expected value
    pub value: String,
}

/// Condition operators
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ConditionOperator {
    /// Equals
    Equals,
    /// Not equals
    NotEquals,
    /// Greater than
    GreaterThan,
    /// Less than
    LessThan,
    /// Contains
    Contains,
    /// Starts with
    StartsWith,
    /// Ends with
    EndsWith,
    /// Regular expression match
    Regex,
}

/// Cached authorization result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedAuthzResult {
    /// Authorization decision
    pub allowed: bool,
    /// Cache expiration time
    pub expires_at: SystemTime,
    /// Cache hit count
    pub hit_count: u64,
}

/// Authorization metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AuthorizationMetrics {
    /// Total authorization requests
    pub total_requests: u64,
    /// Allowed requests
    pub allowed_requests: u64,
    /// Denied requests
    pub denied_requests: u64,
    /// Average authorization time
    pub avg_authz_time: Duration,
    /// Cache hit ratio
    pub cache_hit_ratio: f32,
}

/// Encryption manager
#[derive(Debug)]
pub struct EncryptionManager {
    /// Encryption keys
    keys: Arc<RwLock<HashMap<String, EncryptionKey>>>,
    /// Default key ID
    default_key_id: Arc<RwLock<String>>,
    /// Key rotation schedule
    rotation_schedule: Arc<Mutex<HashMap<String, SystemTime>>>,
    /// Encryption metrics
    metrics: Arc<Mutex<EncryptionMetrics>>,
}

/// Encryption key
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptionKey {
    /// Key ID
    pub key_id: String,
    /// Key data (encrypted)
    pub key_data: Vec<u8>,
    /// Key algorithm
    pub algorithm: EncryptionAlgorithm,
    /// Key size in bits
    pub key_size: u32,
    /// Creation timestamp
    pub created_at: SystemTime,
    /// Expiration timestamp
    pub expires_at: Option<SystemTime>,
    /// Key status
    pub status: KeyStatus,
    /// Key usage counter
    pub usage_count: u64,
}

/// Key status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum KeyStatus {
    /// Active key
    Active,
    /// Deprecated key (can decrypt but not encrypt)
    Deprecated,
    /// Revoked key
    Revoked,
    /// Expired key
    Expired,
}

/// Encryption metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EncryptionMetrics {
    /// Total encryption operations
    pub total_encryptions: u64,
    /// Total decryption operations
    pub total_decryptions: u64,
    /// Failed operations
    pub failed_operations: u64,
    /// Average operation time
    pub avg_operation_time: Duration,
    /// Bytes encrypted
    pub bytes_encrypted: u64,
    /// Bytes decrypted
    pub bytes_decrypted: u64,
}

/// Audit trail manager
#[derive(Debug, Default)]
pub struct AuditTrailManager {
    /// Audit entries
    entries: Vec<AuditEntry>,
    /// Entry counter
    entry_counter: u64,
    /// Audit metrics
    metrics: AuditMetrics,
}

/// Audit entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEntry {
    /// Entry ID
    pub entry_id: u64,
    /// Event type
    pub event_type: AuditEventType,
    /// Timestamp
    pub timestamp: SystemTime,
    /// User principal
    pub user: Option<UserPrincipal>,
    /// Event description
    pub description: String,
    /// Resource involved
    pub resource: Option<String>,
    /// Action performed
    pub action: Option<String>,
    /// Event outcome
    pub outcome: AuditOutcome,
    /// Client IP address
    pub client_ip: Option<String>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Audit outcome
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AuditOutcome {
    /// Success
    Success,
    /// Failure
    Failure,
    /// Partial success
    Partial,
    /// Unknown outcome
    Unknown,
}

/// Audit metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AuditMetrics {
    /// Total audit entries
    pub total_entries: u64,
    /// Entries by type
    pub by_event_type: HashMap<AuditEventType, u64>,
    /// Entries by outcome
    pub by_outcome: HashMap<AuditOutcome, u64>,
    /// Storage size in bytes
    pub storage_size: u64,
}

/// Security policies
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SecurityPolicies {
    /// Password policies by context
    pub password_policies: HashMap<String, PasswordPolicy>,
    /// Session policies
    pub session_policies: HashMap<String, SessionConfig>,
    /// Access policies
    pub access_policies: Vec<AccessPolicy>,
    /// Encryption policies
    pub encryption_policies: HashMap<String, EncryptionConfig>,
}

/// Access policy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessPolicy {
    /// Policy ID
    pub policy_id: String,
    /// Policy name
    pub name: String,
    /// Policy description
    pub description: Option<String>,
    /// Policy rules
    pub rules: Vec<AccessRule>,
    /// Policy priority
    pub priority: i32,
    /// Policy status
    pub enabled: bool,
}

/// Access rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessRule {
    /// Rule ID
    pub rule_id: String,
    /// Subject (user/role pattern)
    pub subject: String,
    /// Resource pattern
    pub resource: String,
    /// Action pattern
    pub action: String,
    /// Rule effect
    pub effect: RuleEffect,
    /// Rule conditions
    pub conditions: Vec<RuleCondition>,
}

/// Rule effect
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RuleEffect {
    /// Allow access
    Allow,
    /// Deny access
    Deny,
}

/// Rule condition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuleCondition {
    /// Condition attribute
    pub attribute: String,
    /// Condition operator
    pub operator: ConditionOperator,
    /// Condition value
    pub value: String,
}

/// Security session
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecuritySession {
    /// Session ID
    pub session_id: String,
    /// User principal
    pub principal: UserPrincipal,
    /// Session creation time
    pub created_at: SystemTime,
    /// Last access time
    pub last_access: SystemTime,
    /// Session expiration time
    pub expires_at: SystemTime,
    /// Session attributes
    pub attributes: HashMap<String, String>,
    /// Session state
    pub state: SessionState,
    /// Client information
    pub client_info: ClientInfo,
}

/// Session state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SessionState {
    /// Active session
    Active,
    /// Inactive session
    Inactive,
    /// Expired session
    Expired,
    /// Terminated session
    Terminated,
}

/// Client information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClientInfo {
    /// Client IP address
    pub ip_address: Option<String>,
    /// User agent
    pub user_agent: Option<String>,
    /// Client platform
    pub platform: Option<String>,
    /// Client location
    pub location: Option<String>,
}

/// Security metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SecurityMetrics {
    /// Active sessions count
    pub active_sessions: usize,
    /// Total security events
    pub total_security_events: u64,
    /// Security violations
    pub security_violations: u64,
    /// Blocked requests
    pub blocked_requests: u64,
    /// Authentication metrics
    pub auth_metrics: AuthenticationMetrics,
    /// Authorization metrics
    pub authz_metrics: AuthorizationMetrics,
    /// Encryption metrics
    pub encryption_metrics: EncryptionMetrics,
    /// Audit metrics
    pub audit_metrics: AuditMetrics,
}

impl SecurityContext {
    /// Create a new security context
    pub fn new(context_id: String) -> ContextResult<Self> {
        let context = Self {
            context_id: context_id.clone(),
            config: Arc::new(RwLock::new(SecurityConfig::default())),
            auth_manager: Arc::new(AuthenticationManager::new()),
            authz_manager: Arc::new(AuthorizationManager::new()),
            encryption_manager: Arc::new(EncryptionManager::new()),
            audit_manager: Arc::new(Mutex::new(AuditTrailManager::default())),
            policies: Arc::new(RwLock::new(SecurityPolicies::default())),
            sessions: Arc::new(RwLock::new(HashMap::new())),
            state: Arc::new(RwLock::new(ContextState::Initializing)),
            metadata: Arc::new(RwLock::new(ContextMetadata::default())),
            metrics: Arc::new(Mutex::new(SecurityMetrics::default())),
        };

        // Update state to active
        *context.state.write().unwrap() = ContextState::Active;

        Ok(context)
    }

    /// Create security context with custom configuration
    pub fn with_config(context_id: String, config: SecurityConfig) -> ContextResult<Self> {
        let mut context = Self::new(context_id)?;
        *context.config.write().unwrap() = config;
        Ok(context)
    }

    /// Authenticate user
    pub fn authenticate(&self, credentials: &AuthenticationCredentials) -> ContextResult<AuthenticationResult> {
        self.auth_manager.authenticate(credentials)
    }

    /// Validate token
    pub fn validate_token(&self, token: &str) -> ContextResult<TokenValidationResult> {
        self.auth_manager.validate_token(token)
    }

    /// Check authorization
    pub fn authorize(&self, user: &UserPrincipal, resource: &str, action: &str) -> ContextResult<bool> {
        self.authz_manager.authorize(user, resource, action)
    }

    /// Create security session
    pub fn create_session(&self, principal: UserPrincipal, client_info: ClientInfo) -> ContextResult<String> {
        let session_id = Uuid::new_v4().to_string();
        let config = self.config.read().unwrap();
        let expires_at = SystemTime::now() + config.session_config.timeout;

        let session = SecuritySession {
            session_id: session_id.clone(),
            principal,
            created_at: SystemTime::now(),
            last_access: SystemTime::now(),
            expires_at,
            attributes: HashMap::new(),
            state: SessionState::Active,
            client_info,
        };

        let mut sessions = self.sessions.write().unwrap();
        sessions.insert(session_id.clone(), session);

        // Update metrics
        let mut metrics = self.metrics.lock().unwrap();
        metrics.active_sessions = sessions.len();

        Ok(session_id)
    }

    /// Get session
    pub fn get_session(&self, session_id: &str) -> ContextResult<Option<SecuritySession>> {
        let sessions = self.sessions.read().unwrap();
        Ok(sessions.get(session_id).cloned())
    }

    /// Terminate session
    pub fn terminate_session(&self, session_id: &str) -> ContextResult<()> {
        let mut sessions = self.sessions.write().unwrap();
        if let Some(session) = sessions.get_mut(session_id) {
            session.state = SessionState::Terminated;
        }

        // Update metrics
        let mut metrics = self.metrics.lock().unwrap();
        metrics.active_sessions = sessions.values().filter(|s| s.state == SessionState::Active).count();

        Ok(())
    }

    /// Log audit event
    pub fn log_audit_event(
        &self,
        event_type: AuditEventType,
        description: String,
        user: Option<UserPrincipal>,
        resource: Option<String>,
        action: Option<String>,
        outcome: AuditOutcome,
        client_ip: Option<String>,
        metadata: HashMap<String, String>,
    ) -> ContextResult<()> {
        let mut audit_manager = self.audit_manager.lock().unwrap();
        audit_manager.log_event(event_type, description, user, resource, action, outcome, client_ip, metadata);
        Ok(())
    }

    /// Get security metrics
    pub fn get_security_metrics(&self) -> ContextResult<SecurityMetrics> {
        let metrics = self.metrics.lock().unwrap();
        Ok(metrics.clone())
    }

    /// Update security configuration
    pub fn update_config<F>(&self, updater: F) -> ContextResult<()>
    where
        F: FnOnce(&mut SecurityConfig) -> ContextResult<()>,
    {
        let mut config = self.config.write().unwrap();
        updater(&mut *config)
    }

    /// Get security configuration
    pub fn get_config(&self) -> ContextResult<SecurityConfig> {
        let config = self.config.read().unwrap();
        Ok(config.clone())
    }
}

impl AuthenticationManager {
    /// Create a new authentication manager
    pub fn new() -> Self {
        Self {
            providers: Arc::new(RwLock::new(HashMap::new())),
            active_attempts: Arc::new(Mutex::new(HashMap::new())),
            auth_cache: Arc::new(Mutex::new(HashMap::new())),
            metrics: Arc::new(Mutex::new(AuthenticationMetrics::default())),
        }
    }

    /// Register authentication provider
    pub fn register_provider(&self, provider: Box<dyn AuthenticationProvider>) -> ContextResult<()> {
        let mut providers = self.providers.write().unwrap();
        providers.insert(provider.name().to_string(), provider);
        Ok(())
    }

    /// Authenticate user
    pub fn authenticate(&self, credentials: &AuthenticationCredentials) -> ContextResult<AuthenticationResult> {
        let providers = self.providers.read().unwrap();

        // Find appropriate provider
        let provider = providers.values()
            .find(|p| p.method() == credentials.credential_type)
            .ok_or_else(|| ContextError::custom(
                "provider_not_found",
                format!("No provider found for method {:?}", credentials.credential_type)
            ))?;

        // Perform authentication
        let result = provider.authenticate(credentials)?;

        // Update metrics
        let mut metrics = self.metrics.lock().unwrap();
        metrics.total_attempts += 1;
        if result.success {
            metrics.successful_attempts += 1;
        } else {
            metrics.failed_attempts += 1;
        }
        *metrics.by_method.entry(credentials.credential_type.clone()).or_insert(0) += 1;

        Ok(result)
    }

    /// Validate token
    pub fn validate_token(&self, token: &str) -> ContextResult<TokenValidationResult> {
        let providers = self.providers.read().unwrap();

        // Try each token-based provider
        for provider in providers.values() {
            if matches!(provider.method(), AuthenticationMethod::Token | AuthenticationMethod::ApiKey) {
                match provider.validate_token(token) {
                    Ok(result) if result.valid => return Ok(result),
                    _ => continue,
                }
            }
        }

        Ok(TokenValidationResult {
            valid: false,
            principal: None,
            expires_at: None,
            error_message: Some("Invalid or expired token".to_string()),
        })
    }
}

impl AuthorizationManager {
    /// Create a new authorization manager
    pub fn new() -> Self {
        Self {
            roles: Arc::new(RwLock::new(HashMap::new())),
            permissions: Arc::new(RwLock::new(HashMap::new())),
            user_roles: Arc::new(RwLock::new(HashMap::new())),
            resource_permissions: Arc::new(RwLock::new(HashMap::new())),
            authz_cache: Arc::new(Mutex::new(HashMap::new())),
            metrics: Arc::new(Mutex::new(AuthorizationMetrics::default())),
        }
    }

    /// Create role
    pub fn create_role(&self, role: Role) -> ContextResult<()> {
        let mut roles = self.roles.write().unwrap();
        roles.insert(role.name.clone(), role);
        Ok(())
    }

    /// Assign role to user
    pub fn assign_role(&self, user_id: &str, role_name: &str) -> ContextResult<()> {
        let mut user_roles = self.user_roles.write().unwrap();
        user_roles.entry(user_id.to_string()).or_insert_with(HashSet::new).insert(role_name.to_string());
        Ok(())
    }

    /// Check authorization
    pub fn authorize(&self, user: &UserPrincipal, resource: &str, action: &str) -> ContextResult<bool> {
        let start_time = std::time::Instant::now();

        // Check user roles and permissions
        let user_roles = self.user_roles.read().unwrap();
        let roles = self.roles.read().unwrap();

        let mut allowed = false;

        if let Some(user_role_names) = user_roles.get(&user.user_id) {
            for role_name in user_role_names {
                if let Some(role) = roles.get(role_name) {
                    for permission in &role.permissions {
                        if permission.resource_type == resource && permission.action == action {
                            allowed = true;
                            break;
                        }
                    }
                    if allowed { break; }
                }
            }
        }

        // Update metrics
        let mut metrics = self.metrics.lock().unwrap();
        metrics.total_requests += 1;
        if allowed {
            metrics.allowed_requests += 1;
        } else {
            metrics.denied_requests += 1;
        }
        metrics.avg_authz_time = Duration::from_nanos(
            ((metrics.avg_authz_time.as_nanos() * (metrics.total_requests - 1) as u128) +
             start_time.elapsed().as_nanos()) / metrics.total_requests as u128
        );

        Ok(allowed)
    }
}

impl EncryptionManager {
    /// Create a new encryption manager
    pub fn new() -> Self {
        Self {
            keys: Arc::new(RwLock::new(HashMap::new())),
            default_key_id: Arc::new(RwLock::new(String::new())),
            rotation_schedule: Arc::new(Mutex::new(HashMap::new())),
            metrics: Arc::new(Mutex::new(EncryptionMetrics::default())),
        }
    }

    /// Generate new encryption key
    pub fn generate_key(&self, algorithm: EncryptionAlgorithm) -> ContextResult<String> {
        let key_id = Uuid::new_v4().to_string();
        let key_size = match algorithm {
            EncryptionAlgorithm::AES128GCM | EncryptionAlgorithm::AES128CBC => 128,
            EncryptionAlgorithm::AES256GCM | EncryptionAlgorithm::AES256CBC => 256,
            EncryptionAlgorithm::ChaCha20Poly1305 => 256,
        };

        let key = EncryptionKey {
            key_id: key_id.clone(),
            key_data: vec![0u8; (key_size / 8) as usize], // Placeholder key data
            algorithm,
            key_size,
            created_at: SystemTime::now(),
            expires_at: None,
            status: KeyStatus::Active,
            usage_count: 0,
        };

        let mut keys = self.keys.write().unwrap();
        keys.insert(key_id.clone(), key);

        Ok(key_id)
    }

    /// Get encryption key
    pub fn get_key(&self, key_id: &str) -> ContextResult<Option<EncryptionKey>> {
        let keys = self.keys.read().unwrap();
        Ok(keys.get(key_id).cloned())
    }

    /// Encrypt data (placeholder implementation)
    pub fn encrypt(&self, data: &[u8], key_id: Option<&str>) -> ContextResult<Vec<u8>> {
        let mut metrics = self.metrics.lock().unwrap();
        metrics.total_encryptions += 1;
        metrics.bytes_encrypted += data.len() as u64;

        // Placeholder encryption - in real implementation would use actual cryptography
        Ok(data.to_vec())
    }

    /// Decrypt data (placeholder implementation)
    pub fn decrypt(&self, encrypted_data: &[u8], key_id: Option<&str>) -> ContextResult<Vec<u8>> {
        let mut metrics = self.metrics.lock().unwrap();
        metrics.total_decryptions += 1;
        metrics.bytes_decrypted += encrypted_data.len() as u64;

        // Placeholder decryption - in real implementation would use actual cryptography
        Ok(encrypted_data.to_vec())
    }
}

impl AuditTrailManager {
    pub fn log_event(
        &mut self,
        event_type: AuditEventType,
        description: String,
        user: Option<UserPrincipal>,
        resource: Option<String>,
        action: Option<String>,
        outcome: AuditOutcome,
        client_ip: Option<String>,
        metadata: HashMap<String, String>,
    ) {
        self.entry_counter += 1;

        let entry = AuditEntry {
            entry_id: self.entry_counter,
            event_type,
            timestamp: SystemTime::now(),
            user,
            description,
            resource,
            action,
            outcome,
            client_ip,
            metadata,
        };

        self.entries.push(entry);

        // Update metrics
        self.metrics.total_entries += 1;
        *self.metrics.by_event_type.entry(event_type).or_insert(0) += 1;
        *self.metrics.by_outcome.entry(outcome).or_insert(0) += 1;
    }

    /// Get audit entries
    pub fn get_entries(&self, limit: Option<usize>) -> Vec<AuditEntry> {
        match limit {
            Some(n) => self.entries.iter().rev().take(n).cloned().collect(),
            None => self.entries.clone(),
        }
    }
}

impl ExecutionContextTrait for SecurityContext {
    fn id(&self) -> &str {
        &self.context_id
    }

    fn context_type(&self) -> ContextType {
        ContextType::Security
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
        // Validate security configuration
        let config = self.config.read().unwrap();

        if config.enable_authentication && config.auth_methods.is_empty() {
            return Err(ContextError::validation("No authentication methods configured"));
        }

        if config.enable_encryption && config.encryption_config.key_size < 128 {
            return Err(ContextError::validation("Insufficient key size for encryption"));
        }

        Ok(())
    }

    fn clone_with_id(&self, new_id: String) -> Result<Box<dyn ExecutionContextTrait>, ContextError> {
        let config = self.get_config()?;
        let new_context = SecurityContext::with_config(new_id, config)?;
        Ok(Box::new(new_context))
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

impl Default for AuthenticationManager {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for AuthorizationManager {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for EncryptionManager {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_security_context_creation() {
        let context = SecurityContext::new("test-security".to_string()).unwrap();
        assert_eq!(context.id(), "test-security");
        assert_eq!(context.context_type(), ContextType::Security);
        assert!(context.is_active());
    }

    #[test]
    fn test_security_session_management() {
        let context = SecurityContext::new("test-session".to_string()).unwrap();

        let principal = UserPrincipal {
            user_id: "user123".to_string(),
            username: "testuser".to_string(),
            display_name: Some("Test User".to_string()),
            email: Some("test@example.com".to_string()),
            roles: HashSet::new(),
            attributes: HashMap::new(),
            account_status: AccountStatus::Active,
            last_login: None,
        };

        let client_info = ClientInfo {
            ip_address: Some("127.0.0.1".to_string()),
            user_agent: Some("test-agent".to_string()),
            platform: Some("test".to_string()),
            location: None,
        };

        // Create session
        let session_id = context.create_session(principal, client_info).unwrap();
        assert!(!session_id.is_empty());

        // Get session
        let session = context.get_session(&session_id).unwrap();
        assert!(session.is_some());
        assert_eq!(session.unwrap().principal.user_id, "user123");

        // Terminate session
        context.terminate_session(&session_id).unwrap();

        let session = context.get_session(&session_id).unwrap();
        assert_eq!(session.unwrap().state, SessionState::Terminated);
    }

    #[test]
    fn test_authorization_manager() {
        let manager = AuthorizationManager::new();

        // Create role
        let mut permissions = HashSet::new();
        permissions.insert(Permission {
            name: "read_data".to_string(),
            resource_type: "data".to_string(),
            action: "read".to_string(),
            resource_instance: None,
            conditions: Vec::new(),
        });

        let role = Role {
            name: "data_reader".to_string(),
            description: Some("Can read data".to_string()),
            permissions,
            parent_roles: HashSet::new(),
            metadata: HashMap::new(),
        };

        manager.create_role(role).unwrap();

        // Assign role to user
        manager.assign_role("user123", "data_reader").unwrap();

        // Test authorization
        let principal = UserPrincipal {
            user_id: "user123".to_string(),
            username: "testuser".to_string(),
            display_name: None,
            email: None,
            roles: HashSet::new(),
            attributes: HashMap::new(),
            account_status: AccountStatus::Active,
            last_login: None,
        };

        let allowed = manager.authorize(&principal, "data", "read").unwrap();
        assert!(allowed);

        let not_allowed = manager.authorize(&principal, "data", "write").unwrap();
        assert!(!not_allowed);
    }

    #[test]
    fn test_encryption_manager() {
        let manager = EncryptionManager::new();

        // Generate key
        let key_id = manager.generate_key(EncryptionAlgorithm::AES256GCM).unwrap();
        assert!(!key_id.is_empty());

        // Get key
        let key = manager.get_key(&key_id).unwrap();
        assert!(key.is_some());
        assert_eq!(key.unwrap().algorithm, EncryptionAlgorithm::AES256GCM);

        // Test encryption/decryption
        let data = b"test data";
        let encrypted = manager.encrypt(data, Some(&key_id)).unwrap();
        let decrypted = manager.decrypt(&encrypted, Some(&key_id)).unwrap();
        assert_eq!(data, decrypted.as_slice());
    }

    #[test]
    fn test_audit_trail_manager() {
        let mut manager = AuditTrailManager::default();

        // Log event
        manager.log_event(
            AuditEventType::Authentication,
            "User login".to_string(),
            None,
            None,
            Some("login".to_string()),
            AuditOutcome::Success,
            Some("127.0.0.1".to_string()),
            HashMap::new(),
        );

        let entries = manager.get_entries(Some(10));
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].event_type, AuditEventType::Authentication);
        assert_eq!(entries[0].outcome, AuditOutcome::Success);
    }

    #[test]
    fn test_password_policy() {
        let policy = PasswordPolicy::default();

        // Test minimum requirements
        assert_eq!(policy.min_length, 8);
        assert!(policy.require_uppercase);
        assert!(policy.require_lowercase);
        assert!(policy.require_digits);
        assert!(policy.require_special_chars);
    }

    #[test]
    fn test_security_levels() {
        assert!(SecurityLevel::TopSecret > SecurityLevel::High);
        assert!(SecurityLevel::High > SecurityLevel::Standard);
        assert!(SecurityLevel::Standard > SecurityLevel::Basic);
    }
}