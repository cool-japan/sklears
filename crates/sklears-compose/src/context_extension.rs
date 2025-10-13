//! Extension context for custom extensions and plugins
//!
//! This module provides a comprehensive extension framework allowing
//! dynamic loading, execution, and management of custom extensions
//! and plugins within execution contexts.

use std::{
    collections::{HashMap, HashSet, VecDeque, BTreeMap},
    sync::{Arc, RwLock, Mutex},
    time::{Duration, SystemTime, Instant},
    fmt::{Debug, Display},
    path::PathBuf,
    any::Any,
    thread,
};
use serde::{Serialize, Deserialize};
use uuid::Uuid;

use crate::context_core::{
    ExecutionContextTrait, ContextType, ContextState, ContextError, ContextResult,
    ContextMetadata, ContextEvent,
};

/// Extension context for managing custom extensions and plugins
#[derive(Debug)]
pub struct ExtensionContext {
    /// Context identifier
    pub id: String,
    /// Extension state
    pub state: Arc<RwLock<ExtensionState>>,
    /// Extension manager
    pub extension_manager: Arc<RwLock<ExtensionManager>>,
    /// Plugin loader
    pub plugin_loader: Arc<Mutex<PluginLoader>>,
    /// Extension registry
    pub registry: Arc<RwLock<ExtensionRegistry>>,
    /// Sandbox manager
    pub sandbox_manager: Arc<RwLock<SandboxManager>>,
    /// Event dispatcher
    pub event_dispatcher: Arc<Mutex<ExtensionEventDispatcher>>,
    /// Resource manager
    pub resource_manager: Arc<Mutex<ExtensionResourceManager>>,
    /// Configuration manager
    pub config_manager: Arc<RwLock<ExtensionConfigManager>>,
    /// Metrics collector
    pub metrics: Arc<Mutex<ExtensionMetrics>>,
    /// Configuration
    pub config: Arc<RwLock<ExtensionConfig>>,
    /// Created timestamp
    pub created_at: SystemTime,
}

/// Extension context states
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExtensionState {
    /// Extension context is initializing
    Initializing,
    /// Extension context is active
    Active,
    /// Extension context is loading extensions
    Loading,
    /// Extension context is in safe mode
    SafeMode,
    /// Extension context is disabled
    Disabled,
    /// Extension context is in maintenance mode
    Maintenance,
    /// Extension context is shutting down
    ShuttingDown,
}

impl Display for ExtensionState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExtensionState::Initializing => write!(f, "initializing"),
            ExtensionState::Active => write!(f, "active"),
            ExtensionState::Loading => write!(f, "loading"),
            ExtensionState::SafeMode => write!(f, "safe_mode"),
            ExtensionState::Disabled => write!(f, "disabled"),
            ExtensionState::Maintenance => write!(f, "maintenance"),
            ExtensionState::ShuttingDown => write!(f, "shutting_down"),
        }
    }
}

/// Extension configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtensionConfig {
    /// Enable extensions
    pub enabled: bool,
    /// Extensions directory
    pub extensions_dir: PathBuf,
    /// Enable sandboxing
    pub enable_sandboxing: bool,
    /// Enable hot reloading
    pub enable_hot_reload: bool,
    /// Extension timeout
    pub extension_timeout: Duration,
    /// Maximum concurrent extensions
    pub max_concurrent_extensions: usize,
    /// Security settings
    pub security_settings: ExtensionSecuritySettings,
    /// Resource limits
    pub resource_limits: ExtensionResourceLimits,
    /// Plugin settings
    pub plugin_settings: PluginSettings,
    /// Custom configuration
    pub custom: HashMap<String, serde_json::Value>,
}

impl Default for ExtensionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            extensions_dir: PathBuf::from("./extensions"),
            enable_sandboxing: true,
            enable_hot_reload: false,
            extension_timeout: Duration::from_secs(30),
            max_concurrent_extensions: 100,
            security_settings: ExtensionSecuritySettings::default(),
            resource_limits: ExtensionResourceLimits::default(),
            plugin_settings: PluginSettings::default(),
            custom: HashMap::new(),
        }
    }
}

/// Extension security settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtensionSecuritySettings {
    /// Require signed extensions
    pub require_signed: bool,
    /// Trusted publishers
    pub trusted_publishers: HashSet<String>,
    /// Allowed capabilities
    pub allowed_capabilities: HashSet<ExtensionCapability>,
    /// Restricted APIs
    pub restricted_apis: HashSet<String>,
    /// Enable permission model
    pub enable_permissions: bool,
    /// Default permission level
    pub default_permission_level: PermissionLevel,
}

impl Default for ExtensionSecuritySettings {
    fn default() -> Self {
        let mut allowed_capabilities = HashSet::new();
        allowed_capabilities.insert(ExtensionCapability::FileRead);
        allowed_capabilities.insert(ExtensionCapability::NetworkClient);

        Self {
            require_signed: false,
            trusted_publishers: HashSet::new(),
            allowed_capabilities,
            restricted_apis: HashSet::new(),
            enable_permissions: true,
            default_permission_level: PermissionLevel::Restricted,
        }
    }
}

/// Extension capabilities
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ExtensionCapability {
    /// File system read access
    FileRead,
    /// File system write access
    FileWrite,
    /// Network client access
    NetworkClient,
    /// Network server access
    NetworkServer,
    /// Database access
    DatabaseAccess,
    /// System process spawning
    ProcessSpawn,
    /// Environment variable access
    EnvironmentAccess,
    /// Registry access
    RegistryAccess,
    /// IPC communication
    IpcCommunication,
    /// Custom capability
    Custom(String),
}

impl Display for ExtensionCapability {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExtensionCapability::FileRead => write!(f, "file_read"),
            ExtensionCapability::FileWrite => write!(f, "file_write"),
            ExtensionCapability::NetworkClient => write!(f, "network_client"),
            ExtensionCapability::NetworkServer => write!(f, "network_server"),
            ExtensionCapability::DatabaseAccess => write!(f, "database_access"),
            ExtensionCapability::ProcessSpawn => write!(f, "process_spawn"),
            ExtensionCapability::EnvironmentAccess => write!(f, "environment_access"),
            ExtensionCapability::RegistryAccess => write!(f, "registry_access"),
            ExtensionCapability::IpcCommunication => write!(f, "ipc_communication"),
            ExtensionCapability::Custom(name) => write!(f, "custom_{}", name),
        }
    }
}

/// Permission levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum PermissionLevel {
    /// No permissions
    None = 0,
    /// Restricted permissions
    Restricted = 1,
    /// Normal permissions
    Normal = 2,
    /// Elevated permissions
    Elevated = 3,
    /// Administrative permissions
    Administrative = 4,
    /// Full permissions
    Full = 5,
}

impl Display for PermissionLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PermissionLevel::None => write!(f, "none"),
            PermissionLevel::Restricted => write!(f, "restricted"),
            PermissionLevel::Normal => write!(f, "normal"),
            PermissionLevel::Elevated => write!(f, "elevated"),
            PermissionLevel::Administrative => write!(f, "administrative"),
            PermissionLevel::Full => write!(f, "full"),
        }
    }
}

/// Extension resource limits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtensionResourceLimits {
    /// Maximum memory per extension (bytes)
    pub max_memory: usize,
    /// Maximum CPU time per extension
    pub max_cpu_time: Duration,
    /// Maximum file handles
    pub max_file_handles: usize,
    /// Maximum network connections
    pub max_network_connections: usize,
    /// Maximum threads
    pub max_threads: usize,
    /// Maximum disk usage
    pub max_disk_usage: usize,
}

impl Default for ExtensionResourceLimits {
    fn default() -> Self {
        Self {
            max_memory: 256 * 1024 * 1024, // 256MB
            max_cpu_time: Duration::from_secs(60),
            max_file_handles: 100,
            max_network_connections: 10,
            max_threads: 4,
            max_disk_usage: 100 * 1024 * 1024, // 100MB
        }
    }
}

/// Plugin settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginSettings {
    /// Plugin search paths
    pub search_paths: Vec<PathBuf>,
    /// Supported plugin formats
    pub supported_formats: HashSet<PluginFormat>,
    /// Plugin cache settings
    pub cache_settings: PluginCacheSettings,
    /// Auto-discovery enabled
    pub auto_discovery: bool,
    /// Plugin verification settings
    pub verification_settings: PluginVerificationSettings,
}

impl Default for PluginSettings {
    fn default() -> Self {
        let mut supported_formats = HashSet::new();
        supported_formats.insert(PluginFormat::DynamicLibrary);
        supported_formats.insert(PluginFormat::WebAssembly);

        Self {
            search_paths: vec![PathBuf::from("./plugins")],
            supported_formats,
            cache_settings: PluginCacheSettings::default(),
            auto_discovery: true,
            verification_settings: PluginVerificationSettings::default(),
        }
    }
}

/// Plugin formats
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PluginFormat {
    /// Dynamic library (.dll, .so, .dylib)
    DynamicLibrary,
    /// WebAssembly module
    WebAssembly,
    /// JavaScript/Node.js
    JavaScript,
    /// Python script
    Python,
    /// Lua script
    Lua,
    /// Custom format
    Custom(String),
}

impl Display for PluginFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PluginFormat::DynamicLibrary => write!(f, "dynamic_library"),
            PluginFormat::WebAssembly => write!(f, "webassembly"),
            PluginFormat::JavaScript => write!(f, "javascript"),
            PluginFormat::Python => write!(f, "python"),
            PluginFormat::Lua => write!(f, "lua"),
            PluginFormat::Custom(name) => write!(f, "custom_{}", name),
        }
    }
}

/// Plugin cache settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginCacheSettings {
    /// Enable plugin caching
    pub enabled: bool,
    /// Cache directory
    pub cache_dir: PathBuf,
    /// Cache TTL
    pub ttl: Duration,
    /// Maximum cache size
    pub max_size: usize,
}

impl Default for PluginCacheSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            cache_dir: PathBuf::from("./cache/plugins"),
            ttl: Duration::from_secs(24 * 60 * 60), // 24 hours
            max_size: 1024 * 1024 * 1024, // 1GB
        }
    }
}

/// Plugin verification settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginVerificationSettings {
    /// Enable signature verification
    pub verify_signature: bool,
    /// Enable checksum verification
    pub verify_checksum: bool,
    /// Trusted certificate authorities
    pub trusted_cas: Vec<String>,
    /// Allow self-signed certificates
    pub allow_self_signed: bool,
}

impl Default for PluginVerificationSettings {
    fn default() -> Self {
        Self {
            verify_signature: false,
            verify_checksum: true,
            trusted_cas: Vec::new(),
            allow_self_signed: false,
        }
    }
}

/// Extension identifier
pub type ExtensionId = Uuid;

/// Extension definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Extension {
    /// Extension ID
    pub id: ExtensionId,
    /// Extension name
    pub name: String,
    /// Extension version
    pub version: String,
    /// Extension description
    pub description: String,
    /// Extension author
    pub author: String,
    /// Extension type
    pub extension_type: ExtensionType,
    /// Extension status
    pub status: ExtensionStatus,
    /// Extension manifest
    pub manifest: ExtensionManifest,
    /// Extension binary/code location
    pub location: ExtensionLocation,
    /// Load timestamp
    pub loaded_at: Option<SystemTime>,
    /// Execution statistics
    pub statistics: ExtensionStatistics,
    /// Extension configuration
    pub configuration: ExtensionConfiguration,
}

/// Extension types
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ExtensionType {
    /// Plugin extension
    Plugin,
    /// Middleware extension
    Middleware,
    /// Filter extension
    Filter,
    /// Processor extension
    Processor,
    /// Renderer extension
    Renderer,
    /// Validator extension
    Validator,
    /// Transformer extension
    Transformer,
    /// Handler extension
    Handler,
    /// Custom extension type
    Custom(String),
}

impl Display for ExtensionType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExtensionType::Plugin => write!(f, "plugin"),
            ExtensionType::Middleware => write!(f, "middleware"),
            ExtensionType::Filter => write!(f, "filter"),
            ExtensionType::Processor => write!(f, "processor"),
            ExtensionType::Renderer => write!(f, "renderer"),
            ExtensionType::Validator => write!(f, "validator"),
            ExtensionType::Transformer => write!(f, "transformer"),
            ExtensionType::Handler => write!(f, "handler"),
            ExtensionType::Custom(name) => write!(f, "custom_{}", name),
        }
    }
}

/// Extension status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExtensionStatus {
    /// Extension is discovered but not loaded
    Discovered,
    /// Extension is loading
    Loading,
    /// Extension is loaded and active
    Active,
    /// Extension is paused
    Paused,
    /// Extension is unloading
    Unloading,
    /// Extension is unloaded
    Unloaded,
    /// Extension encountered an error
    Error,
    /// Extension is disabled
    Disabled,
}

impl Display for ExtensionStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExtensionStatus::Discovered => write!(f, "discovered"),
            ExtensionStatus::Loading => write!(f, "loading"),
            ExtensionStatus::Active => write!(f, "active"),
            ExtensionStatus::Paused => write!(f, "paused"),
            ExtensionStatus::Unloading => write!(f, "unloading"),
            ExtensionStatus::Unloaded => write!(f, "unloaded"),
            ExtensionStatus::Error => write!(f, "error"),
            ExtensionStatus::Disabled => write!(f, "disabled"),
        }
    }
}

/// Extension manifest
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtensionManifest {
    /// Schema version
    pub schema_version: String,
    /// Extension metadata
    pub metadata: ExtensionMetadata,
    /// Required capabilities
    pub capabilities: HashSet<ExtensionCapability>,
    /// Dependencies
    pub dependencies: Vec<ExtensionDependency>,
    /// API definitions
    pub api_definitions: Vec<ApiDefinition>,
    /// Configuration schema
    pub config_schema: Option<serde_json::Value>,
    /// Supported platforms
    pub platforms: Vec<String>,
    /// Minimum runtime version
    pub min_runtime_version: String,
}

/// Extension metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtensionMetadata {
    /// Display name
    pub display_name: String,
    /// Homepage URL
    pub homepage: Option<String>,
    /// Repository URL
    pub repository: Option<String>,
    /// License
    pub license: String,
    /// Keywords/tags
    pub keywords: Vec<String>,
    /// Category
    pub category: String,
    /// Icon URL
    pub icon: Option<String>,
    /// Screenshots
    pub screenshots: Vec<String>,
    /// Release notes
    pub release_notes: Option<String>,
}

/// Extension dependency
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtensionDependency {
    /// Dependency name
    pub name: String,
    /// Version requirement
    pub version: String,
    /// Optional dependency
    pub optional: bool,
    /// Dependency type
    pub dependency_type: DependencyType,
}

/// Dependency types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DependencyType {
    /// Extension dependency
    Extension,
    /// System library
    SystemLibrary,
    /// Runtime requirement
    Runtime,
    /// Custom dependency type
    Custom(String),
}

/// API definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiDefinition {
    /// API name
    pub name: String,
    /// API version
    pub version: String,
    /// API type
    pub api_type: ApiType,
    /// Interface definition
    pub interface: serde_json::Value,
    /// Documentation URL
    pub documentation: Option<String>,
}

/// API types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ApiType {
    /// REST API
    Rest,
    /// GraphQL API
    GraphQL,
    /// gRPC API
    Grpc,
    /// WebSocket API
    WebSocket,
    /// Function API
    Function,
    /// Custom API type
    Custom(String),
}

/// Extension location
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExtensionLocation {
    /// Local file path
    LocalPath(PathBuf),
    /// Remote URL
    RemoteUrl(String),
    /// Registry location
    Registry {
        name: String,
        version: String,
        registry_url: String,
    },
    /// Embedded extension
    Embedded {
        data: Vec<u8>,
    },
    /// Custom location
    Custom {
        location_type: String,
        location_data: serde_json::Value,
    },
}

/// Extension statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ExtensionStatistics {
    /// Load count
    pub load_count: usize,
    /// Execution count
    pub execution_count: usize,
    /// Total execution time
    pub total_execution_time: Duration,
    /// Average execution time
    pub avg_execution_time: Duration,
    /// Error count
    pub error_count: usize,
    /// Last execution time
    pub last_execution: Option<SystemTime>,
    /// Resource usage
    pub resource_usage: ExtensionResourceUsage,
}

/// Extension resource usage
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ExtensionResourceUsage {
    /// Current memory usage
    pub memory_usage: usize,
    /// Peak memory usage
    pub peak_memory_usage: usize,
    /// CPU time used
    pub cpu_time: Duration,
    /// File handles used
    pub file_handles: usize,
    /// Network connections
    pub network_connections: usize,
    /// Disk usage
    pub disk_usage: usize,
}

/// Extension configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtensionConfiguration {
    /// Configuration values
    pub values: HashMap<String, serde_json::Value>,
    /// Configuration source
    pub source: ConfigurationSource,
    /// Last updated
    pub last_updated: SystemTime,
    /// Configuration version
    pub version: String,
}

/// Configuration sources
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConfigurationSource {
    /// Default configuration
    Default,
    /// File configuration
    File(PathBuf),
    /// Environment variables
    Environment,
    /// Registry/database
    Registry,
    /// User-provided
    User,
    /// Custom source
    Custom(String),
}

/// Extension manager
#[derive(Debug)]
pub struct ExtensionManager {
    /// Active extensions
    pub extensions: HashMap<ExtensionId, Extension>,
    /// Extension instances
    pub instances: HashMap<ExtensionId, Box<dyn ExtensionInstance>>,
    /// Load order
    pub load_order: Vec<ExtensionId>,
    /// Dependency resolver
    pub dependency_resolver: DependencyResolver,
    /// Lifecycle manager
    pub lifecycle_manager: ExtensionLifecycleManager,
}

/// Extension instance trait
pub trait ExtensionInstance: Send + Sync {
    /// Initialize the extension
    fn initialize(&mut self, context: &ExtensionContext) -> ContextResult<()>;

    /// Start the extension
    fn start(&mut self) -> ContextResult<()>;

    /// Stop the extension
    fn stop(&mut self) -> ContextResult<()>;

    /// Execute extension functionality
    fn execute(&mut self, input: &ExtensionInput) -> ContextResult<ExtensionOutput>;

    /// Get extension information
    fn info(&self) -> &ExtensionInfo;

    /// Handle configuration update
    fn configure(&mut self, config: &ExtensionConfiguration) -> ContextResult<()>;

    /// Get extension as Any for downcasting
    fn as_any(&self) -> &dyn Any;

    /// Get mutable extension as Any for downcasting
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

/// Extension information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtensionInfo {
    /// Extension ID
    pub id: ExtensionId,
    /// Extension name
    pub name: String,
    /// Extension version
    pub version: String,
    /// Extension description
    pub description: String,
    /// Supported operations
    pub supported_operations: Vec<String>,
    /// Status
    pub status: ExtensionStatus,
}

/// Extension input
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtensionInput {
    /// Operation name
    pub operation: String,
    /// Input parameters
    pub parameters: HashMap<String, serde_json::Value>,
    /// Input data
    pub data: Option<serde_json::Value>,
    /// Context information
    pub context: HashMap<String, String>,
    /// Request metadata
    pub metadata: HashMap<String, String>,
}

/// Extension output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtensionOutput {
    /// Success flag
    pub success: bool,
    /// Output data
    pub data: Option<serde_json::Value>,
    /// Output metadata
    pub metadata: HashMap<String, serde_json::Value>,
    /// Error message if failed
    pub error: Option<String>,
    /// Execution time
    pub execution_time: Duration,
}

/// Dependency resolver
#[derive(Debug, Clone)]
pub struct DependencyResolver {
    /// Dependency graph
    pub dependency_graph: HashMap<ExtensionId, HashSet<ExtensionId>>,
    /// Resolution cache
    pub resolution_cache: HashMap<ExtensionId, Vec<ExtensionId>>,
}

impl DependencyResolver {
    /// Create a new dependency resolver
    pub fn new() -> Self {
        Self {
            dependency_graph: HashMap::new(),
            resolution_cache: HashMap::new(),
        }
    }

    /// Resolve dependencies for an extension
    pub fn resolve(&self, extension_id: ExtensionId) -> ContextResult<Vec<ExtensionId>> {
        // Simplified dependency resolution - in real implementation would do topological sort
        if let Some(cached) = self.resolution_cache.get(&extension_id) {
            return Ok(cached.clone());
        }

        let mut resolved = Vec::new();
        if let Some(deps) = self.dependency_graph.get(&extension_id) {
            resolved.extend(deps.iter().cloned());
        }
        Ok(resolved)
    }
}

/// Extension lifecycle manager
#[derive(Debug)]
pub struct ExtensionLifecycleManager {
    /// Lifecycle hooks
    pub hooks: HashMap<LifecycleEvent, Vec<Box<dyn LifecycleHook>>>,
    /// State transitions
    pub state_machine: ExtensionStateMachine,
}

/// Lifecycle events
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LifecycleEvent {
    /// Before extension load
    BeforeLoad,
    /// After extension load
    AfterLoad,
    /// Before extension start
    BeforeStart,
    /// After extension start
    AfterStart,
    /// Before extension stop
    BeforeStop,
    /// After extension stop
    AfterStop,
    /// Before extension unload
    BeforeUnload,
    /// After extension unload
    AfterUnload,
    /// On extension error
    OnError,
    /// Custom lifecycle event
    Custom(String),
}

/// Lifecycle hook trait
pub trait LifecycleHook: Send + Sync {
    /// Execute the hook
    fn execute(&self, extension_id: ExtensionId, event: &LifecycleEvent) -> ContextResult<()>;

    /// Get hook name
    fn name(&self) -> &str;
}

/// Extension state machine
#[derive(Debug, Clone)]
pub struct ExtensionStateMachine {
    /// Valid state transitions
    pub transitions: HashMap<ExtensionStatus, Vec<ExtensionStatus>>,
}

impl ExtensionStateMachine {
    /// Create a new state machine
    pub fn new() -> Self {
        let mut transitions = HashMap::new();

        // Define valid transitions
        transitions.insert(ExtensionStatus::Discovered, vec![
            ExtensionStatus::Loading, ExtensionStatus::Disabled
        ]);
        transitions.insert(ExtensionStatus::Loading, vec![
            ExtensionStatus::Active, ExtensionStatus::Error
        ]);
        transitions.insert(ExtensionStatus::Active, vec![
            ExtensionStatus::Paused, ExtensionStatus::Unloading, ExtensionStatus::Error
        ]);
        transitions.insert(ExtensionStatus::Paused, vec![
            ExtensionStatus::Active, ExtensionStatus::Unloading
        ]);
        transitions.insert(ExtensionStatus::Unloading, vec![
            ExtensionStatus::Unloaded, ExtensionStatus::Error
        ]);
        transitions.insert(ExtensionStatus::Error, vec![
            ExtensionStatus::Loading, ExtensionStatus::Disabled
        ]);

        Self { transitions }
    }

    /// Check if transition is valid
    pub fn is_valid_transition(&self, from: ExtensionStatus, to: ExtensionStatus) -> bool {
        self.transitions
            .get(&from)
            .map_or(false, |valid_states| valid_states.contains(&to))
    }
}

impl ExtensionLifecycleManager {
    /// Create a new lifecycle manager
    pub fn new() -> Self {
        Self {
            hooks: HashMap::new(),
            state_machine: ExtensionStateMachine::new(),
        }
    }

    /// Add lifecycle hook
    pub fn add_hook(&mut self, event: LifecycleEvent, hook: Box<dyn LifecycleHook>) {
        self.hooks.entry(event).or_insert_with(Vec::new).push(hook);
    }

    /// Execute lifecycle hooks
    pub fn execute_hooks(&self, extension_id: ExtensionId, event: &LifecycleEvent) -> ContextResult<()> {
        if let Some(hooks) = self.hooks.get(event) {
            for hook in hooks {
                hook.execute(extension_id, event)?;
            }
        }
        Ok(())
    }
}

/// Plugin loader
#[derive(Debug)]
pub struct PluginLoader {
    /// Supported loaders
    pub loaders: HashMap<PluginFormat, Box<dyn PluginFormatLoader>>,
    /// Load cache
    pub cache: HashMap<String, CachedPlugin>,
    /// Configuration
    pub config: PluginLoaderConfig,
}

/// Plugin format loader trait
pub trait PluginFormatLoader: Send + Sync {
    /// Load plugin from path
    fn load(&self, path: &PathBuf, manifest: &ExtensionManifest) -> ContextResult<Box<dyn ExtensionInstance>>;

    /// Unload plugin
    fn unload(&self, plugin: Box<dyn ExtensionInstance>) -> ContextResult<()>;

    /// Validate plugin
    fn validate(&self, path: &PathBuf) -> ContextResult<()>;

    /// Get format name
    fn format_name(&self) -> &str;

    /// Get supported extensions
    fn supported_extensions(&self) -> Vec<String>;
}

/// Cached plugin
#[derive(Debug, Clone)]
pub struct CachedPlugin {
    /// Plugin path
    pub path: PathBuf,
    /// Cache timestamp
    pub cached_at: SystemTime,
    /// Plugin checksum
    pub checksum: String,
    /// Plugin metadata
    pub metadata: ExtensionMetadata,
}

/// Plugin loader configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginLoaderConfig {
    /// Enable parallel loading
    pub parallel_loading: bool,
    /// Load timeout
    pub load_timeout: Duration,
    /// Enable preloading
    pub enable_preloading: bool,
    /// Cache validation interval
    pub cache_validation_interval: Duration,
}

impl Default for PluginLoaderConfig {
    fn default() -> Self {
        Self {
            parallel_loading: true,
            load_timeout: Duration::from_secs(30),
            enable_preloading: false,
            cache_validation_interval: Duration::from_secs(60 * 60), // 1 hour
        }
    }
}

impl PluginLoader {
    /// Create a new plugin loader
    pub fn new(config: PluginLoaderConfig) -> Self {
        Self {
            loaders: HashMap::new(),
            cache: HashMap::new(),
            config,
        }
    }

    /// Register plugin format loader
    pub fn register_loader(&mut self, format: PluginFormat, loader: Box<dyn PluginFormatLoader>) {
        self.loaders.insert(format, loader);
    }

    /// Load plugin
    pub fn load_plugin(&mut self, path: &PathBuf, manifest: &ExtensionManifest) -> ContextResult<Box<dyn ExtensionInstance>> {
        // Determine plugin format based on file extension
        let format = self.detect_format(path)?;

        if let Some(loader) = self.loaders.get(&format) {
            loader.load(path, manifest)
        } else {
            Err(ContextError::custom("unsupported_format",
                format!("Unsupported plugin format: {}", format)))
        }
    }

    fn detect_format(&self, path: &PathBuf) -> ContextResult<PluginFormat> {
        let extension = path.extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or("");

        match extension {
            "dll" | "so" | "dylib" => Ok(PluginFormat::DynamicLibrary),
            "wasm" => Ok(PluginFormat::WebAssembly),
            "js" => Ok(PluginFormat::JavaScript),
            "py" => Ok(PluginFormat::Python),
            "lua" => Ok(PluginFormat::Lua),
            _ => Err(ContextError::custom("unknown_format",
                format!("Cannot detect plugin format for: {}", path.display())))
        }
    }
}

/// Extension registry
#[derive(Debug)]
pub struct ExtensionRegistry {
    /// Registry entries
    pub entries: HashMap<String, RegistryEntry>,
    /// Extension sources
    pub sources: Vec<Box<dyn ExtensionSource>>,
    /// Discovery cache
    pub discovery_cache: HashMap<String, DiscoveredExtension>,
    /// Registry configuration
    pub config: RegistryConfig,
}

/// Registry entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegistryEntry {
    /// Entry name
    pub name: String,
    /// Available versions
    pub versions: BTreeMap<String, ExtensionVersion>,
    /// Entry metadata
    pub metadata: RegistryMetadata,
    /// Last updated
    pub last_updated: SystemTime,
}

/// Extension version in registry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtensionVersion {
    /// Version string
    pub version: String,
    /// Extension manifest
    pub manifest: ExtensionManifest,
    /// Download URLs
    pub download_urls: HashMap<String, String>,
    /// Checksum
    pub checksum: String,
    /// Publication date
    pub published_at: SystemTime,
    /// Download count
    pub download_count: usize,
}

/// Registry metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegistryMetadata {
    /// Total downloads
    pub total_downloads: usize,
    /// Rating
    pub rating: f32,
    /// Review count
    pub review_count: usize,
    /// Popularity score
    pub popularity_score: f32,
    /// Maintenance score
    pub maintenance_score: f32,
    /// Security score
    pub security_score: f32,
}

/// Extension source trait
pub trait ExtensionSource: Send + Sync {
    /// Discover available extensions
    fn discover(&self) -> ContextResult<Vec<DiscoveredExtension>>;

    /// Download extension
    fn download(&self, name: &str, version: &str) -> ContextResult<Vec<u8>>;

    /// Get source information
    fn source_info(&self) -> ExtensionSourceInfo;

    /// Search extensions
    fn search(&self, query: &ExtensionQuery) -> ContextResult<Vec<DiscoveredExtension>>;
}

/// Discovered extension
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveredExtension {
    /// Extension name
    pub name: String,
    /// Available versions
    pub versions: Vec<String>,
    /// Extension metadata
    pub metadata: ExtensionMetadata,
    /// Source information
    pub source: String,
    /// Discovery timestamp
    pub discovered_at: SystemTime,
}

/// Extension source information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtensionSourceInfo {
    /// Source name
    pub name: String,
    /// Source URL
    pub url: String,
    /// Source type
    pub source_type: SourceType,
    /// Authentication required
    pub requires_auth: bool,
}

/// Source types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SourceType {
    /// Official registry
    Official,
    /// Community registry
    Community,
    /// Private registry
    Private,
    /// File system
    FileSystem,
    /// Git repository
    Git,
    /// Custom source
    Custom(String),
}

/// Extension query
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtensionQuery {
    /// Search terms
    pub terms: Vec<String>,
    /// Category filter
    pub category: Option<String>,
    /// Author filter
    pub author: Option<String>,
    /// Version filter
    pub version: Option<String>,
    /// Minimum rating
    pub min_rating: Option<f32>,
    /// Tags filter
    pub tags: Vec<String>,
    /// Result limit
    pub limit: Option<usize>,
}

/// Registry configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegistryConfig {
    /// Enable registry
    pub enabled: bool,
    /// Default registry URL
    pub default_registry: String,
    /// Cache TTL
    pub cache_ttl: Duration,
    /// Auto-update interval
    pub auto_update_interval: Duration,
    /// Enable pre-release versions
    pub enable_prerelease: bool,
}

impl Default for RegistryConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            default_registry: "https://registry.extensions.com".to_string(),
            cache_ttl: Duration::from_secs(60 * 60), // 1 hour
            auto_update_interval: Duration::from_secs(24 * 60 * 60), // 24 hours
            enable_prerelease: false,
        }
    }
}

impl ExtensionRegistry {
    /// Create a new extension registry
    pub fn new(config: RegistryConfig) -> Self {
        Self {
            entries: HashMap::new(),
            sources: Vec::new(),
            discovery_cache: HashMap::new(),
            config,
        }
    }

    /// Add extension source
    pub fn add_source(&mut self, source: Box<dyn ExtensionSource>) {
        self.sources.push(source);
    }

    /// Discover extensions from all sources
    pub fn discover_extensions(&mut self) -> ContextResult<Vec<DiscoveredExtension>> {
        let mut discovered = Vec::new();

        for source in &self.sources {
            match source.discover() {
                Ok(mut extensions) => discovered.append(&mut extensions),
                Err(e) => {
                    // Log error but continue with other sources
                    eprintln!("Failed to discover from source: {}", e);
                }
            }
        }

        // Update cache
        for ext in &discovered {
            self.discovery_cache.insert(ext.name.clone(), ext.clone());
        }

        Ok(discovered)
    }
}

/// Sandbox manager
#[derive(Debug)]
pub struct SandboxManager {
    /// Active sandboxes
    pub sandboxes: HashMap<ExtensionId, ExtensionSandbox>,
    /// Sandbox policies
    pub policies: HashMap<String, SandboxPolicy>,
    /// Resource monitors
    pub resource_monitors: HashMap<ExtensionId, ResourceMonitor>,
    /// Configuration
    pub config: SandboxConfig,
}

/// Extension sandbox
#[derive(Debug)]
pub struct ExtensionSandbox {
    /// Sandbox ID
    pub id: String,
    /// Extension ID
    pub extension_id: ExtensionId,
    /// Sandbox type
    pub sandbox_type: SandboxType,
    /// Security policy
    pub policy: String,
    /// Resource limits
    pub resource_limits: ExtensionResourceLimits,
    /// Sandbox status
    pub status: SandboxStatus,
    /// Created timestamp
    pub created_at: SystemTime,
}

/// Sandbox types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SandboxType {
    /// Process isolation
    Process,
    /// Container isolation
    Container,
    /// Virtual machine isolation
    VirtualMachine,
    /// WebAssembly sandbox
    WebAssembly,
    /// JavaScript V8 isolate
    V8Isolate,
    /// Custom sandbox
    Custom(String),
}

/// Sandbox status
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SandboxStatus {
    /// Sandbox is initializing
    Initializing,
    /// Sandbox is active
    Active,
    /// Sandbox is paused
    Paused,
    /// Sandbox is terminating
    Terminating,
    /// Sandbox is terminated
    Terminated,
    /// Sandbox encountered error
    Error,
}

/// Sandbox policy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SandboxPolicy {
    /// Policy name
    pub name: String,
    /// Allowed system calls
    pub allowed_syscalls: HashSet<String>,
    /// Blocked system calls
    pub blocked_syscalls: HashSet<String>,
    /// File system access rules
    pub filesystem_rules: Vec<FileSystemRule>,
    /// Network access rules
    pub network_rules: Vec<NetworkRule>,
    /// Resource limits
    pub resource_limits: ExtensionResourceLimits,
    /// Environment restrictions
    pub environment_restrictions: EnvironmentRestrictions,
}

/// File system access rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileSystemRule {
    /// Path pattern
    pub path_pattern: String,
    /// Access type
    pub access_type: FileSystemAccess,
    /// Allow or deny
    pub allow: bool,
}

/// File system access types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum FileSystemAccess {
    /// Read access
    Read,
    /// Write access
    Write,
    /// Execute access
    Execute,
    /// Create access
    Create,
    /// Delete access
    Delete,
    /// All access
    All,
}

/// Network access rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkRule {
    /// Address pattern
    pub address_pattern: String,
    /// Port range
    pub port_range: Option<(u16, u16)>,
    /// Protocol
    pub protocol: NetworkProtocol,
    /// Allow or deny
    pub allow: bool,
}

/// Network protocols
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum NetworkProtocol {
    /// TCP protocol
    Tcp,
    /// UDP protocol
    Udp,
    /// HTTP protocol
    Http,
    /// HTTPS protocol
    Https,
    /// WebSocket protocol
    WebSocket,
    /// Custom protocol
    Custom(String),
}

/// Environment restrictions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentRestrictions {
    /// Allowed environment variables
    pub allowed_env_vars: HashSet<String>,
    /// Blocked environment variables
    pub blocked_env_vars: HashSet<String>,
    /// Allow process spawning
    pub allow_process_spawn: bool,
    /// Allow IPC
    pub allow_ipc: bool,
}

/// Resource monitor
#[derive(Debug)]
pub struct ResourceMonitor {
    /// Extension ID
    pub extension_id: ExtensionId,
    /// Current usage
    pub current_usage: ExtensionResourceUsage,
    /// Usage history
    pub usage_history: VecDeque<(SystemTime, ExtensionResourceUsage)>,
    /// Limits
    pub limits: ExtensionResourceLimits,
    /// Violations
    pub violations: Vec<ResourceViolation>,
}

/// Resource violation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceViolation {
    /// Violation timestamp
    pub timestamp: SystemTime,
    /// Resource type
    pub resource_type: String,
    /// Current value
    pub current_value: usize,
    /// Limit value
    pub limit_value: usize,
    /// Violation severity
    pub severity: ViolationSeverity,
}

/// Violation severity
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ViolationSeverity {
    /// Warning level
    Warning = 1,
    /// Minor violation
    Minor = 2,
    /// Major violation
    Major = 3,
    /// Critical violation
    Critical = 4,
}

/// Sandbox configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SandboxConfig {
    /// Default sandbox type
    pub default_sandbox_type: SandboxType,
    /// Enable strict mode
    pub strict_mode: bool,
    /// Resource monitoring interval
    pub monitoring_interval: Duration,
    /// Violation threshold
    pub violation_threshold: usize,
    /// Auto-terminate on violations
    pub auto_terminate: bool,
}

impl Default for SandboxConfig {
    fn default() -> Self {
        Self {
            default_sandbox_type: SandboxType::Process,
            strict_mode: true,
            monitoring_interval: Duration::from_secs(5),
            violation_threshold: 3,
            auto_terminate: true,
        }
    }
}

impl SandboxManager {
    /// Create a new sandbox manager
    pub fn new(config: SandboxConfig) -> Self {
        Self {
            sandboxes: HashMap::new(),
            policies: HashMap::new(),
            resource_monitors: HashMap::new(),
            config,
        }
    }

    /// Create sandbox for extension
    pub fn create_sandbox(&mut self, extension_id: ExtensionId, policy_name: &str) -> ContextResult<String> {
        let sandbox_id = Uuid::new_v4().to_string();

        let policy = self.policies.get(policy_name)
            .ok_or_else(|| ContextError::not_found(format!("sandbox_policy:{}", policy_name)))?;

        let sandbox = ExtensionSandbox {
            id: sandbox_id.clone(),
            extension_id,
            sandbox_type: self.config.default_sandbox_type.clone(),
            policy: policy_name.to_string(),
            resource_limits: policy.resource_limits.clone(),
            status: SandboxStatus::Initializing,
            created_at: SystemTime::now(),
        };

        self.sandboxes.insert(extension_id, sandbox);

        // Create resource monitor
        let monitor = ResourceMonitor {
            extension_id,
            current_usage: ExtensionResourceUsage::default(),
            usage_history: VecDeque::new(),
            limits: policy.resource_limits.clone(),
            violations: Vec::new(),
        };

        self.resource_monitors.insert(extension_id, monitor);

        Ok(sandbox_id)
    }
}

/// Extension event dispatcher
#[derive(Debug)]
pub struct ExtensionEventDispatcher {
    /// Event subscribers
    pub subscribers: HashMap<ExtensionEventType, Vec<Box<dyn ExtensionEventHandler>>>,
    /// Event queue
    pub event_queue: VecDeque<ExtensionEvent>,
    /// Configuration
    pub config: EventDispatcherConfig,
}

/// Extension event types
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ExtensionEventType {
    /// Extension loaded
    ExtensionLoaded,
    /// Extension started
    ExtensionStarted,
    /// Extension stopped
    ExtensionStopped,
    /// Extension error
    ExtensionError,
    /// Extension configuration changed
    ConfigurationChanged,
    /// Resource limit exceeded
    ResourceLimitExceeded,
    /// Custom event
    Custom(String),
}

/// Extension event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtensionEvent {
    /// Event ID
    pub id: Uuid,
    /// Event type
    pub event_type: ExtensionEventType,
    /// Extension ID
    pub extension_id: ExtensionId,
    /// Event timestamp
    pub timestamp: SystemTime,
    /// Event data
    pub data: HashMap<String, serde_json::Value>,
    /// Event source
    pub source: String,
}

/// Extension event handler trait
pub trait ExtensionEventHandler: Send + Sync {
    /// Handle extension event
    fn handle(&mut self, event: &ExtensionEvent) -> ContextResult<()>;

    /// Get handler name
    fn name(&self) -> &str;

    /// Check if handler is interested in event type
    fn is_interested_in(&self, event_type: &ExtensionEventType) -> bool;
}

/// Event dispatcher configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventDispatcherConfig {
    /// Enable event dispatching
    pub enabled: bool,
    /// Event queue size
    pub queue_size: usize,
    /// Dispatch interval
    pub dispatch_interval: Duration,
    /// Enable async dispatching
    pub async_dispatching: bool,
}

impl Default for EventDispatcherConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            queue_size: 1000,
            dispatch_interval: Duration::from_millis(100),
            async_dispatching: true,
        }
    }
}

impl ExtensionEventDispatcher {
    /// Create a new event dispatcher
    pub fn new(config: EventDispatcherConfig) -> Self {
        Self {
            subscribers: HashMap::new(),
            event_queue: VecDeque::new(),
            config,
        }
    }

    /// Subscribe to events
    pub fn subscribe(&mut self, event_type: ExtensionEventType, handler: Box<dyn ExtensionEventHandler>) {
        self.subscribers.entry(event_type).or_insert_with(Vec::new).push(handler);
    }

    /// Dispatch event
    pub fn dispatch(&mut self, event: ExtensionEvent) -> ContextResult<()> {
        if !self.config.enabled {
            return Ok(());
        }

        self.event_queue.push_back(event);

        // Process queue if not async
        if !self.config.async_dispatching {
            self.process_queue()?;
        }

        Ok(())
    }

    /// Process event queue
    pub fn process_queue(&mut self) -> ContextResult<()> {
        while let Some(event) = self.event_queue.pop_front() {
            if let Some(handlers) = self.subscribers.get_mut(&event.event_type) {
                for handler in handlers {
                    if handler.is_interested_in(&event.event_type) {
                        if let Err(e) = handler.handle(&event) {
                            eprintln!("Event handler {} failed: {}", handler.name(), e);
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

/// Extension resource manager
#[derive(Debug)]
pub struct ExtensionResourceManager {
    /// Resource pools
    pub resource_pools: HashMap<String, ResourcePool>,
    /// Resource allocations
    pub allocations: HashMap<ExtensionId, Vec<ResourceAllocation>>,
    /// Configuration
    pub config: ResourceManagerConfig,
}

/// Resource pool
#[derive(Debug, Clone)]
pub struct ResourcePool {
    /// Pool name
    pub name: String,
    /// Resource type
    pub resource_type: String,
    /// Total capacity
    pub capacity: usize,
    /// Available resources
    pub available: usize,
    /// Allocated resources
    pub allocated: usize,
    /// Pool status
    pub status: PoolStatus,
}

/// Pool status
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PoolStatus {
    /// Pool is active
    Active,
    /// Pool is full
    Full,
    /// Pool is depleted
    Depleted,
    /// Pool is disabled
    Disabled,
}

/// Resource allocation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceAllocation {
    /// Allocation ID
    pub id: Uuid,
    /// Extension ID
    pub extension_id: ExtensionId,
    /// Resource pool
    pub pool_name: String,
    /// Allocated amount
    pub amount: usize,
    /// Allocation timestamp
    pub allocated_at: SystemTime,
    /// Allocation status
    pub status: AllocationStatus,
}

/// Allocation status
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AllocationStatus {
    /// Allocation is active
    Active,
    /// Allocation is released
    Released,
    /// Allocation is expired
    Expired,
}

/// Resource manager configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceManagerConfig {
    /// Enable resource management
    pub enabled: bool,
    /// Resource monitoring interval
    pub monitoring_interval: Duration,
    /// Enable resource pooling
    pub enable_pooling: bool,
    /// Auto-cleanup interval
    pub cleanup_interval: Duration,
}

impl Default for ResourceManagerConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            monitoring_interval: Duration::from_secs(10),
            enable_pooling: true,
            cleanup_interval: Duration::from_secs(60),
        }
    }
}

impl ExtensionResourceManager {
    /// Create a new resource manager
    pub fn new(config: ResourceManagerConfig) -> Self {
        Self {
            resource_pools: HashMap::new(),
            allocations: HashMap::new(),
            config,
        }
    }

    /// Allocate resource
    pub fn allocate(&mut self, extension_id: ExtensionId, pool_name: &str, amount: usize) -> ContextResult<Uuid> {
        let pool = self.resource_pools.get_mut(pool_name)
            .ok_or_else(|| ContextError::not_found(format!("resource_pool:{}", pool_name)))?;

        if pool.available < amount {
            return Err(ContextError::custom("insufficient_resources",
                format!("Insufficient resources in pool '{}': requested {}, available {}",
                    pool_name, amount, pool.available)));
        }

        let allocation_id = Uuid::new_v4();
        let allocation = ResourceAllocation {
            id: allocation_id,
            extension_id,
            pool_name: pool_name.to_string(),
            amount,
            allocated_at: SystemTime::now(),
            status: AllocationStatus::Active,
        };

        pool.available -= amount;
        pool.allocated += amount;

        self.allocations.entry(extension_id).or_insert_with(Vec::new).push(allocation);

        Ok(allocation_id)
    }
}

/// Extension configuration manager
#[derive(Debug)]
pub struct ExtensionConfigManager {
    /// Extension configurations
    pub configurations: HashMap<ExtensionId, ExtensionConfiguration>,
    /// Configuration sources
    pub sources: Vec<Box<dyn ConfigurationSource>>,
    /// Configuration watchers
    pub watchers: HashMap<ExtensionId, Box<dyn ConfigurationWatcher>>,
    /// Configuration cache
    pub cache: HashMap<String, CachedConfiguration>,
}

/// Configuration source trait
pub trait ConfigurationSource: Send + Sync {
    /// Load configuration
    fn load(&self, extension_id: ExtensionId) -> ContextResult<ExtensionConfiguration>;

    /// Save configuration
    fn save(&self, extension_id: ExtensionId, config: &ExtensionConfiguration) -> ContextResult<()>;

    /// Get source name
    fn name(&self) -> &str;

    /// Check if configuration exists
    fn exists(&self, extension_id: ExtensionId) -> bool;
}

/// Configuration watcher trait
pub trait ConfigurationWatcher: Send + Sync {
    /// Start watching for changes
    fn start_watching(&mut self, extension_id: ExtensionId) -> ContextResult<()>;

    /// Stop watching
    fn stop_watching(&mut self, extension_id: ExtensionId) -> ContextResult<()>;

    /// Check for changes
    fn check_changes(&self, extension_id: ExtensionId) -> ContextResult<bool>;
}

/// Cached configuration
#[derive(Debug, Clone)]
pub struct CachedConfiguration {
    /// Configuration
    pub configuration: ExtensionConfiguration,
    /// Cache timestamp
    pub cached_at: SystemTime,
    /// Cache TTL
    pub ttl: Duration,
}

impl ExtensionConfigManager {
    /// Create a new configuration manager
    pub fn new() -> Self {
        Self {
            configurations: HashMap::new(),
            sources: Vec::new(),
            watchers: HashMap::new(),
            cache: HashMap::new(),
        }
    }

    /// Add configuration source
    pub fn add_source(&mut self, source: Box<dyn ConfigurationSource>) {
        self.sources.push(source);
    }

    /// Load configuration for extension
    pub fn load_configuration(&mut self, extension_id: ExtensionId) -> ContextResult<ExtensionConfiguration> {
        // Try to load from first available source
        for source in &self.sources {
            if source.exists(extension_id) {
                let config = source.load(extension_id)?;
                self.configurations.insert(extension_id, config.clone());
                return Ok(config);
            }
        }

        // Return default configuration if no source has it
        let default_config = ExtensionConfiguration {
            values: HashMap::new(),
            source: ConfigurationSource::Default,
            last_updated: SystemTime::now(),
            version: "1.0.0".to_string(),
        };

        self.configurations.insert(extension_id, default_config.clone());
        Ok(default_config)
    }
}

/// Extension metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ExtensionMetrics {
    /// Total extensions loaded
    pub extensions_loaded: usize,
    /// Active extensions count
    pub active_extensions: usize,
    /// Total extension executions
    pub total_executions: usize,
    /// Total execution time
    pub total_execution_time: Duration,
    /// Average execution time
    pub avg_execution_time: Duration,
    /// Extension errors count
    pub extension_errors: usize,
    /// Resource violations count
    pub resource_violations: usize,
    /// Plugin load failures
    pub plugin_load_failures: usize,
    /// Extensions by type
    pub extensions_by_type: HashMap<ExtensionType, usize>,
    /// Extensions by status
    pub extensions_by_status: HashMap<ExtensionStatus, usize>,
    /// Custom metrics
    pub custom: HashMap<String, serde_json::Value>,
}

impl ExtensionContext {
    /// Create a new extension context
    pub fn new(id: String, config: ExtensionConfig) -> Self {
        Self {
            id,
            state: Arc::new(RwLock::new(ExtensionState::Initializing)),
            extension_manager: Arc::new(RwLock::new(ExtensionManager::new())),
            plugin_loader: Arc::new(Mutex::new(PluginLoader::new(PluginLoaderConfig::default()))),
            registry: Arc::new(RwLock::new(ExtensionRegistry::new(RegistryConfig::default()))),
            sandbox_manager: Arc::new(RwLock::new(SandboxManager::new(SandboxConfig::default()))),
            event_dispatcher: Arc::new(Mutex::new(ExtensionEventDispatcher::new(EventDispatcherConfig::default()))),
            resource_manager: Arc::new(Mutex::new(ExtensionResourceManager::new(ResourceManagerConfig::default()))),
            config_manager: Arc::new(RwLock::new(ExtensionConfigManager::new())),
            metrics: Arc::new(Mutex::new(ExtensionMetrics::default())),
            config: Arc::new(RwLock::new(config)),
            created_at: SystemTime::now(),
        }
    }

    /// Initialize the extension context
    pub fn initialize(&self) -> ContextResult<()> {
        let mut state = self.state.write().map_err(|e|
            ContextError::internal(format!("Failed to acquire state lock: {}", e)))?;

        if *state != ExtensionState::Initializing {
            return Err(ContextError::custom("invalid_state",
                format!("Cannot initialize extension context in state: {}", state)));
        }

        *state = ExtensionState::Active;
        Ok(())
    }

    /// Load extension
    pub fn load_extension(&self, location: ExtensionLocation, manifest: ExtensionManifest) -> ContextResult<ExtensionId> {
        let extension_id = Uuid::new_v4();

        let extension = Extension {
            id: extension_id,
            name: manifest.metadata.display_name.clone(),
            version: "1.0.0".to_string(), // Would come from manifest
            description: manifest.metadata.display_name.clone(),
            author: "unknown".to_string(), // Would come from manifest
            extension_type: ExtensionType::Plugin, // Would be determined from manifest
            status: ExtensionStatus::Loading,
            manifest: manifest.clone(),
            location,
            loaded_at: Some(SystemTime::now()),
            statistics: ExtensionStatistics::default(),
            configuration: ExtensionConfiguration {
                values: HashMap::new(),
                source: ConfigurationSource::Default,
                last_updated: SystemTime::now(),
                version: "1.0.0".to_string(),
            },
        };

        let mut extension_manager = self.extension_manager.write().map_err(|e|
            ContextError::internal(format!("Failed to acquire extension manager lock: {}", e)))?;

        extension_manager.extensions.insert(extension_id, extension);

        // Update metrics
        let mut metrics = self.metrics.lock().map_err(|e|
            ContextError::internal(format!("Failed to acquire metrics lock: {}", e)))?;
        metrics.extensions_loaded += 1;
        metrics.active_extensions += 1;

        Ok(extension_id)
    }

    /// Get extension state
    pub fn get_state(&self) -> ContextResult<ExtensionState> {
        let state = self.state.read().map_err(|e|
            ContextError::internal(format!("Failed to acquire state lock: {}", e)))?;
        Ok(*state)
    }

    /// Get extension metrics
    pub fn get_metrics(&self) -> ContextResult<ExtensionMetrics> {
        let metrics = self.metrics.lock().map_err(|e|
            ContextError::internal(format!("Failed to acquire metrics lock: {}", e)))?;
        Ok(metrics.clone())
    }

    /// List loaded extensions
    pub fn list_extensions(&self) -> ContextResult<Vec<ExtensionInfo>> {
        let extension_manager = self.extension_manager.read().map_err(|e|
            ContextError::internal(format!("Failed to acquire extension manager lock: {}", e)))?;

        let extensions = extension_manager.extensions.values()
            .map(|ext| ExtensionInfo {
                id: ext.id,
                name: ext.name.clone(),
                version: ext.version.clone(),
                description: ext.description.clone(),
                supported_operations: vec![], // Would be populated from manifest
                status: ext.status,
            })
            .collect();

        Ok(extensions)
    }
}

impl ExtensionManager {
    /// Create a new extension manager
    pub fn new() -> Self {
        Self {
            extensions: HashMap::new(),
            instances: HashMap::new(),
            load_order: Vec::new(),
            dependency_resolver: DependencyResolver::new(),
            lifecycle_manager: ExtensionLifecycleManager::new(),
        }
    }
}

// Placeholder implementations (would be implemented properly in real code)
struct DummyExtensionInstance {
    info: ExtensionInfo,
}

impl ExtensionInstance for DummyExtensionInstance {
    fn initialize(&mut self, _context: &ExtensionContext) -> ContextResult<()> {
        Ok(())
    }

    fn start(&mut self) -> ContextResult<()> {
        Ok(())
    }

    fn stop(&mut self) -> ContextResult<()> {
        Ok(())
    }

    fn execute(&mut self, _input: &ExtensionInput) -> ContextResult<ExtensionOutput> {
        Ok(ExtensionOutput {
            success: true,
            data: None,
            metadata: HashMap::new(),
            error: None,
            execution_time: Duration::from_millis(10),
        })
    }

    fn info(&self) -> &ExtensionInfo {
        &self.info
    }

    fn configure(&mut self, _config: &ExtensionConfiguration) -> ContextResult<()> {
        Ok(())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extension_context_creation() {
        let config = ExtensionConfig::default();
        let context = ExtensionContext::new("test-extension".to_string(), config);
        assert_eq!(context.id, "test-extension");
    }

    #[test]
    fn test_extension_states() {
        assert_eq!(ExtensionState::Active.to_string(), "active");
        assert_eq!(ExtensionState::Loading.to_string(), "loading");
    }

    #[test]
    fn test_extension_status() {
        assert_eq!(ExtensionStatus::Active.to_string(), "active");
        assert_eq!(ExtensionStatus::Error.to_string(), "error");
    }

    #[test]
    fn test_extension_types() {
        assert_eq!(ExtensionType::Plugin.to_string(), "plugin");
        assert_eq!(ExtensionType::Custom("test".to_string()).to_string(), "custom_test");
    }

    #[test]
    fn test_extension_capabilities() {
        assert_eq!(ExtensionCapability::FileRead.to_string(), "file_read");
        assert_eq!(ExtensionCapability::NetworkClient.to_string(), "network_client");
    }

    #[test]
    fn test_permission_levels() {
        assert!(PermissionLevel::Full > PermissionLevel::Administrative);
        assert!(PermissionLevel::Administrative > PermissionLevel::Elevated);
        assert_eq!(PermissionLevel::None.to_string(), "none");
    }

    #[test]
    fn test_plugin_formats() {
        assert_eq!(PluginFormat::WebAssembly.to_string(), "webassembly");
        assert_eq!(PluginFormat::JavaScript.to_string(), "javascript");
    }

    #[test]
    fn test_sandbox_types() {
        assert_eq!(SandboxType::Process, SandboxType::Process);
        assert_ne!(SandboxType::Process, SandboxType::Container);
    }

    #[test]
    fn test_extension_state_machine() {
        let state_machine = ExtensionStateMachine::new();
        assert!(state_machine.is_valid_transition(ExtensionStatus::Loading, ExtensionStatus::Active));
        assert!(!state_machine.is_valid_transition(ExtensionStatus::Active, ExtensionStatus::Loading));
    }

    #[test]
    fn test_dependency_resolver() {
        let resolver = DependencyResolver::new();
        let extension_id = Uuid::new_v4();
        let result = resolver.resolve(extension_id);
        assert!(result.is_ok());
    }

    #[test]
    fn test_violation_severity() {
        assert!(ViolationSeverity::Critical > ViolationSeverity::Major);
        assert!(ViolationSeverity::Major > ViolationSeverity::Minor);
    }
}