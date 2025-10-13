//! Core traits and fundamental types for execution context management
//!
//! This module provides the foundational architecture for execution contexts,
//! including core traits, base structures, and coordination mechanisms.

use std::{
    collections::HashMap,
    sync::{Arc, RwLock, Mutex},
    time::{Duration, Instant, SystemTime},
    fmt::{Debug, Display},
    hash::{Hash, Hasher},
    any::Any,
};
use serde::{Serialize, Deserialize};
use uuid::Uuid;

/// Core trait defining execution context behavior
pub trait ExecutionContextTrait: Debug + Send + Sync {
    /// Get the unique identifier for this context
    fn id(&self) -> &str;

    /// Get the context type identifier
    fn context_type(&self) -> ContextType;

    /// Get the current state of the context
    fn state(&self) -> ContextState;

    /// Check if the context is active
    fn is_active(&self) -> bool;

    /// Get context metadata
    fn metadata(&self) -> &ContextMetadata;

    /// Validate the current context state
    fn validate(&self) -> Result<(), ContextError>;

    /// Clone the context with a new identifier
    fn clone_with_id(&self, new_id: String) -> Result<Box<dyn ExecutionContextTrait>, ContextError>;

    /// Get context as Any trait for downcasting
    fn as_any(&self) -> &dyn Any;

    /// Get mutable context as Any trait for downcasting
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

/// Trait for contexts that support hierarchical relationships
pub trait HierarchicalContext: ExecutionContextTrait {
    /// Get parent context ID if exists
    fn parent_id(&self) -> Option<&str>;

    /// Get child context IDs
    fn child_ids(&self) -> Vec<String>;

    /// Add a child context
    fn add_child(&mut self, child_id: String) -> Result<(), ContextError>;

    /// Remove a child context
    fn remove_child(&mut self, child_id: &str) -> Result<(), ContextError>;

    /// Check if this context is an ancestor of another
    fn is_ancestor_of(&self, other_id: &str) -> bool;

    /// Get context depth in hierarchy
    fn depth(&self) -> usize;
}

/// Trait for contexts that support event notifications
pub trait EventCapableContext: ExecutionContextTrait {
    /// Emit an event from this context
    fn emit_event(&self, event: ContextEvent) -> Result<(), ContextError>;

    /// Subscribe to events from this context
    fn subscribe(&mut self, subscriber: Box<dyn ContextEventSubscriber>) -> Result<SubscriptionId, ContextError>;

    /// Unsubscribe from events
    fn unsubscribe(&mut self, subscription_id: SubscriptionId) -> Result<(), ContextError>;

    /// Get current subscribers count
    fn subscriber_count(&self) -> usize;
}

/// Trait for contexts that support resource management
pub trait ResourceCapableContext: ExecutionContextTrait {
    /// Acquire a resource
    fn acquire_resource(&mut self, resource: ContextResource) -> Result<ResourceHandle, ContextError>;

    /// Release a resource
    fn release_resource(&mut self, handle: ResourceHandle) -> Result<(), ContextError>;

    /// Get current resource usage
    fn resource_usage(&self) -> ResourceUsage;

    /// Check resource limits
    fn check_resource_limits(&self) -> Result<(), ContextError>;
}

/// Core execution context types
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ContextType {
    /// Main execution context
    Execution,
    /// Runtime environment context
    Runtime,
    /// Security and authentication context
    Security,
    /// User session context
    Session,
    /// Diagnostic and debugging context
    Diagnostic,
    /// Compliance and regulatory context
    Compliance,
    /// Custom extension context
    Extension(String),
    /// Composite context containing multiple types
    Composite(Vec<ContextType>),
}

impl Display for ContextType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ContextType::Execution => write!(f, "execution"),
            ContextType::Runtime => write!(f, "runtime"),
            ContextType::Security => write!(f, "security"),
            ContextType::Session => write!(f, "session"),
            ContextType::Diagnostic => write!(f, "diagnostic"),
            ContextType::Compliance => write!(f, "compliance"),
            ContextType::Extension(name) => write!(f, "extension:{}", name),
            ContextType::Composite(types) => {
                write!(f, "composite:")?;
                for (i, t) in types.iter().enumerate() {
                    if i > 0 { write!(f, "+")?; }
                    write!(f, "{}", t)?;
                }
                Ok(())
            }
        }
    }
}

/// Context lifecycle states
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ContextState {
    /// Context is being initialized
    Initializing,
    /// Context is active and ready
    Active,
    /// Context is suspended
    Suspended,
    /// Context is being terminated
    Terminating,
    /// Context is terminated
    Terminated,
    /// Context is in error state
    Error,
    /// Context is being migrated
    Migrating,
}

impl Display for ContextState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ContextState::Initializing => write!(f, "initializing"),
            ContextState::Active => write!(f, "active"),
            ContextState::Suspended => write!(f, "suspended"),
            ContextState::Terminating => write!(f, "terminating"),
            ContextState::Terminated => write!(f, "terminated"),
            ContextState::Error => write!(f, "error"),
            ContextState::Migrating => write!(f, "migrating"),
        }
    }
}

/// Context isolation levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum IsolationLevel {
    /// No isolation
    None,
    /// Thread-level isolation
    Thread,
    /// Process-level isolation
    Process,
    /// Container-level isolation
    Container,
    /// Virtual machine isolation
    VirtualMachine,
    /// Hardware-level isolation
    Hardware,
}

/// Context priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum ContextPriority {
    /// Lowest priority
    Lowest = 0,
    /// Low priority
    Low = 1,
    /// Normal priority
    Normal = 2,
    /// High priority
    High = 3,
    /// Highest priority
    Highest = 4,
    /// Critical priority
    Critical = 5,
}

impl Default for ContextPriority {
    fn default() -> Self {
        ContextPriority::Normal
    }
}

/// Context metadata container
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextMetadata {
    /// Context creation timestamp
    pub created_at: SystemTime,
    /// Last update timestamp
    pub updated_at: SystemTime,
    /// Context priority
    pub priority: ContextPriority,
    /// Isolation level
    pub isolation_level: IsolationLevel,
    /// Custom tags
    pub tags: HashMap<String, String>,
    /// Context description
    pub description: Option<String>,
    /// Context owner/creator
    pub owner: Option<String>,
    /// Context version
    pub version: String,
    /// Custom attributes
    pub attributes: HashMap<String, serde_json::Value>,
}

impl Default for ContextMetadata {
    fn default() -> Self {
        let now = SystemTime::now();
        Self {
            created_at: now,
            updated_at: now,
            priority: ContextPriority::default(),
            isolation_level: IsolationLevel::Thread,
            tags: HashMap::new(),
            description: None,
            owner: None,
            version: "1.0.0".to_string(),
            attributes: HashMap::new(),
        }
    }
}

/// Context configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextConfig {
    /// Maximum lifetime duration
    pub max_lifetime: Option<Duration>,
    /// Auto-suspend after inactivity
    pub auto_suspend_timeout: Option<Duration>,
    /// Enable event notifications
    pub enable_events: bool,
    /// Enable metrics collection
    pub enable_metrics: bool,
    /// Enable audit logging
    pub enable_audit: bool,
    /// Resource limits
    pub resource_limits: ResourceLimits,
    /// Custom configuration
    pub custom: HashMap<String, serde_json::Value>,
}

impl Default for ContextConfig {
    fn default() -> Self {
        Self {
            max_lifetime: Some(Duration::from_secs(24 * 60 * 60)), // 24 hours
            auto_suspend_timeout: Some(Duration::from_secs(30 * 60)), // 30 minutes
            enable_events: true,
            enable_metrics: true,
            enable_audit: false,
            resource_limits: ResourceLimits::default(),
            custom: HashMap::new(),
        }
    }
}

/// Resource limits configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceLimits {
    /// Maximum memory usage in bytes
    pub max_memory: Option<usize>,
    /// Maximum CPU usage percentage
    pub max_cpu_percent: Option<f32>,
    /// Maximum number of file handles
    pub max_file_handles: Option<usize>,
    /// Maximum network connections
    pub max_network_connections: Option<usize>,
    /// Maximum execution time
    pub max_execution_time: Option<Duration>,
    /// Custom resource limits
    pub custom: HashMap<String, serde_json::Value>,
}

impl Default for ResourceLimits {
    fn default() -> Self {
        Self {
            max_memory: Some(1024 * 1024 * 1024), // 1GB
            max_cpu_percent: Some(80.0),
            max_file_handles: Some(1024),
            max_network_connections: Some(100),
            max_execution_time: Some(Duration::from_secs(60 * 60)), // 1 hour
            custom: HashMap::new(),
        }
    }
}

/// Resource usage tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUsage {
    /// Current memory usage in bytes
    pub memory_usage: usize,
    /// Current CPU usage percentage
    pub cpu_percent: f32,
    /// Current file handles count
    pub file_handles: usize,
    /// Current network connections count
    pub network_connections: usize,
    /// Current execution duration
    pub execution_duration: Duration,
    /// Custom resource usage
    pub custom: HashMap<String, serde_json::Value>,
}

impl Default for ResourceUsage {
    fn default() -> Self {
        Self {
            memory_usage: 0,
            cpu_percent: 0.0,
            file_handles: 0,
            network_connections: 0,
            execution_duration: Duration::from_secs(0),
            custom: HashMap::new(),
        }
    }
}

/// Context event types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ContextEvent {
    /// Context state changed
    StateChanged {
        context_id: String,
        old_state: ContextState,
        new_state: ContextState,
        timestamp: SystemTime,
    },
    /// Resource usage updated
    ResourceUsageUpdated {
        context_id: String,
        usage: ResourceUsage,
        timestamp: SystemTime,
    },
    /// Error occurred
    ErrorOccurred {
        context_id: String,
        error: String,
        timestamp: SystemTime,
    },
    /// Custom event
    Custom {
        context_id: String,
        event_type: String,
        data: serde_json::Value,
        timestamp: SystemTime,
    },
}

/// Context event subscriber trait
pub trait ContextEventSubscriber: Send + Sync {
    /// Handle a context event
    fn handle_event(&mut self, event: &ContextEvent) -> Result<(), ContextError>;

    /// Get subscriber ID
    fn subscriber_id(&self) -> &str;

    /// Check if subscriber is interested in event type
    fn is_interested_in(&self, event: &ContextEvent) -> bool;
}

/// Subscription identifier
pub type SubscriptionId = Uuid;

/// Resource handle
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ResourceHandle {
    /// Unique handle ID
    pub id: Uuid,
    /// Resource type
    pub resource_type: String,
    /// Acquisition timestamp
    pub acquired_at: SystemTime,
    /// Handle metadata
    pub metadata: HashMap<String, String>,
}

/// Context resource
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextResource {
    /// Resource type
    pub resource_type: String,
    /// Resource identifier
    pub resource_id: String,
    /// Resource metadata
    pub metadata: HashMap<String, serde_json::Value>,
    /// Required access level
    pub access_level: ResourceAccessLevel,
}

/// Resource access levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ResourceAccessLevel {
    /// Read-only access
    Read,
    /// Write access
    Write,
    /// Execute access
    Execute,
    /// Full control
    Full,
}

/// Context error types
#[derive(Debug, Clone, thiserror::Error, Serialize, Deserialize)]
pub enum ContextError {
    /// Context not found
    #[error("Context not found: {id}")]
    NotFound { id: String },

    /// Invalid context state
    #[error("Invalid context state: expected {expected}, got {actual}")]
    InvalidState { expected: String, actual: String },

    /// Context validation failed
    #[error("Context validation failed: {reason}")]
    ValidationFailed { reason: String },

    /// Resource limit exceeded
    #[error("Resource limit exceeded: {resource} ({current} > {limit})")]
    ResourceLimitExceeded { resource: String, current: String, limit: String },

    /// Permission denied
    #[error("Permission denied: {reason}")]
    PermissionDenied { reason: String },

    /// Configuration error
    #[error("Configuration error: {message}")]
    ConfigurationError { message: String },

    /// Serialization error
    #[error("Serialization error: {message}")]
    SerializationError { message: String },

    /// Internal error
    #[error("Internal error: {message}")]
    Internal { message: String },

    /// Custom error
    #[error("Custom error ({error_type}): {message}")]
    Custom { error_type: String, message: String },
}

impl ContextError {
    /// Create a validation error
    pub fn validation(reason: impl Into<String>) -> Self {
        Self::ValidationFailed { reason: reason.into() }
    }

    /// Create a not found error
    pub fn not_found(id: impl Into<String>) -> Self {
        Self::NotFound { id: id.into() }
    }

    /// Create an internal error
    pub fn internal(message: impl Into<String>) -> Self {
        Self::Internal { message: message.into() }
    }

    /// Create a custom error
    pub fn custom(error_type: impl Into<String>, message: impl Into<String>) -> Self {
        Self::Custom {
            error_type: error_type.into(),
            message: message.into()
        }
    }
}

/// Context result type
pub type ContextResult<T> = Result<T, ContextError>;

/// Context coordinator for managing multiple contexts
#[derive(Debug)]
pub struct ContextCoordinator {
    /// Active contexts
    contexts: Arc<RwLock<HashMap<String, Box<dyn ExecutionContextTrait>>>>,
    /// Context metadata
    metadata: Arc<RwLock<HashMap<String, ContextMetadata>>>,
    /// Event subscribers
    subscribers: Arc<Mutex<HashMap<SubscriptionId, Box<dyn ContextEventSubscriber>>>>,
    /// Configuration
    config: Arc<RwLock<ContextCoordinatorConfig>>,
    /// Metrics
    metrics: Arc<Mutex<ContextMetrics>>,
}

/// Context coordinator configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextCoordinatorConfig {
    /// Maximum number of contexts
    pub max_contexts: usize,
    /// Default context config
    pub default_context_config: ContextConfig,
    /// Enable global events
    pub enable_global_events: bool,
    /// Enable metrics aggregation
    pub enable_metrics_aggregation: bool,
    /// Cleanup interval
    pub cleanup_interval: Duration,
}

impl Default for ContextCoordinatorConfig {
    fn default() -> Self {
        Self {
            max_contexts: 10000,
            default_context_config: ContextConfig::default(),
            enable_global_events: true,
            enable_metrics_aggregation: true,
            cleanup_interval: Duration::from_secs(60),
        }
    }
}

/// Context metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ContextMetrics {
    /// Total contexts created
    pub total_created: usize,
    /// Current active contexts
    pub active_count: usize,
    /// Total contexts terminated
    pub total_terminated: usize,
    /// Total errors encountered
    pub total_errors: usize,
    /// Average context lifetime
    pub avg_lifetime: Duration,
    /// Resource usage statistics
    pub resource_stats: HashMap<String, ResourceUsage>,
    /// Custom metrics
    pub custom: HashMap<String, serde_json::Value>,
}

impl ContextCoordinator {
    /// Create a new context coordinator
    pub fn new(config: ContextCoordinatorConfig) -> Self {
        Self {
            contexts: Arc::new(RwLock::new(HashMap::new())),
            metadata: Arc::new(RwLock::new(HashMap::new())),
            subscribers: Arc::new(Mutex::new(HashMap::new())),
            config: Arc::new(RwLock::new(config)),
            metrics: Arc::new(Mutex::new(ContextMetrics::default())),
        }
    }

    /// Register a new context
    pub fn register_context(
        &self,
        context: Box<dyn ExecutionContextTrait>
    ) -> ContextResult<()> {
        let context_id = context.id().to_string();
        let mut contexts = self.contexts.write().map_err(|e|
            ContextError::internal(format!("Failed to acquire contexts lock: {}", e)))?;

        // Check if context already exists
        if contexts.contains_key(&context_id) {
            return Err(ContextError::custom("duplicate_context",
                format!("Context with ID '{}' already exists", context_id)));
        }

        // Check context limits
        let config = self.config.read().map_err(|e|
            ContextError::internal(format!("Failed to acquire config lock: {}", e)))?;
        if contexts.len() >= config.max_contexts {
            return Err(ContextError::custom("context_limit_exceeded",
                format!("Maximum context limit {} exceeded", config.max_contexts)));
        }

        // Store context metadata
        let mut metadata = self.metadata.write().map_err(|e|
            ContextError::internal(format!("Failed to acquire metadata lock: {}", e)))?;
        metadata.insert(context_id.clone(), context.metadata().clone());

        // Register context
        contexts.insert(context_id.clone(), context);

        // Update metrics
        let mut metrics = self.metrics.lock().map_err(|e|
            ContextError::internal(format!("Failed to acquire metrics lock: {}", e)))?;
        metrics.total_created += 1;
        metrics.active_count += 1;

        drop(config);
        drop(contexts);
        drop(metadata);
        drop(metrics);

        // Emit event
        self.emit_global_event(ContextEvent::StateChanged {
            context_id: context_id.clone(),
            old_state: ContextState::Initializing,
            new_state: ContextState::Active,
            timestamp: SystemTime::now(),
        })?;

        Ok(())
    }

    /// Get a context by ID
    pub fn get_context(&self, context_id: &str) -> ContextResult<Option<Arc<RwLock<Box<dyn ExecutionContextTrait>>>>> {
        let contexts = self.contexts.read().map_err(|e|
            ContextError::internal(format!("Failed to acquire contexts lock: {}", e)))?;

        // Note: This is a simplified version. In a real implementation,
        // we would need to handle the trait object more carefully
        if contexts.contains_key(context_id) {
            // For now, return None to indicate the context exists but
            // we need a different approach for sharing trait objects
            Ok(None)
        } else {
            Ok(None)
        }
    }

    /// Check if context exists
    pub fn context_exists(&self, context_id: &str) -> ContextResult<bool> {
        let contexts = self.contexts.read().map_err(|e|
            ContextError::internal(format!("Failed to acquire contexts lock: {}", e)))?;
        Ok(contexts.contains_key(context_id))
    }

    /// Remove a context
    pub fn remove_context(&self, context_id: &str) -> ContextResult<()> {
        let mut contexts = self.contexts.write().map_err(|e|
            ContextError::internal(format!("Failed to acquire contexts lock: {}", e)))?;

        let context = contexts.remove(context_id)
            .ok_or_else(|| ContextError::not_found(context_id))?;

        // Remove metadata
        let mut metadata = self.metadata.write().map_err(|e|
            ContextError::internal(format!("Failed to acquire metadata lock: {}", e)))?;
        metadata.remove(context_id);

        // Update metrics
        let mut metrics = self.metrics.lock().map_err(|e|
            ContextError::internal(format!("Failed to acquire metrics lock: {}", e)))?;
        metrics.active_count = metrics.active_count.saturating_sub(1);
        metrics.total_terminated += 1;

        drop(contexts);
        drop(metadata);
        drop(metrics);

        // Emit event
        self.emit_global_event(ContextEvent::StateChanged {
            context_id: context_id.to_string(),
            old_state: context.state(),
            new_state: ContextState::Terminated,
            timestamp: SystemTime::now(),
        })?;

        Ok(())
    }

    /// Get all context IDs
    pub fn list_contexts(&self) -> ContextResult<Vec<String>> {
        let contexts = self.contexts.read().map_err(|e|
            ContextError::internal(format!("Failed to acquire contexts lock: {}", e)))?;
        Ok(contexts.keys().cloned().collect())
    }

    /// Get context count
    pub fn context_count(&self) -> ContextResult<usize> {
        let contexts = self.contexts.read().map_err(|e|
            ContextError::internal(format!("Failed to acquire contexts lock: {}", e)))?;
        Ok(contexts.len())
    }

    /// Subscribe to global events
    pub fn subscribe_global_events(
        &self,
        subscriber: Box<dyn ContextEventSubscriber>
    ) -> ContextResult<SubscriptionId> {
        let subscription_id = Uuid::new_v4();
        let mut subscribers = self.subscribers.lock().map_err(|e|
            ContextError::internal(format!("Failed to acquire subscribers lock: {}", e)))?;
        subscribers.insert(subscription_id, subscriber);
        Ok(subscription_id)
    }

    /// Unsubscribe from global events
    pub fn unsubscribe_global_events(&self, subscription_id: SubscriptionId) -> ContextResult<()> {
        let mut subscribers = self.subscribers.lock().map_err(|e|
            ContextError::internal(format!("Failed to acquire subscribers lock: {}", e)))?;
        subscribers.remove(&subscription_id);
        Ok(())
    }

    /// Emit a global event
    pub fn emit_global_event(&self, event: ContextEvent) -> ContextResult<()> {
        let config = self.config.read().map_err(|e|
            ContextError::internal(format!("Failed to acquire config lock: {}", e)))?;

        if !config.enable_global_events {
            return Ok(());
        }

        drop(config);

        let mut subscribers = self.subscribers.lock().map_err(|e|
            ContextError::internal(format!("Failed to acquire subscribers lock: {}", e)))?;

        // Notify all interested subscribers
        let mut errors = Vec::new();
        for (sub_id, subscriber) in subscribers.iter_mut() {
            if subscriber.is_interested_in(&event) {
                if let Err(e) = subscriber.handle_event(&event) {
                    errors.push((*sub_id, e));
                }
            }
        }

        // Handle subscriber errors
        if !errors.is_empty() {
            for (sub_id, _error) in &errors {
                subscribers.remove(sub_id);
            }
        }

        Ok(())
    }

    /// Get current metrics
    pub fn get_metrics(&self) -> ContextResult<ContextMetrics> {
        let metrics = self.metrics.lock().map_err(|e|
            ContextError::internal(format!("Failed to acquire metrics lock: {}", e)))?;
        Ok(metrics.clone())
    }

    /// Cleanup terminated contexts
    pub fn cleanup(&self) -> ContextResult<usize> {
        let mut contexts = self.contexts.write().map_err(|e|
            ContextError::internal(format!("Failed to acquire contexts lock: {}", e)))?;

        let mut terminated_contexts = Vec::new();
        for (id, context) in contexts.iter() {
            if matches!(context.state(), ContextState::Terminated | ContextState::Error) {
                terminated_contexts.push(id.clone());
            }
        }

        let cleanup_count = terminated_contexts.len();
        for id in terminated_contexts {
            contexts.remove(&id);
        }

        // Update metrics
        let mut metrics = self.metrics.lock().map_err(|e|
            ContextError::internal(format!("Failed to acquire metrics lock: {}", e)))?;
        metrics.active_count = contexts.len();

        Ok(cleanup_count)
    }
}

/// Context validation engine
#[derive(Debug)]
pub struct ContextValidator {
    /// Validation rules
    rules: Vec<Box<dyn ContextValidationRule>>,
    /// Validation config
    config: ContextValidationConfig,
}

/// Context validation rule trait
pub trait ContextValidationRule: Send + Sync {
    /// Rule name
    fn name(&self) -> &str;

    /// Validate a context
    fn validate(&self, context: &dyn ExecutionContextTrait) -> ContextResult<()>;

    /// Check if rule applies to context type
    fn applies_to(&self, context_type: &ContextType) -> bool;
}

/// Context validation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextValidationConfig {
    /// Enable strict validation
    pub strict_mode: bool,
    /// Stop on first error
    pub fail_fast: bool,
    /// Maximum validation time
    pub max_validation_time: Duration,
}

impl Default for ContextValidationConfig {
    fn default() -> Self {
        Self {
            strict_mode: false,
            fail_fast: true,
            max_validation_time: Duration::from_secs(30),
        }
    }
}

impl ContextValidator {
    /// Create a new validator
    pub fn new(config: ContextValidationConfig) -> Self {
        Self {
            rules: Vec::new(),
            config,
        }
    }

    /// Add a validation rule
    pub fn add_rule(&mut self, rule: Box<dyn ContextValidationRule>) {
        self.rules.push(rule);
    }

    /// Validate a context
    pub fn validate(&self, context: &dyn ExecutionContextTrait) -> ContextResult<()> {
        let start_time = Instant::now();
        let context_type = context.context_type();

        for rule in &self.rules {
            // Check timeout
            if start_time.elapsed() > self.config.max_validation_time {
                return Err(ContextError::internal("Validation timeout exceeded"));
            }

            // Apply rule if applicable
            if rule.applies_to(&context_type) {
                if let Err(e) = rule.validate(context) {
                    if self.config.fail_fast {
                        return Err(e);
                    }
                }
            }
        }

        Ok(())
    }

    /// Get rule count
    pub fn rule_count(&self) -> usize {
        self.rules.len()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_context_type_display() {
        assert_eq!(ContextType::Execution.to_string(), "execution");
        assert_eq!(ContextType::Extension("test".to_string()).to_string(), "extension:test");
        assert_eq!(
            ContextType::Composite(vec![ContextType::Runtime, ContextType::Security]).to_string(),
            "composite:runtime+security"
        );
    }

    #[test]
    fn test_context_state_display() {
        assert_eq!(ContextState::Active.to_string(), "active");
        assert_eq!(ContextState::Terminated.to_string(), "terminated");
    }

    #[test]
    fn test_context_priority_ordering() {
        assert!(ContextPriority::Critical > ContextPriority::High);
        assert!(ContextPriority::High > ContextPriority::Normal);
        assert!(ContextPriority::Normal > ContextPriority::Low);
    }

    #[test]
    fn test_context_error_creation() {
        let err = ContextError::validation("test reason");
        assert!(matches!(err, ContextError::ValidationFailed { .. }));

        let err = ContextError::not_found("test-id");
        assert!(matches!(err, ContextError::NotFound { .. }));

        let err = ContextError::internal("test message");
        assert!(matches!(err, ContextError::Internal { .. }));
    }

    #[test]
    fn test_context_coordinator_creation() {
        let config = ContextCoordinatorConfig::default();
        let coordinator = ContextCoordinator::new(config);
        assert!(coordinator.context_count().unwrap() == 0);
    }

    #[test]
    fn test_context_validator_creation() {
        let config = ContextValidationConfig::default();
        let validator = ContextValidator::new(config);
        assert_eq!(validator.rule_count(), 0);
    }

    #[test]
    fn test_resource_limits_default() {
        let limits = ResourceLimits::default();
        assert!(limits.max_memory.is_some());
        assert!(limits.max_cpu_percent.is_some());
    }

    #[test]
    fn test_context_metadata_default() {
        let metadata = ContextMetadata::default();
        assert_eq!(metadata.priority, ContextPriority::Normal);
        assert_eq!(metadata.isolation_level, IsolationLevel::Thread);
        assert_eq!(metadata.version, "1.0.0");
    }
}