//! Core Component Framework
//!
//! This module provides the fundamental abstractions for modular components including
//! component traits, factory patterns, configuration management, and component validation
//! for building pluggable and composable system architectures.

use serde::{Deserialize, Serialize};
use sklears_core::error::{Result as SklResult, SklearsError};
use std::any::Any;
use std::collections::HashMap;
use std::sync::Arc;
use thiserror::Error;

/// Core component trait defining the interface for all modular components
///
/// Provides standard lifecycle methods, configuration handling, and capability
/// introspection for components within the modular framework.
pub trait PluggableComponent: Send + Sync + Any {
    /// Initialize the component with configuration
    fn initialize(&mut self, config: &ComponentConfig) -> SklResult<()>;

    /// Start the component (transition to active state)
    fn start(&mut self) -> SklResult<()>;

    /// Stop the component (graceful shutdown)
    fn stop(&mut self) -> SklResult<()>;

    /// Pause the component (temporary suspension)
    fn pause(&mut self) -> SklResult<()>;

    /// Resume the component from paused state
    fn resume(&mut self) -> SklResult<()>;

    /// Get component identifier
    fn component_id(&self) -> &str;

    /// Get component type name
    fn component_type(&self) -> &str;

    /// Get component version
    fn version(&self) -> &str;

    /// Get current component state
    fn current_state(&self) -> ComponentState;

    /// Check if component is healthy
    fn health_check(&self) -> SklResult<HealthStatus>;

    /// Get component capabilities
    fn capabilities(&self) -> Vec<ComponentCapability>;

    /// Get component dependencies
    fn dependencies(&self) -> Vec<ComponentDependency>;

    /// Validate component configuration
    fn validate_config(&self, config: &ComponentConfig) -> SklResult<()>;

    /// Get component metrics
    fn get_metrics(&self) -> ComponentMetrics;

    /// Handle component events
    fn handle_event(&mut self, event: &ComponentEvent) -> SklResult<()>;

    /// Clone the component (for factory pattern)
    fn clone_component(&self) -> Box<dyn PluggableComponent>;

    /// Downcast to concrete type
    fn as_any(&self) -> &dyn Any;

    /// Mutable downcast to concrete type
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

/// Component states in the lifecycle
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ComponentState {
    /// Component is created but not initialized
    Created,
    /// Component is being initialized
    Initializing,
    /// Component is initialized but not started
    Ready,
    /// Component is running
    Running,
    /// Component is paused
    Paused,
    /// Component is stopping
    Stopping,
    /// Component is stopped
    Stopped,
    /// Component is in error state
    Error(String),
}

/// Component health status
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum HealthStatus {
    /// Component is healthy
    Healthy,
    /// Component has warnings but is functional
    Warning(String),
    /// Component has degraded performance
    Degraded(String),
    /// Component is unhealthy
    Unhealthy(String),
    /// Component is unresponsive
    Unresponsive,
}

/// Component capability descriptor
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ComponentCapability {
    /// Capability name
    pub name: String,
    /// Capability description
    pub description: String,
    /// Required configuration parameters
    pub required_config: Vec<String>,
    /// Optional configuration parameters
    pub optional_config: Vec<String>,
    /// Capability version
    pub version: String,
}

/// Component dependency descriptor
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ComponentDependency {
    /// Dependency component type
    pub component_type: String,
    /// Required version range
    pub version_requirement: String,
    /// Whether dependency is optional
    pub optional: bool,
    /// Required capabilities from dependency
    pub required_capabilities: Vec<String>,
}

/// Component configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentConfig {
    /// Component identifier
    pub component_id: String,
    /// Component type
    pub component_type: String,
    /// Configuration parameters
    pub parameters: HashMap<String, ConfigValue>,
    /// Environment-specific settings
    pub environment: HashMap<String, String>,
    /// Resource constraints
    pub resources: ResourceConstraints,
    /// Feature flags
    pub features: HashMap<String, bool>,
}

impl ComponentConfig {
    /// Create a new component configuration
    #[must_use]
    pub fn new(component_id: &str, component_type: &str) -> Self {
        Self {
            component_id: component_id.to_string(),
            component_type: component_type.to_string(),
            parameters: HashMap::new(),
            environment: HashMap::new(),
            resources: ResourceConstraints::default(),
            features: HashMap::new(),
        }
    }

    /// Add a configuration parameter
    #[must_use]
    pub fn with_parameter(mut self, key: &str, value: ConfigValue) -> Self {
        self.parameters.insert(key.to_string(), value);
        self
    }

    /// Add an environment variable
    #[must_use]
    pub fn with_environment(mut self, key: &str, value: &str) -> Self {
        self.environment.insert(key.to_string(), value.to_string());
        self
    }

    /// Set resource constraints
    #[must_use]
    pub fn with_resources(mut self, resources: ResourceConstraints) -> Self {
        self.resources = resources;
        self
    }

    /// Add a feature flag
    #[must_use]
    pub fn with_feature(mut self, feature: &str, enabled: bool) -> Self {
        self.features.insert(feature.to_string(), enabled);
        self
    }

    /// Get parameter value
    #[must_use]
    pub fn get_parameter(&self, key: &str) -> Option<&ConfigValue> {
        self.parameters.get(key)
    }

    /// Get environment variable
    pub fn get_environment(&self, key: &str) -> Option<&str> {
        self.environment.get(key).map(String::as_str)
    }

    /// Check if feature is enabled
    #[must_use]
    pub fn is_feature_enabled(&self, feature: &str) -> bool {
        self.features.get(feature).copied().unwrap_or(false)
    }
}

/// Configuration value types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConfigValue {
    /// String
    String(String),
    /// Integer
    Integer(i64),
    /// Float
    Float(f64),
    /// Boolean
    Boolean(bool),
    /// Array
    Array(Vec<ConfigValue>),
    /// Object
    Object(HashMap<String, ConfigValue>),
}

/// Resource constraints for components
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ResourceConstraints {
    /// Maximum memory usage in bytes
    pub max_memory: Option<u64>,
    /// Maximum CPU usage percentage
    pub max_cpu: Option<f64>,
    /// Maximum number of threads
    pub max_threads: Option<u32>,
    /// Maximum disk usage in bytes
    pub max_disk: Option<u64>,
    /// Network bandwidth limit in bytes per second
    pub max_bandwidth: Option<u64>,
}

/// Component metrics for monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentMetrics {
    /// Component uptime
    pub uptime: std::time::Duration,
    /// Memory usage in bytes
    pub memory_usage: u64,
    /// CPU usage percentage
    pub cpu_usage: f64,
    /// Number of processed requests
    pub processed_requests: u64,
    /// Number of failed requests
    pub failed_requests: u64,
    /// Average processing time
    pub average_processing_time: std::time::Duration,
    /// Custom metrics
    pub custom_metrics: HashMap<String, MetricValue>,
}

impl ComponentMetrics {
    #[must_use]
    pub fn new() -> Self {
        Self {
            uptime: std::time::Duration::from_secs(0),
            memory_usage: 0,
            cpu_usage: 0.0,
            processed_requests: 0,
            failed_requests: 0,
            average_processing_time: std::time::Duration::from_secs(0),
            custom_metrics: HashMap::new(),
        }
    }

    /// Add a custom metric
    pub fn add_custom_metric(&mut self, name: &str, value: MetricValue) {
        self.custom_metrics.insert(name.to_string(), value);
    }

    /// Get success rate
    #[must_use]
    pub fn success_rate(&self) -> f64 {
        if self.processed_requests == 0 {
            1.0
        } else {
            (self.processed_requests - self.failed_requests) as f64 / self.processed_requests as f64
        }
    }
}

/// Metric value types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MetricValue {
    /// Counter
    Counter(u64),
    /// Gauge
    Gauge(f64),
    /// Histogram
    Histogram(Vec<f64>),
    /// Timer
    Timer(std::time::Duration),
}

/// Component event for inter-component communication
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentEvent {
    /// Event identifier
    pub event_id: String,
    /// Source component
    pub source: String,
    /// Event type
    pub event_type: String,
    /// Event data
    pub data: HashMap<String, String>,
    /// Event timestamp
    pub timestamp: std::time::SystemTime,
}

/// Component factory trait for creating components
pub trait ComponentFactory: Send + Sync {
    /// Create a new component instance
    fn create_component(&self, config: &ComponentConfig) -> SklResult<Box<dyn PluggableComponent>>;

    /// Get supported component types
    fn supported_types(&self) -> Vec<String>;

    /// Validate component configuration
    fn validate_config(&self, config: &ComponentConfig) -> SklResult<()>;

    /// Get factory metadata
    fn factory_metadata(&self) -> FactoryMetadata;
}

/// Factory metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactoryMetadata {
    /// Factory name
    pub name: String,
    /// Factory version
    pub version: String,
    /// Supported component types
    pub supported_types: Vec<String>,
    /// Factory description
    pub description: String,
}

/// Component registry for managing component types and factories
pub struct ComponentRegistry {
    /// Registered factories by component type
    factories: HashMap<String, Arc<dyn ComponentFactory>>,
    /// Component type metadata
    type_metadata: HashMap<String, ComponentTypeMetadata>,
    /// Registry configuration
    config: RegistryConfig,
}

impl ComponentRegistry {
    /// Create a new component registry
    #[must_use]
    pub fn new() -> Self {
        Self {
            factories: HashMap::new(),
            type_metadata: HashMap::new(),
            config: RegistryConfig::default(),
        }
    }

    /// Register a component factory
    pub fn register_factory(
        &mut self,
        component_type: &str,
        factory: Arc<dyn ComponentFactory>,
    ) -> SklResult<()> {
        if self.factories.contains_key(component_type) && !self.config.allow_overrides {
            return Err(SklearsError::InvalidInput(format!(
                "Component type {component_type} already registered"
            )));
        }

        self.factories
            .insert(component_type.to_string(), factory.clone());

        let metadata = ComponentTypeMetadata {
            component_type: component_type.to_string(),
            factory_metadata: factory.factory_metadata(),
            registration_time: std::time::SystemTime::now(),
        };

        self.type_metadata
            .insert(component_type.to_string(), metadata);
        Ok(())
    }

    /// Create a component instance
    pub fn create_component(
        &self,
        component_type: &str,
        config: &ComponentConfig,
    ) -> SklResult<Box<dyn PluggableComponent>> {
        let factory = self.factories.get(component_type).ok_or_else(|| {
            SklearsError::InvalidInput(format!("Component type {component_type} not registered"))
        })?;

        factory.create_component(config)
    }

    /// Get registered component types
    #[must_use]
    pub fn get_registered_types(&self) -> Vec<String> {
        self.factories.keys().cloned().collect()
    }

    /// Check if component type is registered
    #[must_use]
    pub fn is_registered(&self, component_type: &str) -> bool {
        self.factories.contains_key(component_type)
    }

    /// Get component type metadata
    #[must_use]
    pub fn get_type_metadata(&self, component_type: &str) -> Option<&ComponentTypeMetadata> {
        self.type_metadata.get(component_type)
    }

    /// Unregister a component type
    pub fn unregister(&mut self, component_type: &str) -> SklResult<()> {
        self.factories.remove(component_type);
        self.type_metadata.remove(component_type);
        Ok(())
    }
}

impl std::fmt::Debug for ComponentRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ComponentRegistry")
            .field(
                "factories",
                &format!("<{} factories>", self.factories.len()),
            )
            .field("type_metadata", &self.type_metadata)
            .field("config", &self.config)
            .finish()
    }
}

/// Component type metadata
#[derive(Debug, Clone)]
pub struct ComponentTypeMetadata {
    /// Component type name
    pub component_type: String,
    /// Factory metadata
    pub factory_metadata: FactoryMetadata,
    /// Registration timestamp
    pub registration_time: std::time::SystemTime,
}

/// Registry configuration
#[derive(Debug, Clone)]
pub struct RegistryConfig {
    /// Allow factory overrides
    pub allow_overrides: bool,
    /// Enable type validation
    pub enable_type_validation: bool,
    /// Maximum registered types
    pub max_registered_types: Option<usize>,
}

impl Default for RegistryConfig {
    fn default() -> Self {
        Self {
            allow_overrides: false,
            enable_type_validation: true,
            max_registered_types: None,
        }
    }
}

/// Component framework errors
#[derive(Debug, Error)]
pub enum ComponentError {
    #[error("Component initialization failed: {0}")]
    InitializationFailed(String),

    #[error("Component state transition invalid: {from} -> {to}")]
    InvalidStateTransition { from: String, to: String },

    #[error("Component configuration invalid: {0}")]
    InvalidConfiguration(String),

    #[error("Component dependency not satisfied: {0}")]
    DependencyNotSatisfied(String),

    #[error("Component health check failed: {0}")]
    HealthCheckFailed(String),

    #[error("Component capability not supported: {0}")]
    CapabilityNotSupported(String),
}

impl Default for ComponentRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for ComponentMetrics {
    fn default() -> Self {
        Self::new()
    }
}

// ========== Missing Types for Modular Framework ==========

/// Component information details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentInfo {
    /// Component identifier
    pub id: String,
    /// Component type
    pub component_type: String,
    /// Component version
    pub version: String,
    /// Component description
    pub description: Option<String>,
    /// Component status
    pub status: ComponentStatus,
    /// Component metadata
    pub metadata: ComponentMetadata,
}

/// Component metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentMetadata {
    /// Component author
    pub author: Option<String>,
    /// Creation timestamp
    pub created_at: String,
    /// Last modified timestamp
    pub modified_at: String,
    /// Component tags
    pub tags: Vec<String>,
    /// Custom properties
    pub properties: HashMap<String, String>,
}

/// Component status
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ComponentStatus {
    /// Component is inactive
    Inactive,
    /// Component is initializing
    Initializing,
    /// Component is active and running
    Active,
    /// Component is paused
    Paused,
    /// Component is stopping
    Stopping,
    /// Component has failed
    Failed,
    /// Component is in error state
    Error(String),
}

/// Component node in the dependency graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentNode {
    /// Node identifier
    pub id: String,
    /// Component information
    pub component: ComponentInfo,
    /// Dependencies
    pub dependencies: Vec<String>,
    /// Dependents (reverse dependencies)
    pub dependents: Vec<String>,
    /// Node level in the dependency graph
    pub level: usize,
}

/// Capability mismatch error
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapabilityMismatch {
    /// Required capability
    pub required: String,
    /// Available capability
    pub available: Option<String>,
    /// Mismatch description
    pub description: String,
    /// Severity level
    pub severity: CapabilityMismatchSeverity,
}

/// Capability mismatch severity
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CapabilityMismatchSeverity {
    /// Warning level mismatch
    Warning,
    /// Error level mismatch
    Error,
    /// Critical level mismatch
    Critical,
}

/// Compatibility report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompatibilityReport {
    /// Whether components are compatible
    pub is_compatible: bool,
    /// Compatibility score (0.0 to 1.0)
    pub compatibility_score: f32,
    /// List of capability mismatches
    pub mismatches: Vec<CapabilityMismatch>,
    /// Compatibility warnings
    pub warnings: Vec<String>,
    /// Report generation timestamp
    pub generated_at: String,
}

/// Environment settings for component execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentSettings {
    /// Environment name
    pub name: String,
    /// Environment variables
    pub variables: HashMap<String, String>,
    /// Working directory
    pub working_directory: Option<String>,
    /// Resource limits
    pub resource_limits: Option<ResourceLimits>,
    /// Security settings
    pub security_settings: Option<SecuritySettings>,
}

/// Security settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecuritySettings {
    /// Enable sandboxing
    pub sandbox_enabled: bool,
    /// Allowed network access
    pub network_access: bool,
    /// Allowed file system access
    pub filesystem_access: bool,
    /// Security policy
    pub policy: Option<String>,
}

/// Execution condition for conditional component execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionCondition {
    /// Condition expression
    pub expression: String,
    /// Condition variables
    pub variables: HashMap<String, String>,
    /// Condition type
    pub condition_type: ExecutionConditionType,
}

/// Execution condition types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExecutionConditionType {
    /// Always execute
    Always,
    /// Never execute
    Never,
    /// Execute based on expression
    Expression,
    /// Execute based on dependency status
    DependencyBased,
    /// Execute based on resource availability
    ResourceBased,
}

/// Execution metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionMetadata {
    /// Execution ID
    pub execution_id: String,
    /// Start timestamp
    pub start_time: String,
    /// End timestamp
    pub end_time: Option<String>,
    /// Execution duration in milliseconds
    pub duration_ms: Option<u64>,
    /// Execution status
    pub status: ExecutionStatus,
    /// Resource usage
    pub resource_usage: Option<ResourceUsage>,
}

/// Execution status
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExecutionStatus {
    /// Execution is pending
    Pending,
    /// Execution is running
    Running,
    /// Execution completed successfully
    Completed,
    /// Execution failed
    Failed,
    /// Execution was cancelled
    Cancelled,
    /// Execution timed out
    TimedOut,
}

/// Resource usage information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUsage {
    /// CPU usage percentage
    pub cpu_usage: f32,
    /// Memory usage in MB
    pub memory_usage_mb: f32,
    /// Disk usage in MB
    pub disk_usage_mb: f32,
    /// Network usage in MB
    pub network_usage_mb: f32,
}

/// Log level for component logging
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum LogLevel {
    /// Trace level
    Trace,
    /// Debug level
    Debug,
    /// Info level
    Info,
    /// Warning level
    Warn,
    /// Error level
    Error,
    /// Critical level
    Critical,
}

/// Missing dependency error
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MissingDependency {
    /// Component that has the missing dependency
    pub component_id: String,
    /// Missing dependency name
    pub dependency_name: String,
    /// Required version
    pub required_version: Option<String>,
    /// Suggested resolution
    pub suggested_resolution: Option<String>,
}

/// Version conflict error
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionConflict {
    /// Component with version conflict
    pub component_id: String,
    /// Required version
    pub required_version: String,
    /// Available version
    pub available_version: String,
    /// Conflict description
    pub description: String,
}

/// Resource limits for component execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceLimits {
    /// Maximum CPU cores
    pub max_cpu_cores: Option<usize>,
    /// Maximum memory in MB
    pub max_memory_mb: Option<usize>,
    /// Maximum disk space in MB
    pub max_disk_mb: Option<usize>,
    /// Maximum network bandwidth in Mbps
    pub max_network_mbps: Option<f32>,
    /// Execution timeout in seconds
    pub timeout_sec: Option<u64>,
}

impl Default for ComponentStatus {
    fn default() -> Self {
        Self::Inactive
    }
}

impl Default for CapabilityMismatchSeverity {
    fn default() -> Self {
        Self::Warning
    }
}

impl Default for ExecutionConditionType {
    fn default() -> Self {
        Self::Always
    }
}

impl Default for ExecutionStatus {
    fn default() -> Self {
        Self::Pending
    }
}

impl Default for LogLevel {
    fn default() -> Self {
        Self::Info
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    /// Mock component for testing
    struct MockComponent {
        id: String,
        component_type: String,
        state: ComponentState,
        metrics: ComponentMetrics,
    }

    impl MockComponent {
        fn new(id: &str, component_type: &str) -> Self {
            Self {
                id: id.to_string(),
                component_type: component_type.to_string(),
                state: ComponentState::Created,
                metrics: ComponentMetrics::new(),
            }
        }
    }

    impl PluggableComponent for MockComponent {
        fn initialize(&mut self, _config: &ComponentConfig) -> SklResult<()> {
            self.state = ComponentState::Ready;
            Ok(())
        }

        fn start(&mut self) -> SklResult<()> {
            self.state = ComponentState::Running;
            Ok(())
        }

        fn stop(&mut self) -> SklResult<()> {
            self.state = ComponentState::Stopped;
            Ok(())
        }

        fn pause(&mut self) -> SklResult<()> {
            self.state = ComponentState::Paused;
            Ok(())
        }

        fn resume(&mut self) -> SklResult<()> {
            self.state = ComponentState::Running;
            Ok(())
        }

        fn component_id(&self) -> &str {
            &self.id
        }

        fn component_type(&self) -> &str {
            &self.component_type
        }

        fn version(&self) -> &str {
            "1.0.0"
        }

        fn current_state(&self) -> ComponentState {
            self.state.clone()
        }

        fn health_check(&self) -> SklResult<HealthStatus> {
            Ok(HealthStatus::Healthy)
        }

        fn capabilities(&self) -> Vec<ComponentCapability> {
            vec![ComponentCapability {
                name: "test_capability".to_string(),
                description: "Test capability".to_string(),
                required_config: vec![],
                optional_config: vec![],
                version: "1.0.0".to_string(),
            }]
        }

        fn dependencies(&self) -> Vec<ComponentDependency> {
            vec![]
        }

        fn validate_config(&self, _config: &ComponentConfig) -> SklResult<()> {
            Ok(())
        }

        fn get_metrics(&self) -> ComponentMetrics {
            self.metrics.clone()
        }

        fn handle_event(&mut self, _event: &ComponentEvent) -> SklResult<()> {
            Ok(())
        }

        fn clone_component(&self) -> Box<dyn PluggableComponent> {
            Box::new(MockComponent::new(&self.id, &self.component_type))
        }

        fn as_any(&self) -> &dyn Any {
            self
        }

        fn as_any_mut(&mut self) -> &mut dyn Any {
            self
        }
    }

    /// Mock factory for testing
    struct MockFactory;

    impl ComponentFactory for MockFactory {
        fn create_component(
            &self,
            config: &ComponentConfig,
        ) -> SklResult<Box<dyn PluggableComponent>> {
            Ok(Box::new(MockComponent::new(
                &config.component_id,
                &config.component_type,
            )))
        }

        fn supported_types(&self) -> Vec<String> {
            vec!["mock_component".to_string()]
        }

        fn validate_config(&self, _config: &ComponentConfig) -> SklResult<()> {
            Ok(())
        }

        fn factory_metadata(&self) -> FactoryMetadata {
            /// FactoryMetadata
            FactoryMetadata {
                name: "MockFactory".to_string(),
                version: "1.0.0".to_string(),
                supported_types: vec!["mock_component".to_string()],
                description: "Mock factory for testing".to_string(),
            }
        }
    }

    #[test]
    fn test_component_config_creation() {
        let config = ComponentConfig::new("test_component", "test_type")
            .with_parameter("param1", ConfigValue::String("value1".to_string()))
            .with_environment("ENV_VAR", "env_value")
            .with_feature("feature1", true);

        assert_eq!(config.component_id, "test_component");
        assert_eq!(config.component_type, "test_type");
        assert!(config.is_feature_enabled("feature1"));
        assert_eq!(config.get_environment("ENV_VAR"), Some("env_value"));
    }

    #[test]
    fn test_component_lifecycle() {
        let mut component = MockComponent::new("test", "mock");
        let config = ComponentConfig::new("test", "mock");

        assert_eq!(component.current_state(), ComponentState::Created);

        component.initialize(&config).unwrap();
        assert_eq!(component.current_state(), ComponentState::Ready);

        component.start().unwrap();
        assert_eq!(component.current_state(), ComponentState::Running);

        component.pause().unwrap();
        assert_eq!(component.current_state(), ComponentState::Paused);

        component.resume().unwrap();
        assert_eq!(component.current_state(), ComponentState::Running);

        component.stop().unwrap();
        assert_eq!(component.current_state(), ComponentState::Stopped);
    }

    #[test]
    fn test_component_registry() {
        let mut registry = ComponentRegistry::new();
        let factory = Arc::new(MockFactory);

        registry
            .register_factory("mock_component", factory)
            .unwrap();
        assert!(registry.is_registered("mock_component"));

        let config = ComponentConfig::new("test_instance", "mock_component");
        let component = registry
            .create_component("mock_component", &config)
            .unwrap();

        assert_eq!(component.component_id(), "test_instance");
        assert_eq!(component.component_type(), "mock_component");
    }

    #[test]
    fn test_component_metrics() {
        let mut metrics = ComponentMetrics::new();
        metrics.processed_requests = 100;
        metrics.failed_requests = 5;

        assert_eq!(metrics.success_rate(), 0.95);

        metrics.add_custom_metric("test_metric", MetricValue::Counter(42));
        assert!(metrics.custom_metrics.contains_key("test_metric"));
    }
}
