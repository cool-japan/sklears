//! Component Registry System
//!
//! This module provides comprehensive component registration, discovery, and lookup
//! capabilities including hierarchical registries, plugin management, version control,
//! and runtime component introspection for dynamic system composition.

use serde::{Deserialize, Serialize};
use sklears_core::error::{Result as SklResult, SklearsError};
use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;
use std::sync::{Arc, Mutex, RwLock};
use thiserror::Error;

use super::component_framework::{
    ComponentCapability, ComponentConfig, ComponentDependency, ComponentFactory, FactoryMetadata,
    PluggableComponent,
};

/// Global component registry for system-wide component management
///
/// Provides centralized registration, discovery, and lifecycle management of
/// components with support for hierarchical organization, version control,
/// and dynamic loading capabilities.
pub struct GlobalComponentRegistry {
    /// Component factories by type and version
    factories: Arc<RwLock<BTreeMap<String, BTreeMap<String, Arc<dyn ComponentFactory>>>>>,
    /// Active component instances
    active_components: Arc<RwLock<HashMap<String, Arc<RwLock<Box<dyn PluggableComponent>>>>>>,
    /// Component metadata cache
    metadata_cache: Arc<RwLock<HashMap<String, ComponentRegistrationMetadata>>>,
    /// Plugin directories for dynamic loading
    plugin_directories: Arc<RwLock<Vec<PathBuf>>>,
    /// Registry configuration
    config: RegistryConfiguration,
    /// Registration hooks
    hooks: Arc<RwLock<RegistryHooks>>,
    /// Registry statistics
    stats: Arc<Mutex<RegistryStatistics>>,
}

impl std::fmt::Debug for GlobalComponentRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GlobalComponentRegistry")
            .field("factories", &"[factories: <RwLock>]".to_string())
            .field(
                "active_components",
                &"[active_components: <RwLock>]".to_string(),
            )
            .field("metadata_cache", &"[metadata_cache: <RwLock>]".to_string())
            .field(
                "plugin_directories",
                &"[plugin_directories: <RwLock>]".to_string(),
            )
            .field("config", &self.config)
            .field("hooks", &"[hooks: <RwLock>]".to_string())
            .field("stats", &"[stats: <Mutex>]".to_string())
            .finish()
    }
}

impl GlobalComponentRegistry {
    /// Create a new global component registry
    #[must_use]
    pub fn new() -> Self {
        Self {
            factories: Arc::new(RwLock::new(BTreeMap::new())),
            active_components: Arc::new(RwLock::new(HashMap::new())),
            metadata_cache: Arc::new(RwLock::new(HashMap::new())),
            plugin_directories: Arc::new(RwLock::new(Vec::new())),
            config: RegistryConfiguration::default(),
            hooks: Arc::new(RwLock::new(RegistryHooks::new())),
            stats: Arc::new(Mutex::new(RegistryStatistics::new())),
        }
    }

    /// Register a component factory with version
    pub fn register_factory_versioned(
        &self,
        component_type: &str,
        version: &str,
        factory: Arc<dyn ComponentFactory>,
    ) -> SklResult<()> {
        let mut factories = self.factories.write().unwrap();
        let mut metadata_cache = self.metadata_cache.write().unwrap();
        let mut stats = self.stats.lock().unwrap();

        // Validate version format
        self.validate_version(version)?;

        // Check if already registered and if overrides are allowed
        if let Some(versions) = factories.get(component_type) {
            if versions.contains_key(version) && !self.config.allow_factory_overrides {
                return Err(SklearsError::InvalidInput(format!(
                    "Factory for component type {component_type} version {version} already registered"
                )));
            }
        }

        // Execute pre-registration hooks
        let hooks = self.hooks.read().unwrap();
        for hook in &hooks.pre_registration {
            hook(component_type, version, &factory)?;
        }

        // Register the factory
        factories
            .entry(component_type.to_string())
            .or_default()
            .insert(version.to_string(), factory.clone());

        // Cache metadata
        let metadata = ComponentRegistrationMetadata {
            component_type: component_type.to_string(),
            version: version.to_string(),
            factory_metadata: factory.factory_metadata(),
            registration_time: std::time::SystemTime::now(),
            supported_capabilities: self.extract_capabilities(&factory)?,
            dependency_requirements: self.extract_dependencies(&factory)?,
        };

        metadata_cache.insert(format!("{component_type}:{version}"), metadata);

        // Update statistics
        stats.total_registered_factories += 1;
        stats
            .registrations_by_type
            .entry(component_type.to_string())
            .and_modify(|e| *e += 1)
            .or_insert(1);

        // Execute post-registration hooks
        for hook in &hooks.post_registration {
            hook(component_type, version, &factory)?;
        }

        Ok(())
    }

    /// Register a component factory (latest version)
    pub fn register_factory(
        &self,
        component_type: &str,
        factory: Arc<dyn ComponentFactory>,
    ) -> SklResult<()> {
        let version = factory.factory_metadata().version;
        self.register_factory_versioned(component_type, &version, factory)
    }

    /// Create a component instance with specific version
    pub fn create_component_versioned(
        &self,
        component_type: &str,
        version: &str,
        config: &ComponentConfig,
    ) -> SklResult<Box<dyn PluggableComponent>> {
        let factories = self.factories.read().unwrap();
        let mut stats = self.stats.lock().unwrap();

        let factory = factories
            .get(component_type)
            .and_then(|versions| versions.get(version))
            .ok_or_else(|| {
                SklearsError::InvalidInput(format!(
                    "Component type {component_type} version {version} not registered"
                ))
            })?;

        let component = factory.create_component(config)?;

        // Update statistics
        stats.total_components_created += 1;
        stats
            .creations_by_type
            .entry(component_type.to_string())
            .and_modify(|e| *e += 1)
            .or_insert(1);

        Ok(component)
    }

    /// Create a component instance (latest version)
    pub fn create_component(
        &self,
        component_type: &str,
        config: &ComponentConfig,
    ) -> SklResult<Box<dyn PluggableComponent>> {
        let factories = self.factories.read().unwrap();

        let latest_version = factories
            .get(component_type)
            .and_then(|versions| versions.keys().last())
            .ok_or_else(|| {
                SklearsError::InvalidInput(format!(
                    "Component type {component_type} not registered"
                ))
            })?
            .clone();

        drop(factories);
        self.create_component_versioned(component_type, &latest_version, config)
    }

    /// Register active component instance
    pub fn register_active_component(
        &self,
        component_id: &str,
        component: Box<dyn PluggableComponent>,
    ) -> SklResult<()> {
        let mut active_components = self.active_components.write().unwrap();
        let mut stats = self.stats.lock().unwrap();

        if active_components.contains_key(component_id) && !self.config.allow_instance_overrides {
            return Err(SklearsError::InvalidInput(format!(
                "Component instance {component_id} already registered"
            )));
        }

        active_components.insert(component_id.to_string(), Arc::new(RwLock::new(component)));

        stats.total_active_components += 1;
        Ok(())
    }

    /// Get active component instance
    #[must_use]
    pub fn get_active_component(
        &self,
        component_id: &str,
    ) -> Option<Arc<RwLock<Box<dyn PluggableComponent>>>> {
        let active_components = self.active_components.read().unwrap();
        active_components.get(component_id).cloned()
    }

    /// Unregister active component instance
    pub fn unregister_active_component(&self, component_id: &str) -> SklResult<()> {
        let mut active_components = self.active_components.write().unwrap();
        let mut stats = self.stats.lock().unwrap();

        if active_components.remove(component_id).is_some() {
            stats.total_active_components -= 1;
            Ok(())
        } else {
            Err(SklearsError::InvalidInput(format!(
                "Component instance {component_id} not found"
            )))
        }
    }

    /// Discover available component types
    #[must_use]
    pub fn discover_component_types(&self) -> Vec<ComponentTypeInfo> {
        let factories = self.factories.read().unwrap();
        let metadata_cache = self.metadata_cache.read().unwrap();

        let mut types = Vec::new();
        for (component_type, versions) in factories.iter() {
            let mut version_info = Vec::new();
            for version in versions.keys() {
                if let Some(metadata) = metadata_cache.get(&format!("{component_type}:{version}")) {
                    version_info.push(ComponentVersionInfo {
                        version: version.clone(),
                        capabilities: metadata.supported_capabilities.clone(),
                        dependencies: metadata.dependency_requirements.clone(),
                        registration_time: metadata.registration_time,
                    });
                }
            }

            types.push(ComponentTypeInfo {
                component_type: component_type.clone(),
                available_versions: version_info,
                latest_version: versions.keys().last().cloned(),
            });
        }

        types
    }

    /// Query components by capability
    #[must_use]
    pub fn query_components_by_capability(&self, capability: &str) -> Vec<ComponentQuery> {
        let metadata_cache = self.metadata_cache.read().unwrap();
        let mut results = Vec::new();

        for (key, metadata) in metadata_cache.iter() {
            if metadata
                .supported_capabilities
                .iter()
                .any(|cap| cap.name == capability)
            {
                let parts: Vec<&str> = key.split(':').collect();
                if parts.len() == 2 {
                    results.push(ComponentQuery {
                        component_type: parts[0].to_string(),
                        version: parts[1].to_string(),
                        matching_capabilities: metadata
                            .supported_capabilities
                            .iter()
                            .filter(|cap| cap.name == capability)
                            .cloned()
                            .collect(),
                    });
                }
            }
        }

        results
    }

    /// Query components by dependency
    #[must_use]
    pub fn query_components_by_dependency(&self, dependency_type: &str) -> Vec<ComponentQuery> {
        let metadata_cache = self.metadata_cache.read().unwrap();
        let mut results = Vec::new();

        for (key, metadata) in metadata_cache.iter() {
            if metadata
                .dependency_requirements
                .iter()
                .any(|dep| dep.component_type == dependency_type)
            {
                let parts: Vec<&str> = key.split(':').collect();
                if parts.len() == 2 {
                    results.push(ComponentQuery {
                        component_type: parts[0].to_string(),
                        version: parts[1].to_string(),
                        matching_capabilities: Vec::new(),
                    });
                }
            }
        }

        results
    }

    /// Add plugin directory for dynamic loading
    pub fn add_plugin_directory(&self, path: PathBuf) -> SklResult<()> {
        let mut plugin_directories = self.plugin_directories.write().unwrap();

        if !path.exists() {
            return Err(SklearsError::InvalidInput(format!(
                "Plugin directory does not exist: {path:?}"
            )));
        }

        if !plugin_directories.contains(&path) {
            plugin_directories.push(path);
        }

        Ok(())
    }

    /// Load plugins from registered directories
    pub fn load_plugins(&self) -> SklResult<Vec<PluginLoadResult>> {
        let plugin_directories = self.plugin_directories.read().unwrap();
        let mut results = Vec::new();

        for directory in plugin_directories.iter() {
            match self.load_plugins_from_directory(directory) {
                Ok(mut dir_results) => results.append(&mut dir_results),
                Err(e) => results.push(PluginLoadResult {
                    plugin_path: directory.clone(),
                    success: false,
                    error: Some(format!("Failed to load from directory: {e}")),
                    loaded_components: Vec::new(),
                }),
            }
        }

        Ok(results)
    }

    /// Get registry statistics
    #[must_use]
    pub fn get_statistics(&self) -> RegistryStatistics {
        let stats = self.stats.lock().unwrap();
        stats.clone()
    }

    /// Configure registry hooks
    pub fn configure_hooks(&self, hooks: RegistryHooks) {
        let mut current_hooks = self.hooks.write().unwrap();
        *current_hooks = hooks;
    }

    /// Validate component version format
    fn validate_version(&self, version: &str) -> SklResult<()> {
        if !self.config.enable_version_validation {
            return Ok(());
        }

        // Simple semantic version validation (major.minor.patch)
        let parts: Vec<&str> = version.split('.').collect();
        if parts.len() != 3 {
            return Err(SklearsError::InvalidInput(format!(
                "Invalid version format: {version}"
            )));
        }

        for part in parts {
            if part.parse::<u32>().is_err() {
                return Err(SklearsError::InvalidInput(format!(
                    "Invalid version format: {version}"
                )));
            }
        }

        Ok(())
    }

    /// Extract capabilities from factory
    fn extract_capabilities(
        &self,
        factory: &Arc<dyn ComponentFactory>,
    ) -> SklResult<Vec<ComponentCapability>> {
        // This would typically introspect the factory to extract capabilities
        // For now, we'll return empty list as capabilities are defined by components
        Ok(Vec::new())
    }

    /// Extract dependencies from factory
    fn extract_dependencies(
        &self,
        factory: &Arc<dyn ComponentFactory>,
    ) -> SklResult<Vec<ComponentDependency>> {
        // This would typically introspect the factory to extract dependencies
        // For now, we'll return empty list as dependencies are defined by components
        Ok(Vec::new())
    }

    /// Load plugins from a specific directory
    fn load_plugins_from_directory(&self, directory: &PathBuf) -> SklResult<Vec<PluginLoadResult>> {
        // Plugin loading would typically involve dynamic library loading
        // This is a placeholder implementation
        let mut results = Vec::new();

        results.push(PluginLoadResult {
            plugin_path: directory.clone(),
            success: true,
            error: None,
            loaded_components: Vec::new(),
        });

        Ok(results)
    }
}

/// Component registration metadata
#[derive(Debug, Clone)]
pub struct ComponentRegistrationMetadata {
    /// Component type name
    pub component_type: String,
    /// Component version
    pub version: String,
    /// Factory metadata
    pub factory_metadata: FactoryMetadata,
    /// Registration timestamp
    pub registration_time: std::time::SystemTime,
    /// Supported capabilities
    pub supported_capabilities: Vec<ComponentCapability>,
    /// Dependency requirements
    pub dependency_requirements: Vec<ComponentDependency>,
}

/// Component type information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentTypeInfo {
    /// Component type name
    pub component_type: String,
    /// Available versions
    pub available_versions: Vec<ComponentVersionInfo>,
    /// Latest version
    pub latest_version: Option<String>,
}

/// Component version information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentVersionInfo {
    /// Version string
    pub version: String,
    /// Supported capabilities
    pub capabilities: Vec<ComponentCapability>,
    /// Dependencies
    pub dependencies: Vec<ComponentDependency>,
    /// Registration timestamp
    pub registration_time: std::time::SystemTime,
}

/// Component query result
#[derive(Debug, Clone)]
pub struct ComponentQuery {
    /// Component type
    pub component_type: String,
    /// Version
    pub version: String,
    /// Matching capabilities
    pub matching_capabilities: Vec<ComponentCapability>,
}

/// Registry configuration
#[derive(Debug, Clone)]
pub struct RegistryConfiguration {
    /// Allow factory overrides
    pub allow_factory_overrides: bool,
    /// Allow instance overrides
    pub allow_instance_overrides: bool,
    /// Enable version validation
    pub enable_version_validation: bool,
    /// Maximum registered factories
    pub max_registered_factories: Option<usize>,
    /// Maximum active components
    pub max_active_components: Option<usize>,
    /// Enable plugin auto-discovery
    pub enable_plugin_auto_discovery: bool,
    /// Plugin discovery interval
    pub plugin_discovery_interval: std::time::Duration,
}

impl Default for RegistryConfiguration {
    fn default() -> Self {
        Self {
            allow_factory_overrides: false,
            allow_instance_overrides: false,
            enable_version_validation: true,
            max_registered_factories: None,
            max_active_components: None,
            enable_plugin_auto_discovery: false,
            plugin_discovery_interval: std::time::Duration::from_secs(60),
        }
    }
}

/// Plugin loading result
#[derive(Debug, Clone)]
pub struct PluginLoadResult {
    /// Plugin path
    pub plugin_path: PathBuf,
    /// Load success status
    pub success: bool,
    /// Error message if failed
    pub error: Option<String>,
    /// List of loaded component types
    pub loaded_components: Vec<String>,
}

/// Registry hooks for extensibility
pub struct RegistryHooks {
    /// Pre-registration hooks
    pub pre_registration:
        Vec<Box<dyn Fn(&str, &str, &Arc<dyn ComponentFactory>) -> SklResult<()> + Send + Sync>>,
    /// Post-registration hooks
    pub post_registration:
        Vec<Box<dyn Fn(&str, &str, &Arc<dyn ComponentFactory>) -> SklResult<()> + Send + Sync>>,
    /// Pre-creation hooks
    pub pre_creation: Vec<Box<dyn Fn(&str, &ComponentConfig) -> SklResult<()> + Send + Sync>>,
    /// Post-creation hooks
    pub post_creation:
        Vec<Box<dyn Fn(&str, &Box<dyn PluggableComponent>) -> SklResult<()> + Send + Sync>>,
}

impl RegistryHooks {
    #[must_use]
    pub fn new() -> Self {
        Self {
            pre_registration: Vec::new(),
            post_registration: Vec::new(),
            pre_creation: Vec::new(),
            post_creation: Vec::new(),
        }
    }
}

impl std::fmt::Debug for RegistryHooks {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RegistryHooks")
            .field(
                "pre_registration",
                &format!("<{} hooks>", self.pre_registration.len()),
            )
            .field(
                "post_registration",
                &format!("<{} hooks>", self.post_registration.len()),
            )
            .field(
                "pre_creation",
                &format!("<{} hooks>", self.pre_creation.len()),
            )
            .field(
                "post_creation",
                &format!("<{} hooks>", self.post_creation.len()),
            )
            .finish()
    }
}

/// Registry statistics
#[derive(Debug, Clone)]
pub struct RegistryStatistics {
    /// Total registered factories
    pub total_registered_factories: u64,
    /// Total components created
    pub total_components_created: u64,
    /// Total active components
    pub total_active_components: u64,
    /// Registrations by type
    pub registrations_by_type: HashMap<String, u64>,
    /// Creations by type
    pub creations_by_type: HashMap<String, u64>,
    /// Registry startup time
    pub startup_time: std::time::Instant,
}

impl RegistryStatistics {
    #[must_use]
    pub fn new() -> Self {
        Self {
            total_registered_factories: 0,
            total_components_created: 0,
            total_active_components: 0,
            registrations_by_type: HashMap::new(),
            creations_by_type: HashMap::new(),
            startup_time: std::time::Instant::now(),
        }
    }

    /// Get registry uptime
    #[must_use]
    pub fn uptime(&self) -> std::time::Duration {
        self.startup_time.elapsed()
    }

    /// Get most popular component type by registrations
    #[must_use]
    pub fn most_registered_type(&self) -> Option<String> {
        self.registrations_by_type
            .iter()
            .max_by_key(|(_, count)| *count)
            .map(|(type_name, _)| type_name.clone())
    }

    /// Get most popular component type by creations
    #[must_use]
    pub fn most_created_type(&self) -> Option<String> {
        self.creations_by_type
            .iter()
            .max_by_key(|(_, count)| *count)
            .map(|(type_name, _)| type_name.clone())
    }
}

/// Registry errors
#[derive(Debug, Error)]
pub enum RegistryError {
    #[error("Component type not found: {0}")]
    ComponentTypeNotFound(String),

    #[error("Component version not found: {component_type}:{version}")]
    ComponentVersionNotFound {
        component_type: String,
        version: String,
    },

    #[error("Component already registered: {0}")]
    ComponentAlreadyRegistered(String),

    #[error("Plugin loading failed: {0}")]
    PluginLoadingFailed(String),

    #[error("Registry capacity exceeded: {0}")]
    CapacityExceeded(String),

    #[error("Version validation failed: {0}")]
    VersionValidationFailed(String),
}

impl Default for GlobalComponentRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for RegistryHooks {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for RegistryStatistics {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::modular_framework::component_framework::{ComponentEvent, FactoryMetadata};
    use crate::modular_framework::{
        ComponentCapability, ComponentDependency, ComponentMetrics, ComponentState, HealthStatus,
    };
    use std::any::Any;

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
    fn test_registry_creation() {
        let registry = GlobalComponentRegistry::new();
        let stats = registry.get_statistics();
        assert_eq!(stats.total_registered_factories, 0);
        assert_eq!(stats.total_components_created, 0);
    }

    #[test]
    fn test_factory_registration() {
        let registry = GlobalComponentRegistry::new();
        let factory = Arc::new(MockFactory);

        let result = registry.register_factory_versioned("test_component", "1.0.0", factory);
        assert!(result.is_ok());

        let stats = registry.get_statistics();
        assert_eq!(stats.total_registered_factories, 1);
    }

    #[test]
    fn test_component_creation() {
        let registry = GlobalComponentRegistry::new();
        let factory = Arc::new(MockFactory);

        registry
            .register_factory_versioned("test_component", "1.0.0", factory)
            .unwrap();

        let config = ComponentConfig::new("test_instance", "test_component");
        let component = registry.create_component_versioned("test_component", "1.0.0", &config);

        assert!(component.is_ok());
        let stats = registry.get_statistics();
        assert_eq!(stats.total_components_created, 1);
    }

    #[test]
    fn test_component_discovery() {
        let registry = GlobalComponentRegistry::new();
        let factory = Arc::new(MockFactory);

        registry
            .register_factory_versioned("test_component", "1.0.0", factory)
            .unwrap();

        let types = registry.discover_component_types();
        assert_eq!(types.len(), 1);
        assert_eq!(types[0].component_type, "test_component");
        assert_eq!(types[0].latest_version, Some("1.0.0".to_string()));
    }

    #[test]
    fn test_version_validation() {
        let registry = GlobalComponentRegistry::new();

        assert!(registry.validate_version("1.0.0").is_ok());
        assert!(registry.validate_version("10.20.30").is_ok());
        assert!(registry.validate_version("1.0").is_err());
        assert!(registry.validate_version("1.0.0.0").is_err());
        assert!(registry.validate_version("1.a.0").is_err());
    }

    #[test]
    fn test_active_component_management() {
        let registry = GlobalComponentRegistry::new();
        let component = Box::new(MockComponent::new("test", "mock"));

        let result = registry.register_active_component("test_instance", component);
        assert!(result.is_ok());

        let retrieved = registry.get_active_component("test_instance");
        assert!(retrieved.is_some());

        let unregister_result = registry.unregister_active_component("test_instance");
        assert!(unregister_result.is_ok());

        let retrieved_after = registry.get_active_component("test_instance");
        assert!(retrieved_after.is_none());
    }

    #[test]
    fn test_plugin_directory_management() {
        let registry = GlobalComponentRegistry::new();
        let temp_dir = std::env::temp_dir();

        let result = registry.add_plugin_directory(temp_dir.clone());
        assert!(result.is_ok());

        // Adding the same directory again should not cause issues
        let result2 = registry.add_plugin_directory(temp_dir);
        assert!(result2.is_ok());
    }
}
