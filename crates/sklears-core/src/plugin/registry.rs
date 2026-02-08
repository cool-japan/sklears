//! Plugin Registry
//!
//! This module provides the central registry for managing plugins in the sklears
//! plugin system. It handles plugin registration, discovery, and lifecycle management
//! with thread-safe operations and efficient indexing.

use super::core_traits::Plugin;
use super::types_config::{PluginCategory, PluginMetadata};
use crate::error::{Result, SklearsError};
use std::any::TypeId;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// Registry for managing plugins
///
/// The PluginRegistry provides centralized management of all registered plugins
/// with thread-safe operations, efficient discovery through indexing, and
/// comprehensive lifecycle management.
///
/// # Features
///
/// - Thread-safe plugin registration and unregistration
/// - Category-based plugin organization
/// - Type compatibility indexing for automatic plugin selection
/// - Metadata caching for fast discovery
/// - Search functionality by name and description
/// - Validation and lifecycle management
///
/// # Examples
///
/// ```rust,no_run
/// use sklears_core::plugin::{PluginRegistry, Plugin, PluginMetadata, PluginCategory};
/// use sklears_core::error::Result;
/// use std::any::TypeId;
///
/// // Create a registry
/// let registry = PluginRegistry::new();
///
/// // List available plugins
/// let plugins = registry.list_plugins()?;
/// println!("Available plugins: {:?}", plugins);
///
/// // Search for plugins by category
/// let algorithms = registry.get_plugins_by_category(&PluginCategory::Algorithm)?;
/// println!("Algorithm plugins: {:?}", algorithms);
///
/// // Find plugins compatible with a specific type
/// let compatible = registry.get_compatible_plugins(TypeId::of::<f64>())?;
/// println!("f64-compatible plugins: {:?}", compatible);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug)]
pub struct PluginRegistry {
    /// Registered plugins (plugin_id -> plugin)
    plugins: Arc<RwLock<HashMap<String, Box<dyn Plugin>>>>,
    /// Plugin metadata cache for fast access
    metadata_cache: Arc<RwLock<HashMap<String, PluginMetadata>>>,
    /// Category index for efficient category-based discovery
    category_index: Arc<RwLock<HashMap<PluginCategory, Vec<String>>>>,
    /// Type compatibility index for automatic plugin selection
    type_index: Arc<RwLock<HashMap<TypeId, Vec<String>>>>,
}

impl PluginRegistry {
    /// Create a new plugin registry
    ///
    /// Initializes an empty registry with all necessary internal data structures
    /// for efficient plugin management and discovery.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sklears_core::plugin::PluginRegistry;
    ///
    /// let registry = PluginRegistry::new();
    /// assert_eq!(registry.list_plugins().unwrap().len(), 0);
    /// ```
    pub fn new() -> Self {
        Self {
            plugins: Arc::new(RwLock::new(HashMap::new())),
            metadata_cache: Arc::new(RwLock::new(HashMap::new())),
            category_index: Arc::new(RwLock::new(HashMap::new())),
            type_index: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Register a plugin in the registry
    ///
    /// This method registers a new plugin with the given ID, validates its metadata,
    /// and updates all internal indices for efficient discovery.
    ///
    /// # Arguments
    ///
    /// * `id` - Unique identifier for the plugin
    /// * `plugin` - The plugin instance to register
    ///
    /// # Returns
    ///
    /// Ok(()) on successful registration, or an error if validation fails
    /// or the plugin cannot be registered.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use sklears_core::plugin::{PluginRegistry, Plugin};
    ///
    /// let registry = PluginRegistry::new();
    /// // registry.register("my_plugin", Box::new(my_plugin_instance))?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn register(&self, id: &str, plugin: Box<dyn Plugin>) -> Result<()> {
        let metadata = plugin.metadata();

        // Validate plugin before registration
        self.validate_plugin(&metadata)?;

        // Update all indices for efficient discovery
        self.update_indices(id, &metadata)?;

        // Store plugin in the main registry
        {
            let mut plugins = self.plugins.write().map_err(|_| {
                SklearsError::InvalidOperation("Failed to acquire plugin registry lock".to_string())
            })?;
            plugins.insert(id.to_string(), plugin);
        }

        // Cache metadata for fast access
        {
            let mut cache = self.metadata_cache.write().map_err(|_| {
                SklearsError::InvalidOperation("Failed to acquire metadata cache lock".to_string())
            })?;
            cache.insert(id.to_string(), metadata);
        }

        Ok(())
    }

    /// Unregister a plugin from the registry
    ///
    /// This method removes a plugin from the registry, cleans up its resources,
    /// and updates all internal indices.
    ///
    /// # Arguments
    ///
    /// * `id` - The ID of the plugin to unregister
    ///
    /// # Returns
    ///
    /// Ok(()) on successful unregistration, or an error if the operation fails.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use sklears_core::plugin::PluginRegistry;
    ///
    /// let registry = PluginRegistry::new();
    /// // First register a plugin...
    /// // registry.register("my_plugin", Box::new(my_plugin_instance))?;
    ///
    /// // Then unregister it
    /// registry.unregister("my_plugin")?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn unregister(&self, id: &str) -> Result<()> {
        // Remove from main registry and get the plugin for cleanup
        let mut plugin = {
            let mut plugins = self.plugins.write().map_err(|_| {
                SklearsError::InvalidOperation("Failed to acquire plugin registry lock".to_string())
            })?;
            plugins.remove(id)
        };

        // Cleanup plugin resources if it exists
        if let Some(ref mut plugin) = plugin {
            plugin.cleanup()?;
        }

        // Remove from metadata cache and get metadata for index cleanup
        let metadata = {
            let mut cache = self.metadata_cache.write().map_err(|_| {
                SklearsError::InvalidOperation("Failed to acquire metadata cache lock".to_string())
            })?;
            cache.remove(id)
        };

        // Update indices if metadata existed
        if let Some(metadata) = metadata {
            self.remove_from_indices(id, &metadata)?;
        }

        Ok(())
    }

    /// Get a reference to a plugin by ID
    ///
    /// Note: This is a simplified implementation. In practice, you'd want
    /// to return a proper reference or handle that manages plugin lifetime.
    ///
    /// # Arguments
    ///
    /// * `id` - The ID of the plugin to retrieve
    ///
    /// # Returns
    ///
    /// An error in this simplified implementation, as proper plugin access
    /// requires careful lifetime management.
    pub fn get_plugin(&self, id: &str) -> Result<Arc<RwLock<Box<dyn Plugin>>>> {
        let plugins = self.plugins.read().map_err(|_| {
            SklearsError::InvalidOperation("Failed to acquire plugin registry lock".to_string())
        })?;

        if plugins.contains_key(id) {
            // Note: This is a simplified implementation
            // In practice, you'd want to return a proper reference or clone
            Err(SklearsError::InvalidOperation(
                "Plugin access needs proper lifetime management".to_string(),
            ))
        } else {
            Err(SklearsError::InvalidOperation(format!(
                "Plugin '{id}' not found"
            )))
        }
    }

    /// List all registered plugin IDs
    ///
    /// Returns a vector of all plugin IDs currently registered in the system.
    ///
    /// # Returns
    ///
    /// A vector of plugin IDs, or an error if the registry cannot be accessed.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sklears_core::plugin::PluginRegistry;
    ///
    /// let registry = PluginRegistry::new();
    /// let plugins = registry.list_plugins().unwrap();
    /// assert!(plugins.is_empty()); // No plugins registered yet
    /// ```
    pub fn list_plugins(&self) -> Result<Vec<String>> {
        let plugins = self.plugins.read().map_err(|_| {
            SklearsError::InvalidOperation("Failed to acquire plugin registry lock".to_string())
        })?;
        Ok(plugins.keys().cloned().collect())
    }

    /// Get all plugins in a specific category
    ///
    /// Returns all plugin IDs that belong to the specified category.
    ///
    /// # Arguments
    ///
    /// * `category` - The category to search for
    ///
    /// # Returns
    ///
    /// A vector of plugin IDs in the category, or an error if the index
    /// cannot be accessed.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sklears_core::plugin::{PluginRegistry, PluginCategory};
    ///
    /// let registry = PluginRegistry::new();
    /// let algorithms = registry.get_plugins_by_category(&PluginCategory::Algorithm).unwrap();
    /// assert!(algorithms.is_empty()); // No plugins registered yet
    /// ```
    pub fn get_plugins_by_category(&self, category: &PluginCategory) -> Result<Vec<String>> {
        let index = self.category_index.read().map_err(|_| {
            SklearsError::InvalidOperation("Failed to acquire category index lock".to_string())
        })?;
        Ok(index.get(category).cloned().unwrap_or_default())
    }

    /// Get plugins compatible with a specific data type
    ///
    /// Returns all plugin IDs that declare compatibility with the given TypeId.
    ///
    /// # Arguments
    ///
    /// * `type_id` - The TypeId to check compatibility for
    ///
    /// # Returns
    ///
    /// A vector of compatible plugin IDs, or an error if the index
    /// cannot be accessed.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sklears_core::plugin::PluginRegistry;
    /// use std::any::TypeId;
    ///
    /// let registry = PluginRegistry::new();
    /// let compatible = registry.get_compatible_plugins(TypeId::of::<f64>()).unwrap();
    /// assert!(compatible.is_empty()); // No plugins registered yet
    /// ```
    pub fn get_compatible_plugins(&self, type_id: TypeId) -> Result<Vec<String>> {
        let index = self.type_index.read().map_err(|_| {
            SklearsError::InvalidOperation("Failed to acquire type index lock".to_string())
        })?;
        Ok(index.get(&type_id).cloned().unwrap_or_default())
    }

    /// Get metadata for a specific plugin
    ///
    /// Returns the cached metadata for the plugin with the given ID.
    ///
    /// # Arguments
    ///
    /// * `id` - The ID of the plugin
    ///
    /// # Returns
    ///
    /// The plugin's metadata, or an error if the plugin is not found.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use sklears_core::plugin::PluginRegistry;
    ///
    /// let registry = PluginRegistry::new();
    /// // First register a plugin...
    /// // registry.register("my_plugin", Box::new(my_plugin_instance))?;
    ///
    /// // Then get its metadata
    /// let metadata = registry.get_metadata("my_plugin")?;
    /// println!("Plugin: {} v{}", metadata.name, metadata.version);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn get_metadata(&self, id: &str) -> Result<PluginMetadata> {
        let cache = self.metadata_cache.read().map_err(|_| {
            SklearsError::InvalidOperation("Failed to acquire metadata cache lock".to_string())
        })?;
        cache
            .get(id)
            .cloned()
            .ok_or_else(|| SklearsError::InvalidOperation(format!("Plugin '{id}' not found")))
    }

    /// Search plugins by name or description
    ///
    /// Performs a case-insensitive search through plugin names and descriptions
    /// for the given query string.
    ///
    /// # Arguments
    ///
    /// * `query` - The search query string
    ///
    /// # Returns
    ///
    /// A vector of plugin IDs that match the search query, or an error
    /// if the search cannot be performed.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use sklears_core::plugin::PluginRegistry;
    ///
    /// let registry = PluginRegistry::new();
    /// // Register some plugins first...
    ///
    /// // Search for plugins
    /// let matches = registry.search_plugins("regression")?;
    /// println!("Found regression plugins: {:?}", matches);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn search_plugins(&self, query: &str) -> Result<Vec<String>> {
        let cache = self.metadata_cache.read().map_err(|_| {
            SklearsError::InvalidOperation("Failed to acquire metadata cache lock".to_string())
        })?;

        let query_lower = query.to_lowercase();
        let matches: Vec<String> = cache
            .iter()
            .filter(|(_, metadata)| {
                metadata.name.to_lowercase().contains(&query_lower)
                    || metadata.description.to_lowercase().contains(&query_lower)
            })
            .map(|(id, _)| id.clone())
            .collect();

        Ok(matches)
    }

    /// Get registry statistics
    ///
    /// Returns information about the current state of the registry.
    ///
    /// # Returns
    ///
    /// A HashMap with statistics about the registry.
    pub fn get_statistics(&self) -> Result<HashMap<String, usize>> {
        let plugins = self.plugins.read().map_err(|_| {
            SklearsError::InvalidOperation("Failed to acquire plugin registry lock".to_string())
        })?;

        let category_index = self.category_index.read().map_err(|_| {
            SklearsError::InvalidOperation("Failed to acquire category index lock".to_string())
        })?;

        let mut stats = HashMap::new();
        stats.insert("total_plugins".to_string(), plugins.len());
        stats.insert("categories".to_string(), category_index.len());

        Ok(stats)
    }

    /// Validate a plugin before registration
    ///
    /// Performs validation checks on plugin metadata to ensure it meets
    /// the requirements for registration.
    ///
    /// # Arguments
    ///
    /// * `metadata` - The plugin metadata to validate
    ///
    /// # Returns
    ///
    /// Ok(()) if validation passes, or an error describing the validation failure.
    fn validate_plugin(&self, metadata: &PluginMetadata) -> Result<()> {
        if metadata.name.is_empty() {
            return Err(SklearsError::InvalidOperation(
                "Plugin name cannot be empty".to_string(),
            ));
        }

        if metadata.version.is_empty() {
            return Err(SklearsError::InvalidOperation(
                "Plugin version cannot be empty".to_string(),
            ));
        }

        if metadata.author.is_empty() {
            return Err(SklearsError::InvalidOperation(
                "Plugin author cannot be empty".to_string(),
            ));
        }

        // Validate version format (basic check)
        if !metadata.version.chars().any(|c| c.is_ascii_digit()) {
            return Err(SklearsError::InvalidOperation(
                "Plugin version must contain at least one digit".to_string(),
            ));
        }

        Ok(())
    }

    /// Update internal indices when registering a plugin
    ///
    /// Updates the category and type compatibility indices to include
    /// the newly registered plugin.
    ///
    /// # Arguments
    ///
    /// * `id` - The plugin ID
    /// * `metadata` - The plugin metadata
    ///
    /// # Returns
    ///
    /// Ok(()) on success, or an error if the indices cannot be updated.
    fn update_indices(&self, id: &str, metadata: &PluginMetadata) -> Result<()> {
        // Update category index
        {
            let mut index = self.category_index.write().map_err(|_| {
                SklearsError::InvalidOperation("Failed to acquire category index lock".to_string())
            })?;
            index
                .entry(metadata.category.clone())
                .or_insert_with(Vec::new)
                .push(id.to_string());
        }

        // Update type compatibility index
        {
            let mut index = self.type_index.write().map_err(|_| {
                SklearsError::InvalidOperation("Failed to acquire type index lock".to_string())
            })?;
            for type_id in &metadata.supported_types {
                index
                    .entry(*type_id)
                    .or_insert_with(Vec::new)
                    .push(id.to_string());
            }
        }

        Ok(())
    }

    /// Remove plugin from indices when unregistering
    ///
    /// Removes the plugin from all internal indices when it's unregistered.
    ///
    /// # Arguments
    ///
    /// * `id` - The plugin ID to remove
    /// * `metadata` - The plugin metadata
    ///
    /// # Returns
    ///
    /// Ok(()) on success, or an error if the indices cannot be updated.
    fn remove_from_indices(&self, id: &str, metadata: &PluginMetadata) -> Result<()> {
        // Remove from category index
        {
            let mut index = self.category_index.write().map_err(|_| {
                SklearsError::InvalidOperation("Failed to acquire category index lock".to_string())
            })?;
            if let Some(plugins) = index.get_mut(&metadata.category) {
                plugins.retain(|plugin_id| plugin_id != id);
                if plugins.is_empty() {
                    index.remove(&metadata.category);
                }
            }
        }

        // Remove from type index
        {
            let mut index = self.type_index.write().map_err(|_| {
                SklearsError::InvalidOperation("Failed to acquire type index lock".to_string())
            })?;
            for type_id in &metadata.supported_types {
                if let Some(plugins) = index.get_mut(type_id) {
                    plugins.retain(|plugin_id| plugin_id != id);
                    if plugins.is_empty() {
                        index.remove(type_id);
                    }
                }
            }
        }

        Ok(())
    }

    /// Check if a plugin is registered
    ///
    /// # Arguments
    ///
    /// * `id` - The plugin ID to check
    ///
    /// # Returns
    ///
    /// true if the plugin is registered, false otherwise.
    pub fn is_registered(&self, id: &str) -> bool {
        if let Ok(plugins) = self.plugins.read() {
            plugins.contains_key(id)
        } else {
            false
        }
    }

    /// Get the number of registered plugins
    ///
    /// # Returns
    ///
    /// The number of currently registered plugins.
    pub fn plugin_count(&self) -> usize {
        if let Ok(plugins) = self.plugins.read() {
            plugins.len()
        } else {
            0
        }
    }
}

impl Default for PluginRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for PluginRegistry {
    /// Clone the registry (creates a new registry with the same structure but empty)
    ///
    /// Note: This doesn't clone the actual plugins, just the registry structure.
    /// This is because plugins contain trait objects that can't be easily cloned.
    fn clone(&self) -> Self {
        Self::new()
    }
}
