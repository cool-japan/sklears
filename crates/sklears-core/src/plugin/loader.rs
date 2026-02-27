//! Plugin Loader
//!
//! This module provides dynamic library loading capabilities for the plugin system.
//! It enables loading plugins from shared libraries at runtime with proper
//! lifecycle management and error handling.

use super::registry::PluginRegistry;
#[cfg(feature = "dynamic_loading")]
use super::Plugin;
use crate::error::{Result, SklearsError};
#[cfg(feature = "dynamic_loading")]
use std::collections::HashMap;
use std::sync::Arc;

/// Plugin loader for dynamic library loading
///
/// The PluginLoader provides functionality to load plugins from dynamic libraries
/// at runtime. It manages the lifecycle of loaded libraries and integrates with
/// the plugin registry for automatic plugin registration.
///
/// # Features
///
/// - Dynamic library loading from files or directories
/// - Automatic plugin discovery and registration
/// - Proper library lifecycle management
/// - Cross-platform support (Windows DLL, Linux SO, macOS dylib)
/// - Error handling and validation
///
/// # Examples
///
/// ```rust,ignore
/// use sklears_core::plugin::{PluginLoader, PluginRegistry};
/// use std::sync::Arc;
///
/// let registry = Arc::new(PluginRegistry::new());
/// let mut loader = PluginLoader::new(registry.clone());
///
/// // Load a single plugin
/// # #[cfg(feature = "dynamic_loading")]
/// loader.load_from_library("./plugins/my_plugin.so", "my_plugin")?;
///
/// // Load all plugins from a directory
/// # #[cfg(feature = "dynamic_loading")]
/// let loaded = loader.load_from_directory("./plugins/")?;
/// println!("Loaded {} plugins", loaded.len());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// # Safety
///
/// Dynamic library loading involves unsafe operations. The plugin loader
/// takes precautions to ensure safety, but loaded plugins must be trusted
/// and properly implemented.
#[allow(dead_code)]
#[derive(Debug)]
pub struct PluginLoader {
    /// Loaded libraries (plugin_id -> library)
    /// This field is only available when the dynamic_loading feature is enabled
    #[cfg(feature = "dynamic_loading")]
    libraries: HashMap<String, libloading::Library>,

    /// Plugin registry for automatic registration
    registry: Arc<PluginRegistry>,
}

impl PluginLoader {
    /// Create a new plugin loader
    ///
    /// Creates a new plugin loader that will register loaded plugins
    /// with the provided registry.
    ///
    /// # Arguments
    ///
    /// * `registry` - The plugin registry to use for registration
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sklears_core::plugin::{PluginLoader, PluginRegistry};
    /// use std::sync::Arc;
    ///
    /// let registry = Arc::new(PluginRegistry::new());
    /// let loader = PluginLoader::new(registry);
    /// ```
    pub fn new(registry: Arc<PluginRegistry>) -> Self {
        Self {
            #[cfg(feature = "dynamic_loading")]
            libraries: HashMap::new(),
            registry,
        }
    }

    /// Load a plugin from a dynamic library
    ///
    /// This method loads a plugin from the specified library file and
    /// registers it with the given plugin ID. The library must export
    /// a `create_plugin` function that returns a boxed plugin instance.
    ///
    /// # Arguments
    ///
    /// * `library_path` - Path to the dynamic library file
    /// * `plugin_id` - Unique identifier for the plugin
    ///
    /// # Returns
    ///
    /// Ok(()) on successful loading and registration, or an error if
    /// the library cannot be loaded or the plugin cannot be created.
    ///
    /// # Safety
    ///
    /// This function uses unsafe code to load and call functions from
    /// dynamic libraries. The loaded library must be trusted and properly
    /// implement the expected plugin interface.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use sklears_core::plugin::{PluginLoader, PluginRegistry};
    /// use std::sync::Arc;
    ///
    /// let registry = Arc::new(PluginRegistry::new());
    /// let mut loader = PluginLoader::new(registry);
    ///
    /// # #[cfg(feature = "dynamic_loading")]
    /// loader.load_from_library("./libmy_plugin.so", "my_plugin")?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[cfg(feature = "dynamic_loading")]
    pub fn load_from_library(&mut self, library_path: &str, plugin_id: &str) -> Result<()> {
        unsafe {
            // Load the dynamic library
            let lib = libloading::Library::new(library_path).map_err(|e| {
                SklearsError::InvalidOperation(format!(
                    "Failed to load library '{}': {}",
                    library_path, e
                ))
            })?;

            // Get the plugin creation function
            // The library must export a function with this exact signature
            let create_plugin: libloading::Symbol<fn() -> Box<dyn Plugin>> =
                lib.get(b"create_plugin").map_err(|e| {
                    SklearsError::InvalidOperation(format!(
                        "Failed to get create_plugin symbol from '{}': {}",
                        library_path, e
                    ))
                })?;

            // Create the plugin instance
            let plugin = create_plugin();

            // Validate the plugin ID matches what's expected
            let _plugin_metadata = plugin.metadata();
            if plugin.id() != plugin_id {
                return Err(SklearsError::InvalidOperation(format!(
                    "Plugin ID mismatch: expected '{}', got '{}'",
                    plugin_id,
                    plugin.id()
                )));
            }

            // Register the plugin with the registry
            self.registry.register(plugin_id, plugin).map_err(|e| {
                SklearsError::InvalidOperation(format!(
                    "Failed to register plugin '{}': {}",
                    plugin_id, e
                ))
            })?;

            // Store the library to keep it loaded
            // This prevents the library from being unloaded while the plugin is in use
            self.libraries.insert(plugin_id.to_string(), lib);

            Ok(())
        }
    }

    /// Unload a plugin library
    ///
    /// This method unregisters the plugin and unloads its associated library.
    /// The plugin's cleanup method will be called before unloading.
    ///
    /// # Arguments
    ///
    /// * `plugin_id` - The ID of the plugin to unload
    ///
    /// # Returns
    ///
    /// Ok(()) on successful unloading, or an error if the operation fails.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use sklears_core::plugin::{PluginLoader, PluginRegistry};
    /// use std::sync::Arc;
    ///
    /// let registry = Arc::new(PluginRegistry::new());
    /// let mut loader = PluginLoader::new(registry);
    ///
    /// // Load a plugin first
    /// # #[cfg(feature = "dynamic_loading")]
    /// loader.load_from_library("./libmy_plugin.so", "my_plugin")?;
    ///
    /// // Then unload it
    /// # #[cfg(feature = "dynamic_loading")]
    /// loader.unload_library("my_plugin")?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[cfg(feature = "dynamic_loading")]
    pub fn unload_library(&mut self, plugin_id: &str) -> Result<()> {
        // Unregister the plugin first (this will call cleanup)
        self.registry.unregister(plugin_id).map_err(|e| {
            SklearsError::InvalidOperation(format!(
                "Failed to unregister plugin '{}': {}",
                plugin_id, e
            ))
        })?;

        // Remove the library (this will unload it when dropped)
        if self.libraries.remove(plugin_id).is_none() {
            return Err(SklearsError::InvalidOperation(format!(
                "Library for plugin '{}' was not found",
                plugin_id
            )));
        }

        Ok(())
    }

    /// Load plugins from a directory
    ///
    /// This method scans a directory for dynamic libraries and attempts
    /// to load plugins from each one. It recognizes common library extensions
    /// (.so, .dll, .dylib) and uses the filename (without extension) as the plugin ID.
    ///
    /// # Arguments
    ///
    /// * `directory` - Path to the directory containing plugin libraries
    ///
    /// # Returns
    ///
    /// A vector of successfully loaded plugin IDs, or an error if the
    /// directory cannot be read.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use sklears_core::plugin::{PluginLoader, PluginRegistry};
    /// use std::sync::Arc;
    ///
    /// let registry = Arc::new(PluginRegistry::new());
    /// let mut loader = PluginLoader::new(registry);
    ///
    /// # #[cfg(feature = "dynamic_loading")]
    /// let loaded_plugins = loader.load_from_directory("./plugins/")?;
    /// println!("Successfully loaded {} plugins", loaded_plugins.len());
    /// for plugin_id in &loaded_plugins {
    ///     println!("  - {}", plugin_id);
    /// }
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[cfg(feature = "dynamic_loading")]
    pub fn load_from_directory(&mut self, directory: &str) -> Result<Vec<String>> {
        use std::fs;

        // Read the directory contents
        let entries = fs::read_dir(directory).map_err(|e| {
            SklearsError::InvalidOperation(format!(
                "Failed to read directory '{}': {}",
                directory, e
            ))
        })?;

        let mut loaded_plugins = Vec::new();

        for entry in entries {
            let entry = entry.map_err(|e| {
                SklearsError::InvalidOperation(format!("Failed to read directory entry: {}", e))
            })?;

            let path = entry.path();

            // Only process files (not subdirectories)
            if path.is_file() {
                // Check for known library extensions
                if let Some(extension) = path.extension() {
                    let ext_str = extension.to_string_lossy().to_lowercase();
                    if ext_str == "so" || ext_str == "dll" || ext_str == "dylib" {
                        // Use the filename (without extension) as the plugin ID
                        if let Some(plugin_id) = path.file_stem().and_then(|s| s.to_str()) {
                            match self.load_from_library(
                                path.to_str().expect("load_from_library should succeed"),
                                plugin_id,
                            ) {
                                Ok(()) => {
                                    loaded_plugins.push(plugin_id.to_string());
                                    println!("Successfully loaded plugin: {}", plugin_id);
                                }
                                Err(e) => {
                                    eprintln!(
                                        "Failed to load plugin '{}' from '{}': {}",
                                        plugin_id,
                                        path.display(),
                                        e
                                    );
                                    // Continue loading other plugins despite failures
                                }
                            }
                        } else {
                            eprintln!(
                                "Could not determine plugin ID from filename: {}",
                                path.display()
                            );
                        }
                    }
                }
            }
        }

        Ok(loaded_plugins)
    }

    /// Get the list of loaded library plugin IDs
    ///
    /// Returns a vector of plugin IDs for all currently loaded libraries.
    ///
    /// # Returns
    ///
    /// A vector of plugin IDs for loaded libraries.
    #[cfg(feature = "dynamic_loading")]
    pub fn get_loaded_libraries(&self) -> Vec<String> {
        self.libraries.keys().cloned().collect()
    }

    /// Check if a library is loaded
    ///
    /// # Arguments
    ///
    /// * `plugin_id` - The plugin ID to check
    ///
    /// # Returns
    ///
    /// true if the library is loaded, false otherwise.
    #[cfg(feature = "dynamic_loading")]
    pub fn is_library_loaded(&self, plugin_id: &str) -> bool {
        self.libraries.contains_key(plugin_id)
    }

    /// Get the plugin registry
    ///
    /// Returns a reference to the plugin registry used by this loader.
    ///
    /// # Returns
    ///
    /// A reference to the plugin registry.
    pub fn registry(&self) -> &Arc<PluginRegistry> {
        &self.registry
    }

    /// Unload all libraries
    ///
    /// This method unloads all currently loaded libraries and unregisters
    /// all their associated plugins.
    ///
    /// # Returns
    ///
    /// Ok(()) on success, or an error if any unload operation fails.
    #[cfg(feature = "dynamic_loading")]
    pub fn unload_all(&mut self) -> Result<()> {
        let plugin_ids: Vec<String> = self.libraries.keys().cloned().collect();

        for plugin_id in plugin_ids.into_iter() {
            if let Err(e) = self.unload_library(&plugin_id) {
                eprintln!("Failed to unload plugin '{}': {}", plugin_id, e);
                // Continue unloading other plugins despite failures
            }
        }

        Ok(())
    }

    /// Get statistics about loaded libraries
    ///
    /// Returns information about the current state of the loader.
    ///
    /// # Returns
    ///
    /// A tuple of (number of loaded libraries, list of plugin IDs).
    #[cfg(feature = "dynamic_loading")]
    pub fn get_statistics(&self) -> (usize, Vec<String>) {
        let plugin_ids = self.get_loaded_libraries();
        (plugin_ids.len(), plugin_ids)
    }
}

/// Stub implementations for when dynamic loading is not available
#[cfg(not(feature = "dynamic_loading"))]
impl PluginLoader {
    /// Load a plugin from a dynamic library (stub implementation)
    ///
    /// This method is not available when the `dynamic_loading` feature is disabled.
    /// It will always return an error indicating that dynamic loading is not supported.
    pub fn load_from_library(&mut self, _library_path: &str, _plugin_id: &str) -> Result<()> {
        Err(SklearsError::InvalidOperation(
            "Dynamic loading is not enabled. Rebuild with the 'dynamic_loading' feature to use this functionality.".to_string()
        ))
    }

    /// Unload a plugin library (stub implementation)
    ///
    /// This method is not available when the `dynamic_loading` feature is disabled.
    pub fn unload_library(&mut self, _plugin_id: &str) -> Result<()> {
        Err(SklearsError::InvalidOperation(
            "Dynamic loading is not enabled. Rebuild with the 'dynamic_loading' feature to use this functionality.".to_string()
        ))
    }

    /// Load plugins from a directory (stub implementation)
    ///
    /// This method is not available when the `dynamic_loading` feature is disabled.
    pub fn load_from_directory(&mut self, _directory: &str) -> Result<Vec<String>> {
        Err(SklearsError::InvalidOperation(
            "Dynamic loading is not enabled. Rebuild with the 'dynamic_loading' feature to use this functionality.".to_string()
        ))
    }
}
