//! Plugin architecture for custom SIMD operations
//!
//! This module provides a plugin system that allows users to register and use
//! custom SIMD operations at runtime.

#[cfg(not(feature = "no-std"))]
use std::collections::HashMap;
#[cfg(not(feature = "no-std"))]
use std::fmt;
#[cfg(not(feature = "no-std"))]
use std::string::ToString;
#[cfg(not(feature = "no-std"))]
use std::sync::{Arc, Mutex, RwLock};

#[cfg(feature = "no-std")]
use alloc::collections::BTreeMap as HashMap;
#[cfg(feature = "no-std")]
use alloc::format;
#[cfg(feature = "no-std")]
use alloc::string::{String, ToString};
#[cfg(feature = "no-std")]
use alloc::sync::Arc;
#[cfg(feature = "no-std")]
use alloc::vec::Vec;
#[cfg(feature = "no-std")]
use core::fmt;
#[cfg(feature = "no-std")]
use spin::{Mutex, RwLock};

/// Trait for custom SIMD operations
pub trait SimdOperation: Send + Sync {
    /// The name of the operation
    fn name(&self) -> &str;

    /// The version of the operation
    fn version(&self) -> &str;

    /// Description of what the operation does
    fn description(&self) -> &str;

    /// Execute the operation on f32 data
    fn execute_f32(&self, input: &[f32], output: &mut [f32]) -> Result<(), PluginError>;

    /// Execute the operation on f64 data
    fn execute_f64(&self, input: &[f64], output: &mut [f64]) -> Result<(), PluginError>;

    /// Get the required input size for a given output size
    fn required_input_size(&self, output_size: usize) -> usize {
        output_size // Default 1:1 mapping
    }

    /// Check if the operation supports in-place execution
    fn supports_inplace(&self) -> bool {
        false
    }

    /// Get the SIMD width requirements
    fn simd_requirements(&self) -> SimdRequirements {
        SimdRequirements::default()
    }
}

/// SIMD requirements for an operation
#[derive(Debug, Clone)]
pub struct SimdRequirements {
    pub min_width: usize,
    pub preferred_width: usize,
    pub requires_aligned_memory: bool,
    pub requires_specific_features: Vec<String>,
}

impl Default for SimdRequirements {
    fn default() -> Self {
        Self {
            min_width: 1,
            preferred_width: 4,
            requires_aligned_memory: false,
            requires_specific_features: Vec::new(),
        }
    }
}

/// Plugin error types
#[derive(Debug, Clone)]
pub enum PluginError {
    InvalidInput(String),
    InvalidOutput(String),
    IncompatibleSizes(usize, usize),
    UnsupportedOperation(String),
    ExecutionFailed(String),
    RegistrationFailed(String),
    NotFound(String),
}

impl fmt::Display for PluginError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PluginError::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
            PluginError::InvalidOutput(msg) => write!(f, "Invalid output: {}", msg),
            PluginError::IncompatibleSizes(input, output) => {
                write!(
                    f,
                    "Incompatible sizes: input {} vs output {}",
                    input, output
                )
            }
            PluginError::UnsupportedOperation(op) => {
                write!(f, "Unsupported operation: {}", op)
            }
            PluginError::ExecutionFailed(msg) => write!(f, "Execution failed: {}", msg),
            PluginError::RegistrationFailed(msg) => write!(f, "Registration failed: {}", msg),
            PluginError::NotFound(name) => write!(f, "Plugin not found: {}", name),
        }
    }
}

#[cfg(not(feature = "no-std"))]
impl std::error::Error for PluginError {}

#[cfg(feature = "no-std")]
impl core::error::Error for PluginError {}

/// Plugin metadata
#[derive(Debug, Clone)]
pub struct PluginMetadata {
    pub name: String,
    pub version: String,
    pub description: String,
    pub author: String,
    pub license: String,
    pub dependencies: Vec<String>,
    pub simd_requirements: SimdRequirements,
}

impl Default for PluginMetadata {
    fn default() -> Self {
        Self {
            name: "Unknown".to_string(),
            version: "0.1.0".to_string(),
            description: "Custom SIMD operation".to_string(),
            author: "Unknown".to_string(),
            license: "MIT".to_string(),
            dependencies: Vec::new(),
            simd_requirements: SimdRequirements::default(),
        }
    }
}

/// Plugin wrapper that includes metadata
pub struct Plugin {
    pub metadata: PluginMetadata,
    pub operation: Arc<dyn SimdOperation>,
}

impl Plugin {
    pub fn new(operation: Arc<dyn SimdOperation>) -> Self {
        let metadata = PluginMetadata {
            name: operation.name().to_string(),
            version: operation.version().to_string(),
            description: operation.description().to_string(),
            ..Default::default()
        };

        Self {
            metadata,
            operation,
        }
    }

    pub fn with_metadata(operation: Arc<dyn SimdOperation>, metadata: PluginMetadata) -> Self {
        Self {
            metadata,
            operation,
        }
    }
}

/// Plugin registry for managing custom SIMD operations
pub struct PluginRegistry {
    plugins: RwLock<HashMap<String, Arc<Plugin>>>,
    execution_stats: Mutex<HashMap<String, ExecutionStats>>,
}

#[derive(Debug, Clone, Default)]
pub struct ExecutionStats {
    pub total_calls: u64,
    pub total_elements_processed: u64,
    #[cfg(not(feature = "no-std"))]
    pub total_execution_time: std::time::Duration,
    pub last_error: Option<String>,
}

impl Default for PluginRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl PluginRegistry {
    /// Create a new plugin registry
    pub fn new() -> Self {
        Self {
            plugins: RwLock::new(HashMap::new()),
            execution_stats: Mutex::new(HashMap::new()),
        }
    }

    /// Helper function to handle RwLock read locking in both std and no-std environments
    #[cfg(not(feature = "no-std"))]
    fn read_plugins(&self) -> std::sync::RwLockReadGuard<'_, HashMap<String, Arc<Plugin>>> {
        self.plugins.read().unwrap()
    }

    #[cfg(feature = "no-std")]
    fn read_plugins(&self) -> spin::RwLockReadGuard<'_, HashMap<String, Arc<Plugin>>> {
        self.plugins.read()
    }

    /// Helper function to handle RwLock write locking in both std and no-std environments
    #[cfg(not(feature = "no-std"))]
    fn write_plugins(&self) -> std::sync::RwLockWriteGuard<'_, HashMap<String, Arc<Plugin>>> {
        self.plugins.write().unwrap()
    }

    #[cfg(feature = "no-std")]
    fn write_plugins(&self) -> spin::RwLockWriteGuard<'_, HashMap<String, Arc<Plugin>>> {
        self.plugins.write()
    }

    /// Helper function to handle Mutex locking in both std and no-std environments
    #[cfg(not(feature = "no-std"))]
    fn lock_stats(&self) -> std::sync::MutexGuard<'_, HashMap<String, ExecutionStats>> {
        self.execution_stats.lock().unwrap()
    }

    #[cfg(feature = "no-std")]
    fn lock_stats(&self) -> spin::MutexGuard<'_, HashMap<String, ExecutionStats>> {
        self.execution_stats.lock()
    }

    /// Register a new plugin
    pub fn register(&self, plugin: Plugin) -> Result<(), PluginError> {
        let name = plugin.metadata.name.clone();

        // Validate plugin
        self.validate_plugin(&plugin)?;

        // Register the plugin
        let mut plugins = self.write_plugins();
        plugins.insert(name.clone(), Arc::new(plugin));

        // Initialize stats
        let mut stats = self.lock_stats();
        stats.insert(name, ExecutionStats::default());

        Ok(())
    }

    /// Unregister a plugin
    pub fn unregister(&self, name: &str) -> Result<(), PluginError> {
        let mut plugins = self.write_plugins();
        plugins.remove(name);

        let mut stats = self.lock_stats();
        stats.remove(name);

        Ok(())
    }

    /// Get a plugin by name
    pub fn get(&self, name: &str) -> Result<Arc<Plugin>, PluginError> {
        let plugins = self.read_plugins();
        plugins
            .get(name)
            .cloned()
            .ok_or_else(|| PluginError::NotFound(name.to_string()))
    }

    /// List all registered plugins
    pub fn list(&self) -> Vec<String> {
        let plugins = self.read_plugins();
        plugins.keys().cloned().collect()
    }

    /// Execute a plugin operation on f32 data
    pub fn execute_f32(
        &self,
        name: &str,
        input: &[f32],
        output: &mut [f32],
    ) -> Result<(), PluginError> {
        let plugin = self.get(name)?;

        #[cfg(not(feature = "no-std"))]
        let start_time = std::time::Instant::now();
        let result = plugin.operation.execute_f32(input, output);
        #[cfg(not(feature = "no-std"))]
        let execution_time = start_time.elapsed();

        // Update stats
        #[cfg(not(feature = "no-std"))]
        self.update_stats(name, input.len(), execution_time, result.as_ref().err());
        #[cfg(feature = "no-std")]
        self.update_stats(name, input.len(), result.as_ref().err());

        result
    }

    /// Execute a plugin operation on f64 data
    pub fn execute_f64(
        &self,
        name: &str,
        input: &[f64],
        output: &mut [f64],
    ) -> Result<(), PluginError> {
        let plugin = self.get(name)?;

        #[cfg(not(feature = "no-std"))]
        let start_time = std::time::Instant::now();
        let result = plugin.operation.execute_f64(input, output);
        #[cfg(not(feature = "no-std"))]
        let execution_time = start_time.elapsed();

        // Update stats
        #[cfg(not(feature = "no-std"))]
        self.update_stats(name, input.len(), execution_time, result.as_ref().err());
        #[cfg(feature = "no-std")]
        self.update_stats(name, input.len(), result.as_ref().err());

        result
    }

    /// Get execution statistics for a plugin
    pub fn get_stats(&self, name: &str) -> Option<ExecutionStats> {
        let stats = self.lock_stats();
        stats.get(name).cloned()
    }

    /// Clear execution statistics
    pub fn clear_stats(&self) {
        let mut stats = self.lock_stats();
        for stat in stats.values_mut() {
            *stat = ExecutionStats::default();
        }
    }

    /// Find plugins by capability
    pub fn find_by_capability(&self, requires_inplace: bool, min_width: usize) -> Vec<String> {
        let plugins = self.read_plugins();
        plugins
            .iter()
            .filter(|(_, plugin)| {
                let op = &plugin.operation;
                (!requires_inplace || op.supports_inplace())
                    && op.simd_requirements().min_width <= min_width
            })
            .map(|(name, _)| name.clone())
            .collect()
    }

    fn validate_plugin(&self, plugin: &Plugin) -> Result<(), PluginError> {
        let name = &plugin.metadata.name;

        // Check if already registered
        let plugins = self.read_plugins();
        if plugins.contains_key(name) {
            return Err(PluginError::RegistrationFailed(format!(
                "Plugin '{}' is already registered",
                name
            )));
        }

        // Basic validation
        if name.is_empty() {
            return Err(PluginError::RegistrationFailed(
                "Plugin name cannot be empty".to_string(),
            ));
        }

        Ok(())
    }

    #[cfg(not(feature = "no-std"))]
    fn update_stats(
        &self,
        name: &str,
        elements: usize,
        time: std::time::Duration,
        error: Option<&PluginError>,
    ) {
        let mut stats = self.lock_stats();
        if let Some(stat) = stats.get_mut(name) {
            stat.total_calls += 1;
            stat.total_elements_processed += elements as u64;
            stat.total_execution_time += time;
            if let Some(err) = error {
                stat.last_error = Some(err.to_string());
            }
        }
    }

    #[cfg(feature = "no-std")]
    fn update_stats(&self, name: &str, elements: usize, error: Option<&PluginError>) {
        let mut stats = self.lock_stats();
        if let Some(stat) = stats.get_mut(name) {
            stat.total_calls += 1;
            stat.total_elements_processed += elements as u64;
            if let Some(err) = error {
                stat.last_error = Some(err.to_string());
            }
        }
    }
}

/// Global plugin registry instance
pub static GLOBAL_REGISTRY: once_cell::sync::Lazy<PluginRegistry> =
    once_cell::sync::Lazy::new(PluginRegistry::new);

/// Convenience functions for global registry
pub mod global {
    use super::*;

    /// Register a plugin globally
    pub fn register(plugin: Plugin) -> Result<(), PluginError> {
        GLOBAL_REGISTRY.register(plugin)
    }

    /// Execute a plugin operation globally (f32)
    pub fn execute_f32(name: &str, input: &[f32], output: &mut [f32]) -> Result<(), PluginError> {
        GLOBAL_REGISTRY.execute_f32(name, input, output)
    }

    /// Execute a plugin operation globally (f64)
    pub fn execute_f64(name: &str, input: &[f64], output: &mut [f64]) -> Result<(), PluginError> {
        GLOBAL_REGISTRY.execute_f64(name, input, output)
    }

    /// List all globally registered plugins
    pub fn list() -> Vec<String> {
        GLOBAL_REGISTRY.list()
    }

    /// Get stats for a globally registered plugin
    pub fn get_stats(name: &str) -> Option<ExecutionStats> {
        GLOBAL_REGISTRY.get_stats(name)
    }
}

/// Example plugin implementations
pub mod examples {
    use super::*;

    /// Example: Custom square operation
    pub struct SquareOperation;

    impl SimdOperation for SquareOperation {
        fn name(&self) -> &str {
            "square"
        }
        fn version(&self) -> &str {
            "1.0.0"
        }
        fn description(&self) -> &str {
            "Square each element"
        }

        fn execute_f32(&self, input: &[f32], output: &mut [f32]) -> Result<(), PluginError> {
            if input.len() != output.len() {
                return Err(PluginError::IncompatibleSizes(input.len(), output.len()));
            }

            for (i, &val) in input.iter().enumerate() {
                output[i] = val * val;
            }
            Ok(())
        }

        fn execute_f64(&self, input: &[f64], output: &mut [f64]) -> Result<(), PluginError> {
            if input.len() != output.len() {
                return Err(PluginError::IncompatibleSizes(input.len(), output.len()));
            }

            for (i, &val) in input.iter().enumerate() {
                output[i] = val * val;
            }
            Ok(())
        }

        fn supports_inplace(&self) -> bool {
            true
        }
    }

    /// Example: Custom moving average operation
    pub struct MovingAverageOperation {
        window_size: usize,
    }

    impl MovingAverageOperation {
        pub fn new(window_size: usize) -> Self {
            Self { window_size }
        }
    }

    impl SimdOperation for MovingAverageOperation {
        fn name(&self) -> &str {
            "moving_average"
        }
        fn version(&self) -> &str {
            "1.0.0"
        }
        fn description(&self) -> &str {
            "Compute moving average with configurable window"
        }

        fn execute_f32(&self, input: &[f32], output: &mut [f32]) -> Result<(), PluginError> {
            if input.len() < self.window_size {
                return Err(PluginError::InvalidInput(
                    "Input too small for window size".to_string(),
                ));
            }

            let expected_output_size = input.len() - self.window_size + 1;
            if output.len() != expected_output_size {
                return Err(PluginError::IncompatibleSizes(
                    expected_output_size,
                    output.len(),
                ));
            }

            for i in 0..output.len() {
                let sum: f32 = input[i..i + self.window_size].iter().sum();
                output[i] = sum / self.window_size as f32;
            }
            Ok(())
        }

        fn execute_f64(&self, input: &[f64], output: &mut [f64]) -> Result<(), PluginError> {
            if input.len() < self.window_size {
                return Err(PluginError::InvalidInput(
                    "Input too small for window size".to_string(),
                ));
            }

            let expected_output_size = input.len() - self.window_size + 1;
            if output.len() != expected_output_size {
                return Err(PluginError::IncompatibleSizes(
                    expected_output_size,
                    output.len(),
                ));
            }

            for i in 0..output.len() {
                let sum: f64 = input[i..i + self.window_size].iter().sum();
                output[i] = sum / self.window_size as f64;
            }
            Ok(())
        }

        fn required_input_size(&self, output_size: usize) -> usize {
            output_size + self.window_size - 1
        }
    }
}

#[allow(non_snake_case)]
#[cfg(all(test, not(feature = "no-std")))]
mod tests {
    use super::examples::*;
    use super::*;

    #[cfg(feature = "no-std")]
    use alloc::vec;

    #[test]
    fn test_plugin_registration() {
        let registry = PluginRegistry::new();
        let operation = Arc::new(SquareOperation);
        let plugin = Plugin::new(operation);

        assert!(registry.register(plugin).is_ok());
        assert!(registry.list().contains(&"square".to_string()));
    }

    #[test]
    fn test_plugin_execution() {
        let registry = PluginRegistry::new();
        let operation = Arc::new(SquareOperation);
        let plugin = Plugin::new(operation);

        registry.register(plugin).unwrap();

        let input = vec![1.0, 2.0, 3.0, 4.0];
        let mut output = vec![0.0; 4];

        registry.execute_f32("square", &input, &mut output).unwrap();
        assert_eq!(output, vec![1.0, 4.0, 9.0, 16.0]);
    }

    #[test]
    fn test_moving_average_plugin() {
        let registry = PluginRegistry::new();
        let operation = Arc::new(MovingAverageOperation::new(3));
        let plugin = Plugin::new(operation);

        registry.register(plugin).unwrap();

        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut output = vec![0.0; 3]; // 5 - 3 + 1 = 3

        registry
            .execute_f32("moving_average", &input, &mut output)
            .unwrap();

        // Expected: [(1+2+3)/3, (2+3+4)/3, (3+4+5)/3] = [2.0, 3.0, 4.0]
        assert_eq!(output, vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_plugin_stats() {
        let registry = PluginRegistry::new();
        let operation = Arc::new(SquareOperation);
        let plugin = Plugin::new(operation);

        registry.register(plugin).unwrap();

        let input = vec![1.0, 2.0];
        let mut output = vec![0.0; 2];

        registry.execute_f32("square", &input, &mut output).unwrap();

        let stats = registry.get_stats("square").unwrap();
        assert_eq!(stats.total_calls, 1);
        assert_eq!(stats.total_elements_processed, 2);
    }

    #[test]
    fn test_global_registry() {
        let operation = Arc::new(SquareOperation);
        let plugin = Plugin::new(operation);

        global::register(plugin).unwrap();

        let input = vec![2.0, 3.0];
        let mut output = vec![0.0; 2];

        global::execute_f32("square", &input, &mut output).unwrap();
        assert_eq!(output, vec![4.0, 9.0]);

        let plugins = global::list();
        assert!(plugins.contains(&"square".to_string()));
    }

    #[test]
    fn test_error_handling() {
        let registry = PluginRegistry::new();

        // Test plugin not found
        let input = vec![1.0];
        let mut output = vec![0.0];
        let result = registry.execute_f32("nonexistent", &input, &mut output);
        assert!(matches!(result, Err(PluginError::NotFound(_))));

        // Test incompatible sizes
        let operation = Arc::new(SquareOperation);
        let plugin = Plugin::new(operation);
        registry.register(plugin).unwrap();

        let input = vec![1.0, 2.0];
        let mut output = vec![0.0]; // Wrong size
        let result = registry.execute_f32("square", &input, &mut output);
        assert!(matches!(result, Err(PluginError::IncompatibleSizes(_, _))));
    }
}
