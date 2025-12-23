//! Supporting types and utilities for DSL implementation
//!
//! This module contains helper structures, utility functions, and supporting
//! implementations that are used across the DSL system. It includes error handling,
//! resource management, and common data structures.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Resource configuration for DSL operations
///
/// Manages system resources like memory, CPU, and GPU for efficient
/// DSL compilation and execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceConfig {
    /// Maximum memory usage in megabytes
    pub max_memory_mb: usize,
    /// Maximum CPU cores to use
    pub max_cpu_cores: usize,
    /// GPU memory allocation in megabytes
    pub gpu_memory_mb: usize,
    /// Network bandwidth limit in megabits per second
    pub network_bandwidth_mbps: usize,
    /// Temporary storage allocation in megabytes
    pub temp_storage_mb: usize,
    /// Timeout for operations in seconds
    pub operation_timeout_seconds: u64,
}

impl Default for ResourceConfig {
    fn default() -> Self {
        Self {
            max_memory_mb: 1024,
            max_cpu_cores: num_cpus::get(),
            gpu_memory_mb: 0,
            network_bandwidth_mbps: 100,
            temp_storage_mb: 512,
            operation_timeout_seconds: 300,
        }
    }
}

/// Macro execution context for maintaining state during DSL processing
///
/// Provides a context for macro execution that includes resource management,
/// error tracking, and performance monitoring.
#[derive(Debug)]
pub struct MacroExecutionContext {
    /// Current resource usage
    pub resource_usage: Arc<Mutex<ResourceUsage>>,
    /// Compilation start time
    pub start_time: Instant,
    /// Errors encountered during execution
    pub errors: Arc<Mutex<Vec<DSLError>>>,
    /// Warnings generated during execution
    pub warnings: Arc<Mutex<Vec<DSLWarning>>>,
    /// Performance metrics
    pub metrics: Arc<Mutex<PerformanceMetrics>>,
    /// Configuration for this execution
    pub config: ResourceConfig,
}

impl MacroExecutionContext {
    /// Create a new macro execution context
    pub fn new(config: ResourceConfig) -> Self {
        Self {
            resource_usage: Arc::new(Mutex::new(ResourceUsage::default())),
            start_time: Instant::now(),
            errors: Arc::new(Mutex::new(Vec::new())),
            warnings: Arc::new(Mutex::new(Vec::new())),
            metrics: Arc::new(Mutex::new(PerformanceMetrics::default())),
            config,
        }
    }

    /// Record an error in the execution context
    pub fn add_error(&self, error: DSLError) {
        if let Ok(mut errors) = self.errors.lock() {
            errors.push(error);
        }
    }

    /// Record a warning in the execution context
    pub fn add_warning(&self, warning: DSLWarning) {
        if let Ok(mut warnings) = self.warnings.lock() {
            warnings.push(warning);
        }
    }

    /// Update performance metrics
    pub fn update_metrics<F>(&self, updater: F)
    where
        F: FnOnce(&mut PerformanceMetrics),
    {
        if let Ok(mut metrics) = self.metrics.lock() {
            updater(&mut metrics);
        }
    }

    /// Get current execution duration
    pub fn elapsed_time(&self) -> Duration {
        self.start_time.elapsed()
    }

    /// Check if execution has timed out
    pub fn is_timed_out(&self) -> bool {
        self.elapsed_time().as_secs() > self.config.operation_timeout_seconds
    }

    /// Get execution summary
    pub fn get_summary(&self) -> ExecutionSummary {
        let errors = self.errors.lock().map(|e| e.len()).unwrap_or(0);
        let warnings = self.warnings.lock().map(|w| w.len()).unwrap_or(0);
        let metrics = self
            .metrics
            .lock()
            .ok()
            .map(|m| m.clone())
            .unwrap_or_default();

        ExecutionSummary {
            duration: self.elapsed_time(),
            error_count: errors,
            warning_count: warnings,
            performance_metrics: metrics,
            success: errors == 0,
        }
    }
}

/// Current resource usage tracking
#[derive(Debug, Clone, Default)]
pub struct ResourceUsage {
    /// Current memory usage in bytes
    pub memory_bytes: usize,
    /// Current CPU usage percentage (0-100)
    pub cpu_usage_percent: f64,
    /// Current GPU memory usage in bytes
    pub gpu_memory_bytes: usize,
    /// Number of active threads
    pub active_threads: usize,
    /// Temporary files created
    pub temp_files: Vec<String>,
}

/// Performance metrics for DSL operations
#[derive(Debug, Clone, Default)]
pub struct PerformanceMetrics {
    /// Time spent parsing DSL syntax
    pub parse_time_ms: u64,
    /// Time spent generating code
    pub codegen_time_ms: u64,
    /// Time spent validating configurations
    pub validation_time_ms: u64,
    /// Peak memory usage during execution
    pub peak_memory_bytes: usize,
    /// Number of cache hits
    pub cache_hits: usize,
    /// Number of cache misses
    pub cache_misses: usize,
    /// Lines of code generated
    pub generated_lines: usize,
}

impl PerformanceMetrics {
    /// Calculate cache hit ratio
    pub fn cache_hit_ratio(&self) -> f64 {
        let total = self.cache_hits + self.cache_misses;
        if total > 0 {
            self.cache_hits as f64 / total as f64
        } else {
            0.0
        }
    }

    /// Get total execution time
    pub fn total_time_ms(&self) -> u64 {
        self.parse_time_ms + self.codegen_time_ms + self.validation_time_ms
    }
}

/// Summary of macro execution
#[derive(Debug, Clone)]
pub struct ExecutionSummary {
    /// Total execution duration
    pub duration: Duration,
    /// Number of errors encountered
    pub error_count: usize,
    /// Number of warnings generated
    pub warning_count: usize,
    /// Performance metrics
    pub performance_metrics: PerformanceMetrics,
    /// Whether execution was successful
    pub success: bool,
}

/// DSL-specific error types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DSLError {
    /// Error code for categorization
    pub code: String,
    /// Human-readable error message
    pub message: String,
    /// Source location where error occurred
    pub location: Option<SourceLocation>,
    /// Severity level of the error
    pub severity: ErrorSeverity,
    /// Suggested fixes for the error
    pub suggestions: Vec<String>,
}

/// DSL warning types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DSLWarning {
    /// Warning code for categorization
    pub code: String,
    /// Human-readable warning message
    pub message: String,
    /// Source location where warning occurred
    pub location: Option<SourceLocation>,
    /// Suggested improvements
    pub suggestions: Vec<String>,
}

/// Source code location for error reporting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceLocation {
    /// File name or identifier
    pub file: String,
    /// Line number (1-based)
    pub line: usize,
    /// Column number (1-based)
    pub column: usize,
    /// Length of the problematic span
    pub span_length: usize,
}

/// Error severity levels
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ErrorSeverity {
    /// Critical error that prevents compilation
    Fatal,
    /// Error that should be fixed
    Error,
    /// Warning that should be addressed
    Warning,
    /// Informational notice
    Info,
}

/// Cache for compiled DSL artifacts
///
/// Provides efficient caching of compiled DSL components to avoid
/// recompilation of unchanged code.
#[derive(Debug)]
pub struct DSLCache {
    /// In-memory cache of compiled artifacts
    cache: Arc<Mutex<HashMap<String, CachedArtifact>>>,
    /// Maximum cache size in bytes
    max_size_bytes: usize,
    /// Current cache size in bytes
    current_size_bytes: Arc<Mutex<usize>>,
    /// Cache hit statistics
    stats: Arc<Mutex<CacheStats>>,
}

impl DSLCache {
    /// Create a new DSL cache with specified maximum size
    pub fn new(max_size_bytes: usize) -> Self {
        Self {
            cache: Arc::new(Mutex::new(HashMap::new())),
            max_size_bytes,
            current_size_bytes: Arc::new(Mutex::new(0)),
            stats: Arc::new(Mutex::new(CacheStats::default())),
        }
    }

    /// Store an artifact in the cache
    pub fn store(&self, key: String, artifact: CachedArtifact) -> Result<(), String> {
        let artifact_size = artifact.size_bytes();

        // Check if we need to evict items
        if artifact_size > self.max_size_bytes {
            return Err("Artifact too large for cache".to_string());
        }

        let mut cache = self.cache.lock().map_err(|_| "Lock error")?;
        let mut current_size = self.current_size_bytes.lock().map_err(|_| "Lock error")?;

        // Evict items if necessary
        while *current_size + artifact_size > self.max_size_bytes && !cache.is_empty() {
            if let Some((evicted_key, evicted_artifact)) = cache.iter().next() {
                let evicted_key = evicted_key.clone();
                let evicted_size = evicted_artifact.size_bytes();
                cache.remove(&evicted_key);
                *current_size -= evicted_size;
            } else {
                break;
            }
        }

        // Store the new artifact
        cache.insert(key, artifact);
        *current_size += artifact_size;

        Ok(())
    }

    /// Retrieve an artifact from the cache
    pub fn get(&self, key: &str) -> Option<CachedArtifact> {
        let cache = self.cache.lock().ok()?;
        let mut stats = self.stats.lock().ok()?;

        if let Some(artifact) = cache.get(key) {
            stats.hits += 1;
            Some(artifact.clone())
        } else {
            stats.misses += 1;
            None
        }
    }

    /// Clear the entire cache
    pub fn clear(&self) {
        if let (Ok(mut cache), Ok(mut current_size)) =
            (self.cache.lock(), self.current_size_bytes.lock())
        {
            cache.clear();
            *current_size = 0;
        }
    }

    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        self.stats.lock().map(|s| s.clone()).unwrap_or_default()
    }
}

/// Cached DSL compilation artifact
#[derive(Debug, Clone)]
pub struct CachedArtifact {
    /// Compiled code or data
    pub content: Vec<u8>,
    /// Timestamp when artifact was created
    pub created_at: Instant,
    /// Hash of the source that generated this artifact
    pub source_hash: String,
    /// Metadata about the artifact
    pub metadata: HashMap<String, String>,
}

impl CachedArtifact {
    /// Calculate the size of this artifact in bytes
    pub fn size_bytes(&self) -> usize {
        self.content.len()
            + self.source_hash.len()
            + self
                .metadata
                .iter()
                .map(|(k, v)| k.len() + v.len())
                .sum::<usize>()
    }

    /// Check if this artifact is still valid
    pub fn is_valid(&self, max_age: Duration) -> bool {
        self.created_at.elapsed() < max_age
    }
}

/// Cache usage statistics
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    /// Number of cache hits
    pub hits: usize,
    /// Number of cache misses
    pub misses: usize,
    /// Number of evictions due to size limits
    pub evictions: usize,
}

impl CacheStats {
    /// Calculate hit ratio as a percentage
    pub fn hit_ratio(&self) -> f64 {
        let total = self.hits + self.misses;
        if total > 0 {
            (self.hits as f64 / total as f64) * 100.0
        } else {
            0.0
        }
    }
}

/// Utility functions for DSL operations
pub mod utils {
    use super::*;

    /// Generate a unique identifier for DSL artifacts
    pub fn generate_artifact_id(source: &str) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        source.hash(&mut hasher);
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();

        format!("dsl_{}_{}", hasher.finish(), timestamp)
    }

    /// Validate resource configuration
    pub fn validate_resource_config(config: &ResourceConfig) -> Result<(), String> {
        if config.max_memory_mb == 0 {
            return Err("Memory allocation must be greater than 0".to_string());
        }

        if config.max_cpu_cores == 0 {
            return Err("CPU core count must be greater than 0".to_string());
        }

        if config.operation_timeout_seconds == 0 {
            return Err("Operation timeout must be greater than 0".to_string());
        }

        Ok(())
    }

    /// Format duration for human-readable display
    pub fn format_duration(duration: Duration) -> String {
        let total_ms = duration.as_millis();

        if total_ms < 1000 {
            format!("{}ms", total_ms)
        } else if total_ms < 60_000 {
            format!("{:.2}s", total_ms as f64 / 1000.0)
        } else {
            let minutes = total_ms / 60_000;
            let seconds = (total_ms % 60_000) as f64 / 1000.0;
            format!("{}m {:.2}s", minutes, seconds)
        }
    }

    /// Format bytes for human-readable display
    pub fn format_bytes(bytes: usize) -> String {
        const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
        let mut size = bytes as f64;
        let mut unit_index = 0;

        while size >= 1024.0 && unit_index < UNITS.len() - 1 {
            size /= 1024.0;
            unit_index += 1;
        }

        if unit_index == 0 {
            format!("{} {}", bytes, UNITS[unit_index])
        } else {
            format!("{:.2} {}", size, UNITS[unit_index])
        }
    }

    /// Create a standardized error for DSL operations
    pub fn create_dsl_error(
        code: &str,
        message: &str,
        location: Option<SourceLocation>,
        severity: ErrorSeverity,
    ) -> DSLError {
        DSLError {
            code: code.to_string(),
            message: message.to_string(),
            location,
            severity,
            suggestions: Vec::new(),
        }
    }

    /// Create a standardized warning for DSL operations
    pub fn create_dsl_warning(
        code: &str,
        message: &str,
        location: Option<SourceLocation>,
    ) -> DSLWarning {
        DSLWarning {
            code: code.to_string(),
            message: message.to_string(),
            location,
            suggestions: Vec::new(),
        }
    }
}

/// Registry for managing DSL extensions and plugins
#[derive(Debug)]
pub struct DSLRegistry {
    /// Registered macro implementations
    macros: HashMap<String, MacroImplementation>,
    /// Registered code generators
    generators: HashMap<String, CodeGenerator>,
    /// Registered validators
    validators: HashMap<String, Validator>,
}

impl DSLRegistry {
    /// Create a new DSL registry
    pub fn new() -> Self {
        Self {
            macros: HashMap::new(),
            generators: HashMap::new(),
            validators: HashMap::new(),
        }
    }

    /// Register a macro implementation
    pub fn register_macro(&mut self, name: String, implementation: MacroImplementation) {
        self.macros.insert(name, implementation);
    }

    /// Register a code generator
    pub fn register_generator(&mut self, name: String, generator: CodeGenerator) {
        self.generators.insert(name, generator);
    }

    /// Register a validator
    pub fn register_validator(&mut self, name: String, validator: Validator) {
        self.validators.insert(name, validator);
    }

    /// Get a registered macro implementation
    pub fn get_macro(&self, name: &str) -> Option<&MacroImplementation> {
        self.macros.get(name)
    }

    /// List all registered macros
    pub fn list_macros(&self) -> Vec<&str> {
        self.macros.keys().map(|s| s.as_str()).collect()
    }
}

impl Default for DSLRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// Placeholder types for registry
#[derive(Debug, Clone)]
pub struct MacroImplementation {
    pub name: String,
    pub description: String,
}

#[derive(Debug, Clone)]
pub struct CodeGenerator {
    pub name: String,
    pub target_language: String,
}

#[derive(Debug, Clone)]
pub struct Validator {
    pub name: String,
    pub validation_rules: Vec<String>,
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resource_config_default() {
        let config = ResourceConfig::default();
        assert!(config.max_memory_mb > 0);
        assert!(config.max_cpu_cores > 0);
    }

    #[test]
    fn test_macro_execution_context() {
        let config = ResourceConfig::default();
        let context = MacroExecutionContext::new(config);

        assert_eq!(context.errors.lock().unwrap().len(), 0);
        assert_eq!(context.warnings.lock().unwrap().len(), 0);
    }

    #[test]
    fn test_dsl_cache() {
        let cache = DSLCache::new(1024);
        let artifact = CachedArtifact {
            content: vec![1, 2, 3, 4],
            created_at: Instant::now(),
            source_hash: "test_hash".to_string(),
            metadata: HashMap::new(),
        };

        assert!(cache
            .store("test_key".to_string(), artifact.clone())
            .is_ok());
        assert!(cache.get("test_key").is_some());
        assert!(cache.get("nonexistent_key").is_none());
    }

    #[test]
    fn test_performance_metrics() {
        let mut metrics = PerformanceMetrics::default();
        metrics.cache_hits = 7;
        metrics.cache_misses = 3;

        assert_eq!(metrics.cache_hit_ratio(), 0.7);
        assert_eq!(metrics.total_time_ms(), 0);
    }

    #[test]
    fn test_utils_format_duration() {
        use std::time::Duration;

        assert_eq!(utils::format_duration(Duration::from_millis(500)), "500ms");
        assert_eq!(utils::format_duration(Duration::from_secs(2)), "2.00s");
    }

    #[test]
    fn test_utils_format_bytes() {
        assert_eq!(utils::format_bytes(512), "512 B");
        assert_eq!(utils::format_bytes(2048), "2.00 KB");
        assert_eq!(utils::format_bytes(1_048_576), "1.00 MB");
    }

    #[test]
    fn test_dsl_registry() {
        let mut registry = DSLRegistry::new();

        let macro_impl = MacroImplementation {
            name: "test_macro".to_string(),
            description: "Test macro implementation".to_string(),
        };

        registry.register_macro("test_macro".to_string(), macro_impl);
        assert!(registry.get_macro("test_macro").is_some());
        assert_eq!(registry.list_macros(), vec!["test_macro"]);
    }
}
