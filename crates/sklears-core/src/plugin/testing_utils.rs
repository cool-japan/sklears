//! Plugin Testing Utilities
//!
//! This module provides comprehensive testing infrastructure for plugin development,
//! validation, and quality assurance. It includes mock implementations, test fixtures,
//! performance testing tools, and validation frameworks.

use super::core_traits::Plugin;
use super::security::{DigitalSignature, Permission, PublisherInfo, SecurityPolicy};
use super::types_config::{
    PluginCapability, PluginCategory, PluginConfig, PluginMetadata, PluginParameter,
};
use super::validation::{PluginManifest, PluginValidator, ValidationReport};
use crate::error::Result;
use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::time::{Duration, Instant, SystemTime};

/// Mock plugin implementation for testing
///
/// MockPlugin provides a configurable test implementation of the Plugin trait
/// that can be used for unit testing, integration testing, and validation scenarios.
///
/// # Features
///
/// - Configurable metadata and behavior
/// - Simulation of various plugin states and conditions
/// - Built-in test data generation
/// - Error injection capabilities
///
/// # Examples
///
/// ```rust,ignore
/// use sklears_core::plugin::{MockPlugin, PluginCategory, PluginCapability};
/// use std::any::TypeId;
///
/// // Create a basic mock plugin
/// let mut mock = MockPlugin::new("test_plugin");
/// mock.metadata.category = PluginCategory::Algorithm;
/// mock.metadata.capabilities.push(PluginCapability::Parallel);
///
/// // Add supported types
/// mock.add_supported_type(TypeId::of::<f64>());
///
/// // Configure error behavior
/// mock.set_initialization_error(Some("Test error"));
///
/// // Use in tests
/// assert_eq!(mock.id(), "test_plugin");
/// assert!(mock.is_compatible(TypeId::of::<f64>()));
/// ```
#[derive(Debug, Clone)]
pub struct MockPlugin {
    /// Plugin identifier
    pub id: String,
    /// Plugin metadata
    pub metadata: PluginMetadata,
    /// Configuration for the plugin
    pub config: Option<PluginConfig>,
    /// Whether plugin is initialized
    pub initialized: bool,
    /// Simulated initialization error
    pub initialization_error: Option<String>,
    /// Simulated validation error
    pub validation_error: Option<String>,
    /// Simulated cleanup error
    pub cleanup_error: Option<String>,
    /// Call counters for testing
    pub call_counts: HashMap<String, usize>,
    /// Artificial delays for performance testing
    pub artificial_delays: HashMap<String, Duration>,
}

impl MockPlugin {
    /// Create a new mock plugin with default settings
    ///
    /// # Arguments
    ///
    /// * `id` - The plugin identifier
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use sklears_core::plugin::MockPlugin;
    ///
    /// let mock = MockPlugin::new("test_algorithm");
    /// assert_eq!(mock.id(), "test_algorithm");
    /// ```
    pub fn new(id: &str) -> Self {
        Self {
            id: id.to_string(),
            metadata: PluginMetadata {
                name: format!("Mock Plugin {}", id),
                version: "1.0.0".to_string(),
                description: "A mock plugin for testing".to_string(),
                author: "Test Framework".to_string(),
                category: PluginCategory::Algorithm,
                supported_types: vec![TypeId::of::<f64>(), TypeId::of::<f32>()],
                dependencies: Vec::new(),
                capabilities: vec![PluginCapability::Parallel],
                min_sdk_version: "0.1.0".to_string(),
            },
            config: None,
            initialized: false,
            initialization_error: None,
            validation_error: None,
            cleanup_error: None,
            call_counts: HashMap::new(),
            artificial_delays: HashMap::new(),
        }
    }

    /// Create a mock plugin for a specific algorithm category
    ///
    /// # Arguments
    ///
    /// * `id` - The plugin identifier
    /// * `category` - The plugin category
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use sklears_core::plugin::{MockPlugin, PluginCategory};
    ///
    /// let transformer = MockPlugin::for_category("scaler", PluginCategory::Transformer);
    /// assert_eq!(transformer.metadata.category, PluginCategory::Transformer);
    /// ```
    pub fn for_category(id: &str, category: PluginCategory) -> Self {
        let mut mock = Self::new(id);
        mock.metadata.category = category.clone();
        mock.metadata.name = format!("Mock {} Plugin", Self::category_name(&category));
        mock
    }

    /// Add a supported type to the plugin
    ///
    /// # Arguments
    ///
    /// * `type_id` - The TypeId to add as supported
    pub fn add_supported_type(&mut self, type_id: TypeId) {
        if !self.metadata.supported_types.contains(&type_id) {
            self.metadata.supported_types.push(type_id);
        }
    }

    /// Remove a supported type from the plugin
    ///
    /// # Arguments
    ///
    /// * `type_id` - The TypeId to remove
    pub fn remove_supported_type(&mut self, type_id: TypeId) {
        self.metadata.supported_types.retain(|&t| t != type_id);
    }

    /// Set an initialization error to simulate failure
    ///
    /// # Arguments
    ///
    /// * `error` - Optional error message (None to clear)
    pub fn set_initialization_error(&mut self, error: Option<&str>) {
        self.initialization_error = error.map(|s| s.to_string());
    }

    /// Set a validation error to simulate failure
    ///
    /// # Arguments
    ///
    /// * `error` - Optional error message (None to clear)
    pub fn set_validation_error(&mut self, error: Option<&str>) {
        self.validation_error = error.map(|s| s.to_string());
    }

    /// Set a cleanup error to simulate failure
    ///
    /// # Arguments
    ///
    /// * `error` - Optional error message (None to clear)
    pub fn set_cleanup_error(&mut self, error: Option<&str>) {
        self.cleanup_error = error.map(|s| s.to_string());
    }

    /// Add an artificial delay for a specific method
    ///
    /// # Arguments
    ///
    /// * `method` - The method name
    /// * `delay` - The delay duration
    pub fn add_artificial_delay(&mut self, method: &str, delay: Duration) {
        self.artificial_delays.insert(method.to_string(), delay);
    }

    /// Get the call count for a specific method
    ///
    /// # Arguments
    ///
    /// * `method` - The method name
    ///
    /// # Returns
    ///
    /// The number of times the method was called
    pub fn get_call_count(&self, method: &str) -> usize {
        self.call_counts.get(method).copied().unwrap_or(0)
    }

    /// Reset all call counts
    pub fn reset_call_counts(&mut self) {
        self.call_counts.clear();
    }

    /// Increment call count and apply artificial delay
    fn record_call(&mut self, method: &str) {
        *self.call_counts.entry(method.to_string()).or_insert(0) += 1;

        if let Some(delay) = self.artificial_delays.get(method) {
            std::thread::sleep(*delay);
        }
    }

    /// Get category name as string
    fn category_name(category: &PluginCategory) -> &str {
        match category {
            PluginCategory::Algorithm => "Algorithm",
            PluginCategory::Transformer => "Transformer",
            PluginCategory::DataProcessor => "DataProcessor",
            PluginCategory::Evaluator => "Evaluator",
            PluginCategory::Visualizer => "Visualizer",
            PluginCategory::Custom(name) => name,
        }
    }
}

impl Plugin for MockPlugin {
    fn id(&self) -> &str {
        &self.id
    }

    fn metadata(&self) -> PluginMetadata {
        self.metadata.clone()
    }

    fn initialize(&mut self, config: &PluginConfig) -> Result<()> {
        self.record_call("initialize");

        if let Some(ref error) = self.initialization_error {
            return Err(crate::error::SklearsError::InvalidOperation(error.clone()));
        }

        self.config = Some(config.clone());
        self.initialized = true;
        Ok(())
    }

    fn is_compatible(&self, input_type: TypeId) -> bool {
        self.metadata.supported_types.contains(&input_type)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn validate_config(&self, _config: &PluginConfig) -> Result<()> {
        if let Some(ref error) = self.validation_error {
            return Err(crate::error::SklearsError::InvalidOperation(error.clone()));
        }
        Ok(())
    }

    fn cleanup(&mut self) -> Result<()> {
        self.record_call("cleanup");

        if let Some(ref error) = self.cleanup_error {
            return Err(crate::error::SklearsError::InvalidOperation(error.clone()));
        }

        self.initialized = false;
        self.config = None;
        Ok(())
    }
}

/// Plugin test fixture for comprehensive testing scenarios
///
/// PluginTestFixture provides a complete testing environment with pre-configured
/// plugins, validation scenarios, and test data for thorough plugin testing.
///
/// # Examples
///
/// ```rust,ignore
/// use sklears_core::plugin::PluginTestFixture;
///
/// let fixture = PluginTestFixture::new();
/// let plugins = fixture.create_test_plugins();
/// assert!(!plugins.is_empty());
///
/// let manifests = fixture.create_test_manifests();
/// assert!(!manifests.is_empty());
/// ```
#[derive(Debug)]
pub struct PluginTestFixture {
    /// Security policy for testing
    pub security_policy: SecurityPolicy,
    /// Test plugins
    pub plugins: Vec<Box<dyn Plugin>>,
    /// Test manifests
    pub manifests: Vec<PluginManifest>,
}

impl PluginTestFixture {
    /// Create a new test fixture
    pub fn new() -> Self {
        Self {
            security_policy: SecurityPolicy::permissive(),
            plugins: Vec::new(),
            manifests: Vec::new(),
        }
    }

    /// Create test fixture with strict security policy
    pub fn with_strict_security() -> Self {
        Self {
            security_policy: SecurityPolicy::strict(),
            plugins: Vec::new(),
            manifests: Vec::new(),
        }
    }

    /// Create a set of test plugins covering various scenarios
    ///
    /// # Returns
    ///
    /// Vector of test plugins with different configurations and behaviors.
    pub fn create_test_plugins(&self) -> Vec<Box<dyn Plugin>> {
        vec![
            Box::new(MockPlugin::for_category(
                "linear_regression",
                PluginCategory::Algorithm,
            )),
            Box::new(MockPlugin::for_category(
                "standard_scaler",
                PluginCategory::Transformer,
            )),
            Box::new(MockPlugin::for_category(
                "csv_loader",
                PluginCategory::DataProcessor,
            )),
            Box::new(MockPlugin::for_category(
                "accuracy_metric",
                PluginCategory::Evaluator,
            )),
            Box::new(MockPlugin::for_category(
                "plot_generator",
                PluginCategory::Visualizer,
            )),
        ]
    }

    /// Create test manifests for validation testing
    ///
    /// # Returns
    ///
    /// Vector of plugin manifests with various security profiles and configurations.
    pub fn create_test_manifests(&self) -> Vec<PluginManifest> {
        vec![
            self.create_safe_manifest(),
            self.create_risky_manifest(),
            self.create_invalid_manifest(),
            self.create_signed_manifest(),
        ]
    }

    /// Create a safe plugin manifest for testing
    fn create_safe_manifest(&self) -> PluginManifest {
        PluginManifest {
            metadata: PluginMetadata {
                name: "SafePlugin".to_string(),
                version: "1.0.0".to_string(),
                description: "A safe test plugin".to_string(),
                author: "Test Suite".to_string(),
                category: PluginCategory::Algorithm,
                supported_types: vec![TypeId::of::<f64>()],
                dependencies: Vec::new(),
                capabilities: vec![PluginCapability::Parallel],
                min_sdk_version: "0.1.0".to_string(),
            },
            permissions: vec![Permission::FileSystemRead, Permission::GpuAccess],
            api_usage: None,
            contains_unsafe_code: false,
            dependencies: Vec::new(),
            code_analysis: None,
            signature: None,
            content_hash: "safe_hash_123".to_string(),
            publisher: PublisherInfo {
                name: "Trusted Publisher".to_string(),
                email: "trusted@example.com".to_string(),
                website: Some("https://trusted.example.com".to_string()),
                verified: true,
                trust_score: 9,
            },
            marketplace: super::validation::MarketplaceInfo {
                url: "https://marketplace.example.com/safe-plugin".to_string(),
                downloads: 1000,
                rating: 4.5,
                reviews: 50,
                last_updated: "2024-01-15".to_string(),
            },
        }
    }

    /// Create a risky plugin manifest for testing security validation
    fn create_risky_manifest(&self) -> PluginManifest {
        PluginManifest {
            metadata: PluginMetadata {
                name: "RiskyPlugin".to_string(),
                version: "1.0.0".to_string(),
                description: "A risky test plugin".to_string(),
                author: "Unknown".to_string(),
                category: PluginCategory::Algorithm,
                supported_types: vec![TypeId::of::<f64>()],
                dependencies: Vec::new(),
                capabilities: Vec::new(),
                min_sdk_version: "0.1.0".to_string(),
            },
            permissions: vec![
                Permission::FileSystemWrite,
                Permission::NetworkAccess,
                Permission::SystemCommands,
            ],
            api_usage: Some(super::validation::ApiUsageInfo {
                calls: vec!["std::process::Command".to_string()],
                network_access: vec!["http://api.example.com".to_string()],
                filesystem_access: vec!["/tmp/".to_string()],
            }),
            contains_unsafe_code: true,
            dependencies: Vec::new(),
            code_analysis: Some(super::validation::CodeAnalysisInfo {
                cyclomatic_complexity: 25,
                suspicious_patterns: vec!["eval".to_string()],
                potential_memory_issues: 2,
                lines_of_code: 1500,
                test_coverage: 45.0,
            }),
            signature: None,
            content_hash: "risky_hash_456".to_string(),
            publisher: PublisherInfo {
                name: "Unknown Publisher".to_string(),
                email: "unknown@example.com".to_string(),
                website: None,
                verified: false,
                trust_score: 2,
            },
            marketplace: super::validation::MarketplaceInfo {
                url: "https://marketplace.example.com/risky-plugin".to_string(),
                downloads: 10,
                rating: 2.0,
                reviews: 5,
                last_updated: "2023-06-01".to_string(),
            },
        }
    }

    /// Create an invalid plugin manifest for testing error handling
    fn create_invalid_manifest(&self) -> PluginManifest {
        PluginManifest {
            metadata: PluginMetadata {
                name: "".to_string(),           // Invalid empty name
                version: "invalid".to_string(), // Invalid version format
                description: "Invalid plugin".to_string(),
                author: "".to_string(), // Invalid empty author
                category: PluginCategory::Algorithm,
                supported_types: Vec::new(),
                dependencies: Vec::new(),
                capabilities: Vec::new(),
                min_sdk_version: "999.0.0".to_string(), // Invalid high version
            },
            permissions: vec![Permission::Custom("invalid_permission".to_string())],
            api_usage: None,
            contains_unsafe_code: true,
            dependencies: Vec::new(),
            code_analysis: None,
            signature: None,
            content_hash: "".to_string(), // Invalid empty hash
            publisher: PublisherInfo {
                name: "".to_string(),
                email: "invalid-email".to_string(), // Invalid email format
                website: None,
                verified: false,
                trust_score: 15, // Invalid trust score > 10
            },
            marketplace: super::validation::MarketplaceInfo {
                url: "".to_string(),
                downloads: 0,
                rating: 0.0,
                reviews: 0,
                last_updated: "".to_string(),
            },
        }
    }

    /// Create a signed plugin manifest for testing signature validation
    fn create_signed_manifest(&self) -> PluginManifest {
        let mut manifest = self.create_safe_manifest();
        manifest.signature = Some(DigitalSignature {
            algorithm: "RSA-SHA256".to_string(),
            signature: vec![0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0],
            public_key_fingerprint: "SHA256:test_fingerprint_123".to_string(),
            timestamp: SystemTime::now(),
            signer_certificate: Some(
                "-----BEGIN CERTIFICATE-----\ntest\n-----END CERTIFICATE-----".to_string(),
            ),
        });
        manifest
    }

    /// Create test plugin configurations
    ///
    /// # Returns
    ///
    /// Vector of plugin configurations for testing various scenarios.
    pub fn create_test_configs(&self) -> Vec<PluginConfig> {
        vec![
            self.create_minimal_config(),
            self.create_full_config(),
            self.create_invalid_config(),
        ]
    }

    /// Create a minimal plugin configuration
    fn create_minimal_config(&self) -> PluginConfig {
        PluginConfig::default()
    }

    /// Create a comprehensive plugin configuration
    fn create_full_config(&self) -> PluginConfig {
        let mut config = PluginConfig::default();

        // Add various parameter types
        config
            .parameters
            .insert("learning_rate".to_string(), PluginParameter::Float(0.01));
        config
            .parameters
            .insert("max_iterations".to_string(), PluginParameter::Int(1000));
        config
            .parameters
            .insert("use_bias".to_string(), PluginParameter::Bool(true));
        config.parameters.insert(
            "algorithm".to_string(),
            PluginParameter::String("adam".to_string()),
        );
        config.parameters.insert(
            "layer_sizes".to_string(),
            PluginParameter::IntArray(vec![100, 50, 10]),
        );
        config.parameters.insert(
            "dropout_rates".to_string(),
            PluginParameter::FloatArray(vec![0.2, 0.3, 0.5]),
        );

        // Runtime settings
        config.runtime_settings.num_threads = Some(4);
        config.runtime_settings.memory_limit = Some(1024 * 1024 * 1024); // 1GB
        config.runtime_settings.timeout_ms = Some(30000); // 30 seconds
        config.runtime_settings.use_gpu = true;

        // Plugin-specific settings
        config
            .plugin_settings
            .insert("backend".to_string(), "cuda".to_string());
        config
            .plugin_settings
            .insert("precision".to_string(), "float32".to_string());

        config
    }

    /// Create an invalid plugin configuration for testing error handling
    fn create_invalid_config(&self) -> PluginConfig {
        let mut config = PluginConfig::default();

        // Invalid runtime settings
        config.runtime_settings.num_threads = Some(0); // Invalid: zero threads
        config.runtime_settings.timeout_ms = Some(0); // Invalid: zero timeout
        config.runtime_settings.memory_limit = Some(0); // Invalid: zero memory

        config
    }
}

impl Default for PluginTestFixture {
    fn default() -> Self {
        Self::new()
    }
}

/// Performance testing utilities for plugins
///
/// PluginPerformanceTester provides tools for measuring plugin performance,
/// memory usage, and identifying bottlenecks in plugin implementations.
///
/// # Examples
///
/// ```rust,ignore
/// use sklears_core::plugin::{PluginPerformanceTester, MockPlugin};
///
/// let mut tester = PluginPerformanceTester::new();
/// let mut plugin = MockPlugin::new("test_plugin");
///
/// let result = tester.benchmark_initialization(&mut plugin);
/// println!("Initialization took: {:?}", result.duration);
/// ```
#[derive(Debug)]
pub struct PluginPerformanceTester {
    /// Test configurations for benchmarking
    pub test_configs: Vec<PluginConfig>,
    /// Maximum allowed duration for operations
    pub max_duration: Duration,
    /// Memory usage baseline
    pub memory_baseline: usize,
}

impl PluginPerformanceTester {
    /// Create a new performance tester
    pub fn new() -> Self {
        Self {
            test_configs: Vec::new(),
            max_duration: Duration::from_secs(10),
            memory_baseline: 0,
        }
    }

    /// Create performance tester with custom timeout
    ///
    /// # Arguments
    ///
    /// * `max_duration` - Maximum allowed duration for operations
    pub fn with_timeout(max_duration: Duration) -> Self {
        Self {
            test_configs: Vec::new(),
            max_duration,
            memory_baseline: 0,
        }
    }

    /// Benchmark plugin initialization
    ///
    /// # Arguments
    ///
    /// * `plugin` - The plugin to benchmark
    ///
    /// # Returns
    ///
    /// Performance metrics for the initialization operation.
    pub fn benchmark_initialization(&mut self, plugin: &mut dyn Plugin) -> PerformanceResult {
        let config = PluginConfig::default();
        let start = Instant::now();
        let memory_start = self.get_memory_usage();

        let result = plugin.initialize(&config);

        let duration = start.elapsed();
        let memory_end = self.get_memory_usage();
        let memory_used = memory_end.saturating_sub(memory_start);

        PerformanceResult {
            operation: "initialization".to_string(),
            duration,
            memory_used,
            success: result.is_ok(),
            error: result.err().map(|e| e.to_string()),
            within_limits: duration <= self.max_duration,
        }
    }

    /// Benchmark plugin validation
    ///
    /// # Arguments
    ///
    /// * `plugin` - The plugin to benchmark
    /// * `config` - The configuration to validate
    ///
    /// # Returns
    ///
    /// Performance metrics for the validation operation.
    pub fn benchmark_validation(
        &self,
        plugin: &dyn Plugin,
        config: &PluginConfig,
    ) -> PerformanceResult {
        let start = Instant::now();
        let memory_start = self.get_memory_usage();

        let result = plugin.validate_config(config);

        let duration = start.elapsed();
        let memory_end = self.get_memory_usage();
        let memory_used = memory_end.saturating_sub(memory_start);

        PerformanceResult {
            operation: "validation".to_string(),
            duration,
            memory_used,
            success: result.is_ok(),
            error: result.err().map(|e| e.to_string()),
            within_limits: duration <= self.max_duration,
        }
    }

    /// Run a comprehensive performance test suite
    ///
    /// # Arguments
    ///
    /// * `plugin` - The plugin to test
    ///
    /// # Returns
    ///
    /// Vector of performance results for all operations.
    pub fn run_performance_suite(&mut self, plugin: &mut dyn Plugin) -> Vec<PerformanceResult> {
        let mut results = Vec::new();

        // Test initialization
        results.push(self.benchmark_initialization(plugin));

        // Test validation with different configs
        let fixture = PluginTestFixture::new();
        for config in fixture.create_test_configs() {
            results.push(self.benchmark_validation(plugin, &config));
        }

        // Test cleanup
        results.push(self.benchmark_cleanup(plugin));

        results
    }

    /// Benchmark plugin cleanup
    fn benchmark_cleanup(&self, plugin: &mut dyn Plugin) -> PerformanceResult {
        let start = Instant::now();
        let memory_start = self.get_memory_usage();

        let result = plugin.cleanup();

        let duration = start.elapsed();
        let memory_end = self.get_memory_usage();
        let memory_used = memory_end.saturating_sub(memory_start);

        PerformanceResult {
            operation: "cleanup".to_string(),
            duration,
            memory_used,
            success: result.is_ok(),
            error: result.err().map(|e| e.to_string()),
            within_limits: duration <= self.max_duration,
        }
    }

    /// Get current memory usage (placeholder implementation)
    fn get_memory_usage(&self) -> usize {
        // In a real implementation, this would measure actual memory usage
        // For now, return a placeholder value
        1024 * 1024 // 1MB placeholder
    }
}

impl Default for PluginPerformanceTester {
    fn default() -> Self {
        Self::new()
    }
}

/// Performance measurement result
#[derive(Debug, Clone)]
pub struct PerformanceResult {
    /// Operation name
    pub operation: String,
    /// Time taken for the operation
    pub duration: Duration,
    /// Memory used during operation
    pub memory_used: usize,
    /// Whether the operation succeeded
    pub success: bool,
    /// Error message if operation failed
    pub error: Option<String>,
    /// Whether the operation completed within time limits
    pub within_limits: bool,
}

impl PerformanceResult {
    /// Check if the performance result indicates good performance
    pub fn is_good_performance(&self) -> bool {
        self.success && self.within_limits
    }

    /// Get a performance score (0-100)
    pub fn performance_score(&self) -> u8 {
        if !self.success {
            return 0;
        }

        let time_score = if self.within_limits { 50 } else { 0 };
        let memory_score = if self.memory_used < 10 * 1024 * 1024 {
            50
        } else {
            25
        }; // < 10MB is good

        (time_score + memory_score).min(100)
    }
}

/// Validation test runner for comprehensive plugin testing
///
/// ValidationTestRunner executes a complete test suite including security
/// validation, performance testing, and compatibility checks.
///
/// # Examples
///
/// ```rust,ignore
/// use sklears_core::plugin::{ValidationTestRunner, MockPlugin};
///
/// let runner = ValidationTestRunner::new();
/// let plugin = MockPlugin::new("test_plugin");
/// let fixture = runner.create_test_fixture();
///
/// let report = runner.run_validation_tests(&plugin, &fixture.create_test_manifests()[0]);
/// println!("Validation passed: {}", !report.has_errors());
/// ```
#[derive(Debug)]
pub struct ValidationTestRunner {
    /// Plugin validator
    validator: PluginValidator,
    /// Performance tester
    performance_tester: PluginPerformanceTester,
    /// Test fixture
    fixture: PluginTestFixture,
}

impl ValidationTestRunner {
    /// Create a new validation test runner
    pub fn new() -> Self {
        Self {
            validator: PluginValidator::new(),
            performance_tester: PluginPerformanceTester::new(),
            fixture: PluginTestFixture::new(),
        }
    }

    /// Create test runner with strict security
    pub fn with_strict_security() -> Self {
        Self {
            validator: PluginValidator::new(),
            performance_tester: PluginPerformanceTester::new(),
            fixture: PluginTestFixture::with_strict_security(),
        }
    }

    /// Run comprehensive validation tests
    ///
    /// # Arguments
    ///
    /// * `plugin` - The plugin to test
    /// * `manifest` - The plugin manifest to validate
    ///
    /// # Returns
    ///
    /// Comprehensive validation report.
    pub fn run_validation_tests(
        &self,
        plugin: &dyn Plugin,
        manifest: &PluginManifest,
    ) -> ValidationReport {
        self.validator
            .validate_comprehensive(plugin, manifest)
            .unwrap_or_else(|_| {
                let mut report = ValidationReport::new();
                report.add_error(super::validation::ValidationError::InvalidMetadata(
                    "Failed to run validation tests".to_string(),
                ));
                report
            })
    }

    /// Run performance tests
    ///
    /// # Arguments
    ///
    /// * `plugin` - The plugin to test
    ///
    /// # Returns
    ///
    /// Vector of performance results.
    pub fn run_performance_tests(&mut self, plugin: &mut dyn Plugin) -> Vec<PerformanceResult> {
        self.performance_tester.run_performance_suite(plugin)
    }

    /// Run compatibility tests
    ///
    /// # Arguments
    ///
    /// * `plugin` - The plugin to test
    ///
    /// # Returns
    ///
    /// Compatibility test results.
    pub fn run_compatibility_tests(&self, plugin: &dyn Plugin) -> CompatibilityTestResult {
        let mut supported_types = Vec::new();
        let mut unsupported_types = Vec::new();

        // Test common types
        let test_types = vec![
            TypeId::of::<f32>(),
            TypeId::of::<f64>(),
            TypeId::of::<i32>(),
            TypeId::of::<i64>(),
            TypeId::of::<String>(),
        ];

        for type_id in test_types {
            if plugin.is_compatible(type_id) {
                supported_types.push(type_id);
            } else {
                unsupported_types.push(type_id);
            }
        }

        let compatibility_score = (supported_types.len() as f32 / 5.0 * 100.0) as u8;

        CompatibilityTestResult {
            supported_types,
            unsupported_types,
            total_types_tested: 5,
            compatibility_score,
        }
    }

    /// Create a test fixture for the runner
    pub fn create_test_fixture(&self) -> &PluginTestFixture {
        &self.fixture
    }

    /// Run complete test suite
    ///
    /// # Arguments
    ///
    /// * `plugin` - The plugin to test
    /// * `manifest` - The plugin manifest
    ///
    /// # Returns
    ///
    /// Complete test results.
    pub fn run_complete_test_suite(
        &mut self,
        plugin: &mut dyn Plugin,
        manifest: &PluginManifest,
    ) -> CompleteTestResult {
        let validation_report = self.run_validation_tests(plugin, manifest);
        let performance_results = self.run_performance_tests(plugin);
        let compatibility_result = self.run_compatibility_tests(plugin);

        let overall_score = self.calculate_overall_score(
            &validation_report,
            &performance_results,
            &compatibility_result,
        );

        CompleteTestResult {
            validation_report,
            performance_results,
            compatibility_result,
            overall_score,
            test_passed: overall_score >= 70, // 70% minimum score to pass
        }
    }

    /// Calculate overall test score
    fn calculate_overall_score(
        &self,
        validation: &ValidationReport,
        performance: &[PerformanceResult],
        compatibility: &CompatibilityTestResult,
    ) -> u8 {
        let validation_score = if validation.has_errors() { 0 } else { 40 };

        let avg_performance = if performance.is_empty() {
            0
        } else {
            performance
                .iter()
                .map(|r| r.performance_score() as u32)
                .sum::<u32>()
                / performance.len() as u32
        };
        let performance_score = (avg_performance as f32 * 0.3) as u8;

        let compatibility_score = (compatibility.compatibility_score as f32 * 0.3) as u8;

        (validation_score + performance_score + compatibility_score).min(100)
    }
}

impl Default for ValidationTestRunner {
    fn default() -> Self {
        Self::new()
    }
}

/// Compatibility test result
#[derive(Debug, Clone)]
pub struct CompatibilityTestResult {
    /// Types that the plugin supports
    pub supported_types: Vec<TypeId>,
    /// Types that the plugin doesn't support
    pub unsupported_types: Vec<TypeId>,
    /// Total number of types tested
    pub total_types_tested: usize,
    /// Compatibility score (0-100)
    pub compatibility_score: u8,
}

/// Complete test result containing all test outcomes
#[derive(Debug)]
pub struct CompleteTestResult {
    /// Validation test results
    pub validation_report: ValidationReport,
    /// Performance test results
    pub performance_results: Vec<PerformanceResult>,
    /// Compatibility test results
    pub compatibility_result: CompatibilityTestResult,
    /// Overall test score (0-100)
    pub overall_score: u8,
    /// Whether the plugin passed all tests
    pub test_passed: bool,
}

impl CompleteTestResult {
    /// Get a summary of the test results
    pub fn summary(&self) -> String {
        format!(
            "Plugin Test Summary:\n\
             - Validation: {} errors, {} warnings\n\
             - Performance: {}/{} operations within limits\n\
             - Compatibility: {:.1}% type support\n\
             - Overall Score: {}/100\n\
             - Result: {}",
            self.validation_report.errors.len(),
            self.validation_report.warnings.len(),
            self.performance_results
                .iter()
                .filter(|r| r.within_limits)
                .count(),
            self.performance_results.len(),
            self.compatibility_result.compatibility_score,
            self.overall_score,
            if self.test_passed { "PASSED" } else { "FAILED" }
        )
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mock_plugin_creation() {
        let mock = MockPlugin::new("test_plugin");
        assert_eq!(mock.id(), "test_plugin");
        assert_eq!(mock.metadata.name, "Mock Plugin test_plugin");
        assert!(!mock.initialized);
    }

    #[test]
    fn test_mock_plugin_categories() {
        let algo = MockPlugin::for_category("test", PluginCategory::Algorithm);
        assert_eq!(algo.metadata.category, PluginCategory::Algorithm);

        let transformer = MockPlugin::for_category("test", PluginCategory::Transformer);
        assert_eq!(transformer.metadata.category, PluginCategory::Transformer);
    }

    #[test]
    fn test_mock_plugin_type_support() {
        let mut mock = MockPlugin::new("test");

        // Should support f64 by default
        assert!(mock.is_compatible(TypeId::of::<f64>()));

        // Add i32 support
        mock.add_supported_type(TypeId::of::<i32>());
        assert!(mock.is_compatible(TypeId::of::<i32>()));

        // Remove f64 support
        mock.remove_supported_type(TypeId::of::<f64>());
        assert!(!mock.is_compatible(TypeId::of::<f64>()));
    }

    #[test]
    fn test_mock_plugin_error_simulation() {
        let mut mock = MockPlugin::new("test");

        // Set initialization error
        mock.set_initialization_error(Some("Test error"));
        let result = mock.initialize(&PluginConfig::default());
        assert!(result.is_err());

        // Clear error
        mock.set_initialization_error(None);
        let result = mock.initialize(&PluginConfig::default());
        assert!(result.is_ok());
    }

    #[test]
    fn test_plugin_test_fixture() {
        let fixture = PluginTestFixture::new();

        let plugins = fixture.create_test_plugins();
        assert_eq!(plugins.len(), 5);

        let manifests = fixture.create_test_manifests();
        assert_eq!(manifests.len(), 4);

        let configs = fixture.create_test_configs();
        assert_eq!(configs.len(), 3);
    }

    #[test]
    fn test_performance_tester() {
        let mut tester = PluginPerformanceTester::new();
        let mut mock = MockPlugin::new("test");

        let result = tester.benchmark_initialization(&mut mock);
        assert_eq!(result.operation, "initialization");
        assert!(result.success);
    }

    #[test]
    fn test_validation_test_runner() {
        let runner = ValidationTestRunner::new();
        let mock = MockPlugin::new("test");
        let fixture = runner.create_test_fixture();
        let manifest = &fixture.create_test_manifests()[3]; // Signed manifest for validation

        let report = runner.run_validation_tests(&mock, manifest);
        assert!(!report.has_errors()); // Should pass validation
    }

    #[test]
    fn test_compatibility_tests() {
        let runner = ValidationTestRunner::new();
        let mock = MockPlugin::new("test"); // Supports f64 and f32 by default

        let result = runner.run_compatibility_tests(&mock);
        assert!(!result.supported_types.is_empty());
        assert_eq!(result.total_types_tested, 5);
        assert!(result.compatibility_score > 0);
    }

    #[test]
    fn test_performance_result_scoring() {
        let good_result = PerformanceResult {
            operation: "test".to_string(),
            duration: Duration::from_millis(10),
            memory_used: 1024, // 1KB
            success: true,
            error: None,
            within_limits: true,
        };
        assert!(good_result.is_good_performance());
        assert_eq!(good_result.performance_score(), 100);

        let bad_result = PerformanceResult {
            operation: "test".to_string(),
            duration: Duration::from_secs(20),
            memory_used: 50 * 1024 * 1024, // 50MB
            success: false,
            error: Some("Error".to_string()),
            within_limits: false,
        };
        assert!(!bad_result.is_good_performance());
        assert_eq!(bad_result.performance_score(), 0);
    }
}
