//! Plugin System Module
//!
//! This module provides a comprehensive plugin architecture for the SKLears framework,
//! enabling dynamic loading, validation, and management of machine learning algorithms
//! and data processing components.
//!
//! # Plugin System Architecture
//!
//! The plugin system is organized into several key components:
//!
//! ## Core Components
//!
//! - **[Core Traits](core_traits)**: Foundation traits that all plugins must implement
//! - **[Types & Configuration](types_config)**: Type-safe parameter handling and plugin metadata
//! - **[Registry](registry)**: Thread-safe plugin registration and discovery
//! - **[Loader](loader)**: Dynamic library loading with cross-platform support
//!
//! ## Advanced Features
//!
//! - **[Factory & Builder](factory_builder)**: Factory patterns for plugin creation
//! - **[Validation](validation)**: Comprehensive security and quality validation
//! - **[Security](security)**: Permission management and digital signatures
//! - **[Discovery & Marketplace](discovery_marketplace)**: Remote plugin discovery and community features
//! - **[Testing Utilities](testing_utils)**: Complete testing framework for plugin development
//!
//! # Quick Start
//!
//! ## Creating a Simple Plugin
//!
//! ```rust,ignore
//! use sklears_core::plugin::{Plugin, PluginMetadata, PluginConfig, PluginCategory};
//! use sklears_core::error::Result;
//! use std::any::{Any, TypeId};
//!
//! #[derive(Debug)]
//! struct MyAlgorithm {
//!     name: String,
//! }
//!
//! impl Plugin for MyAlgorithm {
//!     fn id(&self) -> &str {
//!         &self.name
//!     }
//!
//!     fn metadata(&self) -> PluginMetadata {
//!         PluginMetadata {
//!             name: "My Algorithm".to_string(),
//!             version: "1.0.0".to_string(),
//!             description: "An example machine learning algorithm".to_string(),
//!             author: "Your Name".to_string(),
//!             category: PluginCategory::Algorithm,
//!             supported_types: vec![TypeId::of::<f64>()],
//!             ..Default::default()
//!         }
//!     }
//!
//!     fn initialize(&mut self, _config: &PluginConfig) -> Result<()> {
//!         println!("Initializing {}", self.name);
//!         Ok(())
//!     }
//!
//!     fn is_compatible(&self, input_type: TypeId) -> bool {
//!         input_type == TypeId::of::<f64>()
//!     }
//!
//!     fn as_any(&self) -> &dyn Any { self }
//!     fn as_any_mut(&mut self) -> &mut dyn Any { self }
//!     fn validate_config(&self, _config: &PluginConfig) -> Result<()> { Ok(()) }
//!     fn cleanup(&mut self) -> Result<()> { Ok(()) }
//! }
//! ```
//!
//! ## Using the Plugin Registry
//!
//! ```rust,ignore
//! use sklears_core::plugin::{PluginRegistry, Plugin};
//!
//! # use sklears_core::plugin::{PluginMetadata, PluginConfig, PluginCategory};
//! # use sklears_core::error::Result;
//! # use std::any::{Any, TypeId};
//! # #[derive(Debug)]
//! # struct MyAlgorithm { name: String }
//! # impl Plugin for MyAlgorithm {
//! #     fn id(&self) -> &str { &self.name }
//! #     fn metadata(&self) -> PluginMetadata {
//! #         PluginMetadata {
//! #             name: "My Algorithm".to_string(),
//! #             version: "1.0.0".to_string(),
//! #             description: "An example algorithm".to_string(),
//! #             author: "Your Name".to_string(),
//! #             category: PluginCategory::Algorithm,
//! #             supported_types: vec![TypeId::of::<f64>()],
//! #             ..Default::default()
//! #         }
//! #     }
//! #     fn initialize(&mut self, _config: &PluginConfig) -> Result<()> { Ok(()) }
//! #     fn is_compatible(&self, input_type: TypeId) -> bool { input_type == TypeId::of::<f64>() }
//! #     fn as_any(&self) -> &dyn Any { self }
//! #     fn as_any_mut(&mut self) -> &mut dyn Any { self }
//! #     fn validate_config(&self, _config: &PluginConfig) -> Result<()> { Ok(()) }
//! #     fn cleanup(&mut self) -> Result<()> { Ok(()) }
//! # }
//! fn example() -> Result<(), Box<dyn std::error::Error>> {
//!     let registry = PluginRegistry::new();
//!
//!     // Register a plugin
//!     let plugin = MyAlgorithm { name: "my_algo".to_string() };
//!     registry.register("my_algo", Box::new(plugin))?;
//!
//!     // List available plugins
//!     let plugins = registry.list_plugins()?;
//!     println!("Available plugins: {:?}", plugins);
//!
//!     // Search for plugins
//!     let matches = registry.search_plugins("algorithm")?;
//!     println!("Found {} matching plugins", matches.len());
//!
//!     Ok(())
//! }
//! ```
//!
//! ## Building Plugin Configurations
//!
//! ```rust,ignore
//! use sklears_core::plugin::{PluginConfigBuilder, PluginParameter, LogLevel};
//!
//! let config = PluginConfigBuilder::new()
//!     .with_parameter("learning_rate", PluginParameter::Float(0.01))
//!     .with_parameter("max_iterations", PluginParameter::Int(1000))
//!     .with_parameter("use_bias", PluginParameter::Bool(true))
//!     .with_threads(4)
//!     .with_memory_limit(1024 * 1024 * 1024) // 1GB
//!     .with_gpu(true)
//!     .with_log_level(LogLevel::Info)
//!     .build();
//! ```
//!
//! ## Dynamic Library Loading
//!
//! ```rust,ignore
//! use sklears_core::plugin::{PluginLoader, PluginRegistry};
//! use std::sync::Arc;
//!
//! # #[cfg(feature = "dynamic_loading")]
//! fn example() -> Result<(), Box<dyn std::error::Error>> {
//!     let registry = Arc::new(PluginRegistry::new());
//!     let mut loader = PluginLoader::new(registry.clone());
//!
//!     // Load a single plugin
//!     loader.load_from_library("./plugins/my_plugin.so", "my_plugin")?;
//!
//!     // Load all plugins from a directory
//!     let loaded = loader.load_from_directory("./plugins/")?;
//!     println!("Loaded {} plugins", loaded.len());
//!
//!     Ok(())
//! }
//! ```
//!
//! ## Plugin Validation and Security
//!
//! ```rust,ignore
//! use sklears_core::plugin::{PluginValidator, SecurityPolicy, Permission};
//!
//! // Create a validator with custom security policy
//! let mut validator = PluginValidator::new();
//! let mut policy = SecurityPolicy::standard();
//! policy.add_dangerous_permission("network_access".to_string());
//!
//! // Validate plugins before loading
//! # /*
//! let validation_report = validator.validate_comprehensive(&plugin, &manifest)?;
//! if validation_report.has_errors() {
//!     println!("Plugin validation failed: {:?}", validation_report.errors);
//! }
//! # */
//! ```
//!
//! ## Plugin Discovery and Marketplace
//!
//! ```rust,ignore
//! use sklears_core::plugin::{PluginDiscoveryService, PluginMarketplace, SearchQuery, PluginCategory};
//!
//! async fn example() -> Result<(), Box<dyn std::error::Error>> {
//!     let marketplace = PluginMarketplace::new();
//!
//!     // Get featured plugins
//!     let featured = marketplace.get_featured_plugins().await?;
//!     println!("Featured plugins: {}", featured.len());
//!
//!     // Search for plugins
//!     let discovery = PluginDiscoveryService::new();
//!     let query = SearchQuery {
//!         text: "classification".to_string(),
//!         category: Some(PluginCategory::Algorithm),
//!         capabilities: vec![],
//!         limit: Some(10),
//!         min_rating: Some(4.0),
//!     };
//!     let results = discovery.search(&query).await?;
//!
//!     // Install a plugin
//!     if let Some(result) = results.first() {
//!         let install_result = discovery.install_plugin(&result.plugin_id, None).await?;
//!         println!("Installed plugin at: {}", install_result.install_path);
//!     }
//!
//!     Ok(())
//! }
//! ```
//!
//! # Testing Plugins
//!
//! The plugin system includes comprehensive testing utilities:
//!
//! ```rust,ignore
//! use sklears_core::plugin::{MockPlugin, ValidationTestRunner, PluginTestFixture};
//!
//! // Create mock plugins for testing
//! let mut mock = MockPlugin::new("test_plugin");
//! mock.set_initialization_error(Some("Test error"));
//!
//! // Run comprehensive validation tests
//! let runner = ValidationTestRunner::new();
//! let fixture = PluginTestFixture::new();
//! let manifest = &fixture.create_test_manifests()[0];
//!
//! # /*
//! let report = runner.run_validation_tests(&mock, manifest);
//! println!("Validation passed: {}", !report.has_errors());
//! # */
//! ```

// Core plugin functionality
pub mod core_traits;
pub mod loader;
pub mod registry;
pub mod types_config;

// Advanced plugin features
pub mod discovery_marketplace;
pub mod factory_builder;
pub mod security;
pub mod validation;

// Testing and development utilities
pub mod testing_utils;

// Re-export core traits and types for convenient access
pub use core_traits::{AlgorithmPlugin, ClusteringPlugin, Plugin, TransformerPlugin};

pub use types_config::{
    LogLevel, PluginCapability, PluginCategory, PluginConfig, PluginMetadata, PluginParameter,
    RuntimeSettings,
};

pub use registry::PluginRegistry;

pub use loader::PluginLoader;

// Factory and builder patterns
pub use factory_builder::{PluginConfigBuilder, PluginFactory};

// Validation framework
pub use validation::{
    ApiUsageInfo, CodeAnalysisInfo, Dependency, PluginManifest, PluginValidator, ValidationCheck,
    ValidationError, ValidationReport, ValidationResult, ValidationWarning, Vulnerability,
};

// Security framework
pub use security::{
    CertificateInfo, DigitalSignature, Permission, PermissionSet, PublicKeyInfo, PublisherInfo,
    SecurityPolicy, TrustStore,
};

// Discovery and marketplace
pub use discovery_marketplace::{
    FeaturedPlugin, MarketplaceInfo, MarketplaceSummary, PluginDiscoveryService,
    PluginInstallResult, PluginMarketplace, PluginRepository, PluginReview, PluginSearchResult,
    PluginStats, PricingInfo, RepositoryStats, SearchQuery, SubscriptionPeriod, SupportLevel,
    TrendDirection, TrendingPlugin, UsageUnit,
};

// Testing utilities
pub use testing_utils::{
    CompatibilityTestResult, CompleteTestResult, MockPlugin, PerformanceResult,
    PluginPerformanceTester, PluginTestFixture, ValidationTestRunner,
};

// Convenience re-exports for common patterns
/// Common plugin development imports
pub mod prelude {
    pub use super::{
        AlgorithmPlugin, ClusteringPlugin, LogLevel, Plugin, PluginCapability, PluginCategory,
        PluginConfig, PluginConfigBuilder, PluginFactory, PluginMetadata, PluginParameter,
        PluginRegistry, TransformerPlugin,
    };
}

/// Security-focused imports for plugin validation
pub mod security_prelude {
    pub use super::{
        DigitalSignature, Permission, PermissionSet, PluginValidator, SecurityPolicy, TrustStore,
        ValidationReport,
    };
}

/// Marketplace and discovery imports
pub mod marketplace_prelude {
    pub use super::{
        FeaturedPlugin, PluginDiscoveryService, PluginMarketplace, PluginRepository, PluginStats,
        SearchQuery,
    };
}

/// Testing utilities imports
pub mod testing_prelude {
    pub use super::{
        CompleteTestResult, MockPlugin, PluginPerformanceTester, PluginTestFixture,
        ValidationTestRunner,
    };
}

// Convenience type aliases for common use cases
/// Type alias for a boxed plugin instance
pub type BoxedPlugin = Box<dyn Plugin>;

/// Type alias for plugin validation result
pub type PluginValidationResult = Result<ValidationReport, crate::error::SklearsError>;

/// Type alias for plugin creation function (for dynamic loading)
pub type PluginCreateFn = fn() -> BoxedPlugin;

// Module-level documentation and examples

/// # Plugin Development Guide
///
/// ## Creating Custom Plugins
///
/// To create a custom plugin, implement the `Plugin` trait and any relevant
/// specialized traits like `AlgorithmPlugin` or `TransformerPlugin`:
///
/// ```rust,ignore
/// use sklears_core::plugin::prelude::*;
/// use sklears_core::error::Result;
/// use std::any::{Any, TypeId};
/// use std::collections::HashMap;
///
/// #[derive(Debug)]
/// struct LinearRegression {
///     coefficients: Vec`<f64>`,
///     intercept: f64,
/// }
///
/// impl Plugin for LinearRegression {
///     fn id(&self) -> &str { "linear_regression" }
///
///     fn metadata(&self) -> PluginMetadata {
///         PluginMetadata {
///             name: "Linear Regression".to_string(),
///             version: "1.0.0".to_string(),
///             description: "Simple linear regression algorithm".to_string(),
///             author: "SKLears Team".to_string(),
///             category: PluginCategory::Algorithm,
///             supported_types: vec![TypeId::of::`<f64>`()],
///             capabilities: vec![PluginCapability::Parallel],
///             ..Default::default()
///         }
///     }
///
///     fn initialize(&mut self, config: &PluginConfig) -> Result<()> {
///         // Initialize algorithm with configuration
///         if let Some(param) = config.parameters.get("regularization") {
///             let reg_strength = param.as_float()?;
///             println!("Using regularization strength: {}", reg_strength);
///         }
///         Ok(())
///     }
///
///     fn is_compatible(&self, input_type: TypeId) -> bool {
///         input_type == TypeId::of::`<f64>`()
///     }
///
///     fn as_any(&self) -> &dyn Any { self }
///     fn as_any_mut(&mut self) -> &mut dyn Any { self }
///     fn validate_config(&self, _config: &PluginConfig) -> Result<()> { Ok(()) }
///     fn cleanup(&mut self) -> Result<()> { Ok(()) }
/// }
/// ```
///
/// ## Security Best Practices
///
/// When developing plugins, follow these security guidelines:
///
/// 1. **Minimize Permissions**: Only request permissions your plugin actually needs
/// 2. **Validate Inputs**: Always validate configuration parameters and input data
/// 3. **Handle Errors Gracefully**: Don't expose sensitive information in error messages
/// 4. **Use Safe APIs**: Avoid unsafe code blocks unless absolutely necessary
/// 5. **Sign Your Plugins**: Use digital signatures for distribution
///
/// ## Performance Considerations
///
/// - Implement efficient algorithms with O(n log n) or better complexity where possible
/// - Use SIMD operations for numerical computations when available
/// - Minimize memory allocations in hot paths
/// - Support parallel processing through the `Parallel` capability
/// - Use lazy initialization for expensive resources
///
/// ## Testing Your Plugins
///
/// Use the comprehensive testing framework:
///
/// ```rust,ignore
/// use sklears_core::plugin::testing_prelude::*;
///
/// #[test]
/// fn test_my_plugin() {
///     let mut mock = MockPlugin::new("test_plugin");
///     let runner = ValidationTestRunner::new();
///     let fixture = PluginTestFixture::new();
///
///     // Run comprehensive tests
///     let manifest = &fixture.create_test_manifests()[0];
///     let mut test_results = runner.run_complete_test_suite(&mut mock, manifest);
///
///     assert!(test_results.test_passed);
///     assert!(test_results.overall_score >= 80);
/// }
/// ```
#[allow(non_snake_case)]
#[cfg(test)]
mod integration_tests {
    use super::*;
    use std::any::TypeId;

    #[test]
    fn test_plugin_system_integration() {
        // Test basic plugin creation and registration
        let registry = PluginRegistry::new();
        let mock = testing_utils::MockPlugin::new("integration_test");

        assert!(registry
            .register("integration_test", Box::new(mock))
            .is_ok());

        let plugins = registry.list_plugins().unwrap();
        assert!(plugins.contains(&"integration_test".to_string()));
    }

    #[test]
    fn test_plugin_configuration_builder() {
        let config = PluginConfigBuilder::new()
            .with_parameter("test_param", PluginParameter::Float(1.0))
            .with_threads(2)
            .with_gpu(false)
            .build();

        assert_eq!(config.parameters.len(), 1);
        assert_eq!(config.runtime_settings.num_threads, Some(2));
        assert!(!config.runtime_settings.use_gpu);
    }

    #[test]
    fn test_security_policy_levels() {
        let strict = SecurityPolicy::strict();
        let standard = SecurityPolicy::standard();
        let permissive = SecurityPolicy::permissive();

        assert!(strict.require_signatures);
        assert!(standard.require_signatures);
        assert!(!permissive.require_signatures);

        assert!(!strict.allow_unsafe_code);
        assert!(!standard.allow_unsafe_code);
        assert!(permissive.allow_unsafe_code);
    }

    #[test]
    fn test_plugin_validation_framework() {
        let validator = PluginValidator::new();
        let fixture = PluginTestFixture::new();
        let mock = testing_utils::MockPlugin::new("validation_test");
        let manifest = &fixture.create_test_manifests()[3]; // Signed manifest for validation

        let report = validator.validate_comprehensive(&mock, manifest).unwrap();
        assert!(!report.has_errors());
    }

    #[test]
    fn test_permission_system() {
        let fs_read = Permission::FileSystemRead;
        let fs_write = Permission::FileSystemWrite;
        let sys_cmd = Permission::SystemCommands;

        assert_eq!(fs_read.risk_level(), 2);
        assert_eq!(fs_write.risk_level(), 3);
        assert_eq!(sys_cmd.risk_level(), 5);

        assert!(!fs_read.requires_user_consent());
        assert!(fs_write.requires_user_consent());
        assert!(sys_cmd.requires_user_consent());
    }

    #[test]
    fn test_plugin_categories() {
        let categories = vec![
            PluginCategory::Algorithm,
            PluginCategory::Transformer,
            PluginCategory::DataProcessor,
            PluginCategory::Evaluator,
            PluginCategory::Visualizer,
        ];

        for category in categories {
            let mock = testing_utils::MockPlugin::for_category("test", category.clone());
            assert_eq!(mock.metadata().category, category);
        }
    }

    #[test]
    fn test_type_compatibility() {
        let mut mock = testing_utils::MockPlugin::new("type_test");

        // Should support f64 by default
        assert!(mock.is_compatible(TypeId::of::<f64>()));

        // Add i32 support
        mock.add_supported_type(TypeId::of::<i32>());
        assert!(mock.is_compatible(TypeId::of::<i32>()));

        // String should not be supported
        assert!(!mock.is_compatible(TypeId::of::<String>()));
    }

    #[test]
    fn test_performance_testing() {
        let mut tester = PluginPerformanceTester::new();
        let mut mock = testing_utils::MockPlugin::new("perf_test");

        let result = tester.benchmark_initialization(&mut mock);
        assert!(result.success);
        assert_eq!(result.operation, "initialization");
    }

    #[test]
    fn test_marketplace_components() {
        let marketplace = PluginMarketplace::new();

        // Test feature score calculation
        let score = marketplace.calculate_feature_score(4.5, 1000, 50);
        assert!(score > 0.0);
        assert!(score <= 10.0);
    }

    #[test]
    fn test_trust_store() {
        let mut trust_store = TrustStore::new();

        let key_info = PublicKeyInfo {
            algorithm: "RSA".to_string(),
            key_size: 2048,
            added_timestamp: std::time::SystemTime::now(),
            expires_at: None,
            owner: "test@example.com".to_string(),
        };

        trust_store.add_trusted_key("test_key".to_string(), key_info);
        assert!(!trust_store.is_key_revoked("test_key"));

        trust_store.revoke_key("test_key".to_string());
        assert!(trust_store.is_key_revoked("test_key"));
    }
}

// Module feature gates for optional functionality
// PluginLoader already exported above unconditionally

// External dependencies documentation
//
// # External Dependencies
//
// This module uses several external crates for enhanced functionality:
//
// - `libloading` - For dynamic library loading (when `dynamic_loading` feature is enabled)
// - `serde` - For serialization of plugin manifests and configurations
// - `tokio` - For async functionality in discovery and marketplace features
//
// # Feature Flags
//
// - `dynamic_loading` - Enables dynamic library loading capabilities
// - `async` - Enables async functionality for marketplace and discovery
// - `serde` - Enables serialization support for plugin configurations
//
// # Platform Support
//
// The plugin system supports the following platforms:
//
// - **Linux**: Full support including dynamic loading (.so files)
// - **macOS**: Full support including dynamic loading (.dylib files)
// - **Windows**: Full support including dynamic loading (.dll files)
//
// # Safety Considerations
//
// Dynamic loading involves unsafe operations. The plugin system takes several
// precautions to ensure safety:
//
// - Comprehensive validation before loading
// - Digital signature verification
// - Permission-based security model
// - Sandboxed execution environment
// - Memory safety checks
//
// However, users should only load plugins from trusted sources and always
// validate plugin manifests before installation.
