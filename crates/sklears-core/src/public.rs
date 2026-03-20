/// Public API definitions and stability guarantees
///
/// This module clearly defines the public API surface of sklears-core.
/// All items in this module are covered by semantic versioning guarantees.
///
/// # Stability Guarantees
///
/// - **Stable APIs**: Items marked as stable will not have breaking changes
///   without a major version bump
/// - **Experimental APIs**: Items marked as experimental may change in minor versions
/// - **Deprecated APIs**: Items marked as deprecated will be removed in the next major version
///
/// # API Organization
///
/// The public API is organized into logical groups:
/// - Core traits and types
/// - Utility functions
/// - Configuration types
/// - Error handling
use crate::error::{Result, SklearsError};

// =============================================================================
// Stable Public APIs
// =============================================================================

/// Re-export of stable core traits
///
/// These traits form the foundation of the sklears ecosystem and are
/// guaranteed to remain stable across versions.
pub mod stable {
    pub use crate::traits::{
        Estimator, Fit, FitPredict, FitTransform, PartialFit, Predict, Transform,
    };

    // Additional traits will be added here as they are implemented

    pub use crate::types::{
        Array1, Array2, ArrayView1, ArrayView2, ArrayViewMut1, ArrayViewMut2, FeatureCount,
        Features, Float, FloatBounds, Int, IntBounds, Labels, Numeric, Predictions, Probabilities,
        Probability, SampleCount, Target,
    };

    pub use crate::error::{ErrorChain, ErrorContext, Result, SklearsError};

    pub use crate::validation::{Validate, ValidationContext, ValidationRule};

    pub use crate::dataset::{load_iris, make_blobs, make_regression, Dataset};
}

/// Experimental APIs that may change
///
/// These APIs are newer and may undergo breaking changes in minor versions.
/// Use with caution in production code.
pub mod experimental {
    pub use crate::async_traits::*;
    pub use crate::traits::gat_traits::*;
    pub use crate::traits::specialized::*;
    pub use crate::traits::streaming::*;

    pub use crate::plugin::{
        AlgorithmPlugin, ClusteringPlugin, Plugin, PluginConfig, PluginMetadata, PluginRegistry,
        TransformerPlugin,
    };

    pub use crate::parallel::{
        ParallelConfig, ParallelCrossValidation, ParallelFit, ParallelPredict, ParallelTransform,
    };

    #[cfg(feature = "simd")]
    pub use crate::simd::{SimdArrayOps, SimdOps};

    #[cfg(feature = "arrow")]
    pub use crate::arrow::{ArrowDataset, ColumnStats};
}

/// Deprecated APIs scheduled for removal
///
/// These APIs are deprecated and will be removed in the next major version.
/// Migration paths are provided where applicable.
pub mod deprecated {
    // No deprecated APIs yet, but this provides a clear place to put them

    #[deprecated(since = "0.1.0", note = "Use the new Plugin system instead")]
    pub fn old_plugin_system() {
        // Placeholder for deprecated functionality
    }
}

// =============================================================================
// Public API Markers
// =============================================================================

/// Marker trait for stable APIs
///
/// Types implementing this trait are guaranteed to have stable public APIs
/// that follow semantic versioning.
pub trait StableApi {
    /// API version this type was stabilized in
    const STABLE_SINCE: &'static str;

    /// Whether this API has any experimental features
    const HAS_EXPERIMENTAL_FEATURES: bool = false;
}

/// Marker trait for experimental APIs
///
/// Types implementing this trait are experimental and may change
/// without following strict semantic versioning.
pub trait ExperimentalApi {
    /// API version this type was introduced in
    const INTRODUCED_IN: &'static str;

    /// Expected version when this API will be stabilized
    const STABILIZATION_TARGET: Option<&'static str> = None;

    /// Known limitations or issues with this API
    const KNOWN_LIMITATIONS: &'static [&'static str] = &[];
}

// =============================================================================
// API Version Information
// =============================================================================

/// Information about API versions and compatibility
pub struct ApiVersionInfo {
    /// Current version of the core API
    pub core_version: &'static str,
    /// Minimum supported version for compatibility
    pub min_supported_version: &'static str,
    /// List of breaking changes since the minimum version
    pub breaking_changes: &'static [BreakingChange],
}

/// Information about a breaking change
#[derive(Debug, Clone)]
pub struct BreakingChange {
    /// Version where the breaking change was introduced
    pub version: &'static str,
    /// Description of the change
    pub description: &'static str,
    /// Migration guide or workaround
    pub migration: Option<&'static str>,
}

/// Get current API version information
pub fn api_version_info() -> ApiVersionInfo {
    ApiVersionInfo {
        core_version: "0.1.0",
        min_supported_version: "0.1.0",
        breaking_changes: &[
            // Future breaking changes will be documented here
        ],
    }
}

// =============================================================================
// Public Configuration Types
// =============================================================================

/// Configuration for public APIs
#[derive(Debug, Clone)]
pub struct PublicApiConfig {
    /// Enable experimental features
    pub enable_experimental: bool,
    /// Enable deprecated API warnings
    pub warn_deprecated: bool,
    /// Strict compatibility mode
    pub strict_compatibility: bool,
}

impl Default for PublicApiConfig {
    fn default() -> Self {
        Self {
            enable_experimental: false,
            warn_deprecated: true,
            strict_compatibility: true,
        }
    }
}

/// Builder for public API configuration
pub struct PublicApiConfigBuilder {
    config: PublicApiConfig,
}

impl PublicApiConfigBuilder {
    /// Create a new configuration builder
    pub fn new() -> Self {
        Self {
            config: PublicApiConfig::default(),
        }
    }

    /// Enable experimental features
    pub fn enable_experimental(mut self, enable: bool) -> Self {
        self.config.enable_experimental = enable;
        self
    }

    /// Enable deprecated API warnings
    pub fn warn_deprecated(mut self, warn: bool) -> Self {
        self.config.warn_deprecated = warn;
        self
    }

    /// Enable strict compatibility mode
    pub fn strict_compatibility(mut self, strict: bool) -> Self {
        self.config.strict_compatibility = strict;
        self
    }

    /// Build the configuration
    pub fn build(self) -> PublicApiConfig {
        self.config
    }
}

impl Default for PublicApiConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// API Stability Implementations
// =============================================================================

// Implement StableApi for core types
impl<T: crate::traits::Estimator<crate::traits::Untrained>> StableApi for T {
    const STABLE_SINCE: &'static str = "0.1.0";
}

// Note: Cannot implement StableApi for Fit trait without specifying associated types
// impl<X, Y> StableApi for dyn crate::traits::Fit<X, Y> {
//     const STABLE_SINCE: &'static str = "0.1.0";
// }

impl<X, Output> StableApi for dyn crate::traits::Predict<X, Output> {
    const STABLE_SINCE: &'static str = "0.1.0";
}

impl<X, Output> StableApi for dyn crate::traits::Transform<X, Output> {
    const STABLE_SINCE: &'static str = "0.1.0";
}

impl StableApi for crate::error::SklearsError {
    const STABLE_SINCE: &'static str = "0.1.0";
}

impl StableApi for crate::dataset::Dataset {
    const STABLE_SINCE: &'static str = "0.1.0";
}

// Implement ExperimentalApi for newer features
impl ExperimentalApi for crate::plugin::PluginRegistry {
    const INTRODUCED_IN: &'static str = "0.1.0";
    const STABILIZATION_TARGET: Option<&'static str> = Some("0.2.0");
    const KNOWN_LIMITATIONS: &'static [&'static str] = &[
        "Plugin unloading may not clean up all resources",
        "Dynamic loading requires platform-specific libraries",
    ];
}

#[cfg(feature = "simd")]
impl ExperimentalApi for crate::simd::SimdOps {
    const INTRODUCED_IN: &'static str = "0.1.0";
    const STABILIZATION_TARGET: Option<&'static str> = Some("0.3.0");
    const KNOWN_LIMITATIONS: &'static [&'static str] = &[
        "SIMD operations may not be available on all platforms",
        "Performance benefits vary by CPU architecture",
    ];
}

// Note: Cannot implement ExperimentalApi for AsyncFit trait without specifying associated types
// impl<X, Y> ExperimentalApi for dyn crate::traits::async_traits::AsyncFit<X, Y> {
//     const INTRODUCED_IN: &'static str = "0.1.0";
//     const STABILIZATION_TARGET: Option<&'static str> = Some("0.2.0");
//     const KNOWN_LIMITATIONS: &'static [&'static str] = &[
//         "Async runtime dependencies may conflict with user code",
//         "Error handling in async contexts needs improvement",
//     ];
// }

// =============================================================================
// Public Utility Functions
// =============================================================================

/// Check if an API is stable
pub fn is_api_stable<T: StableApi>() -> bool {
    true // All types implementing StableApi are stable by definition
}

/// Check if an API is experimental
pub fn is_api_experimental<T: ExperimentalApi>() -> bool {
    true // All types implementing ExperimentalApi are experimental by definition
}

/// Get stability information for a type
pub fn get_api_stability<T>() -> ApiStability
where
    T: 'static,
{
    let _type_id = std::any::TypeId::of::<T>();

    // This is a simplified implementation
    // In practice, you'd maintain a registry of type stability information
    ApiStability::Unknown
}

/// API stability classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApiStability {
    /// API is stable and follows semantic versioning
    Stable,
    /// API is experimental and may change
    Experimental,
    /// API is deprecated and will be removed
    Deprecated,
    /// API stability is unknown
    Unknown,
}

/// Validate that experimental features are enabled when needed
pub fn validate_experimental_usage<T: ExperimentalApi>(config: &PublicApiConfig) -> Result<()> {
    if !config.enable_experimental {
        return Err(SklearsError::InvalidOperation(format!(
            "Experimental API {} requires enable_experimental = true. \
                 This API was introduced in version {} and may change without notice.",
            std::any::type_name::<T>(),
            T::INTRODUCED_IN
        )));
    }
    Ok(())
}

/// Emit deprecation warning if configured
pub fn warn_if_deprecated<T>(config: &PublicApiConfig, api_name: &str) {
    if config.warn_deprecated {
        eprintln!(
            "Warning: API {api_name} is deprecated and will be removed in a future version. \
             Please update your code to use the recommended alternative."
        );
    }
}

// =============================================================================
// Tests
// =============================================================================

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_api_version_info() {
        let info = api_version_info();
        assert_eq!(info.core_version, "0.1.0");
        assert_eq!(info.min_supported_version, "0.1.0");
    }

    #[test]
    fn test_public_api_config() {
        let config = PublicApiConfigBuilder::new()
            .enable_experimental(true)
            .warn_deprecated(false)
            .strict_compatibility(false)
            .build();

        assert!(config.enable_experimental);
        assert!(!config.warn_deprecated);
        assert!(!config.strict_compatibility);
    }

    #[test]
    fn test_api_stability_traits() {
        // Test that we can determine API stability at compile time
        assert!(is_api_stable::<crate::error::SklearsError>());
        assert!(is_api_experimental::<crate::plugin::PluginRegistry>());
    }

    #[test]
    fn test_experimental_validation() {
        let config_disabled = PublicApiConfig {
            enable_experimental: false,
            ..Default::default()
        };

        let config_enabled = PublicApiConfig {
            enable_experimental: true,
            ..Default::default()
        };

        // This should fail with experimental disabled
        assert!(
            validate_experimental_usage::<crate::plugin::PluginRegistry>(&config_disabled).is_err()
        );

        // This should succeed with experimental enabled
        assert!(
            validate_experimental_usage::<crate::plugin::PluginRegistry>(&config_enabled).is_ok()
        );
    }

    #[test]
    fn test_api_stability_enum() {
        assert_eq!(ApiStability::Stable, ApiStability::Stable);
        assert_ne!(ApiStability::Stable, ApiStability::Experimental);
    }
}
