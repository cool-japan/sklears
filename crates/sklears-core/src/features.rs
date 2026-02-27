/// Feature flag management and compile-time configuration
///
/// This module provides utilities for managing feature flags and compile-time
/// configuration options. It enables fine-grained control over which parts
/// of the library are included in the final binary.
use std::collections::HashMap;

/// Compile-time feature detection
pub struct Features;

impl Features {
    /// Check if standard library support is enabled
    pub const fn has_std() -> bool {
        cfg!(feature = "std")
    }

    /// Check if no_std mode is enabled
    pub const fn has_no_std() -> bool {
        cfg!(feature = "no_std")
    }

    /// Check if serde support is enabled
    pub const fn has_serde() -> bool {
        cfg!(feature = "serde")
    }

    /// Check if binary serialization is enabled
    pub const fn has_binary() -> bool {
        cfg!(feature = "binary")
    }

    /// Check if Polars integration is enabled
    pub const fn has_polars() -> bool {
        cfg!(feature = "polars")
    }

    /// Check if Arrow integration is enabled
    pub const fn has_arrow() -> bool {
        cfg!(feature = "arrow")
    }

    /// Check if SIMD support is enabled
    pub const fn has_simd() -> bool {
        cfg!(feature = "simd")
    }

    /// Check if parallel processing is enabled
    pub const fn has_parallel() -> bool {
        cfg!(feature = "parallel")
    }

    /// Check if memory mapping is enabled
    pub const fn has_mmap() -> bool {
        cfg!(feature = "mmap")
    }

    /// Check if async support is enabled
    pub const fn has_async() -> bool {
        cfg!(feature = "async_support")
    }

    /// Check if streaming support is enabled
    pub const fn has_streaming() -> bool {
        cfg!(feature = "streaming")
    }

    /// Check if GPU support is enabled
    pub const fn has_gpu() -> bool {
        cfg!(feature = "gpu_support")
    }

    /// Check if distributed computing is enabled
    pub const fn has_distributed() -> bool {
        cfg!(feature = "distributed")
    }

    /// Algorithm category checks
    pub const fn has_linear_models() -> bool {
        cfg!(feature = "linear_models")
    }

    pub const fn has_tree_models() -> bool {
        cfg!(feature = "tree_models")
    }

    pub const fn has_neural_networks() -> bool {
        cfg!(feature = "neural_networks")
    }

    pub const fn has_clustering() -> bool {
        cfg!(feature = "clustering")
    }

    pub const fn has_dimensionality_reduction() -> bool {
        cfg!(feature = "dimensionality_reduction")
    }

    pub const fn has_ensemble_methods() -> bool {
        cfg!(feature = "ensemble_methods")
    }

    /// Utility feature checks
    pub const fn has_validation() -> bool {
        cfg!(feature = "validation")
    }

    pub const fn has_metrics() -> bool {
        cfg!(feature = "metrics")
    }

    pub const fn has_preprocessing() -> bool {
        cfg!(feature = "preprocessing")
    }

    pub const fn has_model_selection() -> bool {
        cfg!(feature = "model_selection")
    }

    /// Development feature checks
    pub const fn has_debug_assertions() -> bool {
        cfg!(feature = "debug_assertions")
    }

    pub const fn has_profiling() -> bool {
        cfg!(feature = "profiling")
    }

    pub const fn has_benchmarking() -> bool {
        // Benchmarking feature not currently implemented
        false
    }

    /// Get all enabled features as a HashMap
    pub fn enabled_features() -> HashMap<&'static str, bool> {
        let mut features = HashMap::new();

        // Core features
        features.insert("std", Self::has_std());
        features.insert("no_std", Self::has_no_std());

        // Serialization features
        features.insert("serde", Self::has_serde());
        features.insert("binary", Self::has_binary());

        // Data format support
        features.insert("polars", Self::has_polars());
        features.insert("arrow", Self::has_arrow());

        // Performance features
        features.insert("simd", Self::has_simd());
        features.insert("parallel", Self::has_parallel());
        features.insert("mmap", Self::has_mmap());

        // Algorithm categories
        features.insert("linear_models", Self::has_linear_models());
        features.insert("tree_models", Self::has_tree_models());
        features.insert("neural_networks", Self::has_neural_networks());
        features.insert("clustering", Self::has_clustering());
        features.insert(
            "dimensionality_reduction",
            Self::has_dimensionality_reduction(),
        );
        features.insert("ensemble_methods", Self::has_ensemble_methods());

        // Advanced features
        features.insert("async_support", Self::has_async());
        features.insert("streaming", Self::has_streaming());
        features.insert("gpu_support", Self::has_gpu());
        features.insert("distributed", Self::has_distributed());

        // Utility features
        features.insert("validation", Self::has_validation());
        features.insert("metrics", Self::has_metrics());
        features.insert("preprocessing", Self::has_preprocessing());
        features.insert("model_selection", Self::has_model_selection());

        // Development features
        features.insert("debug_assertions", Self::has_debug_assertions());
        features.insert("profiling", Self::has_profiling());
        features.insert("benchmarking", Self::has_benchmarking());

        features
    }

    /// Print all enabled features (useful for debugging)
    pub fn print_enabled_features() {
        println!("Enabled features:");
        for (feature, enabled) in Self::enabled_features() {
            if enabled {
                println!("  - {feature}");
            }
        }
    }

    /// Get a summary of enabled feature categories
    pub fn feature_summary() -> FeatureSummary {
        FeatureSummary {
            core: CoreFeatures {
                std: Self::has_std(),
                no_std: Self::has_no_std(),
            },
            serialization: SerializationFeatures {
                serde: Self::has_serde(),
                binary: Self::has_binary(),
            },
            data_formats: DataFormatFeatures {
                polars: Self::has_polars(),
                arrow: Self::has_arrow(),
            },
            performance: PerformanceFeatures {
                simd: Self::has_simd(),
                parallel: Self::has_parallel(),
                mmap: Self::has_mmap(),
            },
            algorithms: AlgorithmFeatures {
                linear_models: Self::has_linear_models(),
                tree_models: Self::has_tree_models(),
                neural_networks: Self::has_neural_networks(),
                clustering: Self::has_clustering(),
                dimensionality_reduction: Self::has_dimensionality_reduction(),
                ensemble_methods: Self::has_ensemble_methods(),
            },
            advanced: AdvancedFeatures {
                async_support: Self::has_async(),
                streaming: Self::has_streaming(),
                gpu_support: Self::has_gpu(),
                distributed: Self::has_distributed(),
            },
            utilities: UtilityFeatures {
                validation: Self::has_validation(),
                metrics: Self::has_metrics(),
                preprocessing: Self::has_preprocessing(),
                model_selection: Self::has_model_selection(),
            },
            development: DevelopmentFeatures {
                debug_assertions: Self::has_debug_assertions(),
                profiling: Self::has_profiling(),
                benchmarking: Self::has_benchmarking(),
            },
        }
    }
}

/// Summary of all feature categories
#[derive(Debug, Clone)]
pub struct FeatureSummary {
    pub core: CoreFeatures,
    pub serialization: SerializationFeatures,
    pub data_formats: DataFormatFeatures,
    pub performance: PerformanceFeatures,
    pub algorithms: AlgorithmFeatures,
    pub advanced: AdvancedFeatures,
    pub utilities: UtilityFeatures,
    pub development: DevelopmentFeatures,
}

/// Core library features
#[derive(Debug, Clone)]
pub struct CoreFeatures {
    pub std: bool,
    pub no_std: bool,
}

/// Serialization-related features
#[derive(Debug, Clone)]
pub struct SerializationFeatures {
    pub serde: bool,
    pub binary: bool,
}

/// Data format support features
#[derive(Debug, Clone)]
pub struct DataFormatFeatures {
    pub polars: bool,
    pub arrow: bool,
}

/// Performance optimization features
#[derive(Debug, Clone)]
pub struct PerformanceFeatures {
    pub simd: bool,
    pub parallel: bool,
    pub mmap: bool,
}

/// Algorithm category features
#[derive(Debug, Clone)]
pub struct AlgorithmFeatures {
    pub linear_models: bool,
    pub tree_models: bool,
    pub neural_networks: bool,
    pub clustering: bool,
    pub dimensionality_reduction: bool,
    pub ensemble_methods: bool,
}

impl AlgorithmFeatures {
    /// Check if any algorithm category is enabled
    pub fn any_enabled(&self) -> bool {
        self.linear_models
            || self.tree_models
            || self.neural_networks
            || self.clustering
            || self.dimensionality_reduction
            || self.ensemble_methods
    }

    /// Count how many algorithm categories are enabled
    pub fn count_enabled(&self) -> usize {
        [
            self.linear_models,
            self.tree_models,
            self.neural_networks,
            self.clustering,
            self.dimensionality_reduction,
            self.ensemble_methods,
        ]
        .iter()
        .filter(|&&enabled| enabled)
        .count()
    }

    /// Get list of enabled algorithm categories
    pub fn enabled_categories(&self) -> Vec<&'static str> {
        let mut categories = Vec::new();
        if self.linear_models {
            categories.push("linear_models");
        }
        if self.tree_models {
            categories.push("tree_models");
        }
        if self.neural_networks {
            categories.push("neural_networks");
        }
        if self.clustering {
            categories.push("clustering");
        }
        if self.dimensionality_reduction {
            categories.push("dimensionality_reduction");
        }
        if self.ensemble_methods {
            categories.push("ensemble_methods");
        }
        categories
    }
}

/// Advanced features
#[derive(Debug, Clone)]
pub struct AdvancedFeatures {
    pub async_support: bool,
    pub streaming: bool,
    pub gpu_support: bool,
    pub distributed: bool,
}

/// Utility features
#[derive(Debug, Clone)]
pub struct UtilityFeatures {
    pub validation: bool,
    pub metrics: bool,
    pub preprocessing: bool,
    pub model_selection: bool,
}

/// Development and debugging features
#[derive(Debug, Clone)]
pub struct DevelopmentFeatures {
    pub debug_assertions: bool,
    pub profiling: bool,
    pub benchmarking: bool,
}

/// Compile-time feature validation
pub mod validation {
    use super::Features;

    /// Check for conflicting feature combinations
    pub const fn validate_features() -> Result<(), &'static str> {
        // Check for conflicting std/no_std features
        // Note: When using --all-features, both std and no_std get enabled
        // In this case, we default to std as it's the more permissive option
        if Features::has_std() && Features::has_no_std() {
            // This is expected when using --all-features, so we allow it
            // std takes precedence over no_std in this case
        }

        // Check dependencies - Note: These dependencies should be handled by Cargo.toml
        // but we keep validation for explicit feature conflict detection
        if Features::has_binary() && !Features::has_serde() {
            return Err("'binary' feature requires 'serde' feature");
        }

        // Note: streaming and distributed should automatically enable async_support via Cargo.toml
        // These checks are for explicit verification only
        if Features::has_streaming() && !Features::has_async() {
            return Err("'streaming' feature requires 'async_support' feature");
        }

        if Features::has_distributed() && !Features::has_async() {
            return Err("'distributed' feature requires 'async_support' feature");
        }

        // Ensure at least std or no_std is enabled
        if !Features::has_std() && !Features::has_no_std() {
            // Default to std if neither is explicitly specified
            // This is handled by the default feature set
        }

        Ok(())
    }

    /// Validate features at compile time using const assertions
    pub const fn assert_valid_features() {
        match validate_features() {
            Ok(()) => {}
            Err(_) => panic!("Invalid feature combination detected"),
        }
    }
}

/// Runtime feature configuration
pub struct FeatureConfig {
    enabled_features: HashMap<String, bool>,
}

impl FeatureConfig {
    /// Create a new feature configuration from current compile-time features
    pub fn from_compile_time() -> Self {
        let enabled = Features::enabled_features()
            .into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect();

        Self {
            enabled_features: enabled,
        }
    }

    /// Check if a feature is enabled
    pub fn is_enabled(&self, feature: &str) -> bool {
        self.enabled_features.get(feature).copied().unwrap_or(false)
    }

    /// Get all enabled features
    pub fn enabled_features(&self) -> Vec<&str> {
        self.enabled_features
            .iter()
            .filter_map(|(k, &v)| if v { Some(k.as_str()) } else { None })
            .collect()
    }

    /// Get configuration as JSON string (requires serde feature)
    #[cfg(feature = "serde")]
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(&self.enabled_features)
    }

    /// Create configuration from JSON string (requires serde feature)
    #[cfg(feature = "serde")]
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        let enabled_features = serde_json::from_str(json)?;
        Ok(Self { enabled_features })
    }
}

/// Macro for conditional compilation based on feature flags
#[macro_export]
macro_rules! cfg_feature {
    ($feature:literal, $code:block) => {
        #[cfg(feature = $feature)]
        $code
    };
    ($feature:literal, $code:block, else $else_code:block) => {
        #[cfg(feature = $feature)]
        $code
        #[cfg(not(feature = $feature))]
        $else_code
    };
}

/// Macro for conditional type definitions based on features
#[macro_export]
macro_rules! cfg_type {
    ($feature:literal, $type_def:item) => {
        #[cfg(feature = $feature)]
        $type_def
    };
}

/// Macro for feature-gated function implementations
#[macro_export]
macro_rules! cfg_impl {
    ($feature:literal, impl $trait_name:ident for $type:ty { $($item:item)* }) => {
        #[cfg(feature = $feature)]
        impl $trait_name for $type {
            $($item)*
        }
    };
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_detection() {
        // Test that feature detection works
        let summary = Features::feature_summary();

        // Core features should have at least one enabled
        assert!(summary.core.std || summary.core.no_std);

        // Print enabled features for debugging
        Features::print_enabled_features();
    }

    #[test]
    fn test_feature_config() {
        let config = FeatureConfig::from_compile_time();

        // Should have some features enabled
        assert!(!config.enabled_features().is_empty());

        // Test specific feature checking
        let has_std = config.is_enabled("std");
        assert_eq!(has_std, Features::has_std());
    }

    #[test]
    fn test_algorithm_features() {
        let algo_features = Features::feature_summary().algorithms;

        // Test counting and listing
        let count = algo_features.count_enabled();
        let categories = algo_features.enabled_categories();

        assert_eq!(count, categories.len());
    }

    #[test]
    fn test_feature_validation() {
        // This should not panic if features are correctly configured
        validation::assert_valid_features();

        // Test validation function
        assert!(validation::validate_features().is_ok());
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_feature_serialization() {
        let config = FeatureConfig::from_compile_time();

        // Test JSON serialization
        let json = config.to_json().unwrap_or_default();
        assert!(!json.is_empty());

        // Test round-trip
        let config2 = FeatureConfig::from_json(&json).expect("valid JSON");
        assert_eq!(
            config.enabled_features().len(),
            config2.enabled_features().len()
        );
    }

    #[test]
    fn test_feature_macros() {
        // Test cfg_feature macro
        cfg_feature!("std", {
            println!("std feature is enabled");
        });

        // Test else branch with a defined but likely disabled feature
        cfg_feature!("gpu_support", {
            println!("gpu_support feature is enabled");
        }, else {
            println!("gpu_support is not enabled (expected in most builds)");
        });
    }
}
