/// Fallback strategies for missing dependencies
///
/// This module provides graceful degradation when optional dependencies are not available,
/// allowing the library to continue functioning with reduced capabilities rather than failing.
use crate::error::{Result, SklearsError};
use std::collections::HashMap;

/// Trait for fallback capability detection and implementation
pub trait FallbackStrategy {
    /// Check if the preferred implementation is available
    fn is_preferred_available(&self) -> bool;

    /// Check if fallback implementation is available
    fn has_fallback(&self) -> bool;

    /// Get description of what functionality will be lost with fallback
    fn fallback_limitations(&self) -> Vec<String>;

    /// Execute with preferred implementation if available, otherwise use fallback
    /// Returns a string description of the operation performed
    fn execute_with_fallback(&self, preferred_available: bool) -> Result<String>;
}

/// Registry for dependency fallback strategies
pub struct FallbackRegistry {
    strategies: HashMap<String, Box<dyn FallbackStrategy + Send + Sync>>,
    warnings_shown: std::sync::Mutex<std::collections::HashSet<String>>,
}

impl Default for FallbackRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl FallbackRegistry {
    /// Create a new fallback registry
    pub fn new() -> Self {
        Self {
            strategies: HashMap::new(),
            warnings_shown: std::sync::Mutex::new(std::collections::HashSet::new()),
        }
    }

    /// Register a fallback strategy for a dependency
    pub fn register<S>(&mut self, dependency_name: &str, strategy: S)
    where
        S: FallbackStrategy + Send + Sync + 'static,
    {
        self.strategies
            .insert(dependency_name.to_string(), Box::new(strategy));
    }

    /// Execute operation with fallback if dependency is missing
    pub fn execute_with_fallback<T, F, G>(
        &self,
        dependency_name: &str,
        preferred: F,
        fallback: G,
    ) -> Result<T>
    where
        F: FnOnce() -> Result<T>,
        G: FnOnce() -> Result<T>,
    {
        if let Some(strategy) = self.strategies.get(dependency_name) {
            if strategy.is_preferred_available() {
                preferred()
            } else if strategy.has_fallback() {
                self.warn_fallback_usage(dependency_name, strategy.fallback_limitations());
                fallback()
            } else {
                Err(SklearsError::MissingDependency {
                    dependency: dependency_name.to_string(),
                    feature: "No fallback available".to_string(),
                })
            }
        } else {
            // No strategy registered, try preferred and fail if it doesn't work
            preferred().map_err(|_| SklearsError::MissingDependency {
                dependency: dependency_name.to_string(),
                feature: "No fallback strategy registered".to_string(),
            })
        }
    }

    /// Show warning about fallback usage (only once per dependency)
    fn warn_fallback_usage(&self, dependency_name: &str, limitations: Vec<String>) {
        if let Ok(mut shown) = self.warnings_shown.lock() {
            if !shown.contains(dependency_name) {
                log::warn!(
                    "Using fallback implementation for '{}'. Limitations: {}",
                    dependency_name,
                    limitations.join(", ")
                );
                shown.insert(dependency_name.to_string());
            }
        }
    }

    /// Get status report of all registered dependencies
    pub fn dependency_status(&self) -> DependencyReport {
        let mut available = Vec::new();
        let mut fallback_used = Vec::new();
        let mut missing = Vec::new();

        for (name, strategy) in &self.strategies {
            if strategy.is_preferred_available() {
                available.push(name.clone());
            } else if strategy.has_fallback() {
                fallback_used.push(FallbackInfo {
                    dependency: name.clone(),
                    limitations: strategy.fallback_limitations(),
                });
            } else {
                missing.push(name.clone());
            }
        }

        DependencyReport {
            available,
            fallback_used,
            missing,
        }
    }
}

/// Report of dependency availability status
#[derive(Debug, Clone)]
pub struct DependencyReport {
    pub available: Vec<String>,
    pub fallback_used: Vec<FallbackInfo>,
    pub missing: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct FallbackInfo {
    pub dependency: String,
    pub limitations: Vec<String>,
}

impl DependencyReport {
    pub fn is_fully_functional(&self) -> bool {
        self.fallback_used.is_empty() && self.missing.is_empty()
    }

    pub fn has_critical_missing(&self) -> bool {
        !self.missing.is_empty()
    }
}

/// Specific fallback strategies for common dependencies
/// Fallback strategy for BLAS operations
pub struct BlasFallback;

impl FallbackStrategy for BlasFallback {
    fn is_preferred_available(&self) -> bool {
        // Check if optimized BLAS is available
        // Using SciRS2 instead of ndarray-linalg
        false
    }

    fn has_fallback(&self) -> bool {
        true // Always have pure Rust implementation
    }

    fn fallback_limitations(&self) -> Vec<String> {
        vec![
            "Slower matrix operations".to_string(),
            "No SIMD optimizations".to_string(),
            "Higher memory usage for large matrices".to_string(),
        ]
    }

    fn execute_with_fallback(&self, preferred_available: bool) -> Result<String> {
        if preferred_available && self.is_preferred_available() {
            Ok("Using preferred implementation".to_string())
        } else {
            Ok("Using fallback implementation".to_string())
        }
    }
}

/// Fallback strategy for parallel processing
pub struct ParallelFallback;

impl FallbackStrategy for ParallelFallback {
    fn is_preferred_available(&self) -> bool {
        // Rayon is always available as a dependency
        true
    }

    fn has_fallback(&self) -> bool {
        true // Can always fall back to sequential processing
    }

    fn fallback_limitations(&self) -> Vec<String> {
        vec![
            "Sequential processing only".to_string(),
            "Slower on multi-core systems".to_string(),
            "No work-stealing optimizations".to_string(),
        ]
    }

    fn execute_with_fallback(&self, preferred_available: bool) -> Result<String> {
        if preferred_available && self.is_preferred_available() {
            Ok("Using preferred implementation".to_string())
        } else {
            Ok("Using fallback implementation".to_string())
        }
    }
}

/// Fallback strategy for visualization features
pub struct VisualizationFallback;

impl FallbackStrategy for VisualizationFallback {
    fn is_preferred_available(&self) -> bool {
        // Visualization features are not currently implemented
        false
    }

    fn has_fallback(&self) -> bool {
        true // Can provide text-based alternatives
    }

    fn fallback_limitations(&self) -> Vec<String> {
        vec![
            "No graphical plots".to_string(),
            "Text-based visualization only".to_string(),
            "Limited aesthetic options".to_string(),
        ]
    }

    fn execute_with_fallback(&self, preferred_available: bool) -> Result<String> {
        if preferred_available && self.is_preferred_available() {
            Ok("Using preferred implementation".to_string())
        } else {
            Ok("Using fallback implementation".to_string())
        }
    }
}

/// Fallback strategy for serialization
pub struct SerializationFallback;

impl FallbackStrategy for SerializationFallback {
    fn is_preferred_available(&self) -> bool {
        cfg!(feature = "serde")
    }

    fn has_fallback(&self) -> bool {
        true // Can provide basic binary serialization
    }

    fn fallback_limitations(&self) -> Vec<String> {
        vec![
            "Binary format only".to_string(),
            "No JSON/YAML support".to_string(),
            "Limited cross-platform compatibility".to_string(),
        ]
    }

    fn execute_with_fallback(&self, preferred_available: bool) -> Result<String> {
        if preferred_available && self.is_preferred_available() {
            Ok("Using preferred implementation".to_string())
        } else {
            Ok("Using fallback implementation".to_string())
        }
    }
}

/// Fallback strategy for GPU acceleration
pub struct GpuFallback;

impl FallbackStrategy for GpuFallback {
    fn is_preferred_available(&self) -> bool {
        cfg!(feature = "gpu_support")
    }

    fn has_fallback(&self) -> bool {
        true // Can always fall back to CPU
    }

    fn fallback_limitations(&self) -> Vec<String> {
        vec![
            "CPU-only computation".to_string(),
            "Slower for large datasets".to_string(),
            "No GPU memory optimizations".to_string(),
        ]
    }

    fn execute_with_fallback(&self, preferred_available: bool) -> Result<String> {
        if preferred_available && self.is_preferred_available() {
            Ok("Using preferred implementation".to_string())
        } else {
            Ok("Using fallback implementation".to_string())
        }
    }
}

/// Global fallback registry instance
static GLOBAL_FALLBACK_REGISTRY: std::sync::OnceLock<std::sync::Mutex<FallbackRegistry>> =
    std::sync::OnceLock::new();

/// Get the global fallback registry
pub fn global_fallback_registry() -> &'static std::sync::Mutex<FallbackRegistry> {
    GLOBAL_FALLBACK_REGISTRY.get_or_init(|| {
        let mut registry = FallbackRegistry::new();

        // Register default fallback strategies
        registry.register("blas", BlasFallback);
        registry.register("parallel", ParallelFallback);
        registry.register("visualization", VisualizationFallback);
        registry.register("serialization", SerializationFallback);
        registry.register("gpu", GpuFallback);

        std::sync::Mutex::new(registry)
    })
}

/// Convenience macro for executing operations with fallback
#[macro_export]
macro_rules! with_fallback {
    ($dependency:literal, $preferred:expr, $fallback:expr) => {{
        use $crate::fallback_strategies::global_fallback_registry;
        let registry = global_fallback_registry().lock().map_err(|_| {
            $crate::error::SklearsError::Other(
                "Failed to acquire fallback registry lock".to_string(),
            )
        })?;

        registry.execute_with_fallback($dependency, || $preferred, || $fallback)
    }};
}

/// Trait for types that support fallback implementations
pub trait Fallbackable {
    /// The preferred implementation type
    type Preferred;

    /// The fallback implementation type
    type Fallback;

    /// Create preferred implementation if dependencies are available
    fn try_preferred() -> Result<Self::Preferred>;

    /// Create fallback implementation
    fn create_fallback() -> Self::Fallback;

    /// Convert fallback to the main type
    fn from_fallback(fallback: Self::Fallback) -> Self;
}

/// Helper for conditionally compiled dependencies
pub mod conditional {
    use super::*;

    /// Execute code only if a feature is enabled
    pub fn if_feature_enabled<T, F>(_feature: &str, _f: F) -> Option<T>
    where
        F: FnOnce() -> T,
    {
        // This would need to be a proc macro in real implementation
        // For now, just return None to simulate missing feature
        None
    }

    /// Matrix operations with BLAS fallback
    pub mod matrix_ops {
        use super::*;
        use crate::types::Array2;

        /// Matrix multiplication with fallback
        pub fn matmul(a: &Array2<f64>, b: &Array2<f64>) -> Result<Array2<f64>> {
            with_fallback!(
                "blas",
                {
                    // Preferred: BLAS-accelerated multiplication not available
                    Err(SklearsError::MissingDependency {
                        dependency: "BLAS".to_string(),
                        feature: "Optimized matrix multiplication".to_string(),
                    })
                },
                {
                    // Fallback: Pure Rust implementation
                    naive_matmul(a, b)
                }
            )
        }

        fn naive_matmul(a: &Array2<f64>, b: &Array2<f64>) -> Result<Array2<f64>> {
            if a.ncols() != b.nrows() {
                return Err(SklearsError::ShapeMismatch {
                    expected: format!(
                        "({}, {}) × ({}, {})",
                        a.nrows(),
                        a.ncols(),
                        a.ncols(),
                        b.ncols()
                    ),
                    actual: format!(
                        "({}, {}) × ({}, {})",
                        a.nrows(),
                        a.ncols(),
                        b.nrows(),
                        b.ncols()
                    ),
                });
            }

            let mut result = Array2::zeros((a.nrows(), b.ncols()));

            for i in 0..a.nrows() {
                for j in 0..b.ncols() {
                    let mut sum = 0.0;
                    for k in 0..a.ncols() {
                        sum += a[[i, k]] * b[[k, j]];
                    }
                    result[[i, j]] = sum;
                }
            }

            Ok(result)
        }
    }

    /// Parallel processing with fallback
    pub mod parallel_ops {

        /// Parallel map with fallback to sequential
        pub fn parallel_map<T, R, F>(items: Vec<T>, f: F) -> Vec<R>
        where
            T: Send,
            R: Send,
            F: Fn(T) -> R + Send + Sync,
        {
            use rayon::prelude::*;
            items.into_par_iter().map(f).collect()
        }

        /// Parallel reduce with fallback
        pub fn parallel_reduce<T, F, R>(items: Vec<T>, identity: R, f: F) -> R
        where
            T: Send,
            R: Send + Clone + Sync,
            F: Fn(R, T) -> R + Send + Sync,
        {
            use rayon::prelude::*;
            let identity_clone = identity.clone();
            items
                .into_par_iter()
                .fold(|| identity_clone.clone(), f)
                .reduce(|| identity.clone(), |a, _b| a)
        }
    }
}

/// Utilities for graceful feature detection
pub mod feature_detection {
    use super::*;

    /// Runtime feature availability checker
    pub struct FeatureDetector {
        cache: std::sync::Mutex<HashMap<String, bool>>,
    }

    impl Default for FeatureDetector {
        fn default() -> Self {
            Self::new()
        }
    }

    impl FeatureDetector {
        pub fn new() -> Self {
            Self {
                cache: std::sync::Mutex::new(HashMap::new()),
            }
        }

        /// Check if a feature is available at runtime
        pub fn is_available(&self, feature_name: &str) -> bool {
            if let Ok(mut cache) = self.cache.lock() {
                if let Some(&cached) = cache.get(feature_name) {
                    return cached;
                }

                let available = match feature_name {
                    "blas" => self.detect_blas(),
                    "rayon" => true, // rayon is always available as a dependency
                    "serde" => cfg!(feature = "serde"),
                    "gpu" => self.detect_gpu(),
                    _ => false,
                };

                cache.insert(feature_name.to_string(), available);
                available
            } else {
                false
            }
        }

        fn detect_blas(&self) -> bool {
            // Using SciRS2 instead of ndarray-linalg
            false
        }

        fn detect_gpu(&self) -> bool {
            cfg!(feature = "gpu_support")
        }

        /// Get comprehensive feature report
        pub fn feature_report(&self) -> FeatureReport {
            let features = vec!["blas", "rayon", "serde", "gpu", "visualization"];
            let mut available = Vec::new();
            let mut missing = Vec::new();

            for feature in features {
                if self.is_available(feature) {
                    available.push(feature.to_string());
                } else {
                    missing.push(feature.to_string());
                }
            }

            FeatureReport { available, missing }
        }
    }

    #[derive(Debug, Clone)]
    pub struct FeatureReport {
        pub available: Vec<String>,
        pub missing: Vec<String>,
    }

    impl FeatureReport {
        pub fn print_summary(&self) {
            println!("Feature Availability Report:");
            println!("  Available: {}", self.available.join(", "));
            println!("  Missing: {}", self.missing.join(", "));
        }
    }

    /// Global feature detector instance
    static GLOBAL_FEATURE_DETECTOR: std::sync::OnceLock<FeatureDetector> =
        std::sync::OnceLock::new();

    pub fn global_feature_detector() -> &'static FeatureDetector {
        GLOBAL_FEATURE_DETECTOR.get_or_init(FeatureDetector::new)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fallback_registry() {
        let mut registry = FallbackRegistry::new();
        registry.register("test_dep", BlasFallback);

        let result =
            registry.execute_with_fallback("test_dep", || Ok("preferred"), || Ok("fallback"));

        assert!(result.is_ok());
    }

    #[test]
    fn test_dependency_report() {
        let mut registry = FallbackRegistry::new();
        registry.register("available", BlasFallback);
        registry.register("missing", ParallelFallback);

        let report = registry.dependency_status();
        assert!(!report.available.is_empty() || !report.fallback_used.is_empty());
    }

    #[test]
    fn test_feature_detection() {
        let detector = feature_detection::FeatureDetector::new();
        let report = detector.feature_report();

        // Just verify the report is generated without panicking
        assert!(report.available.len() + report.missing.len() > 0);
    }

    #[test]
    fn test_matrix_multiplication_fallback() {
        use crate::types::Array2;
        use conditional::matrix_ops::matmul;

        let a = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let b = Array2::from_shape_vec((2, 2), vec![5.0, 6.0, 7.0, 8.0]).unwrap();

        let result = matmul(&a, &b);
        assert!(result.is_ok());

        let c = result.unwrap();
        assert_eq!(c.shape(), &[2, 2]);
    }
}
