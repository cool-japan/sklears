//! Memory management and caching for explanation algorithms
//!
//! This module provides comprehensive memory management capabilities for explanation computation,
//! including efficient caching, memory layout optimization, SIMD operations, memory-mapped storage,
//! and shared explanation management with reference counting.
//!
//! # Architecture
//!
//! The memory management system is organized into several specialized modules:
//!
//! - **`cache`**: Core caching system with thread-safe storage and hit/miss tracking
//! - **`cache_ops`**: Cache-friendly computation operations for feature importance and SHAP
//! - **`layout_manager`**: Memory layout optimization with SIMD vectorized operations
//! - **`unsafe_ops`**: High-performance unsafe operations for critical performance paths
//! - **`storage`**: Memory-mapped storage system for persistent explanation data
//! - **`shared_manager`**: Reference-counted shared explanations with lifecycle management
//!
//! # Usage Examples
//!
//! ## Basic Caching
//!
//! ```rust,no_run
//! use sklears_inspection::memory::{CacheConfig, ExplanationCache, CacheKey};
//! # use scirs2_core::ndarray::array;
//! # type Float = f64;
//!
//! let config = CacheConfig::default();
//! let cache = ExplanationCache::new(&config);
//!
//! let data = array![[1.0, 2.0], [3.0, 4.0]];
//! let key = CacheKey::new(&data.view(), "feature_importance", 0);
//!
//! // Cache-friendly computation
//! let result = cache.get_or_compute_feature_importance(&key, || {
//!     // Expensive computation here
//!     Ok(array![0.8, 0.2])
//! }).unwrap();
//! ```
//!
//! ## Memory Layout Optimization
//!
//! ```rust,no_run
//! use sklears_inspection::memory::{MemoryLayoutManager, ExplanationDataLayout};
//!
//! let layout = ExplanationDataLayout::default();
//! let manager = MemoryLayoutManager::new(layout);
//!
//! // Allocate aligned memory for better performance
//! let memory = manager.allocate_aligned(1000);
//! // Use memory for computations...
//! manager.deallocate(memory);
//! ```
//!
//! ## Memory-Mapped Storage
//!
//! ```rust,no_run
//! use sklears_inspection::memory::{MemoryMappedStorage, MemoryMapConfig};
//! use tempfile::TempDir;
//! # use scirs2_core::ndarray::array;
//!
//! let temp_dir = TempDir::new().unwrap();
//! let config = MemoryMapConfig::default();
//! let storage = MemoryMappedStorage::new(temp_dir.path(), config).unwrap();
//!
//! let feature_importance = array![1.0, 2.0, 3.0];
//! let result = storage.store_explanation_results(
//!     "my_explanation",
//!     &feature_importance,
//!     None,
//! ).unwrap();
//!
//! // Later, load the data
//! let loaded = storage.load_feature_importance(&result).unwrap();
//! ```
//!
//! ## Shared Explanation Management
//!
//! ```rust,no_run
//! use sklears_inspection::memory::{SharedExplanationManager, SharedExplanationConfig, ExplanationMetadata};
//! use std::collections::HashMap;
//! # use scirs2_core::ndarray::array;
//!
//! let config = SharedExplanationConfig::default();
//! let manager = SharedExplanationManager::new(config);
//!
//! let feature_importance = array![1.0, 2.0, 3.0];
//! let metadata = ExplanationMetadata {
//!     model_type: "RandomForest".to_string(),
//!     n_features: 3,
//!     n_samples: 100,
//!     method: "SHAP".to_string(),
//!     config: HashMap::new(),
//! };
//!
//! let explanation = manager.create_explanation(
//!     "my_explanation".to_string(),
//!     feature_importance,
//!     None,
//!     metadata,
//! ).unwrap();
//!
//! // Explanation is reference-counted and automatically managed
//! println!("Reference count: {}", explanation.ref_count());
//! ```

// Re-export all public items from submodules
pub mod cache;
pub mod cache_ops;
pub mod layout_manager;
pub mod shared_manager;
pub mod storage;
pub mod unsafe_ops;

// Core caching system
pub use cache::{CacheConfig, CacheKey, CacheStatistics, ExplanationCache};

// Cache-friendly operations
pub use cache_ops::{
    cache_friendly_permutation_importance, cache_friendly_shap_computation, compute_baseline_data,
    compute_r2_score,
};

// Memory layout management
pub use layout_manager::{ExplanationDataLayout, MemoryLayoutManager};

// Shared explanation management
pub use shared_manager::{
    ExplanationData, ExplanationMetadata, ManagerStats, SharedExplanation, SharedExplanationConfig,
    SharedExplanationManager,
};

// Memory-mapped storage
pub use storage::{MappedExplanationResult, MemoryMapConfig, MemoryMappedStorage, StorageStats};

// High-performance unsafe operations
pub use unsafe_ops::{compute_feature_importance_unsafe, UnsafeArrayOps};

/// Convenience prelude for common memory management operations
pub mod prelude {
    pub use super::{
        cache_friendly_permutation_importance, cache_friendly_shap_computation, CacheConfig,
        CacheKey, ExplanationCache, ExplanationDataLayout, MemoryLayoutManager, MemoryMapConfig,
        MemoryMappedStorage, SharedExplanationConfig, SharedExplanationManager,
    };
}

#[cfg(test)]
mod integration_tests {
    use super::*;
    use crate::types::*;
    // ✅ SciRS2 Policy Compliant Import
    use scirs2_core::ndarray::{array, ArrayView2};
    use std::collections::HashMap;
    use tempfile::TempDir;

    #[test]
    fn test_memory_system_integration() {
        // Test integration between cache, layout manager, and shared manager
        let cache_config = CacheConfig::default();
        let cache = ExplanationCache::new(&cache_config);

        let layout = ExplanationDataLayout::default();
        let layout_manager = MemoryLayoutManager::new(layout);

        let shared_config = SharedExplanationConfig::default();
        let shared_manager = SharedExplanationManager::new(shared_config);

        // Test data
        let X = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let y = array![1.0, 2.0, 3.0];

        // Simple model for testing
        let model =
            |x: &ArrayView2<Float>| -> crate::SklResult<scirs2_core::ndarray::Array1<Float>> {
                Ok(x.column(0).to_owned())
            };

        // Test cache-friendly computation
        let cache_key = CacheKey::new(&X.view(), "test_method", 0);
        let importances = cache
            .get_or_compute_feature_importance(&cache_key, || {
                cache_ops::cache_friendly_permutation_importance(
                    &X.view(),
                    &y.view(),
                    &model,
                    &cache,
                    &cache_config,
                )
            })
            .unwrap();

        assert_eq!(importances.len(), 2);

        // Test memory allocation
        let memory = layout_manager.allocate_aligned(100);
        assert_eq!(memory.len(), 100);
        layout_manager.deallocate(memory);

        // Test shared explanation
        let metadata = ExplanationMetadata {
            model_type: "TestModel".to_string(),
            n_features: 2,
            n_samples: 3,
            method: "PermutationImportance".to_string(),
            config: HashMap::new(),
        };

        let explanation = shared_manager
            .create_explanation("integration_test".to_string(), importances, None, metadata)
            .unwrap();

        assert_eq!(explanation.feature_importance().len(), 2);
        assert_eq!(explanation.metadata().model_type, "TestModel");
    }

    #[test]
    fn test_storage_integration() {
        let temp_dir = TempDir::new().unwrap();
        let storage_config = MemoryMapConfig::default();
        let storage = MemoryMappedStorage::new(temp_dir.path(), storage_config).unwrap();

        // Create test data
        let feature_importance = array![0.8, 0.2];
        let shap_values = array![[0.1, 0.2], [0.3, 0.4]];

        // Store data
        let result = storage
            .store_explanation_results("storage_test", &feature_importance, Some(&shap_values))
            .unwrap();

        // Load data back
        let loaded_importance = storage.load_feature_importance(&result).unwrap();
        let loaded_shap = storage.load_shap_values(&result).unwrap();

        // Verify data integrity
        assert_eq!(loaded_importance.len(), feature_importance.len());
        assert_eq!(loaded_shap.shape(), shap_values.shape());

        for (a, b) in loaded_importance.iter().zip(feature_importance.iter()) {
            assert!((a - b).abs() < 1e-10);
        }

        for (a, b) in loaded_shap.iter().zip(shap_values.iter()) {
            assert!((a - b).abs() < 1e-10);
        }

        // Test storage stats
        let stats = storage.get_storage_stats();
        assert_eq!(stats.total_files, 1);
        assert!(stats.total_size > 0);
    }

    #[test]
    fn test_unsafe_ops_integration() {
        let X = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let y = array![3.0, 7.0, 11.0]; // y = x1 + x2

        let model = |x: &ArrayView2<Float>| -> scirs2_core::ndarray::Array1<Float> {
            x.sum_axis(scirs2_core::ndarray::Axis(1))
        };

        // Test unsafe feature importance computation
        let importances = unsafe_ops::compute_feature_importance_unsafe(
            &model,
            &X.view(),
            &y.view(),
            3,        // n_repeats
            Some(42), // random_state
        )
        .unwrap();

        assert_eq!(importances.len(), 2);
        assert!(importances.iter().any(|&x| x.abs() > 0.0));

        // Test unsafe array operations
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![2.0, 3.0, 4.0, 5.0];
        let mut result = vec![0.0; 4];

        unsafe {
            UnsafeArrayOps::fast_multiply(a.as_ptr(), b.as_ptr(), result.as_mut_ptr(), 4);
            assert!((result[0] - 2.0).abs() < 1e-10);
            assert!((result[1] - 6.0).abs() < 1e-10);

            let sum = UnsafeArrayOps::fast_sum(a.as_ptr(), 4);
            assert!((sum - 10.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_cache_statistics_integration() {
        let config = CacheConfig::default();
        let cache = ExplanationCache::new(&config);

        let X = array![[1.0, 2.0], [3.0, 4.0]];
        let key = CacheKey::new(&X.view(), "test", 0);

        // First access - should be miss
        let _result1 = cache
            .get_or_compute_feature_importance(&key, || Ok(array![0.5, 0.3]))
            .unwrap();

        // Second access - should be hit
        let _result2 = cache
            .get_or_compute_feature_importance(&key, || {
                Ok(array![0.1, 0.9]) // Different values - should not be computed
            })
            .unwrap();

        let stats = cache.get_statistics();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
        assert!(stats.total_size > 0);
    }
}
