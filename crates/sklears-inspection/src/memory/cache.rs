//! Core caching system for explanation algorithms
//!
//! This module provides the fundamental caching infrastructure for explanation computation,
//! including cache management, key generation, and statistics tracking.

use crate::types::*;
use crate::SklResult;
// ✅ SciRS2 Policy Compliant Import
use scirs2_core::ndarray::{Array1, Array2, ArrayView2};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex};

/// Cache-friendly explanation computation with memory optimization
pub struct ExplanationCache {
    /// Feature importance cache
    feature_importance_cache: Arc<Mutex<HashMap<CacheKey, Array1<Float>>>>,
    /// Partial dependence cache
    partial_dependence_cache: Arc<Mutex<HashMap<CacheKey, Array2<Float>>>>,
    /// SHAP values cache
    shap_cache: Arc<Mutex<HashMap<CacheKey, Array2<Float>>>>,
    /// Model prediction cache
    prediction_cache: Arc<Mutex<HashMap<CacheKey, Array1<Float>>>>,
    /// Cache size limits
    max_cache_size: usize,
    /// Cache hit statistics
    cache_hits: Arc<Mutex<CacheStatistics>>,
}

/// Cache key for identifying cached computations
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct CacheKey {
    /// Data hash
    data_hash: u64,
    /// Method identifier
    method_id: String,
    /// Configuration hash
    config_hash: u64,
}

/// Cache statistics for monitoring performance
#[derive(Clone, Debug, Default)]
pub struct CacheStatistics {
    /// Number of cache hits
    pub hits: usize,
    /// Number of cache misses
    pub misses: usize,
    /// Total cache size in bytes
    pub total_size: usize,
    /// Average access time
    pub avg_access_time: f64,
}

/// Configuration for cache-friendly algorithms
#[derive(Clone, Debug)]
pub struct CacheConfig {
    /// Maximum cache size in MB
    pub max_cache_size_mb: usize,
    /// Enable data locality optimization
    pub enable_locality_optimization: bool,
    /// Prefetch distance for sequential access
    pub prefetch_distance: usize,
    /// Memory alignment for SIMD operations
    pub memory_alignment: usize,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            max_cache_size_mb: 256,
            enable_locality_optimization: true,
            prefetch_distance: 64,
            memory_alignment: 64,
        }
    }
}

impl ExplanationCache {
    /// Create a new explanation cache with specified configuration
    pub fn new(config: &CacheConfig) -> Self {
        Self {
            feature_importance_cache: Arc::new(Mutex::new(HashMap::new())),
            partial_dependence_cache: Arc::new(Mutex::new(HashMap::new())),
            shap_cache: Arc::new(Mutex::new(HashMap::new())),
            prediction_cache: Arc::new(Mutex::new(HashMap::new())),
            max_cache_size: config.max_cache_size_mb * 1024 * 1024,
            cache_hits: Arc::new(Mutex::new(CacheStatistics::default())),
        }
    }

    /// Get or compute feature importance with caching
    pub fn get_or_compute_feature_importance<F>(
        &self,
        key: &CacheKey,
        compute_fn: F,
    ) -> SklResult<Array1<Float>>
    where
        F: FnOnce() -> SklResult<Array1<Float>>,
    {
        // Check cache first
        {
            let cache = self.feature_importance_cache.lock().unwrap();
            if let Some(result) = cache.get(key) {
                // Cache hit
                let mut stats = self.cache_hits.lock().unwrap();
                stats.hits += 1;
                return Ok(result.clone());
            }
        }

        // Cache miss - compute and store
        let result = compute_fn()?;

        {
            let mut cache = self.feature_importance_cache.lock().unwrap();
            cache.insert(key.clone(), result.clone());

            // Update statistics
            let mut stats = self.cache_hits.lock().unwrap();
            stats.misses += 1;
            stats.total_size += result.len() * std::mem::size_of::<Float>();
        }

        Ok(result)
    }

    /// Get or compute SHAP values with caching
    pub fn get_or_compute_shap<F>(&self, key: &CacheKey, compute_fn: F) -> SklResult<Array2<Float>>
    where
        F: FnOnce() -> SklResult<Array2<Float>>,
    {
        // Check cache first
        {
            let cache = self.shap_cache.lock().unwrap();
            if let Some(result) = cache.get(key) {
                // Cache hit
                let mut stats = self.cache_hits.lock().unwrap();
                stats.hits += 1;
                return Ok(result.clone());
            }
        }

        // Cache miss - compute and store
        let result = compute_fn()?;

        {
            let mut cache = self.shap_cache.lock().unwrap();
            cache.insert(key.clone(), result.clone());

            // Update statistics
            let mut stats = self.cache_hits.lock().unwrap();
            stats.misses += 1;
            stats.total_size += result.len() * std::mem::size_of::<Float>();
        }

        Ok(result)
    }

    /// Get cache statistics
    pub fn get_statistics(&self) -> CacheStatistics {
        self.cache_hits.lock().unwrap().clone()
    }

    /// Clear all caches
    pub fn clear_all(&self) {
        self.feature_importance_cache.lock().unwrap().clear();
        self.partial_dependence_cache.lock().unwrap().clear();
        self.shap_cache.lock().unwrap().clear();
        self.prediction_cache.lock().unwrap().clear();

        let mut stats = self.cache_hits.lock().unwrap();
        *stats = CacheStatistics::default();
    }

    /// Evict least recently used entries if cache is full
    pub fn evict_lru(&self) {
        // Simple size-based eviction for now
        // In a production system, you would implement proper LRU tracking
        let total_size = self.cache_hits.lock().unwrap().total_size;
        if total_size > self.max_cache_size {
            // Clear half the cache
            self.feature_importance_cache.lock().unwrap().clear();
            self.partial_dependence_cache.lock().unwrap().clear();

            let mut stats = self.cache_hits.lock().unwrap();
            stats.total_size /= 2;
        }
    }
}

impl CacheKey {
    /// Create a new cache key from data and configuration
    pub fn new(data: &ArrayView2<Float>, method_id: &str, config_hash: u64) -> Self {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();

        // Hash data dimensions and a sample of values for efficiency
        data.shape().hash(&mut hasher);

        // Hash a sample of data values for uniqueness
        let sample_indices = if data.len() > 1000 {
            (0..data.len())
                .step_by(data.len() / 100)
                .collect::<Vec<_>>()
        } else {
            (0..data.len()).collect::<Vec<_>>()
        };

        for &idx in &sample_indices {
            let (row, col) = (idx / data.ncols(), idx % data.ncols());
            if let Some(val) = data.get((row, col)) {
                val.to_bits().hash(&mut hasher);
            }
        }

        let data_hash = hasher.finish();

        Self {
            data_hash,
            method_id: method_id.to_string(),
            config_hash,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // ✅ SciRS2 Policy Compliant Import
    use scirs2_core::ndarray::array;

    #[test]
    fn test_cache_key_creation() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let key1 = CacheKey::new(&x.view(), "test_method", 123);
        let key2 = CacheKey::new(&x.view(), "test_method", 123);
        let key3 = CacheKey::new(&x.view(), "different_method", 123);

        assert_eq!(key1, key2);
        assert_ne!(key1, key3);
    }

    #[test]
    fn test_explanation_cache_creation() {
        let config = CacheConfig::default();
        let cache = ExplanationCache::new(&config);

        let stats = cache.get_statistics();
        assert_eq!(stats.hits, 0);
        assert_eq!(stats.misses, 0);
    }

    #[test]
    fn test_cache_hit_and_miss() {
        let config = CacheConfig::default();
        let cache = ExplanationCache::new(&config);

        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let key = CacheKey::new(&x.view(), "test", 0);

        // First access should be a miss
        let result1 = cache
            .get_or_compute_feature_importance(&key, || Ok(array![0.5, 0.3]))
            .unwrap();

        let stats = cache.get_statistics();
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.hits, 0);

        // Second access should be a hit
        let result2 = cache
            .get_or_compute_feature_importance(&key, || {
                Ok(array![0.1, 0.9]) // Different values - should not be computed
            })
            .unwrap();

        let stats = cache.get_statistics();
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.hits, 1);

        // Results should be the same (from cache)
        assert_eq!(result1, result2);
    }

    #[test]
    fn test_cache_statistics() {
        let config = CacheConfig::default();
        let cache = ExplanationCache::new(&config);

        let x = array![[1.0, 2.0]];
        let key = CacheKey::new(&x.view(), "test", 0);

        // Perform some operations
        cache
            .get_or_compute_feature_importance(&key, || Ok(array![0.5, 0.3]))
            .unwrap();
        cache
            .get_or_compute_feature_importance(&key, || Ok(array![0.1, 0.9]))
            .unwrap();

        let stats = cache.get_statistics();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
        assert!(stats.total_size > 0);
    }

    #[test]
    fn test_cache_clear() {
        let config = CacheConfig::default();
        let cache = ExplanationCache::new(&config);

        let x = array![[1.0, 2.0]];
        let key = CacheKey::new(&x.view(), "test", 0);

        // Add something to cache
        cache
            .get_or_compute_feature_importance(&key, || Ok(array![0.5, 0.3]))
            .unwrap();

        let stats_before = cache.get_statistics();
        assert_eq!(stats_before.misses, 1);

        // Clear cache
        cache.clear_all();

        let stats_after = cache.get_statistics();
        assert_eq!(stats_after.hits, 0);
        assert_eq!(stats_after.misses, 0);
        assert_eq!(stats_after.total_size, 0);
    }
}
