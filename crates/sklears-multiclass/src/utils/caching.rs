//! Prediction caching utilities for multiclass classifiers
//!
//! This module provides caching functionality to improve performance when making
//! repeated predictions on the same data. It includes LRU (Least Recently Used)
//! caching and hash-based caching for predictions and probabilities.

use scirs2_core::ndarray::{Array1, Array2};
use std::collections::{hash_map::DefaultHasher, HashMap};
use std::hash::{Hash, Hasher};

/// A simple hash-based cache key for arrays
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ArrayKey {
    hash: u64,
    shape: Vec<usize>,
}

impl ArrayKey {
    /// Create a new array key from a 2D array
    pub fn from_array2(arr: &Array2<f64>) -> Self {
        let mut hasher = DefaultHasher::new();

        // Hash the shape
        arr.shape().hash(&mut hasher);

        // Hash a subset of the data to balance performance vs collision risk
        let (nrows, ncols) = arr.dim();
        let sample_size = (nrows * ncols).min(1000); // Sample up to 1000 elements

        for (i, &value) in arr.iter().enumerate() {
            if i >= sample_size {
                break;
            }
            // Convert to bits to handle NaN and infinity consistently
            value.to_bits().hash(&mut hasher);
        }

        let hash = hasher.finish();

        Self {
            hash,
            shape: arr.shape().to_vec(),
        }
    }
}

/// Configuration for prediction caching
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Maximum number of entries in the cache
    pub max_entries: usize,
    /// Whether to enable caching
    pub enabled: bool,
    /// Whether to use approximate matching (for floating point tolerance)
    pub approximate_matching: bool,
    /// Tolerance for approximate matching
    pub tolerance: f64,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            max_entries: 1000,
            enabled: true,
            approximate_matching: false,
            tolerance: 1e-10,
        }
    }
}

/// A simple LRU cache for predictions
#[derive(Debug)]
pub struct PredictionCache {
    /// Cache for predict() results
    predict_cache: HashMap<ArrayKey, Array1<i32>>,
    /// Cache for predict_proba() results
    proba_cache: HashMap<ArrayKey, Array2<f64>>,
    /// Configuration
    config: CacheConfig,
    /// Access order for LRU eviction
    access_order: Vec<ArrayKey>,
}

impl PredictionCache {
    /// Create a new prediction cache with default configuration
    pub fn new() -> Self {
        Self::with_config(CacheConfig::default())
    }

    /// Create a new prediction cache with specified configuration
    pub fn with_config(config: CacheConfig) -> Self {
        Self {
            predict_cache: HashMap::new(),
            proba_cache: HashMap::new(),
            config,
            access_order: Vec::new(),
        }
    }

    /// Get cached prediction if available
    pub fn get_prediction(&mut self, X: &Array2<f64>) -> Option<Array1<i32>> {
        if !self.config.enabled {
            return None;
        }

        let key = ArrayKey::from_array2(X);

        if let Some(prediction) = self.predict_cache.get(&key).cloned() {
            // Update access order
            self.update_access_order(&key);
            Some(prediction)
        } else {
            None
        }
    }

    /// Cache a prediction
    pub fn cache_prediction(&mut self, X: &Array2<f64>, prediction: Array1<i32>) {
        if !self.config.enabled {
            return;
        }

        let key = ArrayKey::from_array2(X);

        // Evict if necessary
        self.maybe_evict();

        self.predict_cache.insert(key.clone(), prediction);
        self.update_access_order(&key);
    }

    /// Get cached probability predictions if available
    pub fn get_prediction_proba(&mut self, X: &Array2<f64>) -> Option<Array2<f64>> {
        if !self.config.enabled {
            return None;
        }

        let key = ArrayKey::from_array2(X);

        if let Some(proba) = self.proba_cache.get(&key).cloned() {
            // Update access order
            self.update_access_order(&key);
            Some(proba)
        } else {
            None
        }
    }

    /// Cache probability predictions
    pub fn cache_prediction_proba(&mut self, X: &Array2<f64>, proba: Array2<f64>) {
        if !self.config.enabled {
            return;
        }

        let key = ArrayKey::from_array2(X);

        // Evict if necessary
        self.maybe_evict();

        self.proba_cache.insert(key.clone(), proba);
        self.update_access_order(&key);
    }

    /// Clear all cached entries
    pub fn clear(&mut self) {
        self.predict_cache.clear();
        self.proba_cache.clear();
        self.access_order.clear();
    }

    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        CacheStats {
            predict_entries: self.predict_cache.len(),
            proba_entries: self.proba_cache.len(),
            total_entries: self.predict_cache.len() + self.proba_cache.len(),
            max_entries: self.config.max_entries,
            enabled: self.config.enabled,
        }
    }

    /// Update access order for LRU eviction
    fn update_access_order(&mut self, key: &ArrayKey) {
        // Remove if already present
        if let Some(pos) = self.access_order.iter().position(|k| k == key) {
            self.access_order.remove(pos);
        }
        // Add to front (most recent)
        self.access_order.insert(0, key.clone());
    }

    /// Evict entries if cache is full
    fn maybe_evict(&mut self) {
        let total_entries = self.predict_cache.len() + self.proba_cache.len();

        if total_entries >= self.config.max_entries {
            // Remove least recently used entry
            if let Some(lru_key) = self.access_order.pop() {
                self.predict_cache.remove(&lru_key);
                self.proba_cache.remove(&lru_key);
            }
        }
    }
}

impl Default for PredictionCache {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about cache usage
#[derive(Debug, Clone)]
pub struct CacheStats {
    /// Number of prediction cache entries
    pub predict_entries: usize,
    /// Number of probability cache entries
    pub proba_entries: usize,
    /// Total number of cache entries
    pub total_entries: usize,
    /// Maximum allowed entries
    pub max_entries: usize,
    /// Whether caching is enabled
    pub enabled: bool,
}

impl CacheStats {
    /// Calculate cache usage as a percentage
    pub fn usage_percentage(&self) -> f64 {
        if self.max_entries == 0 {
            0.0
        } else {
            (self.total_entries as f64 / self.max_entries as f64) * 100.0
        }
    }

    /// Check if cache is nearly full (>90% usage)
    pub fn is_nearly_full(&self) -> bool {
        self.usage_percentage() > 90.0
    }
}

/// A wrapper that adds caching capabilities to any classifier
#[derive(Debug)]
pub struct CachedClassifier<T> {
    /// The underlying classifier
    pub classifier: T,
    /// The prediction cache
    cache: PredictionCache,
}

impl<T> CachedClassifier<T> {
    /// Create a new cached classifier with default cache configuration
    pub fn new(classifier: T) -> Self {
        Self {
            classifier,
            cache: PredictionCache::new(),
        }
    }

    /// Create a new cached classifier with custom cache configuration
    pub fn with_cache_config(classifier: T, config: CacheConfig) -> Self {
        Self {
            classifier,
            cache: PredictionCache::with_config(config),
        }
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> CacheStats {
        self.cache.stats()
    }

    /// Clear the cache
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }

    /// Enable or disable caching
    pub fn set_cache_enabled(&mut self, enabled: bool) {
        self.cache.config.enabled = enabled;
    }
}

impl<T> CachedClassifier<T>
where
    T: sklears_core::traits::Predict<Array2<f64>, Array1<i32>>,
{
    /// Make predictions with caching
    pub fn predict_cached(&mut self, X: &Array2<f64>) -> sklears_core::error::Result<Array1<i32>> {
        // Try cache first
        if let Some(cached_result) = self.cache.get_prediction(X) {
            return Ok(cached_result);
        }

        // Not in cache, compute and cache result
        let prediction = self.classifier.predict(X)?;
        self.cache.cache_prediction(X, prediction.clone());

        Ok(prediction)
    }
}

impl<T> CachedClassifier<T>
where
    T: sklears_core::traits::PredictProba<Array2<f64>, Array2<f64>>,
{
    /// Make probability predictions with caching
    pub fn predict_proba_cached(
        &mut self,
        X: &Array2<f64>,
    ) -> sklears_core::error::Result<Array2<f64>> {
        // Try cache first
        if let Some(cached_result) = self.cache.get_prediction_proba(X) {
            return Ok(cached_result);
        }

        // Not in cache, compute and cache result
        let proba = self.classifier.predict_proba(X)?;
        self.cache.cache_prediction_proba(X, proba.clone());

        Ok(proba)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_autograd::ndarray::array;

    #[test]
    fn test_array_key_creation() {
        let arr = array![[1.0, 2.0], [3.0, 4.0]];
        let key1 = ArrayKey::from_array2(&arr);
        let key2 = ArrayKey::from_array2(&arr);

        assert_eq!(key1, key2);
        assert_eq!(key1.shape, vec![2, 2]);
    }

    #[test]
    fn test_cache_config_default() {
        let config = CacheConfig::default();
        assert_eq!(config.max_entries, 1000);
        assert!(config.enabled);
        assert!(!config.approximate_matching);
        assert_eq!(config.tolerance, 1e-10);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_prediction_cache_basic() {
        let mut cache = PredictionCache::new();
        let X = array![[1.0, 2.0], [3.0, 4.0]];
        let pred = array![0, 1];

        // Initially empty
        assert!(cache.get_prediction(&X).is_none());

        // Cache and retrieve
        cache.cache_prediction(&X, pred.clone());
        let cached = cache.get_prediction(&X);
        assert!(cached.is_some());
        assert_eq!(cached.unwrap(), pred);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_prediction_cache_proba() {
        let mut cache = PredictionCache::new();
        let X = array![[1.0, 2.0], [3.0, 4.0]];
        let proba = array![[0.8, 0.2], [0.3, 0.7]];

        // Initially empty
        assert!(cache.get_prediction_proba(&X).is_none());

        // Cache and retrieve
        cache.cache_prediction_proba(&X, proba.clone());
        let cached = cache.get_prediction_proba(&X);
        assert!(cached.is_some());
        assert_eq!(cached.unwrap(), proba);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_cache_disabled() {
        let config = CacheConfig {
            enabled: false,
            ..CacheConfig::default()
        };
        let mut cache = PredictionCache::with_config(config);
        let X = array![[1.0, 2.0]];
        let pred = array![0];

        cache.cache_prediction(&X, pred);
        assert!(cache.get_prediction(&X).is_none());
    }

    #[test]
    fn test_cache_eviction() {
        let config = CacheConfig {
            max_entries: 2,
            ..CacheConfig::default()
        };
        let mut cache = PredictionCache::with_config(config);

        let X1 = array![[1.0, 2.0]];
        let X2 = array![[3.0, 4.0]];
        let X3 = array![[5.0, 6.0]];

        let pred1 = array![0];
        let pred2 = array![1];
        let pred3 = array![2];

        // Fill cache
        cache.cache_prediction(&X1, pred1.clone());
        cache.cache_prediction(&X2, pred2.clone());

        // Both should be in cache
        assert!(cache.get_prediction(&X1).is_some());
        assert!(cache.get_prediction(&X2).is_some());

        // Add third item (should evict first)
        cache.cache_prediction(&X3, pred3.clone());

        // First should be evicted
        assert!(cache.get_prediction(&X1).is_none());
        assert!(cache.get_prediction(&X2).is_some());
        assert!(cache.get_prediction(&X3).is_some());
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_cache_stats() {
        let mut cache = PredictionCache::new();
        let X = array![[1.0, 2.0]];
        let pred = array![0];
        let proba = array![[0.8, 0.2]];

        let stats = cache.stats();
        assert_eq!(stats.total_entries, 0);
        assert!(!stats.is_nearly_full());

        cache.cache_prediction(&X, pred);
        cache.cache_prediction_proba(&X, proba);

        let stats = cache.stats();
        assert_eq!(stats.predict_entries, 1);
        assert_eq!(stats.proba_entries, 1);
        assert_eq!(stats.total_entries, 2);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_cache_clear() {
        let mut cache = PredictionCache::new();
        let X = array![[1.0, 2.0]];
        let pred = array![0];

        cache.cache_prediction(&X, pred);
        assert!(cache.get_prediction(&X).is_some());

        cache.clear();
        assert!(cache.get_prediction(&X).is_none());
        assert_eq!(cache.stats().total_entries, 0);
    }
}
