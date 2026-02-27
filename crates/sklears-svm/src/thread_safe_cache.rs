//! Thread-safe kernel caching for parallel SVM processing
//!
//! This module provides thread-safe caching implementations for kernel computations
//! that enable efficient parallel processing of SVM algorithms while maintaining
//! cache coherence and avoiding race conditions.

use crate::kernels::Kernel;
use dashmap::DashMap;
use parking_lot::Mutex as ParkingMutex;
use scirs2_core::ndarray::{Array2, ArrayView1};
use scirs2_core::rand_prelude::IndexedRandom;
use sklears_core::error::{Result, SklearsError};
use sklears_core::types::Float;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;

/// Configuration for thread-safe kernel cache
#[derive(Debug, Clone)]
pub struct ThreadSafeKernelCacheConfig {
    /// Maximum number of cached kernel values
    pub max_cache_size: usize,
    /// Number of cache shards for reducing contention
    pub num_shards: usize,
    /// Cache eviction strategy
    pub eviction_strategy: EvictionStrategy,
    /// Whether to enable cache statistics
    pub enable_stats: bool,
    /// Preallocation size for cache maps
    pub prealloc_size: usize,
    /// Concurrency level for DashMap
    pub concurrency_level: usize,
}

impl Default for ThreadSafeKernelCacheConfig {
    fn default() -> Self {
        #[cfg(feature = "parallel")]
        let default_threads = rayon::current_num_threads();
        #[cfg(not(feature = "parallel"))]
        let default_threads = num_cpus::get();

        Self {
            max_cache_size: 100000,
            num_shards: default_threads,
            eviction_strategy: EvictionStrategy::LeastRecentlyUsed,
            enable_stats: true,
            prealloc_size: 1000,
            concurrency_level: default_threads,
        }
    }
}

/// Cache eviction strategies
#[derive(Debug, Clone, Copy)]
pub enum EvictionStrategy {
    /// Least Recently Used eviction
    LeastRecentlyUsed,
    /// Least Frequently Used eviction
    LeastFrequentlyUsed,
    /// Random eviction
    Random,
    /// First In First Out eviction
    FirstInFirstOut,
}

/// Thread-safe kernel cache using multiple strategies
pub trait ThreadSafeKernelCache: Send + Sync {
    /// Get cached kernel value
    fn get(&self, key: &KernelCacheKey) -> Option<Float>;

    /// Insert kernel value into cache
    fn insert(&self, key: KernelCacheKey, value: Float);

    /// Clear the cache
    fn clear(&self);

    /// Get cache statistics
    fn stats(&self) -> CacheStatistics;

    /// Get current cache size
    fn size(&self) -> usize;
}

/// Cache key for kernel values
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct KernelCacheKey {
    /// First sample index (always <= second_index for canonical ordering)
    pub first_index: usize,
    /// Second sample index
    pub second_index: usize,
    /// Hash of kernel parameters for differentiation
    pub kernel_hash: u64,
}

impl KernelCacheKey {
    /// Create a new cache key with canonical ordering
    pub fn new(i: usize, j: usize, kernel_hash: u64) -> Self {
        let (first_index, second_index) = if i <= j { (i, j) } else { (j, i) };
        Self {
            first_index,
            second_index,
            kernel_hash,
        }
    }
}

/// Cache statistics for monitoring performance
#[derive(Debug, Clone)]
pub struct CacheStatistics {
    pub hits: u64,
    pub misses: u64,
    pub insertions: u64,
    pub evictions: u64,
    pub current_size: usize,
    pub max_size: usize,
    pub hit_rate: f64,
}

impl Default for CacheStatistics {
    fn default() -> Self {
        Self::new()
    }
}

impl CacheStatistics {
    pub fn new() -> Self {
        Self {
            hits: 0,
            misses: 0,
            insertions: 0,
            evictions: 0,
            current_size: 0,
            max_size: 0,
            hit_rate: 0.0,
        }
    }

    pub fn update_hit_rate(&mut self) {
        let total = self.hits + self.misses;
        self.hit_rate = if total > 0 {
            self.hits as f64 / total as f64
        } else {
            0.0
        };
    }
}

/// DashMap-based thread-safe cache with high concurrency
pub struct DashMapKernelCache {
    cache: DashMap<KernelCacheKey, CacheEntry>,
    config: ThreadSafeKernelCacheConfig,
    stats: Arc<ParkingMutex<CacheStatistics>>,
    current_size: AtomicUsize,
}

#[derive(Debug)]
struct CacheEntry {
    value: Float,
    access_count: AtomicU64,
    insertion_order: u64,
    last_access: AtomicU64,
}

impl Clone for CacheEntry {
    fn clone(&self) -> Self {
        Self {
            value: self.value,
            access_count: AtomicU64::new(self.access_count.load(Ordering::Relaxed)),
            insertion_order: self.insertion_order,
            last_access: AtomicU64::new(self.last_access.load(Ordering::Relaxed)),
        }
    }
}

impl DashMapKernelCache {
    /// Create a new DashMap-based kernel cache
    pub fn new(config: ThreadSafeKernelCacheConfig) -> Self {
        let cache = DashMap::with_capacity_and_hasher(
            config.prealloc_size,
            std::collections::hash_map::RandomState::new(),
        );

        Self {
            cache,
            config,
            stats: Arc::new(ParkingMutex::new(CacheStatistics::new())),
            current_size: AtomicUsize::new(0),
        }
    }

    /// Evict entries based on the configured strategy
    fn evict_if_needed(&self) {
        let current_size = self.current_size.load(Ordering::Relaxed);
        if current_size >= self.config.max_cache_size {
            let entries_to_evict = current_size - self.config.max_cache_size + 1;
            self.evict_entries(entries_to_evict);
        }
    }

    /// Evict specified number of entries
    fn evict_entries(&self, count: usize) {
        match self.config.eviction_strategy {
            EvictionStrategy::LeastRecentlyUsed => self.evict_lru(count),
            EvictionStrategy::LeastFrequentlyUsed => self.evict_lfu(count),
            EvictionStrategy::Random => self.evict_random(count),
            EvictionStrategy::FirstInFirstOut => self.evict_fifo(count),
        }
    }

    /// Evict least recently used entries
    fn evict_lru(&self, count: usize) {
        let mut entries_to_remove = Vec::new();

        // Collect entries with their last access times
        for entry in self.cache.iter() {
            let last_access = entry.value().last_access.load(Ordering::Relaxed);
            entries_to_remove.push((entry.key().clone(), last_access));
        }

        // Sort by last access time (ascending) and remove oldest
        entries_to_remove.sort_by_key(|(_, last_access)| *last_access);

        let mut evicted = 0;
        for (key, _) in entries_to_remove.into_iter().take(count) {
            if self.cache.remove(&key).is_some() {
                self.current_size.fetch_sub(1, Ordering::Relaxed);
                evicted += 1;
            }
        }

        if self.config.enable_stats {
            let mut stats = self.stats.lock();
            stats.evictions += evicted;
        }
    }

    /// Evict least frequently used entries
    fn evict_lfu(&self, count: usize) {
        let mut entries_to_remove = Vec::new();

        // Collect entries with their access counts
        for entry in self.cache.iter() {
            let access_count = entry.value().access_count.load(Ordering::Relaxed);
            entries_to_remove.push((entry.key().clone(), access_count));
        }

        // Sort by access count (ascending) and remove least used
        entries_to_remove.sort_by_key(|(_, access_count)| *access_count);

        let mut evicted = 0;
        for (key, _) in entries_to_remove.into_iter().take(count) {
            if self.cache.remove(&key).is_some() {
                self.current_size.fetch_sub(1, Ordering::Relaxed);
                evicted += 1;
            }
        }

        if self.config.enable_stats {
            let mut stats = self.stats.lock();
            stats.evictions += evicted;
        }
    }

    /// Evict random entries
    fn evict_random(&self, count: usize) {
        let keys: Vec<_> = self.cache.iter().map(|entry| entry.key().clone()).collect();
        let mut rng = scirs2_core::random::thread_rng();
        let keys_to_remove: Vec<_> = keys
            .as_slice()
            .choose_multiple(&mut rng, count)
            .cloned()
            .collect();

        let mut evicted = 0;
        for key in keys_to_remove {
            if self.cache.remove(&key).is_some() {
                self.current_size.fetch_sub(1, Ordering::Relaxed);
                evicted += 1;
            }
        }

        if self.config.enable_stats {
            let mut stats = self.stats.lock();
            stats.evictions += evicted;
        }
    }

    /// Evict first in first out entries
    fn evict_fifo(&self, count: usize) {
        let mut entries_to_remove = Vec::new();

        // Collect entries with their insertion order
        for entry in self.cache.iter() {
            let insertion_order = entry.value().insertion_order;
            entries_to_remove.push((entry.key().clone(), insertion_order));
        }

        // Sort by insertion order (ascending) and remove oldest
        entries_to_remove.sort_by_key(|(_, insertion_order)| *insertion_order);

        let mut evicted = 0;
        for (key, _) in entries_to_remove.into_iter().take(count) {
            if self.cache.remove(&key).is_some() {
                self.current_size.fetch_sub(1, Ordering::Relaxed);
                evicted += 1;
            }
        }

        if self.config.enable_stats {
            let mut stats = self.stats.lock();
            stats.evictions += evicted;
        }
    }
}

impl ThreadSafeKernelCache for DashMapKernelCache {
    fn get(&self, key: &KernelCacheKey) -> Option<Float> {
        if let Some(entry) = self.cache.get(key) {
            entry.access_count.fetch_add(1, Ordering::Relaxed);
            entry.last_access.store(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos() as u64,
                Ordering::Relaxed,
            );

            if self.config.enable_stats {
                let mut stats = self.stats.lock();
                stats.hits += 1;
                stats.update_hit_rate();
            }

            Some(entry.value)
        } else {
            if self.config.enable_stats {
                let mut stats = self.stats.lock();
                stats.misses += 1;
                stats.update_hit_rate();
            }
            None
        }
    }

    fn insert(&self, key: KernelCacheKey, value: Float) {
        self.evict_if_needed();

        let entry = CacheEntry {
            value,
            access_count: AtomicU64::new(1),
            insertion_order: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64,
            last_access: AtomicU64::new(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos() as u64,
            ),
        };

        if self.cache.insert(key, entry).is_none() {
            self.current_size.fetch_add(1, Ordering::Relaxed);
        }

        if self.config.enable_stats {
            let mut stats = self.stats.lock();
            stats.insertions += 1;
            stats.current_size = self.current_size.load(Ordering::Relaxed);
            stats.max_size = self.config.max_cache_size;
        }
    }

    fn clear(&self) {
        self.cache.clear();
        self.current_size.store(0, Ordering::Relaxed);

        if self.config.enable_stats {
            let mut stats = self.stats.lock();
            *stats = CacheStatistics::new();
        }
    }

    fn stats(&self) -> CacheStatistics {
        if self.config.enable_stats {
            let mut stats = self.stats.lock();
            stats.current_size = self.current_size.load(Ordering::Relaxed);
            stats.max_size = self.config.max_cache_size;
            stats.clone()
        } else {
            CacheStatistics::new()
        }
    }

    fn size(&self) -> usize {
        self.current_size.load(Ordering::Relaxed)
    }
}

/// Sharded kernel cache for reduced contention
pub struct ShardedKernelCache {
    shards: Vec<Arc<DashMapKernelCache>>,
    num_shards: usize,
}

impl ShardedKernelCache {
    /// Create a new sharded kernel cache
    pub fn new(config: ThreadSafeKernelCacheConfig) -> Self {
        let shard_size = config.max_cache_size / config.num_shards;
        let mut shard_config = config.clone();
        shard_config.max_cache_size = shard_size;

        let shards = (0..config.num_shards)
            .map(|_| Arc::new(DashMapKernelCache::new(shard_config.clone())))
            .collect();

        Self {
            shards,
            num_shards: config.num_shards,
        }
    }

    /// Get shard index for a key
    fn get_shard_index(&self, key: &KernelCacheKey) -> usize {
        use std::collections::hash_map::DefaultHasher;

        let mut hasher = DefaultHasher::new();
        key.hash(&mut hasher);
        (hasher.finish() as usize) % self.num_shards
    }
}

impl ThreadSafeKernelCache for ShardedKernelCache {
    fn get(&self, key: &KernelCacheKey) -> Option<Float> {
        let shard_index = self.get_shard_index(key);
        self.shards[shard_index].get(key)
    }

    fn insert(&self, key: KernelCacheKey, value: Float) {
        let shard_index = self.get_shard_index(&key);
        self.shards[shard_index].insert(key, value);
    }

    fn clear(&self) {
        for shard in &self.shards {
            shard.clear();
        }
    }

    fn stats(&self) -> CacheStatistics {
        let mut combined_stats = CacheStatistics::new();

        for shard in &self.shards {
            let shard_stats = shard.stats();
            combined_stats.hits += shard_stats.hits;
            combined_stats.misses += shard_stats.misses;
            combined_stats.insertions += shard_stats.insertions;
            combined_stats.evictions += shard_stats.evictions;
            combined_stats.current_size += shard_stats.current_size;
            combined_stats.max_size += shard_stats.max_size;
        }

        combined_stats.update_hit_rate();
        combined_stats
    }

    fn size(&self) -> usize {
        self.shards.iter().map(|shard| shard.size()).sum()
    }
}

/// Cached kernel wrapper that automatically handles caching
pub struct CachedKernel {
    inner_kernel: Box<dyn Kernel>,
    cache: Arc<dyn ThreadSafeKernelCache>,
    kernel_hash: u64,
    x_data: Option<Array2<Float>>, // Store data for index-based access
}

impl std::fmt::Debug for CachedKernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CachedKernel")
            .field("inner_kernel", &self.inner_kernel)
            .field("kernel_hash", &self.kernel_hash)
            .field("x_data", &self.x_data.as_ref().map(|x| x.dim()))
            .finish()
    }
}

impl CachedKernel {
    /// Create a new cached kernel
    pub fn new(kernel: Box<dyn Kernel>, cache: Arc<dyn ThreadSafeKernelCache>) -> Self {
        // Generate hash for kernel parameters
        let kernel_debug = format!("{kernel:?}");
        let kernel_hash = Self::compute_kernel_hash_from_string(&kernel_debug);

        Self {
            inner_kernel: kernel,
            cache,
            kernel_hash,
            x_data: None,
        }
    }

    /// Set data for index-based caching
    pub fn set_data(&mut self, x: Array2<Float>) {
        self.x_data = Some(x);
    }

    /// Compute kernel value with caching by indices
    pub fn compute_by_indices(&self, i: usize, j: usize) -> Result<Float> {
        let key = KernelCacheKey::new(i, j, self.kernel_hash);

        if let Some(cached_value) = self.cache.get(&key) {
            return Ok(cached_value);
        }

        if let Some(ref x) = self.x_data {
            let value = self
                .inner_kernel
                .compute(x.row(i).to_owned().view(), x.row(j).to_owned().view());
            self.cache.insert(key, value);
            Ok(value)
        } else {
            Err(SklearsError::InvalidInput(
                "No data set for index-based computation".to_string(),
            ))
        }
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> CacheStatistics {
        self.cache.stats()
    }

    /// Clear cache
    pub fn clear_cache(&self) {
        self.cache.clear();
    }

    /// Compute hash for kernel parameters from string
    fn compute_kernel_hash_from_string(kernel_debug: &str) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::Hasher;

        let mut hasher = DefaultHasher::new();
        kernel_debug.hash(&mut hasher);
        hasher.finish()
    }
}

impl Kernel for CachedKernel {
    fn compute(&self, x: ArrayView1<f64>, y: ArrayView1<f64>) -> f64 {
        // For direct vector computation, we can't use index-based caching
        // So we compute directly
        self.inner_kernel.compute(x, y)
    }

    fn compute_matrix(&self, x: &Array2<f64>, y: &Array2<f64>) -> Array2<f64> {
        let n_x = x.nrows();
        let n_y = y.nrows();
        let mut kernel_matrix = Array2::<f64>::zeros((n_x, n_y));

        // Use cached computation for matrix
        for i in 0..n_x {
            for j in 0..n_y {
                let key = KernelCacheKey::new(i, j, self.kernel_hash);

                let k_val = if let Some(cached_value) = self.cache.get(&key) {
                    cached_value
                } else {
                    let value = self.inner_kernel.compute(x.row(i), y.row(j));
                    self.cache.insert(key, value);
                    value
                };

                kernel_matrix[[i, j]] = k_val;
            }
        }

        kernel_matrix
    }

    fn parameters(&self) -> HashMap<String, f64> {
        // Delegate to inner kernel
        self.inner_kernel.parameters()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::LinearKernel;
    use scirs2_core::ndarray::Array1;

    #[test]
    fn test_dashmap_cache_basic_operations() {
        let config = ThreadSafeKernelCacheConfig::default();
        let cache = DashMapKernelCache::new(config);

        let key = KernelCacheKey::new(0, 1, 12345);
        cache.insert(key.clone(), 1.5);

        assert_eq!(cache.get(&key), Some(1.5));
        assert_eq!(cache.size(), 1);
    }

    #[test]
    fn test_sharded_cache() {
        let config = ThreadSafeKernelCacheConfig {
            num_shards: 4,
            ..ThreadSafeKernelCacheConfig::default()
        };
        let cache = ShardedKernelCache::new(config);

        let key = KernelCacheKey::new(0, 1, 12345);
        cache.insert(key.clone(), 2.5);

        assert_eq!(cache.get(&key), Some(2.5));
    }

    #[test]
    fn test_cached_kernel() {
        let kernel = Box::new(LinearKernel);
        let cache_config = ThreadSafeKernelCacheConfig::default();
        let cache = Arc::new(DashMapKernelCache::new(cache_config));

        let cached_kernel = CachedKernel::new(kernel, cache);

        let x1 = Array1::from(vec![1.0, 2.0, 3.0]);
        let x2 = Array1::from(vec![2.0, 3.0, 4.0]);

        let result1 = cached_kernel.compute(x1.view(), x2.view());
        let result2 = cached_kernel.compute(x1.view(), x2.view());

        assert_eq!(result1, result2);
    }

    #[test]
    fn test_cache_eviction() {
        let config = ThreadSafeKernelCacheConfig {
            max_cache_size: 2,
            ..ThreadSafeKernelCacheConfig::default()
        };
        let cache = DashMapKernelCache::new(config);

        // Insert 3 items to trigger eviction
        cache.insert(KernelCacheKey::new(0, 1, 1), 1.0);
        cache.insert(KernelCacheKey::new(1, 2, 1), 2.0);
        cache.insert(KernelCacheKey::new(2, 3, 1), 3.0);

        // Cache should not exceed max size
        assert!(cache.size() <= 2);
    }
}
