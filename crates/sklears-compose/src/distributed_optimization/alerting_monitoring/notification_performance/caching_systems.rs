//! Caching Systems Module
//!
//! This module provides comprehensive caching capabilities including multiple cache types,
//! eviction policies, access tracking, and performance optimization for notification
//! channel performance enhancement.

use std::collections::{HashMap, VecDeque};
use std::time::{Duration, SystemTime};
use serde::{Deserialize, Serialize};

use super::performance_core::CompressionAlgorithm;

/// Cache manager for performance optimization
#[derive(Debug, Clone)]
pub struct CacheManager {
    /// Cache instances
    pub caches: HashMap<String, CacheInstance>,
    /// Cache statistics
    pub statistics: CacheStatistics,
    /// Cache configuration
    pub config: CacheManagerConfig,
    /// Cache eviction policies
    pub eviction_policies: HashMap<String, EvictionPolicy>,
}

/// Cache instance
#[derive(Debug, Clone)]
pub struct CacheInstance {
    /// Cache identifier
    pub cache_id: String,
    /// Cache type
    pub cache_type: CacheType,
    /// Cache data
    pub data: HashMap<String, CacheEntry>,
    /// Cache configuration
    pub config: CacheConfig,
    /// Cache statistics
    pub statistics: CacheInstanceStatistics,
    /// Access tracker
    pub access_tracker: AccessTracker,
}

/// Cache types
#[derive(Debug, Clone)]
pub enum CacheType {
    Memory,
    Disk,
    Hybrid,
    Distributed,
    Custom(String),
}

/// Cache entry with metadata
#[derive(Debug, Clone)]
pub struct CacheEntry {
    /// Entry key
    pub key: String,
    /// Entry value
    pub value: Vec<u8>,
    /// Entry metadata
    pub metadata: CacheEntryMetadata,
    /// Compression information
    pub compression: Option<CompressionInfo>,
}

/// Cache entry metadata
#[derive(Debug, Clone)]
pub struct CacheEntryMetadata {
    /// Creation timestamp
    pub created_at: SystemTime,
    /// Last accessed timestamp
    pub last_accessed: SystemTime,
    /// Access count
    pub access_count: u64,
    /// Expiration timestamp
    pub expires_at: Option<SystemTime>,
    /// Entry size
    pub size: usize,
    /// Entry priority
    pub priority: CachePriority,
}

/// Cache priority levels
#[derive(Debug, Clone)]
pub enum CachePriority {
    Low,
    Normal,
    High,
    Critical,
}

/// Compression information
#[derive(Debug, Clone)]
pub struct CompressionInfo {
    /// Compression algorithm used
    pub algorithm: CompressionAlgorithm,
    /// Original size
    pub original_size: usize,
    /// Compressed size
    pub compressed_size: usize,
    /// Compression ratio
    pub compression_ratio: f64,
    /// Compression time
    pub compression_time: Duration,
}

/// Cache configuration
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Maximum cache size
    pub max_size: usize,
    /// Default TTL
    pub default_ttl: Duration,
    /// Enable compression
    pub enable_compression: bool,
    /// Compression threshold
    pub compression_threshold: usize,
    /// Eviction policy
    pub eviction_policy: EvictionPolicy,
}

/// Cache eviction policies
#[derive(Debug, Clone)]
pub enum EvictionPolicy {
    LRU,
    LFU,
    FIFO,
    TTL,
    Random,
    Priority,
    Custom(String),
}

/// Cache instance statistics
#[derive(Debug, Clone)]
pub struct CacheInstanceStatistics {
    /// Cache hits
    pub hits: u64,
    /// Cache misses
    pub misses: u64,
    /// Hit ratio
    pub hit_ratio: f64,
    /// Total entries
    pub total_entries: usize,
    /// Total size
    pub total_size: usize,
    /// Evictions
    pub evictions: u64,
    /// Average access time
    pub avg_access_time: Duration,
}

/// Access tracker for cache optimization
#[derive(Debug, Clone)]
pub struct AccessTracker {
    /// Access patterns
    pub access_patterns: HashMap<String, AccessPattern>,
    /// Hot keys
    pub hot_keys: VecDeque<String>,
    /// Cold keys
    pub cold_keys: VecDeque<String>,
    /// Access statistics
    pub statistics: AccessStatistics,
    /// Ordered history of recently accessed keys (most recent at the back).
    ///
    /// Bounded by [`AccessTracker::MAX_SEQUENCE_LEN`]. Used both for
    /// recency-weighted scoring and to derive first-order Markov transition
    /// statistics for next-key prediction.
    pub access_sequence: VecDeque<String>,
    /// First-order Markov transition counts.
    ///
    /// `transitions[a][b]` is the number of times key `b` was accessed
    /// immediately after key `a`. This is the empirical basis for predicting
    /// which key is likely to be requested next given the most recent access.
    pub transitions: HashMap<String, HashMap<String, u64>>,
}

/// Access pattern for cache optimization
#[derive(Debug, Clone)]
pub struct AccessPattern {
    /// Key identifier
    pub key: String,
    /// Access frequency
    pub frequency: f64,
    /// Access recency
    pub recency: f64,
    /// Access predictability
    pub predictability: f64,
    /// Access trend
    pub trend: AccessTrend,
}

/// Access trend analysis
#[derive(Debug, Clone)]
pub enum AccessTrend {
    Increasing,
    Decreasing,
    Stable,
    Sporadic,
}

/// Access statistics
#[derive(Debug, Clone)]
pub struct AccessStatistics {
    /// Total accesses
    pub total_accesses: u64,
    /// Unique keys accessed
    pub unique_keys: usize,
    /// Hot key threshold
    pub hot_key_threshold: f64,
    /// Cold key threshold
    pub cold_key_threshold: f64,
}

/// Cache manager statistics
#[derive(Debug, Clone)]
pub struct CacheStatistics {
    /// Total cache hits
    pub total_hits: u64,
    /// Total cache misses
    pub total_misses: u64,
    /// Global hit ratio
    pub global_hit_ratio: f64,
    /// Total cache size
    pub total_cache_size: usize,
    /// Cache efficiency
    pub cache_efficiency: f64,
    /// Memory usage
    pub memory_usage: usize,
}

/// Cache manager configuration
#[derive(Debug, Clone)]
pub struct CacheManagerConfig {
    /// Enable caching
    pub enabled: bool,
    /// Global cache size limit
    pub global_size_limit: usize,
    /// Cache cleanup interval
    pub cleanup_interval: Duration,
    /// Enable cache warming
    pub enable_cache_warming: bool,
    /// Cache warming strategy
    pub warming_strategy: CacheWarmingStrategy,
}

/// Cache warming strategies
#[derive(Debug, Clone)]
pub enum CacheWarmingStrategy {
    Preload,
    OnDemand,
    Predictive,
    Custom(String),
}

impl CacheManager {
    /// Create a new cache manager
    pub fn new() -> Self {
        Self {
            caches: HashMap::new(),
            statistics: CacheStatistics::default(),
            config: CacheManagerConfig::default(),
            eviction_policies: HashMap::new(),
        }
    }

    /// Create a cache instance
    pub fn create_cache(&mut self, cache_id: String, cache_type: CacheType) {
        let cache = CacheInstance::new(cache_id.clone(), cache_type);
        self.caches.insert(cache_id, cache);
    }

    /// Get cache instance
    pub fn get_cache(&self, cache_id: &str) -> Option<&CacheInstance> {
        self.caches.get(cache_id)
    }

    /// Get mutable cache instance
    pub fn get_cache_mut(&mut self, cache_id: &str) -> Option<&mut CacheInstance> {
        self.caches.get_mut(cache_id)
    }

    /// Remove cache instance
    pub fn remove_cache(&mut self, cache_id: &str) -> Option<CacheInstance> {
        self.caches.remove(cache_id)
    }

    /// Get global cache statistics
    pub fn get_global_statistics(&mut self) -> &CacheStatistics {
        // Update global statistics from all cache instances
        let mut total_hits = 0;
        let mut total_misses = 0;
        let mut total_size = 0;

        for cache in self.caches.values() {
            total_hits += cache.statistics.hits;
            total_misses += cache.statistics.misses;
            total_size += cache.statistics.total_size;
        }

        self.statistics.total_hits = total_hits;
        self.statistics.total_misses = total_misses;
        self.statistics.total_cache_size = total_size;

        if total_hits + total_misses > 0 {
            self.statistics.global_hit_ratio = total_hits as f64 / (total_hits + total_misses) as f64;
        }

        &self.statistics
    }

    /// Cleanup expired entries across all caches
    pub fn cleanup_expired_entries(&mut self) {
        let now = SystemTime::now();
        for cache in self.caches.values_mut() {
            cache.cleanup_expired(now);
        }
    }

    /// Optimize cache performance
    pub fn optimize_caches(&mut self) {
        for cache in self.caches.values_mut() {
            cache.optimize_performance();
        }
    }

    /// Predict which keys are most worth preloading for a cache.
    ///
    /// Delegates to the cache's [`AccessTracker`], combining recency-weighted
    /// frequency (LFU) with first-order Markov next-key prediction over the
    /// recorded access history. Keys already resident in the cache are excluded.
    ///
    /// Returns an honest `Err` if the cache does not exist. Returns an empty
    /// vector if there is no usable access history yet (rather than fabricating
    /// recommendations).
    pub fn predict_warm_keys(
        &self,
        cache_id: &str,
        max_keys: usize,
    ) -> Result<Vec<String>, String> {
        let cache = self
            .get_cache(cache_id)
            .ok_or_else(|| format!("Cache {cache_id} not found"))?;
        Ok(cache.predicted_preload_keys(max_keys))
    }

    /// Warm a cache by loading caller-supplied real key/value pairs.
    ///
    /// Cache warming cannot fabricate values: the data must come from the
    /// backing source the caller owns. This method inserts the provided
    /// `entries` (each already fetched from the source) into the cache so that
    /// subsequent reads hit warm. To decide *which* keys to fetch ahead of
    /// time, call [`CacheManager::predict_warm_keys`] first.
    ///
    /// Returns the number of entries actually loaded, or an honest `Err` if the
    /// cache does not exist.
    pub fn warm_cache(
        &mut self,
        cache_id: &str,
        entries: Vec<(String, Vec<u8>)>,
    ) -> Result<usize, String> {
        let cache = self
            .get_cache_mut(cache_id)
            .ok_or_else(|| format!("Cache {cache_id} not found"))?;

        let mut loaded = 0;
        for (key, value) in entries {
            if cache.preload_entry(key, value) {
                loaded += 1;
            }
        }
        Ok(loaded)
    }

    /// Set eviction policy for a cache
    pub fn set_eviction_policy(&mut self, cache_id: &str, policy: EvictionPolicy) -> Result<(), String> {
        let cache = self.get_cache_mut(cache_id)
            .ok_or_else(|| format!("Cache {} not found", cache_id))?;

        cache.config.eviction_policy = policy.clone();
        self.eviction_policies.insert(cache_id.to_string(), policy);
        Ok(())
    }
}

impl CacheInstance {
    /// Create a new cache instance
    pub fn new(cache_id: String, cache_type: CacheType) -> Self {
        Self {
            cache_id,
            cache_type,
            data: HashMap::new(),
            config: CacheConfig::default(),
            statistics: CacheInstanceStatistics::default(),
            access_tracker: AccessTracker::new(),
        }
    }

    /// Get value from cache
    pub fn get(&mut self, key: &str) -> Option<Vec<u8>> {
        if let Some(entry) = self.data.get_mut(key) {
            // Check if entry has expired
            if let Some(expires_at) = entry.metadata.expires_at {
                if SystemTime::now() > expires_at {
                    self.data.remove(key);
                    self.statistics.misses += 1;
                    return None;
                }
            }

            entry.metadata.last_accessed = SystemTime::now();
            entry.metadata.access_count += 1;
            self.statistics.hits += 1;

            // Update access tracker
            self.access_tracker.record_access(key);

            Some(entry.value.clone())
        } else {
            self.statistics.misses += 1;
            None
        }
    }

    /// Put value in cache
    pub fn put(&mut self, key: String, value: Vec<u8>) {
        let entry = CacheEntry {
            key: key.clone(),
            value,
            metadata: CacheEntryMetadata::new(self.config.default_ttl),
            compression: None,
        };

        // Check if eviction is needed
        if self.should_evict() {
            self.evict_entries(1);
        }

        self.data.insert(key, entry);
        self.statistics.total_entries = self.data.len();
        self.update_total_size();
    }

    /// Put value with custom TTL
    pub fn put_with_ttl(&mut self, key: String, value: Vec<u8>, ttl: Duration) {
        let mut metadata = CacheEntryMetadata::new(ttl);
        metadata.expires_at = Some(SystemTime::now() + ttl);

        let entry = CacheEntry {
            key: key.clone(),
            value,
            metadata,
            compression: None,
        };

        if self.should_evict() {
            self.evict_entries(1);
        }

        self.data.insert(key, entry);
        self.statistics.total_entries = self.data.len();
        self.update_total_size();
    }

    /// Remove entry from cache
    pub fn remove(&mut self, key: &str) -> Option<Vec<u8>> {
        if let Some(entry) = self.data.remove(key) {
            self.statistics.total_entries = self.data.len();
            self.update_total_size();
            Some(entry.value)
        } else {
            None
        }
    }

    /// Check if cache contains key
    pub fn contains_key(&self, key: &str) -> bool {
        if let Some(entry) = self.data.get(key) {
            // Check expiration
            if let Some(expires_at) = entry.metadata.expires_at {
                SystemTime::now() <= expires_at
            } else {
                true
            }
        } else {
            false
        }
    }

    /// Clear all entries from cache
    pub fn clear(&mut self) {
        self.data.clear();
        self.statistics.total_entries = 0;
        self.statistics.total_size = 0;
        self.access_tracker = AccessTracker::new();
    }

    /// Get cache size
    pub fn size(&self) -> usize {
        self.data.len()
    }

    /// Get hit ratio
    pub fn hit_ratio(&self) -> f64 {
        let total_requests = self.statistics.hits + self.statistics.misses;
        if total_requests > 0 {
            self.statistics.hits as f64 / total_requests as f64
        } else {
            0.0
        }
    }

    /// Cleanup expired entries
    pub fn cleanup_expired(&mut self, now: SystemTime) {
        let original_size = self.data.len();
        self.data.retain(|_, entry| {
            if let Some(expires_at) = entry.metadata.expires_at {
                now <= expires_at
            } else {
                true
            }
        });

        let removed_count = original_size - self.data.len();
        self.statistics.evictions += removed_count as u64;
        self.statistics.total_entries = self.data.len();
        self.update_total_size();
    }

    /// Optimize cache performance
    pub fn optimize_performance(&mut self) {
        // Update access patterns
        self.access_tracker.update_patterns();

        // Identify hot and cold keys
        self.access_tracker.identify_hot_cold_keys();

        // Apply eviction policy if needed
        if self.should_evict() {
            let evict_count = (self.data.len() as f64 * 0.1) as usize; // Evict 10% when needed
            self.evict_entries(evict_count);
        }
    }

    /// Recommend keys to preload, predicted from this cache's access history.
    ///
    /// Pure read-only recommendation: combines recency-weighted frequency with
    /// Markov next-key prediction (see [`AccessTracker::predict_preload_keys`]).
    /// Keys already present are excluded. The caller is responsible for fetching
    /// the actual values for the returned keys and passing them to
    /// [`CacheInstance::preload_entry`].
    pub fn predicted_preload_keys(&self, max_keys: usize) -> Vec<String> {
        let data = &self.data;
        self.access_tracker
            .predict_preload_keys(max_keys, &|key| data.contains_key(key))
    }

    /// Preload a single real key/value pair fetched from the backing source.
    ///
    /// Returns `true` if the entry was inserted, `false` if the key was already
    /// resident (warming never overwrites live data). No fabrication: the value
    /// must be supplied by the caller from the real data source.
    pub fn preload_entry(&mut self, key: String, value: Vec<u8>) -> bool {
        if self.data.contains_key(&key) {
            return false;
        }
        self.put(key, value);
        true
    }

    fn should_evict(&self) -> bool {
        self.data.len() >= self.config.max_size ||
        self.statistics.total_size >= self.config.max_size * 1024 // Assume max_size is in KB
    }

    fn evict_entries(&mut self, count: usize) {
        match self.config.eviction_policy {
            EvictionPolicy::LRU => self.evict_lru(count),
            EvictionPolicy::LFU => self.evict_lfu(count),
            EvictionPolicy::FIFO => self.evict_fifo(count),
            EvictionPolicy::TTL => self.evict_ttl(count),
            EvictionPolicy::Random => self.evict_random(count),
            EvictionPolicy::Priority => self.evict_priority(count),
            EvictionPolicy::Custom(_) => self.evict_lru(count), // Fallback to LRU
        }

        self.statistics.evictions += count as u64;
        self.statistics.total_entries = self.data.len();
        self.update_total_size();
    }

    fn evict_lru(&mut self, count: usize) {
        // Rank by last-accessed time (oldest first), collecting owned keys so the
        // immutable borrow of `self.data` is released before we mutate it.
        let mut entries: Vec<(String, SystemTime)> = self
            .data
            .iter()
            .map(|(key, entry)| (key.clone(), entry.metadata.last_accessed))
            .collect();
        entries.sort_by_key(|(_, last_accessed)| *last_accessed);

        for (key, _) in entries.into_iter().take(count) {
            self.data.remove(&key);
        }
    }

    fn evict_lfu(&mut self, count: usize) {
        // Rank by access count (least frequently used first).
        let mut entries: Vec<(String, u64)> = self
            .data
            .iter()
            .map(|(key, entry)| (key.clone(), entry.metadata.access_count))
            .collect();
        entries.sort_by_key(|(_, access_count)| *access_count);

        for (key, _) in entries.into_iter().take(count) {
            self.data.remove(&key);
        }
    }

    fn evict_fifo(&mut self, count: usize) {
        // Rank by creation time (oldest first).
        let mut entries: Vec<(String, SystemTime)> = self
            .data
            .iter()
            .map(|(key, entry)| (key.clone(), entry.metadata.created_at))
            .collect();
        entries.sort_by_key(|(_, created_at)| *created_at);

        for (key, _) in entries.into_iter().take(count) {
            self.data.remove(&key);
        }
    }

    fn evict_ttl(&mut self, count: usize) {
        let now = SystemTime::now();
        let expired_keys: Vec<String> = self
            .data
            .iter()
            .filter_map(|(key, entry)| {
                if let Some(expires_at) = entry.metadata.expires_at {
                    if now > expires_at {
                        Some(key.clone())
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
            .take(count)
            .collect();

        for key in expired_keys {
            self.data.remove(&key);
        }
    }

    fn evict_random(&mut self, count: usize) {
        let keys: Vec<_> = self.data.keys().take(count).cloned().collect();
        for key in keys {
            self.data.remove(&key);
        }
    }

    fn evict_priority(&mut self, count: usize) {
        // Evict lowest-priority entries first.
        let mut entries: Vec<(String, u8)> = self
            .data
            .iter()
            .map(|(key, entry)| {
                let rank = match entry.metadata.priority {
                    CachePriority::Critical => 4,
                    CachePriority::High => 3,
                    CachePriority::Normal => 2,
                    CachePriority::Low => 1,
                };
                (key.clone(), rank)
            })
            .collect();
        entries.sort_by_key(|(_, rank)| *rank);

        for (key, _) in entries.into_iter().take(count) {
            self.data.remove(&key);
        }
    }

    fn update_total_size(&mut self) {
        self.statistics.total_size = self.data.values()
            .map(|entry| entry.metadata.size)
            .sum();
    }
}

impl AccessTracker {
    /// Maximum number of accesses retained in the ordered access history.
    ///
    /// Bounds the memory used by [`AccessTracker::access_sequence`] while still
    /// providing a long enough window for stable frequency and transition
    /// estimates.
    pub const MAX_SEQUENCE_LEN: usize = 4096;

    /// Create a new access tracker
    pub fn new() -> Self {
        Self {
            access_patterns: HashMap::new(),
            hot_keys: VecDeque::new(),
            cold_keys: VecDeque::new(),
            statistics: AccessStatistics::default(),
            access_sequence: VecDeque::new(),
            transitions: HashMap::new(),
        }
    }

    /// Record access to a key
    pub fn record_access(&mut self, key: &str) {
        let is_new_key = !self.access_patterns.contains_key(key);

        let pattern = self
            .access_patterns
            .entry(key.to_string())
            .or_insert_with(|| AccessPattern::new(key.to_string()));

        pattern.frequency += 1.0;
        pattern.recency = 1.0; // Reset recency on access
        self.statistics.total_accesses += 1;

        // Update unique keys count (the entry was absent before this access).
        if is_new_key {
            self.statistics.unique_keys += 1;
        }

        // Update the first-order Markov transition counts from the previous
        // access to this one. This records the empirical "what comes next"
        // distribution used by predictive cache warming.
        if let Some(previous) = self.access_sequence.back() {
            if previous != key {
                *self
                    .transitions
                    .entry(previous.clone())
                    .or_default()
                    .entry(key.to_string())
                    .or_insert(0) += 1;
            }
        }

        // Append to the bounded ordered history.
        self.access_sequence.push_back(key.to_string());
        while self.access_sequence.len() > Self::MAX_SEQUENCE_LEN {
            self.access_sequence.pop_front();
        }
    }

    /// Predict the set of keys most worth preloading from recorded access data.
    ///
    /// The selection blends two real, complementary signals derived purely from
    /// observed history (no fabricated values):
    ///
    /// 1. **Frequency (LFU)** — keys accessed most often are most likely to be
    ///    requested again, scored by their recorded [`AccessPattern::frequency`]
    ///    decayed by [`AccessPattern::recency`].
    /// 2. **Markov next-key prediction** — given the most recently accessed key,
    ///    the empirical transition distribution in [`AccessTracker::transitions`]
    ///    predicts which keys typically follow it; their conditional probability
    ///    boosts the score.
    ///
    /// Keys already resident (`already_cached`) are never returned. Results are
    /// ranked by combined score, highest first, and truncated to `max_keys`.
    /// Returns an empty vector only when there is genuinely no usable history.
    pub fn predict_preload_keys(
        &self,
        max_keys: usize,
        already_cached: &dyn Fn(&str) -> bool,
    ) -> Vec<String> {
        if max_keys == 0 || self.access_patterns.is_empty() {
            return Vec::new();
        }

        // Base score: recency-weighted frequency (LFU with recency decay).
        let max_frequency = self
            .access_patterns
            .values()
            .map(|p| p.frequency)
            .fold(0.0_f64, f64::max)
            .max(1.0);

        let mut scores: HashMap<String, f64> = HashMap::new();
        for (key, pattern) in &self.access_patterns {
            if already_cached(key) {
                continue;
            }
            // Normalised frequency in [0, 1] weighted by current recency.
            let lfu_score = (pattern.frequency / max_frequency) * pattern.recency.clamp(0.0, 1.0);
            scores.insert(key.clone(), lfu_score);
        }

        // Markov boost: conditional probability of the next key given the most
        // recent access. This is the actual predictive component.
        if let Some(last_key) = self.access_sequence.back() {
            if let Some(next_counts) = self.transitions.get(last_key) {
                let total: u64 = next_counts.values().copied().sum();
                if total > 0 {
                    for (next_key, &count) in next_counts {
                        if already_cached(next_key) {
                            continue;
                        }
                        let conditional = count as f64 / total as f64;
                        // Weight the transition signal strongly relative to the
                        // raw LFU base so an imminent, predictable access wins.
                        *scores.entry(next_key.clone()).or_insert(0.0) += 2.0 * conditional;
                    }
                }
            }
        }

        if scores.is_empty() {
            return Vec::new();
        }

        let mut ranked: Vec<(String, f64)> = scores.into_iter().collect();
        // Sort by score descending; break ties deterministically by key so the
        // output is stable across runs.
        ranked.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });

        ranked
            .into_iter()
            .take(max_keys)
            .map(|(key, _)| key)
            .collect()
    }

    /// Update access patterns
    pub fn update_patterns(&mut self) {
        for pattern in self.access_patterns.values_mut() {
            pattern.recency *= 0.95; // Decay recency over time
            pattern.update_trend();
        }
    }

    /// Identify hot and cold keys
    pub fn identify_hot_cold_keys(&mut self) {
        self.hot_keys.clear();
        self.cold_keys.clear();

        for (key, pattern) in &self.access_patterns {
            if pattern.frequency > self.statistics.hot_key_threshold {
                self.hot_keys.push_back(key.clone());
            } else if pattern.frequency < self.statistics.cold_key_threshold {
                self.cold_keys.push_back(key.clone());
            }
        }
    }
}

impl AccessPattern {
    /// Create a new access pattern
    pub fn new(key: String) -> Self {
        Self {
            key,
            frequency: 0.0,
            recency: 1.0,
            predictability: 0.5,
            trend: AccessTrend::Stable,
        }
    }

    /// Update access trend
    pub fn update_trend(&mut self) {
        // Simple trend analysis based on frequency changes
        // In a real implementation, this would analyze historical data
        if self.frequency > 10.0 {
            self.trend = AccessTrend::Increasing;
        } else if self.frequency < 1.0 {
            self.trend = AccessTrend::Decreasing;
        } else {
            self.trend = AccessTrend::Stable;
        }
    }
}

impl CacheEntryMetadata {
    /// Create new metadata with TTL
    pub fn new(ttl: Duration) -> Self {
        let now = SystemTime::now();
        Self {
            created_at: now,
            last_accessed: now,
            access_count: 0,
            expires_at: Some(now + ttl),
            size: 0,
            priority: CachePriority::Normal,
        }
    }
}

// Default implementations
impl Default for CacheStatistics {
    fn default() -> Self {
        Self {
            total_hits: 0,
            total_misses: 0,
            global_hit_ratio: 0.0,
            total_cache_size: 0,
            cache_efficiency: 0.0,
            memory_usage: 0,
        }
    }
}

impl Default for CacheManagerConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            global_size_limit: 100_000_000, // 100MB
            cleanup_interval: Duration::from_secs(300), // 5 minutes
            enable_cache_warming: true,
            warming_strategy: CacheWarmingStrategy::OnDemand,
        }
    }
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            max_size: 10_000,
            default_ttl: Duration::from_secs(300), // 5 minutes
            enable_compression: false,
            compression_threshold: 1024, // 1KB
            eviction_policy: EvictionPolicy::LRU,
        }
    }
}

impl Default for CacheInstanceStatistics {
    fn default() -> Self {
        Self {
            hits: 0,
            misses: 0,
            hit_ratio: 0.0,
            total_entries: 0,
            total_size: 0,
            evictions: 0,
            avg_access_time: Duration::from_millis(0),
        }
    }
}

impl Default for AccessStatistics {
    fn default() -> Self {
        Self {
            total_accesses: 0,
            unique_keys: 0,
            hot_key_threshold: 10.0,
            cold_key_threshold: 1.0,
        }
    }
}

#[cfg(test)]
mod predictive_preload_tests {
    use super::*;

    #[test]
    fn lfu_ranks_most_frequent_uncached_keys_first() {
        let mut tracker = AccessTracker::new();
        // "hot" accessed 5x, "warm" 2x, "cold" 1x.
        for _ in 0..5 {
            tracker.record_access("hot");
        }
        for _ in 0..2 {
            tracker.record_access("warm");
        }
        tracker.record_access("cold");

        // Nothing is cached yet.
        let predicted = tracker.predict_preload_keys(2, &|_| false);
        assert_eq!(predicted.len(), 2);
        // "hot" has the highest recency-weighted frequency.
        assert_eq!(predicted[0], "hot");
    }

    #[test]
    fn already_cached_keys_are_excluded() {
        let mut tracker = AccessTracker::new();
        for _ in 0..5 {
            tracker.record_access("a");
        }
        tracker.record_access("b");

        // Pretend "a" is already resident; it must not be recommended.
        let predicted = tracker.predict_preload_keys(5, &|k| k == "a");
        assert!(!predicted.contains(&"a".to_string()));
        assert!(predicted.contains(&"b".to_string()));
    }

    #[test]
    fn markov_predicts_next_key_after_last_access() {
        let mut tracker = AccessTracker::new();
        // Establish a strong x -> y transition pattern.
        for _ in 0..6 {
            tracker.record_access("x");
            tracker.record_access("y");
        }
        // A single, much rarer, standalone key.
        tracker.record_access("z");
        // Make the most recent access "x" so the Markov boost targets "y".
        tracker.record_access("x");

        let predicted = tracker.predict_preload_keys(1, &|_| false);
        assert_eq!(predicted, vec!["y".to_string()]);
    }

    #[test]
    fn empty_history_yields_no_predictions() {
        let tracker = AccessTracker::new();
        assert!(tracker.predict_preload_keys(10, &|_| false).is_empty());
    }

    #[test]
    fn cache_manager_warm_loads_real_entries_only() {
        let mut manager = CacheManager::new();
        manager.create_cache("c1".to_string(), CacheType::Memory);

        let loaded = manager
            .warm_cache(
                "c1",
                vec![
                    ("k1".to_string(), b"v1".to_vec()),
                    ("k2".to_string(), b"v2".to_vec()),
                ],
            )
            .expect("cache exists");
        assert_eq!(loaded, 2);

        // Re-warming the same key does not overwrite (already resident).
        let reloaded = manager
            .warm_cache("c1", vec![("k1".to_string(), b"other".to_vec())])
            .expect("cache exists");
        assert_eq!(reloaded, 0);

        let cache = manager.get_cache_mut("c1").expect("cache exists");
        assert_eq!(cache.get("k1"), Some(b"v1".to_vec()));
    }

    #[test]
    fn predict_warm_keys_errors_for_missing_cache() {
        let manager = CacheManager::new();
        assert!(manager.predict_warm_keys("missing", 5).is_err());
    }
}