//! Profile-guided optimizations for dummy estimator performance

use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Profile-guided optimization configuration
#[derive(Debug, Clone)]
pub struct ProfileConfig {
    /// collect_timings
    pub collect_timings: bool,
    /// adaptive_thresholds
    pub adaptive_thresholds: bool,
    /// cache_predictions
    pub cache_predictions: bool,
    /// use_fast_path
    pub use_fast_path: bool,
}

impl Default for ProfileConfig {
    fn default() -> Self {
        Self {
            collect_timings: true,
            adaptive_thresholds: true,
            cache_predictions: false,
            use_fast_path: true,
        }
    }
}

/// Performance profiler for dummy estimator operations
pub struct PerformanceProfiler {
    timings: HashMap<String, Vec<Duration>>,
    config: ProfileConfig,
}

impl PerformanceProfiler {
    pub fn new(config: ProfileConfig) -> Self {
        Self {
            timings: HashMap::new(),
            config,
        }
    }

    pub fn start_timer(&self, operation: &str) -> ProfileTimer {
        ProfileTimer::new(operation.to_string(), self.config.collect_timings)
    }

    pub fn record_timing(&mut self, operation: String, duration: Duration) {
        if self.config.collect_timings {
            self.timings.entry(operation).or_default().push(duration);
        }
    }

    pub fn get_average_time(&self, operation: &str) -> Option<Duration> {
        self.timings.get(operation).map(|times| {
            let total: Duration = times.iter().sum();
            total / times.len() as u32
        })
    }

    pub fn should_use_fast_path(&self, operation: &str, threshold: Duration) -> bool {
        if !self.config.use_fast_path {
            return false;
        }

        if let Some(avg_time) = self.get_average_time(operation) {
            avg_time > threshold
        } else {
            false
        }
    }
}

/// Timer for profiling operations
pub struct ProfileTimer {
    operation: String,
    start: Instant,
    collect: bool,
}

impl ProfileTimer {
    pub fn new(operation: String, collect: bool) -> Self {
        Self {
            operation,
            start: Instant::now(),
            collect,
        }
    }

    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }
}

impl Drop for ProfileTimer {
    fn drop(&mut self) {
        if self.collect {
            let _duration = self.elapsed();
            // In a real implementation, you'd send this to the profiler
        }
    }
}

/// Adaptive threshold manager
pub struct AdaptiveThresholds {
    thresholds: HashMap<String, f64>,
    adjustments: HashMap<String, f64>,
    config: ProfileConfig,
}

impl AdaptiveThresholds {
    pub fn new(config: ProfileConfig) -> Self {
        Self {
            thresholds: HashMap::new(),
            adjustments: HashMap::new(),
            config,
        }
    }

    pub fn get_threshold(&self, operation: &str) -> f64 {
        self.thresholds.get(operation).copied().unwrap_or(1.0)
    }

    pub fn adjust_threshold(&mut self, operation: &str, performance_ratio: f64) {
        if !self.config.adaptive_thresholds {
            return;
        }

        let current = self.get_threshold(operation);
        let adjustment = if performance_ratio > 1.0 {
            current * 0.9 // Decrease threshold if performance is good
        } else {
            current * 1.1 // Increase threshold if performance is poor
        };

        self.thresholds.insert(operation.to_string(), adjustment);
    }
}

/// Prediction cache for frequently accessed results
pub struct PredictionCache<K, V> {
    cache: HashMap<K, V>,
    max_size: usize,
    access_count: HashMap<K, usize>,
}

impl<K: Clone + Eq + std::hash::Hash, V: Clone> PredictionCache<K, V> {
    pub fn new(max_size: usize) -> Self {
        Self {
            cache: HashMap::new(),
            max_size,
            access_count: HashMap::new(),
        }
    }

    pub fn get(&mut self, key: &K) -> Option<V> {
        if let Some(value) = self.cache.get(key) {
            *self.access_count.entry(key.clone()).or_insert(0) += 1;
            Some(value.clone())
        } else {
            None
        }
    }

    pub fn insert(&mut self, key: K, value: V) {
        if self.cache.len() >= self.max_size {
            self.evict_least_used();
        }
        self.cache.insert(key.clone(), value);
        self.access_count.insert(key, 1);
    }

    fn evict_least_used(&mut self) {
        if let Some((least_used_key, _)) = self.access_count.iter().min_by_key(|(_, &count)| count)
        {
            let key_to_remove = least_used_key.clone();
            self.cache.remove(&key_to_remove);
            self.access_count.remove(&key_to_remove);
        }
    }
}
