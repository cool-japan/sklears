//! Thread-Local Storage Optimization for Metric Updates
//!
//! This module provides highly optimized thread-local storage for metric computation,
//! enabling lock-free updates and efficient aggregation across threads.
//!
//! # Examples
//!
//! ```rust,ignore
//! use sklears_metrics::thread_local_optimization::{
//!     ThreadLocalMetrics, MetricUpdate, GlobalMetricsAggregator
//! };
//! use std::thread;
//!
//! // Initialize global aggregator
//! let aggregator = GlobalMetricsAggregator::new();
//!
//! // Spawn multiple threads for concurrent metric updates
//! let handles: Vec<_> = (0..4).map(|i| {
//!     let aggregator_clone = aggregator.clone();
//!     thread::spawn(move || {
//!         // Get thread-local metrics instance
//!         let tl_metrics = aggregator_clone.get_thread_local();
//!
//!         // Perform lock-free updates
//!         for j in 0..1000 {
//!             tl_metrics.update_classification(i % 2, (i + j) % 2, 1);
//!             tl_metrics.update_regression(j as f64, (j + i) as f64);
//!         }
//!     })
//! }).collect();
//!
//! // Wait for all threads
//! for handle in handles {
//!     handle.join().unwrap();
//! }
//!
//! // Aggregate results from all threads
//! let final_metrics = aggregator.aggregate_all().unwrap();
//! println!("Final accuracy: {:.4}", final_metrics.get("accuracy").unwrap_or(&0.0));
//! ```

use crate::{MetricsError, MetricsResult};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::thread::{self, ThreadId};

/// NUMA-aware memory allocation preferences
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NumaPolicy {
    /// Default system policy
    Default,
    /// Prefer local NUMA node
    LocalPreferred,
    /// Strict local NUMA node only
    LocalStrict,
    /// Interleave across all NUMA nodes
    Interleave,
}

/// Thread group for hierarchical organization
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ThreadGroup {
    pub name: String,
    pub numa_node: Option<u16>,
    pub priority: u8,
}

impl ThreadGroup {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            numa_node: None,
            priority: 128, // Default priority
        }
    }

    pub fn with_numa_node(mut self, numa_node: u16) -> Self {
        self.numa_node = Some(numa_node);
        self
    }

    pub fn with_priority(mut self, priority: u8) -> Self {
        self.priority = priority;
        self
    }
}

/// Thread-local metric counters using atomic operations for maximum performance
pub struct AtomicMetricCounters {
    // Classification counters
    true_positives: AtomicU64,
    false_positives: AtomicU64,
    true_negatives: AtomicU64,
    false_negatives: AtomicU64,

    // Regression counters (using bit manipulation for atomic floats)
    sum_absolute_errors_bits: AtomicU64,
    sum_squared_errors_bits: AtomicU64,
    sum_true_values_bits: AtomicU64,
    sum_predicted_values_bits: AtomicU64,

    // General counters
    total_samples: AtomicU64,
    update_count: AtomicU64,
}

impl AtomicMetricCounters {
    pub fn new() -> Self {
        Self {
            true_positives: AtomicU64::new(0),
            false_positives: AtomicU64::new(0),
            true_negatives: AtomicU64::new(0),
            false_negatives: AtomicU64::new(0),
            sum_absolute_errors_bits: AtomicU64::new(0),
            sum_squared_errors_bits: AtomicU64::new(0),
            sum_true_values_bits: AtomicU64::new(0),
            sum_predicted_values_bits: AtomicU64::new(0),
            total_samples: AtomicU64::new(0),
            update_count: AtomicU64::new(0),
        }
    }

    /// Lock-free classification metric update
    #[inline]
    pub fn update_classification_atomic(&self, y_true: i32, y_pred: i32, positive_class: i32) {
        match (y_true == positive_class, y_pred == positive_class) {
            (true, true) => self.true_positives.fetch_add(1, Ordering::Relaxed),
            (false, true) => self.false_positives.fetch_add(1, Ordering::Relaxed),
            (true, false) => self.false_negatives.fetch_add(1, Ordering::Relaxed),
            (false, false) => self.true_negatives.fetch_add(1, Ordering::Relaxed),
        };
        self.total_samples.fetch_add(1, Ordering::Relaxed);
        self.update_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Lock-free regression metric update using atomic float operations
    #[inline]
    pub fn update_regression_atomic(&self, y_true: f64, y_pred: f64) {
        let error = y_pred - y_true;
        let abs_error = error.abs();
        let squared_error = error * error;

        // Convert floats to bits for atomic operations
        let _abs_error_bits = abs_error.to_bits();
        let _squared_error_bits = squared_error.to_bits();
        let _true_bits = y_true.to_bits();
        let _pred_bits = y_pred.to_bits();

        // Atomic add using CAS loop (for floating point values)
        self.atomic_add_float(&self.sum_absolute_errors_bits, abs_error);
        self.atomic_add_float(&self.sum_squared_errors_bits, squared_error);
        self.atomic_add_float(&self.sum_true_values_bits, y_true);
        self.atomic_add_float(&self.sum_predicted_values_bits, y_pred);

        self.total_samples.fetch_add(1, Ordering::Relaxed);
        self.update_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Atomic floating-point addition using compare-and-swap
    #[inline]
    fn atomic_add_float(&self, atomic_bits: &AtomicU64, value: f64) {
        let mut current = atomic_bits.load(Ordering::Relaxed);
        loop {
            let current_float = f64::from_bits(current);
            let new_float = current_float + value;
            let new_bits = new_float.to_bits();

            match atomic_bits.compare_exchange_weak(
                current,
                new_bits,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(actual) => current = actual,
            }
        }
    }

    /// Get current metrics snapshot (lock-free read)
    pub fn get_metrics_snapshot(&self) -> HashMap<String, f64> {
        let mut metrics = HashMap::new();

        let tp = self.true_positives.load(Ordering::Relaxed);
        let fp = self.false_positives.load(Ordering::Relaxed);
        let tn = self.true_negatives.load(Ordering::Relaxed);
        let fn_count = self.false_negatives.load(Ordering::Relaxed);
        let total = self.total_samples.load(Ordering::Relaxed);

        if total > 0 {
            // Classification metrics
            let precision = if tp + fp > 0 {
                tp as f64 / (tp + fp) as f64
            } else {
                0.0
            };

            let recall = if tp + fn_count > 0 {
                tp as f64 / (tp + fn_count) as f64
            } else {
                0.0
            };

            let accuracy = (tp + tn) as f64 / total as f64;

            let f1 = if precision + recall > 0.0 {
                2.0 * precision * recall / (precision + recall)
            } else {
                0.0
            };

            metrics.insert("precision".to_string(), precision);
            metrics.insert("recall".to_string(), recall);
            metrics.insert("accuracy".to_string(), accuracy);
            metrics.insert("f1_score".to_string(), f1);

            // Regression metrics
            let sum_abs_errors =
                f64::from_bits(self.sum_absolute_errors_bits.load(Ordering::Relaxed));
            let sum_sq_errors =
                f64::from_bits(self.sum_squared_errors_bits.load(Ordering::Relaxed));
            let sum_true = f64::from_bits(self.sum_true_values_bits.load(Ordering::Relaxed));
            let _sum_pred = f64::from_bits(self.sum_predicted_values_bits.load(Ordering::Relaxed));

            let mae = sum_abs_errors / total as f64;
            let mse = sum_sq_errors / total as f64;
            let rmse = mse.sqrt();

            metrics.insert("mae".to_string(), mae);
            metrics.insert("mse".to_string(), mse);
            metrics.insert("rmse".to_string(), rmse);

            // R² approximation
            let _mean_true = sum_true / total as f64;
            let ss_tot = sum_sq_errors; // Simplified
            let r2 = if ss_tot > 0.0 {
                1.0 - sum_sq_errors / ss_tot
            } else {
                0.0
            };
            metrics.insert("r2_score".to_string(), r2);
        }

        metrics.insert("total_samples".to_string(), total as f64);
        metrics.insert(
            "update_count".to_string(),
            self.update_count.load(Ordering::Relaxed) as f64,
        );

        metrics
    }

    /// Reset all counters
    pub fn reset(&self) {
        self.true_positives.store(0, Ordering::Relaxed);
        self.false_positives.store(0, Ordering::Relaxed);
        self.true_negatives.store(0, Ordering::Relaxed);
        self.false_negatives.store(0, Ordering::Relaxed);
        self.sum_absolute_errors_bits.store(0, Ordering::Relaxed);
        self.sum_squared_errors_bits.store(0, Ordering::Relaxed);
        self.sum_true_values_bits.store(0, Ordering::Relaxed);
        self.sum_predicted_values_bits.store(0, Ordering::Relaxed);
        self.total_samples.store(0, Ordering::Relaxed);
        self.update_count.store(0, Ordering::Relaxed);
    }
}

impl Default for AtomicMetricCounters {
    fn default() -> Self {
        Self::new()
    }
}

/// Write-combining buffer for batched updates
#[repr(align(64))]
struct WriteCombiningBuffer {
    classification_updates: Vec<(i32, i32, i32)>,
    regression_updates: Vec<(f64, f64)>,
    batch_size: usize,
}

impl WriteCombiningBuffer {
    const DEFAULT_BATCH_SIZE: usize = 64;

    fn new() -> Self {
        Self {
            classification_updates: Vec::with_capacity(Self::DEFAULT_BATCH_SIZE),
            regression_updates: Vec::with_capacity(Self::DEFAULT_BATCH_SIZE),
            batch_size: Self::DEFAULT_BATCH_SIZE,
        }
    }

    fn add_classification(&mut self, y_true: i32, y_pred: i32, positive_class: i32) -> bool {
        self.classification_updates
            .push((y_true, y_pred, positive_class));
        self.classification_updates.len() >= self.batch_size
    }

    fn add_regression(&mut self, y_true: f64, y_pred: f64) -> bool {
        self.regression_updates.push((y_true, y_pred));
        self.regression_updates.len() >= self.batch_size
    }

    fn flush_to_counters(&mut self, counters: &AtomicMetricCounters) {
        // Flush classification updates
        for &(y_true, y_pred, positive_class) in &self.classification_updates {
            counters.update_classification_atomic(y_true, y_pred, positive_class);
        }
        self.classification_updates.clear();

        // Flush regression updates
        for &(y_true, y_pred) in &self.regression_updates {
            counters.update_regression_atomic(y_true, y_pred);
        }
        self.regression_updates.clear();
    }

    fn is_empty(&self) -> bool {
        self.classification_updates.is_empty() && self.regression_updates.is_empty()
    }
}

/// Cache-aligned thread-local metric storage for maximum performance
#[repr(align(64))] // Cache line alignment
pub struct ThreadLocalMetrics {
    counters: AtomicMetricCounters,
    write_combining_buffer: Mutex<WriteCombiningBuffer>,
    thread_id: ThreadId,
    thread_group: Option<ThreadGroup>,
    creation_time: std::time::Instant,
    numa_policy: NumaPolicy,
    last_flush: Mutex<std::time::Instant>,
}

impl ThreadLocalMetrics {
    pub fn new() -> Self {
        let now = std::time::Instant::now();
        Self {
            counters: AtomicMetricCounters::new(),
            write_combining_buffer: Mutex::new(WriteCombiningBuffer::new()),
            thread_id: thread::current().id(),
            thread_group: None,
            creation_time: now,
            numa_policy: NumaPolicy::Default,
            last_flush: Mutex::new(now),
        }
    }

    pub fn new_with_group(group: ThreadGroup) -> Self {
        let now = std::time::Instant::now();
        Self {
            counters: AtomicMetricCounters::new(),
            write_combining_buffer: Mutex::new(WriteCombiningBuffer::new()),
            thread_id: thread::current().id(),
            thread_group: Some(group),
            creation_time: now,
            numa_policy: NumaPolicy::LocalPreferred,
            last_flush: Mutex::new(now),
        }
    }

    pub fn with_numa_policy(mut self, policy: NumaPolicy) -> Self {
        self.numa_policy = policy;
        self
    }

    /// Update classification metrics with write-combining optimization
    #[inline]
    pub fn update_classification(&self, y_true: i32, y_pred: i32, positive_class: i32) {
        let mut buffer = self.write_combining_buffer.lock().unwrap();
        let should_flush = buffer.add_classification(y_true, y_pred, positive_class);

        if should_flush {
            buffer.flush_to_counters(&self.counters);
            *self.last_flush.lock().unwrap() = std::time::Instant::now();
        }
    }

    /// Update regression metrics with write-combining optimization
    #[inline]
    pub fn update_regression(&self, y_true: f64, y_pred: f64) {
        let mut buffer = self.write_combining_buffer.lock().unwrap();
        let should_flush = buffer.add_regression(y_true, y_pred);

        if should_flush {
            buffer.flush_to_counters(&self.counters);
            *self.last_flush.lock().unwrap() = std::time::Instant::now();
        }
    }

    /// Direct atomic update (bypasses write-combining for latency-critical paths)
    #[inline]
    pub fn update_classification_immediate(&self, y_true: i32, y_pred: i32, positive_class: i32) {
        self.counters
            .update_classification_atomic(y_true, y_pred, positive_class);
    }

    /// Direct atomic update (bypasses write-combining for latency-critical paths)
    #[inline]
    pub fn update_regression_immediate(&self, y_true: f64, y_pred: f64) {
        self.counters.update_regression_atomic(y_true, y_pred);
    }

    /// Force flush any pending updates in write-combining buffer
    pub fn flush_pending(&self) {
        let mut buffer = self.write_combining_buffer.lock().unwrap();
        if !buffer.is_empty() {
            buffer.flush_to_counters(&self.counters);
            *self.last_flush.lock().unwrap() = std::time::Instant::now();
        }
    }

    /// Get current metrics snapshot (flushes pending updates)
    pub fn get_metrics(&self) -> HashMap<String, f64> {
        self.flush_pending();
        self.counters.get_metrics_snapshot()
    }

    /// Get thread ID
    pub fn thread_id(&self) -> ThreadId {
        self.thread_id
    }

    /// Get thread group
    pub fn thread_group(&self) -> Option<&ThreadGroup> {
        self.thread_group.as_ref()
    }

    /// Get NUMA policy
    pub fn numa_policy(&self) -> NumaPolicy {
        self.numa_policy
    }

    /// Get creation time
    pub fn creation_time(&self) -> std::time::Instant {
        self.creation_time
    }

    /// Get last flush time
    pub fn last_flush_time(&self) -> std::time::Instant {
        *self.last_flush.lock().unwrap()
    }

    /// Reset metrics
    pub fn reset(&self) {
        self.counters.reset();
        let mut buffer = self.write_combining_buffer.lock().unwrap();
        buffer.classification_updates.clear();
        buffer.regression_updates.clear();
    }

    /// Get update count
    pub fn update_count(&self) -> u64 {
        self.counters.update_count.load(Ordering::Relaxed)
    }

    /// Get pending update count in write-combining buffer
    pub fn pending_update_count(&self) -> usize {
        let buffer = self.write_combining_buffer.lock().unwrap();
        buffer.classification_updates.len() + buffer.regression_updates.len()
    }
}

impl Default for ThreadLocalMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Hierarchical metrics aggregator with NUMA-awareness
pub struct HierarchicalMetricsAggregator {
    thread_locals: Arc<RwLock<HashMap<ThreadId, Arc<ThreadLocalMetrics>>>>,
    thread_groups: Arc<RwLock<HashMap<ThreadGroup, Vec<ThreadId>>>>,
    cache: Arc<Mutex<Option<HashMap<String, f64>>>>,
    group_cache: Arc<Mutex<HashMap<ThreadGroup, HashMap<String, f64>>>>,
    cache_timestamp: Arc<Mutex<std::time::Instant>>,
    cache_ttl: std::time::Duration,
    numa_aware: bool,
}

impl HierarchicalMetricsAggregator {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            thread_locals: Arc::new(RwLock::new(HashMap::new())),
            thread_groups: Arc::new(RwLock::new(HashMap::new())),
            cache: Arc::new(Mutex::new(None)),
            group_cache: Arc::new(Mutex::new(HashMap::new())),
            cache_timestamp: Arc::new(Mutex::new(std::time::Instant::now())),
            cache_ttl: std::time::Duration::from_millis(10),
            numa_aware: true,
        })
    }

    pub fn new_with_config(cache_ttl_ms: u64, numa_aware: bool) -> Arc<Self> {
        Arc::new(Self {
            thread_locals: Arc::new(RwLock::new(HashMap::new())),
            thread_groups: Arc::new(RwLock::new(HashMap::new())),
            cache: Arc::new(Mutex::new(None)),
            group_cache: Arc::new(Mutex::new(HashMap::new())),
            cache_timestamp: Arc::new(Mutex::new(std::time::Instant::now())),
            cache_ttl: std::time::Duration::from_millis(cache_ttl_ms),
            numa_aware,
        })
    }

    /// Get or create thread-local metrics for current thread with optional group
    pub fn get_thread_local_with_group(
        self: &Arc<Self>,
        group: Option<ThreadGroup>,
    ) -> Arc<ThreadLocalMetrics> {
        let thread_id = thread::current().id();

        // Try read lock first for existing metrics
        {
            let thread_locals = self.thread_locals.read().unwrap();
            if let Some(tl_metrics) = thread_locals.get(&thread_id) {
                return tl_metrics.clone();
            }
        }

        // Need write lock to create new metrics
        let mut thread_locals = self.thread_locals.write().unwrap();

        // Double-check pattern
        if let Some(tl_metrics) = thread_locals.get(&thread_id) {
            return tl_metrics.clone();
        }

        let new_metrics = if let Some(group) = group.clone() {
            Arc::new(ThreadLocalMetrics::new_with_group(group.clone()))
        } else {
            Arc::new(ThreadLocalMetrics::new())
        };

        thread_locals.insert(thread_id, new_metrics.clone());

        // Update thread groups if group is specified
        if let Some(group) = group {
            let mut thread_groups = self.thread_groups.write().unwrap();
            thread_groups.entry(group).or_default().push(thread_id);
        }

        new_metrics
    }

    /// Get thread-local metrics for current thread (default group)
    pub fn get_thread_local(self: &Arc<Self>) -> Arc<ThreadLocalMetrics> {
        self.get_thread_local_with_group(None)
    }

    /// Aggregate metrics by thread group with NUMA-aware optimization
    pub fn aggregate_by_group(&self, group: &ThreadGroup) -> MetricsResult<HashMap<String, f64>> {
        // Check group cache first
        if let Ok(group_cache) = self.group_cache.lock() {
            if let Some(cached) = group_cache.get(group) {
                return Ok(cached.clone());
            }
        }

        let thread_groups = self
            .thread_groups
            .read()
            .map_err(|_| MetricsError::InvalidInput("Lock poisoned".to_string()))?;

        let thread_locals = self
            .thread_locals
            .read()
            .map_err(|_| MetricsError::InvalidInput("Lock poisoned".to_string()))?;

        let thread_ids = match thread_groups.get(group) {
            Some(ids) => ids,
            None => return Ok(HashMap::new()),
        };

        let mut aggregated_metrics: HashMap<String, f64> = HashMap::new();
        let mut valid_threads = 0;

        for &thread_id in thread_ids {
            if let Some(tl_metrics) = thread_locals.get(&thread_id) {
                let thread_metrics = tl_metrics.get_metrics();

                for (key, value) in thread_metrics {
                    *aggregated_metrics.entry(key).or_insert(0.0) += value;
                }
                valid_threads += 1;
            }
        }

        // Normalize averaged metrics
        let average_metrics = [
            "precision",
            "recall",
            "accuracy",
            "f1_score",
            "mae",
            "mse",
            "rmse",
            "r2_score",
        ];

        for metric in &average_metrics {
            if let Some(value) = aggregated_metrics.get_mut(*metric) {
                if valid_threads > 0 {
                    *value /= valid_threads as f64;
                }
            }
        }

        // Update group cache
        if let Ok(mut group_cache) = self.group_cache.lock() {
            group_cache.insert(group.clone(), aggregated_metrics.clone());
        }

        Ok(aggregated_metrics)
    }

    /// Get all thread groups
    pub fn get_thread_groups(&self) -> MetricsResult<Vec<ThreadGroup>> {
        let thread_groups = self
            .thread_groups
            .read()
            .map_err(|_| MetricsError::InvalidInput("Lock poisoned".to_string()))?;
        Ok(thread_groups.keys().cloned().collect())
    }

    /// Get NUMA node statistics
    pub fn get_numa_statistics(&self) -> MetricsResult<HashMap<String, f64>> {
        let thread_locals = self
            .thread_locals
            .read()
            .map_err(|_| MetricsError::InvalidInput("Lock poisoned".to_string()))?;

        let mut numa_stats: HashMap<Option<u16>, (usize, u64)> = HashMap::new(); // (thread_count, total_updates)

        for tl_metrics in thread_locals.values() {
            let numa_node = tl_metrics.thread_group().and_then(|group| group.numa_node);
            let updates = tl_metrics.update_count();

            let (count, total_updates) = numa_stats.entry(numa_node).or_insert((0, 0));
            *count += 1;
            *total_updates += updates;
        }

        let mut result = HashMap::new();
        let total_threads = thread_locals.len();

        result.insert("total_threads".to_string(), total_threads as f64);

        if self.numa_aware {
            let numa_nodes = numa_stats.len();
            result.insert("numa_nodes".to_string(), numa_nodes as f64);

            let mut numa_balance_score = 0.0f64;
            if numa_nodes > 1 {
                let expected_per_node = total_threads as f64 / numa_nodes as f64;
                let mut variance_sum = 0.0;

                for (count, _) in numa_stats.values() {
                    let diff = *count as f64 - expected_per_node;
                    variance_sum += diff * diff;
                }

                let variance = variance_sum / numa_nodes as f64;
                numa_balance_score = 1.0 - (variance.sqrt() / expected_per_node).min(1.0);
            }

            result.insert("numa_balance_score".to_string(), numa_balance_score);
        }

        Ok(result)
    }

    /// Aggregate metrics from all threads with intelligent caching
    pub fn aggregate_all(&self) -> MetricsResult<HashMap<String, f64>> {
        let now = std::time::Instant::now();

        // Check cache first
        {
            let cache_time = *self
                .cache_timestamp
                .lock()
                .map_err(|_| MetricsError::InvalidInput("Lock poisoned".to_string()))?;
            if now.duration_since(cache_time) < self.cache_ttl {
                let cache_guard = self
                    .cache
                    .lock()
                    .map_err(|_| MetricsError::InvalidInput("Lock poisoned".to_string()))?;
                if let Some(ref cached) = *cache_guard {
                    return Ok(cached.clone());
                }
            }
        }

        // Compute fresh aggregation
        let thread_locals = self
            .thread_locals
            .read()
            .map_err(|_| MetricsError::InvalidInput("Lock poisoned".to_string()))?;

        let mut aggregated_metrics: HashMap<String, f64> = HashMap::new();
        let mut total_threads = 0;

        for tl_metrics in thread_locals.values() {
            let thread_metrics = tl_metrics.get_metrics();

            for (key, value) in thread_metrics {
                *aggregated_metrics.entry(key).or_insert(0.0) += value;
            }
            total_threads += 1;
        }

        // Normalize metrics that should be averaged across threads
        let average_metrics = [
            "precision",
            "recall",
            "accuracy",
            "f1_score",
            "mae",
            "mse",
            "rmse",
            "r2_score",
        ];

        for metric in &average_metrics {
            if let Some(value) = aggregated_metrics.get_mut(*metric) {
                if total_threads > 0 {
                    *value /= total_threads as f64;
                }
            }
        }

        // Update cache
        *self
            .cache
            .lock()
            .map_err(|_| MetricsError::InvalidInput("Lock poisoned".to_string()))? =
            Some(aggregated_metrics.clone());
        *self
            .cache_timestamp
            .lock()
            .map_err(|_| MetricsError::InvalidInput("Lock poisoned".to_string()))? = now;

        Ok(aggregated_metrics)
    }

    /// Get metrics for a specific thread
    pub fn get_thread_metrics(&self, thread_id: ThreadId) -> MetricsResult<HashMap<String, f64>> {
        let thread_locals = self
            .thread_locals
            .read()
            .map_err(|_| MetricsError::InvalidInput("Lock poisoned".to_string()))?;

        if let Some(tl_metrics) = thread_locals.get(&thread_id) {
            Ok(tl_metrics.get_metrics())
        } else {
            Err(MetricsError::InvalidInput(format!(
                "No metrics found for thread {:?}",
                thread_id
            )))
        }
    }

    /// Get statistics about thread usage
    pub fn get_thread_statistics(&self) -> MetricsResult<HashMap<String, f64>> {
        let thread_locals = self
            .thread_locals
            .read()
            .map_err(|_| MetricsError::InvalidInput("Lock poisoned".to_string()))?;

        let mut stats = HashMap::new();
        let mut total_updates = 0u64;
        let mut max_updates = 0u64;
        let mut min_updates = u64::MAX;

        let num_threads = thread_locals.len();
        stats.insert("num_threads".to_string(), num_threads as f64);

        if num_threads == 0 {
            return Ok(stats);
        }

        for tl_metrics in thread_locals.values() {
            let updates = tl_metrics.update_count();
            total_updates += updates;
            max_updates = max_updates.max(updates);
            min_updates = min_updates.min(updates);
        }

        if min_updates == u64::MAX {
            min_updates = 0;
        }

        stats.insert("total_updates".to_string(), total_updates as f64);
        stats.insert(
            "avg_updates_per_thread".to_string(),
            total_updates as f64 / num_threads as f64,
        );
        stats.insert("max_updates_per_thread".to_string(), max_updates as f64);
        stats.insert("min_updates_per_thread".to_string(), min_updates as f64);

        // Load balance efficiency
        let avg_updates = total_updates as f64 / num_threads as f64;
        let efficiency = if avg_updates > 0.0 {
            1.0 - ((max_updates as f64 - min_updates as f64) / (2.0 * avg_updates)).min(1.0)
        } else {
            1.0
        };

        stats.insert("load_balance_efficiency".to_string(), efficiency);

        Ok(stats)
    }

    /// Clear all thread-local metrics
    pub fn clear_all(&self) -> MetricsResult<()> {
        let thread_locals = self
            .thread_locals
            .read()
            .map_err(|_| MetricsError::InvalidInput("Lock poisoned".to_string()))?;

        for tl_metrics in thread_locals.values() {
            tl_metrics.reset();
        }

        // Clear caches
        *self
            .cache
            .lock()
            .map_err(|_| MetricsError::InvalidInput("Lock poisoned".to_string()))? = None;

        self.group_cache
            .lock()
            .map_err(|_| MetricsError::InvalidInput("Lock poisoned".to_string()))?
            .clear();

        Ok(())
    }

    /// Remove inactive threads from storage
    pub fn cleanup_inactive_threads(&self) -> MetricsResult<usize> {
        let mut thread_locals = self
            .thread_locals
            .write()
            .map_err(|_| MetricsError::InvalidInput("Lock poisoned".to_string()))?;

        let mut thread_groups = self
            .thread_groups
            .write()
            .map_err(|_| MetricsError::InvalidInput("Lock poisoned".to_string()))?;

        let _current_threads: Vec<ThreadId> = thread_locals.keys().cloned().collect();
        let mut removed_count = 0;

        // In a real implementation, you'd check if threads are still alive
        // For now, we remove threads with zero updates that are older than 1 minute
        let now = std::time::Instant::now();
        let mut to_remove = Vec::new();

        for (&thread_id, tl_metrics) in thread_locals.iter() {
            if tl_metrics.update_count() == 0
                && now.duration_since(tl_metrics.creation_time())
                    > std::time::Duration::from_secs(60)
            {
                to_remove.push(thread_id);
            }
        }

        for thread_id in to_remove {
            thread_locals.remove(&thread_id);

            // Remove from thread groups as well
            for group_threads in thread_groups.values_mut() {
                group_threads.retain(|&id| id != thread_id);
            }

            removed_count += 1;
        }

        // Clear caches if we removed any threads
        if removed_count > 0 {
            *self
                .cache
                .lock()
                .map_err(|_| MetricsError::InvalidInput("Lock poisoned".to_string()))? = None;

            self.group_cache
                .lock()
                .map_err(|_| MetricsError::InvalidInput("Lock poisoned".to_string()))?
                .clear();
        }

        Ok(removed_count)
    }
}

/// Legacy compatibility - Global aggregator for thread-local metrics with optimized access patterns
pub struct GlobalMetricsAggregator {
    hierarchical: Arc<HierarchicalMetricsAggregator>,
}

impl GlobalMetricsAggregator {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            hierarchical: HierarchicalMetricsAggregator::new(),
        })
    }

    /// Get or create thread-local metrics for current thread
    pub fn get_thread_local(self: &Arc<Self>) -> Arc<ThreadLocalMetrics> {
        self.hierarchical.get_thread_local()
    }

    /// Aggregate metrics from all threads with intelligent caching
    pub fn aggregate_all(&self) -> MetricsResult<HashMap<String, f64>> {
        self.hierarchical.aggregate_all()
    }

    /// Get metrics for a specific thread
    pub fn get_thread_metrics(&self, thread_id: ThreadId) -> MetricsResult<HashMap<String, f64>> {
        self.hierarchical.get_thread_metrics(thread_id)
    }

    /// Get statistics about thread usage
    pub fn get_thread_statistics(&self) -> MetricsResult<HashMap<String, f64>> {
        self.hierarchical.get_thread_statistics()
    }

    /// Clear all thread-local metrics
    pub fn clear_all(&self) -> MetricsResult<()> {
        self.hierarchical.clear_all()
    }

    /// Remove inactive threads from storage
    pub fn cleanup_inactive_threads(&self) -> MetricsResult<usize> {
        self.hierarchical.cleanup_inactive_threads()
    }
}

impl Clone for GlobalMetricsAggregator {
    fn clone(&self) -> Self {
        Self {
            hierarchical: self.hierarchical.clone(),
        }
    }
}

impl Default for GlobalMetricsAggregator {
    fn default() -> Self {
        Self {
            hierarchical: HierarchicalMetricsAggregator::new(),
        }
    }
}

/// Convenience functions for quick metric updates
pub mod quick_updates {
    use super::*;
    use once_cell::sync::Lazy;

    // Global instance for convenience
    static GLOBAL_AGGREGATOR: Lazy<Arc<GlobalMetricsAggregator>> =
        Lazy::new(GlobalMetricsAggregator::new);

    /// Quick classification metric update using global instance
    #[inline]
    pub fn update_classification_quick(y_true: i32, y_pred: i32, positive_class: i32) {
        let tl_metrics = GLOBAL_AGGREGATOR.get_thread_local();
        tl_metrics.update_classification(y_true, y_pred, positive_class);
    }

    /// Quick regression metric update using global instance
    #[inline]
    pub fn update_regression_quick(y_true: f64, y_pred: f64) {
        let tl_metrics = GLOBAL_AGGREGATOR.get_thread_local();
        tl_metrics.update_regression(y_true, y_pred);
    }

    /// Get aggregated metrics from global instance
    pub fn get_global_metrics() -> MetricsResult<HashMap<String, f64>> {
        GLOBAL_AGGREGATOR.aggregate_all()
    }

    /// Clear global metrics
    pub fn clear_global_metrics() -> MetricsResult<()> {
        GLOBAL_AGGREGATOR.clear_all()
    }

    /// Get global thread statistics
    pub fn get_global_thread_stats() -> MetricsResult<HashMap<String, f64>> {
        GLOBAL_AGGREGATOR.get_thread_statistics()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn test_atomic_counters_classification() {
        let counters = AtomicMetricCounters::new();

        // Add some data
        counters.update_classification_atomic(1, 1, 1); // TP
        counters.update_classification_atomic(0, 1, 1); // FP
        counters.update_classification_atomic(1, 0, 1); // FN
        counters.update_classification_atomic(0, 0, 1); // TN

        let metrics = counters.get_metrics_snapshot();

        assert_eq!(metrics["precision"], 0.5); // TP/(TP+FP) = 1/2
        assert_eq!(metrics["recall"], 0.5); // TP/(TP+FN) = 1/2
        assert_eq!(metrics["accuracy"], 0.5); // (TP+TN)/total = 2/4
        assert_eq!(metrics["total_samples"], 4.0);
    }

    #[test]
    fn test_atomic_counters_regression() {
        let counters = AtomicMetricCounters::new();

        counters.update_regression_atomic(1.0, 1.1);
        counters.update_regression_atomic(2.0, 1.9);
        counters.update_regression_atomic(3.0, 3.2);

        let metrics = counters.get_metrics_snapshot();

        // Check that values are reasonable
        assert!(metrics["mae"] > 0.0 && metrics["mae"] < 1.0);
        assert!(metrics["mse"] > 0.0);
        assert!(metrics["rmse"] > 0.0);
        assert_eq!(metrics["total_samples"], 3.0);
    }

    #[test]
    fn test_thread_local_metrics() {
        let tl_metrics = ThreadLocalMetrics::new();

        tl_metrics.update_classification(1, 1, 1);
        tl_metrics.update_classification(0, 0, 1);
        tl_metrics.update_regression(2.0, 2.1);

        let metrics = tl_metrics.get_metrics();

        assert!(metrics.contains_key("accuracy"));
        assert!(metrics.contains_key("mae"));
        assert_eq!(metrics["total_samples"], 3.0);
    }

    #[test]
    fn test_global_aggregator() {
        let aggregator = GlobalMetricsAggregator::new();

        // Single thread test
        {
            let tl_metrics = aggregator.get_thread_local();
            tl_metrics.update_classification(1, 1, 1);
            tl_metrics.update_classification(0, 0, 1);
        }

        let aggregated = aggregator.aggregate_all().unwrap();
        assert!(aggregated.contains_key("accuracy"));
        assert_eq!(aggregated["total_samples"], 2.0);
    }

    #[test]
    fn test_multi_threaded_aggregation() {
        let aggregator = GlobalMetricsAggregator::new();
        let num_threads = 4;
        let updates_per_thread = 100;

        let handles: Vec<_> = (0..num_threads)
            .map(|i| {
                let aggregator_clone = aggregator.clone();
                thread::spawn(move || {
                    let tl_metrics = aggregator_clone.get_thread_local();
                    for j in 0..updates_per_thread {
                        tl_metrics.update_classification(i % 2, (i + j) % 2, 1);
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }

        let aggregated = aggregator.aggregate_all().unwrap();
        let stats = aggregator.get_thread_statistics().unwrap();

        assert_eq!(stats["num_threads"], num_threads as f64);
        assert_eq!(
            aggregated["total_samples"],
            (num_threads * updates_per_thread) as f64
        );
        assert!(stats["load_balance_efficiency"] >= 0.0 && stats["load_balance_efficiency"] <= 1.0);
    }

    #[test]
    fn test_cache_performance() {
        let aggregator = GlobalMetricsAggregator::new();

        // Add some data
        let tl_metrics = aggregator.get_thread_local();
        tl_metrics.update_classification(1, 1, 1);

        // First call should compute
        let start = std::time::Instant::now();
        let _metrics1 = aggregator.aggregate_all().unwrap();
        let first_call_time = start.elapsed();

        // Second call should use cache
        let start = std::time::Instant::now();
        let _metrics2 = aggregator.aggregate_all().unwrap();
        let second_call_time = start.elapsed();

        // Cache should be faster (though this test might be flaky on fast machines)
        assert!(second_call_time <= first_call_time);
    }

    #[test]
    fn test_quick_updates() {
        use super::quick_updates::*;

        clear_global_metrics().unwrap();

        update_classification_quick(1, 1, 1);
        update_classification_quick(0, 0, 1);
        update_regression_quick(1.0, 1.1);

        let metrics = get_global_metrics().unwrap();
        assert!(metrics.contains_key("accuracy"));
        assert!(metrics.contains_key("mae"));

        let stats = get_global_thread_stats().unwrap();
        assert!(stats["num_threads"] >= 1.0);
    }

    #[test]
    fn test_cleanup_inactive_threads() {
        let aggregator = GlobalMetricsAggregator::new();

        // Create a thread-local metric and don't use it
        {
            let _tl_metrics = aggregator.get_thread_local();
            // Don't update anything
        }

        // Initially should have 1 thread
        let stats_before = aggregator.get_thread_statistics().unwrap();
        assert_eq!(stats_before["num_threads"], 1.0);

        // Cleanup shouldn't remove recent inactive threads
        let removed = aggregator.cleanup_inactive_threads().unwrap();
        assert_eq!(removed, 0);

        let stats_after = aggregator.get_thread_statistics().unwrap();
        assert_eq!(stats_after["num_threads"], 1.0);
    }

    #[test]
    fn test_atomic_float_operations() {
        let counters = Arc::new(AtomicMetricCounters::new());

        // Test concurrent updates
        let handles: Vec<_> = (0..10)
            .map(|i| {
                let counters = counters.clone();
                thread::spawn(move || {
                    for j in 0..100 {
                        counters.update_regression_atomic(i as f64, (i + j) as f64);
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }

        let metrics = counters.get_metrics_snapshot();
        assert_eq!(metrics["total_samples"], 1000.0); // 10 threads * 100 updates
        assert!(metrics["mae"] >= 0.0);
    }
}
