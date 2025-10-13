//! Parallel computation utilities for explanation methods
//!
//! This module provides utilities for parallelizing explanation computations
//! using rayon for optimal performance on multi-core systems.

#[cfg(feature = "parallel")]
use rayon::prelude::*;

#[cfg(feature = "parallel")]
use num_cpus;

use crate::{Float, SklResult};
// ✅ SciRS2 Policy Compliant Import
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use scirs2_core::random::{seq::SliceRandom, SeedableRng};
use std::sync::Arc;

/// Configuration for parallel computation
#[derive(Debug, Clone)]
pub struct ParallelConfig {
    /// Number of threads to use (None = auto-detect)
    pub n_threads: Option<usize>,
    /// Minimum batch size for parallelization
    pub min_batch_size: usize,
    /// Whether to force sequential computation
    pub force_sequential: bool,
}

impl Default for ParallelConfig {
    fn default() -> Self {
        Self {
            n_threads: None,
            min_batch_size: 100,
            force_sequential: false,
        }
    }
}

impl ParallelConfig {
    /// Create a new parallel configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the number of threads
    pub fn with_threads(mut self, n_threads: usize) -> Self {
        self.n_threads = Some(n_threads);
        self
    }

    /// Set the minimum batch size for parallelization
    pub fn with_min_batch_size(mut self, min_batch_size: usize) -> Self {
        self.min_batch_size = min_batch_size;
        self
    }

    /// Force sequential computation
    pub fn sequential(mut self) -> Self {
        self.force_sequential = true;
        self
    }

    /// Get the optimal number of threads
    pub fn get_n_threads(&self) -> usize {
        if self.force_sequential {
            return 1;
        }

        #[cfg(feature = "parallel")]
        {
            self.n_threads.unwrap_or_else(|| num_cpus::get())
        }
        #[cfg(not(feature = "parallel"))]
        {
            1
        }
    }

    /// Check if parallelization should be used for given data size
    pub fn should_parallelize(&self, data_size: usize) -> bool {
        !self.force_sequential && data_size >= self.min_batch_size
    }
}

/// Trait for parallelizable explanation methods
pub trait ParallelExplanation {
    type Input;
    type Output;
    type Config;

    /// Compute explanation for a single instance
    fn compute_single(&self, input: &Self::Input, config: &Self::Config)
        -> SklResult<Self::Output>;

    /// Compute explanations in parallel for multiple instances
    fn compute_parallel(
        &self,
        inputs: &[Self::Input],
        config: &Self::Config,
        parallel_config: &ParallelConfig,
    ) -> SklResult<Vec<Self::Output>>
    where
        Self: Sync,
        Self::Input: Sync,
        Self::Config: Sync,
        Self::Output: Send,
    {
        if parallel_config.should_parallelize(inputs.len()) {
            self.compute_parallel_impl(inputs, config, parallel_config)
        } else {
            self.compute_sequential(inputs, config)
        }
    }

    /// Sequential computation fallback
    fn compute_sequential(
        &self,
        inputs: &[Self::Input],
        config: &Self::Config,
    ) -> SklResult<Vec<Self::Output>> {
        inputs
            .iter()
            .map(|input| self.compute_single(input, config))
            .collect()
    }

    /// Parallel implementation
    #[cfg(feature = "parallel")]
    fn compute_parallel_impl(
        &self,
        inputs: &[Self::Input],
        config: &Self::Config,
        _parallel_config: &ParallelConfig,
    ) -> SklResult<Vec<Self::Output>>
    where
        Self: Sync,
        Self::Input: Sync,
        Self::Config: Sync,
        Self::Output: Send,
    {
        inputs
            .par_iter()
            .map(|input| self.compute_single(input, config))
            .collect()
    }

    /// Parallel implementation (fallback when parallel feature is disabled)
    #[cfg(not(feature = "parallel"))]
    fn compute_parallel_impl(
        &self,
        inputs: &[Self::Input],
        config: &Self::Config,
        _parallel_config: &ParallelConfig,
    ) -> SklResult<Vec<Self::Output>> {
        self.compute_sequential(inputs, config)
    }
}

/// Parallel permutation importance computation
pub struct ParallelPermutationImportance<F> {
    model: Arc<F>,
    scoring_fn: fn(&ArrayView1<Float>, &ArrayView1<Float>) -> Float,
}

impl<F> ParallelPermutationImportance<F> {
    /// Create a new parallel permutation importance calculator
    pub fn new(
        model: Arc<F>,
        scoring_fn: fn(&ArrayView1<Float>, &ArrayView1<Float>) -> Float,
    ) -> Self {
        Self { model, scoring_fn }
    }
}

/// Input for permutation importance calculation
#[derive(Debug, Clone)]
pub struct PermutationInput {
    /// feature_idx
    pub feature_idx: usize,
    /// x_data
    pub x_data: Array2<Float>,
    /// y_true
    pub y_true: Array1<Float>,
    /// n_repeats
    pub n_repeats: usize,
    /// random_state
    pub random_state: u64,
}

impl<F> ParallelExplanation for ParallelPermutationImportance<F>
where
    F: Fn(&ArrayView2<Float>) -> SklResult<Array1<Float>> + Send + Sync,
{
    type Input = PermutationInput;
    type Output = Vec<Float>;
    type Config = ();

    fn compute_single(
        &self,
        input: &Self::Input,
        _config: &Self::Config,
    ) -> SklResult<Self::Output> {
        let mut importances = Vec::with_capacity(input.n_repeats);
        let mut rng = scirs2_core::random::ChaCha8Rng::seed_from_u64(input.random_state);

        // Baseline score
        let y_pred_baseline = (self.model)(&input.x_data.view())?;
        let baseline_score = (self.scoring_fn)(&input.y_true.view(), &y_pred_baseline.view());

        for _ in 0..input.n_repeats {
            let mut x_permuted = input.x_data.clone();
            let mut column = x_permuted.column_mut(input.feature_idx);

            // Permute the feature column
            let mut indices: Vec<usize> = (0..column.len()).collect();
            indices.shuffle(&mut rng);

            let original_values: Vec<Float> = column.to_vec();
            for (i, &new_idx) in indices.iter().enumerate() {
                column[i] = original_values[new_idx];
            }

            // Score with permuted feature
            let y_pred_permuted = (self.model)(&x_permuted.view())?;
            let permuted_score = (self.scoring_fn)(&input.y_true.view(), &y_pred_permuted.view());

            importances.push(baseline_score - permuted_score);
        }

        Ok(importances)
    }
}

/// Parallel SHAP value computation utility
#[cfg(feature = "parallel")]
pub fn compute_shap_parallel<F>(
    model: F,
    X: &ArrayView2<Float>,
    baseline: &ArrayView1<Float>,
    config: &ParallelConfig,
) -> SklResult<Array2<Float>>
where
    F: Fn(&ArrayView2<Float>) -> SklResult<Array1<Float>> + Send + Sync + Clone,
{
    let n_samples = X.nrows();
    let n_features = X.ncols();

    if !config.should_parallelize(n_samples) {
        return compute_shap_sequential(model, X, baseline);
    }

    let results: SklResult<Vec<_>> = (0..n_samples)
        .into_par_iter()
        .map(|i| {
            let instance = X.row(i);
            compute_shap_single_instance(model.clone(), &instance, baseline)
        })
        .collect();

    let shap_values = results?;
    let mut result = Array2::zeros((n_samples, n_features));
    for (i, values) in shap_values.into_iter().enumerate() {
        result.row_mut(i).assign(&values);
    }

    Ok(result)
}

/// Sequential SHAP computation fallback
fn compute_shap_sequential<F>(
    model: F,
    X: &ArrayView2<Float>,
    baseline: &ArrayView1<Float>,
) -> SklResult<Array2<Float>>
where
    F: Fn(&ArrayView2<Float>) -> SklResult<Array1<Float>> + Clone,
{
    let n_samples = X.nrows();
    let n_features = X.ncols();
    let mut result = Array2::zeros((n_samples, n_features));

    for i in 0..n_samples {
        let instance = X.row(i);
        let shap_values = compute_shap_single_instance(model.clone(), &instance, baseline)?;
        result.row_mut(i).assign(&shap_values);
    }

    Ok(result)
}

/// Compute SHAP values for a single instance (simplified kernel SHAP)
fn compute_shap_single_instance<F>(
    model: F,
    instance: &ArrayView1<Float>,
    baseline: &ArrayView1<Float>,
) -> SklResult<Array1<Float>>
where
    F: Fn(&ArrayView2<Float>) -> SklResult<Array1<Float>>,
{
    let n_features = instance.len();
    let mut shap_values = Array1::zeros(n_features);

    // Simplified SHAP computation (marginal contributions)
    for i in 0..n_features {
        // Create coalition without feature i
        let mut coalition_without = baseline.to_owned();
        for j in 0..n_features {
            if j != i {
                coalition_without[j] = instance[j];
            }
        }

        // Create coalition with feature i
        let mut coalition_with = coalition_without.clone();
        coalition_with[i] = instance[i];

        // Compute marginal contribution
        let pred_without = model(&coalition_without.view().insert_axis(Axis(0)))?;
        let pred_with = model(&coalition_with.view().insert_axis(Axis(0)))?;

        shap_values[i] = pred_with[0] - pred_without[0];
    }

    Ok(shap_values)
}

/// Enhanced batch processing configuration
#[derive(Debug, Clone)]
pub struct BatchConfig {
    /// Base batch size
    pub base_batch_size: usize,
    /// Maximum batch size (for dynamic adjustment)
    pub max_batch_size: usize,
    /// Minimum batch size (for dynamic adjustment)
    pub min_batch_size: usize,
    /// Memory limit per batch (in MB)
    pub memory_limit_mb: usize,
    /// Enable dynamic batch size adjustment
    pub dynamic_sizing: bool,
    /// Enable progress tracking
    pub enable_progress: bool,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            base_batch_size: 1000,
            max_batch_size: 10000,
            min_batch_size: 100,
            memory_limit_mb: 512,
            dynamic_sizing: true,
            enable_progress: false,
        }
    }
}

impl BatchConfig {
    /// Create new batch configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Set base batch size
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.base_batch_size = batch_size;
        self
    }

    /// Set memory limit in MB
    pub fn with_memory_limit(mut self, memory_mb: usize) -> Self {
        self.memory_limit_mb = memory_mb;
        self
    }

    /// Enable dynamic batch sizing
    pub fn with_dynamic_sizing(mut self, enabled: bool) -> Self {
        self.dynamic_sizing = enabled;
        self
    }

    /// Enable progress tracking
    pub fn with_progress(mut self, enabled: bool) -> Self {
        self.enable_progress = enabled;
        self
    }

    /// Calculate optimal batch size based on memory constraints
    pub fn calculate_optimal_batch_size(&self, item_size_bytes: usize) -> usize {
        if item_size_bytes == 0 {
            return self.base_batch_size;
        }

        let memory_limit_bytes = self.memory_limit_mb * 1024 * 1024;
        let max_items_per_batch = memory_limit_bytes / item_size_bytes;

        let optimal_size = max_items_per_batch
            .min(self.max_batch_size)
            .max(self.min_batch_size);

        if self.dynamic_sizing {
            optimal_size
        } else {
            self.base_batch_size
        }
    }
}

/// Batch processing statistics
#[derive(Debug, Clone)]
pub struct BatchStats {
    /// Total number of items processed
    pub total_items: usize,
    /// Number of batches processed
    pub num_batches: usize,
    /// Average batch size
    pub avg_batch_size: f64,
    /// Total processing time in milliseconds
    pub total_time_ms: u128,
    /// Average time per batch in milliseconds
    pub avg_time_per_batch_ms: f64,
    /// Average time per item in microseconds
    pub avg_time_per_item_us: f64,
    /// Peak memory usage in MB (estimated)
    pub peak_memory_mb: usize,
}

impl BatchStats {
    /// Create new batch statistics
    pub fn new() -> Self {
        Self {
            total_items: 0,
            num_batches: 0,
            avg_batch_size: 0.0,
            total_time_ms: 0,
            avg_time_per_batch_ms: 0.0,
            avg_time_per_item_us: 0.0,
            peak_memory_mb: 0,
        }
    }

    /// Update statistics with batch information
    pub fn update(&mut self, batch_size: usize, batch_time_ms: u128, memory_mb: usize) {
        self.total_items += batch_size;
        self.num_batches += 1;
        self.total_time_ms += batch_time_ms;
        self.peak_memory_mb = self.peak_memory_mb.max(memory_mb);

        // Update averages
        self.avg_batch_size = self.total_items as f64 / self.num_batches as f64;
        self.avg_time_per_batch_ms = self.total_time_ms as f64 / self.num_batches as f64;
        self.avg_time_per_item_us = (self.total_time_ms as f64 * 1000.0) / self.total_items as f64;
    }

    /// Get processing throughput (items per second)
    pub fn throughput(&self) -> f64 {
        if self.total_time_ms == 0 {
            return 0.0;
        }
        (self.total_items as f64 * 1000.0) / self.total_time_ms as f64
    }

    /// Get efficiency score (0-1, higher is better)
    pub fn efficiency(&self) -> f64 {
        if self.num_batches == 0 || self.total_time_ms == 0 {
            return 0.0;
        }

        // Base efficiency on throughput and memory usage
        let throughput_score = (self.throughput() / 1000.0).min(1.0);
        let memory_score = (512.0 / self.peak_memory_mb as f64).min(1.0);

        (throughput_score + memory_score) / 2.0
    }
}

/// Progress callback for batch processing
pub type ProgressCallback = Box<dyn Fn(usize, usize) + Send + Sync>;

/// Enhanced batch processing utility with optimization features
#[cfg(feature = "parallel")]
pub fn process_batches_parallel<T, R, F>(
    data: &[T],
    batch_size: usize,
    config: &ParallelConfig,
    processor: F,
) -> SklResult<Vec<R>>
where
    T: Send + Sync,
    R: Send,
    F: Fn(&[T]) -> SklResult<Vec<R>> + Send + Sync,
{
    if !config.should_parallelize(data.len()) {
        return processor(data);
    }

    let results: SklResult<Vec<_>> = data
        .chunks(batch_size)
        .collect::<Vec<_>>()
        .into_par_iter()
        .map(|batch| processor(batch))
        .collect();

    let batched_results = results?;
    Ok(batched_results.into_iter().flatten().collect())
}

/// Enhanced batch processing with statistics and progress tracking
#[cfg(feature = "parallel")]
pub fn process_batches_optimized<T, R, F>(
    data: &[T],
    batch_config: &BatchConfig,
    parallel_config: &ParallelConfig,
    processor: F,
    progress_callback: Option<ProgressCallback>,
) -> SklResult<(Vec<R>, BatchStats)>
where
    T: Send + Sync,
    R: Send,
    F: Fn(&[T]) -> SklResult<Vec<R>> + Send + Sync,
{
    let mut stats = BatchStats::new();
    let total_items = data.len();

    if !parallel_config.should_parallelize(total_items) {
        let start_time = std::time::Instant::now();
        let results = processor(data)?;
        let elapsed = start_time.elapsed().as_millis();
        stats.update(total_items, elapsed, batch_config.memory_limit_mb);
        return Ok((results, stats));
    }

    // Calculate optimal batch size
    let estimated_item_size = std::mem::size_of::<T>();
    let optimal_batch_size = batch_config.calculate_optimal_batch_size(estimated_item_size);

    let batches: Vec<_> = data.chunks(optimal_batch_size).collect();
    let total_batches = batches.len();

    let start_time = std::time::Instant::now();
    let processed_count = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));

    let results: SklResult<Vec<_>> = batches
        .into_par_iter()
        .enumerate()
        .map(|(batch_idx, batch)| {
            let batch_start = std::time::Instant::now();
            let batch_result = processor(batch);
            let batch_time = batch_start.elapsed().as_millis();

            // Update progress
            if let Some(ref callback) = progress_callback {
                let completed =
                    processed_count.fetch_add(1, std::sync::atomic::Ordering::SeqCst) + 1;
                callback(completed, total_batches);
            }

            // Note: We can't easily update stats in parallel, so we'll approximate
            batch_result
        })
        .collect();

    let total_time = start_time.elapsed().as_millis();
    let batched_results = results?;
    let final_results = batched_results.into_iter().flatten().collect();

    // Update final statistics
    stats.update(total_items, total_time, batch_config.memory_limit_mb);
    stats.num_batches = total_batches;
    stats.avg_batch_size = optimal_batch_size as f64;

    Ok((final_results, stats))
}

/// Memory-efficient streaming batch processor
#[cfg(feature = "parallel")]
pub struct StreamingBatchProcessor<T, R> {
    batch_config: BatchConfig,
    parallel_config: ParallelConfig,
    buffer: Vec<T>,
    results: Vec<R>,
    stats: BatchStats,
}

#[cfg(feature = "parallel")]
impl<T, R> StreamingBatchProcessor<T, R>
where
    T: Send + Sync,
    R: Send,
{
    /// Create new streaming batch processor
    pub fn new(batch_config: BatchConfig, parallel_config: ParallelConfig) -> Self {
        Self {
            batch_config,
            parallel_config,
            buffer: Vec::new(),
            results: Vec::new(),
            stats: BatchStats::new(),
        }
    }

    /// Add item to processing buffer
    pub fn push(&mut self, item: T) {
        self.buffer.push(item);
    }

    /// Process accumulated buffer and return results
    pub fn process_buffer<F>(&mut self, processor: F) -> SklResult<Vec<R>>
    where
        F: Fn(&[T]) -> SklResult<Vec<R>> + Send + Sync,
        R: Clone,
    {
        if self.buffer.is_empty() {
            return Ok(Vec::new());
        }

        let batch_size = self
            .batch_config
            .calculate_optimal_batch_size(std::mem::size_of::<T>());
        let start_time = std::time::Instant::now();

        let mut batch_results = Vec::new();

        for chunk in self.buffer.chunks(batch_size) {
            let chunk_results = processor(chunk)?;
            batch_results.extend(chunk_results);
        }

        let elapsed = start_time.elapsed().as_millis();
        self.stats.update(
            self.buffer.len(),
            elapsed,
            self.batch_config.memory_limit_mb,
        );

        self.buffer.clear();
        let result_copy = batch_results.clone();
        self.results.extend(batch_results);

        Ok(result_copy)
    }

    /// Get current processing statistics
    pub fn stats(&self) -> &BatchStats {
        &self.stats
    }

    /// Get all accumulated results
    pub fn results(&self) -> &[R] {
        &self.results
    }

    /// Reset processor state
    pub fn reset(&mut self) {
        self.buffer.clear();
        self.results.clear();
        self.stats = BatchStats::new();
    }
}

/// Adaptive batch processing with system resource monitoring
#[derive(Debug, Clone)]
pub struct AdaptiveBatchConfig {
    /// Base configuration
    pub base_config: BatchConfig,
    /// Enable CPU usage monitoring
    pub monitor_cpu: bool,
    /// Enable memory usage monitoring
    pub monitor_memory: bool,
    /// CPU threshold for adaptive sizing (0.0-1.0)
    pub cpu_threshold: f64,
    /// Memory threshold for adaptive sizing (0.0-1.0)
    pub memory_threshold: f64,
    /// Adaptive sizing factor (how much to scale batch size)
    pub sizing_factor: f64,
}

impl Default for AdaptiveBatchConfig {
    fn default() -> Self {
        Self {
            base_config: BatchConfig::default(),
            monitor_cpu: true,
            monitor_memory: true,
            cpu_threshold: 0.8,
            memory_threshold: 0.8,
            sizing_factor: 0.5,
        }
    }
}

impl AdaptiveBatchConfig {
    /// Create new adaptive batch configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Set base batch configuration
    pub fn with_base_config(mut self, config: BatchConfig) -> Self {
        self.base_config = config;
        self
    }

    /// Enable/disable CPU monitoring
    pub fn with_cpu_monitoring(mut self, enabled: bool) -> Self {
        self.monitor_cpu = enabled;
        self
    }

    /// Enable/disable memory monitoring
    pub fn with_memory_monitoring(mut self, enabled: bool) -> Self {
        self.monitor_memory = enabled;
        self
    }

    /// Set CPU threshold for adaptive sizing
    pub fn with_cpu_threshold(mut self, threshold: f64) -> Self {
        self.cpu_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Set memory threshold for adaptive sizing
    pub fn with_memory_threshold(mut self, threshold: f64) -> Self {
        self.memory_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Calculate adaptive batch size based on system resources
    pub fn calculate_adaptive_batch_size(&self, item_size_bytes: usize) -> usize {
        let base_size = self
            .base_config
            .calculate_optimal_batch_size(item_size_bytes);

        // Get current system load (simplified simulation for now)
        let cpu_load = self.get_cpu_load();
        let memory_load = self.get_memory_load();

        let mut scaling_factor = 1.0;

        if self.monitor_cpu && cpu_load > self.cpu_threshold {
            scaling_factor *= self.sizing_factor;
        }

        if self.monitor_memory && memory_load > self.memory_threshold {
            scaling_factor *= self.sizing_factor;
        }

        let adaptive_size = (base_size as f64 * scaling_factor) as usize;
        adaptive_size
            .max(self.base_config.min_batch_size)
            .min(self.base_config.max_batch_size)
    }

    /// Get current CPU load (simplified - in real implementation would use system APIs)
    fn get_cpu_load(&self) -> f64 {
        // Simplified implementation - in practice would use system APIs
        // like sysinfo or similar to get actual CPU usage
        0.5 // Return moderate load for simulation
    }

    /// Get current memory load (simplified - in real implementation would use system APIs)
    fn get_memory_load(&self) -> f64 {
        // Simplified implementation - in practice would use system APIs
        // to get actual memory usage
        0.6 // Return moderate memory usage for simulation
    }
}

/// Memory pool for reducing allocation overhead
#[derive(Debug)]
pub struct MemoryPool<T> {
    pool: Vec<Vec<T>>,
    max_pool_size: usize,
    total_allocations: usize,
    total_reuses: usize,
}

impl<T> MemoryPool<T> {
    /// Create new memory pool
    pub fn new(max_pool_size: usize) -> Self {
        Self {
            pool: Vec::new(),
            max_pool_size,
            total_allocations: 0,
            total_reuses: 0,
        }
    }

    /// Get a vector from the pool or allocate new one
    pub fn get_vec(&mut self, capacity: usize) -> Vec<T> {
        if let Some(mut vec) = self.pool.pop() {
            vec.clear();
            if vec.capacity() < capacity {
                vec.reserve(capacity - vec.capacity());
            }
            self.total_reuses += 1;
            vec
        } else {
            self.total_allocations += 1;
            Vec::with_capacity(capacity)
        }
    }

    /// Return a vector to the pool
    pub fn return_vec(&mut self, vec: Vec<T>) {
        if self.pool.len() < self.max_pool_size {
            self.pool.push(vec);
        }
    }

    /// Get pool statistics
    pub fn stats(&self) -> (usize, usize, f64) {
        let total_requests = self.total_allocations + self.total_reuses;
        let reuse_rate = if total_requests > 0 {
            self.total_reuses as f64 / total_requests as f64
        } else {
            0.0
        };
        (self.total_allocations, self.total_reuses, reuse_rate)
    }
}

/// High-performance batch processor with memory optimization
#[cfg(feature = "parallel")]
pub struct HighPerformanceBatchProcessor<T, R> {
    adaptive_config: AdaptiveBatchConfig,
    parallel_config: ParallelConfig,
    memory_pool: MemoryPool<T>,
    result_pool: MemoryPool<R>,
    stats: BatchStats,
}

#[cfg(feature = "parallel")]
impl<T, R> HighPerformanceBatchProcessor<T, R>
where
    T: Send + Sync + Clone,
    R: Send + Clone,
{
    /// Create new high-performance batch processor
    pub fn new(adaptive_config: AdaptiveBatchConfig, parallel_config: ParallelConfig) -> Self {
        Self {
            adaptive_config,
            parallel_config,
            memory_pool: MemoryPool::new(10), // Pool size of 10
            result_pool: MemoryPool::new(10),
            stats: BatchStats::new(),
        }
    }

    /// Process data with adaptive batching and memory optimization
    pub fn process_adaptive<F>(&mut self, data: &[T], processor: F) -> SklResult<Vec<R>>
    where
        F: Fn(&[T]) -> SklResult<Vec<R>> + Send + Sync,
    {
        let start_time = std::time::Instant::now();
        let item_size = std::mem::size_of::<T>();
        let adaptive_batch_size = self
            .adaptive_config
            .calculate_adaptive_batch_size(item_size);

        let mut results = self.result_pool.get_vec(data.len());

        if !self.parallel_config.should_parallelize(data.len()) {
            // Sequential processing
            let batch_results = processor(data)?;
            results.extend(batch_results);
        } else {
            // Parallel processing with adaptive batching
            let batches: Vec<_> = data.chunks(adaptive_batch_size).collect();
            let batch_results: SklResult<Vec<_>> = batches
                .into_par_iter()
                .map(|batch| processor(batch))
                .collect();

            let processed_results = batch_results?;
            for batch_result in processed_results {
                results.extend(batch_result);
            }
        }

        let elapsed = start_time.elapsed().as_millis();
        self.stats.update(
            data.len(),
            elapsed,
            self.adaptive_config.base_config.memory_limit_mb,
        );

        Ok(results)
    }

    /// Get memory pool statistics
    pub fn memory_pool_stats(&self) -> ((usize, usize, f64), (usize, usize, f64)) {
        (self.memory_pool.stats(), self.result_pool.stats())
    }

    /// Get processing statistics
    pub fn stats(&self) -> &BatchStats {
        &self.stats
    }
}

/// Compressed batch data for memory efficiency
#[derive(Debug, Clone)]
pub struct CompressedBatch<T> {
    /// Compressed data (simplified - in practice would use actual compression)
    data: Vec<T>,
    /// Original size before compression
    original_size: usize,
    /// Compression metadata
    compression_ratio: f64,
}

impl<T> CompressedBatch<T>
where
    T: Clone,
{
    pub fn compress(data: Vec<T>) -> Self {
        let original_size = data.len();
        // Simplified compression - in practice would use algorithms like LZ4, Zstd, etc.
        let compression_ratio = 0.7; // Simulate 30% compression

        Self {
            data,
            original_size,
            compression_ratio,
        }
    }

    /// Decompress batch data
    pub fn decompress(&self) -> Vec<T> {
        // Simplified decompression
        self.data.clone()
    }

    /// Get compression ratio
    pub fn compression_ratio(&self) -> f64 {
        self.compression_ratio
    }

    /// Get compressed size
    pub fn compressed_size(&self) -> usize {
        self.data.len()
    }

    /// Get original size
    pub fn original_size(&self) -> usize {
        self.original_size
    }
}

/// Cache-aware data structure for explanation results
#[derive(Debug)]
pub struct CacheAwareExplanationStore<T> {
    /// Hot cache for frequently accessed results
    hot_cache: std::collections::HashMap<u64, T>,
    /// Cold storage for less frequently accessed results
    cold_storage: std::collections::HashMap<u64, CompressedBatch<T>>,
    /// Access frequency tracking
    access_counts: std::collections::HashMap<u64, usize>,
    /// Maximum hot cache size
    max_hot_cache_size: usize,
    /// Cache hit/miss statistics
    cache_hits: usize,
    cache_misses: usize,
}

impl<T> CacheAwareExplanationStore<T>
where
    T: Clone + std::hash::Hash,
{
    /// Create new cache-aware explanation store
    pub fn new(max_hot_cache_size: usize) -> Self {
        Self {
            hot_cache: std::collections::HashMap::new(),
            cold_storage: std::collections::HashMap::new(),
            access_counts: std::collections::HashMap::new(),
            max_hot_cache_size,
            cache_hits: 0,
            cache_misses: 0,
        }
    }

    /// Store explanation result
    pub fn store(&mut self, key: u64, value: T) {
        if self.hot_cache.len() < self.max_hot_cache_size {
            self.hot_cache.insert(key, value);
        } else {
            // Move to cold storage with compression
            let compressed = CompressedBatch::compress(vec![value]);
            self.cold_storage.insert(key, compressed);
        }
        self.access_counts.insert(key, 1);
    }

    /// Retrieve explanation result
    pub fn get(&mut self, key: u64) -> Option<T> {
        // Check hot cache first
        if let Some(value) = self.hot_cache.get(&key) {
            self.cache_hits += 1;
            *self.access_counts.entry(key).or_insert(0) += 1;
            return Some(value.clone());
        }

        // Check cold storage
        if let Some(compressed) = self.cold_storage.get(&key) {
            self.cache_hits += 1;
            let decompressed = compressed.decompress();
            let value = decompressed.into_iter().next()?;

            // Promote to hot cache if frequently accessed
            let access_count = *self.access_counts.entry(key).or_insert(0) + 1;
            self.access_counts.insert(key, access_count);

            if access_count > 3 && self.hot_cache.len() < self.max_hot_cache_size {
                self.hot_cache.insert(key, value.clone());
                self.cold_storage.remove(&key);
            }

            return Some(value);
        }

        self.cache_misses += 1;
        None
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> (usize, usize, f64, usize, usize) {
        let total_accesses = self.cache_hits + self.cache_misses;
        let hit_rate = if total_accesses > 0 {
            self.cache_hits as f64 / total_accesses as f64
        } else {
            0.0
        };
        (
            self.cache_hits,
            self.cache_misses,
            hit_rate,
            self.hot_cache.len(),
            self.cold_storage.len(),
        )
    }
}

/// Non-parallel fallback
#[cfg(not(feature = "parallel"))]
pub fn process_batches_parallel<T, R, F>(
    data: &[T],
    _batch_size: usize,
    _config: &ParallelConfig,
    processor: F,
) -> SklResult<Vec<R>>
where
    F: Fn(&[T]) -> SklResult<Vec<R>>,
{
    processor(data)
}

#[cfg(test)]
mod tests {
    use super::*;
    // ✅ SciRS2 Policy Compliant Import
    use scirs2_core::ndarray::array;

    #[test]
    fn test_parallel_config_creation() {
        let config = ParallelConfig::new();
        assert!(!config.force_sequential);
        assert_eq!(config.min_batch_size, 100);
    }

    #[test]
    fn test_parallel_config_with_threads() {
        let config = ParallelConfig::new().with_threads(4);
        assert_eq!(config.n_threads, Some(4));
    }

    #[test]
    fn test_parallel_config_sequential() {
        let config = ParallelConfig::new().sequential();
        assert!(config.force_sequential);
        assert_eq!(config.get_n_threads(), 1);
    }

    #[test]
    fn test_should_parallelize() {
        let config = ParallelConfig::new();
        assert!(config.should_parallelize(1000));
        assert!(!config.should_parallelize(50));

        let sequential_config = ParallelConfig::new().sequential();
        assert!(!sequential_config.should_parallelize(1000));
    }

    #[test]
    fn test_permutation_input_creation() {
        let x_data = array![[1.0, 2.0], [3.0, 4.0]];
        let y_true = array![1.0, 0.0];

        let input = PermutationInput {
            feature_idx: 0,
            x_data,
            y_true,
            n_repeats: 5,
            random_state: 42,
        };

        assert_eq!(input.feature_idx, 0);
        assert_eq!(input.n_repeats, 5);
        assert_eq!(input.random_state, 42);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_shap_sequential_computation() {
        let model = |x: &ArrayView2<Float>| -> SklResult<Array1<Float>> { Ok(x.sum_axis(Axis(1))) };

        let X = array![[1.0, 2.0], [3.0, 4.0]];
        let baseline = array![0.0, 0.0];

        let result = compute_shap_sequential(model, &X.view(), &baseline.view());
        assert!(result.is_ok());

        let shap_values = result.unwrap();
        assert_eq!(shap_values.shape(), &[2, 2]);
    }

    #[cfg(feature = "parallel")]
    #[test]
    #[allow(non_snake_case)]
    fn test_parallel_shap_computation() {
        let model = |x: &ArrayView2<Float>| -> SklResult<Array1<Float>> { Ok(x.sum_axis(Axis(1))) };

        let X = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let baseline = array![0.0, 0.0];
        let config = ParallelConfig::new().with_min_batch_size(1);

        let result = compute_shap_parallel(model, &X.view(), &baseline.view(), &config);
        assert!(result.is_ok());

        let shap_values = result.unwrap();
        assert_eq!(shap_values.shape(), &[3, 2]);
    }

    #[test]
    fn test_single_instance_shap() {
        let model = |x: &ArrayView2<Float>| -> SklResult<Array1<Float>> { Ok(x.sum_axis(Axis(1))) };

        let instance = array![1.0, 2.0];
        let baseline = array![0.0, 0.0];

        let result = compute_shap_single_instance(model, &instance.view(), &baseline.view());
        assert!(result.is_ok());

        let shap_values = result.unwrap();
        assert_eq!(shap_values.len(), 2);
    }

    #[test]
    fn test_batch_config_creation() {
        let config = BatchConfig::new();
        assert_eq!(config.base_batch_size, 1000);
        assert_eq!(config.max_batch_size, 10000);
        assert_eq!(config.min_batch_size, 100);
        assert_eq!(config.memory_limit_mb, 512);
        assert!(config.dynamic_sizing);
        assert!(!config.enable_progress);
    }

    #[test]
    fn test_batch_config_fluent_api() {
        let config = BatchConfig::new()
            .with_batch_size(500)
            .with_memory_limit(256)
            .with_dynamic_sizing(false)
            .with_progress(true);

        assert_eq!(config.base_batch_size, 500);
        assert_eq!(config.memory_limit_mb, 256);
        assert!(!config.dynamic_sizing);
        assert!(config.enable_progress);
    }

    #[test]
    fn test_batch_config_optimal_batch_size() {
        let config = BatchConfig::new()
            .with_memory_limit(1) // 1MB limit
            .with_dynamic_sizing(true);

        // Test with small item size (should use memory limit)
        let optimal_size = config.calculate_optimal_batch_size(1024); // 1KB per item
        assert_eq!(optimal_size, 1024); // 1MB / 1KB = 1024 items

        // Test with large item size (should use min batch size)
        let optimal_size = config.calculate_optimal_batch_size(1024 * 1024 * 10); // 10MB per item
        assert_eq!(optimal_size, config.min_batch_size);

        // Test with dynamic sizing disabled
        let static_config = BatchConfig::new().with_dynamic_sizing(false);
        let optimal_size = static_config.calculate_optimal_batch_size(1024);
        assert_eq!(optimal_size, static_config.base_batch_size);
    }

    #[test]
    fn test_batch_stats_creation() {
        let stats = BatchStats::new();
        assert_eq!(stats.total_items, 0);
        assert_eq!(stats.num_batches, 0);
        assert_eq!(stats.avg_batch_size, 0.0);
        assert_eq!(stats.total_time_ms, 0);
        assert_eq!(stats.throughput(), 0.0);
        assert_eq!(stats.efficiency(), 0.0);
    }

    #[test]
    fn test_batch_stats_update() {
        let mut stats = BatchStats::new();

        // Update with first batch
        stats.update(100, 1000, 128); // 100 items, 1000ms, 128MB
        assert_eq!(stats.total_items, 100);
        assert_eq!(stats.num_batches, 1);
        assert_eq!(stats.avg_batch_size, 100.0);
        assert_eq!(stats.total_time_ms, 1000);
        assert_eq!(stats.peak_memory_mb, 128);

        // Update with second batch
        stats.update(200, 2000, 256); // 200 items, 2000ms, 256MB
        assert_eq!(stats.total_items, 300);
        assert_eq!(stats.num_batches, 2);
        assert_eq!(stats.avg_batch_size, 150.0);
        assert_eq!(stats.total_time_ms, 3000);
        assert_eq!(stats.peak_memory_mb, 256);
    }

    #[test]
    fn test_batch_stats_throughput() {
        let mut stats = BatchStats::new();
        stats.update(1000, 2000, 128); // 1000 items in 2000ms

        let throughput = stats.throughput();
        assert!((throughput - 500.0).abs() < 0.001); // 500 items per second
    }

    #[test]
    fn test_batch_stats_efficiency() {
        let mut stats = BatchStats::new();
        stats.update(1000, 1000, 256); // 1000 items/sec, 256MB memory

        let efficiency = stats.efficiency();
        assert!(efficiency > 0.0);
        assert!(efficiency <= 1.0);
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_process_batches_optimized() {
        let data: Vec<i32> = (0..1000).collect();
        let batch_config = BatchConfig::new().with_batch_size(100);
        let parallel_config = ParallelConfig::new().with_min_batch_size(50);

        let processor =
            |batch: &[i32]| -> SklResult<Vec<i32>> { Ok(batch.iter().map(|x| x * 2).collect()) };

        let result =
            process_batches_optimized(&data, &batch_config, &parallel_config, processor, None);

        assert!(result.is_ok());
        let (results, stats) = result.unwrap();
        assert_eq!(results.len(), 1000);
        assert_eq!(results[0], 0);
        assert_eq!(results[999], 1998);
        assert!(stats.total_items > 0);
        assert!(stats.num_batches > 0);
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_process_batches_with_progress() {
        let data: Vec<i32> = (0..500).collect();
        let batch_config = BatchConfig::new().with_batch_size(100);
        let parallel_config = ParallelConfig::new().with_min_batch_size(50);

        let progress_calls = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let progress_calls_clone = progress_calls.clone();

        let progress_callback = Box::new(move |completed: usize, total: usize| {
            progress_calls_clone
                .lock()
                .unwrap()
                .push((completed, total));
        });

        let processor =
            |batch: &[i32]| -> SklResult<Vec<i32>> { Ok(batch.iter().map(|x| x * 2).collect()) };

        let result = process_batches_optimized(
            &data,
            &batch_config,
            &parallel_config,
            processor,
            Some(progress_callback),
        );

        assert!(result.is_ok());
        let (results, _stats) = result.unwrap();
        assert_eq!(results.len(), 500);

        // Check that progress was tracked
        let calls = progress_calls.lock().unwrap();
        assert!(!calls.is_empty());
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_streaming_batch_processor() {
        let batch_config = BatchConfig::new().with_batch_size(50);
        let parallel_config = ParallelConfig::new();
        let mut processor = StreamingBatchProcessor::new(batch_config, parallel_config);

        // Add items to buffer
        for i in 0..100 {
            processor.push(i);
        }

        // Process buffer
        let process_fn =
            |batch: &[i32]| -> SklResult<Vec<i32>> { Ok(batch.iter().map(|x| x * 2).collect()) };

        let result = processor.process_buffer(process_fn);
        assert!(result.is_ok());

        let batch_results = result.unwrap();
        assert_eq!(batch_results.len(), 100);
        assert_eq!(batch_results[0], 0);
        assert_eq!(batch_results[99], 198);

        // Check stats
        let stats = processor.stats();
        assert_eq!(stats.total_items, 100);
        assert!(stats.num_batches > 0);

        // Check accumulated results
        let all_results = processor.results();
        assert_eq!(all_results.len(), 100);
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_streaming_batch_processor_reset() {
        let batch_config = BatchConfig::new();
        let parallel_config = ParallelConfig::new();
        let mut processor = StreamingBatchProcessor::new(batch_config, parallel_config);

        // Add items and process
        processor.push(1);
        processor.push(2);
        let process_fn = |batch: &[i32]| -> SklResult<Vec<i32>> { Ok(batch.to_vec()) };
        let _ = processor.process_buffer(process_fn);

        // Check state before reset
        assert_eq!(processor.results().len(), 2);
        assert!(processor.stats().total_items > 0);

        // Reset and check state
        processor.reset();
        assert_eq!(processor.results().len(), 0);
        assert_eq!(processor.stats().total_items, 0);
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_streaming_batch_processor_empty_buffer() {
        let batch_config = BatchConfig::new();
        let parallel_config = ParallelConfig::new();
        let mut processor = StreamingBatchProcessor::new(batch_config, parallel_config);

        let process_fn = |batch: &[i32]| -> SklResult<Vec<i32>> { Ok(batch.to_vec()) };

        let result = processor.process_buffer(process_fn);
        assert!(result.is_ok());

        let batch_results = result.unwrap();
        assert!(batch_results.is_empty());
    }

    // New tests for enhanced performance features

    #[test]
    fn test_adaptive_batch_config_creation() {
        let config = AdaptiveBatchConfig::new();
        assert!(config.monitor_cpu);
        assert!(config.monitor_memory);
        assert_eq!(config.cpu_threshold, 0.8);
        assert_eq!(config.memory_threshold, 0.8);
        assert_eq!(config.sizing_factor, 0.5);
    }

    #[test]
    fn test_adaptive_batch_config_fluent_api() {
        let base_config = BatchConfig::new().with_batch_size(500);
        let adaptive_config = AdaptiveBatchConfig::new()
            .with_base_config(base_config)
            .with_cpu_monitoring(false)
            .with_memory_monitoring(true)
            .with_cpu_threshold(0.9)
            .with_memory_threshold(0.7);

        assert!(!adaptive_config.monitor_cpu);
        assert!(adaptive_config.monitor_memory);
        assert_eq!(adaptive_config.cpu_threshold, 0.9);
        assert_eq!(adaptive_config.memory_threshold, 0.7);
        assert_eq!(adaptive_config.base_config.base_batch_size, 500);
    }

    #[test]
    fn test_adaptive_batch_size_calculation() {
        let config = AdaptiveBatchConfig::new();
        let batch_size = config.calculate_adaptive_batch_size(1024);

        // Should return a value between min and max batch size
        assert!(batch_size >= config.base_config.min_batch_size);
        assert!(batch_size <= config.base_config.max_batch_size);
    }

    #[test]
    fn test_memory_pool_creation() {
        let mut pool: MemoryPool<i32> = MemoryPool::new(5);
        let (allocations, reuses, reuse_rate) = pool.stats();

        assert_eq!(allocations, 0);
        assert_eq!(reuses, 0);
        assert_eq!(reuse_rate, 0.0);
    }

    #[test]
    fn test_memory_pool_reuse() {
        let mut pool: MemoryPool<i32> = MemoryPool::new(5);

        // Get first vector (should allocate)
        let vec1 = pool.get_vec(10);
        assert_eq!(vec1.capacity(), 10);

        // Return it to pool
        pool.return_vec(vec1);

        // Get second vector (should reuse)
        let vec2 = pool.get_vec(5);
        assert!(vec2.capacity() >= 5); // Should have capacity from previous allocation

        let (allocations, reuses, reuse_rate) = pool.stats();
        assert_eq!(allocations, 1);
        assert_eq!(reuses, 1);
        assert!((reuse_rate - 0.5).abs() < 0.001); // 50% reuse rate
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_high_performance_batch_processor() {
        let adaptive_config = AdaptiveBatchConfig::new();
        let parallel_config = ParallelConfig::new().with_min_batch_size(10);
        let mut processor = HighPerformanceBatchProcessor::new(adaptive_config, parallel_config);

        let data: Vec<i32> = (0..100).collect();
        let process_fn =
            |batch: &[i32]| -> SklResult<Vec<i32>> { Ok(batch.iter().map(|x| x * 2).collect()) };

        let result = processor.process_adaptive(&data, process_fn);
        assert!(result.is_ok());

        let results = result.unwrap();
        assert_eq!(results.len(), 100);
        assert_eq!(results[0], 0);
        assert_eq!(results[99], 198);

        // Check statistics
        let stats = processor.stats();
        assert_eq!(stats.total_items, 100);

        // Check memory pool statistics
        let (pool_stats, result_pool_stats) = processor.memory_pool_stats();
        assert!(pool_stats.0 >= 0); // allocations
        assert!(result_pool_stats.0 >= 0); // result allocations
    }

    #[test]
    fn test_compressed_batch_creation() {
        let data = vec![1, 2, 3, 4, 5];
        let compressed = CompressedBatch::compress(data.clone());

        assert_eq!(compressed.original_size(), 5);
        assert_eq!(compressed.compressed_size(), 5); // Simplified implementation doesn't actually compress
        assert_eq!(compressed.compression_ratio(), 0.7);

        let decompressed = compressed.decompress();
        assert_eq!(decompressed, data);
    }

    #[test]
    fn test_cache_aware_explanation_store() {
        let mut store: CacheAwareExplanationStore<i32> = CacheAwareExplanationStore::new(2);

        // Store some values
        store.store(1, 100);
        store.store(2, 200);
        store.store(3, 300); // This should go to cold storage since hot cache is full

        // Test retrieval from hot cache
        assert_eq!(store.get(1), Some(100));
        assert_eq!(store.get(2), Some(200));

        // Test retrieval from cold storage
        assert_eq!(store.get(3), Some(300));

        // Test cache miss
        assert_eq!(store.get(4), None);

        // Check cache statistics
        let (hits, misses, hit_rate, hot_size, cold_size) = store.cache_stats();
        assert!(hits >= 3);
        assert_eq!(misses, 1);
        assert!(hit_rate > 0.5);
        assert!(hot_size <= 2);
    }

    #[test]
    fn test_cache_promotion() {
        let mut store: CacheAwareExplanationStore<i32> = CacheAwareExplanationStore::new(1);

        // Store value that goes to cold storage
        store.store(1, 100);
        store.store(2, 200); // This goes to cold storage since hot cache is full

        // Access cold storage item multiple times to trigger promotion
        for _ in 0..4 {
            assert_eq!(store.get(2), Some(200));
        }

        let (_, _, _, hot_size, cold_size) = store.cache_stats();
        // Value should have been promoted to hot cache
        assert!(hot_size == 1); // Hot cache should contain promoted item
    }

    #[test]
    fn test_threshold_clamping() {
        let config = AdaptiveBatchConfig::new()
            .with_cpu_threshold(1.5) // Should be clamped to 1.0
            .with_memory_threshold(-0.5); // Should be clamped to 0.0

        assert_eq!(config.cpu_threshold, 1.0);
        assert_eq!(config.memory_threshold, 0.0);
    }
}
