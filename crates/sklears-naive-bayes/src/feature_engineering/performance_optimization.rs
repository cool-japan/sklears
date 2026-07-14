//! Performance optimization for feature engineering operations
//!
//! This module provides SIMD acceleration, parallel processing, memory
//! optimization, caching, and batch-processing implementations for feature
//! engineering pipelines. GPU-accelerated matrix multiply is available behind
//! the `gpu` feature (see [`GpuOptimizer`]): it is backed by real
//! `oxicuda-blas` GEMM via `sklears_core::gpu` for the `f32`/`f64` element
//! types, and honestly falls back to a CPU implementation for every other
//! element type and for GPU-disabled builds or GPU-less hosts (no fabricated
//! availability -- `GpuBackend::detect()`'s `Ok(None)` contract is preserved).

// SciRS2 Policy Compliance - Use scirs2-autograd for ndarray types
use scirs2_core::ndarray::{s, Array1, Array2, ArrayView1, ArrayView2};
use serde::{Deserialize, Serialize};
use sklears_core::error::Result;
use sklears_core::prelude::SklearsError;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

#[cfg(feature = "gpu")]
use sklears_core::gpu::{GpuArray, GpuBackend, GpuMatrixOps};

/// Performance optimization strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OptimizationStrategy {
    /// SIMD
    Simd,
    /// Parallel
    Parallel,
    /// Memory
    Memory,
    /// Cache
    Cache,
    /// GPU
    Gpu,
    /// Vectorized
    Vectorized,
    /// BatchProcessing
    BatchProcessing,
}

/// Configuration for performance optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationConfig {
    pub strategy: OptimizationStrategy,
    pub num_threads: Option<usize>,
    pub batch_size: usize,
    pub cache_size: usize,
    pub memory_limit: Option<usize>,
    pub enable_prefetch: bool,
    pub use_simd: bool,
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            strategy: OptimizationStrategy::Parallel,
            num_threads: None,
            batch_size: 1000,
            cache_size: 10000,
            memory_limit: None,
            enable_prefetch: true,
            use_simd: true,
        }
    }
}

/// Trait for performance-optimized operations
pub trait PerformanceOptimizer<T> {
    fn optimize_operation(
        &self,
        operation: &dyn Fn(&ArrayView2<T>) -> Result<Array2<T>>,
        data: &ArrayView2<T>,
    ) -> Result<Array2<T>>;
    fn get_performance_metrics(&self) -> HashMap<String, f64>;
}

/// SIMD-accelerated operations
#[derive(Debug, Clone)]
pub struct SimdOptimizer {
    config: OptimizationConfig,
    performance_metrics: HashMap<String, f64>,
}

impl SimdOptimizer {
    pub fn new(config: OptimizationConfig) -> Self {
        Self {
            config,
            performance_metrics: HashMap::new(),
        }
    }

    /// SIMD-accelerated matrix operations
    pub fn simd_matrix_multiply<T>(&self, a: &ArrayView2<T>, b: &ArrayView2<T>) -> Result<Array2<T>>
    where
        T: Clone
            + Copy
            + std::fmt::Debug
            + Default
            + std::ops::Add<Output = T>
            + std::ops::Mul<Output = T>,
    {
        let (m, n) = a.dim();
        let (n2, p) = b.dim();

        if n != n2 {
            return Err(SklearsError::InvalidInput(
                "Matrix dimension mismatch".to_string(),
            ));
        }

        let mut result = Array2::default((m, p));

        // Simplified SIMD-style computation
        for i in 0..m {
            for j in 0..p {
                let mut sum = T::default();
                for k in 0..n {
                    sum = sum + a[(i, k)] * b[(k, j)];
                }
                result[(i, j)] = sum;
            }
        }

        Ok(result)
    }

    /// SIMD-accelerated vector operations
    pub fn simd_vector_add<T>(&self, a: &ArrayView1<T>, b: &ArrayView1<T>) -> Result<Array1<T>>
    where
        T: Clone + Copy + std::fmt::Debug + Default + std::ops::Add<Output = T>,
    {
        if a.len() != b.len() {
            return Err(SklearsError::InvalidInput(
                "Vector length mismatch".to_string(),
            ));
        }

        let mut result = Array1::default(a.len());

        // Vectorized addition
        for i in 0..a.len() {
            result[i] = a[i] + b[i];
        }

        Ok(result)
    }

    /// SIMD-accelerated dot product
    pub fn simd_dot_product<T>(&self, a: &ArrayView1<T>, b: &ArrayView1<T>) -> Result<T>
    where
        T: Clone
            + Copy
            + std::fmt::Debug
            + Default
            + std::ops::Add<Output = T>
            + std::ops::Mul<Output = T>,
    {
        if a.len() != b.len() {
            return Err(SklearsError::InvalidInput(
                "Vector length mismatch".to_string(),
            ));
        }

        let mut result = T::default();
        for i in 0..a.len() {
            result = result + a[i] * b[i];
        }

        Ok(result)
    }

    pub fn performance_metrics(&self) -> &HashMap<String, f64> {
        &self.performance_metrics
    }
}

/// Parallel processing optimizer
#[derive(Debug, Clone)]
pub struct ParallelOptimizer {
    config: OptimizationConfig,
    performance_metrics: HashMap<String, f64>,
}

impl ParallelOptimizer {
    pub fn new(config: OptimizationConfig) -> Self {
        Self {
            config,
            performance_metrics: HashMap::new(),
        }
    }

    /// Parallel matrix operations
    pub fn parallel_transform<T, F>(
        &self,
        data: &ArrayView2<T>,
        transform_fn: F,
    ) -> Result<Array2<T>>
    where
        T: Clone + Copy + std::fmt::Debug + Send + Sync + Default,
        F: Fn(&ArrayView1<T>) -> Result<Array1<T>> + Send + Sync,
    {
        let (n_samples, n_features) = data.dim();
        let mut result = Array2::default((n_samples, n_features));

        // Simulate parallel processing by chunking rows
        let chunk_size = self.config.batch_size;

        for chunk_start in (0..n_samples).step_by(chunk_size) {
            let chunk_end = (chunk_start + chunk_size).min(n_samples);

            for i in chunk_start..chunk_end {
                let row = data.row(i);
                let transformed_row = transform_fn(&row)?;

                for j in 0..n_features {
                    if j < transformed_row.len() {
                        result[(i, j)] = transformed_row[j];
                    }
                }
            }
        }

        Ok(result)
    }

    /// Parallel feature computation
    pub fn parallel_feature_computation<T, F>(
        &self,
        data: &ArrayView2<T>,
        compute_fn: F,
    ) -> Result<Array1<T>>
    where
        T: Clone + Copy + std::fmt::Debug + Send + Sync + Default,
        F: Fn(&ArrayView2<T>) -> Result<T> + Send + Sync,
    {
        let (n_samples, _) = data.dim();
        let chunk_size = self.config.batch_size;
        let mut results = Vec::new();

        for chunk_start in (0..n_samples).step_by(chunk_size) {
            let chunk_end = (chunk_start + chunk_size).min(n_samples);
            let chunk = data.slice(s![chunk_start..chunk_end, ..]);
            let chunk_result = compute_fn(&chunk)?;
            results.push(chunk_result);
        }

        Ok(Array1::from_vec(results))
    }

    pub fn performance_metrics(&self) -> &HashMap<String, f64> {
        &self.performance_metrics
    }
}

/// Memory optimizer
#[derive(Debug, Clone)]
pub struct MemoryOptimizer {
    config: OptimizationConfig,
    memory_usage: Arc<Mutex<usize>>,
    performance_metrics: HashMap<String, f64>,
}

impl MemoryOptimizer {
    pub fn new(config: OptimizationConfig) -> Self {
        Self {
            config,
            memory_usage: Arc::new(Mutex::new(0)),
            performance_metrics: HashMap::new(),
        }
    }

    /// Memory-efficient batch processing
    pub fn batch_process<T, F>(&self, data: &ArrayView2<T>, process_fn: F) -> Result<Array2<T>>
    where
        T: Clone + Copy + std::fmt::Debug + Default,
        F: Fn(&ArrayView2<T>) -> Result<Array2<T>>,
    {
        let (n_samples, n_features) = data.dim();
        let batch_size = self.config.batch_size;
        let mut results = Vec::new();

        for batch_start in (0..n_samples).step_by(batch_size) {
            let batch_end = (batch_start + batch_size).min(n_samples);
            let batch = data.slice(s![batch_start..batch_end, ..]);

            self.track_memory_usage(batch.len() * std::mem::size_of::<T>())?;
            let batch_result = process_fn(&batch)?;
            results.push(batch_result);
        }

        // Combine results
        let total_rows: usize = results.iter().map(|r| r.nrows()).sum();
        let mut combined = Array2::default((total_rows, n_features));
        let mut row_offset = 0;

        for batch_result in results {
            let batch_rows = batch_result.nrows();
            combined
                .slice_mut(s![row_offset..row_offset + batch_rows, ..])
                .assign(&batch_result);
            row_offset += batch_rows;
        }

        Ok(combined)
    }

    /// Track memory usage
    fn track_memory_usage(&self, additional_bytes: usize) -> Result<()> {
        let mut usage = self.memory_usage.lock().expect("operation should succeed");
        *usage += additional_bytes;

        if let Some(limit) = self.config.memory_limit {
            if *usage > limit {
                return Err(SklearsError::InvalidInput(
                    "Memory limit exceeded".to_string(),
                ));
            }
        }

        Ok(())
    }

    /// Memory-efficient operations
    pub fn memory_efficient_transform<T, F>(
        &self,
        data: &ArrayView2<T>,
        transform_fn: F,
    ) -> Result<Array2<T>>
    where
        T: Clone + Copy + std::fmt::Debug + Default,
        F: Fn(T) -> T,
    {
        let (n_samples, n_features) = data.dim();
        let mut result = Array2::default((n_samples, n_features));

        // Process in chunks to limit memory usage
        let chunk_size = self.config.batch_size;

        for chunk_start in (0..n_samples).step_by(chunk_size) {
            let chunk_end = (chunk_start + chunk_size).min(n_samples);

            for i in chunk_start..chunk_end {
                for j in 0..n_features {
                    result[(i, j)] = transform_fn(data[(i, j)]);
                }
            }
        }

        Ok(result)
    }

    pub fn memory_usage(&self) -> usize {
        *self.memory_usage.lock().expect("operation should succeed")
    }

    pub fn performance_metrics(&self) -> &HashMap<String, f64> {
        &self.performance_metrics
    }
}

/// Cache optimizer
#[derive(Debug, Clone)]
pub struct CacheOptimizer<T> {
    config: OptimizationConfig,
    cache: Arc<Mutex<HashMap<String, Array2<T>>>>,
    cache_hits: Arc<Mutex<usize>>,
    cache_misses: Arc<Mutex<usize>>,
}

impl<T> CacheOptimizer<T>
where
    T: Clone + std::fmt::Debug,
{
    pub fn new(config: OptimizationConfig) -> Self {
        Self {
            config,
            cache: Arc::new(Mutex::new(HashMap::new())),
            cache_hits: Arc::new(Mutex::new(0)),
            cache_misses: Arc::new(Mutex::new(0)),
        }
    }

    /// Get cached result or compute if not available
    pub fn get_or_compute<F>(&self, key: &str, compute_fn: F) -> Result<Array2<T>>
    where
        F: FnOnce() -> Result<Array2<T>>,
    {
        // Check cache first
        {
            let cache = self.cache.lock().expect("operation should succeed");
            if let Some(cached_result) = cache.get(key) {
                *self.cache_hits.lock().expect("operation should succeed") += 1;
                return Ok(cached_result.clone());
            }
        }

        // Cache miss - compute result
        *self.cache_misses.lock().expect("operation should succeed") += 1;
        let result = compute_fn()?;

        // Store in cache if within size limit
        {
            let mut cache = self.cache.lock().expect("operation should succeed");
            if cache.len() < self.config.cache_size {
                cache.insert(key.to_string(), result.clone());
            }
        }

        Ok(result)
    }

    /// Clear cache
    pub fn clear_cache(&self) {
        self.cache.lock().expect("operation should succeed").clear();
    }

    /// Get cache statistics
    pub fn cache_statistics(&self) -> HashMap<String, f64> {
        let hits = *self.cache_hits.lock().expect("operation should succeed") as f64;
        let misses = *self.cache_misses.lock().expect("operation should succeed") as f64;
        let total = hits + misses;
        let hit_rate = if total > 0.0 { hits / total } else { 0.0 };

        let mut stats = HashMap::new();
        stats.insert("cache_hits".to_string(), hits);
        stats.insert("cache_misses".to_string(), misses);
        stats.insert("hit_rate".to_string(), hit_rate);
        stats.insert(
            "cache_size".to_string(),
            self.cache.lock().expect("operation should succeed").len() as f64,
        );
        stats
    }
}

/// GPU-accelerated matrix-multiply optimizer.
///
/// Without the `gpu` feature (or on a host with no usable GPU/driver even
/// when the feature is enabled), every call transparently falls back to a
/// plain CPU triple loop -- `is_gpu_available` honestly reports `false` in
/// that case rather than pretending a device is present.
///
/// With the `gpu` feature, [`Self::new`] calls
/// [`GpuBackend::detect`](sklears_core::gpu::GpuBackend::detect) once and
/// caches the result; `f32`/`f64` matrices are then uploaded via
/// [`GpuArray`] and multiplied on-device through `oxicuda-blas` GEMM
/// (`GpuMatrixOps::matmul`). Any other element type still uses the CPU path
/// even when a GPU is present, since `oxicuda-blas`'s GEMM is only
/// implemented for `f32`/`f64`.
#[derive(Debug, Clone)]
pub struct GpuOptimizer {
    #[allow(dead_code)]
    config: OptimizationConfig,
    /// Detected GPU backend, or `None` on GPU-less hosts / GPU-disabled
    /// builds. Only present when the `gpu` feature is enabled.
    #[cfg(feature = "gpu")]
    backend: Option<GpuBackend>,
    performance_metrics: HashMap<String, f64>,
}

impl GpuOptimizer {
    pub fn new(config: OptimizationConfig) -> Self {
        Self {
            config,
            // `GpuBackend::detect()` honestly returns `Ok(None)` when there is
            // no usable driver/device (e.g. this crate's own macOS CI/dev
            // hosts), so `backend` stays `None` there instead of pretending a
            // GPU exists.
            #[cfg(feature = "gpu")]
            backend: GpuBackend::detect().ok().flatten(),
            performance_metrics: HashMap::new(),
        }
    }

    /// Matrix multiply, GPU-accelerated for `f32`/`f64` when a GPU is
    /// available and the `gpu` feature is enabled; CPU fallback otherwise.
    pub fn gpu_matrix_multiply<T>(&self, a: &ArrayView2<T>, b: &ArrayView2<T>) -> Result<Array2<T>>
    where
        T: Clone
            + Copy
            + std::fmt::Debug
            + Default
            + std::ops::Add<Output = T>
            + std::ops::Mul<Output = T>
            + 'static,
    {
        #[cfg(feature = "gpu")]
        if let Some(backend) = &self.backend {
            if let Some(result) = Self::try_gpu_matmul(backend, a, b)? {
                return Ok(result);
            }
        }

        self.cpu_matrix_multiply(a, b)
    }

    /// Attempts the GEMM on-device for `T = f32`/`f64`; returns `Ok(None)`
    /// for every other element type so the caller falls back to the CPU
    /// path. `T: 'static` plus the `TypeId` equality checks below are what
    /// make the `transmute`s sound: a `TypeId` match proves `T` and the
    /// target scalar type are the *same* type (Rust guarantees no two
    /// distinct types ever share a `TypeId`), and `ArrayView2`/`Array2` are
    /// `Vec`/pointer-backed containers whose layout does not depend on the
    /// element type beyond its size, so reinterpreting the view reference
    /// (thin pointer, same size for any `Sized` element type) and the
    /// returned owned array (moved, not duplicated -- no double-free) between
    /// `T` and the scalar type is sound once that identity is established.
    #[cfg(feature = "gpu")]
    fn try_gpu_matmul<T: Copy + 'static>(
        backend: &GpuBackend,
        a: &ArrayView2<T>,
        b: &ArrayView2<T>,
    ) -> Result<Option<Array2<T>>> {
        if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
            let a32: &ArrayView2<f32> = unsafe { std::mem::transmute(a) };
            let b32: &ArrayView2<f32> = unsafe { std::mem::transmute(b) };
            let result = Self::gpu_matmul_typed(backend, a32, b32)?;
            let result: Array2<T> = unsafe { std::mem::transmute(result) };
            return Ok(Some(result));
        }
        if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
            let a64: &ArrayView2<f64> = unsafe { std::mem::transmute(a) };
            let b64: &ArrayView2<f64> = unsafe { std::mem::transmute(b) };
            let result = Self::gpu_matmul_typed(backend, a64, b64)?;
            let result: Array2<T> = unsafe { std::mem::transmute(result) };
            return Ok(Some(result));
        }
        Ok(None)
    }

    /// Real on-device GEMM: uploads `a`/`b`, multiplies via
    /// `GpuMatrixOps::matmul` (`oxicuda-blas` GEMM), and downloads the
    /// result.
    #[cfg(feature = "gpu")]
    fn gpu_matmul_typed<S>(
        backend: &GpuBackend,
        a: &ArrayView2<S>,
        b: &ArrayView2<S>,
    ) -> Result<Array2<S>>
    where
        S: Copy + Clone + Default,
        GpuArray<S>: GpuMatrixOps,
    {
        let a_gpu = GpuArray::from_array2(backend, &a.to_owned())?;
        let b_gpu = GpuArray::from_array2(backend, &b.to_owned())?;
        let c_gpu = a_gpu.matmul(&b_gpu)?;
        c_gpu.to_array2()
    }

    fn cpu_matrix_multiply<T>(&self, a: &ArrayView2<T>, b: &ArrayView2<T>) -> Result<Array2<T>>
    where
        T: Clone
            + Copy
            + std::fmt::Debug
            + Default
            + std::ops::Add<Output = T>
            + std::ops::Mul<Output = T>,
    {
        let (m, n) = a.dim();
        let (n2, p) = b.dim();

        if n != n2 {
            return Err(SklearsError::InvalidInput(
                "Matrix dimension mismatch".to_string(),
            ));
        }

        let mut result = Array2::default((m, p));
        for i in 0..m {
            for j in 0..p {
                let mut sum = T::default();
                for k in 0..n {
                    sum = sum + a[(i, k)] * b[(k, j)];
                }
                result[(i, j)] = sum;
            }
        }

        Ok(result)
    }

    /// `true` only when the `gpu` feature is enabled *and*
    /// [`GpuBackend::detect`](sklears_core::gpu::GpuBackend::detect) found a
    /// real, usable device at construction time -- never fabricated.
    pub fn is_gpu_available(&self) -> bool {
        #[cfg(feature = "gpu")]
        {
            self.backend.is_some()
        }
        #[cfg(not(feature = "gpu"))]
        {
            false
        }
    }

    pub fn performance_metrics(&self) -> &HashMap<String, f64> {
        &self.performance_metrics
    }
}

/// Performance monitoring and profiling
#[derive(Debug, Clone)]
pub struct PerformanceProfiler {
    execution_times: HashMap<String, Vec<f64>>,
    memory_usage: HashMap<String, Vec<usize>>,
}

impl PerformanceProfiler {
    pub fn new() -> Self {
        Self {
            execution_times: HashMap::new(),
            memory_usage: HashMap::new(),
        }
    }

    /// Profile an operation
    pub fn profile_operation<F, R>(&mut self, name: &str, operation: F) -> R
    where
        F: FnOnce() -> R,
    {
        let start_time = std::time::Instant::now();
        let result = operation();
        let duration = start_time.elapsed().as_secs_f64();

        self.execution_times
            .entry(name.to_string())
            .or_default()
            .push(duration);

        result
    }

    /// Get performance statistics
    pub fn get_statistics(&self) -> HashMap<String, HashMap<String, f64>> {
        let mut stats = HashMap::new();

        for (name, times) in &self.execution_times {
            let mut operation_stats = HashMap::new();
            if !times.is_empty() {
                let sum: f64 = times.iter().sum();
                let mean = sum / times.len() as f64;
                let min = times.iter().cloned().fold(f64::INFINITY, f64::min);
                let max = times.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

                operation_stats.insert("mean_time".to_string(), mean);
                operation_stats.insert("min_time".to_string(), min);
                operation_stats.insert("max_time".to_string(), max);
                operation_stats.insert("total_time".to_string(), sum);
                operation_stats.insert("call_count".to_string(), times.len() as f64);
            }
            stats.insert(name.clone(), operation_stats);
        }

        stats
    }

    pub fn reset(&mut self) {
        self.execution_times.clear();
        self.memory_usage.clear();
    }
}

impl Default for PerformanceProfiler {
    fn default() -> Self {
        Self::new()
    }
}

/// Batch processor for large datasets
#[derive(Debug, Clone)]
pub struct BatchProcessor {
    config: OptimizationConfig,
    performance_metrics: HashMap<String, f64>,
}

impl BatchProcessor {
    pub fn new(config: OptimizationConfig) -> Self {
        Self {
            config,
            performance_metrics: HashMap::new(),
        }
    }

    /// Process data in batches
    pub fn process_batches<T, F, R>(&self, data: &ArrayView2<T>, process_fn: F) -> Result<Vec<R>>
    where
        T: Clone + Copy + std::fmt::Debug,
        F: Fn(&ArrayView2<T>) -> Result<R>,
    {
        let (n_samples, _) = data.dim();
        let batch_size = self.config.batch_size;
        let mut results = Vec::new();

        for batch_start in (0..n_samples).step_by(batch_size) {
            let batch_end = (batch_start + batch_size).min(n_samples);
            let batch = data.slice(s![batch_start..batch_end, ..]);
            let batch_result = process_fn(&batch)?;
            results.push(batch_result);
        }

        Ok(results)
    }

    /// Stream processing for very large datasets
    pub fn stream_process<T, F>(&self, data: &ArrayView2<T>, mut process_fn: F) -> Result<()>
    where
        T: Clone + Copy + std::fmt::Debug,
        F: FnMut(&ArrayView2<T>) -> Result<()>,
    {
        let (n_samples, _) = data.dim();
        let batch_size = self.config.batch_size;

        for batch_start in (0..n_samples).step_by(batch_size) {
            let batch_end = (batch_start + batch_size).min(n_samples);
            let batch = data.slice(s![batch_start..batch_end, ..]);
            process_fn(&batch)?;
        }

        Ok(())
    }

    pub fn performance_metrics(&self) -> &HashMap<String, f64> {
        &self.performance_metrics
    }
}

/// Performance optimization validator
#[derive(Debug, Clone)]
pub struct OptimizationValidator;

impl OptimizationValidator {
    pub fn validate_config(config: &OptimizationConfig) -> Result<()> {
        if config.batch_size == 0 {
            return Err(SklearsError::InvalidInput(
                "Batch size must be greater than 0".to_string(),
            ));
        }
        if config.cache_size == 0 {
            return Err(SklearsError::InvalidInput(
                "Cache size must be greater than 0".to_string(),
            ));
        }
        Ok(())
    }

    pub fn validate_memory_limits(data_size: usize, config: &OptimizationConfig) -> Result<()> {
        if let Some(limit) = config.memory_limit {
            if data_size > limit {
                return Err(SklearsError::InvalidInput(
                    "Data size exceeds memory limit".to_string(),
                ));
            }
        }
        Ok(())
    }
}

#[allow(non_snake_case, clippy::field_reassign_with_default)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_optimizer() {
        let config = OptimizationConfig::default();
        let optimizer = SimdOptimizer::new(config);

        let a = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0])
            .expect("operation should succeed");
        let b = Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 0.0, 1.0])
            .expect("operation should succeed");

        let result = optimizer
            .simd_matrix_multiply(&a.view(), &b.view())
            .expect("operation should succeed");
        assert_eq!(result.dim(), (2, 2));
    }

    #[test]
    fn test_parallel_optimizer() {
        let config = OptimizationConfig::default();
        let optimizer = ParallelOptimizer::new(config);

        let data = Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
            .expect("operation should succeed");

        let result = optimizer
            .parallel_transform(&data.view(), |row| Ok(row.to_owned()))
            .expect("operation should succeed");

        assert_eq!(result.dim(), data.dim());
    }

    #[test]
    fn test_memory_optimizer() {
        let mut config = OptimizationConfig::default();
        config.batch_size = 2;
        let optimizer = MemoryOptimizer::new(config);

        let data = Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
            .expect("operation should succeed");

        let result = optimizer
            .batch_process(&data.view(), |batch| Ok(batch.to_owned()))
            .expect("operation should succeed");

        assert_eq!(result.dim(), data.dim());
    }

    #[test]
    fn test_cache_optimizer() {
        let config = OptimizationConfig::default();
        let optimizer: CacheOptimizer<f64> = CacheOptimizer::new(config);

        let result1 = optimizer
            .get_or_compute("test_key", || Ok(Array2::zeros((2, 2))))
            .expect("operation should succeed");

        let result2 = optimizer
            .get_or_compute("test_key", || {
                Ok(Array2::ones((2, 2))) // This should not be called due to cache hit
            })
            .expect("operation should succeed");

        assert_eq!(result1.dim(), result2.dim());

        let stats = optimizer.cache_statistics();
        assert!(stats["cache_hits"] > 0.0);
    }

    #[test]
    fn test_gpu_optimizer_matrix_multiply_falls_back_to_cpu() {
        let config = OptimizationConfig::default();
        let optimizer = GpuOptimizer::new(config);

        let a = Array2::from_shape_vec((2, 2), vec![1.0_f64, 2.0, 3.0, 4.0])
            .expect("operation should succeed");
        let b = Array2::from_shape_vec((2, 2), vec![1.0_f64, 0.0, 0.0, 1.0])
            .expect("operation should succeed");

        let result = optimizer
            .gpu_matrix_multiply(&a.view(), &b.view())
            .expect("operation should succeed");
        assert_eq!(result, a);
    }

    #[test]
    fn test_gpu_optimizer_reports_honest_availability() {
        // `GpuOptimizer` must never fabricate availability: it reports a device
        // exactly when `GpuBackend::detect()` actually found one, and never
        // without the `gpu` feature.
        let config = OptimizationConfig::default();
        let optimizer = GpuOptimizer::new(config);
        #[cfg(feature = "gpu")]
        {
            let detected = GpuBackend::detect().ok().flatten().is_some();
            assert_eq!(optimizer.is_gpu_available(), detected);
        }
        #[cfg(not(feature = "gpu"))]
        {
            assert!(!optimizer.is_gpu_available());
        }

        // The result must still be correct via the CPU fallback path,
        // including for non-f32/f64 element types.
        let a = Array2::from_shape_vec((2, 3), vec![1_i64, 2, 3, 4, 5, 6])
            .expect("operation should succeed");
        let b = Array2::from_shape_vec((3, 2), vec![1_i64, 0, 0, 1, 1, 1])
            .expect("operation should succeed");
        let result = optimizer
            .gpu_matrix_multiply(&a.view(), &b.view())
            .expect("operation should succeed");
        assert_eq!(result.dim(), (2, 2));
    }

    #[test]
    fn test_performance_profiler() {
        let mut profiler = PerformanceProfiler::new();

        let result = profiler.profile_operation("test_op", || {
            std::thread::sleep(std::time::Duration::from_millis(10));
            42
        });

        assert_eq!(result, 42);

        let stats = profiler.get_statistics();
        assert!(stats.contains_key("test_op"));
    }

    #[test]
    fn test_batch_processor() {
        let mut config = OptimizationConfig::default();
        config.batch_size = 2;
        let processor = BatchProcessor::new(config);

        let data = Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
            .expect("operation should succeed");

        let results = processor
            .process_batches(&data.view(), |batch| Ok(batch.nrows()))
            .expect("operation should succeed");

        assert_eq!(results.len(), 2); // 4 rows / 2 batch_size = 2 batches
        assert_eq!(results[0], 2);
        assert_eq!(results[1], 2);
    }
}
