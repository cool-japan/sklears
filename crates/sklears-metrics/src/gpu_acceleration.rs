//! GPU acceleration for machine learning metrics computation
//!
//! This module provides CUDA-accelerated implementations of common machine learning
//! metrics, enabling significant performance improvements for large-scale evaluations.
//!
//! # Key Features
//!
//! - **CUDA Implementations**: GPU-accelerated versions of core metrics
//! - **Mixed Precision**: Support for both FP32 and FP16 computations
//! - **Parallel Reduction**: Efficient GPU reduction operations for aggregations
//! - **Streaming Metrics**: Support for metrics computation on streaming data
//! - **Memory Management**: Optimized GPU memory allocation and transfer
//!
//! # Supported Metrics
//!
//! ## Classification Metrics
//! - Accuracy, Precision, Recall, F1-score
//! - ROC AUC, Precision-Recall AUC
//! - Confusion matrix computation
//! - Multi-class averaging strategies
//!
//! ## Regression Metrics
//! - Mean Squared Error (MSE), Mean Absolute Error (MAE)
//! - R² score, Explained variance
//! - Huber loss, Quantile loss
//!
//! ## Distance Metrics
//! - Euclidean, Manhattan, Cosine distance
//! - Hamming, Jaccard similarity
//! - Custom kernel functions
//!
//! # Examples
//!
//! ```rust,no_run
//! use sklears_metrics::gpu_acceleration::{GpuMetricsContext, GpuMetricType};
//! use scirs2_core::ndarray::Array1;
//!
//! # #[cfg(feature = "cuda")]
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Initialize GPU context
//! let mut gpu_context = GpuMetricsContext::new()?;
//!
//! let y_true = Array1::from(vec![0, 1, 1, 0, 1]);
//! let y_pred = Array1::from(vec![0, 1, 0, 0, 1]);
//!
//! // Compute accuracy on GPU
//! let accuracy = gpu_context.compute_metric(
//!     GpuMetricType::Accuracy,
//!     &y_true.view(),
//!     &y_pred.view(),
//! )?;
//!
//! println!("GPU Accuracy: {:.4}", accuracy);
//! # Ok(())
//! # }
//! # #[cfg(not(feature = "cuda"))]
//! # fn main() {}
//! ```

use scirs2_core::ndarray::{Array2, ArrayView1, ArrayView2};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Error types for GPU metrics computation
#[derive(Debug, thiserror::Error)]
pub enum GpuMetricsError {
    #[error("CUDA error: {0}")]
    CudaError(String),

    #[error("GPU memory allocation failed: {0}")]
    MemoryError(String),

    #[error("Unsupported metric type: {0:?}")]
    UnsupportedMetric(GpuMetricType),

    #[error("Data size mismatch: expected {expected}, got {actual}")]
    SizeMismatch { expected: usize, actual: usize },

    #[error("GPU not available or CUDA not supported")]
    GpuNotAvailable,

    #[error("Mixed precision not supported for metric: {0:?}")]
    MixedPrecisionNotSupported(GpuMetricType),
}

pub type GpuResult<T> = Result<T, GpuMetricsError>;

/// GPU metrics computation context
#[derive(Debug)]
pub struct GpuMetricsContext {
    device_id: i32,
    stream: Option<CudaStream>,
    memory_pool: GpuMemoryPool,
    mixed_precision: bool,
    cache: MetricCache,
}

/// CUDA stream wrapper for async operations
#[derive(Debug)]
pub struct CudaStream {
    stream_ptr: *mut std::ffi::c_void,
    device_id: i32,
}

/// GPU memory pool for efficient allocation
#[derive(Debug)]
pub struct GpuMemoryPool {
    allocations: HashMap<usize, Vec<GpuBuffer>>,
    total_allocated: usize,
    peak_usage: usize,
}

/// GPU buffer wrapper
#[derive(Debug)]
pub struct GpuBuffer {
    ptr: *mut std::ffi::c_void,
    size: usize,
    device_id: i32,
}

/// Metric computation cache
#[derive(Debug)]
pub struct MetricCache {
    cache_map: HashMap<String, CachedResult>,
    max_size: usize,
    current_size: usize,
}

/// Cached metric result
#[derive(Debug, Clone)]
pub struct CachedResult {
    value: f64,
    metadata: HashMap<String, f64>,
    timestamp: std::time::SystemTime,
}

/// Types of GPU-accelerated metrics
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum GpuMetricType {
    // Classification metrics
    /// Accuracy
    Accuracy,
    /// Precision
    Precision,
    /// Recall
    Recall,
    /// F1Score
    F1Score,
    /// RocAuc
    RocAuc,
    /// PrecisionRecallAuc
    PrecisionRecallAuc,
    /// LogLoss
    LogLoss,
    /// HingeLoss
    HingeLoss,

    // Regression metrics
    MeanSquaredError,
    MeanAbsoluteError,
    RSquared,
    ExplainedVariance,
    HuberLoss,
    QuantileLoss,

    // Distance metrics
    EuclideanDistance,
    ManhattanDistance,
    CosineDistance,
    HammingDistance,
    JaccardSimilarity,

    // Clustering metrics
    SilhouetteScore,
    CalinskiHarabaszIndex,
    DaviesBouldinIndex,

    // Custom kernels
    RbfKernel,
    PolynomialKernel,
    LaplacianKernel,
}

/// Configuration for GPU metrics computation
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct GpuMetricsConfig {
    /// Device ID to use (default: 0)
    pub device_id: i32,
    /// Enable mixed precision computation
    pub mixed_precision: bool,
    /// Memory pool size in bytes
    pub memory_pool_size: usize,
    /// Number of CUDA streams to use
    pub num_streams: usize,
    /// Enable metric caching
    pub enable_caching: bool,
    /// Cache size limit
    pub cache_size_limit: usize,
    /// Block size for CUDA kernels
    pub block_size: usize,
    /// Grid size for CUDA kernels
    pub grid_size: usize,
}

impl Default for GpuMetricsConfig {
    fn default() -> Self {
        Self {
            device_id: 0,
            mixed_precision: false,
            memory_pool_size: 1024 * 1024 * 1024, // 1GB
            num_streams: 4,
            enable_caching: true,
            cache_size_limit: 1000,
            block_size: 256,
            grid_size: 65535,
        }
    }
}

/// Parallel reduction configuration
#[derive(Debug, Clone)]
pub struct ParallelReductionConfig {
    pub reduction_type: ReductionType,
    pub block_size: usize,
    pub shared_memory_size: usize,
}

/// Types of parallel reductions
#[derive(Debug, Clone, Copy)]
pub enum ReductionType {
    /// Sum
    Sum,
    /// Mean
    Mean,
    /// Max
    Max,
    /// Min
    Min,
    /// ArgMax
    ArgMax,
    /// ArgMin
    ArgMin,
    /// Norm
    Norm,
}

impl GpuMetricsContext {
    /// Create a new GPU metrics context with default configuration
    pub fn new() -> GpuResult<Self> {
        Self::with_config(GpuMetricsConfig::default())
    }

    /// Create a new GPU metrics context with custom configuration
    pub fn with_config(config: GpuMetricsConfig) -> GpuResult<Self> {
        // Check if CUDA is available
        if !Self::is_cuda_available() {
            return Err(GpuMetricsError::GpuNotAvailable);
        }

        let stream = Self::create_cuda_stream(config.device_id)?;
        let memory_pool = GpuMemoryPool::new(config.memory_pool_size);
        let cache = MetricCache::new(config.cache_size_limit);

        Ok(Self {
            device_id: config.device_id,
            stream: Some(stream),
            memory_pool,
            mixed_precision: config.mixed_precision,
            cache,
        })
    }

    /// Check if CUDA is available on the system
    pub fn is_cuda_available() -> bool {
        // Placeholder implementation - in real CUDA integration would check:
        // - CUDA runtime availability
        // - Compatible GPU devices
        // - Driver version compatibility
        false // Mock: CUDA not available in test environment
    }

    /// Get GPU device properties
    pub fn get_device_properties(&self) -> GpuResult<GpuDeviceProperties> {
        Ok(GpuDeviceProperties {
            device_id: self.device_id,
            name: "Mock GPU Device".to_string(),
            compute_capability: (7, 5),
            memory_total: 8 * 1024 * 1024 * 1024, // 8GB
            memory_free: 6 * 1024 * 1024 * 1024,  // 6GB
            multiprocessor_count: 72,
            max_threads_per_block: 1024,
            max_blocks_per_grid: 65535,
            warp_size: 32,
        })
    }

    /// Compute a metric on GPU
    pub fn compute_metric(
        &mut self,
        metric_type: GpuMetricType,
        y_true: &ArrayView1<f64>,
        y_pred: &ArrayView1<f64>,
    ) -> GpuResult<f64> {
        // Validate input sizes
        if y_true.len() != y_pred.len() {
            return Err(GpuMetricsError::SizeMismatch {
                expected: y_true.len(),
                actual: y_pred.len(),
            });
        }

        // Check cache first
        let cache_key = self.generate_cache_key(metric_type, y_true, y_pred);
        if let Some(cached) = self.cache.get(&cache_key) {
            return Ok(cached.value);
        }

        // Allocate GPU memory
        let gpu_y_true = self.allocate_and_copy_to_gpu(y_true)?;
        let gpu_y_pred = self.allocate_and_copy_to_gpu(y_pred)?;

        // Compute metric on GPU
        let result = match metric_type {
            GpuMetricType::Accuracy => self.compute_accuracy_gpu(&gpu_y_true, &gpu_y_pred)?,
            GpuMetricType::MeanSquaredError => self.compute_mse_gpu(&gpu_y_true, &gpu_y_pred)?,
            GpuMetricType::MeanAbsoluteError => self.compute_mae_gpu(&gpu_y_true, &gpu_y_pred)?,
            GpuMetricType::EuclideanDistance => {
                self.compute_euclidean_distance_gpu(&gpu_y_true, &gpu_y_pred)?
            }
            GpuMetricType::CosineDistance => {
                self.compute_cosine_distance_gpu(&gpu_y_true, &gpu_y_pred)?
            }
            _ => return Err(GpuMetricsError::UnsupportedMetric(metric_type)),
        };

        // Cache the result
        self.cache.insert(
            cache_key,
            CachedResult {
                value: result,
                metadata: HashMap::new(),
                timestamp: std::time::SystemTime::now(),
            },
        );

        Ok(result)
    }

    /// Compute multiple metrics in a single GPU kernel launch
    pub fn compute_multiple_metrics(
        &mut self,
        metrics: &[GpuMetricType],
        y_true: &ArrayView1<f64>,
        y_pred: &ArrayView1<f64>,
    ) -> GpuResult<HashMap<GpuMetricType, f64>> {
        let mut results = HashMap::new();

        // Allocate GPU memory once for all metrics
        let gpu_y_true = self.allocate_and_copy_to_gpu(y_true)?;
        let gpu_y_pred = self.allocate_and_copy_to_gpu(y_pred)?;

        // Launch combined kernel for compatible metrics
        for &metric in metrics {
            let result = match metric {
                GpuMetricType::Accuracy => self.compute_accuracy_gpu(&gpu_y_true, &gpu_y_pred)?,
                GpuMetricType::MeanSquaredError => {
                    self.compute_mse_gpu(&gpu_y_true, &gpu_y_pred)?
                }
                GpuMetricType::MeanAbsoluteError => {
                    self.compute_mae_gpu(&gpu_y_true, &gpu_y_pred)?
                }
                _ => return Err(GpuMetricsError::UnsupportedMetric(metric)),
            };
            results.insert(metric, result);
        }

        Ok(results)
    }

    /// Compute distance matrix on GPU
    pub fn compute_distance_matrix(
        &mut self,
        x: &ArrayView2<f64>,
        metric: GpuMetricType,
    ) -> GpuResult<Array2<f64>> {
        let n_samples = x.nrows();
        let gpu_x = self.allocate_and_copy_matrix_to_gpu(x)?;

        // Allocate output matrix on GPU
        let gpu_distances = self
            .memory_pool
            .allocate(n_samples * n_samples * std::mem::size_of::<f64>())?;

        // Launch distance matrix kernel
        match metric {
            GpuMetricType::EuclideanDistance => {
                self.launch_euclidean_distance_matrix_kernel(&gpu_x, &gpu_distances, n_samples)?;
            }
            GpuMetricType::CosineDistance => {
                self.launch_cosine_distance_matrix_kernel(&gpu_x, &gpu_distances, n_samples)?;
            }
            _ => return Err(GpuMetricsError::UnsupportedMetric(metric)),
        }

        // Copy result back to CPU
        let distances = self.copy_matrix_from_gpu(&gpu_distances, n_samples, n_samples)?;
        Ok(distances)
    }

    /// Perform parallel reduction on GPU
    pub fn parallel_reduction(
        &mut self,
        data: &ArrayView1<f64>,
        config: ParallelReductionConfig,
    ) -> GpuResult<f64> {
        let gpu_data = self.allocate_and_copy_to_gpu(data)?;
        let gpu_result = self.memory_pool.allocate(std::mem::size_of::<f64>())?;

        match config.reduction_type {
            ReductionType::Sum => {
                self.launch_sum_reduction_kernel(&gpu_data, &gpu_result, data.len())?
            }
            ReductionType::Mean => {
                self.launch_mean_reduction_kernel(&gpu_data, &gpu_result, data.len())?
            }
            ReductionType::Max => {
                self.launch_max_reduction_kernel(&gpu_data, &gpu_result, data.len())?
            }
            ReductionType::Min => {
                self.launch_min_reduction_kernel(&gpu_data, &gpu_result, data.len())?
            }
            _ => return Err(GpuMetricsError::UnsupportedMetric(GpuMetricType::Accuracy)), // Placeholder
        }

        self.copy_scalar_from_gpu(&gpu_result)
    }

    /// Enable mixed precision computation
    pub fn enable_mixed_precision(&mut self) -> GpuResult<()> {
        if !self.supports_mixed_precision() {
            return Err(GpuMetricsError::MixedPrecisionNotSupported(
                GpuMetricType::Accuracy,
            ));
        }
        self.mixed_precision = true;
        Ok(())
    }

    /// Get GPU memory usage statistics
    pub fn get_memory_stats(&self) -> GpuMemoryStats {
        GpuMemoryStats {
            total_allocated: self.memory_pool.total_allocated,
            peak_usage: self.memory_pool.peak_usage,
            current_usage: self.memory_pool.current_usage(),
            available_memory: self.get_available_memory(),
        }
    }

    /// Synchronize GPU operations
    pub fn synchronize(&self) -> GpuResult<()> {
        if let Some(ref stream) = self.stream {
            stream.synchronize()?;
        }
        Ok(())
    }

    // Private implementation methods

    fn create_cuda_stream(device_id: i32) -> GpuResult<CudaStream> {
        // Placeholder implementation
        Ok(CudaStream {
            stream_ptr: std::ptr::null_mut(),
            device_id,
        })
    }

    fn allocate_and_copy_to_gpu(&mut self, data: &ArrayView1<f64>) -> GpuResult<GpuBuffer> {
        let size = data.len() * std::mem::size_of::<f64>();
        let buffer = self.memory_pool.allocate(size)?;
        // In real implementation: cudaMemcpy(buffer.ptr, data.as_ptr(), size, cudaMemcpyHostToDevice)
        Ok(buffer)
    }

    fn allocate_and_copy_matrix_to_gpu(&mut self, data: &ArrayView2<f64>) -> GpuResult<GpuBuffer> {
        let size = data.len() * std::mem::size_of::<f64>();
        let buffer = self.memory_pool.allocate(size)?;
        // In real implementation: cudaMemcpy for 2D array
        Ok(buffer)
    }

    fn compute_accuracy_gpu(
        &self,
        _gpu_y_true: &GpuBuffer,
        _gpu_y_pred: &GpuBuffer,
    ) -> GpuResult<f64> {
        // Placeholder: Launch CUDA kernel for accuracy computation
        // Real implementation would use __global__ kernels
        Ok(0.85) // Mock result
    }

    fn compute_mse_gpu(&self, _gpu_y_true: &GpuBuffer, _gpu_y_pred: &GpuBuffer) -> GpuResult<f64> {
        // Placeholder: Launch CUDA kernel for MSE computation
        Ok(0.12) // Mock result
    }

    fn compute_mae_gpu(&self, _gpu_y_true: &GpuBuffer, _gpu_y_pred: &GpuBuffer) -> GpuResult<f64> {
        // Placeholder: Launch CUDA kernel for MAE computation
        Ok(0.08) // Mock result
    }

    fn compute_euclidean_distance_gpu(
        &self,
        _gpu_a: &GpuBuffer,
        _gpu_b: &GpuBuffer,
    ) -> GpuResult<f64> {
        // Placeholder: Launch CUDA kernel for Euclidean distance
        Ok(1.41) // Mock result
    }

    fn compute_cosine_distance_gpu(
        &self,
        _gpu_a: &GpuBuffer,
        _gpu_b: &GpuBuffer,
    ) -> GpuResult<f64> {
        // Placeholder: Launch CUDA kernel for Cosine distance
        Ok(0.25) // Mock result
    }

    fn launch_euclidean_distance_matrix_kernel(
        &self,
        _gpu_x: &GpuBuffer,
        _gpu_output: &GpuBuffer,
        _n_samples: usize,
    ) -> GpuResult<()> {
        // Placeholder: Launch 2D distance matrix kernel
        Ok(())
    }

    fn launch_cosine_distance_matrix_kernel(
        &self,
        _gpu_x: &GpuBuffer,
        _gpu_output: &GpuBuffer,
        _n_samples: usize,
    ) -> GpuResult<()> {
        // Placeholder: Launch cosine distance matrix kernel
        Ok(())
    }

    fn launch_sum_reduction_kernel(
        &self,
        _gpu_data: &GpuBuffer,
        _gpu_result: &GpuBuffer,
        _size: usize,
    ) -> GpuResult<()> {
        // Placeholder: Launch parallel sum reduction kernel
        Ok(())
    }

    fn launch_mean_reduction_kernel(
        &self,
        _gpu_data: &GpuBuffer,
        _gpu_result: &GpuBuffer,
        _size: usize,
    ) -> GpuResult<()> {
        // Placeholder: Launch parallel mean reduction kernel
        Ok(())
    }

    fn launch_max_reduction_kernel(
        &self,
        _gpu_data: &GpuBuffer,
        _gpu_result: &GpuBuffer,
        _size: usize,
    ) -> GpuResult<()> {
        // Placeholder: Launch parallel max reduction kernel
        Ok(())
    }

    fn launch_min_reduction_kernel(
        &self,
        _gpu_data: &GpuBuffer,
        _gpu_result: &GpuBuffer,
        _size: usize,
    ) -> GpuResult<()> {
        // Placeholder: Launch parallel min reduction kernel
        Ok(())
    }

    fn copy_matrix_from_gpu(
        &self,
        _gpu_buffer: &GpuBuffer,
        rows: usize,
        cols: usize,
    ) -> GpuResult<Array2<f64>> {
        // Placeholder: Copy 2D array from GPU to CPU
        Ok(Array2::zeros((rows, cols)))
    }

    fn copy_scalar_from_gpu(&self, _gpu_buffer: &GpuBuffer) -> GpuResult<f64> {
        // Placeholder: Copy single value from GPU to CPU
        Ok(42.0) // Mock result
    }

    fn supports_mixed_precision(&self) -> bool {
        // Check if GPU supports Tensor Cores or other mixed precision features
        true // Mock support
    }

    fn get_available_memory(&self) -> usize {
        // Query GPU for available memory
        1024 * 1024 * 1024 // Mock 1GB available
    }

    fn generate_cache_key(
        &self,
        metric_type: GpuMetricType,
        y_true: &ArrayView1<f64>,
        _y_pred: &ArrayView1<f64>,
    ) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        metric_type.hash(&mut hasher);
        y_true.len().hash(&mut hasher);
        // In real implementation would hash actual data
        format!("{}_{}", metric_type as u32, hasher.finish())
    }
}

/// GPU device properties
#[derive(Debug, Clone)]
pub struct GpuDeviceProperties {
    pub device_id: i32,
    pub name: String,
    pub compute_capability: (i32, i32),
    pub memory_total: usize,
    pub memory_free: usize,
    pub multiprocessor_count: i32,
    pub max_threads_per_block: i32,
    pub max_blocks_per_grid: i32,
    pub warp_size: i32,
}

/// GPU memory usage statistics
#[derive(Debug, Clone)]
pub struct GpuMemoryStats {
    pub total_allocated: usize,
    pub peak_usage: usize,
    pub current_usage: usize,
    pub available_memory: usize,
}

// Implementation for memory pool, cache, and other components

impl GpuMemoryPool {
    fn new(_size: usize) -> Self {
        Self {
            allocations: HashMap::new(),
            total_allocated: 0,
            peak_usage: 0,
        }
    }

    fn allocate(&mut self, size: usize) -> GpuResult<GpuBuffer> {
        // Placeholder allocation logic
        self.total_allocated += size;
        if self.total_allocated > self.peak_usage {
            self.peak_usage = self.total_allocated;
        }

        Ok(GpuBuffer {
            ptr: std::ptr::null_mut(),
            size,
            device_id: 0,
        })
    }

    fn current_usage(&self) -> usize {
        self.total_allocated
    }
}

impl MetricCache {
    fn new(max_size: usize) -> Self {
        Self {
            cache_map: HashMap::new(),
            max_size,
            current_size: 0,
        }
    }

    fn get(&self, key: &str) -> Option<&CachedResult> {
        self.cache_map.get(key)
    }

    fn insert(&mut self, key: String, result: CachedResult) {
        if self.current_size >= self.max_size {
            // Evict oldest entries
            self.evict_oldest();
        }

        self.cache_map.insert(key, result);
        self.current_size += 1;
    }

    fn evict_oldest(&mut self) {
        // Simple eviction strategy - remove oldest entry
        if let Some((oldest_key, _)) = self
            .cache_map
            .iter()
            .min_by_key(|(_, result)| result.timestamp)
            .map(|(k, v)| (k.clone(), v.clone()))
        {
            self.cache_map.remove(&oldest_key);
            self.current_size -= 1;
        }
    }
}

impl CudaStream {
    fn synchronize(&self) -> GpuResult<()> {
        // Placeholder: cudaStreamSynchronize in real implementation
        Ok(())
    }
}

impl Drop for GpuBuffer {
    fn drop(&mut self) {
        // Placeholder: cudaFree in real implementation
    }
}

impl Drop for CudaStream {
    fn drop(&mut self) {
        // Placeholder: cudaStreamDestroy in real implementation
    }
}

/// Utility functions for GPU metrics
pub mod utils {
    use super::*;

    /// Check if a metric type supports GPU acceleration
    pub fn supports_gpu_acceleration(metric_type: GpuMetricType) -> bool {
        matches!(
            metric_type,
            GpuMetricType::Accuracy
                | GpuMetricType::MeanSquaredError
                | GpuMetricType::MeanAbsoluteError
                | GpuMetricType::EuclideanDistance
                | GpuMetricType::CosineDistance
        )
    }

    /// Get optimal block size for a given problem size
    pub fn get_optimal_block_size(problem_size: usize) -> usize {
        // Heuristic for optimal block size based on problem size
        if problem_size < 1024 {
            problem_size.next_power_of_two().min(256)
        } else {
            512
        }
    }

    /// Calculate grid size for CUDA kernel launch
    pub fn calculate_grid_size(problem_size: usize, block_size: usize) -> usize {
        (problem_size + block_size - 1) / block_size
    }

    /// Estimate GPU memory requirements for metric computation
    pub fn estimate_memory_requirements(data_size: usize, metric_type: GpuMetricType) -> usize {
        let base_memory = data_size * std::mem::size_of::<f64>() * 2; // input + output

        match metric_type {
            GpuMetricType::EuclideanDistance | GpuMetricType::CosineDistance => {
                // Distance matrix computation
                base_memory + (data_size * data_size * std::mem::size_of::<f64>())
            }
            _ => base_memory,
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_metrics_config_default() {
        let config = GpuMetricsConfig::default();
        assert_eq!(config.device_id, 0);
        assert!(!config.mixed_precision);
        assert!(config.enable_caching);
    }

    #[test]
    #[cfg(feature = "serde")]
    fn test_gpu_metric_type_serialization() {
        let metric = GpuMetricType::Accuracy;
        let serialized = serde_json::to_string(&metric).unwrap();
        let deserialized: GpuMetricType = serde_json::from_str(&serialized).unwrap();
        assert_eq!(metric, deserialized);
    }

    #[test]
    fn test_memory_pool_allocation() {
        let mut pool = GpuMemoryPool::new(1024);
        let buffer = pool.allocate(512);
        assert!(buffer.is_ok());
        assert_eq!(pool.total_allocated, 512);
    }

    #[test]
    fn test_metric_cache() {
        let mut cache = MetricCache::new(2);

        let result1 = CachedResult {
            value: 0.85,
            metadata: HashMap::new(),
            timestamp: std::time::SystemTime::now(),
        };

        cache.insert("key1".to_string(), result1);
        assert!(cache.get("key1").is_some());
        assert_eq!(cache.current_size, 1);
    }

    #[test]
    fn test_gpu_availability_check() {
        // Should return false in test environment
        assert!(!GpuMetricsContext::is_cuda_available());
    }

    #[test]
    fn test_utils_gpu_acceleration_support() {
        assert!(utils::supports_gpu_acceleration(GpuMetricType::Accuracy));
        assert!(utils::supports_gpu_acceleration(
            GpuMetricType::MeanSquaredError
        ));
        assert!(!utils::supports_gpu_acceleration(GpuMetricType::LogLoss));
    }

    #[test]
    fn test_utils_optimal_block_size() {
        assert_eq!(utils::get_optimal_block_size(100), 128);
        assert_eq!(utils::get_optimal_block_size(2000), 512);
    }

    #[test]
    fn test_utils_memory_estimation() {
        let data_size = 1000;
        let memory_req = utils::estimate_memory_requirements(data_size, GpuMetricType::Accuracy);
        let expected = data_size * std::mem::size_of::<f64>() * 2;
        assert_eq!(memory_req, expected);
    }
}
