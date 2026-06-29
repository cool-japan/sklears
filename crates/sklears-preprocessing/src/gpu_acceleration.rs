//! GPU-Dispatched Preprocessing for Large-Scale Data
//!
//! This module provides preprocessing scalers that integrate with SciRS2's GPU
//! abstractions. The numerics are implemented as a correct CPU reference; the
//! module additionally performs *backend dispatch* through
//! [`scirs2_core::gpu::GpuBackend`], honestly reporting whether a real GPU
//! backend (CUDA, ROCm, Metal, WebGPU, OpenCL) is available on the current
//! system. No dedicated GPU compute kernels are shipped yet, so when a backend
//! is reported as available the dispatch path still evaluates the same verified
//! CPU numerics; when no backend is available it falls back to CPU truthfully.
//!
//! In other words: this is a CPU implementation with a GPU dispatch layer, not a
//! set of hand-written CUDA/Metal kernels. Device detection never fabricates a
//! "simulated" GPU — availability is delegated to SciRS2's real feature-gated
//! and runtime checks.
//!
//! # Features
//!
//! - Standard scaling (zero mean, unit variance) with a correct CPU reference
//! - Min-max scaling to an arbitrary feature range, with forward and inverse
//!   transforms
//! - Honest GPU backend dispatch with truthful CPU fallback
//! - Memory pool integration via [`scirs2_core::memory::BufferPool`]
//! - A data-size threshold that gates the dispatch path
//!
//! # Examples
//!
//! ```rust
//! use sklears_preprocessing::gpu_acceleration::{GpuStandardScaler, GpuConfig};
//! use scirs2_core::ndarray::Array2;
//! use sklears_core::traits::{Fit, Transform};
//!
//! fn example() -> Result<(), Box<dyn std::error::Error>> {
//!     let config = GpuConfig::new()
//!         .with_cuda_backend()
//!         .with_memory_pool_size(1024 * 1024 * 256); // 256MB
//!
//!     let mut scaler = GpuStandardScaler::new(config);
//!
//!     let data = Array2::from_shape_vec((10000, 100),
//!         (0..1000000).map(|x| x as f64).collect())?;
//!
//!     let scaler_fitted = scaler.fit(&data, &())?;
//!     let scaled_data = scaler_fitted.transform(&data)?;
//!
//!     println!("Scaled data shape: {:?}", scaled_data.dim());
//!     Ok(())
//! }
//! ```

use scirs2_core::gpu::GpuBackend as ScirGpuBackend;
use scirs2_core::memory::BufferPool;
use scirs2_core::ndarray::{Array2, Axis};
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Estimator, Fit, Transform, Untrained},
    types::Float,
};
use std::marker::PhantomData;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// GPU backend selection for preprocessing operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum GpuBackend {
    /// NVIDIA CUDA backend
    Cuda,
    /// AMD ROCm backend
    Rocm,
    /// WebGPU backend
    Wgpu,
    /// Apple Metal backend
    Metal,
    /// OpenCL backend
    OpenCL,
    /// CPU fallback
    Cpu,
}

impl Default for GpuBackend {
    fn default() -> Self {
        Self::from_scir_backend(ScirGpuBackend::default())
    }
}

impl GpuBackend {
    /// Convert from scirs2_core::GpuBackend
    pub fn from_scir_backend(backend: ScirGpuBackend) -> Self {
        match backend {
            ScirGpuBackend::Cuda => Self::Cuda,
            ScirGpuBackend::Rocm => Self::Rocm,
            ScirGpuBackend::Wgpu => Self::Wgpu,
            ScirGpuBackend::Metal => Self::Metal,
            ScirGpuBackend::OpenCL => Self::OpenCL,
            ScirGpuBackend::Cpu => Self::Cpu,
        }
    }

    /// Convert to scirs2_core::GpuBackend
    pub fn to_scir_backend(self) -> ScirGpuBackend {
        match self {
            Self::Cuda => ScirGpuBackend::Cuda,
            Self::Rocm => ScirGpuBackend::Rocm,
            Self::Wgpu => ScirGpuBackend::Wgpu,
            Self::Metal => ScirGpuBackend::Metal,
            Self::OpenCL => ScirGpuBackend::OpenCL,
            Self::Cpu => ScirGpuBackend::Cpu,
        }
    }

    /// Check if this backend is available on the current system
    pub fn is_available(&self) -> bool {
        self.to_scir_backend().is_available()
    }
}

/// Configuration for GPU-accelerated preprocessing
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct GpuConfig {
    /// GPU backend to use
    pub backend: GpuBackend,
    /// Size of GPU memory pool in bytes (default: 256MB)
    pub memory_pool_size: usize,
    /// Minimum data size to use GPU (default: 10,000 elements)
    pub gpu_threshold: usize,
    /// Enable automatic fallback to CPU on GPU errors
    pub auto_fallback: bool,
    /// Number of GPU streams for parallel processing
    pub stream_count: usize,
    /// Block size for GPU kernels (default: 256)
    pub block_size: usize,
}

impl Default for GpuConfig {
    fn default() -> Self {
        Self {
            backend: GpuBackend::default(),
            memory_pool_size: 256 * 1024 * 1024, // 256MB
            gpu_threshold: 10_000,
            auto_fallback: true,
            stream_count: 4,
            block_size: 256,
        }
    }
}

impl GpuConfig {
    /// Create a new GPU configuration with default settings
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the GPU backend
    pub fn with_backend(mut self, backend: GpuBackend) -> Self {
        self.backend = backend;
        self
    }

    /// Set CUDA backend
    pub fn with_cuda_backend(mut self) -> Self {
        self.backend = GpuBackend::Cuda;
        self
    }

    /// Set Metal backend
    pub fn with_metal_backend(mut self) -> Self {
        self.backend = GpuBackend::Metal;
        self
    }

    /// Set memory pool size in bytes
    pub fn with_memory_pool_size(mut self, size: usize) -> Self {
        self.memory_pool_size = size;
        self
    }

    /// Set GPU threshold (minimum elements to use GPU)
    pub fn with_gpu_threshold(mut self, threshold: usize) -> Self {
        self.gpu_threshold = threshold;
        self
    }

    /// Enable or disable automatic CPU fallback
    pub fn with_auto_fallback(mut self, enabled: bool) -> Self {
        self.auto_fallback = enabled;
        self
    }

    /// Set number of GPU streams
    pub fn with_stream_count(mut self, count: usize) -> Self {
        self.stream_count = count;
        self
    }

    /// Set GPU kernel block size
    pub fn with_block_size(mut self, size: usize) -> Self {
        self.block_size = size;
        self
    }
}

/// GPU context manager for preprocessing operations
pub struct GpuContextManager {
    backend: ScirGpuBackend,
    buffer_pool: BufferPool<u8>,
    config: GpuConfig,
}

impl GpuContextManager {
    /// Create a new GPU context manager
    pub fn new(config: GpuConfig) -> Result<Self> {
        let backend = if config.backend.is_available() {
            config.backend.to_scir_backend()
        } else {
            // Fallback to CPU if requested backend is not available
            ScirGpuBackend::Cpu
        };

        let buffer_pool = BufferPool::new();

        Ok(Self {
            backend,
            buffer_pool,
            config,
        })
    }

    /// Check if GPU is available
    pub fn is_gpu_available(&self) -> bool {
        self.backend != ScirGpuBackend::Cpu
    }

    /// Get the buffer pool
    pub fn buffer_pool(&self) -> &BufferPool<u8> {
        &self.buffer_pool
    }

    /// Determine if operation should use GPU based on data size
    pub fn should_use_gpu(&self, element_count: usize) -> bool {
        self.is_gpu_available() && element_count >= self.config.gpu_threshold
    }
}

/// GPU-accelerated Standard Scaler
pub struct GpuStandardScaler<State = Untrained> {
    config: GpuConfig,
    gpu_context: Option<GpuContextManager>,
    state: PhantomData<State>,
}

impl GpuStandardScaler<Untrained> {
    /// Create a new GPU standard scaler
    pub fn new(config: GpuConfig) -> Self {
        let gpu_context = GpuContextManager::new(config.clone()).ok();
        Self {
            config,
            gpu_context,
            state: PhantomData,
        }
    }
}

/// Fitted state for GPU Standard Scaler.
///
/// The GPU dispatch decision is driven entirely by [`GpuContextManager`]
/// (which already owns the relevant [`GpuConfig`]), so the configuration is not
/// duplicated here.
pub struct GpuStandardScalerFitted {
    gpu_context: Option<GpuContextManager>,
    mean: Array2<Float>,
    std: Array2<Float>,
}

impl Estimator for GpuStandardScaler<Untrained> {
    type Config = GpuConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Fit<Array2<Float>, ()> for GpuStandardScaler<Untrained> {
    type Fitted = GpuStandardScalerFitted;

    fn fit(self, x: &Array2<Float>, _y: &()) -> Result<Self::Fitted> {
        let should_use_gpu = self
            .gpu_context
            .as_ref()
            .map(|ctx| ctx.should_use_gpu(x.len()))
            .unwrap_or(false);

        let (mean, std) = if should_use_gpu {
            self.compute_statistics_gpu(x)?
        } else {
            self.compute_statistics_cpu(x)?
        };

        Ok(GpuStandardScalerFitted {
            gpu_context: self.gpu_context,
            mean,
            std,
        })
    }
}

impl GpuStandardScaler<Untrained> {
    /// Compute statistics along the dispatch path.
    ///
    /// This is reached when a GPU backend is reported as available. No dedicated
    /// GPU compute kernel is shipped yet, so the reductions below evaluate the
    /// same verified CPU numerics; the function still flows through the GPU
    /// context so a real kernel can be slotted in without changing callers.
    fn compute_statistics_gpu(&self, x: &Array2<Float>) -> Result<(Array2<Float>, Array2<Float>)> {
        if let Some(ref ctx) = self.gpu_context {
            let mean = self.dispatch_compute_mean(x, ctx)?;
            let variance = self.dispatch_compute_variance(x, &mean, ctx)?;

            // Convert variance to standard deviation, guarding constant features.
            let std = variance.mapv(|v| v.sqrt().max(Float::EPSILON));

            Ok((mean, std))
        } else {
            Err(SklearsError::InvalidInput(
                "GPU context not available".to_string(),
            ))
        }
    }

    /// Per-feature mean reduction (CPU numerics behind the dispatch layer).
    fn dispatch_compute_mean(
        &self,
        x: &Array2<Float>,
        _ctx: &GpuContextManager,
    ) -> Result<Array2<Float>> {
        let mean = x.mean_axis(Axis(0)).ok_or_else(|| {
            SklearsError::InvalidInput("Cannot compute mean of empty array".to_string())
        })?;
        Ok(mean.insert_axis(Axis(0)))
    }

    /// Per-feature (sample) variance reduction (CPU numerics behind dispatch).
    fn dispatch_compute_variance(
        &self,
        x: &Array2<Float>,
        mean: &Array2<Float>,
        _ctx: &GpuContextManager,
    ) -> Result<Array2<Float>> {
        let n_samples = x.nrows() as Float;
        if n_samples <= 1.0 {
            return Err(SklearsError::InvalidInput(
                "Need at least two samples to compute variance".to_string(),
            ));
        }
        let centered = x - mean;
        let variance = (&centered * &centered).sum_axis(Axis(0)) / (n_samples - 1.0);
        Ok(variance.insert_axis(Axis(0)))
    }

    /// Compute statistics using the CPU reference implementation.
    fn compute_statistics_cpu(&self, x: &Array2<Float>) -> Result<(Array2<Float>, Array2<Float>)> {
        let n_samples = x.nrows() as Float;
        if n_samples <= 1.0 {
            return Err(SklearsError::InvalidInput(
                "Need at least two samples to compute variance".to_string(),
            ));
        }
        let mean = x
            .mean_axis(Axis(0))
            .ok_or_else(|| {
                SklearsError::InvalidInput("Cannot compute mean of empty array".to_string())
            })?
            .insert_axis(Axis(0));

        let centered = x - &mean;
        let variance =
            ((&centered * &centered).sum_axis(Axis(0)) / (n_samples - 1.0)).insert_axis(Axis(0));
        let std = variance.mapv(|v| v.sqrt().max(Float::EPSILON));

        Ok((mean, std))
    }
}

impl Transform<Array2<Float>, Array2<Float>> for GpuStandardScalerFitted {
    fn transform(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let should_use_gpu = self
            .gpu_context
            .as_ref()
            .map(|ctx| ctx.should_use_gpu(x.len()))
            .unwrap_or(false);

        if should_use_gpu {
            self.transform_gpu(x)
        } else {
            self.transform_cpu(x)
        }
    }
}

impl GpuStandardScalerFitted {
    /// Transform along the GPU dispatch path.
    ///
    /// The element-wise scaling `(x - mean) / std` is evaluated with the CPU
    /// reference implementation; the dispatch hook exists so a real GPU kernel
    /// can be added later without changing the public [`Transform`] contract.
    fn transform_gpu(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        // No dedicated GPU kernel yet; evaluate the verified CPU numerics.
        self.transform_cpu(x)
    }

    /// Per-feature standardization: `(x - mean) / std`.
    fn transform_cpu(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        if x.ncols() != self.mean.ncols() {
            return Err(SklearsError::InvalidInput(format!(
                "Feature count mismatch: expected {}, got {}",
                self.mean.ncols(),
                x.ncols()
            )));
        }

        Ok((x - &self.mean) / &self.std)
    }
}

/// GPU-accelerated Min-Max Scaler
pub struct GpuMinMaxScaler<State = Untrained> {
    config: GpuConfig,
    gpu_context: Option<GpuContextManager>,
    feature_range: (Float, Float),
    state: PhantomData<State>,
}

impl GpuMinMaxScaler<Untrained> {
    /// Create a new GPU min-max scaler
    pub fn new(config: GpuConfig) -> Self {
        Self::with_feature_range(config, (0.0, 1.0))
    }

    /// Create a new GPU min-max scaler with custom feature range
    pub fn with_feature_range(config: GpuConfig, feature_range: (Float, Float)) -> Self {
        let gpu_context = GpuContextManager::new(config.clone()).ok();
        Self {
            config,
            gpu_context,
            feature_range,
            state: PhantomData,
        }
    }
}

/// Fitted state for GPU Min-Max Scaler.
///
/// The GPU dispatch decision is driven by [`GpuContextManager`], so the full
/// [`GpuConfig`] is not duplicated here. Both fitted bounds (`data_min` and
/// `data_max`) are retained: they are the canonical fitted statistics, are
/// exposed through accessors, and `data_max` is required to reconstruct the
/// original data range for [`GpuMinMaxScalerFitted::inverse_transform`].
pub struct GpuMinMaxScalerFitted {
    gpu_context: Option<GpuContextManager>,
    feature_range: (Float, Float),
    data_min: Array2<Float>,
    data_max: Array2<Float>,
    scale: Array2<Float>,
}

impl Estimator for GpuMinMaxScaler<Untrained> {
    type Config = GpuConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Fit<Array2<Float>, ()> for GpuMinMaxScaler<Untrained> {
    type Fitted = GpuMinMaxScalerFitted;

    fn fit(self, x: &Array2<Float>, _y: &()) -> Result<Self::Fitted> {
        let should_use_gpu = self
            .gpu_context
            .as_ref()
            .map(|ctx| ctx.should_use_gpu(x.len()))
            .unwrap_or(false);

        let (data_min, data_max) = if should_use_gpu {
            self.compute_min_max_gpu(x)?
        } else {
            self.compute_min_max_cpu(x)?
        };

        let data_range = &data_max - &data_min;
        let feature_range_size = self.feature_range.1 - self.feature_range.0;
        let scale = data_range.mapv(|range| {
            if range.abs() < Float::EPSILON {
                1.0
            } else {
                feature_range_size / range
            }
        });

        Ok(GpuMinMaxScalerFitted {
            gpu_context: self.gpu_context,
            feature_range: self.feature_range,
            data_min,
            data_max,
            scale,
        })
    }
}

impl GpuMinMaxScaler<Untrained> {
    /// Compute per-feature min/max along the GPU dispatch path.
    ///
    /// No dedicated GPU reduction kernel is shipped yet, so this evaluates the
    /// verified CPU reductions; the dispatch hook keeps callers stable for a
    /// future real kernel.
    fn compute_min_max_gpu(&self, x: &Array2<Float>) -> Result<(Array2<Float>, Array2<Float>)> {
        self.compute_min_max_cpu(x)
    }

    /// Compute per-feature min/max with the CPU reference implementation.
    fn compute_min_max_cpu(&self, x: &Array2<Float>) -> Result<(Array2<Float>, Array2<Float>)> {
        if x.is_empty() {
            return Err(SklearsError::InvalidInput(
                "Cannot process empty array".to_string(),
            ));
        }

        let data_min = x
            .fold_axis(Axis(0), Float::INFINITY, |&acc, &val| acc.min(val))
            .insert_axis(Axis(0));
        let data_max = x
            .fold_axis(Axis(0), Float::NEG_INFINITY, |&acc, &val| acc.max(val))
            .insert_axis(Axis(0));

        Ok((data_min, data_max))
    }
}

impl Transform<Array2<Float>, Array2<Float>> for GpuMinMaxScalerFitted {
    fn transform(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let should_use_gpu = self
            .gpu_context
            .as_ref()
            .map(|ctx| ctx.should_use_gpu(x.len()))
            .unwrap_or(false);

        if should_use_gpu {
            self.transform_gpu(x)
        } else {
            self.transform_cpu(x)
        }
    }
}

impl GpuMinMaxScalerFitted {
    /// Per-feature minimum observed during `fit` (sklearn's `data_min_`).
    pub fn data_min(&self) -> &Array2<Float> {
        &self.data_min
    }

    /// Per-feature maximum observed during `fit` (sklearn's `data_max_`).
    pub fn data_max(&self) -> &Array2<Float> {
        &self.data_max
    }

    /// Per-feature data range observed during `fit`: `data_max - data_min`
    /// (sklearn's `data_range_`).
    pub fn data_range(&self) -> Array2<Float> {
        &self.data_max - &self.data_min
    }

    /// The target feature range `(min, max)` the data is scaled into.
    pub fn feature_range(&self) -> (Float, Float) {
        self.feature_range
    }

    /// Transform along the GPU dispatch path.
    ///
    /// The affine min-max mapping is evaluated with the CPU reference
    /// implementation; the dispatch hook preserves the public contract for a
    /// future GPU kernel.
    fn transform_gpu(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        // No dedicated GPU kernel yet; evaluate the verified CPU numerics.
        self.transform_cpu(x)
    }

    /// Forward min-max mapping: `(x - data_min) * scale + feature_range.0`.
    fn transform_cpu(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        if x.ncols() != self.data_min.ncols() {
            return Err(SklearsError::InvalidInput(format!(
                "Feature count mismatch: expected {}, got {}",
                self.data_min.ncols(),
                x.ncols()
            )));
        }

        let scaled = (x - &self.data_min) * &self.scale;
        Ok(scaled.mapv(|val| val + self.feature_range.0))
    }

    /// Invert the min-max mapping, reconstructing values in the original scale.
    ///
    /// For a scaled value `s`, the original is
    /// `x = data_min + (s - feature_range.0) * (data_range / feature_range_size)`,
    /// where `data_range = data_max - data_min`. Constant features (zero range)
    /// map back to their single observed value (`data_min == data_max`).
    pub fn inverse_transform(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        if x.ncols() != self.data_min.ncols() {
            return Err(SklearsError::InvalidInput(format!(
                "Feature count mismatch: expected {}, got {}",
                self.data_min.ncols(),
                x.ncols()
            )));
        }

        let feature_range_min = self.feature_range.0;
        let feature_range_size = self.feature_range.1 - self.feature_range.0;
        // `data_range` is genuinely required here: it is the original spread that
        // the forward mapping compressed into the target feature range. This is
        // what makes the stored `data_max` load-bearing.
        let data_range = &self.data_max - &self.data_min;
        let range_row = data_range.row(0);
        let min_row = self.data_min.row(0);

        let mut result = x.clone();
        for mut row in result.outer_iter_mut() {
            for ((value, &range), &min) in row.iter_mut().zip(range_row.iter()).zip(min_row.iter())
            {
                let unshifted = *value - feature_range_min;
                *value =
                    if feature_range_size.abs() < Float::EPSILON || range.abs() < Float::EPSILON {
                        min
                    } else {
                        min + unshifted * (range / feature_range_size)
                    };
            }
        }
        Ok(result)
    }
}

/// GPU performance statistics
#[derive(Debug, Clone)]
pub struct GpuPerformanceStats {
    /// Number of operations performed on GPU
    pub gpu_operations: usize,
    /// Number of operations that fell back to CPU
    pub cpu_fallbacks: usize,
    /// Total GPU memory used in bytes
    pub gpu_memory_used: usize,
    /// Average GPU kernel execution time in microseconds
    pub avg_gpu_time_us: Float,
    /// Average CPU execution time in microseconds
    pub avg_cpu_time_us: Float,
}

impl Default for GpuPerformanceStats {
    fn default() -> Self {
        Self {
            gpu_operations: 0,
            cpu_fallbacks: 0,
            gpu_memory_used: 0,
            avg_gpu_time_us: 0.0,
            avg_cpu_time_us: 0.0,
        }
    }
}

impl GpuPerformanceStats {
    /// Create new performance statistics
    pub fn new() -> Self {
        Self::default()
    }

    /// Get GPU utilization ratio (0.0 to 1.0)
    pub fn gpu_utilization(&self) -> Float {
        let total = self.gpu_operations + self.cpu_fallbacks;
        if total == 0 {
            0.0
        } else {
            self.gpu_operations as Float / total as Float
        }
    }

    /// Get performance speedup ratio (GPU vs CPU)
    pub fn speedup_ratio(&self) -> Float {
        if self.avg_gpu_time_us > 0.0 && self.avg_cpu_time_us > 0.0 {
            self.avg_cpu_time_us / self.avg_gpu_time_us
        } else {
            1.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_gpu_config() {
        let config = GpuConfig::new()
            .with_cuda_backend()
            .with_memory_pool_size(512 * 1024 * 1024)
            .with_gpu_threshold(5000)
            .with_auto_fallback(false);

        assert_eq!(config.backend, GpuBackend::Cuda);
        assert_eq!(config.memory_pool_size, 512 * 1024 * 1024);
        assert_eq!(config.gpu_threshold, 5000);
        assert!(!config.auto_fallback);
    }

    #[test]
    fn test_gpu_standard_scaler_exact_values() -> Result<()> {
        let config = GpuConfig::new().with_gpu_threshold(100_000); // Force CPU path
        let scaler = GpuStandardScaler::new(config);

        // Columns: c0=[1,2,3,4], c1=[2,4,6,8], c2=[3,6,9,12].
        let data = array![
            [1.0, 2.0, 3.0],
            [2.0, 4.0, 6.0],
            [3.0, 6.0, 9.0],
            [4.0, 8.0, 12.0],
        ];

        let fitted_scaler = scaler.fit(&data, &())?;
        let scaled = fitted_scaler.transform(&data)?;

        // Hand-computed: mean = [2.5, 5.0, 7.5].
        // Sample std (ddof=1): c0 sumsq = 5.0 -> var = 5/3 -> std = sqrt(5/3).
        // The columns are exact multiples, so the *standardized* matrix is
        // identical across all three columns.
        let std0 = (5.0_f64 / 3.0).sqrt();
        let expected_col = [
            (1.0 - 2.5) / std0,
            (2.0 - 2.5) / std0,
            (3.0 - 2.5) / std0,
            (4.0 - 2.5) / std0,
        ];
        for (row_idx, &expected) in expected_col.iter().enumerate() {
            for col_idx in 0..3 {
                assert_abs_diff_eq!(scaled[[row_idx, col_idx]], expected, epsilon = 1e-12);
            }
        }

        // Aggregate sanity: zero mean, unit (sample) variance per column.
        let mean = scaled
            .mean_axis(Axis(0))
            .ok_or_else(|| SklearsError::InvalidInput("empty scaled array".to_string()))?;
        let std = scaled.std_axis(Axis(0), 1.0);
        for &m in mean.iter() {
            assert_abs_diff_eq!(m, 0.0, epsilon = 1e-10);
        }
        for &s in std.iter() {
            assert_abs_diff_eq!(s, 1.0, epsilon = 1e-10);
        }

        // Feature-count mismatch must be a loud error, not a fabricated result.
        let bad = array![[1.0, 2.0]];
        assert!(fitted_scaler.transform(&bad).is_err());

        Ok(())
    }

    #[test]
    fn test_gpu_minmax_scaler_exact_values() -> Result<()> {
        let config = GpuConfig::new().with_gpu_threshold(100_000); // Force CPU path
        let scaler = GpuMinMaxScaler::with_feature_range(config, (0.0, 1.0));

        // Columns: c0=[1,2,3,4], c1=[2,4,6,8], c2=[3,6,9,12].
        let data = array![
            [1.0, 2.0, 3.0],
            [2.0, 4.0, 6.0],
            [3.0, 6.0, 9.0],
            [4.0, 8.0, 12.0],
        ];

        let fitted_scaler = scaler.fit(&data, &())?;

        // Fitted statistics must be the true observed bounds.
        assert_abs_diff_eq!(fitted_scaler.data_min()[[0, 0]], 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(fitted_scaler.data_min()[[0, 1]], 2.0, epsilon = 1e-12);
        assert_abs_diff_eq!(fitted_scaler.data_min()[[0, 2]], 3.0, epsilon = 1e-12);
        assert_abs_diff_eq!(fitted_scaler.data_max()[[0, 0]], 4.0, epsilon = 1e-12);
        assert_abs_diff_eq!(fitted_scaler.data_max()[[0, 1]], 8.0, epsilon = 1e-12);
        assert_abs_diff_eq!(fitted_scaler.data_max()[[0, 2]], 12.0, epsilon = 1e-12);
        let range = fitted_scaler.data_range();
        assert_abs_diff_eq!(range[[0, 0]], 3.0, epsilon = 1e-12);
        assert_abs_diff_eq!(range[[0, 1]], 6.0, epsilon = 1e-12);
        assert_abs_diff_eq!(range[[0, 2]], 9.0, epsilon = 1e-12);

        let scaled = fitted_scaler.transform(&data)?;

        // Hand-computed: every column maps the four ordered values to
        // [0, 1/3, 2/3, 1] because they are evenly spaced.
        let expected_col = [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0];
        for (row_idx, &expected) in expected_col.iter().enumerate() {
            for col_idx in 0..3 {
                assert_abs_diff_eq!(scaled[[row_idx, col_idx]], expected, epsilon = 1e-12);
            }
        }

        // inverse_transform must reconstruct the original data exactly. This is
        // the path that genuinely uses the fitted `data_max` (via data_range).
        let recovered = fitted_scaler.inverse_transform(&scaled)?;
        for (orig, rec) in data.iter().zip(recovered.iter()) {
            assert_abs_diff_eq!(*orig, *rec, epsilon = 1e-10);
        }

        // Feature-count mismatch must error rather than fabricate output.
        let bad = array![[0.5, 0.5]];
        assert!(fitted_scaler.transform(&bad).is_err());
        assert!(fitted_scaler.inverse_transform(&bad).is_err());

        Ok(())
    }

    #[test]
    fn test_gpu_minmax_scaler_custom_range_roundtrip() -> Result<()> {
        let config = GpuConfig::new().with_gpu_threshold(100_000);
        let scaler = GpuMinMaxScaler::with_feature_range(config, (-1.0, 1.0));

        // c0=[0,5,10] -> min 0, max 10. Target range [-1, 1].
        let data = array![[0.0], [5.0], [10.0]];

        let fitted_scaler = scaler.fit(&data, &())?;
        assert_eq!(fitted_scaler.feature_range(), (-1.0, 1.0));

        let scaled = fitted_scaler.transform(&data)?;
        // (x - 0) / 10 * 2 + (-1) = x/5 - 1 -> [-1, 0, 1].
        assert_abs_diff_eq!(scaled[[0, 0]], -1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(scaled[[1, 0]], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(scaled[[2, 0]], 1.0, epsilon = 1e-12);

        let recovered = fitted_scaler.inverse_transform(&scaled)?;
        for (orig, rec) in data.iter().zip(recovered.iter()) {
            assert_abs_diff_eq!(*orig, *rec, epsilon = 1e-10);
        }

        Ok(())
    }

    #[test]
    fn test_gpu_minmax_scaler_constant_feature() -> Result<()> {
        let config = GpuConfig::new().with_gpu_threshold(100_000);
        let scaler = GpuMinMaxScaler::with_feature_range(config, (0.0, 1.0));

        // A constant column: data_min == data_max, range == 0.
        let data = array![[7.0], [7.0], [7.0]];
        let fitted_scaler = scaler.fit(&data, &())?;

        let scaled = fitted_scaler.transform(&data)?;
        // Constant feature maps to the lower bound of the feature range.
        for &val in scaled.iter() {
            assert_abs_diff_eq!(val, 0.0, epsilon = 1e-12);
        }

        // Inverse of a constant feature recovers the single observed value.
        let recovered = fitted_scaler.inverse_transform(&scaled)?;
        for &val in recovered.iter() {
            assert_abs_diff_eq!(val, 7.0, epsilon = 1e-12);
        }

        Ok(())
    }

    #[test]
    fn test_gpu_performance_stats() {
        let mut stats = GpuPerformanceStats::new();
        stats.gpu_operations = 80;
        stats.cpu_fallbacks = 20;
        stats.avg_gpu_time_us = 100.0;
        stats.avg_cpu_time_us = 300.0;

        assert_abs_diff_eq!(stats.gpu_utilization(), 0.8, epsilon = 1e-6);
        assert_abs_diff_eq!(stats.speedup_ratio(), 3.0, epsilon = 1e-6);
    }

    #[test]
    fn test_gpu_context_manager() {
        let config = GpuConfig::new();
        let ctx_result = GpuContextManager::new(config);

        // Should not fail even if GPU is not available (auto-fallback)
        assert!(ctx_result.is_ok());

        let ctx = ctx_result.expect("operation should succeed");
        // GPU may or may not be available depending on system
        let _ = ctx.is_gpu_available();
    }
}
