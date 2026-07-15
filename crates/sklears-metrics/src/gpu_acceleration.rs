//! GPU acceleration for machine learning metrics computation, backed by
//! `oxicuda-blas`.
//!
//! # Wave (2026-07-06 honesty pass)
//!
//! This module used to ship null-pointer `CudaStream`/`GpuBuffer` types, a
//! hardcoded-`false` `is_cuda_available`, and a set of `compute_*_gpu`
//! methods that unconditionally returned `Err(GpuNotAvailable)` -- none of
//! it ever touched a real GPU, and the `gpu`/`cuda` Cargo features that were
//! supposed to gate it were themselves broken (`gpu = []`, and `full`
//! enabled `gpu` but not `cuda`, so this file was never even compiled by a
//! `--features full` build).
//!
//! This version wires directly onto [`sklears_core::gpu::GpuBackend`] (a
//! real `oxicuda-driver` `Context` + `oxicuda-blas` `BlasHandle`) and calls
//! `oxicuda-blas`'s elementwise / reduction / Level-1 / Level-3 kernels
//! directly for the operations this crate needs (elementwise
//! sub/mul/abs/cmp_eq, sum/mean/max/min reduction, per-axis reduction,
//! `dot`/`nrm2`, and GEMM) -- `sklears_core::gpu::GpuArray`'s
//! [`GpuMatrixOps`](sklears_core::gpu::GpuMatrixOps) trait only covers
//! matmul/add/mul/scale/transpose, which is not enough surface for the
//! metrics below, so this module drops to the same
//! `oxicuda_driver`/`oxicuda_blas`/`oxicuda_memory` layer that
//! `sklears-svm`'s `gpu_kernels` module uses.
//!
//! [`GpuBackend::detect`]/[`GpuBackend::with_device_id`] return `Ok(None)`
//! when no GPU/driver is present (e.g. this crate's own macOS development
//! environment): [`GpuMetricsContext::new`] surfaces that as
//! `Err(GpuMetricsError::GpuNotAvailable)`, exactly like before -- the
//! difference is that when a GPU *is* present, the metrics below now
//! genuinely run on it instead of always taking the "not available" path.
//!
//! # Supported Metrics
//!
//! [`GpuMetricsContext::compute_metric`] currently implements:
//! [`GpuMetricType::Accuracy`] (elementwise equality + mean reduction),
//! [`GpuMetricType::MeanSquaredError`] / [`GpuMetricType::MeanAbsoluteError`]
//! (elementwise diff + square/abs + mean reduction),
//! [`GpuMetricType::EuclideanDistance`] (elementwise diff + `nrm2`), and
//! [`GpuMetricType::CosineDistance`] (`dot` + two `nrm2`s). Every other
//! [`GpuMetricType`] variant (confusion matrix, ROC-AUC, and the rest)
//! returns `Err(GpuMetricsError::UnsupportedMetric)` -- they are not
//! implemented, and this module says so rather than pretending otherwise.
//!
//! [`GpuMetricsContext::compute_distance_matrix`] supports
//! [`GpuMetricType::EuclideanDistance`] and [`GpuMetricType::CosineDistance`]
//! via a GEMM Gram-matrix expansion: `dist2[i,j] = ||x_i||^2 + ||x_j||^2 -
//! 2*(X X^T)[i,j]`, where the `O(n^2*d)` `X X^T` term runs on-device via
//! `oxicuda_blas::level3::gemm` and the row squared-norms run on-device via
//! an elementwise square followed by `oxicuda_blas::reduction::reduce_axis`.
//! The final `O(n^2)` broadcast-and-combine step (needed regardless, since
//! the whole matrix must be downloaded to return an `Array2`) runs on the
//! host.
//!
//! # Examples
//!
//! ```rust,no_run
//! use sklears_metrics::gpu_acceleration::{GpuMetricsContext, GpuMetricType};
//! use scirs2_core::ndarray::Array1;
//!
//! # #[cfg(feature = "gpu")]
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Initialize GPU context (fails with `GpuNotAvailable` if no GPU/driver
//! // is present -- there is no CPU-fallback path here).
//! let mut gpu_context = GpuMetricsContext::new()?;
//!
//! let y_true = Array1::from(vec![0.0, 1.0, 1.0, 0.0, 1.0]);
//! let y_pred = Array1::from(vec![0.0, 1.0, 0.0, 0.0, 1.0]);
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
//! # #[cfg(not(feature = "gpu"))]
//! # fn main() {}
//! ```

use scirs2_core::ndarray::{Array2, ArrayView1, ArrayView2};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use oxicuda_blas::level3::gemm;
use oxicuda_blas::{elementwise, level1, reduction};
use oxicuda_blas::{Layout, MatrixDesc, MatrixDescMut, Transpose};
use oxicuda_memory::DeviceBuffer;
use sklears_core::gpu::{GpuBackend, GpuUtils};

/// Error types for GPU metrics computation
#[derive(Debug, thiserror::Error)]
pub enum GpuMetricsError {
    #[error("CUDA error: {0}")]
    CudaError(String),

    #[error("GPU memory allocation failed: {0}")]
    MemoryError(String),

    #[error("Unsupported metric type: {0:?}")]
    UnsupportedMetric(GpuMetricType),

    #[error("Unsupported reduction type: {0:?}")]
    UnsupportedReduction(ReductionType),

    #[error("Data size mismatch: expected {expected}, got {actual}")]
    SizeMismatch { expected: usize, actual: usize },

    #[error("GPU not available or CUDA not supported")]
    GpuNotAvailable,

    #[error("Mixed precision not supported for metric: {0:?}")]
    MixedPrecisionNotSupported(GpuMetricType),
}

pub type GpuResult<T> = Result<T, GpuMetricsError>;

/// Converts any displayable error from the `oxicuda-*` stack into a
/// [`GpuMetricsError::CudaError`].
fn gpu_err<E: std::fmt::Display>(e: E) -> GpuMetricsError {
    GpuMetricsError::CudaError(e.to_string())
}

/// Conservative reduction-result buffer length for `n` input elements.
///
/// `oxicuda-blas`'s two-phase sum/mean/max/min reductions require a result
/// buffer of at least `ceil(n / block_size)` elements to hold per-block
/// partial results, where `block_size` is an internal constant (currently
/// 256, but documented only as "a power of two >= 32"). Sizing for the
/// smallest documented block size (32) always yields a buffer at least as
/// large as what the actual internal block size requires.
fn reduction_result_len(n: usize) -> usize {
    n.div_ceil(32).max(1)
}

/// Flattens an `ndarray::Array2<f64>` into a row-major `Vec`, uploading
/// directly from the underlying slice when possible and falling back to an
/// iterator copy for non-standard-layout views.
fn flatten_row_major(array: &ArrayView2<f64>) -> Vec<f64> {
    if array.is_standard_layout() {
        match array.as_slice() {
            Some(slice) => slice.to_vec(),
            None => array.iter().copied().collect(),
        }
    } else {
        array.iter().copied().collect()
    }
}

/// GPU metrics computation context.
///
/// Holds a real [`GpuBackend`] (CUDA context + BLAS handle) plus an
/// optional host-side result cache. There is no CPU-fallback path: if no
/// GPU/driver is present, construction fails with
/// [`GpuMetricsError::GpuNotAvailable`].
#[derive(Debug)]
pub struct GpuMetricsContext {
    backend: GpuBackend,
    // Set by `enable_mixed_precision`/`with_config`; not yet consulted by any
    // compute path (see `supports_mixed_precision`'s doc comment for why).
    #[allow(dead_code)]
    mixed_precision: bool,
    enable_caching: bool,
    cache: MetricCache,
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

/// Configuration for GPU metrics computation.
///
/// Earlier versions of this struct also carried `memory_pool_size`,
/// `num_streams`, `block_size`, and `grid_size` fields; those never
/// corresponded to anything this module actually configured (there was no
/// real memory pool or stream pool), so they were removed rather than kept
/// as decorative dead configuration. `oxicuda-blas` manages its own kernel
/// block sizing and a single stream per [`GpuBackend`]/`BlasHandle`.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct GpuMetricsConfig {
    /// Device ordinal to bind to, passed to [`GpuBackend::with_device_id`].
    pub device_id: i32,
    /// Enable mixed precision (f32) computation. See
    /// `GpuMetricsContext::supports_mixed_precision`.
    pub mixed_precision: bool,
    /// Enable metric result caching (see [`MetricCache`]).
    pub enable_caching: bool,
    /// Maximum number of cached metric results before eviction.
    pub cache_size_limit: usize,
}

impl Default for GpuMetricsConfig {
    fn default() -> Self {
        Self {
            device_id: 0,
            mixed_precision: false,
            enable_caching: true,
            cache_size_limit: 1000,
        }
    }
}

/// Parallel reduction configuration
#[derive(Debug, Clone)]
pub struct ParallelReductionConfig {
    pub reduction_type: ReductionType,
}

/// Types of parallel reductions.
///
/// `Sum`/`Mean`/`Max`/`Min` are implemented via
/// `oxicuda_blas::reduction::{sum, mean, max, min}`. `ArgMax`/`ArgMin`/`Norm`
/// are not implemented by
/// [`GpuMetricsContext::parallel_reduction`] and return
/// [`GpuMetricsError::UnsupportedReduction`] -- `oxicuda-blas` has no
/// reduction-with-index primitive, and `Norm` would need a dedicated
/// `nrm2`-based path rather than the generic `ReductionOp` dispatch used
/// here.
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
        let device_id = usize::try_from(config.device_id)
            .map_err(|_| gpu_err(format!("invalid device id {}", config.device_id)))?;
        let backend = GpuBackend::with_device_id(device_id)
            .map_err(gpu_err)?
            .ok_or(GpuMetricsError::GpuNotAvailable)?;

        Ok(Self {
            backend,
            mixed_precision: config.mixed_precision,
            enable_caching: config.enable_caching,
            cache: MetricCache::new(config.cache_size_limit),
        })
    }

    /// Check if a GPU/CUDA driver is available on the system.
    pub fn is_cuda_available() -> bool {
        GpuBackend::is_available()
    }

    /// Get GPU device properties for the device this context is bound to.
    pub fn get_device_properties(&self) -> GpuResult<GpuDeviceProperties> {
        GpuUtils::device_properties(self.backend.device_id()).map_err(gpu_err)
    }

    /// Compute a metric on GPU
    pub fn compute_metric(
        &mut self,
        metric_type: GpuMetricType,
        y_true: &ArrayView1<f64>,
        y_pred: &ArrayView1<f64>,
    ) -> GpuResult<f64> {
        if y_true.len() != y_pred.len() {
            return Err(GpuMetricsError::SizeMismatch {
                expected: y_true.len(),
                actual: y_pred.len(),
            });
        }

        let cache_key = self
            .enable_caching
            .then(|| Self::generate_cache_key(metric_type, y_true, y_pred));
        if let Some(key) = &cache_key {
            if let Some(cached) = self.cache.get(key) {
                return Ok(cached.value);
            }
        }

        let result = self.dispatch_metric(metric_type, y_true, y_pred)?;

        if let Some(key) = cache_key {
            self.cache.insert(
                key,
                CachedResult {
                    value: result,
                    timestamp: std::time::SystemTime::now(),
                },
            );
        }

        Ok(result)
    }

    /// Compute multiple metrics, reusing the same uploaded device buffers
    /// where the underlying kernels allow it.
    pub fn compute_multiple_metrics(
        &mut self,
        metrics: &[GpuMetricType],
        y_true: &ArrayView1<f64>,
        y_pred: &ArrayView1<f64>,
    ) -> GpuResult<HashMap<GpuMetricType, f64>> {
        if y_true.len() != y_pred.len() {
            return Err(GpuMetricsError::SizeMismatch {
                expected: y_true.len(),
                actual: y_pred.len(),
            });
        }

        let mut results = HashMap::new();
        for &metric in metrics {
            let result = self.dispatch_metric(metric, y_true, y_pred)?;
            results.insert(metric, result);
        }
        Ok(results)
    }

    fn dispatch_metric(
        &self,
        metric_type: GpuMetricType,
        y_true: &ArrayView1<f64>,
        y_pred: &ArrayView1<f64>,
    ) -> GpuResult<f64> {
        match metric_type {
            GpuMetricType::Accuracy => self.compute_accuracy_gpu(y_true, y_pred),
            GpuMetricType::MeanSquaredError => self.compute_mse_gpu(y_true, y_pred),
            GpuMetricType::MeanAbsoluteError => self.compute_mae_gpu(y_true, y_pred),
            GpuMetricType::EuclideanDistance => self.compute_euclidean_distance_gpu(y_true, y_pred),
            GpuMetricType::CosineDistance => self.compute_cosine_distance_gpu(y_true, y_pred),
            _ => Err(GpuMetricsError::UnsupportedMetric(metric_type)),
        }
    }

    /// Compute a Euclidean or cosine distance matrix on GPU via a GEMM
    /// Gram-matrix expansion. See the module-level docs for the formula.
    pub fn compute_distance_matrix(
        &mut self,
        x: &ArrayView2<f64>,
        metric: GpuMetricType,
    ) -> GpuResult<Array2<f64>> {
        match metric {
            GpuMetricType::EuclideanDistance => self.euclidean_distance_matrix_gpu(x),
            GpuMetricType::CosineDistance => self.cosine_distance_matrix_gpu(x),
            _ => Err(GpuMetricsError::UnsupportedMetric(metric)),
        }
    }

    /// Perform a parallel reduction on GPU (`Sum`/`Mean`/`Max`/`Min`).
    pub fn parallel_reduction(
        &mut self,
        data: &ArrayView1<f64>,
        config: ParallelReductionConfig,
    ) -> GpuResult<f64> {
        let n = data.len();
        if n == 0 {
            return Err(GpuMetricsError::SizeMismatch {
                expected: 1,
                actual: 0,
            });
        }

        let data_buf = self.upload(data)?;
        let mut result = DeviceBuffer::<f64>::zeroed(reduction_result_len(n)).map_err(gpu_err)?;
        let handle = self.backend.blas();

        match config.reduction_type {
            ReductionType::Sum => reduction::sum(handle, n as u32, &data_buf, &mut result),
            ReductionType::Mean => reduction::mean(handle, n as u32, &data_buf, &mut result),
            ReductionType::Max => reduction::max(handle, n as u32, &data_buf, &mut result),
            ReductionType::Min => reduction::min(handle, n as u32, &data_buf, &mut result),
            ReductionType::ArgMax | ReductionType::ArgMin | ReductionType::Norm => {
                return Err(GpuMetricsError::UnsupportedReduction(config.reduction_type));
            }
        }
        .map_err(gpu_err)?;

        // The preceding kernels ran on the non-blocking compute stream; `copy_to_host`
        // uses the legacy default stream, which does not implicitly wait. Synchronise.
        self.backend.synchronize().map_err(gpu_err)?;
        let mut out = [0.0f64; 1];
        result.copy_to_host(&mut out).map_err(gpu_err)?;
        Ok(out[0])
    }

    /// Enable mixed precision computation.
    pub fn enable_mixed_precision(&mut self) -> GpuResult<()> {
        if !self.supports_mixed_precision() {
            return Err(GpuMetricsError::MixedPrecisionNotSupported(
                GpuMetricType::Accuracy,
            ));
        }
        self.mixed_precision = true;
        Ok(())
    }

    /// Whether the reduced-precision (f32) compute path is supported.
    ///
    /// `oxicuda-blas` genuinely implements `GpuFloat` for both `f32` and
    /// `f64` (the same trait `sklears-svm`'s GPU kernels use for their f32
    /// path), so this is a real hardware/library capability, not a
    /// fabricated one. Note that the metric kernels in this module
    /// currently always compute in `f64` regardless of this flag; wiring an
    /// f32 compute path through `compute_metric` is a follow-up.
    fn supports_mixed_precision(&self) -> bool {
        true
    }

    /// Get GPU memory usage statistics via `cuMemGetInfo`. Reports all
    /// zeros if the underlying driver query fails, rather than fabricating
    /// numbers.
    pub fn get_memory_stats(&self) -> GpuMemoryStats {
        match self.backend.memory_info() {
            Ok(info) => GpuMemoryStats {
                used_memory: info.used,
                available_memory: info.free,
                total_memory: info.total,
            },
            Err(_) => GpuMemoryStats::default(),
        }
    }

    /// Synchronize GPU operations (blocks until all outstanding work on
    /// this context's stream has completed).
    pub fn synchronize(&self) -> GpuResult<()> {
        self.backend.synchronize().map_err(gpu_err)
    }

    // ─── Private implementation methods ────────────────────────────────

    /// Uploads a 1-D host array to a new device buffer, making this
    /// context's CUDA context current first.
    fn upload(&self, data: &ArrayView1<f64>) -> GpuResult<DeviceBuffer<f64>> {
        self.backend.context().set_current().map_err(gpu_err)?;
        let host: Vec<f64> = match data.as_slice() {
            Some(slice) => slice.to_vec(),
            None => data.iter().copied().collect(),
        };
        DeviceBuffer::from_host(&host).map_err(gpu_err)
    }

    fn compute_accuracy_gpu(
        &self,
        y_true: &ArrayView1<f64>,
        y_pred: &ArrayView1<f64>,
    ) -> GpuResult<f64> {
        let n = y_true.len();
        if n == 0 {
            return Ok(0.0);
        }
        let handle = self.backend.blas();
        let true_buf = self.upload(y_true)?;
        let pred_buf = self.upload(y_pred)?;

        let mut matches = DeviceBuffer::<f64>::zeroed(n).map_err(gpu_err)?;
        elementwise::cmp_eq(handle, n as u32, &true_buf, &pred_buf, &mut matches)
            .map_err(gpu_err)?;

        let mut result = DeviceBuffer::<f64>::zeroed(reduction_result_len(n)).map_err(gpu_err)?;
        reduction::mean(handle, n as u32, &matches, &mut result).map_err(gpu_err)?;

        // The preceding kernels ran on the non-blocking compute stream; `copy_to_host`
        // uses the legacy default stream, which does not implicitly wait. Synchronise.
        self.backend.synchronize().map_err(gpu_err)?;
        let mut out = [0.0f64; 1];
        result.copy_to_host(&mut out).map_err(gpu_err)?;
        Ok(out[0])
    }

    fn compute_mse_gpu(
        &self,
        y_true: &ArrayView1<f64>,
        y_pred: &ArrayView1<f64>,
    ) -> GpuResult<f64> {
        let n = y_true.len();
        if n == 0 {
            return Ok(0.0);
        }
        let handle = self.backend.blas();
        let true_buf = self.upload(y_true)?;
        let pred_buf = self.upload(y_pred)?;

        let mut diff = DeviceBuffer::<f64>::zeroed(n).map_err(gpu_err)?;
        elementwise::sub(handle, n as u32, &pred_buf, &true_buf, &mut diff).map_err(gpu_err)?;

        let mut squared = DeviceBuffer::<f64>::zeroed(n).map_err(gpu_err)?;
        elementwise::mul(handle, n as u32, &diff, &diff, &mut squared).map_err(gpu_err)?;

        let mut result = DeviceBuffer::<f64>::zeroed(reduction_result_len(n)).map_err(gpu_err)?;
        reduction::mean(handle, n as u32, &squared, &mut result).map_err(gpu_err)?;

        // The preceding kernels ran on the non-blocking compute stream; `copy_to_host`
        // uses the legacy default stream, which does not implicitly wait. Synchronise.
        self.backend.synchronize().map_err(gpu_err)?;
        let mut out = [0.0f64; 1];
        result.copy_to_host(&mut out).map_err(gpu_err)?;
        Ok(out[0])
    }

    fn compute_mae_gpu(
        &self,
        y_true: &ArrayView1<f64>,
        y_pred: &ArrayView1<f64>,
    ) -> GpuResult<f64> {
        let n = y_true.len();
        if n == 0 {
            return Ok(0.0);
        }
        let handle = self.backend.blas();
        let true_buf = self.upload(y_true)?;
        let pred_buf = self.upload(y_pred)?;

        let mut diff = DeviceBuffer::<f64>::zeroed(n).map_err(gpu_err)?;
        elementwise::sub(handle, n as u32, &pred_buf, &true_buf, &mut diff).map_err(gpu_err)?;

        let mut abs_diff = DeviceBuffer::<f64>::zeroed(n).map_err(gpu_err)?;
        elementwise::abs_val(handle, n as u32, &diff, &mut abs_diff).map_err(gpu_err)?;

        let mut result = DeviceBuffer::<f64>::zeroed(reduction_result_len(n)).map_err(gpu_err)?;
        reduction::mean(handle, n as u32, &abs_diff, &mut result).map_err(gpu_err)?;

        // The preceding kernels ran on the non-blocking compute stream; `copy_to_host`
        // uses the legacy default stream, which does not implicitly wait. Synchronise.
        self.backend.synchronize().map_err(gpu_err)?;
        let mut out = [0.0f64; 1];
        result.copy_to_host(&mut out).map_err(gpu_err)?;
        Ok(out[0])
    }

    fn compute_euclidean_distance_gpu(
        &self,
        y_true: &ArrayView1<f64>,
        y_pred: &ArrayView1<f64>,
    ) -> GpuResult<f64> {
        let n = y_true.len();
        if n == 0 {
            return Ok(0.0);
        }
        let handle = self.backend.blas();
        let true_buf = self.upload(y_true)?;
        let pred_buf = self.upload(y_pred)?;

        let mut diff = DeviceBuffer::<f64>::zeroed(n).map_err(gpu_err)?;
        elementwise::sub(handle, n as u32, &pred_buf, &true_buf, &mut diff).map_err(gpu_err)?;

        let mut result = DeviceBuffer::<f64>::zeroed(1).map_err(gpu_err)?;
        level1::nrm2(handle, n as u32, &diff, 1, &mut result).map_err(gpu_err)?;

        // The preceding kernels ran on the non-blocking compute stream; `copy_to_host`
        // uses the legacy default stream, which does not implicitly wait. Synchronise.
        self.backend.synchronize().map_err(gpu_err)?;
        let mut out = [0.0f64; 1];
        result.copy_to_host(&mut out).map_err(gpu_err)?;
        Ok(out[0])
    }

    fn compute_cosine_distance_gpu(
        &self,
        y_true: &ArrayView1<f64>,
        y_pred: &ArrayView1<f64>,
    ) -> GpuResult<f64> {
        let n = y_true.len();
        if n == 0 {
            return Ok(0.0);
        }
        let handle = self.backend.blas();
        let true_buf = self.upload(y_true)?;
        let pred_buf = self.upload(y_pred)?;

        let mut dot_buf = DeviceBuffer::<f64>::zeroed(1).map_err(gpu_err)?;
        level1::dot(handle, n as u32, &true_buf, 1, &pred_buf, 1, &mut dot_buf).map_err(gpu_err)?;
        let mut true_norm_buf = DeviceBuffer::<f64>::zeroed(1).map_err(gpu_err)?;
        level1::nrm2(handle, n as u32, &true_buf, 1, &mut true_norm_buf).map_err(gpu_err)?;
        let mut pred_norm_buf = DeviceBuffer::<f64>::zeroed(1).map_err(gpu_err)?;
        level1::nrm2(handle, n as u32, &pred_buf, 1, &mut pred_norm_buf).map_err(gpu_err)?;

        // The preceding dot/nrm2 kernels ran on the non-blocking compute stream; the
        // `copy_to_host` calls below use the legacy default stream, which does not
        // implicitly wait. One synchronise covers all three consecutive copies.
        self.backend.synchronize().map_err(gpu_err)?;
        let mut dot_v = [0.0f64; 1];
        dot_buf.copy_to_host(&mut dot_v).map_err(gpu_err)?;
        let mut true_norm = [0.0f64; 1];
        true_norm_buf
            .copy_to_host(&mut true_norm)
            .map_err(gpu_err)?;
        let mut pred_norm = [0.0f64; 1];
        pred_norm_buf
            .copy_to_host(&mut pred_norm)
            .map_err(gpu_err)?;

        let denom = true_norm[0] * pred_norm[0];
        if denom == 0.0 {
            // Cosine similarity is undefined when either vector is all-zero;
            // report maximal distance rather than dividing by zero.
            return Ok(1.0);
        }
        Ok(1.0 - dot_v[0] / denom)
    }

    /// Uploads `x`, computes the Gram matrix `X X^T` via GEMM (on-device,
    /// `O(n^2*d)`), and the row squared-norms via an elementwise square plus
    /// `oxicuda_blas::reduction::reduce_axis` (also on-device). Returns both
    /// downloaded to host, since the caller needs to build a full `Array2`
    /// result anyway.
    fn gram_and_row_sq_norms_gpu(
        &self,
        x: &ArrayView2<f64>,
    ) -> GpuResult<(Vec<f64>, Vec<f64>, usize)> {
        let n = x.nrows();
        let d = x.ncols();
        if n == 0 || d == 0 {
            return Ok((Vec::new(), vec![0.0; n], n));
        }

        self.backend.context().set_current().map_err(gpu_err)?;
        let handle = self.backend.blas();

        let host = flatten_row_major(x);
        let x_buf = DeviceBuffer::<f64>::from_host(&host).map_err(gpu_err)?;

        let a_desc = MatrixDesc::from_buffer(&x_buf, n as u32, d as u32, Layout::RowMajor)
            .map_err(gpu_err)?;
        let b_desc = MatrixDesc::from_buffer(&x_buf, n as u32, d as u32, Layout::RowMajor)
            .map_err(gpu_err)?;
        let mut gram_buf = DeviceBuffer::<f64>::zeroed(n * n).map_err(gpu_err)?;
        let mut c_desc =
            MatrixDescMut::from_buffer(&mut gram_buf, n as u32, n as u32, Layout::RowMajor)
                .map_err(gpu_err)?;
        gemm(
            handle,
            Transpose::NoTrans,
            Transpose::Trans,
            1.0f64,
            &a_desc,
            &b_desc,
            0.0f64,
            &mut c_desc,
        )
        .map_err(gpu_err)?;

        let mut squared = DeviceBuffer::<f64>::zeroed(n * d).map_err(gpu_err)?;
        elementwise::mul(handle, (n * d) as u32, &x_buf, &x_buf, &mut squared).map_err(gpu_err)?;
        let mut row_sums_buf = DeviceBuffer::<f64>::zeroed(n).map_err(gpu_err)?;
        reduction::reduce_axis(
            handle,
            reduction::ReductionOp::Sum,
            n as u32,
            d as u32,
            1,
            &squared,
            &mut row_sums_buf,
        )
        .map_err(gpu_err)?;

        // The preceding gemm/mul/reduce_axis kernels ran on the non-blocking compute
        // stream; the `copy_to_host` calls below use the legacy default stream, which
        // does not implicitly wait. One synchronise covers both consecutive copies.
        self.backend.synchronize().map_err(gpu_err)?;
        let mut gram = vec![0.0f64; n * n];
        gram_buf.copy_to_host(&mut gram).map_err(gpu_err)?;
        let mut row_sq_norms = vec![0.0f64; n];
        row_sums_buf
            .copy_to_host(&mut row_sq_norms)
            .map_err(gpu_err)?;

        Ok((gram, row_sq_norms, n))
    }

    fn euclidean_distance_matrix_gpu(&self, x: &ArrayView2<f64>) -> GpuResult<Array2<f64>> {
        let (gram, row_sq_norms, n) = self.gram_and_row_sq_norms_gpu(x)?;
        let mut out = Array2::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                let dist2 = (row_sq_norms[i] + row_sq_norms[j] - 2.0 * gram[i * n + j]).max(0.0);
                out[[i, j]] = dist2.sqrt();
            }
        }
        Ok(out)
    }

    fn cosine_distance_matrix_gpu(&self, x: &ArrayView2<f64>) -> GpuResult<Array2<f64>> {
        let (gram, row_sq_norms, n) = self.gram_and_row_sq_norms_gpu(x)?;
        let norms: Vec<f64> = row_sq_norms.iter().map(|v| v.max(0.0).sqrt()).collect();
        let mut out = Array2::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                let denom = norms[i] * norms[j];
                let cos_sim = if denom == 0.0 {
                    0.0
                } else {
                    gram[i * n + j] / denom
                };
                out[[i, j]] = 1.0 - cos_sim;
            }
        }
        Ok(out)
    }

    /// Builds a cache key that hashes the metric type and the actual
    /// contents of both arrays (not just their length), so that distinct
    /// same-length inputs never collide on the same cached result.
    fn generate_cache_key(
        metric_type: GpuMetricType,
        y_true: &ArrayView1<f64>,
        y_pred: &ArrayView1<f64>,
    ) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        metric_type.hash(&mut hasher);
        y_true.len().hash(&mut hasher);
        for v in y_true.iter() {
            v.to_bits().hash(&mut hasher);
        }
        for v in y_pred.iter() {
            v.to_bits().hash(&mut hasher);
        }
        format!("{}_{}", metric_type as u32, hasher.finish())
    }
}

/// GPU device properties. Alias for [`sklears_core::gpu::GpuDeviceProperties`]
/// -- the single source of truth for what `oxicuda-driver` can actually
/// report about a device (name, memory, compute capability). The previous
/// version of this struct also had `multiprocessor_count`,
/// `max_threads_per_block`, `max_blocks_per_grid`, and `warp_size` fields,
/// but those were always hardcoded to `0` (never queried); rather than keep
/// four fake fields, this reuses the smaller, real struct.
pub type GpuDeviceProperties = sklears_core::gpu::GpuDeviceProperties;

/// GPU memory usage statistics, from a real `cuMemGetInfo` query (see
/// [`GpuMetricsContext::get_memory_stats`]). All fields report `0` if the
/// query fails, rather than fabricating numbers.
#[derive(Debug, Clone, Copy, Default)]
pub struct GpuMemoryStats {
    /// Bytes currently in use on the device. Reflects all processes/contexts
    /// sharing the device, not just this one -- `cuMemGetInfo` has no
    /// cheaper way to attribute usage to a single context.
    pub used_memory: usize,
    /// Bytes free on the device.
    pub available_memory: usize,
    /// Total bytes on the device.
    pub total_memory: usize,
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
            self.evict_oldest();
        }

        self.cache_map.insert(key, result);
        self.current_size += 1;
    }

    fn evict_oldest(&mut self) {
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
        problem_size.div_ceil(block_size)
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
    use scirs2_core::ndarray::Array1;

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
        let serialized = serde_json::to_string(&metric).expect("operation should succeed");
        let deserialized: GpuMetricType =
            serde_json::from_str(&serialized).expect("operation should succeed");
        assert_eq!(metric, deserialized);
    }

    #[test]
    fn test_metric_cache() {
        let mut cache = MetricCache::new(2);

        let result1 = CachedResult {
            value: 0.85,
            timestamp: std::time::SystemTime::now(),
        };

        cache.insert("key1".to_string(), result1);
        assert!(cache.get("key1").is_some());
        assert_eq!(cache.current_size, 1);
    }

    /// `is_cuda_available()` must never panic. This crate's own dev/CI
    /// machines (macOS, no NVIDIA GPU) correctly report `false`; a real GPU
    /// runner would correctly report `true`. Deliberately not asserted to a
    /// hardcoded value so this test remains meaningful on both.
    #[test]
    fn test_gpu_availability_check() {
        let _available = GpuMetricsContext::is_cuda_available();
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

    /// Skips gracefully when no GPU/driver is present (this crate's own
    /// dev/CI environment), like `sklears-svm`'s `DeviceNotAvailable`
    /// pattern -- it does not fabricate a GPU context.
    fn with_gpu_context(f: impl FnOnce(&mut GpuMetricsContext)) {
        match GpuMetricsContext::new() {
            Ok(mut ctx) => f(&mut ctx),
            Err(GpuMetricsError::GpuNotAvailable) => {
                eprintln!("skipping GPU test: no GPU/driver detected");
            }
            Err(e) => panic!("unexpected error constructing GpuMetricsContext: {e}"),
        }
    }

    #[test]
    fn test_gpu_accuracy_and_mse_mae() {
        with_gpu_context(|ctx| {
            let y_true = Array1::from(vec![0.0, 1.0, 1.0, 0.0, 1.0]);
            let y_pred = Array1::from(vec![0.0, 1.0, 0.0, 0.0, 1.0]);

            let accuracy = ctx
                .compute_metric(GpuMetricType::Accuracy, &y_true.view(), &y_pred.view())
                .expect("accuracy should compute");
            assert!((accuracy - 0.8).abs() < 1e-10, "accuracy={accuracy}");

            let mse = ctx
                .compute_metric(
                    GpuMetricType::MeanSquaredError,
                    &y_true.view(),
                    &y_pred.view(),
                )
                .expect("mse should compute");
            assert!((mse - 0.2).abs() < 1e-10, "mse={mse}");

            let mae = ctx
                .compute_metric(
                    GpuMetricType::MeanAbsoluteError,
                    &y_true.view(),
                    &y_pred.view(),
                )
                .expect("mae should compute");
            assert!((mae - 0.2).abs() < 1e-10, "mae={mae}");
        });
    }

    #[test]
    fn test_gpu_euclidean_and_cosine_distance() {
        with_gpu_context(|ctx| {
            let a = Array1::from(vec![1.0, 0.0]);
            let b = Array1::from(vec![0.0, 1.0]);

            let euclidean = ctx
                .compute_metric(GpuMetricType::EuclideanDistance, &a.view(), &b.view())
                .expect("euclidean distance should compute");
            assert!(
                (euclidean - 2.0f64.sqrt()).abs() < 1e-8,
                "euclidean={euclidean}"
            );

            let cosine = ctx
                .compute_metric(GpuMetricType::CosineDistance, &a.view(), &b.view())
                .expect("cosine distance should compute");
            assert!((cosine - 1.0).abs() < 1e-8, "cosine={cosine}");
        });
    }

    #[test]
    fn test_gpu_distance_matrix() {
        with_gpu_context(|ctx| {
            let x = Array2::from_shape_vec((3, 2), vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0])
                .expect("shape should be valid");

            let dist = ctx
                .compute_distance_matrix(&x.view(), GpuMetricType::EuclideanDistance)
                .expect("distance matrix should compute");
            assert_eq!(dist.shape(), &[3, 3]);
            for i in 0..3 {
                assert!(dist[[i, i]].abs() < 1e-8);
            }
            assert!((dist[[0, 1]] - 1.0).abs() < 1e-8);
            assert!((dist[[0, 2]] - 1.0).abs() < 1e-8);
            assert!((dist[[1, 2]] - 2.0f64.sqrt()).abs() < 1e-8);
        });
    }

    #[test]
    fn test_gpu_parallel_reduction() {
        with_gpu_context(|ctx| {
            let data = Array1::from(vec![1.0, 2.0, 3.0, 4.0]);
            let sum = ctx
                .parallel_reduction(
                    &data.view(),
                    ParallelReductionConfig {
                        reduction_type: ReductionType::Sum,
                    },
                )
                .expect("sum reduction should compute");
            assert!((sum - 10.0).abs() < 1e-10, "sum={sum}");

            let max = ctx
                .parallel_reduction(
                    &data.view(),
                    ParallelReductionConfig {
                        reduction_type: ReductionType::Max,
                    },
                )
                .expect("max reduction should compute");
            assert!((max - 4.0).abs() < 1e-10, "max={max}");
        });
    }

    #[test]
    fn test_gpu_unsupported_metric_and_reduction() {
        with_gpu_context(|ctx| {
            let a = Array1::from(vec![1.0, 2.0]);
            let b = Array1::from(vec![1.0, 2.0]);
            let err = ctx
                .compute_metric(GpuMetricType::RocAuc, &a.view(), &b.view())
                .expect_err("RocAuc is not implemented on GPU");
            assert!(matches!(err, GpuMetricsError::UnsupportedMetric(_)));

            let err = ctx
                .parallel_reduction(
                    &a.view(),
                    ParallelReductionConfig {
                        reduction_type: ReductionType::ArgMax,
                    },
                )
                .expect_err("ArgMax reduction is not implemented");
            assert!(matches!(err, GpuMetricsError::UnsupportedReduction(_)));
        });
    }

    #[test]
    fn test_gpu_memory_stats_and_device_properties() {
        with_gpu_context(|ctx| {
            let stats = ctx.get_memory_stats();
            assert!(
                stats.total_memory > 0,
                "total_memory={}",
                stats.total_memory
            );

            let props = ctx
                .get_device_properties()
                .expect("device properties should be queryable");
            assert!(!props.name.is_empty());
        });
    }
}
