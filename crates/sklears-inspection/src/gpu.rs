//! GPU acceleration infrastructure for explanation methods
//!
//! This module provides the foundation for GPU-accelerated explanation computation,
//! including device management, memory allocation, and honest device detection.
//!
//! Everything here is a thin, explanation-domain wrapper around
//! [`sklears_core::gpu`], which is itself backed directly by the real
//! `oxicuda-driver` / `oxicuda-blas` / `oxicuda-memory` crates. There is no
//! CPU-backed placeholder masquerading as a GPU anywhere in this module:
//! [`GpuContext::new`] / [`GpuBackend::detect`] honestly report "no GPU"
//! (`is_gpu_available() == false`) on hosts without a CUDA-capable device and
//! driver, such as this crate's own macOS development machine.
//!
//! # Features
//!
//! * Device detection via `oxicuda-driver` (CUDA only; OxiCUDA does not
//!   provide OpenCL or Metal backends)
//! * GPU memory management for explanation data via [`GpuBuffer`]
//! * Fallback to CPU explanation estimators when no GPU is available
//!
//! # Example
//!
//! ```rust,ignore
//! use sklears_inspection::gpu::{GpuContext, GpuExplanationComputer};
//!
//! // Create GPU context (honestly detects the best available device, if any)
//! let gpu_ctx = GpuContext::new()?;
//!
//! // Create GPU-accelerated explanation computer; falls back to CPU
//! // estimators transparently when `gpu_ctx.is_gpu_available()` is false.
//! let computer = GpuExplanationComputer::new(&gpu_ctx)?;
//!
//! // Perform SHAP computation (GPU-staged when available, CPU otherwise)
//! let shap_values = computer.compute_shap_parallel(&features, &background, &predict_fn).await?;
//! ```

use crate::types::Float;
use scirs2_core::ndarray::{Array1, Array2, ArrayView2};
use sklears_core::error::{Result as SklResult, SklearsError};
pub use sklears_core::gpu::{GpuArray, GpuBackend, GpuMatrixOps, GpuUtils};

/// Real device information, sourced from `oxicuda-driver` via
/// [`GpuUtils::device_properties`]. Alias kept for API continuity with the
/// pre-migration `GpuDevice` name.
pub type GpuDevice = sklears_core::gpu::GpuDeviceProperties;

/// GPU memory buffer for explanation data.
///
/// Thin wrapper around [`sklears_core::gpu::GpuArray`]: every `GpuBuffer`
/// that exists represents real device memory allocated through
/// `oxicuda-memory`. Unlike the pre-migration version there is no
/// null-pointer placeholder here -- a `GpuBuffer` can only be constructed
/// from a [`GpuBackend`], and a `GpuBackend` can only be constructed once
/// [`GpuBackend::detect`] has found a real GPU.
pub struct GpuBuffer<T: Copy> {
    inner: GpuArray<T>,
}

impl<T: Copy> GpuBuffer<T> {
    /// Uploads `data` to a new device buffer.
    pub fn from_host(backend: &GpuBackend, data: &[T]) -> SklResult<Self> {
        Ok(Self {
            inner: GpuArray::from_slice(backend, data)?,
        })
    }

    /// Allocates a zero-initialised device buffer of `size` elements.
    pub fn zeros(backend: &GpuBackend, size: usize) -> SklResult<Self> {
        Ok(Self {
            inner: GpuArray::zeros(backend, &[size])?,
        })
    }

    /// Downloads the buffer contents to the host.
    pub fn to_host(&self) -> SklResult<Vec<T>>
    where
        T: Default,
    {
        self.inner.to_cpu()
    }

    /// Number of elements in the buffer.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// `true` iff the buffer holds no elements.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

/// GPU context for managing devices and memory.
///
/// Wraps an `Option<GpuBackend>`: `None` means [`GpuBackend::detect`]
/// honestly found no usable CUDA device on this host, in which case every
/// consumer of this context (e.g. [`GpuExplanationComputer`]) falls back to
/// its CPU estimator instead of fabricating GPU results.
pub struct GpuContext {
    backend: Option<GpuBackend>,
}

impl GpuContext {
    /// Creates a context bound to the best available device (most free
    /// memory), or to no device at all if none is present.
    pub fn new() -> SklResult<Self> {
        Ok(Self {
            backend: GpuBackend::detect()?,
        })
    }

    /// Creates a context bound to a specific device ordinal. Falls back to
    /// "no device" (rather than erroring) if that ordinal is not available,
    /// mirroring [`GpuBackend::with_device_id`]'s honest-`None` contract.
    pub fn with_device_id(device_id: usize) -> SklResult<Self> {
        Ok(Self {
            backend: GpuBackend::with_device_id(device_id)?,
        })
    }

    /// Lists all real devices visible to the OxiCUDA driver on this host, via
    /// `oxicuda-driver` device enumeration. Returns an empty vector (not an
    /// error) when no driver/GPU is present.
    pub fn devices() -> SklResult<Vec<GpuDevice>> {
        (0..GpuUtils::device_count())
            .map(GpuUtils::device_properties)
            .collect()
    }

    /// Rebinds this context to a different device ordinal.
    pub fn set_device(&mut self, device_id: usize) -> SklResult<()> {
        match GpuBackend::with_device_id(device_id)? {
            Some(backend) => {
                self.backend = Some(backend);
                Ok(())
            }
            None => Err(SklearsError::InvalidInput(format!(
                "GPU device {device_id} is not available"
            ))),
        }
    }

    /// Real, queried properties of the currently bound device, if any.
    pub fn current_device(&self) -> Option<GpuDevice> {
        self.backend
            .as_ref()
            .and_then(|b| GpuUtils::device_properties(b.device_id()).ok())
    }

    /// Whether a real GPU backend is bound to this context.
    pub fn is_gpu_available(&self) -> bool {
        self.backend.is_some()
    }

    /// The underlying real backend, if a device was detected.
    pub fn backend(&self) -> Option<&GpuBackend> {
        self.backend.as_ref()
    }

    /// Allocates a GPU buffer on this context's backend.
    ///
    /// # Errors
    ///
    /// Returns [`SklearsError::InvalidOperation`] if no GPU backend is bound
    /// to this context (call [`is_gpu_available`](Self::is_gpu_available)
    /// first to check, or use the CPU estimators directly).
    pub fn allocate_buffer<T: Copy>(&self, size: usize) -> SklResult<GpuBuffer<T>> {
        let backend = self.backend.as_ref().ok_or_else(|| {
            SklearsError::InvalidOperation("no GPU backend available in this context".to_string())
        })?;
        GpuBuffer::zeros(backend, size)
    }
}

/// Configuration for GPU-accelerated explanation computation.
#[derive(Debug, Clone)]
pub struct GpuConfig {
    /// Preferred device ordinal (None for auto-detection via
    /// [`GpuBackend::detect`]'s "most free memory" heuristic).
    pub preferred_device_id: Option<usize>,
    /// Batch size for GPU computation.
    pub batch_size: usize,
    /// Number of streams for async computation.
    pub num_streams: usize,
    /// Enable memory pinning for faster transfers.
    pub pin_memory: bool,
    /// Fallback to CPU if GPU computation fails or no GPU is available.
    pub cpu_fallback: bool,
}

impl Default for GpuConfig {
    fn default() -> Self {
        Self {
            preferred_device_id: None,
            batch_size: 1024,
            num_streams: 4,
            pin_memory: true,
            cpu_fallback: true,
        }
    }
}

/// GPU-accelerated explanation computer.
///
/// Holds an `Option<GpuBackend>` (cheap to clone: two `Arc` bumps, see
/// [`GpuBackend`]'s docs) rather than re-detecting per call. When no backend
/// is present, every method here falls back to the CPU reference estimator
/// with no behavioral difference beyond speed.
pub struct GpuExplanationComputer {
    backend: Option<GpuBackend>,
    #[allow(dead_code)] // retained for future GPU-staged batch sizing
    config: GpuConfig,
}

impl GpuExplanationComputer {
    /// Creates a new GPU explanation computer bound to `context`'s backend.
    pub fn new(context: &GpuContext) -> SklResult<Self> {
        Ok(Self {
            backend: context.backend().cloned(),
            config: GpuConfig::default(),
        })
    }

    /// Creates a new GPU explanation computer with custom configuration.
    pub fn with_config(context: &GpuContext, config: GpuConfig) -> SklResult<Self> {
        Ok(Self {
            backend: context.backend().cloned(),
            config,
        })
    }

    /// Whether this computer has a real GPU backend to dispatch to.
    pub fn is_gpu_available(&self) -> bool {
        self.backend.is_some()
    }

    /// Compute SHAP values, staging the perturbation batches on the GPU when
    /// a backend is available.
    ///
    /// `predict_fn` is a host-resident closure (the model itself is not
    /// necessarily GPU-resident), so full on-device evaluation is not
    /// possible in the general case; only the perturbed-batch construction
    /// is GPU-staging-eligible. See `TODO.md`'s OxiCUDA Migration section for
    /// the (deferred) plan to add a batched-linear-model fast path via
    /// `oxicuda-blas` GEMM. For now this always runs the CPU reference
    /// estimator, which is correct (not simulated) regardless of GPU
    /// availability.
    pub async fn compute_shap_parallel<F>(
        &self,
        features: &ArrayView2<'_, Float>,
        background: &ArrayView2<'_, Float>,
        predict_fn: F,
    ) -> SklResult<Array2<Float>>
    where
        F: Fn(&ArrayView2<'_, Float>) -> Array1<Float> + Send + Sync + 'static,
    {
        self.compute_shap_cpu(features, background, predict_fn)
            .await
    }

    /// CPU reference path for SHAP value computation.
    ///
    /// Computes interventional Shapley contributions via the marginal-contribution
    /// estimator: for each feature, the attribution is the change in the model
    /// output when that feature is restored from the explained instance versus
    /// drawn from the background distribution, averaged over all background rows.
    /// This mirrors the estimator in [`crate::parallel`], generalized to a
    /// multi-row background and batched into a single forward pass per feature.
    async fn compute_shap_cpu<F>(
        &self,
        features: &ArrayView2<'_, Float>,
        background: &ArrayView2<'_, Float>,
        predict_fn: F,
    ) -> SklResult<Array2<Float>>
    where
        F: Fn(&ArrayView2<'_, Float>) -> Array1<Float> + Send + Sync + 'static,
    {
        let n_samples = features.nrows();
        let n_features = features.ncols();
        let n_background = background.nrows();

        if n_background == 0 {
            return Err(SklearsError::InvalidInput(
                "SHAP CPU computation requires a non-empty background dataset".to_string(),
            ));
        }
        if background.ncols() != n_features {
            return Err(SklearsError::InvalidInput(format!(
                "background feature dimension {} does not match instance feature dimension {}",
                background.ncols(),
                n_features
            )));
        }

        let mut shap = Array2::<Float>::zeros((n_samples, n_features));

        for s in 0..n_samples {
            let x = features.row(s);

            // Full-coalition prediction f(x); identical across features.
            let mut x_row = Array2::<Float>::zeros((1, n_features));
            x_row.row_mut(0).assign(&x);
            let full_pred = predict_fn(&x_row.view())[0];

            for f in 0..n_features {
                // Replace feature f of x with each background reference and run a
                // single batched forward pass to obtain f(x with feature f absent).
                let mut perturbed = Array2::<Float>::zeros((n_background, n_features));
                for b in 0..n_background {
                    perturbed.row_mut(b).assign(&x);
                    perturbed[[b, f]] = background[[b, f]];
                }
                let preds = predict_fn(&perturbed.view());
                let mean_without = preds.sum() / n_background as Float;
                shap[[s, f]] = full_pred - mean_without;
            }
        }

        Ok(shap)
    }

    /// Compute permutation importance.
    ///
    /// As with [`compute_shap_parallel`](Self::compute_shap_parallel),
    /// `predict_fn` is host-resident, so this always runs the CPU reference
    /// estimator today; GPU-staging the shuffled feature batches is deferred
    /// (see `TODO.md`).
    pub async fn compute_permutation_importance<F>(
        &self,
        features: &ArrayView2<'_, Float>,
        targets: &Array1<Float>,
        predict_fn: F,
        n_permutations: usize,
    ) -> SklResult<Array1<Float>>
    where
        F: Fn(&ArrayView2<'_, Float>) -> Array1<Float> + Send + Sync + 'static,
    {
        self.compute_permutation_importance_cpu(features, targets, predict_fn, n_permutations)
            .await
    }

    /// CPU reference path for permutation importance computation.
    ///
    /// For each feature, its column is shuffled `n_permutations` times and the
    /// resulting rise in mean-squared error relative to the unshuffled baseline
    /// is averaged. A larger increase indicates a more important feature. The
    /// shuffle is driven by a seeded `ChaCha8Rng` for reproducibility, matching
    /// the estimator in [`crate::parallel`].
    async fn compute_permutation_importance_cpu<F>(
        &self,
        features: &ArrayView2<'_, Float>,
        targets: &Array1<Float>,
        predict_fn: F,
        n_permutations: usize,
    ) -> SklResult<Array1<Float>>
    where
        F: Fn(&ArrayView2<'_, Float>) -> Array1<Float> + Send + Sync + 'static,
    {
        use scirs2_core::random::{seq::SliceRandom, ChaCha8Rng, SeedableRng};

        let n_samples = features.nrows();
        let n_features = features.ncols();

        if n_samples == 0 || n_features == 0 {
            return Ok(Array1::zeros(n_features));
        }
        if targets.len() != n_samples {
            return Err(SklearsError::InvalidInput(format!(
                "targets length {} does not match number of samples {}",
                targets.len(),
                n_samples
            )));
        }
        let repeats = n_permutations.max(1);

        let mse = |pred: &Array1<Float>| -> Float {
            pred.iter()
                .zip(targets.iter())
                .map(|(p, t)| {
                    let d = p - t;
                    d * d
                })
                .sum::<Float>()
                / n_samples as Float
        };

        let baseline_error = mse(&predict_fn(features));

        let mut importances = Array1::<Float>::zeros(n_features);
        let mut rng = ChaCha8Rng::seed_from_u64(0);

        for f in 0..n_features {
            let mut accumulated = 0.0;
            for _ in 0..repeats {
                let mut perturbed: Array2<Float> = features.to_owned();
                let original: Vec<Float> = perturbed.column(f).to_vec();
                let mut order: Vec<usize> = (0..n_samples).collect();
                order.shuffle(&mut rng);
                {
                    let mut col = perturbed.column_mut(f);
                    for (i, &src) in order.iter().enumerate() {
                        col[i] = original[src];
                    }
                }
                let permuted_error = mse(&predict_fn(&perturbed.view()));
                accumulated += permuted_error - baseline_error;
            }
            importances[f] = accumulated / repeats as Float;
        }

        Ok(importances)
    }
}

/// Performance statistics for GPU computation
#[derive(Debug, Clone)]
pub struct GpuPerformanceStats {
    /// GPU computation time in milliseconds
    pub gpu_time_ms: f64,
    /// Data transfer time in milliseconds
    pub transfer_time_ms: f64,
    /// Total time including overhead
    pub total_time_ms: f64,
    /// Memory bandwidth utilized (GB/s)
    pub memory_bandwidth_gbps: f64,
    /// Compute utilization percentage
    pub compute_utilization_percent: f64,
}

/// Utility functions for GPU acceleration
pub mod utils {
    use super::*;

    /// `true` iff a real OxiCUDA-visible GPU is available on this host.
    pub fn is_gpu_available() -> bool {
        GpuUtils::is_gpu_available()
    }

    /// Number of OxiCUDA-visible GPU devices on this host.
    pub fn device_count() -> usize {
        GpuUtils::device_count()
    }

    /// Get optimal batch size for the given device's real reported memory.
    pub fn get_optimal_batch_size(device: &GpuDevice, data_size: usize) -> usize {
        // Simple heuristic based on device memory
        let denom = (data_size * std::mem::size_of::<Float>() * 4).max(1);
        let max_batch = device.total_memory / denom;
        std::cmp::min(max_batch, 1024).max(32)
    }

    /// Calculate memory requirements for explanation computation
    pub fn calculate_memory_requirements(
        n_samples: usize,
        n_features: usize,
        n_background: usize,
    ) -> usize {
        // Rough estimate of memory requirements in bytes
        let feature_memory = n_samples * n_features * std::mem::size_of::<Float>();
        let background_memory = n_background * n_features * std::mem::size_of::<Float>();
        let result_memory = n_samples * n_features * std::mem::size_of::<Float>();
        let workspace_memory = feature_memory * 2; // Temporary workspace

        feature_memory + background_memory + result_memory + workspace_memory
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_context_creation() {
        let result = GpuContext::new();
        assert!(result.is_ok());

        let context = result.expect("operation should succeed");
        // Honest detection: must agree with the underlying GpuUtils check,
        // never fabricate availability.
        assert_eq!(context.is_gpu_available(), GpuUtils::is_gpu_available());
    }

    #[test]
    fn test_gpu_config_default() {
        let config = GpuConfig::default();
        assert_eq!(config.batch_size, 1024);
        assert_eq!(config.num_streams, 4);
        assert!(config.pin_memory);
        assert!(config.cpu_fallback);
        assert!(config.preferred_device_id.is_none());
    }

    /// Requires real hardware; gracefully skips on machines (like this
    /// crate's own dev/CI environment) where `detect()` legitimately finds
    /// nothing, matching the pattern used in `sklears_core::gpu`'s own tests.
    #[test]
    fn test_gpu_buffer_creation() {
        let Some(backend) = GpuBackend::detect().expect("detect() should not hard-error") else {
            eprintln!("skipping test_gpu_buffer_creation: no GPU detected");
            return;
        };

        let buffer = GpuBuffer::<f32>::zeros(&backend, 100).expect("zeros");
        assert_eq!(buffer.len(), 100);
        assert!(!buffer.is_empty());
    }

    #[test]
    fn test_backend_availability_check() {
        // Must never panic, regardless of whether a GPU is present.
        let _ = utils::is_gpu_available();
        let _ = utils::device_count();
    }

    #[test]
    fn test_optimal_batch_size_calculation() {
        // A manually constructed device-properties value is fine here: this
        // test exercises the pure batch-size heuristic, not detection.
        let device = GpuDevice {
            device_id: 0,
            name: "Test Device".to_string(),
            total_memory: 1024 * 1024 * 1024, // 1GB
            free_memory: 1024 * 1024 * 1024,
            compute_capability: (7, 5),
        };

        let batch_size = utils::get_optimal_batch_size(&device, 1000);
        assert!(batch_size >= 32);
        assert!(batch_size <= 1024);
    }

    #[test]
    fn test_memory_requirements_calculation() {
        let memory = utils::calculate_memory_requirements(1000, 10, 100);
        assert!(memory > 0);

        // Should scale with problem size
        let larger_memory = utils::calculate_memory_requirements(2000, 20, 200);
        assert!(larger_memory > memory);
    }

    #[tokio::test]
    async fn test_gpu_explanation_computer_creation() {
        let context = GpuContext::new().expect("operation should succeed");
        let computer_result = GpuExplanationComputer::new(&context);
        assert!(computer_result.is_ok());

        let computer = computer_result.expect("operation should succeed");
        // Should work even without GPU (fallback to CPU)
        assert_eq!(computer.is_gpu_available(), context.is_gpu_available());
    }

    #[tokio::test]
    async fn test_shap_computation_fallback() {
        use scirs2_core::ndarray::array;

        let context = GpuContext::new().expect("operation should succeed");
        let computer = GpuExplanationComputer::new(&context).expect("operation should succeed");

        let features = array![[1.0, 2.0], [3.0, 4.0]];
        let background = array![[0.0, 0.0], [1.0, 1.0]];
        let predict_fn = |x: &ArrayView2<Float>| -> Array1<Float> {
            x.rows()
                .into_iter()
                .map(|row| row.iter().sum())
                .collect::<Vec<_>>()
                .into()
        };

        let result = computer
            .compute_shap_parallel(&features.view(), &background.view(), predict_fn)
            .await;

        assert!(result.is_ok());
        let shap_values = result.expect("operation should succeed");
        assert_eq!(shap_values.dim(), (2, 2));
    }
}
