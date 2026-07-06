//! GPU acceleration for ensemble methods, backed by the `oxicuda` crate family.
//!
//! # Honesty pass (v0.2.0)
//!
//! This module used to simulate a GPU: `detect_cuda_device` et al. always
//! returned `NotImplemented`, `is_cuda_available`/`is_opencl_available` were
//! hardcoded `false`, and `GpuMemoryManager` tracked a `Vec` of fake
//! `usize` "pointers" that never referred to any real memory. Behind the new
//! `gpu` feature, CUDA detection and memory allocation now go through real
//! `oxicuda-driver` / `oxicuda-memory` / `oxicuda-blas` calls via
//! [`sklears_core::gpu`]. Without the `gpu` feature (the default), this
//! module compiles to the same honest "no GPU here" behaviour it always
//! had for CUDA, and OpenCL/Metal/Vulkan remain permanently unsupported --
//! `oxicuda` only targets CUDA, so pretending to detect those backends
//! would just be a different flavour of the same dishonesty this pass is
//! removing.
//!
//! The previous `GpuKernel` trait and its four implementors
//! (`HistogramKernel`, `SplitFindingKernel`, `TreeUpdateKernel`,
//! `PredictionKernel`) have been removed rather than reimplemented:
//! every `execute` body only ever returned `NotImplemented`, so
//! `GpuEnsembleTrainer::train_gradient_boosting` was guaranteed to fail for
//! any backend that was not `CpuFallback` -- there was no working code
//! there to preserve. A real on-device histogram-based gradient-boosting
//! trainer would need custom PTX kernels (`oxicuda-ptx`/`oxicuda-launch`),
//! which is out of scope for this pass (deferred 2026-07-06: implementing
//! GPU histogram/split-finding/tree-update kernels is a substantial new
//! kernel-authoring project, not a rewire of existing logic; CPU-side
//! gradient boosting training already exists in
//! [`crate::gradient_boosting`]). [`GpuEnsembleTrainer`] now only offers
//! GPU-accelerated bulk linear-model inference via [`GpuTensorOps::matmul`].

use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::error::{Result, SklearsError};
use sklears_core::types::{Float, Int};
use std::sync::Arc;

#[cfg(feature = "gpu")]
use sklears_core::gpu::{GpuArray, GpuBackend as CoreGpuBackend, GpuMatrixOps};

/// GPU backend enumeration.
///
/// Only [`GpuBackend::Cuda`] can ever resolve to a real device (via
/// `oxicuda-driver`, and only when this crate is built with the `gpu`
/// feature). [`GpuBackend::OpenCL`], [`GpuBackend::Metal`], and
/// [`GpuBackend::Vulkan`] are kept as enum variants for API stability but
/// are permanently unsupported: `oxicuda` is CUDA-only, so selecting them
/// always yields an honest error rather than a fake device.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GpuBackend {
    /// CUDA backend for NVIDIA GPUs, via `oxicuda-driver`/`oxicuda-blas`
    /// (behind the `gpu` feature).
    Cuda,
    /// OpenCL backend -- unsupported by `oxicuda`; always errors.
    OpenCL,
    /// Metal backend -- unsupported by `oxicuda`; always errors.
    Metal,
    /// Vulkan backend -- unsupported by `oxicuda`; always errors.
    Vulkan,
    /// CPU fallback when no GPU is available or requested.
    CpuFallback,
}

/// GPU configuration
#[derive(Debug, Clone)]
pub struct GpuConfig {
    /// GPU backend to use
    pub backend: GpuBackend,
    /// Device ID (for systems with multiple GPUs)
    pub device_id: usize,
    /// Memory limit for GPU usage (in MB)
    pub memory_limit_mb: Option<usize>,
    /// Batch size for GPU operations
    pub batch_size: usize,
    /// Number of GPU streams for parallel execution
    pub n_streams: usize,
    /// Enable mixed precision (FP16/FP32)
    pub mixed_precision: bool,
    /// Enable tensor cores (for supported hardware)
    pub tensor_cores: bool,
    /// Memory pool size for efficient allocation
    pub memory_pool_size_mb: usize,
    /// Enable profiling for performance analysis
    pub enable_profiling: bool,
}

impl Default for GpuConfig {
    fn default() -> Self {
        Self {
            backend: GpuBackend::CpuFallback,
            device_id: 0,
            memory_limit_mb: None,
            batch_size: 1024,
            n_streams: 4,
            mixed_precision: false,
            tensor_cores: false,
            memory_pool_size_mb: 1024,
            enable_profiling: false,
        }
    }
}

/// GPU acceleration context.
///
/// Holds real `oxicuda` device state (behind the `gpu` feature, when a CUDA
/// device was actually found) rather than pretending one exists.
pub struct GpuContext {
    config: GpuConfig,
    device_info: GpuDeviceInfo,
    memory_manager: GpuMemoryManager,
    profiler: Option<GpuProfiler>,
    /// The real `oxicuda`-backed handle, present iff this context is bound
    /// to an actually-detected CUDA device. `None` for `CpuFallback` (and
    /// always `None` when the `gpu` feature is disabled).
    #[cfg(feature = "gpu")]
    core_backend: Option<CoreGpuBackend>,
}

/// GPU device information
#[derive(Debug, Clone)]
pub struct GpuDeviceInfo {
    /// Device name
    pub name: String,
    /// Total memory in MB
    pub total_memory_mb: usize,
    /// Available memory in MB
    pub available_memory_mb: usize,
    /// Number of compute units/cores
    pub compute_units: usize,
    /// Maximum work group size
    pub max_work_group_size: usize,
    /// Supports mixed precision
    pub supports_mixed_precision: bool,
    /// Supports tensor cores
    pub supports_tensor_cores: bool,
}

/// GPU memory manager.
///
/// Backed by real `oxicuda-memory` device allocations when bound to a real
/// CUDA backend (`gpu` feature, [`GpuBackend::Cuda`] detected); otherwise
/// (the `CpuFallback` case, which is the only reachable case without the
/// `gpu` feature) this honestly tracks handles as host-side bookkeeping --
/// there is no device to allocate real GPU memory on, so it does not
/// pretend to move any bytes.
pub struct GpuMemoryManager {
    allocated_bytes: usize,
    peak_allocated_bytes: usize,
    pool_size_bytes: usize,
    kind: MemoryManagerKind,
}

enum MemoryManagerKind {
    /// No real GPU backend bound: `blocks[handle].ptr` is a simulated
    /// handle with no corresponding device allocation.
    Host { blocks: Vec<GpuMemoryBlock> },
    /// Bound to a real `oxicuda` CUDA backend: `blocks[handle]` holds an
    /// actual on-device `DeviceBuffer<u8>` allocation (or `None` once
    /// freed).
    #[cfg(feature = "gpu")]
    Device {
        backend: CoreGpuBackend,
        blocks: Vec<Option<oxicuda_memory::DeviceBuffer<u8>>>,
    },
}

/// GPU memory block (host-side bookkeeping variant only; see
/// [`MemoryManagerKind::Host`]).
#[derive(Debug)]
pub struct GpuMemoryBlock {
    pub ptr: usize,
    pub size_bytes: usize,
    pub in_use: bool,
}

/// GPU profiler for performance analysis
pub struct GpuProfiler {
    enabled: bool,
    kernel_times: Vec<(String, f64)>,
    memory_transfers: Vec<(String, usize, f64)>,
}

/// GPU tensor operations
pub struct GpuTensorOps {
    // Read via `self.context.core_backend()` only under the `gpu` feature
    // (see `matmul`/`elementwise_add`); unused field without it.
    #[allow(dead_code)]
    context: Arc<GpuContext>,
}

/// GPU-accelerated ensemble trainer.
///
/// Training of gradient-boosted trees happens on the CPU (see
/// [`crate::gradient_boosting::GradientBoostingClassifier`] /
/// [`crate::gradient_boosting::GradientBoostingRegressor`]); see this
/// module's top-level doc comment for why the previous on-device
/// histogram/split-finding/tree-update kernel plumbing was removed instead
/// of reimplemented. This type's remaining responsibility is
/// GPU-accelerated bulk linear-model inference via [`GpuTensorOps`].
pub struct GpuEnsembleTrainer {
    context: Arc<GpuContext>,
    tensor_ops: GpuTensorOps,
}

impl GpuContext {
    /// Create new GPU context
    pub fn new(config: GpuConfig) -> Result<Self> {
        #[cfg(feature = "gpu")]
        {
            let (device_info, core_backend) = Self::detect_device(&config)?;
            let memory_manager = match &core_backend {
                Some(backend) => GpuMemoryManager::new_with_backend(
                    config.memory_pool_size_mb * 1024 * 1024,
                    backend.clone(),
                ),
                None => GpuMemoryManager::new(config.memory_pool_size_mb * 1024 * 1024),
            };
            let profiler = if config.enable_profiling {
                Some(GpuProfiler::new())
            } else {
                None
            };

            Ok(Self {
                config,
                device_info,
                memory_manager,
                profiler,
                core_backend,
            })
        }
        #[cfg(not(feature = "gpu"))]
        {
            let device_info = Self::detect_device(&config)?;
            let memory_manager = GpuMemoryManager::new(config.memory_pool_size_mb * 1024 * 1024);
            let profiler = if config.enable_profiling {
                Some(GpuProfiler::new())
            } else {
                None
            };

            Ok(Self {
                config,
                device_info,
                memory_manager,
                profiler,
            })
        }
    }

    /// Detect and initialize GPU device, also returning the live
    /// `oxicuda` backend handle when one was actually found (`gpu` feature
    /// only).
    #[cfg(feature = "gpu")]
    fn detect_device(config: &GpuConfig) -> Result<(GpuDeviceInfo, Option<CoreGpuBackend>)> {
        match config.backend {
            GpuBackend::Cuda => Self::detect_cuda_device(config.device_id),
            GpuBackend::OpenCL => Err(Self::unsupported_backend_err("OpenCL")),
            GpuBackend::Metal => Err(Self::unsupported_backend_err("Metal")),
            GpuBackend::Vulkan => Err(Self::unsupported_backend_err("Vulkan")),
            GpuBackend::CpuFallback => Ok((Self::create_cpu_fallback_info(), None)),
        }
    }

    /// Detect and initialize GPU device (non-`gpu`-feature build: CUDA is
    /// compiled out, so only `CpuFallback` can ever succeed).
    #[cfg(not(feature = "gpu"))]
    fn detect_device(config: &GpuConfig) -> Result<GpuDeviceInfo> {
        match config.backend {
            GpuBackend::Cuda => Self::detect_cuda_device(config.device_id),
            GpuBackend::OpenCL => Err(Self::unsupported_backend_err("OpenCL")),
            GpuBackend::Metal => Err(Self::unsupported_backend_err("Metal")),
            GpuBackend::Vulkan => Err(Self::unsupported_backend_err("Vulkan")),
            GpuBackend::CpuFallback => Ok(Self::create_cpu_fallback_info()),
        }
    }

    /// OpenCL/Metal/Vulkan are not, and cannot currently be, backed by
    /// `oxicuda` (a CUDA-only crate family), regardless of whether this
    /// crate is built with the `gpu` feature.
    fn unsupported_backend_err(name: &str) -> SklearsError {
        SklearsError::NotImplemented(format!(
            "{name} is not supported by the oxicuda backend (CUDA-only); \
             select GpuBackend::Cuda or GpuBackend::CpuFallback instead"
        ))
    }

    /// Detect a real CUDA device via `oxicuda-driver`, when built with the
    /// `gpu` feature.
    #[cfg(feature = "gpu")]
    fn detect_cuda_device(device_id: usize) -> Result<(GpuDeviceInfo, Option<CoreGpuBackend>)> {
        let Some(backend) = CoreGpuBackend::with_device_id(device_id)? else {
            return Err(SklearsError::NotImplemented(format!(
                "no CUDA device available at ordinal {device_id} (oxicuda-driver reported none)"
            )));
        };
        let device = backend.context().device();
        let name = device
            .name()
            .unwrap_or_else(|_| "Unknown CUDA device".to_string());
        let mem_info = backend.memory_info()?;
        let compute_units = device
            .multiprocessor_count()
            .map(|n| n.max(0) as usize)
            .unwrap_or(0);
        let max_work_group_size = device
            .max_threads_per_block()
            .map(|n| n.max(0) as usize)
            .unwrap_or(0);
        // Tensor cores / mixed-precision matmul first shipped with Volta
        // (compute capability 7.0).
        let supports_tensor_cores = device
            .compute_capability()
            .map(|(major, _)| major >= 7)
            .unwrap_or(false);

        Ok((
            GpuDeviceInfo {
                name,
                total_memory_mb: mem_info.total / (1024 * 1024),
                available_memory_mb: mem_info.free / (1024 * 1024),
                compute_units,
                max_work_group_size,
                supports_mixed_precision: supports_tensor_cores,
                supports_tensor_cores,
            },
            Some(backend),
        ))
    }

    /// CUDA detection without the `gpu` feature: the driver crates are not
    /// even compiled in, so this always honestly reports "not available"
    /// rather than pretending to probe for a device.
    #[cfg(not(feature = "gpu"))]
    fn detect_cuda_device(_device_id: usize) -> Result<GpuDeviceInfo> {
        Err(SklearsError::NotImplemented(
            "CUDA support requires building sklears-ensemble with the `gpu` feature".to_string(),
        ))
    }

    /// Create CPU fallback device info (reports real system RAM where available)
    fn create_cpu_fallback_info() -> GpuDeviceInfo {
        let (total_mb, available_mb) = Self::read_system_ram_mb();
        GpuDeviceInfo {
            name: "CPU Fallback (no GPU runtime)".to_string(),
            total_memory_mb: total_mb,
            available_memory_mb: available_mb,
            compute_units: std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1),
            max_work_group_size: 1,
            supports_mixed_precision: false,
            supports_tensor_cores: false,
        }
    }

    /// Read system RAM in MB. Returns (0, 0) if the OS API fails (honest unknown).
    fn read_system_ram_mb() -> (usize, usize) {
        #[cfg(target_os = "linux")]
        {
            if let Ok(content) = std::fs::read_to_string("/proc/meminfo") {
                let mut total_kb: Option<u64> = None;
                let mut avail_kb: Option<u64> = None;
                for line in content.lines() {
                    if let Some(rest) = line.strip_prefix("MemTotal:") {
                        total_kb = rest
                            .split_whitespace()
                            .next()
                            .and_then(|s| s.parse::<u64>().ok());
                    } else if let Some(rest) = line.strip_prefix("MemAvailable:") {
                        avail_kb = rest
                            .split_whitespace()
                            .next()
                            .and_then(|s| s.parse::<u64>().ok());
                    }
                    if total_kb.is_some() && avail_kb.is_some() {
                        break;
                    }
                }
                if let (Some(t), Some(a)) = (total_kb, avail_kb) {
                    return ((t / 1024) as usize, (a / 1024) as usize);
                }
            }
            (0, 0)
        }
        #[cfg(not(target_os = "linux"))]
        {
            (0, 0)
        }
    }

    /// Check if GPU is available
    pub fn is_gpu_available(&self) -> bool {
        #[cfg(feature = "gpu")]
        {
            self.core_backend.is_some()
        }
        #[cfg(not(feature = "gpu"))]
        {
            self.config.backend != GpuBackend::CpuFallback
        }
    }

    /// Get device information
    pub fn device_info(&self) -> &GpuDeviceInfo {
        &self.device_info
    }

    /// The configuration this context was built from.
    pub fn config(&self) -> &GpuConfig {
        &self.config
    }

    /// The real `oxicuda` backend bound to this context, if any (`gpu`
    /// feature only; `None` for `CpuFallback`).
    #[cfg(feature = "gpu")]
    pub fn core_backend(&self) -> Option<&CoreGpuBackend> {
        self.core_backend.as_ref()
    }

    /// Allocate GPU memory
    pub fn allocate_memory(&mut self, size_bytes: usize) -> Result<usize> {
        self.memory_manager.allocate(size_bytes)
    }

    /// Free GPU memory
    pub fn free_memory(&mut self, ptr: usize) -> Result<()> {
        self.memory_manager.free(ptr)
    }

    /// Start profiling
    pub fn start_profiling(&mut self) {
        if let Some(ref mut profiler) = self.profiler {
            profiler.start();
        }
    }

    /// Stop profiling and get results
    pub fn stop_profiling(&mut self) -> Option<ProfilingResults> {
        self.profiler.as_mut().map(|p| p.stop())
    }
}

impl GpuMemoryManager {
    /// Create a new host-side-only memory manager (no real GPU backend
    /// bound). Used for `CpuFallback`, and unconditionally when the `gpu`
    /// feature is disabled.
    pub fn new(pool_size_bytes: usize) -> Self {
        Self {
            allocated_bytes: 0,
            peak_allocated_bytes: 0,
            pool_size_bytes,
            kind: MemoryManagerKind::Host { blocks: Vec::new() },
        }
    }

    /// Create a new memory manager bound to a real `oxicuda` CUDA backend:
    /// [`allocate`](Self::allocate) performs real on-device allocations via
    /// `oxicuda-memory::DeviceBuffer`.
    #[cfg(feature = "gpu")]
    pub fn new_with_backend(pool_size_bytes: usize, backend: CoreGpuBackend) -> Self {
        Self {
            allocated_bytes: 0,
            peak_allocated_bytes: 0,
            pool_size_bytes,
            kind: MemoryManagerKind::Device {
                backend,
                blocks: Vec::new(),
            },
        }
    }

    /// Allocate memory block
    pub fn allocate(&mut self, size_bytes: usize) -> Result<usize> {
        if self.allocated_bytes + size_bytes > self.pool_size_bytes {
            return Err(SklearsError::InvalidInput(
                "GPU memory allocation failed: out of memory".to_string(),
            ));
        }

        let handle = match &mut self.kind {
            MemoryManagerKind::Host { blocks } => {
                // Find suitable free block or allocate new one.
                let mut reused = None;
                for block in blocks.iter_mut() {
                    if !block.in_use && block.size_bytes >= size_bytes {
                        block.in_use = true;
                        reused = Some(block.ptr);
                        break;
                    }
                }
                match reused {
                    Some(ptr) => ptr,
                    None => {
                        let ptr = blocks.len();
                        blocks.push(GpuMemoryBlock {
                            ptr,
                            size_bytes,
                            in_use: true,
                        });
                        ptr
                    }
                }
            }
            #[cfg(feature = "gpu")]
            MemoryManagerKind::Device { backend, blocks } => {
                backend
                    .context()
                    .set_current()
                    .map_err(|e| SklearsError::NumericalError(e.to_string()))?;
                let buf = oxicuda_memory::DeviceBuffer::<u8>::alloc(size_bytes)
                    .map_err(|e| SklearsError::NumericalError(e.to_string()))?;
                let handle = blocks.len();
                blocks.push(Some(buf));
                handle
            }
        };

        self.allocated_bytes += size_bytes;
        self.peak_allocated_bytes = self.peak_allocated_bytes.max(self.allocated_bytes);
        Ok(handle)
    }

    /// Free memory block
    pub fn free(&mut self, ptr: usize) -> Result<()> {
        match &mut self.kind {
            MemoryManagerKind::Host { blocks } => {
                if let Some(block) = blocks.get_mut(ptr) {
                    if block.in_use {
                        block.in_use = false;
                        self.allocated_bytes =
                            self.allocated_bytes.saturating_sub(block.size_bytes);
                        Ok(())
                    } else {
                        Err(SklearsError::InvalidInput(
                            "Attempted to free already freed memory".to_string(),
                        ))
                    }
                } else {
                    Err(SklearsError::InvalidInput(
                        "Invalid memory pointer".to_string(),
                    ))
                }
            }
            #[cfg(feature = "gpu")]
            MemoryManagerKind::Device { blocks, .. } => {
                if let Some(slot) = blocks.get_mut(ptr) {
                    if let Some(buf) = slot.take() {
                        self.allocated_bytes = self.allocated_bytes.saturating_sub(buf.len());
                        // `buf` is dropped here, freeing the real device
                        // allocation via `oxicuda-memory`'s `Drop` impl.
                        Ok(())
                    } else {
                        Err(SklearsError::InvalidInput(
                            "Attempted to free already freed memory".to_string(),
                        ))
                    }
                } else {
                    Err(SklearsError::InvalidInput(
                        "Invalid memory pointer".to_string(),
                    ))
                }
            }
        }
    }

    /// Get memory usage statistics
    pub fn memory_stats(&self) -> (usize, usize, usize) {
        (
            self.allocated_bytes,
            self.peak_allocated_bytes,
            self.pool_size_bytes,
        )
    }
}

impl Default for GpuProfiler {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuProfiler {
    /// Create new profiler
    pub fn new() -> Self {
        Self {
            enabled: false,
            kernel_times: Vec::new(),
            memory_transfers: Vec::new(),
        }
    }

    /// Start profiling
    pub fn start(&mut self) {
        self.enabled = true;
        self.kernel_times.clear();
        self.memory_transfers.clear();
    }

    /// Stop profiling and return results
    pub fn stop(&mut self) -> ProfilingResults {
        self.enabled = false;
        ProfilingResults {
            kernel_times: self.kernel_times.clone(),
            memory_transfers: self.memory_transfers.clone(),
            total_kernel_time: self.kernel_times.iter().map(|(_, t)| t).sum(),
            total_memory_transfer_time: self.memory_transfers.iter().map(|(_, _, t)| t).sum(),
        }
    }

    /// Record kernel execution time
    pub fn record_kernel(&mut self, name: String, time_ms: f64) {
        if self.enabled {
            self.kernel_times.push((name, time_ms));
        }
    }

    /// Record memory transfer
    pub fn record_memory_transfer(&mut self, name: String, bytes: usize, time_ms: f64) {
        if self.enabled {
            self.memory_transfers.push((name, bytes, time_ms));
        }
    }
}

/// Profiling results
#[derive(Debug, Clone)]
pub struct ProfilingResults {
    pub kernel_times: Vec<(String, f64)>,
    pub memory_transfers: Vec<(String, usize, f64)>,
    pub total_kernel_time: f64,
    pub total_memory_transfer_time: f64,
}

impl GpuTensorOps {
    /// Create new GPU tensor operations
    pub fn new(context: Arc<GpuContext>) -> Self {
        Self { context }
    }

    /// Matrix multiplication. Dispatches to `oxicuda-blas` GEMM via
    /// [`sklears_core::gpu::GpuMatrixOps`] when this context is bound to a
    /// real CUDA backend (`gpu` feature); falls back to `ndarray`'s CPU
    /// `dot` otherwise.
    pub fn matmul(&self, a: &Array2<Float>, b: &Array2<Float>) -> Result<Array2<Float>> {
        #[cfg(feature = "gpu")]
        {
            if let Some(backend) = self.context.core_backend() {
                let ga = GpuArray::from_array2(backend, a)?;
                let gb = GpuArray::from_array2(backend, b)?;
                let gc = ga.matmul(&gb)?;
                return gc.to_array2();
            }
        }
        Ok(a.dot(b))
    }

    /// Element-wise addition. Dispatches to `oxicuda-blas`'s elementwise
    /// `add` kernel via [`sklears_core::gpu::GpuMatrixOps`] when bound to a
    /// real CUDA backend; falls back to CPU addition otherwise.
    pub fn elementwise_add(&self, a: &Array2<Float>, b: &Array2<Float>) -> Result<Array2<Float>> {
        #[cfg(feature = "gpu")]
        {
            if let Some(backend) = self.context.core_backend() {
                let ga = GpuArray::from_array2(backend, a)?;
                let gb = GpuArray::from_array2(backend, b)?;
                let gc = ga.add(&gb)?;
                return gc.to_array2();
            }
        }
        Ok(a + b)
    }

    /// Reduction (sum) along an axis, or a full reduction if `axis` is
    /// `None`.
    ///
    /// Always runs on the CPU: as of `oxicuda-blas` 0.4.0 there is no
    /// on-device reduction primitive exposed through
    /// [`sklears_core::gpu::GpuMatrixOps`], so there is nothing to
    /// honestly dispatch to a GPU here yet.
    pub fn reduce_sum(&self, array: &Array2<Float>, axis: Option<usize>) -> Result<Array1<Float>> {
        match axis {
            Some(ax) => Ok(array.sum_axis(scirs2_core::ndarray::Axis(ax))),
            None => Ok(Array1::from_elem(1, array.sum())),
        }
    }

    /// Row-wise softmax.
    ///
    /// Always runs on the CPU: this is a composite max/exp/sum operation
    /// with no matching single on-device primitive in
    /// [`sklears_core::gpu::GpuMatrixOps`] as of `oxicuda-blas` 0.4.0.
    pub fn softmax(&self, array: &Array2<Float>) -> Result<Array2<Float>> {
        let mut result = array.clone();

        for mut row in result.rows_mut() {
            let max_val = row.fold(Float::NEG_INFINITY, |a, &b| a.max(b));
            row.mapv_inplace(|x| (x - max_val).exp());
            let sum = row.sum();
            row /= sum;
        }

        Ok(result)
    }
}

impl GpuEnsembleTrainer {
    /// Create new GPU ensemble trainer
    pub fn new(config: GpuConfig) -> Result<Self> {
        let context = Arc::new(GpuContext::new(config)?);
        let tensor_ops = GpuTensorOps::new(context.clone());

        Ok(Self {
            context,
            tensor_ops,
        })
    }

    /// Predict using a GPU-accelerated batch matmul over an ensemble of
    /// linear models: stacks `models` into an `(n_features, n_models)`
    /// matrix, computes `x . models` in one GEMM call (real on-device GEMM
    /// when a CUDA backend is bound, CPU `dot` otherwise -- see
    /// [`GpuTensorOps::matmul`]), and majority-votes the sign of the
    /// per-model sum.
    pub fn predict_ensemble(
        &self,
        models: &[Array1<Float>],
        x: &Array2<Float>,
    ) -> Result<Array1<Int>> {
        if models.is_empty() {
            return Ok(Array1::zeros(x.nrows()));
        }

        let n_features = x.ncols();
        let mut models_matrix = Array2::<Float>::zeros((n_features, models.len()));
        for (j, model) in models.iter().enumerate() {
            if model.len() != n_features {
                return Err(SklearsError::ShapeMismatch {
                    expected: format!("model feature count = {n_features}"),
                    actual: format!("model[{j}].len() = {}", model.len()),
                });
            }
            models_matrix.column_mut(j).assign(model);
        }

        let per_model_scores = self.tensor_ops.matmul(x, &models_matrix)?;
        let sums = per_model_scores.sum_axis(scirs2_core::ndarray::Axis(1));
        Ok(sums.mapv(|s| if s > 0.0 { 1 } else { 0 }))
    }

    /// Get GPU context
    pub fn context(&self) -> &GpuContext {
        &self.context
    }

    /// Check GPU availability
    pub fn is_gpu_available(&self) -> bool {
        self.context.is_gpu_available()
    }
}

/// GPU backend detection: reports [`GpuBackend::Cuda`] when `oxicuda-driver`
/// successfully initialises and finds at least one device (requires the
/// `gpu` feature; always `false` -- honestly -- without it), plus
/// [`GpuBackend::CpuFallback`], which is always available.
pub fn detect_available_backends() -> Vec<GpuBackend> {
    let mut backends = Vec::new();

    if is_cuda_available() {
        backends.push(GpuBackend::Cuda);
    }

    // Always have CPU fallback
    backends.push(GpuBackend::CpuFallback);

    backends
}

/// Check CUDA availability via a real `oxicuda-driver` probe.
#[cfg(feature = "gpu")]
fn is_cuda_available() -> bool {
    CoreGpuBackend::is_available()
}

/// Without the `gpu` feature, the CUDA driver crates are not even compiled
/// in, so this honestly reports `false` rather than probing for anything.
#[cfg(not(feature = "gpu"))]
fn is_cuda_available() -> bool {
    false
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_gpu_config_default() {
        let config = GpuConfig::default();
        assert_eq!(config.backend, GpuBackend::CpuFallback);
        assert_eq!(config.device_id, 0);
        assert_eq!(config.batch_size, 1024);
    }

    #[test]
    fn test_gpu_context_creation() {
        let config = GpuConfig::default();
        let context = GpuContext::new(config).expect("operation should succeed");
        assert!(!context.is_gpu_available()); // Should be CPU fallback
    }

    #[test]
    fn test_memory_manager() {
        let mut manager = GpuMemoryManager::new(1024 * 1024); // 1MB

        let ptr1 = manager.allocate(1024).expect("operation should succeed");
        let ptr2 = manager.allocate(2048).expect("operation should succeed");

        assert_ne!(ptr1, ptr2);

        manager.free(ptr1).expect("operation should succeed");
        manager.free(ptr2).expect("operation should succeed");

        let (allocated, _, total) = manager.memory_stats();
        assert_eq!(allocated, 0);
        assert_eq!(total, 1024 * 1024);
    }

    #[test]
    fn test_gpu_profiler() {
        let mut profiler = GpuProfiler::new();
        profiler.start();

        profiler.record_kernel("test_kernel".to_string(), 1.5);
        profiler.record_memory_transfer("test_transfer".to_string(), 1024, 0.5);

        let results = profiler.stop();
        assert_eq!(results.kernel_times.len(), 1);
        assert_eq!(results.memory_transfers.len(), 1);
        assert_eq!(results.total_kernel_time, 1.5);
    }

    #[test]
    fn test_gpu_tensor_ops() {
        let config = GpuConfig::default();
        let context = Arc::new(GpuContext::new(config).expect("operation should succeed"));
        let ops = GpuTensorOps::new(context);

        let a = array![[1.0, 2.0], [3.0, 4.0]];
        let b = array![[5.0, 6.0], [7.0, 8.0]];

        let result = ops
            .elementwise_add(&a, &b)
            .expect("operation should succeed");
        let expected = array![[6.0, 8.0], [10.0, 12.0]];

        assert_eq!(result, expected);
    }

    #[test]
    fn test_gpu_tensor_ops_matmul_cpu_fallback() {
        let config = GpuConfig::default();
        let context = Arc::new(GpuContext::new(config).expect("operation should succeed"));
        let ops = GpuTensorOps::new(context);

        let a = array![[1.0, 2.0], [3.0, 4.0]];
        let b = array![[5.0, 6.0], [7.0, 8.0]];

        let result = ops.matmul(&a, &b).expect("operation should succeed");
        let expected = a.dot(&b);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_available_backends() {
        let backends = detect_available_backends();
        assert!(!backends.is_empty());
        assert!(backends.contains(&GpuBackend::CpuFallback));
    }

    #[test]
    fn test_gpu_ensemble_trainer() {
        let config = GpuConfig::default();
        let trainer = GpuEnsembleTrainer::new(config).expect("operation should succeed");

        assert!(!trainer.is_gpu_available()); // Should be CPU fallback
        assert_eq!(
            trainer.context().device_info().name,
            "CPU Fallback (no GPU runtime)"
        );
    }

    #[test]
    fn test_gpu_ensemble_trainer_predict_ensemble_cpu_fallback() {
        let config = GpuConfig::default();
        let trainer = GpuEnsembleTrainer::new(config).expect("operation should succeed");

        let models = vec![array![1.0, -1.0], array![0.5, 0.5]];
        let x = array![[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]];

        let predictions = trainer
            .predict_ensemble(&models, &x)
            .expect("operation should succeed");
        assert_eq!(predictions.len(), 3);
    }
}
