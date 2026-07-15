//! Advanced GPU acceleration with multi-GPU support and performance profiling.
//!
//! Orchestrates work across multiple GPU devices (or CPU-reference backends
//! when real GPUs are absent). The core GEMM dispatch delegates to
//! `sklears_core::gpu::{GpuArray, GpuMatrixOps}` (backed directly by
//! `oxicuda-blas`) through the [`AdvancedGpuOps`] infrastructure when the
//! `gpu` feature is enabled.
//!
//! # Wave B1
//!
//! [`AdvancedGpuOps`] now caches a single `Option<GpuBackend>` (detected
//! once in [`AdvancedGpuOps::new`]) instead of calling the old infallible
//! `GpuContext::new()` fresh on every GEMM call; every GPU-dispatch site
//! extends its existing problem-size-threshold branch to also require that
//! cached backend, falling back to the CPU path whenever no GPU was
//! detected (see `sklears_core::gpu::GpuBackend::detect`).

use scirs2_core::ndarray::{s, Array2, Array3, ArrayView2};
use sklears_core::{error::Result, prelude::SklearsError, types::Float};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

#[cfg(feature = "gpu")]
use half::f16;
#[cfg(feature = "gpu")]
use oxicuda_blas::{BlasHandle, Layout, MatrixDesc, MatrixDescMut, Transpose};
#[cfg(feature = "gpu")]
use oxicuda_driver::{Event, Stream};
#[cfg(feature = "gpu")]
use oxicuda_memory::DeviceBuffer;
#[cfg(feature = "gpu")]
use sklears_core::gpu::{GpuArray, GpuBackend, GpuMatrixOps};

/// Maps any GPU-stack error (`oxicuda-driver`/`oxicuda-blas`) to a
/// `SklearsError`, without needing those crates as direct dependencies just
/// to name their error types.
#[cfg(feature = "gpu")]
fn gpu_err<E: std::fmt::Display>(e: E) -> SklearsError {
    SklearsError::NumericalError(format!("GPU error: {e}"))
}

/// Real effective memory bandwidth for a GEMM-family op: total bytes moved
/// (inputs + output, host<->device plus any intermediate host copies)
/// divided by wall-clock duration. This is what actually elapsed, not a
/// theoretical peak -- unlike `occupancy_percentage` (still `0.0`
/// everywhere in this module: computing real occupancy needs the launch's
/// actual grid/block dimensions, which are internal to
/// `oxicuda-blas::level3::gemm` and not surfaced to callers today).
fn memory_bandwidth_gbps(memory_used_bytes: usize, duration: Duration) -> f64 {
    let seconds = duration.as_secs_f64();
    if seconds <= 0.0 {
        return 0.0;
    }
    (memory_used_bytes as f64 / seconds) / 1e9
}

// ─── AdvancedGpuConfig ─────────────────────────────────────────────────────────

/// Advanced GPU configuration
#[derive(Debug, Clone)]
pub struct AdvancedGpuConfig {
    /// List of GPU device IDs to use
    pub device_ids: Vec<usize>,
    /// Memory pool size per GPU in bytes
    pub memory_pool_size_per_gpu: usize,
    /// Number of compute streams per GPU
    pub streams_per_gpu: usize,
    /// Enable mixed precision computation
    pub enable_mixed_precision: bool,
    /// Enable kernel fusion optimizations
    pub enable_kernel_fusion: bool,
    /// Enable performance profiling
    pub enable_profiling: bool,
    /// Load balancing strategy
    pub load_balancing: LoadBalancingStrategy,
    /// Minimum problem size for GPU acceleration
    pub min_problem_size: usize,
    /// Maximum GPU memory usage percentage
    pub max_memory_usage: f32,
}

impl Default for AdvancedGpuConfig {
    fn default() -> Self {
        Self {
            device_ids: vec![0],
            memory_pool_size_per_gpu: 1 << 30,
            streams_per_gpu: 4,
            enable_mixed_precision: true,
            enable_kernel_fusion: true,
            enable_profiling: false,
            load_balancing: LoadBalancingStrategy::RoundRobin,
            min_problem_size: 1000,
            max_memory_usage: 0.8,
        }
    }
}

// ─── LoadBalancingStrategy ─────────────────────────────────────────────────────

/// Load balancing strategies for multi-GPU operations
#[derive(Debug, Clone, Copy)]
pub enum LoadBalancingStrategy {
    RoundRobin,
    MemoryBased,
    ComputeCapabilityBased,
    Dynamic,
}

// ─── GpuDeviceInfo ─────────────────────────────────────────────────────────────

/// GPU device information
#[derive(Debug, Clone)]
pub struct GpuDeviceInfo {
    pub device_id: usize,
    pub name: String,
    pub memory_total: usize,
    pub memory_free: usize,
    pub compute_capability: (u32, u32),
    pub multiprocessor_count: u32,
    pub max_threads_per_block: u32,
    pub max_shared_memory_per_block: usize,
}

// ─── GpuPerformanceMetrics ─────────────────────────────────────────────────────

/// GPU performance metrics per operation
#[derive(Debug, Clone)]
pub struct GpuPerformanceMetrics {
    pub device_id: usize,
    pub operation_name: String,
    pub duration: Duration,
    pub memory_used: usize,
    pub throughput_gflops: f64,
    pub memory_bandwidth_gbps: f64,
    pub occupancy_percentage: f64,
}

// ─── GpuMemoryPool ─────────────────────────────────────────────────────────────

/// Memory pool with best-fit allocation and coalesced free-block merging,
/// backed by a real reserved device-memory arena when a GPU is available.
///
/// `allocate()`/`deallocate()` manage the same host-side (offset, size)
/// ledger as before; what changed is that construction (when the `gpu`
/// feature is enabled and [`GpuBackend::with_device_id`] finds this pool's
/// device) also reserves `size` bytes of genuine device memory as one
/// `DeviceBuffer<u8>` arena, so the ledger's capacity corresponds to memory
/// the driver actually allocated. See
/// [`is_device_backed`](Self::is_device_backed).
pub struct GpuMemoryPool {
    device_id: usize,
    total_size: usize,
    used_size: usize,
    free_blocks: Vec<(usize, usize)>,
    allocated_blocks: HashMap<usize, usize>,
    #[cfg(feature = "gpu")]
    device_arena: Option<DeviceBuffer<u8>>,
}

impl std::fmt::Debug for GpuMemoryPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuMemoryPool")
            .field("device_id", &self.device_id)
            .field("total_size", &self.total_size)
            .field("used_size", &self.used_size)
            .field("free_blocks", &self.free_blocks)
            .field("allocated_blocks", &self.allocated_blocks)
            .field("is_device_backed", &self.is_device_backed())
            .finish()
    }
}

impl GpuMemoryPool {
    pub fn new(device_id: usize, size: usize) -> Self {
        #[cfg(feature = "gpu")]
        let device_arena = Self::try_reserve_device_arena(device_id, size);
        Self {
            device_id,
            total_size: size,
            used_size: 0,
            free_blocks: vec![(0, size)],
            allocated_blocks: HashMap::new(),
            #[cfg(feature = "gpu")]
            device_arena,
        }
    }

    /// Attempts to reserve `size` bytes of real device memory on device
    /// `device_id`. Returns `None` (never an error) on any failure -- no GPU
    /// at that ordinal, context-current failure, or the allocation itself
    /// failing -- so construction always succeeds and callers transparently
    /// get host-side-only accounting in that case.
    #[cfg(feature = "gpu")]
    fn try_reserve_device_arena(device_id: usize, size: usize) -> Option<DeviceBuffer<u8>> {
        if size == 0 {
            return None;
        }
        let backend = match GpuBackend::with_device_id(device_id) {
            Ok(Some(backend)) => backend,
            Ok(None) => return None,
            Err(e) => {
                log::warn!(
                    "GpuMemoryPool(device {device_id}): backend detection failed ({e}); using host-side accounting only"
                );
                return None;
            }
        };
        if let Err(e) = backend.context().set_current() {
            log::warn!(
                "GpuMemoryPool(device {device_id}): failed to set device context ({e}); using host-side accounting only"
            );
            return None;
        }
        match DeviceBuffer::<u8>::alloc(size) {
            Ok(arena) => Some(arena),
            Err(e) => {
                log::warn!(
                    "GpuMemoryPool(device {device_id}): device arena reservation of {size} bytes failed ({e}); using host-side accounting only"
                );
                None
            }
        }
    }

    /// `true` when this pool's capacity is backed by a genuinely reserved
    /// device-memory arena, as opposed to host-side accounting only (no GPU
    /// at this ordinal, or reservation failed).
    pub fn is_device_backed(&self) -> bool {
        #[cfg(feature = "gpu")]
        {
            self.device_arena.is_some()
        }
        #[cfg(not(feature = "gpu"))]
        {
            false
        }
    }

    pub fn allocate(&mut self, size: usize) -> Result<usize> {
        let aligned_size = (size + 255) & !255;

        for (i, &(offset, block_size)) in self.free_blocks.iter().enumerate() {
            if block_size >= aligned_size {
                let ptr = offset;
                self.allocated_blocks.insert(ptr, aligned_size);
                self.used_size += aligned_size;

                if block_size > aligned_size {
                    self.free_blocks[i] = (offset + aligned_size, block_size - aligned_size);
                } else {
                    self.free_blocks.remove(i);
                }

                return Ok(ptr);
            }
        }

        Err(SklearsError::InvalidInput(format!(
            "Cannot allocate {} bytes on GPU {}",
            size, self.device_id
        )))
    }

    pub fn deallocate(&mut self, ptr: usize) -> Result<()> {
        if let Some(size) = self.allocated_blocks.remove(&ptr) {
            self.used_size -= size;

            self.free_blocks.push((ptr, size));
            self.free_blocks.sort_by_key(|&(offset, _)| offset);

            let mut i = 0;
            while i + 1 < self.free_blocks.len() {
                let (offset1, size1) = self.free_blocks[i];
                let (offset2, size2) = self.free_blocks[i + 1];
                if offset1 + size1 == offset2 {
                    self.free_blocks[i] = (offset1, size1 + size2);
                    self.free_blocks.remove(i + 1);
                } else {
                    i += 1;
                }
            }

            Ok(())
        } else {
            Err(SklearsError::InvalidInput(format!(
                "Invalid pointer {} for deallocation",
                ptr
            )))
        }
    }

    pub fn memory_usage(&self) -> (usize, usize) {
        (self.used_size, self.total_size)
    }

    pub fn fragmentation_ratio(&self) -> f64 {
        if self.free_blocks.is_empty() {
            0.0
        } else {
            let largest_free = self
                .free_blocks
                .iter()
                .map(|(_, size)| *size)
                .max()
                .unwrap_or(0);
            let total_free = self.total_size - self.used_size;
            if total_free == 0 {
                0.0
            } else {
                1.0 - (largest_free as f64 / total_free as f64)
            }
        }
    }
}

// ─── CudaStream ────────────────────────────────────────────────────────────────

/// Compute stream handle for asynchronous operation scheduling.
///
/// When the `gpu` feature is enabled and a real GPU backend was detected at
/// [`AdvancedGpuOps::new`] time, each `CudaStream` wraps its own genuine
/// `oxicuda-driver` [`Stream`] (via its own `oxicuda-blas` [`BlasHandle`]),
/// so GEMMs dispatched on it (see `CudaStream::launch_matmul`) truly run
/// on an independent stream rather than blocking the caller. Falls back to
/// bookkeeping-only (`is_busy` flag) when no GPU is present -- see
/// [`is_real`](Self::is_real).
pub struct CudaStream {
    #[allow(dead_code)]
    stream_id: usize,
    #[allow(dead_code)]
    device_id: usize,
    is_busy: bool,
    #[cfg(feature = "gpu")]
    blas: Option<BlasHandle>,
}

impl std::fmt::Debug for CudaStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaStream")
            .field("stream_id", &self.stream_id)
            .field("device_id", &self.device_id)
            .field("is_busy", &self.is_busy)
            .field("is_real", &self.is_real())
            .finish()
    }
}

impl CudaStream {
    /// Bookkeeping-only stream: no `gpu` feature, or no backend detected.
    pub fn new(device_id: usize, stream_id: usize) -> Self {
        Self {
            stream_id,
            device_id,
            is_busy: false,
            #[cfg(feature = "gpu")]
            blas: None,
        }
    }

    /// Creates a stream genuinely backed by its own `oxicuda-driver` stream
    /// and `oxicuda-blas` handle on `backend`'s context, falling back to
    /// bookkeeping-only when stream/handle creation itself fails (e.g.
    /// resource exhaustion).
    #[cfg(feature = "gpu")]
    fn with_backend(device_id: usize, stream_id: usize, backend: &GpuBackend) -> Self {
        let blas = Stream::new(backend.context())
            .ok()
            .and_then(|stream| BlasHandle::with_stream(backend.context(), stream).ok());
        Self {
            stream_id,
            device_id,
            is_busy: false,
            blas,
        }
    }

    pub fn is_available(&self) -> bool {
        !self.is_busy
    }

    /// `true` when this stream is backed by a real `oxicuda-driver` stream
    /// (rather than bookkeeping only).
    pub fn is_real(&self) -> bool {
        #[cfg(feature = "gpu")]
        {
            self.blas.is_some()
        }
        #[cfg(not(feature = "gpu"))]
        {
            false
        }
    }

    pub fn synchronize(&mut self) -> Result<()> {
        #[cfg(feature = "gpu")]
        if let Some(blas) = self.blas.as_ref() {
            blas.stream().synchronize().map_err(gpu_err)?;
        }
        self.is_busy = false;
        Ok(())
    }

    /// Launches `a x b` on this stream's own `oxicuda-blas` handle and
    /// returns a [`PendingGpuMatmul`] tracking its (genuinely asynchronous)
    /// completion. The device-to-host copy of the result is itself issued
    /// asynchronously (`copy_to_host_async`); callers poll
    /// [`PendingGpuMatmul::is_ready`] (a non-blocking `cuEventQuery`) rather
    /// than blocking here.
    ///
    /// # Errors
    ///
    /// Returns an error if this stream has no real backing (`is_real()` is
    /// `false`), the dimensions are incompatible, or any driver/BLAS call
    /// fails.
    #[cfg(feature = "gpu")]
    fn launch_matmul(&self, a: &Array2<Float>, b: &Array2<Float>) -> Result<PendingGpuMatmul> {
        let blas = self.blas.as_ref().ok_or_else(|| {
            SklearsError::InvalidInput("CudaStream has no real GPU backing".to_string())
        })?;

        let (m, k) = a.dim();
        let (k2, n) = b.dim();
        if k != k2 {
            return Err(SklearsError::InvalidInput(format!(
                "launch_matmul: inner dimensions differ ({k} vs {k2})"
            )));
        }

        blas.context().set_current().map_err(gpu_err)?;

        // Row-major logical iteration order, matching `RowMajor` below --
        // see the identical pattern in `fp16_matrix_multiply`.
        let a_host: Vec<Float> = a.iter().copied().collect();
        let b_host: Vec<Float> = b.iter().copied().collect();

        let a_buf = DeviceBuffer::from_host(&a_host).map_err(gpu_err)?;
        let b_buf = DeviceBuffer::from_host(&b_host).map_err(gpu_err)?;
        let mut c_buf = DeviceBuffer::<Float>::zeroed(m * n).map_err(gpu_err)?;

        let a_desc = MatrixDesc::from_buffer(&a_buf, m as u32, k as u32, Layout::RowMajor)
            .map_err(gpu_err)?;
        let b_desc = MatrixDesc::from_buffer(&b_buf, k as u32, n as u32, Layout::RowMajor)
            .map_err(gpu_err)?;
        let mut c_desc =
            MatrixDescMut::from_buffer(&mut c_buf, m as u32, n as u32, Layout::RowMajor)
                .map_err(gpu_err)?;

        oxicuda_blas::level3::gemm(
            blas,
            Transpose::NoTrans,
            Transpose::NoTrans,
            1.0,
            &a_desc,
            &b_desc,
            0.0,
            &mut c_desc,
        )
        .map_err(gpu_err)?;

        let mut host_result = vec![0.0; m * n];
        c_buf
            .copy_to_host_async(&mut host_result, blas.stream())
            .map_err(gpu_err)?;

        let event = Event::new().map_err(gpu_err)?;
        event.record(blas.stream()).map_err(gpu_err)?;

        Ok(PendingGpuMatmul {
            _a: a_buf,
            _b: b_buf,
            _c: c_buf,
            host_result,
            shape: (m, n),
            event,
        })
    }
}

/// A GEMM in flight on a real `oxicuda-driver` stream (see
/// [`CudaStream::launch_matmul`]).
///
/// The three device buffers are kept alive here (rather than dropped right
/// after the kernel launch) so the asynchronous kernel and device-to-host
/// copy always have valid memory to operate on until [`is_ready`](Self::is_ready)
/// (or [`into_result`](Self::into_result)) confirms completion.
#[cfg(feature = "gpu")]
struct PendingGpuMatmul {
    _a: DeviceBuffer<Float>,
    _b: DeviceBuffer<Float>,
    _c: DeviceBuffer<Float>,
    host_result: Vec<Float>,
    shape: (usize, usize),
    event: Event,
}

#[cfg(feature = "gpu")]
impl PendingGpuMatmul {
    /// Non-blocking poll (`cuEventQuery`) of whether the H2D-upload, GEMM
    /// kernel, and D2H-download pipeline has finished.
    fn is_ready(&self) -> Result<bool> {
        self.event.query().map_err(gpu_err)
    }

    /// Consumes this pending operation, producing the realized result.
    ///
    /// Callers must only call this once [`is_ready`](Self::is_ready) has
    /// returned `Ok(true)`; the host buffer is only guaranteed populated at
    /// that point.
    fn into_result(self) -> Result<Array2<Float>> {
        Array2::from_shape_vec(self.shape, self.host_result)
            .map_err(|e| SklearsError::InvalidOperation(format!("from_shape_vec: {e}")))
    }
}

// ─── AdvancedGpuOps ────────────────────────────────────────────────────────────

/// Multi-GPU operations manager with load balancing and performance profiling.
pub struct AdvancedGpuOps {
    config: AdvancedGpuConfig,
    devices: Vec<GpuDeviceInfo>,
    memory_pools: Vec<Arc<Mutex<GpuMemoryPool>>>,
    streams: Vec<Vec<CudaStream>>,
    performance_metrics: Vec<GpuPerformanceMetrics>,
    load_balancer: LoadBalancer,
    /// Detected once at construction time (see [`GpuBackend::detect`]);
    /// `None` when no GPU is present (or the `gpu` feature is disabled), in
    /// which case every GEMM dispatch below falls back to its CPU
    /// counterpart.
    #[cfg(feature = "gpu")]
    gpu_backend: Option<GpuBackend>,
}

impl AdvancedGpuOps {
    /// Create new advanced GPU operations manager.
    pub fn new(config: AdvancedGpuConfig) -> Result<Self> {
        // Detected up front (rather than per-device below) so that, when a
        // GPU is present, every device's streams can be constructed as real
        // `oxicuda-driver` streams against it -- see
        // [`CudaStream::with_backend`].
        #[cfg(feature = "gpu")]
        let gpu_backend = GpuBackend::detect()?;

        let mut devices = Vec::new();
        let mut memory_pools = Vec::new();
        let mut streams = Vec::new();

        for &device_id in &config.device_ids {
            let device_info = Self::get_device_info(device_id)?;
            devices.push(device_info);

            let pool = Arc::new(Mutex::new(GpuMemoryPool::new(
                device_id,
                config.memory_pool_size_per_gpu,
            )));
            memory_pools.push(pool);

            let device_streams: Vec<CudaStream> = (0..config.streams_per_gpu)
                .map(|i| {
                    #[cfg(feature = "gpu")]
                    if let Some(backend) = gpu_backend.as_ref() {
                        return CudaStream::with_backend(device_id, i, backend);
                    }
                    CudaStream::new(device_id, i)
                })
                .collect();
            streams.push(device_streams);
        }

        let load_balancer = LoadBalancer::new(config.load_balancing, &devices);

        Ok(Self {
            config,
            devices,
            memory_pools,
            streams,
            performance_metrics: Vec::new(),
            load_balancer,
            #[cfg(feature = "gpu")]
            gpu_backend,
        })
    }

    /// Query device information directly from `oxicuda-driver`'s attribute
    /// API, falling back to labelled placeholder defaults when no GPU is
    /// present (or the `gpu` feature is disabled).
    ///
    /// This queries `oxicuda_driver::Device` directly (rather than
    /// `sklears_core::gpu::GpuUtils::device_properties`, whose
    /// `GpuDeviceProperties` does not carry SM count / max threads per
    /// block / shared memory) so that `multiprocessor_count`,
    /// `max_threads_per_block`, and `max_shared_memory_per_block` below are
    /// real driver-reported attributes (`cuDeviceGetAttribute`) rather than
    /// hardcoded guesses, whenever a real GPU is available.
    fn get_device_info(device_id: usize) -> Result<GpuDeviceInfo> {
        #[cfg(feature = "gpu")]
        if let Some(info) = Self::real_device_info(device_id) {
            return Ok(info);
        }
        Ok(GpuDeviceInfo {
            device_id,
            name: format!("GPU Device {} (no GPU detected)", device_id),
            memory_total: 0,
            memory_free: 0,
            compute_capability: (0, 0),
            multiprocessor_count: 0,
            max_threads_per_block: 0,
            max_shared_memory_per_block: 0,
        })
    }

    /// Attempts to gather real `oxicuda-driver` attributes for `device_id`.
    /// Returns `None` on any failure (no driver, no such device, or the
    /// fundamental name/total-memory queries failing) rather than
    /// fabricating a value; [`get_device_info`](Self::get_device_info)
    /// falls back to a clearly-labelled placeholder in that case.
    #[cfg(feature = "gpu")]
    fn real_device_info(device_id: usize) -> Option<GpuDeviceInfo> {
        oxicuda_driver::init().ok()?;
        let ordinal = i32::try_from(device_id).ok()?;
        let device = oxicuda_driver::Device::get(ordinal).ok()?;
        let info = device.info().ok()?;
        let memory_free = oxicuda_driver::memory_info::device_memory_info()
            .map(|(free, _total)| free)
            .unwrap_or(info.total_memory_bytes);

        Some(GpuDeviceInfo {
            device_id,
            name: info.name,
            memory_total: info.total_memory_bytes,
            memory_free,
            compute_capability: (
                info.compute_capability.0 as u32,
                info.compute_capability.1 as u32,
            ),
            multiprocessor_count: info.multiprocessor_count as u32,
            max_threads_per_block: info.max_threads_per_block as u32,
            max_shared_memory_per_block: info.max_shared_memory_per_block as usize,
        })
    }

    // ── Core GEMM dispatch ───────────────────────────────────────────────────

    /// Single-device GEMM via `oxicuda-backend` when `gpu` is enabled.
    fn single_gpu_matrix_multiply(
        &mut self,
        a: &Array2<Float>,
        b: &Array2<Float>,
        _device_id: usize,
    ) -> Result<Array2<Float>> {
        #[cfg(feature = "gpu")]
        if let Some(backend) = self.gpu_backend.as_ref() {
            let (m, k) = a.dim();
            let (_, n) = b.dim();
            if m * k + k * n >= self.config.min_problem_size {
                let a_gpu = GpuArray::<Float>::from_array2(backend, a)?;
                let b_gpu = GpuArray::<Float>::from_array2(backend, b)?;
                let c_gpu = a_gpu.matmul(&b_gpu)?;
                return c_gpu.to_array2();
            }
        }
        Ok(a.dot(b))
    }

    fn single_gpu_matrix_multiply_slice(
        &mut self,
        a: &ArrayView2<Float>,
        b: &Array2<Float>,
        device_id: usize,
    ) -> Result<Array2<Float>> {
        let a_owned = a.to_owned();
        self.single_gpu_matrix_multiply(&a_owned, b, device_id)
    }

    // ── Public operations ────────────────────────────────────────────────────

    /// Multi-GPU matrix multiplication with load-balanced row partitioning.
    pub fn multi_gpu_matrix_multiply(
        &mut self,
        a: &Array2<Float>,
        b: &Array2<Float>,
    ) -> Result<Array2<Float>> {
        let (m, k) = a.dim();
        let (k2, n) = b.dim();

        if k != k2 {
            return Err(SklearsError::InvalidInput(format!(
                "Matrix dimensions incompatible: ({}, {}) * ({}, {})",
                m, k, k2, n
            )));
        }

        let start_time = Instant::now();

        let result = if m * n * k < self.config.min_problem_size * self.config.device_ids.len() {
            self.single_gpu_matrix_multiply(a, b, 0)?
        } else {
            let device_assignments = self.load_balancer.distribute_work(m, &self.devices);
            let mut parts: Vec<(std::ops::Range<usize>, Array2<Float>)> = Vec::new();

            for (device_id, row_range) in device_assignments {
                let a_slice = a.slice(s![row_range.clone(), ..]);
                let part = self.single_gpu_matrix_multiply_slice(&a_slice, b, device_id)?;
                parts.push((row_range, part));
            }

            let mut output = Array2::zeros((m, n));
            for (row_range, part) in parts {
                output.slice_mut(s![row_range, ..]).assign(&part);
            }
            output
        };

        if self.config.enable_profiling {
            let duration = start_time.elapsed();
            let ops = 2.0 * m as f64 * n as f64 * k as f64;
            let throughput = ops / duration.as_secs_f64() / 1e9;
            let memory_used = (m * k + k * n + m * n) * std::mem::size_of::<Float>();
            self.performance_metrics.push(GpuPerformanceMetrics {
                device_id: 0,
                operation_name: "multi_gpu_matrix_multiply".to_string(),
                duration,
                memory_used,
                throughput_gflops: throughput,
                memory_bandwidth_gbps: memory_bandwidth_gbps(memory_used, duration),
                // Real per-kernel occupancy would require the launch's
                // actual grid/block dimensions, which `single_gpu_matrix_multiply`
                // does not surface (the kernel launch is internal to
                // `oxicuda-blas::level3::gemm`) -- see module-level notes.
                occupancy_percentage: 0.0,
            });
        }

        Ok(result)
    }

    /// Fused multiply-add: `C_new = A × B + C`.
    pub fn fused_matrix_multiply_add(
        &mut self,
        a: &Array2<Float>,
        b: &Array2<Float>,
        c: &Array2<Float>,
    ) -> Result<Array2<Float>> {
        let (m, k) = a.dim();
        let (k2, n) = b.dim();
        let (m2, n2) = c.dim();

        if k != k2 || m != m2 || n != n2 {
            return Err(SklearsError::InvalidInput(
                "Matrix dimensions incompatible for fused operation".to_string(),
            ));
        }

        let start_time = Instant::now();

        let ab = if self.config.enable_kernel_fusion {
            self.single_gpu_matrix_multiply(a, b, 0)?
        } else {
            a.dot(b)
        };
        let result = &ab + c;

        if self.config.enable_profiling {
            let duration = start_time.elapsed();
            let ops = 2.0 * m as f64 * n as f64 * k as f64 + m as f64 * n as f64;
            let throughput = ops / duration.as_secs_f64() / 1e9;
            let memory_used = (m * k + k * n + 2 * m * n) * std::mem::size_of::<Float>();
            self.performance_metrics.push(GpuPerformanceMetrics {
                device_id: 0,
                operation_name: "fused_matrix_multiply_add".to_string(),
                duration,
                memory_used,
                throughput_gflops: throughput,
                memory_bandwidth_gbps: memory_bandwidth_gbps(memory_used, duration),
                occupancy_percentage: 0.0,
            });
        }

        Ok(result)
    }

    /// Mixed precision matrix multiplication.
    ///
    /// When mixed precision is enabled and a GPU is available, this runs a
    /// real FP16-input GEMM via `oxicuda_blas::level3::gemm::<half::f16>`.
    /// `GpuFloat::Accumulator` for `f16` is `f32` (see
    /// `oxicuda_blas::types::GpuFloat`), so this is the standard Tensor
    /// Core "mixed precision" mode: every multiply-accumulate happens in
    /// FP32 even though the operands and result are stored as FP16. `a`/`b`
    /// are downcast to FP16 before upload and the FP16 result is upcast
    /// back to `Float` on return -- trading precision for throughput and
    /// memory bandwidth, the same trade-off `torch.cuda.amp` and similar
    /// mixed-precision training modes make.
    ///
    /// Falls back to the plain-precision path (identical to the
    /// `enable_mixed_precision == false` case) whenever no GPU is
    /// available or the FP16 path itself errors (e.g. unsupported
    /// architecture).
    pub fn mixed_precision_matrix_multiply(
        &mut self,
        a: &Array2<Float>,
        b: &Array2<Float>,
    ) -> Result<Array2<Float>> {
        if !self.config.enable_mixed_precision {
            return self.single_gpu_matrix_multiply(a, b, 0);
        }

        let start_time = Instant::now();

        #[cfg(feature = "gpu")]
        if let Some(backend) = self.gpu_backend.as_ref() {
            match Self::fp16_matrix_multiply(backend, a, b) {
                Ok(result) => {
                    self.record_matmul_metrics("mixed_precision_matrix_multiply", a, b, start_time);
                    return Ok(result);
                }
                Err(e) => {
                    log::warn!(
                        "Mixed-precision (FP16) GEMM failed ({e}), falling back to plain-precision path"
                    );
                }
            }
        }

        let result = self.single_gpu_matrix_multiply(a, b, 0)?;
        self.record_matmul_metrics("mixed_precision_matrix_multiply", a, b, start_time);
        Ok(result)
    }

    /// Records a `GpuPerformanceMetrics` sample for a plain `m x k` by
    /// `k x n` GEMM, if profiling is enabled. Shared by both branches of
    /// [`mixed_precision_matrix_multiply`] so they report comparable
    /// throughput numbers.
    fn record_matmul_metrics(
        &mut self,
        operation_name: &str,
        a: &Array2<Float>,
        b: &Array2<Float>,
        start_time: Instant,
    ) {
        if !self.config.enable_profiling {
            return;
        }
        let duration = start_time.elapsed();
        let (m, k) = a.dim();
        let (_, n) = b.dim();
        let ops = 2.0 * m as f64 * n as f64 * k as f64;
        let throughput = ops / duration.as_secs_f64() / 1e9;
        let memory_used = (m * k + k * n + m * n) * std::mem::size_of::<Float>();
        self.performance_metrics.push(GpuPerformanceMetrics {
            device_id: 0,
            operation_name: operation_name.to_string(),
            duration,
            memory_used,
            throughput_gflops: throughput,
            memory_bandwidth_gbps: memory_bandwidth_gbps(memory_used, duration),
            occupancy_percentage: 0.0,
        });
    }

    /// Executes a GEMM in FP16 (see [`mixed_precision_matrix_multiply`] for
    /// why this counts as genuine mixed-precision compute). Implemented
    /// directly against `oxicuda-blas`/`oxicuda-memory` rather than
    /// `sklears_core::gpu::{GpuArray, GpuMatrixOps}`, since that wrapper
    /// only covers `f32`/`f64` today.
    #[cfg(feature = "gpu")]
    fn fp16_matrix_multiply(
        backend: &GpuBackend,
        a: &Array2<Float>,
        b: &Array2<Float>,
    ) -> Result<Array2<Float>> {
        let (m, k) = a.dim();
        let (k2, n) = b.dim();
        if k != k2 {
            return Err(SklearsError::InvalidInput(format!(
                "fp16_matrix_multiply: inner dimensions differ ({k} vs {k2})"
            )));
        }

        backend.context().set_current().map_err(gpu_err)?;

        let a_f16: Vec<f16> = a.iter().map(|&v| f16::from_f64(v)).collect();
        let b_f16: Vec<f16> = b.iter().map(|&v| f16::from_f64(v)).collect();

        let a_buf = DeviceBuffer::from_host(&a_f16).map_err(gpu_err)?;
        let b_buf = DeviceBuffer::from_host(&b_f16).map_err(gpu_err)?;
        let mut c_buf = DeviceBuffer::<f16>::zeroed(m * n).map_err(gpu_err)?;

        let a_desc = MatrixDesc::from_buffer(&a_buf, m as u32, k as u32, Layout::RowMajor)
            .map_err(gpu_err)?;
        let b_desc = MatrixDesc::from_buffer(&b_buf, k as u32, n as u32, Layout::RowMajor)
            .map_err(gpu_err)?;
        let mut c_desc =
            MatrixDescMut::from_buffer(&mut c_buf, m as u32, n as u32, Layout::RowMajor)
                .map_err(gpu_err)?;

        oxicuda_blas::level3::gemm(
            backend.blas(),
            Transpose::NoTrans,
            Transpose::NoTrans,
            f16::ONE,
            &a_desc,
            &b_desc,
            f16::ZERO,
            &mut c_desc,
        )
        .map_err(gpu_err)?;

        // The GEMM ran on the non-blocking compute stream; `copy_to_host`
        // copies on the legacy default stream, which does not implicitly wait
        // on it.
        backend.synchronize()?;
        let mut c_f16 = vec![f16::ZERO; m * n];
        c_buf.copy_to_host(&mut c_f16).map_err(gpu_err)?;

        let c_flat: Vec<Float> = c_f16.iter().map(|v| v.to_f64()).collect();
        Ok(Array2::from_shape_vec((m, n), c_flat)?)
    }

    /// Submit a GEMM on a stream and return a handle tracking its
    /// completion.
    ///
    /// When the chosen stream is genuinely backed by an `oxicuda-driver`
    /// stream (see [`CudaStream::is_real`]), this launches the GEMM via
    /// `CudaStream::launch_matmul`: the kernel and its device-to-host
    /// result copy are issued asynchronously, and the returned
    /// [`AsyncGpuOperation`] tracks completion by polling a real
    /// `cuEventQuery` (see [`AsyncGpuOperation::is_ready`]) rather than
    /// assuming it. Falls back to the eager `single_gpu_matrix_multiply`
    /// path (result known immediately, `is_ready()` trivially `true`) when
    /// no GPU is present or the real-stream launch itself fails.
    pub fn async_matrix_multiply(
        &mut self,
        a: &Array2<Float>,
        b: &Array2<Float>,
        device_id: usize,
    ) -> Result<AsyncGpuOperation> {
        let (m, k) = a.dim();
        let (k2, n) = b.dim();

        if k != k2 {
            return Err(SklearsError::InvalidInput(format!(
                "Matrix dimensions incompatible: ({}, {}) * ({}, {})",
                m, k, k2, n
            )));
        }

        let stream_id = self.find_available_stream(device_id)?;
        self.streams[device_id][stream_id].is_busy = true;
        let start_time = Instant::now();

        #[cfg(feature = "gpu")]
        if self.streams[device_id][stream_id].is_real() {
            match self.streams[device_id][stream_id].launch_matmul(a, b) {
                Ok(pending) => {
                    // The kernel launch has returned control to the host;
                    // this stream is free to accept new work while the
                    // pending op finishes in the background.
                    self.streams[device_id][stream_id].is_busy = false;
                    return Ok(AsyncGpuOperation {
                        operation_id: 0,
                        device_id,
                        stream_id,
                        start_time,
                        result_shape: (m, n),
                        state: std::cell::RefCell::new(AsyncGpuState::Pending(pending)),
                    });
                }
                Err(e) => {
                    log::warn!(
                        "Real async stream GEMM launch failed ({e}), falling back to eager synchronous compute"
                    );
                }
            }
        }

        let result = self.single_gpu_matrix_multiply(a, b, device_id)?;
        self.streams[device_id][stream_id].is_busy = false;

        Ok(AsyncGpuOperation {
            operation_id: 0,
            device_id,
            stream_id,
            start_time,
            result_shape: (m, n),
            state: std::cell::RefCell::new(AsyncGpuState::Ready(result)),
        })
    }

    fn find_available_stream(&self, device_id: usize) -> Result<usize> {
        for (i, stream) in self.streams[device_id].iter().enumerate() {
            if stream.is_available() {
                return Ok(i);
            }
        }
        Err(SklearsError::InvalidInput(
            "No available streams on device".to_string(),
        ))
    }

    /// Distributed GEMM across multiple GPUs via row partitioning.
    pub fn distributed_matrix_multiply(
        &mut self,
        a: &Array2<Float>,
        b: &Array2<Float>,
    ) -> Result<Array2<Float>> {
        let (m, k) = a.dim();
        let (k2, _) = b.dim();

        if k != k2 {
            return Err(SklearsError::InvalidInput(
                "Matrix dimensions incompatible".to_string(),
            ));
        }

        if m * k * b.dim().1 > 1_000_000_000 {
            self.distributed_gemm_algorithm(a, b)
        } else {
            self.multi_gpu_matrix_multiply(a, b)
        }
    }

    fn distributed_gemm_algorithm(
        &mut self,
        a: &Array2<Float>,
        b: &Array2<Float>,
    ) -> Result<Array2<Float>> {
        let (m, _) = a.dim();
        let (_, n) = b.dim();
        let num_gpus = self.config.device_ids.len();
        let block_size = m.div_ceil(num_gpus);
        let device_ids: Vec<usize> = self.config.device_ids.clone();

        let mut parts: Vec<(std::ops::Range<usize>, Array2<Float>)> = Vec::new();
        for (i, device_id) in device_ids.iter().enumerate() {
            let start_row = i * block_size;
            let end_row = ((i + 1) * block_size).min(m);
            if start_row < end_row {
                let a_block = a.slice(s![start_row..end_row, ..]);
                let part = self.single_gpu_matrix_multiply_slice(&a_block, b, *device_id)?;
                parts.push((start_row..end_row, part));
            }
        }

        let mut output = Array2::zeros((m, n));
        for (row_range, part) in parts {
            output.slice_mut(s![row_range, ..]).assign(&part);
        }
        Ok(output)
    }

    /// Batched GEMM (loop over batch dimension).
    pub fn batch_matrix_multiply(
        &mut self,
        a_batch: &Array3<Float>,
        b_batch: &Array3<Float>,
    ) -> Result<Array3<Float>> {
        let (batch_size, m, k) = a_batch.dim();
        let (batch_size2, k2, n) = b_batch.dim();

        if batch_size != batch_size2 || k != k2 {
            return Err(SklearsError::InvalidInput(
                "Batch dimensions incompatible".to_string(),
            ));
        }

        let start_time = Instant::now();
        let num_devices = self.config.device_ids.len();
        let mut results: Vec<Array2<Float>> = Vec::with_capacity(batch_size);

        for i in 0..batch_size {
            let a_slice = a_batch.slice(s![i, .., ..]).to_owned();
            let b_slice = b_batch.slice(s![i, .., ..]).to_owned();
            let device_id = self.config.device_ids[i % num_devices];
            results.push(self.single_gpu_matrix_multiply(&a_slice, &b_slice, device_id)?);
        }

        let mut output = Array3::zeros((batch_size, m, n));
        for (i, result) in results.iter().enumerate() {
            output.slice_mut(s![i, .., ..]).assign(result);
        }

        if self.config.enable_profiling {
            let duration = start_time.elapsed();
            let ops = 2.0 * batch_size as f64 * m as f64 * n as f64 * k as f64;
            let throughput = ops / duration.as_secs_f64() / 1e9;
            let memory_used = batch_size * (m * k + k * n + m * n) * std::mem::size_of::<Float>();
            self.performance_metrics.push(GpuPerformanceMetrics {
                device_id: 0,
                operation_name: "batch_matrix_multiply".to_string(),
                duration,
                memory_used,
                throughput_gflops: throughput,
                memory_bandwidth_gbps: memory_bandwidth_gbps(memory_used, duration),
                occupancy_percentage: 0.0,
            });
        }

        Ok(output)
    }

    // ── Introspection ────────────────────────────────────────────────────────

    pub fn get_performance_metrics(&self) -> &[GpuPerformanceMetrics] {
        &self.performance_metrics
    }

    pub fn get_memory_usage(&self) -> Vec<(usize, usize)> {
        self.memory_pools
            .iter()
            .filter_map(|pool| pool.lock().ok().map(|p| p.memory_usage()))
            .collect()
    }

    pub fn get_devices(&self) -> &[GpuDeviceInfo] {
        &self.devices
    }

    /// Human-readable performance report.
    pub fn generate_performance_report(&self) -> String {
        let mut report = String::new();
        report.push_str("=== GPU Performance Report ===\n");
        report.push_str(&format!(
            "Number of GPUs: {}\n",
            self.config.device_ids.len()
        ));
        report.push_str(&format!(
            "Total operations: {}\n",
            self.performance_metrics.len()
        ));

        if !self.performance_metrics.is_empty() {
            let total_gflops: f64 = self
                .performance_metrics
                .iter()
                .map(|m| m.throughput_gflops)
                .sum();
            let avg_gflops = total_gflops / self.performance_metrics.len() as f64;
            report.push_str(&format!("Average throughput: {:.2} GFLOPS\n", avg_gflops));

            let total_memory: usize = self.performance_metrics.iter().map(|m| m.memory_used).sum();
            report.push_str(&format!(
                "Total memory used: {} MB\n",
                total_memory / 1024 / 1024
            ));
        }

        report.push_str("\nDevice Information:\n");
        for device in self.get_devices() {
            report.push_str(&format!(
                "  Device {}: {} ({} MB)\n",
                device.device_id,
                device.name,
                device.memory_total / 1024 / 1024
            ));
        }

        report.push_str("\nMemory Usage:\n");
        for (i, (used, total)) in self.get_memory_usage().iter().enumerate() {
            let usage_percent = (*used as f64 / *total as f64) * 100.0;
            report.push_str(&format!(
                "  GPU {}: {:.1}% ({} MB / {} MB)\n",
                i,
                usage_percent,
                used / 1024 / 1024,
                total / 1024 / 1024
            ));
        }

        report
    }
}

// ─── LoadBalancer ──────────────────────────────────────────────────────────────

/// Distributes row-work across devices according to the chosen strategy.
#[derive(Debug)]
pub struct LoadBalancer {
    #[allow(dead_code)]
    strategy: LoadBalancingStrategy,
    device_weights: Vec<f32>,
}

impl LoadBalancer {
    pub fn new(strategy: LoadBalancingStrategy, devices: &[GpuDeviceInfo]) -> Self {
        let device_weights = match strategy {
            LoadBalancingStrategy::RoundRobin => vec![1.0; devices.len()],
            LoadBalancingStrategy::MemoryBased => {
                devices.iter().map(|d| d.memory_total as f32).collect()
            }
            LoadBalancingStrategy::ComputeCapabilityBased => devices
                .iter()
                .map(|d| {
                    let (major, minor) = d.compute_capability;
                    (major as f32 * 10.0 + minor as f32) * d.multiprocessor_count as f32
                })
                .collect(),
            LoadBalancingStrategy::Dynamic => vec![1.0; devices.len()],
        };
        Self {
            strategy,
            device_weights,
        }
    }

    pub fn distribute_work(
        &self,
        total_work: usize,
        devices: &[GpuDeviceInfo],
    ) -> Vec<(usize, std::ops::Range<usize>)> {
        let mut assignments = Vec::new();
        let total_weight: f32 = self.device_weights.iter().sum();
        let mut current_offset = 0;

        for (i, &weight) in self.device_weights.iter().enumerate() {
            let work_fraction = weight / total_weight;
            let work_size = ((total_work as f32 * work_fraction) as usize).max(1);
            let end_offset = (current_offset + work_size).min(total_work);

            if current_offset < end_offset {
                assignments.push((devices[i].device_id, current_offset..end_offset));
                current_offset = end_offset;
            }
        }

        assignments
    }
}

// ─── AsyncGpuOperation ─────────────────────────────────────────────────────────

/// Internal completion state for an [`AsyncGpuOperation`].
enum AsyncGpuState {
    /// Result already known: either a real GEMM finished (event confirmed
    /// completion, see [`AsyncGpuOperation::is_ready`]) or this operation
    /// ran eagerly on the CPU/blocking-GPU path from the start.
    Ready(Array2<Float>),
    /// Genuinely in flight on a real `oxicuda-driver` stream.
    #[cfg(feature = "gpu")]
    Pending(PendingGpuMatmul),
    /// The pending GEMM's event fired, but materializing the result failed
    /// (should not happen in practice -- see
    /// [`PendingGpuMatmul::into_result`] -- but this avoids ever silently
    /// handing back a fabricated result if it somehow does). Only reachable
    /// with the `gpu` feature enabled; kept unconditional so
    /// [`AsyncGpuOperation::get_result`] can match it without `#[cfg]`.
    #[allow(dead_code)]
    Failed(String),
}

/// Handle for a submitted GPU operation.
///
/// See [`AdvancedGpuOps::async_matrix_multiply`] for the two ways this can
/// be constructed: a genuinely in-flight real-stream GEMM (state starts as
/// `Pending`), or an already-computed eager result (state starts as
/// `Ready`).
pub struct AsyncGpuOperation {
    pub operation_id: usize,
    pub device_id: usize,
    pub stream_id: usize,
    pub start_time: Instant,
    pub result_shape: (usize, usize),
    state: std::cell::RefCell<AsyncGpuState>,
}

impl std::fmt::Debug for AsyncGpuOperation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AsyncGpuOperation")
            .field("operation_id", &self.operation_id)
            .field("device_id", &self.device_id)
            .field("stream_id", &self.stream_id)
            .field("result_shape", &self.result_shape)
            .field("is_ready", &self.is_ready())
            .finish()
    }
}

impl AsyncGpuOperation {
    /// Polls whether the result is available.
    ///
    /// For an eager (CPU-fallback) operation this is always `true` -- the
    /// computation had already finished by the time this handle was
    /// constructed. For a genuinely in-flight real-stream operation, this
    /// performs a non-blocking `cuEventQuery` on the recorded completion
    /// event rather than assuming completion; once it observes completion
    /// it materializes the result (from the already-populated host buffer)
    /// so subsequent calls don't re-touch device resources.
    pub fn is_ready(&self) -> bool {
        #[cfg(feature = "gpu")]
        {
            let mut state = self.state.borrow_mut();
            let is_pending = matches!(&*state, AsyncGpuState::Pending(_));
            if is_pending {
                let event_fired = matches!(&*state, AsyncGpuState::Pending(p) if matches!(p.is_ready(), Ok(true)));
                if !event_fired {
                    return false;
                }
                // Placeholder swapped in only for the duration of this
                // block; overwritten below with the real outcome before the
                // borrow is released, so it is never observable externally.
                let previous = std::mem::replace(&mut *state, AsyncGpuState::Failed(String::new()));
                let AsyncGpuState::Pending(pending) = previous else {
                    unreachable!("checked above: state is AsyncGpuState::Pending");
                };
                *state = match pending.into_result() {
                    Ok(result) => AsyncGpuState::Ready(result),
                    Err(e) => AsyncGpuState::Failed(format!("{e}")),
                };
            }
            matches!(&*state, AsyncGpuState::Ready(_))
        }
        #[cfg(not(feature = "gpu"))]
        {
            matches!(&*self.state.borrow(), AsyncGpuState::Ready(_))
        }
    }

    /// Returns the result of this operation.
    ///
    /// # Errors
    ///
    /// Returns [`SklearsError::InvalidInput`] if the operation has not
    /// completed yet (`is_ready()` would return `false`). Returns
    /// [`SklearsError::NumericalError`] if the underlying GEMM failed.
    pub fn get_result(&self) -> Result<Array2<Float>> {
        // Give a pending real-stream op a chance to resolve first.
        self.is_ready();
        match &*self.state.borrow() {
            AsyncGpuState::Ready(result) => Ok(result.clone()),
            AsyncGpuState::Failed(msg) => Err(SklearsError::NumericalError(format!(
                "Async GPU operation failed: {msg}"
            ))),
            #[cfg(feature = "gpu")]
            AsyncGpuState::Pending(_) => Err(SklearsError::InvalidInput(
                "Operation not complete".to_string(),
            )),
        }
    }

    pub fn elapsed_time(&self) -> Duration {
        self.start_time.elapsed()
    }
}

// ─── Tests ─────────────────────────────────────────────────────────────────────

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_advanced_gpu_config() {
        let config = AdvancedGpuConfig::default();
        assert_eq!(config.device_ids, vec![0]);
        assert_eq!(config.streams_per_gpu, 4);
        assert!(config.enable_mixed_precision);
        assert!(config.enable_kernel_fusion);
    }

    #[test]
    fn test_memory_pool() {
        let mut pool = GpuMemoryPool::new(0, 1024);

        let ptr1 = pool.allocate(256).expect("operation should succeed");
        let ptr2 = pool.allocate(256).expect("operation should succeed");
        assert_ne!(ptr1, ptr2);

        pool.deallocate(ptr1).expect("operation should succeed");
        let _ptr3 = pool.allocate(128).expect("operation should succeed");

        let (used, total) = pool.memory_usage();
        assert_eq!(total, 1024);
        assert!(used > 0);
    }

    #[test]
    fn test_memory_pool_device_backing_matches_real_detection() {
        // Regression test: `is_device_backed()` must report the *real* backing
        // state, never a hardcoded value. A small (1 KiB) arena reservation
        // succeeds on any detected device, so the pool is device-backed exactly
        // when a GPU backend is present -- and never without the `gpu` feature,
        // where there is no device stack at all.
        let pool = GpuMemoryPool::new(0, 1024);
        #[cfg(feature = "gpu")]
        {
            let gpu_present = GpuBackend::with_device_id(0).ok().flatten().is_some();
            assert_eq!(pool.is_device_backed(), gpu_present);
        }
        #[cfg(not(feature = "gpu"))]
        {
            assert!(!pool.is_device_backed());
        }
    }

    #[test]
    fn test_load_balancer() {
        let devices = vec![
            GpuDeviceInfo {
                device_id: 0,
                name: "GPU 0".to_string(),
                memory_total: 8 * 1024 * 1024 * 1024,
                memory_free: 7 * 1024 * 1024 * 1024,
                compute_capability: (8, 0),
                multiprocessor_count: 68,
                max_threads_per_block: 1024,
                max_shared_memory_per_block: 49152,
            },
            GpuDeviceInfo {
                device_id: 1,
                name: "GPU 1".to_string(),
                memory_total: 8 * 1024 * 1024 * 1024,
                memory_free: 7 * 1024 * 1024 * 1024,
                compute_capability: (8, 0),
                multiprocessor_count: 68,
                max_threads_per_block: 1024,
                max_shared_memory_per_block: 49152,
            },
        ];

        let balancer = LoadBalancer::new(LoadBalancingStrategy::RoundRobin, &devices);
        let assignments = balancer.distribute_work(1000, &devices);

        assert_eq!(assignments.len(), 2);
        assert_eq!(assignments[0].0, 0);
        assert_eq!(assignments[1].0, 1);
    }

    #[test]
    fn test_advanced_gpu_ops_creation() {
        let config = AdvancedGpuConfig {
            device_ids: vec![0],
            ..Default::default()
        };

        let ops = AdvancedGpuOps::new(config);
        assert!(ops.is_ok());

        let ops = ops.expect("operation should succeed");
        assert_eq!(ops.devices.len(), 1);
        assert_eq!(ops.memory_pools.len(), 1);
    }

    #[test]
    fn test_matrix_operations() {
        let config = AdvancedGpuConfig::default();
        let mut ops = AdvancedGpuOps::new(config).expect("operation should succeed");

        let a = Array2::from_shape_vec(
            (4, 3),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        )
        .expect("operation should succeed");
        let b = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .expect("valid array shape");

        let result = ops
            .multi_gpu_matrix_multiply(&a, &b)
            .expect("operation should succeed");
        assert_eq!(result.dim(), (4, 2));

        let c = Array2::from_shape_vec((4, 2), vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
            .expect("valid array shape");
        let fused_result = ops
            .fused_matrix_multiply_add(&a, &b, &c)
            .expect("operation should succeed");
        assert_eq!(fused_result.dim(), (4, 2));
    }

    #[test]
    fn test_performance_metrics() {
        let config = AdvancedGpuConfig {
            enable_profiling: true,
            ..Default::default()
        };
        let mut ops = AdvancedGpuOps::new(config).expect("operation should succeed");

        let a = Array2::from_shape_vec((10, 10), (0..100).map(|i| i as f64).collect())
            .expect("valid array shape");
        let b = Array2::from_shape_vec((10, 10), (0..100).map(|i| i as f64).collect())
            .expect("valid array shape");

        let _result = ops
            .multi_gpu_matrix_multiply(&a, &b)
            .expect("operation should succeed");

        let metrics = ops.get_performance_metrics();
        assert!(!metrics.is_empty());
        assert_eq!(metrics[0].operation_name, "multi_gpu_matrix_multiply");
        // Regression test: bandwidth must be derived from real
        // bytes-moved/duration, not a hardcoded `0.0`.
        assert!(metrics[0].memory_bandwidth_gbps > 0.0);
    }

    #[test]
    fn test_batch_operations() {
        use scirs2_core::ndarray::Array3;
        let config = AdvancedGpuConfig::default();
        let mut ops = AdvancedGpuOps::new(config).expect("operation should succeed");

        let batch_size = 3;
        let m = 4;
        let k = 3;
        let n = 2;

        let a_batch = Array3::from_shape_vec(
            (batch_size, m, k),
            (0..batch_size * m * k).map(|i| i as f64).collect(),
        )
        .expect("operation should succeed");
        let b_batch = Array3::from_shape_vec(
            (batch_size, k, n),
            (0..batch_size * k * n).map(|i| i as f64).collect(),
        )
        .expect("operation should succeed");

        let result = ops
            .batch_matrix_multiply(&a_batch, &b_batch)
            .expect("operation should succeed");
        assert_eq!(result.dim(), (batch_size, m, n));
    }

    #[test]
    fn test_async_operations() {
        let config = AdvancedGpuConfig::default();
        let mut ops = AdvancedGpuOps::new(config).expect("operation should succeed");

        let a = Array2::from_shape_vec(
            (4, 3),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        )
        .expect("operation should succeed");
        let b = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .expect("valid array shape");

        let async_op = ops
            .async_matrix_multiply(&a, &b, 0)
            .expect("operation should succeed");
        assert_eq!(async_op.device_id, 0);
        assert_eq!(async_op.result_shape, (4, 2));

        // Regression test: `get_result()` must return the real product,
        // not a fabricated zero matrix (see `AsyncGpuOperation::get_result`).
        assert!(async_op.is_ready());
        let result = async_op.get_result().expect("result should be available");
        assert_eq!(result.dim(), (4, 2));
        let expected = a.dot(&b);
        for (got, want) in result.iter().zip(expected.iter()) {
            assert!(
                (got - want).abs() < 1e-10,
                "got {got}, want {want} (async result must match a.dot(b))"
            );
        }
    }

    #[test]
    fn test_memory_management() {
        let config = AdvancedGpuConfig::default();
        let ops = AdvancedGpuOps::new(config).expect("operation should succeed");

        let memory_usage = ops.get_memory_usage();
        assert_eq!(memory_usage.len(), 1);

        let (used, total) = memory_usage[0];
        assert_eq!(used, 0);
        assert_eq!(total, 1 << 30);
    }

    #[test]
    fn test_performance_report() {
        let config = AdvancedGpuConfig {
            enable_profiling: true,
            ..Default::default()
        };
        let mut ops = AdvancedGpuOps::new(config).expect("operation should succeed");

        let a = Array2::from_shape_vec((10, 10), (0..100).map(|i| i as f64).collect())
            .expect("valid array shape");
        let b = Array2::from_shape_vec((10, 10), (0..100).map(|i| i as f64).collect())
            .expect("valid array shape");

        let _result = ops
            .multi_gpu_matrix_multiply(&a, &b)
            .expect("operation should succeed");

        let report = ops.generate_performance_report();
        assert!(report.contains("GPU Performance Report"));
        assert!(report.contains("Number of GPUs: 1"));
        assert!(report.contains("Total operations: 1"));
    }
}
