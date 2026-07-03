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
use oxicuda_blas::{Layout, MatrixDesc, MatrixDescMut, Transpose};
#[cfg(feature = "gpu")]
use oxicuda_memory::DeviceBuffer;
#[cfg(feature = "gpu")]
use sklears_core::gpu::{GpuArray, GpuBackend, GpuMatrixOps, GpuUtils};

/// Maps any GPU-stack error (`oxicuda-driver`/`oxicuda-blas`) to a
/// `SklearsError`, without needing those crates as direct dependencies just
/// to name their error types.
#[cfg(feature = "gpu")]
fn gpu_err<E: std::fmt::Display>(e: E) -> SklearsError {
    SklearsError::NumericalError(format!("GPU error: {e}"))
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

/// Memory pool with best-fit allocation and coalesced free-block merging.
#[derive(Debug)]
pub struct GpuMemoryPool {
    device_id: usize,
    total_size: usize,
    used_size: usize,
    free_blocks: Vec<(usize, usize)>,
    allocated_blocks: HashMap<usize, usize>,
}

impl GpuMemoryPool {
    pub fn new(device_id: usize, size: usize) -> Self {
        Self {
            device_id,
            total_size: size,
            used_size: 0,
            free_blocks: vec![(0, size)],
            allocated_blocks: HashMap::new(),
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
#[derive(Debug)]
pub struct CudaStream {
    #[allow(dead_code)]
    stream_id: usize,
    #[allow(dead_code)]
    device_id: usize,
    is_busy: bool,
}

impl CudaStream {
    pub fn new(device_id: usize, stream_id: usize) -> Self {
        Self {
            stream_id,
            device_id,
            is_busy: false,
        }
    }

    pub fn is_available(&self) -> bool {
        !self.is_busy
    }

    pub fn synchronize(&mut self) -> Result<()> {
        self.is_busy = false;
        Ok(())
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
                .map(|i| CudaStream::new(device_id, i))
                .collect();
            streams.push(device_streams);
        }

        let load_balancer = LoadBalancer::new(config.load_balancing, &devices);

        #[cfg(feature = "gpu")]
        let gpu_backend = GpuBackend::detect()?;

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

    /// Query device information from the backend, falling back to sensible defaults.
    fn get_device_info(device_id: usize) -> Result<GpuDeviceInfo> {
        #[cfg(feature = "gpu")]
        {
            if let Ok(props) = GpuUtils::device_properties(device_id) {
                return Ok(GpuDeviceInfo {
                    device_id,
                    name: props.name.clone(),
                    memory_total: props.total_memory,
                    memory_free: props.free_memory,
                    compute_capability: (
                        props.compute_capability.0 as u32,
                        props.compute_capability.1 as u32,
                    ),
                    multiprocessor_count: 1,
                    max_threads_per_block: 1024,
                    max_shared_memory_per_block: 49152,
                });
            }
        }
        Ok(GpuDeviceInfo {
            device_id,
            name: format!("GPU Device {}", device_id),
            memory_total: 8 * 1024 * 1024 * 1024,
            memory_free: 7 * 1024 * 1024 * 1024,
            compute_capability: (8, 0),
            multiprocessor_count: 68,
            max_threads_per_block: 1024,
            max_shared_memory_per_block: 49152,
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
            self.performance_metrics.push(GpuPerformanceMetrics {
                device_id: 0,
                operation_name: "multi_gpu_matrix_multiply".to_string(),
                duration,
                memory_used: (m * k + k * n + m * n) * std::mem::size_of::<Float>(),
                throughput_gflops: throughput,
                memory_bandwidth_gbps: 0.0,
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
            self.performance_metrics.push(GpuPerformanceMetrics {
                device_id: 0,
                operation_name: "fused_matrix_multiply_add".to_string(),
                duration,
                memory_used: (m * k + k * n + 2 * m * n) * std::mem::size_of::<Float>(),
                throughput_gflops: throughput,
                memory_bandwidth_gbps: 0.0,
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
        self.performance_metrics.push(GpuPerformanceMetrics {
            device_id: 0,
            operation_name: operation_name.to_string(),
            duration,
            memory_used: (m * k + k * n + m * n) * std::mem::size_of::<Float>(),
            throughput_gflops: throughput,
            memory_bandwidth_gbps: 0.0,
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

        let mut c_f16 = vec![f16::ZERO; m * n];
        c_buf.copy_to_host(&mut c_f16).map_err(gpu_err)?;

        let c_flat: Vec<Float> = c_f16.iter().map(|v| v.to_f64()).collect();
        Ok(Array2::from_shape_vec((m, n), c_flat)?)
    }

    /// Submit a GEMM and return a handle carrying its (real) result.
    ///
    /// There is no actual asynchronous GPU stream plumbing in this module:
    /// `CudaStream::synchronize` is bookkeeping only, and
    /// `single_gpu_matrix_multiply` is a blocking call. Rather than return a
    /// handle that can never honestly report completion (or, worse, one
    /// that fabricates a zeroed result once "complete"), the multiply is
    /// executed eagerly here and the genuine computed array is stored on
    /// the handle; [`AsyncGpuOperation::is_ready`] is `true` immediately
    /// because the computation has, in fact, already finished by the time
    /// this function returns. See [`AsyncGpuOperation::get_result`].
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
        let result = self.single_gpu_matrix_multiply(a, b, device_id)?;
        // The (synchronous) work is done, so the stream is free again.
        self.streams[device_id][stream_id].is_busy = false;

        Ok(AsyncGpuOperation {
            operation_id: 0,
            device_id,
            stream_id,
            start_time,
            result_shape: (m, n),
            is_complete: true,
            result,
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
            self.performance_metrics.push(GpuPerformanceMetrics {
                device_id: 0,
                operation_name: "batch_matrix_multiply".to_string(),
                duration,
                memory_used: batch_size * (m * k + k * n + m * n) * std::mem::size_of::<Float>(),
                throughput_gflops: throughput,
                memory_bandwidth_gbps: 0.0,
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

/// Handle for a submitted GPU operation.
///
/// See [`AdvancedGpuOps::async_matrix_multiply`] for why `result` is the
/// real, already-computed output rather than a promise resolved later.
#[derive(Debug)]
pub struct AsyncGpuOperation {
    pub operation_id: usize,
    pub device_id: usize,
    pub stream_id: usize,
    pub start_time: Instant,
    pub result_shape: (usize, usize),
    pub is_complete: bool,
    result: Array2<Float>,
}

impl AsyncGpuOperation {
    pub fn is_ready(&self) -> bool {
        self.is_complete
    }

    /// Returns the result of this operation.
    ///
    /// # Errors
    ///
    /// Returns [`SklearsError::InvalidInput`] if the operation has not
    /// completed yet (`is_ready()` is `false`).
    pub fn get_result(&self) -> Result<Array2<Float>> {
        if !self.is_complete {
            return Err(SklearsError::InvalidInput(
                "Operation not complete".to_string(),
            ));
        }
        Ok(self.result.clone())
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
