use std::sync::Arc;

use crate::error::Result;
use crate::prelude::SklearsError;
use scirs2_core::ndarray::Array2;

#[cfg(feature = "gpu_support")]
use oxicuda_backend::CpuBackend;
#[cfg(feature = "gpu_support")]
use oxicuda_backend::{BackendRegistry, BackendTranspose, BinaryOp, ComputeBackend, DeviceInfo};

// ─── GpuContext ────────────────────────────────────────────────────────────────

/// GPU context manager backed by `oxicuda-backend`.
///
/// When `gpu_support` is enabled, wraps a `Box<dyn ComputeBackend>`.
/// `CpuBackend` is always available; a real GPU backend is chosen when present.
#[cfg(feature = "gpu_support")]
#[derive(Debug, Clone)]
pub struct GpuContext {
    backend: Arc<dyn ComputeBackend>,
    device_id: usize,
}

#[cfg(not(feature = "gpu_support"))]
#[derive(Debug, Clone)]
pub struct GpuContext;

#[cfg(feature = "gpu_support")]
impl GpuContext {
    pub fn new() -> Result<Self> {
        Self::with_device_id(0)
    }

    pub fn with_device_id(device_id: usize) -> Result<Self> {
        let mut backend: Box<dyn ComputeBackend> = Box::new(CpuBackend::new());
        backend
            .init()
            .map_err(|e| SklearsError::NumericalError(e.to_string()))?;
        Ok(Self {
            backend: Arc::from(backend),
            device_id,
        })
    }

    pub fn device_id(&self) -> usize {
        self.device_id
    }

    pub fn get_stream(&self) -> Result<usize> {
        Ok(0)
    }

    pub fn return_stream(&self, _stream_id: usize) {}

    pub fn memory_info(&self) -> Result<GpuMemoryInfo> {
        let devices = self
            .backend
            .available_devices()
            .map_err(|e| SklearsError::NumericalError(e.to_string()))?;
        if let Some(dev) = devices.first() {
            let total = dev.total_memory_bytes as usize;
            if total > 0 {
                return Ok(GpuMemoryInfo {
                    total,
                    free: total,
                    used: 0,
                });
            }
        }
        // CpuBackend reports total_memory_bytes = 0; fall back to system RAM.
        let mem = crate::system_info::system_memory()?;
        Ok(GpuMemoryInfo {
            total: mem.total as usize,
            free: mem.available as usize,
            used: mem.used as usize,
        })
    }

    pub fn synchronize(&self) -> Result<()> {
        self.backend
            .synchronize()
            .map_err(|e| SklearsError::NumericalError(e.to_string()))
    }

    pub fn is_gpu(&self) -> bool {
        self.backend.name() != "cpu"
    }

    pub(crate) fn backend(&self) -> Arc<dyn ComputeBackend> {
        Arc::clone(&self.backend)
    }
}

#[cfg(not(feature = "gpu_support"))]
impl GpuContext {
    pub fn new() -> Result<Self> {
        Ok(Self)
    }

    pub fn with_device_id(_device_id: usize) -> Result<Self> {
        Ok(Self)
    }

    pub fn device_id(&self) -> usize {
        0
    }

    pub fn get_stream(&self) -> Result<usize> {
        Err(SklearsError::NotImplemented(
            "GPU support not enabled".into(),
        ))
    }

    pub fn return_stream(&self, _stream_id: usize) {}

    pub fn memory_info(&self) -> Result<GpuMemoryInfo> {
        Err(SklearsError::NotImplemented(
            "GPU support not enabled".into(),
        ))
    }

    pub fn synchronize(&self) -> Result<()> {
        Err(SklearsError::NotImplemented(
            "GPU support not enabled".into(),
        ))
    }

    pub fn is_gpu(&self) -> bool {
        false
    }
}

// ─── GpuMemoryInfo ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy)]
pub struct GpuMemoryInfo {
    pub free: usize,
    pub total: usize,
    pub used: usize,
}

impl GpuMemoryInfo {
    pub fn utilization(&self) -> f32 {
        if self.total == 0 {
            0.0
        } else {
            (self.used as f32 / self.total as f32) * 100.0
        }
    }
}

// ─── GpuArray ──────────────────────────────────────────────────────────────────

/// Device-resident array backed by a `ComputeBackend` allocation.
///
/// Acquired via `backend.alloc()`, released on `Drop` via `backend.free()`.
/// CPU↔device transfers use `bytemuck` for zero-copy byte reinterpretation.
#[cfg(feature = "gpu_support")]
pub struct GpuArray<T: bytemuck::Pod> {
    ptr: u64,
    shape: Vec<usize>,
    size: usize,
    backend: Arc<dyn ComputeBackend>,
    _marker: std::marker::PhantomData<T>,
}

#[cfg(not(feature = "gpu_support"))]
pub struct GpuArray<T> {
    _phantom: std::marker::PhantomData<T>,
}

#[cfg(feature = "gpu_support")]
impl<T: bytemuck::Pod> std::fmt::Debug for GpuArray<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuArray")
            .field("ptr", &format_args!("{:#x}", self.ptr))
            .field("shape", &self.shape)
            .field("size", &self.size)
            .finish()
    }
}

#[cfg(feature = "gpu_support")]
impl<T: bytemuck::Pod> Drop for GpuArray<T> {
    fn drop(&mut self) {
        let _ = self.backend.free(self.ptr);
    }
}

#[cfg(feature = "gpu_support")]
impl<T: bytemuck::Pod> GpuArray<T> {
    pub(crate) fn alloc_upload(
        backend: Arc<dyn ComputeBackend>,
        data: &[T],
        shape: Vec<usize>,
    ) -> Result<Self> {
        let bytes: &[u8] = bytemuck::cast_slice(data);
        let ptr = backend
            .alloc(bytes.len())
            .map_err(|e| SklearsError::NumericalError(e.to_string()))?;
        backend.copy_htod(ptr, bytes).map_err(|e| {
            let _ = backend.free(ptr);
            SklearsError::NumericalError(e.to_string())
        })?;
        Ok(Self {
            ptr,
            size: data.len(),
            shape,
            backend,
            _marker: std::marker::PhantomData,
        })
    }

    pub(crate) fn alloc_zeros(backend: Arc<dyn ComputeBackend>, shape: Vec<usize>) -> Result<Self> {
        let size: usize = shape.iter().product();
        let zeros: Vec<T> = vec![T::zeroed(); size];
        Self::alloc_upload(backend, &zeros, shape)
    }

    pub fn from_slice(context: &GpuContext, data: &[T]) -> Result<Self> {
        Self::alloc_upload(context.backend(), data, vec![data.len()])
    }

    pub fn from_array2(context: &GpuContext, array: &Array2<T>) -> Result<Self>
    where
        T: Clone,
    {
        let data: Vec<T> = if array.is_standard_layout() {
            array.as_slice().unwrap_or(&[]).to_vec()
        } else {
            array.iter().cloned().collect()
        };
        let shape = vec![array.nrows(), array.ncols()];
        Self::alloc_upload(context.backend(), &data, shape)
    }

    pub fn zeros(context: &GpuContext, shape: &[usize]) -> Result<Self> {
        Self::alloc_zeros(context.backend(), shape.to_vec())
    }

    pub fn to_cpu(&self) -> Result<Vec<T>> {
        let n_bytes = self.size * std::mem::size_of::<T>();
        let mut bytes = vec![0u8; n_bytes];
        self.backend
            .copy_dtoh(&mut bytes, self.ptr)
            .map_err(|e| SklearsError::NumericalError(e.to_string()))?;
        Ok(bytemuck::cast_slice(&bytes).to_vec())
    }

    pub fn to_array2(&self) -> Result<Array2<T>>
    where
        T: Clone,
    {
        if self.shape.len() != 2 {
            return Err(SklearsError::InvalidOperation(
                "to_array2 requires a 2-D GpuArray".into(),
            ));
        }
        let data = self.to_cpu()?;
        Array2::from_shape_vec((self.shape[0], self.shape[1]), data)
            .map_err(|e| SklearsError::InvalidOperation(format!("from_shape_vec: {e}")))
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn len(&self) -> usize {
        self.size
    }

    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    pub fn is_gpu_resident(&self) -> bool {
        self.backend.name() != "cpu"
    }

    pub fn to_cpu_only(&mut self) -> Result<()> {
        Ok(())
    }
}

#[cfg(not(feature = "gpu_support"))]
impl<T> GpuArray<T> {
    pub fn from_slice(_context: &GpuContext, _data: &[T]) -> Result<Self> {
        Err(SklearsError::NotImplemented(
            "GPU support not enabled".into(),
        ))
    }

    pub fn zeros(_context: &GpuContext, _shape: &[usize]) -> Result<Self> {
        Err(SklearsError::NotImplemented(
            "GPU support not enabled".into(),
        ))
    }
}

// ─── GpuMatrixOps ──────────────────────────────────────────────────────────────

pub trait GpuMatrixOps {
    fn matmul(&self, other: &Self) -> Result<Self>
    where
        Self: Sized;

    fn add(&self, other: &Self) -> Result<Self>
    where
        Self: Sized;

    fn mul(&self, other: &Self) -> Result<Self>
    where
        Self: Sized;

    fn scale(&self, scalar: f32) -> Result<Self>
    where
        Self: Sized;

    fn transpose(&self) -> Result<Self>
    where
        Self: Sized;
}

/// Row-major `C = A × B` via column-major backend identity `C^T = B^T × A^T`.
///
/// Row-major A[m×k] in memory equals column-major A^T[k×m].
/// Backend call: `gemm(NoTrans, NoTrans, n, m, k, 1, b_ptr, n, a_ptr, k, 0, c_ptr, n)`
#[cfg(feature = "gpu_support")]
fn gemm_row_major_f64(
    backend: &Arc<dyn ComputeBackend>,
    m: usize,
    k: usize,
    n: usize,
    a_ptr: u64,
    b_ptr: u64,
    c_ptr: u64,
) -> Result<()> {
    backend
        .gemm(
            BackendTranspose::NoTrans,
            BackendTranspose::NoTrans,
            n,
            m,
            k,
            1.0,
            b_ptr,
            n,
            a_ptr,
            k,
            0.0,
            c_ptr,
            n,
        )
        .map_err(|e| SklearsError::NumericalError(e.to_string()))
}

#[cfg(feature = "gpu_support")]
impl GpuMatrixOps for GpuArray<f64> {
    fn matmul(&self, other: &Self) -> Result<Self> {
        if self.shape.len() != 2 || other.shape.len() != 2 {
            return Err(SklearsError::InvalidOperation(
                "matmul requires 2-D arrays".into(),
            ));
        }
        let (m, k) = (self.shape[0], self.shape[1]);
        let (k2, n) = (other.shape[0], other.shape[1]);
        if k != k2 {
            return Err(SklearsError::ShapeMismatch {
                expected: format!("k={k}"),
                actual: format!("k2={k2}"),
            });
        }
        let c = GpuArray::<f64>::alloc_zeros(Arc::clone(&self.backend), vec![m, n])?;
        gemm_row_major_f64(&self.backend, m, k, n, self.ptr, other.ptr, c.ptr)?;
        Ok(c)
    }

    fn add(&self, other: &Self) -> Result<Self> {
        if self.shape != other.shape {
            return Err(SklearsError::ShapeMismatch {
                expected: format!("{:?}", self.shape),
                actual: format!("{:?}", other.shape),
            });
        }
        let out = GpuArray::<f64>::alloc_zeros(Arc::clone(&self.backend), self.shape.clone())?;
        self.backend
            .binary(BinaryOp::Add, self.ptr, other.ptr, out.ptr, self.size)
            .map_err(|e| SklearsError::NumericalError(e.to_string()))?;
        Ok(out)
    }

    fn mul(&self, other: &Self) -> Result<Self> {
        if self.shape != other.shape {
            return Err(SklearsError::ShapeMismatch {
                expected: format!("{:?}", self.shape),
                actual: format!("{:?}", other.shape),
            });
        }
        let out = GpuArray::<f64>::alloc_zeros(Arc::clone(&self.backend), self.shape.clone())?;
        self.backend
            .binary(BinaryOp::Mul, self.ptr, other.ptr, out.ptr, self.size)
            .map_err(|e| SklearsError::NumericalError(e.to_string()))?;
        Ok(out)
    }

    fn scale(&self, scalar: f32) -> Result<Self> {
        let mut data = self.to_cpu()?;
        for v in &mut data {
            *v *= scalar as f64;
        }
        Self::alloc_upload(Arc::clone(&self.backend), &data, self.shape.clone())
    }

    fn transpose(&self) -> Result<Self> {
        if self.shape.len() != 2 {
            return Err(SklearsError::InvalidOperation(
                "transpose requires 2-D array".into(),
            ));
        }
        let (m, n) = (self.shape[0], self.shape[1]);
        let src = self.to_cpu()?;
        let mut dst = vec![0.0f64; m * n];
        for i in 0..m {
            for j in 0..n {
                dst[j * m + i] = src[i * n + j];
            }
        }
        Self::alloc_upload(Arc::clone(&self.backend), &dst, vec![n, m])
    }
}

#[cfg(feature = "gpu_support")]
impl GpuMatrixOps for GpuArray<f32> {
    fn matmul(&self, other: &Self) -> Result<Self> {
        if self.shape.len() != 2 || other.shape.len() != 2 {
            return Err(SklearsError::InvalidOperation(
                "matmul requires 2-D arrays".into(),
            ));
        }
        let (m, k) = (self.shape[0], self.shape[1]);
        let (k2, n) = (other.shape[0], other.shape[1]);
        if k != k2 {
            return Err(SklearsError::ShapeMismatch {
                expected: format!("k={k}"),
                actual: format!("k2={k2}"),
            });
        }
        let a64: Vec<f64> = self.to_cpu()?.into_iter().map(|x| x as f64).collect();
        let b64: Vec<f64> = other.to_cpu()?.into_iter().map(|x| x as f64).collect();
        let a = GpuArray::<f64>::alloc_upload(Arc::clone(&self.backend), &a64, self.shape.clone())?;
        let b =
            GpuArray::<f64>::alloc_upload(Arc::clone(&self.backend), &b64, other.shape.clone())?;
        let c = GpuArray::<f64>::alloc_zeros(Arc::clone(&self.backend), vec![m, n])?;
        gemm_row_major_f64(&self.backend, m, k, n, a.ptr, b.ptr, c.ptr)?;
        let c32: Vec<f32> = c.to_cpu()?.into_iter().map(|x| x as f32).collect();
        Self::alloc_upload(Arc::clone(&self.backend), &c32, vec![m, n])
    }

    fn add(&self, other: &Self) -> Result<Self> {
        if self.shape != other.shape {
            return Err(SklearsError::ShapeMismatch {
                expected: format!("{:?}", self.shape),
                actual: format!("{:?}", other.shape),
            });
        }
        let out = GpuArray::<f32>::alloc_zeros(Arc::clone(&self.backend), self.shape.clone())?;
        self.backend
            .binary(BinaryOp::Add, self.ptr, other.ptr, out.ptr, self.size)
            .map_err(|e| SklearsError::NumericalError(e.to_string()))?;
        Ok(out)
    }

    fn mul(&self, other: &Self) -> Result<Self> {
        if self.shape != other.shape {
            return Err(SklearsError::ShapeMismatch {
                expected: format!("{:?}", self.shape),
                actual: format!("{:?}", other.shape),
            });
        }
        let out = GpuArray::<f32>::alloc_zeros(Arc::clone(&self.backend), self.shape.clone())?;
        self.backend
            .binary(BinaryOp::Mul, self.ptr, other.ptr, out.ptr, self.size)
            .map_err(|e| SklearsError::NumericalError(e.to_string()))?;
        Ok(out)
    }

    fn scale(&self, scalar: f32) -> Result<Self> {
        let mut data = self.to_cpu()?;
        for v in &mut data {
            *v *= scalar;
        }
        Self::alloc_upload(Arc::clone(&self.backend), &data, self.shape.clone())
    }

    fn transpose(&self) -> Result<Self> {
        if self.shape.len() != 2 {
            return Err(SklearsError::InvalidOperation(
                "transpose requires 2-D array".into(),
            ));
        }
        let (m, n) = (self.shape[0], self.shape[1]);
        let src = self.to_cpu()?;
        let mut dst = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                dst[j * m + i] = src[i * n + j];
            }
        }
        Self::alloc_upload(Arc::clone(&self.backend), &dst, vec![n, m])
    }
}

#[cfg(not(feature = "gpu_support"))]
impl<T> GpuMatrixOps for GpuArray<T> {
    fn matmul(&self, _other: &Self) -> Result<Self> {
        Err(SklearsError::from("GPU support not enabled"))
    }

    fn add(&self, _other: &Self) -> Result<Self> {
        Err(SklearsError::from("GPU support not enabled"))
    }

    fn mul(&self, _other: &Self) -> Result<Self> {
        Err(SklearsError::from("GPU support not enabled"))
    }

    fn scale(&self, _scalar: f32) -> Result<Self> {
        Err(SklearsError::from("GPU support not enabled"))
    }

    fn transpose(&self) -> Result<Self> {
        Err(SklearsError::from("GPU support not enabled"))
    }
}

// ─── GpuUtils ──────────────────────────────────────────────────────────────────

pub struct GpuUtils;

impl GpuUtils {
    pub fn is_gpu_available() -> bool {
        #[cfg(feature = "gpu_support")]
        {
            let registry = BackendRegistry::with_defaults();
            registry.available_kinds().iter().any(|k| k.is_gpu())
        }
        #[cfg(not(feature = "gpu_support"))]
        {
            false
        }
    }

    pub fn device_count() -> usize {
        #[cfg(feature = "gpu_support")]
        {
            GpuContext::new()
                .ok()
                .and_then(|ctx| ctx.backend().available_devices().ok())
                .map(|v| v.len())
                .unwrap_or(0)
        }
        #[cfg(not(feature = "gpu_support"))]
        {
            0
        }
    }

    #[cfg(feature = "gpu_support")]
    pub fn device_properties(device_id: usize) -> Result<GpuDeviceProperties> {
        let ctx = GpuContext::new()?;
        let devices = ctx
            .backend()
            .available_devices()
            .map_err(|e| SklearsError::NumericalError(e.to_string()))?;
        let info: DeviceInfo = devices
            .into_iter()
            .nth(device_id)
            .ok_or_else(|| SklearsError::InvalidInput(format!("no device at index {device_id}")))?;
        Ok(GpuDeviceProperties {
            device_id,
            name: info.name,
            total_memory: info.total_memory_bytes as usize,
            free_memory: info.total_memory_bytes as usize,
            compute_capability: (0, 0),
        })
    }

    #[cfg(not(feature = "gpu_support"))]
    pub fn device_properties(_device_id: usize) -> Result<GpuDeviceProperties> {
        Err(SklearsError::from("GPU support not enabled"))
    }
}

// ─── GpuDeviceProperties ───────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct GpuDeviceProperties {
    pub device_id: usize,
    pub name: String,
    pub total_memory: usize,
    pub free_memory: usize,
    pub compute_capability: (i32, i32),
}

// ─── MemoryTransferOpts / TransferStrategy ─────────────────────────────────────

pub struct MemoryTransferOpts;

impl MemoryTransferOpts {
    pub fn optimal_transfer_strategy(size_bytes: usize) -> TransferStrategy {
        if size_bytes < 1024 * 1024 {
            TransferStrategy::Synchronous
        } else if size_bytes < 100 * 1024 * 1024 {
            TransferStrategy::Asynchronous
        } else {
            TransferStrategy::Chunked {
                chunk_size: 10 * 1024 * 1024,
            }
        }
    }

    pub fn estimate_transfer_time(size_bytes: usize, pcie_bandwidth_gbps: f32) -> f32 {
        let bandwidth_bytes_per_sec = pcie_bandwidth_gbps * 1e9 / 8.0;
        size_bytes as f32 / bandwidth_bytes_per_sec
    }
}

#[derive(Debug, Clone, Copy)]
pub enum TransferStrategy {
    Synchronous,
    Asynchronous,
    Chunked { chunk_size: usize },
}

// ─── Tests ─────────────────────────────────────────────────────────────────────

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_availability() {
        let _gpu_available = GpuUtils::is_gpu_available();
        let _device_count = GpuUtils::device_count();
    }

    #[test]
    fn test_memory_transfer_strategy() {
        assert!(matches!(
            MemoryTransferOpts::optimal_transfer_strategy(1000),
            TransferStrategy::Synchronous
        ));
        assert!(matches!(
            MemoryTransferOpts::optimal_transfer_strategy(10_000_000),
            TransferStrategy::Asynchronous
        ));
        assert!(matches!(
            MemoryTransferOpts::optimal_transfer_strategy(200_000_000),
            TransferStrategy::Chunked { chunk_size: _ }
        ));
    }

    #[test]
    fn test_transfer_time_estimation() {
        let time = MemoryTransferOpts::estimate_transfer_time(1_000_000, 16.0);
        assert!(time > 0.0);
        assert!(time < 1.0);
    }

    #[cfg(feature = "gpu_support")]
    #[test]
    fn test_gpu_context_creation() {
        let ctx = GpuContext::new().expect("CpuBackend always succeeds");
        let info = ctx.memory_info().expect("memory_info should succeed");
        assert!(info.total > 0);
    }

    #[cfg(feature = "gpu_support")]
    #[test]
    fn test_gpu_array_roundtrip_f64() {
        let ctx = GpuContext::new().unwrap();
        let data = vec![1.0f64, 2.0, 3.0, 4.0];
        let arr = GpuArray::<f64>::from_slice(&ctx, &data).unwrap();
        assert_eq!(arr.len(), 4);
        let back = arr.to_cpu().unwrap();
        assert_eq!(back, data);
    }

    #[cfg(feature = "gpu_support")]
    #[test]
    fn test_gpu_matmul_2x2_f64() {
        use crate::gpu::GpuMatrixOps;
        let ctx = GpuContext::new().unwrap();
        // C = [[1,2],[3,4]] * [[5,6],[7,8]] = [[19,22],[43,50]]
        let a = GpuArray::<f64>::alloc_upload(ctx.backend(), &[1.0, 2.0, 3.0, 4.0], vec![2, 2])
            .unwrap();
        let b = GpuArray::<f64>::alloc_upload(ctx.backend(), &[5.0, 6.0, 7.0, 8.0], vec![2, 2])
            .unwrap();
        let c = a.matmul(&b).unwrap();
        let r = c.to_cpu().unwrap();
        assert!((r[0] - 19.0).abs() < 1e-10, "C[0,0]={}", r[0]);
        assert!((r[1] - 22.0).abs() < 1e-10, "C[0,1]={}", r[1]);
        assert!((r[2] - 43.0).abs() < 1e-10, "C[1,0]={}", r[2]);
        assert!((r[3] - 50.0).abs() < 1e-10, "C[1,1]={}", r[3]);
    }

    #[cfg(feature = "gpu_support")]
    #[test]
    fn test_gpu_array_scale_f64() {
        use crate::gpu::GpuMatrixOps;
        let ctx = GpuContext::new().unwrap();
        let arr = GpuArray::<f64>::from_slice(&ctx, &[1.0, 2.0, 3.0, 4.0]).unwrap();
        let scaled = arr.scale(2.0).unwrap();
        let back = scaled.to_cpu().unwrap();
        assert_eq!(back, vec![2.0, 4.0, 6.0, 8.0]);
    }
}
