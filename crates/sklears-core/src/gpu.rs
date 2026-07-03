//! GPU acceleration types, backed by the `oxicuda` crate family.
//!
//! # Wave A2 rewrite
//!
//! This module used to route through `oxicuda-backend`'s `BackendRegistry` /
//! `ComputeBackend` trait object, which (in this workspace) only ever
//! resolved to `CpuBackend` -- there is no factory anywhere that turns a
//! `BackendKind` into a real `Box<dyn ComputeBackend>` for CUDA/ROCm/Metal.
//! That made every "GPU" type in this file secretly CPU-only.
//!
//! This version wires directly to the concrete `oxicuda-driver` /
//! `oxicuda-blas` / `oxicuda-memory` crates: [`GpuBackend`] owns a real
//! `oxicuda_driver::Context` and `oxicuda_blas::BlasHandle`, and
//! [`GpuArray`] owns a real `oxicuda_memory::DeviceBuffer`. There is no CPU
//! fallback baked into these types any more: [`GpuBackend::detect`] returns
//! `Ok(None)` when no GPU/driver is present (e.g. on this crate's own macOS
//! development machine) instead of silently substituting a fake backend.
//!
//! `GpuContext` is kept as a type alias for [`GpuBackend`] so that downstream
//! crates written against the pre-migration name keep resolving; see the
//! doc comment on the alias for what actually changed.

use std::sync::Arc;

use crate::error::Result;
use crate::prelude::SklearsError;
use scirs2_core::ndarray::Array2;

#[cfg(feature = "gpu_support")]
use oxicuda_blas::{BlasHandle, GpuFloat, Layout, MatrixDesc, MatrixDescMut, Transpose};
#[cfg(feature = "gpu_support")]
use oxicuda_driver::{Context, Device};
#[cfg(feature = "gpu_support")]
use oxicuda_memory::DeviceBuffer;

#[cfg(feature = "gpu_support")]
fn cuda_err(e: oxicuda_driver::CudaError) -> SklearsError {
    SklearsError::NumericalError(e.to_string())
}

#[cfg(feature = "gpu_support")]
fn blas_err(e: oxicuda_blas::BlasError) -> SklearsError {
    SklearsError::NumericalError(e.to_string())
}

// ─── GpuBackend / GpuContext ────────────────────────────────────────────────────

/// Real GPU compute backend: a CUDA [`Context`] paired with a BLAS
/// [`BlasHandle`].
///
/// A `GpuBackend` value is existence-proof that a real GPU is present and
/// usable: it can only be constructed via [`detect`](Self::detect) or
/// [`with_device_id`](Self::with_device_id), both of which fully initialise
/// the driver, open a context, and build a BLAS handle before returning
/// `Some`. Code holding a `GpuBackend` never needs to branch on "is this
/// actually a GPU" the way the pre-migration `CpuBackend`-fallback design
/// did.
///
/// `context` and `blas` are `Arc`-wrapped so that `GpuBackend` -- and, by
/// extension, every [`GpuArray`] built from it -- is cheap to [`Clone`]:
/// cloning only bumps two reference counts, it never re-initialises the
/// driver, opens a new context, or allocates a new stream. `BlasHandle`
/// itself does not implement `Clone` (its `Stream` is a real driver
/// resource), which is why it is stored behind an `Arc` here rather than by
/// value.
#[cfg(feature = "gpu_support")]
#[derive(Clone)]
pub struct GpuBackend {
    context: Arc<Context>,
    blas: Arc<BlasHandle>,
    device_id: usize,
}

#[cfg(feature = "gpu_support")]
impl std::fmt::Debug for GpuBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuBackend")
            .field("device_id", &self.device_id)
            .finish_non_exhaustive()
    }
}

#[cfg(feature = "gpu_support")]
impl GpuBackend {
    /// Detects and initialises the best available GPU (the one with the
    /// most device memory).
    ///
    /// Attempts, in order: [`oxicuda_driver::init`] (loads the CUDA driver
    /// library and calls `cuInit`), [`oxicuda_driver::best_device`],
    /// [`Context::new`], and [`BlasHandle::new`]. If any step fails --
    /// including "the driver loaded but reported zero devices" -- the whole
    /// chain collapses to `Ok(None)`.
    ///
    /// On a machine with no NVIDIA GPU and no CUDA driver installed (for
    /// example, this crate's own macOS development environment), `init()`
    /// itself fails (`UnsupportedPlatform` on macOS, `NotInitialized`
    /// elsewhere) and `detect()` correctly, quietly returns `Ok(None)`. That
    /// is the expected outcome there, not a bug.
    ///
    /// # Errors
    ///
    /// This function is not expected to return `Err` in practice: every
    /// failure mode reachable from driver/device/context/handle
    /// initialisation is hardware- or environment-related and is folded into
    /// `Ok(None)`. The `Result` wrapper exists so callers can use `?`
    /// uniformly with the rest of this crate, and to leave room for a
    /// genuine logic-error variant in the future.
    pub fn detect() -> Result<Option<Self>> {
        if oxicuda_driver::init().is_err() {
            return Ok(None);
        }
        let Ok(Some(device)) = oxicuda_driver::best_device() else {
            return Ok(None);
        };
        Self::from_device(device)
    }

    /// Detects and initialises a specific GPU by ordinal, instead of letting
    /// [`detect`](Self::detect) pick the device with the most memory.
    ///
    /// Returns `Ok(None)` under the same circumstances as
    /// [`detect`](Self::detect) (missing driver, no such device, or
    /// context/handle creation failure) -- including when `device_id`
    /// doesn't fit in the `i32` ordinal the driver API expects.
    ///
    /// # Errors
    ///
    /// See [`detect`](Self::detect).
    pub fn with_device_id(device_id: usize) -> Result<Option<Self>> {
        if oxicuda_driver::init().is_err() {
            return Ok(None);
        }
        let Ok(ordinal) = i32::try_from(device_id) else {
            return Ok(None);
        };
        let Ok(device) = Device::get(ordinal) else {
            return Ok(None);
        };
        Self::from_device(device)
    }

    /// Shared construction logic for [`detect`](Self::detect) and
    /// [`with_device_id`](Self::with_device_id): creates a [`Context`] and a
    /// [`BlasHandle`] for an already-resolved [`Device`].
    fn from_device(device: Device) -> Result<Option<Self>> {
        let Ok(context) = Context::new(&device) else {
            return Ok(None);
        };
        let context = Arc::new(context);
        let Ok(blas) = BlasHandle::new(&context) else {
            return Ok(None);
        };
        let blas = Arc::new(blas);
        Ok(Some(Self {
            context,
            blas,
            device_id: device.ordinal() as usize,
        }))
    }

    /// Cheap existence check: `true` iff [`detect`](Self::detect) would
    /// return `Ok(Some(_))`.
    ///
    /// This still performs full driver/device/context/handle
    /// initialisation -- `oxicuda-driver` has no lighter-weight "just probe"
    /// primitive -- so it is "cheap" only relative to actually using the
    /// resulting backend, not relative to a syscall.
    pub fn is_available() -> bool {
        matches!(Self::detect(), Ok(Some(_)))
    }

    /// The ordinal of the device this backend is bound to.
    pub fn device_id(&self) -> usize {
        self.device_id
    }

    /// Always `true`: a `GpuBackend` value only ever exists once
    /// [`detect`](Self::detect) / [`with_device_id`](Self::with_device_id)
    /// has found a real GPU, so there is no CPU-backed variant to
    /// distinguish from.
    pub fn is_gpu(&self) -> bool {
        true
    }

    /// The CUDA context backing this handle.
    pub fn context(&self) -> &Arc<Context> {
        &self.context
    }

    /// The BLAS handle (stream + math mode + SM version) backing this
    /// backend.
    pub fn blas(&self) -> &BlasHandle {
        &self.blas
    }

    /// Blocks the calling thread until all outstanding work on this
    /// backend's context has completed.
    ///
    /// # Errors
    ///
    /// Returns [`SklearsError::NumericalError`] if the underlying
    /// `cuCtxSynchronize` call fails.
    pub fn synchronize(&self) -> Result<()> {
        self.context.synchronize().map_err(cuda_err)
    }

    /// Live free/total device memory for the GPU bound to this backend, via
    /// `cuMemGetInfo`.
    ///
    /// # Errors
    ///
    /// Returns [`SklearsError::NumericalError`] if the context cannot be
    /// made current or the driver call fails.
    pub fn memory_info(&self) -> Result<GpuMemoryInfo> {
        self.ensure_current()?;
        let (free, total) = oxicuda_driver::memory_info::device_memory_info().map_err(cuda_err)?;
        Ok(GpuMemoryInfo {
            free,
            total,
            used: total.saturating_sub(free),
        })
    }

    /// Makes this backend's CUDA context current on the calling thread.
    ///
    /// The CUDA driver API's legacy memory entry points (`cuMemAlloc`,
    /// `cuMemcpyHtoD`, `cuMemGetInfo`, ...) resolve against whichever
    /// context is current on the calling thread rather than through an
    /// explicit context handle; only stream-based launches (used by every
    /// `oxicuda-blas` kernel call) carry their context implicitly via the
    /// stream. Every [`GpuArray`] operation that allocates or copies device
    /// memory calls this first, so that multiple coexisting `GpuBackend`s
    /// (e.g. one per GPU in a multi-GPU process) do not silently race on
    /// "whichever context happened to be current last".
    fn ensure_current(&self) -> Result<()> {
        self.context.set_current().map_err(cuda_err)
    }
}

/// Compatibility alias: pre-migration code (and, transitionally, downstream
/// crates not yet updated for Wave A2 of the GPU migration) refers to this
/// type as `GpuContext`. The type itself is unchanged in spirit -- "a handle
/// to GPU compute" -- what changed is that construction became honest
/// (`Option`-returning [`GpuBackend::detect`] instead of an
/// always-succeeding `CpuBackend` fallback), so call sites that assumed
/// `GpuContext::new()` always succeeds need to be updated to handle `None`.
/// That downstream update is out of scope for this crate.
#[cfg(feature = "gpu_support")]
pub type GpuContext = GpuBackend;

#[cfg(not(feature = "gpu_support"))]
#[derive(Debug, Clone)]
pub struct GpuContext;

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

/// Non-`gpu_support`-build mirror of [`GpuBackend`]'s detection API, so that
/// code written against `GpuBackend::detect()` / `GpuBackend::is_available()`
/// compiles the same way regardless of whether the `gpu_support` feature is
/// enabled. Always reports "no GPU" -- this build has no GPU code compiled
/// into it at all.
#[cfg(not(feature = "gpu_support"))]
#[derive(Debug, Clone)]
pub struct GpuBackend;

#[cfg(not(feature = "gpu_support"))]
impl GpuBackend {
    pub fn detect() -> Result<Option<Self>> {
        Ok(None)
    }

    pub fn with_device_id(_device_id: usize) -> Result<Option<Self>> {
        Ok(None)
    }

    pub fn is_available() -> bool {
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

/// Device-resident array backed directly by an `oxicuda-memory`
/// [`DeviceBuffer`].
///
/// Every `GpuArray` that exists represents real GPU device memory: unlike
/// the pre-migration version, there is no CPU-backend fallback path through
/// this type, because a `GpuArray` can only be constructed from a
/// [`GpuBackend`], and a `GpuBackend` can only be constructed once
/// [`GpuBackend::detect`] (or [`GpuBackend::with_device_id`]) has found a
/// real GPU. `DeviceBuffer` owns its allocation and frees it on `Drop`, so
/// `GpuArray` does not need a manual `Drop` impl -- unlike the pre-migration
/// version, which manually tracked a raw pointer and called `backend.free()`.
///
/// `backend` is cloned (cheap: two `Arc` bumps, see [`GpuBackend`]) into
/// every `GpuArray` built from it, so that [`GpuMatrixOps`] methods
/// (`matmul`, `add`, `mul`, `scale`, `transpose`) can reach a `BlasHandle`
/// without changing their `&self`-only signatures.
#[cfg(feature = "gpu_support")]
pub struct GpuArray<T: Copy> {
    buf: DeviceBuffer<T>,
    shape: Vec<usize>,
    backend: GpuBackend,
}

#[cfg(not(feature = "gpu_support"))]
pub struct GpuArray<T> {
    _phantom: std::marker::PhantomData<T>,
}

#[cfg(feature = "gpu_support")]
impl<T: Copy> std::fmt::Debug for GpuArray<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuArray")
            .field("ptr", &format_args!("{:#x}", self.buf.as_device_ptr()))
            .field("shape", &self.shape)
            .field("len", &self.buf.len())
            .field("device_id", &self.backend.device_id())
            .finish()
    }
}

#[cfg(feature = "gpu_support")]
impl<T: Copy> GpuArray<T> {
    /// Allocates a device buffer, uploads `data`, and tags it with an
    /// explicit `shape`. Used internally by [`from_slice`](Self::from_slice)
    /// and [`from_array2`](Self::from_array2); also useful for tests and
    /// advanced callers that need an N-D shape from a flat host slice.
    pub(crate) fn from_slice_with_shape(
        backend: &GpuBackend,
        data: &[T],
        shape: Vec<usize>,
    ) -> Result<Self> {
        backend.ensure_current()?;
        let buf = DeviceBuffer::from_host(data).map_err(cuda_err)?;
        Ok(Self {
            buf,
            shape,
            backend: backend.clone(),
        })
    }

    /// Uploads a flat slice to the device as a 1-D array of length
    /// `data.len()`.
    pub fn from_slice(backend: &GpuBackend, data: &[T]) -> Result<Self> {
        Self::from_slice_with_shape(backend, data, vec![data.len()])
    }

    /// Uploads a 2-D `ndarray::Array2` to the device, preserving row-major
    /// element order.
    pub fn from_array2(backend: &GpuBackend, array: &Array2<T>) -> Result<Self>
    where
        T: Clone,
    {
        let data: Vec<T> = if array.is_standard_layout() {
            match array.as_slice() {
                Some(slice) => slice.to_vec(),
                // `is_standard_layout()` is documented to guarantee
                // `as_slice()` succeeds; fall back to the always-correct
                // iterator path rather than silently uploading zero
                // elements if that invariant is ever violated.
                None => array.iter().cloned().collect(),
            }
        } else {
            array.iter().cloned().collect()
        };
        let shape = vec![array.nrows(), array.ncols()];
        Self::from_slice_with_shape(backend, &data, shape)
    }

    /// Allocates a zero-initialised device buffer of the given `shape`.
    ///
    /// Zeroing happens entirely on-device via `cuMemsetD8`; no host-side
    /// buffer is allocated or transferred.
    pub fn zeros(backend: &GpuBackend, shape: &[usize]) -> Result<Self> {
        backend.ensure_current()?;
        let n: usize = shape.iter().product();
        let buf = DeviceBuffer::zeroed(n).map_err(cuda_err)?;
        Ok(Self {
            buf,
            shape: shape.to_vec(),
            backend: backend.clone(),
        })
    }

    /// Downloads the full contents of this array to a host `Vec<T>`, in the
    /// same flattened order used by [`shape`](Self::shape).
    pub fn to_cpu(&self) -> Result<Vec<T>>
    where
        T: Default,
    {
        self.backend.ensure_current()?;
        let mut out = vec![T::default(); self.buf.len()];
        self.buf.copy_to_host(&mut out).map_err(cuda_err)?;
        Ok(out)
    }

    /// Downloads this array as a 2-D `ndarray::Array2`. Requires
    /// [`shape`](Self::shape) to have exactly two dimensions.
    pub fn to_array2(&self) -> Result<Array2<T>>
    where
        T: Clone + Default,
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

    /// The logical shape of this array (e.g. `[rows, cols]` for a matrix).
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Total number of elements (product of [`shape`](Self::shape)).
    pub fn len(&self) -> usize {
        self.buf.len()
    }

    /// `true` iff this array has no elements.
    pub fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }

    /// Always `true`: every `GpuArray` in a `gpu_support` build is real
    /// device memory (see the type-level docs above).
    pub fn is_gpu_resident(&self) -> bool {
        true
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

/// `C = A * B` via `oxicuda-blas`'s native row-major GEMM.
///
/// Unlike the pre-migration implementation, this needs neither a
/// column-major transpose trick nor an f32-to-f64 upcast: `MatrixDesc`
/// supports `Layout::RowMajor` directly, and `oxicuda_blas::level3::gemm` is
/// generic over `T: GpuFloat` (both `f32` and `f64` implement it natively).
#[cfg(feature = "gpu_support")]
fn gpu_matmul<T: GpuFloat>(a: &GpuArray<T>, b: &GpuArray<T>) -> Result<GpuArray<T>> {
    if a.shape.len() != 2 || b.shape.len() != 2 {
        return Err(SklearsError::InvalidOperation(
            "matmul requires 2-D arrays".into(),
        ));
    }
    let (m, k) = (a.shape[0], a.shape[1]);
    let (k2, n) = (b.shape[0], b.shape[1]);
    if k != k2 {
        return Err(SklearsError::ShapeMismatch {
            expected: format!("k={k}"),
            actual: format!("k2={k2}"),
        });
    }

    a.backend.ensure_current()?;
    let a_desc =
        MatrixDesc::from_buffer(&a.buf, m as u32, k as u32, Layout::RowMajor).map_err(blas_err)?;
    let b_desc =
        MatrixDesc::from_buffer(&b.buf, k as u32, n as u32, Layout::RowMajor).map_err(blas_err)?;
    let mut c_buf = DeviceBuffer::<T>::zeroed(m * n).map_err(cuda_err)?;
    let mut c_desc = MatrixDescMut::from_buffer(&mut c_buf, m as u32, n as u32, Layout::RowMajor)
        .map_err(blas_err)?;

    oxicuda_blas::level3::gemm(
        a.backend.blas(),
        Transpose::NoTrans,
        Transpose::NoTrans,
        T::gpu_one(),
        &a_desc,
        &b_desc,
        T::gpu_zero(),
        &mut c_desc,
    )
    .map_err(blas_err)?;

    Ok(GpuArray {
        buf: c_buf,
        shape: vec![m, n],
        backend: a.backend.clone(),
    })
}

/// `C[i] = A[i] + B[i]`, via `oxicuda_blas::elementwise::add` -- a real GPU
/// kernel, not a CPU round-trip.
#[cfg(feature = "gpu_support")]
fn gpu_add<T: GpuFloat>(a: &GpuArray<T>, b: &GpuArray<T>) -> Result<GpuArray<T>> {
    if a.shape != b.shape {
        return Err(SklearsError::ShapeMismatch {
            expected: format!("{:?}", a.shape),
            actual: format!("{:?}", b.shape),
        });
    }
    a.backend.ensure_current()?;
    let n = a.buf.len();
    let mut out = DeviceBuffer::<T>::zeroed(n).map_err(cuda_err)?;
    oxicuda_blas::elementwise::add(a.backend.blas(), n as u32, &a.buf, &b.buf, &mut out)
        .map_err(blas_err)?;
    Ok(GpuArray {
        buf: out,
        shape: a.shape.clone(),
        backend: a.backend.clone(),
    })
}

/// `C[i] = A[i] * B[i]` (Hadamard product), via
/// `oxicuda_blas::elementwise::mul` -- a real GPU kernel, not a CPU
/// round-trip.
#[cfg(feature = "gpu_support")]
fn gpu_mul<T: GpuFloat>(a: &GpuArray<T>, b: &GpuArray<T>) -> Result<GpuArray<T>> {
    if a.shape != b.shape {
        return Err(SklearsError::ShapeMismatch {
            expected: format!("{:?}", a.shape),
            actual: format!("{:?}", b.shape),
        });
    }
    a.backend.ensure_current()?;
    let n = a.buf.len();
    let mut out = DeviceBuffer::<T>::zeroed(n).map_err(cuda_err)?;
    oxicuda_blas::elementwise::mul(a.backend.blas(), n as u32, &a.buf, &b.buf, &mut out)
        .map_err(blas_err)?;
    Ok(GpuArray {
        buf: out,
        shape: a.shape.clone(),
        backend: a.backend.clone(),
    })
}

/// `x = alpha * x`, via `oxicuda_blas::level1::scal`.
///
/// `scal` scales its buffer in place, but [`GpuMatrixOps::scale`] returns a
/// *new* array rather than mutating `self`, so this first makes an on-device
/// copy (device-to-device, no host round-trip) and scales that copy.
#[cfg(feature = "gpu_support")]
fn gpu_scale<T: GpuFloat>(a: &GpuArray<T>, scalar: T) -> Result<GpuArray<T>> {
    a.backend.ensure_current()?;
    let n = a.buf.len();
    let mut out = DeviceBuffer::<T>::alloc(n).map_err(cuda_err)?;
    out.copy_from_device(&a.buf).map_err(cuda_err)?;
    oxicuda_blas::level1::scal(a.backend.blas(), n as u32, scalar, &mut out, 1)
        .map_err(blas_err)?;
    Ok(GpuArray {
        buf: out,
        shape: a.shape.clone(),
        backend: a.backend.clone(),
    })
}

/// 2-D matrix transpose via a host round-trip.
///
/// As of `oxicuda-blas` 0.4.0 there is no on-device transpose primitive
/// (only `MatrixDesc::effective_dims`, which just re-labels the logical
/// extents of a GEMM operand without physically moving any data), so this
/// downloads, transposes in host memory, and re-uploads. This mirrors the
/// pre-migration behaviour and is not a regression.
#[cfg(feature = "gpu_support")]
fn gpu_transpose<T: GpuFloat>(a: &GpuArray<T>) -> Result<GpuArray<T>> {
    if a.shape.len() != 2 {
        return Err(SklearsError::InvalidOperation(
            "transpose requires 2-D array".into(),
        ));
    }
    a.backend.ensure_current()?;
    let (m, n) = (a.shape[0], a.shape[1]);
    let mut src = vec![T::gpu_zero(); m * n];
    a.buf.copy_to_host(&mut src).map_err(cuda_err)?;
    let mut dst = vec![T::gpu_zero(); m * n];
    for i in 0..m {
        for j in 0..n {
            dst[j * m + i] = src[i * n + j];
        }
    }
    let buf = DeviceBuffer::from_host(&dst).map_err(cuda_err)?;
    Ok(GpuArray {
        buf,
        shape: vec![n, m],
        backend: a.backend.clone(),
    })
}

#[cfg(feature = "gpu_support")]
impl GpuMatrixOps for GpuArray<f64> {
    fn matmul(&self, other: &Self) -> Result<Self> {
        gpu_matmul(self, other)
    }

    fn add(&self, other: &Self) -> Result<Self> {
        gpu_add(self, other)
    }

    fn mul(&self, other: &Self) -> Result<Self> {
        gpu_mul(self, other)
    }

    fn scale(&self, scalar: f32) -> Result<Self> {
        gpu_scale(self, scalar as f64)
    }

    fn transpose(&self) -> Result<Self> {
        gpu_transpose(self)
    }
}

#[cfg(feature = "gpu_support")]
impl GpuMatrixOps for GpuArray<f32> {
    fn matmul(&self, other: &Self) -> Result<Self> {
        gpu_matmul(self, other)
    }

    fn add(&self, other: &Self) -> Result<Self> {
        gpu_add(self, other)
    }

    fn mul(&self, other: &Self) -> Result<Self> {
        gpu_mul(self, other)
    }

    fn scale(&self, scalar: f32) -> Result<Self> {
        gpu_scale(self, scalar)
    }

    fn transpose(&self) -> Result<Self> {
        gpu_transpose(self)
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
            GpuBackend::is_available()
        }
        #[cfg(not(feature = "gpu_support"))]
        {
            false
        }
    }

    pub fn device_count() -> usize {
        #[cfg(feature = "gpu_support")]
        {
            if oxicuda_driver::init().is_err() {
                return 0;
            }
            Device::count().map(|c| c.max(0) as usize).unwrap_or(0)
        }
        #[cfg(not(feature = "gpu_support"))]
        {
            0
        }
    }

    #[cfg(feature = "gpu_support")]
    pub fn device_properties(device_id: usize) -> Result<GpuDeviceProperties> {
        oxicuda_driver::init().map_err(cuda_err)?;
        let ordinal = i32::try_from(device_id).map_err(|_| {
            SklearsError::InvalidInput(format!("device id {device_id} out of range"))
        })?;
        let device = Device::get(ordinal).map_err(|e| {
            SklearsError::InvalidInput(format!("no device at index {device_id}: {e}"))
        })?;
        let info = device.info().map_err(cuda_err)?;

        // `DeviceInfo::total_memory_bytes` is a static property queried at
        // device-info time. For a *live* free-memory figure we additionally
        // try `cuMemGetInfo`, which requires a current context; if that
        // query fails (e.g. no context has been made current on this thread
        // yet), fall back to reporting total memory as a conservative
        // (upper-bound) estimate of free memory rather than fabricating a
        // number.
        let free_memory = oxicuda_driver::memory_info::device_memory_info()
            .map(|(free, _total)| free)
            .unwrap_or(info.total_memory_bytes);

        Ok(GpuDeviceProperties {
            device_id,
            name: info.name,
            total_memory: info.total_memory_bytes,
            free_memory,
            compute_capability: info.compute_capability,
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

    /// `detect()` must never panic and must never hard-`Err` just because
    /// there is no GPU/driver present. On this crate's dev/CI machines
    /// (macOS, no NVIDIA GPU) `Ok(None)` is the expected, correct result --
    /// we deliberately do not assert `is_none()` here so this test also
    /// passes unchanged on a real GPU runner.
    #[cfg(feature = "gpu_support")]
    #[test]
    fn test_gpu_backend_detect_is_ok() {
        let result = GpuBackend::detect();
        assert!(
            result.is_ok(),
            "detect() must not hard-error on missing GPU/driver: {result:?}"
        );
    }

    #[cfg(feature = "gpu_support")]
    #[test]
    fn test_gpu_backend_is_available_matches_detect() {
        let available = GpuBackend::is_available();
        let detected = GpuBackend::detect()
            .expect("detect() should not hard-error")
            .is_some();
        assert_eq!(available, detected);
    }

    #[cfg(feature = "gpu_support")]
    #[test]
    fn test_gpu_array_roundtrip_and_matmul_f64() {
        // Requires real hardware; gracefully skip on machines (like this
        // crate's own dev/CI environment) where `detect()` legitimately
        // finds nothing. This keeps the test meaningful on real GPU runners
        // while remaining green everywhere else.
        let Some(backend) = GpuBackend::detect().expect("detect() should not hard-error") else {
            eprintln!("skipping test_gpu_array_roundtrip_and_matmul_f64: no GPU detected");
            return;
        };

        let data = vec![1.0f64, 2.0, 3.0, 4.0];
        let arr = GpuArray::<f64>::from_slice(&backend, &data).expect("from_slice");
        assert_eq!(arr.len(), 4);
        let back = arr.to_cpu().expect("to_cpu");
        assert_eq!(back, data);

        // C = [[1,2],[3,4]] * [[5,6],[7,8]] = [[19,22],[43,50]]
        let a = GpuArray::<f64>::from_slice_with_shape(&backend, &[1.0, 2.0, 3.0, 4.0], vec![2, 2])
            .expect("a");
        let b = GpuArray::<f64>::from_slice_with_shape(&backend, &[5.0, 6.0, 7.0, 8.0], vec![2, 2])
            .expect("b");
        let c = a.matmul(&b).expect("matmul");
        let r = c.to_cpu().expect("to_cpu");
        assert!((r[0] - 19.0).abs() < 1e-10, "C[0,0]={}", r[0]);
        assert!((r[1] - 22.0).abs() < 1e-10, "C[0,1]={}", r[1]);
        assert!((r[2] - 43.0).abs() < 1e-10, "C[1,0]={}", r[2]);
        assert!((r[3] - 50.0).abs() < 1e-10, "C[1,1]={}", r[3]);
    }

    /// Native f32 GEMM (no f32-to-f64 upcast hack). Gracefully skips without
    /// hardware, like the f64 test above.
    #[cfg(feature = "gpu_support")]
    #[test]
    fn test_gpu_matmul_2x2_f32() {
        let Some(backend) = GpuBackend::detect().expect("detect() should not hard-error") else {
            eprintln!("skipping test_gpu_matmul_2x2_f32: no GPU detected");
            return;
        };
        let a = GpuArray::<f32>::from_slice_with_shape(&backend, &[1.0, 2.0, 3.0, 4.0], vec![2, 2])
            .expect("a");
        let b = GpuArray::<f32>::from_slice_with_shape(&backend, &[5.0, 6.0, 7.0, 8.0], vec![2, 2])
            .expect("b");
        let c = a.matmul(&b).expect("matmul");
        let r = c.to_cpu().expect("to_cpu");
        assert!((r[0] - 19.0).abs() < 1e-4, "C[0,0]={}", r[0]);
        assert!((r[1] - 22.0).abs() < 1e-4, "C[0,1]={}", r[1]);
        assert!((r[2] - 43.0).abs() < 1e-4, "C[1,0]={}", r[2]);
        assert!((r[3] - 50.0).abs() < 1e-4, "C[1,1]={}", r[3]);
    }

    #[cfg(feature = "gpu_support")]
    #[test]
    fn test_gpu_array_scale_f64() {
        let Some(backend) = GpuBackend::detect().expect("detect() should not hard-error") else {
            eprintln!("skipping test_gpu_array_scale_f64: no GPU detected");
            return;
        };
        let arr = GpuArray::<f64>::from_slice(&backend, &[1.0, 2.0, 3.0, 4.0]).expect("from_slice");
        let scaled = arr.scale(2.0).expect("scale");
        let back = scaled.to_cpu().expect("to_cpu");
        assert_eq!(back, vec![2.0, 4.0, 6.0, 8.0]);
    }

    /// Native on-device add/mul (via `oxicuda_blas::elementwise`) plus the
    /// host-round-trip transpose, exercised together to keep the test count
    /// down. Gracefully skips without hardware.
    #[cfg(feature = "gpu_support")]
    #[test]
    fn test_gpu_array_add_mul_transpose_f64() {
        let Some(backend) = GpuBackend::detect().expect("detect() should not hard-error") else {
            eprintln!("skipping test_gpu_array_add_mul_transpose_f64: no GPU detected");
            return;
        };
        let a = GpuArray::<f64>::from_slice(&backend, &[1.0, 2.0, 3.0, 4.0]).expect("a");
        let b = GpuArray::<f64>::from_slice(&backend, &[10.0, 20.0, 30.0, 40.0]).expect("b");

        let sum = a.add(&b).expect("add");
        assert_eq!(sum.to_cpu().expect("to_cpu"), vec![11.0, 22.0, 33.0, 44.0]);

        let prod = a.mul(&b).expect("mul");
        assert_eq!(
            prod.to_cpu().expect("to_cpu"),
            vec![10.0, 40.0, 90.0, 160.0]
        );

        let m = GpuArray::<f64>::from_slice_with_shape(
            &backend,
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 3],
        )
        .expect("m");
        let t = m.transpose().expect("transpose");
        assert_eq!(t.shape(), &[3, 2]);
        assert_eq!(
            t.to_cpu().expect("to_cpu"),
            vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
        );
    }

    #[cfg(feature = "gpu_support")]
    #[test]
    fn test_gpu_device_properties_when_available() {
        if GpuUtils::device_count() == 0 {
            eprintln!("skipping test_gpu_device_properties_when_available: no GPU detected");
            return;
        }
        let props = GpuUtils::device_properties(0).expect("device 0 should be queryable");
        assert_eq!(props.device_id, 0);
        assert!(!props.name.is_empty());
        assert!(props.total_memory > 0);
    }
}
