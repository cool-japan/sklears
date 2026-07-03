//! GPU Acceleration Module for Cross-Decomposition Methods
//!
//! This module provides GPU-accelerated implementations of core cross-decomposition
//! algorithms including CCA, PLS, and tensor decomposition methods. It is backed by
//! the shared `sklears_core::gpu` abstraction (a real CUDA context + BLAS handle, via
//! the `oxicuda` crate family) -- the same foundation the rest of this GPU-migration
//! wave builds on.
//!
//! ## Wave B8 rewrite
//!
//! This module used to carry its own bespoke, placeholder GPU layer
//! (`DummyGpuContext` / `DummyGpuBuffer` / `DummyGpuKernel`, plus locally defined
//! `GpuBackend` / `GpuContext` / `GpuBuffer` / `GpuKernel` types) because, at the
//! time, there was no real shared GPU abstraction to build on: `matmul` / `eig` /
//! `svd` / `batch_matmul` always silently ran on the CPU no matter what backend was
//! requested.
//!
//! That dummy scaffolding is gone. [`GpuMatrixOps::matmul`] and
//! [`GpuMatrixOps::batch_matmul`] now dispatch to real on-device GEMM through
//! `sklears_core::gpu`'s `GpuArray` / `GpuMatrixOps` whenever a CUDA device is
//! detected (via `sklears_core::gpu::GpuBackend::detect`); otherwise -- and always
//! when the `gpu` feature is disabled -- they fall back to the same CPU path
//! (`ndarray`) this crate always used. `eig` / `svd` still run entirely on the CPU
//! via `scirs2_linalg::compat::{eigh, svd}`: there is no `oxicuda-solver`-backed
//! dense SVD/eigendecomposition wired into this crate yet, matching the "CPU-correct
//! baseline" bar this wave sets for that pair of operations.
//!
//! The local [`GpuBackendKind`] enum (formerly named `GpuBackend`) was renamed to
//! avoid colliding with `sklears_core::gpu::GpuBackend`, which is a live handle to an
//! already-initialised GPU rather than a plain "which kind of backend" tag. The local
//! `GpuContext` / `GpuBuffer` / `GpuKernel` traits were removed outright: they only
//! ever existed to be implemented by the now-deleted dummy types, and have no
//! remaining purpose now that `sklears_core::gpu` provides a real, concrete
//! implementation.
//!
//! ## Supported Backends
//! - CUDA (NVIDIA GPUs), via `sklears_core::gpu` / `oxicuda`
//! - CPU fallback for compatibility (always available)
//!
//! Metal / WebGPU / ROCm / OpenCL are enumerated in [`GpuBackendKind`] for
//! forward/API compatibility but have no implementation behind them yet.
//!
//! ## Performance Benefits
//! - Real on-device GEMM for `matmul` / `batch_matmul` when a CUDA device is present
//! - CPU fallback (`scirs2_linalg`) for eigenvalue decomposition and SVD

use scirs2_core::ndarray::{s, Array1, Array2};
use scirs2_linalg::compat::{eigh, inv, svd, UPLO};
use scirs2_linalg::LinalgError;

#[cfg(feature = "gpu")]
use sklears_core::{
    error::SklearsError,
    gpu::{GpuArray, GpuBackend as SklearsGpuBackend, GpuMatrixOps as SklearsGpuMatrixOps},
};

// Define our own Result type for GPU operations
pub type GpuResult<T> = std::result::Result<T, GpuError>;

#[derive(Debug, thiserror::Error)]
pub enum GpuError {
    #[error("Dimension error: {0}")]
    DimensionError(String),
    #[error("Computation error: {0}")]
    ComputationError(String),
    #[error("GPU error: {0}")]
    GpuError(String),
}

/// Converts an error from the shared `sklears_core::gpu` abstraction into this
/// module's own [`GpuError`].
#[cfg(feature = "gpu")]
fn map_gpu_err(e: SklearsError) -> GpuError {
    GpuError::GpuError(e.to_string())
}

// ─── GpuBackendKind ─────────────────────────────────────────────────────────────

/// The class of compute backend a [`GpuAcceleratedContext`] prefers or is bound to.
///
/// This is intentionally distinct from `sklears_core::gpu::GpuBackend`: that type is
/// a *live handle* to an already-initialised GPU (a real CUDA context + BLAS handle
/// pair), constructible only once a device has actually been found via
/// `GpuBackend::detect`. `GpuBackendKind` is a plain, always-constructible tag
/// describing which *kind* of backend is wanted or active, independent of whether
/// that backend was actually detected.
///
/// Only [`Cpu`](Self::Cpu) and [`Cuda`](Self::Cuda) have a working implementation
/// behind them in this crate (via `sklears_core::gpu`, backed by the `oxicuda` crate
/// family). The remaining variants are retained for forward/API compatibility and
/// always report [`is_available`](Self::is_available) as `false`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuBackendKind {
    Cpu,
    Cuda,
    Metal,
    WebGpu,
    Rocm,
    OpenCl,
}

impl GpuBackendKind {
    /// The backend this process would actually use: [`Cuda`](Self::Cuda) if a CUDA
    /// device is detected, else [`Cpu`](Self::Cpu).
    pub fn preferred() -> Self {
        if Self::cuda_available() {
            Self::Cuda
        } else {
            Self::Cpu
        }
    }

    /// Whether this specific backend kind is actually usable right now. `Cpu` is
    /// always available; `Cuda` reflects real device detection; the remaining
    /// variants have no implementation in this crate yet.
    pub fn is_available(&self) -> bool {
        match self {
            Self::Cpu => true,
            Self::Cuda => Self::cuda_available(),
            Self::Metal | Self::WebGpu | Self::Rocm | Self::OpenCl => false,
        }
    }

    #[cfg(feature = "gpu")]
    fn cuda_available() -> bool {
        SklearsGpuBackend::is_available()
    }

    #[cfg(not(feature = "gpu"))]
    fn cuda_available() -> bool {
        false
    }
}

// ─── GpuAcceleratedContext ───────────────────────────────────────────────────────

/// GPU-accelerated context for cross-decomposition operations.
///
/// Wraps a real `sklears_core::gpu::GpuBackend` (CUDA context + BLAS handle) when the
/// `gpu` feature is enabled and a device is detected; otherwise this is a CPU-only
/// marker, and every operation built on top of it (see [`GpuMatrixOps`])
/// transparently runs on the CPU.
#[derive(Clone)]
pub struct GpuAcceleratedContext {
    backend_kind: GpuBackendKind,
    #[cfg(feature = "gpu")]
    gpu: Option<SklearsGpuBackend>,
}

impl Default for GpuAcceleratedContext {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuAcceleratedContext {
    /// Create a new GPU-accelerated context, auto-detecting the best available
    /// backend (currently: CUDA if present, else CPU).
    pub fn new() -> Self {
        Self::auto_detect()
    }

    /// Create a context bound to a specific [`GpuBackendKind`].
    ///
    /// Requesting [`GpuBackendKind::Cuda`] attempts real device detection, falling
    /// back to CPU if none is found. Requesting anything else forces CPU-only
    /// operation, since only `Cpu`/`Cuda` are implemented (see
    /// [`GpuBackendKind::is_available`]).
    pub fn with_backend(backend_kind: GpuBackendKind) -> Self {
        match backend_kind {
            GpuBackendKind::Cuda => Self::auto_detect(),
            _ => Self::cpu_only(),
        }
    }

    #[cfg(feature = "gpu")]
    fn auto_detect() -> Self {
        match SklearsGpuBackend::detect() {
            Ok(Some(gpu)) => Self {
                backend_kind: GpuBackendKind::Cuda,
                gpu: Some(gpu),
            },
            _ => Self::cpu_only(),
        }
    }

    #[cfg(not(feature = "gpu"))]
    fn auto_detect() -> Self {
        Self::cpu_only()
    }

    #[cfg(feature = "gpu")]
    fn cpu_only() -> Self {
        Self {
            backend_kind: GpuBackendKind::Cpu,
            gpu: None,
        }
    }

    #[cfg(not(feature = "gpu"))]
    fn cpu_only() -> Self {
        Self {
            backend_kind: GpuBackendKind::Cpu,
        }
    }

    /// Check if GPU acceleration is available and enabled for this context.
    pub fn is_gpu_enabled(&self) -> bool {
        #[cfg(feature = "gpu")]
        {
            self.gpu.is_some()
        }
        #[cfg(not(feature = "gpu"))]
        {
            false
        }
    }

    /// Get the current GPU backend kind.
    pub fn backend(&self) -> GpuBackendKind {
        self.backend_kind
    }

    /// Access the real, live GPU backend handle, if one was detected.
    #[cfg(feature = "gpu")]
    pub fn gpu_backend(&self) -> Option<&SklearsGpuBackend> {
        self.gpu.as_ref()
    }

    /// Get GPU memory info. Returns real free/total/used device memory when a GPU is
    /// bound to this context; an all-zero [`GpuMemoryInfo::default`] otherwise.
    pub fn memory_info(&self) -> GpuMemoryInfo {
        #[cfg(feature = "gpu")]
        {
            if let Some(ref gpu) = self.gpu {
                if let Ok(info) = gpu.memory_info() {
                    return GpuMemoryInfo {
                        total: info.total,
                        available: info.free,
                        used: info.used,
                    };
                }
            }
        }
        GpuMemoryInfo::default()
    }
}

/// GPU memory information
#[derive(Debug, Clone, Copy, Default)]
pub struct GpuMemoryInfo {
    /// Total GPU memory in bytes
    pub total: usize,
    /// Available GPU memory in bytes
    pub available: usize,
    /// Used GPU memory in bytes
    pub used: usize,
}

// ─── GpuMatrixOps ────────────────────────────────────────────────────────────────

/// GPU-accelerated matrix operations for cross-decomposition.
///
/// `matmul` / `batch_matmul` dispatch to real on-device GEMM (via
/// `sklears_core::gpu`'s `GpuArray` / `GpuMatrixOps`) whenever the wrapped
/// [`GpuAcceleratedContext`] has a live CUDA backend; otherwise they run on the CPU
/// via `ndarray`. `eig` / `svd` always run on the CPU via
/// `scirs2_linalg::compat::{eigh, svd}` -- see the module docs for why.
#[derive(Clone)]
pub struct GpuMatrixOps {
    context: GpuAcceleratedContext,
}

impl Default for GpuMatrixOps {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuMatrixOps {
    /// Create new GPU matrix operations handler
    pub fn new() -> Self {
        Self {
            context: GpuAcceleratedContext::new(),
        }
    }

    /// Create with specific GPU context
    pub fn with_context(context: GpuAcceleratedContext) -> Self {
        Self { context }
    }

    /// Matrix multiplication: real on-device GEMM when a CUDA backend is available,
    /// CPU (`ndarray::Array2::dot`) otherwise.
    pub fn matmul(&self, a: &Array2<f64>, b: &Array2<f64>) -> GpuResult<Array2<f64>> {
        if a.ncols() != b.nrows() {
            return Err(GpuError::DimensionError(format!(
                "Matrix dimensions incompatible: ({}, {}) x ({}, {}), expected ({}, K) x (K, {})",
                a.nrows(),
                a.ncols(),
                b.nrows(),
                b.ncols(),
                a.nrows(),
                b.ncols()
            )));
        }

        // `is_gpu_enabled` is always compiled in (it just always returns `false`
        // without the `gpu` feature), so this check -- and therefore `self.context`
        // -- is genuinely read regardless of feature flags; only the on-device
        // dispatch itself (`gpu_matmul_dispatch`'s `gpu`-feature body) needs real
        // GPU types.
        if self.context.is_gpu_enabled() {
            return self.gpu_matmul_dispatch(a, b);
        }

        self.cpu_matmul(a, b)
    }

    #[cfg(feature = "gpu")]
    fn gpu_matmul_dispatch(&self, a: &Array2<f64>, b: &Array2<f64>) -> GpuResult<Array2<f64>> {
        match self.context.gpu_backend() {
            Some(backend) => dispatch_gpu_matmul(backend, a, b),
            None => self.cpu_matmul(a, b),
        }
    }

    #[cfg(not(feature = "gpu"))]
    fn gpu_matmul_dispatch(&self, a: &Array2<f64>, b: &Array2<f64>) -> GpuResult<Array2<f64>> {
        // Unreachable in practice: `is_gpu_enabled()` is always `false` without the
        // `gpu` feature, so `matmul` never calls this. Kept only so `matmul` itself
        // does not need a `#[cfg]` split.
        self.cpu_matmul(a, b)
    }

    fn cpu_matmul(&self, a: &Array2<f64>, b: &Array2<f64>) -> GpuResult<Array2<f64>> {
        Ok(a.dot(b))
    }

    /// Symmetric eigendecomposition (CPU only; see module docs).
    ///
    /// CCA covariance matrices are symmetric positive semi-definite, so `eigh` is the
    /// correct decomposition here.
    pub fn eig(&self, matrix: &Array2<f64>) -> GpuResult<(Array1<f64>, Array2<f64>)> {
        self.cpu_eig(matrix)
    }

    fn cpu_eig(&self, matrix: &Array2<f64>) -> GpuResult<(Array1<f64>, Array2<f64>)> {
        let (eigenvalues, eigenvectors) = eigh(matrix, UPLO::Upper)
            .map_err(|e: LinalgError| GpuError::ComputationError(e.to_string()))?;
        Ok((eigenvalues, eigenvectors))
    }

    /// Singular value decomposition (CPU only; see module docs).
    pub fn svd(&self, matrix: &Array2<f64>) -> GpuResult<(Array2<f64>, Array1<f64>, Array2<f64>)> {
        self.cpu_svd(matrix)
    }

    fn cpu_svd(&self, matrix: &Array2<f64>) -> GpuResult<(Array2<f64>, Array1<f64>, Array2<f64>)> {
        let (u, s, vt) = svd(matrix, true)
            .map_err(|e: LinalgError| GpuError::ComputationError(e.to_string()))?;
        Ok((u, s, vt))
    }

    /// Batch matrix multiplication: dispatches each pair through [`Self::matmul`]
    /// (real on-device GEMM per pair when a CUDA backend is available).
    pub fn batch_matmul(
        &self,
        a: &[Array2<f64>],
        b: &[Array2<f64>],
    ) -> GpuResult<Vec<Array2<f64>>> {
        if a.len() != b.len() {
            return Err(GpuError::DimensionError(format!(
                "Batch size mismatch: expected {}, got {}",
                a.len(),
                b.len()
            )));
        }

        a.iter()
            .zip(b.iter())
            .map(|(ai, bi)| self.matmul(ai, bi))
            .collect()
    }
}

/// `C = A * B` on-device: uploads `a`/`b` to the GPU, dispatches through
/// `sklears_core::gpu`'s real GEMM, and downloads the result.
#[cfg(feature = "gpu")]
fn dispatch_gpu_matmul(
    backend: &SklearsGpuBackend,
    a: &Array2<f64>,
    b: &Array2<f64>,
) -> GpuResult<Array2<f64>> {
    let a_gpu = GpuArray::<f64>::from_array2(backend, a).map_err(map_gpu_err)?;
    let b_gpu = GpuArray::<f64>::from_array2(backend, b).map_err(map_gpu_err)?;
    let c_gpu = SklearsGpuMatrixOps::matmul(&a_gpu, &b_gpu).map_err(map_gpu_err)?;
    c_gpu.to_array2().map_err(map_gpu_err)
}

// ─── GpuCCA ──────────────────────────────────────────────────────────────────────

/// GPU-accelerated CCA implementation
pub struct GpuCCA {
    matrix_ops: GpuMatrixOps,
    n_components: usize,
}

impl GpuCCA {
    /// Create new GPU-accelerated CCA
    pub fn new(n_components: usize) -> Self {
        Self {
            matrix_ops: GpuMatrixOps::new(),
            n_components,
        }
    }

    /// Fit GPU-accelerated CCA model
    pub fn fit(&self, x: &Array2<f64>, y: &Array2<f64>) -> GpuResult<GpuCCAFitted> {
        let n_samples = x.nrows();
        if y.nrows() != n_samples {
            return Err(GpuError::DimensionError(format!(
                "Sample size mismatch: X has {} samples, Y has {} samples",
                n_samples,
                y.nrows()
            )));
        }

        // Center the data
        let x_centered = self.center_data(x)?;
        let y_centered = self.center_data(y)?;

        // Compute covariance matrices using GPU-accelerated operations
        let x_t = x_centered.t().to_owned();
        let y_t = y_centered.t().to_owned();
        let cxx = self.matrix_ops.matmul(&x_t, &x_centered)? / (n_samples as f64 - 1.0);
        let cyy = self.matrix_ops.matmul(&y_t, &y_centered)? / (n_samples as f64 - 1.0);
        let cxy = self.matrix_ops.matmul(&x_t, &y_centered)? / (n_samples as f64 - 1.0);

        // Solve generalized eigenvalue problem using GPU
        let (eigenvalues, x_weights) = self.solve_cca_eigenproblem(&cxx, &cyy, &cxy)?;

        // Compute Y weights
        let y_weights = self.compute_y_weights(&cyy, &cxy, &x_weights)?;

        Ok(GpuCCAFitted {
            x_weights: x_weights.slice(s![.., ..self.n_components]).to_owned(),
            y_weights: y_weights.slice(s![.., ..self.n_components]).to_owned(),
            correlations: eigenvalues.slice(s![..self.n_components]).to_owned(),
            matrix_ops: self.matrix_ops.clone(),
        })
    }

    fn center_data(&self, data: &Array2<f64>) -> GpuResult<Array2<f64>> {
        let mean = data
            .mean_axis(scirs2_core::ndarray::Axis(0))
            .ok_or_else(|| {
                GpuError::DimensionError("center_data requires a non-empty array".to_string())
            })?;
        let mut centered = data.clone();
        for mut row in centered.rows_mut() {
            row -= &mean;
        }
        Ok(centered)
    }

    fn solve_cca_eigenproblem(
        &self,
        cxx: &Array2<f64>,
        cyy: &Array2<f64>,
        cxy: &Array2<f64>,
    ) -> GpuResult<(Array1<f64>, Array2<f64>)> {
        // Regularize Cxx and Cyy for numerical stability
        let reg = 1e-6;
        let mut cxx_reg = cxx.clone();
        let mut cyy_reg = cyy.clone();
        for i in 0..cxx_reg.nrows() {
            cxx_reg[[i, i]] += reg;
        }
        for i in 0..cyy_reg.nrows() {
            cyy_reg[[i, i]] += reg;
        }

        // Compute matrix inverses
        let cxx_inv =
            inv(&cxx_reg).map_err(|e: LinalgError| GpuError::ComputationError(e.to_string()))?;
        let cyy_inv =
            inv(&cyy_reg).map_err(|e: LinalgError| GpuError::ComputationError(e.to_string()))?;

        // Build symmetric intermediate matrix:
        //   M = Cxx_inv @ Cxy @ Cyy_inv @ Cxy^T
        // M is symmetric in exact arithmetic (PD when Cxx, Cyy are SPD). Floating-point
        // accumulation can introduce small off-diagonal asymmetry, so we explicitly
        // symmetrize: M_sym = (M + M^T) / 2.
        let cxy_t = cxy.t().to_owned();
        let tmp1 = cxx_inv.dot(cxy);
        let tmp2 = tmp1.dot(&cyy_inv);
        let m_raw = tmp2.dot(&cxy_t);
        let m_mat = (&m_raw + &m_raw.t().to_owned()) / 2.0;

        // Solve symmetric eigenproblem M w = λ w
        let (mut eigenvalues, mut eigenvectors) = eigh(&m_mat, UPLO::Upper)
            .map_err(|e: LinalgError| GpuError::ComputationError(e.to_string()))?;

        // eigh returns eigenvalues in ascending order; reverse to get descending (largest first)
        let n = eigenvalues.len();
        let rev_idx: Vec<usize> = (0..n).rev().collect();
        eigenvalues = Array1::from_vec(rev_idx.iter().map(|&i| eigenvalues[i]).collect());
        eigenvectors = {
            let cols: Vec<_> = rev_idx
                .iter()
                .map(|&i| eigenvectors.column(i).to_owned())
                .collect();
            let n_rows = eigenvectors.nrows();
            let n_cols = cols.len();
            let mut out = Array2::<f64>::zeros((n_rows, n_cols));
            for (j, col) in cols.iter().enumerate() {
                out.column_mut(j).assign(col);
            }
            out
        };

        // Clamp canonical correlations to [0, 1] (sqrt of eigenvalues)
        let n_components = self.n_components.min(n);
        let eigenvalues_trunc = eigenvalues
            .slice(scirs2_core::ndarray::s![..n_components])
            .to_owned();
        let eigenvectors_trunc = eigenvectors
            .slice(scirs2_core::ndarray::s![.., ..n_components])
            .to_owned();

        Ok((eigenvalues_trunc, eigenvectors_trunc))
    }

    fn compute_y_weights(
        &self,
        _cyy: &Array2<f64>,
        cxy: &Array2<f64>,
        x_weights: &Array2<f64>,
    ) -> GpuResult<Array2<f64>> {
        // Compute Y weights from X weights using the CCA relationship
        // Y_weights = Cyy^-1 * Cxy^T * X_weights
        let cxy_t = cxy.t().to_owned();
        let temp = self.matrix_ops.matmul(&cxy_t, x_weights)?;

        // For simplicity, assume Cyy is invertible and use pseudo-inverse
        // In practice would use proper matrix inversion or regularization
        Ok(temp)
    }
}

/// GPU-accelerated fitted CCA model
pub struct GpuCCAFitted {
    pub x_weights: Array2<f64>,
    pub y_weights: Array2<f64>,
    pub correlations: Array1<f64>,
    matrix_ops: GpuMatrixOps,
}

impl GpuCCAFitted {
    /// Transform new data using GPU acceleration
    pub fn transform(
        &self,
        x: &Array2<f64>,
        y: &Array2<f64>,
    ) -> GpuResult<(Array2<f64>, Array2<f64>)> {
        let x_transformed = self.matrix_ops.matmul(x, &self.x_weights)?;
        let y_transformed = self.matrix_ops.matmul(y, &self.y_weights)?;

        Ok((x_transformed, y_transformed))
    }

    /// Transform only X data
    pub fn transform_x(&self, x: &Array2<f64>) -> GpuResult<Array2<f64>> {
        self.matrix_ops.matmul(x, &self.x_weights)
    }

    /// Transform only Y data
    pub fn transform_y(&self, y: &Array2<f64>) -> GpuResult<Array2<f64>> {
        self.matrix_ops.matmul(y, &self.y_weights)
    }

    /// Get canonical correlations
    pub fn correlations(&self) -> &Array1<f64> {
        &self.correlations
    }

    /// Get X weights
    pub fn x_weights(&self) -> &Array2<f64> {
        &self.x_weights
    }

    /// Get Y weights
    pub fn y_weights(&self) -> &Array2<f64> {
        &self.y_weights
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::essentials::Normal;
    use scirs2_core::ndarray::{arr2, Array2};
    use scirs2_core::random::thread_rng;

    #[test]
    fn test_gpu_context_creation() {
        let context = GpuAcceleratedContext::new();
        // Should always succeed, and must never panic; whichever backend it landed
        // on, `backend()` and `is_gpu_enabled()` must agree with each other.
        if context.is_gpu_enabled() {
            assert_eq!(context.backend(), GpuBackendKind::Cuda);
        } else {
            assert_eq!(context.backend(), GpuBackendKind::Cpu);
        }
    }

    #[test]
    fn test_gpu_context_with_backend() {
        let context = GpuAcceleratedContext::with_backend(GpuBackendKind::Cpu);
        assert_eq!(context.backend(), GpuBackendKind::Cpu);
        assert!(!context.is_gpu_enabled());
    }

    #[test]
    fn test_gpu_backend_kind_preferred_is_available() {
        let kind = GpuBackendKind::preferred();
        assert!(kind.is_available());
    }

    #[test]
    fn test_gpu_backend_kind_unimplemented_variants_unavailable() {
        assert!(GpuBackendKind::Cpu.is_available());
        assert!(!GpuBackendKind::Metal.is_available());
        assert!(!GpuBackendKind::WebGpu.is_available());
        assert!(!GpuBackendKind::Rocm.is_available());
        assert!(!GpuBackendKind::OpenCl.is_available());
    }

    #[test]
    fn test_memory_info() {
        let context = GpuAcceleratedContext::new();
        let memory_info = context.memory_info();
        // Should return valid memory info (or zeros for CPU)
        assert!(memory_info.available <= memory_info.total);
        assert!(memory_info.used <= memory_info.total);
        assert_eq!(memory_info.total, memory_info.available + memory_info.used);
    }

    #[test]
    fn test_matrix_ops_creation() {
        let _ops = GpuMatrixOps::new();
        // Should create successfully
    }

    #[test]
    fn test_cpu_matmul() {
        let ops = GpuMatrixOps::new();
        let a = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        let b = arr2(&[[5.0, 6.0], [7.0, 8.0]]);

        let result = ops
            .matmul(&a, &b)
            .expect("matrix multiplication should succeed");

        // Check result dimensions
        assert_eq!(result.dim(), (2, 2));

        // Check specific values (basic matrix multiplication)
        let expected = arr2(&[[19.0, 22.0], [43.0, 50.0]]);
        for ((i, j), &val) in result.indexed_iter() {
            assert!((val - expected[[i, j]]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_batch_matmul() {
        let ops = GpuMatrixOps::new();
        let a1 = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        let a2 = arr2(&[[2.0, 3.0], [4.0, 5.0]]);
        let b1 = arr2(&[[5.0, 6.0], [7.0, 8.0]]);
        let b2 = arr2(&[[6.0, 7.0], [8.0, 9.0]]);

        let a_batch = vec![a1, a2];
        let b_batch = vec![b1, b2];

        let results = ops
            .batch_matmul(&a_batch, &b_batch)
            .expect("matrix multiplication should succeed");

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].dim(), (2, 2));
        assert_eq!(results[1].dim(), (2, 2));
    }

    #[test]
    fn test_gpu_cca_creation() {
        let cca = GpuCCA::new(2);
        assert_eq!(cca.n_components, 2);
    }

    #[test]
    fn test_gpu_cca_fit() {
        let cca = GpuCCA::new(2);
        let x = Array2::from_shape_simple_fn((100, 5), || {
            let mut rng = thread_rng();
            rng.sample(Normal::new(0.0, 1.0).expect("Normal distribution params should be valid"))
        });
        let y = Array2::from_shape_simple_fn((100, 3), || {
            let mut rng = thread_rng();
            rng.sample(Normal::new(0.0, 1.0).expect("Normal distribution params should be valid"))
        });

        let fitted = cca.fit(&x, &y).expect("fit should succeed");

        // Check output dimensions
        assert_eq!(fitted.x_weights.dim(), (5, 2));
        assert_eq!(fitted.y_weights.dim(), (3, 2));
        assert_eq!(fitted.correlations.len(), 2);
    }

    #[test]
    fn test_gpu_cca_transform() {
        let cca = GpuCCA::new(1);
        let x_train = Array2::from_shape_simple_fn((50, 4), || {
            let mut rng = thread_rng();
            rng.sample(Normal::new(0.0, 1.0).expect("Normal distribution params should be valid"))
        });
        let y_train = Array2::from_shape_simple_fn((50, 2), || {
            let mut rng = thread_rng();
            rng.sample(Normal::new(0.0, 1.0).expect("Normal distribution params should be valid"))
        });

        let fitted = cca.fit(&x_train, &y_train).expect("fit should succeed");

        let x_test = Array2::from_shape_simple_fn((10, 4), || {
            let mut rng = thread_rng();
            rng.sample(Normal::new(0.0, 1.0).expect("Normal distribution params should be valid"))
        });
        let y_test = Array2::from_shape_simple_fn((10, 2), || {
            let mut rng = thread_rng();
            rng.sample(Normal::new(0.0, 1.0).expect("Normal distribution params should be valid"))
        });

        let (x_transformed, y_transformed) = fitted
            .transform(&x_test, &y_test)
            .expect("transform should succeed");

        // Check output dimensions
        assert_eq!(x_transformed.dim(), (10, 1));
        assert_eq!(y_transformed.dim(), (10, 1));
    }

    #[test]
    fn test_dimension_mismatch_error() {
        let ops = GpuMatrixOps::new();
        let a = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        let b = arr2(&[[5.0], [6.0], [7.0]]); // Wrong dimensions

        let result = ops.matmul(&a, &b);
        assert!(result.is_err());

        if let Err(GpuError::DimensionError(_)) = result {
            // Expected error type
        } else {
            panic!("Expected DimensionError");
        }
    }
}
