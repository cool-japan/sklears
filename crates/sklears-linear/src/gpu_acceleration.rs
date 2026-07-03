//! GPU acceleration for linear models, backed by `sklears_core::gpu`
//! (`GpuBackend` / `GpuArray` / `GpuMatrixOps`, wired directly to
//! `oxicuda-blas`).
//!
//! GEMM-based ops (`matrix_multiply`, `matrix_vector_multiply`, `vector_dot`)
//! dispatch through `GpuMatrixOps::matmul` when the `gpu` feature is enabled
//! and a GPU backend was detected. `solve_linear_system` and
//! `qr_decomposition` dispatch through `oxicuda_solver::dense::{lu_factorize,
//! lu_solve, qr_factorize, qr_generate_q}`. `vector_axpy`/`vector_scale`
//! remain CPU-side (out of scope for this wave).
//!
//! # Wave B1
//!
//! `sklears_core::gpu::GpuBackend::detect()` is fallible/optional
//! (`Result<Option<GpuBackend>>`) rather than the old infallible
//! `GpuContext::new()`, so [`GpuLinearOps`] now stores `Option<GpuBackend>`.
//! Every GPU-dispatch site below extends its existing
//! problem-size-threshold CPU/GPU branch to also require a detected
//! backend, falling back to the CPU path both when the problem is too small
//! and when no GPU was detected.

#![allow(clippy::uninlined_format_args)]

use scirs2_core::ndarray::{Array1, Array2, ShapeBuilder};
use scirs2_linalg::compat::{compat_solve, qr};
use sklears_core::{
    error::{Result, SklearsError},
    gpu::GpuBackend,
    types::Float,
};

#[cfg(feature = "gpu")]
use sklears_core::gpu::{GpuArray, GpuMatrixOps};
#[cfg(feature = "gpu")]
use oxicuda_memory::DeviceBuffer;
#[cfg(feature = "gpu")]
use oxicuda_solver::{
    dense::{lu_factorize, lu_solve, qr_factorize, qr_generate_q},
    SolverHandle,
};

/// Maps any GPU-stack error (`oxicuda-driver`/`oxicuda-blas`/`oxicuda-solver`)
/// to a `SklearsError`, without needing those crates as direct dependencies
/// just to name their error types.
#[cfg(feature = "gpu")]
fn gpu_err<E: std::fmt::Display>(e: E) -> SklearsError {
    SklearsError::NumericalError(format!("GPU error: {e}"))
}

// ─── GpuConfig ─────────────────────────────────────────────────────────────────

/// Configuration for GPU acceleration
#[derive(Debug, Clone)]
pub struct GpuConfig {
    /// Device ID to use (default: 0)
    pub device_id: usize,
    /// Memory pool size in bytes (default: 1GB)
    pub memory_pool_size: usize,
    /// Use pinned memory for faster transfers (default: true)
    pub use_pinned_memory: bool,
    /// Minimum number of floating-point elements before GPU is preferred
    pub min_problem_size: usize,
    /// Number of compute streams (default: 4)
    pub num_streams: usize,
    /// Enable automatic mixed precision (default: false)
    pub use_mixed_precision: bool,
}

impl Default for GpuConfig {
    fn default() -> Self {
        Self {
            device_id: 0,
            memory_pool_size: 1024 * 1024 * 1024,
            use_pinned_memory: true,
            min_problem_size: 1000,
            num_streams: 4,
            use_mixed_precision: false,
        }
    }
}

// ─── GpuLinearOps ──────────────────────────────────────────────────────────────

/// GPU-accelerated linear operations backed by oxicuda-backend.
///
/// When the `gpu` feature is enabled, GEMM-based operations use a real
/// `ComputeBackend` (CUDA or CPU reference). All operations provide a CPU
/// fallback for small problem sizes or when the feature is absent.
pub struct GpuLinearOps {
    config: GpuConfig,
    /// `None` when no GPU was detected (or the `gpu` feature is disabled);
    /// every GPU-accelerated method below falls back to its CPU
    /// counterpart in that case. See [`GpuBackend::detect`].
    context: Option<GpuBackend>,
}

impl GpuLinearOps {
    /// Create a new instance.
    ///
    /// GPU detection is best-effort: on a machine with no GPU/driver
    /// present, [`GpuBackend::detect`] returns `Ok(None)` rather than an
    /// error, and every operation below transparently uses its CPU
    /// fallback.
    pub fn new(config: GpuConfig) -> Result<Self> {
        let context = GpuBackend::detect()?;
        Ok(Self { config, context })
    }

    /// Create with default configuration.
    #[allow(clippy::should_implement_trait)]
    pub fn default() -> Result<Self> {
        Self::new(GpuConfig::default())
    }

    /// Returns `true` when a real GPU backend was detected.
    pub fn is_gpu_available(&self) -> bool {
        self.context.is_some()
    }

    /// Returns `(used_bytes, total_bytes)` for the active device.
    pub fn get_memory_info(&self) -> Result<(usize, usize)> {
        #[cfg(feature = "gpu")]
        if let Some(backend) = self.context.as_ref() {
            if let Ok(info) = backend.memory_info() {
                return Ok((info.used, info.total));
            }
        }
        Ok((0, 0))
    }

    // ── Matrix / vector ops ──────────────────────────────────────────────────

    /// GPU-accelerated matrix multiplication: `C = A × B`.
    ///
    /// Uses `ComputeBackend::gemm()` via the row-major ↔ column-major identity
    /// when `gpu` is enabled and problem is large enough; falls back to
    /// `a.dot(b)` otherwise.
    pub fn matrix_multiply(&self, a: &Array2<Float>, b: &Array2<Float>) -> Result<Array2<Float>> {
        let (m, k) = a.dim();
        let (k2, n) = b.dim();

        if k != k2 {
            return Err(SklearsError::InvalidInput(format!(
                "Matrix dimensions incompatible: ({}, {}) * ({}, {})",
                m, k, k2, n
            )));
        }

        #[cfg(feature = "gpu")]
        if m * k + k * n >= self.config.min_problem_size {
            if let Some(backend) = self.context.as_ref() {
                let a_gpu = GpuArray::<Float>::from_array2(backend, a)?;
                let b_gpu = GpuArray::<Float>::from_array2(backend, b)?;
                let c_gpu = a_gpu.matmul(&b_gpu)?;
                return c_gpu.to_array2();
            }
        }

        self.cpu_matrix_multiply(a, b)
    }

    fn cpu_matrix_multiply(&self, a: &Array2<Float>, b: &Array2<Float>) -> Result<Array2<Float>> {
        Ok(a.dot(b))
    }

    /// GPU-accelerated matrix–vector product: `y = A × x`.
    ///
    /// Implemented via GEMM with n = 1 on the GPU path.
    pub fn matrix_vector_multiply(
        &self,
        a: &Array2<Float>,
        x: &Array1<Float>,
    ) -> Result<Array1<Float>> {
        let (m, k) = a.dim();

        if k != x.len() {
            return Err(SklearsError::InvalidInput(format!(
                "matrix_vector_multiply: A is ({}, {}) but x has len {}",
                m,
                k,
                x.len()
            )));
        }

        #[cfg(feature = "gpu")]
        if m * k >= self.config.min_problem_size {
            if let Some(backend) = self.context.as_ref() {
                let x_2d = Array2::from_shape_vec((k, 1), x.to_vec())
                    .map_err(|e| SklearsError::InvalidInput(format!("reshape x failed: {e}")))?;
                let a_gpu = GpuArray::<Float>::from_array2(backend, a)?;
                let x_gpu = GpuArray::<Float>::from_array2(backend, &x_2d)?;
                let y_gpu = a_gpu.matmul(&x_gpu)?;
                let y_2d = y_gpu.to_array2()?;
                return Ok(y_2d.column(0).to_owned());
            }
        }

        Ok(a.dot(x))
    }

    /// AXPY: `y += alpha × x` (in-place, CPU-side for f64 correctness).
    pub fn vector_axpy(
        &self,
        alpha: Float,
        x: &Array1<Float>,
        y: &mut Array1<Float>,
    ) -> Result<()> {
        if x.len() != y.len() {
            return Err(SklearsError::InvalidInput(format!(
                "vector_axpy: x.len()={} != y.len()={}",
                x.len(),
                y.len()
            )));
        }
        y.zip_mut_with(x, |yi, xi| *yi += alpha * xi);
        Ok(())
    }

    /// Scale: `x *= alpha` (in-place, CPU-side).
    pub fn vector_scale(&self, alpha: Float, x: &mut Array1<Float>) -> Result<()> {
        x.mapv_inplace(|v| v * alpha);
        Ok(())
    }

    /// Dot product: `x · y`.
    ///
    /// Uses a 1×n by n×1 GEMM on the GPU path; falls back to `x.dot(y)`.
    pub fn vector_dot(&self, x: &Array1<Float>, y: &Array1<Float>) -> Result<Float> {
        let n = x.len();
        if n != y.len() {
            return Err(SklearsError::InvalidInput(format!(
                "vector_dot: x.len()={} != y.len()={}",
                n,
                y.len()
            )));
        }

        #[cfg(feature = "gpu")]
        if n >= self.config.min_problem_size {
            if let Some(backend) = self.context.as_ref() {
                let x_2d = Array2::from_shape_vec((1, n), x.to_vec())
                    .map_err(|e| SklearsError::InvalidInput(format!("reshape x failed: {e}")))?;
                let y_2d = Array2::from_shape_vec((n, 1), y.to_vec())
                    .map_err(|e| SklearsError::InvalidInput(format!("reshape y failed: {e}")))?;
                let x_gpu = GpuArray::<Float>::from_array2(backend, &x_2d)?;
                let y_gpu = GpuArray::<Float>::from_array2(backend, &y_2d)?;
                let r_gpu = x_gpu.matmul(&y_gpu)?;
                let r_2d = r_gpu.to_array2()?;
                return Ok(r_2d[[0, 0]]);
            }
        }

        Ok(x.dot(y))
    }

    /// Matrix transpose: `Aᵀ`.
    pub fn matrix_transpose(&self, a: &Array2<Float>) -> Result<Array2<Float>> {
        Ok(a.t().to_owned())
    }

    /// Solve the least-squares normal equations: `(AᵀA) x = Aᵀb`.
    ///
    /// Tries a GPU LU solve (`oxicuda_solver::dense::{lu_factorize,
    /// lu_solve}`) first when a GPU backend is available, falling back to
    /// `scirs2_linalg::compat::compat_solve` (CPU LU) when no GPU is
    /// present or the GPU path itself errors.
    pub fn solve_linear_system(
        &self,
        a: &Array2<Float>,
        b: &Array1<Float>,
    ) -> Result<Array1<Float>> {
        let (m, n) = a.dim();
        if m == 0 || n == 0 {
            return Err(SklearsError::InvalidInput(
                "solve_linear_system: empty matrix".to_string(),
            ));
        }
        if m != b.len() {
            return Err(SklearsError::InvalidInput(format!(
                "solve_linear_system: A is ({}, {}) but b has len {}",
                m,
                n,
                b.len()
            )));
        }

        let at = a.t();
        let ata = at.dot(a);
        let atb = at.dot(b);

        #[cfg(feature = "gpu")]
        if let Some(backend) = self.context.as_ref() {
            match Self::gpu_solve_normal_equations(backend, &ata, &atb) {
                Ok(x) => return Ok(x),
                Err(e) => {
                    log::warn!("GPU LU solve failed ({e}), falling back to CPU solve");
                }
            }
        }

        Self::cpu_solve_normal_equations(&ata, &atb)
    }

    fn cpu_solve_normal_equations(
        ata: &Array2<Float>,
        atb: &Array1<Float>,
    ) -> Result<Array1<Float>> {
        compat_solve(ata, atb).map_err(|e| {
            SklearsError::NumericalError(format!(
                "Linear system solve failed (matrix may be singular): {e}"
            ))
        })
    }

    /// GPU LU solve of `ata * x = atb` via
    /// `oxicuda_solver::dense::{lu_factorize, lu_solve}`.
    #[cfg(feature = "gpu")]
    fn gpu_solve_normal_equations(
        backend: &GpuBackend,
        ata: &Array2<Float>,
        atb: &Array1<Float>,
    ) -> Result<Array1<Float>> {
        let n = ata.nrows();

        backend.context().set_current().map_err(gpu_err)?;

        // oxicuda-solver expects column-major storage; `ata.t().iter()`
        // walks `ata` in column-major order without a manual transpose
        // loop (transposing the view swaps which axis is "outer").
        let ata_col_major: Vec<Float> = ata.t().iter().copied().collect();
        let atb_host: Vec<Float> = atb.iter().copied().collect();

        let mut solver = SolverHandle::new(backend.context()).map_err(gpu_err)?;
        let mut d_a = DeviceBuffer::from_host(&ata_col_major).map_err(gpu_err)?;
        let mut d_pivots = DeviceBuffer::<i32>::zeroed(n).map_err(gpu_err)?;

        let lu_result =
            lu_factorize::<Float>(&mut solver, &mut d_a, n as u32, n as u32, &mut d_pivots)
                .map_err(gpu_err)?;
        if lu_result.info != 0 {
            return Err(SklearsError::NumericalError(format!(
                "GPU LU factorization detected a singular matrix at column {}",
                lu_result.info
            )));
        }

        let mut d_b = DeviceBuffer::from_host(&atb_host).map_err(gpu_err)?;
        lu_solve::<Float>(&solver, &d_a, &d_pivots, &mut d_b, n as u32, 1).map_err(gpu_err)?;

        let mut x_host = vec![0.0; n];
        d_b.copy_to_host(&mut x_host).map_err(gpu_err)?;
        Ok(Array1::from_vec(x_host))
    }

    /// QR decomposition: `A = Q × R`.
    ///
    /// Tries a GPU Householder QR (`oxicuda_solver::dense::qr_factorize` +
    /// `qr_generate_q`) first when a GPU backend is available, falling back
    /// to `scirs2_linalg::compat::qr` (CPU) when no GPU is present or the
    /// GPU path itself errors.
    pub fn qr_decomposition(&self, a: &Array2<Float>) -> Result<(Array2<Float>, Array2<Float>)> {
        #[cfg(feature = "gpu")]
        if let Some(backend) = self.context.as_ref() {
            match Self::gpu_qr_decomposition(backend, a) {
                Ok(result) => return Ok(result),
                Err(e) => {
                    log::warn!("GPU QR decomposition failed ({e}), falling back to CPU");
                }
            }
        }

        Self::cpu_qr_decomposition(a)
    }

    fn cpu_qr_decomposition(a: &Array2<Float>) -> Result<(Array2<Float>, Array2<Float>)> {
        qr(a).map_err(|e| SklearsError::NumericalError(format!("QR decomposition failed: {e}")))
    }

    /// GPU QR factorization via `oxicuda_solver::dense::{qr_factorize,
    /// qr_generate_q}`. Matches the CPU path's shape contract: `Q` is
    /// `m x m`, `R` is `m x n`.
    #[cfg(feature = "gpu")]
    fn gpu_qr_decomposition(
        backend: &GpuBackend,
        a: &Array2<Float>,
    ) -> Result<(Array2<Float>, Array2<Float>)> {
        let (m, n) = a.dim();
        if m == 0 || n == 0 {
            return Err(SklearsError::InvalidInput(
                "qr_decomposition: empty matrix".to_string(),
            ));
        }

        backend.context().set_current().map_err(gpu_err)?;

        // oxicuda-solver expects column-major storage with lda = m.
        let a_col_major: Vec<Float> = a.t().iter().copied().collect();
        let k = m.min(n);

        let mut solver = SolverHandle::new(backend.context()).map_err(gpu_err)?;
        let mut d_a = DeviceBuffer::from_host(&a_col_major).map_err(gpu_err)?;
        let mut d_tau = DeviceBuffer::<Float>::zeroed(k).map_err(gpu_err)?;

        qr_factorize::<Float>(&mut solver, &mut d_a, m as u32, n as u32, m as u32, &mut d_tau)
            .map_err(gpu_err)?;

        let mut d_q = DeviceBuffer::<Float>::zeroed(m * m).map_err(gpu_err)?;
        qr_generate_q::<Float>(&solver, &d_a, &d_tau, &mut d_q, m as u32, n as u32)
            .map_err(gpu_err)?;

        let mut qr_packed = vec![0.0; m * n];
        d_a.copy_to_host(&mut qr_packed).map_err(gpu_err)?;
        let mut q_flat = vec![0.0; m * m];
        d_q.copy_to_host(&mut q_flat).map_err(gpu_err)?;

        // `qr_packed` holds R in its upper triangle (column-major, lda = m)
        // and Householder vectors below the diagonal; only copy the upper
        // triangle so the Householder vector data doesn't leak into R.
        let mut r = Array2::<Float>::zeros((m, n));
        for col in 0..n {
            let last_row = col.min(m - 1);
            for row in 0..=last_row {
                r[[row, col]] = qr_packed[col * m + row];
            }
        }

        let q = Array2::from_shape_vec((m, m).f(), q_flat)?;

        Ok((q, r))
    }

    /// Batch matrix multiplication (sequential dispatch per pair).
    pub fn batch_matrix_multiply(
        &self,
        matrices: &[(Array2<Float>, Array2<Float>)],
    ) -> Result<Vec<Array2<Float>>> {
        matrices
            .iter()
            .map(|(a, b)| self.matrix_multiply(a, b))
            .collect()
    }

    /// Synchronize all pending GPU operations (no-op when no GPU is active).
    pub fn synchronize(&self) -> Result<()> {
        #[cfg(feature = "gpu")]
        if let Some(backend) = self.context.as_ref() {
            backend.synchronize()?;
        }
        Ok(())
    }

    /// Snapshot of performance statistics.
    pub fn get_performance_stats(&self) -> GpuPerformanceStats {
        GpuPerformanceStats {
            total_operations: 0,
            gpu_operations: 0,
            cpu_fallback_operations: 0,
            average_gpu_time_ms: 0.0,
            average_cpu_time_ms: 0.0,
            memory_transfers_mb: 0.0,
        }
    }
}

// ─── GpuPerformanceStats ───────────────────────────────────────────────────────

/// Performance statistics snapshot.
#[derive(Debug, Clone)]
pub struct GpuPerformanceStats {
    pub total_operations: usize,
    pub gpu_operations: usize,
    pub cpu_fallback_operations: usize,
    pub average_gpu_time_ms: f64,
    pub average_cpu_time_ms: f64,
    pub memory_transfers_mb: f64,
}

// ─── GpuMemoryPool ─────────────────────────────────────────────────────────────

/// Simple CPU-side memory-pool simulator for capacity accounting.
pub struct GpuMemoryPool {
    pool_size: usize,
    allocated_bytes: usize,
}

impl GpuMemoryPool {
    pub fn new(pool_size: usize) -> Self {
        Self {
            pool_size,
            allocated_bytes: 0,
        }
    }

    pub fn allocate(&mut self, size: usize) -> Result<()> {
        if self.allocated_bytes + size > self.pool_size {
            return Err(SklearsError::InvalidInput(format!(
                "Out of GPU memory: requested {}, available {}",
                size,
                self.pool_size - self.allocated_bytes
            )));
        }
        self.allocated_bytes += size;
        Ok(())
    }

    pub fn deallocate(&mut self, size: usize) {
        self.allocated_bytes = self.allocated_bytes.saturating_sub(size);
    }

    pub fn available_memory(&self) -> usize {
        self.pool_size - self.allocated_bytes
    }
}

// ─── Tests ─────────────────────────────────────────────────────────────────────

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_gpu_ops_creation() {
        let config = GpuConfig::default();
        let gpu_ops = GpuLinearOps::new(config);
        assert!(gpu_ops.is_ok());
    }

    #[test]
    fn test_matrix_multiplication() {
        let gpu_ops = GpuLinearOps::default().expect("operation should succeed");

        let a = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .expect("valid array shape");
        let b = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .expect("valid array shape");

        let result = gpu_ops
            .matrix_multiply(&a, &b)
            .expect("operation should succeed");

        assert_eq!(result.shape(), &[2, 2]);
        assert_abs_diff_eq!(result[[0, 0]], 22.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[[0, 1]], 28.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[[1, 0]], 49.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[[1, 1]], 64.0, epsilon = 1e-10);
    }

    #[test]
    fn test_matrix_vector_multiplication() {
        let gpu_ops = GpuLinearOps::default().expect("operation should succeed");

        let a = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .expect("valid array shape");
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0]);

        let result = gpu_ops
            .matrix_vector_multiply(&a, &x)
            .expect("operation should succeed");

        assert_eq!(result.len(), 2);
        assert_abs_diff_eq!(result[0], 14.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[1], 32.0, epsilon = 1e-10);
    }

    #[test]
    fn test_vector_operations() {
        let gpu_ops = GpuLinearOps::default().expect("operation should succeed");

        let x = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let mut y = Array1::from_vec(vec![4.0, 5.0, 6.0]);

        gpu_ops
            .vector_axpy(2.0, &x, &mut y)
            .expect("operation should succeed");
        assert_abs_diff_eq!(y[0], 6.0, epsilon = 1e-10);
        assert_abs_diff_eq!(y[1], 9.0, epsilon = 1e-10);
        assert_abs_diff_eq!(y[2], 12.0, epsilon = 1e-10);

        let dot = gpu_ops
            .vector_dot(&x, &Array1::from_vec(vec![1.0, 1.0, 1.0]))
            .expect("operation should succeed");
        assert_abs_diff_eq!(dot, 6.0, epsilon = 1e-10);
    }

    #[test]
    fn test_memory_pool() {
        let mut pool = GpuMemoryPool::new(1024);

        assert_eq!(pool.available_memory(), 1024);

        pool.allocate(512).expect("operation should succeed");
        assert_eq!(pool.available_memory(), 512);

        pool.deallocate(256);
        assert_eq!(pool.available_memory(), 768);

        let result = pool.allocate(1024);
        assert!(result.is_err());
    }

    #[test]
    fn test_gpu_availability() {
        let gpu_ops = GpuLinearOps::default().expect("operation should succeed");
        assert!(!gpu_ops.is_gpu_available());
    }

    #[test]
    fn test_performance_stats() {
        let gpu_ops = GpuLinearOps::default().expect("operation should succeed");
        let stats = gpu_ops.get_performance_stats();
        assert_eq!(stats.total_operations, 0);
        assert_eq!(stats.gpu_operations, 0);
        assert_eq!(stats.cpu_fallback_operations, 0);
    }
}
