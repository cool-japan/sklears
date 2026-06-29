//! GPU acceleration for linear models using oxicuda-backend.
//!
//! Provides GPU-accelerated linear algebra for linear models. GEMM-based ops
//! (`matrix_multiply`, `matrix_vector_multiply`, `vector_dot`) dispatch through
//! `oxicuda-backend`'s `ComputeBackend::gemm()` when the `gpu` feature is enabled;
//! vector-only ops (axpy, scale) run CPU-side because the backend's element-wise
//! primitives are f32-only.

#![allow(clippy::uninlined_format_args)]

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_linalg::compat::{compat_solve, qr};
use sklears_core::{
    error::{Result, SklearsError},
    gpu::GpuContext,
    types::Float,
};
use std::sync::Arc;

#[cfg(feature = "gpu")]
use sklears_core::gpu::{GpuArray, GpuMatrixOps};

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
    context: Arc<GpuContext>,
}

impl GpuLinearOps {
    /// Create a new instance.
    pub fn new(config: GpuConfig) -> Result<Self> {
        let context = Arc::new(GpuContext::new()?);
        Ok(Self { config, context })
    }

    /// Create with default configuration.
    #[allow(clippy::should_implement_trait)]
    pub fn default() -> Result<Self> {
        Self::new(GpuConfig::default())
    }

    /// Returns `true` when a non-CPU backend (e.g. CUDA) is active.
    pub fn is_gpu_available(&self) -> bool {
        self.context.is_gpu()
    }

    /// Returns `(used_bytes, total_bytes)` for the active device.
    pub fn get_memory_info(&self) -> Result<(usize, usize)> {
        #[cfg(feature = "gpu")]
        {
            if let Ok(info) = self.context.memory_info() {
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
            let a_gpu = GpuArray::<Float>::from_array2(&self.context, a)?;
            let b_gpu = GpuArray::<Float>::from_array2(&self.context, b)?;
            let c_gpu = a_gpu.matmul(&b_gpu)?;
            return c_gpu.to_array2();
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
            let x_2d = Array2::from_shape_vec((k, 1), x.to_vec())
                .map_err(|e| SklearsError::InvalidInput(format!("reshape x failed: {e}")))?;
            let a_gpu = GpuArray::<Float>::from_array2(&self.context, a)?;
            let x_gpu = GpuArray::<Float>::from_array2(&self.context, &x_2d)?;
            let y_gpu = a_gpu.matmul(&x_gpu)?;
            let y_2d = y_gpu.to_array2()?;
            return Ok(y_2d.column(0).to_owned());
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
            let x_2d = Array2::from_shape_vec((1, n), x.to_vec())
                .map_err(|e| SklearsError::InvalidInput(format!("reshape x failed: {e}")))?;
            let y_2d = Array2::from_shape_vec((n, 1), y.to_vec())
                .map_err(|e| SklearsError::InvalidInput(format!("reshape y failed: {e}")))?;
            let x_gpu = GpuArray::<Float>::from_array2(&self.context, &x_2d)?;
            let y_gpu = GpuArray::<Float>::from_array2(&self.context, &y_2d)?;
            let r_gpu = x_gpu.matmul(&y_gpu)?;
            let r_2d = r_gpu.to_array2()?;
            return Ok(r_2d[[0, 0]]);
        }

        Ok(x.dot(y))
    }

    /// Matrix transpose: `Aᵀ`.
    pub fn matrix_transpose(&self, a: &Array2<Float>) -> Result<Array2<Float>> {
        Ok(a.t().to_owned())
    }

    /// Solve the least-squares normal equations: `(AᵀA) x = Aᵀb`.
    ///
    /// Delegates to `scirs2_linalg::compat::compat_solve` (CPU LU).
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

        self.cpu_solve_linear_system(a, b)
    }

    fn cpu_solve_linear_system(
        &self,
        a: &Array2<Float>,
        b: &Array1<Float>,
    ) -> Result<Array1<Float>> {
        let at = a.t();
        let ata = at.dot(a);
        let atb = at.dot(b);
        compat_solve(&ata, &atb).map_err(|e| {
            SklearsError::NumericalError(format!(
                "Linear system solve failed (matrix may be singular): {e}"
            ))
        })
    }

    /// QR decomposition: `A = Q × R`.
    pub fn qr_decomposition(&self, a: &Array2<Float>) -> Result<(Array2<Float>, Array2<Float>)> {
        self.cpu_qr_decomposition(a)
    }

    fn cpu_qr_decomposition(&self, a: &Array2<Float>) -> Result<(Array2<Float>, Array2<Float>)> {
        qr(a).map_err(|e| SklearsError::NumericalError(format!("QR decomposition failed: {e}")))
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

    /// Synchronize all pending GPU operations (no-op for CPU backend).
    pub fn synchronize(&self) -> Result<()> {
        #[cfg(feature = "gpu")]
        self.context.synchronize()?;
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
