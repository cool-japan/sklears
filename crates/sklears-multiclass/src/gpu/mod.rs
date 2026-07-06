//! GPU Acceleration Framework for Multiclass Classification
//!
//! This module provides GPU-accelerated operations for multiclass
//! classification (parallel voting aggregation, matrix operations for ECOC,
//! and batch prediction), wired directly onto `sklears_core::gpu` -- which
//! in turn is backed by the real `oxicuda-driver` / `oxicuda-blas` /
//! `oxicuda-memory` crates behind the `gpu_support` feature.
//!
//! There is no bespoke CUDA/OpenCL binding in this crate. [`GpuBackend`]
//! (re-exported here as [`GpuContext`], matching the pre-migration name) can
//! only be constructed once [`GpuBackend::detect`] has found a real device;
//! it honestly returns `Ok(None)` when none is present -- including on this
//! crate's own macOS dev/CI environment, which has no NVIDIA GPU. Whenever
//! `detect()` returns `None` (or the `gpu` feature is not compiled in at
//! all), callers should use the always-available, Pure-Rust CPU
//! implementations in [`fallback`] instead of [`OxiCudaMatrixOps`].

pub mod fallback;

use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::error::{Result as SklResult, SklearsError};

#[cfg(feature = "gpu")]
use scirs2_core::ndarray::Axis;

// `sklears_core::gpu` is itself gated behind sklears-core's `gpu_support`
// feature (which this crate's own `gpu` feature turns on -- see
// `Cargo.toml`), so the whole module -- not just individual items in it --
// is simply absent from the dependency graph when `gpu` is off. Re-export
// the real types when available; otherwise stand in with local honest
// equivalents (below) so callers written against `GpuBackend::detect()` /
// `GpuContext` / `GpuArray` compile the same regardless of whether the
// `gpu` feature is enabled.
#[cfg(feature = "gpu")]
pub use sklears_core::gpu::{GpuArray, GpuBackend, GpuContext};

#[cfg(feature = "gpu")]
use sklears_core::gpu::GpuMatrixOps as CoreGpuMatrixOps;

/// Non-`gpu`-feature stand-in for `sklears_core::gpu::GpuBackend`. This
/// build has no GPU code compiled into it at all (`sklears_core::gpu` is
/// itself absent), so [`detect`](Self::detect) always, honestly, reports no
/// device -- there is no CPU-backed fake substituted in its place.
#[cfg(not(feature = "gpu"))]
#[derive(Debug, Clone)]
pub struct GpuBackend;

#[cfg(not(feature = "gpu"))]
impl GpuBackend {
    /// Always `Ok(None)`: this build has no `gpu_support`-backed detection
    /// path compiled in.
    pub fn detect() -> SklResult<Option<Self>> {
        Ok(None)
    }

    /// Always `false`, matching [`detect`](Self::detect).
    pub fn is_available() -> bool {
        false
    }
}

/// Compatibility alias matching the pre-migration name; see [`GpuBackend`].
#[cfg(not(feature = "gpu"))]
pub type GpuContext = GpuBackend;

/// Non-`gpu`-feature stand-in for `sklears_core::gpu::GpuArray`. Since this
/// build has no `GpuBackend` that ever returns `Some`, no value of this type
/// can ever actually be constructed -- it exists purely so type signatures
/// (e.g. [`memory::to_device`]) compile identically with and without the
/// `gpu` feature.
#[cfg(not(feature = "gpu"))]
#[derive(Debug)]
pub struct GpuArray<T> {
    _phantom: std::marker::PhantomData<T>,
}

/// Detects a real GPU backend, following the same honest-detection contract
/// as every other oxicuda-backed crate in this workspace: `Ok(None)`
/// whenever no device is present, whether that's because the `gpu` feature
/// is compiled out entirely or because [`GpuBackend::detect`] legitimately
/// found no hardware.
pub fn detect_context() -> SklResult<Option<GpuContext>> {
    GpuBackend::detect()
}

/// GPU-accelerated voting aggregation
pub trait GpuVotingOps {
    /// Aggregate votes using GPU acceleration
    fn aggregate_votes_gpu(
        &self,
        votes: &Array2<f64>,
        weights: Option<&Array1<f64>>,
        ctx: &GpuContext,
    ) -> SklResult<Array1<i32>>;

    /// Compute weighted probability aggregation on GPU
    fn aggregate_probabilities_gpu(
        &self,
        probabilities: &[Array2<f64>],
        weights: Option<&Array1<f64>>,
        ctx: &GpuContext,
    ) -> SklResult<Array2<f64>>;
}

/// GPU-accelerated matrix operations for ECOC
pub trait GpuMatrixOps {
    /// Matrix-vector multiplication on GPU
    fn matmul_gpu(
        &self,
        matrix: &Array2<f64>,
        vector: &Array1<f64>,
        ctx: &GpuContext,
    ) -> SklResult<Array1<f64>>;

    /// Batch matrix multiplication on GPU
    fn batch_matmul_gpu(
        &self,
        matrices: &[Array2<f64>],
        vectors: &Array2<f64>,
        ctx: &GpuContext,
    ) -> SklResult<Array2<f64>>;

    /// Distance calculation for ECOC decoding on GPU
    fn compute_distances_gpu(
        &self,
        predictions: &Array2<f64>,
        code_matrix: &Array2<i8>,
        ctx: &GpuContext,
    ) -> SklResult<Array2<f64>>;
}

/// GPU-accelerated batch prediction
pub trait GpuBatchPredict {
    /// Predict on large batches using GPU
    fn predict_batch_gpu(&self, x: &Array2<f64>, ctx: &GpuContext) -> SklResult<Array1<i32>>;

    /// Predict probabilities on large batches using GPU
    fn predict_proba_batch_gpu(&self, x: &Array2<f64>, ctx: &GpuContext) -> SklResult<Array2<f64>>;
}

/// Real oxicuda-backed [`GpuMatrixOps`] implementation, used when the `gpu`
/// feature is enabled and a [`GpuContext`] obtained from a successful
/// [`GpuBackend::detect`] is available. This is the honest device path: when
/// no device is present (or the `gpu` feature is off), callers should use
/// [`fallback::CpuMatrixOps`] instead -- there is no CPU emulation baked
/// into this type.
#[cfg(feature = "gpu")]
#[derive(Debug, Default, Clone, Copy)]
pub struct OxiCudaMatrixOps;

#[cfg(feature = "gpu")]
impl GpuMatrixOps for OxiCudaMatrixOps {
    /// `matrix * vector` via a single on-device GEMM (the vector is uploaded
    /// as an `n x 1` matrix so it can go through the same GEMM path as
    /// [`batch_matmul_gpu`](Self::batch_matmul_gpu)).
    fn matmul_gpu(
        &self,
        matrix: &Array2<f64>,
        vector: &Array1<f64>,
        ctx: &GpuContext,
    ) -> SklResult<Array1<f64>> {
        if matrix.ncols() != vector.len() {
            return Err(SklearsError::InvalidInput(format!(
                "Matrix columns ({}) must match vector length ({})",
                matrix.ncols(),
                vector.len()
            )));
        }

        let vector_col: Array2<f64> = vector.clone().insert_axis(Axis(1));
        let gm = GpuArray::<f64>::from_array2(ctx, matrix)?;
        let gv = GpuArray::<f64>::from_array2(ctx, &vector_col)?;
        let gc = gm.matmul(&gv)?;
        let result = gc.to_array2()?;
        Ok(result.column(0).to_owned())
    }

    /// Batch of independent matrix-vector products. `oxicuda-blas` 0.4.0
    /// does expose batched/strided GEMM kernels, but only at the raw
    /// `DeviceBuffer` level; `sklears_core::gpu::GpuArray` deliberately does
    /// not expose its underlying buffer publicly (every `GpuArray` operation
    /// goes through the safe [`sklears_core::gpu::GpuMatrixOps`] surface), so
    /// this dispatches one real on-device GEMM per item via
    /// [`matmul_gpu`](Self::matmul_gpu) rather than reaching past that
    /// boundary. Each iteration is genuine device compute -- this is not a
    /// CPU loop pretending to be a GPU kernel, just a non-fused one.
    fn batch_matmul_gpu(
        &self,
        matrices: &[Array2<f64>],
        vectors: &Array2<f64>,
        ctx: &GpuContext,
    ) -> SklResult<Array2<f64>> {
        if matrices.is_empty() {
            return Err(SklearsError::InvalidInput("Empty matrices".to_string()));
        }

        let n_matrices = matrices.len();
        let n_vectors = vectors.nrows();

        if n_matrices != n_vectors {
            return Err(SklearsError::InvalidInput(
                "Number of matrices must match number of vectors".to_string(),
            ));
        }

        let result_dim = matrices[0].nrows();
        let mut results = Array2::zeros((n_vectors, result_dim));

        for i in 0..n_vectors {
            let vector = vectors.row(i).to_owned();
            let row_result = self.matmul_gpu(&matrices[i], &vector, ctx)?;
            for (j, &val) in row_result.iter().enumerate() {
                results[[i, j]] = val;
            }
        }

        Ok(results)
    }

    /// ECOC squared-Euclidean distance, computed as the GEMM expansion
    /// `||p||^2 + ||c||^2 - 2 * p . c^T`: the `O(n_samples * n_classes *
    /// n_estimators)` cross term runs as a single on-device GEMM
    /// (`predictions * code_matrix^T`), which is the only part of this
    /// computation expensive enough to be worth a device round trip. The
    /// per-row squared norms are `O(n)` host-side reductions and are left on
    /// the CPU.
    fn compute_distances_gpu(
        &self,
        predictions: &Array2<f64>,
        code_matrix: &Array2<i8>,
        ctx: &GpuContext,
    ) -> SklResult<Array2<f64>> {
        let (n_samples, n_estimators) = predictions.dim();
        let n_classes = code_matrix.nrows();

        if code_matrix.ncols() != n_estimators {
            return Err(SklearsError::InvalidInput(
                "Code matrix columns must match number of estimators".to_string(),
            ));
        }

        let p_norms: Vec<f64> = predictions
            .rows()
            .into_iter()
            .map(|row| row.iter().map(|v| v * v).sum())
            .collect();

        let code_f64: Array2<f64> = code_matrix.mapv(f64::from);
        let c_norms: Vec<f64> = code_f64
            .rows()
            .into_iter()
            .map(|row| row.iter().map(|v| v * v).sum())
            .collect();

        let gp = GpuArray::<f64>::from_array2(ctx, predictions)?;
        let gc = GpuArray::<f64>::from_array2(ctx, &code_f64)?;
        let gc_t = gc.transpose()?;
        let g_inner = gp.matmul(&gc_t)?;
        let inner = g_inner.to_array2()?;

        let mut distances = Array2::zeros((n_samples, n_classes));
        for i in 0..n_samples {
            for j in 0..n_classes {
                let squared = (p_norms[i] + c_norms[j] - 2.0 * inner[[i, j]]).max(0.0);
                distances[[i, j]] = squared.sqrt();
            }
        }

        Ok(distances)
    }
}

/// GPU memory management utilities
pub mod memory {
    use super::*;

    /// Transfer a 2-D array to the GPU device.
    #[cfg(feature = "gpu")]
    pub fn to_device<T: Copy + Clone>(data: &Array2<T>, ctx: &GpuContext) -> SklResult<GpuArray<T>> {
        GpuArray::from_array2(ctx, data)
    }

    /// Non-`gpu`-feature mirror of [`to_device`]: this build has no device
    /// path compiled in, so it always errors rather than pretending to
    /// upload anything.
    #[cfg(not(feature = "gpu"))]
    pub fn to_device<T: Clone>(_data: &Array2<T>, _ctx: &GpuContext) -> SklResult<GpuArray<T>> {
        Err(SklearsError::InvalidInput(
            "GPU feature not enabled. Rebuild with --features gpu".to_string(),
        ))
    }

    /// Transfer data from the GPU device back to a host 2-D array.
    #[cfg(feature = "gpu")]
    pub fn from_device<T: Copy + Clone + Default>(
        gpu_data: &GpuArray<T>,
        _ctx: &GpuContext,
    ) -> SklResult<Array2<T>> {
        gpu_data.to_array2()
    }

    /// Non-`gpu`-feature mirror of [`from_device`].
    #[cfg(not(feature = "gpu"))]
    pub fn from_device<T: Clone>(
        _gpu_data: &GpuArray<T>,
        _ctx: &GpuContext,
    ) -> SklResult<Array2<T>> {
        Err(SklearsError::InvalidInput(
            "GPU feature not enabled".to_string(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_context_is_ok() {
        // Must never panic and must never hard-`Err` just because there is
        // no GPU/driver present -- on this crate's dev/CI machine (macOS, no
        // NVIDIA GPU) `Ok(None)` is the expected, correct result.
        let result = detect_context();
        assert!(
            result.is_ok(),
            "detect_context() must not hard-error on missing GPU/driver: {result:?}"
        );
    }

    #[test]
    #[cfg(not(feature = "gpu"))]
    fn test_detect_context_none_without_feature() {
        assert!(detect_context().expect("should not hard-error").is_none());
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn test_oxicuda_matrix_ops_requires_real_device() {
        // No fabricated GPU availability: without real hardware, there is no
        // `GpuContext` to construct, so `OxiCudaMatrixOps` is simply unused.
        // This test only documents that `detect_context()` stays honest even
        // with the `gpu` feature compiled in.
        let ctx = detect_context().expect("detect_context should not hard-error");
        if ctx.is_none() {
            eprintln!(
                "skipping test_oxicuda_matrix_ops_requires_real_device: no GPU detected"
            );
        }
    }
}
