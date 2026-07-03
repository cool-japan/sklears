//! GPU-accelerated kernel computations, backed by `oxicuda-blas`.
//!
//! This module provides GPU-accelerated implementations of common SVM
//! kernels (Linear, RBF, Polynomial, Sigmoid). The inner-product term
//! (`X·Yᵀ`) is always computed on the GPU via `oxicuda-blas`'s BLAS Level-3
//! GEMM (`oxicuda_blas::level3::gemm`). For the RBF and Sigmoid kernels, the
//! post-GEMM non-linear transform (`exp` / `tanh`) also runs on-device via
//! `oxicuda-blas`'s elementwise primitives (`oxicuda_blas::level1::axpy`,
//! `oxicuda_blas::elementwise::{fill, exp, tanh_activation}`), so the
//! `O(n·m)` inner-product matrix never leaves the GPU between the matmul and
//! the activation -- only the final result is downloaded once. The
//! Polynomial kernel's `(γ·inner+c)^d` step still runs on the CPU; see
//! [`GpuKernelComputer`]'s doc comment for why.
//!
//! This has never used WGPU or compute shaders -- that was a stale doc claim
//! left over from an earlier draft of this file (the same issue previously
//! fixed in `sklears-clustering`'s `gpu_distances` module). GEMM here has
//! always gone through the `oxicuda-backend`/`oxicuda-blas` CUDA stack.

use crate::kernels::KernelType;
use scirs2_core::ndarray::Array2;
use thiserror::Error;

#[cfg(feature = "gpu")]
use oxicuda_blas::elementwise::{exp, fill, tanh_activation};
#[cfg(feature = "gpu")]
use oxicuda_blas::level1::axpy;
#[cfg(feature = "gpu")]
use oxicuda_blas::level3::gemm;
#[cfg(feature = "gpu")]
use oxicuda_blas::{Layout, MatrixDesc, MatrixDescMut, Transpose};
#[cfg(feature = "gpu")]
use oxicuda_memory::DeviceBuffer;
#[cfg(feature = "gpu")]
use sklears_core::gpu::GpuBackend;

/// Errors that can occur during GPU kernel computation
#[derive(Error, Debug)]
pub enum GpuKernelError {
    #[error("GPU device not available")]
    DeviceNotAvailable,
    #[error("Insufficient GPU memory")]
    InsufficientMemory,
    #[error("GPU computation failed: {0}")]
    ComputationFailed(String),
    #[error("Shader compilation failed: {0}")]
    ShaderCompilationFailed(String),
    #[error("Buffer creation failed")]
    BufferCreationFailed,
    #[error("GPU feature not supported: {0}")]
    FeatureNotSupported(String),
    #[error("Kernel matrix dimensions mismatch")]
    DimensionMismatch,
}

/// Result type for GPU kernel operations
pub type GpuKernelResult<T> = Result<T, GpuKernelError>;

/// Converts any displayable error from the `oxicuda-*` / `ndarray` stack
/// into a [`GpuKernelError::ComputationFailed`].
#[cfg(feature = "gpu")]
fn gpu_err<E: std::fmt::Display>(e: E) -> GpuKernelError {
    GpuKernelError::ComputationFailed(e.to_string())
}

/// Flattens an `ndarray::Array2` into a row-major `Vec`, uploading directly
/// from the underlying slice when possible and falling back to an iterator
/// copy for non-standard-layout views (e.g. slices/transposes) instead of
/// assuming `as_slice()` always succeeds.
#[cfg(feature = "gpu")]
fn flatten_row_major(array: &Array2<f32>) -> Vec<f32> {
    if array.is_standard_layout() {
        match array.as_slice() {
            Some(slice) => slice.to_vec(),
            None => array.iter().copied().collect(),
        }
    } else {
        array.iter().copied().collect()
    }
}

/// GPU-accelerated kernel matrix computer backed by `oxicuda-blas`.
///
/// The Linear kernel is the raw GEMM result (`X·Yᵀ`), downloaded as-is.
/// RBF and Sigmoid keep the GEMM result on-device and run their non-linear
/// transform there too -- `fill`/`axpy` build the affine pre-activation
/// (`γ·inner+c`, or the RBF equivalent with the norm terms folded in via
/// `axpy`'s accumulate), then `exp`/`tanh_activation` apply the activation,
/// all before a single download of the final matrix.
///
/// Polynomial still downloads `inner` and applies `(γ·inner+c)^d` on the CPU
/// via `f32::powf`: `oxicuda_blas::elementwise::pow` computes `a^b` through
/// an `lg2`/`ex2` approximation (`exp2(b * log2(a))`), which is undefined
/// for a negative `a`. SVM polynomial kernels routinely produce a negative
/// `γ·inner+c` together with an integer `degree` (e.g. `(-2.0)^3 == -8.0`,
/// which `f32::powf` gets right and the on-device approximation cannot), so
/// it is not a safe substitute here. A real signed-integer-power on-device
/// kernel would need its own PTX template -- that's a follow-up, not
/// something to improvise via `oxicuda-ptx::KernelBuilder` in this pass.
#[cfg(feature = "gpu")]
pub struct GpuKernelComputer {
    backend: GpuBackend,
}

#[cfg(feature = "gpu")]
impl GpuKernelComputer {
    pub async fn new() -> GpuKernelResult<Self> {
        let backend = GpuBackend::detect()
            .map_err(gpu_err)?
            .ok_or(GpuKernelError::DeviceNotAvailable)?;
        Ok(Self { backend })
    }

    /// Uploads `x` and `y` and computes `inner = X·Yᵀ` on the GPU via a
    /// single GEMM call (`Transpose::Trans` on the `Y` operand stands in for
    /// a host-side transpose, so `y` is uploaded in its natural layout).
    /// Returns the still-device-resident result together with its
    /// `(rows, cols)` shape.
    fn gemm_inner(
        &self,
        x: &Array2<f32>,
        y: &Array2<f32>,
    ) -> GpuKernelResult<(DeviceBuffer<f32>, usize, usize)> {
        self.backend.context().set_current().map_err(gpu_err)?;

        let (n_x, n_features) = x.dim();
        let (n_y, _) = y.dim();

        let x_buf = DeviceBuffer::<f32>::from_host(&flatten_row_major(x)).map_err(gpu_err)?;
        let y_buf = DeviceBuffer::<f32>::from_host(&flatten_row_major(y)).map_err(gpu_err)?;
        let mut inner_buf = DeviceBuffer::<f32>::zeroed(n_x * n_y).map_err(gpu_err)?;

        let a_desc =
            MatrixDesc::from_buffer(&x_buf, n_x as u32, n_features as u32, Layout::RowMajor)
                .map_err(gpu_err)?;
        let b_desc =
            MatrixDesc::from_buffer(&y_buf, n_y as u32, n_features as u32, Layout::RowMajor)
                .map_err(gpu_err)?;
        let mut c_desc =
            MatrixDescMut::from_buffer(&mut inner_buf, n_x as u32, n_y as u32, Layout::RowMajor)
                .map_err(gpu_err)?;

        gemm(
            self.backend.blas(),
            Transpose::NoTrans,
            Transpose::Trans,
            1.0f32,
            &a_desc,
            &b_desc,
            0.0f32,
            &mut c_desc,
        )
        .map_err(gpu_err)?;

        Ok((inner_buf, n_x, n_y))
    }

    /// Compute the kernel matrix `K[i,j] = k(x_i, y_j)` on the GPU.
    pub async fn compute_kernel_matrix(
        &self,
        x: &Array2<f32>,
        y: &Array2<f32>,
        kernel_type: &KernelType,
    ) -> GpuKernelResult<Array2<f32>> {
        if x.ncols() != y.ncols() {
            return Err(GpuKernelError::DimensionMismatch);
        }

        let (inner_buf, n_x, n_y) = self.gemm_inner(x, y)?;
        let n = n_x * n_y;
        let handle = self.backend.blas();

        let result_buf = match kernel_type {
            KernelType::Linear => inner_buf,
            KernelType::Rbf { gamma } => {
                let g = *gamma as f32;
                let x_norms: Vec<f32> = x
                    .rows()
                    .into_iter()
                    .map(|r| r.iter().map(|v| v * v).sum::<f32>())
                    .collect();
                let y_norms: Vec<f32> = y
                    .rows()
                    .into_iter()
                    .map(|r| r.iter().map(|v| v * v).sum::<f32>())
                    .collect();

                // pre[i,j] = -g * (x_norms[i] + y_norms[j]).
                // The norm vectors themselves are O(n_x)+O(n_y) -- cheap to
                // build on the CPU regardless of where the O(n_x*n_y)
                // transform runs.
                let mut pre_host = vec![0.0f32; n];
                for (i, x_norm) in x_norms.iter().enumerate() {
                    for (j, y_norm) in y_norms.iter().enumerate() {
                        pre_host[i * n_y + j] = -g * (x_norm + y_norm);
                    }
                }
                let mut pre_buf = DeviceBuffer::<f32>::from_host(&pre_host).map_err(gpu_err)?;

                // pre += 2g * inner
                //   => pre[i,j] = 2g*inner[i,j] - g*(x_norms[i]+y_norms[j])
                //               = -g * (x_norms[i] + y_norms[j] - 2*inner[i,j])
                //               = -g * sq[i,j]
                // This is the O(n_x*n_y) transform: it runs entirely
                // on-device, directly against the un-downloaded GEMM output.
                axpy(handle, n as u32, 2.0 * g, &inner_buf, 1, &mut pre_buf, 1)
                    .map_err(gpu_err)?;

                let mut out_buf = DeviceBuffer::<f32>::zeroed(n).map_err(gpu_err)?;
                exp(handle, n as u32, &pre_buf, &mut out_buf).map_err(gpu_err)?;
                out_buf
            }
            KernelType::Sigmoid { gamma, coef0 } => {
                let (g, c) = (*gamma as f32, *coef0 as f32);

                let mut pre_buf = DeviceBuffer::<f32>::zeroed(n).map_err(gpu_err)?;
                fill(handle, &mut pre_buf, c, n as u32).map_err(gpu_err)?;
                // pre = g*inner + pre = g*inner + c
                axpy(handle, n as u32, g, &inner_buf, 1, &mut pre_buf, 1).map_err(gpu_err)?;

                let mut out_buf = DeviceBuffer::<f32>::zeroed(n).map_err(gpu_err)?;
                tanh_activation(handle, n as u32, &pre_buf, &mut out_buf).map_err(gpu_err)?;
                out_buf
            }
            KernelType::Polynomial {
                gamma,
                coef0,
                degree,
            } => {
                // See the struct-level doc comment: the on-device `pow` is
                // unsound for a negative base, so this transform stays on
                // the CPU after a single download of `inner`.
                let mut inner_host = vec![0.0f32; n];
                inner_buf.copy_to_host(&mut inner_host).map_err(gpu_err)?;
                let (g, c, d) = (*gamma as f32, *coef0 as f32, *degree as f32);
                for value in &mut inner_host {
                    *value = (g * *value + c).powf(d);
                }
                return Array2::from_shape_vec((n_x, n_y), inner_host).map_err(gpu_err);
            }
            _ => {
                return Err(GpuKernelError::FeatureNotSupported(format!(
                    "Kernel {kernel_type:?} not supported on GPU"
                )));
            }
        };

        let mut out_host = vec![0.0f32; n];
        result_buf.copy_to_host(&mut out_host).map_err(gpu_err)?;
        Array2::from_shape_vec((n_x, n_y), out_host).map_err(gpu_err)
    }

    pub fn device_info(&self) -> String {
        if let Ok(mem) = self.backend.memory_info() {
            format!(
                "oxicuda-backend | total={} MB | free={} MB",
                mem.total / 1024 / 1024,
                mem.free / 1024 / 1024,
            )
        } else {
            "oxicuda-backend (CPU fallback)".to_string()
        }
    }

    pub fn supports_compute(&self) -> bool {
        true
    }
}

#[cfg(not(feature = "gpu"))]
pub struct GpuKernelComputer;

#[cfg(not(feature = "gpu"))]
impl GpuKernelComputer {
    pub async fn new() -> GpuKernelResult<Self> {
        Err(GpuKernelError::FeatureNotSupported(
            "GPU support not enabled".to_string(),
        ))
    }

    pub async fn compute_kernel_matrix(
        &self,
        _x: &Array2<f32>,
        _y: &Array2<f32>,
        _kernel_type: &KernelType,
    ) -> GpuKernelResult<Array2<f32>> {
        Err(GpuKernelError::FeatureNotSupported(
            "GPU support not enabled".to_string(),
        ))
    }
}

/// GPU-accelerated kernel function wrapper
pub struct GpuKernel {
    #[cfg(feature = "gpu")]
    computer: Option<GpuKernelComputer>,
    kernel_type: KernelType,
    use_gpu: bool,
}

impl GpuKernel {
    /// Create a new GPU-accelerated kernel
    pub fn new(kernel_type: KernelType, use_gpu: bool) -> Self {
        Self {
            #[cfg(feature = "gpu")]
            computer: None,
            kernel_type,
            use_gpu,
        }
    }

    /// Initialize GPU acceleration
    #[cfg(feature = "gpu")]
    pub async fn init_gpu(&mut self) -> GpuKernelResult<()> {
        if self.use_gpu {
            match GpuKernelComputer::new().await {
                Ok(computer) => self.computer = Some(computer),
                Err(GpuKernelError::DeviceNotAvailable) => {
                    // No GPU/driver present: leave `computer` unset so
                    // `compute_matrix` gracefully falls back to
                    // `compute_cpu_kernel_matrix` below. This mirrors
                    // `GpuBackend::detect()`'s `Ok(None)` contract -- missing
                    // hardware is an expected, non-error outcome, not a
                    // reason to fail `init_gpu` itself.
                    self.computer = None;
                }
                Err(e) => return Err(e),
            }
        }
        Ok(())
    }

    #[cfg(not(feature = "gpu"))]
    pub async fn init_gpu(&mut self) -> GpuKernelResult<()> {
        if self.use_gpu {
            return Err(GpuKernelError::FeatureNotSupported(
                "GPU support not enabled".to_string(),
            ));
        }
        Ok(())
    }

    /// Compute kernel matrix with GPU acceleration if available
    pub async fn compute_matrix(&self, x: &Array2<f32>, y: &Array2<f32>) -> Array2<f32> {
        #[cfg(feature = "gpu")]
        if let Some(computer) = &self.computer {
            if let Ok(result) = computer
                .compute_kernel_matrix(x, y, &self.kernel_type)
                .await
            {
                return result;
            }
        }

        // Fallback to CPU computation
        self.compute_cpu_kernel_matrix(x, y)
    }

    /// Compute kernel matrix on CPU
    pub fn compute_cpu_kernel_matrix(&self, x: &Array2<f32>, y: &Array2<f32>) -> Array2<f32> {
        let (n_x, _n_features) = x.dim();
        let (n_y, _) = y.dim();
        let mut result = Array2::zeros((n_x, n_y));

        for i in 0..n_x {
            for j in 0..n_y {
                let x_i = x.row(i);
                let y_j = y.row(j);

                let kernel_value = match &self.kernel_type {
                    KernelType::Linear => x_i.dot(&y_j) as f64,
                    KernelType::Rbf { gamma } => {
                        let diff = &x_i - &y_j;
                        let squared_distance = diff.dot(&diff) as f64;
                        (-gamma * squared_distance).exp()
                    }
                    KernelType::Polynomial {
                        gamma,
                        coef0,
                        degree,
                    } => {
                        let dot_product = x_i.dot(&y_j) as f64;
                        (gamma * dot_product + coef0).powf(*degree)
                    }
                    KernelType::Sigmoid { gamma, coef0 } => {
                        let dot_product = x_i.dot(&y_j) as f64;
                        (gamma * dot_product + coef0).tanh()
                    }
                    _ => 0.0, // Unsupported kernel types default to 0
                };

                result[(i, j)] = kernel_value as f32;
            }
        }

        result
    }

    /// Check if GPU is available and initialized
    pub fn is_gpu_available(&self) -> bool {
        #[cfg(feature = "gpu")]
        return self.computer.is_some();
        #[cfg(not(feature = "gpu"))]
        false
    }

    /// Get device information
    pub fn device_info(&self) -> String {
        #[cfg(feature = "gpu")]
        if let Some(computer) = &self.computer {
            return computer.device_info();
        }
        "CPU".to_string()
    }
}

/// Benchmark GPU vs CPU kernel computation
pub struct GpuKernelBenchmark {
    pub gpu_time: Option<std::time::Duration>,
    pub cpu_time: std::time::Duration,
    pub speedup: Option<f64>,
    pub accuracy: f64,
}

impl GpuKernelBenchmark {
    /// Run benchmark comparing GPU and CPU kernel computation
    pub async fn run(
        x: &Array2<f32>,
        y: &Array2<f32>,
        kernel_type: KernelType,
    ) -> GpuKernelResult<Self> {
        // CPU benchmark
        let cpu_start = std::time::Instant::now();
        let cpu_kernel = GpuKernel::new(kernel_type.clone(), false);
        #[cfg_attr(not(feature = "gpu"), allow(unused_variables))]
        let cpu_result = cpu_kernel.compute_cpu_kernel_matrix(x, y);
        let cpu_time = cpu_start.elapsed();

        // GPU benchmark
        #[cfg(feature = "gpu")]
        let (gpu_time, speedup, accuracy) = {
            if let Ok(computer) = GpuKernelComputer::new().await {
                let gpu_start = std::time::Instant::now();
                let gpu_result = computer.compute_kernel_matrix(x, y, &kernel_type).await?;
                let gpu_time = gpu_start.elapsed();

                // Compute accuracy
                let diff = &cpu_result - &gpu_result;
                let mse = diff.mapv(|x| x * x).mean().unwrap_or(0.0);
                let accuracy = 1.0 - (mse as f64).sqrt();

                let speedup = cpu_time.as_secs_f64() / gpu_time.as_secs_f64();

                (Some(gpu_time), Some(speedup), accuracy)
            } else {
                (None, None, 0.0)
            }
        };

        #[cfg(not(feature = "gpu"))]
        let (gpu_time, speedup, accuracy) = (None, None, 0.0);

        Ok(GpuKernelBenchmark {
            gpu_time,
            cpu_time,
            speedup,
            accuracy,
        })
    }
}

/// Utilities for GPU kernel optimization
pub mod gpu_utils {
    use super::*;

    /// Optimal batch size for GPU computation
    pub fn optimal_batch_size(n_samples: usize, n_features: usize) -> usize {
        // Heuristic: balance memory usage and parallelism
        let memory_limit = 1024 * 1024 * 1024; // 1GB limit
        let sample_size = n_features * std::mem::size_of::<f32>();
        let max_batch = memory_limit / sample_size;

        (max_batch.min(n_samples)).max(1)
    }

    /// Check if GPU acceleration is beneficial
    pub fn should_use_gpu(n_samples: usize, n_features: usize) -> bool {
        // Use GPU for large datasets where parallel computation is beneficial
        let computation_size = n_samples * n_samples * n_features;
        computation_size > 1_000_000 // Threshold for GPU acceleration
    }

    /// Batch kernel matrix computation for large datasets
    pub async fn compute_kernel_matrix_batched(
        computer: &GpuKernelComputer,
        x: &Array2<f32>,
        y: &Array2<f32>,
        kernel_type: &KernelType,
        batch_size: usize,
    ) -> GpuKernelResult<Array2<f32>> {
        let (n_x, _n_features) = x.dim();
        let (n_y, _) = y.dim();

        let mut result = Array2::zeros((n_x, n_y));

        let n_feat = x.ncols();
        for i in (0..n_x).step_by(batch_size) {
            let end_i = (i + batch_size).min(n_x);
            let n_xi = end_i - i;
            let x_batch: Array2<f32> = Array2::from_shape_vec(
                (n_xi, n_feat),
                x.rows()
                    .into_iter()
                    .skip(i)
                    .take(n_xi)
                    .flat_map(|r| r.to_vec())
                    .collect(),
            )
            .map_err(|e| GpuKernelError::ComputationFailed(e.to_string()))?;

            for j in (0..n_y).step_by(batch_size) {
                let end_j = (j + batch_size).min(n_y);
                let n_yj = end_j - j;
                let y_batch: Array2<f32> = Array2::from_shape_vec(
                    (n_yj, n_feat),
                    y.rows()
                        .into_iter()
                        .skip(j)
                        .take(n_yj)
                        .flat_map(|r| r.to_vec())
                        .collect(),
                )
                .map_err(|e| GpuKernelError::ComputationFailed(e.to_string()))?;

                let batch_result: Array2<f32> = computer
                    .compute_kernel_matrix(&x_batch, &y_batch, kernel_type)
                    .await?;

                for bi in 0..n_xi {
                    for bj in 0..n_yj {
                        result[(i + bi, j + bj)] = batch_result[(bi, bj)];
                    }
                }
            }
        }

        Ok(result)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_kernel_creation() {
        let kernel = GpuKernel::new(KernelType::Rbf { gamma: 1.0 }, true);
        assert!(!kernel.is_gpu_available()); // Not initialized yet
    }

    #[test]
    fn test_gpu_kernel_sync() {
        let kernel = GpuKernel::new(KernelType::Linear, true);
        // Test synchronous operations only for now
        assert_eq!(kernel.device_info(), "CPU");
    }

    #[test]
    fn test_gpu_utils() {
        let batch_size = gpu_utils::optimal_batch_size(1000, 100);
        assert!(batch_size > 0);

        let should_use = gpu_utils::should_use_gpu(1000, 1000);
        assert!(should_use);

        let should_not_use = gpu_utils::should_use_gpu(10, 10);
        assert!(!should_not_use);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_benchmark_sync() {
        let X_var = Array2::from_shape_vec((10, 5), (0..50).map(|x| x as f32).collect())
            .expect("array shape mismatch");
        let Y_var = Array2::from_shape_vec((8, 5), (0..40).map(|x| x as f32).collect())
            .expect("array shape mismatch");

        // Test that we can create the benchmark struct
        assert_eq!(X_var.dim(), (10, 5));
        assert_eq!(Y_var.dim(), (8, 5));
    }

    /// Compares the on-device GEMM+transform path against
    /// `compute_cpu_kernel_matrix` for every GPU-supported kernel type.
    ///
    /// On this crate's own dev/CI machine (no CUDA device),
    /// `GpuKernelComputer::new()` returns `Err(DeviceNotAvailable)` and the
    /// test exits early -- exactly the graceful-fallback branch
    /// `GpuKernel::init_gpu`/`compute_matrix` take in production. On a real
    /// GPU runner this drives the on-device Rbf/Sigmoid transform (and the
    /// GEMM-only Linear/Polynomial paths) end-to-end and checks the result
    /// against the CPU reference within an f32-appropriate tolerance, so the
    /// comparison is meaningful there too.
    #[cfg(feature = "gpu")]
    #[test]
    fn test_gpu_kernel_matches_cpu_reference() {
        let x = Array2::from_shape_vec(
            (4, 3),
            vec![
                1.0, 2.0, 3.0, -1.0, 0.5, 2.0, 0.0, 0.0, 1.0, 2.5, -1.5, 0.5,
            ],
        )
        .expect("valid 4x3 shape");
        let y = Array2::from_shape_vec((3, 3), vec![1.0, 1.0, 1.0, -2.0, 1.0, 0.0, 0.5, 0.5, 0.5])
            .expect("valid 3x3 shape");

        let kernel_types = [
            KernelType::Linear,
            KernelType::Rbf { gamma: 0.5 },
            KernelType::Polynomial {
                gamma: 0.3,
                coef0: 1.0,
                degree: 3.0,
            },
            KernelType::Sigmoid {
                gamma: 0.2,
                coef0: -0.5,
            },
        ];

        pollster::block_on(async {
            let computer = match GpuKernelComputer::new().await {
                Ok(computer) => computer,
                Err(GpuKernelError::DeviceNotAvailable) => {
                    eprintln!(
                        "test_gpu_kernel_matches_cpu_reference: no GPU detected, skipping on-device comparison"
                    );
                    return;
                }
                Err(e) => panic!("unexpected GPU init error: {e}"),
            };

            for kernel_type in &kernel_types {
                let gpu_result = computer
                    .compute_kernel_matrix(&x, &y, kernel_type)
                    .await
                    .expect("GPU kernel computation should succeed");

                let cpu_kernel = GpuKernel::new(kernel_type.clone(), false);
                let cpu_result = cpu_kernel.compute_cpu_kernel_matrix(&x, &y);

                assert_eq!(
                    gpu_result.dim(),
                    cpu_result.dim(),
                    "shape mismatch for {kernel_type:?}"
                );
                for (g, c) in gpu_result.iter().zip(cpu_result.iter()) {
                    assert!(
                        (g - c).abs() < 1e-3,
                        "{kernel_type:?}: gpu={g} cpu={c} diff={}",
                        (g - c).abs()
                    );
                }
            }
        });
    }
}
