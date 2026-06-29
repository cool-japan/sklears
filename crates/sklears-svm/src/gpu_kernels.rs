//! GPU-accelerated kernel computations using WGPU
//!
//! This module provides GPU-accelerated implementations of common SVM kernels
//! including RBF, Polynomial, and Linear kernels. It uses WGPU for cross-platform
//! GPU acceleration and compute shaders for high-performance kernel matrix computation.

use crate::kernels::KernelType;
use scirs2_core::ndarray::Array2;
use thiserror::Error;

#[cfg(feature = "gpu")]
use sklears_core::gpu::{GpuArray, GpuContext, GpuMatrixOps};

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

/// GPU-accelerated kernel matrix computer backed by `oxicuda-backend`.
///
/// The Linear kernel uses direct GEMM (`X·Y^T`).
/// RBF, Polynomial, and Sigmoid kernels use GEMM for inner products, then
/// apply the non-linear function on the CPU.
#[cfg(feature = "gpu")]
pub struct GpuKernelComputer {
    context: GpuContext,
}

#[cfg(feature = "gpu")]
impl GpuKernelComputer {
    pub async fn new() -> GpuKernelResult<Self> {
        let context =
            GpuContext::new().map_err(|e| GpuKernelError::ComputationFailed(e.to_string()))?;
        Ok(Self { context })
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

        // GPU GEMM: inner = X · Y^T  [n_x × n_y]
        let x_gpu = GpuArray::<f32>::from_array2(&self.context, x)
            .map_err(|e| GpuKernelError::ComputationFailed(e.to_string()))?;
        let yt_data: Vec<f32> = {
            let (nr, nc) = y.dim();
            let mut t = vec![0.0f32; nr * nc];
            for r in 0..nr {
                for c in 0..nc {
                    t[c * nr + r] = y[[r, c]];
                }
            }
            t
        };
        let yt = scirs2_core::ndarray::Array2::from_shape_vec((y.ncols(), y.nrows()), yt_data)
            .map_err(|e| GpuKernelError::ComputationFailed(e.to_string()))?;
        let yt_gpu = GpuArray::<f32>::from_array2(&self.context, &yt)
            .map_err(|e| GpuKernelError::ComputationFailed(e.to_string()))?;
        let inner_gpu = x_gpu
            .matmul(&yt_gpu)
            .map_err(|e| GpuKernelError::ComputationFailed(e.to_string()))?;
        let inner = inner_gpu
            .to_array2()
            .map_err(|e| GpuKernelError::ComputationFailed(e.to_string()))?;

        let (n_x, n_y) = inner.dim();
        let mut out = Array2::<f32>::zeros((n_x, n_y));

        match kernel_type {
            KernelType::Linear => {
                for i in 0..n_x {
                    for j in 0..n_y {
                        out[[i, j]] = inner[[i, j]];
                    }
                }
            }
            KernelType::Rbf { gamma } => {
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
                let g = *gamma as f32;
                for i in 0..n_x {
                    for j in 0..n_y {
                        let sq = (x_norms[i] + y_norms[j] - 2.0 * inner[[i, j]]).max(0.0);
                        out[[i, j]] = (-g * sq).exp();
                    }
                }
            }
            KernelType::Polynomial {
                gamma,
                coef0,
                degree,
            } => {
                let (g, c, d) = (*gamma as f32, *coef0 as f32, *degree as f32);
                for i in 0..n_x {
                    for j in 0..n_y {
                        out[[i, j]] = (g * inner[[i, j]] + c).powf(d);
                    }
                }
            }
            KernelType::Sigmoid { gamma, coef0 } => {
                let (g, c) = (*gamma as f32, *coef0 as f32);
                for i in 0..n_x {
                    for j in 0..n_y {
                        out[[i, j]] = (g * inner[[i, j]] + c).tanh();
                    }
                }
            }
            _ => {
                return Err(GpuKernelError::FeatureNotSupported(format!(
                    "Kernel {:?} not supported on GPU",
                    kernel_type
                )))
            }
        }

        Ok(out)
    }

    pub fn device_info(&self) -> String {
        if let Ok(mem) = self.context.memory_info() {
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
            self.computer = Some(GpuKernelComputer::new().await?);
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
}
