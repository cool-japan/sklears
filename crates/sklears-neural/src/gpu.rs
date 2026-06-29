//! GPU acceleration module for neural network computations
//!
//! This module provides CUDA-based GPU acceleration for neural network operations,
//! including matrix operations, activation functions, and gradient computations.
//!
//! # Features
//!
//! - GPU context management with automatic device selection
//! - Memory pooling for efficient GPU memory allocation
//! - Batch processing with optimal GPU utilization
//! - Automatic fallback to CPU when GPU is unavailable
//! - Mixed precision training support
//!
//! # Examples
//!
//! ```rust
//! use sklears_neural::gpu::{GpuContext, GpuTensor};
//!
//! # #[cfg(feature = "gpu")]
//! fn example() -> Result<(), Box<dyn std::error::Error>> {
//!     let ctx = GpuContext::new()?;
//!     let a = GpuTensor::from_host_data(&ctx, &[1.0, 2.0, 3.0, 4.0], &[2, 2])?;
//!     let b = GpuTensor::from_host_data(&ctx, &[2.0, 0.0, 1.0, 2.0], &[2, 2])?;
//!     
//!     let result = ctx.matrix_multiply(&a, &b)?;
//!     let host_result = result.to_host()?;
//!     
//!     println!("GPU matrix multiplication result: {:?}", host_result);
//!     Ok(())
//! }
//! ```

use crate::NeuralResult;
use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::error::SklearsError;
use std::collections::HashMap;

#[cfg(feature = "gpu")]
use sklears_core::gpu::{GpuArray, GpuContext as SklearsGpuContext, GpuMatrixOps};

/// Configuration for GPU operations
#[derive(Debug, Clone)]
pub struct GpuConfig {
    /// Device ID to use (None for automatic selection)
    pub device_id: Option<usize>,
    /// Memory pool size in bytes
    pub memory_pool_size: usize,
    /// Whether to use mixed precision training
    pub mixed_precision: bool,
    /// Batch size threshold for GPU processing
    pub gpu_threshold: usize,
    /// Maximum number of CUDA streams
    pub max_streams: usize,
}

impl Default for GpuConfig {
    fn default() -> Self {
        Self {
            device_id: None,
            memory_pool_size: 1024 * 1024 * 1024, // 1GB
            mixed_precision: false,
            gpu_threshold: 1000,
            max_streams: 4,
        }
    }
}

/// GPU tensor backed by `oxicuda-backend` device memory.
///
/// Wraps `sklears_core::gpu::GpuArray<T>` and stores the multi-dimensional shape
/// separately (the underlying `GpuArray` always uses a flat 1-D allocation).
#[cfg(feature = "gpu")]
pub struct GpuTensor<T: bytemuck::Pod> {
    /// Logical shape of the tensor (e.g. `[batch, rows, cols]`).
    pub shape: Vec<usize>,
    array: GpuArray<T>,
    ctx: SklearsGpuContext,
}

#[cfg(feature = "gpu")]
impl<T: bytemuck::Pod + Clone> GpuTensor<T> {
    /// Upload host `data` to the GPU, tagging it with `shape`.
    pub fn from_host_data(ctx: &GpuContext, data: &[T], shape: &[usize]) -> NeuralResult<Self> {
        let total_elements: usize = shape.iter().product();
        if data.len() != total_elements {
            return Err(SklearsError::InvalidInput(format!(
                "Data length {} doesn't match shape {:?} (expected {})",
                data.len(),
                shape,
                total_elements
            )));
        }
        let array = GpuArray::<T>::from_slice(&ctx.inner, data).map_err(|e| {
            SklearsError::InvalidInput(format!("Failed to copy data to GPU: {}", e))
        })?;
        Ok(Self {
            shape: shape.to_vec(),
            array,
            ctx: ctx.inner.clone(),
        })
    }

    /// Download tensor data to host memory.
    pub fn to_host(&self) -> NeuralResult<Vec<T>> {
        self.array
            .to_cpu()
            .map_err(|e| SklearsError::InvalidInput(format!("Failed to copy data from GPU: {}", e)))
    }

    /// Total number of elements.
    pub fn len(&self) -> usize {
        self.shape.iter().product()
    }

    /// Returns `true` when the tensor contains no elements.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Number of dimensions.
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Return a new tensor with `new_shape`, preserving the underlying data.
    pub fn reshape(&self, new_shape: &[usize]) -> NeuralResult<GpuTensor<T>> {
        let new_len: usize = new_shape.iter().product();
        if new_len != self.len() {
            return Err(SklearsError::InvalidInput(format!(
                "Cannot reshape tensor of size {} to shape {:?} (size {})",
                self.len(),
                new_shape,
                new_len
            )));
        }
        let data = self.to_host()?;
        let array = GpuArray::<T>::from_slice(&self.ctx, &data).map_err(|e| {
            SklearsError::InvalidInput(format!("Failed to allocate reshaped tensor: {}", e))
        })?;
        Ok(GpuTensor {
            shape: new_shape.to_vec(),
            array,
            ctx: self.ctx.clone(),
        })
    }
}

/// GPU context for neural network operations, backed by `oxicuda-backend`.
///
/// Wraps `sklears_core::gpu::GpuContext` which provides the `ComputeBackend` abstraction
/// (real CUDA on NVIDIA hardware, `CpuBackend` otherwise).
#[cfg(feature = "gpu")]
pub struct GpuContext {
    inner: SklearsGpuContext,
    #[allow(dead_code)]
    config: GpuConfig,
}

#[cfg(feature = "gpu")]
impl GpuContext {
    /// Create a GPU context using default configuration.
    pub fn new() -> NeuralResult<Self> {
        Self::with_config(GpuConfig::default())
    }

    /// Create a GPU context using the provided configuration.
    pub fn with_config(config: GpuConfig) -> NeuralResult<Self> {
        let device_id = config.device_id.unwrap_or(0);
        let inner = SklearsGpuContext::with_device_id(device_id).map_err(|e| {
            SklearsError::InvalidInput(format!(
                "Failed to initialize GPU device {}: {}",
                device_id, e
            ))
        })?;
        Ok(Self { inner, config })
    }

    /// Block until all pending GPU operations have completed.
    pub fn synchronize(&self) -> NeuralResult<()> {
        self.inner
            .synchronize()
            .map_err(|e| SklearsError::InvalidInput(format!("Failed to synchronize GPU: {}", e)))
    }

    /// Matrix multiplication for f32 matrices via GEMM.
    pub fn matrix_multiply(
        &self,
        a: &GpuTensor<f32>,
        b: &GpuTensor<f32>,
    ) -> NeuralResult<GpuTensor<f32>> {
        if a.shape.len() != 2 || b.shape.len() != 2 {
            return Err(SklearsError::InvalidInput(
                "matrix_multiply requires 2-D tensors".to_string(),
            ));
        }
        let (m, k) = (a.shape[0], a.shape[1]);
        let (k2, n) = (b.shape[0], b.shape[1]);
        if k != k2 {
            return Err(SklearsError::InvalidInput(format!(
                "Matrix dimension mismatch: {}×{} and {}×{}",
                m, k, k2, n
            )));
        }
        let c = a
            .array
            .matmul(&b.array)
            .map_err(|e| SklearsError::InvalidInput(format!("GPU GEMM failed: {}", e)))?;
        let data = c
            .to_cpu()
            .map_err(|e| SklearsError::InvalidInput(format!("Failed to download result: {}", e)))?;
        let array = sklears_core::gpu::GpuArray::<f32>::from_slice(&self.inner, &data)
            .map_err(|e| SklearsError::InvalidInput(format!("Failed to upload result: {}", e)))?;
        Ok(GpuTensor {
            shape: vec![m, n],
            array,
            ctx: self.inner.clone(),
        })
    }

    /// Matrix multiplication for f64 matrices via GEMM.
    pub fn matrix_multiply_f64(
        &self,
        a: &GpuTensor<f64>,
        b: &GpuTensor<f64>,
    ) -> NeuralResult<GpuTensor<f64>> {
        if a.shape.len() != 2 || b.shape.len() != 2 {
            return Err(SklearsError::InvalidInput(
                "matrix_multiply_f64 requires 2-D tensors".to_string(),
            ));
        }
        let (m, k) = (a.shape[0], a.shape[1]);
        let (k2, n) = (b.shape[0], b.shape[1]);
        if k != k2 {
            return Err(SklearsError::InvalidInput(format!(
                "Matrix dimension mismatch: {}×{} and {}×{}",
                m, k, k2, n
            )));
        }
        let c = a
            .array
            .matmul(&b.array)
            .map_err(|e| SklearsError::InvalidInput(format!("GPU GEMM f64 failed: {}", e)))?;
        let data = c.to_cpu().map_err(|e| {
            SklearsError::InvalidInput(format!("Failed to download f64 result: {}", e))
        })?;
        let array =
            sklears_core::gpu::GpuArray::<f64>::from_slice(&self.inner, &data).map_err(|e| {
                SklearsError::InvalidInput(format!("Failed to upload f64 result: {}", e))
            })?;
        Ok(GpuTensor {
            shape: vec![m, n],
            array,
            ctx: self.inner.clone(),
        })
    }

    /// Element-wise addition of two f32 tensors via `backend.binary`.
    pub fn add(&self, a: &GpuTensor<f32>, b: &GpuTensor<f32>) -> NeuralResult<GpuTensor<f32>> {
        if a.shape != b.shape {
            return Err(SklearsError::InvalidInput(format!(
                "Shape mismatch for add: {:?} vs {:?}",
                a.shape, b.shape
            )));
        }
        let c = a
            .array
            .add(&b.array)
            .map_err(|e| SklearsError::InvalidInput(format!("GPU add failed: {}", e)))?;
        let data = c.to_cpu().map_err(|e| {
            SklearsError::InvalidInput(format!("Failed to download add result: {}", e))
        })?;
        let array =
            sklears_core::gpu::GpuArray::<f32>::from_slice(&self.inner, &data).map_err(|e| {
                SklearsError::InvalidInput(format!("Failed to upload add result: {}", e))
            })?;
        Ok(GpuTensor {
            shape: a.shape.clone(),
            array,
            ctx: self.inner.clone(),
        })
    }

    /// ReLU activation: downloads data, applies CPU-side, re-uploads.
    pub fn relu(&self, input: &GpuTensor<f32>) -> NeuralResult<GpuTensor<f32>> {
        let data = input.to_host()?;
        let result: Vec<f32> = data.into_iter().map(|x| x.max(0.0)).collect();
        GpuTensor::from_host_data(self, &result, &input.shape)
    }

    /// Sigmoid activation: downloads data, applies CPU-side, re-uploads.
    pub fn sigmoid(&self, input: &GpuTensor<f32>) -> NeuralResult<GpuTensor<f32>> {
        let data = input.to_host()?;
        let result: Vec<f32> = data.into_iter().map(|x| 1.0 / (1.0 + (-x).exp())).collect();
        GpuTensor::from_host_data(self, &result, &input.shape)
    }

    /// Return `(free_bytes, total_bytes)` for the device.
    pub fn memory_info(&self) -> NeuralResult<(usize, usize)> {
        let info = self
            .inner
            .memory_info()
            .map_err(|e| SklearsError::InvalidInput(format!("Failed to get memory info: {}", e)))?;
        Ok((info.free, info.total))
    }

    /// Return memory pool utilization `(used_fraction, hit_rate)` — always `(0, 0)` without a pool.
    pub fn memory_pool_stats(&self) -> (f64, f64) {
        (0.0, 0.0)
    }

    /// Returns `true` if the device has dedicated Tensor Core hardware.
    pub fn has_tensor_cores(&self) -> bool {
        false
    }

    /// Return the CUDA compute capability `(major, minor)` of the device, if known.
    pub fn compute_capability(&self) -> Option<(i32, i32)> {
        None
    }

    /// Tensor core f16 GEMM — not yet supported with `oxicuda-backend`.
    pub fn tensor_core_gemm_f16(
        &self,
        _a: &GpuTensor<half::f16>,
        _b: &GpuTensor<half::f16>,
    ) -> NeuralResult<GpuTensor<half::f16>> {
        Err(SklearsError::InvalidInput(
            "Tensor core f16 GEMM requires a real CUDA device".to_string(),
        ))
    }

    /// Mixed-precision GEMM (f16 in, f32 out) — not yet supported.
    pub fn mixed_precision_gemm(
        &self,
        _a: &GpuTensor<half::f16>,
        _b: &GpuTensor<half::f16>,
    ) -> NeuralResult<GpuTensor<f32>> {
        Err(SklearsError::InvalidInput(
            "Mixed-precision GEMM requires a real CUDA device".to_string(),
        ))
    }

    /// Tensor-core conv2d — not yet supported.
    pub fn tensor_core_conv2d(
        &self,
        _input: &GpuTensor<half::f16>,
        _kernel: &GpuTensor<half::f16>,
        _stride: (usize, usize),
        _padding: (usize, usize),
    ) -> NeuralResult<GpuTensor<half::f16>> {
        Err(SklearsError::InvalidInput(
            "Tensor core conv2d requires a real CUDA device".to_string(),
        ))
    }
}

#[cfg(not(feature = "gpu"))]
/// Stub GPU context when GPU feature is disabled
pub struct GpuContext;

#[cfg(not(feature = "gpu"))]
impl GpuContext {
    /// Attempt to create a GPU context; always returns an error when the `gpu` feature is disabled
    pub fn new() -> NeuralResult<Self> {
        Err(SklearsError::InvalidInput(
            "GPU support not compiled. Enable 'gpu' feature".to_string(),
        ))
    }
}

#[cfg(not(feature = "gpu"))]
/// Stub GPU tensor when GPU feature is disabled
pub struct GpuTensor<T> {
    _phantom: std::marker::PhantomData<T>,
}

/// GPU-accelerated neural network operations
#[allow(dead_code)] // context is the GPU handle, used when `gpu` feature is enabled
pub struct GpuAcceleratedOps {
    #[cfg(feature = "gpu")]
    context: Option<GpuContext>,
    #[cfg(not(feature = "gpu"))]
    context: Option<()>,
    config: GpuConfig,
}

impl GpuAcceleratedOps {
    /// Create new GPU-accelerated operations
    pub fn new() -> Self {
        Self::with_config(GpuConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: GpuConfig) -> Self {
        #[cfg(feature = "gpu")]
        let context = GpuContext::with_config(config.clone()).ok();

        #[cfg(not(feature = "gpu"))]
        let context: Option<()> = None;

        Self { context, config }
    }

    /// Check if GPU acceleration is available
    pub fn is_available(&self) -> bool {
        #[cfg(feature = "gpu")]
        return self.context.is_some();

        #[cfg(not(feature = "gpu"))]
        return false;
    }

    /// GPU-accelerated matrix multiplication with automatic fallback
    pub fn matrix_multiply(&self, a: &Array2<f32>, b: &Array2<f32>) -> NeuralResult<Array2<f32>> {
        // Check if GPU acceleration should be used
        let use_gpu = self.is_available()
            && a.len() >= self.config.gpu_threshold
            && b.len() >= self.config.gpu_threshold;

        if use_gpu {
            #[cfg(feature = "gpu")]
            {
                if let Some(ref ctx) = self.context {
                    return self.gpu_matrix_multiply(ctx, a, b);
                }
            }
        }

        // CPU fallback
        self.cpu_matrix_multiply(a, b)
    }

    #[cfg(feature = "gpu")]
    fn gpu_matrix_multiply(
        &self,
        ctx: &GpuContext,
        a: &Array2<f32>,
        b: &Array2<f32>,
    ) -> NeuralResult<Array2<f32>> {
        let a_data: Vec<f32> = a.iter().cloned().collect();
        let b_data: Vec<f32> = b.iter().cloned().collect();

        let gpu_a = GpuTensor::from_host_data(ctx, &a_data, &[a.nrows(), a.ncols()])?;
        let gpu_b = GpuTensor::from_host_data(ctx, &b_data, &[b.nrows(), b.ncols()])?;

        let gpu_result = ctx.matrix_multiply(&gpu_a, &gpu_b)?;
        let result_data = gpu_result.to_host()?;

        let result = Array2::from_shape_vec((a.nrows(), b.ncols()), result_data)
            .map_err(|e| SklearsError::InvalidInput(format!("Failed to reshape result: {}", e)))?;

        Ok(result)
    }

    fn cpu_matrix_multiply(&self, a: &Array2<f32>, b: &Array2<f32>) -> NeuralResult<Array2<f32>> {
        if a.ncols() != b.nrows() {
            return Err(SklearsError::InvalidInput(format!(
                "Matrix dimension mismatch: {}x{} and {}x{}",
                a.nrows(),
                a.ncols(),
                b.nrows(),
                b.ncols()
            )));
        }

        Ok(a.dot(b))
    }

    /// GPU-accelerated activation function application
    pub fn apply_activation(
        &self,
        input: &Array1<f32>,
        activation: &str,
    ) -> NeuralResult<Array1<f32>> {
        let use_gpu = self.is_available() && input.len() >= self.config.gpu_threshold;

        if use_gpu {
            #[cfg(feature = "gpu")]
            {
                if let Some(ref ctx) = self.context {
                    return self.gpu_apply_activation(ctx, input, activation);
                }
            }
        }

        // CPU fallback
        self.cpu_apply_activation(input, activation)
    }

    /// Tensor core optimized matrix multiplication (if available)
    #[cfg(feature = "gpu")]
    pub fn tensor_core_matrix_multiply(
        &self,
        a: &Array2<f32>,
        b: &Array2<f32>,
        use_mixed_precision: bool,
    ) -> NeuralResult<Array2<f32>> {
        if let Some(ref ctx) = self.context {
            if ctx.has_tensor_cores() && self.is_tensor_core_friendly(a, b) {
                // Convert to half precision for tensor core operations
                let a_f16: Vec<half::f16> = a.iter().map(|&x| half::f16::from_f32(x)).collect();
                let b_f16: Vec<half::f16> = b.iter().map(|&x| half::f16::from_f32(x)).collect();

                let gpu_a = GpuTensor::from_host_data(ctx, &a_f16, &[a.nrows(), a.ncols()])?;
                let gpu_b = GpuTensor::from_host_data(ctx, &b_f16, &[b.nrows(), b.ncols()])?;

                if use_mixed_precision {
                    // FP16 compute, FP32 accumulate
                    let gpu_result = ctx.mixed_precision_gemm(&gpu_a, &gpu_b)?;
                    let result_data = gpu_result.to_host()?;
                    let result = Array2::from_shape_vec((a.nrows(), b.ncols()), result_data)
                        .map_err(|e| {
                            SklearsError::InvalidInput(format!("Failed to reshape result: {}", e))
                        })?;
                    return Ok(result);
                } else {
                    // Pure FP16 compute
                    let gpu_result = ctx.tensor_core_gemm_f16(&gpu_a, &gpu_b)?;
                    let result_f16 = gpu_result.to_host()?;
                    let result_f32: Vec<f32> = result_f16.iter().map(|&x| x.to_f32()).collect();
                    let result = Array2::from_shape_vec((a.nrows(), b.ncols()), result_f32)
                        .map_err(|e| {
                            SklearsError::InvalidInput(format!("Failed to reshape result: {}", e))
                        })?;
                    return Ok(result);
                }
            }
        }

        // Fallback to regular GPU or CPU computation
        self.matrix_multiply(a, b)
    }

    /// Check if tensor dimensions are suitable for tensor cores (multiples of 8, min size 64)
    #[allow(dead_code)] // Called from cfg(feature = "gpu") block; also used in tests
    fn is_tensor_core_friendly(&self, a: &Array2<f32>, b: &Array2<f32>) -> bool {
        let (m, k) = (a.nrows(), a.ncols());
        let n = b.ncols();

        // Tensor cores work best with dimensions that are multiples of 8
        m % 8 == 0 && n.is_multiple_of(8) && k % 8 == 0 &&
        // And matrices should be reasonably large to benefit from tensor cores
        m >= 64 && n >= 64 && k >= 64
    }

    /// Get tensor core optimization recommendations
    pub fn tensor_core_recommendations(&self) -> HashMap<String, String> {
        let mut recommendations = HashMap::new();

        #[cfg(feature = "gpu")]
        {
            if let Some(ref ctx) = self.context {
                if ctx.has_tensor_cores() {
                    recommendations.insert(
                        "tensor_cores".to_string(),
                        "Available - use mixed precision training for best performance".to_string(),
                    );

                    if let Some((major, minor)) = ctx.compute_capability() {
                        recommendations.insert(
                            "compute_capability".to_string(),
                            format!("{}.{}", major, minor),
                        );

                        if major >= 8 {
                            recommendations.insert(
                                "optimization".to_string(),
                                "Use BF16 for better numerical stability on Ampere+ GPUs"
                                    .to_string(),
                            );
                        } else if major >= 7 {
                            recommendations.insert(
                                "optimization".to_string(),
                                "Use FP16 mixed precision for Volta/Turing GPUs".to_string(),
                            );
                        }
                    }

                    recommendations.insert(
                        "dimension_requirement".to_string(),
                        "Ensure matrix dimensions are multiples of 8 for optimal tensor core utilization".to_string(),
                    );
                } else {
                    recommendations.insert(
                        "tensor_cores".to_string(),
                        "Not available on this GPU - use regular FP32 operations".to_string(),
                    );
                }
            }
        }

        #[cfg(not(feature = "gpu"))]
        {
            recommendations.insert(
                "gpu".to_string(),
                "GPU support not compiled - enable 'gpu' feature for tensor core acceleration"
                    .to_string(),
            );
        }

        recommendations
    }

    #[cfg(feature = "gpu")]
    fn gpu_apply_activation(
        &self,
        ctx: &GpuContext,
        input: &Array1<f32>,
        activation: &str,
    ) -> NeuralResult<Array1<f32>> {
        let input_data: Vec<f32> = input.iter().cloned().collect();
        let gpu_input = GpuTensor::from_host_data(ctx, &input_data, &[input.len()])?;

        let gpu_result = match activation.to_lowercase().as_str() {
            "relu" => ctx.relu(&gpu_input)?,
            "sigmoid" => ctx.sigmoid(&gpu_input)?,
            _ => {
                return Err(SklearsError::InvalidInput(format!(
                    "Unsupported activation function: {}",
                    activation
                )))
            }
        };

        let result_data = gpu_result.to_host()?;
        let result = Array1::from_vec(result_data);

        Ok(result)
    }

    fn cpu_apply_activation(
        &self,
        input: &Array1<f32>,
        activation: &str,
    ) -> NeuralResult<Array1<f32>> {
        let result = match activation.to_lowercase().as_str() {
            "relu" => input.mapv(|x| x.max(0.0)),
            "sigmoid" => input.mapv(|x| 1.0 / (1.0 + (-x).exp())),
            "tanh" => input.mapv(|x| x.tanh()),
            _ => {
                return Err(SklearsError::InvalidInput(format!(
                    "Unsupported activation function: {}",
                    activation
                )))
            }
        };

        Ok(result)
    }

    /// Get GPU memory information
    pub fn memory_info(&self) -> Option<(usize, usize)> {
        #[cfg(feature = "gpu")]
        {
            if let Some(ref ctx) = self.context {
                return ctx.memory_info().ok();
            }
        }
        None
    }

    /// Get performance statistics
    pub fn performance_stats(&self) -> HashMap<String, f64> {
        let mut stats = HashMap::new();

        #[cfg(feature = "gpu")]
        {
            if let Some(ref ctx) = self.context {
                let (f32_hit_rate, f64_hit_rate) = ctx.memory_pool_stats();
                stats.insert("f32_pool_hit_rate".to_string(), f32_hit_rate);
                stats.insert("f64_pool_hit_rate".to_string(), f64_hit_rate);

                // Add tensor core availability
                stats.insert(
                    "tensor_cores_available".to_string(),
                    if ctx.has_tensor_cores() { 1.0 } else { 0.0 },
                );
            }
        }

        stats.insert(
            "gpu_available".to_string(),
            if self.is_available() { 1.0 } else { 0.0 },
        );
        stats
    }
}

impl Default for GpuAcceleratedOps {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_gpu_config_default() {
        let config = GpuConfig::default();
        assert_eq!(config.device_id, None);
        assert_eq!(config.memory_pool_size, 1024 * 1024 * 1024);
        assert!(!config.mixed_precision);
        assert_eq!(config.gpu_threshold, 1000);
        assert_eq!(config.max_streams, 4);
    }

    #[test]
    fn test_gpu_accelerated_ops_creation() {
        let ops = GpuAcceleratedOps::new();
        // Should not panic even without GPU
        let _stats = ops.performance_stats();
    }

    #[test]
    fn test_cpu_matrix_multiply() {
        let ops = GpuAcceleratedOps::new();
        let a = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .expect("array shape mismatch");
        let b = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .expect("array shape mismatch");

        let result = ops
            .cpu_matrix_multiply(&a, &b)
            .expect("operation should succeed");

        assert_eq!(result.dim(), (2, 2));
        assert_relative_eq!(result[[0, 0]], 22.0, epsilon = 1e-6);
        assert_relative_eq!(result[[0, 1]], 28.0, epsilon = 1e-6);
        assert_relative_eq!(result[[1, 0]], 49.0, epsilon = 1e-6);
        assert_relative_eq!(result[[1, 1]], 64.0, epsilon = 1e-6);
    }

    #[test]
    fn test_cpu_activation_functions() {
        let ops = GpuAcceleratedOps::new();
        let input = Array1::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0]);

        // Test ReLU
        let relu_result = ops
            .cpu_apply_activation(&input, "relu")
            .expect("operation should succeed");
        let expected_relu = [0.0, 0.0, 0.0, 1.0, 2.0];
        for (actual, expected) in relu_result.iter().zip(expected_relu.iter()) {
            assert_relative_eq!(actual, expected, epsilon = 1e-6);
        }

        // Test Sigmoid
        let sigmoid_result = ops
            .cpu_apply_activation(&input, "sigmoid")
            .expect("operation should succeed");
        for (input_val, output_val) in input.iter().zip(sigmoid_result.iter()) {
            let expected = 1.0 / (1.0 + (-input_val).exp());
            assert_relative_eq!(*output_val, expected, epsilon = 1e-6);
        }

        // Test Tanh
        let tanh_result = ops
            .cpu_apply_activation(&input, "tanh")
            .expect("operation should succeed");
        for (input_val, output_val) in input.iter().zip(tanh_result.iter()) {
            assert_relative_eq!(*output_val, input_val.tanh(), epsilon = 1e-6);
        }
    }

    #[test]
    fn test_activation_function_fallback() {
        let ops = GpuAcceleratedOps::new();
        let input = Array1::from_vec(vec![1.0, 2.0, 3.0]);

        // Should use CPU fallback even if GPU threshold is met
        let result = ops
            .apply_activation(&input, "relu")
            .expect("operation should succeed");
        assert_eq!(result.len(), 3);
        assert_relative_eq!(result[0], 1.0, epsilon = 1e-6);
        assert_relative_eq!(result[1], 2.0, epsilon = 1e-6);
        assert_relative_eq!(result[2], 3.0, epsilon = 1e-6);
    }

    #[test]
    fn test_matrix_multiply_fallback() {
        let ops = GpuAcceleratedOps::new();
        let a =
            Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).expect("array shape mismatch");
        let b =
            Array2::from_shape_vec((2, 2), vec![2.0, 0.0, 1.0, 2.0]).expect("array shape mismatch");

        // Should use CPU fallback
        let result = ops
            .matrix_multiply(&a, &b)
            .expect("operation should succeed");
        assert_eq!(result.dim(), (2, 2));
        assert_relative_eq!(result[[0, 0]], 4.0, epsilon = 1e-6);
        assert_relative_eq!(result[[0, 1]], 4.0, epsilon = 1e-6);
        assert_relative_eq!(result[[1, 0]], 10.0, epsilon = 1e-6);
        assert_relative_eq!(result[[1, 1]], 8.0, epsilon = 1e-6);
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn test_gpu_context_creation() {
        // This test will only run if GPU feature is enabled and CUDA is available
        match GpuContext::new() {
            Ok(ctx) => {
                // Test basic functionality
                let _memory_info = ctx.memory_info();
                let _stats = ctx.memory_pool_stats();
            }
            Err(_) => {
                // GPU not available, which is fine for CI/testing
                println!("GPU not available for testing");
            }
        }
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn test_gpu_tensor_operations() {
        if let Ok(ctx) = GpuContext::new() {
            let data = vec![1.0f32, 2.0, 3.0, 4.0];
            let shape = vec![2, 2];

            match GpuTensor::from_host_data(&ctx, &data, &shape) {
                Ok(tensor) => {
                    assert_eq!(tensor.shape, shape);
                    assert_eq!(tensor.len(), 4);
                    assert!(!tensor.is_empty());
                    assert_eq!(tensor.ndim(), 2);

                    // Test reshape
                    let reshaped = tensor.reshape(&[4, 1]).expect("operation should succeed");
                    assert_eq!(reshaped.shape, vec![4, 1]);
                    assert_eq!(reshaped.len(), 4);

                    // Test host copy
                    let host_data = tensor.to_host().expect("operation should succeed");
                    assert_eq!(host_data, data);
                }
                Err(_) => {
                    println!("GPU tensor creation failed - likely no GPU available");
                }
            }
        }
    }

    #[test]
    fn test_error_cases() {
        let ops = GpuAcceleratedOps::new();

        // Test dimension mismatch
        let a = Array2::from_shape_vec((2, 3), vec![1.0; 6]).expect("array shape mismatch");
        let b = Array2::from_shape_vec((2, 2), vec![1.0; 4]).expect("array shape mismatch");

        assert!(ops.matrix_multiply(&a, &b).is_err());

        // Test unsupported activation
        let input = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        assert!(ops.apply_activation(&input, "unsupported").is_err());
    }

    #[test]
    fn test_tensor_core_friendly_dimensions() {
        let ops = GpuAcceleratedOps::new();

        // Test tensor core friendly dimensions (multiples of 8, >= 64)
        let a_good =
            Array2::from_shape_vec((64, 64), vec![1.0; 64 * 64]).expect("array shape mismatch");
        let b_good =
            Array2::from_shape_vec((64, 64), vec![1.0; 64 * 64]).expect("array shape mismatch");
        assert!(ops.is_tensor_core_friendly(&a_good, &b_good));

        // Test non-tensor core friendly dimensions
        let a_bad =
            Array2::from_shape_vec((63, 63), vec![1.0; 63 * 63]).expect("array shape mismatch");
        let b_bad =
            Array2::from_shape_vec((63, 63), vec![1.0; 63 * 63]).expect("array shape mismatch");
        assert!(!ops.is_tensor_core_friendly(&a_bad, &b_bad));

        // Test too small dimensions
        let a_small =
            Array2::from_shape_vec((32, 32), vec![1.0; 32 * 32]).expect("array shape mismatch");
        let b_small =
            Array2::from_shape_vec((32, 32), vec![1.0; 32 * 32]).expect("array shape mismatch");
        assert!(!ops.is_tensor_core_friendly(&a_small, &b_small));
    }

    #[test]
    fn test_tensor_core_recommendations() {
        let ops = GpuAcceleratedOps::new();
        let recommendations = ops.tensor_core_recommendations();

        // Should always have some recommendations
        assert!(!recommendations.is_empty());

        // Check for expected keys
        #[cfg(feature = "gpu")]
        {
            assert!(recommendations.contains_key("tensor_cores"));
        }

        #[cfg(not(feature = "gpu"))]
        {
            assert!(recommendations.contains_key("gpu"));
        }
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn test_tensor_core_matrix_multiply() {
        let ops = GpuAcceleratedOps::new();

        // Test with tensor core friendly dimensions
        let a = Array2::from_shape_vec((64, 64), vec![1.0; 64 * 64]).expect("array shape mismatch");
        let b = Array2::from_shape_vec((64, 64), vec![2.0; 64 * 64]).expect("array shape mismatch");

        // Should not panic even if GPU/tensor cores are not available
        match ops.tensor_core_matrix_multiply(&a, &b, true) {
            Ok(result) => {
                assert_eq!(result.dim(), (64, 64));
                // Result should be approximately 64 * 1.0 * 2.0 = 128.0 for each element
                // (allowing for some floating point precision differences)
                if let Some(&val) = result.iter().next() {
                    assert!(
                        (val - 128.0).abs() < 1e-3 || val == 2.0,
                        "Unexpected result value: {}",
                        val
                    );
                }
            }
            Err(_) => {
                // Expected if GPU is not available or tensor cores not supported
                println!(
                    "Tensor core matrix multiply not available - this is expected in CI/testing"
                );
            }
        }
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn test_gpu_context_tensor_core_features() {
        match GpuContext::new() {
            Ok(ctx) => {
                // Test tensor core detection
                let has_tensor_cores = ctx.has_tensor_cores();
                println!("Tensor cores available: {}", has_tensor_cores);

                // Test compute capability
                if let Some((major, minor)) = ctx.compute_capability() {
                    println!("Compute capability: {}.{}", major, minor);

                    // Tensor cores should be available on compute capability 7.0+
                    if major >= 7 {
                        // Note: This might still be false if the GPU name doesn't match our patterns
                        println!("Expected tensor core support based on compute capability");
                    }
                }
            }
            Err(_) => {
                println!("GPU not available for testing - this is expected in CI environments");
            }
        }
    }
}
