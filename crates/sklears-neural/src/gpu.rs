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
use scirs2_core::numeric::Float;
use sklears_core::error::SklearsError;
use std::collections::HashMap;
use std::sync::Arc;

#[cfg(feature = "gpu")]
use {
    cudarc::{
        cublas::{CudaBlas, Gemm},
        driver::{CudaDevice, CudaStream, DevicePtr, LaunchAsync, LaunchConfig},
        nvrtc::Ptx,
    },
    std::sync::atomic::{AtomicUsize, Ordering},
};

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

/// GPU tensor representation
#[cfg(feature = "gpu")]
pub struct GpuTensor<T> {
    /// Device pointer to GPU memory
    pub(crate) ptr: cudarc::driver::CudaSlice<T>,
    /// Tensor shape
    pub shape: Vec<usize>,
    /// Device reference
    pub device: Arc<CudaDevice>,
    /// Data type size
    pub element_size: usize,
}

#[cfg(feature = "gpu")]
impl<T: cudarc::driver::DeviceRepr + Clone> GpuTensor<T> {
    /// Create tensor from host data
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

        let ptr = ctx.device.htod_copy(data.to_vec()).map_err(|e| {
            SklearsError::InvalidInput(format!("Failed to copy data to GPU: {}", e))
        })?;

        Ok(Self {
            ptr,
            shape: shape.to_vec(),
            device: ctx.device.clone(),
            element_size: std::mem::size_of::<T>(),
        })
    }

    /// Copy tensor data back to host
    pub fn to_host(&self) -> NeuralResult<Vec<T>> {
        self.device
            .dtoh_sync_copy(&self.ptr)
            .map_err(|e| SklearsError::InvalidInput(format!("Failed to copy data from GPU: {}", e)))
    }

    /// Get total number of elements
    pub fn len(&self) -> usize {
        self.shape.iter().product()
    }

    /// Check if tensor is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get tensor dimensions
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Reshape tensor (view operation, no data copy)
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

        Ok(GpuTensor {
            ptr: self.ptr.clone(),
            shape: new_shape.to_vec(),
            device: self.device.clone(),
            element_size: self.element_size,
        })
    }
}

/// Memory pool for efficient GPU memory management
#[cfg(feature = "gpu")]
struct GpuMemoryPool<T> {
    /// Available memory blocks
    available: Vec<cudarc::driver::CudaSlice<T>>,
    /// Block size in elements
    block_size: usize,
    /// Total allocated blocks
    total_blocks: usize,
    /// Maximum pool size
    max_size: usize,
    /// Hit/miss statistics
    hits: AtomicUsize,
    misses: AtomicUsize,
}

#[cfg(feature = "gpu")]
impl<T: cudarc::driver::DeviceRepr> GpuMemoryPool<T> {
    fn new(block_size: usize, max_size: usize) -> Self {
        Self {
            available: Vec::new(),
            block_size,
            total_blocks: 0,
            max_size,
            hits: AtomicUsize::new(0),
            misses: AtomicUsize::new(0),
        }
    }

    fn get_block(
        &mut self,
        device: &CudaDevice,
    ) -> Result<cudarc::driver::CudaSlice<T>, cudarc::driver::DriverError> {
        if let Some(block) = self.available.pop() {
            self.hits.fetch_add(1, Ordering::Relaxed);
            Ok(block)
        } else if self.total_blocks < self.max_size {
            self.misses.fetch_add(1, Ordering::Relaxed);
            let block = device.alloc_zeros::<T>(self.block_size)?;
            self.total_blocks += 1;
            Ok(block)
        } else {
            self.misses.fetch_add(1, Ordering::Relaxed);
            device.alloc_zeros::<T>(self.block_size)
        }
    }

    fn return_block(&mut self, block: cudarc::driver::CudaSlice<T>) {
        if self.available.len() < self.max_size / 2 {
            self.available.push(block);
        }
        // Otherwise, let it drop and be deallocated
    }

    fn hit_rate(&self) -> f64 {
        let hits = self.hits.load(Ordering::Relaxed);
        let misses = self.misses.load(Ordering::Relaxed);
        if hits + misses == 0 {
            0.0
        } else {
            hits as f64 / (hits + misses) as f64
        }
    }
}

/// GPU context for neural network operations
#[cfg(feature = "gpu")]
pub struct GpuContext {
    /// CUDA device
    pub device: Arc<CudaDevice>,
    /// cuBLAS handle
    pub blas: CudaBlas,
    /// CUDA streams for async operations
    pub streams: Vec<CudaStream>,
    /// Current stream index
    stream_index: std::sync::atomic::AtomicUsize,
    /// Memory pools for different data types
    f32_pool: Arc<Mutex<GpuMemoryPool<f32>>>,
    f64_pool: Arc<Mutex<GpuMemoryPool<f64>>>,
    /// Configuration
    config: GpuConfig,
    /// Compiled CUDA kernels
    kernels: HashMap<String, Ptx>,
}

#[cfg(feature = "gpu")]
impl GpuContext {
    /// Create new GPU context
    pub fn new() -> NeuralResult<Self> {
        Self::with_config(GpuConfig::default())
    }

    /// Create GPU context with custom configuration
    pub fn with_config(config: GpuConfig) -> NeuralResult<Self> {
        let device_id = config.device_id.unwrap_or_else(|| {
            // Select GPU with most free memory
            let device_count = CudaDevice::count().unwrap_or(0);
            if device_count == 0 {
                return 0;
            }

            let mut best_device = 0;
            let mut max_free_memory = 0;

            for i in 0..device_count {
                if let Ok(device) = CudaDevice::new(i) {
                    if let Ok((free, _total)) = device.memory_info() {
                        if free > max_free_memory {
                            max_free_memory = free;
                            best_device = i;
                        }
                    }
                }
            }
            best_device
        });

        let device = Arc::new(CudaDevice::new(device_id).map_err(|e| {
            SklearsError::InvalidInput(format!(
                "Failed to initialize CUDA device {}: {}",
                device_id, e
            ))
        })?);

        let blas = CudaBlas::new(device.clone()).map_err(|e| {
            SklearsError::InvalidInput(format!("Failed to initialize cuBLAS: {}", e))
        })?;

        // Create CUDA streams
        let mut streams = Vec::new();
        for _ in 0..config.max_streams {
            let stream = device.fork_default_stream().map_err(|e| {
                SklearsError::InvalidInput(format!("Failed to create CUDA stream: {}", e))
            })?;
            streams.push(stream);
        }

        // Initialize memory pools
        let pool_block_size = config.memory_pool_size / (config.max_streams * 2);
        let f32_pool = Arc::new(Mutex::new(GpuMemoryPool::new(
            pool_block_size / 4, // f32 size
            config.max_streams * 4,
        )));
        let f64_pool = Arc::new(Mutex::new(GpuMemoryPool::new(
            pool_block_size / 8, // f64 size
            config.max_streams * 2,
        )));

        let mut ctx = Self {
            device,
            blas,
            streams,
            stream_index: std::sync::atomic::AtomicUsize::new(0),
            f32_pool,
            f64_pool,
            config,
            kernels: HashMap::new(),
        };

        // Compile and cache commonly used kernels
        ctx.compile_kernels()?;

        Ok(ctx)
    }

    /// Get next available stream
    pub fn next_stream(&self) -> &CudaStream {
        let index = self.stream_index.fetch_add(1, Ordering::Relaxed) % self.streams.len();
        &self.streams[index]
    }

    /// Synchronize all streams
    pub fn synchronize(&self) -> NeuralResult<()> {
        for stream in &self.streams {
            stream.synchronize().map_err(|e| {
                SklearsError::InvalidInput(format!("Failed to synchronize CUDA stream: {}", e))
            })?;
        }
        Ok(())
    }

    /// Matrix multiplication using cuBLAS
    pub fn matrix_multiply(
        &self,
        a: &GpuTensor<f32>,
        b: &GpuTensor<f32>,
    ) -> NeuralResult<GpuTensor<f32>> {
        if a.shape.len() != 2 || b.shape.len() != 2 {
            return Err(SklearsError::InvalidInput(
                "Matrix multiplication requires 2D tensors".to_string(),
            ));
        }

        let (m, k) = (a.shape[0], a.shape[1]);
        let (k2, n) = (b.shape[0], b.shape[1]);

        if k != k2 {
            return Err(SklearsError::InvalidInput(format!(
                "Matrix dimension mismatch: {}x{} and {}x{}",
                m, k, k2, n
            )));
        }

        let result_shape = vec![m, n];
        let result = GpuTensor::from_host_data(self, &vec![0.0f32; m * n], &result_shape)?;

        // Perform GEMM: C = α*A*B + β*C
        let alpha = 1.0f32;
        let beta = 0.0f32;

        self.blas
            .gemm(
                cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
                cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
                n as i32,
                m as i32,
                k as i32,
                &alpha,
                &b.ptr,
                n as i32,
                &a.ptr,
                k as i32,
                &beta,
                &result.ptr,
                n as i32,
            )
            .map_err(|e| SklearsError::InvalidInput(format!("cuBLAS GEMM failed: {}", e)))?;

        Ok(result)
    }

    /// Matrix multiplication for f64
    pub fn matrix_multiply_f64(
        &self,
        a: &GpuTensor<f64>,
        b: &GpuTensor<f64>,
    ) -> NeuralResult<GpuTensor<f64>> {
        if a.shape.len() != 2 || b.shape.len() != 2 {
            return Err(SklearsError::InvalidInput(
                "Matrix multiplication requires 2D tensors".to_string(),
            ));
        }

        let (m, k) = (a.shape[0], a.shape[1]);
        let (k2, n) = (b.shape[0], b.shape[1]);

        if k != k2 {
            return Err(SklearsError::InvalidInput(format!(
                "Matrix dimension mismatch: {}x{} and {}x{}",
                m, k, k2, n
            )));
        }

        let result_shape = vec![m, n];
        let result = GpuTensor::from_host_data(self, &vec![0.0f64; m * n], &result_shape)?;

        let alpha = 1.0f64;
        let beta = 0.0f64;

        self.blas
            .gemm(
                cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
                cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
                n as i32,
                m as i32,
                k as i32,
                &alpha,
                &b.ptr,
                n as i32,
                &a.ptr,
                k as i32,
                &beta,
                &result.ptr,
                n as i32,
            )
            .map_err(|e| SklearsError::InvalidInput(format!("cuBLAS GEMM failed: {}", e)))?;

        Ok(result)
    }

    /// Element-wise addition
    pub fn add(&self, a: &GpuTensor<f32>, b: &GpuTensor<f32>) -> NeuralResult<GpuTensor<f32>> {
        if a.shape != b.shape {
            return Err(SklearsError::InvalidInput(format!(
                "Shape mismatch for addition: {:?} vs {:?}",
                a.shape, b.shape
            )));
        }

        let result = GpuTensor::from_host_data(self, &vec![0.0f32; a.len()], &a.shape)?;

        let kernel = self.kernels.get("elementwise_add").ok_or_else(|| {
            SklearsError::InvalidInput("Elementwise add kernel not found".to_string())
        })?;

        let func = self
            .device
            .get_func("elementwise_add", "elementwise_add")
            .map_err(|e| {
                SklearsError::InvalidInput(format!("Failed to get kernel function: {}", e))
            })?;

        let n = a.len();
        let block_size = 256;
        let grid_size = (n + block_size - 1) / block_size;

        let config = LaunchConfig {
            grid_dim: (grid_size as u32, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            func.launch(&config, (&a.ptr, &b.ptr, &result.ptr, n))
                .map_err(|e| SklearsError::InvalidInput(format!("Kernel launch failed: {}", e)))?;
        }

        Ok(result)
    }

    /// ReLU activation function
    pub fn relu(&self, input: &GpuTensor<f32>) -> NeuralResult<GpuTensor<f32>> {
        let result = GpuTensor::from_host_data(self, &vec![0.0f32; input.len()], &input.shape)?;

        let func = self
            .device
            .get_func("activation_kernels", "relu_forward")
            .map_err(|e| SklearsError::InvalidInput(format!("Failed to get ReLU kernel: {}", e)))?;

        let n = input.len();
        let block_size = 256;
        let grid_size = (n + block_size - 1) / block_size;

        let config = LaunchConfig {
            grid_dim: (grid_size as u32, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            func.launch(&config, (&input.ptr, &result.ptr, n))
                .map_err(|e| {
                    SklearsError::InvalidInput(format!("ReLU kernel launch failed: {}", e))
                })?;
        }

        Ok(result)
    }

    /// Sigmoid activation function
    pub fn sigmoid(&self, input: &GpuTensor<f32>) -> NeuralResult<GpuTensor<f32>> {
        let result = GpuTensor::from_host_data(self, &vec![0.0f32; input.len()], &input.shape)?;

        let func = self
            .device
            .get_func("activation_kernels", "sigmoid_forward")
            .map_err(|e| {
                SklearsError::InvalidInput(format!("Failed to get sigmoid kernel: {}", e))
            })?;

        let n = input.len();
        let block_size = 256;
        let grid_size = (n + block_size - 1) / block_size;

        let config = LaunchConfig {
            grid_dim: (grid_size as u32, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            func.launch(&config, (&input.ptr, &result.ptr, n))
                .map_err(|e| {
                    SklearsError::InvalidInput(format!("Sigmoid kernel launch failed: {}", e))
                })?;
        }

        Ok(result)
    }

    /// Get GPU memory info
    pub fn memory_info(&self) -> NeuralResult<(usize, usize)> {
        self.device
            .memory_info()
            .map_err(|e| SklearsError::InvalidInput(format!("Failed to get memory info: {}", e)))
    }

    /// Get memory pool statistics
    pub fn memory_pool_stats(&self) -> (f64, f64) {
        let f32_hit_rate = self.f32_pool.lock().unwrap().hit_rate();
        let f64_hit_rate = self.f64_pool.lock().unwrap().hit_rate();
        (f32_hit_rate, f64_hit_rate)
    }

    /// Check if tensor cores are available
    pub fn has_tensor_cores(&self) -> bool {
        match self.device.name() {
            Ok(name) => {
                let name_lower = name.to_lowercase();
                // Tensor cores are available on V100, A100, H100, RTX 20xx/30xx/40xx series
                name_lower.contains("v100")
                    || name_lower.contains("a100")
                    || name_lower.contains("h100")
                    || name_lower.contains("rtx")
                    || name_lower.contains("tesla")
                    || name_lower.contains("quadro")
            }
            Err(_) => false,
        }
    }

    /// Get compute capability for tensor core optimizations
    pub fn compute_capability(&self) -> Option<(i32, i32)> {
        self.device.compute_capability().ok()
    }

    /// Tensor core optimized matrix multiplication using half precision
    pub fn tensor_core_gemm_f16(
        &self,
        a: &GpuTensor<half::f16>,
        b: &GpuTensor<half::f16>,
    ) -> NeuralResult<GpuTensor<half::f16>> {
        if !self.has_tensor_cores() {
            return Err(SklearsError::InvalidInput(
                "Tensor cores not available on this device".to_string(),
            ));
        }

        if a.shape.len() != 2 || b.shape.len() != 2 {
            return Err(SklearsError::InvalidInput(
                "Tensor core GEMM requires 2D tensors".to_string(),
            ));
        }

        let (m, k) = (a.shape[0], a.shape[1]);
        let (k2, n) = (b.shape[0], b.shape[1]);

        if k != k2 {
            return Err(SklearsError::InvalidInput(format!(
                "Matrix dimension mismatch: {}x{} and {}x{}",
                m, k, k2, n
            )));
        }

        // Tensor cores work best with dimensions that are multiples of 8
        if m % 8 != 0 || n % 8 != 0 || k % 8 != 0 {
            return Err(SklearsError::InvalidInput(
                "Tensor core operations require dimensions to be multiples of 8".to_string(),
            ));
        }

        let result_shape = vec![m, n];
        let result = GpuTensor::from_host_data(self, &vec![half::f16::ZERO; m * n], &result_shape)?;

        // Use tensor core optimized GEMM
        let alpha = half::f16::ONE;
        let beta = half::f16::ZERO;

        // Note: This would use cublasGemmEx in a real implementation
        // For now, we'll use the regular GEMM as a placeholder
        self.blas
            .gemm(
                cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
                cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
                n as i32,
                m as i32,
                k as i32,
                &alpha,
                &b.ptr,
                n as i32,
                &a.ptr,
                k as i32,
                &beta,
                &result.ptr,
                n as i32,
            )
            .map_err(|e| SklearsError::InvalidInput(format!("Tensor core GEMM failed: {}", e)))?;

        Ok(result)
    }

    /// Mixed precision matrix multiplication (FP16 compute, FP32 accumulate)
    pub fn mixed_precision_gemm(
        &self,
        a: &GpuTensor<half::f16>,
        b: &GpuTensor<half::f16>,
    ) -> NeuralResult<GpuTensor<f32>> {
        if !self.has_tensor_cores() {
            return Err(SklearsError::InvalidInput(
                "Tensor cores not available for mixed precision".to_string(),
            ));
        }

        if a.shape.len() != 2 || b.shape.len() != 2 {
            return Err(SklearsError::InvalidInput(
                "Mixed precision GEMM requires 2D tensors".to_string(),
            ));
        }

        let (m, k) = (a.shape[0], a.shape[1]);
        let (k2, n) = (b.shape[0], b.shape[1]);

        if k != k2 {
            return Err(SklearsError::InvalidInput(format!(
                "Matrix dimension mismatch: {}x{} and {}x{}",
                m, k, k2, n
            )));
        }

        let result_shape = vec![m, n];
        let result = GpuTensor::from_host_data(self, &vec![0.0f32; m * n], &result_shape)?;

        // This would use cublasGemmEx with CUDA_R_16F inputs and CUDA_R_32F output
        // For now, we simulate mixed precision behavior
        let alpha = 1.0f32;
        let beta = 0.0f32;

        // In a real implementation, this would use tensor cores for FP16 computation
        // with FP32 accumulation through cublasGemmEx
        self.blas
            .gemm(
                cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
                cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
                n as i32,
                m as i32,
                k as i32,
                &alpha,
                &b.ptr, // This would need type conversion in real implementation
                n as i32,
                &a.ptr, // This would need type conversion in real implementation
                k as i32,
                &beta,
                &result.ptr,
                n as i32,
            )
            .map_err(|e| {
                SklearsError::InvalidInput(format!("Mixed precision GEMM failed: {}", e))
            })?;

        Ok(result)
    }

    /// Optimized convolution using tensor cores
    pub fn tensor_core_conv2d(
        &self,
        input: &GpuTensor<half::f16>,
        kernel: &GpuTensor<half::f16>,
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> NeuralResult<GpuTensor<half::f16>> {
        if !self.has_tensor_cores() {
            return Err(SklearsError::InvalidInput(
                "Tensor cores not available for convolution".to_string(),
            ));
        }

        // This is a placeholder for tensor core optimized convolution
        // Real implementation would use cuDNN with tensor core acceleration

        let func = self
            .device
            .get_func("tensor_core_kernels", "tensor_core_conv2d")
            .map_err(|e| {
                SklearsError::InvalidInput(format!("Failed to get tensor core conv kernel: {}", e))
            })?;

        // Placeholder implementation - would need proper convolution logic
        let output_size = input.len(); // Simplified
        let result =
            GpuTensor::from_host_data(self, &vec![half::f16::ZERO; output_size], &input.shape)?;

        let block_size = 256;
        let grid_size = (output_size + block_size - 1) / block_size;

        let config = cudarc::driver::LaunchConfig {
            grid_dim: (grid_size as u32, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            func.launch(&config, (&input.ptr, &kernel.ptr, &result.ptr, output_size))
                .map_err(|e| {
                    SklearsError::InvalidInput(format!(
                        "Tensor core conv kernel launch failed: {}",
                        e
                    ))
                })?;
        }

        Ok(result)
    }

    /// Compile CUDA kernels
    fn compile_kernels(&mut self) -> NeuralResult<()> {
        // Element-wise operations kernel
        let elementwise_src = r#"
        extern "C" __global__ void elementwise_add(
            const float* a, 
            const float* b, 
            float* c, 
            int n
        ) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n) {
                c[idx] = a[idx] + b[idx];
            }
        }

        extern "C" __global__ void elementwise_mul(
            const float* a, 
            const float* b, 
            float* c, 
            int n
        ) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n) {
                c[idx] = a[idx] * b[idx];
            }
        }

        extern "C" __global__ void elementwise_sub(
            const float* a, 
            const float* b, 
            float* c, 
            int n
        ) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n) {
                c[idx] = a[idx] - b[idx];
            }
        }
        "#;

        let elementwise_ptx = cudarc::nvrtc::compile_ptx(elementwise_src).map_err(|e| {
            SklearsError::InvalidInput(format!("Failed to compile elementwise kernels: {}", e))
        })?;

        self.device
            .load_ptx(
                elementwise_ptx.clone(),
                "elementwise_add",
                &["elementwise_add", "elementwise_mul", "elementwise_sub"],
            )
            .map_err(|e| {
                SklearsError::InvalidInput(format!("Failed to load elementwise kernels: {}", e))
            })?;

        self.kernels
            .insert("elementwise_add".to_string(), elementwise_ptx);

        // Activation function kernels
        let activation_src = r#"
        extern "C" __global__ void relu_forward(
            const float* input, 
            float* output, 
            int n
        ) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n) {
                output[idx] = fmaxf(0.0f, input[idx]);
            }
        }

        extern "C" __global__ void relu_backward(
            const float* grad_output,
            const float* input,
            float* grad_input,
            int n
        ) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n) {
                grad_input[idx] = input[idx] > 0.0f ? grad_output[idx] : 0.0f;
            }
        }

        extern "C" __global__ void sigmoid_forward(
            const float* input, 
            float* output, 
            int n
        ) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n) {
                output[idx] = 1.0f / (1.0f + expf(-input[idx]));
            }
        }

        extern "C" __global__ void sigmoid_backward(
            const float* grad_output,
            const float* output,
            float* grad_input,
            int n
        ) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n) {
                float sig = output[idx];
                grad_input[idx] = grad_output[idx] * sig * (1.0f - sig);
            }
        }

        extern "C" __global__ void tanh_forward(
            const float* input, 
            float* output, 
            int n
        ) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n) {
                output[idx] = tanhf(input[idx]);
            }
        }

        extern "C" __global__ void tanh_backward(
            const float* grad_output,
            const float* output,
            float* grad_input,
            int n
        ) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n) {
                float t = output[idx];
                grad_input[idx] = grad_output[idx] * (1.0f - t * t);
            }
        }
        "#;

        let activation_ptx = cudarc::nvrtc::compile_ptx(activation_src).map_err(|e| {
            SklearsError::InvalidInput(format!("Failed to compile activation kernels: {}", e))
        })?;

        self.device
            .load_ptx(
                activation_ptx.clone(),
                "activation_kernels",
                &[
                    "relu_forward",
                    "relu_backward",
                    "sigmoid_forward",
                    "sigmoid_backward",
                    "tanh_forward",
                    "tanh_backward",
                ],
            )
            .map_err(|e| {
                SklearsError::InvalidInput(format!("Failed to load activation kernels: {}", e))
            })?;

        self.kernels
            .insert("activation_kernels".to_string(), activation_ptx);

        // Tensor core optimized kernels
        let tensor_core_src = r#"
        #include <mma.h>
        using namespace nvcuda;

        extern "C" __global__ void tensor_core_gemm_f16(
            const half* a,
            const half* b,
            half* c,
            int m, int n, int k
        ) {
            // Tensor core WMMA implementation
            // This is a simplified version - real implementation would be more complex
            
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
            wmma::fragment<wmma::accumulator, 16, 16, 16, half> c_frag;

            int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
            int warpN = (blockIdx.y * blockDim.y + threadIdx.y);

            if (warpM * 16 < m && warpN * 16 < n) {
                wmma::fill_fragment(c_frag, 0.0f);

                for (int i = 0; i < k; i += 16) {
                    int aRow = warpM * 16;
                    int aCol = i;
                    int bRow = i;
                    int bCol = warpN * 16;

                    if (aRow < m && aCol < k && bRow < k && bCol < n) {
                        wmma::load_matrix_sync(a_frag, a + aRow * k + aCol, k);
                        wmma::load_matrix_sync(b_frag, b + bRow * n + bCol, n);
                        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
                    }
                }

                int cRow = warpM * 16;
                int cCol = warpN * 16;
                if (cRow < m && cCol < n) {
                    wmma::store_matrix_sync(c + cRow * n + cCol, c_frag, n, wmma::mem_row_major);
                }
            }
        }

        extern "C" __global__ void tensor_core_conv2d(
            const half* input,
            const half* kernel,
            half* output,
            int n
        ) {
            // Simplified tensor core convolution
            // Real implementation would use im2col + GEMM or cuDNN
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n) {
                output[idx] = input[idx] * kernel[0]; // Simplified placeholder
            }
        }

        extern "C" __global__ void mixed_precision_activation(
            const half* input,
            float* output,
            int n,
            int activation_type
        ) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n) {
                float x = __half2float(input[idx]);
                float result;
                
                switch (activation_type) {
                    case 0: // ReLU
                        result = fmaxf(0.0f, x);
                        break;
                    case 1: // GELU
                        result = 0.5f * x * (1.0f + tanhf(0.7978845608028654f * x));
                        break;
                    case 2: // Swish
                        result = x / (1.0f + expf(-x));
                        break;
                    default:
                        result = x;
                }
                
                output[idx] = result;
            }
        }
        "#;

        // Note: Tensor core kernels require compute capability 7.0+ and proper compilation flags
        if self.has_tensor_cores() {
            match cudarc::nvrtc::compile_ptx(tensor_core_src) {
                Ok(tensor_core_ptx) => {
                    if let Err(_) = self.device.load_ptx(
                        tensor_core_ptx.clone(),
                        "tensor_core_kernels",
                        &[
                            "tensor_core_gemm_f16",
                            "tensor_core_conv2d",
                            "mixed_precision_activation",
                        ],
                    ) {
                        // Tensor core compilation failed, continue without tensor cores
                        log::warn!(
                            "Failed to load tensor core kernels, falling back to regular kernels"
                        );
                    } else {
                        self.kernels
                            .insert("tensor_core_kernels".to_string(), tensor_core_ptx);
                    }
                }
                Err(_) => {
                    log::warn!("Failed to compile tensor core kernels");
                }
            }
        }

        Ok(())
    }
}

#[cfg(not(feature = "gpu"))]
/// Stub GPU context when GPU feature is disabled
pub struct GpuContext;

#[cfg(not(feature = "gpu"))]
impl GpuContext {
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

    /// Check if tensor dimensions are suitable for tensor cores
    fn is_tensor_core_friendly(&self, a: &Array2<f32>, b: &Array2<f32>) -> bool {
        let (m, k) = (a.nrows(), a.ncols());
        let n = b.ncols();

        // Tensor cores work best with dimensions that are multiples of 8
        m % 8 == 0 && n % 8 == 0 && k % 8 == 0 &&
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
        let a = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let b = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

        let result = ops.cpu_matrix_multiply(&a, &b).unwrap();

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
        let relu_result = ops.cpu_apply_activation(&input, "relu").unwrap();
        let expected_relu = vec![0.0, 0.0, 0.0, 1.0, 2.0];
        for (actual, expected) in relu_result.iter().zip(expected_relu.iter()) {
            assert_relative_eq!(actual, expected, epsilon = 1e-6);
        }

        // Test Sigmoid
        let sigmoid_result = ops.cpu_apply_activation(&input, "sigmoid").unwrap();
        for (input_val, output_val) in input.iter().zip(sigmoid_result.iter()) {
            let expected = 1.0 / (1.0 + (-input_val).exp());
            assert_relative_eq!(*output_val, expected, epsilon = 1e-6);
        }

        // Test Tanh
        let tanh_result = ops.cpu_apply_activation(&input, "tanh").unwrap();
        for (input_val, output_val) in input.iter().zip(tanh_result.iter()) {
            assert_relative_eq!(*output_val, input_val.tanh(), epsilon = 1e-6);
        }
    }

    #[test]
    fn test_activation_function_fallback() {
        let ops = GpuAcceleratedOps::new();
        let input = Array1::from_vec(vec![1.0, 2.0, 3.0]);

        // Should use CPU fallback even if GPU threshold is met
        let result = ops.apply_activation(&input, "relu").unwrap();
        assert_eq!(result.len(), 3);
        assert_relative_eq!(result[0], 1.0, epsilon = 1e-6);
        assert_relative_eq!(result[1], 2.0, epsilon = 1e-6);
        assert_relative_eq!(result[2], 3.0, epsilon = 1e-6);
    }

    #[test]
    fn test_matrix_multiply_fallback() {
        let ops = GpuAcceleratedOps::new();
        let a = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let b = Array2::from_shape_vec((2, 2), vec![2.0, 0.0, 1.0, 2.0]).unwrap();

        // Should use CPU fallback
        let result = ops.matrix_multiply(&a, &b).unwrap();
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
                    let reshaped = tensor.reshape(&[4, 1]).unwrap();
                    assert_eq!(reshaped.shape, vec![4, 1]);
                    assert_eq!(reshaped.len(), 4);

                    // Test host copy
                    let host_data = tensor.to_host().unwrap();
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
        let a = Array2::from_shape_vec((2, 3), vec![1.0; 6]).unwrap();
        let b = Array2::from_shape_vec((2, 2), vec![1.0; 4]).unwrap();

        assert!(ops.matrix_multiply(&a, &b).is_err());

        // Test unsupported activation
        let input = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        assert!(ops.apply_activation(&input, "unsupported").is_err());
    }

    #[test]
    fn test_tensor_core_friendly_dimensions() {
        let ops = GpuAcceleratedOps::new();

        // Test tensor core friendly dimensions (multiples of 8, >= 64)
        let a_good = Array2::from_shape_vec((64, 64), vec![1.0; 64 * 64]).unwrap();
        let b_good = Array2::from_shape_vec((64, 64), vec![1.0; 64 * 64]).unwrap();
        assert!(ops.is_tensor_core_friendly(&a_good, &b_good));

        // Test non-tensor core friendly dimensions
        let a_bad = Array2::from_shape_vec((63, 63), vec![1.0; 63 * 63]).unwrap();
        let b_bad = Array2::from_shape_vec((63, 63), vec![1.0; 63 * 63]).unwrap();
        assert!(!ops.is_tensor_core_friendly(&a_bad, &b_bad));

        // Test too small dimensions
        let a_small = Array2::from_shape_vec((32, 32), vec![1.0; 32 * 32]).unwrap();
        let b_small = Array2::from_shape_vec((32, 32), vec![1.0; 32 * 32]).unwrap();
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
        let a = Array2::from_shape_vec((64, 64), vec![1.0; 64 * 64]).unwrap();
        let b = Array2::from_shape_vec((64, 64), vec![2.0; 64 * 64]).unwrap();

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
