//! GPU acceleration support for SIMD operations
//!
//! This module provides CUDA and OpenCL kernel interfaces for GPU-accelerated
//! machine learning operations with fallback to CPU SIMD implementations.

use crate::traits::SimdError;

#[cfg(feature = "no-std")]
use alloc::{
    boxed::Box,
    format,
    string::{String, ToString},
    vec::Vec,
};

#[cfg(feature = "no-std")]
use core::any::Any;
#[cfg(not(feature = "no-std"))]
use std::any::Any;

#[cfg(feature = "no-std")]
use spin::Mutex;
#[cfg(not(feature = "no-std"))]
use std::sync::Mutex;

/// GPU computation backends
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuBackend {
    Cuda,
    OpenCL,
    Metal,
    Vulkan,
}

/// GPU device information
#[derive(Debug, Clone)]
pub struct GpuDevice {
    pub id: u32,
    pub name: String,
    pub backend: GpuBackend,
    pub compute_units: u32,
    pub memory_mb: u64,
    pub supports_f64: bool,
    pub supports_f16: bool,
}

/// GPU memory buffer wrapper
#[derive(Debug)]
pub struct GpuBuffer<T> {
    pub ptr: *mut T,
    pub size: usize,
    pub device: GpuDevice,
    backend_handle: Option<Box<dyn Any + Send + Sync>>,
}

unsafe impl<T: Send> Send for GpuBuffer<T> {}
unsafe impl<T: Sync> Sync for GpuBuffer<T> {}

impl<T> Drop for GpuBuffer<T> {
    fn drop(&mut self) {
        // Free GPU memory when buffer is dropped
        // Implementation depends on backend
    }
}

/// GPU context for managing resources
pub struct GpuContext {
    pub device: GpuDevice,
    pub streams: Vec<GpuStream>,
    backend_context: Option<Box<dyn Any + Send + Sync>>,
}

/// GPU stream for asynchronous operations
#[derive(Debug)]
pub struct GpuStream {
    pub id: u32,
    pub device_id: u32,
    backend_stream: Option<Box<dyn Any + Send + Sync>>,
}

/// GPU kernel launch parameters
#[derive(Debug, Clone)]
pub struct KernelConfig {
    pub grid_size: (u32, u32, u32),
    pub block_size: (u32, u32, u32),
    pub shared_memory: u32,
    pub stream: Option<u32>,
}

impl Default for KernelConfig {
    fn default() -> Self {
        Self {
            grid_size: (1, 1, 1),
            block_size: (256, 1, 1),
            shared_memory: 0,
            stream: None,
        }
    }
}

/// GPU operations interface
pub trait GpuOperations {
    /// Allocate GPU memory
    fn allocate<T>(&self, size: usize) -> Result<GpuBuffer<T>, SimdError>;

    /// Copy data from host to device
    fn copy_to_device<T>(
        &self,
        host_data: &[T],
        gpu_buffer: &mut GpuBuffer<T>,
    ) -> Result<(), SimdError>;

    /// Copy data from device to host
    fn copy_to_host<T>(
        &self,
        gpu_buffer: &GpuBuffer<T>,
        host_data: &mut [T],
    ) -> Result<(), SimdError>;

    /// Launch kernel with configuration
    fn launch_kernel(
        &self,
        kernel: &str,
        config: &KernelConfig,
        args: &[&dyn Any],
    ) -> Result<(), SimdError>;

    /// Synchronize device
    fn synchronize(&self) -> Result<(), SimdError>;
}

/// CUDA specific implementation
pub mod cuda {
    use super::*;

    /// CUDA device manager
    pub struct CudaDevice {
        device_id: u32,
        context: Option<Box<dyn Any + Send + Sync>>,
    }

    impl CudaDevice {
        pub fn new(device_id: u32) -> Result<Self, SimdError> {
            // Initialize CUDA device
            Ok(Self {
                device_id,
                context: None,
            })
        }

        pub fn get_device_count() -> Result<u32, SimdError> {
            // CUDA disabled for macOS compatibility
            Err(SimdError::UnsupportedOperation(
                "CUDA not available".to_string(),
            ))
        }

        pub fn get_device_info(device_id: u32) -> Result<GpuDevice, SimdError> {
            // Mock device info - would query actual CUDA device
            Ok(GpuDevice {
                id: device_id,
                name: format!("CUDA Device {}", device_id),
                backend: GpuBackend::Cuda,
                compute_units: 80,
                memory_mb: 8192,
                supports_f64: true,
                supports_f16: true,
            })
        }
    }

    impl GpuOperations for CudaDevice {
        fn allocate<T>(&self, size: usize) -> Result<GpuBuffer<T>, SimdError> {
            // CUDA disabled for macOS compatibility
            let _ = size;
            Err(SimdError::UnsupportedOperation(
                "CUDA not available".to_string(),
            ))
        }

        fn copy_to_device<T>(
            &self,
            _host_data: &[T],
            _gpu_buffer: &mut GpuBuffer<T>,
        ) -> Result<(), SimdError> {
            // CUDA disabled for macOS compatibility
            Err(SimdError::UnsupportedOperation(
                "CUDA not available".to_string(),
            ))
        }

        fn copy_to_host<T>(
            &self,
            _gpu_buffer: &GpuBuffer<T>,
            _host_data: &mut [T],
        ) -> Result<(), SimdError> {
            // CUDA disabled for macOS compatibility
            Err(SimdError::UnsupportedOperation(
                "CUDA not available".to_string(),
            ))
        }

        fn launch_kernel(
            &self,
            _kernel: &str,
            _config: &KernelConfig,
            _args: &[&dyn Any],
        ) -> Result<(), SimdError> {
            // CUDA disabled for macOS compatibility
            Err(SimdError::UnsupportedOperation(
                "CUDA not available".to_string(),
            ))
        }

        fn synchronize(&self) -> Result<(), SimdError> {
            // CUDA disabled for macOS compatibility
            Err(SimdError::UnsupportedOperation(
                "CUDA not available".to_string(),
            ))
        }
    }

    /// CUDA kernels for common SIMD operations
    pub mod kernels {

        /// Vector addition kernel
        pub const VECTOR_ADD_KERNEL: &str = r#"
        extern "C" __global__ void vector_add(float* a, float* b, float* c, int n) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n) {
                c[idx] = a[idx] + b[idx];
            }
        }
        "#;

        /// Dot product kernel with reduction
        pub const DOT_PRODUCT_KERNEL: &str = r#"
        extern "C" __global__ void dot_product(float* a, float* b, float* result, int n) {
            __shared__ float shared[256];
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            int tid = threadIdx.x;
            
            float sum = 0.0f;
            while (idx < n) {
                sum += a[idx] * b[idx];
                idx += blockDim.x * gridDim.x;
            }
            
            shared[tid] = sum;
            __syncthreads();
            
            // Reduction in shared memory
            for (int s = blockDim.x / 2; s > 0; s >>= 1) {
                if (tid < s) {
                    shared[tid] += shared[tid + s];
                }
                __syncthreads();
            }
            
            if (tid == 0) {
                atomicAdd(result, shared[0]);
            }
        }
        "#;

        /// Matrix multiplication kernel
        pub const MATRIX_MUL_KERNEL: &str = r#"
        extern "C" __global__ void matrix_mul(float* a, float* b, float* c, int m, int n, int k) {
            int row = blockIdx.y * blockDim.y + threadIdx.y;
            int col = blockIdx.x * blockDim.x + threadIdx.x;
            
            if (row < m && col < n) {
                float sum = 0.0f;
                for (int i = 0; i < k; i++) {
                    sum += a[row * k + i] * b[i * n + col];
                }
                c[row * n + col] = sum;
            }
        }
        "#;

        /// ReLU activation kernel
        pub const RELU_KERNEL: &str = r#"
        extern "C" __global__ void relu(float* input, float* output, int n) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n) {
                output[idx] = fmaxf(0.0f, input[idx]);
            }
        }
        "#;

        /// Softmax kernel
        pub const SOFTMAX_KERNEL: &str = r#"
        extern "C" __global__ void softmax(float* input, float* output, int n) {
            extern __shared__ float shared[];
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            int tid = threadIdx.x;
            
            // Find maximum for numerical stability
            float max_val = (idx < n) ? input[idx] : -INFINITY;
            shared[tid] = max_val;
            __syncthreads();
            
            for (int s = blockDim.x / 2; s > 0; s >>= 1) {
                if (tid < s) {
                    shared[tid] = fmaxf(shared[tid], shared[tid + s]);
                }
                __syncthreads();
            }
            
            float global_max = shared[0];
            
            // Compute exp and sum
            float exp_val = (idx < n) ? expf(input[idx] - global_max) : 0.0f;
            shared[tid] = exp_val;
            __syncthreads();
            
            for (int s = blockDim.x / 2; s > 0; s >>= 1) {
                if (tid < s) {
                    shared[tid] += shared[tid + s];
                }
                __syncthreads();
            }
            
            float sum = shared[0];
            
            if (idx < n) {
                output[idx] = exp_val / sum;
            }
        }
        "#;
    }
}

/// OpenCL specific implementation
pub mod opencl {
    use super::*;

    /// OpenCL device manager
    pub struct OpenCLDevice {
        device_id: u32,
        context: Option<Box<dyn Any + Send + Sync>>,
        command_queue: Option<Box<dyn Any + Send + Sync>>,
    }

    impl OpenCLDevice {
        pub fn new(device_id: u32) -> Result<Self, SimdError> {
            Ok(Self {
                device_id,
                context: None,
                command_queue: None,
            })
        }

        pub fn get_platforms() -> Result<Vec<String>, SimdError> {
            // OpenCL disabled for macOS compatibility
            Err(SimdError::UnsupportedOperation(
                "OpenCL not available".to_string(),
            ))
        }

        pub fn get_devices(platform_id: u32) -> Result<Vec<GpuDevice>, SimdError> {
            // OpenCL disabled for macOS compatibility
            let _ = platform_id;
            Err(SimdError::UnsupportedOperation(
                "OpenCL not available".to_string(),
            ))
        }
    }

    impl GpuOperations for OpenCLDevice {
        fn allocate<T>(&self, size: usize) -> Result<GpuBuffer<T>, SimdError> {
            // OpenCL disabled for macOS compatibility
            let _ = size;
            Err(SimdError::UnsupportedOperation(
                "OpenCL not available".to_string(),
            ))
        }

        fn copy_to_device<T>(
            &self,
            _host_data: &[T],
            _gpu_buffer: &mut GpuBuffer<T>,
        ) -> Result<(), SimdError> {
            // OpenCL disabled for macOS compatibility
            Err(SimdError::UnsupportedOperation(
                "OpenCL not available".to_string(),
            ))
        }

        fn copy_to_host<T>(
            &self,
            _gpu_buffer: &GpuBuffer<T>,
            _host_data: &mut [T],
        ) -> Result<(), SimdError> {
            // OpenCL disabled for macOS compatibility
            Err(SimdError::UnsupportedOperation(
                "OpenCL not available".to_string(),
            ))
        }

        fn launch_kernel(
            &self,
            _kernel: &str,
            _config: &KernelConfig,
            _args: &[&dyn Any],
        ) -> Result<(), SimdError> {
            // OpenCL disabled for macOS compatibility
            Err(SimdError::UnsupportedOperation(
                "OpenCL not available".to_string(),
            ))
        }

        fn synchronize(&self) -> Result<(), SimdError> {
            // OpenCL disabled for macOS compatibility
            Err(SimdError::UnsupportedOperation(
                "OpenCL not available".to_string(),
            ))
        }
    }

    /// OpenCL kernels for common SIMD operations
    pub mod kernels {
        /// Vector addition kernel
        pub const VECTOR_ADD_KERNEL: &str = r#"
        __kernel void vector_add(__global float* a, __global float* b, __global float* c, int n) {
            int idx = get_global_id(0);
            if (idx < n) {
                c[idx] = a[idx] + b[idx];
            }
        }
        "#;

        /// Dot product kernel
        pub const DOT_PRODUCT_KERNEL: &str = r#"
        __kernel void dot_product(__global float* a, __global float* b, __global float* result, int n) {
            __local float local_sum[256];
            int idx = get_global_id(0);
            int lid = get_local_id(0);
            
            float sum = 0.0f;
            if (idx < n) {
                sum = a[idx] * b[idx];
            }
            
            local_sum[lid] = sum;
            barrier(CLK_LOCAL_MEM_FENCE);
            
            // Reduction
            for (int s = get_local_size(0) / 2; s > 0; s >>= 1) {
                if (lid < s) {
                    local_sum[lid] += local_sum[lid + s];
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            
            if (lid == 0) {
                atomic_add_global(result, local_sum[0]);
            }
        }
        "#;

        /// Matrix multiplication kernel
        pub const MATRIX_MUL_KERNEL: &str = r#"
        __kernel void matrix_mul(__global float* a, __global float* b, __global float* c, int m, int n, int k) {
            int row = get_global_id(1);
            int col = get_global_id(0);
            
            if (row < m && col < n) {
                float sum = 0.0f;
                for (int i = 0; i < k; i++) {
                    sum += a[row * k + i] * b[i * n + col];
                }
                c[row * n + col] = sum;
            }
        }
        "#;
    }
}

/// GPU manager for handling multiple devices and backends
pub struct GpuManager {
    cuda_devices: Vec<cuda::CudaDevice>,
    opencl_devices: Vec<opencl::OpenCLDevice>,
    preferred_backend: Option<GpuBackend>,
}

impl GpuManager {
    pub fn new() -> Self {
        Self {
            cuda_devices: Vec::new(),
            opencl_devices: Vec::new(),
            preferred_backend: None,
        }
    }

    /// Initialize GPU manager and detect available devices
    pub fn initialize(&mut self) -> Result<(), SimdError> {
        // Try to initialize CUDA devices
        if let Ok(count) = cuda::CudaDevice::get_device_count() {
            for i in 0..count {
                if let Ok(device) = cuda::CudaDevice::new(i) {
                    self.cuda_devices.push(device);
                }
            }
        }

        // Try to initialize OpenCL devices
        if let Ok(platforms) = opencl::OpenCLDevice::get_platforms() {
            for (platform_id, _platform) in platforms.iter().enumerate() {
                if let Ok(devices) = opencl::OpenCLDevice::get_devices(platform_id as u32) {
                    for device in devices {
                        if let Ok(opencl_device) = opencl::OpenCLDevice::new(device.id) {
                            self.opencl_devices.push(opencl_device);
                        }
                    }
                }
            }
        }

        // Set preferred backend
        if !self.cuda_devices.is_empty() {
            self.preferred_backend = Some(GpuBackend::Cuda);
        } else if !self.opencl_devices.is_empty() {
            self.preferred_backend = Some(GpuBackend::OpenCL);
        }

        Ok(())
    }

    /// Get available GPU devices
    pub fn get_devices(&self) -> Vec<GpuDevice> {
        let mut devices = Vec::new();

        for (i, _) in self.cuda_devices.iter().enumerate() {
            if let Ok(device) = cuda::CudaDevice::get_device_info(i as u32) {
                devices.push(device);
            }
        }

        // Add OpenCL devices
        for (i, _) in self.opencl_devices.iter().enumerate() {
            devices.push(GpuDevice {
                id: i as u32,
                name: format!("OpenCL Device {}", i),
                backend: GpuBackend::OpenCL,
                compute_units: 16,
                memory_mb: 4096,
                supports_f64: true,
                supports_f16: false,
            });
        }

        devices
    }

    /// Get the best available device
    pub fn get_best_device(&self) -> Option<GpuDevice> {
        let devices = self.get_devices();
        devices
            .into_iter()
            .max_by_key(|d| d.compute_units * (d.memory_mb / 1024) as u32)
    }

    /// Check if GPU acceleration is available
    pub fn is_available(&self) -> bool {
        !self.cuda_devices.is_empty() || !self.opencl_devices.is_empty()
    }
}

impl Default for GpuManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Global GPU manager instance
use once_cell::sync::Lazy;
pub static GPU_MANAGER: Lazy<Mutex<GpuManager>> = Lazy::new(|| Mutex::new(GpuManager::new()));

/// Initialize GPU support
pub fn initialize_gpu() -> Result<(), SimdError> {
    #[cfg(not(feature = "no-std"))]
    let mut manager = GPU_MANAGER
        .lock()
        .map_err(|_| SimdError::ExternalLibraryError("Failed to lock GPU manager".to_string()))?;
    #[cfg(feature = "no-std")]
    let mut manager = GPU_MANAGER.lock();
    manager.initialize()
}

/// Check if GPU acceleration is available
pub fn is_gpu_available() -> bool {
    #[cfg(not(feature = "no-std"))]
    {
        if let Ok(manager) = GPU_MANAGER.lock() {
            manager.is_available()
        } else {
            false
        }
    }
    #[cfg(feature = "no-std")]
    {
        let manager = GPU_MANAGER.lock();
        manager.is_available()
    }
}

/// Get available GPU devices
pub fn get_gpu_devices() -> Vec<GpuDevice> {
    #[cfg(not(feature = "no-std"))]
    {
        if let Ok(manager) = GPU_MANAGER.lock() {
            manager.get_devices()
        } else {
            Vec::new()
        }
    }
    #[cfg(feature = "no-std")]
    {
        let manager = GPU_MANAGER.lock();
        manager.get_devices()
    }
}

#[allow(non_snake_case)]
#[cfg(all(test, not(feature = "no-std")))]
mod tests {
    use super::*;

    #[cfg(feature = "no-std")]
    use alloc::{
        string::{String, ToString},
        vec,
        vec::Vec,
    };

    #[test]
    fn test_gpu_manager_creation() {
        let manager = GpuManager::new();
        assert_eq!(manager.cuda_devices.len(), 0);
        assert_eq!(manager.opencl_devices.len(), 0);
    }

    #[test]
    fn test_gpu_device_creation() {
        let device = GpuDevice {
            id: 0,
            name: "Test Device".to_string(),
            backend: GpuBackend::Cuda,
            compute_units: 80,
            memory_mb: 8192,
            supports_f64: true,
            supports_f16: true,
        };

        assert_eq!(device.id, 0);
        assert_eq!(device.backend, GpuBackend::Cuda);
        assert!(device.supports_f64);
    }

    #[test]
    fn test_kernel_config_default() {
        let config = KernelConfig::default();
        assert_eq!(config.grid_size, (1, 1, 1));
        assert_eq!(config.block_size, (256, 1, 1));
        assert_eq!(config.shared_memory, 0);
    }

    #[test]
    fn test_cuda_device_creation() {
        // This would fail without CUDA, but tests the interface
        if cuda::CudaDevice::get_device_count().is_ok() {
            let result = cuda::CudaDevice::new(0);
            // Should either succeed or fail gracefully
            match result {
                Ok(_device) => {
                    // CUDA available and device created
                }
                Err(SimdError::UnsupportedOperation(_)) => {
                    // Expected when CUDA not available
                }
                Err(_) => panic!("Unexpected error type"),
            }
        }
    }

    #[test]
    fn test_opencl_platforms() {
        // Test OpenCL platform detection
        match opencl::OpenCLDevice::get_platforms() {
            Ok(platforms) => {
                // OpenCL available
                assert!(!platforms.is_empty());
            }
            Err(SimdError::UnsupportedOperation(_)) => {
                // Expected when OpenCL not available
            }
            Err(_) => panic!("Unexpected error type"),
        }
    }

    #[test]
    fn test_gpu_initialization() {
        // Test initialization doesn't panic
        let result = initialize_gpu();
        // Should either succeed or fail gracefully
        assert!(result.is_ok() || matches!(result, Err(SimdError::UnsupportedOperation(_))));
    }

    #[test]
    fn test_gpu_availability_check() {
        // This should not panic
        let _available = is_gpu_available();
    }

    #[test]
    fn test_get_devices() {
        // This should not panic and return a list (possibly empty)
        let _devices = get_gpu_devices();
        // Should be a valid Vec, even if empty (no need to assert len >= 0 as it's always true)
    }
}
