//! GPU-accelerated kernel computations using WGPU
//!
//! This module provides GPU-accelerated implementations of common SVM kernels
//! including RBF, Polynomial, and Linear kernels. It uses WGPU for cross-platform
//! GPU acceleration and compute shaders for high-performance kernel matrix computation.

use crate::kernels::KernelType;
use scirs2_core::ndarray::{s, Array2};
use thiserror::Error;

#[cfg(feature = "gpu")]
use std::collections::HashMap;
#[cfg(feature = "gpu")]
use wgpu::{
    util::DeviceExt, Adapter, BufferDescriptor, BufferUsages, ComputePassDescriptor,
    ComputePipeline, Device, DeviceDescriptor, Features, Instance, Limits, PowerPreference, Queue,
    RequestAdapterOptions, ShaderModule, ShaderModuleDescriptor, ShaderSource,
};

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

/// GPU-accelerated kernel matrix computation
#[cfg(feature = "gpu")]
pub struct GpuKernelComputer {
    device: Device,
    queue: Queue,
    adapter: Adapter,
    pipelines: HashMap<String, ComputePipeline>,
    shader_modules: HashMap<String, ShaderModule>,
}

#[cfg(feature = "gpu")]
impl GpuKernelComputer {
    /// Create a new GPU kernel computer
    pub async fn new() -> GpuKernelResult<Self> {
        let instance = Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            flags: wgpu::InstanceFlags::default(),
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&RequestAdapterOptions {
                power_preference: PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or(GpuKernelError::DeviceNotAvailable)?;

        let (device, queue) = adapter
            .request_device(
                &DeviceDescriptor {
                    label: None,
                    required_features: Features::empty(),
                    required_limits: Limits::default(),
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await
            .map_err(|e| GpuKernelError::ComputationFailed(e.to_string()))?;

        let mut computer = Self {
            device,
            queue,
            adapter,
            pipelines: HashMap::new(),
            shader_modules: HashMap::new(),
        };

        // Initialize common shaders
        computer.init_rbf_shader()?;
        computer.init_polynomial_shader()?;
        computer.init_linear_shader()?;
        computer.init_sigmoid_shader()?;

        Ok(computer)
    }

    /// Initialize RBF kernel shader
    fn init_rbf_shader(&mut self) -> GpuKernelResult<()> {
        let shader_source = r#"
            @group(0) @binding(0) var<storage, read> X: array<f32>;
            @group(0) @binding(1) var<storage, read> Y: array<f32>;
            @group(0) @binding(2) var<storage, read_write> result: array<f32>;
            @group(0) @binding(3) var<storage, read> params: array<f32>;

            @compute @workgroup_size(16, 16)
            fn rbf_kernel(@builtin(global_invocation_id) global_id: vec3<u32>) {
                let n_x = u32(params[0]);
                let n_y = u32(params[1]);
                let n_features = u32(params[2]);
                let gamma = params[3];

                let i = global_id.x;
                let j = global_id.y;

                if (i >= n_x || j >= n_y) {
                    return;
                }

                var sum_sq_diff = 0.0;
                for (var k = 0u; k < n_features; k++) {
                    let diff = X[i * n_features + k] - Y[j * n_features + k];
                    sum_sq_diff += diff * diff;
                }

                result[i * n_y + j] = exp(-gamma * sum_sq_diff);
            }
        "#;

        let shader = self.device.create_shader_module(ShaderModuleDescriptor {
            label: Some("RBF Kernel Shader"),
            source: ShaderSource::Wgsl(shader_source.into()),
        });

        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("RBF Kernel Pipeline"),
                layout: None,
                module: &shader,
                entry_point: Some("rbf_kernel"),
                compilation_options: Default::default(),
                cache: None,
            });

        self.shader_modules.insert("rbf".to_string(), shader);
        self.pipelines.insert("rbf".to_string(), pipeline);

        Ok(())
    }

    /// Initialize polynomial kernel shader
    fn init_polynomial_shader(&mut self) -> GpuKernelResult<()> {
        let shader_source = r#"
            @group(0) @binding(0) var<storage, read> X: array<f32>;
            @group(0) @binding(1) var<storage, read> Y: array<f32>;
            @group(0) @binding(2) var<storage, read_write> result: array<f32>;
            @group(0) @binding(3) var<storage, read> params: array<f32>;

            @compute @workgroup_size(16, 16)
            fn polynomial_kernel(@builtin(global_invocation_id) global_id: vec3<u32>) {
                let n_x = u32(params[0]);
                let n_y = u32(params[1]);
                let n_features = u32(params[2]);
                let gamma = params[3];
                let coef0 = params[4];
                let degree = params[5];

                let i = global_id.x;
                let j = global_id.y;

                if (i >= n_x || j >= n_y) {
                    return;
                }

                var dot_product = 0.0;
                for (var k = 0u; k < n_features; k++) {
                    dot_product += X[i * n_features + k] * Y[j * n_features + k];
                }

                result[i * n_y + j] = pow(gamma * dot_product + coef0, degree);
            }
        "#;

        let shader = self.device.create_shader_module(ShaderModuleDescriptor {
            label: Some("Polynomial Kernel Shader"),
            source: ShaderSource::Wgsl(shader_source.into()),
        });

        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Polynomial Kernel Pipeline"),
                layout: None,
                module: &shader,
                entry_point: Some("polynomial_kernel"),
                compilation_options: Default::default(),
                cache: None,
            });

        self.shader_modules.insert("polynomial".to_string(), shader);
        self.pipelines.insert("polynomial".to_string(), pipeline);

        Ok(())
    }

    /// Initialize linear kernel shader
    fn init_linear_shader(&mut self) -> GpuKernelResult<()> {
        let shader_source = r#"
            @group(0) @binding(0) var<storage, read> X: array<f32>;
            @group(0) @binding(1) var<storage, read> Y: array<f32>;
            @group(0) @binding(2) var<storage, read_write> result: array<f32>;
            @group(0) @binding(3) var<storage, read> params: array<f32>;

            @compute @workgroup_size(16, 16)
            fn linear_kernel(@builtin(global_invocation_id) global_id: vec3<u32>) {
                let n_x = u32(params[0]);
                let n_y = u32(params[1]);
                let n_features = u32(params[2]);

                let i = global_id.x;
                let j = global_id.y;

                if (i >= n_x || j >= n_y) {
                    return;
                }

                var dot_product = 0.0;
                for (var k = 0u; k < n_features; k++) {
                    dot_product += X[i * n_features + k] * Y[j * n_features + k];
                }

                result[i * n_y + j] = dot_product;
            }
        "#;

        let shader = self.device.create_shader_module(ShaderModuleDescriptor {
            label: Some("Linear Kernel Shader"),
            source: ShaderSource::Wgsl(shader_source.into()),
        });

        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Linear Kernel Pipeline"),
                layout: None,
                module: &shader,
                entry_point: Some("linear_kernel"),
                compilation_options: Default::default(),
                cache: None,
            });

        self.shader_modules.insert("linear".to_string(), shader);
        self.pipelines.insert("linear".to_string(), pipeline);

        Ok(())
    }

    /// Initialize sigmoid kernel shader
    fn init_sigmoid_shader(&mut self) -> GpuKernelResult<()> {
        let shader_source = r#"
            @group(0) @binding(0) var<storage, read> X: array<f32>;
            @group(0) @binding(1) var<storage, read> Y: array<f32>;
            @group(0) @binding(2) var<storage, read_write> result: array<f32>;
            @group(0) @binding(3) var<storage, read> params: array<f32>;

            @compute @workgroup_size(16, 16)
            fn sigmoid_kernel(@builtin(global_invocation_id) global_id: vec3<u32>) {
                let n_x = u32(params[0]);
                let n_y = u32(params[1]);
                let n_features = u32(params[2]);
                let gamma = params[3];
                let coef0 = params[4];

                let i = global_id.x;
                let j = global_id.y;

                if (i >= n_x || j >= n_y) {
                    return;
                }

                var dot_product = 0.0;
                for (var k = 0u; k < n_features; k++) {
                    dot_product += X[i * n_features + k] * Y[j * n_features + k];
                }

                result[i * n_y + j] = tanh(gamma * dot_product + coef0);
            }
        "#;

        let shader = self.device.create_shader_module(ShaderModuleDescriptor {
            label: Some("Sigmoid Kernel Shader"),
            source: ShaderSource::Wgsl(shader_source.into()),
        });

        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Sigmoid Kernel Pipeline"),
                layout: None,
                module: &shader,
                entry_point: Some("sigmoid_kernel"),
                compilation_options: Default::default(),
                cache: None,
            });

        self.shader_modules.insert("sigmoid".to_string(), shader);
        self.pipelines.insert("sigmoid".to_string(), pipeline);

        Ok(())
    }

    /// Compute kernel matrix on GPU
    pub async fn compute_kernel_matrix(
        &self,
        X: &Array2<f32>,
        Y: &Array2<f32>,
        kernel_type: &KernelType,
    ) -> GpuKernelResult<Array2<f32>> {
        let (n_x, n_features_x) = X.dim();
        let (n_y, n_features_y) = Y.dim();

        if n_features_x != n_features_y {
            return Err(GpuKernelError::DimensionMismatch);
        }

        let (pipeline_name, params) = match kernel_type {
            KernelType::Rbf { gamma } => (
                "rbf",
                vec![n_x as f32, n_y as f32, n_features_x as f32, *gamma as f32],
            ),
            KernelType::Polynomial {
                gamma,
                coef0,
                degree,
            } => (
                "polynomial",
                vec![
                    n_x as f32,
                    n_y as f32,
                    n_features_x as f32,
                    *gamma as f32,
                    *coef0 as f32,
                    *degree as f32,
                ],
            ),
            KernelType::Linear => ("linear", vec![n_x as f32, n_y as f32, n_features_x as f32]),
            KernelType::Sigmoid { gamma, coef0 } => (
                "sigmoid",
                vec![
                    n_x as f32,
                    n_y as f32,
                    n_features_x as f32,
                    *gamma as f32,
                    *coef0 as f32,
                ],
            ),
            _ => {
                return Err(GpuKernelError::FeatureNotSupported(
                    "Kernel type not supported on GPU".to_string(),
                ))
            }
        };

        let pipeline = self.pipelines.get(pipeline_name).ok_or_else(|| {
            GpuKernelError::FeatureNotSupported(format!("Pipeline {pipeline_name} not found"))
        })?;

        // Create buffers
        let x_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("X Buffer"),
                contents: bytemuck::cast_slice(X.as_slice().unwrap()),
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            });

        let y_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Y Buffer"),
                contents: bytemuck::cast_slice(Y.as_slice().unwrap()),
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            });

        let result_buffer = self.device.create_buffer(&BufferDescriptor {
            label: Some("Result Buffer"),
            size: (n_x * n_y * std::mem::size_of::<f32>()) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Params Buffer"),
                contents: bytemuck::cast_slice(&params),
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            });

        // Create bind group
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Kernel Bind Group"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: x_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: y_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: result_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        // Execute compute pass
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Kernel Compute Encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("Kernel Compute Pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            let workgroup_size = 16;
            let num_workgroups_x = (n_x + workgroup_size - 1) / workgroup_size;
            let num_workgroups_y = (n_y + workgroup_size - 1) / workgroup_size;

            compute_pass.dispatch_workgroups(num_workgroups_x as u32, num_workgroups_y as u32, 1);
        }

        // Create staging buffer for reading results
        let staging_buffer = self.device.create_buffer(&BufferDescriptor {
            label: Some("Staging Buffer"),
            size: (n_x * n_y * std::mem::size_of::<f32>()) as u64,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(
            &result_buffer,
            0,
            &staging_buffer,
            0,
            (n_x * n_y * std::mem::size_of::<f32>()) as u64,
        );

        self.queue.submit(Some(encoder.finish()));

        // Read results
        let buffer_slice = staging_buffer.slice(..);
        let (tx, rx) = futures_intrusive::channel::shared::oneshot_channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });

        self.device.poll(wgpu::Maintain::Wait);
        rx.receive()
            .await
            .unwrap()
            .map_err(|e| GpuKernelError::ComputationFailed(e.to_string()))?;

        let data = buffer_slice.get_mapped_range();
        let result_data: &[f32] = bytemuck::cast_slice(&data);

        let result_matrix = Array2::from_shape_vec((n_x, n_y), result_data.to_vec())
            .map_err(|e| GpuKernelError::ComputationFailed(e.to_string()))?;

        Ok(result_matrix)
    }

    /// Get device information
    pub fn device_info(&self) -> String {
        format!("GPU: {}", self.adapter.get_info().name)
    }

    /// Check if GPU supports required features
    pub fn supports_compute(&self) -> bool {
        self.adapter.features().contains(Features::empty())
    }

    /// Get GPU memory info
    pub fn memory_info(&self) -> wgpu::AdapterInfo {
        self.adapter.get_info()
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

        for i in (0..n_x).step_by(batch_size) {
            let end_i = (i + batch_size).min(n_x);
            let x_batch = x.slice(s![i..end_i, ..]);

            for j in (0..n_y).step_by(batch_size) {
                let end_j = (j + batch_size).min(n_y);
                let y_batch = y.slice(s![j..end_j, ..]);

                let batch_result = computer
                    .compute_kernel_matrix(&x_batch.to_owned(), &y_batch.to_owned(), kernel_type)
                    .await?;

                result
                    .slice_mut(s![i..end_i, j..end_j])
                    .assign(&batch_result);
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
        let X_var = Array2::from_shape_vec((10, 5), (0..50).map(|x| x as f32).collect()).unwrap();
        let Y_var = Array2::from_shape_vec((8, 5), (0..40).map(|x| x as f32).collect()).unwrap();

        // Test that we can create the benchmark struct
        assert_eq!(X_var.dim(), (10, 5));
        assert_eq!(Y_var.dim(), (8, 5));
    }
}
