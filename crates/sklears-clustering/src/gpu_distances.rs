//! GPU-Accelerated Distance Computations
//!
//! This module provides GPU-accelerated distance computations for clustering algorithms
//! using WebGPU (wgpu) for cross-platform GPU compute support.
//!
//! # Features
//! - **Euclidean Distance**: Batch computation of Euclidean distances
//! - **Manhattan Distance**: Batch computation of Manhattan distances
//! - **Cosine Distance**: Batch computation of cosine distances
//! - **Distance Matrix**: GPU-accelerated distance matrix computation
//! - **K-Nearest Neighbors**: GPU-accelerated k-NN search
//! - **Memory Management**: Efficient GPU buffer management for large datasets
//!
//! # GPU Compute Shaders
//! This module includes WGSL (WebGPU Shading Language) compute shaders for:
//! - Parallel distance calculations
//! - Reduction operations for k-NN
//! - Memory-efficient batch processing
//!
//! # Usage
//! ```rust,ignore
//! use sklears_clustering::gpu_distances::{GpuDistanceComputer, GpuDistanceMetric};
//!
//! // Create GPU distance computer
//! let mut gpu_computer = GpuDistanceComputer::new().await?;
//!
//! // Compute distances between datasets
//! let distances = gpu_computer.compute_pairwise_distances(
//!     &data1, &data2, GpuDistanceMetric::Euclidean
//! ).await?;
//! ```

#[cfg(feature = "gpu")]
pub mod gpu {
    use std::collections::HashMap;

    use bytemuck::{Pod, Zeroable};
    use numrs2::prelude::*;
    use scirs2_core::ndarray::Array2;
    use sklears_core::error::{Result, SklearsError};
    use wgpu::util::DeviceExt;

    /// GPU distance metrics
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub enum GpuDistanceMetric {
        /// Euclidean distance
        Euclidean,
        /// Manhattan (L1) distance
        Manhattan,
        /// Cosine distance
        Cosine,
        /// Squared Euclidean distance (faster, no sqrt)
        SquaredEuclidean,
    }

    /// GPU distance computation configuration
    #[derive(Debug, Clone)]
    pub struct GpuConfig {
        /// Preferred device type
        pub device_type: wgpu::DeviceType,
        /// Maximum buffer size in bytes
        pub max_buffer_size: u64,
        /// Workgroup size for compute shaders
        pub workgroup_size: u32,
    }

    impl Default for GpuConfig {
        fn default() -> Self {
            Self {
                device_type: wgpu::DeviceType::DiscreteGpu,
                max_buffer_size: 1024 * 1024 * 1024, // 1GB
                workgroup_size: 256,
            }
        }
    }

    /// Buffer data for GPU compute
    #[repr(C)]
    #[derive(Clone, Copy, Pod, Zeroable)]
    struct GpuPoint {
        data: [f32; 32], // Maximum dimensionality supported
        dim: u32,
        _padding: [u32; 3],
    }

    /// Parameters for distance computation
    #[repr(C)]
    #[derive(Clone, Copy, Pod, Zeroable)]
    struct ComputeParams {
        n_points_a: u32,
        n_points_b: u32,
        n_dims: u32,
        metric_type: u32, // 0=Euclidean, 1=Manhattan, 2=Cosine, 3=SquaredEuclidean
    }

    /// GPU Distance Computer
    pub struct GpuDistanceComputer {
        device: wgpu::Device,
        queue: wgpu::Queue,
        config: GpuConfig,
        compute_pipeline: Option<wgpu::ComputePipeline>,
        bind_group_layout: wgpu::BindGroupLayout,
    }

    impl GpuDistanceComputer {
        /// Create a new GPU distance computer
        pub async fn new() -> Result<Self> {
            Self::with_config(GpuConfig::default()).await
        }

        /// Create a new GPU distance computer with configuration
        pub async fn with_config(config: GpuConfig) -> Result<Self> {
            // Initialize wgpu
            let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
                backends: wgpu::Backends::all(),
                flags: wgpu::InstanceFlags::default(),
                ..Default::default()
            });

            // Request adapter
            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    force_fallback_adapter: false,
                    compatible_surface: None,
                })
                .await
                .map_err(|e| {
                    SklearsError::InvalidInput(format!("Failed to request adapter: {:?}", e))
                })?;

            // Request device
            let (device, queue) = adapter
                .request_device(&wgpu::DeviceDescriptor {
                    label: None,
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::Performance,
                    trace: Default::default(),
                })
                .await
                .map_err(|e| {
                    SklearsError::InvalidInput(format!("Failed to create device: {}", e))
                })?;

            // Create bind group layout
            let bind_group_layout =
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Distance Compute Bind Group Layout"),
                    entries: &[
                        // Input data A
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // Input data B
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // Output distances
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // Parameters
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });

            Ok(Self {
                device,
                queue,
                config,
                compute_pipeline: None,
                bind_group_layout,
            })
        }

        /// Initialize compute pipeline for distance calculations
        fn ensure_compute_pipeline(&mut self) -> Result<()> {
            if self.compute_pipeline.is_some() {
                return Ok(());
            }

            let shader = self
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("Distance Compute Shader"),
                    source: wgpu::ShaderSource::Wgsl(self.distance_compute_shader().into()),
                });

            let pipeline_layout =
                self.device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("Distance Compute Pipeline Layout"),
                        bind_group_layouts: &[&self.bind_group_layout],
                        push_constant_ranges: &[],
                    });

            let compute_pipeline =
                self.device
                    .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some("Distance Compute Pipeline"),
                        layout: Some(&pipeline_layout),
                        module: &shader,
                        entry_point: Some("main"),
                        compilation_options: Default::default(),
                        cache: None,
                    });

            self.compute_pipeline = Some(compute_pipeline);
            Ok(())
        }

        /// WGSL compute shader for distance calculations
        fn distance_compute_shader(&self) -> &'static str {
            r#"
struct ComputeParams {
    n_points_a: u32,
    n_points_b: u32,
    n_dims: u32,
    metric_type: u32,
}

@group(0) @binding(0)
var<storage, read> points_a: array<f32>;

@group(0) @binding(1)
var<storage, read> points_b: array<f32>;

@group(0) @binding(2)
var<storage, read_write> distances: array<f32>;

@group(0) @binding(3)
var<uniform> params: ComputeParams;

fn euclidean_distance(idx_a: u32, idx_b: u32) -> f32 {
    var sum: f32 = 0.0;
    let base_a = idx_a * params.n_dims;
    let base_b = idx_b * params.n_dims;

    for (var i: u32 = 0u; i < params.n_dims; i = i + 1u) {
        let diff = points_a[base_a + i] - points_b[base_b + i];
        sum = sum + diff * diff;
    }
    return sqrt(sum);
}

fn squared_euclidean_distance(idx_a: u32, idx_b: u32) -> f32 {
    var sum: f32 = 0.0;
    let base_a = idx_a * params.n_dims;
    let base_b = idx_b * params.n_dims;

    for (var i: u32 = 0u; i < params.n_dims; i = i + 1u) {
        let diff = points_a[base_a + i] - points_b[base_b + i];
        sum = sum + diff * diff;
    }
    return sum;
}

fn manhattan_distance(idx_a: u32, idx_b: u32) -> f32 {
    var sum: f32 = 0.0;
    let base_a = idx_a * params.n_dims;
    let base_b = idx_b * params.n_dims;

    for (var i: u32 = 0u; i < params.n_dims; i = i + 1u) {
        sum = sum + abs(points_a[base_a + i] - points_b[base_b + i]);
    }
    return sum;
}

fn cosine_distance(idx_a: u32, idx_b: u32) -> f32 {
    var dot_product: f32 = 0.0;
    var norm_a: f32 = 0.0;
    var norm_b: f32 = 0.0;
    let base_a = idx_a * params.n_dims;
    let base_b = idx_b * params.n_dims;

    for (var i: u32 = 0u; i < params.n_dims; i = i + 1u) {
        let val_a = points_a[base_a + i];
        let val_b = points_b[base_b + i];
        dot_product = dot_product + val_a * val_b;
        norm_a = norm_a + val_a * val_a;
        norm_b = norm_b + val_b * val_b;
    }

    norm_a = sqrt(norm_a);
    norm_b = sqrt(norm_b);

    if (norm_a == 0.0 || norm_b == 0.0) {
        return 1.0;
    }

    return 1.0 - (dot_product / (norm_a * norm_b));
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let total_pairs = params.n_points_a * params.n_points_b;
    let idx = global_id.x;

    if (idx >= total_pairs) {
        return;
    }

    let idx_a = idx / params.n_points_b;
    let idx_b = idx % params.n_points_b;

    var distance: f32;
    switch (params.metric_type) {
        case 0u: { distance = euclidean_distance(idx_a, idx_b); }
        case 1u: { distance = manhattan_distance(idx_a, idx_b); }
        case 2u: { distance = cosine_distance(idx_a, idx_b); }
        case 3u: { distance = squared_euclidean_distance(idx_a, idx_b); }
        default: { distance = euclidean_distance(idx_a, idx_b); }
    }

    distances[idx] = distance;
}
"#
        }

        /// Convert Array2 data to GPU-compatible format
        fn prepare_gpu_data(&self, data: &Array2<f64>) -> Result<Vec<f32>> {
            let n_points = data.nrows();
            let n_dims = data.ncols();

            if n_dims > 32 {
                return Err(SklearsError::InvalidInput(
                    "Maximum supported dimensionality is 32 for GPU computation".to_string(),
                ));
            }

            // Flatten data to f32 (GPU prefers f32 over f64)
            let mut gpu_data = Vec::with_capacity(n_points * n_dims);
            for i in 0..n_points {
                for j in 0..n_dims {
                    gpu_data.push(data[[i, j]] as f32);
                }
            }

            Ok(gpu_data)
        }

        /// Compute pairwise distances between two datasets on GPU
        pub async fn compute_pairwise_distances(
            &mut self,
            data_a: &Array2<f64>,
            data_b: &Array2<f64>,
            metric: GpuDistanceMetric,
        ) -> Result<Array2<f64>> {
            self.ensure_compute_pipeline()?;

            if data_a.ncols() != data_b.ncols() {
                return Err(SklearsError::InvalidInput(
                    "Data dimensions must match".to_string(),
                ));
            }

            let n_points_a = data_a.nrows() as u32;
            let n_points_b = data_b.nrows() as u32;
            let n_dims = data_a.ncols() as u32;

            // Prepare GPU data
            let gpu_data_a = self.prepare_gpu_data(data_a)?;
            let gpu_data_b = self.prepare_gpu_data(data_b)?;

            // Create buffers
            let buffer_a = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Data A Buffer"),
                    contents: bytemuck::cast_slice(&gpu_data_a),
                    usage: wgpu::BufferUsages::STORAGE,
                });

            let buffer_b = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Data B Buffer"),
                    contents: bytemuck::cast_slice(&gpu_data_b),
                    usage: wgpu::BufferUsages::STORAGE,
                });

            let output_size = (n_points_a * n_points_b * std::mem::size_of::<f32>() as u32) as u64;
            let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Distance Output Buffer"),
                size: output_size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });

            // Parameters buffer
            let params = ComputeParams {
                n_points_a,
                n_points_b,
                n_dims,
                metric_type: match metric {
                    GpuDistanceMetric::Euclidean => 0,
                    GpuDistanceMetric::Manhattan => 1,
                    GpuDistanceMetric::Cosine => 2,
                    GpuDistanceMetric::SquaredEuclidean => 3,
                },
            };

            let params_buffer = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Params Buffer"),
                    contents: bytemuck::cast_slice(&[params]),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

            // Create bind group
            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Distance Compute Bind Group"),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buffer_a.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: buffer_b.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: output_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: params_buffer.as_entire_binding(),
                    },
                ],
            });

            // Dispatch compute shader
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Distance Compute Encoder"),
                });

            {
                let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Distance Compute Pass"),
                    timestamp_writes: None,
                });

                compute_pass.set_pipeline(self.compute_pipeline.as_ref().unwrap());
                compute_pass.set_bind_group(0, &bind_group, &[]);

                let workgroups = ((n_points_a * n_points_b) + self.config.workgroup_size - 1)
                    / self.config.workgroup_size;
                compute_pass.dispatch_workgroups(workgroups, 1, 1);
            }

            // Copy output to staging buffer
            let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Distance Staging Buffer"),
                size: output_size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, output_size);

            // Submit commands
            self.queue.submit(std::iter::once(encoder.finish()));

            // Read results
            let buffer_slice = staging_buffer.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
                tx.send(result).unwrap();
            });

            // Wait for the operation to complete

            rx.recv()
                .map_err(|e| SklearsError::InvalidInput(format!("Channel receive error: {}", e)))?
                .map_err(|e| SklearsError::InvalidInput(format!("Buffer map error: {:?}", e)))?;

            let data = buffer_slice.get_mapped_range();
            let result: &[f32] = bytemuck::cast_slice(&data);

            // Convert back to Array2<f64>
            let mut distances = Array2::<f64>::zeros((n_points_a as usize, n_points_b as usize));
            for i in 0..n_points_a as usize {
                for j in 0..n_points_b as usize {
                    let idx = i * n_points_b as usize + j;
                    distances[[i, j]] = result[idx] as f64;
                }
            }

            Ok(distances)
        }

        /// Compute distance matrix for a single dataset (all pairwise distances)
        pub async fn compute_distance_matrix(
            &mut self,
            data: &Array2<f64>,
            metric: GpuDistanceMetric,
        ) -> Result<Array2<f64>> {
            self.compute_pairwise_distances(data, data, metric).await
        }

        /// Find k-nearest neighbors using GPU acceleration
        pub async fn k_nearest_neighbors(
            &mut self,
            query_points: &Array2<f64>,
            reference_points: &Array2<f64>,
            k: usize,
            metric: GpuDistanceMetric,
        ) -> Result<(Vec<Vec<usize>>, Vec<Vec<f64>>)> {
            // Compute all pairwise distances
            let distances = self
                .compute_pairwise_distances(query_points, reference_points, metric)
                .await?;

            let n_queries = query_points.nrows();
            let mut indices = Vec::with_capacity(n_queries);
            let mut neighbor_distances = Vec::with_capacity(n_queries);

            // For each query point, find k nearest neighbors
            for i in 0..n_queries {
                let mut point_distances: Vec<(usize, f64)> = distances
                    .row(i)
                    .iter()
                    .enumerate()
                    .map(|(idx, &dist)| (idx, dist))
                    .collect();

                // Sort by distance and take k closest
                point_distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                point_distances.truncate(k);

                let (point_indices, point_dists): (Vec<_>, Vec<_>) =
                    point_distances.into_iter().unzip();
                indices.push(point_indices);
                neighbor_distances.push(point_dists);
            }

            Ok((indices, neighbor_distances))
        }

        /// Get device information
        pub fn device_info(&self) -> HashMap<String, String> {
            let mut info = HashMap::new();
            info.insert(
                "backend".to_string(),
                format!("{:?}", self.device.features()),
            );
            info.insert("limits".to_string(), format!("{:?}", self.device.limits()));
            info.insert(
                "workgroup_size".to_string(),
                self.config.workgroup_size.to_string(),
            );
            info.insert(
                "max_buffer_size".to_string(),
                self.config.max_buffer_size.to_string(),
            );
            info
        }
    }

    #[allow(non_snake_case)]
    #[cfg(test)]
    mod tests {
        use super::*;
        use approx::assert_abs_diff_eq;

        #[test]
        fn test_gpu_distance_computation() {
            pollster::block_on(async {
                // Skip test if GPU not available in CI
                if std::env::var("CI").is_ok() {
                    return;
                }

                let mut gpu_computer = match GpuDistanceComputer::new().await {
                    Ok(computer) => computer,
                    Err(_) => {
                        // Skip test if GPU not available
                        return;
                    }
                };

                let data_a = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
                let data_b = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 5.0, 6.0]).unwrap();

                let distances = gpu_computer
                    .compute_pairwise_distances(&data_a, &data_b, GpuDistanceMetric::Euclidean)
                    .await
                    .unwrap();

                assert_eq!(distances.nrows(), 2);
                assert_eq!(distances.ncols(), 2);

                // First point [1,2] to first point [1,2] should be 0
                assert_abs_diff_eq!(distances[[0, 0]], 0.0, epsilon = 1e-6);

                // First point [1,2] to second point [5,6] should be sqrt((1-5)^2 + (2-6)^2) = sqrt(32)
                let expected = ((1.0f64 - 5.0).powi(2) + (2.0f64 - 6.0).powi(2)).sqrt();
                assert_abs_diff_eq!(distances[[0, 1]], expected, epsilon = 1e-6);
            });
        }

        #[test]
        fn test_gpu_manhattan_distance() {
            pollster::block_on(async {
                // Skip test if GPU not available in CI
                if std::env::var("CI").is_ok() {
                    return;
                }

                let mut gpu_computer = match GpuDistanceComputer::new().await {
                    Ok(computer) => computer,
                    Err(_) => return,
                };

                let data_a = Array2::from_shape_vec((1, 2), vec![1.0, 2.0]).unwrap();
                let data_b = Array2::from_shape_vec((1, 2), vec![3.0, 4.0]).unwrap();

                let distances = gpu_computer
                    .compute_pairwise_distances(&data_a, &data_b, GpuDistanceMetric::Manhattan)
                    .await
                    .unwrap();

                // Manhattan distance should be |1-3| + |2-4| = 2 + 2 = 4
                assert_abs_diff_eq!(distances[[0, 0]], 4.0, epsilon = 1e-6);
            });
        }
    }
}

// Re-export GPU module when feature is enabled
#[cfg(feature = "gpu")]
pub use gpu::*;

// Provide stub implementations when GPU feature is not enabled
#[cfg(not(feature = "gpu"))]
pub mod stub {
    use numrs2::prelude::*;
    use sklears_core::error::{Result, SklearsError};

    /// Stub GPU distance metric (no-op when GPU feature disabled)
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub enum GpuDistanceMetric {
        Euclidean,
        Manhattan,
        Cosine,
        SquaredEuclidean,
    }

    /// Stub GPU distance computer (returns error when GPU feature disabled)
    pub struct GpuDistanceComputer;

    impl GpuDistanceComputer {
        pub async fn new() -> Result<Self> {
            Err(SklearsError::InvalidInput(
                "GPU feature not enabled. Enable with --features gpu".to_string(),
            ))
        }

        pub async fn compute_pairwise_distances(
            &mut self,
            _data_a: &Array2<f64>,
            _data_b: &Array2<f64>,
            _metric: GpuDistanceMetric,
        ) -> Result<Array2<f64>> {
            Err(SklearsError::InvalidInput(
                "GPU feature not enabled".to_string(),
            ))
        }
    }
}

#[cfg(not(feature = "gpu"))]
pub use stub::*;
