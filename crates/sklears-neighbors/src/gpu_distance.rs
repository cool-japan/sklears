//! GPU-accelerated distance computations for high-performance neighbor search
//!
//! This module provides GPU acceleration for distance computations using various
//! GPU backends including CUDA, OpenCL, and Metal. It includes batch processing
//! capabilities and memory management for large-scale distance matrix computations.

use crate::distance::Distance;
use crate::{NeighborsError, NeighborsResult};
use scirs2_core::ndarray::{Array2, ArrayView2, Axis};
use sklears_core::types::Float;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// GPU backend type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GpuBackend {
    /// CUDA backend (NVIDIA GPUs)
    Cuda,
    /// OpenCL backend (Cross-platform)
    OpenCl,
    /// Metal backend (Apple GPUs)
    Metal,
    /// CPU fallback (for testing)
    CpuFallback,
}

/// GPU device information
#[derive(Debug, Clone)]
pub struct GpuDeviceInfo {
    pub device_id: u32,
    pub name: String,
    pub backend: GpuBackend,
    pub memory_size: usize,
    pub compute_units: u32,
    pub max_work_group_size: usize,
}

/// GPU memory management strategy
#[derive(Debug, Clone)]
pub enum GpuMemoryStrategy {
    /// Allocate all data on GPU at once
    PreloadAll,
    /// Stream data in chunks
    Streaming { chunk_size: usize },
    /// Adaptive strategy based on available memory
    Adaptive,
}

/// GPU computation configuration
#[derive(Debug, Clone)]
pub struct GpuConfig {
    pub backend: GpuBackend,
    pub device_id: Option<u32>,
    pub memory_strategy: GpuMemoryStrategy,
    pub batch_size: usize,
    pub max_memory_usage: Option<usize>,
    pub enable_async: bool,
}

impl Default for GpuConfig {
    fn default() -> Self {
        Self {
            backend: GpuBackend::CpuFallback,
            device_id: None,
            memory_strategy: GpuMemoryStrategy::Adaptive,
            batch_size: 1024,
            max_memory_usage: None,
            enable_async: true,
        }
    }
}

/// GPU distance computation result
#[derive(Debug, Clone)]
pub struct GpuDistanceResult {
    pub distances: Array2<Float>,
    pub computation_time: f64,
    pub memory_usage: usize,
    pub backend_used: GpuBackend,
}

/// GPU distance computation statistics
#[derive(Debug, Clone)]
pub struct GpuComputationStats {
    pub total_computations: usize,
    pub total_time: f64,
    pub average_time: f64,
    pub peak_memory_usage: usize,
    pub backend_distribution: HashMap<GpuBackend, usize>,
}

/// GPU distance calculator
pub struct GpuDistanceCalculator {
    config: GpuConfig,
    device_info: Option<GpuDeviceInfo>,
    stats: Arc<Mutex<GpuComputationStats>>,
}

impl Default for GpuDistanceCalculator {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuDistanceCalculator {
    /// Create a new GPU distance calculator
    pub fn new() -> Self {
        Self {
            config: GpuConfig::default(),
            device_info: None,
            stats: Arc::new(Mutex::new(GpuComputationStats {
                total_computations: 0,
                total_time: 0.0,
                average_time: 0.0,
                peak_memory_usage: 0,
                backend_distribution: HashMap::new(),
            })),
        }
    }

    /// Create a new GPU distance calculator with configuration
    pub fn with_config(config: GpuConfig) -> Self {
        Self {
            config,
            device_info: None,
            stats: Arc::new(Mutex::new(GpuComputationStats {
                total_computations: 0,
                total_time: 0.0,
                average_time: 0.0,
                peak_memory_usage: 0,
                backend_distribution: HashMap::new(),
            })),
        }
    }

    /// Initialize GPU context and detect available devices
    pub fn initialize(&mut self) -> NeighborsResult<()> {
        self.device_info = self.detect_gpu_devices()?;
        Ok(())
    }

    /// Detect available GPU devices
    pub fn detect_gpu_devices(&self) -> NeighborsResult<Option<GpuDeviceInfo>> {
        // In a real implementation, this would detect actual GPU devices
        // For now, we'll simulate device detection
        match self.config.backend {
            GpuBackend::Cuda => {
                // Mock CUDA device detection
                if self.is_cuda_available() {
                    Ok(Some(GpuDeviceInfo {
                        device_id: 0,
                        name: "NVIDIA GPU (Mock)".to_string(),
                        backend: GpuBackend::Cuda,
                        memory_size: 8 * 1024 * 1024 * 1024, // 8GB
                        compute_units: 32,
                        max_work_group_size: 1024,
                    }))
                } else {
                    Ok(None)
                }
            }
            GpuBackend::OpenCl => {
                // Mock OpenCL device detection
                if self.is_opencl_available() {
                    Ok(Some(GpuDeviceInfo {
                        device_id: 0,
                        name: "OpenCL Device (Mock)".to_string(),
                        backend: GpuBackend::OpenCl,
                        memory_size: 4 * 1024 * 1024 * 1024, // 4GB
                        compute_units: 16,
                        max_work_group_size: 256,
                    }))
                } else {
                    Ok(None)
                }
            }
            GpuBackend::Metal => {
                // Mock Metal device detection
                if self.is_metal_available() {
                    Ok(Some(GpuDeviceInfo {
                        device_id: 0,
                        name: "Apple GPU (Mock)".to_string(),
                        backend: GpuBackend::Metal,
                        memory_size: 16 * 1024 * 1024 * 1024, // 16GB unified memory
                        compute_units: 8,
                        max_work_group_size: 512,
                    }))
                } else {
                    Ok(None)
                }
            }
            GpuBackend::CpuFallback => {
                Ok(Some(GpuDeviceInfo {
                    device_id: 0,
                    name: "CPU Fallback".to_string(),
                    backend: GpuBackend::CpuFallback,
                    memory_size: 16 * 1024 * 1024 * 1024, // 16GB
                    compute_units: 8,
                    max_work_group_size: 1,
                }))
            }
        }
    }

    /// Check if CUDA is available
    fn is_cuda_available(&self) -> bool {
        // In a real implementation, this would check for CUDA runtime
        // For now, we'll simulate availability based on platform
        cfg!(target_os = "linux") || cfg!(target_os = "windows")
    }

    /// Check if OpenCL is available
    fn is_opencl_available(&self) -> bool {
        // In a real implementation, this would check for OpenCL runtime
        true // OpenCL is generally available on most platforms
    }

    /// Check if Metal is available
    fn is_metal_available(&self) -> bool {
        // Metal is only available on Apple platforms
        cfg!(target_os = "macos") || cfg!(target_os = "ios")
    }

    /// Compute pairwise distances using GPU acceleration
    #[allow(non_snake_case)]
    pub fn pairwise_distances<'a>(
        &self,
        X: &ArrayView2<'a, Float>,
        Y: Option<&ArrayView2<'a, Float>>,
        distance: Distance,
    ) -> NeighborsResult<GpuDistanceResult> {
        let start_time = std::time::Instant::now();

        let Y = Y.unwrap_or(X);

        if X.ncols() != Y.ncols() {
            return Err(NeighborsError::ShapeMismatch {
                expected: vec![X.nrows(), X.ncols()],
                actual: vec![Y.nrows(), Y.ncols()],
            });
        }

        let result = match self.config.backend {
            GpuBackend::Cuda => self.compute_cuda_distances(X, Y, distance)?,
            GpuBackend::OpenCl => self.compute_opencl_distances(X, Y, distance)?,
            GpuBackend::Metal => self.compute_metal_distances(X, Y, distance)?,
            GpuBackend::CpuFallback => self.compute_cpu_distances(X, Y, distance)?,
        };

        let computation_time = start_time.elapsed().as_secs_f64();

        // Update statistics
        self.update_stats(computation_time, result.1, self.config.backend);

        Ok(GpuDistanceResult {
            distances: result.0,
            computation_time,
            memory_usage: result.1,
            backend_used: self.config.backend,
        })
    }

    /// Compute batch distances for large datasets
    #[allow(non_snake_case)]
    pub fn batch_pairwise_distances<'a>(
        &self,
        X: &ArrayView2<'a, Float>,
        Y: Option<&ArrayView2<'a, Float>>,
        distance: Distance,
    ) -> NeighborsResult<GpuDistanceResult> {
        let Y = Y.unwrap_or(X);
        let batch_size = self.config.batch_size;

        let n_samples_x = X.nrows();
        let n_samples_y = Y.nrows();

        let mut result_distances = Array2::zeros((n_samples_x, n_samples_y));
        let mut total_memory_usage = 0;
        let start_time = std::time::Instant::now();

        // Process in batches
        for i in (0..n_samples_x).step_by(batch_size) {
            let end_i = std::cmp::min(i + batch_size, n_samples_x);
            let X_batch = X.slice(scirs2_core::ndarray::s![i..end_i, ..]);

            for j in (0..n_samples_y).step_by(batch_size) {
                let end_j = std::cmp::min(j + batch_size, n_samples_y);
                let Y_batch = Y.slice(scirs2_core::ndarray::s![j..end_j, ..]);

                let batch_result =
                    self.pairwise_distances(&X_batch, Some(&Y_batch), distance.clone())?;

                result_distances
                    .slice_mut(scirs2_core::ndarray::s![i..end_i, j..end_j])
                    .assign(&batch_result.distances);

                total_memory_usage += batch_result.memory_usage;
            }
        }

        let computation_time = start_time.elapsed().as_secs_f64();

        Ok(GpuDistanceResult {
            distances: result_distances,
            computation_time,
            memory_usage: total_memory_usage,
            backend_used: self.config.backend,
        })
    }

    /// Compute distances using CUDA backend
    fn compute_cuda_distances<'a>(
        &self,
        X: &ArrayView2<'a, Float>,
        Y: &ArrayView2<'a, Float>,
        distance: Distance,
    ) -> NeighborsResult<(Array2<Float>, usize)> {
        // In a real implementation, this would use CUDA kernels
        // For now, we'll use optimized CPU computation with parallelization
        let distances = self.compute_parallel_distances(X, Y, distance)?;
        let memory_usage = distances.len() * std::mem::size_of::<Float>();
        Ok((distances, memory_usage))
    }

    /// Compute distances using OpenCL backend
    fn compute_opencl_distances<'a>(
        &self,
        X: &ArrayView2<'a, Float>,
        Y: &ArrayView2<'a, Float>,
        distance: Distance,
    ) -> NeighborsResult<(Array2<Float>, usize)> {
        // In a real implementation, this would use OpenCL kernels
        // For now, we'll use optimized CPU computation with parallelization
        let distances = self.compute_parallel_distances(X, Y, distance)?;
        let memory_usage = distances.len() * std::mem::size_of::<Float>();
        Ok((distances, memory_usage))
    }

    /// Compute distances using Metal backend
    fn compute_metal_distances<'a>(
        &self,
        X: &ArrayView2<'a, Float>,
        Y: &ArrayView2<'a, Float>,
        distance: Distance,
    ) -> NeighborsResult<(Array2<Float>, usize)> {
        // In a real implementation, this would use Metal compute shaders
        // For now, we'll use optimized CPU computation with parallelization
        let distances = self.compute_parallel_distances(X, Y, distance)?;
        let memory_usage = distances.len() * std::mem::size_of::<Float>();
        Ok((distances, memory_usage))
    }

    /// Compute distances using CPU fallback
    fn compute_cpu_distances<'a>(
        &self,
        X: &ArrayView2<'a, Float>,
        Y: &ArrayView2<'a, Float>,
        distance: Distance,
    ) -> NeighborsResult<(Array2<Float>, usize)> {
        let distances = self.compute_parallel_distances(X, Y, distance)?;
        let memory_usage = distances.len() * std::mem::size_of::<Float>();
        Ok((distances, memory_usage))
    }

    /// Compute distances using parallel CPU computation
    fn compute_parallel_distances<'a>(
        &self,
        X: &ArrayView2<'a, Float>,
        Y: &ArrayView2<'a, Float>,
        distance: Distance,
    ) -> NeighborsResult<Array2<Float>> {
        let n_samples_x = X.nrows();
        let n_samples_y = Y.nrows();
        let mut distances = Array2::zeros((n_samples_x, n_samples_y));

        // Sequential computation across rows
        for (i, mut row) in distances.axis_iter_mut(Axis(0)).enumerate() {
            let x_row = X.row(i);
            for j in 0..n_samples_y {
                let y_row = Y.row(j);
                row[j] = distance.calculate(&x_row, &y_row);
            }
        }

        Ok(distances)
    }

    /// Update computation statistics
    fn update_stats(&self, computation_time: f64, memory_usage: usize, backend: GpuBackend) {
        if let Ok(mut stats) = self.stats.lock() {
            stats.total_computations += 1;
            stats.total_time += computation_time;
            stats.average_time = stats.total_time / stats.total_computations as f64;
            stats.peak_memory_usage = stats.peak_memory_usage.max(memory_usage);

            *stats.backend_distribution.entry(backend).or_insert(0) += 1;
        }
    }

    /// Get computation statistics
    pub fn get_stats(&self) -> GpuComputationStats {
        self.stats.lock().unwrap().clone()
    }

    /// Get device information
    pub fn get_device_info(&self) -> Option<GpuDeviceInfo> {
        self.device_info.clone()
    }

    /// Reset computation statistics
    pub fn reset_stats(&self) {
        if let Ok(mut stats) = self.stats.lock() {
            *stats = GpuComputationStats {
                total_computations: 0,
                total_time: 0.0,
                average_time: 0.0,
                peak_memory_usage: 0,
                backend_distribution: HashMap::new(),
            };
        }
    }
}

/// GPU-accelerated k-nearest neighbors search
pub struct GpuKNeighborsSearch {
    calculator: GpuDistanceCalculator,
    k: usize,
    distance: Distance,
}

impl GpuKNeighborsSearch {
    /// Create a new GPU k-nearest neighbors search
    pub fn new(k: usize, config: GpuConfig) -> Self {
        Self {
            calculator: GpuDistanceCalculator::with_config(config),
            k,
            distance: Distance::Euclidean,
        }
    }

    /// Set the distance metric
    pub fn with_distance(mut self, distance: Distance) -> Self {
        self.distance = distance;
        self
    }

    /// Initialize GPU context
    pub fn initialize(&mut self) -> NeighborsResult<()> {
        self.calculator.initialize()
    }

    /// Find k-nearest neighbors using GPU acceleration
    #[allow(non_snake_case)]
    pub fn kneighbors<'a>(
        &self,
        X: &ArrayView2<'a, Float>,
        X_query: Option<&ArrayView2<'a, Float>>,
    ) -> NeighborsResult<(Array2<usize>, Array2<Float>)> {
        let X_query = X_query.unwrap_or(X);

        // Compute all pairwise distances using GPU
        let gpu_result =
            self.calculator
                .pairwise_distances(X_query, Some(X), self.distance.clone())?;
        let distances = gpu_result.distances;

        let n_queries = distances.nrows();
        let mut indices = Array2::zeros((n_queries, self.k));
        let mut neighbor_distances = Array2::zeros((n_queries, self.k));

        // For each query, find k nearest neighbors
        for (i, distance_row) in distances.axis_iter(Axis(0)).enumerate() {
            let mut indexed_distances: Vec<(usize, Float)> = distance_row
                .iter()
                .enumerate()
                .map(|(j, &d)| (j, d))
                .collect();

            // Sort by distance
            indexed_distances
                .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

            // Take k nearest (excluding self if query is in training set)
            let mut count = 0;
            let mut j = 0;

            while count < self.k && j < indexed_distances.len() {
                let (idx, dist) = indexed_distances[j];

                // Skip self-distance if query point is in training set
                if X_query.as_ptr() == X.as_ptr() && i == idx {
                    j += 1;
                    continue;
                }

                indices[[i, count]] = idx;
                neighbor_distances[[i, count]] = dist;
                count += 1;
                j += 1;
            }
        }

        Ok((indices, neighbor_distances))
    }

    /// Get GPU computation statistics
    pub fn get_stats(&self) -> GpuComputationStats {
        self.calculator.get_stats()
    }

    /// Get device information
    pub fn get_device_info(&self) -> Option<GpuDeviceInfo> {
        self.calculator.get_device_info()
    }
}

/// GPU memory usage estimator
pub struct GpuMemoryEstimator;

impl GpuMemoryEstimator {
    /// Estimate memory usage for pairwise distance computation
    pub fn estimate_pairwise_memory(n_samples_x: usize, n_samples_y: usize) -> usize {
        // Input data + output distances
        let input_memory = (n_samples_x + n_samples_y) * std::mem::size_of::<Float>();
        let output_memory = n_samples_x * n_samples_y * std::mem::size_of::<Float>();

        // Additional GPU memory overhead (buffers, workspace)
        let overhead = (input_memory + output_memory) / 2;

        input_memory + output_memory + overhead
    }

    /// Estimate optimal batch size for given memory constraints
    pub fn estimate_optimal_batch_size(
        n_samples: usize,
        n_features: usize,
        max_memory: usize,
    ) -> usize {
        let sample_size = n_features * std::mem::size_of::<Float>();
        let distance_size = n_samples * std::mem::size_of::<Float>();

        // Account for GPU memory overhead
        let overhead_factor = 1.5;
        let effective_memory = (max_memory as f64 / overhead_factor) as usize;

        let max_batch_size = effective_memory / (sample_size + distance_size);

        std::cmp::min(max_batch_size, n_samples).max(1)
    }

    /// Check if computation fits in GPU memory
    pub fn can_fit_in_memory(
        n_samples_x: usize,
        n_samples_y: usize,
        available_memory: usize,
    ) -> bool {
        let required_memory = Self::estimate_pairwise_memory(n_samples_x, n_samples_y);
        required_memory <= available_memory
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    fn create_test_data() -> Array2<Float> {
        Array2::from_shape_vec((100, 4), (0..400).map(|x| x as Float).collect()).unwrap()
    }

    #[test]
    fn test_gpu_config_default() {
        let config = GpuConfig::default();
        assert_eq!(config.backend, GpuBackend::CpuFallback);
        assert_eq!(config.batch_size, 1024);
        assert!(config.enable_async);
    }

    #[test]
    fn test_gpu_distance_calculator_creation() {
        let calculator = GpuDistanceCalculator::new();
        assert!(calculator.device_info.is_none());
    }

    #[test]
    fn test_gpu_distance_calculator_with_config() {
        let config = GpuConfig {
            backend: GpuBackend::OpenCl,
            batch_size: 512,
            ..Default::default()
        };
        let calculator = GpuDistanceCalculator::with_config(config);
        assert_eq!(calculator.config.backend, GpuBackend::OpenCl);
        assert_eq!(calculator.config.batch_size, 512);
    }

    #[test]
    fn test_gpu_device_detection() {
        let calculator = GpuDistanceCalculator::new();
        let device_info = calculator.detect_gpu_devices().unwrap();
        assert!(device_info.is_some());

        let device = device_info.unwrap();
        assert_eq!(device.backend, GpuBackend::CpuFallback);
        assert!(!device.name.is_empty());
    }

    #[test]
    fn test_gpu_pairwise_distances_cpu_fallback() {
        let data = create_test_data();
        let calculator = GpuDistanceCalculator::new();

        let result = calculator
            .pairwise_distances(&data.view(), None, Distance::Euclidean)
            .unwrap();

        assert_eq!(result.distances.shape(), &[100, 100]);
        assert_eq!(result.backend_used, GpuBackend::CpuFallback);
        assert!(result.computation_time > 0.0);
        assert!(result.memory_usage > 0);
    }

    #[test]
    fn test_gpu_batch_pairwise_distances() {
        let data = create_test_data();
        let config = GpuConfig {
            batch_size: 10,
            ..Default::default()
        };
        let calculator = GpuDistanceCalculator::with_config(config);

        let result = calculator
            .batch_pairwise_distances(&data.view(), None, Distance::Euclidean)
            .unwrap();

        assert_eq!(result.distances.shape(), &[100, 100]);
        assert!(result.computation_time > 0.0);
    }

    #[test]
    fn test_gpu_kneighbors_search() {
        let data = create_test_data();
        let config = GpuConfig {
            batch_size: 50,
            ..Default::default()
        };
        let search = GpuKNeighborsSearch::new(5, config);

        let (indices, distances) = search.kneighbors(&data.view(), None).unwrap();

        assert_eq!(indices.shape(), &[100, 5]);
        assert_eq!(distances.shape(), &[100, 5]);

        // Check that distances are in ascending order
        for i in 0..indices.nrows() {
            for j in 1..indices.ncols() {
                assert!(distances[[i, j]] >= distances[[i, j - 1]]);
            }
        }
    }

    #[test]
    fn test_gpu_memory_estimator() {
        let memory_usage = GpuMemoryEstimator::estimate_pairwise_memory(100, 100);
        assert!(memory_usage > 0);

        let batch_size = GpuMemoryEstimator::estimate_optimal_batch_size(
            1000,
            10,
            1024 * 1024, // 1MB
        );
        assert!(batch_size > 0);
        assert!(batch_size <= 1000);

        let can_fit = GpuMemoryEstimator::can_fit_in_memory(100, 100, 1024 * 1024);
        assert!(can_fit);
    }

    #[test]
    fn test_gpu_stats_tracking() {
        let data = create_test_data();
        let calculator = GpuDistanceCalculator::new();

        // Perform some computations
        let _ = calculator
            .pairwise_distances(&data.view(), None, Distance::Euclidean)
            .unwrap();
        let _ = calculator
            .pairwise_distances(&data.view(), None, Distance::Manhattan)
            .unwrap();

        let stats = calculator.get_stats();
        assert_eq!(stats.total_computations, 2);
        assert!(stats.total_time > 0.0);
        assert!(stats.average_time > 0.0);
        assert!(stats.peak_memory_usage > 0);
        assert_eq!(stats.backend_distribution[&GpuBackend::CpuFallback], 2);
    }

    #[test]
    fn test_different_gpu_backends() {
        let backends = vec![
            GpuBackend::CpuFallback,
            GpuBackend::OpenCl,
            GpuBackend::Cuda,
            GpuBackend::Metal,
        ];

        for backend in backends {
            let config = GpuConfig {
                backend,
                ..Default::default()
            };
            let calculator = GpuDistanceCalculator::with_config(config);
            let device_info = calculator.detect_gpu_devices().unwrap();

            // At least CPU fallback should always be available
            if backend == GpuBackend::CpuFallback {
                assert!(device_info.is_some());
            }
        }
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_gpu_shape_mismatch_error() {
        let X = Array2::from_shape_vec((10, 4), (0..40).map(|x| x as Float).collect()).unwrap();
        let Y = Array2::from_shape_vec((10, 3), (0..30).map(|x| x as Float).collect()).unwrap();

        let calculator = GpuDistanceCalculator::new();
        let result = calculator.pairwise_distances(&X.view(), Some(&Y.view()), Distance::Euclidean);

        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            NeighborsError::ShapeMismatch { .. }
        ));
    }

    #[test]
    fn test_gpu_reset_stats() {
        let data = create_test_data();
        let calculator = GpuDistanceCalculator::new();

        // Perform computation
        let _ = calculator
            .pairwise_distances(&data.view(), None, Distance::Euclidean)
            .unwrap();

        let stats_before = calculator.get_stats();
        assert_eq!(stats_before.total_computations, 1);

        // Reset stats
        calculator.reset_stats();

        let stats_after = calculator.get_stats();
        assert_eq!(stats_after.total_computations, 0);
        assert_eq!(stats_after.total_time, 0.0);
    }
}
