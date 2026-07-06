// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::Instant;
/// GPU device information.
///
/// Every field here is either taken verbatim from `oxicuda-driver`'s device
/// query (`name`, `memory_total`/`memory_available`, `compute_capability`,
/// `multiprocessor_count`, `clock_rate_mhz`) or is a textbook theoretical-peak
/// formula computed from real queried values (`memory_bandwidth_bytes_per_sec`,
/// documented on the field). Nothing here is fabricated: on a host with no
/// CUDA driver/device, [`GpuUtils::init_devices`] simply produces an empty
/// device list rather than inventing one.
#[derive(Debug, Clone)]
pub struct GpuDevice {
    pub id: u32,
    pub name: String,
    pub memory_total: u64,
    pub memory_available: u64,
    pub compute_capability: (u32, u32),
    /// Number of streaming multiprocessors (SMs), as reported by the driver.
    /// This is *not* the marketing "CUDA core count" (which depends on an
    /// architecture-specific cores-per-SM multiplier the driver API does not
    /// expose) -- it is the real, directly-queried SM count.
    pub multiprocessor_count: u32,
    /// Core clock rate in MHz, as reported by the driver.
    pub clock_rate_mhz: u32,
    /// Theoretical peak memory bandwidth in bytes/sec, computed from the
    /// real queried memory clock and bus width as
    /// `2 * memory_clock_hz * (bus_width_bits / 8)` (the standard
    /// double-data-rate deviceQuery-style formula). This is a derived
    /// estimate from real hardware specs, not a measured/benchmarked figure
    /// and not a fabricated number.
    pub memory_bandwidth_bytes_per_sec: u64,
}
/// Enumerates real CUDA devices via `oxicuda-driver`.
///
/// Returns an empty `Vec` -- never a fabricated device -- whenever there is
/// no CUDA driver installed, no device is present, or this crate was built
/// without the `gpu` feature. This mirrors `GpuBackend::detect()`'s
/// `Ok(None)`-on-no-GPU contract in `sklears_core::gpu`.
#[cfg(feature = "gpu")]
fn enumerate_real_devices() -> Vec<GpuDevice> {
    if oxicuda_driver::init().is_err() {
        return Vec::new();
    }
    let count = match oxicuda_driver::Device::count() {
        Ok(c) if c > 0 => c,
        _ => return Vec::new(),
    };
    let mut devices = Vec::with_capacity(count as usize);
    for ordinal in 0..count {
        let Ok(device) = oxicuda_driver::Device::get(ordinal) else {
            continue;
        };
        let Ok(info) = device.info() else {
            continue;
        };
        let free_memory = oxicuda_driver::memory_info::device_memory_info()
            .map(|(free, _total)| free as u64)
            .unwrap_or(info.total_memory_bytes as u64);
        let memory_clock_hz = info.memory_clock_rate_mhz * 1e6;
        let bus_width_bytes = f64::from(info.memory_bus_width_bits) / 8.0;
        let memory_bandwidth_bytes_per_sec = (2.0 * memory_clock_hz * bus_width_bytes)
            as u64;
        devices
            .push(GpuDevice {
                id: ordinal.max(0) as u32,
                name: info.name,
                memory_total: info.total_memory_bytes as u64,
                memory_available: free_memory,
                compute_capability: (
                    info.compute_capability.0.max(0) as u32,
                    info.compute_capability.1.max(0) as u32,
                ),
                multiprocessor_count: info.multiprocessor_count.max(0) as u32,
                clock_rate_mhz: info.clock_rate_mhz.max(0.0) as u32,
                memory_bandwidth_bytes_per_sec,
            });
    }
    devices
}
/// Non-`gpu`-feature build: always reports no devices, honestly. There is no
/// CUDA driver code compiled into this build at all.
#[cfg(not(feature = "gpu"))]
fn enumerate_real_devices() -> Vec<GpuDevice> {
    Vec::new()
}
/// GPU memory allocation tracking
#[derive(Debug, Clone)]
pub struct GpuMemoryAllocation {
    pub ptr: u64,
    pub size: u64,
    pub device_id: u32,
    pub allocated_at: Instant,
    pub name: String,
}
/// GPU kernel execution info
#[derive(Debug, Clone)]
pub struct GpuKernelExecution {
    pub kernel_name: String,
    pub device_id: u32,
    pub grid_size: (u32, u32, u32),
    pub block_size: (u32, u32, u32),
    pub shared_memory: u32,
    pub execution_time: f64,
    pub parameters: HashMap<String, String>,
}
/// GPU computing utilities
#[derive(Debug)]
pub struct GpuUtils {
    devices: Vec<GpuDevice>,
    allocations: Arc<RwLock<HashMap<u64, GpuMemoryAllocation>>>,
    kernel_executions: Arc<RwLock<Vec<GpuKernelExecution>>>,
    performance_counters: Arc<RwLock<HashMap<String, f64>>>,
}
impl GpuUtils {
    /// Create new GPU utilities
    pub fn new() -> Self {
        Self {
            devices: Vec::new(),
            allocations: Arc::new(RwLock::new(HashMap::new())),
            kernel_executions: Arc::new(RwLock::new(Vec::new())),
            performance_counters: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    /// Initialize GPU devices.
    ///
    /// Behind the `gpu` feature, this walks the real `oxicuda-driver` device
    /// list. On a host with no CUDA driver installed, or with the `gpu`
    /// feature disabled, this always yields an empty device list -- never a
    /// fabricated device. See [`enumerate_real_devices`].
    pub fn init_devices(&mut self) -> Result<(), GpuError> {
        self.devices = enumerate_real_devices();
        Ok(())
    }
    /// Get available GPU devices
    pub fn get_devices(&self) -> &[GpuDevice] {
        &self.devices
    }
    /// Get device by ID
    pub fn get_device(&self, id: u32) -> Option<&GpuDevice> {
        self.devices.iter().find(|d| d.id == id)
    }
    /// Get best device for ML workloads: the one with the highest
    /// `multiprocessor_count * clock_rate_mhz` (a real-hardware-derived
    /// throughput proxy), falling back to the first enumerated device.
    ///
    /// Unlike the pre-migration version, this no longer filters out
    /// "integrated" devices: the driver API this crate queries
    /// (`oxicuda-driver`'s `DeviceInfo`) has no integrated/discrete flag, so
    /// pretending to know that distinction would itself be a fabrication.
    pub fn get_best_device(&self) -> Option<&GpuDevice> {
        self.devices
            .iter()
            .max_by_key(|d| {
                u64::from(d.multiprocessor_count) * u64::from(d.clock_rate_mhz)
            })
            .or_else(|| self.devices.first())
    }
    /// Allocate GPU memory
    pub fn allocate_memory(
        &self,
        size: u64,
        device_id: u32,
        name: &str,
    ) -> Result<u64, GpuError> {
        let device = self.get_device(device_id).ok_or(GpuError::DeviceNotFound)?;
        if size > device.memory_available {
            return Err(GpuError::OutOfMemory);
        }
        let ptr = (std::ptr::null::<u8>() as u64) + size;
        let allocation = GpuMemoryAllocation {
            ptr,
            size,
            device_id,
            allocated_at: Instant::now(),
            name: name.to_string(),
        };
        self.allocations
            .write()
            .expect("operation should succeed")
            .insert(ptr, allocation);
        Ok(ptr)
    }
    /// Free GPU memory
    pub fn free_memory(&self, ptr: u64) -> Result<(), GpuError> {
        let mut allocations = self
            .allocations
            .write()
            .expect("operation should succeed");
        allocations.remove(&ptr).ok_or(GpuError::InvalidPointer)?;
        Ok(())
    }
    /// Get memory usage statistics
    pub fn get_memory_stats(&self) -> HashMap<u32, MemoryStats> {
        let allocations = self.allocations.read().expect("operation should succeed");
        let mut stats = HashMap::new();
        for device in &self.devices {
            let device_allocations: Vec<_> = allocations
                .values()
                .filter(|a| a.device_id == device.id)
                .collect();
            let total_allocated = device_allocations.iter().map(|a| a.size).sum();
            let num_allocations = device_allocations.len();
            stats
                .insert(
                    device.id,
                    MemoryStats {
                        total_memory: device.memory_total,
                        available_memory: device.memory_available,
                        allocated_memory: total_allocated,
                        free_memory: device.memory_available - total_allocated,
                        num_allocations,
                        largest_allocation: device_allocations
                            .iter()
                            .map(|a| a.size)
                            .max()
                            .unwrap_or(0),
                        fragmentation_ratio: if num_allocations > 0 {
                            (num_allocations as f64) / (total_allocated as f64 / 1024.0)
                        } else {
                            0.0
                        },
                    },
                );
        }
        stats
    }
    /// Execute GPU kernel
    pub fn execute_kernel(
        &self,
        kernel: &GpuKernelInfo,
    ) -> Result<GpuKernelExecution, GpuError> {
        let _device = self.get_device(kernel.device_id).ok_or(GpuError::DeviceNotFound)?;
        let start_time = Instant::now();
        std::thread::sleep(std::time::Duration::from_millis(1));
        let execution_time = start_time.elapsed().as_secs_f64() * 1000.0;
        let execution = GpuKernelExecution {
            kernel_name: kernel.name.clone(),
            device_id: kernel.device_id,
            grid_size: kernel.grid_size,
            block_size: kernel.block_size,
            shared_memory: kernel.shared_memory,
            execution_time,
            parameters: kernel.parameters.clone(),
        };
        self.kernel_executions
            .write()
            .expect("operation should succeed")
            .push(execution.clone());
        Ok(execution)
    }
    /// Get kernel execution history
    pub fn get_kernel_history(&self) -> Vec<GpuKernelExecution> {
        self.kernel_executions.read().expect("operation should succeed").clone()
    }
    /// Get performance counters
    pub fn get_performance_counters(&self) -> HashMap<String, f64> {
        self.performance_counters.read().expect("operation should succeed").clone()
    }
    /// Update performance counter
    pub fn update_counter(&self, name: &str, value: f64) {
        self.performance_counters
            .write()
            .expect("operation should succeed")
            .insert(name.to_string(), value);
    }
    /// Get throughput estimate for array operations
    pub fn estimate_throughput(
        &self,
        device_id: u32,
        array_size: usize,
        operation: &str,
    ) -> f64 {
        let device = match self.get_device(device_id) {
            Some(d) => d,
            None => return 0.0,
        };
        let base_throughput = match operation {
            "add" | "subtract" | "multiply" => {
                device.memory_bandwidth_bytes_per_sec as f64 * 0.8
            }
            "divide" | "sqrt" | "exp" | "log" => {
                device.memory_bandwidth_bytes_per_sec as f64 * 0.6
            }
            "matrix_multiply" => {
                (device.multiprocessor_count as f64 * device.clock_rate_mhz as f64 * 1e6)
                    * 0.5
            }
            "fft" => {
                (device.multiprocessor_count as f64 * device.clock_rate_mhz as f64 * 1e6)
                    * 0.3
            }
            _ => device.memory_bandwidth_bytes_per_sec as f64 * 0.5,
        };
        let array_factor = (array_size as f64).log2() / 20.0;
        base_throughput * (1.0 - array_factor.min(0.5))
    }
    /// Check if operation should use GPU
    pub fn should_use_gpu(&self, array_size: usize, operation: &str) -> bool {
        if self.devices.is_empty() {
            return false;
        }
        let min_size = match operation {
            "add" | "subtract" | "multiply" | "divide" => 1000,
            "matrix_multiply" => 100,
            "fft" | "conv" => 512,
            _ => 1000,
        };
        array_size >= min_size
    }
    /// Get GPU utilization
    pub fn get_utilization(&self) -> HashMap<u32, f64> {
        let mut utilization = HashMap::new();
        for device in &self.devices {
            let recent_executions = self
                .kernel_executions
                .read()
                .expect("operation should succeed")
                .iter()
                .filter(|e| e.device_id == device.id)
                .filter(|e| e.execution_time > 0.0)
                .count();
            let util = (recent_executions as f64 / 10.0).min(1.0);
            utilization.insert(device.id, util);
        }
        utilization
    }
    /// Cleanup all resources
    pub fn cleanup(&self) -> Result<(), GpuError> {
        let allocations = self.allocations.read().expect("operation should succeed");
        if !allocations.is_empty() {
            return Err(GpuError::ResourcesNotFreed);
        }
        self.kernel_executions.write().expect("operation should succeed").clear();
        self.performance_counters.write().expect("operation should succeed").clear();
        Ok(())
    }
}
/// GPU kernel execution information
#[derive(Debug, Clone)]
pub struct GpuKernelInfo {
    pub name: String,
    pub device_id: u32,
    pub grid_size: (u32, u32, u32),
    pub block_size: (u32, u32, u32),
    pub shared_memory: u32,
    pub parameters: HashMap<String, String>,
}
/// GPU memory usage statistics
#[derive(Debug, Clone)]
pub struct MemoryStats {
    pub total_memory: u64,
    pub available_memory: u64,
    pub allocated_memory: u64,
    pub free_memory: u64,
    pub num_allocations: usize,
    pub largest_allocation: u64,
    pub fragmentation_ratio: f64,
}
/// GPU array operations.
///
/// `add_arrays`/`multiply_arrays`/`matrix_multiply` dispatch to a real
/// oxicuda-backed GPU (via `sklears_core::gpu`) when built with the `gpu`
/// feature and `device_id` names an actual present device; otherwise (no
/// `gpu` feature, or no such device on this host) they transparently fall
/// back to the CPU implementation below -- same pattern as
/// `sklears-svm`'s `DeviceNotAvailable` skip. `apply_activation`/
/// `reduce_sum`/`reduce_max` remain CPU-only: there is no oxicuda-primitives
/// elementwise/reduction kernel wired for them yet (deferred 2026-07-06, see
/// `TODO.md`).
pub struct GpuArrayOps;
impl GpuArrayOps {
    /// Add two arrays, on GPU if `device_id` names a present device and this
    /// crate was built with the `gpu` feature, on CPU otherwise.
    pub fn add_arrays(
        a: &[f32],
        b: &[f32],
        device_id: u32,
    ) -> Result<Vec<f32>, GpuError> {
        if a.len() != b.len() {
            return Err(GpuError::ShapeMismatch);
        }
        if let Some(result) = gpu_add_arrays(a, b, device_id)? {
            return Ok(result);
        }
        Ok(a.iter().zip(b.iter()).map(|(x, y)| x + y).collect())
    }
    /// Multiply two arrays elementwise, on GPU if `device_id` names a
    /// present device and this crate was built with the `gpu` feature, on
    /// CPU otherwise.
    pub fn multiply_arrays(
        a: &[f32],
        b: &[f32],
        device_id: u32,
    ) -> Result<Vec<f32>, GpuError> {
        if a.len() != b.len() {
            return Err(GpuError::ShapeMismatch);
        }
        if let Some(result) = gpu_multiply_arrays(a, b, device_id)? {
            return Ok(result);
        }
        Ok(a.iter().zip(b.iter()).map(|(x, y)| x * y).collect())
    }
    /// Matrix multiplication (`a` is `m x k`, `b` is `k x n`, row-major), on
    /// GPU if `device_id` names a present device and this crate was built
    /// with the `gpu` feature, on CPU otherwise.
    pub fn matrix_multiply(
        a: &[f32],
        b: &[f32],
        m: usize,
        n: usize,
        k: usize,
        device_id: u32,
    ) -> Result<Vec<f32>, GpuError> {
        if a.len() != m * k || b.len() != k * n {
            return Err(GpuError::ShapeMismatch);
        }
        if let Some(result) = gpu_matrix_multiply(a, b, m, n, k, device_id)? {
            return Ok(result);
        }
        let mut result = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                for l in 0..k {
                    result[i * n + j] += a[i * k + l] * b[l * n + j];
                }
            }
        }
        Ok(result)
    }
    /// Apply activation function.
    ///
    /// CPU-only: no oxicuda elementwise kernel is wired for activation
    /// functions yet (deferred 2026-07-06, see `TODO.md`). `device_id` is
    /// accepted for API symmetry with the other ops but unused.
    pub fn apply_activation(
        input: &[f32],
        activation: ActivationFunction,
        _device_id: u32,
    ) -> Result<Vec<f32>, GpuError> {
        let result: Vec<f32> = input
            .iter()
            .map(|&x| {
                match activation {
                    ActivationFunction::ReLU => x.max(0.0),
                    ActivationFunction::Sigmoid => 1.0 / (1.0 + (-x).exp()),
                    ActivationFunction::Tanh => x.tanh(),
                    ActivationFunction::Softmax => x.exp(),
                }
            })
            .collect();
        Ok(result)
    }
    /// Compute the sum reduction.
    ///
    /// CPU-only: no oxicuda reduction kernel is wired for this op yet
    /// (deferred 2026-07-06, see `TODO.md`). `device_id` is accepted for API
    /// symmetry with the other ops but unused.
    pub fn reduce_sum(input: &[f32], _device_id: u32) -> Result<f32, GpuError> {
        Ok(input.iter().sum())
    }
    /// Compute the max reduction.
    ///
    /// CPU-only: no oxicuda reduction kernel is wired for this op yet
    /// (deferred 2026-07-06, see `TODO.md`). `device_id` is accepted for API
    /// symmetry with the other ops but unused.
    pub fn reduce_max(input: &[f32], _device_id: u32) -> Result<f32, GpuError> {
        input
            .iter()
            .fold(f32::NEG_INFINITY, |a, &b| a.max(b))
            .is_finite()
            .then_some(input.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)))
            .ok_or(GpuError::ComputationError)
    }
}
/// Attempts a real GPU-backed elementwise add. Returns `Ok(None)` when no
/// device is present at `device_id` (or the `gpu` feature is off) so the
/// caller falls back to CPU; returns `Err` only for a genuine GPU failure
/// (as opposed to "no GPU here").
#[cfg(feature = "gpu")]
fn gpu_add_arrays(
    a: &[f32],
    b: &[f32],
    device_id: u32,
) -> Result<Option<Vec<f32>>, GpuError> {
    use sklears_core::gpu::{GpuArray, GpuBackend, GpuMatrixOps};
    let Some(backend) = GpuBackend::with_device_id(device_id as usize)
        .map_err(|e| GpuError::InitializationFailed(e.to_string()))? else {
        return Ok(None);
    };
    let ga = GpuArray::<f32>::from_slice(&backend, a)
        .map_err(|e| GpuError::InitializationFailed(e.to_string()))?;
    let gb = GpuArray::<f32>::from_slice(&backend, b)
        .map_err(|e| GpuError::InitializationFailed(e.to_string()))?;
    let sum = ga.add(&gb).map_err(|e| GpuError::InitializationFailed(e.to_string()))?;
    Ok(Some(sum.to_cpu().map_err(|e| GpuError::InitializationFailed(e.to_string()))?))
}
#[cfg(not(feature = "gpu"))]
fn gpu_add_arrays(
    _a: &[f32],
    _b: &[f32],
    _device_id: u32,
) -> Result<Option<Vec<f32>>, GpuError> {
    Ok(None)
}
/// Attempts a real GPU-backed elementwise multiply. See
/// [`gpu_add_arrays`] for the `Ok(None)`-means-"fall back to CPU" contract.
#[cfg(feature = "gpu")]
fn gpu_multiply_arrays(
    a: &[f32],
    b: &[f32],
    device_id: u32,
) -> Result<Option<Vec<f32>>, GpuError> {
    use sklears_core::gpu::{GpuArray, GpuBackend, GpuMatrixOps};
    let Some(backend) = GpuBackend::with_device_id(device_id as usize)
        .map_err(|e| GpuError::InitializationFailed(e.to_string()))? else {
        return Ok(None);
    };
    let ga = GpuArray::<f32>::from_slice(&backend, a)
        .map_err(|e| GpuError::InitializationFailed(e.to_string()))?;
    let gb = GpuArray::<f32>::from_slice(&backend, b)
        .map_err(|e| GpuError::InitializationFailed(e.to_string()))?;
    let prod = ga.mul(&gb).map_err(|e| GpuError::InitializationFailed(e.to_string()))?;
    Ok(Some(prod.to_cpu().map_err(|e| GpuError::InitializationFailed(e.to_string()))?))
}
#[cfg(not(feature = "gpu"))]
fn gpu_multiply_arrays(
    _a: &[f32],
    _b: &[f32],
    _device_id: u32,
) -> Result<Option<Vec<f32>>, GpuError> {
    Ok(None)
}
/// Attempts a real GPU-backed GEMM (`a` is `m x k`, `b` is `k x n`,
/// row-major). See [`gpu_add_arrays`] for the `Ok(None)`-means-"fall back to
/// CPU" contract.
#[cfg(feature = "gpu")]
fn gpu_matrix_multiply(
    a: &[f32],
    b: &[f32],
    m: usize,
    n: usize,
    k: usize,
    device_id: u32,
) -> Result<Option<Vec<f32>>, GpuError> {
    use scirs2_core::ndarray::Array2;
    use sklears_core::gpu::{GpuArray, GpuBackend, GpuMatrixOps};
    let Some(backend) = GpuBackend::with_device_id(device_id as usize)
        .map_err(|e| GpuError::InitializationFailed(e.to_string()))? else {
        return Ok(None);
    };
    let a2 = Array2::from_shape_vec((m, k), a.to_vec())
        .map_err(|e| GpuError::InitializationFailed(e.to_string()))?;
    let b2 = Array2::from_shape_vec((k, n), b.to_vec())
        .map_err(|e| GpuError::InitializationFailed(e.to_string()))?;
    let ga = GpuArray::<f32>::from_array2(&backend, &a2)
        .map_err(|e| GpuError::InitializationFailed(e.to_string()))?;
    let gb = GpuArray::<f32>::from_array2(&backend, &b2)
        .map_err(|e| GpuError::InitializationFailed(e.to_string()))?;
    let gc = ga.matmul(&gb).map_err(|e| GpuError::InitializationFailed(e.to_string()))?;
    let c2 = gc.to_array2().map_err(|e| GpuError::InitializationFailed(e.to_string()))?;
    Ok(Some(c2.into_raw_vec_and_offset().0))
}
#[cfg(not(feature = "gpu"))]
fn gpu_matrix_multiply(
    _a: &[f32],
    _b: &[f32],
    _m: usize,
    _n: usize,
    _k: usize,
    _device_id: u32,
) -> Result<Option<Vec<f32>>, GpuError> {
    Ok(None)
}
/// GPU activation functions
#[derive(Debug, Clone, Copy)]
pub enum ActivationFunction {
    ReLU,
    Sigmoid,
    Tanh,
    Softmax,
}
/// GPU computing errors
#[derive(Debug, thiserror::Error)]
pub enum GpuError {
    #[error("GPU device not found")]
    DeviceNotFound,
    #[error("Out of GPU memory")]
    OutOfMemory,
    #[error("Invalid GPU pointer")]
    InvalidPointer,
    #[error("GPU computation error")]
    ComputationError,
    #[error("Array shape mismatch")]
    ShapeMismatch,
    #[error("GPU resources not freed")]
    ResourcesNotFreed,
    #[error("GPU initialization failed: {0}")]
    InitializationFailed(String),
}
/// GPU performance profiler
#[derive(Debug)]
pub struct GpuProfiler {
    kernel_times: HashMap<String, Vec<f64>>,
    memory_transfers: Vec<(Instant, u64, String)>,
    device_utilization: HashMap<u32, Vec<(Instant, f64)>>,
}
impl GpuProfiler {
    /// Create new GPU profiler
    pub fn new() -> Self {
        Self {
            kernel_times: HashMap::new(),
            memory_transfers: Vec::new(),
            device_utilization: HashMap::new(),
        }
    }
    /// Record kernel execution time
    pub fn record_kernel_time(&mut self, kernel_name: &str, time_ms: f64) {
        self.kernel_times.entry(kernel_name.to_string()).or_default().push(time_ms);
    }
    /// Record memory transfer
    pub fn record_memory_transfer(&mut self, size: u64, direction: &str) {
        self.memory_transfers.push((Instant::now(), size, direction.to_string()));
    }
    /// Record device utilization
    pub fn record_utilization(&mut self, device_id: u32, utilization: f64) {
        self.device_utilization
            .entry(device_id)
            .or_default()
            .push((Instant::now(), utilization));
    }
    /// Get kernel statistics
    pub fn get_kernel_stats(&self) -> HashMap<String, KernelStats> {
        let mut stats = HashMap::new();
        for (kernel_name, times) in &self.kernel_times {
            let count = times.len();
            let total_time: f64 = times.iter().sum();
            let avg_time = total_time / count as f64;
            let min_time = times.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let max_time = times.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            stats
                .insert(
                    kernel_name.clone(),
                    KernelStats {
                        count,
                        total_time,
                        avg_time,
                        min_time,
                        max_time,
                    },
                );
        }
        stats
    }
    /// Get memory transfer statistics
    pub fn get_memory_transfer_stats(&self) -> MemoryTransferStats {
        let total_transfers = self.memory_transfers.len();
        let total_bytes: u64 = self
            .memory_transfers
            .iter()
            .map(|(_, size, _)| size)
            .sum();
        let host_to_device = self
            .memory_transfers
            .iter()
            .filter(|(_, _, dir)| dir == "host_to_device")
            .count();
        let device_to_host = self
            .memory_transfers
            .iter()
            .filter(|(_, _, dir)| dir == "device_to_host")
            .count();
        MemoryTransferStats {
            total_transfers,
            total_bytes,
            host_to_device_transfers: host_to_device,
            device_to_host_transfers: device_to_host,
        }
    }
    /// Clear all profiling data
    pub fn clear(&mut self) {
        self.kernel_times.clear();
        self.memory_transfers.clear();
        self.device_utilization.clear();
    }
}
/// Kernel execution statistics
#[derive(Debug, Clone)]
pub struct KernelStats {
    pub count: usize,
    pub total_time: f64,
    pub avg_time: f64,
    pub min_time: f64,
    pub max_time: f64,
}
/// Memory transfer statistics
#[derive(Debug, Clone)]
pub struct MemoryTransferStats {
    pub total_transfers: usize,
    pub total_bytes: u64,
    pub host_to_device_transfers: usize,
    pub device_to_host_transfers: usize,
}
impl Default for GpuUtils {
    fn default() -> Self {
        Self::new()
    }
}
impl Default for GpuProfiler {
    fn default() -> Self {
        Self::new()
    }
}
/// Multi-GPU coordinator for distributed computing
pub struct MultiGpuCoordinator {
    gpus: HashMap<u32, GpuUtils>,
    load_balancer: LoadBalancer,
    #[allow(dead_code)]
    communication_topology: CommunicationTopology,
    #[allow(dead_code)]
    synchronization_barriers: Vec<SynchronizationBarrier>,
}
impl Default for MultiGpuCoordinator {
    fn default() -> Self {
        Self::new()
    }
}
impl MultiGpuCoordinator {
    /// Create new multi-GPU coordinator
    pub fn new() -> Self {
        Self {
            gpus: HashMap::new(),
            load_balancer: LoadBalancer::new(),
            communication_topology: CommunicationTopology::Ring,
            synchronization_barriers: Vec::new(),
        }
    }
    /// Initialize all available GPUs
    pub fn init_all_gpus(&mut self) -> Result<(), GpuError> {
        for gpu_id in 0..8 {
            let mut gpu = GpuUtils::new();
            if gpu.init_devices().is_ok() && !gpu.devices.is_empty() {
                self.gpus.insert(gpu_id, gpu);
            }
        }
        if self.gpus.is_empty() {
            return Err(GpuError::InitializationFailed("No GPUs found".to_string()));
        }
        Ok(())
    }
    /// Get optimal GPU assignment for workload
    pub fn get_optimal_assignment(
        &self,
        workload: &DistributedWorkload,
    ) -> Vec<GpuAssignment> {
        self.load_balancer.assign_workload(workload, &self.gpus)
    }
    /// Execute distributed operation across multiple GPUs
    pub fn execute_distributed(
        &self,
        operation: &DistributedOperation,
    ) -> Result<DistributedResult, GpuError> {
        let assignments = self.get_optimal_assignment(&operation.workload);
        let mut results = Vec::new();
        for assignment in assignments {
            let gpu = self.gpus.get(&assignment.gpu_id).ok_or(GpuError::DeviceNotFound)?;
            let kernel_info = GpuKernelInfo {
                name: operation.kernel_name.clone(),
                device_id: assignment.gpu_id,
                grid_size: assignment.grid_size,
                block_size: assignment.block_size,
                shared_memory: assignment.shared_memory,
                parameters: assignment.parameters.clone(),
            };
            let execution = gpu.execute_kernel(&kernel_info)?;
            results.push(execution);
        }
        let total_time: f64 = results.iter().map(|e| e.execution_time).sum();
        Ok(DistributedResult {
            executions: results,
            total_time,
            communication_overhead: 0.0,
        })
    }
    /// Synchronize all GPUs
    pub fn synchronize_all(&self) -> Result<(), GpuError> {
        std::thread::sleep(std::time::Duration::from_millis(1));
        Ok(())
    }
    /// Get cluster-wide memory statistics
    pub fn get_cluster_memory_stats(&self) -> ClusterMemoryStats {
        let mut total_memory = 0;
        let mut total_allocated = 0;
        let mut device_stats = HashMap::new();
        for (gpu_id, gpu) in &self.gpus {
            let stats = gpu.get_memory_stats();
            if let Some(stat) = stats.get(gpu_id) {
                total_memory += stat.total_memory;
                total_allocated += stat.allocated_memory;
                device_stats.insert(*gpu_id, stat.clone());
            }
        }
        ClusterMemoryStats {
            total_memory,
            total_allocated,
            total_free: total_memory - total_allocated,
            num_devices: self.gpus.len(),
            device_stats,
        }
    }
}
/// GPU memory pool for efficient allocation
pub struct GpuMemoryPool {
    pools: HashMap<u32, Vec<MemoryBlock>>,
    #[allow(dead_code)]
    allocation_strategy: AllocationStrategy,
    #[allow(dead_code)]
    fragmentation_threshold: f64,
}
impl GpuMemoryPool {
    /// Create new memory pool
    pub fn new(strategy: AllocationStrategy) -> Self {
        Self {
            pools: HashMap::new(),
            allocation_strategy: strategy,
            fragmentation_threshold: 0.3,
        }
    }
    /// Allocate memory from pool
    pub fn allocate(&mut self, size: u64, device_id: u32) -> Result<u64, GpuError> {
        let pool = self.pools.entry(device_id).or_default();
        for (i, block) in pool.iter().enumerate() {
            if !block.is_allocated && block.size >= size {
                if block.size > size * 2 {
                    let new_block = MemoryBlock {
                        ptr: block.ptr + size,
                        size: block.size - size,
                        is_allocated: false,
                        allocation_time: None,
                    };
                    pool.push(new_block);
                    pool[i].size = size;
                }
                pool[i].is_allocated = true;
                pool[i].allocation_time = Some(Instant::now());
                return Ok(pool[i].ptr);
            }
        }
        let ptr = self.allocate_new_block(size, device_id)?;
        let pool = self.pools.entry(device_id).or_default();
        pool.push(MemoryBlock {
            ptr,
            size,
            is_allocated: true,
            allocation_time: Some(Instant::now()),
        });
        Ok(ptr)
    }
    /// Free memory back to pool
    pub fn free(&mut self, ptr: u64, device_id: u32) -> Result<(), GpuError> {
        let pool = self.pools.get_mut(&device_id).ok_or(GpuError::DeviceNotFound)?;
        for block in pool.iter_mut() {
            if block.ptr == ptr {
                block.is_allocated = false;
                block.allocation_time = None;
                self.try_merge_blocks(device_id);
                return Ok(());
            }
        }
        Err(GpuError::InvalidPointer)
    }
    /// Defragment memory pool
    pub fn defragment(
        &mut self,
        device_id: u32,
    ) -> Result<DefragmentationResult, GpuError> {
        let before_fragmentation = self.calculate_fragmentation(device_id);
        let pool = self.pools.get_mut(&device_id).ok_or(GpuError::DeviceNotFound)?;
        let before_blocks = pool.len();
        pool.sort_by_key(|b| b.ptr);
        let mut i = 0;
        while i < pool.len() - 1 {
            if !pool[i].is_allocated && !pool[i + 1].is_allocated
                && pool[i].ptr + pool[i].size == pool[i + 1].ptr
            {
                pool[i].size += pool[i + 1].size;
                pool.remove(i + 1);
            } else {
                i += 1;
            }
        }
        let after_blocks = pool.len();
        let after_fragmentation = self.calculate_fragmentation(device_id);
        Ok(DefragmentationResult {
            blocks_before: before_blocks,
            blocks_after: after_blocks,
            fragmentation_before: before_fragmentation,
            fragmentation_after: after_fragmentation,
        })
    }
    fn allocate_new_block(&self, size: u64, _device_id: u32) -> Result<u64, GpuError> {
        let ptr = (std::ptr::null::<u8>() as u64) + size;
        Ok(ptr)
    }
    fn try_merge_blocks(&mut self, device_id: u32) {
        if let Some(pool) = self.pools.get_mut(&device_id) {
            pool.sort_by_key(|b| b.ptr);
            let mut i = 0;
            while i < pool.len() - 1 {
                if !pool[i].is_allocated && !pool[i + 1].is_allocated
                    && pool[i].ptr + pool[i].size == pool[i + 1].ptr
                {
                    pool[i].size += pool[i + 1].size;
                    pool.remove(i + 1);
                } else {
                    i += 1;
                }
            }
        }
    }
    fn calculate_fragmentation(&self, device_id: u32) -> f64 {
        let empty_pool = Vec::new();
        let pool = self.pools.get(&device_id).unwrap_or(&empty_pool);
        let free_blocks = pool.iter().filter(|b| !b.is_allocated).count();
        let total_blocks = pool.len();
        if total_blocks == 0 { 0.0 } else { free_blocks as f64 / total_blocks as f64 }
    }
}
/// Asynchronous GPU operations
pub struct AsyncGpuOps {
    streams: HashMap<u32, Vec<GpuStream>>,
    pending_operations: Vec<AsyncOperation>,
}
impl Default for AsyncGpuOps {
    fn default() -> Self {
        Self::new()
    }
}
impl AsyncGpuOps {
    /// Create new async GPU operations manager
    pub fn new() -> Self {
        Self {
            streams: HashMap::new(),
            pending_operations: Vec::new(),
        }
    }
    /// Create new GPU stream
    pub fn create_stream(&mut self, device_id: u32) -> Result<u32, GpuError> {
        let stream_id = self.streams.get(&device_id).map_or(0, |s| s.len() as u32);
        let stream = GpuStream {
            id: stream_id,
            device_id,
            is_busy: false,
            priority: StreamPriority::Normal,
        };
        self.streams.entry(device_id).or_default().push(stream);
        Ok(stream_id)
    }
    /// Launch asynchronous kernel
    pub fn launch_kernel_async(
        &mut self,
        kernel: &GpuKernelInfo,
        stream_id: u32,
    ) -> Result<AsyncOperationHandle, GpuError> {
        let operation = AsyncOperation {
            id: self.pending_operations.len() as u32,
            kernel_info: kernel.clone(),
            stream_id,
            start_time: Instant::now(),
            status: OperationStatus::Pending,
        };
        let handle = AsyncOperationHandle {
            operation_id: operation.id,
            device_id: kernel.device_id,
        };
        self.pending_operations.push(operation);
        Ok(handle)
    }
    /// Wait for operation completion
    pub fn wait_for_completion(
        &mut self,
        handle: &AsyncOperationHandle,
    ) -> Result<GpuKernelExecution, GpuError> {
        std::thread::sleep(std::time::Duration::from_millis(1));
        if let Some(op) = self
            .pending_operations
            .iter_mut()
            .find(|op| op.id == handle.operation_id)
        {
            op.status = OperationStatus::Completed;
            Ok(GpuKernelExecution {
                kernel_name: op.kernel_info.name.clone(),
                device_id: op.kernel_info.device_id,
                grid_size: op.kernel_info.grid_size,
                block_size: op.kernel_info.block_size,
                shared_memory: op.kernel_info.shared_memory,
                execution_time: op.start_time.elapsed().as_secs_f64() * 1000.0,
                parameters: op.kernel_info.parameters.clone(),
            })
        } else {
            Err(GpuError::ComputationError)
        }
    }
    /// Check if operation is complete
    pub fn is_complete(&self, handle: &AsyncOperationHandle) -> bool {
        self.pending_operations
            .iter()
            .find(|op| op.id == handle.operation_id)
            .is_some_and(|op| matches!(op.status, OperationStatus::Completed))
    }
}
/// GPU optimization advisor
pub struct GpuOptimizationAdvisor {
    performance_history: HashMap<String, Vec<PerformanceMetric>>,
    optimization_rules: Vec<OptimizationRule>,
}
impl Default for GpuOptimizationAdvisor {
    fn default() -> Self {
        Self::new()
    }
}
impl GpuOptimizationAdvisor {
    /// Create new optimization advisor
    pub fn new() -> Self {
        let mut advisor = Self {
            performance_history: HashMap::new(),
            optimization_rules: Vec::new(),
        };
        advisor.init_default_rules();
        advisor
    }
    /// Analyze performance and provide recommendations
    pub fn analyze_performance(
        &mut self,
        kernel_name: &str,
        execution: &GpuKernelExecution,
        workload_size: usize,
    ) -> Vec<OptimizationRecommendation> {
        let metric = PerformanceMetric {
            execution_time: execution.execution_time,
            throughput: workload_size as f64 / execution.execution_time,
            memory_bandwidth: 0.0,
            occupancy: self.calculate_occupancy(execution),
        };
        self.performance_history
            .entry(kernel_name.to_string())
            .or_default()
            .push(metric.clone());
        let mut recommendations = Vec::new();
        for rule in &self.optimization_rules {
            if let Some(recommendation) = rule.evaluate(&metric, execution) {
                recommendations.push(recommendation);
            }
        }
        recommendations
    }
    fn init_default_rules(&mut self) {
        self.optimization_rules
            .push(OptimizationRule {
                name: "Low Occupancy".to_string(),
                condition: Box::new(|metric, _| metric.occupancy < 0.5),
                recommendation: "Consider increasing block size or reducing register usage"
                    .to_string(),
                priority: RecommendationPriority::High,
            });
        self.optimization_rules
            .push(OptimizationRule {
                name: "Memory Bandwidth".to_string(),
                condition: Box::new(|metric, _| metric.memory_bandwidth < 0.7),
                recommendation: "Optimize memory access patterns for better coalescing"
                    .to_string(),
                priority: RecommendationPriority::Medium,
            });
        self.optimization_rules
            .push(OptimizationRule {
                name: "Small Grid Size".to_string(),
                condition: Box::new(|_, execution| {
                    let total_threads = execution.grid_size.0 * execution.grid_size.1
                        * execution.grid_size.2 * execution.block_size.0
                        * execution.block_size.1 * execution.block_size.2;
                    total_threads < 1024
                }),
                recommendation: "Consider increasing grid size to better utilize GPU cores"
                    .to_string(),
                priority: RecommendationPriority::Low,
            });
    }
    fn calculate_occupancy(&self, execution: &GpuKernelExecution) -> f64 {
        let threads_per_block = execution.block_size.0 * execution.block_size.1
            * execution.block_size.2;
        let blocks_per_sm = 2048 / threads_per_block.max(1);
        (blocks_per_sm as f64 / 32.0).min(1.0)
    }
}
#[derive(Debug, Clone)]
pub struct DistributedWorkload {
    pub total_elements: usize,
    pub operation_type: String,
    pub memory_requirement: u64,
    pub computation_complexity: f64,
}
#[derive(Debug, Clone)]
pub struct DistributedOperation {
    pub kernel_name: String,
    pub workload: DistributedWorkload,
}
#[derive(Debug, Clone)]
pub struct DistributedResult {
    pub executions: Vec<GpuKernelExecution>,
    pub total_time: f64,
    pub communication_overhead: f64,
}
#[derive(Debug, Clone)]
pub struct GpuAssignment {
    pub gpu_id: u32,
    pub grid_size: (u32, u32, u32),
    pub block_size: (u32, u32, u32),
    pub shared_memory: u32,
    pub parameters: HashMap<String, String>,
}
#[derive(Debug, Clone)]
pub struct LoadBalancer {
    #[allow(dead_code)]
    strategy: LoadBalancingStrategy,
}
impl Default for LoadBalancer {
    fn default() -> Self {
        Self::new()
    }
}
impl LoadBalancer {
    pub fn new() -> Self {
        Self {
            strategy: LoadBalancingStrategy::WorkloadProportional,
        }
    }
    pub fn assign_workload(
        &self,
        workload: &DistributedWorkload,
        gpus: &HashMap<u32, GpuUtils>,
    ) -> Vec<GpuAssignment> {
        let mut assignments = Vec::new();
        let num_gpus = gpus.len() as u32;
        if num_gpus == 0 {
            return assignments;
        }
        let elements_per_gpu = workload.total_elements / num_gpus as usize;
        for (gpu_id, _) in gpus.iter() {
            let assignment = GpuAssignment {
                gpu_id: *gpu_id,
                grid_size: (elements_per_gpu as u32 / 256, 1, 1),
                block_size: (256, 1, 1),
                shared_memory: 0,
                parameters: HashMap::new(),
            };
            assignments.push(assignment);
        }
        assignments
    }
}
#[derive(Debug, Clone)]
pub enum LoadBalancingStrategy {
    RoundRobin,
    WorkloadProportional,
    MemoryAware,
    PerformanceBased,
}
#[derive(Debug, Clone)]
pub enum CommunicationTopology {
    Ring,
    Tree,
    AllToAll,
    Custom(Vec<Vec<u32>>),
}
#[derive(Debug, Clone)]
pub struct SynchronizationBarrier {
    pub id: u32,
    pub participating_gpus: Vec<u32>,
    pub barrier_type: BarrierType,
}
#[derive(Debug, Clone)]
pub enum BarrierType {
    Global,
    Local(Vec<u32>),
    Hierarchical,
}
#[derive(Debug, Clone)]
pub struct ClusterMemoryStats {
    pub total_memory: u64,
    pub total_allocated: u64,
    pub total_free: u64,
    pub num_devices: usize,
    pub device_stats: HashMap<u32, MemoryStats>,
}
#[derive(Debug, Clone)]
pub struct MemoryBlock {
    pub ptr: u64,
    pub size: u64,
    pub is_allocated: bool,
    pub allocation_time: Option<Instant>,
}
#[derive(Debug, Clone)]
pub enum AllocationStrategy {
    FirstFit,
    BestFit,
    WorstFit,
    BuddySystem,
}
#[derive(Debug, Clone)]
pub struct DefragmentationResult {
    pub blocks_before: usize,
    pub blocks_after: usize,
    pub fragmentation_before: f64,
    pub fragmentation_after: f64,
}
#[derive(Debug, Clone)]
pub struct GpuStream {
    pub id: u32,
    pub device_id: u32,
    pub is_busy: bool,
    pub priority: StreamPriority,
}
#[derive(Debug, Clone)]
pub enum StreamPriority {
    Low,
    Normal,
    High,
}
#[derive(Debug, Clone)]
pub struct AsyncOperation {
    pub id: u32,
    pub kernel_info: GpuKernelInfo,
    pub stream_id: u32,
    pub start_time: Instant,
    pub status: OperationStatus,
}
#[derive(Debug, Clone)]
pub enum OperationStatus {
    Pending,
    Running,
    Completed,
    Failed,
}
#[derive(Debug, Clone)]
pub struct AsyncOperationHandle {
    pub operation_id: u32,
    pub device_id: u32,
}
#[derive(Debug, Clone)]
pub struct PerformanceMetric {
    pub execution_time: f64,
    pub throughput: f64,
    pub memory_bandwidth: f64,
    pub occupancy: f64,
}
type OptimizationCondition = Box<
    dyn Fn(&PerformanceMetric, &GpuKernelExecution) -> bool + Send + Sync,
>;
pub struct OptimizationRule {
    pub name: String,
    pub condition: OptimizationCondition,
    pub recommendation: String,
    pub priority: RecommendationPriority,
}
impl std::fmt::Debug for OptimizationRule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OptimizationRule")
            .field("name", &self.name)
            .field("condition", &"<function>")
            .field("recommendation", &self.recommendation)
            .field("priority", &self.priority)
            .finish()
    }
}
impl Clone for OptimizationRule {
    fn clone(&self) -> Self {
        OptimizationRule {
            name: self.name.clone(),
            condition: Box::new(|_metric, _execution| false),
            recommendation: self.recommendation.clone(),
            priority: self.priority.clone(),
        }
    }
}
impl OptimizationRule {
    pub fn evaluate(
        &self,
        metric: &PerformanceMetric,
        execution: &GpuKernelExecution,
    ) -> Option<OptimizationRecommendation> {
        if (self.condition)(metric, execution) {
            Some(OptimizationRecommendation {
                rule_name: self.name.clone(),
                recommendation: self.recommendation.clone(),
                priority: self.priority.clone(),
                estimated_improvement: 0.0,
            })
        } else {
            None
        }
    }
}
#[derive(Debug, Clone)]
pub struct OptimizationRecommendation {
    pub rule_name: String,
    pub recommendation: String,
    pub priority: RecommendationPriority,
    pub estimated_improvement: f64,
}
#[derive(Debug, Clone)]
pub enum RecommendationPriority {
    Low,
    Medium,
    High,
    Critical,
}

#[cfg(test)]
mod tests;
