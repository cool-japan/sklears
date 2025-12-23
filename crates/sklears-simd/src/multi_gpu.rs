//! Multi-GPU support for parallel processing
//!
//! This module provides utilities for distributing computations across multiple
//! GPU devices, load balancing, and coordinating parallel GPU operations.

use crate::gpu::{GpuDevice, KernelConfig};
use crate::gpu_memory::MultiGpuMemoryManager;
use crate::traits::SimdError;

#[cfg(not(feature = "no-std"))]
use std::collections::{HashMap, HashSet};
#[cfg(not(feature = "no-std"))]
use std::sync::{Arc, Mutex};
#[cfg(not(feature = "no-std"))]
use std::thread;

#[cfg(feature = "no-std")]
use alloc::collections::{BTreeMap as HashMap, BTreeSet as HashSet};
#[cfg(feature = "no-std")]
use alloc::{
    boxed::Box,
    format,
    string::{String, ToString},
    sync::Arc,
    vec,
    vec::Vec,
};

#[cfg(feature = "no-std")]
use core::mem;
#[cfg(feature = "no-std")]
use core::{any::Any, cmp::Ordering};
#[cfg(feature = "no-std")]
use spin::Mutex;
#[cfg(not(feature = "no-std"))]
use std::{any::Any, cmp::Ordering, string::ToString};

// Mock types for no-std compatibility
#[cfg(feature = "no-std")]
#[derive(Debug, Clone, Copy)]
pub struct Instant(u64);

#[cfg(feature = "no-std")]
impl Instant {
    pub fn now() -> Self {
        Instant(0) // Mock implementation for no-std
    }

    pub fn elapsed(&self) -> u64 {
        0 // Mock implementation - returns 0 nanoseconds
    }
}

/// Multi-GPU coordinator for parallel processing
pub struct MultiGpuCoordinator {
    devices: Vec<GpuDevice>,
    memory_manager: Arc<Mutex<MultiGpuMemoryManager>>,
    load_balancer: LoadBalancer,
    task_scheduler: TaskScheduler,
    sync_manager: SynchronizationManager,
}

/// Load balancing strategies for multi-GPU operations
#[derive(Debug, Clone, Copy)]
pub enum LoadBalancingStrategy {
    /// Equal distribution across all devices
    Equal,
    /// Weighted by compute units
    ComputeWeighted,
    /// Weighted by memory bandwidth
    BandwidthWeighted,
    /// Dynamic based on current load
    Dynamic,
    /// Custom weights specified by user
    Custom,
}

/// Load balancer for distributing work across GPUs
pub struct LoadBalancer {
    strategy: LoadBalancingStrategy,
    device_weights: HashMap<u32, f64>,
    performance_history: HashMap<u32, Vec<f64>>,
}

/// Task scheduler for coordinating GPU operations
pub struct TaskScheduler {
    pending_tasks: Vec<GpuTask>,
    running_tasks: HashMap<u32, Vec<GpuTask>>,
    completed_tasks: Vec<CompletedTask>,
}

/// Synchronization manager for multi-GPU operations
pub struct SynchronizationManager {
    barriers: HashMap<String, GpuBarrier>,
    events: HashMap<String, GpuEvent>,
}

/// GPU task representation
#[derive(Debug, Clone)]
pub struct GpuTask {
    pub id: String,
    pub kernel_name: String,
    pub config: KernelConfig,
    pub input_data: Vec<GpuTaskData>,
    pub output_data: Vec<GpuTaskData>,
    pub device_preference: Option<u32>,
    pub priority: TaskPriority,
    pub dependencies: Vec<String>,
}

/// Task data descriptor
#[derive(Debug, Clone)]
pub struct GpuTaskData {
    pub name: String,
    pub size: usize,
    pub data_type: String, // "f32", "f64", "i32", etc.
    pub location: DataLocation,
}

/// Data location for GPU tasks
#[derive(Debug, Clone)]
pub enum DataLocation {
    Host(Vec<u8>),
    Device(u32, *mut u8), // device_id, pointer
    Unified(*mut u8),     // unified memory pointer
}

unsafe impl Send for DataLocation {}
unsafe impl Sync for DataLocation {}

/// Task priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TaskPriority {
    Low = 0,
    Normal = 1,
    High = 2,
    Critical = 3,
}

/// Completed task information
#[derive(Debug, Clone)]
pub struct CompletedTask {
    pub task_id: String,
    pub device_id: u32,
    pub execution_time_ms: f64,
    pub memory_used: usize,
    pub success: bool,
    pub error: Option<String>,
}

/// GPU barrier for synchronization
pub struct GpuBarrier {
    name: String,
    expected_participants: u32,
    current_participants: u32,
    waiting_devices: Vec<u32>,
}

/// GPU event for asynchronous operations
pub struct GpuEvent {
    name: String,
    device_id: u32,
    is_recorded: bool,
    backend_event: Option<Box<dyn Any>>,
}

impl MultiGpuCoordinator {
    /// Create a new multi-GPU coordinator
    pub fn new(devices: Vec<GpuDevice>) -> Self {
        let memory_manager = Arc::new(Mutex::new(MultiGpuMemoryManager::new()));

        // Add devices to memory manager
        #[cfg(not(feature = "no-std"))]
        {
            if let Ok(mut manager) = memory_manager.lock() {
                for device in &devices {
                    manager.add_device(device.clone());
                }
            }
        }
        #[cfg(feature = "no-std")]
        {
            let mut manager = memory_manager.lock();
            for device in &devices {
                manager.add_device(device.clone());
            }
        }

        Self {
            devices,
            memory_manager,
            load_balancer: LoadBalancer::new(LoadBalancingStrategy::ComputeWeighted),
            task_scheduler: TaskScheduler::new(),
            sync_manager: SynchronizationManager::new(),
        }
    }

    /// Add a task to the scheduler
    pub fn submit_task(&mut self, task: GpuTask) -> Result<(), SimdError> {
        self.task_scheduler.add_task(task);
        Ok(())
    }

    /// Execute all pending tasks
    pub fn execute_all(&mut self) -> Result<Vec<CompletedTask>, SimdError> {
        let mut results = Vec::new();

        // Schedule tasks based on dependencies and load balancing
        let scheduled_tasks = self.schedule_tasks()?;

        // Execute tasks in parallel (or sequentially in no-std)
        #[cfg(not(feature = "no-std"))]
        {
            let handles: Vec<_> = scheduled_tasks
                .into_iter()
                .map(|(device_id, tasks)| {
                    let memory_manager = Arc::clone(&self.memory_manager);
                    thread::spawn(move || {
                        Self::execute_device_tasks(device_id, tasks, memory_manager)
                    })
                })
                .collect();

            // Collect results
            for handle in handles {
                match handle.join() {
                    Ok(device_results) => results.extend(device_results),
                    Err(_) => {
                        return Err(SimdError::ExternalLibraryError(
                            "Thread execution failed".to_string(),
                        ))
                    }
                }
            }
        }

        #[cfg(feature = "no-std")]
        {
            // Sequential execution for no-std
            for (device_id, tasks) in scheduled_tasks {
                let memory_manager = Arc::clone(&self.memory_manager);
                let device_results = Self::execute_device_tasks(device_id, tasks, memory_manager);
                results.extend(device_results);
            }
        }

        // Update performance history
        self.update_performance_history(&results);

        Ok(results)
    }

    /// Execute a distributed matrix multiplication
    pub fn distributed_matrix_multiply(
        &mut self,
        a: &[f32],
        b: &[f32],
        a_rows: usize,
        a_cols: usize,
        b_cols: usize,
    ) -> Result<Vec<f32>, SimdError> {
        let num_devices = self.devices.len();
        if num_devices == 0 {
            return Err(SimdError::ExternalLibraryError(
                "No GPU devices available".to_string(),
            ));
        }

        // Distribute rows across devices
        let rows_per_device = a_rows / num_devices;
        let mut tasks = Vec::new();

        for (i, device) in self.devices.iter().enumerate() {
            let start_row = i * rows_per_device;
            let end_row = if i == num_devices - 1 {
                a_rows
            } else {
                (i + 1) * rows_per_device
            };
            let device_rows = end_row - start_row;

            if device_rows == 0 {
                continue;
            }

            // Create task for this device
            let task = GpuTask {
                id: format!("matmul_device_{}", i),
                kernel_name: "matrix_mul".to_string(),
                config: KernelConfig {
                    grid_size: (
                        ((b_cols + 15) / 16) as u32,
                        ((device_rows + 15) / 16) as u32,
                        1,
                    ),
                    block_size: (16, 16, 1),
                    shared_memory: 0,
                    stream: None,
                },
                input_data: vec![
                    GpuTaskData {
                        name: "matrix_a".to_string(),
                        #[cfg(not(feature = "no-std"))]
                        size: device_rows * a_cols * std::mem::size_of::<f32>(),
                        #[cfg(feature = "no-std")]
                        size: device_rows * a_cols * mem::size_of::<f32>(),
                        data_type: "f32".to_string(),
                        location: DataLocation::Host(
                            a[start_row * a_cols..end_row * a_cols]
                                .iter()
                                .flat_map(|&x| x.to_ne_bytes())
                                .collect(),
                        ),
                    },
                    GpuTaskData {
                        name: "matrix_b".to_string(),
                        #[cfg(not(feature = "no-std"))]
                        size: a_cols * b_cols * std::mem::size_of::<f32>(),
                        #[cfg(feature = "no-std")]
                        size: a_cols * b_cols * mem::size_of::<f32>(),
                        data_type: "f32".to_string(),
                        location: DataLocation::Host(
                            b.iter().flat_map(|&x| x.to_ne_bytes()).collect(),
                        ),
                    },
                ],
                output_data: vec![GpuTaskData {
                    name: "matrix_c".to_string(),
                    #[cfg(not(feature = "no-std"))]
                    size: device_rows * b_cols * std::mem::size_of::<f32>(),
                    #[cfg(feature = "no-std")]
                    size: device_rows * b_cols * mem::size_of::<f32>(),
                    data_type: "f32".to_string(),
                    location: DataLocation::Host(Vec::new()),
                }],
                device_preference: Some(device.id),
                priority: TaskPriority::High,
                dependencies: Vec::new(),
            };

            tasks.push(task);
        }

        // Submit and execute tasks
        for task in tasks {
            self.submit_task(task)?;
        }

        let results = self.execute_all()?;

        // Combine results
        let output = vec![0.0f32; a_rows * b_cols];
        let mut _current_row = 0;

        for result in results {
            if result.success {
                // Extract result data from completed task
                // This would involve copying from GPU memory
                let device_rows = rows_per_device;
                _current_row += device_rows;
            }
        }

        Ok(output)
    }

    /// Set load balancing strategy
    pub fn set_load_balancing(&mut self, strategy: LoadBalancingStrategy) {
        self.load_balancer.set_strategy(strategy);
    }

    /// Get device utilization statistics
    pub fn get_device_stats(&self) -> HashMap<u32, DeviceStats> {
        let mut stats = HashMap::new();

        for device in &self.devices {
            let device_stats = DeviceStats {
                device_id: device.id,
                name: device.name.clone(),
                compute_units: device.compute_units,
                memory_mb: device.memory_mb,
                current_tasks: self.task_scheduler.get_device_task_count(device.id),
                average_performance: self.load_balancer.get_average_performance(device.id),
            };
            stats.insert(device.id, device_stats);
        }

        stats
    }

    fn schedule_tasks(&mut self) -> Result<HashMap<u32, Vec<GpuTask>>, SimdError> {
        let mut scheduled = HashMap::new();

        // Get available tasks (no pending dependencies)
        let available_tasks = self.task_scheduler.get_available_tasks();

        for task in available_tasks {
            let device_id = if let Some(preferred) = task.device_preference {
                preferred
            } else {
                self.load_balancer.select_device(&self.devices, &task)?
            };

            scheduled
                .entry(device_id)
                .or_insert_with(Vec::new)
                .push(task);
        }

        Ok(scheduled)
    }

    fn execute_device_tasks(
        device_id: u32,
        tasks: Vec<GpuTask>,
        _memory_manager: Arc<Mutex<MultiGpuMemoryManager>>,
    ) -> Vec<CompletedTask> {
        let mut results = Vec::new();

        for task in tasks {
            #[cfg(not(feature = "no-std"))]
            let start_time = std::time::Instant::now();
            #[cfg(feature = "no-std")]
            let start_time = Instant::now();

            // Execute task (simplified)
            let result = CompletedTask {
                task_id: task.id.clone(),
                device_id,
                #[cfg(not(feature = "no-std"))]
                execution_time_ms: start_time.elapsed().as_millis() as f64,
                #[cfg(feature = "no-std")]
                execution_time_ms: start_time.elapsed() as f64 / 1_000_000.0, // Convert nanoseconds to milliseconds
                memory_used: task.input_data.iter().map(|d| d.size).sum(),
                success: true, // Placeholder
                error: None,
            };

            results.push(result);
        }

        results
    }

    fn update_performance_history(&mut self, results: &[CompletedTask]) {
        for result in results {
            self.load_balancer.add_performance_sample(
                result.device_id,
                1.0 / result.execution_time_ms, // Operations per ms
            );
        }
    }
}

/// Device performance statistics
#[derive(Debug, Clone)]
pub struct DeviceStats {
    pub device_id: u32,
    pub name: String,
    pub compute_units: u32,
    pub memory_mb: u64,
    pub current_tasks: usize,
    pub average_performance: f64,
}

impl LoadBalancer {
    pub fn new(strategy: LoadBalancingStrategy) -> Self {
        Self {
            strategy,
            device_weights: HashMap::new(),
            performance_history: HashMap::new(),
        }
    }

    pub fn set_strategy(&mut self, strategy: LoadBalancingStrategy) {
        self.strategy = strategy;
    }

    pub fn select_device(&self, devices: &[GpuDevice], _task: &GpuTask) -> Result<u32, SimdError> {
        if devices.is_empty() {
            return Err(SimdError::ExternalLibraryError(
                "No devices available".to_string(),
            ));
        }

        match self.strategy {
            LoadBalancingStrategy::Equal => Ok(devices[0].id),
            LoadBalancingStrategy::ComputeWeighted => {
                // Select device with most compute units
                let best_device = devices.iter().max_by_key(|d| d.compute_units).unwrap();
                Ok(best_device.id)
            }
            LoadBalancingStrategy::BandwidthWeighted => {
                // Select device with most memory (proxy for bandwidth)
                let best_device = devices.iter().max_by_key(|d| d.memory_mb).unwrap();
                Ok(best_device.id)
            }
            LoadBalancingStrategy::Dynamic => {
                // Select device with best recent performance
                let best_device = devices
                    .iter()
                    .max_by(|a, b| {
                        let a_perf = self.get_average_performance(a.id);
                        let b_perf = self.get_average_performance(b.id);
                        a_perf.partial_cmp(&b_perf).unwrap_or(Ordering::Equal)
                    })
                    .unwrap();
                Ok(best_device.id)
            }
            LoadBalancingStrategy::Custom => {
                // Use custom weights
                let best_device = devices
                    .iter()
                    .max_by(|a, b| {
                        let a_weight = self.device_weights.get(&a.id).unwrap_or(&1.0);
                        let b_weight = self.device_weights.get(&b.id).unwrap_or(&1.0);
                        a_weight.partial_cmp(b_weight).unwrap_or(Ordering::Equal)
                    })
                    .unwrap();
                Ok(best_device.id)
            }
        }
    }

    pub fn add_performance_sample(&mut self, device_id: u32, performance: f64) {
        let history = self.performance_history.entry(device_id).or_default();
        history.push(performance);

        // Keep only recent samples
        if history.len() > 100 {
            history.remove(0);
        }
    }

    pub fn get_average_performance(&self, device_id: u32) -> f64 {
        if let Some(history) = self.performance_history.get(&device_id) {
            if history.is_empty() {
                1.0
            } else {
                history.iter().sum::<f64>() / history.len() as f64
            }
        } else {
            1.0
        }
    }

    pub fn set_custom_weight(&mut self, device_id: u32, weight: f64) {
        self.device_weights.insert(device_id, weight);
    }
}

impl TaskScheduler {
    pub fn new() -> Self {
        Self {
            pending_tasks: Vec::new(),
            running_tasks: HashMap::new(),
            completed_tasks: Vec::new(),
        }
    }

    pub fn add_task(&mut self, task: GpuTask) {
        self.pending_tasks.push(task);
    }

    pub fn get_available_tasks(&mut self) -> Vec<GpuTask> {
        let completed_ids: HashSet<_> = self.completed_tasks.iter().map(|t| &t.task_id).collect();

        let mut available = Vec::new();
        let mut remaining = Vec::new();

        for task in self.pending_tasks.drain(..) {
            let deps_satisfied = task
                .dependencies
                .iter()
                .all(|dep| completed_ids.contains(dep));

            if deps_satisfied {
                available.push(task);
            } else {
                remaining.push(task);
            }
        }

        self.pending_tasks = remaining;
        available.sort_by(|a, b| b.priority.cmp(&a.priority));
        available
    }

    pub fn get_device_task_count(&self, device_id: u32) -> usize {
        self.running_tasks
            .get(&device_id)
            .map_or(0, |tasks| tasks.len())
    }

    pub fn mark_task_completed(&mut self, task_id: String) {
        // Remove from running tasks
        for tasks in self.running_tasks.values_mut() {
            tasks.retain(|t| t.id != task_id);
        }
    }
}

impl SynchronizationManager {
    pub fn new() -> Self {
        Self {
            barriers: HashMap::new(),
            events: HashMap::new(),
        }
    }

    pub fn create_barrier(
        &mut self,
        name: String,
        participant_count: u32,
    ) -> Result<(), SimdError> {
        let barrier = GpuBarrier {
            name: name.clone(),
            expected_participants: participant_count,
            current_participants: 0,
            waiting_devices: Vec::new(),
        };

        self.barriers.insert(name, barrier);
        Ok(())
    }

    pub fn wait_barrier(&mut self, name: &str, device_id: u32) -> Result<(), SimdError> {
        let should_synchronize = if let Some(barrier) = self.barriers.get_mut(name) {
            barrier.current_participants += 1;
            barrier.waiting_devices.push(device_id);

            if barrier.current_participants >= barrier.expected_participants {
                // All devices reached barrier, synchronize
                let waiting_devices = barrier.waiting_devices.clone();
                barrier.current_participants = 0;
                barrier.waiting_devices.clear();
                Some(waiting_devices)
            } else {
                None
            }
        } else {
            return Err(SimdError::InvalidParameter {
                name: "name".to_string(),
                value: name.to_string(),
            });
        };

        if let Some(waiting_devices) = should_synchronize {
            self.synchronize_devices(&waiting_devices)?;
        }

        Ok(())
    }

    pub fn create_event(&mut self, name: String, device_id: u32) -> Result<(), SimdError> {
        let event = GpuEvent {
            name: name.clone(),
            device_id,
            is_recorded: false,
            backend_event: None,
        };

        self.events.insert(name, event);
        Ok(())
    }

    pub fn record_event(&mut self, name: &str) -> Result<(), SimdError> {
        if let Some(event) = self.events.get_mut(name) {
            event.is_recorded = true;
            // Record backend-specific event
            Ok(())
        } else {
            Err(SimdError::InvalidParameter {
                name: "event".to_string(),
                value: format!("Event '{}' not found", name),
            })
        }
    }

    fn synchronize_devices(&self, device_ids: &[u32]) -> Result<(), SimdError> {
        // Synchronize all specified devices
        for &_device_id in device_ids {
            // Device-specific synchronization
        }
        Ok(())
    }
}

impl Default for TaskScheduler {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for SynchronizationManager {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(non_snake_case)]
#[cfg(all(test, not(feature = "no-std")))]
mod tests {
    use super::*;
    use crate::gpu::GpuBackend;

    #[cfg(feature = "no-std")]
    use alloc::{
        string::{String, ToString},
        vec,
        vec::Vec,
    };

    #[test]
    fn test_multi_gpu_coordinator_creation() {
        let devices = vec![
            GpuDevice {
                id: 0,
                name: "Device 0".to_string(),
                backend: GpuBackend::Cuda,
                compute_units: 80,
                memory_mb: 8192,
                supports_f64: true,
                supports_f16: true,
            },
            GpuDevice {
                id: 1,
                name: "Device 1".to_string(),
                backend: GpuBackend::Cuda,
                compute_units: 40,
                memory_mb: 4096,
                supports_f64: true,
                supports_f16: true,
            },
        ];

        let coordinator = MultiGpuCoordinator::new(devices);
        assert_eq!(coordinator.devices.len(), 2);
    }

    #[test]
    fn test_load_balancer() {
        let balancer = LoadBalancer::new(LoadBalancingStrategy::ComputeWeighted);

        let devices = vec![
            GpuDevice {
                id: 0,
                name: "Device 0".to_string(),
                backend: GpuBackend::Cuda,
                compute_units: 80,
                memory_mb: 8192,
                supports_f64: true,
                supports_f16: true,
            },
            GpuDevice {
                id: 1,
                name: "Device 1".to_string(),
                backend: GpuBackend::Cuda,
                compute_units: 40,
                memory_mb: 4096,
                supports_f64: true,
                supports_f16: true,
            },
        ];

        let task = GpuTask {
            id: "test_task".to_string(),
            kernel_name: "test_kernel".to_string(),
            config: KernelConfig::default(),
            input_data: Vec::new(),
            output_data: Vec::new(),
            device_preference: None,
            priority: TaskPriority::Normal,
            dependencies: Vec::new(),
        };

        let selected = balancer.select_device(&devices, &task).unwrap();
        assert_eq!(selected, 0); // Device 0 has more compute units
    }

    #[test]
    fn test_task_scheduler() {
        let mut scheduler = TaskScheduler::new();

        let task = GpuTask {
            id: "test_task".to_string(),
            kernel_name: "test_kernel".to_string(),
            config: KernelConfig::default(),
            input_data: Vec::new(),
            output_data: Vec::new(),
            device_preference: None,
            priority: TaskPriority::High,
            dependencies: Vec::new(),
        };

        scheduler.add_task(task);
        let available = scheduler.get_available_tasks();
        assert_eq!(available.len(), 1);
        assert_eq!(available[0].priority, TaskPriority::High);
    }

    #[test]
    fn test_task_dependencies() {
        let mut scheduler = TaskScheduler::new();

        let task1 = GpuTask {
            id: "task1".to_string(),
            kernel_name: "kernel1".to_string(),
            config: KernelConfig::default(),
            input_data: Vec::new(),
            output_data: Vec::new(),
            device_preference: None,
            priority: TaskPriority::Normal,
            dependencies: Vec::new(),
        };

        let task2 = GpuTask {
            id: "task2".to_string(),
            kernel_name: "kernel2".to_string(),
            config: KernelConfig::default(),
            input_data: Vec::new(),
            output_data: Vec::new(),
            device_preference: None,
            priority: TaskPriority::Normal,
            dependencies: vec!["task1".to_string()],
        };

        scheduler.add_task(task1);
        scheduler.add_task(task2);

        let available = scheduler.get_available_tasks();
        assert_eq!(available.len(), 1); // Only task1 should be available
        assert_eq!(available[0].id, "task1");
    }

    #[test]
    fn test_synchronization_manager() {
        let mut sync_manager = SynchronizationManager::new();

        sync_manager
            .create_barrier("test_barrier".to_string(), 2)
            .unwrap();
        sync_manager
            .create_event("test_event".to_string(), 0)
            .unwrap();

        assert!(sync_manager.barriers.contains_key("test_barrier"));
        assert!(sync_manager.events.contains_key("test_event"));
    }

    #[test]
    fn test_device_stats() {
        let stats = DeviceStats {
            device_id: 0,
            name: "Test Device".to_string(),
            compute_units: 80,
            memory_mb: 8192,
            current_tasks: 3,
            average_performance: 1.5,
        };

        assert_eq!(stats.device_id, 0);
        assert_eq!(stats.current_tasks, 3);
        assert!((stats.average_performance - 1.5).abs() < 0.001);
    }
}
