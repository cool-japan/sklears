//! Resource management implementations for the composable execution engine
//!
//! This module provides comprehensive resource management including CPU, memory,
//! I/O, network, and GPU resource allocation with pluggable policies and monitoring.

use std::collections::HashMap;
use std::sync::{Mutex, RwLock};
use std::time::SystemTime;

use sklears_core::error::{Result as SklResult, SklearsError};

use super::tasks::{ExecutionTask, ResourceRequirements};

/// Resource utilization information
#[derive(Debug, Clone)]
pub struct ResourceUtilization {
    /// CPU utilization percentage
    pub cpu_percent: f64,
    /// Memory utilization percentage
    pub memory_percent: f64,
    /// I/O utilization percentage
    pub io_percent: f64,
    /// Network utilization percentage
    pub network_percent: f64,
    /// Queue utilization percentage
    pub queue_percent: f64,
}

impl Default for ResourceUtilization {
    fn default() -> Self {
        Self {
            cpu_percent: 0.0,
            memory_percent: 0.0,
            io_percent: 0.0,
            network_percent: 0.0,
            queue_percent: 0.0,
        }
    }
}

/// Resource manager for execution engines
pub struct ResourceManager {
    /// Available resources
    resources: RwLock<AvailableResources>,
    /// Resource allocations
    allocations: Mutex<HashMap<String, ResourceAllocation>>,
    /// Resource policies
    policies: ResourcePolicies,
    /// Resource monitors
    monitors: Vec<Box<dyn ResourceMonitor>>,
}

impl ResourceManager {
    /// Create a new resource manager
    pub fn new() -> SklResult<Self> {
        let resources = AvailableResources {
            cpu_cores: num_cpus::get(),
            total_memory: detect_total_memory(),
            available_memory: detect_available_memory(),
            io_bandwidth: detect_io_bandwidth(),
            network_bandwidth: detect_network_bandwidth(),
            available_disk_space: detect_available_disk_space(),
            gpu_info: detect_gpu_info(),
        };

        Ok(Self {
            resources: RwLock::new(resources),
            allocations: Mutex::new(HashMap::new()),
            policies: ResourcePolicies::default(),
            monitors: Vec::new(),
        })
    }

    /// Check resource availability for task requirements
    pub fn check_availability(&self, requirements: &ResourceRequirements) -> SklResult<()> {
        let resources = self.resources.read().map_err(|_| {
            SklearsError::InvalidInput("Failed to acquire resource lock".to_string())
        })?;

        // Check CPU cores
        if requirements.cpu_cores > resources.cpu_cores as f64 {
            return Err(SklearsError::InvalidInput(format!(
                "Insufficient CPU cores: required {}, available {}",
                requirements.cpu_cores, resources.cpu_cores
            )));
        }

        // Check memory
        if requirements.memory_bytes > resources.available_memory {
            return Err(SklearsError::InvalidInput(format!(
                "Insufficient memory: required {} bytes, available {} bytes",
                requirements.memory_bytes, resources.available_memory
            )));
        }

        // Check disk space
        if requirements.disk_bytes > 0 && requirements.disk_bytes > resources.available_disk_space {
            return Err(SklearsError::InvalidInput(format!(
                "Insufficient disk space: required {} bytes, available {} bytes",
                requirements.disk_bytes, resources.available_disk_space
            )));
        }

        // Check network bandwidth
        if requirements.network_bandwidth > resources.network_bandwidth {
            return Err(SklearsError::InvalidInput(
                "Insufficient network bandwidth".to_string(),
            ));
        }

        // Check GPU memory
        if requirements.gpu_memory_bytes > 0 {
            let available_gpu_memory: u64 = resources
                .gpu_info
                .iter()
                .map(|gpu| gpu.available_memory)
                .sum();

            if requirements.gpu_memory_bytes > available_gpu_memory {
                return Err(SklearsError::InvalidInput(
                    "Insufficient GPU memory".to_string(),
                ));
            }
        }

        Ok(())
    }

    /// Allocate resources for a task
    pub fn allocate_resources(&self, task: &ExecutionTask) -> SklResult<ResourceAllocation> {
        self.check_availability(&task.requirements)?;

        let allocation = match self.policies.cpu_policy {
            CpuAllocationPolicy::Exclusive => self.allocate_exclusive_resources(task)?,
            CpuAllocationPolicy::Shared => self.allocate_shared_resources(task)?,
            CpuAllocationPolicy::NumaAware => self.allocate_numa_aware_resources(task)?,
            CpuAllocationPolicy::PowerEfficient => self.allocate_power_efficient_resources(task)?,
        };

        let mut allocations = self.allocations.lock().map_err(|_| {
            SklearsError::InvalidInput("Failed to acquire allocation lock".to_string())
        })?;
        allocations.insert(task.id.clone(), allocation.clone());

        // Update available resources
        self.update_available_resources(&allocation, false)?;

        Ok(allocation)
    }

    /// Release resources from an allocation
    pub fn release_resources(&self, allocation: &ResourceAllocation) -> SklResult<()> {
        let mut allocations = self.allocations.lock().map_err(|_| {
            SklearsError::InvalidInput("Failed to acquire allocation lock".to_string())
        })?;

        if allocations.remove(&allocation.task_id).is_some() {
            // Update available resources
            self.update_available_resources(allocation, true)?;
        }

        Ok(())
    }

    /// Get resource utilization statistics
    pub fn get_utilization(&self) -> SklResult<ResourceUtilization> {
        let resources = self.resources.read().map_err(|_| {
            SklearsError::InvalidInput("Failed to acquire resource lock".to_string())
        })?;

        let allocations = self.allocations.lock().map_err(|_| {
            SklearsError::InvalidInput("Failed to acquire allocation lock".to_string())
        })?;

        let allocated_cores: usize = allocations
            .values()
            .map(|alloc| alloc.cpu_cores.len())
            .sum();

        let allocated_memory: u64 = allocations
            .values()
            .filter_map(|alloc| alloc.memory_range.map(|(_, size)| size))
            .sum();

        let allocated_io: u64 = allocations
            .values()
            .filter_map(|alloc| alloc.io_bandwidth)
            .sum();

        let cpu_percent = (allocated_cores as f64 / resources.cpu_cores as f64) * 100.0;
        let memory_percent = (allocated_memory as f64 / resources.total_memory as f64) * 100.0;

        let io_percent = if resources.io_bandwidth > 0 {
            (allocated_io as f64 / resources.io_bandwidth as f64) * 100.0
        } else {
            0.0
        };

        // Network utilization: sum of all allocated network bandwidth requirements
        // Note: Allocations don't currently track network bandwidth separately,
        // so we estimate based on task requirements if available
        let network_percent = 0.0; // Conservative estimate until network tracking is added to allocations

        // Queue utilization: percentage of active allocations vs some maximum
        // We use a heuristic of max 1000 concurrent tasks
        let max_concurrent_tasks = 1000;
        let queue_percent = (allocations.len() as f64 / max_concurrent_tasks as f64) * 100.0;

        Ok(ResourceUtilization {
            cpu_percent,
            memory_percent,
            io_percent,
            network_percent,
            queue_percent,
        })
    }

    /// Add a resource monitor
    pub fn add_monitor(&mut self, monitor: Box<dyn ResourceMonitor>) {
        self.monitors.push(monitor);
    }

    /// Collect metrics from all monitors
    pub fn collect_all_metrics(&self) -> Vec<ResourceMetrics> {
        self.monitors
            .iter()
            .map(|monitor| monitor.collect_metrics())
            .collect()
    }

    /// Update resource policies
    pub fn update_policies(&mut self, policies: ResourcePolicies) {
        self.policies = policies;
    }

    /// Get current resource policies
    pub fn get_policies(&self) -> ResourcePolicies {
        self.policies.clone()
    }

    // Private helper methods

    fn allocate_exclusive_resources(&self, task: &ExecutionTask) -> SklResult<ResourceAllocation> {
        let resources = self.resources.read().map_err(|_| {
            SklearsError::InvalidInput("Failed to acquire resource lock".to_string())
        })?;

        let required_cores = task.requirements.cpu_cores.ceil() as usize;
        let cpu_cores: Vec<usize> = (0..required_cores.min(resources.cpu_cores)).collect();

        Ok(ResourceAllocation {
            task_id: task.id.clone(),
            cpu_cores,
            memory_range: Some((0, task.requirements.memory_bytes)),
            io_bandwidth: Some(task.requirements.network_bandwidth),
            gpu_id: if task.requirements.gpu_memory_bytes > 0 {
                Some(0)
            } else {
                None
            },
            allocated_at: SystemTime::now(),
        })
    }

    fn allocate_shared_resources(&self, task: &ExecutionTask) -> SklResult<ResourceAllocation> {
        let resources = self.resources.read().map_err(|_| {
            SklearsError::InvalidInput("Failed to acquire resource lock".to_string())
        })?;

        // For shared allocation, assign virtual cores
        let required_cores = task.requirements.cpu_cores.ceil() as usize;
        let cpu_cores: Vec<usize> = (0..required_cores.min(resources.cpu_cores)).collect();

        Ok(ResourceAllocation {
            task_id: task.id.clone(),
            cpu_cores,
            memory_range: Some((0, task.requirements.memory_bytes)),
            io_bandwidth: Some(task.requirements.network_bandwidth),
            gpu_id: if task.requirements.gpu_memory_bytes > 0 {
                Some(0)
            } else {
                None
            },
            allocated_at: SystemTime::now(),
        })
    }

    fn allocate_numa_aware_resources(&self, task: &ExecutionTask) -> SklResult<ResourceAllocation> {
        // Simplified NUMA-aware allocation
        self.allocate_shared_resources(task)
    }

    fn allocate_power_efficient_resources(
        &self,
        task: &ExecutionTask,
    ) -> SklResult<ResourceAllocation> {
        // Simplified power-efficient allocation
        self.allocate_shared_resources(task)
    }

    fn update_available_resources(
        &self,
        allocation: &ResourceAllocation,
        release: bool,
    ) -> SklResult<()> {
        let mut resources = self.resources.write().map_err(|_| {
            SklearsError::InvalidInput("Failed to acquire resource lock".to_string())
        })?;

        if let Some((_, memory_size)) = allocation.memory_range {
            if release {
                resources.available_memory += memory_size;
            } else {
                resources.available_memory = resources.available_memory.saturating_sub(memory_size);
            }
        }

        Ok(())
    }
}

/// Available system resources
#[derive(Debug, Clone)]
pub struct AvailableResources {
    /// CPU cores
    pub cpu_cores: usize,
    /// Total memory (bytes)
    pub total_memory: u64,
    /// Available memory (bytes)
    pub available_memory: u64,
    /// I/O bandwidth (bytes/sec)
    pub io_bandwidth: u64,
    /// Network bandwidth (bytes/sec)
    pub network_bandwidth: u64,
    /// Available disk space (bytes)
    pub available_disk_space: u64,
    /// GPU information
    pub gpu_info: Vec<GpuInfo>,
}

/// GPU information
#[derive(Debug, Clone)]
pub struct GpuInfo {
    /// GPU identifier
    pub id: usize,
    /// GPU name
    pub name: String,
    /// Total memory (bytes)
    pub total_memory: u64,
    /// Available memory (bytes)
    pub available_memory: u64,
    /// Compute capability
    pub compute_capability: String,
    /// GPU vendor
    pub vendor: GpuVendor,
    /// Device utilization percentage
    pub utilization: f64,
}

/// GPU vendor information
#[derive(Debug, Clone, PartialEq)]
pub enum GpuVendor {
    /// Nvidia
    Nvidia,
    /// AMD
    AMD,
    /// Intel
    Intel,
    /// Apple
    Apple,
    /// Unknown
    Unknown,
}

/// Resource allocation for a task
#[derive(Debug, Clone)]
pub struct ResourceAllocation {
    /// Task identifier
    pub task_id: String,
    /// Allocated CPU cores
    pub cpu_cores: Vec<usize>,
    /// Allocated memory range (start, size)
    pub memory_range: Option<(u64, u64)>,
    /// Allocated I/O bandwidth
    pub io_bandwidth: Option<u64>,
    /// Allocated GPU
    pub gpu_id: Option<usize>,
    /// Allocation timestamp
    pub allocated_at: SystemTime,
}

/// Resource management policies
#[derive(Debug, Clone)]
pub struct ResourcePolicies {
    /// CPU allocation policy
    pub cpu_policy: CpuAllocationPolicy,
    /// Memory allocation policy
    pub memory_policy: MemoryAllocationPolicy,
    /// I/O allocation policy
    pub io_policy: IoAllocationPolicy,
    /// Priority-based allocation
    pub priority_allocation: bool,
    /// Fair share allocation
    pub fair_share: bool,
}

impl Default for ResourcePolicies {
    fn default() -> Self {
        Self {
            cpu_policy: CpuAllocationPolicy::Shared,
            memory_policy: MemoryAllocationPolicy::FirstFit,
            io_policy: IoAllocationPolicy::FairQueuing,
            priority_allocation: true,
            fair_share: true,
        }
    }
}

/// CPU allocation policies
#[derive(Debug, Clone, PartialEq)]
pub enum CpuAllocationPolicy {
    /// Exclusive CPU allocation
    Exclusive,
    /// Shared CPU allocation
    Shared,
    /// NUMA-aware allocation
    NumaAware,
    /// Power-efficient allocation
    PowerEfficient,
}

/// Memory allocation policies
#[derive(Debug, Clone, PartialEq)]
pub enum MemoryAllocationPolicy {
    /// First fit allocation
    FirstFit,
    /// Best fit allocation
    BestFit,
    /// Worst fit allocation
    WorstFit,
    /// Buddy system allocation
    BuddySystem,
    /// Slab allocator
    SlabAllocator,
}

/// I/O allocation policies
#[derive(Debug, Clone, PartialEq)]
pub enum IoAllocationPolicy {
    /// First In, First Out
    FIFO,
    /// Fair queuing
    FairQueuing,
    /// Weighted fair queuing
    WeightedFairQueuing,
    /// Deadline scheduling
    DeadlineScheduling,
}

/// Resource monitor trait for pluggable monitoring
pub trait ResourceMonitor: Send + Sync {
    /// Monitor name
    fn name(&self) -> &str;

    /// Collect resource metrics
    fn collect_metrics(&self) -> ResourceMetrics;

    /// Check resource health
    fn check_health(&self) -> ResourceHealth;

    /// Get monitor configuration
    fn get_config(&self) -> MonitorConfig;

    /// Update monitor configuration
    fn update_config(&mut self, config: MonitorConfig) -> SklResult<()>;
}

/// Resource metrics collected by monitors
#[derive(Debug, Clone)]
pub struct ResourceMetrics {
    /// Timestamp when metrics were collected
    pub timestamp: SystemTime,
    /// CPU metrics
    pub cpu: CpuMetrics,
    /// Memory metrics
    pub memory: MemoryMetrics,
    /// I/O metrics
    pub io: IoMetrics,
    /// Network metrics
    pub network: NetworkMetrics,
    /// GPU metrics
    pub gpu: Vec<GpuMetrics>,
}

impl Default for ResourceMetrics {
    fn default() -> Self {
        Self {
            timestamp: SystemTime::now(),
            cpu: CpuMetrics::default(),
            memory: MemoryMetrics::default(),
            io: IoMetrics::default(),
            network: NetworkMetrics::default(),
            gpu: Vec::new(),
        }
    }
}

/// CPU metrics
#[derive(Debug, Clone)]
pub struct CpuMetrics {
    /// Overall CPU utilization
    pub utilization: f64,
    /// Per-core utilization
    pub per_core: Vec<f64>,
    /// CPU frequency
    pub frequency: f64,
    /// Temperature (if available)
    pub temperature: Option<f64>,
    /// Power consumption (if available)
    pub power_consumption: Option<f64>,
}

impl Default for CpuMetrics {
    fn default() -> Self {
        Self {
            utilization: 0.0,
            per_core: Vec::new(),
            frequency: 0.0,
            temperature: None,
            power_consumption: None,
        }
    }
}

/// Memory metrics
#[derive(Debug, Clone, Default)]
pub struct MemoryMetrics {
    /// Used memory (bytes)
    pub used: u64,
    /// Available memory (bytes)
    pub available: u64,
    /// Cached memory (bytes)
    pub cached: u64,
    /// Buffer memory (bytes)
    pub buffers: u64,
    /// Swap usage (bytes)
    pub swap_used: u64,
}

/// I/O metrics
#[derive(Debug, Clone)]
pub struct IoMetrics {
    /// Read operations per second
    pub read_ops: f64,
    /// Write operations per second
    pub write_ops: f64,
    /// Read bandwidth (bytes/sec)
    pub read_bandwidth: f64,
    /// Write bandwidth (bytes/sec)
    pub write_bandwidth: f64,
    /// I/O wait time
    pub io_wait: f64,
}

impl Default for IoMetrics {
    fn default() -> Self {
        Self {
            read_ops: 0.0,
            write_ops: 0.0,
            read_bandwidth: 0.0,
            write_bandwidth: 0.0,
            io_wait: 0.0,
        }
    }
}

/// Network metrics
#[derive(Debug, Clone)]
pub struct NetworkMetrics {
    /// Bytes received per second
    pub rx_bytes: f64,
    /// Bytes transmitted per second
    pub tx_bytes: f64,
    /// Packets received per second
    pub rx_packets: f64,
    /// Packets transmitted per second
    pub tx_packets: f64,
    /// Network latency (if available)
    pub latency: Option<f64>,
}

impl Default for NetworkMetrics {
    fn default() -> Self {
        Self {
            rx_bytes: 0.0,
            tx_bytes: 0.0,
            rx_packets: 0.0,
            tx_packets: 0.0,
            latency: None,
        }
    }
}

/// GPU metrics
#[derive(Debug, Clone)]
pub struct GpuMetrics {
    /// GPU identifier
    pub gpu_id: usize,
    /// GPU utilization percentage
    pub utilization: f64,
    /// Memory utilization percentage
    pub memory_utilization: f64,
    /// Temperature (if available)
    pub temperature: Option<f64>,
    /// Power consumption (if available)
    pub power_consumption: Option<f64>,
}

/// Resource health status
#[derive(Debug, Clone, PartialEq)]
pub enum ResourceHealth {
    /// Resources are healthy
    Healthy,
    /// Resources have warnings
    Warning { reason: String },
    /// Resources are in critical state
    Critical { reason: String },
    /// Resources are unavailable
    Unavailable,
}

/// Monitor configuration
#[derive(Debug, Clone)]
pub struct MonitorConfig {
    /// Monitoring interval
    pub interval: std::time::Duration,
    /// Enable detailed monitoring
    pub detailed: bool,
    /// Alert thresholds
    pub thresholds: AlertThresholds,
}

impl Default for MonitorConfig {
    fn default() -> Self {
        Self {
            interval: std::time::Duration::from_secs(5),
            detailed: false,
            thresholds: AlertThresholds::default(),
        }
    }
}

/// Alert thresholds for resource monitoring
#[derive(Debug, Clone)]
pub struct AlertThresholds {
    /// CPU utilization threshold
    pub cpu_threshold: f64,
    /// Memory utilization threshold
    pub memory_threshold: f64,
    /// I/O utilization threshold
    pub io_threshold: f64,
    /// Network utilization threshold
    pub network_threshold: f64,
    /// GPU utilization threshold
    pub gpu_threshold: f64,
}

impl Default for AlertThresholds {
    fn default() -> Self {
        Self {
            cpu_threshold: 90.0,
            memory_threshold: 85.0,
            io_threshold: 80.0,
            network_threshold: 75.0,
            gpu_threshold: 90.0,
        }
    }
}

/// System resource monitor implementation
pub struct SystemResourceMonitor {
    config: MonitorConfig,
}

impl SystemResourceMonitor {
    /// Create a new system resource monitor
    #[must_use]
    pub fn new(config: MonitorConfig) -> Self {
        Self { config }
    }
}

impl ResourceMonitor for SystemResourceMonitor {
    fn name(&self) -> &'static str {
        "system_resource_monitor"
    }

    fn collect_metrics(&self) -> ResourceMetrics {
        /// ResourceMetrics
        ResourceMetrics {
            timestamp: SystemTime::now(),
            cpu: collect_cpu_metrics(),
            memory: collect_memory_metrics(),
            io: collect_io_metrics(),
            network: collect_network_metrics(),
            gpu: collect_gpu_metrics(),
        }
    }

    fn check_health(&self) -> ResourceHealth {
        let metrics = self.collect_metrics();

        if metrics.cpu.utilization > self.config.thresholds.cpu_threshold {
            return ResourceHealth::Critical {
                reason: format!("CPU utilization too high: {:.1}%", metrics.cpu.utilization),
            };
        }

        if metrics.memory.used as f64 / (metrics.memory.used + metrics.memory.available) as f64
            * 100.0
            > self.config.thresholds.memory_threshold
        {
            return ResourceHealth::Warning {
                reason: "Memory utilization high".to_string(),
            };
        }

        ResourceHealth::Healthy
    }

    fn get_config(&self) -> MonitorConfig {
        self.config.clone()
    }

    fn update_config(&mut self, config: MonitorConfig) -> SklResult<()> {
        self.config = config;
        Ok(())
    }
}

// Platform-specific resource detection functions
// These are placeholder implementations

fn detect_total_memory() -> u64 {
    // Placeholder: 8GB
    8 * 1024 * 1024 * 1024
}

fn detect_available_memory() -> u64 {
    // Placeholder: 6GB
    6 * 1024 * 1024 * 1024
}

fn detect_io_bandwidth() -> u64 {
    // Placeholder: 1GB/s
    1000 * 1024 * 1024
}

fn detect_network_bandwidth() -> u64 {
    // Placeholder: 100MB/s
    100 * 1024 * 1024
}

fn detect_gpu_info() -> Vec<GpuInfo> {
    // Placeholder: Empty GPU info
    Vec::new()
}

fn detect_available_disk_space() -> u64 {
    // Attempt to detect available disk space on the current directory
    // Falls back to a conservative placeholder if detection fails
    #[cfg(unix)]
    {
        use std::ffi::CString;
        use std::mem;
        use std::os::raw::{c_char, c_ulong};

        #[repr(C)]
        struct StatVfs {
            f_bsize: c_ulong,
            f_frsize: c_ulong,
            f_blocks: c_ulong,
            f_bfree: c_ulong,
            f_bavail: c_ulong,
            // ... other fields omitted for brevity
        }

        extern "C" {
            fn statvfs(path: *const c_char, buf: *mut StatVfs) -> i32;
        }

        let path = CString::new(".").unwrap();
        let mut stat: StatVfs = unsafe { mem::zeroed() };

        if unsafe { statvfs(path.as_ptr(), &mut stat) } == 0 {
            // Available space = f_bavail * f_frsize (blocks available to non-root * fragment size)
            return stat.f_bavail.saturating_mul(stat.f_frsize);
        }
    }

    // Placeholder fallback: 100GB
    100 * 1024 * 1024 * 1024
}

fn collect_cpu_metrics() -> CpuMetrics {
    CpuMetrics::default()
}

fn collect_memory_metrics() -> MemoryMetrics {
    MemoryMetrics::default()
}

fn collect_io_metrics() -> IoMetrics {
    IoMetrics::default()
}

fn collect_network_metrics() -> NetworkMetrics {
    NetworkMetrics::default()
}

fn collect_gpu_metrics() -> Vec<GpuMetrics> {
    Vec::new()
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::tasks::*;

    #[test]
    fn test_resource_manager_creation() {
        let resource_manager = ResourceManager::new();
        assert!(resource_manager.is_ok());
    }

    #[test]
    fn test_resource_allocation() {
        let resource_manager = ResourceManager::new().unwrap();
        let task = create_test_task();

        let allocation = resource_manager.allocate_resources(&task);
        assert!(allocation.is_ok());

        let alloc = allocation.unwrap();
        assert_eq!(alloc.task_id, task.id);
        assert!(!alloc.cpu_cores.is_empty());
    }

    #[test]
    fn test_resource_release() {
        let resource_manager = ResourceManager::new().unwrap();
        let task = create_test_task();

        let allocation = resource_manager.allocate_resources(&task).unwrap();
        let result = resource_manager.release_resources(&allocation);
        assert!(result.is_ok());
    }

    #[test]
    fn test_system_resource_monitor() {
        let config = MonitorConfig::default();
        let monitor = SystemResourceMonitor::new(config);

        assert_eq!(monitor.name(), "system_resource_monitor");

        let metrics = monitor.collect_metrics();
        assert!(metrics.timestamp <= SystemTime::now());

        let health = monitor.check_health();
        assert!(matches!(health, ResourceHealth::Healthy));
    }

    fn create_test_task() -> ExecutionTask {
        /// ExecutionTask
        ExecutionTask {
            id: "test_task_1".to_string(),
            task_type: TaskType::Computation,
            metadata: TaskMetadata {
                name: "Test Task".to_string(),
                description: "A test task".to_string(),
                priority: TaskPriority::Normal,
                estimated_duration: Some(std::time::Duration::from_secs(10)),
                deadline: None,
                dependencies: Vec::new(),
                tags: Vec::new(),
                created_at: SystemTime::now(),
            },
            requirements: ResourceRequirements {
                cpu_cores: 1.0,
                memory_bytes: 1024 * 1024,
                disk_bytes: 0,
                network_bandwidth: 0,
                gpu_memory_bytes: 0,
                special_resources: Vec::new(),
            },
            input_data: None,
            configuration: TaskConfiguration::default(),
        }
    }
}
