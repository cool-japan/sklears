//! CPU resource management
//!
//! This module provides comprehensive CPU resource allocation, NUMA awareness,
//! and performance optimization for the resource management system.

use super::resource_types::CpuAllocation;
use sklears_core::error::{Result as SklResult, SklearsError};
use std::collections::HashMap;
use std::time::Duration;

/// CPU resource manager for core allocation and NUMA optimization
#[derive(Debug)]
pub struct CpuResourceManager {
    /// CPU topology information
    topology: CpuTopology,
    /// Core allocations
    allocations: HashMap<String, CpuAllocation>,
    /// NUMA nodes
    numa_nodes: Vec<NumaNode>,
    /// CPU cores
    cpu_cores: Vec<CpuCore>,
    /// CPU configuration
    config: CpuManagerConfig,
    /// Performance statistics
    stats: CpuStats,
}

/// CPU topology information
#[derive(Debug, Clone)]
pub struct CpuTopology {
    /// Number of physical CPUs
    pub physical_cpus: u32,
    /// Cores per CPU
    pub cores_per_cpu: u32,
    /// Threads per core
    pub threads_per_core: u32,
    /// Total logical CPUs
    pub logical_cpus: u32,
    /// CPU architecture
    pub architecture: String,
    /// CPU features
    pub features: Vec<String>,
    /// Cache hierarchy
    pub cache_hierarchy: CacheHierarchy,
}

/// CPU cache hierarchy
#[derive(Debug, Clone)]
pub struct CacheHierarchy {
    /// L1 instruction cache
    pub l1i_cache: CacheInfo,
    /// L1 data cache
    pub l1d_cache: CacheInfo,
    /// L2 cache
    pub l2_cache: CacheInfo,
    /// L3 cache
    pub l3_cache: Option<CacheInfo>,
    /// L4 cache
    pub l4_cache: Option<CacheInfo>,
}

/// CPU cache information
#[derive(Debug, Clone)]
pub struct CacheInfo {
    /// Cache size in bytes
    pub size: u64,
    /// Cache line size
    pub line_size: u32,
    /// Cache associativity
    pub associativity: u32,
    /// Shared among cores
    pub shared: bool,
}

/// NUMA node information
#[derive(Debug, Clone)]
pub struct NumaNode {
    /// Node ID
    pub node_id: usize,
    /// CPU cores in this node
    pub cpu_cores: Vec<usize>,
    /// Memory size
    pub memory_size: u64,
    /// Memory bandwidth
    pub memory_bandwidth: f64,
    /// Distance to other nodes
    pub distances: HashMap<usize, u32>,
    /// Node utilization
    pub utilization: f64,
}

/// Individual CPU core
#[derive(Debug, Clone)]
pub struct CpuCore {
    /// Core ID
    pub core_id: usize,
    /// Physical core ID
    pub physical_id: usize,
    /// NUMA node
    pub numa_node: usize,
    /// Core state
    pub state: CoreState,
    /// Current frequency
    pub current_frequency: f64,
    /// Available frequencies
    pub available_frequencies: Vec<f64>,
    /// Current governor
    pub governor: String,
    /// Core utilization
    pub utilization: f64,
    /// Temperature
    pub temperature: Option<f64>,
}

/// CPU core states
#[derive(Debug, Clone, PartialEq)]
pub enum CoreState {
    /// Idle
    Idle,
    /// Running
    Running,
    /// Offline
    Offline,
    /// Throttled
    Throttled,
    /// Error
    Error,
}

/// CPU manager configuration
#[derive(Debug, Clone)]
pub struct CpuManagerConfig {
    /// Enable NUMA awareness
    pub numa_aware: bool,
    /// CPU affinity strategy
    pub affinity_strategy: AffinityStrategy,
    /// Enable dynamic frequency scaling
    pub enable_dvfs: bool,
    /// Default CPU governor
    pub default_governor: String,
    /// Enable thermal management
    pub enable_thermal_management: bool,
    /// Thermal throttling threshold
    pub thermal_threshold: f64,
    /// Core utilization threshold
    pub utilization_threshold: f64,
}

/// CPU affinity strategies
#[derive(Debug, Clone)]
pub enum AffinityStrategy {
    /// No specific affinity
    None,
    /// Prefer cores on same NUMA node
    NumaLocal,
    /// Prefer physical cores over hyperthreads
    PhysicalCores,
    /// Spread across all cores
    Spread,
    /// Pack cores together
    Pack,
    /// Custom affinity pattern
    Custom(Vec<usize>),
}

/// CPU performance statistics
#[derive(Debug, Clone)]
pub struct CpuStats {
    /// Total allocations
    pub total_allocations: u64,
    /// Active allocations
    pub active_allocations: u64,
    /// Average core utilization
    pub avg_utilization: f64,
    /// Peak utilization
    pub peak_utilization: f64,
    /// Context switches per second
    pub context_switches_per_sec: f64,
    /// Cache hit rate
    pub cache_hit_rate: f64,
    /// NUMA misses
    pub numa_misses: u64,
    /// Thermal throttling events
    pub thermal_events: u64,
}

/// CPU allocation request
#[derive(Debug, Clone)]
pub struct CpuAllocationRequest {
    /// Task ID
    pub task_id: String,
    /// Number of cores requested
    pub cores_requested: u32,
    /// Minimum frequency requirement
    pub min_frequency: Option<f64>,
    /// NUMA node preference
    pub numa_preference: Option<usize>,
    /// CPU affinity requirements
    pub affinity_requirements: Option<Vec<usize>>,
    /// Exclusive core access
    pub exclusive_access: bool,
    /// Performance requirements
    pub performance_requirements: PerformanceRequirements,
}

/// Performance requirements for CPU allocation
#[derive(Debug, Clone)]
pub struct PerformanceRequirements {
    /// Minimum CPU performance
    pub min_performance: f64,
    /// Maximum acceptable latency
    pub max_latency: Duration,
    /// Required cache coherency
    pub cache_coherency: bool,
    /// SIMD requirements
    pub simd_requirements: Vec<String>,
}

impl Default for CpuResourceManager {
    fn default() -> Self {
        Self::new()
    }
}

impl CpuResourceManager {
    /// Create a new CPU resource manager
    #[must_use]
    pub fn new() -> Self {
        let topology = Self::detect_topology();
        let numa_nodes = Self::detect_numa_topology();
        let cpu_cores = Self::initialize_cores(&topology, &numa_nodes);

        Self {
            topology,
            allocations: HashMap::new(),
            numa_nodes,
            cpu_cores,
            config: CpuManagerConfig::default(),
            stats: CpuStats::default(),
        }
    }

    /// Detect CPU topology
    fn detect_topology() -> CpuTopology {
        // In a real implementation, this would query the system
        // For now, return a default topology
        /// CpuTopology
        CpuTopology {
            physical_cpus: num_cpus::get_physical() as u32,
            cores_per_cpu: 1,
            threads_per_core: if num_cpus::get() > num_cpus::get_physical() {
                2
            } else {
                1
            },
            logical_cpus: num_cpus::get() as u32,
            architecture: std::env::consts::ARCH.to_string(),
            features: vec!["sse".to_string(), "sse2".to_string(), "avx".to_string()],
            cache_hierarchy: CacheHierarchy {
                l1i_cache: CacheInfo {
                    size: 32 * 1024,
                    line_size: 64,
                    associativity: 8,
                    shared: false,
                },
                l1d_cache: CacheInfo {
                    size: 32 * 1024,
                    line_size: 64,
                    associativity: 8,
                    shared: false,
                },
                l2_cache: CacheInfo {
                    size: 256 * 1024,
                    line_size: 64,
                    associativity: 8,
                    shared: false,
                },
                l3_cache: Some(CacheInfo {
                    size: 8 * 1024 * 1024,
                    line_size: 64,
                    associativity: 16,
                    shared: true,
                }),
                l4_cache: None,
            },
        }
    }

    /// Detect NUMA topology
    fn detect_numa_topology() -> Vec<NumaNode> {
        // In a real implementation, this would query NUMA information
        // For now, return a single NUMA node
        vec![NumaNode {
            node_id: 0,
            cpu_cores: (0..num_cpus::get()).collect(),
            memory_size: 16 * 1024 * 1024 * 1024, // 16GB default
            memory_bandwidth: 25600.0,            // MB/s
            distances: [(0, 10)].iter().copied().collect(),
            utilization: 0.0,
        }]
    }

    /// Initialize CPU cores
    fn initialize_cores(topology: &CpuTopology, numa_nodes: &[NumaNode]) -> Vec<CpuCore> {
        (0..topology.logical_cpus)
            .map(|core_id| {
                let numa_node = numa_nodes
                    .iter()
                    .find(|node| node.cpu_cores.contains(&(core_id as usize)))
                    .map_or(0, |node| node.node_id);

                /// CpuCore
                CpuCore {
                    core_id: core_id as usize,
                    physical_id: core_id as usize / topology.threads_per_core as usize,
                    numa_node,
                    state: CoreState::Idle,
                    current_frequency: 2400.0, // MHz
                    available_frequencies: vec![1200.0, 1800.0, 2400.0, 3200.0],
                    governor: "ondemand".to_string(),
                    utilization: 0.0,
                    temperature: None,
                }
            })
            .collect()
    }

    /// Allocate CPU resources
    pub fn allocate_cpu(&mut self, request: CpuAllocationRequest) -> SklResult<CpuAllocation> {
        // Find available cores
        let available_cores = self.find_available_cores(&request)?;

        if available_cores.len() < request.cores_requested as usize {
            return Err(SklearsError::ResourceAllocationError(format!(
                "Not enough CPU cores available: requested {}, available {}",
                request.cores_requested,
                available_cores.len()
            )));
        }

        // Select optimal cores based on strategy
        let selected_cores = self.select_optimal_cores(&available_cores, &request)?;

        // Create allocation
        let allocation = CpuAllocation {
            cores: selected_cores.clone(),
            threads: selected_cores.clone(), // Simplified: assume 1:1 mapping
            affinity_mask: self.create_affinity_mask(&selected_cores),
            numa_node: self.get_numa_node_for_cores(&selected_cores),
            frequency: request.min_frequency,
            governor: Some(self.config.default_governor.clone()),
        };

        // Update core states
        for &core_id in &selected_cores {
            if let Some(core) = self.cpu_cores.iter_mut().find(|c| c.core_id == core_id) {
                core.state = CoreState::Running;
            }
        }

        // Store allocation
        self.allocations.insert(request.task_id, allocation.clone());
        self.stats.total_allocations += 1;
        self.stats.active_allocations += 1;

        Ok(allocation)
    }

    /// Release CPU allocation
    pub fn release_cpu(&mut self, task_id: &str) -> SklResult<()> {
        if let Some(allocation) = self.allocations.remove(task_id) {
            // Update core states back to idle
            for &core_id in &allocation.cores {
                if let Some(core) = self.cpu_cores.iter_mut().find(|c| c.core_id == core_id) {
                    core.state = CoreState::Idle;
                }
            }
            self.stats.active_allocations = self.stats.active_allocations.saturating_sub(1);
            Ok(())
        } else {
            Err(SklearsError::ResourceAllocationError(format!(
                "No CPU allocation found for task {task_id}"
            )))
        }
    }

    /// Find available CPU cores
    fn find_available_cores(&self, request: &CpuAllocationRequest) -> SklResult<Vec<usize>> {
        let mut available = Vec::new();

        for core in &self.cpu_cores {
            if core.state == CoreState::Idle {
                // Check NUMA preference
                if let Some(preferred_numa) = request.numa_preference {
                    if core.numa_node != preferred_numa {
                        continue;
                    }
                }

                // Check frequency requirement
                if let Some(min_freq) = request.min_frequency {
                    if core.available_frequencies.iter().all(|&f| f < min_freq) {
                        continue;
                    }
                }

                available.push(core.core_id);
            }
        }

        Ok(available)
    }

    /// Select optimal cores based on affinity strategy
    fn select_optimal_cores(
        &self,
        available_cores: &[usize],
        request: &CpuAllocationRequest,
    ) -> SklResult<Vec<usize>> {
        let mut selected = Vec::new();
        let cores_needed = request.cores_requested as usize;

        if available_cores.len() < cores_needed {
            return Err(SklearsError::ResourceAllocationError(
                "Not enough cores available".to_string(),
            ));
        }

        match &self.config.affinity_strategy {
            AffinityStrategy::NumaLocal => {
                // Group cores by NUMA node and prefer local allocation
                let mut numa_groups: HashMap<usize, Vec<usize>> = HashMap::new();
                for &core_id in available_cores {
                    if let Some(core) = self.cpu_cores.iter().find(|c| c.core_id == core_id) {
                        numa_groups.entry(core.numa_node).or_default().push(core_id);
                    }
                }

                // Try to satisfy the request from a single NUMA node first
                for (_, mut cores) in numa_groups {
                    if cores.len() >= cores_needed {
                        cores.sort_unstable();
                        selected.extend(cores.into_iter().take(cores_needed));
                        break;
                    }
                }

                // If no single NUMA node can satisfy, distribute across nodes
                if selected.is_empty() {
                    let mut remaining = cores_needed;
                    for &core_id in available_cores {
                        if remaining == 0 {
                            break;
                        }
                        selected.push(core_id);
                        remaining -= 1;
                    }
                }
            }
            AffinityStrategy::PhysicalCores => {
                // Prefer physical cores over hyperthreads
                let mut physical_cores = Vec::new();
                let mut hyperthreads = Vec::new();

                for &core_id in available_cores {
                    if let Some(core) = self.cpu_cores.iter().find(|c| c.core_id == core_id) {
                        if core.core_id == core.physical_id {
                            physical_cores.push(core_id);
                        } else {
                            hyperthreads.push(core_id);
                        }
                    }
                }

                // First use physical cores
                let from_physical = physical_cores.into_iter().take(cores_needed);
                selected.extend(from_physical);

                // Then use hyperthreads if needed
                let remaining = cores_needed - selected.len();
                if remaining > 0 {
                    selected.extend(hyperthreads.into_iter().take(remaining));
                }
            }
            AffinityStrategy::Spread => {
                // Distribute across all available NUMA nodes
                let mut numa_assignment: HashMap<usize, Vec<usize>> = HashMap::new();
                for &core_id in available_cores {
                    if let Some(core) = self.cpu_cores.iter().find(|c| c.core_id == core_id) {
                        numa_assignment
                            .entry(core.numa_node)
                            .or_default()
                            .push(core_id);
                    }
                }

                let numa_count = numa_assignment.len();
                let cores_per_numa = cores_needed / numa_count;
                let mut extra_cores = cores_needed % numa_count;

                for (_, mut cores) in numa_assignment {
                    let take_count = if extra_cores > 0 {
                        extra_cores -= 1;
                        cores_per_numa + 1
                    } else {
                        cores_per_numa
                    };

                    cores.sort_unstable();
                    let cores_len = cores.len();
                    selected.extend(cores.into_iter().take(take_count.min(cores_len)));
                    if selected.len() >= cores_needed {
                        break;
                    }
                }
            }
            AffinityStrategy::Pack => {
                // Pack cores together on same NUMA node
                let mut cores_sorted = available_cores.to_vec();
                cores_sorted.sort_unstable();
                selected.extend(cores_sorted.into_iter().take(cores_needed));
            }
            AffinityStrategy::Custom(pattern) => {
                // Use custom affinity pattern
                for &core_id in pattern {
                    if available_cores.contains(&core_id) && selected.len() < cores_needed {
                        selected.push(core_id);
                    }
                }
                // Fill remaining with any available cores
                for &core_id in available_cores {
                    if !selected.contains(&core_id) && selected.len() < cores_needed {
                        selected.push(core_id);
                    }
                }
            }
            AffinityStrategy::None => {
                // Just take first available cores
                selected.extend(available_cores.iter().take(cores_needed));
            }
        }

        if selected.len() < cores_needed {
            return Err(SklearsError::ResourceAllocationError(format!(
                "Could not satisfy core allocation: needed {}, got {}",
                cores_needed,
                selected.len()
            )));
        }

        Ok(selected)
    }

    /// Create CPU affinity mask from core list
    fn create_affinity_mask(&self, cores: &[usize]) -> u64 {
        let mut mask = 0u64;
        for &core_id in cores {
            if core_id < 64 {
                mask |= 1u64 << core_id;
            }
        }
        mask
    }

    /// Get NUMA node for a set of cores
    fn get_numa_node_for_cores(&self, cores: &[usize]) -> Option<usize> {
        // Return the most common NUMA node among the cores
        let mut numa_counts: HashMap<usize, usize> = HashMap::new();

        for &core_id in cores {
            if let Some(core) = self.cpu_cores.iter().find(|c| c.core_id == core_id) {
                *numa_counts.entry(core.numa_node).or_insert(0) += 1;
            }
        }

        numa_counts
            .into_iter()
            .max_by_key(|(_, count)| *count)
            .map(|(numa_node, _)| numa_node)
    }

    /// Get current CPU statistics
    #[must_use]
    pub fn get_stats(&self) -> &CpuStats {
        &self.stats
    }

    /// Get CPU topology information
    #[must_use]
    pub fn get_topology(&self) -> &CpuTopology {
        &self.topology
    }

    /// Update CPU utilization statistics
    pub fn update_utilization(&mut self) -> SklResult<()> {
        // In a real implementation, this would query system CPU usage
        // For now, simulate some utilization
        let total_utilization: f64 = self.cpu_cores.iter().map(|core| core.utilization).sum();

        self.stats.avg_utilization = total_utilization / self.cpu_cores.len() as f64;
        self.stats.peak_utilization = self
            .cpu_cores
            .iter()
            .map(|core| core.utilization)
            .fold(0.0, f64::max);

        Ok(())
    }
}

impl Default for CpuManagerConfig {
    fn default() -> Self {
        Self {
            numa_aware: true,
            affinity_strategy: AffinityStrategy::NumaLocal,
            enable_dvfs: true,
            default_governor: "ondemand".to_string(),
            enable_thermal_management: true,
            thermal_threshold: 85.0,     // Celsius
            utilization_threshold: 80.0, // Percent
        }
    }
}

impl Default for CpuStats {
    fn default() -> Self {
        Self {
            total_allocations: 0,
            active_allocations: 0,
            avg_utilization: 0.0,
            peak_utilization: 0.0,
            context_switches_per_sec: 0.0,
            cache_hit_rate: 0.0,
            numa_misses: 0,
            thermal_events: 0,
        }
    }
}
