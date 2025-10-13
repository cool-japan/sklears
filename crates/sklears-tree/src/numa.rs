//! NUMA-aware memory allocation and optimization for tree algorithms
//!
//! This module provides Non-Uniform Memory Access (NUMA) optimizations
//! for improved performance on multi-socket systems.

use scirs2_core::ndarray::{Array1, Array2};
use num_cpus;
use sklears_core::error::{Result, SklearsError};
use std::alloc::{alloc, dealloc, Layout};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::thread;

#[cfg(target_os = "linux")]
use std::fs;

/// NUMA topology information
#[derive(Debug, Clone)]
pub struct NumaTopology {
    /// Number of NUMA nodes
    pub num_nodes: usize,
    /// CPU IDs for each NUMA node
    pub node_cpus: HashMap<usize, Vec<usize>>,
    /// Memory sizes for each NUMA node (in bytes)
    pub node_memory: HashMap<usize, u64>,
    /// Distance matrix between NUMA nodes
    pub distance_matrix: Array2<u32>,
    /// Available memory per node
    pub available_memory: HashMap<usize, u64>,
}

impl NumaTopology {
    /// Detect NUMA topology on the current system
    pub fn detect() -> Result<Self> {
        #[cfg(target_os = "linux")]
        {
            Self::detect_linux()
        }
        #[cfg(not(target_os = "linux"))]
        {
            Self::detect_fallback()
        }
    }

    #[cfg(target_os = "linux")]
    fn detect_linux() -> Result<Self> {
        let mut num_nodes = 0;
        let mut node_cpus = HashMap::new();
        let mut node_memory = HashMap::new();
        let mut available_memory = HashMap::new();

        // Detect number of NUMA nodes
        if let Ok(entries) = fs::read_dir("/sys/devices/system/node") {
            for entry in entries {
                if let Ok(entry) = entry {
                    let name = entry.file_name();
                    if let Some(name_str) = name.to_str() {
                        if name_str.starts_with("node") {
                            if let Ok(node_id) = name_str[4..].parse::<usize>() {
                                num_nodes = num_nodes.max(node_id + 1);

                                // Read CPU list for this node
                                let cpulist_path =
                                    format!("/sys/devices/system/node/{}/cpulist", name_str);
                                if let Ok(cpulist) = fs::read_to_string(&cpulist_path) {
                                    let cpus = Self::parse_cpu_list(&cpulist.trim())?;
                                    node_cpus.insert(node_id, cpus);
                                }

                                // Read memory info for this node
                                let meminfo_path =
                                    format!("/sys/devices/system/node/{}/meminfo", name_str);
                                if let Ok(meminfo) = fs::read_to_string(&meminfo_path) {
                                    let (total, available) = Self::parse_meminfo(&meminfo)?;
                                    node_memory.insert(node_id, total);
                                    available_memory.insert(node_id, available);
                                }
                            }
                        }
                    }
                }
            }
        }

        // Create distance matrix
        let distance_matrix = Self::create_distance_matrix(num_nodes)?;

        if num_nodes == 0 {
            // Fallback if NUMA detection fails
            return Self::detect_fallback();
        }

        Ok(NumaTopology {
            num_nodes,
            node_cpus,
            node_memory,
            distance_matrix,
            available_memory,
        })
    }

    #[cfg(not(target_os = "linux"))]
    fn detect_fallback() -> Result<Self> {
        // Fallback for non-Linux systems - assume single NUMA node
        let num_cpus = num_cpus::get();
        let mut node_cpus = HashMap::new();
        node_cpus.insert(0, (0..num_cpus).collect());

        let mut node_memory = HashMap::new();
        let mut available_memory = HashMap::new();

        // Estimate system memory (rough approximation)
        let estimated_memory = 8 * 1024 * 1024 * 1024u64; // 8GB default
        node_memory.insert(0, estimated_memory);
        available_memory.insert(0, estimated_memory / 2);

        let distance_matrix = Array2::from_elem((1, 1), 10u32);

        Ok(NumaTopology {
            num_nodes: 1,
            node_cpus,
            node_memory,
            distance_matrix,
            available_memory,
        })
    }

    #[cfg(target_os = "linux")]
    fn parse_cpu_list(cpulist: &str) -> Result<Vec<usize>> {
        let mut cpus = Vec::new();

        for part in cpulist.split(',') {
            if part.contains('-') {
                let range: Vec<&str> = part.split('-').collect();
                if range.len() == 2 {
                    let start: usize = range[0].parse().map_err(|_| {
                        sklears_core::error::SklearsError::InvalidData {
                            reason: "Invalid CPU range start".to_string(),
                        }
                    })?;
                    let end: usize = range[1].parse().map_err(|_| {
                        sklears_core::error::SklearsError::InvalidData {
                            reason: "Invalid CPU range end".to_string(),
                        }
                    })?;
                    cpus.extend(start..=end);
                }
            } else {
                let cpu: usize =
                    part.parse()
                        .map_err(|_| sklears_core::error::SklearsError::InvalidData {
                            reason: "Invalid CPU ID".to_string(),
                        })?;
                cpus.push(cpu);
            }
        }

        Ok(cpus)
    }

    #[cfg(target_os = "linux")]
    fn parse_meminfo(meminfo: &str) -> Result<(u64, u64)> {
        let mut total = 0u64;
        let mut available = 0u64;

        for line in meminfo.lines() {
            if line.contains("MemTotal:") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 2 {
                    total = parts[1].parse::<u64>().unwrap_or(0) * 1024; // Convert KB to bytes
                }
            } else if line.contains("MemFree:") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 2 {
                    available = parts[1].parse::<u64>().unwrap_or(0) * 1024; // Convert KB to bytes
                }
            }
        }

        Ok((total, available))
    }

    fn create_distance_matrix(num_nodes: usize) -> Result<Array2<u32>> {
        let mut matrix = Array2::zeros((num_nodes, num_nodes));

        #[cfg(target_os = "linux")]
        {
            // Try to read NUMA distances from sysfs
            for i in 0..num_nodes {
                let distance_path = format!("/sys/devices/system/node/node{}/distance", i);
                if let Ok(distances) = fs::read_to_string(&distance_path) {
                    let distances: Vec<u32> = distances
                        .split_whitespace()
                        .filter_map(|s| s.parse().ok())
                        .collect();

                    for (j, &distance) in distances.iter().enumerate() {
                        if j < num_nodes {
                            matrix[[i, j]] = distance;
                        }
                    }
                }
            }
        }

        // Fallback: create reasonable distance matrix
        for i in 0..num_nodes {
            for j in 0..num_nodes {
                if matrix[[i, j]] == 0 {
                    if i == j {
                        matrix[[i, j]] = 10; // Local access
                    } else {
                        matrix[[i, j]] = 20; // Remote access
                    }
                }
            }
        }

        Ok(matrix)
    }

    /// Get the best NUMA node for a given CPU
    pub fn get_node_for_cpu(&self, cpu_id: usize) -> Option<usize> {
        for (node_id, cpus) in &self.node_cpus {
            if cpus.contains(&cpu_id) {
                return Some(*node_id);
            }
        }
        None
    }

    /// Get CPUs for a given NUMA node
    pub fn get_cpus_for_node(&self, node_id: usize) -> Option<&Vec<usize>> {
        self.node_cpus.get(&node_id)
    }

    /// Find the closest NUMA node to a given node
    pub fn find_closest_node(&self, node_id: usize) -> Option<usize> {
        if node_id >= self.num_nodes {
            return None;
        }

        let mut min_distance = u32::MAX;
        let mut closest_node = None;

        for other_node in 0..self.num_nodes {
            if other_node != node_id {
                let distance = self.distance_matrix[[node_id, other_node]];
                if distance < min_distance {
                    min_distance = distance;
                    closest_node = Some(other_node);
                }
            }
        }

        closest_node
    }

    /// Get available memory for a NUMA node
    pub fn get_available_memory(&self, node_id: usize) -> Option<u64> {
        self.available_memory.get(&node_id).copied()
    }
}

/// NUMA-aware memory allocator
pub struct NumaAllocator {
    topology: NumaTopology,
    allocations: Arc<Mutex<HashMap<*mut u8, (usize, Layout)>>>,
}

impl NumaAllocator {
    /// Create a new NUMA-aware allocator
    pub fn new() -> Result<Self> {
        let topology = NumaTopology::detect()?;
        let allocations = Arc::new(Mutex::new(HashMap::new()));

        Ok(Self {
            topology,
            allocations,
        })
    }

    /// Allocate memory on a specific NUMA node
    pub fn allocate_on_node(&self, size: usize, node_id: usize) -> Result<*mut u8> {
        let layout = Layout::from_size_align(size, 8).map_err(|_| {
            sklears_core::error::SklearsError::InvalidData {
                reason: "Invalid memory layout".to_string(),
            }
        })?;

        // For now, use standard allocation
        // In a real implementation, this would use numa_alloc_onnode or similar
        let ptr = unsafe { alloc(layout) };

        if ptr.is_null() {
            return Err(sklears_core::error::SklearsError::NumericalError(format!(
                "Failed to allocate {} bytes on NUMA node {}",
                size, node_id
            )));
        }

        // Track allocation
        {
            let mut allocations = self.allocations.lock().unwrap();
            allocations.insert(ptr, (node_id, layout));
        }

        log::debug!("Allocated {} bytes on NUMA node {}", size, node_id);
        Ok(ptr)
    }

    /// Deallocate NUMA-aware memory
    ///
    /// # Safety
    /// The caller must ensure that `ptr` was allocated with this allocator and is valid
    pub unsafe fn deallocate(&self, ptr: *mut u8) -> Result<()> {
        let (node_id, layout) = {
            let mut allocations = self.allocations.lock().unwrap();
            allocations.remove(&ptr).ok_or_else(|| {
                sklears_core::error::SklearsError::InvalidData {
                    reason: "Pointer not found in allocations".to_string(),
                }
            })?
        };

        unsafe {
            dealloc(ptr, layout);
        }

        log::debug!("Deallocated memory from NUMA node {}", node_id);
        Ok(())
    }

    /// Get the best NUMA node for current thread
    pub fn get_current_node(&self) -> usize {
        // Try to get current CPU and map to NUMA node
        #[cfg(target_os = "linux")]
        {
            if let Ok(cpu_str) = std::fs::read_to_string("/proc/self/stat") {
                let fields: Vec<&str> = cpu_str.split_whitespace().collect();
                if fields.len() > 38 {
                    if let Ok(cpu) = fields[38].parse::<usize>() {
                        if let Some(node) = self.topology.get_node_for_cpu(cpu) {
                            return node;
                        }
                    }
                }
            }
        }

        // Fallback to node 0
        0
    }

    /// Allocate array on optimal NUMA node
    pub fn allocate_array<T>(&self, shape: (usize, usize)) -> Result<Array2<T>>
    where
        T: Clone + Default,
    {
        let total_elements = shape.0 * shape.1;
        let element_size = std::mem::size_of::<T>();
        let total_size = total_elements * element_size;

        let optimal_node = self.choose_optimal_node(total_size)?;

        // For demonstration, create array normally
        // In real implementation, would use custom allocation
        let array = Array2::default(shape);

        log::debug!(
            "Allocated array of {} elements on NUMA node {}",
            total_elements,
            optimal_node
        );
        Ok(array)
    }

    /// Choose optimal NUMA node based on available memory and current thread
    fn choose_optimal_node(&self, required_size: usize) -> Result<usize> {
        let current_node = self.get_current_node();

        // Check if current node has enough memory
        if let Some(available) = self.topology.get_available_memory(current_node) {
            if available >= required_size as u64 {
                return Ok(current_node);
            }
        }

        // Find node with most available memory
        let mut best_node = 0;
        let mut max_memory = 0u64;

        for node_id in 0..self.topology.num_nodes {
            if let Some(available) = self.topology.get_available_memory(node_id) {
                if available > max_memory && available >= required_size as u64 {
                    max_memory = available;
                    best_node = node_id;
                }
            }
        }

        Ok(best_node)
    }

    /// Get topology information
    pub fn topology(&self) -> &NumaTopology {
        &self.topology
    }
}

/// NUMA-aware data placement strategies
#[derive(Debug, Clone)]
pub enum DataPlacementStrategy {
    /// Distribute data evenly across all NUMA nodes
    Interleaved,
    /// Place data on the same node as the accessing thread
    LocalFirst,
    /// Place data based on access patterns
    AccessAware,
    /// Custom placement strategy
    Custom(HashMap<usize, usize>), // data_segment -> node_id
}

/// NUMA-aware tree data structure
pub struct NumaTreeData<T> {
    /// Data segments distributed across NUMA nodes
    segments: Vec<(Array2<T>, usize)>, // (data, node_id)
    /// Placement strategy used
    strategy: DataPlacementStrategy,
    /// NUMA allocator
    allocator: NumaAllocator,
}

impl<T> NumaTreeData<T>
where
    T: Clone + Default + Send + Sync,
{
    /// Create NUMA-aware tree data with specified strategy
    pub fn new(data: Array2<T>, strategy: DataPlacementStrategy) -> Result<Self> {
        let allocator = NumaAllocator::new()?;
        let segments = Self::distribute_data(data, &strategy, &allocator)?;

        Ok(Self {
            segments,
            strategy,
            allocator,
        })
    }

    /// Distribute data across NUMA nodes according to strategy
    fn distribute_data(
        data: Array2<T>,
        strategy: &DataPlacementStrategy,
        allocator: &NumaAllocator,
    ) -> Result<Vec<(Array2<T>, usize)>> {
        let topology = allocator.topology();
        let num_nodes = topology.num_nodes;
        let mut segments = Vec::new();

        match strategy {
            DataPlacementStrategy::Interleaved => {
                // Split data into chunks for each NUMA node
                let rows_per_node = data.nrows() / num_nodes;
                let remaining_rows = data.nrows() % num_nodes;

                let mut start_row = 0;
                for node_id in 0..num_nodes {
                    let end_row =
                        start_row + rows_per_node + if node_id < remaining_rows { 1 } else { 0 };

                    if start_row < data.nrows() {
                        let segment = data
                            .slice(s![start_row..end_row.min(data.nrows()), ..])
                            .to_owned();
                        segments.push((segment, node_id));
                    }

                    start_row = end_row;
                }
            }

            DataPlacementStrategy::LocalFirst => {
                // Place all data on current node
                let current_node = allocator.get_current_node();
                segments.push((data, current_node));
            }

            DataPlacementStrategy::AccessAware => {
                // Simplified access-aware placement
                // In practice, this would analyze access patterns
                let optimal_node =
                    allocator.choose_optimal_node(data.len() * std::mem::size_of::<T>())?;
                segments.push((data, optimal_node));
            }

            DataPlacementStrategy::Custom(placement_map) => {
                // Use custom placement mapping
                for (&segment_id, &node_id) in placement_map {
                    if segment_id == 0 {
                        // For simplicity, place all data on specified node
                        segments.push((data.clone(), node_id));
                        break;
                    }
                }

                if segments.is_empty() {
                    segments.push((data, 0)); // Fallback to node 0
                }
            }
        }

        log::info!(
            "Distributed data into {} segments across NUMA nodes",
            segments.len()
        );
        Ok(segments)
    }

    /// Get data segment for a specific NUMA node
    pub fn get_segment_for_node(&self, node_id: usize) -> Option<&Array2<T>> {
        self.segments
            .iter()
            .find(|(_, segment_node)| *segment_node == node_id)
            .map(|(data, _)| data)
    }

    /// Get all segments
    pub fn segments(&self) -> &[(Array2<T>, usize)] {
        &self.segments
    }

    /// Get placement strategy
    pub fn strategy(&self) -> &DataPlacementStrategy {
        &self.strategy
    }

    /// Optimize data access for current thread
    pub fn optimize_for_current_thread(&mut self) -> Result<()> {
        let current_node = self.allocator.get_current_node();

        // Move frequently accessed data to current node if beneficial
        let topology = self.allocator.topology();

        for (segment, node_id) in &mut self.segments {
            let distance = topology.distance_matrix[[current_node, *node_id]];

            // If data is on a distant node and current node has capacity, consider moving
            if distance > 15 && current_node != *node_id {
                let segment_size = segment.len() * std::mem::size_of::<T>();

                if let Some(available) = topology.get_available_memory(current_node) {
                    if available >= segment_size as u64 {
                        log::debug!(
                            "Moving data segment from node {} to node {} for better locality",
                            *node_id,
                            current_node
                        );
                        *node_id = current_node;
                    }
                }
            }
        }

        Ok(())
    }
}

/// NUMA-aware thread affinity manager
pub struct NumaAffinityManager {
    topology: NumaTopology,
    thread_assignments: Arc<Mutex<HashMap<thread::ThreadId, usize>>>,
}

impl NumaAffinityManager {
    /// Create new affinity manager
    pub fn new() -> Result<Self> {
        let topology = NumaTopology::detect()?;
        let thread_assignments = Arc::new(Mutex::new(HashMap::new()));

        Ok(Self {
            topology,
            thread_assignments,
        })
    }

    /// Assign current thread to a NUMA node
    pub fn bind_current_thread_to_node(&self, node_id: usize) -> Result<()> {
        if node_id >= self.topology.num_nodes {
            return Err(sklears_core::error::SklearsError::InvalidData {
                reason: format!("Invalid NUMA node ID: {}", node_id),
            });
        }

        let thread_id = thread::current().id();

        {
            let mut assignments = self.thread_assignments.lock().unwrap();
            assignments.insert(thread_id, node_id);
        }

        log::debug!("Bound thread {:?} to NUMA node {}", thread_id, node_id);

        // In a real implementation, this would call sched_setaffinity or similar
        // For now, we just track the assignment

        Ok(())
    }

    /// Get optimal node assignment for load balancing
    pub fn get_optimal_node_assignment(&self) -> Result<Vec<usize>> {
        let num_threads = num_cpus::get();
        let mut assignments = Vec::new();

        // Distribute threads evenly across NUMA nodes
        for i in 0..num_threads {
            let node_id = i % self.topology.num_nodes;
            assignments.push(node_id);
        }

        Ok(assignments)
    }

    /// Get current thread's NUMA node assignment
    pub fn get_current_thread_node(&self) -> Option<usize> {
        let thread_id = thread::current().id();
        let assignments = self.thread_assignments.lock().unwrap();
        assignments.get(&thread_id).copied()
    }

    /// Get topology information
    pub fn topology(&self) -> &NumaTopology {
        &self.topology
    }
}

/// NUMA optimization utilities for tree algorithms
pub struct NumaOptimizer {
    allocator: NumaAllocator,
    affinity_manager: NumaAffinityManager,
}

impl NumaOptimizer {
    /// Create new NUMA optimizer
    pub fn new() -> Result<Self> {
        let allocator = NumaAllocator::new()?;
        let affinity_manager = NumaAffinityManager::new()?;

        Ok(Self {
            allocator,
            affinity_manager,
        })
    }

    /// Optimize tree construction for NUMA
    pub fn optimize_tree_construction<F, R>(&self, data_size: usize, f: F) -> Result<R>
    where
        F: FnOnce() -> R + Send,
        R: Send,
    {
        // Choose optimal NUMA node based on data size and current thread
        let optimal_node = self.allocator.choose_optimal_node(data_size)?;

        // Bind current thread to optimal node
        self.affinity_manager
            .bind_current_thread_to_node(optimal_node)?;

        log::debug!(
            "Optimizing tree construction on NUMA node {} for {} bytes",
            optimal_node,
            data_size
        );

        // Execute the function
        Ok(f())
    }

    /// Create NUMA-optimized data structure for training data
    pub fn create_optimized_training_data(
        &self,
        x: &Array2<f64>,
        y: &Array1<i32>,
        strategy: DataPlacementStrategy,
    ) -> Result<(NumaTreeData<f64>, NumaTreeData<i32>)> {
        // Create NUMA-aware data structures
        let x_data = NumaTreeData::new(x.clone(), strategy.clone())?;

        // Convert y to 2D for compatibility
        let y_2d = y.clone().insert_axis(scirs2_core::ndarray::Axis(1));
        let y_data = NumaTreeData::new(y_2d, strategy)?;

        Ok((x_data, y_data))
    }

    /// Get allocator reference
    pub fn allocator(&self) -> &NumaAllocator {
        &self.allocator
    }

    /// Get affinity manager reference
    pub fn affinity_manager(&self) -> &NumaAffinityManager {
        &self.affinity_manager
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_numa_topology_detection() {
        let topology = NumaTopology::detect();
        assert!(topology.is_ok());

        let topo = topology.unwrap();
        assert!(topo.num_nodes > 0);
        assert!(!topo.node_cpus.is_empty());
    }

    #[test]
    fn test_numa_allocator_creation() {
        let allocator = NumaAllocator::new();
        assert!(allocator.is_ok());
    }

    #[test]
    fn test_data_placement_interleaved() {
        let data: Array2<f64> = Array2::zeros((100, 10));
        let strategy = DataPlacementStrategy::Interleaved;

        let numa_data = NumaTreeData::new(data, strategy);
        assert!(numa_data.is_ok());

        let numa_data = numa_data.unwrap();
        assert!(!numa_data.segments().is_empty());
    }

    #[test]
    fn test_data_placement_local_first() {
        let data: Array2<f64> = Array2::zeros((50, 5));
        let strategy = DataPlacementStrategy::LocalFirst;

        let numa_data = NumaTreeData::new(data, strategy);
        assert!(numa_data.is_ok());
    }

    #[test]
    fn test_affinity_manager() {
        let manager = NumaAffinityManager::new();
        assert!(manager.is_ok());

        let manager = manager.unwrap();
        let assignments = manager.get_optimal_node_assignment();
        assert!(assignments.is_ok());
    }

    #[test]
    fn test_numa_optimizer() {
        let optimizer = NumaOptimizer::new();
        assert!(optimizer.is_ok());

        let optimizer = optimizer.unwrap();
        let result = optimizer.optimize_tree_construction(1024, || {
            // Simulate tree construction work
            42
        });

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 42);
    }
}

// Add ndarray slice macro for compatibility
use scirs2_core::ndarray::s;
