use crate::distributed_optimization::core_types::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::SystemTime;

/// Node capacity information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeCapacity {
    pub node_id: NodeId,
    pub cpu_capacity: CpuCapacity,
    pub memory_capacity: MemoryCapacity,
    pub storage_capacity: StorageCapacity,
    pub network_capacity: NetworkCapacity,
    pub gpu_capacity: Option<GpuCapacity>,
    pub custom_resources: HashMap<String, ResourceCapacity>,
    pub total_capacity_score: f64,
    pub last_updated: SystemTime,
}

/// CPU capacity metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuCapacity {
    pub total_cores: u32,
    pub available_cores: u32,
    pub cpu_utilization: f64,
    pub cpu_frequency: f64,
    pub cpu_architecture: CpuArchitecture,
    pub instruction_sets: Vec<InstructionSet>,
    pub cache_sizes: CacheSizes,
    pub thermal_state: ThermalState,
    pub power_consumption: f64,
}

/// CPU architecture types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CpuArchitecture {
    X86_64,
    ARM64,
    RISC_V,
    PowerPC,
    SPARC,
    Custom(String),
}

/// Instruction set support
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InstructionSet {
    SSE4_2,
    AVX,
    AVX2,
    AVX512,
    NEON,
    SVE,
    Custom(String),
}

/// CPU cache information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheSizes {
    pub l1_cache_kb: u64,
    pub l2_cache_kb: u64,
    pub l3_cache_kb: u64,
    pub cache_line_size: u32,
}

/// Thermal state monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ThermalState {
    Normal,
    Warm,
    Hot,
    Critical,
    Throttling,
}

/// Memory capacity metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryCapacity {
    pub total_memory_gb: f64,
    pub available_memory_gb: f64,
    pub memory_utilization: f64,
    pub memory_type: MemoryType,
    pub memory_speed: f64,
    pub memory_bandwidth: f64,
    pub swap_capacity: SwapCapacity,
    pub numa_topology: Option<NumaTopology>,
}

/// Memory types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MemoryType {
    DDR4,
    DDR5,
    LPDDR4,
    LPDDR5,
    HBM2,
    HBM3,
    Custom(String),
}

/// Swap memory information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwapCapacity {
    pub total_swap_gb: f64,
    pub used_swap_gb: f64,
    pub swap_utilization: f64,
}

/// NUMA topology information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NumaTopology {
    pub numa_nodes: Vec<NumaNode>,
    pub memory_affinity: HashMap<u32, Vec<u32>>,
    pub cpu_affinity: HashMap<u32, Vec<u32>>,
}

/// NUMA node information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NumaNode {
    pub node_id: u32,
    pub cpus: Vec<u32>,
    pub memory_gb: f64,
    pub distance_matrix: Vec<u32>,
}

/// Storage capacity metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageCapacity {
    pub storage_devices: Vec<StorageDevice>,
    pub total_storage_gb: f64,
    pub available_storage_gb: f64,
    pub storage_utilization: f64,
    pub iops_capacity: IopsCapacity,
    pub storage_tiers: Vec<StorageTier>,
}

/// Storage device information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageDevice {
    pub device_id: String,
    pub device_type: StorageType,
    pub capacity_gb: f64,
    pub available_gb: f64,
    pub read_speed_mbps: f64,
    pub write_speed_mbps: f64,
    pub iops_read: u64,
    pub iops_write: u64,
    pub wear_level: f64,
    pub temperature: f64,
}

/// Storage types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StorageType {
    SSD,
    NVMe,
    HDD,
    Optane,
    RAM_Disk,
    Network,
    Custom(String),
}

/// IOPS capacity metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IopsCapacity {
    pub max_read_iops: u64,
    pub max_write_iops: u64,
    pub current_read_iops: u64,
    pub current_write_iops: u64,
    pub iops_utilization: f64,
}

/// Storage tier information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageTier {
    pub tier_name: String,
    pub tier_type: StorageType,
    pub capacity_gb: f64,
    pub performance_class: PerformanceClass,
    pub cost_per_gb: f64,
}

/// Performance classification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PerformanceClass {
    HighPerformance,
    Standard,
    Archive,
    Backup,
    Custom(String),
}

/// Network capacity metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkCapacity {
    pub network_interfaces: Vec<NetworkInterface>,
    pub total_bandwidth_gbps: f64,
    pub available_bandwidth_gbps: f64,
    pub network_utilization: f64,
    pub latency_metrics: LatencyMetrics,
    pub packet_processing: PacketProcessingCapacity,
}

/// Network interface information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkInterface {
    pub interface_name: String,
    pub interface_type: NetworkInterfaceType,
    pub bandwidth_gbps: f64,
    pub current_utilization: f64,
    pub link_status: LinkStatus,
    pub mtu_size: u32,
    pub duplex_mode: DuplexMode,
}

/// Network interface types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NetworkInterfaceType {
    Ethernet,
    InfiniBand,
    WiFi,
    Bluetooth,
    Loopback,
    Virtual,
    Custom(String),
}

/// Link status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LinkStatus {
    Up,
    Down,
    Unknown,
}

/// Duplex modes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DuplexMode {
    FullDuplex,
    HalfDuplex,
    Simplex,
}

/// Network latency metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyMetrics {
    pub min_latency_ms: f64,
    pub max_latency_ms: f64,
    pub avg_latency_ms: f64,
    pub p95_latency_ms: f64,
    pub p99_latency_ms: f64,
    pub jitter_ms: f64,
}

/// Packet processing capacity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PacketProcessingCapacity {
    pub max_packets_per_second: u64,
    pub current_packets_per_second: u64,
    pub packet_drop_rate: f64,
    pub buffer_utilization: f64,
}

/// GPU capacity metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuCapacity {
    pub gpu_devices: Vec<GpuDevice>,
    pub total_compute_units: u32,
    pub available_compute_units: u32,
    pub total_memory_gb: f64,
    pub available_memory_gb: f64,
    pub gpu_utilization: f64,
    pub memory_utilization: f64,
}

/// GPU device information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuDevice {
    pub device_id: String,
    pub gpu_type: GpuType,
    pub compute_capability: ComputeCapability,
    pub memory_gb: f64,
    pub memory_bandwidth: f64,
    pub cuda_cores: Option<u32>,
    pub tensor_cores: Option<u32>,
    pub temperature: f64,
    pub power_usage: f64,
}

/// GPU types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GpuType {
    NVIDIA,
    AMD,
    Intel,
    Apple,
    Custom(String),
}

/// Compute capability
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputeCapability {
    pub major: u32,
    pub minor: u32,
    pub architecture: String,
}

/// Generic resource capacity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceCapacity {
    pub resource_type: String,
    pub total_capacity: f64,
    pub available_capacity: f64,
    pub utilization: f64,
    pub unit: String,
    pub measurement_type: MeasurementType,
}

/// Measurement types for resources
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MeasurementType {
    Absolute,
    Percentage,
    Rate,
    Count,
    Custom(String),
}