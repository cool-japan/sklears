//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::{
    collections::{HashMap, HashSet, VecDeque},
    sync::{Arc, RwLock, Mutex},
    time::{Duration, Instant, SystemTime},
    fmt::{Debug, Display},
    cmp::{Ordering, max, min},
};
use serde::{Serialize, Deserialize};
use uuid::Uuid;
use crate::context_core::{
    ExecutionContextTrait, ContextType, ContextState, ContextMetadata, ContextError, ContextResult,
    ContextEvent, IsolationLevel, ContextPriority
};
use super::functions::*;

/// Firewall rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FirewallRule {
    /// Rule ID
    pub id: String,
    /// Rule action
    pub action: FirewallAction,
    /// Source address pattern
    pub source: Option<String>,
    /// Destination address pattern
    pub destination: Option<String>,
    /// Port range
    pub port_range: Option<(u16, u16)>,
    /// Protocol
    pub protocol: Option<String>,
}
/// Scheduling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulingConfig {
    /// Scheduling policy
    pub policy: SchedulingPolicy,
    /// Scheduling priority
    pub priority: i32,
    /// CPU affinity
    pub cpu_affinity: Option<Vec<usize>>,
    /// Memory affinity (NUMA)
    pub memory_affinity: Option<Vec<u32>>,
    /// Preemption policy
    pub preemption: PreemptionPolicy,
}
/// Billing information
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BillingInfo {
    /// Total cost
    pub total_cost: f64,
    /// Cost by resource type
    pub cost_by_resource: HashMap<String, f64>,
    /// Billing period
    pub billing_period: Duration,
    /// Currency
    pub currency: String,
    /// Billing history
    pub billing_history: Vec<BillingRecord>,
}
/// Memory configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    /// Maximum memory in bytes
    pub max_memory: Option<usize>,
    /// Memory limit enforcement
    pub limit_enforcement: MemoryLimitEnforcement,
    /// Memory swapping policy
    pub swap_policy: SwapPolicy,
    /// Memory allocation strategy
    pub allocation_strategy: MemoryAllocationStrategy,
    /// Huge pages configuration
    pub huge_pages: HugePagesConfig,
    /// Memory protection settings
    pub protection: MemoryProtectionConfig,
}
/// Network limits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkLimits {
    /// Maximum bandwidth
    pub max_bandwidth: Option<u64>,
    /// Maximum connections
    pub max_connections: Option<u32>,
    /// Maximum packets per second
    pub max_pps: Option<u32>,
}
/// Quality of Service configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QosConfig {
    /// Enable QoS
    pub enabled: bool,
    /// Traffic class
    pub traffic_class: TrafficClass,
    /// Differentiated Services Code Point
    pub dscp: Option<u8>,
    /// Priority queuing
    pub priority_queuing: bool,
}
/// Resource requirement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceRequirement {
    /// Required CPU cores
    pub cpu_cores: Option<f32>,
    /// Required memory in bytes
    pub memory: Option<u64>,
    /// Required storage in bytes
    pub storage: Option<u64>,
    /// Required network bandwidth
    pub network_bandwidth: Option<u64>,
    /// Required GPU devices
    pub gpu_devices: Option<usize>,
    /// Custom resource requirements
    pub custom: HashMap<String, f64>,
    /// Minimum requirements (vs preferred)
    pub minimum: bool,
}
/// Storage allocation policy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StorageAllocationPolicy {
    /// Pre-allocated storage
    PreAllocated,
    /// On-demand allocation
    OnDemand,
    /// Thin provisioning
    ThinProvisioning,
    /// Thick provisioning
    ThickProvisioning,
}
/// CPU allocation policy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CpuAllocationPolicy {
    /// Exclusive CPU allocation
    Exclusive,
    /// Shared CPU allocation
    Shared,
    /// Proportional allocation
    Proportional,
    /// Guaranteed allocation
    Guaranteed,
}
/// Traffic classes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrafficClass {
    /// Best effort
    BestEffort,
    /// Background
    Background,
    /// Bulk data
    BulkData,
    /// Interactive
    Interactive,
    /// Voice
    Voice,
    /// Video
    Video,
    /// Network control
    NetworkControl,
}
/// Memory protection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryProtectionConfig {
    /// Enable memory protection
    pub enabled: bool,
    /// Stack guard pages
    pub stack_guard: bool,
    /// Heap guard pages
    pub heap_guard: bool,
    /// Address space layout randomization
    pub aslr: bool,
    /// Data execution prevention
    pub dep: bool,
}
/// Priority level definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriorityLevel {
    /// Priority level
    pub level: i32,
    /// Scheduling weight
    pub weight: f32,
}
/// Resource scheduler
#[derive(Debug)]
pub struct ResourceScheduler {
    /// Scheduling queue
    queue: Arc<Mutex<VecDeque<SchedulingRequest>>>,
    /// Scheduler policies
    policies: Arc<RwLock<SchedulerPolicies>>,
    /// Scheduler metrics
    metrics: Arc<Mutex<SchedulerMetrics>>,
}
impl ResourceScheduler {
    /// Create a new resource scheduler
    pub fn new() -> Self {
        Self {
            queue: Arc::new(Mutex::new(VecDeque::new())),
            policies: Arc::new(RwLock::new(SchedulerPolicies::default())),
            metrics: Arc::new(Mutex::new(SchedulerMetrics::default())),
        }
    }
    /// Submit scheduling request
    pub fn submit_request(&self, request: SchedulingRequest) -> ContextResult<()> {
        let mut queue = self.queue.lock().unwrap();
        queue.push_back(request);
        let mut metrics = self.metrics.lock().unwrap();
        metrics.total_requests += 1;
        metrics.queue_length = queue.len();
        Ok(())
    }
    /// Get next request from queue
    pub fn get_next_request(&self) -> ContextResult<Option<SchedulingRequest>> {
        let mut queue = self.queue.lock().unwrap();
        let request = queue.pop_front();
        let mut metrics = self.metrics.lock().unwrap();
        metrics.queue_length = queue.len();
        Ok(request)
    }
}
/// Quota reset policies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuotaResetPolicy {
    /// Never reset
    Never,
    /// Reset hourly
    Hourly,
    /// Reset daily
    Daily,
    /// Reset weekly
    Weekly,
    /// Reset monthly
    Monthly,
    /// Custom period
    Custom,
}
/// Connection limits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectionLimits {
    /// Maximum concurrent connections
    pub max_connections: Option<u32>,
    /// Maximum connections per IP
    pub max_connections_per_ip: Option<u32>,
    /// Connection timeout
    pub connection_timeout: Option<Duration>,
    /// Keep-alive timeout
    pub keepalive_timeout: Option<Duration>,
}
/// Alert manager
#[derive(Debug, Default)]
pub struct AlertManager {
    /// Active alerts
    active_alerts: HashMap<String, ResourceAlert>,
    /// Alert history
    alert_history: VecDeque<ResourceAlert>,
    /// Alert rules
    alert_rules: Vec<AlertRule>,
}
/// CPU configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuConfig {
    /// Maximum CPU cores
    pub max_cores: Option<usize>,
    /// CPU affinity mask
    pub affinity_mask: Option<u64>,
    /// CPU scheduling priority
    pub priority: CpuPriority,
    /// CPU throttling configuration
    pub throttling: CpuThrottling,
    /// NUMA node preference
    pub numa_preference: NumaPreference,
}
/// CPU priorities
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum CpuPriority {
    /// Idle priority
    Idle = -20,
    /// Low priority
    Low = -10,
    /// Below normal priority
    BelowNormal = -5,
    /// Normal priority
    Normal = 0,
    /// Above normal priority
    AboveNormal = 5,
    /// High priority
    High = 10,
    /// Real-time priority
    RealTime = 20,
}
/// Storage encryption configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageEncryptionConfig {
    /// Enable encryption at rest
    pub enabled: bool,
    /// Encryption algorithm
    pub algorithm: String,
    /// Key size in bits
    pub key_size: u32,
    /// Encryption mode
    pub mode: String,
}
/// Overcommit policies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OvercommitPolicies {
    /// CPU overcommit ratio
    pub cpu_overcommit_ratio: f32,
    /// Memory overcommit ratio
    pub memory_overcommit_ratio: f32,
    /// Storage overcommit ratio
    pub storage_overcommit_ratio: f32,
    /// Network overcommit ratio
    pub network_overcommit_ratio: f32,
    /// Enable dynamic overcommit adjustment
    pub dynamic_adjustment: bool,
}
/// Resource reservation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceReservation {
    /// Reservation ID
    pub reservation_id: String,
    /// Context ID
    pub context_id: String,
    /// Reserved resources
    pub resources: ResourceRequirement,
    /// Reservation start time
    pub start_time: SystemTime,
    /// Reservation duration
    pub duration: Duration,
    /// Reservation priority
    pub priority: ContextPriority,
    /// Reservation status
    pub status: ReservationStatus,
}
/// Resource manager metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ResourceManagerMetrics {
    /// Total allocations made
    pub total_allocations: u64,
    /// Active allocations
    pub active_allocations: u64,
    /// Failed allocations
    pub failed_allocations: u64,
    /// Average allocation time
    pub avg_allocation_time: Duration,
    /// Resource utilization
    pub utilization: ResourceUtilization,
    /// Allocation success rate
    pub success_rate: f32,
}
/// Monitoring granularity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MonitoringGranularity {
    /// Low granularity (basic metrics)
    Low,
    /// Standard granularity
    Standard,
    /// High granularity (detailed metrics)
    High,
    /// Maximum granularity (all metrics)
    Maximum,
}
/// Huge pages configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HugePagesConfig {
    /// Enable huge pages
    pub enabled: bool,
    /// Huge page size
    pub page_size: HugePageSize,
    /// Number of huge pages
    pub page_count: Option<usize>,
    /// Huge page allocation policy
    pub allocation_policy: HugePagePolicy,
}
/// Allocation policy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AllocationPolicy {
    /// First fit allocation
    FirstFit,
    /// Best fit allocation
    BestFit,
    /// Worst fit allocation
    WorstFit,
    /// Priority-based allocation
    Priority,
    /// Fair share allocation
    FairShare,
}
/// Resource usage
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ResourceUsage {
    /// CPU usage in cores
    pub cpu_usage: f32,
    /// Memory usage in bytes
    pub memory_usage: u64,
    /// Storage usage in bytes
    pub storage_usage: u64,
    /// Network ingress bytes per second
    pub network_ingress: u64,
    /// Network egress bytes per second
    pub network_egress: u64,
    /// GPU usage percentage
    pub gpu_usage: f32,
    /// Custom resource usage
    pub custom_usage: HashMap<String, f64>,
    /// Timestamp
    pub timestamp: SystemTime,
}
/// Alert severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum AlertSeverity {
    /// Info level
    Info = 1,
    /// Warning level
    Warning = 2,
    /// Error level
    Error = 3,
    /// Critical level
    Critical = 4,
}
/// Monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringConfig {
    /// Enable monitoring
    pub enabled: bool,
    /// Monitoring interval
    pub interval: Duration,
    /// Metrics retention period
    pub retention_period: Duration,
    /// Enable alerts
    pub enable_alerts: bool,
    /// Alert thresholds
    pub alert_thresholds: AlertThresholds,
    /// Monitoring granularity
    pub granularity: MonitoringGranularity,
}
/// Custom resource configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomResourceConfig {
    /// Resource name
    pub name: String,
    /// Resource type
    pub resource_type: String,
    /// Resource units
    pub units: String,
    /// Maximum allocation
    pub max_allocation: Option<f64>,
    /// Resource provider
    pub provider: Option<String>,
    /// Custom attributes
    pub attributes: HashMap<String, serde_json::Value>,
}
/// Usage point for tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsagePoint {
    /// Usage value
    pub value: f64,
    /// Timestamp
    pub timestamp: SystemTime,
}
/// Scheduling request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulingRequest {
    /// Request ID
    pub request_id: String,
    /// Context ID
    pub context_id: String,
    /// Resource requirements
    pub requirements: ResourceRequirement,
    /// Scheduling priority
    pub priority: ContextPriority,
    /// Deadline (if any)
    pub deadline: Option<SystemTime>,
    /// Request timestamp
    pub timestamp: SystemTime,
    /// Estimated duration
    pub estimated_duration: Option<Duration>,
}
/// Network isolation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkIsolationConfig {
    /// Enable network isolation
    pub enabled: bool,
    /// Network namespace
    pub namespace: Option<String>,
    /// VLAN ID
    pub vlan_id: Option<u16>,
    /// Firewall rules
    pub firewall_rules: Vec<FirewallRule>,
}
/// Huge page allocation policies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HugePagePolicy {
    /// Automatic allocation
    Auto,
    /// Always allocate huge pages
    Always,
    /// Never allocate huge pages
    Never,
    /// Advisory allocation
    Advisory,
}
/// Memory allocation policy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemoryAllocationPolicy {
    /// Strict memory allocation
    Strict,
    /// Overcommit allowed
    Overcommit,
    /// Adaptive allocation
    Adaptive,
    /// Balloon allocation
    Balloon,
}
/// GPU configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuConfig {
    /// Maximum GPU devices
    pub max_devices: Option<usize>,
    /// GPU memory limit per device
    pub memory_limit_per_device: Option<u64>,
    /// GPU utilization limit
    pub utilization_limit: Option<f32>,
    /// GPU scheduling priority
    pub priority: GpuPriority,
    /// CUDA configuration
    pub cuda_config: Option<CudaConfig>,
}
/// Alert types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AlertType {
    /// Resource usage exceeded threshold
    UsageExceeded,
    /// Resource exhausted
    ResourceExhausted,
    /// Performance degradation
    PerformanceDegradation,
    /// Resource unavailable
    ResourceUnavailable,
    /// Quota exceeded
    QuotaExceeded,
}
/// Resource alert
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceAlert {
    /// Alert ID
    pub alert_id: String,
    /// Alert type
    pub alert_type: AlertType,
    /// Alert severity
    pub severity: AlertSeverity,
    /// Alert message
    pub message: String,
    /// Resource involved
    pub resource: String,
    /// Current value
    pub current_value: f64,
    /// Threshold value
    pub threshold_value: f64,
    /// Alert timestamp
    pub timestamp: SystemTime,
    /// Alert status
    pub status: AlertStatus,
}
/// CUDA configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CudaConfig {
    /// CUDA version
    pub version: String,
    /// Compute capability
    pub compute_capability: (u32, u32),
    /// Maximum threads per block
    pub max_threads_per_block: u32,
    /// Maximum blocks per grid
    pub max_blocks_per_grid: (u32, u32, u32),
}
/// Huge page sizes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HugePageSize {
    /// 2MB pages
    Size2MB,
    /// 1GB pages
    Size1GB,
}
/// Quota usage tracking
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct QuotaUsage {
    /// Current usage
    pub current_usage: f64,
    /// Peak usage in period
    pub peak_usage: f64,
    /// Usage history
    pub usage_history: Vec<UsagePoint>,
    /// Last reset timestamp
    pub last_reset: SystemTime,
    /// Next reset timestamp
    pub next_reset: SystemTime,
    /// Quota exceeded flag
    pub exceeded: bool,
}
/// GPU priorities
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum GpuPriority {
    /// Low priority
    Low = 1,
    /// Normal priority
    Normal = 5,
    /// High priority
    High = 10,
}
/// Traffic shaping algorithms
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrafficShapingAlgorithm {
    /// Token bucket algorithm
    TokenBucket,
    /// Leaky bucket algorithm
    LeakyBucket,
    /// Hierarchical token bucket
    HTB,
    /// Fair queuing
    FairQueuing,
}
/// Resource utilization
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ResourceUtilization {
    /// CPU utilization percentage
    pub cpu_utilization: f32,
    /// Memory utilization percentage
    pub memory_utilization: f32,
    /// Storage utilization percentage
    pub storage_utilization: f32,
    /// Network utilization percentage
    pub network_utilization: f32,
    /// GPU utilization percentage
    pub gpu_utilization: f32,
    /// Custom resource utilization
    pub custom_utilization: HashMap<String, f32>,
}
/// Billing record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BillingRecord {
    /// Record ID
    pub record_id: String,
    /// Resource type
    pub resource_type: String,
    /// Usage amount
    pub usage_amount: f64,
    /// Rate per unit
    pub rate: f64,
    /// Total cost
    pub cost: f64,
    /// Billing timestamp
    pub timestamp: SystemTime,
}
/// Memory limits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryLimits {
    /// Maximum resident set size
    pub max_rss: Option<usize>,
    /// Maximum virtual memory size
    pub max_vm: Option<usize>,
    /// Maximum stack size
    pub max_stack: Option<usize>,
    /// Maximum heap size
    pub max_heap: Option<usize>,
}
/// Resource quotas
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceQuotas {
    /// CPU quota in cores-hours
    pub cpu_quota: Option<f64>,
    /// Memory quota in byte-hours
    pub memory_quota: Option<u64>,
    /// Storage quota in byte-hours
    pub storage_quota: Option<u64>,
    /// Network quota in byte-hours
    pub network_quota: Option<u64>,
    /// GPU quota in device-hours
    pub gpu_quota: Option<f64>,
    /// Custom resource quotas
    pub custom_quotas: HashMap<String, f64>,
    /// Quota period
    pub quota_period: Duration,
    /// Quota reset policy
    pub reset_policy: QuotaResetPolicy,
}
/// Alert thresholds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertThresholds {
    /// CPU usage threshold percentage
    pub cpu_usage: f32,
    /// Memory usage threshold percentage
    pub memory_usage: f32,
    /// Storage usage threshold percentage
    pub storage_usage: f32,
    /// Network usage threshold percentage
    pub network_usage: f32,
    /// GPU usage threshold percentage
    pub gpu_usage: Option<f32>,
}
/// Scheduler metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SchedulerMetrics {
    /// Total requests processed
    pub total_requests: u64,
    /// Average wait time
    pub avg_wait_time: Duration,
    /// Average turnaround time
    pub avg_turnaround_time: Duration,
    /// Throughput (requests per second)
    pub throughput: f32,
    /// Queue length
    pub queue_length: usize,
    /// Utilization efficiency
    pub utilization_efficiency: f32,
}
/// Quota manager
#[derive(Debug)]
pub struct QuotaManager {
    /// Quota definitions
    quotas: Arc<RwLock<HashMap<String, Quota>>>,
    /// Quota usage tracking
    usage: Arc<Mutex<HashMap<String, QuotaUsage>>>,
    /// Billing information
    billing: Arc<Mutex<BillingInfo>>,
}
impl QuotaManager {
    /// Create a new quota manager
    pub fn new() -> Self {
        Self {
            quotas: Arc::new(RwLock::new(HashMap::new())),
            usage: Arc::new(Mutex::new(HashMap::new())),
            billing: Arc::new(Mutex::new(BillingInfo::default())),
        }
    }
    /// Create quota
    pub fn create_quota(&self, quota: Quota) -> ContextResult<()> {
        let mut quotas = self.quotas.write().unwrap();
        quotas.insert(quota.quota_id.clone(), quota);
        Ok(())
    }
    /// Update quota usage
    pub fn update_usage(&self, quota_id: &str, usage_delta: f64) -> ContextResult<()> {
        let mut usage = self.usage.lock().unwrap();
        let quota_usage = usage
            .entry(quota_id.to_string())
            .or_insert_with(QuotaUsage::default);
        quota_usage.current_usage += usage_delta;
        quota_usage.peak_usage = quota_usage.peak_usage.max(quota_usage.current_usage);
        quota_usage
            .usage_history
            .push(UsagePoint {
                value: quota_usage.current_usage,
                timestamp: SystemTime::now(),
            });
        Ok(())
    }
    /// Check quota limits
    pub fn check_quota(&self, quota_id: &str) -> ContextResult<bool> {
        let quotas = self.quotas.read().unwrap();
        let usage = self.usage.lock().unwrap();
        if let (Some(quota), Some(quota_usage)) = (
            quotas.get(quota_id),
            usage.get(quota_id),
        ) {
            Ok(quota_usage.current_usage <= quota.limit)
        } else {
            Err(ContextError::not_found(quota_id))
        }
    }
}
/// Pool status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PoolStatus {
    /// Pool active
    Active,
    /// Pool inactive
    Inactive,
    /// Pool maintenance
    Maintenance,
    /// Pool drained
    Drained,
}
/// Resource limits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceLimits {
    /// CPU limits
    pub cpu: CpuLimits,
    /// Memory limits
    pub memory: MemoryLimits,
    /// Storage limits
    pub storage: StorageLimits,
    /// Network limits
    pub network: NetworkLimits,
    /// GPU limits
    pub gpu: Option<GpuLimits>,
    /// Custom resource limits
    pub custom: HashMap<String, f64>,
}
/// Resource monitor
#[derive(Debug)]
pub struct ResourceMonitor {
    /// Monitoring enabled
    enabled: Arc<RwLock<bool>>,
    /// Monitoring interval
    interval: Arc<RwLock<Duration>>,
    /// Current usage
    current_usage: Arc<Mutex<ResourceUsage>>,
    /// Usage history
    usage_history: Arc<Mutex<VecDeque<ResourceUsageSnapshot>>>,
    /// Alert manager
    alert_manager: Arc<Mutex<AlertManager>>,
}
impl ResourceMonitor {
    /// Create a new resource monitor
    pub fn new() -> Self {
        Self {
            enabled: Arc::new(RwLock::new(true)),
            interval: Arc::new(RwLock::new(Duration::from_secs(60))),
            current_usage: Arc::new(Mutex::new(ResourceUsage::default())),
            usage_history: Arc::new(Mutex::new(VecDeque::new())),
            alert_manager: Arc::new(Mutex::new(AlertManager::default())),
        }
    }
    /// Get current usage
    pub fn get_current_usage(&self) -> ContextResult<ResourceUsage> {
        let usage = self.current_usage.lock().unwrap();
        Ok(usage.clone())
    }
    /// Update current usage
    pub fn update_usage(&self, usage: ResourceUsage) -> ContextResult<()> {
        let mut current = self.current_usage.lock().unwrap();
        *current = usage.clone();
        let mut history = self.usage_history.lock().unwrap();
        history
            .push_back(ResourceUsageSnapshot {
                usage,
                timestamp: SystemTime::now(),
                duration: Duration::from_secs(60),
            });
        if history.len() > 1000 {
            history.pop_front();
        }
        Ok(())
    }
    /// Get usage history
    pub fn get_usage_history(
        &self,
        limit: Option<usize>,
    ) -> ContextResult<Vec<ResourceUsageSnapshot>> {
        let history = self.usage_history.lock().unwrap();
        match limit {
            Some(n) => Ok(history.iter().rev().take(n).cloned().collect()),
            None => Ok(history.iter().cloned().collect()),
        }
    }
}
/// Alert status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AlertStatus {
    /// Alert active
    Active,
    /// Alert acknowledged
    Acknowledged,
    /// Alert resolved
    Resolved,
    /// Alert suppressed
    Suppressed,
}
/// Resource allocation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceAllocation {
    /// Allocation ID
    pub allocation_id: String,
    /// Context ID that owns this allocation
    pub context_id: String,
    /// Allocated CPU cores
    pub cpu_cores: f32,
    /// Allocated memory in bytes
    pub memory: u64,
    /// Allocated storage in bytes
    pub storage: u64,
    /// Allocated network bandwidth
    pub network_bandwidth: u64,
    /// Allocated GPU devices
    pub gpu_devices: usize,
    /// Custom resource allocations
    pub custom: HashMap<String, f64>,
    /// Allocation timestamp
    pub allocated_at: SystemTime,
    /// Allocation duration
    pub duration: Option<Duration>,
    /// Allocation priority
    pub priority: ContextPriority,
    /// Allocation status
    pub status: AllocationStatus,
}
/// Traffic shaping configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrafficShapingConfig {
    /// Enable traffic shaping
    pub enabled: bool,
    /// Ingress bandwidth limit
    pub ingress_limit: Option<u64>,
    /// Egress bandwidth limit
    pub egress_limit: Option<u64>,
    /// Burst size
    pub burst_size: Option<u64>,
    /// Shaping algorithm
    pub algorithm: TrafficShapingAlgorithm,
}
/// Storage configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    /// Maximum storage space in bytes
    pub max_storage: Option<u64>,
    /// Temporary storage limit
    pub temp_storage_limit: Option<u64>,
    /// I/O operations per second limit
    pub iops_limit: Option<u32>,
    /// Storage bandwidth limit in bytes/second
    pub bandwidth_limit: Option<u64>,
    /// Storage priority
    pub priority: StoragePriority,
    /// File system quotas
    pub fs_quotas: HashMap<String, u64>,
    /// Storage encryption
    pub encryption: StorageEncryptionConfig,
}
/// CPU throttling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuThrottling {
    /// Enable CPU throttling
    pub enabled: bool,
    /// Maximum CPU usage percentage
    pub max_usage_percent: f32,
    /// Throttling period
    pub period: Duration,
    /// Throttling quota
    pub quota: Duration,
}
/// Resource usage snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUsageSnapshot {
    /// Usage data
    pub usage: ResourceUsage,
    /// Snapshot timestamp
    pub timestamp: SystemTime,
    /// Snapshot duration
    pub duration: Duration,
}
/// Quota definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Quota {
    /// Quota ID
    pub quota_id: String,
    /// Context ID
    pub context_id: String,
    /// Resource type
    pub resource_type: String,
    /// Quota limit
    pub limit: f64,
    /// Quota period
    pub period: Duration,
    /// Quota reset policy
    pub reset_policy: QuotaResetPolicy,
    /// Soft limit (warning threshold)
    pub soft_limit: Option<f64>,
    /// Quota enabled
    pub enabled: bool,
}
/// Alert rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertRule {
    /// Rule ID
    pub rule_id: String,
    /// Rule name
    pub name: String,
    /// Resource to monitor
    pub resource: String,
    /// Metric to check
    pub metric: String,
    /// Threshold value
    pub threshold: f64,
    /// Comparison operator
    pub operator: ComparisonOperator,
    /// Alert severity
    pub severity: AlertSeverity,
    /// Rule enabled
    pub enabled: bool,
}
/// Resource metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ResourceMetrics {
    /// Total resource allocations
    pub total_allocations: u64,
    /// Active allocations
    pub active_allocations: u64,
    /// Failed allocations
    pub failed_allocations: u64,
    /// Resource utilization
    pub utilization: ResourceUtilization,
    /// Manager metrics
    pub manager_metrics: ResourceManagerMetrics,
    /// Scheduler metrics
    pub scheduler_metrics: SchedulerMetrics,
    /// Alert count by severity
    pub alerts_by_severity: HashMap<AlertSeverity, u64>,
    /// Quota usage summary
    pub quota_usage_summary: HashMap<String, f64>,
}
/// Scheduler policies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerPolicies {
    /// Primary scheduling policy
    pub primary_policy: SchedulingPolicy,
    /// Preemption enabled
    pub preemption_enabled: bool,
    /// Fair share weights
    pub fair_share_weights: HashMap<String, f32>,
    /// Priority levels
    pub priority_levels: Vec<PriorityLevel>,
    /// Backfill enabled
    pub backfill_enabled: bool,
}
/// Network configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConfig {
    /// Maximum bandwidth in bytes/second
    pub max_bandwidth: Option<u64>,
    /// Network traffic shaping
    pub traffic_shaping: TrafficShapingConfig,
    /// Network isolation
    pub isolation: NetworkIsolationConfig,
    /// Quality of Service
    pub qos: QosConfig,
    /// Connection limits
    pub connection_limits: ConnectionLimits,
}
/// Storage priorities
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum StoragePriority {
    /// Low priority
    Low = 1,
    /// Normal priority
    Normal = 5,
    /// High priority
    High = 10,
    /// Critical priority
    Critical = 15,
}
/// Resource context for comprehensive resource management
#[derive(Debug)]
pub struct ResourceContext {
    /// Context identifier
    context_id: String,
    /// Resource configuration
    config: Arc<RwLock<ResourceConfig>>,
    /// Resource manager
    resource_manager: Arc<ResourceManager>,
    /// Resource monitor
    monitor: Arc<ResourceMonitor>,
    /// Quota manager
    quota_manager: Arc<QuotaManager>,
    /// Resource scheduler
    scheduler: Arc<ResourceScheduler>,
    /// Resource pools
    pools: Arc<RwLock<HashMap<String, ResourcePool>>>,
    /// Active allocations
    allocations: Arc<RwLock<HashMap<String, ResourceAllocation>>>,
    /// Context state
    state: Arc<RwLock<ContextState>>,
    /// Metadata
    metadata: Arc<RwLock<ContextMetadata>>,
    /// Resource metrics
    metrics: Arc<Mutex<ResourceMetrics>>,
}
impl ResourceContext {
    /// Create a new resource context
    pub fn new(context_id: String) -> ContextResult<Self> {
        let context = Self {
            context_id: context_id.clone(),
            config: Arc::new(RwLock::new(ResourceConfig::default())),
            resource_manager: Arc::new(ResourceManager::new()),
            monitor: Arc::new(ResourceMonitor::new()),
            quota_manager: Arc::new(QuotaManager::new()),
            scheduler: Arc::new(ResourceScheduler::new()),
            pools: Arc::new(RwLock::new(HashMap::new())),
            allocations: Arc::new(RwLock::new(HashMap::new())),
            state: Arc::new(RwLock::new(ContextState::Initializing)),
            metadata: Arc::new(RwLock::new(ContextMetadata::default())),
            metrics: Arc::new(Mutex::new(ResourceMetrics::default())),
        };
        *context.state.write().unwrap() = ContextState::Active;
        Ok(context)
    }
    /// Create resource context with custom configuration
    pub fn with_config(
        context_id: String,
        config: ResourceConfig,
    ) -> ContextResult<Self> {
        let mut context = Self::new(context_id)?;
        *context.config.write().unwrap() = config;
        Ok(context)
    }
    /// Allocate resources
    pub fn allocate_resources(
        &self,
        requirements: ResourceRequirement,
    ) -> ContextResult<String> {
        self.resource_manager.allocate(self.id(), requirements)
    }
    /// Release resource allocation
    pub fn release_allocation(&self, allocation_id: &str) -> ContextResult<()> {
        self.resource_manager.release(allocation_id)
    }
    /// Get current resource usage
    pub fn get_resource_usage(&self) -> ContextResult<ResourceUsage> {
        self.monitor.get_current_usage()
    }
    /// Get resource metrics
    pub fn get_resource_metrics(&self) -> ContextResult<ResourceMetrics> {
        let metrics = self.metrics.lock().unwrap();
        Ok(metrics.clone())
    }
    /// Create resource pool
    pub fn create_pool(&self, pool_id: String, pool: ResourcePool) -> ContextResult<()> {
        let mut pools = self.pools.write().unwrap();
        pools.insert(pool_id, pool);
        Ok(())
    }
    /// Get resource pool
    pub fn get_pool(&self, pool_id: &str) -> ContextResult<Option<ResourcePool>> {
        let pools = self.pools.read().unwrap();
        Ok(pools.get(pool_id).cloned())
    }
    /// Update resource configuration
    pub fn update_config<F>(&self, updater: F) -> ContextResult<()>
    where
        F: FnOnce(&mut ResourceConfig) -> ContextResult<()>,
    {
        let mut config = self.config.write().unwrap();
        updater(&mut *config)
    }
    /// Get resource configuration
    pub fn get_config(&self) -> ContextResult<ResourceConfig> {
        let config = self.config.read().unwrap();
        Ok(config.clone())
    }
    /// Check resource limits
    pub fn check_resource_limits(&self) -> ContextResult<()> {
        let config = self.config.read().unwrap();
        let usage = self.monitor.get_current_usage()?;
        if let Some(max_cores) = config.limits.cpu.max_cores {
            if usage.cpu_usage > max_cores {
                return Err(
                    ContextError::custom(
                        "cpu_limit_exceeded",
                        format!(
                            "CPU usage {} exceeds limit {}", usage.cpu_usage, max_cores
                        ),
                    ),
                );
            }
        }
        if let Some(max_memory) = config.limits.memory.max_rss {
            if usage.memory_usage as usize > max_memory {
                return Err(
                    ContextError::custom(
                        "memory_limit_exceeded",
                        format!(
                            "Memory usage {} exceeds limit {}", usage.memory_usage,
                            max_memory
                        ),
                    ),
                );
            }
        }
        if let Some(max_storage) = config.limits.storage.max_space {
            if usage.storage_usage > max_storage {
                return Err(
                    ContextError::custom(
                        "storage_limit_exceeded",
                        format!(
                            "Storage usage {} exceeds limit {}", usage.storage_usage,
                            max_storage
                        ),
                    ),
                );
            }
        }
        Ok(())
    }
}
/// Memory limit enforcement modes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemoryLimitEnforcement {
    /// No enforcement
    None,
    /// Soft limit (warnings only)
    Soft,
    /// Hard limit (strict enforcement)
    Strict,
    /// Adaptive limit (dynamic adjustment)
    Adaptive,
}
/// Resource manager
#[derive(Debug)]
pub struct ResourceManager {
    /// Available resources
    available_resources: Arc<RwLock<AvailableResources>>,
    /// Resource allocations
    allocations: Arc<RwLock<HashMap<String, ResourceAllocation>>>,
    /// Resource reservations
    reservations: Arc<RwLock<HashMap<String, ResourceReservation>>>,
    /// Allocation policies
    policies: Arc<RwLock<AllocationPolicies>>,
    /// Resource metrics
    metrics: Arc<Mutex<ResourceManagerMetrics>>,
}
impl ResourceManager {
    /// Create a new resource manager
    pub fn new() -> Self {
        Self {
            available_resources: Arc::new(RwLock::new(AvailableResources::default())),
            allocations: Arc::new(RwLock::new(HashMap::new())),
            reservations: Arc::new(RwLock::new(HashMap::new())),
            policies: Arc::new(RwLock::new(AllocationPolicies::default())),
            metrics: Arc::new(Mutex::new(ResourceManagerMetrics::default())),
        }
    }
    /// Allocate resources
    pub fn allocate(
        &self,
        context_id: &str,
        requirements: ResourceRequirement,
    ) -> ContextResult<String> {
        let allocation_id = Uuid::new_v4().to_string();
        let available = self.available_resources.read().unwrap();
        if let Some(cpu_req) = requirements.cpu_cores {
            if available.cpu_cores < cpu_req {
                return Err(
                    ContextError::custom(
                        "insufficient_cpu",
                        "Not enough CPU cores available",
                    ),
                );
            }
        }
        if let Some(memory_req) = requirements.memory {
            if available.memory < memory_req {
                return Err(
                    ContextError::custom(
                        "insufficient_memory",
                        "Not enough memory available",
                    ),
                );
            }
        }
        drop(available);
        let allocation = ResourceAllocation {
            allocation_id: allocation_id.clone(),
            context_id: context_id.to_string(),
            cpu_cores: requirements.cpu_cores.unwrap_or(0.0),
            memory: requirements.memory.unwrap_or(0),
            storage: requirements.storage.unwrap_or(0),
            network_bandwidth: requirements.network_bandwidth.unwrap_or(0),
            gpu_devices: requirements.gpu_devices.unwrap_or(0),
            custom: requirements.custom,
            allocated_at: SystemTime::now(),
            duration: None,
            priority: ContextPriority::Normal,
            status: AllocationStatus::Active,
        };
        let mut allocations = self.allocations.write().unwrap();
        allocations.insert(allocation_id.clone(), allocation.clone());
        let mut available = self.available_resources.write().unwrap();
        available.cpu_cores -= allocation.cpu_cores;
        available.memory -= allocation.memory;
        available.storage -= allocation.storage;
        available.network_bandwidth -= allocation.network_bandwidth;
        available.gpu_devices -= allocation.gpu_devices;
        available.last_updated = SystemTime::now();
        let mut metrics = self.metrics.lock().unwrap();
        metrics.total_allocations += 1;
        metrics.active_allocations += 1;
        Ok(allocation_id)
    }
    /// Release allocation
    pub fn release(&self, allocation_id: &str) -> ContextResult<()> {
        let mut allocations = self.allocations.write().unwrap();
        if let Some(mut allocation) = allocations.remove(allocation_id) {
            allocation.status = AllocationStatus::Released;
            let mut available = self.available_resources.write().unwrap();
            available.cpu_cores += allocation.cpu_cores;
            available.memory += allocation.memory;
            available.storage += allocation.storage;
            available.network_bandwidth += allocation.network_bandwidth;
            available.gpu_devices += allocation.gpu_devices;
            available.last_updated = SystemTime::now();
            let mut metrics = self.metrics.lock().unwrap();
            metrics.active_allocations -= 1;
            Ok(())
        } else {
            Err(ContextError::not_found(allocation_id))
        }
    }
    /// Get allocation
    pub fn get_allocation(
        &self,
        allocation_id: &str,
    ) -> ContextResult<Option<ResourceAllocation>> {
        let allocations = self.allocations.read().unwrap();
        Ok(allocations.get(allocation_id).cloned())
    }
    /// Get available resources
    pub fn get_available_resources(&self) -> ContextResult<AvailableResources> {
        let available = self.available_resources.read().unwrap();
        Ok(available.clone())
    }
}
/// Allocation status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AllocationStatus {
    /// Allocation requested
    Requested,
    /// Allocation pending
    Pending,
    /// Allocation active
    Active,
    /// Allocation suspended
    Suspended,
    /// Allocation released
    Released,
    /// Allocation failed
    Failed,
}
/// NUMA node preferences
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NumaPreference {
    /// Any NUMA node
    Any,
    /// Specific NUMA node
    Node(u32),
    /// Local NUMA node preferred
    Local,
    /// Interleaved across nodes
    Interleaved,
}
/// Memory allocation strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemoryAllocationStrategy {
    /// First fit allocation
    FirstFit,
    /// Best fit allocation
    BestFit,
    /// Worst fit allocation
    WorstFit,
    /// Buddy allocation
    Buddy,
    /// Slab allocation
    Slab,
}
/// Reservation status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReservationStatus {
    /// Reservation pending
    Pending,
    /// Reservation confirmed
    Confirmed,
    /// Reservation active
    Active,
    /// Reservation completed
    Completed,
    /// Reservation cancelled
    Cancelled,
    /// Reservation expired
    Expired,
}
/// Allocation policies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllocationPolicies {
    /// Default allocation policy
    pub default_policy: AllocationPolicy,
    /// CPU allocation policy
    pub cpu_policy: CpuAllocationPolicy,
    /// Memory allocation policy
    pub memory_policy: MemoryAllocationPolicy,
    /// Storage allocation policy
    pub storage_policy: StorageAllocationPolicy,
    /// Network allocation policy
    pub network_policy: NetworkAllocationPolicy,
    /// Overcommit policies
    pub overcommit: OvercommitPolicies,
}
/// Firewall actions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FirewallAction {
    /// Allow traffic
    Allow,
    /// Deny traffic
    Deny,
    /// Drop traffic silently
    Drop,
    /// Log traffic
    Log,
}
/// Storage limits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageLimits {
    /// Maximum storage space
    pub max_space: Option<u64>,
    /// Maximum number of files
    pub max_files: Option<u64>,
    /// Maximum I/O operations per second
    pub max_iops: Option<u32>,
    /// Maximum bandwidth
    pub max_bandwidth: Option<u64>,
}
/// Preemption policies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PreemptionPolicy {
    /// Non-preemptive
    NonPreemptive,
    /// Preemptive
    Preemptive,
    /// Cooperative
    Cooperative,
}
/// Comparison operators
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ComparisonOperator {
    /// Greater than
    GreaterThan,
    /// Greater than or equal
    GreaterThanOrEqual,
    /// Less than
    LessThan,
    /// Less than or equal
    LessThanOrEqual,
    /// Equal
    Equal,
    /// Not equal
    NotEqual,
}
/// Resource pool
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourcePool {
    /// Pool ID
    pub pool_id: String,
    /// Pool name
    pub name: String,
    /// Pool description
    pub description: Option<String>,
    /// Pool resources
    pub resources: AvailableResources,
    /// Pool policies
    pub policies: AllocationPolicies,
    /// Pool status
    pub status: PoolStatus,
    /// Associated contexts
    pub contexts: HashSet<String>,
}
/// Scheduling policies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SchedulingPolicy {
    /// First-come, first-served
    FCFS,
    /// Shortest job first
    SJF,
    /// Round robin
    RoundRobin,
    /// Priority scheduling
    Priority,
    /// Fair scheduling
    Fair,
    /// Lottery scheduling
    Lottery,
    /// Completely fair scheduler
    CFS,
}
/// GPU limits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuLimits {
    /// Maximum number of devices
    pub max_devices: Option<usize>,
    /// Maximum memory per device
    pub max_memory_per_device: Option<u64>,
    /// Maximum utilization percentage
    pub max_utilization: Option<f32>,
}
/// Network allocation policy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NetworkAllocationPolicy {
    /// Dedicated bandwidth
    Dedicated,
    /// Shared bandwidth
    Shared,
    /// Guaranteed minimum
    GuaranteedMin,
    /// Best effort
    BestEffort,
}
/// CPU limits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuLimits {
    /// Maximum CPU cores
    pub max_cores: Option<f32>,
    /// Maximum CPU usage percentage
    pub max_usage_percent: Option<f32>,
    /// CPU time limit
    pub cpu_time_limit: Option<Duration>,
}
/// Resource configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceConfig {
    /// CPU configuration
    pub cpu_config: CpuConfig,
    /// Memory configuration
    pub memory_config: MemoryConfig,
    /// Storage configuration
    pub storage_config: StorageConfig,
    /// Network configuration
    pub network_config: NetworkConfig,
    /// GPU configuration
    pub gpu_config: Option<GpuConfig>,
    /// Custom resource configurations
    pub custom_resources: HashMap<String, CustomResourceConfig>,
    /// Resource limits
    pub limits: ResourceLimits,
    /// Resource quotas
    pub quotas: ResourceQuotas,
    /// Monitoring configuration
    pub monitoring: MonitoringConfig,
    /// Scheduling configuration
    pub scheduling: SchedulingConfig,
}
/// Available resources
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AvailableResources {
    /// Available CPU cores
    pub cpu_cores: f32,
    /// Available memory in bytes
    pub memory: u64,
    /// Available storage in bytes
    pub storage: u64,
    /// Available network bandwidth
    pub network_bandwidth: u64,
    /// Available GPU devices
    pub gpu_devices: usize,
    /// Custom resources
    pub custom: HashMap<String, f64>,
    /// Last updated timestamp
    pub last_updated: SystemTime,
}
/// Memory swapping policies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SwapPolicy {
    /// No swapping allowed
    NoSwap,
    /// Allow swapping
    Allow,
    /// Automatic swapping
    Auto,
    /// Aggressive swapping
    Aggressive,
}
