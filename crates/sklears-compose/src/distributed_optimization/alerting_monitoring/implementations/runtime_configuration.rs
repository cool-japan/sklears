use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// Comprehensive runtime configuration system providing performance optimization,
/// resource allocation management, error handling strategies, and runtime monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeConfig {
    /// Performance configuration
    pub performance: RuntimePerformanceConfig,
    /// Resource allocation
    pub resource_allocation: ResourceAllocationConfig,
    /// Error handling
    pub error_handling: RuntimeErrorHandlingConfig,
    /// Monitoring
    pub monitoring: RuntimeMonitoringConfig,
}

/// Runtime performance configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimePerformanceConfig {
    /// Optimization level
    pub optimization_level: RuntimeOptimizationLevel,
    /// Garbage collection configuration
    pub gc_config: GarbageCollectionConfig,
    /// Thread pool configuration
    pub thread_pool: ThreadPoolConfig,
}

/// Runtime optimization levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RuntimeOptimizationLevel {
    /// Debug mode (no optimization)
    Debug,
    /// Development mode (basic optimization)
    Development,
    /// Production mode (full optimization)
    Production,
    /// Custom optimization level
    Custom(String),
}

/// Garbage collection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GarbageCollectionConfig {
    /// GC algorithm
    pub algorithm: GCAlgorithm,
    /// GC tuning parameters
    pub tuning: GCTuning,
}

/// Garbage collection algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GCAlgorithm {
    /// Serial GC
    Serial,
    /// Parallel GC
    Parallel,
    /// Concurrent Mark Sweep
    CMS,
    /// G1 GC
    G1,
    /// Custom GC
    Custom(String),
}

/// GC tuning parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GCTuning {
    /// Heap size
    pub heap_size_mb: usize,
    /// Young generation size
    pub young_gen_size_mb: usize,
    /// GC frequency threshold
    pub frequency_threshold: f64,
}

/// Thread pool configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreadPoolConfig {
    /// Core pool size
    pub core_size: usize,
    /// Maximum pool size
    pub max_size: usize,
    /// Keep alive time
    pub keep_alive: Duration,
    /// Queue capacity
    pub queue_capacity: usize,
}

/// Resource allocation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceAllocationConfig {
    /// CPU allocation
    pub cpu_allocation: CPUAllocationConfig,
    /// Memory allocation
    pub memory_allocation: MemoryAllocationConfig,
    /// Storage allocation
    pub storage_allocation: StorageAllocationConfig,
    /// Network allocation
    pub network_allocation: NetworkAllocationConfig,
}

/// CPU allocation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CPUAllocationConfig {
    /// CPU cores allocated
    pub cores: u32,
    /// CPU affinity
    pub affinity: CPUAffinity,
    /// Priority
    pub priority: ProcessPriority,
}

/// CPU affinity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CPUAffinity {
    /// No affinity
    None,
    /// Specific cores
    Cores(Vec<u32>),
    /// NUMA node
    NumaNode(u32),
    /// Custom affinity
    Custom(String),
}

/// Process priority
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProcessPriority {
    /// Low priority
    Low,
    /// Normal priority
    Normal,
    /// High priority
    High,
    /// Real-time priority
    RealTime,
}

/// Memory allocation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryAllocationConfig {
    /// Memory limit in megabytes
    pub limit_mb: usize,
    /// Memory pools
    pub pools: Vec<MemoryPool>,
    /// Garbage collection configuration
    pub gc_config: GarbageCollectionConfig,
}

/// Memory pool
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryPool {
    /// Pool name
    pub name: String,
    /// Pool size in megabytes
    pub size_mb: usize,
    /// Pool type
    pub pool_type: MemoryPoolType,
}

/// Memory pool types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MemoryPoolType {
    /// Heap memory pool
    Heap,
    /// Stack memory pool
    Stack,
    /// Direct memory pool
    Direct,
    /// Custom memory pool
    Custom(String),
}

/// Storage allocation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageAllocationConfig {
    /// Storage tiers
    pub tiers: Vec<StorageTier>,
    /// Default tier
    pub default_tier: String,
    /// Allocation strategy
    pub allocation_strategy: StorageAllocationStrategy,
}

/// Storage tier
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageTier {
    /// Tier name
    pub name: String,
    /// Tier type
    pub tier_type: StorageTierType,
    /// Capacity in gigabytes
    pub capacity_gb: u64,
    /// Performance characteristics
    pub performance: StoragePerformance,
}

/// Storage tier types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StorageTierType {
    /// High-performance storage (SSD)
    HighPerformance,
    /// Standard storage (HDD)
    Standard,
    /// Archive storage
    Archive,
    /// Cloud storage
    Cloud,
    /// Custom storage
    Custom(String),
}

/// Storage performance characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoragePerformance {
    /// Read IOPS
    pub read_iops: u32,
    /// Write IOPS
    pub write_iops: u32,
    /// Latency in milliseconds
    pub latency_ms: f64,
    /// Throughput in MB/s
    pub throughput_mbps: u32,
}

/// Storage allocation strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StorageAllocationStrategy {
    /// First available
    FirstAvailable,
    /// Best fit
    BestFit,
    /// Performance optimized
    PerformanceOptimized,
    /// Cost optimized
    CostOptimized,
    /// Custom strategy
    Custom(String),
}

/// Network allocation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkAllocationConfig {
    /// Bandwidth allocation
    pub bandwidth: BandwidthAllocation,
    /// Quality of Service
    pub qos: QoSConfig,
    /// Network interfaces
    pub interfaces: Vec<NetworkInterface>,
}

/// Bandwidth allocation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BandwidthAllocation {
    /// Total bandwidth in Mbps
    pub total_mbps: u32,
    /// Ingress limit in Mbps
    pub ingress_limit_mbps: u32,
    /// Egress limit in Mbps
    pub egress_limit_mbps: u32,
}

/// Quality of Service configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QoSConfig {
    /// Traffic classes
    pub traffic_classes: Vec<TrafficClass>,
    /// Priority queuing
    pub priority_queuing: PriorityQueuingConfig,
    /// Rate limiting
    pub rate_limiting: NetworkRateLimitingConfig,
}

/// Traffic class
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrafficClass {
    /// Class name
    pub name: String,
    /// Priority level
    pub priority: u8,
    /// Bandwidth guarantee
    pub bandwidth_guarantee_mbps: u32,
    /// Maximum bandwidth
    pub max_bandwidth_mbps: u32,
}

/// Priority queuing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriorityQueuingConfig {
    /// Queue levels
    pub levels: u8,
    /// Scheduling algorithm
    pub algorithm: SchedulingAlgorithm,
}

/// Scheduling algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SchedulingAlgorithm {
    /// Strict priority
    StrictPriority,
    /// Weighted fair queuing
    WeightedFairQueuing,
    /// Deficit round robin
    DeficitRoundRobin,
    /// Custom algorithm
    Custom(String),
}

/// Network rate limiting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkRateLimitingConfig {
    /// Rate limit per connection
    pub per_connection_mbps: u32,
    /// Rate limit per source IP
    pub per_source_ip_mbps: u32,
    /// Burst allowance
    pub burst_allowance_mb: u32,
}

/// Network interface
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkInterface {
    /// Interface name
    pub name: String,
    /// Interface type
    pub interface_type: InterfaceType,
    /// Configuration
    pub config: InterfaceConfig,
}

/// Interface types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InterfaceType {
    /// Ethernet interface
    Ethernet,
    /// Wi-Fi interface
    WiFi,
    /// Virtual interface
    Virtual,
    /// Custom interface
    Custom(String),
}

/// Interface configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterfaceConfig {
    /// IP address
    pub ip_address: String,
    /// Subnet mask
    pub subnet_mask: String,
    /// Gateway
    pub gateway: String,
    /// MTU size
    pub mtu: u16,
}

/// Runtime error handling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeErrorHandlingConfig {
    /// Error recovery strategies
    pub recovery_strategies: Vec<ErrorRecoveryStrategy>,
    /// Error aggregation
    pub aggregation: ErrorAggregationConfig,
    /// Error notification
    pub notification: ErrorNotificationConfig,
}

/// Error recovery strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorRecoveryStrategy {
    /// Strategy name
    pub name: String,
    /// Error types handled
    pub error_types: Vec<String>,
    /// Recovery actions
    pub actions: Vec<RecoveryAction>,
}

/// Recovery action
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecoveryAction {
    /// Retry operation
    Retry { max_attempts: u32, delay: Duration },
    /// Failover to backup
    Failover { backup_id: String },
    /// Restart component
    Restart { component_id: String },
    /// Custom action
    Custom(String),
}

/// Error aggregation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorAggregationConfig {
    /// Aggregation window
    pub window: Duration,
    /// Aggregation threshold
    pub threshold: u32,
    /// Aggregation actions
    pub actions: Vec<AggregationAction>,
}

/// Aggregation action
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AggregationAction {
    /// Generate summary alert
    SummaryAlert,
    /// Escalate to higher level
    Escalate,
    /// Custom action
    Custom(String),
}

/// Error notification configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorNotificationConfig {
    /// Notification channels
    pub channels: Vec<String>,
    /// Notification thresholds
    pub thresholds: ErrorThresholds,
    /// Notification templates
    pub templates: HashMap<String, String>,
}

/// Error thresholds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorThresholds {
    /// Error rate threshold (errors per minute)
    pub error_rate: f64,
    /// Critical error threshold
    pub critical_errors: u32,
    /// Error burst threshold
    pub error_burst: u32,
}

/// Runtime monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeMonitoringConfig {
    /// Health checks
    pub health_checks: Vec<RuntimeHealthCheck>,
    /// Performance metrics
    pub performance_metrics: Vec<PerformanceMetric>,
    /// Alerting rules
    pub alerting_rules: Vec<AlertingRule>,
}

/// Runtime health check
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeHealthCheck {
    /// Check name
    pub name: String,
    /// Check type
    pub check_type: HealthCheckType,
    /// Check interval
    pub interval: Duration,
    /// Failure threshold
    pub failure_threshold: u32,
}

/// Health check types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HealthCheckType {
    /// Liveness check
    Liveness,
    /// Readiness check
    Readiness,
    /// Custom check
    Custom(String),
}

/// Performance metric
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetric {
    /// Metric name
    pub name: String,
    /// Metric type
    pub metric_type: PerformanceMetricType,
    /// Collection interval
    pub interval: Duration,
    /// Retention period
    pub retention: Duration,
}

/// Performance metric types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PerformanceMetricType {
    /// Response time metric
    ResponseTime,
    /// Throughput metric
    Throughput,
    /// Error rate metric
    ErrorRate,
    /// Resource utilization metric
    ResourceUtilization,
    /// Custom metric
    Custom(String),
}

/// Alerting rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertingRule {
    /// Rule name
    pub name: String,
    /// Rule expression
    pub expression: String,
    /// Rule severity
    pub severity: AlertSeverity,
    /// Rule actions
    pub actions: Vec<AlertAction>,
}

/// Alert severity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertSeverity {
    /// Info level
    Info,
    /// Warning level
    Warning,
    /// Error level
    Error,
    /// Critical level
    Critical,
}

/// Alert action
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertAction {
    /// Send notification
    Notify { channel: String, template: String },
    /// Execute script
    Script { path: String, args: Vec<String> },
    /// Custom action
    Custom(String),
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            performance: RuntimePerformanceConfig {
                optimization_level: RuntimeOptimizationLevel::Production,
                gc_config: GarbageCollectionConfig {
                    algorithm: GCAlgorithm::G1,
                    tuning: GCTuning {
                        heap_size_mb: 2048,
                        young_gen_size_mb: 512,
                        frequency_threshold: 0.1,
                    },
                },
                thread_pool: ThreadPoolConfig {
                    core_size: 8,
                    max_size: 32,
                    keep_alive: Duration::from_secs(60),
                    queue_capacity: 1000,
                },
            },
            resource_allocation: ResourceAllocationConfig {
                cpu_allocation: CPUAllocationConfig {
                    cores: 4,
                    affinity: CPUAffinity::None,
                    priority: ProcessPriority::Normal,
                },
                memory_allocation: MemoryAllocationConfig {
                    limit_mb: 4096,
                    pools: vec![
                        MemoryPool {
                            name: "heap".to_string(),
                            size_mb: 2048,
                            pool_type: MemoryPoolType::Heap,
                        },
                        MemoryPool {
                            name: "direct".to_string(),
                            size_mb: 1024,
                            pool_type: MemoryPoolType::Direct,
                        },
                    ],
                    gc_config: GarbageCollectionConfig {
                        algorithm: GCAlgorithm::G1,
                        tuning: GCTuning {
                            heap_size_mb: 2048,
                            young_gen_size_mb: 512,
                            frequency_threshold: 0.1,
                        },
                    },
                },
                storage_allocation: StorageAllocationConfig {
                    tiers: vec![
                        StorageTier {
                            name: "ssd".to_string(),
                            tier_type: StorageTierType::HighPerformance,
                            capacity_gb: 1000,
                            performance: StoragePerformance {
                                read_iops: 50000,
                                write_iops: 30000,
                                latency_ms: 0.1,
                                throughput_mbps: 500,
                            },
                        }
                    ],
                    default_tier: "ssd".to_string(),
                    allocation_strategy: StorageAllocationStrategy::PerformanceOptimized,
                },
                network_allocation: NetworkAllocationConfig {
                    bandwidth: BandwidthAllocation {
                        total_mbps: 1000,
                        ingress_limit_mbps: 800,
                        egress_limit_mbps: 800,
                    },
                    qos: QoSConfig {
                        traffic_classes: vec![
                            TrafficClass {
                                name: "critical".to_string(),
                                priority: 1,
                                bandwidth_guarantee_mbps: 200,
                                max_bandwidth_mbps: 500,
                            }
                        ],
                        priority_queuing: PriorityQueuingConfig {
                            levels: 4,
                            algorithm: SchedulingAlgorithm::WeightedFairQueuing,
                        },
                        rate_limiting: NetworkRateLimitingConfig {
                            per_connection_mbps: 100,
                            per_source_ip_mbps: 200,
                            burst_allowance_mb: 10,
                        },
                    },
                    interfaces: Vec::new(),
                },
            },
            error_handling: RuntimeErrorHandlingConfig {
                recovery_strategies: Vec::new(),
                aggregation: ErrorAggregationConfig {
                    window: Duration::from_secs(300), // 5 minutes
                    threshold: 100,
                    actions: vec![AggregationAction::SummaryAlert],
                },
                notification: ErrorNotificationConfig {
                    channels: vec!["email".to_string()],
                    thresholds: ErrorThresholds {
                        error_rate: 10.0,
                        critical_errors: 5,
                        error_burst: 50,
                    },
                    templates: HashMap::new(),
                },
            },
            monitoring: RuntimeMonitoringConfig {
                health_checks: vec![
                    RuntimeHealthCheck {
                        name: "liveness".to_string(),
                        check_type: HealthCheckType::Liveness,
                        interval: Duration::from_secs(30),
                        failure_threshold: 3,
                    }
                ],
                performance_metrics: vec![
                    PerformanceMetric {
                        name: "response_time".to_string(),
                        metric_type: PerformanceMetricType::ResponseTime,
                        interval: Duration::from_secs(10),
                        retention: Duration::from_secs(24 * 3600), // 24 hours
                    }
                ],
                alerting_rules: Vec::new(),
            },
        }
    }
}