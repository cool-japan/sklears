//! Resource Monitoring Configuration
//!
//! This module contains all configuration structures related to resource monitoring,
//! capacity planning, and resource optimization. It provides comprehensive control
//! over system resource tracking and management.

use std::collections::HashMap;
use std::time::Duration;

/// Resource monitoring configuration
///
/// Controls all aspects of resource monitoring including CPU, memory, disk, network,
/// and other system resources. Provides capacity planning and optimization insights.
///
/// # Architecture
///
/// The resource monitoring system tracks multiple resource dimensions:
///
/// ```text
/// Resource Monitoring
/// ├── CPU Monitoring (cores, utilization, frequency)
/// ├── Memory Monitoring (RAM, swap, allocation patterns)
/// ├── Storage Monitoring (disk usage, I/O patterns)
/// ├── Network Monitoring (bandwidth, connections, latency)
/// ├── Capacity Planning (forecasting, scaling recommendations)
/// └── Resource Optimization (efficiency analysis, recommendations)
/// ```
///
/// # Resource Categories
///
/// Resources are organized into categories for efficient monitoring:
/// - Compute resources: CPU, GPU, processing units
/// - Memory resources: RAM, swap, cache
/// - Storage resources: Disk space, I/O bandwidth
/// - Network resources: Bandwidth, connections, latency
/// - Application resources: Threads, handles, connections
///
/// # Usage Examples
///
/// ## Production Resource Monitoring
/// ```rust
/// use sklears_compose::monitoring_config::ResourceMonitoringConfig;
///
/// let config = ResourceMonitoringConfig::production();
/// ```
///
/// ## Development Resource Monitoring
/// ```rust
/// let config = ResourceMonitoringConfig::development();
/// ```
///
/// ## Cloud Resource Monitoring
/// ```rust
/// let config = ResourceMonitoringConfig::cloud_optimized();
/// ```
#[derive(Debug, Clone)]
pub struct ResourceMonitoringConfig {
    /// Enable resource monitoring
    ///
    /// Global switch for all resource monitoring functionality.
    pub enabled: bool,

    /// Resource categories to monitor
    ///
    /// Defines which types of resources should be tracked.
    pub resource_types: Vec<ResourceType>,

    /// Resource thresholds and alerting
    ///
    /// Defines thresholds that trigger alerts when resource usage exceeds limits.
    pub thresholds: ResourceThresholds,

    /// Capacity planning configuration
    ///
    /// Controls forecasting and capacity planning features.
    pub capacity_planning: CapacityPlanningConfig,

    /// Resource optimization configuration
    ///
    /// Controls automated resource optimization and efficiency analysis.
    pub optimization: ResourceOptimizationConfig,

    /// Collection settings
    ///
    /// Controls how resource data is collected and processed.
    pub collection: ResourceCollectionConfig,

    /// Historical data retention
    ///
    /// Controls how long resource monitoring data is retained.
    pub retention: ResourceRetentionConfig,
}

/// Types of resources to monitor
///
/// Defines the categories of system resources that can be tracked
/// and analyzed for performance and capacity planning.
#[derive(Debug, Clone, PartialEq)]
pub enum ResourceType {
    /// CPU resources
    ///
    /// Monitors CPU usage, core utilization, frequency scaling,
    /// and processing efficiency across all available cores.
    Cpu {
        /// Monitor per-core utilization
        per_core: bool,
        /// Monitor CPU frequency scaling
        frequency_scaling: bool,
        /// Monitor CPU temperature (if available)
        temperature: bool,
    },

    /// Memory resources
    ///
    /// Tracks RAM usage, swap utilization, memory allocation patterns,
    /// and garbage collection behavior.
    Memory {
        /// Monitor detailed allocation patterns
        allocation_tracking: bool,
        /// Monitor swap usage
        swap_monitoring: bool,
        /// Monitor memory fragmentation
        fragmentation_analysis: bool,
    },

    /// Storage resources
    ///
    /// Monitors disk space usage, I/O operations, and storage performance
    /// across all mounted filesystems and storage devices.
    Storage {
        /// Monitor all filesystems
        all_filesystems: bool,
        /// Monitor I/O operations
        io_monitoring: bool,
        /// Monitor disk health (SMART data)
        health_monitoring: bool,
    },

    /// Network resources
    ///
    /// Tracks network bandwidth usage, connection counts, latency,
    /// and network interface statistics.
    Network {
        /// Monitor all network interfaces
        all_interfaces: bool,
        /// Monitor connection states
        connection_tracking: bool,
        /// Monitor network latency
        latency_monitoring: bool,
    },

    /// GPU resources (if available)
    ///
    /// Monitors GPU utilization, memory usage, and temperature
    /// for systems with dedicated graphics processing units.
    Gpu {
        /// Monitor GPU memory usage
        memory_monitoring: bool,
        /// Monitor GPU temperature
        temperature_monitoring: bool,
        /// Monitor GPU power consumption
        power_monitoring: bool,
    },

    /// Process-level resources
    ///
    /// Tracks resource usage by individual processes including
    /// CPU time, memory usage, and file handle consumption.
    Process {
        /// Monitor all processes
        all_processes: bool,
        /// Monitor file descriptors
        file_descriptors: bool,
        /// Monitor thread counts
        thread_monitoring: bool,
    },

    /// Container resources (if applicable)
    ///
    /// Monitors resource usage within containerized environments
    /// including container-specific limits and cgroup statistics.
    Container {
        /// Monitor all containers
        all_containers: bool,
        /// Monitor container limits
        limit_monitoring: bool,
        /// Monitor container health
        health_monitoring: bool,
    },

    /// Custom resource type
    ///
    /// User-defined resource category for application-specific monitoring.
    Custom {
        /// Resource name
        name: String,
        /// Resource description
        description: String,
        /// Collection method
        collection_method: String,
    },
}

/// Resource thresholds configuration
///
/// Defines warning and critical thresholds for different resource types
/// that trigger alerts and automated responses when exceeded.
#[derive(Debug, Clone)]
pub struct ResourceThresholds {
    /// CPU utilization thresholds
    pub cpu: CpuThresholds,

    /// Memory usage thresholds
    pub memory: MemoryThresholds,

    /// Storage usage thresholds
    pub storage: StorageThresholds,

    /// Network usage thresholds
    pub network: NetworkThresholds,

    /// GPU usage thresholds (if applicable)
    pub gpu: Option<GpuThresholds>,

    /// Custom resource thresholds
    pub custom: HashMap<String, CustomThresholds>,

    /// Global threshold settings
    pub global: GlobalThresholdSettings,
}

/// CPU threshold configuration
#[derive(Debug, Clone)]
pub struct CpuThresholds {
    /// Overall CPU utilization warning threshold (percentage)
    pub utilization_warning: f64,

    /// Overall CPU utilization critical threshold (percentage)
    pub utilization_critical: f64,

    /// Per-core utilization warning threshold (percentage)
    pub per_core_warning: f64,

    /// Per-core utilization critical threshold (percentage)
    pub per_core_critical: f64,

    /// Load average warning threshold
    pub load_average_warning: f64,

    /// Load average critical threshold
    pub load_average_critical: f64,

    /// CPU temperature warning threshold (Celsius)
    pub temperature_warning: Option<f64>,

    /// CPU temperature critical threshold (Celsius)
    pub temperature_critical: Option<f64>,
}

/// Memory threshold configuration
#[derive(Debug, Clone)]
pub struct MemoryThresholds {
    /// Memory usage warning threshold (percentage)
    pub usage_warning: f64,

    /// Memory usage critical threshold (percentage)
    pub usage_critical: f64,

    /// Available memory warning threshold (bytes)
    pub available_warning: u64,

    /// Available memory critical threshold (bytes)
    pub available_critical: u64,

    /// Swap usage warning threshold (percentage)
    pub swap_warning: f64,

    /// Swap usage critical threshold (percentage)
    pub swap_critical: f64,

    /// Memory leak detection threshold (bytes/hour)
    pub leak_detection_threshold: Option<u64>,
}

/// Storage threshold configuration
#[derive(Debug, Clone)]
pub struct StorageThresholds {
    /// Disk usage warning threshold (percentage)
    pub usage_warning: f64,

    /// Disk usage critical threshold (percentage)
    pub usage_critical: f64,

    /// Available space warning threshold (bytes)
    pub available_warning: u64,

    /// Available space critical threshold (bytes)
    pub available_critical: f64,

    /// I/O wait time warning threshold (percentage)
    pub io_wait_warning: f64,

    /// I/O wait time critical threshold (percentage)
    pub io_wait_critical: f64,

    /// Disk health warning threshold (SMART score)
    pub health_warning: Option<u32>,

    /// Disk health critical threshold (SMART score)
    pub health_critical: Option<u32>,
}

/// Network threshold configuration
#[derive(Debug, Clone)]
pub struct NetworkThresholds {
    /// Bandwidth utilization warning threshold (percentage)
    pub bandwidth_warning: f64,

    /// Bandwidth utilization critical threshold (percentage)
    pub bandwidth_critical: f64,

    /// Connection count warning threshold
    pub connection_warning: u32,

    /// Connection count critical threshold
    pub connection_critical: u32,

    /// Network latency warning threshold (milliseconds)
    pub latency_warning: f64,

    /// Network latency critical threshold (milliseconds)
    pub latency_critical: f64,

    /// Packet loss warning threshold (percentage)
    pub packet_loss_warning: f64,

    /// Packet loss critical threshold (percentage)
    pub packet_loss_critical: f64,
}

/// GPU threshold configuration
#[derive(Debug, Clone)]
pub struct GpuThresholds {
    /// GPU utilization warning threshold (percentage)
    pub utilization_warning: f64,

    /// GPU utilization critical threshold (percentage)
    pub utilization_critical: f64,

    /// GPU memory warning threshold (percentage)
    pub memory_warning: f64,

    /// GPU memory critical threshold (percentage)
    pub memory_critical: f64,

    /// GPU temperature warning threshold (Celsius)
    pub temperature_warning: f64,

    /// GPU temperature critical threshold (Celsius)
    pub temperature_critical: f64,

    /// GPU power warning threshold (watts)
    pub power_warning: Option<f64>,

    /// GPU power critical threshold (watts)
    pub power_critical: Option<f64>,
}

/// Custom resource threshold configuration
#[derive(Debug, Clone)]
pub struct CustomThresholds {
    /// Warning threshold value
    pub warning: f64,

    /// Critical threshold value
    pub critical: f64,

    /// Unit of measurement
    pub unit: String,

    /// Comparison direction (true for "greater than", false for "less than")
    pub greater_than: bool,
}

/// Global threshold settings
#[derive(Debug, Clone)]
pub struct GlobalThresholdSettings {
    /// Duration threshold must be exceeded before triggering
    pub duration_threshold: Duration,

    /// Number of consecutive violations before triggering
    pub consecutive_violations: u32,

    /// Enable hysteresis to prevent flapping
    pub hysteresis_enabled: bool,

    /// Hysteresis percentage (amount to reduce threshold for clearing alerts)
    pub hysteresis_percentage: f64,
}

/// Capacity planning configuration
///
/// Controls forecasting and capacity planning features that help predict
/// future resource needs and optimize resource allocation.
#[derive(Debug, Clone)]
pub struct CapacityPlanningConfig {
    /// Enable capacity planning
    pub enabled: bool,

    /// Forecasting algorithms to use
    pub algorithms: Vec<ForecastingAlgorithm>,

    /// Forecasting horizon (how far into the future to predict)
    pub forecast_horizon: Duration,

    /// Historical data window for analysis
    pub analysis_window: Duration,

    /// Confidence level for predictions
    pub confidence_level: f64,

    /// Growth rate assumptions
    pub growth_assumptions: GrowthAssumptions,

    /// Capacity planning strategies
    pub strategies: Vec<CapacityPlanningStrategy>,

    /// Recommendation settings
    pub recommendations: RecommendationSettings,
}

/// Forecasting algorithms for capacity planning
#[derive(Debug, Clone)]
pub enum ForecastingAlgorithm {
    LinearRegression,

    ExponentialSmoothing,

    Arima {
        p: u32,
        d: u32,
        q: u32,
    },

    /// Seasonal decomposition forecasting
    SeasonalDecomposition,

    /// Machine learning based forecasting
    MachineLearning {
        /// Model type
        model_type: String,
        /// Model parameters
        parameters: HashMap<String, f64>,
    },

    /// Trend analysis
    TrendAnalysis,

    /// Custom forecasting algorithm
    Custom {
        /// Algorithm name
        name: String,
        /// Algorithm configuration
        config: HashMap<String, String>,
    },
}

/// Growth rate assumptions for capacity planning
#[derive(Debug, Clone)]
pub struct GrowthAssumptions {
    /// Expected annual growth rate (percentage)
    pub annual_growth_rate: f64,

    /// Seasonal variations (month -> multiplier)
    pub seasonal_patterns: HashMap<u32, f64>,

    /// Growth acceleration factor
    pub acceleration_factor: f64,

    /// Maximum sustainable growth rate
    pub max_growth_rate: f64,
}

/// Capacity planning strategies
#[derive(Debug, Clone)]
pub enum CapacityPlanningStrategy {
    /// Conservative planning with high safety margins
    Conservative {
        /// Safety buffer percentage
        buffer_percentage: f64
    },

    /// Aggressive planning for cost optimization
    Aggressive {
        /// Efficiency target percentage
        efficiency_target: f64
    },

    /// Predictive planning based on machine learning
    Predictive {
        /// Confidence level for predictions
        confidence_level: f64
    },

    /// Cost-optimized planning
    CostOptimized {
        /// Cost threshold for resource scaling
        cost_threshold: f64
    },

    /// Custom planning strategy
    Custom {
        /// Strategy name
        name: String,
        /// Strategy parameters
        parameters: HashMap<String, f64>,
    },
}

/// Recommendation settings for capacity planning
#[derive(Debug, Clone)]
pub struct RecommendationSettings {
    /// Enable automated recommendations
    pub enabled: bool,

    /// Minimum confidence level for recommendations
    pub min_confidence: f64,

    /// Recommendation update frequency
    pub update_frequency: Duration,

    /// Include cost analysis in recommendations
    pub include_cost_analysis: bool,

    /// Maximum recommendation lookahead
    pub max_lookahead: Duration,
}

/// Resource optimization configuration
///
/// Controls automated analysis and optimization of resource usage
/// to improve efficiency and reduce costs.
#[derive(Debug, Clone)]
pub struct ResourceOptimizationConfig {
    /// Enable resource optimization
    pub enabled: bool,

    /// Optimization algorithms to use
    pub algorithms: Vec<OptimizationAlgorithm>,

    /// Optimization frequency
    pub frequency: Duration,

    /// Minimum improvement threshold
    pub min_improvement: f64,

    /// Optimization targets
    pub targets: OptimizationTargets,

    /// Constraints for optimization
    pub constraints: OptimizationConstraints,
}

/// Resource optimization algorithms
#[derive(Debug, Clone)]
pub enum OptimizationAlgorithm {
    UtilizationAnalysis,

    CostOptimization,

    PerformanceOptimization,

    EnergyOptimization,

    RightSizing,

    LoadBalancing,

    Custom {
        name: String,
        config: HashMap<String, String>,
    },
}

/// Optimization targets
#[derive(Debug, Clone)]
pub struct OptimizationTargets {
    /// Target CPU utilization (percentage)
    pub cpu_utilization_target: f64,

    /// Target memory utilization (percentage)
    pub memory_utilization_target: f64,

    /// Target cost reduction (percentage)
    pub cost_reduction_target: f64,

    /// Target performance improvement (percentage)
    pub performance_improvement_target: f64,

    /// Target energy efficiency improvement (percentage)
    pub energy_efficiency_target: f64,
}

/// Optimization constraints
#[derive(Debug, Clone)]
pub struct OptimizationConstraints {
    /// Maximum allowable performance degradation (percentage)
    pub max_performance_degradation: f64,

    /// Minimum resource headroom (percentage)
    pub min_headroom: f64,

    /// Budget constraints
    pub budget_limit: Option<f64>,

    /// Service level agreement constraints
    pub sla_constraints: HashMap<String, f64>,
}

/// Resource collection configuration
///
/// Controls how resource monitoring data is collected, processed, and stored.
#[derive(Debug, Clone)]
pub struct ResourceCollectionConfig {
    /// Collection interval for resource metrics
    pub collection_interval: Duration,

    /// Data aggregation settings
    pub aggregation: ResourceAggregationConfig,

    /// Collection methods for different resource types
    pub collection_methods: HashMap<String, CollectionMethod>,

    /// Performance tuning settings
    pub performance: CollectionPerformanceConfig,
}

/// Resource aggregation configuration
#[derive(Debug, Clone)]
pub struct ResourceAggregationConfig {
    /// Aggregation window size
    pub window_size: Duration,

    /// Aggregation functions to apply
    pub functions: Vec<AggregationFunction>,

    /// Maximum number of data points to keep in memory
    pub max_data_points: usize,

    /// Enable real-time aggregation
    pub real_time: bool,
}

/// Aggregation functions for resource data
#[derive(Debug, Clone, PartialEq)]
pub enum AggregationFunction {
    /// Average value over the window
    Average,
    /// Minimum value over the window
    Minimum,
    /// Maximum value over the window
    Maximum,
    /// Sum of values over the window
    Sum,
    /// Standard deviation over the window
    StandardDeviation,
    /// Percentile calculation
    Percentile(f64),
    /// Rate of change
    Rate,
}

/// Collection methods for different resource types
#[derive(Debug, Clone)]
pub enum CollectionMethod {
    /// System API calls (e.g., /proc, WMI)
    SystemApi,

    /// Performance monitoring tools (e.g., perf, sar)
    PerformanceTools {
        /// Tool name
        tool: String,
        /// Tool arguments
        arguments: Vec<String>,
    },

    /// Hardware monitoring (e.g., sensors, SNMP)
    HardwareMonitoring {
        /// Protocol or interface
        protocol: String,
        /// Connection details
        connection: String,
    },

    /// Container runtime API (e.g., Docker, containerd)
    ContainerRuntime {
        /// Runtime type
        runtime: String,
        /// API endpoint
        endpoint: String,
    },

    /// Cloud provider API
    CloudApi {
        /// Provider name
        provider: String,
        /// API configuration
        config: HashMap<String, String>,
    },

    /// Custom collection method
    Custom {
        /// Method name
        name: String,
        /// Method configuration
        config: HashMap<String, String>,
    },
}

/// Performance settings for resource collection
#[derive(Debug, Clone)]
pub struct CollectionPerformanceConfig {
    /// Maximum collection latency
    pub max_latency: Duration,

    /// Collection timeout
    pub timeout: Duration,

    /// Number of collection worker threads
    pub worker_threads: usize,

    /// Buffer size for collected data
    pub buffer_size: usize,

    /// Enable collection caching
    pub enable_caching: bool,

    /// Cache TTL for collection results
    pub cache_ttl: Duration,
}

/// Resource retention configuration
///
/// Controls how long resource monitoring data is retained at different
/// aggregation levels.
#[derive(Debug, Clone)]
pub struct ResourceRetentionConfig {
    /// Raw data retention period
    pub raw_data_retention: Duration,

    /// Aggregated data retention periods
    pub aggregated_retention: HashMap<String, Duration>,

    /// Archive settings for long-term storage
    pub archive: ArchiveConfig,

    /// Cleanup settings
    pub cleanup: CleanupConfig,
}

/// Archive configuration for long-term data storage
#[derive(Debug, Clone)]
pub struct ArchiveConfig {
    /// Enable archiving
    pub enabled: bool,

    /// Archive storage location
    pub storage_location: String,

    /// Archive format
    pub format: ArchiveFormat,

    /// Compression settings
    pub compression: CompressionSettings,

    /// Archive frequency
    pub frequency: Duration,
}

/// Archive storage formats
#[derive(Debug, Clone)]
pub enum ArchiveFormat {
    /// Parquet format for analytics
    Parquet,
    /// CSV format for compatibility
    Csv,
    /// JSON format for flexibility
    Json,
    /// Binary format for efficiency
    Binary,
    /// Custom format
    Custom { format_name: String },
}

/// Compression settings for archived data
#[derive(Debug, Clone)]
pub struct CompressionSettings {
    /// Enable compression
    pub enabled: bool,

    /// Compression algorithm
    pub algorithm: CompressionAlgorithm,

    /// Compression level
    pub level: u8,
}

/// Compression algorithms
#[derive(Debug, Clone)]
pub enum CompressionAlgorithm {
    /// GZIP compression
    Gzip,
    /// LZ4 compression (fast)
    Lz4,
    /// Zstandard compression (balanced)
    Zstd,
    /// BZIP2 compression (high ratio)
    Bzip2,
}

/// Cleanup configuration for resource data
#[derive(Debug, Clone)]
pub struct CleanupConfig {
    /// Enable automatic cleanup
    pub enabled: bool,

    /// Cleanup frequency
    pub frequency: Duration,

    /// Cleanup strategy
    pub strategy: CleanupStrategy,

    /// Retention policies
    pub policies: Vec<RetentionPolicy>,
}

/// Cleanup strategies
#[derive(Debug, Clone)]
pub enum CleanupStrategy {
    /// Time-based cleanup (delete data older than threshold)
    TimeBased,
    /// Size-based cleanup (delete oldest data when size limit reached)
    SizeBased,
    /// Priority-based cleanup (delete low-priority data first)
    PriorityBased,
    /// Custom cleanup strategy
    Custom { strategy_name: String },
}

/// Retention policies for different data types
#[derive(Debug, Clone)]
pub struct RetentionPolicy {
    /// Data type this policy applies to
    pub data_type: String,

    /// Retention duration
    pub retention_duration: Duration,

    /// Policy priority
    pub priority: u32,

    /// Conditions for applying this policy
    pub conditions: Vec<String>,
}

impl ResourceMonitoringConfig {
    /// Create configuration optimized for production environments
    pub fn production() -> Self {
        Self {
            enabled: true,
            resource_types: vec![
                ResourceType::Cpu { per_core: true, frequency_scaling: true, temperature: true },
                ResourceType::Memory { allocation_tracking: false, swap_monitoring: true, fragmentation_analysis: false },
                ResourceType::Storage { all_filesystems: true, io_monitoring: true, health_monitoring: true },
                ResourceType::Network { all_interfaces: true, connection_tracking: true, latency_monitoring: true },
                ResourceType::Process { all_processes: false, file_descriptors: true, thread_monitoring: true },
            ],
            thresholds: ResourceThresholds {
                cpu: CpuThresholds {
                    utilization_warning: 70.0,
                    utilization_critical: 90.0,
                    per_core_warning: 80.0,
                    per_core_critical: 95.0,
                    load_average_warning: 4.0,
                    load_average_critical: 8.0,
                    temperature_warning: Some(70.0),
                    temperature_critical: Some(85.0),
                },
                memory: MemoryThresholds {
                    usage_warning: 80.0,
                    usage_critical: 95.0,
                    available_warning: 1_073_741_824, // 1GB
                    available_critical: 536_870_912,   // 512MB
                    swap_warning: 50.0,
                    swap_critical: 80.0,
                    leak_detection_threshold: Some(104_857_600), // 100MB/hour
                },
                storage: StorageThresholds {
                    usage_warning: 80.0,
                    usage_critical: 95.0,
                    available_warning: 10_737_418_240, // 10GB
                    available_critical: 2_147_483_648.0, // 2GB
                    io_wait_warning: 20.0,
                    io_wait_critical: 40.0,
                    health_warning: Some(180),
                    health_critical: Some(160),
                },
                network: NetworkThresholds {
                    bandwidth_warning: 70.0,
                    bandwidth_critical: 90.0,
                    connection_warning: 1000,
                    connection_critical: 2000,
                    latency_warning: 100.0,
                    latency_critical: 500.0,
                    packet_loss_warning: 0.1,
                    packet_loss_critical: 1.0,
                },
                gpu: None,
                custom: HashMap::new(),
                global: GlobalThresholdSettings {
                    duration_threshold: Duration::from_secs(300),
                    consecutive_violations: 3,
                    hysteresis_enabled: true,
                    hysteresis_percentage: 10.0,
                },
            },
            capacity_planning: CapacityPlanningConfig {
                enabled: true,
                algorithms: vec![
                    ForecastingAlgorithm::LinearRegression,
                    ForecastingAlgorithm::ExponentialSmoothing,
                ],
                forecast_horizon: Duration::from_secs(86400 * 30), // 30 days
                analysis_window: Duration::from_secs(86400 * 90),  // 90 days
                confidence_level: 0.95,
                growth_assumptions: GrowthAssumptions {
                    annual_growth_rate: 15.0,
                    seasonal_patterns: HashMap::new(),
                    acceleration_factor: 1.1,
                    max_growth_rate: 50.0,
                },
                strategies: vec![CapacityPlanningStrategy::Conservative { buffer_percentage: 20.0 }],
                recommendations: RecommendationSettings {
                    enabled: true,
                    min_confidence: 0.8,
                    update_frequency: Duration::from_secs(86400), // Daily
                    include_cost_analysis: true,
                    max_lookahead: Duration::from_secs(86400 * 90), // 90 days
                },
            },
            optimization: ResourceOptimizationConfig {
                enabled: true,
                algorithms: vec![
                    OptimizationAlgorithm::UtilizationAnalysis,
                    OptimizationAlgorithm::CostOptimization,
                    OptimizationAlgorithm::RightSizing,
                ],
                frequency: Duration::from_secs(86400), // Daily
                min_improvement: 0.05, // 5%
                targets: OptimizationTargets {
                    cpu_utilization_target: 70.0,
                    memory_utilization_target: 75.0,
                    cost_reduction_target: 10.0,
                    performance_improvement_target: 5.0,
                    energy_efficiency_target: 15.0,
                },
                constraints: OptimizationConstraints {
                    max_performance_degradation: 2.0,
                    min_headroom: 15.0,
                    budget_limit: None,
                    sla_constraints: HashMap::new(),
                },
            },
            collection: ResourceCollectionConfig {
                collection_interval: Duration::from_secs(30),
                aggregation: ResourceAggregationConfig {
                    window_size: Duration::from_secs(300),
                    functions: vec![
                        AggregationFunction::Average,
                        AggregationFunction::Maximum,
                        AggregationFunction::Percentile(95.0),
                    ],
                    max_data_points: 2880, // 24 hours at 30-second intervals
                    real_time: true,
                },
                collection_methods: HashMap::new(),
                performance: CollectionPerformanceConfig {
                    max_latency: Duration::from_millis(100),
                    timeout: Duration::from_secs(10),
                    worker_threads: 4,
                    buffer_size: 1000,
                    enable_caching: true,
                    cache_ttl: Duration::from_secs(60),
                },
            },
            retention: ResourceRetentionConfig {
                raw_data_retention: Duration::from_secs(86400 * 7), // 7 days
                aggregated_retention: {
                    let mut map = HashMap::new();
                    map.insert("5min".to_string(), Duration::from_secs(86400 * 30)); // 30 days
                    map.insert("1hour".to_string(), Duration::from_secs(86400 * 365)); // 1 year
                    map.insert("1day".to_string(), Duration::from_secs(86400 * 365 * 5)); // 5 years
                    map
                },
                archive: ArchiveConfig {
                    enabled: true,
                    storage_location: "./data/archive".to_string(),
                    format: ArchiveFormat::Parquet,
                    compression: CompressionSettings {
                        enabled: true,
                        algorithm: CompressionAlgorithm::Zstd,
                        level: 6,
                    },
                    frequency: Duration::from_secs(86400), // Daily
                },
                cleanup: CleanupConfig {
                    enabled: true,
                    frequency: Duration::from_secs(86400), // Daily
                    strategy: CleanupStrategy::TimeBased,
                    policies: Vec::new(),
                },
            },
        }
    }

    /// Create configuration optimized for development environments
    pub fn development() -> Self {
        Self {
            enabled: true,
            resource_types: vec![
                ResourceType::Cpu { per_core: false, frequency_scaling: false, temperature: false },
                ResourceType::Memory { allocation_tracking: true, swap_monitoring: false, fragmentation_analysis: true },
                ResourceType::Storage { all_filesystems: false, io_monitoring: false, health_monitoring: false },
                ResourceType::Process { all_processes: true, file_descriptors: true, thread_monitoring: true },
            ],
            thresholds: ResourceThresholds {
                cpu: CpuThresholds {
                    utilization_warning: 90.0,
                    utilization_critical: 98.0,
                    per_core_warning: 95.0,
                    per_core_critical: 99.0,
                    load_average_warning: 8.0,
                    load_average_critical: 16.0,
                    temperature_warning: None,
                    temperature_critical: None,
                },
                memory: MemoryThresholds {
                    usage_warning: 90.0,
                    usage_critical: 98.0,
                    available_warning: 536_870_912,   // 512MB
                    available_critical: 268_435_456, // 256MB
                    swap_warning: 80.0,
                    swap_critical: 95.0,
                    leak_detection_threshold: Some(52_428_800), // 50MB/hour
                },
                storage: StorageThresholds {
                    usage_warning: 90.0,
                    usage_critical: 98.0,
                    available_warning: 1_073_741_824, // 1GB
                    available_critical: 536_870_912.0, // 512MB
                    io_wait_warning: 50.0,
                    io_wait_critical: 80.0,
                    health_warning: None,
                    health_critical: None,
                },
                network: NetworkThresholds {
                    bandwidth_warning: 90.0,
                    bandwidth_critical: 98.0,
                    connection_warning: 100,
                    connection_critical: 200,
                    latency_warning: 1000.0,
                    latency_critical: 5000.0,
                    packet_loss_warning: 1.0,
                    packet_loss_critical: 5.0,
                },
                gpu: None,
                custom: HashMap::new(),
                global: GlobalThresholdSettings {
                    duration_threshold: Duration::from_secs(60),
                    consecutive_violations: 1,
                    hysteresis_enabled: false,
                    hysteresis_percentage: 5.0,
                },
            },
            capacity_planning: CapacityPlanningConfig {
                enabled: false,
                algorithms: vec![],
                forecast_horizon: Duration::from_secs(86400 * 7), // 7 days
                analysis_window: Duration::from_secs(86400 * 7),  // 7 days
                confidence_level: 0.8,
                growth_assumptions: GrowthAssumptions {
                    annual_growth_rate: 0.0,
                    seasonal_patterns: HashMap::new(),
                    acceleration_factor: 1.0,
                    max_growth_rate: 0.0,
                },
                strategies: vec![],
                recommendations: RecommendationSettings {
                    enabled: false,
                    min_confidence: 0.5,
                    update_frequency: Duration::from_secs(86400 * 7), // Weekly
                    include_cost_analysis: false,
                    max_lookahead: Duration::from_secs(86400 * 7), // 7 days
                },
            },
            optimization: ResourceOptimizationConfig {
                enabled: false,
                algorithms: vec![],
                frequency: Duration::from_secs(86400 * 7), // Weekly
                min_improvement: 0.1, // 10%
                targets: OptimizationTargets {
                    cpu_utilization_target: 50.0,
                    memory_utilization_target: 60.0,
                    cost_reduction_target: 0.0,
                    performance_improvement_target: 0.0,
                    energy_efficiency_target: 0.0,
                },
                constraints: OptimizationConstraints {
                    max_performance_degradation: 5.0,
                    min_headroom: 20.0,
                    budget_limit: None,
                    sla_constraints: HashMap::new(),
                },
            },
            collection: ResourceCollectionConfig {
                collection_interval: Duration::from_secs(10),
                aggregation: ResourceAggregationConfig {
                    window_size: Duration::from_secs(60),
                    functions: vec![AggregationFunction::Average, AggregationFunction::Maximum],
                    max_data_points: 720, // 2 hours at 10-second intervals
                    real_time: false,
                },
                collection_methods: HashMap::new(),
                performance: CollectionPerformanceConfig {
                    max_latency: Duration::from_millis(500),
                    timeout: Duration::from_secs(5),
                    worker_threads: 2,
                    buffer_size: 100,
                    enable_caching: false,
                    cache_ttl: Duration::from_secs(30),
                },
            },
            retention: ResourceRetentionConfig {
                raw_data_retention: Duration::from_secs(86400), // 1 day
                aggregated_retention: {
                    let mut map = HashMap::new();
                    map.insert("1min".to_string(), Duration::from_secs(86400 * 7)); // 7 days
                    map
                },
                archive: ArchiveConfig {
                    enabled: false,
                    storage_location: "./dev/archive".to_string(),
                    format: ArchiveFormat::Json,
                    compression: CompressionSettings {
                        enabled: false,
                        algorithm: CompressionAlgorithm::Gzip,
                        level: 1,
                    },
                    frequency: Duration::from_secs(86400 * 7), // Weekly
                },
                cleanup: CleanupConfig {
                    enabled: true,
                    frequency: Duration::from_secs(86400), // Daily
                    strategy: CleanupStrategy::TimeBased,
                    policies: Vec::new(),
                },
            },
        }
    }

    /// Validate the resource monitoring configuration
    pub fn validate(&self) -> Result<(), String> {
        // Validate thresholds
        if self.thresholds.cpu.utilization_warning >= self.thresholds.cpu.utilization_critical {
            return Err("CPU warning threshold must be less than critical threshold".to_string());
        }

        if self.thresholds.memory.usage_warning >= self.thresholds.memory.usage_critical {
            return Err("Memory warning threshold must be less than critical threshold".to_string());
        }

        if self.thresholds.storage.usage_warning >= self.thresholds.storage.usage_critical {
            return Err("Storage warning threshold must be less than critical threshold".to_string());
        }

        // Validate collection settings
        if self.collection.collection_interval < Duration::from_millis(100) {
            return Err("Collection interval must be at least 100ms".to_string());
        }

        if self.collection.aggregation.window_size < self.collection.collection_interval {
            return Err("Aggregation window must be at least as large as collection interval".to_string());
        }

        // Validate capacity planning settings
        if self.capacity_planning.enabled {
            if self.capacity_planning.confidence_level < 0.0 || self.capacity_planning.confidence_level > 1.0 {
                return Err("Confidence level must be between 0.0 and 1.0".to_string());
            }
        }

        Ok(())
    }
}

impl Default for ResourceMonitoringConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            resource_types: vec![
                ResourceType::Cpu { per_core: false, frequency_scaling: false, temperature: false },
                ResourceType::Memory { allocation_tracking: false, swap_monitoring: true, fragmentation_analysis: false },
                ResourceType::Storage { all_filesystems: true, io_monitoring: false, health_monitoring: false },
            ],
            thresholds: ResourceThresholds {
                cpu: CpuThresholds {
                    utilization_warning: 80.0,
                    utilization_critical: 95.0,
                    per_core_warning: 85.0,
                    per_core_critical: 98.0,
                    load_average_warning: 4.0,
                    load_average_critical: 8.0,
                    temperature_warning: None,
                    temperature_critical: None,
                },
                memory: MemoryThresholds {
                    usage_warning: 85.0,
                    usage_critical: 95.0,
                    available_warning: 1_073_741_824, // 1GB
                    available_critical: 536_870_912,   // 512MB
                    swap_warning: 50.0,
                    swap_critical: 80.0,
                    leak_detection_threshold: None,
                },
                storage: StorageThresholds {
                    usage_warning: 85.0,
                    usage_critical: 95.0,
                    available_warning: 5_368_709_120, // 5GB
                    available_critical: 1_073_741_824.0, // 1GB
                    io_wait_warning: 30.0,
                    io_wait_critical: 50.0,
                    health_warning: None,
                    health_critical: None,
                },
                network: NetworkThresholds {
                    bandwidth_warning: 80.0,
                    bandwidth_critical: 95.0,
                    connection_warning: 500,
                    connection_critical: 1000,
                    latency_warning: 200.0,
                    latency_critical: 1000.0,
                    packet_loss_warning: 0.5,
                    packet_loss_critical: 2.0,
                },
                gpu: None,
                custom: HashMap::new(),
                global: GlobalThresholdSettings {
                    duration_threshold: Duration::from_secs(300),
                    consecutive_violations: 2,
                    hysteresis_enabled: true,
                    hysteresis_percentage: 10.0,
                },
            },
            capacity_planning: CapacityPlanningConfig {
                enabled: false,
                algorithms: vec![],
                forecast_horizon: Duration::from_secs(86400 * 30), // 30 days
                analysis_window: Duration::from_secs(86400 * 30),  // 30 days
                confidence_level: 0.9,
                growth_assumptions: GrowthAssumptions {
                    annual_growth_rate: 10.0,
                    seasonal_patterns: HashMap::new(),
                    acceleration_factor: 1.0,
                    max_growth_rate: 100.0,
                },
                strategies: vec![],
                recommendations: RecommendationSettings {
                    enabled: false,
                    min_confidence: 0.7,
                    update_frequency: Duration::from_secs(86400), // Daily
                    include_cost_analysis: false,
                    max_lookahead: Duration::from_secs(86400 * 30), // 30 days
                },
            },
            optimization: ResourceOptimizationConfig {
                enabled: false,
                algorithms: vec![],
                frequency: Duration::from_secs(86400), // Daily
                min_improvement: 0.05, // 5%
                targets: OptimizationTargets {
                    cpu_utilization_target: 70.0,
                    memory_utilization_target: 75.0,
                    cost_reduction_target: 5.0,
                    performance_improvement_target: 5.0,
                    energy_efficiency_target: 10.0,
                },
                constraints: OptimizationConstraints {
                    max_performance_degradation: 5.0,
                    min_headroom: 20.0,
                    budget_limit: None,
                    sla_constraints: HashMap::new(),
                },
            },
            collection: ResourceCollectionConfig {
                collection_interval: Duration::from_secs(60),
                aggregation: ResourceAggregationConfig {
                    window_size: Duration::from_secs(300),
                    functions: vec![AggregationFunction::Average, AggregationFunction::Maximum],
                    max_data_points: 1440, // 24 hours at 1-minute intervals
                    real_time: false,
                },
                collection_methods: HashMap::new(),
                performance: CollectionPerformanceConfig {
                    max_latency: Duration::from_millis(200),
                    timeout: Duration::from_secs(10),
                    worker_threads: 2,
                    buffer_size: 500,
                    enable_caching: true,
                    cache_ttl: Duration::from_secs(60),
                },
            },
            retention: ResourceRetentionConfig {
                raw_data_retention: Duration::from_secs(86400 * 7), // 7 days
                aggregated_retention: {
                    let mut map = HashMap::new();
                    map.insert("5min".to_string(), Duration::from_secs(86400 * 30)); // 30 days
                    map.insert("1hour".to_string(), Duration::from_secs(86400 * 90)); // 90 days
                    map
                },
                archive: ArchiveConfig {
                    enabled: false,
                    storage_location: "./data/archive".to_string(),
                    format: ArchiveFormat::Json,
                    compression: CompressionSettings {
                        enabled: false,
                        algorithm: CompressionAlgorithm::Gzip,
                        level: 6,
                    },
                    frequency: Duration::from_secs(86400), // Daily
                },
                cleanup: CleanupConfig {
                    enabled: true,
                    frequency: Duration::from_secs(86400), // Daily
                    strategy: CleanupStrategy::TimeBased,
                    policies: Vec::new(),
                },
            },
        }
    }
}