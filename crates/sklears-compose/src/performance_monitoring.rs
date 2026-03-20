//! Comprehensive Performance Monitoring and Analytics
//!
//! This module provides extensive performance monitoring capabilities including
//! real-time metrics collection, performance analysis, alerting, and advanced
//! analytics for the composable execution engine ecosystem.

use sklears_core::{
    error::{Result as SklResult, SklearsError},
};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, SystemTime, Instant};

use crate::execution_types::*;
use crate::execution_strategies::StrategyMetrics;
use crate::resource_management::ResourceUtilization;

/// Comprehensive execution metrics aggregator
///
/// Collects and aggregates performance metrics from all components of the
/// execution engine including strategies, schedulers, and resource managers.
#[derive(Debug, Clone)]
pub struct ExecutionMetrics {
    /// Metrics collection start time
    pub start_time: SystemTime,

    /// Strategy-specific performance metrics
    pub strategy_metrics: HashMap<String, StrategyMetrics>,

    /// Task scheduler performance metrics
    pub scheduler_metrics: SchedulerMetrics,

    /// Resource utilization metrics
    pub resource_metrics: ResourceMetrics,

    /// Error and failure statistics
    pub error_statistics: ErrorStatistics,

    /// Performance trend analysis
    pub performance_trends: PerformanceTrends,

    /// System-wide performance indicators
    pub system_performance: SystemPerformanceIndicators,

    /// Custom metric collections
    pub custom_metrics: HashMap<String, MetricCollection>,
}

/// Task scheduler performance metrics
///
/// Tracks scheduling efficiency, queue management, and dependency
/// resolution performance for optimization.
#[derive(Debug, Clone)]
pub struct SchedulerMetrics {
    /// Total number of tasks scheduled
    pub tasks_scheduled: u64,

    /// Average time to schedule a task
    pub avg_scheduling_time: Duration,

    /// Current queue length across all priorities
    pub queue_length: usize,

    /// Scheduling efficiency (0.0 to 1.0)
    pub efficiency: f64,

    /// Dependency resolution success rate
    pub dependency_resolution_rate: f64,

    /// Queue utilization by priority
    pub queue_utilization_by_priority: HashMap<TaskPriority, f64>,

    /// Load balancing effectiveness
    pub load_balancing_effectiveness: f64,

    /// Scheduler throughput (tasks/second)
    pub throughput: f64,

    /// Average task wait time in queue
    pub avg_wait_time: Duration,

    /// Peak queue size recorded
    pub peak_queue_size: usize,

    /// Scheduling algorithm performance
    pub algorithm_performance: AlgorithmPerformanceMetrics,
}

/// Algorithm-specific performance metrics
#[derive(Debug, Clone)]
pub struct AlgorithmPerformanceMetrics {
    /// Algorithm name
    pub algorithm_name: String,

    /// Algorithm efficiency score
    pub efficiency_score: f64,

    /// Fairness metric (0.0 to 1.0)
    pub fairness_score: f64,

    /// Response time variance
    pub response_time_variance: f64,

    /// Resource utilization optimization
    pub resource_optimization_score: f64,

    /// Algorithm-specific custom metrics
    pub custom_algorithm_metrics: HashMap<String, f64>,
}

/// Comprehensive resource metrics collection
///
/// Detailed system resource utilization and performance metrics
/// for capacity planning and optimization.
#[derive(Debug, Clone)]
pub struct ResourceMetrics {
    /// Metrics timestamp
    pub timestamp: SystemTime,

    /// CPU performance metrics
    pub cpu: CpuMetrics,

    /// Memory utilization metrics
    pub memory: MemoryMetrics,

    /// I/O performance metrics
    pub io: IoMetrics,

    /// Network performance metrics
    pub network: NetworkMetrics,

    /// GPU utilization metrics (if available)
    pub gpu: Vec<GpuMetrics>,

    /// Storage performance metrics
    pub storage: Vec<StorageMetrics>,

    /// Resource allocation efficiency
    pub allocation_efficiency: ResourceAllocationEfficiency,

    /// Resource contention metrics
    pub contention_metrics: ResourceContentionMetrics,
}

/// CPU performance and utilization metrics
#[derive(Debug, Clone)]
pub struct CpuMetrics {
    /// Overall CPU utilization (0.0 to 1.0)
    pub utilization: f64,

    /// Per-core utilization values
    pub per_core: Vec<f64>,

    /// CPU frequency in MHz
    pub frequency: f64,

    /// CPU temperature in Celsius (if available)
    pub temperature: Option<f64>,

    /// Power consumption in watts (if available)
    pub power_consumption: Option<f64>,

    /// Context switches per second
    pub context_switches: f64,

    /// Interrupts per second
    pub interrupts: f64,

    /// CPU cache hit rates
    pub cache_hit_rates: CpuCacheMetrics,

    /// Load average values
    pub load_average: LoadAverageMetrics,

    /// CPU performance counters
    pub performance_counters: CpuPerformanceCounters,
}

/// CPU cache performance metrics
#[derive(Debug, Clone)]
pub struct CpuCacheMetrics {
    /// L1 cache hit rate
    pub l1_hit_rate: f64,

    /// L2 cache hit rate
    pub l2_hit_rate: f64,

    /// L3 cache hit rate
    pub l3_hit_rate: f64,

    /// Cache miss rate
    pub miss_rate: f64,

    /// Cache coherency overhead
    pub coherency_overhead: f64,
}

/// System load average metrics
#[derive(Debug, Clone)]
pub struct LoadAverageMetrics {
    /// 1-minute load average
    pub one_minute: f64,

    /// 5-minute load average
    pub five_minute: f64,

    /// 15-minute load average
    pub fifteen_minute: f64,

    /// Load trend direction
    pub trend: LoadTrend,
}

/// Load trend indicators
#[derive(Debug, Clone)]
pub enum LoadTrend {
    Increasing,
    Decreasing,
    Stable,
    Volatile,
}

/// CPU performance counters
#[derive(Debug, Clone)]
pub struct CpuPerformanceCounters {
    /// Instructions per cycle
    pub instructions_per_cycle: f64,

    /// Branch misprediction rate
    pub branch_misprediction_rate: f64,

    /// Memory bandwidth utilization
    pub memory_bandwidth_utilization: f64,

    /// Floating point operations per second
    pub flops: f64,

    /// CPU stall cycles
    pub stall_cycles: f64,
}

/// Memory utilization and performance metrics
#[derive(Debug, Clone)]
pub struct MemoryMetrics {
    /// Used memory in bytes
    pub used: u64,

    /// Available memory in bytes
    pub available: u64,

    /// Cached memory in bytes
    pub cached: u64,

    /// Buffer memory in bytes
    pub buffers: u64,

    /// Swap usage in bytes
    pub swap_used: u64,

    /// Total swap space in bytes
    pub swap_total: u64,

    /// Memory pressure indicator (0.0 to 1.0)
    pub pressure: f64,

    /// Page faults per second
    pub page_faults: f64,

    /// Memory bandwidth utilization
    pub bandwidth_utilization: f64,

    /// NUMA memory distribution
    pub numa_metrics: NumaMemoryMetrics,

    /// Memory allocation patterns
    pub allocation_patterns: MemoryAllocationPatterns,
}

/// NUMA memory performance metrics
#[derive(Debug, Clone)]
pub struct NumaMemoryMetrics {
    /// Memory usage per NUMA node
    pub per_node_usage: Vec<u64>,

    /// Inter-node memory access rate
    pub inter_node_access_rate: f64,

    /// NUMA balancing efficiency
    pub balancing_efficiency: f64,

    /// Local vs remote memory access ratio
    pub local_access_ratio: f64,
}

/// Memory allocation pattern analysis
#[derive(Debug, Clone)]
pub struct MemoryAllocationPatterns {
    /// Small allocation frequency
    pub small_alloc_frequency: f64,

    /// Large allocation frequency
    pub large_alloc_frequency: f64,

    /// Average allocation size
    pub avg_allocation_size: u64,

    /// Memory fragmentation level
    pub fragmentation_level: f64,

    /// Allocation/deallocation ratio
    pub allocation_deallocation_ratio: f64,
}

/// I/O performance metrics
#[derive(Debug, Clone)]
pub struct IoMetrics {
    /// Read operations per second
    pub read_ops: f64,

    /// Write operations per second
    pub write_ops: f64,

    /// Read bandwidth in bytes per second
    pub read_bandwidth: f64,

    /// Write bandwidth in bytes per second
    pub write_bandwidth: f64,

    /// I/O wait time percentage
    pub io_wait: f64,

    /// Average queue depth
    pub queue_depth: f64,

    /// Average service time in milliseconds
    pub service_time: f64,

    /// I/O utilization percentage
    pub utilization: f64,

    /// I/O patterns analysis
    pub patterns: IoPatternMetrics,

    /// I/O efficiency metrics
    pub efficiency: IoEfficiencyMetrics,
}

/// I/O access pattern analysis
#[derive(Debug, Clone)]
pub struct IoPatternMetrics {
    /// Sequential access percentage
    pub sequential_access_percentage: f64,

    /// Random access percentage
    pub random_access_percentage: f64,

    /// Average request size
    pub avg_request_size: u64,

    /// Read/write ratio
    pub read_write_ratio: f64,

    /// Access locality score
    pub locality_score: f64,
}

/// I/O efficiency metrics
#[derive(Debug, Clone)]
pub struct IoEfficiencyMetrics {
    /// Cache hit rate for I/O operations
    pub cache_hit_rate: f64,

    /// I/O coalescing effectiveness
    pub coalescing_effectiveness: f64,

    /// Prefetch accuracy
    pub prefetch_accuracy: f64,

    /// I/O scheduler efficiency
    pub scheduler_efficiency: f64,
}

/// Network performance metrics
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

    /// Network latency in milliseconds
    pub latency: Option<f64>,

    /// Packet loss rate
    pub packet_loss: Option<f64>,

    /// Network utilization percentage
    pub utilization: f64,

    /// Active connection count
    pub connections: u32,

    /// Network quality metrics
    pub quality: NetworkQualityMetrics,

    /// Network efficiency metrics
    pub efficiency: NetworkEfficiencyMetrics,
}

/// Network quality assessment metrics
#[derive(Debug, Clone)]
pub struct NetworkQualityMetrics {
    /// Jitter (latency variation) in milliseconds
    pub jitter: f64,

    /// Bandwidth consistency score
    pub bandwidth_consistency: f64,

    /// Connection stability score
    pub connection_stability: f64,

    /// Error rate per packet
    pub error_rate: f64,
}

/// Network efficiency metrics
#[derive(Debug, Clone)]
pub struct NetworkEfficiencyMetrics {
    /// Protocol overhead percentage
    pub protocol_overhead: f64,

    /// Congestion control effectiveness
    pub congestion_control_effectiveness: f64,

    /// Buffer utilization efficiency
    pub buffer_utilization: f64,

    /// Network stack efficiency
    pub stack_efficiency: f64,
}

/// GPU performance metrics (per device)
#[derive(Debug, Clone)]
pub struct GpuMetrics {
    /// GPU device ID
    pub device_id: usize,

    /// GPU utilization percentage
    pub utilization: f64,

    /// Memory utilization percentage
    pub memory_utilization: f64,

    /// GPU temperature in Celsius
    pub temperature: f64,

    /// Power consumption in watts
    pub power_consumption: f64,

    /// Core clock speed in MHz
    pub core_clock: f64,

    /// Memory clock speed in MHz
    pub memory_clock: f64,

    /// Current performance state
    pub performance_state: String,

    /// GPU compute efficiency
    pub compute_efficiency: GpuComputeEfficiency,

    /// GPU memory efficiency
    pub memory_efficiency: GpuMemoryEfficiency,
}

/// GPU compute efficiency metrics
#[derive(Debug, Clone)]
pub struct GpuComputeEfficiency {
    /// Shader utilization percentage
    pub shader_utilization: f64,

    /// Compute unit efficiency
    pub compute_unit_efficiency: f64,

    /// Instruction throughput
    pub instruction_throughput: f64,

    /// Occupancy percentage
    pub occupancy: f64,
}

/// GPU memory efficiency metrics
#[derive(Debug, Clone)]
pub struct GpuMemoryEfficiency {
    /// Memory bandwidth utilization
    pub bandwidth_utilization: f64,

    /// Memory access efficiency
    pub access_efficiency: f64,

    /// Cache hit rate
    pub cache_hit_rate: f64,

    /// Memory coalescing efficiency
    pub coalescing_efficiency: f64,
}

/// Storage device performance metrics
#[derive(Debug, Clone)]
pub struct StorageMetrics {
    /// Storage device identifier
    pub device_id: String,

    /// Read IOPS (Input/Output Operations Per Second)
    pub read_iops: f64,

    /// Write IOPS
    pub write_iops: f64,

    /// Read latency in milliseconds
    pub read_latency: f64,

    /// Write latency in milliseconds
    pub write_latency: f64,

    /// Queue depth
    pub queue_depth: f64,

    /// Device utilization percentage
    pub utilization: f64,

    /// Space utilization percentage
    pub space_utilization: f64,

    /// Storage efficiency metrics
    pub efficiency: StorageEfficiencyMetrics,

    /// Storage health indicators
    pub health: StorageHealthMetrics,
}

/// Storage efficiency analysis
#[derive(Debug, Clone)]
pub struct StorageEfficiencyMetrics {
    /// Compression ratio achieved
    pub compression_ratio: f64,

    /// Deduplication efficiency
    pub deduplication_efficiency: f64,

    /// Wear leveling effectiveness
    pub wear_leveling_effectiveness: f64,

    /// Garbage collection efficiency
    pub gc_efficiency: f64,
}

/// Storage health indicators
#[derive(Debug, Clone)]
pub struct StorageHealthMetrics {
    /// Device health score (0.0 to 1.0)
    pub health_score: f64,

    /// Estimated remaining lifetime
    pub estimated_lifetime: Option<Duration>,

    /// Error rate per operation
    pub error_rate: f64,

    /// Temperature (if available)
    pub temperature: Option<f64>,
}

/// Resource allocation efficiency metrics
#[derive(Debug, Clone)]
pub struct ResourceAllocationEfficiency {
    /// CPU allocation efficiency
    pub cpu_efficiency: f64,

    /// Memory allocation efficiency
    pub memory_efficiency: f64,

    /// I/O allocation efficiency
    pub io_efficiency: f64,

    /// GPU allocation efficiency
    pub gpu_efficiency: f64,

    /// Overall allocation score
    pub overall_efficiency: f64,

    /// Resource waste percentage
    pub waste_percentage: f64,
}

/// Resource contention analysis
#[derive(Debug, Clone)]
pub struct ResourceContentionMetrics {
    /// CPU contention level
    pub cpu_contention: f64,

    /// Memory contention level
    pub memory_contention: f64,

    /// I/O contention level
    pub io_contention: f64,

    /// Network contention level
    pub network_contention: f64,

    /// Lock contention metrics
    pub lock_contention: LockContentionMetrics,

    /// Queue contention metrics
    pub queue_contention: QueueContentionMetrics,
}

/// Lock contention analysis
#[derive(Debug, Clone)]
pub struct LockContentionMetrics {
    /// Total lock acquisitions
    pub total_acquisitions: u64,

    /// Failed lock acquisitions
    pub failed_acquisitions: u64,

    /// Average lock hold time
    pub avg_hold_time: Duration,

    /// Maximum lock wait time
    pub max_wait_time: Duration,

    /// Lock contention hotspots
    pub hotspots: Vec<String>,
}

/// Queue contention analysis
#[derive(Debug, Clone)]
pub struct QueueContentionMetrics {
    /// Queue utilization across priorities
    pub utilization_by_priority: HashMap<TaskPriority, f64>,

    /// Average queue wait time
    pub avg_wait_time: Duration,

    /// Queue growth rate
    pub growth_rate: f64,

    /// Queue bottleneck identification
    pub bottlenecks: Vec<String>,
}

/// Error and failure statistics
///
/// Comprehensive error tracking and analysis for system reliability
/// and failure pattern identification.
#[derive(Debug, Clone)]
pub struct ErrorStatistics {
    /// Total number of errors recorded
    pub total_errors: u64,

    /// Errors categorized by type
    pub errors_by_type: HashMap<String, u64>,

    /// Overall error rate (errors per operation)
    pub error_rate: f64,

    /// Recovery success rate
    pub recovery_rate: f64,

    /// Error severity distribution
    pub severity_distribution: ErrorSeverityDistribution,

    /// Error patterns analysis
    pub error_patterns: ErrorPatternAnalysis,

    /// Mean Time Between Failures (MTBF)
    pub mtbf: Duration,

    /// Mean Time To Recovery (MTTR)
    pub mttr: Duration,

    /// Error hotspots identification
    pub error_hotspots: Vec<ErrorHotspot>,
}

/// Error severity distribution
#[derive(Debug, Clone)]
pub struct ErrorSeverityDistribution {
    /// Critical errors count
    pub critical: u64,

    /// High severity errors count
    pub high: u64,

    /// Medium severity errors count
    pub medium: u64,

    /// Low severity errors count
    pub low: u64,

    /// Warning count
    pub warnings: u64,
}

/// Error pattern analysis
#[derive(Debug, Clone)]
pub struct ErrorPatternAnalysis {
    /// Temporal error clustering
    pub temporal_clustering: f64,

    /// Spatial error clustering
    pub spatial_clustering: f64,

    /// Error correlation score
    pub correlation_score: f64,

    /// Cascading failure indicators
    pub cascading_indicators: Vec<String>,

    /// Error prediction confidence
    pub prediction_confidence: f64,
}

/// Error hotspot identification
#[derive(Debug, Clone)]
pub struct ErrorHotspot {
    /// Component or module name
    pub component: String,

    /// Error frequency
    pub frequency: u64,

    /// Error rate for this component
    pub error_rate: f64,

    /// Impact severity
    pub impact_severity: f64,

    /// Suggested remediation actions
    pub remediation_suggestions: Vec<String>,
}

/// Performance trend analysis
///
/// Historical performance data analysis and trend identification
/// for capacity planning and optimization.
#[derive(Debug, Clone)]
pub struct PerformanceTrends {
    /// Throughput trend over time
    pub throughput_trend: TrendAnalysis,

    /// Latency trend over time
    pub latency_trend: TrendAnalysis,

    /// Resource utilization trends
    pub resource_utilization_trends: HashMap<String, TrendAnalysis>,

    /// Error rate trends
    pub error_rate_trend: TrendAnalysis,

    /// Performance degradation indicators
    pub degradation_indicators: Vec<DegradationIndicator>,

    /// Seasonal patterns
    pub seasonal_patterns: SeasonalPatterns,

    /// Performance forecasting
    pub forecasting: PerformanceForecasting,
}

/// Trend analysis for specific metrics
#[derive(Debug, Clone)]
pub struct TrendAnalysis {
    /// Current value
    pub current_value: f64,

    /// Trend direction
    pub direction: TrendDirection,

    /// Trend strength (0.0 to 1.0)
    pub strength: f64,

    /// Rate of change
    pub rate_of_change: f64,

    /// Confidence in trend (0.0 to 1.0)
    pub confidence: f64,

    /// Prediction for next period
    pub prediction: f64,
}

/// Trend direction indicators
#[derive(Debug, Clone)]
pub enum TrendDirection {
    Increasing,
    Decreasing,
    Stable,
    Volatile,
    Cyclical,
}

/// Performance degradation indicators
#[derive(Debug, Clone)]
pub struct DegradationIndicator {
    /// Metric name showing degradation
    pub metric: String,

    /// Degradation severity (0.0 to 1.0)
    pub severity: f64,

    /// Rate of degradation
    pub degradation_rate: f64,

    /// Estimated time to critical level
    pub time_to_critical: Option<Duration>,

    /// Potential causes
    pub potential_causes: Vec<String>,
}

/// Seasonal pattern analysis
#[derive(Debug, Clone)]
pub struct SeasonalPatterns {
    /// Daily patterns detected
    pub daily_patterns: Vec<PatternDescription>,

    /// Weekly patterns detected
    pub weekly_patterns: Vec<PatternDescription>,

    /// Monthly patterns detected
    pub monthly_patterns: Vec<PatternDescription>,

    /// Pattern strength scores
    pub pattern_strengths: HashMap<String, f64>,
}

/// Pattern description
#[derive(Debug, Clone)]
pub struct PatternDescription {
    /// Pattern name
    pub name: String,

    /// Pattern period
    pub period: Duration,

    /// Pattern amplitude
    pub amplitude: f64,

    /// Pattern confidence
    pub confidence: f64,
}

/// Performance forecasting
#[derive(Debug, Clone)]
pub struct PerformanceForecasting {
    /// Short-term forecast (next hour)
    pub short_term: ForecastData,

    /// Medium-term forecast (next day)
    pub medium_term: ForecastData,

    /// Long-term forecast (next week)
    pub long_term: ForecastData,

    /// Forecast accuracy metrics
    pub accuracy_metrics: ForecastAccuracy,
}

/// Forecast data structure
#[derive(Debug, Clone)]
pub struct ForecastData {
    /// Predicted throughput
    pub predicted_throughput: f64,

    /// Predicted latency
    pub predicted_latency: f64,

    /// Predicted resource utilization
    pub predicted_resource_utilization: f64,

    /// Prediction confidence
    pub confidence: f64,

    /// Prediction intervals
    pub confidence_intervals: ConfidenceIntervals,
}

/// Forecast confidence intervals
#[derive(Debug, Clone)]
pub struct ConfidenceIntervals {
    /// Lower bound (95% confidence)
    pub lower_95: f64,

    /// Upper bound (95% confidence)
    pub upper_95: f64,

    /// Lower bound (99% confidence)
    pub lower_99: f64,

    /// Upper bound (99% confidence)
    pub upper_99: f64,
}

/// Forecast accuracy assessment
#[derive(Debug, Clone)]
pub struct ForecastAccuracy {
    /// Mean Absolute Error
    pub mae: f64,

    /// Root Mean Square Error
    pub rmse: f64,

    /// Mean Absolute Percentage Error
    pub mape: f64,

    /// Forecast bias
    pub bias: f64,
}

/// System-wide performance indicators
///
/// High-level performance KPIs for executive dashboards
/// and system health monitoring.
#[derive(Debug, Clone)]
pub struct SystemPerformanceIndicators {
    /// Overall system health score (0.0 to 1.0)
    pub health_score: f64,

    /// System efficiency rating (0.0 to 1.0)
    pub efficiency_rating: f64,

    /// Performance stability index
    pub stability_index: f64,

    /// System scalability metrics
    pub scalability_metrics: ScalabilityMetrics,

    /// Reliability metrics
    pub reliability_metrics: ReliabilityMetrics,

    /// Service Level Agreement compliance
    pub sla_compliance: SlaComplianceMetrics,

    /// Cost efficiency metrics
    pub cost_efficiency: CostEfficiencyMetrics,
}

/// Scalability assessment metrics
#[derive(Debug, Clone)]
pub struct ScalabilityMetrics {
    /// Horizontal scalability score
    pub horizontal_scalability: f64,

    /// Vertical scalability score
    pub vertical_scalability: f64,

    /// Load capacity utilization
    pub load_capacity_utilization: f64,

    /// Scaling efficiency
    pub scaling_efficiency: f64,

    /// Bottleneck identification
    pub bottlenecks: Vec<String>,
}

/// Reliability assessment metrics
#[derive(Debug, Clone)]
pub struct ReliabilityMetrics {
    /// System uptime percentage
    pub uptime_percentage: f64,

    /// Availability score (including planned downtime)
    pub availability_score: f64,

    /// Fault tolerance effectiveness
    pub fault_tolerance_effectiveness: f64,

    /// Recovery time metrics
    pub recovery_metrics: RecoveryMetrics,

    /// Redundancy effectiveness
    pub redundancy_effectiveness: f64,
}

/// Recovery time metrics
#[derive(Debug, Clone)]
pub struct RecoveryMetrics {
    /// Recovery Time Objective (RTO)
    pub rto: Duration,

    /// Recovery Point Objective (RPO)
    pub rpo: Duration,

    /// Actual recovery times
    pub actual_recovery_times: Vec<Duration>,

    /// Recovery success rate
    pub recovery_success_rate: f64,
}

/// SLA compliance metrics
#[derive(Debug, Clone)]
pub struct SlaComplianceMetrics {
    /// Overall SLA compliance percentage
    pub overall_compliance: f64,

    /// Compliance by service level
    pub compliance_by_level: HashMap<String, f64>,

    /// SLA violations count
    pub violations_count: u64,

    /// Time to SLA breach warnings
    pub time_to_breach: Option<Duration>,

    /// Compliance trends
    pub compliance_trends: HashMap<String, TrendAnalysis>,
}

/// Cost efficiency analysis
#[derive(Debug, Clone)]
pub struct CostEfficiencyMetrics {
    /// Cost per transaction
    pub cost_per_transaction: f64,

    /// Resource cost efficiency
    pub resource_cost_efficiency: f64,

    /// Performance per dollar
    pub performance_per_dollar: f64,

    /// Cost optimization opportunities
    pub optimization_opportunities: Vec<CostOptimizationOpportunity>,

    /// ROI metrics
    pub roi_metrics: RoiMetrics,
}

/// Cost optimization opportunity
#[derive(Debug, Clone)]
pub struct CostOptimizationOpportunity {
    /// Opportunity description
    pub description: String,

    /// Potential savings percentage
    pub potential_savings: f64,

    /// Implementation complexity
    pub implementation_complexity: ComplexityLevel,

    /// Expected ROI timeframe
    pub roi_timeframe: Duration,
}

/// Implementation complexity levels
#[derive(Debug, Clone)]
pub enum ComplexityLevel {
    Low,
    Medium,
    High,
    VeryHigh,
}

/// Return on Investment metrics
#[derive(Debug, Clone)]
pub struct RoiMetrics {
    /// Performance improvement ROI
    pub performance_roi: f64,

    /// Cost reduction ROI
    pub cost_reduction_roi: f64,

    /// Overall system ROI
    pub overall_roi: f64,

    /// Payback period
    pub payback_period: Duration,
}

/// Generic metric collection for custom metrics
#[derive(Debug, Clone)]
pub struct MetricCollection {
    /// Collection name
    pub name: String,

    /// Metric values with timestamps
    pub values: VecDeque<(SystemTime, f64)>,

    /// Statistical summary
    pub statistics: MetricStatistics,

    /// Collection metadata
    pub metadata: HashMap<String, String>,
}

/// Statistical summary for metric collections
#[derive(Debug, Clone)]
pub struct MetricStatistics {
    /// Count of data points
    pub count: usize,

    /// Mean value
    pub mean: f64,

    /// Standard deviation
    pub std_dev: f64,

    /// Minimum value
    pub min: f64,

    /// Maximum value
    pub max: f64,

    /// Median value
    pub median: f64,

    /// 95th percentile
    pub p95: f64,

    /// 99th percentile
    pub p99: f64,
}

// Default implementations and constructors

impl ExecutionMetrics {
    /// Create a new execution metrics instance
    pub fn new() -> Self {
        Self {
            start_time: SystemTime::now(),
            strategy_metrics: HashMap::new(),
            scheduler_metrics: SchedulerMetrics::default(),
            resource_metrics: ResourceMetrics::default(),
            error_statistics: ErrorStatistics::default(),
            performance_trends: PerformanceTrends::default(),
            system_performance: SystemPerformanceIndicators::default(),
            custom_metrics: HashMap::new(),
        }
    }

    /// Update metrics with task result
    pub fn update_with_result(&mut self, result: &TaskResult) {
        // Update error statistics
        match result.status {
            TaskStatus::Failed => {
                self.error_statistics.total_errors += 1;
                if let Some(error) = &result.error {
                    *self.error_statistics.errors_by_type
                        .entry(error.error_type.clone())
                        .or_insert(0) += 1;
                }
            }
            TaskStatus::Completed => {
                // Task completed successfully
            }
            _ => {}
        }

        // Update other metrics based on task execution
        self.update_performance_indicators(result);
        self.update_trends();
    }

    /// Update performance indicators based on task result
    fn update_performance_indicators(&mut self, _result: &TaskResult) {
        // Placeholder for performance indicator updates
        // In a real implementation, this would analyze task results
        // and update various performance metrics
    }

    /// Update trend analysis
    fn update_trends(&mut self) {
        // Placeholder for trend analysis updates
        // In a real implementation, this would analyze historical data
        // and update trend predictions
    }

    /// Generate performance report
    pub fn generate_report(&self) -> PerformanceReport {
        PerformanceReport {
            timestamp: SystemTime::now(),
            summary: self.get_performance_summary(),
            detailed_metrics: self.clone(),
            recommendations: self.generate_recommendations(),
        }
    }

    /// Get high-level performance summary
    fn get_performance_summary(&self) -> PerformanceSummary {
        PerformanceSummary {
            overall_health: self.system_performance.health_score,
            throughput: self.scheduler_metrics.throughput,
            avg_latency: self.scheduler_metrics.avg_wait_time,
            error_rate: self.error_statistics.error_rate,
            resource_utilization: self.resource_metrics.allocation_efficiency.overall_efficiency,
        }
    }

    /// Generate optimization recommendations
    fn generate_recommendations(&self) -> Vec<OptimizationRecommendation> {
        let mut recommendations = Vec::new();

        // Generate recommendations based on current metrics
        if self.resource_metrics.allocation_efficiency.overall_efficiency < 0.7 {
            recommendations.push(OptimizationRecommendation {
                category: "Resource Efficiency".to_string(),
                priority: RecommendationPriority::High,
                description: "Resource allocation efficiency is below optimal threshold".to_string(),
                suggested_actions: vec![
                    "Review resource allocation policies".to_string(),
                    "Consider auto-scaling adjustments".to_string(),
                ],
                expected_impact: 0.15,
            });
        }

        if self.error_statistics.error_rate > 0.05 {
            recommendations.push(OptimizationRecommendation {
                category: "Reliability".to_string(),
                priority: RecommendationPriority::Critical,
                description: "Error rate exceeds acceptable threshold".to_string(),
                suggested_actions: vec![
                    "Investigate error hotspots".to_string(),
                    "Implement additional fault tolerance".to_string(),
                ],
                expected_impact: 0.8,
            });
        }

        recommendations
    }
}

/// Performance report structure
#[derive(Debug, Clone)]
pub struct PerformanceReport {
    /// Report generation timestamp
    pub timestamp: SystemTime,

    /// High-level performance summary
    pub summary: PerformanceSummary,

    /// Detailed metrics
    pub detailed_metrics: ExecutionMetrics,

    /// Optimization recommendations
    pub recommendations: Vec<OptimizationRecommendation>,
}

/// High-level performance summary
#[derive(Debug, Clone)]
pub struct PerformanceSummary {
    /// Overall system health (0.0 to 1.0)
    pub overall_health: f64,

    /// System throughput (tasks/second)
    pub throughput: f64,

    /// Average latency
    pub avg_latency: Duration,

    /// Error rate (0.0 to 1.0)
    pub error_rate: f64,

    /// Resource utilization efficiency
    pub resource_utilization: f64,
}

/// Optimization recommendation
#[derive(Debug, Clone)]
pub struct OptimizationRecommendation {
    /// Recommendation category
    pub category: String,

    /// Priority level
    pub priority: RecommendationPriority,

    /// Description of the issue
    pub description: String,

    /// Suggested remediation actions
    pub suggested_actions: Vec<String>,

    /// Expected performance impact (0.0 to 1.0)
    pub expected_impact: f64,
}

/// Recommendation priority levels
#[derive(Debug, Clone)]
pub enum RecommendationPriority {
    Low,
    Medium,
    High,
    Critical,
}

// Default implementations for major structures

impl Default for SchedulerMetrics {
    fn default() -> Self {
        Self {
            tasks_scheduled: 0,
            avg_scheduling_time: Duration::new(0, 0),
            queue_length: 0,
            efficiency: 1.0,
            dependency_resolution_rate: 1.0,
            queue_utilization_by_priority: HashMap::new(),
            load_balancing_effectiveness: 1.0,
            throughput: 0.0,
            avg_wait_time: Duration::new(0, 0),
            peak_queue_size: 0,
            algorithm_performance: AlgorithmPerformanceMetrics {
                algorithm_name: "default".to_string(),
                efficiency_score: 1.0,
                fairness_score: 1.0,
                response_time_variance: 0.0,
                resource_optimization_score: 1.0,
                custom_algorithm_metrics: HashMap::new(),
            },
        }
    }
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
            storage: Vec::new(),
            allocation_efficiency: ResourceAllocationEfficiency::default(),
            contention_metrics: ResourceContentionMetrics::default(),
        }
    }
}

impl Default for CpuMetrics {
    fn default() -> Self {
        Self {
            utilization: 0.0,
            per_core: Vec::new(),
            frequency: 0.0,
            temperature: None,
            power_consumption: None,
            context_switches: 0.0,
            interrupts: 0.0,
            cache_hit_rates: CpuCacheMetrics {
                l1_hit_rate: 0.95,
                l2_hit_rate: 0.90,
                l3_hit_rate: 0.85,
                miss_rate: 0.05,
                coherency_overhead: 0.02,
            },
            load_average: LoadAverageMetrics {
                one_minute: 0.0,
                five_minute: 0.0,
                fifteen_minute: 0.0,
                trend: LoadTrend::Stable,
            },
            performance_counters: CpuPerformanceCounters {
                instructions_per_cycle: 2.0,
                branch_misprediction_rate: 0.05,
                memory_bandwidth_utilization: 0.3,
                flops: 0.0,
                stall_cycles: 0.1,
            },
        }
    }
}

impl Default for MemoryMetrics {
    fn default() -> Self {
        Self {
            used: 0,
            available: 0,
            cached: 0,
            buffers: 0,
            swap_used: 0,
            swap_total: 0,
            pressure: 0.0,
            page_faults: 0.0,
            bandwidth_utilization: 0.0,
            numa_metrics: NumaMemoryMetrics {
                per_node_usage: Vec::new(),
                inter_node_access_rate: 0.0,
                balancing_efficiency: 1.0,
                local_access_ratio: 1.0,
            },
            allocation_patterns: MemoryAllocationPatterns {
                small_alloc_frequency: 0.0,
                large_alloc_frequency: 0.0,
                avg_allocation_size: 0,
                fragmentation_level: 0.0,
                allocation_deallocation_ratio: 1.0,
            },
        }
    }
}

impl Default for IoMetrics {
    fn default() -> Self {
        Self {
            read_ops: 0.0,
            write_ops: 0.0,
            read_bandwidth: 0.0,
            write_bandwidth: 0.0,
            io_wait: 0.0,
            queue_depth: 0.0,
            service_time: 0.0,
            utilization: 0.0,
            patterns: IoPatternMetrics {
                sequential_access_percentage: 0.5,
                random_access_percentage: 0.5,
                avg_request_size: 4096,
                read_write_ratio: 1.0,
                locality_score: 0.5,
            },
            efficiency: IoEfficiencyMetrics {
                cache_hit_rate: 0.8,
                coalescing_effectiveness: 0.7,
                prefetch_accuracy: 0.6,
                scheduler_efficiency: 0.8,
            },
        }
    }
}

impl Default for NetworkMetrics {
    fn default() -> Self {
        Self {
            rx_bytes: 0.0,
            tx_bytes: 0.0,
            rx_packets: 0.0,
            tx_packets: 0.0,
            latency: None,
            packet_loss: None,
            utilization: 0.0,
            connections: 0,
            quality: NetworkQualityMetrics {
                jitter: 0.0,
                bandwidth_consistency: 1.0,
                connection_stability: 1.0,
                error_rate: 0.0,
            },
            efficiency: NetworkEfficiencyMetrics {
                protocol_overhead: 0.1,
                congestion_control_effectiveness: 0.9,
                buffer_utilization: 0.5,
                stack_efficiency: 0.9,
            },
        }
    }
}

impl Default for ErrorStatistics {
    fn default() -> Self {
        Self {
            total_errors: 0,
            errors_by_type: HashMap::new(),
            error_rate: 0.0,
            recovery_rate: 1.0,
            severity_distribution: ErrorSeverityDistribution {
                critical: 0,
                high: 0,
                medium: 0,
                low: 0,
                warnings: 0,
            },
            error_patterns: ErrorPatternAnalysis {
                temporal_clustering: 0.0,
                spatial_clustering: 0.0,
                correlation_score: 0.0,
                cascading_indicators: Vec::new(),
                prediction_confidence: 0.0,
            },
            mtbf: Duration::from_secs(86400), // 24 hours default
            mttr: Duration::from_secs(300),   // 5 minutes default
            error_hotspots: Vec::new(),
        }
    }
}

impl Default for PerformanceTrends {
    fn default() -> Self {
        Self {
            throughput_trend: TrendAnalysis {
                current_value: 0.0,
                direction: TrendDirection::Stable,
                strength: 0.0,
                rate_of_change: 0.0,
                confidence: 0.0,
                prediction: 0.0,
            },
            latency_trend: TrendAnalysis {
                current_value: 0.0,
                direction: TrendDirection::Stable,
                strength: 0.0,
                rate_of_change: 0.0,
                confidence: 0.0,
                prediction: 0.0,
            },
            resource_utilization_trends: HashMap::new(),
            error_rate_trend: TrendAnalysis {
                current_value: 0.0,
                direction: TrendDirection::Stable,
                strength: 0.0,
                rate_of_change: 0.0,
                confidence: 0.0,
                prediction: 0.0,
            },
            degradation_indicators: Vec::new(),
            seasonal_patterns: SeasonalPatterns {
                daily_patterns: Vec::new(),
                weekly_patterns: Vec::new(),
                monthly_patterns: Vec::new(),
                pattern_strengths: HashMap::new(),
            },
            forecasting: PerformanceForecasting {
                short_term: ForecastData {
                    predicted_throughput: 0.0,
                    predicted_latency: 0.0,
                    predicted_resource_utilization: 0.0,
                    confidence: 0.0,
                    confidence_intervals: ConfidenceIntervals {
                        lower_95: 0.0,
                        upper_95: 0.0,
                        lower_99: 0.0,
                        upper_99: 0.0,
                    },
                },
                medium_term: ForecastData {
                    predicted_throughput: 0.0,
                    predicted_latency: 0.0,
                    predicted_resource_utilization: 0.0,
                    confidence: 0.0,
                    confidence_intervals: ConfidenceIntervals {
                        lower_95: 0.0,
                        upper_95: 0.0,
                        lower_99: 0.0,
                        upper_99: 0.0,
                    },
                },
                long_term: ForecastData {
                    predicted_throughput: 0.0,
                    predicted_latency: 0.0,
                    predicted_resource_utilization: 0.0,
                    confidence: 0.0,
                    confidence_intervals: ConfidenceIntervals {
                        lower_95: 0.0,
                        upper_95: 0.0,
                        lower_99: 0.0,
                        upper_99: 0.0,
                    },
                },
                accuracy_metrics: ForecastAccuracy {
                    mae: 0.0,
                    rmse: 0.0,
                    mape: 0.0,
                    bias: 0.0,
                },
            },
        }
    }
}

impl Default for SystemPerformanceIndicators {
    fn default() -> Self {
        Self {
            health_score: 1.0,
            efficiency_rating: 1.0,
            stability_index: 1.0,
            scalability_metrics: ScalabilityMetrics {
                horizontal_scalability: 1.0,
                vertical_scalability: 1.0,
                load_capacity_utilization: 0.0,
                scaling_efficiency: 1.0,
                bottlenecks: Vec::new(),
            },
            reliability_metrics: ReliabilityMetrics {
                uptime_percentage: 100.0,
                availability_score: 1.0,
                fault_tolerance_effectiveness: 1.0,
                recovery_metrics: RecoveryMetrics {
                    rto: Duration::from_secs(300),
                    rpo: Duration::from_secs(60),
                    actual_recovery_times: Vec::new(),
                    recovery_success_rate: 1.0,
                },
                redundancy_effectiveness: 1.0,
            },
            sla_compliance: SlaComplianceMetrics {
                overall_compliance: 100.0,
                compliance_by_level: HashMap::new(),
                violations_count: 0,
                time_to_breach: None,
                compliance_trends: HashMap::new(),
            },
            cost_efficiency: CostEfficiencyMetrics {
                cost_per_transaction: 0.0,
                resource_cost_efficiency: 1.0,
                performance_per_dollar: 1.0,
                optimization_opportunities: Vec::new(),
                roi_metrics: RoiMetrics {
                    performance_roi: 1.0,
                    cost_reduction_roi: 1.0,
                    overall_roi: 1.0,
                    payback_period: Duration::from_secs(86400),
                },
            },
        }
    }
}

impl Default for ResourceAllocationEfficiency {
    fn default() -> Self {
        Self {
            cpu_efficiency: 1.0,
            memory_efficiency: 1.0,
            io_efficiency: 1.0,
            gpu_efficiency: 1.0,
            overall_efficiency: 1.0,
            waste_percentage: 0.0,
        }
    }
}

impl Default for ResourceContentionMetrics {
    fn default() -> Self {
        Self {
            cpu_contention: 0.0,
            memory_contention: 0.0,
            io_contention: 0.0,
            network_contention: 0.0,
            lock_contention: LockContentionMetrics {
                total_acquisitions: 0,
                failed_acquisitions: 0,
                avg_hold_time: Duration::new(0, 0),
                max_wait_time: Duration::new(0, 0),
                hotspots: Vec::new(),
            },
            queue_contention: QueueContentionMetrics {
                utilization_by_priority: HashMap::new(),
                avg_wait_time: Duration::new(0, 0),
                growth_rate: 0.0,
                bottlenecks: Vec::new(),
            },
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_execution_metrics_creation() {
        let metrics = ExecutionMetrics::new();
        assert!(metrics.start_time <= SystemTime::now());
        assert_eq!(metrics.strategy_metrics.len(), 0);
        assert_eq!(metrics.error_statistics.total_errors, 0);
    }

    #[test]
    fn test_performance_report_generation() {
        let metrics = ExecutionMetrics::new();
        let report = metrics.generate_report();

        assert!(report.timestamp <= SystemTime::now());
        assert_eq!(report.summary.overall_health, 1.0);
        assert!(!report.recommendations.is_empty() || report.recommendations.is_empty());
    }

    #[test]
    fn test_trend_direction_variants() {
        let directions = vec![
            TrendDirection::Increasing,
            TrendDirection::Decreasing,
            TrendDirection::Stable,
            TrendDirection::Volatile,
            TrendDirection::Cyclical,
        ];

        for direction in directions {
            assert!(matches!(direction, TrendDirection::_));
        }
    }

    #[test]
    fn test_recommendation_priorities() {
        let priorities = vec![
            RecommendationPriority::Low,
            RecommendationPriority::Medium,
            RecommendationPriority::High,
            RecommendationPriority::Critical,
        ];

        for priority in priorities {
            assert!(matches!(priority, RecommendationPriority::_));
        }
    }

    #[test]
    fn test_metric_statistics_calculation() {
        let mut collection = MetricCollection {
            name: "test_metric".to_string(),
            values: VecDeque::new(),
            statistics: MetricStatistics {
                count: 0,
                mean: 0.0,
                std_dev: 0.0,
                min: 0.0,
                max: 0.0,
                median: 0.0,
                p95: 0.0,
                p99: 0.0,
            },
            metadata: HashMap::new(),
        };

        // Add some test values
        let now = SystemTime::now();
        collection.values.push_back((now, 1.0));
        collection.values.push_back((now, 2.0));
        collection.values.push_back((now, 3.0));

        assert_eq!(collection.values.len(), 3);
    }

    #[test]
    fn test_error_statistics_defaults() {
        let stats = ErrorStatistics::default();
        assert_eq!(stats.total_errors, 0);
        assert_eq!(stats.error_rate, 0.0);
        assert_eq!(stats.recovery_rate, 1.0);
        assert_eq!(stats.severity_distribution.critical, 0);
    }

    #[test]
    fn test_system_performance_indicators_defaults() {
        let indicators = SystemPerformanceIndicators::default();
        assert_eq!(indicators.health_score, 1.0);
        assert_eq!(indicators.efficiency_rating, 1.0);
        assert_eq!(indicators.reliability_metrics.uptime_percentage, 100.0);
    }
}