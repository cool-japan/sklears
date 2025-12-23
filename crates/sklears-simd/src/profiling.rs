//! Performance analysis and profiling tools
//!
//! This module provides comprehensive profiling capabilities for SIMD operations,
//! including instruction-level profiling, cache analysis, and vectorization efficiency metrics.

#[cfg(not(feature = "no-std"))]
use std::{
    collections::HashMap,
    string::ToString,
    sync::atomic::{AtomicU64, Ordering},
    time::{Duration, Instant},
};

#[cfg(feature = "no-std")]
use alloc::{
    collections::BTreeMap as HashMap,
    format,
    string::{String, ToString},
    vec,
    vec::Vec,
};
#[cfg(feature = "no-std")]
use core::sync::atomic::{AtomicU64, Ordering};

// Type aliases for conditional compilation (reusing from performance_hooks)
#[cfg(feature = "no-std")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Duration(u64); // Mock duration in microseconds
#[cfg(feature = "no-std")]
#[derive(Debug, Clone, Copy)]
pub struct Instant(u64); // Mock instant as counter

#[cfg(feature = "no-std")]
impl Instant {
    pub fn now() -> Self {
        Instant(0) // Mock implementation
    }

    pub fn elapsed(&self) -> Duration {
        Duration(0) // Mock implementation
    }
}

#[cfg(feature = "no-std")]
impl Duration {
    pub fn as_nanos(&self) -> u128 {
        self.0 as u128 * 1000 // Mock implementation
    }

    pub fn from_nanos(nanos: u64) -> Self {
        Duration(nanos / 1000) // Convert nanos to mock microseconds
    }

    pub fn from_micros(micros: u64) -> Self {
        Duration(micros) // Mock implementation - directly use microseconds
    }

    pub fn as_micros(&self) -> u128 {
        self.0 as u128 // Mock implementation
    }
}

#[cfg(feature = "no-std")]
impl core::ops::Add for Duration {
    type Output = Duration;

    fn add(self, rhs: Duration) -> Self::Output {
        Duration(self.0 + rhs.0)
    }
}

#[cfg(feature = "no-std")]
impl core::ops::Div<u32> for Duration {
    type Output = Duration;

    fn div(self, rhs: u32) -> Self::Output {
        Duration(self.0 / rhs as u64)
    }
}

#[cfg(feature = "no-std")]
impl core::iter::Sum for Duration {
    fn sum<I: Iterator<Item = Duration>>(iter: I) -> Self {
        Duration(iter.map(|d| d.0).sum())
    }
}

#[cfg(feature = "no-std")]
impl<'a> core::iter::Sum<&'a Duration> for Duration {
    fn sum<I: Iterator<Item = &'a Duration>>(iter: I) -> Self {
        Duration(iter.map(|d| d.0).sum())
    }
}

/// Performance counter for tracking SIMD operation metrics
#[derive(Debug, Clone)]
pub struct SimdProfiler {
    /// Execution time measurements for different operations
    operation_times: HashMap<String, Vec<Duration>>,
    /// Instruction counts for SIMD vs scalar operations
    instruction_counts: HashMap<String, InstructionCount>,
    /// Cache performance metrics
    cache_metrics: CacheMetrics,
    /// Vectorization efficiency tracking
    vectorization_metrics: VectorizationMetrics,
}

/// Instruction count tracking for performance analysis
#[derive(Debug, Clone, Default)]
pub struct InstructionCount {
    /// Number of SIMD instructions executed
    pub simd_instructions: u64,
    /// Number of scalar instructions executed
    pub scalar_instructions: u64,
    /// Number of memory load operations
    pub memory_loads: u64,
    /// Number of memory store operations
    pub memory_stores: u64,
    /// Number of branch instructions
    pub branches: u64,
}

/// Cache performance metrics
#[derive(Debug, Clone, Default)]
pub struct CacheMetrics {
    /// L1 cache hit rate (0.0 - 1.0)
    pub l1_hit_rate: f64,
    /// L2 cache hit rate (0.0 - 1.0)
    pub l2_hit_rate: f64,
    /// L3 cache hit rate (0.0 - 1.0)
    pub l3_hit_rate: f64,
    /// Total cache misses
    pub total_misses: u64,
    /// Memory bandwidth utilization (bytes/second)
    pub bandwidth_utilization: f64,
}

/// Vectorization efficiency metrics
#[derive(Debug, Clone, Default)]
pub struct VectorizationMetrics {
    /// Percentage of operations that were vectorized (0.0 - 1.0)
    pub vectorization_rate: f64,
    /// SIMD lane utilization efficiency (0.0 - 1.0)
    pub lane_utilization: f64,
    /// Theoretical vs actual throughput ratio
    pub throughput_efficiency: f64,
    /// Number of elements processed per SIMD operation
    pub elements_per_operation: f64,
}

/// Performance bottleneck identification
#[derive(Debug, Clone)]
pub struct BottleneckAnalysis {
    /// Primary bottleneck type
    pub primary_bottleneck: BottleneckType,
    /// Performance limiters in order of impact
    pub limiters: Vec<(BottleneckType, f64)>,
    /// Optimization recommendations
    pub recommendations: Vec<String>,
}

/// Types of performance bottlenecks
#[derive(Debug, Clone, PartialEq)]
pub enum BottleneckType {
    /// Computation-bound (CPU intensive)
    Compute,
    /// Memory bandwidth limited
    MemoryBandwidth,
    /// Memory latency limited
    MemoryLatency,
    /// Cache miss limited
    CacheMiss,
    /// Branch prediction limited
    BranchPrediction,
    /// SIMD lane underutilization
    SimdUnderutilization,
    /// Instruction dependency chains
    InstructionDependency,
}

/// Global performance counter for thread-safe profiling
static GLOBAL_OPERATION_COUNT: AtomicU64 = AtomicU64::new(0);
static GLOBAL_SIMD_COUNT: AtomicU64 = AtomicU64::new(0);
static GLOBAL_SCALAR_COUNT: AtomicU64 = AtomicU64::new(0);

impl Default for SimdProfiler {
    fn default() -> Self {
        Self::new()
    }
}

impl SimdProfiler {
    /// Create a new SIMD profiler instance
    pub fn new() -> Self {
        Self {
            operation_times: HashMap::new(),
            instruction_counts: HashMap::new(),
            cache_metrics: CacheMetrics::default(),
            vectorization_metrics: VectorizationMetrics::default(),
        }
    }

    /// Start profiling a SIMD operation
    pub fn start_operation(&mut self, operation_name: &str) -> OperationProfiler {
        OperationProfiler::new(operation_name.to_string())
    }

    /// Record execution time for an operation
    pub fn record_time(&mut self, operation: &str, duration: Duration) {
        self.operation_times
            .entry(operation.to_string())
            .or_default()
            .push(duration);
    }

    /// Record instruction counts for an operation
    pub fn record_instructions(&mut self, operation: &str, counts: InstructionCount) {
        self.instruction_counts
            .insert(operation.to_string(), counts);
    }

    /// Update cache metrics
    pub fn update_cache_metrics(&mut self, metrics: CacheMetrics) {
        self.cache_metrics = metrics;
    }

    /// Update vectorization metrics
    pub fn update_vectorization_metrics(&mut self, metrics: VectorizationMetrics) {
        self.vectorization_metrics = metrics;
    }

    /// Get average execution time for an operation
    pub fn average_time(&self, operation: &str) -> Option<Duration> {
        self.operation_times.get(operation).map(|times| {
            let total: Duration = times.iter().sum();
            total / times.len() as u32
        })
    }

    /// Get operation statistics
    pub fn get_statistics(&self, operation: &str) -> Option<OperationStats> {
        self.operation_times.get(operation).map(|times| {
            let count = times.len();
            let total: Duration = times.iter().sum();
            let average = total / count as u32;

            let mut sorted_times = times.clone();
            sorted_times.sort();

            let median = if count % 2 == 0 {
                (sorted_times[count / 2 - 1] + sorted_times[count / 2]) / 2
            } else {
                sorted_times[count / 2]
            };

            let min = *sorted_times.first().unwrap();
            let max = *sorted_times.last().unwrap();

            OperationStats {
                count,
                total,
                average,
                median,
                min,
                max,
                std_deviation: self.calculate_std_deviation(times, average),
            }
        })
    }

    /// Calculate standard deviation of execution times
    fn calculate_std_deviation(&self, times: &[Duration], average: Duration) -> Duration {
        if times.len() <= 1 {
            return Duration::from_nanos(0);
        }

        let variance: f64 = times
            .iter()
            .map(|&time| {
                let diff = time.as_nanos() as f64 - average.as_nanos() as f64;
                diff * diff
            })
            .sum::<f64>()
            / times.len() as f64;

        Duration::from_nanos(variance.sqrt() as u64)
    }

    /// Analyze performance bottlenecks
    pub fn analyze_bottlenecks(&self) -> BottleneckAnalysis {
        let mut limiters = Vec::new();

        // Analyze vectorization efficiency
        if self.vectorization_metrics.vectorization_rate < 0.7 {
            limiters.push((
                BottleneckType::SimdUnderutilization,
                1.0 - self.vectorization_metrics.vectorization_rate,
            ));
        }

        // Analyze cache performance
        if self.cache_metrics.l1_hit_rate < 0.9 {
            limiters.push((
                BottleneckType::CacheMiss,
                1.0 - self.cache_metrics.l1_hit_rate,
            ));
        }

        // Analyze memory bandwidth utilization
        if self.cache_metrics.bandwidth_utilization < 0.8 {
            limiters.push((
                BottleneckType::MemoryBandwidth,
                1.0 - self.cache_metrics.bandwidth_utilization,
            ));
        }

        // Sort limiters by impact
        limiters.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let primary_bottleneck = limiters
            .first()
            .map(|(t, _)| t.clone())
            .unwrap_or(BottleneckType::Compute);

        let recommendations = self.generate_recommendations(&limiters);

        BottleneckAnalysis {
            primary_bottleneck,
            limiters,
            recommendations,
        }
    }

    /// Generate optimization recommendations based on bottleneck analysis
    fn generate_recommendations(&self, limiters: &[(BottleneckType, f64)]) -> Vec<String> {
        let mut recommendations = Vec::new();

        for (bottleneck_type, impact) in limiters {
            match bottleneck_type {
                BottleneckType::SimdUnderutilization => {
                    recommendations.push(format!(
                        "Improve SIMD utilization (current: {:.1}%): Consider wider SIMD instructions or better data layout",
                        self.vectorization_metrics.vectorization_rate * 100.0
                    ));
                }
                BottleneckType::CacheMiss => {
                    recommendations.push(format!(
                        "Reduce cache misses (impact: {:.1}%): Improve data locality or use cache-friendly algorithms",
                        impact * 100.0
                    ));
                }
                BottleneckType::MemoryBandwidth => {
                    recommendations.push(format!(
                        "Optimize memory bandwidth (utilization: {:.1}%): Use prefetching or reduce memory traffic",
                        self.cache_metrics.bandwidth_utilization * 100.0
                    ));
                }
                BottleneckType::BranchPrediction => {
                    recommendations.push(
                        "Reduce branching: Use branchless algorithms or improve predictability"
                            .to_string(),
                    );
                }
                _ => {}
            }
        }

        recommendations
    }

    /// Generate comprehensive performance report
    pub fn generate_report(&self) -> String {
        let mut report = String::new();
        report.push_str("=== SIMD Performance Analysis Report ===\n\n");

        // Operation timing summary
        report.push_str("## Operation Performance Summary\n");
        for operation in self.operation_times.keys() {
            if let Some(stats) = self.get_statistics(operation) {
                report.push_str(&format!(
                    "{}: avg={:.2}μs, min={:.2}μs, max={:.2}μs, count={}\n",
                    operation,
                    stats.average.as_micros(),
                    stats.min.as_micros(),
                    stats.max.as_micros(),
                    stats.count
                ));
            }
        }

        // Vectorization metrics
        report.push_str(&format!(
            "\n## Vectorization Efficiency\n\
            Vectorization Rate: {:.1}%\n\
            Lane Utilization: {:.1}%\n\
            Throughput Efficiency: {:.1}%\n",
            self.vectorization_metrics.vectorization_rate * 100.0,
            self.vectorization_metrics.lane_utilization * 100.0,
            self.vectorization_metrics.throughput_efficiency * 100.0
        ));

        // Cache performance
        report.push_str(&format!(
            "\n## Cache Performance\n\
            L1 Hit Rate: {:.1}%\n\
            L2 Hit Rate: {:.1}%\n\
            L3 Hit Rate: {:.1}%\n\
            Bandwidth Utilization: {:.1}%\n",
            self.cache_metrics.l1_hit_rate * 100.0,
            self.cache_metrics.l2_hit_rate * 100.0,
            self.cache_metrics.l3_hit_rate * 100.0,
            self.cache_metrics.bandwidth_utilization * 100.0
        ));

        // Bottleneck analysis
        let analysis = self.analyze_bottlenecks();
        report.push_str(&format!(
            "\n## Bottleneck Analysis\n\
            Primary Bottleneck: {:?}\n",
            analysis.primary_bottleneck
        ));

        report.push_str("\n## Optimization Recommendations\n");
        for (i, recommendation) in analysis.recommendations.iter().enumerate() {
            report.push_str(&format!("{}. {}\n", i + 1, recommendation));
        }

        report
    }
}

/// Statistics for a specific operation
#[derive(Debug, Clone)]
pub struct OperationStats {
    pub count: usize,
    pub total: Duration,
    pub average: Duration,
    pub median: Duration,
    pub min: Duration,
    pub max: Duration,
    pub std_deviation: Duration,
}

/// Individual operation profiler for timing measurements
pub struct OperationProfiler {
    operation_name: String,
    start_time: Instant,
    instruction_count: InstructionCount,
}

impl OperationProfiler {
    /// Create a new operation profiler
    pub fn new(operation_name: String) -> Self {
        GLOBAL_OPERATION_COUNT.fetch_add(1, Ordering::Relaxed);

        Self {
            operation_name,
            start_time: Instant::now(),
            instruction_count: InstructionCount::default(),
        }
    }

    /// Record a SIMD instruction execution
    pub fn record_simd_instruction(&mut self) {
        self.instruction_count.simd_instructions += 1;
        GLOBAL_SIMD_COUNT.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a scalar instruction execution
    pub fn record_scalar_instruction(&mut self) {
        self.instruction_count.scalar_instructions += 1;
        GLOBAL_SCALAR_COUNT.fetch_add(1, Ordering::Relaxed);
    }

    /// Record memory operations
    pub fn record_memory_load(&mut self) {
        self.instruction_count.memory_loads += 1;
    }

    pub fn record_memory_store(&mut self) {
        self.instruction_count.memory_stores += 1;
    }

    /// Finish profiling and return results
    pub fn finish(self) -> (Duration, InstructionCount) {
        let duration = self.start_time.elapsed();
        (duration, self.instruction_count)
    }
}

/// Cache-aware algorithm performance analyzer
pub struct CacheAnalyzer {
    /// Cache sizes for different levels (in bytes)
    cache_sizes: Vec<usize>,
    /// Cache line size (typically 64 bytes)
    cache_line_size: usize,
}

impl Default for CacheAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl CacheAnalyzer {
    /// Create a new cache analyzer with typical x86-64 cache hierarchy
    pub fn new() -> Self {
        Self {
            cache_sizes: vec![32 * 1024, 256 * 1024, 8 * 1024 * 1024], // L1, L2, L3
            cache_line_size: 64,
        }
    }

    /// Analyze cache efficiency for a given data access pattern
    pub fn analyze_access_pattern(&self, data_size: usize, stride: usize) -> CacheAnalysis {
        let cache_lines_accessed = (data_size + self.cache_line_size - 1) / self.cache_line_size;

        // Estimate cache misses based on stride and cache sizes
        let l1_working_set = cache_lines_accessed * self.cache_line_size;
        let l1_fit = l1_working_set <= self.cache_sizes[0];
        let l2_fit = l1_working_set <= self.cache_sizes[1];
        let l3_fit = l1_working_set <= self.cache_sizes[2];

        let estimated_l1_hit_rate = if l1_fit { 0.95 } else { 0.1 };
        let estimated_l2_hit_rate = if l2_fit { 0.9 } else { 0.3 };
        let estimated_l3_hit_rate = if l3_fit { 0.8 } else { 0.1 };

        CacheAnalysis {
            l1_hit_rate: estimated_l1_hit_rate,
            l2_hit_rate: estimated_l2_hit_rate,
            l3_hit_rate: estimated_l3_hit_rate,
            cache_lines_accessed,
            working_set_size: l1_working_set,
            stride_efficiency: self.calculate_stride_efficiency(stride),
        }
    }

    /// Calculate stride efficiency (how cache-friendly the access pattern is)
    fn calculate_stride_efficiency(&self, stride: usize) -> f64 {
        if stride <= self.cache_line_size {
            1.0 // Perfect locality
        } else if stride <= self.cache_line_size * 2 {
            0.8 // Good locality
        } else if stride <= self.cache_line_size * 4 {
            0.6 // Moderate locality
        } else {
            0.3 // Poor locality
        }
    }
}

/// Cache analysis results
#[derive(Debug, Clone)]
pub struct CacheAnalysis {
    pub l1_hit_rate: f64,
    pub l2_hit_rate: f64,
    pub l3_hit_rate: f64,
    pub cache_lines_accessed: usize,
    pub working_set_size: usize,
    pub stride_efficiency: f64,
}

/// Vectorization efficiency analyzer
pub struct VectorizationAnalyzer;

impl VectorizationAnalyzer {
    /// Analyze vectorization efficiency for a given operation
    pub fn analyze_operation(
        elements_processed: usize,
        simd_width: usize,
        actual_simd_ops: usize,
        scalar_ops: usize,
    ) -> VectorizationAnalysis {
        let theoretical_simd_ops = (elements_processed + simd_width - 1) / simd_width;
        let total_ops = actual_simd_ops + scalar_ops;

        let vectorization_rate = if total_ops > 0 {
            actual_simd_ops as f64 / total_ops as f64
        } else {
            0.0
        };

        let lane_utilization = if actual_simd_ops > 0 {
            elements_processed as f64 / (actual_simd_ops * simd_width) as f64
        } else {
            0.0
        };

        let throughput_efficiency = if theoretical_simd_ops > 0 {
            actual_simd_ops as f64 / theoretical_simd_ops as f64
        } else {
            0.0
        };

        VectorizationAnalysis {
            vectorization_rate,
            lane_utilization,
            throughput_efficiency,
            theoretical_simd_ops,
            actual_simd_ops,
            scalar_fallback_ops: scalar_ops,
        }
    }
}

/// Vectorization analysis results
#[derive(Debug, Clone)]
pub struct VectorizationAnalysis {
    pub vectorization_rate: f64,
    pub lane_utilization: f64,
    pub throughput_efficiency: f64,
    pub theoretical_simd_ops: usize,
    pub actual_simd_ops: usize,
    pub scalar_fallback_ops: usize,
}

/// Global profiling statistics
pub fn get_global_stats() -> GlobalStats {
    GlobalStats {
        total_operations: GLOBAL_OPERATION_COUNT.load(Ordering::Relaxed),
        total_simd_instructions: GLOBAL_SIMD_COUNT.load(Ordering::Relaxed),
        total_scalar_instructions: GLOBAL_SCALAR_COUNT.load(Ordering::Relaxed),
    }
}

/// Global profiling statistics
#[derive(Debug, Clone)]
pub struct GlobalStats {
    pub total_operations: u64,
    pub total_simd_instructions: u64,
    pub total_scalar_instructions: u64,
}

impl GlobalStats {
    /// Get the SIMD vs scalar instruction ratio
    pub fn simd_ratio(&self) -> f64 {
        let total = self.total_simd_instructions + self.total_scalar_instructions;
        if total > 0 {
            self.total_simd_instructions as f64 / total as f64
        } else {
            0.0
        }
    }
}

#[allow(non_snake_case)]
#[cfg(all(test, not(feature = "no-std")))]
mod tests {
    use super::*;
    #[cfg(not(feature = "no-std"))]
    use std::time::Duration;

    #[cfg(feature = "no-std")]
    use alloc::{
        string::{String, ToString},
        vec,
        vec::Vec,
    };

    #[test]
    fn test_profiler_basic_functionality() {
        let mut profiler = SimdProfiler::new();

        // Record some operation times
        profiler.record_time("vector_add", Duration::from_micros(10));
        profiler.record_time("vector_add", Duration::from_micros(12));
        profiler.record_time("vector_add", Duration::from_micros(8));

        let avg_time = profiler.average_time("vector_add").unwrap();
        assert!(avg_time >= Duration::from_micros(8));
        assert!(avg_time <= Duration::from_micros(12));

        let stats = profiler.get_statistics("vector_add").unwrap();
        assert_eq!(stats.count, 3);
        assert_eq!(stats.min, Duration::from_micros(8));
        assert_eq!(stats.max, Duration::from_micros(12));
    }

    #[test]
    fn test_operation_profiler() {
        let mut op_profiler = OperationProfiler::new("test_op".to_string());

        op_profiler.record_simd_instruction();
        op_profiler.record_simd_instruction();
        op_profiler.record_scalar_instruction();
        op_profiler.record_memory_load();

        let (duration, counts) = op_profiler.finish();

        assert!(duration >= Duration::from_nanos(0));
        assert_eq!(counts.simd_instructions, 2);
        assert_eq!(counts.scalar_instructions, 1);
        assert_eq!(counts.memory_loads, 1);
    }

    #[test]
    fn test_cache_analyzer() {
        let analyzer = CacheAnalyzer::new();

        // Small data should fit in L1 cache
        let analysis = analyzer.analyze_access_pattern(16 * 1024, 4);
        assert!(analysis.l1_hit_rate > 0.9);
        assert_eq!(analysis.stride_efficiency, 1.0);

        // Large stride should have poor efficiency
        let analysis = analyzer.analyze_access_pattern(64 * 1024, 1024);
        assert!(analysis.stride_efficiency < 0.5);
    }

    #[test]
    fn test_vectorization_analyzer() {
        let analysis = VectorizationAnalyzer::analyze_operation(
            1000, // elements processed
            8,    // SIMD width
            120,  // actual SIMD ops (should be 125 theoretical)
            10,   // scalar ops
        );

        assert!(analysis.vectorization_rate > 0.9); // High vectorization
        assert!(analysis.lane_utilization > 0.95); // Good lane utilization
        assert!(analysis.throughput_efficiency > 0.9); // Good efficiency
    }

    #[test]
    fn test_bottleneck_analysis() {
        let mut profiler = SimdProfiler::new();

        // Set up good cache metrics so SIMD becomes the primary bottleneck
        profiler.update_cache_metrics(CacheMetrics {
            l1_hit_rate: 0.95, // Good cache performance
            l2_hit_rate: 0.9,
            l3_hit_rate: 0.85,
            total_misses: 100,
            bandwidth_utilization: 0.9, // Good bandwidth utilization
        });

        // Set up poor vectorization metrics
        profiler.update_vectorization_metrics(VectorizationMetrics {
            vectorization_rate: 0.3, // Poor vectorization
            lane_utilization: 0.5,
            throughput_efficiency: 0.4,
            elements_per_operation: 2.0,
        });

        let analysis = profiler.analyze_bottlenecks();
        assert_eq!(
            analysis.primary_bottleneck,
            BottleneckType::SimdUnderutilization
        );
        assert!(!analysis.recommendations.is_empty());
    }

    #[test]
    fn test_global_stats() {
        // Create some operations to test with
        let _profiler1 = OperationProfiler::new("test_op1".to_string());
        let mut profiler2 = OperationProfiler::new("test_op2".to_string());
        profiler2.record_simd_instruction();
        profiler2.record_scalar_instruction();

        let stats = get_global_stats();
        assert!(stats.total_operations >= 2); // At least the operations we just created

        let simd_ratio = stats.simd_ratio();
        assert!(simd_ratio >= 0.0 && simd_ratio <= 1.0);
    }

    #[test]
    fn test_performance_report_generation() {
        let mut profiler = SimdProfiler::new();

        profiler.record_time("test_operation", Duration::from_micros(100));
        profiler.update_vectorization_metrics(VectorizationMetrics {
            vectorization_rate: 0.85,
            lane_utilization: 0.92,
            throughput_efficiency: 0.88,
            elements_per_operation: 7.5,
        });

        let report = profiler.generate_report();
        assert!(report.contains("SIMD Performance Analysis Report"));
        assert!(report.contains("Vectorization Rate: 85.0%"));
        assert!(report.contains("Lane Utilization: 92.0%"));
    }

    #[test]
    fn test_instruction_count_tracking() {
        let mut count = InstructionCount::default();
        count.simd_instructions = 100;
        count.scalar_instructions = 50;
        count.memory_loads = 75;
        count.memory_stores = 25;
        count.branches = 10;

        // Verify all fields are tracked correctly
        assert_eq!(count.simd_instructions, 100);
        assert_eq!(count.scalar_instructions, 50);
        assert_eq!(count.memory_loads, 75);
        assert_eq!(count.memory_stores, 25);
        assert_eq!(count.branches, 10);
    }
}
