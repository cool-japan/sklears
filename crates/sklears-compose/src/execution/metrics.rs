//! Metrics collection and analysis for the composable execution engine
//!
//! This module provides comprehensive metrics collection, aggregation, and analysis
//! with SIMD-accelerated operations for high-performance monitoring.

use std::collections::HashMap;
use std::time::{Duration, SystemTime};

use super::resources::ResourceMetrics;
use super::strategies::StrategyMetrics;
use super::tasks::{TaskPriority, TaskResult, TaskStatus, TaskType};

/// Overall execution metrics for the engine
#[derive(Debug, Clone)]
pub struct ExecutionMetrics {
    /// Execution start time
    pub start_time: SystemTime,
    /// Strategy metrics by name
    pub strategy_metrics: HashMap<String, StrategyMetrics>,
    /// Scheduler metrics
    pub scheduler_metrics: SchedulerMetrics,
    /// Resource metrics
    pub resource_metrics: ResourceMetrics,
    /// Error statistics
    pub error_statistics: ErrorStatistics,
}

impl Default for ExecutionMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl ExecutionMetrics {
    /// Create new execution metrics
    #[must_use]
    pub fn new() -> Self {
        Self {
            start_time: SystemTime::now(),
            strategy_metrics: HashMap::new(),
            scheduler_metrics: SchedulerMetrics::default(),
            resource_metrics: ResourceMetrics::default(),
            error_statistics: ErrorStatistics::default(),
        }
    }

    /// Update strategy metrics
    pub fn update_strategy_metrics(&mut self, strategy_name: String, metrics: StrategyMetrics) {
        self.strategy_metrics.insert(strategy_name, metrics);
    }

    /// Update scheduler metrics
    pub fn update_scheduler_metrics(&mut self, metrics: SchedulerMetrics) {
        self.scheduler_metrics = metrics;
    }

    /// Update resource metrics
    pub fn update_resource_metrics(&mut self, metrics: ResourceMetrics) {
        self.resource_metrics = metrics;
    }

    /// Record error
    pub fn record_error(&mut self, error_type: String) {
        self.error_statistics.total_errors += 1;
        *self
            .error_statistics
            .errors_by_type
            .entry(error_type)
            .or_insert(0) += 1;

        // Update error rate (simple calculation)
        let total_operations = self
            .strategy_metrics
            .values()
            .map(|m| m.tasks_executed)
            .sum::<u64>();

        if total_operations > 0 {
            self.error_statistics.error_rate =
                self.error_statistics.total_errors as f64 / total_operations as f64;
        }
    }

    /// Get execution duration
    #[must_use]
    pub fn execution_duration(&self) -> Duration {
        self.start_time.elapsed().unwrap_or_default()
    }
}

/// Scheduler metrics for performance tracking
#[derive(Debug, Clone)]
pub struct SchedulerMetrics {
    /// Tasks scheduled
    pub tasks_scheduled: u64,
    /// Average scheduling time
    pub avg_scheduling_time: Duration,
    /// Queue length
    pub queue_length: usize,
    /// Scheduling efficiency
    pub efficiency: f64,
}

impl Default for SchedulerMetrics {
    fn default() -> Self {
        Self {
            tasks_scheduled: 0,
            avg_scheduling_time: Duration::ZERO,
            queue_length: 0,
            efficiency: 1.0,
        }
    }
}

/// Error statistics for monitoring failures
#[derive(Debug, Clone)]
pub struct ErrorStatistics {
    /// Total errors
    pub total_errors: u64,
    /// Errors by type
    pub errors_by_type: HashMap<String, u64>,
    /// Error rate
    pub error_rate: f64,
    /// Recovery rate
    pub recovery_rate: f64,
}

impl Default for ErrorStatistics {
    fn default() -> Self {
        Self {
            total_errors: 0,
            errors_by_type: HashMap::new(),
            error_rate: 0.0,
            recovery_rate: 0.0,
        }
    }
}

/// SIMD-accelerated metrics calculations for high-performance resource monitoring
pub mod simd_metrics {
    use super::{
        AggregatedPerformanceMetrics, Duration, PerformanceBounds, ResourceUtilizationStatistics,
        TaskPriority, TaskType,
    };

    /// Aggregate CPU utilization across multiple cores using SIMD operations
    #[must_use]
    pub fn aggregate_cpu_utilization(per_core_utilizations: &[f64]) -> f64 {
        if per_core_utilizations.is_empty() {
            return 0.0;
        }

        // Convert to f32 for SIMD operations (scalar fallback)
        let core_utils_f32: Vec<f32> = per_core_utilizations.iter().map(|&x| x as f32).collect();

        // Use SIMD mean calculation (scalar fallback)
        f64::from(mean_vec(&core_utils_f32))
    }

    /// Calculate resource utilization statistics using SIMD operations
    #[must_use]
    pub fn calculate_resource_statistics(
        cpu_values: &[f64],
        memory_values: &[f64],
        io_values: &[f64],
        network_values: &[f64],
    ) -> ResourceUtilizationStatistics {
        if cpu_values.is_empty() {
            return ResourceUtilizationStatistics::default();
        }

        // Convert all arrays to f32 for SIMD processing (scalar fallback)
        let cpu_f32: Vec<f32> = cpu_values.iter().map(|&x| x as f32).collect();
        let memory_f32: Vec<f32> = memory_values.iter().map(|&x| x as f32).collect();
        let io_f32: Vec<f32> = io_values.iter().map(|&x| x as f32).collect();
        let network_f32: Vec<f32> = network_values.iter().map(|&x| x as f32).collect();

        ResourceUtilizationStatistics {
            cpu_mean: f64::from(mean_vec(&cpu_f32)),
            cpu_variance: f64::from(variance_vec(&cpu_f32)),
            memory_mean: f64::from(mean_vec(&memory_f32)),
            memory_variance: f64::from(variance_vec(&memory_f32)),
            io_mean: f64::from(mean_vec(&io_f32)),
            io_variance: f64::from(variance_vec(&io_f32)),
            network_mean: f64::from(mean_vec(&network_f32)),
            network_variance: f64::from(variance_vec(&network_f32)),
            sample_count: cpu_values.len(),
        }
    }

    /// Aggregate task performance metrics across multiple tasks using SIMD
    #[must_use]
    pub fn aggregate_task_performance(
        throughputs: &[f64],
        latencies: &[f64],
        cache_hit_rates: &[f64],
        error_rates: &[f64],
    ) -> AggregatedPerformanceMetrics {
        if throughputs.is_empty() {
            return AggregatedPerformanceMetrics::default();
        }

        // Convert to f32 for SIMD processing (scalar fallback)
        let throughput_f32: Vec<f32> = throughputs.iter().map(|&x| x as f32).collect();
        let latency_f32: Vec<f32> = latencies.iter().map(|&x| x as f32).collect();
        let cache_f32: Vec<f32> = cache_hit_rates.iter().map(|&x| x as f32).collect();
        let error_f32: Vec<f32> = error_rates.iter().map(|&x| x as f32).collect();

        // Use SIMD operations for aggregation (scalar fallback)
        let total_throughput = f64::from(sum_vec(&throughput_f32));
        let avg_latency = f64::from(mean_vec(&latency_f32));
        let avg_cache_hit_rate = f64::from(mean_vec(&cache_f32));
        let avg_error_rate = f64::from(mean_vec(&error_f32));

        AggregatedPerformanceMetrics {
            total_throughput,
            average_latency: avg_latency,
            average_cache_hit_rate: avg_cache_hit_rate,
            average_error_rate: avg_error_rate,
            task_count: throughputs.len(),
        }
    }

    /// Calculate weighted resource allocation using SIMD dot product
    #[must_use]
    pub fn calculate_weighted_allocation(
        resource_demands: &[f64],
        priority_weights: &[f64],
    ) -> f64 {
        if resource_demands.len() != priority_weights.len() || resource_demands.is_empty() {
            return 0.0;
        }

        // Convert to f32 for SIMD processing (scalar fallback)
        let demands_f32: Vec<f32> = resource_demands.iter().map(|&x| x as f32).collect();
        let weights_f32: Vec<f32> = priority_weights.iter().map(|&x| x as f32).collect();

        // Use SIMD dot product for weighted sum (scalar fallback)
        f64::from(dot_product(&demands_f32, &weights_f32))
    }

    /// Vectorized min-max analysis for performance bounds
    #[must_use]
    pub fn analyze_performance_bounds(values: &[f64]) -> PerformanceBounds {
        if values.is_empty() {
            return PerformanceBounds::default();
        }

        // Convert to f32 for SIMD processing (scalar fallback)
        let values_f32: Vec<f32> = values.iter().map(|&x| x as f32).collect();

        // Use SIMD min-max operations (scalar fallback)
        let (min_val, max_val) = min_max_vec(&values_f32);

        PerformanceBounds {
            minimum: f64::from(min_val),
            maximum: f64::from(max_val),
            range: f64::from(max_val - min_val),
            mean: f64::from(mean_vec(&values_f32)),
        }
    }

    /// Batch process task priorities using SIMD operations for efficient scheduling
    #[must_use]
    pub fn optimize_batch_scheduling(
        task_priorities: &[TaskPriority],
        estimated_durations: &[Duration],
        resource_requirements: &[f64],
    ) -> Vec<usize> {
        if task_priorities.is_empty() {
            return Vec::new();
        }

        // Convert priorities to numeric values for SIMD processing
        let priority_values: Vec<f32> = task_priorities
            .iter()
            .map(|p| match p {
                TaskPriority::Critical => 4.0,
                TaskPriority::High => 3.0,
                TaskPriority::Normal => 2.0,
                TaskPriority::Low => 1.0,
            })
            .collect();

        // Convert durations to seconds as f32
        let duration_values: Vec<f32> = estimated_durations
            .iter()
            .map(std::time::Duration::as_secs_f32)
            .collect();

        // Ensure resource_requirements is in f32 format
        let resource_f32: Vec<f32> = resource_requirements.iter().map(|&x| x as f32).collect();

        // Calculate weighted priority using SIMD dot product (scalar fallback)
        // Higher priority and lower duration/resource usage gets higher score
        let mut duration_weights = vec![1.0f32; duration_values.len()];
        divide_vec_inplace(&mut duration_weights, &duration_values);

        let mut resource_weights = vec![1.0f32; resource_f32.len()];
        divide_vec_inplace(&mut resource_weights, &resource_f32);

        // Combine all factors using SIMD operations (scalar fallback)
        let mut priority_scores = vec![0.0f32; priority_values.len()];
        multiply_vec(&priority_values, &duration_weights, &mut priority_scores);
        multiply_vec_inplace(&mut priority_scores, &resource_weights);

        // Sort indices by priority scores (descending order)
        let mut indexed_scores: Vec<(usize, f32)> = priority_scores
            .iter()
            .enumerate()
            .map(|(i, &score)| (i, score))
            .collect();

        indexed_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        indexed_scores.into_iter().map(|(i, _)| i).collect()
    }

    /// Vectorized load balancing calculation for distributed execution
    #[must_use]
    pub fn calculate_load_distribution(
        node_capacities: &[f64],
        current_loads: &[f64],
        _task_weights: &[f64],
    ) -> Vec<f64> {
        if node_capacities.is_empty() || current_loads.is_empty() {
            return Vec::new();
        }

        // Convert to f32 for SIMD processing (scalar fallback)
        let capacities_f32: Vec<f32> = node_capacities.iter().map(|&x| x as f32).collect();
        let loads_f32: Vec<f32> = current_loads.iter().map(|&x| x as f32).collect();

        // Calculate available capacity using SIMD subtraction (scalar fallback)
        let mut available_capacity = vec![0.0f32; capacities_f32.len()];
        subtract_vec(&capacities_f32, &loads_f32, &mut available_capacity);

        // Normalize available capacities
        let total_available = sum_vec(&available_capacity);
        if total_available > 0.0 {
            let mut distribution = vec![0.0f32; available_capacity.len()];
            let total_vec = vec![total_available; available_capacity.len()];
            divide_vec(&available_capacity, &total_vec, &mut distribution);
            distribution.into_iter().map(f64::from).collect()
        } else {
            vec![0.0; node_capacities.len()]
        }
    }

    /// SIMD-accelerated task clustering for batch optimization
    #[must_use]
    pub fn cluster_tasks_for_batch_processing(
        task_types: &[TaskType],
        resource_requirements: &[f64],
        similarity_threshold: f64,
    ) -> Vec<Vec<usize>> {
        if task_types.is_empty() {
            return Vec::new();
        }

        let mut clusters = Vec::new();
        let mut assigned = vec![false; task_types.len()];

        for i in 0..task_types.len() {
            if assigned[i] {
                continue;
            }

            let mut cluster = vec![i];
            assigned[i] = true;

            // Find similar tasks using SIMD operations
            for j in (i + 1)..task_types.len() {
                if assigned[j] {
                    continue;
                }

                // Check type similarity
                let type_similar = match (&task_types[i], &task_types[j]) {
                    (TaskType::Computation, TaskType::Computation)
                    | (TaskType::IoOperation, TaskType::IoOperation)
                    | (TaskType::NetworkOperation, TaskType::NetworkOperation)
                    | (TaskType::Custom, TaskType::Custom) => true,
                    _ => false,
                };

                // Check resource similarity using SIMD (scalar fallback)
                if type_similar {
                    let resource_diff = (resource_requirements[i] - resource_requirements[j]).abs();
                    let resource_avg = (resource_requirements[i] + resource_requirements[j]) / 2.0;
                    let similarity = if resource_avg > 0.0 {
                        1.0 - (resource_diff / resource_avg)
                    } else {
                        1.0
                    };

                    if similarity >= similarity_threshold {
                        cluster.push(j);
                        assigned[j] = true;
                    }
                }
            }

            clusters.push(cluster);
        }

        clusters
    }

    // Scalar fallback implementations for SIMD operations
    // These would be replaced with actual SIMD implementations in production

    /// Calculate mean of vector (scalar fallback)
    #[must_use]
    pub fn mean_vec(values: &[f32]) -> f32 {
        if values.is_empty() {
            return 0.0;
        }
        values.iter().sum::<f32>() / values.len() as f32
    }

    /// Calculate variance of vector (scalar fallback)
    #[must_use]
    pub fn variance_vec(values: &[f32]) -> f32 {
        if values.len() <= 1 {
            return 0.0;
        }
        let mean = mean_vec(values);
        let sum_sq_diff: f32 = values.iter().map(|&x| (x - mean).powi(2)).sum();
        sum_sq_diff / (values.len() - 1) as f32
    }

    /// Calculate sum of vector (scalar fallback)
    #[must_use]
    pub fn sum_vec(values: &[f32]) -> f32 {
        values.iter().sum()
    }

    /// Calculate dot product (scalar fallback)
    #[must_use]
    pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return 0.0;
        }
        a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
    }

    /// Calculate min and max of vector (scalar fallback)
    #[must_use]
    pub fn min_max_vec(values: &[f32]) -> (f32, f32) {
        if values.is_empty() {
            return (0.0, 0.0);
        }
        let min_val = values.iter().fold(f32::INFINITY, |acc, &x| acc.min(x));
        let max_val = values.iter().fold(f32::NEG_INFINITY, |acc, &x| acc.max(x));
        (min_val, max_val)
    }

    /// Element-wise division with in-place mutation (scalar fallback)
    pub fn divide_vec_inplace(numerator: &mut [f32], denominator: &[f32]) {
        for (n, &d) in numerator.iter_mut().zip(denominator.iter()) {
            if d != 0.0 {
                *n /= d;
            }
        }
    }

    /// Element-wise multiplication (scalar fallback)
    pub fn multiply_vec(a: &[f32], b: &[f32], result: &mut [f32]) {
        for ((r, &a_val), &b_val) in result.iter_mut().zip(a.iter()).zip(b.iter()) {
            *r = a_val * b_val;
        }
    }

    /// Element-wise multiplication with in-place mutation (scalar fallback)
    pub fn multiply_vec_inplace(a: &mut [f32], b: &[f32]) {
        for (a_val, &b_val) in a.iter_mut().zip(b.iter()) {
            *a_val *= b_val;
        }
    }

    /// Element-wise subtraction (scalar fallback)
    pub fn subtract_vec(a: &[f32], b: &[f32], result: &mut [f32]) {
        for ((r, &a_val), &b_val) in result.iter_mut().zip(a.iter()).zip(b.iter()) {
            *r = a_val - b_val;
        }
    }

    /// Element-wise division (scalar fallback)
    pub fn divide_vec(numerator: &[f32], denominator: &[f32], result: &mut [f32]) {
        for ((r, &n), &d) in result
            .iter_mut()
            .zip(numerator.iter())
            .zip(denominator.iter())
        {
            *r = if d == 0.0 { 0.0 } else { n / d };
        }
    }
}

/// Resource utilization statistics computed using SIMD operations
#[derive(Debug, Clone)]
pub struct ResourceUtilizationStatistics {
    /// CPU mean utilization
    pub cpu_mean: f64,
    /// CPU utilization variance
    pub cpu_variance: f64,
    /// Memory mean utilization
    pub memory_mean: f64,
    /// Memory utilization variance
    pub memory_variance: f64,
    /// I/O mean utilization
    pub io_mean: f64,
    /// I/O utilization variance
    pub io_variance: f64,
    /// Network mean utilization
    pub network_mean: f64,
    /// Network utilization variance
    pub network_variance: f64,
    /// Number of samples
    pub sample_count: usize,
}

impl Default for ResourceUtilizationStatistics {
    fn default() -> Self {
        Self {
            cpu_mean: 0.0,
            cpu_variance: 0.0,
            memory_mean: 0.0,
            memory_variance: 0.0,
            io_mean: 0.0,
            io_variance: 0.0,
            network_mean: 0.0,
            network_variance: 0.0,
            sample_count: 0,
        }
    }
}

/// Aggregated performance metrics computed using SIMD operations
#[derive(Debug, Clone)]
pub struct AggregatedPerformanceMetrics {
    /// Total throughput across all tasks
    pub total_throughput: f64,
    /// Average latency
    pub average_latency: f64,
    /// Average cache hit rate
    pub average_cache_hit_rate: f64,
    /// Average error rate
    pub average_error_rate: f64,
    /// Number of tasks analyzed
    pub task_count: usize,
}

impl Default for AggregatedPerformanceMetrics {
    fn default() -> Self {
        Self {
            total_throughput: 0.0,
            average_latency: 0.0,
            average_cache_hit_rate: 0.0,
            average_error_rate: 0.0,
            task_count: 0,
        }
    }
}

/// Performance bounds analysis using SIMD operations
#[derive(Debug, Clone)]
pub struct PerformanceBounds {
    /// Minimum performance value
    pub minimum: f64,
    /// Maximum performance value
    pub maximum: f64,
    /// Performance range (max - min)
    pub range: f64,
    /// Mean performance value
    pub mean: f64,
}

impl Default for PerformanceBounds {
    fn default() -> Self {
        Self {
            minimum: 0.0,
            maximum: 0.0,
            range: 0.0,
            mean: 0.0,
        }
    }
}

/// Metrics collector for gathering execution metrics
pub struct MetricsCollector {
    /// Collected metrics
    metrics: ExecutionMetrics,
    /// Collection interval
    interval: Duration,
    /// Enable detailed collection
    detailed: bool,
}

impl MetricsCollector {
    /// Create a new metrics collector
    #[must_use]
    pub fn new(interval: Duration, detailed: bool) -> Self {
        Self {
            metrics: ExecutionMetrics::new(),
            interval,
            detailed,
        }
    }

    /// Collect current metrics
    pub fn collect(&mut self) -> ExecutionMetrics {
        // In a real implementation, this would gather metrics from various sources
        self.metrics.clone()
    }

    /// Record task completion
    pub fn record_task_completion(&mut self, task_result: &TaskResult) {
        if let Some(strategy) = self.metrics.strategy_metrics.get_mut("default") {
            strategy.tasks_executed += 1;
            match task_result.status {
                TaskStatus::Completed => {
                    // Update success metrics
                }
                TaskStatus::Failed => {
                    self.metrics
                        .record_error("task_execution_error".to_string());
                }
                _ => {}
            }
        }
    }

    /// Update configuration
    pub fn update_config(&mut self, interval: Duration, detailed: bool) {
        self.interval = interval;
        self.detailed = detailed;
    }

    /// Get metrics interval
    #[must_use]
    pub fn get_interval(&self) -> Duration {
        self.interval
    }

    /// Check if detailed collection is enabled
    #[must_use]
    pub fn is_detailed(&self) -> bool {
        self.detailed
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
    fn test_simd_metrics_cpu_aggregation() {
        let per_core_utils = vec![75.0, 80.0, 85.0, 70.0, 90.0, 88.0, 72.0, 78.0];
        let avg_util = simd_metrics::aggregate_cpu_utilization(&per_core_utils);

        // Should be close to the arithmetic mean (79.75)
        assert!((avg_util - 79.75).abs() < 0.1);
    }

    #[test]
    fn test_simd_metrics_resource_statistics() {
        let cpu_values = vec![75.0, 80.0, 85.0, 70.0];
        let memory_values = vec![60.0, 65.0, 70.0, 55.0];
        let io_values = vec![40.0, 45.0, 35.0, 50.0];
        let network_values = vec![20.0, 25.0, 30.0, 15.0];

        let stats = simd_metrics::calculate_resource_statistics(
            &cpu_values,
            &memory_values,
            &io_values,
            &network_values,
        );

        assert_eq!(stats.sample_count, 4);
        assert!((stats.cpu_mean - 77.5).abs() < 0.1);
        assert!((stats.memory_mean - 62.5).abs() < 0.1);
        assert!(stats.cpu_variance > 0.0); // Should have some variance
    }

    #[test]
    fn test_simd_batch_scheduling_optimization() {
        let priorities = vec![
            TaskPriority::High,
            TaskPriority::Low,
            TaskPriority::Critical,
            TaskPriority::Normal,
        ];
        let durations = vec![
            Duration::from_secs(10),
            Duration::from_secs(60),
            Duration::from_secs(5),
            Duration::from_secs(30),
        ];
        let resources = vec![100.0, 500.0, 50.0, 200.0];

        let optimized_order =
            simd_metrics::optimize_batch_scheduling(&priorities, &durations, &resources);

        // Critical priority task with short duration should be first
        assert_eq!(optimized_order[0], 2); // Critical task with 5 second duration
        assert_eq!(optimized_order.len(), 4);
    }

    #[test]
    fn test_simd_load_distribution() {
        let capacities = vec![100.0, 150.0, 120.0, 80.0];
        let current_loads = vec![20.0, 30.0, 40.0, 10.0];
        let task_weights = vec![10.0, 15.0, 12.0, 8.0];

        let distribution =
            simd_metrics::calculate_load_distribution(&capacities, &current_loads, &task_weights);

        assert_eq!(distribution.len(), 4);

        // Sum of distribution should be approximately 1.0 (normalized)
        let total: f64 = distribution.iter().sum();
        assert!((total - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_simd_performance_bounds_analysis() {
        let performance_values = vec![95.5, 87.2, 91.8, 89.4, 93.1, 88.7, 94.3];

        let bounds = simd_metrics::analyze_performance_bounds(&performance_values);

        assert!(bounds.minimum < bounds.maximum);
        assert!(bounds.range > 0.0);
        assert!(bounds.mean > 0.0);
        assert!(bounds.mean >= bounds.minimum && bounds.mean <= bounds.maximum);
    }

    #[test]
    fn test_metrics_collector() {
        let mut collector = MetricsCollector::new(Duration::from_secs(1), false);

        assert_eq!(collector.get_interval(), Duration::from_secs(1));
        assert!(!collector.is_detailed());

        let metrics = collector.collect();
        assert!(metrics.start_time <= SystemTime::now());

        collector.update_config(Duration::from_secs(5), true);
        assert_eq!(collector.get_interval(), Duration::from_secs(5));
        assert!(collector.is_detailed());
    }
}
