//! Execution Utilities and Helper Functions
//!
//! This module provides utility functions, helpers, and common functionality
//! used across the composable execution system. It includes:
//!
//! - **Configuration Builders**: Helper builders for creating configurations
//! - **Validation Utilities**: Functions for validating execution parameters
//! - **Conversion Helpers**: Type conversion and transformation utilities
//! - **Metrics Helpers**: Common metrics calculation and aggregation functions
//! - **Test Utilities**: Helper functions for creating test fixtures and mocks
//! - **Resource Calculators**: Utilities for resource estimation and calculation
//! - **Time Utilities**: Duration and timing helper functions
//! - **Error Utilities**: Common error handling and recovery helpers

use crate::execution_types::{
    ExecutionTask, TaskResult, TaskStatus, TaskType, TaskPriority, TaskHandle,
    TaskMetadata, TaskRequirements, TaskConstraints, TaskError,
    TaskExecutionMetrics, TaskResourceUsage, TaskPerformanceMetrics,
    ExecutionContext, ResourceConstraints, PerformanceGoals,
    FaultToleranceConfig, MonitoringConfig, BackoffStrategy
};
use crate::resource_management::{
    ResourceManager, AvailableResources, ResourceAllocation,
    ResourceUsage, ResourceMonitoring
};
use crate::task_scheduling::{
    SchedulerConfig, SchedulerStatus, QueueStatistics
};
use crate::execution_strategies::{
    StrategyConfig, StrategyMetrics, StrategyHealth
};
use crate::performance_monitoring::{
    PerformanceMetrics, SystemPerformanceIndicators,
    MetricsCollection, PerformanceTrend, PerformanceAlert
};
use sklears_core::SklResult;
use std::time::{Duration, SystemTime, Instant};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

/// Configuration builder utilities for creating execution configurations
pub mod config_builders {
    use super::*;

    /// Builder for ExecutionContext with sensible defaults
    pub struct ExecutionContextBuilder {
        id: Option<String>,
        environment: Option<String>,
        trace_id: Option<String>,
        parent_span_id: Option<String>,
        tags: HashMap<String, String>,
        properties: HashMap<String, String>,
    }

    impl ExecutionContextBuilder {
        /// Create new execution context builder
        pub fn new() -> Self {
            Self {
                id: None,
                environment: None,
                trace_id: None,
                parent_span_id: None,
                tags: HashMap::new(),
                properties: HashMap::new(),
            }
        }

        /// Set execution context ID
        pub fn with_id(mut self, id: String) -> Self {
            self.id = Some(id);
            self
        }

        /// Set environment name
        pub fn with_environment(mut self, environment: String) -> Self {
            self.environment = Some(environment);
            self
        }

        /// Set trace ID for distributed tracing
        pub fn with_trace_id(mut self, trace_id: String) -> Self {
            self.trace_id = Some(trace_id);
            self
        }

        /// Set parent span ID for distributed tracing
        pub fn with_parent_span_id(mut self, parent_span_id: String) -> Self {
            self.parent_span_id = Some(parent_span_id);
            self
        }

        /// Add tag to context
        pub fn with_tag(mut self, key: String, value: String) -> Self {
            self.tags.insert(key, value);
            self
        }

        /// Add property to context
        pub fn with_property(mut self, key: String, value: String) -> Self {
            self.properties.insert(key, value);
            self
        }

        /// Build the execution context
        pub fn build(self) -> ExecutionContext {
            ExecutionContext {
                id: self.id.unwrap_or_else(generate_unique_id),
                environment: self.environment.unwrap_or_else(|| "default".to_string()),
                trace_id: self.trace_id,
                parent_span_id: self.parent_span_id,
                tags: self.tags,
                properties: self.properties,
            }
        }
    }

    /// Builder for TaskMetadata with intelligent defaults
    pub struct TaskMetadataBuilder {
        name: String,
        description: Option<String>,
        tags: Vec<String>,
        estimated_duration: Option<Duration>,
        priority: TaskPriority,
        dependencies: Vec<String>,
    }

    impl TaskMetadataBuilder {
        /// Create new task metadata builder
        pub fn new(name: String) -> Self {
            Self {
                name,
                description: None,
                tags: Vec::new(),
                estimated_duration: None,
                priority: TaskPriority::Normal,
                dependencies: Vec::new(),
            }
        }

        /// Set task description
        pub fn with_description(mut self, description: String) -> Self {
            self.description = Some(description);
            self
        }

        /// Add tag to task
        pub fn with_tag(mut self, tag: String) -> Self {
            self.tags.push(tag);
            self
        }

        /// Set estimated duration
        pub fn with_estimated_duration(mut self, duration: Duration) -> Self {
            self.estimated_duration = Some(duration);
            self
        }

        /// Set task priority
        pub fn with_priority(mut self, priority: TaskPriority) -> Self {
            self.priority = priority;
            self
        }

        /// Add dependency
        pub fn with_dependency(mut self, dependency: String) -> Self {
            self.dependencies.push(dependency);
            self
        }

        /// Build the task metadata
        pub fn build(self) -> TaskMetadata {
            TaskMetadata {
                name: self.name,
                description: self.description,
                tags: self.tags,
                created_at: SystemTime::now(),
                estimated_duration: self.estimated_duration,
                priority: self.priority,
                dependencies: self.dependencies,
            }
        }
    }

    /// Builder for ResourceConstraints with smart defaults
    pub struct ResourceConstraintsBuilder {
        max_cpu_cores: Option<usize>,
        max_memory_bytes: Option<usize>,
        max_io_operations: Option<usize>,
        max_network_bandwidth: Option<usize>,
        max_gpu_memory: Option<usize>,
        max_concurrent_tasks: Option<usize>,
        min_free_memory_percent: Option<f64>,
        max_cpu_usage_percent: Option<f64>,
        timeout: Option<Duration>,
    }

    impl ResourceConstraintsBuilder {
        /// Create new resource constraints builder
        pub fn new() -> Self {
            Self {
                max_cpu_cores: None,
                max_memory_bytes: None,
                max_io_operations: None,
                max_network_bandwidth: None,
                max_gpu_memory: None,
                max_concurrent_tasks: None,
                min_free_memory_percent: None,
                max_cpu_usage_percent: None,
                timeout: None,
            }
        }

        /// Set maximum CPU cores
        pub fn with_max_cpu_cores(mut self, cores: usize) -> Self {
            self.max_cpu_cores = Some(cores);
            self
        }

        /// Set maximum memory bytes
        pub fn with_max_memory_bytes(mut self, bytes: usize) -> Self {
            self.max_memory_bytes = Some(bytes);
            self
        }

        /// Set maximum I/O operations
        pub fn with_max_io_operations(mut self, operations: usize) -> Self {
            self.max_io_operations = Some(operations);
            self
        }

        /// Set maximum network bandwidth
        pub fn with_max_network_bandwidth(mut self, bandwidth: usize) -> Self {
            self.max_network_bandwidth = Some(bandwidth);
            self
        }

        /// Set maximum GPU memory
        pub fn with_max_gpu_memory(mut self, memory: usize) -> Self {
            self.max_gpu_memory = Some(memory);
            self
        }

        /// Set maximum concurrent tasks
        pub fn with_max_concurrent_tasks(mut self, tasks: usize) -> Self {
            self.max_concurrent_tasks = Some(tasks);
            self
        }

        /// Set minimum free memory percentage
        pub fn with_min_free_memory_percent(mut self, percent: f64) -> Self {
            self.min_free_memory_percent = Some(percent);
            self
        }

        /// Set maximum CPU usage percentage
        pub fn with_max_cpu_usage_percent(mut self, percent: f64) -> Self {
            self.max_cpu_usage_percent = Some(percent);
            self
        }

        /// Set timeout duration
        pub fn with_timeout(mut self, timeout: Duration) -> Self {
            self.timeout = Some(timeout);
            self
        }

        /// Build the resource constraints
        pub fn build(self) -> ResourceConstraints {
            ResourceConstraints {
                max_cpu_cores: self.max_cpu_cores.or(Some(num_cpus::get())),
                max_memory_bytes: self.max_memory_bytes.or(Some(8 * 1024 * 1024 * 1024)), // 8GB default
                max_io_operations: self.max_io_operations.or(Some(1000)),
                max_network_bandwidth: self.max_network_bandwidth.or(Some(100 * 1024 * 1024)), // 100MB/s
                max_gpu_memory: self.max_gpu_memory,
                max_concurrent_tasks: self.max_concurrent_tasks.or(Some(10)),
                min_free_memory_percent: self.min_free_memory_percent.or(Some(20.0)),
                max_cpu_usage_percent: self.max_cpu_usage_percent.or(Some(80.0)),
                timeout: self.timeout.or(Some(Duration::from_secs(3600))), // 1 hour default
            }
        }
    }

    impl Default for ExecutionContextBuilder {
        fn default() -> Self {
            Self::new()
        }
    }

    impl Default for ResourceConstraintsBuilder {
        fn default() -> Self {
            Self::new()
        }
    }
}

/// Validation utilities for execution parameters and configurations
pub mod validation {
    use super::*;

    /// Validation result with specific error details
    #[derive(Debug)]
    pub struct ValidationResult {
        pub is_valid: bool,
        pub errors: Vec<String>,
        pub warnings: Vec<String>,
    }

    impl ValidationResult {
        /// Create new successful validation result
        pub fn success() -> Self {
            Self {
                is_valid: true,
                errors: Vec::new(),
                warnings: Vec::new(),
            }
        }

        /// Create new failed validation result
        pub fn failure(error: String) -> Self {
            Self {
                is_valid: false,
                errors: vec![error],
                warnings: Vec::new(),
            }
        }

        /// Add error to validation result
        pub fn add_error(&mut self, error: String) {
            self.errors.push(error);
            self.is_valid = false;
        }

        /// Add warning to validation result
        pub fn add_warning(&mut self, warning: String) {
            self.warnings.push(warning);
        }
    }

    /// Validate execution task configuration
    pub fn validate_execution_task(task: &ExecutionTask) -> ValidationResult {
        let mut result = ValidationResult::success();

        // Validate task ID
        if task.id.is_empty() {
            result.add_error("Task ID cannot be empty".to_string());
        }

        if task.id.len() > 256 {
            result.add_error("Task ID cannot exceed 256 characters".to_string());
        }

        // Validate task name
        if task.metadata.name.is_empty() {
            result.add_error("Task name cannot be empty".to_string());
        }

        // Validate estimated duration
        if let Some(duration) = task.metadata.estimated_duration {
            if duration.as_secs() > 86400 {  // 24 hours
                result.add_warning("Task estimated duration exceeds 24 hours".to_string());
            }
        }

        // Validate resource requirements
        if let Some(cpu_cores) = task.requirements.cpu_cores {
            let max_cpu_cores = num_cpus::get();
            if cpu_cores > max_cpu_cores {
                result.add_error(format!(
                    "Requested CPU cores ({}) exceeds available cores ({})",
                    cpu_cores, max_cpu_cores
                ));
            }
        }

        if let Some(memory_bytes) = task.requirements.memory_bytes {
            if memory_bytes > 128 * 1024 * 1024 * 1024 { // 128GB
                result.add_warning("Task requires more than 128GB of memory".to_string());
            }
        }

        // Validate constraints
        if let Some(max_execution_time) = task.constraints.max_execution_time {
            if max_execution_time.as_secs() == 0 {
                result.add_error("Maximum execution time cannot be zero".to_string());
            }
        }

        if let Some(deadline) = task.constraints.deadline {
            if deadline < SystemTime::now() {
                result.add_error("Task deadline is in the past".to_string());
            }
        }

        result
    }

    /// Validate resource constraints configuration
    pub fn validate_resource_constraints(constraints: &ResourceConstraints) -> ValidationResult {
        let mut result = ValidationResult::success();

        if let Some(cpu_cores) = constraints.max_cpu_cores {
            let available_cores = num_cpus::get();
            if cpu_cores > available_cores {
                result.add_error(format!(
                    "Max CPU cores ({}) exceeds available cores ({})",
                    cpu_cores, available_cores
                ));
            }
        }

        if let Some(memory_bytes) = constraints.max_memory_bytes {
            if memory_bytes < 1024 * 1024 { // 1MB minimum
                result.add_error("Maximum memory bytes must be at least 1MB".to_string());
            }
        }

        if let Some(concurrent_tasks) = constraints.max_concurrent_tasks {
            if concurrent_tasks == 0 {
                result.add_error("Maximum concurrent tasks cannot be zero".to_string());
            }
            if concurrent_tasks > 1000 {
                result.add_warning("Maximum concurrent tasks exceeds 1000, may impact performance".to_string());
            }
        }

        if let Some(free_memory_percent) = constraints.min_free_memory_percent {
            if free_memory_percent < 0.0 || free_memory_percent > 100.0 {
                result.add_error("Minimum free memory percent must be between 0 and 100".to_string());
            }
        }

        if let Some(cpu_usage_percent) = constraints.max_cpu_usage_percent {
            if cpu_usage_percent < 0.0 || cpu_usage_percent > 100.0 {
                result.add_error("Maximum CPU usage percent must be between 0 and 100".to_string());
            }
        }

        result
    }

    /// Validate scheduler configuration
    pub fn validate_scheduler_config(config: &SchedulerConfig) -> ValidationResult {
        let mut result = ValidationResult::success();

        if config.max_queue_size == 0 {
            result.add_error("Maximum queue size cannot be zero".to_string());
        }

        if config.max_queue_size > 100000 {
            result.add_warning("Maximum queue size exceeds 100,000, may impact memory usage".to_string());
        }

        if config.worker_threads == 0 {
            result.add_error("Worker thread count cannot be zero".to_string());
        }

        if config.worker_threads > num_cpus::get() * 2 {
            result.add_warning("Worker thread count significantly exceeds CPU cores".to_string());
        }

        if config.heartbeat_interval.as_secs() == 0 {
            result.add_error("Heartbeat interval cannot be zero".to_string());
        }

        if config.task_timeout.as_secs() < 1 {
            result.add_error("Task timeout must be at least 1 second".to_string());
        }

        result
    }
}

/// Conversion utilities for transforming between different types
pub mod conversion {
    use super::*;

    /// Convert task priority to numeric weight for sorting
    pub fn priority_to_weight(priority: TaskPriority) -> i32 {
        match priority {
            TaskPriority::Critical => 1000,
            TaskPriority::High => 100,
            TaskPriority::Normal => 10,
            TaskPriority::Low => 1,
        }
    }

    /// Convert numeric weight back to task priority
    pub fn weight_to_priority(weight: i32) -> TaskPriority {
        match weight {
            1000.. => TaskPriority::Critical,
            100..=999 => TaskPriority::High,
            10..=99 => TaskPriority::Normal,
            _ => TaskPriority::Low,
        }
    }

    /// Convert duration to human-readable string
    pub fn duration_to_human_readable(duration: Duration) -> String {
        let total_seconds = duration.as_secs();
        let hours = total_seconds / 3600;
        let minutes = (total_seconds % 3600) / 60;
        let seconds = total_seconds % 60;
        let millis = duration.subsec_millis();

        if hours > 0 {
            format!("{}h {}m {}s", hours, minutes, seconds)
        } else if minutes > 0 {
            format!("{}m {}s", minutes, seconds)
        } else if seconds > 0 {
            format!("{}s {}ms", seconds, millis)
        } else {
            format!("{}ms", millis)
        }
    }

    /// Convert bytes to human-readable string
    pub fn bytes_to_human_readable(bytes: usize) -> String {
        const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
        let mut size = bytes as f64;
        let mut unit_index = 0;

        while size >= 1024.0 && unit_index < UNITS.len() - 1 {
            size /= 1024.0;
            unit_index += 1;
        }

        if unit_index == 0 {
            format!("{} {}", bytes, UNITS[unit_index])
        } else {
            format!("{:.2} {}", size, UNITS[unit_index])
        }
    }

    /// Convert system time to ISO 8601 string
    pub fn system_time_to_iso8601(time: SystemTime) -> String {
        match time.duration_since(SystemTime::UNIX_EPOCH) {
            Ok(duration) => {
                let secs = duration.as_secs();
                let nanos = duration.subsec_nanos();
                format!("{}T{:02}:{:02}:{:02}.{:06}Z",
                    "1970-01-01",  // Simplified date format
                    (secs / 3600) % 24,
                    (secs / 60) % 60,
                    secs % 60,
                    nanos / 1000
                )
            },
            Err(_) => "1970-01-01T00:00:00.000000Z".to_string(),
        }
    }

    /// Convert task status to progress percentage
    pub fn status_to_progress(status: &TaskStatus) -> f64 {
        match status {
            TaskStatus::Pending => 0.0,
            TaskStatus::Running => 50.0,
            TaskStatus::Completed => 100.0,
            TaskStatus::Failed => 0.0,
            TaskStatus::Cancelled => 0.0,
        }
    }
}

/// Metrics calculation and aggregation utilities
pub mod metrics {
    use super::*;

    /// Calculate average from a collection of values
    pub fn calculate_average(values: &[f64]) -> f64 {
        if values.is_empty() {
            0.0
        } else {
            values.iter().sum::<f64>() / values.len() as f64
        }
    }

    /// Calculate median from a collection of values
    pub fn calculate_median(values: &mut [f64]) -> f64 {
        if values.is_empty() {
            return 0.0;
        }

        values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let len = values.len();

        if len % 2 == 0 {
            (values[len / 2 - 1] + values[len / 2]) / 2.0
        } else {
            values[len / 2]
        }
    }

    /// Calculate 95th percentile from a collection of values
    pub fn calculate_p95(values: &mut [f64]) -> f64 {
        if values.is_empty() {
            return 0.0;
        }

        values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let index = ((values.len() - 1) as f64 * 0.95) as usize;
        values[index]
    }

    /// Calculate standard deviation from a collection of values
    pub fn calculate_standard_deviation(values: &[f64]) -> f64 {
        if values.len() < 2 {
            return 0.0;
        }

        let mean = calculate_average(values);
        let variance = values.iter()
            .map(|value| {
                let diff = mean - value;
                diff * diff
            })
            .sum::<f64>() / (values.len() - 1) as f64;

        variance.sqrt()
    }

    /// Calculate throughput (tasks per second)
    pub fn calculate_throughput(completed_tasks: usize, duration: Duration) -> f64 {
        if duration.as_secs_f64() == 0.0 {
            0.0
        } else {
            completed_tasks as f64 / duration.as_secs_f64()
        }
    }

    /// Calculate error rate percentage
    pub fn calculate_error_rate(failed_tasks: usize, total_tasks: usize) -> f64 {
        if total_tasks == 0 {
            0.0
        } else {
            (failed_tasks as f64 / total_tasks as f64) * 100.0
        }
    }

    /// Calculate resource utilization percentage
    pub fn calculate_resource_utilization(used: usize, total: usize) -> f64 {
        if total == 0 {
            0.0
        } else {
            (used as f64 / total as f64) * 100.0
        }
    }

    /// Aggregate task execution metrics
    pub fn aggregate_task_metrics(metrics: &[TaskExecutionMetrics]) -> AggregatedMetrics {
        if metrics.is_empty() {
            return AggregatedMetrics::default();
        }

        let mut durations: Vec<f64> = metrics.iter()
            .filter_map(|m| m.duration.map(|d| d.as_secs_f64()))
            .collect();

        let mut cpu_usages: Vec<f64> = metrics.iter()
            .map(|m| m.resource_usage.cpu_percent)
            .collect();

        let memory_usages: Vec<usize> = metrics.iter()
            .map(|m| m.resource_usage.memory_bytes)
            .collect();

        let throughputs: Vec<f64> = metrics.iter()
            .map(|m| m.performance.throughput)
            .collect();

        AggregatedMetrics {
            count: metrics.len(),
            avg_duration: calculate_average(&durations),
            median_duration: calculate_median(&mut durations),
            p95_duration: calculate_p95(&mut durations),
            avg_cpu_usage: calculate_average(&cpu_usages),
            max_memory_usage: memory_usages.iter().copied().max().unwrap_or(0),
            total_memory_usage: memory_usages.iter().sum(),
            avg_throughput: calculate_average(&throughputs),
            max_throughput: throughputs.iter().copied().fold(0.0, f64::max),
        }
    }

    /// Aggregated metrics summary
    #[derive(Debug, Clone)]
    pub struct AggregatedMetrics {
        pub count: usize,
        pub avg_duration: f64,
        pub median_duration: f64,
        pub p95_duration: f64,
        pub avg_cpu_usage: f64,
        pub max_memory_usage: usize,
        pub total_memory_usage: usize,
        pub avg_throughput: f64,
        pub max_throughput: f64,
    }

    impl Default for AggregatedMetrics {
        fn default() -> Self {
            Self {
                count: 0,
                avg_duration: 0.0,
                median_duration: 0.0,
                p95_duration: 0.0,
                avg_cpu_usage: 0.0,
                max_memory_usage: 0,
                total_memory_usage: 0,
                avg_throughput: 0.0,
                max_throughput: 0.0,
            }
        }
    }
}

/// Time and duration utility functions
pub mod time_utils {
    use super::*;

    /// Get current timestamp as milliseconds since UNIX epoch
    pub fn current_timestamp_millis() -> u64 {
        SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64
    }

    /// Get current timestamp as seconds since UNIX epoch
    pub fn current_timestamp_secs() -> u64 {
        SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    }

    /// Create duration from milliseconds
    pub fn duration_from_millis(millis: u64) -> Duration {
        Duration::from_millis(millis)
    }

    /// Create duration from seconds
    pub fn duration_from_secs(secs: u64) -> Duration {
        Duration::from_secs(secs)
    }

    /// Add duration to system time safely
    pub fn add_duration_to_time(time: SystemTime, duration: Duration) -> SystemTime {
        time.checked_add(duration).unwrap_or(time)
    }

    /// Subtract duration from system time safely
    pub fn subtract_duration_from_time(time: SystemTime, duration: Duration) -> SystemTime {
        time.checked_sub(duration).unwrap_or(time)
    }

    /// Check if a time is expired relative to current time
    pub fn is_expired(time: SystemTime, timeout: Duration) -> bool {
        let now = SystemTime::now();
        if let Ok(elapsed) = now.duration_since(time) {
            elapsed > timeout
        } else {
            false
        }
    }

    /// Calculate time remaining until deadline
    pub fn time_until_deadline(deadline: SystemTime) -> Option<Duration> {
        let now = SystemTime::now();
        if deadline > now {
            Some(deadline.duration_since(now).unwrap_or_default())
        } else {
            None
        }
    }

    /// Format duration as HH:MM:SS
    pub fn format_duration_hms(duration: Duration) -> String {
        let total_seconds = duration.as_secs();
        let hours = total_seconds / 3600;
        let minutes = (total_seconds % 3600) / 60;
        let seconds = total_seconds % 60;
        format!("{:02}:{:02}:{:02}", hours, minutes, seconds)
    }

    /// Sleep for specified duration (async-friendly)
    pub async fn sleep(duration: Duration) {
        tokio::time::sleep(duration).await;
    }

    /// Create timeout future
    pub fn timeout<F>(duration: Duration, future: F) -> tokio::time::Timeout<F>
    where
        F: std::future::Future,
    {
        tokio::time::timeout(duration, future)
    }
}

/// Resource calculation and estimation utilities
pub mod resource_utils {
    use super::*;

    /// Estimate memory usage based on task type and input size
    pub fn estimate_memory_usage(task_type: &TaskType, input_size_bytes: usize) -> usize {
        match task_type {
            TaskType::Fit => input_size_bytes * 3,      // Model training typically requires 3x input size
            TaskType::Transform => input_size_bytes * 2, // Transformation requires 2x for input + output
            TaskType::Predict => input_size_bytes / 2,   // Prediction is usually lightweight
            TaskType::Evaluate => input_size_bytes,      // Evaluation similar to input size
            TaskType::Preprocess => input_size_bytes * 2, // Preprocessing requires buffer space
            TaskType::PostProcess => input_size_bytes,    // Post-processing is usually in-place
            TaskType::Custom => input_size_bytes,        // Conservative estimate for custom tasks
        }
    }

    /// Estimate CPU cores needed based on task type and complexity
    pub fn estimate_cpu_cores(task_type: &TaskType, complexity_factor: f64) -> usize {
        let base_cores = match task_type {
            TaskType::Fit => 4,          // Model training is CPU-intensive
            TaskType::Transform => 2,    // Transformation is moderately intensive
            TaskType::Predict => 1,      // Prediction is usually single-threaded
            TaskType::Evaluate => 2,     // Evaluation benefits from parallel processing
            TaskType::Preprocess => 2,   // Preprocessing can be parallelized
            TaskType::PostProcess => 1,  // Post-processing is usually sequential
            TaskType::Custom => 2,       // Conservative estimate for custom tasks
        };

        let estimated = (base_cores as f64 * complexity_factor).ceil() as usize;
        estimated.max(1).min(num_cpus::get())
    }

    /// Estimate execution time based on historical data and task characteristics
    pub fn estimate_execution_time(
        task_type: &TaskType,
        input_size_bytes: usize,
        historical_avg_ms: Option<u64>,
    ) -> Duration {
        // Use historical average if available
        if let Some(avg_ms) = historical_avg_ms {
            return Duration::from_millis((avg_ms as f64 * 1.2) as u64); // Add 20% buffer
        }

        // Estimate based on task type and input size
        let base_time_ms = match task_type {
            TaskType::Fit => 5000,       // 5 seconds base for model training
            TaskType::Transform => 1000, // 1 second base for transformation
            TaskType::Predict => 100,    // 100ms base for prediction
            TaskType::Evaluate => 2000,  // 2 seconds base for evaluation
            TaskType::Preprocess => 1500, // 1.5 seconds base for preprocessing
            TaskType::PostProcess => 500,  // 500ms base for post-processing
            TaskType::Custom => 1000,     // 1 second base for custom tasks
        };

        // Scale by input size (assuming linear relationship)
        let size_mb = input_size_bytes / (1024 * 1024);
        let scaling_factor = 1.0 + (size_mb as f64 / 100.0); // +1% per MB

        Duration::from_millis((base_time_ms as f64 * scaling_factor) as u64)
    }

    /// Calculate optimal batch size based on available resources
    pub fn calculate_optimal_batch_size(
        available_memory_bytes: usize,
        item_memory_bytes: usize,
        max_concurrent_tasks: usize,
    ) -> usize {
        let memory_based_batch_size = if item_memory_bytes > 0 {
            available_memory_bytes / item_memory_bytes
        } else {
            max_concurrent_tasks
        };

        memory_based_batch_size.min(max_concurrent_tasks).max(1)
    }

    /// Check if resources are sufficient for task execution
    pub fn check_resource_availability(
        task: &ExecutionTask,
        available: &AvailableResources,
    ) -> bool {
        // Check CPU cores
        if let Some(required_cores) = task.requirements.cpu_cores {
            if required_cores > available.cpu_cores {
                return false;
            }
        }

        // Check memory
        if let Some(required_memory) = task.requirements.memory_bytes {
            if required_memory > available.memory_bytes {
                return false;
            }
        }

        // Check GPU memory
        if let Some(required_gpu_memory) = task.requirements.gpu_memory {
            if required_gpu_memory > available.gpu_memory.unwrap_or(0) {
                return false;
            }
        }

        // Check network bandwidth
        if let Some(required_bandwidth) = task.requirements.network_bandwidth {
            if required_bandwidth > available.network_bandwidth.unwrap_or(0) {
                return false;
            }
        }

        // Check I/O bandwidth
        if let Some(required_io) = task.requirements.io_bandwidth {
            if required_io > available.io_bandwidth.unwrap_or(0) {
                return false;
            }
        }

        true
    }
}

/// Error handling and recovery utilities
pub mod error_utils {
    use super::*;

    /// Create task error with recovery suggestions
    pub fn create_task_error(
        error_type: &str,
        message: String,
        suggestions: Vec<String>,
    ) -> TaskError {
        TaskError {
            error_type: error_type.to_string(),
            message,
            code: None,
            stack_trace: None,
            recovery_suggestions: suggestions,
        }
    }

    /// Create resource exhaustion error
    pub fn create_resource_exhaustion_error(resource: &str, required: usize, available: usize) -> TaskError {
        create_task_error(
            "ResourceExhaustionError",
            format!("Insufficient {}: required {}, available {}", resource, required, available),
            vec![
                format!("Free up {} resources", resource),
                "Reduce task resource requirements".to_string(),
                "Wait for resources to become available".to_string(),
            ],
        )
    }

    /// Create timeout error
    pub fn create_timeout_error(duration: Duration, task_id: &str) -> TaskError {
        create_task_error(
            "TimeoutError",
            format!("Task {} exceeded maximum execution time of {:?}", task_id, duration),
            vec![
                "Increase task timeout duration".to_string(),
                "Optimize task implementation for better performance".to_string(),
                "Break task into smaller chunks".to_string(),
            ],
        )
    }

    /// Create dependency error
    pub fn create_dependency_error(task_id: &str, missing_deps: Vec<String>) -> TaskError {
        create_task_error(
            "DependencyError",
            format!("Task {} has unresolved dependencies: {:?}", task_id, missing_deps),
            vec![
                "Ensure all dependencies are completed".to_string(),
                "Check dependency task IDs for correctness".to_string(),
                "Review task scheduling order".to_string(),
            ],
        )
    }

    /// Get error severity level
    pub fn get_error_severity(error: &TaskError) -> ErrorSeverity {
        match error.error_type.as_str() {
            "ResourceExhaustionError" => ErrorSeverity::High,
            "TimeoutError" => ErrorSeverity::Medium,
            "DependencyError" => ErrorSeverity::Medium,
            "ValidationError" => ErrorSeverity::High,
            "ConfigurationError" => ErrorSeverity::High,
            "NetworkError" => ErrorSeverity::Medium,
            "IOError" => ErrorSeverity::Medium,
            _ => ErrorSeverity::Low,
        }
    }

    /// Error severity levels
    #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
    pub enum ErrorSeverity {
        Low = 1,
        Medium = 2,
        High = 3,
        Critical = 4,
    }

    /// Determine if error is retryable
    pub fn is_retryable_error(error: &TaskError) -> bool {
        match error.error_type.as_str() {
            "NetworkError" | "IOError" | "TemporaryResourceError" => true,
            "ResourceExhaustionError" | "DependencyError" | "ValidationError" => false,
            _ => false,
        }
    }

    /// Calculate backoff delay for retry attempts
    pub fn calculate_backoff_delay(attempt: u32, strategy: &BackoffStrategy) -> Duration {
        match strategy {
            BackoffStrategy::Fixed { delay } => *delay,
            BackoffStrategy::Linear { base_delay, increment } => {
                *base_delay + Duration::from_millis(increment.as_millis() as u64 * attempt as u64)
            },
            BackoffStrategy::Exponential { base_delay, multiplier, max_delay } => {
                let delay_ms = base_delay.as_millis() as f64 * multiplier.powi(attempt as i32);
                let capped_delay = Duration::from_millis(delay_ms as u64).min(*max_delay);
                capped_delay
            },
        }
    }
}

/// Test utilities and mock data creation
pub mod test_utils {
    use super::*;

    /// Create a simple test execution task
    pub fn create_test_task(id: &str, task_type: TaskType) -> ExecutionTask {
        ExecutionTask {
            id: id.to_string(),
            task_type,
            task_fn: Box::new(|| Ok(())),
            metadata: TaskMetadata {
                name: format!("Test Task {}", id),
                description: Some("A test task for validation".to_string()),
                tags: vec!["test".to_string()],
                created_at: SystemTime::now(),
                estimated_duration: Some(Duration::from_secs(5)),
                priority: TaskPriority::Normal,
                dependencies: vec![],
            },
            requirements: TaskRequirements {
                cpu_cores: Some(2),
                memory_bytes: Some(1024 * 1024),
                io_bandwidth: None,
                gpu_memory: None,
                network_bandwidth: None,
            },
            constraints: TaskConstraints {
                max_execution_time: Some(Duration::from_secs(60)),
                deadline: None,
                location: None,
                affinity: None,
            },
        }
    }

    /// Create test task result
    pub fn create_test_task_result(task_id: &str, status: TaskStatus) -> TaskResult {
        TaskResult {
            task_id: task_id.to_string(),
            status,
            data: None,
            metrics: TaskExecutionMetrics {
                start_time: SystemTime::now(),
                end_time: Some(SystemTime::now() + Duration::from_secs(5)),
                duration: Some(Duration::from_secs(5)),
                resource_usage: TaskResourceUsage {
                    cpu_percent: 50.0,
                    memory_bytes: 1024 * 1024,
                    io_operations: 100,
                    network_bytes: 0,
                },
                performance: TaskPerformanceMetrics {
                    throughput: 100.0,
                    latency: 50.0,
                    cache_hit_rate: 0.95,
                    error_rate: 0.0,
                },
            },
            error: None,
        }
    }

    /// Create mock available resources
    pub fn create_mock_available_resources() -> AvailableResources {
        AvailableResources {
            cpu_cores: num_cpus::get(),
            memory_bytes: 8 * 1024 * 1024 * 1024, // 8GB
            io_bandwidth: Some(1000 * 1024 * 1024), // 1GB/s
            gpu_memory: Some(4 * 1024 * 1024 * 1024), // 4GB
            network_bandwidth: Some(100 * 1024 * 1024), // 100MB/s
        }
    }

    /// Create test scheduler config
    pub fn create_test_scheduler_config() -> SchedulerConfig {
        SchedulerConfig {
            max_queue_size: 1000,
            worker_threads: 4,
            enable_priority_scheduling: true,
            enable_load_balancing: true,
            enable_fault_tolerance: true,
            heartbeat_interval: Duration::from_secs(30),
            task_timeout: Duration::from_secs(300),
        }
    }

    /// Create test resource constraints
    pub fn create_test_resource_constraints() -> ResourceConstraints {
        ResourceConstraints {
            max_cpu_cores: Some(8),
            max_memory_bytes: Some(16 * 1024 * 1024 * 1024), // 16GB
            max_io_operations: Some(10000),
            max_network_bandwidth: Some(1000 * 1024 * 1024), // 1GB/s
            max_gpu_memory: Some(8 * 1024 * 1024 * 1024), // 8GB
            max_concurrent_tasks: Some(20),
            min_free_memory_percent: Some(10.0),
            max_cpu_usage_percent: Some(85.0),
            timeout: Some(Duration::from_secs(3600)), // 1 hour
        }
    }

    /// Generate random task ID
    pub fn generate_random_task_id() -> String {
        format!("task_{}", current_timestamp_millis())
    }

    /// Create batch of test tasks
    pub fn create_test_task_batch(count: usize, task_type: TaskType) -> Vec<ExecutionTask> {
        (0..count)
            .map(|i| create_test_task(&format!("task_{}", i), task_type.clone()))
            .collect()
    }
}

/// Common utility functions
/// Generate unique ID for execution contexts, tasks, etc.
pub fn generate_unique_id() -> String {
    static COUNTER: AtomicU64 = AtomicU64::new(1);
    let timestamp = time_utils::current_timestamp_millis();
    let counter = COUNTER.fetch_add(1, Ordering::SeqCst);
    format!("{}_{:06}", timestamp, counter)
}

/// Generate task handle from task ID
pub fn generate_task_handle(task_id: &str) -> TaskHandle {
    TaskHandle {
        id: task_id.to_string(),
        created_at: SystemTime::now(),
    }
}

/// Check if system has sufficient resources for a batch of tasks
pub fn check_batch_resources(
    tasks: &[ExecutionTask],
    available: &AvailableResources,
) -> SklResult<bool> {
    let total_cpu_cores: usize = tasks.iter()
        .filter_map(|t| t.requirements.cpu_cores)
        .sum();

    let total_memory_bytes: usize = tasks.iter()
        .filter_map(|t| t.requirements.memory_bytes)
        .sum();

    let total_gpu_memory: usize = tasks.iter()
        .filter_map(|t| t.requirements.gpu_memory)
        .sum();

    let sufficient = total_cpu_cores <= available.cpu_cores
        && total_memory_bytes <= available.memory_bytes
        && total_gpu_memory <= available.gpu_memory.unwrap_or(0);

    Ok(sufficient)
}

/// Calculate system load factor (0.0 = no load, 1.0 = fully loaded)
pub fn calculate_system_load_factor(
    active_tasks: usize,
    max_concurrent_tasks: usize,
    cpu_usage_percent: f64,
    memory_usage_percent: f64,
) -> f64 {
    let task_load = active_tasks as f64 / max_concurrent_tasks as f64;
    let cpu_load = cpu_usage_percent / 100.0;
    let memory_load = memory_usage_percent / 100.0;

    // Take the maximum of the three load factors
    task_load.max(cpu_load).max(memory_load).min(1.0)
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::*;

    #[test]
    fn test_generate_unique_id() {
        let id1 = generate_unique_id();
        let id2 = generate_unique_id();
        assert_ne!(id1, id2);
        assert!(!id1.is_empty());
        assert!(!id2.is_empty());
    }

    #[test]
    fn test_execution_context_builder() {
        let context = config_builders::ExecutionContextBuilder::new()
            .with_environment("test".to_string())
            .with_tag("type".to_string(), "unit_test".to_string())
            .build();

        assert_eq!(context.environment, "test");
        assert_eq!(context.tags.get("type"), Some(&"unit_test".to_string()));
    }

    #[test]
    fn test_task_validation() {
        let task = create_test_task("test_task", TaskType::Transform);
        let result = validation::validate_execution_task(&task);
        assert!(result.is_valid);
        assert!(result.errors.is_empty());
    }

    #[test]
    fn test_priority_conversion() {
        assert_eq!(conversion::priority_to_weight(TaskPriority::Critical), 1000);
        assert_eq!(conversion::priority_to_weight(TaskPriority::Normal), 10);

        assert_eq!(conversion::weight_to_priority(1000), TaskPriority::Critical);
        assert_eq!(conversion::weight_to_priority(50), TaskPriority::Normal);
    }

    #[test]
    fn test_duration_formatting() {
        let duration = Duration::from_secs(3661); // 1h 1m 1s
        let formatted = conversion::duration_to_human_readable(duration);
        assert_eq!(formatted, "1h 1m 1s");
    }

    #[test]
    fn test_bytes_formatting() {
        assert_eq!(conversion::bytes_to_human_readable(1024), "1.00 KB");
        assert_eq!(conversion::bytes_to_human_readable(1048576), "1.00 MB");
    }

    #[test]
    fn test_metrics_calculation() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(metrics::calculate_average(&values), 3.0);

        let mut values_mut = values.clone();
        assert_eq!(metrics::calculate_median(&mut values_mut), 3.0);
    }

    #[test]
    fn test_resource_estimation() {
        let memory = resource_utils::estimate_memory_usage(&TaskType::Fit, 1024);
        assert_eq!(memory, 3072); // 3x input size

        let cores = resource_utils::estimate_cpu_cores(&TaskType::Transform, 1.5);
        assert_eq!(cores, 3); // 2 base cores * 1.5 complexity factor
    }

    #[test]
    fn test_error_creation() {
        let error = error_utils::create_timeout_error(
            Duration::from_secs(300),
            "test_task"
        );

        assert_eq!(error.error_type, "TimeoutError");
        assert!(error.message.contains("test_task"));
        assert!(!error.recovery_suggestions.is_empty());
    }

    #[test]
    fn test_system_load_calculation() {
        let load = calculate_system_load_factor(5, 10, 50.0, 30.0);
        assert_eq!(load, 0.5); // Max of 0.5 task load, 0.5 CPU load, 0.3 memory load
    }
}