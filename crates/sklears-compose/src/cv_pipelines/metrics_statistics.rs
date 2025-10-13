//! Metrics and statistics collection for computer vision pipelines
//!
//! This module provides comprehensive metrics collection, statistical analysis,
//! and performance tracking capabilities for computer vision processing pipelines.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, SystemTime};

/// Computer vision metrics collector
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CVMetrics {
    /// Processing statistics
    pub processing_stats: ProcessingStatistics,
    /// Quality metrics
    pub quality_metrics: QualityMetrics,
    /// Performance metrics
    pub performance_metrics: PerformanceMetrics,
    /// Resource utilization metrics
    pub resource_utilization: ResourceUtilization,
    /// Error tracking
    pub error_tracking: ErrorTracking,
    /// Historical data
    pub historical_data: HistoricalMetrics,
}

impl CVMetrics {
    /// Create new metrics collector
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Record processing of a single image
    pub fn record_image_processed(&mut self, processing_time: Duration, quality_score: f64) {
        self.processing_stats.record_image(processing_time);
        self.quality_metrics.record_quality(quality_score);
        self.performance_metrics.update_throughput();
    }

    /// Record an error occurrence
    pub fn record_error(&mut self, error_type: ErrorType, error_message: String) {
        self.error_tracking.record_error(error_type, error_message);
    }

    /// Record resource usage
    pub fn record_resource_usage(&mut self, cpu: f64, memory: u64, gpu: Option<f64>) {
        self.resource_utilization.update_usage(cpu, memory, gpu);
    }

    /// Get current processing rate (images per second)
    #[must_use]
    pub fn current_processing_rate(&self) -> f64 {
        self.processing_stats.throughput
    }

    /// Get average quality score
    #[must_use]
    pub fn average_quality(&self) -> f64 {
        self.quality_metrics.avg_confidence
    }

    /// Get error rate
    #[must_use]
    pub fn error_rate(&self) -> f64 {
        self.error_tracking.error_rate
    }

    /// Reset all metrics
    pub fn reset(&mut self) {
        self.processing_stats.reset();
        self.quality_metrics.reset();
        self.performance_metrics.reset();
        self.resource_utilization.reset();
        self.error_tracking.reset();
    }

    /// Generate summary report
    #[must_use]
    pub fn generate_summary(&self) -> MetricsSummary {
        MetricsSummary {
            total_images_processed: self.processing_stats.total_images,
            average_processing_time: self.processing_stats.avg_time_per_image,
            current_throughput: self.processing_stats.throughput,
            average_quality: self.quality_metrics.avg_confidence,
            error_rate: self.error_tracking.error_rate,
            cpu_utilization: self.resource_utilization.cpu_utilization,
            memory_utilization: self.resource_utilization.memory_usage_mb,
            gpu_utilization: self.resource_utilization.gpu_utilization,
            uptime: self.processing_stats.total_time,
        }
    }
}

/// Processing statistics for computer vision pipelines
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingStatistics {
    /// Total number of images processed
    pub total_images: u64,
    /// Total processing time
    pub total_time: Duration,
    /// Average processing time per image
    pub avg_time_per_image: Duration,
    /// Minimum processing time recorded
    pub min_time: Duration,
    /// Maximum processing time recorded
    pub max_time: Duration,
    /// Number of successful predictions
    pub successful_predictions: u64,
    /// Number of failed predictions
    pub failed_predictions: u64,
    /// Current processing throughput (images/second)
    pub throughput: f64,
    /// Processing start time
    pub start_time: SystemTime,
    /// Recent processing times for rolling statistics
    pub recent_times: VecDeque<Duration>,
}

impl Default for ProcessingStatistics {
    fn default() -> Self {
        Self {
            total_images: 0,
            total_time: Duration::from_secs(0),
            avg_time_per_image: Duration::from_secs(0),
            min_time: Duration::from_secs(u64::MAX),
            max_time: Duration::from_secs(0),
            successful_predictions: 0,
            failed_predictions: 0,
            throughput: 0.0,
            start_time: SystemTime::now(),
            recent_times: VecDeque::with_capacity(100),
        }
    }
}

impl ProcessingStatistics {
    /// Record processing of a single image
    pub fn record_image(&mut self, processing_time: Duration) {
        self.total_images += 1;
        self.total_time += processing_time;
        self.successful_predictions += 1;

        // Update min/max times
        if processing_time < self.min_time {
            self.min_time = processing_time;
        }
        if processing_time > self.max_time {
            self.max_time = processing_time;
        }

        // Update rolling average
        self.recent_times.push_back(processing_time);
        if self.recent_times.len() > 100 {
            self.recent_times.pop_front();
        }

        // Calculate average
        if self.total_images > 0 {
            self.avg_time_per_image = self.total_time / self.total_images as u32;
        }

        // Calculate throughput
        self.update_throughput();
    }

    /// Record a failed prediction
    pub fn record_failure(&mut self) {
        self.failed_predictions += 1;
    }

    /// Update throughput calculation
    pub fn update_throughput(&mut self) {
        let elapsed = SystemTime::now()
            .duration_since(self.start_time)
            .unwrap_or(Duration::from_secs(1));

        if elapsed.as_secs() > 0 {
            self.throughput = self.total_images as f64 / elapsed.as_secs_f64();
        }
    }

    /// Get success rate
    #[must_use]
    pub fn success_rate(&self) -> f64 {
        let total = self.successful_predictions + self.failed_predictions;
        if total > 0 {
            self.successful_predictions as f64 / total as f64
        } else {
            1.0
        }
    }

    /// Get recent average processing time
    #[must_use]
    pub fn recent_avg_time(&self) -> Duration {
        if self.recent_times.is_empty() {
            return Duration::from_secs(0);
        }

        let total: Duration = self.recent_times.iter().sum();
        total / self.recent_times.len() as u32
    }

    /// Reset statistics
    pub fn reset(&mut self) {
        self.total_images = 0;
        self.total_time = Duration::from_secs(0);
        self.avg_time_per_image = Duration::from_secs(0);
        self.min_time = Duration::from_secs(u64::MAX);
        self.max_time = Duration::from_secs(0);
        self.successful_predictions = 0;
        self.failed_predictions = 0;
        self.throughput = 0.0;
        self.start_time = SystemTime::now();
        self.recent_times.clear();
    }
}

/// Quality metrics for computer vision processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMetrics {
    /// Average confidence score
    pub avg_confidence: f64,
    /// Minimum confidence recorded
    pub min_confidence: f64,
    /// Maximum confidence recorded
    pub max_confidence: f64,
    /// Prediction accuracy (if ground truth available)
    pub accuracy: Option<f64>,
    /// Quality consistency score
    pub consistency: f64,
    /// Error rate
    pub error_rate: f64,
    /// Quality distribution histogram
    pub quality_distribution: HashMap<String, u64>,
    /// Recent confidence scores for rolling statistics
    pub recent_confidences: VecDeque<f64>,
}

impl Default for QualityMetrics {
    fn default() -> Self {
        Self {
            avg_confidence: 0.0,
            min_confidence: 1.0,
            max_confidence: 0.0,
            accuracy: None,
            consistency: 0.0,
            error_rate: 0.0,
            quality_distribution: HashMap::new(),
            recent_confidences: VecDeque::with_capacity(100),
        }
    }
}

impl QualityMetrics {
    /// Record a quality measurement
    pub fn record_quality(&mut self, confidence: f64) {
        // Update min/max
        if confidence < self.min_confidence {
            self.min_confidence = confidence;
        }
        if confidence > self.max_confidence {
            self.max_confidence = confidence;
        }

        // Update rolling average
        self.recent_confidences.push_back(confidence);
        if self.recent_confidences.len() > 100 {
            self.recent_confidences.pop_front();
        }

        // Calculate average confidence
        let sum: f64 = self.recent_confidences.iter().sum();
        self.avg_confidence = sum / self.recent_confidences.len() as f64;

        // Update quality distribution
        let bucket = self.get_quality_bucket(confidence);
        *self.quality_distribution.entry(bucket).or_insert(0) += 1;

        // Update consistency
        self.update_consistency();
    }

    /// Update accuracy with ground truth
    pub fn update_accuracy(&mut self, predicted: &str, actual: &str) {
        // Simple accuracy calculation - in practice this would be more sophisticated
        let is_correct = predicted == actual;

        // Update rolling accuracy
        match self.accuracy {
            Some(current_acc) => {
                // Simple exponential moving average
                let alpha = 0.1;
                self.accuracy = Some(
                    current_acc * (1.0 - alpha) + (if is_correct { 1.0 } else { 0.0 }) * alpha,
                );
            }
            None => {
                self.accuracy = Some(if is_correct { 1.0 } else { 0.0 });
            }
        }
    }

    /// Get quality bucket for distribution tracking
    fn get_quality_bucket(&self, confidence: f64) -> String {
        match confidence {
            x if x < 0.2 => "very_low".to_string(),
            x if x < 0.4 => "low".to_string(),
            x if x < 0.6 => "medium".to_string(),
            x if x < 0.8 => "high".to_string(),
            _ => "very_high".to_string(),
        }
    }

    /// Update consistency metric
    fn update_consistency(&mut self) {
        if self.recent_confidences.len() < 2 {
            return;
        }

        // Calculate standard deviation of recent confidences
        let mean = self.avg_confidence;
        let variance: f64 = self
            .recent_confidences
            .iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>()
            / self.recent_confidences.len() as f64;

        let std_dev = variance.sqrt();

        // Consistency is inversely related to standard deviation
        self.consistency = (1.0 - std_dev.min(1.0)).max(0.0);
    }

    /// Reset quality metrics
    pub fn reset(&mut self) {
        self.avg_confidence = 0.0;
        self.min_confidence = 1.0;
        self.max_confidence = 0.0;
        self.accuracy = None;
        self.consistency = 0.0;
        self.error_rate = 0.0;
        self.quality_distribution.clear();
        self.recent_confidences.clear();
    }
}

/// Performance metrics for system monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// CPU utilization percentage
    pub cpu_utilization: f64,
    /// GPU utilization percentage (if applicable)
    pub gpu_utilization: Option<f64>,
    /// Memory usage in bytes
    pub memory_usage: u64,
    /// Disk I/O metrics
    pub disk_io: DiskIOMetrics,
    /// Network I/O metrics
    pub network_io: NetworkIOMetrics,
    /// Latency measurements
    pub latency: LatencyMetrics,
    /// Thermal metrics
    pub thermal: ThermalMetrics,
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            cpu_utilization: 0.0,
            gpu_utilization: None,
            memory_usage: 0,
            disk_io: DiskIOMetrics::default(),
            network_io: NetworkIOMetrics::default(),
            latency: LatencyMetrics::default(),
            thermal: ThermalMetrics::default(),
        }
    }
}

impl PerformanceMetrics {
    /// Update throughput metrics
    pub fn update_throughput(&mut self) {
        // This would typically be updated by a monitoring system
        // For now, we'll just update the timestamp
    }

    /// Reset performance metrics
    pub fn reset(&mut self) {
        self.cpu_utilization = 0.0;
        self.gpu_utilization = None;
        self.memory_usage = 0;
        self.disk_io.reset();
        self.network_io.reset();
        self.latency.reset();
        self.thermal.reset();
    }
}

/// Disk I/O metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiskIOMetrics {
    /// Bytes read from disk
    pub bytes_read: u64,
    /// Bytes written to disk
    pub bytes_written: u64,
    /// Read operations per second
    pub read_ops_per_sec: f64,
    /// Write operations per second
    pub write_ops_per_sec: f64,
    /// Average read latency
    pub avg_read_latency: Duration,
    /// Average write latency
    pub avg_write_latency: Duration,
}

impl Default for DiskIOMetrics {
    fn default() -> Self {
        Self {
            bytes_read: 0,
            bytes_written: 0,
            read_ops_per_sec: 0.0,
            write_ops_per_sec: 0.0,
            avg_read_latency: Duration::from_millis(0),
            avg_write_latency: Duration::from_millis(0),
        }
    }
}

impl DiskIOMetrics {
    pub fn reset(&mut self) {
        self.bytes_read = 0;
        self.bytes_written = 0;
        self.read_ops_per_sec = 0.0;
        self.write_ops_per_sec = 0.0;
        self.avg_read_latency = Duration::from_millis(0);
        self.avg_write_latency = Duration::from_millis(0);
    }
}

/// Network I/O metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkIOMetrics {
    /// Bytes received
    pub bytes_received: u64,
    /// Bytes sent
    pub bytes_sent: u64,
    /// Packets received
    pub packets_received: u64,
    /// Packets sent
    pub packets_sent: u64,
    /// Network bandwidth utilization
    pub bandwidth_utilization: f64,
    /// Network latency
    pub network_latency: Duration,
}

impl Default for NetworkIOMetrics {
    fn default() -> Self {
        Self {
            bytes_received: 0,
            bytes_sent: 0,
            packets_received: 0,
            packets_sent: 0,
            bandwidth_utilization: 0.0,
            network_latency: Duration::from_millis(0),
        }
    }
}

impl NetworkIOMetrics {
    pub fn reset(&mut self) {
        self.bytes_received = 0;
        self.bytes_sent = 0;
        self.packets_received = 0;
        self.packets_sent = 0;
        self.bandwidth_utilization = 0.0;
        self.network_latency = Duration::from_millis(0);
    }
}

/// Latency metrics for various operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyMetrics {
    /// End-to-end processing latency
    pub end_to_end_latency: Duration,
    /// Preprocessing latency
    pub preprocessing_latency: Duration,
    /// Inference latency
    pub inference_latency: Duration,
    /// Post-processing latency
    pub postprocessing_latency: Duration,
    /// Percentile latencies
    pub latency_percentiles: LatencyPercentiles,
}

impl Default for LatencyMetrics {
    fn default() -> Self {
        Self {
            end_to_end_latency: Duration::from_millis(0),
            preprocessing_latency: Duration::from_millis(0),
            inference_latency: Duration::from_millis(0),
            postprocessing_latency: Duration::from_millis(0),
            latency_percentiles: LatencyPercentiles::default(),
        }
    }
}

impl LatencyMetrics {
    pub fn reset(&mut self) {
        self.end_to_end_latency = Duration::from_millis(0);
        self.preprocessing_latency = Duration::from_millis(0);
        self.inference_latency = Duration::from_millis(0);
        self.postprocessing_latency = Duration::from_millis(0);
        self.latency_percentiles.reset();
    }
}

/// Latency percentile measurements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyPercentiles {
    /// 50th percentile (median)
    pub p50: Duration,
    /// 90th percentile
    pub p90: Duration,
    /// 95th percentile
    pub p95: Duration,
    /// 99th percentile
    pub p99: Duration,
    /// 99.9th percentile
    pub p999: Duration,
}

impl Default for LatencyPercentiles {
    fn default() -> Self {
        Self {
            p50: Duration::from_millis(0),
            p90: Duration::from_millis(0),
            p95: Duration::from_millis(0),
            p99: Duration::from_millis(0),
            p999: Duration::from_millis(0),
        }
    }
}

impl LatencyPercentiles {
    pub fn reset(&mut self) {
        self.p50 = Duration::from_millis(0);
        self.p90 = Duration::from_millis(0);
        self.p95 = Duration::from_millis(0);
        self.p99 = Duration::from_millis(0);
        self.p999 = Duration::from_millis(0);
    }
}

/// Thermal metrics for monitoring system temperature
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermalMetrics {
    /// CPU temperature in Celsius
    pub cpu_temperature: f32,
    /// GPU temperature in Celsius (if applicable)
    pub gpu_temperature: Option<f32>,
    /// Ambient temperature
    pub ambient_temperature: Option<f32>,
    /// Thermal throttling events
    pub throttling_events: u32,
    /// Fan speeds (RPM)
    pub fan_speeds: Vec<u32>,
}

impl Default for ThermalMetrics {
    fn default() -> Self {
        Self {
            cpu_temperature: 0.0,
            gpu_temperature: None,
            ambient_temperature: None,
            throttling_events: 0,
            fan_speeds: vec![],
        }
    }
}

impl ThermalMetrics {
    pub fn reset(&mut self) {
        self.cpu_temperature = 0.0;
        self.gpu_temperature = None;
        self.ambient_temperature = None;
        self.throttling_events = 0;
        self.fan_speeds.clear();
    }
}

/// Resource utilization metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUtilization {
    /// CPU utilization percentage
    pub cpu_utilization: f64,
    /// GPU utilization percentage (if applicable)
    pub gpu_utilization: Option<f64>,
    /// Memory usage in MB
    pub memory_usage_mb: f64,
    /// GPU memory usage in MB (if applicable)
    pub gpu_memory_usage_mb: Option<f64>,
    /// Disk space usage
    pub disk_usage: DiskUsage,
    /// Process-specific resource usage
    pub process_usage: ProcessResourceUsage,
}

impl Default for ResourceUtilization {
    fn default() -> Self {
        Self {
            cpu_utilization: 0.0,
            gpu_utilization: None,
            memory_usage_mb: 0.0,
            gpu_memory_usage_mb: None,
            disk_usage: DiskUsage::default(),
            process_usage: ProcessResourceUsage::default(),
        }
    }
}

impl ResourceUtilization {
    /// Update resource usage measurements
    pub fn update_usage(&mut self, cpu: f64, memory: u64, gpu: Option<f64>) {
        self.cpu_utilization = cpu;
        self.memory_usage_mb = memory as f64 / (1024.0 * 1024.0);
        self.gpu_utilization = gpu;
    }

    pub fn reset(&mut self) {
        self.cpu_utilization = 0.0;
        self.gpu_utilization = None;
        self.memory_usage_mb = 0.0;
        self.gpu_memory_usage_mb = None;
        self.disk_usage.reset();
        self.process_usage.reset();
    }
}

/// Disk usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiskUsage {
    /// Total disk space in bytes
    pub total_space: u64,
    /// Used disk space in bytes
    pub used_space: u64,
    /// Available disk space in bytes
    pub available_space: u64,
    /// Usage percentage
    pub usage_percentage: f64,
}

impl Default for DiskUsage {
    fn default() -> Self {
        Self {
            total_space: 0,
            used_space: 0,
            available_space: 0,
            usage_percentage: 0.0,
        }
    }
}

impl DiskUsage {
    pub fn reset(&mut self) {
        self.total_space = 0;
        self.used_space = 0;
        self.available_space = 0;
        self.usage_percentage = 0.0;
    }
}

/// Process-specific resource usage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessResourceUsage {
    /// Process CPU usage percentage
    pub process_cpu: f64,
    /// Process memory usage in bytes
    pub process_memory: u64,
    /// Process thread count
    pub thread_count: u32,
    /// Process handle count
    pub handle_count: u32,
}

impl Default for ProcessResourceUsage {
    fn default() -> Self {
        Self {
            process_cpu: 0.0,
            process_memory: 0,
            thread_count: 0,
            handle_count: 0,
        }
    }
}

impl ProcessResourceUsage {
    pub fn reset(&mut self) {
        self.process_cpu = 0.0;
        self.process_memory = 0;
        self.thread_count = 0;
        self.handle_count = 0;
    }
}

/// Error tracking and analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorTracking {
    /// Total number of errors
    pub total_errors: u64,
    /// Error rate (errors per operation)
    pub error_rate: f64,
    /// Error counts by type
    pub error_counts: HashMap<ErrorType, u64>,
    /// Recent errors for analysis
    pub recent_errors: VecDeque<ErrorRecord>,
    /// Error trends
    pub error_trends: ErrorTrends,
}

impl Default for ErrorTracking {
    fn default() -> Self {
        Self {
            total_errors: 0,
            error_rate: 0.0,
            error_counts: HashMap::new(),
            recent_errors: VecDeque::with_capacity(100),
            error_trends: ErrorTrends::default(),
        }
    }
}

impl ErrorTracking {
    /// Record an error occurrence
    pub fn record_error(&mut self, error_type: ErrorType, message: String) {
        self.total_errors += 1;
        *self.error_counts.entry(error_type).or_insert(0) += 1;

        let error_record = ErrorRecord {
            error_type,
            message,
            timestamp: SystemTime::now(),
        };

        self.recent_errors.push_back(error_record);
        if self.recent_errors.len() > 100 {
            self.recent_errors.pop_front();
        }

        self.update_error_rate();
        self.error_trends.update();
    }

    /// Update error rate calculation
    fn update_error_rate(&mut self) {
        // This is a simplified calculation - in practice you'd want to consider
        // the total number of operations
        if self.total_errors > 0 {
            self.error_rate = self.total_errors as f64 / 1000.0; // Assuming 1000 operations
        }
    }

    pub fn reset(&mut self) {
        self.total_errors = 0;
        self.error_rate = 0.0;
        self.error_counts.clear();
        self.recent_errors.clear();
        self.error_trends.reset();
    }
}

/// Error types for classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ErrorType {
    /// Input validation error
    InputValidation,
    /// Preprocessing error
    Preprocessing,
    /// Model inference error
    Inference,
    /// Post-processing error
    PostProcessing,
    /// Memory allocation error
    Memory,
    /// Network/IO error
    Network,
    /// Configuration error
    Configuration,
    /// Hardware error
    Hardware,
    /// Unknown error
    Unknown,
}

/// Error record for tracking individual errors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorRecord {
    /// Type of error
    pub error_type: ErrorType,
    /// Error message
    pub message: String,
    /// Timestamp when error occurred
    pub timestamp: SystemTime,
}

/// Error trends analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorTrends {
    /// Errors in the last hour
    pub errors_last_hour: u64,
    /// Errors in the last day
    pub errors_last_day: u64,
    /// Error rate trend (increasing/decreasing)
    pub trend_direction: TrendDirection,
    /// Error pattern analysis
    pub patterns: Vec<ErrorPattern>,
}

impl Default for ErrorTrends {
    fn default() -> Self {
        Self {
            errors_last_hour: 0,
            errors_last_day: 0,
            trend_direction: TrendDirection::Stable,
            patterns: vec![],
        }
    }
}

impl ErrorTrends {
    pub fn update(&mut self) {
        // This would typically analyze recent errors to identify trends
        // For now, we'll just increment counters
        self.errors_last_hour += 1;
        self.errors_last_day += 1;
    }

    pub fn reset(&mut self) {
        self.errors_last_hour = 0;
        self.errors_last_day = 0;
        self.trend_direction = TrendDirection::Stable;
        self.patterns.clear();
    }
}

/// Trend direction for metrics
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrendDirection {
    /// Metric is increasing
    Increasing,
    /// Metric is decreasing
    Decreasing,
    /// Metric is stable
    Stable,
    /// Trend is unknown/insufficient data
    Unknown,
}

/// Error pattern for analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorPattern {
    /// Pattern description
    pub description: String,
    /// Pattern frequency
    pub frequency: u64,
    /// Pattern confidence
    pub confidence: f64,
}

/// Historical metrics storage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoricalMetrics {
    /// Hourly aggregated metrics
    pub hourly_metrics: VecDeque<MetricsSummary>,
    /// Daily aggregated metrics
    pub daily_metrics: VecDeque<MetricsSummary>,
    /// Maximum history length
    pub max_history_length: usize,
}

impl Default for HistoricalMetrics {
    fn default() -> Self {
        Self {
            hourly_metrics: VecDeque::with_capacity(24), // 24 hours
            daily_metrics: VecDeque::with_capacity(30),  // 30 days
            max_history_length: 1000,
        }
    }
}

impl HistoricalMetrics {
    /// Add hourly metrics snapshot
    pub fn add_hourly_snapshot(&mut self, summary: MetricsSummary) {
        self.hourly_metrics.push_back(summary);
        if self.hourly_metrics.len() > 24 {
            self.hourly_metrics.pop_front();
        }
    }

    /// Add daily metrics snapshot
    pub fn add_daily_snapshot(&mut self, summary: MetricsSummary) {
        self.daily_metrics.push_back(summary);
        if self.daily_metrics.len() > 30 {
            self.daily_metrics.pop_front();
        }
    }
}

/// Summary of all metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsSummary {
    /// Total images processed
    pub total_images_processed: u64,
    /// Average processing time
    pub average_processing_time: Duration,
    /// Current throughput
    pub current_throughput: f64,
    /// Average quality score
    pub average_quality: f64,
    /// Error rate
    pub error_rate: f64,
    /// CPU utilization
    pub cpu_utilization: f64,
    /// Memory utilization in MB
    pub memory_utilization: f64,
    /// GPU utilization (if applicable)
    pub gpu_utilization: Option<f64>,
    /// System uptime
    pub uptime: Duration,
}

impl Default for MetricsSummary {
    fn default() -> Self {
        Self {
            total_images_processed: 0,
            average_processing_time: Duration::from_millis(0),
            current_throughput: 0.0,
            average_quality: 0.0,
            error_rate: 0.0,
            cpu_utilization: 0.0,
            memory_utilization: 0.0,
            gpu_utilization: None,
            uptime: Duration::from_secs(0),
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cv_metrics_basic_operations() {
        let mut metrics = CVMetrics::new();

        // Test recording image processing
        metrics.record_image_processed(Duration::from_millis(100), 0.85);
        assert_eq!(metrics.processing_stats.total_images, 1);
        assert_eq!(metrics.quality_metrics.recent_confidences.len(), 1);

        // Test recording error
        metrics.record_error(ErrorType::Inference, "Test error".to_string());
        assert_eq!(metrics.error_tracking.total_errors, 1);

        // Test recording resource usage
        metrics.record_resource_usage(50.0, 1024 * 1024 * 1024, Some(75.0));
        assert_eq!(metrics.resource_utilization.cpu_utilization, 50.0);
    }

    #[test]
    fn test_processing_statistics() {
        let mut stats = ProcessingStatistics::default();

        stats.record_image(Duration::from_millis(100));
        stats.record_image(Duration::from_millis(200));
        stats.record_failure();

        assert_eq!(stats.total_images, 2);
        assert_eq!(stats.successful_predictions, 2);
        assert_eq!(stats.failed_predictions, 1);
        assert_eq!(stats.success_rate(), 2.0 / 3.0);
    }

    #[test]
    fn test_quality_metrics() {
        let mut quality = QualityMetrics::default();

        quality.record_quality(0.8);
        quality.record_quality(0.9);
        quality.record_quality(0.7);

        assert_eq!(quality.recent_confidences.len(), 3);
        assert!((quality.avg_confidence - 0.8).abs() < 0.01);
        assert_eq!(quality.min_confidence, 0.7);
        assert_eq!(quality.max_confidence, 0.9);
    }

    #[test]
    fn test_error_tracking() {
        let mut error_tracking = ErrorTracking::default();

        error_tracking.record_error(ErrorType::Inference, "Model failed".to_string());
        error_tracking.record_error(ErrorType::Preprocessing, "Invalid input".to_string());
        error_tracking.record_error(ErrorType::Inference, "Another inference error".to_string());

        assert_eq!(error_tracking.total_errors, 3);
        assert_eq!(error_tracking.error_counts[&ErrorType::Inference], 2);
        assert_eq!(error_tracking.error_counts[&ErrorType::Preprocessing], 1);
        assert_eq!(error_tracking.recent_errors.len(), 3);
    }

    #[test]
    fn test_metrics_summary() {
        let mut metrics = CVMetrics::new();

        metrics.record_image_processed(Duration::from_millis(150), 0.75);
        metrics.record_image_processed(Duration::from_millis(120), 0.85);
        metrics.record_resource_usage(60.0, 512 * 1024 * 1024, Some(70.0));

        let summary = metrics.generate_summary();
        assert_eq!(summary.total_images_processed, 2);
        assert_eq!(summary.cpu_utilization, 60.0);
        assert_eq!(summary.gpu_utilization, Some(70.0));
    }

    #[test]
    fn test_metrics_reset() {
        let mut metrics = CVMetrics::new();

        metrics.record_image_processed(Duration::from_millis(100), 0.8);
        metrics.record_error(ErrorType::Inference, "Test".to_string());

        metrics.reset();

        assert_eq!(metrics.processing_stats.total_images, 0);
        assert_eq!(metrics.error_tracking.total_errors, 0);
        assert_eq!(metrics.quality_metrics.recent_confidences.len(), 0);
    }

    #[test]
    fn test_historical_metrics() {
        let mut historical = HistoricalMetrics::default();
        let summary = MetricsSummary::default();

        historical.add_hourly_snapshot(summary.clone());
        historical.add_daily_snapshot(summary);

        assert_eq!(historical.hourly_metrics.len(), 1);
        assert_eq!(historical.daily_metrics.len(), 1);
    }
}
