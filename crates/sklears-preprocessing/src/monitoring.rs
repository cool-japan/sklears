//! Transformation Monitoring and Performance Metrics
//!
//! Provides comprehensive monitoring, logging, and performance tracking
//! for preprocessing transformations.

use scirs2_core::ndarray::Array2;
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Transformation metrics collected during preprocessing
#[derive(Debug, Clone)]
pub struct TransformationMetrics {
    /// Transformation name
    pub name: String,
    /// Start time
    pub start_time: Option<Instant>,
    /// End time
    pub end_time: Option<Instant>,
    /// Duration in milliseconds
    pub duration_ms: Option<f64>,
    /// Input shape (rows, cols)
    pub input_shape: (usize, usize),
    /// Output shape (rows, cols)
    pub output_shape: (usize, usize),
    /// Memory usage in bytes
    pub memory_bytes: Option<usize>,
    /// Number of NaN values in input
    pub input_nan_count: usize,
    /// Number of NaN values in output
    pub output_nan_count: usize,
    /// Transformation success
    pub success: bool,
    /// Error message if failed
    pub error_message: Option<String>,
    /// Custom metrics
    pub custom_metrics: HashMap<String, f64>,
}

impl TransformationMetrics {
    /// Create new transformation metrics
    pub fn new(name: String, input_shape: (usize, usize)) -> Self {
        Self {
            name,
            start_time: None,
            end_time: None,
            duration_ms: None,
            input_shape,
            output_shape: (0, 0),
            memory_bytes: None,
            input_nan_count: 0,
            output_nan_count: 0,
            success: false,
            error_message: None,
            custom_metrics: HashMap::new(),
        }
    }

    /// Mark transformation as started
    pub fn start(&mut self) {
        self.start_time = Some(Instant::now());
    }

    /// Mark transformation as completed
    pub fn complete(&mut self, output_shape: (usize, usize)) {
        self.end_time = Some(Instant::now());
        self.output_shape = output_shape;
        self.success = true;

        if let (Some(start), Some(end)) = (self.start_time, self.end_time) {
            let duration = end.duration_since(start);
            self.duration_ms = Some(duration.as_secs_f64() * 1000.0);
        }
    }

    /// Mark transformation as failed
    pub fn fail(&mut self, error: String) {
        self.end_time = Some(Instant::now());
        self.success = false;
        self.error_message = Some(error);

        if let (Some(start), Some(end)) = (self.start_time, self.end_time) {
            let duration = end.duration_since(start);
            self.duration_ms = Some(duration.as_secs_f64() * 1000.0);
        }
    }

    /// Add custom metric
    pub fn add_metric(&mut self, name: String, value: f64) {
        self.custom_metrics.insert(name, value);
    }

    /// Get throughput (elements per second)
    pub fn throughput(&self) -> Option<f64> {
        if let Some(duration_ms) = self.duration_ms {
            if duration_ms > 0.0 {
                let elements = (self.input_shape.0 * self.input_shape.1) as f64;
                Some(elements / (duration_ms / 1000.0))
            } else {
                None
            }
        } else {
            None
        }
    }

    /// Get memory efficiency (bytes per element)
    pub fn memory_efficiency(&self) -> Option<f64> {
        if let Some(mem) = self.memory_bytes {
            let elements = (self.input_shape.0 * self.input_shape.1) as f64;
            Some(mem as f64 / elements)
        } else {
            None
        }
    }
}

/// Pipeline monitoring session
#[derive(Debug)]
pub struct MonitoringSession {
    /// Session name
    pub name: String,
    /// Start time
    pub start_time: Instant,
    /// All transformation metrics
    pub metrics: Vec<TransformationMetrics>,
    /// Session configuration
    pub config: MonitoringConfig,
}

/// Monitoring configuration
#[derive(Debug, Clone)]
pub struct MonitoringConfig {
    /// Enable detailed logging
    pub enable_logging: bool,
    /// Log level
    pub log_level: LogLevel,
    /// Track memory usage
    pub track_memory: bool,
    /// Track NaN values
    pub track_nan: bool,
    /// Collect custom metrics
    pub collect_custom_metrics: bool,
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            enable_logging: true,
            log_level: LogLevel::Info,
            track_memory: true,
            track_nan: true,
            collect_custom_metrics: false,
        }
    }
}

/// Log levels
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum LogLevel {
    Debug,
    Info,
    Warning,
    Error,
}

impl MonitoringSession {
    /// Create a new monitoring session
    pub fn new(name: String) -> Self {
        Self {
            name,
            start_time: Instant::now(),
            metrics: Vec::new(),
            config: MonitoringConfig::default(),
        }
    }

    /// Create a session with custom configuration
    pub fn with_config(name: String, config: MonitoringConfig) -> Self {
        Self {
            name,
            start_time: Instant::now(),
            metrics: Vec::new(),
            config,
        }
    }

    /// Start tracking a transformation
    pub fn start_transformation(&mut self, name: String, input: &Array2<f64>) -> usize {
        let mut metrics = TransformationMetrics::new(name.clone(), (input.nrows(), input.ncols()));
        metrics.start();

        if self.config.track_nan {
            metrics.input_nan_count = input.iter().filter(|v| v.is_nan()).count();
        }

        if self.config.enable_logging && self.config.log_level <= LogLevel::Debug {
            println!(
                "[DEBUG] Starting transformation '{}' on data shape {:?}",
                name,
                (input.nrows(), input.ncols())
            );
        }

        self.metrics.push(metrics);
        self.metrics.len() - 1 // Return index
    }

    /// Complete a transformation
    pub fn complete_transformation(
        &mut self,
        index: usize,
        output: &Array2<f64>,
    ) -> Result<(), String> {
        if index >= self.metrics.len() {
            return Err("Invalid metrics index".to_string());
        }

        let metrics = &mut self.metrics[index];
        metrics.complete((output.nrows(), output.ncols()));

        if self.config.track_nan {
            metrics.output_nan_count = output.iter().filter(|v| v.is_nan()).count();
        }

        if self.config.track_memory {
            let memory = (output.nrows() * output.ncols()) * std::mem::size_of::<f64>();
            metrics.memory_bytes = Some(memory);
        }

        if self.config.enable_logging && self.config.log_level <= LogLevel::Info {
            println!(
                "[INFO] Completed transformation '{}' in {:.2}ms (throughput: {:.0} elem/s)",
                metrics.name,
                metrics.duration_ms.unwrap_or(0.0),
                metrics.throughput().unwrap_or(0.0)
            );
        }

        Ok(())
    }

    /// Fail a transformation
    pub fn fail_transformation(&mut self, index: usize, error: String) -> Result<(), String> {
        if index >= self.metrics.len() {
            return Err("Invalid metrics index".to_string());
        }

        let metrics = &mut self.metrics[index];
        metrics.fail(error.clone());

        if self.config.enable_logging && self.config.log_level <= LogLevel::Error {
            println!(
                "[ERROR] Transformation '{}' failed: {}",
                metrics.name, error
            );
        }

        Ok(())
    }

    /// Get total session duration
    pub fn total_duration(&self) -> Duration {
        Instant::now().duration_since(self.start_time)
    }

    /// Get total processing time (sum of all transformations)
    pub fn total_processing_time(&self) -> f64 {
        self.metrics.iter().filter_map(|m| m.duration_ms).sum()
    }

    /// Get average throughput
    pub fn average_throughput(&self) -> Option<f64> {
        let throughputs: Vec<f64> = self.metrics.iter().filter_map(|m| m.throughput()).collect();

        if throughputs.is_empty() {
            None
        } else {
            Some(throughputs.iter().sum::<f64>() / throughputs.len() as f64)
        }
    }

    /// Get successful transformations
    pub fn successful_count(&self) -> usize {
        self.metrics.iter().filter(|m| m.success).count()
    }

    /// Get failed transformations
    pub fn failed_count(&self) -> usize {
        self.metrics.iter().filter(|m| !m.success).count()
    }

    /// Generate summary report
    pub fn summary(&self) -> MonitoringSummary {
        let total_transformations = self.metrics.len();
        let successful = self.successful_count();
        let failed = self.failed_count();
        let total_duration = self.total_duration();
        let processing_time = self.total_processing_time();
        let avg_throughput = self.average_throughput();

        let total_input_elements: usize = self
            .metrics
            .iter()
            .map(|m| m.input_shape.0 * m.input_shape.1)
            .sum();

        let total_output_elements: usize = self
            .metrics
            .iter()
            .map(|m| m.output_shape.0 * m.output_shape.1)
            .sum();

        let total_memory: usize = self.metrics.iter().filter_map(|m| m.memory_bytes).sum();

        MonitoringSummary {
            session_name: self.name.clone(),
            total_transformations,
            successful_transformations: successful,
            failed_transformations: failed,
            total_duration_ms: total_duration.as_secs_f64() * 1000.0,
            total_processing_ms: processing_time,
            overhead_ms: total_duration.as_secs_f64() * 1000.0 - processing_time,
            average_throughput: avg_throughput,
            total_input_elements,
            total_output_elements,
            total_memory_bytes: total_memory,
            slowest_transformation: self.find_slowest(),
            fastest_transformation: self.find_fastest(),
        }
    }

    /// Find slowest transformation
    fn find_slowest(&self) -> Option<String> {
        self.metrics
            .iter()
            .filter(|m| m.duration_ms.is_some())
            .max_by(|a, b| {
                a.duration_ms
                    .unwrap()
                    .partial_cmp(&b.duration_ms.unwrap())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|m| format!("{} ({:.2}ms)", m.name, m.duration_ms.unwrap()))
    }

    /// Find fastest transformation
    fn find_fastest(&self) -> Option<String> {
        self.metrics
            .iter()
            .filter(|m| m.duration_ms.is_some())
            .min_by(|a, b| {
                a.duration_ms
                    .unwrap()
                    .partial_cmp(&b.duration_ms.unwrap())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|m| format!("{} ({:.2}ms)", m.name, m.duration_ms.unwrap()))
    }

    /// Print summary
    pub fn print_summary(&self) {
        let summary = self.summary();
        summary.print();
    }
}

/// Monitoring summary report
#[derive(Debug, Clone)]
pub struct MonitoringSummary {
    pub session_name: String,
    pub total_transformations: usize,
    pub successful_transformations: usize,
    pub failed_transformations: usize,
    pub total_duration_ms: f64,
    pub total_processing_ms: f64,
    pub overhead_ms: f64,
    pub average_throughput: Option<f64>,
    pub total_input_elements: usize,
    pub total_output_elements: usize,
    pub total_memory_bytes: usize,
    pub slowest_transformation: Option<String>,
    pub fastest_transformation: Option<String>,
}

impl MonitoringSummary {
    /// Print formatted summary
    pub fn print(&self) {
        println!("\n{}", "=".repeat(60));
        println!("Monitoring Summary: {}", self.session_name);
        println!("{}", "=".repeat(60));
        println!();

        println!("Transformations:");
        println!("  Total: {}", self.total_transformations);
        println!("  Successful: {}", self.successful_transformations);
        println!("  Failed: {}", self.failed_transformations);
        println!();

        println!("Performance:");
        println!("  Total Duration: {:.2} ms", self.total_duration_ms);
        println!("  Processing Time: {:.2} ms", self.total_processing_ms);
        println!(
            "  Overhead: {:.2} ms ({:.1}%)",
            self.overhead_ms,
            (self.overhead_ms / self.total_duration_ms) * 100.0
        );
        if let Some(throughput) = self.average_throughput {
            println!("  Average Throughput: {:.0} elements/s", throughput);
        }
        println!();

        println!("Data:");
        println!("  Total Input Elements: {}", self.total_input_elements);
        println!("  Total Output Elements: {}", self.total_output_elements);
        println!(
            "  Total Memory: {:.2} MB",
            self.total_memory_bytes as f64 / 1024.0 / 1024.0
        );
        println!();

        if let Some(slowest) = &self.slowest_transformation {
            println!("Slowest Transformation: {}", slowest);
        }
        if let Some(fastest) = &self.fastest_transformation {
            println!("Fastest Transformation: {}", fastest);
        }

        println!("{}", "=".repeat(60));
    }

    /// Get efficiency percentage
    pub fn efficiency(&self) -> f64 {
        if self.total_duration_ms > 0.0 {
            (self.total_processing_ms / self.total_duration_ms) * 100.0
        } else {
            0.0
        }
    }

    /// Check if performance is acceptable
    pub fn is_acceptable(&self, min_throughput: f64, max_overhead_percent: f64) -> bool {
        let throughput_ok = self
            .average_throughput
            .map(|t| t >= min_throughput)
            .unwrap_or(false);

        let overhead_percent = (self.overhead_ms / self.total_duration_ms) * 100.0;
        let overhead_ok = overhead_percent <= max_overhead_percent;

        throughput_ok && overhead_ok
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::random::essentials::Normal;
    use scirs2_core::random::{seeded_rng, Distribution};
    use std::thread;

    fn generate_test_data(nrows: usize, ncols: usize, seed: u64) -> Array2<f64> {
        let mut rng = seeded_rng(seed);
        let normal = Normal::new(0.0, 1.0).unwrap();

        let data: Vec<f64> = (0..nrows * ncols)
            .map(|_| normal.sample(&mut rng))
            .collect();

        Array2::from_shape_vec((nrows, ncols), data).unwrap()
    }

    #[test]
    fn test_transformation_metrics() {
        let mut metrics = TransformationMetrics::new("test".to_string(), (100, 10));
        metrics.start();

        thread::sleep(Duration::from_millis(10));

        metrics.complete((100, 10));

        assert!(metrics.success);
        assert!(metrics.duration_ms.is_some());
        assert!(metrics.duration_ms.unwrap() >= 10.0);
    }

    #[test]
    fn test_monitoring_session() {
        let mut session = MonitoringSession::new("test_session".to_string());

        let input = generate_test_data(100, 10, 42);
        let output = input.clone();

        let idx = session.start_transformation("StandardScaler".to_string(), &input);
        thread::sleep(Duration::from_millis(5));
        session.complete_transformation(idx, &output).unwrap();

        assert_eq!(session.successful_count(), 1);
        assert_eq!(session.failed_count(), 0);
    }

    #[test]
    fn test_monitoring_session_failure() {
        let mut session = MonitoringSession::new("test_session".to_string());

        let input = generate_test_data(100, 10, 123);

        let idx = session.start_transformation("Faulty".to_string(), &input);
        session
            .fail_transformation(idx, "Test error".to_string())
            .unwrap();

        assert_eq!(session.successful_count(), 0);
        assert_eq!(session.failed_count(), 1);
    }

    #[test]
    fn test_throughput_calculation() {
        let mut metrics = TransformationMetrics::new("test".to_string(), (1000, 100));
        metrics.start();

        thread::sleep(Duration::from_millis(100));

        metrics.complete((1000, 100));

        let throughput = metrics.throughput();
        assert!(throughput.is_some());
        assert!(throughput.unwrap() > 0.0);
    }

    #[test]
    fn test_monitoring_summary() {
        let mut session = MonitoringSession::new("test".to_string());

        let input1 = generate_test_data(100, 10, 42);
        let input2 = generate_test_data(200, 20, 123);

        let idx1 = session.start_transformation("Step1".to_string(), &input1);
        thread::sleep(Duration::from_millis(5));
        session.complete_transformation(idx1, &input1).unwrap();

        let idx2 = session.start_transformation("Step2".to_string(), &input2);
        thread::sleep(Duration::from_millis(10));
        session.complete_transformation(idx2, &input2).unwrap();

        let summary = session.summary();

        assert_eq!(summary.total_transformations, 2);
        assert_eq!(summary.successful_transformations, 2);
        assert!(summary.slowest_transformation.is_some());
        assert!(summary.fastest_transformation.is_some());
    }

    #[test]
    fn test_custom_metrics() {
        let mut metrics = TransformationMetrics::new("test".to_string(), (100, 10));
        metrics.add_metric("accuracy".to_string(), 0.95);
        metrics.add_metric("loss".to_string(), 0.05);

        assert_eq!(metrics.custom_metrics.len(), 2);
        assert_eq!(metrics.custom_metrics.get("accuracy"), Some(&0.95));
    }
}
