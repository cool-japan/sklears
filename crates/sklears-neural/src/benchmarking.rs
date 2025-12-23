//! Comprehensive benchmarking framework for neural network performance evaluation.
//!
//! This module provides tools for benchmarking neural network training and inference:
//! - Training speed benchmarks
//! - Inference latency measurements
//! - Memory usage profiling
//! - Throughput measurements
//! - Accuracy vs speed tradeoff analysis
//! - Comparison with baseline implementations

use crate::NeuralResult;
use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::{error::SklearsError, types::FloatBounds};
use std::time::{Duration, Instant};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Benchmark configuration
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct BenchmarkConfig {
    /// Number of warmup iterations
    pub warmup_iterations: usize,
    /// Number of benchmark iterations
    pub benchmark_iterations: usize,
    /// Batch sizes to test
    pub batch_sizes: Vec<usize>,
    /// Whether to measure memory usage
    pub measure_memory: bool,
    /// Whether to measure throughput
    pub measure_throughput: bool,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            warmup_iterations: 10,
            benchmark_iterations: 100,
            batch_sizes: vec![1, 8, 16, 32, 64, 128],
            measure_memory: true,
            measure_throughput: true,
        }
    }
}

/// Benchmark results for a single configuration
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct BenchmarkResult {
    /// Batch size used
    pub batch_size: usize,
    /// Mean latency (milliseconds)
    pub mean_latency_ms: f64,
    /// Standard deviation of latency
    pub std_latency_ms: f64,
    /// Minimum latency
    pub min_latency_ms: f64,
    /// Maximum latency
    pub max_latency_ms: f64,
    /// Median latency
    pub median_latency_ms: f64,
    /// Throughput (samples per second)
    pub throughput_samples_per_sec: f64,
    /// Memory usage (MB) - estimated
    pub memory_mb: Option<f64>,
}

impl BenchmarkResult {
    /// Create a new benchmark result
    pub fn new(batch_size: usize, latencies: Vec<Duration>) -> Self {
        let latencies_ms: Vec<f64> = latencies.iter().map(|d| d.as_secs_f64() * 1000.0).collect();

        let mean = latencies_ms.iter().sum::<f64>() / latencies_ms.len() as f64;
        let variance = latencies_ms.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
            / latencies_ms.len() as f64;
        let std = variance.sqrt();

        let mut sorted_latencies = latencies_ms.clone();
        sorted_latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let min = *sorted_latencies.first().unwrap();
        let max = *sorted_latencies.last().unwrap();
        let median = sorted_latencies[sorted_latencies.len() / 2];

        // Throughput: samples per second
        let throughput = (batch_size as f64) / (mean / 1000.0);

        Self {
            batch_size,
            mean_latency_ms: mean,
            std_latency_ms: std,
            min_latency_ms: min,
            max_latency_ms: max,
            median_latency_ms: median,
            throughput_samples_per_sec: throughput,
            memory_mb: None,
        }
    }

    /// Set memory usage
    pub fn with_memory(mut self, memory_mb: f64) -> Self {
        self.memory_mb = Some(memory_mb);
        self
    }

    /// Get latency percentile
    pub fn latency_percentile(&self, percentile: f64) -> f64 {
        // Simplified: use mean + std approximation
        self.mean_latency_ms + self.std_latency_ms * percentile
    }
}

/// Complete benchmark suite results
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct BenchmarkSuite {
    /// Name of the benchmark
    pub name: String,
    /// Results for each batch size
    pub results: Vec<BenchmarkResult>,
    /// Total benchmark duration
    pub total_duration_secs: f64,
}

impl BenchmarkSuite {
    /// Create a new benchmark suite
    pub fn new(name: String) -> Self {
        Self {
            name,
            results: Vec::new(),
            total_duration_secs: 0.0,
        }
    }

    /// Add a result
    pub fn add_result(&mut self, result: BenchmarkResult) {
        self.results.push(result);
    }

    /// Set total duration
    pub fn set_duration(&mut self, duration: Duration) {
        self.total_duration_secs = duration.as_secs_f64();
    }

    /// Get result for specific batch size
    pub fn get_result(&self, batch_size: usize) -> Option<&BenchmarkResult> {
        self.results.iter().find(|r| r.batch_size == batch_size)
    }

    /// Get best throughput result
    pub fn best_throughput(&self) -> Option<&BenchmarkResult> {
        self.results.iter().max_by(|a, b| {
            a.throughput_samples_per_sec
                .partial_cmp(&b.throughput_samples_per_sec)
                .unwrap()
        })
    }

    /// Get best latency result
    pub fn best_latency(&self) -> Option<&BenchmarkResult> {
        self.results
            .iter()
            .min_by(|a, b| a.mean_latency_ms.partial_cmp(&b.mean_latency_ms).unwrap())
    }

    /// Print summary
    pub fn print_summary(&self) {
        println!("\n=== Benchmark Suite: {} ===", self.name);
        println!("Total Duration: {:.2}s", self.total_duration_secs);
        println!(
            "\n{:<12} {:<15} {:<15} {:<15} {:<15}",
            "Batch Size", "Mean (ms)", "Std (ms)", "Throughput", "Memory (MB)"
        );
        println!("{}", "-".repeat(75));

        for result in &self.results {
            let memory_str = result
                .memory_mb
                .map(|m| format!("{:.2}", m))
                .unwrap_or_else(|| "N/A".to_string());

            println!(
                "{:<12} {:<15.2} {:<15.2} {:<15.2} {:<15}",
                result.batch_size,
                result.mean_latency_ms,
                result.std_latency_ms,
                result.throughput_samples_per_sec,
                memory_str
            );
        }

        if let Some(best_throughput) = self.best_throughput() {
            println!(
                "\nBest Throughput: {:.2} samples/sec @ batch_size={}",
                best_throughput.throughput_samples_per_sec, best_throughput.batch_size
            );
        }

        if let Some(best_latency) = self.best_latency() {
            println!(
                "Best Latency: {:.2}ms @ batch_size={}",
                best_latency.mean_latency_ms, best_latency.batch_size
            );
        }
    }
}

/// Benchmarking utilities
pub struct Benchmarker {
    config: BenchmarkConfig,
}

impl Benchmarker {
    /// Create a new benchmarker
    pub fn new(config: BenchmarkConfig) -> Self {
        Self { config }
    }

    /// Create with default configuration
    pub fn default() -> Self {
        Self::new(BenchmarkConfig::default())
    }

    /// Benchmark a function
    pub fn benchmark<F, T>(&self, name: &str, mut f: F) -> BenchmarkSuite
    where
        F: FnMut() -> T,
    {
        let total_start = Instant::now();
        let mut suite = BenchmarkSuite::new(name.to_string());

        // Single batch size benchmark (for non-batch operations)
        let mut latencies = Vec::new();

        // Warmup
        for _ in 0..self.config.warmup_iterations {
            let _ = f();
        }

        // Benchmark
        for _ in 0..self.config.benchmark_iterations {
            let start = Instant::now();
            let _ = f();
            let duration = start.elapsed();
            latencies.push(duration);
        }

        let result = BenchmarkResult::new(1, latencies);
        suite.add_result(result);

        suite.set_duration(total_start.elapsed());
        suite
    }

    /// Benchmark a batch operation
    pub fn benchmark_batch<F, T>(&self, name: &str, mut f: F) -> BenchmarkSuite
    where
        F: FnMut(usize) -> T,
    {
        let total_start = Instant::now();
        let mut suite = BenchmarkSuite::new(name.to_string());

        for &batch_size in &self.config.batch_sizes {
            let mut latencies = Vec::new();

            // Warmup
            for _ in 0..self.config.warmup_iterations {
                let _ = f(batch_size);
            }

            // Benchmark
            for _ in 0..self.config.benchmark_iterations {
                let start = Instant::now();
                let _ = f(batch_size);
                let duration = start.elapsed();
                latencies.push(duration);
            }

            let result = BenchmarkResult::new(batch_size, latencies);
            suite.add_result(result);
        }

        suite.set_duration(total_start.elapsed());
        suite
    }

    /// Compare two implementations
    pub fn compare<F1, F2, T1, T2>(
        &self,
        name1: &str,
        mut f1: F1,
        name2: &str,
        mut f2: F2,
    ) -> (BenchmarkSuite, BenchmarkSuite, f64)
    where
        F1: FnMut() -> T1,
        F2: FnMut() -> T2,
    {
        let suite1 = self.benchmark(name1, &mut f1);
        let suite2 = self.benchmark(name2, &mut f2);

        // Calculate speedup
        let speedup = if let (Some(r1), Some(r2)) = (suite1.results.first(), suite2.results.first())
        {
            r2.mean_latency_ms / r1.mean_latency_ms
        } else {
            1.0
        };

        (suite1, suite2, speedup)
    }
}

/// Training benchmark metrics
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct TrainingBenchmark {
    /// Epochs completed
    pub epochs: usize,
    /// Total training time
    pub total_time_secs: f64,
    /// Time per epoch
    pub time_per_epoch_secs: f64,
    /// Samples per second
    pub samples_per_sec: f64,
    /// Final training loss
    pub final_loss: f64,
    /// Final validation accuracy (if available)
    pub final_accuracy: Option<f64>,
}

impl TrainingBenchmark {
    /// Create a new training benchmark
    pub fn new(
        epochs: usize,
        total_time: Duration,
        total_samples: usize,
        final_loss: f64,
        final_accuracy: Option<f64>,
    ) -> Self {
        let total_time_secs = total_time.as_secs_f64();
        let time_per_epoch_secs = total_time_secs / epochs as f64;
        let samples_per_sec = total_samples as f64 / total_time_secs;

        Self {
            epochs,
            total_time_secs,
            time_per_epoch_secs,
            samples_per_sec,
            final_loss,
            final_accuracy,
        }
    }

    /// Print summary
    pub fn print_summary(&self) {
        println!("\n=== Training Benchmark ===");
        println!("Epochs: {}", self.epochs);
        println!("Total Time: {:.2}s", self.total_time_secs);
        println!("Time per Epoch: {:.2}s", self.time_per_epoch_secs);
        println!("Throughput: {:.2} samples/sec", self.samples_per_sec);
        println!("Final Loss: {:.4}", self.final_loss);
        if let Some(acc) = self.final_accuracy {
            println!("Final Accuracy: {:.2}%", acc * 100.0);
        }
    }
}

/// Memory profiling results
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct MemoryProfile {
    /// Peak memory usage (MB)
    pub peak_memory_mb: f64,
    /// Average memory usage (MB)
    pub avg_memory_mb: f64,
    /// Memory usage by component
    pub component_memory: Vec<(String, f64)>,
}

impl MemoryProfile {
    /// Create a new memory profile
    pub fn new() -> Self {
        Self {
            peak_memory_mb: 0.0,
            avg_memory_mb: 0.0,
            component_memory: Vec::new(),
        }
    }

    /// Add component memory
    pub fn add_component(&mut self, name: String, memory_mb: f64) {
        self.component_memory.push((name, memory_mb));
    }

    /// Print summary
    pub fn print_summary(&self) {
        println!("\n=== Memory Profile ===");
        println!("Peak Memory: {:.2} MB", self.peak_memory_mb);
        println!("Average Memory: {:.2} MB", self.avg_memory_mb);

        if !self.component_memory.is_empty() {
            println!("\nMemory by Component:");
            for (name, memory) in &self.component_memory {
                println!("  {}: {:.2} MB", name, memory);
            }
        }
    }
}

impl Default for MemoryProfile {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benchmark_config_default() {
        let config = BenchmarkConfig::default();
        assert_eq!(config.warmup_iterations, 10);
        assert_eq!(config.benchmark_iterations, 100);
        assert!(!config.batch_sizes.is_empty());
    }

    #[test]
    fn test_benchmark_result_creation() {
        let durations = vec![
            Duration::from_millis(10),
            Duration::from_millis(12),
            Duration::from_millis(11),
        ];

        let result = BenchmarkResult::new(32, durations);

        assert_eq!(result.batch_size, 32);
        assert!(result.mean_latency_ms > 0.0);
        assert!(result.std_latency_ms >= 0.0);
        assert!(result.throughput_samples_per_sec > 0.0);
    }

    #[test]
    fn test_benchmark_suite() {
        let mut suite = BenchmarkSuite::new("Test Suite".to_string());

        let result1 = BenchmarkResult::new(8, vec![Duration::from_millis(10)]);
        let result2 = BenchmarkResult::new(16, vec![Duration::from_millis(8)]);

        suite.add_result(result1);
        suite.add_result(result2);

        assert_eq!(suite.results.len(), 2);
        assert!(suite.get_result(8).is_some());
        assert!(suite.get_result(16).is_some());
        assert!(suite.get_result(32).is_none());
    }

    #[test]
    fn test_benchmark_best_throughput() {
        let mut suite = BenchmarkSuite::new("Test".to_string());

        let result1 = BenchmarkResult::new(8, vec![Duration::from_millis(10)]);
        let result2 = BenchmarkResult::new(16, vec![Duration::from_millis(8)]);

        suite.add_result(result1);
        suite.add_result(result2);

        let best = suite.best_throughput().unwrap();
        assert_eq!(best.batch_size, 16); // Higher throughput
    }

    #[test]
    fn test_benchmark_best_latency() {
        let mut suite = BenchmarkSuite::new("Test".to_string());

        let result1 = BenchmarkResult::new(8, vec![Duration::from_millis(10)]);
        let result2 = BenchmarkResult::new(16, vec![Duration::from_millis(8)]);

        suite.add_result(result1);
        suite.add_result(result2);

        let best = suite.best_latency().unwrap();
        assert_eq!(best.batch_size, 16); // Lower latency
    }

    #[test]
    fn test_benchmarker_creation() {
        let config = BenchmarkConfig::default();
        let benchmarker = Benchmarker::new(config);

        assert!(benchmarker.config.warmup_iterations > 0);
    }

    #[test]
    fn test_benchmarker_simple() {
        let config = BenchmarkConfig {
            warmup_iterations: 2,
            benchmark_iterations: 5,
            batch_sizes: vec![1],
            measure_memory: false,
            measure_throughput: true,
        };

        let benchmarker = Benchmarker::new(config);

        let suite = benchmarker.benchmark("simple_test", || {
            // Simple operation
            let mut sum = 0;
            for i in 0..100 {
                sum += i;
            }
            sum
        });

        assert_eq!(suite.results.len(), 1);
        assert!(suite.results[0].mean_latency_ms >= 0.0);
    }

    #[test]
    fn test_training_benchmark() {
        let benchmark =
            TrainingBenchmark::new(10, Duration::from_secs(100), 1000, 0.123, Some(0.95));

        assert_eq!(benchmark.epochs, 10);
        assert_eq!(benchmark.total_time_secs, 100.0);
        assert_eq!(benchmark.time_per_epoch_secs, 10.0);
        assert_eq!(benchmark.samples_per_sec, 10.0);
        assert_eq!(benchmark.final_loss, 0.123);
        assert_eq!(benchmark.final_accuracy, Some(0.95));
    }

    #[test]
    fn test_memory_profile() {
        let mut profile = MemoryProfile::new();
        profile.peak_memory_mb = 512.0;
        profile.avg_memory_mb = 256.0;

        profile.add_component("Weights".to_string(), 100.0);
        profile.add_component("Activations".to_string(), 50.0);

        assert_eq!(profile.peak_memory_mb, 512.0);
        assert_eq!(profile.component_memory.len(), 2);
    }

    #[test]
    fn test_benchmark_result_with_memory() {
        let result = BenchmarkResult::new(32, vec![Duration::from_millis(10)]);
        let result_with_mem = result.with_memory(256.0);

        assert_eq!(result_with_mem.memory_mb, Some(256.0));
    }

    #[test]
    fn test_benchmarker_compare() {
        let config = BenchmarkConfig {
            warmup_iterations: 1,
            benchmark_iterations: 3,
            batch_sizes: vec![1],
            measure_memory: false,
            measure_throughput: true,
        };

        let benchmarker = Benchmarker::new(config);

        let (suite1, suite2, speedup) = benchmarker.compare(
            "impl1",
            || {
                std::thread::sleep(Duration::from_micros(100));
                42
            },
            "impl2",
            || {
                std::thread::sleep(Duration::from_micros(200));
                42
            },
        );

        assert!(!suite1.results.is_empty());
        assert!(!suite2.results.is_empty());
        assert!(speedup > 0.0);
    }
}
