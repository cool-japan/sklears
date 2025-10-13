//! Benchmark utilities for performance testing of dummy estimator operations

use std::time::{Duration, Instant};

/// Benchmark result structure
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    /// mean_time
    pub mean_time: Duration,
    /// std_dev
    pub std_dev: Duration,
    /// throughput
    pub throughput: f64,
    /// min_time
    pub min_time: Duration,
    /// max_time
    pub max_time: Duration,
    /// iterations
    pub iterations: usize,
}

/// Simple benchmark function
pub fn benchmark_function<F>(mut func: F, iterations: usize) -> BenchmarkResult
where
    F: FnMut(),
{
    let mut times = Vec::with_capacity(iterations);

    for _ in 0..iterations {
        let start = Instant::now();
        func();
        times.push(start.elapsed());
    }

    let total_time: Duration = times.iter().sum();
    let mean_time = total_time / iterations as u32;

    let min_time = times.iter().min().copied().unwrap_or(Duration::ZERO);
    let max_time = times.iter().max().copied().unwrap_or(Duration::ZERO);

    let variance: f64 = times
        .iter()
        .map(|&t| {
            let diff = t.as_nanos() as f64 - mean_time.as_nanos() as f64;
            diff * diff
        })
        .sum::<f64>()
        / iterations as f64;

    let std_dev = Duration::from_nanos(variance.sqrt() as u64);
    let throughput = 1_000_000_000.0 / mean_time.as_nanos() as f64;

    BenchmarkResult {
        mean_time,
        std_dev,
        throughput,
        min_time,
        max_time,
        iterations,
    }
}

/// Comparative benchmark for multiple functions
pub fn comparative_benchmark<F>(
    functions: Vec<(&str, F)>,
    iterations: usize,
) -> Vec<(String, BenchmarkResult)>
where
    F: FnMut() + Clone,
{
    functions
        .into_iter()
        .map(|(name, func)| {
            let result = benchmark_function(func, iterations);
            (name.to_string(), result)
        })
        .collect()
}

/// Memory usage tracker for benchmarks
pub struct MemoryTracker {
    initial_usage: usize,
    peak_usage: usize,
}

impl Default for MemoryTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl MemoryTracker {
    pub fn new() -> Self {
        Self {
            initial_usage: get_memory_usage(),
            peak_usage: 0,
        }
    }

    pub fn update_peak(&mut self) {
        self.peak_usage = self.peak_usage.max(get_memory_usage());
    }

    pub fn memory_delta(&self) -> usize {
        self.peak_usage.saturating_sub(self.initial_usage)
    }
}

/// Get current memory usage (simplified implementation)
fn get_memory_usage() -> usize {
    // In a real implementation, you'd use platform-specific APIs
    // For now, return a placeholder value
    0
}

/// Performance regression detector
pub struct RegressionDetector {
    baseline_results: Vec<BenchmarkResult>,
    threshold: f64,
}

impl RegressionDetector {
    pub fn new(baseline_results: Vec<BenchmarkResult>, threshold: f64) -> Self {
        Self {
            baseline_results,
            threshold,
        }
    }

    pub fn detect_regression(&self, current_result: &BenchmarkResult) -> bool {
        if let Some(baseline) = self.baseline_results.first() {
            let ratio =
                current_result.mean_time.as_nanos() as f64 / baseline.mean_time.as_nanos() as f64;
            ratio > (1.0 + self.threshold)
        } else {
            false
        }
    }

    pub fn performance_ratio(&self, current_result: &BenchmarkResult) -> Option<f64> {
        self.baseline_results.first().map(|baseline| {
            current_result.mean_time.as_nanos() as f64 / baseline.mean_time.as_nanos() as f64
        })
    }
}

/// Throughput measurement utilities
pub mod throughput {
    use super::*;

    pub fn measure_ops_per_second<F>(mut operation: F, duration: Duration) -> f64
    where
        F: FnMut(),
    {
        let start = Instant::now();
        let mut operations = 0;

        while start.elapsed() < duration {
            operation();
            operations += 1;
        }

        operations as f64 / duration.as_secs_f64()
    }

    pub fn measure_data_throughput<F>(operation: F, data_size: usize, duration: Duration) -> f64
    where
        F: FnMut(),
    {
        let ops_per_sec = measure_ops_per_second(operation, duration);
        ops_per_sec * data_size as f64
    }
}
