//! Advanced benchmarking suite for isotonic regression
//!
//! This module provides comprehensive benchmarking utilities for comparing
//! different isotonic regression algorithms and implementations.

use crate::fluent_api::FluentIsotonicRegression;
use crate::pav::{
    pool_adjacent_violators_huber, pool_adjacent_violators_l1, pool_adjacent_violators_l2,
};
use crate::unsafe_optimizations::pav_optimized;
use scirs2_core::ndarray::Array1;
use sklears_core::{error::Result, traits::Fit, types::Float};
use std::time::{Duration, Instant};

/// Benchmark result for a single algorithm
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    /// Algorithm name
    pub name: String,
    /// Total execution time
    pub total_time: Duration,
    /// Average time per run
    pub avg_time: Duration,
    /// Minimum time
    pub min_time: Duration,
    /// Maximum time
    pub max_time: Duration,
    /// Standard deviation
    pub std_dev: Duration,
    /// Number of iterations
    pub iterations: usize,
    /// Mean squared error (if ground truth available)
    pub mse: Option<Float>,
    /// Memory usage estimate (bytes)
    pub memory_usage: Option<usize>,
}

impl BenchmarkResult {
    /// Create a summary string
    pub fn summary(&self) -> String {
        format!(
            "{}: avg={:.2?}, min={:.2?}, max={:.2?}, std={:.2?}, iters={}{}",
            self.name,
            self.avg_time,
            self.min_time,
            self.max_time,
            self.std_dev,
            self.iterations,
            if let Some(mse) = self.mse {
                format!(", MSE={:.6}", mse)
            } else {
                String::new()
            }
        )
    }
}

/// Benchmark configuration
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    /// Number of iterations per benchmark
    pub iterations: usize,
    /// Warmup iterations (not counted)
    pub warmup: usize,
    /// Data sizes to benchmark
    pub data_sizes: Vec<usize>,
    /// Include unsafe optimizations
    pub include_unsafe: bool,
    /// Include all loss functions
    pub all_loss_functions: bool,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            iterations: 100,
            warmup: 10,
            data_sizes: vec![100, 500, 1000, 5000],
            include_unsafe: true,
            all_loss_functions: true,
        }
    }
}

/// Benchmark suite for isotonic regression
pub struct IsotonicBenchmarkSuite {
    config: BenchmarkConfig,
    results: Vec<BenchmarkResult>,
}

impl IsotonicBenchmarkSuite {
    /// Create new benchmark suite
    pub fn new(config: BenchmarkConfig) -> Self {
        Self {
            config,
            results: Vec::new(),
        }
    }

    /// Run all benchmarks
    pub fn run_all(&mut self) -> Result<()> {
        let data_sizes = self.config.data_sizes.clone();
        let all_loss = self.config.all_loss_functions;
        let include_unsafe = self.config.include_unsafe;

        for size in data_sizes {
            println!("\n=== Benchmarking with data size: {} ===", size);

            // Generate test data
            let (y, weights) = Self::generate_data(size);

            // Benchmark L2 PAV
            self.benchmark_pav_l2(&y, &weights)?;

            if all_loss {
                // Benchmark L1 PAV
                self.benchmark_pav_l1(&y, &weights)?;

                // Benchmark Huber PAV
                self.benchmark_pav_huber(&y, &weights)?;
            }

            if include_unsafe {
                // Benchmark unsafe optimized PAV
                self.benchmark_pav_unsafe(&y, &weights)?;
            }

            // Benchmark fluent API
            self.benchmark_fluent_api(&y, &weights)?;
        }

        Ok(())
    }

    fn generate_data(size: usize) -> (Array1<Float>, Array1<Float>) {
        // Generate random data with some noise
        let y: Vec<Float> = (0..size)
            .map(|i| {
                let trend = i as Float / size as Float * 100.0;
                let noise = ((i * 7919) % 100) as Float / 100.0 * 10.0;
                trend + noise
            })
            .collect();

        let weights = Array1::from_elem(size, 1.0);
        (Array1::from_vec(y), weights)
    }

    fn benchmark_pav_l2(&mut self, y: &Array1<Float>, weights: &Array1<Float>) -> Result<()> {
        let mut times = Vec::new();

        // Warmup
        for _ in 0..self.config.warmup {
            let _ = pool_adjacent_violators_l2(y, Some(weights), true)?;
        }

        // Benchmark
        for _ in 0..self.config.iterations {
            let start = Instant::now();
            let _ = pool_adjacent_violators_l2(y, Some(weights), true)?;
            times.push(start.elapsed());
        }

        let result = Self::compute_stats("PAV-L2", times, None);
        println!("{}", result.summary());
        self.results.push(result);

        Ok(())
    }

    fn benchmark_pav_l1(&mut self, y: &Array1<Float>, weights: &Array1<Float>) -> Result<()> {
        let mut times = Vec::new();

        // Warmup
        for _ in 0..self.config.warmup {
            let _ = pool_adjacent_violators_l1(y, Some(weights), true)?;
        }

        // Benchmark
        for _ in 0..self.config.iterations {
            let start = Instant::now();
            let _ = pool_adjacent_violators_l1(y, Some(weights), true)?;
            times.push(start.elapsed());
        }

        let result = Self::compute_stats("PAV-L1", times, None);
        println!("{}", result.summary());
        self.results.push(result);

        Ok(())
    }

    fn benchmark_pav_huber(&mut self, y: &Array1<Float>, weights: &Array1<Float>) -> Result<()> {
        let mut times = Vec::new();

        // Warmup
        for _ in 0..self.config.warmup {
            let _ = pool_adjacent_violators_huber(y, Some(weights), 1.35, true)?;
        }

        // Benchmark
        for _ in 0..self.config.iterations {
            let start = Instant::now();
            let _ = pool_adjacent_violators_huber(y, Some(weights), 1.35, true)?;
            times.push(start.elapsed());
        }

        let result = Self::compute_stats("PAV-Huber", times, None);
        println!("{}", result.summary());
        self.results.push(result);

        Ok(())
    }

    fn benchmark_pav_unsafe(&mut self, y: &Array1<Float>, weights: &Array1<Float>) -> Result<()> {
        let mut times = Vec::new();

        // Warmup
        for _ in 0..self.config.warmup {
            let _ = pav_optimized(y, weights, true);
        }

        // Benchmark
        for _ in 0..self.config.iterations {
            let start = Instant::now();
            let _ = pav_optimized(y, weights, true);
            times.push(start.elapsed());
        }

        let result = Self::compute_stats("PAV-Unsafe", times, None);
        println!("{}", result.summary());
        self.results.push(result);

        Ok(())
    }

    fn benchmark_fluent_api(&mut self, y: &Array1<Float>, _weights: &Array1<Float>) -> Result<()> {
        let mut times = Vec::new();
        let x = Array1::from_vec((0..y.len()).map(|i| i as Float).collect());

        // Warmup
        for _ in 0..self.config.warmup {
            let model = FluentIsotonicRegression::new().increasing();
            let _ = model.fit(&x, y)?;
        }

        // Benchmark
        for _ in 0..self.config.iterations {
            let start = Instant::now();
            let model = FluentIsotonicRegression::new().increasing();
            let _ = model.fit(&x, y)?;
            times.push(start.elapsed());
        }

        let result = Self::compute_stats("Fluent-API", times, None);
        println!("{}", result.summary());
        self.results.push(result);

        Ok(())
    }

    fn compute_stats(name: &str, times: Vec<Duration>, mse: Option<Float>) -> BenchmarkResult {
        let total_time: Duration = times.iter().sum();
        let iterations = times.len();
        let avg_time = total_time / iterations as u32;

        let min_time = times.iter().min().copied().unwrap_or(Duration::ZERO);
        let max_time = times.iter().max().copied().unwrap_or(Duration::ZERO);

        // Compute standard deviation
        let mean_nanos = avg_time.as_nanos() as f64;
        let variance: f64 = times
            .iter()
            .map(|t| {
                let diff = t.as_nanos() as f64 - mean_nanos;
                diff * diff
            })
            .sum::<f64>()
            / iterations as f64;
        let std_dev = Duration::from_nanos(variance.sqrt() as u64);

        BenchmarkResult {
            name: name.to_string(),
            total_time,
            avg_time,
            min_time,
            max_time,
            std_dev,
            iterations,
            mse,
            memory_usage: None,
        }
    }

    /// Get all benchmark results
    pub fn results(&self) -> &[BenchmarkResult] {
        &self.results
    }

    /// Print comparison summary
    pub fn print_summary(&self) {
        println!("\n=== Benchmark Summary ===");

        // Group by data size
        let mut size_groups: Vec<Vec<&BenchmarkResult>> = Vec::new();
        let mut current_group = Vec::new();

        for result in &self.results {
            if result.name.contains("PAV-L2") && !current_group.is_empty() {
                size_groups.push(current_group);
                current_group = Vec::new();
            }
            current_group.push(result);
        }
        if !current_group.is_empty() {
            size_groups.push(current_group);
        }

        for (idx, group) in size_groups.iter().enumerate() {
            if idx < self.config.data_sizes.len() {
                println!("\nData size: {}", self.config.data_sizes[idx]);
                println!("{:-<80}", "");

                // Find baseline (PAV-L2)
                let baseline = group
                    .iter()
                    .find(|r| r.name == "PAV-L2")
                    .map(|r| r.avg_time);

                for result in group {
                    let speedup = if let Some(base) = baseline {
                        format!(
                            " ({}x)",
                            base.as_nanos() as f64 / result.avg_time.as_nanos() as f64
                        )
                    } else {
                        String::new()
                    };

                    println!(
                        "  {:<15} {:>12.2?}{}",
                        result.name, result.avg_time, speedup
                    );
                }
            }
        }
    }

    /// Export results to CSV
    pub fn export_csv(&self, path: &str) -> std::io::Result<()> {
        use std::fs::File;
        use std::io::Write;

        let mut file = File::create(path)?;

        writeln!(
            file,
            "name,total_time_ns,avg_time_ns,min_time_ns,max_time_ns,std_dev_ns,iterations,mse"
        )?;

        for result in &self.results {
            writeln!(
                file,
                "{},{},{},{},{},{},{},{}",
                result.name,
                result.total_time.as_nanos(),
                result.avg_time.as_nanos(),
                result.min_time.as_nanos(),
                result.max_time.as_nanos(),
                result.std_dev.as_nanos(),
                result.iterations,
                result.mse.map(|m| m.to_string()).unwrap_or_default()
            )?;
        }

        Ok(())
    }
}

/// Quick benchmark function for convenience
pub fn quick_benchmark() -> Result<()> {
    let config = BenchmarkConfig {
        iterations: 50,
        warmup: 5,
        data_sizes: vec![100, 1000, 5000],
        include_unsafe: true,
        all_loss_functions: true,
    };

    let mut suite = IsotonicBenchmarkSuite::new(config);
    suite.run_all()?;
    suite.print_summary();

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benchmark_result_summary() {
        let result = BenchmarkResult {
            name: "Test".to_string(),
            total_time: Duration::from_millis(1000),
            avg_time: Duration::from_millis(10),
            min_time: Duration::from_millis(8),
            max_time: Duration::from_millis(15),
            std_dev: Duration::from_millis(2),
            iterations: 100,
            mse: Some(0.123),
            memory_usage: None,
        };

        let summary = result.summary();
        assert!(summary.contains("Test"));
        assert!(summary.contains("MSE=0.123"));
    }

    #[test]
    fn test_generate_data() {
        let (y, weights) = IsotonicBenchmarkSuite::generate_data(100);
        assert_eq!(y.len(), 100);
        assert_eq!(weights.len(), 100);
    }

    #[test]
    fn test_benchmark_config_default() {
        let config = BenchmarkConfig::default();
        assert_eq!(config.iterations, 100);
        assert_eq!(config.warmup, 10);
        assert!(config.include_unsafe);
        assert!(config.all_loss_functions);
    }

    #[test]
    fn test_benchmark_suite_creation() {
        let config = BenchmarkConfig::default();
        let suite = IsotonicBenchmarkSuite::new(config);
        assert_eq!(suite.results().len(), 0);
    }

    #[test]
    fn test_small_benchmark() {
        let config = BenchmarkConfig {
            iterations: 5,
            warmup: 1,
            data_sizes: vec![10],
            include_unsafe: true,
            all_loss_functions: false,
        };

        let mut suite = IsotonicBenchmarkSuite::new(config);
        assert!(suite.run_all().is_ok());
        assert!(!suite.results().is_empty());
    }
}
