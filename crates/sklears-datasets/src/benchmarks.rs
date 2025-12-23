//! Performance benchmarking framework for dataset generation
//!
//! This module provides comprehensive benchmarking capabilities to measure
//! generation speed, memory usage, and scalability of dataset generators.

use std::collections::HashMap;
use std::fmt;
use std::time::{Duration, Instant};

/// Performance metrics for a single benchmark run
#[derive(Debug, Clone)]
pub struct BenchmarkMetrics {
    pub name: String,
    pub duration: Duration,
    pub samples_per_second: f64,
    pub memory_usage_mb: f64,
    pub n_samples: usize,
    pub n_features: usize,
    pub parallel_workers: Option<usize>,
    pub chunk_size: Option<usize>,
}

/// Comprehensive benchmark report
#[derive(Debug, Clone)]
pub struct BenchmarkReport {
    pub total_benchmarks: usize,
    pub metrics: Vec<BenchmarkMetrics>,
    pub fastest_generator: Option<String>,
    pub slowest_generator: Option<String>,
    pub avg_samples_per_second: f64,
    pub total_duration: Duration,
}

impl BenchmarkReport {
    /// Create a new benchmark report
    pub fn new() -> Self {
        Self {
            total_benchmarks: 0,
            metrics: Vec::new(),
            fastest_generator: None,
            slowest_generator: None,
            avg_samples_per_second: 0.0,
            total_duration: Duration::new(0, 0),
        }
    }

    /// Add benchmark metrics to the report
    pub fn add_metrics(&mut self, metrics: BenchmarkMetrics) {
        self.total_benchmarks += 1;
        self.total_duration += metrics.duration;

        // Update fastest and slowest generators
        if self.fastest_generator.is_none() || metrics.samples_per_second > self.get_fastest_speed()
        {
            self.fastest_generator = Some(metrics.name.clone());
        }

        if self.slowest_generator.is_none() || metrics.samples_per_second < self.get_slowest_speed()
        {
            self.slowest_generator = Some(metrics.name.clone());
        }

        self.metrics.push(metrics);
        self.update_averages();
    }

    fn get_fastest_speed(&self) -> f64 {
        self.metrics
            .iter()
            .filter(|m| Some(&m.name) == self.fastest_generator.as_ref())
            .map(|m| m.samples_per_second)
            .next()
            .unwrap_or(0.0)
    }

    fn get_slowest_speed(&self) -> f64 {
        self.metrics
            .iter()
            .filter(|m| Some(&m.name) == self.slowest_generator.as_ref())
            .map(|m| m.samples_per_second)
            .next()
            .unwrap_or(f64::INFINITY)
    }

    fn update_averages(&mut self) {
        if self.total_benchmarks > 0 {
            self.avg_samples_per_second = self
                .metrics
                .iter()
                .map(|m| m.samples_per_second)
                .sum::<f64>()
                / self.total_benchmarks as f64;
        }
    }

    /// Get performance ranking of generators
    pub fn get_performance_ranking(&self) -> Vec<(&str, f64)> {
        let mut ranking: Vec<(&str, f64)> = self
            .metrics
            .iter()
            .map(|m| (m.name.as_str(), m.samples_per_second))
            .collect();

        ranking.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        ranking
    }

    /// Get scalability analysis
    pub fn get_scalability_analysis(&self) -> HashMap<String, Vec<(usize, f64)>> {
        let mut scalability = HashMap::new();

        for metric in &self.metrics {
            let entry = scalability
                .entry(metric.name.clone())
                .or_insert_with(Vec::new);
            entry.push((metric.n_samples, metric.samples_per_second));
        }

        // Sort by sample size for each generator
        for (_, samples) in scalability.iter_mut() {
            samples.sort_by(|a, b| a.0.cmp(&b.0));
        }

        scalability
    }
}

impl fmt::Display for BenchmarkReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "=== Performance Benchmark Report ===")?;
        writeln!(f, "Total benchmarks: {}", self.total_benchmarks)?;
        writeln!(f, "Total duration: {:.2?}", self.total_duration)?;
        writeln!(f, "Average samples/sec: {:.2}", self.avg_samples_per_second)?;
        writeln!(f, "")?;

        if let Some(fastest) = &self.fastest_generator {
            writeln!(f, "Fastest generator: {}", fastest)?;
        }

        if let Some(slowest) = &self.slowest_generator {
            writeln!(f, "Slowest generator: {}", slowest)?;
        }

        writeln!(f, "")?;
        writeln!(f, "Performance Rankings:")?;

        for (i, (name, speed)) in self.get_performance_ranking().iter().enumerate() {
            writeln!(f, "{}. {} - {:.2} samples/sec", i + 1, name, speed)?;
        }

        writeln!(f, "")?;
        writeln!(f, "Detailed Metrics:")?;

        for metric in &self.metrics {
            writeln!(
                f,
                "{}: {:.2} samples/sec, {:.2?}, {:.2} MB",
                metric.name, metric.samples_per_second, metric.duration, metric.memory_usage_mb
            )?;
        }

        Ok(())
    }
}

/// Configuration for benchmarking
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    pub sample_sizes: Vec<usize>,
    pub feature_sizes: Vec<usize>,
    pub num_runs: usize,
    pub warmup_runs: usize,
    pub measure_memory: bool,
    pub parallel_workers: Vec<Option<usize>>,
    pub chunk_sizes: Vec<Option<usize>>,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            sample_sizes: vec![1000, 5000, 10000],
            feature_sizes: vec![10, 50, 100],
            num_runs: 3,
            warmup_runs: 1,
            measure_memory: true,
            parallel_workers: vec![None, Some(2), Some(4)],
            chunk_sizes: vec![None, Some(1000), Some(5000)],
        }
    }
}

/// Memory usage estimation (simplified)
fn estimate_memory_usage(n_samples: usize, n_features: usize) -> f64 {
    // Rough estimate: 8 bytes per f64 value + overhead
    (n_samples * n_features * 8) as f64 / 1024.0 / 1024.0
}

/// Benchmark a generator function
pub fn benchmark_generator<F>(
    name: &str,
    generator_fn: F,
    n_samples: usize,
    n_features: usize,
    config: &BenchmarkConfig,
) -> BenchmarkMetrics
where
    F: Fn(usize, usize) -> Result<Vec<Vec<f64>>, Box<dyn std::error::Error>>,
{
    let mut durations = Vec::new();

    // Warmup runs
    for _ in 0..config.warmup_runs {
        let _ = generator_fn(n_samples, n_features);
    }

    // Actual benchmark runs
    for _ in 0..config.num_runs {
        let start = Instant::now();
        let result = generator_fn(n_samples, n_features);
        let duration = start.elapsed();

        if result.is_ok() {
            durations.push(duration);
        }
    }

    let avg_duration = if durations.is_empty() {
        Duration::new(0, 0)
    } else {
        durations.iter().sum::<Duration>() / durations.len() as u32
    };

    let samples_per_second = if avg_duration.as_secs_f64() > 0.0 {
        n_samples as f64 / avg_duration.as_secs_f64()
    } else {
        0.0
    };

    let memory_usage = if config.measure_memory {
        estimate_memory_usage(n_samples, n_features)
    } else {
        0.0
    };

    BenchmarkMetrics {
        name: name.to_string(),
        duration: avg_duration,
        samples_per_second,
        memory_usage_mb: memory_usage,
        n_samples,
        n_features,
        parallel_workers: None,
        chunk_size: None,
    }
}

/// Benchmark parallel generator function
pub fn benchmark_parallel_generator<F>(
    name: &str,
    generator_fn: F,
    n_samples: usize,
    n_features: usize,
    workers: usize,
    config: &BenchmarkConfig,
) -> BenchmarkMetrics
where
    F: Fn(usize, usize, usize) -> Result<Vec<Vec<f64>>, Box<dyn std::error::Error>>,
{
    let mut durations = Vec::new();

    // Warmup runs
    for _ in 0..config.warmup_runs {
        let _ = generator_fn(n_samples, n_features, workers);
    }

    // Actual benchmark runs
    for _ in 0..config.num_runs {
        let start = Instant::now();
        let result = generator_fn(n_samples, n_features, workers);
        let duration = start.elapsed();

        if result.is_ok() {
            durations.push(duration);
        }
    }

    let avg_duration = if durations.is_empty() {
        Duration::new(0, 0)
    } else {
        durations.iter().sum::<Duration>() / durations.len() as u32
    };

    let samples_per_second = if avg_duration.as_secs_f64() > 0.0 {
        n_samples as f64 / avg_duration.as_secs_f64()
    } else {
        0.0
    };

    let memory_usage = if config.measure_memory {
        estimate_memory_usage(n_samples, n_features)
    } else {
        0.0
    };

    BenchmarkMetrics {
        name: format!("{}_parallel_{}", name, workers),
        duration: avg_duration,
        samples_per_second,
        memory_usage_mb: memory_usage,
        n_samples,
        n_features,
        parallel_workers: Some(workers),
        chunk_size: None,
    }
}

/// Comprehensive benchmark suite for dataset generators
pub fn run_comprehensive_benchmarks() -> BenchmarkReport {
    let config = BenchmarkConfig::default();
    let mut report = BenchmarkReport::new();

    // Import necessary functions
    use crate::generators_legacy::{make_blobs, make_classification, make_regression};

    // Benchmark different generators with various configurations
    for &n_samples in &config.sample_sizes {
        for &n_features in &config.feature_sizes {
            // Benchmark make_classification
            let classification_metrics = benchmark_generator(
                "make_classification",
                |n_samples, n_features| {
                    make_classification(n_samples, n_features, n_features / 2, 0, 2, None)
                        .map(|(x, _y)| {
                            // Convert ndarray to Vec<Vec<f64>>
                            (0..x.nrows())
                                .map(|i| (0..x.ncols()).map(|j| x[[i, j]]).collect())
                                .collect()
                        })
                        .map_err(|e| e.into())
                },
                n_samples,
                n_features,
                &config,
            );
            report.add_metrics(classification_metrics);

            // Benchmark make_regression
            let regression_metrics = benchmark_generator(
                "make_regression",
                |n_samples, n_features| {
                    make_regression(n_samples, n_features, n_features / 2, 0.1, None)
                        .map(|(x, _y)| {
                            // Convert ndarray to Vec<Vec<f64>>
                            (0..x.nrows())
                                .map(|i| (0..x.ncols()).map(|j| x[[i, j]]).collect())
                                .collect()
                        })
                        .map_err(|e| e.into())
                },
                n_samples,
                n_features,
                &config,
            );
            report.add_metrics(regression_metrics);

            // Benchmark make_blobs
            let blobs_metrics = benchmark_generator(
                "make_blobs",
                |n_samples, n_features| {
                    make_blobs(n_samples, n_features, 3, 1.0, None)
                        .map(|(x, _y)| {
                            // Convert ndarray to Vec<Vec<f64>>
                            (0..x.nrows())
                                .map(|i| (0..x.ncols()).map(|j| x[[i, j]]).collect())
                                .collect()
                        })
                        .map_err(|e| e.into())
                },
                n_samples,
                n_features,
                &config,
            );
            report.add_metrics(blobs_metrics);
        }
    }

    report
}

/// Benchmark memory usage scalability
pub fn benchmark_memory_scalability() -> HashMap<String, Vec<(usize, f64)>> {
    let mut scalability = HashMap::new();

    // Test different sample sizes
    let sample_sizes = vec![1000, 5000, 10000, 50000, 100000];
    let n_features = 10;

    for &n_samples in &sample_sizes {
        let estimated_memory = estimate_memory_usage(n_samples, n_features);

        scalability
            .entry("memory_usage".to_string())
            .or_insert_with(Vec::new)
            .push((n_samples, estimated_memory));
    }

    scalability
}

/// Run streaming performance benchmarks
pub fn benchmark_streaming_performance() -> BenchmarkReport {
    let mut report = BenchmarkReport::new();

    // Test different chunk sizes
    let chunk_sizes = vec![1000, 5000, 10000];
    let total_samples = 50000;
    let n_features = 10;

    for &chunk_size in &chunk_sizes {
        let start = Instant::now();

        // Simulate streaming by generating data in chunks
        let mut total_generated = 0;
        while total_generated < total_samples {
            let current_chunk = std::cmp::min(chunk_size, total_samples - total_generated);

            // Generate a chunk (using a simple generator)
            let _chunk: Vec<Vec<f64>> = (0..current_chunk)
                .map(|_| (0..n_features).map(|_| scirs2_core::random::thread_rng().random_range(0.0, 1.0)).collect())
                .collect();

            total_generated += current_chunk;
        }

        let duration = start.elapsed();
        let samples_per_second = total_samples as f64 / duration.as_secs_f64();

        let metrics = BenchmarkMetrics {
            name: format!("streaming_chunk_{}", chunk_size),
            duration,
            samples_per_second,
            memory_usage_mb: estimate_memory_usage(chunk_size, n_features),
            n_samples: total_samples,
            n_features,
            parallel_workers: None,
            chunk_size: Some(chunk_size),
        };

        report.add_metrics(metrics);
    }

    report
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benchmark_metrics() {
        let metrics = BenchmarkMetrics {
            name: "test_generator".to_string(),
            duration: Duration::from_millis(100),
            samples_per_second: 10000.0,
            memory_usage_mb: 1.5,
            n_samples: 1000,
            n_features: 10,
            parallel_workers: None,
            chunk_size: None,
        };

        assert_eq!(metrics.name, "test_generator");
        assert_eq!(metrics.samples_per_second, 10000.0);
    }

    #[test]
    fn test_benchmark_report() {
        let mut report = BenchmarkReport::new();

        let metrics1 = BenchmarkMetrics {
            name: "fast_generator".to_string(),
            duration: Duration::from_millis(50),
            samples_per_second: 20000.0,
            memory_usage_mb: 1.0,
            n_samples: 1000,
            n_features: 10,
            parallel_workers: None,
            chunk_size: None,
        };

        let metrics2 = BenchmarkMetrics {
            name: "slow_generator".to_string(),
            duration: Duration::from_millis(200),
            samples_per_second: 5000.0,
            memory_usage_mb: 2.0,
            n_samples: 1000,
            n_features: 10,
            parallel_workers: None,
            chunk_size: None,
        };

        report.add_metrics(metrics1);
        report.add_metrics(metrics2);

        assert_eq!(report.total_benchmarks, 2);
        assert_eq!(report.fastest_generator, Some("fast_generator".to_string()));
        assert_eq!(report.slowest_generator, Some("slow_generator".to_string()));
    }

    #[test]
    fn test_memory_estimation() {
        let memory_mb = estimate_memory_usage(1000, 10);
        assert!(memory_mb > 0.0);

        // Should scale with sample size
        let memory_mb_2x = estimate_memory_usage(2000, 10);
        assert!(memory_mb_2x > memory_mb);
    }

    #[test]
    fn test_benchmark_simple_generator() {
        let config = BenchmarkConfig {
            num_runs: 1,
            warmup_runs: 0,
            ..BenchmarkConfig::default()
        };

        let metrics = benchmark_generator(
            "simple_generator",
            |n_samples, n_features| {
                Ok((0..n_samples)
                    .map(|_| (0..n_features).map(|_| 1.0).collect())
                    .collect())
            },
            100,
            5,
            &config,
        );

        assert_eq!(metrics.name, "simple_generator");
        assert_eq!(metrics.n_samples, 100);
        assert_eq!(metrics.n_features, 5);
        assert!(metrics.samples_per_second > 0.0);
    }

    #[test]
    fn test_performance_ranking() {
        let mut report = BenchmarkReport::new();

        let metrics1 = BenchmarkMetrics {
            name: "fast".to_string(),
            duration: Duration::from_millis(10),
            samples_per_second: 100000.0,
            memory_usage_mb: 1.0,
            n_samples: 1000,
            n_features: 10,
            parallel_workers: None,
            chunk_size: None,
        };

        let metrics2 = BenchmarkMetrics {
            name: "slow".to_string(),
            duration: Duration::from_millis(100),
            samples_per_second: 10000.0,
            memory_usage_mb: 1.0,
            n_samples: 1000,
            n_features: 10,
            parallel_workers: None,
            chunk_size: None,
        };

        report.add_metrics(metrics1);
        report.add_metrics(metrics2);

        let ranking = report.get_performance_ranking();
        assert_eq!(ranking[0].0, "fast");
        assert_eq!(ranking[1].0, "slow");
    }
}
