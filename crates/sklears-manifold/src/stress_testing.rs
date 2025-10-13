//! Stress testing utilities for manifold learning algorithms
//!
//! This module provides utilities to test the scalability and performance of manifold
//! learning algorithms on large datasets with various characteristics:
//! - **Scale testing**: Performance on datasets of increasing size
//! - **Dimensionality testing**: Behavior with high-dimensional input data
//! - **Memory profiling**: Memory usage patterns during algorithm execution
//! - **Time complexity validation**: Empirical verification of theoretical complexity
//! - **Stress testing**: Robustness under extreme conditions

use scirs2_core::essentials::Normal;
use scirs2_core::ndarray::{Array2, ArrayView2};
use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::thread_rng;
use scirs2_core::random::Rng;
use scirs2_core::random::SeedableRng;
use scirs2_core::Distribution;
use sklears_core::types::Float;
use std::time::{Duration, Instant};

/// Dataset generation utilities for stress testing
pub mod synthetic_data {
    use super::*;

    /// Generate synthetic clustered data for testing
    pub fn generate_clustered_data(
        n_samples: usize,
        n_features: usize,
        n_clusters: usize,
        cluster_std: f64,
        seed: Option<u64>,
    ) -> Array2<Float> {
        let mut rng = if let Some(s) = seed {
            StdRng::seed_from_u64(s)
        } else {
            StdRng::seed_from_u64(thread_rng().gen::<u64>())
        };

        let mut data = Array2::zeros((n_samples, n_features));
        let samples_per_cluster = n_samples / n_clusters;

        for cluster_idx in 0..n_clusters {
            let start_idx = cluster_idx * samples_per_cluster;
            let end_idx = if cluster_idx == n_clusters - 1 {
                n_samples
            } else {
                (cluster_idx + 1) * samples_per_cluster
            };

            // Generate cluster center
            let mut center = vec![0.0; n_features];
            for j in 0..n_features {
                center[j] = rng.gen_range(-5.0..5.0);
            }

            // Generate samples around cluster center
            let normal = Normal::new(0.0, cluster_std).unwrap();
            for i in start_idx..end_idx {
                for j in 0..n_features {
                    data[(i, j)] = center[j] + normal.sample(&mut rng);
                }
            }
        }

        data
    }

    /// Generate synthetic manifold data (Swiss roll)
    pub fn generate_swiss_roll(n_samples: usize, noise: f64, seed: Option<u64>) -> Array2<Float> {
        let mut rng = if let Some(s) = seed {
            StdRng::seed_from_u64(s)
        } else {
            StdRng::seed_from_u64(thread_rng().gen::<u64>())
        };

        let mut data = Array2::zeros((n_samples, 3));
        let normal = Normal::new(0.0, noise).unwrap();

        for i in 0..n_samples {
            let t = 1.5 * std::f64::consts::PI * (1.0 + 2.0 * rng.gen::<f64>());
            let height = 21.0 * rng.gen::<f64>();

            data[(i, 0)] = t * t.cos() + normal.sample(&mut rng);
            data[(i, 1)] = height + normal.sample(&mut rng);
            data[(i, 2)] = t * t.sin() + normal.sample(&mut rng);
        }

        data
    }

    /// Generate high-dimensional sparse data
    pub fn generate_sparse_data(
        n_samples: usize,
        n_features: usize,
        sparsity: f64,
        seed: Option<u64>,
    ) -> Array2<Float> {
        let mut rng = if let Some(s) = seed {
            StdRng::seed_from_u64(s)
        } else {
            StdRng::seed_from_u64(thread_rng().gen::<u64>())
        };

        let mut data = Array2::zeros((n_samples, n_features));

        for i in 0..n_samples {
            for j in 0..n_features {
                if rng.gen::<f64>() > sparsity {
                    data[(i, j)] = rng.sample(scirs2_core::StandardNormal);
                }
            }
        }

        data
    }

    /// Generate data with known manifold structure (circle/sphere)
    pub fn generate_manifold_sphere(
        n_samples: usize,
        intrinsic_dim: usize,
        embedding_dim: usize,
        noise: f64,
        seed: Option<u64>,
    ) -> Array2<Float> {
        let mut rng = if let Some(s) = seed {
            StdRng::seed_from_u64(s)
        } else {
            StdRng::seed_from_u64(thread_rng().gen::<u64>())
        };

        let mut data = Array2::zeros((n_samples, embedding_dim));
        let normal = Normal::new(0.0, noise).unwrap();

        for i in 0..n_samples {
            // Generate point on unit sphere in intrinsic_dim space
            let mut sphere_point = vec![0.0; intrinsic_dim + 1];
            for j in 0..=intrinsic_dim {
                sphere_point[j] = rng.sample(scirs2_core::StandardNormal);
            }

            // Normalize to unit sphere
            let norm: f64 = sphere_point.iter().map(|x| x * x).sum::<f64>().sqrt();
            for j in 0..=intrinsic_dim {
                sphere_point[j] /= norm;
            }

            // Embed in higher dimensional space and add noise
            for j in 0..embedding_dim {
                if j <= intrinsic_dim {
                    data[(i, j)] = sphere_point[j] + normal.sample(&mut rng);
                } else {
                    data[(i, j)] = normal.sample(&mut rng);
                }
            }
        }

        data
    }
}

/// Performance measurement utilities
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    /// execution_time
    pub execution_time: Duration,
    /// memory_peak_mb
    pub memory_peak_mb: Option<f64>,
    /// n_samples
    pub n_samples: usize,
    /// n_features
    pub n_features: usize,
    /// n_components
    pub n_components: usize,
    /// algorithm_name
    pub algorithm_name: String,
    /// success
    pub success: bool,
    /// error_message
    pub error_message: Option<String>,
}

impl PerformanceMetrics {
    /// Calculate samples per second throughput
    pub fn throughput(&self) -> f64 {
        self.n_samples as f64 / self.execution_time.as_secs_f64()
    }

    /// Calculate time per sample in milliseconds
    pub fn time_per_sample_ms(&self) -> f64 {
        self.execution_time.as_secs_f64() * 1000.0 / self.n_samples as f64
    }
}

impl std::fmt::Display for PerformanceMetrics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Performance Metrics for {}", self.algorithm_name)?;
        writeln!(
            f,
            "========================{}",
            "=".repeat(self.algorithm_name.len())
        )?;
        writeln!(
            f,
            "Dataset: {} samples × {} features → {} components",
            self.n_samples, self.n_features, self.n_components
        )?;
        writeln!(
            f,
            "Execution time: {:.2}s",
            self.execution_time.as_secs_f64()
        )?;
        writeln!(f, "Throughput: {:.1} samples/sec", self.throughput())?;
        writeln!(f, "Time per sample: {:.2}ms", self.time_per_sample_ms())?;
        if let Some(mem) = self.memory_peak_mb {
            writeln!(f, "Peak memory: {:.1}MB", mem)?;
        }
        writeln!(f, "Success: {}", self.success)?;
        if let Some(ref err) = self.error_message {
            writeln!(f, "Error: {}", err)?;
        }
        Ok(())
    }
}

/// Scalability test suite for manifold algorithms
pub struct ScalabilityTester {
    max_execution_time: Duration,
    memory_limit_mb: Option<f64>,
    seed: Option<u64>,
}

impl Default for ScalabilityTester {
    fn default() -> Self {
        Self::new()
    }
}

impl ScalabilityTester {
    /// Create a new scalability tester
    pub fn new() -> Self {
        Self {
            max_execution_time: Duration::from_secs(300), // 5 minutes default
            memory_limit_mb: None,
            seed: Some(42),
        }
    }

    /// Set maximum allowed execution time
    pub fn max_execution_time(mut self, duration: Duration) -> Self {
        self.max_execution_time = duration;
        self
    }

    /// Set memory limit in megabytes
    pub fn memory_limit_mb(mut self, limit: f64) -> Self {
        self.memory_limit_mb = Some(limit);
        self
    }

    /// Set random seed for reproducibility
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Test algorithm performance across different dataset sizes
    pub fn scale_test<F>(
        &self,
        algorithm_name: &str,
        base_size: usize,
        max_size: usize,
        size_multiplier: f64,
        n_features: usize,
        n_components: usize,
        algorithm_fn: F,
    ) -> Vec<PerformanceMetrics>
    where
        F: Fn(&ArrayView2<Float>) -> Result<Array2<Float>, String>,
    {
        let mut results = Vec::new();
        let mut current_size = base_size;

        while current_size <= max_size {
            println!(
                "Testing {} with {} samples...",
                algorithm_name, current_size
            );

            // Generate test data
            let data = synthetic_data::generate_clustered_data(
                current_size,
                n_features,
                (current_size / 100).max(2),
                1.0,
                self.seed,
            );

            // Measure performance
            let metrics =
                self.measure_performance(algorithm_name, &data.view(), n_components, &algorithm_fn);

            let should_continue =
                metrics.success && metrics.execution_time <= self.max_execution_time;

            results.push(metrics);

            if !should_continue {
                println!("Stopping scale test due to timeout or failure");
                break;
            }

            current_size = (current_size as f64 * size_multiplier) as usize;
        }

        results
    }

    /// Test algorithm performance across different feature dimensions
    pub fn dimensionality_test<F>(
        &self,
        algorithm_name: &str,
        n_samples: usize,
        base_features: usize,
        max_features: usize,
        feature_multiplier: f64,
        n_components: usize,
        algorithm_fn: F,
    ) -> Vec<PerformanceMetrics>
    where
        F: Fn(&ArrayView2<Float>) -> Result<Array2<Float>, String>,
    {
        let mut results = Vec::new();
        let mut current_features = base_features;

        while current_features <= max_features {
            println!(
                "Testing {} with {} features...",
                algorithm_name, current_features
            );

            // Generate test data
            let data = synthetic_data::generate_manifold_sphere(
                n_samples,
                n_components.min(current_features - 1),
                current_features,
                0.1,
                self.seed,
            );

            // Measure performance
            let metrics =
                self.measure_performance(algorithm_name, &data.view(), n_components, &algorithm_fn);

            let should_continue =
                metrics.success && metrics.execution_time <= self.max_execution_time;

            results.push(metrics);

            if !should_continue {
                println!("Stopping dimensionality test due to timeout or failure");
                break;
            }

            current_features = (current_features as f64 * feature_multiplier) as usize;
        }

        results
    }

    /// Stress test with extreme conditions
    pub fn stress_test<F>(&self, algorithm_name: &str, algorithm_fn: F) -> Vec<PerformanceMetrics>
    where
        F: Fn(&ArrayView2<Float>) -> Result<Array2<Float>, String>,
    {
        let mut results = Vec::new();

        // Test cases with extreme conditions
        let test_cases = vec![
            ("Large dataset", 10000, 50, 2),
            ("High dimensionality", 1000, 500, 2),
            ("Extreme compression", 1000, 100, 1),
            ("Minimal samples", 10, 5, 2),
            ("Many components", 1000, 20, 15),
        ];

        for (test_name, n_samples, n_features, n_components) in test_cases {
            println!(
                "Stress test - {}: {} × {} → {}",
                test_name, n_samples, n_features, n_components
            );

            let data = synthetic_data::generate_clustered_data(
                n_samples,
                n_features,
                (n_samples / 100).max(2),
                1.0,
                self.seed,
            );

            let metrics = self.measure_performance(
                &format!("{} ({})", algorithm_name, test_name),
                &data.view(),
                n_components,
                &algorithm_fn,
            );

            results.push(metrics);
        }

        results
    }

    /// Measure performance of a single algorithm run
    fn measure_performance<F>(
        &self,
        algorithm_name: &str,
        data: &ArrayView2<Float>,
        n_components: usize,
        algorithm_fn: &F,
    ) -> PerformanceMetrics
    where
        F: Fn(&ArrayView2<Float>) -> Result<Array2<Float>, String>,
    {
        let start_time = Instant::now();

        let result = algorithm_fn(data);

        let execution_time = start_time.elapsed();

        match result {
            Ok(_embedding) => PerformanceMetrics {
                execution_time,
                memory_peak_mb: None, // TODO: Implement memory tracking
                n_samples: data.nrows(),
                n_features: data.ncols(),
                n_components,
                algorithm_name: algorithm_name.to_string(),
                success: true,
                error_message: None,
            },
            Err(error) => PerformanceMetrics {
                execution_time,
                memory_peak_mb: None,
                n_samples: data.nrows(),
                n_features: data.ncols(),
                n_components,
                algorithm_name: algorithm_name.to_string(),
                success: false,
                error_message: Some(error),
            },
        }
    }
}

/// Generate a comprehensive performance report
pub fn generate_performance_report(results: &[PerformanceMetrics]) -> String {
    let mut report = String::new();

    report.push_str("Manifold Learning Performance Report\n");
    report.push_str("===================================\n\n");

    if results.is_empty() {
        report.push_str("No performance data available.\n");
        return report;
    }

    // Summary statistics
    let successful_runs: Vec<_> = results.iter().filter(|r| r.success).collect();
    let failed_runs: Vec<_> = results.iter().filter(|r| !r.success).collect();

    report.push_str(&format!("Total runs: {}\n", results.len()));
    report.push_str(&format!("Successful: {}\n", successful_runs.len()));
    report.push_str(&format!("Failed: {}\n\n", failed_runs.len()));

    if !successful_runs.is_empty() {
        let total_time: Duration = successful_runs.iter().map(|r| r.execution_time).sum();
        let avg_throughput: f64 = successful_runs.iter().map(|r| r.throughput()).sum::<f64>()
            / successful_runs.len() as f64;
        let max_samples = successful_runs.iter().map(|r| r.n_samples).max().unwrap();
        let max_features = successful_runs.iter().map(|r| r.n_features).max().unwrap();

        report.push_str("Performance Summary:\n");
        report.push_str(&format!(
            "  Total execution time: {:.2}s\n",
            total_time.as_secs_f64()
        ));
        report.push_str(&format!(
            "  Average throughput: {:.1} samples/sec\n",
            avg_throughput
        ));
        report.push_str(&format!("  Largest dataset: {} samples\n", max_samples));
        report.push_str(&format!(
            "  Highest dimensionality: {} features\n\n",
            max_features
        ));
    }

    // Detailed results
    report.push_str("Detailed Results:\n");
    report.push_str("-----------------\n");
    for (i, result) in results.iter().enumerate() {
        report.push_str(&format!("Run {}: {}\n", i + 1, result));
    }

    report
}

/// Complexity analysis utilities
pub mod complexity_analysis {
    use super::PerformanceMetrics;

    /// Estimate time complexity from performance data
    pub fn estimate_time_complexity(results: &[PerformanceMetrics]) -> Option<String> {
        if results.len() < 3 {
            return None;
        }

        let successful: Vec<_> = results.iter().filter(|r| r.success).collect();
        if successful.len() < 3 {
            return None;
        }

        // Simple heuristic: compare time scaling with n and n^2
        let mut linear_fit = 0.0;
        let mut quadratic_fit = 0.0;
        let mut nlogn_fit = 0.0;

        for window in successful.windows(2) {
            let r1 = &window[0];
            let r2 = &window[1];

            if r1.n_samples == r2.n_samples {
                continue;
            }

            let n1 = r1.n_samples as f64;
            let n2 = r2.n_samples as f64;
            let t1 = r1.execution_time.as_secs_f64();
            let t2 = r2.execution_time.as_secs_f64();

            let actual_ratio = t2 / t1;
            let linear_ratio = n2 / n1;
            let quadratic_ratio = (n2 * n2) / (n1 * n1);
            let nlogn_ratio = (n2 * n2.ln()) / (n1 * n1.ln());

            linear_fit += (actual_ratio - linear_ratio).abs();
            quadratic_fit += (actual_ratio - quadratic_ratio).abs();
            nlogn_fit += (actual_ratio - nlogn_ratio).abs();
        }

        if linear_fit <= quadratic_fit && linear_fit <= nlogn_fit {
            Some("O(n)".to_string())
        } else if nlogn_fit <= quadratic_fit {
            Some("O(n log n)".to_string())
        } else {
            Some("O(n²)".to_string())
        }
    }

    /// Check if algorithm meets expected complexity
    pub fn verify_complexity(results: &[PerformanceMetrics], expected: &str) -> bool {
        if let Some(estimated) = estimate_time_complexity(results) {
            estimated == expected
        } else {
            false
        }
    }
}
