//! Comprehensive Benchmarking Framework for Kernel Approximation Methods
//!
//! This module provides tools for benchmarking and comparing different kernel approximation
//! methods in terms of approximation quality, computational performance, and memory usage.

use scirs2_core::ndarray::{Array1, Array2, Axis};
use scirs2_core::random::essentials::Uniform as RandUniform;
use scirs2_core::random::rngs::StdRng as RealStdRng;
use scirs2_core::random::Rng;
use scirs2_core::random::SeedableRng;
use sklears_core::prelude::Result;
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Benchmarking suite for kernel approximation methods
#[derive(Debug, Clone)]
/// KernelApproximationBenchmark
pub struct KernelApproximationBenchmark {
    config: BenchmarkConfig,
    datasets: Vec<BenchmarkDataset>,
    results: Vec<BenchmarkResult>,
}

/// Configuration for benchmarking
#[derive(Debug, Clone)]
/// BenchmarkConfig
pub struct BenchmarkConfig {
    /// repetitions
    pub repetitions: usize,
    /// quality_metrics
    pub quality_metrics: Vec<QualityMetric>,
    /// performance_metrics
    pub performance_metrics: Vec<PerformanceMetric>,
    /// approximation_dimensions
    pub approximation_dimensions: Vec<usize>,
    /// random_state
    pub random_state: Option<u64>,
    /// warmup_iterations
    pub warmup_iterations: usize,
    /// timeout_seconds
    pub timeout_seconds: Option<u64>,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            repetitions: 5,
            quality_metrics: vec![
                QualityMetric::KernelAlignment,
                QualityMetric::FrobeniusError,
                QualityMetric::SpectralError,
                QualityMetric::RelativeError,
            ],
            performance_metrics: vec![
                PerformanceMetric::FitTime,
                PerformanceMetric::TransformTime,
                PerformanceMetric::MemoryUsage,
                PerformanceMetric::Throughput,
            ],
            approximation_dimensions: vec![50, 100, 200, 500, 1000],
            random_state: Some(42),
            warmup_iterations: 3,
            timeout_seconds: Some(300),
        }
    }
}

/// Quality metrics for evaluating approximation quality
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
/// QualityMetric
pub enum QualityMetric {
    /// Centered kernel alignment
    KernelAlignment,
    /// Frobenius norm error
    FrobeniusError,
    /// Spectral norm error
    SpectralError,
    /// Relative approximation error
    RelativeError,
    /// Nuclear norm error
    NuclearError,
    /// Effective rank comparison
    EffectiveRank,
    /// Eigenvalue preservation
    EigenvalueError,
}

/// Performance metrics for computational evaluation
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
/// PerformanceMetric
pub enum PerformanceMetric {
    /// Time to fit the approximation
    FitTime,
    /// Time to transform data
    TransformTime,
    /// Peak memory usage
    MemoryUsage,
    /// Throughput (samples per second)
    Throughput,
    /// Total computational cost
    TotalTime,
    /// Memory efficiency (quality per MB)
    MemoryEfficiency,
}

/// Benchmark dataset specification
#[derive(Debug, Clone)]
/// BenchmarkDataset
pub struct BenchmarkDataset {
    /// name
    pub name: String,
    /// data
    pub data: Array2<f64>,
    /// target
    pub target: Option<Array1<f64>>,
    /// true_kernel
    pub true_kernel: Option<Array2<f64>>,
    /// description
    pub description: String,
}

impl BenchmarkDataset {
    /// Create a synthetic Gaussian dataset
    pub fn gaussian(n_samples: usize, n_features: usize, noise: f64, seed: u64) -> Self {
        let mut rng = RealStdRng::seed_from_u64(seed);

        // Helper function to generate normal distribution samples using Box-Muller
        let normal_sample = |rng: &mut RealStdRng, mean: f64, std_dev: f64| -> f64 {
            let u1: f64 = rng.gen();
            let u2: f64 = rng.gen();
            let z = (-2.0_f64 * u1.ln()).sqrt() * (2.0_f64 * std::f64::consts::PI * u2).cos();
            mean + z * std_dev
        };

        let mut data = Array2::zeros((n_samples, n_features));
        for i in 0..n_samples {
            for j in 0..n_features {
                data[[i, j]] =
                    normal_sample(&mut rng, 0.0, 1.0) + normal_sample(&mut rng, 0.0, noise);
            }
        }

        // Generate a simple target based on linear combination
        let mut target = Array1::zeros(n_samples);
        let weights = Array1::from_shape_fn(n_features, |_| normal_sample(&mut rng, 0.0, 1.0));

        for (i, sample) in data.axis_iter(Axis(0)).enumerate() {
            target[i] = sample.dot(&weights) + normal_sample(&mut rng, 0.0, noise);
        }

        Self {
            name: format!("Gaussian_{}x{}_noise{}", n_samples, n_features, noise),
            data,
            target: Some(target),
            true_kernel: None,
            description: format!(
                "Gaussian dataset with {} samples, {} features, and noise level {}",
                n_samples, n_features, noise
            ),
        }
    }

    /// Create a polynomial dataset
    pub fn polynomial(n_samples: usize, n_features: usize, degree: usize, seed: u64) -> Self {
        let mut rng = RealStdRng::seed_from_u64(seed);
        let uniform = RandUniform::new(-1.0, 1.0).unwrap();

        let mut data = Array2::zeros((n_samples, n_features));
        for i in 0..n_samples {
            for j in 0..n_features {
                data[[i, j]] = rng.sample(uniform);
            }
        }

        // Generate polynomial target
        let mut target = Array1::zeros(n_samples);
        for (i, sample) in data.axis_iter(Axis(0)).enumerate() {
            let mut poly_value = 0.0;
            for &x in sample.iter() {
                let x: f64 = x;
                poly_value += x.powi(degree as i32);
            }
            target[i] = poly_value / n_features as f64;
        }

        Self {
            name: format!("Polynomial_{}x{}_deg{}", n_samples, n_features, degree),
            data,
            target: Some(target),
            true_kernel: None,
            description: format!(
                "Polynomial dataset with {} samples, {} features, and degree {}",
                n_samples, n_features, degree
            ),
        }
    }

    /// Create a classification dataset with distinct clusters
    pub fn classification_clusters(
        n_samples: usize,
        n_features: usize,
        n_clusters: usize,
        seed: u64,
    ) -> Self {
        let mut rng = RealStdRng::seed_from_u64(seed);

        // Helper function to generate normal distribution samples using Box-Muller
        let normal_sample = |rng: &mut RealStdRng, mean: f64, std_dev: f64| -> f64 {
            let u1: f64 = rng.gen();
            let u2: f64 = rng.gen();
            let z = (-2.0_f64 * u1.ln()).sqrt() * (2.0_f64 * std::f64::consts::PI * u2).cos();
            mean + z * std_dev
        };

        let samples_per_cluster = n_samples / n_clusters;
        let mut data = Array2::zeros((n_samples, n_features));
        let mut target = Array1::zeros(n_samples);

        // Generate cluster centers
        let cluster_separation = 3.0;
        let mut centers = Array2::zeros((n_clusters, n_features));
        for i in 0..n_clusters {
            for j in 0..n_features {
                centers[[i, j]] =
                    (i as f64 * cluster_separation) * if j % 2 == 0 { 1.0 } else { -1.0 };
            }
        }

        // Generate samples around centers
        let mut sample_idx = 0;
        for cluster in 0..n_clusters {
            for _ in 0..samples_per_cluster {
                if sample_idx >= n_samples {
                    break;
                }

                for j in 0..n_features {
                    data[[sample_idx, j]] =
                        centers[[cluster, j]] + normal_sample(&mut rng, 0.0, 0.5);
                }
                target[sample_idx] = cluster as f64;
                sample_idx += 1;
            }
        }

        Self {
            name: format!("Clusters_{}x{}_k{}", n_samples, n_features, n_clusters),
            data,
            target: Some(target),
            true_kernel: None,
            description: format!(
                "Classification dataset with {} samples, {} features, and {} clusters",
                n_samples, n_features, n_clusters
            ),
        }
    }
}

/// Result of a benchmark run
#[derive(Debug, Clone)]
/// BenchmarkResult
pub struct BenchmarkResult {
    /// method_name
    pub method_name: String,
    /// dataset_name
    pub dataset_name: String,
    /// approximation_dimension
    pub approximation_dimension: usize,
    /// quality_scores
    pub quality_scores: HashMap<QualityMetric, f64>,
    /// performance_scores
    pub performance_scores: HashMap<PerformanceMetric, Duration>,
    /// memory_usage_mb
    pub memory_usage_mb: f64,
    /// success
    pub success: bool,
    /// error_message
    pub error_message: Option<String>,
    /// repetition
    pub repetition: usize,
}

/// Summary statistics across multiple benchmark runs
#[derive(Debug, Clone)]
/// BenchmarkSummary
pub struct BenchmarkSummary {
    /// method_name
    pub method_name: String,
    /// dataset_name
    pub dataset_name: String,
    /// approximation_dimension
    pub approximation_dimension: usize,
    /// quality_means
    pub quality_means: HashMap<QualityMetric, f64>,
    /// quality_stds
    pub quality_stds: HashMap<QualityMetric, f64>,
    /// performance_means
    pub performance_means: HashMap<PerformanceMetric, Duration>,
    /// performance_stds
    pub performance_stds: HashMap<PerformanceMetric, Duration>,
    /// success_rate
    pub success_rate: f64,
    /// memory_usage_mean_mb
    pub memory_usage_mean_mb: f64,
    /// memory_usage_std_mb
    pub memory_usage_std_mb: f64,
}

/// Trait for benchmarkable kernel approximation methods
pub trait BenchmarkableKernelMethod {
    /// Get the name of the method
    fn method_name(&self) -> String;

    /// Fit the approximation method and return timing
    fn benchmark_fit(
        &self,
        data: &Array2<f64>,
        target: Option<&Array1<f64>>,
    ) -> Result<(Box<dyn BenchmarkableTransformer>, Duration)>;

    /// Clone the method for multiple repetitions
    fn clone_method(&self) -> Box<dyn BenchmarkableKernelMethod>;
}

/// Trait for fitted kernel approximation methods that can transform data
pub trait BenchmarkableTransformer {
    /// Transform data and return timing
    fn benchmark_transform(&self, data: &Array2<f64>) -> Result<(Array2<f64>, Duration)>;

    /// Get approximation dimension
    fn approximation_dimension(&self) -> usize;

    /// Estimate memory usage in MB
    fn memory_usage_mb(&self) -> f64;
}

impl KernelApproximationBenchmark {
    /// Create a new benchmark suite
    pub fn new(config: BenchmarkConfig) -> Self {
        Self {
            config,
            datasets: Vec::new(),
            results: Vec::new(),
        }
    }

    /// Add a dataset to benchmark
    pub fn add_dataset(&mut self, dataset: BenchmarkDataset) {
        self.datasets.push(dataset);
    }

    /// Add multiple synthetic datasets
    pub fn add_synthetic_datasets(&mut self, seed: u64) {
        // Small datasets for quick testing
        self.add_dataset(BenchmarkDataset::gaussian(100, 10, 0.1, seed));
        self.add_dataset(BenchmarkDataset::polynomial(100, 5, 2, seed + 1));
        self.add_dataset(BenchmarkDataset::classification_clusters(
            120,
            8,
            3,
            seed + 2,
        ));

        // Medium datasets
        self.add_dataset(BenchmarkDataset::gaussian(500, 20, 0.2, seed + 3));
        self.add_dataset(BenchmarkDataset::polynomial(500, 15, 3, seed + 4));
        self.add_dataset(BenchmarkDataset::classification_clusters(
            600,
            25,
            4,
            seed + 5,
        ));

        // Large datasets
        self.add_dataset(BenchmarkDataset::gaussian(1000, 50, 0.1, seed + 6));
        self.add_dataset(BenchmarkDataset::polynomial(1000, 30, 2, seed + 7));
        self.add_dataset(BenchmarkDataset::classification_clusters(
            1200,
            40,
            5,
            seed + 8,
        ));
    }

    /// Run benchmark for a specific method
    pub fn benchmark_method(
        &mut self,
        method: &dyn BenchmarkableKernelMethod,
    ) -> Result<Vec<BenchmarkResult>> {
        let mut method_results = Vec::new();
        let _method_name = method.method_name();

        for dataset in &self.datasets {
            for &n_components in &self.config.approximation_dimensions {
                for repetition in 0..self.config.repetitions {
                    let result =
                        self.run_single_benchmark(method, dataset, n_components, repetition)?;

                    method_results.push(result.clone());
                    self.results.push(result);
                }
            }
        }

        Ok(method_results)
    }

    fn run_single_benchmark(
        &self,
        method: &dyn BenchmarkableKernelMethod,
        dataset: &BenchmarkDataset,
        n_components: usize,
        repetition: usize,
    ) -> Result<BenchmarkResult> {
        let start_time = Instant::now();

        // Check timeout
        if let Some(timeout) = self.config.timeout_seconds {
            if start_time.elapsed().as_secs() > timeout {
                return Ok(BenchmarkResult {
                    method_name: method.method_name(),
                    dataset_name: dataset.name.clone(),
                    approximation_dimension: n_components,
                    quality_scores: HashMap::new(),
                    performance_scores: HashMap::new(),
                    memory_usage_mb: 0.0,
                    success: false,
                    error_message: Some("Timeout exceeded".to_string()),
                    repetition,
                });
            }
        }

        // Warmup iterations
        for _ in 0..self.config.warmup_iterations {
            let _ = method.benchmark_fit(&dataset.data, dataset.target.as_ref());
        }

        // Actual benchmark run
        let _fit_start = Instant::now();
        let (fitted_method, fit_time) =
            match method.benchmark_fit(&dataset.data, dataset.target.as_ref()) {
                Ok(result) => result,
                Err(e) => {
                    return Ok(BenchmarkResult {
                        method_name: method.method_name(),
                        dataset_name: dataset.name.clone(),
                        approximation_dimension: n_components,
                        quality_scores: HashMap::new(),
                        performance_scores: HashMap::new(),
                        memory_usage_mb: 0.0,
                        success: false,
                        error_message: Some(format!("Fit error: {}", e)),
                        repetition,
                    });
                }
            };

        let (transformed_data, transform_time) =
            match fitted_method.benchmark_transform(&dataset.data) {
                Ok(result) => result,
                Err(e) => {
                    return Ok(BenchmarkResult {
                        method_name: method.method_name(),
                        dataset_name: dataset.name.clone(),
                        approximation_dimension: n_components,
                        quality_scores: HashMap::new(),
                        performance_scores: HashMap::new(),
                        memory_usage_mb: 0.0,
                        success: false,
                        error_message: Some(format!("Transform error: {}", e)),
                        repetition,
                    });
                }
            };

        let memory_usage = fitted_method.memory_usage_mb();
        let total_time = fit_time + transform_time;
        let throughput_time =
            Duration::from_secs_f64(dataset.data.nrows() as f64 / transform_time.as_secs_f64());

        // Compute quality metrics
        let mut quality_scores = HashMap::new();

        if let Some(ref true_kernel) = dataset.true_kernel {
            // Use provided true kernel for comparison
            let approx_kernel = self.compute_approximate_kernel(&transformed_data)?;

            for metric in &self.config.quality_metrics {
                let score = self.compute_quality_metric(metric, true_kernel, &approx_kernel)?;
                quality_scores.insert(metric.clone(), score);
            }
        } else {
            // Compute approximate quality metrics without true kernel
            for metric in &self.config.quality_metrics {
                let score = self.compute_approximate_quality_metric(
                    metric,
                    &dataset.data,
                    &transformed_data,
                )?;
                quality_scores.insert(metric.clone(), score);
            }
        }

        // Collect performance metrics
        let mut performance_scores = HashMap::new();
        performance_scores.insert(PerformanceMetric::FitTime, fit_time);
        performance_scores.insert(PerformanceMetric::TransformTime, transform_time);
        performance_scores.insert(PerformanceMetric::TotalTime, total_time);
        performance_scores.insert(PerformanceMetric::Throughput, throughput_time);
        performance_scores.insert(
            PerformanceMetric::MemoryUsage,
            Duration::from_secs_f64(memory_usage),
        );
        performance_scores.insert(
            PerformanceMetric::MemoryEfficiency,
            Duration::from_secs_f64(
                quality_scores
                    .get(&QualityMetric::KernelAlignment)
                    .unwrap_or(&0.0)
                    / memory_usage.max(1.0),
            ),
        );

        Ok(BenchmarkResult {
            method_name: method.method_name(),
            dataset_name: dataset.name.clone(),
            approximation_dimension: n_components,
            quality_scores,
            performance_scores,
            memory_usage_mb: memory_usage,
            success: true,
            error_message: None,
            repetition,
        })
    }

    fn compute_approximate_kernel(&self, features: &Array2<f64>) -> Result<Array2<f64>> {
        let n_samples = features.nrows();
        let mut kernel = Array2::zeros((n_samples, n_samples));

        for i in 0..n_samples {
            for j in i..n_samples {
                let similarity = features.row(i).dot(&features.row(j));
                kernel[[i, j]] = similarity;
                kernel[[j, i]] = similarity;
            }
        }

        Ok(kernel)
    }

    fn compute_quality_metric(
        &self,
        metric: &QualityMetric,
        true_kernel: &Array2<f64>,
        approx_kernel: &Array2<f64>,
    ) -> Result<f64> {
        match metric {
            QualityMetric::KernelAlignment => {
                Ok(self.compute_kernel_alignment(true_kernel, approx_kernel)?)
            }
            QualityMetric::FrobeniusError => {
                let diff = true_kernel - approx_kernel;
                let frobenius_norm = diff.mapv(|x| x * x).sum().sqrt();
                Ok(frobenius_norm)
            }
            QualityMetric::SpectralError => {
                // Simplified spectral norm approximation
                let diff = true_kernel - approx_kernel;
                let max_abs = diff
                    .mapv(|x: f64| x.abs())
                    .fold(0.0_f64, |acc, &x| acc.max(x));
                Ok(max_abs)
            }
            QualityMetric::RelativeError => {
                let diff = true_kernel - approx_kernel;
                let frobenius_diff = diff.mapv(|x| x * x).sum().sqrt();
                let frobenius_true = true_kernel.mapv(|x| x * x).sum().sqrt();
                Ok(frobenius_diff / frobenius_true.max(1e-8))
            }
            QualityMetric::NuclearError => {
                // Simplified nuclear norm (sum of absolute values as approximation)
                let diff = true_kernel - approx_kernel;
                let nuclear_norm = diff.mapv(|x| x.abs()).sum();
                Ok(nuclear_norm)
            }
            QualityMetric::EffectiveRank => {
                // Simplified effective rank computation
                let eigenvalues = self.compute_approximate_eigenvalues(approx_kernel)?;
                let sum_eigenvalues = eigenvalues.sum();
                let sum_squared_eigenvalues = eigenvalues.mapv(|x| x * x).sum();
                Ok(sum_eigenvalues * sum_eigenvalues / sum_squared_eigenvalues.max(1e-8))
            }
            QualityMetric::EigenvalueError => {
                let true_eigenvalues = self.compute_approximate_eigenvalues(true_kernel)?;
                let approx_eigenvalues = self.compute_approximate_eigenvalues(approx_kernel)?;

                let min_len = true_eigenvalues.len().min(approx_eigenvalues.len());
                let mut error = 0.0;
                for i in 0..min_len {
                    error += (true_eigenvalues[i] - approx_eigenvalues[i]).abs();
                }
                Ok(error / min_len as f64)
            }
        }
    }

    fn compute_approximate_quality_metric(
        &self,
        metric: &QualityMetric,
        original_data: &Array2<f64>,
        features: &Array2<f64>,
    ) -> Result<f64> {
        match metric {
            QualityMetric::KernelAlignment => {
                // Compute alignment with RBF kernel on original data
                let rbf_kernel = self.compute_rbf_kernel(original_data, 1.0)?;
                let approx_kernel = self.compute_approximate_kernel(features)?;
                Ok(self.compute_kernel_alignment(&rbf_kernel, &approx_kernel)?)
            }
            QualityMetric::EffectiveRank => {
                let approx_kernel = self.compute_approximate_kernel(features)?;
                let eigenvalues = self.compute_approximate_eigenvalues(&approx_kernel)?;
                let sum_eigenvalues = eigenvalues.sum();
                let sum_squared_eigenvalues = eigenvalues.mapv(|x| x * x).sum();
                Ok(sum_eigenvalues * sum_eigenvalues / sum_squared_eigenvalues.max(1e-8))
            }
            _ => {
                // For other metrics without ground truth, return approximation dimension as proxy
                Ok(features.ncols() as f64)
            }
        }
    }

    fn compute_kernel_alignment(&self, k1: &Array2<f64>, k2: &Array2<f64>) -> Result<f64> {
        let n = k1.nrows();

        // Center the kernels
        let row_means_k1 = k1.mean_axis(Axis(1)).unwrap();
        let row_means_k2 = k2.mean_axis(Axis(1)).unwrap();
        let total_mean_k1 = row_means_k1.mean().unwrap();
        let total_mean_k2 = row_means_k2.mean().unwrap();

        let mut k1_centered = k1.clone();
        let mut k2_centered = k2.clone();

        for i in 0..n {
            for j in 0..n {
                k1_centered[[i, j]] -= row_means_k1[i] + row_means_k1[j] - total_mean_k1;
                k2_centered[[i, j]] -= row_means_k2[i] + row_means_k2[j] - total_mean_k2;
            }
        }

        // Compute centered kernel alignment
        let numerator = (&k1_centered * &k2_centered).sum();
        let denom1 = (&k1_centered * &k1_centered).sum().sqrt();
        let denom2 = (&k2_centered * &k2_centered).sum().sqrt();

        Ok(numerator / (denom1 * denom2).max(1e-8))
    }

    fn compute_rbf_kernel(&self, data: &Array2<f64>, gamma: f64) -> Result<Array2<f64>> {
        let n_samples = data.nrows();
        let mut kernel = Array2::zeros((n_samples, n_samples));

        for i in 0..n_samples {
            for j in i..n_samples {
                let diff = &data.row(i) - &data.row(j);
                let dist_sq = diff.mapv(|x| x * x).sum();
                let similarity = (-gamma * dist_sq).exp();
                kernel[[i, j]] = similarity;
                kernel[[j, i]] = similarity;
            }
        }

        Ok(kernel)
    }

    fn compute_approximate_eigenvalues(&self, matrix: &Array2<f64>) -> Result<Array1<f64>> {
        // Simplified eigenvalue computation (diagonal elements as approximation)
        let n = matrix.nrows();
        let mut eigenvalues = Array1::zeros(n);

        for i in 0..n {
            eigenvalues[i] = matrix[[i, i]];
        }

        // Sort in descending order
        let mut eigenvalues_vec: Vec<f64> = eigenvalues.to_vec();
        eigenvalues_vec.sort_by(|a, b| b.partial_cmp(a).unwrap());

        Ok(Array1::from_vec(eigenvalues_vec))
    }

    /// Generate summary statistics from benchmark results
    pub fn summarize_results(&self) -> Vec<BenchmarkSummary> {
        let mut summaries = Vec::new();
        let mut grouped_results: HashMap<(String, String, usize), Vec<&BenchmarkResult>> =
            HashMap::new();

        // Group results by method, dataset, and approximation dimension
        for result in &self.results {
            let key = (
                result.method_name.clone(),
                result.dataset_name.clone(),
                result.approximation_dimension,
            );
            grouped_results.entry(key).or_default().push(result);
        }

        // Compute summary statistics for each group
        for ((method_name, dataset_name, approximation_dimension), results) in grouped_results {
            let total_results = results.len();
            let successful_results: Vec<_> = results.into_iter().filter(|r| r.success).collect();

            if successful_results.is_empty() {
                continue;
            }

            let mut quality_means = HashMap::new();
            let mut quality_stds = HashMap::new();
            let mut performance_means = HashMap::new();
            let mut performance_stds = HashMap::new();

            // Compute quality metric statistics
            for metric in &self.config.quality_metrics {
                let values: Vec<f64> = successful_results
                    .iter()
                    .filter_map(|r| r.quality_scores.get(metric))
                    .cloned()
                    .collect();

                if !values.is_empty() {
                    let mean = values.iter().sum::<f64>() / values.len() as f64;
                    let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
                        / values.len() as f64;
                    let std = variance.sqrt();

                    quality_means.insert(metric.clone(), mean);
                    quality_stds.insert(metric.clone(), std);
                }
            }

            // Compute performance metric statistics
            for metric in &self.config.performance_metrics {
                let values: Vec<Duration> = successful_results
                    .iter()
                    .filter_map(|r| r.performance_scores.get(metric))
                    .cloned()
                    .collect();

                if !values.is_empty() {
                    let mean_secs: f64 =
                        values.iter().map(|d| d.as_secs_f64()).sum::<f64>() / values.len() as f64;
                    let variance: f64 = values
                        .iter()
                        .map(|d| (d.as_secs_f64() - mean_secs).powi(2))
                        .sum::<f64>()
                        / values.len() as f64;
                    let std_secs = variance.sqrt();

                    performance_means.insert(metric.clone(), Duration::from_secs_f64(mean_secs));
                    performance_stds.insert(metric.clone(), Duration::from_secs_f64(std_secs));
                }
            }

            // Compute memory usage statistics
            let memory_values: Vec<f64> = successful_results
                .iter()
                .map(|r| r.memory_usage_mb)
                .collect();

            let memory_mean = memory_values.iter().sum::<f64>() / memory_values.len() as f64;
            let memory_variance = memory_values
                .iter()
                .map(|x| (x - memory_mean).powi(2))
                .sum::<f64>()
                / memory_values.len() as f64;
            let memory_std = memory_variance.sqrt();

            let success_rate = successful_results.len() as f64 / total_results as f64;

            summaries.push(BenchmarkSummary {
                method_name,
                dataset_name,
                approximation_dimension,
                quality_means,
                quality_stds,
                performance_means,
                performance_stds,
                success_rate,
                memory_usage_mean_mb: memory_mean,
                memory_usage_std_mb: memory_std,
            });
        }

        summaries
    }

    /// Get all benchmark results
    pub fn get_results(&self) -> &[BenchmarkResult] {
        &self.results
    }

    /// Clear all results
    pub fn clear_results(&mut self) {
        self.results.clear();
    }

    /// Export results to CSV format
    pub fn export_results_csv(&self) -> String {
        let mut csv = String::new();

        // Header
        csv.push_str("method,dataset,n_components,repetition,success,");
        csv.push_str("kernel_alignment,frobenius_error,spectral_error,relative_error,");
        csv.push_str("fit_time_ms,transform_time_ms,total_time_ms,memory_mb\n");

        // Data rows
        for result in &self.results {
            csv.push_str(&format!(
                "{},{},{},{},{},",
                result.method_name,
                result.dataset_name,
                result.approximation_dimension,
                result.repetition,
                result.success
            ));

            // Quality metrics
            let kernel_alignment = result
                .quality_scores
                .get(&QualityMetric::KernelAlignment)
                .map(|x| x.to_string())
                .unwrap_or_else(|| "".to_string());
            let frobenius_error = result
                .quality_scores
                .get(&QualityMetric::FrobeniusError)
                .map(|x| x.to_string())
                .unwrap_or_else(|| "".to_string());
            let spectral_error = result
                .quality_scores
                .get(&QualityMetric::SpectralError)
                .map(|x| x.to_string())
                .unwrap_or_else(|| "".to_string());
            let relative_error = result
                .quality_scores
                .get(&QualityMetric::RelativeError)
                .map(|x| x.to_string())
                .unwrap_or_else(|| "".to_string());

            csv.push_str(&format!(
                "{},{},{},{},",
                kernel_alignment, frobenius_error, spectral_error, relative_error
            ));

            // Performance metrics
            let fit_time = result
                .performance_scores
                .get(&PerformanceMetric::FitTime)
                .map(|d| d.as_millis().to_string())
                .unwrap_or_else(|| "".to_string());
            let transform_time = result
                .performance_scores
                .get(&PerformanceMetric::TransformTime)
                .map(|d| d.as_millis().to_string())
                .unwrap_or_else(|| "".to_string());
            let total_time = result
                .performance_scores
                .get(&PerformanceMetric::TotalTime)
                .map(|d| d.as_millis().to_string())
                .unwrap_or_else(|| "".to_string());

            csv.push_str(&format!(
                "{},{},{},{}\n",
                fit_time, transform_time, total_time, result.memory_usage_mb
            ));
        }

        csv
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    // Mock implementation for testing
    struct MockRBFMethod {
        n_components: usize,
    }

    impl BenchmarkableKernelMethod for MockRBFMethod {
        fn method_name(&self) -> String {
            "MockRBF".to_string()
        }

        fn benchmark_fit(
            &self,
            _data: &Array2<f64>,
            _target: Option<&Array1<f64>>,
        ) -> Result<(Box<dyn BenchmarkableTransformer>, Duration)> {
            let start = Instant::now();

            // Simple mock implementation
            let duration = start.elapsed();
            Ok((
                Box::new(MockFittedRBF {
                    n_components: self.n_components,
                }),
                duration,
            ))
        }

        fn clone_method(&self) -> Box<dyn BenchmarkableKernelMethod> {
            Box::new(MockRBFMethod {
                n_components: self.n_components,
            })
        }
    }

    struct MockFittedRBF {
        n_components: usize,
    }

    impl BenchmarkableTransformer for MockFittedRBF {
        fn benchmark_transform(&self, data: &Array2<f64>) -> Result<(Array2<f64>, Duration)> {
            let start = Instant::now();
            // Simple mock transformation
            let result = Array2::zeros((data.nrows(), self.n_components));
            let duration = start.elapsed();
            Ok((result, duration))
        }

        fn approximation_dimension(&self) -> usize {
            self.n_components
        }

        fn memory_usage_mb(&self) -> f64 {
            // Rough estimate: n_components * n_features * 8 bytes
            (self.n_components * 10 * 8) as f64 / (1024.0 * 1024.0)
        }
    }

    #[test]
    fn test_benchmark_dataset_creation() {
        let dataset = BenchmarkDataset::gaussian(100, 10, 0.1, 42);
        assert_eq!(dataset.data.shape(), &[100, 10]);
        assert!(dataset.target.is_some());
        assert_eq!(dataset.target.as_ref().unwrap().len(), 100);
    }

    #[test]
    fn test_benchmark_execution() {
        let mut benchmark = KernelApproximationBenchmark::new(BenchmarkConfig {
            repetitions: 2,
            approximation_dimensions: vec![10, 20],
            ..Default::default()
        });

        let dataset = BenchmarkDataset::gaussian(50, 5, 0.1, 42);
        benchmark.add_dataset(dataset);

        let method = MockRBFMethod { n_components: 10 };
        let results = benchmark.benchmark_method(&method).unwrap();

        assert_eq!(results.len(), 4); // 1 dataset * 2 dimensions * 2 repetitions
        assert!(results.iter().all(|r| r.success));
    }

    #[test]
    fn test_benchmark_summary() {
        let mut benchmark = KernelApproximationBenchmark::new(BenchmarkConfig {
            repetitions: 3,
            approximation_dimensions: vec![10],
            ..Default::default()
        });

        let dataset = BenchmarkDataset::gaussian(50, 5, 0.1, 42);
        benchmark.add_dataset(dataset);

        let method = MockRBFMethod { n_components: 10 };
        benchmark.benchmark_method(&method).unwrap();

        let summaries = benchmark.summarize_results();
        assert_eq!(summaries.len(), 1);
        assert_eq!(summaries[0].success_rate, 1.0);
    }

    #[test]
    fn test_csv_export() {
        let mut benchmark = KernelApproximationBenchmark::new(BenchmarkConfig {
            repetitions: 1,
            approximation_dimensions: vec![5],
            ..Default::default()
        });

        let dataset = BenchmarkDataset::gaussian(20, 3, 0.1, 42);
        benchmark.add_dataset(dataset);

        let method = MockRBFMethod { n_components: 5 };
        benchmark.benchmark_method(&method).unwrap();

        let csv = benchmark.export_results_csv();
        assert!(csv.contains("method,dataset"));
        assert!(csv.contains("MockRBF"));
        assert!(csv.contains("Gaussian"));
    }
}
