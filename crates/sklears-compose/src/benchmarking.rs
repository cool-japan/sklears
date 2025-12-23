//! Benchmarking utilities for pipeline composition performance analysis
//!
//! This module provides comprehensive benchmarking frameworks for measuring
//! and comparing the performance of different composition strategies.

use scirs2_core::ndarray::{Array2, ArrayView2};
use scirs2_core::random::Rng;
use serde::{Deserialize, Serialize};
use sklears_core::{
    error::Result as SklResult,
    traits::{Estimator, Transform},
};
use std::collections::{BTreeMap, HashMap};
use std::fmt::Write as FmtWrite;
use std::time::{Duration, Instant};

/// Advanced benchmark configuration
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    /// Number of iterations for each benchmark
    pub iterations: usize,
    /// Warmup iterations before timing
    pub warmup_iterations: usize,
    /// Sample sizes to test
    pub sample_sizes: Vec<usize>,
    /// Feature counts to test
    pub feature_counts: Vec<usize>,
    /// Whether to include memory usage measurements
    pub measure_memory: bool,
    /// Whether to include throughput measurements
    pub measure_throughput: bool,
    /// Enable CPU profiling
    pub enable_cpu_profiling: bool,
    /// Enable cache miss analysis
    pub enable_cache_analysis: bool,
    /// Enable parallel execution profiling
    pub enable_parallel_profiling: bool,
    /// Maximum benchmark duration per test
    pub max_duration: Duration,
    /// Statistical confidence level (0.0 to 1.0)
    pub confidence_level: f64,
    /// Enable convergence detection
    pub enable_convergence_detection: bool,
    /// Custom benchmark tags for categorization
    pub tags: HashMap<String, String>,
    /// Outlier detection threshold (number of standard deviations)
    pub outlier_threshold: f64,
}

impl BenchmarkConfig {
    /// Create a new benchmark configuration
    #[must_use]
    pub fn new() -> Self {
        Self {
            iterations: 100,
            warmup_iterations: 10,
            sample_sizes: vec![100, 1000, 10000],
            feature_counts: vec![5, 10, 20, 50],
            measure_memory: true,
            measure_throughput: true,
            enable_cpu_profiling: true,
            enable_cache_analysis: false, // Requires special hardware support
            enable_parallel_profiling: true,
            max_duration: Duration::from_secs(300), // 5 minutes max
            confidence_level: 0.95,
            enable_convergence_detection: true,
            tags: HashMap::new(),
            outlier_threshold: 2.0, // 2 standard deviations
        }
    }

    /// Set number of iterations
    #[must_use]
    pub fn iterations(mut self, iterations: usize) -> Self {
        self.iterations = iterations;
        self
    }

    /// Set warmup iterations
    #[must_use]
    pub fn warmup_iterations(mut self, warmup: usize) -> Self {
        self.warmup_iterations = warmup;
        self
    }

    /// Set sample sizes to test
    #[must_use]
    pub fn sample_sizes(mut self, sizes: Vec<usize>) -> Self {
        self.sample_sizes = sizes;
        self
    }

    /// Set feature counts to test
    #[must_use]
    pub fn feature_counts(mut self, counts: Vec<usize>) -> Self {
        self.feature_counts = counts;
        self
    }

    /// Enable/disable memory measurements
    #[must_use]
    pub fn measure_memory(mut self, enable: bool) -> Self {
        self.measure_memory = enable;
        self
    }

    /// Enable/disable throughput measurements
    #[must_use]
    pub fn measure_throughput(mut self, enable: bool) -> Self {
        self.measure_throughput = enable;
        self
    }

    /// Enable/disable CPU profiling
    #[must_use]
    pub fn enable_cpu_profiling(mut self, enable: bool) -> Self {
        self.enable_cpu_profiling = enable;
        self
    }

    /// Enable/disable cache analysis
    #[must_use]
    pub fn enable_cache_analysis(mut self, enable: bool) -> Self {
        self.enable_cache_analysis = enable;
        self
    }

    /// Enable/disable parallel profiling
    #[must_use]
    pub fn enable_parallel_profiling(mut self, enable: bool) -> Self {
        self.enable_parallel_profiling = enable;
        self
    }

    /// Set maximum benchmark duration
    #[must_use]
    pub fn max_duration(mut self, duration: Duration) -> Self {
        self.max_duration = duration;
        self
    }

    /// Set statistical confidence level
    #[must_use]
    pub fn confidence_level(mut self, level: f64) -> Self {
        self.confidence_level = level.clamp(0.0, 1.0);
        self
    }

    /// Enable/disable convergence detection
    #[must_use]
    pub fn enable_convergence_detection(mut self, enable: bool) -> Self {
        self.enable_convergence_detection = enable;
        self
    }

    /// Add custom tags
    #[must_use]
    pub fn add_tag(mut self, key: String, value: String) -> Self {
        self.tags.insert(key, value);
        self
    }

    /// Set outlier detection threshold
    #[must_use]
    pub fn outlier_threshold(mut self, threshold: f64) -> Self {
        self.outlier_threshold = threshold;
        self
    }
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Enhanced benchmark result with comprehensive metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    /// Name of the benchmark
    pub name: String,
    /// Mean execution time
    pub mean_time: Duration,
    /// Standard deviation of execution time
    pub std_dev_time: Duration,
    /// Minimum execution time
    pub min_time: Duration,
    /// Maximum execution time
    pub max_time: Duration,
    /// Median execution time
    pub median_time: Duration,
    /// 95th percentile execution time
    pub p95_time: Duration,
    /// 99th percentile execution time
    pub p99_time: Duration,
    /// Number of samples processed per second
    pub throughput: Option<f64>,
    /// Memory usage statistics
    pub memory_usage: Option<MemoryUsage>,
    /// CPU utilization metrics
    pub cpu_metrics: Option<CpuMetrics>,
    /// Cache performance metrics
    pub cache_metrics: Option<CacheMetrics>,
    /// Parallel execution metrics
    pub parallel_metrics: Option<ParallelMetrics>,
    /// Input data dimensions
    pub data_dimensions: (usize, usize),
    /// Performance classification
    pub performance_class: PerformanceClass,
    /// Statistical confidence interval
    pub confidence_interval: Option<(Duration, Duration)>,
    /// Number of outliers detected
    pub outliers_detected: usize,
    /// Timestamp of benchmark execution
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Custom metrics and metadata
    pub custom_metrics: HashMap<String, f64>,
}

impl BenchmarkResult {
    /// Create a new benchmark result
    #[must_use]
    pub fn new(name: String, times: Vec<Duration>, data_dimensions: (usize, usize)) -> Self {
        let mean_time = Duration::from_nanos(
            (times
                .iter()
                .map(std::time::Duration::as_nanos)
                .sum::<u128>()
                / times.len() as u128) as u64,
        );

        let mean_nanos = mean_time.as_nanos() as f64;
        let variance = times
            .iter()
            .map(|d| {
                let diff = d.as_nanos() as f64 - mean_nanos;
                diff * diff
            })
            .sum::<f64>()
            / times.len() as f64;

        let std_dev_time = Duration::from_nanos(variance.sqrt() as u64);
        let min_time = times.iter().min().copied().unwrap_or(Duration::ZERO);
        let max_time = times.iter().max().copied().unwrap_or(Duration::ZERO);

        // Calculate percentiles
        let mut sorted_times = times.clone();
        sorted_times.sort();

        let median_time = Self::calculate_percentile(&sorted_times, 50.0);
        let p95_time = Self::calculate_percentile(&sorted_times, 95.0);
        let p99_time = Self::calculate_percentile(&sorted_times, 99.0);

        Self {
            name,
            mean_time,
            std_dev_time,
            min_time,
            max_time,
            median_time,
            p95_time,
            p99_time,
            throughput: None,
            memory_usage: None,
            cpu_metrics: None,
            cache_metrics: None,
            parallel_metrics: None,
            data_dimensions,
            performance_class: PerformanceClass::Normal,
            confidence_interval: None,
            outliers_detected: 0,
            timestamp: chrono::Utc::now(),
            custom_metrics: HashMap::new(),
        }
    }

    /// Set throughput
    #[must_use]
    pub fn with_throughput(mut self, throughput: f64) -> Self {
        self.throughput = Some(throughput);
        self
    }

    /// Set memory usage
    #[must_use]
    pub fn with_memory_usage(mut self, memory_usage: MemoryUsage) -> Self {
        self.memory_usage = Some(memory_usage);
        self
    }

    /// Calculate performance score (higher is better)
    #[must_use]
    pub fn performance_score(&self) -> f64 {
        let time_score = 1.0 / self.mean_time.as_secs_f64();
        let throughput_score = self.throughput.unwrap_or(1.0);
        let memory_score = if let Some(ref mem) = self.memory_usage {
            1.0 / (mem.peak_usage_mb + 1.0)
        } else {
            1.0
        };

        // Weighted combination
        0.5 * time_score + 0.3 * throughput_score + 0.2 * memory_score
    }

    /// Calculate percentile from sorted duration vector
    ///
    /// Uses linear interpolation between closest values for fractional indices.
    ///
    /// # Arguments
    /// * `sorted_times` - Pre-sorted vector of durations
    /// * `percentile` - Percentile to calculate (0-100)
    ///
    /// # Returns
    /// Duration at the specified percentile
    fn calculate_percentile(sorted_times: &[Duration], percentile: f64) -> Duration {
        if sorted_times.is_empty() {
            return Duration::ZERO;
        }

        if sorted_times.len() == 1 {
            return sorted_times[0];
        }

        // Calculate index using linear interpolation
        // Percentile rank formula: index = (percentile / 100) * (n - 1)
        let index = (percentile / 100.0) * (sorted_times.len() - 1) as f64;
        let lower_index = index.floor() as usize;
        let upper_index = index.ceil() as usize;

        if lower_index == upper_index {
            // Exact index
            sorted_times[lower_index]
        } else {
            // Linear interpolation between two values
            let lower_value = sorted_times[lower_index].as_nanos() as f64;
            let upper_value = sorted_times[upper_index].as_nanos() as f64;
            let fraction = index - lower_index as f64;
            let interpolated = lower_value + fraction * (upper_value - lower_value);

            Duration::from_nanos(interpolated as u64)
        }
    }
}

/// Memory usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryUsage {
    /// Peak memory usage in MB
    pub peak_usage_mb: f64,
    /// Average memory usage in MB
    pub average_usage_mb: f64,
    /// Number of allocations
    pub allocations: usize,
    /// Number of deallocations
    pub deallocations: usize,
    /// Memory leak detection (bytes not freed)
    pub memory_leaks_bytes: u64,
    /// Allocation pattern efficiency score (0.0 to 1.0)
    pub allocation_efficiency: f64,
}

/// CPU utilization metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuMetrics {
    /// Average CPU utilization (0.0 to 1.0)
    pub average_utilization: f64,
    /// Peak CPU utilization (0.0 to 1.0)
    pub peak_utilization: f64,
    /// User mode CPU time percentage
    pub user_time_percent: f64,
    /// Kernel mode CPU time percentage
    pub kernel_time_percent: f64,
    /// Context switches per second
    pub context_switches_per_sec: f64,
    /// CPU efficiency score (0.0 to 1.0)
    pub efficiency_score: f64,
}

/// Cache performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheMetrics {
    /// L1 cache hit rate (0.0 to 1.0)
    pub l1_hit_rate: f64,
    /// L2 cache hit rate (0.0 to 1.0)
    pub l2_hit_rate: f64,
    /// L3 cache hit rate (0.0 to 1.0)
    pub l3_hit_rate: f64,
    /// Cache miss penalty (average cycles)
    pub miss_penalty_cycles: f64,
    /// Cache efficiency score (0.0 to 1.0)
    pub cache_efficiency: f64,
}

/// Parallel execution metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParallelMetrics {
    /// Number of threads used
    pub thread_count: usize,
    /// Parallel efficiency (0.0 to 1.0)
    pub parallel_efficiency: f64,
    /// Load balancing score (0.0 to 1.0)
    pub load_balance_score: f64,
    /// Thread contention events
    pub contention_events: usize,
    /// Synchronization overhead percentage
    pub sync_overhead_percent: f64,
    /// Speedup achieved vs single thread
    pub speedup_ratio: f64,
}

/// Performance classification categories
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PerformanceClass {
    /// Excellent performance
    Excellent,
    /// Good performance
    Good,
    /// Normal performance
    Normal,
    /// Acceptable performance
    Acceptable,
    /// Poor performance needing optimization
    Poor,
    /// Critical performance issues
    Critical,
}

/// Benchmark suite for comparing multiple strategies
#[derive(Debug)]
pub struct BenchmarkSuite {
    config: BenchmarkConfig,
    results: HashMap<String, Vec<BenchmarkResult>>,
}

impl BenchmarkSuite {
    /// Create a new benchmark suite
    #[must_use]
    pub fn new(config: BenchmarkConfig) -> Self {
        Self {
            config,
            results: HashMap::new(),
        }
    }

    /// Benchmark a transformation strategy
    pub fn benchmark_transformer<T>(&mut self, name: &str, transformer: &T) -> SklResult<()>
    where
        T: for<'a> Transform<ArrayView2<'a, f64>, Array2<f64>>,
    {
        let mut strategy_results = Vec::new();

        for &n_samples in &self.config.sample_sizes {
            for &n_features in &self.config.feature_counts {
                let data = self.generate_data(n_samples, n_features);
                let result = self.benchmark_single_transform(
                    &format!("{}_{}_{}_{}", name, "transform", n_samples, n_features),
                    transformer,
                    &data,
                    (n_samples, n_features),
                )?;
                strategy_results.push(result);
            }
        }

        self.results.insert(name.to_string(), strategy_results);
        Ok(())
    }

    /// Benchmark a single transformation
    fn benchmark_single_transform<T>(
        &self,
        bench_name: &str,
        transformer: &T,
        data: &Array2<f64>,
        dimensions: (usize, usize),
    ) -> SklResult<BenchmarkResult>
    where
        T: for<'a> Transform<ArrayView2<'a, f64>, Array2<f64>>,
    {
        let mut times = Vec::new();

        // Warmup
        for _ in 0..self.config.warmup_iterations {
            let _ = transformer.transform(&data.view())?;
        }

        // Benchmark
        for _ in 0..self.config.iterations {
            let start = Instant::now();
            let _ = transformer.transform(&data.view())?;
            times.push(start.elapsed());
        }

        let mut result = BenchmarkResult::new(bench_name.to_string(), times, dimensions);

        // Calculate throughput
        if self.config.measure_throughput {
            let samples_per_sec = dimensions.0 as f64 / result.mean_time.as_secs_f64();
            result = result.with_throughput(samples_per_sec);
        }

        // Measure memory usage (simplified version)
        if self.config.measure_memory {
            let memory_usage = self.estimate_memory_usage(dimensions);
            result = result.with_memory_usage(memory_usage);
        }

        Ok(result)
    }

    /// Benchmark pipeline composition strategies
    pub fn benchmark_composition_strategies(&mut self) -> SklResult<()> {
        // This would benchmark different composition approaches
        // For now, we'll simulate some benchmark results

        let strategies = vec![
            "sequential_pipeline",
            "parallel_feature_union",
            "dag_pipeline",
            "zero_cost_composition",
        ];

        for strategy in strategies {
            let mut strategy_results = Vec::new();

            for &n_samples in &self.config.sample_sizes {
                for &n_features in &self.config.feature_counts {
                    // Simulate benchmark results for different strategies
                    let base_time =
                        Duration::from_micros(100 + (n_samples * n_features / 1000) as u64);
                    let strategy_multiplier = match strategy {
                        "sequential_pipeline" => 1.0,
                        "parallel_feature_union" => 0.6, // Faster due to parallelization
                        "dag_pipeline" => 0.8,           // Moderate speedup
                        "zero_cost_composition" => 0.4,  // Fastest due to zero-cost abstractions
                        _ => 1.0,
                    };

                    let adjusted_time = Duration::from_nanos(
                        (base_time.as_nanos() as f64 * strategy_multiplier) as u64,
                    );

                    let times = vec![adjusted_time; self.config.iterations];
                    let mut result = BenchmarkResult::new(
                        format!(
                            "{}_{}_{}_{}",
                            strategy, "composition", n_samples, n_features
                        ),
                        times,
                        (n_samples, n_features),
                    );

                    if self.config.measure_throughput {
                        let throughput = n_samples as f64 / adjusted_time.as_secs_f64();
                        result = result.with_throughput(throughput);
                    }

                    if self.config.measure_memory {
                        let memory_usage = self.estimate_memory_usage((n_samples, n_features));
                        result = result.with_memory_usage(memory_usage);
                    }

                    strategy_results.push(result);
                }
            }

            self.results.insert(strategy.to_string(), strategy_results);
        }

        Ok(())
    }

    /// Generate test data
    fn generate_data(&self, n_samples: usize, n_features: usize) -> Array2<f64> {
        use scirs2_core::random::rngs::StdRng;
        use scirs2_core::random::SeedableRng;

        let mut rng = StdRng::seed_from_u64(42);
        Array2::from_shape_fn((n_samples, n_features), |_| rng.gen_range(-1.0..1.0))
    }

    /// Estimate memory usage (simplified)
    fn estimate_memory_usage(&self, dimensions: (usize, usize)) -> MemoryUsage {
        let (n_samples, n_features) = dimensions;
        let data_size_mb = (n_samples * n_features * 8) as f64 / (1024.0 * 1024.0);

        /// MemoryUsage
        MemoryUsage {
            peak_usage_mb: data_size_mb * 2.5, // Assume some overhead
            average_usage_mb: data_size_mb * 1.8,
            allocations: n_samples + n_features,
            deallocations: n_samples + n_features,
            allocation_efficiency: 0.95, // Default efficiency
            memory_leaks_bytes: 0,       // No leaks assumed
        }
    }

    /// Get benchmark results
    #[must_use]
    pub fn results(&self) -> &HashMap<String, Vec<BenchmarkResult>> {
        &self.results
    }

    /// Generate comparison report
    #[must_use]
    pub fn comparison_report(&self) -> BenchmarkReport {
        BenchmarkReport::new(self.results.clone())
    }

    /// Get the best performing strategy for given dimensions
    #[must_use]
    pub fn best_strategy(&self, dimensions: (usize, usize)) -> Option<(&str, &BenchmarkResult)> {
        let mut best_strategy = None;
        let mut best_score = 0.0;

        for (strategy_name, results) in &self.results {
            for result in results {
                if result.data_dimensions == dimensions {
                    let score = result.performance_score();
                    if score > best_score {
                        best_score = score;
                        best_strategy = Some((strategy_name.as_str(), result));
                    }
                }
            }
        }

        best_strategy
    }
}

/// Comprehensive benchmark report
#[derive(Debug, Clone)]
pub struct BenchmarkReport {
    /// Results by strategy
    pub strategy_results: HashMap<String, Vec<BenchmarkResult>>,
    /// Performance rankings
    pub performance_rankings: Vec<(String, f64)>,
    /// Scalability analysis
    pub scalability_analysis: HashMap<String, ScalabilityMetrics>,
}

impl BenchmarkReport {
    /// Create a new benchmark report
    #[must_use]
    pub fn new(strategy_results: HashMap<String, Vec<BenchmarkResult>>) -> Self {
        let performance_rankings = Self::calculate_performance_rankings(&strategy_results);
        let scalability_analysis = Self::analyze_scalability(&strategy_results);

        Self {
            strategy_results,
            performance_rankings,
            scalability_analysis,
        }
    }

    /// Calculate performance rankings across all strategies
    fn calculate_performance_rankings(
        results: &HashMap<String, Vec<BenchmarkResult>>,
    ) -> Vec<(String, f64)> {
        let mut rankings = Vec::new();

        for (strategy, strategy_results) in results {
            let avg_score = strategy_results
                .iter()
                .map(BenchmarkResult::performance_score)
                .sum::<f64>()
                / strategy_results.len() as f64;
            rankings.push((strategy.clone(), avg_score));
        }

        rankings.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        rankings
    }

    /// Analyze scalability characteristics
    fn analyze_scalability(
        results: &HashMap<String, Vec<BenchmarkResult>>,
    ) -> HashMap<String, ScalabilityMetrics> {
        let mut analysis = HashMap::new();

        for (strategy, strategy_results) in results {
            let mut sample_scalability = Vec::new();
            let mut feature_scalability = Vec::new();

            // Analyze how performance scales with data size
            for result in strategy_results {
                let (n_samples, n_features) = result.data_dimensions;
                let time_per_sample = result.mean_time.as_secs_f64() / n_samples as f64;
                let time_per_feature = result.mean_time.as_secs_f64() / n_features as f64;

                sample_scalability.push((n_samples, time_per_sample));
                feature_scalability.push((n_features, time_per_feature));
            }

            let metrics = ScalabilityMetrics {
                sample_complexity: Self::estimate_complexity(&sample_scalability),
                feature_complexity: Self::estimate_complexity(&feature_scalability),
                memory_efficiency: strategy_results
                    .iter()
                    .filter_map(|r| r.memory_usage.as_ref())
                    .map(|m| m.peak_usage_mb)
                    .sum::<f64>()
                    / strategy_results.len() as f64,
            };

            analysis.insert(strategy.clone(), metrics);
        }

        analysis
    }

    /// Estimate computational complexity
    fn estimate_complexity(data_points: &[(usize, f64)]) -> ComplexityClass {
        if data_points.len() < 2 {
            return ComplexityClass::Unknown;
        }

        // Simple heuristic: if time grows linearly with size, it's O(n)
        // If it grows quadratically, it's O(n²), etc.
        let mut growth_ratios = Vec::new();

        for i in 1..data_points.len() {
            let (size1, time1) = data_points[i - 1];
            let (size2, time2) = data_points[i];

            if size1 > 0 && time1 > 0.0 {
                let size_ratio = size2 as f64 / size1 as f64;
                let time_ratio = time2 / time1;

                if size_ratio > 1.0 {
                    growth_ratios.push(time_ratio / size_ratio);
                }
            }
        }

        if growth_ratios.is_empty() {
            return ComplexityClass::Unknown;
        }

        let avg_growth = growth_ratios.iter().sum::<f64>() / growth_ratios.len() as f64;

        match avg_growth {
            x if x < 1.2 => ComplexityClass::Linear,
            x if x < 2.0 => ComplexityClass::LogLinear,
            x if x < 4.0 => ComplexityClass::Quadratic,
            _ => ComplexityClass::Higher,
        }
    }

    /// Generate summary statistics
    #[must_use]
    pub fn summary(&self) -> String {
        let mut summary = String::new();
        summary.push_str("Benchmark Summary\n");
        summary.push_str("================\n\n");

        summary.push_str("Performance Rankings:\n");
        for (i, (strategy, score)) in self.performance_rankings.iter().enumerate() {
            let _ = write!(summary, "{}. {} (score: {:.3})\n", i + 1, strategy, score);
        }

        summary.push_str("\nScalability Analysis:\n");
        for (strategy, metrics) in &self.scalability_analysis {
            let _ = write!(
                summary,
                "{}: Sample complexity: {:?}, Feature complexity: {:?}\n",
                strategy, metrics.sample_complexity, metrics.feature_complexity
            );
        }

        summary
    }

    /// Get recommendations based on use case
    #[must_use]
    pub fn recommendations(&self, use_case: UseCase) -> Vec<String> {
        let mut recommendations = Vec::new();

        match use_case {
            UseCase::HighThroughput => {
                // Recommend strategies with best throughput
                let mut throughput_rankings = self
                    .strategy_results
                    .iter()
                    .map(|(name, results)| {
                        let avg_throughput =
                            results.iter().filter_map(|r| r.throughput).sum::<f64>()
                                / results.len() as f64;
                        (name, avg_throughput)
                    })
                    .collect::<Vec<_>>();

                throughput_rankings.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

                for (strategy, throughput) in throughput_rankings.iter().take(3) {
                    recommendations.push(format!(
                        "{strategy} (throughput: {throughput:.2} samples/sec)"
                    ));
                }
            }
            UseCase::LowLatency => {
                // Recommend strategies with lowest latency
                let mut latency_rankings = self
                    .strategy_results
                    .iter()
                    .map(|(name, results)| {
                        let avg_latency = results
                            .iter()
                            .map(|r| r.mean_time.as_secs_f64())
                            .sum::<f64>()
                            / results.len() as f64;
                        (name, avg_latency)
                    })
                    .collect::<Vec<_>>();

                latency_rankings.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

                for (strategy, latency) in latency_rankings.iter().take(3) {
                    recommendations.push(format!(
                        "{} (latency: {:.3}ms)",
                        strategy,
                        latency * 1000.0
                    ));
                }
            }
            UseCase::MemoryConstrained => {
                // Recommend strategies with lowest memory usage
                let mut memory_rankings = self
                    .strategy_results
                    .iter()
                    .map(|(name, results)| {
                        let avg_memory = results
                            .iter()
                            .filter_map(|r| r.memory_usage.as_ref())
                            .map(|m| m.peak_usage_mb)
                            .sum::<f64>()
                            / results.len() as f64;
                        (name, avg_memory)
                    })
                    .collect::<Vec<_>>();

                memory_rankings.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

                for (strategy, memory) in memory_rankings.iter().take(3) {
                    recommendations.push(format!("{strategy} (memory: {memory:.2} MB)"));
                }
            }
        }

        recommendations
    }
}

/// Scalability metrics for a strategy
#[derive(Debug, Clone)]
pub struct ScalabilityMetrics {
    /// Computational complexity with respect to sample count
    pub sample_complexity: ComplexityClass,
    /// Computational complexity with respect to feature count
    pub feature_complexity: ComplexityClass,
    /// Memory efficiency score
    pub memory_efficiency: f64,
}

/// Computational complexity classification
#[derive(Debug, Clone, PartialEq)]
pub enum ComplexityClass {
    /// O(1) - Constant time
    Constant,
    /// O(log n) - Logarithmic time
    Logarithmic,
    /// O(n) - Linear time
    Linear,
    /// O(n log n) - Linearithmic time
    LogLinear,
    /// O(n²) - Quadratic time
    Quadratic,
    /// O(n³) or higher - Higher polynomial time
    Higher,
    /// Unknown complexity
    Unknown,
}

/// Use case for benchmark recommendations
#[derive(Debug, Clone, PartialEq)]
pub enum UseCase {
    /// High throughput processing
    HighThroughput,
    /// Low latency requirements
    LowLatency,
    /// Memory-constrained environments
    MemoryConstrained,
}

/// Advanced benchmarking extensions for specialized analysis
pub mod advanced_benchmarking {
    use super::{BTreeMap, BenchmarkResult, HashMap};

    /// Advanced benchmark analyzer with statistical modeling
    pub struct AdvancedBenchmarkAnalyzer {
        results_database: HashMap<String, Vec<BenchmarkResult>>,
        statistical_models: HashMap<String, StatisticalModel>,
        trend_analyzer: TrendAnalyzer,
    }

    /// Statistical model for performance prediction
    #[derive(Debug, Clone)]
    pub struct StatisticalModel {
        pub model_type: ModelType,
        pub coefficients: Vec<f64>,
        pub r_squared: f64,
        pub confidence_intervals: Vec<(f64, f64)>,
        pub prediction_accuracy: f64,
    }

    /// Types of statistical models
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub enum ModelType {
        /// Linear
        Linear,
        /// Polynomial
        Polynomial,
        /// Exponential
        Exponential,
        /// Logarithmic
        Logarithmic,
        /// PowerLaw
        PowerLaw,
    }

    /// Trend analysis for performance over time
    pub struct TrendAnalyzer {
        historical_data: BTreeMap<chrono::DateTime<chrono::Utc>, f64>,
        trend_models: HashMap<String, TrendModel>,
    }

    /// Trend model for performance forecasting
    #[derive(Debug, Clone)]
    pub struct TrendModel {
        pub trend_direction: TrendDirection,
        pub slope: f64,
        pub seasonal_component: Option<f64>,
        pub confidence: f64,
        pub forecast_horizon: chrono::Duration,
    }

    /// Trend directions
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub enum TrendDirection {
        /// Improving
        Improving,
        /// Degrading
        Degrading,
        /// Stable
        Stable,
        /// Cyclical
        Cyclical,
        /// Unknown
        Unknown,
    }

    /// Resource efficiency analyzer
    pub struct ResourceEfficiencyAnalyzer {
        cpu_profiles: HashMap<String, CpuProfile>,
        memory_profiles: HashMap<String, MemoryProfile>,
        energy_profiles: HashMap<String, EnergyProfile>,
    }

    /// CPU usage profile
    #[derive(Debug, Clone)]
    pub struct CpuProfile {
        pub utilization_history: Vec<f64>,
        pub peak_utilization: f64,
        pub average_utilization: f64,
        pub efficiency_score: f64,
        pub hotspots: Vec<String>,
    }

    /// Memory usage profile
    #[derive(Debug, Clone)]
    pub struct MemoryProfile {
        pub allocation_pattern: AllocationPattern,
        pub peak_usage_mb: f64,
        pub average_usage_mb: f64,
        pub fragmentation_score: f64,
        pub gc_impact: f64,
    }

    /// Memory allocation patterns
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub enum AllocationPattern {
        /// Constant
        Constant,
        /// Linear
        Linear,
        /// Exponential
        Exponential,
        /// Spiky
        Spiky,
        /// Cyclical
        Cyclical,
    }

    /// Energy consumption profile
    #[derive(Debug, Clone)]
    pub struct EnergyProfile {
        pub total_energy_joules: f64,
        pub average_power_watts: f64,
        pub efficiency_score: f64,
        pub carbon_footprint_kg: f64,
    }

    /// Comparative benchmark analysis
    pub struct ComparativeBenchmarkAnalysis {
        pub baseline_component: String,
        pub comparison_components: Vec<String>,
        pub metrics: Vec<ComparisonMetric>,
        pub statistical_significance: f64,
        pub effect_sizes: HashMap<String, f64>,
        pub confidence_intervals: HashMap<String, (f64, f64)>,
    }

    /// Comparison metrics
    #[derive(Debug, Clone)]
    pub struct ComparisonMetric {
        pub name: String,
        pub baseline_value: f64,
        pub comparison_values: HashMap<String, f64>,
        pub relative_improvements: HashMap<String, f64>,
        pub statistical_tests: HashMap<String, StatisticalTest>,
    }

    /// Statistical test results
    #[derive(Debug, Clone)]
    pub struct StatisticalTest {
        pub test_type: TestType,
        pub p_value: f64,
        pub test_statistic: f64,
        pub is_significant: bool,
        pub effect_size: f64,
    }

    /// Types of statistical tests
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub enum TestType {
        /// TTest
        TTest,
        /// WilcoxonTest
        WilcoxonTest,
        /// MannWhitneyU
        MannWhitneyU,
        /// KruskalWallis
        KruskalWallis,
        /// ANOVA
        ANOVA,
    }

    impl Default for AdvancedBenchmarkAnalyzer {
        fn default() -> Self {
            Self::new()
        }
    }

    impl AdvancedBenchmarkAnalyzer {
        /// Create a new advanced benchmark analyzer
        #[must_use]
        pub fn new() -> Self {
            Self {
                results_database: HashMap::new(),
                statistical_models: HashMap::new(),
                trend_analyzer: TrendAnalyzer::new(),
            }
        }

        /// Add benchmark results for analysis
        pub fn add_results(&mut self, component_name: String, results: Vec<BenchmarkResult>) {
            self.results_database.insert(component_name, results);
        }

        /// Build statistical model for performance prediction
        pub fn build_model(&mut self, component_name: &str) -> Option<StatisticalModel> {
            if let Some(results) = self.results_database.get(component_name) {
                let data_points: Vec<(f64, f64)> = results
                    .iter()
                    .enumerate()
                    .map(|(i, result)| (i as f64, result.mean_time.as_millis() as f64))
                    .collect();

                let model = self.fit_model(&data_points);
                self.statistical_models
                    .insert(component_name.to_string(), model.clone());
                Some(model)
            } else {
                None
            }
        }

        /// Predict performance for given input size
        #[must_use]
        pub fn predict_performance(&self, component_name: &str, input_size: f64) -> Option<f64> {
            self.statistical_models
                .get(component_name)
                .map(|model| self.apply_model(model, input_size))
        }

        /// Analyze performance trends over time
        pub fn analyze_trends(&mut self, component_name: &str) -> Option<TrendModel> {
            self.trend_analyzer.analyze_component_trends(component_name)
        }

        /// Generate comprehensive analysis report
        #[must_use]
        pub fn generate_analysis_report(&self) -> AdvancedAnalysisReport {
            let mut component_analyses = HashMap::new();

            for (component_name, results) in &self.results_database {
                let analysis = ComponentAnalysis {
                    component_name: component_name.clone(),
                    total_benchmarks: results.len(),
                    performance_model: self.statistical_models.get(component_name).cloned(),
                    trend_analysis: None, // Would be populated from trend analyzer
                    efficiency_scores: self.calculate_efficiency_scores(results),
                    recommendations: self
                        .generate_component_recommendations(component_name, results),
                };
                component_analyses.insert(component_name.clone(), analysis);
            }

            /// AdvancedAnalysisReport
            AdvancedAnalysisReport {
                analysis_timestamp: chrono::Utc::now(),
                total_components: component_analyses.len(),
                component_analyses,
                cross_component_insights: self.generate_cross_component_insights(),
                optimization_recommendations: self.generate_optimization_recommendations(),
            }
        }

        /// Perform comparative analysis between components
        #[must_use]
        pub fn comparative_analysis(
            &self,
            baseline: &str,
            comparisons: &[&str],
        ) -> Option<ComparativeBenchmarkAnalysis> {
            if let Some(baseline_results) = self.results_database.get(baseline) {
                let mut comparison_values = HashMap::new();
                let mut effect_sizes = HashMap::new();

                for &component in comparisons {
                    if let Some(comp_results) = self.results_database.get(component) {
                        let baseline_mean = self.calculate_mean_performance(baseline_results);
                        let comp_mean = self.calculate_mean_performance(comp_results);

                        comparison_values.insert(component.to_string(), comp_mean);

                        // Calculate effect size (Cohen's d)
                        let effect_size =
                            self.calculate_effect_size(baseline_results, comp_results);
                        effect_sizes.insert(component.to_string(), effect_size);
                    }
                }

                Some(ComparativeBenchmarkAnalysis {
                    baseline_component: baseline.to_string(),
                    comparison_components: comparisons.iter().map(|s| (*s).to_string()).collect(),
                    metrics: vec![], // Would be populated with detailed metrics
                    statistical_significance: 0.95, // Placeholder
                    effect_sizes,
                    confidence_intervals: HashMap::new(), // Would calculate actual CIs
                })
            } else {
                None
            }
        }

        // Private implementation methods

        fn fit_model(&self, data: &[(f64, f64)]) -> StatisticalModel {
            // Simplified linear regression
            let n = data.len() as f64;
            let sum_x: f64 = data.iter().map(|(x, _)| x).sum();
            let sum_y: f64 = data.iter().map(|(_, y)| y).sum();
            let sum_xy: f64 = data.iter().map(|(x, y)| x * y).sum();
            let sum_x2: f64 = data.iter().map(|(x, _)| x * x).sum();

            let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
            let intercept = (sum_y - slope * sum_x) / n;

            // Calculate R-squared
            let mean_y = sum_y / n;
            let ss_tot: f64 = data.iter().map(|(_, y)| (y - mean_y).powi(2)).sum();
            let ss_res: f64 = data
                .iter()
                .map(|(x, y)| (y - (slope * x + intercept)).powi(2))
                .sum();
            let r_squared = 1.0 - (ss_res / ss_tot);

            /// StatisticalModel
            StatisticalModel {
                model_type: ModelType::Linear,
                coefficients: vec![intercept, slope],
                r_squared,
                confidence_intervals: vec![], // Would calculate actual CIs
                prediction_accuracy: r_squared,
            }
        }

        fn apply_model(&self, model: &StatisticalModel, input: f64) -> f64 {
            match model.model_type {
                ModelType::Linear => model.coefficients[0] + model.coefficients[1] * input,
                ModelType::Polynomial => model
                    .coefficients
                    .iter()
                    .enumerate()
                    .map(|(i, &coef)| coef * input.powi(i as i32))
                    .sum(),
                _ => model.coefficients[0] + model.coefficients[1] * input, // Fallback to linear
            }
        }

        fn calculate_mean_performance(&self, results: &[BenchmarkResult]) -> f64 {
            results
                .iter()
                .map(|r| r.mean_time.as_millis() as f64)
                .sum::<f64>()
                / results.len() as f64
        }

        fn calculate_effect_size(
            &self,
            baseline: &[BenchmarkResult],
            comparison: &[BenchmarkResult],
        ) -> f64 {
            let baseline_mean = self.calculate_mean_performance(baseline);
            let comparison_mean = self.calculate_mean_performance(comparison);

            // Simplified effect size calculation
            let pooled_std = 1.0; // Would calculate actual pooled standard deviation
            (comparison_mean - baseline_mean) / pooled_std
        }

        fn calculate_efficiency_scores(
            &self,
            _results: &[BenchmarkResult],
        ) -> HashMap<String, f64> {
            let mut scores = HashMap::new();
            scores.insert("cpu_efficiency".to_string(), 0.85);
            scores.insert("memory_efficiency".to_string(), 0.78);
            scores.insert("energy_efficiency".to_string(), 0.82);
            scores
        }

        fn generate_component_recommendations(
            &self,
            _component_name: &str,
            _results: &[BenchmarkResult],
        ) -> Vec<String> {
            vec![
                "Consider optimizing memory allocation patterns".to_string(),
                "Investigate opportunities for parallel processing".to_string(),
                "Profile CPU-intensive operations for bottlenecks".to_string(),
            ]
        }

        fn generate_cross_component_insights(&self) -> Vec<String> {
            vec![
                "Linear scaling algorithms show better performance on large datasets".to_string(),
                "Memory-intensive components benefit from batch processing".to_string(),
                "CPU-bound operations should prioritize algorithmic optimizations".to_string(),
            ]
        }

        fn generate_optimization_recommendations(&self) -> Vec<String> {
            vec![
                "Implement adaptive batch sizing based on available memory".to_string(),
                "Consider using SIMD instructions for vectorizable operations".to_string(),
                "Implement memory pooling for frequently allocated objects".to_string(),
            ]
        }
    }

    impl TrendAnalyzer {
        #[must_use]
        pub fn new() -> Self {
            Self {
                historical_data: BTreeMap::new(),
                trend_models: HashMap::new(),
            }
        }

        #[must_use]
        pub fn analyze_component_trends(&self, _component_name: &str) -> Option<TrendModel> {
            // Simplified trend analysis
            Some(TrendModel {
                trend_direction: TrendDirection::Stable,
                slope: 0.01,
                seasonal_component: None,
                confidence: 0.8,
                forecast_horizon: chrono::Duration::days(30),
            })
        }
    }

    /// Component-specific analysis results
    #[derive(Debug, Clone)]
    pub struct ComponentAnalysis {
        pub component_name: String,
        pub total_benchmarks: usize,
        pub performance_model: Option<StatisticalModel>,
        pub trend_analysis: Option<TrendModel>,
        pub efficiency_scores: HashMap<String, f64>,
        pub recommendations: Vec<String>,
    }

    /// Advanced analysis report
    #[derive(Debug, Clone)]
    pub struct AdvancedAnalysisReport {
        pub analysis_timestamp: chrono::DateTime<chrono::Utc>,
        pub total_components: usize,
        pub component_analyses: HashMap<String, ComponentAnalysis>,
        pub cross_component_insights: Vec<String>,
        pub optimization_recommendations: Vec<String>,
    }

    impl Default for TrendAnalyzer {
        fn default() -> Self {
            Self::new()
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::mock::MockTransformer;

    #[test]
    fn test_benchmark_config() {
        let config = BenchmarkConfig::new()
            .iterations(50)
            .sample_sizes(vec![100, 500])
            .feature_counts(vec![5, 10]);

        assert_eq!(config.iterations, 50);
        assert_eq!(config.sample_sizes, vec![100, 500]);
        assert_eq!(config.feature_counts, vec![5, 10]);
    }

    #[test]
    fn test_benchmark_result() {
        let times = vec![
            Duration::from_millis(10),
            Duration::from_millis(12),
            Duration::from_millis(11),
        ];

        let result = BenchmarkResult::new("test".to_string(), times, (100, 5));
        assert_eq!(result.name, "test");
        assert_eq!(result.data_dimensions, (100, 5));
        assert!(result.performance_score() > 0.0);
    }

    #[test]
    fn test_benchmark_suite() {
        let config = BenchmarkConfig::new()
            .iterations(5)
            .sample_sizes(vec![10])
            .feature_counts(vec![3]);

        let mut suite = BenchmarkSuite::new(config);
        let transformer = MockTransformer::new();

        suite.benchmark_transformer("mock", &transformer).unwrap();
        assert!(suite.results().contains_key("mock"));
    }

    #[test]
    fn test_benchmark_report() {
        let mut results = HashMap::new();
        let times = vec![Duration::from_millis(10); 3];
        let result = BenchmarkResult::new("test".to_string(), times, (100, 5));
        results.insert("strategy1".to_string(), vec![result]);

        let report = BenchmarkReport::new(results);
        assert_eq!(report.performance_rankings.len(), 1);
        assert!(!report.summary().is_empty());
    }

    #[test]
    fn test_complexity_estimation() {
        let data = vec![(10, 0.1), (20, 0.2), (30, 0.3)];
        let complexity = BenchmarkReport::estimate_complexity(&data);
        assert_eq!(complexity, ComplexityClass::Linear);
    }

    #[test]
    fn test_use_case_recommendations() {
        let mut results = HashMap::new();
        let times = vec![Duration::from_millis(10); 3];
        let result =
            BenchmarkResult::new("test".to_string(), times, (100, 5)).with_throughput(1000.0);
        results.insert("fast_strategy".to_string(), vec![result]);

        let report = BenchmarkReport::new(results);
        let recommendations = report.recommendations(UseCase::HighThroughput);
        assert!(!recommendations.is_empty());
    }
}
