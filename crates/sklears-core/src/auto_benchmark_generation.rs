//! # Automatic Benchmark Generation System
//!
//! This module provides a sophisticated system for automatically generating comprehensive
//! benchmarks for ML algorithms, including:
//! - Performance regression detection
//! - Scalability analysis
//! - Cross-platform performance validation
//! - Automated performance optimization suggestions
//! - Comparative analysis against baselines

use crate::error::Result;
use quote::quote;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::time::{Duration, Instant};

// ============================================================================
// Core Benchmark Generation Framework
// ============================================================================

/// Configuration for automatic benchmark generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoBenchmarkConfig {
    pub benchmark_types: Vec<BenchmarkType>,
    pub scaling_dimensions: Vec<ScalingDimension>,
    pub performance_targets: PerformanceTargets,
    pub comparison_baselines: Vec<Baseline>,
    pub statistical_config: StatisticalConfig,
    pub output_formats: Vec<OutputFormat>,
    pub regression_detection: RegressionDetectionConfig,
    pub optimization_hints: bool,
}

/// Types of benchmarks to generate
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum BenchmarkType {
    Microbenchmark,       // Single function/operation
    IntegrationBenchmark, // Full algorithm workflow
    ScalabilityBenchmark, // Performance vs. input size
    MemoryBenchmark,      // Memory usage analysis
    LatencyBenchmark,     // Latency distribution
    ThroughputBenchmark,  // Operations per second
    AccuracyBenchmark,    // Accuracy vs. performance trade-offs
    RegressionBenchmark,  // Performance regression detection
    ComparativeBenchmark, // Against other implementations
    StressBenchmark,      // Under high load/large inputs
}

/// Dimension along which to scale benchmarks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalingDimension {
    pub name: String,
    pub parameter_path: String, // Path to the parameter (e.g., "config.num_features")
    pub values: ScalingValues,
    pub expected_complexity: ComplexityClass,
    pub units: String,
}

/// Values to use for scaling benchmarks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ScalingValues {
    Linear { start: f64, end: f64, steps: usize },
    Exponential { start: f64, base: f64, steps: usize },
    Custom(Vec<f64>),
    Fibonacci { max_value: f64 },
    PowersOfTwo { min_power: i32, max_power: i32 },
}

/// Algorithmic complexity classes
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ComplexityClass {
    Constant,       // O(1)
    Logarithmic,    // O(log n)
    Linear,         // O(n)
    Linearithmic,   // O(n log n)
    Quadratic,      // O(n²)
    Cubic,          // O(n³)
    Exponential,    // O(2^n)
    Factorial,      // O(n!)
    Custom(String), // Custom complexity description
}

/// Performance targets and thresholds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceTargets {
    pub max_latency_ms: f64,
    pub min_throughput_ops_sec: f64,
    pub max_memory_mb: f64,
    pub max_accuracy_loss_percent: f64,
    pub regression_threshold_percent: f64,
    pub stability_coefficient_of_variation: f64, // CV = std/mean
}

/// Baseline implementations for comparison
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Baseline {
    pub name: String,
    pub implementation: BaselineType,
    pub expected_performance_ratio: f64, // How much faster/slower than baseline
    pub accuracy_expectation: AccuracyExpectation,
    pub availability: BaselineAvailability,
}

/// Types of baseline implementations
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum BaselineType {
    ScikitLearn,
    NumPy,
    Scipy,
    NativeRust,
    BLAS,
    LAPACK,
    Custom(String),
    Theoretical, // Theoretical lower bound
}

/// Expected accuracy relationship with baseline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AccuracyExpectation {
    Identical,
    WithinTolerance(f64),
    Approximate(f64), // Allowed relative error
    Different,        // Different algorithm, accuracy not comparable
}

/// Availability of baseline for testing
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum BaselineAvailability {
    Always,
    ConditionalOnFeature(String),
    Manual, // Requires manual setup
    Unavailable,
}

/// Statistical configuration for benchmarks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalConfig {
    pub min_iterations: usize,
    pub max_iterations: usize,
    pub warmup_iterations: usize,
    pub confidence_level: f64, // 0.95 for 95% confidence interval
    pub outlier_detection: OutlierDetectionMethod,
    pub measurement_precision: MeasurementPrecision,
}

/// Methods for detecting outliers in benchmark results
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum OutlierDetectionMethod {
    None,
    IQR,            // Interquartile range
    ZScore,         // Z-score based
    ModifiedZScore, // Modified Z-score (using median)
    Isolation,      // Isolation forest
    Custom(String),
}

/// Precision requirements for measurements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeasurementPrecision {
    pub timing_precision_ns: u64,
    pub memory_precision_bytes: u64,
    pub accuracy_precision_digits: u8,
    pub min_relative_precision: f64, // Minimum relative precision required
}

/// Output formats for benchmark results
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum OutputFormat {
    Json,
    Csv,
    Html,
    Markdown,
    PlotlyJson,
    CriterionCompatible,
    Custom(String),
}

/// Configuration for regression detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionDetectionConfig {
    pub enabled: bool,
    pub historical_data_path: String,
    pub regression_threshold_percent: f64,
    pub minimum_effect_size: f64,
    pub statistical_test: StatisticalTest,
    pub alert_on_regression: bool,
}

/// Statistical tests for regression detection
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum StatisticalTest {
    TTest,
    MannWhitneyU,
    WelchTTest,
    Bootstrap,
    PermutationTest,
}

// ============================================================================
// Benchmark Generation Engine
// ============================================================================

/// Main engine for generating benchmarks
pub struct BenchmarkGenerator {
    config: AutoBenchmarkConfig,
    generated_benchmarks: Vec<GeneratedBenchmark>,
    #[allow(dead_code)]
    performance_models: HashMap<String, PerformanceModel>,
}

impl BenchmarkGenerator {
    /// Create new benchmark generator
    pub fn new(config: AutoBenchmarkConfig) -> Self {
        Self {
            config,
            generated_benchmarks: Vec::new(),
            performance_models: HashMap::new(),
        }
    }

    /// Generate benchmarks for a given type
    pub fn generate_for_type<T>(&mut self, type_name: &str) -> Result<Vec<GeneratedBenchmark>> {
        let mut benchmarks = Vec::new();

        for benchmark_type in &self.config.benchmark_types {
            let benchmark = match benchmark_type {
                BenchmarkType::Microbenchmark => self.generate_microbenchmark(type_name)?,
                BenchmarkType::IntegrationBenchmark => {
                    self.generate_integration_benchmark(type_name)?
                }
                BenchmarkType::ScalabilityBenchmark => {
                    self.generate_scalability_benchmark(type_name)?
                }
                BenchmarkType::MemoryBenchmark => self.generate_memory_benchmark(type_name)?,
                BenchmarkType::LatencyBenchmark => self.generate_latency_benchmark(type_name)?,
                BenchmarkType::ThroughputBenchmark => {
                    self.generate_throughput_benchmark(type_name)?
                }
                BenchmarkType::AccuracyBenchmark => self.generate_accuracy_benchmark(type_name)?,
                BenchmarkType::RegressionBenchmark => {
                    self.generate_regression_benchmark(type_name)?
                }
                BenchmarkType::ComparativeBenchmark => {
                    self.generate_comparative_benchmark(type_name)?
                }
                BenchmarkType::StressBenchmark => self.generate_stress_benchmark(type_name)?,
            };

            benchmarks.push(benchmark);
        }

        self.generated_benchmarks.extend(benchmarks.clone());
        Ok(benchmarks)
    }

    /// Generate microbenchmark for individual operations
    fn generate_microbenchmark(&self, type_name: &str) -> Result<GeneratedBenchmark> {
        let benchmark_name = format!("microbench_{}", type_name.to_lowercase());

        let code = quote! {
            use criterion::{criterion_group, criterion_main, Criterion, black_box};

            fn #benchmark_name(c: &mut Criterion) {
                let mut group = c.benchmark_group(stringify!(#type_name));

                // Setup test data
                let test_data = generate_test_data();

                group.bench_function("fit", |b| {
                    let mut model = #type_name::default();
                    b.iter(|| {
                        black_box(model.fit(&test_data.x, &test_data.y).expect("model fitting should succeed"))
                    })
                });

                group.bench_function("predict", |b| {
                    let model = #type_name::default().fit(&test_data.x, &test_data.y).expect("model fitting should succeed");
                    b.iter(|| {
                        black_box(model.predict(&test_data.x_test).expect("prediction should succeed"))
                    })
                });

                group.finish();
            }

            criterion_group!(benches, #benchmark_name);
            criterion_main!(benches);
        }
        .to_string();

        Ok(GeneratedBenchmark {
            name: benchmark_name,
            benchmark_type: BenchmarkType::Microbenchmark,
            code,
            setup_code: self.generate_setup_code(type_name),
            dependencies: self.get_benchmark_dependencies(),
            expected_performance: self
                .estimate_performance(type_name, BenchmarkType::Microbenchmark),
            scaling_analysis: None,
        })
    }

    /// Generate integration benchmark for full workflows
    fn generate_integration_benchmark(&self, type_name: &str) -> Result<GeneratedBenchmark> {
        let benchmark_name = format!("integration_bench_{}", type_name.to_lowercase());

        let code = quote! {
            use criterion::{criterion_group, criterion_main, Criterion, black_box};

            fn #benchmark_name(c: &mut Criterion) {
                let mut group = c.benchmark_group("integration");

                // Full ML pipeline benchmark
                group.bench_function("full_pipeline", |b| {
                    b.iter(|| {
                        // Data loading and preprocessing
                        let (x_train, y_train, x_test, y_test) = load_and_preprocess_data();

                        // Model training
                        let model = #type_name::default()
                            .fit(&x_train, &y_train)
                            .expect("expected valid value");

                        // Prediction and evaluation
                        let predictions = model.predict(&x_test).expect("prediction should succeed");
                        let score = evaluate_predictions(&predictions, &y_test);

                        black_box(score)
                    })
                });

                group.finish();
            }

            criterion_group!(benches, #benchmark_name);
            criterion_main!(benches);
        }
        .to_string();

        Ok(GeneratedBenchmark {
            name: benchmark_name,
            benchmark_type: BenchmarkType::IntegrationBenchmark,
            code,
            setup_code: self.generate_setup_code(type_name),
            dependencies: self.get_benchmark_dependencies(),
            expected_performance: self
                .estimate_performance(type_name, BenchmarkType::IntegrationBenchmark),
            scaling_analysis: None,
        })
    }

    /// Generate scalability benchmark
    fn generate_scalability_benchmark(&self, type_name: &str) -> Result<GeneratedBenchmark> {
        let benchmark_name = format!("scalability_bench_{}", type_name.to_lowercase());

        let scaling_tests = self.config.scaling_dimensions.iter().map(|dim| {
            let values = self.generate_scaling_values(&dim.values);
            let param_name = &dim.name;

            quote! {
                // Benchmark scaling with #param_name
                for &value in &[#(#values),*] {
                    group.bench_with_input(
                        criterion::BenchmarkId::new(#param_name, value),
                        &value,
                        |b, &size| {
                            let test_data = generate_test_data_with_size(size as usize);
                            let mut model = #type_name::default();

                            b.iter(|| {
                                black_box(model.fit(&test_data.x, &test_data.y).expect("model fitting should succeed"))
                            })
                        }
                    );
                }
            }
        });

        let code = quote! {
            use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId, black_box};

            fn #benchmark_name(c: &mut Criterion) {
                let mut group = c.benchmark_group("scalability");

                #(#scaling_tests)*

                group.finish();
            }

            criterion_group!(benches, #benchmark_name);
            criterion_main!(benches);
        }
        .to_string();

        Ok(GeneratedBenchmark {
            name: benchmark_name,
            benchmark_type: BenchmarkType::ScalabilityBenchmark,
            code,
            setup_code: self.generate_setup_code(type_name),
            dependencies: self.get_benchmark_dependencies(),
            expected_performance: self
                .estimate_performance(type_name, BenchmarkType::ScalabilityBenchmark),
            scaling_analysis: Some(self.generate_scaling_analysis()),
        })
    }

    /// Generate memory benchmark
    fn generate_memory_benchmark(&self, type_name: &str) -> Result<GeneratedBenchmark> {
        let benchmark_name = format!("memory_bench_{}", type_name.to_lowercase());

        let code = quote! {
            use criterion::{criterion_group, criterion_main, Criterion, black_box};
            use std::alloc::{GlobalAlloc, Layout, System};
            use std::sync::atomic::{AtomicUsize, Ordering};

            struct MemoryTracker;

            static ALLOCATED: AtomicUsize = AtomicUsize::new(0);

            unsafe impl GlobalAlloc for MemoryTracker {
                unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
                    let ret = System.alloc(layout);
                    if !ret.is_null() {
                        ALLOCATED.fetch_add(layout.size(), Ordering::SeqCst);
                    }
                    ret
                }

                unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
                    System.dealloc(ptr, layout);
                    ALLOCATED.fetch_sub(layout.size(), Ordering::SeqCst);
                }
            }

            #[global_allocator]
            static GLOBAL: MemoryTracker = MemoryTracker;

            fn #benchmark_name(c: &mut Criterion) {
                let mut group = c.benchmark_group("memory");

                group.bench_function("memory_usage", |b| {
                    b.iter_custom(|iters| {
                        let start_memory = ALLOCATED.load(Ordering::SeqCst);
                        let start_time = std::time::Instant::now();

                        for _ in 0..iters {
                            let test_data = generate_test_data();
                            let model = #type_name::default()
                                .fit(&test_data.x, &test_data.y)
                                .expect("expected valid value");
                            black_box(model);
                        }

                        let duration = start_time.elapsed();
                        let end_memory = ALLOCATED.load(Ordering::SeqCst);
                        let memory_used = end_memory.saturating_sub(start_memory);

                        println!("Memory used: {} bytes", memory_used);
                        duration
                    })
                });

                group.finish();
            }

            criterion_group!(benches, #benchmark_name);
            criterion_main!(benches);
        }
        .to_string();

        Ok(GeneratedBenchmark {
            name: benchmark_name,
            benchmark_type: BenchmarkType::MemoryBenchmark,
            code,
            setup_code: self.generate_setup_code(type_name),
            dependencies: self.get_benchmark_dependencies(),
            expected_performance: self
                .estimate_performance(type_name, BenchmarkType::MemoryBenchmark),
            scaling_analysis: None,
        })
    }

    /// Generate remaining benchmark types (simplified for brevity)
    fn generate_latency_benchmark(&self, type_name: &str) -> Result<GeneratedBenchmark> {
        self.generate_simple_benchmark(type_name, BenchmarkType::LatencyBenchmark, "latency")
    }

    fn generate_throughput_benchmark(&self, type_name: &str) -> Result<GeneratedBenchmark> {
        self.generate_simple_benchmark(type_name, BenchmarkType::ThroughputBenchmark, "throughput")
    }

    fn generate_accuracy_benchmark(&self, type_name: &str) -> Result<GeneratedBenchmark> {
        self.generate_simple_benchmark(type_name, BenchmarkType::AccuracyBenchmark, "accuracy")
    }

    fn generate_regression_benchmark(&self, type_name: &str) -> Result<GeneratedBenchmark> {
        self.generate_simple_benchmark(type_name, BenchmarkType::RegressionBenchmark, "regression")
    }

    fn generate_comparative_benchmark(&self, type_name: &str) -> Result<GeneratedBenchmark> {
        self.generate_simple_benchmark(
            type_name,
            BenchmarkType::ComparativeBenchmark,
            "comparative",
        )
    }

    fn generate_stress_benchmark(&self, type_name: &str) -> Result<GeneratedBenchmark> {
        self.generate_simple_benchmark(type_name, BenchmarkType::StressBenchmark, "stress")
    }

    /// Helper to generate simple benchmarks
    fn generate_simple_benchmark(
        &self,
        type_name: &str,
        benchmark_type: BenchmarkType,
        prefix: &str,
    ) -> Result<GeneratedBenchmark> {
        let benchmark_name = format!("{}_{}", prefix, type_name.to_lowercase());

        let code = format!(
            r#"
            use criterion::{{criterion_group, criterion_main, Criterion, black_box}};

            fn {}(c: &mut Criterion) {{
                let mut group = c.benchmark_group("{}");

                group.bench_function("operation", |b| {{
                    let test_data = generate_test_data();
                    let mut model = {}::default();

                    b.iter(|| {{
                        black_box(model.fit(&test_data.x, &test_data.y).expect("model fitting should succeed"))
                    }})
                }});

                group.finish();
            }}

            criterion_group!(benches, {});
            criterion_main!(benches);
            "#,
            benchmark_name, prefix, type_name, benchmark_name
        );

        Ok(GeneratedBenchmark {
            name: benchmark_name,
            benchmark_type: benchmark_type.clone(),
            code,
            setup_code: self.generate_setup_code(type_name),
            dependencies: self.get_benchmark_dependencies(),
            expected_performance: self.estimate_performance(type_name, benchmark_type.clone()),
            scaling_analysis: None,
        })
    }

    /// Generate setup code for benchmarks
    fn generate_setup_code(&self, type_name: &str) -> String {
        format!(
            r#"
            use {}::*;

            struct TestData {{
                x: Array2<f64>,
                y: Array1<f64>,
                x_test: Array2<f64>,
                y_test: Array1<f64>,
            }}

            fn generate_test_data() -> TestData {{
                let n_samples = 1000;
                let n_features = 20;

                let x = Array2::random((n_samples, n_features), Normal::new(0.0, 1.0).unwrap_or_else(|_| Normal::new(0.0, 1.0).expect("default normal distribution")));
                let y = Array1::random(n_samples, Normal::new(0.0, 1.0).unwrap_or_else(|_| Normal::new(0.0, 1.0).expect("default normal distribution")));
                let x_test = Array2::random((100, n_features), Normal::new(0.0, 1.0).unwrap_or_else(|_| Normal::new(0.0, 1.0).expect("default normal distribution")));
                let y_test = Array1::random(100, Normal::new(0.0, 1.0).unwrap_or_else(|_| Normal::new(0.0, 1.0).expect("default normal distribution")));

                TestData {{ x, y, x_test, y_test }}
            }}

            fn generate_test_data_with_size(size: usize) -> TestData {{
                let n_features = 20;

                let x = Array2::random((size, n_features), Normal::new(0.0, 1.0).unwrap_or_else(|_| Normal::new(0.0, 1.0).expect("default normal distribution")));
                let y = Array1::random(size, Normal::new(0.0, 1.0).unwrap_or_else(|_| Normal::new(0.0, 1.0).expect("default normal distribution")));
                let x_test = Array2::random((size / 10, n_features), Normal::new(0.0, 1.0).unwrap_or_else(|_| Normal::new(0.0, 1.0).expect("default normal distribution")));
                let y_test = Array1::random(size / 10, Normal::new(0.0, 1.0).unwrap_or_else(|_| Normal::new(0.0, 1.0).expect("default normal distribution")));

                TestData {{ x, y, x_test, y_test }}
            }}

            fn load_and_preprocess_data() -> (Array2<f64>, Array1<f64>, Array2<f64>, Array1<f64>) {{
                let test_data = generate_test_data();
                (test_data.x, test_data.y, test_data.x_test, test_data.y_test)
            }}

            fn evaluate_predictions(predictions: &Array1<f64>, y_true: &Array1<f64>) -> f64 {{
                // Mean squared error
                let diff = predictions - y_true;
                diff.mapv(|x| x * x).mean().unwrap_or_default()
            }}
            "#,
            type_name
        )
    }

    /// Get dependencies needed for benchmarks
    fn get_benchmark_dependencies(&self) -> Vec<String> {
        vec![
            "criterion".to_string(),
            "ndarray".to_string(),
            "ndarray-rand".to_string(),
            "rand_distr".to_string(),
        ]
    }

    /// Estimate performance for different benchmark types
    fn estimate_performance(
        &self,
        _type_name: &str,
        benchmark_type: BenchmarkType,
    ) -> PerformanceEstimate {
        // This would use historical data or heuristics to estimate performance
        PerformanceEstimate {
            expected_latency_ms: match benchmark_type {
                BenchmarkType::Microbenchmark => 1.0,
                BenchmarkType::IntegrationBenchmark => 100.0,
                BenchmarkType::ScalabilityBenchmark => 50.0,
                _ => 10.0,
            },
            expected_throughput_ops_sec: 1000.0,
            expected_memory_mb: 10.0,
            confidence_interval: 0.95,
        }
    }

    /// Generate scaling values from configuration
    fn generate_scaling_values(&self, scaling_values: &ScalingValues) -> Vec<f64> {
        match scaling_values {
            ScalingValues::Linear { start, end, steps } => {
                let step_size = (end - start) / (*steps as f64 - 1.0);
                (0..*steps)
                    .map(|i| start + (i as f64) * step_size)
                    .collect()
            }
            ScalingValues::Exponential { start, base, steps } => {
                (0..*steps).map(|i| start * base.powi(i as i32)).collect()
            }
            ScalingValues::Custom(values) => values.clone(),
            ScalingValues::Fibonacci { max_value } => {
                let mut fib = vec![1.0, 1.0];
                while fib[fib.len() - 1] < *max_value {
                    let next = fib[fib.len() - 1] + fib[fib.len() - 2];
                    fib.push(next);
                }
                fib
            }
            ScalingValues::PowersOfTwo {
                min_power,
                max_power,
            } => (*min_power..=*max_power).map(|p| 2.0_f64.powi(p)).collect(),
        }
    }

    /// Generate scaling analysis
    fn generate_scaling_analysis(&self) -> ScalingAnalysis {
        ScalingAnalysis {
            complexity_models: self
                .config
                .scaling_dimensions
                .iter()
                .map(|dim| {
                    ComplexityModel {
                        dimension: dim.name.clone(),
                        expected_complexity: dim.expected_complexity.clone(),
                        coefficients: vec![1.0, 0.1, 0.01], // Mock coefficients
                        r_squared: 0.95,
                    }
                })
                .collect(),
            performance_predictions: HashMap::new(),
            optimization_recommendations: vec![],
        }
    }
}

// ============================================================================
// Generated Benchmark Structure
// ============================================================================

/// A generated benchmark with all necessary components
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratedBenchmark {
    pub name: String,
    pub benchmark_type: BenchmarkType,
    pub code: String,
    pub setup_code: String,
    pub dependencies: Vec<String>,
    pub expected_performance: PerformanceEstimate,
    pub scaling_analysis: Option<ScalingAnalysis>,
}

/// Performance estimate for a benchmark
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceEstimate {
    pub expected_latency_ms: f64,
    pub expected_throughput_ops_sec: f64,
    pub expected_memory_mb: f64,
    pub confidence_interval: f64,
}

/// Analysis of scaling behavior
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalingAnalysis {
    pub complexity_models: Vec<ComplexityModel>,
    pub performance_predictions: HashMap<String, f64>,
    pub optimization_recommendations: Vec<String>,
}

/// Model of algorithmic complexity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplexityModel {
    pub dimension: String,
    pub expected_complexity: ComplexityClass,
    pub coefficients: Vec<f64>, // Polynomial coefficients
    pub r_squared: f64,         // Goodness of fit
}

/// Performance model for predicting behavior
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceModel {
    pub algorithm_name: String,
    pub complexity_class: ComplexityClass,
    pub base_performance: f64,
    pub scaling_factors: HashMap<String, f64>,
    pub confidence: f64,
}

// ============================================================================
// Benchmark Execution and Analysis
// ============================================================================

/// Executor for generated benchmarks
pub struct BenchmarkExecutor {
    results: Vec<BenchmarkResult>,
    regression_detector: RegressionDetector,
}

impl Default for BenchmarkExecutor {
    fn default() -> Self {
        Self::new()
    }
}

impl BenchmarkExecutor {
    /// Create new benchmark executor
    pub fn new() -> Self {
        Self {
            results: Vec::new(),
            regression_detector: RegressionDetector::new(),
        }
    }

    /// Execute a generated benchmark
    pub fn execute_benchmark(&mut self, benchmark: &GeneratedBenchmark) -> Result<BenchmarkResult> {
        let _start_time = Instant::now();

        // This would actually compile and run the benchmark
        // For now, we'll simulate execution
        let execution_time = Duration::from_millis(100);

        let result = BenchmarkResult {
            benchmark_name: benchmark.name.clone(),
            benchmark_type: benchmark.benchmark_type.clone(),
            execution_time,
            memory_usage_bytes: 1024 * 1024, // 1 MB
            throughput_ops_sec: 1000.0,
            accuracy_score: Some(0.95),
            regression_detected: false,
            performance_vs_baseline: 1.2, // 20% faster than baseline
            statistical_significance: 0.99,
        };

        self.results.push(result.clone());
        Ok(result)
    }

    /// Analyze benchmark results for regressions
    pub fn analyze_results(&mut self) -> AnalysisReport {
        let regressions = self.regression_detector.detect_regressions(&self.results);
        let recommendations = self.generate_optimization_recommendations();

        AnalysisReport {
            total_benchmarks: self.results.len(),
            regressions_detected: regressions.len(),
            average_performance_change: self.calculate_average_performance_change(),
            recommendations,
            detailed_results: self.results.clone(),
        }
    }

    /// Generate optimization recommendations
    fn generate_optimization_recommendations(&self) -> Vec<OptimizationRecommendation> {
        let mut recommendations = Vec::new();

        for result in &self.results {
            if result.performance_vs_baseline < 1.0 {
                recommendations.push(OptimizationRecommendation {
                    benchmark_name: result.benchmark_name.clone(),
                    issue: "Performance below baseline".to_string(),
                    suggestion: "Consider algorithm optimization or SIMD vectorization".to_string(),
                    expected_improvement: 1.5,
                    implementation_effort: ImplementationEffort::Medium,
                });
            }
        }

        recommendations
    }

    /// Calculate average performance change
    fn calculate_average_performance_change(&self) -> f64 {
        if self.results.is_empty() {
            return 0.0;
        }

        let sum: f64 = self.results.iter().map(|r| r.performance_vs_baseline).sum();
        sum / self.results.len() as f64
    }
}

/// Result of executing a benchmark
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    pub benchmark_name: String,
    pub benchmark_type: BenchmarkType,
    pub execution_time: Duration,
    pub memory_usage_bytes: u64,
    pub throughput_ops_sec: f64,
    pub accuracy_score: Option<f64>,
    pub regression_detected: bool,
    pub performance_vs_baseline: f64, // Ratio: new_performance / baseline_performance
    pub statistical_significance: f64,
}

/// Analysis report for all benchmarks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisReport {
    pub total_benchmarks: usize,
    pub regressions_detected: usize,
    pub average_performance_change: f64,
    pub recommendations: Vec<OptimizationRecommendation>,
    pub detailed_results: Vec<BenchmarkResult>,
}

/// Optimization recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationRecommendation {
    pub benchmark_name: String,
    pub issue: String,
    pub suggestion: String,
    pub expected_improvement: f64,
    pub implementation_effort: ImplementationEffort,
}

/// Effort required to implement optimization
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ImplementationEffort {
    Low,
    Medium,
    High,
    Expert,
}

/// Regression detector
pub struct RegressionDetector {
    #[allow(dead_code)]
    historical_data: HashMap<String, Vec<f64>>,
    threshold: f64,
}

impl Default for RegressionDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl RegressionDetector {
    pub fn new() -> Self {
        Self {
            historical_data: HashMap::new(),
            threshold: 0.05, // 5% regression threshold
        }
    }

    pub fn detect_regressions(&self, results: &[BenchmarkResult]) -> Vec<RegressionAlert> {
        let mut alerts = Vec::new();

        for result in results {
            if result.performance_vs_baseline < (1.0 - self.threshold) {
                alerts.push(RegressionAlert {
                    benchmark_name: result.benchmark_name.clone(),
                    performance_drop_percent: (1.0 - result.performance_vs_baseline) * 100.0,
                    severity: if result.performance_vs_baseline < 0.8 {
                        RegressionSeverity::High
                    } else {
                        RegressionSeverity::Medium
                    },
                    recommendation: "Investigate performance regression".to_string(),
                });
            }
        }

        alerts
    }
}

/// Regression alert
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionAlert {
    pub benchmark_name: String,
    pub performance_drop_percent: f64,
    pub severity: RegressionSeverity,
    pub recommendation: String,
}

/// Severity of performance regression
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RegressionSeverity {
    Low,
    Medium,
    High,
    Critical,
}

// ============================================================================
// Macro Interface
// ============================================================================

/// Macro to automatically generate benchmarks for a type
#[macro_export]
macro_rules! auto_benchmark {
    ($type:ty) => {
        $crate::auto_benchmark_generation::generate_benchmarks_for_type::<$type>()
    };
    ($type:ty, $config:expr) => {
        $crate::auto_benchmark_generation::generate_benchmarks_with_config::<$type>($config)
    };
}

/// Generate benchmarks for a type using default configuration
pub fn generate_benchmarks_for_type<T>() -> Result<Vec<GeneratedBenchmark>> {
    let config = AutoBenchmarkConfig::default();
    let mut generator = BenchmarkGenerator::new(config);
    generator.generate_for_type::<T>(std::any::type_name::<T>())
}

/// Generate benchmarks for a type with custom configuration
pub fn generate_benchmarks_with_config<T>(
    config: AutoBenchmarkConfig,
) -> Result<Vec<GeneratedBenchmark>> {
    let mut generator = BenchmarkGenerator::new(config);
    generator.generate_for_type::<T>(std::any::type_name::<T>())
}

// ============================================================================
// Default Implementations
// ============================================================================

impl Default for AutoBenchmarkConfig {
    fn default() -> Self {
        Self {
            benchmark_types: vec![
                BenchmarkType::Microbenchmark,
                BenchmarkType::IntegrationBenchmark,
                BenchmarkType::ScalabilityBenchmark,
            ],
            scaling_dimensions: vec![
                ScalingDimension {
                    name: "n_samples".to_string(),
                    parameter_path: "n_samples".to_string(),
                    values: ScalingValues::PowersOfTwo {
                        min_power: 6,
                        max_power: 16,
                    }, // 64 to 65536
                    expected_complexity: ComplexityClass::Linear,
                    units: "samples".to_string(),
                },
                ScalingDimension {
                    name: "n_features".to_string(),
                    parameter_path: "n_features".to_string(),
                    values: ScalingValues::Linear {
                        start: 10.0,
                        end: 1000.0,
                        steps: 10,
                    },
                    expected_complexity: ComplexityClass::Linear,
                    units: "features".to_string(),
                },
            ],
            performance_targets: PerformanceTargets::default(),
            comparison_baselines: vec![Baseline {
                name: "scikit-learn".to_string(),
                implementation: BaselineType::ScikitLearn,
                expected_performance_ratio: 3.0, // 3x faster
                accuracy_expectation: AccuracyExpectation::WithinTolerance(0.01),
                availability: BaselineAvailability::ConditionalOnFeature(
                    "python-comparison".to_string(),
                ),
            }],
            statistical_config: StatisticalConfig::default(),
            output_formats: vec![OutputFormat::Json, OutputFormat::Html],
            regression_detection: RegressionDetectionConfig::default(),
            optimization_hints: true,
        }
    }
}

impl Default for PerformanceTargets {
    fn default() -> Self {
        Self {
            max_latency_ms: 100.0,
            min_throughput_ops_sec: 1000.0,
            max_memory_mb: 1024.0,
            max_accuracy_loss_percent: 1.0,
            regression_threshold_percent: 5.0,
            stability_coefficient_of_variation: 0.1, // 10% CV
        }
    }
}

impl Default for StatisticalConfig {
    fn default() -> Self {
        Self {
            min_iterations: 10,
            max_iterations: 1000,
            warmup_iterations: 3,
            confidence_level: 0.95,
            outlier_detection: OutlierDetectionMethod::IQR,
            measurement_precision: MeasurementPrecision {
                timing_precision_ns: 1000,    // 1 microsecond
                memory_precision_bytes: 1024, // 1 KB
                accuracy_precision_digits: 6,
                min_relative_precision: 0.01, // 1%
            },
        }
    }
}

impl Default for RegressionDetectionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            historical_data_path: "benchmark_history.json".to_string(),
            regression_threshold_percent: 5.0,
            minimum_effect_size: 0.1,
            statistical_test: StatisticalTest::TTest,
            alert_on_regression: true,
        }
    }
}

impl fmt::Display for ComplexityClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ComplexityClass::Constant => write!(f, "O(1)"),
            ComplexityClass::Logarithmic => write!(f, "O(log n)"),
            ComplexityClass::Linear => write!(f, "O(n)"),
            ComplexityClass::Linearithmic => write!(f, "O(n log n)"),
            ComplexityClass::Quadratic => write!(f, "O(n²)"),
            ComplexityClass::Cubic => write!(f, "O(n³)"),
            ComplexityClass::Exponential => write!(f, "O(2^n)"),
            ComplexityClass::Factorial => write!(f, "O(n!)"),
            ComplexityClass::Custom(s) => write!(f, "O({})", s),
        }
    }
}

impl fmt::Display for BenchmarkType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BenchmarkType::Microbenchmark => write!(f, "Microbenchmark"),
            BenchmarkType::IntegrationBenchmark => write!(f, "Integration Benchmark"),
            BenchmarkType::ScalabilityBenchmark => write!(f, "Scalability Benchmark"),
            BenchmarkType::MemoryBenchmark => write!(f, "Memory Benchmark"),
            BenchmarkType::LatencyBenchmark => write!(f, "Latency Benchmark"),
            BenchmarkType::ThroughputBenchmark => write!(f, "Throughput Benchmark"),
            BenchmarkType::AccuracyBenchmark => write!(f, "Accuracy Benchmark"),
            BenchmarkType::RegressionBenchmark => write!(f, "Regression Benchmark"),
            BenchmarkType::ComparativeBenchmark => write!(f, "Comparative Benchmark"),
            BenchmarkType::StressBenchmark => write!(f, "Stress Benchmark"),
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_auto_benchmark_config_default() {
        let config = AutoBenchmarkConfig::default();
        assert!(!config.benchmark_types.is_empty());
        assert!(!config.scaling_dimensions.is_empty());
        assert!(config.optimization_hints);
    }

    #[test]
    fn test_scaling_values_generation() {
        let generator = BenchmarkGenerator::new(AutoBenchmarkConfig::default());

        let linear_values = generator.generate_scaling_values(&ScalingValues::Linear {
            start: 0.0,
            end: 10.0,
            steps: 5,
        });
        assert_eq!(linear_values.len(), 5);
        assert_eq!(linear_values[0], 0.0);
        assert_eq!(linear_values[4], 10.0);

        let powers_of_two = generator.generate_scaling_values(&ScalingValues::PowersOfTwo {
            min_power: 2,
            max_power: 4,
        });
        assert_eq!(powers_of_two, vec![4.0, 8.0, 16.0]);
    }

    #[test]
    fn test_benchmark_generation() {
        let config = AutoBenchmarkConfig::default();
        let mut generator = BenchmarkGenerator::new(config);

        let benchmarks = generator
            .generate_for_type::<String>("TestType")
            .expect("expected valid value");
        assert!(!benchmarks.is_empty());

        for benchmark in &benchmarks {
            assert!(!benchmark.name.is_empty());
            assert!(!benchmark.code.is_empty());
            assert!(!benchmark.setup_code.is_empty());
        }
    }

    #[test]
    fn test_benchmark_executor() {
        let mut executor = BenchmarkExecutor::new();

        let benchmark = GeneratedBenchmark {
            name: "test_benchmark".to_string(),
            benchmark_type: BenchmarkType::Microbenchmark,
            code: "mock code".to_string(),
            setup_code: "mock setup".to_string(),
            dependencies: vec![],
            expected_performance: PerformanceEstimate {
                expected_latency_ms: 1.0,
                expected_throughput_ops_sec: 1000.0,
                expected_memory_mb: 1.0,
                confidence_interval: 0.95,
            },
            scaling_analysis: None,
        };

        let result = executor
            .execute_benchmark(&benchmark)
            .expect("execute_benchmark should succeed");
        assert_eq!(result.benchmark_name, "test_benchmark");
        assert_eq!(result.benchmark_type, BenchmarkType::Microbenchmark);
    }

    #[test]
    fn test_regression_detection() {
        let detector = RegressionDetector::new();

        let results = vec![
            BenchmarkResult {
                benchmark_name: "test1".to_string(),
                benchmark_type: BenchmarkType::Microbenchmark,
                execution_time: Duration::from_millis(100),
                memory_usage_bytes: 1024,
                throughput_ops_sec: 1000.0,
                accuracy_score: Some(0.95),
                regression_detected: false,
                performance_vs_baseline: 0.8, // 20% slower - regression
                statistical_significance: 0.99,
            },
            BenchmarkResult {
                benchmark_name: "test2".to_string(),
                benchmark_type: BenchmarkType::Microbenchmark,
                execution_time: Duration::from_millis(50),
                memory_usage_bytes: 512,
                throughput_ops_sec: 2000.0,
                accuracy_score: Some(0.97),
                regression_detected: false,
                performance_vs_baseline: 1.2, // 20% faster - improvement
                statistical_significance: 0.99,
            },
        ];

        let alerts = detector.detect_regressions(&results);
        assert_eq!(alerts.len(), 1); // Only test1 should trigger regression alert
        assert_eq!(alerts[0].benchmark_name, "test1");
    }

    #[test]
    fn test_complexity_class_display() {
        assert_eq!(format!("{}", ComplexityClass::Linear), "O(n)");
        assert_eq!(format!("{}", ComplexityClass::Quadratic), "O(n²)");
        assert_eq!(
            format!("{}", ComplexityClass::Custom("n log n".to_string())),
            "O(n log n)"
        );
    }
}
