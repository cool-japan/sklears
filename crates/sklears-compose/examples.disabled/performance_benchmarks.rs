//! Performance Benchmarks for Pipeline Composition
//!
//! This example provides comprehensive benchmarks comparing different pipeline
//! composition strategies and their performance characteristics. It demonstrates:
//!
//! - Sequential vs parallel pipeline execution
//! - Memory-efficient vs standard pipeline execution
//! - SIMD-optimized operations vs standard operations
//! - Different feature union strategies
//! - Caching vs non-caching approaches
//!
//! Run with: cargo run --example performance_benchmarks --release

use scirs2_autograd::ndarray::{Array1, Array2, Axis};
use scirs2_core::parallel::{par_chunks, ParallelExecutor};
use scirs2_core::random::{rng, Random};
use scirs2_core::simd::{auto_vectorize, SimdOps};
use sklears_compose::{
    advanced_pipeline::{CachedPipeline, MemoryEfficientPipeline},
    benchmarking::{BenchmarkConfig, BenchmarkRunner},
    monitoring::PipelineMonitor,
    simd_optimizations::{SIMDPolynomialFeatures, SIMDStandardScaler},
    FeatureUnion, Pipeline, PipelineBuilder,
};
use sklears_core::{
    error::Result as SklResult,
    traits::{Fit, Predict, Transform},
    types::Float,
};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Benchmark configuration for different test scenarios
#[derive(Debug, Clone)]
struct BenchmarkSuite {
    small_dataset: (Array2<Float>, Array1<Float>),
    medium_dataset: (Array2<Float>, Array1<Float>),
    large_dataset: (Array2<Float>, Array1<Float>),
}

impl BenchmarkSuite {
    /// Create benchmark suite with different dataset sizes
    fn new() -> SklResult<Self> {
        let mut rng = Random::new(42)?;

        Ok(Self {
            small_dataset: Self::generate_dataset(&mut rng, 100, 10)?,
            medium_dataset: Self::generate_dataset(&mut rng, 1_000, 50)?,
            large_dataset: Self::generate_dataset(&mut rng, 10_000, 100)?,
        })
    }

    fn generate_dataset(
        rng: &mut Random,
        n_samples: usize,
        n_features: usize,
    ) -> SklResult<(Array2<Float>, Array1<Float>)> {
        let mut X = Array2::<Float>::zeros((n_samples, n_features));
        let mut y = Array1::<Float>::zeros(n_samples);

        // Generate features
        for i in 0..n_samples {
            for j in 0..n_features {
                X[[i, j]] = rng.normal(0.0, 1.0)?;
            }
        }

        // Generate target as linear combination with noise
        for i in 0..n_samples {
            let mut target = 0.0;
            for j in 0..n_features {
                target += X[[i, j]] * (j as Float + 1.0) / n_features as Float;
            }
            target += rng.normal(0.0, 0.1)?;
            y[i] = target;
        }

        Ok((X, y))
    }
}

/// Standard transformer for comparison
#[derive(Debug, Clone)]
struct StandardTransformer {
    scale_factor: Float,
}

impl StandardTransformer {
    fn new(scale_factor: Float) -> Self {
        Self { scale_factor }
    }
}

impl Transform for StandardTransformer {
    type Input = Array2<Float>;
    type Output = Array2<Float>;

    fn transform(&self, x: &Self::Input) -> SklResult<Self::Output> {
        Ok(x * self.scale_factor)
    }
}

impl Fit for StandardTransformer {
    type Input = Array2<Float>;
    type Target = Array1<Float>;
    type Fitted = Self;

    fn fit(self, _x: &Self::Input, _y: Option<&Self::Target>) -> SklResult<Self::Fitted> {
        Ok(self)
    }
}

/// SIMD-optimized transformer for performance comparison
#[derive(Debug, Clone)]
struct SIMDTransformer {
    scale_factor: Float,
}

impl SIMDTransformer {
    fn new(scale_factor: Float) -> Self {
        Self { scale_factor }
    }
}

impl Transform for SIMDTransformer {
    type Input = Array2<Float>;
    type Output = Array2<Float>;

    fn transform(&self, x: &Self::Input) -> SklResult<Self::Output> {
        // Simulate SIMD operations for demonstration
        let mut result = x.clone();

        // Use chunked processing to simulate SIMD benefits
        for mut chunk in result.axis_chunks_iter_mut(Axis(0), 4) {
            chunk *= self.scale_factor;
        }

        Ok(result)
    }
}

impl Fit for SIMDTransformer {
    type Input = Array2<Float>;
    type Target = Array1<Float>;
    type Fitted = Self;

    fn fit(self, _x: &Self::Input, _y: Option<&Self::Target>) -> SklResult<Self::Fitted> {
        Ok(self)
    }
}

/// Simple predictor for benchmarking
#[derive(Debug, Clone)]
struct SimplePredictor {
    weights: Option<Array1<Float>>,
}

impl SimplePredictor {
    fn new() -> Self {
        Self { weights: None }
    }
}

impl Fit for SimplePredictor {
    type Input = Array2<Float>;
    type Target = Array1<Float>;
    type Fitted = Self;

    fn fit(mut self, x: &Self::Input, y: &Self::Target) -> SklResult<Self::Fitted> {
        // Simple average weights
        let n_features = x.ncols();
        let mut weights = Array1::<Float>::zeros(n_features);

        for j in 0..n_features {
            weights[j] = 1.0 / n_features as Float;
        }

        self.weights = Some(weights);
        Ok(self)
    }
}

impl Predict for SimplePredictor {
    type Input = Array2<Float>;
    type Output = Array1<Float>;

    fn predict(&self, x: &Self::Input) -> SklResult<Self::Output> {
        if let Some(ref weights) = self.weights {
            Ok(x.dot(weights))
        } else {
            Err(sklears_core::error::SklearsError::InvalidState(
                "Predictor not fitted".to_string(),
            ))
        }
    }
}

/// Benchmark results aggregator
#[derive(Debug)]
struct BenchmarkResults {
    results: HashMap<String, BenchmarkMetrics>,
}

#[derive(Debug, Clone)]
struct BenchmarkMetrics {
    mean_duration: Duration,
    std_deviation: Duration,
    throughput_samples_per_sec: f64,
    memory_usage_mb: f64,
    iterations: usize,
}

impl BenchmarkResults {
    fn new() -> Self {
        Self {
            results: HashMap::new(),
        }
    }

    fn add_result(&mut self, name: String, metrics: BenchmarkMetrics) {
        self.results.insert(name, metrics);
    }

    fn print_comparison(&self) {
        println!("\n📊 Benchmark Results Comparison");
        println!("{}", "=".repeat(80));

        println!(
            "{:<25} {:<15} {:<15} {:<15} {:<10}",
            "Method", "Mean Time", "Throughput", "Memory (MB)", "Iterations"
        );
        println!("{}", "-".repeat(80));

        let mut sorted_results: Vec<_> = self.results.iter().collect();
        sorted_results.sort_by(|a, b| a.1.mean_duration.cmp(&b.1.mean_duration));

        for (name, metrics) in sorted_results {
            println!(
                "{:<25} {:>12.2?} {:>12.0}/s {:>12.1} MB {:>10}",
                name,
                metrics.mean_duration,
                metrics.throughput_samples_per_sec,
                metrics.memory_usage_mb,
                metrics.iterations
            );
        }

        // Find best and worst performers
        if let (Some(best), Some(worst)) = (
            self.results
                .iter()
                .min_by(|a, b| a.1.mean_duration.cmp(&b.1.mean_duration)),
            self.results
                .iter()
                .max_by(|a, b| a.1.mean_duration.cmp(&b.1.mean_duration)),
        ) {
            let speedup =
                worst.1.mean_duration.as_nanos() as f64 / best.1.mean_duration.as_nanos() as f64;
            println!("\n🏆 Best: {} | 🐌 Worst: {}", best.0, worst.0);
            println!("⚡ Speedup: {:.2}x faster", speedup);
        }
    }
}

/// Run benchmarks for different pipeline strategies
fn main() -> SklResult<()> {
    println!("🚀 Performance Benchmarks for Pipeline Composition");
    println!("{}", "=".repeat(60));

    let benchmark_suite = BenchmarkSuite::new()?;
    let mut results = BenchmarkResults::new();

    // Benchmark different pipeline strategies
    benchmark_basic_pipeline(&benchmark_suite, &mut results)?;
    benchmark_simd_pipeline(&benchmark_suite, &mut results)?;
    benchmark_parallel_pipeline(&benchmark_suite, &mut results)?;
    benchmark_memory_efficient_pipeline(&benchmark_suite, &mut results)?;
    benchmark_cached_pipeline(&benchmark_suite, &mut results)?;
    benchmark_feature_union_strategies(&benchmark_suite, &mut results)?;

    results.print_comparison();
    print_recommendations();

    Ok(())
}

/// Benchmark basic sequential pipeline
fn benchmark_basic_pipeline(
    suite: &BenchmarkSuite,
    results: &mut BenchmarkResults,
) -> SklResult<()> {
    println!("\n🔧 Benchmarking Basic Sequential Pipeline");
    println!("{}", "-".repeat(50));

    let (X, y) = &suite.medium_dataset;
    let iterations = 100;
    let mut durations = Vec::new();

    // Warm up
    for _ in 0..5 {
        let pipeline = Pipeline::builder()
            .step("transformer1", Box::new(StandardTransformer::new(0.5)))
            .step("transformer2", Box::new(StandardTransformer::new(2.0)))
            .estimator(Box::new(SimplePredictor::new()))
            .build();

        let _ = pipeline.fit(X, Some(y))?.predict(X)?;
    }

    // Actual benchmarking
    for _ in 0..iterations {
        let start = Instant::now();

        let pipeline = Pipeline::builder()
            .step("transformer1", Box::new(StandardTransformer::new(0.5)))
            .step("transformer2", Box::new(StandardTransformer::new(2.0)))
            .estimator(Box::new(SimplePredictor::new()))
            .build();

        let fitted = pipeline.fit(X, Some(y))?;
        let _ = fitted.predict(X)?;

        durations.push(start.elapsed());
    }

    let metrics = calculate_metrics(&durations, X.nrows(), iterations);
    results.add_result("Basic Pipeline".to_string(), metrics);

    println!(
        "✅ Basic pipeline: {:?} avg",
        durations.iter().sum::<Duration>() / durations.len() as u32
    );

    Ok(())
}

/// Benchmark SIMD-optimized pipeline
fn benchmark_simd_pipeline(
    suite: &BenchmarkSuite,
    results: &mut BenchmarkResults,
) -> SklResult<()> {
    println!("\n⚡ Benchmarking SIMD-Optimized Pipeline");
    println!("{}", "-".repeat(50));

    let (X, y) = &suite.medium_dataset;
    let iterations = 100;
    let mut durations = Vec::new();

    // Warm up
    for _ in 0..5 {
        let pipeline = Pipeline::builder()
            .step("simd_transformer1", Box::new(SIMDTransformer::new(0.5)))
            .step("simd_transformer2", Box::new(SIMDTransformer::new(2.0)))
            .estimator(Box::new(SimplePredictor::new()))
            .build();

        let _ = pipeline.fit(X, Some(y))?.predict(X)?;
    }

    // Actual benchmarking
    for _ in 0..iterations {
        let start = Instant::now();

        let pipeline = Pipeline::builder()
            .step("simd_transformer1", Box::new(SIMDTransformer::new(0.5)))
            .step("simd_transformer2", Box::new(SIMDTransformer::new(2.0)))
            .estimator(Box::new(SimplePredictor::new()))
            .build();

        let fitted = pipeline.fit(X, Some(y))?;
        let _ = fitted.predict(X)?;

        durations.push(start.elapsed());
    }

    let metrics = calculate_metrics(&durations, X.nrows(), iterations);
    results.add_result("SIMD Pipeline".to_string(), metrics);

    println!(
        "✅ SIMD pipeline: {:?} avg",
        durations.iter().sum::<Duration>() / durations.len() as u32
    );

    Ok(())
}

/// Benchmark parallel pipeline execution
fn benchmark_parallel_pipeline(
    suite: &BenchmarkSuite,
    results: &mut BenchmarkResults,
) -> SklResult<()> {
    println!("\n🔀 Benchmarking Parallel Pipeline");
    println!("{}", "-".repeat(50));

    let (X, y) = &suite.large_dataset; // Use larger dataset for parallel benefits
    let iterations = 50; // Fewer iterations due to larger dataset
    let mut durations = Vec::new();

    // Actual benchmarking
    for _ in 0..iterations {
        let start = Instant::now();

        // Simulate parallel processing by using chunked operations
        let chunk_size = X.nrows() / 4; // Simulate 4-thread parallelism
        let mut results_vec = Vec::new();

        for chunk_start in (0..X.nrows()).step_by(chunk_size) {
            let chunk_end = (chunk_start + chunk_size).min(X.nrows());
            let x_chunk = X.slice(s![chunk_start..chunk_end, ..]);
            let y_chunk = y.slice(s![chunk_start..chunk_end]);

            let pipeline = Pipeline::builder()
                .step("transformer", Box::new(StandardTransformer::new(1.5)))
                .estimator(Box::new(SimplePredictor::new()))
                .build();

            let fitted = pipeline.fit(&x_chunk.to_owned(), Some(&y_chunk.to_owned()))?;
            let pred = fitted.predict(&x_chunk.to_owned())?;
            results_vec.push(pred);
        }

        durations.push(start.elapsed());
    }

    let metrics = calculate_metrics(&durations, X.nrows(), iterations);
    results.add_result("Parallel Pipeline".to_string(), metrics);

    println!(
        "✅ Parallel pipeline: {:?} avg",
        durations.iter().sum::<Duration>() / durations.len() as u32
    );

    Ok(())
}

/// Benchmark memory-efficient pipeline
fn benchmark_memory_efficient_pipeline(
    suite: &BenchmarkSuite,
    results: &mut BenchmarkResults,
) -> SklResult<()> {
    println!("\n💾 Benchmarking Memory-Efficient Pipeline");
    println!("{}", "-".repeat(50));

    let (X, y) = &suite.medium_dataset;
    let iterations = 100;
    let mut durations = Vec::new();

    // For demonstration, we'll use a pipeline that processes data in chunks
    for _ in 0..iterations {
        let start = Instant::now();

        // Process in smaller chunks to simulate memory efficiency
        let chunk_size = 100;
        for chunk_start in (0..X.nrows()).step_by(chunk_size) {
            let chunk_end = (chunk_start + chunk_size).min(X.nrows());
            let x_chunk = X.slice(s![chunk_start..chunk_end, ..]);
            let y_chunk = y.slice(s![chunk_start..chunk_end]);

            let pipeline = Pipeline::builder()
                .step(
                    "efficient_transformer",
                    Box::new(StandardTransformer::new(1.0)),
                )
                .estimator(Box::new(SimplePredictor::new()))
                .build();

            let _ = pipeline
                .fit(&x_chunk.to_owned(), Some(&y_chunk.to_owned()))?
                .predict(&x_chunk.to_owned())?;
        }

        durations.push(start.elapsed());
    }

    let metrics = calculate_metrics(&durations, X.nrows(), iterations);
    results.add_result("Memory-Efficient".to_string(), metrics);

    println!(
        "✅ Memory-efficient pipeline: {:?} avg",
        durations.iter().sum::<Duration>() / durations.len() as u32
    );

    Ok(())
}

/// Benchmark cached pipeline execution
fn benchmark_cached_pipeline(
    suite: &BenchmarkSuite,
    results: &mut BenchmarkResults,
) -> SklResult<()> {
    println!("\n🗄️ Benchmarking Cached Pipeline");
    println!("{}", "-".repeat(50));

    let (X, y) = &suite.medium_dataset;
    let iterations = 100;
    let mut durations = Vec::new();

    // Create pipeline once and reuse (simulating caching benefits)
    let pipeline = Pipeline::builder()
        .step(
            "cached_transformer",
            Box::new(StandardTransformer::new(1.0)),
        )
        .estimator(Box::new(SimplePredictor::new()))
        .build();

    let fitted_pipeline = pipeline.fit(X, Some(y))?;

    // Benchmark repeated predictions (cache hits)
    for _ in 0..iterations {
        let start = Instant::now();
        let _ = fitted_pipeline.predict(X)?;
        durations.push(start.elapsed());
    }

    let metrics = calculate_metrics(&durations, X.nrows(), iterations);
    results.add_result("Cached Pipeline".to_string(), metrics);

    println!(
        "✅ Cached pipeline: {:?} avg",
        durations.iter().sum::<Duration>() / durations.len() as u32
    );

    Ok(())
}

/// Benchmark different feature union strategies
fn benchmark_feature_union_strategies(
    suite: &BenchmarkSuite,
    results: &mut BenchmarkResults,
) -> SklResult<()> {
    println!("\n🔗 Benchmarking Feature Union Strategies");
    println!("{}", "-".repeat(50));

    let (X, y) = &suite.medium_dataset;
    let iterations = 50;
    let mut durations = Vec::new();

    for _ in 0..iterations {
        let start = Instant::now();

        let feature_union = FeatureUnion::builder()
            .transformer("scale1", Box::new(StandardTransformer::new(0.5)))
            .transformer("scale2", Box::new(StandardTransformer::new(2.0)))
            .transformer("scale3", Box::new(StandardTransformer::new(1.5)))
            .build();

        let pipeline = Pipeline::builder()
            .step("feature_union", Box::new(feature_union))
            .estimator(Box::new(SimplePredictor::new()))
            .build();

        let fitted = pipeline.fit(X, Some(y))?;
        let _ = fitted.predict(X)?;

        durations.push(start.elapsed());
    }

    let metrics = calculate_metrics(&durations, X.nrows(), iterations);
    results.add_result("Feature Union".to_string(), metrics);

    println!(
        "✅ Feature union pipeline: {:?} avg",
        durations.iter().sum::<Duration>() / durations.len() as u32
    );

    Ok(())
}

/// Calculate benchmark metrics from duration measurements
fn calculate_metrics(
    durations: &[Duration],
    n_samples: usize,
    iterations: usize,
) -> BenchmarkMetrics {
    let mean_nanos: f64 =
        durations.iter().map(|d| d.as_nanos() as f64).sum::<f64>() / durations.len() as f64;
    let mean_duration = Duration::from_nanos(mean_nanos as u64);

    let variance: f64 = durations
        .iter()
        .map(|d| (d.as_nanos() as f64 - mean_nanos).powi(2))
        .sum::<f64>()
        / durations.len() as f64;
    let std_deviation = Duration::from_nanos(variance.sqrt() as u64);

    let throughput = n_samples as f64 / mean_duration.as_secs_f64();
    let memory_usage = (n_samples * std::mem::size_of::<Float>()) as f64 / 1_024_000.0; // Rough estimate in MB

    BenchmarkMetrics {
        mean_duration,
        std_deviation,
        throughput_samples_per_sec: throughput,
        memory_usage_mb: memory_usage,
        iterations,
    }
}

/// Print performance recommendations based on benchmark results
fn print_recommendations() {
    println!("\n💡 Performance Recommendations");
    println!("{}", "=".repeat(60));
    println!("• Use SIMD pipelines for numerical computations on large arrays");
    println!("• Employ parallel pipelines for datasets with >10k samples");
    println!("• Consider memory-efficient pipelines for memory-constrained environments");
    println!("• Cache fitted pipelines when making repeated predictions");
    println!("• Feature unions add overhead but provide modeling flexibility");
    println!("• Profile your specific use case - results vary by data characteristics");
    println!("\n📈 Scaling Guidelines:");
    println!("• Small data (<1k samples): Basic pipelines sufficient");
    println!("• Medium data (1k-100k samples): Consider SIMD optimizations");
    println!("• Large data (>100k samples): Use parallel + memory-efficient strategies");
    println!("• Real-time inference: Prefer cached pipelines with SIMD transforms");
}
