//! Benchmarking framework for domain-specific feature selection methods

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::{thread_rng, Distribution, StandardNormal};
use sklears_core::error::SklearsError;
use sklears_core::traits::{Fit, Transform};
use std::time::{Duration, Instant};

use crate::domain_specific::advanced_nlp::NLPStrategy;
use crate::domain_specific::bioinformatics::BioinformaticsStrategy;
use crate::domain_specific::finance::FinanceStrategy;
use crate::domain_specific::*;

/// Result of a single benchmark run
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    pub method_name: String,
    pub domain: String,
    pub strategy: String,
    pub dataset_size: (usize, usize), // (n_samples, n_features)
    pub k_features: usize,
    pub fit_time: Duration,
    pub transform_time: Duration,
    pub total_time: Duration,
    pub memory_usage_mb: f64,
    pub selected_features_count: usize,
    pub feature_quality_score: f64,
}

/// Collection of benchmark results
#[derive(Debug, Clone)]
pub struct BenchmarkSuite {
    pub results: Vec<BenchmarkResult>,
    pub summary: BenchmarkSummary,
}

/// Summary statistics for benchmark results
#[derive(Debug, Clone)]
pub struct BenchmarkSummary {
    pub total_methods_tested: usize,
    pub fastest_method: String,
    pub slowest_method: String,
    pub most_memory_efficient: String,
    pub highest_quality_score: String,
    pub average_fit_time: Duration,
    pub average_transform_time: Duration,
}

/// Configuration for benchmarking
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    pub dataset_sizes: Vec<(usize, usize)>,
    pub k_values: Vec<usize>,
    pub repetitions: usize,
    pub include_bioinformatics: bool,
    pub include_finance: bool,
    pub include_nlp: bool,
    pub measure_memory: bool,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            dataset_sizes: vec![(100, 50), (200, 100), (500, 200)],
            k_values: vec![10, 20, 50],
            repetitions: 3,
            include_bioinformatics: true,
            include_finance: true,
            include_nlp: true,
            measure_memory: false, // Simplified for demonstration
        }
    }
}

/// Domain-specific benchmarking framework
pub struct DomainBenchmarkFramework {
    config: BenchmarkConfig,
}

impl DomainBenchmarkFramework {
    /// Create a new benchmarking framework
    pub fn new(config: BenchmarkConfig) -> Self {
        Self { config }
    }

    /// Run comprehensive benchmarks across all domain-specific methods
    pub fn run_comprehensive_benchmark(&self) -> Result<BenchmarkSuite, SklearsError> {
        let mut all_results = Vec::new();

        // Benchmark bioinformatics methods
        if self.config.include_bioinformatics {
            let bio_results = self.benchmark_bioinformatics_methods()?;
            all_results.extend(bio_results);
        }

        // Benchmark finance methods
        if self.config.include_finance {
            let finance_results = self.benchmark_finance_methods()?;
            all_results.extend(finance_results);
        }

        // Benchmark NLP methods
        if self.config.include_nlp {
            let nlp_results = self.benchmark_nlp_methods()?;
            all_results.extend(nlp_results);
        }

        let summary = self.generate_summary(&all_results);

        Ok(BenchmarkSuite {
            results: all_results,
            summary,
        })
    }

    /// Benchmark bioinformatics feature selection methods
    fn benchmark_bioinformatics_methods(&self) -> Result<Vec<BenchmarkResult>, SklearsError> {
        let mut results = Vec::new();

        let strategies = vec![
            BioinformaticsStrategy::DifferentialExpression,
            BioinformaticsStrategy::FunctionalAnnotation,
            BioinformaticsStrategy::PathwayEnrichment,
            BioinformaticsStrategy::CoExpressionAnalysis,
        ];

        for &(n_samples, n_features) in &self.config.dataset_sizes {
            for &k in &self.config.k_values {
                if k >= n_features {
                    continue;
                }

                for strategy in &strategies {
                    let mut avg_fit_time = Duration::new(0, 0);
                    let mut avg_transform_time = Duration::new(0, 0);
                    let mut avg_quality = 0.0;
                    let mut successful_runs = 0;

                    for _ in 0..self.config.repetitions {
                        match self.benchmark_single_bioinformatics_run(
                            n_samples,
                            n_features,
                            k,
                            strategy.clone(),
                        ) {
                            Ok((fit_time, transform_time, quality, _selected_count)) => {
                                avg_fit_time += fit_time;
                                avg_transform_time += transform_time;
                                avg_quality += quality;
                                successful_runs += 1;
                            }
                            Err(_) => continue, // Skip failed runs
                        }
                    }

                    if successful_runs > 0 {
                        avg_fit_time /= successful_runs as u32;
                        avg_transform_time /= successful_runs as u32;
                        avg_quality /= successful_runs as f64;

                        results.push(BenchmarkResult {
                            method_name: "BioinformaticsFeatureSelector".to_string(),
                            domain: "bioinformatics".to_string(),
                            strategy: format!("{:?}", strategy),
                            dataset_size: (n_samples, n_features),
                            k_features: k,
                            fit_time: avg_fit_time,
                            transform_time: avg_transform_time,
                            total_time: avg_fit_time + avg_transform_time,
                            memory_usage_mb: self.estimate_memory_usage(n_samples, n_features, k),
                            selected_features_count: k,
                            feature_quality_score: avg_quality,
                        });
                    }
                }
            }
        }

        Ok(results)
    }

    /// Benchmark finance feature selection methods
    fn benchmark_finance_methods(&self) -> Result<Vec<BenchmarkResult>, SklearsError> {
        let mut results = Vec::new();

        let strategies = vec![
            FinanceStrategy::Momentum,
            FinanceStrategy::TechnicalIndicators,
            FinanceStrategy::RiskAdjusted,
            FinanceStrategy::Volatility,
        ];

        for &(n_samples, n_features) in &self.config.dataset_sizes {
            for &k in &self.config.k_values {
                if k >= n_features {
                    continue;
                }

                for strategy in &strategies {
                    let mut avg_fit_time = Duration::new(0, 0);
                    let mut avg_transform_time = Duration::new(0, 0);
                    let mut avg_quality = 0.0;
                    let mut successful_runs = 0;

                    for _ in 0..self.config.repetitions {
                        match self.benchmark_single_finance_run(
                            n_samples,
                            n_features,
                            k,
                            strategy.clone(),
                        ) {
                            Ok((fit_time, transform_time, quality, _selected_count)) => {
                                avg_fit_time += fit_time;
                                avg_transform_time += transform_time;
                                avg_quality += quality;
                                successful_runs += 1;
                            }
                            Err(_) => continue,
                        }
                    }

                    if successful_runs > 0 {
                        avg_fit_time /= successful_runs as u32;
                        avg_transform_time /= successful_runs as u32;
                        avg_quality /= successful_runs as f64;

                        results.push(BenchmarkResult {
                            method_name: "FinanceFeatureSelector".to_string(),
                            domain: "finance".to_string(),
                            strategy: format!("{:?}", strategy),
                            dataset_size: (n_samples, n_features),
                            k_features: k,
                            fit_time: avg_fit_time,
                            transform_time: avg_transform_time,
                            total_time: avg_fit_time + avg_transform_time,
                            memory_usage_mb: self.estimate_memory_usage(n_samples, n_features, k),
                            selected_features_count: k,
                            feature_quality_score: avg_quality,
                        });
                    }
                }
            }
        }

        Ok(results)
    }

    /// Benchmark NLP feature selection methods
    fn benchmark_nlp_methods(&self) -> Result<Vec<BenchmarkResult>, SklearsError> {
        let mut results = Vec::new();

        let strategies = vec![
            NLPStrategy::InformationTheoretic,
            NLPStrategy::SyntacticAnalysis,
            NLPStrategy::SemanticAnalysis,
            NLPStrategy::TransformerBased,
        ];

        for &(n_samples, n_features) in &self.config.dataset_sizes {
            for &k in &self.config.k_values {
                if k >= n_features {
                    continue;
                }

                for strategy in &strategies {
                    let mut avg_fit_time = Duration::new(0, 0);
                    let mut avg_transform_time = Duration::new(0, 0);
                    let mut avg_quality = 0.0;
                    let mut successful_runs = 0;

                    for _ in 0..self.config.repetitions {
                        match self.benchmark_single_nlp_run(
                            n_samples,
                            n_features,
                            k,
                            strategy.clone(),
                        ) {
                            Ok((fit_time, transform_time, quality, _selected_count)) => {
                                avg_fit_time += fit_time;
                                avg_transform_time += transform_time;
                                avg_quality += quality;
                                successful_runs += 1;
                            }
                            Err(_) => continue,
                        }
                    }

                    if successful_runs > 0 {
                        avg_fit_time /= successful_runs as u32;
                        avg_transform_time /= successful_runs as u32;
                        avg_quality /= successful_runs as f64;

                        results.push(BenchmarkResult {
                            method_name: "AdvancedNLPFeatureSelector".to_string(),
                            domain: "nlp".to_string(),
                            strategy: format!("{:?}", strategy),
                            dataset_size: (n_samples, n_features),
                            k_features: k,
                            fit_time: avg_fit_time,
                            transform_time: avg_transform_time,
                            total_time: avg_fit_time + avg_transform_time,
                            memory_usage_mb: self.estimate_memory_usage(n_samples, n_features, k),
                            selected_features_count: k,
                            feature_quality_score: avg_quality,
                        });
                    }
                }
            }
        }

        Ok(results)
    }

    /// Benchmark a single bioinformatics run
    #[allow(non_snake_case)]
    fn benchmark_single_bioinformatics_run(
        &self,
        n_samples: usize,
        n_features: usize,
        k: usize,
        strategy: BioinformaticsStrategy,
    ) -> Result<(Duration, Duration, f64, usize), SklearsError> {
        // Generate synthetic genomic data
        let X = Array2::from_shape_fn((n_samples, n_features), |_| {
            let mut rng = thread_rng();
            StandardNormal.sample(&mut rng)
        });
        let y = Array1::from_shape_fn(n_samples, |_| {
            let mut rng = thread_rng();
            StandardNormal.sample(&mut rng)
        });

        let selector = BioinformaticsFeatureSelector::new().k(k).strategy(strategy);

        // Measure fit time
        let fit_start = Instant::now();
        let trained_selector = selector
            .fit(&X, &y)
            .map_err(|_| SklearsError::InvalidInput("Fit failed".to_string()))?;
        let fit_time = fit_start.elapsed();

        // Measure transform time
        let transform_start = Instant::now();
        let transformed = trained_selector
            .transform(&X)
            .map_err(|_| SklearsError::InvalidInput("Transform failed".to_string()))?;
        let transform_time = transform_start.elapsed();

        let selected_count = transformed.ncols();
        let quality_score = self.calculate_feature_quality(&transformed, &y);

        Ok((fit_time, transform_time, quality_score, selected_count))
    }

    /// Benchmark a single finance run
    #[allow(non_snake_case)]
    fn benchmark_single_finance_run(
        &self,
        n_samples: usize,
        n_features: usize,
        k: usize,
        strategy: FinanceStrategy,
    ) -> Result<(Duration, Duration, f64, usize), SklearsError> {
        // Generate synthetic financial time series data
        let X = Array2::from_shape_fn((n_samples, n_features), |_| {
            let mut rng = thread_rng();
            StandardNormal.sample(&mut rng)
        });
        let y = Array1::from_shape_fn(n_samples, |_| {
            let mut rng = thread_rng();
            StandardNormal.sample(&mut rng)
        });

        let selector = FinanceFeatureSelector::new().k(k).strategy(strategy);

        let fit_start = Instant::now();
        let trained_selector = selector
            .fit(&X, &y)
            .map_err(|_| SklearsError::InvalidInput("Fit failed".to_string()))?;
        let fit_time = fit_start.elapsed();

        let transform_start = Instant::now();
        let transformed = trained_selector
            .transform(&X)
            .map_err(|_| SklearsError::InvalidInput("Transform failed".to_string()))?;
        let transform_time = transform_start.elapsed();

        let selected_count = transformed.ncols();
        let quality_score = self.calculate_feature_quality(&transformed, &y);

        Ok((fit_time, transform_time, quality_score, selected_count))
    }

    /// Benchmark a single NLP run
    #[allow(non_snake_case)]
    fn benchmark_single_nlp_run(
        &self,
        n_samples: usize,
        n_features: usize,
        k: usize,
        strategy: NLPStrategy,
    ) -> Result<(Duration, Duration, f64, usize), SklearsError> {
        // Generate synthetic text feature data
        let X = Array2::from_shape_fn((n_samples, n_features), |_| {
            let mut rng = thread_rng();
            StandardNormal.sample(&mut rng)
        });
        let y = Array1::from_shape_fn(n_samples, |_| {
            let mut rng = thread_rng();
            StandardNormal.sample(&mut rng)
        });

        let selector = AdvancedNLPFeatureSelector::new().k(k).strategy(strategy);

        let fit_start = Instant::now();
        let trained_selector = selector
            .fit(&X, &y)
            .map_err(|_| SklearsError::InvalidInput("Fit failed".to_string()))?;
        let fit_time = fit_start.elapsed();

        let transform_start = Instant::now();
        let transformed = trained_selector
            .transform(&X)
            .map_err(|_| SklearsError::InvalidInput("Transform failed".to_string()))?;
        let transform_time = transform_start.elapsed();

        let selected_count = transformed.ncols();
        let quality_score = self.calculate_feature_quality(&transformed, &y);

        Ok((fit_time, transform_time, quality_score, selected_count))
    }

    /// Calculate feature quality score (simplified correlation-based metric)
    fn calculate_feature_quality(&self, X: &Array2<f64>, y: &Array1<f64>) -> f64 {
        let mut total_correlation = 0.0;
        let n_features = X.ncols();

        for i in 0..n_features {
            let feature_values = X.column(i);
            let correlation = self.pearson_correlation(&feature_values.to_owned(), y);
            total_correlation += correlation.abs();
        }

        total_correlation / n_features as f64
    }

    /// Calculate Pearson correlation
    fn pearson_correlation(&self, x: &Array1<f64>, y: &Array1<f64>) -> f64 {
        let x_mean = x.mean().unwrap_or(0.0);
        let y_mean = y.mean().unwrap_or(0.0);

        let numerator: f64 = x
            .iter()
            .zip(y.iter())
            .map(|(&xi, &yi)| (xi - x_mean) * (yi - y_mean))
            .sum();

        let x_sq_sum: f64 = x.iter().map(|&xi| (xi - x_mean).powi(2)).sum();
        let y_sq_sum: f64 = y.iter().map(|&yi| (yi - y_mean).powi(2)).sum();

        let denominator = (x_sq_sum * y_sq_sum).sqrt();

        if denominator != 0.0 {
            numerator / denominator
        } else {
            0.0
        }
    }

    /// Estimate memory usage (simplified calculation)
    fn estimate_memory_usage(&self, n_samples: usize, n_features: usize, k: usize) -> f64 {
        // Rough estimate: input data + selected features + metadata
        let input_size = n_samples * n_features * 8; // 8 bytes per f64
        let selected_size = n_samples * k * 8;
        let metadata_size = 1024; // Approximate metadata overhead

        (input_size + selected_size + metadata_size) as f64 / (1024.0 * 1024.0) // Convert to MB
    }

    /// Generate benchmark summary
    fn generate_summary(&self, results: &[BenchmarkResult]) -> BenchmarkSummary {
        if results.is_empty() {
            return BenchmarkSummary {
                total_methods_tested: 0,
                fastest_method: "None".to_string(),
                slowest_method: "None".to_string(),
                most_memory_efficient: "None".to_string(),
                highest_quality_score: "None".to_string(),
                average_fit_time: Duration::new(0, 0),
                average_transform_time: Duration::new(0, 0),
            };
        }

        let fastest = results.iter().min_by_key(|r| r.total_time).unwrap();

        let slowest = results.iter().max_by_key(|r| r.total_time).unwrap();

        let most_memory_efficient = results
            .iter()
            .min_by(|a, b| a.memory_usage_mb.partial_cmp(&b.memory_usage_mb).unwrap())
            .unwrap();

        let highest_quality = results
            .iter()
            .max_by(|a, b| {
                a.feature_quality_score
                    .partial_cmp(&b.feature_quality_score)
                    .unwrap()
            })
            .unwrap();

        let total_fit_time: Duration = results.iter().map(|r| r.fit_time).sum();
        let total_transform_time: Duration = results.iter().map(|r| r.transform_time).sum();
        let n_results = results.len() as u32;

        BenchmarkSummary {
            total_methods_tested: results.len(),
            fastest_method: format!("{} ({})", fastest.method_name, fastest.strategy),
            slowest_method: format!("{} ({})", slowest.method_name, slowest.strategy),
            most_memory_efficient: format!(
                "{} ({})",
                most_memory_efficient.method_name, most_memory_efficient.strategy
            ),
            highest_quality_score: format!(
                "{} ({})",
                highest_quality.method_name, highest_quality.strategy
            ),
            average_fit_time: total_fit_time / n_results,
            average_transform_time: total_transform_time / n_results,
        }
    }

    /// Export benchmark results to CSV format
    pub fn export_to_csv(&self, results: &BenchmarkSuite) -> String {
        let mut csv = String::new();
        csv.push_str("method_name,domain,strategy,n_samples,n_features,k_features,fit_time_ms,transform_time_ms,total_time_ms,memory_mb,selected_count,quality_score\n");

        for result in &results.results {
            csv.push_str(&format!(
                "{},{},{},{},{},{},{},{},{},{:.2},{},{:.4}\n",
                result.method_name,
                result.domain,
                result.strategy,
                result.dataset_size.0,
                result.dataset_size.1,
                result.k_features,
                result.fit_time.as_millis(),
                result.transform_time.as_millis(),
                result.total_time.as_millis(),
                result.memory_usage_mb,
                result.selected_features_count,
                result.feature_quality_score
            ));
        }

        csv
    }
}

/// Convenience function to run a quick benchmark with default settings
pub fn run_quick_benchmark() -> Result<BenchmarkSuite, SklearsError> {
    let config = BenchmarkConfig {
        dataset_sizes: vec![(50, 30), (100, 50)],
        k_values: vec![10, 20],
        repetitions: 2,
        ..Default::default()
    };

    let framework = DomainBenchmarkFramework::new(config);
    framework.run_comprehensive_benchmark()
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benchmark_framework_creation() {
        let config = BenchmarkConfig::default();
        let framework = DomainBenchmarkFramework::new(config);

        // Test that framework is created successfully
        assert_eq!(framework.config.repetitions, 3);
        assert!(framework.config.include_bioinformatics);
        assert!(framework.config.include_finance);
        assert!(framework.config.include_nlp);
    }

    #[test]
    fn test_memory_estimation() {
        let config = BenchmarkConfig::default();
        let framework = DomainBenchmarkFramework::new(config);

        let memory_mb = framework.estimate_memory_usage(100, 50, 10);
        assert!(memory_mb > 0.0);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_feature_quality_calculation() {
        let config = BenchmarkConfig::default();
        let framework = DomainBenchmarkFramework::new(config);

        let X = Array2::from_shape_fn((20, 5), |_| {
            let mut rng = thread_rng();
            StandardNormal.sample(&mut rng)
        });
        let y = Array1::from_shape_fn(20, |_| {
            let mut rng = thread_rng();
            StandardNormal.sample(&mut rng)
        });

        let quality = framework.calculate_feature_quality(&X, &y);
        assert!(quality >= 0.0 && quality <= 1.0);
    }

    #[test]
    fn test_csv_export() {
        let config = BenchmarkConfig::default();
        let framework = DomainBenchmarkFramework::new(config);

        let dummy_result = BenchmarkResult {
            method_name: "TestMethod".to_string(),
            domain: "test".to_string(),
            strategy: "TestStrategy".to_string(),
            dataset_size: (100, 50),
            k_features: 10,
            fit_time: Duration::from_millis(100),
            transform_time: Duration::from_millis(50),
            total_time: Duration::from_millis(150),
            memory_usage_mb: 5.0,
            selected_features_count: 10,
            feature_quality_score: 0.75,
        };

        let dummy_summary = BenchmarkSummary {
            total_methods_tested: 1,
            fastest_method: "TestMethod".to_string(),
            slowest_method: "TestMethod".to_string(),
            most_memory_efficient: "TestMethod".to_string(),
            highest_quality_score: "TestMethod".to_string(),
            average_fit_time: Duration::from_millis(100),
            average_transform_time: Duration::from_millis(50),
        };

        let suite = BenchmarkSuite {
            results: vec![dummy_result],
            summary: dummy_summary,
        };

        let csv = framework.export_to_csv(&suite);
        assert!(csv.contains("method_name"));
        assert!(csv.contains("TestMethod"));
    }
}
