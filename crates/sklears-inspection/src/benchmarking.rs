//! Benchmarking and Performance Analysis
//!
//! This module provides comprehensive benchmarking capabilities for model inspection methods,
//! including performance comparisons, speed analysis, memory profiling, and explanation quality benchmarks.

use crate::SklResult;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use sklears_core::{error::SklearsError, types::Float};
use std::collections::HashMap;
use std::time::{Duration, Instant};

#[cfg(feature = "serde")]
use chrono::{DateTime, Utc};

/// Configuration for benchmarking
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct BenchmarkConfig {
    /// Number of warmup iterations
    pub warmup_iterations: usize,
    /// Number of benchmark iterations
    pub benchmark_iterations: usize,
    /// Memory profiling enabled
    pub memory_profiling: bool,
    /// Statistical significance level
    pub significance_level: Float,
    /// Benchmark categories to run
    pub categories: Vec<BenchmarkCategory>,
    /// Reference implementation for comparison
    pub reference_implementation: Option<String>,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            warmup_iterations: 10,
            benchmark_iterations: 100,
            memory_profiling: true,
            significance_level: 0.05,
            categories: vec![
                BenchmarkCategory::FeatureImportance,
                BenchmarkCategory::LocalExplanations,
                BenchmarkCategory::GlobalExplanations,
                BenchmarkCategory::Visualization,
            ],
            reference_implementation: None,
        }
    }
}

/// Benchmark categories
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum BenchmarkCategory {
    /// Feature importance methods
    FeatureImportance,
    /// Local explanation methods
    LocalExplanations,
    /// Global explanation methods
    GlobalExplanations,
    /// Visualization generation
    Visualization,
    /// Model comparison
    ModelComparison,
    /// Uncertainty quantification
    UncertaintyQuantification,
    /// All categories
    All,
}

/// Benchmark result for a single method
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct BenchmarkResult {
    /// Method name
    pub method_name: String,
    /// Category
    pub category: BenchmarkCategory,
    /// Timing statistics
    pub timing_stats: TimingStatistics,
    /// Memory statistics
    pub memory_stats: Option<MemoryStatistics>,
    /// Quality metrics
    pub quality_metrics: QualityMetrics,
    /// Comparison with reference
    pub reference_comparison: Option<ReferenceComparison>,
    /// Test configuration
    pub test_config: TestConfiguration,
    /// Timestamp
    #[cfg(feature = "serde")]
    pub timestamp: DateTime<Utc>,
}

/// Timing statistics
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct TimingStatistics {
    /// Mean execution time
    pub mean_time: Duration,
    /// Standard deviation
    pub std_dev: Duration,
    /// Median time
    pub median_time: Duration,
    /// Minimum time
    pub min_time: Duration,
    /// Maximum time
    pub max_time: Duration,
    /// 95th percentile
    pub percentile_95: Duration,
    /// Throughput (operations per second)
    pub throughput: Float,
}

/// Memory statistics
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct MemoryStatistics {
    /// Peak memory usage in bytes
    pub peak_memory: usize,
    /// Average memory usage
    pub avg_memory: usize,
    /// Memory allocations
    pub allocations: usize,
    /// Memory deallocations
    pub deallocations: usize,
    /// Memory efficiency score
    pub efficiency_score: Float,
}

/// Quality metrics for explanations
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct QualityMetrics {
    /// Fidelity score
    pub fidelity: Float,
    /// Stability score
    pub stability: Float,
    /// Consistency score
    pub consistency: Float,
    /// Completeness score
    pub completeness: Float,
    /// Interpretability score
    pub interpretability: Float,
    /// Overall quality score
    pub overall_score: Float,
}

/// Comparison with reference implementation
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ReferenceComparison {
    /// Speed improvement factor
    pub speed_improvement: Float,
    /// Memory improvement factor
    pub memory_improvement: Float,
    /// Quality difference
    pub quality_difference: Float,
    /// Statistical significance
    pub is_significant: bool,
    /// P-value of comparison
    pub p_value: Float,
}

/// Test configuration used for benchmarking
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct TestConfiguration {
    /// Dataset size (number of samples)
    pub dataset_size: usize,
    /// Number of features
    pub num_features: usize,
    /// Model type
    pub model_type: String,
    /// Problem type (classification/regression)
    pub problem_type: ProblemType,
    /// Additional parameters
    pub parameters: HashMap<String, String>,
}

/// Problem type for benchmarking
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum ProblemType {
    /// Binary classification
    BinaryClassification,
    /// Multi-class classification
    MultiClassification,
    /// Regression
    Regression,
}

/// Main benchmarking suite
pub struct BenchmarkingSuite {
    config: BenchmarkConfig,
    results: Vec<BenchmarkResult>,
    reference_results: HashMap<String, BenchmarkResult>,
}

impl BenchmarkingSuite {
    /// Create a new benchmarking suite
    pub fn new(config: BenchmarkConfig) -> Self {
        Self {
            config,
            results: Vec::new(),
            reference_results: HashMap::new(),
        }
    }

    /// Run complete benchmark suite
    pub fn run_benchmarks(&mut self) -> SklResult<BenchmarkReport> {
        for category in &self.config.categories.clone() {
            self.benchmark_category(*category)?;
        }

        self.generate_report()
    }

    /// Benchmark a specific method
    pub fn benchmark_method<F, T>(
        &mut self,
        method_name: String,
        category: BenchmarkCategory,
        method_fn: F,
        test_config: TestConfiguration,
    ) -> SklResult<BenchmarkResult>
    where
        F: Fn() -> SklResult<T>,
        T: std::fmt::Debug,
    {
        // Warmup phase
        for _ in 0..self.config.warmup_iterations {
            let _ = method_fn()?;
        }

        // Timing measurements
        let mut times = Vec::new();
        let mut memory_snapshots = Vec::new();

        for _ in 0..self.config.benchmark_iterations {
            let memory_before = if self.config.memory_profiling {
                Some(self.get_memory_usage())
            } else {
                None
            };

            let start_time = Instant::now();
            let _result = method_fn()?;
            let elapsed = start_time.elapsed();

            times.push(elapsed);

            if let Some(mem_before) = memory_before {
                let memory_after = self.get_memory_usage();
                memory_snapshots.push((mem_before, memory_after));
            }
        }

        // Calculate timing statistics
        let timing_stats = self.calculate_timing_statistics(&times);

        // Calculate memory statistics
        let memory_stats = if self.config.memory_profiling {
            Some(self.calculate_memory_statistics(&memory_snapshots))
        } else {
            None
        };

        // Calculate quality metrics (placeholder implementation)
        let quality_metrics = self.calculate_quality_metrics(&method_name, &test_config)?;

        // Compare with reference if available
        let reference_comparison = if let Some(ref_name) = &self.config.reference_implementation {
            self.reference_results.get(ref_name).map(|ref_result| {
                self.compare_with_reference(
                    &timing_stats,
                    &memory_stats,
                    &quality_metrics,
                    ref_result,
                )
            })
        } else {
            None
        };

        let result = BenchmarkResult {
            method_name,
            category,
            timing_stats,
            memory_stats,
            quality_metrics,
            reference_comparison,
            test_config,
            #[cfg(feature = "serde")]
            timestamp: Utc::now(),
        };

        self.results.push(result.clone());
        Ok(result)
    }

    /// Benchmark feature importance methods
    pub fn benchmark_feature_importance(&mut self) -> SklResult<Vec<BenchmarkResult>> {
        let mut results = Vec::new();

        let test_configs = vec![
            TestConfiguration {
                dataset_size: 1000,
                num_features: 10,
                model_type: "RandomForest".to_string(),
                problem_type: ProblemType::BinaryClassification,
                parameters: HashMap::new(),
            },
            TestConfiguration {
                dataset_size: 5000,
                num_features: 50,
                model_type: "RandomForest".to_string(),
                problem_type: ProblemType::MultiClassification,
                parameters: HashMap::new(),
            },
        ];

        for config in test_configs {
            // Benchmark permutation importance
            let config_clone = config.clone();
            let perm_result = self.benchmark_method(
                "PermutationImportance".to_string(),
                BenchmarkCategory::FeatureImportance,
                move || Self::simulate_permutation_importance_static(&config_clone),
                config.clone(),
            )?;
            results.push(perm_result);

            // Benchmark SHAP
            let config_clone = config.clone();
            let shap_result = self.benchmark_method(
                "SHAP".to_string(),
                BenchmarkCategory::FeatureImportance,
                move || Self::simulate_shap_computation_static(&config_clone),
                config.clone(),
            )?;
            results.push(shap_result);
        }

        Ok(results)
    }

    /// Add reference implementation results
    pub fn add_reference_result(&mut self, name: String, result: BenchmarkResult) {
        self.reference_results.insert(name, result);
    }

    /// Generate comprehensive benchmark report
    pub fn generate_report(&self) -> SklResult<BenchmarkReport> {
        let mut category_summaries = HashMap::new();

        // Group results by category
        for result in &self.results {
            let category_results = category_summaries
                .entry(result.category)
                .or_insert_with(Vec::new);
            category_results.push(result.clone());
        }

        // Generate summaries for each category
        let mut summaries = HashMap::new();
        for (category, results) in category_summaries {
            summaries.insert(category, self.generate_category_summary(&results));
        }

        // Overall performance insights
        let insights = self.generate_performance_insights();

        // Recommendations
        let recommendations = self.generate_recommendations();

        Ok(BenchmarkReport {
            config: self.config.clone(),
            results: self.results.clone(),
            category_summaries: summaries,
            performance_insights: insights,
            recommendations,
            #[cfg(feature = "serde")]
            generated_at: Utc::now(),
        })
    }

    fn benchmark_category(&mut self, category: BenchmarkCategory) -> SklResult<()> {
        match category {
            BenchmarkCategory::FeatureImportance => {
                self.benchmark_feature_importance()?;
            }
            BenchmarkCategory::LocalExplanations => {
                self.benchmark_local_explanations()?;
            }
            BenchmarkCategory::GlobalExplanations => {
                self.benchmark_global_explanations()?;
            }
            BenchmarkCategory::Visualization => {
                self.benchmark_visualization()?;
            }
            BenchmarkCategory::All => {
                self.benchmark_feature_importance()?;
                self.benchmark_local_explanations()?;
                self.benchmark_global_explanations()?;
                self.benchmark_visualization()?;
            }
            _ => {} // Other categories
        }
        Ok(())
    }

    fn benchmark_local_explanations(&mut self) -> SklResult<Vec<BenchmarkResult>> {
        // Placeholder implementation
        Ok(Vec::new())
    }

    fn benchmark_global_explanations(&mut self) -> SklResult<Vec<BenchmarkResult>> {
        // Placeholder implementation
        Ok(Vec::new())
    }

    fn benchmark_visualization(&mut self) -> SklResult<Vec<BenchmarkResult>> {
        // Placeholder implementation
        Ok(Vec::new())
    }

    fn calculate_timing_statistics(&self, times: &[Duration]) -> TimingStatistics {
        if times.is_empty() {
            return TimingStatistics {
                mean_time: Duration::from_secs(0),
                std_dev: Duration::from_secs(0),
                median_time: Duration::from_secs(0),
                min_time: Duration::from_secs(0),
                max_time: Duration::from_secs(0),
                percentile_95: Duration::from_secs(0),
                throughput: 0.0,
            };
        }

        let mut sorted_times = times.to_vec();
        sorted_times.sort();

        let mean_nanos: u128 =
            times.iter().map(|d| d.as_nanos()).sum::<u128>() / times.len() as u128;
        let mean_time = Duration::from_nanos(mean_nanos as u64);

        let variance: f64 = times
            .iter()
            .map(|d| {
                let diff = d.as_nanos() as f64 - mean_nanos as f64;
                diff * diff
            })
            .sum::<f64>()
            / times.len() as f64;
        let std_dev = Duration::from_nanos(variance.sqrt() as u64);

        let median_time = sorted_times[times.len() / 2];
        let min_time = *sorted_times.first().unwrap();
        let max_time = *sorted_times.last().unwrap();
        let percentile_95 = sorted_times[(times.len() as f64 * 0.95) as usize];

        let throughput = if mean_time.as_secs_f64() > 0.0 {
            1.0 / mean_time.as_secs_f64()
        } else {
            0.0
        };

        TimingStatistics {
            mean_time,
            std_dev,
            median_time,
            min_time,
            max_time,
            percentile_95,
            throughput,
        }
    }

    fn calculate_memory_statistics(&self, snapshots: &[(usize, usize)]) -> MemoryStatistics {
        if snapshots.is_empty() {
            return MemoryStatistics {
                peak_memory: 0,
                avg_memory: 0,
                allocations: 0,
                deallocations: 0,
                efficiency_score: 0.0,
            };
        }

        let peak_memory = snapshots.iter().map(|(_, after)| *after).max().unwrap_or(0);

        let avg_memory = snapshots.iter().map(|(_, after)| *after).sum::<usize>() / snapshots.len();

        // Simplified memory statistics
        MemoryStatistics {
            peak_memory,
            avg_memory,
            allocations: snapshots.len(),
            deallocations: snapshots.len(),
            efficiency_score: 0.8, // Placeholder
        }
    }

    fn calculate_quality_metrics(
        &self,
        method_name: &str,
        _config: &TestConfiguration,
    ) -> SklResult<QualityMetrics> {
        // Placeholder implementation - in practice, these would be calculated
        // based on actual explanation quality assessments
        let base_score = match method_name {
            "PermutationImportance" => 0.85,
            "SHAP" => 0.90,
            "LIME" => 0.80,
            _ => 0.75,
        };

        Ok(QualityMetrics {
            fidelity: base_score + 0.05,
            stability: base_score - 0.02,
            consistency: base_score + 0.03,
            completeness: base_score - 0.05,
            interpretability: base_score + 0.02,
            overall_score: base_score,
        })
    }

    fn compare_with_reference(
        &self,
        timing: &TimingStatistics,
        memory: &Option<MemoryStatistics>,
        quality: &QualityMetrics,
        reference: &BenchmarkResult,
    ) -> ReferenceComparison {
        let speed_improvement =
            reference.timing_stats.mean_time.as_secs_f64() / timing.mean_time.as_secs_f64();

        let memory_improvement =
            if let (Some(current_mem), Some(ref_mem)) = (memory, &reference.memory_stats) {
                ref_mem.peak_memory as f64 / current_mem.peak_memory as f64
            } else {
                1.0
            };

        let quality_difference = quality.overall_score - reference.quality_metrics.overall_score;

        // Simple statistical test (placeholder)
        let p_value = 0.01; // Would be calculated based on actual statistical test
        let is_significant = p_value < self.config.significance_level;

        ReferenceComparison {
            speed_improvement,
            memory_improvement,
            quality_difference,
            is_significant,
            p_value,
        }
    }

    fn generate_category_summary(&self, results: &[BenchmarkResult]) -> CategorySummary {
        if results.is_empty() {
            return CategorySummary {
                best_method: "None".to_string(),
                worst_method: "None".to_string(),
                avg_throughput: 0.0,
                avg_quality: 0.0,
                performance_ranking: Vec::new(),
            };
        }

        let best_method = results
            .iter()
            .max_by(|a, b| {
                a.timing_stats
                    .throughput
                    .partial_cmp(&b.timing_stats.throughput)
                    .unwrap()
            })
            .unwrap()
            .method_name
            .clone();

        let worst_method = results
            .iter()
            .min_by(|a, b| {
                a.timing_stats
                    .throughput
                    .partial_cmp(&b.timing_stats.throughput)
                    .unwrap()
            })
            .unwrap()
            .method_name
            .clone();

        let avg_throughput = results
            .iter()
            .map(|r| r.timing_stats.throughput)
            .sum::<Float>()
            / results.len() as Float;

        let avg_quality = results
            .iter()
            .map(|r| r.quality_metrics.overall_score)
            .sum::<Float>()
            / results.len() as Float;

        let mut performance_ranking: Vec<(String, Float)> = results
            .iter()
            .map(|r| (r.method_name.clone(), r.timing_stats.throughput))
            .collect();
        performance_ranking.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        CategorySummary {
            best_method,
            worst_method,
            avg_throughput,
            avg_quality,
            performance_ranking,
        }
    }

    fn generate_performance_insights(&self) -> Vec<PerformanceInsight> {
        let mut insights = Vec::new();

        // Speed insights
        let fastest_method = self.results.iter().max_by(|a, b| {
            a.timing_stats
                .throughput
                .partial_cmp(&b.timing_stats.throughput)
                .unwrap()
        });

        if let Some(method) = fastest_method {
            insights.push(PerformanceInsight {
                insight_type: InsightType::Speed,
                message: format!(
                    "{} is the fastest method with {:.2} ops/sec",
                    method.method_name, method.timing_stats.throughput
                ),
                severity: InsightSeverity::Info,
            });
        }

        // Quality insights
        let highest_quality = self.results.iter().max_by(|a, b| {
            a.quality_metrics
                .overall_score
                .partial_cmp(&b.quality_metrics.overall_score)
                .unwrap()
        });

        if let Some(method) = highest_quality {
            insights.push(PerformanceInsight {
                insight_type: InsightType::Quality,
                message: format!(
                    "{} has the highest quality score: {:.3}",
                    method.method_name, method.quality_metrics.overall_score
                ),
                severity: InsightSeverity::Info,
            });
        }

        insights
    }

    fn generate_recommendations(&self) -> Vec<String> {
        let mut recommendations = Vec::new();

        recommendations.push("Consider using parallel processing for large datasets".to_string());
        recommendations.push("Enable caching for repeated computations".to_string());
        recommendations.push("Profile memory usage for memory-intensive operations".to_string());

        recommendations
    }

    fn get_memory_usage(&self) -> usize {
        // Placeholder implementation - would use actual memory profiling

        // This is a simplified placeholder
        1024 * 1024 // 1MB placeholder
    }

    fn simulate_permutation_importance_static(config: &TestConfiguration) -> SklResult<String> {
        // Simulate computation time based on dataset size
        let computation_time =
            Duration::from_millis((config.dataset_size * config.num_features / 100) as u64);
        std::thread::sleep(computation_time);
        Ok("Permutation importance computed".to_string())
    }

    fn simulate_shap_computation_static(config: &TestConfiguration) -> SklResult<String> {
        // SHAP typically takes longer than permutation importance
        let computation_time =
            Duration::from_millis((config.dataset_size * config.num_features / 50) as u64);
        std::thread::sleep(computation_time);
        Ok("SHAP values computed".to_string())
    }
}

/// Category summary for benchmarking results
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct CategorySummary {
    /// Best performing method in category
    pub best_method: String,
    /// Worst performing method in category
    pub worst_method: String,
    /// Average throughput across methods
    pub avg_throughput: Float,
    /// Average quality score
    pub avg_quality: Float,
    /// Performance ranking (method, throughput)
    pub performance_ranking: Vec<(String, Float)>,
}

/// Performance insight
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct PerformanceInsight {
    /// Type of insight
    pub insight_type: InsightType,
    /// Insight message
    pub message: String,
    /// Severity level
    pub severity: InsightSeverity,
}

/// Types of performance insights
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum InsightType {
    /// Speed-related insight
    Speed,
    /// Memory-related insight
    Memory,
    /// Quality-related insight
    Quality,
    /// Scalability insight
    Scalability,
}

/// Insight severity levels
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum InsightSeverity {
    /// Informational
    Info,
    /// Warning
    Warning,
    /// Critical issue
    Critical,
}

/// Complete benchmark report
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct BenchmarkReport {
    /// Benchmark configuration used
    pub config: BenchmarkConfig,
    /// All benchmark results
    pub results: Vec<BenchmarkResult>,
    /// Category summaries
    pub category_summaries: HashMap<BenchmarkCategory, CategorySummary>,
    /// Performance insights
    pub performance_insights: Vec<PerformanceInsight>,
    /// Recommendations for improvement
    pub recommendations: Vec<String>,
    /// Report generation timestamp
    #[cfg(feature = "serde")]
    pub generated_at: DateTime<Utc>,
}

impl BenchmarkReport {
    /// Export report to JSON
    #[cfg(feature = "serde")]
    pub fn to_json(&self) -> SklResult<String> {
        serde_json::to_string_pretty(self)
            .map_err(|e| SklearsError::InvalidInput(format!("Failed to serialize report: {}", e)))
    }

    /// Generate HTML report
    pub fn to_html(&self) -> String {
        let generated_time = {
            #[cfg(feature = "serde")]
            {
                self.generated_at
                    .format("%Y-%m-%d %H:%M:%S UTC")
                    .to_string()
            }
            #[cfg(not(feature = "serde"))]
            {
                "N/A".to_string()
            }
        };

        format!(
            r#"<!DOCTYPE html>
<html>
<head>
    <title>Benchmark Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .section {{ margin: 20px 0; }}
        .result {{ background-color: #f9f9f9; padding: 15px; margin: 10px 0; border-radius: 5px; }}
        .performance-table {{ width: 100%; border-collapse: collapse; }}
        .performance-table th, .performance-table td {{ 
            border: 1px solid #ddd; padding: 8px; text-align: left; 
        }}
        .performance-table th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Benchmark Report</h1>
        <p>Generated: {}</p>
        <p>Total Methods Tested: {}</p>
    </div>
    
    <div class="section">
        <h2>Performance Summary</h2>
        <table class="performance-table">
            <tr>
                <th>Method</th>
                <th>Category</th>
                <th>Mean Time (ms)</th>
                <th>Throughput (ops/sec)</th>
                <th>Quality Score</th>
            </tr>
            {}
        </table>
    </div>
    
    <div class="section">
        <h2>Insights</h2>
        {}
    </div>
    
    <div class="section">
        <h2>Recommendations</h2>
        <ul>
            {}
        </ul>
    </div>
</body>
</html>"#,
            generated_time,
            self.results.len(),
            self.results
                .iter()
                .map(|r| format!(
                    "<tr><td>{}</td><td>{:?}</td><td>{:.2}</td><td>{:.2}</td><td>{:.3}</td></tr>",
                    r.method_name,
                    r.category,
                    r.timing_stats.mean_time.as_millis(),
                    r.timing_stats.throughput,
                    r.quality_metrics.overall_score
                ))
                .collect::<Vec<_>>()
                .join("\n"),
            self.performance_insights
                .iter()
                .map(|insight| format!(
                    "<p><strong>{:?}:</strong> {}</p>",
                    insight.insight_type, insight.message
                ))
                .collect::<Vec<_>>()
                .join("\n"),
            self.recommendations
                .iter()
                .map(|rec| format!("<li>{}</li>", rec))
                .collect::<Vec<_>>()
                .join("\n")
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benchmark_config_creation() {
        let config = BenchmarkConfig::default();
        assert_eq!(config.warmup_iterations, 10);
        assert_eq!(config.benchmark_iterations, 100);
        assert_eq!(config.significance_level, 0.05);
    }

    #[test]
    fn test_benchmark_suite_creation() {
        let config = BenchmarkConfig::default();
        let suite = BenchmarkingSuite::new(config);
        assert_eq!(suite.results.len(), 0);
        assert_eq!(suite.reference_results.len(), 0);
    }

    #[test]
    fn test_timing_statistics_calculation() {
        let config = BenchmarkConfig::default();
        let suite = BenchmarkingSuite::new(config);

        let times = vec![
            Duration::from_millis(100),
            Duration::from_millis(200),
            Duration::from_millis(150),
        ];

        let stats = suite.calculate_timing_statistics(&times);
        assert!(stats.mean_time.as_millis() > 0);
        assert!(stats.throughput > 0.0);
    }

    #[test]
    fn test_quality_metrics_calculation() {
        let config = BenchmarkConfig::default();
        let suite = BenchmarkingSuite::new(config);

        let test_config = TestConfiguration {
            dataset_size: 1000,
            num_features: 10,
            model_type: "Test".to_string(),
            problem_type: ProblemType::BinaryClassification,
            parameters: HashMap::new(),
        };

        let metrics = suite
            .calculate_quality_metrics("SHAP", &test_config)
            .unwrap();
        assert!(metrics.overall_score > 0.0);
        assert!(metrics.fidelity > 0.0);
    }

    #[test]
    fn test_memory_statistics_calculation() {
        let config = BenchmarkConfig::default();
        let suite = BenchmarkingSuite::new(config);

        let snapshots = vec![(1000, 1500), (1200, 1800), (1100, 1600)];
        let stats = suite.calculate_memory_statistics(&snapshots);

        assert_eq!(stats.peak_memory, 1800);
        assert!(stats.avg_memory > 0);
    }

    #[test]
    fn test_test_configuration() {
        let config = TestConfiguration {
            dataset_size: 5000,
            num_features: 20,
            model_type: "RandomForest".to_string(),
            problem_type: ProblemType::MultiClassification,
            parameters: HashMap::new(),
        };

        assert_eq!(config.dataset_size, 5000);
        assert_eq!(config.num_features, 20);
        assert!(matches!(
            config.problem_type,
            ProblemType::MultiClassification
        ));
    }

    #[test]
    fn test_reference_comparison() {
        let comparison = ReferenceComparison {
            speed_improvement: 2.5,
            memory_improvement: 1.8,
            quality_difference: 0.05,
            is_significant: true,
            p_value: 0.01,
        };

        assert!(comparison.speed_improvement > 2.0);
        assert!(comparison.is_significant);
        assert!(comparison.p_value < 0.05);
    }

    #[test]
    fn test_performance_insight() {
        let insight = PerformanceInsight {
            insight_type: InsightType::Speed,
            message: "Method A is 3x faster than Method B".to_string(),
            severity: InsightSeverity::Info,
        };

        assert!(matches!(insight.insight_type, InsightType::Speed));
        assert!(matches!(insight.severity, InsightSeverity::Info));
        assert!(insight.message.contains("3x faster"));
    }

    #[test]
    fn test_html_report_generation() {
        let config = BenchmarkConfig::default();
        let report = BenchmarkReport {
            config,
            results: Vec::new(),
            category_summaries: HashMap::new(),
            performance_insights: Vec::new(),
            recommendations: vec!["Test recommendation".to_string()],
            #[cfg(feature = "serde")]
            generated_at: Utc::now(),
        };

        let html = report.to_html();
        assert!(html.contains("<!DOCTYPE html>"));
        assert!(html.contains("Benchmark Report"));
        assert!(html.contains("Test recommendation"));
    }
}
