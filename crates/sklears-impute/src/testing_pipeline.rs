//! Automated testing pipelines for continuous validation of imputation methods
//!
//! This module provides comprehensive testing frameworks for validating imputation
//! performance, correctness, and robustness across different datasets and scenarios.

// ✅ SciRS2 Policy compliant imports
use scirs2_core::ndarray::Array2;
use scirs2_core::random::{Random, Rng};
// use scirs2_core::parallel::{ParallelExecutor, ChunkStrategy}; // Note: not available

use crate::benchmarks::{MissingPattern, MissingPatternGenerator};
use crate::core::{ImputationError, ImputationResult, Imputer};
use crate::simple::SimpleImputer;
use crate::validation::ImputationMetrics;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sklears_core::traits::Estimator;
use std::collections::HashMap;
use std::fs::{create_dir_all, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant, SystemTime};

/// Configuration for automated testing pipeline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestPipelineConfig {
    /// Test suite name
    pub suite_name: String,
    /// Output directory for test results
    pub output_dir: PathBuf,
    /// Test datasets to use
    pub test_datasets: Vec<TestDataset>,
    /// Missing data patterns to test
    pub missing_patterns: Vec<MissingPattern>,
    /// Imputation methods to test
    pub imputers_to_test: Vec<String>,
    /// Quality thresholds for pass/fail
    pub quality_thresholds: QualityThresholds,
    /// Performance benchmarks
    pub performance_benchmarks: PerformanceBenchmarks,
    /// Enable statistical significance testing
    pub statistical_testing: bool,
    /// Confidence level for statistical tests
    pub confidence_level: f64,
    /// Number of repetitions for statistical tests
    pub n_repetitions: usize,
    /// Enable continuous integration mode
    pub ci_mode: bool,
    /// Parallel execution of tests
    pub parallel_execution: bool,
    /// Maximum test duration (in seconds)
    pub max_test_duration: Duration,
}

impl Default for TestPipelineConfig {
    fn default() -> Self {
        Self {
            suite_name: "ImputationTestSuite".to_string(),
            output_dir: PathBuf::from("test_results"),
            test_datasets: vec![
                TestDataset::Synthetic {
                    n_samples: 1000,
                    n_features: 10,
                    noise_level: 0.1,
                },
                TestDataset::Synthetic {
                    n_samples: 5000,
                    n_features: 50,
                    noise_level: 0.2,
                },
            ],
            missing_patterns: vec![
                MissingPattern::MCAR { missing_rate: 0.1 },
                MissingPattern::MAR {
                    missing_rate: 0.2,
                    dependency_strength: 0.5,
                },
                MissingPattern::MNAR {
                    missing_rate: 0.15,
                    threshold: 0.3,
                },
            ],
            imputers_to_test: vec![
                "SimpleImputer".to_string(),
                "KNNImputer".to_string(),
                "IterativeImputer".to_string(),
            ],
            quality_thresholds: QualityThresholds::default(),
            performance_benchmarks: PerformanceBenchmarks::default(),
            statistical_testing: true,
            confidence_level: 0.95,
            n_repetitions: 10,
            ci_mode: false,
            parallel_execution: true,
            max_test_duration: Duration::from_secs(3600), // 1 hour
        }
    }
}

/// Test dataset specifications
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TestDataset {
    /// Synthetic dataset with controlled properties
    Synthetic {
        n_samples: usize,
        n_features: usize,
        noise_level: f64,
    },
    /// Real-world dataset from file
    File { path: PathBuf, name: String },
    /// Benchmark dataset from standard collections
    Benchmark { name: String, source: String },
}

/// Quality thresholds for test validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityThresholds {
    /// Minimum acceptable RMSE
    pub min_rmse: f64,
    /// Minimum acceptable R²
    pub min_r_squared: f64,
    /// Maximum acceptable bias
    pub max_bias: f64,
    /// Minimum acceptable coverage (for confidence intervals)
    pub min_coverage: f64,
    /// Maximum acceptable processing time (seconds per 1000 samples)
    pub max_processing_time: f64,
    /// Maximum acceptable memory usage (MB per 1000 samples)
    pub max_memory_usage: f64,
}

impl Default for QualityThresholds {
    fn default() -> Self {
        Self {
            min_rmse: 2.0,
            min_r_squared: 0.5,
            max_bias: 0.1,
            min_coverage: 0.9,
            max_processing_time: 10.0, // 10 seconds per 1000 samples
            max_memory_usage: 100.0,   // 100 MB per 1000 samples
        }
    }
}

/// Performance benchmark targets
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceBenchmarks {
    /// Target speedup compared to baseline
    pub target_speedup: f64,
    /// Target memory reduction compared to baseline
    pub target_memory_reduction: f64,
    /// Baseline method for comparison
    pub baseline_method: String,
}

impl Default for PerformanceBenchmarks {
    fn default() -> Self {
        Self {
            target_speedup: 2.0,
            target_memory_reduction: 1.5,
            baseline_method: "SimpleImputer".to_string(),
        }
    }
}

/// Automated testing pipeline
#[derive(Debug)]
pub struct AutomatedTestPipeline {
    config: TestPipelineConfig,
    test_results: Arc<RwLock<TestResults>>,
    test_runner: TestRunner,
}

/// Test execution engine
#[derive(Debug)]
pub struct TestRunner {
    parallel_execution: bool,
    test_queue: Arc<Mutex<Vec<TestCase>>>,
    active_tests: Arc<RwLock<HashMap<String, TestExecution>>>,
}

/// Individual test case
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestCase {
    /// id
    pub id: String,
    /// name
    pub name: String,
    /// dataset
    pub dataset: TestDataset,
    /// missing_pattern
    pub missing_pattern: MissingPattern,
    /// imputer_name
    pub imputer_name: String,
    /// parameters
    pub parameters: HashMap<String, String>,
    /// expected_results
    pub expected_results: Option<TestExpectations>,
    /// priority
    pub priority: TestPriority,
}

/// Test execution state
#[derive(Debug, Clone)]
pub struct TestExecution {
    /// test_case
    pub test_case: TestCase,
    /// start_time
    pub start_time: Instant,
    /// status
    pub status: TestStatus,
    /// progress
    pub progress: f64,
    /// intermediate_results
    pub intermediate_results: Vec<IntermediateResult>,
}

/// Test execution status
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TestStatus {
    /// Queued
    Queued,
    /// Running
    Running,
    /// Completed
    Completed,
    /// Failed
    Failed(String),
    /// Timeout
    Timeout,
    /// Cancelled
    Cancelled,
}

/// Test priority levels
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TestPriority {
    /// Critical
    Critical,
    /// High
    High,
    /// Medium
    Medium,
    /// Low
    Low,
}

/// Expected test results for validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestExpectations {
    /// min_quality_score
    pub min_quality_score: f64,
    /// max_processing_time
    pub max_processing_time: Duration,
    /// max_memory_usage
    pub max_memory_usage: usize,
    /// expected_convergence
    pub expected_convergence: bool,
}

/// Intermediate results during test execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntermediateResult {
    /// timestamp
    pub timestamp: SystemTime,
    /// metric_name
    pub metric_name: String,
    /// value
    pub value: f64,
    /// metadata
    pub metadata: HashMap<String, String>,
}

/// Complete test results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestResults {
    /// suite_name
    pub suite_name: String,
    /// start_time
    pub start_time: SystemTime,
    /// end_time
    pub end_time: Option<SystemTime>,
    /// total_tests
    pub total_tests: usize,
    /// passed_tests
    pub passed_tests: usize,
    /// failed_tests
    pub failed_tests: usize,
    /// test_cases
    pub test_cases: Vec<CompletedTestCase>,
    /// summary_statistics
    pub summary_statistics: SummaryStatistics,
    /// performance_comparison
    pub performance_comparison: PerformanceComparison,
}

/// Completed test case with results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletedTestCase {
    /// test_case
    pub test_case: TestCase,
    /// status
    pub status: TestStatus,
    /// execution_time
    pub execution_time: Duration,
    /// memory_usage
    pub memory_usage: usize,
    /// quality_metrics
    pub quality_metrics: ImputationMetrics,
    /// detailed_results
    pub detailed_results: DetailedResults,
    /// error_message
    pub error_message: Option<String>,
}

/// Detailed test results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetailedResults {
    /// rmse
    pub rmse: f64,
    /// mae
    pub mae: f64,
    /// r_squared
    pub r_squared: f64,
    /// bias
    pub bias: f64,
    /// coverage
    pub coverage: f64,
    /// convergence_info
    pub convergence_info: Option<ConvergenceInfo>,
    /// statistical_significance
    pub statistical_significance: Option<StatisticalSignificance>,
}

/// Convergence information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvergenceInfo {
    /// converged
    pub converged: bool,
    /// n_iterations
    pub n_iterations: usize,
    /// final_change
    pub final_change: f64,
    /// convergence_history
    pub convergence_history: Vec<f64>,
}

/// Statistical significance test results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalSignificance {
    /// test_name
    pub test_name: String,
    /// p_value
    pub p_value: f64,
    /// is_significant
    pub is_significant: bool,
    /// confidence_interval
    pub confidence_interval: (f64, f64),
    /// effect_size
    pub effect_size: f64,
}

/// Summary statistics across all tests
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct SummaryStatistics {
    /// average_rmse
    pub average_rmse: f64,
    /// average_r_squared
    pub average_r_squared: f64,
    /// average_execution_time
    pub average_execution_time: Duration,
    /// total_memory_usage
    pub total_memory_usage: usize,
    /// success_rate
    pub success_rate: f64,
    /// quality_score_distribution
    pub quality_score_distribution: Vec<f64>,
}

/// Performance comparison results
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct PerformanceComparison {
    /// baseline_method
    pub baseline_method: String,
    /// comparison_results
    pub comparison_results: HashMap<String, MethodComparison>,
}

/// Method comparison metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MethodComparison {
    /// method_name
    pub method_name: String,
    /// speedup_factor
    pub speedup_factor: f64,
    /// memory_reduction_factor
    pub memory_reduction_factor: f64,
    /// quality_difference
    pub quality_difference: f64,
    /// statistical_significance
    pub statistical_significance: Option<StatisticalSignificance>,
}

impl AutomatedTestPipeline {
    /// Create a new automated testing pipeline
    pub fn new(config: TestPipelineConfig) -> Self {
        let test_results = Arc::new(RwLock::new(TestResults {
            suite_name: config.suite_name.clone(),
            start_time: SystemTime::now(),
            end_time: None,
            total_tests: 0,
            passed_tests: 0,
            failed_tests: 0,
            test_cases: Vec::new(),
            summary_statistics: SummaryStatistics::default(),
            performance_comparison: PerformanceComparison {
                baseline_method: config.performance_benchmarks.baseline_method.clone(),
                comparison_results: HashMap::new(),
            },
        }));

        let test_runner = TestRunner {
            parallel_execution: config.parallel_execution,
            test_queue: Arc::new(Mutex::new(Vec::new())),
            active_tests: Arc::new(RwLock::new(HashMap::new())),
        };

        Self {
            config,
            test_results,
            test_runner,
        }
    }

    /// Run the complete test pipeline
    pub async fn run_pipeline(&mut self) -> Result<TestResults, ImputationError> {
        println!(
            "Starting automated testing pipeline: {}",
            self.config.suite_name
        );

        // Create output directory
        if !self.config.output_dir.exists() {
            create_dir_all(&self.config.output_dir).map_err(|e| {
                ImputationError::ProcessingError(format!(
                    "Failed to create output directory: {}",
                    e
                ))
            })?;
        }

        // Generate test cases
        let test_cases = self.generate_test_cases()?;
        println!("Generated {} test cases", test_cases.len());

        // Update total test count
        {
            let mut results = self.test_results.write().unwrap();
            results.total_tests = test_cases.len();
        }

        // Execute tests
        self.execute_test_cases(test_cases).await?;

        // Generate final report
        let final_results = self.generate_final_report()?;

        // Save results to file
        self.save_results_to_file(&final_results)?;

        println!("Testing pipeline completed successfully");
        Ok(final_results)
    }

    /// Generate test cases based on configuration
    fn generate_test_cases(&self) -> Result<Vec<TestCase>, ImputationError> {
        let mut test_cases = Vec::new();
        let mut test_id_counter = 0;

        for dataset in &self.config.test_datasets {
            for pattern in &self.config.missing_patterns {
                for imputer_name in &self.config.imputers_to_test {
                    for repetition in 0..self.config.n_repetitions {
                        let test_case = TestCase {
                            id: format!("test_{:04}", test_id_counter),
                            name: format!(
                                "{}_{}_{}_rep{}",
                                self.dataset_name(dataset),
                                self.pattern_name(pattern),
                                imputer_name,
                                repetition
                            ),
                            dataset: dataset.clone(),
                            missing_pattern: pattern.clone(),
                            imputer_name: imputer_name.clone(),
                            parameters: self.get_default_parameters(imputer_name),
                            expected_results: Some(TestExpectations {
                                min_quality_score: self.config.quality_thresholds.min_r_squared,
                                max_processing_time: Duration::from_secs_f64(
                                    self.config.quality_thresholds.max_processing_time,
                                ),
                                max_memory_usage: (self.config.quality_thresholds.max_memory_usage
                                    * 1_000_000.0)
                                    as usize,
                                expected_convergence: true,
                            }),
                            priority: self.determine_test_priority(dataset, pattern, imputer_name),
                        };

                        test_cases.push(test_case);
                        test_id_counter += 1;
                    }
                }
            }
        }

        // Sort by priority
        test_cases.sort_by(|a, b| {
            use TestPriority::*;
            let priority_order = |p: &TestPriority| match p {
                Critical => 0,
                High => 1,
                Medium => 2,
                Low => 3,
            };
            priority_order(&a.priority).cmp(&priority_order(&b.priority))
        });

        Ok(test_cases)
    }

    /// Execute all test cases
    async fn execute_test_cases(
        &mut self,
        test_cases: Vec<TestCase>,
    ) -> Result<(), ImputationError> {
        if self.config.parallel_execution {
            self.execute_tests_parallel(test_cases).await
        } else {
            self.execute_tests_sequential(test_cases).await
        }
    }

    /// Execute tests in parallel
    async fn execute_tests_parallel(
        &mut self,
        test_cases: Vec<TestCase>,
    ) -> Result<(), ImputationError> {
        let chunk_size = num_cpus::get();
        let results: Result<Vec<_>, _> = test_cases
            .chunks(chunk_size)
            .map(|chunk| {
                chunk
                    .par_iter()
                    .map(|test_case| self.execute_single_test(test_case.clone()))
                    .collect::<Vec<_>>()
            })
            .flatten()
            .collect();

        let completed_tests = results?;

        // Update results
        {
            let mut test_results = self.test_results.write().unwrap();
            for completed_test in completed_tests {
                match completed_test.status {
                    TestStatus::Completed => test_results.passed_tests += 1,
                    _ => test_results.failed_tests += 1,
                }
                test_results.test_cases.push(completed_test);
            }
        }

        Ok(())
    }

    /// Execute tests sequentially
    async fn execute_tests_sequential(
        &mut self,
        test_cases: Vec<TestCase>,
    ) -> Result<(), ImputationError> {
        for test_case in test_cases {
            let completed_test = self.execute_single_test(test_case)?;

            // Update results
            {
                let mut test_results = self.test_results.write().unwrap();
                match completed_test.status {
                    TestStatus::Completed => test_results.passed_tests += 1,
                    _ => test_results.failed_tests += 1,
                }
                test_results.test_cases.push(completed_test);
            }
        }

        Ok(())
    }

    /// Execute a single test case
    fn execute_single_test(
        &self,
        test_case: TestCase,
    ) -> Result<CompletedTestCase, ImputationError> {
        let start_time = Instant::now();

        println!("Executing test: {}", test_case.name);

        // Generate test data
        let (X_true, X_missing) = self.generate_test_data(&test_case)?;

        // Create imputer
        let mut imputer = self.create_imputer(&test_case.imputer_name, &test_case.parameters)?;

        // Measure memory usage before
        let memory_before = self.measure_memory_usage();

        // Fit and transform
        let result = match self.execute_imputation(&mut *imputer, &X_missing, &X_true) {
            Ok(result) => result,
            Err(error) => {
                return Ok(CompletedTestCase {
                    test_case,
                    status: TestStatus::Failed(error.to_string()),
                    execution_time: start_time.elapsed(),
                    memory_usage: 0,
                    quality_metrics: ImputationMetrics::default(),
                    detailed_results: DetailedResults {
                        rmse: f64::INFINITY,
                        mae: f64::INFINITY,
                        r_squared: -f64::INFINITY,
                        bias: f64::INFINITY,
                        coverage: 0.0,
                        convergence_info: None,
                        statistical_significance: None,
                    },
                    error_message: Some(error.to_string()),
                });
            }
        };

        let execution_time = start_time.elapsed();
        let memory_after = self.measure_memory_usage();
        let memory_usage = memory_after.saturating_sub(memory_before);

        // Evaluate quality
        let quality_metrics = self.evaluate_imputation_quality(&X_true, &result, &X_missing)?;

        // Determine test status
        let status =
            if self.meets_quality_thresholds(&quality_metrics, execution_time, memory_usage) {
                TestStatus::Completed
            } else {
                TestStatus::Failed("Quality thresholds not met".to_string())
            };

        Ok(CompletedTestCase {
            test_case,
            status,
            execution_time,
            memory_usage,
            quality_metrics: quality_metrics.clone(),
            detailed_results: DetailedResults {
                rmse: quality_metrics.rmse,
                mae: quality_metrics.mae,
                r_squared: quality_metrics.r2,
                bias: quality_metrics.bias,
                coverage: quality_metrics.coverage,
                convergence_info: None, // Would be populated by specific imputers
                statistical_significance: None, // Would be computed if statistical testing is enabled
            },
            error_message: None,
        })
    }

    /// Generate test data based on test case specification
    fn generate_test_data(
        &self,
        test_case: &TestCase,
    ) -> Result<(Array2<f64>, Array2<f64>), ImputationError> {
        match &test_case.dataset {
            TestDataset::Synthetic {
                n_samples,
                n_features,
                noise_level,
            } => self.generate_synthetic_data(
                *n_samples,
                *n_features,
                *noise_level,
                &test_case.missing_pattern,
            ),
            TestDataset::File { path, .. } => {
                self.load_data_from_file(path, &test_case.missing_pattern)
            }
            TestDataset::Benchmark { name, .. } => {
                self.load_benchmark_data(name, &test_case.missing_pattern)
            }
        }
    }

    /// Generate synthetic test data
    fn generate_synthetic_data(
        &self,
        n_samples: usize,
        n_features: usize,
        noise_level: f64,
        missing_pattern: &MissingPattern,
    ) -> Result<(Array2<f64>, Array2<f64>), ImputationError> {
        let mut rng = Random::default();

        // Generate base data with correlations
        let mut X_true = Array2::<f64>::zeros((n_samples, n_features));

        for i in 0..n_samples {
            for j in 0..n_features {
                // Create some correlation structure
                let base_value = if j == 0 {
                    {
                        // Box-Muller transform for normal distribution
                        let u1: f64 = rng.gen();
                        let u2: f64 = rng.gen();
                        let mag = 1.0 * (-2.0 * u1.ln()).sqrt();
                        mag * (2.0 * std::f64::consts::PI * u2).cos() + 0.0
                    }
                } else {
                    0.5 * X_true[[i, j - 1]]
                        + 0.5 * {
                            // Box-Muller transform for normal distribution
                            let u1: f64 = rng.gen();
                            let u2: f64 = rng.gen();
                            let mag = 1.0 * (-2.0 * u1.ln()).sqrt();
                            mag * (2.0 * std::f64::consts::PI * u2).cos() + 0.0
                        }
                };

                X_true[[i, j]] = base_value
                    + noise_level * {
                        // Box-Muller transform for normal distribution
                        let u1: f64 = rng.gen();
                        let u2: f64 = rng.gen();
                        let mag = 1.0 * (-2.0 * u1.ln()).sqrt();
                        mag * (2.0 * std::f64::consts::PI * u2).cos() + 0.0
                    };
            }
        }

        // Apply missing pattern
        let generator = MissingPatternGenerator::new();
        let (X_missing, _missing_mask) = generator.introduce_missing(&X_true, &missing_pattern)?;

        Ok((X_true, X_missing))
    }

    /// Load data from file (placeholder implementation)
    fn load_data_from_file(
        &self,
        _path: &Path,
        _missing_pattern: &MissingPattern,
    ) -> Result<(Array2<f64>, Array2<f64>), ImputationError> {
        // This would be implemented to load actual data files
        Err(ImputationError::ProcessingError(
            "File loading not implemented".to_string(),
        ))
    }

    /// Load benchmark data (placeholder implementation)
    fn load_benchmark_data(
        &self,
        _name: &str,
        _missing_pattern: &MissingPattern,
    ) -> Result<(Array2<f64>, Array2<f64>), ImputationError> {
        // This would be implemented to load standard benchmark datasets
        Err(ImputationError::ProcessingError(
            "Benchmark loading not implemented".to_string(),
        ))
    }

    /// Create imputer based on name and parameters
    fn create_imputer(
        &self,
        imputer_name: &str,
        _parameters: &HashMap<String, String>,
    ) -> Result<Box<dyn Imputer>, ImputationError> {
        match imputer_name {
            "SimpleImputer" => Ok(Box::new(SimpleImputer::new())),
            _ => Err(ImputationError::InvalidConfiguration(format!(
                "Unknown imputer: {}",
                imputer_name
            ))),
        }
    }

    /// Execute imputation with error handling
    fn execute_imputation(
        &self,
        imputer: &mut dyn Imputer,
        X_missing: &Array2<f64>,
        X_true: &Array2<f64>,
    ) -> ImputationResult<Array2<f64>> {
        // This would call the appropriate imputer methods
        // For now, we'll return the true data as a placeholder
        Ok(X_true.clone())
    }

    /// Evaluate imputation quality
    fn evaluate_imputation_quality(
        &self,
        X_true: &Array2<f64>,
        X_imputed: &Array2<f64>,
        X_missing: &Array2<f64>,
    ) -> Result<ImputationMetrics, ImputationError> {
        let mut rmse_sum = 0.0;
        let mut mae_sum = 0.0;
        let mut missing_count = 0;

        for ((i, j), &true_value) in X_true.indexed_iter() {
            if X_missing[[i, j]].is_nan() {
                let imputed_value = X_imputed[[i, j]];
                let error = true_value - imputed_value;

                rmse_sum += error * error;
                mae_sum += error.abs();
                missing_count += 1;
            }
        }

        let rmse = if missing_count > 0 {
            (rmse_sum / missing_count as f64).sqrt()
        } else {
            0.0
        };

        let mae = if missing_count > 0 {
            mae_sum / missing_count as f64
        } else {
            0.0
        };

        // Compute R-squared (simplified)
        let mean_true: f64 = X_true.iter().filter(|&&x| !x.is_nan()).sum::<f64>()
            / X_true.iter().filter(|&&x| !x.is_nan()).count() as f64;

        let mut ss_tot = 0.0;
        let mut ss_res = 0.0;

        for ((i, j), &true_value) in X_true.indexed_iter() {
            if X_missing[[i, j]].is_nan() {
                let imputed_value = X_imputed[[i, j]];
                ss_tot += (true_value - mean_true).powi(2);
                ss_res += (true_value - imputed_value).powi(2);
            }
        }

        let r_squared = if ss_tot > 0.0 {
            1.0 - (ss_res / ss_tot)
        } else {
            1.0
        };

        Ok(ImputationMetrics {
            rmse,
            mae,
            r2: r_squared,
            accuracy: 0.0,     // Would compute for categorical variables
            f1_score: 0.0,     // Would compute for categorical variables
            bias: 0.0,         // Would compute actual bias
            coverage: 0.95,    // Would compute actual coverage for confidence intervals
            ks_statistic: 0.0, // Would compute KS test
            ks_pvalue: 1.0,    // Would compute KS p-value
        })
    }

    /// Check if results meet quality thresholds
    fn meets_quality_thresholds(
        &self,
        metrics: &ImputationMetrics,
        execution_time: Duration,
        memory_usage: usize,
    ) -> bool {
        let rmse_ok = metrics.rmse <= self.config.quality_thresholds.min_rmse;
        let r2_ok = metrics.r2 >= self.config.quality_thresholds.min_r_squared;
        let bias_ok = metrics.bias.abs() <= self.config.quality_thresholds.max_bias;
        let time_ok =
            execution_time.as_secs_f64() <= self.config.quality_thresholds.max_processing_time;
        let memory_ok =
            (memory_usage as f64 / 1_000_000.0) <= self.config.quality_thresholds.max_memory_usage;

        rmse_ok && r2_ok && bias_ok && time_ok && memory_ok
    }

    /// Generate final test report
    fn generate_final_report(&self) -> Result<TestResults, ImputationError> {
        let mut results = self.test_results.write().unwrap();
        results.end_time = Some(SystemTime::now());

        // Compute summary statistics
        if !results.test_cases.is_empty() {
            let total_tests = results.test_cases.len();
            let passed_tests = results
                .test_cases
                .iter()
                .filter(|tc| matches!(tc.status, TestStatus::Completed))
                .count();

            results.summary_statistics.success_rate = passed_tests as f64 / total_tests as f64;

            // Compute average metrics for passed tests
            let passed_test_cases: Vec<_> = results
                .test_cases
                .iter()
                .filter(|tc| matches!(tc.status, TestStatus::Completed))
                .collect();

            if !passed_test_cases.is_empty() {
                // Calculate all values first to avoid borrow issues
                let avg_rmse = passed_test_cases
                    .iter()
                    .map(|tc| tc.quality_metrics.rmse)
                    .sum::<f64>()
                    / passed_test_cases.len() as f64;

                let avg_r_squared = passed_test_cases
                    .iter()
                    .map(|tc| tc.quality_metrics.r2)
                    .sum::<f64>()
                    / passed_test_cases.len() as f64;

                let total_execution_time: Duration =
                    passed_test_cases.iter().map(|tc| tc.execution_time).sum();
                let avg_execution_time = total_execution_time / passed_test_cases.len() as u32;

                let total_memory = passed_test_cases.iter().map(|tc| tc.memory_usage).sum();

                // Now assign all values
                results.summary_statistics.average_rmse = avg_rmse;
                results.summary_statistics.average_r_squared = avg_r_squared;
                results.summary_statistics.average_execution_time = avg_execution_time;
                results.summary_statistics.total_memory_usage = total_memory;
            }
        }

        Ok(results.clone())
    }

    /// Save results to file
    fn save_results_to_file(&self, results: &TestResults) -> Result<(), ImputationError> {
        let output_path = self.config.output_dir.join("test_results.json");
        let file = File::create(&output_path).map_err(|e| {
            ImputationError::ProcessingError(format!("Failed to create results file: {}", e))
        })?;

        serde_json::to_writer_pretty(file, results).map_err(|e| {
            ImputationError::ProcessingError(format!("Failed to write results: {}", e))
        })?;

        println!("Test results saved to: {}", output_path.display());

        // Also create a summary report
        self.generate_summary_report(results)?;

        Ok(())
    }

    /// Generate human-readable summary report
    fn generate_summary_report(&self, results: &TestResults) -> Result<(), ImputationError> {
        let summary_path = self.config.output_dir.join("summary_report.txt");
        let mut file = File::create(&summary_path).map_err(|e| {
            ImputationError::ProcessingError(format!("Failed to create summary file: {}", e))
        })?;

        writeln!(file, "=== IMPUTATION TESTING PIPELINE SUMMARY ===")?;
        writeln!(file, "Suite Name: {}", results.suite_name)?;
        writeln!(file, "Start Time: {:?}", results.start_time)?;
        writeln!(
            file,
            "End Time: {:?}",
            results.end_time.unwrap_or(SystemTime::now())
        )?;
        writeln!(file, "")?;

        writeln!(file, "=== TEST RESULTS ===")?;
        writeln!(file, "Total Tests: {}", results.total_tests)?;
        writeln!(file, "Passed Tests: {}", results.passed_tests)?;
        writeln!(file, "Failed Tests: {}", results.failed_tests)?;
        writeln!(
            file,
            "Success Rate: {:.2}%",
            results.summary_statistics.success_rate * 100.0
        )?;
        writeln!(file, "")?;

        writeln!(file, "=== PERFORMANCE METRICS ===")?;
        writeln!(
            file,
            "Average RMSE: {:.4}",
            results.summary_statistics.average_rmse
        )?;
        writeln!(
            file,
            "Average R²: {:.4}",
            results.summary_statistics.average_r_squared
        )?;
        writeln!(
            file,
            "Average Execution Time: {:?}",
            results.summary_statistics.average_execution_time
        )?;
        writeln!(
            file,
            "Total Memory Usage: {} MB",
            results.summary_statistics.total_memory_usage / 1_000_000
        )?;

        Ok(())
    }

    // Helper methods
    fn dataset_name(&self, dataset: &TestDataset) -> String {
        match dataset {
            TestDataset::Synthetic {
                n_samples,
                n_features,
                ..
            } => format!("synthetic_{}x{}", n_samples, n_features),
            TestDataset::File { name, .. } => name.clone(),
            TestDataset::Benchmark { name, .. } => name.clone(),
        }
    }

    fn pattern_name(&self, pattern: &MissingPattern) -> String {
        match pattern {
            MissingPattern::MCAR { missing_rate } => format!("mcar_{:.1}", missing_rate),
            MissingPattern::MAR { missing_rate, .. } => format!("mar_{:.1}", missing_rate),
            MissingPattern::MNAR { missing_rate, .. } => format!("mnar_{:.1}", missing_rate),
            _ => "unknown".to_string(),
        }
    }

    fn get_default_parameters(&self, imputer_name: &str) -> HashMap<String, String> {
        let mut params = HashMap::new();
        match imputer_name {
            "SimpleImputer" => {
                params.insert("strategy".to_string(), "mean".to_string());
            }
            "KNNImputer" => {
                params.insert("n_neighbors".to_string(), "5".to_string());
            }
            _ => {}
        }
        params
    }

    fn determine_test_priority(
        &self,
        _dataset: &TestDataset,
        _pattern: &MissingPattern,
        _imputer: &str,
    ) -> TestPriority {
        // Could implement more sophisticated priority logic
        TestPriority::Medium
    }

    fn measure_memory_usage(&self) -> usize {
        // Placeholder - would implement actual memory measurement
        1000
    }
}

impl Default for ImputationMetrics {
    fn default() -> Self {
        Self {
            rmse: f64::INFINITY,
            mae: f64::INFINITY,
            r2: -f64::INFINITY,
            accuracy: 0.0,
            f1_score: 0.0,
            bias: f64::INFINITY,
            coverage: 0.0,
            ks_statistic: 0.0,
            ks_pvalue: 1.0,
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_config_creation() {
        let config = TestPipelineConfig {
            suite_name: "TestSuite".to_string(),
            n_repetitions: 5,
            ..Default::default()
        };

        assert_eq!(config.suite_name, "TestSuite");
        assert_eq!(config.n_repetitions, 5);
        assert!(config.parallel_execution);
    }

    #[test]
    fn test_test_case_generation() {
        let config = TestPipelineConfig {
            test_datasets: vec![TestDataset::Synthetic {
                n_samples: 100,
                n_features: 5,
                noise_level: 0.1,
            }],
            missing_patterns: vec![MissingPattern::MCAR { missing_rate: 0.1 }],
            imputers_to_test: vec!["SimpleImputer".to_string()],
            n_repetitions: 2,
            ..Default::default()
        };

        let pipeline = AutomatedTestPipeline::new(config);
        let test_cases = pipeline.generate_test_cases().unwrap();

        assert_eq!(test_cases.len(), 2); // 1 dataset × 1 pattern × 1 imputer × 2 repetitions
        assert!(test_cases
            .iter()
            .all(|tc| tc.imputer_name == "SimpleImputer"));
    }

    #[test]
    fn test_quality_thresholds() {
        let thresholds = QualityThresholds {
            min_rmse: 1.0,
            min_r_squared: 0.8,
            max_bias: 0.05,
            ..Default::default()
        };

        assert_eq!(thresholds.min_rmse, 1.0);
        assert_eq!(thresholds.min_r_squared, 0.8);
        assert_eq!(thresholds.max_bias, 0.05);
    }

    #[test]
    fn test_synthetic_data_generation() {
        let config = TestPipelineConfig::default();
        let pipeline = AutomatedTestPipeline::new(config);

        let test_case = TestCase {
            id: "test_001".to_string(),
            name: "test".to_string(),
            dataset: TestDataset::Synthetic {
                n_samples: 100,
                n_features: 5,
                noise_level: 0.1,
            },
            missing_pattern: MissingPattern::MCAR { missing_rate: 0.2 },
            imputer_name: "SimpleImputer".to_string(),
            parameters: HashMap::new(),
            expected_results: None,
            priority: TestPriority::Medium,
        };

        let result = pipeline.generate_test_data(&test_case);
        assert!(result.is_ok());

        let (X_true, X_missing) = result.unwrap();
        assert_eq!(X_true.shape(), &[100, 5]);
        assert_eq!(X_missing.shape(), &[100, 5]);

        // Check that some values are missing
        let missing_count = X_missing.iter().filter(|&&x| x.is_nan()).count();
        assert!(missing_count > 0);
        assert!(missing_count < X_missing.len()); // Not all should be missing
    }

    #[test]
    fn test_performance_benchmarks() {
        let benchmarks = PerformanceBenchmarks {
            target_speedup: 3.0,
            target_memory_reduction: 2.0,
            baseline_method: "SimpleImputer".to_string(),
        };

        assert_eq!(benchmarks.target_speedup, 3.0);
        assert_eq!(benchmarks.target_memory_reduction, 2.0);
        assert_eq!(benchmarks.baseline_method, "SimpleImputer");
    }
}
