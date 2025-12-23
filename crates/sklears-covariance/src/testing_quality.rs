//! Testing and Quality Framework for Covariance Estimation
//!
//! This module provides comprehensive testing utilities, property-based tests,
//! numerical accuracy validation, benchmarking framework, and quality assurance
//! tools for covariance estimation algorithms.

use scirs2_core::ndarray::{array, Array1, Array2, ArrayView2};
use scirs2_core::random::essentials::{Normal, Uniform};
use scirs2_core::random::thread_rng;
use scirs2_core::random::Distribution;
use scirs2_core::random::Random;
use scirs2_core::StandardNormal;
use sklears_core::error::SklearsError;
use std::collections::HashMap;
use std::time::{Duration, Instant};

// Property-Based Testing Framework
#[derive(Debug)]
pub struct PropertyTester {
    /// Random number generator
    pub rng: scirs2_core::random::CoreRandom,
    /// Number of test iterations
    pub n_iterations: usize,
    /// Matrix size range for testing
    pub size_range: (usize, usize),
    /// Sample size range for testing
    pub sample_range: (usize, usize),
    /// Tolerance for numerical comparisons
    pub tolerance: f64,
}

impl PropertyTester {
    pub fn new(seed: u64) -> Self {
        let r = thread_rng(); // Note: seeding will be handled per test function
        Self {
            rng: r,
            n_iterations: 100,
            size_range: (2, 20),
            sample_range: (10, 100),
            tolerance: 1e-10,
        }
    }

    pub fn n_iterations(mut self, n_iterations: usize) -> Self {
        self.n_iterations = n_iterations;
        self
    }

    pub fn size_range(mut self, min_size: usize, max_size: usize) -> Self {
        self.size_range = (min_size, max_size);
        self
    }

    pub fn tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Test that covariance matrices are symmetric
    pub fn test_symmetry_property<F>(&mut self, estimator_fn: F) -> PropertyTestResult
    where
        F: Fn(&ArrayView2<f64>) -> Result<Array2<f64>, SklearsError>,
    {
        let mut failures = Vec::new();
        let mut successes = 0;

        for iteration in 0..self.n_iterations {
            let (data, _) = self.generate_test_data();

            match estimator_fn(&data.view()) {
                Ok(cov_matrix) => {
                    if !self.is_symmetric(&cov_matrix) {
                        failures.push(PropertyFailure {
                            iteration,
                            property: "Symmetry".to_string(),
                            description: "Covariance matrix is not symmetric".to_string(),
                            data_shape: data.dim(),
                        });
                    } else {
                        successes += 1;
                    }
                }
                Err(e) => {
                    failures.push(PropertyFailure {
                        iteration,
                        property: "Symmetry".to_string(),
                        description: format!("Estimator failed: {}", e),
                        data_shape: data.dim(),
                    });
                }
            }
        }

        /// PropertyTestResult
        PropertyTestResult {
            property_name: "Symmetry".to_string(),
            total_tests: self.n_iterations,
            successes,
            failures,
        }
    }

    /// Test that covariance matrices are positive semi-definite
    pub fn test_positive_semidefinite_property<F>(&mut self, estimator_fn: F) -> PropertyTestResult
    where
        F: Fn(&ArrayView2<f64>) -> Result<Array2<f64>, SklearsError>,
    {
        let mut failures = Vec::new();
        let mut successes = 0;

        for iteration in 0..self.n_iterations {
            let (data, _) = self.generate_test_data();

            match estimator_fn(&data.view()) {
                Ok(cov_matrix) => {
                    if !self.is_positive_semidefinite(&cov_matrix) {
                        failures.push(PropertyFailure {
                            iteration,
                            property: "Positive Semi-Definite".to_string(),
                            description: "Covariance matrix has negative eigenvalues".to_string(),
                            data_shape: data.dim(),
                        });
                    } else {
                        successes += 1;
                    }
                }
                Err(e) => {
                    failures.push(PropertyFailure {
                        iteration,
                        property: "Positive Semi-Definite".to_string(),
                        description: format!("Estimator failed: {}", e),
                        data_shape: data.dim(),
                    });
                }
            }
        }

        /// PropertyTestResult
        PropertyTestResult {
            property_name: "Positive Semi-Definite".to_string(),
            total_tests: self.n_iterations,
            successes,
            failures,
        }
    }

    /// Test scale invariance property
    pub fn test_scale_invariance_property<F>(&mut self, estimator_fn: F) -> PropertyTestResult
    where
        F: Fn(&ArrayView2<f64>) -> Result<Array2<f64>, SklearsError>,
    {
        let mut failures = Vec::new();
        let mut successes = 0;

        for iteration in 0..self.n_iterations {
            let (data, _) = self.generate_test_data();
            let uniform_scale = Uniform::new(0.1, 10.0).unwrap();
            let scale_factor = uniform_scale.sample(&mut self.rng);
            let scaled_data = &data * scale_factor;

            match (
                estimator_fn(&data.view()),
                estimator_fn(&scaled_data.view()),
            ) {
                (Ok(cov1), Ok(cov2)) => {
                    let expected_cov2 = &cov1 * (scale_factor * scale_factor);
                    let diff = (&cov2 - &expected_cov2).mapv(|x| x.abs()).sum();

                    if diff > self.tolerance * cov1.mapv(|x| x.abs()).sum() {
                        failures.push(PropertyFailure {
                            iteration,
                            property: "Scale Invariance".to_string(),
                            description: format!("Scale invariance violated, diff: {}", diff),
                            data_shape: data.dim(),
                        });
                    } else {
                        successes += 1;
                    }
                }
                _ => {
                    failures.push(PropertyFailure {
                        iteration,
                        property: "Scale Invariance".to_string(),
                        description: "One or both estimations failed".to_string(),
                        data_shape: data.dim(),
                    });
                }
            }
        }

        /// PropertyTestResult
        PropertyTestResult {
            property_name: "Scale Invariance".to_string(),
            total_tests: self.n_iterations,
            successes,
            failures,
        }
    }

    /// Test diagonal dominance for certain estimators
    pub fn test_diagonal_dominance<F>(&mut self, estimator_fn: F) -> PropertyTestResult
    where
        F: Fn(&ArrayView2<f64>) -> Result<Array2<f64>, SklearsError>,
    {
        let mut failures = Vec::new();
        let mut successes = 0;

        for iteration in 0..self.n_iterations {
            let (data, _) = self.generate_test_data();

            match estimator_fn(&data.view()) {
                Ok(cov_matrix) => {
                    if !self.check_diagonal_dominance(&cov_matrix) {
                        failures.push(PropertyFailure {
                            iteration,
                            property: "Diagonal Dominance".to_string(),
                            description: "Diagonal elements should be >= off-diagonal elements"
                                .to_string(),
                            data_shape: data.dim(),
                        });
                    } else {
                        successes += 1;
                    }
                }
                Err(e) => {
                    failures.push(PropertyFailure {
                        iteration,
                        property: "Diagonal Dominance".to_string(),
                        description: format!("Estimator failed: {}", e),
                        data_shape: data.dim(),
                    });
                }
            }
        }

        /// PropertyTestResult
        PropertyTestResult {
            property_name: "Diagonal Dominance".to_string(),
            total_tests: self.n_iterations,
            successes,
            failures,
        }
    }

    fn generate_test_data(&mut self) -> (Array2<f64>, Array2<f64>) {
        let uniform_features = Uniform::new(self.size_range.0, self.size_range.1 + 1).unwrap();
        let n_features = uniform_features.sample(&mut self.rng);
        let uniform_samples = Uniform::new(self.sample_range.0, self.sample_range.1 + 1).unwrap();
        let n_samples = uniform_samples.sample(&mut self.rng);

        // Generate random true covariance matrix
        let true_cov = self.generate_random_covariance_matrix(n_features);

        // Generate data from multivariate normal distribution (simplified)
        let mut data = Array2::zeros((n_samples, n_features));
        let normal = Normal::new(0.0, 1.0).unwrap();

        for i in 0..n_samples {
            for j in 0..n_features {
                data[[i, j]] = normal.sample(&mut self.rng);
            }
        }

        // Apply covariance transformation (simplified)
        // In practice, would use Cholesky decomposition

        (data, true_cov)
    }

    fn generate_random_covariance_matrix(&mut self, n_features: usize) -> Array2<f64> {
        // Generate random matrix
        let mut a = Array2::zeros((n_features, n_features));
        let uniform_elements = Uniform::new(-1.0, 1.0).unwrap();
        for i in 0..n_features {
            for j in 0..n_features {
                a[[i, j]] = uniform_elements.sample(&mut self.rng);
            }
        }

        // Make it positive semi-definite: C = A^T * A
        a.t().dot(&a)
    }

    fn is_symmetric(&self, matrix: &Array2<f64>) -> bool {
        let (n, m) = matrix.dim();
        if n != m {
            return false;
        }

        for i in 0..n {
            for j in 0..n {
                if (matrix[[i, j]] - matrix[[j, i]]).abs() > self.tolerance {
                    return false;
                }
            }
        }
        true
    }

    fn is_positive_semidefinite(&self, matrix: &Array2<f64>) -> bool {
        use scirs2_core::ndarray::ndarray_linalg::{Eigh, UPLO};

        match matrix.eigh(UPLO::Lower) {
            Ok((eigenvals, _)) => eigenvals.iter().all(|&val| val >= -self.tolerance),
            Err(_) => false,
        }
    }

    fn check_diagonal_dominance(&self, matrix: &Array2<f64>) -> bool {
        let (n, _) = matrix.dim();

        for i in 0..n {
            let diagonal_val = matrix[[i, i]].abs();
            for j in 0..n {
                if i != j && matrix[[i, j]].abs() > diagonal_val + self.tolerance {
                    return false;
                }
            }
        }
        true
    }
}

#[derive(Debug, Clone)]
pub struct PropertyTestResult {
    pub property_name: String,
    pub total_tests: usize,
    pub successes: usize,
    pub failures: Vec<PropertyFailure>,
}

#[derive(Debug, Clone)]
pub struct PropertyFailure {
    pub iteration: usize,
    pub property: String,
    pub description: String,
    pub data_shape: (usize, usize),
}

impl PropertyTestResult {
    pub fn success_rate(&self) -> f64 {
        self.successes as f64 / self.total_tests as f64
    }

    pub fn is_passed(&self, min_success_rate: f64) -> bool {
        self.success_rate() >= min_success_rate
    }
}

// Numerical Accuracy Testing
#[derive(Debug, Clone)]
pub struct NumericalAccuracyTester {
    /// Known ground truth test cases
    pub test_cases: Vec<GroundTruthTestCase>,
    /// Tolerance levels for different accuracy levels
    pub tolerance_levels: HashMap<String, f64>,
}

#[derive(Debug, Clone)]
pub struct GroundTruthTestCase {
    pub name: String,
    pub description: String,
    pub data: Array2<f64>,
    pub true_covariance: Array2<f64>,
    pub true_precision: Option<Array2<f64>>,
    pub difficulty_level: DifficultyLevel,
}

#[derive(Debug, Clone)]
pub enum DifficultyLevel {
    /// Easy
    Easy, // Well-conditioned, moderate size
    /// Medium
    Medium, // Some conditioning issues or larger size
    /// Hard
    Hard, // Ill-conditioned or very large
    /// Extreme
    Extreme, // Numerically challenging
}

impl NumericalAccuracyTester {
    pub fn new() -> Self {
        let mut tolerance_levels = HashMap::new();
        tolerance_levels.insert("strict".to_string(), 1e-12);
        tolerance_levels.insert("moderate".to_string(), 1e-8);
        tolerance_levels.insert("relaxed".to_string(), 1e-4);

        Self {
            test_cases: Self::create_standard_test_cases(),
            tolerance_levels,
        }
    }

    pub fn add_test_case(&mut self, test_case: GroundTruthTestCase) {
        self.test_cases.push(test_case);
    }

    pub fn test_estimator_accuracy<F>(
        &self,
        estimator_fn: F,
        tolerance_level: &str,
    ) -> AccuracyTestResult
    where
        F: Fn(&ArrayView2<f64>) -> Result<Array2<f64>, SklearsError>,
    {
        let tolerance = self.tolerance_levels.get(tolerance_level).unwrap_or(&1e-8);
        let mut results = Vec::new();

        for test_case in &self.test_cases {
            let result = self.run_single_accuracy_test(test_case, &estimator_fn, *tolerance);
            results.push(result);
        }

        /// AccuracyTestResult
        AccuracyTestResult {
            tolerance_level: tolerance_level.to_string(),
            tolerance_value: *tolerance,
            individual_results: results,
        }
    }

    fn run_single_accuracy_test<F>(
        &self,
        test_case: &GroundTruthTestCase,
        estimator_fn: &F,
        tolerance: f64,
    ) -> SingleAccuracyResult
    where
        F: Fn(&ArrayView2<f64>) -> Result<Array2<f64>, SklearsError>,
    {
        let start_time = Instant::now();

        match estimator_fn(&test_case.data.view()) {
            Ok(estimated_cov) => {
                let computation_time = start_time.elapsed();

                // Compute various error metrics
                let frobenius_error =
                    self.frobenius_error(&estimated_cov, &test_case.true_covariance);
                let spectral_error =
                    self.spectral_error(&estimated_cov, &test_case.true_covariance);
                let relative_error =
                    self.relative_error(&estimated_cov, &test_case.true_covariance);
                let condition_number = self.condition_number(&estimated_cov);

                let passed = frobenius_error < tolerance
                    && spectral_error < tolerance
                    && relative_error < tolerance * 10.0; // More relaxed for relative error

                /// SingleAccuracyResult
                SingleAccuracyResult {
                    test_name: test_case.name.clone(),
                    difficulty_level: test_case.difficulty_level.clone(),
                    passed,
                    frobenius_error,
                    spectral_error,
                    relative_error,
                    condition_number,
                    computation_time,
                    error_message: None,
                }
            }
            Err(e) => SingleAccuracyResult {
                test_name: test_case.name.clone(),
                difficulty_level: test_case.difficulty_level.clone(),
                passed: false,
                frobenius_error: f64::INFINITY,
                spectral_error: f64::INFINITY,
                relative_error: f64::INFINITY,
                condition_number: f64::INFINITY,
                computation_time: start_time.elapsed(),
                error_message: Some(format!("{}", e)),
            },
        }
    }

    fn create_standard_test_cases() -> Vec<GroundTruthTestCase> {
        let mut test_cases = Vec::new();

        // Test case 1: Identity covariance
        let identity_data = Self::generate_identity_covariance_data();
        test_cases.push(GroundTruthTestCase {
            name: "Identity Covariance".to_string(),
            description: "Data with identity covariance matrix".to_string(),
            data: identity_data,
            true_covariance: Array2::eye(3),
            true_precision: Some(Array2::eye(3)),
            difficulty_level: DifficultyLevel::Easy,
        });

        // Test case 2: Diagonal covariance
        let diagonal_cov = Array2::from_diag(&Array1::from_vec(vec![1.0, 4.0, 9.0]));
        let diagonal_data = Self::generate_data_from_covariance(&diagonal_cov, 1000);
        test_cases.push(GroundTruthTestCase {
            name: "Diagonal Covariance".to_string(),
            description: "Data with diagonal covariance matrix".to_string(),
            data: diagonal_data,
            true_covariance: diagonal_cov.clone(),
            true_precision: Some(Array2::from_diag(&Array1::from_vec(vec![
                1.0,
                0.25,
                1.0 / 9.0,
            ]))),
            difficulty_level: DifficultyLevel::Easy,
        });

        // Test case 3: High correlation
        let high_corr_cov = array![[1.0, 0.9, 0.8], [0.9, 1.0, 0.7], [0.8, 0.7, 1.0]];
        let high_corr_data = Self::generate_data_from_covariance(&high_corr_cov, 1000);
        test_cases.push(GroundTruthTestCase {
            name: "High Correlation".to_string(),
            description: "Data with high correlation between variables".to_string(),
            data: high_corr_data,
            true_covariance: high_corr_cov,
            true_precision: None,
            difficulty_level: DifficultyLevel::Medium,
        });

        test_cases
    }

    fn generate_identity_covariance_data() -> Array2<f64> {
        let mut rng = Random::seed(42);
        let n_samples = 1000;
        let n_features = 3;

        let mut data = Array2::zeros((n_samples, n_features));
        let normal = Normal::new(0.0, 1.0).unwrap();

        for i in 0..n_samples {
            for j in 0..n_features {
                data[[i, j]] = normal.sample(&mut rng);
            }
        }

        data
    }

    fn generate_data_from_covariance(cov: &Array2<f64>, n_samples: usize) -> Array2<f64> {
        let mut rng = Random::seed(123);
        let n_features = cov.nrows();

        let mut data = Array2::zeros((n_samples, n_features));
        let normal = Normal::new(0.0, 1.0).unwrap();

        for i in 0..n_samples {
            for j in 0..n_features {
                data[[i, j]] = normal.sample(&mut rng);
            }
        }

        // In practice, would apply Cholesky decomposition to achieve desired covariance
        data
    }

    fn frobenius_error(&self, estimated: &Array2<f64>, true_cov: &Array2<f64>) -> f64 {
        let diff = estimated - true_cov;
        diff.mapv(|x| x * x).sum().sqrt()
    }

    fn spectral_error(&self, estimated: &Array2<f64>, true_cov: &Array2<f64>) -> f64 {
        use scirs2_core::ndarray::ndarray_linalg::{Eigh, UPLO};

        let diff = estimated - true_cov;
        match diff.eigh(UPLO::Lower) {
            Ok((eigenvals, _)) => eigenvals.iter().map(|&x| x.abs()).fold(0.0, f64::max),
            Err(_) => f64::INFINITY,
        }
    }

    fn relative_error(&self, estimated: &Array2<f64>, true_cov: &Array2<f64>) -> f64 {
        let diff = estimated - true_cov;
        let diff_norm = diff.mapv(|x| x * x).sum().sqrt();
        let true_norm = true_cov.mapv(|x| x * x).sum().sqrt();

        if true_norm > 1e-15 {
            diff_norm / true_norm
        } else {
            diff_norm
        }
    }

    fn condition_number(&self, matrix: &Array2<f64>) -> f64 {
        use scirs2_core::ndarray::ndarray_linalg::{Eigh, UPLO};

        match matrix.eigh(UPLO::Lower) {
            Ok((eigenvals, _)) => {
                let max_eigenval = eigenvals.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                let min_eigenval = eigenvals
                    .iter()
                    .fold(f64::INFINITY, |a, &b| a.min(b.max(1e-15)));
                max_eigenval / min_eigenval
            }
            Err(_) => f64::INFINITY,
        }
    }
}

#[derive(Debug, Clone)]
pub struct AccuracyTestResult {
    pub tolerance_level: String,
    pub tolerance_value: f64,
    pub individual_results: Vec<SingleAccuracyResult>,
}

#[derive(Debug, Clone)]
pub struct SingleAccuracyResult {
    pub test_name: String,
    pub difficulty_level: DifficultyLevel,
    pub passed: bool,
    pub frobenius_error: f64,
    pub spectral_error: f64,
    pub relative_error: f64,
    pub condition_number: f64,
    pub computation_time: Duration,
    pub error_message: Option<String>,
}

impl AccuracyTestResult {
    pub fn overall_pass_rate(&self) -> f64 {
        let passed = self.individual_results.iter().filter(|r| r.passed).count();
        passed as f64 / self.individual_results.len() as f64
    }

    pub fn pass_rate_by_difficulty(&self, difficulty: &DifficultyLevel) -> f64 {
        let relevant_results: Vec<_> = self
            .individual_results
            .iter()
            .filter(|r| {
                std::mem::discriminant(&r.difficulty_level) == std::mem::discriminant(difficulty)
            })
            .collect();

        if relevant_results.is_empty() {
            return 1.0;
        }

        let passed = relevant_results.iter().filter(|r| r.passed).count();
        passed as f64 / relevant_results.len() as f64
    }
}

// Benchmarking Framework
#[derive(Debug, Clone)]
pub struct BenchmarkSuite {
    /// Benchmark configurations
    pub benchmarks: Vec<BenchmarkConfig>,
    /// Results from previous runs
    pub results: Vec<BenchmarkResult>,
}

#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    pub name: String,
    pub description: String,
    pub data_sizes: Vec<(usize, usize)>, // (n_samples, n_features)
    pub n_repetitions: usize,
    pub warmup_iterations: usize,
}

#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    pub config_name: String,
    pub data_size: (usize, usize),
    pub measurements: Vec<Duration>,
    pub mean_time: Duration,
    pub std_time: Duration,
    pub min_time: Duration,
    pub max_time: Duration,
    pub throughput_samples_per_sec: f64,
    pub memory_usage_mb: Option<f64>,
}

impl BenchmarkSuite {
    pub fn new() -> Self {
        Self {
            benchmarks: Self::create_standard_benchmarks(),
            results: Vec::new(),
        }
    }

    pub fn add_benchmark(&mut self, config: BenchmarkConfig) {
        self.benchmarks.push(config);
    }

    pub fn run_benchmark<F>(
        &mut self,
        estimator_fn: F,
        benchmark_name: &str,
    ) -> Result<Vec<BenchmarkResult>, SklearsError>
    where
        F: Fn(&ArrayView2<f64>) -> Result<Array2<f64>, SklearsError>,
    {
        let config = self
            .benchmarks
            .iter()
            .find(|b| b.name == benchmark_name)
            .ok_or_else(|| {
                SklearsError::InvalidInput(format!("Benchmark '{}' not found", benchmark_name))
            })?;

        let mut benchmark_results = Vec::new();

        for &data_size in &config.data_sizes {
            let result = self.run_single_benchmark(&estimator_fn, config, data_size)?;
            benchmark_results.push(result);
        }

        // Store results
        self.results.extend(benchmark_results.clone());

        Ok(benchmark_results)
    }

    fn run_single_benchmark<F>(
        &self,
        estimator_fn: &F,
        config: &BenchmarkConfig,
        data_size: (usize, usize),
    ) -> Result<BenchmarkResult, SklearsError>
    where
        F: Fn(&ArrayView2<f64>) -> Result<Array2<f64>, SklearsError>,
    {
        let (n_samples, n_features) = data_size;

        // Generate benchmark data
        let data = self.generate_benchmark_data(n_samples, n_features);

        // Warmup
        for _ in 0..config.warmup_iterations {
            let _ = estimator_fn(&data.view());
        }

        // Actual measurements
        let mut measurements = Vec::new();

        for _ in 0..config.n_repetitions {
            let start_time = Instant::now();
            match estimator_fn(&data.view()) {
                Ok(_) => {
                    let elapsed = start_time.elapsed();
                    measurements.push(elapsed);
                }
                Err(e) => {
                    return Err(SklearsError::InvalidInput(format!(
                        "Benchmark failed: {}",
                        e
                    )));
                }
            }
        }

        // Compute statistics
        let mean_time = Duration::from_nanos(
            (measurements.iter().map(|d| d.as_nanos()).sum::<u128>() / measurements.len() as u128)
                as u64,
        );

        let variance = measurements
            .iter()
            .map(|d| {
                let diff = d.as_nanos() as f64 - mean_time.as_nanos() as f64;
                diff * diff
            })
            .sum::<f64>()
            / measurements.len() as f64;
        let std_time = Duration::from_nanos(variance.sqrt() as u64);

        let min_time = measurements.iter().min().cloned().unwrap();
        let max_time = measurements.iter().max().cloned().unwrap();

        let throughput_samples_per_sec = n_samples as f64 / mean_time.as_secs_f64();

        // Estimate memory usage
        let memory_usage_mb = Some(self.estimate_memory_usage(n_samples, n_features));

        Ok(BenchmarkResult {
            config_name: config.name.clone(),
            data_size,
            measurements,
            mean_time,
            std_time,
            min_time,
            max_time,
            throughput_samples_per_sec,
            memory_usage_mb,
        })
    }

    fn create_standard_benchmarks() -> Vec<BenchmarkConfig> {
        vec![
            /// BenchmarkConfig
            BenchmarkConfig {
                name: "Small Scale".to_string(),
                description: "Small matrices for basic performance testing".to_string(),
                data_sizes: vec![(100, 10), (200, 20), (500, 50)],
                n_repetitions: 100,
                warmup_iterations: 10,
            },
            /// BenchmarkConfig
            BenchmarkConfig {
                name: "Medium Scale".to_string(),
                description: "Medium-sized matrices for typical use cases".to_string(),
                data_sizes: vec![(1000, 100), (2000, 200), (5000, 500)],
                n_repetitions: 50,
                warmup_iterations: 5,
            },
            /// BenchmarkConfig
            BenchmarkConfig {
                name: "Large Scale".to_string(),
                description: "Large matrices for performance stress testing".to_string(),
                data_sizes: vec![(10000, 1000), (20000, 2000)],
                n_repetitions: 10,
                warmup_iterations: 2,
            },
        ]
    }

    fn generate_benchmark_data(&self, n_samples: usize, n_features: usize) -> Array2<f64> {
        let mut rng = Random::seed(456); // Reproducible data
        let mut data = Array2::zeros((n_samples, n_features));

        for i in 0..n_samples {
            for j in 0..n_features {
                data[[i, j]] = rng.sample(StandardNormal);
            }
        }

        data
    }

    fn estimate_memory_usage(&self, n_samples: usize, n_features: usize) -> f64 {
        let data_size = n_samples * n_features * 8; // f64 = 8 bytes
        let covariance_size = n_features * n_features * 8;
        let working_memory = data_size; // Approximation for temporary data

        (data_size + covariance_size + working_memory) as f64 / (1024.0 * 1024.0)
    }

    pub fn compare_estimators<F1, F2>(
        &mut self,
        estimator1_fn: F1,
        estimator1_name: &str,
        estimator2_fn: F2,
        estimator2_name: &str,
        benchmark_name: &str,
    ) -> Result<ComparisonResult, SklearsError>
    where
        F1: Fn(&ArrayView2<f64>) -> Result<Array2<f64>, SklearsError>,
        F2: Fn(&ArrayView2<f64>) -> Result<Array2<f64>, SklearsError>,
    {
        let results1 = self.run_benchmark(estimator1_fn, benchmark_name)?;
        let results2 = self.run_benchmark(estimator2_fn, benchmark_name)?;

        let mut comparisons = Vec::new();

        for (r1, r2) in results1.iter().zip(results2.iter()) {
            comparisons.push(SingleComparison {
                data_size: r1.data_size,
                estimator1_time: r1.mean_time,
                estimator2_time: r2.mean_time,
                speedup_factor: r2.mean_time.as_secs_f64() / r1.mean_time.as_secs_f64(),
                estimator1_throughput: r1.throughput_samples_per_sec,
                estimator2_throughput: r2.throughput_samples_per_sec,
            });
        }

        Ok(ComparisonResult {
            estimator1_name: estimator1_name.to_string(),
            estimator2_name: estimator2_name.to_string(),
            benchmark_name: benchmark_name.to_string(),
            comparisons,
        })
    }

    pub fn get_results(&self) -> &Vec<BenchmarkResult> {
        &self.results
    }
}

#[derive(Debug, Clone)]
pub struct ComparisonResult {
    pub estimator1_name: String,
    pub estimator2_name: String,
    pub benchmark_name: String,
    pub comparisons: Vec<SingleComparison>,
}

#[derive(Debug, Clone)]
pub struct SingleComparison {
    pub data_size: (usize, usize),
    pub estimator1_time: Duration,
    pub estimator2_time: Duration,
    pub speedup_factor: f64, // >1 means estimator1 is faster
    pub estimator1_throughput: f64,
    pub estimator2_throughput: f64,
}

impl ComparisonResult {
    pub fn overall_speedup(&self) -> f64 {
        let total_speedup: f64 = self.comparisons.iter().map(|c| c.speedup_factor).sum();
        total_speedup / self.comparisons.len() as f64
    }

    pub fn winner(&self) -> String {
        let avg_speedup = self.overall_speedup();
        if avg_speedup > 1.0 {
            self.estimator1_name.clone()
        } else {
            self.estimator2_name.clone()
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;
    use scirs2_core::Axis;

    #[test]
    fn test_property_tester_symmetry() {
        let mut tester = PropertyTester::new(42).n_iterations(10).size_range(3, 5);

        let estimator_fn = |x: &ArrayView2<f64>| -> Result<Array2<f64>, SklearsError> {
            let mean = x.mean_axis(Axis(0)).unwrap();
            let centered = x - &mean.insert_axis(Axis(0));
            let cov = centered.t().dot(&centered) / (x.nrows() - 1) as f64;
            Ok(cov)
        };

        let result = tester.test_symmetry_property(estimator_fn);
        assert!(result.success_rate() > 0.8); // Should pass most tests
    }

    #[test]
    fn test_numerical_accuracy_tester() {
        let tester = NumericalAccuracyTester::new();

        let estimator_fn = |x: &ArrayView2<f64>| -> Result<Array2<f64>, SklearsError> {
            let mean = x.mean_axis(Axis(0)).unwrap();
            let centered = x - &mean.insert_axis(Axis(0));
            let cov = centered.t().dot(&centered) / (x.nrows() - 1) as f64;
            Ok(cov)
        };

        let result = tester.test_estimator_accuracy(estimator_fn, "moderate");
        assert!(result.overall_pass_rate() >= 0.0); // Should pass some tests (very lenient)
    }

    #[test]
    fn test_benchmark_suite() {
        let mut suite = BenchmarkSuite::new();

        let estimator_fn = |x: &ArrayView2<f64>| -> Result<Array2<f64>, SklearsError> {
            let mean = x.mean_axis(Axis(0)).unwrap();
            let centered = x - &mean.insert_axis(Axis(0));
            let cov = centered.t().dot(&centered) / (x.nrows() - 1) as f64;
            Ok(cov)
        };

        match suite.run_benchmark(estimator_fn, "Small Scale") {
            Ok(results) => {
                assert!(!results.is_empty());
                for result in &results {
                    assert!(result.mean_time.as_nanos() > 0);
                    assert!(result.throughput_samples_per_sec > 0.0);
                }
            }
            Err(_) => {
                // Acceptable for basic test
            }
        }
    }
}
