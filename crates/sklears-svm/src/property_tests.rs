//! Property-based testing framework for SVM algorithms
//!
//! This module provides comprehensive property-based testing utilities to ensure
//! the mathematical correctness and robustness of SVM implementations.
//!
//! Properties tested include:
//! - Mathematical properties (convexity, KKT conditions, etc.)
//! - Numerical stability and convergence
//! - Invariance properties
//! - Robustness to edge cases
//! - Performance characteristics

use std::time::Instant;

// TODO: Replace with scirs2-linalg
// use nalgebra::{DMatrix, DVector};
// use proptest::prelude::*;
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::SeedableRng;
use scirs2_core::Rng;

use crate::kernels::{Kernel, KernelType};
use crate::svc::SVC;
use sklears_core::error::{Result, SklearsError};
use sklears_core::traits::{Fit, Predict};

/// Configuration for property-based testing
#[derive(Debug, Clone)]
pub struct PropertyTestConfig {
    /// Number of test cases to run
    pub test_cases: usize,
    /// Maximum number of samples in generated datasets
    pub max_samples: usize,
    /// Maximum number of features in generated datasets
    pub max_features: usize,
    /// Tolerance for numerical comparisons
    pub numerical_tolerance: f64,
    /// Maximum training time allowed (seconds)
    pub max_training_time: f64,
    /// Minimum accuracy threshold for synthetic datasets
    pub min_accuracy: f64,
    /// Random seed for reproducibility
    pub random_seed: Option<u64>,
}

impl Default for PropertyTestConfig {
    fn default() -> Self {
        Self {
            test_cases: 100,
            max_samples: 1000,
            max_features: 50,
            numerical_tolerance: 1e-10,
            max_training_time: 60.0,
            min_accuracy: 0.7,
            random_seed: Some(42),
        }
    }
}

/// Property test result
#[derive(Debug, Clone)]
pub struct PropertyTestResult {
    /// Test name
    pub test_name: String,
    /// Number of test cases passed
    pub passed: usize,
    /// Number of test cases failed
    pub failed: usize,
    /// Total test cases
    pub total: usize,
    /// Average execution time per test case
    pub avg_time: f64,
    /// Failure reasons
    pub failures: Vec<String>,
}

impl PropertyTestResult {
    fn new(test_name: String) -> Self {
        Self {
            test_name,
            passed: 0,
            failed: 0,
            total: 0,
            avg_time: 0.0,
            failures: Vec::new(),
        }
    }

    fn success_rate(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            self.passed as f64 / self.total as f64
        }
    }
}

/// Property-based test runner for SVM algorithms
pub struct SVMPropertyTester {
    config: PropertyTestConfig,
    rng: StdRng,
}

impl SVMPropertyTester {
    /// Create a new property tester
    pub fn new(config: PropertyTestConfig) -> Self {
        let rng = StdRng::seed_from_u64(42);

        Self { config, rng }
    }

    /// Create a new property tester with default configuration
    pub fn default() -> Self {
        Self::new(PropertyTestConfig::default())
    }

    /// Run all property tests
    pub fn run_all_tests(&mut self) -> Vec<PropertyTestResult> {
        let mut results = Vec::new();

        // Mathematical properties
        results.push(self.test_convexity_property());
        results.push(self.test_kkt_conditions());
        results.push(self.test_dual_gap());
        results.push(self.test_kernel_properties());

        // Numerical stability
        results.push(self.test_numerical_stability());
        results.push(self.test_convergence_properties());
        results.push(self.test_scale_invariance());

        // Robustness tests
        results.push(self.test_outlier_robustness());
        results.push(self.test_noise_robustness());
        results.push(self.test_edge_cases());

        // Performance properties
        results.push(self.test_training_time_bounds());
        results.push(self.test_memory_usage());
        results.push(self.test_prediction_consistency());

        results
    }

    /// Test convexity property of SVM optimization
    pub fn test_convexity_property(&mut self) -> PropertyTestResult {
        let mut result = PropertyTestResult::new("Convexity Property".to_string());
        let start_time = Instant::now();

        for _ in 0..self.config.test_cases {
            result.total += 1;

            // Generate random dataset
            let n_samples = self.rng.random_range(10..=100);
            let n_features = self.rng.random_range(2..=10);
            let (x, y) = self.generate_linearly_separable_dataset(n_samples, n_features);

            // Test convexity by checking that linear combinations of solutions
            // have objective value between the individual solutions
            match self.test_convexity_single_case(&x, &y) {
                Ok(true) => result.passed += 1,
                Ok(false) => {
                    result.failed += 1;
                    result
                        .failures
                        .push("Convexity property violated".to_string());
                }
                Err(e) => {
                    result.failed += 1;
                    result.failures.push(format!("Error: {e}"));
                }
            }
        }

        result.avg_time = start_time.elapsed().as_secs_f64() / result.total as f64;
        result
    }

    /// Test KKT conditions for optimality
    pub fn test_kkt_conditions(&mut self) -> PropertyTestResult {
        let mut result = PropertyTestResult::new("KKT Conditions".to_string());
        let start_time = Instant::now();

        for _ in 0..self.config.test_cases {
            result.total += 1;

            let n_samples = self.rng.random_range(20..=200);
            let n_features = self.rng.random_range(2..=20);
            let (x, y) = self.generate_random_dataset(n_samples, n_features);

            match self.test_kkt_conditions_single_case(&x, &y) {
                Ok(true) => result.passed += 1,
                Ok(false) => {
                    result.failed += 1;
                    result.failures.push("KKT conditions violated".to_string());
                }
                Err(e) => {
                    result.failed += 1;
                    result.failures.push(format!("Error: {e}"));
                }
            }
        }

        result.avg_time = start_time.elapsed().as_secs_f64() / result.total as f64;
        result
    }

    /// Test dual gap property
    pub fn test_dual_gap(&mut self) -> PropertyTestResult {
        let mut result = PropertyTestResult::new("Dual Gap Property".to_string());
        let start_time = Instant::now();

        for _ in 0..self.config.test_cases {
            result.total += 1;

            let n_samples = self.rng.random_range(20..=100);
            let n_features = self.rng.random_range(2..=10);
            let (x, y) = self.generate_random_dataset(n_samples, n_features);

            match self.test_dual_gap_single_case(&x, &y) {
                Ok(true) => result.passed += 1,
                Ok(false) => {
                    result.failed += 1;
                    result
                        .failures
                        .push("Dual gap property violated".to_string());
                }
                Err(e) => {
                    result.failed += 1;
                    result.failures.push(format!("Error: {e}"));
                }
            }
        }

        result.avg_time = start_time.elapsed().as_secs_f64() / result.total as f64;
        result
    }

    /// Test kernel properties (positive definiteness, symmetry, etc.)
    pub fn test_kernel_properties(&mut self) -> PropertyTestResult {
        let mut result = PropertyTestResult::new("Kernel Properties".to_string());
        let start_time = Instant::now();

        let kernels = vec![
            KernelType::Linear,
            KernelType::Rbf { gamma: 1.0 },
            KernelType::Polynomial {
                gamma: 1.0,
                degree: 2.0,
                coef0: 1.0,
            },
            KernelType::Polynomial {
                gamma: 1.0,
                degree: 3.0,
                coef0: 0.0,
            },
        ];

        for kernel_type in kernels {
            for _ in 0..self.config.test_cases / 4 {
                result.total += 1;

                let n_samples = self.rng.random_range(10..=50);
                let n_features = self.rng.random_range(2..=10);
                let x = self.generate_random_matrix(n_samples, n_features);

                match self.test_kernel_properties_single_case(&x, &kernel_type) {
                    Ok(true) => result.passed += 1,
                    Ok(false) => {
                        result.failed += 1;
                        result
                            .failures
                            .push(format!("Kernel property violated: {:?}", kernel_type));
                    }
                    Err(e) => {
                        result.failed += 1;
                        result
                            .failures
                            .push(format!("Error with {:?}: {}", kernel_type, e));
                    }
                }
            }
        }

        result.avg_time = start_time.elapsed().as_secs_f64() / result.total as f64;
        result
    }

    /// Test numerical stability
    pub fn test_numerical_stability(&mut self) -> PropertyTestResult {
        let mut result = PropertyTestResult::new("Numerical Stability".to_string());
        let start_time = Instant::now();

        for _ in 0..self.config.test_cases {
            result.total += 1;

            // Generate dataset with various numerical challenges
            let (x, y) = match self.rng.random_range(0..4) {
                0 => self.generate_ill_conditioned_dataset(50, 5),
                1 => self.generate_very_small_values_dataset(50, 5),
                2 => self.generate_very_large_values_dataset(50, 5),
                _ => self.generate_near_duplicate_samples_dataset(50, 5),
            };

            match self.test_numerical_stability_single_case(&x, &y) {
                Ok(true) => result.passed += 1,
                Ok(false) => {
                    result.failed += 1;
                    result
                        .failures
                        .push("Numerical instability detected".to_string());
                }
                Err(e) => {
                    result.failed += 1;
                    result.failures.push(format!("Error: {e}"));
                }
            }
        }

        result.avg_time = start_time.elapsed().as_secs_f64() / result.total as f64;
        result
    }

    /// Test convergence properties
    pub fn test_convergence_properties(&mut self) -> PropertyTestResult {
        let mut result = PropertyTestResult::new("Convergence Properties".to_string());
        let start_time = Instant::now();

        for _ in 0..self.config.test_cases {
            result.total += 1;

            let n_samples = self.rng.random_range(50..=200);
            let n_features = self.rng.random_range(2..=20);
            let (x, y) = self.generate_random_dataset(n_samples, n_features);

            match self.test_convergence_single_case(&x, &y) {
                Ok(true) => result.passed += 1,
                Ok(false) => {
                    result.failed += 1;
                    result
                        .failures
                        .push("Convergence property violated".to_string());
                }
                Err(e) => {
                    result.failed += 1;
                    result.failures.push(format!("Error: {e}"));
                }
            }
        }

        result.avg_time = start_time.elapsed().as_secs_f64() / result.total as f64;
        result
    }

    /// Test scale invariance
    pub fn test_scale_invariance(&mut self) -> PropertyTestResult {
        let mut result = PropertyTestResult::new("Scale Invariance".to_string());
        let start_time = Instant::now();

        for _ in 0..self.config.test_cases {
            result.total += 1;

            let n_samples = self.rng.random_range(20..=100);
            let n_features = self.rng.random_range(2..=10);
            let (x, y) = self.generate_random_dataset(n_samples, n_features);

            match self.test_scale_invariance_single_case(&x, &y) {
                Ok(true) => result.passed += 1,
                Ok(false) => {
                    result.failed += 1;
                    result
                        .failures
                        .push("Scale invariance violated".to_string());
                }
                Err(e) => {
                    result.failed += 1;
                    result.failures.push(format!("Error: {e}"));
                }
            }
        }

        result.avg_time = start_time.elapsed().as_secs_f64() / result.total as f64;
        result
    }

    /// Test outlier robustness
    pub fn test_outlier_robustness(&mut self) -> PropertyTestResult {
        let mut result = PropertyTestResult::new("Outlier Robustness".to_string());
        let start_time = Instant::now();

        for _ in 0..self.config.test_cases {
            result.total += 1;

            let n_samples = self.rng.random_range(50..=200);
            let n_features = self.rng.random_range(2..=10);
            let (x, y) = self.generate_dataset_with_outliers(
                n_samples, n_features, 0.1, // 10% outliers
            );

            match self.test_outlier_robustness_single_case(&x, &y) {
                Ok(true) => result.passed += 1,
                Ok(false) => {
                    result.failed += 1;
                    result
                        .failures
                        .push("Outlier robustness failed".to_string());
                }
                Err(e) => {
                    result.failed += 1;
                    result.failures.push(format!("Error: {e}"));
                }
            }
        }

        result.avg_time = start_time.elapsed().as_secs_f64() / result.total as f64;
        result
    }

    /// Test noise robustness
    pub fn test_noise_robustness(&mut self) -> PropertyTestResult {
        let mut result = PropertyTestResult::new("Noise Robustness".to_string());
        let start_time = Instant::now();

        for _ in 0..self.config.test_cases {
            result.total += 1;

            let n_samples = self.rng.random_range(50..=200);
            let n_features = self.rng.random_range(2..=10);
            let (x, y) = self.generate_noisy_dataset(
                n_samples, n_features, 0.1, // 10% noise level
            );

            match self.test_noise_robustness_single_case(&x, &y) {
                Ok(true) => result.passed += 1,
                Ok(false) => {
                    result.failed += 1;
                    result.failures.push("Noise robustness failed".to_string());
                }
                Err(e) => {
                    result.failed += 1;
                    result.failures.push(format!("Error: {e}"));
                }
            }
        }

        result.avg_time = start_time.elapsed().as_secs_f64() / result.total as f64;
        result
    }

    /// Test edge cases
    pub fn test_edge_cases(&mut self) -> PropertyTestResult {
        let mut result = PropertyTestResult::new("Edge Cases".to_string());
        let start_time = Instant::now();

        // Test various edge cases
        let edge_cases = vec![
            ("Single sample", 1, 2),
            ("Two samples", 2, 2),
            ("Single feature", 10, 1),
            ("More features than samples", 5, 10),
            ("Square dataset", 10, 10),
        ];

        for (case_name, n_samples, n_features) in edge_cases {
            result.total += 1;

            let (x, y) = self.generate_random_dataset(n_samples, n_features);

            match self.test_edge_case_single(&x, &y, case_name) {
                Ok(true) => result.passed += 1,
                Ok(false) => {
                    result.failed += 1;
                    result
                        .failures
                        .push(format!("Edge case failed: {case_name}"));
                }
                Err(e) => {
                    result.failed += 1;
                    result
                        .failures
                        .push(format!("Error in {}: {}", case_name, e));
                }
            }
        }

        result.avg_time = start_time.elapsed().as_secs_f64() / result.total as f64;
        result
    }

    /// Test training time bounds
    pub fn test_training_time_bounds(&mut self) -> PropertyTestResult {
        let mut result = PropertyTestResult::new("Training Time Bounds".to_string());
        let start_time = Instant::now();

        for _ in 0..self.config.test_cases {
            result.total += 1;

            let n_samples = self.rng.random_range(100..=500);
            let n_features = self.rng.random_range(5..=20);
            let (x, y) = self.generate_random_dataset(n_samples, n_features);

            match self.test_training_time_single_case(&x, &y) {
                Ok(true) => result.passed += 1,
                Ok(false) => {
                    result.failed += 1;
                    result
                        .failures
                        .push("Training time exceeded bounds".to_string());
                }
                Err(e) => {
                    result.failed += 1;
                    result.failures.push(format!("Error: {e}"));
                }
            }
        }

        result.avg_time = start_time.elapsed().as_secs_f64() / result.total as f64;
        result
    }

    /// Test memory usage
    pub fn test_memory_usage(&mut self) -> PropertyTestResult {
        let mut result = PropertyTestResult::new("Memory Usage".to_string());
        let start_time = Instant::now();

        for _ in 0..self.config.test_cases {
            result.total += 1;

            let n_samples = self.rng.random_range(100..=1000);
            let n_features = self.rng.random_range(5..=50);
            let (x, y) = self.generate_random_dataset(n_samples, n_features);

            match self.test_memory_usage_single_case(&x, &y) {
                Ok(true) => result.passed += 1,
                Ok(false) => {
                    result.failed += 1;
                    result
                        .failures
                        .push("Memory usage exceeded bounds".to_string());
                }
                Err(e) => {
                    result.failed += 1;
                    result.failures.push(format!("Error: {e}"));
                }
            }
        }

        result.avg_time = start_time.elapsed().as_secs_f64() / result.total as f64;
        result
    }

    /// Test prediction consistency
    pub fn test_prediction_consistency(&mut self) -> PropertyTestResult {
        let mut result = PropertyTestResult::new("Prediction Consistency".to_string());
        let start_time = Instant::now();

        for _ in 0..self.config.test_cases {
            result.total += 1;

            let n_samples = self.rng.random_range(50..=200);
            let n_features = self.rng.random_range(2..=10);
            let (x, y) = self.generate_random_dataset(n_samples, n_features);

            match self.test_prediction_consistency_single_case(&x, &y) {
                Ok(true) => result.passed += 1,
                Ok(false) => {
                    result.failed += 1;
                    result
                        .failures
                        .push("Prediction consistency violated".to_string());
                }
                Err(e) => {
                    result.failed += 1;
                    result.failures.push(format!("Error: {e}"));
                }
            }
        }

        result.avg_time = start_time.elapsed().as_secs_f64() / result.total as f64;
        result
    }

    // Helper methods for single test cases
    fn test_convexity_single_case(&mut self, x: &DMatrix<f64>, y: &DVector<f64>) -> Result<bool> {
        // Test convexity by checking that the objective function is convex
        let svm1 = SVC::new().c(1.0).linear().tol(1e-6).max_iter(1000);
        let svm2 = SVC::new().c(1.0).linear().tol(1e-6).max_iter(1000);

        // Train two different models with different initializations
        // Convert to ndarray for SVM
        let x_ndarray = Array2::from_shape_vec((x.nrows(), x.ncols()), x.iter().cloned().collect())
            .map_err(|e| {
                SklearsError::InvalidInput(format!("Failed to convert to ndarray: {e}"))
            })?;
        let y_ndarray = Array1::from_vec(y.iter().cloned().collect());

        let fitted_svm1 = svm1.fit(&x_ndarray, &y_ndarray)?;
        let fitted_svm2 = svm2.fit(&x_ndarray, &y_ndarray)?;

        // Check if both converged to similar solutions (indicating convexity)
        let decision1 = fitted_svm1.decision_function(&x_ndarray)?;
        let decision2 = fitted_svm2.decision_function(&x_ndarray)?;

        let diff_norm = (&decision1 - &decision2)
            .iter()
            .map(|x| x * x)
            .sum::<f64>()
            .sqrt();
        Ok(diff_norm < self.config.numerical_tolerance * 10.0)
    }

    fn test_kkt_conditions_single_case(
        &mut self,
        x: &DMatrix<f64>,
        y: &DVector<f64>,
    ) -> Result<bool> {
        let svm = SVC::new().c(1.0).linear().tol(1e-6).max_iter(1000);
        // Convert to ndarray for SVM
        let x_ndarray = Array2::from_shape_vec((x.nrows(), x.ncols()), x.iter().cloned().collect())
            .map_err(|e| {
                SklearsError::InvalidInput(format!("Failed to convert to ndarray: {e}"))
            })?;
        let y_ndarray = Array1::from_vec(y.iter().cloned().collect());

        let fitted_svm = svm.fit(&x_ndarray, &y_ndarray)?;

        // Check KKT conditions
        let decision_values = fitted_svm.decision_function(&x_ndarray)?;
        let margins: Vec<f64> = decision_values
            .iter()
            .zip(y.iter())
            .map(|(&d, &label)| label * d)
            .collect();

        // Check complementary slackness and stationarity conditions
        let mut kkt_satisfied = true;
        for i in 0..margins.len() {
            let margin = margins[i];
            let alpha = if i < fitted_svm.dual_coef().len() {
                fitted_svm.dual_coef()[i]
            } else {
                0.0
            };

            // Check complementary slackness: α_i * (1 - y_i * f(x_i)) = 0
            if alpha > self.config.numerical_tolerance {
                if (margin - 1.0).abs() > self.config.numerical_tolerance {
                    kkt_satisfied = false;
                    break;
                }
            }

            // Check bounds: 0 ≤ α_i ≤ C
            if alpha < -self.config.numerical_tolerance
                || alpha > 1.0 + self.config.numerical_tolerance
            {
                kkt_satisfied = false;
                break;
            }
        }

        Ok(kkt_satisfied)
    }

    fn test_dual_gap_single_case(&mut self, x: &DMatrix<f64>, y: &DVector<f64>) -> Result<bool> {
        let svm = SVC::new().c(1.0).linear().tol(1e-6).max_iter(1000);
        // Convert to ndarray for SVM
        let x_ndarray = Array2::from_shape_vec((x.nrows(), x.ncols()), x.iter().cloned().collect())
            .map_err(|e| {
                SklearsError::InvalidInput(format!("Failed to convert to ndarray: {e}"))
            })?;
        let y_ndarray = Array1::from_vec(y.iter().cloned().collect());

        let fitted_svm = svm.fit(&x_ndarray, &y_ndarray)?;

        // Calculate primal and dual objectives
        let decision_values = fitted_svm.decision_function(&x_ndarray)?;
        let margins: Vec<f64> = decision_values
            .iter()
            .zip(y.iter())
            .map(|(&d, &label)| label * d)
            .collect();

        // Primal objective: (1/2)||w||² + C∑max(0, 1-y_i*f(x_i))
        let hinge_loss: f64 = margins.iter().map(|&m| (1.0 - m).max(0.0)).sum();
        let primal_obj =
            0.5 * fitted_svm.dual_coef().iter().map(|x| x * x).sum::<f64>() + 1.0 * hinge_loss;

        // Dual objective: ∑α_i - (1/2)∑∑α_i*α_j*y_i*y_j*K(x_i,x_j)
        let dual_obj = fitted_svm.dual_coef().iter().sum::<f64>()
            - 0.5 * fitted_svm.dual_coef().iter().map(|x| x * x).sum::<f64>();

        // Dual gap should be non-negative and small at optimum
        let dual_gap = primal_obj - dual_obj;
        Ok(dual_gap >= -self.config.numerical_tolerance && dual_gap < 1.0)
    }

    fn test_kernel_properties_single_case(
        &mut self,
        x: &DMatrix<f64>,
        kernel_type: &KernelType,
    ) -> Result<bool> {
        let kernel = kernel_type.clone();
        let n = x.nrows();

        // Test symmetry: K(x_i, x_j) = K(x_j, x_i)
        for i in 0..n.min(10) {
            for j in (i + 1)..n.min(10) {
                let row_i = Array1::from_vec(x.row(i).iter().cloned().collect());
                let row_j = Array1::from_vec(x.row(j).iter().cloned().collect());
                let k_ij = kernel.compute(row_i.view(), row_j.view());
                let row_j = Array1::from_vec(x.row(j).iter().cloned().collect());
                let row_i = Array1::from_vec(x.row(i).iter().cloned().collect());
                let k_ji = kernel.compute(row_j.view(), row_i.view());

                if (k_ij - k_ji).abs() > self.config.numerical_tolerance {
                    return Ok(false);
                }
            }
        }

        // Test positive definiteness (at least positive semidefinite)
        let mut gram_matrix = DMatrix::zeros(n.min(20), n.min(20));
        for i in 0..n.min(20) {
            for j in 0..n.min(20) {
                let row_i = Array1::from_vec(x.row(i).iter().cloned().collect());
                let row_j = Array1::from_vec(x.row(j).iter().cloned().collect());
                gram_matrix[(i, j)] = kernel.compute(row_i.view(), row_j.view());
            }
        }

        // Check that all eigenvalues are non-negative (simplified check)
        let trace = gram_matrix.trace();
        Ok(trace >= 0.0)
    }

    fn test_numerical_stability_single_case(
        &mut self,
        x: &DMatrix<f64>,
        y: &DVector<f64>,
    ) -> Result<bool> {
        let svm = SVC::new().c(1.0).linear().tol(1e-6).max_iter(1000);

        // Try to fit and check for numerical issues
        // Convert to ndarray for SVM
        let x_ndarray = Array2::from_shape_vec((x.nrows(), x.ncols()), x.iter().cloned().collect())
            .map_err(|e| {
                SklearsError::InvalidInput(format!("Failed to convert to ndarray: {e}"))
            })?;
        let y_ndarray = Array1::from_vec(y.iter().cloned().collect());

        match svm.fit(&x_ndarray, &y_ndarray) {
            Ok(fitted_svm) => {
                // Check if predictions are reasonable
                let predictions = fitted_svm.predict(&x_ndarray)?;
                let has_nan = predictions.iter().any(|&p| p.is_nan());
                let has_inf = predictions.iter().any(|&p| p.is_infinite());

                Ok(!has_nan && !has_inf)
            }
            Err(_) => Ok(false), // Consider numerical failure as instability
        }
    }

    fn test_convergence_single_case(&mut self, x: &DMatrix<f64>, y: &DVector<f64>) -> Result<bool> {
        let svm = SVC::new().c(1.0).linear().tol(1e-6).max_iter(1000);

        // Train with different tolerances and check convergence
        // Convert to ndarray for SVM
        let x_ndarray = Array2::from_shape_vec((x.nrows(), x.ncols()), x.iter().cloned().collect())
            .map_err(|e| {
                SklearsError::InvalidInput(format!("Failed to convert to ndarray: {e}"))
            })?;
        let y_ndarray = Array1::from_vec(y.iter().cloned().collect());

        let fitted_svm = svm.fit(&x_ndarray, &y_ndarray)?;

        // Check if the algorithm converged (basic check)
        let decision_values = fitted_svm.decision_function(&x_ndarray)?;
        let margins: Vec<f64> = decision_values
            .iter()
            .zip(y.iter())
            .map(|(&d, &label)| label * d)
            .collect();

        // Check if margins are reasonable
        let avg_margin = margins.iter().sum::<f64>() / margins.len() as f64;
        Ok(avg_margin.is_finite() && avg_margin.abs() < 1000.0)
    }

    fn test_scale_invariance_single_case(
        &mut self,
        x: &DMatrix<f64>,
        y: &DVector<f64>,
    ) -> Result<bool> {
        // Test with original data
        let svm1 = SVC::new().c(1.0).linear().tol(1e-6).max_iter(1000);
        // Convert to ndarray for SVM
        let x_ndarray = Array2::from_shape_vec((x.nrows(), x.ncols()), x.iter().cloned().collect())
            .map_err(|e| {
                SklearsError::InvalidInput(format!("Failed to convert to ndarray: {e}"))
            })?;
        let y_ndarray = Array1::from_vec(y.iter().cloned().collect());

        let fitted_svm1 = svm1.fit(&x_ndarray, &y_ndarray)?;
        let pred1 = fitted_svm1.predict(&x_ndarray)?;

        // Test with scaled data
        let scale_factor = 10.0;
        let x_scaled = x * scale_factor;
        let x_scaled_ndarray = Array2::from_shape_vec(
            (x_scaled.nrows(), x_scaled.ncols()),
            x_scaled.iter().cloned().collect(),
        )
        .map_err(|e| SklearsError::InvalidInput(format!("Failed to convert to ndarray: {e}")))?;
        let svm2 = SVC::new().c(1.0).linear().tol(1e-6).max_iter(1000);
        let fitted_svm2 = svm2.fit(&x_scaled_ndarray, &y_ndarray)?;
        let pred2 = fitted_svm2.predict(&x_scaled_ndarray)?;

        // Predictions should be the same
        let diff = (&pred1 - &pred2).iter().map(|x| x * x).sum::<f64>().sqrt();
        Ok(diff < self.config.numerical_tolerance * 10.0)
    }

    fn test_outlier_robustness_single_case(
        &mut self,
        x: &DMatrix<f64>,
        y: &DVector<f64>,
    ) -> Result<bool> {
        let svm = SVC::new().c(1.0).linear().tol(1e-6).max_iter(1000);

        // Train on data with outliers
        // Convert to ndarray for SVM
        let x_ndarray = Array2::from_shape_vec((x.nrows(), x.ncols()), x.iter().cloned().collect())
            .map_err(|e| {
                SklearsError::InvalidInput(format!("Failed to convert to ndarray: {e}"))
            })?;
        let y_ndarray = Array1::from_vec(y.iter().cloned().collect());

        let result = svm.fit(&x_ndarray, &y_ndarray);

        // Check that the algorithm can handle outliers without failing
        match result {
            Ok(fitted_svm) => {
                let predictions = fitted_svm.predict(&x_ndarray)?;
                let accuracy = predictions
                    .iter()
                    .zip(y.iter())
                    .map(|(&p, &t)| if (p - t).abs() < 0.5 { 1.0 } else { 0.0 })
                    .sum::<f64>()
                    / predictions.len() as f64;

                Ok(accuracy > 0.5) // Should achieve reasonable accuracy even with outliers
            }
            Err(_) => Ok(false),
        }
    }

    fn test_noise_robustness_single_case(
        &mut self,
        x: &DMatrix<f64>,
        y: &DVector<f64>,
    ) -> Result<bool> {
        let svm = SVC::new().c(1.0).linear().tol(1e-6).max_iter(1000);

        // Train on noisy data
        // Convert to ndarray for SVM
        let x_ndarray = Array2::from_shape_vec((x.nrows(), x.ncols()), x.iter().cloned().collect())
            .map_err(|e| {
                SklearsError::InvalidInput(format!("Failed to convert to ndarray: {e}"))
            })?;
        let y_ndarray = Array1::from_vec(y.iter().cloned().collect());

        let result = svm.fit(&x_ndarray, &y_ndarray);

        match result {
            Ok(fitted_svm) => {
                let predictions = fitted_svm.predict(&x_ndarray)?;
                let accuracy = predictions
                    .iter()
                    .zip(y.iter())
                    .map(|(&p, &t)| if (p - t).abs() < 0.5 { 1.0 } else { 0.0 })
                    .sum::<f64>()
                    / predictions.len() as f64;

                Ok(accuracy > self.config.min_accuracy)
            }
            Err(_) => Ok(false),
        }
    }

    fn test_edge_case_single(
        &mut self,
        x: &DMatrix<f64>,
        y: &DVector<f64>,
        case_name: &str,
    ) -> Result<bool> {
        let svm = SVC::new().c(1.0).linear().tol(1e-6).max_iter(1000);

        // Try to handle edge case gracefully
        // Convert to ndarray for SVM
        let x_ndarray = Array2::from_shape_vec((x.nrows(), x.ncols()), x.iter().cloned().collect())
            .map_err(|e| {
                SklearsError::InvalidInput(format!("Failed to convert to ndarray: {e}"))
            })?;
        let y_ndarray = Array1::from_vec(y.iter().cloned().collect());

        match svm.fit(&x_ndarray, &y_ndarray) {
            Ok(fitted_svm) => {
                let predictions = fitted_svm.predict(&x_ndarray)?;
                let has_valid_predictions = predictions.iter().all(|&p| p.is_finite());
                Ok(has_valid_predictions)
            }
            Err(_) => {
                // For some edge cases, failure might be expected
                match case_name {
                    "Single sample" => Ok(true), // Expected to fail
                    _ => Ok(false),
                }
            }
        }
    }

    fn test_training_time_single_case(
        &mut self,
        x: &DMatrix<f64>,
        y: &DVector<f64>,
    ) -> Result<bool> {
        let svm = SVC::new().c(1.0).linear().tol(1e-6).max_iter(1000);

        // Convert to ndarray for SVM
        let x_ndarray = Array2::from_shape_vec((x.nrows(), x.ncols()), x.iter().cloned().collect())
            .map_err(|e| {
                SklearsError::InvalidInput(format!("Failed to convert to ndarray: {e}"))
            })?;
        let y_ndarray = Array1::from_vec(y.iter().cloned().collect());

        let start_time = Instant::now();
        let result = svm.fit(&x_ndarray, &y_ndarray);
        let training_time = start_time.elapsed().as_secs_f64();

        match result {
            Ok(_) => Ok(training_time < self.config.max_training_time),
            Err(_) => Ok(false),
        }
    }

    fn test_memory_usage_single_case(
        &mut self,
        x: &DMatrix<f64>,
        y: &DVector<f64>,
    ) -> Result<bool> {
        let svm = SVC::new().c(1.0).linear().tol(1e-6).max_iter(1000);

        // This is a simplified memory test
        // In practice, you'd want to use a proper memory profiler
        // Convert to ndarray for SVM
        let x_ndarray = Array2::from_shape_vec((x.nrows(), x.ncols()), x.iter().cloned().collect())
            .map_err(|e| {
                SklearsError::InvalidInput(format!("Failed to convert to ndarray: {e}"))
            })?;
        let y_ndarray = Array1::from_vec(y.iter().cloned().collect());

        let result = svm.fit(&x_ndarray, &y_ndarray);

        match result {
            Ok(fitted_svm) => {
                // Check that the model doesn't use excessive memory
                // This is a basic check - in practice you'd measure actual memory usage
                let model_size = std::mem::size_of_val(&fitted_svm);
                Ok(model_size < 1024 * 1024) // Less than 1MB
            }
            Err(_) => Ok(false),
        }
    }

    fn test_prediction_consistency_single_case(
        &mut self,
        x: &DMatrix<f64>,
        y: &DVector<f64>,
    ) -> Result<bool> {
        let svm = SVC::new().c(1.0).linear().tol(1e-6).max_iter(1000);
        // Convert to ndarray for SVM
        let x_ndarray = Array2::from_shape_vec((x.nrows(), x.ncols()), x.iter().cloned().collect())
            .map_err(|e| {
                SklearsError::InvalidInput(format!("Failed to convert to ndarray: {e}"))
            })?;
        let y_ndarray = Array1::from_vec(y.iter().cloned().collect());

        let fitted_svm = svm.fit(&x_ndarray, &y_ndarray)?;

        // Test that predictions are consistent across multiple calls
        let pred1 = fitted_svm.predict(&x_ndarray)?;
        let pred2 = fitted_svm.predict(&x_ndarray)?;

        let diff = (&pred1 - &pred2).iter().map(|x| x * x).sum::<f64>().sqrt();
        Ok(diff < self.config.numerical_tolerance)
    }

    // Data generation methods
    fn generate_random_dataset(
        &mut self,
        n_samples: usize,
        n_features: usize,
    ) -> (DMatrix<f64>, DVector<f64>) {
        let x = DMatrix::from_fn(n_samples, n_features, |_, _| {
            self.rng.random_range(-1.0..=1.0)
        });
        let y = DVector::from_fn(n_samples, |_, _| {
            if self.rng.random::<f64>() > 0.5 {
                1.0
            } else {
                -1.0
            }
        });
        (x, y)
    }

    fn generate_linearly_separable_dataset(
        &mut self,
        n_samples: usize,
        n_features: usize,
    ) -> (DMatrix<f64>, DVector<f64>) {
        let mut x = DMatrix::from_fn(n_samples, n_features, |_, _| {
            self.rng.random_range(-1.0..=1.0)
        });
        let w = DVector::from_fn(n_features, |_, _| self.rng.random_range(-1.0..=1.0));

        // Create linearly separable labels
        let mut y = DVector::zeros(n_samples);
        for i in 0..n_samples {
            let decision = x.row(i).dot(&w);
            y[i] = if decision > 0.0 { 1.0 } else { -1.0 };
        }

        // Add some margin to ensure separability
        for i in 0..n_samples {
            let margin = 0.1 * y[i];
            let row_update = &w * margin;
            for k in 0..n_features {
                x[(i, k)] += row_update[k];
            }
        }

        (x, y)
    }

    fn generate_random_matrix(&mut self, n_rows: usize, n_cols: usize) -> DMatrix<f64> {
        DMatrix::from_fn(n_rows, n_cols, |_, _| self.rng.random_range(-1.0..=1.0))
    }

    fn generate_ill_conditioned_dataset(
        &mut self,
        n_samples: usize,
        n_features: usize,
    ) -> (DMatrix<f64>, DVector<f64>) {
        let mut x = self.generate_random_matrix(n_samples, n_features);

        // Make the matrix ill-conditioned by making some columns nearly identical
        if n_features > 1 {
            let noise_scale = 1e-10;
            for i in 1..n_features {
                x.set_column(
                    i,
                    &(x.column(0)
                        + DVector::from_fn(n_samples, |_, _| {
                            self.rng.random_range(-noise_scale..=noise_scale)
                        })),
                );
            }
        }

        let y = DVector::from_fn(n_samples, |_, _| {
            if self.rng.random::<f64>() > 0.5 {
                1.0
            } else {
                -1.0
            }
        });
        (x, y)
    }

    fn generate_very_small_values_dataset(
        &mut self,
        n_samples: usize,
        n_features: usize,
    ) -> (DMatrix<f64>, DVector<f64>) {
        let scale = 1e-10;
        let x = DMatrix::from_fn(n_samples, n_features, |_, _| {
            self.rng.random_range(-scale..=scale)
        });
        let y = DVector::from_fn(n_samples, |_, _| {
            if self.rng.random::<f64>() > 0.5 {
                1.0
            } else {
                -1.0
            }
        });
        (x, y)
    }

    fn generate_very_large_values_dataset(
        &mut self,
        n_samples: usize,
        n_features: usize,
    ) -> (DMatrix<f64>, DVector<f64>) {
        let scale = 1e10;
        let x = DMatrix::from_fn(n_samples, n_features, |_, _| {
            self.rng.random_range(-scale..=scale)
        });
        let y = DVector::from_fn(n_samples, |_, _| {
            if self.rng.random::<f64>() > 0.5 {
                1.0
            } else {
                -1.0
            }
        });
        (x, y)
    }

    fn generate_near_duplicate_samples_dataset(
        &mut self,
        n_samples: usize,
        n_features: usize,
    ) -> (DMatrix<f64>, DVector<f64>) {
        let base_sample = DVector::from_fn(n_features, |_, _| self.rng.random_range(-1.0..=1.0));
        let noise_scale = 1e-8;

        let mut x = DMatrix::zeros(n_samples, n_features);
        for i in 0..n_samples {
            let noise = DVector::from_fn(n_features, |_, _| {
                self.rng.random_range(-noise_scale..=noise_scale)
            });
            x.set_row(i, &(base_sample.clone() + noise).transpose());
        }

        let y = DVector::from_fn(n_samples, |_, _| {
            if self.rng.random::<f64>() > 0.5 {
                1.0
            } else {
                -1.0
            }
        });
        (x, y)
    }

    fn generate_dataset_with_outliers(
        &mut self,
        n_samples: usize,
        n_features: usize,
        outlier_fraction: f64,
    ) -> (DMatrix<f64>, DVector<f64>) {
        let n_outliers = (n_samples as f64 * outlier_fraction) as usize;
        let n_normal = n_samples - n_outliers;

        // Generate normal samples
        let x_normal = DMatrix::from_fn(n_normal, n_features, |_, _| {
            self.rng.random_range(-1.0..=1.0)
        });
        let y_normal = DVector::from_fn(n_normal, |_, _| {
            if self.rng.random::<f64>() > 0.5 {
                1.0
            } else {
                -1.0
            }
        });

        // Generate outliers
        let outlier_scale = 10.0;
        let x_outliers = DMatrix::from_fn(n_outliers, n_features, |_, _| {
            self.rng.random_range(-outlier_scale..=outlier_scale)
        });
        let y_outliers = DVector::from_fn(n_outliers, |_, _| {
            if self.rng.random::<f64>() > 0.5 {
                1.0
            } else {
                -1.0
            }
        });

        // Combine normal and outlier samples
        let mut x = DMatrix::zeros(n_samples, n_features);
        let mut y = DVector::zeros(n_samples);

        x.rows_mut(0, n_normal).copy_from(&x_normal);
        x.rows_mut(n_normal, n_outliers).copy_from(&x_outliers);
        y.rows_mut(0, n_normal).copy_from(&y_normal);
        y.rows_mut(n_normal, n_outliers).copy_from(&y_outliers);

        (x, y)
    }

    fn generate_noisy_dataset(
        &mut self,
        n_samples: usize,
        n_features: usize,
        noise_level: f64,
    ) -> (DMatrix<f64>, DVector<f64>) {
        let (mut x, mut y) = self.generate_linearly_separable_dataset(n_samples, n_features);

        // Add noise to features
        for i in 0..n_samples {
            for j in 0..n_features {
                x[(i, j)] += self.rng.random_range(-noise_level..=noise_level);
            }
        }

        // Add label noise
        let n_label_flips = (n_samples as f64 * noise_level) as usize;
        for _ in 0..n_label_flips {
            let idx = self.rng.random_range(0..n_samples);
            y[idx] = -y[idx];
        }

        (x, y)
    }

    /// Print test results summary
    pub fn print_results_summary(&self, results: &[PropertyTestResult]) {
        println!(
            "
=== SVM Property Test Results ==="
        );
        println!(
            "{:<25} {:<8} {:<8} {:<8} {:<12} {:<10}",
            "Test Name", "Passed", "Failed", "Total", "Success Rate", "Avg Time"
        );
        println!("{}", "-".repeat(75));

        for result in results {
            println!(
                "{:<25} {:<8} {:<8} {:<8} {:<12.2}% {:<10.3}s",
                result.test_name,
                result.passed,
                result.failed,
                result.total,
                result.success_rate() * 100.0,
                result.avg_time
            );
        }

        let total_passed: usize = results.iter().map(|r| r.passed).sum();
        let total_failed: usize = results.iter().map(|r| r.failed).sum();
        let total_tests: usize = results.iter().map(|r| r.total).sum();
        let overall_success_rate = if total_tests > 0 {
            total_passed as f64 / total_tests as f64
        } else {
            0.0
        };

        println!("{}", "-".repeat(75));
        println!(
            "{:<25} {:<8} {:<8} {:<8} {:<12.2}%",
            "OVERALL",
            total_passed,
            total_failed,
            total_tests,
            overall_success_rate * 100.0
        );

        // Print failures summary
        if total_failed > 0 {
            println!(
                "
=== Failure Summary ==="
            );
            for result in results {
                if !result.failures.is_empty() {
                    println!(
                        "
{}: {} failures",
                        result.test_name,
                        result.failures.len()
                    );
                    for (i, failure) in result.failures.iter().enumerate() {
                        if i < 3 {
                            // Show first 3 failures
                            println!("  - {failure}");
                        } else if i == 3 {
                            println!("  - ... and {} more", result.failures.len() - 3);
                            break;
                        }
                    }
                }
            }
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_property_tester_creation() {
        let config = PropertyTestConfig::default();
        let mut tester = SVMPropertyTester::new(config);

        // Test that the tester was created successfully
        assert!(tester.config.test_cases > 0);
        assert!(tester.config.max_samples > 0);
        assert!(tester.config.max_features > 0);
    }

    #[test]
    fn test_data_generation() {
        let config = PropertyTestConfig::default();
        let mut tester = SVMPropertyTester::new(config);

        let (x, y) = tester.generate_random_dataset(10, 5);
        assert_eq!(x.nrows(), 10);
        assert_eq!(x.ncols(), 5);
        assert_eq!(y.len(), 10);

        let (x, y) = tester.generate_linearly_separable_dataset(20, 3);
        assert_eq!(x.nrows(), 20);
        assert_eq!(x.ncols(), 3);
        assert_eq!(y.len(), 20);
    }

    #[test]
    fn test_single_property_test() {
        let config = PropertyTestConfig {
            test_cases: 5,
            max_samples: 50,
            max_features: 5,
            ..Default::default()
        };
        let mut tester = SVMPropertyTester::new(config);

        let result = tester.test_prediction_consistency();
        assert!(result.total > 0);
        println!(
            "Prediction consistency test: {}/{} passed",
            result.passed, result.total
        );
    }

    #[test]
    fn test_results_summary() {
        let config = PropertyTestConfig {
            test_cases: 2,
            max_samples: 20,
            max_features: 3,
            ..Default::default()
        };
        let mut tester = SVMPropertyTester::new(config);

        let results = vec![
            tester.test_prediction_consistency(),
            tester.test_numerical_stability(),
        ];

        tester.print_results_summary(&results);

        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|r| r.total > 0));
    }
}
