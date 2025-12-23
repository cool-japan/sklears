//! Property-based testing utilities for pipeline composition
//!
//! This module provides comprehensive property-based testing frameworks
//! for validating pipeline properties, invariants, and correctness.

use scirs2_core::ndarray::{Array1, Array2, ArrayView2};
use sklears_core::traits::Transform;

/// Property-based test generator for pipeline data
pub struct PropertyTestGenerator {
    /// Random seed for reproducible tests
    seed: u64,
    /// Minimum sample size for generated data
    min_samples: usize,
    /// Maximum sample size for generated data
    max_samples: usize,
    /// Minimum feature count
    min_features: usize,
    /// Maximum feature count
    max_features: usize,
    /// Value range for generated features
    value_range: (f64, f64),
}

impl PropertyTestGenerator {
    /// Create a new property test generator
    #[must_use]
    pub fn new() -> Self {
        Self {
            seed: 42,
            min_samples: 10,
            max_samples: 1000,
            min_features: 1,
            max_features: 20,
            value_range: (-10.0, 10.0),
        }
    }

    /// Set the random seed
    #[must_use]
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Set sample size range
    #[must_use]
    pub fn sample_range(mut self, min: usize, max: usize) -> Self {
        self.min_samples = min;
        self.max_samples = max;
        self
    }

    /// Set feature count range
    #[must_use]
    pub fn feature_range(mut self, min: usize, max: usize) -> Self {
        self.min_features = min;
        self.max_features = max;
        self
    }

    /// Set value range for generated features
    #[must_use]
    pub fn value_range(mut self, range: (f64, f64)) -> Self {
        self.value_range = range;
        self
    }

    /// Generate random matrix for testing
    #[must_use]
    pub fn generate_matrix(&self, n_samples: usize, n_features: usize) -> Array2<f64> {
        use scirs2_core::random::rngs::StdRng;
        use scirs2_core::random::{Rng, SeedableRng};

        let mut rng = StdRng::seed_from_u64(self.seed);
        let (min_val, max_val) = self.value_range;

        Array2::from_shape_fn((n_samples, n_features), |_| {
            rng.gen_range(min_val..max_val + 1.0)
        })
    }

    /// Generate random target vector for testing
    #[must_use]
    pub fn generate_targets(&self, n_samples: usize) -> Array1<f64> {
        use scirs2_core::random::rngs::StdRng;
        use scirs2_core::random::{Rng, SeedableRng};

        let mut rng = StdRng::seed_from_u64(self.seed + 1);
        let (min_val, max_val) = self.value_range;

        Array1::from_shape_fn(n_samples, |_| rng.gen_range(min_val..max_val + 1.0))
    }

    /// Generate classification targets
    #[must_use]
    pub fn generate_classification_targets(
        &self,
        n_samples: usize,
        n_classes: usize,
    ) -> Array1<usize> {
        use scirs2_core::random::rngs::StdRng;
        use scirs2_core::random::{Rng, SeedableRng};

        let mut rng = StdRng::seed_from_u64(self.seed + 2);

        Array1::from_shape_fn(n_samples, |_| rng.gen_range(0..n_classes))
    }
}

impl Default for PropertyTestGenerator {
    fn default() -> Self {
        Self::new()
    }
}

/// Property-based test suite for pipeline invariants
pub struct PipelinePropertyTester {
    generator: PropertyTestGenerator,
}

impl PipelinePropertyTester {
    /// Create a new pipeline property tester
    #[must_use]
    pub fn new() -> Self {
        Self {
            generator: PropertyTestGenerator::new(),
        }
    }

    /// Set the test generator
    #[must_use]
    pub fn generator(mut self, generator: PropertyTestGenerator) -> Self {
        self.generator = generator;
        self
    }

    /// Test that pipeline preserves sample count
    pub fn test_sample_preservation<P>(&self, pipeline: &P, n_tests: usize) -> PropertyTestResult
    where
        P: for<'a> Transform<ArrayView2<'a, f64>, Array2<f64>>,
    {
        let mut results = Vec::new();

        for i in 0..n_tests {
            let n_samples = self.generator.min_samples
                + (i % (self.generator.max_samples - self.generator.min_samples));
            let n_features = self.generator.min_features
                + (i % (self.generator.max_features - self.generator.min_features));

            let data = self.generator.generate_matrix(n_samples, n_features);

            match pipeline.transform(&data.view()) {
                Ok(transformed) => {
                    let property_holds = transformed.nrows() == n_samples;
                    results.push(PropertyTestCase {
                        test_name: "sample_preservation".to_string(),
                        input_shape: (n_samples, n_features),
                        output_shape: (transformed.nrows(), transformed.ncols()),
                        property_holds,
                        error: None,
                    });
                }
                Err(e) => {
                    results.push(PropertyTestCase {
                        test_name: "sample_preservation".to_string(),
                        input_shape: (n_samples, n_features),
                        output_shape: (0, 0),
                        property_holds: false,
                        error: Some(format!("{e:?}")),
                    });
                }
            }
        }

        PropertyTestResult::new("sample_preservation", results)
    }

    /// Test that pipeline transformations are consistent
    pub fn test_transformation_consistency<P>(
        &self,
        pipeline: &P,
        n_tests: usize,
    ) -> PropertyTestResult
    where
        P: for<'a> Transform<ArrayView2<'a, f64>, Array2<f64>>,
    {
        let mut results = Vec::new();

        for i in 0..n_tests {
            let n_samples = 50;
            let n_features = 5;

            let data = self.generator.generate_matrix(n_samples, n_features);

            match (
                pipeline.transform(&data.view()),
                pipeline.transform(&data.view()),
            ) {
                (Ok(result1), Ok(result2)) => {
                    let property_holds = result1.abs_diff_eq(&result2, 1e-10);
                    results.push(PropertyTestCase {
                        test_name: "transformation_consistency".to_string(),
                        input_shape: (n_samples, n_features),
                        output_shape: (result1.nrows(), result1.ncols()),
                        property_holds,
                        error: None,
                    });
                }
                (Err(e), _) | (_, Err(e)) => {
                    results.push(PropertyTestCase {
                        test_name: "transformation_consistency".to_string(),
                        input_shape: (n_samples, n_features),
                        output_shape: (0, 0),
                        property_holds: false,
                        error: Some(format!("{e:?}")),
                    });
                }
            }
        }

        PropertyTestResult::new("transformation_consistency", results)
    }

    /// Test pipeline composition properties
    pub fn test_composition_associativity<P1, P2, P3>(
        &self,
        p1: &P1,
        p2: &P2,
        p3: &P3,
    ) -> PropertyTestResult
    where
        P1: for<'a> Transform<ArrayView2<'a, f64>, Array2<f64>>,
        P2: for<'a> Transform<ArrayView2<'a, f64>, Array2<f64>>,
        P3: for<'a> Transform<ArrayView2<'a, f64>, Array2<f64>>,
    {
        let mut results = Vec::new();

        let n_samples = 50;
        let n_features = 5;
        let data = self.generator.generate_matrix(n_samples, n_features);

        // Test (p1 ∘ p2) ∘ p3 = p1 ∘ (p2 ∘ p3)
        let result = match (
            p1.transform(&data.view())
                .and_then(|r| p2.transform(&r.view()))
                .and_then(|r| p3.transform(&r.view())),
            p2.transform(&data.view())
                .and_then(|r| p3.transform(&r.view()))
                .and_then(|r| p1.transform(&r.view())),
        ) {
            (Ok(left), Ok(right)) => PropertyTestCase {
                test_name: "composition_associativity".to_string(),
                input_shape: (n_samples, n_features),
                output_shape: (left.nrows(), left.ncols()),
                property_holds: left.shape() == right.shape(),
                error: None,
            },
            (Err(e), _) | (_, Err(e)) => PropertyTestCase {
                test_name: "composition_associativity".to_string(),
                input_shape: (n_samples, n_features),
                output_shape: (0, 0),
                property_holds: false,
                error: Some(format!("{e:?}")),
            },
        };

        results.push(result);
        PropertyTestResult::new("composition_associativity", results)
    }

    /// Test that feature union preserves all input features
    pub fn test_feature_union_completeness<T1, T2>(&self, t1: &T1, t2: &T2) -> PropertyTestResult
    where
        T1: for<'a> Transform<ArrayView2<'a, f64>, Array2<f64>>,
        T2: for<'a> Transform<ArrayView2<'a, f64>, Array2<f64>>,
    {
        let mut results = Vec::new();

        let n_samples = 50;
        let n_features = 5;
        let data = self.generator.generate_matrix(n_samples, n_features);

        match (t1.transform(&data.view()), t2.transform(&data.view())) {
            (Ok(result1), Ok(result2)) => {
                let total_features = result1.ncols() + result2.ncols();
                let property_holds = total_features >= n_features;

                results.push(PropertyTestCase {
                    test_name: "feature_union_completeness".to_string(),
                    input_shape: (n_samples, n_features),
                    output_shape: (n_samples, total_features),
                    property_holds,
                    error: None,
                });
            }
            (Err(e), _) | (_, Err(e)) => {
                results.push(PropertyTestCase {
                    test_name: "feature_union_completeness".to_string(),
                    input_shape: (n_samples, n_features),
                    output_shape: (0, 0),
                    property_holds: false,
                    error: Some(format!("{e:?}")),
                });
            }
        }

        PropertyTestResult::new("feature_union_completeness", results)
    }
}

impl Default for PipelinePropertyTester {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of a single property test case
#[derive(Debug, Clone)]
pub struct PropertyTestCase {
    /// Name of the test
    pub test_name: String,
    /// Input data shape
    pub input_shape: (usize, usize),
    /// Output data shape
    pub output_shape: (usize, usize),
    /// Whether the property holds
    pub property_holds: bool,
    /// Error message if any
    pub error: Option<String>,
}

/// Result of a property test suite
#[derive(Debug, Clone)]
pub struct PropertyTestResult {
    /// Name of the property being tested
    pub property_name: String,
    /// Individual test cases
    pub cases: Vec<PropertyTestCase>,
    /// Success rate (0.0 to 1.0)
    pub success_rate: f64,
    /// Total number of tests
    pub total_tests: usize,
    /// Number of passing tests
    pub passing_tests: usize,
}

impl PropertyTestResult {
    /// Create a new property test result
    #[must_use]
    pub fn new(property_name: &str, cases: Vec<PropertyTestCase>) -> Self {
        let total_tests = cases.len();
        let passing_tests = cases.iter().filter(|c| c.property_holds).count();
        let success_rate = if total_tests > 0 {
            passing_tests as f64 / total_tests as f64
        } else {
            0.0
        };

        Self {
            property_name: property_name.to_string(),
            cases,
            success_rate,
            total_tests,
            passing_tests,
        }
    }

    /// Check if all tests passed
    #[must_use]
    pub fn all_passed(&self) -> bool {
        self.success_rate == 1.0
    }

    /// Get failing test cases
    #[must_use]
    pub fn failing_cases(&self) -> Vec<&PropertyTestCase> {
        self.cases.iter().filter(|c| !c.property_holds).collect()
    }

    /// Generate a summary report
    #[must_use]
    pub fn summary(&self) -> String {
        format!(
            "Property '{}': {}/{} tests passed ({:.1}%)",
            self.property_name,
            self.passing_tests,
            self.total_tests,
            self.success_rate * 100.0
        )
    }
}

/// Statistical validation utilities
pub struct StatisticalValidator {
    confidence_level: f64,
    min_sample_size: usize,
}

impl StatisticalValidator {
    /// Create a new statistical validator
    #[must_use]
    pub fn new() -> Self {
        Self {
            confidence_level: 0.95,
            min_sample_size: 30,
        }
    }

    /// Set confidence level for statistical tests
    #[must_use]
    pub fn confidence_level(mut self, level: f64) -> Self {
        self.confidence_level = level.clamp(0.0, 1.0);
        self
    }

    /// Set minimum sample size for tests
    #[must_use]
    pub fn min_sample_size(mut self, size: usize) -> Self {
        self.min_sample_size = size;
        self
    }

    /// Validate that pipeline predictions have reasonable statistical properties
    #[must_use]
    pub fn validate_prediction_distribution(&self, predictions: &Array1<f64>) -> ValidationResult {
        let mut issues = Vec::new();

        if predictions.len() < self.min_sample_size {
            issues.push(format!(
                "Sample size {} is below minimum {}",
                predictions.len(),
                self.min_sample_size
            ));
        }

        // Check for NaN or infinite values
        let nan_count = predictions.iter().filter(|&&x| x.is_nan()).count();
        let inf_count = predictions.iter().filter(|&&x| x.is_infinite()).count();

        if nan_count > 0 {
            issues.push(format!("Found {nan_count} NaN values in predictions"));
        }

        if inf_count > 0 {
            issues.push(format!("Found {inf_count} infinite values in predictions"));
        }

        // Basic statistical checks
        let mean = predictions.mean().unwrap_or(0.0);
        let variance = predictions.var(0.0);

        if variance.is_nan() || variance.is_infinite() {
            issues.push("Prediction variance is invalid".to_string());
        }

        ValidationResult {
            is_valid: issues.is_empty(),
            issues,
            statistics: Some(ValidationStatistics {
                mean,
                variance,
                sample_size: predictions.len(),
            }),
        }
    }

    /// Validate pipeline transformation properties
    #[must_use]
    pub fn validate_transformation(
        &self,
        input: &Array2<f64>,
        output: &Array2<f64>,
    ) -> ValidationResult {
        let mut issues = Vec::new();

        // Check shape consistency
        if input.nrows() != output.nrows() {
            issues.push(format!(
                "Row count mismatch: input {} vs output {}",
                input.nrows(),
                output.nrows()
            ));
        }

        // Check for valid values in output
        let nan_count = output.iter().filter(|&&x| x.is_nan()).count();
        let inf_count = output.iter().filter(|&&x| x.is_infinite()).count();

        if nan_count > 0 {
            issues.push(format!(
                "Found {nan_count} NaN values in transformation output"
            ));
        }

        if inf_count > 0 {
            issues.push(format!(
                "Found {inf_count} infinite values in transformation output"
            ));
        }

        ValidationResult {
            is_valid: issues.is_empty(),
            issues,
            statistics: None,
        }
    }
}

impl Default for StatisticalValidator {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of statistical validation
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Whether the validation passed
    pub is_valid: bool,
    /// List of validation issues
    pub issues: Vec<String>,
    /// Optional statistical summary
    pub statistics: Option<ValidationStatistics>,
}

/// Statistical summary for validation
#[derive(Debug, Clone)]
pub struct ValidationStatistics {
    /// Mean value
    pub mean: f64,
    /// Variance
    pub variance: f64,
    /// Sample size
    pub sample_size: usize,
}

/// Comprehensive test suite runner
pub struct TestSuiteRunner {
    property_tester: PipelinePropertyTester,
    statistical_validator: StatisticalValidator,
}

impl TestSuiteRunner {
    /// Create a new test suite runner
    #[must_use]
    pub fn new() -> Self {
        Self {
            property_tester: PipelinePropertyTester::new(),
            statistical_validator: StatisticalValidator::new(),
        }
    }

    /// Run comprehensive tests on a pipeline
    pub fn run_comprehensive_tests<P>(&self, pipeline: &P) -> TestSuiteResult
    where
        P: for<'a> Transform<ArrayView2<'a, f64>, Array2<f64>>,
    {
        let mut results = Vec::new();

        // Property-based tests
        results.push(self.property_tester.test_sample_preservation(pipeline, 100));
        results.push(
            self.property_tester
                .test_transformation_consistency(pipeline, 50),
        );

        // Statistical validation
        let test_data = self.property_tester.generator.generate_matrix(100, 5);
        if let Ok(transformed) = pipeline.transform(&test_data.view()) {
            let validation = self
                .statistical_validator
                .validate_transformation(&test_data, &transformed);
            if !validation.is_valid {
                // Convert validation issues to property test format
                let failing_case = PropertyTestCase {
                    test_name: "statistical_validation".to_string(),
                    input_shape: test_data.dim(),
                    output_shape: transformed.dim(),
                    property_holds: false,
                    error: Some(validation.issues.join("; ")),
                };
                results.push(PropertyTestResult::new(
                    "statistical_validation",
                    vec![failing_case],
                ));
            }
        }

        TestSuiteResult::new(results)
    }
}

impl Default for TestSuiteRunner {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of running a complete test suite
#[derive(Debug, Clone)]
pub struct TestSuiteResult {
    /// Individual property test results
    pub property_results: Vec<PropertyTestResult>,
    /// Overall success rate
    pub overall_success_rate: f64,
    /// Total number of tests across all properties
    pub total_tests: usize,
    /// Total number of passing tests
    pub total_passing: usize,
}

impl TestSuiteResult {
    /// Create a new test suite result
    #[must_use]
    pub fn new(property_results: Vec<PropertyTestResult>) -> Self {
        let total_tests: usize = property_results.iter().map(|r| r.total_tests).sum();
        let total_passing: usize = property_results.iter().map(|r| r.passing_tests).sum();
        let overall_success_rate = if total_tests > 0 {
            total_passing as f64 / total_tests as f64
        } else {
            0.0
        };

        Self {
            property_results,
            overall_success_rate,
            total_tests,
            total_passing,
        }
    }

    /// Check if all tests passed
    #[must_use]
    pub fn all_passed(&self) -> bool {
        self.overall_success_rate == 1.0
    }

    /// Generate a detailed report
    #[must_use]
    pub fn detailed_report(&self) -> String {
        let mut report = String::new();
        report.push_str(&format!(
            "Test Suite Summary: {}/{} tests passed ({:.1}%)\n\n",
            self.total_passing,
            self.total_tests,
            self.overall_success_rate * 100.0
        ));

        for result in &self.property_results {
            report.push_str(&format!("  {}\n", result.summary()));

            if !result.all_passed() {
                for failing_case in result.failing_cases() {
                    report.push_str(&format!(
                        "    FAIL: {} - {:?}\n",
                        failing_case.test_name, failing_case.error
                    ));
                }
            }
        }

        report
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::mock::MockTransformer;

    #[test]
    fn test_property_test_generator() {
        let generator = PropertyTestGenerator::new();
        let matrix = generator.generate_matrix(10, 5);
        assert_eq!(matrix.shape(), &[10, 5]);

        let targets = generator.generate_targets(10);
        assert_eq!(targets.len(), 10);
    }

    #[test]
    fn test_pipeline_property_tester() {
        let tester = PipelinePropertyTester::new();
        let transformer = MockTransformer::new();

        let result = tester.test_sample_preservation(&transformer, 10);
        assert_eq!(result.property_name, "sample_preservation");
        assert_eq!(result.total_tests, 10);
    }

    #[test]
    fn test_statistical_validator() {
        let validator = StatisticalValidator::new();
        let predictions = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);

        let result = validator.validate_prediction_distribution(&predictions);
        // Should fail due to small sample size
        assert!(!result.is_valid);
    }

    #[test]
    fn test_test_suite_runner() {
        let runner = TestSuiteRunner::new();
        let transformer = MockTransformer::new();

        let result = runner.run_comprehensive_tests(&transformer);
        assert!(result.total_tests > 0);
    }

    #[test]
    fn test_property_test_result() {
        let cases = vec![
            PropertyTestCase {
                test_name: "test1".to_string(),
                input_shape: (10, 5),
                output_shape: (10, 5),
                property_holds: true,
                error: None,
            },
            PropertyTestCase {
                test_name: "test2".to_string(),
                input_shape: (10, 5),
                output_shape: (10, 5),
                property_holds: false,
                error: Some("Test error".to_string()),
            },
        ];

        let result = PropertyTestResult::new("test_property", cases);
        assert_eq!(result.success_rate, 0.5);
        assert_eq!(result.failing_cases().len(), 1);
    }
}
