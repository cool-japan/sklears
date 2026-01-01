/// Test utilities for common testing patterns in machine learning
///
/// This module provides utilities for testing machine learning algorithms,
/// including data generation, test assertions, property-based testing helpers,
/// and benchmarking utilities.
#[cfg(test)]
use crate::error::SklearsError;
use crate::traits::Fit;
#[cfg(test)]
use crate::types::{Array1, Array2, Float};
#[cfg(test)]
use approx::abs_diff_eq;
// SciRS2 Policy: Using scirs2_core for unified access (COMPLIANT)
#[cfg(test)]
use proptest::prelude::*;
#[cfg(test)]
use scirs2_core::ndarray::Array;
#[cfg(test)]
use scirs2_core::random::{thread_rng, Random};
#[cfg(test)]
use std::collections::HashMap;
#[cfg(test)]
use std::sync::Mutex;
#[cfg(test)]
use std::time::Duration;

/// Test data generation utilities
#[cfg(test)]
pub mod generators {
    use super::*;
    use crate::error::Result;

    /// Generate synthetic regression data with known relationships
    pub fn make_regression_data(
        n_samples: usize,
        n_features: usize,
        noise: f64,
        seed: Option<u64>,
    ) -> Result<(Array2<Float>, Array1<Float>)> {
        let mut rng = match seed {
            Some(s) => Random::seed(s),
            None => Random::seed(42), // Default seed for reproducible tests
        };

        // Generate random features using Box-Muller transform
        let mut x_data = Vec::with_capacity(n_samples * n_features);
        for _ in 0..(n_samples * n_features + 1) / 2 {
            let u1: f64 = rng.gen();
            let u2: f64 = rng.gen();
            let z0 = (-2.0f64 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            let z1 = (-2.0f64 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).sin();
            x_data.push(z0);
            if x_data.len() < n_samples * n_features {
                x_data.push(z1);
            }
        }
        x_data.truncate(n_samples * n_features);
        let x = Array::from_shape_vec((n_samples, n_features), x_data)
            .map_err(|e| SklearsError::Other(e.to_string()))?;

        // Generate random coefficients
        let mut coef = Vec::with_capacity(n_features);
        for _ in 0..n_features {
            coef.push(rng.random_range(-5.0..5.0));
        }

        // Generate targets: y = X @ coef + noise
        let mut y_data = Vec::with_capacity(n_samples);

        for i in 0..n_samples {
            let mut y_i = 0.0;
            for j in 0..n_features {
                y_i += x[[i, j]] * coef[j];
            }
            // Add noise using Box-Muller transform
            let u1: f64 = rng.gen();
            let u2: f64 = rng.gen();
            let noise_val =
                noise * (-2.0f64 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            y_i += noise_val;
            y_data.push(y_i);
        }
        let y = Array::from_vec(y_data);

        Ok((x, y))
    }

    /// Generate synthetic classification data with known classes
    pub fn make_classification_data(
        n_samples: usize,
        n_features: usize,
        n_classes: usize,
        cluster_std: f64,
        seed: Option<u64>,
    ) -> Result<(Array2<Float>, Array1<Float>)> {
        let mut rng = match seed {
            Some(s) => Random::seed(s),
            None => Random::seed(42), // Default seed for reproducible tests
        };

        if n_classes < 2 {
            return Err(SklearsError::InvalidParameter {
                name: "n_classes".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }

        let samples_per_class = n_samples / n_classes;
        let remaining_samples = n_samples % n_classes;

        // Generate random class centers
        let mut centers = Vec::with_capacity(n_classes * n_features);
        for _ in 0..n_classes * n_features {
            centers.push(rng.random_range(-10.0..10.0));
        }

        let mut x_data = Vec::with_capacity(n_samples * n_features);
        let mut y_data = Vec::with_capacity(n_samples);

        // Generate samples for each class
        for class_idx in 0..n_classes {
            let class_samples = if class_idx < remaining_samples {
                samples_per_class + 1
            } else {
                samples_per_class
            };

            for _ in 0..class_samples {
                for feature_idx in 0..n_features {
                    let center_value = centers[class_idx * n_features + feature_idx];
                    // Generate normal random value using Box-Muller transform
                    let u1: f64 = rng.gen();
                    let u2: f64 = rng.gen();
                    let normal_val = cluster_std
                        * (-2.0f64 * u1.ln()).sqrt()
                        * (2.0 * std::f64::consts::PI * u2).cos();
                    x_data.push(center_value + normal_val);
                }
                y_data.push(class_idx as Float);
            }
        }

        let x = Array::from_shape_vec((n_samples, n_features), x_data)
            .map_err(|e| SklearsError::Other(e.to_string()))?;
        let y = Array::from_vec(y_data);

        Ok((x, y))
    }

    /// Generate data with specific properties for testing edge cases
    pub fn make_edge_case_data(case: EdgeCase) -> Result<(Array2<Float>, Array1<Float>)> {
        match case {
            EdgeCase::Empty => Ok((Array2::zeros((0, 0)), Array1::zeros(0))),
            EdgeCase::SingleSample => {
                let x = Array2::from_shape_vec((1, 3), vec![1.0, 2.0, 3.0])
                    .map_err(|e| SklearsError::Other(e.to_string()))?;
                let y = Array1::from_vec(vec![1.0]);
                Ok((x, y))
            }
            EdgeCase::SingleFeature => {
                let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0])
                    .map_err(|e| SklearsError::Other(e.to_string()))?;
                let y = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
                Ok((x, y))
            }
            EdgeCase::MoreFeaturesThanSamples => {
                let x = Array2::from_shape_vec(
                    (2, 5),
                    vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                )
                .map_err(|e| SklearsError::Other(e.to_string()))?;
                let y = Array1::from_vec(vec![1.0, 2.0]);
                Ok((x, y))
            }
            EdgeCase::PerfectlyCorrelated => {
                let x_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
                let x = Array2::from_shape_vec((5, 1), x_data.clone())
                    .map_err(|e| SklearsError::Other(e.to_string()))?;
                let y = Array1::from_vec(x_data.iter().map(|&x| 2.0 * x + 1.0).collect());
                Ok((x, y))
            }
            EdgeCase::ConstantTarget => {
                let x = Array2::from_shape_vec(
                    (5, 2),
                    vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                )
                .map_err(|e| SklearsError::Other(e.to_string()))?;
                let y = Array1::from_vec(vec![5.0, 5.0, 5.0, 5.0, 5.0]);
                Ok((x, y))
            }
            EdgeCase::WithOutliers => {
                let mut x_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
                x_data.extend(vec![100.0, -100.0]); // Outliers
                let x = Array2::from_shape_vec((5, 2), x_data)
                    .map_err(|e| SklearsError::Other(e.to_string()))?;
                let y = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 1000.0]); // Outlier in target
                Ok((x, y))
            }
        }
    }

    /// Common edge cases for testing
    #[derive(Debug, Clone)]
    pub enum EdgeCase {
        Empty,
        SingleSample,
        SingleFeature,
        MoreFeaturesThanSamples,
        PerfectlyCorrelated,
        ConstantTarget,
        WithOutliers,
    }

    /// Property-based test generators
    pub mod proptest_generators {
        use super::*;

        /// Generate arbitrary feature matrices
        pub fn feature_matrix(
            max_samples: usize,
            max_features: usize,
        ) -> impl Strategy<Value = Array2<Float>> {
            (1..=max_samples, 1..=max_features).prop_flat_map(|(n_samples, n_features)| {
                prop::collection::vec(-10.0..10.0, n_samples * n_features).prop_map(move |data| {
                    Array::from_shape_vec((n_samples, n_features), data).unwrap()
                })
            })
        }

        /// Generate arbitrary target vectors
        pub fn target_vector(max_samples: usize) -> impl Strategy<Value = Array1<Float>> {
            (1..=max_samples).prop_flat_map(|n_samples| {
                prop::collection::vec(-10.0..10.0, n_samples).prop_map(Array::from_vec)
            })
        }

        /// Generate classification targets
        pub fn classification_targets(
            max_samples: usize,
            n_classes: usize,
        ) -> impl Strategy<Value = Array1<Float>> {
            (1..=max_samples).prop_flat_map(move |n_samples| {
                prop::collection::vec(0.0..(n_classes as Float), n_samples)
                    .prop_map(|data| Array::from_vec(data.into_iter().map(|x| x.floor()).collect()))
            })
        }

        /// Generate valid train/test split ratios
        pub fn train_test_ratio() -> impl Strategy<Value = f64> {
            0.1..0.9
        }

        /// Generate valid learning rates
        pub fn learning_rate() -> impl Strategy<Value = f64> {
            1e-6..1.0
        }

        /// Generate valid regularization strengths
        pub fn regularization_strength() -> impl Strategy<Value = f64> {
            0.0..10.0
        }
    }
}

/// Assertion utilities for machine learning tests
#[cfg(test)]
pub mod assertions {
    use super::*;
    use crate::error::Result;

    /// Assert that two arrays are approximately equal
    pub fn assert_arrays_close(
        actual: &Array2<Float>,
        expected: &Array2<Float>,
        tolerance: Float,
    ) -> Result<()> {
        if actual.shape() != expected.shape() {
            return Err(crate::error::SklearsError::ShapeMismatch {
                expected: format!("{:?}", expected.shape()),
                actual: format!("{:?}", actual.shape()),
            });
        }

        for ((&a, &e), idx) in actual
            .iter()
            .zip(expected.iter())
            .zip(scirs2_core::ndarray::indices(actual.dim()))
        {
            if !abs_diff_eq!(a, e, epsilon = tolerance) {
                return Err(crate::error::SklearsError::Other(format!(
                    "Arrays differ at index {:?}: actual={}, expected={}, tolerance={}",
                    idx, a, e, tolerance
                )));
            }
        }

        Ok(())
    }

    /// Assert that a model's predictions are within expected bounds
    pub fn assert_predictions_bounded(
        predictions: &Array1<Float>,
        min_val: Float,
        max_val: Float,
    ) -> Result<()> {
        for (i, &pred) in predictions.iter().enumerate() {
            if pred < min_val || pred > max_val {
                return Err(crate::error::SklearsError::Other(format!(
                    "Prediction at index {} is out of bounds: {} not in [{}, {}]",
                    i, pred, min_val, max_val
                )));
            }
        }
        Ok(())
    }

    /// Assert that probabilities sum to 1 for each sample
    pub fn assert_probabilities_valid(
        probabilities: &Array2<Float>,
        tolerance: Float,
    ) -> Result<()> {
        for (i, row) in probabilities
            .axis_iter(scirs2_core::ndarray::Axis(0))
            .enumerate()
        {
            let sum: Float = row.sum();
            if !abs_diff_eq!(sum, 1.0, epsilon = tolerance) {
                return Err(crate::error::SklearsError::Other(format!(
                    "Probabilities for sample {} sum to {}, expected 1.0 ± {}",
                    i, sum, tolerance
                )));
            }

            for (j, &prob) in row.iter().enumerate() {
                if prob < 0.0 || prob > 1.0 {
                    return Err(crate::error::SklearsError::Other(format!(
                        "Invalid probability at sample {}, class {}: {}",
                        i, j, prob
                    )));
                }
            }
        }
        Ok(())
    }

    /// Assert that a model shows improvement in metrics
    pub fn assert_learning_progress(
        metrics_history: &[Float],
        improvement_threshold: Float,
    ) -> Result<()> {
        if metrics_history.len() < 2 {
            return Err(crate::error::SklearsError::InvalidInput(
                "Need at least 2 metric values to check progress".to_string(),
            ));
        }

        let initial_metric = metrics_history[0];
        let final_metric = metrics_history[metrics_history.len() - 1];
        let improvement = (final_metric - initial_metric).abs();

        if improvement < improvement_threshold {
            return Err(crate::error::SklearsError::Other(format!(
                "Insufficient learning progress: improvement {} < threshold {}",
                improvement, improvement_threshold
            )));
        }

        Ok(())
    }

    /// Assert that training and validation metrics don't show severe overfitting
    pub fn assert_no_severe_overfitting(
        train_scores: &[Float],
        val_scores: &[Float],
        max_gap: Float,
    ) -> Result<()> {
        if train_scores.len() != val_scores.len() {
            return Err(crate::error::SklearsError::ShapeMismatch {
                expected: format!("{}", train_scores.len()),
                actual: format!("{}", val_scores.len()),
            });
        }

        for (i, (&train, &val)) in train_scores.iter().zip(val_scores.iter()).enumerate() {
            let gap = (train - val).abs();
            if gap > max_gap {
                return Err(crate::error::SklearsError::Other(format!(
                    "Severe overfitting detected at epoch {}: train={}, val={}, gap={}",
                    i, train, val, gap
                )));
            }
        }

        Ok(())
    }

    /// Assert that model performance is above baseline
    pub fn assert_above_baseline(score: Float, baseline: Float, metric_name: &str) -> Result<()> {
        if score <= baseline {
            return Err(crate::error::SklearsError::Other(format!(
                "{} score {} is not above baseline {}",
                metric_name, score, baseline
            )));
        }
        Ok(())
    }
}

/// Performance testing utilities
#[cfg(test)]
pub mod performance {
    use super::*;
    use std::time::Instant;

    /// Measure execution time of a function
    pub fn measure_time<F, R>(f: F) -> (R, Duration)
    where
        F: FnOnce() -> R,
    {
        let start = Instant::now();
        let result = f();
        let duration = start.elapsed();
        (result, duration)
    }

    /// Performance test configuration
    #[derive(Debug, Clone)]
    pub struct PerformanceConfig {
        pub max_duration: Duration,
        pub max_memory_mb: usize,
        pub min_throughput: Option<f64>, // operations per second
    }

    impl Default for PerformanceConfig {
        fn default() -> Self {
            Self {
                max_duration: Duration::from_secs(10),
                max_memory_mb: 1000, // 1GB
                min_throughput: None,
            }
        }
    }

    /// Performance test result
    #[derive(Debug, Clone)]
    pub struct PerformanceResult {
        pub duration: Duration,
        pub memory_used_mb: usize,
        pub throughput: Option<f64>,
    }

    /// Run a performance test with monitoring
    pub fn performance_test<F, R>(
        f: F,
        config: PerformanceConfig,
        operation_count: Option<usize>,
    ) -> crate::error::Result<(R, PerformanceResult)>
    where
        F: FnOnce() -> R,
    {
        let start_memory = get_memory_usage_mb();
        let (result, duration) = measure_time(f);
        let end_memory = get_memory_usage_mb();

        let memory_used = end_memory.saturating_sub(start_memory);
        let throughput = operation_count.map(|count| count as f64 / duration.as_secs_f64());

        // Check performance constraints
        if duration > config.max_duration {
            return Err(crate::error::SklearsError::Other(format!(
                "Performance test exceeded max duration: {:?} > {:?}",
                duration, config.max_duration
            )));
        }

        if memory_used > config.max_memory_mb {
            return Err(crate::error::SklearsError::Other(format!(
                "Performance test exceeded memory limit: {} MB > {} MB",
                memory_used, config.max_memory_mb
            )));
        }

        if let (Some(min_throughput), Some(actual_throughput)) = (config.min_throughput, throughput)
        {
            if actual_throughput < min_throughput {
                return Err(crate::error::SklearsError::Other(format!(
                    "Performance test below minimum throughput: {} ops/s < {} ops/s",
                    actual_throughput, min_throughput
                )));
            }
        }

        let perf_result = PerformanceResult {
            duration,
            memory_used_mb: memory_used,
            throughput,
        };

        Ok((result, perf_result))
    }

    /// Estimate memory usage (simplified implementation)
    fn get_memory_usage_mb() -> usize {
        // This is a simplified implementation
        // In a real scenario, you'd use system-specific APIs or libraries like `sysinfo`
        0
    }

    /// Benchmark different algorithm implementations
    pub fn benchmark_algorithms<F1, F2, R>(
        name1: &str,
        algorithm1: F1,
        name2: &str,
        algorithm2: F2,
    ) -> BenchmarkResult
    where
        F1: FnOnce() -> R,
        F2: FnOnce() -> R,
    {
        let (_, duration1) = measure_time(algorithm1);
        let (_, duration2) = measure_time(algorithm2);

        BenchmarkResult {
            algorithm1: AlgorithmResult {
                name: name1.to_string(),
                duration: duration1,
            },
            algorithm2: AlgorithmResult {
                name: name2.to_string(),
                duration: duration2,
            },
        }
    }

    #[derive(Debug, Clone)]
    pub struct BenchmarkResult {
        pub algorithm1: AlgorithmResult,
        pub algorithm2: AlgorithmResult,
    }

    #[derive(Debug, Clone)]
    pub struct AlgorithmResult {
        pub name: String,
        pub duration: Duration,
    }

    impl BenchmarkResult {
        pub fn faster_algorithm(&self) -> &AlgorithmResult {
            if self.algorithm1.duration < self.algorithm2.duration {
                &self.algorithm1
            } else {
                &self.algorithm2
            }
        }

        pub fn speedup_factor(&self) -> f64 {
            let slower_duration = self.algorithm1.duration.max(self.algorithm2.duration);
            let faster_duration = self.algorithm1.duration.min(self.algorithm2.duration);
            slower_duration.as_secs_f64() / faster_duration.as_secs_f64()
        }
    }
}

/// Test fixtures and common test data
#[cfg(test)]
pub mod fixtures {
    use super::*;
    use once_cell::sync::Lazy;

    /// Common test datasets that are expensive to generate
    pub static IRIS_DATASET: Lazy<(Array2<Float>, Array1<Float>)> = Lazy::new(|| {
        // Simplified iris dataset for testing
        let features = Array::from_shape_vec(
            (6, 4),
            vec![
                5.1, 3.5, 1.4, 0.2, // setosa
                4.9, 3.0, 1.4, 0.2, // setosa
                7.0, 3.2, 4.7, 1.4, // versicolor
                6.4, 3.2, 4.5, 1.5, // versicolor
                6.3, 3.3, 6.0, 2.5, // virginica
                5.8, 2.7, 5.1, 1.9, // virginica
            ],
        )
        .expect("Failed to create iris features");

        let targets = Array::from_vec(vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0]);

        (features, targets)
    });

    /// Regression dataset fixture
    pub static BOSTON_HOUSING: Lazy<(Array2<Float>, Array1<Float>)> = Lazy::new(|| {
        // Simplified version of Boston housing dataset
        generators::make_regression_data(100, 5, 0.1, Some(42))
            .expect("Failed to generate regression data")
    });

    /// Large dataset for performance testing
    pub static LARGE_DATASET: Lazy<(Array2<Float>, Array1<Float>)> = Lazy::new(|| {
        generators::make_regression_data(10000, 20, 0.1, Some(42))
            .expect("Failed to generate large dataset")
    });

    /// Test configuration presets
    pub mod configs {
        use super::performance::PerformanceConfig;
        use std::time::Duration;

        pub fn fast_test_config() -> PerformanceConfig {
            PerformanceConfig {
                max_duration: Duration::from_millis(100),
                max_memory_mb: 100,
                min_throughput: Some(1000.0),
            }
        }

        pub fn standard_test_config() -> PerformanceConfig {
            PerformanceConfig {
                max_duration: Duration::from_secs(5),
                max_memory_mb: 500,
                min_throughput: Some(100.0),
            }
        }

        pub fn intensive_test_config() -> PerformanceConfig {
            PerformanceConfig {
                max_duration: Duration::from_secs(60),
                max_memory_mb: 2000,
                min_throughput: Some(10.0),
            }
        }
    }
}

/// Mock implementations for testing
#[cfg(test)]
pub mod mocks {
    use super::*;
    use crate::traits::Predict;
    use std::marker::PhantomData;

    /// Mock linear model for testing
    #[derive(Debug, Clone)]
    pub struct MockLinearModel<X, Y> {
        pub coefficients: Option<Array1<Float>>,
        pub intercept: Float,
        _phantom: PhantomData<(X, Y)>,
    }

    impl<X, Y> MockLinearModel<X, Y> {
        pub fn new() -> Self {
            Self {
                coefficients: None,
                intercept: 0.0,
                _phantom: PhantomData,
            }
        }
    }

    impl<X, Y> Default for MockLinearModel<X, Y> {
        fn default() -> Self {
            Self::new()
        }
    }

    /// Mock implementation that always returns the same prediction
    #[derive(Debug, Clone)]
    pub struct ConstantPredictor {
        pub value: Float,
    }

    impl ConstantPredictor {
        pub fn new(value: Float) -> Self {
            Self { value }
        }
    }

    /// Mock implementation that introduces controlled errors
    #[derive(Debug, Clone)]
    pub struct ErrorProneModel {
        pub error_probability: f64,
        pub error_message: String,
    }

    impl ErrorProneModel {
        pub fn new(error_probability: f64, error_message: String) -> Self {
            Self {
                error_probability,
                error_message,
            }
        }

        fn should_error(&self) -> bool {
            thread_rng().gen::<f64>() < self.error_probability
        }
    }

    /// Trait for creating test doubles
    pub trait TestDouble {
        fn create_stub() -> Self;
        fn create_mock() -> Self;
        fn create_fake() -> Self;
    }

    /// Advanced mock framework for ML algorithms
    pub mod advanced_mocks {
        use super::*;
        use crate::traits::*;
        use std::sync::Arc;

        /// Mock behavior configuration
        #[derive(Debug, Clone)]
        pub enum MockBehavior<T> {
            /// Always return the same value
            ReturnConstant(T),
            /// Return values in sequence
            ReturnSequence(Vec<T>),
            /// Return value based on input
            ReturnFunction(fn(&dyn std::any::Any) -> T),
            /// Throw an error
            ThrowError(String),
            /// Call through to real implementation
            CallThrough,
        }

        /// Mock call recording
        #[derive(Debug, Clone)]
        pub struct MockCall {
            pub method_name: String,
            pub args: Vec<String>, // Simplified - would be more complex in real implementation
            pub timestamp: std::time::Instant,
        }

        /// Advanced mock that can record calls and configure behaviors
        #[derive(Debug)]
        pub struct AdvancedMock<T> {
            pub behaviors: HashMap<String, MockBehavior<T>>,
            pub call_history: Arc<Mutex<Vec<MockCall>>>,
            pub verification_failures: Arc<Mutex<Vec<String>>>,
        }

        impl<T: Clone> AdvancedMock<T> {
            pub fn new() -> Self {
                Self {
                    behaviors: HashMap::new(),
                    call_history: Arc::new(Mutex::new(Vec::new())),
                    verification_failures: Arc::new(Mutex::new(Vec::new())),
                }
            }

            /// Configure behavior for a method
            pub fn when(&mut self, method: &str, behavior: MockBehavior<T>) -> &mut Self {
                self.behaviors.insert(method.to_string(), behavior);
                self
            }

            /// Record a method call
            pub fn record_call(&self, method_name: &str, args: Vec<String>) {
                let call = MockCall {
                    method_name: method_name.to_string(),
                    args,
                    timestamp: std::time::Instant::now(),
                };

                if let Ok(mut history) = self.call_history.lock() {
                    history.push(call);
                }
            }

            /// Get call history
            pub fn get_call_history(&self) -> Vec<MockCall> {
                self.call_history
                    .lock()
                    .unwrap_or_else(|_| panic!("Failed to lock call history"))
                    .clone()
            }

            /// Verify that a method was called
            pub fn verify_called(&self, method_name: &str) -> bool {
                self.get_call_history()
                    .iter()
                    .any(|call| call.method_name == method_name)
            }

            /// Verify that a method was called with specific arguments
            pub fn verify_called_with(&self, method_name: &str, expected_args: &[&str]) -> bool {
                self.get_call_history().iter().any(|call| {
                    call.method_name == method_name
                        && call.args.len() == expected_args.len()
                        && call
                            .args
                            .iter()
                            .zip(expected_args.iter())
                            .all(|(a, &e)| a == e)
                })
            }

            /// Verify call count
            pub fn verify_call_count(&self, method_name: &str, expected_count: usize) -> bool {
                let actual_count = self
                    .get_call_history()
                    .iter()
                    .filter(|call| call.method_name == method_name)
                    .count();
                actual_count == expected_count
            }

            /// Reset all recorded calls
            pub fn reset(&self) {
                if let Ok(mut history) = self.call_history.lock() {
                    history.clear();
                }
                if let Ok(mut failures) = self.verification_failures.lock() {
                    failures.clear();
                }
            }
        }

        /// Mock estimator for testing trait implementations
        #[derive(Debug)]
        pub struct MockEstimator {
            mock: AdvancedMock<Array1<Float>>,
            fitted_state: bool,
            config: MockEstimatorConfig,
        }

        #[derive(Debug, Clone)]
        pub struct MockEstimatorConfig {
            pub learning_rate: f64,
            pub max_iter: usize,
            pub tolerance: f64,
        }

        impl Default for MockEstimatorConfig {
            fn default() -> Self {
                Self {
                    learning_rate: 0.01,
                    max_iter: 1000,
                    tolerance: 1e-6,
                }
            }
        }

        impl MockEstimator {
            pub fn new() -> Self {
                Self {
                    mock: AdvancedMock::new(),
                    fitted_state: false,
                    config: MockEstimatorConfig::default(),
                }
            }

            pub fn with_config(config: MockEstimatorConfig) -> Self {
                Self {
                    mock: AdvancedMock::new(),
                    fitted_state: false,
                    config,
                }
            }

            /// Configure mock to return specific predictions
            pub fn will_predict(&mut self, predictions: Array1<Float>) -> &mut Self {
                self.mock
                    .when("predict", MockBehavior::ReturnConstant(predictions));
                self
            }

            /// Configure mock to simulate fitting failure
            pub fn will_fail_fitting(&mut self, error_message: &str) -> &mut Self {
                self.mock
                    .when("fit", MockBehavior::ThrowError(error_message.to_string()));
                self
            }

            /// Get verification methods
            pub fn verify(&self) -> MockVerification {
                MockVerification {
                    call_history: self.mock.get_call_history(),
                }
            }
        }

        pub struct MockVerification {
            call_history: Vec<MockCall>,
        }

        impl MockVerification {
            pub fn fit_was_called(&self) -> bool {
                self.call_history
                    .iter()
                    .any(|call| call.method_name == "fit")
            }

            pub fn predict_was_called(&self) -> bool {
                self.call_history
                    .iter()
                    .any(|call| call.method_name == "predict")
            }

            pub fn fit_called_before_predict(&self) -> bool {
                let fit_time = self
                    .call_history
                    .iter()
                    .find(|call| call.method_name == "fit")
                    .map(|call| call.timestamp);

                let predict_time = self
                    .call_history
                    .iter()
                    .find(|call| call.method_name == "predict")
                    .map(|call| call.timestamp);

                match (fit_time, predict_time) {
                    (Some(fit), Some(predict)) => fit < predict,
                    _ => false,
                }
            }

            pub fn method_call_count(&self, method_name: &str) -> usize {
                self.call_history
                    .iter()
                    .filter(|call| call.method_name == method_name)
                    .count()
            }
        }

        /// Implement core traits for the mock estimator
        impl Estimator for MockEstimator {
            type Config = MockEstimatorConfig;
            type Error = crate::error::SklearsError;
            type Float = Float;

            fn config(&self) -> &Self::Config {
                &self.config
            }
        }

        impl GetParams for MockEstimator {
            fn get_params(&self) -> HashMap<String, String> {
                let mut params = HashMap::new();
                params.insert(
                    "learning_rate".to_string(),
                    self.config.learning_rate.to_string(),
                );
                params.insert("max_iter".to_string(), self.config.max_iter.to_string());
                params.insert("tolerance".to_string(), self.config.tolerance.to_string());
                params
            }
        }

        impl SetParams for MockEstimator {
            fn set_params(&mut self, params: HashMap<String, String>) -> crate::error::Result<()> {
                self.mock
                    .record_call("set_params", params.keys().cloned().collect());

                if let Some(lr) = params.get("learning_rate") {
                    self.config.learning_rate =
                        lr.parse()
                            .map_err(|_| crate::error::SklearsError::InvalidParameter {
                                name: "learning_rate".to_string(),
                                reason: "Invalid float value".to_string(),
                            })?;
                }

                Ok(())
            }
        }

        /// Mock fitted estimator
        #[derive(Debug)]
        pub struct MockFittedEstimator {
            predictions: Array1<Float>,
            mock: AdvancedMock<Array1<Float>>,
        }

        impl MockFittedEstimator {
            pub fn new(predictions: Array1<Float>) -> Self {
                Self {
                    predictions,
                    mock: AdvancedMock::new(),
                }
            }
        }

        impl Predict<Array2<Float>, Array1<Float>> for MockFittedEstimator {
            fn predict(&self, _x: &Array2<Float>) -> crate::error::Result<Array1<Float>> {
                Ok(self.predictions.clone())
            }
        }

        impl Fit<Array2<Float>, Array1<Float>> for MockEstimator {
            type Fitted = MockFittedEstimator;

            fn fit(
                self,
                x: &Array2<Float>,
                y: &Array1<Float>,
            ) -> crate::error::Result<Self::Fitted> {
                self.mock.record_call(
                    "fit",
                    vec![
                        format!("x_shape: {:?}", x.shape()),
                        format!("y_len: {}", y.len()),
                    ],
                );

                // Check if we should simulate an error
                if let Some(MockBehavior::ThrowError(msg)) = self.mock.behaviors.get("fit") {
                    return Err(crate::error::SklearsError::Other(msg.clone()));
                }

                // Simulate successful fitting by creating predictions
                let predictions = Array1::zeros(x.nrows());
                Ok(MockFittedEstimator::new(predictions))
            }
        }
    }
}

/// Contract testing framework for verifying trait implementations
#[cfg(test)]
pub mod contract_testing {
    use super::*;
    use crate::error::Result;
    use crate::traits::*;

    /// Contract specification for ML algorithm traits
    pub trait Contract<T> {
        /// Verify the contract is satisfied
        fn verify(&self, implementation: &T) -> Result<()>;

        /// Get contract name for reporting
        fn name(&self) -> &str;

        /// Get contract description
        fn description(&self) -> &str;
    }

    /// Contract for Estimator trait compliance
    pub struct EstimatorContract;

    impl EstimatorContract {
        pub fn new() -> Self {
            Self
        }
    }

    impl<T> Contract<T> for EstimatorContract
    where
        T: Estimator + GetParams + SetParams,
    {
        fn verify(&self, implementation: &T) -> Result<()> {
            // Test that configuration is accessible
            let _config = implementation.config();

            // Test that parameters can be retrieved
            let params = implementation.get_params();
            if params.is_empty() {
                log::warn!("Estimator has no parameters - this may be intentional");
            }

            // Test metadata accessibility
            let metadata = implementation.metadata();
            if metadata.name.is_empty() {
                return Err(crate::error::SklearsError::Other(
                    "Estimator metadata must have a non-empty name".to_string(),
                ));
            }

            Ok(())
        }

        fn name(&self) -> &str {
            "EstimatorContract"
        }

        fn description(&self) -> &str {
            "Verifies that an implementation correctly follows the Estimator trait contract"
        }
    }

    /// Contract for supervised learning algorithms
    pub struct SupervisedLearningContract {
        test_data: (Array2<Float>, Array1<Float>),
    }

    impl SupervisedLearningContract {
        pub fn new(test_data: (Array2<Float>, Array1<Float>)) -> Self {
            Self { test_data }
        }
    }

    impl<T> Contract<T> for SupervisedLearningContract
    where
        T: Fit<Array2<Float>, Array1<Float>> + Clone,
        T::Fitted: Predict<Array2<Float>, Array1<Float>>,
    {
        fn verify(&self, implementation: &T) -> Result<()> {
            let (x, y) = &self.test_data;

            // Verify that fitting returns a fitted model
            let fitted = implementation.clone().fit(x, y)?;

            // Verify that fitted model can make predictions
            let predictions = fitted.predict(x)?;

            // Verify prediction shape matches input
            if predictions.len() != x.nrows() {
                return Err(crate::error::SklearsError::ShapeMismatch {
                    expected: format!("{}", x.nrows()),
                    actual: format!("{}", predictions.len()),
                });
            }

            // Verify predictions are finite
            for (i, &pred) in predictions.iter().enumerate() {
                if !pred.is_finite() {
                    return Err(crate::error::SklearsError::Other(format!(
                        "Prediction at index {} is not finite: {}",
                        i, pred
                    )));
                }
            }

            Ok(())
        }

        fn name(&self) -> &str {
            "SupervisedLearningContract"
        }

        fn description(&self) -> &str {
            "Verifies that supervised learning algorithms follow the fit-predict contract"
        }
    }

    /// Contract for classification algorithms
    pub struct ClassificationContract {
        test_data: (Array2<Float>, Array1<Float>),
        n_classes: usize,
    }

    impl ClassificationContract {
        pub fn new(test_data: (Array2<Float>, Array1<Float>), n_classes: usize) -> Self {
            Self {
                test_data,
                n_classes,
            }
        }
    }

    impl<T> Contract<T> for ClassificationContract
    where
        T: Fit<Array2<Float>, Array1<Float>> + Clone,
        T::Fitted:
            Predict<Array2<Float>, Array1<Float>> + PredictProba<Array2<Float>, Array2<Float>>,
    {
        fn verify(&self, implementation: &T) -> Result<()> {
            let (x, y) = &self.test_data;

            // Verify supervised learning contract first
            let supervised_contract = SupervisedLearningContract::new(self.test_data.clone());
            supervised_contract.verify(implementation)?;

            // Additional classification-specific verification
            let fitted = implementation.clone().fit(x, y)?;

            // Verify probability predictions
            let probabilities = fitted.predict_proba(x)?;

            // Check probability shape
            if probabilities.shape()[0] != x.nrows() {
                return Err(crate::error::SklearsError::ShapeMismatch {
                    expected: format!("({}, {})", x.nrows(), self.n_classes),
                    actual: format!("{:?}", probabilities.shape()),
                });
            }

            // Verify probabilities are valid
            assertions::assert_probabilities_valid(&probabilities, 1e-6)?;

            Ok(())
        }

        fn name(&self) -> &str {
            "ClassificationContract"
        }

        fn description(&self) -> &str {
            "Verifies that classification algorithms follow probability prediction contracts"
        }
    }

    /// Contract for clustering algorithms
    pub struct ClusteringContract {
        test_data: Array2<Float>,
        expected_n_clusters: usize,
    }

    impl ClusteringContract {
        pub fn new(test_data: Array2<Float>, expected_n_clusters: usize) -> Self {
            Self {
                test_data,
                expected_n_clusters,
            }
        }
    }

    impl<T> Contract<T> for ClusteringContract
    where
        T: Cluster<Array2<Float>, Labels = Array1<Float>> + Clone,
    {
        fn verify(&self, implementation: &T) -> Result<()> {
            // Verify clustering produces labels
            let labels = implementation.clone().fit_predict(&self.test_data)?;

            // Verify label count matches sample count
            if labels.len() != self.test_data.nrows() {
                return Err(crate::error::SklearsError::ShapeMismatch {
                    expected: format!("{}", self.test_data.nrows()),
                    actual: format!("{}", labels.len()),
                });
            }

            // Verify all labels are non-negative integers
            for (i, &label) in labels.iter().enumerate() {
                if label < 0.0 || label.fract() != 0.0 {
                    return Err(crate::error::SklearsError::Other(format!(
                        "Invalid cluster label at index {}: {}",
                        i, label
                    )));
                }
            }

            // Verify the number of unique clusters
            let mut unique_labels: Vec<Float> = labels.to_vec();
            unique_labels.sort_by(|a, b| a.partial_cmp(b).unwrap());
            unique_labels.dedup();

            if unique_labels.len() > self.expected_n_clusters {
                return Err(crate::error::SklearsError::Other(format!(
                    "Too many clusters found: {} > {}",
                    unique_labels.len(),
                    self.expected_n_clusters
                )));
            }

            Ok(())
        }

        fn name(&self) -> &str {
            "ClusteringContract"
        }

        fn description(&self) -> &str {
            "Verifies that clustering algorithms produce valid cluster assignments"
        }
    }

    /// Contract test runner for comprehensive validation
    pub struct ContractTestRunner {
        contracts: Vec<Box<dyn ContractDyn>>,
        results: Vec<ContractResult>,
    }

    /// Dynamic contract trait for different types
    trait ContractDyn {
        fn verify_dyn(&self, implementation: &dyn std::any::Any) -> Result<()>;
        fn name(&self) -> &str;
        fn description(&self) -> &str;
    }

    /// Contract test result
    #[derive(Debug, Clone)]
    pub struct ContractResult {
        pub contract_name: String,
        pub passed: bool,
        pub error_message: Option<String>,
        pub execution_time: std::time::Duration,
    }

    impl ContractTestRunner {
        pub fn new() -> Self {
            Self {
                contracts: Vec::new(),
                results: Vec::new(),
            }
        }

        /// Add a contract to the test suite
        pub fn add_contract<T, C>(&mut self, _contract: C)
        where
            T: 'static,
            C: Contract<T> + 'static,
        {
            // In a real implementation, we would need better type erasure
            // This is a simplified version
        }

        /// Run all contracts against an implementation
        pub fn run_all<T>(&mut self, _implementation: &T) -> Vec<ContractResult>
        where
            T: 'static,
        {
            self.results.clear();

            // In a real implementation, would iterate through all contracts
            // For now, return empty results
            self.results.clone()
        }

        /// Get summary of contract test results
        pub fn summary(&self) -> ContractSummary {
            let total = self.results.len();
            let passed = self.results.iter().filter(|r| r.passed).count();
            let failed = total - passed;

            ContractSummary {
                total_contracts: total,
                passed,
                failed,
                total_execution_time: self.results.iter().map(|r| r.execution_time).sum(),
            }
        }
    }

    /// Summary of contract test execution
    #[derive(Debug, Clone)]
    pub struct ContractSummary {
        pub total_contracts: usize,
        pub passed: usize,
        pub failed: usize,
        pub total_execution_time: std::time::Duration,
    }

    impl ContractSummary {
        pub fn success_rate(&self) -> f64 {
            if self.total_contracts == 0 {
                1.0
            } else {
                self.passed as f64 / self.total_contracts as f64
            }
        }

        pub fn all_passed(&self) -> bool {
            self.failed == 0
        }
    }

    /// Convenience macro for running contract tests
    #[macro_export]
    macro_rules! assert_contract {
        ($implementation:expr, $contract:expr) => {
            match $contract.verify(&$implementation) {
                Ok(()) => {}
                Err(e) => panic!("Contract '{}' failed: {}", $contract.name(), e),
            }
        };
    }

    /// Test data generators for contract testing
    pub mod contract_data {
        use super::*;

        /// Generate appropriate test data for supervised learning contracts
        pub fn supervised_learning_data() -> Result<(Array2<Float>, Array1<Float>)> {
            generators::make_regression_data(50, 5, 0.1, Some(42))
        }

        /// Generate appropriate test data for classification contracts
        pub fn classification_data(n_classes: usize) -> Result<(Array2<Float>, Array1<Float>)> {
            generators::make_classification_data(100, 4, n_classes, 1.0, Some(42))
        }

        /// Generate appropriate test data for clustering contracts
        pub fn clustering_data() -> Result<Array2<Float>> {
            let (x, _) = generators::make_classification_data(75, 3, 3, 2.0, Some(42))?;
            Ok(x)
        }
    }
}

/// Utilities for property-based testing
#[cfg(test)]
pub mod property_testing {
    use super::*;

    /// Property test for checking algorithm invariants
    pub fn check_algorithm_invariants<F, A>(
        algorithm_factory: F,
        property_name: String,
    ) -> BoxedStrategy<TestCase>
    where
        F: Fn() -> A + Clone + 'static,
        A: 'static,
    {
        prop::collection::vec(
            generators::proptest_generators::feature_matrix(100, 10),
            1..=10,
        )
        .prop_map(move |_test_data| {
            TestCase {
                name: property_name.clone(),
                algorithm: Box::new(algorithm_factory()),
                expected_properties: vec![], // Would be defined based on the algorithm
            }
        })
        .boxed()
    }

    #[derive(Debug)]
    pub struct TestCase {
        pub name: String,
        pub algorithm: Box<dyn std::any::Any>,
        pub expected_properties: Vec<Property>,
    }

    #[derive(Debug, Clone)]
    pub enum Property {
        Deterministic,
        MonotonicIncrease,
        MonotonicDecrease,
        BoundedOutput { min: Float, max: Float },
        PreservesShape,
        Idempotent,
    }

    /// Common property test strategies
    pub mod strategies {
        use super::*;

        /// Strategy for testing with different data sizes
        pub fn data_sizes() -> BoxedStrategy<(usize, usize)> {
            prop_oneof![
                // Small datasets
                (1..=10usize, 1..=5usize),
                // Medium datasets
                (100..=1000usize, 5..=20usize),
                // Large datasets (less frequent)
                (5000..=10000usize, 10..=50usize).prop_map(|x| x).boxed(),
            ]
            .boxed()
        }

        /// Strategy for testing with different hyperparameters
        pub fn hyperparameters() -> BoxedStrategy<HashMap<String, Float>> {
            prop::collection::hash_map("[a-z_]+".prop_map(|s| s.to_string()), 0.0..10.0, 0..=10)
                .boxed()
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_make_regression_data() {
        let result = generators::make_regression_data(100, 5, 0.1, Some(42));
        assert!(result.is_ok());

        let (x, y) = result.unwrap();
        assert_eq!(x.shape(), &[100, 5]);
        assert_eq!(y.len(), 100);
    }

    #[test]
    fn test_make_classification_data() {
        let result = generators::make_classification_data(150, 4, 3, 1.0, Some(42));
        assert!(result.is_ok());

        let (x, y) = result.unwrap();
        assert_eq!(x.shape(), &[150, 4]);
        assert_eq!(y.len(), 150);

        // Check that all target values are in valid range
        for &target in y.iter() {
            assert!(target >= 0.0 && target < 3.0);
        }
    }

    #[test]
    fn test_edge_case_data() {
        use generators::EdgeCase;

        let (x, y) = generators::make_edge_case_data(EdgeCase::SingleSample).unwrap();
        assert_eq!(x.shape(), &[1, 3]);
        assert_eq!(y.len(), 1);

        let (x, y) = generators::make_edge_case_data(EdgeCase::SingleFeature).unwrap();
        assert_eq!(x.shape(), &[5, 1]);
        assert_eq!(y.len(), 5);
    }

    #[test]
    fn test_array_assertions() {
        let a = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let b = Array2::from_shape_vec((2, 2), vec![1.1, 2.1, 3.1, 4.1]).unwrap();

        assert!(assertions::assert_arrays_close(&a, &b, 0.2).is_ok());
        assert!(assertions::assert_arrays_close(&a, &b, 0.05).is_err());
    }

    #[test]
    fn test_probability_assertions() {
        let probs = Array2::from_shape_vec((2, 3), vec![0.3, 0.4, 0.3, 0.2, 0.5, 0.3]).unwrap();
        assert!(assertions::assert_probabilities_valid(&probs, 1e-6).is_ok());

        let invalid_probs =
            Array2::from_shape_vec((2, 3), vec![0.3, 0.4, 0.4, 0.2, 0.5, 0.3]).unwrap();
        assert!(assertions::assert_probabilities_valid(&invalid_probs, 1e-6).is_err());
    }

    #[test]
    fn test_performance_measurement() {
        let (result, duration) = performance::measure_time(|| {
            std::thread::sleep(std::time::Duration::from_millis(10));
            42
        });

        assert_eq!(result, 42);
        assert!(duration >= std::time::Duration::from_millis(10));
    }

    #[test]
    fn test_benchmark_comparison() {
        let result = performance::benchmark_algorithms(
            "fast",
            || std::thread::sleep(std::time::Duration::from_millis(1)),
            "slow",
            || std::thread::sleep(std::time::Duration::from_millis(10)),
        );

        assert_eq!(result.faster_algorithm().name, "fast");
        assert!(result.speedup_factor() > 1.0);
    }

    #[test]
    fn test_fixtures() {
        let (x, y) = &*fixtures::IRIS_DATASET;
        assert_eq!(x.shape(), &[6, 4]);
        assert_eq!(y.len(), 6);

        let (x, y) = &*fixtures::BOSTON_HOUSING;
        assert_eq!(x.shape(), &[100, 5]);
        assert_eq!(y.len(), 100);
    }

    #[test]
    fn test_mock_models() {
        let mock = mocks::MockLinearModel::<Array2<Float>, Array1<Float>>::new();
        assert!(mock.coefficients.is_none());
        assert_eq!(mock.intercept, 0.0);

        let constant_predictor = mocks::ConstantPredictor::new(42.0);
        assert_eq!(constant_predictor.value, 42.0);
    }
}
