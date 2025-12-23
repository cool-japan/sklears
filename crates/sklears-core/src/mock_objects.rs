/// Mock objects for testing complex machine learning interactions
///
/// This module provides comprehensive mock implementations of machine learning
/// components to enable sophisticated testing scenarios, particularly for:
///
/// - Integration testing between multiple ML components
/// - Behavior verification in ensemble methods
/// - Error condition simulation and recovery testing
/// - Performance benchmarking with controlled behavior
/// - Pipeline testing with predictable components
///
/// # Key Features
///
/// - Configurable mock estimators with predictable behavior
/// - Controllable failure modes for error testing
/// - Performance simulation for benchmarking
/// - State tracking for behavior verification
/// - Builder pattern for easy mock configuration
///
/// # Examples
///
/// ```rust
/// use sklears_core::mock_objects::{MockEstimator, MockBehavior, MockConfig};
/// use sklears_core::traits::{Predict, Fit};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// // Create a mock classifier that always predicts class 1
/// let mock = MockEstimator::builder()
///     .with_behavior(MockBehavior::ConstantPrediction(1.0))
///     .with_fit_delay(std::time::Duration::from_millis(10))
///     .build();
///
/// // Use it like any other estimator
/// let features = ndarray::Array2::zeros((100, 10));
/// let targets = ndarray::Array1::zeros(100);
///
/// let trained = mock.fit(&features.view(), &targets.view())?;
/// let predictions = trained.predict(&features.view())?;
///
/// // All predictions should be 1.0
/// assert!(predictions.iter().all(|&p| p == 1.0));
/// # Ok(())
/// # }
/// ```
use crate::error::{Result, SklearsError};
use crate::traits::{Estimator, Fit, Predict, PredictProba, Score, Transform};
// SciRS2 Policy: Using scirs2_core::ndarray and scirs2_core::random (COMPLIANT)
use scirs2_core::ndarray::{s, Array1, Array2, ArrayView1, ArrayView2};
use scirs2_core::random::Random;
use scirs2_core::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Mock estimator with configurable behavior for testing
#[derive(Debug, Clone)]
pub struct MockEstimator {
    config: MockConfig,
    state: Arc<Mutex<MockState>>,
}

/// Configuration for mock estimator behavior
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MockConfig {
    /// Behavior pattern for predictions
    pub behavior: MockBehavior,
    /// Artificial delay during fit operation
    pub fit_delay: Duration,
    /// Artificial delay during predict operation
    pub predict_delay: Duration,
    /// Whether to simulate fit failures
    pub fit_failure_probability: f64,
    /// Whether to simulate predict failures
    pub predict_failure_probability: f64,
    /// Maximum number of fit calls before failure
    pub max_fit_calls: Option<usize>,
    /// Random seed for reproducible behavior
    pub random_seed: u64,
}

impl Default for MockConfig {
    fn default() -> Self {
        Self {
            behavior: MockBehavior::ConstantPrediction(0.0),
            fit_delay: Duration::from_millis(0),
            predict_delay: Duration::from_millis(0),
            fit_failure_probability: 0.0,
            predict_failure_probability: 0.0,
            max_fit_calls: None,
            random_seed: 42,
        }
    }
}

/// Different mock behavior patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MockBehavior {
    /// Always return the same prediction value
    ConstantPrediction(f64),
    /// Return predictions based on feature sum
    FeatureSum,
    /// Return random predictions (using seed)
    Random { min: f64, max: f64 },
    /// Return predictions based on a simple linear model
    LinearModel { weights: Vec<f64>, bias: f64 },
    /// Return values from a predefined sequence
    Sequence(Vec<f64>),
    /// Mirror the target values during training
    MirrorTargets,
    /// Always predict the class with highest frequency in training
    MajorityClass,
    /// Simulate overfitting by perfect training accuracy, poor test accuracy
    Overfitting {
        train_accuracy: f64,
        test_accuracy: f64,
    },
}

/// Internal state tracking for mock estimator
#[derive(Debug, Default)]
struct MockState {
    fit_count: usize,
    predict_count: usize,
    last_fit_time: Option<Instant>,
    last_predict_time: Option<Instant>,
    training_targets: Option<Array1<f64>>,
    fitted: bool,
    fit_call_history: Vec<Instant>,
    predict_call_history: Vec<Instant>,
    performance_metrics: HashMap<String, f64>,
}

impl MockEstimator {
    /// Create a new mock estimator with default configuration
    pub fn new() -> Self {
        Self::with_config(MockConfig::default())
    }

    /// Create a mock estimator with custom configuration
    pub fn with_config(config: MockConfig) -> Self {
        Self {
            config,
            state: Arc::new(Mutex::new(MockState::default())),
        }
    }

    /// Create a builder for configuring mock estimator
    pub fn builder() -> MockEstimatorBuilder {
        MockEstimatorBuilder::new()
    }

    /// Get the current configuration
    pub fn config(&self) -> &MockConfig {
        &self.config
    }

    /// Get mock state information for testing
    pub fn mock_state(&self) -> MockStateSnapshot {
        let state = self.state.lock().unwrap();
        MockStateSnapshot {
            fit_count: state.fit_count,
            predict_count: state.predict_count,
            fitted: state.fitted,
            fit_call_history: state.fit_call_history.clone(),
            predict_call_history: state.predict_call_history.clone(),
            performance_metrics: state.performance_metrics.clone(),
        }
    }

    /// Reset the mock state (useful for test setup)
    pub fn reset_state(&self) {
        let mut state = self.state.lock().unwrap();
        *state = MockState::default();
    }

    /// Simulate a specific error condition
    pub fn simulate_error(&self, error_type: MockErrorType) -> Result<()> {
        match error_type {
            MockErrorType::FitFailure => {
                Err(SklearsError::FitError("Simulated fit failure".to_string()))
            }
            MockErrorType::PredictFailure => Err(SklearsError::PredictError(
                "Simulated predict failure".to_string(),
            )),
            MockErrorType::InvalidInput => Err(SklearsError::InvalidInput(
                "Simulated invalid input".to_string(),
            )),
            MockErrorType::NotFitted => Err(SklearsError::NotFitted {
                operation: "predict".to_string(),
            }),
        }
    }
}

impl Default for MockEstimator {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for configuring mock estimators
#[derive(Debug)]
pub struct MockEstimatorBuilder {
    config: MockConfig,
}

impl MockEstimatorBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            config: MockConfig::default(),
        }
    }

    /// Set the behavior pattern
    pub fn with_behavior(mut self, behavior: MockBehavior) -> Self {
        self.config.behavior = behavior;
        self
    }

    /// Set the fit delay
    pub fn with_fit_delay(mut self, delay: Duration) -> Self {
        self.config.fit_delay = delay;
        self
    }

    /// Set the predict delay
    pub fn with_predict_delay(mut self, delay: Duration) -> Self {
        self.config.predict_delay = delay;
        self
    }

    /// Set fit failure probability
    pub fn with_fit_failure_probability(mut self, probability: f64) -> Self {
        self.config.fit_failure_probability = probability.clamp(0.0, 1.0);
        self
    }

    /// Set predict failure probability
    pub fn with_predict_failure_probability(mut self, probability: f64) -> Self {
        self.config.predict_failure_probability = probability.clamp(0.0, 1.0);
        self
    }

    /// Set maximum number of fit calls before failure
    pub fn with_max_fit_calls(mut self, max_calls: usize) -> Self {
        self.config.max_fit_calls = Some(max_calls);
        self
    }

    /// Set random seed for reproducible behavior
    pub fn with_random_seed(mut self, seed: u64) -> Self {
        self.config.random_seed = seed;
        self
    }

    /// Build the mock estimator
    pub fn build(self) -> MockEstimator {
        MockEstimator::with_config(self.config)
    }
}

impl Default for MockEstimatorBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Snapshot of mock state for testing
#[derive(Debug, Clone)]
pub struct MockStateSnapshot {
    pub fit_count: usize,
    pub predict_count: usize,
    pub fitted: bool,
    pub fit_call_history: Vec<Instant>,
    pub predict_call_history: Vec<Instant>,
    pub performance_metrics: HashMap<String, f64>,
}

/// Types of errors that can be simulated
#[derive(Debug, Clone, Copy)]
pub enum MockErrorType {
    FitFailure,
    PredictFailure,
    InvalidInput,
    NotFitted,
}

/// Trained mock estimator
#[derive(Debug, Clone)]
pub struct TrainedMockEstimator {
    estimator: MockEstimator,
    training_data_shape: (usize, usize),
}

impl Estimator for MockEstimator {
    type Config = MockConfig;
    type Error = crate::error::SklearsError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl<'a> Fit<ArrayView2<'a, f64>, ArrayView1<'a, f64>> for MockEstimator {
    type Fitted = TrainedMockEstimator;

    fn fit(self, x: &ArrayView2<'a, f64>, y: &ArrayView1<'a, f64>) -> Result<Self::Fitted> {
        let mut state = self.state.lock().unwrap();

        // Track fit call
        state.fit_count += 1;
        state.last_fit_time = Some(Instant::now());
        state.fit_call_history.push(Instant::now());

        // Check for max fit calls limit
        if let Some(max_calls) = self.config.max_fit_calls {
            if state.fit_count > max_calls {
                return Err(SklearsError::FitError(format!(
                    "Maximum fit calls ({max_calls}) exceeded"
                )));
            }
        }

        // Simulate fit failure probability
        if self.config.fit_failure_probability > 0.0 {
            let mut rng = Random::seed(self.config.random_seed + state.fit_count as u64);
            if rng.gen::<f64>() < self.config.fit_failure_probability {
                return Err(SklearsError::FitError(
                    "Simulated random fit failure".to_string(),
                ));
            }
        }

        // Validate input dimensions
        if x.nrows() != y.len() {
            return Err(SklearsError::ShapeMismatch {
                expected: format!("({}, n_features)", y.len()),
                actual: format!("({}, {})", x.nrows(), x.ncols()),
            });
        }

        // Store training targets for certain behaviors
        match self.config.behavior {
            MockBehavior::MirrorTargets | MockBehavior::MajorityClass => {
                state.training_targets = Some(y.to_owned());
            }
            _ => {}
        }

        // Simulate fit delay
        if !self.config.fit_delay.is_zero() {
            std::thread::sleep(self.config.fit_delay);
        }

        state.fitted = true;
        drop(state); // Release lock before creating output

        Ok(TrainedMockEstimator {
            estimator: self.clone(),
            training_data_shape: (x.nrows(), x.ncols()),
        })
    }
}

impl<'a> Predict<ArrayView2<'a, f64>, Array1<f64>> for TrainedMockEstimator {
    fn predict(&self, x: &ArrayView2<'a, f64>) -> Result<Array1<f64>> {
        let mut state = self.estimator.state.lock().unwrap();

        // Track predict call
        state.predict_count += 1;
        state.last_predict_time = Some(Instant::now());
        state.predict_call_history.push(Instant::now());

        // Simulate predict failure probability
        if self.estimator.config.predict_failure_probability > 0.0 {
            let mut rng =
                Random::seed(self.estimator.config.random_seed + state.predict_count as u64);
            if rng.gen::<f64>() < self.estimator.config.predict_failure_probability {
                return Err(SklearsError::PredictError(
                    "Simulated random predict failure".to_string(),
                ));
            }
        }

        // Validate input dimensions
        if x.ncols() != self.training_data_shape.1 {
            return Err(SklearsError::FeatureMismatch {
                expected: self.training_data_shape.1,
                actual: x.ncols(),
            });
        }

        // Simulate predict delay
        if !self.estimator.config.predict_delay.is_zero() {
            std::thread::sleep(self.estimator.config.predict_delay);
        }

        // Generate predictions based on behavior
        let predictions = match &self.estimator.config.behavior {
            MockBehavior::ConstantPrediction(value) => Array1::from_elem(x.nrows(), *value),
            MockBehavior::FeatureSum => {
                Array1::from_iter(x.rows().into_iter().map(|row| row.sum()))
            }
            MockBehavior::Random { min, max } => {
                let mut rng = Random::seed(self.estimator.config.random_seed);
                Array1::from_iter((0..x.nrows()).map(|_| rng.gen_range(*min..*max)))
            }
            MockBehavior::LinearModel { weights, bias } => {
                if weights.len() != x.ncols() {
                    return Err(SklearsError::InvalidInput(
                        "Weight dimension mismatch".to_string(),
                    ));
                }
                Array1::from_iter(x.rows().into_iter().map(|row| {
                    let dot_product: f64 = row.iter().zip(weights.iter()).map(|(x, w)| x * w).sum();
                    dot_product + bias
                }))
            }
            MockBehavior::Sequence(values) => {
                Array1::from_iter((0..x.nrows()).map(|i| values[i % values.len()]))
            }
            MockBehavior::MirrorTargets => {
                if let Some(ref targets) = state.training_targets {
                    // Return targets corresponding to input indices (simplified)
                    Array1::from_iter((0..x.nrows()).map(|i| targets[i % targets.len()]))
                } else {
                    Array1::zeros(x.nrows())
                }
            }
            MockBehavior::MajorityClass => {
                if let Some(ref targets) = state.training_targets {
                    // Find most common class
                    let mut counts = HashMap::new();
                    for &target in targets {
                        *counts.entry(target as i32).or_insert(0) += 1;
                    }
                    let majority_class = counts
                        .into_iter()
                        .max_by_key(|(_, count)| *count)
                        .map(|(class, _)| class as f64)
                        .unwrap_or(0.0);
                    Array1::from_elem(x.nrows(), majority_class)
                } else {
                    Array1::zeros(x.nrows())
                }
            }
            MockBehavior::Overfitting {
                train_accuracy: _,
                test_accuracy,
            } => {
                // Simulate poor generalization
                let mut rng = Random::seed(self.estimator.config.random_seed);
                Array1::from_iter((0..x.nrows()).map(|_| {
                    if rng.gen::<f64>() < *test_accuracy {
                        1.0 // Correct prediction
                    } else {
                        0.0 // Incorrect prediction
                    }
                }))
            }
        };

        Ok(predictions)
    }
}

impl<'a> PredictProba<ArrayView2<'a, f64>, Array2<f64>> for TrainedMockEstimator {
    fn predict_proba(&self, x: &ArrayView2<'a, f64>) -> Result<Array2<f64>> {
        // Convert predictions to probabilities (simplified for binary classification)
        let predictions = self.predict(x)?;
        let mut probabilities = Array2::zeros((x.nrows(), 2));

        for (i, &pred) in predictions.iter().enumerate() {
            let prob_positive = (pred.tanh() + 1.0) / 2.0; // Map to [0, 1]
            probabilities[[i, 0]] = 1.0 - prob_positive;
            probabilities[[i, 1]] = prob_positive;
        }

        Ok(probabilities)
    }
}

impl<'a> Score<ArrayView2<'a, f64>, ArrayView1<'a, f64>> for TrainedMockEstimator {
    type Float = f64;
    fn score(&self, x: &ArrayView2<'a, f64>, y: &ArrayView1<'a, f64>) -> Result<f64> {
        let predictions = self.predict(x)?;

        // Calculate R² score for regression or accuracy for classification
        match &self.estimator.config.behavior {
            MockBehavior::Overfitting {
                train_accuracy,
                test_accuracy: _,
            } => {
                // Return perfect score for training data simulation
                Ok(*train_accuracy)
            }
            _ => {
                // Simple accuracy calculation (assuming classification)
                let correct = predictions
                    .iter()
                    .zip(y.iter())
                    .map(|(pred, actual)| {
                        if (pred - actual).abs() < 0.5 {
                            1.0
                        } else {
                            0.0
                        }
                    })
                    .sum::<f64>();
                Ok(correct / y.len() as f64)
            }
        }
    }
}

/// Mock transformer for testing transformation pipelines
#[derive(Debug, Clone)]
pub struct MockTransformer {
    config: MockTransformConfig,
    fitted: bool,
    input_shape: Option<(usize, usize)>,
}

/// Configuration for mock transformer
#[derive(Debug, Clone)]
pub struct MockTransformConfig {
    pub transform_type: MockTransformType,
    pub output_features: Option<usize>,
    pub transform_delay: Duration,
}

/// Types of transformations to simulate
#[derive(Debug, Clone)]
pub enum MockTransformType {
    /// Identity transformation (no change)
    Identity,
    /// Scale all values by a constant
    Scale(f64),
    /// Add constant to all values
    Shift(f64),
    /// Reduce feature dimensions
    FeatureReduction { keep_ratio: f64 },
    /// Expand feature dimensions
    FeatureExpansion { expansion_factor: usize },
    /// Simulate standardization (mean=0, std=1)
    Standardization,
}

impl MockTransformer {
    /// Create a new mock transformer
    pub fn new(transform_type: MockTransformType) -> Self {
        Self {
            config: MockTransformConfig {
                transform_type,
                output_features: None,
                transform_delay: Duration::from_millis(0),
            },
            fitted: false,
            input_shape: None,
        }
    }

    /// Create a mock transformer builder
    pub fn builder() -> MockTransformerBuilder {
        MockTransformerBuilder::new()
    }
}

/// Builder for mock transformers
#[derive(Debug)]
pub struct MockTransformerBuilder {
    transform_type: MockTransformType,
    output_features: Option<usize>,
    transform_delay: Duration,
}

impl MockTransformerBuilder {
    pub fn new() -> Self {
        Self {
            transform_type: MockTransformType::Identity,
            output_features: None,
            transform_delay: Duration::from_millis(0),
        }
    }

    pub fn with_transform_type(mut self, transform_type: MockTransformType) -> Self {
        self.transform_type = transform_type;
        self
    }

    pub fn with_output_features(mut self, features: usize) -> Self {
        self.output_features = Some(features);
        self
    }

    pub fn with_transform_delay(mut self, delay: Duration) -> Self {
        self.transform_delay = delay;
        self
    }

    pub fn build(self) -> MockTransformer {
        MockTransformer {
            config: MockTransformConfig {
                transform_type: self.transform_type,
                output_features: self.output_features,
                transform_delay: self.transform_delay,
            },
            fitted: false,
            input_shape: None,
        }
    }
}

impl Default for MockTransformerBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a> Fit<ArrayView2<'a, f64>, ArrayView1<'a, f64>> for MockTransformer {
    type Fitted = MockTransformer;

    fn fit(self, x: &ArrayView2<'a, f64>, _y: &ArrayView1<'a, f64>) -> Result<Self::Fitted> {
        let mut fitted = self.clone();
        fitted.fitted = true;
        fitted.input_shape = Some((x.nrows(), x.ncols()));
        Ok(fitted)
    }
}

impl<'a> Transform<ArrayView2<'a, f64>, Array2<f64>> for MockTransformer {
    fn transform(&self, x: &ArrayView2<'a, f64>) -> Result<Array2<f64>> {
        if !self.fitted {
            return Err(SklearsError::NotFitted {
                operation: "transform".to_string(),
            });
        }

        // Simulate transform delay
        if !self.config.transform_delay.is_zero() {
            std::thread::sleep(self.config.transform_delay);
        }

        match &self.config.transform_type {
            MockTransformType::Identity => Ok(x.to_owned()),
            MockTransformType::Scale(factor) => Ok(x * *factor),
            MockTransformType::Shift(offset) => Ok(x + *offset),
            MockTransformType::FeatureReduction { keep_ratio } => {
                let keep_features = ((x.ncols() as f64) * keep_ratio).ceil() as usize;
                let keep_features = keep_features.max(1).min(x.ncols());
                Ok(x.slice(s![.., 0..keep_features]).to_owned())
            }
            MockTransformType::FeatureExpansion { expansion_factor } => {
                let new_features = x.ncols() * expansion_factor;
                let mut expanded = Array2::zeros((x.nrows(), new_features));

                // Tile the original features
                for i in 0..*expansion_factor {
                    let start_col = i * x.ncols();
                    let end_col = start_col + x.ncols();
                    expanded.slice_mut(s![.., start_col..end_col]).assign(x);
                }
                Ok(expanded)
            }
            MockTransformType::Standardization => {
                // Simple standardization simulation
                let mean = x.mean().unwrap_or(0.0);
                let std = x.std(0.0);
                if std == 0.0 {
                    Ok(x - mean)
                } else {
                    Ok((x - mean) / std)
                }
            }
        }
    }
}

/// Mock ensemble for testing ensemble methods
#[derive(Debug)]
#[allow(dead_code)]
pub struct MockEnsemble {
    estimators: Vec<MockEstimator>,
    voting_strategy: VotingStrategy,
    fitted: bool,
}

/// Voting strategies for mock ensemble
#[derive(Debug, Clone)]
pub enum VotingStrategy {
    MajorityVote,
    AverageVote,
    WeightedVote(Vec<f64>),
}

impl MockEnsemble {
    /// Create a new mock ensemble
    pub fn new(estimators: Vec<MockEstimator>, voting_strategy: VotingStrategy) -> Self {
        Self {
            estimators,
            voting_strategy,
            fitted: false,
        }
    }

    /// Get the number of base estimators
    pub fn n_estimators(&self) -> usize {
        self.estimators.len()
    }

    /// Get voting strategy
    pub fn voting_strategy(&self) -> &VotingStrategy {
        &self.voting_strategy
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_mock_estimator_constant_prediction() {
        let mock = MockEstimator::builder()
            .with_behavior(MockBehavior::ConstantPrediction(5.0))
            .build();

        let features = Array2::zeros((10, 3));
        let targets = Array1::zeros(10);

        let trained = mock.clone().fit(&features.view(), &targets.view()).unwrap();
        let predictions = trained.predict(&features.view()).unwrap();

        assert_eq!(predictions.len(), 10);
        assert!(predictions.iter().all(|&p| p == 5.0));
    }

    #[test]
    fn test_mock_estimator_state_tracking() {
        let mock = MockEstimator::new();
        let features = Array2::zeros((5, 2));
        let targets = Array1::zeros(5);

        // Initial state
        let state = mock.mock_state();
        assert_eq!(state.fit_count, 0);
        assert_eq!(state.predict_count, 0);
        assert!(!state.fitted);

        // After fitting
        let trained = mock.clone().fit(&features.view(), &targets.view()).unwrap();
        let state = mock.mock_state();
        assert_eq!(state.fit_count, 1);
        assert!(state.fitted);

        // After predicting
        let _ = trained.predict(&features.view()).unwrap();
        let state = mock.mock_state();
        assert_eq!(state.predict_count, 1);
    }

    #[test]
    fn test_mock_estimator_feature_sum() {
        let mock = MockEstimator::builder()
            .with_behavior(MockBehavior::FeatureSum)
            .build();

        let features = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let targets = Array1::zeros(2);

        let trained = mock.clone().fit(&features.view(), &targets.view()).unwrap();
        let predictions = trained.predict(&features.view()).unwrap();

        assert_eq!(predictions[0], 6.0); // 1 + 2 + 3
        assert_eq!(predictions[1], 15.0); // 4 + 5 + 6
    }

    #[test]
    fn test_mock_estimator_linear_model() {
        let weights = vec![1.0, 2.0, 3.0];
        let bias = 1.0;

        let mock = MockEstimator::builder()
            .with_behavior(MockBehavior::LinearModel { weights, bias })
            .build();

        let features = Array2::from_shape_vec((1, 3), vec![1.0, 1.0, 1.0]).unwrap();
        let targets = Array1::zeros(1);

        let trained = mock.fit(&features.view(), &targets.view()).unwrap();
        let predictions = trained.predict(&features.view()).unwrap();

        assert_eq!(predictions[0], 7.0); // 1*1 + 2*1 + 3*1 + 1
    }

    #[test]
    fn test_mock_transformer_identity() {
        let transformer = MockTransformer::new(MockTransformType::Identity);
        let data = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let targets = Array1::zeros(2);

        let fitted = transformer
            .clone()
            .fit(&data.view(), &targets.view())
            .unwrap();
        let transformed = fitted.transform(&data.view()).unwrap();

        assert_eq!(transformed, data);
    }

    #[test]
    fn test_mock_transformer_scale() {
        let transformer = MockTransformer::new(MockTransformType::Scale(2.0));
        let data = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let targets = Array1::zeros(2);

        let fitted = transformer
            .clone()
            .fit(&data.view(), &targets.view())
            .unwrap();
        let transformed = fitted.transform(&data.view()).unwrap();

        let expected = Array2::from_shape_vec((2, 2), vec![2.0, 4.0, 6.0, 8.0]).unwrap();
        assert_eq!(transformed, expected);
    }

    #[test]
    fn test_mock_estimator_failure_simulation() {
        let mock = MockEstimator::builder()
            .with_fit_failure_probability(1.0) // Always fail
            .build();

        let features = Array2::zeros((5, 2));
        let targets = Array1::zeros(5);

        let result = mock.clone().fit(&features.view(), &targets.view());
        assert!(result.is_err());
    }

    #[test]
    fn test_mock_estimator_max_fit_calls() {
        let mock = MockEstimator::builder().with_max_fit_calls(2).build();

        let features = Array2::zeros((5, 2));
        let targets = Array1::zeros(5);

        // First two fits should succeed
        assert!(mock.clone().fit(&features.view(), &targets.view()).is_ok());
        assert!(mock.clone().fit(&features.view(), &targets.view()).is_ok());

        // Third fit should fail
        assert!(mock.clone().fit(&features.view(), &targets.view()).is_err());
    }

    #[test]
    fn test_mock_estimator_predict_proba() {
        let mock = MockEstimator::builder()
            .with_behavior(MockBehavior::ConstantPrediction(0.0))
            .build();

        let features = Array2::zeros((3, 2));
        let targets = Array1::zeros(3);

        let trained = mock.clone().fit(&features.view(), &targets.view()).unwrap();
        let probabilities = trained.predict_proba(&features.view()).unwrap();

        assert_eq!(probabilities.shape(), &[3, 2]);
        // All predictions should have probabilities that sum to 1
        for row in probabilities.rows() {
            let sum: f64 = row.sum();
            assert!((sum - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_mock_ensemble_creation() {
        let est1 = MockEstimator::builder()
            .with_behavior(MockBehavior::ConstantPrediction(1.0))
            .build();
        let est2 = MockEstimator::builder()
            .with_behavior(MockBehavior::ConstantPrediction(2.0))
            .build();

        let ensemble = MockEnsemble::new(vec![est1, est2], VotingStrategy::AverageVote);

        assert_eq!(ensemble.n_estimators(), 2);
        assert!(matches!(
            ensemble.voting_strategy(),
            VotingStrategy::AverageVote
        ));
    }
}
