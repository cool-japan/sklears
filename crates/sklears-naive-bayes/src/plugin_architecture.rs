//! Plugin architecture for extensible Naive Bayes framework
//!
//! This module provides a trait-based plugin system that allows for:
//! - Pluggable probability distributions
//! - Composable smoothing methods
//! - Extensible parameter estimation
//! - Flexible model selection system
//! - Custom middleware for prediction pipelines

// SciRS2 Policy Compliance - Use scirs2-autograd for ndarray types
use scirs2_core::ndarray::{Array1, Array2};
use std::collections::HashMap;
use std::fmt::Debug;
use std::sync::Arc;

use sklears_core::{error::Result, prelude::SklearsError, types::Float};

/// Trait for probability distributions that can be plugged into Naive Bayes models
pub trait PluggableDistribution: Debug + Send + Sync {
    /// The name/identifier of this distribution
    fn name(&self) -> &'static str;

    /// Estimate parameters from training data
    fn fit(&mut self, x: &Array1<Float>) -> Result<()>;

    /// Compute log probability density/mass function
    fn log_prob(&self, x: Float) -> Float;

    /// Compute multiple log probabilities efficiently
    fn log_prob_batch(&self, x: &Array1<Float>) -> Array1<Float> {
        x.map(|&val| self.log_prob(val))
    }

    /// Get the number of parameters
    fn n_parameters(&self) -> usize;

    /// Get parameter values as a vector
    fn parameters(&self) -> Vec<Float>;

    /// Set parameter values from a vector
    fn set_parameters(&mut self, params: &[Float]) -> Result<()>;

    /// Check if the distribution supports the given data type
    fn supports_data_type(&self, data_type: &DataType) -> bool;

    /// Validate input data for this distribution
    fn validate_data(&self, x: &Array1<Float>) -> Result<()>;

    /// Clone the distribution as a trait object
    fn clone_box(&self) -> Box<dyn PluggableDistribution>;
}

/// Trait for composable smoothing methods
pub trait ComposableSmoothingMethod: Debug + Send + Sync {
    /// The name/identifier of this smoothing method
    fn name(&self) -> &'static str;

    /// Apply smoothing to counts
    fn smooth(&self, counts: &Array1<Float>, alpha: Float) -> Array1<Float>;

    /// Apply smoothing to a matrix of counts
    fn smooth_matrix(&self, counts: &Array2<Float>, alpha: Float) -> Array2<Float> {
        let mut result = counts.clone();
        for mut row in result.rows_mut() {
            let smoothed = self.smooth(&row.to_owned(), alpha);
            row.assign(&smoothed);
        }
        result
    }

    /// Get the number of hyperparameters
    fn n_hyperparameters(&self) -> usize;

    /// Get hyperparameter values
    fn hyperparameters(&self) -> Vec<Float>;

    /// Set hyperparameter values
    fn set_hyperparameters(&mut self, params: &[Float]) -> Result<()>;

    /// Check if this method is applicable to the given data characteristics
    fn is_applicable(&self, data_characteristics: &DataCharacteristics) -> bool;

    /// Clone the smoothing method as a trait object
    fn clone_box(&self) -> Box<dyn ComposableSmoothingMethod>;
}

/// Trait for extensible parameter estimation methods
pub trait ExtensibleParameterEstimator: Debug + Send + Sync {
    /// The name/identifier of this estimator
    fn name(&self) -> &'static str;

    /// Estimate parameters using this method
    fn estimate(&self, x: &Array2<Float>, y: &Array1<i32>) -> Result<EstimationResult>;

    /// Check if this estimator supports the given configuration
    fn supports_configuration(&self, config: &EstimatorConfiguration) -> bool;

    /// Get the computational complexity of this estimator
    fn complexity(&self) -> ComputationalComplexity;

    /// Clone the estimator as a trait object
    fn clone_box(&self) -> Box<dyn ExtensibleParameterEstimator>;
}

/// Trait for flexible model selection systems
pub trait FlexibleModelSelector: Debug + Send + Sync {
    /// The name/identifier of this selector
    fn name(&self) -> &'static str;

    /// Select the best model configuration
    fn select_model(&self, candidates: &[ModelCandidate]) -> Result<ModelCandidate>;

    /// Evaluate a single model configuration
    fn evaluate_model(&self, candidate: &ModelCandidate) -> Result<Float>;

    /// Get selection criteria used by this selector
    fn selection_criteria(&self) -> Vec<SelectionCriterion>;

    /// Clone the selector as a trait object
    fn clone_box(&self) -> Box<dyn FlexibleModelSelector>;
}

/// Trait for middleware components in prediction pipelines
pub trait PredictionMiddleware: Debug + Send + Sync {
    /// The name/identifier of this middleware
    fn name(&self) -> &'static str;

    /// Process input before prediction
    fn before_predict(&self, x: &Array2<Float>) -> Result<Array2<Float>>;

    /// Process output after prediction
    fn after_predict(&self, predictions: &Array1<i32>) -> Result<Array1<i32>>;

    /// Process probabilistic output after prediction
    fn after_predict_proba(&self, probabilities: &Array2<Float>) -> Result<Array2<Float>>;

    /// Check if this middleware should be applied
    fn should_apply(&self, context: &PredictionContext) -> bool;

    /// Get the execution order priority (lower = earlier)
    fn priority(&self) -> i32;
}

/// Registry for managing pluggable components
#[derive(Debug, Default)]
pub struct PluginRegistry {
    distributions: HashMap<String, Box<dyn PluggableDistribution>>,
    smoothing_methods: HashMap<String, Box<dyn ComposableSmoothingMethod>>,
    parameter_estimators: HashMap<String, Box<dyn ExtensibleParameterEstimator>>,
    model_selectors: HashMap<String, Box<dyn FlexibleModelSelector>>,
    middleware: Vec<Arc<dyn PredictionMiddleware>>,
}

impl PluginRegistry {
    /// Create a new plugin registry
    pub fn new() -> Self {
        let mut registry = Self::default();
        registry.register_builtin_components();
        registry
    }

    /// Register a distribution plugin
    pub fn register_distribution(&mut self, distribution: Box<dyn PluggableDistribution>) {
        let name = distribution.name().to_string();
        self.distributions.insert(name, distribution);
    }

    /// Register a smoothing method plugin
    pub fn register_smoothing_method(&mut self, method: Box<dyn ComposableSmoothingMethod>) {
        let name = method.name().to_string();
        self.smoothing_methods.insert(name, method);
    }

    /// Register a parameter estimator plugin
    pub fn register_parameter_estimator(
        &mut self,
        estimator: Box<dyn ExtensibleParameterEstimator>,
    ) {
        let name = estimator.name().to_string();
        self.parameter_estimators.insert(name, estimator);
    }

    /// Register a model selector plugin
    pub fn register_model_selector(&mut self, selector: Box<dyn FlexibleModelSelector>) {
        let name = selector.name().to_string();
        self.model_selectors.insert(name, selector);
    }

    /// Register middleware
    pub fn register_middleware(&mut self, middleware: Arc<dyn PredictionMiddleware>) {
        self.middleware.push(middleware);
        // Sort by priority
        self.middleware.sort_by_key(|m| m.priority());
    }

    /// Get a distribution by name
    pub fn get_distribution(&self, name: &str) -> Option<Box<dyn PluggableDistribution>> {
        self.distributions.get(name).map(|d| d.clone_box())
    }

    /// Get a smoothing method by name
    pub fn get_smoothing_method(&self, name: &str) -> Option<Box<dyn ComposableSmoothingMethod>> {
        self.smoothing_methods.get(name).map(|s| s.clone_box())
    }

    /// Get a parameter estimator by name
    pub fn get_parameter_estimator(
        &self,
        name: &str,
    ) -> Option<Box<dyn ExtensibleParameterEstimator>> {
        self.parameter_estimators.get(name).map(|e| e.clone_box())
    }

    /// Get a model selector by name
    pub fn get_model_selector(&self, name: &str) -> Option<Box<dyn FlexibleModelSelector>> {
        self.model_selectors.get(name).map(|s| s.clone_box())
    }

    /// Get all middleware components
    pub fn get_middleware(&self) -> Vec<Arc<dyn PredictionMiddleware>> {
        self.middleware.clone()
    }

    /// List available distributions
    pub fn list_distributions(&self) -> Vec<String> {
        self.distributions.keys().cloned().collect()
    }

    /// List available smoothing methods
    pub fn list_smoothing_methods(&self) -> Vec<String> {
        self.smoothing_methods.keys().cloned().collect()
    }

    /// List available parameter estimators
    pub fn list_parameter_estimators(&self) -> Vec<String> {
        self.parameter_estimators.keys().cloned().collect()
    }

    /// List available model selectors
    pub fn list_model_selectors(&self) -> Vec<String> {
        self.model_selectors.keys().cloned().collect()
    }

    /// Register built-in components
    fn register_builtin_components(&mut self) {
        // Register built-in distributions
        self.register_distribution(Box::new(BuiltinGaussianDistribution::default()));
        self.register_distribution(Box::new(BuiltinMultinomialDistribution::default()));
        self.register_distribution(Box::new(BuiltinBernoulliDistribution::default()));

        // Register built-in smoothing methods
        self.register_smoothing_method(Box::new(BuiltinLaplaceSmoothing));
        self.register_smoothing_method(Box::new(BuiltinLidstoneSmoothing::default()));

        // Register built-in parameter estimators
        self.register_parameter_estimator(Box::new(BuiltinMLEEstimator));
        self.register_parameter_estimator(Box::new(BuiltinMAPEstimator));

        // Register built-in model selectors
        self.register_model_selector(Box::new(BuiltinCrossValidationSelector));
        self.register_model_selector(Box::new(BuiltinInformationCriterionSelector));
    }
}

/// Supporting types and structures

#[derive(Debug, Clone, PartialEq)]
pub enum DataType {
    /// Continuous
    Continuous,
    /// Discrete
    Discrete,
    /// Binary
    Binary,
    /// Categorical
    Categorical,
    /// Count
    Count,
}

#[derive(Debug, Clone)]
pub struct DataCharacteristics {
    pub data_type: DataType,
    pub n_samples: usize,
    pub n_features: usize,
    pub sparsity: Float,
    pub has_missing_values: bool,
    pub is_balanced: bool,
}

#[derive(Debug, Clone)]
pub struct EstimationResult {
    pub parameters: HashMap<String, Vec<Float>>,
    pub log_likelihood: Float,
    pub aic: Option<Float>,
    pub bic: Option<Float>,
    pub convergence_info: ConvergenceInfo,
}

#[derive(Debug, Clone)]
pub struct ConvergenceInfo {
    pub converged: bool,
    pub n_iterations: usize,
    pub final_log_likelihood: Float,
    pub convergence_threshold: Float,
}

#[derive(Debug, Clone)]
pub struct EstimatorConfiguration {
    pub method: String,
    pub hyperparameters: HashMap<String, Float>,
    pub constraints: Vec<ParameterConstraint>,
}

#[derive(Debug, Clone)]
pub enum ComputationalComplexity {
    /// Constant
    Constant,
    /// Linear
    Linear,
    /// Quadratic
    Quadratic,
    /// Cubic
    Cubic,
    /// Exponential
    Exponential,
}

#[derive(Debug, Clone)]
pub struct ModelCandidate {
    pub model_type: String,
    pub configuration: HashMap<String, serde_json::Value>,
    pub score: Option<Float>,
    pub cross_validation_scores: Option<Vec<Float>>,
    pub training_time: Option<std::time::Duration>,
}

#[derive(Debug, Clone)]
pub enum SelectionCriterion {
    /// Accuracy
    Accuracy,
    /// LogLikelihood
    LogLikelihood,
    /// AIC
    AIC,
    /// BIC
    BIC,
    /// CrossValidationScore
    CrossValidationScore,
    /// TrainingTime
    TrainingTime,
    /// ModelComplexity
    ModelComplexity,
}

#[derive(Debug, Clone)]
pub struct PredictionContext {
    pub model_type: String,
    pub input_shape: (usize, usize),
    pub has_missing_values: bool,
    pub is_streaming: bool,
    pub require_probabilities: bool,
}

#[derive(Debug, Clone)]
pub enum ParameterConstraint {
    /// Positive
    Positive,
    /// NonNegative
    NonNegative,
    /// Normalized
    Normalized,
    /// Range
    Range(Float, Float),
    /// SumToOne
    SumToOne,
}

// Built-in implementations

#[derive(Debug, Clone, Default)]
pub struct BuiltinGaussianDistribution {
    mean: Float,
    variance: Float,
    fitted: bool,
}

impl PluggableDistribution for BuiltinGaussianDistribution {
    fn name(&self) -> &'static str {
        "gaussian"
    }

    fn fit(&mut self, x: &Array1<Float>) -> Result<()> {
        if x.is_empty() {
            return Err(SklearsError::InvalidInput("Empty input data".to_string()));
        }

        self.mean = x.mean().unwrap_or(0.0);
        self.variance = x.var(0.0);
        self.fitted = true;
        Ok(())
    }

    fn log_prob(&self, x: Float) -> Float {
        if !self.fitted {
            return Float::NEG_INFINITY;
        }

        let diff = x - self.mean;
        -0.5 * (diff * diff / self.variance
            + self.variance.ln()
            + (2.0 * std::f64::consts::PI).ln())
    }

    fn n_parameters(&self) -> usize {
        2
    }

    fn parameters(&self) -> Vec<Float> {
        vec![self.mean, self.variance]
    }

    fn set_parameters(&mut self, params: &[Float]) -> Result<()> {
        if params.len() != 2 {
            return Err(SklearsError::InvalidInput(
                "Expected 2 parameters".to_string(),
            ));
        }
        self.mean = params[0];
        self.variance = params[1];
        self.fitted = true;
        Ok(())
    }

    fn supports_data_type(&self, data_type: &DataType) -> bool {
        matches!(data_type, DataType::Continuous)
    }

    fn validate_data(&self, x: &Array1<Float>) -> Result<()> {
        if x.iter().any(|&val| val.is_nan() || val.is_infinite()) {
            return Err(SklearsError::InvalidInput(
                "Data contains NaN or infinite values".to_string(),
            ));
        }
        Ok(())
    }

    fn clone_box(&self) -> Box<dyn PluggableDistribution> {
        Box::new(self.clone())
    }
}

#[derive(Debug, Clone, Default)]
pub struct BuiltinMultinomialDistribution {
    probabilities: Vec<Float>,
    fitted: bool,
}

impl PluggableDistribution for BuiltinMultinomialDistribution {
    fn name(&self) -> &'static str {
        "multinomial"
    }

    fn fit(&mut self, x: &Array1<Float>) -> Result<()> {
        if x.is_empty() {
            return Err(SklearsError::InvalidInput("Empty input data".to_string()));
        }

        let total: Float = x.sum();
        if total <= 0.0 {
            return Err(SklearsError::InvalidInput(
                "Total count must be positive".to_string(),
            ));
        }

        self.probabilities = x.iter().map(|&count| count / total).collect();
        self.fitted = true;
        Ok(())
    }

    fn log_prob(&self, x: Float) -> Float {
        if !self.fitted || x < 0.0 || x >= self.probabilities.len() as Float {
            return Float::NEG_INFINITY;
        }

        let index = x as usize;
        self.probabilities[index].ln()
    }

    fn n_parameters(&self) -> usize {
        self.probabilities.len()
    }

    fn parameters(&self) -> Vec<Float> {
        self.probabilities.clone()
    }

    fn set_parameters(&mut self, params: &[Float]) -> Result<()> {
        let sum: Float = params.iter().sum();
        if (sum - 1.0).abs() > 1e-10 {
            return Err(SklearsError::InvalidInput(
                "Parameters must sum to 1".to_string(),
            ));
        }
        self.probabilities = params.to_vec();
        self.fitted = true;
        Ok(())
    }

    fn supports_data_type(&self, data_type: &DataType) -> bool {
        matches!(data_type, DataType::Discrete | DataType::Count)
    }

    fn validate_data(&self, x: &Array1<Float>) -> Result<()> {
        if x.iter().any(|&val| val < 0.0 || val.fract() != 0.0) {
            return Err(SklearsError::InvalidInput(
                "Data must be non-negative integers".to_string(),
            ));
        }
        Ok(())
    }

    fn clone_box(&self) -> Box<dyn PluggableDistribution> {
        Box::new(self.clone())
    }
}

#[derive(Debug, Clone, Default)]
pub struct BuiltinBernoulliDistribution {
    probability: Float,
    fitted: bool,
}

impl PluggableDistribution for BuiltinBernoulliDistribution {
    fn name(&self) -> &'static str {
        "bernoulli"
    }

    fn fit(&mut self, x: &Array1<Float>) -> Result<()> {
        if x.is_empty() {
            return Err(SklearsError::InvalidInput("Empty input data".to_string()));
        }

        self.probability = x.mean().unwrap_or(0.5);
        self.fitted = true;
        Ok(())
    }

    fn log_prob(&self, x: Float) -> Float {
        if !self.fitted {
            return Float::NEG_INFINITY;
        }

        if x == 1.0 {
            self.probability.ln()
        } else if x == 0.0 {
            (1.0 - self.probability).ln()
        } else {
            Float::NEG_INFINITY
        }
    }

    fn n_parameters(&self) -> usize {
        1
    }

    fn parameters(&self) -> Vec<Float> {
        vec![self.probability]
    }

    fn set_parameters(&mut self, params: &[Float]) -> Result<()> {
        if params.len() != 1 {
            return Err(SklearsError::InvalidInput(
                "Expected 1 parameter".to_string(),
            ));
        }
        if params[0] < 0.0 || params[0] > 1.0 {
            return Err(SklearsError::InvalidInput(
                "Probability must be between 0 and 1".to_string(),
            ));
        }
        self.probability = params[0];
        self.fitted = true;
        Ok(())
    }

    fn supports_data_type(&self, data_type: &DataType) -> bool {
        matches!(data_type, DataType::Binary)
    }

    fn validate_data(&self, x: &Array1<Float>) -> Result<()> {
        if x.iter().any(|&val| val != 0.0 && val != 1.0) {
            return Err(SklearsError::InvalidInput(
                "Data must be binary (0 or 1)".to_string(),
            ));
        }
        Ok(())
    }

    fn clone_box(&self) -> Box<dyn PluggableDistribution> {
        Box::new(self.clone())
    }
}

// Additional built-in implementations for other traits...

#[derive(Debug, Clone, Default)]
pub struct BuiltinLaplaceSmoothing;

impl ComposableSmoothingMethod for BuiltinLaplaceSmoothing {
    fn name(&self) -> &'static str {
        "laplace"
    }

    fn smooth(&self, counts: &Array1<Float>, alpha: Float) -> Array1<Float> {
        counts + alpha
    }

    fn n_hyperparameters(&self) -> usize {
        0
    }

    fn hyperparameters(&self) -> Vec<Float> {
        vec![]
    }

    fn set_hyperparameters(&mut self, _params: &[Float]) -> Result<()> {
        Ok(())
    }

    fn is_applicable(&self, _data_characteristics: &DataCharacteristics) -> bool {
        true
    }

    fn clone_box(&self) -> Box<dyn ComposableSmoothingMethod> {
        Box::new(self.clone())
    }
}

#[derive(Debug, Clone, Default)]
pub struct BuiltinLidstoneSmoothing {
    alpha: Float,
}

impl ComposableSmoothingMethod for BuiltinLidstoneSmoothing {
    fn name(&self) -> &'static str {
        "lidstone"
    }

    fn smooth(&self, counts: &Array1<Float>, alpha: Float) -> Array1<Float> {
        counts + (alpha * self.alpha)
    }

    fn n_hyperparameters(&self) -> usize {
        1
    }

    fn hyperparameters(&self) -> Vec<Float> {
        vec![self.alpha]
    }

    fn set_hyperparameters(&mut self, params: &[Float]) -> Result<()> {
        if params.len() != 1 {
            return Err(SklearsError::InvalidInput(
                "Expected 1 hyperparameter".to_string(),
            ));
        }
        self.alpha = params[0];
        Ok(())
    }

    fn is_applicable(&self, _data_characteristics: &DataCharacteristics) -> bool {
        true
    }

    fn clone_box(&self) -> Box<dyn ComposableSmoothingMethod> {
        Box::new(self.clone())
    }
}

#[derive(Debug, Clone, Default)]
pub struct BuiltinMLEEstimator;

impl ExtensibleParameterEstimator for BuiltinMLEEstimator {
    fn name(&self) -> &'static str {
        "mle"
    }

    fn estimate(&self, x: &Array2<Float>, y: &Array1<i32>) -> Result<EstimationResult> {
        // Simple MLE implementation
        let mut parameters = HashMap::new();

        // Calculate class means for Gaussian features
        let classes: Vec<i32> = {
            let mut c = y.to_vec();
            c.sort();
            c.dedup();
            c
        };

        for (class_idx, &class) in classes.iter().enumerate() {
            let class_mask: Vec<bool> = y.iter().map(|&label| label == class).collect();
            let class_data: Array2<Float> = x.select(
                scirs2_core::ndarray::Axis(0),
                &class_mask
                    .iter()
                    .enumerate()
                    .filter(|(_, &mask)| mask)
                    .map(|(idx, _)| idx)
                    .collect::<Vec<_>>(),
            );

            if !class_data.is_empty() {
                let means: Vec<Float> = class_data
                    .mean_axis(scirs2_core::ndarray::Axis(0))
                    .unwrap()
                    .to_vec();
                parameters.insert(format!("class_{}_means", class_idx), means);
            }
        }

        Ok(EstimationResult {
            parameters,
            log_likelihood: 0.0, // Would compute actual log-likelihood
            aic: None,
            bic: None,
            convergence_info: ConvergenceInfo {
                converged: true,
                n_iterations: 1,
                final_log_likelihood: 0.0,
                convergence_threshold: 1e-6,
            },
        })
    }

    fn supports_configuration(&self, _config: &EstimatorConfiguration) -> bool {
        true
    }

    fn complexity(&self) -> ComputationalComplexity {
        ComputationalComplexity::Linear
    }

    fn clone_box(&self) -> Box<dyn ExtensibleParameterEstimator> {
        Box::new(self.clone())
    }
}

#[derive(Debug, Clone, Default)]
pub struct BuiltinMAPEstimator;

impl ExtensibleParameterEstimator for BuiltinMAPEstimator {
    fn name(&self) -> &'static str {
        "map"
    }

    fn estimate(&self, x: &Array2<Float>, y: &Array1<i32>) -> Result<EstimationResult> {
        // MAP estimation with priors
        let mut parameters = HashMap::new();

        // Similar to MLE but with prior regularization
        let classes: Vec<i32> = {
            let mut c = y.to_vec();
            c.sort();
            c.dedup();
            c
        };

        for (class_idx, &class) in classes.iter().enumerate() {
            let class_mask: Vec<bool> = y.iter().map(|&label| label == class).collect();
            let class_data: Array2<Float> = x.select(
                scirs2_core::ndarray::Axis(0),
                &class_mask
                    .iter()
                    .enumerate()
                    .filter(|(_, &mask)| mask)
                    .map(|(idx, _)| idx)
                    .collect::<Vec<_>>(),
            );

            if !class_data.is_empty() {
                // Add some regularization for MAP estimation
                let means: Vec<Float> = class_data
                    .mean_axis(scirs2_core::ndarray::Axis(0))
                    .unwrap()
                    .iter()
                    .map(|&mean| mean * 0.99) // Simple regularization
                    .collect();
                parameters.insert(format!("class_{}_means", class_idx), means);
            }
        }

        Ok(EstimationResult {
            parameters,
            log_likelihood: 0.0,
            aic: None,
            bic: None,
            convergence_info: ConvergenceInfo {
                converged: true,
                n_iterations: 1,
                final_log_likelihood: 0.0,
                convergence_threshold: 1e-6,
            },
        })
    }

    fn supports_configuration(&self, _config: &EstimatorConfiguration) -> bool {
        true
    }

    fn complexity(&self) -> ComputationalComplexity {
        ComputationalComplexity::Linear
    }

    fn clone_box(&self) -> Box<dyn ExtensibleParameterEstimator> {
        Box::new(self.clone())
    }
}

#[derive(Debug, Clone, Default)]
pub struct BuiltinCrossValidationSelector;

impl FlexibleModelSelector for BuiltinCrossValidationSelector {
    fn name(&self) -> &'static str {
        "cross_validation"
    }

    fn select_model(&self, candidates: &[ModelCandidate]) -> Result<ModelCandidate> {
        candidates
            .iter()
            .max_by(|a, b| {
                let score_a = a.score.unwrap_or(Float::NEG_INFINITY);
                let score_b = b.score.unwrap_or(Float::NEG_INFINITY);
                score_a
                    .partial_cmp(&score_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .cloned()
            .ok_or_else(|| SklearsError::InvalidInput("No candidates provided".to_string()))
    }

    fn evaluate_model(&self, candidate: &ModelCandidate) -> Result<Float> {
        candidate
            .score
            .ok_or_else(|| SklearsError::InvalidInput("Candidate has no score".to_string()))
    }

    fn selection_criteria(&self) -> Vec<SelectionCriterion> {
        vec![SelectionCriterion::CrossValidationScore]
    }

    fn clone_box(&self) -> Box<dyn FlexibleModelSelector> {
        Box::new(self.clone())
    }
}

#[derive(Debug, Clone, Default)]
pub struct BuiltinInformationCriterionSelector;

impl FlexibleModelSelector for BuiltinInformationCriterionSelector {
    fn name(&self) -> &'static str {
        "information_criterion"
    }

    fn select_model(&self, candidates: &[ModelCandidate]) -> Result<ModelCandidate> {
        // Select based on AIC/BIC (lower is better for IC)
        candidates
            .iter()
            .min_by(|a, b| {
                let score_a = a.score.unwrap_or(Float::INFINITY);
                let score_b = b.score.unwrap_or(Float::INFINITY);
                score_a
                    .partial_cmp(&score_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .cloned()
            .ok_or_else(|| SklearsError::InvalidInput("No candidates provided".to_string()))
    }

    fn evaluate_model(&self, candidate: &ModelCandidate) -> Result<Float> {
        candidate
            .score
            .ok_or_else(|| SklearsError::InvalidInput("Candidate has no score".to_string()))
    }

    fn selection_criteria(&self) -> Vec<SelectionCriterion> {
        vec![SelectionCriterion::AIC, SelectionCriterion::BIC]
    }

    fn clone_box(&self) -> Box<dyn FlexibleModelSelector> {
        Box::new(self.clone())
    }
}

/// Global plugin registry instance
static GLOBAL_REGISTRY: once_cell::sync::Lazy<std::sync::Mutex<PluginRegistry>> =
    once_cell::sync::Lazy::new(|| std::sync::Mutex::new(PluginRegistry::new()));

/// Get a reference to the global plugin registry
pub fn get_global_registry() -> std::sync::MutexGuard<'static, PluginRegistry> {
    GLOBAL_REGISTRY.lock().unwrap()
}

/// Register a custom distribution globally
pub fn register_distribution(distribution: Box<dyn PluggableDistribution>) {
    get_global_registry().register_distribution(distribution);
}

/// Register a custom smoothing method globally
pub fn register_smoothing_method(method: Box<dyn ComposableSmoothingMethod>) {
    get_global_registry().register_smoothing_method(method);
}

/// Register a custom parameter estimator globally
pub fn register_parameter_estimator(estimator: Box<dyn ExtensibleParameterEstimator>) {
    get_global_registry().register_parameter_estimator(estimator);
}

/// Register a custom model selector globally
pub fn register_model_selector(selector: Box<dyn FlexibleModelSelector>) {
    get_global_registry().register_model_selector(selector);
}

/// Register middleware globally
pub fn register_middleware(middleware: Arc<dyn PredictionMiddleware>) {
    get_global_registry().register_middleware(middleware);
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plugin_registry_creation() {
        let registry = PluginRegistry::new();

        // Check that built-in components are registered
        assert!(!registry.list_distributions().is_empty());
        assert!(!registry.list_smoothing_methods().is_empty());
        assert!(!registry.list_parameter_estimators().is_empty());
        assert!(!registry.list_model_selectors().is_empty());
    }

    #[test]
    fn test_gaussian_distribution() {
        let mut dist = BuiltinGaussianDistribution::default();

        let data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        dist.fit(&data).unwrap();

        assert!(dist.log_prob(3.0).is_finite());
        assert_eq!(dist.n_parameters(), 2);

        let params = dist.parameters();
        assert_eq!(params.len(), 2);
        assert!((params[0] - 3.0).abs() < 1e-10); // mean should be 3.0
    }

    #[test]
    fn test_laplace_smoothing() {
        let smoothing = BuiltinLaplaceSmoothing::default();

        let counts = Array1::from_vec(vec![1.0, 2.0, 0.0]);
        let smoothed = smoothing.smooth(&counts, 1.0);

        assert_eq!(smoothed, Array1::from_vec(vec![2.0, 3.0, 1.0]));
    }

    #[test]
    fn test_mle_estimator() {
        let estimator = BuiltinMLEEstimator::default();

        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let y = Array1::from_vec(vec![0, 0, 1, 1]);

        let result = estimator.estimate(&x, &y).unwrap();
        assert!(!result.parameters.is_empty());
        assert!(result.convergence_info.converged);
    }

    #[test]
    fn test_model_selector() {
        let selector = BuiltinCrossValidationSelector::default();

        let candidates = vec![
            ModelCandidate {
                model_type: "gaussian".to_string(),
                configuration: HashMap::new(),
                score: Some(0.8),
                cross_validation_scores: None,
                training_time: None,
            },
            ModelCandidate {
                model_type: "multinomial".to_string(),
                configuration: HashMap::new(),
                score: Some(0.9),
                cross_validation_scores: None,
                training_time: None,
            },
        ];

        let best = selector.select_model(&candidates).unwrap();
        assert_eq!(best.model_type, "multinomial");
        assert_eq!(best.score, Some(0.9));
    }

    #[test]
    fn test_distribution_registration() {
        let mut registry = PluginRegistry::new();
        let custom_dist = Box::new(BuiltinGaussianDistribution::default());

        registry.register_distribution(custom_dist);
        assert!(registry.get_distribution("gaussian").is_some());
    }
}
