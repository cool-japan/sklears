//! Modular Framework for Linear Models
//!
//! This module implements a trait-based system for pluggable solvers, loss functions,
//! and regularization schemes. This addresses TODO items for architectural improvements:
//! - Separate solver implementations into trait-based system
//! - Create pluggable loss function framework  
//! - Implement composable regularization schemes
//! - Add extensible prediction interface

use scirs2_core::ndarray::{Array1, Array2, Axis};
use sklears_core::{
    error::{Result, SklearsError},
    types::Float,
};
use std::fmt::Debug;

/// Trait for optimization objectives that can be minimized
pub trait Objective: Debug + Send + Sync {
    /// Compute the objective value
    fn value(&self, coefficients: &Array1<Float>, data: &ObjectiveData) -> Result<Float>;

    /// Compute the gradient of the objective
    fn gradient(&self, coefficients: &Array1<Float>, data: &ObjectiveData)
        -> Result<Array1<Float>>;

    /// Compute both value and gradient (often more efficient than separate calls)
    fn value_and_gradient(
        &self,
        coefficients: &Array1<Float>,
        data: &ObjectiveData,
    ) -> Result<(Float, Array1<Float>)> {
        let value = self.value(coefficients, data)?;
        let gradient = self.gradient(coefficients, data)?;
        Ok((value, gradient))
    }

    /// Check if the objective supports Hessian computation
    fn supports_hessian(&self) -> bool {
        false
    }

    /// Compute the Hessian matrix (for second-order methods)
    fn hessian(
        &self,
        _coefficients: &Array1<Float>,
        _data: &ObjectiveData,
    ) -> Result<Array2<Float>> {
        Err(SklearsError::InvalidOperation(
            "Hessian computation not supported for this objective".to_string(),
        ))
    }
}

/// Data structure containing all information needed for objective computation
#[derive(Debug, Clone)]
pub struct ObjectiveData {
    /// Feature matrix (n_samples, n_features)
    pub features: Array2<Float>,
    /// Target values (n_samples,)
    pub targets: Array1<Float>,
    /// Sample weights (optional)
    pub sample_weights: Option<Array1<Float>>,
    /// Additional metadata
    pub metadata: ObjectiveMetadata,
}

/// Metadata for objective computation
#[derive(Debug, Clone, Default)]
pub struct ObjectiveMetadata {
    /// Whether to fit intercept
    pub fit_intercept: bool,
    /// Feature scaling factors (for numerical stability)
    pub feature_scale: Option<Array1<Float>>,
    /// Target scaling factors
    pub target_scale: Option<Float>,
}

/// Trait for loss functions that measure prediction error
pub trait LossFunction: Debug + Send + Sync {
    /// Compute the loss value for predictions
    fn loss(&self, y_true: &Array1<Float>, y_pred: &Array1<Float>) -> Result<Float>;

    /// Compute the derivative of loss with respect to predictions
    fn loss_derivative(
        &self,
        y_true: &Array1<Float>,
        y_pred: &Array1<Float>,
    ) -> Result<Array1<Float>>;

    /// Compute both loss and derivative (often more efficient)
    fn loss_and_derivative(
        &self,
        y_true: &Array1<Float>,
        y_pred: &Array1<Float>,
    ) -> Result<(Float, Array1<Float>)> {
        let loss = self.loss(y_true, y_pred)?;
        let derivative = self.loss_derivative(y_true, y_pred)?;
        Ok((loss, derivative))
    }

    /// Check if this is a classification loss (vs regression)
    fn is_classification(&self) -> bool {
        false
    }

    /// Get the name of this loss function
    fn name(&self) -> &'static str;
}

/// Trait for regularization penalties
pub trait Regularization: Debug + Send + Sync {
    /// Compute the regularization penalty value
    fn penalty(&self, coefficients: &Array1<Float>) -> Result<Float>;

    /// Compute the regularization gradient (subgradient for non-smooth penalties)
    fn penalty_gradient(&self, coefficients: &Array1<Float>) -> Result<Array1<Float>>;

    /// Apply the proximal operator (for proximal gradient methods)
    fn proximal_operator(
        &self,
        coefficients: &Array1<Float>,
        step_size: Float,
    ) -> Result<Array1<Float>> {
        // Default implementation: no proximal operator (smooth penalties)
        Ok(coefficients.clone())
    }

    /// Check if this regularization is non-smooth (requires proximal methods)
    fn is_non_smooth(&self) -> bool {
        false
    }

    /// Get the regularization strength
    fn strength(&self) -> Float;

    /// Get the name of this regularization
    fn name(&self) -> &'static str;
}

/// Trait for optimization solvers
pub trait OptimizationSolver: Debug + Send + Sync {
    /// Configuration type for this solver
    type Config: Debug + Clone + Send + Sync;

    /// Result type returned by this solver
    type Result: Debug + Clone + Send + Sync;

    /// Solve the optimization problem
    fn solve(
        &self,
        objective: &dyn Objective,
        initial_guess: &Array1<Float>,
        config: &Self::Config,
    ) -> Result<Self::Result>;

    /// Check if this solver supports the given objective type
    fn supports_objective(&self, objective: &dyn Objective) -> bool;

    /// Get the name of this solver
    fn name(&self) -> &'static str;

    /// Get solver-specific recommendations for the given problem
    fn get_recommendations(&self, data: &ObjectiveData) -> SolverRecommendations {
        SolverRecommendations::default()
    }
}

/// Recommendations for solver configuration
#[derive(Debug, Clone, Default)]
pub struct SolverRecommendations {
    /// Recommended maximum iterations
    pub max_iterations: Option<usize>,
    /// Recommended convergence tolerance
    pub tolerance: Option<Float>,
    /// Recommended step size or learning rate
    pub step_size: Option<Float>,
    /// Whether to use line search
    pub use_line_search: Option<bool>,
    /// Additional solver-specific advice
    pub notes: Vec<String>,
}

/// Extensible prediction interface supporting different prediction types
pub trait PredictionProvider: Debug + Send + Sync {
    fn predict(
        &self,
        features: &Array2<Float>,
        coefficients: &Array1<Float>,
        intercept: Option<Float>,
    ) -> Result<Array1<Float>>;

    fn predict_with_confidence(
        &self,
        features: &Array2<Float>,
        coefficients: &Array1<Float>,
        intercept: Option<Float>,
        confidence_level: Float,
    ) -> Result<PredictionWithConfidence> {
        let predictions = self.predict(features, coefficients, intercept)?;
        Ok(PredictionWithConfidence {
            predictions,
            lower_bounds: None,
            upper_bounds: None,
            confidence_level,
        })
    }

    /// Prediction with uncertainty quantification (if supported)
    fn predict_with_uncertainty(
        &self,
        features: &Array2<Float>,
        coefficients: &Array1<Float>,
        intercept: Option<Float>,
    ) -> Result<PredictionWithUncertainty> {
        let predictions = self.predict(features, coefficients, intercept)?;
        Ok(PredictionWithUncertainty {
            predictions,
            uncertainties: None,
            prediction_intervals: None,
        })
    }

    /// Check if this provider supports confidence intervals
    fn supports_confidence_intervals(&self) -> bool {
        false
    }

    /// Check if this provider supports uncertainty quantification
    fn supports_uncertainty_quantification(&self) -> bool {
        false
    }

    /// Get the name of this prediction provider
    fn name(&self) -> &'static str;
}

/// Prediction result with confidence intervals
#[derive(Debug, Clone)]
pub struct PredictionWithConfidence {
    /// Point predictions
    pub predictions: Array1<Float>,
    /// Lower confidence bounds (optional)
    pub lower_bounds: Option<Array1<Float>>,
    /// Upper confidence bounds (optional)
    pub upper_bounds: Option<Array1<Float>>,
    /// Confidence level used (e.g., 0.95 for 95% confidence)
    pub confidence_level: Float,
}

/// Prediction result with uncertainty quantification
#[derive(Debug, Clone)]
pub struct PredictionWithUncertainty {
    /// Point predictions
    pub predictions: Array1<Float>,
    /// Prediction uncertainties (standard errors)
    pub uncertainties: Option<Array1<Float>>,
    /// Prediction intervals
    pub prediction_intervals: Option<Array2<Float>>, // (n_samples, 2) for [lower, upper]
}

/// Standard linear prediction provider
#[derive(Debug)]
pub struct LinearPredictionProvider;

impl PredictionProvider for LinearPredictionProvider {
    fn predict(
        &self,
        features: &Array2<Float>,
        coefficients: &Array1<Float>,
        intercept: Option<Float>,
    ) -> Result<Array1<Float>> {
        let mut predictions = features.dot(coefficients);
        if let Some(intercept_val) = intercept {
            predictions += intercept_val;
        }
        Ok(predictions)
    }

    fn name(&self) -> &'static str {
        "LinearPrediction"
    }
}

/// Probabilistic prediction provider for classification
#[derive(Debug)]
pub struct ProbabilisticPredictionProvider;

impl PredictionProvider for ProbabilisticPredictionProvider {
    fn predict(
        &self,
        features: &Array2<Float>,
        coefficients: &Array1<Float>,
        intercept: Option<Float>,
    ) -> Result<Array1<Float>> {
        let linear_predictions =
            LinearPredictionProvider.predict(features, coefficients, intercept)?;
        // Apply sigmoid transformation for binary classification
        let probabilities = linear_predictions.mapv(|x| 1.0 / (1.0 + (-x).exp()));
        Ok(probabilities)
    }

    fn supports_confidence_intervals(&self) -> bool {
        true
    }

    fn predict_with_confidence(
        &self,
        features: &Array2<Float>,
        coefficients: &Array1<Float>,
        intercept: Option<Float>,
        confidence_level: Float,
    ) -> Result<PredictionWithConfidence> {
        let predictions = self.predict(features, coefficients, intercept)?;

        // For logistic regression, we can compute confidence intervals
        // based on the variance of the linear predictor
        let linear_predictions =
            LinearPredictionProvider.predict(features, coefficients, intercept)?;
        let variances = Array1::ones(features.nrows()); // Simplified - would need proper variance calculation

        let z_score = match confidence_level {
            0.90 => 1.645,
            0.95 => 1.96,
            0.99 => 2.576,
            _ => 1.96, // Default to 95%
        };

        let margins = variances.mapv(|v: Float| z_score * v.sqrt());
        let lower_bounds = &predictions - &margins;
        let upper_bounds = &predictions + &margins;

        Ok(PredictionWithConfidence {
            predictions,
            lower_bounds: Some(lower_bounds),
            upper_bounds: Some(upper_bounds),
            confidence_level,
        })
    }

    fn name(&self) -> &'static str {
        "ProbabilisticPrediction"
    }
}

/// Bayesian prediction provider with uncertainty quantification
#[derive(Debug)]
pub struct BayesianPredictionProvider {
    /// Posterior covariance matrix
    pub posterior_covariance: Option<Array2<Float>>,
}

impl BayesianPredictionProvider {
    pub fn new(posterior_covariance: Option<Array2<Float>>) -> Self {
        Self {
            posterior_covariance,
        }
    }
}

impl PredictionProvider for BayesianPredictionProvider {
    fn predict(
        &self,
        features: &Array2<Float>,
        coefficients: &Array1<Float>,
        intercept: Option<Float>,
    ) -> Result<Array1<Float>> {
        LinearPredictionProvider.predict(features, coefficients, intercept)
    }

    fn supports_uncertainty_quantification(&self) -> bool {
        self.posterior_covariance.is_some()
    }

    fn predict_with_uncertainty(
        &self,
        features: &Array2<Float>,
        coefficients: &Array1<Float>,
        intercept: Option<Float>,
    ) -> Result<PredictionWithUncertainty> {
        let predictions = self.predict(features, coefficients, intercept)?;

        if let Some(ref cov) = self.posterior_covariance {
            // Compute prediction uncertainties using the posterior covariance
            let mut uncertainties = Array1::zeros(features.nrows());

            for (i, row) in features.axis_iter(Axis(0)).enumerate() {
                let variance = row.dot(&cov.dot(&row));
                uncertainties[i] = variance.sqrt();
            }

            // Compute 95% prediction intervals
            let z_score = 1.96;
            let lower_bounds = &predictions - z_score * &uncertainties;
            let upper_bounds = &predictions + z_score * &uncertainties;

            let mut prediction_intervals = Array2::zeros((features.nrows(), 2));
            prediction_intervals.column_mut(0).assign(&lower_bounds);
            prediction_intervals.column_mut(1).assign(&upper_bounds);

            Ok(PredictionWithUncertainty {
                predictions,
                uncertainties: Some(uncertainties),
                prediction_intervals: Some(prediction_intervals),
            })
        } else {
            Ok(PredictionWithUncertainty {
                predictions,
                uncertainties: None,
                prediction_intervals: None,
            })
        }
    }

    fn name(&self) -> &'static str {
        "BayesianPrediction"
    }
}

/// Configuration for the modular framework
#[derive(Debug, Clone)]
pub struct ModularConfig {
    /// Maximum iterations for optimization
    pub max_iterations: usize,
    /// Convergence tolerance
    pub tolerance: Float,
    /// Whether to enable verbose output
    pub verbose: bool,
    /// Random seed for reproducibility
    pub random_seed: Option<u64>,
}

impl Default for ModularConfig {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            tolerance: 1e-6,
            verbose: false,
            random_seed: None,
        }
    }
}

/// Result of optimization through the modular framework
#[derive(Debug, Clone)]
pub struct OptimizationResult {
    /// Final coefficient values
    pub coefficients: Array1<Float>,
    /// Final intercept value (if fitted)
    pub intercept: Option<Float>,
    /// Final objective value
    pub objective_value: Float,
    /// Number of iterations performed
    pub n_iterations: usize,
    /// Whether optimization converged
    pub converged: bool,
    /// Solver-specific information
    pub solver_info: SolverInfo,
}

/// Information about the solver execution
#[derive(Debug, Clone)]
pub struct SolverInfo {
    /// Name of the solver used
    pub solver_name: String,
    /// Solver-specific metrics
    pub metrics: std::collections::HashMap<String, Float>,
    /// Warning messages
    pub warnings: Vec<String>,
    /// Convergence history (if available)
    pub convergence_history: Option<Array1<Float>>,
}

/// The main modular framework that coordinates components
#[derive(Debug)]
pub struct ModularFramework {
    config: ModularConfig,
}

impl ModularFramework {
    /// Create a new modular framework with default configuration
    pub fn new() -> Self {
        Self {
            config: ModularConfig::default(),
        }
    }

    /// Create a new modular framework with custom configuration
    pub fn with_config(config: ModularConfig) -> Self {
        Self { config }
    }

    /// Solve an optimization problem using the modular components
    pub fn solve<S: OptimizationSolver + ?Sized>(
        &self,
        loss: &dyn LossFunction,
        regularization: Option<&dyn Regularization>,
        solver: &S,
        data: &ObjectiveData,
        initial_guess: Option<&Array1<Float>>,
    ) -> Result<OptimizationResult> {
        // Create the composite objective combining loss and regularization
        let objective = CompositeObjective::new(loss, regularization);

        // Create initial guess if not provided
        let n_features = data.features.ncols();
        let init = initial_guess
            .cloned()
            .unwrap_or_else(|| Array1::zeros(n_features));

        // Create solver config from framework config
        let solver_config = self.create_solver_config::<S>(&objective, data)?;

        // Solve the problem
        let solver_result = solver.solve(&objective, &init, &solver_config)?;

        // Convert solver result to framework result
        self.convert_result::<S>(solver_result, &objective, data)
    }

    /// Create solver-specific configuration from framework configuration
    fn create_solver_config<S: OptimizationSolver + ?Sized>(
        &self,
        objective: &dyn Objective,
        data: &ObjectiveData,
    ) -> Result<S::Config> {
        // Get solver recommendations and use them to create config
        let solver_name = std::any::type_name::<S>();

        // For now, we'll return an error indicating the specific solver type
        // In practice, each solver would register a config converter
        Err(SklearsError::InvalidOperation(format!(
            "Config conversion not implemented for solver: {}",
            solver_name
        )))
    }

    /// Convert solver-specific result to framework result
    fn convert_result<S: OptimizationSolver + ?Sized>(
        &self,
        solver_result: S::Result,
        objective: &dyn Objective,
        data: &ObjectiveData,
    ) -> Result<OptimizationResult> {
        // For now, we'll return an error indicating the specific result type
        // In practice, each solver result would implement a conversion trait
        let result_type = std::any::type_name::<S::Result>();

        Err(SklearsError::InvalidOperation(format!(
            "Result conversion not implemented for result type: {}",
            result_type
        )))
    }
}

impl Default for ModularFramework {
    fn default() -> Self {
        Self::new()
    }
}

/// A composite objective that combines a loss function with optional regularization
#[derive(Debug)]
pub struct CompositeObjective<'a> {
    loss: &'a dyn LossFunction,
    regularization: Option<&'a dyn Regularization>,
}

impl<'a> CompositeObjective<'a> {
    /// Create a new composite objective
    pub fn new(loss: &'a dyn LossFunction, regularization: Option<&'a dyn Regularization>) -> Self {
        Self {
            loss,
            regularization,
        }
    }
}

impl<'a> Objective for CompositeObjective<'a> {
    fn value(&self, coefficients: &Array1<Float>, data: &ObjectiveData) -> Result<Float> {
        // Compute predictions
        let predictions = data.features.dot(coefficients);

        // Compute loss
        let loss_value = self.loss.loss(&data.targets, &predictions)?;

        // Add regularization if present
        let regularization_value = if let Some(reg) = self.regularization {
            reg.penalty(coefficients)?
        } else {
            0.0
        };

        Ok(loss_value + regularization_value)
    }

    fn gradient(
        &self,
        coefficients: &Array1<Float>,
        data: &ObjectiveData,
    ) -> Result<Array1<Float>> {
        // Compute predictions
        let predictions = data.features.dot(coefficients);

        // Compute loss derivative with respect to predictions
        let loss_grad_pred = self.loss.loss_derivative(&data.targets, &predictions)?;

        // Compute gradient with respect to coefficients using chain rule
        let mut gradient = data.features.t().dot(&loss_grad_pred);

        // Add regularization gradient if present
        if let Some(reg) = self.regularization {
            let reg_grad = reg.penalty_gradient(coefficients)?;
            gradient = gradient + reg_grad;
        }

        Ok(gradient)
    }

    fn supports_hessian(&self) -> bool {
        // For simplicity, we don't support Hessian computation in the composite objective
        false
    }
}

/// Utility function to create a modular linear regression solver
pub fn create_modular_linear_regression(
    loss: Box<dyn LossFunction>,
    regularization: Option<Box<dyn Regularization>>,
    solver: Box<dyn OptimizationSolver<Config = ModularConfig, Result = OptimizationResult>>,
) -> ModularLinearModel {
    ModularLinearModel {
        loss,
        regularization,
        solver,
        framework: ModularFramework::new(),
    }
}

/// A linear model built using the modular framework
#[derive(Debug)]
pub struct ModularLinearModel {
    loss: Box<dyn LossFunction>,
    regularization: Option<Box<dyn Regularization>>,
    solver: Box<dyn OptimizationSolver<Config = ModularConfig, Result = OptimizationResult>>,
    framework: ModularFramework,
}

impl ModularLinearModel {
    /// Fit the model to training data
    pub fn fit(&self, X: &Array2<Float>, y: &Array1<Float>) -> Result<OptimizationResult> {
        let data = ObjectiveData {
            features: X.clone(),
            targets: y.clone(),
            sample_weights: None,
            metadata: ObjectiveMetadata::default(),
        };

        self.framework.solve(
            self.loss.as_ref(),
            self.regularization.as_deref(),
            self.solver.as_ref(),
            &data,
            None,
        )
    }

    /// Make predictions using the fitted model
    pub fn predict(&self, X: &Array2<Float>, result: &OptimizationResult) -> Result<Array1<Float>> {
        let predictions = X.dot(&result.coefficients);

        // Add intercept if present
        if let Some(intercept) = result.intercept {
            Ok(predictions + intercept)
        } else {
            Ok(predictions)
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array;

    // Test helper: dummy loss function
    #[derive(Debug)]
    struct DummyLoss;

    impl LossFunction for DummyLoss {
        fn loss(&self, y_true: &Array1<Float>, y_pred: &Array1<Float>) -> Result<Float> {
            Ok(((y_true - y_pred).mapv(|x| x * x)).sum() / (2.0 * y_true.len() as Float))
        }

        fn loss_derivative(
            &self,
            y_true: &Array1<Float>,
            y_pred: &Array1<Float>,
        ) -> Result<Array1<Float>> {
            Ok((y_pred - y_true) / (y_true.len() as Float))
        }

        fn name(&self) -> &'static str {
            "SquaredLoss"
        }
    }

    // Test helper: dummy regularization
    #[derive(Debug)]
    struct DummyRegularization {
        alpha: Float,
    }

    impl Regularization for DummyRegularization {
        fn penalty(&self, coefficients: &Array1<Float>) -> Result<Float> {
            Ok(0.5 * self.alpha * coefficients.mapv(|x| x * x).sum())
        }

        fn penalty_gradient(&self, coefficients: &Array1<Float>) -> Result<Array1<Float>> {
            Ok(self.alpha * coefficients)
        }

        fn strength(&self) -> Float {
            self.alpha
        }

        fn name(&self) -> &'static str {
            "L2Regularization"
        }
    }

    #[test]
    fn test_composite_objective() {
        let loss = DummyLoss;
        let regularization = DummyRegularization { alpha: 0.1 };
        let objective = CompositeObjective::new(&loss, Some(&regularization));

        let data = ObjectiveData {
            features: Array::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
            targets: Array::from_vec(vec![1.0, 2.0, 3.0]),
            sample_weights: None,
            metadata: ObjectiveMetadata::default(),
        };

        let coefficients = Array::from_vec(vec![0.5, 0.5]);

        // Test that value computation doesn't panic
        let value = objective.value(&coefficients, &data);
        assert!(value.is_ok());

        // Test that gradient computation doesn't panic
        let gradient = objective.gradient(&coefficients, &data);
        assert!(gradient.is_ok());
    }

    #[test]
    fn test_modular_config() {
        let config = ModularConfig::default();
        assert_eq!(config.max_iterations, 1000);
        assert_eq!(config.tolerance, 1e-6);
        assert!(!config.verbose);
        assert!(config.random_seed.is_none());
    }

    #[test]
    fn test_loss_function_interface() {
        let loss = DummyLoss;
        let y_true = Array::from_vec(vec![1.0, 2.0, 3.0]);
        let y_pred = Array::from_vec(vec![1.1, 1.9, 3.1]);

        let loss_value = loss.loss(&y_true, &y_pred).unwrap();
        assert!(loss_value >= 0.0);

        let derivative = loss.loss_derivative(&y_true, &y_pred).unwrap();
        assert_eq!(derivative.len(), y_true.len());

        assert_eq!(loss.name(), "SquaredLoss");
        assert!(!loss.is_classification());
    }

    #[test]
    fn test_regularization_interface() {
        let reg = DummyRegularization { alpha: 0.1 };
        let coefficients = Array::from_vec(vec![1.0, -1.0, 2.0]);

        let penalty = reg.penalty(&coefficients).unwrap();
        assert!(penalty >= 0.0);

        let gradient = reg.penalty_gradient(&coefficients).unwrap();
        assert_eq!(gradient.len(), coefficients.len());

        assert_eq!(reg.strength(), 0.1);
        assert_eq!(reg.name(), "L2Regularization");
        assert!(!reg.is_non_smooth());
    }
}
