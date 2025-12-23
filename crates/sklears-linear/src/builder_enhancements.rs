//! Enhanced builder patterns for linear models
//!
//! This module provides advanced builder patterns with:
//! - Fluent API for complex model configurations
//! - Compile-time parameter validation
//! - Configuration presets for common use cases
//! - Method chaining for model configuration
//!
//! The builders use phantom types to ensure type safety and prevent invalid configurations.

use std::marker::PhantomData;

use sklears_core::{
    error::{Result, SklearsError},
    traits::Estimator,
    types::Float,
};

use crate::{LinearRegression, LinearRegressionConfig, Penalty, Solver};

#[cfg(feature = "logistic-regression")]
use crate::{LogisticRegression, LogisticRegressionConfig};

/// Configuration presets for common use cases
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ModelPreset {
    Quick,
    Balanced,
    HighAccuracy,
    Robust,
    MemoryEfficient,
    Production,
}

/// Marker traits for compile-time validation
pub mod validation {
    /// Marker for models that have been properly configured
    pub trait Configured {}

    /// Marker for models that have regularization configured
    pub trait WithRegularization {}

    /// Marker for models that have solver configured
    pub trait WithSolver {}
}

/// Enhanced builder for Linear Regression with preset configurations
#[derive(Debug, Clone)]
pub struct EnhancedLinearRegressionBuilder<State = Unconfigured> {
    config: LinearRegressionConfig,
    validation_config: ValidationConfig,
    _state: PhantomData<State>,
}

/// Enhanced builder for Logistic Regression with preset configurations
#[cfg(feature = "logistic-regression")]
#[derive(Debug, Clone)]
pub struct EnhancedLogisticRegressionBuilder<State = Unconfigured> {
    config: LogisticRegressionConfig,
    validation_config: ValidationConfig,
    _state: PhantomData<State>,
}

/// Marker type for unconfigured builders
#[derive(Debug, Clone, Copy)]
pub struct Unconfigured;

/// Marker type for configured builders
#[derive(Debug, Clone, Copy)]
pub struct Configured;

/// Marker type for builders with regularization
#[derive(Debug, Clone, Copy)]
pub struct WithRegularization;

/// Marker type for builders with solver
#[derive(Debug, Clone, Copy)]
pub struct WithSolver;

/// Validation configuration for enhanced models
#[derive(Debug, Clone, Default)]
pub struct ValidationConfig {
    /// Number of cross-validation folds
    pub cross_validation_folds: Option<usize>,
    /// Validation split ratio
    pub validation_split: Option<Float>,
    /// Whether to use early stopping
    pub early_stopping: bool,
    /// Random state for reproducibility
    pub random_state: Option<u64>,
}

// Enhanced Linear Regression Builder Implementation
impl Default for EnhancedLinearRegressionBuilder<Unconfigured> {
    fn default() -> Self {
        Self {
            config: LinearRegressionConfig::default(),
            validation_config: ValidationConfig::default(),
            _state: PhantomData,
        }
    }
}

impl EnhancedLinearRegressionBuilder<Unconfigured> {
    /// Create a new enhanced linear regression builder
    pub fn new() -> Self {
        Self::default()
    }

    /// Start with a preset configuration
    pub fn with_preset(preset: ModelPreset) -> EnhancedLinearRegressionBuilder<Configured> {
        let builder = Self::new();
        builder.apply_preset(preset)
    }

    /// Apply a configuration preset
    pub fn apply_preset(
        mut self,
        preset: ModelPreset,
    ) -> EnhancedLinearRegressionBuilder<Configured> {
        match preset {
            ModelPreset::Quick => {
                self.config.solver = Solver::Normal;
                self.config.fit_intercept = true;
                self.config.max_iter = 100;
            }
            ModelPreset::Balanced => {
                self.config.solver = Solver::Auto;
                self.config.fit_intercept = true;
                self.config.max_iter = 1000;
                self.config.penalty = Penalty::L2(0.1);
            }
            ModelPreset::HighAccuracy => {
                self.config.solver = Solver::Normal;
                self.config.fit_intercept = true;
                self.config.max_iter = 5000;
                self.config.penalty = Penalty::L2(0.01);
                self.validation_config.cross_validation_folds = Some(5);
            }
            ModelPreset::Robust => {
                self.config.solver = Solver::Auto;
                self.config.fit_intercept = true;
                self.config.penalty = Penalty::L1(0.1);
                self.config.max_iter = 2000;
            }
            ModelPreset::MemoryEfficient => {
                self.config.solver = Solver::Normal;
                self.config.fit_intercept = true;
                self.config.max_iter = 500;
            }
            ModelPreset::Production => {
                self.config.solver = Solver::Auto;
                self.config.fit_intercept = true;
                self.config.penalty = Penalty::ElasticNet {
                    l1_ratio: 0.5,
                    alpha: 0.1,
                };
                self.config.max_iter = 3000;
                self.validation_config.cross_validation_folds = Some(10);
                self.validation_config.early_stopping = true;
            }
        }

        EnhancedLinearRegressionBuilder {
            config: self.config,
            validation_config: self.validation_config,
            _state: PhantomData,
        }
    }
}

impl<State> EnhancedLinearRegressionBuilder<State> {
    /// Set the solver
    pub fn solver(mut self, solver: Solver) -> EnhancedLinearRegressionBuilder<WithSolver> {
        self.config.solver = solver;
        EnhancedLinearRegressionBuilder {
            config: self.config,
            validation_config: self.validation_config,
            _state: PhantomData,
        }
    }

    /// Set whether to fit intercept
    pub fn fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.config.fit_intercept = fit_intercept;
        self
    }

    /// Set regularization penalty
    pub fn penalty(
        mut self,
        penalty: Penalty,
    ) -> EnhancedLinearRegressionBuilder<WithRegularization> {
        self.config.penalty = penalty;
        EnhancedLinearRegressionBuilder {
            config: self.config,
            validation_config: self.validation_config,
            _state: PhantomData,
        }
    }

    /// Set maximum iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.config.max_iter = max_iter;
        self
    }

    /// Set tolerance for convergence
    pub fn tolerance(mut self, tol: f64) -> Self {
        self.config.tol = tol;
        self
    }

    /// Enable warm start
    pub fn warm_start(mut self, warm_start: bool) -> Self {
        self.config.warm_start = warm_start;
        self
    }

    /// Configure cross-validation
    pub fn with_cross_validation(mut self, folds: usize) -> Self {
        self.validation_config.cross_validation_folds = Some(folds);
        self
    }

    /// Configure validation split
    pub fn with_validation_split(mut self, split: Float) -> Self {
        self.validation_config.validation_split = Some(split);
        self
    }

    /// Enable early stopping
    pub fn with_early_stopping(mut self) -> Self {
        self.validation_config.early_stopping = true;
        self
    }

    /// Set random state for reproducibility
    pub fn random_state(mut self, seed: u64) -> Self {
        self.validation_config.random_state = Some(seed);
        self
    }

    /// Build the linear regression model
    pub fn build(self) -> Result<LinearRegression> {
        LinearRegression::new()
            .penalty(self.config.penalty)
            .solver(self.config.solver)
            .fit_intercept(self.config.fit_intercept)
            .max_iter(self.config.max_iter)
            .warm_start(self.config.warm_start)
            .validate_config()
    }

    /// Get the configuration
    pub fn config(&self) -> &LinearRegressionConfig {
        &self.config
    }

    /// Get the validation configuration
    pub fn validation_config(&self) -> &ValidationConfig {
        &self.validation_config
    }
}

// Enhanced Logistic Regression Builder Implementation
#[cfg(feature = "logistic-regression")]
impl Default for EnhancedLogisticRegressionBuilder<Unconfigured> {
    fn default() -> Self {
        Self {
            config: LogisticRegressionConfig::default(),
            validation_config: ValidationConfig::default(),
            _state: PhantomData,
        }
    }
}

#[cfg(feature = "logistic-regression")]
impl EnhancedLogisticRegressionBuilder<Unconfigured> {
    /// Create a new enhanced logistic regression builder
    pub fn new() -> Self {
        Self::default()
    }

    /// Start with a preset configuration
    pub fn with_preset(preset: ModelPreset) -> EnhancedLogisticRegressionBuilder<Configured> {
        let builder = Self::new();
        builder.apply_preset(preset)
    }

    /// Apply a configuration preset
    pub fn apply_preset(
        mut self,
        preset: ModelPreset,
    ) -> EnhancedLogisticRegressionBuilder<Configured> {
        match preset {
            ModelPreset::Quick => {
                self.config.solver = Solver::Lbfgs;
                self.config.max_iter = 100;
                self.config.penalty = Penalty::L2(1.0);
                self.config.tol = 1e-3;
            }
            ModelPreset::Balanced => {
                self.config.solver = Solver::Auto;
                self.config.max_iter = 1000;
                self.config.penalty = Penalty::L2(1.0);
                self.config.tol = 1e-4;
            }
            ModelPreset::HighAccuracy => {
                self.config.solver = Solver::Lbfgs;
                self.config.max_iter = 10000;
                self.config.penalty = Penalty::ElasticNet {
                    l1_ratio: 0.5,
                    alpha: 1.0,
                };
                self.config.tol = 1e-6;
                self.validation_config.cross_validation_folds = Some(5);
            }
            ModelPreset::Robust => {
                self.config.solver = Solver::Saga;
                self.config.penalty = Penalty::L1(1.0);
                self.config.max_iter = 2000;
                self.config.tol = 1e-4;
            }
            ModelPreset::MemoryEfficient => {
                self.config.solver = Solver::Sag;
                self.config.max_iter = 1000;
                self.config.penalty = Penalty::L2(1.0);
                self.config.tol = 1e-3;
            }
            ModelPreset::Production => {
                self.config.solver = Solver::Lbfgs;
                self.config.max_iter = 5000;
                self.config.penalty = Penalty::ElasticNet {
                    l1_ratio: 0.1,
                    alpha: 1.0,
                };
                self.config.tol = 1e-5;
                self.validation_config.cross_validation_folds = Some(5);
                self.validation_config.early_stopping = true;
            }
        }

        EnhancedLogisticRegressionBuilder {
            config: self.config,
            validation_config: self.validation_config,
            _state: PhantomData,
        }
    }
}

#[cfg(feature = "logistic-regression")]
impl<State> EnhancedLogisticRegressionBuilder<State> {
    /// Set the penalty
    pub fn penalty(
        mut self,
        penalty: Penalty,
    ) -> EnhancedLogisticRegressionBuilder<WithRegularization> {
        self.config.penalty = penalty;
        EnhancedLogisticRegressionBuilder {
            config: self.config,
            validation_config: self.validation_config,
            _state: PhantomData,
        }
    }

    /// Set the solver
    pub fn solver(mut self, solver: Solver) -> EnhancedLogisticRegressionBuilder<WithSolver> {
        self.config.solver = solver;
        EnhancedLogisticRegressionBuilder {
            config: self.config,
            validation_config: self.validation_config,
            _state: PhantomData,
        }
    }

    /// Set maximum iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.config.max_iter = max_iter;
        self
    }

    /// Set tolerance
    pub fn tolerance(mut self, tol: f64) -> Self {
        self.config.tol = tol;
        self
    }

    /// Set whether to fit intercept
    pub fn fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.config.fit_intercept = fit_intercept;
        self
    }

    /// Configure cross-validation
    pub fn with_cross_validation(mut self, folds: usize) -> Self {
        self.validation_config.cross_validation_folds = Some(folds);
        self
    }

    /// Configure validation split
    pub fn with_validation_split(mut self, split: Float) -> Self {
        self.validation_config.validation_split = Some(split);
        self
    }

    /// Enable early stopping
    pub fn with_early_stopping(mut self) -> Self {
        self.validation_config.early_stopping = true;
        self
    }

    /// Set random state for reproducibility
    pub fn random_state(mut self, seed: u64) -> Self {
        self.config.random_state = Some(seed);
        self.validation_config.random_state = Some(seed);
        self
    }

    /// Build the logistic regression model
    pub fn build(self) -> Result<LogisticRegression> {
        Ok(LogisticRegression::new()
            .penalty(self.config.penalty)
            .solver(self.config.solver)
            .max_iter(self.config.max_iter)
            .fit_intercept(self.config.fit_intercept))
    }

    /// Get the configuration
    pub fn config(&self) -> &LogisticRegressionConfig {
        &self.config
    }

    /// Get the validation configuration
    pub fn validation_config(&self) -> &ValidationConfig {
        &self.validation_config
    }
}

/// Compile-time validation trait implementations
impl validation::Configured for EnhancedLinearRegressionBuilder<Configured> {}
impl validation::WithRegularization for EnhancedLinearRegressionBuilder<WithRegularization> {}
impl validation::WithSolver for EnhancedLinearRegressionBuilder<WithSolver> {}

#[cfg(feature = "logistic-regression")]
impl validation::Configured for EnhancedLogisticRegressionBuilder<Configured> {}
#[cfg(feature = "logistic-regression")]
impl validation::WithRegularization for EnhancedLogisticRegressionBuilder<WithRegularization> {}
#[cfg(feature = "logistic-regression")]
impl validation::WithSolver for EnhancedLogisticRegressionBuilder<WithSolver> {}

/// Extension trait for model validation
pub trait ModelValidation {
    type Error;

    /// Validate the model configuration
    fn validate_config(self) -> std::result::Result<Self, Self::Error>
    where
        Self: Sized;
}

impl ModelValidation for LinearRegression {
    type Error = SklearsError;

    fn validate_config(self) -> std::result::Result<Self, Self::Error> {
        // Add validation logic here
        match self.config().penalty {
            Penalty::L1(_) | Penalty::ElasticNet { .. } => {
                if matches!(self.config().solver, Solver::Normal) {
                    return Err(SklearsError::InvalidInput(
                        "Normal equations solver does not support L1 regularization. Use CoordinateDescent or other iterative solver.".to_string()
                    ));
                }
            }
            _ => {}
        }

        if self.config().max_iter == 0 {
            return Err(SklearsError::InvalidInput(
                "max_iter must be greater than 0".to_string(),
            ));
        }

        Ok(self)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_enhanced_linear_regression_builder_presets() {
        let quick_model = EnhancedLinearRegressionBuilder::with_preset(ModelPreset::Quick)
            .build()
            .unwrap();
        assert_eq!(quick_model.config().solver, Solver::Normal);

        let balanced_model = EnhancedLinearRegressionBuilder::with_preset(ModelPreset::Balanced)
            .build()
            .unwrap();
        assert_eq!(balanced_model.config().solver, Solver::Auto);

        let production_model =
            EnhancedLinearRegressionBuilder::with_preset(ModelPreset::Production)
                .build()
                .unwrap();
        assert!(matches!(
            production_model.config().penalty,
            Penalty::ElasticNet { .. }
        ));
    }

    #[test]
    #[cfg(feature = "logistic-regression")]
    fn test_enhanced_logistic_regression_builder_presets() {
        let quick_model = EnhancedLogisticRegressionBuilder::with_preset(ModelPreset::Quick)
            .build()
            .unwrap();
        assert_eq!(quick_model.config().solver, Solver::Lbfgs);

        let robust_model = EnhancedLogisticRegressionBuilder::with_preset(ModelPreset::Robust)
            .build()
            .unwrap();
        assert_eq!(robust_model.config().solver, Solver::Saga);
    }

    #[test]
    fn test_builder_method_chaining() {
        let model = EnhancedLinearRegressionBuilder::new()
            .solver(Solver::CoordinateDescent)
            .penalty(Penalty::L1(0.5))
            .max_iter(2000)
            .fit_intercept(false)
            .with_cross_validation(5)
            .with_early_stopping()
            .build()
            .unwrap();

        assert_eq!(model.config().solver, Solver::CoordinateDescent);
        assert!(matches!(model.config().penalty, Penalty::L1(_)));
        assert_eq!(model.config().max_iter, 2000);
        assert!(!model.config().fit_intercept);
    }

    #[test]
    #[cfg(feature = "logistic-regression")]
    fn test_fluent_api() {
        let builder = EnhancedLogisticRegressionBuilder::new()
            .penalty(Penalty::L2(2.0))
            .solver(Solver::Saga)
            .max_iter(1500)
            .tolerance(1e-5)
            .random_state(42);

        assert!(matches!(builder.config().penalty, Penalty::L2(_)));
        assert_eq!(builder.config().solver, Solver::Saga);
        assert_eq!(builder.config().max_iter, 1500);
        assert_eq!(builder.config().random_state, Some(42));
    }

    #[test]
    fn test_configuration_validation() {
        // Test that L1 penalty with Normal solver fails validation
        let result = EnhancedLinearRegressionBuilder::new()
            .solver(Solver::Normal)
            .penalty(Penalty::L1(1.0))
            .build();

        assert!(result.is_err());
    }

    #[test]
    fn test_preset_configurations_differ() {
        let quick = EnhancedLinearRegressionBuilder::with_preset(ModelPreset::Quick);
        let production = EnhancedLinearRegressionBuilder::with_preset(ModelPreset::Production);

        assert_ne!(quick.config().max_iter, production.config().max_iter);
        assert_ne!(
            quick.validation_config().cross_validation_folds,
            production.validation_config().cross_validation_folds
        );
    }
}
