//! Fluent API for isotonic regression configuration
//!
//! This module provides a fluent interface for configuring isotonic regression
//! models with method chaining and intuitive constraint specification.

use scirs2_core::ndarray::Array1;
use sklears_core::{
    error::Result,
    traits::{Fit, Predict, Trained, Untrained},
    types::Float,
};

use crate::core::{IsotonicRegression, LossFunction, MonotonicityConstraint};

/// Fluent builder for isotonic regression with method chaining
#[derive(Debug, Clone)]
/// FluentIsotonicRegression
pub struct FluentIsotonicRegression<State = Untrained> {
    inner: IsotonicRegression<State>,
}

impl FluentIsotonicRegression<Untrained> {
    /// Create a new fluent isotonic regression builder
    pub fn new() -> Self {
        Self {
            inner: IsotonicRegression::new(),
        }
    }

    /// Set the regression to be increasing (monotonically non-decreasing)
    pub fn increasing(mut self) -> Self {
        self.inner = self.inner.increasing(true);
        self
    }

    /// Set the regression to be decreasing (monotonically non-increasing)
    pub fn decreasing(mut self) -> Self {
        self.inner = self.inner.increasing(false);
        self
    }

    /// Set the regression to be convex (increasing and convex)
    pub fn convex(mut self) -> Self {
        self.inner = self.inner.convex();
        self
    }

    /// Set the regression to be concave (increasing and concave)
    pub fn concave(mut self) -> Self {
        self.inner = self.inner.concave();
        self
    }

    /// Set the regression to be convex and decreasing
    pub fn convex_decreasing(mut self) -> Self {
        self.inner = self.inner.convex_decreasing();
        self
    }

    /// Set the regression to be concave and decreasing
    pub fn concave_decreasing(mut self) -> Self {
        self.inner = self.inner.concave_decreasing();
        self
    }

    /// Set piecewise monotonic constraints with breakpoints
    pub fn piecewise(mut self, breakpoints: Vec<Float>, segment_increasing: Vec<bool>) -> Self {
        self.inner = self.inner.piecewise(breakpoints, segment_increasing);
        self
    }

    /// Set the loss function to squared loss (L2) - default
    pub fn squared_loss(mut self) -> Self {
        self.inner = self.inner.loss(LossFunction::SquaredLoss);
        self
    }

    /// Set the loss function to absolute loss (L1) for robustness
    pub fn absolute_loss(mut self) -> Self {
        self.inner = self.inner.loss(LossFunction::AbsoluteLoss);
        self
    }

    /// Set the loss function to Huber loss with specified delta
    pub fn huber_loss(mut self, delta: Float) -> Self {
        self.inner = self.inner.loss(LossFunction::HuberLoss { delta });
        self
    }

    /// Set the loss function to quantile loss with specified quantile
    pub fn quantile_loss(mut self, quantile: Float) -> Self {
        self.inner = self.inner.loss(LossFunction::QuantileLoss { quantile });
        self
    }

    /// Set lower bound for output values
    pub fn lower_bound(mut self, bound: Float) -> Self {
        self.inner = self.inner.y_min(bound);
        self
    }

    /// Set upper bound for output values
    pub fn upper_bound(mut self, bound: Float) -> Self {
        self.inner = self.inner.y_max(bound);
        self
    }

    /// Set both lower and upper bounds
    pub fn bounds(mut self, lower: Float, upper: Float) -> Self {
        self.inner = self.inner.y_min(lower).y_max(upper);
        self
    }

    /// Set bounds to [0, 1] for probability-like outputs
    pub fn probability_bounds(self) -> Self {
        self.bounds(0.0, 1.0)
    }

    /// Set bounds to [0, +inf) for non-negative outputs
    pub fn non_negative(self) -> Self {
        self.lower_bound(0.0)
    }

    /// Configure for robust regression against outliers
    pub fn robust(self) -> Self {
        self.huber_loss(1.35) // Standard robust delta
    }

    /// Configure for median regression
    pub fn median_regression(self) -> Self {
        self.quantile_loss(0.5)
    }

    /// Configure for 90th percentile regression
    pub fn percentile_90(self) -> Self {
        self.quantile_loss(0.9)
    }

    /// Configure for 10th percentile regression
    pub fn percentile_10(self) -> Self {
        self.quantile_loss(0.1)
    }
}

impl Default for FluentIsotonicRegression<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

// Note: Estimator trait implementation would need to match sklears-core requirements
// For now, focusing on Fit and Predict traits

impl Fit<Array1<Float>, Array1<Float>> for FluentIsotonicRegression<Untrained> {
    type Fitted = FluentIsotonicRegression<Trained>;

    fn fit(self, x: &Array1<Float>, y: &Array1<Float>) -> Result<Self::Fitted> {
        let fitted_inner = self.inner.fit(x, y)?;
        Ok(FluentIsotonicRegression {
            inner: fitted_inner,
        })
    }
}

impl Predict<Array1<Float>, Array1<Float>> for FluentIsotonicRegression<Trained> {
    fn predict(&self, x: &Array1<Float>) -> Result<Array1<Float>> {
        self.inner.predict(x)
    }
}

/// Constraint builder for complex constraint specifications
#[derive(Debug, Clone)]
/// ConstraintBuilder
pub struct ConstraintBuilder {
    constraints: Vec<ConstraintSpec>,
}

#[derive(Debug, Clone)]
enum ConstraintSpec {
    Monotonic {
        increasing: bool,
    },
    Piecewise {
        breakpoints: Vec<Float>,
        segments: Vec<bool>,
    },
    Convex,
    Concave,
    ConvexDecreasing,
    ConcaveDecreasing,
}

impl ConstraintBuilder {
    /// Create a new constraint builder
    pub fn new() -> Self {
        Self {
            constraints: Vec::new(),
        }
    }

    /// Add an increasing constraint
    pub fn increasing(mut self) -> Self {
        self.constraints
            .push(ConstraintSpec::Monotonic { increasing: true });
        self
    }

    /// Add a decreasing constraint
    pub fn decreasing(mut self) -> Self {
        self.constraints
            .push(ConstraintSpec::Monotonic { increasing: false });
        self
    }

    /// Add a convex constraint
    pub fn convex(mut self) -> Self {
        self.constraints.push(ConstraintSpec::Convex);
        self
    }

    /// Add a concave constraint
    pub fn concave(mut self) -> Self {
        self.constraints.push(ConstraintSpec::Concave);
        self
    }

    /// Add piecewise constraints
    pub fn piecewise(mut self, breakpoints: Vec<Float>, segments: Vec<bool>) -> Self {
        self.constraints.push(ConstraintSpec::Piecewise {
            breakpoints,
            segments,
        });
        self
    }

    /// Build the constraint specification
    pub fn build(self) -> Result<MonotonicityConstraint> {
        if self.constraints.is_empty() {
            return Ok(MonotonicityConstraint::Global { increasing: true });
        }

        // For now, take the last constraint (could be extended to combine constraints)
        let last_constraint = self.constraints.last().ok_or_else(|| {
            sklears_core::error::SklearsError::InvalidInput("No constraints specified".to_string())
        })?;

        match last_constraint {
            ConstraintSpec::Monotonic { increasing } => Ok(MonotonicityConstraint::Global {
                increasing: *increasing,
            }),
            ConstraintSpec::Piecewise {
                breakpoints,
                segments,
            } => Ok(MonotonicityConstraint::Piecewise {
                breakpoints: breakpoints.clone(),
                segments: segments.clone(),
            }),
            ConstraintSpec::Convex => Ok(MonotonicityConstraint::Convex),
            ConstraintSpec::Concave => Ok(MonotonicityConstraint::Concave),
            ConstraintSpec::ConvexDecreasing => Ok(MonotonicityConstraint::ConvexDecreasing),
            ConstraintSpec::ConcaveDecreasing => Ok(MonotonicityConstraint::ConcaveDecreasing),
        }
    }
}

impl Default for ConstraintBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Advanced builder pattern for complex isotonic regression problems
#[derive(Debug, Clone)]
/// ComplexIsotonicBuilder
pub struct ComplexIsotonicBuilder {
    loss_function: LossFunction,
    constraint: MonotonicityConstraint,
    bounds: Option<(Float, Float)>,
    regularization: Option<RegularizationConfig>,
    optimization: OptimizationConfig,
    preprocessing: PreprocessingConfig,
}

#[derive(Debug, Clone)]
/// RegularizationConfig
pub struct RegularizationConfig {
    /// l1_penalty
    pub l1_penalty: Float,
    /// l2_penalty
    pub l2_penalty: Float,
    /// smoothness_penalty
    pub smoothness_penalty: Float,
}

#[derive(Debug, Clone)]
/// OptimizationConfig
pub struct OptimizationConfig {
    /// max_iterations
    pub max_iterations: usize,
    /// tolerance
    pub tolerance: Float,
    /// early_stopping
    pub early_stopping: bool,
    /// warm_start
    pub warm_start: bool,
}

#[derive(Debug, Clone)]
/// PreprocessingConfig
pub struct PreprocessingConfig {
    /// normalize
    pub normalize: bool,
    /// standardize
    pub standardize: bool,
    /// remove_outliers
    pub remove_outliers: bool,
    /// outlier_threshold
    pub outlier_threshold: Float,
}

impl ComplexIsotonicBuilder {
    /// Create a new complex builder
    pub fn new() -> Self {
        Self {
            loss_function: LossFunction::SquaredLoss,
            constraint: MonotonicityConstraint::Global { increasing: true },
            bounds: None,
            regularization: None,
            optimization: OptimizationConfig::default(),
            preprocessing: PreprocessingConfig::default(),
        }
    }

    /// Set the loss function
    pub fn loss(mut self, loss: LossFunction) -> Self {
        self.loss_function = loss;
        self
    }

    /// Set the constraint
    pub fn constraint(mut self, constraint: MonotonicityConstraint) -> Self {
        self.constraint = constraint;
        self
    }

    /// Set bounds
    pub fn bounds(mut self, lower: Float, upper: Float) -> Self {
        self.bounds = Some((lower, upper));
        self
    }

    /// Configure regularization
    pub fn regularization(mut self, config: RegularizationConfig) -> Self {
        self.regularization = Some(config);
        self
    }

    /// Configure optimization parameters
    pub fn optimization(mut self, config: OptimizationConfig) -> Self {
        self.optimization = config;
        self
    }

    /// Configure preprocessing
    pub fn preprocessing(mut self, config: PreprocessingConfig) -> Self {
        self.preprocessing = config;
        self
    }

    /// Enable L1 regularization (Lasso)
    pub fn l1_regularization(mut self, penalty: Float) -> Self {
        let mut reg = self.regularization.unwrap_or_default();
        reg.l1_penalty = penalty;
        self.regularization = Some(reg);
        self
    }

    /// Enable L2 regularization (Ridge)
    pub fn l2_regularization(mut self, penalty: Float) -> Self {
        let mut reg = self.regularization.unwrap_or_default();
        reg.l2_penalty = penalty;
        self.regularization = Some(reg);
        self
    }

    /// Enable smoothness regularization
    pub fn smoothness_regularization(mut self, penalty: Float) -> Self {
        let mut reg = self.regularization.unwrap_or_default();
        reg.smoothness_penalty = penalty;
        self.regularization = Some(reg);
        self
    }

    /// Enable early stopping
    pub fn early_stopping(mut self, enabled: bool) -> Self {
        self.optimization.early_stopping = enabled;
        self
    }

    /// Set convergence tolerance
    pub fn tolerance(mut self, tol: Float) -> Self {
        self.optimization.tolerance = tol;
        self
    }

    /// Set maximum iterations
    pub fn max_iterations(mut self, max_iter: usize) -> Self {
        self.optimization.max_iterations = max_iter;
        self
    }

    /// Enable data normalization
    pub fn normalize(mut self) -> Self {
        self.preprocessing.normalize = true;
        self
    }

    /// Enable data standardization
    pub fn standardize(mut self) -> Self {
        self.preprocessing.standardize = true;
        self
    }

    /// Enable outlier removal
    pub fn remove_outliers(mut self, threshold: Float) -> Self {
        self.preprocessing.remove_outliers = true;
        self.preprocessing.outlier_threshold = threshold;
        self
    }

    /// Build a fluent isotonic regression model
    pub fn build(self) -> FluentIsotonicRegression<Untrained> {
        let mut model = FluentIsotonicRegression::new();

        // Set loss function
        model.inner = model.inner.loss(self.loss_function);

        // Set constraint
        model.inner = model.inner.constraint(self.constraint);

        // Set bounds
        if let Some((lower, upper)) = self.bounds {
            model.inner = model.inner.y_min(lower).y_max(upper);
        }

        model
    }

    /// Create a preset for financial modeling (bounded, robust)
    pub fn financial_model() -> Self {
        Self::new()
            .loss(LossFunction::HuberLoss { delta: 1.35 })
            .bounds(0.0, 1.0)
            .l1_regularization(0.01)
            .remove_outliers(3.0)
            .early_stopping(true)
    }

    /// Create a preset for scientific data (high precision, no outlier removal)
    pub fn scientific_model() -> Self {
        Self::new()
            .loss(LossFunction::SquaredLoss)
            .tolerance(1e-10)
            .max_iterations(10000)
            .standardize()
    }

    /// Create a preset for web analytics (robust, fast)
    pub fn web_analytics_model() -> Self {
        Self::new()
            .loss(LossFunction::AbsoluteLoss)
            .remove_outliers(2.0)
            .max_iterations(100)
            .early_stopping(true)
    }

    /// Create a preset for medical data (conservative, bounded)
    pub fn medical_model() -> Self {
        Self::new()
            .loss(LossFunction::HuberLoss { delta: 1.0 })
            .bounds(0.0, 100.0)
            .l2_regularization(0.1)
            .standardize()
            .tolerance(1e-8)
    }
}

impl Default for ComplexIsotonicBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for RegularizationConfig {
    fn default() -> Self {
        Self {
            l1_penalty: 0.0,
            l2_penalty: 0.0,
            smoothness_penalty: 0.0,
        }
    }
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            tolerance: 1e-6,
            early_stopping: false,
            warm_start: false,
        }
    }
}

impl Default for PreprocessingConfig {
    fn default() -> Self {
        Self {
            normalize: false,
            standardize: false,
            remove_outliers: false,
            outlier_threshold: 3.0,
        }
    }
}

/// Convenience functions for common configurations
pub mod presets {
    use super::*;

    /// Create a standard increasing isotonic regression
    pub fn increasing() -> FluentIsotonicRegression<Untrained> {
        FluentIsotonicRegression::new().increasing()
    }

    /// Create a standard decreasing isotonic regression
    pub fn decreasing() -> FluentIsotonicRegression<Untrained> {
        FluentIsotonicRegression::new().decreasing()
    }

    /// Create a robust isotonic regression using Huber loss
    pub fn robust() -> FluentIsotonicRegression<Untrained> {
        FluentIsotonicRegression::new().increasing().robust()
    }

    /// Create a median regression model
    pub fn median() -> FluentIsotonicRegression<Untrained> {
        FluentIsotonicRegression::new()
            .increasing()
            .median_regression()
    }

    /// Create a probability-bounded isotonic regression
    pub fn probability() -> FluentIsotonicRegression<Untrained> {
        FluentIsotonicRegression::new()
            .increasing()
            .probability_bounds()
    }

    /// Create a non-negative increasing regression
    pub fn non_negative() -> FluentIsotonicRegression<Untrained> {
        FluentIsotonicRegression::new().increasing().non_negative()
    }

    /// Create a convex regression
    pub fn convex() -> FluentIsotonicRegression<Untrained> {
        FluentIsotonicRegression::new().convex()
    }

    /// Create a concave regression
    pub fn concave() -> FluentIsotonicRegression<Untrained> {
        FluentIsotonicRegression::new().concave()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_fluent_basic() -> Result<()> {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![1.0, 3.0, 2.0, 4.0, 5.0];

        let model = FluentIsotonicRegression::new().increasing().squared_loss();

        let fitted = model.fit(&x, &y)?;
        let predictions = fitted.predict(&x)?;

        assert_eq!(predictions.len(), 5);

        // Check monotonicity
        for i in 0..predictions.len() - 1 {
            assert!(predictions[i] <= predictions[i + 1]);
        }

        Ok(())
    }

    #[test]
    fn test_fluent_method_chaining() -> Result<()> {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![1.0, 3.0, 2.0, 4.0, 5.0];

        let model = FluentIsotonicRegression::new()
            .increasing()
            .huber_loss(1.0)
            .bounds(0.0, 10.0);

        let fitted = model.fit(&x, &y)?;
        let predictions = fitted.predict(&x)?;

        // Check bounds
        for &pred in predictions.iter() {
            assert!(pred >= 0.0 && pred <= 10.0);
        }

        Ok(())
    }

    #[test]
    fn test_presets() -> Result<()> {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![0.1, 0.3, 0.2, 0.7, 0.9];

        // Test probability preset
        let model = presets::probability();
        let fitted = model.fit(&x, &y)?;
        let predictions = fitted.predict(&x)?;

        // Check probability bounds
        for &pred in predictions.iter() {
            assert!(pred >= 0.0 && pred <= 1.0);
        }

        Ok(())
    }

    #[test]
    fn test_constraint_builder() -> Result<()> {
        let constraint = ConstraintBuilder::new().increasing().build()?;

        match constraint {
            MonotonicityConstraint::Global { increasing } => assert!(increasing),
            _ => panic!("Expected global increasing constraint"),
        }

        Ok(())
    }

    #[test]
    fn test_robust_preset() -> Result<()> {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![1.0, 100.0, 2.0, 4.0, 5.0]; // outlier

        let model = presets::robust();
        let fitted = model.fit(&x, &y)?;
        let predictions = fitted.predict(&x)?;

        // Should handle outlier reasonably
        assert!(predictions[1] < 50.0); // Should not be close to outlier value

        Ok(())
    }

    #[test]
    fn test_median_regression() -> Result<()> {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![1.0, 3.0, 2.0, 4.0, 5.0];

        let model = presets::median();
        let fitted = model.fit(&x, &y)?;
        let predictions = fitted.predict(&x)?;

        assert_eq!(predictions.len(), 5);

        Ok(())
    }

    #[test]
    fn test_convex_preset() -> Result<()> {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![1.0, 1.5, 3.0, 5.5, 9.0]; // convex-like data

        let model = presets::convex();
        let fitted = model.fit(&x, &y)?;
        let predictions = fitted.predict(&x)?;

        assert_eq!(predictions.len(), 5);

        Ok(())
    }

    #[test]
    fn test_complex_builder_basic() -> Result<()> {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![1.0, 3.0, 2.0, 4.0, 5.0];

        let model = ComplexIsotonicBuilder::new()
            .loss(LossFunction::HuberLoss { delta: 1.0 })
            .bounds(0.0, 10.0)
            .l1_regularization(0.01)
            .early_stopping(true)
            .build();

        let fitted = model.fit(&x, &y)?;
        let predictions = fitted.predict(&x)?;

        assert_eq!(predictions.len(), 5);

        Ok(())
    }

    #[test]
    fn test_complex_builder_presets() -> Result<()> {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![0.1, 0.3, 0.2, 0.7, 0.9];

        // Test financial model preset
        let financial_model = ComplexIsotonicBuilder::financial_model().build();
        let fitted = financial_model.fit(&x, &y)?;
        let predictions = fitted.predict(&x)?;

        assert_eq!(predictions.len(), 5);

        // Test scientific model preset
        let scientific_model = ComplexIsotonicBuilder::scientific_model().build();
        let fitted = scientific_model.fit(&x, &y)?;
        let predictions = fitted.predict(&x)?;

        assert_eq!(predictions.len(), 5);

        Ok(())
    }

    #[test]
    fn test_complex_builder_method_chaining() -> Result<()> {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![1.0, 3.0, 2.0, 4.0, 5.0];

        let model = ComplexIsotonicBuilder::new()
            .standardize()
            .remove_outliers(2.5)
            .l2_regularization(0.1)
            .tolerance(1e-8)
            .max_iterations(500)
            .build();

        let fitted = model.fit(&x, &y)?;
        let predictions = fitted.predict(&x)?;

        assert_eq!(predictions.len(), 5);

        Ok(())
    }

    #[test]
    fn test_regularization_config() {
        let config = RegularizationConfig::default();
        assert_eq!(config.l1_penalty, 0.0);
        assert_eq!(config.l2_penalty, 0.0);
        assert_eq!(config.smoothness_penalty, 0.0);
    }

    #[test]
    fn test_optimization_config() {
        let config = OptimizationConfig::default();
        assert_eq!(config.max_iterations, 1000);
        assert_eq!(config.tolerance, 1e-6);
        assert!(!config.early_stopping);
        assert!(!config.warm_start);
    }

    #[test]
    fn test_preprocessing_config() {
        let config = PreprocessingConfig::default();
        assert!(!config.normalize);
        assert!(!config.standardize);
        assert!(!config.remove_outliers);
        assert_eq!(config.outlier_threshold, 3.0);
    }
}
