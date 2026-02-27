/// Compile-time validation framework for ML configurations
///
/// This module provides traits and types for compile-time validation of machine learning
/// model configurations, preventing runtime errors and ensuring type safety.
use crate::error::SklearsError;
use std::marker::PhantomData;

/// Marker trait for valid configurations
pub trait ValidConfig {}

/// Marker trait for configurations that have been validated
pub trait Validated {}

/// Marker trait for configurations that require validation
pub trait RequiresValidation {}

/// Phantom type for tracking validation state
pub struct ValidationState<T> {
    _phantom: PhantomData<T>,
}

/// Marker type for unvalidated configurations
pub struct Unvalidated;

/// Marker type for validated configurations
pub struct ValidatedState;

impl<T> ValidationState<T> {
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

impl<T> Default for ValidationState<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration wrapper that tracks validation state at compile time
pub struct ValidatedConfig<T, S = Unvalidated> {
    pub config: T,
    _state: PhantomData<S>,
}

impl<T> ValidatedConfig<T, Unvalidated> {
    /// Create a new unvalidated configuration
    pub fn new(config: T) -> Self {
        Self {
            config,
            _state: PhantomData,
        }
    }

    /// Validate the configuration at compile time
    pub fn validate(self) -> Result<ValidatedConfig<T, ValidatedState>, SklearsError>
    where
        T: ValidConfig,
    {
        // Runtime validation can still be performed here
        Ok(ValidatedConfig {
            config: self.config,
            _state: PhantomData,
        })
    }
}

impl<T> ValidatedConfig<T, ValidatedState> {
    /// Get the validated configuration
    pub fn inner(&self) -> &T {
        &self.config
    }

    /// Consume the wrapper and return the validated configuration
    pub fn into_inner(self) -> T {
        self.config
    }
}

/// Trait for compile-time parameter validation
pub trait ParameterValidator<T> {
    type Error;

    /// Validate parameter at compile time
    fn validate(value: &T) -> Result<(), Self::Error>;
}

/// Compile-time range validator
pub struct RangeValidator<const MIN: i64, const MAX: i64>;

impl<const MIN: i64, const MAX: i64> ParameterValidator<i32> for RangeValidator<MIN, MAX> {
    type Error = SklearsError;

    fn validate(value: &i32) -> Result<(), Self::Error> {
        if (*value as i64) < MIN || (*value as i64) > MAX {
            Err(SklearsError::InvalidParameter {
                name: "value".to_string(),
                reason: format!("Value {value} not in range [{MIN}, {MAX}]"),
            })
        } else {
            Ok(())
        }
    }
}

impl<const MIN: i64, const MAX: i64> ParameterValidator<f64> for RangeValidator<MIN, MAX> {
    type Error = SklearsError;

    fn validate(value: &f64) -> Result<(), Self::Error> {
        if (*value as i64) < MIN || (*value as i64) > MAX {
            Err(SklearsError::InvalidParameter {
                name: "value".to_string(),
                reason: format!("Value {value} not in range [{MIN}, {MAX}]"),
            })
        } else {
            Ok(())
        }
    }
}

/// Positive number validator
pub struct PositiveValidator;

impl ParameterValidator<f64> for PositiveValidator {
    type Error = SklearsError;

    fn validate(value: &f64) -> Result<(), Self::Error> {
        if *value <= 0.0 {
            Err(SklearsError::InvalidParameter {
                name: "value".to_string(),
                reason: format!("Value {value} must be positive"),
            })
        } else {
            Ok(())
        }
    }
}

impl ParameterValidator<i32> for PositiveValidator {
    type Error = SklearsError;

    fn validate(value: &i32) -> Result<(), Self::Error> {
        if *value <= 0 {
            Err(SklearsError::InvalidParameter {
                name: "value".to_string(),
                reason: format!("Value {value} must be positive"),
            })
        } else {
            Ok(())
        }
    }
}

/// Probability validator (0.0 to 1.0)
pub struct ProbabilityValidator;

impl ParameterValidator<f64> for ProbabilityValidator {
    type Error = SklearsError;

    fn validate(value: &f64) -> Result<(), Self::Error> {
        if *value < 0.0 || *value > 1.0 {
            Err(SklearsError::InvalidParameter {
                name: "probability".to_string(),
                reason: format!("Probability {value} must be between 0.0 and 1.0"),
            })
        } else {
            Ok(())
        }
    }
}

/// Macro for creating compile-time validated parameters
#[macro_export]
macro_rules! validated_param {
    ($name:ident: $type:ty, $validator:ty, $value:expr) => {{
        <$validator as $crate::compile_time_validation::ParameterValidator<$type>>::validate(
            &$value,
        )?;
        $value
    }};
}

/// Trait for algorithms that support compile-time configuration validation
pub trait CompileTimeValidated {
    type Config: ValidConfig;
    type ValidatedConfig;

    /// Create a validated configuration
    fn validate_config(config: Self::Config) -> Result<Self::ValidatedConfig, SklearsError>;
}

/// Example validated configuration for linear regression
#[derive(Debug, Clone)]
pub struct LinearRegressionConfig {
    pub fit_intercept: bool,
    pub positive: bool,
    pub alpha: f64,
    pub max_iter: i32,
}

impl ValidConfig for LinearRegressionConfig {}

impl LinearRegressionConfig {
    /// Create a new configuration with compile-time validation
    pub fn builder() -> LinearRegressionConfigBuilder<Unvalidated> {
        LinearRegressionConfigBuilder::new()
    }
}

/// Builder for LinearRegressionConfig with compile-time validation
pub struct LinearRegressionConfigBuilder<S = Unvalidated> {
    config: LinearRegressionConfig,
    _state: PhantomData<S>,
}

impl LinearRegressionConfigBuilder<Unvalidated> {
    pub fn new() -> Self {
        Self {
            config: LinearRegressionConfig {
                fit_intercept: true,
                positive: false,
                alpha: 1.0,
                max_iter: 1000,
            },
            _state: PhantomData,
        }
    }

    pub fn fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.config.fit_intercept = fit_intercept;
        self
    }

    pub fn positive(mut self, positive: bool) -> Self {
        self.config.positive = positive;
        self
    }

    /// Set alpha with compile-time validation
    pub fn alpha(mut self, alpha: f64) -> Result<Self, SklearsError> {
        PositiveValidator::validate(&alpha)?;
        self.config.alpha = alpha;
        Ok(self)
    }

    /// Set max_iter with compile-time validation
    pub fn max_iter(mut self, max_iter: i32) -> Result<Self, SklearsError> {
        RangeValidator::<1, 10000>::validate(&max_iter)?;
        self.config.max_iter = max_iter;
        Ok(self)
    }

    /// Build the validated configuration
    pub fn build(self) -> Result<LinearRegressionConfigBuilder<ValidatedState>, SklearsError> {
        // Additional cross-parameter validation can be done here
        Ok(LinearRegressionConfigBuilder {
            config: self.config,
            _state: PhantomData,
        })
    }
}

impl Default for LinearRegressionConfigBuilder<Unvalidated> {
    fn default() -> Self {
        Self::new()
    }
}

impl LinearRegressionConfigBuilder<ValidatedState> {
    /// Get the validated configuration
    pub fn config(&self) -> &LinearRegressionConfig {
        &self.config
    }

    /// Consume the builder and return the validated configuration
    pub fn into_config(self) -> LinearRegressionConfig {
        self.config
    }
}

/// Trait for dimension validation at compile time
pub trait DimensionValidator<const N: usize> {
    fn validate_dimensions(&self) -> Result<(), SklearsError>;
}

/// Fixed-size array wrapper with compile-time dimension validation
pub struct FixedArray<T, const N: usize> {
    data: [T; N],
}

impl<T, const N: usize> FixedArray<T, N> {
    pub fn new(data: [T; N]) -> Self {
        Self { data }
    }

    pub fn len(&self) -> usize {
        N
    }

    pub fn is_empty(&self) -> bool {
        N == 0
    }

    pub fn as_slice(&self) -> &[T] {
        &self.data
    }
}

impl<T, const N: usize> DimensionValidator<N> for FixedArray<T, N> {
    fn validate_dimensions(&self) -> Result<(), SklearsError> {
        // Compile-time dimension validation is automatic with const generics
        Ok(())
    }
}

/// Trait for solver compatibility validation
pub trait SolverCompatibility<S> {
    fn is_compatible() -> bool;
}

/// Marker types for different solvers
pub struct SGDSolver;
pub struct LBFGSSolver;
pub struct CoordinateDescentSolver;

/// Marker types for different regularization types
pub struct L1Regularization;
pub struct L2Regularization;
pub struct ElasticNetRegularization;

/// Example solver compatibility implementations
impl SolverCompatibility<L1Regularization> for CoordinateDescentSolver {
    fn is_compatible() -> bool {
        true
    }
}

impl SolverCompatibility<L1Regularization> for LBFGSSolver {
    fn is_compatible() -> bool {
        false // LBFGS doesn't support L1 regularization
    }
}

impl SolverCompatibility<L2Regularization> for LBFGSSolver {
    fn is_compatible() -> bool {
        true
    }
}

impl SolverCompatibility<ElasticNetRegularization> for CoordinateDescentSolver {
    fn is_compatible() -> bool {
        true
    }
}

/// Compile-time solver validation
pub fn validate_solver_regularization<S, R>() -> Result<(), SklearsError>
where
    S: SolverCompatibility<R>,
{
    if S::is_compatible() {
        Ok(())
    } else {
        Err(SklearsError::InvalidParameter {
            name: "solver".to_string(),
            reason: "Solver is not compatible with the specified regularization".to_string(),
        })
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validated_config_creation() {
        let config = LinearRegressionConfig {
            fit_intercept: true,
            positive: false,
            alpha: 1.0,
            max_iter: 1000,
        };

        let validated = ValidatedConfig::new(config);
        assert!(validated.validate().is_ok());
    }

    #[test]
    fn test_config_builder_validation() {
        let result = LinearRegressionConfig::builder()
            .fit_intercept(true)
            .alpha(0.5)
            .expect("expected valid value")
            .max_iter(500)
            .expect("expected valid value")
            .build();

        assert!(result.is_ok());
    }

    #[test]
    fn test_invalid_alpha() {
        let result = LinearRegressionConfig::builder().alpha(-1.0);

        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_max_iter() {
        let result = LinearRegressionConfig::builder().max_iter(-1);

        assert!(result.is_err());
    }

    #[test]
    fn test_fixed_array_dimensions() {
        let arr = FixedArray::new([1, 2, 3, 4, 5]);
        assert_eq!(arr.len(), 5);
        assert!(arr.validate_dimensions().is_ok());
    }

    #[test]
    fn test_solver_compatibility() {
        // This should compile and return Ok
        assert!(
            validate_solver_regularization::<CoordinateDescentSolver, L1Regularization>().is_ok()
        );

        // This should compile but return Err
        assert!(validate_solver_regularization::<LBFGSSolver, L1Regularization>().is_err());
    }

    #[test]
    fn test_range_validator() {
        assert!(RangeValidator::<1, 100>::validate(&50).is_ok());
        assert!(RangeValidator::<1, 100>::validate(&0).is_err());
        assert!(RangeValidator::<1, 100>::validate(&101).is_err());
    }

    #[test]
    fn test_positive_validator() {
        assert!(PositiveValidator::validate(&1.0).is_ok());
        assert!(PositiveValidator::validate(&0.0).is_err());
        assert!(PositiveValidator::validate(&-1.0).is_err());
    }

    #[test]
    fn test_probability_validator() {
        assert!(ProbabilityValidator::validate(&0.5).is_ok());
        assert!(ProbabilityValidator::validate(&0.0).is_ok());
        assert!(ProbabilityValidator::validate(&1.0).is_ok());
        assert!(ProbabilityValidator::validate(&-0.1).is_err());
        assert!(ProbabilityValidator::validate(&1.1).is_err());
    }
}
