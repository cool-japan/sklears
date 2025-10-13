//! Type-safe isotonic regression with phantom types and const generics
//!
//! This module provides compile-time guarantees for monotonicity constraints,
//! zero-cost abstractions, and type-safe optimization operations.

use crate::core::{isotonic_regression, IsotonicRegression, LossFunction};
use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::{prelude::SklearsError, traits::Fit, types::Float};
use std::marker::PhantomData;

/// Phantom type for increasing monotonicity
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Increasing
pub struct Increasing;

/// Phantom type for decreasing monotonicity
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Decreasing
pub struct Decreasing;

/// Phantom type for no monotonicity constraint
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// NoConstraint
pub struct NoConstraint;

/// Trait for monotonicity constraint types
pub trait MonotonicityType {
    /// Whether this constraint enforces increasing monotonicity
    const IS_INCREASING: Option<bool>;

    /// Name of the constraint type for error messages
    const NAME: &'static str;
}

impl MonotonicityType for Increasing {
    const IS_INCREASING: Option<bool> = Some(true);
    const NAME: &'static str = "Increasing";
}

impl MonotonicityType for Decreasing {
    const IS_INCREASING: Option<bool> = Some(false);
    const NAME: &'static str = "Decreasing";
}

impl MonotonicityType for NoConstraint {
    const IS_INCREASING: Option<bool> = None;
    const NAME: &'static str = "NoConstraint";
}

/// Phantom type for fitted state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Fitted
pub struct Fitted;

/// Phantom type for unfitted state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Unfitted
pub struct Unfitted;

/// Trait for fitting state types
pub trait FittingState {
    /// Whether this state allows prediction
    const CAN_PREDICT: bool;

    /// Name of the state for error messages
    const NAME: &'static str;
}

impl FittingState for Fitted {
    const CAN_PREDICT: bool = true;
    const NAME: &'static str = "Fitted";
}

impl FittingState for Unfitted {
    const CAN_PREDICT: bool = false;
    const NAME: &'static str = "Unfitted";
}

/// Type-safe isotonic regression with phantom types
#[derive(Debug, Clone)]
/// TypeSafeIsotonicRegression
pub struct TypeSafeIsotonicRegression<M, S, const N: usize = 0>
where
    M: MonotonicityType,
    S: FittingState,
{
    /// Loss function
    loss: LossFunction,
    /// Lower bound constraint
    y_min: Option<Float>,
    /// Upper bound constraint
    y_max: Option<Float>,
    /// Fitted values (only available in Fitted state)
    fitted_values: Option<Array1<Float>>,
    /// Fitted x values (only available in Fitted state)
    fitted_x: Option<Array1<Float>>,
    /// Phantom data for type safety
    _phantom: PhantomData<(M, S)>,
}

impl<M: MonotonicityType> TypeSafeIsotonicRegression<M, Unfitted> {
    /// Create a new unfitted isotonic regression model
    pub fn new() -> Self {
        Self {
            loss: LossFunction::SquaredLoss,
            y_min: None,
            y_max: None,
            fitted_values: None,
            fitted_x: None,
            _phantom: PhantomData,
        }
    }

    /// Set the loss function
    pub fn loss(mut self, loss: LossFunction) -> Self {
        self.loss = loss;
        self
    }

    /// Set lower bound constraint
    pub fn y_min(mut self, y_min: Float) -> Self {
        self.y_min = Some(y_min);
        self
    }

    /// Set upper bound constraint
    pub fn y_max(mut self, y_max: Float) -> Self {
        self.y_max = Some(y_max);
        self
    }

    /// Simple linear interpolation for fitting stage
    fn interpolate_simple(
        &self,
        x: Float,
        fitted_x: &Array1<Float>,
        fitted_values: &Array1<Float>,
    ) -> Result<Float, SklearsError> {
        if fitted_x.is_empty() || fitted_values.is_empty() {
            return Err(SklearsError::InvalidInput("No fitted data".to_string()));
        }

        let n = fitted_x.len().min(fitted_values.len());

        // Handle single point case
        if n == 0 {
            return Err(SklearsError::InvalidInput("No data points".to_string()));
        }
        if n == 1 {
            return Ok(fitted_values[0]);
        }

        // Handle boundary cases
        if x <= fitted_x[0] {
            return Ok(fitted_values[0]);
        }
        if x >= fitted_x[n - 1] {
            return Ok(fitted_values[n - 1]);
        }

        // Find interpolation interval
        for i in 0..n - 1 {
            if x >= fitted_x[i] && x <= fitted_x[i + 1] {
                if fitted_x[i + 1] == fitted_x[i] {
                    return Ok(fitted_values[i]);
                }
                let t = (x - fitted_x[i]) / (fitted_x[i + 1] - fitted_x[i]);
                return Ok(fitted_values[i] + t * (fitted_values[i + 1] - fitted_values[i]));
            }
        }

        Ok(fitted_values[n - 1])
    }

    /// Fit the model, transitioning to fitted state
    pub fn fit(
        self,
        x: &Array1<Float>,
        y: &Array1<Float>,
    ) -> Result<TypeSafeIsotonicRegression<M, Fitted>, SklearsError> {
        if x.len() != y.len() {
            return Err(SklearsError::ShapeMismatch {
                expected: format!("{}", x.len()),
                actual: format!("{}", y.len()),
            });
        }

        if x.is_empty() {
            return Err(SklearsError::InvalidInput("Empty input data".to_string()));
        }

        // Sort data by x values
        let mut data: Vec<(Float, Float)> =
            x.iter().zip(y.iter()).map(|(&xi, &yi)| (xi, yi)).collect();
        data.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        let sorted_x: Array1<Float> = Array1::from_vec(data.iter().map(|(xi, _)| *xi).collect());
        let sorted_y: Array1<Float> = Array1::from_vec(data.iter().map(|(_, yi)| *yi).collect());

        // Apply isotonic regression with type-safe monotonicity
        // Use the full IsotonicRegression struct to get both fitted x and y values
        let mut iso = IsotonicRegression::new();
        iso = iso.loss(self.loss);

        if let Some(inc) = M::IS_INCREASING {
            iso = iso.increasing(inc);
        }
        if let Some(min_val) = self.y_min {
            iso = iso.y_min(min_val);
        }
        if let Some(max_val) = self.y_max {
            iso = iso.y_max(max_val);
        }

        let fitted = iso.fit(&sorted_x, &sorted_y)?;
        let pava_fitted_x = fitted.fitted_x();
        let pava_fitted_y = fitted.fitted_y();

        // For type-safe isotonic regression, we need fitted_x and fitted_values to have the same length as input
        // So we'll use the original sorted_x and interpolate the fitted values at those points
        let fitted_x = sorted_x.clone();
        let mut fitted_y = Array1::zeros(sorted_x.len());

        for (i, &xi) in sorted_x.iter().enumerate() {
            fitted_y[i] = self.interpolate_simple(xi, pava_fitted_x, pava_fitted_y)?;
        }

        Ok(TypeSafeIsotonicRegression {
            loss: self.loss,
            y_min: self.y_min,
            y_max: self.y_max,
            fitted_values: Some(fitted_y),
            fitted_x: Some(fitted_x),
            _phantom: PhantomData,
        })
    }
}

impl<M: MonotonicityType> TypeSafeIsotonicRegression<M, Fitted> {
    /// Predict values at given points (only available for fitted models)
    pub fn predict(&self, x: &Array1<Float>) -> Result<Array1<Float>, SklearsError> {
        let fitted_x = self.fitted_x.as_ref().unwrap(); // Safe because of type system
        let fitted_values = self.fitted_values.as_ref().unwrap(); // Safe because of type system

        let mut predictions = Array1::zeros(x.len());

        for (i, &xi) in x.iter().enumerate() {
            predictions[i] = self.interpolate(xi, fitted_x, fitted_values)?;
        }

        Ok(predictions)
    }

    /// Get fitted values (only available for fitted models)
    pub fn fitted_values(&self) -> &Array1<Float> {
        self.fitted_values.as_ref().unwrap() // Safe because of type system
    }

    /// Get fitted x values (only available for fitted models)
    pub fn fitted_x(&self) -> &Array1<Float> {
        self.fitted_x.as_ref().unwrap() // Safe because of type system
    }

    /// Convert to different monotonicity constraint (requires refitting)
    pub fn change_constraint<N: MonotonicityType>(self) -> TypeSafeIsotonicRegression<N, Unfitted> {
        TypeSafeIsotonicRegression {
            loss: self.loss,
            y_min: self.y_min,
            y_max: self.y_max,
            fitted_values: None,
            fitted_x: None,
            _phantom: PhantomData,
        }
    }

    /// Linear interpolation for prediction
    fn interpolate(
        &self,
        x: Float,
        fitted_x: &Array1<Float>,
        fitted_values: &Array1<Float>,
    ) -> Result<Float, SklearsError> {
        if fitted_x.is_empty() {
            return Err(SklearsError::InvalidInput("No fitted data".to_string()));
        }

        if fitted_values.is_empty() {
            return Err(SklearsError::InvalidInput("No fitted values".to_string()));
        }

        let n = fitted_x.len().min(fitted_values.len());

        // Handle single point case
        if n == 0 {
            return Err(SklearsError::InvalidInput("No data points".to_string()));
        }
        if n == 1 {
            return Ok(fitted_values[0]);
        }

        // Handle boundary cases
        if x <= fitted_x[0] {
            return Ok(fitted_values[0]);
        }
        if x >= fitted_x[n - 1] {
            return Ok(fitted_values[n - 1]);
        }

        // Find interpolation interval
        for i in 0..n - 1 {
            if x >= fitted_x[i] && x <= fitted_x[i + 1] {
                if fitted_x[i + 1] == fitted_x[i] {
                    return Ok(fitted_values[i]);
                }
                let t = (x - fitted_x[i]) / (fitted_x[i + 1] - fitted_x[i]);
                return Ok(fitted_values[i] + t * (fitted_values[i + 1] - fitted_values[i]));
            }
        }

        Ok(fitted_values[n - 1])
    }
}

/// Type aliases for common configurations
pub type IncreasingIsotonicRegression<S> = TypeSafeIsotonicRegression<Increasing, S>;
pub type DecreasingIsotonicRegression<S> = TypeSafeIsotonicRegression<Decreasing, S>;
pub type UnconstrainedIsotonicRegression<S> = TypeSafeIsotonicRegression<NoConstraint, S>;

/// Fixed-size isotonic regression with const generics
#[derive(Debug, Clone)]
/// FixedSizeIsotonicRegression
pub struct FixedSizeIsotonicRegression<M, S, const N: usize>
where
    M: MonotonicityType,
    S: FittingState,
{
    /// Loss function
    loss: LossFunction,
    /// Lower bound constraint
    y_min: Option<Float>,
    /// Upper bound constraint
    y_max: Option<Float>,
    /// Fitted values (fixed size)
    fitted_values: Option<[Float; N]>,
    /// Fitted x values (fixed size)
    fitted_x: Option<[Float; N]>,
    /// Phantom data for type safety
    _phantom: PhantomData<(M, S)>,
}

impl<M: MonotonicityType, const N: usize> FixedSizeIsotonicRegression<M, Unfitted, N> {
    /// Create a new fixed-size unfitted isotonic regression model
    pub fn new() -> Self {
        Self {
            loss: LossFunction::SquaredLoss,
            y_min: None,
            y_max: None,
            fitted_values: None,
            fitted_x: None,
            _phantom: PhantomData,
        }
    }

    /// Set the loss function
    pub fn loss(mut self, loss: LossFunction) -> Self {
        self.loss = loss;
        self
    }

    /// Set lower bound constraint
    pub fn y_min(mut self, y_min: Float) -> Self {
        self.y_min = Some(y_min);
        self
    }

    /// Set upper bound constraint
    pub fn y_max(mut self, y_max: Float) -> Self {
        self.y_max = Some(y_max);
        self
    }

    /// Linear interpolation for a single point
    fn interpolate_single(
        &self,
        x: Float,
        fitted_x: &Array1<Float>,
        fitted_values: &Array1<Float>,
    ) -> Result<Float, SklearsError> {
        if fitted_x.is_empty() {
            return Err(SklearsError::InvalidInput("No fitted data".to_string()));
        }

        if fitted_values.is_empty() {
            return Err(SklearsError::InvalidInput("No fitted values".to_string()));
        }

        let n = fitted_x.len().min(fitted_values.len());

        // Handle single point case
        if n == 0 {
            return Err(SklearsError::InvalidInput("No data points".to_string()));
        }
        if n == 1 {
            return Ok(fitted_values[0]);
        }

        // Handle boundary cases
        if x <= fitted_x[0] {
            return Ok(fitted_values[0]);
        }
        if x >= fitted_x[n - 1] {
            return Ok(fitted_values[n - 1]);
        }

        // Find interpolation interval
        for i in 0..n - 1 {
            if x >= fitted_x[i] && x <= fitted_x[i + 1] {
                if fitted_x[i + 1] == fitted_x[i] {
                    return Ok(fitted_values[i]);
                }
                let t = (x - fitted_x[i]) / (fitted_x[i + 1] - fitted_x[i]);
                return Ok(fitted_values[i] + t * (fitted_values[i + 1] - fitted_values[i]));
            }
        }

        Ok(fitted_values[n - 1])
    }

    /// Fit the model with fixed-size arrays
    pub fn fit(
        self,
        x: &[Float; N],
        y: &[Float; N],
    ) -> Result<FixedSizeIsotonicRegression<M, Fitted, N>, SklearsError> {
        if N == 0 {
            return Err(SklearsError::InvalidInput(
                "Empty fixed-size array".to_string(),
            ));
        }

        // Convert to Array1 for processing
        let x_array = Array1::from_vec(x.to_vec());
        let y_array = Array1::from_vec(y.to_vec());

        // Sort data by x values
        let mut data: Vec<(Float, Float)> =
            x.iter().zip(y.iter()).map(|(&xi, &yi)| (xi, yi)).collect();
        data.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        let sorted_x: Array1<Float> = Array1::from_vec(data.iter().map(|(xi, _)| *xi).collect());
        let sorted_y: Array1<Float> = Array1::from_vec(data.iter().map(|(_, yi)| *yi).collect());

        // Apply isotonic regression using the full IsotonicRegression struct
        let mut iso = IsotonicRegression::new();
        iso = iso.loss(self.loss);

        if let Some(inc) = M::IS_INCREASING {
            iso = iso.increasing(inc);
        }
        if let Some(min_val) = self.y_min {
            iso = iso.y_min(min_val);
        }
        if let Some(max_val) = self.y_max {
            iso = iso.y_max(max_val);
        }

        let fitted = iso.fit(&sorted_x, &sorted_y)?;
        let pava_fitted_x = fitted.fitted_x();
        let pava_fitted_y = fitted.fitted_y();

        // Convert back to fixed-size arrays using interpolation
        let mut fitted_x_fixed = [0.0; N];
        let mut fitted_y_fixed = [0.0; N];

        for i in 0..N {
            fitted_x_fixed[i] = sorted_x[i];
            // Interpolate to get the fitted y value at this x
            fitted_y_fixed[i] =
                self.interpolate_single(sorted_x[i], pava_fitted_x, pava_fitted_y)?;
        }

        Ok(FixedSizeIsotonicRegression {
            loss: self.loss,
            y_min: self.y_min,
            y_max: self.y_max,
            fitted_values: Some(fitted_y_fixed),
            fitted_x: Some(fitted_x_fixed),
            _phantom: PhantomData,
        })
    }
}

impl<M: MonotonicityType, const N: usize> FixedSizeIsotonicRegression<M, Fitted, N> {
    /// Predict values at given points (fixed-size)
    pub fn predict(&self, x: &[Float; N]) -> Result<[Float; N], SklearsError> {
        let fitted_x = self.fitted_x.as_ref().unwrap(); // Safe because of type system
        let fitted_values = self.fitted_values.as_ref().unwrap(); // Safe because of type system

        let mut predictions = [0.0; N];

        for (i, &xi) in x.iter().enumerate() {
            predictions[i] = self.interpolate(xi, fitted_x, fitted_values)?;
        }

        Ok(predictions)
    }

    /// Get fitted values (fixed-size)
    pub fn fitted_values(&self) -> &[Float; N] {
        self.fitted_values.as_ref().unwrap() // Safe because of type system
    }

    /// Get fitted x values (fixed-size)
    pub fn fitted_x(&self) -> &[Float; N] {
        self.fitted_x.as_ref().unwrap() // Safe because of type system
    }

    /// Linear interpolation for prediction
    fn interpolate(
        &self,
        x: Float,
        fitted_x: &[Float; N],
        fitted_values: &[Float; N],
    ) -> Result<Float, SklearsError> {
        if N == 0 {
            return Err(SklearsError::InvalidInput("No fitted data".to_string()));
        }

        // Handle boundary cases
        if x <= fitted_x[0] {
            return Ok(fitted_values[0]);
        }
        if x >= fitted_x[N - 1] {
            return Ok(fitted_values[N - 1]);
        }

        // Find interpolation interval
        for i in 0..N - 1 {
            if x >= fitted_x[i] && x <= fitted_x[i + 1] {
                let t = (x - fitted_x[i]) / (fitted_x[i + 1] - fitted_x[i]);
                return Ok(fitted_values[i] + t * (fitted_values[i + 1] - fitted_values[i]));
            }
        }

        Ok(fitted_values[N - 1])
    }
}

/// Zero-cost constraint validator
#[derive(Debug, Clone)]
/// ConstraintValidator
pub struct ConstraintValidator<M: MonotonicityType> {
    _phantom: PhantomData<M>,
}

impl<M: MonotonicityType> ConstraintValidator<M> {
    /// Create a new constraint validator
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }

    /// Validate monotonicity constraints at compile time
    pub fn validate(&self, values: &Array1<Float>) -> Result<(), SklearsError> {
        if let Some(increasing) = M::IS_INCREASING {
            for i in 0..values.len() - 1 {
                if increasing && values[i] > values[i + 1] {
                    return Err(SklearsError::InvalidInput(format!(
                        "Increasing constraint violated at index {} ({} > {})",
                        i,
                        values[i],
                        values[i + 1]
                    )));
                }
                if !increasing && values[i] < values[i + 1] {
                    return Err(SklearsError::InvalidInput(format!(
                        "Decreasing constraint violated at index {} ({} < {})",
                        i,
                        values[i],
                        values[i + 1]
                    )));
                }
            }
        }
        Ok(())
    }

    /// Check if values satisfy the constraint (compile-time type-safe)
    pub fn check(&self, values: &Array1<Float>) -> bool {
        if let Some(increasing) = M::IS_INCREASING {
            for i in 0..values.len() - 1 {
                if increasing && values[i] > values[i + 1] {
                    return false;
                }
                if !increasing && values[i] < values[i + 1] {
                    return false;
                }
            }
        }
        true
    }
}

/// Type-safe optimization operations
pub trait TypeSafeOptimization<M: MonotonicityType> {
    /// Apply optimization step with constraint preservation
    fn optimize_step(
        &mut self,
        gradient: &Array1<Float>,
        step_size: Float,
    ) -> Result<(), SklearsError>;

    /// Project onto constraint set
    fn project_constraints(&mut self) -> Result<(), SklearsError>;
}

/// Type-safe optimization implementation
#[derive(Debug, Clone)]
/// TypeSafeOptimizer
pub struct TypeSafeOptimizer<M: MonotonicityType> {
    /// Current solution
    solution: Array1<Float>,
    /// Constraint validator
    validator: ConstraintValidator<M>,
}

impl<M: MonotonicityType> TypeSafeOptimizer<M> {
    /// Create a new type-safe optimizer
    pub fn new(initial_solution: Array1<Float>) -> Self {
        Self {
            solution: initial_solution,
            validator: ConstraintValidator::new(),
        }
    }

    /// Get current solution
    pub fn solution(&self) -> &Array1<Float> {
        &self.solution
    }

    /// Set solution (with constraint validation)
    pub fn set_solution(&mut self, solution: Array1<Float>) -> Result<(), SklearsError> {
        self.validator.validate(&solution)?;
        self.solution = solution;
        Ok(())
    }
}

impl<M: MonotonicityType> TypeSafeOptimization<M> for TypeSafeOptimizer<M> {
    fn optimize_step(
        &mut self,
        gradient: &Array1<Float>,
        step_size: Float,
    ) -> Result<(), SklearsError> {
        if gradient.len() != self.solution.len() {
            return Err(SklearsError::ShapeMismatch {
                expected: format!("{}", self.solution.len()),
                actual: format!("{}", gradient.len()),
            });
        }

        // Take gradient step
        self.solution = &self.solution - &(gradient * step_size);

        // Project onto constraints
        self.project_constraints()?;

        Ok(())
    }

    fn project_constraints(&mut self) -> Result<(), SklearsError> {
        // Apply monotonicity constraint using pool-adjacent-violators
        if M::IS_INCREASING.is_some() {
            self.solution = isotonic_regression(
                &Array1::from_vec((0..self.solution.len()).map(|i| i as Float).collect()),
                &self.solution,
                M::IS_INCREASING,
                None,
                None,
            )?;
        }

        // Validate the result
        self.validator.validate(&self.solution)?;

        Ok(())
    }
}

/// Convenience functions for creating type-safe models
pub fn increasing_isotonic_regression() -> TypeSafeIsotonicRegression<Increasing, Unfitted> {
    TypeSafeIsotonicRegression::new()
}

pub fn decreasing_isotonic_regression() -> TypeSafeIsotonicRegression<Decreasing, Unfitted> {
    TypeSafeIsotonicRegression::new()
}

pub fn unconstrained_isotonic_regression() -> TypeSafeIsotonicRegression<NoConstraint, Unfitted> {
    TypeSafeIsotonicRegression::new()
}

/// Convenience functions for fixed-size models
pub fn fixed_size_increasing_isotonic_regression<const N: usize>(
) -> FixedSizeIsotonicRegression<Increasing, Unfitted, N> {
    FixedSizeIsotonicRegression::new()
}

pub fn fixed_size_decreasing_isotonic_regression<const N: usize>(
) -> FixedSizeIsotonicRegression<Decreasing, Unfitted, N> {
    FixedSizeIsotonicRegression::new()
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_type_safe_increasing_isotonic_regression() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![1.0, 3.0, 2.0, 4.0, 5.0]; // Non-monotonic

        let model = increasing_isotonic_regression();
        let fitted_model = model.fit(&x, &y).unwrap();

        let predictions = fitted_model.predict(&x).unwrap();

        // Check that predictions are increasing
        for i in 0..predictions.len() - 1 {
            assert!(predictions[i] <= predictions[i + 1]);
        }

        // Verify type safety - this should not compile:
        // let _ = model.predict(&x); // Error: model is not fitted
    }

    #[test]
    fn test_type_safe_decreasing_isotonic_regression() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![5.0, 3.0, 4.0, 2.0, 1.0]; // Non-monotonic

        let model = decreasing_isotonic_regression();
        let fitted_model = model.fit(&x, &y).unwrap();

        let predictions = fitted_model.predict(&x).unwrap();

        // Check that predictions are decreasing
        for i in 0..predictions.len() - 1 {
            assert!(predictions[i] >= predictions[i + 1]);
        }
    }

    #[test]
    fn test_fixed_size_isotonic_regression() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [1.0, 3.0, 2.0, 4.0, 5.0];

        let model: FixedSizeIsotonicRegression<Increasing, Unfitted, 5> =
            fixed_size_increasing_isotonic_regression();
        let fitted_model = model.fit(&x, &y).unwrap();

        let predictions = fitted_model.predict(&x).unwrap();

        // Check that predictions are increasing
        for i in 0..predictions.len() - 1 {
            assert!(predictions[i] <= predictions[i + 1]);
        }

        // Type safety: arrays must be the same size
        // let wrong_x = [1.0, 2.0, 3.0]; // This would not compile
    }

    #[test]
    fn test_constraint_validator() {
        let validator = ConstraintValidator::<Increasing>::new();

        let increasing_values = array![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!(validator.check(&increasing_values));
        assert!(validator.validate(&increasing_values).is_ok());

        let non_increasing_values = array![1.0, 3.0, 2.0, 4.0, 5.0];
        assert!(!validator.check(&non_increasing_values));
        assert!(validator.validate(&non_increasing_values).is_err());
    }

    #[test]
    fn test_type_safe_optimizer() {
        let initial_solution = array![1.0, 1.0, 1.0, 1.0, 1.0];
        let mut optimizer = TypeSafeOptimizer::<Increasing>::new(initial_solution);

        let gradient = array![0.1, -0.1, 0.2, -0.2, 0.1];
        assert!(optimizer.optimize_step(&gradient, 0.1).is_ok());

        // Solution should still satisfy increasing constraint
        let solution = optimizer.solution();
        for i in 0..solution.len() - 1 {
            assert!(solution[i] <= solution[i + 1]);
        }
    }

    #[test]
    fn test_constraint_type_conversion() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![1.0, 3.0, 2.0, 4.0, 5.0];

        let model = increasing_isotonic_regression();
        let fitted_model = model.fit(&x, &y).unwrap();

        // Convert to decreasing constraint (requires refitting)
        let new_model: TypeSafeIsotonicRegression<Decreasing, Unfitted> =
            fitted_model.change_constraint();

        // Can't predict with unfitted model
        // let _ = new_model.predict(&x); // This would not compile

        // Must refit with new constraint
        let refitted_model = new_model.fit(&x, &y).unwrap();
        let predictions = refitted_model.predict(&x).unwrap();

        // Should now be decreasing
        for i in 0..predictions.len() - 1 {
            assert!(predictions[i] >= predictions[i + 1]);
        }
    }

    #[test]
    fn test_type_safety_compilation() {
        // These tests verify that the type system prevents misuse

        let model = increasing_isotonic_regression();

        // Cannot predict before fitting
        // let x = array![1.0, 2.0, 3.0];
        // let _ = model.predict(&x); // Should not compile

        // Type system enforces correct state transitions
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![1.0, 3.0, 2.0, 4.0, 5.0];

        let fitted_model = model.fit(&x, &y).unwrap();
        let _ = fitted_model.predict(&x).unwrap(); // This should compile
    }

    #[test]
    fn test_monotonicity_type_constants() {
        assert_eq!(Increasing::IS_INCREASING, Some(true));
        assert_eq!(Decreasing::IS_INCREASING, Some(false));
        assert_eq!(NoConstraint::IS_INCREASING, None);

        assert_eq!(Increasing::NAME, "Increasing");
        assert_eq!(Decreasing::NAME, "Decreasing");
        assert_eq!(NoConstraint::NAME, "NoConstraint");
    }

    #[test]
    fn test_fitting_state_constants() {
        assert_eq!(Fitted::CAN_PREDICT, true);
        assert_eq!(Unfitted::CAN_PREDICT, false);

        assert_eq!(Fitted::NAME, "Fitted");
        assert_eq!(Unfitted::NAME, "Unfitted");
    }
}
