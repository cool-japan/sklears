//! Disciplined Convex Programming Approach for Isotonic Regression
//!
//! This module implements disciplined convex programming (DCP) techniques for isotonic regression,
//! providing a flexible framework for combining different convex objectives and constraints
//! with automated verification and optimization.

use scirs2_core::ndarray::Array1;
use sklears_core::{prelude::SklearsError, types::Float};

/// Disciplined convex programming approach for isotonic regression
///
/// This implementation uses DCP principles to provide a flexible framework
/// for combining different convex objectives and constraints in isotonic regression.
/// It supports various objectives (least squares, LAD, Huber, quantile) and
/// constraints (monotonicity, bounds, smoothness, sparsity).
#[derive(Debug, Clone)]
pub struct DisciplinedConvexIsotonicRegression {
    increasing: bool,
    objective: ConvexObjective,
    constraints: Vec<ConvexConstraint>,
    regularization: Float,
    max_iterations: usize,
    tolerance: Float,
    fitted_values: Option<Array1<Float>>,
    fitted_x: Option<Array1<Float>>,
}

/// Types of convex objectives
///
/// Each objective provides different characteristics:
/// - LeastSquares: Standard L2 loss, differentiable
/// - LeastAbsolute: L1 loss, robust to outliers
/// - Huber: Combination of L2 and L1, robust with smoothness
/// - Quantile: Asymmetric loss for quantile regression
#[derive(Debug, Clone)]
pub enum ConvexObjective {
    /// Least squares (L2 loss)
    LeastSquares,
    /// Least absolute deviations (L1 loss)
    LeastAbsolute,
    /// Huber loss with robustness parameter
    Huber { delta: Float },
    /// Quantile loss with quantile parameter
    Quantile { tau: Float },
}

/// Types of convex constraints
///
/// Constraints can be combined to create complex optimization problems:
/// - Monotonic: Core isotonic constraint
/// - Bounds: Box constraints for feasible regions
/// - Smoothness: Total variation regularization
/// - Sparsity: L1 penalty for sparse solutions
#[derive(Debug, Clone)]
pub enum ConvexConstraint {
    /// Monotonicity constraint (increasing or decreasing)
    Monotonic { increasing: bool },
    /// Bound constraints (box constraints)
    Bounds {
        lower: Option<Float>,

        upper: Option<Float>,
    },
    /// Smoothness constraint (total variation)
    Smoothness { lambda: Float },
    /// Sparsity constraint (L1 penalty)
    Sparsity { lambda: Float },
}

impl DisciplinedConvexIsotonicRegression {
    /// Create a new disciplined convex programming isotonic regression model
    pub fn new() -> Self {
        Self {
            increasing: true,
            objective: ConvexObjective::LeastSquares,
            constraints: vec![ConvexConstraint::Monotonic { increasing: true }],
            regularization: 1e-6,
            max_iterations: 1000,
            tolerance: 1e-8,
            fitted_values: None,
            fitted_x: None,
        }
    }

    /// Set whether the function should be increasing or decreasing
    pub fn increasing(mut self, increasing: bool) -> Self {
        self.increasing = increasing;
        // Update monotonic constraint
        self.constraints = self
            .constraints
            .into_iter()
            .map(|constraint| match constraint {
                ConvexConstraint::Monotonic { .. } => ConvexConstraint::Monotonic { increasing },
                other => other,
            })
            .collect();
        self
    }

    /// Set the objective function
    pub fn objective(mut self, objective: ConvexObjective) -> Self {
        self.objective = objective;
        self
    }

    /// Add a constraint
    pub fn add_constraint(mut self, constraint: ConvexConstraint) -> Self {
        self.constraints.push(constraint);
        self
    }

    /// Set regularization parameter
    pub fn regularization(mut self, regularization: Float) -> Self {
        self.regularization = regularization;
        self
    }

    /// Set maximum iterations
    pub fn max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Set convergence tolerance
    pub fn tolerance(mut self, tolerance: Float) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Fit the disciplined convex programming isotonic regression model
    pub fn fit(&mut self, x: &Array1<Float>, y: &Array1<Float>) -> Result<(), SklearsError> {
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
        data.sort_by(|a, b| a.0.total_cmp(&b.0));

        let sorted_x: Array1<Float> = Array1::from_vec(data.iter().map(|(xi, _)| *xi).collect());
        let sorted_y: Array1<Float> = Array1::from_vec(data.iter().map(|(_, yi)| *yi).collect());

        // Apply disciplined convex programming
        let fitted_y = self.dcp_isotonic_regression(&sorted_x, &sorted_y)?;

        self.fitted_x = Some(sorted_x);
        self.fitted_values = Some(fitted_y);

        Ok(())
    }

    /// Predict values at given points
    pub fn predict(&self, x: &Array1<Float>) -> Result<Array1<Float>, SklearsError> {
        let fitted_x = self
            .fitted_x
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "predict".to_string(),
            })?;

        let fitted_values = self
            .fitted_values
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "predict".to_string(),
            })?;

        let mut predictions = Array1::zeros(x.len());

        for (i, &xi) in x.iter().enumerate() {
            predictions[i] = self.interpolate(xi, fitted_x, fitted_values)?;
        }

        Ok(predictions)
    }

    /// Disciplined convex programming based isotonic regression
    fn dcp_isotonic_regression(
        &self,
        x: &Array1<Float>,
        y: &Array1<Float>,
    ) -> Result<Array1<Float>, SklearsError> {
        let _n = x.len();
        let mut fitted_y = y.clone();

        // Iterative optimization using alternating minimization
        for _iteration in 0..self.max_iterations {
            let old_y = fitted_y.clone();

            // Minimize objective function
            fitted_y = self.minimize_objective(x, y, &fitted_y)?;

            // Project onto constraints
            for constraint in &self.constraints {
                fitted_y = self.project_constraint(&fitted_y, constraint)?;
            }

            // Check convergence
            let change = (&fitted_y - &old_y).map(|x| x.abs()).sum();
            if change < self.tolerance {
                break;
            }
        }

        Ok(fitted_y)
    }

    /// Minimize the objective function
    fn minimize_objective(
        &self,
        _x: &Array1<Float>,
        y: &Array1<Float>,
        current_y: &Array1<Float>,
    ) -> Result<Array1<Float>, SklearsError> {
        match &self.objective {
            ConvexObjective::LeastSquares => {
                // Analytical solution for least squares
                Ok(y.clone())
            }
            ConvexObjective::LeastAbsolute => {
                // Use iteratively reweighted least squares for L1
                let mut result = current_y.clone();
                for _ in 0..10 {
                    let residuals = &result - y;
                    let weights = residuals.map(|&r| 1.0 / (r.abs() + 1e-8));

                    // Weighted least squares update
                    let weighted_sum: Float =
                        weights.iter().zip(y.iter()).map(|(&w, &yi)| w * yi).sum();
                    let weight_sum: Float = weights.sum();

                    if weight_sum > 0.0 {
                        let mean = weighted_sum / weight_sum;
                        result.fill(mean);
                    }
                }
                Ok(result)
            }
            ConvexObjective::Huber { delta } => {
                // Huber loss minimization
                let mut result = current_y.clone();
                for i in 0..result.len() {
                    let residual = result[i] - y[i];
                    if residual.abs() <= *delta {
                        result[i] = y[i];
                    } else {
                        result[i] = y[i] + delta * residual.signum();
                    }
                }
                Ok(result)
            }
            ConvexObjective::Quantile { tau } => {
                // Quantile loss minimization (simplified)
                let mut values: Vec<Float> = y.to_vec();
                values.sort_by(|a, b| a.total_cmp(b));
                let quantile_idx = (tau * values.len() as Float) as usize;
                let quantile_value = values[quantile_idx.min(values.len() - 1)];
                Ok(Array1::from_elem(y.len(), quantile_value))
            }
        }
    }

    /// Project onto a specific constraint
    fn project_constraint(
        &self,
        y: &Array1<Float>,
        constraint: &ConvexConstraint,
    ) -> Result<Array1<Float>, SklearsError> {
        match constraint {
            ConvexConstraint::Monotonic { increasing } => {
                self.project_isotonic_constraint(y, *increasing)
            }
            ConvexConstraint::Bounds { lower, upper } => {
                let mut result = y.clone();
                for i in 0..result.len() {
                    if let Some(lower_bound) = lower {
                        result[i] = result[i].max(*lower_bound);
                    }
                    if let Some(upper_bound) = upper {
                        result[i] = result[i].min(*upper_bound);
                    }
                }
                Ok(result)
            }
            ConvexConstraint::Smoothness { lambda: _ } => {
                // Apply smoothness constraint (simplified as moving average)
                let mut result = y.clone();
                for i in 1..result.len() - 1 {
                    result[i] = (y[i - 1] + 2.0 * y[i] + y[i + 1]) / 4.0;
                }
                Ok(result)
            }
            ConvexConstraint::Sparsity { lambda } => {
                // Apply soft thresholding for sparsity
                Ok(y.map(|&x| {
                    if x.abs() > *lambda {
                        x - lambda * x.signum()
                    } else {
                        0.0
                    }
                }))
            }
        }
    }

    /// Project onto isotonic constraints using Pool Adjacent Violators algorithm
    fn project_isotonic_constraint(
        &self,
        y: &Array1<Float>,
        increasing: bool,
    ) -> Result<Array1<Float>, SklearsError> {
        let n = y.len();
        let mut result = y.clone();

        if increasing {
            // Enforce increasing constraints
            for i in 1..n {
                if result[i] < result[i - 1] {
                    // Pool adjacent violators
                    let mut j = i;
                    let mut sum = result[i - 1] + result[i];
                    let mut count = 2;

                    // Extend pool backwards
                    while j > 1 && result[j - 2] > sum / count as Float {
                        j -= 1;
                        sum += result[j - 1];
                        count += 1;
                    }

                    // Set pooled values
                    let pooled_value = sum / count as Float;
                    for k in (j - 1)..=i {
                        result[k] = pooled_value;
                    }
                }
            }
        } else {
            // Enforce decreasing constraints
            for i in 1..n {
                if result[i] > result[i - 1] {
                    // Pool adjacent violators
                    let mut j = i;
                    let mut sum = result[i - 1] + result[i];
                    let mut count = 2;

                    // Extend pool backwards
                    while j > 1 && result[j - 2] < sum / count as Float {
                        j -= 1;
                        sum += result[j - 1];
                        count += 1;
                    }

                    // Set pooled values
                    let pooled_value = sum / count as Float;
                    for k in (j - 1)..=i {
                        result[k] = pooled_value;
                    }
                }
            }
        }

        Ok(result)
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

        let n = fitted_x.len();

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
                let t = (x - fitted_x[i]) / (fitted_x[i + 1] - fitted_x[i]);
                return Ok(fitted_values[i] + t * (fitted_values[i + 1] - fitted_values[i]));
            }
        }

        Ok(fitted_values[n - 1])
    }

    /// Get the fitted values (for analysis)
    pub fn fitted_values(&self) -> Option<&Array1<Float>> {
        self.fitted_values.as_ref()
    }

    /// Get the fitted x values (for analysis)
    pub fn fitted_x(&self) -> Option<&Array1<Float>> {
        self.fitted_x.as_ref()
    }

    /// Get current objective function
    pub fn get_objective(&self) -> &ConvexObjective {
        &self.objective
    }

    /// Get current constraints
    pub fn get_constraints(&self) -> &[ConvexConstraint] {
        &self.constraints
    }

    /// Get current regularization parameter
    pub fn get_regularization(&self) -> Float {
        self.regularization
    }

    /// Get current tolerance
    pub fn get_tolerance(&self) -> Float {
        self.tolerance
    }

    /// Get maximum iterations setting
    pub fn get_max_iterations(&self) -> usize {
        self.max_iterations
    }

    /// Check if model enforces increasing monotonicity
    pub fn is_increasing(&self) -> bool {
        self.increasing
    }
}

impl Default for DisciplinedConvexIsotonicRegression {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience function for disciplined convex programming isotonic regression
///
/// This function provides a simple interface for one-shot DCP isotonic regression.
///
/// # Arguments
///
/// * `x` - Input features (must be sorted)
/// * `y` - Target values
/// * `increasing` - Whether to enforce increasing monotonicity
/// * `objective` - Convex objective function
/// * `constraints` - Additional convex constraints
///
/// # Returns
///
/// Fitted isotonic values or error
pub fn disciplined_convex_isotonic_regression(
    x: &Array1<Float>,
    y: &Array1<Float>,
    increasing: bool,
    objective: ConvexObjective,
    constraints: Vec<ConvexConstraint>,
) -> Result<Array1<Float>, SklearsError> {
    let mut model = DisciplinedConvexIsotonicRegression::new()
        .increasing(increasing)
        .objective(objective);

    for constraint in constraints {
        model = model.add_constraint(constraint);
    }

    model.fit(x, y)?;
    model
        .fitted_values()
        .ok_or_else(|| SklearsError::NotFitted {
            operation: "disciplined_convex_isotonic_regression".to_string(),
        })
        .map(|v| v.clone())
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_disciplined_convex_least_squares() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![1.0, 3.0, 2.0, 4.0, 5.0]; // Non-monotonic

        let mut model = DisciplinedConvexIsotonicRegression::new()
            .increasing(true)
            .objective(ConvexObjective::LeastSquares);

        assert!(model.fit(&x, &y).is_ok());

        let predictions = model.predict(&x).unwrap();

        // Check that predictions are increasing
        for i in 0..predictions.len() - 1 {
            assert!(predictions[i] <= predictions[i + 1]);
        }
    }

    #[test]
    fn test_disciplined_convex_huber() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![5.0, 3.0, 4.0, 2.0, 1.0]; // Non-monotonic

        let mut model = DisciplinedConvexIsotonicRegression::new()
            .increasing(false)
            .objective(ConvexObjective::Huber { delta: 1.0 });

        assert!(model.fit(&x, &y).is_ok());

        let predictions = model.predict(&x).unwrap();

        // Check that predictions are decreasing
        for i in 0..predictions.len() - 1 {
            assert!(predictions[i] >= predictions[i + 1]);
        }
    }

    #[test]
    fn test_disciplined_convex_with_bounds() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![1.0, 3.0, 2.0, 4.0, 5.0]; // Non-monotonic

        let mut model = DisciplinedConvexIsotonicRegression::new()
            .increasing(true)
            .objective(ConvexObjective::LeastSquares)
            .add_constraint(ConvexConstraint::Bounds {
                lower: Some(0.0),
                upper: Some(10.0),
            });

        assert!(model.fit(&x, &y).is_ok());

        let predictions = model.predict(&x).unwrap();

        // Check that predictions are increasing and within bounds
        for i in 0..predictions.len() - 1 {
            assert!(predictions[i] <= predictions[i + 1]);
            assert!(predictions[i] >= 0.0);
            assert!(predictions[i] <= 10.0);
        }
    }

    #[test]
    fn test_disciplined_convex_convenience_function() {
        let x = array![1.0, 2.0, 3.0, 4.0];
        let y = array![1.5, 1.0, 2.5, 3.0];

        let result = disciplined_convex_isotonic_regression(
            &x,
            &y,
            true,
            ConvexObjective::LeastSquares,
            vec![ConvexConstraint::Monotonic { increasing: true }],
        );
        assert!(result.is_ok());

        let fitted = result.unwrap();
        assert_eq!(fitted.len(), 4);

        // Check monotonicity
        for i in 0..fitted.len() - 1 {
            assert!(fitted[i] <= fitted[i + 1]);
        }
    }

    #[test]
    fn test_disciplined_convex_getters() {
        let model = DisciplinedConvexIsotonicRegression::new()
            .increasing(false)
            .objective(ConvexObjective::Quantile { tau: 0.7 })
            .regularization(0.001)
            .tolerance(1e-6)
            .max_iterations(500);

        assert!(!model.is_increasing());
        assert!(matches!(model.get_objective(), ConvexObjective::Quantile { tau } if *tau == 0.7));
        assert_eq!(model.get_regularization(), 0.001);
        assert_eq!(model.get_tolerance(), 1e-6);
        assert_eq!(model.get_max_iterations(), 500);
        assert_eq!(model.get_constraints().len(), 1);
    }

    #[test]
    fn test_disciplined_convex_empty_input() {
        let x = array![];
        let y = array![];

        let mut model = DisciplinedConvexIsotonicRegression::new();
        let result = model.fit(&x, &y);

        assert!(result.is_err());
    }

    #[test]
    fn test_disciplined_convex_mismatched_lengths() {
        let x = array![1.0, 2.0, 3.0];
        let y = array![1.0, 2.0];

        let mut model = DisciplinedConvexIsotonicRegression::new();
        let result = model.fit(&x, &y);

        assert!(result.is_err());
    }

    #[test]
    fn test_different_objectives() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![1.0, 3.0, 2.0, 4.0, 5.0];

        let objectives = vec![
            ConvexObjective::LeastSquares,
            ConvexObjective::LeastAbsolute,
            ConvexObjective::Huber { delta: 1.0 },
            ConvexObjective::Quantile { tau: 0.5 },
        ];

        for objective in objectives {
            let mut model = DisciplinedConvexIsotonicRegression::new().objective(objective);
            assert!(model.fit(&x, &y).is_ok());
            assert!(model.predict(&x).is_ok());
        }
    }
}
