//! Alternating Direction Method of Multipliers (ADMM) for Isotonic Regression
//!
//! This module implements the ADMM algorithm for isotonic regression,
//! providing a robust and efficient optimization approach that decomposes
//! the problem into simpler subproblems with excellent convergence properties.

use scirs2_core::ndarray::{Array1};
use sklears_core::{prelude::SklearsError, types::Float};

/// Alternating Direction Method of Multipliers (ADMM) for isotonic regression
///
/// ADMM decomposes the isotonic regression problem into alternating optimization
/// steps, providing robust convergence and handling of various constraint types.
/// The algorithm alternates between solving an unconstrained quadratic problem
/// and projecting onto the isotonic constraint set.
///
/// # Mathematical Formulation
///
/// The ADMM algorithm solves:
/// ```text
/// minimize   ||x - y||² + ρ/2 ||x - z + u||²
/// subject to z ∈ isotonic constraint set
/// ```
///
/// where x is the primal variable, z is the auxiliary variable,
/// and u is the scaled dual variable.
///
/// # Examples
///
/// ```rust
/// use sklears_isotonic::convex_optimization::AdmmIsotonicRegression;
/// use scirs2_core::ndarray::array;
///
/// let mut model = AdmmIsotonicRegression::new()
///     .increasing(true)
///     .rho(1.0)
///     .adaptive_rho(true)
///     .max_iterations(1000);
///
/// let x = array![1.0, 2.0, 3.0, 4.0];
/// let y = array![1.5, 1.0, 2.5, 3.0]; // Non-monotonic
///
/// model.fit(&x, &y).unwrap();
/// let predictions = model.predict(&x).unwrap();
/// ```
#[derive(Debug, Clone)]
/// AdmmIsotonicRegression
pub struct AdmmIsotonicRegression {
    /// Whether to enforce increasing or decreasing monotonicity
    increasing: bool,
    /// Penalty parameter for ADMM (ρ)
    rho: Float,
    /// Maximum number of iterations
    max_iterations: usize,
    /// Convergence tolerance for primal residual
    primal_tolerance: Float,
    /// Convergence tolerance for dual residual
    dual_tolerance: Float,
    /// Adaptive penalty parameter adjustment
    adaptive_rho: bool,
    /// Fitted values
    fitted_values: Option<Array1<Float>>,
    /// Fitted x values (for interpolation)
    fitted_x: Option<Array1<Float>>,
}

impl AdmmIsotonicRegression {
    /// Create a new ADMM isotonic regression model
    pub fn new() -> Self {
        Self {
            increasing: true,
            rho: 1.0,
            max_iterations: 1000,
            primal_tolerance: 1e-6,
            dual_tolerance: 1e-6,
            adaptive_rho: true,
            fitted_values: None,
            fitted_x: None,
        }
    }

    /// Set whether the function should be increasing or decreasing
    pub fn increasing(mut self, increasing: bool) -> Self {
        self.increasing = increasing;
        self
    }

    /// Set penalty parameter (ρ)
    ///
    /// The penalty parameter controls the augmented Lagrangian term.
    /// Larger values enforce constraint satisfaction more strongly
    /// but may lead to ill-conditioning.
    pub fn rho(mut self, rho: Float) -> Self {
        self.rho = rho;
        self
    }

    /// Set maximum number of iterations
    pub fn max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Set primal tolerance
    ///
    /// Convergence tolerance for the primal residual ||x - z||
    pub fn primal_tolerance(mut self, tolerance: Float) -> Self {
        self.primal_tolerance = tolerance;
        self
    }

    /// Set dual tolerance
    ///
    /// Convergence tolerance for the dual residual ||ρ(z^{k+1} - z^k)||
    pub fn dual_tolerance(mut self, tolerance: Float) -> Self {
        self.dual_tolerance = tolerance;
        self
    }

    /// Enable or disable adaptive penalty parameter adjustment
    ///
    /// Adaptive ρ adjustment helps balance primal and dual residuals
    /// for better convergence properties.
    pub fn adaptive_rho(mut self, adaptive: bool) -> Self {
        self.adaptive_rho = adaptive;
        self
    }

    /// Fit the ADMM isotonic regression model
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

        let n = x.len();

        // Sort data by x values
        let mut data: Vec<(Float, Float)> =
            x.iter().zip(y.iter()).map(|(&xi, &yi)| (xi, yi)).collect();
        data.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        let sorted_x: Array1<Float> = Array1::from_vec(data.iter().map(|(xi, _)| *xi).collect());
        let sorted_y: Array1<Float> = Array1::from_vec(data.iter().map(|(_, yi)| *yi).collect());

        // ADMM variables
        let mut z = sorted_y.clone(); // Primal variable (isotonic)
        let mut u = Array1::zeros(n); // Dual variable (scaled)
        let mut rho = self.rho;

        for iter in 0..self.max_iterations {
            // x-update: solve for unconstrained problem
            let x_var = self.x_update(&sorted_y, &z, &u, rho);

            // z-update: projection onto isotonic constraint set
            let z_new = self.z_update(&x_var, &u, rho);

            // u-update: dual variable update
            let u_new = &u + (&x_var - &z_new);

            // Check convergence
            let primal_residual = (&x_var - &z_new).mapv(|x| x.abs()).sum();
            let dual_residual = rho * (&z_new - &z).mapv(|x| x.abs()).sum();

            if primal_residual < self.primal_tolerance && dual_residual < self.dual_tolerance {
                self.fitted_values = Some(z_new);
                self.fitted_x = Some(sorted_x);
                return Ok(());
            }

            // Adaptive penalty parameter adjustment
            if self.adaptive_rho && iter % 10 == 0 {
                if primal_residual > 10.0 * dual_residual {
                    rho *= 2.0;
                } else if dual_residual > 10.0 * primal_residual {
                    rho /= 2.0;
                }
            }

            z = z_new;
            u = u_new;
        }

        // Use final values even if not converged
        self.fitted_values = Some(z);
        self.fitted_x = Some(sorted_x);
        Ok(())
    }

    /// X-update step: solve unconstrained quadratic problem
    ///
    /// Solves: argmin_x ||x - y||² + ρ/2 ||x - z + u||²
    ///
    /// The analytical solution is: x = (y + ρ(z - u)) / (1 + ρ)
    fn x_update(
        &self,
        y: &Array1<Float>,
        z: &Array1<Float>,
        u: &Array1<Float>,
        rho: Float,
    ) -> Array1<Float> {
        // Solve: argmin_x ||x - y||^2 + rho/2 ||x - z + u||^2
        // Solution: x = (y + rho*(z - u)) / (1 + rho)
        (y + &(rho * (z - u))) / (1.0 + rho)
    }

    /// Z-update step: projection onto isotonic constraint set
    ///
    /// Projects x + u onto the isotonic constraint set using the
    /// Pool Adjacent Violators Algorithm (PAVA).
    fn z_update(&self, x: &Array1<Float>, u: &Array1<Float>, rho: Float) -> Array1<Float> {
        // Project x + u onto the isotonic constraint set
        let target = x + u;

        // For ADMM, we need to maintain the same array length, so we'll use a simpler projection
        // that doesn't merge points. We'll use Pool Adjacent Violators directly on the target.
        self.pava_projection(&target)
    }

    /// Pool Adjacent Violators Algorithm for projection
    ///
    /// Projects the input array onto the isotonic constraint set
    /// while maintaining the original array length.
    fn pava_projection(&self, y: &Array1<Float>) -> Array1<Float> {
        let n = y.len();
        let mut result = y.clone();

        if self.increasing {
            // Increasing PAVA
            for i in 1..n {
                if result[i] < result[i - 1] {
                    // Pool adjacent violators
                    let mut j = i;
                    let mut sum = result[i] + result[i - 1];
                    let mut count = 2;

                    // Find the range to pool
                    while j > 1 && sum / (count as Float) < result[j - 2] {
                        j -= 1;
                        sum += result[j - 1];
                        count += 1;
                    }

                    // Update the pooled values
                    let avg = sum / (count as Float);
                    for k in (j - 1)..=i {
                        result[k] = avg;
                    }
                }
            }
        } else {
            // Decreasing PAVA
            for i in 1..n {
                if result[i] > result[i - 1] {
                    // Pool adjacent violators
                    let mut j = i;
                    let mut sum = result[i] + result[i - 1];
                    let mut count = 2;

                    // Find the range to pool
                    while j > 1 && sum / (count as Float) > result[j - 2] {
                        j -= 1;
                        sum += result[j - 1];
                        count += 1;
                    }

                    // Update the pooled values
                    let avg = sum / (count as Float);
                    for k in (j - 1)..=i {
                        result[k] = avg;
                    }
                }
            }
        }

        result
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
            predictions[i] = self.interpolate(xi, fitted_x, fitted_values);
        }

        Ok(predictions)
    }

    /// Linear interpolation for prediction
    fn interpolate(
        &self,
        x: Float,
        fitted_x: &Array1<Float>,
        fitted_values: &Array1<Float>,
    ) -> Float {
        if fitted_x.is_empty() {
            return Float::NAN;
        }

        let n = fitted_x.len().min(fitted_values.len());

        if n == 1 {
            return fitted_values[0];
        }

        // Handle boundary cases
        if x <= fitted_x[0] {
            return fitted_values[0];
        }
        if x >= fitted_x[n - 1] {
            return fitted_values[n - 1];
        }

        // Find interpolation interval
        for i in 0..n - 1 {
            if x >= fitted_x[i] && x <= fitted_x[i + 1] {
                if fitted_x[i + 1] == fitted_x[i] {
                    return fitted_values[i];
                }
                let t = (x - fitted_x[i]) / (fitted_x[i + 1] - fitted_x[i]);
                return fitted_values[i] + t * (fitted_values[i + 1] - fitted_values[i]);
            }
        }

        fitted_values[n - 1]
    }

    /// Get the fitted values (for analysis)
    pub fn fitted_values(&self) -> Option<&Array1<Float>> {
        self.fitted_values.as_ref()
    }

    /// Get the fitted x values (for analysis)
    pub fn fitted_x(&self) -> Option<&Array1<Float>> {
        self.fitted_x.as_ref()
    }

    /// Get current penalty parameter (ρ)
    pub fn get_rho(&self) -> Float {
        self.rho
    }

    /// Get current primal tolerance
    pub fn get_primal_tolerance(&self) -> Float {
        self.primal_tolerance
    }

    /// Get current dual tolerance
    pub fn get_dual_tolerance(&self) -> Float {
        self.dual_tolerance
    }

    /// Get maximum iterations setting
    pub fn get_max_iterations(&self) -> usize {
        self.max_iterations
    }

    /// Check if adaptive ρ adjustment is enabled
    pub fn is_adaptive_rho(&self) -> bool {
        self.adaptive_rho
    }

    /// Check if model enforces increasing monotonicity
    pub fn is_increasing(&self) -> bool {
        self.increasing
    }
}

impl Default for AdmmIsotonicRegression {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience function for ADMM isotonic regression
///
/// This function provides a simple interface for one-shot ADMM isotonic regression.
///
/// # Arguments
///
/// * `x` - Input features (must be sorted)
/// * `y` - Target values
/// * `increasing` - Whether to enforce increasing monotonicity
/// * `rho` - Penalty parameter for ADMM algorithm
///
/// # Returns
///
/// Fitted isotonic values or error
pub fn admm_isotonic_regression(
    x: &Array1<Float>,
    y: &Array1<Float>,
    increasing: bool,
    rho: Option<Float>,
) -> Result<Array1<Float>, SklearsError> {
    let mut model = AdmmIsotonicRegression::new().increasing(increasing);

    if let Some(rho_val) = rho {
        model = model.rho(rho_val);
    }

    model.fit(x, y)?;
    Ok(model.fitted_values().unwrap().clone())
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_admm_increasing() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![1.0, 3.0, 2.0, 4.0, 5.0]; // Non-monotonic

        let mut model = AdmmIsotonicRegression::new()
            .increasing(true)
            .rho(1.0)
            .max_iterations(100);

        assert!(model.fit(&x, &y).is_ok());

        let predictions = model.predict(&x).unwrap();

        // Check that predictions are increasing
        for i in 0..predictions.len() - 1 {
            assert!(predictions[i] <= predictions[i + 1]);
        }
    }

    #[test]
    fn test_admm_decreasing() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![5.0, 3.0, 4.0, 2.0, 1.0]; // Non-monotonic

        let mut model = AdmmIsotonicRegression::new()
            .increasing(false)
            .adaptive_rho(true);

        assert!(model.fit(&x, &y).is_ok());

        let predictions = model.predict(&x).unwrap();

        // Check that predictions are decreasing
        for i in 0..predictions.len() - 1 {
            assert!(predictions[i] >= predictions[i + 1]);
        }
    }

    #[test]
    fn test_admm_convenience_function() {
        let x = array![1.0, 2.0, 3.0, 4.0];
        let y = array![1.5, 1.0, 2.5, 3.0];

        let result = admm_isotonic_regression(&x, &y, true, Some(2.0));
        assert!(result.is_ok());

        let fitted = result.unwrap();
        assert_eq!(fitted.len(), 4);

        // Check monotonicity
        for i in 0..fitted.len() - 1 {
            assert!(fitted[i] <= fitted[i + 1]);
        }
    }

    #[test]
    fn test_admm_getters() {
        let model = AdmmIsotonicRegression::new()
            .increasing(false)
            .rho(2.5)
            .primal_tolerance(1e-7)
            .dual_tolerance(1e-5)
            .max_iterations(500)
            .adaptive_rho(false);

        assert!(!model.is_increasing());
        assert_eq!(model.get_rho(), 2.5);
        assert_eq!(model.get_primal_tolerance(), 1e-7);
        assert_eq!(model.get_dual_tolerance(), 1e-5);
        assert_eq!(model.get_max_iterations(), 500);
        assert!(!model.is_adaptive_rho());
    }

    #[test]
    fn test_admm_empty_input() {
        let x = array![];
        let y = array![];

        let mut model = AdmmIsotonicRegression::new();
        let result = model.fit(&x, &y);

        assert!(result.is_err());
    }

    #[test]
    fn test_admm_mismatched_lengths() {
        let x = array![1.0, 2.0, 3.0];
        let y = array![1.0, 2.0];

        let mut model = AdmmIsotonicRegression::new();
        let result = model.fit(&x, &y);

        assert!(result.is_err());
    }

    #[test]
    fn test_admm_convergence_tolerance() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![1.0, 3.0, 2.0, 4.0, 5.0]; // Non-monotonic

        let mut model = AdmmIsotonicRegression::new()
            .increasing(true)
            .primal_tolerance(1e-8)
            .dual_tolerance(1e-8)
            .max_iterations(1000);

        assert!(model.fit(&x, &y).is_ok());

        let predictions = model.predict(&x).unwrap();

        // Check that predictions are increasing
        for i in 0..predictions.len() - 1 {
            assert!(predictions[i] <= predictions[i + 1]);
        }
    }

    #[test]
    fn test_admm_adaptive_rho() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![1.0, 3.0, 2.0, 4.0, 5.0]; // Non-monotonic

        let mut model = AdmmIsotonicRegression::new()
            .increasing(true)
            .rho(0.1) // Start with small rho
            .adaptive_rho(true)
            .max_iterations(100);

        assert!(model.fit(&x, &y).is_ok());

        let predictions = model.predict(&x).unwrap();

        // Check that predictions are increasing
        for i in 0..predictions.len() - 1 {
            assert!(predictions[i] <= predictions[i + 1]);
        }
    }
}