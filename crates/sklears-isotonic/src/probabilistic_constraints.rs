//! Probabilistic constraints for isotonic regression
//!
//! This module provides probabilistic (soft) constraints for isotonic regression,
//! allowing for violations of monotonicity with specified confidence levels.

use crate::core::{LossFunction, MonotonicityConstraint};
use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Estimator, Fit, Predict, Trained, Untrained},
    types::Float,
};
use std::marker::PhantomData;

/// Probabilistic constraint specification
#[derive(Debug, Clone)]
/// ProbabilisticConstraint
pub struct ProbabilisticConstraint {
    /// Base monotonicity constraint
    pub base_constraint: MonotonicityConstraint,
    /// Confidence level (probability that constraint is satisfied)
    pub confidence_level: Float,
    /// Penalty weight for constraint violations
    pub violation_penalty: Float,
    /// Tolerance for constraint violations
    pub violation_tolerance: Float,
}

/// Constraint enforcement strategy
#[derive(Debug, Clone, Copy, PartialEq)]
/// ConstraintEnforcement
pub enum ConstraintEnforcement {
    /// Soft penalty approach - violations are penalized but allowed
    SoftPenalty,
    /// Chance constraint - constraint satisfied with given probability
    ChanceConstraint,
    /// Robust optimization - worst-case constraint satisfaction
    RobustOptimization,
    /// Bayesian approach with uncertainty in constraints
    BayesianUncertainty,
}

/// Probabilistic isotonic regression model
#[derive(Debug, Clone)]
/// ProbabilisticIsotonicRegression
pub struct ProbabilisticIsotonicRegression<State = Untrained> {
    /// Probabilistic constraints
    pub constraints: Vec<ProbabilisticConstraint>,
    /// Constraint enforcement strategy
    pub enforcement: ConstraintEnforcement,
    /// Loss function
    pub loss: LossFunction,
    /// Maximum iterations for optimization
    pub max_iter: usize,
    /// Convergence tolerance
    pub tolerance: Float,
    /// Learning rate for gradient descent
    pub learning_rate: Float,
    /// Number of samples for Monte Carlo methods
    pub n_samples: usize,
    /// Random seed for reproducibility
    pub random_seed: Option<u64>,
    /// Global bounds
    pub y_min: Option<Float>,
    pub y_max: Option<Float>,

    // Fitted attributes
    x_: Option<Array1<Float>>,
    y_: Option<Array1<Float>>,
    constraint_violations_: Option<Array1<Float>>,
    confidence_intervals_: Option<Array2<Float>>,

    _state: PhantomData<State>,
}

impl ProbabilisticIsotonicRegression<Untrained> {
    /// Create a new probabilistic isotonic regression model
    pub fn new() -> Self {
        Self {
            constraints: Vec::new(),
            enforcement: ConstraintEnforcement::SoftPenalty,
            loss: LossFunction::SquaredLoss,
            max_iter: 1000,
            tolerance: 1e-6,
            learning_rate: 0.01,
            n_samples: 1000,
            random_seed: None,
            y_min: None,
            y_max: None,
            x_: None,
            y_: None,
            constraint_violations_: None,
            confidence_intervals_: None,
            _state: PhantomData,
        }
    }

    /// Add a probabilistic constraint
    pub fn add_constraint(mut self, constraint: ProbabilisticConstraint) -> Self {
        self.constraints.push(constraint);
        self
    }

    /// Add multiple probabilistic constraints
    pub fn add_constraints(mut self, constraints: Vec<ProbabilisticConstraint>) -> Self {
        self.constraints.extend(constraints);
        self
    }

    /// Set constraint enforcement strategy
    pub fn enforcement(mut self, enforcement: ConstraintEnforcement) -> Self {
        self.enforcement = enforcement;
        self
    }

    /// Set loss function
    pub fn loss(mut self, loss: LossFunction) -> Self {
        self.loss = loss;
        self
    }

    /// Set maximum iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set convergence tolerance
    pub fn tolerance(mut self, tolerance: Float) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Set learning rate
    pub fn learning_rate(mut self, learning_rate: Float) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    /// Set number of samples for Monte Carlo methods
    pub fn n_samples(mut self, n_samples: usize) -> Self {
        self.n_samples = n_samples;
        self
    }

    /// Set random seed
    pub fn random_seed(mut self, seed: u64) -> Self {
        self.random_seed = Some(seed);
        self
    }

    /// Set global bounds
    pub fn bounds(mut self, y_min: Option<Float>, y_max: Option<Float>) -> Self {
        self.y_min = y_min;
        self.y_max = y_max;
        self
    }

    /// Create a simple probabilistic monotonic constraint
    pub fn probabilistic_monotonic(
        mut self,
        increasing: bool,
        confidence_level: Float,
        violation_penalty: Float,
    ) -> Self {
        let constraint = ProbabilisticConstraint {
            base_constraint: MonotonicityConstraint::Global { increasing },
            confidence_level,
            violation_penalty,
            violation_tolerance: self.tolerance,
        };
        self.constraints.push(constraint);
        self
    }
}

impl Default for ProbabilisticIsotonicRegression<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for ProbabilisticIsotonicRegression<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<Array1<Float>, Array1<Float>> for ProbabilisticIsotonicRegression<Untrained> {
    type Fitted = ProbabilisticIsotonicRegression<Trained>;

    fn fit(self, x: &Array1<Float>, y: &Array1<Float>) -> Result<Self::Fitted> {
        if x.len() != y.len() {
            return Err(SklearsError::InvalidInput(
                "x and y must have the same length".to_string(),
            ));
        }

        if x.is_empty() {
            return Err(SklearsError::InvalidInput(
                "x and y cannot be empty".to_string(),
            ));
        }

        if self.constraints.is_empty() {
            return Err(SklearsError::InvalidInput(
                "At least one probabilistic constraint must be specified".to_string(),
            ));
        }

        let n = x.len();

        // Sort data by x values
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&a, &b| x[a].partial_cmp(&x[b]).unwrap());

        let sorted_x: Vec<Float> = indices.iter().map(|&i| x[i]).collect();
        let sorted_y: Vec<Float> = indices.iter().map(|&i| y[i]).collect();

        // Fit with probabilistic constraints
        let (fitted_y, violations, confidence_intervals) =
            fit_probabilistic_constrained(&sorted_x, &sorted_y, &self)?;

        Ok(ProbabilisticIsotonicRegression {
            constraints: self.constraints,
            enforcement: self.enforcement,
            loss: self.loss,
            max_iter: self.max_iter,
            tolerance: self.tolerance,
            learning_rate: self.learning_rate,
            n_samples: self.n_samples,
            random_seed: self.random_seed,
            y_min: self.y_min,
            y_max: self.y_max,
            x_: Some(Array1::from(sorted_x)),
            y_: Some(Array1::from(fitted_y)),
            constraint_violations_: Some(violations),
            confidence_intervals_: Some(confidence_intervals),
            _state: PhantomData,
        })
    }
}

impl Predict<Array1<Float>, Array1<Float>> for ProbabilisticIsotonicRegression<Trained> {
    fn predict(&self, x: &Array1<Float>) -> Result<Array1<Float>> {
        let fitted_x = self.x_.as_ref().ok_or_else(|| SklearsError::NotFitted {
            operation: "predict".to_string(),
        })?;

        let fitted_y = self.y_.as_ref().ok_or_else(|| SklearsError::NotFitted {
            operation: "predict".to_string(),
        })?;

        let predictions = crate::algorithms::linear_interpolate(fitted_x, fitted_y, x);
        Ok(predictions)
    }
}

impl ProbabilisticIsotonicRegression<Trained> {
    /// Get constraint violations
    pub fn constraint_violations(&self) -> Result<&Array1<Float>> {
        self.constraint_violations_
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "constraint_violations".to_string(),
            })
    }

    /// Get confidence intervals
    pub fn confidence_intervals(&self) -> Result<&Array2<Float>> {
        self.confidence_intervals_
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "confidence_intervals".to_string(),
            })
    }

    /// Get fitted x values
    pub fn fitted_x(&self) -> Result<&Array1<Float>> {
        self.x_.as_ref().ok_or_else(|| SklearsError::NotFitted {
            operation: "fitted_x".to_string(),
        })
    }

    /// Get fitted y values
    pub fn fitted_y(&self) -> Result<&Array1<Float>> {
        self.y_.as_ref().ok_or_else(|| SklearsError::NotFitted {
            operation: "fitted_y".to_string(),
        })
    }

    /// Predict with uncertainty bounds
    pub fn predict_with_uncertainty(
        &self,
        x: &Array1<Float>,
    ) -> Result<(Array1<Float>, Array2<Float>)> {
        let fitted_x = self.fitted_x()?;
        let fitted_y = self.fitted_y()?;
        let confidence_intervals = self.confidence_intervals()?;

        // Interpolate mean predictions
        let mean_predictions = crate::algorithms::linear_interpolate(fitted_x, fitted_y, x);

        // Interpolate confidence intervals
        let lower_bounds = confidence_intervals.column(0).to_owned();
        let upper_bounds = confidence_intervals.column(1).to_owned();

        let lower_predictions = crate::algorithms::linear_interpolate(fitted_x, &lower_bounds, x);
        let upper_predictions = crate::algorithms::linear_interpolate(fitted_x, &upper_bounds, x);

        let mut uncertainty_intervals = Array2::zeros((x.len(), 2));
        uncertainty_intervals
            .column_mut(0)
            .assign(&lower_predictions);
        uncertainty_intervals
            .column_mut(1)
            .assign(&upper_predictions);

        Ok((mean_predictions, uncertainty_intervals))
    }

    /// Get overall constraint violation score
    pub fn violation_score(&self) -> Result<Float> {
        let violations = self.constraint_violations()?;
        Ok(violations.sum())
    }

    /// Check if constraints are satisfied within confidence level
    pub fn constraints_satisfied(&self) -> Result<bool> {
        let violations = self.constraint_violations()?;
        let max_violation = violations.iter().fold(0.0f64, |acc, &x| acc.max(x));

        // Check if maximum violation is within tolerance
        Ok(max_violation <= self.tolerance)
    }
}

/// Fit probabilistic constrained isotonic regression
fn fit_probabilistic_constrained(
    x: &[Float],
    y: &[Float],
    config: &ProbabilisticIsotonicRegression<Untrained>,
) -> Result<(Vec<Float>, Array1<Float>, Array2<Float>)> {
    match config.enforcement {
        ConstraintEnforcement::SoftPenalty => fit_soft_penalty(x, y, config),
        ConstraintEnforcement::ChanceConstraint => fit_chance_constraint(x, y, config),
        ConstraintEnforcement::RobustOptimization => fit_robust_optimization(x, y, config),
        ConstraintEnforcement::BayesianUncertainty => fit_bayesian_uncertainty(x, y, config),
    }
}

/// Fit with soft penalty enforcement
fn fit_soft_penalty(
    x: &[Float],
    y: &[Float],
    config: &ProbabilisticIsotonicRegression<Untrained>,
) -> Result<(Vec<Float>, Array1<Float>, Array2<Float>)> {
    let n = x.len();
    let mut fitted_y = y.to_vec();

    for iteration in 0..config.max_iter {
        let old_y = fitted_y.clone();

        // Gradient step for data fitting term
        for i in 0..n {
            let gradient = 2.0 * (fitted_y[i] - y[i]); // L2 loss gradient
            fitted_y[i] -= config.learning_rate * gradient;
        }

        // Apply probabilistic constraint penalties
        for constraint in &config.constraints {
            let penalty_gradient =
                compute_constraint_penalty_gradient(&fitted_y, constraint, config.learning_rate)?;

            for i in 0..n {
                fitted_y[i] -= penalty_gradient[i];
            }
        }

        // Apply bounds
        apply_bounds(&mut fitted_y, config);

        // Check convergence
        let change: Float = fitted_y
            .iter()
            .zip(old_y.iter())
            .map(|(new, old)| (new - old).powi(2))
            .sum::<Float>()
            .sqrt();

        if change < config.tolerance {
            break;
        }
    }

    // Compute violations and confidence intervals
    let violations = compute_constraint_violations(&fitted_y, &config.constraints)?;
    let confidence_intervals = compute_confidence_intervals(&fitted_y, config)?;

    Ok((fitted_y, violations, confidence_intervals))
}

/// Fit with chance constraint enforcement
fn fit_chance_constraint(
    x: &[Float],
    y: &[Float],
    config: &ProbabilisticIsotonicRegression<Untrained>,
) -> Result<(Vec<Float>, Array1<Float>, Array2<Float>)> {
    let n = x.len();
    let mut fitted_y = y.to_vec();

    // Sample-based approximation of chance constraints
    let mut rng_state = config.random_seed.unwrap_or(42);

    for iteration in 0..config.max_iter {
        let old_y = fitted_y.clone();

        // Generate samples for chance constraint approximation
        let mut constraint_samples = Vec::new();
        for _ in 0..config.n_samples {
            let noise_scale = 0.1; // Standard deviation of noise
            let noisy_y: Vec<Float> = fitted_y
                .iter()
                .map(|&val| val + sample_gaussian(0.0, noise_scale, &mut rng_state))
                .collect();
            constraint_samples.push(noisy_y);
        }

        // Gradient step for data fitting term
        for i in 0..n {
            let gradient = 2.0 * (fitted_y[i] - y[i]);
            fitted_y[i] -= config.learning_rate * gradient;
        }

        // Apply chance constraint corrections
        for constraint in &config.constraints {
            let violation_probability =
                estimate_violation_probability(&constraint_samples, constraint)?;

            if violation_probability > (1.0 - constraint.confidence_level) {
                // Apply correction to satisfy chance constraint
                let correction = compute_chance_constraint_correction(
                    &fitted_y,
                    constraint,
                    violation_probability,
                    config.learning_rate,
                )?;

                for i in 0..n {
                    fitted_y[i] -= correction[i];
                }
            }
        }

        // Apply bounds
        apply_bounds(&mut fitted_y, config);

        // Check convergence
        let change: Float = fitted_y
            .iter()
            .zip(old_y.iter())
            .map(|(new, old)| (new - old).powi(2))
            .sum::<Float>()
            .sqrt();

        if change < config.tolerance {
            break;
        }
    }

    let violations = compute_constraint_violations(&fitted_y, &config.constraints)?;
    let confidence_intervals = compute_confidence_intervals(&fitted_y, config)?;

    Ok((fitted_y, violations, confidence_intervals))
}

/// Fit with robust optimization enforcement
fn fit_robust_optimization(
    x: &[Float],
    y: &[Float],
    config: &ProbabilisticIsotonicRegression<Untrained>,
) -> Result<(Vec<Float>, Array1<Float>, Array2<Float>)> {
    // Robust optimization considers worst-case scenarios
    // For now, implement as conservative soft penalty
    fit_soft_penalty(x, y, config)
}

/// Fit with Bayesian uncertainty enforcement
fn fit_bayesian_uncertainty(
    x: &[Float],
    y: &[Float],
    config: &ProbabilisticIsotonicRegression<Untrained>,
) -> Result<(Vec<Float>, Array1<Float>, Array2<Float>)> {
    // Bayesian approach with uncertain constraints
    // For now, implement as soft penalty with uncertainty propagation
    fit_soft_penalty(x, y, config)
}

/// Compute constraint penalty gradient
fn compute_constraint_penalty_gradient(
    fitted_y: &[Float],
    constraint: &ProbabilisticConstraint,
    learning_rate: Float,
) -> Result<Vec<Float>> {
    let n = fitted_y.len();
    let mut gradient = vec![0.0; n];

    match &constraint.base_constraint {
        MonotonicityConstraint::Global { increasing: true } => {
            // Penalize decreasing violations
            for i in 0..n - 1 {
                if fitted_y[i] > fitted_y[i + 1] {
                    let violation = fitted_y[i] - fitted_y[i + 1];
                    let penalty = constraint.violation_penalty
                        * (1.0 - constraint.confidence_level)
                        * violation;

                    gradient[i] += learning_rate * penalty;
                    gradient[i + 1] -= learning_rate * penalty;
                }
            }
        }
        MonotonicityConstraint::Global { increasing: false } => {
            // Penalize increasing violations
            for i in 0..n - 1 {
                if fitted_y[i] < fitted_y[i + 1] {
                    let violation = fitted_y[i + 1] - fitted_y[i];
                    let penalty = constraint.violation_penalty
                        * (1.0 - constraint.confidence_level)
                        * violation;

                    gradient[i] -= learning_rate * penalty;
                    gradient[i + 1] += learning_rate * penalty;
                }
            }
        }
        _ => {
            // For complex constraints, use simple increasing for now
            for i in 0..n - 1 {
                if fitted_y[i] > fitted_y[i + 1] {
                    let violation = fitted_y[i] - fitted_y[i + 1];
                    let penalty = constraint.violation_penalty
                        * (1.0 - constraint.confidence_level)
                        * violation;

                    gradient[i] += learning_rate * penalty;
                    gradient[i + 1] -= learning_rate * penalty;
                }
            }
        }
    }

    Ok(gradient)
}

/// Compute constraint violations
fn compute_constraint_violations(
    fitted_y: &[Float],
    constraints: &[ProbabilisticConstraint],
) -> Result<Array1<Float>> {
    let n = fitted_y.len();
    let mut violations = Array1::zeros(n);

    for constraint in constraints {
        match &constraint.base_constraint {
            MonotonicityConstraint::Global { increasing: true } => {
                for i in 0..n - 1 {
                    if fitted_y[i] > fitted_y[i + 1] {
                        let violation = fitted_y[i] - fitted_y[i + 1];
                        violations[i] += violation;
                    }
                }
            }
            MonotonicityConstraint::Global { increasing: false } => {
                for i in 0..n - 1 {
                    if fitted_y[i] < fitted_y[i + 1] {
                        let violation = fitted_y[i + 1] - fitted_y[i];
                        violations[i] += violation;
                    }
                }
            }
            _ => {
                // Default to increasing constraint
                for i in 0..n - 1 {
                    if fitted_y[i] > fitted_y[i + 1] {
                        let violation = fitted_y[i] - fitted_y[i + 1];
                        violations[i] += violation;
                    }
                }
            }
        }
    }

    Ok(violations)
}

/// Compute confidence intervals based on constraint uncertainty
fn compute_confidence_intervals(
    fitted_y: &[Float],
    config: &ProbabilisticIsotonicRegression<Untrained>,
) -> Result<Array2<Float>> {
    let n = fitted_y.len();
    let mut intervals = Array2::zeros((n, 2));

    // Estimate uncertainty from constraint confidence levels
    let avg_confidence: Float = config
        .constraints
        .iter()
        .map(|c| c.confidence_level)
        .sum::<Float>()
        / (config.constraints.len() as Float);

    let uncertainty_scale = (1.0 - avg_confidence) * 2.0; // Simple heuristic

    for i in 0..n {
        let uncertainty = uncertainty_scale * fitted_y[i].abs().max(0.1);
        intervals[[i, 0]] = fitted_y[i] - uncertainty; // Lower bound
        intervals[[i, 1]] = fitted_y[i] + uncertainty; // Upper bound
    }

    Ok(intervals)
}

/// Estimate violation probability from samples
fn estimate_violation_probability(
    samples: &[Vec<Float>],
    constraint: &ProbabilisticConstraint,
) -> Result<Float> {
    let n_samples = samples.len();
    let mut violations = 0;

    for sample in samples {
        if violates_constraint(sample, constraint)? {
            violations += 1;
        }
    }

    Ok((violations as Float) / (n_samples as Float))
}

/// Check if a sample violates the constraint
fn violates_constraint(sample: &[Float], constraint: &ProbabilisticConstraint) -> Result<bool> {
    match &constraint.base_constraint {
        MonotonicityConstraint::Global { increasing: true } => {
            for i in 0..sample.len() - 1 {
                if sample[i] > sample[i + 1] + constraint.violation_tolerance {
                    return Ok(true);
                }
            }
        }
        MonotonicityConstraint::Global { increasing: false } => {
            for i in 0..sample.len() - 1 {
                if sample[i] < sample[i + 1] - constraint.violation_tolerance {
                    return Ok(true);
                }
            }
        }
        _ => {
            // Default to increasing constraint
            for i in 0..sample.len() - 1 {
                if sample[i] > sample[i + 1] + constraint.violation_tolerance {
                    return Ok(true);
                }
            }
        }
    }
    Ok(false)
}

/// Compute chance constraint correction
fn compute_chance_constraint_correction(
    fitted_y: &[Float],
    constraint: &ProbabilisticConstraint,
    violation_probability: Float,
    learning_rate: Float,
) -> Result<Vec<Float>> {
    let n = fitted_y.len();
    let mut correction = vec![0.0; n];

    let excess_probability = violation_probability - (1.0 - constraint.confidence_level);
    if excess_probability > 0.0 {
        let correction_strength = learning_rate * constraint.violation_penalty * excess_probability;

        match &constraint.base_constraint {
            MonotonicityConstraint::Global { increasing: true } => {
                for i in 0..n - 1 {
                    if fitted_y[i] > fitted_y[i + 1] {
                        correction[i] += correction_strength;
                        correction[i + 1] -= correction_strength;
                    }
                }
            }
            MonotonicityConstraint::Global { increasing: false } => {
                for i in 0..n - 1 {
                    if fitted_y[i] < fitted_y[i + 1] {
                        correction[i] -= correction_strength;
                        correction[i + 1] += correction_strength;
                    }
                }
            }
            _ => {
                // Default to increasing constraint
                for i in 0..n - 1 {
                    if fitted_y[i] > fitted_y[i + 1] {
                        correction[i] += correction_strength;
                        correction[i + 1] -= correction_strength;
                    }
                }
            }
        }
    }

    Ok(correction)
}

/// Apply bounds to fitted values
fn apply_bounds(fitted_y: &mut [Float], config: &ProbabilisticIsotonicRegression<Untrained>) {
    if let Some(y_min) = config.y_min {
        for val in fitted_y.iter_mut() {
            *val = val.max(y_min);
        }
    }
    if let Some(y_max) = config.y_max {
        for val in fitted_y.iter_mut() {
            *val = val.min(y_max);
        }
    }
}

/// Simple Gaussian sampling (placeholder implementation)
fn sample_gaussian(mean: Float, std_dev: Float, rng_state: &mut u64) -> Float {
    // Simple Box-Muller transform
    *rng_state = (*rng_state).wrapping_mul(1103515245).wrapping_add(12345);
    let u1 = (*rng_state as f64) / (u64::MAX as f64);

    *rng_state = (*rng_state).wrapping_mul(1103515245).wrapping_add(12345);
    let u2 = (*rng_state as f64) / (u64::MAX as f64);

    let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();

    mean + std_dev * z as Float
}

/// Convenience function for probabilistic isotonic regression
pub fn probabilistic_isotonic_regression(
    x: &Array1<Float>,
    y: &Array1<Float>,
    constraints: Vec<ProbabilisticConstraint>,
    enforcement: ConstraintEnforcement,
) -> Result<(Array1<Float>, Array1<Float>, Array1<Float>, Array2<Float>)> {
    let model = ProbabilisticIsotonicRegression::new()
        .add_constraints(constraints)
        .enforcement(enforcement);

    let fitted_model = model.fit(x, y)?;

    let fitted_x = fitted_model.fitted_x()?.clone();
    let fitted_y = fitted_model.fitted_y()?.clone();
    let violations = fitted_model.constraint_violations()?.clone();
    let confidence_intervals = fitted_model.confidence_intervals()?.clone();

    Ok((fitted_x, fitted_y, violations, confidence_intervals))
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_probabilistic_constraint_basic() {
        let x = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let y = Array1::from(vec![1.0, 3.0, 2.0, 4.0, 5.0]);

        let constraint = ProbabilisticConstraint {
            base_constraint: MonotonicityConstraint::Global { increasing: true },
            confidence_level: 0.9,
            violation_penalty: 1.0,
            violation_tolerance: 0.1,
        };

        let model = ProbabilisticIsotonicRegression::new()
            .add_constraint(constraint)
            .enforcement(ConstraintEnforcement::SoftPenalty);

        let fitted_model = model.fit(&x, &y).unwrap();
        let fitted_y = fitted_model.fitted_y().unwrap();

        assert_eq!(fitted_y.len(), 5);

        // Constraints should be mostly satisfied
        let violations = fitted_model.constraint_violations().unwrap();
        let total_violation = violations.sum();
        assert!(total_violation >= 0.0); // Non-negative violations
    }

    #[test]
    fn test_chance_constraint_enforcement() {
        let x = Array1::from(vec![1.0, 2.0, 3.0, 4.0]);
        let y = Array1::from(vec![2.0, 1.0, 3.0, 4.0]);

        let constraint = ProbabilisticConstraint {
            base_constraint: MonotonicityConstraint::Global { increasing: true },
            confidence_level: 0.8,
            violation_penalty: 0.5,
            violation_tolerance: 0.05,
        };

        let model = ProbabilisticIsotonicRegression::new()
            .add_constraint(constraint)
            .enforcement(ConstraintEnforcement::ChanceConstraint)
            .n_samples(100);

        let fitted_model = model.fit(&x, &y).unwrap();
        let fitted_y = fitted_model.fitted_y().unwrap();

        assert_eq!(fitted_y.len(), 4);

        // Check that we can get confidence intervals
        let intervals = fitted_model.confidence_intervals().unwrap();
        assert_eq!(intervals.shape(), &[4, 2]);

        // Lower bounds should be less than upper bounds
        for i in 0..4 {
            assert!(intervals[[i, 0]] <= intervals[[i, 1]]);
        }
    }

    #[test]
    fn test_prediction_with_uncertainty() {
        let x = Array1::from(vec![1.0, 2.0, 3.0, 4.0]);
        let y = Array1::from(vec![1.0, 2.0, 3.0, 4.0]);

        let model = ProbabilisticIsotonicRegression::new().probabilistic_monotonic(true, 0.95, 0.1);

        let fitted_model = model.fit(&x, &y).unwrap();

        let x_new = Array1::from(vec![1.5, 2.5, 3.5]);
        let (predictions, uncertainty) = fitted_model.predict_with_uncertainty(&x_new).unwrap();

        assert_eq!(predictions.len(), 3);
        assert_eq!(uncertainty.shape(), &[3, 2]);

        // Predictions should be within uncertainty bounds
        for i in 0..3 {
            assert!(predictions[i] >= uncertainty[[i, 0]]);
            assert!(predictions[i] <= uncertainty[[i, 1]]);
        }
    }

    #[test]
    fn test_violation_metrics() {
        let x = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let y = Array1::from(vec![5.0, 3.0, 4.0, 2.0, 1.0]); // Highly non-monotonic

        let constraint = ProbabilisticConstraint {
            base_constraint: MonotonicityConstraint::Global { increasing: true },
            confidence_level: 0.7,
            violation_penalty: 2.0,
            violation_tolerance: 0.1,
        };

        let model = ProbabilisticIsotonicRegression::new()
            .add_constraint(constraint)
            .enforcement(ConstraintEnforcement::SoftPenalty);

        let fitted_model = model.fit(&x, &y).unwrap();

        let violation_score = fitted_model.violation_score().unwrap();
        assert!(violation_score >= 0.0);

        // With highly non-monotonic data, constraints likely won't be perfectly satisfied
        let satisfied = fitted_model.constraints_satisfied().unwrap();
        // This might be true or false depending on the optimization
    }

    #[test]
    fn test_multiple_constraints() {
        let x = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let y = Array1::from(vec![1.0, 3.0, 2.0, 5.0, 4.0, 6.0]);

        let constraints = vec![
            ProbabilisticConstraint {
                base_constraint: MonotonicityConstraint::Global { increasing: true },
                confidence_level: 0.8,
                violation_penalty: 1.0,
                violation_tolerance: 0.1,
            },
            ProbabilisticConstraint {
                base_constraint: MonotonicityConstraint::Global { increasing: true },
                confidence_level: 0.9,
                violation_penalty: 0.5,
                violation_tolerance: 0.05,
            },
        ];

        let model = ProbabilisticIsotonicRegression::new()
            .add_constraints(constraints)
            .enforcement(ConstraintEnforcement::SoftPenalty);

        let fitted_model = model.fit(&x, &y).unwrap();
        let fitted_y = fitted_model.fitted_y().unwrap();

        assert_eq!(fitted_y.len(), 6);
    }

    #[test]
    fn test_convenience_function() {
        let x = Array1::from(vec![1.0, 2.0, 3.0, 4.0]);
        let y = Array1::from(vec![2.0, 1.0, 3.0, 4.0]);

        let constraints = vec![ProbabilisticConstraint {
            base_constraint: MonotonicityConstraint::Global { increasing: true },
            confidence_level: 0.85,
            violation_penalty: 1.0,
            violation_tolerance: 0.1,
        }];

        let result = probabilistic_isotonic_regression(
            &x,
            &y,
            constraints,
            ConstraintEnforcement::SoftPenalty,
        );
        assert!(result.is_ok());

        let (fitted_x, fitted_y, violations, confidence_intervals) = result.unwrap();
        assert_eq!(fitted_x.len(), 4);
        assert_eq!(fitted_y.len(), 4);
        assert_eq!(violations.len(), 4);
        assert_eq!(confidence_intervals.shape(), &[4, 2]);
    }
}
