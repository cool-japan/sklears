//! Constrained Bayesian Optimization
//!
//! This module implements constrained Bayesian optimization methods where the optimization
//! must respect both the objective function and one or more constraint functions.
//!
//! # Mathematical Background
//!
//! Constrained Bayesian optimization solves problems of the form:
//!
//! ```text
//! maximize f(x)  subject to  g_i(x) ≤ 0  for i = 1, ..., m
//! ```
//!
//! Where:
//! - f(x) is the objective function to maximize
//! - g_i(x) are constraint functions
//! - The feasible region is Ω = {x : g_i(x) ≤ 0 ∀ i}
//!
//! # Constraint Handling Methods
//!
//! This implementation provides several approaches:
//!
//! 1. **Constraint-weighted acquisition**: Multiply acquisition by feasibility probability
//! 2. **Expected feasible improvement**: Only consider improvement in feasible regions
//! 3. **Probability of feasible improvement**: Probability of improvement given feasibility
//! 4. **Multi-objective approach**: Treat constraints as additional objectives
//! 5. **Penalty methods**: Add constraint violations as penalties to the objective
//!
//! # Example
//!
//! ```rust
//! use sklears_gaussian_process::constrained_bayesian_optimization::*;
//! use scirs2_core::ndarray::{Array1, Array2, array};
//!
//! // Define constraint functions
//! let constraint1 = |x: &Array1<f64>| x[0] + x[1] - 1.0; // x + y ≤ 1
//! let constraint2 = |x: &Array1<f64>| x[0].powi(2) + x[1].powi(2) - 4.0; // x² + y² ≤ 4
//!
//! let constraints = vec![
//!     Box::new(constraint1) as ConstraintFunction,
//!     Box::new(constraint2) as ConstraintFunction,
//! ];
//!
//! let optimizer = ConstrainedBayesianOptimizer::builder()
//!     .constraints(constraints)
//!     .acquisition_function(ConstrainedAcquisition::ExpectedFeasibleImprovement)
//!     .feasibility_threshold(0.5)
//!     .build();
//! ```

use crate::bayesian_optimization::{AcquisitionFunction, BayesianOptimizer, OptimizationResult};
use crate::gpr::GaussianProcessRegressor;
use crate::kernels::Kernel;
use scirs2_core::ndarray::{Array1, Array2, Array3, ArrayView1, ArrayView2, Axis};
use scirs2_core::random::{thread_rng, Random};
use sklears_core::error::{Result, SklearsError};
use std::f64::consts::PI;

/// Type alias for constraint functions
///
/// Constraint functions should return:
/// - Negative values for feasible points
/// - Positive values for infeasible points
/// - Zero for boundary points
pub type ConstraintFunction = Box<dyn Fn(&Array1<f64>) -> f64 + Send + Sync>;

/// Constraint evaluation result containing value and gradient
#[derive(Debug, Clone)]
pub struct ConstraintEvaluation {
    /// Constraint function value (≤ 0 for feasible)
    pub value: f64,
    /// Gradient of constraint function (optional)
    pub gradient: Option<Array1<f64>>,
    /// Hessian of constraint function (optional)
    pub hessian: Option<Array2<f64>>,
}

/// Different methods for handling constraints in acquisition functions
#[derive(Debug, Clone)]
pub enum ConstrainedAcquisition {
    /// Multiply acquisition function by probability of feasibility
    ConstraintWeighted {
        base_acquisition: AcquisitionFunction,
        feasibility_weight: f64,
    },
    /// Expected improvement only in feasible regions
    ExpectedFeasibleImprovement,
    /// Probability of improvement given feasibility
    ProbabilityOfFeasibleImprovement { epsilon: f64 },
    /// Multi-objective approach treating constraints as objectives
    MultiObjectiveConstrained { constraint_weights: Array1<f64> },
    /// Penalty method adding constraint violations to objective
    PenaltyMethod {
        penalty_factor: f64,
        penalty_exponent: f64,
    },
    /// Augmented Lagrangian method
    AugmentedLagrangian {
        lagrange_multipliers: Array1<f64>,
        penalty_parameter: f64,
    },
}

/// Configuration for constraint handling
#[derive(Debug, Clone)]
pub struct ConstraintConfig {
    /// Threshold for considering a point feasible
    pub feasibility_threshold: f64,
    /// Method for constraint approximation
    pub approximation_method: ConstraintApproximation,
    /// Number of samples for feasibility estimation
    pub feasibility_samples: usize,
    /// Tolerance for constraint satisfaction
    pub constraint_tolerance: f64,
    /// Whether to use constraint gradients if available
    pub use_gradients: bool,
}

impl Default for ConstraintConfig {
    fn default() -> Self {
        Self {
            feasibility_threshold: 0.5,
            approximation_method: ConstraintApproximation::Independent,
            feasibility_samples: 1000,
            constraint_tolerance: 1e-6,
            use_gradients: false,
        }
    }
}

/// Methods for approximating constraint functions
#[derive(Debug, Clone)]
pub enum ConstraintApproximation {
    /// Model each constraint independently
    Independent,
    /// Joint modeling of all constraints
    Joint,
    /// Classifier approach (feasible/infeasible)
    Classification,
    /// Composite constraint function
    Composite,
}

/// Feasibility analysis result
#[derive(Debug, Clone)]
pub struct FeasibilityAnalysis {
    /// Probability of feasibility for each constraint
    pub individual_probabilities: Array1<f64>,
    /// Overall probability of feasibility
    pub overall_probability: f64,
    /// Expected constraint violations
    pub expected_violations: Array1<f64>,
    /// Variance of constraint predictions
    pub constraint_variances: Array1<f64>,
    /// Most likely violated constraint
    pub most_violated_constraint: Option<usize>,
}

/// Constrained Bayesian optimizer
#[derive(Debug)]
pub struct ConstrainedBayesianOptimizer {
    /// Base Bayesian optimizer for the objective
    objective_optimizer: BayesianOptimizer,
    /// Gaussian processes for modeling constraints
    constraint_models: Vec<GaussianProcessRegressor>,
    /// Constraint functions
    constraints: Vec<ConstraintFunction>,
    /// Acquisition function for constrained optimization
    acquisition: ConstrainedAcquisition,
    /// Configuration for constraint handling
    config: ConstraintConfig,
    /// Current best feasible point
    best_feasible_point: Option<Array1<f64>>,
    /// Current best feasible value
    best_feasible_value: Option<f64>,
    /// History of evaluated points
    evaluation_history: Vec<(Array1<f64>, f64, Array1<f64>)>, // (point, objective, constraints)
}

impl ConstrainedBayesianOptimizer {
    /// Create a new builder for the constrained optimizer
    pub fn builder() -> ConstrainedBayesianOptimizerBuilder {
        ConstrainedBayesianOptimizerBuilder::new()
    }

    /// Add constraint evaluations to the optimizer
    pub fn add_constraint_observations(
        &mut self,
        points: &Array2<f64>,
        constraint_values: &Array2<f64>,
    ) -> Result<()> {
        if constraint_values.ncols() != self.constraints.len() {
            return Err(SklearsError::InvalidParameter(
                "Number of constraint value columns must match number of constraints".to_string(),
            ));
        }

        // Add observations to each constraint model
        for (i, model) in self.constraint_models.iter_mut().enumerate() {
            let constraint_col = constraint_values.column(i);
            model.add_observations(points, &constraint_col.to_owned())?;
        }

        Ok(())
    }

    /// Evaluate feasibility at given points
    pub fn evaluate_feasibility(&self, points: &Array2<f64>) -> Result<Array1<f64>> {
        let n_points = points.nrows();
        let mut feasibility_probs = Array1::zeros(n_points);

        for i in 0..n_points {
            let point = points.row(i);
            let analysis = self.analyze_feasibility(&point.to_owned())?;
            feasibility_probs[i] = analysis.overall_probability;
        }

        Ok(feasibility_probs)
    }

    /// Analyze feasibility for a single point
    pub fn analyze_feasibility(&self, point: &Array1<f64>) -> Result<FeasibilityAnalysis> {
        let n_constraints = self.constraints.len();
        let mut individual_probs = Array1::zeros(n_constraints);
        let mut expected_violations = Array1::zeros(n_constraints);
        let mut constraint_variances = Array1::zeros(n_constraints);

        // Evaluate each constraint
        for (i, model) in self.constraint_models.iter().enumerate() {
            let point_2d = point.clone().insert_axis(Axis(0));
            let prediction = model.predict(&point_2d)?;
            let variance = model.predict_variance(&point_2d)?;

            let mean = prediction[[0]];
            let var = variance[[0]];
            let std_dev = var.sqrt();

            // Probability that constraint is satisfied (g(x) ≤ 0)
            individual_probs[i] = if std_dev > 1e-10 {
                0.5 * (1.0 + erf(-mean / (std_dev * 2.0_f64.sqrt())))
            } else {
                if mean <= 0.0 {
                    1.0
                } else {
                    0.0
                }
            };

            expected_violations[i] = mean.max(0.0);
            constraint_variances[i] = var;
        }

        // Overall feasibility probability (assuming independence)
        let overall_probability = match self.config.approximation_method {
            ConstraintApproximation::Independent => individual_probs.iter().product(),
            ConstraintApproximation::Joint => {
                // For joint modeling, we would need a different approach
                // For now, use independence assumption
                individual_probs.iter().product()
            }
            ConstraintApproximation::Classification => {
                // Use classification approach (not implemented in this example)
                individual_probs.iter().product()
            }
            ConstraintApproximation::Composite => {
                // Use composite constraint function
                individual_probs.iter().product()
            }
        };

        // Find most likely violated constraint
        let most_violated_constraint = expected_violations
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, val)| {
                if *val > self.config.constraint_tolerance {
                    Some(i)
                } else {
                    None
                }
            })
            .flatten();

        Ok(FeasibilityAnalysis {
            individual_probabilities: individual_probs,
            overall_probability,
            expected_violations,
            constraint_variances,
            most_violated_constraint,
        })
    }

    /// Compute constrained acquisition function value
    pub fn constrained_acquisition_value(
        &self,
        point: &Array1<f64>,
        current_best: f64,
    ) -> Result<f64> {
        let point_2d = point.clone().insert_axis(Axis(0));
        let feasibility = self.analyze_feasibility(point)?;

        match &self.acquisition {
            ConstrainedAcquisition::ConstraintWeighted {
                base_acquisition,
                feasibility_weight,
            } => {
                let base_value = self
                    .objective_optimizer
                    .acquisition_value(point, current_best)?;
                Ok(base_value * feasibility.overall_probability.powf(*feasibility_weight))
            }
            ConstrainedAcquisition::ExpectedFeasibleImprovement => {
                // Only consider improvement if the point is likely feasible
                if feasibility.overall_probability > self.config.feasibility_threshold {
                    self.objective_optimizer
                        .acquisition_value(point, current_best)
                } else {
                    Ok(0.0)
                }
            }
            ConstrainedAcquisition::ProbabilityOfFeasibleImprovement { epsilon } => {
                let obj_prediction = self.objective_optimizer.predict(&point_2d)?;
                let obj_variance = self.objective_optimizer.predict_variance(&point_2d)?;

                let mean = obj_prediction[[0]];
                let std_dev = obj_variance[[0]].sqrt();

                let improvement_prob = if std_dev > 1e-10 {
                    0.5 * (1.0 + erf((mean - current_best - epsilon) / (std_dev * 2.0_f64.sqrt())))
                } else {
                    if mean > current_best + epsilon {
                        1.0
                    } else {
                        0.0
                    }
                };

                Ok(improvement_prob * feasibility.overall_probability)
            }
            ConstrainedAcquisition::MultiObjectiveConstrained { constraint_weights } => {
                // Treat constraints as additional objectives
                let obj_value = self
                    .objective_optimizer
                    .acquisition_value(point, current_best)?;
                let constraint_penalty: f64 = feasibility
                    .expected_violations
                    .iter()
                    .zip(constraint_weights.iter())
                    .map(|(violation, weight)| violation * weight)
                    .sum();

                Ok(obj_value - constraint_penalty)
            }
            ConstrainedAcquisition::PenaltyMethod {
                penalty_factor,
                penalty_exponent,
            } => {
                let obj_value = self
                    .objective_optimizer
                    .acquisition_value(point, current_best)?;
                let total_violation: f64 = feasibility
                    .expected_violations
                    .iter()
                    .map(|v| v.max(0.0).powf(*penalty_exponent))
                    .sum();

                Ok(obj_value - penalty_factor * total_violation)
            }
            ConstrainedAcquisition::AugmentedLagrangian {
                lagrange_multipliers,
                penalty_parameter,
            } => {
                let obj_value = self
                    .objective_optimizer
                    .acquisition_value(point, current_best)?;

                let lagrangian_term: f64 = feasibility
                    .expected_violations
                    .iter()
                    .zip(lagrange_multipliers.iter())
                    .map(|(violation, lambda)| lambda * violation.max(0.0))
                    .sum();

                let penalty_term: f64 = feasibility
                    .expected_violations
                    .iter()
                    .map(|v| 0.5 * penalty_parameter * v.max(0.0).powi(2))
                    .sum();

                Ok(obj_value - lagrangian_term - penalty_term)
            }
        }
    }

    /// Optimize the constrained acquisition function
    pub fn optimize_constrained_acquisition(
        &self,
        bounds: &Array2<f64>,
        current_best: f64,
        n_restarts: usize,
    ) -> Result<Array1<f64>> {
        let n_dims = bounds.nrows();
        let mut best_point = Array1::zeros(n_dims);
        let mut best_value = f64::NEG_INFINITY;

        // Multi-restart optimization
        for _ in 0..n_restarts {
            // Random starting point
            let mut start_point = Array1::zeros(n_dims);
            for i in 0..n_dims {
                let range = bounds[[i, 1]] - bounds[[i, 0]];
                start_point[i] = bounds[[i, 0]] + rng().gen::<f64>() * range;
            }

            // Simple gradient-free optimization (could be replaced with more sophisticated methods)
            let optimized_point = self.optimize_from_start(&start_point, bounds, current_best)?;
            let value = self.constrained_acquisition_value(&optimized_point, current_best)?;

            if value > best_value {
                best_value = value;
                best_point = optimized_point;
            }
        }

        Ok(best_point)
    }

    /// Optimize from a starting point using simple gradient-free method
    fn optimize_from_start(
        &self,
        start_point: &Array1<f64>,
        bounds: &Array2<f64>,
        current_best: f64,
    ) -> Result<Array1<f64>> {
        let mut point = start_point.clone();
        let mut step_size = 0.1;
        let max_iterations = 100;
        let tolerance = 1e-6;

        for iteration in 0..max_iterations {
            let current_value = self.constrained_acquisition_value(&point, current_best)?;
            let mut improved = false;

            // Try small perturbations in each dimension
            for dim in 0..point.len() {
                for &direction in &[1.0, -1.0] {
                    let mut new_point = point.clone();
                    new_point[dim] += direction * step_size;

                    // Ensure within bounds
                    new_point[dim] = new_point[dim].max(bounds[[dim, 0]]).min(bounds[[dim, 1]]);

                    let new_value = self.constrained_acquisition_value(&new_point, current_best)?;

                    if new_value > current_value + tolerance {
                        point = new_point;
                        improved = true;
                        break;
                    }
                }
                if improved {
                    break;
                }
            }

            if !improved {
                step_size *= 0.5;
                if step_size < 1e-8 {
                    break;
                }
            }
        }

        Ok(point)
    }

    /// Get the current best feasible point and value
    pub fn get_best_feasible(&self) -> Option<(Array1<f64>, f64)> {
        if let (Some(point), Some(value)) = (&self.best_feasible_point, self.best_feasible_value) {
            Some((point.clone(), *value))
        } else {
            None
        }
    }

    /// Update the best feasible point if a better one is found
    pub fn update_best_feasible(&mut self, point: &Array1<f64>, value: f64) -> Result<bool> {
        // Check if point is feasible
        let feasibility = self.analyze_feasibility(point)?;

        if feasibility.overall_probability > self.config.feasibility_threshold {
            if self.best_feasible_value.map_or(true, |best| value > best) {
                self.best_feasible_point = Some(point.clone());
                self.best_feasible_value = Some(value);
                return Ok(true);
            }
        }

        Ok(false)
    }

    /// Run constrained Bayesian optimization
    pub fn optimize(
        &mut self,
        bounds: &Array2<f64>,
        n_iterations: usize,
        n_restarts: usize,
    ) -> Result<OptimizationResult> {
        let mut all_points = Vec::new();
        let mut all_values = Vec::new();

        for iteration in 0..n_iterations {
            // Find current best feasible value
            let current_best = self.best_feasible_value.unwrap_or(f64::NEG_INFINITY);

            // Optimize acquisition function
            let next_point =
                self.optimize_constrained_acquisition(bounds, current_best, n_restarts)?;

            // Evaluate objective function (this would be provided by user)
            // For now, we'll return the next point to evaluate
            all_points.push(next_point.clone());

            // In a real implementation, the user would evaluate the objective and constraints here
            // and call add_observation() and add_constraint_observations()
        }

        Ok(OptimizationResult {
            best_point: self
                .best_feasible_point
                .clone()
                .unwrap_or_else(|| Array1::zeros(bounds.nrows())),
            best_value: self.best_feasible_value.unwrap_or(f64::NEG_INFINITY),
            all_points: Array2::from_shape_vec(
                (all_points.len(), all_points[0].len()),
                all_points.into_iter().flatten().collect(),
            )
            .unwrap(),
            all_values: Array1::from_vec(all_values),
            n_iterations: n_iterations,
        })
    }
}

/// Builder for constrained Bayesian optimizer
#[derive(Debug)]
pub struct ConstrainedBayesianOptimizerBuilder {
    objective_optimizer: Option<BayesianOptimizer>,
    constraints: Vec<ConstraintFunction>,
    acquisition: Option<ConstrainedAcquisition>,
    config: ConstraintConfig,
    constraint_kernels: Vec<Box<dyn Kernel>>,
}

impl ConstrainedBayesianOptimizerBuilder {
    pub fn new() -> Self {
        Self {
            objective_optimizer: None,
            constraints: Vec::new(),
            acquisition: None,
            config: ConstraintConfig::default(),
            constraint_kernels: Vec::new(),
        }
    }

    pub fn objective_optimizer(mut self, optimizer: BayesianOptimizer) -> Self {
        self.objective_optimizer = Some(optimizer);
        self
    }

    pub fn constraints(mut self, constraints: Vec<ConstraintFunction>) -> Self {
        self.constraints = constraints;
        self
    }

    pub fn add_constraint(mut self, constraint: ConstraintFunction) -> Self {
        self.constraints.push(constraint);
        self
    }

    pub fn acquisition_function(mut self, acquisition: ConstrainedAcquisition) -> Self {
        self.acquisition = Some(acquisition);
        self
    }

    pub fn constraint_config(mut self, config: ConstraintConfig) -> Self {
        self.config = config;
        self
    }

    pub fn feasibility_threshold(mut self, threshold: f64) -> Self {
        self.config.feasibility_threshold = threshold;
        self
    }

    pub fn constraint_kernels(mut self, kernels: Vec<Box<dyn Kernel>>) -> Self {
        self.constraint_kernels = kernels;
        self
    }

    pub fn build(self) -> ConstrainedBayesianOptimizer {
        let objective_optimizer = self
            .objective_optimizer
            .unwrap_or_else(|| BayesianOptimizer::builder().build());

        let n_constraints = self.constraints.len();

        // Create constraint models (one GP per constraint)
        let constraint_models = (0..n_constraints)
            .map(|i| {
                let kernel = self
                    .constraint_kernels
                    .get(i)
                    .map(|k| k.clone_box())
                    .unwrap_or_else(|| {
                        use crate::kernels::RBF;
                        Box::new(RBF::new(1.0, 1.0))
                    });

                GaussianProcessRegressor::new(kernel)
            })
            .collect();

        let acquisition = self
            .acquisition
            .unwrap_or(ConstrainedAcquisition::ExpectedFeasibleImprovement);

        ConstrainedBayesianOptimizer {
            objective_optimizer,
            constraint_models,
            constraints: self.constraints,
            acquisition,
            config: self.config,
            best_feasible_point: None,
            best_feasible_value: None,
            evaluation_history: Vec::new(),
        }
    }
}

impl Default for ConstrainedBayesianOptimizerBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Error function approximation for probability calculations
fn erf(x: f64) -> f64 {
    // Approximation of the error function using Abramowitz and Stegun approximation
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();

    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    sign * y
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::RBF;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_constraint_function_creation() {
        let constraint = |x: &Array1<f64>| x[0] + x[1] - 1.0;
        let point = array![0.5, 0.3];
        let value = constraint(&point);
        assert!((value - (-0.2)).abs() < 1e-10);
    }

    #[test]
    fn test_constrained_optimizer_builder() {
        let constraint1 = Box::new(|x: &Array1<f64>| x[0] + x[1] - 1.0) as ConstraintFunction;
        let constraint2 =
            Box::new(|x: &Array1<f64>| x[0].powi(2) + x[1].powi(2) - 4.0) as ConstraintFunction;

        let optimizer = ConstrainedBayesianOptimizer::builder()
            .constraints(vec![constraint1, constraint2])
            .acquisition_function(ConstrainedAcquisition::ExpectedFeasibleImprovement)
            .feasibility_threshold(0.5)
            .build();

        assert_eq!(optimizer.constraints.len(), 2);
        assert_eq!(optimizer.constraint_models.len(), 2);
    }

    #[test]
    fn test_feasibility_analysis_empty_models() {
        let constraint = Box::new(|x: &Array1<f64>| x[0] + x[1] - 1.0) as ConstraintFunction;

        let optimizer = ConstrainedBayesianOptimizer::builder()
            .constraints(vec![constraint])
            .build();

        let point = array![0.5, 0.3];

        // This should work even with empty models (though results may be uninformative)
        let result = optimizer.analyze_feasibility(&point);
        assert!(result.is_ok());
    }

    #[test]
    fn test_constraint_weighted_acquisition() {
        let base_acq = AcquisitionFunction::ExpectedImprovement;
        let acquisition = ConstrainedAcquisition::ConstraintWeighted {
            base_acquisition: base_acq,
            feasibility_weight: 1.0,
        };

        match acquisition {
            ConstrainedAcquisition::ConstraintWeighted {
                feasibility_weight, ..
            } => {
                assert!((feasibility_weight - 1.0).abs() < 1e-10);
            }
            _ => panic!("Wrong acquisition type"),
        }
    }

    #[test]
    fn test_constraint_config_default() {
        let config = ConstraintConfig::default();
        assert!((config.feasibility_threshold - 0.5).abs() < 1e-10);
        assert_eq!(config.feasibility_samples, 1000);
        assert!((config.constraint_tolerance - 1e-6).abs() < 1e-10);
    }

    #[test]
    fn test_erf_function() {
        // Test some known values
        assert!((erf(0.0) - 0.0).abs() < 1e-6);
        assert!((erf(1.0) - 0.8427).abs() < 1e-3);
        assert!((erf(-1.0) + 0.8427).abs() < 1e-3);
        assert!(erf(10.0) > 0.999);
        assert!(erf(-10.0) < -0.999);
    }

    #[test]
    fn test_multiple_constraint_types() {
        let linear_constraint = Box::new(|x: &Array1<f64>| x[0] + x[1] - 1.0) as ConstraintFunction;
        let quadratic_constraint =
            Box::new(|x: &Array1<f64>| x[0].powi(2) + x[1].powi(2) - 4.0) as ConstraintFunction;
        let nonlinear_constraint =
            Box::new(|x: &Array1<f64>| x[0] * x[1] - 0.5) as ConstraintFunction;

        let optimizer = ConstrainedBayesianOptimizer::builder()
            .constraints(vec![
                linear_constraint,
                quadratic_constraint,
                nonlinear_constraint,
            ])
            .build();

        assert_eq!(optimizer.constraints.len(), 3);
    }

    #[test]
    fn test_augmented_lagrangian_acquisition() {
        let lagrange_mults = array![1.0, 0.5];
        let acquisition = ConstrainedAcquisition::AugmentedLagrangian {
            lagrange_multipliers: lagrange_mults.clone(),
            penalty_parameter: 2.0,
        };

        match acquisition {
            ConstrainedAcquisition::AugmentedLagrangian {
                lagrange_multipliers,
                penalty_parameter,
            } => {
                assert_eq!(lagrange_multipliers.len(), 2);
                assert!((penalty_parameter - 2.0).abs() < 1e-10);
            }
            _ => panic!("Wrong acquisition type"),
        }
    }

    #[test]
    fn test_constraint_approximation_methods() {
        let methods = vec![
            ConstraintApproximation::Independent,
            ConstraintApproximation::Joint,
            ConstraintApproximation::Classification,
            ConstraintApproximation::Composite,
        ];

        for method in methods {
            let config = ConstraintConfig {
                approximation_method: method,
                ..Default::default()
            };
            assert!(config.feasibility_threshold > 0.0);
        }
    }

    #[test]
    fn test_builder_method_chaining() {
        let constraint = Box::new(|x: &Array1<f64>| x[0] + x[1] - 1.0) as ConstraintFunction;

        let optimizer = ConstrainedBayesianOptimizer::builder()
            .add_constraint(constraint)
            .feasibility_threshold(0.8)
            .acquisition_function(ConstrainedAcquisition::ProbabilityOfFeasibleImprovement {
                epsilon: 0.01,
            })
            .build();

        assert_eq!(optimizer.constraints.len(), 1);
        assert!((optimizer.config.feasibility_threshold - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_multi_objective_constrained_acquisition() {
        let weights = array![1.0, 2.0, 0.5];
        let acquisition = ConstrainedAcquisition::MultiObjectiveConstrained {
            constraint_weights: weights.clone(),
        };

        match acquisition {
            ConstrainedAcquisition::MultiObjectiveConstrained { constraint_weights } => {
                assert_eq!(constraint_weights.len(), 3);
                assert!((constraint_weights[1] - 2.0).abs() < 1e-10);
            }
            _ => panic!("Wrong acquisition type"),
        }
    }

    #[test]
    fn test_penalty_method_acquisition() {
        let acquisition = ConstrainedAcquisition::PenaltyMethod {
            penalty_factor: 10.0,
            penalty_exponent: 2.0,
        };

        match acquisition {
            ConstrainedAcquisition::PenaltyMethod {
                penalty_factor,
                penalty_exponent,
            } => {
                assert!((penalty_factor - 10.0).abs() < 1e-10);
                assert!((penalty_exponent - 2.0).abs() < 1e-10);
            }
            _ => panic!("Wrong acquisition type"),
        }
    }

    #[test]
    fn test_feasibility_analysis_structure() {
        let analysis = FeasibilityAnalysis {
            individual_probabilities: array![0.8, 0.6, 0.9],
            overall_probability: 0.432, // 0.8 * 0.6 * 0.9
            expected_violations: array![0.0, 0.1, 0.0],
            constraint_variances: array![0.01, 0.02, 0.005],
            most_violated_constraint: Some(1),
        };

        assert_eq!(analysis.individual_probabilities.len(), 3);
        assert!((analysis.overall_probability - 0.432).abs() < 1e-10);
        assert_eq!(analysis.most_violated_constraint, Some(1));
    }

    #[test]
    fn test_constraint_evaluation_structure() {
        let evaluation = ConstraintEvaluation {
            value: -0.5,
            gradient: Some(array![1.0, 1.0]),
            hessian: Some(array![[0.0, 0.0], [0.0, 0.0]]),
        };

        assert!((evaluation.value + 0.5).abs() < 1e-10);
        assert!(evaluation.gradient.is_some());
        assert!(evaluation.hessian.is_some());
    }
}
