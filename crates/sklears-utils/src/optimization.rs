//! Optimization utilities for numerical optimization algorithms
//!
//! This module provides utilities for optimization algorithms including:
//! - Line search methods (Armijo, Wolfe conditions)
//! - Convergence criteria checking
//! - Gradient computation helpers
//! - Constraint handling utilities
//!
//! # Examples
//!
//! ```rust
//! use sklears_utils::optimization::{LineSearch, ConvergenceCriteria, GradientComputer};
//! use scirs2_core::ndarray::Array1;
//!
//! let line_search = LineSearch::armijo(1e-4);
//! let conv_criteria = ConvergenceCriteria::new()
//!     .with_tolerance(1e-6)
//!     .with_max_iterations(1000);
//! ```

use crate::UtilsResult;
use scirs2_core::ndarray::{Array1, Array2, ArrayView1};
use std::collections::VecDeque;

/// Line search methods for optimization algorithms
#[derive(Debug, Clone)]
pub struct LineSearch {
    pub method: LineSearchMethod,
    pub c1: f64, // Armijo condition parameter
    pub c2: f64, // Wolfe condition parameter
    pub max_iterations: usize,
    pub initial_step: f64,
    pub step_decay: f64,
}

#[derive(Debug, Clone)]
pub enum LineSearchMethod {
    Armijo,
    Wolfe,
    StrongWolfe,
    Backtracking,
}

impl LineSearch {
    /// Create Armijo line search with default parameters
    pub fn armijo(c1: f64) -> Self {
        Self {
            method: LineSearchMethod::Armijo,
            c1,
            c2: 0.9,
            max_iterations: 50,
            initial_step: 1.0,
            step_decay: 0.5,
        }
    }

    /// Create Wolfe line search with default parameters
    pub fn wolfe(c1: f64, c2: f64) -> Self {
        Self {
            method: LineSearchMethod::Wolfe,
            c1,
            c2,
            max_iterations: 50,
            initial_step: 1.0,
            step_decay: 0.5,
        }
    }

    /// Create strong Wolfe line search
    pub fn strong_wolfe(c1: f64, c2: f64) -> Self {
        Self {
            method: LineSearchMethod::StrongWolfe,
            c1,
            c2,
            max_iterations: 50,
            initial_step: 1.0,
            step_decay: 0.5,
        }
    }

    /// Create backtracking line search
    pub fn backtracking(c1: f64) -> Self {
        Self {
            method: LineSearchMethod::Backtracking,
            c1,
            c2: 0.9,
            max_iterations: 50,
            initial_step: 1.0,
            step_decay: 0.5,
        }
    }

    /// Perform line search to find appropriate step size
    pub fn search<F, G>(
        &self,
        f: F,
        grad_f: G,
        x: &ArrayView1<f64>,
        direction: &ArrayView1<f64>,
        f_x: f64,
        grad_x: &ArrayView1<f64>,
    ) -> UtilsResult<f64>
    where
        F: Fn(&ArrayView1<f64>) -> f64,
        G: Fn(&ArrayView1<f64>) -> Array1<f64>,
    {
        match self.method {
            LineSearchMethod::Armijo => self.armijo_search(f, x, direction, f_x, grad_x),
            LineSearchMethod::Backtracking => {
                self.backtracking_search(f, x, direction, f_x, grad_x)
            }
            LineSearchMethod::Wolfe => self.wolfe_search(f, grad_f, x, direction, f_x, grad_x),
            LineSearchMethod::StrongWolfe => {
                self.strong_wolfe_search(f, grad_f, x, direction, f_x, grad_x)
            }
        }
    }

    fn armijo_search<F>(
        &self,
        f: F,
        x: &ArrayView1<f64>,
        direction: &ArrayView1<f64>,
        f_x: f64,
        grad_x: &ArrayView1<f64>,
    ) -> UtilsResult<f64>
    where
        F: Fn(&ArrayView1<f64>) -> f64,
    {
        let mut alpha = self.initial_step;
        let directional_derivative = grad_x.dot(direction);

        for _ in 0..self.max_iterations {
            let x_new = x + &(direction * alpha);
            let f_new = f(&x_new.view());

            // Armijo condition: f(x + α*p) ≤ f(x) + c₁*α*∇f(x)ᵀp
            if f_new <= f_x + self.c1 * alpha * directional_derivative {
                return Ok(alpha);
            }

            alpha *= self.step_decay;
        }

        Ok(alpha) // Return last alpha even if conditions not met
    }

    fn backtracking_search<F>(
        &self,
        f: F,
        x: &ArrayView1<f64>,
        direction: &ArrayView1<f64>,
        f_x: f64,
        grad_x: &ArrayView1<f64>,
    ) -> UtilsResult<f64>
    where
        F: Fn(&ArrayView1<f64>) -> f64,
    {
        self.armijo_search(f, x, direction, f_x, grad_x)
    }

    fn wolfe_search<F, G>(
        &self,
        f: F,
        grad_f: G,
        x: &ArrayView1<f64>,
        direction: &ArrayView1<f64>,
        f_x: f64,
        grad_x: &ArrayView1<f64>,
    ) -> UtilsResult<f64>
    where
        F: Fn(&ArrayView1<f64>) -> f64,
        G: Fn(&ArrayView1<f64>) -> Array1<f64>,
    {
        let mut alpha = self.initial_step;
        let directional_derivative = grad_x.dot(direction);

        for _ in 0..self.max_iterations {
            let x_new = x + &(direction * alpha);
            let f_new = f(&x_new.view());

            // Armijo condition
            if f_new > f_x + self.c1 * alpha * directional_derivative {
                alpha *= self.step_decay;
                continue;
            }

            // Wolfe condition: ∇f(x + α*p)ᵀp ≥ c₂*∇f(x)ᵀp
            let grad_new = grad_f(&x_new.view());
            let new_directional_derivative = grad_new.dot(direction);

            if new_directional_derivative >= self.c2 * directional_derivative {
                return Ok(alpha);
            }

            alpha /= self.step_decay; // Increase step size
        }

        Ok(alpha)
    }

    fn strong_wolfe_search<F, G>(
        &self,
        f: F,
        grad_f: G,
        x: &ArrayView1<f64>,
        direction: &ArrayView1<f64>,
        f_x: f64,
        grad_x: &ArrayView1<f64>,
    ) -> UtilsResult<f64>
    where
        F: Fn(&ArrayView1<f64>) -> f64,
        G: Fn(&ArrayView1<f64>) -> Array1<f64>,
    {
        let mut alpha = self.initial_step;
        let directional_derivative = grad_x.dot(direction);

        for _ in 0..self.max_iterations {
            let x_new = x + &(direction * alpha);
            let f_new = f(&x_new.view());

            // Armijo condition
            if f_new > f_x + self.c1 * alpha * directional_derivative {
                alpha *= self.step_decay;
                continue;
            }

            // Strong Wolfe condition: |∇f(x + α*p)ᵀp| ≤ c₂*|∇f(x)ᵀp|
            let grad_new = grad_f(&x_new.view());
            let new_directional_derivative = grad_new.dot(direction);

            if new_directional_derivative.abs() <= self.c2 * directional_derivative.abs() {
                return Ok(alpha);
            }

            alpha *= if new_directional_derivative < 0.0 {
                1.0 / self.step_decay
            } else {
                self.step_decay
            };
        }

        Ok(alpha)
    }
}

/// Convergence criteria for optimization algorithms
#[derive(Debug, Clone)]
pub struct ConvergenceCriteria {
    pub tolerance: f64,
    pub gradient_tolerance: f64,
    pub parameter_tolerance: f64,
    pub max_iterations: usize,
    pub min_iterations: usize,
    pub function_tolerance: f64,
    pub patience: usize,
}

impl Default for ConvergenceCriteria {
    fn default() -> Self {
        Self {
            tolerance: 1e-6,
            gradient_tolerance: 1e-6,
            parameter_tolerance: 1e-8,
            max_iterations: 1000,
            min_iterations: 1,
            function_tolerance: 1e-9,
            patience: 10,
        }
    }
}

impl ConvergenceCriteria {
    /// Create new convergence criteria with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Set gradient tolerance
    pub fn with_tolerance(mut self, tol: f64) -> Self {
        self.tolerance = tol;
        self
    }

    /// Set gradient tolerance
    pub fn with_gradient_tolerance(mut self, tol: f64) -> Self {
        self.gradient_tolerance = tol;
        self
    }

    /// Set parameter tolerance
    pub fn with_parameter_tolerance(mut self, tol: f64) -> Self {
        self.parameter_tolerance = tol;
        self
    }

    /// Set maximum iterations
    pub fn with_max_iterations(mut self, max_iter: usize) -> Self {
        self.max_iterations = max_iter;
        self
    }

    /// Set minimum iterations
    pub fn with_min_iterations(mut self, min_iter: usize) -> Self {
        self.min_iterations = min_iter;
        self
    }

    /// Set function tolerance
    pub fn with_function_tolerance(mut self, tol: f64) -> Self {
        self.function_tolerance = tol;
        self
    }

    /// Set patience for early stopping
    pub fn with_patience(mut self, patience: usize) -> Self {
        self.patience = patience;
        self
    }

    /// Check if convergence is achieved
    pub fn is_converged(
        &self,
        iteration: usize,
        current_f: f64,
        previous_f: Option<f64>,
        gradient: Option<&ArrayView1<f64>>,
        parameter_change: Option<f64>,
        no_improvement_count: usize,
    ) -> ConvergenceStatus {
        // Check minimum iterations
        if iteration < self.min_iterations {
            return ConvergenceStatus::Continuing;
        }

        // Check maximum iterations
        if iteration >= self.max_iterations {
            return ConvergenceStatus::MaxIterationsReached;
        }

        // Check gradient tolerance
        if let Some(grad) = gradient {
            let grad_norm = grad.iter().map(|x| x * x).sum::<f64>().sqrt();
            if grad_norm < self.gradient_tolerance {
                return ConvergenceStatus::GradientTolerance;
            }
        }

        // Check function tolerance
        if let Some(prev_f) = previous_f {
            let f_change = (current_f - prev_f).abs();
            if f_change < self.function_tolerance {
                return ConvergenceStatus::FunctionTolerance;
            }
        }

        // Check parameter tolerance
        if let Some(param_change) = parameter_change {
            if param_change < self.parameter_tolerance {
                return ConvergenceStatus::ParameterTolerance;
            }
        }

        // Check early stopping patience
        if no_improvement_count >= self.patience {
            return ConvergenceStatus::NoImprovement;
        }

        ConvergenceStatus::Continuing
    }
}

/// Status of convergence checking
#[derive(Debug, Clone, PartialEq)]
pub enum ConvergenceStatus {
    Continuing,
    GradientTolerance,
    FunctionTolerance,
    ParameterTolerance,
    MaxIterationsReached,
    NoImprovement,
}

impl ConvergenceStatus {
    pub fn is_converged(&self) -> bool {
        !matches!(self, ConvergenceStatus::Continuing)
    }

    pub fn is_successful(&self) -> bool {
        matches!(
            self,
            ConvergenceStatus::GradientTolerance
                | ConvergenceStatus::FunctionTolerance
                | ConvergenceStatus::ParameterTolerance
        )
    }
}

/// Gradient computation utilities
#[derive(Debug, Clone)]
pub struct GradientComputer {
    pub method: GradientMethod,
    pub epsilon: f64,
    pub parallel: bool,
}

#[derive(Debug, Clone)]
pub enum GradientMethod {
    Forward,
    Backward,
    Central,
}

impl Default for GradientComputer {
    fn default() -> Self {
        Self {
            method: GradientMethod::Central,
            epsilon: 1e-8,
            parallel: false,
        }
    }
}

impl GradientComputer {
    /// Create new gradient computer
    pub fn new() -> Self {
        Self::default()
    }

    /// Set gradient method
    pub fn with_method(mut self, method: GradientMethod) -> Self {
        self.method = method;
        self
    }

    /// Set finite difference epsilon
    pub fn with_epsilon(mut self, eps: f64) -> Self {
        self.epsilon = eps;
        self
    }

    /// Enable parallel computation
    pub fn with_parallel(mut self, parallel: bool) -> Self {
        self.parallel = parallel;
        self
    }

    /// Compute numerical gradient using finite differences
    pub fn compute_gradient<F>(&self, f: F, x: &ArrayView1<f64>) -> UtilsResult<Array1<f64>>
    where
        F: Fn(&ArrayView1<f64>) -> f64 + Sync,
    {
        let n = x.len();
        let mut gradient = Array1::zeros(n);

        match self.method {
            GradientMethod::Forward => {
                for i in 0..n {
                    let mut x_plus = x.to_owned();
                    x_plus[i] += self.epsilon;
                    gradient[i] = (f(&x_plus.view()) - f(x)) / self.epsilon;
                }
            }
            GradientMethod::Backward => {
                for i in 0..n {
                    let mut x_minus = x.to_owned();
                    x_minus[i] -= self.epsilon;
                    gradient[i] = (f(x) - f(&x_minus.view())) / self.epsilon;
                }
            }
            GradientMethod::Central => {
                for i in 0..n {
                    let mut x_plus = x.to_owned();
                    let mut x_minus = x.to_owned();
                    x_plus[i] += self.epsilon;
                    x_minus[i] -= self.epsilon;
                    gradient[i] = (f(&x_plus.view()) - f(&x_minus.view())) / (2.0 * self.epsilon);
                }
            }
        }

        Ok(gradient)
    }

    /// Compute Jacobian matrix for vector-valued functions
    pub fn compute_jacobian<F>(
        &self,
        f: F,
        x: &ArrayView1<f64>,
        m: usize,
    ) -> UtilsResult<Array2<f64>>
    where
        F: Fn(&ArrayView1<f64>) -> Array1<f64> + Sync,
    {
        let n = x.len();
        let mut jacobian = Array2::zeros((m, n));

        match self.method {
            GradientMethod::Central => {
                for j in 0..n {
                    let mut x_plus = x.to_owned();
                    let mut x_minus = x.to_owned();
                    x_plus[j] += self.epsilon;
                    x_minus[j] -= self.epsilon;

                    let f_plus = f(&x_plus.view());
                    let f_minus = f(&x_minus.view());

                    for i in 0..m {
                        jacobian[[i, j]] = (f_plus[i] - f_minus[i]) / (2.0 * self.epsilon);
                    }
                }
            }
            _ => {
                // For Jacobian, prefer central differences for accuracy
                return self
                    .clone()
                    .with_method(GradientMethod::Central)
                    .compute_jacobian(f, x, m);
            }
        }

        Ok(jacobian)
    }
}

/// Type alias for constraint function
pub type ConstraintFunction = Box<dyn Fn(&ArrayView1<f64>) -> f64 + Send + Sync>;

/// Constraint handling utilities for constrained optimization
pub struct ConstraintHandler {
    pub equality_constraints: Vec<ConstraintFunction>,
    pub inequality_constraints: Vec<ConstraintFunction>,
    pub bounds: Option<(Array1<f64>, Array1<f64>)>, // (lower, upper)
    pub penalty_parameter: f64,
    pub tolerance: f64,
}

impl Default for ConstraintHandler {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for ConstraintHandler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConstraintHandler")
            .field(
                "equality_constraints",
                &format!("{} functions", self.equality_constraints.len()),
            )
            .field(
                "inequality_constraints",
                &format!("{} functions", self.inequality_constraints.len()),
            )
            .field("bounds", &self.bounds)
            .field("penalty_parameter", &self.penalty_parameter)
            .field("tolerance", &self.tolerance)
            .finish()
    }
}

#[derive(Debug, Clone)]
pub struct ConstraintViolation {
    pub equality_violations: Vec<f64>,
    pub inequality_violations: Vec<f64>,
    pub bound_violations: Vec<f64>,
    pub max_violation: f64,
    pub total_violation: f64,
}

impl ConstraintHandler {
    /// Create new constraint handler
    pub fn new() -> Self {
        Self {
            equality_constraints: Vec::new(),
            inequality_constraints: Vec::new(),
            bounds: None,
            penalty_parameter: 1.0,
            tolerance: 1e-6,
        }
    }

    /// Set bounds constraints
    pub fn with_bounds(mut self, lower: Array1<f64>, upper: Array1<f64>) -> Self {
        self.bounds = Some((lower, upper));
        self
    }

    /// Set penalty parameter
    pub fn with_penalty_parameter(mut self, penalty: f64) -> Self {
        self.penalty_parameter = penalty;
        self
    }

    /// Set tolerance
    pub fn with_tolerance(mut self, tol: f64) -> Self {
        self.tolerance = tol;
        self
    }

    /// Project point onto bounds
    pub fn project_bounds(&self, x: &Array1<f64>) -> Array1<f64> {
        if let Some((ref lower, ref upper)) = self.bounds {
            x.iter()
                .zip(lower.iter())
                .zip(upper.iter())
                .map(|((x_i, l_i), u_i)| x_i.max(*l_i).min(*u_i))
                .collect::<Array1<f64>>()
        } else {
            x.clone()
        }
    }

    /// Check constraint violations
    pub fn check_violations(&self, x: &ArrayView1<f64>) -> ConstraintViolation {
        let mut equality_violations = Vec::new();
        let mut inequality_violations = Vec::new();
        let mut bound_violations = Vec::new();

        // Check equality constraints: c_eq(x) = 0
        for constraint in &self.equality_constraints {
            let violation = constraint(x).abs();
            equality_violations.push(violation);
        }

        // Check inequality constraints: c_ineq(x) <= 0
        for constraint in &self.inequality_constraints {
            let value = constraint(x);
            let violation = if value > 0.0 { value } else { 0.0 };
            inequality_violations.push(violation);
        }

        // Check bound constraints
        if let Some((ref lower, ref upper)) = self.bounds {
            for i in 0..x.len() {
                let x_i = x[i];
                let lower_violation = if x_i < lower[i] { lower[i] - x_i } else { 0.0 };
                let upper_violation = if x_i > upper[i] { x_i - upper[i] } else { 0.0 };
                bound_violations.push(lower_violation + upper_violation);
            }
        }

        let max_violation = equality_violations
            .iter()
            .chain(&inequality_violations)
            .chain(&bound_violations)
            .fold(0.0f64, |acc, &x| acc.max(x));

        let total_violation = equality_violations.iter().sum::<f64>()
            + inequality_violations.iter().sum::<f64>()
            + bound_violations.iter().sum::<f64>();

        ConstraintViolation {
            equality_violations,
            inequality_violations,
            bound_violations,
            max_violation,
            total_violation,
        }
    }

    /// Check if constraints are satisfied
    pub fn is_feasible(&self, x: &ArrayView1<f64>) -> bool {
        let violations = self.check_violations(x);
        violations.max_violation <= self.tolerance
    }

    /// Compute penalty function value
    pub fn penalty_function(&self, x: &ArrayView1<f64>) -> f64 {
        let violations = self.check_violations(x);
        self.penalty_parameter * violations.total_violation
    }
}

/// History tracking for optimization algorithms
#[derive(Debug, Clone)]
pub struct OptimizationHistory {
    pub function_values: VecDeque<f64>,
    pub gradient_norms: VecDeque<f64>,
    pub parameter_changes: VecDeque<f64>,
    pub step_sizes: VecDeque<f64>,
    pub max_history_size: usize,
}

impl OptimizationHistory {
    /// Create new optimization history tracker
    pub fn new(max_size: usize) -> Self {
        Self {
            function_values: VecDeque::new(),
            gradient_norms: VecDeque::new(),
            parameter_changes: VecDeque::new(),
            step_sizes: VecDeque::new(),
            max_history_size: max_size,
        }
    }

    /// Add function value to history
    pub fn add_function_value(&mut self, value: f64) {
        if self.function_values.len() >= self.max_history_size {
            self.function_values.pop_front();
        }
        self.function_values.push_back(value);
    }

    /// Add gradient norm to history
    pub fn add_gradient_norm(&mut self, norm: f64) {
        if self.gradient_norms.len() >= self.max_history_size {
            self.gradient_norms.pop_front();
        }
        self.gradient_norms.push_back(norm);
    }

    /// Add parameter change to history
    pub fn add_parameter_change(&mut self, change: f64) {
        if self.parameter_changes.len() >= self.max_history_size {
            self.parameter_changes.pop_front();
        }
        self.parameter_changes.push_back(change);
    }

    /// Add step size to history
    pub fn add_step_size(&mut self, step: f64) {
        if self.step_sizes.len() >= self.max_history_size {
            self.step_sizes.pop_front();
        }
        self.step_sizes.push_back(step);
    }

    /// Get recent function values
    pub fn recent_function_values(&self, n: usize) -> Vec<f64> {
        self.function_values.iter().rev().take(n).cloned().collect()
    }

    /// Check for improvement trend
    pub fn has_improvement_trend(&self, window_size: usize) -> bool {
        if self.function_values.len() < window_size + 1 {
            return false;
        }

        let recent = self.recent_function_values(window_size + 1);
        if recent.len() < 2 {
            return false;
        }

        // Check if function values are decreasing
        recent.windows(2).all(|w| w[0] < w[1])
    }

    /// Get average improvement rate
    pub fn average_improvement_rate(&self, window_size: usize) -> Option<f64> {
        if self.function_values.len() < window_size + 1 {
            return None;
        }

        let recent = self.recent_function_values(window_size + 1);
        if recent.len() < 2 {
            return None;
        }

        let improvements: Vec<f64> = recent.windows(2).map(|w| w[1] - w[0]).collect();

        let avg_improvement = improvements.iter().sum::<f64>() / improvements.len() as f64;
        Some(avg_improvement)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_line_search_armijo() {
        let line_search = LineSearch::armijo(1e-4);

        // Simple quadratic function: f(x) = x^2
        let f = |x: &ArrayView1<f64>| x[0] * x[0];
        let grad_f = |x: &ArrayView1<f64>| array![2.0 * x[0]];

        let x = array![2.0];
        let direction = array![-1.0]; // Descent direction
        let f_x = f(&x.view());
        let grad_x = grad_f(&x.view());

        let alpha = line_search
            .search(f, grad_f, &x.view(), &direction.view(), f_x, &grad_x.view())
            .unwrap();

        assert!(alpha > 0.0);
        assert!(alpha <= 1.0);
    }

    #[test]
    fn test_convergence_criteria() {
        let criteria = ConvergenceCriteria::new()
            .with_tolerance(1e-6)
            .with_max_iterations(100);

        // Test continuing
        let status = criteria.is_converged(50, 1.0, Some(1.1), None, None, 0);
        assert_eq!(status, ConvergenceStatus::Continuing);

        // Test max iterations
        let status = criteria.is_converged(100, 1.0, Some(1.1), None, None, 0);
        assert_eq!(status, ConvergenceStatus::MaxIterationsReached);

        // Test gradient tolerance
        let small_grad = array![1e-7];
        let status = criteria.is_converged(50, 1.0, Some(1.1), Some(&small_grad.view()), None, 0);
        assert_eq!(status, ConvergenceStatus::GradientTolerance);
    }

    #[test]
    fn test_gradient_computer() {
        let grad_computer = GradientComputer::new();

        // Test on quadratic function: f(x) = x₁² + x₂²
        let f = |x: &ArrayView1<f64>| x[0] * x[0] + x[1] * x[1];
        let x = array![2.0, 3.0];

        let gradient = grad_computer.compute_gradient(f, &x.view()).unwrap();

        // Analytical gradient should be [2*x₁, 2*x₂] = [4.0, 6.0]
        assert!((gradient[0] - 4.0).abs() < 1e-6);
        assert!((gradient[1] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_constraint_handler_bounds() {
        let lower = array![-1.0, -2.0];
        let upper = array![1.0, 2.0];
        let handler = ConstraintHandler::new().with_bounds(lower, upper);

        // Test point within bounds
        let x_feasible = array![0.5, 1.0];
        assert!(handler.is_feasible(&x_feasible.view()));

        // Test point outside bounds
        let x_infeasible = array![2.0, -3.0];
        assert!(!handler.is_feasible(&x_infeasible.view()));

        // Test projection
        let x_projected = handler.project_bounds(&x_infeasible);
        assert_eq!(x_projected, array![1.0, -2.0]);
        assert!(handler.is_feasible(&x_projected.view()));
    }

    #[test]
    fn test_optimization_history() {
        let mut history = OptimizationHistory::new(5);

        // Add some function values
        for i in 0..7 {
            history.add_function_value(i as f64);
        }

        // Should only keep last 5 values
        assert_eq!(history.function_values.len(), 5);
        assert_eq!(history.recent_function_values(3), vec![6.0, 5.0, 4.0]);

        // Test improvement trend (values are increasing, so no improvement)
        assert!(!history.has_improvement_trend(3));

        // Add decreasing values
        let mut history2 = OptimizationHistory::new(10);
        for i in (0..5).rev() {
            history2.add_function_value(i as f64);
        }

        assert!(history2.has_improvement_trend(3));
    }

    #[test]
    fn test_jacobian_computation() {
        let grad_computer = GradientComputer::new();

        // Test vector function: f(x) = [x₁², x₁*x₂]
        let f = |x: &ArrayView1<f64>| array![x[0] * x[0], x[0] * x[1]];
        let x = array![2.0, 3.0];

        let jacobian = grad_computer.compute_jacobian(f, &x.view(), 2).unwrap();

        // Analytical Jacobian should be:
        // [[2*x₁, 0  ],     [[4, 0],
        //  [x₂,   x₁]]  =    [3, 2]]
        assert!((jacobian[[0, 0]] - 4.0).abs() < 1e-6);
        assert!((jacobian[[0, 1]] - 0.0).abs() < 1e-6);
        assert!((jacobian[[1, 0]] - 3.0).abs() < 1e-6);
        assert!((jacobian[[1, 1]] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_constraint_violations() {
        let handler = ConstraintHandler::new()
            .with_bounds(array![-1.0, -1.0], array![1.0, 1.0])
            .with_tolerance(1e-6);

        let x_violating = array![2.0, -2.0];
        let violations = handler.check_violations(&x_violating.view());

        assert!(violations.max_violation > 0.0);
        assert!(violations.total_violation > 0.0);
        assert_eq!(violations.bound_violations.len(), 2);
        assert!(violations.bound_violations[0] > 0.0); // Upper bound violation
        assert!(violations.bound_violations[1] > 0.0); // Lower bound violation
    }
}
