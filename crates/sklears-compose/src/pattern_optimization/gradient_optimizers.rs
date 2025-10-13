//! Gradient-based optimization algorithms and first/second-order methods
//!
//! This module provides gradient-based optimization algorithms including:
//! - Gradient descent variants (vanilla, momentum, Adam, RMSprop)
//! - Quasi-Newton methods (BFGS, L-BFGS, DFP, SR1)
//! - Trust region methods with subproblem solvers
//! - Line search methods with Wolfe conditions
//! - Conjugate gradient methods (Fletcher-Reeves, Polak-Ribière, etc.)
//! - Second-order methods (Newton's method, modified Newton)
//! - Gradient and Hessian estimation techniques

use std::collections::HashMap;
use std::time::SystemTime;

use scirs2_core::ndarray::{Array1, Array2};
use crate::core::SklResult;
use super::optimization_core::OptimizationProblem;

/// Gradient-based optimizer coordinating derivative-based methods
///
/// Manages various gradient-based algorithms and provides automatic
/// method selection based on problem characteristics and available derivatives.
#[derive(Debug)]
pub struct GradientBasedOptimizer {
    /// Unique optimizer identifier
    pub optimizer_id: String,
    /// Gradient descent algorithm variants
    pub gradient_descent_variants: HashMap<String, Box<dyn GradientDescentAlgorithm>>,
    /// Quasi-Newton method implementations
    pub quasi_newton_methods: HashMap<String, Box<dyn QuasiNewtonMethod>>,
    /// Trust region method implementations
    pub trust_region_methods: HashMap<String, Box<dyn TrustRegionMethod>>,
    /// Line search method implementations
    pub line_search_methods: HashMap<String, Box<dyn LineSearchMethod>>,
    /// Conjugate gradient method implementations
    pub conjugate_gradient_methods: HashMap<String, Box<dyn ConjugateGradientMethod>>,
    /// Second-order method implementations
    pub second_order_methods: HashMap<String, Box<dyn SecondOrderMethod>>,
    /// Gradient estimation utilities
    pub gradient_estimator: GradientEstimator,
    /// Hessian estimation utilities
    pub hessian_estimator: HessianEstimator,
    /// Step size control mechanisms
    pub step_size_controller: StepSizeController,
}

/// Gradient descent algorithm trait
///
/// Defines interface for first-order optimization methods including
/// vanilla gradient descent, momentum methods, and adaptive methods.
pub trait GradientDescentAlgorithm: Send + Sync {
    /// Initialize algorithm with problem and starting point
    fn initialize(&mut self, problem: &OptimizationProblem, initial_point: &Array1<f64>) -> SklResult<()>;

    /// Compute gradient at given point
    fn compute_gradient(&self, point: &Array1<f64>) -> SklResult<Array1<f64>>;

    /// Compute step direction and size
    fn compute_step(&self, point: &Array1<f64>, gradient: &Array1<f64>) -> SklResult<Array1<f64>>;

    /// Update algorithm parameters based on progress
    fn update_parameters(&mut self, iteration: u64, improvement: f64) -> SklResult<()>;

    /// Get current algorithm state
    fn get_algorithm_state(&self) -> GradientDescentState;

    /// Check convergence criteria
    fn is_converged(&self, current_point: &Array1<f64>, gradient: &Array1<f64>) -> bool;
}

/// Gradient descent algorithm state
#[derive(Debug, Clone)]
pub struct GradientDescentState {
    /// Current optimization point
    pub current_point: Array1<f64>,
    /// Current gradient vector
    pub current_gradient: Array1<f64>,
    /// Current learning rate
    pub learning_rate: f64,
    /// Momentum term
    pub momentum: Array1<f64>,
    /// Current iteration number
    pub iteration: u64,
    /// History of objective values
    pub objective_history: Vec<f64>,
    /// History of gradient norms
    pub gradient_norm_history: Vec<f64>,
    /// History of step sizes
    pub step_size_history: Vec<f64>,
}

/// Quasi-Newton method trait
///
/// Defines interface for quasi-Newton methods that approximate
/// the Hessian matrix using gradient information.
pub trait QuasiNewtonMethod: Send + Sync {
    /// Initialize Hessian approximation
    fn initialize_hessian_approximation(&mut self, dimension: usize) -> SklResult<()>;

    /// Update Hessian approximation using secant condition
    fn update_hessian_approximation(&mut self, s: &Array1<f64>, y: &Array1<f64>) -> SklResult<()>;

    /// Compute search direction using Hessian approximation
    fn compute_search_direction(&self, gradient: &Array1<f64>) -> SklResult<Array1<f64>>;

    /// Get current Hessian approximation
    fn get_hessian_approximation(&self) -> Array2<f64>;

    /// Reset Hessian approximation to identity
    fn reset_hessian_approximation(&mut self) -> SklResult<()>;
}

/// Trust region method trait
///
/// Defines interface for trust region optimization methods that
/// solve quadratic subproblems within a trust region.
pub trait TrustRegionMethod: Send + Sync {
    /// Solve trust region subproblem
    fn solve_trust_region_subproblem(
        &self,
        gradient: &Array1<f64>,
        hessian: &Array2<f64>,
        radius: f64,
    ) -> SklResult<Array1<f64>>;

    /// Evaluate step quality for radius update
    fn evaluate_step_quality(&self, predicted_reduction: f64, actual_reduction: f64) -> f64;

    /// Update trust region radius based on step quality
    fn update_trust_region_radius(&self, step_quality: f64, current_radius: f64) -> f64;

    /// Get trust region parameters
    fn get_trust_region_parameters(&self) -> TrustRegionParameters;
}

/// Trust region parameters
#[derive(Debug, Clone)]
pub struct TrustRegionParameters {
    /// Initial trust region radius
    pub initial_radius: f64,
    /// Maximum trust region radius
    pub max_radius: f64,
    /// Factor for decreasing radius
    pub radius_decrease_factor: f64,
    /// Factor for increasing radius
    pub radius_increase_factor: f64,
    /// Threshold for accepting steps
    pub step_acceptance_threshold: f64,
    /// Lower threshold for radius update
    pub radius_update_threshold_low: f64,
    /// Upper threshold for radius update
    pub radius_update_threshold_high: f64,
}

/// Line search method trait
///
/// Defines interface for line search algorithms that find
/// appropriate step sizes satisfying Wolfe conditions.
pub trait LineSearchMethod: Send + Sync {
    /// Perform line search along given direction
    fn search(
        &self,
        f: &dyn Fn(&Array1<f64>) -> f64,
        point: &Array1<f64>,
        direction: &Array1<f64>,
    ) -> SklResult<f64>;

    /// Check if Wolfe conditions are satisfied
    fn wolfe_conditions_met(
        &self,
        step_size: f64,
        f_current: f64,
        f_new: f64,
        gradient_current: &Array1<f64>,
        gradient_new: &Array1<f64>,
        direction: &Array1<f64>,
    ) -> bool;

    /// Get line search parameters
    fn get_line_search_parameters(&self) -> LineSearchParameters;
}

/// Line search parameters
#[derive(Debug, Clone)]
pub struct LineSearchParameters {
    /// Armijo condition parameter (sufficient decrease)
    pub c1: f64,
    /// Curvature condition parameter
    pub c2: f64,
    /// Maximum line search iterations
    pub max_iterations: u32,
    /// Initial step size
    pub initial_step_size: f64,
    /// Step size reduction factor
    pub step_size_reduction: f64,
    /// Step size expansion factor
    pub step_size_expansion: f64,
}

/// Conjugate gradient method trait
///
/// Defines interface for conjugate gradient methods that
/// generate conjugate search directions.
pub trait ConjugateGradientMethod: Send + Sync {
    /// Compute beta parameter for conjugate direction
    fn compute_beta(
        &self,
        gradient_current: &Array1<f64>,
        gradient_previous: &Array1<f64>,
        direction_previous: &Array1<f64>,
    ) -> f64;

    /// Get conjugate gradient formula type
    fn get_cg_formula(&self) -> ConjugateGradientFormula;

    /// Check if restart is needed
    fn restart_criterion(&self, gradient: &Array1<f64>, direction: &Array1<f64>) -> bool;
}

/// Conjugate gradient formula variants
#[derive(Debug, Clone)]
pub enum ConjugateGradientFormula {
    /// Fletcher-Reeves formula
    FletcherReeves,
    /// Polak-Ribière formula
    PolakRibiere,
    /// Hestenes-Stiefel formula
    HestenesStiefel,
    /// Dai-Yuan formula
    DaiYuan,
    /// Hybrid formula
    Hybrid,
}

/// Second-order method trait
///
/// Defines interface for methods using second-order derivatives
/// including Newton's method and modifications.
pub trait SecondOrderMethod: Send + Sync {
    /// Compute Newton step direction
    fn compute_newton_step(&self, gradient: &Array1<f64>, hessian: &Array2<f64>) -> SklResult<Array1<f64>>;

    /// Regularize Hessian for numerical stability
    fn regularize_hessian(&self, hessian: &Array2<f64>) -> SklResult<Array2<f64>>;

    /// Check if matrix is positive definite
    fn check_positive_definiteness(&self, matrix: &Array2<f64>) -> bool;

    /// Apply modified Cholesky factorization
    fn modified_cholesky(&self, matrix: &Array2<f64>) -> SklResult<Array2<f64>>;
}

/// Gradient estimation utilities
#[derive(Debug, Default)]
pub struct GradientEstimator;

/// Hessian estimation utilities
#[derive(Debug, Default)]
pub struct HessianEstimator;

/// Step size control mechanisms
#[derive(Debug, Default)]
pub struct StepSizeController;

impl Default for GradientBasedOptimizer {
    fn default() -> Self {
        Self {
            optimizer_id: format!(
                "gradient_{}",
                SystemTime::now()
                    .duration_since(SystemTime::UNIX_EPOCH)
                    .unwrap()
                    .as_millis()
            ),
            gradient_descent_variants: HashMap::new(),
            quasi_newton_methods: HashMap::new(),
            trust_region_methods: HashMap::new(),
            line_search_methods: HashMap::new(),
            conjugate_gradient_methods: HashMap::new(),
            second_order_methods: HashMap::new(),
            gradient_estimator: GradientEstimator::default(),
            hessian_estimator: HessianEstimator::default(),
            step_size_controller: StepSizeController::default(),
        }
    }
}

impl GradientBasedOptimizer {
    /// Create a new gradient-based optimizer
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a gradient descent algorithm
    pub fn register_gradient_descent(&mut self, name: String, algorithm: Box<dyn GradientDescentAlgorithm>) {
        self.gradient_descent_variants.insert(name, algorithm);
    }

    /// Register a quasi-Newton method
    pub fn register_quasi_newton(&mut self, name: String, method: Box<dyn QuasiNewtonMethod>) {
        self.quasi_newton_methods.insert(name, method);
    }

    /// Register a trust region method
    pub fn register_trust_region(&mut self, name: String, method: Box<dyn TrustRegionMethod>) {
        self.trust_region_methods.insert(name, method);
    }

    /// Register a line search method
    pub fn register_line_search(&mut self, name: String, method: Box<dyn LineSearchMethod>) {
        self.line_search_methods.insert(name, method);
    }

    /// Register a conjugate gradient method
    pub fn register_conjugate_gradient(&mut self, name: String, method: Box<dyn ConjugateGradientMethod>) {
        self.conjugate_gradient_methods.insert(name, method);
    }

    /// Register a second-order method
    pub fn register_second_order(&mut self, name: String, method: Box<dyn SecondOrderMethod>) {
        self.second_order_methods.insert(name, method);
    }

    /// Get available methods by category
    pub fn get_available_methods(&self) -> HashMap<String, Vec<String>> {
        let mut methods = HashMap::new();
        methods.insert("gradient_descent".to_string(), self.gradient_descent_variants.keys().cloned().collect());
        methods.insert("quasi_newton".to_string(), self.quasi_newton_methods.keys().cloned().collect());
        methods.insert("trust_region".to_string(), self.trust_region_methods.keys().cloned().collect());
        methods.insert("line_search".to_string(), self.line_search_methods.keys().cloned().collect());
        methods.insert("conjugate_gradient".to_string(), self.conjugate_gradient_methods.keys().cloned().collect());
        methods.insert("second_order".to_string(), self.second_order_methods.keys().cloned().collect());
        methods
    }
}

impl Default for TrustRegionParameters {
    fn default() -> Self {
        Self {
            initial_radius: 1.0,
            max_radius: 100.0,
            radius_decrease_factor: 0.25,
            radius_increase_factor: 2.0,
            step_acceptance_threshold: 0.1,
            radius_update_threshold_low: 0.25,
            radius_update_threshold_high: 0.75,
        }
    }
}

impl Default for LineSearchParameters {
    fn default() -> Self {
        Self {
            c1: 1e-4,      // Armijo parameter
            c2: 0.9,       // Curvature parameter
            max_iterations: 20,
            initial_step_size: 1.0,
            step_size_reduction: 0.5,
            step_size_expansion: 2.0,
        }
    }
}

impl GradientEstimator {
    /// Create a new gradient estimator
    pub fn new() -> Self {
        Self::default()
    }

    /// Estimate gradient using finite differences
    pub fn finite_difference(
        &self,
        f: &dyn Fn(&Array1<f64>) -> f64,
        point: &Array1<f64>,
        epsilon: f64,
    ) -> SklResult<Array1<f64>> {
        let n = point.len();
        let mut gradient = Array1::zeros(n);

        for i in 0..n {
            let mut point_plus = point.clone();
            let mut point_minus = point.clone();

            point_plus[i] += epsilon;
            point_minus[i] -= epsilon;

            let f_plus = f(&point_plus);
            let f_minus = f(&point_minus);

            gradient[i] = (f_plus - f_minus) / (2.0 * epsilon);
        }

        Ok(gradient)
    }

    /// Estimate gradient using complex step method
    pub fn complex_step(
        &self,
        f: &dyn Fn(&Array1<f64>) -> f64,
        point: &Array1<f64>,
        step_size: f64,
    ) -> SklResult<Array1<f64>> {
        // Simplified implementation - would use complex arithmetic in practice
        self.finite_difference(f, point, step_size)
    }
}

impl HessianEstimator {
    pub fn new() -> Self {
        Self::default()
    }

    /// Estimate Hessian using finite differences
    pub fn finite_difference(
        &self,
        gradient_func: &dyn Fn(&Array1<f64>) -> SklResult<Array1<f64>>,
        point: &Array1<f64>,
        epsilon: f64,
    ) -> SklResult<Array2<f64>> {
        let n = point.len();
        let mut hessian = Array2::zeros((n, n));

        for i in 0..n {
            let mut point_plus = point.clone();
            let mut point_minus = point.clone();

            point_plus[i] += epsilon;
            point_minus[i] -= epsilon;

            let grad_plus = gradient_func(&point_plus)?;
            let grad_minus = gradient_func(&point_minus)?;

            for j in 0..n {
                hessian[[i, j]] = (grad_plus[j] - grad_minus[j]) / (2.0 * epsilon);
            }
        }

        Ok(hessian)
    }

    /// Estimate Hessian using BFGS approximation
    pub fn bfgs_approximation(
        &self,
        current_hessian: &Array2<f64>,
        s: &Array1<f64>,  // step vector
        y: &Array1<f64>,  // gradient difference
    ) -> SklResult<Array2<f64>> {
        let n = current_hessian.nrows();
        let mut h_new = current_hessian.clone();

        let ys = s.dot(y);
        if ys.abs() < 1e-12 {
            return Ok(h_new); // Skip update if denominator too small
        }

        let hy = current_hessian.dot(y);
        let yhy = y.dot(&hy);

        // BFGS update: H_{k+1} = H_k - (H_k y y^T H_k)/(y^T H_k y) + (s s^T)/(s^T y)
        for i in 0..n {
            for j in 0..n {
                h_new[[i, j]] = h_new[[i, j]]
                    - (hy[i] * hy[j]) / yhy
                    + (s[i] * s[j]) / ys;
            }
        }

        Ok(h_new)
    }
}

impl StepSizeController {
    /// Create a new step size controller
    pub fn new() -> Self {
        Self::default()
    }

    /// Adapt step size based on gradient norm
    pub fn adaptive_step_size(
        &self,
        current_step: f64,
        gradient_norm: f64,
        previous_gradient_norm: f64,
        improvement: f64,
    ) -> f64 {
        let ratio = gradient_norm / previous_gradient_norm;

        if improvement > 0.0 && ratio < 0.9 {
            // Good progress, increase step size
            (current_step * 1.1).min(10.0)
        } else if improvement < 0.0 || ratio > 1.1 {
            // Poor progress, decrease step size
            (current_step * 0.5).max(1e-8)
        } else {
            // Maintain current step size
            current_step
        }
    }

    /// Backtracking line search for step size
    pub fn backtracking_line_search(
        &self,
        f: &dyn Fn(&Array1<f64>) -> f64,
        point: &Array1<f64>,
        direction: &Array1<f64>,
        gradient: &Array1<f64>,
        alpha: f64,
        rho: f64,
        c: f64,
    ) -> f64 {
        let mut step_size = alpha;
        let f_current = f(point);
        let directional_derivative = gradient.dot(direction);

        loop {
            let new_point = point + &(direction * step_size);
            let f_new = f(&new_point);

            // Armijo condition
            if f_new <= f_current + c * step_size * directional_derivative {
                break;
            }

            step_size *= rho;

            if step_size < 1e-12 {
                break;
            }
        }

        step_size
    }
}