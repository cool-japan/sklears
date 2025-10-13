//! Core Gradient-Based Optimization Algorithms
//!
//! This module provides the fundamental optimization algorithms and core trait definitions
//! for gradient-based optimization. It includes the main `GradientBasedOptimizer` coordinator
//! and essential algorithm traits for different optimization methods.
//!
//! # Features
//!
//! - **GradientBasedOptimizer**: Central coordinator managing optimization algorithms
//! - **Core Algorithm Traits**: Unified interfaces for gradient descent, quasi-Newton, trust region, etc.
//! - **Algorithm State Management**: Comprehensive state tracking and history
//! - **Type Safety**: Strongly-typed optimization states and parameters
//! - **SciRS2 Compliance**: Full integration with SciRS2 ecosystem

use std::collections::HashMap;
use std::time::{Duration, SystemTime};
use std::sync::{Arc, Mutex, RwLock, atomic::{AtomicU64, AtomicBool, Ordering}};
use std::fmt;

// SciRS2 Core Dependencies for Sklears Compliance
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2, array};
use scirs2_core::ndarray_ext::{stats, manipulation, matrix};
use scirs2_core::random::{Random, rng, DistributionExt};

// SIMD and Performance Optimization (with conditional compilation)
#[cfg(feature = "simd_ops")]
use scirs2_core::simd_ops::{simd_dot_product, simd_matrix_multiply};

#[cfg(feature = "parallel_ops")]
use scirs2_core::parallel_ops::{par_chunks, par_join, par_scope};

// Use standard Rust Result type
type SklResult<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;

// Forward declarations for types defined in other modules
use super::gradient_computation::{GradientEstimator, HessianEstimator};
use super::algorithm_selection::{AlgorithmSelector, StepSizeController, ConvergenceAnalyzer};
use super::performance_monitoring::{GradientPerformanceMonitor, GradientPerformanceMetrics};
use super::simd_acceleration::SimdConfiguration;

// Basic compatibility types
#[derive(Debug, Clone)]
pub struct OptimizationProblem {
    pub problem_id: String,
    pub dimension: usize,
    pub objectives: Vec<String>,
    pub constraints: Vec<String>,
    pub variable_bounds: Option<(Array1<f64>, Array1<f64>)>,
}

#[derive(Debug, Clone)]
pub struct Solution {
    pub variables: Array1<f64>,
    pub objective_value: f64,
    pub constraint_violations: Vec<f64>,
    pub metadata: HashMap<String, String>,
}

/// Central gradient-based optimization engine coordinating various gradient methods
///
/// The `GradientBasedOptimizer` serves as the main entry point for gradient-based
/// optimization algorithms. It maintains collections of different algorithm types
/// and provides intelligent algorithm selection based on problem characteristics.
#[derive(Debug)]
pub struct GradientBasedOptimizer {
    /// Unique identifier for this optimizer instance
    optimizer_id: String,

    /// Collection of gradient descent algorithm implementations
    gradient_descent_variants: HashMap<String, Box<dyn GradientDescentAlgorithm>>,

    /// Collection of quasi-Newton method implementations
    quasi_newton_methods: HashMap<String, Box<dyn QuasiNewtonMethod>>,

    /// Collection of trust region method implementations
    trust_region_methods: HashMap<String, Box<dyn TrustRegionMethod>>,

    /// Collection of line search algorithm implementations
    line_search_methods: HashMap<String, Box<dyn LineSearchMethod>>,

    /// Collection of conjugate gradient method implementations
    conjugate_gradient_methods: HashMap<String, Box<dyn ConjugateGradientMethod>>,

    /// Collection of second-order optimization methods
    second_order_methods: HashMap<String, Box<dyn SecondOrderMethod>>,

    /// Gradient computation and estimation system
    gradient_estimator: GradientEstimator,

    /// Hessian approximation and computation system
    hessian_estimator: HessianEstimator,

    /// Adaptive step size control system
    step_size_controller: StepSizeController,

    /// Algorithm selection and configuration system
    algorithm_selector: AlgorithmSelector,

    /// Convergence analysis and detection system
    convergence_analyzer: ConvergenceAnalyzer,

    /// Performance monitoring and metrics collection
    performance_monitor: GradientPerformanceMonitor,

    /// Current optimization state and history
    optimization_state: Arc<RwLock<OptimizationState>>,

    /// SIMD acceleration configuration
    simd_config: SimdConfiguration,

    /// Thread-safe counters for performance tracking
    total_gradient_evaluations: Arc<AtomicU64>,
    total_hessian_evaluations: Arc<AtomicU64>,
    total_function_evaluations: Arc<AtomicU64>,

    /// Active optimization flag for safe shutdown
    is_optimizing: Arc<AtomicBool>,
}

/// Current state of optimization process
#[derive(Debug, Clone)]
pub struct OptimizationState {
    /// Current best solution found
    pub current_solution: Option<Solution>,

    /// Current objective function value
    pub current_objective: f64,

    /// Current gradient vector
    pub current_gradient: Array1<f64>,

    /// Current iteration number
    pub iteration: u64,

    /// Optimization start time
    pub start_time: SystemTime,

    /// Algorithm-specific state information
    pub algorithm_state: GradientDescentState,

    /// Convergence status
    pub convergence_status: ConvergenceStatus,

    /// Current performance metrics
    pub performance_metrics: GradientPerformanceMetrics,
}

/// Convergence status enumeration
#[derive(Debug, Clone, PartialEq)]
pub enum ConvergenceStatus {
    /// Optimization in progress
    InProgress,
    /// Converged to optimal solution
    Converged,
    /// Converged to local minimum
    LocalConvergence,
    /// Stagnated - no further progress
    Stagnated,
    /// Failed to converge within limits
    Failed,
    /// Stopped by user request
    UserStopped,
    /// Stopped due to resource limits
    ResourceLimitReached,
}

/// Trait defining the interface for gradient descent algorithm implementations
///
/// This trait provides a unified interface for various gradient descent methods
/// including classical gradient descent, momentum variants, and adaptive methods.
pub trait GradientDescentAlgorithm: Send + Sync {
    /// Initialize the algorithm with problem setup and starting point
    fn initialize(&mut self, problem: &OptimizationProblem, initial_point: &Array1<f64>) -> SklResult<()>;

    /// Compute gradient at the given point using problem-specific methods
    fn compute_gradient(&self, point: &Array1<f64>) -> SklResult<Array1<f64>>;

    /// Compute optimization step from current point and gradient
    fn compute_step(&self, point: &Array1<f64>, gradient: &Array1<f64>) -> SklResult<Array1<f64>>;

    /// Update algorithm parameters based on iteration progress
    fn update_parameters(&mut self, iteration: u64, improvement: f64) -> SklResult<()>;

    /// Get current algorithm state for monitoring and analysis
    fn get_algorithm_state(&self) -> GradientDescentState;

    /// Check if optimization has converged based on gradient norms
    fn is_converged(&self, current_point: &Array1<f64>, gradient: &Array1<f64>) -> bool;

    /// Perform SIMD-accelerated gradient computation when available
    fn compute_gradient_simd(&self, point: &Array1<f64>) -> SklResult<Array1<f64>> {
        // Default implementation falls back to standard gradient computation
        self.compute_gradient(point)
    }

    /// Compute momentum-adjusted step for algorithms supporting momentum
    fn compute_momentum_step(&self, gradient: &Array1<f64>, previous_step: &Array1<f64>, momentum: f64) -> SklResult<Array1<f64>> {
        // Default momentum computation: momentum * previous_step - learning_rate * gradient
        let momentum_term = previous_step * momentum;
        let gradient_term = gradient * 0.01; // Default learning rate
        Ok(&momentum_term - &gradient_term)
    }
}

/// State information for gradient descent algorithms
///
/// This structure maintains the complete state of gradient descent optimization
/// including current position, gradients, learning parameters, and history.
#[derive(Debug, Clone)]
pub struct GradientDescentState {
    /// Current optimization point in parameter space
    pub current_point: Array1<f64>,

    /// Current gradient vector at the optimization point
    pub current_gradient: Array1<f64>,

    /// Current learning rate or step size
    pub learning_rate: f64,

    /// Momentum vector for momentum-based methods
    pub momentum: Array1<f64>,

    /// Current iteration number
    pub iteration: u64,

    /// History of objective function values
    pub objective_history: Vec<f64>,

    /// History of gradient norms for convergence analysis
    pub gradient_norm_history: Vec<f64>,

    /// History of step sizes for adaptive methods
    pub step_size_history: Vec<f64>,

    /// Adaptive parameter state (for Adam, RMSprop, etc.)
    pub adaptive_state: AdaptiveGradientState,

    /// Performance metrics for this algorithm run
    pub performance_metrics: GradientPerformanceMetrics,
}

/// State for adaptive gradient methods like Adam, RMSprop, AdaGrad
#[derive(Debug, Clone)]
pub struct AdaptiveGradientState {
    /// First moment estimate (Adam, AdaMax)
    pub first_moment: Array1<f64>,

    /// Second moment estimate (Adam, RMSprop)
    pub second_moment: Array1<f64>,

    /// Accumulated squared gradients (AdaGrad)
    pub accumulated_gradients: Array1<f64>,

    /// Bias correction terms
    pub bias_correction_1: f64,
    pub bias_correction_2: f64,

    /// Algorithm-specific parameters
    pub beta1: f64,
    pub beta2: f64,
    pub epsilon: f64,
}

/// Trait for quasi-Newton optimization methods
///
/// Quasi-Newton methods approximate the Hessian matrix using gradient information
/// from previous iterations, providing superlinear convergence without computing
/// the full Hessian at each iteration.
pub trait QuasiNewtonMethod: Send + Sync {
    /// Initialize Hessian approximation (typically identity matrix)
    fn initialize_hessian_approximation(&mut self, dimension: usize) -> SklResult<()>;

    /// Update Hessian approximation using secant condition
    fn update_hessian_approximation(&mut self, s: &Array1<f64>, y: &Array1<f64>) -> SklResult<()>;

    /// Compute search direction using current Hessian approximation
    fn compute_search_direction(&self, gradient: &Array1<f64>) -> SklResult<Array1<f64>>;

    /// Get current Hessian approximation matrix
    fn get_hessian_approximation(&self) -> Array2<f64>;

    /// Reset Hessian approximation to initial state
    fn reset_hessian_approximation(&mut self) -> SklResult<()>;

    /// Check if Hessian approximation maintains positive definiteness
    fn is_positive_definite(&self) -> bool;

    /// Get algorithm-specific parameters (BFGS, L-BFGS, etc.)
    fn get_method_parameters(&self) -> QuasiNewtonParameters;

    /// Perform limited-memory update for L-BFGS style methods
    fn limited_memory_update(&mut self, s_history: &[Array1<f64>], y_history: &[Array1<f64>]) -> SklResult<()> {
        // Default implementation for full-memory methods
        if let (Some(s), Some(y)) = (s_history.last(), y_history.last()) {
            self.update_hessian_approximation(s, y)
        } else {
            Ok(())
        }
    }
}

/// Parameters for quasi-Newton methods
#[derive(Debug, Clone)]
pub struct QuasiNewtonParameters {
    /// Method type (BFGS, L-BFGS, DFP, SR1)
    pub method_type: QuasiNewtonMethodType,

    /// Memory limit for limited-memory methods
    pub memory_limit: Option<usize>,

    /// Damping parameter for stability
    pub damping_parameter: f64,

    /// Tolerance for curvature condition
    pub curvature_tolerance: f64,

    /// Maximum condition number for Hessian approximation
    pub max_condition_number: f64,
}

/// Types of quasi-Newton methods
#[derive(Debug, Clone, PartialEq)]
pub enum QuasiNewtonMethodType {
    /// Broyden-Fletcher-Goldfarb-Shanno method
    BFGS,
    /// Limited-memory BFGS
    LBFGS,
    /// Davidon-Fletcher-Powell method
    DFP,
    /// Symmetric Rank-1 updates
    SR1,
    /// Broyden family methods
    Broyden(f64),
}

/// Trait for trust region optimization methods
///
/// Trust region methods maintain a region around the current iterate where
/// a model is trusted to be accurate. The step is computed by solving
/// a subproblem within this region.
pub trait TrustRegionMethod: Send + Sync {
    /// Solve trust region subproblem to find step within trust region
    fn solve_trust_region_subproblem(&self, gradient: &Array1<f64>, hessian: &Array2<f64>, radius: f64) -> SklResult<Array1<f64>>;

    /// Evaluate quality of step by comparing actual vs predicted reduction
    fn evaluate_step_quality(&self, predicted_reduction: f64, actual_reduction: f64) -> f64;

    /// Update trust region radius based on step quality
    fn update_trust_region_radius(&self, step_quality: f64, current_radius: f64) -> f64;

    /// Get trust region method parameters
    fn get_trust_region_parameters(&self) -> TrustRegionParameters;

    /// Compute Cauchy point for trust region methods
    fn compute_cauchy_point(&self, gradient: &Array1<f64>, hessian: &Array2<f64>, radius: f64) -> SklResult<Array1<f64>>;

    /// Solve using Dogleg method
    fn solve_dogleg(&self, gradient: &Array1<f64>, hessian: &Array2<f64>, radius: f64) -> SklResult<Array1<f64>>;

    /// Solve using Steihaug-CG method for large problems
    fn solve_steihaug_cg(&self, gradient: &Array1<f64>, hessian: &Array2<f64>, radius: f64, max_iterations: usize) -> SklResult<Array1<f64>>;
}

/// Parameters for trust region methods
#[derive(Debug, Clone)]
pub struct TrustRegionParameters {
    /// Initial trust region radius
    pub initial_radius: f64,

    /// Maximum trust region radius
    pub max_radius: f64,

    /// Factor for decreasing radius on poor steps
    pub radius_decrease_factor: f64,

    /// Factor for increasing radius on good steps
    pub radius_increase_factor: f64,

    /// Threshold for accepting a step
    pub step_acceptance_threshold: f64,

    /// Lower threshold for radius update decisions
    pub radius_update_threshold_low: f64,

    /// Upper threshold for radius update decisions
    pub radius_update_threshold_high: f64,

    /// Trust region method variant
    pub method_variant: TrustRegionMethodVariant,

    /// Maximum iterations for CG subproblem solver
    pub max_cg_iterations: usize,

    /// Tolerance for CG subproblem convergence
    pub cg_tolerance: f64,
}

/// Variants of trust region methods
#[derive(Debug, Clone, PartialEq)]
pub enum TrustRegionMethodVariant {
    /// Dogleg method
    Dogleg,
    /// Two-dimensional subspace minimization
    TwoDimensional,
    /// Steihaug conjugate gradient method
    SteihaugCG,
    /// More-Sorensen method
    MoreSorensen,
}

/// Trait for line search methods
///
/// Line search methods find an appropriate step size along a given direction
/// by satisfying various conditions such as Armijo rule or Wolfe conditions.
pub trait LineSearchMethod: Send + Sync {
    /// Perform line search to find appropriate step size
    fn search(&self, f: &dyn Fn(&Array1<f64>) -> f64, point: &Array1<f64>, direction: &Array1<f64>) -> SklResult<f64>;

    /// Check if Wolfe conditions are satisfied
    fn wolfe_conditions_met(&self, step_size: f64, f_current: f64, f_new: f64, gradient_current: &Array1<f64>, gradient_new: &Array1<f64>, direction: &Array1<f64>) -> bool;

    /// Get line search parameters
    fn get_line_search_parameters(&self) -> LineSearchParameters;

    /// Perform backtracking line search with Armijo condition
    fn backtracking_search(&self, f: &dyn Fn(&Array1<f64>) -> f64, point: &Array1<f64>, direction: &Array1<f64>, initial_step: f64) -> SklResult<f64>;

    /// Zoom phase for strong Wolfe line search
    fn zoom_phase(&self, f: &dyn Fn(&Array1<f64>) -> f64, point: &Array1<f64>, direction: &Array1<f64>,
                  alpha_low: f64, alpha_high: f64, f_low: f64, f_high: f64) -> SklResult<f64>;

    /// Cubic interpolation for step size refinement
    fn cubic_interpolation(&self, alpha1: f64, f1: f64, df1: f64, alpha2: f64, f2: f64, df2: f64) -> f64;

    /// Quadratic interpolation for step size refinement
    fn quadratic_interpolation(&self, alpha1: f64, f1: f64, df1: f64, alpha2: f64, f2: f64) -> f64;
}

/// Parameters for line search algorithms
#[derive(Debug, Clone)]
pub struct LineSearchParameters {
    /// Armijo condition parameter (sufficient decrease)
    pub c1: f64,

    /// Curvature condition parameter (Wolfe conditions)
    pub c2: f64,

    /// Maximum number of line search iterations
    pub max_iterations: u32,

    /// Initial step size
    pub initial_step_size: f64,

    /// Step size reduction factor for backtracking
    pub step_size_reduction: f64,

    /// Step size expansion factor for aggressive search
    pub step_size_expansion: f64,

    /// Line search method variant
    pub method_variant: LineSearchVariant,

    /// Tolerance for convergence
    pub tolerance: f64,

    /// Safety bounds for step size
    pub min_step_size: f64,
    pub max_step_size: f64,
}

/// Variants of line search methods
#[derive(Debug, Clone, PartialEq)]
pub enum LineSearchVariant {
    /// Simple backtracking with Armijo condition
    Backtracking,
    /// Weak Wolfe conditions
    WeakWolfe,
    /// Strong Wolfe conditions
    StrongWolfe,
    /// Goldstein conditions
    Goldstein,
    /// More-Thuente line search
    MoreThuente,
}

/// Trait for conjugate gradient methods
///
/// Conjugate gradient methods generate search directions that are conjugate
/// with respect to the Hessian, providing excellent convergence for quadratic
/// functions and good performance for general nonlinear optimization.
pub trait ConjugateGradientMethod: Send + Sync {
    /// Compute beta parameter for conjugate direction
    fn compute_beta(&self, gradient_current: &Array1<f64>, gradient_previous: &Array1<f64>, direction_previous: &Array1<f64>) -> f64;

    /// Get conjugate gradient formula being used
    fn get_cg_formula(&self) -> ConjugateGradientFormula;

    /// Check if restart criterion is met
    fn restart_criterion(&self, gradient: &Array1<f64>, direction: &Array1<f64>) -> bool;

    /// Compute conjugate direction
    fn compute_conjugate_direction(&self, gradient: &Array1<f64>, previous_direction: &Array1<f64>, beta: f64) -> Array1<f64>;

    /// Preconditioning for conjugate gradient
    fn apply_preconditioning(&self, gradient: &Array1<f64>, preconditioner: Option<&Array2<f64>>) -> SklResult<Array1<f64>>;
}

/// Conjugate gradient formulas for computing beta parameter
#[derive(Debug, Clone, PartialEq)]
pub enum ConjugateGradientFormula {
    /// Fletcher-Reeves formula
    FletcherReeves,
    /// Polak-Ribière formula
    PolakRibiere,
    /// Hestenes-Stiefel formula
    HestenesStiefel,
    /// Dai-Yuan formula
    DaiYuan,
    /// Hager-Zhang formula
    HagerZhang,
    /// Custom formula with identifier
    Custom(String),
}

/// Trait for second-order optimization methods
///
/// Second-order methods use both gradient and Hessian information to achieve
/// quadratic convergence near the optimum, including Newton's method and
/// its variants.
pub trait SecondOrderMethod: Send + Sync {
    fn compute_newton_step(&self, gradient: &Array1<f64>, hessian: &Array2<f64>) -> SklResult<Array1<f64>>;

    fn regularize_hessian(&self, hessian: &Array2<f64>) -> SklResult<Array2<f64>>;

    fn check_positive_definiteness(&self, matrix: &Array2<f64>) -> bool;

    fn modified_cholesky(&self, matrix: &Array2<f64>) -> SklResult<Array2<f64>>;

    fn gauss_newton_step(&self, residuals: &Array1<f64>, jacobian: &Array2<f64>) -> SklResult<Array1<f64>>;

    fn levenberg_marquardt_step(&self, residuals: &Array1<f64>, jacobian: &Array2<f64>, damping: f64) -> SklResult<Array1<f64>>;

    fn natural_gradient_step(&self, gradient: &Array1<f64>, metric_tensor: &Array2<f64>) -> SklResult<Array1<f64>>;
}

// Default implementations

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
            method_variant: TrustRegionMethodVariant::Dogleg,
            max_cg_iterations: 100,
            cg_tolerance: 1e-8,
        }
    }
}

impl Default for LineSearchParameters {
    fn default() -> Self {
        Self {
            c1: 1e-4,
            c2: 0.9,
            max_iterations: 50,
            initial_step_size: 1.0,
            step_size_reduction: 0.5,
            step_size_expansion: 2.0,
            method_variant: LineSearchVariant::StrongWolfe,
            tolerance: 1e-8,
            min_step_size: 1e-16,
            max_step_size: 1e16,
        }
    }
}

impl Default for AdaptiveGradientState {
    fn default() -> Self {
        Self {
            first_moment: Array1::zeros(1),
            second_moment: Array1::zeros(1),
            accumulated_gradients: Array1::zeros(1),
            bias_correction_1: 1.0,
            bias_correction_2: 1.0,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
        }
    }
}

impl Default for GradientDescentState {
    fn default() -> Self {
        Self {
            current_point: Array1::zeros(1),
            current_gradient: Array1::zeros(1),
            learning_rate: 1.0,
            momentum: Array1::zeros(1),
            iteration: 0,
            objective_history: Vec::with_capacity(1000),
            gradient_norm_history: Vec::with_capacity(1000),
            step_size_history: Vec::with_capacity(1000),
            adaptive_state: AdaptiveGradientState::default(),
            performance_metrics: GradientPerformanceMetrics::default(),
        }
    }
}

impl Default for OptimizationState {
    fn default() -> Self {
        Self {
            current_solution: None,
            current_objective: f64::INFINITY,
            current_gradient: Array1::zeros(1),
            iteration: 0,
            start_time: SystemTime::now(),
            algorithm_state: GradientDescentState::default(),
            convergence_status: ConvergenceStatus::InProgress,
            performance_metrics: GradientPerformanceMetrics::default(),
        }
    }
}

impl Default for GradientBasedOptimizer {
    fn default() -> Self {
        Self {
            optimizer_id: "gradient_optimizer_default".to_string(),
            gradient_descent_variants: HashMap::new(),
            quasi_newton_methods: HashMap::new(),
            trust_region_methods: HashMap::new(),
            line_search_methods: HashMap::new(),
            conjugate_gradient_methods: HashMap::new(),
            second_order_methods: HashMap::new(),
            gradient_estimator: GradientEstimator::default(),
            hessian_estimator: HessianEstimator::default(),
            step_size_controller: StepSizeController::default(),
            algorithm_selector: AlgorithmSelector::default(),
            convergence_analyzer: ConvergenceAnalyzer::default(),
            performance_monitor: GradientPerformanceMonitor::default(),
            optimization_state: Arc::new(RwLock::new(OptimizationState::default())),
            simd_config: SimdConfiguration::default(),
            total_gradient_evaluations: Arc::new(AtomicU64::new(0)),
            total_hessian_evaluations: Arc::new(AtomicU64::new(0)),
            total_function_evaluations: Arc::new(AtomicU64::new(0)),
            is_optimizing: Arc::new(AtomicBool::new(false)),
        }
    }
}

// Core algorithm implementations
impl GradientBasedOptimizer {
    /// Create a new gradient-based optimizer with default configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a new optimizer with custom identifier
    pub fn with_id(id: String) -> Self {
        let mut optimizer = Self::default();
        optimizer.optimizer_id = id;
        optimizer
    }

    /// Register a gradient descent algorithm variant
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

    /// Get current optimization state
    pub fn get_state(&self) -> Arc<RwLock<OptimizationState>> {
        self.optimization_state.clone()
    }

    /// Check if optimization is currently running
    pub fn is_optimizing(&self) -> bool {
        self.is_optimizing.load(Ordering::Relaxed)
    }

    /// Get performance counters
    pub fn get_performance_counters(&self) -> (u64, u64, u64) {
        (
            self.total_gradient_evaluations.load(Ordering::Relaxed),
            self.total_hessian_evaluations.load(Ordering::Relaxed),
            self.total_function_evaluations.load(Ordering::Relaxed),
        )
    }

    /// Reset performance counters
    pub fn reset_counters(&self) {
        self.total_gradient_evaluations.store(0, Ordering::Relaxed);
        self.total_hessian_evaluations.store(0, Ordering::Relaxed);
        self.total_function_evaluations.store(0, Ordering::Relaxed);
    }
}