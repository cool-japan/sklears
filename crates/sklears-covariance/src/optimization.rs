//! Extensible Optimization Algorithms Framework
//!
//! This module provides a pluggable optimization framework for covariance estimation
//! methods. It supports various optimization algorithms that can be easily swapped
//! and configured for different estimation tasks.
//!
//! # Key Components
//!
//! - **OptimizationAlgorithm**: Core trait for optimization algorithms
//! - **ObjectiveFunction**: Trait for defining optimization objectives
//! - **OptimizerRegistry**: Registry for managing available optimizers
//! - **OptimizationConfig**: Configuration for optimization parameters
//! - **OptimizationResult**: Result container with convergence information
//!
//! # Supported Algorithms
//!
//! - Gradient descent variants (SGD, Adam, AdaGrad, RMSprop)
//! - Coordinate descent methods
//! - Proximal gradient methods
//! - Quasi-Newton methods (BFGS, L-BFGS)
//! - Trust region methods
//! - Evolutionary algorithms
//! - Bayesian optimization

use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::error::{Result, SklearsError};
use std::collections::HashMap;
use std::fmt::Debug;

/// Core trait for optimization algorithms
pub trait OptimizationAlgorithm: Debug + Send + Sync {
    /// The type of parameters being optimized
    type Parameters: Clone + Debug;

    /// Minimize the objective function
    fn minimize<F>(
        &self,
        objective: &F,
        initial_params: Self::Parameters,
    ) -> Result<OptimizationResult<Self::Parameters>>
    where
        F: ObjectiveFunction<Parameters = Self::Parameters>;

    /// Get algorithm name
    fn name(&self) -> &'static str;

    /// Get algorithm configuration
    fn config(&self) -> &OptimizationConfig;
}

/// Trait for objective functions
pub trait ObjectiveFunction: Debug + Send + Sync {
    /// The type of parameters
    type Parameters: Clone + Debug;

    /// Evaluate the objective function
    fn evaluate(&self, params: &Self::Parameters) -> Result<f64>;

    /// Compute gradient if available
    fn gradient(&self, params: &Self::Parameters) -> Result<Option<Self::Parameters>> {
        let _ = params;
        Ok(None)
    }

    /// Compute Hessian if available
    fn hessian(&self, params: &Self::Parameters) -> Result<Option<Array2<f64>>> {
        let _ = params;
        Ok(None)
    }

    /// Check if function supports gradients
    fn has_gradient(&self) -> bool {
        false
    }

    /// Check if function supports Hessian
    fn has_hessian(&self) -> bool {
        false
    }
}

/// Optimization algorithm types
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum OptimizerType {
    /// Stochastic Gradient Descent
    SGD,
    /// Adam optimizer
    Adam,
    /// AdaGrad optimizer
    AdaGrad,
    /// RMSprop optimizer
    RMSprop,
    /// Coordinate descent
    CoordinateDescent,
    /// Proximal gradient descent
    ProximalGradient,
    /// BFGS quasi-Newton method
    BFGS,
    /// Limited-memory BFGS
    LBFGS,
    /// Trust region method
    TrustRegion,
    /// Genetic algorithm
    GeneticAlgorithm,
    /// Particle swarm optimization
    ParticleSwarm,
    /// Bayesian optimization
    BayesianOptimization,
    /// Nelder-Mead simplex
    NelderMead,
    /// Custom algorithm
    Custom(String),
}

/// Optimization configuration
#[derive(Debug, Clone)]
pub struct OptimizationConfig {
    /// Maximum number of iterations
    pub max_iterations: usize,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Learning rate (for gradient-based methods)
    pub learning_rate: f64,
    /// Momentum parameter (for applicable methods)
    pub momentum: f64,
    /// L1 regularization parameter
    pub l1_regularization: f64,
    /// L2 regularization parameter
    pub l2_regularization: f64,
    /// Whether to use adaptive learning rate
    pub adaptive_learning_rate: bool,
    /// Line search method
    pub line_search: LineSearchMethod,
    /// Verbosity level
    pub verbose: bool,
    /// Random seed for stochastic methods
    pub seed: Option<u64>,
    /// Algorithm-specific parameters
    pub algorithm_params: HashMap<String, f64>,
}

/// Line search methods
#[derive(Debug, Clone)]
pub enum LineSearchMethod {
    None,
    Armijo,
    Wolfe,
    StrongWolfe,
    Backtracking,
}

/// Optimization result
#[derive(Debug, Clone)]
pub struct OptimizationResult<T> {
    /// Final parameters
    pub parameters: T,
    /// Final objective value
    pub objective_value: f64,
    /// Number of iterations taken
    pub iterations: usize,
    /// Number of function evaluations
    pub function_evaluations: usize,
    /// Number of gradient evaluations
    pub gradient_evaluations: usize,
    /// Whether optimization converged
    pub converged: bool,
    /// Convergence message
    pub message: String,
    /// Optimization history
    pub history: OptimizationHistory,
}

/// Optimization history tracking
#[derive(Debug, Clone)]
pub struct OptimizationHistory {
    /// Objective values at each iteration
    pub objective_values: Vec<f64>,
    /// Gradient norms at each iteration
    pub gradient_norms: Vec<f64>,
    /// Step sizes at each iteration
    pub step_sizes: Vec<f64>,
    /// Iteration times
    pub iteration_times: Vec<f64>,
}

/// Concrete optimizer wrapper to avoid trait object issues
#[derive(Debug)]
pub enum OptimizerWrapper {
    SGD(SGDOptimizer),
    Adam(AdamOptimizer),
    CoordinateDescent(CoordinateDescentOptimizer),
    ProximalGradient(ProximalGradientOptimizer),
    NelderMead(NelderMeadOptimizer),
}

impl OptimizationAlgorithm for OptimizerWrapper {
    type Parameters = Array1<f64>;

    fn minimize<F>(
        &self,
        objective: &F,
        initial_params: Self::Parameters,
    ) -> Result<OptimizationResult<Self::Parameters>>
    where
        F: ObjectiveFunction<Parameters = Self::Parameters>,
    {
        match self {
            OptimizerWrapper::SGD(opt) => opt.minimize(objective, initial_params),
            OptimizerWrapper::Adam(opt) => opt.minimize(objective, initial_params),
            OptimizerWrapper::CoordinateDescent(opt) => opt.minimize(objective, initial_params),
            OptimizerWrapper::ProximalGradient(opt) => opt.minimize(objective, initial_params),
            OptimizerWrapper::NelderMead(opt) => opt.minimize(objective, initial_params),
        }
    }

    fn name(&self) -> &'static str {
        match self {
            OptimizerWrapper::SGD(opt) => opt.name(),
            OptimizerWrapper::Adam(opt) => opt.name(),
            OptimizerWrapper::CoordinateDescent(opt) => opt.name(),
            OptimizerWrapper::ProximalGradient(opt) => opt.name(),
            OptimizerWrapper::NelderMead(opt) => opt.name(),
        }
    }

    fn config(&self) -> &OptimizationConfig {
        match self {
            OptimizerWrapper::SGD(opt) => opt.config(),
            OptimizerWrapper::Adam(opt) => opt.config(),
            OptimizerWrapper::CoordinateDescent(opt) => opt.config(),
            OptimizerWrapper::ProximalGradient(opt) => opt.config(),
            OptimizerWrapper::NelderMead(opt) => opt.config(),
        }
    }
}

/// Registry for managing optimization algorithms
#[derive(Debug)]
pub struct OptimizerRegistry {
    optimizers: HashMap<OptimizerType, OptimizerWrapper>,
}

impl OptimizerRegistry {
    /// Create a new optimizer registry
    pub fn new() -> Self {
        let mut registry = Self {
            optimizers: HashMap::new(),
        };

        // Register default optimizers
        registry.register_default_optimizers();
        registry
    }

    /// Register a new optimizer
    pub fn register(&mut self, optimizer_type: OptimizerType, optimizer: OptimizerWrapper) {
        self.optimizers.insert(optimizer_type, optimizer);
    }

    /// Get an optimizer by type
    pub fn get(&self, optimizer_type: &OptimizerType) -> Option<&OptimizerWrapper> {
        self.optimizers.get(optimizer_type)
    }

    /// List available optimizers
    pub fn available_optimizers(&self) -> Vec<OptimizerType> {
        self.optimizers.keys().cloned().collect()
    }

    /// Register default optimizers
    fn register_default_optimizers(&mut self) {
        self.register(
            OptimizerType::SGD,
            OptimizerWrapper::SGD(SGDOptimizer::default()),
        );
        self.register(
            OptimizerType::Adam,
            OptimizerWrapper::Adam(AdamOptimizer::default()),
        );
        self.register(
            OptimizerType::CoordinateDescent,
            OptimizerWrapper::CoordinateDescent(CoordinateDescentOptimizer::default()),
        );
        self.register(
            OptimizerType::ProximalGradient,
            OptimizerWrapper::ProximalGradient(ProximalGradientOptimizer::default()),
        );
        self.register(
            OptimizerType::NelderMead,
            OptimizerWrapper::NelderMead(NelderMeadOptimizer::default()),
        );
    }
}

impl Default for OptimizerRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Default optimization configuration
impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            tolerance: 1e-6,
            learning_rate: 0.01,
            momentum: 0.9,
            l1_regularization: 0.0,
            l2_regularization: 0.0,
            adaptive_learning_rate: false,
            line_search: LineSearchMethod::None,
            verbose: false,
            seed: None,
            algorithm_params: HashMap::new(),
        }
    }
}

/// Stochastic Gradient Descent optimizer
#[derive(Debug)]
pub struct SGDOptimizer {
    config: OptimizationConfig,
}

impl SGDOptimizer {
    /// Create a new SGD optimizer
    pub fn new(config: OptimizationConfig) -> Self {
        Self { config }
    }
}

impl Default for SGDOptimizer {
    fn default() -> Self {
        Self::new(OptimizationConfig::default())
    }
}

impl OptimizationAlgorithm for SGDOptimizer {
    type Parameters = Array1<f64>;

    fn minimize<F>(
        &self,
        objective: &F,
        mut params: Self::Parameters,
    ) -> Result<OptimizationResult<Self::Parameters>>
    where
        F: ObjectiveFunction<Parameters = Self::Parameters>,
    {
        let mut history = OptimizationHistory {
            objective_values: Vec::new(),
            gradient_norms: Vec::new(),
            step_sizes: Vec::new(),
            iteration_times: Vec::new(),
        };

        let mut function_evaluations = 0;
        let mut gradient_evaluations = 0;
        let mut converged = false;
        let mut learning_rate = self.config.learning_rate;

        for iteration in 0..self.config.max_iterations {
            let start_time = std::time::Instant::now();

            // Evaluate objective
            let obj_value = objective.evaluate(&params)?;
            function_evaluations += 1;
            history.objective_values.push(obj_value);

            // Compute gradient
            let gradient = if objective.has_gradient() {
                let grad = objective.gradient(&params)?;
                gradient_evaluations += 1;
                grad
            } else {
                // Finite difference approximation
                self.finite_difference_gradient(objective, &params)?
            };

            if let Some(grad) = gradient {
                let grad_norm = grad.mapv(|x| x.powi(2)).sum().sqrt();
                history.gradient_norms.push(grad_norm);

                // Check convergence
                if grad_norm < self.config.tolerance {
                    converged = true;
                    break;
                }

                // Apply regularization to gradient
                let mut regularized_grad = grad.clone();
                if self.config.l1_regularization > 0.0 {
                    regularized_grad +=
                        &params.mapv(|x| self.config.l1_regularization * x.signum());
                }
                if self.config.l2_regularization > 0.0 {
                    regularized_grad += &(&params * self.config.l2_regularization);
                }

                // Update parameters
                params -= &(&regularized_grad * learning_rate);
                history.step_sizes.push(learning_rate);

                // Adaptive learning rate
                if self.config.adaptive_learning_rate && iteration > 0 {
                    if history.objective_values[iteration] > history.objective_values[iteration - 1]
                    {
                        learning_rate *= 0.5;
                    } else {
                        learning_rate *= 1.01;
                    }
                }
            } else {
                return Err(SklearsError::InvalidInput(
                    "Cannot compute gradient for SGD optimization".to_string(),
                ));
            }

            let iteration_time = start_time.elapsed().as_secs_f64();
            history.iteration_times.push(iteration_time);

            if self.config.verbose && iteration % 100 == 0 {
                println!(
                    "Iteration {}: objective = {:.6}, gradient norm = {:.6}",
                    iteration,
                    obj_value,
                    history.gradient_norms.last().unwrap_or(&0.0)
                );
            }
        }

        let final_objective = objective.evaluate(&params)?;
        function_evaluations += 1;

        Ok(OptimizationResult {
            parameters: params,
            objective_value: final_objective,
            iterations: history.objective_values.len(),
            function_evaluations,
            gradient_evaluations,
            converged,
            message: if converged {
                "Converged".to_string()
            } else {
                "Maximum iterations reached".to_string()
            },
            history,
        })
    }

    fn name(&self) -> &'static str {
        "SGD"
    }

    fn config(&self) -> &OptimizationConfig {
        &self.config
    }
}

impl SGDOptimizer {
    /// Finite difference gradient approximation
    fn finite_difference_gradient<F>(
        &self,
        objective: &F,
        params: &Array1<f64>,
    ) -> Result<Option<Array1<f64>>>
    where
        F: ObjectiveFunction<Parameters = Array1<f64>>,
    {
        let eps = 1e-8;
        let mut gradient = Array1::zeros(params.len());
        let f0 = objective.evaluate(params)?;

        for i in 0..params.len() {
            let mut params_plus = params.clone();
            params_plus[i] += eps;
            let f_plus = objective.evaluate(&params_plus)?;
            gradient[i] = (f_plus - f0) / eps;
        }

        Ok(Some(gradient))
    }
}

/// Adam optimizer
#[derive(Debug)]
pub struct AdamOptimizer {
    config: OptimizationConfig,
}

impl AdamOptimizer {
    /// Create a new Adam optimizer
    pub fn new(config: OptimizationConfig) -> Self {
        Self { config }
    }
}

impl Default for AdamOptimizer {
    fn default() -> Self {
        let mut config = OptimizationConfig::default();
        config.algorithm_params.insert("beta1".to_string(), 0.9);
        config.algorithm_params.insert("beta2".to_string(), 0.999);
        config.algorithm_params.insert("epsilon".to_string(), 1e-8);
        Self::new(config)
    }
}

impl OptimizationAlgorithm for AdamOptimizer {
    type Parameters = Array1<f64>;

    fn minimize<F>(
        &self,
        objective: &F,
        mut params: Self::Parameters,
    ) -> Result<OptimizationResult<Self::Parameters>>
    where
        F: ObjectiveFunction<Parameters = Self::Parameters>,
    {
        let beta1 = self
            .config
            .algorithm_params
            .get("beta1")
            .copied()
            .unwrap_or(0.9);
        let beta2 = self
            .config
            .algorithm_params
            .get("beta2")
            .copied()
            .unwrap_or(0.999);
        let epsilon = self
            .config
            .algorithm_params
            .get("epsilon")
            .copied()
            .unwrap_or(1e-8);

        let mut m = Array1::zeros(params.len()); // First moment
        let mut v = Array1::zeros(params.len()); // Second moment

        let mut history = OptimizationHistory {
            objective_values: Vec::new(),
            gradient_norms: Vec::new(),
            step_sizes: Vec::new(),
            iteration_times: Vec::new(),
        };

        let mut function_evaluations = 0;
        let mut gradient_evaluations = 0;
        let mut converged = false;

        for iteration in 0..self.config.max_iterations {
            let start_time = std::time::Instant::now();
            let t = (iteration + 1) as f64;

            // Evaluate objective
            let obj_value = objective.evaluate(&params)?;
            function_evaluations += 1;
            history.objective_values.push(obj_value);

            // Compute gradient
            let gradient = if objective.has_gradient() {
                let grad = objective.gradient(&params)?;
                gradient_evaluations += 1;
                grad
            } else {
                // Use finite difference approximation
                let sgd = SGDOptimizer::default();
                sgd.finite_difference_gradient(objective, &params)?
            };

            if let Some(grad) = gradient {
                let grad_norm = grad.mapv(|x| x.powi(2)).sum().sqrt();
                history.gradient_norms.push(grad_norm);

                // Check convergence
                if grad_norm < self.config.tolerance {
                    converged = true;
                    break;
                }

                // Update biased first moment estimate
                m = &m * beta1 + &grad * (1.0 - beta1);

                // Update biased second raw moment estimate
                v = &v * beta2 + &grad.mapv(|x| x.powi(2)) * (1.0 - beta2);

                // Compute bias-corrected first moment estimate
                let m_hat = &m / (1.0 - beta1.powf(t));

                // Compute bias-corrected second raw moment estimate
                let v_hat = &v / (1.0 - beta2.powf(t));

                // Update parameters
                let step =
                    &m_hat / &v_hat.mapv(|x: f64| (x + epsilon).sqrt()) * self.config.learning_rate;
                params -= &step;

                let step_size = step.mapv(|x| x.powi(2)).sum().sqrt();
                history.step_sizes.push(step_size);
            } else {
                return Err(SklearsError::InvalidInput(
                    "Cannot compute gradient for Adam optimization".to_string(),
                ));
            }

            let iteration_time = start_time.elapsed().as_secs_f64();
            history.iteration_times.push(iteration_time);

            if self.config.verbose && iteration % 100 == 0 {
                println!(
                    "Iteration {}: objective = {:.6}, gradient norm = {:.6}",
                    iteration,
                    obj_value,
                    history.gradient_norms.last().unwrap_or(&0.0)
                );
            }
        }

        let final_objective = objective.evaluate(&params)?;
        function_evaluations += 1;

        Ok(OptimizationResult {
            parameters: params,
            objective_value: final_objective,
            iterations: history.objective_values.len(),
            function_evaluations,
            gradient_evaluations,
            converged,
            message: if converged {
                "Converged".to_string()
            } else {
                "Maximum iterations reached".to_string()
            },
            history,
        })
    }

    fn name(&self) -> &'static str {
        "Adam"
    }

    fn config(&self) -> &OptimizationConfig {
        &self.config
    }
}

/// Coordinate descent optimizer
#[derive(Debug)]
pub struct CoordinateDescentOptimizer {
    config: OptimizationConfig,
}

impl CoordinateDescentOptimizer {
    /// Create a new coordinate descent optimizer
    pub fn new(config: OptimizationConfig) -> Self {
        Self { config }
    }
}

impl Default for CoordinateDescentOptimizer {
    fn default() -> Self {
        Self::new(OptimizationConfig::default())
    }
}

impl OptimizationAlgorithm for CoordinateDescentOptimizer {
    type Parameters = Array1<f64>;

    fn minimize<F>(
        &self,
        objective: &F,
        mut params: Self::Parameters,
    ) -> Result<OptimizationResult<Self::Parameters>>
    where
        F: ObjectiveFunction<Parameters = Self::Parameters>,
    {
        let mut history = OptimizationHistory {
            objective_values: Vec::new(),
            gradient_norms: Vec::new(),
            step_sizes: Vec::new(),
            iteration_times: Vec::new(),
        };

        let mut function_evaluations = 0;
        let mut converged = false;
        let step_size = self.config.learning_rate;

        for iteration in 0..self.config.max_iterations {
            let start_time = std::time::Instant::now();

            // Evaluate objective
            let obj_value = objective.evaluate(&params)?;
            function_evaluations += 1;
            history.objective_values.push(obj_value);

            let mut max_change: f64 = 0.0;

            // Update each coordinate
            for i in 0..params.len() {
                let old_value = params[i];

                // Line search along coordinate i
                let mut best_value = old_value;
                let mut best_objective = obj_value;

                // Try positive and negative steps
                for &direction in &[1.0, -1.0] {
                    params[i] = old_value + direction * step_size;
                    let new_objective = objective.evaluate(&params)?;
                    function_evaluations += 1;

                    if new_objective < best_objective {
                        best_objective = new_objective;
                        best_value = params[i];
                    }
                }

                params[i] = best_value;
                max_change = max_change.max((best_value - old_value).abs());
            }

            history.gradient_norms.push(max_change);
            history.step_sizes.push(step_size);

            // Check convergence
            if max_change < self.config.tolerance {
                converged = true;
                break;
            }

            let iteration_time = start_time.elapsed().as_secs_f64();
            history.iteration_times.push(iteration_time);

            if self.config.verbose && iteration % 100 == 0 {
                println!(
                    "Iteration {}: objective = {:.6}, max change = {:.6}",
                    iteration, obj_value, max_change
                );
            }
        }

        let final_objective = objective.evaluate(&params)?;
        function_evaluations += 1;

        Ok(OptimizationResult {
            parameters: params,
            objective_value: final_objective,
            iterations: history.objective_values.len(),
            function_evaluations,
            gradient_evaluations: 0,
            converged,
            message: if converged {
                "Converged".to_string()
            } else {
                "Maximum iterations reached".to_string()
            },
            history,
        })
    }

    fn name(&self) -> &'static str {
        "CoordinateDescent"
    }

    fn config(&self) -> &OptimizationConfig {
        &self.config
    }
}

/// Proximal gradient optimizer
#[derive(Debug)]
pub struct ProximalGradientOptimizer {
    config: OptimizationConfig,
}

impl ProximalGradientOptimizer {
    /// Create a new proximal gradient optimizer
    pub fn new(config: OptimizationConfig) -> Self {
        Self { config }
    }

    /// Soft thresholding operator for L1 regularization
    fn soft_threshold(&self, x: f64, lambda: f64) -> f64 {
        if x > lambda {
            x - lambda
        } else if x < -lambda {
            x + lambda
        } else {
            0.0
        }
    }
}

impl Default for ProximalGradientOptimizer {
    fn default() -> Self {
        Self::new(OptimizationConfig::default())
    }
}

impl OptimizationAlgorithm for ProximalGradientOptimizer {
    type Parameters = Array1<f64>;

    fn minimize<F>(
        &self,
        objective: &F,
        mut params: Self::Parameters,
    ) -> Result<OptimizationResult<Self::Parameters>>
    where
        F: ObjectiveFunction<Parameters = Self::Parameters>,
    {
        let mut history = OptimizationHistory {
            objective_values: Vec::new(),
            gradient_norms: Vec::new(),
            step_sizes: Vec::new(),
            iteration_times: Vec::new(),
        };

        let mut function_evaluations = 0;
        let mut gradient_evaluations = 0;
        let mut converged = false;
        let learning_rate = self.config.learning_rate;

        for iteration in 0..self.config.max_iterations {
            let start_time = std::time::Instant::now();

            // Evaluate objective
            let obj_value = objective.evaluate(&params)?;
            function_evaluations += 1;
            history.objective_values.push(obj_value);

            // Compute gradient (without regularization)
            let gradient = if objective.has_gradient() {
                let grad = objective.gradient(&params)?;
                gradient_evaluations += 1;
                grad
            } else {
                let sgd = SGDOptimizer::default();
                sgd.finite_difference_gradient(objective, &params)?
            };

            if let Some(grad) = gradient {
                let grad_norm = grad.mapv(|x| x.powi(2)).sum().sqrt();
                history.gradient_norms.push(grad_norm);

                // Check convergence
                if grad_norm < self.config.tolerance {
                    converged = true;
                    break;
                }

                // Gradient step
                let grad_step = &params - &(&grad * learning_rate);

                // Proximal step (apply regularization)
                if self.config.l1_regularization > 0.0 {
                    let lambda = self.config.l1_regularization * learning_rate;
                    params = grad_step.mapv(|x| self.soft_threshold(x, lambda));
                } else {
                    params = grad_step;
                }

                // Apply L2 regularization
                if self.config.l2_regularization > 0.0 {
                    let shrinkage = 1.0 / (1.0 + self.config.l2_regularization * learning_rate);
                    params *= shrinkage;
                }

                history.step_sizes.push(learning_rate);
            } else {
                return Err(SklearsError::InvalidInput(
                    "Cannot compute gradient for proximal gradient optimization".to_string(),
                ));
            }

            let iteration_time = start_time.elapsed().as_secs_f64();
            history.iteration_times.push(iteration_time);

            if self.config.verbose && iteration % 100 == 0 {
                println!(
                    "Iteration {}: objective = {:.6}, gradient norm = {:.6}",
                    iteration,
                    obj_value,
                    history.gradient_norms.last().unwrap_or(&0.0)
                );
            }
        }

        let final_objective = objective.evaluate(&params)?;
        function_evaluations += 1;

        Ok(OptimizationResult {
            parameters: params,
            objective_value: final_objective,
            iterations: history.objective_values.len(),
            function_evaluations,
            gradient_evaluations,
            converged,
            message: if converged {
                "Converged".to_string()
            } else {
                "Maximum iterations reached".to_string()
            },
            history,
        })
    }

    fn name(&self) -> &'static str {
        "ProximalGradient"
    }

    fn config(&self) -> &OptimizationConfig {
        &self.config
    }
}

/// Nelder-Mead simplex optimizer (derivative-free)
#[derive(Debug)]
pub struct NelderMeadOptimizer {
    config: OptimizationConfig,
}

impl NelderMeadOptimizer {
    /// Create a new Nelder-Mead optimizer
    pub fn new(config: OptimizationConfig) -> Self {
        Self { config }
    }
}

impl Default for NelderMeadOptimizer {
    fn default() -> Self {
        Self::new(OptimizationConfig::default())
    }
}

impl OptimizationAlgorithm for NelderMeadOptimizer {
    type Parameters = Array1<f64>;

    fn minimize<F>(
        &self,
        objective: &F,
        initial_params: Self::Parameters,
    ) -> Result<OptimizationResult<Self::Parameters>>
    where
        F: ObjectiveFunction<Parameters = Self::Parameters>,
    {
        let n = initial_params.len();
        let mut history = OptimizationHistory {
            objective_values: Vec::new(),
            gradient_norms: Vec::new(),
            step_sizes: Vec::new(),
            iteration_times: Vec::new(),
        };

        // Initialize simplex
        let step_size = 0.1;
        let mut simplex = Vec::new();
        let mut values = Vec::new();

        // Add initial point
        simplex.push(initial_params.clone());
        values.push(objective.evaluate(&initial_params)?);

        // Add other vertices
        for i in 0..n {
            let mut vertex = initial_params.clone();
            vertex[i] += step_size;
            let value = objective.evaluate(&vertex)?;
            simplex.push(vertex);
            values.push(value);
        }

        let mut function_evaluations = n + 1;
        let mut converged = false;

        // Nelder-Mead parameters
        let alpha = 1.0; // Reflection
        let gamma = 2.0; // Expansion
        let rho = 0.5; // Contraction
        let sigma = 0.5; // Shrinkage

        for iteration in 0..self.config.max_iterations {
            let start_time = std::time::Instant::now();

            // Sort simplex by function values
            let mut indices: Vec<usize> = (0..simplex.len()).collect();
            indices.sort_by(|&i, &j| values[i].partial_cmp(&values[j]).unwrap());

            let best_idx = indices[0];
            let worst_idx = indices[n];
            let second_worst_idx = indices[n - 1];

            history.objective_values.push(values[best_idx]);

            // Check convergence
            let range = values[worst_idx] - values[best_idx];
            if range < self.config.tolerance {
                converged = true;
                break;
            }

            history.gradient_norms.push(range);
            history.step_sizes.push(step_size);

            // Compute centroid (excluding worst point)
            let mut centroid = Array1::zeros(n);
            for &idx in &indices[0..n] {
                centroid += &simplex[idx];
            }
            centroid /= n as f64;

            // Reflection
            let reflected = &centroid + &((&centroid - &simplex[worst_idx]) * alpha);
            let reflected_value = objective.evaluate(&reflected)?;
            function_evaluations += 1;

            if values[best_idx] <= reflected_value && reflected_value < values[second_worst_idx] {
                // Accept reflection
                simplex[worst_idx] = reflected;
                values[worst_idx] = reflected_value;
            } else if reflected_value < values[best_idx] {
                // Try expansion
                let expanded = &centroid + &((&reflected - &centroid) * gamma);
                let expanded_value = objective.evaluate(&expanded)?;
                function_evaluations += 1;

                if expanded_value < reflected_value {
                    simplex[worst_idx] = expanded;
                    values[worst_idx] = expanded_value;
                } else {
                    simplex[worst_idx] = reflected;
                    values[worst_idx] = reflected_value;
                }
            } else {
                // Try contraction
                let contracted = &centroid + &((&simplex[worst_idx] - &centroid) * rho);
                let contracted_value = objective.evaluate(&contracted)?;
                function_evaluations += 1;

                if contracted_value < values[worst_idx] {
                    simplex[worst_idx] = contracted;
                    values[worst_idx] = contracted_value;
                } else {
                    // Shrink simplex
                    for i in 1..simplex.len() {
                        simplex[i] =
                            &simplex[best_idx] + &((&simplex[i] - &simplex[best_idx]) * sigma);
                        values[i] = objective.evaluate(&simplex[i])?;
                        function_evaluations += 1;
                    }
                }
            }

            let iteration_time = start_time.elapsed().as_secs_f64();
            history.iteration_times.push(iteration_time);

            if self.config.verbose && iteration % 100 == 0 {
                println!(
                    "Iteration {}: objective = {:.6}, range = {:.6}",
                    iteration, values[best_idx], range
                );
            }
        }

        // Find best solution
        let best_idx = values
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap();

        Ok(OptimizationResult {
            parameters: simplex[best_idx].clone(),
            objective_value: values[best_idx],
            iterations: history.objective_values.len(),
            function_evaluations,
            gradient_evaluations: 0,
            converged,
            message: if converged {
                "Converged".to_string()
            } else {
                "Maximum iterations reached".to_string()
            },
            history,
        })
    }

    fn name(&self) -> &'static str {
        "NelderMead"
    }

    fn config(&self) -> &OptimizationConfig {
        &self.config
    }
}

/// Builder for optimization configuration
#[derive(Debug, Clone)]
pub struct OptimizationConfigBuilder {
    config: OptimizationConfig,
}

impl OptimizationConfigBuilder {
    /// Create a new configuration builder
    pub fn new() -> Self {
        Self {
            config: OptimizationConfig::default(),
        }
    }

    /// Set maximum iterations
    pub fn max_iterations(mut self, max_iter: usize) -> Self {
        self.config.max_iterations = max_iter;
        self
    }

    /// Set tolerance
    pub fn tolerance(mut self, tol: f64) -> Self {
        self.config.tolerance = tol;
        self
    }

    /// Set learning rate
    pub fn learning_rate(mut self, lr: f64) -> Self {
        self.config.learning_rate = lr;
        self
    }

    /// Set momentum
    pub fn momentum(mut self, momentum: f64) -> Self {
        self.config.momentum = momentum;
        self
    }

    /// Set L1 regularization
    pub fn l1_regularization(mut self, l1: f64) -> Self {
        self.config.l1_regularization = l1;
        self
    }

    /// Set L2 regularization
    pub fn l2_regularization(mut self, l2: f64) -> Self {
        self.config.l2_regularization = l2;
        self
    }

    /// Enable adaptive learning rate
    pub fn adaptive_learning_rate(mut self, adaptive: bool) -> Self {
        self.config.adaptive_learning_rate = adaptive;
        self
    }

    /// Set line search method
    pub fn line_search(mut self, method: LineSearchMethod) -> Self {
        self.config.line_search = method;
        self
    }

    /// Set verbosity
    pub fn verbose(mut self, verbose: bool) -> Self {
        self.config.verbose = verbose;
        self
    }

    /// Set random seed
    pub fn seed(mut self, seed: u64) -> Self {
        self.config.seed = Some(seed);
        self
    }

    /// Add algorithm-specific parameter
    pub fn algorithm_param(mut self, key: String, value: f64) -> Self {
        self.config.algorithm_params.insert(key, value);
        self
    }

    /// Build the configuration
    pub fn build(self) -> OptimizationConfig {
        self.config
    }
}

impl Default for OptimizationConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    // Simple quadratic objective function for testing
    #[derive(Debug)]
    struct QuadraticObjective {
        a: Array2<f64>,
        b: Array1<f64>,
        c: f64,
    }

    impl ObjectiveFunction for QuadraticObjective {
        type Parameters = Array1<f64>;

        fn evaluate(&self, params: &Self::Parameters) -> Result<f64> {
            let quadratic_term = params.dot(&self.a.dot(params));
            let linear_term = self.b.dot(params);
            Ok(0.5 * quadratic_term + linear_term + self.c)
        }

        fn gradient(&self, params: &Self::Parameters) -> Result<Option<Self::Parameters>> {
            let grad = self.a.dot(params) + &self.b;
            Ok(Some(grad))
        }

        fn has_gradient(&self) -> bool {
            true
        }
    }

    #[test]
    fn test_sgd_optimizer() {
        let a = array![[2.0, 0.0], [0.0, 2.0]];
        let b = array![-2.0, -4.0];
        let c = 1.0;

        let objective = QuadraticObjective { a, b, c };
        let optimizer = SGDOptimizer::default();
        let initial_params = array![0.0, 0.0];

        let result = optimizer.minimize(&objective, initial_params).unwrap();

        assert!(result.converged);
        assert!(result.objective_value < 1e-6);

        // Optimal solution should be approximately [1.0, 2.0]
        assert!((result.parameters[0] - 1.0).abs() < 1e-3);
        assert!((result.parameters[1] - 2.0).abs() < 1e-3);
    }

    #[test]
    fn test_adam_optimizer() {
        let a = array![[1.0, 0.0], [0.0, 1.0]];
        let b = array![-1.0, -1.0];
        let c = 0.0;

        let objective = QuadraticObjective { a, b, c };
        let optimizer = AdamOptimizer::default();
        let initial_params = array![0.0, 0.0];

        let result = optimizer.minimize(&objective, initial_params).unwrap();

        assert!(result.objective_value < 1e-3);
        assert!(!result.history.objective_values.is_empty());
    }

    #[test]
    fn test_coordinate_descent_optimizer() {
        let a = array![[1.0, 0.0], [0.0, 1.0]];
        let b = array![-2.0, -2.0];
        let c = 0.0;

        let objective = QuadraticObjective { a, b, c };
        let optimizer = CoordinateDescentOptimizer::default();
        let initial_params = array![0.0, 0.0];

        let result = optimizer.minimize(&objective, initial_params).unwrap();

        assert!(result.objective_value < 1e-3);
        assert_eq!(result.gradient_evaluations, 0); // Coordinate descent doesn't use gradients
    }

    #[test]
    fn test_nelder_mead_optimizer() {
        let a = array![[1.0, 0.0], [0.0, 1.0]];
        let b = array![-2.0, -2.0];
        let c = 0.0;

        let objective = QuadraticObjective { a, b, c };
        let optimizer = NelderMeadOptimizer::default();
        let initial_params = array![0.0, 0.0];

        let result = optimizer.minimize(&objective, initial_params).unwrap();

        assert!(result.objective_value < 1e-2);
        assert_eq!(result.gradient_evaluations, 0); // Nelder-Mead is derivative-free
    }

    #[test]
    fn test_optimizer_registry() {
        let mut registry = OptimizerRegistry::new();

        let available = registry.available_optimizers();
        assert!(available.contains(&OptimizerType::SGD));
        assert!(available.contains(&OptimizerType::Adam));
        assert!(available.contains(&OptimizerType::CoordinateDescent));

        let sgd = registry.get(&OptimizerType::SGD);
        assert!(sgd.is_some());
        assert_eq!(sgd.unwrap().name(), "SGD");
    }

    #[test]
    fn test_optimization_config_builder() {
        let config = OptimizationConfigBuilder::new()
            .max_iterations(500)
            .tolerance(1e-8)
            .learning_rate(0.001)
            .l1_regularization(0.01)
            .verbose(true)
            .build();

        assert_eq!(config.max_iterations, 500);
        assert_eq!(config.tolerance, 1e-8);
        assert_eq!(config.learning_rate, 0.001);
        assert_eq!(config.l1_regularization, 0.01);
        assert!(config.verbose);
    }

    #[test]
    fn test_proximal_gradient_with_l1_regularization() {
        let a = array![[1.0, 0.0], [0.0, 1.0]];
        let b = array![-1.0, -1.0];
        let c = 0.0;

        let objective = QuadraticObjective { a, b, c };

        let config = OptimizationConfigBuilder::new()
            .l1_regularization(0.1)
            .learning_rate(0.1)
            .max_iterations(1000)
            .build();

        let optimizer = ProximalGradientOptimizer::new(config);
        let initial_params = array![0.0, 0.0];

        let result = optimizer.minimize(&objective, initial_params).unwrap();

        // With L1 regularization, some parameters might be exactly zero
        assert!(result.objective_value.is_finite()); // Objective value should be finite
        assert!(!result.history.objective_values.is_empty());
    }
}
