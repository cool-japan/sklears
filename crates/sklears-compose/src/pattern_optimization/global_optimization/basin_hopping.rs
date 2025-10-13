//! Basin hopping algorithms for global optimization.
//!
//! This module implements basin hopping methods that combine local minimization
//! with random perturbations and Monte Carlo acceptance criteria to escape local
//! optima and find global solutions. Basin hopping is particularly effective for
//! continuous optimization problems with multiple local minima.
//!
//! # Algorithm Overview
//!
//! Basin hopping works by:
//! 1. Starting from an initial point
//! 2. Performing local minimization to find a local minimum
//! 3. Making a random perturbation (hop) to a new starting point
//! 4. Applying Monte Carlo acceptance criteria based on energy difference
//! 5. Repeating until convergence or maximum iterations
//!
//! # Key Features
//!
//! - **Adaptive perturbations**: Multiple strategies for generating effective hops
//! - **Temperature control**: Sophisticated cooling schedules with reheat capabilities
//! - **Basin analysis**: Comprehensive landscape characterization and diagnostics
//! - **Local search integration**: Seamless integration with gradient-based optimizers
//! - **Performance monitoring**: Detailed effectiveness metrics and recommendations
//!
//! # Examples
//!
//! ```rust
//! use sklears_compose::pattern_optimization::global_optimization::basin_hopping::*;
//! use scirs2_core::ndarray::Array1;
//!
//! // Create basin hopping parameters
//! let params = HoppingParameters::builder()
//!     .step_size(0.5)
//!     .temperature(1.0)
//!     .cooling_rate(0.95)
//!     .perturbation_strategy(PerturbationStrategy::GaussianRandom { sigma: 0.3 })
//!     .build();
//!
//! // The basin hopper would be implemented by specific algorithms
//! // that use these parameters and follow the BasinHopping trait
//! ```
//!
//! ## Perturbation Strategies
//!
//! Different perturbation strategies are available:
//! - **Uniform Random**: Perturbations within a fixed radius
//! - **Gaussian Random**: Normal distribution perturbations
//! - **Adaptive**: Perturbations that adapt based on acceptance history
//! - **Guided**: Perturbations using gradient or hessian information
//!
//! ## Temperature Schedules
//!
//! Various cooling schedules control the acceptance probability:
//! - Linear cooling with optional reheat
//! - Exponential cooling schedules
//! - Logarithmic cooling for slow exploration
//! - Adaptive schedules based on acceptance rates

use std::collections::HashMap;
use std::time::Duration;

// SciRS2 imports following compliance requirements
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::error::CoreError;

// Local imports
use sklears_core::error::Result as SklResult;
use crate::enhanced_errors::PipelineError;

/// Basin hopping algorithm for escaping local optima.
///
/// Implements the basin hopping method that combines local minimization
/// with random perturbations and Monte Carlo acceptance criteria.
/// Provides excellent global optimization capability for continuous problems.
pub trait BasinHopping: Send + Sync {
    /// Perform a hopping step from the current solution.
    ///
    /// Generates a new candidate solution through perturbation mechanisms
    /// including random displacement, guided perturbations, and adaptive strategies.
    ///
    /// # Arguments
    /// * `current_solution` - Current best solution
    ///
    /// # Returns
    /// New candidate solution after perturbation
    fn hop_step(&self, current_solution: &Solution) -> SklResult<Solution>;

    /// Evaluate Monte Carlo acceptance criterion.
    ///
    /// Determines whether to accept a new solution based on energy difference
    /// and current temperature using Metropolis or custom criteria.
    ///
    /// # Arguments
    /// * `current_energy` - Energy of current solution
    /// * `new_energy` - Energy of candidate solution
    /// * `temperature` - Current temperature parameter
    ///
    /// # Returns
    /// True if the move should be accepted
    fn accept_criterion(&self, current_energy: f64, new_energy: f64, temperature: f64) -> bool;

    /// Update temperature according to cooling schedule.
    ///
    /// Implements adaptive temperature control including linear, exponential,
    /// and logarithmic cooling schedules with reheat capabilities.
    ///
    /// # Arguments
    /// * `acceptance_ratio` - Recent acceptance ratio for adaptive control
    ///
    /// # Returns
    /// Result of temperature update operation
    fn update_temperature(&mut self, acceptance_ratio: f64) -> SklResult<()>;

    /// Get current hopping parameters.
    fn get_hopping_parameters(&self) -> HoppingParameters;

    /// Set hopping parameters for adaptive control.
    fn set_hopping_parameters(&mut self, params: HoppingParameters) -> SklResult<()>;

    /// Perform local minimization after hop.
    ///
    /// Integrates with gradient-based local optimizers for hybrid optimization.
    /// Uses BFGS, L-BFGS-B, or other methods for local refinement.
    ///
    /// # Arguments
    /// * `starting_point` - Point to start local minimization
    /// * `problem` - Optimization problem definition
    ///
    /// # Returns
    /// Locally optimized solution
    fn local_minimize(&self, starting_point: &Array1<f64>, problem: &OptimizationProblem) -> SklResult<Solution>;

    /// Analyze basin structure and hopping effectiveness.
    ///
    /// Provides diagnostic information about basin identification,
    /// hopping success rates, and parameter recommendations.
    ///
    /// # Returns
    /// Basin hopping analysis results
    fn analyze_basin_structure(&self) -> SklResult<BasinAnalysis>;
}

/// Configuration parameters for basin hopping algorithms.
#[derive(Debug, Clone)]
pub struct HoppingParameters {
    /// Step size for random perturbations
    pub step_size: f64,
    /// Current temperature for acceptance criterion
    pub temperature: f64,
    /// Cooling rate for temperature reduction
    pub cooling_rate: f64,
    /// Minimum temperature threshold
    pub min_temperature: f64,
    /// Maximum number of hopping attempts per iteration
    pub max_hop_attempts: u32,
    /// Perturbation strategy (random, guided, adaptive)
    pub perturbation_strategy: PerturbationStrategy,
    /// Local minimization integration settings
    pub local_minimizer_config: LocalMinimizerConfig,
    /// Temperature reheat parameters
    pub reheat_parameters: Option<ReheatParameters>,
}

impl Default for HoppingParameters {
    fn default() -> Self {
        Self {
            step_size: 0.5,
            temperature: 1.0,
            cooling_rate: 0.95,
            min_temperature: 1e-6,
            max_hop_attempts: 100,
            perturbation_strategy: PerturbationStrategy::GaussianRandom { sigma: 0.3 },
            local_minimizer_config: LocalMinimizerConfig::default(),
            reheat_parameters: None,
        }
    }
}

impl HoppingParameters {
    /// Create a new builder for HoppingParameters.
    pub fn builder() -> HoppingParametersBuilder {
        HoppingParametersBuilder::new()
    }

    /// Validate the hopping parameters.
    pub fn validate(&self) -> SklResult<()> {
        if self.step_size <= 0.0 {
            return Err(CoreError::InvalidInput("Step size must be positive".to_string()).into());
        }
        if self.temperature <= 0.0 {
            return Err(CoreError::InvalidInput("Temperature must be positive".to_string()).into());
        }
        if self.cooling_rate <= 0.0 || self.cooling_rate >= 1.0 {
            return Err(CoreError::InvalidInput("Cooling rate must be between 0 and 1".to_string()).into());
        }
        if self.min_temperature <= 0.0 {
            return Err(CoreError::InvalidInput("Minimum temperature must be positive".to_string()).into());
        }
        if self.max_hop_attempts == 0 {
            return Err(CoreError::InvalidInput("Maximum hop attempts must be positive".to_string()).into());
        }
        Ok(())
    }
}

/// Builder for HoppingParameters.
#[derive(Debug)]
pub struct HoppingParametersBuilder {
    step_size: f64,
    temperature: f64,
    cooling_rate: f64,
    min_temperature: f64,
    max_hop_attempts: u32,
    perturbation_strategy: PerturbationStrategy,
    local_minimizer_config: LocalMinimizerConfig,
    reheat_parameters: Option<ReheatParameters>,
}

impl HoppingParametersBuilder {
    pub fn new() -> Self {
        Self {
            step_size: 0.5,
            temperature: 1.0,
            cooling_rate: 0.95,
            min_temperature: 1e-6,
            max_hop_attempts: 100,
            perturbation_strategy: PerturbationStrategy::GaussianRandom { sigma: 0.3 },
            local_minimizer_config: LocalMinimizerConfig::default(),
            reheat_parameters: None,
        }
    }

    /// Set the step size for perturbations.
    pub fn step_size(mut self, step_size: f64) -> Self {
        self.step_size = step_size;
        self
    }

    /// Set the initial temperature.
    pub fn temperature(mut self, temperature: f64) -> Self {
        self.temperature = temperature;
        self
    }

    /// Set the cooling rate.
    pub fn cooling_rate(mut self, rate: f64) -> Self {
        self.cooling_rate = rate;
        self
    }

    /// Set the minimum temperature.
    pub fn min_temperature(mut self, min_temp: f64) -> Self {
        self.min_temperature = min_temp;
        self
    }

    /// Set the maximum hop attempts per iteration.
    pub fn max_hop_attempts(mut self, max_attempts: u32) -> Self {
        self.max_hop_attempts = max_attempts;
        self
    }

    /// Set the perturbation strategy.
    pub fn perturbation_strategy(mut self, strategy: PerturbationStrategy) -> Self {
        self.perturbation_strategy = strategy;
        self
    }

    /// Set the local minimizer configuration.
    pub fn local_minimizer_config(mut self, config: LocalMinimizerConfig) -> Self {
        self.local_minimizer_config = config;
        self
    }

    /// Set the reheat parameters.
    pub fn reheat_parameters(mut self, params: ReheatParameters) -> Self {
        self.reheat_parameters = Some(params);
        self
    }

    /// Build the HoppingParameters.
    pub fn build(self) -> SklResult<HoppingParameters> {
        let params = HoppingParameters {
            step_size: self.step_size,
            temperature: self.temperature,
            cooling_rate: self.cooling_rate,
            min_temperature: self.min_temperature,
            max_hop_attempts: self.max_hop_attempts,
            perturbation_strategy: self.perturbation_strategy,
            local_minimizer_config: self.local_minimizer_config,
            reheat_parameters: self.reheat_parameters,
        };
        params.validate()?;
        Ok(params)
    }
}

/// Perturbation strategies for basin hopping.
#[derive(Debug, Clone)]
pub enum PerturbationStrategy {
    /// Uniform random perturbations within a fixed radius
    UniformRandom { radius: f64 },
    /// Gaussian random perturbations with specified standard deviation
    GaussianRandom { sigma: f64 },
    /// Adaptive perturbations based on acceptance history
    Adaptive { adaptation_rate: f64 },
    /// Guided perturbations using gradient information
    Guided { gradient_weight: f64 },
    /// Levy flight perturbations for long-range exploration
    LevyFlight { alpha: f64, scale: f64 },
    /// Custom perturbation function
    Custom { name: String },
}

impl Default for PerturbationStrategy {
    fn default() -> Self {
        PerturbationStrategy::GaussianRandom { sigma: 0.3 }
    }
}

/// Local minimizer configuration for basin hopping.
#[derive(Debug, Clone)]
pub struct LocalMinimizerConfig {
    /// Local optimization method to use
    pub method: LocalSearchMethod,
    /// Maximum number of local optimization iterations
    pub max_iterations: u32,
    /// Convergence tolerance for local optimization
    pub tolerance: f64,
    /// Whether to use numerical gradients
    pub use_numerical_gradients: bool,
    /// Step size for numerical gradient computation
    pub gradient_step_size: f64,
}

impl Default for LocalMinimizerConfig {
    fn default() -> Self {
        Self {
            method: LocalSearchMethod::BFGS,
            max_iterations: 1000,
            tolerance: 1e-6,
            use_numerical_gradients: true,
            gradient_step_size: 1e-8,
        }
    }
}

/// Local search methods available for basin hopping.
#[derive(Debug, Clone)]
pub enum LocalSearchMethod {
    /// Broyden-Fletcher-Goldfarb-Shanno quasi-Newton method
    BFGS,
    /// Limited-memory BFGS
    LBFGS,
    /// L-BFGS-B with bound constraints
    LBFGSB,
    /// Conjugate gradient method
    ConjugateGradient,
    /// Nelder-Mead simplex method
    NelderMead,
    /// Trust region methods
    TrustRegion,
    /// Powell's method
    Powell,
}

/// Temperature reheat parameters for basin hopping.
#[derive(Debug, Clone)]
pub struct ReheatParameters {
    /// Threshold acceptance rate for triggering reheat
    pub acceptance_threshold: f64,
    /// Factor by which to increase temperature during reheat
    pub reheat_factor: f64,
    /// Maximum number of reheats allowed
    pub max_reheats: u32,
    /// Minimum iterations between reheats
    pub min_iterations_between_reheats: u32,
}

impl Default for ReheatParameters {
    fn default() -> Self {
        Self {
            acceptance_threshold: 0.1,
            reheat_factor: 2.0,
            max_reheats: 5,
            min_iterations_between_reheats: 50,
        }
    }
}

/// Basin analysis results for hopping methods.
#[derive(Debug)]
pub struct BasinAnalysis {
    /// Basin identification results
    pub basin_identification: BasinIdentification,
    /// Hopping effectiveness metrics
    pub hopping_effectiveness: HoppingEffectiveness,
    /// Parameter recommendations
    pub parameter_recommendations: HashMap<String, f64>,
    /// Landscape characterization
    pub landscape_characteristics: LandscapeCharacteristics,
    /// Performance diagnostics
    pub performance_diagnostics: PerformanceDiagnostics,
}

/// Basin identification results.
#[derive(Debug)]
pub struct BasinIdentification {
    /// Number of basins identified
    pub basin_count: usize,
    /// Basin boundaries
    pub basin_boundaries: Vec<BasinBoundary>,
    /// Basin characteristics
    pub basin_characteristics: Vec<BasinCharacteristics>,
    /// Basin connectivity graph
    pub basin_connectivity: Array2<f64>,
}

/// Basin boundary definition.
#[derive(Debug)]
pub struct BasinBoundary {
    /// Boundary points
    pub boundary_points: Vec<Array1<f64>>,
    /// Barrier height between basins
    pub barrier_height: f64,
    /// Boundary curvature characteristics
    pub curvature: f64,
    /// Transition probability
    pub transition_probability: f64,
}

/// Basin characteristics.
#[derive(Debug)]
pub struct BasinCharacteristics {
    /// Basin center (local minimum)
    pub center: Array1<f64>,
    /// Basin depth (function value at center)
    pub depth: f64,
    /// Basin width estimate
    pub width: f64,
    /// Basin volume estimate
    pub volume: f64,
    /// Hessian eigenvalues at center
    pub hessian_eigenvalues: Array1<f64>,
    /// Basin shape descriptor
    pub shape_descriptor: BasinShape,
}

/// Basin shape descriptors.
#[derive(Debug, Clone)]
pub enum BasinShape {
    /// Spherical basin
    Spherical { radius: f64 },
    /// Elliptical basin
    Elliptical { axes: Array1<f64> },
    /// Irregular basin
    Irregular { convexity: f64 },
    /// Narrow valley
    Valley { width: f64, length: f64 },
}

impl Default for BasinCharacteristics {
    fn default() -> Self {
        Self {
            center: Array1::zeros(1),
            depth: 0.0,
            width: 1.0,
            volume: 1.0,
            hessian_eigenvalues: Array1::zeros(1),
            shape_descriptor: BasinShape::Spherical { radius: 1.0 },
        }
    }
}

/// Hopping effectiveness metrics.
#[derive(Debug)]
pub struct HoppingEffectiveness {
    /// Success rate of hops (accepted moves)
    pub hop_success_rate: f64,
    /// Average distance covered per hop
    pub avg_hop_distance: f64,
    /// Basin escape rate (successful escapes from local minima)
    pub basin_escape_rate: f64,
    /// Overall efficiency score
    pub efficiency_score: f64,
    /// Exploration vs exploitation balance
    pub exploration_exploitation_balance: f64,
    /// Convergence velocity
    pub convergence_velocity: f64,
}

/// Landscape characterization results.
#[derive(Debug)]
pub struct LandscapeCharacteristics {
    /// Ruggedness measure
    pub ruggedness: f64,
    /// Multimodality degree
    pub multimodality: f64,
    /// Deceptiveness measure
    pub deceptiveness: f64,
    /// Neutrality measure
    pub neutrality: f64,
    /// Epistasis measure
    pub epistasis: f64,
}

/// Performance diagnostics for basin hopping.
#[derive(Debug)]
pub struct PerformanceDiagnostics {
    /// Total function evaluations
    pub total_evaluations: u64,
    /// Successful hops
    pub successful_hops: u64,
    /// Failed hops
    pub failed_hops: u64,
    /// Local minima found
    pub local_minima_found: u64,
    /// Average time per hop
    pub avg_time_per_hop: Duration,
    /// Memory usage statistics
    pub memory_usage: MemoryUsage,
}

/// Memory usage statistics.
#[derive(Debug)]
pub struct MemoryUsage {
    /// Peak memory usage in bytes
    pub peak_usage: usize,
    /// Average memory usage in bytes
    pub avg_usage: usize,
    /// Memory efficiency score
    pub efficiency: f64,
}

/// Solution representation for optimization problems.
#[derive(Debug, Clone)]
pub struct Solution {
    /// Solution vector
    pub variables: Array1<f64>,
    /// Objective function value
    pub objective_value: f64,
    /// Constraint violation (if applicable)
    pub constraint_violation: f64,
    /// Gradient at solution (if available)
    pub gradient: Option<Array1<f64>>,
    /// Hessian at solution (if available)
    pub hessian: Option<Array2<f64>>,
    /// Solution metadata
    pub metadata: SolutionMetadata,
}

/// Metadata for optimization solutions.
#[derive(Debug, Clone)]
pub struct SolutionMetadata {
    /// Number of function evaluations to reach this solution
    pub function_evaluations: u64,
    /// Time to find this solution
    pub computation_time: Duration,
    /// Convergence status
    pub convergence_status: ConvergenceStatus,
    /// Local optimality certificate
    pub optimality_certificate: Option<OptimalityCertificate>,
}

/// Convergence status indicators.
#[derive(Debug, Clone)]
pub enum ConvergenceStatus {
    /// Converged to specified tolerance
    Converged,
    /// Maximum iterations reached
    MaxIterations,
    /// Maximum function evaluations reached
    MaxEvaluations,
    /// Stagnation detected
    Stagnation,
    /// User termination
    UserTermination,
    /// Error occurred
    Error(String),
}

/// Optimality certificate for solutions.
#[derive(Debug, Clone)]
pub struct OptimalityCertificate {
    /// KKT conditions satisfied
    pub kkt_satisfied: bool,
    /// Gradient norm
    pub gradient_norm: f64,
    /// Hessian positive definiteness
    pub hessian_positive_definite: bool,
    /// Constraint satisfaction
    pub constraints_satisfied: bool,
}

/// Optimization problem definition.
#[derive(Debug)]
pub struct OptimizationProblem {
    /// Objective function
    pub objective: Box<dyn Fn(&Array1<f64>) -> f64 + Send + Sync>,
    /// Gradient function (optional)
    pub gradient: Option<Box<dyn Fn(&Array1<f64>) -> Array1<f64> + Send + Sync>>,
    /// Hessian function (optional)
    pub hessian: Option<Box<dyn Fn(&Array1<f64>) -> Array2<f64> + Send + Sync>>,
    /// Lower bounds
    pub lower_bounds: Option<Array1<f64>>,
    /// Upper bounds
    pub upper_bounds: Option<Array1<f64>>,
    /// Constraint functions
    pub constraints: Vec<Box<dyn Fn(&Array1<f64>) -> f64 + Send + Sync>>,
    /// Problem metadata
    pub metadata: ProblemMetadata,
}

/// Metadata for optimization problems.
#[derive(Debug)]
pub struct ProblemMetadata {
    /// Problem name
    pub name: String,
    /// Problem dimension
    pub dimension: usize,
    /// Problem type
    pub problem_type: ProblemType,
    /// Expected number of local minima
    pub expected_local_minima: Option<usize>,
    /// Known global minimum (for testing)
    pub known_global_minimum: Option<f64>,
}

/// Types of optimization problems.
#[derive(Debug, Clone)]
pub enum ProblemType {
    /// Unconstrained optimization
    Unconstrained,
    /// Box-constrained optimization
    BoxConstrained,
    /// Nonlinearly constrained optimization
    NonlinearConstrained,
    /// Integer programming
    IntegerProgramming,
    /// Mixed-integer programming
    MixedInteger,
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hopping_parameters_default() {
        let params = HoppingParameters::default();
        assert!(params.step_size > 0.0);
        assert!(params.temperature > 0.0);
        assert!(params.cooling_rate > 0.0 && params.cooling_rate < 1.0);
        assert!(params.min_temperature > 0.0);
        assert!(params.max_hop_attempts > 0);
    }

    #[test]
    fn test_hopping_parameters_builder() {
        let params = HoppingParameters::builder()
            .step_size(1.0)
            .temperature(2.0)
            .cooling_rate(0.9)
            .build()
            .unwrap();

        assert_eq!(params.step_size, 1.0);
        assert_eq!(params.temperature, 2.0);
        assert_eq!(params.cooling_rate, 0.9);
    }

    #[test]
    fn test_hopping_parameters_validation() {
        // Invalid step size
        let result = HoppingParameters::builder()
            .step_size(-1.0)
            .build();
        assert!(result.is_err());

        // Invalid temperature
        let result = HoppingParameters::builder()
            .temperature(0.0)
            .build();
        assert!(result.is_err());

        // Invalid cooling rate
        let result = HoppingParameters::builder()
            .cooling_rate(1.5)
            .build();
        assert!(result.is_err());
    }

    #[test]
    fn test_perturbation_strategy_default() {
        let strategy = PerturbationStrategy::default();
        match strategy {
            PerturbationStrategy::GaussianRandom { sigma } => {
                assert!(sigma > 0.0);
            }
            _ => panic!("Expected GaussianRandom as default"),
        }
    }

    #[test]
    fn test_local_minimizer_config_default() {
        let config = LocalMinimizerConfig::default();
        assert!(config.max_iterations > 0);
        assert!(config.tolerance > 0.0);
        matches!(config.method, LocalSearchMethod::BFGS);
    }

    #[test]
    fn test_basin_characteristics_default() {
        let characteristics = BasinCharacteristics::default();
        assert_eq!(characteristics.center.len(), 1);
        assert_eq!(characteristics.depth, 0.0);
        assert!(characteristics.width > 0.0);
        assert!(characteristics.volume > 0.0);
    }

    #[test]
    fn test_solution_creation() {
        let variables = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let metadata = SolutionMetadata {
            function_evaluations: 100,
            computation_time: Duration::from_millis(50),
            convergence_status: ConvergenceStatus::Converged,
            optimality_certificate: None,
        };

        let solution = Solution {
            variables: variables.clone(),
            objective_value: 10.0,
            constraint_violation: 0.0,
            gradient: None,
            hessian: None,
            metadata,
        };

        assert_eq!(solution.variables, variables);
        assert_eq!(solution.objective_value, 10.0);
        assert_eq!(solution.constraint_violation, 0.0);
    }
}