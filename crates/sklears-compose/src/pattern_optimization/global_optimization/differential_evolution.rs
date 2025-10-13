//! Differential Evolution algorithms for global optimization.
//!
//! This module implements Differential Evolution (DE), a powerful evolutionary algorithm
//! for global optimization of real-valued functions. DE is particularly effective for
//! continuous optimization problems and is known for its simplicity and robustness.
//!
//! # Algorithm Overview
//!
//! Differential Evolution works by:
//! 1. Initializing a population of candidate solutions
//! 2. For each generation:
//!    - Creating mutant vectors through differential mutation
//!    - Performing crossover to create trial vectors
//!    - Selecting better solutions for the next generation
//! 3. Repeating until convergence or maximum generations
//!
//! # Key Features
//!
//! - **Multiple mutation strategies**: DE/rand/1, DE/best/1, DE/current-to-best/1, etc.
//! - **Adaptive parameters**: Self-adaptive F and CR parameters
//! - **Boundary handling**: Various strategies for constraint handling
//! - **Population analysis**: Comprehensive diversity and convergence metrics
//! - **Parallel execution**: Support for parallel fitness evaluation
//! - **Advanced variants**: jDE, SHADE, SUCCESS-History based adaptation
//!
//! # Examples
//!
//! ```rust
//! use sklears_compose::pattern_optimization::global_optimization::differential_evolution::*;
//! use scirs2_core::ndarray::Array1;
//!
//! // Create differential evolution parameters
//! let params = DifferentialEvolutionParameters::builder()
//!     .population_size(100)
//!     .mutation_strategy(MutationStrategy::Rand1)
//!     .crossover_rate(0.7)
//!     .scaling_factor(0.5)
//!     .max_generations(1000)
//!     .build();
//!
//! // The DE algorithm would be implemented by specific optimizers
//! // that use these parameters and follow the DifferentialEvolution trait
//! ```
//!
//! ## Mutation Strategies
//!
//! Various mutation strategies are available:
//! - **DE/rand/1**: Random selection with one difference vector
//! - **DE/best/1**: Best individual with one difference vector
//! - **DE/current-to-best/1**: Current-to-best with one difference vector
//! - **DE/rand/2**: Random selection with two difference vectors
//! - **DE/best/2**: Best individual with two difference vectors
//!
//! ## Boundary Handling
//!
//! Different strategies for handling bound constraints:
//! - **Reflection**: Reflect violated components back into bounds
//! - **Clipping**: Clip violated components to bound values
//! - **Wrapping**: Wrap violated components around bounds
//! - **Resampling**: Regenerate violated components randomly

use std::collections::HashMap;
use std::time::Duration;

// SciRS2 imports following compliance requirements
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::error::CoreError;
use scirs2_core::random::{Random, rng};

// Local imports
use sklears_core::error::Result as SklResult;
use crate::enhanced_errors::PipelineError;
use super::basin_hopping::{Solution, OptimizationProblem, ConvergenceStatus};

/// Differential Evolution algorithm for global optimization.
///
/// Implements various DE strategies for evolutionary optimization with
/// adaptive parameters and advanced population management techniques.
pub trait DifferentialEvolution: Send + Sync {
    /// Initialize the population for differential evolution.
    ///
    /// Creates an initial population of candidate solutions within the
    /// problem bounds using various initialization strategies.
    ///
    /// # Arguments
    /// * `problem` - Optimization problem definition
    /// * `population_size` - Size of the population
    ///
    /// # Returns
    /// Initial population of solutions
    fn initialize_population(&self, problem: &OptimizationProblem, population_size: usize) -> SklResult<Vec<Array1<f64>>>;

    /// Perform mutation operation to create mutant vectors.
    ///
    /// Applies the specified mutation strategy to create mutant vectors
    /// from the current population using differential information.
    ///
    /// # Arguments
    /// * `population` - Current population
    /// * `target_index` - Index of target vector
    /// * `best_index` - Index of best individual
    /// * `scaling_factor` - Scaling factor F
    ///
    /// # Returns
    /// Mutant vector
    fn mutate(&self, population: &[Array1<f64>], target_index: usize, best_index: usize, scaling_factor: f64) -> SklResult<Array1<f64>>;

    /// Perform crossover operation between target and mutant vectors.
    ///
    /// Creates trial vectors by combining target and mutant vectors
    /// according to the crossover strategy and rate.
    ///
    /// # Arguments
    /// * `target` - Target vector
    /// * `mutant` - Mutant vector
    /// * `crossover_rate` - Crossover rate CR
    ///
    /// # Returns
    /// Trial vector
    fn crossover(&self, target: &Array1<f64>, mutant: &Array1<f64>, crossover_rate: f64) -> SklResult<Array1<f64>>;

    /// Perform selection between target and trial vectors.
    ///
    /// Compares target and trial vectors and selects the better one
    /// for the next generation based on fitness values.
    ///
    /// # Arguments
    /// * `target` - Target vector with fitness
    /// * `trial` - Trial vector with fitness
    /// * `problem` - Optimization problem
    ///
    /// # Returns
    /// Selected vector for next generation
    fn select(&self, target: &Solution, trial: &Solution, problem: &OptimizationProblem) -> SklResult<Solution>;

    /// Handle boundary constraint violations.
    ///
    /// Applies boundary handling strategies when vectors exceed
    /// the problem bounds during mutation or crossover.
    ///
    /// # Arguments
    /// * `vector` - Vector that may violate bounds
    /// * `bounds` - Problem bounds
    /// * `strategy` - Boundary handling strategy
    ///
    /// # Returns
    /// Corrected vector within bounds
    fn handle_boundaries(&self, vector: &Array1<f64>, bounds: &(Array1<f64>, Array1<f64>), strategy: BoundaryHandling) -> SklResult<Array1<f64>>;

    /// Adapt parameters based on population dynamics.
    ///
    /// Implements adaptive parameter control for scaling factor and
    /// crossover rate based on population diversity and convergence.
    ///
    /// # Arguments
    /// * `population` - Current population
    /// * `generation` - Current generation number
    /// * `success_history` - History of successful parameter values
    ///
    /// # Returns
    /// Adapted parameters
    fn adapt_parameters(&self, population: &[Solution], generation: usize, success_history: &ParameterHistory) -> SklResult<AdaptedParameters>;

    /// Analyze population diversity and convergence.
    ///
    /// Computes various diversity metrics and convergence indicators
    /// to monitor the optimization progress and population health.
    ///
    /// # Arguments
    /// * `population` - Current population
    /// * `generation` - Current generation number
    ///
    /// # Returns
    /// Population analysis results
    fn analyze_population(&self, population: &[Solution], generation: usize) -> SklResult<PopulationAnalysis>;

    /// Get differential evolution statistics.
    ///
    /// Provides comprehensive statistics about the DE optimization
    /// including convergence behavior and parameter adaptation.
    ///
    /// # Returns
    /// DE optimization statistics
    fn get_statistics(&self) -> SklResult<DifferentialEvolutionStatistics>;
}

/// Configuration parameters for Differential Evolution.
#[derive(Debug, Clone)]
pub struct DifferentialEvolutionParameters {
    /// Population size
    pub population_size: usize,
    /// Maximum number of generations
    pub max_generations: usize,
    /// Mutation strategy to use
    pub mutation_strategy: MutationStrategy,
    /// Crossover strategy
    pub crossover_strategy: CrossoverStrategy,
    /// Initial scaling factor F
    pub scaling_factor: f64,
    /// Initial crossover rate CR
    pub crossover_rate: f64,
    /// Boundary handling strategy
    pub boundary_handling: BoundaryHandling,
    /// Whether to use adaptive parameters
    pub adaptive_parameters: bool,
    /// Convergence tolerance
    pub convergence_tolerance: f64,
    /// Stagnation detection parameters
    pub stagnation_detection: StagnationDetection,
    /// Population initialization strategy
    pub initialization_strategy: InitializationStrategy,
    /// Whether to enable parallel evaluation
    pub parallel_evaluation: bool,
}

impl Default for DifferentialEvolutionParameters {
    fn default() -> Self {
        Self {
            population_size: 100,
            max_generations: 1000,
            mutation_strategy: MutationStrategy::Rand1,
            crossover_strategy: CrossoverStrategy::Binomial,
            scaling_factor: 0.5,
            crossover_rate: 0.7,
            boundary_handling: BoundaryHandling::Reflection,
            adaptive_parameters: true,
            convergence_tolerance: 1e-6,
            stagnation_detection: StagnationDetection::default(),
            initialization_strategy: InitializationStrategy::Random,
            parallel_evaluation: true,
        }
    }
}

impl DifferentialEvolutionParameters {
    /// Create a new builder for DifferentialEvolutionParameters.
    pub fn builder() -> DifferentialEvolutionParametersBuilder {
        DifferentialEvolutionParametersBuilder::new()
    }

    /// Validate the DE parameters.
    pub fn validate(&self) -> SklResult<()> {
        if self.population_size < 4 {
            return Err(CoreError::InvalidInput("Population size must be at least 4".to_string()).into());
        }
        if self.max_generations == 0 {
            return Err(CoreError::InvalidInput("Maximum generations must be positive".to_string()).into());
        }
        if self.scaling_factor <= 0.0 || self.scaling_factor > 2.0 {
            return Err(CoreError::InvalidInput("Scaling factor must be in (0, 2]".to_string()).into());
        }
        if self.crossover_rate < 0.0 || self.crossover_rate > 1.0 {
            return Err(CoreError::InvalidInput("Crossover rate must be in [0, 1]".to_string()).into());
        }
        if self.convergence_tolerance <= 0.0 {
            return Err(CoreError::InvalidInput("Convergence tolerance must be positive".to_string()).into());
        }
        Ok(())
    }
}

/// Builder for DifferentialEvolutionParameters.
#[derive(Debug)]
pub struct DifferentialEvolutionParametersBuilder {
    population_size: usize,
    max_generations: usize,
    mutation_strategy: MutationStrategy,
    crossover_strategy: CrossoverStrategy,
    scaling_factor: f64,
    crossover_rate: f64,
    boundary_handling: BoundaryHandling,
    adaptive_parameters: bool,
    convergence_tolerance: f64,
    stagnation_detection: StagnationDetection,
    initialization_strategy: InitializationStrategy,
    parallel_evaluation: bool,
}

impl DifferentialEvolutionParametersBuilder {
    pub fn new() -> Self {
        Self {
            population_size: 100,
            max_generations: 1000,
            mutation_strategy: MutationStrategy::Rand1,
            crossover_strategy: CrossoverStrategy::Binomial,
            scaling_factor: 0.5,
            crossover_rate: 0.7,
            boundary_handling: BoundaryHandling::Reflection,
            adaptive_parameters: true,
            convergence_tolerance: 1e-6,
            stagnation_detection: StagnationDetection::default(),
            initialization_strategy: InitializationStrategy::Random,
            parallel_evaluation: true,
        }
    }

    /// Set the population size.
    pub fn population_size(mut self, size: usize) -> Self {
        self.population_size = size;
        self
    }

    /// Set the maximum number of generations.
    pub fn max_generations(mut self, generations: usize) -> Self {
        self.max_generations = generations;
        self
    }

    /// Set the mutation strategy.
    pub fn mutation_strategy(mut self, strategy: MutationStrategy) -> Self {
        self.mutation_strategy = strategy;
        self
    }

    /// Set the crossover strategy.
    pub fn crossover_strategy(mut self, strategy: CrossoverStrategy) -> Self {
        self.crossover_strategy = strategy;
        self
    }

    /// Set the scaling factor.
    pub fn scaling_factor(mut self, factor: f64) -> Self {
        self.scaling_factor = factor;
        self
    }

    /// Set the crossover rate.
    pub fn crossover_rate(mut self, rate: f64) -> Self {
        self.crossover_rate = rate;
        self
    }

    /// Set the boundary handling strategy.
    pub fn boundary_handling(mut self, handling: BoundaryHandling) -> Self {
        self.boundary_handling = handling;
        self
    }

    /// Enable or disable adaptive parameters.
    pub fn adaptive_parameters(mut self, adaptive: bool) -> Self {
        self.adaptive_parameters = adaptive;
        self
    }

    /// Set the convergence tolerance.
    pub fn convergence_tolerance(mut self, tolerance: f64) -> Self {
        self.convergence_tolerance = tolerance;
        self
    }

    /// Set the stagnation detection parameters.
    pub fn stagnation_detection(mut self, detection: StagnationDetection) -> Self {
        self.stagnation_detection = detection;
        self
    }

    /// Set the initialization strategy.
    pub fn initialization_strategy(mut self, strategy: InitializationStrategy) -> Self {
        self.initialization_strategy = strategy;
        self
    }

    /// Enable or disable parallel evaluation.
    pub fn parallel_evaluation(mut self, parallel: bool) -> Self {
        self.parallel_evaluation = parallel;
        self
    }

    /// Build the DifferentialEvolutionParameters.
    pub fn build(self) -> SklResult<DifferentialEvolutionParameters> {
        let params = DifferentialEvolutionParameters {
            population_size: self.population_size,
            max_generations: self.max_generations,
            mutation_strategy: self.mutation_strategy,
            crossover_strategy: self.crossover_strategy,
            scaling_factor: self.scaling_factor,
            crossover_rate: self.crossover_rate,
            boundary_handling: self.boundary_handling,
            adaptive_parameters: self.adaptive_parameters,
            convergence_tolerance: self.convergence_tolerance,
            stagnation_detection: self.stagnation_detection,
            initialization_strategy: self.initialization_strategy,
            parallel_evaluation: self.parallel_evaluation,
        };
        params.validate()?;
        Ok(params)
    }
}

/// Mutation strategies for Differential Evolution.
#[derive(Debug, Clone)]
pub enum MutationStrategy {
    /// DE/rand/1: vi = xr1 + F * (xr2 - xr3)
    Rand1,
    /// DE/rand/2: vi = xr1 + F * (xr2 - xr3) + F * (xr4 - xr5)
    Rand2,
    /// DE/best/1: vi = xbest + F * (xr1 - xr2)
    Best1,
    /// DE/best/2: vi = xbest + F * (xr1 - xr2) + F * (xr3 - xr4)
    Best2,
    /// DE/current-to-best/1: vi = xi + F * (xbest - xi) + F * (xr1 - xr2)
    CurrentToBest1,
    /// DE/current-to-rand/1: vi = xi + F * (xr1 - xi) + F * (xr2 - xr3)
    CurrentToRand1,
    /// Adaptive mutation strategy selection
    Adaptive,
}

/// Crossover strategies for Differential Evolution.
#[derive(Debug, Clone)]
pub enum CrossoverStrategy {
    /// Binomial crossover
    Binomial,
    /// Exponential crossover
    Exponential,
    /// Arithmetic crossover
    Arithmetic,
}

/// Boundary handling strategies for constraint violations.
#[derive(Debug, Clone)]
pub enum BoundaryHandling {
    /// Reflect violated components back into bounds
    Reflection,
    /// Clip violated components to bound values
    Clipping,
    /// Wrap violated components around bounds (periodic)
    Wrapping,
    /// Regenerate violated components randomly
    Resampling,
    /// Penalize violations in fitness function
    Penalty { penalty_factor: f64 },
}

/// Population initialization strategies.
#[derive(Debug, Clone)]
pub enum InitializationStrategy {
    /// Uniform random initialization
    Random,
    /// Latin hypercube sampling
    LatinHypercube,
    /// Sobol sequence initialization
    SobolSequence,
    /// Opposition-based initialization
    OppositionBased,
}

/// Stagnation detection parameters.
#[derive(Debug, Clone)]
pub struct StagnationDetection {
    /// Window size for stagnation detection
    pub window_size: usize,
    /// Improvement threshold for stagnation
    pub improvement_threshold: f64,
    /// Enable stagnation detection
    pub enabled: bool,
}

impl Default for StagnationDetection {
    fn default() -> Self {
        Self {
            window_size: 20,
            improvement_threshold: 1e-10,
            enabled: true,
        }
    }
}

/// Adapted parameters from adaptive control.
#[derive(Debug, Clone)]
pub struct AdaptedParameters {
    /// Adapted scaling factor
    pub scaling_factor: f64,
    /// Adapted crossover rate
    pub crossover_rate: f64,
    /// Adapted mutation strategy
    pub mutation_strategy: Option<MutationStrategy>,
    /// Confidence in adaptation
    pub adaptation_confidence: f64,
}

/// History of successful parameter values for adaptation.
#[derive(Debug)]
pub struct ParameterHistory {
    /// Successful scaling factors
    pub successful_f_values: Vec<f64>,
    /// Successful crossover rates
    pub successful_cr_values: Vec<f64>,
    /// History window size
    pub window_size: usize,
    /// Success weights
    pub success_weights: Vec<f64>,
}

impl ParameterHistory {
    pub fn new(window_size: usize) -> Self {
        Self {
            successful_f_values: Vec::new(),
            successful_cr_values: Vec::new(),
            window_size,
            success_weights: Vec::new(),
        }
    }

    /// Add successful parameter values.
    pub fn add_success(&mut self, f: f64, cr: f64, weight: f64) {
        self.successful_f_values.push(f);
        self.successful_cr_values.push(cr);
        self.success_weights.push(weight);

        // Maintain window size
        if self.successful_f_values.len() > self.window_size {
            self.successful_f_values.remove(0);
            self.successful_cr_values.remove(0);
            self.success_weights.remove(0);
        }
    }

    /// Compute weighted mean of successful parameters.
    pub fn compute_weighted_mean(&self) -> (f64, f64) {
        if self.successful_f_values.is_empty() {
            return (0.5, 0.7); // Default values
        }

        let total_weight: f64 = self.success_weights.iter().sum();
        if total_weight == 0.0 {
            return (0.5, 0.7);
        }

        let weighted_f: f64 = self.successful_f_values.iter()
            .zip(&self.success_weights)
            .map(|(f, w)| f * w)
            .sum::<f64>() / total_weight;

        let weighted_cr: f64 = self.successful_cr_values.iter()
            .zip(&self.success_weights)
            .map(|(cr, w)| cr * w)
            .sum::<f64>() / total_weight;

        (weighted_f, weighted_cr)
    }
}

/// Population analysis results.
#[derive(Debug)]
pub struct PopulationAnalysis {
    /// Population diversity metrics
    pub diversity_metrics: DiversityMetrics,
    /// Convergence indicators
    pub convergence_indicators: ConvergenceIndicators,
    /// Population statistics
    pub population_statistics: PopulationStatistics,
    /// Performance metrics
    pub performance_metrics: PerformanceMetrics,
}

/// Population diversity metrics.
#[derive(Debug)]
pub struct DiversityMetrics {
    /// Average pairwise distance between individuals
    pub average_distance: f64,
    /// Standard deviation of fitness values
    pub fitness_standard_deviation: f64,
    /// Population diameter (maximum distance)
    pub population_diameter: f64,
    /// Diversity entropy measure
    pub diversity_entropy: f64,
}

/// Convergence indicators for DE.
#[derive(Debug)]
pub struct ConvergenceIndicators {
    /// Best fitness improvement rate
    pub improvement_rate: f64,
    /// Convergence velocity
    pub convergence_velocity: f64,
    /// Stagnation indicator
    pub is_stagnating: bool,
    /// Generations since improvement
    pub generations_since_improvement: usize,
}

/// Population statistics.
#[derive(Debug)]
pub struct PopulationStatistics {
    /// Best fitness value
    pub best_fitness: f64,
    /// Worst fitness value
    pub worst_fitness: f64,
    /// Average fitness value
    pub average_fitness: f64,
    /// Median fitness value
    pub median_fitness: f64,
    /// Fitness variance
    pub fitness_variance: f64,
}

/// Performance metrics for DE optimization.
#[derive(Debug)]
pub struct PerformanceMetrics {
    /// Function evaluations per generation
    pub evaluations_per_generation: f64,
    /// Success rate of mutations
    pub mutation_success_rate: f64,
    /// Success rate of crossovers
    pub crossover_success_rate: f64,
    /// Parameter adaptation efficiency
    pub adaptation_efficiency: f64,
}

/// Comprehensive DE optimization statistics.
#[derive(Debug)]
pub struct DifferentialEvolutionStatistics {
    /// Total generations executed
    pub total_generations: usize,
    /// Total function evaluations
    pub total_evaluations: u64,
    /// Best solution found
    pub best_solution: Solution,
    /// Convergence history
    pub convergence_history: Vec<f64>,
    /// Parameter adaptation history
    pub parameter_history: ParameterAdaptationHistory,
    /// Final population analysis
    pub final_population_analysis: PopulationAnalysis,
    /// Optimization time
    pub optimization_time: Duration,
}

/// History of parameter adaptations during optimization.
#[derive(Debug)]
pub struct ParameterAdaptationHistory {
    /// History of scaling factor values
    pub f_values: Vec<f64>,
    /// History of crossover rate values
    pub cr_values: Vec<f64>,
    /// History of mutation strategy selections
    pub mutation_strategies: Vec<String>,
    /// Success rates for each parameter setting
    pub success_rates: Vec<f64>,
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_de_parameters_default() {
        let params = DifferentialEvolutionParameters::default();
        assert!(params.population_size >= 4);
        assert!(params.max_generations > 0);
        assert!(params.scaling_factor > 0.0 && params.scaling_factor <= 2.0);
        assert!(params.crossover_rate >= 0.0 && params.crossover_rate <= 1.0);
        assert!(params.convergence_tolerance > 0.0);
    }

    #[test]
    fn test_de_parameters_builder() {
        let params = DifferentialEvolutionParameters::builder()
            .population_size(50)
            .max_generations(500)
            .scaling_factor(0.8)
            .crossover_rate(0.9)
            .build()
            .unwrap();

        assert_eq!(params.population_size, 50);
        assert_eq!(params.max_generations, 500);
        assert_eq!(params.scaling_factor, 0.8);
        assert_eq!(params.crossover_rate, 0.9);
    }

    #[test]
    fn test_de_parameters_validation() {
        // Invalid population size
        let result = DifferentialEvolutionParameters::builder()
            .population_size(3)
            .build();
        assert!(result.is_err());

        // Invalid scaling factor
        let result = DifferentialEvolutionParameters::builder()
            .scaling_factor(2.5)
            .build();
        assert!(result.is_err());

        // Invalid crossover rate
        let result = DifferentialEvolutionParameters::builder()
            .crossover_rate(1.5)
            .build();
        assert!(result.is_err());

        // Invalid convergence tolerance
        let result = DifferentialEvolutionParameters::builder()
            .convergence_tolerance(0.0)
            .build();
        assert!(result.is_err());
    }

    #[test]
    fn test_mutation_strategies() {
        let strategies = vec![
            MutationStrategy::Rand1,
            MutationStrategy::Rand2,
            MutationStrategy::Best1,
            MutationStrategy::Best2,
            MutationStrategy::CurrentToBest1,
            MutationStrategy::CurrentToRand1,
            MutationStrategy::Adaptive,
        ];

        for strategy in strategies {
            let _cloned = strategy.clone();
        }
    }

    #[test]
    fn test_crossover_strategies() {
        let strategies = vec![
            CrossoverStrategy::Binomial,
            CrossoverStrategy::Exponential,
            CrossoverStrategy::Arithmetic,
        ];

        for strategy in strategies {
            let _cloned = strategy.clone();
        }
    }

    #[test]
    fn test_boundary_handling() {
        let strategies = vec![
            BoundaryHandling::Reflection,
            BoundaryHandling::Clipping,
            BoundaryHandling::Wrapping,
            BoundaryHandling::Resampling,
            BoundaryHandling::Penalty { penalty_factor: 1.0 },
        ];

        for strategy in strategies {
            let _cloned = strategy.clone();
        }
    }

    #[test]
    fn test_stagnation_detection_default() {
        let detection = StagnationDetection::default();
        assert!(detection.window_size > 0);
        assert!(detection.improvement_threshold > 0.0);
        assert!(detection.enabled);
    }

    #[test]
    fn test_parameter_history() {
        let mut history = ParameterHistory::new(5);

        // Add some successful parameters
        history.add_success(0.5, 0.7, 1.0);
        history.add_success(0.6, 0.8, 1.5);
        history.add_success(0.4, 0.6, 0.8);

        let (mean_f, mean_cr) = history.compute_weighted_mean();
        assert!(mean_f > 0.0 && mean_f < 1.0);
        assert!(mean_cr > 0.0 && mean_cr < 1.0);

        // Test window size maintenance
        for i in 0..10 {
            history.add_success(0.5, 0.7, 1.0);
        }
        assert!(history.successful_f_values.len() <= 5);
    }

    #[test]
    fn test_adapted_parameters() {
        let adapted = AdaptedParameters {
            scaling_factor: 0.6,
            crossover_rate: 0.8,
            mutation_strategy: Some(MutationStrategy::Best1),
            adaptation_confidence: 0.95,
        };

        assert!(adapted.scaling_factor > 0.0);
        assert!(adapted.crossover_rate >= 0.0 && adapted.crossover_rate <= 1.0);
        assert!(adapted.adaptation_confidence >= 0.0 && adapted.adaptation_confidence <= 1.0);
    }

    #[test]
    fn test_initialization_strategies() {
        let strategies = vec![
            InitializationStrategy::Random,
            InitializationStrategy::LatinHypercube,
            InitializationStrategy::SobolSequence,
            InitializationStrategy::OppositionBased,
        ];

        for strategy in strategies {
            let _cloned = strategy.clone();
        }
    }
}