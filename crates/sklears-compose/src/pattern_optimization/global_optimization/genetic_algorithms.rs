//! Genetic algorithms for global optimization.
//!
//! This module implements genetic algorithms (GA), a class of evolutionary algorithms
//! inspired by natural selection and genetics. GAs are particularly effective for
//! complex optimization problems with large search spaces and multiple local optima.
//!
//! # Algorithm Overview
//!
//! Genetic algorithms work by:
//! 1. Initializing a population of candidate solutions (chromosomes)
//! 2. Evaluating fitness of each individual
//! 3. Selecting parents based on fitness
//! 4. Creating offspring through crossover and mutation
//! 5. Replacing old population with new generation
//! 6. Repeating until convergence or maximum generations
//!
//! # Key Features
//!
//! - **Multiple selection strategies**: Tournament, roulette wheel, rank-based selection
//! - **Various crossover operators**: One-point, two-point, uniform, arithmetic crossover
//! - **Mutation operators**: Gaussian, uniform, polynomial mutation
//! - **Adaptive parameters**: Self-adaptive crossover and mutation rates
//! - **Elitism**: Preservation of best individuals across generations
//! - **Population diversity**: Niching and speciation techniques
//!
//! # Examples
//!
//! ```rust
//! use sklears_compose::pattern_optimization::global_optimization::genetic_algorithms::*;
//! use scirs2_core::ndarray::Array1;
//!
//! // Create genetic algorithm parameters
//! let params = GeneticAlgorithmParameters::builder()
//!     .population_size(100)
//!     .selection_strategy(SelectionStrategy::Tournament { size: 3 })
//!     .crossover_strategy(CrossoverStrategy::TwoPoint)
//!     .mutation_strategy(MutationStrategy::Gaussian { sigma: 0.1 })
//!     .crossover_rate(0.8)
//!     .mutation_rate(0.1)
//!     .max_generations(1000)
//!     .build();
//!
//! // The GA algorithm would be implemented by specific optimizers
//! // that use these parameters and follow the GeneticAlgorithm trait
//! ```

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

/// Genetic algorithm for global optimization.
///
/// Implements various GA strategies with adaptive parameters and
/// advanced population management techniques.
pub trait GeneticAlgorithm: Send + Sync {
    /// Initialize the population for genetic algorithm.
    ///
    /// Creates an initial population of candidate solutions within the
    /// problem bounds using various initialization strategies.
    ///
    /// # Arguments
    /// * `problem` - Optimization problem definition
    /// * `population_size` - Size of the population
    ///
    /// # Returns
    /// Initial population of chromosomes
    fn initialize_population(&self, problem: &OptimizationProblem, population_size: usize) -> SklResult<Vec<Chromosome>>;

    /// Evaluate fitness of the population.
    ///
    /// Computes fitness values for all individuals in the population
    /// using the objective function and constraint handling.
    ///
    /// # Arguments
    /// * `population` - Population to evaluate
    /// * `problem` - Optimization problem
    ///
    /// # Returns
    /// Population with updated fitness values
    fn evaluate_fitness(&self, population: &mut [Chromosome], problem: &OptimizationProblem) -> SklResult<()>;

    /// Select parents for reproduction.
    ///
    /// Applies selection strategy to choose parents for creating
    /// the next generation based on fitness values.
    ///
    /// # Arguments
    /// * `population` - Current population
    /// * `num_parents` - Number of parents to select
    ///
    /// # Returns
    /// Selected parent chromosomes
    fn select_parents(&self, population: &[Chromosome], num_parents: usize) -> SklResult<Vec<Chromosome>>;

    /// Perform crossover between parent chromosomes.
    ///
    /// Creates offspring by combining genetic material from two
    /// parent chromosomes using the specified crossover strategy.
    ///
    /// # Arguments
    /// * `parent1` - First parent chromosome
    /// * `parent2` - Second parent chromosome
    ///
    /// # Returns
    /// Offspring chromosomes
    fn crossover(&self, parent1: &Chromosome, parent2: &Chromosome) -> SklResult<(Chromosome, Chromosome)>;

    /// Apply mutation to chromosomes.
    ///
    /// Introduces random variations to maintain population diversity
    /// and enable exploration of new regions in the search space.
    ///
    /// # Arguments
    /// * `chromosome` - Chromosome to mutate
    /// * `mutation_rate` - Probability of mutation
    ///
    /// # Returns
    /// Mutated chromosome
    fn mutate(&self, chromosome: &Chromosome, mutation_rate: f64) -> SklResult<Chromosome>;

    /// Replace old population with new generation.
    ///
    /// Implements replacement strategy to determine which individuals
    /// survive to the next generation, often including elitism.
    ///
    /// # Arguments
    /// * `old_population` - Current population
    /// * `offspring` - New offspring from reproduction
    ///
    /// # Returns
    /// New population for next generation
    fn replace_population(&self, old_population: &[Chromosome], offspring: &[Chromosome]) -> SklResult<Vec<Chromosome>>;

    /// Analyze population diversity and convergence.
    ///
    /// Computes diversity metrics and convergence indicators to
    /// monitor optimization progress and population health.
    ///
    /// # Arguments
    /// * `population` - Current population
    /// * `generation` - Current generation number
    ///
    /// # Returns
    /// Population analysis results
    fn analyze_population(&self, population: &[Chromosome], generation: usize) -> SklResult<GeneticAnalysis>;

    /// Adapt genetic algorithm parameters.
    ///
    /// Implements adaptive parameter control for crossover rate,
    /// mutation rate, and other GA parameters based on evolution progress.
    ///
    /// # Arguments
    /// * `population` - Current population
    /// * `generation` - Current generation number
    ///
    /// # Returns
    /// Adapted parameters
    fn adapt_parameters(&self, population: &[Chromosome], generation: usize) -> SklResult<AdaptedGAParameters>;
}

/// Chromosome representation for genetic algorithms.
#[derive(Debug, Clone)]
pub struct Chromosome {
    /// Genetic material (variable values)
    pub genes: Array1<f64>,
    /// Fitness value
    pub fitness: f64,
    /// Age of the chromosome (generations survived)
    pub age: usize,
    /// Parent information for lineage tracking
    pub parent_info: Option<ParentInfo>,
    /// Additional metadata
    pub metadata: ChromosomeMetadata,
}

/// Parent information for lineage tracking.
#[derive(Debug, Clone)]
pub struct ParentInfo {
    /// Indices of parent chromosomes
    pub parent_indices: Vec<usize>,
    /// Crossover point(s) used
    pub crossover_points: Vec<usize>,
    /// Mutation information
    pub mutation_applied: bool,
}

/// Metadata for chromosomes.
#[derive(Debug, Clone)]
pub struct ChromosomeMetadata {
    /// Generation when chromosome was created
    pub generation_created: usize,
    /// Number of mutations applied
    pub mutation_count: usize,
    /// Genetic diversity contribution
    pub diversity_contribution: f64,
}

/// Configuration parameters for Genetic Algorithms.
#[derive(Debug, Clone)]
pub struct GeneticAlgorithmParameters {
    /// Population size
    pub population_size: usize,
    /// Maximum number of generations
    pub max_generations: usize,
    /// Selection strategy
    pub selection_strategy: SelectionStrategy,
    /// Crossover strategy
    pub crossover_strategy: CrossoverStrategy,
    /// Mutation strategy
    pub mutation_strategy: MutationStrategy,
    /// Crossover rate
    pub crossover_rate: f64,
    /// Mutation rate
    pub mutation_rate: f64,
    /// Elitism rate (fraction of best individuals to preserve)
    pub elitism_rate: f64,
    /// Replacement strategy
    pub replacement_strategy: ReplacementStrategy,
    /// Convergence criteria
    pub convergence_criteria: ConvergenceCriteria,
    /// Population initialization strategy
    pub initialization_strategy: PopulationInitialization,
    /// Whether to enable adaptive parameters
    pub adaptive_parameters: bool,
}

impl Default for GeneticAlgorithmParameters {
    fn default() -> Self {
        Self {
            population_size: 100,
            max_generations: 1000,
            selection_strategy: SelectionStrategy::Tournament { size: 3 },
            crossover_strategy: CrossoverStrategy::TwoPoint,
            mutation_strategy: MutationStrategy::Gaussian { sigma: 0.1 },
            crossover_rate: 0.8,
            mutation_rate: 0.1,
            elitism_rate: 0.1,
            replacement_strategy: ReplacementStrategy::Generational,
            convergence_criteria: ConvergenceCriteria::default(),
            initialization_strategy: PopulationInitialization::Random,
            adaptive_parameters: true,
        }
    }
}

impl GeneticAlgorithmParameters {
    /// Create a new builder for GeneticAlgorithmParameters.
    pub fn builder() -> GeneticAlgorithmParametersBuilder {
        GeneticAlgorithmParametersBuilder::new()
    }

    /// Validate the GA parameters.
    pub fn validate(&self) -> SklResult<()> {
        if self.population_size < 2 {
            return Err(CoreError::InvalidInput("Population size must be at least 2".to_string()).into());
        }
        if self.max_generations == 0 {
            return Err(CoreError::InvalidInput("Maximum generations must be positive".to_string()).into());
        }
        if self.crossover_rate < 0.0 || self.crossover_rate > 1.0 {
            return Err(CoreError::InvalidInput("Crossover rate must be in [0, 1]".to_string()).into());
        }
        if self.mutation_rate < 0.0 || self.mutation_rate > 1.0 {
            return Err(CoreError::InvalidInput("Mutation rate must be in [0, 1]".to_string()).into());
        }
        if self.elitism_rate < 0.0 || self.elitism_rate > 1.0 {
            return Err(CoreError::InvalidInput("Elitism rate must be in [0, 1]".to_string()).into());
        }
        Ok(())
    }
}

/// Builder for GeneticAlgorithmParameters.
#[derive(Debug)]
pub struct GeneticAlgorithmParametersBuilder {
    population_size: usize,
    max_generations: usize,
    selection_strategy: SelectionStrategy,
    crossover_strategy: CrossoverStrategy,
    mutation_strategy: MutationStrategy,
    crossover_rate: f64,
    mutation_rate: f64,
    elitism_rate: f64,
    replacement_strategy: ReplacementStrategy,
    convergence_criteria: ConvergenceCriteria,
    initialization_strategy: PopulationInitialization,
    adaptive_parameters: bool,
}

impl GeneticAlgorithmParametersBuilder {
    pub fn new() -> Self {
        Self {
            population_size: 100,
            max_generations: 1000,
            selection_strategy: SelectionStrategy::Tournament { size: 3 },
            crossover_strategy: CrossoverStrategy::TwoPoint,
            mutation_strategy: MutationStrategy::Gaussian { sigma: 0.1 },
            crossover_rate: 0.8,
            mutation_rate: 0.1,
            elitism_rate: 0.1,
            replacement_strategy: ReplacementStrategy::Generational,
            convergence_criteria: ConvergenceCriteria::default(),
            initialization_strategy: PopulationInitialization::Random,
            adaptive_parameters: true,
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

    /// Set the selection strategy.
    pub fn selection_strategy(mut self, strategy: SelectionStrategy) -> Self {
        self.selection_strategy = strategy;
        self
    }

    /// Set the crossover strategy.
    pub fn crossover_strategy(mut self, strategy: CrossoverStrategy) -> Self {
        self.crossover_strategy = strategy;
        self
    }

    /// Set the mutation strategy.
    pub fn mutation_strategy(mut self, strategy: MutationStrategy) -> Self {
        self.mutation_strategy = strategy;
        self
    }

    /// Set the crossover rate.
    pub fn crossover_rate(mut self, rate: f64) -> Self {
        self.crossover_rate = rate;
        self
    }

    /// Set the mutation rate.
    pub fn mutation_rate(mut self, rate: f64) -> Self {
        self.mutation_rate = rate;
        self
    }

    /// Set the elitism rate.
    pub fn elitism_rate(mut self, rate: f64) -> Self {
        self.elitism_rate = rate;
        self
    }

    /// Set the replacement strategy.
    pub fn replacement_strategy(mut self, strategy: ReplacementStrategy) -> Self {
        self.replacement_strategy = strategy;
        self
    }

    /// Set the convergence criteria.
    pub fn convergence_criteria(mut self, criteria: ConvergenceCriteria) -> Self {
        self.convergence_criteria = criteria;
        self
    }

    /// Set the initialization strategy.
    pub fn initialization_strategy(mut self, strategy: PopulationInitialization) -> Self {
        self.initialization_strategy = strategy;
        self
    }

    /// Enable or disable adaptive parameters.
    pub fn adaptive_parameters(mut self, adaptive: bool) -> Self {
        self.adaptive_parameters = adaptive;
        self
    }

    /// Build the GeneticAlgorithmParameters.
    pub fn build(self) -> SklResult<GeneticAlgorithmParameters> {
        let params = GeneticAlgorithmParameters {
            population_size: self.population_size,
            max_generations: self.max_generations,
            selection_strategy: self.selection_strategy,
            crossover_strategy: self.crossover_strategy,
            mutation_strategy: self.mutation_strategy,
            crossover_rate: self.crossover_rate,
            mutation_rate: self.mutation_rate,
            elitism_rate: self.elitism_rate,
            replacement_strategy: self.replacement_strategy,
            convergence_criteria: self.convergence_criteria,
            initialization_strategy: self.initialization_strategy,
            adaptive_parameters: self.adaptive_parameters,
        };
        params.validate()?;
        Ok(params)
    }
}

/// Selection strategies for genetic algorithms.
#[derive(Debug, Clone)]
pub enum SelectionStrategy {
    /// Tournament selection with specified tournament size
    Tournament { size: usize },
    /// Roulette wheel selection (fitness proportionate)
    RouletteWheel,
    /// Rank-based selection
    RankBased,
    /// Stochastic universal sampling
    StochasticUniversalSampling,
    /// Linear ranking selection
    LinearRanking { selective_pressure: f64 },
    /// Truncation selection
    Truncation { threshold: f64 },
}

/// Crossover strategies for genetic algorithms.
#[derive(Debug, Clone)]
pub enum CrossoverStrategy {
    /// Single-point crossover
    OnePoint,
    /// Two-point crossover
    TwoPoint,
    /// Uniform crossover
    Uniform { probability: f64 },
    /// Arithmetic crossover
    Arithmetic { alpha: f64 },
    /// Simulated binary crossover (SBX)
    SimulatedBinary { eta: f64 },
    /// Blend crossover (BLX-α)
    Blend { alpha: f64 },
}

/// Mutation strategies for genetic algorithms.
#[derive(Debug, Clone)]
pub enum MutationStrategy {
    /// Gaussian mutation with specified standard deviation
    Gaussian { sigma: f64 },
    /// Uniform mutation within bounds
    Uniform { range: f64 },
    /// Polynomial mutation
    Polynomial { eta: f64 },
    /// Non-uniform mutation
    NonUniform { b: f64 },
    /// Cauchy mutation
    Cauchy { scale: f64 },
}

/// Replacement strategies for population management.
#[derive(Debug, Clone)]
pub enum ReplacementStrategy {
    /// Generational replacement (replace entire population)
    Generational,
    /// Steady-state replacement (replace few individuals)
    SteadyState { num_replacements: usize },
    /// Elitist replacement with specified elite size
    Elitist { elite_size: usize },
    /// Tournament replacement
    TournamentReplacement { tournament_size: usize },
}

/// Population initialization strategies.
#[derive(Debug, Clone)]
pub enum PopulationInitialization {
    /// Random initialization within bounds
    Random,
    /// Latin hypercube sampling
    LatinHypercube,
    /// Sobol sequence initialization
    SobolSequence,
    /// Opposition-based initialization
    OppositionBased,
}

/// Convergence criteria for genetic algorithms.
#[derive(Debug, Clone)]
pub struct ConvergenceCriteria {
    /// Fitness tolerance for convergence
    pub fitness_tolerance: f64,
    /// Maximum generations without improvement
    pub max_stagnant_generations: usize,
    /// Target fitness value
    pub target_fitness: Option<f64>,
    /// Population diversity threshold
    pub diversity_threshold: f64,
}

impl Default for ConvergenceCriteria {
    fn default() -> Self {
        Self {
            fitness_tolerance: 1e-6,
            max_stagnant_generations: 50,
            target_fitness: None,
            diversity_threshold: 1e-8,
        }
    }
}

/// Adapted parameters from adaptive control.
#[derive(Debug, Clone)]
pub struct AdaptedGAParameters {
    /// Adapted crossover rate
    pub crossover_rate: f64,
    /// Adapted mutation rate
    pub mutation_rate: f64,
    /// Adapted selection pressure
    pub selection_pressure: f64,
    /// Confidence in adaptation
    pub adaptation_confidence: f64,
}

/// Genetic algorithm analysis results.
#[derive(Debug)]
pub struct GeneticAnalysis {
    /// Population diversity metrics
    pub diversity_metrics: GADiversityMetrics,
    /// Convergence indicators
    pub convergence_indicators: GAConvergenceIndicators,
    /// Selection pressure analysis
    pub selection_pressure: SelectionPressureAnalysis,
    /// Genetic operator effectiveness
    pub operator_effectiveness: OperatorEffectiveness,
}

/// Diversity metrics for genetic algorithms.
#[derive(Debug)]
pub struct GADiversityMetrics {
    /// Genetic diversity (average pairwise distance)
    pub genetic_diversity: f64,
    /// Phenotypic diversity (fitness variance)
    pub phenotypic_diversity: f64,
    /// Population entropy
    pub population_entropy: f64,
    /// Diversity loss rate
    pub diversity_loss_rate: f64,
}

/// Convergence indicators for GA.
#[derive(Debug)]
pub struct GAConvergenceIndicators {
    /// Best fitness improvement rate
    pub improvement_rate: f64,
    /// Average fitness trend
    pub average_fitness_trend: f64,
    /// Population convergence measure
    pub population_convergence: f64,
    /// Premature convergence indicator
    pub premature_convergence_risk: f64,
}

/// Selection pressure analysis.
#[derive(Debug)]
pub struct SelectionPressureAnalysis {
    /// Current selection pressure
    pub current_pressure: f64,
    /// Optimal pressure estimate
    pub optimal_pressure: f64,
    /// Pressure balance (exploration vs exploitation)
    pub pressure_balance: f64,
}

/// Effectiveness of genetic operators.
#[derive(Debug)]
pub struct OperatorEffectiveness {
    /// Crossover success rate
    pub crossover_success_rate: f64,
    /// Mutation success rate
    pub mutation_success_rate: f64,
    /// Selection efficiency
    pub selection_efficiency: f64,
    /// Overall operator balance
    pub operator_balance: f64,
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ga_parameters_default() {
        let params = GeneticAlgorithmParameters::default();
        assert!(params.population_size >= 2);
        assert!(params.max_generations > 0);
        assert!(params.crossover_rate >= 0.0 && params.crossover_rate <= 1.0);
        assert!(params.mutation_rate >= 0.0 && params.mutation_rate <= 1.0);
        assert!(params.elitism_rate >= 0.0 && params.elitism_rate <= 1.0);
    }

    #[test]
    fn test_ga_parameters_builder() {
        let params = GeneticAlgorithmParameters::builder()
            .population_size(50)
            .max_generations(500)
            .crossover_rate(0.9)
            .mutation_rate(0.05)
            .elitism_rate(0.2)
            .build()
            .unwrap();

        assert_eq!(params.population_size, 50);
        assert_eq!(params.max_generations, 500);
        assert_eq!(params.crossover_rate, 0.9);
        assert_eq!(params.mutation_rate, 0.05);
        assert_eq!(params.elitism_rate, 0.2);
    }

    #[test]
    fn test_ga_parameters_validation() {
        // Invalid population size
        let result = GeneticAlgorithmParameters::builder()
            .population_size(1)
            .build();
        assert!(result.is_err());

        // Invalid crossover rate
        let result = GeneticAlgorithmParameters::builder()
            .crossover_rate(1.5)
            .build();
        assert!(result.is_err());

        // Invalid mutation rate
        let result = GeneticAlgorithmParameters::builder()
            .mutation_rate(-0.1)
            .build();
        assert!(result.is_err());

        // Invalid elitism rate
        let result = GeneticAlgorithmParameters::builder()
            .elitism_rate(1.1)
            .build();
        assert!(result.is_err());
    }

    #[test]
    fn test_selection_strategies() {
        let strategies = vec![
            SelectionStrategy::Tournament { size: 3 },
            SelectionStrategy::RouletteWheel,
            SelectionStrategy::RankBased,
            SelectionStrategy::StochasticUniversalSampling,
            SelectionStrategy::LinearRanking { selective_pressure: 2.0 },
            SelectionStrategy::Truncation { threshold: 0.5 },
        ];

        for strategy in strategies {
            let _cloned = strategy.clone();
        }
    }

    #[test]
    fn test_crossover_strategies() {
        let strategies = vec![
            CrossoverStrategy::OnePoint,
            CrossoverStrategy::TwoPoint,
            CrossoverStrategy::Uniform { probability: 0.5 },
            CrossoverStrategy::Arithmetic { alpha: 0.5 },
            CrossoverStrategy::SimulatedBinary { eta: 20.0 },
            CrossoverStrategy::Blend { alpha: 0.5 },
        ];

        for strategy in strategies {
            let _cloned = strategy.clone();
        }
    }

    #[test]
    fn test_mutation_strategies() {
        let strategies = vec![
            MutationStrategy::Gaussian { sigma: 0.1 },
            MutationStrategy::Uniform { range: 0.1 },
            MutationStrategy::Polynomial { eta: 20.0 },
            MutationStrategy::NonUniform { b: 5.0 },
            MutationStrategy::Cauchy { scale: 0.1 },
        ];

        for strategy in strategies {
            let _cloned = strategy.clone();
        }
    }

    #[test]
    fn test_replacement_strategies() {
        let strategies = vec![
            ReplacementStrategy::Generational,
            ReplacementStrategy::SteadyState { num_replacements: 10 },
            ReplacementStrategy::Elitist { elite_size: 5 },
            ReplacementStrategy::TournamentReplacement { tournament_size: 3 },
        ];

        for strategy in strategies {
            let _cloned = strategy.clone();
        }
    }

    #[test]
    fn test_chromosome_creation() {
        let genes = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let metadata = ChromosomeMetadata {
            generation_created: 0,
            mutation_count: 0,
            diversity_contribution: 0.5,
        };

        let chromosome = Chromosome {
            genes: genes.clone(),
            fitness: 10.0,
            age: 0,
            parent_info: None,
            metadata,
        };

        assert_eq!(chromosome.genes, genes);
        assert_eq!(chromosome.fitness, 10.0);
        assert_eq!(chromosome.age, 0);
    }

    #[test]
    fn test_convergence_criteria_default() {
        let criteria = ConvergenceCriteria::default();
        assert!(criteria.fitness_tolerance > 0.0);
        assert!(criteria.max_stagnant_generations > 0);
        assert!(criteria.diversity_threshold > 0.0);
    }

    #[test]
    fn test_population_initialization() {
        let strategies = vec![
            PopulationInitialization::Random,
            PopulationInitialization::LatinHypercube,
            PopulationInitialization::SobolSequence,
            PopulationInitialization::OppositionBased,
        ];

        for strategy in strategies {
            let _cloned = strategy.clone();
        }
    }
}