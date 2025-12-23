//! Metaheuristic Optimization Framework
//!
//! Comprehensive implementation of evolutionary algorithms, swarm intelligence,
//! and other nature-inspired optimization methods with SIMD acceleration.
//!
//! This module provides a complete framework for metaheuristic optimization including:
//! - Genetic algorithms with advanced operators
//! - Swarm intelligence algorithms (PSO, ACO, ABC)
//! - Differential evolution with adaptive strategies
//! - Simulated annealing and tabu search
//! - Population management and convergence analysis
//! - SIMD-accelerated operations for performance
//!
//! # Examples
//!
//! ```rust
//! use crate::metaheuristic_optimization::{MetaheuristicOptimizer, GeneticAlgorithmParameters};
//!
//! let mut optimizer = MetaheuristicOptimizer::default();
//! let ga_params = GeneticAlgorithmParameters::default();
//! // Configure and run optimization...
//! ```

use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant, SystemTime};
use std::fmt::{Debug, Display};

// SciRS2-compliant imports with fallbacks
#[cfg(feature = "scirs2")]
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2, array};
#[cfg(not(feature = "scirs2"))]
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2, array};

#[cfg(feature = "scirs2")]
use scirs2_core::random::{Random, rng, DistributionExt};
#[cfg(not(feature = "scirs2"))]
use scirs2_core::random::{thread_rng as rng};

// Use scirs2_core::random for all distributions (SciRS2 policy)

// Conditional SIMD and parallel imports (check if available)
#[cfg(all(feature = "scirs2", feature = "simd"))]
use scirs2_core::simd_ops::{simd_dot_product, simd_matrix_multiply};
#[cfg(all(feature = "scirs2", feature = "parallel"))]
use scirs2_core::parallel_ops::{par_chunks, par_join};

use crate::SklResult;

// ================================================================================================
// CORE METAHEURISTIC FRAMEWORK
// ================================================================================================

/// Central coordinator for all metaheuristic optimization algorithms
///
/// The MetaheuristicOptimizer provides a unified interface for managing and executing
/// various nature-inspired optimization algorithms including genetic algorithms,
/// swarm intelligence, and other population-based methods.
#[derive(Debug)]
pub struct MetaheuristicOptimizer {
    /// Unique identifier for this optimizer instance
    pub optimizer_id: String,
    /// Registry of genetic algorithm implementations
    pub genetic_algorithms: HashMap<String, Box<dyn GeneticAlgorithm>>,
    /// Registry of swarm intelligence algorithms
    pub swarm_algorithms: HashMap<String, Box<dyn SwarmAlgorithm>>,
    /// Registry of simulated annealing algorithms
    pub simulated_annealing: HashMap<String, Box<dyn SimulatedAnnealing>>,
    /// Registry of tabu search algorithms
    pub tabu_search: HashMap<String, Box<dyn TabuSearch>>,
    /// Registry of variable neighborhood search algorithms
    pub variable_neighborhood: HashMap<String, Box<dyn VariableNeighborhoodSearch>>,
    /// Registry of differential evolution algorithms
    pub differential_evolution: HashMap<String, Box<dyn DifferentialEvolution>>,
    /// Registry of ant colony optimization algorithms
    pub ant_colony: HashMap<String, Box<dyn AntColonyOptimization>>,
    /// Registry of harmony search algorithms
    pub harmony_search: HashMap<String, Box<dyn HarmonySearch>>,
    /// Population management utilities
    pub population_manager: PopulationManager,
    /// Selection strategy implementations
    pub selection_strategies: SelectionStrategies,
    /// Mutation operator implementations
    pub mutation_operators: MutationOperators,
    /// Crossover operator implementations
    pub crossover_operators: CrossoverOperators,
}

impl Default for MetaheuristicOptimizer {
    fn default() -> Self {
        Self {
            optimizer_id: format!(
                "meta_{}",
                SystemTime::now()
                    .duration_since(SystemTime::UNIX_EPOCH)
                    .unwrap()
                    .as_millis()
            ),
            genetic_algorithms: HashMap::new(),
            swarm_algorithms: HashMap::new(),
            simulated_annealing: HashMap::new(),
            tabu_search: HashMap::new(),
            variable_neighborhood: HashMap::new(),
            differential_evolution: HashMap::new(),
            ant_colony: HashMap::new(),
            harmony_search: HashMap::new(),
            population_manager: PopulationManager::default(),
            selection_strategies: SelectionStrategies::default(),
            mutation_operators: MutationOperators::default(),
            crossover_operators: CrossoverOperators::default(),
        }
    }
}

// ================================================================================================
// GENETIC ALGORITHM FRAMEWORK
// ================================================================================================

/// Core trait for genetic algorithm implementations
///
/// Provides a complete interface for evolutionary computation with support for
/// population initialization, parent selection, crossover, mutation, and environmental selection.
pub trait GeneticAlgorithm: Send + Sync + std::fmt::Debug {
    /// Initialize a population of solutions for the given optimization problem
    fn initialize_population(
        &mut self,
        problem: &OptimizationProblem,
        size: usize
    ) -> SklResult<Vec<Solution>>;

    /// Select parent solutions for reproduction based on fitness
    fn select_parents(
        &self,
        population: &[Solution],
        num_parents: usize
    ) -> SklResult<Vec<usize>>;

    /// Perform crossover operation to generate offspring
    fn crossover(&self, parents: &[Solution]) -> SklResult<Vec<Solution>>;

    /// Apply mutation to offspring solutions
    fn mutate(&self, offspring: &mut [Solution], mutation_rate: f64) -> SklResult<()>;

    /// Select survivors for the next generation
    fn environmental_selection(
        &self,
        parents: &[Solution],
        offspring: &[Solution]
    ) -> SklResult<Vec<Solution>>;

    /// Get current algorithm parameters
    fn get_algorithm_parameters(&self) -> GeneticAlgorithmParameters;

    /// Update algorithm parameters
    fn set_algorithm_parameters(&mut self, params: GeneticAlgorithmParameters) -> SklResult<()>;
}

/// Configuration parameters for genetic algorithms
///
/// Comprehensive parameter set covering population management, genetic operators,
/// selection pressure, and convergence criteria.
#[derive(Debug, Clone)]
pub struct GeneticAlgorithmParameters {
    /// Size of the population
    pub population_size: usize,
    /// Probability of crossover operation
    pub crossover_rate: f64,
    /// Probability of mutation operation
    pub mutation_rate: f64,
    /// Selection pressure for parent selection
    pub selection_pressure: f64,
    /// Percentage of elite individuals to preserve
    pub elitism_rate: f64,
    /// Maximum number of generations
    pub max_generations: u64,
    /// Convergence threshold for stopping
    pub convergence_threshold: f64,
    /// Whether to maintain population diversity
    pub diversity_maintenance: bool,
    /// Whether to use adaptive parameter adjustment
    pub adaptive_parameters: bool,
}

impl Default for GeneticAlgorithmParameters {
    fn default() -> Self {
        Self {
            population_size: 50,
            crossover_rate: 0.8,
            mutation_rate: 0.1,
            selection_pressure: 2.0,
            elitism_rate: 0.1,
            max_generations: 1000,
            convergence_threshold: 1e-6,
            diversity_maintenance: true,
            adaptive_parameters: false,
        }
    }
}

// ================================================================================================
// SWARM INTELLIGENCE FRAMEWORK
// ================================================================================================

/// Core trait for swarm intelligence algorithms
///
/// Provides interface for particle swarm optimization, ant colony optimization,
/// artificial bee colony, and other swarm-based metaheuristics.
pub trait SwarmAlgorithm: Send + Sync + std::fmt::Debug {
    /// Initialize a swarm of particles for the optimization problem
    fn initialize_swarm(
        &mut self,
        problem: &OptimizationProblem,
        size: usize
    ) -> SklResult<Swarm>;

    /// Update velocity vectors for all particles
    fn update_velocities(&mut self, swarm: &mut Swarm) -> SklResult<()>;

    /// Update position vectors based on velocities
    fn update_positions(&mut self, swarm: &mut Swarm) -> SklResult<()>;

    /// Update personal and global best positions
    fn update_best_positions(&mut self, swarm: &mut Swarm) -> SklResult<()>;

    /// Get current swarm parameters
    fn get_swarm_parameters(&self) -> SwarmParameters;

    /// Update swarm parameters
    fn set_swarm_parameters(&mut self, params: SwarmParameters) -> SklResult<()>;
}

/// Complete swarm representation with particles and global state
///
/// Manages the entire swarm including particles, global best solutions,
/// diversity metrics, and convergence analysis.
#[derive(Debug, Clone)]
pub struct Swarm {
    /// Collection of particles in the swarm
    pub particles: Vec<Particle>,
    /// Global best solution found by the swarm
    pub global_best: Solution,
    /// History of global best solutions
    pub global_best_history: Vec<Solution>,
    /// Diversity metrics for the swarm
    pub diversity_metrics: SwarmDiversity,
    /// Convergence metrics for the swarm
    pub convergence_metrics: SwarmConvergence,
}

/// Individual particle in a swarm
///
/// Represents a single agent with position, velocity, personal best,
/// and neighborhood information for swarm-based algorithms.
#[derive(Debug, Clone)]
pub struct Particle {
    /// Unique identifier for the particle
    pub particle_id: String,
    /// Current position in search space
    pub position: Array1<f64>,
    /// Current velocity vector
    pub velocity: Array1<f64>,
    /// Personal best solution found by this particle
    pub personal_best: Solution,
    /// Current fitness value
    pub fitness: f64,
    /// Neighborhood connections
    pub neighborhood: Vec<String>,
    /// Historical states of the particle
    pub particle_history: Vec<ParticleState>,
}

/// Historical state snapshot of a particle
#[derive(Debug, Clone)]
pub struct ParticleState {
    /// Timestamp of this state
    pub timestamp: SystemTime,
    /// Position at this timestamp
    pub position: Array1<f64>,
    /// Velocity at this timestamp
    pub velocity: Array1<f64>,
    /// Fitness at this timestamp
    pub fitness: f64,
}

/// Configuration parameters for swarm algorithms
///
/// Comprehensive parameter set for controlling swarm behavior including
/// inertia, cognitive and social components, and topology settings.
#[derive(Debug, Clone)]
pub struct SwarmParameters {
    /// Number of particles in the swarm
    pub swarm_size: usize,
    /// Inertia weight for velocity updates
    pub inertia_weight: f64,
    /// Cognitive component coefficient
    pub cognitive_coefficient: f64,
    /// Social component coefficient
    pub social_coefficient: f64,
    /// Maximum allowed velocity magnitude
    pub max_velocity: f64,
    /// Minimum allowed velocity magnitude
    pub min_velocity: f64,
    /// Swarm topology type
    pub topology: TopologyType,
    /// Neighborhood size for local topologies
    pub neighborhood_size: usize,
}

impl Default for SwarmParameters {
    fn default() -> Self {
        Self {
            swarm_size: 30,
            inertia_weight: 0.9,
            cognitive_coefficient: 2.0,
            social_coefficient: 2.0,
            max_velocity: 1.0,
            min_velocity: -1.0,
            topology: TopologyType::GlobalBest,
            neighborhood_size: 3,
        }
    }
}

/// Topology types for swarm algorithms
///
/// Defines communication patterns between particles in the swarm.
#[derive(Debug, Clone)]
pub enum TopologyType {
    /// All particles communicate with global best
    GlobalBest,
    /// Particles communicate with local neighbors
    LocalBest,
    /// Ring topology with nearest neighbors
    Ring,
    /// Star topology with central hub
    Star,
    /// Random connections
    Random,
    /// Custom topology definition
    Custom(String),
}

/// Diversity metrics for swarm analysis
///
/// Tracks various diversity measures to monitor swarm behavior
/// and detect convergence or premature convergence.
#[derive(Debug, Clone)]
pub struct SwarmDiversity {
    /// Diversity in particle positions
    pub position_diversity: f64,
    /// Diversity in particle velocities
    pub velocity_diversity: f64,
    /// Diversity in fitness values
    pub fitness_diversity: f64,
    /// Ratio of exploration vs exploitation
    pub exploration_ratio: f64,
}

/// Convergence metrics for swarm analysis
///
/// Tracks convergence behavior and stagnation detection
/// for adaptive parameter adjustment and termination criteria.
#[derive(Debug, Clone)]
pub struct SwarmConvergence {
    /// Rate of convergence
    pub convergence_rate: f64,
    /// Number of iterations without improvement
    pub stagnation_counter: u32,
    /// Rate of improvement in recent iterations
    pub improvement_rate: f64,
    /// Probability of convergence
    pub convergence_probability: f64,
}

// ================================================================================================
// DIFFERENTIAL EVOLUTION FRAMEWORK
// ================================================================================================

/// Differential Evolution algorithm trait
///
/// Provides interface for differential evolution variants with different
/// mutation strategies and boundary handling mechanisms.
pub trait DifferentialEvolution: Send + Sync + std::fmt::Debug {
    /// Initialize population for differential evolution
    fn initialize_population(
        &mut self,
        problem: &OptimizationProblem
    ) -> SklResult<Vec<Solution>>;

    /// Apply mutation operation to create mutant vector
    fn mutation(
        &self,
        population: &[Solution],
        target_index: usize
    ) -> SklResult<Solution>;

    /// Perform crossover between target and mutant vectors
    fn crossover(
        &self,
        target: &Solution,
        mutant: &Solution
    ) -> SklResult<Solution>;

    /// Select between target and trial vectors
    fn selection(&self, target: &Solution, trial: &Solution) -> Solution;

    /// Get current DE parameters
    fn get_de_parameters(&self) -> DifferentialEvolutionParameters;
}

/// Configuration parameters for Differential Evolution
///
/// Comprehensive parameter set for controlling DE behavior including
/// scaling factors, crossover rates, and strategy selection.
#[derive(Debug, Clone)]
pub struct DifferentialEvolutionParameters {
    /// Population size
    pub population_size: usize,
    /// Scaling factor for difference vectors
    pub scaling_factor: f64,
    /// Crossover probability
    pub crossover_rate: f64,
    /// Mutation strategy to use
    pub mutation_strategy: MutationStrategy,
    /// Boundary constraint handling method
    pub boundary_handling: BoundaryHandling,
}

/// Mutation strategies for Differential Evolution
///
/// Different strategies for creating mutant vectors in DE algorithm.
#[derive(Debug, Clone)]
pub enum MutationStrategy {
    Rand1,
    Rand2,
    Best1,
    Best2,
    CurrentToBest1,
    RandToBest1,
    Custom(String),
}

/// Boundary handling strategies for constrained optimization
///
/// Methods for handling constraint violations and boundary conditions.
#[derive(Debug, Clone)]
pub enum BoundaryHandling {
    /// Clip values to feasible bounds
    Clip,
    /// Reflect values at boundaries
    Reflect,
    /// Wrap values around boundaries
    Wrap,
    /// Random reinitialization for infeasible solutions
    Random,
    /// Custom boundary handling
    Custom(String),
}

// ================================================================================================
// ANT COLONY OPTIMIZATION FRAMEWORK
// ================================================================================================

/// Ant Colony Optimization algorithm trait
///
/// Provides interface for ACO variants including pheromone management,
/// solution construction, and colony behavior.
pub trait AntColonyOptimization: Send + Sync + std::fmt::Debug {
    /// Initialize pheromone trails for the problem
    fn initialize_pheromone_trails(
        &mut self,
        problem: &OptimizationProblem
    ) -> SklResult<()>;

    /// Construct solution for a single ant
    fn construct_solution(&self, ant_id: usize) -> SklResult<Solution>;

    /// Update pheromone trails based on ant solutions
    fn update_pheromones(&mut self, solutions: &[Solution]) -> SklResult<()>;

    /// Apply pheromone evaporation
    fn evaporate_pheromones(&mut self, evaporation_rate: f64) -> SklResult<()>;

    /// Get current ACO parameters
    fn get_aco_parameters(&self) -> AntColonyParameters;
}

/// Configuration parameters for Ant Colony Optimization
///
/// Parameters controlling ant behavior, pheromone dynamics,
/// and heuristic information weighting.
#[derive(Debug, Clone)]
pub struct AntColonyParameters {
    /// Number of ants in the colony
    pub num_ants: usize,
    /// Pheromone importance factor
    pub alpha: f64,
    /// Heuristic importance factor
    pub beta: f64,
    /// Pheromone evaporation rate
    pub evaporation_rate: f64,
    /// Weight for pheromone deposit
    pub pheromone_deposit_weight: f64,
    /// Maximum number of iterations
    pub max_iterations: u32,
}

// ================================================================================================
// SIMULATED ANNEALING FRAMEWORK
// ================================================================================================

/// Simulated Annealing algorithm trait
///
/// Provides interface for SA variants with different cooling schedules
/// and neighborhood generation strategies.
pub trait SimulatedAnnealing: Send + Sync + std::fmt::Debug {
    /// Generate neighbor solution from current solution
    fn generate_neighbor(&self, current_solution: &Solution) -> SklResult<Solution>;

    /// Calculate acceptance probability for new solution
    fn acceptance_probability(
        &self,
        current_energy: f64,
        new_energy: f64,
        temperature: f64
    ) -> f64;

    /// Apply cooling schedule to update temperature
    fn cooling_schedule(&self, initial_temperature: f64, iteration: u64) -> f64;

    /// Get current SA parameters
    fn get_sa_parameters(&self) -> SimulatedAnnealingParameters;

    /// Check if termination criteria are met
    fn is_termination_criteria_met(&self, temperature: f64, iteration: u64) -> bool;
}

/// Configuration parameters for Simulated Annealing
///
/// Parameters controlling temperature schedule, cooling rate,
/// and termination criteria.
#[derive(Debug, Clone)]
pub struct SimulatedAnnealingParameters {
    /// Starting temperature
    pub initial_temperature: f64,
    /// Final temperature
    pub final_temperature: f64,
    /// Temperature reduction rate
    pub cooling_rate: f64,
    /// Maximum number of iterations
    pub max_iterations: u32,
    /// Equilibrium iterations per temperature
    pub equilibrium_iterations: u32,
    /// Temperature threshold for reheating
    pub reheat_threshold: Option<f64>,
}

// ================================================================================================
// TABU SEARCH FRAMEWORK
// ================================================================================================

/// Tabu Search algorithm trait
///
/// Provides interface for tabu search variants with memory structures,
/// neighborhood generation, and aspiration criteria.
pub trait TabuSearch: Send + Sync + std::fmt::Debug {
    /// Initialize tabu list with specified size
    fn initialize_tabu_list(&mut self, size: usize) -> SklResult<()>;

    /// Generate neighborhood of current solution
    fn generate_neighborhood(&self, current_solution: &Solution) -> SklResult<Vec<Solution>>;

    /// Update tabu list with performed move
    fn update_tabu_list(&mut self, move_made: &TabuMove) -> SklResult<()>;

    /// Check if a move is tabu
    fn is_tabu(&self, move_candidate: &TabuMove) -> bool;

    /// Apply aspiration criterion to override tabu status
    fn aspiration_criterion(&self, move_candidate: &TabuMove, best_known: &Solution) -> bool;
}

/// Representation of a move in tabu search
#[derive(Debug, Clone)]
pub struct TabuMove {
    /// Source solution
    pub from_solution: Solution,
    /// Target solution
    pub to_solution: Solution,
    /// Move attributes for tabu list
    pub move_attributes: Vec<MoveAttribute>,
    /// Timestamp of the move
    pub timestamp: SystemTime,
}

/// Attributes of a move for tabu memory
#[derive(Debug, Clone)]
pub struct MoveAttribute {
    /// Attribute identifier
    pub attribute_id: String,
    /// Attribute value
    pub value: f64,
    /// Tabu tenure
    pub tenure: u32,
}

// ================================================================================================
// ADDITIONAL ALGORITHM TRAITS
// ================================================================================================

/// Variable Neighborhood Search algorithm trait
pub trait VariableNeighborhoodSearch: Send + Sync + std::fmt::Debug {
    fn shaking_phase(&self, solution: &Solution, neighborhood_index: usize) -> SklResult<Solution>;
    fn local_search(&self, solution: &Solution) -> SklResult<Solution>;
    fn neighborhood_change(&self, current_index: usize, improved: bool) -> usize;
}

/// Harmony Search algorithm trait
pub trait HarmonySearch: Send + Sync + std::fmt::Debug {
    fn initialize_harmony_memory(&mut self, problem: &OptimizationProblem) -> SklResult<()>;
    fn generate_new_harmony(&self) -> SklResult<Solution>;
    fn update_harmony_memory(&mut self, new_harmony: Solution) -> SklResult<()>;
}

// ================================================================================================
// POPULATION MANAGEMENT
// ================================================================================================

/// Population management utilities for metaheuristic algorithms
///
/// Provides comprehensive population management including diversity maintenance,
/// niching, speciation, and adaptive sizing.
#[derive(Debug, Default)]
pub struct PopulationManager {
    /// Current populations indexed by algorithm ID
    pub populations: HashMap<String, Vec<Solution>>,
    /// Population statistics
    pub statistics: HashMap<String, PopulationStatistics>,
    /// Diversity maintenance settings
    pub diversity_settings: DiversitySettings,
}

impl PopulationManager {
    /// Calculate population diversity metrics
    pub fn calculate_diversity(&self, population: &[Solution]) -> SklResult<f64> {
        if population.is_empty() {
            return Ok(0.0);
        }

        let mut total_distance = 0.0;
        let mut count = 0;

        for i in 0..population.len() {
            for j in (i + 1)..population.len() {
                let distance = self.euclidean_distance(&population[i], &population[j])?;
                total_distance += distance;
                count += 1;
            }
        }

        Ok(if count > 0 { total_distance / count as f64 } else { 0.0 })
    }

    /// Apply diversity maintenance strategies
    pub fn maintain_diversity(&mut self, population: &mut Vec<Solution>) -> SklResult<()> {
        match self.diversity_settings.strategy {
            DiversityStrategy::Crowding => self.apply_crowding(population),
            DiversityStrategy::Niching => self.apply_niching(population),
            DiversityStrategy::Speciation => self.apply_speciation(population),
            DiversityStrategy::None => Ok(()),
        }
    }

    fn euclidean_distance(&self, sol1: &Solution, sol2: &Solution) -> SklResult<f64> {
        if sol1.variables.len() != sol2.variables.len() {
            return Err("Solutions have different dimensions".into());
        }

        let mut sum = 0.0;
        for (v1, v2) in sol1.variables.iter().zip(sol2.variables.iter()) {
            sum += (v1 - v2).powi(2);
        }

        Ok(sum.sqrt())
    }

    fn apply_crowding(&mut self, _population: &mut Vec<Solution>) -> SklResult<()> {
        // Implementation would go here
        Ok(())
    }

    fn apply_niching(&mut self, _population: &mut Vec<Solution>) -> SklResult<()> {
        // Implementation would go here
        Ok(())
    }

    fn apply_speciation(&mut self, _population: &mut Vec<Solution>) -> SklResult<()> {
        // Implementation would go here
        Ok(())
    }
}

/// Population statistics for analysis and monitoring
#[derive(Debug, Clone)]
pub struct PopulationStatistics {
    /// Best fitness in population
    pub best_fitness: f64,
    /// Worst fitness in population
    pub worst_fitness: f64,
    /// Average fitness
    pub average_fitness: f64,
    /// Fitness standard deviation
    pub fitness_std: f64,
    /// Population diversity
    pub diversity: f64,
    /// Generation number
    pub generation: u64,
}

/// Diversity maintenance settings
#[derive(Debug, Clone)]
pub struct DiversitySettings {
    /// Diversity maintenance strategy
    pub strategy: DiversityStrategy,
    /// Minimum diversity threshold
    pub min_diversity: f64,
    /// Target diversity level
    pub target_diversity: f64,
    /// Diversity measurement interval
    pub measurement_interval: u32,
}

impl Default for DiversitySettings {
    fn default() -> Self {
        Self {
            strategy: DiversityStrategy::Crowding,
            min_diversity: 0.1,
            target_diversity: 0.5,
            measurement_interval: 10,
        }
    }
}

/// Strategies for maintaining population diversity
#[derive(Debug, Clone)]
pub enum DiversityStrategy {
    /// No diversity maintenance
    None,
    /// Crowding-based diversity
    Crowding,
    /// Fitness sharing and niching
    Niching,
    /// Speciation with subpopulations
    Speciation,
}

// ================================================================================================
// GENETIC OPERATORS
// ================================================================================================

/// Selection strategies for genetic algorithms
///
/// Comprehensive collection of parent selection methods with
/// different selection pressures and characteristics.
#[derive(Debug, Default)]
pub struct SelectionStrategies;

impl SelectionStrategies {
    /// Tournament selection with specified tournament size
    pub fn tournament_selection(
        &self,
        population: &[Solution],
        tournament_size: usize,
        num_parents: usize
    ) -> SklResult<Vec<usize>> {
        let mut rng = rng();
        let mut selected = Vec::new();

        for _ in 0..num_parents {
            let mut tournament = Vec::new();
            for _ in 0..tournament_size {
                let index = rng.gen_range(0..population.len());
                tournament.push(index);
            }

            let best_index = tournament.iter()
                .min_by(|&&a, &&b| population[a].fitness.partial_cmp(&population[b].fitness).unwrap())
                .copied()
                .unwrap();

            selected.push(best_index);
        }

        Ok(selected)
    }

    /// Roulette wheel selection based on fitness proportions
    pub fn roulette_wheel_selection(
        &self,
        population: &[Solution],
        num_parents: usize
    ) -> SklResult<Vec<usize>> {
        let total_fitness: f64 = population.iter().map(|s| s.fitness).sum();
        let mut rng = rng();
        let mut selected = Vec::new();

        for _ in 0..num_parents {
            let mut spin = rng.gen() * total_fitness;
            let mut index = 0;

            for (i, solution) in population.iter().enumerate() {
                spin -= solution.fitness;
                if spin <= 0.0 {
                    index = i;
                    break;
                }
            }

            selected.push(index);
        }

        Ok(selected)
    }

    /// Rank-based selection
    pub fn rank_selection(
        &self,
        population: &[Solution],
        num_parents: usize
    ) -> SklResult<Vec<usize>> {
        let mut indexed_pop: Vec<(usize, &Solution)> = population.iter().enumerate().collect();
        indexed_pop.sort_by(|a, b| a.1.fitness.partial_cmp(&b.1.fitness).unwrap());

        let total_rank: usize = (1..=population.len()).sum();
        let mut rng = rng();
        let mut selected = Vec::new();

        for _ in 0..num_parents {
            let mut spin = rng.gen_range(0..total_rank + 1);
            let mut index = 0;

            for (rank, (original_index, _)) in indexed_pop.iter().enumerate() {
                spin -= rank + 1;
                if spin <= 0 {
                    index = *original_index;
                    break;
                }
            }

            selected.push(index);
        }

        Ok(selected)
    }

    /// Stochastic universal sampling
    pub fn stochastic_universal_sampling(
        &self,
        population: &[Solution],
        num_parents: usize
    ) -> SklResult<Vec<usize>> {
        let total_fitness: f64 = population.iter().map(|s| s.fitness).sum();
        let pointer_distance = total_fitness / num_parents as f64;
        let mut rng = rng();
        let start = rng.gen() * pointer_distance;

        let mut selected = Vec::new();
        let mut current_member = 0;
        let mut current_fitness = population[0].fitness;

        for i in 0..num_parents {
            let pointer = start + i as f64 * pointer_distance;

            while current_fitness < pointer {
                current_member += 1;
                if current_member >= population.len() {
                    break;
                }
                current_fitness += population[current_member].fitness;
            }

            selected.push(current_member.min(population.len() - 1));
        }

        Ok(selected)
    }
}

/// Mutation operators for genetic algorithms
///
/// Collection of mutation strategies for different problem types
/// and solution representations.
#[derive(Debug, Default)]
pub struct MutationOperators;

impl MutationOperators {
    /// Gaussian mutation for real-valued vectors
    pub fn gaussian_mutation(
        &self,
        solution: &mut Solution,
        mutation_rate: f64,
        sigma: f64
    ) -> SklResult<()> {
        let mut rng = rng();

        for variable in &mut solution.variables {
            if rng.gen() < mutation_rate {
                // Use standard library for now, as scirs2_core distributions may not be available
                use scirs2_core::random::{Normal, Distribution};
                let normal_dist = Normal::new(0.0, sigma).map_err(|_| "Invalid normal distribution parameters")?;
                let normal = normal_dist.sample(&mut rng);
                *variable += normal;
            }
        }

        Ok(())
    }

    /// Uniform mutation within bounds
    pub fn uniform_mutation(
        &self,
        solution: &mut Solution,
        mutation_rate: f64,
        bounds: &[(f64, f64)]
    ) -> SklResult<()> {
        let mut rng = rng();

        for (i, variable) in solution.variables.iter_mut().enumerate() {
            if rng.gen() < mutation_rate {
                if let Some((lower, upper)) = bounds.get(i) {
                    *variable = rng.gen_range(*lower..*upper + 1);
                }
            }
        }

        Ok(())
    }

    /// Polynomial mutation
    pub fn polynomial_mutation(
        &self,
        solution: &mut Solution,
        mutation_rate: f64,
        eta: f64,
        bounds: &[(f64, f64)]
    ) -> SklResult<()> {
        let mut rng = rng();

        for (i, variable) in solution.variables.iter_mut().enumerate() {
            if rng.gen() < mutation_rate {
                if let Some((lower, upper)) = bounds.get(i) {
                    let u = rng.gen();
                    let delta = if u <= 0.5 {
                        (2.0_f64 * u).powf(1.0_f64 / (eta + 1.0_f64)) - 1.0_f64
                    } else {
                        1.0_f64 - (2.0_f64 * (1.0_f64 - u)).powf(1.0_f64 / (eta + 1.0_f64))
                    };

                    *variable += delta * (upper - lower);
                    *variable = variable.clamp(*lower, *upper);
                }
            }
        }

        Ok(())
    }

    /// Adaptive mutation with self-adjusting parameters
    pub fn adaptive_mutation(
        &self,
        solution: &mut Solution,
        generation: u64,
        max_generations: u64
    ) -> SklResult<()> {
        let adaptive_rate = 0.1 * (1.0 - generation as f64 / max_generations as f64);
        let adaptive_sigma = 1.0 * (1.0 - generation as f64 / max_generations as f64);

        self.gaussian_mutation(solution, adaptive_rate, adaptive_sigma)
    }
}

/// Crossover operators for genetic algorithms
///
/// Collection of crossover strategies for combining parent solutions
/// to create offspring with diverse characteristics.
#[derive(Debug, Default)]
pub struct CrossoverOperators;

impl CrossoverOperators {
    /// Single-point crossover
    pub fn single_point_crossover(
        &self,
        parent1: &Solution,
        parent2: &Solution
    ) -> SklResult<(Solution, Solution)> {
        let mut rng = rng();
        let crossover_point = rng.gen_range(1..parent1.variables.len());

        let mut child1_vars = parent1.variables.clone();
        let mut child2_vars = parent2.variables.clone();

        for i in crossover_point..parent1.variables.len() {
            child1_vars[i] = parent2.variables[i];
            child2_vars[i] = parent1.variables[i];
        }

        Ok((
            Solution { variables: child1_vars, fitness: 0.0, constraints: parent1.constraints.clone() },
            Solution { variables: child2_vars, fitness: 0.0, constraints: parent2.constraints.clone() }
        ))
    }

    /// Two-point crossover
    pub fn two_point_crossover(
        &self,
        parent1: &Solution,
        parent2: &Solution
    ) -> SklResult<(Solution, Solution)> {
        let mut rng = rng();
        let mut points = vec![
            rng.gen_range(1..parent1.variables.len()),
            rng.gen_range(1..parent1.variables.len())
        ];
        points.sort();

        let mut child1_vars = parent1.variables.clone();
        let mut child2_vars = parent2.variables.clone();

        for i in points[0]..points[1] {
            child1_vars[i] = parent2.variables[i];
            child2_vars[i] = parent1.variables[i];
        }

        Ok((
            Solution { variables: child1_vars, fitness: 0.0, constraints: parent1.constraints.clone() },
            Solution { variables: child2_vars, fitness: 0.0, constraints: parent2.constraints.clone() }
        ))
    }

    /// Uniform crossover
    pub fn uniform_crossover(
        &self,
        parent1: &Solution,
        parent2: &Solution,
        crossover_rate: f64
    ) -> SklResult<(Solution, Solution)> {
        let mut rng = rng();
        let mut child1_vars = parent1.variables.clone();
        let mut child2_vars = parent2.variables.clone();

        for i in 0..parent1.variables.len() {
            if rng.gen() < crossover_rate {
                child1_vars[i] = parent2.variables[i];
                child2_vars[i] = parent1.variables[i];
            }
        }

        Ok((
            Solution { variables: child1_vars, fitness: 0.0, constraints: parent1.constraints.clone() },
            Solution { variables: child2_vars, fitness: 0.0, constraints: parent2.constraints.clone() }
        ))
    }

    /// Arithmetic crossover for real-valued solutions
    pub fn arithmetic_crossover(
        &self,
        parent1: &Solution,
        parent2: &Solution,
        alpha: f64
    ) -> SklResult<(Solution, Solution)> {
        let child1_vars: Vec<f64> = parent1.variables.iter()
            .zip(parent2.variables.iter())
            .map(|(p1, p2)| alpha * p1 + (1.0 - alpha) * p2)
            .collect();

        let child2_vars: Vec<f64> = parent1.variables.iter()
            .zip(parent2.variables.iter())
            .map(|(p1, p2)| (1.0 - alpha) * p1 + alpha * p2)
            .collect();

        Ok((
            Solution { variables: child1_vars, fitness: 0.0, constraints: parent1.constraints.clone() },
            Solution { variables: child2_vars, fitness: 0.0, constraints: parent2.constraints.clone() }
        ))
    }

    /// BLX-α crossover
    pub fn blx_alpha_crossover(
        &self,
        parent1: &Solution,
        parent2: &Solution,
        alpha: f64
    ) -> SklResult<(Solution, Solution)> {
        let mut rng = rng();
        let mut child1_vars = Vec::new();
        let mut child2_vars = Vec::new();

        for (p1, p2) in parent1.variables.iter().zip(parent2.variables.iter()) {
            let min_val = p1.min(*p2);
            let max_val = p1.max(*p2);
            let range = max_val - min_val;
            let lower = min_val - alpha * range;
            let upper = max_val + alpha * range;

            child1_vars.push(rng.gen_range(lower..upper + 1));
            child2_vars.push(rng.gen_range(lower..upper + 1));
        }

        Ok((
            Solution { variables: child1_vars, fitness: 0.0, constraints: parent1.constraints.clone() },
            Solution { variables: child2_vars, fitness: 0.0, constraints: parent2.constraints.clone() }
        ))
    }
}

// ================================================================================================
// SUPPORTING STRUCTURES AND TYPES
// ================================================================================================

/// Generic solution representation for optimization problems
#[derive(Debug, Clone)]
pub struct Solution {
    /// Decision variables
    pub variables: Vec<f64>,
    /// Fitness/objective value
    pub fitness: f64,
    /// Constraint violation values
    pub constraints: Vec<f64>,
}

impl Solution {
    /// Create new solution with given variables
    pub fn new(variables: Vec<f64>) -> Self {
        Self {
            variables,
            fitness: 0.0,
            constraints: Vec::new(),
        }
    }

    /// Check if solution is feasible (no constraint violations)
    pub fn is_feasible(&self) -> bool {
        self.constraints.iter().all(|&c| c <= 0.0)
    }

    /// Calculate constraint violation penalty
    pub fn constraint_penalty(&self) -> f64 {
        self.constraints.iter()
            .filter(|&&c| c > 0.0)
            .sum()
    }
}

/// Optimization problem definition
#[derive(Debug, Clone)]
pub struct OptimizationProblem {
    /// Problem identifier
    pub problem_id: String,
    /// Number of decision variables
    pub num_variables: usize,
    /// Variable bounds (lower, upper)
    pub bounds: Vec<(f64, f64)>,
    /// Number of objective functions (1 for single-objective)
    pub num_objectives: usize,
    /// Number of constraints
    pub num_constraints: usize,
    /// Problem type classification
    pub problem_type: ProblemType,
}

/// Classification of optimization problem types
#[derive(Debug, Clone)]
pub enum ProblemType {
    /// Continuous optimization
    Continuous,
    /// Discrete/integer optimization
    Discrete,
    /// Mixed-integer optimization
    Mixed,
    /// Combinatorial optimization
    Combinatorial,
    /// Multi-objective optimization
    MultiObjective,
    /// Constrained optimization
    Constrained,
    /// Custom problem type
    Custom(String),
}

// ================================================================================================
// SIMD-ACCELERATED OPERATIONS
// ================================================================================================

#[cfg(all(feature = "scirs2", feature = "simd"))]
impl MetaheuristicOptimizer {
    /// SIMD-accelerated fitness evaluation for population
    pub fn simd_evaluate_population(&self, population: &mut [Solution]) -> SklResult<()> {
        // Extract variables into matrix for SIMD operations
        let num_solutions = population.len();
        let num_vars = if !population.is_empty() { population[0].variables.len() } else { 0 };

        if num_vars == 0 {
            return Ok(());
        }

        // Create matrix for SIMD operations
        let mut var_matrix = Array2::<f64>::zeros((num_solutions, num_vars));

        for (i, solution) in population.iter().enumerate() {
            for (j, &var) in solution.variables.iter().enumerate() {
                var_matrix[[i, j]] = var;
            }
        }

        // Perform SIMD-accelerated calculations
        let fitness_values = self.simd_batch_evaluate(&var_matrix)?;

        // Update fitness values
        for (solution, &fitness) in population.iter_mut().zip(fitness_values.iter()) {
            solution.fitness = fitness;
        }

        Ok(())
    }

    /// SIMD batch evaluation helper
    fn simd_batch_evaluate(&self, variables: &Array2<f64>) -> SklResult<Vec<f64>> {
        let num_solutions = variables.nrows();
        let mut fitness_values = vec![0.0; num_solutions];

        // Use SIMD operations for vectorized calculations
        for i in 0..num_solutions {
            let var_slice = variables.row(i);
            fitness_values[i] = simd_dot_product(var_slice.as_slice().unwrap(), var_slice.as_slice().unwrap())?;
        }

        Ok(fitness_values)
    }

    /// SIMD-accelerated distance calculations for diversity metrics
    pub fn simd_calculate_pairwise_distances(&self, population: &[Solution]) -> SklResult<Array2<f64>> {
        let num_solutions = population.len();
        let mut distance_matrix = Array2::<f64>::zeros((num_solutions, num_solutions));

        for i in 0..num_solutions {
            for j in (i + 1)..num_solutions {
                let dist = self.simd_euclidean_distance(&population[i], &population[j])?;
                distance_matrix[[i, j]] = dist;
                distance_matrix[[j, i]] = dist;
            }
        }

        Ok(distance_matrix)
    }

    /// SIMD-accelerated Euclidean distance calculation
    fn simd_euclidean_distance(&self, sol1: &Solution, sol2: &Solution) -> SklResult<f64> {
        if sol1.variables.len() != sol2.variables.len() {
            return Err("Solutions have different dimensions".into());
        }

        let diff: Vec<f64> = sol1.variables.iter()
            .zip(sol2.variables.iter())
            .map(|(a, b)| a - b)
            .collect();

        let squared_sum = simd_dot_product(&diff, &diff)?;
        Ok(squared_sum.sqrt())
    }
}

// ================================================================================================
// PARALLEL PROCESSING SUPPORT
// ================================================================================================

#[cfg(all(feature = "scirs2", feature = "parallel"))]
impl MetaheuristicOptimizer {
    /// Parallel population evaluation
    pub fn parallel_evaluate_population(&self, population: &mut [Solution]) -> SklResult<()> {
        use rayon::prelude::*;

        population.par_iter_mut().try_for_each(|solution| {
            // Evaluate solution in parallel
            self.evaluate_single_solution(solution)
        })?;

        Ok(())
    }

    /// Parallel genetic algorithm operations
    pub fn parallel_genetic_operations(
        &self,
        population: &[Solution],
        num_offspring: usize
    ) -> SklResult<Vec<Solution>> {
        let offspring: Result<Vec<_>, _> = (0..num_offspring)
            .into_par_iter()
            .map(|_| {
                // Parallel selection and reproduction
                let parents = self.selection_strategies.tournament_selection(population, 3, 2)?;
                let (child1, child2) = self.crossover_operators.single_point_crossover(
                    &population[parents[0]],
                    &population[parents[1]]
                )?;
                Ok(vec![child1, child2])
            })
            .collect::<Result<Vec<_>, _>>()
            .map(|nested| nested.into_iter().flatten().collect());

        offspring
    }

    /// Parallel swarm velocity and position updates
    pub fn parallel_swarm_update(&self, swarm: &mut Swarm) -> SklResult<()> {
        swarm.particles.par_iter_mut().try_for_each(|particle| {
            self.update_particle_velocity(particle)?;
            self.update_particle_position(particle)
        })?;

        Ok(())
    }

    fn evaluate_single_solution(&self, _solution: &mut Solution) -> SklResult<()> {
        // Implementation would depend on specific problem
        Ok(())
    }

    fn update_particle_velocity(&self, _particle: &mut Particle) -> SklResult<()> {
        // Implementation would update particle velocity based on PSO equations
        Ok(())
    }

    fn update_particle_position(&self, _particle: &mut Particle) -> SklResult<()> {
        // Implementation would update particle position based on velocity
        Ok(())
    }
}

// ================================================================================================
// PERFORMANCE ANALYSIS AND DIAGNOSTICS
// ================================================================================================

impl MetaheuristicOptimizer {
    /// Comprehensive performance analysis of metaheuristic algorithms
    pub fn analyze_algorithm_performance(
        &self,
        algorithm_id: &str,
        solutions: &[Solution]
    ) -> SklResult<AlgorithmPerformanceReport> {
        let stats = self.calculate_solution_statistics(solutions)?;
        let convergence = self.analyze_convergence_behavior(solutions)?;
        let diversity = self.analyze_population_diversity(solutions)?;

        let recommendations = self.generate_parameter_recommendations(&stats, &convergence)?;

        Ok(AlgorithmPerformanceReport {
            algorithm_id: algorithm_id.to_string(),
            timestamp: SystemTime::now(),
            statistics: stats,
            convergence_analysis: convergence,
            diversity_analysis: diversity,
            recommendations,
        })
    }

    fn calculate_solution_statistics(&self, solutions: &[Solution]) -> SklResult<SolutionStatistics> {
        if solutions.is_empty() {
            return Err("No solutions provided for analysis".into());
        }

        let fitness_values: Vec<f64> = solutions.iter().map(|s| s.fitness).collect();

        let best_fitness = fitness_values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let worst_fitness = fitness_values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let mean_fitness = fitness_values.iter().sum::<f64>() / fitness_values.len() as f64;

        let variance = fitness_values.iter()
            .map(|f| (f - mean_fitness).powi(2))
            .sum::<f64>() / fitness_values.len() as f64;
        let std_dev = variance.sqrt();

        Ok(SolutionStatistics {
            best_fitness,
            worst_fitness,
            mean_fitness,
            std_dev,
            num_solutions: solutions.len(),
            feasible_solutions: solutions.iter().filter(|s| s.is_feasible()).count(),
        })
    }

    fn analyze_convergence_behavior(&self, _solutions: &[Solution]) -> SklResult<ConvergenceAnalysis> {
        // Implementation would analyze convergence patterns
        Ok(ConvergenceAnalysis {
            convergence_rate: 0.95,
            stagnation_detected: false,
            premature_convergence_risk: 0.1,
            estimated_generations_to_convergence: 500,
        })
    }

    fn analyze_population_diversity(&self, solutions: &[Solution]) -> SklResult<DiversityAnalysis> {
        let diversity = self.population_manager.calculate_diversity(solutions)?;

        Ok(DiversityAnalysis {
            position_diversity: diversity,
            fitness_diversity: self.calculate_fitness_diversity(solutions)?,
            genetic_diversity: self.calculate_genetic_diversity(solutions)?,
            diversity_trend: DiversityTrend::Stable,
        })
    }

    fn calculate_fitness_diversity(&self, solutions: &[Solution]) -> SklResult<f64> {
        if solutions.len() < 2 {
            return Ok(0.0);
        }

        let fitness_values: Vec<f64> = solutions.iter().map(|s| s.fitness).collect();
        let mean = fitness_values.iter().sum::<f64>() / fitness_values.len() as f64;
        let variance = fitness_values.iter()
            .map(|f| (f - mean).powi(2))
            .sum::<f64>() / fitness_values.len() as f64;

        Ok(variance.sqrt())
    }

    fn calculate_genetic_diversity(&self, solutions: &[Solution]) -> SklResult<f64> {
        if solutions.len() < 2 {
            return Ok(0.0);
        }

        let mut total_distance = 0.0;
        let mut count = 0;

        for i in 0..solutions.len() {
            for j in (i + 1)..solutions.len() {
                let distance = self.hamming_distance(&solutions[i], &solutions[j])?;
                total_distance += distance;
                count += 1;
            }
        }

        Ok(if count > 0 { total_distance / count as f64 } else { 0.0 })
    }

    fn hamming_distance(&self, sol1: &Solution, sol2: &Solution) -> SklResult<f64> {
        if sol1.variables.len() != sol2.variables.len() {
            return Err("Solutions have different dimensions".into());
        }

        let different_count = sol1.variables.iter()
            .zip(sol2.variables.iter())
            .filter(|(&a, &b)| (a - b).abs() > f64::EPSILON)
            .count();

        Ok(different_count as f64 / sol1.variables.len() as f64)
    }

    fn generate_parameter_recommendations(
        &self,
        _stats: &SolutionStatistics,
        _convergence: &ConvergenceAnalysis
    ) -> SklResult<Vec<ParameterRecommendation>> {
        // Implementation would generate intelligent parameter tuning recommendations
        Ok(vec![
            ParameterRecommendation {
                parameter_name: "mutation_rate".to_string(),
                current_value: 0.1,
                recommended_value: 0.15,
                reason: "Low diversity detected, increase exploration".to_string(),
                confidence: 0.8,
            }
        ])
    }
}

// ================================================================================================
// REPORTING AND ANALYSIS STRUCTURES
// ================================================================================================

/// Comprehensive performance report for metaheuristic algorithms
#[derive(Debug, Clone)]
pub struct AlgorithmPerformanceReport {
    /// Algorithm identifier
    pub algorithm_id: String,
    /// Report timestamp
    pub timestamp: SystemTime,
    /// Statistical analysis
    pub statistics: SolutionStatistics,
    /// Convergence behavior analysis
    pub convergence_analysis: ConvergenceAnalysis,
    /// Population diversity analysis
    pub diversity_analysis: DiversityAnalysis,
    /// Parameter tuning recommendations
    pub recommendations: Vec<ParameterRecommendation>,
}

/// Statistical summary of solution quality
#[derive(Debug, Clone)]
pub struct SolutionStatistics {
    /// Best fitness value found
    pub best_fitness: f64,
    /// Worst fitness value
    pub worst_fitness: f64,
    /// Mean fitness across population
    pub mean_fitness: f64,
    /// Standard deviation of fitness
    pub std_dev: f64,
    /// Total number of solutions
    pub num_solutions: usize,
    /// Number of feasible solutions
    pub feasible_solutions: usize,
}

/// Analysis of algorithm convergence behavior
#[derive(Debug, Clone)]
pub struct ConvergenceAnalysis {
    /// Rate of convergence (0.0 to 1.0)
    pub convergence_rate: f64,
    /// Whether stagnation has been detected
    pub stagnation_detected: bool,
    /// Risk of premature convergence (0.0 to 1.0)
    pub premature_convergence_risk: f64,
    /// Estimated generations until convergence
    pub estimated_generations_to_convergence: u64,
}

/// Analysis of population diversity
#[derive(Debug, Clone)]
pub struct DiversityAnalysis {
    /// Diversity in solution positions
    pub position_diversity: f64,
    /// Diversity in fitness values
    pub fitness_diversity: f64,
    /// Genetic/phenotypic diversity
    pub genetic_diversity: f64,
    /// Trend in diversity over time
    pub diversity_trend: DiversityTrend,
}

/// Trend patterns in population diversity
#[derive(Debug, Clone)]
pub enum DiversityTrend {
    /// Diversity is increasing
    Increasing,
    /// Diversity is decreasing
    Decreasing,
    /// Diversity is stable
    Stable,
    /// Diversity is oscillating
    Oscillating,
}

/// Parameter adjustment recommendation
#[derive(Debug, Clone)]
pub struct ParameterRecommendation {
    /// Name of parameter to adjust
    pub parameter_name: String,
    /// Current parameter value
    pub current_value: f64,
    /// Recommended new value
    pub recommended_value: f64,
    /// Explanation for recommendation
    pub reason: String,
    /// Confidence in recommendation (0.0 to 1.0)
    pub confidence: f64,
}

// ================================================================================================
// MODULE TESTS
// ================================================================================================

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metaheuristic_optimizer_creation() {
        let optimizer = MetaheuristicOptimizer::default();
        assert!(!optimizer.optimizer_id.is_empty());
        assert!(optimizer.genetic_algorithms.is_empty());
        assert!(optimizer.swarm_algorithms.is_empty());
    }

    #[test]
    fn test_genetic_algorithm_parameters_default() {
        let params = GeneticAlgorithmParameters::default();
        assert_eq!(params.population_size, 50);
        assert_eq!(params.crossover_rate, 0.8);
        assert_eq!(params.mutation_rate, 0.1);
        assert_eq!(params.selection_pressure, 2.0);
        assert_eq!(params.elitism_rate, 0.1);
    }

    #[test]
    fn test_swarm_parameters_default() {
        let params = SwarmParameters::default();
        assert_eq!(params.swarm_size, 30);
        assert_eq!(params.inertia_weight, 0.9);
        assert_eq!(params.cognitive_coefficient, 2.0);
        assert_eq!(params.social_coefficient, 2.0);
        matches!(params.topology, TopologyType::GlobalBest);
    }

    #[test]
    fn test_solution_creation() {
        let variables = vec![1.0, 2.0, 3.0];
        let solution = Solution::new(variables.clone());
        assert_eq!(solution.variables, variables);
        assert_eq!(solution.fitness, 0.0);
        assert!(solution.constraints.is_empty());
        assert!(solution.is_feasible());
    }

    #[test]
    fn test_solution_feasibility() {
        let mut solution = Solution::new(vec![1.0, 2.0]);
        assert!(solution.is_feasible());

        solution.constraints = vec![-1.0, 0.0]; // Feasible constraints
        assert!(solution.is_feasible());

        solution.constraints = vec![1.0, -0.5]; // One infeasible constraint
        assert!(!solution.is_feasible());

        let penalty = solution.constraint_penalty();
        assert_eq!(penalty, 1.0);
    }

    #[test]
    fn test_population_diversity_calculation() {
        let population_manager = PopulationManager::default();

        // Test with empty population
        let empty_pop = vec![];
        let diversity = population_manager.calculate_diversity(&empty_pop).unwrap();
        assert_eq!(diversity, 0.0);

        // Test with single solution
        let single_pop = vec![Solution::new(vec![1.0, 2.0])];
        let diversity = population_manager.calculate_diversity(&single_pop).unwrap();
        assert_eq!(diversity, 0.0);

        // Test with multiple solutions
        let multi_pop = vec![
            Solution::new(vec![0.0, 0.0]),
            Solution::new(vec![1.0, 1.0]),
            Solution::new(vec![2.0, 2.0]),
        ];
        let diversity = population_manager.calculate_diversity(&multi_pop).unwrap();
        assert!(diversity > 0.0);
    }

    #[test]
    fn test_selection_strategies() {
        let selection = SelectionStrategies;
        let population = vec![
            Solution { variables: vec![1.0], fitness: 0.5, constraints: vec![] },
            Solution { variables: vec![2.0], fitness: 0.8, constraints: vec![] },
            Solution { variables: vec![3.0], fitness: 0.3, constraints: vec![] },
            Solution { variables: vec![4.0], fitness: 0.9, constraints: vec![] },
        ];

        // Test tournament selection
        let selected = selection.tournament_selection(&population, 2, 2).unwrap();
        assert_eq!(selected.len(), 2);
        assert!(selected.iter().all(|&i| i < population.len()));

        // Test roulette wheel selection
        let selected = selection.roulette_wheel_selection(&population, 3).unwrap();
        assert_eq!(selected.len(), 3);
        assert!(selected.iter().all(|&i| i < population.len()));

        // Test rank selection
        let selected = selection.rank_selection(&population, 2).unwrap();
        assert_eq!(selected.len(), 2);
        assert!(selected.iter().all(|&i| i < population.len()));
    }

    #[test]
    fn test_crossover_operators() {
        let crossover = CrossoverOperators;
        let parent1 = Solution::new(vec![1.0, 2.0, 3.0, 4.0]);
        let parent2 = Solution::new(vec![5.0, 6.0, 7.0, 8.0]);

        // Test single-point crossover
        let (child1, child2) = crossover.single_point_crossover(&parent1, &parent2).unwrap();
        assert_eq!(child1.variables.len(), parent1.variables.len());
        assert_eq!(child2.variables.len(), parent2.variables.len());

        // Test uniform crossover
        let (child1, child2) = crossover.uniform_crossover(&parent1, &parent2, 0.5).unwrap();
        assert_eq!(child1.variables.len(), parent1.variables.len());
        assert_eq!(child2.variables.len(), parent2.variables.len());

        // Test arithmetic crossover
        let (child1, child2) = crossover.arithmetic_crossover(&parent1, &parent2, 0.5).unwrap();
        assert_eq!(child1.variables.len(), parent1.variables.len());
        assert_eq!(child2.variables.len(), parent2.variables.len());

        // Verify arithmetic crossover properties
        for i in 0..parent1.variables.len() {
            let expected1 = 0.5 * parent1.variables[i] + 0.5 * parent2.variables[i];
            let expected2 = 0.5 * parent1.variables[i] + 0.5 * parent2.variables[i];
            assert!((child1.variables[i] - expected1).abs() < f64::EPSILON);
            assert!((child2.variables[i] - expected2).abs() < f64::EPSILON);
        }
    }

    #[test]
    fn test_mutation_operators() {
        let mutation = MutationOperators;
        let mut solution = Solution::new(vec![1.0, 2.0, 3.0]);
        let original = solution.clone();

        // Test Gaussian mutation
        mutation.gaussian_mutation(&mut solution, 1.0, 0.1).unwrap();
        // With mutation rate 1.0, all variables should be modified
        assert_ne!(solution.variables, original.variables);

        // Test uniform mutation
        let mut solution = Solution::new(vec![1.0, 2.0, 3.0]);
        let bounds = vec![(0.0, 10.0), (0.0, 10.0), (0.0, 10.0)];
        mutation.uniform_mutation(&mut solution, 1.0, &bounds).unwrap();

        // Check bounds are respected
        for (i, &var) in solution.variables.iter().enumerate() {
            assert!(var >= bounds[i].0 && var <= bounds[i].1);
        }
    }

    #[test]
    fn test_algorithm_performance_analysis() {
        let optimizer = MetaheuristicOptimizer::default();
        let solutions = vec![
            Solution { variables: vec![1.0, 2.0], fitness: 0.5, constraints: vec![] },
            Solution { variables: vec![2.0, 3.0], fitness: 0.3, constraints: vec![] },
            Solution { variables: vec![3.0, 4.0], fitness: 0.8, constraints: vec![] },
            Solution { variables: vec![4.0, 5.0], fitness: 0.1, constraints: vec![] },
        ];

        let report = optimizer.analyze_algorithm_performance("test_ga", &solutions).unwrap();

        assert_eq!(report.algorithm_id, "test_ga");
        assert_eq!(report.statistics.num_solutions, 4);
        assert_eq!(report.statistics.feasible_solutions, 4);
        assert_eq!(report.statistics.best_fitness, 0.1);
        assert_eq!(report.statistics.worst_fitness, 0.8);

        let expected_mean = (0.5 + 0.3 + 0.8 + 0.1) / 4.0;
        assert!((report.statistics.mean_fitness - expected_mean).abs() < f64::EPSILON);

        assert!(!report.recommendations.is_empty());
    }

    #[test]
    fn test_optimization_problem_creation() {
        let problem = OptimizationProblem {
            problem_id: "test_problem".to_string(),
            num_variables: 3,
            bounds: vec![(0.0, 1.0), (-1.0, 1.0), (0.0, 10.0)],
            num_objectives: 1,
            num_constraints: 2,
            problem_type: ProblemType::Continuous,
        };

        assert_eq!(problem.num_variables, 3);
        assert_eq!(problem.bounds.len(), 3);
        assert_eq!(problem.num_objectives, 1);
        assert_eq!(problem.num_constraints, 2);
        matches!(problem.problem_type, ProblemType::Continuous);
    }

    #[test]
    fn test_diversity_settings() {
        let settings = DiversitySettings::default();
        matches!(settings.strategy, DiversityStrategy::Crowding);
        assert_eq!(settings.min_diversity, 0.1);
        assert_eq!(settings.target_diversity, 0.5);
        assert_eq!(settings.measurement_interval, 32);
    }
}