//! Metaheuristic optimization algorithms and population-based methods
//!
//! This module provides nature-inspired and population-based optimization algorithms including:
//! - Genetic algorithms (GA) with selection, crossover, and mutation operators
//! - Particle swarm optimization (PSO) with various topologies and parameters
//! - Simulated annealing (SA) with adaptive cooling schedules
//! - Tabu search with adaptive memory and aspiration criteria
//! - Variable neighborhood search (VNS) with multiple neighborhood structures
//! - Differential evolution (DE) with multiple mutation and crossover strategies
//! - Ant colony optimization (ACO) with pheromone trail management
//! - Harmony search (HS) with pitch adjustment mechanisms

use std::collections::HashMap;
use std::time::SystemTime;

use scirs2_core::ndarray::{Array1, Array2};
use crate::core::SklResult;
use super::optimization_core::{OptimizationProblem, Solution};

/// Metaheuristic optimizer coordinating nature-inspired algorithms
///
/// Manages various metaheuristic algorithms and provides unified interface
/// for population-based and single-solution optimization methods.
#[derive(Debug)]
pub struct MetaheuristicOptimizer {
    /// Unique optimizer identifier
    pub optimizer_id: String,
    /// Genetic algorithm implementations
    pub genetic_algorithms: HashMap<String, Box<dyn GeneticAlgorithm>>,
    /// Swarm intelligence algorithms
    pub swarm_algorithms: HashMap<String, Box<dyn SwarmAlgorithm>>,
    /// Simulated annealing implementations
    pub simulated_annealing: HashMap<String, Box<dyn SimulatedAnnealing>>,
    /// Tabu search algorithms
    pub tabu_search: HashMap<String, Box<dyn TabuSearch>>,
    /// Variable neighborhood search methods
    pub variable_neighborhood: HashMap<String, Box<dyn VariableNeighborhoodSearch>>,
    /// Differential evolution variants
    pub differential_evolution: HashMap<String, Box<dyn DifferentialEvolution>>,
    /// Ant colony optimization algorithms
    pub ant_colony: HashMap<String, Box<dyn AntColonyOptimization>>,
    /// Harmony search implementations
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

/// Genetic algorithm trait for evolutionary computation
///
/// Implements the canonical genetic algorithm paradigm with selection,
/// crossover, mutation, and environmental selection operations.
pub trait GeneticAlgorithm: Send + Sync {
    /// Initialize population with random individuals
    fn initialize_population(&mut self, problem: &OptimizationProblem, size: usize) -> SklResult<Vec<Solution>>;

    /// Select parents for reproduction
    fn select_parents(&self, population: &[Solution], num_parents: usize) -> SklResult<Vec<usize>>;

    /// Perform crossover to create offspring
    fn crossover(&self, parents: &[Solution]) -> SklResult<Vec<Solution>>;

    /// Apply mutation to offspring
    fn mutate(&self, offspring: &mut [Solution], mutation_rate: f64) -> SklResult<()>;

    /// Select survivors for next generation
    fn environmental_selection(&self, parents: &[Solution], offspring: &[Solution]) -> SklResult<Vec<Solution>>;

    /// Get algorithm parameters
    fn get_algorithm_parameters(&self) -> GeneticAlgorithmParameters;

    /// Set algorithm parameters
    fn set_algorithm_parameters(&mut self, params: GeneticAlgorithmParameters) -> SklResult<()>;
}

/// Genetic algorithm parameters
#[derive(Debug, Clone)]
pub struct GeneticAlgorithmParameters {
    /// Population size
    pub population_size: usize,
    /// Crossover probability
    pub crossover_rate: f64,
    /// Mutation probability
    pub mutation_rate: f64,
    /// Selection pressure
    pub selection_pressure: f64,
    /// Elitism rate (fraction of best individuals preserved)
    pub elitism_rate: f64,
    /// Maximum generations
    pub max_generations: u64,
    /// Convergence threshold
    pub convergence_threshold: f64,
    /// Whether to maintain diversity
    pub diversity_maintenance: bool,
    /// Whether parameters adapt during evolution
    pub adaptive_parameters: bool,
}

/// Swarm algorithm trait for particle swarm optimization
///
/// Implements particle swarm optimization with velocity updates,
/// position updates, and neighborhood topologies.
pub trait SwarmAlgorithm: Send + Sync {
    /// Initialize swarm with particles
    fn initialize_swarm(&mut self, problem: &OptimizationProblem, size: usize) -> SklResult<Swarm>;

    /// Update particle velocities
    fn update_velocities(&mut self, swarm: &mut Swarm) -> SklResult<()>;

    /// Update particle positions
    fn update_positions(&mut self, swarm: &mut Swarm) -> SklResult<()>;

    /// Update personal and global best positions
    fn update_best_positions(&mut self, swarm: &mut Swarm) -> SklResult<()>;

    /// Get swarm parameters
    fn get_swarm_parameters(&self) -> SwarmParameters;

    /// Set swarm parameters
    fn set_swarm_parameters(&mut self, params: SwarmParameters) -> SklResult<()>;
}

/// Swarm structure containing particles and global information
#[derive(Debug, Clone)]
pub struct Swarm {
    /// Individual particles in the swarm
    pub particles: Vec<Particle>,
    /// Global best solution found
    pub global_best: Solution,
    /// History of global best solutions
    pub global_best_history: Vec<Solution>,
    /// Swarm diversity metrics
    pub diversity_metrics: SwarmDiversity,
    /// Convergence metrics
    pub convergence_metrics: SwarmConvergence,
}

/// Individual particle in particle swarm optimization
#[derive(Debug, Clone)]
pub struct Particle {
    /// Unique particle identifier
    pub particle_id: String,
    /// Current position in search space
    pub position: Array1<f64>,
    /// Current velocity vector
    pub velocity: Array1<f64>,
    /// Personal best solution found
    pub personal_best: Solution,
    /// Current fitness value
    pub fitness: f64,
    /// Neighborhood particle IDs
    pub neighborhood: Vec<String>,
    /// Historical states of the particle
    pub particle_history: Vec<ParticleState>,
}

/// Historical state of a particle
#[derive(Debug, Clone)]
pub struct ParticleState {
    /// Timestamp of the state
    pub timestamp: SystemTime,
    /// Position at this time
    pub position: Array1<f64>,
    /// Velocity at this time
    pub velocity: Array1<f64>,
    /// Fitness at this time
    pub fitness: f64,
}

/// Swarm algorithm parameters
#[derive(Debug, Clone)]
pub struct SwarmParameters {
    /// Number of particles in swarm
    pub swarm_size: usize,
    /// Inertia weight for velocity update
    pub inertia_weight: f64,
    /// Cognitive coefficient (personal best attraction)
    pub cognitive_coefficient: f64,
    /// Social coefficient (global best attraction)
    pub social_coefficient: f64,
    /// Maximum velocity magnitude
    pub max_velocity: f64,
    /// Neighborhood topology structure
    pub neighborhood_topology: TopologyType,
    /// Whether to clamp velocities
    pub velocity_clamping: bool,
    /// Whether parameters adapt over time
    pub adaptive_parameters: bool,
}

/// Neighborhood topology types for PSO
#[derive(Debug, Clone)]
pub enum TopologyType {
    /// All particles connected (global PSO)
    GlobalBest,
    /// Local neighborhoods (local PSO)
    LocalBest,
    /// Ring topology
    Ring,
    /// Star topology
    Star,
    /// Random connections
    Random,
    /// Custom topology definition
    Custom(String),
}

/// Swarm diversity metrics
#[derive(Debug, Clone)]
pub struct SwarmDiversity {
    /// Diversity in particle positions
    pub position_diversity: f64,
    /// Diversity in particle velocities
    pub velocity_diversity: f64,
    /// Diversity in fitness values
    pub fitness_diversity: f64,
    /// Exploration vs exploitation ratio
    pub exploration_ratio: f64,
}

/// Swarm convergence metrics
#[derive(Debug, Clone)]
pub struct SwarmConvergence {
    /// Rate of convergence
    pub convergence_rate: f64,
    /// Number of iterations without improvement
    pub stagnation_counter: u32,
    /// Rate of improvement per iteration
    pub improvement_rate: f64,
    /// Probability of convergence
    pub convergence_probability: f64,
}

/// Simulated annealing trait for probabilistic optimization
///
/// Implements simulated annealing with adaptive cooling schedules
/// and probabilistic acceptance of worse solutions.
pub trait SimulatedAnnealing: Send + Sync {
    /// Generate neighbor solution
    fn generate_neighbor(&self, current_solution: &Solution) -> SklResult<Solution>;

    /// Calculate acceptance probability
    fn acceptance_probability(&self, current_energy: f64, new_energy: f64, temperature: f64) -> f64;

    /// Update temperature according to cooling schedule
    fn cooling_schedule(&self, initial_temperature: f64, iteration: u64) -> f64;

    /// Get simulated annealing parameters
    fn get_sa_parameters(&self) -> SimulatedAnnealingParameters;

    /// Check termination criteria
    fn is_termination_criteria_met(&self, temperature: f64, iteration: u64) -> bool;
}

/// Simulated annealing parameters
#[derive(Debug, Clone)]
pub struct SimulatedAnnealingParameters {
    /// Initial temperature
    pub initial_temperature: f64,
    /// Final temperature
    pub final_temperature: f64,
    /// Cooling rate
    pub cooling_rate: f64,
    /// Number of iterations per temperature
    pub iterations_per_temperature: u32,
    /// Maximum iterations
    pub max_iterations: u64,
    /// Neighborhood size for move generation
    pub neighborhood_size: f64,
}

/// Tabu search trait for memory-based optimization
///
/// Implements tabu search with adaptive memory structures
/// and aspiration criteria for escaping local optima.
pub trait TabuSearch: Send + Sync {
    /// Initialize tabu list with given size
    fn initialize_tabu_list(&mut self, size: usize) -> SklResult<()>;

    /// Generate neighborhood solutions
    fn generate_neighborhood(&self, current_solution: &Solution) -> SklResult<Vec<Solution>>;

    /// Update tabu list with new move
    fn update_tabu_list(&mut self, move_made: &TabuMove) -> SklResult<()>;

    /// Check if move is tabu
    fn is_tabu(&self, move_candidate: &TabuMove) -> bool;

    /// Apply aspiration criterion
    fn aspiration_criterion(&self, move_candidate: &TabuMove, best_known: &Solution) -> bool;
}

/// Tabu move representation
#[derive(Debug, Clone)]
pub struct TabuMove {
    /// Move identifier
    pub move_id: String,
    /// Source solution
    pub from_solution: Solution,
    /// Target solution
    pub to_solution: Solution,
    /// Move attributes
    pub move_attributes: HashMap<String, f64>,
    /// Tabu tenure
    pub tabu_tenure: u32,
}

/// Variable neighborhood search trait
///
/// Implements VNS with multiple neighborhood structures
/// and systematic neighborhood change strategies.
pub trait VariableNeighborhoodSearch: Send + Sync {
    /// Get available neighborhood structures
    fn get_neighborhood_structures(&self) -> Vec<NeighborhoodStructure>;

    /// Perform local search in given neighborhood
    fn local_search(&self, solution: &Solution, structure: &NeighborhoodStructure) -> SklResult<Solution>;

    /// Shake solution using neighborhood structure
    fn shake(&self, solution: &Solution, structure: &NeighborhoodStructure) -> SklResult<Solution>;

    /// Change to next neighborhood structure
    fn change_neighborhood(&mut self, improvement_found: bool) -> SklResult<()>;
}

/// Neighborhood structure definition
#[derive(Debug, Clone)]
pub struct NeighborhoodStructure {
    /// Structure identifier
    pub structure_id: String,
    /// Structure name
    pub structure_name: String,
    /// Structure type
    pub structure_type: String,
    /// Structure parameters
    pub parameters: HashMap<String, f64>,
    /// Effective search radius
    pub search_radius: f64,
}

/// Differential evolution trait
///
/// Implements differential evolution with various mutation
/// and crossover strategies for continuous optimization.
pub trait DifferentialEvolution: Send + Sync {
    /// Initialize population
    fn initialize_population(&mut self, problem: &OptimizationProblem) -> SklResult<Vec<Solution>>;

    /// Perform mutation operation
    fn mutation(&self, population: &[Solution], target_index: usize) -> SklResult<Solution>;

    /// Perform crossover operation
    fn crossover(&self, target: &Solution, mutant: &Solution) -> SklResult<Solution>;

    /// Select between target and trial solution
    fn selection(&self, target: &Solution, trial: &Solution) -> Solution;

    /// Get DE parameters
    fn get_de_parameters(&self) -> DifferentialEvolutionParameters;
}

/// Differential evolution parameters
#[derive(Debug, Clone)]
pub struct DifferentialEvolutionParameters {
    /// Population size
    pub population_size: usize,
    /// Mutation factor F
    pub mutation_factor: f64,
    /// Crossover probability
    pub crossover_probability: f64,
    /// Mutation strategy
    pub mutation_strategy: MutationStrategy,
    /// Boundary handling method
    pub boundary_handling: BoundaryHandling,
    /// Whether parameters adapt
    pub adaptive_parameters: bool,
}

/// Mutation strategies for differential evolution
#[derive(Debug, Clone)]
pub enum MutationStrategy {
    /// DE/rand/1
    Rand1,
    /// DE/best/1
    Best1,
    /// DE/current-to-best/1
    CurrentToBest1,
    /// DE/rand/2
    Rand2,
    /// DE/best/2
    Best2,
}

/// Boundary handling methods
#[derive(Debug, Clone)]
pub enum BoundaryHandling {
    /// Clamp to bounds
    Clamp,
    /// Reflect at boundaries
    Reflect,
    /// Wrap around boundaries
    Wrap,
    /// Random reinitialization
    Random,
}

/// Ant colony optimization trait
///
/// Implements ACO with pheromone trail management
/// and probabilistic solution construction.
pub trait AntColonyOptimization: Send + Sync {
    /// Initialize pheromone trails
    fn initialize_pheromone_trails(&mut self, problem: &OptimizationProblem) -> SklResult<()>;

    /// Construct solution using pheromone trails
    fn construct_solution(&self, ant_id: usize) -> SklResult<Solution>;

    /// Update pheromone trails based on solutions
    fn update_pheromones(&mut self, solutions: &[Solution]) -> SklResult<()>;

    /// Evaporate pheromones
    fn evaporate_pheromones(&mut self, evaporation_rate: f64) -> SklResult<()>;

    /// Get ACO parameters
    fn get_aco_parameters(&self) -> AntColonyParameters;
}

/// Ant colony optimization parameters
#[derive(Debug, Clone)]
pub struct AntColonyParameters {
    /// Number of ants
    pub num_ants: usize,
    /// Alpha parameter (pheromone importance)
    pub alpha: f64,
    /// Beta parameter (heuristic importance)
    pub beta: f64,
    /// Evaporation rate
    pub evaporation_rate: f64,
    /// Pheromone deposit amount
    pub pheromone_deposit: f64,
    /// Maximum iterations
    pub max_iterations: u64,
}

/// Harmony search trait
///
/// Implements harmony search with harmony memory
/// and pitch adjustment mechanisms.
pub trait HarmonySearch: Send + Sync {
    /// Initialize harmony memory
    fn initialize_harmony_memory(&mut self, problem: &OptimizationProblem) -> SklResult<()>;

    /// Improvise new harmony
    fn improvise_harmony(&self) -> SklResult<Solution>;

    /// Update harmony memory with new harmony
    fn update_harmony_memory(&mut self, new_harmony: Solution) -> SklResult<()>;

    /// Apply pitch adjustment
    fn pitch_adjustment(&self, harmony: &mut Solution) -> SklResult<()>;

    /// Get harmony search parameters
    fn get_hs_parameters(&self) -> HarmonySearchParameters;
}

/// Harmony search parameters
#[derive(Debug, Clone)]
pub struct HarmonySearchParameters {
    /// Harmony memory size
    pub harmony_memory_size: usize,
    /// Harmony memory considering rate
    pub hmcr: f64,
    /// Pitch adjusting rate
    pub par: f64,
    /// Bandwidth for pitch adjustment
    pub bandwidth: f64,
    /// Maximum iterations
    pub max_iterations: u64,
}

/// Population management utilities
#[derive(Debug, Default)]
pub struct PopulationManager;

/// Selection strategy implementations
#[derive(Debug, Default)]
pub struct SelectionStrategies;

/// Mutation operator implementations
#[derive(Debug, Default)]
pub struct MutationOperators;

/// Crossover operator implementations
#[derive(Debug, Default)]
pub struct CrossoverOperators;

impl Default for MetaheuristicOptimizer {
    fn default() -> Self {
        Self {
            optimizer_id: format!(
                "metaheuristic_{}",
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

impl MetaheuristicOptimizer {
    /// Create a new metaheuristic optimizer
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a genetic algorithm
    pub fn register_genetic_algorithm(&mut self, name: String, algorithm: Box<dyn GeneticAlgorithm>) {
        self.genetic_algorithms.insert(name, algorithm);
    }

    /// Register a swarm algorithm
    pub fn register_swarm_algorithm(&mut self, name: String, algorithm: Box<dyn SwarmAlgorithm>) {
        self.swarm_algorithms.insert(name, algorithm);
    }

    /// Register a simulated annealing algorithm
    pub fn register_simulated_annealing(&mut self, name: String, algorithm: Box<dyn SimulatedAnnealing>) {
        self.simulated_annealing.insert(name, algorithm);
    }

    /// Get available algorithms by type
    pub fn get_available_algorithms(&self) -> HashMap<String, Vec<String>> {
        let mut algorithms = HashMap::new();
        algorithms.insert("genetic".to_string(), self.genetic_algorithms.keys().cloned().collect());
        algorithms.insert("swarm".to_string(), self.swarm_algorithms.keys().cloned().collect());
        algorithms.insert("simulated_annealing".to_string(), self.simulated_annealing.keys().cloned().collect());
        algorithms.insert("tabu_search".to_string(), self.tabu_search.keys().cloned().collect());
        algorithms.insert("variable_neighborhood".to_string(), self.variable_neighborhood.keys().cloned().collect());
        algorithms.insert("differential_evolution".to_string(), self.differential_evolution.keys().cloned().collect());
        algorithms.insert("ant_colony".to_string(), self.ant_colony.keys().cloned().collect());
        algorithms.insert("harmony_search".to_string(), self.harmony_search.keys().cloned().collect());
        algorithms
    }
}

impl Default for GeneticAlgorithmParameters {
    fn default() -> Self {
        Self {
            population_size: 100,
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

impl Default for SwarmParameters {
    fn default() -> Self {
        Self {
            swarm_size: 50,
            inertia_weight: 0.729,
            cognitive_coefficient: 1.494,
            social_coefficient: 1.494,
            max_velocity: 1.0,
            neighborhood_topology: TopologyType::GlobalBest,
            velocity_clamping: true,
            adaptive_parameters: false,
        }
    }
}

impl Default for SimulatedAnnealingParameters {
    fn default() -> Self {
        Self {
            initial_temperature: 1000.0,
            final_temperature: 0.01,
            cooling_rate: 0.95,
            iterations_per_temperature: 100,
            max_iterations: 10000,
            neighborhood_size: 0.1,
        }
    }
}

impl Default for DifferentialEvolutionParameters {
    fn default() -> Self {
        Self {
            population_size: 100,
            mutation_factor: 0.5,
            crossover_probability: 0.9,
            mutation_strategy: MutationStrategy::Rand1,
            boundary_handling: BoundaryHandling::Clamp,
            adaptive_parameters: false,
        }
    }
}

impl Default for AntColonyParameters {
    fn default() -> Self {
        Self {
            num_ants: 50,
            alpha: 1.0,
            beta: 2.0,
            evaporation_rate: 0.1,
            pheromone_deposit: 1.0,
            max_iterations: 1000,
        }
    }
}

impl Default for HarmonySearchParameters {
    fn default() -> Self {
        Self {
            harmony_memory_size: 20,
            hmcr: 0.9,
            par: 0.3,
            bandwidth: 0.1,
            max_iterations: 1000,
        }
    }
}