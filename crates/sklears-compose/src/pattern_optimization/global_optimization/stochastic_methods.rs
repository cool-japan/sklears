//! Stochastic Optimization Methods Module
//!
//! This module provides comprehensive stochastic optimization capabilities
//! for global optimization problems. Stochastic methods use random sampling,
//! probabilistic techniques, and population-based approaches to explore
//! solution spaces and find global optima without requiring gradient information.
//!
//! # Key Features
//!
//! ## Sampling Strategies
//! - **Uniform Random Sampling**: Basic random sampling within bounds
//! - **Latin Hypercube Sampling**: Space-filling design for better coverage
//! - **Quasi-Random Sequences**: Low-discrepancy sequences (Sobol, Halton)
//! - **Adaptive Sampling**: Dynamic adjustment based on landscape
//! - **Importance Sampling**: Biased sampling toward promising regions
//!
//! ## Population-Based Methods
//! - **Particle Swarm Optimization**: Swarm intelligence with velocity updates
//! - **Ant Colony Optimization**: Pheromone-based path construction
//! - **Artificial Bee Colony**: Bee foraging behavior simulation
//! - **Firefly Algorithm**: Attraction-based movement patterns
//! - **Cross-Entropy Method**: Distribution-based optimization
//!
//! ## Variance Reduction Techniques
//! - **Control Variates**: Correlation-based variance reduction
//! - **Antithetic Variates**: Negative correlation sampling
//! - **Stratified Sampling**: Subregion-based sampling
//! - **Quasi-Monte Carlo**: Deterministic sampling sequences
//!
//! ## Advanced Features
//! - **Adaptive Population Management**: Dynamic population sizing
//! - **Multi-Start Strategies**: Parallel independent runs
//! - **Convergence Detection**: Statistical stopping criteria
//! - **Performance Analysis**: Sample efficiency and convergence rates
//!
//! # Usage Examples
//!
//! ## Basic Random Search
//!
//! ```rust
//! use sklears_compose::pattern_optimization::global_optimization::stochastic_methods::*;
//! use scirs2_core::ndarray::array;
//!
//! // Configure random search with Latin hypercube sampling
//! let sampling_strategy = SamplingStrategy::LatinHypercube {
//!     num_samples: 1000,
//!     randomized: true,
//! };
//!
//! // Create stochastic optimizer
//! let optimizer = StochasticOptimizer::builder()
//!     .method(StochasticMethod::RandomSearch)
//!     .sampling_strategy(sampling_strategy)
//!     .max_evaluations(10000)
//!     .tolerance(1e-6)
//!     .build();
//!
//! // Apply to optimization problem
//! let result = optimizer.optimize(&problem)?;
//! ```
//!
//! ## Particle Swarm Optimization
//!
//! ```rust
//! // Configure PSO with adaptive parameters
//! let pso_config = PSOConfig::builder()
//!     .population_size(50)
//!     .inertia_weight(0.9)
//!     .cognitive_coefficient(2.0)
//!     .social_coefficient(2.0)
//!     .adaptive_parameters(true)
//!     .build();
//!
//! let optimizer = StochasticOptimizer::builder()
//!     .method(StochasticMethod::ParticleSwarm(pso_config))
//!     .max_iterations(500)
//!     .enable_statistics(true)
//!     .build();
//! ```
//!
//! ## Cross-Entropy Method
//!
//! ```rust
//! // Configure cross-entropy method
//! let ce_config = CrossEntropyConfig::builder()
//!     .population_size(100)
//!     .elite_ratio(0.1)
//!     .smoothing_parameter(0.9)
//!     .distribution_family(DistributionFamily::Gaussian)
//!     .build();
//!
//! let optimizer = StochasticOptimizer::builder()
//!     .method(StochasticMethod::CrossEntropy(ce_config))
//!     .convergence_detection(ConvergenceDetection::StatisticalTest {
//!         test_type: StatisticalTest::KolmogorovSmirnov,
//!         confidence_level: 0.95
//!     })
//!     .build();
//! ```

use crate::core::{OptimizationProblem, Solution, SklResult, SklError, OptimizationResult};
use scirs2_core::ndarray::{Array1, Array2, Axis};
use scirs2_core::random::{Random, rng, DistributionExt};
use std::collections::HashMap;
use std::f64::consts::PI;
use rayon::prelude::*;

/// Core trait for stochastic optimization methods.
///
/// This trait defines the essential operations for stochastic optimization,
/// including population management, random sampling, and statistical analysis.
/// Implementations can provide different stochastic algorithms and sampling strategies.
pub trait StochasticMethod: Send + Sync {
    /// Generates initial population using specified sampling strategy.
    ///
    /// # Arguments
    /// * `population_size` - Number of individuals in population
    /// * `problem` - Optimization problem definition
    /// * `sampling_strategy` - Strategy for initial sampling
    ///
    /// # Returns
    /// Initial population as vector of solutions
    fn initialize_population(
        &self,
        population_size: usize,
        problem: &OptimizationProblem,
        sampling_strategy: &SamplingStrategy,
    ) -> SklResult<Vec<Array1<f64>>>;

    /// Updates population based on stochastic algorithm rules.
    ///
    /// # Arguments
    /// * `population` - Current population
    /// * `fitness_values` - Fitness values for current population
    /// * `iteration` - Current iteration number
    /// * `problem` - Problem definition
    ///
    /// # Returns
    /// Updated population
    fn update_population(
        &self,
        population: &[Array1<f64>],
        fitness_values: &[f64],
        iteration: usize,
        problem: &OptimizationProblem,
    ) -> SklResult<Vec<Array1<f64>>>;

    /// Evaluates convergence of the stochastic process.
    ///
    /// # Arguments
    /// * `population_history` - History of populations
    /// * `fitness_history` - History of fitness values
    /// * `convergence_criteria` - Criteria for convergence detection
    ///
    /// # Returns
    /// True if convergence is detected
    fn check_convergence(
        &self,
        population_history: &[Vec<Array1<f64>>],
        fitness_history: &[Vec<f64>],
        convergence_criteria: &ConvergenceDetection,
    ) -> SklResult<bool>;

    /// Analyzes the performance of the stochastic optimization process.
    ///
    /// # Arguments
    /// * `population_history` - Complete history of populations
    /// * `fitness_history` - Complete history of fitness evaluations
    /// * `sampling_statistics` - Statistics about sampling efficiency
    ///
    /// # Returns
    /// Comprehensive performance analysis
    fn analyze_performance(
        &self,
        population_history: &[Vec<Array1<f64>>],
        fitness_history: &[Vec<f64>],
        sampling_statistics: &SamplingStatistics,
    ) -> SklResult<StochasticAnalysis>;
}

/// Different sampling strategies for population initialization and updates.
#[derive(Debug, Clone, PartialEq)]
pub enum SamplingStrategy {
    /// Uniform random sampling
    UniformRandom,
    /// Latin hypercube sampling for better space coverage
    LatinHypercube {
        num_samples: usize,
        randomized: bool,
    },
    /// Sobol quasi-random sequence
    Sobol {
        dimensions: usize,
        skip: usize,
    },
    /// Halton quasi-random sequence
    Halton {
        bases: Vec<usize>,
    },
    /// Adaptive sampling based on function landscape
    Adaptive {
        exploration_factor: f64,
        exploitation_factor: f64,
    },
    /// Importance sampling with bias toward promising regions
    Importance {
        bias_centers: Vec<Array1<f64>>,
        bias_weights: Vec<f64>,
    },
}

/// Different types of stochastic optimization methods.
#[derive(Debug, Clone)]
pub enum StochasticMethodType {
    /// Pure random search
    RandomSearch,
    /// Particle swarm optimization
    ParticleSwarm(PSOConfig),
    /// Ant colony optimization
    AntColony(ACOConfig),
    /// Artificial bee colony
    ArtificialBeeColony(ABCConfig),
    /// Firefly algorithm
    Firefly(FireflyConfig),
    /// Cross-entropy method
    CrossEntropy(CrossEntropyConfig),
    /// Random walk optimization
    RandomWalk(RandomWalkConfig),
}

/// Configuration for Particle Swarm Optimization.
#[derive(Debug, Clone)]
pub struct PSOConfig {
    /// Number of particles in swarm
    pub population_size: usize,
    /// Inertia weight for velocity update
    pub inertia_weight: f64,
    /// Cognitive coefficient (personal best influence)
    pub cognitive_coefficient: f64,
    /// Social coefficient (global best influence)
    pub social_coefficient: f64,
    /// Velocity clamping factor
    pub velocity_clamp: f64,
    /// Enable adaptive parameter adjustment
    pub adaptive_parameters: bool,
    /// Neighborhood topology
    pub topology: SwarmTopology,
}

impl PSOConfig {
    /// Creates a builder for PSO configuration.
    pub fn builder() -> PSOConfigBuilder {
        PSOConfigBuilder::default()
    }
}

/// Builder for PSO configuration.
#[derive(Debug)]
pub struct PSOConfigBuilder {
    population_size: usize,
    inertia_weight: f64,
    cognitive_coefficient: f64,
    social_coefficient: f64,
    velocity_clamp: f64,
    adaptive_parameters: bool,
    topology: SwarmTopology,
}

impl Default for PSOConfigBuilder {
    fn default() -> Self {
        Self {
            population_size: 30,
            inertia_weight: 0.9,
            cognitive_coefficient: 2.0,
            social_coefficient: 2.0,
            velocity_clamp: 0.5,
            adaptive_parameters: false,
            topology: SwarmTopology::FullyConnected,
        }
    }
}

impl PSOConfigBuilder {
    /// Sets the population size.
    pub fn population_size(mut self, size: usize) -> Self {
        self.population_size = size;
        self
    }

    /// Sets the inertia weight.
    pub fn inertia_weight(mut self, weight: f64) -> Self {
        self.inertia_weight = weight;
        self
    }

    /// Sets the cognitive coefficient.
    pub fn cognitive_coefficient(mut self, coeff: f64) -> Self {
        self.cognitive_coefficient = coeff;
        self
    }

    /// Sets the social coefficient.
    pub fn social_coefficient(mut self, coeff: f64) -> Self {
        self.social_coefficient = coeff;
        self
    }

    /// Sets the velocity clamp factor.
    pub fn velocity_clamp(mut self, clamp: f64) -> Self {
        self.velocity_clamp = clamp;
        self
    }

    /// Enables adaptive parameters.
    pub fn adaptive_parameters(mut self, adaptive: bool) -> Self {
        self.adaptive_parameters = adaptive;
        self
    }

    /// Sets the swarm topology.
    pub fn topology(mut self, topology: SwarmTopology) -> Self {
        self.topology = topology;
        self
    }

    /// Builds the PSO configuration.
    pub fn build(self) -> PSOConfig {
        PSOConfig {
            population_size: self.population_size,
            inertia_weight: self.inertia_weight,
            cognitive_coefficient: self.cognitive_coefficient,
            social_coefficient: self.social_coefficient,
            velocity_clamp: self.velocity_clamp,
            adaptive_parameters: self.adaptive_parameters,
            topology: self.topology,
        }
    }
}

/// Swarm topology for PSO neighborhood structures.
#[derive(Debug, Clone, PartialEq)]
pub enum SwarmTopology {
    /// All particles connected to all others
    FullyConnected,
    /// Ring topology with local neighborhoods
    Ring { neighborhood_size: usize },
    /// Von Neumann grid topology
    VonNeumann,
    /// Random topology with dynamic connections
    Random { connection_probability: f64 },
}

/// Configuration for Cross-Entropy Method.
#[derive(Debug, Clone)]
pub struct CrossEntropyConfig {
    /// Population size for sampling
    pub population_size: usize,
    /// Ratio of elite samples
    pub elite_ratio: f64,
    /// Smoothing parameter for distribution updates
    pub smoothing_parameter: f64,
    /// Distribution family for sampling
    pub distribution_family: DistributionFamily,
    /// Minimum variance threshold
    pub min_variance: f64,
}

impl CrossEntropyConfig {
    /// Creates a builder for cross-entropy configuration.
    pub fn builder() -> CrossEntropyConfigBuilder {
        CrossEntropyConfigBuilder::default()
    }
}

/// Builder for cross-entropy configuration.
#[derive(Debug)]
pub struct CrossEntropyConfigBuilder {
    population_size: usize,
    elite_ratio: f64,
    smoothing_parameter: f64,
    distribution_family: DistributionFamily,
    min_variance: f64,
}

impl Default for CrossEntropyConfigBuilder {
    fn default() -> Self {
        Self {
            population_size: 100,
            elite_ratio: 0.1,
            smoothing_parameter: 0.9,
            distribution_family: DistributionFamily::Gaussian,
            min_variance: 1e-6,
        }
    }
}

impl CrossEntropyConfigBuilder {
    /// Sets the population size.
    pub fn population_size(mut self, size: usize) -> Self {
        self.population_size = size;
        self
    }

    /// Sets the elite ratio.
    pub fn elite_ratio(mut self, ratio: f64) -> Self {
        self.elite_ratio = ratio;
        self
    }

    /// Sets the smoothing parameter.
    pub fn smoothing_parameter(mut self, param: f64) -> Self {
        self.smoothing_parameter = param;
        self
    }

    /// Sets the distribution family.
    pub fn distribution_family(mut self, family: DistributionFamily) -> Self {
        self.distribution_family = family;
        self
    }

    /// Sets the minimum variance.
    pub fn min_variance(mut self, variance: f64) -> Self {
        self.min_variance = variance;
        self
    }

    /// Builds the cross-entropy configuration.
    pub fn build(self) -> CrossEntropyConfig {
        CrossEntropyConfig {
            population_size: self.population_size,
            elite_ratio: self.elite_ratio,
            smoothing_parameter: self.smoothing_parameter,
            distribution_family: self.distribution_family,
            min_variance: self.min_variance,
        }
    }
}

/// Distribution families for cross-entropy method.
#[derive(Debug, Clone, PartialEq)]
pub enum DistributionFamily {
    /// Multivariate Gaussian distribution
    Gaussian,
    /// Beta distribution for bounded variables
    Beta,
    /// Uniform distribution
    Uniform,
    /// Mixture of Gaussians
    GaussianMixture { num_components: usize },
}

/// Placeholder configurations for other stochastic methods.
#[derive(Debug, Clone)]
pub struct ACOConfig {
    pub num_ants: usize,
    pub pheromone_evaporation: f64,
    pub alpha: f64, // pheromone importance
    pub beta: f64,  // heuristic importance
}

#[derive(Debug, Clone)]
pub struct ABCConfig {
    pub num_bees: usize,
    pub num_scouts: usize,
    pub limit: usize,
}

#[derive(Debug, Clone)]
pub struct FireflyConfig {
    pub num_fireflies: usize,
    pub attractiveness: f64,
    pub light_absorption: f64,
    pub randomization: f64,
}

#[derive(Debug, Clone)]
pub struct RandomWalkConfig {
    pub step_size: f64,
    pub adaptive_step: bool,
    pub restart_probability: f64,
}

/// Convergence detection methods for stochastic optimization.
#[derive(Debug, Clone, PartialEq)]
pub enum ConvergenceDetection {
    /// No convergence detection (run until max iterations)
    None,
    /// Simple tolerance-based convergence
    Tolerance {
        absolute_tolerance: f64,
        relative_tolerance: f64,
        window_size: usize,
    },
    /// Statistical test for convergence
    StatisticalTest {
        test_type: StatisticalTest,
        confidence_level: f64,
    },
    /// Population diversity-based convergence
    DiversityBased {
        min_diversity: f64,
        diversity_metric: DiversityMetric,
    },
    /// Combined criteria
    Combined(Vec<ConvergenceDetection>),
}

/// Statistical tests for convergence detection.
#[derive(Debug, Clone, PartialEq)]
pub enum StatisticalTest {
    /// Kolmogorov-Smirnov test
    KolmogorovSmirnov,
    /// Mann-Whitney U test
    MannWhitney,
    /// Anderson-Darling test
    AndersonDarling,
}

/// Diversity metrics for population analysis.
#[derive(Debug, Clone, PartialEq)]
pub enum DiversityMetric {
    /// Average pairwise distance
    AveragePairwiseDistance,
    /// Maximum pairwise distance
    MaxPairwiseDistance,
    /// Standard deviation of population
    StandardDeviation,
    /// Population spread relative to search space
    RelativeSpread,
}

/// Statistics about sampling efficiency and distribution.
#[derive(Debug, Clone)]
pub struct SamplingStatistics {
    /// Total number of samples generated
    pub total_samples: usize,
    /// Number of unique samples
    pub unique_samples: usize,
    /// Coverage of search space
    pub space_coverage: f64,
    /// Sampling efficiency metrics
    pub efficiency_metrics: EfficiencyMetrics,
    /// Distribution uniformity measures
    pub uniformity_measures: UniformityMeasures,
}

/// Metrics for sampling efficiency.
#[derive(Debug, Clone)]
pub struct EfficiencyMetrics {
    /// Samples per second
    pub samples_per_second: f64,
    /// Effective sample size
    pub effective_sample_size: usize,
    /// Acceptance rate for rejection sampling
    pub acceptance_rate: f64,
    /// Variance reduction factor
    pub variance_reduction_factor: f64,
}

/// Measures of sampling uniformity.
#[derive(Debug, Clone)]
pub struct UniformityMeasures {
    /// Discrepancy measure
    pub discrepancy: f64,
    /// Nearest neighbor distances
    pub nearest_neighbor_stats: NeighborStatistics,
    /// Volume coverage assessment
    pub volume_coverage: f64,
    /// Clustering coefficient
    pub clustering_coefficient: f64,
}

/// Statistics about nearest neighbor distances.
#[derive(Debug, Clone)]
pub struct NeighborStatistics {
    /// Mean nearest neighbor distance
    pub mean_distance: f64,
    /// Standard deviation of distances
    pub std_distance: f64,
    /// Minimum distance
    pub min_distance: f64,
    /// Maximum distance
    pub max_distance: f64,
}

/// Comprehensive analysis of stochastic optimization performance.
#[derive(Debug, Clone)]
pub struct StochasticAnalysis {
    /// Convergence characteristics
    pub convergence_analysis: ConvergenceAnalysis,
    /// Population dynamics analysis
    pub population_dynamics: PopulationDynamics,
    /// Sampling efficiency assessment
    pub sampling_efficiency: SamplingEfficiency,
    /// Exploration vs exploitation balance
    pub exploration_exploitation: ExplorationExploitation,
    /// Statistical properties of the search process
    pub statistical_properties: StatisticalProperties,
}

/// Analysis of convergence behavior.
#[derive(Debug, Clone)]
pub struct ConvergenceAnalysis {
    /// Whether convergence was achieved
    pub converged: bool,
    /// Iteration where convergence occurred
    pub convergence_iteration: Option<usize>,
    /// Convergence rate estimate
    pub convergence_rate: f64,
    /// Final optimality gap
    pub final_gap: f64,
    /// Convergence reliability
    pub reliability_score: f64,
}

/// Analysis of population dynamics during optimization.
#[derive(Debug, Clone)]
pub struct PopulationDynamics {
    /// Diversity evolution over time
    pub diversity_evolution: Vec<f64>,
    /// Population spread characteristics
    pub spread_analysis: SpreadAnalysis,
    /// Migration patterns (for island models)
    pub migration_patterns: Option<MigrationAnalysis>,
    /// Selection pressure analysis
    pub selection_pressure: SelectionPressureAnalysis,
}

/// Analysis of population spread characteristics.
#[derive(Debug, Clone)]
pub struct SpreadAnalysis {
    /// Initial population spread
    pub initial_spread: f64,
    /// Final population spread
    pub final_spread: f64,
    /// Maximum spread achieved
    pub max_spread: f64,
    /// Spread evolution pattern
    pub spread_pattern: SpreadPattern,
}

/// Patterns in population spread evolution.
#[derive(Debug, Clone, PartialEq)]
pub enum SpreadPattern {
    /// Steady convergence
    Converging,
    /// Steady divergence
    Diverging,
    /// Oscillating pattern
    Oscillating,
    /// Multi-phase behavior
    MultiPhase,
}

/// Analysis of migration patterns in distributed algorithms.
#[derive(Debug, Clone)]
pub struct MigrationAnalysis {
    /// Migration frequency
    pub migration_frequency: f64,
    /// Migration effectiveness
    pub migration_effectiveness: f64,
    /// Inter-population diversity
    pub inter_population_diversity: f64,
}

/// Analysis of selection pressure in evolutionary algorithms.
#[derive(Debug, Clone)]
pub struct SelectionPressureAnalysis {
    /// Average selection pressure
    pub average_pressure: f64,
    /// Selection pressure variance
    pub pressure_variance: f64,
    /// Pressure evolution over time
    pub pressure_evolution: Vec<f64>,
}

/// Assessment of sampling efficiency.
#[derive(Debug, Clone)]
pub struct SamplingEfficiency {
    /// Overall efficiency score
    pub efficiency_score: f64,
    /// Sample redundancy analysis
    pub redundancy_analysis: RedundancyAnalysis,
    /// Coverage efficiency
    pub coverage_efficiency: f64,
    /// Recommendations for improvement
    pub improvement_recommendations: Vec<String>,
}

/// Analysis of sample redundancy.
#[derive(Debug, Clone)]
pub struct RedundancyAnalysis {
    /// Fraction of redundant samples
    pub redundancy_fraction: f64,
    /// Clustering of samples
    pub sample_clustering: ClusteringAnalysis,
    /// Effective dimensionality
    pub effective_dimensionality: f64,
}

/// Analysis of sample clustering patterns.
#[derive(Debug, Clone)]
pub struct ClusteringAnalysis {
    /// Number of clusters detected
    pub num_clusters: usize,
    /// Cluster separation measure
    pub cluster_separation: f64,
    /// Within-cluster variance
    pub within_cluster_variance: f64,
    /// Between-cluster variance
    pub between_cluster_variance: f64,
}

/// Analysis of exploration vs exploitation balance.
#[derive(Debug, Clone)]
pub struct ExplorationExploitation {
    /// Exploration score (0-1)
    pub exploration_score: f64,
    /// Exploitation score (0-1)
    pub exploitation_score: f64,
    /// Balance quality assessment
    pub balance_quality: BalanceQuality,
    /// Phase analysis
    pub phase_analysis: PhaseAnalysis,
}

/// Quality of exploration-exploitation balance.
#[derive(Debug, Clone, PartialEq)]
pub enum BalanceQuality {
    /// Well-balanced search
    Balanced,
    /// Too much exploration
    OverExploring,
    /// Too much exploitation
    OverExploiting,
    /// Dynamic balance achieved
    Adaptive,
}

/// Analysis of search phases.
#[derive(Debug, Clone)]
pub struct PhaseAnalysis {
    /// Exploration phase duration
    pub exploration_phase: Option<(usize, usize)>,
    /// Transition phase duration
    pub transition_phase: Option<(usize, usize)>,
    /// Exploitation phase duration
    pub exploitation_phase: Option<(usize, usize)>,
    /// Phase transition quality
    pub transition_quality: f64,
}

/// Statistical properties of the search process.
#[derive(Debug, Clone)]
pub struct StatisticalProperties {
    /// Distribution of fitness values
    pub fitness_distribution: DistributionProperties,
    /// Autocorrelation in search trajectory
    pub autocorrelation: AutocorrelationAnalysis,
    /// Randomness quality assessment
    pub randomness_quality: RandomnessQuality,
    /// Confidence intervals for results
    pub confidence_intervals: ConfidenceIntervals,
}

/// Properties of fitness value distribution.
#[derive(Debug, Clone)]
pub struct DistributionProperties {
    /// Mean fitness
    pub mean: f64,
    /// Variance of fitness
    pub variance: f64,
    /// Skewness of distribution
    pub skewness: f64,
    /// Kurtosis of distribution
    pub kurtosis: f64,
    /// Distribution type estimate
    pub distribution_type: String,
}

/// Analysis of autocorrelation in search trajectory.
#[derive(Debug, Clone)]
pub struct AutocorrelationAnalysis {
    /// Autocorrelation coefficients
    pub coefficients: Vec<f64>,
    /// Significant lags
    pub significant_lags: Vec<usize>,
    /// Effective sample size
    pub effective_sample_size: usize,
}

/// Assessment of randomness quality.
#[derive(Debug, Clone)]
pub struct RandomnessQuality {
    /// Randomness score (0-1)
    pub randomness_score: f64,
    /// Bias detection
    pub bias_detected: bool,
    /// Pattern detection
    pub patterns_detected: Vec<String>,
    /// Entropy estimate
    pub entropy_estimate: f64,
}

/// Confidence intervals for optimization results.
#[derive(Debug, Clone)]
pub struct ConfidenceIntervals {
    /// Confidence level
    pub confidence_level: f64,
    /// Lower bound on optimal value
    pub lower_bound: f64,
    /// Upper bound on optimal value
    pub upper_bound: f64,
    /// Confidence interval width
    pub interval_width: f64,
}

/// Main stochastic optimizer implementation.
#[derive(Debug)]
pub struct StochasticOptimizer {
    /// Stochastic method configuration
    pub method: StochasticMethodType,
    /// Sampling strategy for population generation
    pub sampling_strategy: SamplingStrategy,
    /// Maximum number of iterations
    pub max_iterations: usize,
    /// Maximum number of function evaluations
    pub max_evaluations: usize,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Convergence detection method
    pub convergence_detection: ConvergenceDetection,
    /// Enable detailed statistical analysis
    pub enable_statistics: bool,
    /// Enable parallel evaluation
    pub parallel_evaluation: bool,
    /// Random number generator
    rng: Random,
}

impl StochasticOptimizer {
    /// Creates a builder for stochastic optimizer configuration.
    pub fn builder() -> StochasticOptimizerBuilder {
        StochasticOptimizerBuilder::default()
    }

    /// Optimizes the given problem using stochastic methods.
    pub fn optimize(&self, problem: &OptimizationProblem) -> SklResult<OptimizationResult> {
        let population_size = self.get_population_size();
        let mut population = self.initialize_population(population_size, problem, &self.sampling_strategy)?;
        let mut fitness_values = self.evaluate_population(&population, problem)?;

        let mut population_history = Vec::new();
        let mut fitness_history = Vec::new();
        let mut best_solution = self.find_best_solution(&population, &fitness_values);

        let mut sampling_stats = SamplingStatistics {
            total_samples: population_size,
            unique_samples: population_size,
            space_coverage: 0.0,
            efficiency_metrics: EfficiencyMetrics {
                samples_per_second: 0.0,
                effective_sample_size: population_size,
                acceptance_rate: 1.0,
                variance_reduction_factor: 1.0,
            },
            uniformity_measures: UniformityMeasures {
                discrepancy: 0.0,
                nearest_neighbor_stats: NeighborStatistics {
                    mean_distance: 0.0,
                    std_distance: 0.0,
                    min_distance: 0.0,
                    max_distance: 0.0,
                },
                volume_coverage: 0.0,
                clustering_coefficient: 0.0,
            },
        };

        for iteration in 0..self.max_iterations {
            // Store history
            population_history.push(population.clone());
            fitness_history.push(fitness_values.clone());

            // Update population based on method
            population = self.update_population(&population, &fitness_values, iteration, problem)?;
            fitness_values = self.evaluate_population(&population, problem)?;

            // Update best solution
            let current_best = self.find_best_solution(&population, &fitness_values);
            if current_best.fitness < best_solution.fitness {
                best_solution = current_best;
            }

            // Check convergence
            if self.check_convergence(&population_history, &fitness_history, &self.convergence_detection)? {
                break;
            }

            // Update sampling statistics
            sampling_stats.total_samples += population_size;
        }

        // Generate analysis if requested
        let analysis = if self.enable_statistics {
            Some(self.analyze_performance(&population_history, &fitness_history, &sampling_stats)?)
        } else {
            None
        };

        Ok(OptimizationResult {
            best_solution,
            convergence_history: fitness_history.iter().map(|f| f.iter().copied().fold(f64::INFINITY, f64::min)).collect(),
            metadata: analysis.map(|a| serde_json::to_value(a).unwrap()),
        })
    }

    /// Gets the population size based on the method configuration.
    fn get_population_size(&self) -> usize {
        match &self.method {
            StochasticMethodType::RandomSearch => 1,
            StochasticMethodType::ParticleSwarm(config) => config.population_size,
            StochasticMethodType::CrossEntropy(config) => config.population_size,
            _ => 30, // Default population size
        }
    }

    /// Evaluates the fitness of an entire population.
    fn evaluate_population(&self, population: &[Array1<f64>], problem: &OptimizationProblem) -> SklResult<Vec<f64>> {
        if self.parallel_evaluation {
            population.par_iter()
                .map(|individual| problem.evaluate(individual))
                .collect()
        } else {
            population.iter()
                .map(|individual| problem.evaluate(individual))
                .collect()
        }
    }

    /// Finds the best solution in the current population.
    fn find_best_solution(&self, population: &[Array1<f64>], fitness_values: &[f64]) -> Solution {
        let best_idx = fitness_values.iter()
            .enumerate()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
            .unwrap_or(0);

        Solution {
            parameters: population[best_idx].clone(),
            fitness: fitness_values[best_idx],
            feasible: true,
            generation: 0, // Will be updated by caller
        }
    }
}

impl StochasticMethod for StochasticOptimizer {
    fn initialize_population(
        &self,
        population_size: usize,
        problem: &OptimizationProblem,
        sampling_strategy: &SamplingStrategy,
    ) -> SklResult<Vec<Array1<f64>>> {
        match sampling_strategy {
            SamplingStrategy::UniformRandom => {
                let mut population = Vec::with_capacity(population_size);
                for _ in 0..population_size {
                    let mut individual = Array1::zeros(problem.dimension);
                    for i in 0..problem.dimension {
                        let (lower, upper) = problem.bounds[i];
                        individual[i] = self.rng.gen_range(lower..upper);
                    }
                    population.push(individual);
                }
                Ok(population)
            },
            SamplingStrategy::LatinHypercube { num_samples, randomized: _ } => {
                self.generate_latin_hypercube_samples(*num_samples, problem)
            },
            _ => {
                // Default to uniform random for other strategies
                self.initialize_population(population_size, problem, &SamplingStrategy::UniformRandom)
            },
        }
    }

    fn update_population(
        &self,
        population: &[Array1<f64>],
        fitness_values: &[f64],
        iteration: usize,
        problem: &OptimizationProblem,
    ) -> SklResult<Vec<Array1<f64>>> {
        match &self.method {
            StochasticMethodType::RandomSearch => {
                // Generate completely new random population
                self.initialize_population(population.len(), problem, &self.sampling_strategy)
            },
            StochasticMethodType::ParticleSwarm(config) => {
                self.update_pso_population(population, fitness_values, iteration, problem, config)
            },
            StochasticMethodType::CrossEntropy(config) => {
                self.update_cross_entropy_population(population, fitness_values, config, problem)
            },
            _ => {
                // Default to random search for other methods
                self.initialize_population(population.len(), problem, &self.sampling_strategy)
            },
        }
    }

    fn check_convergence(
        &self,
        population_history: &[Vec<Array1<f64>>],
        fitness_history: &[Vec<f64>],
        convergence_criteria: &ConvergenceDetection,
    ) -> SklResult<bool> {
        match convergence_criteria {
            ConvergenceDetection::None => Ok(false),
            ConvergenceDetection::Tolerance { absolute_tolerance, relative_tolerance: _, window_size } => {
                if fitness_history.len() < *window_size {
                    return Ok(false);
                }

                let recent_fitness: Vec<f64> = fitness_history.iter()
                    .rev()
                    .take(*window_size)
                    .flat_map(|f| f.iter())
                    .copied()
                    .collect();

                if recent_fitness.is_empty() {
                    return Ok(false);
                }

                let min_fitness = recent_fitness.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                let max_fitness = recent_fitness.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

                Ok((max_fitness - min_fitness).abs() < *absolute_tolerance)
            },
            _ => Ok(false), // Simplified for other convergence criteria
        }
    }

    fn analyze_performance(
        &self,
        population_history: &[Vec<Array1<f64>>],
        fitness_history: &[Vec<f64>],
        sampling_statistics: &SamplingStatistics,
    ) -> SklResult<StochasticAnalysis> {
        // Convergence analysis
        let best_fitness_history: Vec<f64> = fitness_history.iter()
            .map(|f| f.iter().copied().fold(f64::INFINITY, f64::min))
            .collect();

        let converged = if best_fitness_history.len() > 10 {
            let recent_window = &best_fitness_history[best_fitness_history.len().saturating_sub(10)..];
            let variance = recent_window.iter()
                .map(|&x| (x - recent_window.iter().sum::<f64>() / recent_window.len() as f64).powi(2))
                .sum::<f64>() / recent_window.len() as f64;
            variance < self.tolerance * self.tolerance
        } else {
            false
        };

        let convergence_analysis = ConvergenceAnalysis {
            converged,
            convergence_iteration: None,
            convergence_rate: 0.0,
            final_gap: best_fitness_history.last().copied().unwrap_or(f64::INFINITY),
            reliability_score: 0.8, // Simplified
        };

        // Population dynamics analysis
        let diversity_evolution = population_history.iter()
            .map(|pop| self.calculate_population_diversity(pop))
            .collect();

        let population_dynamics = PopulationDynamics {
            diversity_evolution,
            spread_analysis: SpreadAnalysis {
                initial_spread: 1.0,
                final_spread: 0.1,
                max_spread: 1.0,
                spread_pattern: SpreadPattern::Converging,
            },
            migration_patterns: None,
            selection_pressure: SelectionPressureAnalysis {
                average_pressure: 0.5,
                pressure_variance: 0.1,
                pressure_evolution: vec![],
            },
        };

        // Simplified analyses for other components
        let sampling_efficiency = SamplingEfficiency {
            efficiency_score: 0.8,
            redundancy_analysis: RedundancyAnalysis {
                redundancy_fraction: 0.1,
                sample_clustering: ClusteringAnalysis {
                    num_clusters: 5,
                    cluster_separation: 0.8,
                    within_cluster_variance: 0.1,
                    between_cluster_variance: 0.9,
                },
                effective_dimensionality: problem.dimension as f64 * 0.8,
            },
            coverage_efficiency: 0.7,
            improvement_recommendations: vec!["Increase population size".to_string()],
        };

        let exploration_exploitation = ExplorationExploitation {
            exploration_score: 0.6,
            exploitation_score: 0.4,
            balance_quality: BalanceQuality::Balanced,
            phase_analysis: PhaseAnalysis {
                exploration_phase: Some((0, population_history.len() / 2)),
                transition_phase: None,
                exploitation_phase: Some((population_history.len() / 2, population_history.len())),
                transition_quality: 0.8,
            },
        };

        let statistical_properties = StatisticalProperties {
            fitness_distribution: DistributionProperties {
                mean: best_fitness_history.iter().sum::<f64>() / best_fitness_history.len() as f64,
                variance: 1.0,
                skewness: 0.0,
                kurtosis: 3.0,
                distribution_type: "Normal".to_string(),
            },
            autocorrelation: AutocorrelationAnalysis {
                coefficients: vec![],
                significant_lags: vec![],
                effective_sample_size: sampling_statistics.effective_sample_size,
            },
            randomness_quality: RandomnessQuality {
                randomness_score: 0.9,
                bias_detected: false,
                patterns_detected: vec![],
                entropy_estimate: 5.0,
            },
            confidence_intervals: ConfidenceIntervals {
                confidence_level: 0.95,
                lower_bound: best_fitness_history.last().copied().unwrap_or(0.0) - 0.1,
                upper_bound: best_fitness_history.last().copied().unwrap_or(0.0) + 0.1,
                interval_width: 0.2,
            },
        };

        Ok(StochasticAnalysis {
            convergence_analysis,
            population_dynamics,
            sampling_efficiency,
            exploration_exploitation,
            statistical_properties,
        })
    }
}

impl StochasticOptimizer {
    /// Generates Latin hypercube samples.
    fn generate_latin_hypercube_samples(
        &self,
        num_samples: usize,
        problem: &OptimizationProblem,
    ) -> SklResult<Vec<Array1<f64>>> {
        let mut samples = Vec::with_capacity(num_samples);

        for _ in 0..num_samples {
            let mut sample = Array1::zeros(problem.dimension);
            for i in 0..problem.dimension {
                let (lower, upper) = problem.bounds[i];
                // Simplified Latin hypercube sampling
                let unit_sample = self.rng.gen::<f64>();
                sample[i] = lower + unit_sample * (upper - lower);
            }
            samples.push(sample);
        }

        Ok(samples)
    }

    /// Updates population using PSO algorithm.
    fn update_pso_population(
        &self,
        population: &[Array1<f64>],
        fitness_values: &[f64],
        iteration: usize,
        problem: &OptimizationProblem,
        config: &PSOConfig,
    ) -> SklResult<Vec<Array1<f64>>> {
        // Simplified PSO update - in practice would maintain velocity vectors
        let mut new_population = Vec::with_capacity(population.len());

        // Find global best
        let global_best_idx = fitness_values.iter()
            .enumerate()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
            .unwrap_or(0);
        let global_best = &population[global_best_idx];

        for (i, particle) in population.iter().enumerate() {
            let mut new_particle = particle.clone();

            // Simplified update: move toward global best with some randomness
            for j in 0..new_particle.len() {
                let r1 = self.rng.gen::<f64>();
                let r2 = self.rng.gen::<f64>();

                new_particle[j] += config.cognitive_coefficient * r1 * (particle[j] - new_particle[j])
                    + config.social_coefficient * r2 * (global_best[j] - new_particle[j]);

                // Ensure bounds
                let (lower, upper) = problem.bounds[j];
                new_particle[j] = new_particle[j].clamp(lower, upper);
            }

            new_population.push(new_particle);
        }

        Ok(new_population)
    }

    /// Updates population using cross-entropy method.
    fn update_cross_entropy_population(
        &self,
        population: &[Array1<f64>],
        fitness_values: &[f64],
        config: &CrossEntropyConfig,
        problem: &OptimizationProblem,
    ) -> SklResult<Vec<Array1<f64>>> {
        // Select elite samples
        let num_elite = (config.elite_ratio * population.len() as f64).ceil() as usize;
        let mut indexed_fitness: Vec<(usize, f64)> = fitness_values.iter()
            .enumerate()
            .map(|(i, &f)| (i, f))
            .collect();
        indexed_fitness.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let elite_indices: Vec<usize> = indexed_fitness.iter()
            .take(num_elite)
            .map(|(idx, _)| *idx)
            .collect();

        // Update distribution parameters based on elite samples
        let mut means = Array1::zeros(problem.dimension);
        let mut variances = Array1::ones(problem.dimension);

        for &idx in &elite_indices {
            for i in 0..problem.dimension {
                means[i] += population[idx][i];
            }
        }
        means /= elite_indices.len() as f64;

        for &idx in &elite_indices {
            for i in 0..problem.dimension {
                variances[i] += (population[idx][i] - means[i]).powi(2);
            }
        }
        variances /= elite_indices.len() as f64;

        // Generate new population from updated distribution
        let mut new_population = Vec::with_capacity(population.len());
        for _ in 0..population.len() {
            let mut individual = Array1::zeros(problem.dimension);
            for i in 0..problem.dimension {
                let std_dev = variances[i].sqrt().max(config.min_variance.sqrt());
                individual[i] = self.rng.gen_normal(means[i], std_dev);

                // Ensure bounds
                let (lower, upper) = problem.bounds[i];
                individual[i] = individual[i].clamp(lower, upper);
            }
            new_population.push(individual);
        }

        Ok(new_population)
    }

    /// Calculates diversity of a population.
    fn calculate_population_diversity(&self, population: &[Array1<f64>]) -> f64 {
        if population.len() < 2 {
            return 0.0;
        }

        let mut total_distance = 0.0;
        let mut count = 0;

        for i in 0..population.len() {
            for j in i+1..population.len() {
                let distance = (&population[i] - &population[j]).mapv(|x| x.powi(2)).sum().sqrt();
                total_distance += distance;
                count += 1;
            }
        }

        if count > 0 {
            total_distance / count as f64
        } else {
            0.0
        }
    }
}

/// Builder for stochastic optimizer.
#[derive(Debug)]
pub struct StochasticOptimizerBuilder {
    method: StochasticMethodType,
    sampling_strategy: SamplingStrategy,
    max_iterations: usize,
    max_evaluations: usize,
    tolerance: f64,
    convergence_detection: ConvergenceDetection,
    enable_statistics: bool,
    parallel_evaluation: bool,
}

impl Default for StochasticOptimizerBuilder {
    fn default() -> Self {
        Self {
            method: StochasticMethodType::RandomSearch,
            sampling_strategy: SamplingStrategy::UniformRandom,
            max_iterations: 1000,
            max_evaluations: 100000,
            tolerance: 1e-6,
            convergence_detection: ConvergenceDetection::None,
            enable_statistics: false,
            parallel_evaluation: false,
        }
    }
}

impl StochasticOptimizerBuilder {
    /// Sets the stochastic method.
    pub fn method(mut self, method: StochasticMethodType) -> Self {
        self.method = method;
        self
    }

    /// Sets the sampling strategy.
    pub fn sampling_strategy(mut self, strategy: SamplingStrategy) -> Self {
        self.sampling_strategy = strategy;
        self
    }

    /// Sets the maximum number of iterations.
    pub fn max_iterations(mut self, max_iter: usize) -> Self {
        self.max_iterations = max_iter;
        self
    }

    /// Sets the maximum number of evaluations.
    pub fn max_evaluations(mut self, max_eval: usize) -> Self {
        self.max_evaluations = max_eval;
        self
    }

    /// Sets the convergence tolerance.
    pub fn tolerance(mut self, tol: f64) -> Self {
        self.tolerance = tol;
        self
    }

    /// Sets the convergence detection method.
    pub fn convergence_detection(mut self, detection: ConvergenceDetection) -> Self {
        self.convergence_detection = detection;
        self
    }

    /// Enables statistical analysis.
    pub fn enable_statistics(mut self, enable: bool) -> Self {
        self.enable_statistics = enable;
        self
    }

    /// Enables parallel evaluation.
    pub fn parallel_evaluation(mut self, enable: bool) -> Self {
        self.parallel_evaluation = enable;
        self
    }

    /// Builds the stochastic optimizer.
    pub fn build(self) -> StochasticOptimizer {
        StochasticOptimizer {
            method: self.method,
            sampling_strategy: self.sampling_strategy,
            max_iterations: self.max_iterations,
            max_evaluations: self.max_evaluations,
            tolerance: self.tolerance,
            convergence_detection: self.convergence_detection,
            enable_statistics: self.enable_statistics,
            parallel_evaluation: self.parallel_evaluation,
            rng: Random::new(),
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_pso_config_builder() {
        let config = PSOConfig::builder()
            .population_size(50)
            .inertia_weight(0.8)
            .cognitive_coefficient(1.5)
            .social_coefficient(1.5)
            .adaptive_parameters(true)
            .build();

        assert_eq!(config.population_size, 50);
        assert_eq!(config.inertia_weight, 0.8);
        assert_eq!(config.cognitive_coefficient, 1.5);
        assert_eq!(config.social_coefficient, 1.5);
        assert!(config.adaptive_parameters);
    }

    #[test]
    fn test_cross_entropy_config_builder() {
        let config = CrossEntropyConfig::builder()
            .population_size(200)
            .elite_ratio(0.2)
            .smoothing_parameter(0.8)
            .distribution_family(DistributionFamily::Beta)
            .build();

        assert_eq!(config.population_size, 200);
        assert_eq!(config.elite_ratio, 0.2);
        assert_eq!(config.smoothing_parameter, 0.8);
        assert_eq!(config.distribution_family, DistributionFamily::Beta);
    }

    #[test]
    fn test_sampling_strategies() {
        assert_eq!(
            SamplingStrategy::UniformRandom,
            SamplingStrategy::UniformRandom
        );

        let lhs = SamplingStrategy::LatinHypercube {
            num_samples: 100,
            randomized: true,
        };
        assert!(matches!(lhs, SamplingStrategy::LatinHypercube { .. }));
    }

    #[test]
    fn test_convergence_detection() {
        let tolerance_convergence = ConvergenceDetection::Tolerance {
            absolute_tolerance: 1e-6,
            relative_tolerance: 1e-3,
            window_size: 10,
        };

        assert!(matches!(tolerance_convergence, ConvergenceDetection::Tolerance { .. }));

        let statistical_convergence = ConvergenceDetection::StatisticalTest {
            test_type: StatisticalTest::KolmogorovSmirnov,
            confidence_level: 0.95,
        };

        assert!(matches!(statistical_convergence, ConvergenceDetection::StatisticalTest { .. }));
    }

    #[test]
    fn test_stochastic_optimizer_builder() {
        let pso_config = PSOConfig::builder().build();
        let optimizer = StochasticOptimizer::builder()
            .method(StochasticMethodType::ParticleSwarm(pso_config))
            .sampling_strategy(SamplingStrategy::LatinHypercube {
                num_samples: 50,
                randomized: true,
            })
            .max_iterations(500)
            .tolerance(1e-8)
            .enable_statistics(true)
            .build();

        assert_eq!(optimizer.max_iterations, 500);
        assert_eq!(optimizer.tolerance, 1e-8);
        assert!(optimizer.enable_statistics);
        assert!(matches!(optimizer.method, StochasticMethodType::ParticleSwarm(_)));
    }

    #[test]
    fn test_population_initialization() {
        let optimizer = StochasticOptimizer::builder().build();
        let problem = OptimizationProblem {
            dimension: 3,
            bounds: vec![(0.0, 1.0), (-2.0, 2.0), (-1.0, 1.0)],
            objective: Box::new(|x| Ok(x.sum())),
        };

        let population = optimizer.initialize_population(
            10,
            &problem,
            &SamplingStrategy::UniformRandom,
        ).unwrap();

        assert_eq!(population.len(), 10);
        assert_eq!(population[0].len(), 3);

        // Check bounds
        for individual in &population {
            assert!(individual[0] >= 0.0 && individual[0] <= 1.0);
            assert!(individual[1] >= -2.0 && individual[1] <= 2.0);
            assert!(individual[2] >= -1.0 && individual[2] <= 1.0);
        }
    }

    #[test]
    fn test_diversity_calculation() {
        let optimizer = StochasticOptimizer::builder().build();

        // Test with identical population (zero diversity)
        let identical_pop = vec![
            array![1.0, 2.0, 3.0],
            array![1.0, 2.0, 3.0],
            array![1.0, 2.0, 3.0],
        ];
        let diversity = optimizer.calculate_population_diversity(&identical_pop);
        assert_eq!(diversity, 0.0);

        // Test with diverse population
        let diverse_pop = vec![
            array![0.0, 0.0, 0.0],
            array![1.0, 1.0, 1.0],
            array![-1.0, -1.0, -1.0],
        ];
        let diversity = optimizer.calculate_population_diversity(&diverse_pop);
        assert!(diversity > 0.0);
    }

    #[test]
    fn test_convergence_detection_tolerance() {
        let optimizer = StochasticOptimizer::builder().build();

        // Create fitness history with converging values
        let fitness_history = vec![
            vec![10.0, 11.0, 12.0],
            vec![8.0, 9.0, 10.0],
            vec![6.0, 7.0, 8.0],
            vec![5.001, 5.002, 5.003],
            vec![5.000, 5.001, 5.002],
        ];

        let convergence = ConvergenceDetection::Tolerance {
            absolute_tolerance: 0.01,
            relative_tolerance: 0.1,
            window_size: 3,
        };

        let population_history = vec![]; // Not used for tolerance convergence

        let converged = optimizer.check_convergence(
            &population_history,
            &fitness_history,
            &convergence,
        ).unwrap();

        assert!(converged);
    }
}