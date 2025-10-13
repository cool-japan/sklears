//! Global Optimization Core Framework
//!
//! This module provides the central framework for global optimization, offering
//! a unified interface to all optimization methods and comprehensive analysis
//! capabilities. It serves as the coordination layer that integrates various
//! global optimization algorithms and provides intelligent method selection,
//! performance comparison, and adaptive optimization strategies.
//!
//! # Key Features
//!
//! ## Unified Interface
//! - **GlobalOptimizer Trait**: Common interface for all optimization methods
//! - **Method Registry**: Centralized registration and discovery of optimization algorithms
//! - **Strategy Pattern**: Pluggable optimization strategies with runtime selection
//! - **Adaptive Selection**: Intelligent method selection based on problem characteristics
//!
//! ## Multi-Method Orchestration
//! - **Portfolio Optimization**: Running multiple methods in parallel
//! - **Sequential Refinement**: Chaining methods for progressive improvement
//! - **Hybrid Approaches**: Combining complementary optimization strategies
//! - **Resource Management**: Optimal allocation of computational resources
//!
//! ## Comprehensive Analysis
//! - **Performance Profiling**: Detailed analysis of optimization performance
//! - **Convergence Comparison**: Side-by-side convergence analysis
//! - **Method Benchmarking**: Systematic evaluation across problem classes
//! - **Statistical Validation**: Rigorous statistical analysis of results
//!
//! ## Advanced Features
//! - **Problem Classification**: Automatic categorization of optimization problems
//! - **Meta-Learning**: Learning from previous optimization experiences
//! - **Warm-Starting**: Intelligent initialization from prior solutions
//! - **Early Stopping**: Sophisticated termination criteria
//!
//! # Usage Examples
//!
//! ## Basic Global Optimization
//!
//! ```rust
//! use sklears_compose::pattern_optimization::global_optimization::core::*;
//! use scirs2_core::ndarray::array;
//!
//! // Create global optimizer with automatic method selection
//! let optimizer = GlobalOptimizerBuilder::new()
//!     .auto_select_method(true)
//!     .max_evaluations(10000)
//!     .target_accuracy(1e-6)
//!     .build();
//!
//! // Optimize problem with intelligent method selection
//! let result = optimizer.optimize(&problem)?;
//! println!("Best solution: {:?}", result.best_solution);
//! println!("Method used: {}", result.method_used);
//! ```
//!
//! ## Multi-Method Portfolio Optimization
//!
//! ```rust
//! // Configure portfolio of optimization methods
//! let portfolio = MethodPortfolio::builder()
//!     .add_method(OptimizationMethod::SimulatedAnnealing, 0.3)
//!     .add_method(OptimizationMethod::GeneticAlgorithm, 0.3)
//!     .add_method(OptimizationMethod::ParticleSwarm, 0.2)
//!     .add_method(OptimizationMethod::DifferentialEvolution, 0.2)
//!     .parallel_execution(true)
//!     .result_aggregation(AggregationStrategy::BestResult)
//!     .build();
//!
//! let optimizer = GlobalOptimizerBuilder::new()
//!     .portfolio(portfolio)
//!     .enable_benchmarking(true)
//!     .build();
//!
//! let result = optimizer.optimize(&problem)?;
//! ```
//!
//! ## Adaptive Optimization with Learning
//!
//! ```rust
//! // Configure adaptive optimizer with meta-learning
//! let adaptive_config = AdaptiveConfig::builder()
//!     .problem_classification(true)
//!     .meta_learning(true)
//!     .warm_starting(true)
//!     .dynamic_method_switching(true)
//!     .build();
//!
//! let optimizer = GlobalOptimizerBuilder::new()
//!     .adaptive_config(adaptive_config)
//!     .learning_database("optimization_history.db")
//!     .build();
//!
//! // Optimizer learns and adapts based on problem characteristics
//! let result = optimizer.optimize(&problem)?;
//! ```

use crate::core::{OptimizationProblem, Solution, SklResult, SklError, OptimizationResult};
use super::{
    basin_hopping::{BasinHopping, BasinHoppingOptimizer},
    multi_start::{MultiStartMethod, MultiStartOptimizer},
    differential_evolution::{DifferentialEvolution, DifferentialEvolutionOptimizer},
    genetic_algorithms::{GeneticAlgorithm, GeneticAlgorithmOptimizer},
    simulated_annealing::{SimulatedAnnealing, SimulatedAnnealingOptimizer},
    branch_and_bound::{BranchAndBound, BranchAndBoundOptimizer},
    stochastic_methods::{StochasticMethod, StochasticOptimizer},
};
use scirs2_core::ndarray::{Array1, Array2, Axis};
use scirs2_core::random::{Random, rng, DistributionExt};
use std::collections::{HashMap, BTreeMap};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};
use rayon::prelude::*;

/// Core trait for global optimization algorithms.
///
/// This trait provides a unified interface for all global optimization methods,
/// enabling seamless method selection, comparison, and orchestration. All
/// optimization algorithms in the framework implement this trait.
pub trait GlobalOptimizer: Send + Sync {
    /// Optimizes the given problem using the algorithm's strategy.
    ///
    /// # Arguments
    /// * `problem` - The optimization problem to solve
    ///
    /// # Returns
    /// Optimization result with solution and metadata
    fn optimize(&self, problem: &OptimizationProblem) -> SklResult<OptimizationResult>;

    /// Returns the name/identifier of the optimization method.
    fn method_name(&self) -> &'static str;

    /// Provides metadata about the optimization method capabilities.
    fn method_info(&self) -> MethodInfo;

    /// Estimates the computational cost for the given problem.
    ///
    /// # Arguments
    /// * `problem` - Problem to estimate cost for
    ///
    /// # Returns
    /// Estimated computational cost (relative scale)
    fn estimate_cost(&self, problem: &OptimizationProblem) -> f64;

    /// Assesses suitability for the given problem type.
    ///
    /// # Arguments
    /// * `problem_characteristics` - Characteristics of the problem
    ///
    /// # Returns
    /// Suitability score (0.0 to 1.0)
    fn assess_suitability(&self, problem_characteristics: &ProblemCharacteristics) -> f64;

    /// Provides warm-start initialization if supported.
    ///
    /// # Arguments
    /// * `previous_results` - Results from previous optimization runs
    /// * `problem` - Current problem to optimize
    ///
    /// # Returns
    /// Warm-start configuration or None if not supported
    fn warm_start(
        &self,
        previous_results: &[OptimizationResult],
        problem: &OptimizationProblem,
    ) -> Option<WarmStartConfig>;
}

/// Information about an optimization method.
#[derive(Debug, Clone)]
pub struct MethodInfo {
    /// Method category
    pub category: MethodCategory,
    /// Supported problem types
    pub supported_types: Vec<ProblemType>,
    /// Algorithmic complexity
    pub complexity: Complexity,
    /// Required resources
    pub resource_requirements: ResourceRequirements,
    /// Method characteristics
    pub characteristics: MethodCharacteristics,
}

/// Categories of optimization methods.
#[derive(Debug, Clone, PartialEq)]
pub enum MethodCategory {
    /// Evolutionary algorithms
    Evolutionary,
    /// Swarm intelligence
    SwarmIntelligence,
    /// Simulated physical processes
    PhysicalSimulation,
    /// Tree search methods
    TreeSearch,
    /// Stochastic sampling
    StochasticSampling,
    /// Hybrid approaches
    Hybrid,
}

/// Types of optimization problems.
#[derive(Debug, Clone, PartialEq)]
pub enum ProblemType {
    /// Continuous optimization
    Continuous,
    /// Discrete optimization
    Discrete,
    /// Mixed-integer optimization
    MixedInteger,
    /// Constrained optimization
    Constrained,
    /// Multi-objective optimization
    MultiObjective,
    /// Black-box optimization
    BlackBox,
}

/// Algorithmic complexity characteristics.
#[derive(Debug, Clone)]
pub struct Complexity {
    /// Time complexity order
    pub time_complexity: String,
    /// Space complexity order
    pub space_complexity: String,
    /// Scalability with problem dimension
    pub dimension_scalability: ScalabilityLevel,
    /// Parallelization potential
    pub parallelization: ParallelizationLevel,
}

/// Levels of scalability.
#[derive(Debug, Clone, PartialEq)]
pub enum ScalabilityLevel {
    /// Excellent scalability
    Excellent,
    /// Good scalability
    Good,
    /// Moderate scalability
    Moderate,
    /// Limited scalability
    Limited,
    /// Poor scalability
    Poor,
}

/// Levels of parallelization support.
#[derive(Debug, Clone, PartialEq)]
pub enum ParallelizationLevel {
    /// Fully parallelizable
    Full,
    /// Partially parallelizable
    Partial,
    /// Limited parallelization
    Limited,
    /// Sequential only
    Sequential,
}

/// Resource requirements for optimization methods.
#[derive(Debug, Clone)]
pub struct ResourceRequirements {
    /// Memory requirements
    pub memory: MemoryRequirement,
    /// CPU requirements
    pub cpu: CpuRequirement,
    /// Storage requirements
    pub storage: StorageRequirement,
}

/// Memory requirement levels.
#[derive(Debug, Clone, PartialEq)]
pub enum MemoryRequirement {
    /// Low memory usage
    Low,
    /// Moderate memory usage
    Moderate,
    /// High memory usage
    High,
    /// Very high memory usage
    VeryHigh,
}

/// CPU requirement levels.
#[derive(Debug, Clone, PartialEq)]
pub enum CpuRequirement {
    /// Low CPU usage
    Low,
    /// Moderate CPU usage
    Moderate,
    /// High CPU usage
    High,
    /// Intensive CPU usage
    Intensive,
}

/// Storage requirement levels.
#[derive(Debug, Clone, PartialEq)]
pub enum StorageRequirement {
    /// Minimal storage
    Minimal,
    /// Low storage
    Low,
    /// Moderate storage
    Moderate,
    /// High storage
    High,
}

/// Characteristics of optimization methods.
#[derive(Debug, Clone)]
pub struct MethodCharacteristics {
    /// Exploration capability
    pub exploration: f64,
    /// Exploitation capability
    pub exploitation: f64,
    /// Convergence speed
    pub convergence_speed: ConvergenceSpeed,
    /// Solution quality
    pub solution_quality: QualityLevel,
    /// Robustness to noise
    pub noise_robustness: RobustnessLevel,
    /// Parameter sensitivity
    pub parameter_sensitivity: SensitivityLevel,
}

/// Convergence speed levels.
#[derive(Debug, Clone, PartialEq)]
pub enum ConvergenceSpeed {
    /// Very fast convergence
    VeryFast,
    /// Fast convergence
    Fast,
    /// Moderate convergence
    Moderate,
    /// Slow convergence
    Slow,
    /// Very slow convergence
    VerySlow,
}

/// Quality levels for various aspects.
#[derive(Debug, Clone, PartialEq)]
pub enum QualityLevel {
    /// Excellent quality
    Excellent,
    /// Good quality
    Good,
    /// Moderate quality
    Moderate,
    /// Fair quality
    Fair,
    /// Poor quality
    Poor,
}

/// Robustness levels.
#[derive(Debug, Clone, PartialEq)]
pub enum RobustnessLevel {
    /// Very robust
    VeryRobust,
    /// Robust
    Robust,
    /// Moderate robustness
    Moderate,
    /// Sensitive
    Sensitive,
    /// Very sensitive
    VerySensitive,
}

/// Sensitivity levels.
#[derive(Debug, Clone, PartialEq)]
pub enum SensitivityLevel {
    /// Very low sensitivity
    VeryLow,
    /// Low sensitivity
    Low,
    /// Moderate sensitivity
    Moderate,
    /// High sensitivity
    High,
    /// Very high sensitivity
    VeryHigh,
}

/// Characteristics of optimization problems for method selection.
#[derive(Debug, Clone)]
pub struct ProblemCharacteristics {
    /// Problem dimension
    pub dimension: usize,
    /// Problem type
    pub problem_type: ProblemType,
    /// Estimated modality
    pub modality: Modality,
    /// Noise level
    pub noise_level: NoiseLevel,
    /// Constraint density
    pub constraint_density: f64,
    /// Evaluation cost
    pub evaluation_cost: EvaluationCost,
    /// Landscape properties
    pub landscape_properties: LandscapeProperties,
}

/// Modality of the optimization landscape.
#[derive(Debug, Clone, PartialEq)]
pub enum Modality {
    /// Single optimum
    Unimodal,
    /// Few local optima
    FewModal,
    /// Many local optima
    Multimodal,
    /// Extremely complex landscape
    Highly Multimodal,
    /// Unknown modality
    Unknown,
}

/// Noise levels in objective function.
#[derive(Debug, Clone, PartialEq)]
pub enum NoiseLevel {
    /// No noise
    None,
    /// Low noise
    Low,
    /// Moderate noise
    Moderate,
    /// High noise
    High,
    /// Very high noise
    VeryHigh,
}

/// Cost of objective function evaluation.
#[derive(Debug, Clone, PartialEq)]
pub enum EvaluationCost {
    /// Very cheap evaluation
    VeryCheap,
    /// Cheap evaluation
    Cheap,
    /// Moderate cost
    Moderate,
    /// Expensive evaluation
    Expensive,
    /// Very expensive evaluation
    VeryExpensive,
}

/// Properties of the optimization landscape.
#[derive(Debug, Clone)]
pub struct LandscapeProperties {
    /// Smoothness estimate
    pub smoothness: f64,
    /// Gradient availability
    pub gradient_available: bool,
    /// Separability
    pub separability: f64,
    /// Symmetry
    pub symmetry: f64,
    /// Conditioning estimate
    pub conditioning: f64,
}

/// Configuration for warm-starting optimization.
#[derive(Debug, Clone)]
pub struct WarmStartConfig {
    /// Initial population/starting points
    pub initial_points: Vec<Array1<f64>>,
    /// Method-specific parameters
    pub parameters: HashMap<String, f64>,
    /// Confidence in warm-start quality
    pub confidence: f64,
}

/// Available optimization methods.
#[derive(Debug, Clone, PartialEq)]
pub enum OptimizationMethod {
    /// Basin hopping optimization
    BasinHopping,
    /// Multi-start optimization
    MultiStart,
    /// Differential evolution
    DifferentialEvolution,
    /// Genetic algorithm
    GeneticAlgorithm,
    /// Simulated annealing
    SimulatedAnnealing,
    /// Branch and bound
    BranchAndBound,
    /// Particle swarm optimization
    ParticleSwarm,
    /// Random search
    RandomSearch,
    /// Cross-entropy method
    CrossEntropy,
}

/// Portfolio of optimization methods for multi-method optimization.
#[derive(Debug, Clone)]
pub struct MethodPortfolio {
    /// Methods and their resource allocation
    pub methods: Vec<(OptimizationMethod, f64)>,
    /// Parallel execution flag
    pub parallel_execution: bool,
    /// Result aggregation strategy
    pub aggregation_strategy: AggregationStrategy,
    /// Resource distribution
    pub resource_distribution: ResourceDistribution,
}

impl MethodPortfolio {
    /// Creates a builder for method portfolio configuration.
    pub fn builder() -> MethodPortfolioBuilder {
        MethodPortfolioBuilder::default()
    }
}

/// Builder for method portfolio.
#[derive(Debug)]
pub struct MethodPortfolioBuilder {
    methods: Vec<(OptimizationMethod, f64)>,
    parallel_execution: bool,
    aggregation_strategy: AggregationStrategy,
    resource_distribution: ResourceDistribution,
}

impl Default for MethodPortfolioBuilder {
    fn default() -> Self {
        Self {
            methods: Vec::new(),
            parallel_execution: false,
            aggregation_strategy: AggregationStrategy::BestResult,
            resource_distribution: ResourceDistribution::Equal,
        }
    }
}

impl MethodPortfolioBuilder {
    /// Adds a method to the portfolio with resource allocation.
    pub fn add_method(mut self, method: OptimizationMethod, allocation: f64) -> Self {
        self.methods.push((method, allocation));
        self
    }

    /// Enables parallel execution of methods.
    pub fn parallel_execution(mut self, parallel: bool) -> Self {
        self.parallel_execution = parallel;
        self
    }

    /// Sets the result aggregation strategy.
    pub fn result_aggregation(mut self, strategy: AggregationStrategy) -> Self {
        self.aggregation_strategy = strategy;
        self
    }

    /// Sets the resource distribution strategy.
    pub fn resource_distribution(mut self, distribution: ResourceDistribution) -> Self {
        self.resource_distribution = distribution;
        self
    }

    /// Builds the method portfolio.
    pub fn build(self) -> MethodPortfolio {
        MethodPortfolio {
            methods: self.methods,
            parallel_execution: self.parallel_execution,
            aggregation_strategy: self.aggregation_strategy,
            resource_distribution: self.resource_distribution,
        }
    }
}

/// Strategies for aggregating results from multiple methods.
#[derive(Debug, Clone, PartialEq)]
pub enum AggregationStrategy {
    /// Use the best result found
    BestResult,
    /// Weighted combination of results
    WeightedCombination,
    /// Ensemble of top results
    Ensemble { top_k: usize },
    /// Consensus-based selection
    Consensus,
}

/// Strategies for distributing resources among methods.
#[derive(Debug, Clone, PartialEq)]
pub enum ResourceDistribution {
    /// Equal resource allocation
    Equal,
    /// Performance-based allocation
    PerformanceBased,
    /// Adaptive allocation during execution
    Adaptive,
    /// User-defined allocation
    Custom(Vec<f64>),
}

/// Configuration for adaptive optimization behavior.
#[derive(Debug, Clone)]
pub struct AdaptiveConfig {
    /// Enable problem classification
    pub problem_classification: bool,
    /// Enable meta-learning from previous runs
    pub meta_learning: bool,
    /// Enable warm-starting
    pub warm_starting: bool,
    /// Enable dynamic method switching
    pub dynamic_method_switching: bool,
    /// Learning rate for adaptation
    pub learning_rate: f64,
    /// Exploration factor for method selection
    pub exploration_factor: f64,
}

impl AdaptiveConfig {
    /// Creates a builder for adaptive configuration.
    pub fn builder() -> AdaptiveConfigBuilder {
        AdaptiveConfigBuilder::default()
    }
}

/// Builder for adaptive configuration.
#[derive(Debug)]
pub struct AdaptiveConfigBuilder {
    problem_classification: bool,
    meta_learning: bool,
    warm_starting: bool,
    dynamic_method_switching: bool,
    learning_rate: f64,
    exploration_factor: f64,
}

impl Default for AdaptiveConfigBuilder {
    fn default() -> Self {
        Self {
            problem_classification: false,
            meta_learning: false,
            warm_starting: false,
            dynamic_method_switching: false,
            learning_rate: 0.1,
            exploration_factor: 0.1,
        }
    }
}

impl AdaptiveConfigBuilder {
    /// Enables problem classification.
    pub fn problem_classification(mut self, enable: bool) -> Self {
        self.problem_classification = enable;
        self
    }

    /// Enables meta-learning.
    pub fn meta_learning(mut self, enable: bool) -> Self {
        self.meta_learning = enable;
        self
    }

    /// Enables warm-starting.
    pub fn warm_starting(mut self, enable: bool) -> Self {
        self.warm_starting = enable;
        self
    }

    /// Enables dynamic method switching.
    pub fn dynamic_method_switching(mut self, enable: bool) -> Self {
        self.dynamic_method_switching = enable;
        self
    }

    /// Sets the learning rate.
    pub fn learning_rate(mut self, rate: f64) -> Self {
        self.learning_rate = rate;
        self
    }

    /// Sets the exploration factor.
    pub fn exploration_factor(mut self, factor: f64) -> Self {
        self.exploration_factor = factor;
        self
    }

    /// Builds the adaptive configuration.
    pub fn build(self) -> AdaptiveConfig {
        AdaptiveConfig {
            problem_classification: self.problem_classification,
            meta_learning: self.meta_learning,
            warm_starting: self.warm_starting,
            dynamic_method_switching: self.dynamic_method_switching,
            learning_rate: self.learning_rate,
            exploration_factor: self.exploration_factor,
        }
    }
}

/// Comprehensive performance analysis for global optimization.
#[derive(Debug, Clone)]
pub struct GlobalOptimizationAnalysis {
    /// Method comparison results
    pub method_comparison: MethodComparison,
    /// Performance profiling
    pub performance_profile: PerformanceProfile,
    /// Convergence analysis
    pub convergence_analysis: GlobalConvergenceAnalysis,
    /// Resource utilization
    pub resource_utilization: ResourceUtilization,
    /// Recommendations for improvement
    pub recommendations: OptimizationRecommendations,
}

/// Comparison between different optimization methods.
#[derive(Debug, Clone)]
pub struct MethodComparison {
    /// Methods evaluated
    pub methods_tested: Vec<String>,
    /// Performance ranking
    pub performance_ranking: Vec<MethodPerformance>,
    /// Statistical significance tests
    pub significance_tests: StatisticalTests,
    /// Pareto frontier of methods
    pub pareto_frontier: Vec<(String, f64, f64)>, // (method, quality, cost)
}

/// Performance metrics for a single method.
#[derive(Debug, Clone)]
pub struct MethodPerformance {
    /// Method name
    pub method_name: String,
    /// Solution quality
    pub solution_quality: f64,
    /// Convergence speed
    pub convergence_speed: f64,
    /// Computational cost
    pub computational_cost: f64,
    /// Reliability score
    pub reliability: f64,
    /// Overall score
    pub overall_score: f64,
}

/// Statistical tests for method comparison.
#[derive(Debug, Clone)]
pub struct StatisticalTests {
    /// Wilcoxon signed-rank test results
    pub wilcoxon_tests: HashMap<(String, String), f64>,
    /// Friedman test result
    pub friedman_test: Option<f64>,
    /// Effect size measurements
    pub effect_sizes: HashMap<(String, String), f64>,
}

/// Performance profiling of optimization runs.
#[derive(Debug, Clone)]
pub struct PerformanceProfile {
    /// Time breakdown
    pub time_breakdown: TimeBreakdown,
    /// Memory usage patterns
    pub memory_usage: MemoryUsagePattern,
    /// Function evaluation statistics
    pub evaluation_stats: EvaluationStatistics,
    /// Scalability analysis
    pub scalability_analysis: ScalabilityAnalysis,
}

/// Breakdown of time spent in different phases.
#[derive(Debug, Clone)]
pub struct TimeBreakdown {
    /// Time spent in initialization
    pub initialization_time: Duration,
    /// Time spent in main optimization loop
    pub optimization_time: Duration,
    /// Time spent in function evaluations
    pub evaluation_time: Duration,
    /// Time spent in analysis
    pub analysis_time: Duration,
    /// Total runtime
    pub total_time: Duration,
}

/// Memory usage patterns during optimization.
#[derive(Debug, Clone)]
pub struct MemoryUsagePattern {
    /// Peak memory usage
    pub peak_memory: usize,
    /// Average memory usage
    pub average_memory: usize,
    /// Memory usage over time
    pub memory_timeline: Vec<(Duration, usize)>,
    /// Memory efficiency score
    pub efficiency_score: f64,
}

/// Statistics about function evaluations.
#[derive(Debug, Clone)]
pub struct EvaluationStatistics {
    /// Total evaluations performed
    pub total_evaluations: usize,
    /// Unique evaluations (no duplicates)
    pub unique_evaluations: usize,
    /// Evaluations per second
    pub evaluations_per_second: f64,
    /// Cache hit rate
    pub cache_hit_rate: f64,
    /// Evaluation distribution
    pub evaluation_distribution: HashMap<String, usize>,
}

/// Analysis of algorithm scalability.
#[derive(Debug, Clone)]
pub struct ScalabilityAnalysis {
    /// Scalability with problem dimension
    pub dimension_scalability: Vec<(usize, f64)>,
    /// Scalability with population size
    pub population_scalability: Vec<(usize, f64)>,
    /// Parallel efficiency
    pub parallel_efficiency: f64,
    /// Predicted performance for larger problems
    pub performance_predictions: HashMap<usize, f64>,
}

/// Global convergence analysis across methods.
#[derive(Debug, Clone)]
pub struct GlobalConvergenceAnalysis {
    /// Convergence curves for all methods
    pub convergence_curves: HashMap<String, Vec<f64>>,
    /// Convergence rate comparison
    pub convergence_rates: HashMap<String, f64>,
    /// Stagnation analysis
    pub stagnation_analysis: StagnationAnalysis,
    /// Early stopping effectiveness
    pub early_stopping_analysis: EarlyStoppingAnalysis,
}

/// Analysis of optimization stagnation.
#[derive(Debug, Clone)]
pub struct StagnationAnalysis {
    /// Stagnation periods by method
    pub stagnation_periods: HashMap<String, Vec<(usize, usize)>>,
    /// Average stagnation duration
    pub average_stagnation: HashMap<String, f64>,
    /// Stagnation recovery strategies
    pub recovery_strategies: Vec<String>,
}

/// Analysis of early stopping effectiveness.
#[derive(Debug, Clone)]
pub struct EarlyStoppingAnalysis {
    /// Optimal stopping points
    pub optimal_stopping_points: HashMap<String, usize>,
    /// Savings from early stopping
    pub computational_savings: HashMap<String, f64>,
    /// Quality loss from early stopping
    pub quality_loss: HashMap<String, f64>,
}

/// Resource utilization analysis.
#[derive(Debug, Clone)]
pub struct ResourceUtilization {
    /// CPU utilization
    pub cpu_utilization: UtilizationMetrics,
    /// Memory utilization
    pub memory_utilization: UtilizationMetrics,
    /// I/O utilization
    pub io_utilization: UtilizationMetrics,
    /// Overall efficiency score
    pub efficiency_score: f64,
}

/// Utilization metrics for resources.
#[derive(Debug, Clone)]
pub struct UtilizationMetrics {
    /// Average utilization
    pub average: f64,
    /// Peak utilization
    pub peak: f64,
    /// Utilization variance
    pub variance: f64,
    /// Efficiency score
    pub efficiency: f64,
}

/// Recommendations for optimization improvement.
#[derive(Debug, Clone)]
pub struct OptimizationRecommendations {
    /// Method selection recommendations
    pub method_recommendations: Vec<MethodRecommendation>,
    /// Parameter tuning suggestions
    pub parameter_suggestions: HashMap<String, Vec<ParameterSuggestion>>,
    /// Resource allocation recommendations
    pub resource_recommendations: Vec<ResourceRecommendation>,
    /// General improvement suggestions
    pub general_suggestions: Vec<String>,
}

/// Recommendation for method selection.
#[derive(Debug, Clone)]
pub struct MethodRecommendation {
    /// Recommended method
    pub method: String,
    /// Confidence in recommendation
    pub confidence: f64,
    /// Reasons for recommendation
    pub reasons: Vec<String>,
    /// Expected improvement
    pub expected_improvement: f64,
}

/// Suggestion for parameter tuning.
#[derive(Debug, Clone)]
pub struct ParameterSuggestion {
    /// Parameter name
    pub parameter: String,
    /// Current value
    pub current_value: f64,
    /// Suggested value
    pub suggested_value: f64,
    /// Expected impact
    pub expected_impact: f64,
    /// Confidence in suggestion
    pub confidence: f64,
}

/// Recommendation for resource allocation.
#[derive(Debug, Clone)]
pub struct ResourceRecommendation {
    /// Resource type
    pub resource_type: String,
    /// Current allocation
    pub current_allocation: f64,
    /// Recommended allocation
    pub recommended_allocation: f64,
    /// Expected benefit
    pub expected_benefit: f64,
}

/// Main global optimization framework implementation.
#[derive(Debug)]
pub struct UnifiedGlobalOptimizer {
    /// Optimization strategy
    pub strategy: OptimizationStrategy,
    /// Method portfolio (if using portfolio strategy)
    pub portfolio: Option<MethodPortfolio>,
    /// Adaptive configuration
    pub adaptive_config: Option<AdaptiveConfig>,
    /// Method registry
    pub method_registry: Arc<RwLock<MethodRegistry>>,
    /// Performance history
    pub performance_history: Arc<Mutex<Vec<OptimizationResult>>>,
    /// Learning database path
    pub learning_database: Option<String>,
    /// Configuration parameters
    pub config: GlobalOptimizerConfig,
}

/// Strategy for global optimization.
#[derive(Debug, Clone, PartialEq)]
pub enum OptimizationStrategy {
    /// Use a single method
    SingleMethod(OptimizationMethod),
    /// Automatically select best method
    AutoSelect,
    /// Use portfolio of methods
    Portfolio,
    /// Adaptive method selection with learning
    Adaptive,
}

/// Registry of available optimization methods.
#[derive(Debug)]
pub struct MethodRegistry {
    /// Registered optimizers
    optimizers: HashMap<OptimizationMethod, Box<dyn GlobalOptimizer>>,
    /// Method performance database
    performance_db: BTreeMap<String, MethodPerformance>,
    /// Problem-method suitability matrix
    suitability_matrix: HashMap<(ProblemType, OptimizationMethod), f64>,
}

impl MethodRegistry {
    /// Creates a new method registry with default optimizers.
    pub fn new() -> Self {
        let mut registry = Self {
            optimizers: HashMap::new(),
            performance_db: BTreeMap::new(),
            suitability_matrix: HashMap::new(),
        };

        // Register default optimizers would go here
        // registry.register_optimizer(OptimizationMethod::SimulatedAnnealing, Box::new(...));

        registry
    }

    /// Registers a new optimization method.
    pub fn register_optimizer(
        &mut self,
        method: OptimizationMethod,
        optimizer: Box<dyn GlobalOptimizer>,
    ) {
        self.optimizers.insert(method, optimizer);
    }

    /// Gets an optimizer for the specified method.
    pub fn get_optimizer(&self, method: &OptimizationMethod) -> Option<&dyn GlobalOptimizer> {
        self.optimizers.get(method).map(|o| o.as_ref())
    }

    /// Selects the best method for the given problem characteristics.
    pub fn select_best_method(&self, characteristics: &ProblemCharacteristics) -> OptimizationMethod {
        let mut best_method = OptimizationMethod::SimulatedAnnealing;
        let mut best_score = 0.0;

        for (method, optimizer) in &self.optimizers {
            let score = optimizer.assess_suitability(characteristics);
            if score > best_score {
                best_score = score;
                best_method = method.clone();
            }
        }

        best_method
    }
}

/// Configuration for global optimizer.
#[derive(Debug, Clone)]
pub struct GlobalOptimizerConfig {
    /// Maximum total evaluations
    pub max_evaluations: usize,
    /// Target accuracy
    pub target_accuracy: f64,
    /// Maximum runtime
    pub max_runtime: Duration,
    /// Enable benchmarking
    pub enable_benchmarking: bool,
    /// Enable detailed analysis
    pub enable_analysis: bool,
    /// Parallel execution
    pub parallel_execution: bool,
    /// Caching of function evaluations
    pub enable_caching: bool,
}

impl Default for GlobalOptimizerConfig {
    fn default() -> Self {
        Self {
            max_evaluations: 100000,
            target_accuracy: 1e-6,
            max_runtime: Duration::from_secs(3600),
            enable_benchmarking: false,
            enable_analysis: false,
            parallel_execution: false,
            enable_caching: true,
        }
    }
}

impl UnifiedGlobalOptimizer {
    /// Creates a builder for global optimizer configuration.
    pub fn builder() -> GlobalOptimizerBuilder {
        GlobalOptimizerBuilder::default()
    }

    /// Analyzes problem characteristics for method selection.
    pub fn analyze_problem(&self, problem: &OptimizationProblem) -> ProblemCharacteristics {
        // Simplified problem analysis - in practice would be more sophisticated
        ProblemCharacteristics {
            dimension: problem.dimension,
            problem_type: self.infer_problem_type(problem),
            modality: self.estimate_modality(problem),
            noise_level: NoiseLevel::Low,
            constraint_density: 0.0,
            evaluation_cost: self.estimate_evaluation_cost(problem),
            landscape_properties: self.analyze_landscape(problem),
        }
    }

    /// Infers the problem type from the problem definition.
    fn infer_problem_type(&self, problem: &OptimizationProblem) -> ProblemType {
        // Simplified inference - would analyze constraints, variable types, etc.
        ProblemType::Continuous
    }

    /// Estimates the modality of the optimization landscape.
    fn estimate_modality(&self, problem: &OptimizationProblem) -> Modality {
        // Simplified estimation - would sample the landscape
        Modality::Unknown
    }

    /// Estimates the cost of function evaluation.
    fn estimate_evaluation_cost(&self, problem: &OptimizationProblem) -> EvaluationCost {
        // Simplified estimation based on dimension
        match problem.dimension {
            1..=10 => EvaluationCost::Cheap,
            11..=100 => EvaluationCost::Moderate,
            101..=1000 => EvaluationCost::Expensive,
            _ => EvaluationCost::VeryExpensive,
        }
    }

    /// Analyzes landscape properties.
    fn analyze_landscape(&self, problem: &OptimizationProblem) -> LandscapeProperties {
        // Simplified analysis - would sample gradients, Hessians, etc.
        LandscapeProperties {
            smoothness: 0.5,
            gradient_available: false,
            separability: 0.5,
            symmetry: 0.5,
            conditioning: 1.0,
        }
    }
}

impl GlobalOptimizer for UnifiedGlobalOptimizer {
    fn optimize(&self, problem: &OptimizationProblem) -> SklResult<OptimizationResult> {
        let start_time = Instant::now();

        match &self.strategy {
            OptimizationStrategy::SingleMethod(method) => {
                let registry = self.method_registry.read().unwrap();
                if let Some(optimizer) = registry.get_optimizer(method) {
                    optimizer.optimize(problem)
                } else {
                    Err(SklError::InvalidParameter(format!("Method {:?} not available", method)))
                }
            },
            OptimizationStrategy::AutoSelect => {
                let characteristics = self.analyze_problem(problem);
                let registry = self.method_registry.read().unwrap();
                let best_method = registry.select_best_method(&characteristics);

                if let Some(optimizer) = registry.get_optimizer(&best_method) {
                    let mut result = optimizer.optimize(problem)?;
                    // Add metadata about method selection
                    result.metadata = Some(serde_json::json!({
                        "selected_method": format!("{:?}", best_method),
                        "selection_reason": "automatic",
                        "problem_characteristics": characteristics
                    }));
                    Ok(result)
                } else {
                    Err(SklError::InvalidParameter("No suitable method found".to_string()))
                }
            },
            OptimizationStrategy::Portfolio => {
                if let Some(portfolio) = &self.portfolio {
                    self.optimize_with_portfolio(problem, portfolio)
                } else {
                    Err(SklError::InvalidParameter("Portfolio not configured".to_string()))
                }
            },
            OptimizationStrategy::Adaptive => {
                self.optimize_adaptively(problem)
            },
        }
    }

    fn method_name(&self) -> &'static str {
        "UnifiedGlobalOptimizer"
    }

    fn method_info(&self) -> MethodInfo {
        MethodInfo {
            category: MethodCategory::Hybrid,
            supported_types: vec![
                ProblemType::Continuous,
                ProblemType::Discrete,
                ProblemType::MixedInteger,
                ProblemType::Constrained,
                ProblemType::BlackBox,
            ],
            complexity: Complexity {
                time_complexity: "Variable".to_string(),
                space_complexity: "Variable".to_string(),
                dimension_scalability: ScalabilityLevel::Good,
                parallelization: ParallelizationLevel::Full,
            },
            resource_requirements: ResourceRequirements {
                memory: MemoryRequirement::Moderate,
                cpu: CpuRequirement::Moderate,
                storage: StorageRequirement::Low,
            },
            characteristics: MethodCharacteristics {
                exploration: 0.8,
                exploitation: 0.8,
                convergence_speed: ConvergenceSpeed::Moderate,
                solution_quality: QualityLevel::Excellent,
                noise_robustness: RobustnessLevel::Robust,
                parameter_sensitivity: SensitivityLevel::Low,
            },
        }
    }

    fn estimate_cost(&self, problem: &OptimizationProblem) -> f64 {
        // Base cost estimate
        problem.dimension as f64 * 1000.0
    }

    fn assess_suitability(&self, _characteristics: &ProblemCharacteristics) -> f64 {
        // Unified optimizer is generally suitable for most problems
        0.8
    }

    fn warm_start(
        &self,
        previous_results: &[OptimizationResult],
        _problem: &OptimizationProblem,
    ) -> Option<WarmStartConfig> {
        if previous_results.is_empty() {
            return None;
        }

        // Extract best solutions for warm-starting
        let mut best_points = Vec::new();
        for result in previous_results.iter().take(5) {
            best_points.push(result.best_solution.parameters.clone());
        }

        Some(WarmStartConfig {
            initial_points: best_points,
            parameters: HashMap::new(),
            confidence: 0.7,
        })
    }
}

impl UnifiedGlobalOptimizer {
    /// Optimizes using a portfolio of methods.
    fn optimize_with_portfolio(
        &self,
        problem: &OptimizationProblem,
        portfolio: &MethodPortfolio,
    ) -> SklResult<OptimizationResult> {
        let registry = self.method_registry.read().unwrap();

        if portfolio.parallel_execution {
            // Parallel execution of methods
            let results: Result<Vec<_>, _> = portfolio.methods.par_iter()
                .map(|(method, _allocation)| {
                    if let Some(optimizer) = registry.get_optimizer(method) {
                        optimizer.optimize(problem)
                    } else {
                        Err(SklError::InvalidParameter(format!("Method {:?} not available", method)))
                    }
                })
                .collect();

            let results = results?;
            self.aggregate_results(results, &portfolio.aggregation_strategy)
        } else {
            // Sequential execution
            let mut best_result = None;
            let mut all_results = Vec::new();

            for (method, _allocation) in &portfolio.methods {
                if let Some(optimizer) = registry.get_optimizer(method) {
                    let result = optimizer.optimize(problem)?;

                    if best_result.is_none() || result.best_solution.fitness < best_result.as_ref().unwrap().best_solution.fitness {
                        best_result = Some(result.clone());
                    }
                    all_results.push(result);
                }
            }

            best_result.ok_or_else(|| SklError::InvalidParameter("No methods executed successfully".to_string()))
        }
    }

    /// Optimizes using adaptive method selection.
    fn optimize_adaptively(&self, problem: &OptimizationProblem) -> SklResult<OptimizationResult> {
        // Simplified adaptive optimization - would implement learning and adaptation
        let characteristics = self.analyze_problem(problem);
        let registry = self.method_registry.read().unwrap();
        let selected_method = registry.select_best_method(&characteristics);

        if let Some(optimizer) = registry.get_optimizer(&selected_method) {
            optimizer.optimize(problem)
        } else {
            Err(SklError::InvalidParameter("No suitable method found".to_string()))
        }
    }

    /// Aggregates results from multiple optimization runs.
    fn aggregate_results(
        &self,
        results: Vec<OptimizationResult>,
        strategy: &AggregationStrategy,
    ) -> SklResult<OptimizationResult> {
        match strategy {
            AggregationStrategy::BestResult => {
                results.into_iter()
                    .min_by(|a, b| a.best_solution.fitness.partial_cmp(&b.best_solution.fitness).unwrap())
                    .ok_or_else(|| SklError::InvalidParameter("No results to aggregate".to_string()))
            },
            _ => {
                // Simplified - would implement other aggregation strategies
                results.into_iter()
                    .min_by(|a, b| a.best_solution.fitness.partial_cmp(&b.best_solution.fitness).unwrap())
                    .ok_or_else(|| SklError::InvalidParameter("No results to aggregate".to_string()))
            },
        }
    }
}

/// Builder for unified global optimizer.
#[derive(Debug)]
pub struct GlobalOptimizerBuilder {
    strategy: OptimizationStrategy,
    portfolio: Option<MethodPortfolio>,
    adaptive_config: Option<AdaptiveConfig>,
    learning_database: Option<String>,
    config: GlobalOptimizerConfig,
}

impl Default for GlobalOptimizerBuilder {
    fn default() -> Self {
        Self {
            strategy: OptimizationStrategy::AutoSelect,
            portfolio: None,
            adaptive_config: None,
            learning_database: None,
            config: GlobalOptimizerConfig::default(),
        }
    }
}

impl GlobalOptimizerBuilder {
    /// Creates a new builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the optimization strategy.
    pub fn strategy(mut self, strategy: OptimizationStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Enables automatic method selection.
    pub fn auto_select_method(mut self, enable: bool) -> Self {
        if enable {
            self.strategy = OptimizationStrategy::AutoSelect;
        }
        self
    }

    /// Sets the method portfolio.
    pub fn portfolio(mut self, portfolio: MethodPortfolio) -> Self {
        self.strategy = OptimizationStrategy::Portfolio;
        self.portfolio = Some(portfolio);
        self
    }

    /// Sets the adaptive configuration.
    pub fn adaptive_config(mut self, config: AdaptiveConfig) -> Self {
        self.strategy = OptimizationStrategy::Adaptive;
        self.adaptive_config = Some(config);
        self
    }

    /// Sets the learning database path.
    pub fn learning_database(mut self, path: &str) -> Self {
        self.learning_database = Some(path.to_string());
        self
    }

    /// Sets maximum evaluations.
    pub fn max_evaluations(mut self, max_eval: usize) -> Self {
        self.config.max_evaluations = max_eval;
        self
    }

    /// Sets target accuracy.
    pub fn target_accuracy(mut self, accuracy: f64) -> Self {
        self.config.target_accuracy = accuracy;
        self
    }

    /// Sets maximum runtime.
    pub fn max_runtime(mut self, duration: Duration) -> Self {
        self.config.max_runtime = duration;
        self
    }

    /// Enables benchmarking.
    pub fn enable_benchmarking(mut self, enable: bool) -> Self {
        self.config.enable_benchmarking = enable;
        self
    }

    /// Enables detailed analysis.
    pub fn enable_analysis(mut self, enable: bool) -> Self {
        self.config.enable_analysis = enable;
        self
    }

    /// Enables parallel execution.
    pub fn parallel_execution(mut self, enable: bool) -> Self {
        self.config.parallel_execution = enable;
        self
    }

    /// Builds the global optimizer.
    pub fn build(self) -> UnifiedGlobalOptimizer {
        UnifiedGlobalOptimizer {
            strategy: self.strategy,
            portfolio: self.portfolio,
            adaptive_config: self.adaptive_config,
            method_registry: Arc::new(RwLock::new(MethodRegistry::new())),
            performance_history: Arc::new(Mutex::new(Vec::new())),
            learning_database: self.learning_database,
            config: self.config,
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_method_portfolio_builder() {
        let portfolio = MethodPortfolio::builder()
            .add_method(OptimizationMethod::SimulatedAnnealing, 0.4)
            .add_method(OptimizationMethod::GeneticAlgorithm, 0.3)
            .add_method(OptimizationMethod::ParticleSwarm, 0.3)
            .parallel_execution(true)
            .result_aggregation(AggregationStrategy::BestResult)
            .build();

        assert_eq!(portfolio.methods.len(), 3);
        assert!(portfolio.parallel_execution);
        assert_eq!(portfolio.aggregation_strategy, AggregationStrategy::BestResult);
    }

    #[test]
    fn test_adaptive_config_builder() {
        let config = AdaptiveConfig::builder()
            .problem_classification(true)
            .meta_learning(true)
            .warm_starting(false)
            .learning_rate(0.05)
            .exploration_factor(0.2)
            .build();

        assert!(config.problem_classification);
        assert!(config.meta_learning);
        assert!(!config.warm_starting);
        assert_eq!(config.learning_rate, 0.05);
        assert_eq!(config.exploration_factor, 0.2);
    }

    #[test]
    fn test_global_optimizer_builder() {
        let optimizer = UnifiedGlobalOptimizer::builder()
            .auto_select_method(true)
            .max_evaluations(50000)
            .target_accuracy(1e-8)
            .enable_benchmarking(true)
            .parallel_execution(true)
            .build();

        assert_eq!(optimizer.strategy, OptimizationStrategy::AutoSelect);
        assert_eq!(optimizer.config.max_evaluations, 50000);
        assert_eq!(optimizer.config.target_accuracy, 1e-8);
        assert!(optimizer.config.enable_benchmarking);
        assert!(optimizer.config.parallel_execution);
    }

    #[test]
    fn test_method_info_creation() {
        let info = MethodInfo {
            category: MethodCategory::Evolutionary,
            supported_types: vec![ProblemType::Continuous, ProblemType::Discrete],
            complexity: Complexity {
                time_complexity: "O(n^2)".to_string(),
                space_complexity: "O(n)".to_string(),
                dimension_scalability: ScalabilityLevel::Good,
                parallelization: ParallelizationLevel::Partial,
            },
            resource_requirements: ResourceRequirements {
                memory: MemoryRequirement::Moderate,
                cpu: CpuRequirement::High,
                storage: StorageRequirement::Low,
            },
            characteristics: MethodCharacteristics {
                exploration: 0.8,
                exploitation: 0.6,
                convergence_speed: ConvergenceSpeed::Moderate,
                solution_quality: QualityLevel::Good,
                noise_robustness: RobustnessLevel::Robust,
                parameter_sensitivity: SensitivityLevel::Moderate,
            },
        };

        assert_eq!(info.category, MethodCategory::Evolutionary);
        assert_eq!(info.supported_types.len(), 2);
        assert_eq!(info.complexity.dimension_scalability, ScalabilityLevel::Good);
    }

    #[test]
    fn test_problem_characteristics() {
        let characteristics = ProblemCharacteristics {
            dimension: 10,
            problem_type: ProblemType::Continuous,
            modality: Modality::Multimodal,
            noise_level: NoiseLevel::Low,
            constraint_density: 0.2,
            evaluation_cost: EvaluationCost::Moderate,
            landscape_properties: LandscapeProperties {
                smoothness: 0.7,
                gradient_available: false,
                separability: 0.5,
                symmetry: 0.3,
                conditioning: 1.2,
            },
        };

        assert_eq!(characteristics.dimension, 10);
        assert_eq!(characteristics.problem_type, ProblemType::Continuous);
        assert_eq!(characteristics.modality, Modality::Multimodal);
        assert!((characteristics.landscape_properties.smoothness - 0.7).abs() < 1e-10);
    }

    #[test]
    fn test_optimization_strategy() {
        let single_method = OptimizationStrategy::SingleMethod(OptimizationMethod::SimulatedAnnealing);
        let auto_select = OptimizationStrategy::AutoSelect;
        let portfolio = OptimizationStrategy::Portfolio;
        let adaptive = OptimizationStrategy::Adaptive;

        assert_eq!(single_method, OptimizationStrategy::SingleMethod(OptimizationMethod::SimulatedAnnealing));
        assert_eq!(auto_select, OptimizationStrategy::AutoSelect);
        assert_eq!(portfolio, OptimizationStrategy::Portfolio);
        assert_eq!(adaptive, OptimizationStrategy::Adaptive);
    }

    #[test]
    fn test_warm_start_config() {
        let warm_start = WarmStartConfig {
            initial_points: vec![
                array![1.0, 2.0, 3.0],
                array![4.0, 5.0, 6.0],
            ],
            parameters: {
                let mut params = HashMap::new();
                params.insert("temperature".to_string(), 100.0);
                params.insert("cooling_rate".to_string(), 0.95);
                params
            },
            confidence: 0.8,
        };

        assert_eq!(warm_start.initial_points.len(), 2);
        assert_eq!(warm_start.parameters.len(), 2);
        assert_eq!(warm_start.confidence, 0.8);
        assert_eq!(warm_start.parameters.get("temperature"), Some(&100.0));
    }
}