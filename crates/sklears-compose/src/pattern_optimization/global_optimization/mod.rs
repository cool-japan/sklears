//! Global Optimization Framework
//!
//! This module provides a comprehensive global optimization framework with multiple
//! algorithms, intelligent method selection, and advanced analysis capabilities.
//! The framework is designed to handle complex optimization problems across various
//! domains with state-of-the-art algorithms and sophisticated orchestration.
//!
//! # Architecture Overview
//!
//! The global optimization framework is structured into specialized modules:
//!
//! ## Core Framework
//! - **[`core`]**: Central coordination, method selection, and unified interface
//! - **[`basin_hopping`]**: Basin hopping with Monte Carlo acceptance criteria
//! - **[`multi_start`]**: Multi-start optimization with coverage analysis
//!
//! ## Evolutionary and Swarm Methods
//! - **[`differential_evolution`]**: Differential evolution with adaptive parameters
//! - **[`genetic_algorithms`]**: Genetic algorithms with various operators
//! - **[`stochastic_methods`]**: Particle swarm, ant colony, and stochastic sampling
//!
//! ## Advanced Search Methods
//! - **[`simulated_annealing`]**: Simulated annealing with cooling schedules
//! - **[`branch_and_bound`]**: Branch and bound for discrete optimization
//!
//! # Key Features
//!
//! ## Intelligent Method Selection
//! - Automatic algorithm selection based on problem characteristics
//! - Performance-based method recommendation
//! - Adaptive strategy switching during optimization
//!
//! ## Multi-Method Orchestration
//! - Portfolio optimization with parallel execution
//! - Result aggregation and ensemble methods
//! - Resource allocation and load balancing
//!
//! ## Comprehensive Analysis
//! - Performance profiling and benchmarking
//! - Convergence analysis and comparison
//! - Statistical validation of results
//!
//! ## Advanced Features
//! - Meta-learning from optimization history
//! - Warm-starting from previous solutions
//! - Early stopping with intelligent criteria
//!
//! # Quick Start Guide
//!
//! ## Basic Usage with Automatic Method Selection
//!
//! ```rust
//! use sklears_compose::pattern_optimization::global_optimization::{
//!     UnifiedGlobalOptimizer, GlobalOptimizerBuilder
//! };
//! use scirs2_core::ndarray::Array1;
//!
//! // Create optimizer with automatic method selection
//! let optimizer = GlobalOptimizerBuilder::new()
//!     .auto_select_method(true)
//!     .max_evaluations(10000)
//!     .target_accuracy(1e-6)
//!     .build();
//!
//! // Define optimization problem
//! let problem = OptimizationProblem {
//!     dimension: 2,
//!     bounds: vec![(-5.0, 5.0), (-5.0, 5.0)],
//!     objective: Box::new(|x: &Array1<f64>| {
//!         // Rosenbrock function
//!         let a = 1.0;
//!         let b = 100.0;
//!         Ok((a - x[0]).powi(2) + b * (x[1] - x[0].powi(2)).powi(2))
//!     }),
//! };
//!
//! // Optimize with intelligent method selection
//! let result = optimizer.optimize(&problem)?;
//! println!("Best solution: {:?}", result.best_solution.parameters);
//! println!("Objective value: {}", result.best_solution.fitness);
//! ```
//!
//! ## Portfolio Optimization
//!
//! ```rust
//! use sklears_compose::pattern_optimization::global_optimization::{
//!     MethodPortfolio, OptimizationMethod, AggregationStrategy
//! };
//!
//! // Create portfolio with multiple methods
//! let portfolio = MethodPortfolio::builder()
//!     .add_method(OptimizationMethod::SimulatedAnnealing, 0.3)
//!     .add_method(OptimizationMethod::GeneticAlgorithm, 0.3)
//!     .add_method(OptimizationMethod::DifferentialEvolution, 0.2)
//!     .add_method(OptimizationMethod::ParticleSwarm, 0.2)
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
//! ## Specific Algorithm Usage
//!
//! ```rust
//! use sklears_compose::pattern_optimization::global_optimization::{
//!     SimulatedAnnealingOptimizer, ExponentialCooling
//! };
//!
//! // Configure specific algorithm
//! let cooling_schedule = ExponentialCooling::builder()
//!     .cooling_factor(0.95)
//!     .min_temperature(0.001)
//!     .build();
//!
//! let sa_optimizer = SimulatedAnnealingOptimizer::builder()
//!     .cooling_schedule(Box::new(cooling_schedule))
//!     .max_iterations(10000)
//!     .tolerance(1e-6)
//!     .build()?;
//!
//! let result = sa_optimizer.optimize(&problem)?;
//! ```
//!
//! # Supported Optimization Methods
//!
//! ## Evolutionary Algorithms
//! - **Genetic Algorithms**: Traditional and advanced GA variants
//! - **Differential Evolution**: DE with adaptive parameter control
//! - **Evolution Strategies**: CMA-ES and related methods
//!
//! ## Swarm Intelligence
//! - **Particle Swarm Optimization**: PSO with various topologies
//! - **Ant Colony Optimization**: ACO for combinatorial problems
//! - **Artificial Bee Colony**: ABC with scout bee strategies
//!
//! ## Physical Simulation
//! - **Simulated Annealing**: SA with multiple cooling schedules
//! - **Basin Hopping**: Global optimization with local refinement
//! - **Firefly Algorithm**: Bio-inspired optimization
//!
//! ## Tree Search
//! - **Branch and Bound**: Exact methods for discrete problems
//! - **Monte Carlo Tree Search**: MCTS for complex spaces
//!
//! ## Stochastic Methods
//! - **Random Search**: Pure and guided random sampling
//! - **Cross-Entropy Method**: Distribution-based optimization
//! - **Multi-Start Methods**: Parallel local optimization
//!
//! # Performance Considerations
//!
//! ## Method Selection Guidelines
//!
//! | Problem Type | Recommended Methods | Reasoning |
//! |--------------|-------------------|-----------|
//! | Smooth, Unimodal | Simulated Annealing, Multi-Start | Fast convergence |
//! | Multimodal | Genetic Algorithm, PSO | Good exploration |
//! | High-Dimensional | Differential Evolution, CMA-ES | Dimension scaling |
//! | Discrete | Branch and Bound, GA | Discrete handling |
//! | Noisy | Evolutionary Methods | Noise robustness |
//! | Expensive | Efficient Global Optimization | Sample efficiency |
//!
//! ## Scalability
//!
//! - **Low Dimension (≤10)**: All methods suitable
//! - **Medium Dimension (10-100)**: Evolutionary methods preferred
//! - **High Dimension (≥100)**: Specialized high-dim methods
//!
//! ## Parallel Computing
//!
//! Most methods support parallel execution:
//! - Population-based methods: Natural parallelization
//! - Multi-start methods: Parallel independent runs
//! - Portfolio optimization: Parallel method execution
//!
//! # Advanced Usage
//!
//! ## Custom Problem Definition
//!
//! ```rust
//! use sklears_compose::pattern_optimization::global_optimization::*;
//!
//! struct CustomProblem {
//!     // Problem-specific data
//! }
//!
//! impl CustomProblem {
//!     fn evaluate(&self, x: &Array1<f64>) -> f64 {
//!         // Custom objective function
//!         x.iter().map(|&xi| xi.sin()).sum()
//!     }
//! }
//! ```
//!
//! ## Performance Monitoring
//!
//! ```rust
//! let optimizer = GlobalOptimizerBuilder::new()
//!     .enable_benchmarking(true)
//!     .enable_analysis(true)
//!     .build();
//!
//! let result = optimizer.optimize(&problem)?;
//!
//! // Access detailed analysis
//! if let Some(metadata) = result.metadata {
//!     let analysis: GlobalOptimizationAnalysis =
//!         serde_json::from_value(metadata)?;
//!
//!     println!("Method used: {}", analysis.method_comparison.methods_tested[0]);
//!     println!("Convergence rate: {:.4}",
//!         analysis.convergence_analysis.convergence_rates.values().next().unwrap());
//! }
//! ```
//!
//! # Integration with Other Modules
//!
//! The global optimization framework integrates seamlessly with:
//! - **Pattern Recognition**: Optimization of ML hyperparameters
//! - **Neural Networks**: Architecture and weight optimization
//! - **Feature Selection**: Optimal feature subset selection
//! - **Model Selection**: Automated model configuration

// Module declarations
pub mod basin_hopping;
pub mod multi_start;
pub mod differential_evolution;
pub mod genetic_algorithms;
pub mod simulated_annealing;
pub mod branch_and_bound;
pub mod stochastic_methods;
pub mod core;

// Re-export main public interfaces from core module
pub use core::{
    GlobalOptimizer,
    UnifiedGlobalOptimizer,
    GlobalOptimizerBuilder,
    MethodInfo,
    MethodCategory,
    ProblemType,
    ProblemCharacteristics,
    OptimizationMethod,
    OptimizationStrategy,
    MethodPortfolio,
    MethodPortfolioBuilder,
    AdaptiveConfig,
    AdaptiveConfigBuilder,
    AggregationStrategy,
    ResourceDistribution,
    WarmStartConfig,
    GlobalOptimizationAnalysis,
    MethodComparison,
    PerformanceProfile,
    GlobalConvergenceAnalysis,
    ResourceUtilization,
    OptimizationRecommendations,
};

// Re-export specific optimizers
pub use basin_hopping::{
    BasinHopping,
    BasinHoppingOptimizer,
    BasinHoppingBuilder,
    HoppingStrategy,
    TemperatureSchedule,
    BasinAnalysis,
};

pub use multi_start::{
    MultiStartMethod,
    MultiStartOptimizer,
    MultiStartBuilder,
    StartingPointStrategy,
    CoverageAnalysis,
    SolutionClusters,
};

pub use differential_evolution::{
    DifferentialEvolution,
    DifferentialEvolutionOptimizer,
    DifferentialEvolutionBuilder,
    MutationStrategy,
    CrossoverStrategy,
    SelectionStrategy,
    AdaptedParameters,
};

pub use genetic_algorithms::{
    GeneticAlgorithm,
    GeneticAlgorithmOptimizer,
    GeneticAlgorithmBuilder,
    SelectionMethod,
    CrossoverOperator,
    MutationOperator,
    PopulationAnalysis,
};

pub use simulated_annealing::{
    SimulatedAnnealing,
    SimulatedAnnealingOptimizer,
    SimulatedAnnealingBuilder,
    CoolingSchedule,
    LinearCooling,
    ExponentialCooling,
    AdaptiveCooling,
    AcceptanceCriterion,
    AnnealingAnalysis,
};

pub use branch_and_bound::{
    BranchAndBound,
    BranchAndBoundOptimizer,
    BranchAndBoundBuilder,
    TreeNode,
    NodeConstraints,
    SearchStrategy,
    NodeSelection,
    BranchingRule,
    BoundingMethod,
    TreeAnalysis,
};

pub use stochastic_methods::{
    StochasticMethod,
    StochasticOptimizer,
    StochasticOptimizerBuilder,
    SamplingStrategy,
    StochasticMethodType,
    PSOConfig,
    CrossEntropyConfig,
    ConvergenceDetection,
    StochasticAnalysis,
};

// Convenience type aliases for common use cases
pub type DefaultOptimizer = UnifiedGlobalOptimizer;
pub type OptimizationResult = crate::core::OptimizationResult;
pub type OptimizationProblem = crate::core::OptimizationProblem;
pub type Solution = crate::core::Solution;
pub type SklResult<T> = crate::core::SklResult<T>;

/// Creates a default global optimizer with automatic method selection.
///
/// This is a convenience function for quickly creating an optimizer with
/// sensible defaults for most optimization problems.
///
/// # Returns
/// A configured global optimizer ready for use
///
/// # Example
///
/// ```rust
/// use sklears_compose::pattern_optimization::global_optimization::default_optimizer;
///
/// let optimizer = default_optimizer();
/// let result = optimizer.optimize(&problem)?;
/// ```
pub fn default_optimizer() -> UnifiedGlobalOptimizer {
    GlobalOptimizerBuilder::new()
        .auto_select_method(true)
        .max_evaluations(50000)
        .target_accuracy(1e-6)
        .enable_analysis(true)
        .build()
}

/// Creates a fast optimizer optimized for quick results.
///
/// This optimizer prioritizes speed over solution quality, making it
/// suitable for rapid prototyping or time-constrained scenarios.
///
/// # Returns
/// A configured fast optimizer
///
/// # Example
///
/// ```rust
/// use sklears_compose::pattern_optimization::global_optimization::fast_optimizer;
///
/// let optimizer = fast_optimizer();
/// let result = optimizer.optimize(&problem)?;
/// ```
pub fn fast_optimizer() -> UnifiedGlobalOptimizer {
    GlobalOptimizerBuilder::new()
        .strategy(OptimizationStrategy::SingleMethod(OptimizationMethod::SimulatedAnnealing))
        .max_evaluations(5000)
        .target_accuracy(1e-3)
        .build()
}

/// Creates a thorough optimizer optimized for solution quality.
///
/// This optimizer prioritizes solution quality over speed, using multiple
/// methods and extensive analysis to find the best possible solution.
///
/// # Returns
/// A configured thorough optimizer
///
/// # Example
///
/// ```rust
/// use sklears_compose::pattern_optimization::global_optimization::thorough_optimizer;
///
/// let optimizer = thorough_optimizer();
/// let result = optimizer.optimize(&problem)?;
/// ```
pub fn thorough_optimizer() -> UnifiedGlobalOptimizer {
    let portfolio = MethodPortfolio::builder()
        .add_method(OptimizationMethod::SimulatedAnnealing, 0.25)
        .add_method(OptimizationMethod::GeneticAlgorithm, 0.25)
        .add_method(OptimizationMethod::DifferentialEvolution, 0.25)
        .add_method(OptimizationMethod::ParticleSwarm, 0.25)
        .parallel_execution(true)
        .result_aggregation(AggregationStrategy::BestResult)
        .build();

    GlobalOptimizerBuilder::new()
        .portfolio(portfolio)
        .max_evaluations(200000)
        .target_accuracy(1e-8)
        .enable_benchmarking(true)
        .enable_analysis(true)
        .parallel_execution(true)
        .build()
}

/// Creates an adaptive optimizer with meta-learning capabilities.
///
/// This optimizer learns from previous optimization runs and adapts
/// its strategy based on problem characteristics and historical performance.
///
/// # Arguments
/// * `learning_database` - Optional path to learning database
///
/// # Returns
/// A configured adaptive optimizer
///
/// # Example
///
/// ```rust
/// use sklears_compose::pattern_optimization::global_optimization::adaptive_optimizer;
///
/// let optimizer = adaptive_optimizer(Some("optimization_history.db"));
/// let result = optimizer.optimize(&problem)?;
/// ```
pub fn adaptive_optimizer(learning_database: Option<&str>) -> UnifiedGlobalOptimizer {
    let adaptive_config = AdaptiveConfig::builder()
        .problem_classification(true)
        .meta_learning(true)
        .warm_starting(true)
        .dynamic_method_switching(true)
        .learning_rate(0.1)
        .exploration_factor(0.2)
        .build();

    let mut builder = GlobalOptimizerBuilder::new()
        .adaptive_config(adaptive_config)
        .max_evaluations(100000)
        .target_accuracy(1e-6)
        .enable_analysis(true);

    if let Some(db_path) = learning_database {
        builder = builder.learning_database(db_path);
    }

    builder.build()
}

/// Utility function to create a problem-specific optimizer.
///
/// This function analyzes the problem characteristics and creates an
/// optimizer specifically tailored for the given problem type.
///
/// # Arguments
/// * `problem` - The optimization problem to analyze
///
/// # Returns
/// A configured optimizer tailored for the problem
///
/// # Example
///
/// ```rust
/// use sklears_compose::pattern_optimization::global_optimization::problem_specific_optimizer;
///
/// let optimizer = problem_specific_optimizer(&problem);
/// let result = optimizer.optimize(&problem)?;
/// ```
pub fn problem_specific_optimizer(problem: &OptimizationProblem) -> UnifiedGlobalOptimizer {
    // Analyze problem characteristics
    let dimension = problem.dimension;
    let bounds_range = problem.bounds.iter()
        .map(|(lower, upper)| upper - lower)
        .fold(0.0, |acc, range| acc + range) / problem.bounds.len() as f64;

    // Select strategy based on problem characteristics
    let strategy = if dimension <= 5 {
        // Low dimensional: use thorough search
        OptimizationStrategy::SingleMethod(OptimizationMethod::SimulatedAnnealing)
    } else if dimension <= 20 {
        // Medium dimensional: use evolutionary methods
        OptimizationStrategy::SingleMethod(OptimizationMethod::DifferentialEvolution)
    } else if dimension <= 100 {
        // High dimensional: use specialized methods
        OptimizationStrategy::SingleMethod(OptimizationMethod::GeneticAlgorithm)
    } else {
        // Very high dimensional: use portfolio approach
        OptimizationStrategy::Portfolio
    };

    let mut builder = GlobalOptimizerBuilder::new()
        .strategy(strategy)
        .max_evaluations(std::cmp::min(dimension * 10000, 500000))
        .target_accuracy(1e-6);

    // Add portfolio for high-dimensional problems
    if dimension > 100 {
        let portfolio = MethodPortfolio::builder()
            .add_method(OptimizationMethod::DifferentialEvolution, 0.4)
            .add_method(OptimizationMethod::GeneticAlgorithm, 0.3)
            .add_method(OptimizationMethod::ParticleSwarm, 0.3)
            .parallel_execution(true)
            .build();
        builder = builder.portfolio(portfolio);
    }

    builder.build()
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{Array1, array};

    /// Test problem: Sphere function
    fn create_sphere_problem(dimension: usize) -> OptimizationProblem {
        OptimizationProblem {
            dimension,
            bounds: vec![(-5.0, 5.0); dimension],
            objective: Box::new(|x: &Array1<f64>| {
                Ok(x.iter().map(|xi| xi * xi).sum::<f64>())
            }),
        }
    }

    /// Test problem: Rosenbrock function
    fn create_rosenbrock_problem() -> OptimizationProblem {
        OptimizationProblem {
            dimension: 2,
            bounds: vec![(-5.0, 5.0), (-5.0, 5.0)],
            objective: Box::new(|x: &Array1<f64>| {
                let a = 1.0;
                let b = 100.0;
                Ok((a - x[0]).powi(2) + b * (x[1] - x[0].powi(2)).powi(2))
            }),
        }
    }

    #[test]
    fn test_default_optimizer() {
        let optimizer = default_optimizer();
        let problem = create_sphere_problem(2);

        // Test that optimizer can be created and basic optimization works
        let result = optimizer.optimize(&problem);
        assert!(result.is_ok());

        let result = result.unwrap();
        assert!(result.best_solution.fitness >= 0.0); // Sphere function is non-negative
    }

    #[test]
    fn test_fast_optimizer() {
        let optimizer = fast_optimizer();
        let problem = create_sphere_problem(3);

        let result = optimizer.optimize(&problem);
        assert!(result.is_ok());

        let result = result.unwrap();
        assert!(result.best_solution.parameters.len() == 3);
    }

    #[test]
    fn test_problem_specific_optimizer() {
        // Test low-dimensional problem
        let problem_2d = create_sphere_problem(2);
        let optimizer_2d = problem_specific_optimizer(&problem_2d);
        let result_2d = optimizer_2d.optimize(&problem_2d);
        assert!(result_2d.is_ok());

        // Test medium-dimensional problem
        let problem_10d = create_sphere_problem(10);
        let optimizer_10d = problem_specific_optimizer(&problem_10d);
        let result_10d = optimizer_10d.optimize(&problem_10d);
        assert!(result_10d.is_ok());
    }

    #[test]
    fn test_method_portfolio_creation() {
        let portfolio = MethodPortfolio::builder()
            .add_method(OptimizationMethod::SimulatedAnnealing, 0.5)
            .add_method(OptimizationMethod::GeneticAlgorithm, 0.5)
            .parallel_execution(true)
            .build();

        assert_eq!(portfolio.methods.len(), 2);
        assert!(portfolio.parallel_execution);
    }

    #[test]
    fn test_adaptive_config_creation() {
        let config = AdaptiveConfig::builder()
            .problem_classification(true)
            .meta_learning(true)
            .learning_rate(0.05)
            .build();

        assert!(config.problem_classification);
        assert!(config.meta_learning);
        assert_eq!(config.learning_rate, 0.05);
    }

    #[test]
    fn test_optimization_strategies() {
        let single_method = OptimizationStrategy::SingleMethod(OptimizationMethod::SimulatedAnnealing);
        let auto_select = OptimizationStrategy::AutoSelect;
        let portfolio = OptimizationStrategy::Portfolio;

        // Test that strategies can be created and compared
        assert_eq!(single_method, OptimizationStrategy::SingleMethod(OptimizationMethod::SimulatedAnnealing));
        assert_eq!(auto_select, OptimizationStrategy::AutoSelect);
        assert_eq!(portfolio, OptimizationStrategy::Portfolio);
    }

    #[test]
    fn test_module_exports() {
        // Test that main types are accessible
        let _optimizer: UnifiedGlobalOptimizer = default_optimizer();
        let _method = OptimizationMethod::SimulatedAnnealing;
        let _strategy = OptimizationStrategy::AutoSelect;

        // Test that specific optimizers are accessible
        let _sa_builder = SimulatedAnnealingBuilder::default();
        let _ga_builder = GeneticAlgorithmBuilder::default();
        let _de_builder = DifferentialEvolutionBuilder::default();
    }

    #[test]
    fn test_convenience_functions() {
        let problem = create_rosenbrock_problem();

        // Test default optimizer
        let default_opt = default_optimizer();
        assert!(default_opt.optimize(&problem).is_ok());

        // Test fast optimizer
        let fast_opt = fast_optimizer();
        assert!(fast_opt.optimize(&problem).is_ok());

        // Test problem-specific optimizer
        let specific_opt = problem_specific_optimizer(&problem);
        assert!(specific_opt.optimize(&problem).is_ok());
    }

    #[test]
    fn test_type_aliases() {
        // Test that type aliases work correctly
        let problem: OptimizationProblem = create_sphere_problem(2);
        let optimizer: DefaultOptimizer = default_optimizer();
        let result: SklResult<OptimizationResult> = optimizer.optimize(&problem);

        assert!(result.is_ok());
        let result = result.unwrap();
        let _solution: Solution = result.best_solution;
    }
}