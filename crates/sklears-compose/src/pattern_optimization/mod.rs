//! Pattern optimization module providing comprehensive optimization capabilities
//!
//! This module provides a complete optimization framework for pattern recognition,
//! machine learning, and general optimization problems. It includes:
//!
//! ## Core Components
//! - [`PatternOptimizationEngine`] - Main optimization engine orchestrating all algorithms
//! - [`OptimizationProblem`] - Problem definition with objectives and constraints
//! - [`Solution`] - High-performance solution representation with SIMD operations
//! - [`OptimizationResult`] - Comprehensive optimization results with metrics
//!
//! ## Optimization Paradigms
//!
//! ### Multi-Objective Optimization
//! - Pareto front management and non-dominated sorting
//! - Scalarization methods and preference modeling
//! - Multi-objective evolutionary algorithms (NSGA-II, SPEA2, etc.)
//!
//! ### Mathematical Programming
//! - Linear programming with simplex and interior point methods
//! - Quadratic programming with active set and barrier methods
//! - Nonlinear programming with SQP and trust region methods
//!
//! ### Metaheuristic Algorithms
//! - Genetic algorithms with advanced operators
//! - Particle swarm optimization and differential evolution
//! - Simulated annealing with adaptive cooling schedules
//!
//! ### Gradient-Based Methods
//! - Classical gradient descent with momentum and adaptive learning rates
//! - Quasi-Newton methods (BFGS, L-BFGS) with line search
//! - Trust region methods with adaptive radius adjustment
//!
//! ### Constraint Handling
//! - Penalty methods with adaptive penalty parameters
//! - Barrier methods for inequality constraints
//! - Augmented Lagrangian and SQP for general constraints
//!
//! ### Global Optimization
//! - Basin hopping with acceptance criteria and temperature control
//! - Multi-start methods with diversification strategies
//! - Branch-and-bound for discrete and continuous problems
//!
//! ### Online and Streaming Optimization
//! - Online gradient descent with regret bounds
//! - Multi-armed bandit algorithms with exploration strategies
//! - Streaming optimization with concept drift detection
//!
//! ### Distributed Optimization
//! - Federated optimization with privacy preservation
//! - Consensus algorithms for distributed coordination
//! - Load balancing and fault tolerance mechanisms
//!
//! ## Performance Features
//! - SIMD-accelerated fitness calculations and distance metrics
//! - GPU acceleration for large-scale problems
//! - Memory-efficient operations for streaming data
//! - Parallel execution with load balancing
//!
//! ## Analysis and Validation
//! - Convergence analysis with multiple detection algorithms
//! - Sensitivity analysis for parameter robustness
//! - Configuration validation with performance estimation
//! - Comprehensive metrics collection and reporting
//!
//! ## Example Usage
//!
//! ```rust
//! use sklears_compose::pattern_optimization::{
//!     PatternOptimizationEngine, OptimizationProblem, ProblemType
//! };
//! use scirs2_core::ndarray::Array1;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Create optimization engine
//! let mut engine = PatternOptimizationEngine::new();
//!
//! // Define optimization problem
//! let problem = OptimizationProblem {
//!     problem_id: "test_problem".to_string(),
//!     problem_type: ProblemType::NonlinearProgramming,
//!     dimension: 10,
//!     objectives: vec![], // Add objective functions
//!     constraints: vec![], // Add constraints
//!     bounds: (Array1::zeros(10), Array1::ones(10)),
//!     ..Default::default()
//! };
//!
//! // Optimize
//! let result = engine.optimize(&problem)?;
//! println!("Best solution: {:?}", result.best_solution);
//! println!("Objective value: {}", result.best_objective_value);
//! # Ok(())
//! # }
//! ```

// Core optimization engine and fundamental types
pub mod optimization_core;
pub use optimization_core::*;

// Multi-objective optimization and Pareto front management
pub mod multi_objective;
pub use multi_objective::*;

// Classical mathematical programming solvers
pub mod mathematical_optimizers;
pub use mathematical_optimizers::*;

// Nature-inspired and population-based algorithms
pub mod metaheuristic_optimizers;
pub use metaheuristic_optimizers::*;

// Gradient-based optimization methods
pub mod gradient_optimizers;
pub use gradient_optimizers::*;

// Constraint handling techniques
pub mod constraint_handling;
pub use constraint_handling::*;

// Global optimization strategies
pub mod global_optimization;
pub use global_optimization::*;

// Online and streaming optimization
pub mod online_optimization;
pub use online_optimization::*;

// Distributed and federated optimization
pub mod distributed_optimization;
pub use distributed_optimization::*;

// Solution management with SIMD acceleration
pub mod solution_management;
pub use solution_management::*;

// Performance analysis and convergence metrics
pub mod analysis_metrics;
pub use analysis_metrics::*;

// Configuration management and validation
pub mod configuration_validation;
pub use configuration_validation::*;

// Re-export commonly used types for convenience
pub use optimization_core::{
    PatternOptimizationEngine,
    OptimizationEngine,
    OptimizationProblem,
    OptimizationResult,
    Solution as CoreSolution,
    ObjectiveFunction,
    ConstraintFunction,
    ProblemType,
    OptimizationStatus,
};

pub use multi_objective::{
    MultiObjectiveOptimizer,
    MOOAlgorithm,
    ParetoFrontManager,
    ScalarizationMethod,
    ParetoSolution,
    ReferencePointMethod,
};

pub use mathematical_optimizers::{
    MathematicalOptimizer,
    LinearSolver,
    QuadraticSolver,
    NonlinearSolver,
    SimplexMethod,
    InteriorPointMethod,
    SequentialQuadraticProgramming,
};

pub use metaheuristic_optimizers::{
    MetaheuristicOptimizer,
    GeneticAlgorithm,
    SwarmAlgorithm,
    DifferentialEvolution,
    ParticleSwarmOptimization,
    SimulatedAnnealing,
    Individual,
    Population,
    GeneticOperators,
};

pub use gradient_optimizers::{
    GradientBasedOptimizer,
    GradientDescentAlgorithm,
    QuasiNewtonMethod,
    TrustRegionMethod,
    LineSearchMethod,
    GradientDescentVariant,
    BFGSOptimizer,
    LBFGSOptimizer,
};

pub use constraint_handling::{
    ConstraintOptimizer,
    PenaltyMethod,
    BarrierMethod,
    AugmentedLagrangian,
    ConstraintType,
    ConstraintViolation,
    LagrangeMultipliers,
};

pub use global_optimization::{
    GlobalOptimizer,
    BasinHopping,
    MultiStartMethod,
    BranchAndBound,
    StochasticMethod,
    HoppingParameters,
    SolutionCluster,
    BranchNode,
    StochasticParameters,
};

pub use online_optimization::{
    OnlineOptimizer,
    OnlineLearningAlgorithm,
    BanditAlgorithm,
    StreamingOptimizer,
    OnlineGradientDescent,
    MultiArmedBandit,
    ConceptDriftDetector,
    RegretBound,
    ExplorationStrategy,
};

pub use distributed_optimization::{
    DistributedOptimizer,
    FederatedOptimizer,
    ConsensusAlgorithm,
    CoordinationProtocol,
    ParticipantInfo,
    LocalUpdate,
    GlobalUpdate,
    Proposal,
    Vote,
    ConsensusResponse,
};

pub use solution_management::{
    Solution,
    SolutionArchive,
    SolutionValidator,
    ArchiveStatistics,
    DiversitySettings,
    ValidationRule,
    ValidationRuleType,
    ValidationSeverity,
    QualityThresholds,
    StatisticalChecks,
};

pub use analysis_metrics::{
    OptimizationMetrics,
    PerformanceMetrics,
    ConvergenceAnalyzer,
    ConvergenceMetrics,
    SensitivityAnalyzer,
    SensitivityResult,
    RobustnessAssessment,
    ResourceMetrics,
    QualityMetrics,
    StatisticalMetrics,
    ConvergenceDetector,
    SensitivityMethod,
    TrendDirection,
    ConvergenceStatus,
};

pub use configuration_validation::{
    OptimizerConfiguration,
    ConfigValue,
    StoppingCriteria,
    PerformanceSettings,
    LoggingSettings,
    ParallelSettings,
    ValidationSettings,
    OptimizationValidator,
    ConfigValidation,
    ConfigValidationIssue,
    PrecisionLevel,
    LogLevel,
    LogFrequency,
    LoadBalancingStrategy,
    CommunicationProtocol,
    IssueSeverity,
    IssueCategory,
    ValidationRule as ConfigValidationRule,
    ValidationRuleType as ConfigValidationRuleType,
};