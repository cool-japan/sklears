//! Multi-Output Learning Optimization Framework
//!
//! This module provides a comprehensive suite of optimization algorithms for multi-output
//! learning problems. It has been refactored from a monolithic 2697-line file into 5
//! specialized modules for improved maintainability and focused functionality.
//!
//! ## Module Organization
//!
//! - **joint_loss_optimization**: Joint loss optimization with multiple loss function combinations
//! - **multi_objective_optimization**: Genetic algorithm-based Pareto optimization
//! - **scalarization_methods**: Scalarization techniques for multi-objective problems
//! - **nsga2_algorithms**: NSGA-II evolutionary algorithms with advanced operators
//! - **evolutionary_multi_objective**: Complete NSGA-II implementation with SBX crossover and polynomial mutation
//! - **tests**: Comprehensive integration test suite
//!
//! ## Key Features
//!
//! - **Joint Loss Functions**: MSE, MAE, Huber, Cross-entropy, and Hinge losses
//! - **Loss Combination Strategies**: Sum, weighted sum, max, geometric mean, and adaptive
//! - **Multi-Objective Optimization**: Pareto-optimal solution discovery
//! - **Scalarization Methods**: Weighted sum, epsilon-constraint, and Tchebycheff approaches
//! - **Advanced Evolutionary Algorithms**: NSGA-II with SBX crossover and polynomial mutation
//! - **Performance Monitoring**: Convergence tracking with hypervolume indicator

pub mod evolutionary_multi_objective;
pub mod joint_loss_optimization;
pub mod multi_objective_optimization;
pub mod nsga2_algorithms;
pub mod scalarization_methods;

#[allow(non_snake_case)]
#[cfg(test)]
pub mod tests;

// Re-export main public items for backward compatibility
pub use joint_loss_optimization::{
    JointLossConfig, JointLossOptimizer, JointLossOptimizerTrained, LossCombination, LossFunction,
};

pub use multi_objective_optimization::{
    MultiObjectiveConfig, MultiObjectiveOptimizer, MultiObjectiveOptimizerTrained, ParetoSolution,
};

pub use scalarization_methods::{
    ScalarizationConfig, ScalarizationMethod, ScalarizationOptimizer, ScalarizationOptimizerTrained,
};

pub use nsga2_algorithms::{NSGA2Algorithm, NSGA2Config, NSGA2Optimizer, NSGA2OptimizerTrained};

pub use evolutionary_multi_objective::{GenerationStats, Individual, OptimizationResult, NSGAII};
