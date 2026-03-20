//! Hyperparameter optimization for SVM algorithms
//!
//! This module provides comprehensive hyperparameter optimization utilities for SVM models,
//! including grid search, random search, and Bayesian optimization approaches.
//!
//! Methods included:
//! - Grid Search: Exhaustive search over parameter grid
//! - Random Search: Random sampling from parameter distributions
//! - Bayesian Optimization: Gaussian Process-based optimization
//! - Evolutionary Algorithms: Genetic algorithm for parameter search
//! - Cross-validation integration for robust evaluation

pub mod bayesian_optimization;
pub mod evolutionary_optimization;
pub mod grid_search;
pub mod random_search;

pub use bayesian_optimization::BayesianOptimizationCV;
pub use evolutionary_optimization::EvolutionaryOptimizationCV;
pub use grid_search::GridSearchCV;
pub use random_search::RandomSearchCV;

use crate::kernels::KernelType;

/// Parameter specification for optimization
#[derive(Debug, Clone)]
pub enum ParameterSpec {
    /// Fixed value
    Fixed(f64),
    /// Uniform distribution between min and max
    Uniform { min: f64, max: f64 },
    /// Log-uniform distribution between min and max
    LogUniform { min: f64, max: f64 },
    /// Discrete choices
    Choice(Vec<f64>),
    /// Kernel choices
    KernelChoice(Vec<KernelType>),
}

/// Hyperparameter search space
#[derive(Debug, Clone)]
pub struct SearchSpace {
    /// Regularization parameter C
    pub c: ParameterSpec,
    /// Kernel parameter gamma (for RBF kernel)
    pub gamma: Option<ParameterSpec>,
    /// Polynomial degree (for polynomial kernel)
    pub degree: Option<ParameterSpec>,
    /// Kernel coefficient (for polynomial/sigmoid kernels)
    pub coef0: Option<ParameterSpec>,
    /// Kernel type
    pub kernel: Option<ParameterSpec>,
    /// Tolerance
    pub tol: Option<ParameterSpec>,
    /// Maximum iterations
    pub max_iter: Option<ParameterSpec>,
}

impl Default for SearchSpace {
    fn default() -> Self {
        Self {
            c: ParameterSpec::LogUniform {
                min: 1e-3,
                max: 1e3,
            },
            gamma: Some(ParameterSpec::LogUniform {
                min: 1e-4,
                max: 1e2,
            }),
            degree: Some(ParameterSpec::Choice(vec![2.0, 3.0, 4.0, 5.0])),
            coef0: Some(ParameterSpec::Uniform { min: 0.0, max: 1.0 }),
            kernel: Some(ParameterSpec::KernelChoice(vec![
                KernelType::Linear,
                KernelType::Rbf { gamma: 1.0 },
                KernelType::Polynomial {
                    gamma: 1.0,
                    degree: 3.0,
                    coef0: 1.0,
                },
            ])),
            tol: Some(ParameterSpec::LogUniform {
                min: 1e-6,
                max: 1e-2,
            }),
            max_iter: Some(ParameterSpec::Choice(vec![100.0, 500.0, 1000.0, 5000.0])),
        }
    }
}

/// Configuration for optimization algorithms
#[derive(Debug, Clone)]
pub struct OptimizationConfig {
    /// Number of iterations for optimization
    pub n_iterations: usize,
    /// Number of cross-validation folds
    pub cv_folds: usize,
    /// Scoring metric
    pub scoring: ScoringMetric,
    /// Random seed for reproducibility
    pub random_state: Option<u64>,
    /// Parallel execution
    pub n_jobs: Option<usize>,
    /// Verbose output
    pub verbose: bool,
    /// Early stopping patience
    pub early_stopping_patience: Option<usize>,
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            n_iterations: 100,
            cv_folds: 5,
            scoring: ScoringMetric::Accuracy,
            random_state: Some(42),
            n_jobs: None,
            verbose: false,
            early_stopping_patience: Some(10),
        }
    }
}

/// Scoring metrics for evaluation
#[derive(Debug, Clone)]
pub enum ScoringMetric {
    Accuracy,
    Precision,
    Recall,
    F1Score,
    AUC,
    MeanSquaredError,
    MeanAbsoluteError,
    R2Score,
}

/// Parameter set for SVM
#[derive(Debug, Clone)]
pub struct ParameterSet {
    pub c: f64,
    pub kernel: KernelType,
    pub tol: f64,
    pub max_iter: usize,
}

impl ParameterSet {
    pub fn new() -> Self {
        Self {
            c: 1.0,
            kernel: KernelType::Rbf { gamma: 1.0 },
            tol: 1e-3,
            max_iter: 1000,
        }
    }
}

impl Default for ParameterSet {
    fn default() -> Self {
        Self::new()
    }
}

/// Optimization result
#[derive(Debug, Clone)]
pub struct OptimizationResult {
    /// Best parameter set found
    pub best_params: ParameterSet,
    /// Best cross-validation score
    pub best_score: f64,
    /// All parameter sets and scores tried
    pub cv_results: Vec<(ParameterSet, f64)>,
    /// Number of iterations performed
    pub n_iterations: usize,
    /// Total optimization time
    pub optimization_time: f64,
    /// Convergence history
    pub score_history: Vec<f64>,
}
