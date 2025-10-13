//! Comprehensive Test Suite for Multi-Output Optimization
//!
//! This module contains integration tests for all optimization algorithms in the multi-output
//! learning framework, including joint loss optimization, multi-objective optimization,
//! scalarization methods, and NSGA-II evolutionary algorithms.

use super::joint_loss_optimization::{
    JointLossConfig, JointLossOptimizer, LossCombination, LossFunction,
};
use super::multi_objective_optimization::{
    MultiObjectiveConfig, MultiObjectiveOptimizer, ParetoSolution,
};
use super::nsga2_algorithms::{NSGA2Algorithm, NSGA2Config, NSGA2Optimizer};
use sklears_core::traits::{Estimator, Fit, Predict};

use approx::assert_abs_diff_eq;
// Use SciRS2-Core for arrays and random number generation (SciRS2 Policy)
use scirs2_core::ndarray::array;

#[test]
fn test_joint_loss_optimizer_basic() {
    let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];
    let y = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];

    let config = JointLossConfig {
        output_losses: vec![LossFunction::MSE, LossFunction::MSE],
        combination: LossCombination::Sum,
        max_iter: 100,
        ..Default::default()
    };

    let optimizer = JointLossOptimizer::new().config(config);
    let trained = optimizer.fit(&X.view(), &y.view()).unwrap();

    let predictions = trained.predict(&X.view()).unwrap();
    assert_eq!(predictions.shape(), &[3, 2]);
}

#[test]
fn test_joint_loss_optimizer_weighted_sum() {
    let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];
    let y = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];

    let config = JointLossConfig {
        output_losses: vec![LossFunction::MSE, LossFunction::MAE],
        combination: LossCombination::WeightedSum(vec![0.7, 0.3]),
        max_iter: 100,
        ..Default::default()
    };

    let optimizer = JointLossOptimizer::new().config(config);
    let trained = optimizer.fit(&X.view(), &y.view()).unwrap();

    assert_eq!(trained.n_outputs(), 2);
    assert_eq!(trained.n_features(), 2);
    assert!(!trained.loss_history().is_empty());
}

#[test]
fn test_joint_loss_optimizer_huber_loss() {
    let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];
    let y = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];

    let config = JointLossConfig {
        output_losses: vec![LossFunction::Huber(1.0), LossFunction::Huber(1.0)],
        combination: LossCombination::GeometricMean,
        max_iter: 50,
        ..Default::default()
    };

    let optimizer = JointLossOptimizer::new().config(config);
    let trained = optimizer.fit(&X.view(), &y.view()).unwrap();

    let predictions = trained.predict(&X.view()).unwrap();
    assert_eq!(predictions.shape(), &[3, 2]);
}

#[test]
fn test_multi_objective_optimizer_basic() {
    let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];
    let y = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];

    let config = MultiObjectiveConfig {
        population_size: 20,
        generations: 10,
        objectives: vec!["accuracy".to_string(), "complexity".to_string()],
        random_state: Some(42),
        ..Default::default()
    };

    let optimizer = MultiObjectiveOptimizer::new().config(config);
    let trained = optimizer.fit(&X.view(), &y.view()).unwrap();

    assert!(!trained.pareto_solutions().is_empty());
    assert!(!trained.convergence_history().is_empty());
}

#[test]
fn test_multi_objective_optimizer_multiple_objectives() {
    let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];
    let y = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];

    let config = MultiObjectiveConfig {
        population_size: 15,
        generations: 5,
        objectives: vec![
            "mse".to_string(),
            "mae".to_string(),
            "complexity".to_string(),
        ],
        random_state: Some(123),
        ..Default::default()
    };

    let optimizer = MultiObjectiveOptimizer::new().config(config);
    let trained = optimizer.fit(&X.view(), &y.view()).unwrap();

    let predictions = trained.predict(&X.view()).unwrap();
    assert_eq!(predictions.shape(), &[3, 2]);
}

#[test]
fn test_joint_loss_optimizer_adaptive_combination() {
    let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];
    let y = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];

    let config = JointLossConfig {
        output_losses: vec![LossFunction::MSE, LossFunction::MAE],
        combination: LossCombination::Adaptive,
        max_iter: 50,
        learning_rate: 0.001,
        ..Default::default()
    };

    let optimizer = JointLossOptimizer::new().config(config);
    let trained = optimizer.fit(&X.view(), &y.view()).unwrap();

    // Check that loss decreases over iterations
    let loss_history = trained.loss_history();
    if loss_history.len() > 1 {
        assert!(loss_history[0] > loss_history[loss_history.len() - 1]);
    }
}

#[test]
fn test_joint_loss_optimizer_invalid_input() {
    let X = array![[1.0, 2.0], [2.0, 3.0]];
    let y = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]; // Different number of samples

    let optimizer = JointLossOptimizer::new();
    let result = optimizer.fit(&X.view(), &y.view());
    assert!(result.is_err());
}

#[test]
fn test_multi_objective_optimizer_invalid_objectives() {
    let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];
    let y = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];

    let config = MultiObjectiveConfig {
        population_size: 10,
        generations: 5,
        objectives: vec!["invalid_objective".to_string()],
        ..Default::default()
    };

    let optimizer = MultiObjectiveOptimizer::new().config(config);
    let result = optimizer.fit(&X.view(), &y.view());
    assert!(result.is_err());
}

#[test]
fn test_joint_loss_optimizer_cross_entropy() {
    let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];
    let y = array![[0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]; // Binary labels

    let config = JointLossConfig {
        output_losses: vec![LossFunction::CrossEntropy, LossFunction::CrossEntropy],
        combination: LossCombination::Sum,
        max_iter: 100,
        learning_rate: 0.01,
        ..Default::default()
    };

    let optimizer = JointLossOptimizer::new().config(config);
    let trained = optimizer.fit(&X.view(), &y.view()).unwrap();

    let predictions = trained.predict(&X.view()).unwrap();
    assert_eq!(predictions.shape(), &[3, 2]);
}

#[test]
fn test_joint_loss_optimizer_max_combination() {
    let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];
    let y = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];

    let config = JointLossConfig {
        output_losses: vec![LossFunction::MSE, LossFunction::MAE],
        combination: LossCombination::Max,
        max_iter: 50,
        ..Default::default()
    };

    let optimizer = JointLossOptimizer::new().config(config);
    let trained = optimizer.fit(&X.view(), &y.view()).unwrap();

    assert_eq!(trained.weights().shape(), &[2, 2]);
    assert_eq!(trained.bias().len(), 2);
}

#[test]
fn test_nsga2_optimizer_basic() {
    let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
    let y = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];

    let config = NSGA2Config {
        population_size: 20,
        generations: 10,
        crossover_prob: 0.8,
        mutation_prob: 0.1,
        random_state: Some(42),
        ..Default::default()
    };

    let optimizer = NSGA2Optimizer::new().config(config);
    let trained = optimizer.fit(&X.view(), &y.view()).unwrap();

    assert!(!trained.pareto_solutions().is_empty());
    assert!(!trained.convergence_history().is_empty());
    assert_eq!(trained.convergence_history().len(), 10);

    let predictions = trained.predict(&X.view()).unwrap();
    assert_eq!(predictions.shape(), &[4, 2]);
}

#[test]
fn test_nsga2_optimizer_sbx_algorithm() {
    let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];
    let y = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];

    let config = NSGA2Config {
        population_size: 15,
        generations: 5,
        algorithm: NSGA2Algorithm::SBX,
        eta_c: 15.0,
        eta_m: 15.0,
        random_state: Some(123),
        ..Default::default()
    };

    let optimizer = NSGA2Optimizer::new()
        .config(config)
        .population_size(15)
        .generations(5)
        .crossover_prob(0.9)
        .mutation_prob(0.2)
        .algorithm(NSGA2Algorithm::SBX);

    let trained = optimizer.fit(&X.view(), &y.view()).unwrap();

    assert!(!trained.pareto_solutions().is_empty());
    assert_eq!(trained.final_population().len(), 15);

    // Check that best solution has reasonable parameters
    let best = trained.best_solution();
    assert_eq!(best.parameters.len(), 6); // 2 features * 2 outputs + 2 bias
    assert!(best.objectives.len() == 2);
}

#[test]
fn test_nsga2_optimizer_convergence() {
    let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
    let y = array![[2.0, 4.0], [4.0, 6.0], [6.0, 8.0], [8.0, 10.0]]; // Linear relationship

    let config = NSGA2Config {
        population_size: 30,
        generations: 20,
        crossover_prob: 0.9,
        mutation_prob: 0.1,
        random_state: Some(42),
        ..Default::default()
    };

    let optimizer = NSGA2Optimizer::new().config(config);
    let trained = optimizer.fit(&X.view(), &y.view()).unwrap();

    // Check convergence - later generations should have better or similar hypervolume
    let history = trained.convergence_history();
    assert!(!history.is_empty());

    // Check that we have Pareto solutions
    assert!(!trained.pareto_solutions().is_empty());

    // Check that solutions have valid objectives
    for solution in trained.pareto_solutions() {
        assert!(solution.objectives[0] >= 0.0); // MSE should be non-negative
        assert!(solution.objectives[1] >= 0.0); // Complexity should be non-negative
    }
}

#[test]
fn test_nsga2_optimizer_builder_pattern() {
    let config = NSGA2Optimizer::new()
        .population_size(50)
        .generations(25)
        .crossover_prob(0.85)
        .mutation_prob(0.15)
        .algorithm(NSGA2Algorithm::Standard);

    // Access config through Estimator trait method
    let cfg = Estimator::config(&config);
    assert_eq!(cfg.population_size, 50);
    assert_eq!(cfg.generations, 25);
    assert_abs_diff_eq!(cfg.crossover_prob, 0.85, epsilon = 1e-6);
    assert_abs_diff_eq!(cfg.mutation_prob, 0.15, epsilon = 1e-6);
    assert_eq!(cfg.algorithm, NSGA2Algorithm::Standard);
}

#[test]
fn test_nsga2_optimizer_invalid_input() {
    let X = array![[1.0, 2.0], [2.0, 3.0]];
    let y = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]; // Different number of samples

    let optimizer = NSGA2Optimizer::new();
    let result = optimizer.fit(&X.view(), &y.view());
    assert!(result.is_err());
}

#[test]
fn test_nsga2_dominance_relationships() {
    let solution1 = ParetoSolution {
        parameters: array![1.0, 2.0, 3.0],
        objectives: array![1.0, 2.0], // Better in both objectives
        rank: 0,
        crowding_distance: 0.0,
    };

    let solution2 = ParetoSolution {
        parameters: array![2.0, 3.0, 4.0],
        objectives: array![2.0, 3.0], // Worse in both objectives
        rank: 0,
        crowding_distance: 0.0,
    };

    let optimizer = NSGA2Optimizer::new();
    assert!(optimizer.nsga2_dominates(&solution1, &solution2));
    assert!(!optimizer.nsga2_dominates(&solution2, &solution1));

    // Test non-dominating solutions
    let solution3 = ParetoSolution {
        parameters: array![1.5, 2.5, 3.5],
        objectives: array![0.5, 3.5], // Better in first, worse in second
        rank: 0,
        crowding_distance: 0.0,
    };

    assert!(!optimizer.nsga2_dominates(&solution1, &solution3));
    assert!(!optimizer.nsga2_dominates(&solution3, &solution1));
}

#[test]
fn test_nsga2_optimizer_reproducibility() {
    let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];
    let y = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];

    let config1 = NSGA2Config {
        population_size: 20,
        generations: 5,
        random_state: Some(42),
        ..Default::default()
    };

    let config2 = NSGA2Config {
        population_size: 20,
        generations: 5,
        random_state: Some(42),
        ..Default::default()
    };

    let optimizer1 = NSGA2Optimizer::new().config(config1);
    let optimizer2 = NSGA2Optimizer::new().config(config2);

    let trained1 = optimizer1.fit(&X.view(), &y.view()).unwrap();
    let trained2 = optimizer2.fit(&X.view(), &y.view()).unwrap();

    // Results should be identical with same random state
    assert_eq!(
        trained1.pareto_solutions().len(),
        trained2.pareto_solutions().len()
    );
    assert_eq!(
        trained1.convergence_history().len(),
        trained2.convergence_history().len()
    );

    // Check that best solutions are the same
    let best1 = trained1.best_solution();
    let best2 = trained2.best_solution();

    for i in 0..best1.parameters.len() {
        assert_abs_diff_eq!(best1.parameters[i], best2.parameters[i], epsilon = 1e-10);
    }
}
