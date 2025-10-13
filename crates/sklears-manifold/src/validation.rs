//! Validation framework for manifold learning algorithms
//! This module provides comprehensive validation and hyperparameter tuning
//! capabilities for manifold learning algorithms, including cross-validation,
//! grid search, random search, and advanced hyperparameter optimization.

use crate::benchmark_datasets::PerformanceEvaluator;
use scirs2_core::essentials::Normal;
use scirs2_core::ndarray::{Array1, Array2, Axis};
use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::thread_rng;
use scirs2_core::random::Rng;
use scirs2_core::random::SeedableRng;
use scirs2_core::Distribution;
use scirs2_core::SliceRandomExt;
use std::fmt;

use std::collections::HashMap;
#[derive(Debug, Clone)]
pub enum CrossValidationStrategy {
    /// K-fold cross-validation
    KFold {
        k: usize,
        shuffle: bool,
        random_state: Option<u64>,
    },
    /// Stratified K-fold (for labeled data)
    StratifiedKFold {
        k: usize,
        shuffle: bool,
        random_state: Option<u64>,
    },
    /// Leave-one-out cross-validation
    LeaveOneOut,
    /// Time series split (for temporal data)
    TimeSeriesSplit {
        n_splits: usize,
        test_size: Option<usize>,
    },
    /// Custom split using provided indices
    CustomSplit {
        train_indices: Vec<Vec<usize>>,
        test_indices: Vec<Vec<usize>>,
    },
}

/// Hyperparameter space definition
#[derive(Debug, Clone)]
pub enum ParameterSpace {
    /// Discrete choice from a list of values
    Choice(Vec<ParameterValue>),
    /// Uniform distribution over a range
    Uniform { low: f64, high: f64 },
    /// Log-uniform distribution
    LogUniform { low: f64, high: f64 },
    /// Normal distribution
    Normal { mean: f64, std: f64 },
    /// Integer range
    IntRange { low: i32, high: i32 },
}

/// Parameter value types
#[derive(Debug, Clone, PartialEq)]
pub enum ParameterValue {
    /// Float
    Float(f64),
    /// Int
    Int(i32),
    /// String
    String(String),
    /// Bool
    Bool(bool),
}

impl fmt::Display for ParameterValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParameterValue::Float(v) => write!(f, "{}", v),
            ParameterValue::Int(v) => write!(f, "{}", v),
            ParameterValue::String(v) => write!(f, "{}", v),
            ParameterValue::Bool(v) => write!(f, "{}", v),
        }
    }
}

/// Hyperparameter configuration
pub type ParameterGrid = HashMap<String, ParameterSpace>;
pub type ParameterSet = HashMap<String, ParameterValue>;

/// Cross-validation result for a single fold
#[derive(Debug, Clone)]
pub struct FoldResult {
    /// fold_index
    pub fold_index: usize,
    /// train_indices
    pub train_indices: Vec<usize>,
    /// test_indices
    pub test_indices: Vec<usize>,
    /// parameters
    pub parameters: ParameterSet,
    /// trustworthiness
    pub trustworthiness: f64,
    /// continuity
    pub continuity: f64,
    /// normalized_stress
    pub normalized_stress: f64,
    /// neighborhood_preservation
    pub neighborhood_preservation: f64,
    /// execution_time
    pub execution_time: std::time::Duration,
    /// memory_usage
    pub memory_usage: Option<usize>,
}

/// Complete cross-validation results
#[derive(Debug, Clone)]
pub struct CrossValidationResult {
    /// strategy
    pub strategy: CrossValidationStrategy,
    /// parameters
    pub parameters: ParameterSet,
    /// fold_results
    pub fold_results: Vec<FoldResult>,
    /// mean_scores
    pub mean_scores: ValidationScores,
    /// std_scores
    pub std_scores: ValidationScores,
    /// best_fold_index
    pub best_fold_index: usize,
    /// total_time
    pub total_time: std::time::Duration,
}

/// Validation scores summary
#[derive(Debug, Clone)]
pub struct ValidationScores {
    /// trustworthiness
    pub trustworthiness: f64,
    /// continuity
    pub continuity: f64,
    /// normalized_stress
    pub normalized_stress: f64,
    /// neighborhood_preservation
    pub neighborhood_preservation: f64,
}

impl ValidationScores {
    /// Create new validation scores
    pub fn new(trust: f64, cont: f64, stress: f64, neigh: f64) -> Self {
        Self {
            trustworthiness: trust,
            continuity: cont,
            normalized_stress: stress,
            neighborhood_preservation: neigh,
        }
    }

    /// Compute composite score (higher is better)
    pub fn composite_score(&self) -> f64 {
        // Weighted combination of metrics (stress is inverted since lower is better)
        0.3 * self.trustworthiness
            + 0.3 * self.continuity
            + 0.2 * (1.0 - self.normalized_stress.min(1.0))
            + 0.2 * self.neighborhood_preservation
    }
}

/// Cross-validation splitter
pub struct CrossValidationSplitter {
    strategy: CrossValidationStrategy,
}

impl CrossValidationSplitter {
    /// Create a new cross-validation splitter
    pub fn new(strategy: CrossValidationStrategy) -> Self {
        Self { strategy }
    }

    /// Generate train/test splits for the given data
    pub fn split(
        &self,
        n_samples: usize,
        labels: Option<&Array1<usize>>,
    ) -> Vec<(Vec<usize>, Vec<usize>)> {
        match &self.strategy {
            CrossValidationStrategy::KFold {
                k,
                shuffle,
                random_state,
            } => self.k_fold_split(n_samples, *k, *shuffle, *random_state),
            CrossValidationStrategy::StratifiedKFold {
                k,
                shuffle,
                random_state,
            } => {
                if let Some(labels) = labels {
                    self.stratified_k_fold_split(n_samples, labels, *k, *shuffle, *random_state)
                } else {
                    // Fallback to regular k-fold if no labels provided
                    self.k_fold_split(n_samples, *k, *shuffle, *random_state)
                }
            }
            CrossValidationStrategy::LeaveOneOut => self.leave_one_out_split(n_samples),
            CrossValidationStrategy::TimeSeriesSplit {
                n_splits,
                test_size,
            } => self.time_series_split(n_samples, *n_splits, *test_size),
            CrossValidationStrategy::CustomSplit {
                train_indices,
                test_indices,
            } => train_indices
                .iter()
                .zip(test_indices.iter())
                .map(|(train, test)| (train.clone(), test.clone()))
                .collect(),
        }
    }

    fn k_fold_split(
        &self,
        n_samples: usize,
        k: usize,
        shuffle: bool,
        random_state: Option<u64>,
    ) -> Vec<(Vec<usize>, Vec<usize>)> {
        let mut indices: Vec<usize> = (0..n_samples).collect();

        if shuffle {
            if let Some(seed) = random_state {
                let mut rng = StdRng::seed_from_u64(seed);
                indices.shuffle(&mut rng);
            } else {
                let mut rng = thread_rng();
                indices.shuffle(&mut rng);
            }
        }

        let mut splits = Vec::new();
        let fold_size = n_samples / k;

        for i in 0..k {
            let start = i * fold_size;
            let end = if i == k - 1 {
                n_samples
            } else {
                (i + 1) * fold_size
            };

            let test_indices = indices[start..end].to_vec();
            let train_indices = indices[0..start]
                .iter()
                .chain(indices[end..].iter())
                .copied()
                .collect();

            splits.push((train_indices, test_indices));
        }

        splits
    }

    fn stratified_k_fold_split(
        &self,
        n_samples: usize,
        labels: &Array1<usize>,
        k: usize,
        shuffle: bool,
        random_state: Option<u64>,
    ) -> Vec<(Vec<usize>, Vec<usize>)> {
        // Group indices by label
        let mut label_groups: HashMap<usize, Vec<usize>> = HashMap::new();
        for (idx, &label) in labels.iter().enumerate() {
            label_groups.entry(label).or_default().push(idx);
        }

        // Shuffle within each group if requested
        if shuffle {
            let mut rng = if let Some(seed) = random_state {
                StdRng::seed_from_u64(seed)
            } else {
                StdRng::seed_from_u64(thread_rng().gen())
            };

            for group in label_groups.values_mut() {
                group.shuffle(&mut rng);
            }
        }

        let mut splits = vec![(Vec::new(), Vec::new()); k];

        // Distribute samples from each group across folds
        for (_label, indices) in label_groups {
            let group_size = indices.len();
            let base_fold_size = group_size / k;
            let remainder = group_size % k;

            let mut start = 0;
            for fold in 0..k {
                let fold_size = base_fold_size + if fold < remainder { 1 } else { 0 };
                let end = start + fold_size;

                for &idx in &indices[start..end] {
                    splits[fold].1.push(idx); // test set
                }

                // Add to train sets of other folds
                for other_fold in 0..k {
                    if other_fold != fold {
                        for &idx in &indices[start..end] {
                            splits[other_fold].0.push(idx);
                        }
                    }
                }

                start = end;
            }
        }

        splits
    }

    fn leave_one_out_split(&self, n_samples: usize) -> Vec<(Vec<usize>, Vec<usize>)> {
        let mut splits = Vec::new();

        for i in 0..n_samples {
            let test_indices = vec![i];
            let train_indices = (0..n_samples).filter(|&x| x != i).collect();
            splits.push((train_indices, test_indices));
        }

        splits
    }

    fn time_series_split(
        &self,
        n_samples: usize,
        n_splits: usize,
        test_size: Option<usize>,
    ) -> Vec<(Vec<usize>, Vec<usize>)> {
        let test_size = test_size.unwrap_or(n_samples / (n_splits + 1));
        let mut splits = Vec::new();

        for i in 0..n_splits {
            let test_start = n_samples - test_size * (n_splits - i);
            let test_end = test_start + test_size;

            if test_start < n_samples {
                let train_indices = (0..test_start).collect();
                let test_indices = (test_start..test_end.min(n_samples)).collect();
                splits.push((train_indices, test_indices));
            }
        }

        splits
    }
}

/// Hyperparameter optimization strategies
#[derive(Debug, Clone)]
pub enum OptimizationStrategy {
    /// Exhaustive grid search
    GridSearch,
    /// Random search with specified number of iterations
    RandomSearch {
        n_iter: usize,
        random_state: Option<u64>,
    },
    /// Bayesian optimization (simplified implementation)
    BayesianOptimization {
        n_iter: usize,
        exploration_weight: f64,
    },
    /// Genetic algorithm
    GeneticAlgorithm {
        population_size: usize,
        n_generations: usize,
        mutation_rate: f64,
    },
}

/// Hyperparameter optimizer
pub struct HyperparameterOptimizer {
    parameter_grid: ParameterGrid,
    strategy: OptimizationStrategy,
    cv_strategy: CrossValidationStrategy,
    scoring_metric: ScoringMetric,
}

/// Scoring metrics for hyperparameter optimization
#[derive(Debug, Clone)]
pub enum ScoringMetric {
    /// Trustworthiness
    Trustworthiness,
    /// Continuity
    Continuity,
    /// NormalizedStress
    NormalizedStress, // Lower is better
    /// NeighborhoodPreservation
    NeighborhoodPreservation,
    /// CompositeScore
    CompositeScore,
}

impl HyperparameterOptimizer {
    /// Create a new hyperparameter optimizer
    pub fn new(
        parameter_grid: ParameterGrid,
        strategy: OptimizationStrategy,
        cv_strategy: CrossValidationStrategy,
        scoring_metric: ScoringMetric,
    ) -> Self {
        Self {
            parameter_grid,
            strategy,
            cv_strategy,
            scoring_metric,
        }
    }

    /// Optimize hyperparameters for a given dataset
    pub fn optimize<F>(
        &self,
        data: &Array2<f64>,
        labels: Option<&Array1<usize>>,
        fit_transform_fn: F,
    ) -> OptimizationResult
    where
        F: Fn(&Array2<f64>, &ParameterSet) -> Result<Array2<f64>, String> + Send + Sync,
    {
        let start_time = std::time::Instant::now();

        let parameter_combinations = match &self.strategy {
            OptimizationStrategy::GridSearch => self.generate_grid_combinations(),
            OptimizationStrategy::RandomSearch {
                n_iter,
                random_state,
            } => self.generate_random_combinations(*n_iter, *random_state),
            OptimizationStrategy::BayesianOptimization {
                n_iter,
                exploration_weight,
            } => self.generate_bayesian_combinations(*n_iter, *exploration_weight),
            OptimizationStrategy::GeneticAlgorithm {
                population_size,
                n_generations,
                mutation_rate,
            } => {
                self.generate_genetic_combinations(*population_size, *n_generations, *mutation_rate)
            }
        };

        let mut cv_results = Vec::new();
        let splitter = CrossValidationSplitter::new(self.cv_strategy.clone());

        for (combo_idx, parameters) in parameter_combinations.iter().enumerate() {
            let cv_result =
                self.cross_validate(data, labels, parameters, &splitter, &fit_transform_fn);
            cv_results.push(cv_result);

            // Progress reporting could be added here
            if combo_idx % 10 == 0 && combo_idx > 0 {
                println!(
                    "Completed {}/{} parameter combinations",
                    combo_idx,
                    parameter_combinations.len()
                );
            }
        }

        // Find best parameters based on scoring metric
        let best_idx = self.find_best_parameters(&cv_results);
        let best_result = cv_results[best_idx].clone();

        /// OptimizationResult
        OptimizationResult {
            best_parameters: best_result.parameters.clone(),
            best_score: self.extract_score(&best_result.mean_scores),
            best_cv_result: best_result,
            all_results: cv_results,
            optimization_time: start_time.elapsed(),
            n_parameter_combinations: parameter_combinations.len(),
        }
    }

    fn cross_validate<F>(
        &self,
        data: &Array2<f64>,
        labels: Option<&Array1<usize>>,
        parameters: &ParameterSet,
        splitter: &CrossValidationSplitter,
        fit_transform_fn: &F,
    ) -> CrossValidationResult
    where
        F: Fn(&Array2<f64>, &ParameterSet) -> Result<Array2<f64>, String>,
    {
        let splits = splitter.split(data.nrows(), labels);
        let mut fold_results = Vec::new();
        let cv_start = std::time::Instant::now();

        for (fold_idx, (train_indices, test_indices)) in splits.iter().enumerate() {
            let fold_start = std::time::Instant::now();

            // Create train/test data
            let train_data = data.select(Axis(0), train_indices);

            // Fit and transform
            match fit_transform_fn(&train_data, parameters) {
                Ok(embedded_data) => {
                    // Evaluate embedding quality
                    let k_neighbors = 10.min(train_data.nrows() - 1);

                    let trustworthiness = PerformanceEvaluator::trustworthiness(
                        &train_data,
                        &embedded_data,
                        k_neighbors,
                    );
                    let continuity =
                        PerformanceEvaluator::continuity(&train_data, &embedded_data, k_neighbors);
                    let stress =
                        PerformanceEvaluator::normalized_stress(&train_data, &embedded_data);
                    let neighborhood_preservation = PerformanceEvaluator::neighborhood_hit_rate(
                        &train_data,
                        &embedded_data,
                        k_neighbors,
                    );

                    fold_results.push(FoldResult {
                        fold_index: fold_idx,
                        train_indices: train_indices.clone(),
                        test_indices: test_indices.clone(),
                        parameters: parameters.clone(),
                        trustworthiness,
                        continuity,
                        normalized_stress: stress,
                        neighborhood_preservation,
                        execution_time: fold_start.elapsed(),
                        memory_usage: None, // Could be implemented with memory profiling
                    });
                }
                Err(_) => {
                    // Handle failed fits with poor scores
                    fold_results.push(FoldResult {
                        fold_index: fold_idx,
                        train_indices: train_indices.clone(),
                        test_indices: test_indices.clone(),
                        parameters: parameters.clone(),
                        trustworthiness: 0.0,
                        continuity: 0.0,
                        normalized_stress: f64::INFINITY,
                        neighborhood_preservation: 0.0,
                        execution_time: fold_start.elapsed(),
                        memory_usage: None,
                    });
                }
            }
        }

        // Compute mean and std scores
        let n_folds = fold_results.len() as f64;
        let mean_trust = fold_results.iter().map(|r| r.trustworthiness).sum::<f64>() / n_folds;
        let mean_cont = fold_results.iter().map(|r| r.continuity).sum::<f64>() / n_folds;
        let mean_stress = fold_results
            .iter()
            .map(|r| r.normalized_stress)
            .sum::<f64>()
            / n_folds;
        let mean_neigh = fold_results
            .iter()
            .map(|r| r.neighborhood_preservation)
            .sum::<f64>()
            / n_folds;

        let std_trust = (fold_results
            .iter()
            .map(|r| (r.trustworthiness - mean_trust).powi(2))
            .sum::<f64>()
            / n_folds)
            .sqrt();
        let std_cont = (fold_results
            .iter()
            .map(|r| (r.continuity - mean_cont).powi(2))
            .sum::<f64>()
            / n_folds)
            .sqrt();
        let std_stress = (fold_results
            .iter()
            .map(|r| (r.normalized_stress - mean_stress).powi(2))
            .sum::<f64>()
            / n_folds)
            .sqrt();
        let std_neigh = (fold_results
            .iter()
            .map(|r| (r.neighborhood_preservation - mean_neigh).powi(2))
            .sum::<f64>()
            / n_folds)
            .sqrt();

        let best_fold_index = fold_results
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                let score_a = ValidationScores::new(
                    a.trustworthiness,
                    a.continuity,
                    a.normalized_stress,
                    a.neighborhood_preservation,
                )
                .composite_score();
                let score_b = ValidationScores::new(
                    b.trustworthiness,
                    b.continuity,
                    b.normalized_stress,
                    b.neighborhood_preservation,
                )
                .composite_score();
                score_a.partial_cmp(&score_b).unwrap()
            })
            .map(|(idx, _)| idx)
            .unwrap_or(0);

        /// CrossValidationResult
        CrossValidationResult {
            strategy: self.cv_strategy.clone(),
            parameters: parameters.clone(),
            fold_results,
            mean_scores: ValidationScores::new(mean_trust, mean_cont, mean_stress, mean_neigh),
            std_scores: ValidationScores::new(std_trust, std_cont, std_stress, std_neigh),
            best_fold_index,
            total_time: cv_start.elapsed(),
        }
    }

    fn generate_grid_combinations(&self) -> Vec<ParameterSet> {
        let mut combinations = vec![HashMap::new()];

        for (param_name, param_space) in &self.parameter_grid {
            let mut new_combinations = Vec::new();

            let values = match param_space {
                ParameterSpace::Choice(values) => values.clone(),
                ParameterSpace::Uniform { low, high } => {
                    // Generate 10 evenly spaced values
                    (0..10)
                        .map(|i| {
                            let val = low + (high - low) * i as f64 / 9.0;
                            ParameterValue::Float(val)
                        })
                        .collect()
                }
                ParameterSpace::IntRange { low, high } => {
                    (*low..=*high).map(ParameterValue::Int).collect()
                }
                _ => {
                    // For other spaces, use a few sample values
                    vec![
                        ParameterValue::Float(0.1),
                        ParameterValue::Float(0.5),
                        ParameterValue::Float(1.0),
                    ]
                }
            };

            for combo in &combinations {
                for value in &values {
                    let mut new_combo = combo.clone();
                    new_combo.insert(param_name.clone(), value.clone());
                    new_combinations.push(new_combo);
                }
            }

            combinations = new_combinations;
        }

        combinations
    }

    fn generate_random_combinations(
        &self,
        n_iter: usize,
        random_state: Option<u64>,
    ) -> Vec<ParameterSet> {
        let mut combinations = Vec::new();
        let mut rng = if let Some(seed) = random_state {
            StdRng::seed_from_u64(seed)
        } else {
            StdRng::seed_from_u64(thread_rng().gen())
        };

        for _ in 0..n_iter {
            let mut combo = HashMap::new();

            for (param_name, param_space) in &self.parameter_grid {
                let value = match param_space {
                    ParameterSpace::Choice(values) => {
                        values[rng.gen_range(0..values.len())].clone()
                    }
                    ParameterSpace::Uniform { low, high } => {
                        ParameterValue::Float(rng.gen_range(*low..*high))
                    }
                    ParameterSpace::LogUniform { low, high } => {
                        let log_val = rng.gen_range(low.ln()..high.ln());
                        ParameterValue::Float(log_val.exp())
                    }
                    ParameterSpace::Normal { mean, std } => {
                        let normal = Normal::new(*mean, *std).unwrap();
                        let val = normal.sample(&mut rng);
                        ParameterValue::Float(val)
                    }
                    ParameterSpace::IntRange { low, high } => {
                        ParameterValue::Int(rng.gen_range(*low..=*high))
                    }
                };

                combo.insert(param_name.clone(), value);
            }

            combinations.push(combo);
        }

        combinations
    }

    fn generate_bayesian_combinations(
        &self,
        n_iter: usize,
        _exploration_weight: f64,
    ) -> Vec<ParameterSet> {
        // Simplified Bayesian optimization - in practice would use Gaussian processes
        // For now, use random search with some guided exploration
        self.generate_random_combinations(n_iter, Some(42))
    }

    fn generate_genetic_combinations(
        &self,
        population_size: usize,
        n_generations: usize,
        _mutation_rate: f64,
    ) -> Vec<ParameterSet> {
        // Simplified genetic algorithm - start with random population
        // In practice would implement selection, crossover, and mutation
        self.generate_random_combinations(population_size * n_generations / 10, Some(42))
    }

    fn find_best_parameters(&self, cv_results: &[CrossValidationResult]) -> usize {
        cv_results
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                let score_a = self.extract_score(&a.mean_scores);
                let score_b = self.extract_score(&b.mean_scores);
                score_a.partial_cmp(&score_b).unwrap()
            })
            .map(|(idx, _)| idx)
            .unwrap_or(0)
    }

    fn extract_score(&self, scores: &ValidationScores) -> f64 {
        match self.scoring_metric {
            ScoringMetric::Trustworthiness => scores.trustworthiness,
            ScoringMetric::Continuity => scores.continuity,
            ScoringMetric::NormalizedStress => 1.0 - scores.normalized_stress.min(1.0), // Invert since lower is better
            ScoringMetric::NeighborhoodPreservation => scores.neighborhood_preservation,
            ScoringMetric::CompositeScore => scores.composite_score(),
        }
    }
}

/// Hyperparameter optimization result
#[derive(Debug, Clone)]
pub struct OptimizationResult {
    /// best_parameters
    pub best_parameters: ParameterSet,
    /// best_score
    pub best_score: f64,
    /// best_cv_result
    pub best_cv_result: CrossValidationResult,
    /// all_results
    pub all_results: Vec<CrossValidationResult>,
    /// optimization_time
    pub optimization_time: std::time::Duration,
    /// n_parameter_combinations
    pub n_parameter_combinations: usize,
}

impl OptimizationResult {
    /// Generate a summary report of the optimization
    pub fn summary(&self) -> String {
        format!(
            "Hyperparameter Optimization Results\n\
             ====================================\n\
             Best Score: {:.4}\n\
             Best Parameters:\n{}\n\
             Cross-validation Results:\n\
             - Mean Trustworthiness: {:.4} ± {:.4}\n\
             - Mean Continuity: {:.4} ± {:.4}\n\
             - Mean Normalized Stress: {:.4} ± {:.4}\n\
             - Mean Neighborhood Preservation: {:.4} ± {:.4}\n\
             \n\
             Optimization Details:\n\
             - Total Time: {:.2}s\n\
             - Parameter Combinations Tested: {}\n\
             - Best Fold Index: {}",
            self.best_score,
            self.format_parameters(&self.best_parameters),
            self.best_cv_result.mean_scores.trustworthiness,
            self.best_cv_result.std_scores.trustworthiness,
            self.best_cv_result.mean_scores.continuity,
            self.best_cv_result.std_scores.continuity,
            self.best_cv_result.mean_scores.normalized_stress,
            self.best_cv_result.std_scores.normalized_stress,
            self.best_cv_result.mean_scores.neighborhood_preservation,
            self.best_cv_result.std_scores.neighborhood_preservation,
            self.optimization_time.as_secs_f64(),
            self.n_parameter_combinations,
            self.best_cv_result.best_fold_index
        )
    }

    fn format_parameters(&self, params: &ParameterSet) -> String {
        params
            .iter()
            .map(|(key, value)| format!("  {}: {}", key, value))
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Get parameter importance analysis
    pub fn parameter_importance(&self) -> HashMap<String, f64> {
        let mut importance = HashMap::new();

        // Simplified parameter importance based on score variance
        for param_name in self.best_parameters.keys() {
            let mut param_scores = Vec::new();

            for result in &self.all_results {
                if let Some(_param_value) = result.parameters.get(param_name) {
                    param_scores.push(result.mean_scores.composite_score());
                }
            }

            if param_scores.len() > 1 {
                let mean_score = param_scores.iter().sum::<f64>() / param_scores.len() as f64;
                let variance = param_scores
                    .iter()
                    .map(|score| (score - mean_score).powi(2))
                    .sum::<f64>()
                    / param_scores.len() as f64;

                importance.insert(param_name.clone(), variance.sqrt());
            }
        }

        importance
    }
}

/// Validation utilities
pub mod utils {
    use super::*;

    /// Create a standard parameter grid for t-SNE
    pub fn create_tsne_parameter_grid() -> ParameterGrid {
        let mut grid = HashMap::new();

        grid.insert(
            "perplexity".to_string(),
            ParameterSpace::Choice(vec![
                ParameterValue::Float(5.0),
                ParameterValue::Float(10.0),
                ParameterValue::Float(30.0),
                ParameterValue::Float(50.0),
                ParameterValue::Float(100.0),
            ]),
        );

        grid.insert(
            "learning_rate".to_string(),
            ParameterSpace::LogUniform {
                low: 10.0,
                high: 1000.0,
            },
        );

        grid.insert(
            "n_iter".to_string(),
            ParameterSpace::Choice(vec![
                ParameterValue::Int(250),
                ParameterValue::Int(500),
                ParameterValue::Int(1000),
            ]),
        );

        grid
    }

    /// Create a standard parameter grid for UMAP
    pub fn create_umap_parameter_grid() -> ParameterGrid {
        let mut grid = HashMap::new();

        grid.insert(
            "n_neighbors".to_string(),
            ParameterSpace::Choice(vec![
                ParameterValue::Int(5),
                ParameterValue::Int(15),
                ParameterValue::Int(50),
                ParameterValue::Int(100),
            ]),
        );

        grid.insert(
            "min_dist".to_string(),
            ParameterSpace::LogUniform {
                low: 0.001,
                high: 0.5,
            },
        );

        grid.insert(
            "spread".to_string(),
            ParameterSpace::Uniform {
                low: 0.5,
                high: 2.0,
            },
        );

        grid
    }

    /// Create a standard parameter grid for Isomap
    pub fn create_isomap_parameter_grid() -> ParameterGrid {
        let mut grid = HashMap::new();

        grid.insert(
            "n_neighbors".to_string(),
            ParameterSpace::Choice(vec![
                ParameterValue::Int(5),
                ParameterValue::Int(10),
                ParameterValue::Int(20),
                ParameterValue::Int(30),
            ]),
        );

        grid.insert(
            "eigen_solver".to_string(),
            ParameterSpace::Choice(vec![
                ParameterValue::String("auto".to_string()),
                ParameterValue::String("dense".to_string()),
            ]),
        );

        grid
    }

    /// Create a quick validation setup for testing
    pub fn create_quick_validation() -> (CrossValidationStrategy, OptimizationStrategy) {
        let cv_strategy = CrossValidationStrategy::KFold {
            k: 3,
            shuffle: true,
            random_state: Some(42),
        };

        let opt_strategy = OptimizationStrategy::RandomSearch {
            n_iter: 10,
            random_state: Some(42),
        };

        (cv_strategy, opt_strategy)
    }

    /// Create a comprehensive validation setup
    pub fn create_comprehensive_validation() -> (CrossValidationStrategy, OptimizationStrategy) {
        let cv_strategy = CrossValidationStrategy::KFold {
            k: 5,
            shuffle: true,
            random_state: Some(42),
        };

        let opt_strategy = OptimizationStrategy::GridSearch;

        (cv_strategy, opt_strategy)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::benchmark_datasets::BenchmarkDatasets;
    use scirs2_core::s;

    #[test]
    fn test_k_fold_splitter() {
        let strategy = CrossValidationStrategy::KFold {
            k: 3,
            shuffle: false,
            random_state: None,
        };
        let splitter = CrossValidationSplitter::new(strategy);
        let splits = splitter.split(9, None);

        assert_eq!(splits.len(), 3);

        // Check that all indices are covered
        let mut all_indices: Vec<usize> = Vec::new();
        for (train, test) in &splits {
            all_indices.extend(train);
            all_indices.extend(test);
        }
        all_indices.sort();
        all_indices.dedup();
        assert_eq!(all_indices, (0..9).collect::<Vec<_>>());
    }

    #[test]
    fn test_parameter_space_generation() {
        let mut grid = HashMap::new();
        grid.insert(
            "test_param".to_string(),
            ParameterSpace::Choice(vec![ParameterValue::Float(1.0), ParameterValue::Float(2.0)]),
        );

        let optimizer = HyperparameterOptimizer::new(
            grid,
            OptimizationStrategy::GridSearch,
            CrossValidationStrategy::KFold {
                k: 2,
                shuffle: false,
                random_state: None,
            },
            ScoringMetric::CompositeScore,
        );

        let combinations = optimizer.generate_grid_combinations();
        assert_eq!(combinations.len(), 2);

        assert!(combinations[0].contains_key("test_param"));
        assert!(combinations[1].contains_key("test_param"));
    }

    #[test]
    fn test_validation_scores() {
        let scores = ValidationScores::new(0.8, 0.7, 0.2, 0.9);
        let composite = scores.composite_score();

        // Should be a weighted combination
        assert!(composite > 0.0 && composite <= 1.0);
    }

    #[test]
    fn test_cross_validation_with_mock_function() {
        let (data, _) = BenchmarkDatasets::swiss_roll(50, 0.1, 42);

        // Mock fit_transform function that just returns first 2 dimensions
        let fit_transform_fn =
            |data: &Array2<f64>, _params: &ParameterSet| -> Result<Array2<f64>, String> {
                Ok(data.slice(s![.., 0..2]).to_owned())
            };

        let strategy = CrossValidationStrategy::KFold {
            k: 3,
            shuffle: true,
            random_state: Some(42),
        };
        let splitter = CrossValidationSplitter::new(strategy.clone());

        let mut parameters = HashMap::new();
        parameters.insert("test_param".to_string(), ParameterValue::Float(1.0));

        let optimizer = HyperparameterOptimizer::new(
            HashMap::new(),
            OptimizationStrategy::GridSearch,
            strategy,
            ScoringMetric::CompositeScore,
        );

        let result =
            optimizer.cross_validate(&data, None, &parameters, &splitter, &fit_transform_fn);

        assert_eq!(result.fold_results.len(), 3);
        assert!(result.mean_scores.trustworthiness >= 0.0);
        assert!(result.mean_scores.continuity >= 0.0);
    }

    #[test]
    fn test_standard_parameter_grids() {
        let tsne_grid = utils::create_tsne_parameter_grid();
        assert!(tsne_grid.contains_key("perplexity"));
        assert!(tsne_grid.contains_key("learning_rate"));

        let umap_grid = utils::create_umap_parameter_grid();
        assert!(umap_grid.contains_key("n_neighbors"));
        assert!(umap_grid.contains_key("min_dist"));

        let isomap_grid = utils::create_isomap_parameter_grid();
        assert!(isomap_grid.contains_key("n_neighbors"));
    }
}
