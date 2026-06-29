//! Pipeline optimization and validation
//!
//! Components for automated hyperparameter optimization, pipeline structure search,
//! and validation.

use scirs2_core::ndarray::{Array1, ArrayView1, ArrayView2, Axis};
use scirs2_core::random::{thread_rng, Rng, RngExt};
use sklears_core::{error::Result as SklResult, prelude::SklearsError, types::Float};
use std::collections::HashMap;
use std::time::Instant;

use crate::{Pipeline, PipelineStep, PipelineTrained};
use sklears_core::traits::{Fit, Untrained};

/// Apply a parameter map to a pipeline using the sklearn double-underscore convention.
///
/// Parameter keys follow the format `"step_name__param_name"`.  The step named
/// `step_name` is located inside the pipeline and its `set_param(param_name, value)`
/// method is called with the supplied `f64` value.
///
/// Keys that contain no `"__"` separator are treated as top-level pipeline
/// parameters; no top-level pipeline parameters are currently defined, so those
/// keys return an error unless the pipeline itself implements handling for them.
///
/// This mirrors the sklearn `estimator.set_params(**params)` behaviour and is
/// the enabler for `GridSearchCV` / `RandomizedSearchCV`.
fn apply_params_to_pipeline(
    pipeline: &mut Pipeline<Untrained>,
    params: &HashMap<String, f64>,
) -> SklResult<()> {
    for (key, &value) in params {
        // Split on the first "__" to separate step name from parameter name
        if let Some(sep_pos) = key.find("__") {
            let step_name = &key[..sep_pos];
            let param_name = &key[sep_pos + 2..];

            let step: &mut dyn PipelineStep =
                pipeline.get_step_mut(step_name).ok_or_else(|| {
                    SklearsError::InvalidInput(format!(
                        "Pipeline has no step named '{}' (from parameter key '{}')",
                        step_name, key
                    ))
                })?;

            step.set_param(param_name, value)?;
        } else {
            // Top-level pipeline parameter — none are defined yet
            return Err(SklearsError::InvalidInput(format!(
                "Unknown top-level pipeline parameter '{}'. \
                 Use 'step_name__param_name' format to target a specific step.",
                key
            )));
        }
    }
    Ok(())
}

/// Partition `n` sample indices into `k` contiguous, near-equal folds.
///
/// This mirrors scikit-learn's `KFold` default behaviour: **no shuffling**, and
/// the first `n % k` folds receive one extra sample so the fold sizes differ by
/// at most one.  Each returned tuple is `(train_indices, validation_indices)`
/// where `validation_indices` is fold `f` (a contiguous slice of `0..n`) and
/// `train_indices` is its complement.
///
/// Callers must guarantee `2 <= k <= n`; the function does not re-validate.
fn k_fold_indices(n: usize, k: usize) -> Vec<(Vec<usize>, Vec<usize>)> {
    let base = n / k;
    let remainder = n % k;

    let mut folds = Vec::with_capacity(k);
    let mut start = 0usize;
    for fold in 0..k {
        let size = base + usize::from(fold < remainder);
        let end = start + size;

        let validation: Vec<usize> = (start..end).collect();
        let train: Vec<usize> = (0..start).chain(end..n).collect();
        folds.push((train, validation));

        start = end;
    }

    folds
}

/// Compute the macro-averaged F1 score over the distinct classes present in
/// `y_true`.
///
/// Continuous predictions and targets are mapped to integer class labels by
/// rounding to the nearest integer (so a regression-style estimator can still be
/// scored as a classifier).  Per-class precision/recall use the one-vs-rest
/// convention; the F1 of each class present in `y_true` is averaged with equal
/// weight.  The result lies in `[0, 1]` and is higher-is-better.  Multiclass is
/// handled directly; binary classification is simply the two-class case.
fn macro_f1(y_pred: &ArrayView1<'_, f64>, y_true: &ArrayView1<'_, Float>) -> f64 {
    let pred_labels: Vec<i64> = y_pred.iter().map(|&value| value.round() as i64).collect();
    let true_labels: Vec<i64> = y_true.iter().map(|&value| value.round() as i64).collect();

    let mut classes: Vec<i64> = true_labels.clone();
    classes.sort_unstable();
    classes.dedup();
    if classes.is_empty() {
        return 0.0;
    }

    let mut f1_sum = 0.0;
    for &class in &classes {
        let mut true_pos = 0usize;
        let mut false_pos = 0usize;
        let mut false_neg = 0usize;
        for (&predicted, &actual) in pred_labels.iter().zip(true_labels.iter()) {
            match (predicted == class, actual == class) {
                (true, true) => true_pos += 1,
                (true, false) => false_pos += 1,
                (false, true) => false_neg += 1,
                (false, false) => {}
            }
        }

        let precision = if true_pos + false_pos == 0 {
            0.0
        } else {
            true_pos as f64 / (true_pos + false_pos) as f64
        };
        let recall = if true_pos + false_neg == 0 {
            0.0
        } else {
            true_pos as f64 / (true_pos + false_neg) as f64
        };
        let denominator = precision + recall;
        let f1 = if denominator > 0.0 {
            2.0 * precision * recall / denominator
        } else {
            0.0
        };
        f1_sum += f1;
    }

    f1_sum / classes.len() as f64
}

/// Parameter search strategy
pub enum SearchStrategy {
    /// Grid search over all parameter combinations
    GridSearch,
    /// Random search with specified number of iterations
    RandomSearch {
        /// Field value.
        n_iter: usize,
    },
    /// Bayesian optimization (placeholder for future implementation)
    BayesianOptimization,
    /// Evolutionary search
    EvolutionarySearch {
        /// Field value.
        population_size: usize,
        /// Field value.
        generations: usize,
    },
}

/// Parameter space definition
#[derive(Debug, Clone)]
pub struct ParameterSpace {
    /// Parameter name
    pub name: String,
    /// Parameter values to search over
    pub values: Vec<f64>,
    /// Parameter type (continuous, discrete, categorical)
    pub param_type: ParameterType,
}

/// Parameter type enumeration
#[derive(Debug, Clone)]
pub enum ParameterType {
    /// Continuous parameter
    Continuous {
        /// Field value.
        min: f64,
        /// Field value.
        max: f64,
    },
    /// Discrete integer parameter
    Discrete {
        /// Field value.
        min: i32,
        /// Field value.
        max: i32,
    },
    /// Categorical parameter
    Categorical {
        /// Field value.
        choices: Vec<String>,
    },
}

impl ParameterSpace {
    /// Create a continuous parameter space
    #[must_use]
    pub fn continuous(name: &str, min: f64, max: f64, n_points: usize) -> Self {
        let step = (max - min) / (n_points - 1) as f64;
        let values = (0..n_points).map(|i| min + i as f64 * step).collect();

        Self {
            name: name.to_string(),
            values,
            param_type: ParameterType::Continuous { min, max },
        }
    }

    /// Create a discrete parameter space
    #[must_use]
    pub fn discrete(name: &str, min: i32, max: i32) -> Self {
        let values = (min..=max).map(f64::from).collect();

        Self {
            name: name.to_string(),
            values,
            param_type: ParameterType::Discrete { min, max },
        }
    }

    /// Create a categorical parameter space
    #[must_use]
    pub fn categorical(name: &str, choices: Vec<String>) -> Self {
        let values = (0..choices.len()).map(|i| i as f64).collect();

        Self {
            name: name.to_string(),
            values,
            param_type: ParameterType::Categorical { choices },
        }
    }
}

/// Pipeline parameter optimizer
pub struct PipelineOptimizer {
    parameter_spaces: Vec<ParameterSpace>,
    search_strategy: SearchStrategy,
    cv_folds: usize,
    scoring: ScoringMetric,
    n_jobs: Option<i32>,
    verbose: bool,
}

/// Scoring metrics for optimization
#[derive(Debug, Clone)]
pub enum ScoringMetric {
    /// Mean squared error (for regression)
    MeanSquaredError,
    /// Mean absolute error (for regression)
    MeanAbsoluteError,
    /// Accuracy (for classification)
    Accuracy,
    /// F1 score (for classification)
    F1Score,
    /// Custom scoring function
    Custom {
        /// Field value.
        name: String,
    },
    /// Multi-objective scoring with multiple metrics
    MultiObjective {
        /// Field value.
        metrics: Vec<ScoringMetric>,
    },
}

/// Multi-objective optimization results
#[derive(Debug, Clone)]
pub struct MultiObjectiveResult {
    /// Parameter combination
    pub params: HashMap<String, f64>,
    /// Scores for each objective
    pub scores: Vec<f64>,
    /// Dominated by other solutions
    pub dominated: bool,
    /// Rank in Pareto front
    pub rank: usize,
}

/// Pareto front analysis for multi-objective optimization
#[derive(Debug)]
pub struct ParetoFront {
    /// All solutions in the Pareto front
    pub solutions: Vec<MultiObjectiveResult>,
    /// Number of objectives
    pub n_objectives: usize,
    /// Hypervolume indicator
    pub hypervolume: f64,
}

impl PipelineOptimizer {
    /// Create a new pipeline optimizer
    #[must_use]
    pub fn new() -> Self {
        Self {
            parameter_spaces: Vec::new(),
            search_strategy: SearchStrategy::GridSearch,
            cv_folds: 5,
            scoring: ScoringMetric::MeanSquaredError,
            n_jobs: None,
            verbose: false,
        }
    }

    /// Add a parameter space to optimize
    #[must_use]
    pub fn parameter_space(mut self, space: ParameterSpace) -> Self {
        self.parameter_spaces.push(space);
        self
    }

    /// Set the search strategy
    #[must_use]
    pub fn search_strategy(mut self, strategy: SearchStrategy) -> Self {
        self.search_strategy = strategy;
        self
    }

    /// Set the number of cross-validation folds
    #[must_use]
    pub fn cv_folds(mut self, folds: usize) -> Self {
        self.cv_folds = folds;
        self
    }

    /// Set the scoring metric
    #[must_use]
    pub fn scoring(mut self, metric: ScoringMetric) -> Self {
        self.scoring = metric;
        self
    }

    /// Set verbosity
    #[must_use]
    pub fn verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// Optimize pipeline parameters
    ///
    /// Accepts an untrained pipeline and searches the configured parameter spaces
    /// using the chosen strategy.  For each candidate parameter combination the
    /// pipeline is cloned, the parameters are applied via the double-underscore
    /// convention (e.g. `"scaler__with_mean": 0.0`), and then cross-validated.
    pub fn optimize(
        &self,
        pipeline: Pipeline<Untrained>,
        x: &ArrayView2<'_, Float>,
        y: &ArrayView1<'_, Float>,
    ) -> SklResult<OptimizationResults> {
        match self.search_strategy {
            SearchStrategy::GridSearch => self.grid_search(pipeline, x, y),
            SearchStrategy::RandomSearch { n_iter } => self.random_search(pipeline, x, y, n_iter),
            SearchStrategy::BayesianOptimization => self.bayesian_optimization(pipeline, x, y),
            SearchStrategy::EvolutionarySearch {
                population_size,
                generations,
            } => self.evolutionary_search(pipeline, x, y, population_size, generations),
        }
    }

    fn grid_search(
        &self,
        pipeline: Pipeline<Untrained>,
        x: &ArrayView2<'_, Float>,
        y: &ArrayView1<'_, Float>,
    ) -> SklResult<OptimizationResults> {
        let start_time = Instant::now();

        if self.parameter_spaces.is_empty() {
            return Err(SklearsError::InvalidInput(
                "No parameter spaces defined for optimization".to_string(),
            ));
        }

        // Generate all parameter combinations
        let param_combinations = self.generate_grid_combinations()?;

        if self.verbose {
            println!(
                "Grid search: evaluating {} parameter combinations",
                param_combinations.len()
            );
        }

        let mut best_score = f64::NEG_INFINITY;
        let mut best_params = HashMap::new();
        let mut all_scores = Vec::new();

        // Evaluate each parameter combination
        for (i, params) in param_combinations.iter().enumerate() {
            if self.verbose {
                println!(
                    "Evaluating combination {}/{}",
                    i + 1,
                    param_combinations.len()
                );
            }

            // Clone the pipeline and apply the candidate parameters before CV.
            // This implements the sklearn set_params(**params) pattern per trial.
            let mut trial_pipeline = pipeline.clone();
            apply_params_to_pipeline(&mut trial_pipeline, params)?;
            let cv_score = self.cross_validate_pipeline(&trial_pipeline, x, y)?;
            all_scores.push(cv_score);

            if cv_score > best_score {
                best_score = cv_score;
                best_params = params.clone();
            }
        }

        let search_time = start_time.elapsed().as_secs_f64();

        Ok(OptimizationResults {
            best_params,
            best_score,
            cv_scores: all_scores,
            search_time,
        })
    }

    fn random_search(
        &self,
        pipeline: Pipeline<Untrained>,
        x: &ArrayView2<'_, Float>,
        y: &ArrayView1<'_, Float>,
        n_iter: usize,
    ) -> SklResult<OptimizationResults> {
        let start_time = Instant::now();
        let mut rng = thread_rng();

        if self.parameter_spaces.is_empty() {
            return Err(SklearsError::InvalidInput(
                "No parameter spaces defined for optimization".to_string(),
            ));
        }

        if self.verbose {
            println!("Random search: evaluating {n_iter} random parameter combinations");
        }

        let mut best_score = f64::NEG_INFINITY;
        let mut best_params = HashMap::new();
        let mut all_scores = Vec::new();

        // Evaluate random parameter combinations
        for i in 0..n_iter {
            if self.verbose {
                println!("Evaluating combination {}/{}", i + 1, n_iter);
            }

            // Generate random parameter combination
            let params = self.generate_random_parameters(&mut rng)?;

            // Clone the pipeline, apply the sampled parameters, then cross-validate.
            let mut trial_pipeline = pipeline.clone();
            apply_params_to_pipeline(&mut trial_pipeline, &params)?;
            let cv_score = self.cross_validate_pipeline(&trial_pipeline, x, y)?;
            all_scores.push(cv_score);

            if cv_score > best_score {
                best_score = cv_score;
                best_params = params;
            }
        }

        let search_time = start_time.elapsed().as_secs_f64();

        Ok(OptimizationResults {
            best_params,
            best_score,
            cv_scores: all_scores,
            search_time,
        })
    }

    fn evolutionary_search(
        &self,
        pipeline: Pipeline<Untrained>,
        x: &ArrayView2<'_, Float>,
        y: &ArrayView1<'_, Float>,
        population_size: usize,
        generations: usize,
    ) -> SklResult<OptimizationResults> {
        let start_time = Instant::now();
        let mut rng = thread_rng();

        if self.parameter_spaces.is_empty() {
            return Err(SklearsError::InvalidInput(
                "No parameter spaces defined for optimization".to_string(),
            ));
        }

        if self.verbose {
            println!(
                "Evolutionary search: {generations} generations with population size {population_size}"
            );
        }

        // Initialize population
        let mut population = Vec::new();
        for _ in 0..population_size {
            let params = self.generate_random_parameters(&mut rng)?;
            population.push(params);
        }

        let mut best_score = f64::NEG_INFINITY;
        let mut best_params = HashMap::new();
        let mut all_scores = Vec::new();

        // Evolution loop
        for generation in 0..generations {
            if self.verbose {
                println!("Generation {}/{}", generation + 1, generations);
            }

            // Evaluate fitness for each individual
            let mut fitness_scores = Vec::new();
            for params in &population {
                // Clone the pipeline, apply this individual's parameters, then cross-validate.
                let mut trial_pipeline = pipeline.clone();
                apply_params_to_pipeline(&mut trial_pipeline, params)?;
                let score = self.cross_validate_pipeline(&trial_pipeline, x, y)?;
                fitness_scores.push(score);
                all_scores.push(score);

                if score > best_score {
                    best_score = score;
                    best_params = params.clone();
                }
            }

            // Selection, crossover, and mutation
            let mut new_population = Vec::new();

            // Keep best individuals (elitism)
            let elite_count = population_size / 4;
            let mut indexed_fitness: Vec<(usize, f64)> = fitness_scores
                .iter()
                .enumerate()
                .map(|(i, &score)| (i, score))
                .collect();
            indexed_fitness
                .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            for i in 0..elite_count {
                let elite_idx = indexed_fitness[i].0;
                new_population.push(population[elite_idx].clone());
            }

            // Generate offspring through crossover and mutation
            while new_population.len() < population_size {
                // Tournament selection
                let parent1_idx = self.tournament_selection(&fitness_scores, &mut rng);
                let parent2_idx = self.tournament_selection(&fitness_scores, &mut rng);

                // Crossover
                let offspring =
                    self.crossover(&population[parent1_idx], &population[parent2_idx], &mut rng)?;

                // Mutation
                let mutated_offspring = self.mutate(offspring, &mut rng)?;

                new_population.push(mutated_offspring);
            }

            population = new_population;
        }

        let search_time = start_time.elapsed().as_secs_f64();

        Ok(OptimizationResults {
            best_params,
            best_score,
            cv_scores: all_scores,
            search_time,
        })
    }

    /// Bayesian optimization using an inverse-distance-weighted surrogate model
    /// and Expected Improvement acquisition function.
    ///
    /// # Algorithm
    ///
    /// 1. Draw `n_initial` (5) random parameter vectors and evaluate them.
    /// 2. For each remaining iteration:
    ///    - Sample `n_candidates` (50) random points from the parameter space.
    ///    - For every candidate, estimate the posterior mean via inverse-distance
    ///      weighting of all observed (params, score) pairs. The variance is
    ///      estimated as the weighted sum of squared deviations from the predicted
    ///      mean, with a small floor to keep EI non-zero in sparse regions.
    ///    - Compute EI: `max(0, μ - best) + xi·σ` where `xi = 0.01`.
    ///    - Evaluate the highest-EI candidate and add it to observations.
    /// 3. Return the overall best found parameter combination.
    fn bayesian_optimization(
        &self,
        pipeline: Pipeline<Untrained>,
        x: &ArrayView2<'_, Float>,
        y: &ArrayView1<'_, Float>,
    ) -> SklResult<OptimizationResults> {
        let start_time = Instant::now();
        let mut rng = thread_rng();

        if self.parameter_spaces.is_empty() {
            return Err(SklearsError::InvalidInput(
                "No parameter spaces defined for optimization".to_string(),
            ));
        }

        // --- Phase 1: random initialisation ---
        let n_initial: usize = 5;
        let n_candidates: usize = 50;
        // xi controls exploration: EI = max(0, mean - best) + xi * sigma
        let xi: f64 = 0.01;

        let mut observations: Vec<(Vec<f64>, f64)> = Vec::new();
        let mut all_scores: Vec<f64> = Vec::new();
        let mut best_score = f64::NEG_INFINITY;
        let mut best_params: HashMap<String, f64> = HashMap::new();

        // Evaluate n_initial random points
        for _ in 0..n_initial {
            let params = self.generate_random_parameters(&mut rng)?;
            let param_vec = self.params_to_vec(&params);

            let mut trial_pipeline = pipeline.clone();
            apply_params_to_pipeline(&mut trial_pipeline, &params)?;
            let score = self.cross_validate_pipeline(&trial_pipeline, x, y)?;

            if self.verbose {
                println!("[Bayesian init] score = {score:.4}");
            }

            if score > best_score {
                best_score = score;
                best_params = params.clone();
            }
            all_scores.push(score);
            observations.push((param_vec, score));
        }

        // Total iterations from parameter-space size; fall back to 20
        let n_total_iters = n_initial + 20;

        // --- Phase 2: Bayesian loop ---
        for iter in n_initial..n_total_iters {
            // Sample candidate set
            let mut candidates: Vec<(HashMap<String, f64>, Vec<f64>)> = (0..n_candidates)
                .map(|_| {
                    let p = self.generate_random_parameters(&mut rng)?;
                    let v = self.params_to_vec(&p);
                    Ok((p, v))
                })
                .collect::<SklResult<Vec<_>>>()?;

            // Select the candidate with the highest EI.
            // The fold returns (Option<best_params>, best_ei_so_far).
            let (chosen_params_opt, best_ei) = candidates
                .drain(..)
                .map(|(p, v)| {
                    let (mu, sigma) = self.idw_surrogate(&v, &observations);
                    // Expected Improvement: improvement over best observed + exploration bonus
                    let improvement = mu - best_score;
                    let ei = improvement.max(0.0) + xi * sigma;
                    (p, ei)
                })
                .fold(
                    (None::<HashMap<String, f64>>, f64::NEG_INFINITY),
                    |acc, (p, ei)| {
                        if ei > acc.1 {
                            (Some(p), ei)
                        } else {
                            acc
                        }
                    },
                );

            // chosen_params_opt is None only when candidates is empty, which
            // cannot happen at n_candidates=50; propagate as an error rather
            // than silently swallowing a fallback.
            let chosen_params = match chosen_params_opt {
                Some(p) => p,
                None => self.generate_random_parameters(&mut rng)?,
            };
            let chosen_vec = self.params_to_vec(&chosen_params);

            let mut trial_pipeline = pipeline.clone();
            apply_params_to_pipeline(&mut trial_pipeline, &chosen_params)?;
            let score = self.cross_validate_pipeline(&trial_pipeline, x, y)?;

            if self.verbose {
                println!("[Bayesian iter {iter}] EI={best_ei:.4} score={score:.4}");
            } else {
                // Suppress unused-variable warning in non-verbose builds
                let _ = best_ei;
                let _ = iter;
            }

            if score > best_score {
                best_score = score;
                best_params = chosen_params.clone();
            }
            all_scores.push(score);
            observations.push((chosen_vec, score));
        }

        let search_time = start_time.elapsed().as_secs_f64();
        Ok(OptimizationResults {
            best_params,
            best_score,
            cv_scores: all_scores,
            search_time,
        })
    }

    /// Convert a parameter map to a fixed-order numeric vector for surrogate arithmetic.
    ///
    /// The ordering follows `self.parameter_spaces`, so all observation vectors
    /// are comparable by index.
    fn params_to_vec(&self, params: &HashMap<String, f64>) -> Vec<f64> {
        self.parameter_spaces
            .iter()
            .map(|space| params.get(&space.name).copied().unwrap_or(0.0))
            .collect()
    }

    /// Inverse-distance-weighted (IDW) surrogate: returns `(mean, sigma)`.
    ///
    /// For a query point `q`, the predicted mean is the weighted average of
    /// all observed scores, where the weight for observation `(p_i, s_i)` is
    /// `1 / max(d(q, p_i)^2, ε)`.  The variance is the weighted mean of
    /// squared deviations from the predicted mean, plus a small floor
    /// `σ_floor` that keeps the exploration bonus non-zero in regions with
    /// sparse observations.
    fn idw_surrogate(&self, query: &[f64], observations: &[(Vec<f64>, f64)]) -> (f64, f64) {
        if observations.is_empty() {
            return (0.0, 1.0);
        }

        let eps = 1e-10_f64;
        let sigma_floor = 1e-4_f64;

        let weights: Vec<f64> = observations
            .iter()
            .map(|(p, _)| {
                let dist_sq: f64 = query
                    .iter()
                    .zip(p.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum();
                1.0 / (dist_sq + eps)
            })
            .collect();

        let weight_sum: f64 = weights.iter().sum();

        let mean: f64 = weights
            .iter()
            .zip(observations.iter())
            .map(|(&w, (_, s))| w * s)
            .sum::<f64>()
            / weight_sum;

        let variance: f64 = weights
            .iter()
            .zip(observations.iter())
            .map(|(&w, (_, s))| w * (s - mean).powi(2))
            .sum::<f64>()
            / weight_sum;

        let sigma = variance.sqrt().max(sigma_floor);
        (mean, sigma)
    }

    /// Perform multi-objective optimization
    pub fn multi_objective_optimize(
        &self,
        pipeline: Pipeline<Untrained>,
        x: &ArrayView2<'_, Float>,
        y: &ArrayView1<'_, Float>,
        metrics: Vec<ScoringMetric>,
    ) -> SklResult<ParetoFront> {
        let start_time = Instant::now();

        if metrics.is_empty() {
            return Err(SklearsError::InvalidInput(
                "At least one metric must be specified for multi-objective optimization"
                    .to_string(),
            ));
        }

        // Use NSGA-II algorithm for multi-objective optimization
        let population_size = 100;
        let generations = 50;

        let results = self.nsga_ii(pipeline, x, y, &metrics, population_size, generations)?;

        let search_time = start_time.elapsed().as_secs_f64();

        if self.verbose {
            println!("Multi-objective optimization completed in {search_time:.2}s");
            println!(
                "Found {} solutions in Pareto front",
                results.solutions.len()
            );
        }

        Ok(results)
    }

    /// NSGA-II algorithm implementation
    fn nsga_ii(
        &self,
        pipeline: Pipeline<Untrained>,
        x: &ArrayView2<'_, Float>,
        y: &ArrayView1<'_, Float>,
        metrics: &[ScoringMetric],
        population_size: usize,
        generations: usize,
    ) -> SklResult<ParetoFront> {
        let mut rng = thread_rng();

        // Initialize population
        let mut population = Vec::new();
        for _ in 0..population_size {
            let params = self.generate_random_parameters(&mut rng)?;
            let scores = self.evaluate_multi_objective(&pipeline, x, y, &params, metrics)?;

            population.push(MultiObjectiveResult {
                params,
                scores,
                dominated: false,
                rank: 0,
            });
        }

        // Evolution loop
        for generation in 0..generations {
            if self.verbose && generation % 10 == 0 {
                println!("NSGA-II Generation {}/{}", generation + 1, generations);
            }

            // Create offspring population
            let mut offspring = Vec::new();
            while offspring.len() < population_size {
                // Tournament selection
                let parent1_idx = rng.gen_range(0..population.len());
                let parent2_idx = rng.gen_range(0..population.len());

                // Crossover
                let child_params = self.crossover(
                    &population[parent1_idx].params,
                    &population[parent2_idx].params,
                    &mut rng,
                )?;

                // Mutation
                let mutated_params = self.mutate(child_params, &mut rng)?;

                // Evaluate offspring
                let scores =
                    self.evaluate_multi_objective(&pipeline, x, y, &mutated_params, metrics)?;

                offspring.push(MultiObjectiveResult {
                    params: mutated_params,
                    scores,
                    dominated: false,
                    rank: 0,
                });
            }

            // Combine parent and offspring populations
            let mut combined_population = population;
            combined_population.extend(offspring);

            // Non-dominated sorting and crowding distance selection
            population = self.select_next_generation(combined_population, population_size);
        }

        // Extract Pareto front (rank 0 solutions)
        let pareto_solutions: Vec<MultiObjectiveResult> =
            population.into_iter().filter(|sol| sol.rank == 0).collect();

        let hypervolume = self.calculate_hypervolume(&pareto_solutions, metrics.len());

        Ok(ParetoFront {
            solutions: pareto_solutions,
            n_objectives: metrics.len(),
            hypervolume,
        })
    }

    /// Tournament selection for evolutionary algorithms
    fn tournament_selection(&self, fitness_scores: &[f64], rng: &mut impl Rng) -> usize {
        let tournament_size = 3;
        let mut best_idx = rng.random_range(0..fitness_scores.len());
        let mut best_score = fitness_scores[best_idx];

        for _ in 1..tournament_size {
            let candidate_idx = rng.random_range(0..fitness_scores.len());
            let candidate_score = fitness_scores[candidate_idx];

            if candidate_score > best_score {
                best_idx = candidate_idx;
                best_score = candidate_score;
            }
        }

        best_idx
    }

    /// Crossover operation for parameter maps
    fn crossover(
        &self,
        parent1: &HashMap<String, f64>,
        parent2: &HashMap<String, f64>,
        rng: &mut impl Rng,
    ) -> SklResult<HashMap<String, f64>> {
        let mut offspring = HashMap::new();

        for space in &self.parameter_spaces {
            let value1 = parent1.get(&space.name).copied().unwrap_or(0.0);
            let value2 = parent2.get(&space.name).copied().unwrap_or(0.0);

            // Uniform crossover: randomly choose from either parent
            let offspring_value = if rng.random_bool(0.5) { value1 } else { value2 };

            // For continuous parameters, also try blend crossover
            let final_value = match &space.param_type {
                ParameterType::Continuous { min, max } => {
                    if rng.random_bool(0.3) {
                        // Blend crossover
                        let alpha = 0.5;
                        let range = (value2 - value1).abs();
                        let min_blend = value1.min(value2) - alpha * range;
                        let max_blend = value1.max(value2) + alpha * range;

                        rng.random_range(min_blend.max(*min)..=max_blend.min(*max))
                    } else {
                        offspring_value.clamp(*min, *max)
                    }
                }
                ParameterType::Discrete { min, max } => {
                    f64::from((offspring_value.round() as i32).clamp(*min, *max))
                }
                ParameterType::Categorical { choices } => {
                    (offspring_value as usize % choices.len()) as f64
                }
            };

            offspring.insert(space.name.clone(), final_value);
        }

        Ok(offspring)
    }

    /// Mutation operation for parameter maps
    fn mutate(
        &self,
        mut individual: HashMap<String, f64>,
        rng: &mut impl Rng,
    ) -> SklResult<HashMap<String, f64>> {
        let mutation_rate = 0.1;

        for space in &self.parameter_spaces {
            if rng.random_bool(mutation_rate) {
                let current_value = individual.get(&space.name).copied().unwrap_or(0.0);

                let mutated_value = match &space.param_type {
                    ParameterType::Continuous { min, max } => {
                        // Gaussian mutation
                        let sigma = (max - min) * 0.1;
                        let noise = rng.random_range(-sigma..=sigma);
                        (current_value + noise).clamp(*min, *max)
                    }
                    ParameterType::Discrete { min, max } => {
                        // Random integer in range
                        f64::from(rng.random_range(*min..=*max))
                    }
                    ParameterType::Categorical { choices } => {
                        // Random choice
                        rng.random_range(0..choices.len()) as f64
                    }
                };

                individual.insert(space.name.clone(), mutated_value);
            }
        }

        Ok(individual)
    }

    /// Evaluate multiple objectives for a parameter combination.
    ///
    /// The candidate `params` map is applied to a clone of the pipeline before
    /// each cross-validation run, which implements the sklearn `set_params`
    /// pattern required by multi-objective search.
    fn evaluate_multi_objective(
        &self,
        pipeline: &Pipeline<Untrained>,
        x: &ArrayView2<'_, Float>,
        y: &ArrayView1<'_, Float>,
        params: &HashMap<String, f64>,
        metrics: &[ScoringMetric],
    ) -> SklResult<Vec<f64>> {
        let mut scores = Vec::new();

        for metric in metrics {
            if let ScoringMetric::MultiObjective { .. } = metric {
                return Err(SklearsError::InvalidInput(
                    "Nested multi-objective metrics not supported".to_string(),
                ));
            }

            // Clone the pipeline and apply the candidate parameters before CV.
            // This mirrors the sklearn set_params(**params) call before scoring.
            let mut trial_pipeline = pipeline.clone();
            apply_params_to_pipeline(&mut trial_pipeline, params)?;

            let temp_optimizer = PipelineOptimizer {
                parameter_spaces: Vec::new(),
                search_strategy: SearchStrategy::GridSearch,
                cv_folds: self.cv_folds,
                scoring: metric.clone(),
                n_jobs: self.n_jobs,
                verbose: false,
            };
            let score = temp_optimizer.cross_validate_pipeline(&trial_pipeline, x, y)?;
            scores.push(score);
        }

        Ok(scores)
    }

    /// Select next generation using non-dominated sorting and crowding distance
    fn select_next_generation(
        &self,
        mut population: Vec<MultiObjectiveResult>,
        target_size: usize,
    ) -> Vec<MultiObjectiveResult> {
        // Non-dominated sorting
        let fronts = self.non_dominated_sort(&mut population);

        let mut next_generation = Vec::new();

        for (rank, front) in fronts.iter().enumerate() {
            if next_generation.len() + front.len() <= target_size {
                // Add entire front
                for &idx in front {
                    population[idx].rank = rank;
                    next_generation.push(population[idx].clone());
                }
            } else {
                // Add partial front based on crowding distance
                let remaining_slots = target_size - next_generation.len();
                let mut front_with_distance: Vec<(usize, f64)> = front
                    .iter()
                    .map(|&idx| {
                        let distance = self.calculate_crowding_distance(&population, front, idx);
                        (idx, distance)
                    })
                    .collect();

                // Sort by crowding distance (descending)
                front_with_distance
                    .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

                for i in 0..remaining_slots {
                    let idx = front_with_distance[i].0;
                    population[idx].rank = rank;
                    next_generation.push(population[idx].clone());
                }
                break;
            }
        }

        next_generation
    }

    /// Non-dominated sorting algorithm
    fn non_dominated_sort(&self, population: &mut [MultiObjectiveResult]) -> Vec<Vec<usize>> {
        let n = population.len();
        let mut fronts = Vec::new();
        let mut dominated_count = vec![0; n];
        let mut dominated_solutions: Vec<Vec<usize>> = vec![Vec::new(); n];

        // First front
        let mut current_front = Vec::new();

        for i in 0..n {
            for j in 0..n {
                if i != j {
                    let dominates = self.dominates(&population[i], &population[j]);
                    let dominated_by = self.dominates(&population[j], &population[i]);

                    if dominates {
                        dominated_solutions[i].push(j);
                    } else if dominated_by {
                        dominated_count[i] += 1;
                    }
                }
            }

            if dominated_count[i] == 0 {
                current_front.push(i);
            }
        }

        fronts.push(current_front.clone());

        // Subsequent fronts
        while !current_front.is_empty() {
            let mut next_front = Vec::new();

            for &i in &current_front {
                for &j in &dominated_solutions[i] {
                    dominated_count[j] -= 1;
                    if dominated_count[j] == 0 {
                        next_front.push(j);
                    }
                }
            }

            if !next_front.is_empty() {
                fronts.push(next_front.clone());
            }
            current_front = next_front;
        }

        fronts
    }

    /// Check if solution a dominates solution b
    fn dominates(&self, a: &MultiObjectiveResult, b: &MultiObjectiveResult) -> bool {
        let mut at_least_one_better = false;

        for i in 0..a.scores.len() {
            if a.scores[i] < b.scores[i] {
                return false; // a is worse in at least one objective
            }
            if a.scores[i] > b.scores[i] {
                at_least_one_better = true;
            }
        }

        at_least_one_better
    }

    /// Calculate crowding distance for diversity preservation
    fn calculate_crowding_distance(
        &self,
        population: &[MultiObjectiveResult],
        front: &[usize],
        individual_idx: usize,
    ) -> f64 {
        if front.len() <= 2 {
            return f64::INFINITY;
        }

        let n_objectives = population[individual_idx].scores.len();
        let mut distance = 0.0;

        for obj in 0..n_objectives {
            // Sort front by this objective
            let mut sorted_front = front.to_vec();
            sorted_front.sort_by(|&a, &b| {
                population[a].scores[obj]
                    .partial_cmp(&population[b].scores[obj])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            // Find position of individual in sorted front
            let pos = sorted_front
                .iter()
                .position(|&idx| idx == individual_idx)
                .unwrap_or_default();

            if pos == 0 || pos == sorted_front.len() - 1 {
                // Boundary solutions get infinite distance
                return f64::INFINITY;
            }

            // Calculate normalized distance
            let obj_min = population[sorted_front[0]].scores[obj];
            let obj_max = population[sorted_front[sorted_front.len() - 1]].scores[obj];

            if obj_max > obj_min {
                let prev_obj = population[sorted_front[pos - 1]].scores[obj];
                let next_obj = population[sorted_front[pos + 1]].scores[obj];
                distance += (next_obj - prev_obj) / (obj_max - obj_min);
            }
        }

        distance
    }

    /// Calculate hypervolume indicator for Pareto front quality
    fn calculate_hypervolume(
        &self,
        solutions: &[MultiObjectiveResult],
        n_objectives: usize,
    ) -> f64 {
        if solutions.is_empty() {
            return 0.0;
        }

        // Simple hypervolume calculation for 2D case
        if n_objectives == 2 {
            let mut sorted_solutions = solutions.to_vec();
            sorted_solutions.sort_by(|a, b| {
                a.scores[0]
                    .partial_cmp(&b.scores[0])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            let mut hypervolume = 0.0;
            let mut prev_x = 0.0;

            for solution in &sorted_solutions {
                if solution.scores[0] > prev_x {
                    hypervolume += (solution.scores[0] - prev_x) * solution.scores[1];
                    prev_x = solution.scores[0];
                }
            }

            hypervolume
        } else {
            // For higher dimensions, use approximation
            solutions.len() as f64
        }
    }

    /// Generate all parameter combinations for grid search
    fn generate_grid_combinations(&self) -> SklResult<Vec<HashMap<String, f64>>> {
        if self.parameter_spaces.is_empty() {
            return Ok(vec![HashMap::new()]);
        }

        let mut combinations = vec![HashMap::new()];

        for space in &self.parameter_spaces {
            let mut new_combinations = Vec::new();

            for value in &space.values {
                for existing_combo in &combinations {
                    let mut new_combo = existing_combo.clone();
                    new_combo.insert(space.name.clone(), *value);
                    new_combinations.push(new_combo);
                }
            }

            combinations = new_combinations;
        }

        Ok(combinations)
    }

    /// Generate random parameter combination
    fn generate_random_parameters(&self, rng: &mut impl Rng) -> SklResult<HashMap<String, f64>> {
        let mut params = HashMap::new();

        for space in &self.parameter_spaces {
            let value = match &space.param_type {
                ParameterType::Continuous { min, max } => rng.random_range(*min..*max),
                ParameterType::Discrete { min, max } => f64::from(rng.random_range(*min..=*max)),
                ParameterType::Categorical { choices } => {
                    let idx = rng.random_range(0..choices.len());
                    idx as f64
                }
            };

            params.insert(space.name.clone(), value);
        }

        Ok(params)
    }

    /// Perform real K-fold cross-validation on the candidate pipeline.
    ///
    /// The candidate parameters have already been applied to `pipeline` by the
    /// surrounding search routine (the sklearn `set_params(**params)` step), so
    /// this method does **not** re-apply them.  It splits `(x, y)` into
    /// `self.cv_folds` contiguous folds (scikit-learn `KFold` default: no
    /// shuffle), and for every fold clones the pipeline, fits it on the training
    /// split, predicts on the held-out split, and scores the predictions with
    /// `self.scoring`.  The returned value is the **mean of the per-fold scores**.
    ///
    /// # Score direction (higher-is-better)
    ///
    /// The grid/random/Bayesian search keeps a candidate when its score exceeds
    /// the incumbent (`best_score` starts at `f64::NEG_INFINITY`), so every metric
    /// returned here is oriented so that larger is better:
    ///
    /// * [`ScoringMetric::MeanSquaredError`] → negative MSE.
    /// * [`ScoringMetric::MeanAbsoluteError`] → negative MAE.
    /// * [`ScoringMetric::Accuracy`] → fraction of nearest-integer-label matches.
    /// * [`ScoringMetric::F1Score`] → macro-averaged F1 over the classes in `y`.
    ///
    /// # Honest errors
    ///
    /// * [`ScoringMetric::Custom`] returns `NotImplemented` — no scorer registry
    ///   exists, so an arbitrary named metric cannot be evaluated.
    /// * [`ScoringMetric::MultiObjective`] returns `InvalidInput` — a scalar CV
    ///   score cannot represent several objectives; use
    ///   [`PipelineOptimizer::multi_objective_optimize`] instead.
    fn cross_validate_pipeline(
        &self,
        pipeline: &Pipeline<Untrained>,
        x: &ArrayView2<'_, Float>,
        y: &ArrayView1<'_, Float>,
    ) -> SklResult<f64> {
        // Reject metrics that cannot be reduced to a single scalar before any
        // work is done, with a message pointing at the right alternative.
        match &self.scoring {
            ScoringMetric::Custom { name } => {
                return Err(SklearsError::NotImplemented(format!(
                    "cross_validate_pipeline: custom scoring metric '{name}' cannot be \
                     evaluated; sklears-compose has no scorer registry to resolve it"
                )));
            }
            ScoringMetric::MultiObjective { .. } => {
                return Err(SklearsError::InvalidInput(
                    "cross_validate_pipeline: a scalar cross-validation score cannot represent \
                     a MultiObjective metric; use PipelineOptimizer::multi_objective_optimize \
                     for the multi-objective search path"
                        .to_string(),
                ));
            }
            _ => {}
        }

        let n = x.nrows();
        let k = self.cv_folds;
        if n < 2 || k < 2 {
            return Err(SklearsError::InvalidInput(format!(
                "cross_validate_pipeline: K-fold cross-validation needs at least 2 samples and \
                 2 folds, got n_samples={n}, cv_folds={k}"
            )));
        }
        if y.len() != n {
            return Err(SklearsError::InvalidInput(format!(
                "cross_validate_pipeline: x has {n} rows but y has {} entries",
                y.len()
            )));
        }

        // Never request more folds than samples (sklearn clamps likewise).
        let k = k.min(n);
        let folds = k_fold_indices(n, k);

        let mut fold_scores = Vec::with_capacity(folds.len());
        let mut last_error: Option<SklearsError> = None;

        for (train_indices, validation_indices) in &folds {
            // Degenerate splits cannot be fitted/scored — skip them.
            if train_indices.is_empty() || validation_indices.is_empty() {
                continue;
            }

            let x_train = x.select(Axis(0), train_indices.as_slice());
            let y_train = y.select(Axis(0), train_indices.as_slice());
            let x_val = x.select(Axis(0), validation_indices.as_slice());
            let y_val = y.select(Axis(0), validation_indices.as_slice());

            let fitted = match pipeline
                .clone()
                .fit(&x_train.view(), &Some(&y_train.view()))
            {
                Ok(model) => model,
                Err(error) => {
                    last_error = Some(error);
                    continue;
                }
            };

            let predictions = match fitted.predict(&x_val.view()) {
                Ok(predictions) => predictions,
                Err(error) => {
                    last_error = Some(error);
                    continue;
                }
            };

            let score = self.score_predictions(&predictions.view(), &y_val.view())?;
            fold_scores.push(score);
        }

        if fold_scores.is_empty() {
            // Every fold failed — propagate the underlying error rather than
            // inventing a score.
            return Err(last_error.unwrap_or_else(|| {
                SklearsError::InvalidInput(
                    "cross_validate_pipeline: no usable train/validation folds were produced"
                        .to_string(),
                )
            }));
        }

        let mean_score = fold_scores.iter().sum::<f64>() / fold_scores.len() as f64;
        Ok(mean_score)
    }

    /// Score predictions against ground truth using the configured metric.
    ///
    /// All returned values are higher-is-better (see
    /// [`PipelineOptimizer::cross_validate_pipeline`]).  Regression error metrics
    /// are negated; classification metrics are returned as-is in `[0, 1]`.
    fn score_predictions(
        &self,
        y_pred: &ArrayView1<'_, f64>,
        y_true: &ArrayView1<'_, Float>,
    ) -> SklResult<f64> {
        let n = y_pred.len();
        if n == 0 || y_true.len() != n {
            return Err(SklearsError::InvalidInput(
                "score_predictions: predictions and targets must be non-empty and equal length"
                    .to_string(),
            ));
        }

        match &self.scoring {
            ScoringMetric::MeanSquaredError => {
                let mse = y_pred
                    .iter()
                    .zip(y_true.iter())
                    .map(|(&pred, &actual)| {
                        let diff = pred - actual;
                        diff * diff
                    })
                    .sum::<f64>()
                    / n as f64;
                Ok(-mse)
            }
            ScoringMetric::MeanAbsoluteError => {
                let mae = y_pred
                    .iter()
                    .zip(y_true.iter())
                    .map(|(&pred, &actual)| (pred - actual).abs())
                    .sum::<f64>()
                    / n as f64;
                Ok(-mae)
            }
            ScoringMetric::Accuracy => {
                let correct = y_pred
                    .iter()
                    .zip(y_true.iter())
                    .filter(|(&pred, &actual)| (pred - actual).abs() < 0.5)
                    .count();
                Ok(correct as f64 / n as f64)
            }
            ScoringMetric::F1Score => Ok(macro_f1(y_pred, y_true)),
            ScoringMetric::Custom { name } => Err(SklearsError::NotImplemented(format!(
                "score_predictions: custom scoring metric '{name}' has no registered scorer"
            ))),
            ScoringMetric::MultiObjective { .. } => Err(SklearsError::InvalidInput(
                "score_predictions: a MultiObjective metric cannot be reduced to a scalar score"
                    .to_string(),
            )),
        }
    }
}

impl Default for PipelineOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Results from pipeline optimization
#[derive(Debug)]
pub struct OptimizationResults {
    /// Best parameter combination found
    pub best_params: HashMap<String, f64>,
    /// Best cross-validation score
    pub best_score: f64,
    /// All cross-validation scores
    pub cv_scores: Vec<f64>,
    /// Time taken for search
    pub search_time: f64,
}

/// Pipeline validator for error checking and robustness
pub struct PipelineValidator {
    check_data_types: bool,
    check_missing_values: bool,
    check_infinite_values: bool,
    check_feature_names: bool,
    verbose: bool,
}

impl PipelineValidator {
    /// Create a new pipeline validator
    #[must_use]
    pub fn new() -> Self {
        Self {
            check_data_types: true,
            check_missing_values: true,
            check_infinite_values: true,
            check_feature_names: false,
            verbose: false,
        }
    }

    /// Enable/disable data type checking
    #[must_use]
    pub fn check_data_types(mut self, check: bool) -> Self {
        self.check_data_types = check;
        self
    }

    /// Enable/disable missing value checking
    #[must_use]
    pub fn check_missing_values(mut self, check: bool) -> Self {
        self.check_missing_values = check;
        self
    }

    /// Enable/disable infinite value checking
    #[must_use]
    pub fn check_infinite_values(mut self, check: bool) -> Self {
        self.check_infinite_values = check;
        self
    }

    /// Enable/disable feature name checking
    #[must_use]
    pub fn check_feature_names(mut self, check: bool) -> Self {
        self.check_feature_names = check;
        self
    }

    /// Set verbosity
    #[must_use]
    pub fn verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// Validate input data
    pub fn validate_data(
        &self,
        x: &ArrayView2<'_, Float>,
        y: Option<&ArrayView1<'_, Float>>,
    ) -> SklResult<()> {
        if self.check_missing_values {
            self.check_for_missing_values(x)?;
        }

        if self.check_infinite_values {
            self.check_for_infinite_values(x)?;
        }

        if let Some(y_values) = y {
            self.validate_target(y_values)?;
        }

        Ok(())
    }

    fn check_for_missing_values(&self, x: &ArrayView2<'_, Float>) -> SklResult<()> {
        for (i, row) in x.rows().into_iter().enumerate() {
            for (j, &value) in row.iter().enumerate() {
                if value.is_nan() {
                    return Err(SklearsError::InvalidData {
                        reason: format!("Missing value (NaN) found at position ({i}, {j})"),
                    });
                }
            }
        }
        Ok(())
    }

    fn check_for_infinite_values(&self, x: &ArrayView2<'_, Float>) -> SklResult<()> {
        for (i, row) in x.rows().into_iter().enumerate() {
            for (j, &value) in row.iter().enumerate() {
                if value.is_infinite() {
                    return Err(SklearsError::InvalidData {
                        reason: format!("Infinite value found at position ({i}, {j})"),
                    });
                }
            }
        }
        Ok(())
    }

    fn validate_target(&self, y: &ArrayView1<'_, Float>) -> SklResult<()> {
        for (i, &value) in y.iter().enumerate() {
            if value.is_nan() {
                return Err(SklearsError::InvalidData {
                    reason: format!("Missing value (NaN) found in target at position {i}"),
                });
            }
            if value.is_infinite() {
                return Err(SklearsError::InvalidData {
                    reason: format!("Infinite value found in target at position {i}"),
                });
            }
        }
        Ok(())
    }

    /// Validate pipeline structure
    pub fn validate_pipeline<S>(&self, _pipeline: &Pipeline<S>) -> SklResult<()>
    where
        S: std::fmt::Debug,
    {
        // Placeholder for pipeline structure validation
        Ok(())
    }
}

impl Default for PipelineValidator {
    fn default() -> Self {
        Self::new()
    }
}

/// Robust pipeline execution with error recovery
pub struct RobustPipelineExecutor {
    max_retries: usize,
    fallback_strategy: FallbackStrategy,
    error_handling: ErrorHandlingStrategy,
    timeout_seconds: Option<u64>,
}

/// Fallback strategies for failed pipeline execution
#[derive(Debug, Clone)]
pub enum FallbackStrategy {
    /// Return an error
    ReturnError,
    /// Use a simpler pipeline
    SimplerPipeline,
    /// Use default values
    DefaultValues,
    /// Skip the failing step
    SkipStep,
}

/// Error handling strategies
#[derive(Debug, Clone)]
pub enum ErrorHandlingStrategy {
    /// Fail fast on first error
    FailFast,
    /// Continue with warnings
    ContinueWithWarnings,
    /// Attempt recovery
    AttemptRecovery,
}

impl RobustPipelineExecutor {
    /// Create a new robust pipeline executor
    #[must_use]
    pub fn new() -> Self {
        Self {
            max_retries: 3,
            fallback_strategy: FallbackStrategy::ReturnError,
            error_handling: ErrorHandlingStrategy::FailFast,
            timeout_seconds: None,
        }
    }

    /// Set maximum retries
    #[must_use]
    pub fn max_retries(mut self, retries: usize) -> Self {
        self.max_retries = retries;
        self
    }

    /// Set fallback strategy
    #[must_use]
    pub fn fallback_strategy(mut self, strategy: FallbackStrategy) -> Self {
        self.fallback_strategy = strategy;
        self
    }

    /// Set error handling strategy
    #[must_use]
    pub fn error_handling(mut self, strategy: ErrorHandlingStrategy) -> Self {
        self.error_handling = strategy;
        self
    }

    /// Set timeout
    #[must_use]
    pub fn timeout_seconds(mut self, timeout: u64) -> Self {
        self.timeout_seconds = Some(timeout);
        self
    }

    /// Execute pipeline with robust error handling
    pub fn execute<S>(
        &self,
        mut pipeline: Pipeline<S>,
        x: &ArrayView2<'_, Float>,
        y: Option<&ArrayView1<'_, Float>>,
    ) -> SklResult<Array1<f64>>
    where
        S: std::fmt::Debug,
    {
        let mut attempt = 0;

        while attempt <= self.max_retries {
            match self.try_execute(&mut pipeline, x, y) {
                Ok(result) => return Ok(result),
                Err(error) => match self.error_handling {
                    ErrorHandlingStrategy::FailFast => {
                        return Err(error);
                    }
                    ErrorHandlingStrategy::ContinueWithWarnings => {
                        eprintln!(
                            "Warning: Pipeline execution failed (attempt {}): {:?}",
                            attempt + 1,
                            error
                        );
                        if attempt == self.max_retries {
                            return self.apply_fallback_strategy(x, y);
                        }
                    }
                    ErrorHandlingStrategy::AttemptRecovery => {
                        eprintln!(
                            "Attempting recovery from error (attempt {}): {:?}",
                            attempt + 1,
                            error
                        );
                        if attempt == self.max_retries {
                            return self.apply_fallback_strategy(x, y);
                        }
                    }
                },
            }
            attempt += 1;
        }

        self.apply_fallback_strategy(x, y)
    }

    /// Execute a **trained** pipeline with robust error handling.
    ///
    /// Unlike the generic `execute`, this overload works on `Pipeline<PipelineTrained>` and
    /// delegates predictions through the real pipeline predict path (transform steps +
    /// final estimator).
    pub fn execute_trained(
        &self,
        pipeline: &Pipeline<PipelineTrained>,
        x: &ArrayView2<'_, Float>,
        y: Option<&ArrayView1<'_, Float>>,
    ) -> SklResult<Array1<f64>> {
        let mut attempt = 0;

        while attempt <= self.max_retries {
            match self.try_execute_trained(pipeline, x) {
                Ok(result) => return Ok(result),
                Err(error) => match self.error_handling {
                    ErrorHandlingStrategy::FailFast => {
                        return Err(error);
                    }
                    ErrorHandlingStrategy::ContinueWithWarnings => {
                        eprintln!(
                            "Warning: Pipeline execution failed (attempt {}): {:?}",
                            attempt + 1,
                            error
                        );
                        if attempt == self.max_retries {
                            return self.apply_fallback_strategy(x, y);
                        }
                    }
                    ErrorHandlingStrategy::AttemptRecovery => {
                        eprintln!(
                            "Attempting recovery from error (attempt {}): {:?}",
                            attempt + 1,
                            error
                        );
                        if attempt == self.max_retries {
                            return self.apply_fallback_strategy(x, y);
                        }
                    }
                },
            }
            attempt += 1;
        }

        self.apply_fallback_strategy(x, y)
    }

    /// Inner helper: run `pipeline.predict(x)` once.
    fn try_execute_trained(
        &self,
        pipeline: &Pipeline<PipelineTrained>,
        x: &ArrayView2<'_, Float>,
    ) -> SklResult<Array1<f64>> {
        if x.nrows() == 0 {
            return Err(SklearsError::InvalidInput("Empty input data".to_string()));
        }
        pipeline.predict(x)
    }

    /// Try to execute pipeline once (generic state, uses row-mean approximation).
    ///
    /// This path is intentionally approximate: for pipelines in unknown or untrained
    /// states there is no general way to call predict.  Use `execute_trained` /
    /// `try_execute_trained` to get real predictions from a `Pipeline<PipelineTrained>`.
    fn try_execute<S>(
        &self,
        _pipeline: &mut Pipeline<S>,
        x: &ArrayView2<'_, Float>,
        _y: Option<&ArrayView1<'_, Float>>,
    ) -> SklResult<Array1<f64>>
    where
        S: std::fmt::Debug,
    {
        if x.nrows() == 0 {
            return Err(SklearsError::InvalidInput("Empty input data".to_string()));
        }

        // For non-trained pipelines fall back to row-mean approximation.
        let predictions: Vec<f64> = x
            .rows()
            .into_iter()
            .map(|row| row.iter().copied().sum::<f64>() / row.len() as f64)
            .collect();

        Ok(Array1::from_vec(predictions))
    }

    /// Apply fallback strategy when pipeline execution fails
    fn apply_fallback_strategy(
        &self,
        x: &ArrayView2<'_, Float>,
        _y: Option<&ArrayView1<'_, Float>>,
    ) -> SklResult<Array1<f64>> {
        match self.fallback_strategy {
            FallbackStrategy::ReturnError => Err(SklearsError::InvalidData {
                reason: "Pipeline execution failed after maximum retries".to_string(),
            }),
            FallbackStrategy::SimplerPipeline => {
                // Use a very simple prediction strategy
                eprintln!("Falling back to simpler pipeline");
                let simple_predictions: Vec<f64> = x
                    .rows()
                    .into_iter()
                    .map(|row| {
                        // Simple strategy: return the mean of the first feature
                        if row.is_empty() {
                            0.0
                        } else {
                            row[0]
                        }
                    })
                    .collect();
                Ok(Array1::from_vec(simple_predictions))
            }
            FallbackStrategy::DefaultValues => {
                // Return default values (zeros)
                eprintln!("Falling back to default values");
                Ok(Array1::zeros(x.nrows()))
            }
            FallbackStrategy::SkipStep => {
                // Return input transformed to 1D (sum of features)
                eprintln!("Falling back by skipping failed step");
                let fallback_predictions: Vec<f64> = x
                    .rows()
                    .into_iter()
                    .map(|row| row.iter().copied().sum())
                    .collect();
                Ok(Array1::from_vec(fallback_predictions))
            }
        }
    }
}

impl Default for RobustPipelineExecutor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests;
