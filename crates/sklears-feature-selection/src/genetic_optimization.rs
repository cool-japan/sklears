//! Genetic Algorithm and Multi-Objective Optimization for Feature Selection
//!
//! This module contains genetic algorithm-based feature selection and multi-objective
//! optimization components for finding optimal feature subsets.

use crate::ensemble_selectors::extract_features;
use crate::{base::SelectorMixin, IndexableTarget};
use scirs2_core::ndarray::{Array1, Array2, Axis};
use scirs2_core::rand_prelude::SliceRandom;
use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use sklears_core::{
    error::{validate, Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Score, Trained, Transform, Untrained},
    types::Float,
};
use std::marker::PhantomData;

/// Genetic Algorithm Feature Selection
/// Uses evolutionary algorithms to find optimal feature subsets
#[derive(Debug, Clone)]
pub struct GeneticSelector<E, State = Untrained> {
    estimator: E,
    population_size: usize,
    generations: usize,
    crossover_rate: f64,
    mutation_rate: f64,
    tournament_size: usize,
    min_features: usize,
    max_features: Option<usize>,
    cv_folds: usize,
    random_state: Option<u64>,
    state: PhantomData<State>,
    // Trained state
    best_features_: Option<Vec<usize>>,
    best_score_: Option<f64>,
    n_features_: Option<usize>,
}

/// Simple cross-validator trait replacement
pub trait CrossValidator {
    fn split(&self, n_samples: usize) -> Vec<(Vec<usize>, Vec<usize>)>;
}

/// Simple KFold implementation replacement
#[derive(Debug, Clone)]
pub struct KFold {
    n_splits: usize,
    shuffle: bool,
}

impl KFold {
    pub fn new(n_splits: usize) -> Self {
        Self {
            n_splits,
            shuffle: false,
        }
    }

    pub fn shuffle(mut self, shuffle: bool) -> Self {
        self.shuffle = shuffle;
        self
    }
}

impl CrossValidator for KFold {
    fn split(&self, n_samples: usize) -> Vec<(Vec<usize>, Vec<usize>)> {
        let mut splits = Vec::new();
        let fold_size = n_samples / self.n_splits;

        for i in 0..self.n_splits {
            let test_start = i * fold_size;
            let test_end = if i == self.n_splits - 1 {
                n_samples
            } else {
                (i + 1) * fold_size
            };

            let test_indices: Vec<usize> = (test_start..test_end).collect();
            let train_indices: Vec<usize> = (0..test_start).chain(test_end..n_samples).collect();

            splits.push((train_indices, test_indices));
        }

        splits
    }
}

impl<E: Clone> GeneticSelector<E, Untrained> {
    /// Create a new GeneticSelector
    pub fn new(estimator: E) -> Self {
        Self {
            estimator,
            population_size: 50,
            generations: 100,
            crossover_rate: 0.8,
            mutation_rate: 0.1,
            tournament_size: 3,
            min_features: 1,
            max_features: None,
            cv_folds: 5,
            random_state: None,
            state: PhantomData,
            best_features_: None,
            best_score_: None,
            n_features_: None,
        }
    }

    /// Set the population size for the genetic algorithm
    pub fn population_size(mut self, size: usize) -> Self {
        if size < 2 {
            panic!("population_size must be at least 2");
        }
        self.population_size = size;
        self
    }

    /// Set the number of generations
    pub fn generations(mut self, generations: usize) -> Self {
        self.generations = generations;
        self
    }

    /// Set the crossover rate (probability of crossover between 0 and 1)
    pub fn crossover_rate(mut self, rate: f64) -> Self {
        if !(0.0..=1.0).contains(&rate) {
            panic!("crossover_rate must be between 0 and 1");
        }
        self.crossover_rate = rate;
        self
    }

    /// Set the mutation rate (probability of mutation between 0 and 1)
    pub fn mutation_rate(mut self, rate: f64) -> Self {
        if !(0.0..=1.0).contains(&rate) {
            panic!("mutation_rate must be between 0 and 1");
        }
        self.mutation_rate = rate;
        self
    }

    /// Set the tournament size for selection
    pub fn tournament_size(mut self, size: usize) -> Self {
        if size < 1 {
            panic!("tournament_size must be at least 1");
        }
        self.tournament_size = size;
        self
    }

    /// Set the minimum number of features
    pub fn min_features(mut self, min: usize) -> Self {
        if min < 1 {
            panic!("min_features must be at least 1");
        }
        self.min_features = min;
        self
    }

    /// Set the maximum number of features
    pub fn max_features(mut self, max: Option<usize>) -> Self {
        self.max_features = max;
        self
    }

    /// Set the number of cross-validation folds
    pub fn cv_folds(mut self, folds: usize) -> Self {
        if folds < 2 {
            panic!("cv_folds must be at least 2");
        }
        self.cv_folds = folds;
        self
    }

    /// Set the random state for reproducibility
    pub fn random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }
}

impl<E> Estimator for GeneticSelector<E, Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl<E, Y> Fit<Array2<Float>, Y> for GeneticSelector<E, Untrained>
where
    E: Clone + Fit<Array2<Float>, Y> + Send + Sync,
    E::Fitted: Score<Array2<Float>, Y> + Send + Sync,
    Y: Clone + IndexableTarget + Send + Sync,
    <E::Fitted as Score<Array2<Float>, Y>>::Float: Into<f64>,
{
    type Fitted = GeneticSelector<E, Trained>;

    fn fit(self, x: &Array2<Float>, y: &Y) -> SklResult<Self::Fitted> {
        let (n_samples, n_features) = x.dim();

        if n_samples < self.cv_folds {
            return Err(SklearsError::InvalidInput(
                "Number of samples must be at least cv_folds".to_string(),
            ));
        }

        // Initialize random number generator
        let mut rng = if let Some(seed) = self.random_state {
            StdRng::seed_from_u64(seed)
        } else {
            StdRng::seed_from_u64(42)
        };

        // Set max_features default to n_features if not specified
        let max_features = self.max_features.unwrap_or(n_features);
        if self.min_features > max_features {
            return Err(SklearsError::InvalidInput(
                "min_features cannot be greater than max_features".to_string(),
            ));
        }

        // Initialize population with random chromosomes
        let mut population = Vec::with_capacity(self.population_size);
        for _ in 0..self.population_size {
            let chromosome =
                generate_random_chromosome(n_features, self.min_features, max_features, &mut rng);
            population.push(chromosome);
        }

        // Evolution loop
        let mut best_score = f64::NEG_INFINITY;
        let mut best_features = Vec::new();

        for generation in 0..self.generations {
            // Evaluate fitness for all chromosomes
            let mut fitness_scores = Vec::with_capacity(self.population_size);

            for chromosome in &population {
                let features: Vec<usize> = chromosome
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &selected)| if selected { Some(i) } else { None })
                    .collect();

                if features.is_empty() {
                    fitness_scores.push(f64::NEG_INFINITY);
                    continue;
                }

                // Extract selected features
                let x_subset = extract_features(x, &features)?;

                // Evaluate using cross-validation
                let score = cross_validate_score(&self.estimator, &x_subset, y, self.cv_folds)?;
                fitness_scores.push(score);

                // Update best solution
                if score > best_score {
                    best_score = score;
                    best_features = features;
                }
            }

            // Create next generation
            let mut new_population = Vec::with_capacity(self.population_size);

            // Elitism: keep best chromosome
            let best_idx = fitness_scores
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap();
            new_population.push(population[best_idx].clone());

            // Generate rest of population through selection, crossover, and mutation
            while new_population.len() < self.population_size {
                // Tournament selection
                let parent1 = tournament_selection(
                    &population,
                    &fitness_scores,
                    self.tournament_size,
                    &mut rng,
                );
                let parent2 = tournament_selection(
                    &population,
                    &fitness_scores,
                    self.tournament_size,
                    &mut rng,
                );

                // Crossover
                let (mut child1, mut child2) = if rng.gen::<f64>() < self.crossover_rate {
                    uniform_crossover(parent1, parent2, &mut rng)
                } else {
                    (parent1.clone(), parent2.clone())
                };

                // Mutation
                if rng.gen::<f64>() < self.mutation_rate {
                    flip_mutation(&mut child1, self.min_features, max_features, &mut rng);
                }
                if rng.gen::<f64>() < self.mutation_rate {
                    flip_mutation(&mut child2, self.min_features, max_features, &mut rng);
                }

                new_population.push(child1.to_vec());
                if new_population.len() < self.population_size {
                    new_population.push(child2.to_vec());
                }
            }

            population = new_population;
        }

        if best_features.is_empty() {
            return Err(SklearsError::InvalidInput(
                "No valid feature subset found during genetic algorithm".to_string(),
            ));
        }

        Ok(GeneticSelector {
            estimator: self.estimator,
            population_size: self.population_size,
            generations: self.generations,
            crossover_rate: self.crossover_rate,
            mutation_rate: self.mutation_rate,
            tournament_size: self.tournament_size,
            min_features: self.min_features,
            max_features: self.max_features,
            cv_folds: self.cv_folds,
            random_state: self.random_state,
            state: PhantomData,
            best_features_: Some(best_features),
            best_score_: Some(best_score),
            n_features_: Some(n_features),
        })
    }
}

impl<E> Transform<Array2<Float>> for GeneticSelector<E, Trained> {
    fn transform(&self, x: &Array2<Float>) -> SklResult<Array2<Float>> {
        validate::check_n_features(x, self.n_features_.unwrap())?;

        let selected_features = self.best_features_.as_ref().unwrap();
        extract_features(x, selected_features)
    }
}

impl<E> SelectorMixin for GeneticSelector<E, Trained> {
    fn get_support(&self) -> SklResult<Array1<bool>> {
        let n_features = self.n_features_.unwrap();
        let selected_features = self.best_features_.as_ref().unwrap();
        let mut support = Array1::from_elem(n_features, false);

        for &idx in selected_features {
            support[idx] = true;
        }

        Ok(support)
    }

    fn transform_features(&self, indices: &[usize]) -> SklResult<Vec<usize>> {
        let selected_features = self.best_features_.as_ref().unwrap();
        Ok(indices
            .iter()
            .filter_map(|&idx| selected_features.iter().position(|&f| f == idx))
            .collect())
    }
}

impl<E> GeneticSelector<E, Trained> {
    /// Get the best feature subset found
    pub fn best_features(&self) -> &[usize] {
        self.best_features_.as_ref().unwrap()
    }

    /// Get the best cross-validation score achieved
    pub fn best_score(&self) -> f64 {
        self.best_score_.unwrap()
    }
}

// Helper functions for genetic algorithm

/// Generate a random chromosome (binary vector representing feature selection)
fn generate_random_chromosome(
    n_features: usize,
    min_features: usize,
    max_features: usize,
    rng: &mut StdRng,
) -> Vec<bool> {
    let mut chromosome = vec![false; n_features];
    let n_selected = rng.gen_range(min_features..=max_features.min(n_features));

    let mut selected_indices: Vec<usize> = (0..n_features).collect();
    for _ in 0..n_selected {
        let idx = rng.gen_range(0..selected_indices.len());
        let feature_idx = selected_indices.swap_remove(idx);
        chromosome[feature_idx] = true;
    }

    chromosome
}

/// Tournament selection
fn tournament_selection<'a>(
    population: &'a [Vec<bool>],
    fitness_scores: &[f64],
    tournament_size: usize,
    rng: &mut StdRng,
) -> &'a Vec<bool> {
    let mut best_idx = rng.gen_range(0..population.len());
    let mut best_fitness = fitness_scores[best_idx];

    for _ in 1..tournament_size {
        let idx = rng.gen_range(0..population.len());
        if fitness_scores[idx] > best_fitness {
            best_idx = idx;
            best_fitness = fitness_scores[idx];
        }
    }

    &population[best_idx]
}

/// Uniform crossover
fn uniform_crossover(
    parent1: &[bool],
    parent2: &[bool],
    rng: &mut StdRng,
) -> (Vec<bool>, Vec<bool>) {
    let mut child1 = Vec::with_capacity(parent1.len());
    let mut child2 = Vec::with_capacity(parent1.len());

    for i in 0..parent1.len() {
        if rng.gen::<bool>() {
            child1.push(parent1[i]);
            child2.push(parent2[i]);
        } else {
            child1.push(parent2[i]);
            child2.push(parent1[i]);
        }
    }

    (child1, child2)
}

/// Flip mutation
fn flip_mutation(
    chromosome: &mut [bool],
    min_features: usize,
    max_features: usize,
    rng: &mut StdRng,
) {
    let current_count = chromosome.iter().filter(|&&x| x).count();

    // Randomly flip a bit
    let flip_idx = rng.gen_range(0..chromosome.len());
    chromosome[flip_idx] = !chromosome[flip_idx];

    // Ensure constraints are satisfied
    let new_count = chromosome.iter().filter(|&&x| x).count();

    if new_count < min_features {
        // Need to add features
        let need_to_add = min_features - new_count;
        let false_indices: Vec<usize> = chromosome
            .iter()
            .enumerate()
            .filter_map(|(i, &val)| if !val { Some(i) } else { None })
            .collect();

        for _ in 0..need_to_add {
            if !false_indices.is_empty() {
                let idx = rng.gen_range(0..false_indices.len());
                chromosome[false_indices[idx]] = true;
            }
        }
    } else if new_count > max_features {
        // Need to remove features
        let need_to_remove = new_count - max_features;
        let true_indices: Vec<usize> = chromosome
            .iter()
            .enumerate()
            .filter_map(|(i, &val)| if val { Some(i) } else { None })
            .collect();

        for _ in 0..need_to_remove {
            if !true_indices.is_empty() {
                let idx = rng.gen_range(0..true_indices.len());
                chromosome[true_indices[idx]] = false;
            }
        }
    }
}

/// Cross-validate score for a given feature subset
fn cross_validate_score<E, Y>(
    estimator: &E,
    x: &Array2<Float>,
    y: &Y,
    cv_folds: usize,
) -> SklResult<f64>
where
    E: Clone + Fit<Array2<Float>, Y>,
    E::Fitted: Score<Array2<Float>, Y>,
    Y: Clone + IndexableTarget,
    <E::Fitted as Score<Array2<Float>, Y>>::Float: Into<f64>,
{
    let n_samples = x.nrows();
    let cv = KFold::new(cv_folds).shuffle(true);
    let mut scores = Vec::new();

    for (train_idx, test_idx) in cv.split(n_samples) {
        // Extract training and test sets
        let x_train = x.select(Axis(0), &train_idx);
        let y_train = y.select(&train_idx);
        let x_test = x.select(Axis(0), &test_idx);
        let y_test = y.select(&test_idx);

        // Fit and score
        let fitted = estimator.clone().fit(&x_train, &y_train)?;
        let score = fitted.score(&x_test, &y_test)?;
        scores.push(score);
    }

    // Return mean score
    let mut sum = 0.0;
    for score in &scores {
        sum += (*score).into();
    }
    Ok(sum / scores.len() as f64)
}

/// Multi-objective feature selector
#[derive(Debug, Clone)]
pub struct MultiObjectiveFeatureSelector<State = Untrained> {
    objectives: Vec<Box<dyn ObjectiveFunction>>,
    optimization_method: MultiObjectiveMethod,
    population_size: usize,
    n_generations: usize,
    crossover_prob: f64,
    mutation_prob: f64,
    random_state: Option<u64>,
    state: PhantomData<State>,
    // Trained state
    pareto_front_: Option<Vec<Individual>>,
    best_solutions_: Option<Vec<Vec<usize>>>, // Multiple Pareto-optimal solutions
    objective_values_: Option<Array2<f64>>,   // Objective values for each solution
    n_features_: Option<usize>,
}

/// Method for multi-objective optimization
#[derive(Debug, Clone)]
pub enum MultiObjectiveMethod {
    /// Non-dominated Sorting Genetic Algorithm II (NSGA-II)
    NSGAII,
    /// Multi-Objective Evolutionary Algorithm based on Decomposition (MOEA/D)
    MOEAD,
    /// Pareto Archived Evolution Strategy (PAES)
    PAES,
    /// Simple weighted sum approach
    WeightedSum(Vec<f64>), // weights for each objective
    /// Lexicographic ordering
    Lexicographic,
}

/// Individual in the population for evolutionary algorithms
#[derive(Debug, Clone)]
pub struct Individual {
    /// Binary feature selection (true = selected, false = not selected)
    pub features: Vec<bool>,
    /// Objective function values
    pub objectives: Vec<f64>,
    /// Dominance rank (for NSGA-II)
    pub rank: Option<usize>,
    /// Crowding distance (for NSGA-II)
    pub crowding_distance: Option<f64>,
}

/// Trait for objective functions in multi-objective optimization
pub trait ObjectiveFunction: Send + Sync + std::fmt::Debug {
    /// Evaluate the objective for a given feature subset
    /// Returns higher values for better solutions (maximization)
    fn evaluate(
        &self,
        x: &Array2<f64>,
        y_classif: Option<&Array1<i32>>,
        y_regression: Option<&Array1<f64>>,
        feature_mask: &[bool],
    ) -> SklResult<f64>;

    /// Get the name of this objective
    fn name(&self) -> &str;

    /// Whether this is a minimization objective (will be converted to maximization)
    fn is_minimization(&self) -> bool {
        false
    }

    fn clone_box(&self) -> Box<dyn ObjectiveFunction>;
}

impl Clone for Box<dyn ObjectiveFunction> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

/// Objective to minimize the number of selected features
#[derive(Debug, Clone)]
pub struct FeatureCountObjective;

impl ObjectiveFunction for FeatureCountObjective {
    fn evaluate(
        &self,
        _x: &Array2<f64>,
        _y_classif: Option<&Array1<i32>>,
        _y_regression: Option<&Array1<f64>>,
        feature_mask: &[bool],
    ) -> SklResult<f64> {
        let n_selected = feature_mask.iter().filter(|&&selected| selected).count();
        let n_total = feature_mask.len();
        // Return negative count for maximization (fewer features is better)
        Ok(-(n_selected as f64) / (n_total as f64))
    }

    fn name(&self) -> &str {
        "feature_count"
    }

    fn is_minimization(&self) -> bool {
        true
    }

    fn clone_box(&self) -> Box<dyn ObjectiveFunction> {
        Box::new(self.clone())
    }
}

/// Objective to maximize feature importance (using a given importance score)
#[derive(Debug, Clone)]
pub struct FeatureImportanceObjective {
    importance_scores: Array1<f64>,
}

impl FeatureImportanceObjective {
    pub fn new(importance_scores: Array1<f64>) -> Self {
        Self { importance_scores }
    }
}

impl ObjectiveFunction for FeatureImportanceObjective {
    fn evaluate(
        &self,
        _x: &Array2<f64>,
        _y_classif: Option<&Array1<i32>>,
        _y_regression: Option<&Array1<f64>>,
        feature_mask: &[bool],
    ) -> SklResult<f64> {
        let total_importance: f64 = feature_mask
            .iter()
            .enumerate()
            .filter_map(|(i, &selected)| {
                if selected {
                    Some(self.importance_scores[i])
                } else {
                    None
                }
            })
            .sum();

        let max_possible: f64 = self.importance_scores.iter().sum();

        Ok(total_importance / max_possible)
    }

    fn name(&self) -> &str {
        "feature_importance"
    }

    fn clone_box(&self) -> Box<dyn ObjectiveFunction> {
        Box::new(self.clone())
    }
}

/// Objective to maximize diversity (minimize correlation between selected features)
#[derive(Debug, Clone)]
pub struct FeatureDiversityObjective;

impl ObjectiveFunction for FeatureDiversityObjective {
    fn evaluate(
        &self,
        x: &Array2<f64>,
        _y_classif: Option<&Array1<i32>>,
        _y_regression: Option<&Array1<f64>>,
        feature_mask: &[bool],
    ) -> SklResult<f64> {
        let selected_indices: Vec<usize> = feature_mask
            .iter()
            .enumerate()
            .filter_map(|(i, &selected)| if selected { Some(i) } else { None })
            .collect();

        if selected_indices.len() < 2 {
            return Ok(1.0); // Maximum diversity for 0 or 1 features
        }

        let mut total_correlation = 0.0;
        let mut pair_count = 0;

        for i in 0..selected_indices.len() {
            for j in (i + 1)..selected_indices.len() {
                let feature_i = x.column(selected_indices[i]).to_owned();
                let feature_j = x.column(selected_indices[j]).to_owned();

                let correlation = compute_pearson_correlation(&feature_i, &feature_j).abs();
                total_correlation += correlation;
                pair_count += 1;
            }
        }

        let avg_correlation = if pair_count > 0 {
            total_correlation / pair_count as f64
        } else {
            0.0
        };

        // Return 1 - avg_correlation (higher diversity = lower correlation)
        Ok(1.0 - avg_correlation)
    }

    fn name(&self) -> &str {
        "feature_diversity"
    }

    fn clone_box(&self) -> Box<dyn ObjectiveFunction> {
        Box::new(self.clone())
    }
}

/// Objective to maximize predictive performance (simplified with correlation)
#[derive(Debug, Clone)]
pub struct PredictivePerformanceObjective;

impl ObjectiveFunction for PredictivePerformanceObjective {
    fn evaluate(
        &self,
        x: &Array2<f64>,
        y_classif: Option<&Array1<i32>>,
        y_regression: Option<&Array1<f64>>,
        feature_mask: &[bool],
    ) -> SklResult<f64> {
        let selected_indices: Vec<usize> = feature_mask
            .iter()
            .enumerate()
            .filter_map(|(i, &selected)| if selected { Some(i) } else { None })
            .collect();

        if selected_indices.is_empty() {
            return Ok(0.0);
        }

        let mut total_performance = 0.0;

        if let Some(y) = y_classif {
            // For classification, use absolute correlation with target
            for &idx in &selected_indices {
                let feature = x.column(idx).to_owned();
                let y_float = y.mapv(|v| v as f64);
                let correlation = compute_pearson_correlation(&feature, &y_float).abs();
                total_performance += correlation;
            }
        } else if let Some(y) = y_regression {
            // For regression, use absolute correlation with target
            for &idx in &selected_indices {
                let feature = x.column(idx).to_owned();
                let correlation = compute_pearson_correlation(&feature, y).abs();
                total_performance += correlation;
            }
        } else {
            return Err(SklearsError::InvalidInput(
                "Need either classification or regression target".to_string(),
            ));
        }

        Ok(total_performance / selected_indices.len() as f64)
    }

    fn name(&self) -> &str {
        "predictive_performance"
    }

    fn clone_box(&self) -> Box<dyn ObjectiveFunction> {
        Box::new(self.clone())
    }
}

/// Cost-sensitive objective that incorporates feature acquisition costs
/// Balances predictive performance with the cost of acquiring/using features
#[derive(Debug, Clone)]
pub struct CostSensitiveObjective {
    feature_costs: Array1<f64>,
    cost_weight: f64,
}

impl CostSensitiveObjective {
    /// Create a new cost-sensitive objective
    ///
    /// # Arguments
    /// * `feature_costs` - Cost associated with each feature (higher = more expensive)
    /// * `cost_weight` - Weight for cost vs performance trade-off (0.0 = ignore costs, 1.0 = only costs)
    pub fn new(feature_costs: Array1<f64>, cost_weight: f64) -> Self {
        Self {
            feature_costs,
            cost_weight: cost_weight.clamp(0.0, 1.0),
        }
    }
}

impl ObjectiveFunction for CostSensitiveObjective {
    fn evaluate(
        &self,
        x: &Array2<f64>,
        y_classif: Option<&Array1<i32>>,
        y_regression: Option<&Array1<f64>>,
        feature_mask: &[bool],
    ) -> SklResult<f64> {
        let selected_indices: Vec<usize> = feature_mask
            .iter()
            .enumerate()
            .filter_map(|(i, &selected)| if selected { Some(i) } else { None })
            .collect();

        if selected_indices.is_empty() {
            return Ok(0.0);
        }

        // Calculate total cost of selected features
        let total_cost: f64 = selected_indices
            .iter()
            .map(|&idx| self.feature_costs[idx])
            .sum();
        let max_cost: f64 = self.feature_costs.sum();
        let normalized_cost = if max_cost > 0.0 {
            total_cost / max_cost
        } else {
            0.0
        };

        // Calculate predictive performance (similar to PredictivePerformanceObjective)
        let mut total_performance = 0.0;
        if let Some(y) = y_classif {
            for &idx in &selected_indices {
                let feature = x.column(idx).to_owned();
                let y_float = y.mapv(|v| v as f64);
                let correlation = compute_pearson_correlation(&feature, &y_float).abs();
                total_performance += correlation;
            }
        } else if let Some(y) = y_regression {
            for &idx in &selected_indices {
                let feature = x.column(idx).to_owned();
                let correlation = compute_pearson_correlation(&feature, y).abs();
                total_performance += correlation;
            }
        } else {
            return Err(SklearsError::InvalidInput(
                "Need either classification or regression target".to_string(),
            ));
        }
        let avg_performance = total_performance / selected_indices.len() as f64;

        // Combine performance and cost (minimize cost, maximize performance)
        let cost_sensitive_score =
            (1.0 - self.cost_weight) * avg_performance + self.cost_weight * (1.0 - normalized_cost);

        Ok(cost_sensitive_score)
    }

    fn name(&self) -> &str {
        "cost_sensitive"
    }

    fn clone_box(&self) -> Box<dyn ObjectiveFunction> {
        Box::new(self.clone())
    }
}

/// Fairness-aware objective that promotes feature selection fairness across protected groups
/// Minimizes disparity in feature representation across different demographic groups
#[derive(Debug, Clone)]
pub struct FairnessAwareObjective {
    protected_groups: Array1<i32>, // Group membership for each sample
    fairness_weight: f64,
    fairness_metric: FairnessMetric,
}

#[derive(Debug, Clone)]
pub enum FairnessMetric {
    /// Demographic parity - equal representation across groups
    DemographicParity,
    /// Equal opportunity - equal true positive rates across groups
    EqualOpportunity,
    /// Equalized odds - equal true positive and false positive rates across groups
    EqualizedOdds,
    /// Statistical parity difference
    StatisticalParityDifference,
}

impl FairnessAwareObjective {
    /// Create a new fairness-aware objective
    ///
    /// # Arguments
    /// * `protected_groups` - Group membership for each sample (0, 1, 2, etc.)
    /// * `fairness_weight` - Weight for fairness vs performance trade-off
    /// * `fairness_metric` - Type of fairness metric to optimize for
    pub fn new(
        protected_groups: Array1<i32>,
        fairness_weight: f64,
        fairness_metric: FairnessMetric,
    ) -> Self {
        Self {
            protected_groups,
            fairness_weight: fairness_weight.clamp(0.0, 1.0),
            fairness_metric,
        }
    }
}

impl ObjectiveFunction for FairnessAwareObjective {
    fn evaluate(
        &self,
        x: &Array2<f64>,
        y_classif: Option<&Array1<i32>>,
        y_regression: Option<&Array1<f64>>,
        feature_mask: &[bool],
    ) -> SklResult<f64> {
        let selected_indices: Vec<usize> = feature_mask
            .iter()
            .enumerate()
            .filter_map(|(i, &selected)| if selected { Some(i) } else { None })
            .collect();

        if selected_indices.is_empty() {
            return Ok(1.0); // Maximum fairness for empty feature set
        }

        // Calculate predictive performance
        let mut total_performance = 0.0;
        if let Some(y) = y_classif {
            for &idx in &selected_indices {
                let feature = x.column(idx).to_owned();
                let y_float = y.mapv(|v| v as f64);
                let correlation = compute_pearson_correlation(&feature, &y_float).abs();
                total_performance += correlation;
            }
        } else if let Some(y) = y_regression {
            for &idx in &selected_indices {
                let feature = x.column(idx).to_owned();
                let correlation = compute_pearson_correlation(&feature, y).abs();
                total_performance += correlation;
            }
        } else {
            return Err(SklearsError::InvalidInput(
                "Need either classification or regression target".to_string(),
            ));
        }
        let avg_performance = total_performance / selected_indices.len() as f64;

        // Calculate fairness based on feature correlations with protected groups
        let mut fairness_score = 1.0; // Start with perfect fairness

        match self.fairness_metric {
            FairnessMetric::DemographicParity | FairnessMetric::StatisticalParityDifference => {
                // Measure how correlated selected features are with protected attributes
                let protected_float = self.protected_groups.mapv(|v| v as f64);
                let mut total_bias = 0.0;

                for &idx in &selected_indices {
                    let feature = x.column(idx).to_owned();
                    let correlation = compute_pearson_correlation(&feature, &protected_float).abs();
                    total_bias += correlation;
                }

                let avg_bias = total_bias / selected_indices.len() as f64;
                fairness_score = 1.0 - avg_bias; // Lower correlation with protected attributes = more fair
            }
            _ => {
                // For more complex fairness metrics, use simplified correlation-based approach
                let protected_float = self.protected_groups.mapv(|v| v as f64);
                let mut total_bias = 0.0;

                for &idx in &selected_indices {
                    let feature = x.column(idx).to_owned();
                    let correlation = compute_pearson_correlation(&feature, &protected_float).abs();
                    total_bias += correlation;
                }

                let avg_bias = total_bias / selected_indices.len() as f64;
                fairness_score = 1.0 - avg_bias;
            }
        }

        // Combine performance and fairness
        let fairness_aware_score =
            (1.0 - self.fairness_weight) * avg_performance + self.fairness_weight * fairness_score;

        Ok(fairness_aware_score)
    }

    fn name(&self) -> &str {
        "fairness_aware"
    }

    fn clone_box(&self) -> Box<dyn ObjectiveFunction> {
        Box::new(self.clone())
    }
}

impl Default for MultiObjectiveFeatureSelector<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl MultiObjectiveFeatureSelector<Untrained> {
    /// Create a new multi-objective feature selector
    pub fn new() -> Self {
        Self {
            objectives: Vec::new(),
            optimization_method: MultiObjectiveMethod::NSGAII,
            population_size: 100,
            n_generations: 50,
            crossover_prob: 0.8,
            mutation_prob: 0.1,
            random_state: None,
            state: PhantomData,
            pareto_front_: None,
            best_solutions_: None,
            objective_values_: None,
            n_features_: None,
        }
    }

    /// Add an objective function
    pub fn add_objective(mut self, objective: Box<dyn ObjectiveFunction>) -> Self {
        self.objectives.push(objective);
        self
    }

    /// Set the optimization method
    pub fn optimization_method(mut self, method: MultiObjectiveMethod) -> Self {
        self.optimization_method = method;
        self
    }

    /// Set population size for evolutionary algorithms
    pub fn population_size(mut self, size: usize) -> Self {
        self.population_size = size;
        self
    }

    /// Set number of generations for evolutionary algorithms
    pub fn n_generations(mut self, generations: usize) -> Self {
        self.n_generations = generations;
        self
    }

    /// Set crossover probability
    pub fn crossover_prob(mut self, prob: f64) -> Self {
        self.crossover_prob = prob;
        self
    }

    /// Set mutation probability
    pub fn mutation_prob(mut self, prob: f64) -> Self {
        self.mutation_prob = prob;
        self
    }

    /// Set random state for reproducibility
    pub fn random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Convenience method to add standard objectives
    pub fn add_standard_objectives(mut self, importance_scores: Option<Array1<f64>>) -> Self {
        self.objectives.push(Box::new(FeatureCountObjective));
        self.objectives
            .push(Box::new(PredictivePerformanceObjective));
        self.objectives.push(Box::new(FeatureDiversityObjective));

        if let Some(scores) = importance_scores {
            self.objectives
                .push(Box::new(FeatureImportanceObjective::new(scores)));
        }

        self
    }
}

impl Fit<Array2<f64>, Array1<i32>> for MultiObjectiveFeatureSelector<Untrained> {
    type Fitted = MultiObjectiveFeatureSelector<Trained>;

    fn fit(self, x: &Array2<f64>, y: &Array1<i32>) -> SklResult<Self::Fitted> {
        validate::check_consistent_length(x, y)?;

        let n_features = x.ncols();

        if self.objectives.is_empty() {
            return Err(SklearsError::InvalidInput(
                "No objectives added".to_string(),
            ));
        }

        let (pareto_front, best_solutions, objective_values) =
            self.optimize(x, Some(y), None, n_features)?;

        Ok(MultiObjectiveFeatureSelector {
            objectives: self.objectives,
            optimization_method: self.optimization_method,
            population_size: self.population_size,
            n_generations: self.n_generations,
            crossover_prob: self.crossover_prob,
            mutation_prob: self.mutation_prob,
            random_state: self.random_state,
            state: PhantomData,
            pareto_front_: Some(pareto_front),
            best_solutions_: Some(best_solutions),
            objective_values_: Some(objective_values),
            n_features_: Some(n_features),
        })
    }
}

impl Fit<Array2<f64>, Array1<f64>> for MultiObjectiveFeatureSelector<Untrained> {
    type Fitted = MultiObjectiveFeatureSelector<Trained>;

    fn fit(self, x: &Array2<f64>, y: &Array1<f64>) -> SklResult<Self::Fitted> {
        validate::check_consistent_length(x, y)?;

        let n_features = x.ncols();

        if self.objectives.is_empty() {
            return Err(SklearsError::InvalidInput(
                "No objectives added".to_string(),
            ));
        }

        let (pareto_front, best_solutions, objective_values) =
            self.optimize(x, None, Some(y), n_features)?;

        Ok(MultiObjectiveFeatureSelector {
            objectives: self.objectives,
            optimization_method: self.optimization_method,
            population_size: self.population_size,
            n_generations: self.n_generations,
            crossover_prob: self.crossover_prob,
            mutation_prob: self.mutation_prob,
            random_state: self.random_state,
            state: PhantomData,
            pareto_front_: Some(pareto_front),
            best_solutions_: Some(best_solutions),
            objective_values_: Some(objective_values),
            n_features_: Some(n_features),
        })
    }
}

impl MultiObjectiveFeatureSelector<Untrained> {
    /// Main optimization function
    fn optimize(
        &self,
        x: &Array2<f64>,
        y_classif: Option<&Array1<i32>>,
        y_regression: Option<&Array1<f64>>,
        n_features: usize,
    ) -> SklResult<(Vec<Individual>, Vec<Vec<usize>>, Array2<f64>)> {
        match &self.optimization_method {
            MultiObjectiveMethod::NSGAII => self.nsga_ii(x, y_classif, y_regression, n_features),
            MultiObjectiveMethod::WeightedSum(weights) => {
                self.weighted_sum(x, y_classif, y_regression, n_features, weights)
            }
            _ => {
                // For now, default to NSGA-II for other methods
                self.nsga_ii(x, y_classif, y_regression, n_features)
            }
        }
    }

    /// NSGA-II implementation (simplified)
    fn nsga_ii(
        &self,
        x: &Array2<f64>,
        y_classif: Option<&Array1<i32>>,
        y_regression: Option<&Array1<f64>>,
        n_features: usize,
    ) -> SklResult<(Vec<Individual>, Vec<Vec<usize>>, Array2<f64>)> {
        let mut rng = if let Some(seed) = self.random_state {
            StdRng::seed_from_u64(seed)
        } else {
            StdRng::seed_from_u64(42)
        };

        // Initialize population
        let mut population = self.initialize_population(n_features, &mut rng);

        // Evaluate initial population
        for individual in &mut population {
            individual.objectives =
                self.evaluate_objectives(x, y_classif, y_regression, &individual.features)?;
        }

        // Evolution loop
        for _generation in 0..self.n_generations {
            // Create offspring
            let mut offspring = self.create_offspring(&population, n_features, &mut rng)?;

            // Evaluate offspring
            for individual in &mut offspring {
                individual.objectives =
                    self.evaluate_objectives(x, y_classif, y_regression, &individual.features)?;
            }

            // Combine population and offspring
            let mut combined = population;
            combined.extend(offspring);

            // Non-dominated sorting and selection
            population = self.select_next_generation(combined, self.population_size);
        }

        // Extract Pareto front
        let pareto_front = self.extract_pareto_front(&population);
        let best_solutions: Vec<Vec<usize>> = pareto_front
            .iter()
            .map(|ind| {
                ind.features
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &selected)| if selected { Some(i) } else { None })
                    .collect()
            })
            .collect();

        // Create objective values matrix
        let n_objectives = self.objectives.len();
        let mut objective_values = Array2::<f64>::zeros((pareto_front.len(), n_objectives));
        for (i, individual) in pareto_front.iter().enumerate() {
            for (j, &value) in individual.objectives.iter().enumerate() {
                objective_values[[i, j]] = value;
            }
        }

        Ok((pareto_front, best_solutions, objective_values))
    }

    /// Weighted sum approach
    fn weighted_sum(
        &self,
        x: &Array2<f64>,
        y_classif: Option<&Array1<i32>>,
        y_regression: Option<&Array1<f64>>,
        n_features: usize,
        weights: &[f64],
    ) -> SklResult<(Vec<Individual>, Vec<Vec<usize>>, Array2<f64>)> {
        if weights.len() != self.objectives.len() {
            return Err(SklearsError::InvalidInput(
                "Number of weights must match number of objectives".to_string(),
            ));
        }

        let mut rng = if let Some(seed) = self.random_state {
            StdRng::seed_from_u64(seed)
        } else {
            StdRng::seed_from_u64(42)
        };

        // Initialize population
        let mut population = self.initialize_population(n_features, &mut rng);

        // Evaluate and select best individual based on weighted sum
        let mut best_individual = None;
        let mut best_score = f64::NEG_INFINITY;

        for individual in &mut population {
            individual.objectives =
                self.evaluate_objectives(x, y_classif, y_regression, &individual.features)?;

            let weighted_score: f64 = individual
                .objectives
                .iter()
                .zip(weights.iter())
                .map(|(&obj, &weight)| obj * weight)
                .sum();

            if weighted_score > best_score {
                best_score = weighted_score;
                best_individual = Some(individual.clone());
            }
        }

        let best = best_individual.unwrap();
        let best_features: Vec<usize> = best
            .features
            .iter()
            .enumerate()
            .filter_map(|(i, &selected)| if selected { Some(i) } else { None })
            .collect();

        let mut objective_values = Array2::<f64>::zeros((1, self.objectives.len()));
        for (j, &value) in best.objectives.iter().enumerate() {
            objective_values[[0, j]] = value;
        }

        Ok((vec![best], vec![best_features], objective_values))
    }

    /// Initialize random population
    fn initialize_population(&self, n_features: usize, rng: &mut StdRng) -> Vec<Individual> {
        let mut population = Vec::with_capacity(self.population_size);

        for _ in 0..self.population_size {
            let mut features = vec![false; n_features];
            // Randomly select some features (10-50% of total)
            let n_selected = rng.gen_range((n_features / 10).max(1)..=(n_features / 2).max(1));

            let mut indices: Vec<usize> = (0..n_features).collect();
            indices.shuffle(rng);

            for &idx in indices.iter().take(n_selected) {
                features[idx] = true;
            }

            population.push(Individual {
                features,
                objectives: Vec::new(),
                rank: None,
                crowding_distance: None,
            });
        }

        population
    }

    /// Evaluate all objectives for a feature selection
    fn evaluate_objectives(
        &self,
        x: &Array2<f64>,
        y_classif: Option<&Array1<i32>>,
        y_regression: Option<&Array1<f64>>,
        feature_mask: &[bool],
    ) -> SklResult<Vec<f64>> {
        let mut objectives = Vec::with_capacity(self.objectives.len());

        for objective in &self.objectives {
            let mut value = objective.evaluate(x, y_classif, y_regression, feature_mask)?;

            // Convert minimization objectives to maximization
            if objective.is_minimization() {
                value = -value;
            }

            objectives.push(value);
        }

        Ok(objectives)
    }

    /// Create offspring through crossover and mutation
    fn create_offspring(
        &self,
        population: &[Individual],
        n_features: usize,
        rng: &mut StdRng,
    ) -> SklResult<Vec<Individual>> {
        let mut offspring = Vec::with_capacity(self.population_size);

        for _ in 0..self.population_size {
            // Tournament selection for parents
            let parent1 = self.tournament_selection(population, rng);
            let parent2 = self.tournament_selection(population, rng);

            // Crossover
            let mut child = if rng.gen::<f64>() < self.crossover_prob {
                self.uniform_crossover(&parent1.features, &parent2.features, rng)
            } else {
                parent1.features.clone()
            };

            // Mutation
            if rng.gen::<f64>() < self.mutation_prob {
                self.bit_flip_mutation(&mut child, rng);
            }

            offspring.push(Individual {
                features: child,
                objectives: Vec::new(),
                rank: None,
                crowding_distance: None,
            });
        }

        Ok(offspring)
    }

    /// Tournament selection
    fn tournament_selection<'a>(
        &self,
        population: &'a [Individual],
        rng: &mut StdRng,
    ) -> &'a Individual {
        let tournament_size = 3;
        let mut best = &population[rng.gen_range(0..population.len())];

        for _ in 1..tournament_size {
            let candidate = &population[rng.gen_range(0..population.len())];
            if self.dominates(&candidate.objectives, &best.objectives) {
                best = candidate;
            }
        }

        best
    }

    /// Uniform crossover
    fn uniform_crossover(&self, parent1: &[bool], parent2: &[bool], rng: &mut StdRng) -> Vec<bool> {
        parent1
            .iter()
            .zip(parent2.iter())
            .map(|(&p1, &p2)| if rng.gen::<f64>() < 0.5 { p1 } else { p2 })
            .collect()
    }

    /// Bit flip mutation
    fn bit_flip_mutation(&self, individual: &mut [bool], rng: &mut StdRng) {
        for gene in individual.iter_mut() {
            if rng.gen::<f64>() < 0.1 {
                // 10% mutation rate per bit
                *gene = !*gene;
            }
        }
    }

    /// Check if objectives1 dominates objectives2
    fn dominates(&self, objectives1: &[f64], objectives2: &[f64]) -> bool {
        let at_least_one_better = objectives1
            .iter()
            .zip(objectives2.iter())
            .any(|(&a, &b)| a > b);

        let all_as_good = objectives1
            .iter()
            .zip(objectives2.iter())
            .all(|(&a, &b)| a >= b);

        at_least_one_better && all_as_good
    }

    /// Select next generation using NSGA-II selection
    fn select_next_generation(
        &self,
        mut population: Vec<Individual>,
        target_size: usize,
    ) -> Vec<Individual> {
        // Non-dominated sorting (simplified)
        let fronts = self.non_dominated_sort(&mut population);

        let mut next_generation = Vec::new();

        for front in fronts {
            if next_generation.len() + front.len() <= target_size {
                next_generation.extend(front);
            } else {
                // Add remaining individuals based on crowding distance
                let remaining = target_size - next_generation.len();
                let mut front = front;
                front.sort_by(|a, b| {
                    b.crowding_distance
                        .unwrap_or(0.0)
                        .partial_cmp(&a.crowding_distance.unwrap_or(0.0))
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                next_generation.extend(front.into_iter().take(remaining));
                break;
            }
        }

        next_generation
    }

    /// Non-dominated sorting (simplified)
    fn non_dominated_sort(&self, population: &mut [Individual]) -> Vec<Vec<Individual>> {
        let mut fronts = Vec::new();
        let mut current_front: Vec<Individual> = Vec::new();
        let mut remaining: Vec<Individual> = population.to_vec();

        while !remaining.is_empty() {
            current_front.clear();
            let mut next_remaining = Vec::new();

            for individual in remaining {
                let mut is_dominated = false;
                for other in &current_front {
                    if self.dominates(&other.objectives, &individual.objectives) {
                        is_dominated = true;
                        break;
                    }
                }

                if !is_dominated {
                    // Remove any dominated individuals from current front
                    current_front
                        .retain(|other| !self.dominates(&individual.objectives, &other.objectives));
                    current_front.push(individual);
                } else {
                    next_remaining.push(individual);
                }
            }

            fronts.push(current_front.clone());
            remaining = next_remaining;
        }

        fronts
    }

    /// Extract Pareto front (first front from non-dominated sorting)
    fn extract_pareto_front(&self, population: &[Individual]) -> Vec<Individual> {
        let mut pareto_front = Vec::new();

        for individual in population {
            let mut is_dominated = false;
            for other in population {
                if self.dominates(&other.objectives, &individual.objectives) {
                    is_dominated = true;
                    break;
                }
            }

            if !is_dominated {
                pareto_front.push(individual.clone());
            }
        }

        pareto_front
    }
}

impl MultiObjectiveFeatureSelector<Trained> {
    /// Get the Pareto front solutions
    pub fn pareto_front(&self) -> &[Individual] {
        self.pareto_front_.as_ref().unwrap()
    }

    /// Get all Pareto-optimal feature selections
    pub fn pareto_optimal_features(&self) -> &[Vec<usize>] {
        self.best_solutions_.as_ref().unwrap()
    }

    /// Get objective values for all Pareto-optimal solutions
    pub fn objective_values(&self) -> &Array2<f64> {
        self.objective_values_.as_ref().unwrap()
    }

    /// Select a specific solution from the Pareto front by index
    pub fn select_solution(&self, index: usize) -> Option<&[usize]> {
        self.best_solutions_
            .as_ref()
            .unwrap()
            .get(index)
            .map(|v| v.as_slice())
    }

    /// Select solution based on preference weights for objectives
    pub fn select_by_preference(&self, weights: &[f64]) -> SklResult<&[usize]> {
        if weights.len() != self.objectives.len() {
            return Err(SklearsError::InvalidInput(
                "Number of weights must match number of objectives".to_string(),
            ));
        }

        let objective_values = self.objective_values_.as_ref().unwrap();
        let mut best_score = f64::NEG_INFINITY;
        let mut best_index = 0;

        for i in 0..objective_values.nrows() {
            let score: f64 = (0..objective_values.ncols())
                .map(|j| objective_values[[i, j]] * weights[j])
                .sum();

            if score > best_score {
                best_score = score;
                best_index = i;
            }
        }

        Ok(self.best_solutions_.as_ref().unwrap()[best_index].as_slice())
    }

    /// Get a balanced solution (equal weights for all objectives)
    pub fn balanced_solution(&self) -> SklResult<&[usize]> {
        let n_objectives = self.objectives.len();
        let weights = vec![1.0 / n_objectives as f64; n_objectives];
        self.select_by_preference(&weights)
    }

    /// Select solution with minimum total cost while maintaining performance threshold
    pub fn select_cost_optimal(&self, performance_threshold: f64) -> SklResult<&[usize]> {
        let objective_values = self.objective_values_.as_ref().unwrap();
        let solutions = self.best_solutions_.as_ref().unwrap();

        // Find cost-sensitive and performance objectives
        let mut cost_idx = None;
        let mut perf_idx = None;

        for (i, obj) in self.objectives.iter().enumerate() {
            if obj.name() == "cost_sensitive" {
                cost_idx = Some(i);
            } else if obj.name() == "predictive_performance" {
                perf_idx = Some(i);
            }
        }

        let mut best_cost = f64::INFINITY;
        let mut best_solution_idx = 0;

        for i in 0..objective_values.nrows() {
            // Check if performance meets threshold
            let performance = if let Some(perf_idx) = perf_idx {
                objective_values[[i, perf_idx]]
            } else {
                // Use average of all objectives as performance proxy
                (0..objective_values.ncols())
                    .map(|j| objective_values[[i, j]])
                    .sum::<f64>()
                    / objective_values.ncols() as f64
            };

            if performance >= performance_threshold {
                let cost = if let Some(cost_idx) = cost_idx {
                    1.0 - objective_values[[i, cost_idx]] // Convert back to cost (higher is worse)
                } else {
                    // Use feature count as cost proxy
                    solutions[i].len() as f64
                };

                if cost < best_cost {
                    best_cost = cost;
                    best_solution_idx = i;
                }
            }
        }

        if best_cost == f64::INFINITY {
            return Err(SklearsError::InvalidInput(
                "No solution meets the performance threshold".to_string(),
            ));
        }

        Ok(&solutions[best_solution_idx])
    }

    /// Select solution with best cost-performance ratio
    pub fn select_cost_efficient(&self) -> SklResult<&[usize]> {
        let objective_values = self.objective_values_.as_ref().unwrap();
        let solutions = self.best_solutions_.as_ref().unwrap();

        // Find cost-sensitive and performance objectives
        let mut cost_idx = None;
        let mut perf_idx = None;

        for (i, obj) in self.objectives.iter().enumerate() {
            if obj.name() == "cost_sensitive" {
                cost_idx = Some(i);
            } else if obj.name() == "predictive_performance" {
                perf_idx = Some(i);
            }
        }

        let mut best_ratio = f64::NEG_INFINITY;
        let mut best_solution_idx = 0;

        for i in 0..objective_values.nrows() {
            let performance = if let Some(perf_idx) = perf_idx {
                objective_values[[i, perf_idx]]
            } else {
                // Use average of all objectives as performance proxy
                (0..objective_values.ncols())
                    .map(|j| objective_values[[i, j]])
                    .sum::<f64>()
                    / objective_values.ncols() as f64
            };

            let cost = if let Some(cost_idx) = cost_idx {
                1.0 - objective_values[[i, cost_idx]] // Convert back to cost (higher is worse)
            } else {
                // Use feature count as cost proxy
                solutions[i].len() as f64
            };

            let ratio = if cost > 0.0 {
                performance / cost
            } else {
                f64::INFINITY
            };

            if ratio > best_ratio {
                best_ratio = ratio;
                best_solution_idx = i;
            }
        }

        Ok(&solutions[best_solution_idx])
    }

    /// Get cost breakdown for each Pareto-optimal solution
    pub fn cost_breakdown(&self) -> SklResult<Vec<f64>> {
        let solutions = self.best_solutions_.as_ref().unwrap();
        let mut costs = Vec::new();

        // Try to find cost-sensitive objective
        let mut cost_objective = None;
        for obj in &self.objectives {
            if obj.name() == "cost_sensitive" {
                cost_objective = Some(obj);
                break;
            }
        }

        if let Some(cost_obj) = cost_objective {
            // We can't directly access the feature costs from the objective,
            // so we'll use feature count as a proxy
            for solution in solutions {
                costs.push(solution.len() as f64);
            }
        } else {
            // Use feature count as cost proxy
            for solution in solutions {
                costs.push(solution.len() as f64);
            }
        }

        Ok(costs)
    }

    /// Select solution that maximizes fairness metrics
    pub fn select_fairness_optimal(&self) -> SklResult<&[usize]> {
        let objective_values = self.objective_values_.as_ref().unwrap();
        let solutions = self.best_solutions_.as_ref().unwrap();

        // Find fairness-aware objective
        let mut fairness_idx = None;
        for (i, obj) in self.objectives.iter().enumerate() {
            if obj.name() == "fairness_aware" {
                fairness_idx = Some(i);
                break;
            }
        }

        let mut best_fairness = f64::NEG_INFINITY;
        let mut best_solution_idx = 0;

        for i in 0..objective_values.nrows() {
            let fairness = if let Some(fairness_idx) = fairness_idx {
                objective_values[[i, fairness_idx]]
            } else {
                // Use average of all objectives as fairness proxy
                (0..objective_values.ncols())
                    .map(|j| objective_values[[i, j]])
                    .sum::<f64>()
                    / objective_values.ncols() as f64
            };

            if fairness > best_fairness {
                best_fairness = fairness;
                best_solution_idx = i;
            }
        }

        Ok(&solutions[best_solution_idx])
    }

    /// Select solution that balances performance and fairness
    pub fn select_fair_balanced(&self, fairness_threshold: f64) -> SklResult<&[usize]> {
        let objective_values = self.objective_values_.as_ref().unwrap();
        let solutions = self.best_solutions_.as_ref().unwrap();

        // Find fairness and performance objectives
        let mut fairness_idx = None;
        let mut perf_idx = None;

        for (i, obj) in self.objectives.iter().enumerate() {
            if obj.name() == "fairness_aware" {
                fairness_idx = Some(i);
            } else if obj.name() == "predictive_performance" {
                perf_idx = Some(i);
            }
        }

        let mut best_score = f64::NEG_INFINITY;
        let mut best_solution_idx = 0;

        for i in 0..objective_values.nrows() {
            let fairness = if let Some(fairness_idx) = fairness_idx {
                objective_values[[i, fairness_idx]]
            } else {
                // Use average of all objectives as fairness proxy
                (0..objective_values.ncols())
                    .map(|j| objective_values[[i, j]])
                    .sum::<f64>()
                    / objective_values.ncols() as f64
            };

            // Only consider solutions that meet fairness threshold
            if fairness >= fairness_threshold {
                let performance = if let Some(perf_idx) = perf_idx {
                    objective_values[[i, perf_idx]]
                } else {
                    // Use average of all objectives as performance proxy
                    (0..objective_values.ncols())
                        .map(|j| objective_values[[i, j]])
                        .sum::<f64>()
                        / objective_values.ncols() as f64
                };

                let balanced_score = 0.5 * performance + 0.5 * fairness;

                if balanced_score > best_score {
                    best_score = balanced_score;
                    best_solution_idx = i;
                }
            }
        }

        if best_score == f64::NEG_INFINITY {
            return Err(SklearsError::InvalidInput(
                "No solution meets the fairness threshold".to_string(),
            ));
        }

        Ok(&solutions[best_solution_idx])
    }

    /// Get fairness metrics for each Pareto-optimal solution
    pub fn fairness_metrics(&self) -> SklResult<Vec<f64>> {
        let objective_values = self.objective_values_.as_ref().unwrap();
        let mut fairness_values = Vec::new();

        // Find fairness-aware objective
        let mut fairness_idx = None;
        for (i, obj) in self.objectives.iter().enumerate() {
            if obj.name() == "fairness_aware" {
                fairness_idx = Some(i);
                break;
            }
        }

        if let Some(fairness_idx) = fairness_idx {
            for i in 0..objective_values.nrows() {
                fairness_values.push(objective_values[[i, fairness_idx]]);
            }
        } else {
            // Use average of all objectives as fairness proxy
            for i in 0..objective_values.nrows() {
                let avg = (0..objective_values.ncols())
                    .map(|j| objective_values[[i, j]])
                    .sum::<f64>()
                    / objective_values.ncols() as f64;
                fairness_values.push(avg);
            }
        }

        Ok(fairness_values)
    }

    /// Get hypervolume indicator for solution quality assessment
    pub fn hypervolume(&self, reference_point: &[f64]) -> SklResult<f64> {
        if reference_point.len() != self.objectives.len() {
            return Err(SklearsError::InvalidInput(
                "Reference point must have same dimensions as objectives".to_string(),
            ));
        }

        let objective_values = self.objective_values_.as_ref().unwrap();
        let mut hypervolume = 0.0;

        // Simple hypervolume calculation for 2D case
        if self.objectives.len() == 2 {
            let mut points: Vec<(f64, f64)> = Vec::new();
            for i in 0..objective_values.nrows() {
                let x = objective_values[[i, 0]];
                let y = objective_values[[i, 1]];
                points.push((x, y));
            }

            // Sort by first objective
            points.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

            let mut prev_y = reference_point[1];
            for (x, y) in points {
                if x > reference_point[0] && y > reference_point[1] {
                    hypervolume += (x - reference_point[0]) * (y - prev_y);
                    prev_y = y;
                }
            }
        } else {
            // For higher dimensions, use approximation
            for i in 0..objective_values.nrows() {
                let mut volume = 1.0;
                for j in 0..objective_values.ncols() {
                    let diff = objective_values[[i, j]] - reference_point[j];
                    if diff > 0.0 {
                        volume *= diff;
                    } else {
                        volume = 0.0;
                        break;
                    }
                }
                hypervolume += volume;
            }
        }

        Ok(hypervolume)
    }

    /// Get diversity metrics of the Pareto front
    pub fn pareto_diversity(&self) -> SklResult<f64> {
        let objective_values = self.objective_values_.as_ref().unwrap();
        let n_solutions = objective_values.nrows();

        if n_solutions < 2 {
            return Ok(0.0);
        }

        let mut total_distance = 0.0;
        let mut count = 0;

        // Calculate average pairwise distance in objective space
        for i in 0..n_solutions {
            for j in (i + 1)..n_solutions {
                let mut distance = 0.0;
                for k in 0..objective_values.ncols() {
                    let diff = objective_values[[i, k]] - objective_values[[j, k]];
                    distance += diff * diff;
                }
                total_distance += distance.sqrt();
                count += 1;
            }
        }

        Ok(total_distance / count as f64)
    }

    /// Find knee point solution (best trade-off)
    pub fn knee_point_solution(&self) -> SklResult<&[usize]> {
        let objective_values = self.objective_values_.as_ref().unwrap();
        let solutions = self.best_solutions_.as_ref().unwrap();

        if self.objectives.len() != 2 {
            return Err(SklearsError::InvalidInput(
                "Knee point detection only supported for 2-objective problems".to_string(),
            ));
        }

        let mut max_distance = f64::NEG_INFINITY;
        let mut knee_idx = 0;

        // Find the solution with maximum distance from the line connecting extreme points
        let mut min_obj1 = f64::INFINITY;
        let mut max_obj1 = f64::NEG_INFINITY;
        let mut min_obj2 = f64::INFINITY;
        let mut max_obj2 = f64::NEG_INFINITY;

        for i in 0..objective_values.nrows() {
            let obj1 = objective_values[[i, 0]];
            let obj2 = objective_values[[i, 1]];

            min_obj1 = min_obj1.min(obj1);
            max_obj1 = max_obj1.max(obj1);
            min_obj2 = min_obj2.min(obj2);
            max_obj2 = max_obj2.max(obj2);
        }

        // Calculate distance from line for each point
        for i in 0..objective_values.nrows() {
            let obj1 = objective_values[[i, 0]];
            let obj2 = objective_values[[i, 1]];

            // Normalize objectives
            let norm_obj1 = (obj1 - min_obj1) / (max_obj1 - min_obj1);
            let norm_obj2 = (obj2 - min_obj2) / (max_obj2 - min_obj2);

            // Distance from line connecting (0,1) to (1,0)
            let distance = (norm_obj1 + norm_obj2 - 1.0).abs() / 2.0_f64.sqrt();

            if distance > max_distance {
                max_distance = distance;
                knee_idx = i;
            }
        }

        Ok(&solutions[knee_idx])
    }

    /// Get dominated solutions count for each objective
    pub fn domination_analysis(&self) -> SklResult<Vec<usize>> {
        let objective_values = self.objective_values_.as_ref().unwrap();
        let n_solutions = objective_values.nrows();
        let mut domination_counts = vec![0; n_solutions];

        // Count how many solutions each solution dominates
        for i in 0..n_solutions {
            for j in 0..n_solutions {
                if i != j {
                    let mut dominates = true;
                    let mut strictly_better = false;

                    for k in 0..objective_values.ncols() {
                        let val_i = objective_values[[i, k]];
                        let val_j = objective_values[[j, k]];

                        if val_i < val_j {
                            dominates = false;
                            break;
                        } else if val_i > val_j {
                            strictly_better = true;
                        }
                    }

                    if dominates && strictly_better {
                        domination_counts[i] += 1;
                    }
                }
            }
        }

        Ok(domination_counts)
    }

    /// Select solution closest to ideal point in objective space
    pub fn select_closest_to_ideal(&self, ideal_point: &[f64]) -> SklResult<&[usize]> {
        if ideal_point.len() != self.objectives.len() {
            return Err(SklearsError::InvalidInput(
                "Ideal point must have same dimensions as objectives".to_string(),
            ));
        }

        let objective_values = self.objective_values_.as_ref().unwrap();
        let solutions = self.best_solutions_.as_ref().unwrap();

        let mut min_distance = f64::INFINITY;
        let mut best_solution_idx = 0;

        for i in 0..objective_values.nrows() {
            let mut distance = 0.0;
            for j in 0..objective_values.ncols() {
                let diff = objective_values[[i, j]] - ideal_point[j];
                distance += diff * diff;
            }
            distance = distance.sqrt();

            if distance < min_distance {
                min_distance = distance;
                best_solution_idx = i;
            }
        }

        Ok(&solutions[best_solution_idx])
    }

    /// Select solution with minimum distance to reference point
    pub fn select_reference_point(&self, reference: &[f64]) -> SklResult<&[usize]> {
        self.select_closest_to_ideal(reference)
    }

    /// Get solutions within specific objective ranges
    pub fn filter_by_objectives(
        &self,
        min_values: &[f64],
        max_values: &[f64],
    ) -> SklResult<Vec<usize>> {
        if min_values.len() != self.objectives.len() || max_values.len() != self.objectives.len() {
            return Err(SklearsError::InvalidInput(
                "Min and max values must have same dimensions as objectives".to_string(),
            ));
        }

        let objective_values = self.objective_values_.as_ref().unwrap();
        let mut filtered_indices = Vec::new();

        for i in 0..objective_values.nrows() {
            let mut within_range = true;

            for j in 0..objective_values.ncols() {
                let value = objective_values[[i, j]];
                if value < min_values[j] || value > max_values[j] {
                    within_range = false;
                    break;
                }
            }

            if within_range {
                filtered_indices.push(i);
            }
        }

        Ok(filtered_indices)
    }
}

impl Transform<Array2<f64>> for MultiObjectiveFeatureSelector<Trained> {
    fn transform(&self, x: &Array2<f64>) -> SklResult<Array2<f64>> {
        let n_features = self.n_features_.unwrap();
        if x.ncols() != n_features {
            return Err(SklearsError::InvalidInput(format!(
                "Expected {} features, got {}",
                n_features,
                x.ncols()
            )));
        }

        let selected = self.balanced_solution().unwrap_or(&[]);
        let n_samples = x.nrows();
        let n_selected = selected.len();

        let mut x_new = Array2::<f64>::zeros((n_samples, n_selected));
        for (new_idx, &old_idx) in selected.iter().enumerate() {
            x_new.column_mut(new_idx).assign(&x.column(old_idx));
        }

        Ok(x_new)
    }
}

impl SelectorMixin for MultiObjectiveFeatureSelector<Trained> {
    fn get_support(&self) -> SklResult<Array1<bool>> {
        // Default: return balanced solution
        let selected_features = self.balanced_solution().unwrap_or(&[]);
        let n_features = self.n_features_.unwrap();
        let mut support = vec![false; n_features];

        for &idx in selected_features {
            support[idx] = true;
        }

        Ok(Array1::from(support))
    }

    fn transform_features(&self, indices: &[usize]) -> SklResult<Vec<usize>> {
        let selected = self.balanced_solution().unwrap_or(&[]);
        Ok(indices
            .iter()
            .filter_map(|&idx| selected.iter().position(|&f| f == idx))
            .collect())
    }
}

/// Helper function to compute Pearson correlation between two arrays
fn compute_pearson_correlation(x: &Array1<f64>, y: &Array1<f64>) -> f64 {
    let n = x.len() as f64;
    if n < 2.0 {
        return 0.0;
    }

    let mean_x = x.mean().unwrap_or(0.0);
    let mean_y = y.mean().unwrap_or(0.0);

    let mut numerator = 0.0;
    let mut sum_sq_x = 0.0;
    let mut sum_sq_y = 0.0;

    for (&xi, &yi) in x.iter().zip(y.iter()) {
        let dx = xi - mean_x;
        let dy = yi - mean_y;
        numerator += dx * dy;
        sum_sq_x += dx * dx;
        sum_sq_y += dy * dy;
    }

    let denominator = (sum_sq_x * sum_sq_y).sqrt();
    if denominator == 0.0 {
        0.0
    } else {
        numerator / denominator
    }
}
