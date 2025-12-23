//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::functions::*;
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::rand_prelude::SliceRandom;
use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::Rng;
use scirs2_core::random::SeedableRng;
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Trained, Untrained},
};
use std::marker::PhantomData;

/// Simple cross-validator trait replacement
pub trait CrossValidator {
    fn split(&self, n_samples: usize) -> Vec<(Vec<usize>, Vec<usize>)>;
}

/// Genetic Algorithm Feature Selection
/// Uses evolutionary algorithms to find optimal feature subsets
#[derive(Debug, Clone)]
pub struct GeneticSelector<E, State = Untrained> {
    pub(crate) estimator: E,
    pub(crate) population_size: usize,
    pub(crate) generations: usize,
    pub(crate) crossover_rate: f64,
    pub(crate) mutation_rate: f64,
    pub(crate) tournament_size: usize,
    pub(crate) min_features: usize,
    pub(crate) max_features: Option<usize>,
    pub(crate) cv_folds: usize,
    pub(crate) random_state: Option<u64>,
    pub(crate) state: PhantomData<State>,
    pub(crate) best_features_: Option<Vec<usize>>,
    pub(crate) best_score_: Option<f64>,
    pub(crate) n_features_: Option<usize>,
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
/// Objective to maximize feature importance (using a given importance score)
#[derive(Debug, Clone)]
pub struct FeatureImportanceObjective {
    pub(crate) importance_scores: Array1<f64>,
}
impl FeatureImportanceObjective {
    pub fn new(importance_scores: Array1<f64>) -> Self {
        Self { importance_scores }
    }
}
/// Cost-sensitive objective that incorporates feature acquisition costs
/// Balances predictive performance with the cost of acquiring/using features
#[derive(Debug, Clone)]
pub struct CostSensitiveObjective {
    pub(crate) feature_costs: Array1<f64>,
    pub(crate) cost_weight: f64,
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
/// Objective to maximize diversity (minimize correlation between selected features)
#[derive(Debug, Clone)]
pub struct FeatureDiversityObjective;
/// Multi-objective feature selector
#[derive(Debug, Clone)]
pub struct MultiObjectiveFeatureSelector<State = Untrained> {
    pub(crate) objectives: Vec<Box<dyn ObjectiveFunction>>,
    pub(crate) optimization_method: MultiObjectiveMethod,
    pub(crate) population_size: usize,
    pub(crate) n_generations: usize,
    pub(crate) crossover_prob: f64,
    pub(crate) mutation_prob: f64,
    pub(crate) random_state: Option<u64>,
    pub(crate) state: PhantomData<State>,
    pub(crate) pareto_front_: Option<Vec<Individual>>,
    pub(crate) best_solutions_: Option<Vec<Vec<usize>>>,
    pub(crate) objective_values_: Option<Array2<f64>>,
    pub(crate) n_features_: Option<usize>,
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
impl MultiObjectiveFeatureSelector<Untrained> {
    /// Main optimization function
    pub(crate) fn optimize(
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
            _ => self.nsga_ii(x, y_classif, y_regression, n_features),
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
        let mut population = self.initialize_population(n_features, &mut rng);
        for individual in &mut population {
            individual.objectives =
                self.evaluate_objectives(x, y_classif, y_regression, &individual.features)?;
        }
        for _generation in 0..self.n_generations {
            let mut offspring = self.create_offspring(&population, n_features, &mut rng)?;
            for individual in &mut offspring {
                individual.objectives =
                    self.evaluate_objectives(x, y_classif, y_regression, &individual.features)?;
            }
            let mut combined = population;
            combined.extend(offspring);
            population = self.select_next_generation(combined, self.population_size);
        }
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
        let mut population = self.initialize_population(n_features, &mut rng);
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
        _n_features: usize,
        rng: &mut StdRng,
    ) -> SklResult<Vec<Individual>> {
        let mut offspring = Vec::with_capacity(self.population_size);
        for _ in 0..self.population_size {
            let parent1 = self.tournament_selection(population, rng);
            let parent2 = self.tournament_selection(population, rng);
            let mut child = if rng.gen::<f64>() < self.crossover_prob {
                self.uniform_crossover(&parent1.features, &parent2.features, rng)
            } else {
                parent1.features.clone()
            };
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
        let fronts = self.non_dominated_sort(&mut population);
        let mut next_generation = Vec::new();
        for front in fronts {
            if next_generation.len() + front.len() <= target_size {
                next_generation.extend(front);
            } else {
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
            let performance = if let Some(perf_idx) = perf_idx {
                objective_values[[i, perf_idx]]
            } else {
                (0..objective_values.ncols())
                    .map(|j| objective_values[[i, j]])
                    .sum::<f64>()
                    / objective_values.ncols() as f64
            };
            if performance >= performance_threshold {
                let cost = if let Some(cost_idx) = cost_idx {
                    1.0 - objective_values[[i, cost_idx]]
                } else {
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
                (0..objective_values.ncols())
                    .map(|j| objective_values[[i, j]])
                    .sum::<f64>()
                    / objective_values.ncols() as f64
            };
            let cost = if let Some(cost_idx) = cost_idx {
                1.0 - objective_values[[i, cost_idx]]
            } else {
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
        let mut cost_objective = None;
        for obj in &self.objectives {
            if obj.name() == "cost_sensitive" {
                cost_objective = Some(obj);
                break;
            }
        }
        if let Some(_cost_obj) = cost_objective {
            for solution in solutions {
                costs.push(solution.len() as f64);
            }
        } else {
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
                (0..objective_values.ncols())
                    .map(|j| objective_values[[i, j]])
                    .sum::<f64>()
                    / objective_values.ncols() as f64
            };
            if fairness >= fairness_threshold {
                let performance = if let Some(perf_idx) = perf_idx {
                    objective_values[[i, perf_idx]]
                } else {
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
        if self.objectives.len() == 2 {
            let mut points: Vec<(f64, f64)> = Vec::new();
            for i in 0..objective_values.nrows() {
                let x = objective_values[[i, 0]];
                let y = objective_values[[i, 1]];
                points.push((x, y));
            }
            points.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            let mut prev_y = reference_point[1];
            for (x, y) in points {
                if x > reference_point[0] && y > reference_point[1] {
                    hypervolume += (x - reference_point[0]) * (y - prev_y);
                    prev_y = y;
                }
            }
        } else {
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
        for i in 0..objective_values.nrows() {
            let obj1 = objective_values[[i, 0]];
            let obj2 = objective_values[[i, 1]];
            let norm_obj1 = (obj1 - min_obj1) / (max_obj1 - min_obj1);
            let norm_obj2 = (obj2 - min_obj2) / (max_obj2 - min_obj2);
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
    WeightedSum(Vec<f64>),
    /// Lexicographic ordering
    Lexicographic,
}
/// Objective to maximize predictive performance (simplified with correlation)
#[derive(Debug, Clone)]
pub struct PredictivePerformanceObjective;
/// Simple KFold implementation replacement
#[derive(Debug, Clone)]
pub struct KFold {
    pub(crate) n_splits: usize,
    pub(crate) shuffle: bool,
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
/// Fairness-aware objective that promotes feature selection fairness across protected groups
/// Minimizes disparity in feature representation across different demographic groups
#[derive(Debug, Clone)]
pub struct FairnessAwareObjective {
    pub(crate) protected_groups: Array1<i32>,
    pub(crate) fairness_weight: f64,
    pub(crate) fairness_metric: FairnessMetric,
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
/// Objective to minimize the number of selected features
#[derive(Debug, Clone)]
pub struct FeatureCountObjective;
