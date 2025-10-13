//! Evolutionary algorithms for hyperparameter optimization
//!
//! This module implements genetic algorithms and other evolutionary approaches for
//! hyperparameter optimization. These methods are particularly useful for:
//! - Non-differentiable objective functions
//! - Mixed parameter types (categorical, continuous, integer)
//! - Multi-modal optimization landscapes
//! - When gradient information is not available

use crate::{CrossValidator, ParameterValue};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::rand_prelude::IndexedRandom;
use scirs2_core::random::prelude::*;
use scirs2_core::random::{rngs::StdRng, SeedableRng};
use sklears_core::{
    error::Result,
    traits::{Fit, Predict, Score},
    types::Float,
};

/// Individual in the genetic algorithm population
#[derive(Debug, Clone)]
pub struct Individual {
    /// Parameter values (chromosome)
    pub parameters: Vec<ParameterValue>,
    /// Fitness score (higher is better)
    pub fitness: Option<f64>,
    /// Multi-objective fitness scores
    pub multi_fitness: Option<Vec<f64>>,
    /// Age of the individual (generations since creation)
    pub age: usize,
    /// Domination rank (for multi-objective optimization)
    pub rank: Option<usize>,
    /// Crowding distance (for multi-objective optimization)
    pub crowding_distance: Option<f64>,
}

impl Individual {
    /// Create a new individual with given parameters
    pub fn new(parameters: Vec<ParameterValue>) -> Self {
        Self {
            parameters,
            fitness: None,
            multi_fitness: None,
            age: 0,
            rank: None,
            crowding_distance: None,
        }
    }

    /// Increment the age of the individual
    pub fn age_increment(&mut self) {
        self.age += 1;
    }
}

/// Configuration for genetic algorithm
#[derive(Debug, Clone)]
pub struct GeneticAlgorithmConfig {
    /// Population size
    pub population_size: usize,
    /// Number of generations to run
    pub n_generations: usize,
    /// Crossover probability
    pub crossover_rate: f64,
    /// Mutation probability
    pub mutation_rate: f64,
    /// Elite size (top individuals to preserve)
    pub elite_size: usize,
    /// Tournament size for selection
    pub tournament_size: usize,
    /// Random state for reproducibility
    pub random_state: Option<u64>,
}

impl Default for GeneticAlgorithmConfig {
    fn default() -> Self {
        Self {
            population_size: 50,
            n_generations: 100,
            crossover_rate: 0.8,
            mutation_rate: 0.1,
            elite_size: 5,
            tournament_size: 3,
            random_state: None,
        }
    }
}

/// Parameter space definition for evolutionary optimization
#[derive(Debug, Clone)]
pub struct ParameterSpace {
    /// Parameter definitions
    pub parameters: Vec<ParameterDef>,
}

/// Definition of a single parameter
#[derive(Debug, Clone)]
pub enum ParameterDef {
    /// Continuous parameter with bounds
    Continuous { name: String, low: f64, high: f64 },
    /// Integer parameter with bounds
    Integer { name: String, low: i64, high: i64 },
    /// Categorical parameter with choices
    Categorical { name: String, choices: Vec<String> },
    /// Boolean parameter
    Boolean { name: String },
}

impl ParameterSpace {
    /// Create a new parameter space
    pub fn new() -> Self {
        Self {
            parameters: Vec::new(),
        }
    }

    /// Add a continuous parameter
    pub fn add_continuous(mut self, name: &str, low: f64, high: f64) -> Self {
        self.parameters.push(ParameterDef::Continuous {
            name: name.to_string(),
            low,
            high,
        });
        self
    }

    /// Add an integer parameter
    pub fn add_integer(mut self, name: &str, low: i64, high: i64) -> Self {
        self.parameters.push(ParameterDef::Integer {
            name: name.to_string(),
            low,
            high,
        });
        self
    }

    /// Add a categorical parameter
    pub fn add_categorical(mut self, name: &str, choices: Vec<&str>) -> Self {
        self.parameters.push(ParameterDef::Categorical {
            name: name.to_string(),
            choices: choices.iter().map(|s| s.to_string()).collect(),
        });
        self
    }

    /// Add a boolean parameter
    pub fn add_boolean(mut self, name: &str) -> Self {
        self.parameters.push(ParameterDef::Boolean {
            name: name.to_string(),
        });
        self
    }

    /// Sample parameter values
    pub fn sample(&self, rng: &mut StdRng) -> Vec<ParameterValue> {
        let mut parameters = Vec::new();

        for param_def in &self.parameters {
            let value = match param_def {
                ParameterDef::Continuous { low, high, .. } => {
                    ParameterValue::Float(rng.gen_range(*low..*high))
                }
                ParameterDef::Integer { low, high, .. } => {
                    ParameterValue::Integer(rng.gen_range(*low..*high))
                }
                ParameterDef::Categorical { choices, .. } => {
                    let choice = choices.as_slice().choose(rng).unwrap();
                    ParameterValue::String(choice.clone())
                }
                ParameterDef::Boolean { .. } => ParameterValue::Boolean(rng.gen_bool(0.5)),
            };
            parameters.push(value);
        }

        parameters
    }

    /// Generate a random individual
    pub fn random_individual(&self, rng: &mut StdRng) -> Individual {
        let mut parameters = Vec::new();

        for param_def in &self.parameters {
            let value = match param_def {
                ParameterDef::Continuous { low, high, .. } => {
                    ParameterValue::Float(rng.gen_range(*low..*high))
                }
                ParameterDef::Integer { low, high, .. } => {
                    ParameterValue::Integer(rng.gen_range(*low..*high))
                }
                ParameterDef::Categorical { choices, .. } => {
                    let choice = choices.as_slice().choose(rng).unwrap();
                    ParameterValue::String(choice.clone())
                }
                ParameterDef::Boolean { .. } => ParameterValue::Boolean(rng.gen_bool(0.5)),
            };
            parameters.push(value);
        }

        Individual::new(parameters)
    }

    /// Mutate an individual
    pub fn mutate(&self, individual: &mut Individual, rng: &mut StdRng, mutation_rate: f64) {
        for (i, param_def) in self.parameters.iter().enumerate() {
            if rng.gen_bool(mutation_rate) {
                match param_def {
                    ParameterDef::Continuous { low, high, .. } => {
                        // Gaussian mutation
                        if let ParameterValue::Float(ref mut val) = individual.parameters[i] {
                            let range = high - low;
                            let stddev = range * 0.1; // 10% of range as std deviation
                            let noise: f64 = rng.gen_range(-stddev..stddev);
                            *val = (*val + noise).clamp(*low, *high);
                        }
                    }
                    ParameterDef::Integer { low, high, .. } => {
                        // Random integer in range
                        individual.parameters[i] =
                            ParameterValue::Integer(rng.gen_range(*low..*high));
                    }
                    ParameterDef::Categorical { choices, .. } => {
                        // Random choice
                        let choice = choices.as_slice().choose(rng).unwrap();
                        individual.parameters[i] = ParameterValue::String(choice.clone());
                    }
                    ParameterDef::Boolean { .. } => {
                        // Flip boolean
                        if let ParameterValue::Boolean(ref mut val) = individual.parameters[i] {
                            *val = !*val;
                        }
                    }
                }
            }
        }
        individual.fitness = None; // Reset fitness after mutation
    }

    /// Perform crossover between two individuals
    pub fn crossover(
        &self,
        parent1: &Individual,
        parent2: &Individual,
        rng: &mut StdRng,
    ) -> (Individual, Individual) {
        let mut child1_params = Vec::new();
        let mut child2_params = Vec::new();

        for i in 0..self.parameters.len() {
            match &self.parameters[i] {
                ParameterDef::Continuous { .. } => {
                    // Arithmetic crossover for continuous parameters
                    if let (ParameterValue::Float(val1), ParameterValue::Float(val2)) =
                        (&parent1.parameters[i], &parent2.parameters[i])
                    {
                        let alpha = rng.gen_range(0.0..1.0);
                        let child1_val = alpha * val1 + (1.0 - alpha) * val2;
                        let child2_val = alpha * val2 + (1.0 - alpha) * val1;
                        child1_params.push(ParameterValue::Float(child1_val));
                        child2_params.push(ParameterValue::Float(child2_val));
                    }
                }
                _ => {
                    // Uniform crossover for discrete parameters
                    if rng.gen_bool(0.5) {
                        child1_params.push(parent1.parameters[i].clone());
                        child2_params.push(parent2.parameters[i].clone());
                    } else {
                        child1_params.push(parent2.parameters[i].clone());
                        child2_params.push(parent1.parameters[i].clone());
                    }
                }
            }
        }

        (
            Individual::new(child1_params),
            Individual::new(child2_params),
        )
    }
}

impl Default for ParameterSpace {
    fn default() -> Self {
        Self::new()
    }
}

/// Genetic Algorithm for hyperparameter optimization
pub struct GeneticAlgorithmCV<E, F, C> {
    estimator: E,
    parameter_space: ParameterSpace,
    cv: C,
    config: GeneticAlgorithmConfig,
    _phantom: std::marker::PhantomData<F>,
}

impl<E, F, C> GeneticAlgorithmCV<E, F, C>
where
    E: Clone,
    F: Clone,
    E: Fit<Array2<Float>, Array1<Float>, Fitted = F>,
    F: Predict<Array2<Float>, Array1<Float>>,
    F: Score<Array2<Float>, Array1<Float>, Float = f64>,
    C: CrossValidator,
{
    /// Create a new genetic algorithm optimizer
    pub fn new(estimator: E, parameter_space: ParameterSpace, cv: C) -> Self {
        Self {
            estimator,
            parameter_space,
            cv,
            config: GeneticAlgorithmConfig::default(),
            _phantom: std::marker::PhantomData,
        }
    }

    /// Set the configuration
    pub fn config(mut self, config: GeneticAlgorithmConfig) -> Self {
        self.config = config;
        self
    }

    /// Evaluate the fitness of an individual
    fn evaluate_fitness(
        &self,
        individual: &Individual,
        x: &Array2<Float>,
        y: &Array1<Float>,
    ) -> Result<f64> {
        // Configure estimator with parameters
        let configured_estimator =
            self.configure_estimator(&self.estimator, &individual.parameters)?;

        // Perform cross-validation
        let splits = self.cv.split(x.nrows(), None);
        let mut scores = Vec::new();

        for (train_idx, test_idx) in splits {
            let x_train = x.select(scirs2_core::ndarray::Axis(0), &train_idx);
            let y_train = y.select(scirs2_core::ndarray::Axis(0), &train_idx);
            let x_test = x.select(scirs2_core::ndarray::Axis(0), &test_idx);
            let y_test = y.select(scirs2_core::ndarray::Axis(0), &test_idx);

            let fitted = configured_estimator.clone().fit(&x_train, &y_train)?;
            let score = fitted.score(&x_test, &y_test)?;
            scores.push(score);
        }

        // Return mean cross-validation score
        Ok(scores.iter().sum::<f64>() / scores.len() as f64)
    }

    /// Configure estimator with parameters (placeholder - needs specific implementation)
    fn configure_estimator(&self, estimator: &E, _parameters: &[ParameterValue]) -> Result<E> {
        // This is a placeholder - in a real implementation, you would need a way to
        // configure the estimator with the given parameters
        Ok(estimator.clone())
    }

    /// Tournament selection
    fn tournament_selection<'a>(
        &self,
        population: &'a [Individual],
        rng: &mut StdRng,
    ) -> &'a Individual {
        let mut tournament: Vec<&Individual> = population
            .choose_multiple(rng, self.config.tournament_size)
            .collect();

        tournament.sort_by(|a, b| {
            b.fitness
                .unwrap_or(f64::NEG_INFINITY)
                .partial_cmp(&a.fitness.unwrap_or(f64::NEG_INFINITY))
                .unwrap()
        });

        tournament[0]
    }

    /// Run the genetic algorithm optimization
    pub fn fit(&mut self, x: &Array2<Float>, y: &Array1<Float>) -> Result<GeneticAlgorithmResult> {
        let mut rng = match self.config.random_state {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::seed_from_u64(42),
        };

        // Initialize population
        let mut population: Vec<Individual> = (0..self.config.population_size)
            .map(|_| self.parameter_space.random_individual(&mut rng))
            .collect();

        // Evaluate initial population
        for individual in &mut population {
            let fitness = self.evaluate_fitness(individual, x, y)?;
            individual.fitness = Some(fitness);
        }

        let mut best_scores = Vec::new();
        let mut best_individual = population[0].clone();

        // Evolution loop
        for generation in 0..self.config.n_generations {
            // Sort population by fitness
            population.sort_by(|a, b| {
                b.fitness
                    .unwrap_or(f64::NEG_INFINITY)
                    .partial_cmp(&a.fitness.unwrap_or(f64::NEG_INFINITY))
                    .unwrap()
            });

            // Track best individual
            if population[0].fitness.unwrap_or(f64::NEG_INFINITY)
                > best_individual.fitness.unwrap_or(f64::NEG_INFINITY)
            {
                best_individual = population[0].clone();
            }

            best_scores.push(best_individual.fitness.unwrap_or(f64::NEG_INFINITY));

            // Create new generation
            let mut new_population = Vec::new();

            // Elitism: preserve best individuals
            for i in 0..self.config.elite_size {
                new_population.push(population[i].clone());
            }

            // Generate offspring
            while new_population.len() < self.config.population_size {
                // Selection
                let parent1 = self.tournament_selection(&population, &mut rng);
                let parent2 = self.tournament_selection(&population, &mut rng);

                // Crossover
                let (mut child1, mut child2) = if rng.gen_bool(self.config.crossover_rate) {
                    self.parameter_space.crossover(parent1, parent2, &mut rng)
                } else {
                    (parent1.clone(), parent2.clone())
                };

                // Mutation
                self.parameter_space
                    .mutate(&mut child1, &mut rng, self.config.mutation_rate);
                self.parameter_space
                    .mutate(&mut child2, &mut rng, self.config.mutation_rate);

                // Evaluate offspring
                if child1.fitness.is_none() {
                    let fitness = self.evaluate_fitness(&child1, x, y)?;
                    child1.fitness = Some(fitness);
                }
                if child2.fitness.is_none() {
                    let fitness = self.evaluate_fitness(&child2, x, y)?;
                    child2.fitness = Some(fitness);
                }

                new_population.push(child1);
                if new_population.len() < self.config.population_size {
                    new_population.push(child2);
                }
            }

            // Age population
            for individual in &mut new_population {
                individual.age_increment();
            }

            population = new_population;

            // Optional: early stopping could be implemented here
            println!(
                "Generation {}: Best score = {:.6}",
                generation, best_scores[generation]
            );
        }

        Ok(GeneticAlgorithmResult {
            best_parameters: best_individual.parameters,
            best_score: best_individual.fitness.unwrap_or(f64::NEG_INFINITY),
            score_history: best_scores,
            n_evaluations: self.config.n_generations * self.config.population_size,
        })
    }
}

/// Result of genetic algorithm optimization
#[derive(Debug, Clone)]
pub struct GeneticAlgorithmResult {
    /// Best parameters found
    pub best_parameters: Vec<ParameterValue>,
    /// Best score achieved
    pub best_score: f64,
    /// History of best scores per generation
    pub score_history: Vec<f64>,
    /// Total number of evaluations performed
    pub n_evaluations: usize,
}

/// Multi-objective optimization using NSGA-II algorithm
///
/// This implements the Non-dominated Sorting Genetic Algorithm II (NSGA-II)
/// for multi-objective optimization problems where multiple conflicting objectives
/// need to be optimized simultaneously.
pub struct MultiObjectiveGA<E, F, C> {
    estimator: E,
    parameter_space: ParameterSpace,
    cv: C,
    config: GeneticAlgorithmConfig,
    objective_functions:
        Vec<Box<dyn Fn(&F, &Array2<Float>, &Array1<Float>) -> Result<f64> + Send + Sync>>,
    _phantom: std::marker::PhantomData<F>,
}

impl<E, F, C> MultiObjectiveGA<E, F, C>
where
    E: Clone,
    F: Clone,
    E: Fit<Array2<Float>, Array1<Float>, Fitted = F>,
    F: Predict<Array2<Float>, Array1<Float>>,
    F: Score<Array2<Float>, Array1<Float>, Float = f64>,
    C: CrossValidator,
{
    /// Create a new multi-objective genetic algorithm optimizer
    pub fn new(estimator: E, parameter_space: ParameterSpace, cv: C) -> Self {
        Self {
            estimator,
            parameter_space,
            cv,
            config: GeneticAlgorithmConfig::default(),
            objective_functions: Vec::new(),
            _phantom: std::marker::PhantomData,
        }
    }

    /// Set the configuration
    pub fn config(mut self, config: GeneticAlgorithmConfig) -> Self {
        self.config = config;
        self
    }

    /// Add an objective function
    pub fn add_objective<FObj>(mut self, objective: FObj) -> Self
    where
        FObj: Fn(&F, &Array2<Float>, &Array1<Float>) -> Result<f64> + Send + Sync + 'static,
    {
        self.objective_functions.push(Box::new(objective));
        self
    }

    /// Check if individual A dominates individual B
    fn dominates(a: &Individual, b: &Individual) -> bool {
        if let (Some(fitness_a), Some(fitness_b)) = (&a.multi_fitness, &b.multi_fitness) {
            let mut a_better = false;
            let mut a_worse = false;

            for (fa, fb) in fitness_a.iter().zip(fitness_b.iter()) {
                if fa > fb {
                    a_better = true;
                } else if fa < fb {
                    a_worse = true;
                }
            }

            a_better && !a_worse
        } else {
            false
        }
    }

    /// Perform non-dominated sorting
    fn non_dominated_sort(population: &mut [Individual]) -> Vec<Vec<usize>> {
        let n = population.len();
        let mut fronts = Vec::new();
        let mut domination_count = vec![0; n];
        let mut dominated_solutions = vec![Vec::new(); n];

        // Calculate domination relationships
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    if Self::dominates(&population[i], &population[j]) {
                        dominated_solutions[i].push(j);
                    } else if Self::dominates(&population[j], &population[i]) {
                        domination_count[i] += 1;
                    }
                }
            }
        }

        // First front
        let mut current_front = Vec::new();
        for i in 0..n {
            if domination_count[i] == 0 {
                population[i].rank = Some(0);
                current_front.push(i);
            }
        }
        fronts.push(current_front.clone());

        // Subsequent fronts
        let mut front_num = 0;
        while !current_front.is_empty() {
            let mut next_front = Vec::new();
            for &i in &current_front {
                for &j in &dominated_solutions[i] {
                    domination_count[j] -= 1;
                    if domination_count[j] == 0 {
                        population[j].rank = Some(front_num + 1);
                        next_front.push(j);
                    }
                }
            }
            front_num += 1;
            current_front = next_front.clone();
            if !next_front.is_empty() {
                fronts.push(next_front);
            }
        }

        fronts
    }

    /// Calculate crowding distance for diversity preservation
    fn calculate_crowding_distance(population: &mut [Individual], front: &[usize]) {
        let front_size = front.len();
        if front_size <= 2 {
            for &idx in front {
                population[idx].crowding_distance = Some(f64::INFINITY);
            }
            return;
        }

        // Initialize crowding distance
        for &idx in front {
            population[idx].crowding_distance = Some(0.0);
        }

        if let Some(n_objectives) = population[front[0]].multi_fitness.as_ref().map(|f| f.len()) {
            for obj in 0..n_objectives {
                // Sort by objective
                let mut front_sorted = front.to_vec();
                front_sorted.sort_by(|&a, &b| {
                    let fitness_a = &population[a].multi_fitness.as_ref().unwrap()[obj];
                    let fitness_b = &population[b].multi_fitness.as_ref().unwrap()[obj];
                    fitness_a
                        .partial_cmp(fitness_b)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });

                // Set boundary points to infinity
                population[front_sorted[0]].crowding_distance = Some(f64::INFINITY);
                population[front_sorted[front_size - 1]].crowding_distance = Some(f64::INFINITY);

                // Calculate crowding distance for internal points
                let obj_min = population[front_sorted[0]].multi_fitness.as_ref().unwrap()[obj];
                let obj_max = population[front_sorted[front_size - 1]]
                    .multi_fitness
                    .as_ref()
                    .unwrap()[obj];
                let obj_range = obj_max - obj_min;

                if obj_range > 0.0 {
                    for i in 1..front_size - 1 {
                        let current_idx = front_sorted[i];
                        let prev_fitness = population[front_sorted[i - 1]]
                            .multi_fitness
                            .as_ref()
                            .unwrap()[obj];
                        let next_fitness = population[front_sorted[i + 1]]
                            .multi_fitness
                            .as_ref()
                            .unwrap()[obj];

                        let distance = population[current_idx].crowding_distance.unwrap_or(0.0);
                        population[current_idx].crowding_distance =
                            Some(distance + (next_fitness - prev_fitness) / obj_range);
                    }
                }
            }
        }
    }

    /// Evaluate multi-objective fitness
    fn evaluate_multi_fitness(
        &self,
        individual: &Individual,
        x: &Array2<Float>,
        y: &Array1<Float>,
    ) -> Result<Vec<f64>> {
        let configured_estimator =
            self.configure_estimator(&self.estimator, &individual.parameters)?;
        let fitted = configured_estimator.fit(x, y)?;

        let mut objectives = Vec::new();
        for objective_fn in &self.objective_functions {
            let score = objective_fn(&fitted, x, y)?;
            objectives.push(score);
        }

        Ok(objectives)
    }

    /// Configure estimator with parameters (placeholder)
    fn configure_estimator(&self, estimator: &E, _parameters: &[ParameterValue]) -> Result<E> {
        Ok(estimator.clone())
    }

    /// Run multi-objective optimization
    pub fn fit(&mut self, x: &Array2<Float>, y: &Array1<Float>) -> Result<MultiObjectiveResult> {
        if self.objective_functions.is_empty() {
            return Err(sklears_core::error::SklearsError::InvalidParameter {
                name: "objective_functions".to_string(),
                reason: "At least one objective function must be specified".to_string(),
            });
        }

        let mut rng = match self.config.random_state {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::seed_from_u64(42),
        };

        // Initialize population
        let mut population: Vec<Individual> = (0..self.config.population_size)
            .map(|_| self.parameter_space.random_individual(&mut rng))
            .collect();

        // Evaluate initial population
        for individual in &mut population {
            let multi_fitness = self.evaluate_multi_fitness(individual, x, y)?;
            individual.multi_fitness = Some(multi_fitness);
        }

        let mut pareto_history = Vec::new();

        // Evolution loop
        for generation in 0..self.config.n_generations {
            // Non-dominated sorting
            let fronts = Self::non_dominated_sort(&mut population);

            // Calculate crowding distance for each front
            for front in &fronts {
                Self::calculate_crowding_distance(&mut population, front);
            }

            // Store Pareto front
            let pareto_front: Vec<Individual> = fronts[0]
                .iter()
                .map(|&idx| population[idx].clone())
                .collect();
            pareto_history.push(pareto_front);

            // Generate offspring
            let mut offspring = Vec::new();
            while offspring.len() < self.config.population_size {
                // Tournament selection based on rank and crowding distance
                let parent1 = self.nsga_tournament_selection(&population, &mut rng);
                let parent2 = self.nsga_tournament_selection(&population, &mut rng);

                // Crossover
                let (mut child1, mut child2) = if rng.gen_bool(self.config.crossover_rate) {
                    self.parameter_space.crossover(parent1, parent2, &mut rng)
                } else {
                    (parent1.clone(), parent2.clone())
                };

                // Mutation
                self.parameter_space
                    .mutate(&mut child1, &mut rng, self.config.mutation_rate);
                self.parameter_space
                    .mutate(&mut child2, &mut rng, self.config.mutation_rate);

                // Evaluate offspring
                if child1.multi_fitness.is_none() {
                    let multi_fitness = self.evaluate_multi_fitness(&child1, x, y)?;
                    child1.multi_fitness = Some(multi_fitness);
                }
                if child2.multi_fitness.is_none() {
                    let multi_fitness = self.evaluate_multi_fitness(&child2, x, y)?;
                    child2.multi_fitness = Some(multi_fitness);
                }

                offspring.push(child1);
                if offspring.len() < self.config.population_size {
                    offspring.push(child2);
                }
            }

            // Combine population and offspring
            population.extend(offspring);

            // Environmental selection
            let fronts = Self::non_dominated_sort(&mut population);
            let mut new_population = Vec::new();

            // Add fronts until population size is reached
            for front in &fronts {
                if new_population.len() + front.len() <= self.config.population_size {
                    for &idx in front {
                        new_population.push(population[idx].clone());
                    }
                } else {
                    // Calculate crowding distance and select best individuals
                    Self::calculate_crowding_distance(&mut population, front);
                    let mut front_individuals: Vec<_> =
                        front.iter().map(|&idx| &population[idx]).collect();

                    // Sort by crowding distance (descending)
                    front_individuals.sort_by(|a, b| {
                        b.crowding_distance
                            .unwrap_or(0.0)
                            .partial_cmp(&a.crowding_distance.unwrap_or(0.0))
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });

                    let remaining_slots = self.config.population_size - new_population.len();
                    for individual in front_individuals.into_iter().take(remaining_slots) {
                        new_population.push(individual.clone());
                    }
                    break;
                }
            }

            population = new_population;

            println!(
                "Generation {}: Pareto front size = {}",
                generation,
                pareto_history.last().unwrap().len()
            );
        }

        // Final non-dominated sorting
        let fronts = Self::non_dominated_sort(&mut population);
        let final_pareto_front: Vec<Individual> = fronts[0]
            .iter()
            .map(|&idx| population[idx].clone())
            .collect();

        Ok(MultiObjectiveResult {
            pareto_front: final_pareto_front,
            pareto_history,
            n_evaluations: self.config.n_generations * self.config.population_size * 2, // Including offspring
        })
    }

    /// NSGA-II tournament selection
    fn nsga_tournament_selection<'a>(
        &self,
        population: &'a [Individual],
        rng: &mut StdRng,
    ) -> &'a Individual {
        let tournament: Vec<&Individual> = population
            .choose_multiple(rng, self.config.tournament_size)
            .collect();

        // Select best individual based on rank and crowding distance
        tournament
            .into_iter()
            .min_by(|a, b| {
                // First compare rank
                let rank_cmp = a
                    .rank
                    .unwrap_or(usize::MAX)
                    .cmp(&b.rank.unwrap_or(usize::MAX));
                if rank_cmp != std::cmp::Ordering::Equal {
                    return rank_cmp;
                }

                // If ranks are equal, compare crowding distance (higher is better)
                b.crowding_distance
                    .unwrap_or(0.0)
                    .partial_cmp(&a.crowding_distance.unwrap_or(0.0))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap()
    }
}

/// Result of multi-objective optimization
#[derive(Debug, Clone)]
pub struct MultiObjectiveResult {
    /// Final Pareto front
    pub pareto_front: Vec<Individual>,
    /// History of Pareto fronts for each generation
    pub pareto_history: Vec<Vec<Individual>>,
    /// Total number of evaluations performed
    pub n_evaluations: usize,
}

impl MultiObjectiveResult {
    /// Get the hypervolume of the Pareto front
    pub fn hypervolume(&self, reference_point: &[f64]) -> f64 {
        if self.pareto_front.is_empty() {
            return 0.0;
        }

        // Simple hypervolume calculation for 2D case
        if reference_point.len() == 2 {
            let mut points: Vec<(f64, f64)> = self
                .pareto_front
                .iter()
                .filter_map(|ind| {
                    if let Some(fitness) = &ind.multi_fitness {
                        if fitness.len() >= 2 {
                            Some((fitness[0], fitness[1]))
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                })
                .collect();

            if points.is_empty() {
                return 0.0;
            }

            // Sort by first objective
            points.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

            let mut hv = 0.0;
            let mut prev_x = reference_point[0];

            for (x, y) in points {
                if x > prev_x && y > reference_point[1] {
                    hv += (x - prev_x) * (y - reference_point[1]);
                    prev_x = x;
                }
            }

            hv
        } else {
            // For higher dimensions, return 0 (would need more complex algorithm)
            0.0
        }
    }
}

/// Cross-validation wrapper for evolutionary search
#[derive(Debug, Clone)]
pub struct EvolutionarySearchCV<E, S> {
    estimator: E,
    parameter_space: ParameterSpace,
    scorer: Box<S>,
    cv_folds: usize,
    population_size: usize,
    n_generations: usize,
    crossover_rate: f64,
    mutation_rate: f64,
    random_state: Option<u64>,
}

impl<E, S> EvolutionarySearchCV<E, S>
where
    E: Clone + Fit<Array2<Float>, Array1<Float>> + Predict<Array2<Float>, Array1<Float>>,
    S: Fn(&E::Fitted, &Array2<Float>, &Array1<Float>) -> Result<f64>,
{
    /// Create a new evolutionary search with cross-validation
    pub fn new(
        estimator: E,
        parameter_space: ParameterSpace,
        scorer: Box<S>,
        cv_folds: usize,
    ) -> Self {
        Self {
            estimator,
            parameter_space,
            scorer,
            cv_folds,
            population_size: 50,
            n_generations: 100,
            crossover_rate: 0.8,
            mutation_rate: 0.1,
            random_state: None,
        }
    }

    /// Set population size
    pub fn set_population_size(&mut self, population_size: usize) {
        self.population_size = population_size;
    }

    /// Set number of generations
    pub fn set_n_generations(&mut self, n_generations: usize) {
        self.n_generations = n_generations;
    }

    /// Set random state for reproducibility
    pub fn set_random_state(&mut self, seed: u64) {
        self.random_state = Some(seed);
    }

    /// Set crossover rate
    pub fn set_crossover_rate(&mut self, rate: f64) {
        self.crossover_rate = rate.clamp(0.0, 1.0);
    }

    /// Set mutation rate
    pub fn set_mutation_rate(&mut self, rate: f64) {
        self.mutation_rate = rate.clamp(0.0, 1.0);
    }

    /// Perform evolutionary search with cross-validation
    pub fn fit(&self, x: &Array2<Float>, y: &Array1<Float>) -> Result<EvolutionarySearchResult> {
        let mut rng = match self.random_state {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::seed_from_u64(42),
        };

        // Initialize population
        let mut population: Vec<Individual> = (0..self.population_size)
            .map(|_| {
                let params = self.parameter_space.sample(&mut rng);
                Individual::new(params)
            })
            .collect();

        // Evaluate initial population
        for individual in &mut population {
            individual.fitness = Some(self.evaluate_individual(individual, x, y)?);
        }

        let mut best_individual = population[0].clone();
        let mut best_fitness = population[0].fitness.unwrap();

        // Evolution loop
        for generation in 0..self.n_generations {
            // Selection and reproduction
            let mut new_population = Vec::new();

            // Elitism: keep best individuals
            let elite_size = (self.population_size as f64 * 0.1) as usize;
            population.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap());
            new_population.extend_from_slice(&population[..elite_size]);

            // Generate offspring
            while new_population.len() < self.population_size {
                // Tournament selection
                let parent1 = self.tournament_selection(&population, &mut rng);
                let parent2 = self.tournament_selection(&population, &mut rng);

                // Crossover
                let (mut child1, mut child2) = if rng.gen::<f64>() < self.crossover_rate {
                    self.parameter_space.crossover(parent1, parent2, &mut rng)
                } else {
                    (parent1.clone(), parent2.clone())
                };

                // Mutation
                if rng.gen::<f64>() < self.mutation_rate {
                    self.parameter_space.mutate(&mut child1, &mut rng, 0.1);
                }
                if rng.gen::<f64>() < self.mutation_rate {
                    self.parameter_space.mutate(&mut child2, &mut rng, 0.1);
                }

                // Evaluate offspring
                child1.fitness = Some(self.evaluate_individual(&child1, x, y)?);
                child2.fitness = Some(self.evaluate_individual(&child2, x, y)?);

                new_population.push(child1);
                if new_population.len() < self.population_size {
                    new_population.push(child2);
                }
            }

            population = new_population;

            // Update best individual
            for individual in &population {
                let fitness = individual.fitness.unwrap();
                if fitness > best_fitness {
                    best_fitness = fitness;
                    best_individual = individual.clone();
                }
            }
        }

        Ok(EvolutionarySearchResult {
            best_params: best_individual.parameters,
            best_score: best_fitness,
            n_generations: self.n_generations,
            population_size: self.population_size,
        })
    }

    /// Evaluate individual using cross-validation
    fn evaluate_individual(
        &self,
        individual: &Individual,
        x: &Array2<Float>,
        y: &Array1<Float>,
    ) -> Result<f64> {
        // Simple train-test split for evaluation (placeholder for full CV)
        let n_samples = x.nrows();
        let train_size = (n_samples as f64 * 0.8) as usize;

        let x_train = x
            .slice(scirs2_core::ndarray::s![..train_size, ..])
            .to_owned();
        let y_train = y.slice(scirs2_core::ndarray::s![..train_size]).to_owned();
        let x_test = x
            .slice(scirs2_core::ndarray::s![train_size.., ..])
            .to_owned();
        let y_test = y.slice(scirs2_core::ndarray::s![train_size..]).to_owned();

        // Configure estimator with parameters (simplified)
        let estimator = self.estimator.clone();
        let fitted = estimator.fit(&x_train, &y_train)?;

        (self.scorer)(&fitted, &x_test, &y_test)
    }

    /// Tournament selection
    fn tournament_selection<'a>(
        &self,
        population: &'a [Individual],
        rng: &mut StdRng,
    ) -> &'a Individual {
        let tournament_size = 3;
        let tournament: Vec<&Individual> =
            population.choose_multiple(rng, tournament_size).collect();

        tournament
            .into_iter()
            .max_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap())
            .unwrap()
    }
}

/// Result of evolutionary search
#[derive(Debug, Clone)]
pub struct EvolutionarySearchResult {
    pub best_params: Vec<ParameterValue>,
    pub best_score: f64,
    pub n_generations: usize,
    pub population_size: usize,
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parameter_space_creation() {
        let param_space = ParameterSpace::new()
            .add_continuous("alpha", 0.1, 10.0)
            .add_integer("max_depth", 1, 20)
            .add_categorical("kernel", vec!["linear", "rbf", "poly"])
            .add_boolean("fit_intercept");

        assert_eq!(param_space.parameters.len(), 4);
    }

    #[test]
    fn test_random_individual_generation() {
        let param_space = ParameterSpace::new()
            .add_continuous("alpha", 0.1, 10.0)
            .add_integer("max_depth", 1, 20)
            .add_categorical("kernel", vec!["linear", "rbf"])
            .add_boolean("fit_intercept");

        let mut rng = StdRng::seed_from_u64(42);
        let individual = param_space.random_individual(&mut rng);

        assert_eq!(individual.parameters.len(), 4);
        assert!(matches!(individual.parameters[0], ParameterValue::Float(_)));
        assert!(matches!(
            individual.parameters[1],
            ParameterValue::Integer(_)
        ));
        assert!(matches!(
            individual.parameters[2],
            ParameterValue::String(_)
        ));
        assert!(matches!(
            individual.parameters[3],
            ParameterValue::Boolean(_)
        ));
    }

    #[test]
    fn test_crossover() {
        let param_space = ParameterSpace::new()
            .add_continuous("alpha", 0.1, 10.0)
            .add_categorical("kernel", vec!["linear", "rbf"]);

        let parent1 = Individual::new(vec![
            ParameterValue::Float(1.0),
            ParameterValue::String("linear".to_string()),
        ]);
        let parent2 = Individual::new(vec![
            ParameterValue::Float(5.0),
            ParameterValue::String("rbf".to_string()),
        ]);

        let mut rng = StdRng::seed_from_u64(42);
        let (child1, child2) = param_space.crossover(&parent1, &parent2, &mut rng);

        assert_eq!(child1.parameters.len(), 2);
        assert_eq!(child2.parameters.len(), 2);
    }

    #[test]
    fn test_mutation() {
        let param_space = ParameterSpace::new()
            .add_continuous("alpha", 0.1, 10.0)
            .add_boolean("fit_intercept");

        let mut individual = Individual::new(vec![
            ParameterValue::Float(5.0),
            ParameterValue::Boolean(true),
        ]);

        let mut rng = StdRng::seed_from_u64(42);
        param_space.mutate(&mut individual, &mut rng, 1.0); // 100% mutation rate

        // Values should have changed (though specific values depend on RNG)
        assert_eq!(individual.parameters.len(), 2);
        assert!(individual.fitness.is_none()); // Fitness should be reset
    }

    #[test]
    fn test_genetic_algorithm_config() {
        let config = GeneticAlgorithmConfig {
            population_size: 30,
            n_generations: 50,
            crossover_rate: 0.9,
            mutation_rate: 0.05,
            elite_size: 3,
            tournament_size: 5,
            random_state: Some(123),
        };

        assert_eq!(config.population_size, 30);
        assert_eq!(config.n_generations, 50);
        assert_eq!(config.crossover_rate, 0.9);
        assert_eq!(config.mutation_rate, 0.05);
        assert_eq!(config.elite_size, 3);
        assert_eq!(config.tournament_size, 5);
        assert_eq!(config.random_state, Some(123));
    }
}
