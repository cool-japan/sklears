//! Multi-Objective Optimization for Multi-Output Learning
//!
//! This module provides advanced multi-objective optimization techniques for multi-output
//! learning problems, where multiple conflicting objectives need to be optimized simultaneously.
//! It implements genetic algorithm-based approaches to find Pareto-optimal solutions.
//!
//! ## Key Features
//!
//! - **Genetic Algorithm**: Population-based evolutionary optimization
//! - **Pareto Optimization**: Find trade-off solutions between conflicting objectives
//! - **Non-dominated Sorting**: Efficient ranking of solutions using NSGA-II principles
//! - **Crowding Distance**: Maintain diversity in the Pareto front
//! - **Multiple Objectives**: Support for accuracy, complexity, MSE, MAE, and custom objectives
//! - **Tournament Selection**: Efficient parent selection for reproduction

// Use SciRS2-Core for arrays and random number generation (SciRS2 Policy)
use scirs2_core::ndarray::{s, Array1, Array2, ArrayView2};
use scirs2_core::random::thread_rng;
use scirs2_core::random::{RandNormal, Rng};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Predict, Untrained},
    types::Float,
};

/// Multi-Objective Optimization for multi-output learning
#[derive(Debug, Clone)]
pub struct MultiObjectiveOptimizer<S = Untrained> {
    state: S,
    config: MultiObjectiveConfig,
}

/// Configuration for Multi-Objective Optimization
#[derive(Debug, Clone)]
pub struct MultiObjectiveConfig {
    /// Population size for genetic algorithm
    pub population_size: usize,
    /// Number of generations
    pub generations: usize,
    /// Mutation rate
    pub mutation_rate: Float,
    /// Crossover rate
    pub crossover_rate: Float,
    /// Selection pressure
    pub selection_pressure: Float,
    /// Objective functions
    pub objectives: Vec<String>,
    /// Random state for reproducibility
    pub random_state: Option<u64>,
}

impl Default for MultiObjectiveConfig {
    fn default() -> Self {
        Self {
            population_size: 100,
            generations: 100,
            mutation_rate: 0.1,
            crossover_rate: 0.8,
            selection_pressure: 2.0,
            objectives: vec!["accuracy".to_string(), "complexity".to_string()],
            random_state: None,
        }
    }
}

/// Pareto-optimal solution
#[derive(Debug, Clone)]
pub struct ParetoSolution {
    /// Solution parameters
    pub parameters: Array1<Float>,
    /// Objective values
    pub objectives: Array1<Float>,
    /// Dominance rank
    pub rank: usize,
    /// Crowding distance
    pub crowding_distance: Float,
}

/// Trained state for Multi-Objective Optimizer
#[derive(Debug, Clone)]
pub struct MultiObjectiveOptimizerTrained {
    /// Pareto-optimal solutions
    pub pareto_solutions: Vec<ParetoSolution>,
    /// Best compromise solution
    pub best_solution: ParetoSolution,
    /// Convergence history
    pub convergence_history: Vec<Float>,
    /// Configuration used for optimization
    pub config: MultiObjectiveConfig,
    /// Number of outputs
    pub n_outputs: usize,
}

impl MultiObjectiveOptimizer<Untrained> {
    /// Create a new Multi-Objective Optimizer
    pub fn new() -> Self {
        Self {
            state: Untrained,
            config: MultiObjectiveConfig::default(),
        }
    }

    /// Set the configuration
    pub fn config(mut self, config: MultiObjectiveConfig) -> Self {
        self.config = config;
        self
    }

    /// Set the population size
    pub fn population_size(mut self, population_size: usize) -> Self {
        self.config.population_size = population_size;
        self
    }

    /// Set the number of generations
    pub fn generations(mut self, generations: usize) -> Self {
        self.config.generations = generations;
        self
    }

    /// Set the mutation rate
    pub fn mutation_rate(mut self, mutation_rate: Float) -> Self {
        self.config.mutation_rate = mutation_rate;
        self
    }

    /// Set the crossover rate
    pub fn crossover_rate(mut self, crossover_rate: Float) -> Self {
        self.config.crossover_rate = crossover_rate;
        self
    }

    /// Set the selection pressure
    pub fn selection_pressure(mut self, selection_pressure: Float) -> Self {
        self.config.selection_pressure = selection_pressure;
        self
    }

    /// Set the objective functions
    pub fn objectives(mut self, objectives: Vec<String>) -> Self {
        self.config.objectives = objectives;
        self
    }

    /// Set the random state
    pub fn random_state(mut self, random_state: Option<u64>) -> Self {
        self.config.random_state = random_state;
        self
    }
}

impl Default for MultiObjectiveOptimizer<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for MultiObjectiveOptimizer<Untrained> {
    type Config = MultiObjectiveConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Fit<ArrayView2<'_, Float>, ArrayView2<'_, Float>> for MultiObjectiveOptimizer<Untrained> {
    type Fitted = MultiObjectiveOptimizer<MultiObjectiveOptimizerTrained>;

    fn fit(self, X: &ArrayView2<'_, Float>, y: &ArrayView2<'_, Float>) -> SklResult<Self::Fitted> {
        let (n_samples, n_features) = X.dim();
        let (y_samples, n_outputs) = y.dim();

        if n_samples != y_samples {
            return Err(SklearsError::InvalidInput(
                "X and y must have the same number of samples".to_string(),
            ));
        }

        let mut rng = thread_rng();

        // Initialize population
        let mut population = self.initialize_population(n_features, n_outputs, &mut rng)?;
        let mut convergence_history = Vec::new();

        for generation in 0..self.config.generations {
            // Evaluate objectives for all solutions
            self.evaluate_population(&mut population, X, y)?;

            // Non-dominated sorting
            self.non_dominated_sort(&mut population)?;

            // Calculate crowding distance
            self.calculate_crowding_distance(&mut population)?;

            // Selection, crossover, and mutation
            population = self.evolve_population(population, &mut rng)?;

            // Track convergence
            let hypervolume = self.calculate_hypervolume(&population)?;
            convergence_history.push(hypervolume);
        }

        // Final evaluation and sorting
        self.evaluate_population(&mut population, X, y)?;
        self.non_dominated_sort(&mut population)?;

        // Extract Pareto-optimal solutions
        let pareto_solutions: Vec<ParetoSolution> =
            population.into_iter().filter(|sol| sol.rank == 0).collect();

        // Find best compromise solution (closest to ideal point)
        let best_solution = self.find_best_compromise(&pareto_solutions)?;

        Ok(MultiObjectiveOptimizer {
            state: MultiObjectiveOptimizerTrained {
                pareto_solutions,
                best_solution,
                convergence_history,
                config: self.config.clone(),
                n_outputs,
            },
            config: self.config,
        })
    }
}

impl MultiObjectiveOptimizer<Untrained> {
    /// Initialize population with random solutions
    fn initialize_population(
        &self,
        n_features: usize,
        n_outputs: usize,
        rng: &mut scirs2_core::random::CoreRandom,
    ) -> SklResult<Vec<ParetoSolution>> {
        let mut population = Vec::new();

        for _ in 0..self.config.population_size {
            // Random parameters (weights and bias)
            let param_size = n_features * n_outputs + n_outputs;
            let normal_dist = RandNormal::new(0.0, 1.0).unwrap();
            let mut parameters = Array1::<Float>::zeros(param_size);
            for i in 0..param_size {
                parameters[i] = rng.sample(normal_dist);
            }

            let solution = ParetoSolution {
                parameters,
                objectives: Array1::<Float>::zeros(self.config.objectives.len()),
                rank: 0,
                crowding_distance: 0.0,
            };

            population.push(solution);
        }

        Ok(population)
    }

    /// Evaluate objectives for all solutions in the population
    fn evaluate_population(
        &self,
        population: &mut [ParetoSolution],
        X: &ArrayView2<'_, Float>,
        y: &ArrayView2<'_, Float>,
    ) -> SklResult<()> {
        let (n_samples, n_features) = X.dim();
        let n_outputs = y.ncols();

        for solution in population.iter_mut() {
            // Extract weights and bias from parameters
            let weights_size = n_features * n_outputs;
            let weights = solution
                .parameters
                .slice(s![..weights_size])
                .to_owned()
                .into_shape((n_features, n_outputs))
                .unwrap();
            let bias = solution.parameters.slice(s![weights_size..]).to_owned();

            // Make predictions
            let predictions = X.dot(&weights) + &bias;

            // Calculate objectives
            let mut objectives = Array1::<Float>::zeros(self.config.objectives.len());

            for (i, objective) in self.config.objectives.iter().enumerate() {
                let objective_value = match objective.as_str() {
                    "accuracy" => self.calculate_accuracy(&predictions, y)?,
                    "complexity" => self.calculate_complexity(&weights, &bias)?,
                    "mse" => self.calculate_mse(&predictions, y)?,
                    "mae" => self.calculate_mae(&predictions, y)?,
                    _ => {
                        return Err(SklearsError::InvalidInput(format!(
                            "Unknown objective: {}",
                            objective
                        )))
                    }
                };
                objectives[i] = objective_value;
            }

            solution.objectives = objectives;
        }

        Ok(())
    }

    /// Calculate accuracy objective
    fn calculate_accuracy(
        &self,
        predictions: &Array2<Float>,
        y: &ArrayView2<'_, Float>,
    ) -> SklResult<Float> {
        let mse = predictions
            .iter()
            .zip(y.iter())
            .map(|(pred, true_val)| (pred - true_val).powi(2))
            .sum::<Float>()
            / (predictions.len() as Float);
        Ok(-mse) // Negative because we want to minimize MSE (maximize accuracy)
    }

    /// Calculate complexity objective
    fn calculate_complexity(
        &self,
        weights: &Array2<Float>,
        bias: &Array1<Float>,
    ) -> SklResult<Float> {
        let weight_complexity = weights.mapv(|x| x.abs()).sum();
        let bias_complexity = bias.mapv(|x| x.abs()).sum();
        Ok(weight_complexity + bias_complexity)
    }

    /// Calculate MSE objective
    fn calculate_mse(
        &self,
        predictions: &Array2<Float>,
        y: &ArrayView2<'_, Float>,
    ) -> SklResult<Float> {
        let mse = predictions
            .iter()
            .zip(y.iter())
            .map(|(pred, true_val)| (pred - true_val).powi(2))
            .sum::<Float>()
            / (predictions.len() as Float);
        Ok(mse)
    }

    /// Calculate MAE objective
    fn calculate_mae(
        &self,
        predictions: &Array2<Float>,
        y: &ArrayView2<'_, Float>,
    ) -> SklResult<Float> {
        let mae = predictions
            .iter()
            .zip(y.iter())
            .map(|(pred, true_val)| (pred - true_val).abs())
            .sum::<Float>()
            / (predictions.len() as Float);
        Ok(mae)
    }

    /// Non-dominated sorting
    fn non_dominated_sort(&self, population: &mut [ParetoSolution]) -> SklResult<()> {
        let n = population.len();
        let mut domination_count = vec![0; n];
        let mut dominated_solutions = vec![Vec::new(); n];

        // Calculate domination relationships
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    if self.dominates(&population[i], &population[j]) {
                        dominated_solutions[i].push(j);
                    } else if self.dominates(&population[j], &population[i]) {
                        domination_count[i] += 1;
                    }
                }
            }
        }

        // Assign ranks
        let mut current_rank = 0;
        let mut current_front: Vec<usize> = (0..n).filter(|&i| domination_count[i] == 0).collect();

        while !current_front.is_empty() {
            let mut next_front = Vec::new();

            for &i in &current_front {
                population[i].rank = current_rank;

                for &j in &dominated_solutions[i] {
                    domination_count[j] -= 1;
                    if domination_count[j] == 0 {
                        next_front.push(j);
                    }
                }
            }

            current_front = next_front;
            current_rank += 1;
        }

        Ok(())
    }

    /// Check if solution a dominates solution b
    fn dominates(&self, a: &ParetoSolution, b: &ParetoSolution) -> bool {
        let mut at_least_one_better = false;

        for i in 0..a.objectives.len() {
            if a.objectives[i] < b.objectives[i] {
                return false; // a is worse in at least one objective
            } else if a.objectives[i] > b.objectives[i] {
                at_least_one_better = true;
            }
        }

        at_least_one_better
    }

    /// Calculate crowding distance
    fn calculate_crowding_distance(&self, population: &mut [ParetoSolution]) -> SklResult<()> {
        let n = population.len();
        let n_objectives = self.config.objectives.len();

        // Initialize crowding distances
        for solution in population.iter_mut() {
            solution.crowding_distance = 0.0;
        }

        // Calculate crowding distance for each objective
        for obj_idx in 0..n_objectives {
            // Sort by objective value
            let mut indices: Vec<usize> = (0..n).collect();
            indices.sort_by(|&i, &j| {
                population[i].objectives[obj_idx]
                    .partial_cmp(&population[j].objectives[obj_idx])
                    .unwrap()
            });

            // Set boundary points to infinite distance
            population[indices[0]].crowding_distance = Float::INFINITY;
            population[indices[n - 1]].crowding_distance = Float::INFINITY;

            // Calculate crowding distance for middle points
            let obj_range = population[indices[n - 1]].objectives[obj_idx]
                - population[indices[0]].objectives[obj_idx];

            if obj_range > 0.0 {
                for i in 1..n - 1 {
                    let distance = (population[indices[i + 1]].objectives[obj_idx]
                        - population[indices[i - 1]].objectives[obj_idx])
                        / obj_range;
                    population[indices[i]].crowding_distance += distance;
                }
            }
        }

        Ok(())
    }

    /// Evolve population through selection, crossover, and mutation
    fn evolve_population(
        &self,
        population: Vec<ParetoSolution>,
        rng: &mut scirs2_core::random::CoreRandom,
    ) -> SklResult<Vec<ParetoSolution>> {
        let mut new_population = Vec::new();

        while new_population.len() < self.config.population_size {
            // Tournament selection
            let parent1 = self.tournament_selection(&population, rng)?;
            let parent2 = self.tournament_selection(&population, rng)?;

            // Crossover
            let (mut child1, mut child2) = self.crossover(&parent1, &parent2, rng)?;

            // Mutation
            self.mutate(&mut child1, rng)?;
            self.mutate(&mut child2, rng)?;

            new_population.push(child1);
            if new_population.len() < self.config.population_size {
                new_population.push(child2);
            }
        }

        Ok(new_population)
    }

    /// Tournament selection
    fn tournament_selection(
        &self,
        population: &[ParetoSolution],
        rng: &mut scirs2_core::random::CoreRandom,
    ) -> SklResult<ParetoSolution> {
        let tournament_size = 3;
        let mut best_solution = None;

        for _ in 0..tournament_size {
            let idx = rng.gen_range(0..population.len());
            let candidate = &population[idx];

            if let Some(ref current_best) = best_solution {
                if self.is_better_solution(candidate, current_best) {
                    best_solution = Some(candidate.clone());
                }
            } else {
                best_solution = Some(candidate.clone());
            }
        }

        best_solution
            .ok_or_else(|| SklearsError::InvalidInput("Tournament selection failed".to_string()))
    }

    /// Check if solution a is better than solution b
    fn is_better_solution(&self, a: &ParetoSolution, b: &ParetoSolution) -> bool {
        if a.rank < b.rank {
            true
        } else if a.rank == b.rank {
            a.crowding_distance > b.crowding_distance
        } else {
            false
        }
    }

    /// Crossover operation
    fn crossover(
        &self,
        parent1: &ParetoSolution,
        parent2: &ParetoSolution,
        rng: &mut scirs2_core::random::CoreRandom,
    ) -> SklResult<(ParetoSolution, ParetoSolution)> {
        let mut child1 = parent1.clone();
        let mut child2 = parent2.clone();

        if rng.gen::<Float>() < self.config.crossover_rate {
            // Uniform crossover
            for i in 0..parent1.parameters.len() {
                if rng.gen::<Float>() < 0.5 {
                    child1.parameters[i] = parent2.parameters[i];
                    child2.parameters[i] = parent1.parameters[i];
                }
            }
        }

        Ok((child1, child2))
    }

    /// Mutation operation
    fn mutate(
        &self,
        solution: &mut ParetoSolution,
        rng: &mut scirs2_core::random::CoreRandom,
    ) -> SklResult<()> {
        for param in solution.parameters.iter_mut() {
            if rng.gen::<Float>() < self.config.mutation_rate {
                let mutation = rng.gen_range(-0.1..0.1);
                *param += mutation;
            }
        }
        Ok(())
    }

    /// Calculate hypervolume (convergence metric)
    fn calculate_hypervolume(&self, population: &[ParetoSolution]) -> SklResult<Float> {
        // Simplified hypervolume calculation
        let pareto_front: Vec<&ParetoSolution> =
            population.iter().filter(|sol| sol.rank == 0).collect();

        if pareto_front.is_empty() {
            return Ok(0.0);
        }

        // Use the sum of objective values as a proxy for hypervolume
        let hypervolume = pareto_front
            .iter()
            .map(|sol| sol.objectives.sum())
            .sum::<Float>()
            / pareto_front.len() as Float;

        Ok(hypervolume)
    }

    /// Find best compromise solution
    fn find_best_compromise(
        &self,
        pareto_solutions: &[ParetoSolution],
    ) -> SklResult<ParetoSolution> {
        if pareto_solutions.is_empty() {
            return Err(SklearsError::InvalidInput(
                "No Pareto solutions available".to_string(),
            ));
        }

        // Find the solution closest to the ideal point (origin)
        let mut best_solution = pareto_solutions[0].clone();
        let mut best_distance = Float::INFINITY;

        for solution in pareto_solutions {
            let distance = solution.objectives.mapv(|x| x * x).sum().sqrt();
            if distance < best_distance {
                best_distance = distance;
                best_solution = solution.clone();
            }
        }

        Ok(best_solution)
    }
}

impl Predict<ArrayView2<'_, Float>, Array2<Float>>
    for MultiObjectiveOptimizer<MultiObjectiveOptimizerTrained>
{
    fn predict(&self, X: &ArrayView2<'_, Float>) -> SklResult<Array2<Float>> {
        let (n_samples, n_features) = X.dim();
        let best_solution = &self.state.best_solution;

        // Extract weights and bias from best solution parameters
        let n_outputs = self.state.n_outputs;
        let weights_size = n_features * n_outputs;
        let weights = best_solution
            .parameters
            .slice(s![..weights_size])
            .to_owned()
            .into_shape((n_features, n_outputs))
            .unwrap();
        let bias = best_solution
            .parameters
            .slice(s![weights_size..weights_size + n_outputs])
            .to_owned();

        let predictions = X.dot(&weights) + &bias;
        Ok(predictions)
    }
}

impl Estimator for MultiObjectiveOptimizer<MultiObjectiveOptimizerTrained> {
    type Config = MultiObjectiveConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.state.config
    }
}

impl MultiObjectiveOptimizer<MultiObjectiveOptimizerTrained> {
    /// Get the Pareto-optimal solutions
    pub fn pareto_solutions(&self) -> &[ParetoSolution] {
        &self.state.pareto_solutions
    }

    /// Get the best compromise solution
    pub fn best_solution(&self) -> &ParetoSolution {
        &self.state.best_solution
    }

    /// Get the convergence history
    pub fn convergence_history(&self) -> &[Float] {
        &self.state.convergence_history
    }
}
