//! Enhanced Evolutionary Multi-Objective Optimization Algorithms
//!
//! This module provides NSGA-II (Non-dominated Sorting Genetic Algorithm II) and related
//! evolutionary algorithms for multi-objective optimization. NSGA-II is one of the most
//! popular and effective multi-objective evolutionary algorithms.
//!
//! ## Key Features
//!
//! - **Non-dominated Sorting**: Fast and efficient ranking of solutions based on Pareto dominance
//! - **Crowding Distance**: Maintains diversity in the population and Pareto front
//! - **Multiple Algorithm Variants**: Standard NSGA-II, SBX crossover, and differential evolution
//! - **Advanced Operators**: Simulated binary crossover (SBX) and polynomial mutation
//! - **Elitism**: Preserves good solutions across generations
//! - **Hypervolume Tracking**: Monitors convergence quality over generations

// Use SciRS2-Core for arrays and random number generation (SciRS2 Policy)
use scirs2_core::ndarray::{array, s, Array1, Array2, ArrayView2};
use scirs2_core::random::RandNormal;
use scirs2_core::random::Rng;
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Predict, Untrained},
    types::Float,
};

use super::multi_objective_optimization::ParetoSolution;

/// NSGA-II (Non-dominated Sorting Genetic Algorithm II) algorithm types
#[derive(Debug, Clone, PartialEq)]
pub enum NSGA2Algorithm {
    /// Standard NSGA-II
    Standard,
    /// NSGA-II with simulated binary crossover
    SBX,
    /// NSGA-II with differential evolution
    DE,
}

/// NSGA-II Configuration
#[derive(Debug, Clone)]
pub struct NSGA2Config {
    /// Population size
    pub population_size: usize,
    /// Number of generations
    pub generations: usize,
    /// Crossover probability
    pub crossover_prob: Float,
    /// Mutation probability
    pub mutation_prob: Float,
    /// Distribution index for SBX crossover
    pub eta_c: Float,
    /// Distribution index for polynomial mutation
    pub eta_m: Float,
    /// Algorithm variant
    pub algorithm: NSGA2Algorithm,
    /// Random state for reproducibility
    pub random_state: Option<u64>,
}

impl Default for NSGA2Config {
    fn default() -> Self {
        Self {
            population_size: 100,
            generations: 250,
            crossover_prob: 0.9,
            mutation_prob: 0.1,
            eta_c: 20.0,
            eta_m: 20.0,
            algorithm: NSGA2Algorithm::Standard,
            random_state: None,
        }
    }
}

/// NSGA-II Multi-Objective Optimizer
#[derive(Debug, Clone)]
pub struct NSGA2Optimizer<S = Untrained> {
    state: S,
    config: NSGA2Config,
}

/// Trained state for NSGA-II Optimizer
#[derive(Debug, Clone)]
pub struct NSGA2OptimizerTrained {
    /// Pareto-optimal solutions
    pub pareto_solutions: Vec<ParetoSolution>,
    /// Best compromise solution
    pub best_solution: ParetoSolution,
    /// Convergence history (hypervolume indicator)
    pub convergence_history: Vec<Float>,
    /// Final population
    pub final_population: Vec<ParetoSolution>,
    /// Configuration used for optimization
    pub config: NSGA2Config,
    /// Number of objectives
    pub n_objectives: usize,
}

impl Default for NSGA2Optimizer<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl NSGA2Optimizer<Untrained> {
    /// Create a new NSGA-II Optimizer
    pub fn new() -> Self {
        Self {
            state: Untrained,
            config: NSGA2Config::default(),
        }
    }

    /// Set the configuration
    pub fn config(mut self, config: NSGA2Config) -> Self {
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

    /// Set the crossover probability
    pub fn crossover_prob(mut self, crossover_prob: Float) -> Self {
        self.config.crossover_prob = crossover_prob;
        self
    }

    /// Set the mutation probability
    pub fn mutation_prob(mut self, mutation_prob: Float) -> Self {
        self.config.mutation_prob = mutation_prob;
        self
    }

    /// Set the algorithm variant
    pub fn algorithm(mut self, algorithm: NSGA2Algorithm) -> Self {
        self.config.algorithm = algorithm;
        self
    }
}

impl Estimator for NSGA2Optimizer<Untrained> {
    type Config = NSGA2Config;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Estimator for NSGA2Optimizer<NSGA2OptimizerTrained> {
    type Config = NSGA2Config;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.state.config
    }
}

impl Fit<ArrayView2<'_, Float>, ArrayView2<'_, Float>> for NSGA2Optimizer<Untrained> {
    type Fitted = NSGA2Optimizer<NSGA2OptimizerTrained>;

    fn fit(self, X: &ArrayView2<'_, Float>, y: &ArrayView2<'_, Float>) -> SklResult<Self::Fitted> {
        if X.nrows() != y.nrows() {
            return Err(SklearsError::InvalidInput(
                "X and y must have the same number of samples".to_string(),
            ));
        }

        let n_samples = X.nrows();
        let n_features = X.ncols();
        let n_outputs = y.ncols();
        let n_objectives = 2; // Default: accuracy and complexity

        let mut rng = match self.config.random_state {
            Some(seed) => scirs2_core::random::seeded_rng(seed),
            None => scirs2_core::random::seeded_rng(42), // Default seed
        };

        // Initialize population
        let mut population = self.initialize_population(n_features, n_outputs, &mut rng)?;

        // Evaluate initial population
        self.evaluate_population(&mut population, X, y)?;

        let mut convergence_history = Vec::new();

        // Main evolutionary loop
        for generation in 0..self.config.generations {
            // Non-dominated sorting
            self.nsga2_non_dominated_sort(&mut population)?;

            // Calculate crowding distance
            self.nsga2_crowding_distance(&mut population)?;

            // Calculate hypervolume for convergence tracking
            let hypervolume = self.calculate_hypervolume(&population)?;
            convergence_history.push(hypervolume);

            // Generate offspring population
            let mut offspring = self.nsga2_generate_offspring(&population, &mut rng)?;

            // Evaluate offspring
            self.evaluate_population(&mut offspring, X, y)?;

            // Combine parent and offspring populations
            population.extend(offspring);

            // Environmental selection
            population = self.nsga2_environmental_selection(population)?;

            if generation % 50 == 0 {
                println!(
                    "Generation {}: Hypervolume = {:.6}",
                    generation, hypervolume
                );
            }
        }

        // Final evaluation
        self.nsga2_non_dominated_sort(&mut population)?;
        let pareto_solutions = self.extract_pareto_front(&population)?;
        let best_solution = self.find_best_compromise(&pareto_solutions)?;

        Ok(NSGA2Optimizer {
            state: NSGA2OptimizerTrained {
                pareto_solutions: pareto_solutions.clone(),
                best_solution,
                convergence_history,
                final_population: population,
                config: self.config.clone(),
                n_objectives,
            },
            config: self.config,
        })
    }
}

impl NSGA2Optimizer<Untrained> {
    /// Initialize population for NSGA-II
    fn initialize_population<R: Rng>(
        &self,
        n_features: usize,
        n_outputs: usize,
        rng: &mut R,
    ) -> SklResult<Vec<ParetoSolution>> {
        let mut population = Vec::with_capacity(self.config.population_size);
        let param_size = n_features * n_outputs + n_outputs; // weights + bias

        for _ in 0..self.config.population_size {
            let parameters = Array1::from_shape_fn(param_size, |_| rng.gen_range(-1.0..1.0));
            let solution = ParetoSolution {
                parameters,
                objectives: Array1::zeros(2), // Will be filled during evaluation
                rank: 0,
                crowding_distance: 0.0,
            };
            population.push(solution);
        }

        Ok(population)
    }

    /// NSGA-II Non-dominated sorting
    fn nsga2_non_dominated_sort(&self, population: &mut Vec<ParetoSolution>) -> SklResult<()> {
        let n = population.len();
        let mut domination_counts = vec![0; n];
        let mut dominated_solutions = vec![Vec::new(); n];
        let mut fronts: Vec<Vec<usize>> = Vec::new();

        // Calculate domination relationships
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    if self.nsga2_dominates(&population[i], &population[j]) {
                        dominated_solutions[i].push(j);
                    } else if self.nsga2_dominates(&population[j], &population[i]) {
                        domination_counts[i] += 1;
                    }
                }
            }
        }

        // First front
        let mut current_front: Vec<usize> = (0..n).filter(|&i| domination_counts[i] == 0).collect();
        let mut rank = 0;

        while !current_front.is_empty() {
            // Assign rank to current front
            for &i in &current_front {
                population[i].rank = rank;
            }

            fronts.push(current_front.clone());

            // Generate next front
            let mut next_front = Vec::new();
            for &i in &current_front {
                for &j in &dominated_solutions[i] {
                    domination_counts[j] -= 1;
                    if domination_counts[j] == 0 {
                        next_front.push(j);
                    }
                }
            }

            current_front = next_front;
            rank += 1;
        }

        Ok(())
    }

    /// Check if solution a dominates solution b for NSGA-II
    pub fn nsga2_dominates(&self, a: &ParetoSolution, b: &ParetoSolution) -> bool {
        let mut at_least_one_better = false;

        for i in 0..a.objectives.len() {
            if a.objectives[i] > b.objectives[i] {
                return false; // b is better in at least one objective
            }
            if a.objectives[i] < b.objectives[i] {
                at_least_one_better = true;
            }
        }

        at_least_one_better
    }

    /// Calculate crowding distance for NSGA-II
    fn nsga2_crowding_distance(&self, population: &mut Vec<ParetoSolution>) -> SklResult<()> {
        let n = population.len();
        if n == 0 {
            return Ok(());
        }

        // Initialize crowding distances
        for solution in population.iter_mut() {
            solution.crowding_distance = 0.0;
        }

        let n_objectives = population[0].objectives.len();

        for obj_idx in 0..n_objectives {
            // Sort by objective value
            let mut indices: Vec<usize> = (0..n).collect();
            indices.sort_by(|&a, &b| {
                population[a].objectives[obj_idx]
                    .partial_cmp(&population[b].objectives[obj_idx])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            // Set boundary points to infinite distance
            population[indices[0]].crowding_distance = Float::INFINITY;
            population[indices[n - 1]].crowding_distance = Float::INFINITY;

            // Calculate distances for intermediate points
            let obj_range = population[indices[n - 1]].objectives[obj_idx]
                - population[indices[0]].objectives[obj_idx];

            if obj_range > 0.0 {
                for i in 1..(n - 1) {
                    let distance = (population[indices[i + 1]].objectives[obj_idx]
                        - population[indices[i - 1]].objectives[obj_idx])
                        / obj_range;
                    population[indices[i]].crowding_distance += distance;
                }
            }
        }

        Ok(())
    }

    /// Generate offspring population using NSGA-II
    fn nsga2_generate_offspring<R: Rng>(
        &self,
        population: &[ParetoSolution],
        rng: &mut R,
    ) -> SklResult<Vec<ParetoSolution>> {
        let mut offspring = Vec::new();

        for _ in 0..self.config.population_size {
            // Binary tournament selection
            let parent1 = self.nsga2_tournament_selection(population, rng)?;
            let parent2 = self.nsga2_tournament_selection(population, rng)?;

            // Crossover
            let mut child = match self.config.algorithm {
                NSGA2Algorithm::SBX => self.simulated_binary_crossover(&parent1, &parent2, rng)?,
                _ => self.uniform_crossover(&parent1, &parent2, rng)?,
            };

            // Mutation
            match self.config.algorithm {
                NSGA2Algorithm::SBX => self.polynomial_mutation(&mut child, rng)?,
                _ => self.gaussian_mutation(&mut child, rng)?,
            }

            offspring.push(child);
        }

        Ok(offspring)
    }

    /// Binary tournament selection for NSGA-II
    fn nsga2_tournament_selection<R: Rng>(
        &self,
        population: &[ParetoSolution],
        rng: &mut R,
    ) -> SklResult<ParetoSolution> {
        let idx1 = rng.gen_range(0..population.len());
        let idx2 = rng.gen_range(0..population.len());

        let solution1 = &population[idx1];
        let solution2 = &population[idx2];

        // Compare by rank first, then by crowding distance
        if solution1.rank < solution2.rank {
            Ok(solution1.clone())
        } else if solution1.rank > solution2.rank {
            Ok(solution2.clone())
        } else {
            // Same rank, compare by crowding distance (higher is better)
            if solution1.crowding_distance > solution2.crowding_distance {
                Ok(solution1.clone())
            } else {
                Ok(solution2.clone())
            }
        }
    }

    /// Simulated Binary Crossover (SBX)
    fn simulated_binary_crossover<R: Rng>(
        &self,
        parent1: &ParetoSolution,
        parent2: &ParetoSolution,
        rng: &mut R,
    ) -> SklResult<ParetoSolution> {
        let mut child_params = parent1.parameters.clone();

        if rng.gen::<Float>() <= self.config.crossover_prob {
            for i in 0..child_params.len() {
                let p1 = parent1.parameters[i];
                let p2 = parent2.parameters[i];

                if rng.gen::<Float>() <= 0.5 {
                    let u = rng.gen::<Float>();
                    let beta = if u <= 0.5 {
                        (2.0 * u).powf(1.0 / (self.config.eta_c + 1.0))
                    } else {
                        (1.0 / (2.0 * (1.0 - u))).powf(1.0 / (self.config.eta_c + 1.0))
                    };

                    let child_val = 0.5 * ((1.0 + beta) * p1 + (1.0 - beta) * p2);
                    child_params[i] = child_val.clamp(-2.0, 2.0);
                }
            }
        }

        Ok(ParetoSolution {
            parameters: child_params,
            objectives: Array1::zeros(parent1.objectives.len()),
            rank: 0,
            crowding_distance: 0.0,
        })
    }

    /// Polynomial mutation
    fn polynomial_mutation<R: Rng>(
        &self,
        solution: &mut ParetoSolution,
        rng: &mut R,
    ) -> SklResult<()> {
        for i in 0..solution.parameters.len() {
            if rng.gen::<Float>() <= self.config.mutation_prob {
                let u = rng.gen::<Float>();
                let delta = if u < 0.5 {
                    (2.0 * u).powf(1.0 / (self.config.eta_m + 1.0)) - 1.0
                } else {
                    1.0 - (2.0 * (1.0 - u)).powf(1.0 / (self.config.eta_m + 1.0))
                };

                solution.parameters[i] += delta * 0.1;
                solution.parameters[i] = solution.parameters[i].clamp(-2.0, 2.0);
            }
        }
        Ok(())
    }

    /// Environmental selection for NSGA-II
    fn nsga2_environmental_selection(
        &self,
        mut population: Vec<ParetoSolution>,
    ) -> SklResult<Vec<ParetoSolution>> {
        // Sort by rank and crowding distance
        self.nsga2_non_dominated_sort(&mut population)?;
        self.nsga2_crowding_distance(&mut population)?;

        // Sort population by rank, then by crowding distance
        population.sort_by(|a, b| {
            match a.rank.cmp(&b.rank) {
                std::cmp::Ordering::Equal => {
                    // Higher crowding distance is better
                    b.crowding_distance
                        .partial_cmp(&a.crowding_distance)
                        .unwrap_or(std::cmp::Ordering::Equal)
                }
                other => other,
            }
        });

        // Take the best individuals up to population size
        population.truncate(self.config.population_size);
        Ok(population)
    }

    /// Extract Pareto front (rank 0 solutions)
    fn extract_pareto_front(
        &self,
        population: &[ParetoSolution],
    ) -> SklResult<Vec<ParetoSolution>> {
        Ok(population
            .iter()
            .filter(|sol| sol.rank == 0)
            .cloned()
            .collect())
    }

    /// Evaluate population fitness
    fn evaluate_population(
        &self,
        population: &mut [ParetoSolution],
        X: &ArrayView2<Float>,
        y: &ArrayView2<Float>,
    ) -> SklResult<()> {
        let n_features = X.ncols();
        let n_outputs = y.ncols();

        for solution in population.iter_mut() {
            // Extract weights and bias from parameters
            let weights = solution
                .parameters
                .slice(s![..n_features * n_outputs])
                .to_owned()
                .into_shape((n_features, n_outputs))
                .map_err(|e| SklearsError::InvalidInput(format!("Shape error: {}", e)))?;

            let bias = solution
                .parameters
                .slice(s![n_features * n_outputs..])
                .to_owned();

            // Make predictions
            let mut predictions = X.dot(&weights);
            for mut row in predictions.rows_mut() {
                row += &bias;
            }

            // Calculate objectives
            let mse = self.calculate_mse(&predictions.view(), y)?;
            let complexity = self.calculate_complexity(&weights)?;

            solution.objectives = array![mse, complexity];
        }

        Ok(())
    }

    /// Calculate Mean Squared Error
    fn calculate_mse(
        &self,
        predictions: &ArrayView2<Float>,
        y: &ArrayView2<Float>,
    ) -> SklResult<Float> {
        let diff = predictions - y;
        let squared_diff = &diff * &diff;
        Ok(squared_diff.sum() / (predictions.nrows() * predictions.ncols()) as Float)
    }

    /// Calculate model complexity (based on parameter magnitudes)
    fn calculate_complexity(&self, weights: &Array2<Float>) -> SklResult<Float> {
        Ok(weights.mapv(|x| x.abs()).sum())
    }

    /// Uniform crossover operation
    fn uniform_crossover<R: Rng>(
        &self,
        parent1: &ParetoSolution,
        parent2: &ParetoSolution,
        rng: &mut R,
    ) -> SklResult<ParetoSolution> {
        let mut child_params = parent1.parameters.clone();

        if rng.gen::<Float>() <= self.config.crossover_prob {
            for i in 0..child_params.len() {
                if rng.gen::<Float>() <= 0.5 {
                    child_params[i] = parent2.parameters[i];
                }
            }
        }

        Ok(ParetoSolution {
            parameters: child_params,
            objectives: Array1::zeros(parent1.objectives.len()),
            rank: 0,
            crowding_distance: 0.0,
        })
    }

    /// Gaussian mutation operation
    fn gaussian_mutation<R: Rng>(
        &self,
        solution: &mut ParetoSolution,
        rng: &mut R,
    ) -> SklResult<()> {
        for i in 0..solution.parameters.len() {
            if rng.gen::<Float>() <= self.config.mutation_prob {
                let normal = RandNormal::new(0.0, 0.1).map_err(|e| {
                    SklearsError::InvalidInput(format!(
                        "Failed to create normal distribution: {}",
                        e
                    ))
                })?;
                let mutation = rng.sample(normal);
                solution.parameters[i] += mutation;
                solution.parameters[i] = solution.parameters[i].clamp(-2.0, 2.0);
            }
        }
        Ok(())
    }

    /// Calculate hypervolume indicator
    fn calculate_hypervolume(&self, population: &[ParetoSolution]) -> SklResult<Float> {
        // Extract non-dominated solutions (Pareto front)
        let pareto_front: Vec<&ParetoSolution> =
            population.iter().filter(|sol| sol.rank == 0).collect();

        if pareto_front.is_empty() {
            return Ok(0.0);
        }

        // Simple hypervolume calculation using reference point (1.0, 1.0)
        let reference_point = array![1.0, 1.0];
        let mut hypervolume = 0.0;

        for solution in &pareto_front {
            let mut volume = 1.0;
            for i in 0..solution.objectives.len() {
                let contribution = (reference_point[i] - solution.objectives[i]).max(0.0);
                volume *= contribution;
            }
            hypervolume += volume;
        }

        Ok(hypervolume / pareto_front.len() as Float)
    }

    /// Find best compromise solution from Pareto solutions
    fn find_best_compromise(
        &self,
        pareto_solutions: &[ParetoSolution],
    ) -> SklResult<ParetoSolution> {
        if pareto_solutions.is_empty() {
            return Err(SklearsError::InvalidInput(
                "No Pareto solutions available".to_string(),
            ));
        }

        let mut best_solution = pareto_solutions[0].clone();
        let mut best_distance = Float::INFINITY;

        // Find solution closest to ideal point (0, 0)
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

impl Predict<ArrayView2<'_, Float>, Array2<Float>> for NSGA2Optimizer<NSGA2OptimizerTrained> {
    fn predict(&self, X: &ArrayView2<'_, Float>) -> SklResult<Array2<Float>> {
        let n_samples = X.nrows();
        let n_features = X.ncols();
        let n_outputs = self.state.best_solution.parameters.len() / (n_features + 1);

        // Extract weights and bias from best solution
        let weights = self
            .state
            .best_solution
            .parameters
            .slice(s![..n_features * n_outputs])
            .to_owned()
            .into_shape((n_features, n_outputs))
            .map_err(|e| SklearsError::InvalidInput(format!("Shape error: {}", e)))?;

        let bias = self
            .state
            .best_solution
            .parameters
            .slice(s![n_features * n_outputs..])
            .to_owned();

        // Make predictions: y = X * W + b
        let mut predictions = X.dot(&weights);
        for mut row in predictions.rows_mut() {
            row += &bias;
        }

        Ok(predictions)
    }
}

impl NSGA2Optimizer<NSGA2OptimizerTrained> {
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

    /// Get the final population
    pub fn final_population(&self) -> &[ParetoSolution] {
        &self.state.final_population
    }
}
