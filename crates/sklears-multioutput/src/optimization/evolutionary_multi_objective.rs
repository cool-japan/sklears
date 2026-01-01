//! Evolutionary Multi-Objective Optimization Algorithms
//!
//! This module implements advanced evolutionary algorithms for multi-objective optimization
//! problems, particularly suited for multi-output learning scenarios where multiple
//! conflicting objectives need to be optimized simultaneously.

// Use SciRS2-Core for arrays and random number generation (SciRS2 Policy)
use scirs2_core::ndarray::{Array1, Array2, ArrayView1};
use scirs2_core::random::Rng;
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    types::Float,
};
use std::cmp::Ordering;

/// Individual in the evolutionary algorithm population
#[derive(Debug, Clone)]
pub struct Individual {
    /// Decision variables (parameters to optimize)
    pub variables: Array1<Float>,
    /// Objective function values for this individual
    pub objectives: Array1<Float>,
    /// Non-dominated rank (lower is better)
    pub rank: usize,
    /// Crowding distance (higher is better for diversity)
    pub crowding_distance: Float,
}

impl Individual {
    /// Create a new individual with given variables
    pub fn new(variables: Array1<Float>) -> Self {
        Self {
            variables,
            objectives: Array1::zeros(0), // Will be evaluated later
            rank: 0,
            crowding_distance: 0.0,
        }
    }

    /// Check if this individual dominates another
    pub fn dominates(&self, other: &Individual) -> bool {
        let mut at_least_one_better = false;

        for i in 0..self.objectives.len() {
            if self.objectives[i] > other.objectives[i] {
                return false; // This individual is worse in at least one objective
            }
            if self.objectives[i] < other.objectives[i] {
                at_least_one_better = true;
            }
        }

        at_least_one_better
    }
}

/// NSGA-II (Non-dominated Sorting Genetic Algorithm II) implementation
///
/// A state-of-the-art evolutionary algorithm for multi-objective optimization
/// that maintains a diverse set of Pareto-optimal solutions.
///
/// # Examples
///
/// ```rust,ignore
/// use sklears_multioutput::optimization::evolutionary_multi_objective::NSGAII;
/// // Use SciRS2-Core for arrays and random number generation (SciRS2 Policy)
/// use scirs2_core::ndarray::array;
///
/// // Define a multi-objective function (minimize both objectives)
/// let objectives = |x: &scirs2_autograd::scirs2_core::ndarray::ArrayView1`<f64>`| {
///     let obj1 = x[0].powi(2) + x[1].powi(2);           // Minimize distance from origin
///     let obj2 = (x[0] - 1.0).powi(2) + (x[1] - 1.0).powi(2); // Minimize distance from (1,1)
///     array![obj1, obj2]
/// };
///
/// let nsga2 = NSGAII::new()
///     .population_size(100)
///     .n_generations(50)
///     .crossover_probability(0.9)
///     .mutation_probability(0.1)
///     .variable_bounds(vec![(-2.0, 2.0), (-2.0, 2.0)])
///     .random_state(42);
///
/// let result = nsga2.optimize(objectives, 2).unwrap();
/// assert!(result.pareto_front().len() > 0);
/// ```
#[derive(Debug, Clone)]
pub struct NSGAII {
    /// Size of the population
    population_size: usize,
    /// Number of generations to evolve
    n_generations: usize,
    /// Probability of crossover
    crossover_probability: Float,
    /// Probability of mutation
    mutation_probability: Float,
    /// Bounds for each decision variable (min, max)
    variable_bounds: Vec<(Float, Float)>,
    /// Random state for reproducibility
    random_state: Option<u64>,
    /// Crossover distribution index
    crossover_eta: Float,
    /// Mutation distribution index
    mutation_eta: Float,
}

impl Default for NSGAII {
    fn default() -> Self {
        Self {
            population_size: 100,
            n_generations: 100,
            crossover_probability: 0.9,
            mutation_probability: 0.1,
            variable_bounds: Vec::new(),
            random_state: None,
            crossover_eta: 20.0,
            mutation_eta: 20.0,
        }
    }
}

impl NSGAII {
    /// Create a new NSGA-II optimizer with default parameters
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the population size
    pub fn population_size(mut self, size: usize) -> Self {
        self.population_size = size;
        self
    }

    /// Set the number of generations
    pub fn n_generations(mut self, generations: usize) -> Self {
        self.n_generations = generations;
        self
    }

    /// Set the crossover probability
    pub fn crossover_probability(mut self, prob: Float) -> Self {
        self.crossover_probability = prob;
        self
    }

    /// Set the mutation probability
    pub fn mutation_probability(mut self, prob: Float) -> Self {
        self.mutation_probability = prob;
        self
    }

    /// Set bounds for decision variables
    pub fn variable_bounds(mut self, bounds: Vec<(Float, Float)>) -> Self {
        self.variable_bounds = bounds;
        self
    }

    /// Set random state for reproducibility
    pub fn random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Set crossover distribution index (higher values = more uniform crossover)
    pub fn crossover_eta(mut self, eta: Float) -> Self {
        self.crossover_eta = eta;
        self
    }

    /// Set mutation distribution index (higher values = smaller mutations)
    pub fn mutation_eta(mut self, eta: Float) -> Self {
        self.mutation_eta = eta;
        self
    }

    /// Optimize a multi-objective function using NSGA-II
    pub fn optimize<F>(&self, objective_fn: F, n_objectives: usize) -> SklResult<OptimizationResult>
    where
        F: Fn(&ArrayView1<Float>) -> Array1<Float>,
    {
        if self.variable_bounds.is_empty() {
            return Err(SklearsError::InvalidInput(
                "Variable bounds must be specified".to_string(),
            ));
        }

        let n_variables = self.variable_bounds.len();
        let mut rng = if let Some(seed) = self.random_state {
            scirs2_core::random::seeded_rng(seed)
        } else {
            scirs2_core::random::seeded_rng(42)
        };

        // Initialize population
        let mut population = self.initialize_population(&mut rng, n_variables)?;

        // Evaluate initial population
        for individual in &mut population {
            individual.objectives = objective_fn(&individual.variables.view());
        }

        let mut generation_stats = Vec::new();

        // Evolution loop
        for generation in 0..self.n_generations {
            // Create offspring through crossover and mutation
            let offspring = self.create_offspring(&population, &mut rng, n_variables)?;

            // Combine parent and offspring populations
            let mut combined_population = population;
            combined_population.extend(offspring);

            // Evaluate new individuals
            for individual in &mut combined_population {
                if individual.objectives.len() != n_objectives {
                    individual.objectives = objective_fn(&individual.variables.view());
                }
            }

            // Perform non-dominated sorting
            let fronts = self.non_dominated_sort(&combined_population);

            // Select next generation using NSGA-II selection
            population = self.environmental_selection(combined_population, fronts)?;

            // Record generation statistics
            let stats = self.calculate_generation_stats(&population);
            generation_stats.push(stats);

            // Optional: Early stopping based on convergence criteria could be added here
        }

        // Extract final Pareto front
        let fronts = self.non_dominated_sort(&population);
        let pareto_front = if fronts.is_empty() {
            Vec::new()
        } else {
            fronts[0].clone()
        };

        Ok(OptimizationResult {
            pareto_front,
            final_population: population,
            generation_stats,
            n_generations: self.n_generations,
        })
    }

    /// Initialize a random population
    fn initialize_population<R: Rng>(
        &self,
        rng: &mut scirs2_core::random::CoreRandom<R>,
        n_variables: usize,
    ) -> SklResult<Vec<Individual>> {
        let mut population = Vec::with_capacity(self.population_size);

        for _ in 0..self.population_size {
            let mut variables = Array1::zeros(n_variables);

            for j in 0..n_variables {
                let (min_val, max_val) = self.variable_bounds[j];
                variables[j] = rng.gen_range(min_val..max_val + 1.0);
            }

            population.push(Individual::new(variables));
        }

        Ok(population)
    }

    /// Create offspring through crossover and mutation
    fn create_offspring<R: Rng>(
        &self,
        population: &[Individual],
        rng: &mut scirs2_core::random::CoreRandom<R>,
        n_variables: usize,
    ) -> SklResult<Vec<Individual>> {
        let mut offspring = Vec::with_capacity(self.population_size);

        for _ in 0..self.population_size {
            // Tournament selection for parents
            let parent1 = self.tournament_selection(population, rng);
            let parent2 = self.tournament_selection(population, rng);

            // Crossover
            let mut child = if rng.gen::<Float>() < self.crossover_probability {
                self.sbx_crossover(parent1, parent2, rng)?
            } else {
                parent1.clone()
            };

            // Mutation
            if rng.gen::<Float>() < self.mutation_probability {
                self.polynomial_mutation(&mut child, rng);
            }

            offspring.push(child);
        }

        Ok(offspring)
    }

    /// Tournament selection
    fn tournament_selection<'a, R: Rng>(
        &self,
        population: &'a [Individual],
        rng: &mut scirs2_core::random::CoreRandom<R>,
    ) -> &'a Individual {
        let tournament_size = 2;
        let mut best = &population[rng.gen_range(0..population.len())];

        for _ in 1..tournament_size {
            let candidate = &population[rng.gen_range(0..population.len())];
            if self.compare_individuals(candidate, best) == Ordering::Less {
                best = candidate;
            }
        }

        best
    }

    /// Compare two individuals using NSGA-II criteria
    fn compare_individuals(&self, a: &Individual, b: &Individual) -> Ordering {
        // First, compare by rank (lower is better)
        match a.rank.cmp(&b.rank) {
            Ordering::Equal => {
                // If ranks are equal, compare by crowding distance (higher is better)
                b.crowding_distance
                    .partial_cmp(&a.crowding_distance)
                    .unwrap_or(Ordering::Equal)
            }
            other => other,
        }
    }

    /// Simulated Binary Crossover (SBX)
    fn sbx_crossover<R: Rng>(
        &self,
        parent1: &Individual,
        parent2: &Individual,
        rng: &mut scirs2_core::random::CoreRandom<R>,
    ) -> SklResult<Individual> {
        let n_variables = parent1.variables.len();
        let mut child_variables = Array1::zeros(n_variables);

        for i in 0..n_variables {
            let p1 = parent1.variables[i];
            let p2 = parent2.variables[i];
            let (min_val, max_val) = self.variable_bounds[i];

            if (p1 - p2).abs() > 1e-14 {
                let u = rng.gen::<Float>();
                let beta = if u <= 0.5 {
                    (2.0 * u).powf(1.0 / (self.crossover_eta + 1.0))
                } else {
                    (1.0 / (2.0 * (1.0 - u))).powf(1.0 / (self.crossover_eta + 1.0))
                };

                let c1 = 0.5 * (p1 + p2 - beta * (p1 - p2).abs());
                child_variables[i] = c1.clamp(min_val, max_val);
            } else {
                child_variables[i] = p1;
            }
        }

        Ok(Individual::new(child_variables))
    }

    /// Polynomial mutation
    fn polynomial_mutation<R: Rng>(
        &self,
        individual: &mut Individual,
        rng: &mut scirs2_core::random::CoreRandom<R>,
    ) {
        for i in 0..individual.variables.len() {
            if rng.gen::<Float>() < (1.0 / individual.variables.len() as Float) {
                let (min_val, max_val) = self.variable_bounds[i];
                let x = individual.variables[i];
                let u = rng.gen::<Float>();

                let delta = if u <= 0.5 {
                    let bl = (x - min_val) / (max_val - min_val);
                    let b = 2.0 * u + (1.0 - 2.0 * u) * (1.0 - bl).powf(self.mutation_eta + 1.0);
                    b.powf(1.0 / (self.mutation_eta + 1.0)) - 1.0
                } else {
                    let bu = (max_val - x) / (max_val - min_val);
                    let b = 2.0 * (1.0 - u)
                        + 2.0 * (u - 0.5) * (1.0 - bu).powf(self.mutation_eta + 1.0);
                    1.0 - b.powf(1.0 / (self.mutation_eta + 1.0))
                };

                individual.variables[i] = (x + delta * (max_val - min_val)).clamp(min_val, max_val);
            }
        }
    }

    /// Perform non-dominated sorting
    fn non_dominated_sort(&self, population: &[Individual]) -> Vec<Vec<Individual>> {
        let mut fronts = Vec::new();
        let mut domination_count = vec![0; population.len()];
        let mut dominated_solutions = vec![Vec::new(); population.len()];

        // Calculate domination relationships
        for i in 0..population.len() {
            for j in 0..population.len() {
                if population[i].dominates(&population[j]) {
                    dominated_solutions[i].push(j);
                } else if population[j].dominates(&population[i]) {
                    domination_count[i] += 1;
                }
            }
        }

        // Find first front
        let mut current_front = Vec::new();
        for i in 0..population.len() {
            if domination_count[i] == 0 {
                current_front.push(population[i].clone());
            }
        }

        let mut front_index = 0;
        while !current_front.is_empty() {
            fronts.push(current_front.clone());
            let mut next_front = Vec::new();

            for individual in &current_front {
                // Find the index of this individual in the original population
                if let Some(ind_idx) = population.iter().position(|p| {
                    p.variables
                        .iter()
                        .zip(individual.variables.iter())
                        .all(|(a, b)| (a - b).abs() < 1e-10)
                }) {
                    for &dominated_idx in &dominated_solutions[ind_idx] {
                        domination_count[dominated_idx] -= 1;
                        if domination_count[dominated_idx] == 0 {
                            next_front.push(population[dominated_idx].clone());
                        }
                    }
                }
            }

            current_front = next_front;
            front_index += 1;
        }

        fronts
    }

    /// Environmental selection using NSGA-II
    fn environmental_selection(
        &self,
        population: Vec<Individual>,
        fronts: Vec<Vec<Individual>>,
    ) -> SklResult<Vec<Individual>> {
        let mut selected = Vec::new();

        for front in &fronts {
            if selected.len() + front.len() <= self.population_size {
                // Add entire front
                selected.extend(front.clone());
            } else {
                // Partially add front based on crowding distance
                let remaining = self.population_size - selected.len();
                let mut front_with_distance = front.clone();

                // Calculate crowding distances for this front
                self.calculate_crowding_distance(&mut front_with_distance);

                // Sort by crowding distance (descending)
                front_with_distance.sort_by(|a, b| {
                    b.crowding_distance
                        .partial_cmp(&a.crowding_distance)
                        .unwrap_or(Ordering::Equal)
                });

                selected.extend(front_with_distance.into_iter().take(remaining));
                break;
            }
        }

        // Assign ranks to selected individuals
        for (rank, front) in fronts.iter().enumerate() {
            for individual in &mut selected {
                if front.iter().any(|f| {
                    f.variables
                        .iter()
                        .zip(individual.variables.iter())
                        .all(|(a, b)| (a - b).abs() < 1e-10)
                }) {
                    individual.rank = rank;
                }
            }
        }

        Ok(selected)
    }

    /// Calculate crowding distance for a front
    fn calculate_crowding_distance(&self, front: &mut [Individual]) {
        if front.len() <= 2 {
            // Set infinite crowding distance for boundary solutions
            for individual in front {
                individual.crowding_distance = Float::INFINITY;
            }
            return;
        }

        let n_objectives = front[0].objectives.len();

        // Initialize crowding distances
        for individual in front.iter_mut() {
            individual.crowding_distance = 0.0;
        }

        // Calculate for each objective
        for obj_idx in 0..n_objectives {
            // Sort by objective value
            front.sort_by(|a, b| {
                a.objectives[obj_idx]
                    .partial_cmp(&b.objectives[obj_idx])
                    .unwrap_or(Ordering::Equal)
            });

            // Set boundary points to infinity
            front[0].crowding_distance = Float::INFINITY;
            front[front.len() - 1].crowding_distance = Float::INFINITY;

            // Calculate crowding distance for intermediate points
            let max_obj = front[front.len() - 1].objectives[obj_idx];
            let min_obj = front[0].objectives[obj_idx];

            if max_obj - min_obj > 0.0 {
                for i in 1..front.len() - 1 {
                    if front[i].crowding_distance != Float::INFINITY {
                        front[i].crowding_distance += (front[i + 1].objectives[obj_idx]
                            - front[i - 1].objectives[obj_idx])
                            / (max_obj - min_obj);
                    }
                }
            }
        }
    }

    /// Calculate generation statistics
    fn calculate_generation_stats(&self, population: &[Individual]) -> GenerationStats {
        let mut hypervolume = 0.0;
        let spacing = 0.0;
        let pareto_front_size = population.iter().filter(|ind| ind.rank == 0).count();

        // Simple hypervolume calculation (could be improved)
        let pareto_individuals: Vec<_> = population.iter().filter(|ind| ind.rank == 0).collect();
        if !pareto_individuals.is_empty() {
            let n_objectives = pareto_individuals[0].objectives.len();
            let reference_point = Array1::from_elem(n_objectives, 10.0); // Simple reference point

            for individual in &pareto_individuals {
                let mut volume = 1.0;
                for obj_idx in 0..n_objectives {
                    volume *= (reference_point[obj_idx] - individual.objectives[obj_idx]).max(0.0);
                }
                hypervolume += volume;
            }
        }

        GenerationStats {
            hypervolume,
            spacing,
            pareto_front_size,
        }
    }
}

/// Results from evolutionary multi-objective optimization
#[derive(Debug, Clone)]
pub struct OptimizationResult {
    /// Final Pareto front (non-dominated solutions)
    pub pareto_front: Vec<Individual>,
    /// Final population
    pub final_population: Vec<Individual>,
    /// Statistics for each generation
    pub generation_stats: Vec<GenerationStats>,
    /// Number of generations run
    pub n_generations: usize,
}

impl OptimizationResult {
    /// Get the Pareto front solutions
    pub fn pareto_front(&self) -> &[Individual] {
        &self.pareto_front
    }

    /// Get the final population
    pub fn final_population(&self) -> &[Individual] {
        &self.final_population
    }

    /// Get statistics for each generation
    pub fn generation_stats(&self) -> &[GenerationStats] {
        &self.generation_stats
    }

    /// Extract objective values from the Pareto front
    pub fn pareto_objectives(&self) -> Array2<Float> {
        if self.pareto_front.is_empty() {
            return Array2::zeros((0, 0));
        }

        let n_objectives = self.pareto_front[0].objectives.len();
        let mut objectives = Array2::zeros((self.pareto_front.len(), n_objectives));

        for (i, individual) in self.pareto_front.iter().enumerate() {
            for (j, &obj_val) in individual.objectives.iter().enumerate() {
                objectives[[i, j]] = obj_val;
            }
        }

        objectives
    }

    /// Extract decision variables from the Pareto front
    pub fn pareto_variables(&self) -> Array2<Float> {
        if self.pareto_front.is_empty() {
            return Array2::zeros((0, 0));
        }

        let n_variables = self.pareto_front[0].variables.len();
        let mut variables = Array2::zeros((self.pareto_front.len(), n_variables));

        for (i, individual) in self.pareto_front.iter().enumerate() {
            for (j, &var_val) in individual.variables.iter().enumerate() {
                variables[[i, j]] = var_val;
            }
        }

        variables
    }
}

/// Statistics for a generation of evolution
#[derive(Debug, Clone)]
pub struct GenerationStats {
    /// Hypervolume indicator
    pub hypervolume: Float,
    /// Spacing metric
    pub spacing: Float,
    /// Size of Pareto front
    pub pareto_front_size: usize,
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    // Use SciRS2-Core for arrays and random number generation (SciRS2 Policy)
    use scirs2_core::ndarray::array;

    #[test]
    fn test_nsga2_creation() {
        let nsga2 = NSGAII::new()
            .population_size(50)
            .n_generations(25)
            .crossover_probability(0.8)
            .mutation_probability(0.2)
            .variable_bounds(vec![(-1.0, 1.0), (-1.0, 1.0)])
            .random_state(42);

        assert_eq!(nsga2.population_size, 50);
        assert_eq!(nsga2.n_generations, 25);
        assert_abs_diff_eq!(nsga2.crossover_probability, 0.8);
        assert_abs_diff_eq!(nsga2.mutation_probability, 0.2);
        assert_eq!(nsga2.variable_bounds.len(), 2);
        assert_eq!(nsga2.random_state, Some(42));
    }

    #[test]
    fn test_nsga2_optimization() {
        // Simple bi-objective optimization problem (ZDT1-like)
        let objective_fn = |x: &ArrayView1<Float>| {
            let f1 = x[0];
            let g = 1.0 + 9.0 * x.iter().skip(1).sum::<Float>() / (x.len() - 1) as Float;
            let f2 = g * (1.0 - (f1 / g).sqrt());
            array![f1, f2]
        };

        let nsga2 = NSGAII::new()
            .population_size(20)
            .n_generations(10)
            .variable_bounds(vec![(0.0, 1.0), (0.0, 1.0)])
            .random_state(42);

        let result = nsga2.optimize(objective_fn, 2).unwrap();

        // Check that we got a Pareto front
        assert!(!result.pareto_front().is_empty());
        assert!(result.pareto_front().len() <= 20); // Should not exceed population size

        // Check that objectives are properly calculated
        for individual in result.pareto_front() {
            assert_eq!(individual.objectives.len(), 2);
            assert!(individual.objectives[0] >= 0.0 && individual.objectives[0] <= 1.0);
            assert!(individual.objectives[1] >= 0.0);
        }

        // Check generation stats
        assert_eq!(result.generation_stats().len(), 10);
        assert!(result.generation_stats()[0].pareto_front_size > 0);
    }

    #[test]
    fn test_individual_dominance() {
        let ind1 = Individual {
            variables: array![1.0, 2.0],
            objectives: array![1.0, 2.0], // Better in both objectives
            rank: 0,
            crowding_distance: 0.0,
        };

        let ind2 = Individual {
            variables: array![2.0, 3.0],
            objectives: array![2.0, 3.0], // Worse in both objectives
            rank: 0,
            crowding_distance: 0.0,
        };

        assert!(ind1.dominates(&ind2));
        assert!(!ind2.dominates(&ind1));
    }

    #[test]
    fn test_non_dominated_sort() {
        let nsga2 = NSGAII::new();

        let population = vec![
            Individual {
                variables: array![1.0],
                objectives: array![1.0, 3.0], // Front 0
                rank: 0,
                crowding_distance: 0.0,
            },
            Individual {
                variables: array![2.0],
                objectives: array![2.0, 2.0], // Front 0
                rank: 0,
                crowding_distance: 0.0,
            },
            Individual {
                variables: array![3.0],
                objectives: array![3.0, 1.0], // Front 0
                rank: 0,
                crowding_distance: 0.0,
            },
            Individual {
                variables: array![4.0],
                objectives: array![2.0, 3.0], // Front 1 (dominated by ind1)
                rank: 0,
                crowding_distance: 0.0,
            },
        ];

        let fronts = nsga2.non_dominated_sort(&population);

        assert!(fronts.len() >= 1);
        assert_eq!(fronts[0].len(), 3); // First three form Pareto front
        if fronts.len() > 1 {
            assert_eq!(fronts[1].len(), 1); // Last one is dominated
        }
    }

    #[test]
    fn test_optimization_result_accessors() {
        let individuals = vec![
            Individual {
                variables: array![1.0, 2.0],
                objectives: array![1.0, 2.0],
                rank: 0,
                crowding_distance: 0.0,
            },
            Individual {
                variables: array![2.0, 1.0],
                objectives: array![2.0, 1.0],
                rank: 0,
                crowding_distance: 0.0,
            },
        ];

        let result = OptimizationResult {
            pareto_front: individuals.clone(),
            final_population: individuals,
            generation_stats: vec![GenerationStats {
                hypervolume: 1.0,
                spacing: 0.5,
                pareto_front_size: 2,
            }],
            n_generations: 10,
        };

        let objectives = result.pareto_objectives();
        assert_eq!(objectives.shape(), &[2, 2]);
        assert_abs_diff_eq!(objectives[[0, 0]], 1.0);
        assert_abs_diff_eq!(objectives[[0, 1]], 2.0);

        let variables = result.pareto_variables();
        assert_eq!(variables.shape(), &[2, 2]);
        assert_abs_diff_eq!(variables[[0, 0]], 1.0);
        assert_abs_diff_eq!(variables[[0, 1]], 2.0);
    }
}
