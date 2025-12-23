//! Evolutionary Algorithms for hyperparameter optimization

use std::time::Instant;

#[cfg(feature = "parallel")]
use rayon::prelude::*;
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::Random;

use crate::kernels::KernelType;
use crate::svc::SVC;
use sklears_core::error::{Result, SklearsError};
use sklears_core::traits::{Fit, Predict};

use super::{
    OptimizationConfig, OptimizationResult, ParameterSet, ParameterSpec, ScoringMetric, SearchSpace,
};

// Type aliases for compatibility
type DMatrix<T> = Array2<T>;
type DVector<T> = Array1<T>;

/// Selection methods for evolutionary algorithm
#[derive(Debug, Clone)]
pub enum SelectionMethod {
    Tournament { size: usize },
    RouletteWheel,
    RankBased,
}

/// Individual in the genetic algorithm population
#[derive(Debug, Clone)]
pub struct Individual {
    pub params: ParameterSet,
    pub fitness: f64,
}

impl Individual {
    pub fn new(params: ParameterSet) -> Self {
        Self {
            params,
            fitness: -f64::INFINITY,
        }
    }
}

/// Evolutionary Optimization hyperparameter optimizer
pub struct EvolutionaryOptimizationCV {
    config: OptimizationConfig,
    search_space: SearchSpace,
    rng: Random<scirs2_core::random::rngs::StdRng>,
    population_size: usize,
    selection_method: SelectionMethod,
    mutation_rate: f64,
    crossover_rate: f64,
    elite_ratio: f64,
}

impl EvolutionaryOptimizationCV {
    /// Create a new evolutionary optimization optimizer
    pub fn new(config: OptimizationConfig, search_space: SearchSpace) -> Self {
        let rng = if let Some(seed) = config.random_state {
            Random::seed(seed)
        } else {
            Random::seed(42) // Default seed for reproducibility
        };

        Self {
            config,
            search_space,
            rng,
            population_size: 50,
            selection_method: SelectionMethod::Tournament { size: 3 },
            mutation_rate: 0.1,
            crossover_rate: 0.8,
            elite_ratio: 0.1,
        }
    }

    /// Builder method to set population size
    pub fn population_size(mut self, size: usize) -> Self {
        self.population_size = size;
        self
    }

    /// Builder method to set selection method
    pub fn selection_method(mut self, method: SelectionMethod) -> Self {
        self.selection_method = method;
        self
    }

    /// Builder method to set mutation rate
    pub fn mutation_rate(mut self, rate: f64) -> Self {
        self.mutation_rate = rate;
        self
    }

    /// Builder method to set crossover rate
    pub fn crossover_rate(mut self, rate: f64) -> Self {
        self.crossover_rate = rate;
        self
    }

    /// Builder method to set elite ratio
    pub fn elite_ratio(mut self, ratio: f64) -> Self {
        self.elite_ratio = ratio;
        self
    }

    /// Run evolutionary optimization
    pub fn fit(&mut self, x: &DMatrix<f64>, y: &DVector<f64>) -> Result<OptimizationResult> {
        let start_time = Instant::now();

        if self.config.verbose {
            println!(
                "Evolutionary optimization with {} generations, population size {}",
                self.config.n_iterations, self.population_size
            );
        }

        // Initialize population
        let mut population = self.initialize_population()?;

        // Evaluate initial population
        self.evaluate_population(&mut population, x, y)?;

        let mut best_individual = population[0].clone();
        let mut cv_results = Vec::new();
        let mut score_history = Vec::new();
        let mut generations_without_improvement = 0;

        // Evolution loop
        for generation in 0..self.config.n_iterations {
            // Sort population by fitness (descending)
            population.sort_by(|a, b| {
                b.fitness
                    .partial_cmp(&a.fitness)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            // Track best individual
            if population[0].fitness > best_individual.fitness {
                best_individual = population[0].clone();
                generations_without_improvement = 0;
            } else {
                generations_without_improvement += 1;
            }

            // Store results
            for ind in &population {
                cv_results.push((ind.params.clone(), ind.fitness));
            }
            score_history.push(best_individual.fitness);

            if self.config.verbose && (generation + 1) % 10 == 0 {
                println!(
                    "Generation {}/{}: Best score {:.6}",
                    generation + 1,
                    self.config.n_iterations,
                    best_individual.fitness
                );
            }

            // Early stopping check
            if let Some(patience) = self.config.early_stopping_patience {
                if generations_without_improvement >= patience {
                    if self.config.verbose {
                        println!("Early stopping at generation {}", generation + 1);
                    }
                    break;
                }
            }

            // Create next generation
            let mut next_generation = Vec::new();

            // Elitism: preserve top individuals
            let n_elite = (self.population_size as f64 * self.elite_ratio) as usize;
            for individual in population.iter().take(n_elite.min(population.len())) {
                next_generation.push(individual.clone());
            }

            // Generate offspring
            while next_generation.len() < self.population_size {
                // Selection
                let parent1 = self.select_individual(&population)?;
                let parent2 = self.select_individual(&population)?;

                // Crossover
                use scirs2_core::random::essentials::Uniform;
                let dist = Uniform::new(0.0, 1.0).map_err(|e| {
                    SklearsError::InvalidInput(format!(
                        "Failed to create uniform distribution: {}",
                        e
                    ))
                })?;
                let offspring = if self.rng.sample(dist) < self.crossover_rate {
                    self.crossover(&parent1.params, &parent2.params)?
                } else {
                    parent1.params.clone()
                };

                // Mutation
                let dist = Uniform::new(0.0, 1.0).map_err(|e| {
                    SklearsError::InvalidInput(format!(
                        "Failed to create uniform distribution: {}",
                        e
                    ))
                })?;
                let mutated = if self.rng.sample(dist) < self.mutation_rate {
                    self.mutate(&offspring)?
                } else {
                    offspring
                };

                next_generation.push(Individual::new(mutated));
            }

            // Evaluate new generation
            self.evaluate_population(&mut next_generation, x, y)?;
            population = next_generation;
        }

        // Final evaluation to get best
        population.sort_by(|a, b| {
            b.fitness
                .partial_cmp(&a.fitness)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let best_individual = population[0].clone();

        if self.config.verbose {
            println!("Best score: {:.6}", best_individual.fitness);
            println!("Best params: {:?}", best_individual.params);
        }

        Ok(OptimizationResult {
            best_params: best_individual.params,
            best_score: best_individual.fitness,
            cv_results,
            n_iterations: score_history.len(),
            optimization_time: start_time.elapsed().as_secs_f64(),
            score_history,
        })
    }

    /// Initialize population with random individuals
    fn initialize_population(&mut self) -> Result<Vec<Individual>> {
        let mut population = Vec::with_capacity(self.population_size);

        // Clone search space specs to avoid borrow checker issues
        let c_spec = self.search_space.c.clone();
        let kernel_spec = self.search_space.kernel.clone();
        let tol_spec = self.search_space.tol.clone();
        let max_iter_spec = self.search_space.max_iter.clone();

        for _ in 0..self.population_size {
            let c = self.sample_value(&c_spec)?;

            let kernel = if let Some(ref spec) = kernel_spec {
                self.sample_kernel(spec)?
            } else {
                KernelType::Rbf { gamma: 1.0 }
            };

            let tol = if let Some(ref spec) = tol_spec {
                self.sample_value(spec)?
            } else {
                1e-3
            };

            let max_iter = if let Some(ref spec) = max_iter_spec {
                self.sample_value(spec)? as usize
            } else {
                1000
            };

            population.push(Individual::new(ParameterSet {
                c,
                kernel,
                tol,
                max_iter,
            }));
        }

        Ok(population)
    }

    /// Evaluate fitness for all individuals in population
    fn evaluate_population(
        &self,
        population: &mut [Individual],
        x: &DMatrix<f64>,
        y: &DVector<f64>,
    ) -> Result<()> {
        #[cfg(feature = "parallel")]
        if self.config.n_jobs.is_some() {
            // Parallel evaluation
            let fitnesses: Vec<f64> = population
                .par_iter()
                .map(|ind| {
                    self.evaluate_params(&ind.params, x, y)
                        .unwrap_or(-f64::INFINITY)
                })
                .collect();

            for (ind, fitness) in population.iter_mut().zip(fitnesses.iter()) {
                ind.fitness = *fitness;
            }
        } else {
            // Sequential evaluation
            for ind in population.iter_mut() {
                ind.fitness = self.evaluate_params(&ind.params, x, y)?;
            }
        }

        #[cfg(not(feature = "parallel"))]
        {
            // Sequential evaluation (parallel feature disabled)
            for ind in population.iter_mut() {
                ind.fitness = self.evaluate_params(&ind.params, x, y)?;
            }
        }

        Ok(())
    }

    /// Select an individual from population using selection method
    fn select_individual(&mut self, population: &[Individual]) -> Result<Individual> {
        match &self.selection_method {
            SelectionMethod::Tournament { size } => self.tournament_selection(population, *size),
            SelectionMethod::RouletteWheel => self.roulette_wheel_selection(population),
            SelectionMethod::RankBased => self.rank_based_selection(population),
        }
    }

    /// Tournament selection
    fn tournament_selection(
        &mut self,
        population: &[Individual],
        tournament_size: usize,
    ) -> Result<Individual> {
        use scirs2_core::random::essentials::Uniform;
        let dist = Uniform::new(0, population.len()).map_err(|e| {
            SklearsError::InvalidInput(format!("Failed to create uniform distribution: {}", e))
        })?;

        let mut best_idx = self.rng.sample(dist);
        let mut best_fitness = population[best_idx].fitness;

        for _ in 1..tournament_size {
            let idx = self.rng.sample(dist);
            if population[idx].fitness > best_fitness {
                best_idx = idx;
                best_fitness = population[idx].fitness;
            }
        }

        Ok(population[best_idx].clone())
    }

    /// Roulette wheel selection (fitness proportionate)
    fn roulette_wheel_selection(&mut self, population: &[Individual]) -> Result<Individual> {
        // Shift fitnesses to be non-negative
        let min_fitness = population
            .iter()
            .map(|ind| ind.fitness)
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(0.0);
        let offset = if min_fitness < 0.0 {
            -min_fitness + 1.0
        } else {
            0.0
        };

        let total_fitness: f64 = population.iter().map(|ind| ind.fitness + offset).sum();

        if total_fitness <= 0.0 {
            // All fitnesses are equal or negative, select randomly
            use scirs2_core::random::essentials::Uniform;
            let dist = Uniform::new(0, population.len()).map_err(|e| {
                SklearsError::InvalidInput(format!("Failed to create uniform distribution: {}", e))
            })?;
            let idx = self.rng.sample(dist);
            return Ok(population[idx].clone());
        }

        use scirs2_core::random::essentials::Uniform;
        let dist = Uniform::new(0.0, total_fitness).map_err(|e| {
            SklearsError::InvalidInput(format!("Failed to create uniform distribution: {}", e))
        })?;
        let mut spin = self.rng.sample(dist);

        for ind in population {
            spin -= ind.fitness + offset;
            if spin <= 0.0 {
                return Ok(ind.clone());
            }
        }

        // Fallback (shouldn't reach here)
        Ok(population[population.len() - 1].clone())
    }

    /// Rank-based selection
    fn rank_based_selection(&mut self, population: &[Individual]) -> Result<Individual> {
        // Assign ranks (higher fitness = higher rank)
        let total_rank = population.len() * (population.len() + 1) / 2;

        use scirs2_core::random::essentials::Uniform;
        let dist = Uniform::new(0.0, total_rank as f64).map_err(|e| {
            SklearsError::InvalidInput(format!("Failed to create uniform distribution: {}", e))
        })?;
        let mut spin = self.rng.sample(dist);

        for (i, ind) in population.iter().enumerate() {
            let rank = population.len() - i; // Higher index = better fitness (assumes sorted)
            spin -= rank as f64;
            if spin <= 0.0 {
                return Ok(ind.clone());
            }
        }

        // Fallback
        Ok(population[population.len() - 1].clone())
    }

    /// Crossover two parameter sets
    fn crossover(
        &mut self,
        parent1: &ParameterSet,
        parent2: &ParameterSet,
    ) -> Result<ParameterSet> {
        use scirs2_core::random::essentials::Uniform;
        let dist = Uniform::new(0.0, 1.0).map_err(|e| {
            SklearsError::InvalidInput(format!("Failed to create uniform distribution: {}", e))
        })?;

        // Uniform crossover: each gene comes from either parent with 50% probability
        let c = if self.rng.sample(dist) < 0.5 {
            parent1.c
        } else {
            parent2.c
        };

        let kernel = if self.rng.sample(dist) < 0.5 {
            parent1.kernel.clone()
        } else {
            parent2.kernel.clone()
        };

        let tol = if self.rng.sample(dist) < 0.5 {
            parent1.tol
        } else {
            parent2.tol
        };

        let max_iter = if self.rng.sample(dist) < 0.5 {
            parent1.max_iter
        } else {
            parent2.max_iter
        };

        Ok(ParameterSet {
            c,
            kernel,
            tol,
            max_iter,
        })
    }

    /// Mutate a parameter set
    fn mutate(&mut self, params: &ParameterSet) -> Result<ParameterSet> {
        use scirs2_core::random::essentials::Uniform;
        let dist = Uniform::new(0.0, 1.0).map_err(|e| {
            SklearsError::InvalidInput(format!("Failed to create uniform distribution: {}", e))
        })?;

        // Clone search space specs to avoid borrow checker issues
        let c_spec = self.search_space.c.clone();
        let kernel_spec = self.search_space.kernel.clone();
        let tol_spec = self.search_space.tol.clone();
        let max_iter_spec = self.search_space.max_iter.clone();

        // Mutate each gene with 20% probability
        let c = if self.rng.sample(dist) < 0.2 {
            self.sample_value(&c_spec)?
        } else {
            params.c
        };

        let kernel = if self.rng.sample(dist) < 0.2 {
            if let Some(ref spec) = kernel_spec {
                self.sample_kernel(spec)?
            } else {
                params.kernel.clone()
            }
        } else {
            params.kernel.clone()
        };

        let tol = if self.rng.sample(dist) < 0.2 {
            if let Some(ref spec) = tol_spec {
                self.sample_value(spec)?
            } else {
                params.tol
            }
        } else {
            params.tol
        };

        let max_iter = if self.rng.sample(dist) < 0.2 {
            if let Some(ref spec) = max_iter_spec {
                self.sample_value(spec)? as usize
            } else {
                params.max_iter
            }
        } else {
            params.max_iter
        };

        Ok(ParameterSet {
            c,
            kernel,
            tol,
            max_iter,
        })
    }

    /// Sample a single value from parameter specification
    fn sample_value(&mut self, spec: &ParameterSpec) -> Result<f64> {
        match spec {
            ParameterSpec::Fixed(value) => Ok(*value),
            ParameterSpec::Uniform { min, max } => {
                use scirs2_core::random::essentials::Uniform;
                let dist = Uniform::new(*min, *max).map_err(|e| {
                    SklearsError::InvalidInput(format!(
                        "Failed to create uniform distribution: {}",
                        e
                    ))
                })?;
                Ok(self.rng.sample(dist))
            }
            ParameterSpec::LogUniform { min, max } => {
                use scirs2_core::random::essentials::Uniform;
                let log_min = min.ln();
                let log_max = max.ln();
                let dist = Uniform::new(log_min, log_max).map_err(|e| {
                    SklearsError::InvalidInput(format!(
                        "Failed to create uniform distribution: {}",
                        e
                    ))
                })?;
                let log_val = self.rng.sample(dist);
                Ok(log_val.exp())
            }
            ParameterSpec::Choice(choices) => {
                if choices.is_empty() {
                    return Err(SklearsError::InvalidInput("Empty choice list".to_string()));
                }
                use scirs2_core::random::essentials::Uniform;
                let dist = Uniform::new(0, choices.len()).map_err(|e| {
                    SklearsError::InvalidInput(format!(
                        "Failed to create uniform distribution: {}",
                        e
                    ))
                })?;
                let idx = self.rng.sample(dist);
                Ok(choices[idx])
            }
            ParameterSpec::KernelChoice(_) => Err(SklearsError::InvalidInput(
                "Use sample_kernel for kernel specs".to_string(),
            )),
        }
    }

    /// Sample a kernel from kernel specification
    fn sample_kernel(&mut self, spec: &ParameterSpec) -> Result<KernelType> {
        match spec {
            ParameterSpec::KernelChoice(kernels) => {
                if kernels.is_empty() {
                    return Err(SklearsError::InvalidInput(
                        "Empty kernel choice list".to_string(),
                    ));
                }
                use scirs2_core::random::essentials::Uniform;
                let dist = Uniform::new(0, kernels.len()).map_err(|e| {
                    SklearsError::InvalidInput(format!(
                        "Failed to create uniform distribution: {}",
                        e
                    ))
                })?;
                let idx = self.rng.sample(dist);
                Ok(kernels[idx].clone())
            }
            _ => Err(SklearsError::InvalidInput(
                "Invalid kernel specification".to_string(),
            )),
        }
    }

    /// Evaluate parameter set using cross-validation
    fn evaluate_params(
        &self,
        params: &ParameterSet,
        x: &DMatrix<f64>,
        y: &DVector<f64>,
    ) -> Result<f64> {
        let scores = self.cross_validate(params, x, y)?;
        Ok(scores.iter().sum::<f64>() / scores.len() as f64)
    }

    /// Perform cross-validation
    fn cross_validate(
        &self,
        params: &ParameterSet,
        x: &DMatrix<f64>,
        y: &DVector<f64>,
    ) -> Result<Vec<f64>> {
        let n_samples = x.nrows();
        let fold_size = n_samples / self.config.cv_folds;
        let mut scores = Vec::new();

        for fold in 0..self.config.cv_folds {
            let start_idx = fold * fold_size;
            let end_idx = if fold == self.config.cv_folds - 1 {
                n_samples
            } else {
                (fold + 1) * fold_size
            };

            // Create train/test splits
            let mut x_train_data = Vec::new();
            let mut y_train_vals = Vec::new();
            let mut x_test_data = Vec::new();
            let mut y_test_vals = Vec::new();

            for i in 0..n_samples {
                if i >= start_idx && i < end_idx {
                    // Test set
                    for j in 0..x.ncols() {
                        x_test_data.push(x[[i, j]]);
                    }
                    y_test_vals.push(y[i]);
                } else {
                    // Training set
                    for j in 0..x.ncols() {
                        x_train_data.push(x[[i, j]]);
                    }
                    y_train_vals.push(y[i]);
                }
            }

            let n_train = y_train_vals.len();
            let n_test = y_test_vals.len();
            let n_features = x.ncols();

            let x_train = Array2::from_shape_vec((n_train, n_features), x_train_data)?;
            let y_train = Array1::from_vec(y_train_vals);
            let x_test = Array2::from_shape_vec((n_test, n_features), x_test_data)?;
            let y_test = Array1::from_vec(y_test_vals);

            // Train and evaluate model
            let svm = SVC::new()
                .c(params.c)
                .kernel(params.kernel.clone())
                .tol(params.tol)
                .max_iter(params.max_iter);

            let fitted_svm = svm.fit(&x_train, &y_train)?;
            let y_pred = fitted_svm.predict(&x_test)?;

            let score = self.calculate_score(&y_test, &y_pred)?;
            scores.push(score);
        }

        Ok(scores)
    }

    /// Calculate score based on scoring metric
    fn calculate_score(&self, y_true: &DVector<f64>, y_pred: &DVector<f64>) -> Result<f64> {
        match self.config.scoring {
            ScoringMetric::Accuracy => {
                let correct = y_true
                    .iter()
                    .zip(y_pred.iter())
                    .map(|(&t, &p)| if (t - p).abs() < 0.5 { 1.0 } else { 0.0 })
                    .sum::<f64>();
                Ok(correct / y_true.len() as f64)
            }
            ScoringMetric::MeanSquaredError => {
                let mse = y_true
                    .iter()
                    .zip(y_pred.iter())
                    .map(|(&t, &p)| (t - p).powi(2))
                    .sum::<f64>()
                    / y_true.len() as f64;
                Ok(-mse) // Negative because we want to maximize
            }
            ScoringMetric::MeanAbsoluteError => {
                let mae = y_true
                    .iter()
                    .zip(y_pred.iter())
                    .map(|(&t, &p)| (t - p).abs())
                    .sum::<f64>()
                    / y_true.len() as f64;
                Ok(-mae) // Negative because we want to maximize
            }
            _ => {
                // For now, default to accuracy for other metrics
                let correct = y_true
                    .iter()
                    .zip(y_pred.iter())
                    .map(|(&t, &p)| if (t - p).abs() < 0.5 { 1.0 } else { 0.0 })
                    .sum::<f64>();
                Ok(correct / y_true.len() as f64)
            }
        }
    }
}
