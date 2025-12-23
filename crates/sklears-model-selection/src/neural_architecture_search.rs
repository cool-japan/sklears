//! Neural Architecture Search (NAS) Integration
//!
//! This module provides automated neural network architecture optimization using various
//! search strategies including evolutionary algorithms, reinforcement learning, and
//! gradient-based methods for finding optimal neural network architectures.

use scirs2_core::rand_prelude::IndexedRandom;
use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::Rng;
use scirs2_core::random::SeedableRng;
use sklears_core::types::Float;
use std::collections::HashMap;

/// Neural Architecture Search strategies
#[derive(Debug, Clone)]
pub enum NASStrategy {
    /// Evolutionary algorithm-based search
    Evolutionary {
        population_size: usize,

        generations: usize,

        mutation_rate: Float,

        crossover_rate: Float,
    },
    /// Reinforcement learning-based search
    ReinforcementLearning {
        episodes: usize,
        learning_rate: Float,
        exploration_rate: Float,
    },
    /// Gradient-based differentiable architecture search
    GDAS {
        search_epochs: usize,
        learning_rate: Float,
        weight_decay: Float,
    },
    /// Random search baseline
    RandomSearch { n_trials: usize, max_depth: usize },
    /// Progressive search with increasing complexity
    Progressive {
        stages: usize,
        complexity_growth: Float,
    },
    /// Bayesian optimization for architecture search
    BayesianOptimization {
        n_trials: usize,
        acquisition_function: String,
    },
}

/// Neural network architecture representation
#[derive(Debug, Clone)]
pub struct NeuralArchitecture {
    /// Number of layers
    pub num_layers: usize,
    /// Hidden layer sizes
    pub layer_sizes: Vec<usize>,
    /// Activation functions for each layer
    pub activations: Vec<String>,
    /// Dropout rates for each layer
    pub dropout_rates: Vec<Float>,
    /// Batch normalization flags
    pub batch_norm: Vec<bool>,
    /// Skip connections (ResNet-style)
    pub skip_connections: Vec<(usize, usize)>,
    /// Architecture complexity score
    pub complexity_score: Float,
}

/// Architecture search space definition
#[derive(Debug, Clone)]
pub struct ArchitectureSearchSpace {
    /// Range of layer counts
    pub layer_count_range: (usize, usize),
    /// Range of neurons per layer
    pub neuron_count_range: (usize, usize),
    /// Available activation functions
    pub activation_options: Vec<String>,
    /// Dropout rate range
    pub dropout_range: (Float, Float),
    /// Maximum skip connection distance
    pub max_skip_distance: usize,
    /// Whether to use batch normalization
    pub use_batch_norm: bool,
}

/// NAS configuration
#[derive(Debug, Clone)]
pub struct NASConfig {
    pub strategy: NASStrategy,
    pub search_space: ArchitectureSearchSpace,
    pub evaluation_metric: String,
    pub max_evaluation_time: Option<u64>,
    pub early_stopping_patience: usize,
    pub validation_split: Float,
    pub random_state: Option<u64>,
    pub parallel_evaluations: usize,
}

/// Architecture evaluation result
#[derive(Debug, Clone)]
pub struct ArchitectureEvaluation {
    pub architecture: NeuralArchitecture,
    pub validation_score: Float,
    pub training_time: Float,
    pub parameters_count: usize,
    pub flops: usize,
    pub memory_usage: Float,
}

/// NAS optimization result
#[derive(Debug, Clone)]
pub struct NASResult {
    pub best_architecture: NeuralArchitecture,
    pub best_score: Float,
    pub search_history: Vec<ArchitectureEvaluation>,
    pub total_search_time: Float,
    pub architectures_evaluated: usize,
    pub convergence_curve: Vec<Float>,
}

/// Neural Architecture Search optimizer
#[derive(Debug, Clone)]
pub struct NASOptimizer {
    config: NASConfig,
    rng: StdRng,
}

impl Default for ArchitectureSearchSpace {
    fn default() -> Self {
        Self {
            layer_count_range: (1, 10),
            neuron_count_range: (16, 1024),
            activation_options: vec![
                "relu".to_string(),
                "tanh".to_string(),
                "sigmoid".to_string(),
                "swish".to_string(),
                "gelu".to_string(),
            ],
            dropout_range: (0.0, 0.5),
            max_skip_distance: 3,
            use_batch_norm: true,
        }
    }
}

impl Default for NASConfig {
    fn default() -> Self {
        Self {
            strategy: NASStrategy::Evolutionary {
                population_size: 20,
                generations: 50,
                mutation_rate: 0.1,
                crossover_rate: 0.7,
            },
            search_space: ArchitectureSearchSpace::default(),
            evaluation_metric: "accuracy".to_string(),
            max_evaluation_time: Some(3600), // 1 hour
            early_stopping_patience: 10,
            validation_split: 0.2,
            random_state: None,
            parallel_evaluations: 4,
        }
    }
}

impl NASOptimizer {
    /// Create a new NAS optimizer
    pub fn new(config: NASConfig) -> Self {
        let rng = match config.random_state {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => {
                use scirs2_core::random::thread_rng;
                StdRng::from_rng(&mut thread_rng())
            }
        };

        Self { config, rng }
    }

    /// Search for optimal neural architecture
    pub fn search<F>(&mut self, evaluation_fn: F) -> Result<NASResult, Box<dyn std::error::Error>>
    where
        F: Fn(&NeuralArchitecture) -> Result<ArchitectureEvaluation, Box<dyn std::error::Error>>,
    {
        let start_time = std::time::Instant::now();

        let result = match &self.config.strategy {
            NASStrategy::Evolutionary { .. } => self.evolutionary_search(&evaluation_fn)?,
            NASStrategy::ReinforcementLearning { .. } => {
                self.reinforcement_learning_search(&evaluation_fn)?
            }
            NASStrategy::GDAS { .. } => self.gradient_based_search(&evaluation_fn)?,
            NASStrategy::RandomSearch { .. } => self.random_search(&evaluation_fn)?,
            NASStrategy::Progressive { .. } => self.progressive_search(&evaluation_fn)?,
            NASStrategy::BayesianOptimization { .. } => {
                self.bayesian_optimization_search(&evaluation_fn)?
            }
        };

        let total_time = start_time.elapsed().as_secs_f64() as Float;

        Ok(NASResult {
            best_architecture: result.0,
            best_score: result.1,
            search_history: result.2,
            total_search_time: total_time,
            architectures_evaluated: result.3,
            convergence_curve: result.4,
        })
    }

    /// Evolutionary algorithm-based architecture search
    fn evolutionary_search<F>(
        &mut self,
        evaluation_fn: &F,
    ) -> Result<
        (
            NeuralArchitecture,
            Float,
            Vec<ArchitectureEvaluation>,
            usize,
            Vec<Float>,
        ),
        Box<dyn std::error::Error>,
    >
    where
        F: Fn(&NeuralArchitecture) -> Result<ArchitectureEvaluation, Box<dyn std::error::Error>>,
    {
        let (population_size, generations, mutation_rate, crossover_rate) =
            match &self.config.strategy {
                NASStrategy::Evolutionary {
                    population_size,
                    generations,
                    mutation_rate,
                    crossover_rate,
                } => (
                    *population_size,
                    *generations,
                    *mutation_rate,
                    *crossover_rate,
                ),
                _ => unreachable!(),
            };

        let mut population = self.initialize_population(population_size)?;
        let mut search_history = Vec::new();
        let mut convergence_curve = Vec::new();
        let mut best_architecture = population[0].clone();
        let mut best_score = Float::NEG_INFINITY;
        let mut evaluations_count = 0;

        for _generation in 0..generations {
            // Evaluate population
            let mut evaluations = Vec::new();
            for architecture in &population {
                let evaluation = evaluation_fn(architecture)?;
                evaluations.push(evaluation.clone());
                search_history.push(evaluation.clone());
                evaluations_count += 1;

                if evaluation.validation_score > best_score {
                    best_score = evaluation.validation_score;
                    best_architecture = architecture.clone();
                }
            }

            // Selection, crossover, and mutation
            population =
                self.evolve_population(&population, &evaluations, crossover_rate, mutation_rate)?;

            // Track convergence
            let generation_best = evaluations
                .iter()
                .map(|e| e.validation_score)
                .fold(Float::NEG_INFINITY, |a, b| a.max(b));
            convergence_curve.push(generation_best);

            // Early stopping check
            if self.check_early_stopping(&convergence_curve) {
                break;
            }
        }

        Ok((
            best_architecture,
            best_score,
            search_history,
            evaluations_count,
            convergence_curve,
        ))
    }

    /// Reinforcement learning-based architecture search
    fn reinforcement_learning_search<F>(
        &mut self,
        evaluation_fn: &F,
    ) -> Result<
        (
            NeuralArchitecture,
            Float,
            Vec<ArchitectureEvaluation>,
            usize,
            Vec<Float>,
        ),
        Box<dyn std::error::Error>,
    >
    where
        F: Fn(&NeuralArchitecture) -> Result<ArchitectureEvaluation, Box<dyn std::error::Error>>,
    {
        let (episodes, learning_rate, exploration_rate) = match &self.config.strategy {
            NASStrategy::ReinforcementLearning {
                episodes,
                learning_rate,
                exploration_rate,
            } => (*episodes, *learning_rate, *exploration_rate),
            _ => unreachable!(),
        };

        let mut search_history = Vec::new();
        let mut convergence_curve = Vec::new();
        let mut best_architecture = self.generate_random_architecture()?;
        let mut best_score = Float::NEG_INFINITY;
        let mut evaluations_count = 0;

        // Simple policy-based RL approach
        let mut policy_weights = HashMap::new();
        let mut epsilon = exploration_rate;

        for _episode in 0..episodes {
            // Generate architecture using current policy
            let architecture = if self.rng.random::<Float>() < epsilon {
                self.generate_random_architecture()?
            } else {
                self.generate_architecture_from_policy(&policy_weights)?
            };

            // Evaluate architecture
            let evaluation = evaluation_fn(&architecture)?;
            search_history.push(evaluation.clone());
            evaluations_count += 1;

            // Update policy weights based on reward
            let reward = evaluation.validation_score;
            self.update_policy_weights(&mut policy_weights, &architecture, reward, learning_rate);

            if evaluation.validation_score > best_score {
                best_score = evaluation.validation_score;
                best_architecture = architecture.clone();
            }

            convergence_curve.push(best_score);

            // Decay exploration rate
            epsilon *= 0.99;

            // Early stopping check
            if self.check_early_stopping(&convergence_curve) {
                break;
            }
        }

        Ok((
            best_architecture,
            best_score,
            search_history,
            evaluations_count,
            convergence_curve,
        ))
    }

    /// Gradient-based differentiable architecture search
    fn gradient_based_search<F>(
        &mut self,
        evaluation_fn: &F,
    ) -> Result<
        (
            NeuralArchitecture,
            Float,
            Vec<ArchitectureEvaluation>,
            usize,
            Vec<Float>,
        ),
        Box<dyn std::error::Error>,
    >
    where
        F: Fn(&NeuralArchitecture) -> Result<ArchitectureEvaluation, Box<dyn std::error::Error>>,
    {
        let (search_epochs, learning_rate, _weight_decay) = match &self.config.strategy {
            NASStrategy::GDAS {
                search_epochs,
                learning_rate,
                weight_decay,
            } => (*search_epochs, *learning_rate, *weight_decay),
            _ => unreachable!(),
        };

        let mut search_history = Vec::new();
        let mut convergence_curve = Vec::new();
        let mut best_architecture = self.generate_random_architecture()?;
        let mut best_score = Float::NEG_INFINITY;
        let mut evaluations_count = 0;

        // Simulate gradient-based search with architecture parameters
        let mut architecture_params = self.initialize_architecture_parameters()?;

        for _epoch in 0..search_epochs {
            // Sample architecture from current parameters
            let architecture = self.sample_architecture_from_params(&architecture_params)?;

            // Evaluate architecture
            let evaluation = evaluation_fn(&architecture)?;
            search_history.push(evaluation.clone());
            evaluations_count += 1;

            // Update architecture parameters (simplified gradient update)
            self.update_architecture_parameters(
                &mut architecture_params,
                &evaluation,
                learning_rate,
            );

            if evaluation.validation_score > best_score {
                best_score = evaluation.validation_score;
                best_architecture = architecture.clone();
            }

            convergence_curve.push(best_score);

            // Early stopping check
            if self.check_early_stopping(&convergence_curve) {
                break;
            }
        }

        Ok((
            best_architecture,
            best_score,
            search_history,
            evaluations_count,
            convergence_curve,
        ))
    }

    /// Random search baseline
    fn random_search<F>(
        &mut self,
        evaluation_fn: &F,
    ) -> Result<
        (
            NeuralArchitecture,
            Float,
            Vec<ArchitectureEvaluation>,
            usize,
            Vec<Float>,
        ),
        Box<dyn std::error::Error>,
    >
    where
        F: Fn(&NeuralArchitecture) -> Result<ArchitectureEvaluation, Box<dyn std::error::Error>>,
    {
        let (n_trials, _max_depth) = match &self.config.strategy {
            NASStrategy::RandomSearch {
                n_trials,
                max_depth,
            } => (*n_trials, *max_depth),
            _ => unreachable!(),
        };

        let mut search_history = Vec::new();
        let mut convergence_curve = Vec::new();
        let mut best_architecture = self.generate_random_architecture()?;
        let mut best_score = Float::NEG_INFINITY;

        for _trial in 0..n_trials {
            let architecture = self.generate_random_architecture()?;
            let evaluation = evaluation_fn(&architecture)?;
            search_history.push(evaluation.clone());

            if evaluation.validation_score > best_score {
                best_score = evaluation.validation_score;
                best_architecture = architecture.clone();
            }

            convergence_curve.push(best_score);

            // Early stopping check
            if self.check_early_stopping(&convergence_curve) {
                break;
            }
        }

        Ok((
            best_architecture,
            best_score,
            search_history,
            n_trials,
            convergence_curve,
        ))
    }

    /// Progressive search with increasing complexity
    fn progressive_search<F>(
        &mut self,
        evaluation_fn: &F,
    ) -> Result<
        (
            NeuralArchitecture,
            Float,
            Vec<ArchitectureEvaluation>,
            usize,
            Vec<Float>,
        ),
        Box<dyn std::error::Error>,
    >
    where
        F: Fn(&NeuralArchitecture) -> Result<ArchitectureEvaluation, Box<dyn std::error::Error>>,
    {
        let (stages, complexity_growth) = match &self.config.strategy {
            NASStrategy::Progressive {
                stages,
                complexity_growth,
            } => (*stages, *complexity_growth),
            _ => unreachable!(),
        };

        let mut search_history = Vec::new();
        let mut convergence_curve = Vec::new();
        let mut best_architecture = self.generate_random_architecture()?;
        let mut best_score = Float::NEG_INFINITY;
        let mut evaluations_count = 0;

        let mut current_complexity = 1.0;
        let trials_per_stage = 20;

        for _stage in 0..stages {
            // Search with current complexity constraint
            for _ in 0..trials_per_stage {
                let architecture =
                    self.generate_architecture_with_complexity(current_complexity)?;
                let evaluation = evaluation_fn(&architecture)?;
                search_history.push(evaluation.clone());
                evaluations_count += 1;

                if evaluation.validation_score > best_score {
                    best_score = evaluation.validation_score;
                    best_architecture = architecture.clone();
                }

                convergence_curve.push(best_score);
            }

            // Increase complexity for next stage
            current_complexity *= complexity_growth;

            // Early stopping check
            if self.check_early_stopping(&convergence_curve) {
                break;
            }
        }

        Ok((
            best_architecture,
            best_score,
            search_history,
            evaluations_count,
            convergence_curve,
        ))
    }

    /// Bayesian optimization for architecture search
    fn bayesian_optimization_search<F>(
        &mut self,
        evaluation_fn: &F,
    ) -> Result<
        (
            NeuralArchitecture,
            Float,
            Vec<ArchitectureEvaluation>,
            usize,
            Vec<Float>,
        ),
        Box<dyn std::error::Error>,
    >
    where
        F: Fn(&NeuralArchitecture) -> Result<ArchitectureEvaluation, Box<dyn std::error::Error>>,
    {
        let (n_trials, _acquisition_function) = match &self.config.strategy {
            NASStrategy::BayesianOptimization {
                n_trials,
                acquisition_function,
            } => (*n_trials, acquisition_function),
            _ => unreachable!(),
        };

        let mut search_history = Vec::new();
        let mut convergence_curve = Vec::new();
        let mut best_architecture = self.generate_random_architecture()?;
        let mut best_score = Float::NEG_INFINITY;

        // Initialize with random samples
        let init_samples = 5;
        let mut evaluated_architectures = Vec::new();

        for _ in 0..init_samples {
            let architecture = self.generate_random_architecture()?;
            let evaluation = evaluation_fn(&architecture)?;
            search_history.push(evaluation.clone());
            evaluated_architectures.push((architecture.clone(), evaluation.validation_score));

            if evaluation.validation_score > best_score {
                best_score = evaluation.validation_score;
                best_architecture = architecture.clone();
            }

            convergence_curve.push(best_score);
        }

        // Bayesian optimization loop
        for _ in init_samples..n_trials {
            // Select next architecture using acquisition function
            let architecture = self.select_next_architecture_bayesian(&evaluated_architectures)?;
            let evaluation = evaluation_fn(&architecture)?;
            search_history.push(evaluation.clone());
            evaluated_architectures.push((architecture.clone(), evaluation.validation_score));

            if evaluation.validation_score > best_score {
                best_score = evaluation.validation_score;
                best_architecture = architecture.clone();
            }

            convergence_curve.push(best_score);

            // Early stopping check
            if self.check_early_stopping(&convergence_curve) {
                break;
            }
        }

        Ok((
            best_architecture,
            best_score,
            search_history,
            n_trials,
            convergence_curve,
        ))
    }

    /// Generate random neural architecture
    fn generate_random_architecture(
        &mut self,
    ) -> Result<NeuralArchitecture, Box<dyn std::error::Error>> {
        let num_layers = self.rng.gen_range(
            self.config.search_space.layer_count_range.0
                ..=self.config.search_space.layer_count_range.1,
        );

        let mut layer_sizes = Vec::new();
        let mut activations = Vec::new();
        let mut dropout_rates = Vec::new();
        let mut batch_norm = Vec::new();

        for _ in 0..num_layers {
            let size = self.rng.gen_range(
                self.config.search_space.neuron_count_range.0
                    ..=self.config.search_space.neuron_count_range.1,
            );
            layer_sizes.push(size);

            let activation = self
                .config
                .search_space
                .activation_options
                .choose(&mut self.rng)
                .unwrap()
                .clone();
            activations.push(activation);

            let dropout = self.rng.gen_range(
                self.config.search_space.dropout_range.0..=self.config.search_space.dropout_range.1,
            );
            dropout_rates.push(dropout);

            batch_norm.push(self.config.search_space.use_batch_norm && self.rng.gen_bool(0.5));
        }

        // Generate skip connections
        let mut skip_connections = Vec::new();
        if num_layers > 2 {
            let n_skip = self.rng.gen_range(0..num_layers / 2 + 1);
            for _ in 0..n_skip {
                let from = self.rng.gen_range(0..num_layers - 1);
                let max_to =
                    (from + self.config.search_space.max_skip_distance).min(num_layers - 1);
                if max_to > from {
                    let to = self.rng.gen_range(from + 1..max_to + 1);
                    skip_connections.push((from, to));
                }
            }
        }

        let complexity_score = self.calculate_complexity_score(&layer_sizes, &skip_connections);

        Ok(NeuralArchitecture {
            num_layers,
            layer_sizes,
            activations,
            dropout_rates,
            batch_norm,
            skip_connections,
            complexity_score,
        })
    }

    /// Initialize population for evolutionary algorithm
    fn initialize_population(
        &mut self,
        size: usize,
    ) -> Result<Vec<NeuralArchitecture>, Box<dyn std::error::Error>> {
        let mut population = Vec::new();
        for _ in 0..size {
            population.push(self.generate_random_architecture()?);
        }
        Ok(population)
    }

    /// Evolve population using genetic operators
    fn evolve_population(
        &mut self,
        population: &[NeuralArchitecture],
        evaluations: &[ArchitectureEvaluation],
        crossover_rate: Float,
        mutation_rate: Float,
    ) -> Result<Vec<NeuralArchitecture>, Box<dyn std::error::Error>> {
        let mut new_population = Vec::new();
        let population_size = population.len();

        // Keep best individuals (elitism)
        let mut sorted_indices: Vec<usize> = (0..population_size).collect();
        sorted_indices.sort_by(|&a, &b| {
            evaluations[b]
                .validation_score
                .partial_cmp(&evaluations[a].validation_score)
                .unwrap()
        });

        let elite_count = population_size / 4;
        for &idx in sorted_indices.iter().take(elite_count) {
            new_population.push(population[idx].clone());
        }

        // Generate offspring through crossover and mutation
        while new_population.len() < population_size {
            // Extract all random values first to avoid multiple mutable borrows
            let crossover_prob = self.rng.random::<Float>();
            let mutation_prob = self.rng.random::<Float>();

            let (parent1, parent2) = self.tournament_selection_pair(population, evaluations, 3)?;

            let mut offspring = if crossover_prob < crossover_rate {
                self.crossover(parent1, parent2)?
            } else {
                parent1.clone()
            };

            if mutation_prob < mutation_rate {
                offspring = self.mutate(&offspring)?;
            }

            new_population.push(offspring);
        }

        Ok(new_population)
    }

    /// Tournament selection for evolutionary algorithm
    fn tournament_selection<'a>(
        &mut self,
        population: &'a [NeuralArchitecture],
        evaluations: &[ArchitectureEvaluation],
        tournament_size: usize,
    ) -> Result<&'a NeuralArchitecture, Box<dyn std::error::Error>> {
        let mut best_idx = 0;
        let mut best_score = Float::NEG_INFINITY;

        for _ in 0..tournament_size {
            let idx = self.rng.gen_range(0..population.len());
            if evaluations[idx].validation_score > best_score {
                best_score = evaluations[idx].validation_score;
                best_idx = idx;
            }
        }

        Ok(&population[best_idx])
    }

    /// Tournament selection for two parents
    fn tournament_selection_pair<'a>(
        &mut self,
        population: &'a [NeuralArchitecture],
        evaluations: &[ArchitectureEvaluation],
        tournament_size: usize,
    ) -> Result<(&'a NeuralArchitecture, &'a NeuralArchitecture), Box<dyn std::error::Error>> {
        let parent1_idx =
            self.tournament_selection_idx(population, evaluations, tournament_size)?;
        let parent2_idx =
            self.tournament_selection_idx(population, evaluations, tournament_size)?;

        Ok((&population[parent1_idx], &population[parent2_idx]))
    }

    /// Tournament selection returning index
    fn tournament_selection_idx(
        &mut self,
        population: &[NeuralArchitecture],
        evaluations: &[ArchitectureEvaluation],
        tournament_size: usize,
    ) -> Result<usize, Box<dyn std::error::Error>> {
        let mut best_idx = 0;
        let mut best_score = Float::NEG_INFINITY;

        for _ in 0..tournament_size {
            let idx = self.rng.gen_range(0..population.len());
            if evaluations[idx].validation_score > best_score {
                best_score = evaluations[idx].validation_score;
                best_idx = idx;
            }
        }

        Ok(best_idx)
    }

    /// Crossover two architectures
    fn crossover(
        &mut self,
        parent1: &NeuralArchitecture,
        parent2: &NeuralArchitecture,
    ) -> Result<NeuralArchitecture, Box<dyn std::error::Error>> {
        let num_layers = if self.rng.gen_bool(0.5) {
            parent1.num_layers
        } else {
            parent2.num_layers
        };

        let mut layer_sizes = Vec::new();
        let mut activations = Vec::new();
        let mut dropout_rates = Vec::new();
        let mut batch_norm = Vec::new();

        for i in 0..num_layers {
            let parent1_idx = i.min(parent1.num_layers - 1);
            let parent2_idx = i.min(parent2.num_layers - 1);

            layer_sizes.push(if self.rng.gen_bool(0.5) {
                parent1.layer_sizes[parent1_idx]
            } else {
                parent2.layer_sizes[parent2_idx]
            });

            activations.push(if self.rng.gen_bool(0.5) {
                parent1.activations[parent1_idx].clone()
            } else {
                parent2.activations[parent2_idx].clone()
            });

            dropout_rates.push(if self.rng.gen_bool(0.5) {
                parent1.dropout_rates[parent1_idx]
            } else {
                parent2.dropout_rates[parent2_idx]
            });

            batch_norm.push(if self.rng.gen_bool(0.5) {
                parent1.batch_norm[parent1_idx]
            } else {
                parent2.batch_norm[parent2_idx]
            });
        }

        // Combine skip connections
        let mut skip_connections = Vec::new();
        for &(from, to) in &parent1.skip_connections {
            if from < num_layers && to < num_layers {
                skip_connections.push((from, to));
            }
        }
        for &(from, to) in &parent2.skip_connections {
            if from < num_layers && to < num_layers && !skip_connections.contains(&(from, to)) {
                skip_connections.push((from, to));
            }
        }

        let complexity_score = self.calculate_complexity_score(&layer_sizes, &skip_connections);

        Ok(NeuralArchitecture {
            num_layers,
            layer_sizes,
            activations,
            dropout_rates,
            batch_norm,
            skip_connections,
            complexity_score,
        })
    }

    /// Mutate an architecture
    fn mutate(
        &mut self,
        architecture: &NeuralArchitecture,
    ) -> Result<NeuralArchitecture, Box<dyn std::error::Error>> {
        let mut mutated = architecture.clone();

        // Mutate number of layers
        if self.rng.gen_bool(0.1) {
            let change = if self.rng.gen_bool(0.5) { 1 } else { -1 };
            mutated.num_layers = ((mutated.num_layers as i32 + change) as usize)
                .max(self.config.search_space.layer_count_range.0)
                .min(self.config.search_space.layer_count_range.1);
        }

        // Adjust vectors to match new layer count
        while mutated.layer_sizes.len() < mutated.num_layers {
            mutated.layer_sizes.push(self.rng.gen_range(
                self.config.search_space.neuron_count_range.0
                    ..=self.config.search_space.neuron_count_range.1,
            ));
            mutated.activations.push(
                self.config
                    .search_space
                    .activation_options
                    .choose(&mut self.rng)
                    .unwrap()
                    .clone(),
            );
            mutated.dropout_rates.push(self.rng.gen_range(
                self.config.search_space.dropout_range.0..=self.config.search_space.dropout_range.1,
            ));
            mutated
                .batch_norm
                .push(self.config.search_space.use_batch_norm && self.rng.gen_bool(0.5));
        }
        mutated.layer_sizes.truncate(mutated.num_layers);
        mutated.activations.truncate(mutated.num_layers);
        mutated.dropout_rates.truncate(mutated.num_layers);
        mutated.batch_norm.truncate(mutated.num_layers);

        // Mutate layer sizes
        for size in &mut mutated.layer_sizes {
            if self.rng.gen_bool(0.2) {
                let change = self.rng.gen_range(-50..50 + 1);
                *size = ((*size as i32 + change) as usize)
                    .max(self.config.search_space.neuron_count_range.0)
                    .min(self.config.search_space.neuron_count_range.1);
            }
        }

        // Mutate activations
        for activation in &mut mutated.activations {
            if self.rng.gen_bool(0.1) {
                *activation = self
                    .config
                    .search_space
                    .activation_options
                    .choose(&mut self.rng)
                    .unwrap()
                    .clone();
            }
        }

        // Mutate dropout rates
        for dropout in &mut mutated.dropout_rates {
            if self.rng.gen_bool(0.2) {
                let change = self.rng.gen_range(-0.1..1.1);
                *dropout = (*dropout + change)
                    .max(self.config.search_space.dropout_range.0)
                    .min(self.config.search_space.dropout_range.1);
            }
        }

        // Mutate batch normalization
        for bn in &mut mutated.batch_norm {
            if self.rng.gen_bool(0.1) {
                *bn = !*bn;
            }
        }

        // Mutate skip connections
        if self.rng.gen_bool(0.1) {
            if self.rng.gen_bool(0.5) && !mutated.skip_connections.is_empty() {
                // Remove a connection
                let idx = self.rng.gen_range(0..mutated.skip_connections.len());
                mutated.skip_connections.remove(idx);
            } else if mutated.num_layers > 2 {
                // Add a connection
                let from = self.rng.gen_range(0..mutated.num_layers - 1);
                let max_to =
                    (from + self.config.search_space.max_skip_distance).min(mutated.num_layers - 1);
                if max_to > from {
                    let to = self.rng.gen_range(from + 1..max_to + 1);
                    let connection = (from, to);
                    if !mutated.skip_connections.contains(&connection) {
                        mutated.skip_connections.push(connection);
                    }
                }
            }
        }

        mutated.complexity_score =
            self.calculate_complexity_score(&mutated.layer_sizes, &mutated.skip_connections);

        Ok(mutated)
    }

    /// Calculate architecture complexity score
    fn calculate_complexity_score(
        &self,
        layer_sizes: &[usize],
        skip_connections: &[(usize, usize)],
    ) -> Float {
        let total_params = layer_sizes.iter().sum::<usize>() as Float;
        let skip_penalty = skip_connections.len() as Float * 0.1;
        total_params + skip_penalty
    }

    /// Check early stopping condition
    fn check_early_stopping(&self, convergence_curve: &[Float]) -> bool {
        if convergence_curve.len() < self.config.early_stopping_patience {
            return false;
        }

        let recent_scores =
            &convergence_curve[convergence_curve.len() - self.config.early_stopping_patience..];
        let improvement = recent_scores.last().unwrap() - recent_scores.first().unwrap();
        improvement < 1e-6
    }

    /// Generate architecture from policy weights (for RL)
    fn generate_architecture_from_policy(
        &mut self,
        _policy_weights: &HashMap<String, Float>,
    ) -> Result<NeuralArchitecture, Box<dyn std::error::Error>> {
        // Simplified policy-based generation
        // In practice, this would use learned policy weights
        self.generate_random_architecture()
    }

    /// Update policy weights based on reward (for RL)
    fn update_policy_weights(
        &mut self,
        policy_weights: &mut HashMap<String, Float>,
        architecture: &NeuralArchitecture,
        reward: Float,
        learning_rate: Float,
    ) {
        // Simplified policy update
        // In practice, this would update weights based on architecture features
        let key = format!("layers_{}", architecture.num_layers);
        let current_weight = policy_weights.get(&key).unwrap_or(&0.0);
        policy_weights.insert(key, current_weight + learning_rate * reward);
    }

    /// Initialize architecture parameters (for GDAS)
    fn initialize_architecture_parameters(
        &mut self,
    ) -> Result<HashMap<String, Float>, Box<dyn std::error::Error>> {
        let mut params = HashMap::new();
        params.insert("layer_weight".to_string(), 0.5);
        params.insert("activation_weight".to_string(), 0.5);
        params.insert("dropout_weight".to_string(), 0.5);
        Ok(params)
    }

    /// Sample architecture from parameters (for GDAS)
    fn sample_architecture_from_params(
        &mut self,
        _params: &HashMap<String, Float>,
    ) -> Result<NeuralArchitecture, Box<dyn std::error::Error>> {
        // Simplified sampling based on parameters
        self.generate_random_architecture()
    }

    /// Update architecture parameters (for GDAS)
    fn update_architecture_parameters(
        &mut self,
        params: &mut HashMap<String, Float>,
        evaluation: &ArchitectureEvaluation,
        learning_rate: Float,
    ) {
        // Simplified parameter update
        for (_key, value) in params.iter_mut() {
            *value += learning_rate * evaluation.validation_score * 0.01;
        }
    }

    /// Generate architecture with complexity constraint
    fn generate_architecture_with_complexity(
        &mut self,
        max_complexity: Float,
    ) -> Result<NeuralArchitecture, Box<dyn std::error::Error>> {
        let mut architecture = self.generate_random_architecture()?;

        // Adjust architecture to meet complexity constraint
        while architecture.complexity_score > max_complexity {
            if architecture.num_layers > 1 {
                architecture.num_layers -= 1;
                architecture.layer_sizes.pop();
                architecture.activations.pop();
                architecture.dropout_rates.pop();
                architecture.batch_norm.pop();
            }

            // Reduce layer sizes
            for size in &mut architecture.layer_sizes {
                *size = ((*size as Float) * 0.8) as usize;
            }

            architecture.complexity_score = self.calculate_complexity_score(
                &architecture.layer_sizes,
                &architecture.skip_connections,
            );
        }

        Ok(architecture)
    }

    /// Select next architecture using Bayesian optimization
    fn select_next_architecture_bayesian(
        &mut self,
        evaluated_architectures: &[(NeuralArchitecture, Float)],
    ) -> Result<NeuralArchitecture, Box<dyn std::error::Error>> {
        // Simplified Bayesian optimization - in practice would use GP surrogate model
        // For now, generate random architecture with slight bias towards better performing ones
        let mut architecture = self.generate_random_architecture()?;

        // Bias towards architectures similar to best performing ones
        if !evaluated_architectures.is_empty() {
            let best_score = evaluated_architectures
                .iter()
                .map(|(_, score)| *score)
                .fold(Float::NEG_INFINITY, |a, b| a.max(b));

            for (arch, score) in evaluated_architectures {
                if *score > best_score * 0.9 {
                    // Slightly bias towards similar architectures
                    if self.rng.gen_bool(0.3) {
                        architecture.num_layers = arch.num_layers;
                    }
                    break;
                }
            }
        }

        Ok(architecture)
    }
}

impl NeuralArchitecture {
    /// Calculate the number of parameters in the architecture
    pub fn parameter_count(&self) -> usize {
        if self.layer_sizes.is_empty() {
            return 0;
        }

        let mut total = 0;
        for i in 0..self.layer_sizes.len() - 1 {
            total += self.layer_sizes[i] * self.layer_sizes[i + 1];
        }
        total
    }

    /// Calculate estimated FLOPs for the architecture
    pub fn estimated_flops(&self) -> usize {
        self.parameter_count() * 2 // Rough estimate
    }

    /// Get architecture summary
    pub fn summary(&self) -> String {
        format!(
            "Architecture: {} layers, {} parameters, complexity: {:.2}",
            self.num_layers,
            self.parameter_count(),
            self.complexity_score
        )
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nas_optimizer_creation() {
        let config = NASConfig::default();
        let optimizer = NASOptimizer::new(config);
        assert!(optimizer.config.search_space.layer_count_range.0 > 0);
    }

    #[test]
    fn test_random_architecture_generation() {
        let config = NASConfig::default();
        let mut optimizer = NASOptimizer::new(config);

        let architecture = optimizer.generate_random_architecture().unwrap();
        assert!(architecture.num_layers > 0);
        assert!(architecture.layer_sizes.len() == architecture.num_layers);
        assert!(architecture.activations.len() == architecture.num_layers);
    }

    #[test]
    fn test_architecture_complexity_calculation() {
        let config = NASConfig::default();
        let optimizer = NASOptimizer::new(config);

        let layer_sizes = vec![64, 128, 32];
        let skip_connections = vec![(0, 2)];

        let complexity = optimizer.calculate_complexity_score(&layer_sizes, &skip_connections);
        assert!(complexity > 0.0);
    }

    #[test]
    fn test_architecture_parameter_count() {
        let architecture = NeuralArchitecture {
            num_layers: 3,
            layer_sizes: vec![64, 128, 32],
            activations: vec![
                "relu".to_string(),
                "relu".to_string(),
                "sigmoid".to_string(),
            ],
            dropout_rates: vec![0.1, 0.2, 0.0],
            batch_norm: vec![true, true, false],
            skip_connections: vec![],
            complexity_score: 224.0,
        };

        let param_count = architecture.parameter_count();
        assert_eq!(param_count, 64 * 128 + 128 * 32); // 8192 + 4096 = 12288
    }

    #[test]
    fn test_nas_random_search() {
        let config = NASConfig {
            strategy: NASStrategy::RandomSearch {
                n_trials: 5,
                max_depth: 5,
            },
            ..Default::default()
        };
        let mut optimizer = NASOptimizer::new(config);

        let evaluation_fn = |arch: &NeuralArchitecture| -> Result<ArchitectureEvaluation, Box<dyn std::error::Error>> {
            Ok(ArchitectureEvaluation {
                architecture: arch.clone(),
                validation_score: 0.8,
                training_time: 10.0,
                parameters_count: arch.parameter_count(),
                flops: arch.estimated_flops(),
                memory_usage: 100.0,
            })
        };

        let result = optimizer.search(evaluation_fn).unwrap();
        assert!(result.best_score > 0.0);
        assert_eq!(result.architectures_evaluated, 5);
    }
}
