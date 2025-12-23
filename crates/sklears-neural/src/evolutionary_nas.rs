//! Evolutionary Neural Architecture Search (ENAS) using genetic algorithms.
//!
//! This module implements evolutionary approaches to neural architecture search:
//! - Genetic algorithm-based architecture evolution
//! - Mutation and crossover operators for architectures
//! - Multi-objective optimization (accuracy, efficiency, size)
//! - Population-based architecture search
//! - Fitness evaluation and selection strategies
//! - Architecture encoding and decoding

use crate::{nas::OperationType, NeuralResult};
use scirs2_core::random::{thread_rng, CoreRandom, Rng};
use sklears_core::{error::SklearsError, types::FloatBounds};
use std::collections::HashSet;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Architecture genome representation
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ArchitectureGenome {
    /// Sequence of operations
    pub operations: Vec<OperationType>,
    /// Connection pattern (adjacency list)
    pub connections: Vec<Vec<usize>>,
    /// Number of nodes
    pub num_nodes: usize,
}

impl ArchitectureGenome {
    /// Create a new random genome
    pub fn random(num_nodes: usize) -> Self {
        let mut rng = thread_rng();
        let available_ops = OperationType::all_operations();

        let mut operations = Vec::new();
        for _ in 0..num_nodes {
            let op_idx = rng.random_range(0..available_ops.len());
            operations.push(available_ops[op_idx]);
        }

        // Create random connections (DAG)
        let mut connections = vec![Vec::new(); num_nodes];
        for (i, conn) in connections.iter_mut().enumerate().skip(1) {
            // Each node connects to a subset of previous nodes
            let num_inputs = rng.random_range(1..=i.min(3));
            let mut selected = HashSet::new();

            while selected.len() < num_inputs {
                let input_node = rng.random_range(0..i);
                selected.insert(input_node);
            }

            *conn = selected.into_iter().collect();
            conn.sort_unstable();
        }

        Self {
            operations,
            connections,
            num_nodes,
        }
    }

    /// Mutate the genome
    pub fn mutate(&mut self, mutation_rate: f64) {
        let mut rng = thread_rng();
        let available_ops = OperationType::all_operations();

        // Mutate operations
        for op in &mut self.operations {
            if rng.random::<f64>() < mutation_rate {
                let new_op_idx = rng.random_range(0..available_ops.len());
                *op = available_ops[new_op_idx];
            }
        }

        // Mutate connections
        for (i, conns) in self.connections.iter_mut().enumerate() {
            if i == 0 {
                continue; // First node has no inputs
            }

            if rng.random::<f64>() < mutation_rate && !conns.is_empty() {
                // Change one connection
                let idx_to_change = rng.random_range(0..conns.len());

                // Only mutate if there are available alternatives
                // (avoid infinite loop when conns.len() == i)
                if conns.len() < i {
                    let mut new_conn = rng.random_range(0..i);

                    // Make sure it's different
                    while conns.contains(&new_conn) {
                        new_conn = rng.random_range(0..i);
                    }

                    conns[idx_to_change] = new_conn;
                    conns.sort_unstable();
                }
            }
        }
    }

    /// Crossover with another genome
    pub fn crossover(
        &self,
        other: &ArchitectureGenome,
    ) -> (ArchitectureGenome, ArchitectureGenome) {
        let mut rng = thread_rng();

        // Single-point crossover
        let crossover_point = rng.random_range(1..self.num_nodes);

        let mut child1_ops = self.operations[..crossover_point].to_vec();
        child1_ops.extend_from_slice(&other.operations[crossover_point..]);

        let mut child2_ops = other.operations[..crossover_point].to_vec();
        child2_ops.extend_from_slice(&self.operations[crossover_point..]);

        let mut child1_conns = self.connections[..crossover_point].to_vec();
        child1_conns.extend_from_slice(&other.connections[crossover_point..]);

        let mut child2_conns = other.connections[..crossover_point].to_vec();
        child2_conns.extend_from_slice(&self.connections[crossover_point..]);

        let child1 = ArchitectureGenome {
            operations: child1_ops,
            connections: child1_conns,
            num_nodes: self.num_nodes,
        };

        let child2 = ArchitectureGenome {
            operations: child2_ops,
            connections: child2_conns,
            num_nodes: other.num_nodes,
        };

        (child1, child2)
    }

    /// Compute complexity score
    pub fn complexity(&self) -> f64 {
        let mut score = 0.0;

        for op in &self.operations {
            score += match op {
                OperationType::None => 0.0,
                OperationType::Skip => 0.1,
                OperationType::MaxPool3x3 | OperationType::AvgPool3x3 => 0.5,
                OperationType::SepConv3x3 => 1.0,
                OperationType::SepConv5x5 => 1.5,
                OperationType::DilConv3x3 => 1.2,
                OperationType::DilConv5x5 => 1.8,
            };
        }

        // Add connection complexity
        let total_connections: usize = self.connections.iter().map(|c| c.len()).sum();
        score += total_connections as f64 * 0.1;

        score
    }

    /// Get number of parameters (estimated)
    pub fn num_parameters(&self, hidden_dim: usize) -> usize {
        let mut params = 0;

        for (i, op) in self.operations.iter().enumerate() {
            let n_inputs = if i == 0 { 1 } else { self.connections[i].len() };

            params += match op {
                OperationType::None => 0,
                OperationType::Skip => 0,
                OperationType::MaxPool3x3 | OperationType::AvgPool3x3 => 0,
                OperationType::SepConv3x3 => n_inputs * hidden_dim * 3 * 3,
                OperationType::SepConv5x5 => n_inputs * hidden_dim * 5 * 5,
                OperationType::DilConv3x3 => n_inputs * hidden_dim * 3 * 3,
                OperationType::DilConv5x5 => n_inputs * hidden_dim * 5 * 5,
            };
        }

        params
    }
}

/// Fitness metrics for architecture evaluation
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Fitness {
    /// Accuracy score
    pub accuracy: f64,
    /// Model complexity (lower is better)
    pub complexity: f64,
    /// Number of parameters (lower is better)
    pub num_parameters: usize,
    /// Inference latency in milliseconds (lower is better)
    pub latency_ms: f64,
}

impl Fitness {
    /// Create new fitness
    pub fn new(accuracy: f64, complexity: f64, num_parameters: usize, latency_ms: f64) -> Self {
        Self {
            accuracy,
            complexity,
            num_parameters,
            latency_ms,
        }
    }

    /// Compute weighted fitness score
    pub fn score(&self, weights: &FitnessWeights) -> f64 {
        weights.accuracy * self.accuracy
            - weights.complexity * self.complexity
            - weights.params * (self.num_parameters as f64 / 1_000_000.0)
            - weights.latency * self.latency_ms
    }

    /// Dominates another fitness (Pareto dominance)
    pub fn dominates(&self, other: &Fitness) -> bool {
        let better_or_equal = self.accuracy >= other.accuracy
            && self.complexity <= other.complexity
            && self.num_parameters <= other.num_parameters
            && self.latency_ms <= other.latency_ms;

        let strictly_better = self.accuracy > other.accuracy
            || self.complexity < other.complexity
            || self.num_parameters < other.num_parameters
            || self.latency_ms < other.latency_ms;

        better_or_equal && strictly_better
    }
}

/// Weights for multi-objective fitness
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct FitnessWeights {
    /// Weight for accuracy
    pub accuracy: f64,
    /// Weight for complexity penalty
    pub complexity: f64,
    /// Weight for parameter count penalty
    pub params: f64,
    /// Weight for latency penalty
    pub latency: f64,
}

impl Default for FitnessWeights {
    fn default() -> Self {
        Self {
            accuracy: 1.0,
            complexity: 0.1,
            params: 0.1,
            latency: 0.1,
        }
    }
}

/// Individual in the evolutionary population
#[derive(Debug, Clone)]
pub struct Individual {
    /// Genome
    pub genome: ArchitectureGenome,
    /// Fitness
    pub fitness: Option<Fitness>,
    /// Age (generations survived)
    pub age: usize,
}

impl Individual {
    /// Create new individual
    pub fn new(genome: ArchitectureGenome) -> Self {
        Self {
            genome,
            fitness: None,
            age: 0,
        }
    }

    /// Create random individual
    pub fn random(num_nodes: usize) -> Self {
        Self::new(ArchitectureGenome::random(num_nodes))
    }

    /// Set fitness
    pub fn set_fitness(&mut self, fitness: Fitness) {
        self.fitness = Some(fitness);
    }

    /// Increment age
    pub fn increment_age(&mut self) {
        self.age += 1;
    }
}

/// Selection strategy for evolutionary algorithm
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum SelectionStrategy {
    /// Tournament selection
    Tournament { size: usize },
    /// Roulette wheel selection
    RouletteWheel,
    /// Rank-based selection
    Rank,
    /// Elitism (keep top k)
    Elitism { top_k: usize },
}

/// Evolutionary NAS configuration
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct EvolutionaryNASConfig {
    /// Population size
    pub population_size: usize,
    /// Number of generations
    pub num_generations: usize,
    /// Mutation rate
    pub mutation_rate: f64,
    /// Crossover rate
    pub crossover_rate: f64,
    /// Number of nodes in architecture
    pub num_nodes: usize,
    /// Selection strategy
    pub selection_strategy: SelectionStrategy,
    /// Fitness weights
    pub fitness_weights: FitnessWeights,
    /// Elitism count (best individuals to preserve)
    pub elitism_count: usize,
}

impl Default for EvolutionaryNASConfig {
    fn default() -> Self {
        Self {
            population_size: 50,
            num_generations: 100,
            mutation_rate: 0.1,
            crossover_rate: 0.7,
            num_nodes: 6,
            selection_strategy: SelectionStrategy::Tournament { size: 3 },
            fitness_weights: FitnessWeights::default(),
            elitism_count: 5,
        }
    }
}

/// Evolutionary Neural Architecture Search
pub struct EvolutionaryNAS {
    /// Configuration
    config: EvolutionaryNASConfig,
    /// Current population
    population: Vec<Individual>,
    /// Best individual ever found
    best_individual: Option<Individual>,
    /// Current generation
    generation: usize,
}

impl EvolutionaryNAS {
    /// Create new evolutionary NAS
    pub fn new(config: EvolutionaryNASConfig) -> Self {
        let mut population = Vec::with_capacity(config.population_size);

        for _ in 0..config.population_size {
            population.push(Individual::random(config.num_nodes));
        }

        Self {
            config,
            population,
            best_individual: None,
            generation: 0,
        }
    }

    /// Select parent using configured strategy
    fn select_parent(&self) -> &Individual {
        let mut rng = thread_rng();

        match self.config.selection_strategy {
            SelectionStrategy::Tournament { size } => {
                let mut best: Option<&Individual> = None;
                let mut best_score = f64::NEG_INFINITY;

                for _ in 0..size {
                    let idx = rng.random_range(0..self.population.len());
                    let individual = &self.population[idx];

                    if let Some(fitness) = &individual.fitness {
                        let score = fitness.score(&self.config.fitness_weights);
                        if score > best_score {
                            best_score = score;
                            best = Some(individual);
                        }
                    }
                }

                best.unwrap_or(&self.population[0])
            }
            SelectionStrategy::RouletteWheel => {
                // Compute total fitness
                let total_fitness: f64 = self
                    .population
                    .iter()
                    .filter_map(|ind| {
                        ind.fitness
                            .map(|f| f.score(&self.config.fitness_weights).max(0.0))
                    })
                    .sum();

                if total_fitness == 0.0 {
                    return &self.population[rng.random_range(0..self.population.len())];
                }

                let mut target = rng.random::<f64>() * total_fitness;

                for individual in &self.population {
                    if let Some(fitness) = &individual.fitness {
                        let score = fitness.score(&self.config.fitness_weights).max(0.0);
                        target -= score;
                        if target <= 0.0 {
                            return individual;
                        }
                    }
                }

                &self.population[self.population.len() - 1]
            }
            SelectionStrategy::Rank => {
                // Sort by fitness and select based on rank
                let idx = rng.random_range(0..self.population.len());
                &self.population[idx]
            }
            SelectionStrategy::Elitism { top_k } => {
                let idx = rng.random_range(0..top_k.min(self.population.len()));
                &self.population[idx]
            }
        }
    }

    /// Evolve population for one generation
    pub fn evolve_generation(&mut self) -> NeuralResult<()> {
        let mut rng = thread_rng();

        // Sort population by fitness
        self.population.sort_by(|a, b| {
            let score_a = a
                .fitness
                .map(|f| f.score(&self.config.fitness_weights))
                .unwrap_or(f64::NEG_INFINITY);
            let score_b = b
                .fitness
                .map(|f| f.score(&self.config.fitness_weights))
                .unwrap_or(f64::NEG_INFINITY);

            score_b.partial_cmp(&score_a).unwrap()
        });

        // Update best individual
        if let Some(fitness) = &self.population[0].fitness {
            if self.best_individual.is_none()
                || fitness.score(&self.config.fitness_weights)
                    > self
                        .best_individual
                        .as_ref()
                        .unwrap()
                        .fitness
                        .unwrap()
                        .score(&self.config.fitness_weights)
            {
                self.best_individual = Some(self.population[0].clone());
            }
        }

        // Create new population
        let mut new_population = Vec::with_capacity(self.config.population_size);

        // Elitism: keep best individuals
        for i in 0..self.config.elitism_count.min(self.population.len()) {
            let mut elite = self.population[i].clone();
            elite.increment_age();
            new_population.push(elite);
        }

        // Generate offspring
        while new_population.len() < self.config.population_size {
            if rng.random::<f64>() < self.config.crossover_rate {
                // Crossover
                let parent1 = self.select_parent();
                let parent2 = self.select_parent();

                let (child1_genome, child2_genome) = parent1.genome.crossover(&parent2.genome);

                let mut child1 = Individual::new(child1_genome);
                child1.genome.mutate(self.config.mutation_rate);

                new_population.push(child1);

                if new_population.len() < self.config.population_size {
                    let mut child2 = Individual::new(child2_genome);
                    child2.genome.mutate(self.config.mutation_rate);
                    new_population.push(child2);
                }
            } else {
                // Mutation only
                let parent = self.select_parent();
                let mut child_genome = parent.genome.clone();
                child_genome.mutate(self.config.mutation_rate);

                new_population.push(Individual::new(child_genome));
            }
        }

        self.population = new_population;
        self.generation += 1;

        Ok(())
    }

    /// Get current population
    pub fn get_population(&self) -> &[Individual] {
        &self.population
    }

    /// Get population mutably for fitness evaluation
    pub fn get_population_mut(&mut self) -> &mut [Individual] {
        &mut self.population
    }

    /// Get best individual
    pub fn get_best_individual(&self) -> Option<&Individual> {
        self.best_individual.as_ref()
    }

    /// Get current generation
    pub fn get_generation(&self) -> usize {
        self.generation
    }

    /// Get Pareto front (non-dominated solutions)
    pub fn get_pareto_front(&self) -> Vec<Individual> {
        let mut front = Vec::new();

        for individual in &self.population {
            if let Some(fitness) = &individual.fitness {
                let mut dominated = false;

                for other in &self.population {
                    if let Some(other_fitness) = &other.fitness {
                        if other_fitness.dominates(fitness) {
                            dominated = true;
                            break;
                        }
                    }
                }

                if !dominated {
                    front.push(individual.clone());
                }
            }
        }

        front
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_architecture_genome_creation() {
        let genome = ArchitectureGenome::random(5);
        assert_eq!(genome.num_nodes, 5);
        assert_eq!(genome.operations.len(), 5);
        assert_eq!(genome.connections.len(), 5);
    }

    #[test]
    fn test_architecture_genome_mutate() {
        let mut genome = ArchitectureGenome::random(4);
        let original = genome.clone();

        genome.mutate(1.0); // 100% mutation rate

        // At least something should have changed
        assert!(
            genome.operations != original.operations || genome.connections != original.connections
        );
    }

    #[test]
    fn test_architecture_genome_crossover() {
        let genome1 = ArchitectureGenome::random(6);
        let genome2 = ArchitectureGenome::random(6);

        let (child1, child2) = genome1.crossover(&genome2);

        assert_eq!(child1.num_nodes, 6);
        assert_eq!(child2.num_nodes, 6);
    }

    #[test]
    fn test_architecture_genome_complexity() {
        let genome = ArchitectureGenome::random(5);
        let complexity = genome.complexity();

        assert!(complexity >= 0.0);
    }

    #[test]
    fn test_fitness_creation() {
        let fitness = Fitness::new(0.95, 10.5, 1_000_000, 5.2);

        assert_eq!(fitness.accuracy, 0.95);
        assert_eq!(fitness.complexity, 10.5);
        assert_eq!(fitness.num_parameters, 1_000_000);
        assert_eq!(fitness.latency_ms, 5.2);
    }

    #[test]
    fn test_fitness_score() {
        let fitness = Fitness::new(0.9, 5.0, 500_000, 3.0);
        let weights = FitnessWeights::default();

        let score = fitness.score(&weights);
        assert!(score.is_finite());
    }

    #[test]
    fn test_fitness_dominance() {
        let fitness1 = Fitness::new(0.95, 5.0, 1_000_000, 3.0);
        let fitness2 = Fitness::new(0.90, 6.0, 1_200_000, 4.0);

        assert!(fitness1.dominates(&fitness2));
        assert!(!fitness2.dominates(&fitness1));
    }

    #[test]
    fn test_individual_creation() {
        let genome = ArchitectureGenome::random(4);
        let individual = Individual::new(genome);

        assert!(individual.fitness.is_none());
        assert_eq!(individual.age, 0);
    }

    #[test]
    fn test_individual_random() {
        let individual = Individual::random(5);
        assert_eq!(individual.genome.num_nodes, 5);
    }

    #[test]
    fn test_individual_set_fitness() {
        let mut individual = Individual::random(4);
        let fitness = Fitness::new(0.85, 8.0, 800_000, 4.5);

        individual.set_fitness(fitness);
        assert!(individual.fitness.is_some());
    }

    #[test]
    fn test_evolutionary_nas_creation() {
        let config = EvolutionaryNASConfig::default();
        let nas = EvolutionaryNAS::new(config);

        assert_eq!(nas.population.len(), 50);
        assert_eq!(nas.generation, 0);
    }

    #[test]
    fn test_evolutionary_nas_evolve() {
        let config = EvolutionaryNASConfig {
            population_size: 10,
            num_generations: 5,
            num_nodes: 3,
            ..Default::default()
        };
        let mut nas = EvolutionaryNAS::new(config);

        // Set some random fitness values
        for individual in nas.get_population_mut() {
            let fitness = Fitness::new(0.7, 5.0, 500_000, 3.0);
            individual.set_fitness(fitness);
        }

        nas.evolve_generation().unwrap();

        assert_eq!(nas.get_generation(), 1);
        assert_eq!(nas.population.len(), 10);
    }

    #[test]
    fn test_evolutionary_nas_best_individual() {
        let config = EvolutionaryNASConfig {
            population_size: 5,
            num_nodes: 3,
            ..Default::default()
        };
        let mut nas = EvolutionaryNAS::new(config);

        // Set fitness for all individuals
        for (i, individual) in nas.get_population_mut().iter_mut().enumerate() {
            let fitness = Fitness::new(0.5 + i as f64 * 0.1, 5.0, 500_000, 3.0);
            individual.set_fitness(fitness);
        }

        nas.evolve_generation().unwrap();

        assert!(nas.get_best_individual().is_some());
    }

    #[test]
    fn test_pareto_front() {
        let config = EvolutionaryNASConfig {
            population_size: 5,
            num_nodes: 3,
            ..Default::default()
        };
        let mut nas = EvolutionaryNAS::new(config);

        // Create diverse fitness values
        let fitness_values = vec![
            Fitness::new(0.9, 10.0, 1_000_000, 5.0),
            Fitness::new(0.85, 8.0, 800_000, 4.0),
            Fitness::new(0.95, 12.0, 1_200_000, 6.0),
            Fitness::new(0.80, 6.0, 600_000, 3.0),
            Fitness::new(0.88, 9.0, 900_000, 4.5),
        ];

        for (individual, fitness) in nas.get_population_mut().iter_mut().zip(fitness_values) {
            individual.set_fitness(fitness);
        }

        let pareto_front = nas.get_pareto_front();
        assert!(!pareto_front.is_empty());
    }

    #[test]
    fn test_selection_strategies() {
        let config = EvolutionaryNASConfig {
            population_size: 10,
            selection_strategy: SelectionStrategy::Tournament { size: 3 },
            num_nodes: 3,
            ..Default::default()
        };
        let mut nas = EvolutionaryNAS::new(config);

        // Set fitness
        for individual in nas.get_population_mut() {
            individual.set_fitness(Fitness::new(0.7, 5.0, 500_000, 3.0));
        }

        let parent = nas.select_parent();
        assert!(parent.fitness.is_some());
    }

    #[test]
    fn test_genome_num_parameters() {
        // Create genome with specific operations to ensure deterministic parameter count
        let genome = ArchitectureGenome {
            operations: vec![
                OperationType::SepConv3x3,
                OperationType::SepConv5x5,
                OperationType::DilConv3x3,
                OperationType::Skip,
            ],
            connections: vec![
                vec![],        // node 0: no inputs
                vec![0],       // node 1: 1 input
                vec![0, 1],    // node 2: 2 inputs
                vec![1, 2],    // node 3: 2 inputs
            ],
            num_nodes: 4,
        };

        let params = genome.num_parameters(64);

        // Calculate expected parameters:
        // node 0 (SepConv3x3): 1 * 64 * 3 * 3 = 576
        // node 1 (SepConv5x5): 1 * 64 * 5 * 5 = 1600
        // node 2 (DilConv3x3): 2 * 64 * 3 * 3 = 1152
        // node 3 (Skip): 0
        // Total: 576 + 1600 + 1152 = 3328
        assert_eq!(params, 3328);

        // Also test with a random genome to ensure it doesn't panic
        let random_genome = ArchitectureGenome::random(4);
        let _random_params = random_genome.num_parameters(64);
        // random_params can be 0 if all ops are pooling/skip/none, which is valid
    }
}
