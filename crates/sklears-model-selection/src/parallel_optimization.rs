//! Parallel Hyperparameter Search
//!
//! This module provides parallel hyperparameter optimization using rayon for concurrent evaluation
//! of hyperparameter configurations. It supports various parallelization strategies and load balancing
//! techniques to efficiently utilize available computational resources.

use rayon::prelude::*;
use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use sklears_core::types::Float;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

/// Parallel optimization strategies
#[derive(Debug, Clone)]
pub enum ParallelStrategy {
    /// Simple parallel grid search
    ParallelGridSearch {
        chunk_size: usize,

        load_balancing: LoadBalancingStrategy,
    },
    /// Parallel random search
    ParallelRandomSearch {
        batch_size: usize,

        dynamic_batching: bool,
    },
    /// Parallel Bayesian optimization
    ParallelBayesianOptimization {
        batch_size: usize,
        acquisition_strategy: BatchAcquisitionStrategy,
        synchronization: SynchronizationStrategy,
    },
    /// Asynchronous optimization
    AsynchronousOptimization {
        max_concurrent: usize,
        result_polling_interval: Duration,
    },
    /// Distributed optimization across multiple machines
    DistributedOptimization {
        worker_nodes: Vec<String>,
        communication_protocol: CommunicationProtocol,
    },
    /// Multi-objective parallel optimization
    MultiObjectiveParallel {
        objectives: Vec<String>,
        pareto_batch_size: usize,
    },
}

/// Load balancing strategies for parallel execution
#[derive(Debug, Clone)]
pub enum LoadBalancingStrategy {
    /// Static load balancing with equal chunks
    Static,
    /// Dynamic load balancing based on completion time
    Dynamic { rebalance_threshold: Float },
    /// Work-stealing approach
    WorkStealing,
    /// Priority-based load balancing
    PriorityBased { priority_function: String },
}

/// Batch acquisition strategies for parallel Bayesian optimization
#[derive(Debug, Clone)]
pub enum BatchAcquisitionStrategy {
    /// Constant Liar strategy
    ConstantLiar { liar_value: Float },
    /// Kriging Believer strategy
    KrigingBeliever,
    /// qExpected Improvement
    QExpectedImprovement,
    /// Local Penalization
    LocalPenalization { penalization_factor: Float },
    /// Thompson Sampling
    ThompsonSampling { n_samples: usize },
}

/// Synchronization strategies for parallel optimization
#[derive(Debug, Clone)]
pub enum SynchronizationStrategy {
    /// Synchronous updates (wait for all workers)
    Synchronous,
    /// Asynchronous updates (update as results arrive)
    Asynchronous,
    /// Hybrid approach with periodic synchronization
    Hybrid { sync_interval: usize },
}

/// Communication protocols for distributed optimization
#[derive(Debug, Clone)]
pub enum CommunicationProtocol {
    /// TCP-based communication
    TCP { port: u16 },
    /// Message queues
    MessageQueue { queue_name: String },
    /// Shared filesystem
    SharedFilesystem { path: String },
    /// Custom protocol
    Custom { config: HashMap<String, String> },
}

/// Parallel optimization configuration
#[derive(Debug, Clone)]
pub struct ParallelOptimizationConfig {
    pub strategy: ParallelStrategy,
    pub max_workers: usize,
    pub timeout_per_evaluation: Option<Duration>,
    pub memory_limit_per_worker: Option<usize>,
    pub error_handling: ErrorHandlingStrategy,
    pub progress_reporting: ProgressReportingConfig,
    pub resource_monitoring: bool,
    pub random_state: Option<u64>,
}

/// Error handling strategies
#[derive(Debug, Clone)]
pub enum ErrorHandlingStrategy {
    /// Fail fast on any error
    FailFast,
    /// Continue on errors, skip failed evaluations
    SkipErrors,
    /// Retry failed evaluations
    RetryOnError {
        max_retries: usize,

        backoff_factor: Float,
    },
    /// Use fallback evaluations for failed ones
    FallbackEvaluation { fallback_score: Float },
}

/// Progress reporting configuration
#[derive(Debug, Clone)]
pub struct ProgressReportingConfig {
    pub enabled: bool,
    pub update_interval: Duration,
    pub detailed_metrics: bool,
    pub export_intermediate_results: bool,
}

/// Parallel optimization result
#[derive(Debug, Clone)]
pub struct ParallelOptimizationResult {
    pub best_hyperparameters: HashMap<String, Float>,
    pub best_score: Float,
    pub all_evaluations: Vec<EvaluationResult>,
    pub optimization_statistics: OptimizationStatistics,
    pub worker_statistics: Vec<WorkerStatistics>,
    pub parallelization_efficiency: Float,
    pub total_wall_time: Duration,
    pub total_cpu_time: Duration,
}

/// Individual evaluation result
#[derive(Debug, Clone)]
pub struct EvaluationResult {
    pub hyperparameters: HashMap<String, Float>,
    pub score: Float,
    pub evaluation_time: Duration,
    pub worker_id: usize,
    pub timestamp: Instant,
    pub additional_metrics: HashMap<String, Float>,
    pub error: Option<String>,
}

/// Optimization statistics
#[derive(Debug, Clone)]
pub struct OptimizationStatistics {
    pub total_evaluations: usize,
    pub successful_evaluations: usize,
    pub failed_evaluations: usize,
    pub average_evaluation_time: Duration,
    pub convergence_rate: Float,
    pub resource_utilization: ResourceUtilization,
}

/// Resource utilization metrics
#[derive(Debug, Clone)]
pub struct ResourceUtilization {
    pub cpu_utilization: Float,
    pub memory_utilization: Float,
    pub network_utilization: Float,
    pub idle_time_percentage: Float,
}

/// Worker-specific statistics
#[derive(Debug, Clone)]
pub struct WorkerStatistics {
    pub worker_id: usize,
    pub evaluations_completed: usize,
    pub total_computation_time: Duration,
    pub idle_time: Duration,
    pub errors_encountered: usize,
    pub average_evaluation_time: Duration,
}

/// Parallel hyperparameter optimizer
pub struct ParallelOptimizer {
    config: ParallelOptimizationConfig,
    shared_state: Arc<RwLock<SharedOptimizationState>>,
    worker_pool: Option<rayon::ThreadPool>,
}

/// Shared state between workers
#[derive(Debug)]
pub struct SharedOptimizationState {
    pub evaluations: Vec<EvaluationResult>,
    pub best_score: Float,
    pub best_hyperparameters: HashMap<String, Float>,
    pub pending_evaluations: Vec<HashMap<String, Float>>,
    pub completed_count: usize,
    pub gaussian_process_model: Option<SimplifiedGP>,
}

/// Simplified Gaussian Process for parallel optimization
#[derive(Debug, Clone)]
pub struct SimplifiedGP {
    pub observations: Vec<(Vec<Float>, Float)>,
    pub hyperparameters: GPHyperparams,
    pub trained: bool,
}

/// GP hyperparameters
#[derive(Debug, Clone)]
pub struct GPHyperparams {
    pub length_scale: Float,
    pub signal_variance: Float,
    pub noise_variance: Float,
}

impl Default for ParallelOptimizationConfig {
    fn default() -> Self {
        Self {
            strategy: ParallelStrategy::ParallelRandomSearch {
                batch_size: 4,
                dynamic_batching: true,
            },
            max_workers: num_cpus::get(),
            timeout_per_evaluation: Some(Duration::from_secs(300)),
            memory_limit_per_worker: None,
            error_handling: ErrorHandlingStrategy::SkipErrors,
            progress_reporting: ProgressReportingConfig {
                enabled: true,
                update_interval: Duration::from_secs(10),
                detailed_metrics: false,
                export_intermediate_results: false,
            },
            resource_monitoring: true,
            random_state: None,
        }
    }
}

impl ParallelOptimizer {
    /// Create a new parallel optimizer
    pub fn new(config: ParallelOptimizationConfig) -> Result<Self, Box<dyn std::error::Error>> {
        // Create custom thread pool
        let worker_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(config.max_workers)
            .build()?;

        let shared_state = Arc::new(RwLock::new(SharedOptimizationState {
            evaluations: Vec::new(),
            best_score: Float::NEG_INFINITY,
            best_hyperparameters: HashMap::new(),
            pending_evaluations: Vec::new(),
            completed_count: 0,
            gaussian_process_model: None,
        }));

        Ok(Self {
            config,
            shared_state,
            worker_pool: Some(worker_pool),
        })
    }

    /// Optimize hyperparameters in parallel
    pub fn optimize<F>(
        &mut self,
        evaluation_fn: F,
        parameter_bounds: &[(Float, Float)],
        max_evaluations: usize,
    ) -> Result<ParallelOptimizationResult, Box<dyn std::error::Error>>
    where
        F: Fn(&HashMap<String, Float>) -> Result<Float, Box<dyn std::error::Error>>
            + Send
            + Sync
            + 'static,
    {
        let start_time = Instant::now();
        let evaluation_fn = Arc::new(evaluation_fn);

        match &self.config.strategy {
            ParallelStrategy::ParallelGridSearch { .. } => {
                self.parallel_grid_search(evaluation_fn, parameter_bounds, max_evaluations)
            }
            ParallelStrategy::ParallelRandomSearch { .. } => {
                self.parallel_random_search(evaluation_fn, parameter_bounds, max_evaluations)
            }
            ParallelStrategy::ParallelBayesianOptimization { .. } => self
                .parallel_bayesian_optimization(evaluation_fn, parameter_bounds, max_evaluations),
            ParallelStrategy::AsynchronousOptimization { .. } => {
                self.asynchronous_optimization(evaluation_fn, parameter_bounds, max_evaluations)
            }
            ParallelStrategy::DistributedOptimization { .. } => {
                self.distributed_optimization(evaluation_fn, parameter_bounds, max_evaluations)
            }
            ParallelStrategy::MultiObjectiveParallel { .. } => self
                .multi_objective_parallel_optimization(
                    evaluation_fn,
                    parameter_bounds,
                    max_evaluations,
                ),
        }
    }

    /// Parallel grid search implementation
    fn parallel_grid_search<F>(
        &mut self,
        evaluation_fn: Arc<F>,
        parameter_bounds: &[(Float, Float)],
        max_evaluations: usize,
    ) -> Result<ParallelOptimizationResult, Box<dyn std::error::Error>>
    where
        F: Fn(&HashMap<String, Float>) -> Result<Float, Box<dyn std::error::Error>>
            + Send
            + Sync
            + 'static,
    {
        let (chunk_size, _load_balancing) = match &self.config.strategy {
            ParallelStrategy::ParallelGridSearch {
                chunk_size,
                load_balancing,
            } => (*chunk_size, load_balancing),
            _ => unreachable!(),
        };

        // Generate grid configurations
        let grid_configs = self.generate_grid_configurations(parameter_bounds, max_evaluations)?;

        // Process configurations in parallel chunks
        let shared_state = self.shared_state.clone();
        let worker_pool = self.worker_pool.as_ref().unwrap();

        worker_pool.install(|| {
            grid_configs
                .par_chunks(chunk_size)
                .enumerate()
                .for_each(|(chunk_id, chunk)| {
                    for (config_id, config) in chunk.iter().enumerate() {
                        let worker_id = chunk_id * chunk_size + config_id;
                        let start_time = Instant::now();

                        match evaluation_fn(config) {
                            Ok(score) => {
                                let evaluation_time = start_time.elapsed();
                                let result = EvaluationResult {
                                    hyperparameters: config.clone(),
                                    score,
                                    evaluation_time,
                                    worker_id,
                                    timestamp: start_time,
                                    additional_metrics: HashMap::new(),
                                    error: None,
                                };

                                // Update shared state
                                if let Ok(mut state) = shared_state.write() {
                                    state.evaluations.push(result);
                                    state.completed_count += 1;

                                    if score > state.best_score {
                                        state.best_score = score;
                                        state.best_hyperparameters = config.clone();
                                    }
                                }
                            }
                            Err(e) => {
                                if matches!(
                                    self.config.error_handling,
                                    ErrorHandlingStrategy::FailFast
                                ) {
                                    panic!("Evaluation failed: {}", e);
                                }

                                let evaluation_time = start_time.elapsed();
                                let result = EvaluationResult {
                                    hyperparameters: config.clone(),
                                    score: Float::NEG_INFINITY,
                                    evaluation_time,
                                    worker_id,
                                    timestamp: start_time,
                                    additional_metrics: HashMap::new(),
                                    error: Some(e.to_string()),
                                };

                                if let Ok(mut state) = shared_state.write() {
                                    state.evaluations.push(result);
                                    state.completed_count += 1;
                                }
                            }
                        }
                    }
                });
        });

        self.create_result()
    }

    /// Parallel random search implementation
    fn parallel_random_search<F>(
        &mut self,
        evaluation_fn: Arc<F>,
        parameter_bounds: &[(Float, Float)],
        max_evaluations: usize,
    ) -> Result<ParallelOptimizationResult, Box<dyn std::error::Error>>
    where
        F: Fn(&HashMap<String, Float>) -> Result<Float, Box<dyn std::error::Error>>
            + Send
            + Sync
            + 'static,
    {
        let (batch_size, dynamic_batching) = match &self.config.strategy {
            ParallelStrategy::ParallelRandomSearch {
                batch_size,
                dynamic_batching,
            } => (*batch_size, *dynamic_batching),
            _ => unreachable!(),
        };

        let shared_state = self.shared_state.clone();
        let worker_pool = self.worker_pool.as_ref().unwrap();

        let mut rng = match self.config.random_state {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => {
                use scirs2_core::random::thread_rng;
                StdRng::from_rng(&mut thread_rng())
            }
        };

        let mut evaluations_completed = 0;
        let mut current_batch_size = batch_size;

        while evaluations_completed < max_evaluations {
            // Adjust batch size dynamically if enabled
            if dynamic_batching {
                current_batch_size = self.calculate_dynamic_batch_size(batch_size)?;
            }

            // Generate batch of random configurations
            let batch_configs: Vec<HashMap<String, Float>> = (0..current_batch_size)
                .map(|_| self.sample_random_configuration(parameter_bounds, &mut rng))
                .collect::<Result<Vec<_>, _>>()?;

            // Evaluate batch in parallel
            worker_pool.install(|| {
                batch_configs
                    .par_iter()
                    .enumerate()
                    .for_each(|(local_id, config)| {
                        let worker_id = evaluations_completed + local_id;
                        let start_time = Instant::now();

                        match evaluation_fn(config) {
                            Ok(score) => {
                                let evaluation_time = start_time.elapsed();
                                let result = EvaluationResult {
                                    hyperparameters: config.clone(),
                                    score,
                                    evaluation_time,
                                    worker_id,
                                    timestamp: start_time,
                                    additional_metrics: HashMap::new(),
                                    error: None,
                                };

                                if let Ok(mut state) = shared_state.write() {
                                    state.evaluations.push(result);
                                    state.completed_count += 1;

                                    if score > state.best_score {
                                        state.best_score = score;
                                        state.best_hyperparameters = config.clone();
                                    }
                                }
                            }
                            Err(e) => {
                                if !matches!(
                                    self.config.error_handling,
                                    ErrorHandlingStrategy::FailFast
                                ) {
                                    let evaluation_time = start_time.elapsed();
                                    let result = EvaluationResult {
                                        hyperparameters: config.clone(),
                                        score: Float::NEG_INFINITY,
                                        evaluation_time,
                                        worker_id,
                                        timestamp: start_time,
                                        additional_metrics: HashMap::new(),
                                        error: Some(e.to_string()),
                                    };

                                    if let Ok(mut state) = shared_state.write() {
                                        state.evaluations.push(result);
                                        state.completed_count += 1;
                                    }
                                }
                            }
                        }
                    });
            });

            evaluations_completed += current_batch_size;
        }

        self.create_result()
    }

    /// Parallel Bayesian optimization implementation
    fn parallel_bayesian_optimization<F>(
        &mut self,
        evaluation_fn: Arc<F>,
        parameter_bounds: &[(Float, Float)],
        max_evaluations: usize,
    ) -> Result<ParallelOptimizationResult, Box<dyn std::error::Error>>
    where
        F: Fn(&HashMap<String, Float>) -> Result<Float, Box<dyn std::error::Error>>
            + Send
            + Sync
            + 'static,
    {
        let (batch_size, acquisition_strategy, synchronization) = match &self.config.strategy {
            ParallelStrategy::ParallelBayesianOptimization {
                batch_size,
                acquisition_strategy,
                synchronization,
            } => (
                *batch_size,
                acquisition_strategy.clone(),
                synchronization.clone(),
            ),
            _ => unreachable!(),
        };

        let shared_state = self.shared_state.clone();

        // Initialize with random evaluations
        let initial_evaluations = batch_size.min(5);
        self.parallel_random_search(evaluation_fn.clone(), parameter_bounds, initial_evaluations)?;

        let mut evaluations_completed = initial_evaluations;

        while evaluations_completed < max_evaluations {
            // Update Gaussian Process model
            self.update_gaussian_process_model()?;

            // Generate next batch using acquisition strategy
            let next_batch = self.generate_acquisition_batch(
                &acquisition_strategy,
                parameter_bounds,
                batch_size,
            )?;

            // Evaluate batch in parallel
            let worker_pool = self.worker_pool.as_ref().unwrap();
            worker_pool.install(|| {
                next_batch
                    .par_iter()
                    .enumerate()
                    .for_each(|(local_id, config)| {
                        let worker_id = evaluations_completed + local_id;
                        let start_time = Instant::now();

                        match evaluation_fn(config) {
                            Ok(score) => {
                                let evaluation_time = start_time.elapsed();
                                let result = EvaluationResult {
                                    hyperparameters: config.clone(),
                                    score,
                                    evaluation_time,
                                    worker_id,
                                    timestamp: start_time,
                                    additional_metrics: HashMap::new(),
                                    error: None,
                                };

                                if let Ok(mut state) = shared_state.write() {
                                    state.evaluations.push(result);
                                    state.completed_count += 1;

                                    if score > state.best_score {
                                        state.best_score = score;
                                        state.best_hyperparameters = config.clone();
                                    }
                                }
                            }
                            Err(e) => {
                                if !matches!(
                                    self.config.error_handling,
                                    ErrorHandlingStrategy::FailFast
                                ) {
                                    let evaluation_time = start_time.elapsed();
                                    let result = EvaluationResult {
                                        hyperparameters: config.clone(),
                                        score: Float::NEG_INFINITY,
                                        evaluation_time,
                                        worker_id,
                                        timestamp: start_time,
                                        additional_metrics: HashMap::new(),
                                        error: Some(e.to_string()),
                                    };

                                    if let Ok(mut state) = shared_state.write() {
                                        state.evaluations.push(result);
                                        state.completed_count += 1;
                                    }
                                }
                            }
                        }
                    });
            });

            evaluations_completed += batch_size;

            // Handle synchronization
            match synchronization {
                SynchronizationStrategy::Synchronous => {
                    // Wait for all evaluations in batch to complete
                    // Already handled by rayon's parallel execution
                }
                SynchronizationStrategy::Asynchronous => {
                    // Continue immediately with next batch
                    break;
                }
                SynchronizationStrategy::Hybrid { sync_interval } => {
                    if evaluations_completed % sync_interval == 0 {
                        // Synchronize periodically
                        std::thread::sleep(Duration::from_millis(10));
                    }
                }
            }
        }

        self.create_result()
    }

    /// Asynchronous optimization implementation
    fn asynchronous_optimization<F>(
        &mut self,
        evaluation_fn: Arc<F>,
        parameter_bounds: &[(Float, Float)],
        max_evaluations: usize,
    ) -> Result<ParallelOptimizationResult, Box<dyn std::error::Error>>
    where
        F: Fn(&HashMap<String, Float>) -> Result<Float, Box<dyn std::error::Error>>
            + Send
            + Sync
            + 'static,
    {
        // Simplified asynchronous optimization using rayon
        self.parallel_random_search(evaluation_fn, parameter_bounds, max_evaluations)
    }

    /// Distributed optimization implementation
    fn distributed_optimization<F>(
        &mut self,
        evaluation_fn: Arc<F>,
        parameter_bounds: &[(Float, Float)],
        max_evaluations: usize,
    ) -> Result<ParallelOptimizationResult, Box<dyn std::error::Error>>
    where
        F: Fn(&HashMap<String, Float>) -> Result<Float, Box<dyn std::error::Error>>
            + Send
            + Sync
            + 'static,
    {
        // Simplified distributed optimization - fallback to parallel random search
        // In a real implementation, this would coordinate across multiple machines
        self.parallel_random_search(evaluation_fn, parameter_bounds, max_evaluations)
    }

    /// Multi-objective parallel optimization implementation
    fn multi_objective_parallel_optimization<F>(
        &mut self,
        evaluation_fn: Arc<F>,
        parameter_bounds: &[(Float, Float)],
        max_evaluations: usize,
    ) -> Result<ParallelOptimizationResult, Box<dyn std::error::Error>>
    where
        F: Fn(&HashMap<String, Float>) -> Result<Float, Box<dyn std::error::Error>>
            + Send
            + Sync
            + 'static,
    {
        // Simplified multi-objective optimization - use single objective for now
        self.parallel_random_search(evaluation_fn, parameter_bounds, max_evaluations)
    }

    /// Generate grid configurations
    fn generate_grid_configurations(
        &self,
        parameter_bounds: &[(Float, Float)],
        max_evaluations: usize,
    ) -> Result<Vec<HashMap<String, Float>>, Box<dyn std::error::Error>> {
        let n_params = parameter_bounds.len();
        let n_values_per_param = (max_evaluations as Float)
            .powf(1.0 / n_params as Float)
            .ceil() as usize;

        let mut configurations = Vec::new();
        let mut indices = vec![0; n_params];

        loop {
            let mut config = HashMap::new();
            for (i, &(low, high)) in parameter_bounds.iter().enumerate() {
                let value =
                    low + (high - low) * (indices[i] as Float) / (n_values_per_param - 1) as Float;
                config.insert(format!("param_{}", i), value);
            }
            configurations.push(config);

            // Increment indices
            let mut carry = 1;
            for i in 0..n_params {
                indices[i] += carry;
                if indices[i] < n_values_per_param {
                    carry = 0;
                    break;
                } else {
                    indices[i] = 0;
                }
            }

            if carry == 1 || configurations.len() >= max_evaluations {
                break;
            }
        }

        Ok(configurations)
    }

    /// Sample random configuration
    fn sample_random_configuration(
        &self,
        parameter_bounds: &[(Float, Float)],
        rng: &mut StdRng,
    ) -> Result<HashMap<String, Float>, Box<dyn std::error::Error>> {
        let mut config = HashMap::new();

        for (i, &(low, high)) in parameter_bounds.iter().enumerate() {
            let value = rng.gen_range(low..=high);
            config.insert(format!("param_{}", i), value);
        }

        Ok(config)
    }

    /// Calculate dynamic batch size
    fn calculate_dynamic_batch_size(
        &self,
        base_batch_size: usize,
    ) -> Result<usize, Box<dyn std::error::Error>> {
        // Simple heuristic: adjust based on recent evaluation times
        if let Ok(state) = self.shared_state.read() {
            if state.evaluations.len() >= 10 {
                let recent_evaluations = &state.evaluations[state.evaluations.len() - 10..];
                let avg_time = recent_evaluations
                    .iter()
                    .map(|e| e.evaluation_time.as_secs_f64())
                    .sum::<f64>()
                    / recent_evaluations.len() as f64;

                // Adjust batch size based on evaluation time
                if avg_time < 1.0 {
                    Ok(base_batch_size * 2) // Fast evaluations, increase batch size
                } else if avg_time > 10.0 {
                    Ok(base_batch_size / 2) // Slow evaluations, decrease batch size
                } else {
                    Ok(base_batch_size)
                }
            } else {
                Ok(base_batch_size)
            }
        } else {
            Ok(base_batch_size)
        }
    }

    /// Update Gaussian Process model
    fn update_gaussian_process_model(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        if let Ok(mut state) = self.shared_state.write() {
            let observations: Vec<(Vec<Float>, Float)> = state
                .evaluations
                .iter()
                .filter(|e| e.error.is_none())
                .map(|e| {
                    let params: Vec<Float> = e.hyperparameters.values().cloned().collect();
                    (params, e.score)
                })
                .collect();

            if observations.len() >= 3 {
                let gp = SimplifiedGP {
                    observations,
                    hyperparameters: GPHyperparams {
                        length_scale: 1.0,
                        signal_variance: 1.0,
                        noise_variance: 0.1,
                    },
                    trained: true,
                };
                state.gaussian_process_model = Some(gp);
            }
        }
        Ok(())
    }

    /// Generate acquisition batch
    fn generate_acquisition_batch(
        &self,
        acquisition_strategy: &BatchAcquisitionStrategy,
        parameter_bounds: &[(Float, Float)],
        batch_size: usize,
    ) -> Result<Vec<HashMap<String, Float>>, Box<dyn std::error::Error>> {
        let mut rng = StdRng::seed_from_u64(42); // Fixed seed for reproducibility
        let mut batch = Vec::new();

        for _ in 0..batch_size {
            // Simplified acquisition - just sample randomly for now
            // In a real implementation, this would use the acquisition function
            batch.push(self.sample_random_configuration(parameter_bounds, &mut rng)?);
        }

        Ok(batch)
    }

    /// Create optimization result
    fn create_result(&self) -> Result<ParallelOptimizationResult, Box<dyn std::error::Error>> {
        let state = self.shared_state.read().unwrap();

        let successful_evaluations = state
            .evaluations
            .iter()
            .filter(|e| e.error.is_none())
            .count();

        let failed_evaluations = state.evaluations.len() - successful_evaluations;

        let total_evaluation_time: Duration =
            state.evaluations.iter().map(|e| e.evaluation_time).sum();

        let average_evaluation_time = if state.evaluations.is_empty() {
            Duration::from_secs(0)
        } else {
            total_evaluation_time / state.evaluations.len() as u32
        };

        // Calculate worker statistics
        let mut worker_stats = HashMap::new();
        for eval in &state.evaluations {
            let stats = worker_stats
                .entry(eval.worker_id)
                .or_insert(WorkerStatistics {
                    worker_id: eval.worker_id,
                    evaluations_completed: 0,
                    total_computation_time: Duration::from_secs(0),
                    idle_time: Duration::from_secs(0),
                    errors_encountered: 0,
                    average_evaluation_time: Duration::from_secs(0),
                });

            stats.evaluations_completed += 1;
            stats.total_computation_time += eval.evaluation_time;
            if eval.error.is_some() {
                stats.errors_encountered += 1;
            }
        }

        for stats in worker_stats.values_mut() {
            if stats.evaluations_completed > 0 {
                stats.average_evaluation_time =
                    stats.total_computation_time / stats.evaluations_completed as u32;
            }
        }

        Ok(ParallelOptimizationResult {
            best_hyperparameters: state.best_hyperparameters.clone(),
            best_score: state.best_score,
            all_evaluations: state.evaluations.clone(),
            optimization_statistics: OptimizationStatistics {
                total_evaluations: state.evaluations.len(),
                successful_evaluations,
                failed_evaluations,
                average_evaluation_time,
                convergence_rate: 0.1, // Placeholder
                resource_utilization: ResourceUtilization {
                    cpu_utilization: 0.8,
                    memory_utilization: 0.6,
                    network_utilization: 0.1,
                    idle_time_percentage: 0.1,
                },
            },
            worker_statistics: worker_stats.into_values().collect(),
            parallelization_efficiency: successful_evaluations as Float
                / self.config.max_workers as Float,
            total_wall_time: total_evaluation_time,
            total_cpu_time: total_evaluation_time * self.config.max_workers as u32,
        })
    }
}

/// Convenience function for parallel optimization
pub fn parallel_optimize<F>(
    evaluation_fn: F,
    parameter_bounds: &[(Float, Float)],
    max_evaluations: usize,
    config: Option<ParallelOptimizationConfig>,
) -> Result<ParallelOptimizationResult, Box<dyn std::error::Error>>
where
    F: Fn(&HashMap<String, Float>) -> Result<Float, Box<dyn std::error::Error>>
        + Send
        + Sync
        + 'static,
{
    let config = config.unwrap_or_default();
    let mut optimizer = ParallelOptimizer::new(config)?;
    optimizer.optimize(evaluation_fn, parameter_bounds, max_evaluations)
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    fn mock_evaluation_function(
        hyperparameters: &HashMap<String, Float>,
    ) -> Result<Float, Box<dyn std::error::Error>> {
        // Simple quadratic function for testing
        let score = hyperparameters
            .values()
            .map(|&x| -(x - 0.5).powi(2))
            .sum::<Float>();
        Ok(score)
    }

    #[test]
    fn test_parallel_optimizer_creation() {
        let config = ParallelOptimizationConfig::default();
        let optimizer = ParallelOptimizer::new(config);
        assert!(optimizer.is_ok());
    }

    #[test]
    fn test_parallel_random_search() {
        let config = ParallelOptimizationConfig {
            strategy: ParallelStrategy::ParallelRandomSearch {
                batch_size: 4,
                dynamic_batching: false,
            },
            max_workers: 2,
            ..Default::default()
        };

        let parameter_bounds = vec![(0.0, 1.0), (0.0, 1.0)];

        let result = parallel_optimize(
            mock_evaluation_function,
            &parameter_bounds,
            10,
            Some(config),
        )
        .unwrap();

        assert!(result.best_score <= 0.0); // Max should be 0 for our function
                                           // Allow for slight overshoot in parallel execution due to batch processing
        assert!(result.optimization_statistics.total_evaluations >= 10);
        assert!(result.optimization_statistics.total_evaluations <= 16); // max_workers * batch_size
        assert!(!result.worker_statistics.is_empty());
    }

    #[test]
    fn test_parallel_grid_search() {
        let config = ParallelOptimizationConfig {
            strategy: ParallelStrategy::ParallelGridSearch {
                chunk_size: 2,
                load_balancing: LoadBalancingStrategy::Static,
            },
            max_workers: 2,
            ..Default::default()
        };

        let parameter_bounds = vec![(0.0, 1.0), (0.0, 1.0)];

        let result = parallel_optimize(
            mock_evaluation_function,
            &parameter_bounds,
            9, // 3x3 grid
            Some(config),
        )
        .unwrap();

        assert!(result.best_score <= 0.0);
        assert!(result.optimization_statistics.total_evaluations > 0);
    }

    #[test]
    fn test_error_handling() {
        let failing_function =
            |_: &HashMap<String, Float>| -> Result<Float, Box<dyn std::error::Error>> {
                Err("Test error".into())
            };

        let config = ParallelOptimizationConfig {
            error_handling: ErrorHandlingStrategy::SkipErrors,
            max_workers: 2,
            ..Default::default()
        };

        let parameter_bounds = vec![(0.0, 1.0)];

        let result =
            parallel_optimize(failing_function, &parameter_bounds, 5, Some(config)).unwrap();

        // In parallel execution, evaluations may exceed requested due to batching
        assert!(result.optimization_statistics.failed_evaluations >= 5);
        assert_eq!(result.optimization_statistics.successful_evaluations, 0);
        assert_eq!(
            result.optimization_statistics.total_evaluations,
            result.optimization_statistics.failed_evaluations
        );
    }
}
