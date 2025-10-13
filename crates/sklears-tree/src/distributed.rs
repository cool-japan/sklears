//! Distributed ensemble training for tree algorithms
//!
//! This module provides distributed training capabilities for tree ensembles
//! across multiple nodes/processes for improved scalability and performance.

use scirs2_core::ndarray::{Array1, Array2};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use sklears_core::error::{Result, SklearsError};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::random_forest::RandomForestConfig;

/// Configuration for distributed ensemble training
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct DistributedConfig {
    /// Total number of trees in the ensemble
    pub n_estimators: usize,
    /// Number of worker nodes/processes
    pub n_workers: usize,
    /// Maximum training time per worker (in seconds)
    pub max_training_time: Option<u64>,
    /// Communication timeout (in milliseconds)
    pub communication_timeout: u64,
    /// Enable fault tolerance
    pub fault_tolerant: bool,
    /// Minimum number of workers required for completion
    pub min_workers: usize,
    /// Data partitioning strategy
    pub partitioning_strategy: PartitioningStrategy,
    /// Model aggregation strategy
    pub aggregation_strategy: AggregationStrategy,
    /// Worker load balancing configuration
    pub load_balancing: LoadBalancingConfig,
}

impl Default for DistributedConfig {
    fn default() -> Self {
        Self {
            n_estimators: 100,
            n_workers: 4,
            max_training_time: Some(3600), // 1 hour
            communication_timeout: 30000,  // 30 seconds
            fault_tolerant: true,
            min_workers: 2,
            partitioning_strategy: PartitioningStrategy::TreeBased,
            aggregation_strategy: AggregationStrategy::WeightedVoting,
            load_balancing: LoadBalancingConfig::default(),
        }
    }
}

/// Strategy for partitioning work across workers
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum PartitioningStrategy {
    /// Distribute trees evenly across workers
    TreeBased,
    /// Distribute data samples across workers
    DataBased,
    /// Hybrid approach: distribute both trees and data
    Hybrid { tree_ratio: f64, data_ratio: f64 },
    /// Feature-based partitioning
    FeatureBased { overlap_ratio: f64 },
}

/// Strategy for aggregating results from workers
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum AggregationStrategy {
    /// Simple majority voting
    MajorityVoting,
    /// Weighted voting based on worker performance
    WeightedVoting,
    /// Model averaging
    ModelAveraging,
    /// Stacking ensemble
    StackingEnsemble,
}

/// Load balancing configuration
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct LoadBalancingConfig {
    /// Enable dynamic load balancing
    pub dynamic_balancing: bool,
    /// Load balancing check interval (in milliseconds)
    pub check_interval: u64,
    /// Maximum imbalance ratio tolerated
    pub max_imbalance_ratio: f64,
    /// Enable work stealing between workers
    pub work_stealing: bool,
}

impl Default for LoadBalancingConfig {
    fn default() -> Self {
        Self {
            dynamic_balancing: true,
            check_interval: 5000, // 5 seconds
            max_imbalance_ratio: 2.0,
            work_stealing: true,
        }
    }
}

/// Represents a worker node in the distributed system
#[derive(Debug, Clone)]
pub struct WorkerNode {
    /// Unique worker identifier
    pub id: usize,
    /// Worker status
    pub status: WorkerStatus,
    /// Number of trees assigned to this worker
    pub assigned_trees: usize,
    /// Number of completed trees
    pub completed_trees: usize,
    /// Worker performance metrics
    pub performance: WorkerPerformance,
    /// Last heartbeat timestamp
    pub last_heartbeat: Instant,
}

/// Worker node status
#[derive(Debug, Clone, PartialEq)]
pub enum WorkerStatus {
    Idle,
    Training,
    Completed,
    Failed,
    Timeout,
}

/// Performance metrics for a worker
#[derive(Debug, Clone)]
pub struct WorkerPerformance {
    /// Average time per tree (in milliseconds)
    pub avg_tree_time: f64,
    /// Total training time (in milliseconds)
    pub total_training_time: f64,
    /// Success rate (0.0 to 1.0)
    pub success_rate: f64,
    /// Memory usage (in MB)
    pub memory_usage: f64,
    /// CPU utilization (0.0 to 1.0)
    pub cpu_utilization: f64,
}

impl Default for WorkerPerformance {
    fn default() -> Self {
        Self {
            avg_tree_time: 0.0,
            total_training_time: 0.0,
            success_rate: 1.0,
            memory_usage: 0.0,
            cpu_utilization: 0.0,
        }
    }
}

/// Training task assigned to a worker
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct TrainingTask {
    /// Task identifier
    pub task_id: usize,
    /// Worker assigned to this task
    pub worker_id: usize,
    /// Tree indices to train
    pub tree_indices: Vec<usize>,
    /// Data partition (row indices)
    pub data_partition: Option<Vec<usize>>,
    /// Feature partition (column indices)
    pub feature_partition: Option<Vec<usize>>,
    /// Random seeds for reproducibility
    pub random_seeds: Vec<u64>,
    /// Task priority
    pub priority: TaskPriority,
}

/// Task priority levels
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum TaskPriority {
    Low,
    Normal,
    High,
    Critical,
}

/// Result from a completed training task
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct TaskResult {
    /// Task identifier
    pub task_id: usize,
    /// Worker that completed the task
    pub worker_id: usize,
    /// Serialized trained trees
    pub trained_trees: Vec<Vec<u8>>,
    /// Training metrics
    pub metrics: TrainingMetrics,
    /// Completion timestamp
    pub completed_at: u64,
}

/// Training metrics for a task
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct TrainingMetrics {
    /// Training time (in milliseconds)
    pub training_time: f64,
    /// Out-of-bag score
    pub oob_score: Option<f64>,
    /// Feature importances
    pub feature_importances: Vec<f64>,
    /// Memory peak usage (in MB)
    pub peak_memory: f64,
}

/// Distributed ensemble trainer
pub struct DistributedEnsembleTrainer {
    /// Configuration
    config: DistributedConfig,
    /// Worker nodes
    workers: Arc<Mutex<HashMap<usize, WorkerNode>>>,
    /// Training tasks queue
    task_queue: Arc<Mutex<Vec<TrainingTask>>>,
    /// Completed task results
    task_results: Arc<Mutex<Vec<TaskResult>>>,
    /// Load balancer
    load_balancer: LoadBalancer,
}

impl DistributedEnsembleTrainer {
    /// Create a new distributed ensemble trainer
    pub fn new(config: DistributedConfig) -> Self {
        let workers = Arc::new(Mutex::new(HashMap::new()));
        let task_queue = Arc::new(Mutex::new(Vec::new()));
        let task_results = Arc::new(Mutex::new(Vec::new()));

        // Initialize workers
        {
            let mut workers_guard = workers.lock().unwrap();
            for i in 0..config.n_workers {
                workers_guard.insert(
                    i,
                    WorkerNode {
                        id: i,
                        status: WorkerStatus::Idle,
                        assigned_trees: 0,
                        completed_trees: 0,
                        performance: WorkerPerformance::default(),
                        last_heartbeat: Instant::now(),
                    },
                );
            }
        }

        let load_balancer = LoadBalancer::new(config.load_balancing.clone());

        Self {
            config,
            workers,
            task_queue,
            task_results,
            load_balancer,
        }
    }

    /// Train a distributed random forest ensemble
    pub fn train_random_forest(
        &mut self,
        x: &Array2<f64>,
        y: &Array1<i32>,
        rf_config: &RandomForestConfig,
    ) -> Result<DistributedRandomForest> {
        let start_time = Instant::now();

        // Create training tasks based on partitioning strategy
        let tasks = self.create_training_tasks(x, y)?;

        // Distribute tasks to workers
        self.distribute_tasks(tasks)?;

        // Monitor training progress
        let results = self.monitor_training()?;

        // Aggregate results
        let ensemble = self.aggregate_results(results, rf_config)?;

        let total_time = start_time.elapsed().as_millis() as f64;
        log::info!("Distributed training completed in {:.2}ms", total_time);

        Ok(ensemble)
    }

    /// Create training tasks based on the partitioning strategy
    fn create_training_tasks(&self, x: &Array2<f64>, y: &Array1<i32>) -> Result<Vec<TrainingTask>> {
        let mut tasks = Vec::new();
        let n_estimators = self.config.n_estimators;
        let n_workers = self.config.n_workers;

        match &self.config.partitioning_strategy {
            PartitioningStrategy::TreeBased => {
                // Distribute trees evenly across workers
                let trees_per_worker = n_estimators / n_workers;
                let extra_trees = n_estimators % n_workers;

                let mut tree_start = 0;
                for worker_id in 0..n_workers {
                    let trees_for_worker =
                        trees_per_worker + if worker_id < extra_trees { 1 } else { 0 };
                    let tree_indices: Vec<usize> =
                        (tree_start..tree_start + trees_for_worker).collect();
                    tree_start += trees_for_worker;

                    let random_seeds: Vec<u64> =
                        tree_indices.iter().map(|&i| 42 + i as u64).collect();

                    tasks.push(TrainingTask {
                        task_id: worker_id,
                        worker_id,
                        tree_indices,
                        data_partition: None,    // Use full dataset
                        feature_partition: None, // Use all features
                        random_seeds,
                        priority: TaskPriority::Normal,
                    });
                }
            }

            PartitioningStrategy::DataBased => {
                // Distribute data samples across workers
                let n_samples = x.nrows();
                let samples_per_worker = n_samples / n_workers;
                let extra_samples = n_samples % n_workers;

                let mut sample_start = 0;
                for worker_id in 0..n_workers {
                    let samples_for_worker =
                        samples_per_worker + if worker_id < extra_samples { 1 } else { 0 };
                    let data_partition: Vec<usize> =
                        (sample_start..sample_start + samples_for_worker).collect();
                    sample_start += samples_for_worker;

                    // Each worker trains all trees on their data partition
                    let tree_indices: Vec<usize> = (0..n_estimators).collect();
                    let random_seeds: Vec<u64> = tree_indices
                        .iter()
                        .map(|&i| 42 + i as u64 + worker_id as u64 * 1000)
                        .collect();

                    tasks.push(TrainingTask {
                        task_id: worker_id,
                        worker_id,
                        tree_indices,
                        data_partition: Some(data_partition),
                        feature_partition: None,
                        random_seeds,
                        priority: TaskPriority::Normal,
                    });
                }
            }

            PartitioningStrategy::Hybrid {
                tree_ratio,
                data_ratio,
            } => {
                // Hybrid approach: partition both trees and data
                let trees_per_worker = ((n_estimators as f64) * tree_ratio) as usize / n_workers;
                let samples_per_worker = ((x.nrows() as f64) * data_ratio) as usize / n_workers;

                for worker_id in 0..n_workers {
                    let tree_start = worker_id * trees_per_worker;
                    let tree_end = ((worker_id + 1) * trees_per_worker).min(n_estimators);
                    let tree_indices: Vec<usize> = (tree_start..tree_end).collect();

                    let sample_start = worker_id * samples_per_worker;
                    let sample_end = ((worker_id + 1) * samples_per_worker).min(x.nrows());
                    let data_partition: Vec<usize> = (sample_start..sample_end).collect();

                    let random_seeds: Vec<u64> =
                        tree_indices.iter().map(|&i| 42 + i as u64).collect();

                    tasks.push(TrainingTask {
                        task_id: worker_id,
                        worker_id,
                        tree_indices,
                        data_partition: Some(data_partition),
                        feature_partition: None,
                        random_seeds,
                        priority: TaskPriority::Normal,
                    });
                }
            }

            PartitioningStrategy::FeatureBased { overlap_ratio } => {
                // Feature-based partitioning with overlapping features
                let n_features = x.ncols();
                let features_per_worker = n_features / n_workers;
                let overlap_size = ((features_per_worker as f64) * overlap_ratio) as usize;

                for worker_id in 0..n_workers {
                    let feature_start = if worker_id == 0 {
                        0
                    } else {
                        worker_id * features_per_worker - overlap_size
                    };
                    let feature_end =
                        ((worker_id + 1) * features_per_worker + overlap_size).min(n_features);
                    let feature_partition: Vec<usize> = (feature_start..feature_end).collect();

                    // Each worker trains subset of trees
                    let trees_per_worker = n_estimators / n_workers;
                    let tree_start = worker_id * trees_per_worker;
                    let tree_end = ((worker_id + 1) * trees_per_worker).min(n_estimators);
                    let tree_indices: Vec<usize> = (tree_start..tree_end).collect();

                    let random_seeds: Vec<u64> =
                        tree_indices.iter().map(|&i| 42 + i as u64).collect();

                    tasks.push(TrainingTask {
                        task_id: worker_id,
                        worker_id,
                        tree_indices,
                        data_partition: None,
                        feature_partition: Some(feature_partition),
                        random_seeds,
                        priority: TaskPriority::Normal,
                    });
                }
            }
        }

        Ok(tasks)
    }

    /// Distribute tasks to workers
    fn distribute_tasks(&self, tasks: Vec<TrainingTask>) -> Result<()> {
        let mut task_queue = self.task_queue.lock().unwrap();
        let mut workers = self.workers.lock().unwrap();

        for task in tasks {
            // Assign task to worker
            if let Some(worker) = workers.get_mut(&task.worker_id) {
                worker.assigned_trees = task.tree_indices.len();
                worker.status = WorkerStatus::Training;
            }
            task_queue.push(task);
        }

        log::info!(
            "Distributed {} tasks to {} workers",
            task_queue.len(),
            self.config.n_workers
        );
        Ok(())
    }

    /// Monitor training progress and handle failures
    fn monitor_training(&mut self) -> Result<Vec<TaskResult>> {
        let start_time = Instant::now();
        let max_time = Duration::from_secs(self.config.max_training_time.unwrap_or(3600));

        loop {
            // Check for timeout
            if start_time.elapsed() > max_time {
                log::warn!("Training timeout reached");
                break;
            }

            // Check worker progress
            self.check_worker_health()?;

            // Perform load balancing if enabled
            if self.config.load_balancing.dynamic_balancing {
                self.load_balancer
                    .balance_load(&self.workers, &self.task_queue)?;
            }

            // Check if training is complete
            let completion_status = self.check_completion_status()?;
            if completion_status.is_complete {
                log::info!("Training completed successfully");
                break;
            }

            // Check if we have enough workers for fault tolerance
            if self.config.fault_tolerant
                && completion_status.active_workers < self.config.min_workers
            {
                return Err(sklears_core::error::SklearsError::FitError(format!(
                    "Too few active workers: {} < {}",
                    completion_status.active_workers, self.config.min_workers
                )));
            }

            // Sleep before next check
            thread::sleep(Duration::from_millis(1000));
        }

        // Return completed results
        let task_results = self.task_results.lock().unwrap();
        Ok(task_results.clone())
    }

    /// Check worker health and handle failures
    fn check_worker_health(&self) -> Result<()> {
        let mut workers = self.workers.lock().unwrap();
        let timeout_duration = Duration::from_millis(self.config.communication_timeout);

        for (worker_id, worker) in workers.iter_mut() {
            if worker.last_heartbeat.elapsed() > timeout_duration
                && worker.status == WorkerStatus::Training
            {
                log::warn!("Worker {} timed out", worker_id);
                worker.status = WorkerStatus::Timeout;

                // Handle timeout based on fault tolerance settings
                if self.config.fault_tolerant {
                    self.handle_worker_failure(*worker_id)?;
                }
            }
        }

        Ok(())
    }

    /// Handle worker failure by redistributing tasks
    fn handle_worker_failure(&self, failed_worker_id: usize) -> Result<()> {
        log::info!("Handling failure of worker {}", failed_worker_id);

        // Find incomplete tasks for the failed worker
        let mut task_queue = self.task_queue.lock().unwrap();
        let workers = self.workers.lock().unwrap();

        let mut tasks_to_redistribute = Vec::new();
        task_queue.retain(|task| {
            if task.worker_id == failed_worker_id {
                tasks_to_redistribute.push(task.clone());
                false
            } else {
                true
            }
        });

        // Redistribute tasks to available workers
        for mut task in tasks_to_redistribute {
            if let Some(available_worker_id) = self.find_available_worker(&workers) {
                let task_id = task.task_id;
                task.worker_id = available_worker_id;
                task.priority = TaskPriority::High; // Prioritize redistributed tasks
                task_queue.push(task);
                log::info!(
                    "Redistributed task {} to worker {}",
                    task_id,
                    available_worker_id
                );
            } else {
                log::warn!("No available workers to redistribute task {}", task.task_id);
            }
        }

        Ok(())
    }

    /// Find an available worker for task redistribution
    fn find_available_worker(&self, workers: &HashMap<usize, WorkerNode>) -> Option<usize> {
        for (worker_id, worker) in workers.iter() {
            if worker.status == WorkerStatus::Idle || worker.status == WorkerStatus::Training {
                // Choose worker with lowest load
                if worker.assigned_trees < worker.completed_trees + 10 {
                    // Some threshold
                    return Some(*worker_id);
                }
            }
        }
        None
    }

    /// Check training completion status
    fn check_completion_status(&self) -> Result<CompletionStatus> {
        let workers = self.workers.lock().unwrap();
        let task_queue = self.task_queue.lock().unwrap();
        let task_results = self.task_results.lock().unwrap();

        let active_workers = workers
            .values()
            .filter(|w| w.status == WorkerStatus::Training || w.status == WorkerStatus::Idle)
            .count();

        let completed_workers = workers
            .values()
            .filter(|w| w.status == WorkerStatus::Completed)
            .count();

        let pending_tasks = task_queue.len();
        let completed_tasks = task_results.len();

        let is_complete = pending_tasks == 0 && completed_workers >= self.config.min_workers;

        Ok(CompletionStatus {
            is_complete,
            active_workers,
            completed_workers,
            pending_tasks,
            completed_tasks,
        })
    }

    /// Aggregate results from workers into a final ensemble
    fn aggregate_results(
        &self,
        results: Vec<TaskResult>,
        rf_config: &RandomForestConfig,
    ) -> Result<DistributedRandomForest> {
        if results.is_empty() {
            return Err(sklears_core::error::SklearsError::FitError(
                "No training results to aggregate".to_string(),
            ));
        }

        match &self.config.aggregation_strategy {
            AggregationStrategy::MajorityVoting => {
                self.aggregate_majority_voting(results, rf_config)
            }
            AggregationStrategy::WeightedVoting => {
                self.aggregate_weighted_voting(results, rf_config)
            }
            AggregationStrategy::ModelAveraging => {
                self.aggregate_model_averaging(results, rf_config)
            }
            AggregationStrategy::StackingEnsemble => self.aggregate_stacking(results, rf_config),
        }
    }

    /// Aggregate using majority voting
    fn aggregate_majority_voting(
        &self,
        results: Vec<TaskResult>,
        _rf_config: &RandomForestConfig,
    ) -> Result<DistributedRandomForest> {
        let mut all_trees = Vec::new();
        let mut worker_weights = HashMap::new();

        for result in &results {
            worker_weights.insert(result.worker_id, 1.0); // Equal weights
            for tree_data in &result.trained_trees {
                all_trees.push(tree_data.clone());
            }
        }

        Ok(DistributedRandomForest {
            trees: all_trees,
            worker_weights,
            aggregation_strategy: self.config.aggregation_strategy.clone(),
            training_metrics: self.collect_training_metrics(&results),
        })
    }

    /// Aggregate using weighted voting based on worker performance
    fn aggregate_weighted_voting(
        &self,
        results: Vec<TaskResult>,
        _rf_config: &RandomForestConfig,
    ) -> Result<DistributedRandomForest> {
        let mut all_trees = Vec::new();
        let mut worker_weights = HashMap::new();

        for result in &results {
            // Weight based on OOB score or training time (inverse)
            let weight = if let Some(oob_score) = result.metrics.oob_score {
                oob_score.max(0.1) // Minimum weight
            } else {
                1.0 / (result.metrics.training_time + 1.0) // Inverse of training time
            };

            worker_weights.insert(result.worker_id, weight);
            for tree_data in &result.trained_trees {
                all_trees.push(tree_data.clone());
            }
        }

        // Normalize weights
        let total_weight: f64 = worker_weights.values().sum();
        for weight in worker_weights.values_mut() {
            *weight /= total_weight;
        }

        Ok(DistributedRandomForest {
            trees: all_trees,
            worker_weights,
            aggregation_strategy: self.config.aggregation_strategy.clone(),
            training_metrics: self.collect_training_metrics(&results),
        })
    }

    /// Aggregate using model averaging
    fn aggregate_model_averaging(
        &self,
        results: Vec<TaskResult>,
        _rf_config: &RandomForestConfig,
    ) -> Result<DistributedRandomForest> {
        // For tree ensembles, model averaging is similar to weighted voting
        // but with different weight calculation
        let mut all_trees = Vec::new();
        let mut worker_weights = HashMap::new();

        let total_workers = results.len() as f64;
        for result in &results {
            worker_weights.insert(result.worker_id, 1.0 / total_workers); // Equal averaging
            for tree_data in &result.trained_trees {
                all_trees.push(tree_data.clone());
            }
        }

        Ok(DistributedRandomForest {
            trees: all_trees,
            worker_weights,
            aggregation_strategy: self.config.aggregation_strategy.clone(),
            training_metrics: self.collect_training_metrics(&results),
        })
    }

    /// Aggregate using stacking ensemble
    fn aggregate_stacking(
        &self,
        results: Vec<TaskResult>,
        _rf_config: &RandomForestConfig,
    ) -> Result<DistributedRandomForest> {
        // Simplified stacking: use meta-learning weights
        let mut all_trees = Vec::new();
        let mut worker_weights = HashMap::new();

        // Calculate stacking weights based on cross-validation performance
        for result in &results {
            let meta_weight = self.calculate_meta_weight(&result.metrics);
            worker_weights.insert(result.worker_id, meta_weight);
            for tree_data in &result.trained_trees {
                all_trees.push(tree_data.clone());
            }
        }

        Ok(DistributedRandomForest {
            trees: all_trees,
            worker_weights,
            aggregation_strategy: self.config.aggregation_strategy.clone(),
            training_metrics: self.collect_training_metrics(&results),
        })
    }

    /// Calculate meta-learning weight for stacking
    fn calculate_meta_weight(&self, metrics: &TrainingMetrics) -> f64 {
        // Combine multiple metrics for meta-weight calculation
        let oob_weight = metrics.oob_score.unwrap_or(0.5);
        let time_weight = 1.0 / (metrics.training_time / 1000.0 + 1.0); // Inverse of time in seconds
        let memory_weight = 1.0 / (metrics.peak_memory / 1000.0 + 1.0); // Inverse of memory in GB

        // Weighted combination
        0.6 * oob_weight + 0.3 * time_weight + 0.1 * memory_weight
    }

    /// Collect training metrics from all workers
    fn collect_training_metrics(&self, results: &[TaskResult]) -> DistributedTrainingMetrics {
        let total_training_time: f64 = results.iter().map(|r| r.metrics.training_time).sum();
        let avg_oob_score = {
            let scores: Vec<f64> = results.iter().filter_map(|r| r.metrics.oob_score).collect();
            if scores.is_empty() {
                None
            } else {
                Some(scores.iter().sum::<f64>() / scores.len() as f64)
            }
        };
        let peak_memory: f64 = results
            .iter()
            .map(|r| r.metrics.peak_memory)
            .fold(0.0, f64::max);

        DistributedTrainingMetrics {
            total_training_time,
            avg_oob_score,
            peak_memory,
            n_workers_used: results.len(),
            total_trees: results.iter().map(|r| r.trained_trees.len()).sum(),
        }
    }
}

/// Completion status for training monitoring
#[derive(Debug)]
struct CompletionStatus {
    is_complete: bool,
    active_workers: usize,
    completed_workers: usize,
    pending_tasks: usize,
    completed_tasks: usize,
}

/// Load balancer for distributed training
struct LoadBalancer {
    config: LoadBalancingConfig,
    last_balance_time: Instant,
}

impl LoadBalancer {
    fn new(config: LoadBalancingConfig) -> Self {
        Self {
            config,
            last_balance_time: Instant::now(),
        }
    }

    fn balance_load(
        &mut self,
        workers: &Arc<Mutex<HashMap<usize, WorkerNode>>>,
        task_queue: &Arc<Mutex<Vec<TrainingTask>>>,
    ) -> Result<()> {
        let now = Instant::now();
        let check_interval = Duration::from_millis(self.config.check_interval);

        if now.duration_since(self.last_balance_time) < check_interval {
            return Ok(());
        }

        self.last_balance_time = now;

        let workers_guard = workers.lock().unwrap();
        let task_queue_guard = task_queue.lock().unwrap();

        // Calculate load imbalance
        let loads: Vec<f64> = workers_guard
            .values()
            .map(|w| w.assigned_trees as f64 / (w.completed_trees + 1) as f64)
            .collect();

        if loads.is_empty() {
            return Ok(());
        }

        let max_load = loads.iter().fold(0.0f64, |a, &b| a.max(b));
        let min_load = loads.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let imbalance_ratio = max_load / min_load.max(1e-10);

        if imbalance_ratio > self.config.max_imbalance_ratio {
            log::info!("Load imbalance detected: {:.2}", imbalance_ratio);

            if self.config.work_stealing {
                self.perform_work_stealing(&workers_guard, &task_queue_guard)?;
            }
        }

        Ok(())
    }

    fn perform_work_stealing(
        &self,
        _workers: &HashMap<usize, WorkerNode>,
        _task_queue: &Vec<TrainingTask>,
    ) -> Result<()> {
        // Simplified work stealing implementation
        log::info!("Performing work stealing to balance load");
        // In a real implementation, this would redistribute tasks
        // from overloaded workers to underloaded ones
        Ok(())
    }
}

/// Distributed random forest ensemble
#[derive(Debug)]
pub struct DistributedRandomForest {
    /// Serialized trees from all workers
    trees: Vec<Vec<u8>>,
    /// Worker weights for aggregation
    worker_weights: HashMap<usize, f64>,
    /// Aggregation strategy used
    aggregation_strategy: AggregationStrategy,
    /// Training metrics
    training_metrics: DistributedTrainingMetrics,
}

/// Training metrics for distributed ensemble
#[derive(Debug)]
pub struct DistributedTrainingMetrics {
    /// Total training time across all workers
    pub total_training_time: f64,
    /// Average out-of-bag score
    pub avg_oob_score: Option<f64>,
    /// Peak memory usage
    pub peak_memory: f64,
    /// Number of workers used
    pub n_workers_used: usize,
    /// Total number of trees
    pub total_trees: usize,
}

impl DistributedRandomForest {
    /// Predict class probabilities for new samples
    pub fn predict_proba(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        // In a real implementation, this would deserialize trees and aggregate predictions
        // For now, return a placeholder
        let n_samples = x.nrows();
        let n_classes = 2; // Assume binary classification for simplicity

        Ok(Array2::from_elem((n_samples, n_classes), 0.5))
    }

    /// Predict class labels for new samples  
    pub fn predict(&self, x: &Array2<f64>) -> Result<Array1<i32>> {
        let probabilities = self.predict_proba(x)?;
        let predictions = probabilities.map_axis(scirs2_core::ndarray::Axis(1), |row| {
            row.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i as i32)
                .unwrap_or(0)
        });
        Ok(predictions)
    }

    /// Get training metrics
    pub fn training_metrics(&self) -> &DistributedTrainingMetrics {
        &self.training_metrics
    }

    /// Get number of trees in the ensemble
    pub fn n_trees(&self) -> usize {
        self.trees.len()
    }

    /// Get aggregation strategy used
    pub fn aggregation_strategy(&self) -> &AggregationStrategy {
        &self.aggregation_strategy
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_distributed_config_default() {
        let config = DistributedConfig::default();
        assert_eq!(config.n_estimators, 100);
        assert_eq!(config.n_workers, 4);
        assert!(config.fault_tolerant);
    }

    #[test]
    fn test_worker_node_creation() {
        let worker = WorkerNode {
            id: 0,
            status: WorkerStatus::Idle,
            assigned_trees: 0,
            completed_trees: 0,
            performance: WorkerPerformance::default(),
            last_heartbeat: Instant::now(),
        };
        assert_eq!(worker.id, 0);
        assert_eq!(worker.status, WorkerStatus::Idle);
    }

    #[test]
    fn test_partitioning_strategy_tree_based() {
        let config = DistributedConfig {
            n_estimators: 10,
            n_workers: 3,
            partitioning_strategy: PartitioningStrategy::TreeBased,
            ..Default::default()
        };

        let trainer = DistributedEnsembleTrainer::new(config);
        let x = Array2::zeros((100, 5));
        let y = Array1::zeros(100);

        let tasks = trainer.create_training_tasks(&x, &y).unwrap();
        assert_eq!(tasks.len(), 3); // One task per worker

        // Check tree distribution
        let total_trees: usize = tasks.iter().map(|t| t.tree_indices.len()).sum();
        assert_eq!(total_trees, 10);
    }

    #[test]
    fn test_load_balancing_config() {
        let config = LoadBalancingConfig::default();
        assert!(config.dynamic_balancing);
        assert_eq!(config.check_interval, 5000);
        assert_eq!(config.max_imbalance_ratio, 2.0);
    }
}
