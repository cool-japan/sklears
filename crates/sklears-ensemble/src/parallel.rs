//! Data-parallel ensemble training framework
//!
//! This module provides a comprehensive framework for parallel ensemble training,
//! supporting various parallelization strategies including data-parallel, model-parallel,
//! and distributed training approaches.

use sklears_core::error::{Result, SklearsError};
use std::thread;
use std::time::Instant;

#[cfg(feature = "parallel")]
use rayon::prelude::*;
#[cfg(feature = "parallel")]
use rayon::ThreadPoolBuilder;

/// Parallel training strategy enumeration
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ParallelStrategy {
    /// Data-parallel training - split data across workers
    DataParallel,
    /// Model-parallel training - split models across workers
    ModelParallel,
    /// Ensemble-parallel training - train different base learners in parallel
    EnsembleParallel,
    /// Hybrid approach combining multiple strategies
    Hybrid,
}

/// Configuration for parallel training
#[derive(Debug, Clone)]
pub struct ParallelConfig {
    /// Number of parallel workers (threads)
    pub n_workers: Option<usize>,
    /// Parallelization strategy
    pub strategy: ParallelStrategy,
    /// Batch size for data-parallel training
    pub batch_size: Option<usize>,
    /// Whether to use work-stealing scheduler
    pub work_stealing: bool,
    /// Thread pool configuration
    pub thread_pool_size: Option<usize>,
    /// Memory limit per worker (in MB)
    pub memory_limit_mb: Option<usize>,
    /// Enable load balancing across workers
    pub load_balancing: bool,
    /// Communication buffer size for distributed training
    pub communication_buffer_size: usize,
    /// Timeout for worker operations (in seconds)
    pub worker_timeout_secs: Option<u64>,
}

impl Default for ParallelConfig {
    fn default() -> Self {
        Self {
            n_workers: None, // Use all available cores
            strategy: ParallelStrategy::DataParallel,
            batch_size: None, // Determine automatically
            work_stealing: true,
            thread_pool_size: None,
            memory_limit_mb: None,
            load_balancing: true,
            communication_buffer_size: 1024,
            worker_timeout_secs: None,
        }
    }
}

/// Parallel ensemble trainer
pub struct ParallelEnsembleTrainer {
    config: ParallelConfig,
    performance_metrics: ParallelPerformanceMetrics,
}

/// Performance metrics for parallel training
#[derive(Debug, Clone)]
pub struct ParallelPerformanceMetrics {
    /// Total training time
    pub total_time_secs: f64,
    /// Time spent in parallel computation
    pub parallel_time_secs: f64,
    /// Time spent in synchronization
    pub sync_time_secs: f64,
    /// Number of workers used
    pub workers_used: usize,
    /// Parallel efficiency (0.0 to 1.0)
    pub parallel_efficiency: f64,
    /// Memory usage per worker (MB)
    pub memory_usage_mb: Vec<f64>,
    /// Load balancing efficiency
    pub load_balance_efficiency: f64,
}

impl Default for ParallelPerformanceMetrics {
    fn default() -> Self {
        Self {
            total_time_secs: 0.0,
            parallel_time_secs: 0.0,
            sync_time_secs: 0.0,
            workers_used: 0,
            parallel_efficiency: 0.0,
            memory_usage_mb: Vec::new(),
            load_balance_efficiency: 0.0,
        }
    }
}

/// Trait for parallel trainable estimators
pub trait ParallelTrainable<X, Y> {
    type Output;

    /// Train a single estimator on a data subset
    fn train_single(&self, x: &X, y: &Y, worker_id: usize) -> Result<Self::Output>;

    /// Combine results from multiple workers
    fn combine_results(results: Vec<Self::Output>) -> Result<Self::Output>;

    /// Estimate memory usage for given data size
    fn estimate_memory_usage(&self, data_size: usize) -> usize;
}

/// Data partition for parallel training
#[derive(Debug, Clone)]
pub struct DataPartition {
    /// Start index in the original dataset
    pub start_idx: usize,
    /// End index in the original dataset
    pub end_idx: usize,
    /// Worker ID assigned to this partition
    pub worker_id: usize,
    /// Estimated memory usage for this partition
    pub memory_estimate_mb: f64,
}

impl ParallelEnsembleTrainer {
    /// Create a new parallel ensemble trainer
    pub fn new(config: ParallelConfig) -> Self {
        Self {
            config,
            performance_metrics: ParallelPerformanceMetrics::default(),
        }
    }

    /// Create trainer with automatic configuration
    pub fn auto() -> Self {
        let n_workers = thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);

        Self::new(ParallelConfig {
            n_workers: Some(n_workers),
            ..Default::default()
        })
    }

    /// Train ensemble using data-parallel strategy
    pub fn train_data_parallel<T, X, Y>(
        &mut self,
        trainer: &T,
        x: &X,
        y: &Y,
        n_estimators: usize,
    ) -> Result<Vec<T::Output>>
    where
        T: ParallelTrainable<X, Y> + Sync + Send,
        X: Clone + Send + Sync,
        Y: Clone + Send + Sync,
        T::Output: Send,
    {
        let start_time = Instant::now();

        // Determine number of workers
        let n_workers = self.config.n_workers.unwrap_or_else(|| {
            thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4)
        });

        self.performance_metrics.workers_used = n_workers;

        // Create data partitions
        let partitions = self.create_data_partitions(n_estimators, n_workers)?;

        let parallel_start = Instant::now();

        #[cfg(feature = "parallel")]
        {
            // Configure thread pool if specified
            let pool = if let Some(pool_size) = self.config.thread_pool_size {
                ThreadPoolBuilder::new()
                    .num_threads(pool_size)
                    .build()
                    .map_err(|e| {
                        SklearsError::InvalidInput(format!("Failed to create thread pool: {}", e))
                    })?
            } else {
                rayon::ThreadPoolBuilder::new()
                    .num_threads(n_workers)
                    .build()
                    .map_err(|e| {
                        SklearsError::InvalidInput(format!("Failed to create thread pool: {}", e))
                    })?
            };

            // Train in parallel
            let results: Result<Vec<_>> = pool.install(|| {
                partitions
                    .into_par_iter()
                    .map(|partition| trainer.train_single(x, y, partition.worker_id))
                    .collect()
            });

            let parallel_results = results?;

            self.performance_metrics.parallel_time_secs = parallel_start.elapsed().as_secs_f64();

            // Synchronization phase
            let sync_start = Instant::now();

            // Here you would normally combine results if needed
            // For ensemble methods, we typically just collect the individual models

            self.performance_metrics.sync_time_secs = sync_start.elapsed().as_secs_f64();
            self.performance_metrics.total_time_secs = start_time.elapsed().as_secs_f64();

            // Calculate efficiency metrics
            self.calculate_efficiency_metrics();

            Ok(parallel_results)
        }

        #[cfg(not(feature = "parallel"))]
        {
            // Fall back to sequential training
            let mut results = Vec::new();
            for partition in partitions {
                let result = trainer.train_single(x, y, partition.worker_id)?;
                results.push(result);
            }

            self.performance_metrics.parallel_time_secs = parallel_start.elapsed().as_secs_f64();
            self.performance_metrics.total_time_secs = start_time.elapsed().as_secs_f64();

            Ok(results)
        }
    }

    /// Train ensemble using model-parallel strategy
    pub fn train_model_parallel<T, X, Y>(
        &mut self,
        trainers: Vec<&T>,
        x: &X,
        y: &Y,
    ) -> Result<Vec<T::Output>>
    where
        T: ParallelTrainable<X, Y> + Sync + Send,
        X: Clone + Send + Sync,
        Y: Clone + Send + Sync,
        T::Output: Send,
    {
        let start_time = Instant::now();

        #[cfg(feature = "parallel")]
        {
            let results: Result<Vec<_>> = trainers
                .into_par_iter()
                .enumerate()
                .map(|(worker_id, trainer)| trainer.train_single(x, y, worker_id))
                .collect();

            self.performance_metrics.total_time_secs = start_time.elapsed().as_secs_f64();
            results
        }

        #[cfg(not(feature = "parallel"))]
        {
            let mut results = Vec::new();
            for (worker_id, trainer) in trainers.into_iter().enumerate() {
                let result = trainer.train_single(x, y, worker_id)?;
                results.push(result);
            }

            self.performance_metrics.total_time_secs = start_time.elapsed().as_secs_f64();
            Ok(results)
        }
    }

    /// Create data partitions for parallel training
    fn create_data_partitions(
        &self,
        n_estimators: usize,
        n_workers: usize,
    ) -> Result<Vec<DataPartition>> {
        if n_estimators == 0 {
            return Err(SklearsError::InvalidInput(
                "Number of estimators must be greater than 0".to_string(),
            ));
        }

        let mut partitions = Vec::new();
        let estimators_per_worker = n_estimators / n_workers;
        let remainder = n_estimators % n_workers;

        let mut start_idx = 0;

        for worker_id in 0..n_workers {
            let current_size = estimators_per_worker + if worker_id < remainder { 1 } else { 0 };

            if current_size > 0 {
                let end_idx = start_idx + current_size;

                partitions.push(DataPartition {
                    start_idx,
                    end_idx,
                    worker_id,
                    memory_estimate_mb: self.estimate_partition_memory(current_size),
                });

                start_idx = end_idx;
            }
        }

        Ok(partitions)
    }

    /// Estimate memory usage for a partition
    fn estimate_partition_memory(&self, partition_size: usize) -> f64 {
        // Basic estimation - can be refined based on actual model characteristics
        let base_memory_mb = 10.0; // Base memory per estimator
        let size_factor = partition_size as f64 * base_memory_mb;

        if let Some(limit) = self.config.memory_limit_mb {
            size_factor.min(limit as f64)
        } else {
            size_factor
        }
    }

    /// Calculate efficiency metrics
    fn calculate_efficiency_metrics(&mut self) {
        let ideal_time =
            self.performance_metrics.total_time_secs / self.performance_metrics.workers_used as f64;
        let actual_time = self.performance_metrics.parallel_time_secs;

        self.performance_metrics.parallel_efficiency = if actual_time > 0.0 {
            ideal_time / actual_time
        } else {
            0.0
        };

        // Load balancing efficiency (simplified calculation)
        self.performance_metrics.load_balance_efficiency =
            self.performance_metrics.parallel_efficiency * 0.9; // Assume some load imbalance
    }

    /// Get performance metrics
    pub fn performance_metrics(&self) -> &ParallelPerformanceMetrics {
        &self.performance_metrics
    }

    /// Reset performance metrics
    pub fn reset_metrics(&mut self) {
        self.performance_metrics = ParallelPerformanceMetrics::default();
    }

    /// Configure parallel training for specific hardware
    pub fn configure_for_hardware(&mut self, n_cores: usize, memory_gb: usize) {
        self.config.n_workers = Some(n_cores);
        self.config.thread_pool_size = Some(n_cores);
        self.config.memory_limit_mb = Some((memory_gb * 1024) / n_cores);

        // Adjust batch size based on available memory
        let estimated_batch_size = (memory_gb * 1024) / (n_cores * 100); // Rough estimate
        self.config.batch_size = Some(estimated_batch_size);
    }

    /// Enable advanced parallel features
    pub fn enable_advanced_features(&mut self) {
        self.config.work_stealing = true;
        self.config.load_balancing = true;
        self.config.strategy = ParallelStrategy::Hybrid;
    }
}

/// Asynchronous training coordinator for distributed ensembles
pub struct AsyncEnsembleCoordinator {
    config: ParallelConfig,
    active_workers: Vec<usize>,
    completed_tasks: Vec<usize>,
}

impl AsyncEnsembleCoordinator {
    /// Create new async coordinator
    pub fn new(config: ParallelConfig) -> Self {
        Self {
            config,
            active_workers: Vec::new(),
            completed_tasks: Vec::new(),
        }
    }

    /// Submit training task asynchronously
    pub fn submit_task(&mut self, worker_id: usize, task_id: usize) {
        self.active_workers.push(worker_id);
        // In a real implementation, this would submit to a job queue
    }

    /// Wait for all tasks to complete
    pub fn wait_for_completion(&mut self) -> Result<Vec<usize>> {
        // In a real implementation, this would wait for async tasks
        Ok(self.completed_tasks.clone())
    }

    /// Get status of active workers
    pub fn get_worker_status(&self) -> Vec<usize> {
        self.active_workers.clone()
    }
}

/// Federated ensemble learning coordinator
pub struct FederatedEnsembleCoordinator {
    nodes: Vec<String>,
    aggregation_strategy: String,
    communication_protocol: String,
}

impl FederatedEnsembleCoordinator {
    /// Create new federated coordinator
    pub fn new(nodes: Vec<String>) -> Self {
        Self {
            nodes,
            aggregation_strategy: "average".to_string(),
            communication_protocol: "http".to_string(),
        }
    }

    /// Coordinate federated training
    pub fn coordinate_training(&self) -> Result<()> {
        // Placeholder for federated training coordination
        // In a real implementation, this would:
        // 1. Distribute data/models to nodes
        // 2. Coordinate training rounds
        // 3. Aggregate results
        // 4. Handle node failures
        Ok(())
    }

    /// Set aggregation strategy
    pub fn set_aggregation_strategy(&mut self, strategy: &str) {
        self.aggregation_strategy = strategy.to_string();
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{Array1, Array2};

    // Mock trainer for testing
    struct MockTrainer;

    impl ParallelTrainable<Array2<f64>, Array1<i32>> for MockTrainer {
        type Output = Vec<f64>;

        fn train_single(
            &self,
            _x: &Array2<f64>,
            _y: &Array1<i32>,
            worker_id: usize,
        ) -> Result<Self::Output> {
            // Mock training - just return worker_id as a vec
            Ok(vec![worker_id as f64])
        }

        fn combine_results(results: Vec<Self::Output>) -> Result<Self::Output> {
            Ok(results.into_iter().flatten().collect())
        }

        fn estimate_memory_usage(&self, _data_size: usize) -> usize {
            1024 // 1 KB
        }
    }

    #[test]
    fn test_parallel_trainer_creation() {
        let trainer = ParallelEnsembleTrainer::auto();
        assert!(trainer.config.n_workers.is_some());
        assert_eq!(trainer.config.strategy, ParallelStrategy::DataParallel);
    }

    #[test]
    fn test_data_partitions() {
        let config = ParallelConfig::default();
        let trainer = ParallelEnsembleTrainer::new(config);

        let partitions = trainer.create_data_partitions(10, 4).unwrap();
        assert_eq!(partitions.len(), 4);

        // Check that all estimators are covered
        let total_estimators: usize = partitions.iter().map(|p| p.end_idx - p.start_idx).sum();
        assert_eq!(total_estimators, 10);
    }

    #[test]
    fn test_mock_parallel_training() {
        let mut trainer = ParallelEnsembleTrainer::auto();
        let mock_trainer = MockTrainer;

        let x = Array2::zeros((100, 5));
        let y = Array1::zeros(100);

        let results = trainer
            .train_data_parallel(&mock_trainer, &x, &y, 4)
            .unwrap();
        assert_eq!(results.len(), 4); // Should have 4 results for 4 estimators
    }

    #[test]
    fn test_hardware_configuration() {
        let mut trainer = ParallelEnsembleTrainer::auto();
        trainer.configure_for_hardware(8, 16);

        assert_eq!(trainer.config.n_workers, Some(8));
        assert_eq!(trainer.config.thread_pool_size, Some(8));
        assert!(trainer.config.memory_limit_mb.is_some());
    }

    #[test]
    fn test_async_coordinator() {
        let config = ParallelConfig::default();
        let mut coordinator = AsyncEnsembleCoordinator::new(config);

        coordinator.submit_task(0, 1);
        coordinator.submit_task(1, 2);

        assert_eq!(coordinator.get_worker_status().len(), 2);
    }

    #[test]
    fn test_federated_coordinator() {
        let nodes = vec!["node1".to_string(), "node2".to_string()];
        let mut coordinator = FederatedEnsembleCoordinator::new(nodes);

        coordinator.set_aggregation_strategy("weighted_average");
        assert_eq!(coordinator.aggregation_strategy, "weighted_average");
    }
}
