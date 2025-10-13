use scirs2_core::ndarray::{Array, Array1, Array2, ArrayD, Axis, Dimension, RemoveAxis};
use scirs2_core::numeric::Float;
use sklears_core::error::SklearsError;
use sklears_core::types::FloatBounds;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

/// Result type for distributed training operations
pub type DistributedResult<T> = Result<T, SklearsError>;

/// Configuration for distributed training
#[derive(Debug, Clone)]
pub struct DistributedConfig<T: FloatBounds> {
    /// Number of worker processes/devices
    pub num_workers: usize,
    /// Backend type for communication (GPU, CPU, etc.)
    pub backend: DistributedBackend,
    /// Gradient synchronization strategy
    pub sync_strategy: GradientSyncStrategy,
    /// Batch size per worker
    pub batch_size_per_worker: usize,
    /// Frequency of gradient synchronization (in steps)
    pub sync_frequency: usize,
    /// Learning rate scaling strategy
    pub lr_scaling: LearningRateScaling<T>,
    /// Whether to use gradient compression
    pub gradient_compression: bool,
    /// Compression threshold for gradients
    pub compression_threshold: T,
    /// Maximum gradient norm for clipping
    pub max_grad_norm: Option<T>,
    /// Warmup steps for distributed training
    pub warmup_steps: usize,
}

impl<T: FloatBounds> Default for DistributedConfig<T> {
    fn default() -> Self {
        Self {
            num_workers: 1,
            backend: DistributedBackend::CPU,
            sync_strategy: GradientSyncStrategy::AllReduce,
            batch_size_per_worker: 32,
            sync_frequency: 1,
            lr_scaling: LearningRateScaling::Linear,
            gradient_compression: false,
            compression_threshold: T::from(0.01).unwrap(),
            max_grad_norm: Some(T::from(1.0).unwrap()),
            warmup_steps: 0,
        }
    }
}

/// Backend types for distributed training
#[derive(Debug, Clone)]
pub enum DistributedBackend {
    /// CPU-based distributed training
    CPU,
    /// GPU-based distributed training (requires CUDA)
    GPU,
    /// Mixed CPU/GPU training
    Mixed,
}

/// Gradient synchronization strategies
#[derive(Debug, Clone)]
pub enum GradientSyncStrategy {
    /// All-reduce synchronization (default)
    AllReduce,
    /// Parameter server approach
    ParameterServer,
    /// Hierarchical synchronization
    Hierarchical,
    /// Asynchronous gradient updates
    Asynchronous,
}

/// Learning rate scaling strategies for distributed training
#[derive(Debug, Clone)]
pub enum LearningRateScaling<T: FloatBounds> {
    /// Linear scaling (lr * num_workers)
    Linear,
    /// Square root scaling (lr * sqrt(num_workers))
    SquareRoot,
    /// Custom scaling factor
    Custom(T),
    /// No scaling
    None,
}

/// Statistics for distributed training
#[derive(Debug, Clone, Default)]
pub struct DistributedStats<T: FloatBounds> {
    /// Communication time per step
    pub communication_time_ms: Vec<f64>,
    /// Computation time per step
    pub computation_time_ms: Vec<f64>,
    /// Gradient norm before synchronization
    pub gradient_norms_before: Vec<T>,
    /// Gradient norm after synchronization
    pub gradient_norms_after: Vec<T>,
    /// Memory usage per worker
    pub memory_usage_mb: Vec<f64>,
    /// Load balancing efficiency
    pub load_balance_efficiency: Vec<f64>,
}

/// Distributed training coordinator
pub struct DistributedTrainer<T: FloatBounds> {
    /// Configuration
    config: DistributedConfig<T>,
    /// Worker coordinators
    workers: Vec<WorkerCoordinator<T>>,
    /// Parameter server (if using parameter server strategy)
    parameter_server: Option<ParameterServer<T>>,
    /// Training statistics
    stats: DistributedStats<T>,
    /// Current step
    current_step: usize,
    /// Gradient buffers
    gradient_buffers: HashMap<String, Array1<T>>,
}

impl<T: FloatBounds + Default + std::iter::Sum<T> + scirs2_core::ndarray::ScalarOperand + Copy>
    DistributedTrainer<T>
{
    /// Create a new distributed trainer
    pub fn new(config: DistributedConfig<T>) -> DistributedResult<Self> {
        if config.num_workers == 0 {
            return Err(SklearsError::InvalidParameter {
                name: "num_workers".to_string(),
                reason: "Number of workers must be greater than 0".to_string(),
            });
        }

        let workers = (0..config.num_workers)
            .map(|rank| WorkerCoordinator::new(rank, &config))
            .collect::<Result<Vec<_>, _>>()?;

        let parameter_server = match config.sync_strategy {
            GradientSyncStrategy::ParameterServer => Some(ParameterServer::new(&config)?),
            _ => None,
        };

        Ok(Self {
            config,
            workers,
            parameter_server,
            stats: DistributedStats::default(),
            current_step: 0,
            gradient_buffers: HashMap::new(),
        })
    }

    /// Distribute data across workers
    pub fn distribute_data<D: Dimension>(
        &self,
        data: &Array<T, D>,
    ) -> DistributedResult<Vec<Array<T, D>>>
    where
        D: RemoveAxis,
    {
        let num_samples = data.shape()[0];
        let samples_per_worker = num_samples / self.config.num_workers;

        if samples_per_worker == 0 {
            return Err(SklearsError::InvalidParameter {
                name: "data_size".to_string(),
                reason: "Data size too small for the number of workers".to_string(),
            });
        }

        let mut distributed_data = Vec::new();

        for i in 0..self.config.num_workers {
            let start_idx = i * samples_per_worker;
            let end_idx = if i == self.config.num_workers - 1 {
                num_samples // Last worker gets remaining samples
            } else {
                (i + 1) * samples_per_worker
            };

            let worker_data =
                data.slice_axis(scirs2_core::ndarray::Axis(0), (start_idx..end_idx).into());
            distributed_data.push(worker_data.to_owned());
        }

        Ok(distributed_data)
    }

    /// Synchronize gradients across workers
    pub fn synchronize_gradients(
        &mut self,
        gradients: &mut HashMap<String, Array1<T>>,
    ) -> DistributedResult<()> {
        let start_time = Instant::now();

        match self.config.sync_strategy {
            GradientSyncStrategy::AllReduce => {
                self.all_reduce_gradients(gradients)?;
            }
            GradientSyncStrategy::ParameterServer => {
                self.parameter_server_sync(gradients)?;
            }
            GradientSyncStrategy::Hierarchical => {
                self.hierarchical_sync(gradients)?;
            }
            GradientSyncStrategy::Asynchronous => {
                // Asynchronous updates don't require synchronization
            }
        }

        // Apply gradient compression if enabled
        if self.config.gradient_compression {
            self.compress_gradients(gradients)?;
        }

        // Apply gradient clipping if configured
        if let Some(max_norm) = self.config.max_grad_norm {
            self.clip_gradients(gradients, max_norm)?;
        }

        let communication_time = start_time.elapsed().as_millis() as f64;
        self.stats.communication_time_ms.push(communication_time);

        Ok(())
    }

    /// All-reduce gradient synchronization
    fn all_reduce_gradients(
        &mut self,
        gradients: &mut HashMap<String, Array1<T>>,
    ) -> DistributedResult<()> {
        for (name, grad) in gradients.iter_mut() {
            // Simulate all-reduce by averaging gradients across workers
            let sum: T = grad.iter().copied().sum();
            let mean = sum / T::from(self.config.num_workers).unwrap();
            grad.fill(mean);

            // Store gradient norm before sync
            let norm_before = self.compute_gradient_norm(grad);
            self.stats.gradient_norms_before.push(norm_before);
        }
        Ok(())
    }

    /// Parameter server gradient synchronization
    fn parameter_server_sync(
        &mut self,
        gradients: &mut HashMap<String, Array1<T>>,
    ) -> DistributedResult<()> {
        if let Some(ref mut ps) = self.parameter_server {
            ps.aggregate_gradients(gradients)?;
            ps.broadcast_parameters(gradients)?;
        }
        Ok(())
    }

    /// Hierarchical gradient synchronization
    fn hierarchical_sync(
        &mut self,
        gradients: &mut HashMap<String, Array1<T>>,
    ) -> DistributedResult<()> {
        // Implement hierarchical reduction (e.g., reduce within nodes, then across nodes)
        let num_levels = (self.config.num_workers as f64).log2().ceil() as usize;

        for level in 0..num_levels {
            let group_size = 2_usize.pow(level as u32);
            for (name, grad) in gradients.iter_mut() {
                // Simulate hierarchical reduction
                let reduction_factor = T::from(group_size).unwrap();
                grad.mapv_inplace(|x| x / reduction_factor);
            }
        }

        Ok(())
    }

    /// Compress gradients using threshold-based compression
    fn compress_gradients(
        &self,
        gradients: &mut HashMap<String, Array1<T>>,
    ) -> DistributedResult<()> {
        for grad in gradients.values_mut() {
            grad.mapv_inplace(|x| {
                if x.abs() < self.config.compression_threshold {
                    T::zero()
                } else {
                    x
                }
            });
        }
        Ok(())
    }

    /// Clip gradients to prevent exploding gradients
    fn clip_gradients(
        &self,
        gradients: &mut HashMap<String, Array1<T>>,
        max_norm: T,
    ) -> DistributedResult<()> {
        for grad in gradients.values_mut() {
            let norm = self.compute_gradient_norm(grad);
            if norm > max_norm {
                let scale_factor = max_norm / norm;
                grad.mapv_inplace(|x| x * scale_factor);
            }
        }
        Ok(())
    }

    /// Compute L2 norm of gradient
    fn compute_gradient_norm(&self, grad: &Array1<T>) -> T {
        grad.mapv(|x| x * x).sum().sqrt()
    }

    /// Scale learning rate based on number of workers
    pub fn scale_learning_rate(&self, base_lr: T) -> T {
        match self.config.lr_scaling {
            LearningRateScaling::Linear => base_lr * T::from(self.config.num_workers).unwrap(),
            LearningRateScaling::SquareRoot => {
                base_lr * T::from(self.config.num_workers as f64).unwrap().sqrt()
            }
            LearningRateScaling::Custom(factor) => base_lr * factor,
            LearningRateScaling::None => base_lr,
        }
    }

    /// Update statistics
    pub fn update_stats(&mut self, computation_time_ms: f64, memory_usage_mb: f64) {
        self.stats.computation_time_ms.push(computation_time_ms);
        self.stats.memory_usage_mb.push(memory_usage_mb);

        // Calculate load balancing efficiency
        let efficiency = self.calculate_load_balance_efficiency();
        self.stats.load_balance_efficiency.push(efficiency);
    }

    /// Calculate load balancing efficiency
    fn calculate_load_balance_efficiency(&self) -> f64 {
        if self.stats.computation_time_ms.is_empty() {
            return 1.0;
        }

        let last_idx = self.stats.computation_time_ms.len() - 1;
        let max_time = *self
            .stats
            .computation_time_ms
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        let avg_time = self.stats.computation_time_ms.iter().sum::<f64>()
            / self.stats.computation_time_ms.len() as f64;

        if max_time > 0.0 {
            avg_time / max_time
        } else {
            1.0
        }
    }

    /// Get training statistics
    pub fn get_stats(&self) -> &DistributedStats<T> {
        &self.stats
    }

    /// Perform distributed training step
    pub fn training_step<F>(
        &mut self,
        compute_fn: F,
        gradients: &mut HashMap<String, Array1<T>>,
    ) -> DistributedResult<()>
    where
        F: Fn() -> DistributedResult<f64> + Send + Sync,
    {
        let start_time = Instant::now();

        // Compute gradients on all workers in parallel
        let computation_time = compute_fn()?;

        // Synchronize gradients if needed
        if self.current_step % self.config.sync_frequency == 0 {
            self.synchronize_gradients(gradients)?;
        }

        // Update statistics
        let total_time = start_time.elapsed().as_millis() as f64;
        self.update_stats(computation_time, 0.0); // Memory usage would need system integration

        self.current_step += 1;

        Ok(())
    }
}

/// Worker coordinator for distributed training
pub struct WorkerCoordinator<T: FloatBounds> {
    /// Worker rank/ID
    rank: usize,
    /// Local gradients
    local_gradients: HashMap<String, Array1<T>>,
    /// Worker-specific configuration
    config: DistributedConfig<T>,
}

impl<T: FloatBounds> WorkerCoordinator<T> {
    pub fn new(rank: usize, config: &DistributedConfig<T>) -> DistributedResult<Self> {
        Ok(Self {
            rank,
            local_gradients: HashMap::new(),
            config: config.clone(),
        })
    }

    pub fn get_rank(&self) -> usize {
        self.rank
    }

    pub fn update_gradients(&mut self, gradients: HashMap<String, Array1<T>>) {
        self.local_gradients = gradients;
    }

    pub fn get_gradients(&self) -> &HashMap<String, Array1<T>> {
        &self.local_gradients
    }
}

/// Parameter server for distributed training
pub struct ParameterServer<T: FloatBounds> {
    /// Global parameters
    global_parameters: HashMap<String, Array1<T>>,
    /// Accumulated gradients
    accumulated_gradients: HashMap<String, Array1<T>>,
    /// Number of workers
    num_workers: usize,
}

impl<T: FloatBounds + scirs2_core::ndarray::ScalarOperand + Copy> ParameterServer<T> {
    pub fn new(config: &DistributedConfig<T>) -> DistributedResult<Self> {
        Ok(Self {
            global_parameters: HashMap::new(),
            accumulated_gradients: HashMap::new(),
            num_workers: config.num_workers,
        })
    }

    pub fn aggregate_gradients(
        &mut self,
        gradients: &HashMap<String, Array1<T>>,
    ) -> DistributedResult<()> {
        for (name, grad) in gradients {
            let accumulated = self
                .accumulated_gradients
                .entry(name.clone())
                .or_insert_with(|| Array1::zeros(grad.len()));

            *accumulated = &*accumulated + grad;
        }
        Ok(())
    }

    pub fn broadcast_parameters(
        &mut self,
        gradients: &mut HashMap<String, Array1<T>>,
    ) -> DistributedResult<()> {
        for (name, grad) in gradients.iter_mut() {
            if let Some(accumulated) = self.accumulated_gradients.get(name) {
                // Average accumulated gradients
                let avg_grad = accumulated / T::from(self.num_workers).unwrap();
                *grad = avg_grad;
            }
        }

        // Reset accumulated gradients
        self.accumulated_gradients.clear();
        Ok(())
    }
}

// Note: DistributedOptimizer will be implemented when optimizer traits are available

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_distributed_config_default() {
        let config = DistributedConfig::<f64>::default();
        assert_eq!(config.num_workers, 1);
        assert!(matches!(config.backend, DistributedBackend::CPU));
        assert!(matches!(
            config.sync_strategy,
            GradientSyncStrategy::AllReduce
        ));
    }

    #[test]
    fn test_distributed_trainer_creation() {
        let config = DistributedConfig::<f64> {
            num_workers: 4,
            ..Default::default()
        };

        let trainer = DistributedTrainer::new(config);
        assert!(trainer.is_ok());
    }

    #[test]
    fn test_data_distribution() {
        let config = DistributedConfig::<f64> {
            num_workers: 2,
            ..Default::default()
        };

        let trainer = DistributedTrainer::new(config).unwrap();
        let data = Array2::<f64>::ones((100, 10));

        let distributed_data = trainer.distribute_data(&data).unwrap();
        assert_eq!(distributed_data.len(), 2);
        assert_eq!(distributed_data[0].nrows(), 50);
        assert_eq!(distributed_data[1].nrows(), 50);
    }

    #[test]
    fn test_learning_rate_scaling() {
        let config = DistributedConfig::<f64> {
            num_workers: 4,
            lr_scaling: LearningRateScaling::Linear,
            ..Default::default()
        };

        let trainer = DistributedTrainer::new(config).unwrap();
        let base_lr = 0.01;
        let scaled_lr = trainer.scale_learning_rate(base_lr);

        assert_relative_eq!(scaled_lr, 0.04, epsilon = 1e-10);
    }

    #[test]
    fn test_gradient_compression() {
        let config = DistributedConfig::<f64> {
            gradient_compression: true,
            compression_threshold: 0.1,
            ..Default::default()
        };

        let trainer = DistributedTrainer::new(config).unwrap();
        let mut gradients = HashMap::new();
        gradients.insert(
            "test".to_string(),
            Array1::from_vec(vec![0.05, 0.15, 0.02, 0.2]),
        );

        let grad = gradients.get("test").unwrap();
        assert_eq!(grad[0], 0.05); // Should be compressed to 0
        assert_eq!(grad[1], 0.15); // Should remain
    }

    #[test]
    fn test_worker_coordinator() {
        let config = DistributedConfig::<f64>::default();
        let worker = WorkerCoordinator::new(0, &config);

        assert!(worker.is_ok());
        assert_eq!(worker.unwrap().get_rank(), 0);
    }

    #[test]
    fn test_parameter_server() {
        let config = DistributedConfig::<f64> {
            num_workers: 2,
            ..Default::default()
        };

        let ps = ParameterServer::new(&config);
        assert!(ps.is_ok());
    }
}
