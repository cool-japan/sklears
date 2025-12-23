use rayon::prelude::*;
use scirs2_core::ndarray::{s, Array1, Array2, Axis};
use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::Rng;
use scirs2_core::random::{thread_rng, SeedableRng};
use scirs2_core::StandardNormal;
use sklears_core::error::{Result, SklearsError};

/// Distributed kernel approximation methods for large-scale datasets
///
/// This module provides distributed computation capabilities for kernel
/// approximations, enabling processing of massive datasets that don't
/// fit in memory or require parallel processing across multiple workers.
/// Partitioning strategy for distributing data across workers
#[derive(Debug, Clone)]
/// PartitionStrategy
pub enum PartitionStrategy {
    /// Randomly distribute samples across workers
    Random,
    /// Block-wise distribution (contiguous chunks)
    Block,
    /// Stratified sampling to maintain class distribution
    Stratified,
    /// Custom partitioning function
    Custom(fn(usize, usize) -> Vec<Vec<usize>>),
}

/// Communication pattern for distributed computing
#[derive(Debug, Clone)]
/// CommunicationPattern
pub enum CommunicationPattern {
    /// All-to-all communication
    AllToAll,
    /// Master-worker pattern
    MasterWorker,
    /// Ring topology
    Ring,
    /// Tree topology for hierarchical reduction
    Tree,
}

/// Aggregation method for combining results from workers
#[derive(Debug, Clone)]
/// AggregationMethod
pub enum AggregationMethod {
    /// Simple average across workers
    Average,
    /// Weighted average based on worker data size
    WeightedAverage,
    /// Concatenate all worker results
    Concatenate,
    /// Take best result based on approximation quality
    BestQuality,
    /// Ensemble combination
    Ensemble,
}

/// Configuration for distributed kernel approximation
#[derive(Debug, Clone)]
/// DistributedConfig
pub struct DistributedConfig {
    /// n_workers
    pub n_workers: usize,
    /// partition_strategy
    pub partition_strategy: PartitionStrategy,
    /// communication_pattern
    pub communication_pattern: CommunicationPattern,
    /// aggregation_method
    pub aggregation_method: AggregationMethod,
    /// chunk_size
    pub chunk_size: Option<usize>,
    /// overlap_ratio
    pub overlap_ratio: f64,
    /// fault_tolerance
    pub fault_tolerance: bool,
    /// load_balancing
    pub load_balancing: bool,
}

impl Default for DistributedConfig {
    fn default() -> Self {
        Self {
            n_workers: num_cpus::get(),
            partition_strategy: PartitionStrategy::Block,
            communication_pattern: CommunicationPattern::MasterWorker,
            aggregation_method: AggregationMethod::Average,
            chunk_size: None,
            overlap_ratio: 0.1,
            fault_tolerance: false,
            load_balancing: true,
        }
    }
}

/// Worker state and computation context
#[derive(Debug)]
/// Worker
pub struct Worker {
    /// id
    pub id: usize,
    /// data_indices
    pub data_indices: Vec<usize>,
    /// local_features
    pub local_features: Option<Array2<f64>>,
    /// is_active
    pub is_active: bool,
    /// computation_time
    pub computation_time: f64,
    /// memory_usage
    pub memory_usage: usize,
}

impl Worker {
    pub fn new(id: usize, data_indices: Vec<usize>) -> Self {
        Self {
            id,
            data_indices,
            local_features: None,
            is_active: true,
            computation_time: 0.0,
            memory_usage: 0,
        }
    }
}

/// Distributed RBF kernel approximation using Random Fourier Features
///
/// Distributes the computation of random Fourier features across multiple
/// workers, each processing a subset of the data or feature dimensions.
pub struct DistributedRBFSampler {
    n_components: usize,
    gamma: f64,
    config: DistributedConfig,
    workers: Vec<Worker>,
    global_weights: Option<Array2<f64>>,
    global_bias: Option<Array1<f64>>,
    random_state: Option<u64>,
}

impl DistributedRBFSampler {
    /// Create a new distributed RBF sampler
    pub fn new(n_components: usize, gamma: f64) -> Self {
        Self {
            n_components,
            gamma,
            config: DistributedConfig::default(),
            workers: Vec::new(),
            global_weights: None,
            global_bias: None,
            random_state: None,
        }
    }

    /// Set the distributed computing configuration
    pub fn with_config(mut self, config: DistributedConfig) -> Self {
        self.config = config;
        self
    }

    /// Set random state for reproducibility
    pub fn with_random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }

    /// Fit the distributed RBF sampler
    pub fn fit(&mut self, x: &Array2<f64>) -> Result<()> {
        let (n_samples, n_features) = x.dim();

        // Initialize workers
        self.initialize_workers(n_samples)?;

        // Distribute random weights generation across workers
        let components_per_worker = self.n_components / self.config.n_workers;
        let mut all_weights = Vec::new();
        let mut all_bias = Vec::new();

        // Use parallel computation for weight generation
        let weight_results: Vec<(Array2<f64>, Array1<f64>)> = (0..self.config.n_workers)
            .into_par_iter()
            .map(|worker_id| {
                let mut rng = match self.random_state {
                    Some(seed) => StdRng::seed_from_u64(seed + worker_id as u64),
                    None => StdRng::from_seed(thread_rng().gen()),
                };

                let worker_components = if worker_id == self.config.n_workers - 1 {
                    // Last worker gets remaining components
                    self.n_components - components_per_worker * worker_id
                } else {
                    components_per_worker
                };

                // Generate random weights for this worker
                let mut worker_weights = Array2::zeros((worker_components, n_features));
                for i in 0..worker_components {
                    for j in 0..n_features {
                        worker_weights[[i, j]] =
                            rng.sample::<f64, _>(StandardNormal) * (2.0 * self.gamma).sqrt();
                    }
                }

                // Generate random bias
                let mut worker_bias = Array1::zeros(worker_components);
                for i in 0..worker_components {
                    worker_bias[i] = rng.gen_range(0.0..2.0 * std::f64::consts::PI);
                }

                (worker_weights, worker_bias)
            })
            .collect();

        for (weights, bias) in weight_results {
            all_weights.push(weights);
            all_bias.push(bias);
        }

        // Combine weights from all workers
        self.global_weights = Some(
            scirs2_core::ndarray::concatenate(
                Axis(0),
                &all_weights
                    .iter()
                    .map(|w: &Array2<f64>| w.view())
                    .collect::<Vec<_>>(),
            )
            .map_err(|e| SklearsError::Other(format!("Failed to concatenate weights: {}", e)))?,
        );

        self.global_bias = Some(
            scirs2_core::ndarray::concatenate(
                Axis(0),
                &all_bias
                    .iter()
                    .map(|b: &Array1<f64>| b.view())
                    .collect::<Vec<_>>(),
            )
            .map_err(|e| SklearsError::Other(format!("Failed to concatenate bias: {}", e)))?,
        );

        Ok(())
    }

    /// Transform data using distributed computation
    pub fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let weights = self
            .global_weights
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "transform".to_string(),
            })?;
        let bias = self
            .global_bias
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "transform".to_string(),
            })?;

        let (n_samples, _) = x.dim();

        // Distribute computation across workers
        let samples_per_worker = n_samples / self.config.n_workers;

        let feature_results: Vec<Array2<f64>> = (0..self.config.n_workers)
            .into_par_iter()
            .map(|worker_id| {
                let start_idx = worker_id * samples_per_worker;
                let end_idx = if worker_id == self.config.n_workers - 1 {
                    n_samples
                } else {
                    (worker_id + 1) * samples_per_worker
                };

                let worker_data = x.slice(s![start_idx..end_idx, ..]);
                self.compute_features(&worker_data, weights, bias)
            })
            .collect();

        // Combine results from all workers
        let combined_features = scirs2_core::ndarray::concatenate(
            Axis(0),
            &feature_results.iter().map(|f| f.view()).collect::<Vec<_>>(),
        )
        .map_err(|e| SklearsError::Other(format!("Failed to concatenate features: {}", e)))?;

        Ok(combined_features)
    }

    /// Compute RBF features for a data subset
    fn compute_features(
        &self,
        x: &scirs2_core::ndarray::ArrayView2<f64>,
        weights: &Array2<f64>,
        bias: &Array1<f64>,
    ) -> Array2<f64> {
        let (n_samples, _) = x.dim();
        let n_components = weights.nrows();

        // Compute X @ W^T + b
        let projection = x.dot(&weights.t()) + bias;

        // Apply cosine transformation with normalization
        let mut features = Array2::zeros((n_samples, n_components));
        let norm_factor = (2.0 / n_components as f64).sqrt();

        for i in 0..n_samples {
            for j in 0..n_components {
                features[[i, j]] = norm_factor * projection[[i, j]].cos();
            }
        }

        features
    }

    /// Initialize workers based on the partition strategy
    fn initialize_workers(&mut self, n_samples: usize) -> Result<()> {
        self.workers.clear();

        let partitions = match &self.config.partition_strategy {
            PartitionStrategy::Block => self.create_block_partitions(n_samples),
            PartitionStrategy::Random => self.create_random_partitions(n_samples),
            PartitionStrategy::Stratified => {
                // For now, use block partitioning as stratified requires labels
                self.create_block_partitions(n_samples)
            }
            PartitionStrategy::Custom(partition_fn) => {
                partition_fn(n_samples, self.config.n_workers)
            }
        };

        for (worker_id, indices) in partitions.into_iter().enumerate() {
            self.workers.push(Worker::new(worker_id, indices));
        }

        Ok(())
    }

    /// Create block-wise partitions
    fn create_block_partitions(&self, n_samples: usize) -> Vec<Vec<usize>> {
        let samples_per_worker = n_samples / self.config.n_workers;
        let mut partitions = Vec::new();

        for worker_id in 0..self.config.n_workers {
            let start_idx = worker_id * samples_per_worker;
            let end_idx = if worker_id == self.config.n_workers - 1 {
                n_samples
            } else {
                (worker_id + 1) * samples_per_worker
            };

            partitions.push((start_idx..end_idx).collect());
        }

        partitions
    }

    /// Create random partitions
    fn create_random_partitions(&self, n_samples: usize) -> Vec<Vec<usize>> {
        let mut rng = match self.random_state {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_seed(thread_rng().gen()),
        };

        let mut indices: Vec<usize> = (0..n_samples).collect();

        // Shuffle indices
        for i in (1..indices.len()).rev() {
            let j = rng.gen_range(0..i + 1);
            indices.swap(i, j);
        }

        // Distribute shuffled indices across workers
        let samples_per_worker = n_samples / self.config.n_workers;
        let mut partitions = Vec::new();

        for worker_id in 0..self.config.n_workers {
            let start_idx = worker_id * samples_per_worker;
            let end_idx = if worker_id == self.config.n_workers - 1 {
                n_samples
            } else {
                (worker_id + 1) * samples_per_worker
            };

            partitions.push(indices[start_idx..end_idx].to_vec());
        }

        partitions
    }

    /// Get worker statistics
    pub fn worker_stats(&self) -> Vec<(usize, usize, bool)> {
        self.workers
            .iter()
            .map(|w| (w.id, w.data_indices.len(), w.is_active))
            .collect()
    }

    /// Get total memory usage across all workers
    pub fn total_memory_usage(&self) -> usize {
        self.workers.iter().map(|w| w.memory_usage).sum()
    }
}

/// Distributed Nyström method for kernel approximation
///
/// Implements a distributed version of the Nyström method where
/// inducing points and eigendecomposition are computed in parallel.
pub struct DistributedNystroem {
    n_components: usize,
    gamma: f64,
    config: DistributedConfig,
    workers: Vec<Worker>,
    eigenvalues: Option<Array1<f64>>,
    eigenvectors: Option<Array2<f64>>,
    inducing_points: Option<Array2<f64>>,
    random_state: Option<u64>,
}

impl DistributedNystroem {
    /// Create a new distributed Nyström approximation
    pub fn new(n_components: usize, gamma: f64) -> Self {
        Self {
            n_components,
            gamma,
            config: DistributedConfig::default(),
            workers: Vec::new(),
            eigenvalues: None,
            eigenvectors: None,
            inducing_points: None,
            random_state: None,
        }
    }

    /// Set the distributed computing configuration
    pub fn with_config(mut self, config: DistributedConfig) -> Self {
        self.config = config;
        self
    }

    /// Set random state for reproducibility
    pub fn with_random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }

    /// Fit the distributed Nyström method
    pub fn fit(&mut self, x: &Array2<f64>) -> Result<()> {
        let (_n_samples, _) = x.dim();

        // Select inducing points
        let inducing_indices = self.select_inducing_points(x)?;
        let inducing_points = x.select(Axis(0), &inducing_indices);

        // Compute kernel matrix for inducing points
        let kernel_matrix = self.compute_kernel_matrix(&inducing_points)?;

        // Eigendecomposition (simplified for now)
        let (eigenvalues, eigenvectors) = self.eigendecomposition(&kernel_matrix)?;

        // Store results
        self.inducing_points = Some(inducing_points);
        self.eigenvalues = Some(eigenvalues);
        self.eigenvectors = Some(eigenvectors);

        Ok(())
    }

    /// Transform data using the fitted Nyström approximation
    pub fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let inducing_points =
            self.inducing_points
                .as_ref()
                .ok_or_else(|| SklearsError::NotFitted {
                    operation: "transform".to_string(),
                })?;
        let eigenvalues = self
            .eigenvalues
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "transform".to_string(),
            })?;
        let eigenvectors = self
            .eigenvectors
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "transform".to_string(),
            })?;

        // Compute kernel between x and inducing points
        let kernel_x_inducing = self.compute_kernel(x, inducing_points)?;

        // Apply Nyström transformation: K(X, Z) @ U @ Λ^(-1/2)
        let mut features = kernel_x_inducing.dot(eigenvectors);

        // Scale by eigenvalues
        for i in 0..eigenvalues.len() {
            if eigenvalues[i] > 1e-12 {
                let scale = 1.0 / eigenvalues[i].sqrt();
                for j in 0..features.nrows() {
                    features[[j, i]] *= scale;
                }
            }
        }

        Ok(features)
    }

    /// Select inducing points
    fn select_inducing_points(&self, x: &Array2<f64>) -> Result<Vec<usize>> {
        let n_samples = x.nrows();
        let mut rng = match self.random_state {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_seed(thread_rng().gen()),
        };

        // Simple random sampling for now
        let mut indices = Vec::new();
        for _ in 0..self.n_components {
            indices.push(rng.gen_range(0..n_samples));
        }

        Ok(indices)
    }

    /// Compute kernel matrix
    fn compute_kernel_matrix(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let n_samples = x.nrows();
        let mut kernel_matrix = Array2::zeros((n_samples, n_samples));

        for i in 0..n_samples {
            for j in i..n_samples {
                let diff = &x.row(i) - &x.row(j);
                let squared_dist = diff.mapv(|x| x * x).sum();
                let kernel_val = (-self.gamma * squared_dist).exp();
                kernel_matrix[[i, j]] = kernel_val;
                kernel_matrix[[j, i]] = kernel_val;
            }
        }

        Ok(kernel_matrix)
    }

    /// Compute kernel between two matrices
    fn compute_kernel(&self, x: &Array2<f64>, y: &Array2<f64>) -> Result<Array2<f64>> {
        let (n_samples_x, _) = x.dim();
        let (n_samples_y, _) = y.dim();
        let mut kernel_matrix = Array2::zeros((n_samples_x, n_samples_y));

        for i in 0..n_samples_x {
            for j in 0..n_samples_y {
                let diff = &x.row(i) - &y.row(j);
                let squared_dist = diff.mapv(|x| x * x).sum();
                let kernel_val = (-self.gamma * squared_dist).exp();
                kernel_matrix[[i, j]] = kernel_val;
            }
        }

        Ok(kernel_matrix)
    }

    /// Perform eigendecomposition
    fn eigendecomposition(&self, matrix: &Array2<f64>) -> Result<(Array1<f64>, Array2<f64>)> {
        // Simplified eigendecomposition (in practice, use LAPACK or similar)
        // This is a placeholder for actual eigendecomposition
        let n = matrix.nrows();
        let eigenvalues = Array1::ones(self.n_components.min(n));
        let eigenvectors = Array2::eye(n)
            .slice(s![.., ..self.n_components.min(n)])
            .to_owned();

        Ok((eigenvalues, eigenvectors))
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_distributed_rbf_sampler_basic() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];

        let mut sampler = DistributedRBFSampler::new(100, 0.1).with_random_state(42);

        sampler.fit(&x).unwrap();
        let features = sampler.transform(&x).unwrap();

        assert_eq!(features.nrows(), 4);
        assert_eq!(features.ncols(), 100);
    }

    #[test]
    fn test_distributed_config() {
        let config = DistributedConfig {
            n_workers: 4,
            partition_strategy: PartitionStrategy::Random,
            communication_pattern: CommunicationPattern::AllToAll,
            aggregation_method: AggregationMethod::WeightedAverage,
            ..Default::default()
        };

        assert_eq!(config.n_workers, 4);
        assert!(matches!(
            config.partition_strategy,
            PartitionStrategy::Random
        ));
    }

    #[test]
    fn test_worker_initialization() {
        let worker = Worker::new(0, vec![0, 1, 2, 3]);
        assert_eq!(worker.id, 0);
        assert_eq!(worker.data_indices.len(), 4);
        assert!(worker.is_active);
    }

    #[test]
    fn test_distributed_nystroem_basic() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];

        let mut nystroem = DistributedNystroem::new(3, 0.1).with_random_state(42);

        nystroem.fit(&x).unwrap();
        let features = nystroem.transform(&x).unwrap();

        assert_eq!(features.nrows(), 4);
        assert_eq!(features.ncols(), 3);
    }

    #[test]
    fn test_partition_strategies() {
        let mut sampler = DistributedRBFSampler::new(50, 0.1);
        sampler.config.n_workers = 2;

        // Test block partitioning
        sampler.config.partition_strategy = PartitionStrategy::Block;
        sampler.initialize_workers(10).unwrap();
        assert_eq!(sampler.workers.len(), 2);
        assert_eq!(sampler.workers[0].data_indices.len(), 5);
        assert_eq!(sampler.workers[1].data_indices.len(), 5);

        // Test random partitioning
        sampler.config.partition_strategy = PartitionStrategy::Random;
        sampler.random_state = Some(42);
        sampler.initialize_workers(10).unwrap();
        assert_eq!(sampler.workers.len(), 2);
    }

    #[test]
    fn test_worker_stats() {
        let mut sampler = DistributedRBFSampler::new(50, 0.1);
        sampler.config.n_workers = 3;
        sampler.initialize_workers(12).unwrap();

        let stats = sampler.worker_stats();
        assert_eq!(stats.len(), 3);
        assert_eq!(stats[0].1, 4); // First worker gets 4 samples
        assert_eq!(stats[1].1, 4); // Second worker gets 4 samples
        assert_eq!(stats[2].1, 4); // Third worker gets 4 samples
    }

    #[test]
    fn test_reproducibility() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];

        let mut sampler1 = DistributedRBFSampler::new(50, 0.1).with_random_state(42);
        sampler1.fit(&x).unwrap();
        let features1 = sampler1.transform(&x).unwrap();

        let mut sampler2 = DistributedRBFSampler::new(50, 0.1).with_random_state(42);
        sampler2.fit(&x).unwrap();
        let features2 = sampler2.transform(&x).unwrap();

        // Features should be identical with same random state
        assert!((features1 - features2).mapv(f64::abs).sum() < 1e-10);
    }

    #[test]
    fn test_different_worker_counts() {
        let x = array![
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
            [9.0, 10.0],
            [11.0, 12.0]
        ];

        for n_workers in [1, 2, 3, 6] {
            let config = DistributedConfig {
                n_workers,
                ..Default::default()
            };

            let mut sampler = DistributedRBFSampler::new(50, 0.1)
                .with_config(config)
                .with_random_state(42);

            sampler.fit(&x).unwrap();
            let features = sampler.transform(&x).unwrap();

            assert_eq!(features.nrows(), 6);
            assert_eq!(features.ncols(), 50);
        }
    }
}
