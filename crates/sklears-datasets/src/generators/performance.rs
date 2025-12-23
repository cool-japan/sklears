//! Performance-optimized dataset generation
//!
//! This module provides high-performance dataset generation capabilities including:
//! - Streaming dataset generation with lazy evaluation
//! - Parallel data generation using rayon
//! - Memory-efficient generation for large datasets
//! - Chunked processing for distributed systems
//! - Distributed dataset generation across multiple nodes

use crate::generators::basic::{make_blobs, make_classification, make_regression};
use scirs2_core::ndarray::{Array1, Array2};
use std::collections::HashMap;
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

/// Configuration for streaming dataset generation
#[derive(Debug, Clone)]
pub struct StreamConfig {
    /// Chunk size for streaming generation
    pub chunk_size: usize,
    /// Total number of samples to generate
    pub total_samples: usize,
    /// Random seed for reproducibility
    pub random_state: Option<u64>,
    /// Number of parallel workers
    pub n_workers: usize,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            chunk_size: 1000,
            total_samples: 10000,
            random_state: None,
            n_workers: num_cpus::get(),
        }
    }
}

/// Iterator for streaming dataset generation
pub struct DatasetStream<T> {
    config: StreamConfig,
    current_chunk: usize,
    total_chunks: usize,
    generator_fn: Box<dyn Fn(usize, usize, Option<u64>) -> T + Send + Sync>,
}

impl<T> DatasetStream<T> {
    fn new<F>(config: StreamConfig, generator_fn: F) -> Self
    where
        F: Fn(usize, usize, Option<u64>) -> T + Send + Sync + 'static,
    {
        let total_chunks = (config.total_samples + config.chunk_size - 1) / config.chunk_size;

        Self {
            config,
            current_chunk: 0,
            total_chunks,
            generator_fn: Box::new(generator_fn),
        }
    }
}

impl<T> Iterator for DatasetStream<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_chunk >= self.total_chunks {
            return None;
        }

        let chunk_start = self.current_chunk * self.config.chunk_size;
        let chunk_end = std::cmp::min(
            chunk_start + self.config.chunk_size,
            self.config.total_samples,
        );
        let chunk_size = chunk_end - chunk_start;

        // Generate unique seed for this chunk if random_state is provided
        let chunk_seed = self
            .config
            .random_state
            .map(|seed| seed + self.current_chunk as u64);

        let result = (self.generator_fn)(chunk_size, self.current_chunk, chunk_seed);
        self.current_chunk += 1;

        Some(result)
    }
}

/// Streaming classification dataset generator
pub fn stream_classification(
    n_features: usize,
    n_classes: usize,
    config: StreamConfig,
) -> DatasetStream<(Array2<f64>, Array1<i32>)> {
    DatasetStream::new(config, move |chunk_size, _chunk_idx, seed| {
        make_classification(
            chunk_size, n_features, n_features, // n_informative
            0,          // n_redundant
            n_classes, seed,
        )
        .unwrap()
    })
}

/// Streaming regression dataset generator
pub fn stream_regression(
    n_features: usize,
    config: StreamConfig,
) -> DatasetStream<(Array2<f64>, Array1<f64>)> {
    DatasetStream::new(config, move |chunk_size, _chunk_idx, seed| {
        make_regression(
            chunk_size, n_features, n_features, // n_informative
            0.1,        // noise
            seed,
        )
        .unwrap()
    })
}

/// Streaming blob dataset generator
pub fn stream_blobs(
    n_features: usize,
    centers: usize,
    config: StreamConfig,
) -> DatasetStream<(Array2<f64>, Array1<i32>)> {
    DatasetStream::new(config, move |chunk_size, _chunk_idx, seed| {
        make_blobs(
            chunk_size, n_features, centers, 1.0, // cluster_std
            seed,
        )
        .unwrap()
    })
}

/// Parallel dataset generation result
#[derive(Debug)]
pub struct ParallelGenerationResult<T> {
    pub chunks: Vec<T>,
    pub generation_time: std::time::Duration,
    pub n_workers_used: usize,
}

/// Generate datasets in parallel using multiple threads
pub fn parallel_generate<T, F>(
    n_samples: usize,
    n_workers: usize,
    generator_fn: F,
) -> Result<ParallelGenerationResult<T>, Box<dyn std::error::Error + Send + Sync>>
where
    T: Send + 'static,
    F: Fn(usize, Option<u64>) -> Result<T, Box<dyn std::error::Error + Send + Sync>>
        + Send
        + Sync
        + Copy
        + 'static,
{
    let start_time = std::time::Instant::now();

    let chunk_size = (n_samples + n_workers - 1) / n_workers;
    let (tx, rx) = mpsc::channel();

    let mut handles = Vec::new();

    for worker_id in 0..n_workers {
        let tx = tx.clone();
        let handle = thread::spawn(move || {
            let chunk_start = worker_id * chunk_size;
            let chunk_end = std::cmp::min(chunk_start + chunk_size, n_samples);
            let actual_chunk_size = chunk_end - chunk_start;

            if actual_chunk_size == 0 {
                return;
            }

            // Use worker_id as part of the seed for reproducibility
            let seed = Some(worker_id as u64 * 12345);

            match generator_fn(actual_chunk_size, seed) {
                Ok(result) => {
                    if tx.send((worker_id, Ok(result))).is_err() {
                        eprintln!("Failed to send result from worker {}", worker_id);
                    }
                }
                Err(e) => {
                    if tx.send((worker_id, Err(e))).is_err() {
                        eprintln!("Failed to send error from worker {}", worker_id);
                    }
                }
            }
        });
        handles.push(handle);
    }

    // Drop the sender so the receiver knows when all workers are done
    drop(tx);

    // Collect results in order
    let mut results: Vec<Option<T>> = (0..n_workers).map(|_| None).collect();
    let mut successful_workers = 0;

    for (worker_id, result) in rx {
        match result {
            Ok(data) => {
                results[worker_id] = Some(data);
                successful_workers += 1;
            }
            Err(e) => {
                return Err(format!("Worker {} failed: {}", worker_id, e).into());
            }
        }
    }

    // Wait for all threads to complete
    for handle in handles {
        handle.join().map_err(|_| "Thread panicked")?;
    }

    // Filter out None values and collect results
    let chunks: Vec<T> = results.into_iter().flatten().collect();

    let generation_time = start_time.elapsed();

    Ok(ParallelGenerationResult {
        chunks,
        generation_time,
        n_workers_used: successful_workers,
    })
}

/// Parallel classification dataset generation
pub fn parallel_classification(
    n_samples: usize,
    n_features: usize,
    n_classes: usize,
    n_workers: usize,
) -> Result<
    ParallelGenerationResult<(Array2<f64>, Array1<i32>)>,
    Box<dyn std::error::Error + Send + Sync>,
> {
    parallel_generate(n_samples, n_workers, move |chunk_size, seed| {
        make_classification(
            chunk_size, n_features, n_features, // n_informative
            0,          // n_redundant
            n_classes, seed,
        )
        .map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)
    })
}

/// Parallel regression dataset generation
pub fn parallel_regression(
    n_samples: usize,
    n_features: usize,
    n_workers: usize,
) -> Result<
    ParallelGenerationResult<(Array2<f64>, Array1<f64>)>,
    Box<dyn std::error::Error + Send + Sync>,
> {
    parallel_generate(n_samples, n_workers, move |chunk_size, seed| {
        make_regression(
            chunk_size, n_features, n_features, // n_informative
            0.1,        // noise
            seed,
        )
        .map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)
    })
}

/// Parallel blob dataset generation
pub fn parallel_blobs(
    n_samples: usize,
    n_features: usize,
    centers: usize,
    n_workers: usize,
) -> Result<
    ParallelGenerationResult<(Array2<f64>, Array1<i32>)>,
    Box<dyn std::error::Error + Send + Sync>,
> {
    parallel_generate(n_samples, n_workers, move |chunk_size, seed| {
        make_blobs(
            chunk_size, n_features, centers, 1.0, // cluster_std
            seed,
        )
        .map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)
    })
}

/// Memory-efficient dataset generator that yields chunks on demand
pub struct LazyDatasetGenerator<T> {
    chunk_size: usize,
    total_samples: usize,
    generated_samples: usize,
    generator_fn:
        Box<dyn Fn(usize, Option<u64>) -> Result<T, Box<dyn std::error::Error + Send + Sync>>>,
    random_state: Option<u64>,
}

impl<T> LazyDatasetGenerator<T> {
    pub fn new<F>(
        total_samples: usize,
        chunk_size: usize,
        random_state: Option<u64>,
        generator_fn: F,
    ) -> Self
    where
        F: Fn(usize, Option<u64>) -> Result<T, Box<dyn std::error::Error + Send + Sync>> + 'static,
    {
        Self {
            chunk_size,
            total_samples,
            generated_samples: 0,
            generator_fn: Box::new(generator_fn),
            random_state,
        }
    }

    /// Generate the next chunk of data
    pub fn next_chunk(&mut self) -> Option<Result<T, Box<dyn std::error::Error + Send + Sync>>> {
        if self.generated_samples >= self.total_samples {
            return None;
        }

        let remaining_samples = self.total_samples - self.generated_samples;
        let current_chunk_size = std::cmp::min(self.chunk_size, remaining_samples);

        // Generate seed based on current position for reproducibility
        let seed = self.random_state.map(|s| s + self.generated_samples as u64);

        let result = (self.generator_fn)(current_chunk_size, seed);
        self.generated_samples += current_chunk_size;

        Some(result)
    }

    /// Get progress information
    pub fn progress(&self) -> (usize, usize, f64) {
        let progress_ratio = self.generated_samples as f64 / self.total_samples as f64;
        (self.generated_samples, self.total_samples, progress_ratio)
    }

    /// Check if generation is complete
    pub fn is_complete(&self) -> bool {
        self.generated_samples >= self.total_samples
    }
}

/// Create a lazy classification dataset generator
pub fn lazy_classification(
    total_samples: usize,
    n_features: usize,
    n_classes: usize,
    chunk_size: usize,
    random_state: Option<u64>,
) -> LazyDatasetGenerator<(Array2<f64>, Array1<i32>)> {
    LazyDatasetGenerator::new(
        total_samples,
        chunk_size,
        random_state,
        move |chunk_size, seed| {
            make_classification(
                chunk_size, n_features, n_features, // n_informative
                0,          // n_redundant
                n_classes, seed,
            )
            .map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)
        },
    )
}

/// Create a lazy regression dataset generator
pub fn lazy_regression(
    total_samples: usize,
    n_features: usize,
    chunk_size: usize,
    random_state: Option<u64>,
) -> LazyDatasetGenerator<(Array2<f64>, Array1<f64>)> {
    LazyDatasetGenerator::new(
        total_samples,
        chunk_size,
        random_state,
        move |chunk_size, seed| {
            make_regression(
                chunk_size, n_features, n_features, // n_informative
                0.1,        // noise
                seed,
            )
            .map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)
        },
    )
}

/// Configuration for distributed dataset generation
#[derive(Debug, Clone)]
pub struct DistributedConfig {
    /// Total number of samples to generate
    pub total_samples: usize,
    /// Number of nodes in the distributed cluster
    pub n_nodes: usize,
    /// Node identifier (0 to n_nodes-1)
    pub node_id: usize,
    /// Random seed for reproducibility
    pub random_state: Option<u64>,
    /// Timeout for node communication
    pub timeout: Duration,
    /// Load balancing strategy
    pub load_balancing: LoadBalancingStrategy,
}

/// Load balancing strategies for distributed generation
#[derive(Debug, Clone)]
pub enum LoadBalancingStrategy {
    /// Equal distribution of samples across nodes
    EqualSplit,
    /// Weighted distribution based on node capabilities
    Weighted(Vec<f64>),
    /// Dynamic load balancing based on node performance
    Dynamic,
}

impl Default for DistributedConfig {
    fn default() -> Self {
        Self {
            total_samples: 100000,
            n_nodes: 1,
            node_id: 0,
            random_state: None,
            timeout: Duration::from_secs(300), // 5 minutes
            load_balancing: LoadBalancingStrategy::EqualSplit,
        }
    }
}

/// Node information for distributed generation
#[derive(Debug, Clone)]
pub struct NodeInfo {
    pub node_id: usize,
    pub samples_assigned: usize,
    pub samples_generated: usize,
    pub status: NodeStatus,
    pub start_time: Option<Instant>,
    pub completion_time: Option<Instant>,
}

/// Status of a node in distributed generation
#[derive(Debug, Clone, PartialEq)]
pub enum NodeStatus {
    /// Idle
    Idle,
    /// Working
    Working,
    /// Completed
    Completed,
    /// Failed
    Failed,
}

/// Result of distributed dataset generation
#[derive(Debug)]
pub struct DistributedGenerationResult<T> {
    pub data: T,
    pub node_results: HashMap<usize, NodeResult<T>>,
    pub total_generation_time: Duration,
    pub coordination_overhead: Duration,
    pub n_nodes_used: usize,
    pub load_balance_efficiency: f64,
}

/// Result from a single node
#[derive(Debug)]
pub struct NodeResult<T> {
    pub node_id: usize,
    pub data: T,
    pub generation_time: Duration,
    pub samples_generated: usize,
}

/// Distributed dataset generator coordinator
#[derive(Debug)]
pub struct DistributedGenerator {
    config: DistributedConfig,
    nodes: HashMap<usize, NodeInfo>,
}

impl DistributedGenerator {
    /// Create a new distributed generator
    pub fn new(
        config: DistributedConfig,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        if config.node_id >= config.n_nodes {
            return Err("Node ID must be less than total number of nodes".into());
        }

        let mut nodes = HashMap::new();
        for i in 0..config.n_nodes {
            nodes.insert(
                i,
                NodeInfo {
                    node_id: i,
                    samples_assigned: 0,
                    samples_generated: 0,
                    status: NodeStatus::Idle,
                    start_time: None,
                    completion_time: None,
                },
            );
        }

        Ok(Self { config, nodes })
    }

    /// Calculate sample distribution across nodes
    pub fn calculate_sample_distribution(
        &mut self,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        match &self.config.load_balancing {
            LoadBalancingStrategy::EqualSplit => {
                let base_samples = self.config.total_samples / self.config.n_nodes;
                let remainder = self.config.total_samples % self.config.n_nodes;

                for i in 0..self.config.n_nodes {
                    if let Some(node) = self.nodes.get_mut(&i) {
                        node.samples_assigned = base_samples + if i < remainder { 1 } else { 0 };
                    }
                }
            }
            LoadBalancingStrategy::Weighted(weights) => {
                if weights.len() != self.config.n_nodes {
                    return Err("Number of weights must match number of nodes".into());
                }

                let total_weight: f64 = weights.iter().sum();
                if total_weight <= 0.0 {
                    return Err("Total weight must be positive".into());
                }

                let mut assigned_samples = 0;
                // Iterate through nodes in order (0 to n_nodes-1) to ensure deterministic assignment
                for i in 0..self.config.n_nodes {
                    if let Some(node) = self.nodes.get_mut(&i) {
                        if i < weights.len() - 1 {
                            node.samples_assigned = ((weights[i] / total_weight)
                                * self.config.total_samples as f64)
                                as usize;
                            assigned_samples += node.samples_assigned;
                        } else {
                            // Assign remaining samples to last node to ensure exact total
                            node.samples_assigned = self.config.total_samples - assigned_samples;
                        }
                    }
                }
            }
            LoadBalancingStrategy::Dynamic => {
                // Start with equal split, can be adjusted during runtime
                let base_samples = self.config.total_samples / self.config.n_nodes;
                let remainder = self.config.total_samples % self.config.n_nodes;

                for i in 0..self.config.n_nodes {
                    if let Some(node) = self.nodes.get_mut(&i) {
                        node.samples_assigned = base_samples + if i < remainder { 1 } else { 0 };
                    }
                }
            }
        }

        Ok(())
    }

    /// Get the samples assigned to current node
    pub fn get_current_node_samples(&self) -> usize {
        self.nodes
            .get(&self.config.node_id)
            .map(|node| node.samples_assigned)
            .unwrap_or(0)
    }

    /// Mark current node as working
    pub fn start_generation(&mut self) {
        if let Some(node) = self.nodes.get_mut(&self.config.node_id) {
            node.status = NodeStatus::Working;
            node.start_time = Some(Instant::now());
        }
    }

    /// Mark current node as completed
    pub fn complete_generation(&mut self, samples_generated: usize) {
        if let Some(node) = self.nodes.get_mut(&self.config.node_id) {
            node.status = NodeStatus::Completed;
            node.samples_generated = samples_generated;
            node.completion_time = Some(Instant::now());
        }
    }

    /// Calculate load balance efficiency
    pub fn calculate_load_balance_efficiency(&self) -> f64 {
        let completed_nodes: Vec<_> = self
            .nodes
            .values()
            .filter(|node| node.status == NodeStatus::Completed)
            .collect();

        if completed_nodes.is_empty() {
            return 0.0;
        }

        let generation_times: Vec<Duration> = completed_nodes
            .iter()
            .filter_map(|node| {
                if let (Some(start), Some(end)) = (node.start_time, node.completion_time) {
                    Some(end - start)
                } else {
                    None
                }
            })
            .collect();

        if generation_times.is_empty() {
            return 0.0;
        }

        let total_time: Duration = generation_times.iter().sum();
        let avg_time = total_time / generation_times.len() as u32;
        let max_time = generation_times.iter().max().unwrap();

        if max_time.as_nanos() == 0 {
            return 1.0;
        }

        (avg_time.as_nanos() as f64) / (max_time.as_nanos() as f64)
    }
}

/// Generate distributed classification dataset
pub fn distributed_classification(
    n_features: usize,
    n_classes: usize,
    config: DistributedConfig,
) -> Result<
    DistributedGenerationResult<(Array2<f64>, Array1<i32>)>,
    Box<dyn std::error::Error + Send + Sync>,
> {
    let start_time = Instant::now();

    let mut generator = DistributedGenerator::new(config.clone())?;
    generator.calculate_sample_distribution()?;

    let samples_for_this_node = generator.get_current_node_samples();

    // Generate unique seed for this node
    let node_seed = config
        .random_state
        .map(|seed| seed + config.node_id as u64 * 12345);

    generator.start_generation();

    // Generate data for this node
    let generation_start = Instant::now();
    let (x, y) = make_classification(
        samples_for_this_node,
        n_features,
        n_features, // n_informative
        0,          // n_redundant
        n_classes,
        node_seed,
    )?;
    let generation_time = generation_start.elapsed();

    generator.complete_generation(samples_for_this_node);

    // Create node result
    let node_result = NodeResult {
        node_id: config.node_id,
        data: (x.clone(), y.clone()),
        generation_time,
        samples_generated: samples_for_this_node,
    };

    let mut node_results = HashMap::new();
    node_results.insert(config.node_id, node_result);

    let total_generation_time = start_time.elapsed();
    let coordination_overhead = total_generation_time - generation_time;
    let load_balance_efficiency = generator.calculate_load_balance_efficiency();

    Ok(DistributedGenerationResult {
        data: (x, y),
        node_results,
        total_generation_time,
        coordination_overhead,
        n_nodes_used: 1, // Only current node in this implementation
        load_balance_efficiency,
    })
}

/// Generate distributed regression dataset
pub fn distributed_regression(
    n_features: usize,
    config: DistributedConfig,
) -> Result<
    DistributedGenerationResult<(Array2<f64>, Array1<f64>)>,
    Box<dyn std::error::Error + Send + Sync>,
> {
    let start_time = Instant::now();

    let mut generator = DistributedGenerator::new(config.clone())?;
    generator.calculate_sample_distribution()?;

    let samples_for_this_node = generator.get_current_node_samples();

    // Generate unique seed for this node
    let node_seed = config
        .random_state
        .map(|seed| seed + config.node_id as u64 * 12345);

    generator.start_generation();

    // Generate data for this node
    let generation_start = Instant::now();
    let (x, y) = make_regression(
        samples_for_this_node,
        n_features,
        n_features, // n_informative
        0.1,        // noise
        node_seed,
    )?;
    let generation_time = generation_start.elapsed();

    generator.complete_generation(samples_for_this_node);

    // Create node result
    let node_result = NodeResult {
        node_id: config.node_id,
        data: (x.clone(), y.clone()),
        generation_time,
        samples_generated: samples_for_this_node,
    };

    let mut node_results = HashMap::new();
    node_results.insert(config.node_id, node_result);

    let total_generation_time = start_time.elapsed();
    let coordination_overhead = total_generation_time - generation_time;
    let load_balance_efficiency = generator.calculate_load_balance_efficiency();

    Ok(DistributedGenerationResult {
        data: (x, y),
        node_results,
        total_generation_time,
        coordination_overhead,
        n_nodes_used: 1, // Only current node in this implementation
        load_balance_efficiency,
    })
}

/// Generate distributed blob dataset
pub fn distributed_blobs(
    n_features: usize,
    centers: usize,
    config: DistributedConfig,
) -> Result<
    DistributedGenerationResult<(Array2<f64>, Array1<i32>)>,
    Box<dyn std::error::Error + Send + Sync>,
> {
    let start_time = Instant::now();

    let mut generator = DistributedGenerator::new(config.clone())?;
    generator.calculate_sample_distribution()?;

    let samples_for_this_node = generator.get_current_node_samples();

    // Generate unique seed for this node
    let node_seed = config
        .random_state
        .map(|seed| seed + config.node_id as u64 * 12345);

    generator.start_generation();

    // Generate data for this node
    let generation_start = Instant::now();
    let (x, y) = make_blobs(
        samples_for_this_node,
        n_features,
        centers,
        1.0, // cluster_std
        node_seed,
    )?;
    let generation_time = generation_start.elapsed();

    generator.complete_generation(samples_for_this_node);

    // Create node result
    let node_result = NodeResult {
        node_id: config.node_id,
        data: (x.clone(), y.clone()),
        generation_time,
        samples_generated: samples_for_this_node,
    };

    let mut node_results = HashMap::new();
    node_results.insert(config.node_id, node_result);

    let total_generation_time = start_time.elapsed();
    let coordination_overhead = total_generation_time - generation_time;
    let load_balance_efficiency = generator.calculate_load_balance_efficiency();

    Ok(DistributedGenerationResult {
        data: (x, y),
        node_results,
        total_generation_time,
        coordination_overhead,
        n_nodes_used: 1, // Only current node in this implementation
        load_balance_efficiency,
    })
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stream_classification() {
        let config = StreamConfig {
            chunk_size: 100,
            total_samples: 300,
            random_state: Some(42),
            n_workers: 2,
        };

        let stream = stream_classification(4, 3, config);
        let mut total_samples = 0;

        for (i, (x, y)) in stream.enumerate() {
            assert_eq!(x.ncols(), 4); // 4 features
            assert!(y.iter().all(|&label| label < 3)); // 3 classes (0, 1, 2)

            if i < 2 {
                assert_eq!(x.nrows(), 100); // First two chunks should be full size
                assert_eq!(y.len(), 100);
            } else {
                assert_eq!(x.nrows(), 100); // Last chunk should be remaining samples
                assert_eq!(y.len(), 100);
            }

            total_samples += x.nrows();
        }

        assert_eq!(total_samples, 300);
    }

    #[test]
    fn test_parallel_classification() {
        let result = parallel_classification(1000, 5, 3, 4).unwrap();

        assert_eq!(result.n_workers_used, 4);
        assert_eq!(result.chunks.len(), 4);

        let total_samples: usize = result.chunks.iter().map(|(x, _)| x.nrows()).sum();
        assert_eq!(total_samples, 1000);

        // Verify all chunks have correct number of features
        for (x, y) in &result.chunks {
            assert_eq!(x.ncols(), 5);
            assert!(y.iter().all(|&label| label < 3));
        }
    }

    #[test]
    fn test_lazy_generator() {
        let mut generator = lazy_classification(500, 3, 2, 150, Some(42));

        let mut total_samples = 0;
        let mut chunk_count = 0;

        while !generator.is_complete() {
            if let Some(result) = generator.next_chunk() {
                let (x, y) = result.unwrap();
                assert_eq!(x.ncols(), 3);
                assert!(y.iter().all(|&label| label < 2));

                total_samples += x.nrows();
                chunk_count += 1;

                let (generated, total, progress) = generator.progress();
                assert_eq!(generated, total_samples);
                assert_eq!(total, 500);
                assert!((0.0..=1.0).contains(&progress));
            } else {
                break;
            }
        }

        assert_eq!(total_samples, 500);
        assert_eq!(chunk_count, 4); // 150 + 150 + 150 + 50
        assert!(generator.is_complete());
    }

    #[test]
    fn test_stream_config_default() {
        let config = StreamConfig::default();
        assert_eq!(config.chunk_size, 1000);
        assert_eq!(config.total_samples, 10000);
        assert!(config.random_state.is_none());
        assert!(config.n_workers > 0);
    }

    #[test]
    fn test_parallel_generation_timing() {
        let start = std::time::Instant::now();
        let result = parallel_regression(2000, 10, 2).unwrap();
        let sequential_time = start.elapsed();

        assert!(result.generation_time <= sequential_time * 2); // Should be reasonably fast
        assert_eq!(result.n_workers_used, 2);

        let total_samples: usize = result.chunks.iter().map(|(x, _)| x.nrows()).sum();
        assert_eq!(total_samples, 2000);
    }

    #[test]
    fn test_distributed_config_default() {
        let config = DistributedConfig::default();
        assert_eq!(config.total_samples, 100000);
        assert_eq!(config.n_nodes, 1);
        assert_eq!(config.node_id, 0);
        assert!(config.random_state.is_none());
        assert_eq!(config.timeout, Duration::from_secs(300));
        assert!(matches!(
            config.load_balancing,
            LoadBalancingStrategy::EqualSplit
        ));
    }

    #[test]
    fn test_distributed_generator_sample_distribution() {
        let mut config = DistributedConfig::default();
        config.total_samples = 1000;
        config.n_nodes = 3;
        config.node_id = 0;

        let mut generator = DistributedGenerator::new(config).unwrap();
        generator.calculate_sample_distribution().unwrap();

        // Check equal split: 1000 / 3 = 333 remainder 1
        // Node 0: 334, Node 1: 333, Node 2: 333
        assert_eq!(generator.nodes[&0].samples_assigned, 334);
        assert_eq!(generator.nodes[&1].samples_assigned, 333);
        assert_eq!(generator.nodes[&2].samples_assigned, 333);

        let total_assigned: usize = generator
            .nodes
            .values()
            .map(|node| node.samples_assigned)
            .sum();
        assert_eq!(total_assigned, 1000);
    }

    #[test]
    fn test_distributed_generator_weighted_distribution() {
        let mut config = DistributedConfig::default();
        config.total_samples = 1000;
        config.n_nodes = 3;
        config.node_id = 0;
        config.load_balancing = LoadBalancingStrategy::Weighted(vec![0.5, 0.3, 0.2]);

        let mut generator = DistributedGenerator::new(config).unwrap();
        generator.calculate_sample_distribution().unwrap();

        // Check weighted split: 500, 300, 200
        assert_eq!(generator.nodes[&0].samples_assigned, 500);
        assert_eq!(generator.nodes[&1].samples_assigned, 300);
        assert_eq!(generator.nodes[&2].samples_assigned, 200);

        let total_assigned: usize = generator
            .nodes
            .values()
            .map(|node| node.samples_assigned)
            .sum();
        assert_eq!(total_assigned, 1000);
    }

    #[test]
    fn test_distributed_classification() {
        let config = DistributedConfig {
            total_samples: 1000,
            n_nodes: 4,
            node_id: 1,
            random_state: Some(42),
            ..Default::default()
        };

        let result = distributed_classification(5, 3, config).unwrap();

        // Check that node 1 gets approximately 250 samples (1000 / 4)
        assert_eq!(result.data.0.nrows(), 250);
        assert_eq!(result.data.1.len(), 250);
        assert_eq!(result.data.0.ncols(), 5);
        assert!(result.data.1.iter().all(|&label| label < 3));

        assert_eq!(result.n_nodes_used, 1);
        assert!(result.node_results.contains_key(&1));
        assert_eq!(result.node_results[&1].samples_generated, 250);
        assert!(result.total_generation_time > Duration::from_nanos(0));
    }

    #[test]
    fn test_distributed_regression() {
        let config = DistributedConfig {
            total_samples: 800,
            n_nodes: 2,
            node_id: 0,
            random_state: Some(123),
            ..Default::default()
        };

        let result = distributed_regression(7, config).unwrap();

        // Check that node 0 gets 400 samples (800 / 2)
        assert_eq!(result.data.0.nrows(), 400);
        assert_eq!(result.data.1.len(), 400);
        assert_eq!(result.data.0.ncols(), 7);

        assert_eq!(result.n_nodes_used, 1);
        assert!(result.node_results.contains_key(&0));
        assert_eq!(result.node_results[&0].samples_generated, 400);
        assert!(result.load_balance_efficiency >= 0.0);
        assert!(result.load_balance_efficiency <= 1.0);
    }

    #[test]
    fn test_distributed_blobs() {
        let config = DistributedConfig {
            total_samples: 600,
            n_nodes: 3,
            node_id: 2,
            random_state: Some(456),
            ..Default::default()
        };

        let result = distributed_blobs(4, 5, config).unwrap();

        // Check that node 2 gets 200 samples (600 / 3)
        assert_eq!(result.data.0.nrows(), 200);
        assert_eq!(result.data.1.len(), 200);
        assert_eq!(result.data.0.ncols(), 4);
        assert!(result.data.1.iter().all(|&label| label < 5));

        assert_eq!(result.n_nodes_used, 1);
        assert!(result.node_results.contains_key(&2));
        assert_eq!(result.node_results[&2].samples_generated, 200);
        assert!(result.coordination_overhead >= Duration::from_nanos(0));
    }

    #[test]
    fn test_distributed_generator_invalid_node_id() {
        let config = DistributedConfig {
            total_samples: 1000,
            n_nodes: 3,
            node_id: 3, // Invalid: should be 0, 1, or 2
            ..Default::default()
        };

        let result = DistributedGenerator::new(config);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Node ID must be less than total number of nodes"));
    }

    #[test]
    fn test_distributed_generator_weighted_validation() {
        let mut config = DistributedConfig::default();
        config.n_nodes = 3;
        config.load_balancing = LoadBalancingStrategy::Weighted(vec![0.5, 0.3]); // Wrong length

        let mut generator = DistributedGenerator::new(config).unwrap();
        let result = generator.calculate_sample_distribution();
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Number of weights must match number of nodes"));
    }

    #[test]
    fn test_load_balance_efficiency_calculation() {
        let mut config = DistributedConfig::default();
        config.n_nodes = 2;
        config.node_id = 0;

        let mut generator = DistributedGenerator::new(config).unwrap();

        // Simulate nodes completing at different times
        generator.nodes.get_mut(&0).unwrap().status = NodeStatus::Completed;
        generator.nodes.get_mut(&0).unwrap().start_time =
            Some(Instant::now() - Duration::from_millis(100));
        generator.nodes.get_mut(&0).unwrap().completion_time = Some(Instant::now());

        generator.nodes.get_mut(&1).unwrap().status = NodeStatus::Completed;
        generator.nodes.get_mut(&1).unwrap().start_time =
            Some(Instant::now() - Duration::from_millis(200));
        generator.nodes.get_mut(&1).unwrap().completion_time = Some(Instant::now());

        let efficiency = generator.calculate_load_balance_efficiency();
        assert!(efficiency >= 0.0);
        assert!(efficiency <= 1.0);
    }
}
