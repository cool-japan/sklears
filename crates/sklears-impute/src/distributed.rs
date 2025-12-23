//! Distributed imputation algorithms for large-scale missing data processing
//!
//! This module provides distributed implementations that can process datasets
//! across multiple machines or cores, enabling imputation of very large datasets
//! that don't fit on a single machine.

// ✅ SciRS2 Policy compliant imports
use scirs2_core::ndarray::{s, Array1, Array2, ArrayView1, ArrayView2, Axis};
// use scirs2_core::parallel::{LoadBalancer}; // Note: ParallelExecutor, ChunkStrategy not available
// use scirs2_core::memory_efficient::{ChunkedArray, AdaptiveChunking}; // Note: memory_efficient feature-gated
// use scirs2_core::simd::{SimdOps}; // Note: SimdArray not available

use crate::core::{ImputationError, ImputationMetadata, Imputer};
use crate::simple::SimpleImputer;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Transform, Untrained},
    types::Float,
};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use std::thread;
use std::time::{Duration, Instant};

/// Configuration for distributed imputation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedConfig {
    /// Number of worker nodes/processes
    pub num_workers: usize,
    /// Chunk size for data partitioning
    pub chunk_size: usize,
    /// Communication strategy between workers
    pub communication_strategy: CommunicationStrategy,
    /// Load balancing enabled
    pub load_balancing: bool,
    /// Fault tolerance enabled
    pub fault_tolerance: bool,
    /// Maximum retry attempts for failed operations
    pub max_retries: usize,
    /// Timeout for worker operations (in seconds)
    pub worker_timeout: Duration,
}

impl Default for DistributedConfig {
    fn default() -> Self {
        Self {
            num_workers: num_cpus::get(),
            chunk_size: 10000,
            communication_strategy: CommunicationStrategy::SharedMemory,
            load_balancing: true,
            fault_tolerance: true,
            max_retries: 3,
            worker_timeout: Duration::from_secs(300),
        }
    }
}

/// Communication strategies for distributed processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CommunicationStrategy {
    /// Use shared memory for communication (single machine)
    SharedMemory,
    /// Use message passing for communication (multi-machine)
    MessagePassing,
    /// Use parameter server architecture
    ParameterServer,
    /// Use all-reduce communication pattern
    AllReduce,
}

/// Distributed data partition
#[derive(Debug, Clone)]
pub struct DataPartition {
    /// Partition identifier
    pub id: usize,
    /// Start row index
    pub start_row: usize,
    /// End row index
    pub end_row: usize,
    /// Data chunk
    pub data: Array2<f64>,
    /// Missing value mask
    pub missing_mask: Array2<bool>,
    /// Partition metadata
    pub metadata: PartitionMetadata,
}

/// Metadata for data partitions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartitionMetadata {
    /// partition_id
    pub partition_id: usize,
    /// worker_id
    pub worker_id: usize,
    /// num_samples
    pub num_samples: usize,
    /// num_features
    pub num_features: usize,
    /// missing_ratio
    pub missing_ratio: f64,
    /// processing_time
    pub processing_time: Duration,
    /// memory_usage
    pub memory_usage: usize,
}

/// Worker node for distributed processing
pub struct DistributedWorker {
    /// id
    pub id: usize,
    /// config
    pub config: DistributedConfig,
    /// partitions
    pub partitions: Vec<DataPartition>,
    /// local_imputer
    pub local_imputer: Box<dyn Imputer + Send + Sync>,
    /// statistics
    pub statistics: WorkerStatistics,
}

/// Statistics tracked by each worker
#[derive(Debug, Default, Clone)]
pub struct WorkerStatistics {
    /// samples_processed
    pub samples_processed: usize,
    /// features_imputed
    pub features_imputed: usize,
    /// processing_time
    pub processing_time: Duration,
    /// memory_peak
    pub memory_peak: usize,
    /// errors_count
    pub errors_count: usize,
    /// retries_count
    pub retries_count: usize,
}

/// Distributed KNN Imputer
pub struct DistributedKNNImputer<S = Untrained> {
    state: S,
    n_neighbors: usize,
    weights: String,
    missing_values: f64,
    config: DistributedConfig,
    workers: Vec<DistributedWorker>,
    coordinator: Option<ImputationCoordinator>,
}

/// Trained state for distributed KNN imputer
pub struct DistributedKNNImputerTrained {
    reference_data: Arc<RwLock<Array2<f64>>>,
    n_features_in_: usize,
    config: DistributedConfig,
    workers: Vec<Arc<Mutex<DistributedWorker>>>,
    coordinator: ImputationCoordinator,
}

/// Coordinator for managing distributed imputation
#[derive(Debug)]
pub struct ImputationCoordinator {
    /// config
    pub config: DistributedConfig,
    /// workers
    pub workers: HashMap<usize, WorkerHandle>,
    /// data_partitioner
    pub data_partitioner: DataPartitioner,
    /// result_aggregator
    pub result_aggregator: ResultAggregator,
    /// fault_handler
    pub fault_handler: FaultHandler,
}

/// Handle for a worker process/thread
#[derive(Debug)]
pub struct WorkerHandle {
    /// id
    pub id: usize,
    /// thread_handle
    pub thread_handle: Option<thread::JoinHandle<Result<WorkerResult, ImputationError>>>,
    /// status
    pub status: WorkerStatus,
    /// last_heartbeat
    pub last_heartbeat: Instant,
}

/// Status of a worker
#[derive(Debug, Clone, PartialEq)]
pub enum WorkerStatus {
    /// Idle
    Idle,
    /// Processing
    Processing,
    /// Completed
    Completed,
    /// Failed
    Failed,
    /// Timeout
    Timeout,
}

/// Result from a worker
#[derive(Debug, Clone)]
pub struct WorkerResult {
    /// worker_id
    pub worker_id: usize,
    /// partition_id
    pub partition_id: usize,
    /// imputed_data
    pub imputed_data: Array2<f64>,
    /// statistics
    pub statistics: WorkerStatistics,
    /// metadata
    pub metadata: ImputationMetadata,
}

/// Data partitioning strategies
#[derive(Debug)]
pub struct DataPartitioner {
    strategy: PartitioningStrategy,
}

/// Partitioning strategies
#[derive(Debug, Clone)]
pub enum PartitioningStrategy {
    /// Horizontal partitioning (row-wise)
    Horizontal,
    /// Vertical partitioning (column-wise)
    Vertical,
    /// Random partitioning
    Random,
    /// Stratified partitioning based on missing patterns
    Stratified,
    /// Hash-based partitioning
    Hash,
}

/// Result aggregation strategies
#[derive(Debug)]
pub struct ResultAggregator {
    strategy: AggregationStrategy,
}

/// Aggregation strategies for combining results
#[derive(Debug, Clone)]
pub enum AggregationStrategy {
    /// Simple concatenation
    Concatenate,
    /// Weighted averaging
    WeightedAverage,
    /// Consensus-based aggregation
    Consensus,
    /// Model averaging
    ModelAveraging,
}

/// Fault handling for distributed processing
#[derive(Debug)]
pub struct FaultHandler {
    /// max_retries
    pub max_retries: usize,
    /// retry_delay
    pub retry_delay: Duration,
    /// checkpointing_enabled
    pub checkpointing_enabled: bool,
    /// checkpoint_interval
    pub checkpoint_interval: Duration,
}

impl DistributedKNNImputer<Untrained> {
    /// Create a new distributed KNN imputer
    pub fn new() -> Self {
        Self {
            state: Untrained,
            n_neighbors: 5,
            weights: "uniform".to_string(),
            missing_values: f64::NAN,
            config: DistributedConfig::default(),
            workers: Vec::new(),
            coordinator: None,
        }
    }

    /// Set the number of neighbors
    pub fn n_neighbors(mut self, n_neighbors: usize) -> Self {
        self.n_neighbors = n_neighbors;
        self
    }

    /// Set the weight function
    pub fn weights(mut self, weights: String) -> Self {
        self.weights = weights;
        self
    }

    /// Set the distributed configuration
    pub fn distributed_config(mut self, config: DistributedConfig) -> Self {
        self.config = config;
        self
    }

    /// Set the number of workers
    pub fn num_workers(mut self, num_workers: usize) -> Self {
        self.config.num_workers = num_workers;
        self
    }

    /// Set the chunk size
    pub fn chunk_size(mut self, chunk_size: usize) -> Self {
        self.config.chunk_size = chunk_size;
        self
    }

    /// Set the communication strategy
    pub fn communication_strategy(mut self, strategy: CommunicationStrategy) -> Self {
        self.config.communication_strategy = strategy;
        self
    }

    /// Enable fault tolerance
    pub fn fault_tolerance(mut self, enabled: bool) -> Self {
        self.config.fault_tolerance = enabled;
        self
    }

    fn is_missing(&self, value: f64) -> bool {
        if self.missing_values.is_nan() {
            value.is_nan()
        } else {
            (value - self.missing_values).abs() < f64::EPSILON
        }
    }
}

impl Default for DistributedKNNImputer<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for DistributedKNNImputer<Untrained> {
    type Config = DistributedConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Fit<ArrayView2<'_, Float>, ()> for DistributedKNNImputer<Untrained> {
    type Fitted = DistributedKNNImputer<DistributedKNNImputerTrained>;

    #[allow(non_snake_case)]
    fn fit(self, X: &ArrayView2<'_, Float>, _y: &()) -> SklResult<Self::Fitted> {
        let X = X.mapv(|x| x);
        let (n_samples, n_features) = X.dim();

        if n_samples < self.config.num_workers {
            return Err(SklearsError::InvalidInput(
                "Dataset too small for distributed processing. Use regular KNN imputer."
                    .to_string(),
            ));
        }

        // Create data partitioner
        let data_partitioner = DataPartitioner {
            strategy: PartitioningStrategy::Horizontal,
        };

        // Create result aggregator
        let result_aggregator = ResultAggregator {
            strategy: AggregationStrategy::Concatenate,
        };

        // Create fault handler
        let fault_handler = FaultHandler {
            max_retries: self.config.max_retries,
            retry_delay: Duration::from_secs(1),
            checkpointing_enabled: false,
            checkpoint_interval: Duration::from_secs(60),
        };

        // Create coordinator
        let coordinator = ImputationCoordinator {
            config: self.config.clone(),
            workers: HashMap::new(),
            data_partitioner,
            result_aggregator,
            fault_handler,
        };

        // Initialize workers
        let mut workers = Vec::new();
        for worker_id in 0..self.config.num_workers {
            let worker = DistributedWorker {
                id: worker_id,
                config: self.config.clone(),
                partitions: Vec::new(),
                local_imputer: Box::new(SimpleImputer::default()),
                statistics: WorkerStatistics::default(),
            };
            workers.push(Arc::new(Mutex::new(worker)));
        }

        Ok(DistributedKNNImputer {
            state: DistributedKNNImputerTrained {
                reference_data: Arc::new(RwLock::new(X.clone())),
                n_features_in_: n_features,
                config: self.config,
                workers,
                coordinator,
            },
            n_neighbors: self.n_neighbors,
            weights: self.weights,
            missing_values: self.missing_values,
            config: Default::default(),
            workers: Vec::new(),
            coordinator: None,
        })
    }
}

impl Transform<ArrayView2<'_, Float>, Array2<Float>>
    for DistributedKNNImputer<DistributedKNNImputerTrained>
{
    #[allow(non_snake_case)]
    fn transform(&self, X: &ArrayView2<'_, Float>) -> SklResult<Array2<Float>> {
        let X = X.mapv(|x| x);
        let (_n_samples, n_features) = X.dim();

        if n_features != self.state.n_features_in_ {
            return Err(SklearsError::InvalidInput(format!(
                "Number of features {} does not match training features {}",
                n_features, self.state.n_features_in_
            )));
        }

        // Partition data across workers
        let partitions = self.partition_data(&X)?;

        // Process partitions in parallel using workers
        let results = self.process_partitions_distributed(partitions)?;

        // Aggregate results
        let X_imputed = self.aggregate_results(results)?;

        Ok(X_imputed.mapv(|x| x as Float))
    }
}

impl DistributedKNNImputer<DistributedKNNImputerTrained> {
    /// Partition data for distributed processing
    fn partition_data(&self, X: &Array2<f64>) -> Result<Vec<DataPartition>, ImputationError> {
        let (n_samples, _n_features) = X.dim();
        let chunk_size = self
            .state
            .config
            .chunk_size
            .min(n_samples / self.state.config.num_workers);
        let mut partitions = Vec::new();

        for (partition_id, chunk) in X.axis_chunks_iter(Axis(0), chunk_size).enumerate() {
            let start_row = partition_id * chunk_size;
            let end_row = (start_row + chunk.nrows()).min(n_samples);

            // Create missing value mask
            let mut missing_mask = Array2::<bool>::from_elem(chunk.dim(), false);
            for ((i, j), &value) in chunk.indexed_iter() {
                missing_mask[[i, j]] = self.is_missing(value);
            }

            // Calculate missing ratio
            let total_elements = chunk.len();
            let missing_count = missing_mask.iter().filter(|&&x| x).count();
            let missing_ratio = missing_count as f64 / total_elements as f64;

            let metadata = PartitionMetadata {
                partition_id,
                worker_id: partition_id % self.state.config.num_workers,
                num_samples: chunk.nrows(),
                num_features: chunk.ncols(),
                missing_ratio,
                processing_time: Duration::default(),
                memory_usage: chunk.len() * std::mem::size_of::<f64>(),
            };

            partitions.push(DataPartition {
                id: partition_id,
                start_row,
                end_row,
                data: chunk.to_owned(),
                missing_mask,
                metadata,
            });
        }

        Ok(partitions)
    }

    /// Process partitions using distributed workers
    fn process_partitions_distributed(
        &self,
        partitions: Vec<DataPartition>,
    ) -> Result<Vec<WorkerResult>, ImputationError> {
        let reference_data = self.state.reference_data.clone();
        let n_neighbors = self.n_neighbors;
        let weights = self.weights.clone();
        let _missing_values = self.missing_values;

        // Process partitions in parallel
        let results: Result<Vec<_>, _> = partitions
            .into_par_iter()
            .map(|partition| -> Result<WorkerResult, ImputationError> {
                let start_time = Instant::now();
                let worker_id = partition.metadata.worker_id;

                // Access reference data
                let ref_data = reference_data.read().map_err(|_| {
                    ImputationError::ProcessingError("Failed to access reference data".to_string())
                })?;

                // Perform KNN imputation on this partition
                let mut imputed_data = partition.data.clone();

                for i in 0..imputed_data.nrows() {
                    for j in 0..imputed_data.ncols() {
                        if partition.missing_mask[[i, j]] {
                            // Find k nearest neighbors from reference data
                            let query_row = imputed_data.row(i);
                            let query_row_2d = query_row.insert_axis(Axis(0));
                            let neighbors =
                                self.find_knn_neighbors(&ref_data, query_row_2d, n_neighbors, j)?;

                            // Compute weighted average
                            let imputed_value =
                                self.compute_weighted_average(&neighbors, &weights)?;
                            imputed_data[[i, j]] = imputed_value;
                        }
                    }
                }

                let processing_time = start_time.elapsed();

                let statistics = WorkerStatistics {
                    samples_processed: partition.metadata.num_samples,
                    features_imputed: partition.missing_mask.iter().filter(|&&x| x).count(),
                    processing_time,
                    memory_peak: partition.metadata.memory_usage,
                    errors_count: 0,
                    retries_count: 0,
                };

                let metadata = ImputationMetadata {
                    method: "DistributedKNN".to_string(),
                    parameters: {
                        let mut params = std::collections::HashMap::new();
                        params.insert("n_neighbors".to_string(), n_neighbors.to_string());
                        params.insert("weights".to_string(), weights.clone());
                        params
                    },
                    processing_time_ms: Some(processing_time.as_millis() as u64),
                    n_imputed: partition.missing_mask.iter().filter(|&&x| x).count(),
                    convergence_info: None,
                    quality_metrics: None,
                };

                Ok(WorkerResult {
                    worker_id,
                    partition_id: partition.id,
                    imputed_data,
                    statistics,
                    metadata,
                })
            })
            .collect();

        results.map_err(|_| {
            ImputationError::ProcessingError("Distributed processing failed".to_string())
        })
    }

    /// Find k nearest neighbors for imputation
    fn find_knn_neighbors(
        &self,
        reference_data: &Array2<f64>,
        query_row: ArrayView2<f64>,
        k: usize,
        target_feature: usize,
    ) -> Result<Vec<(f64, f64)>, ImputationError> {
        let mut neighbors = Vec::new();

        for ref_row_idx in 0..reference_data.nrows() {
            let ref_row = reference_data.row(ref_row_idx);

            // Skip if reference row has missing value for target feature
            if self.is_missing(ref_row[target_feature]) {
                continue;
            }

            // Calculate distance (ignoring missing values)
            let distance = self.calculate_nan_euclidean_distance(query_row.row(0), ref_row);

            if distance.is_finite() {
                neighbors.push((distance, ref_row[target_feature]));
            }
        }

        // Sort by distance and take k nearest
        neighbors.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        neighbors.truncate(k);

        Ok(neighbors)
    }

    /// Calculate Euclidean distance ignoring NaN values
    fn calculate_nan_euclidean_distance(
        &self,
        row1: ArrayView1<f64>,
        row2: ArrayView1<f64>,
    ) -> f64 {
        let mut sum_sq = 0.0;
        let mut valid_count = 0;

        for (&x1, &x2) in row1.iter().zip(row2.iter()) {
            if !self.is_missing(x1) && !self.is_missing(x2) {
                sum_sq += (x1 - x2).powi(2);
                valid_count += 1;
            }
        }

        if valid_count > 0 {
            (sum_sq / valid_count as f64).sqrt()
        } else {
            f64::INFINITY
        }
    }

    /// Compute weighted average of neighbor values
    fn compute_weighted_average(
        &self,
        neighbors: &[(f64, f64)],
        weights_type: &str,
    ) -> Result<f64, ImputationError> {
        if neighbors.is_empty() {
            return Err(ImputationError::ProcessingError(
                "No valid neighbors found".to_string(),
            ));
        }

        match weights_type {
            "uniform" => {
                let sum: f64 = neighbors.iter().map(|(_, value)| value).sum();
                Ok(sum / neighbors.len() as f64)
            }
            "distance" => {
                let mut weighted_sum = 0.0;
                let mut weight_sum = 0.0;

                for &(distance, value) in neighbors {
                    let weight = if distance > 0.0 { 1.0 / distance } else { 1e6 };
                    weighted_sum += weight * value;
                    weight_sum += weight;
                }

                if weight_sum > 0.0 {
                    Ok(weighted_sum / weight_sum)
                } else {
                    Ok(neighbors[0].1) // Fallback to first neighbor
                }
            }
            _ => Err(ImputationError::InvalidConfiguration(format!(
                "Unknown weights type: {}",
                weights_type
            ))),
        }
    }

    /// Aggregate results from distributed workers
    fn aggregate_results(
        &self,
        results: Vec<WorkerResult>,
    ) -> Result<Array2<f64>, ImputationError> {
        if results.is_empty() {
            return Err(ImputationError::ProcessingError(
                "No results to aggregate".to_string(),
            ));
        }

        // Sort results by partition ID to maintain order
        let mut sorted_results = results;
        sorted_results.sort_by_key(|r| r.partition_id);

        // Concatenate imputed data
        let first_result = &sorted_results[0];
        let n_features = first_result.imputed_data.ncols();
        let total_rows: usize = sorted_results.iter().map(|r| r.imputed_data.nrows()).sum();

        let mut aggregated_data = Array2::<f64>::zeros((total_rows, n_features));
        let mut current_row = 0;

        for result in sorted_results {
            let chunk_rows = result.imputed_data.nrows();
            aggregated_data
                .slice_mut(s![current_row..current_row + chunk_rows, ..])
                .assign(&result.imputed_data);
            current_row += chunk_rows;
        }

        Ok(aggregated_data)
    }

    fn is_missing(&self, value: f64) -> bool {
        if self.missing_values.is_nan() {
            value.is_nan()
        } else {
            (value - self.missing_values).abs() < f64::EPSILON
        }
    }
}

/// Distributed Simple Imputer for basic imputation strategies
#[derive(Debug)]
pub struct DistributedSimpleImputer<S = Untrained> {
    state: S,
    strategy: String,
    missing_values: f64,
    config: DistributedConfig,
}

/// Trained state for distributed simple imputer
#[derive(Debug)]
pub struct DistributedSimpleImputerTrained {
    statistics_: Array1<f64>,
    n_features_in_: usize,
    config: DistributedConfig,
}

impl DistributedSimpleImputer<Untrained> {
    pub fn new() -> Self {
        Self {
            state: Untrained,
            strategy: "mean".to_string(),
            missing_values: f64::NAN,
            config: DistributedConfig::default(),
        }
    }

    pub fn strategy(mut self, strategy: String) -> Self {
        self.strategy = strategy;
        self
    }

    pub fn distributed_config(mut self, config: DistributedConfig) -> Self {
        self.config = config;
        self
    }

    fn is_missing(&self, value: f64) -> bool {
        if self.missing_values.is_nan() {
            value.is_nan()
        } else {
            (value - self.missing_values).abs() < f64::EPSILON
        }
    }
}

impl Default for DistributedSimpleImputer<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for DistributedSimpleImputer<Untrained> {
    type Config = DistributedConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Fit<ArrayView2<'_, Float>, ()> for DistributedSimpleImputer<Untrained> {
    type Fitted = DistributedSimpleImputer<DistributedSimpleImputerTrained>;

    #[allow(non_snake_case)]
    fn fit(self, X: &ArrayView2<'_, Float>, _y: &()) -> SklResult<Self::Fitted> {
        let X = X.mapv(|x| x);
        let (_, n_features) = X.dim();

        // Compute statistics in parallel across features
        let statistics: Vec<f64> = (0..n_features)
            .into_par_iter()
            .map(|j| {
                let column = X.column(j);
                let valid_values: Vec<f64> = column
                    .iter()
                    .filter(|&&x| !self.is_missing(x))
                    .cloned()
                    .collect();

                if valid_values.is_empty() {
                    0.0
                } else {
                    match self.strategy.as_str() {
                        "mean" => valid_values.iter().sum::<f64>() / valid_values.len() as f64,
                        "median" => {
                            let mut sorted_values = valid_values.clone();
                            sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
                            let mid = sorted_values.len() / 2;
                            if sorted_values.len() % 2 == 0 {
                                (sorted_values[mid - 1] + sorted_values[mid]) / 2.0
                            } else {
                                sorted_values[mid]
                            }
                        }
                        "most_frequent" => {
                            let mut frequency_map = HashMap::new();
                            for &value in &valid_values {
                                *frequency_map.entry(value as i64).or_insert(0) += 1;
                            }
                            frequency_map
                                .into_iter()
                                .max_by_key(|(_, count)| *count)
                                .map(|(value, _)| value as f64)
                                .unwrap_or(0.0)
                        }
                        _ => valid_values.iter().sum::<f64>() / valid_values.len() as f64,
                    }
                }
            })
            .collect();

        Ok(DistributedSimpleImputer {
            state: DistributedSimpleImputerTrained {
                statistics_: Array1::from_vec(statistics),
                n_features_in_: n_features,
                config: self.config,
            },
            strategy: self.strategy,
            missing_values: self.missing_values,
            config: Default::default(),
        })
    }
}

impl Transform<ArrayView2<'_, Float>, Array2<Float>>
    for DistributedSimpleImputer<DistributedSimpleImputerTrained>
{
    #[allow(non_snake_case)]
    fn transform(&self, X: &ArrayView2<'_, Float>) -> SklResult<Array2<Float>> {
        let X = X.mapv(|x| x);
        let (n_samples, n_features) = X.dim();

        if n_features != self.state.n_features_in_ {
            return Err(SklearsError::InvalidInput(format!(
                "Number of features {} does not match training features {}",
                n_features, self.state.n_features_in_
            )));
        }

        // Parallel imputation across rows
        let imputed_rows: Vec<Array1<f64>> = (0..n_samples)
            .into_par_iter()
            .map(|i| {
                let mut row = X.row(i).to_owned();
                for j in 0..n_features {
                    if self.is_missing(row[j]) {
                        row[j] = self.state.statistics_[j];
                    }
                }
                row
            })
            .collect();

        // Reconstruct the array
        let mut X_imputed = Array2::zeros((n_samples, n_features));
        for (i, row) in imputed_rows.into_iter().enumerate() {
            X_imputed.row_mut(i).assign(&row);
        }

        Ok(X_imputed.mapv(|x| x as Float))
    }
}

impl DistributedSimpleImputer<DistributedSimpleImputerTrained> {
    fn is_missing(&self, value: f64) -> bool {
        if self.missing_values.is_nan() {
            value.is_nan()
        } else {
            (value - self.missing_values).abs() < f64::EPSILON
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    #[test]
    #[allow(non_snake_case)]
    fn test_distributed_simple_imputer() {
        let X = array![[1.0, 2.0, 3.0], [4.0, f64::NAN, 6.0], [7.0, 8.0, 9.0]];

        let imputer = DistributedSimpleImputer::new()
            .strategy("mean".to_string())
            .distributed_config(DistributedConfig {
                num_workers: 2,
                ..Default::default()
            });

        let fitted = imputer.fit(&X.view(), &()).unwrap();
        let X_imputed = fitted.transform(&X.view()).unwrap();

        // Check that NaN was replaced with mean of column (2.0 + 8.0) / 2 = 5.0
        assert_abs_diff_eq!(X_imputed[[1, 1]], 5.0, epsilon = 1e-10);
        assert_abs_diff_eq!(X_imputed[[0, 0]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(X_imputed[[2, 2]], 9.0, epsilon = 1e-10);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_distributed_knn_imputer() {
        let X = array![
            [1.0, 2.0, 3.0],
            [4.0, f64::NAN, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0]
        ];

        let imputer = DistributedKNNImputer::new()
            .n_neighbors(2)
            .weights("uniform".to_string())
            .distributed_config(DistributedConfig {
                num_workers: 2,
                chunk_size: 2,
                ..Default::default()
            });

        let fitted = imputer.fit(&X.view(), &()).unwrap();
        let X_imputed = fitted.transform(&X.view()).unwrap();

        // Verify that missing value was imputed
        assert!(!X_imputed[[1, 1]].is_nan());
        assert_abs_diff_eq!(X_imputed[[0, 0]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(X_imputed[[2, 2]], 9.0, epsilon = 1e-10);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_data_partitioning() {
        let X = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];

        let imputer = DistributedKNNImputer::new().distributed_config(DistributedConfig {
            num_workers: 2,
            chunk_size: 2,
            ..Default::default()
        });

        let fitted = imputer.fit(&X.view(), &()).unwrap();
        let partitions = fitted.partition_data(&X.mapv(|x| x)).unwrap();

        assert_eq!(partitions.len(), 2);
        assert_eq!(partitions[0].data.nrows(), 2);
        assert_eq!(partitions[1].data.nrows(), 2);
        assert_eq!(partitions[0].start_row, 0);
        assert_eq!(partitions[0].end_row, 2);
        assert_eq!(partitions[1].start_row, 2);
        assert_eq!(partitions[1].end_row, 4);
    }
}
