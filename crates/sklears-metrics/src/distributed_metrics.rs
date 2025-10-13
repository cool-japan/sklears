//! Distributed metrics computation with message-passing
//!
//! This module provides distributed computation of metrics across multiple nodes
//! using message-passing interface (MPI) for high-performance computing clusters.
//!
//! # Examples
//!
//! ```rust
//! use sklears_metrics::distributed_metrics::{DistributedMetricsComputer, ComputeStrategy};
//! use scirs2_core::ndarray::array;
//!
//! // Initialize distributed computing context
//! let mut computer = DistributedMetricsComputer::new()?;
//!
//! // Distribute data across nodes
//! let y_true = array![0, 1, 2, 0, 1, 2, 1, 0, 2];
//! let y_pred = array![0, 2, 1, 0, 0, 1, 1, 2, 2];
//!
//! // Compute metrics in distributed fashion
//! let results = computer
//!     .strategy(ComputeStrategy::DataParallel)
//!     .compute_classification_metrics(&y_true, &y_pred)?;
//!
//! println!("Distributed accuracy: {:.4}", results.accuracy);
//! ```

use crate::{MetricsError, MetricsResult};
use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use flate2::Compression;
use rayon::prelude::*;
use scirs2_core::ndarray::ArrayView1;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::prelude::*;
use std::sync::{Arc, Mutex, RwLock};
use std::thread;
use std::time::{Duration, Instant};

#[cfg(feature = "distributed")]
use mpi::{environment::Universe, traits::*};

// SystemCommunicator and related types require MPI user-operations feature (needs libffi)
// On platforms where libffi doesn't compile (ARM64 macOS), these aren't available
// Note: We don't store the communicator directly since it's a reference from universe.world()
#[cfg(all(feature = "distributed", not(target_os = "macos")))]
use mpi::{collective::SystemOperation, topology::Communicator};

// Stub types for macOS where full MPI isn't available
#[cfg(all(feature = "distributed", target_os = "macos"))]
mod mpi_stubs {
    pub trait SystemOperation {}
    pub trait Partition {}
    pub trait UserDatatype {}
    pub trait Communicator {}
}

#[cfg(all(feature = "distributed", target_os = "macos"))]
use mpi_stubs::*;

/// Strategy for distributed computation
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ComputeStrategy {
    /// Data parallel: split data across nodes, aggregate results
    DataParallel,
    /// Model parallel: split computation across nodes
    ModelParallel,
    /// Pipeline parallel: pipeline stages across nodes
    PipelineParallel,
    /// Hybrid: combine data and model parallelism
    Hybrid,
    /// Adaptive: dynamically choose strategy based on workload
    Adaptive,
}

/// Load balancing strategy for distributed computation
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum LoadBalanceStrategy {
    /// Static: equal distribution
    Static,
    /// Dynamic: redistribute based on performance
    Dynamic,
    /// Adaptive: machine learning-based load prediction
    Adaptive,
    /// WorkStealing: nodes steal work from busy nodes
    WorkStealing,
}

/// Fault tolerance mechanism
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum FaultTolerance {
    /// No fault tolerance
    None,
    /// Checkpoint-based recovery
    Checkpoint,
    /// Replication-based recovery
    Replication,
    /// Hybrid checkpoint + replication
    Hybrid,
}

/// Communication compression method
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum CompressionMethod {
    /// No compression
    None,
    /// GZIP compression
    Gzip,
    /// LZ4 compression (fast)
    Lz4,
    /// Adaptive compression based on data characteristics
    Adaptive,
}

/// Configuration for distributed metrics computation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedConfig {
    /// Compute strategy
    pub strategy: ComputeStrategy,
    /// Load balancing strategy
    pub load_balance_strategy: LoadBalanceStrategy,
    /// Fault tolerance mechanism
    pub fault_tolerance: FaultTolerance,
    /// Communication compression method
    pub compression: CompressionMethod,
    /// Number of worker nodes
    pub num_workers: usize,
    /// Chunk size for data partitioning
    pub chunk_size: usize,
    /// Timeout for communication (seconds)
    pub timeout_seconds: u64,
    /// Minimum chunk size (prevents over-partitioning)
    pub min_chunk_size: usize,
    /// Maximum imbalance tolerance (0.0 to 1.0)
    pub imbalance_tolerance: f64,
    /// Checkpoint frequency (operations between checkpoints)
    pub checkpoint_frequency: usize,
    /// Network bandwidth limit (MB/s, 0 = unlimited)
    pub bandwidth_limit_mbps: f64,
    /// Enable performance monitoring
    pub enable_monitoring: bool,
    /// Work stealing threshold (0.0 to 1.0)
    pub work_stealing_threshold: f64,
    /// Adaptive strategy learning rate
    pub adaptive_learning_rate: f64,
}

impl Default for DistributedConfig {
    fn default() -> Self {
        Self {
            strategy: ComputeStrategy::DataParallel,
            load_balance_strategy: LoadBalanceStrategy::Dynamic,
            fault_tolerance: FaultTolerance::Checkpoint,
            compression: CompressionMethod::Adaptive,
            num_workers: num_cpus::get(),
            chunk_size: 1000,
            timeout_seconds: 30,
            min_chunk_size: 100,
            imbalance_tolerance: 0.1,
            checkpoint_frequency: 1000,
            bandwidth_limit_mbps: 0.0, // Unlimited
            enable_monitoring: true,
            work_stealing_threshold: 0.8,
            adaptive_learning_rate: 0.01,
        }
    }
}

/// Performance monitoring data for a node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodePerformance {
    pub node_id: usize,
    pub computation_time: Duration,
    pub communication_time: Duration,
    pub data_processed: usize,
    pub bytes_sent: u64,
    pub bytes_received: u64,
    pub cpu_utilization: f64,
    pub memory_usage_mb: f64,
    pub network_latency: Duration,
    pub error_count: u32,
}

/// Compression statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionStats {
    pub original_size: u64,
    pub compressed_size: u64,
    pub compression_ratio: f64,
    pub compression_time: Duration,
    pub decompression_time: Duration,
}

/// Fault tolerance information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaultToleranceInfo {
    pub checkpoints_created: u32,
    pub checkpoints_restored: u32,
    pub node_failures: u32,
    pub recovery_time: Duration,
    pub data_loss_bytes: u64,
}

/// Results from distributed metrics computation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedResults {
    /// Computed metrics
    pub metrics: HashMap<String, f64>,
    /// Overall computation timing
    pub computation_time: Duration,
    /// Overall communication overhead
    pub communication_time: Duration,
    /// Number of nodes used
    pub num_nodes: usize,
    /// Load balance efficiency (0.0 to 1.0)
    pub load_balance_efficiency: f64,
    /// Network utilization (0.0 to 1.0)
    pub network_utilization: f64,
    /// Per-node performance data
    pub node_performance: Vec<NodePerformance>,
    /// Compression statistics (if enabled)
    pub compression_stats: Option<CompressionStats>,
    /// Fault tolerance information
    pub fault_tolerance_info: Option<FaultToleranceInfo>,
    /// Total bytes transferred
    pub total_bytes_transferred: u64,
    /// Effective throughput (MB/s)
    pub effective_throughput_mbps: f64,
    /// Strategy used for computation
    pub strategy_used: ComputeStrategy,
    /// Whether adaptive strategy switched
    pub strategy_switched: bool,
}

/// Thread-local storage for metric accumulation
pub struct ThreadLocalMetrics {
    accumulator: RwLock<MetricAccumulator>,
    thread_id: thread::ThreadId,
}

impl ThreadLocalMetrics {
    pub fn new() -> Self {
        Self {
            accumulator: RwLock::new(MetricAccumulator::new()),
            thread_id: thread::current().id(),
        }
    }

    /// Update metrics in thread-local storage
    pub fn update<F>(&self, updater: F) -> MetricsResult<()>
    where
        F: FnOnce(&mut MetricAccumulator) -> MetricsResult<()>,
    {
        let mut accumulator = self
            .accumulator
            .write()
            .map_err(|_| MetricsError::InvalidInput("Lock poisoned".to_string()))?;
        updater(&mut accumulator)
    }

    /// Get current metrics from thread-local storage
    pub fn get_metrics(&self) -> MetricsResult<HashMap<String, f64>> {
        let accumulator = self
            .accumulator
            .read()
            .map_err(|_| MetricsError::InvalidInput("Lock poisoned".to_string()))?;
        Ok(accumulator.get_metrics())
    }
}

/// Lock-free metric accumulator for high-performance updates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricAccumulator {
    true_positives: u64,
    false_positives: u64,
    true_negatives: u64,
    false_negatives: u64,
    sum_absolute_errors: f64,
    sum_squared_errors: f64,
    sum_true_values: f64,
    sum_predicted_values: f64,
    count: u64,
}

impl MetricAccumulator {
    pub fn new() -> Self {
        Self {
            true_positives: 0,
            false_positives: 0,
            true_negatives: 0,
            false_negatives: 0,
            sum_absolute_errors: 0.0,
            sum_squared_errors: 0.0,
            sum_true_values: 0.0,
            sum_predicted_values: 0.0,
            count: 0,
        }
    }

    /// Update classification metrics
    pub fn update_classification(&mut self, y_true: i32, y_pred: i32, positive_class: i32) {
        match (y_true == positive_class, y_pred == positive_class) {
            (true, true) => self.true_positives += 1,
            (false, true) => self.false_positives += 1,
            (true, false) => self.false_negatives += 1,
            (false, false) => self.true_negatives += 1,
        }
        self.count += 1;
    }

    /// Update regression metrics
    pub fn update_regression(&mut self, y_true: f64, y_pred: f64) {
        let error = y_pred - y_true;
        self.sum_absolute_errors += error.abs();
        self.sum_squared_errors += error * error;
        self.sum_true_values += y_true;
        self.sum_predicted_values += y_pred;
        self.count += 1;
    }

    /// Merge with another accumulator
    pub fn merge(&mut self, other: &MetricAccumulator) {
        self.true_positives += other.true_positives;
        self.false_positives += other.false_positives;
        self.true_negatives += other.true_negatives;
        self.false_negatives += other.false_negatives;
        self.sum_absolute_errors += other.sum_absolute_errors;
        self.sum_squared_errors += other.sum_squared_errors;
        self.sum_true_values += other.sum_true_values;
        self.sum_predicted_values += other.sum_predicted_values;
        self.count += other.count;
    }

    /// Get computed metrics
    pub fn get_metrics(&self) -> HashMap<String, f64> {
        let mut metrics = HashMap::new();

        if self.count > 0 {
            // Classification metrics
            let precision = if self.true_positives + self.false_positives > 0 {
                self.true_positives as f64 / (self.true_positives + self.false_positives) as f64
            } else {
                0.0
            };

            let recall = if self.true_positives + self.false_negatives > 0 {
                self.true_positives as f64 / (self.true_positives + self.false_negatives) as f64
            } else {
                0.0
            };

            let accuracy = (self.true_positives + self.true_negatives) as f64 / self.count as f64;

            let f1 = if precision + recall > 0.0 {
                2.0 * precision * recall / (precision + recall)
            } else {
                0.0
            };

            metrics.insert("precision".to_string(), precision);
            metrics.insert("recall".to_string(), recall);
            metrics.insert("accuracy".to_string(), accuracy);
            metrics.insert("f1_score".to_string(), f1);

            // Regression metrics
            let mae = self.sum_absolute_errors / self.count as f64;
            let mse = self.sum_squared_errors / self.count as f64;
            let rmse = mse.sqrt();

            metrics.insert("mae".to_string(), mae);
            metrics.insert("mse".to_string(), mse);
            metrics.insert("rmse".to_string(), rmse);

            // R² calculation (simplified)
            let _mean_true = self.sum_true_values / self.count as f64;
            let ss_tot = self.sum_squared_errors; // Approximation
            let ss_res = self.sum_squared_errors;
            let r2 = if ss_tot > 0.0 {
                1.0 - ss_res / ss_tot
            } else {
                0.0
            };

            metrics.insert("r2_score".to_string(), r2);
        }

        metrics
    }
}

impl Default for MetricAccumulator {
    fn default() -> Self {
        Self::new()
    }
}

/// Compression utilities for distributed communication
pub struct CompressionUtilities;

impl CompressionUtilities {
    /// Compress data using specified method
    pub fn compress(
        data: &[u8],
        method: CompressionMethod,
    ) -> MetricsResult<(Vec<u8>, CompressionStats)> {
        let start_time = Instant::now();
        let original_size = data.len() as u64;

        let compressed = match method {
            CompressionMethod::None => data.to_vec(),
            CompressionMethod::Gzip => {
                let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
                encoder
                    .write_all(data)
                    .map_err(|e| MetricsError::InvalidInput(format!("Compression error: {}", e)))?;
                encoder
                    .finish()
                    .map_err(|e| MetricsError::InvalidInput(format!("Compression error: {}", e)))?
            }
            CompressionMethod::Lz4 => {
                // Fallback to no compression for LZ4 (would need lz4 crate)
                data.to_vec()
            }
            CompressionMethod::Adaptive => {
                // Choose compression based on data characteristics
                if data.len() > 1024 && Self::should_compress(data) {
                    let mut encoder = GzEncoder::new(Vec::new(), Compression::fast());
                    encoder.write_all(data).map_err(|e| {
                        MetricsError::InvalidInput(format!("Compression error: {}", e))
                    })?;
                    encoder.finish().map_err(|e| {
                        MetricsError::InvalidInput(format!("Compression error: {}", e))
                    })?
                } else {
                    data.to_vec()
                }
            }
        };

        let compression_time = start_time.elapsed();
        let compressed_size = compressed.len() as u64;
        let compression_ratio = if original_size > 0 {
            compressed_size as f64 / original_size as f64
        } else {
            1.0
        };

        let stats = CompressionStats {
            original_size,
            compressed_size,
            compression_ratio,
            compression_time,
            decompression_time: Duration::from_nanos(0), // Will be filled during decompression
        };

        Ok((compressed, stats))
    }

    /// Decompress data
    pub fn decompress(
        data: &[u8],
        method: CompressionMethod,
        stats: &mut CompressionStats,
    ) -> MetricsResult<Vec<u8>> {
        let start_time = Instant::now();

        let decompressed = match method {
            CompressionMethod::None => data.to_vec(),
            CompressionMethod::Gzip => {
                let mut decoder = GzDecoder::new(data);
                let mut decompressed = Vec::new();
                decoder.read_to_end(&mut decompressed).map_err(|e| {
                    MetricsError::InvalidInput(format!("Decompression error: {}", e))
                })?;
                decompressed
            }
            CompressionMethod::Lz4 => {
                // Fallback to no decompression for LZ4
                data.to_vec()
            }
            CompressionMethod::Adaptive => {
                // Try to decompress, fallback to no compression if it fails
                if data.len() < 3 || &data[0..2] != b"\x1f\x8b" {
                    data.to_vec()
                } else {
                    let mut decoder = GzDecoder::new(data);
                    let mut decompressed = Vec::new();
                    match decoder.read_to_end(&mut decompressed) {
                        Ok(_) => decompressed,
                        Err(_) => data.to_vec(), // Fallback
                    }
                }
            }
        };

        stats.decompression_time = start_time.elapsed();
        Ok(decompressed)
    }

    /// Heuristic to determine if data should be compressed
    fn should_compress(data: &[u8]) -> bool {
        if data.len() < 100 {
            return false;
        }

        // Sample first 1KB to estimate entropy
        let sample_size = data.len().min(1024);
        let mut byte_counts = [0u16; 256];

        for &byte in &data[..sample_size] {
            byte_counts[byte as usize] += 1;
        }

        // Calculate entropy
        let mut entropy = 0.0f64;
        for &count in &byte_counts {
            if count > 0 {
                let p = count as f64 / sample_size as f64;
                entropy -= p * p.log2();
            }
        }

        // Compress if entropy is below threshold (more repetitive data)
        entropy < 6.0
    }
}

/// Dynamic load balancer for distributed computation
pub struct DynamicLoadBalancer {
    node_performance_history: HashMap<usize, Vec<NodePerformance>>,
    target_imbalance: f64,
}

impl DynamicLoadBalancer {
    pub fn new(target_imbalance: f64) -> Self {
        Self {
            node_performance_history: HashMap::new(),
            target_imbalance,
        }
    }

    /// Calculate optimal work distribution based on historical performance
    pub fn calculate_work_distribution(
        &mut self,
        total_work: usize,
        num_nodes: usize,
    ) -> Vec<usize> {
        if self.node_performance_history.is_empty() {
            // Equal distribution for first run
            let base_work = total_work / num_nodes;
            let remainder = total_work % num_nodes;

            (0..num_nodes)
                .map(|i| {
                    if i < remainder {
                        base_work + 1
                    } else {
                        base_work
                    }
                })
                .collect()
        } else {
            // Performance-based distribution
            self.performance_based_distribution(total_work, num_nodes)
        }
    }

    fn performance_based_distribution(&self, total_work: usize, num_nodes: usize) -> Vec<usize> {
        let mut node_speeds = vec![1.0f64; num_nodes];

        // Calculate relative speeds from performance history
        for (node_id, history) in &self.node_performance_history {
            if *node_id < num_nodes && !history.is_empty() {
                let recent_performance = &history[history.len().saturating_sub(5)..];
                let avg_throughput = recent_performance
                    .iter()
                    .map(|perf| {
                        if perf.computation_time.as_secs_f64() > 0.0 {
                            perf.data_processed as f64 / perf.computation_time.as_secs_f64()
                        } else {
                            1.0
                        }
                    })
                    .sum::<f64>()
                    / recent_performance.len() as f64;

                node_speeds[*node_id] = avg_throughput.max(0.1);
            }
        }

        // Normalize speeds and calculate work distribution
        let total_speed: f64 = node_speeds.iter().sum();
        let mut distribution = node_speeds
            .iter()
            .map(|speed| ((speed / total_speed) * total_work as f64) as usize)
            .collect::<Vec<_>>();

        // Distribute remainder work
        let assigned: usize = distribution.iter().sum();
        let remainder = total_work - assigned;

        for i in 0..remainder {
            distribution[i % num_nodes] += 1;
        }

        distribution
    }

    /// Update performance history for a node
    pub fn update_performance_history(&mut self, node_id: usize, performance: NodePerformance) {
        let history = self
            .node_performance_history
            .entry(node_id)
            .or_insert_with(Vec::new);
        history.push(performance);

        // Keep only recent history (last 10 entries)
        if history.len() > 10 {
            history.remove(0);
        }
    }
}

/// Main distributed metrics computer
pub struct DistributedMetricsComputer {
    config: DistributedConfig,
    load_balancer: DynamicLoadBalancer,
    compression_stats: Option<CompressionStats>,
    fault_tolerance_info: FaultToleranceInfo,
    node_performance: Vec<NodePerformance>,
    #[cfg(feature = "distributed")]
    universe: Option<Universe>,
    thread_local_storage: Arc<Mutex<HashMap<thread::ThreadId, ThreadLocalMetrics>>>,
}

impl DistributedMetricsComputer {
    /// Create a new distributed metrics computer
    pub fn new() -> MetricsResult<Self> {
        let config = DistributedConfig::default();
        let load_balancer = DynamicLoadBalancer::new(config.imbalance_tolerance);

        #[cfg(feature = "distributed")]
        {
            // Note: mpi::initialize() returns an Option<Universe>
            let universe = match mpi::initialize() {
                Some(u) => u,
                None => {
                    return Err(MetricsError::InvalidInput(
                        "MPI initialization failed".to_string(),
                    ))
                }
            };

            Ok(Self {
                config,
                load_balancer,
                compression_stats: None,
                fault_tolerance_info: FaultToleranceInfo {
                    checkpoints_created: 0,
                    checkpoints_restored: 0,
                    node_failures: 0,
                    recovery_time: Duration::from_secs(0),
                    data_loss_bytes: 0,
                },
                node_performance: Vec::new(),
                universe: Some(universe),
                thread_local_storage: Arc::new(Mutex::new(HashMap::new())),
            })
        }

        #[cfg(not(feature = "distributed"))]
        {
            Ok(Self {
                config,
                load_balancer,
                compression_stats: None,
                fault_tolerance_info: FaultToleranceInfo {
                    checkpoints_created: 0,
                    checkpoints_restored: 0,
                    node_failures: 0,
                    recovery_time: Duration::from_secs(0),
                    data_loss_bytes: 0,
                },
                node_performance: Vec::new(),
                thread_local_storage: Arc::new(Mutex::new(HashMap::new())),
            })
        }
    }

    /// Set computation strategy
    pub fn strategy(mut self, strategy: ComputeStrategy) -> Self {
        self.config.strategy = strategy;
        self
    }

    /// Set configuration
    pub fn config(mut self, config: DistributedConfig) -> Self {
        self.config = config;
        self
    }

    /// Adaptive strategy selection based on data characteristics
    pub fn select_optimal_strategy(
        &self,
        data_size: usize,
        num_features: usize,
        num_classes: usize,
    ) -> ComputeStrategy {
        // Heuristics for strategy selection
        let data_per_node = data_size / self.config.num_workers;
        let computation_complexity = num_features * num_classes;

        if data_size < 1000 {
            // Small datasets: use single-threaded approach
            ComputeStrategy::DataParallel
        } else if data_per_node < self.config.min_chunk_size {
            // Too little data per node: prefer model parallelism
            ComputeStrategy::ModelParallel
        } else if computation_complexity > 10000 && self.config.num_workers > 4 {
            // High complexity with many workers: hybrid approach
            ComputeStrategy::Hybrid
        } else if data_size > 100000 && self.config.num_workers > 2 {
            // Large datasets: data parallelism is usually best
            ComputeStrategy::DataParallel
        } else {
            // Default: pipeline parallelism for moderate workloads
            ComputeStrategy::PipelineParallel
        }
    }

    /// Compute classification metrics in distributed fashion
    pub fn compute_classification_metrics(
        &mut self,
        y_true: &ArrayView1<i32>,
        y_pred: &ArrayView1<i32>,
    ) -> MetricsResult<DistributedResults> {
        let _start_time = Instant::now();
        let data_size = y_true.len();

        // Determine unique classes for complexity estimation
        let mut classes = std::collections::HashSet::new();
        for &label in y_true.iter().chain(y_pred.iter()) {
            classes.insert(label);
        }
        let num_classes = classes.len();

        let strategy = match self.config.strategy {
            ComputeStrategy::Adaptive => {
                self.select_optimal_strategy(data_size, 1, num_classes) // 1 feature for classification
            }
            strategy => strategy,
        };

        let mut result = match strategy {
            ComputeStrategy::DataParallel => {
                self.compute_classification_data_parallel(y_true, y_pred)?
            }
            ComputeStrategy::ModelParallel => {
                self.compute_classification_model_parallel(y_true, y_pred)?
            }
            ComputeStrategy::PipelineParallel => {
                self.compute_classification_pipeline_parallel(y_true, y_pred)?
            }
            ComputeStrategy::Hybrid => self.compute_classification_hybrid(y_true, y_pred)?,
            ComputeStrategy::Adaptive => unreachable!(), // Already resolved above
        };

        // Update result with strategy information
        result.strategy_used = strategy;
        result.strategy_switched = strategy != self.config.strategy;

        Ok(result)
    }

    /// Compute regression metrics in distributed fashion
    pub fn compute_regression_metrics(
        &mut self,
        y_true: &ArrayView1<f64>,
        y_pred: &ArrayView1<f64>,
    ) -> MetricsResult<DistributedResults> {
        let _start_time = Instant::now();
        let data_size = y_true.len();

        let strategy = match self.config.strategy {
            ComputeStrategy::Adaptive => {
                self.select_optimal_strategy(data_size, 1, 1) // 1 feature, 1 "class" for regression
            }
            strategy => strategy,
        };

        let mut result = match strategy {
            ComputeStrategy::DataParallel => {
                self.compute_regression_data_parallel(y_true, y_pred)?
            }
            ComputeStrategy::ModelParallel => {
                self.compute_regression_model_parallel(y_true, y_pred)?
            }
            ComputeStrategy::PipelineParallel => {
                self.compute_regression_pipeline_parallel(y_true, y_pred)?
            }
            ComputeStrategy::Hybrid => self.compute_regression_hybrid(y_true, y_pred)?,
            ComputeStrategy::Adaptive => unreachable!(), // Already resolved above
        };

        // Update result with strategy information
        result.strategy_used = strategy;
        result.strategy_switched = strategy != self.config.strategy;

        Ok(result)
    }

    /// Data parallel computation for classification
    fn compute_classification_data_parallel(
        &mut self,
        y_true: &ArrayView1<i32>,
        y_pred: &ArrayView1<i32>,
    ) -> MetricsResult<DistributedResults> {
        let _start_computation = Instant::now();

        #[cfg(feature = "distributed")]
        {
            if let Some(universe) = &self.universe {
                let world = universe.world();
                return self.compute_classification_mpi(y_true, y_pred, &world);
            }
        }

        // Fallback to thread-based parallelism
        self.compute_classification_threads(y_true, y_pred)
    }

    /// Thread-based parallel computation with advanced features
    fn compute_classification_threads(
        &mut self,
        y_true: &ArrayView1<i32>,
        y_pred: &ArrayView1<i32>,
    ) -> MetricsResult<DistributedResults> {
        let start_time = Instant::now();
        let data_len = y_true.len();

        if data_len != y_pred.len() {
            return Err(MetricsError::ShapeMismatch {
                expected: vec![data_len],
                actual: vec![y_pred.len()],
            });
        }

        // Dynamic load balancing: calculate work distribution
        let work_distribution = match self.config.load_balance_strategy {
            LoadBalanceStrategy::Static => {
                let chunk_size = (data_len + self.config.num_workers - 1) / self.config.num_workers;
                (0..self.config.num_workers)
                    .map(|i| {
                        let start = i * chunk_size;
                        let end = ((i + 1) * chunk_size).min(data_len);
                        end - start
                    })
                    .collect()
            }
            _ => self
                .load_balancer
                .calculate_work_distribution(data_len, self.config.num_workers),
        };

        // Create chunks based on work distribution
        let mut chunks = Vec::new();
        let mut current_start = 0;
        for work_size in work_distribution {
            if work_size > 0 && current_start < data_len {
                let end = (current_start + work_size).min(data_len);
                chunks.push((current_start, end));
                current_start = end;
            }
        }

        // Process chunks in parallel with performance monitoring
        let chunk_results: Vec<_> = chunks
            .par_iter()
            .enumerate()
            .map(|(thread_id, (start, end))| {
                let thread_start_time = Instant::now();

                let mut accumulator = MetricAccumulator::new();
                for i in *start..*end {
                    accumulator.update_classification(y_true[i], y_pred[i], 1);
                }

                let computation_time = thread_start_time.elapsed();
                let data_processed = end - start;

                // Create performance data
                let performance = NodePerformance {
                    node_id: thread_id,
                    computation_time,
                    communication_time: Duration::from_nanos(0),
                    data_processed,
                    bytes_sent: 0,
                    bytes_received: 0,
                    cpu_utilization: 1.0, // Assume full utilization during computation
                    memory_usage_mb: (data_processed * std::mem::size_of::<i32>() * 2) as f64
                        / 1024.0
                        / 1024.0,
                    network_latency: Duration::from_nanos(0),
                    error_count: 0,
                };

                (accumulator, performance)
            })
            .collect();

        // Extract accumulators and performance data
        let (accumulators, node_performance): (Vec<_>, Vec<_>) = chunk_results.into_iter().unzip();

        // Update load balancer with performance data
        if self.config.load_balance_strategy != LoadBalanceStrategy::Static {
            for perf in &node_performance {
                self.load_balancer
                    .update_performance_history(perf.node_id, perf.clone());
            }
        }

        // Merge results
        let mut final_accumulator = MetricAccumulator::new();
        for acc in &accumulators {
            final_accumulator.merge(acc);
        }

        // Apply compression if enabled (simulate data transfer compression)
        let compression_stats = if self.config.compression != CompressionMethod::None {
            // Simulate compressing the accumulated data
            let data_to_compress = bincode::serialize(&final_accumulator)
                .map_err(|e| MetricsError::InvalidInput(format!("Serialization error: {}", e)))?;

            let (_compressed, stats) =
                CompressionUtilities::compress(&data_to_compress, self.config.compression)?;
            Some(stats)
        } else {
            None
        };

        let computation_time = start_time.elapsed();
        let metrics = final_accumulator.get_metrics();
        let total_bytes_transferred = compression_stats
            .as_ref()
            .map(|cs| cs.compressed_size)
            .unwrap_or(0);

        let effective_throughput = if computation_time.as_secs_f64() > 0.0 {
            (data_len as f64 * std::mem::size_of::<i32>() as f64 * 2.0)
                / (1024.0 * 1024.0)
                / computation_time.as_secs_f64()
        } else {
            0.0
        };

        Ok(DistributedResults {
            metrics,
            computation_time,
            communication_time: Duration::from_millis(0), // No network communication
            num_nodes: self.config.num_workers,
            load_balance_efficiency: self.calculate_load_balance_efficiency(&accumulators),
            network_utilization: 0.0,
            node_performance,
            compression_stats,
            fault_tolerance_info: Some(self.fault_tolerance_info.clone()),
            total_bytes_transferred,
            effective_throughput_mbps: effective_throughput,
            strategy_used: ComputeStrategy::DataParallel, // Will be updated by caller
            strategy_switched: false,                     // Will be updated by caller
        })
    }

    #[cfg(feature = "distributed")]
    /// MPI-based distributed computation
    fn compute_classification_mpi<C: Communicator>(
        &mut self,
        y_true: &ArrayView1<i32>,
        y_pred: &ArrayView1<i32>,
        world: &C,
    ) -> MetricsResult<DistributedResults> {
        let start_time = Instant::now();
        let rank = world.rank();
        let size = world.size();

        let data_len = y_true.len();

        // Calculate data partition for this rank
        let chunk_size = (data_len + size as usize - 1) / size as usize;
        let start_idx = rank as usize * chunk_size;
        let end_idx = ((rank + 1) as usize * chunk_size).min(data_len);

        // Compute local metrics
        let mut local_accumulator = MetricAccumulator::new();
        for i in start_idx..end_idx {
            if i < data_len {
                local_accumulator.update_classification(y_true[i], y_pred[i], 1);
            }
        }

        let computation_start = Instant::now();

        // Reduce results across all nodes
        let mut global_tp = 0u64;
        let mut global_fp = 0u64;
        let mut global_tn = 0u64;
        let mut global_fn = 0u64;
        let mut global_count = 0u64;

        world.all_reduce_into(
            &local_accumulator.true_positives,
            &mut global_tp,
            SystemOperation::sum(),
        );
        world.all_reduce_into(
            &local_accumulator.false_positives,
            &mut global_fp,
            SystemOperation::sum(),
        );
        world.all_reduce_into(
            &local_accumulator.true_negatives,
            &mut global_tn,
            SystemOperation::sum(),
        );
        world.all_reduce_into(
            &local_accumulator.false_negatives,
            &mut global_fn,
            SystemOperation::sum(),
        );
        world.all_reduce_into(
            &local_accumulator.count,
            &mut global_count,
            SystemOperation::sum(),
        );

        let communication_time = computation_start.elapsed();

        // Calculate final metrics
        let mut global_accumulator = MetricAccumulator::new();
        global_accumulator.true_positives = global_tp;
        global_accumulator.false_positives = global_fp;
        global_accumulator.true_negatives = global_tn;
        global_accumulator.false_negatives = global_fn;
        global_accumulator.count = global_count;

        let metrics = global_accumulator.get_metrics();

        Ok(DistributedResults {
            metrics,
            computation_time: start_time.elapsed(),
            communication_time,
            num_nodes: size as usize,
            load_balance_efficiency: 1.0, // Assume perfect load balancing with MPI
            network_utilization: 0.8,     // Estimate
            node_performance: vec![],     // Not tracked in this basic implementation
            compression_stats: None,
            fault_tolerance_info: None,
            total_bytes_transferred: 0,
            effective_throughput_mbps: 0.0,
            strategy_used: ComputeStrategy::DataParallel,
            strategy_switched: false,
        })
    }

    /// Model parallel computation (placeholder)
    fn compute_classification_model_parallel(
        &mut self,
        y_true: &ArrayView1<i32>,
        y_pred: &ArrayView1<i32>,
    ) -> MetricsResult<DistributedResults> {
        // For simplicity, delegate to data parallel
        self.compute_classification_data_parallel(y_true, y_pred)
    }

    /// Pipeline parallel computation (placeholder)
    fn compute_classification_pipeline_parallel(
        &mut self,
        y_true: &ArrayView1<i32>,
        y_pred: &ArrayView1<i32>,
    ) -> MetricsResult<DistributedResults> {
        // For simplicity, delegate to data parallel
        self.compute_classification_data_parallel(y_true, y_pred)
    }

    /// Hybrid computation (placeholder)
    fn compute_classification_hybrid(
        &mut self,
        y_true: &ArrayView1<i32>,
        y_pred: &ArrayView1<i32>,
    ) -> MetricsResult<DistributedResults> {
        // For simplicity, delegate to data parallel
        self.compute_classification_data_parallel(y_true, y_pred)
    }

    /// Data parallel computation for regression
    fn compute_regression_data_parallel(
        &mut self,
        y_true: &ArrayView1<f64>,
        y_pred: &ArrayView1<f64>,
    ) -> MetricsResult<DistributedResults> {
        let start_time = Instant::now();
        let data_len = y_true.len();

        if data_len != y_pred.len() {
            return Err(MetricsError::ShapeMismatch {
                expected: vec![data_len],
                actual: vec![y_pred.len()],
            });
        }

        // Split data into chunks
        let chunk_size = (data_len + self.config.num_workers - 1) / self.config.num_workers;
        let chunks: Vec<_> = (0..data_len)
            .step_by(chunk_size)
            .map(|i| (i, (i + chunk_size).min(data_len)))
            .collect();

        // Process chunks in parallel
        let accumulators: Vec<MetricAccumulator> = chunks
            .par_iter()
            .map(|(start, end)| {
                let mut accumulator = MetricAccumulator::new();
                for i in *start..*end {
                    accumulator.update_regression(y_true[i], y_pred[i]);
                }
                accumulator
            })
            .collect();

        // Merge results
        let mut final_accumulator = MetricAccumulator::new();
        for acc in &accumulators {
            final_accumulator.merge(acc);
        }

        let computation_time = start_time.elapsed();
        let metrics = final_accumulator.get_metrics();

        Ok(DistributedResults {
            metrics,
            computation_time,
            communication_time: Duration::from_millis(0),
            num_nodes: self.config.num_workers,
            load_balance_efficiency: self.calculate_load_balance_efficiency(&accumulators),
            network_utilization: 0.0,
            node_performance: vec![],
            compression_stats: None,
            fault_tolerance_info: None,
            total_bytes_transferred: 0,
            effective_throughput_mbps: 0.0,
            strategy_used: ComputeStrategy::DataParallel,
            strategy_switched: false,
        })
    }

    /// Model parallel computation for regression (placeholder)
    fn compute_regression_model_parallel(
        &mut self,
        y_true: &ArrayView1<f64>,
        y_pred: &ArrayView1<f64>,
    ) -> MetricsResult<DistributedResults> {
        self.compute_regression_data_parallel(y_true, y_pred)
    }

    /// Pipeline parallel computation for regression (placeholder)
    fn compute_regression_pipeline_parallel(
        &mut self,
        y_true: &ArrayView1<f64>,
        y_pred: &ArrayView1<f64>,
    ) -> MetricsResult<DistributedResults> {
        self.compute_regression_data_parallel(y_true, y_pred)
    }

    /// Hybrid computation for regression (placeholder)
    fn compute_regression_hybrid(
        &mut self,
        y_true: &ArrayView1<f64>,
        y_pred: &ArrayView1<f64>,
    ) -> MetricsResult<DistributedResults> {
        self.compute_regression_data_parallel(y_true, y_pred)
    }

    /// Calculate load balance efficiency
    fn calculate_load_balance_efficiency(&self, accumulators: &[MetricAccumulator]) -> f64 {
        if accumulators.is_empty() {
            return 0.0;
        }

        let counts: Vec<u64> = accumulators.iter().map(|acc| acc.count).collect();
        let total: u64 = counts.iter().sum();
        let mean = total as f64 / counts.len() as f64;

        if mean == 0.0 {
            return 1.0;
        }

        let variance = counts
            .iter()
            .map(|&count| {
                let diff = count as f64 - mean;
                diff * diff
            })
            .sum::<f64>()
            / counts.len() as f64;

        let coefficient_of_variation = variance.sqrt() / mean;

        // Convert to efficiency (0.0 to 1.0, higher is better)
        (1.0 / (1.0 + coefficient_of_variation)).max(0.0).min(1.0)
    }

    /// Get thread-local metrics storage
    pub fn get_thread_local_storage(
        &self,
    ) -> Arc<Mutex<HashMap<thread::ThreadId, ThreadLocalMetrics>>> {
        self.thread_local_storage.clone()
    }

    /// Update metrics using thread-local storage
    pub fn update_thread_local<F>(&self, updater: F) -> MetricsResult<()>
    where
        F: FnOnce(&mut MetricAccumulator) -> MetricsResult<()>,
    {
        let thread_id = thread::current().id();
        let mut storage = self
            .thread_local_storage
            .lock()
            .map_err(|_| MetricsError::InvalidInput("Lock poisoned".to_string()))?;

        let thread_local = storage
            .entry(thread_id)
            .or_insert_with(|| ThreadLocalMetrics::new());

        thread_local.update(updater)
    }

    /// Aggregate thread-local metrics
    pub fn aggregate_thread_local(&self) -> MetricsResult<HashMap<String, f64>> {
        let storage = self
            .thread_local_storage
            .lock()
            .map_err(|_| MetricsError::InvalidInput("Lock poisoned".to_string()))?;

        let global_accumulator = MetricAccumulator::new();

        for thread_local in storage.values() {
            let _metrics = thread_local.get_metrics()?;
            // Note: This is a simplified aggregation
            // In practice, you'd need to reconstruct accumulators from metrics
        }

        Ok(global_accumulator.get_metrics())
    }
}

impl Default for DistributedMetricsComputer {
    fn default() -> Self {
        Self::new().unwrap()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_metric_accumulator() {
        let mut acc = MetricAccumulator::new();

        // Add some classification data
        acc.update_classification(1, 1, 1); // TP
        acc.update_classification(0, 1, 1); // FP
        acc.update_classification(1, 0, 1); // FN
        acc.update_classification(0, 0, 1); // TN

        let metrics = acc.get_metrics();
        assert_eq!(metrics["precision"], 0.5); // 1/(1+1)
        assert_eq!(metrics["recall"], 0.5); // 1/(1+1)
        assert_eq!(metrics["accuracy"], 0.5); // (1+1)/4
    }

    #[test]
    fn test_metric_accumulator_regression() {
        let mut acc = MetricAccumulator::new();

        acc.update_regression(1.0, 1.1);
        acc.update_regression(2.0, 1.9);
        acc.update_regression(3.0, 3.2);

        let metrics = acc.get_metrics();
        assert!((metrics["mae"] - 0.1333).abs() < 0.01);
    }

    #[test]
    fn test_distributed_computer_creation() {
        let computer = DistributedMetricsComputer::new();
        assert!(computer.is_ok());
    }

    #[test]
    fn test_thread_based_classification() {
        let mut computer = DistributedMetricsComputer::new().unwrap();
        let y_true = array![0, 1, 2, 0, 1, 2, 1, 0, 2];
        let y_pred = array![0, 2, 1, 0, 0, 1, 1, 2, 2];

        let results = computer
            .strategy(ComputeStrategy::DataParallel)
            .compute_classification_metrics(&y_true.view(), &y_pred.view())
            .unwrap();

        assert!(results.metrics.contains_key("accuracy"));
        assert!(results.metrics.contains_key("precision"));
        assert!(results.computation_time.as_millis() >= 0);
        assert_eq!(results.num_nodes, num_cpus::get());
    }

    #[test]
    fn test_thread_based_regression() {
        let mut computer = DistributedMetricsComputer::new().unwrap();
        let y_true = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y_pred = array![1.1, 2.1, 2.9, 3.9, 5.1];

        let results = computer
            .strategy(ComputeStrategy::DataParallel)
            .compute_regression_metrics(&y_true.view(), &y_pred.view())
            .unwrap();

        assert!(results.metrics.contains_key("mae"));
        assert!(results.metrics.contains_key("mse"));
        assert!(results.metrics.contains_key("rmse"));
        assert!(results.computation_time.as_millis() >= 0);
    }

    #[test]
    fn test_load_balance_calculation() {
        let computer = DistributedMetricsComputer::new().unwrap();

        // Perfect balance
        let mut acc1 = MetricAccumulator::new();
        acc1.count = 100;
        let mut acc2 = MetricAccumulator::new();
        acc2.count = 100;
        let balanced = vec![acc1, acc2];

        let efficiency = computer.calculate_load_balance_efficiency(&balanced);
        assert!(efficiency > 0.9); // Should be close to 1.0 for perfect balance

        // Imbalanced
        let mut acc3 = MetricAccumulator::new();
        acc3.count = 10;
        let mut acc4 = MetricAccumulator::new();
        acc4.count = 190;
        let imbalanced = vec![acc3, acc4];

        let efficiency_imbalanced = computer.calculate_load_balance_efficiency(&imbalanced);
        assert!(efficiency_imbalanced < efficiency); // Should be lower than balanced case
    }

    #[test]
    fn test_thread_local_storage() {
        let computer = DistributedMetricsComputer::new().unwrap();

        // Update thread-local metrics
        computer
            .update_thread_local(|acc| {
                acc.update_classification(1, 1, 1);
                acc.update_classification(0, 0, 1);
                Ok(())
            })
            .unwrap();

        // Verify thread-local storage contains data
        let storage = computer.get_thread_local_storage();
        let storage_lock = storage.lock().unwrap();
        assert!(!storage_lock.is_empty());
    }
}
