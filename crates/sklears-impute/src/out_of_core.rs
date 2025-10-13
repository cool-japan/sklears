//! Out-of-core imputation algorithms for datasets larger than memory
//!
//! This module provides imputation methods that can process datasets that don't
//! fit entirely in memory by streaming data from disk and processing it in chunks.

// ✅ SciRS2 Policy compliant imports
use scirs2_core::ndarray::{s, Array1, Array2, ArrayView2, Axis};
// use scirs2_core::memory_efficient::{
//     MemoryMappedArray, ChunkedArray, AdaptiveChunking, DiskBackedArray
// }; // Note: memory_efficient feature-gated
// use scirs2_core::memory::{GlobalBufferPool, ChunkProcessor}; // Note: these may not be available

use crate::core::ImputationError;
use memmap2::Mmap;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Transform, Untrained},
    types::Float,
};
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// Configuration for out-of-core processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutOfCoreConfig {
    /// Maximum memory usage in bytes
    pub max_memory_usage: usize,
    /// Chunk size for processing
    pub chunk_size: usize,
    /// Number of chunks to keep in memory
    pub memory_pool_size: usize,
    /// Temporary directory for intermediate files
    pub temp_dir: PathBuf,
    /// Enable compression for temporary files
    pub compression_enabled: bool,
    /// Enable memory mapping for large files
    pub memory_mapping_enabled: bool,
    /// Buffer size for I/O operations
    pub io_buffer_size: usize,
    /// Prefetch strategy for data loading
    pub prefetch_strategy: PrefetchStrategy,
}

impl Default for OutOfCoreConfig {
    fn default() -> Self {
        Self {
            max_memory_usage: 1_073_741_824, // 1 GB
            chunk_size: 100_000,             // 100k rows
            memory_pool_size: 10,
            temp_dir: std::env::temp_dir(),
            compression_enabled: false,
            memory_mapping_enabled: true,
            io_buffer_size: 65_536, // 64 KB
            prefetch_strategy: PrefetchStrategy::Sequential,
        }
    }
}

/// Prefetch strategies for data loading
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PrefetchStrategy {
    /// No prefetching
    None,
    /// Sequential prefetching
    Sequential,
    /// Random access prefetching
    Random,
    /// Adaptive prefetching based on access patterns
    Adaptive,
}

/// Out-of-core data chunk
#[derive(Debug)]
pub struct OutOfCoreChunk {
    /// id
    pub id: usize,
    /// start_row
    pub start_row: usize,
    /// end_row
    pub end_row: usize,
    /// data
    pub data: Option<Array2<f64>>,
    /// file_path
    pub file_path: Option<PathBuf>,
    /// memory_mapped
    pub memory_mapped: Option<Mmap>,
    /// missing_mask
    pub missing_mask: Array2<bool>,
    /// in_memory
    pub in_memory: bool,
    /// last_accessed
    pub last_accessed: Instant,
    /// dirty
    pub dirty: bool,
}

/// Memory manager for out-of-core processing
#[derive(Debug)]
pub struct MemoryManager {
    config: OutOfCoreConfig,
    chunks: HashMap<usize, OutOfCoreChunk>,
    memory_usage: usize,
    buffer_pool: Arc<Mutex<Vec<Vec<u8>>>>, // Simplified buffer pool
    lru_order: Vec<usize>,
}

/// File-based data storage for out-of-core processing
#[derive(Debug)]
pub struct FileBackedDataset {
    /// file_path
    pub file_path: PathBuf,
    /// n_samples
    pub n_samples: usize,
    /// n_features
    pub n_features: usize,
    /// chunk_size
    pub chunk_size: usize,
    /// memory_map
    pub memory_map: Option<Mmap>,
    /// metadata
    pub metadata: DatasetMetadata,
}

/// Metadata for file-backed datasets
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetMetadata {
    /// dtype
    pub dtype: String,
    /// shape
    pub shape: (usize, usize),
    /// chunk_size
    pub chunk_size: usize,
    /// missing_value_encoding
    pub missing_value_encoding: f64,
    /// creation_time
    pub creation_time: std::time::SystemTime,
    /// file_size
    pub file_size: u64,
}

/// Out-of-core Simple Imputer
#[derive(Debug)]
pub struct OutOfCoreSimpleImputer<S = Untrained> {
    state: S,
    strategy: String,
    missing_values: f64,
    config: OutOfCoreConfig,
    memory_manager: Option<MemoryManager>,
}

/// Trained state for out-of-core simple imputer
#[derive(Debug)]
pub struct OutOfCoreSimpleImputerTrained {
    statistics_: Array1<f64>,
    n_features_in_: usize,
    config: OutOfCoreConfig,
    memory_manager: MemoryManager,
    dataset_metadata: DatasetMetadata,
}

/// Out-of-core KNN Imputer
#[derive(Debug)]
pub struct OutOfCoreKNNImputer<S = Untrained> {
    state: S,
    n_neighbors: usize,
    weights: String,
    missing_values: f64,
    config: OutOfCoreConfig,
    reference_dataset: Option<FileBackedDataset>,
}

/// Trained state for out-of-core KNN imputer
#[derive(Debug)]
pub struct OutOfCoreKNNImputerTrained {
    reference_dataset: FileBackedDataset,
    n_features_in_: usize,
    config: OutOfCoreConfig,
    memory_manager: MemoryManager,
    neighbor_index: Option<NeighborIndex>,
}

/// Index structure for efficient neighbor search
#[derive(Debug)]
pub struct NeighborIndex {
    /// index_type
    pub index_type: IndexType,
    /// index_file
    pub index_file: PathBuf,
    /// chunk_centroids
    pub chunk_centroids: Array2<f64>,
    /// chunk_bounds
    pub chunk_bounds: Vec<(usize, usize)>,
}

/// Types of neighbor indices
#[derive(Debug, Clone)]
pub enum IndexType {
    /// Brute force search
    BruteForce,
    /// LSH (Locality Sensitive Hashing)
    LSH,
    /// KD-Tree based index
    KDTree,
    /// Approximate nearest neighbor index
    ANN,
}

impl OutOfCoreSimpleImputer<Untrained> {
    pub fn new() -> Self {
        Self {
            state: Untrained,
            strategy: "mean".to_string(),
            missing_values: f64::NAN,
            config: OutOfCoreConfig::default(),
            memory_manager: None,
        }
    }

    pub fn strategy(mut self, strategy: String) -> Self {
        self.strategy = strategy;
        self
    }

    pub fn out_of_core_config(mut self, config: OutOfCoreConfig) -> Self {
        self.config = config;
        self
    }

    pub fn max_memory_usage(mut self, max_memory: usize) -> Self {
        self.config.max_memory_usage = max_memory;
        self
    }

    pub fn chunk_size(mut self, chunk_size: usize) -> Self {
        self.config.chunk_size = chunk_size;
        self
    }

    pub fn temp_dir<P: AsRef<Path>>(mut self, temp_dir: P) -> Self {
        self.config.temp_dir = temp_dir.as_ref().to_path_buf();
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

impl Default for OutOfCoreSimpleImputer<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for OutOfCoreSimpleImputer<Untrained> {
    type Config = OutOfCoreConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Fit<ArrayView2<'_, Float>, ()> for OutOfCoreSimpleImputer<Untrained> {
    type Fitted = OutOfCoreSimpleImputer<OutOfCoreSimpleImputerTrained>;

    #[allow(non_snake_case)]
    fn fit(self, X: &ArrayView2<'_, Float>, _y: &()) -> SklResult<Self::Fitted> {
        let X = X.mapv(|x| x as f64);
        let (n_samples, n_features) = X.dim();

        // Create memory manager
        let buffer_pool = Arc::new(Mutex::new(Vec::with_capacity(self.config.memory_pool_size)));
        let memory_manager = MemoryManager {
            config: self.config.clone(),
            chunks: HashMap::new(),
            memory_usage: 0,
            buffer_pool,
            lru_order: Vec::new(),
        };

        // Create dataset metadata
        let dataset_metadata = DatasetMetadata {
            dtype: "f64".to_string(),
            shape: (n_samples, n_features),
            chunk_size: self.config.chunk_size,
            missing_value_encoding: self.missing_values,
            creation_time: std::time::SystemTime::now(),
            file_size: (n_samples * n_features * 8) as u64, // 8 bytes per f64
        };

        // Check if we need out-of-core processing
        let data_size = n_samples * n_features * std::mem::size_of::<f64>();
        let use_out_of_core = data_size > self.config.max_memory_usage;

        let statistics = if use_out_of_core {
            self.compute_statistics_out_of_core(&X)?
        } else {
            self.compute_statistics_in_memory(&X)?
        };

        Ok(OutOfCoreSimpleImputer {
            state: OutOfCoreSimpleImputerTrained {
                statistics_: statistics,
                n_features_in_: n_features,
                config: self.config,
                memory_manager,
                dataset_metadata,
            },
            strategy: self.strategy,
            missing_values: self.missing_values,
            config: Default::default(),
            memory_manager: None,
        })
    }
}

impl Transform<ArrayView2<'_, Float>, Array2<Float>>
    for OutOfCoreSimpleImputer<OutOfCoreSimpleImputerTrained>
{
    #[allow(non_snake_case)]
    fn transform(&self, X: &ArrayView2<'_, Float>) -> SklResult<Array2<Float>> {
        let X = X.mapv(|x| x as f64);
        let (n_samples, n_features) = X.dim();

        if n_features != self.state.n_features_in_ {
            return Err(SklearsError::InvalidInput(format!(
                "Number of features {} does not match training features {}",
                n_features, self.state.n_features_in_
            )));
        }

        // Check if we need out-of-core processing
        let data_size = n_samples * n_features * std::mem::size_of::<f64>();
        let use_out_of_core = data_size > self.state.config.max_memory_usage;

        let X_imputed = if use_out_of_core {
            self.transform_out_of_core(&X)?
        } else {
            self.transform_in_memory(&X)?
        };

        Ok(X_imputed.mapv(|x| x as Float))
    }
}

impl OutOfCoreSimpleImputer<Untrained> {
    /// Compute statistics using out-of-core processing
    fn compute_statistics_out_of_core(&self, X: &Array2<f64>) -> Result<Array1<f64>, SklearsError> {
        let (n_samples, n_features) = X.dim();
        let mut statistics = Array1::<f64>::zeros(n_features);

        // Process data in chunks
        let chunk_size = self.config.chunk_size.min(n_samples);

        match self.strategy.as_str() {
            "mean" => {
                let mut sums = Array1::<f64>::zeros(n_features);
                let mut counts = Array1::<usize>::zeros(n_features);

                for chunk_start in (0..n_samples).step_by(chunk_size) {
                    let chunk_end = (chunk_start + chunk_size).min(n_samples);
                    let chunk = X.slice(s![chunk_start..chunk_end, ..]);

                    for ((_, j), &value) in chunk.indexed_iter() {
                        if !self.is_missing(value) {
                            sums[j] += value;
                            counts[j] += 1;
                        }
                    }
                }

                for j in 0..n_features {
                    statistics[j] = if counts[j] > 0 {
                        sums[j] / counts[j] as f64
                    } else {
                        0.0
                    };
                }
            }
            "median" => {
                // For median, we need to collect all values (more memory intensive)
                for j in 0..n_features {
                    let mut values = Vec::new();

                    for chunk_start in (0..n_samples).step_by(chunk_size) {
                        let chunk_end = (chunk_start + chunk_size).min(n_samples);
                        let chunk = X.slice(s![chunk_start..chunk_end, j..j + 1]);

                        for &value in chunk.iter() {
                            if !self.is_missing(value) {
                                values.push(value);
                            }
                        }
                    }

                    if !values.is_empty() {
                        values.sort_by(|a, b| a.partial_cmp(b).unwrap());
                        let mid = values.len() / 2;
                        statistics[j] = if values.len() % 2 == 0 {
                            (values[mid - 1] + values[mid]) / 2.0
                        } else {
                            values[mid]
                        };
                    }
                }
            }
            "most_frequent" => {
                for j in 0..n_features {
                    let mut frequency_map = HashMap::new();

                    for chunk_start in (0..n_samples).step_by(chunk_size) {
                        let chunk_end = (chunk_start + chunk_size).min(n_samples);
                        let chunk = X.slice(s![chunk_start..chunk_end, j..j + 1]);

                        for &value in chunk.iter() {
                            if !self.is_missing(value) {
                                let key = (value * 1e6) as i64; // Handle floating point precision
                                *frequency_map.entry(key).or_insert(0) += 1;
                            }
                        }
                    }

                    statistics[j] = frequency_map
                        .into_iter()
                        .max_by_key(|(_, count)| *count)
                        .map(|(value, _)| value as f64 / 1e6)
                        .unwrap_or(0.0);
                }
            }
            _ => {
                return Err(SklearsError::InvalidInput(format!(
                    "Unknown strategy: {}",
                    self.strategy
                )));
            }
        }

        Ok(statistics)
    }

    /// Compute statistics using in-memory processing
    fn compute_statistics_in_memory(&self, X: &Array2<f64>) -> Result<Array1<f64>, SklearsError> {
        let (_, n_features) = X.dim();
        let mut statistics = Array1::<f64>::zeros(n_features);

        for j in 0..n_features {
            let column = X.column(j);
            let valid_values: Vec<f64> = column
                .iter()
                .filter(|&&x| !self.is_missing(x))
                .cloned()
                .collect();

            if valid_values.is_empty() {
                statistics[j] = 0.0;
            } else {
                statistics[j] = match self.strategy.as_str() {
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
                            let key = (value * 1e6) as i64;
                            *frequency_map.entry(key).or_insert(0) += 1;
                        }
                        frequency_map
                            .into_iter()
                            .max_by_key(|(_, count)| *count)
                            .map(|(value, _)| value as f64 / 1e6)
                            .unwrap_or(0.0)
                    }
                    _ => valid_values.iter().sum::<f64>() / valid_values.len() as f64,
                };
            }
        }

        Ok(statistics)
    }
}

impl OutOfCoreSimpleImputer<OutOfCoreSimpleImputerTrained> {
    /// Transform data using out-of-core processing
    fn transform_out_of_core(&self, X: &Array2<f64>) -> Result<Array2<f64>, SklearsError> {
        let (n_samples, n_features) = X.dim();
        let mut X_imputed = X.clone();
        let chunk_size = self.state.config.chunk_size.min(n_samples);

        // Process data in chunks
        for chunk_start in (0..n_samples).step_by(chunk_size) {
            let chunk_end = (chunk_start + chunk_size).min(n_samples);
            let mut chunk = X_imputed.slice_mut(s![chunk_start..chunk_end, ..]);

            // Impute missing values in this chunk
            for ((i, j), value) in chunk.indexed_iter_mut() {
                if self.is_missing(*value) {
                    *value = self.state.statistics_[j];
                }
            }
        }

        Ok(X_imputed)
    }

    /// Transform data using in-memory processing
    fn transform_in_memory(&self, X: &Array2<f64>) -> Result<Array2<f64>, SklearsError> {
        let (n_samples, n_features) = X.dim();
        let mut X_imputed = X.clone();

        // Parallel imputation across rows
        X_imputed
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .for_each(|mut row| {
                for (j, value) in row.iter_mut().enumerate() {
                    if self.is_missing(*value) {
                        *value = self.state.statistics_[j];
                    }
                }
            });

        Ok(X_imputed)
    }

    fn is_missing(&self, value: f64) -> bool {
        if self.missing_values.is_nan() {
            value.is_nan()
        } else {
            (value - self.missing_values).abs() < f64::EPSILON
        }
    }
}

impl OutOfCoreKNNImputer<Untrained> {
    pub fn new() -> Self {
        Self {
            state: Untrained,
            n_neighbors: 5,
            weights: "uniform".to_string(),
            missing_values: f64::NAN,
            config: OutOfCoreConfig::default(),
            reference_dataset: None,
        }
    }

    pub fn n_neighbors(mut self, n_neighbors: usize) -> Self {
        self.n_neighbors = n_neighbors;
        self
    }

    pub fn weights(mut self, weights: String) -> Self {
        self.weights = weights;
        self
    }

    pub fn out_of_core_config(mut self, config: OutOfCoreConfig) -> Self {
        self.config = config;
        self
    }

    pub fn max_memory_usage(mut self, max_memory: usize) -> Self {
        self.config.max_memory_usage = max_memory;
        self
    }

    pub fn chunk_size(mut self, chunk_size: usize) -> Self {
        self.config.chunk_size = chunk_size;
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

impl Default for OutOfCoreKNNImputer<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for OutOfCoreKNNImputer<Untrained> {
    type Config = OutOfCoreConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Fit<ArrayView2<'_, Float>, ()> for OutOfCoreKNNImputer<Untrained> {
    type Fitted = OutOfCoreKNNImputer<OutOfCoreKNNImputerTrained>;

    #[allow(non_snake_case)]
    fn fit(self, X: &ArrayView2<'_, Float>, _y: &()) -> SklResult<Self::Fitted> {
        let X = X.mapv(|x| x as f64);
        let (n_samples, n_features) = X.dim();

        // Create temporary file for reference dataset
        let temp_file = self
            .config
            .temp_dir
            .join(format!("reference_data_{}.bin", std::process::id()));

        // Create file-backed dataset
        let reference_dataset = FileBackedDataset {
            file_path: temp_file.clone(),
            n_samples,
            n_features,
            chunk_size: self.config.chunk_size,
            memory_map: None,
            metadata: DatasetMetadata {
                dtype: "f64".to_string(),
                shape: (n_samples, n_features),
                chunk_size: self.config.chunk_size,
                missing_value_encoding: self.missing_values,
                creation_time: std::time::SystemTime::now(),
                file_size: (n_samples * n_features * 8) as u64,
            },
        };

        // Write data to file
        self.write_dataset_to_file(&X, &temp_file)?;

        // Create memory manager
        let buffer_pool = Arc::new(Mutex::new(Vec::with_capacity(self.config.memory_pool_size)));
        let memory_manager = MemoryManager {
            config: self.config.clone(),
            chunks: HashMap::new(),
            memory_usage: 0,
            buffer_pool,
            lru_order: Vec::new(),
        };

        // Build neighbor index for efficient search
        let neighbor_index = self.build_neighbor_index(&X)?;

        Ok(OutOfCoreKNNImputer {
            state: OutOfCoreKNNImputerTrained {
                reference_dataset,
                n_features_in_: n_features,
                config: self.config,
                memory_manager,
                neighbor_index: Some(neighbor_index),
            },
            n_neighbors: self.n_neighbors,
            weights: self.weights,
            missing_values: self.missing_values,
            config: Default::default(),
            reference_dataset: None,
        })
    }
}

impl Transform<ArrayView2<'_, Float>, Array2<Float>>
    for OutOfCoreKNNImputer<OutOfCoreKNNImputerTrained>
{
    #[allow(non_snake_case)]
    fn transform(&self, X: &ArrayView2<'_, Float>) -> SklResult<Array2<Float>> {
        let X = X.mapv(|x| x as f64);
        let (n_samples, n_features) = X.dim();

        if n_features != self.state.n_features_in_ {
            return Err(SklearsError::InvalidInput(format!(
                "Number of features {} does not match training features {}",
                n_features, self.state.n_features_in_
            )));
        }

        // Check if we need out-of-core processing
        let data_size = n_samples * n_features * std::mem::size_of::<f64>();
        let use_out_of_core = data_size > self.state.config.max_memory_usage;

        let X_imputed = if use_out_of_core {
            self.transform_out_of_core(&X)?
        } else {
            self.transform_in_memory(&X)?
        };

        Ok(X_imputed.mapv(|x| x as Float))
    }
}

impl OutOfCoreKNNImputer<Untrained> {
    /// Write dataset to file
    fn write_dataset_to_file(&self, X: &Array2<f64>, file_path: &Path) -> Result<(), SklearsError> {
        let mut file = File::create(file_path)
            .map_err(|e| SklearsError::InvalidInput(format!("Failed to create file: {}", e)))?;

        // Write data in binary format
        for row in X.rows() {
            for &value in row {
                file.write_all(&value.to_le_bytes()).map_err(|e| {
                    SklearsError::InvalidInput(format!("Failed to write data: {}", e))
                })?;
            }
        }

        file.flush()
            .map_err(|e| SklearsError::InvalidInput(format!("Failed to flush file: {}", e)))?;

        Ok(())
    }

    /// Build neighbor index for efficient search
    fn build_neighbor_index(&self, X: &Array2<f64>) -> Result<NeighborIndex, SklearsError> {
        let (n_samples, n_features) = X.dim();
        let chunk_size = self.config.chunk_size;
        let num_chunks = (n_samples + chunk_size - 1) / chunk_size;

        // Compute chunk centroids for approximate search
        let mut chunk_centroids = Array2::<f64>::zeros((num_chunks, n_features));
        let mut chunk_bounds = Vec::new();

        for (chunk_idx, chunk_start) in (0..n_samples).step_by(chunk_size).enumerate() {
            let chunk_end = (chunk_start + chunk_size).min(n_samples);
            let chunk = X.slice(s![chunk_start..chunk_end, ..]);

            // Compute centroid
            let mut centroid = Array1::<f64>::zeros(n_features);
            let mut valid_counts = Array1::<usize>::zeros(n_features);

            for row in chunk.rows() {
                for (j, &value) in row.iter().enumerate() {
                    if !self.is_missing(value) {
                        centroid[j] += value;
                        valid_counts[j] += 1;
                    }
                }
            }

            for j in 0..n_features {
                if valid_counts[j] > 0 {
                    centroid[j] /= valid_counts[j] as f64;
                }
            }

            chunk_centroids.row_mut(chunk_idx).assign(&centroid);
            chunk_bounds.push((chunk_start, chunk_end));
        }

        // Create index file path
        let index_file = self
            .config
            .temp_dir
            .join(format!("neighbor_index_{}.bin", std::process::id()));

        Ok(NeighborIndex {
            index_type: IndexType::BruteForce,
            index_file,
            chunk_centroids,
            chunk_bounds,
        })
    }
}

impl OutOfCoreKNNImputer<OutOfCoreKNNImputerTrained> {
    /// Transform data using out-of-core processing
    fn transform_out_of_core(&self, X: &Array2<f64>) -> Result<Array2<f64>, SklearsError> {
        let (n_samples, n_features) = X.dim();
        let mut X_imputed = X.clone();
        let chunk_size = self.state.config.chunk_size.min(n_samples);

        // Process data in chunks
        for chunk_start in (0..n_samples).step_by(chunk_size) {
            let chunk_end = (chunk_start + chunk_size).min(n_samples);
            let mut chunk = X_imputed.slice_mut(s![chunk_start..chunk_end, ..]);

            // Process each row in the chunk
            for (row_idx, mut row) in chunk.rows_mut().into_iter().enumerate() {
                // Clone row data before mutable iteration to avoid borrow conflicts
                let row_data = row.to_owned();
                for (j, value) in row.iter_mut().enumerate() {
                    if self.is_missing(*value) {
                        // Find k nearest neighbors for this missing value
                        let neighbors = self.find_neighbors_out_of_core(&row_data, j)?;

                        if !neighbors.is_empty() {
                            *value = self.compute_weighted_average(&neighbors)?;
                        }
                    }
                }
            }
        }

        Ok(X_imputed)
    }

    /// Transform data using in-memory processing
    fn transform_in_memory(&self, X: &Array2<f64>) -> Result<Array2<f64>, SklearsError> {
        // Load reference data into memory if possible
        let reference_data = self.load_reference_data_chunk(0, None)?;

        let mut X_imputed = X.clone();

        // Process each sample
        for (i, mut row) in X_imputed.rows_mut().into_iter().enumerate() {
            // Clone row data before mutable iteration to avoid borrow conflicts
            let row_data = row.to_owned();
            for (j, value) in row.iter_mut().enumerate() {
                if self.is_missing(*value) {
                    let neighbors = self.find_neighbors_in_memory(&reference_data, &row_data, j)?;

                    if !neighbors.is_empty() {
                        *value = self.compute_weighted_average(&neighbors)?;
                    }
                }
            }
        }

        Ok(X_imputed)
    }

    /// Find neighbors using out-of-core processing
    fn find_neighbors_out_of_core(
        &self,
        query_row: &Array1<f64>,
        target_feature: usize,
    ) -> Result<Vec<(f64, f64)>, SklearsError> {
        let mut all_neighbors = Vec::new();

        // Use neighbor index to identify relevant chunks
        if let Some(ref index) = self.state.neighbor_index {
            let query_distances: Vec<f64> = index
                .chunk_centroids
                .rows()
                .into_iter()
                .map(|centroid| self.calculate_distance(query_row, &centroid.to_owned()))
                .collect();

            // Sort chunks by distance to query
            let mut chunk_distances: Vec<(usize, f64)> =
                query_distances.into_iter().enumerate().collect();
            chunk_distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

            // Search the closest chunks
            let max_chunks_to_search = 3.min(chunk_distances.len());
            for (chunk_idx, _) in chunk_distances.into_iter().take(max_chunks_to_search) {
                let (chunk_start, chunk_end) = index.chunk_bounds[chunk_idx];
                let chunk_data =
                    self.load_reference_data_chunk(chunk_start, Some(chunk_end - chunk_start))?;

                let chunk_neighbors =
                    self.find_neighbors_in_memory(&chunk_data, query_row, target_feature)?;
                all_neighbors.extend(chunk_neighbors);
            }
        }

        // Sort all neighbors by distance and take k best
        all_neighbors.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        all_neighbors.truncate(self.n_neighbors);

        Ok(all_neighbors)
    }

    /// Find neighbors in memory
    fn find_neighbors_in_memory(
        &self,
        reference_data: &Array2<f64>,
        query_row: &Array1<f64>,
        target_feature: usize,
    ) -> Result<Vec<(f64, f64)>, SklearsError> {
        let mut neighbors = Vec::new();

        for ref_row in reference_data.rows() {
            // Skip if reference row has missing value for target feature
            if self.is_missing(ref_row[target_feature]) {
                continue;
            }

            let distance = self.calculate_distance(query_row, &ref_row.to_owned());
            if distance.is_finite() {
                neighbors.push((distance, ref_row[target_feature]));
            }
        }

        // Sort by distance and take k nearest
        neighbors.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        neighbors.truncate(self.n_neighbors);

        Ok(neighbors)
    }

    /// Load a chunk of reference data
    fn load_reference_data_chunk(
        &self,
        start_row: usize,
        num_rows: Option<usize>,
    ) -> Result<Array2<f64>, SklearsError> {
        let file_path = &self.state.reference_dataset.file_path;
        let n_features = self.state.n_features_in_;
        let rows_to_read = num_rows.unwrap_or(self.state.reference_dataset.n_samples - start_row);

        let mut file = File::open(file_path).map_err(|e| {
            SklearsError::InvalidInput(format!("Failed to open reference file: {}", e))
        })?;

        // Seek to the correct position
        let byte_offset = start_row * n_features * std::mem::size_of::<f64>();
        file.seek(SeekFrom::Start(byte_offset as u64))
            .map_err(|e| SklearsError::InvalidInput(format!("Failed to seek in file: {}", e)))?;

        // Read data
        let mut data = Array2::<f64>::zeros((rows_to_read, n_features));
        let mut buffer = [0u8; 8]; // Buffer for one f64

        for i in 0..rows_to_read {
            for j in 0..n_features {
                file.read_exact(&mut buffer).map_err(|e| {
                    SklearsError::InvalidInput(format!("Failed to read data: {}", e))
                })?;
                data[[i, j]] = f64::from_le_bytes(buffer);
            }
        }

        Ok(data)
    }

    /// Calculate distance between two rows
    fn calculate_distance(&self, row1: &Array1<f64>, row2: &Array1<f64>) -> f64 {
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
    fn compute_weighted_average(&self, neighbors: &[(f64, f64)]) -> Result<f64, SklearsError> {
        if neighbors.is_empty() {
            return Ok(0.0);
        }

        match self.weights.as_str() {
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
                    Ok(neighbors[0].1)
                }
            }
            _ => Err(SklearsError::InvalidInput(format!(
                "Unknown weights: {}",
                self.weights
            ))),
        }
    }

    fn is_missing(&self, value: f64) -> bool {
        if self.missing_values.is_nan() {
            value.is_nan()
        } else {
            (value - self.missing_values).abs() < f64::EPSILON
        }
    }
}

// Implement memory management utilities
impl MemoryManager {
    pub fn new(config: OutOfCoreConfig) -> Self {
        let memory_pool_size = config.memory_pool_size;
        Self {
            config,
            chunks: HashMap::new(),
            memory_usage: 0,
            buffer_pool: Arc::new(Mutex::new(Vec::with_capacity(memory_pool_size))),
            lru_order: Vec::new(),
        }
    }

    pub fn load_chunk(&mut self, chunk_id: usize) -> Result<&OutOfCoreChunk, ImputationError> {
        // Check if chunk is already in memory
        let chunk_exists = self
            .chunks
            .get(&chunk_id)
            .map(|c| c.in_memory)
            .unwrap_or(false);

        if chunk_exists {
            // Update access time and LRU order
            if let Some(chunk) = self.chunks.get_mut(&chunk_id) {
                chunk.last_accessed = Instant::now();
            }
            self.update_lru_order(chunk_id);
            return Ok(self.chunks.get(&chunk_id).unwrap());
        }

        // Evict chunks if memory limit exceeded
        self.evict_if_necessary()?;

        // Load chunk from disk
        self.load_chunk_from_disk(chunk_id)?;

        Ok(self.chunks.get(&chunk_id).unwrap())
    }

    fn update_lru_order(&mut self, chunk_id: usize) {
        // Remove chunk_id if it exists and add it to the end
        self.lru_order.retain(|&id| id != chunk_id);
        self.lru_order.push(chunk_id);
    }

    fn evict_if_necessary(&mut self) -> Result<(), ImputationError> {
        while self.memory_usage > self.config.max_memory_usage && !self.lru_order.is_empty() {
            let chunk_id = self.lru_order.remove(0);

            // Check if chunk needs writing
            let (should_write, mem_to_free) = if let Some(chunk) = self.chunks.get(&chunk_id) {
                let should_write = chunk.in_memory && chunk.dirty;
                let mem = chunk
                    .data
                    .as_ref()
                    .map(|d| d.len() * std::mem::size_of::<f64>())
                    .unwrap_or(0);
                (should_write, mem)
            } else {
                (false, 0)
            };

            // Write if needed
            if should_write {
                if let Some(chunk) = self.chunks.get(&chunk_id) {
                    self.write_chunk_to_disk(chunk)?;
                }
            }

            // Now modify the chunk
            if let Some(chunk) = self.chunks.get_mut(&chunk_id) {
                if chunk.in_memory {
                    chunk.dirty = false;
                    chunk.data = None;
                    chunk.in_memory = false;
                    self.memory_usage -= mem_to_free;
                }
            }
        }
        Ok(())
    }

    fn load_chunk_from_disk(&mut self, chunk_id: usize) -> Result<(), ImputationError> {
        // Implementation would load chunk data from disk
        // For now, we'll create a placeholder implementation
        Err(ImputationError::ProcessingError(
            "Chunk loading not implemented".to_string(),
        ))
    }

    fn write_chunk_to_disk(&self, chunk: &OutOfCoreChunk) -> Result<(), ImputationError> {
        // Implementation would write chunk data to disk
        // For now, we'll create a placeholder implementation
        Ok(())
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
    fn test_out_of_core_simple_imputer_in_memory() {
        let X = array![[1.0, 2.0, 3.0], [4.0, f64::NAN, 6.0], [7.0, 8.0, 9.0]];

        let imputer = OutOfCoreSimpleImputer::new()
            .strategy("mean".to_string())
            .max_memory_usage(1_000_000_000); // Large enough to process in memory

        let fitted = imputer.fit(&X.view(), &()).unwrap();
        let X_imputed = fitted.transform(&X.view()).unwrap();

        // Check that NaN was replaced with mean of column (2.0 + 8.0) / 2 = 5.0
        assert_abs_diff_eq!(X_imputed[[1, 1]], 5.0, epsilon = 1e-10);
        assert_abs_diff_eq!(X_imputed[[0, 0]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(X_imputed[[2, 2]], 9.0, epsilon = 1e-10);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_out_of_core_simple_imputer_median() {
        let X = array![[1.0, 2.0], [3.0, f64::NAN], [5.0, 8.0], [7.0, 10.0]];

        let imputer = OutOfCoreSimpleImputer::new().strategy("median".to_string());

        let fitted = imputer.fit(&X.view(), &()).unwrap();
        let X_imputed = fitted.transform(&X.view()).unwrap();

        // Median of [2.0, 8.0, 10.0] = 8.0
        assert_abs_diff_eq!(X_imputed[[1, 1]], 8.0, epsilon = 1e-10);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_out_of_core_knn_imputer() {
        let X = array![
            [1.0, 2.0, 3.0],
            [4.0, f64::NAN, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0]
        ];

        let imputer = OutOfCoreKNNImputer::new()
            .n_neighbors(2)
            .weights("uniform".to_string())
            .max_memory_usage(1_000_000_000); // Process in memory for test

        let fitted = imputer.fit(&X.view(), &()).unwrap();
        let X_imputed = fitted.transform(&X.view()).unwrap();

        // Verify that missing value was imputed
        assert!(!X_imputed[[1, 1]].is_nan());
        assert_abs_diff_eq!(X_imputed[[0, 0]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(X_imputed[[2, 2]], 9.0, epsilon = 1e-10);
    }

    #[test]
    fn test_out_of_core_config() {
        let config = OutOfCoreConfig {
            max_memory_usage: 500_000_000,
            chunk_size: 50_000,
            temp_dir: PathBuf::from("/tmp/test"),
            ..Default::default()
        };

        let imputer = OutOfCoreSimpleImputer::new().out_of_core_config(config.clone());

        assert_eq!(imputer.config.max_memory_usage, 500_000_000);
        assert_eq!(imputer.config.chunk_size, 50_000);
        assert_eq!(imputer.config.temp_dir, PathBuf::from("/tmp/test"));
    }
}
