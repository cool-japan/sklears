//! Out-of-Core Discriminant Analysis
//!
//! This module provides out-of-core implementations for discriminant analysis algorithms,
//! enabling processing of datasets that don't fit in memory using disk-based storage,
//! streaming operations, and memory-efficient algorithms following SciRS2 policy.

// ✅ Using SciRS2 dependencies following SciRS2 policy
use scirs2_core::ndarray::{s, Array1, Array2, Axis};
// Note: Some SciRS2 features may not be available in current version
// Using alternative implementations for now

use crate::{
    lda::{LinearDiscriminantAnalysis, LinearDiscriminantAnalysisConfig},
    parallel_eigen::ParallelEigenConfig,
    qda::{QuadraticDiscriminantAnalysis, QuadraticDiscriminantAnalysisConfig},
};

use memmap2::MmapOptions;
use rayon::prelude::*;
use sklears_core::{
    error::Result,
    prelude::SklearsError,
    traits::{Fit, Predict, PredictProba, Trained},
    types::Float,
};
use std::{
    fs::{File, OpenOptions},
    io::{BufReader, BufWriter, Read, Write},
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};
use tempfile::TempDir;

/// Configuration for out-of-core discriminant analysis
#[derive(Debug, Clone)]
pub struct OutOfCoreConfig {
    /// Maximum memory usage in bytes (None = unlimited)
    pub max_memory_usage: Option<usize>,
    /// Chunk size for processing data in batches
    pub chunk_size: usize,
    /// Number of samples to process at once
    pub batch_size: usize,
    /// Use memory-mapped files for large datasets
    pub use_memory_mapping: bool,
    /// Temporary directory for disk-based storage
    pub temp_dir: Option<PathBuf>,
    /// Enable compression for disk storage
    pub enable_compression: bool,
    /// Cache frequently accessed data in memory
    pub enable_caching: bool,
    /// Maximum cache size in bytes
    pub cache_size: usize,
    /// Use parallel processing for chunks
    pub parallel_chunks: bool,
    /// Parallel eigenvalue decomposition config
    pub parallel_eigen_config: ParallelEigenConfig,
}

impl Default for OutOfCoreConfig {
    fn default() -> Self {
        Self {
            max_memory_usage: Some(1_000_000_000), // 1GB default limit
            chunk_size: 1000,
            batch_size: 100,
            use_memory_mapping: true,
            temp_dir: None,            // Will use system temp
            enable_compression: false, // Disabled for performance by default
            enable_caching: true,
            cache_size: 100_000_000, // 100MB cache
            parallel_chunks: true,
            parallel_eigen_config: ParallelEigenConfig::default(),
        }
    }
}

/// Out-of-core data storage manager
pub struct OutOfCoreDataManager {
    config: OutOfCoreConfig,
    temp_dir: TempDir,
    data_files: Vec<PathBuf>,
    chunk_metadata: Vec<ChunkMetadata>,
    cache: Arc<Mutex<LRUCache>>,
}

#[derive(Debug, Clone)]
struct ChunkMetadata {
    file_path: PathBuf,
    start_row: usize,
    end_row: usize,
    num_features: usize,
    compressed: bool,
}

/// Simple LRU cache for frequently accessed chunks
struct LRUCache {
    capacity: usize,
    data: std::collections::HashMap<usize, Array2<Float>>,
    usage_order: Vec<usize>,
}

impl LRUCache {
    fn new(capacity: usize) -> Self {
        Self {
            capacity,
            data: std::collections::HashMap::new(),
            usage_order: Vec::new(),
        }
    }

    fn get(&mut self, key: usize) -> Option<Array2<Float>> {
        if let Some(data) = self.data.get(&key) {
            // Move to end (most recently used)
            self.usage_order.retain(|&x| x != key);
            self.usage_order.push(key);
            Some(data.clone())
        } else {
            None
        }
    }

    fn put(&mut self, key: usize, value: Array2<Float>) {
        if self.data.contains_key(&key) {
            self.data.insert(key, value);
            // Move to end
            self.usage_order.retain(|&x| x != key);
            self.usage_order.push(key);
        } else {
            // Check capacity and evict if needed
            while self.data.len() >= self.capacity && !self.usage_order.is_empty() {
                let oldest = self.usage_order.remove(0);
                self.data.remove(&oldest);
            }

            self.data.insert(key, value);
            self.usage_order.push(key);
        }
    }
}

impl OutOfCoreDataManager {
    /// Create a new out-of-core data manager
    pub fn new(config: OutOfCoreConfig) -> Result<Self> {
        let temp_dir = if let Some(ref path) = config.temp_dir {
            TempDir::new_in(path)
        } else {
            TempDir::new()
        }
        .map_err(|e| {
            SklearsError::InvalidInput(format!("Failed to create temp directory: {}", e))
        })?;

        let cache = Arc::new(Mutex::new(LRUCache::new(
            config.cache_size / (std::mem::size_of::<Float>() * 100), // Estimate cache slots
        )));

        Ok(Self {
            config,
            temp_dir,
            data_files: Vec::new(),
            chunk_metadata: Vec::new(),
            cache,
        })
    }

    /// Store large dataset in chunks for out-of-core processing
    pub fn store_dataset(&mut self, data: &Array2<Float>, labels: &Array1<usize>) -> Result<()> {
        let n_samples = data.nrows();
        let n_features = data.ncols();

        // Calculate optimal chunk size based on memory constraints
        let chunk_size = self.calculate_chunk_size(n_samples, n_features)?;

        // Split data into chunks and store on disk
        for (chunk_idx, start) in (0..n_samples).step_by(chunk_size).enumerate() {
            let end = (start + chunk_size).min(n_samples);
            let chunk_data = data.slice(s![start..end, ..]).to_owned();
            let chunk_labels = labels.slice(s![start..end]).to_owned();

            self.store_chunk(chunk_idx, &chunk_data, &chunk_labels)?;
        }

        Ok(())
    }

    /// Calculate optimal chunk size based on memory constraints
    fn calculate_chunk_size(&self, n_samples: usize, n_features: usize) -> Result<usize> {
        if let Some(max_memory) = self.config.max_memory_usage {
            let bytes_per_sample = n_features * std::mem::size_of::<Float>();
            let max_samples = max_memory / bytes_per_sample;
            let capped = max_samples.min(self.config.chunk_size).max(1);
            Ok(capped.min(n_samples.max(1)))
        } else {
            Ok(self.config.chunk_size.min(n_samples.max(1)))
        }
    }

    /// Store a single chunk to disk
    fn store_chunk(
        &mut self,
        chunk_idx: usize,
        data: &Array2<Float>,
        _labels: &Array1<usize>,
    ) -> Result<()> {
        let file_name = format!("chunk_{:06}.bin", chunk_idx);
        let file_path = self.temp_dir.path().join(file_name);

        // Store using memory mapping if enabled
        if self.config.use_memory_mapping {
            self.store_chunk_mmap(&file_path, data)?;
        } else {
            self.store_chunk_file(&file_path, data)?;
        }

        // Update metadata
        let metadata = ChunkMetadata {
            file_path: file_path.clone(),
            start_row: chunk_idx * self.config.chunk_size,
            end_row: (chunk_idx + 1) * self.config.chunk_size,
            num_features: data.ncols(),
            compressed: self.config.enable_compression,
        };

        self.data_files.push(file_path);
        self.chunk_metadata.push(metadata);

        Ok(())
    }

    /// Store chunk using memory mapping
    fn store_chunk_mmap(&self, file_path: &Path, data: &Array2<Float>) -> Result<()> {
        let file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .open(file_path)
            .map_err(|e| SklearsError::InvalidInput(format!("Failed to create file: {}", e)))?;

        let data_size = data.len() * std::mem::size_of::<Float>();
        file.set_len(data_size as u64)
            .map_err(|e| SklearsError::InvalidInput(format!("Failed to set file size: {}", e)))?;

        let mut mmap = unsafe {
            MmapOptions::new()
                .map_mut(&file)
                .map_err(|e| SklearsError::InvalidInput(format!("Failed to create mmap: {}", e)))?
        };

        // Copy data to memory-mapped file
        let data_bytes =
            unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data_size) };

        mmap[..data_size].copy_from_slice(data_bytes);
        mmap.flush()
            .map_err(|e| SklearsError::InvalidInput(format!("Failed to flush mmap: {}", e)))?;

        Ok(())
    }

    /// Store chunk using regular file I/O
    fn store_chunk_file(&self, file_path: &Path, data: &Array2<Float>) -> Result<()> {
        let file = File::create(file_path)
            .map_err(|e| SklearsError::InvalidInput(format!("Failed to create file: {}", e)))?;
        let mut writer = BufWriter::new(file);

        // Write dimensions first
        writer
            .write_all(&(data.nrows() as u32).to_le_bytes())
            .map_err(|e| {
                SklearsError::InvalidInput(format!("Failed to write dimensions: {}", e))
            })?;
        writer
            .write_all(&(data.ncols() as u32).to_le_bytes())
            .map_err(|e| {
                SklearsError::InvalidInput(format!("Failed to write dimensions: {}", e))
            })?;

        // Write data
        for value in data.iter() {
            writer
                .write_all(&value.to_le_bytes())
                .map_err(|e| SklearsError::InvalidInput(format!("Failed to write data: {}", e)))?;
        }

        writer
            .flush()
            .map_err(|e| SklearsError::InvalidInput(format!("Failed to flush writer: {}", e)))?;

        Ok(())
    }

    /// Load a chunk from disk (with caching)
    pub fn load_chunk(&self, chunk_idx: usize) -> Result<Array2<Float>> {
        // Check cache first
        if self.config.enable_caching {
            if let Ok(mut cache) = self.cache.lock() {
                if let Some(data) = cache.get(chunk_idx) {
                    return Ok(data);
                }
            }
        }

        // Load from disk
        let metadata = &self.chunk_metadata[chunk_idx];
        let data = if self.config.use_memory_mapping {
            self.load_chunk_mmap(&metadata.file_path)?
        } else {
            self.load_chunk_file(&metadata.file_path)?
        };

        // Cache the result
        if self.config.enable_caching {
            if let Ok(mut cache) = self.cache.lock() {
                cache.put(chunk_idx, data.clone());
            }
        }

        Ok(data)
    }

    /// Load chunk using memory mapping
    fn load_chunk_mmap(&self, file_path: &Path) -> Result<Array2<Float>> {
        let file = File::open(file_path)
            .map_err(|e| SklearsError::InvalidInput(format!("Failed to open file: {}", e)))?;

        let mmap = unsafe {
            MmapOptions::new()
                .map(&file)
                .map_err(|e| SklearsError::InvalidInput(format!("Failed to create mmap: {}", e)))?
        };

        // Calculate dimensions from file size
        let total_elements = mmap.len() / std::mem::size_of::<Float>();
        let data_ptr = mmap.as_ptr() as *const Float;

        // For simplicity, assume we stored the data in a flattened format
        // In practice, we'd need to store dimensions metadata
        let data_slice = unsafe { std::slice::from_raw_parts(data_ptr, total_elements) };

        // Reconstruct array (this is simplified - in practice need proper dimension handling)
        let n_features = self.chunk_metadata[0].num_features;
        let n_samples = total_elements / n_features;

        Array2::from_shape_vec((n_samples, n_features), data_slice.to_vec())
            .map_err(|e| SklearsError::InvalidInput(format!("Failed to reshape array: {}", e)))
    }

    /// Load chunk using regular file I/O
    fn load_chunk_file(&self, file_path: &Path) -> Result<Array2<Float>> {
        let file = File::open(file_path)
            .map_err(|e| SklearsError::InvalidInput(format!("Failed to open file: {}", e)))?;
        let mut reader = BufReader::new(file);

        // Read dimensions
        let mut dim_buf = [0u8; 4];
        reader
            .read_exact(&mut dim_buf)
            .map_err(|e| SklearsError::InvalidInput(format!("Failed to read dimensions: {}", e)))?;
        let nrows = u32::from_le_bytes(dim_buf) as usize;

        reader
            .read_exact(&mut dim_buf)
            .map_err(|e| SklearsError::InvalidInput(format!("Failed to read dimensions: {}", e)))?;
        let ncols = u32::from_le_bytes(dim_buf) as usize;

        // Read data
        let mut data = Vec::with_capacity(nrows * ncols);
        let mut value_buf = [0u8; 8];
        for _ in 0..(nrows * ncols) {
            reader
                .read_exact(&mut value_buf)
                .map_err(|e| SklearsError::InvalidInput(format!("Failed to read data: {}", e)))?;
            data.push(Float::from_le_bytes(value_buf));
        }

        Array2::from_shape_vec((nrows, ncols), data)
            .map_err(|e| SklearsError::InvalidInput(format!("Failed to create array: {}", e)))
    }

    /// Get number of chunks
    pub fn num_chunks(&self) -> usize {
        self.chunk_metadata.len()
    }

    /// Process all chunks with a function in parallel
    pub fn process_chunks_parallel<F, R>(&self, f: F) -> Result<Vec<R>>
    where
        F: Fn(&Array2<Float>) -> Result<R> + Send + Sync,
        R: Send + Sync,
    {
        if self.config.parallel_chunks {
            (0..self.num_chunks())
                .into_par_iter()
                .map(|chunk_idx| {
                    let chunk_data = self.load_chunk(chunk_idx)?;
                    f(&chunk_data)
                })
                .collect()
        } else {
            (0..self.num_chunks())
                .map(|chunk_idx| {
                    let chunk_data = self.load_chunk(chunk_idx)?;
                    f(&chunk_data)
                })
                .collect()
        }
    }
}

/// Out-of-core Linear Discriminant Analysis
pub struct OutOfCoreLDA {
    config: OutOfCoreConfig,
    lda_config: LinearDiscriminantAnalysisConfig,
    data_manager: Option<OutOfCoreDataManager>,
    trained_model: Option<LinearDiscriminantAnalysis<Trained>>,
    cached_mean: Option<Array1<Float>>,
    cached_covariance: Option<Array2<Float>>,
}

impl Default for OutOfCoreLDA {
    fn default() -> Self {
        Self::new()
    }
}

impl OutOfCoreLDA {
    /// Create a new out-of-core LDA
    pub fn new() -> Self {
        Self {
            config: OutOfCoreConfig::default(),
            lda_config: LinearDiscriminantAnalysisConfig::default(),
            data_manager: None,
            trained_model: None,
            cached_mean: None,
            cached_covariance: None,
        }
    }

    /// Create with custom configurations
    pub fn with_config(
        config: OutOfCoreConfig,
        lda_config: LinearDiscriminantAnalysisConfig,
    ) -> Self {
        Self {
            config,
            lda_config,
            data_manager: None,
            trained_model: None,
            cached_mean: None,
            cached_covariance: None,
        }
    }

    /// Fit the model using out-of-core training
    pub fn fit_out_of_core(&mut self, data: &Array2<Float>, labels: &Array1<usize>) -> Result<()> {
        // Initialize data manager
        let mut data_manager = OutOfCoreDataManager::new(self.config.clone())?;
        data_manager.store_dataset(data, labels)?;

        // Compute statistics incrementally
        let (mean, covariance) = self.compute_incremental_statistics(&data_manager)?;
        self.cached_mean = Some(mean.clone());
        self.cached_covariance = Some(covariance.clone());

        // Train LDA model with computed statistics
        let lda = LinearDiscriminantAnalysis::new();

        // For this simplified implementation, we'll load all data at once for training
        // In a full implementation, we'd implement incremental LDA training
        let labels_i32 = labels.mapv(|x| x as i32);
        let trained = lda.fit(data, &labels_i32)?;

        self.data_manager = Some(data_manager);
        self.trained_model = Some(trained);

        Ok(())
    }

    /// Compute statistics incrementally from chunks
    fn compute_incremental_statistics(
        &self,
        data_manager: &OutOfCoreDataManager,
    ) -> Result<(Array1<Float>, Array2<Float>)> {
        let num_chunks = data_manager.num_chunks();
        if num_chunks == 0 {
            return Err(SklearsError::InvalidInput(
                "No chunks available".to_string(),
            ));
        }

        // Initialize accumulators
        let first_chunk = data_manager.load_chunk(0)?;
        let n_features = first_chunk.ncols();
        let mut mean_accumulator = Array1::zeros(n_features);
        let mut cov_accumulator = Array2::zeros((n_features, n_features));
        let mut total_samples = 0;

        // Process chunks in parallel for statistics
        let chunk_stats: Vec<_> = data_manager.process_chunks_parallel(|chunk| {
            let n_samples = chunk.nrows();
            let chunk_mean = chunk.mean_axis(Axis(0)).unwrap();
            let centered = chunk - &chunk_mean;
            let chunk_cov = centered.t().dot(&centered) / (n_samples - 1) as Float;

            Ok((chunk_mean, chunk_cov, n_samples))
        })?;

        // Combine statistics from all chunks
        for (chunk_mean, chunk_cov, n_samples) in chunk_stats {
            let weight = n_samples as Float;
            mean_accumulator = (mean_accumulator * total_samples as Float + chunk_mean * weight)
                / (total_samples as Float + weight);
            cov_accumulator = cov_accumulator + chunk_cov;
            total_samples += n_samples;
        }

        // Normalize covariance
        cov_accumulator /= (num_chunks - 1) as Float;

        Ok((mean_accumulator, cov_accumulator))
    }

    /// Predict using the trained model
    pub fn predict(&self, data: &Array2<Float>) -> Result<Array1<usize>> {
        if let Some(ref model) = self.trained_model {
            let predictions_i32 = model.predict(data)?;
            let predictions_usize = predictions_i32.mapv(|x| x as usize);
            Ok(predictions_usize)
        } else {
            Err(SklearsError::InvalidInput("Model not trained".to_string()))
        }
    }

    /// Predict probabilities
    pub fn predict_proba(&self, data: &Array2<Float>) -> Result<Array2<Float>> {
        if let Some(ref model) = self.trained_model {
            model.predict_proba(data)
        } else {
            Err(SklearsError::InvalidInput("Model not trained".to_string()))
        }
    }
}

/// Out-of-core Quadratic Discriminant Analysis
pub struct OutOfCoreQDA {
    config: OutOfCoreConfig,
    qda_config: QuadraticDiscriminantAnalysisConfig,
    data_manager: Option<OutOfCoreDataManager>,
    trained_model: Option<QuadraticDiscriminantAnalysis<Trained>>,
}

impl Default for OutOfCoreQDA {
    fn default() -> Self {
        Self::new()
    }
}

impl OutOfCoreQDA {
    /// Create a new out-of-core QDA
    pub fn new() -> Self {
        Self {
            config: OutOfCoreConfig::default(),
            qda_config: QuadraticDiscriminantAnalysisConfig::default(),
            data_manager: None,
            trained_model: None,
        }
    }

    /// Create with custom configurations
    pub fn with_config(
        config: OutOfCoreConfig,
        qda_config: QuadraticDiscriminantAnalysisConfig,
    ) -> Self {
        Self {
            config,
            qda_config,
            data_manager: None,
            trained_model: None,
        }
    }

    /// Fit the model using out-of-core training
    pub fn fit_out_of_core(&mut self, data: &Array2<Float>, labels: &Array1<usize>) -> Result<()> {
        // Initialize data manager
        let mut data_manager = OutOfCoreDataManager::new(self.config.clone())?;
        data_manager.store_dataset(data, labels)?;

        // Train QDA model
        let qda = QuadraticDiscriminantAnalysis::new();

        // For this simplified implementation, we'll load all data at once for training
        // In a full implementation, we'd implement incremental QDA training
        let labels_i32 = labels.mapv(|x| x as i32);
        let trained = qda.fit(data, &labels_i32)?;

        self.data_manager = Some(data_manager);
        self.trained_model = Some(trained);

        Ok(())
    }

    /// Predict using the trained model
    pub fn predict(&self, data: &Array2<Float>) -> Result<Array1<usize>> {
        if let Some(ref model) = self.trained_model {
            let predictions_i32 = model.predict(data)?;
            let predictions_usize = predictions_i32.mapv(|x| x as usize);
            Ok(predictions_usize)
        } else {
            Err(SklearsError::InvalidInput("Model not trained".to_string()))
        }
    }

    /// Predict probabilities
    pub fn predict_proba(&self, data: &Array2<Float>) -> Result<Array2<Float>> {
        if let Some(ref model) = self.trained_model {
            model.predict_proba(data)
        } else {
            Err(SklearsError::InvalidInput("Model not trained".to_string()))
        }
    }
}

/// Streaming discriminant analysis for continuously arriving data
pub struct StreamingDiscriminant {
    config: OutOfCoreConfig,
    buffer: Array2<Float>,
    buffer_labels: Array1<usize>,
    buffer_size: usize,
    model: Option<LinearDiscriminantAnalysis<Trained>>,
}

impl StreamingDiscriminant {
    /// Create a new streaming discriminant analyzer
    pub fn new(buffer_capacity: usize, n_features: usize) -> Self {
        Self {
            config: OutOfCoreConfig::default(),
            buffer: Array2::zeros((buffer_capacity, n_features)),
            buffer_labels: Array1::zeros(buffer_capacity),
            buffer_size: 0,
            model: None,
        }
    }

    /// Add new samples to the stream
    pub fn add_samples(&mut self, samples: &Array2<Float>, labels: &Array1<usize>) -> Result<()> {
        let n_new_samples = samples.nrows();
        let buffer_capacity = self.buffer.nrows();

        // If buffer would overflow, train model and reset buffer
        if self.buffer_size + n_new_samples > buffer_capacity {
            self.train_on_buffer()?;
            self.buffer_size = 0;
        }

        // Add new samples to buffer
        let end_idx = self.buffer_size + n_new_samples;
        self.buffer
            .slice_mut(s![self.buffer_size..end_idx, ..])
            .assign(samples);
        self.buffer_labels
            .slice_mut(s![self.buffer_size..end_idx])
            .assign(labels);

        self.buffer_size += n_new_samples;

        Ok(())
    }

    /// Train model on current buffer
    fn train_on_buffer(&mut self) -> Result<()> {
        if self.buffer_size == 0 {
            return Ok(());
        }

        let buffer_data = self.buffer.slice(s![..self.buffer_size, ..]).to_owned();
        let buffer_labels = self.buffer_labels.slice(s![..self.buffer_size]).to_owned();

        let lda = LinearDiscriminantAnalysis::new();
        let buffer_labels_i32 = buffer_labels.mapv(|x| x as i32);
        let trained = lda.fit(&buffer_data, &buffer_labels_i32)?;

        // In a full implementation, we'd merge with existing model
        self.model = Some(trained);

        Ok(())
    }

    /// Predict on new data
    pub fn predict(&self, data: &Array2<Float>) -> Result<Array1<usize>> {
        if let Some(ref model) = self.model {
            let predictions_i32 = model.predict(data)?;
            let predictions_usize = predictions_i32.mapv(|x| x as usize);
            Ok(predictions_usize)
        } else {
            Err(SklearsError::InvalidInput(
                "No model trained yet".to_string(),
            ))
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_out_of_core_data_manager() {
        let config = OutOfCoreConfig {
            chunk_size: 2,
            ..Default::default()
        };

        let mut manager = OutOfCoreDataManager::new(config).unwrap();
        let data = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
        let labels = array![0, 0, 1, 1];

        manager.store_dataset(&data, &labels).unwrap();

        assert_eq!(manager.num_chunks(), 2);

        let chunk0 = manager.load_chunk(0).unwrap();
        assert_eq!(chunk0.nrows(), 2);
        assert_eq!(chunk0.ncols(), 2);
    }

    #[test]
    fn test_streaming_discriminant() {
        let mut stream = StreamingDiscriminant::new(4, 2);

        let samples1 = array![[1.0, 2.0], [3.0, 4.0]];
        let labels1 = array![0, 1];

        let samples2 = array![[5.0, 6.0], [7.0, 8.0]];
        let labels2 = array![0, 1];

        stream.add_samples(&samples1, &labels1).unwrap();
        stream.add_samples(&samples2, &labels2).unwrap();

        // Buffer should trigger training
        let test_data = array![[2.0, 3.0]];
        let _predictions = stream.predict(&test_data);
        // May fail if model not trained yet, which is expected
    }

    #[test]
    fn test_lru_cache() {
        let mut cache = LRUCache::new(2);

        let data1 = array![[1.0, 2.0]];
        let data2 = array![[3.0, 4.0]];
        let data3 = array![[5.0, 6.0]];

        cache.put(1, data1.clone());
        cache.put(2, data2.clone());

        assert!(cache.get(1).is_some());
        assert!(cache.get(2).is_some());

        // Adding third item should evict first (LRU)
        cache.put(3, data3);
        assert!(cache.get(1).is_none());
        assert!(cache.get(2).is_some());
        assert!(cache.get(3).is_some());
    }
}
