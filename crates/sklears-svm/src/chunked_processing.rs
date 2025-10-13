//! Chunked processing for large-scale SVM training
//!
//! This module provides chunked processing capabilities for handling datasets
//! that are too large to fit in memory at once. It implements efficient
//! strategies for loading, processing, and managing data chunks during SVM training.

use crate::kernels::Kernel;
use scirs2_core::ndarray::{s, Array1, Array2, Axis};
use sklears_core::{
    error::{Result, SklearsError},
    types::Float,
};
use std::collections::VecDeque;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

/// Configuration for chunked processing
#[derive(Debug, Clone)]
pub struct ChunkedProcessingConfig {
    /// Maximum chunk size in number of samples
    pub max_chunk_size: usize,
    /// Maximum memory usage in MB
    pub max_memory_mb: usize,
    /// Number of chunks to keep in memory
    pub cache_chunks: usize,
    /// Temporary directory for storing chunks
    pub temp_dir: Option<PathBuf>,
    /// Overlap size between chunks for continuity
    pub chunk_overlap: usize,
    /// Compression level for stored chunks (0-9)
    pub compression_level: u8,
}

impl Default for ChunkedProcessingConfig {
    fn default() -> Self {
        Self {
            max_chunk_size: 10000,
            max_memory_mb: 1024, // 1GB
            cache_chunks: 3,
            temp_dir: None,
            chunk_overlap: 100,
            compression_level: 6,
        }
    }
}

/// Chunked dataset manager
pub struct ChunkedDataset {
    config: ChunkedProcessingConfig,
    chunks: Vec<DataChunk>,
    cached_chunks: VecDeque<(usize, CachedChunk)>,
    temp_files: Vec<PathBuf>,
    total_samples: usize,
    n_features: usize,
}

/// Information about a data chunk
#[derive(Debug, Clone)]
struct DataChunk {
    /// Chunk identifier
    id: usize,
    /// Start index in the original dataset
    start_idx: usize,
    /// End index in the original dataset
    end_idx: usize,
    /// Number of samples in this chunk
    n_samples: usize,
    /// File path if stored on disk
    file_path: Option<PathBuf>,
    /// Whether chunk is currently in memory
    in_memory: bool,
}

/// Cached chunk data
#[derive(Debug, Clone)]
struct CachedChunk {
    x: Array2<Float>,
    y: Array1<Float>,
    last_accessed: std::time::Instant,
}

impl ChunkedDataset {
    /// Create a new chunked dataset from arrays
    pub fn from_arrays(
        x: &Array2<Float>,
        y: &Array1<Float>,
        config: ChunkedProcessingConfig,
    ) -> Result<Self> {
        let total_samples = x.nrows();
        let n_features = x.ncols();

        if total_samples != y.len() {
            return Err(SklearsError::InvalidInput(
                "X and y must have the same number of samples".to_string(),
            ));
        }

        let chunk_size = config.max_chunk_size.min(total_samples);
        let mut chunks = Vec::new();
        let mut chunk_id = 0;

        // Create chunks
        let mut start = 0;
        while start < total_samples {
            let end = (start + chunk_size).min(total_samples);

            chunks.push(DataChunk {
                id: chunk_id,
                start_idx: start,
                end_idx: end,
                n_samples: end - start,
                file_path: None,
                in_memory: false,
            });

            start = end - config.chunk_overlap.min(end - start);
            chunk_id += 1;
        }

        let mut dataset = Self {
            config,
            chunks,
            cached_chunks: VecDeque::new(),
            temp_files: Vec::new(),
            total_samples,
            n_features,
        };

        // Store chunks to disk if needed
        dataset.store_chunks_to_disk(x, y)?;

        Ok(dataset)
    }

    /// Create chunked dataset from files
    pub fn from_files(data_files: Vec<PathBuf>, config: ChunkedProcessingConfig) -> Result<Self> {
        // This would implement loading from multiple files
        // For now, return a placeholder
        Err(SklearsError::InvalidInput(
            "File-based chunked loading not yet implemented".to_string(),
        ))
    }

    /// Store chunks to disk
    fn store_chunks_to_disk(&mut self, x: &Array2<Float>, y: &Array1<Float>) -> Result<()> {
        let temp_dir = self
            .config
            .temp_dir
            .clone()
            .unwrap_or_else(std::env::temp_dir);

        // Collect chunk information first to avoid borrowing conflicts
        let chunk_info: Vec<(usize, usize, usize)> = self
            .chunks
            .iter()
            .map(|chunk| (chunk.id, chunk.start_idx, chunk.end_idx))
            .collect();

        for (i, (chunk_id, start_idx, end_idx)) in chunk_info.into_iter().enumerate() {
            let chunk_x = x.slice(s![start_idx..end_idx, ..]);
            let chunk_y = y.slice(s![start_idx..end_idx]);

            let file_path = temp_dir.join(format!("chunk_{chunk_id}.bin"));
            self.serialize_chunk(&chunk_x.to_owned(), &chunk_y.to_owned(), &file_path)?;

            self.chunks[i].file_path = Some(file_path.clone());
            self.temp_files.push(file_path);
        }

        Ok(())
    }

    /// Serialize chunk to disk
    fn serialize_chunk(
        &self,
        x: &Array2<Float>,
        y: &Array1<Float>,
        file_path: &Path,
    ) -> Result<()> {
        let file = File::create(file_path)
            .map_err(|e| SklearsError::InvalidInput(format!("Failed to create chunk file: {e}")))?;
        let mut writer = BufWriter::new(file);

        // Write dimensions
        let dims = [x.nrows() as u64, x.ncols() as u64];
        for &dim in &dims {
            writer.write_all(&dim.to_le_bytes()).map_err(|e| {
                SklearsError::InvalidInput(format!("Failed to write dimensions: {e}"))
            })?;
        }

        // Write X data
        for row in x.axis_iter(Axis(0)) {
            for &value in row.iter() {
                writer.write_all(&value.to_le_bytes()).map_err(|e| {
                    SklearsError::InvalidInput(format!("Failed to write X data: {e}"))
                })?;
            }
        }

        // Write y data
        for &value in y.iter() {
            writer
                .write_all(&value.to_le_bytes())
                .map_err(|e| SklearsError::InvalidInput(format!("Failed to write y data: {e}")))?;
        }

        writer
            .flush()
            .map_err(|e| SklearsError::InvalidInput(format!("Failed to flush writer: {e}")))?;

        Ok(())
    }

    /// Deserialize chunk from disk
    fn deserialize_chunk(&self, file_path: &Path) -> Result<(Array2<Float>, Array1<Float>)> {
        let file = File::open(file_path)
            .map_err(|e| SklearsError::InvalidInput(format!("Failed to open chunk file: {e}")))?;
        let mut reader = BufReader::new(file);

        // Read dimensions
        let mut dim_bytes = [0u8; 8];
        reader
            .read_exact(&mut dim_bytes)
            .map_err(|e| SklearsError::InvalidInput(format!("Failed to read rows: {e}")))?;
        let n_rows = u64::from_le_bytes(dim_bytes) as usize;

        reader
            .read_exact(&mut dim_bytes)
            .map_err(|e| SklearsError::InvalidInput(format!("Failed to read cols: {e}")))?;
        let n_cols = u64::from_le_bytes(dim_bytes) as usize;

        // Read X data
        let mut x = Array2::zeros((n_rows, n_cols));
        let mut value_bytes = [0u8; 8]; // Assuming Float is f64
        for mut row in x.axis_iter_mut(Axis(0)) {
            for value in row.iter_mut() {
                reader.read_exact(&mut value_bytes).map_err(|e| {
                    SklearsError::InvalidInput(format!("Failed to read X value: {e}"))
                })?;
                *value = f64::from_le_bytes(value_bytes);
            }
        }

        // Read y data
        let mut y = Array1::zeros(n_rows);
        for value in y.iter_mut() {
            reader
                .read_exact(&mut value_bytes)
                .map_err(|e| SklearsError::InvalidInput(format!("Failed to read y value: {e}")))?;
            *value = f64::from_le_bytes(value_bytes);
        }

        Ok((x, y))
    }

    /// Get a chunk by ID
    pub fn get_chunk(&mut self, chunk_id: usize) -> Result<(&Array2<Float>, &Array1<Float>)> {
        // Check if chunk is already cached
        if let Some(pos) = self
            .cached_chunks
            .iter()
            .position(|(id, _)| *id == chunk_id)
        {
            let (_, chunk) = &mut self.cached_chunks[pos];
            chunk.last_accessed = std::time::Instant::now();
            return Ok((&chunk.x, &chunk.y));
        }

        // Load chunk from disk
        if chunk_id >= self.chunks.len() {
            return Err(SklearsError::InvalidInput(format!(
                "Chunk ID {} out of range",
                chunk_id
            )));
        }

        let chunk_info = &self.chunks[chunk_id];
        let file_path = chunk_info
            .file_path
            .as_ref()
            .ok_or_else(|| SklearsError::InvalidInput("Chunk file path not set".to_string()))?;

        let (x, y) = self.deserialize_chunk(file_path)?;

        // Add to cache
        self.add_to_cache(chunk_id, x, y);

        // Return reference to cached chunk
        let (_, cached_chunk) = self.cached_chunks.back().unwrap();
        Ok((&cached_chunk.x, &cached_chunk.y))
    }

    /// Add chunk to cache, managing cache size
    fn add_to_cache(&mut self, chunk_id: usize, x: Array2<Float>, y: Array1<Float>) {
        let cached_chunk = CachedChunk {
            x,
            y,
            last_accessed: std::time::Instant::now(),
        };

        // Remove oldest chunk if cache is full
        if self.cached_chunks.len() >= self.config.cache_chunks {
            self.cached_chunks.pop_front();
        }

        self.cached_chunks.push_back((chunk_id, cached_chunk));
    }

    /// Get chunk iterator
    pub fn chunk_iter(&mut self) -> ChunkIterator {
        ChunkIterator {
            dataset: self,
            current_chunk: 0,
        }
    }

    /// Get number of chunks
    pub fn n_chunks(&self) -> usize {
        self.chunks.len()
    }

    /// Get total number of samples
    pub fn n_samples(&self) -> usize {
        self.total_samples
    }

    /// Get number of features
    pub fn n_features(&self) -> usize {
        self.n_features
    }

    /// Process chunks with a given function
    pub fn process_chunks<F, R>(&mut self, mut processor: F) -> Result<Vec<R>>
    where
        F: FnMut(usize, &Array2<Float>, &Array1<Float>) -> Result<R>,
    {
        let mut results = Vec::new();

        for chunk_id in 0..self.n_chunks() {
            let (x, y) = self.get_chunk(chunk_id)?;
            let result = processor(chunk_id, x, y)?;
            results.push(result);
        }

        Ok(results)
    }

    /// Compute statistics across all chunks
    pub fn compute_stats(&mut self) -> Result<ChunkedDatasetStats> {
        let mut total_samples = 0;
        let mut sum_x = Array1::zeros(self.n_features);
        let mut sum_y = 0.0;
        let mut sum_x_squared = Array1::zeros(self.n_features);
        let mut sum_y_squared = 0.0;

        for chunk_id in 0..self.n_chunks() {
            let (x, y) = self.get_chunk(chunk_id)?;

            total_samples += x.nrows();

            // Update sums for X
            for row in x.axis_iter(Axis(0)) {
                for (i, &value) in row.iter().enumerate() {
                    sum_x[i] += value;
                    sum_x_squared[i] += value * value;
                }
            }

            // Update sums for y
            for &value in y.iter() {
                sum_y += value;
                sum_y_squared += value * value;
            }
        }

        let n_samples = total_samples as Float;
        let mean_x = &sum_x / n_samples;
        let mean_y = sum_y / n_samples;

        let var_x = (&sum_x_squared / n_samples) - &mean_x * &mean_x;
        let var_y = (sum_y_squared / n_samples) - mean_y * mean_y;

        Ok(ChunkedDatasetStats {
            n_samples: total_samples,
            n_features: self.n_features,
            mean_x,
            mean_y,
            var_x,
            var_y,
        })
    }
}

/// Iterator over chunks
pub struct ChunkIterator<'life> {
    dataset: &'life mut ChunkedDataset,
    current_chunk: usize,
}

impl<'life> ChunkIterator<'life> {
    /// Get next chunk
    pub fn next(&mut self) -> Option<Result<(usize, &Array2<Float>, &Array1<Float>)>> {
        if self.current_chunk >= self.dataset.n_chunks() {
            return None;
        }

        let chunk_id = self.current_chunk;
        self.current_chunk += 1;

        match self.dataset.get_chunk(chunk_id) {
            Ok((x, y)) => Some(Ok((chunk_id, x, y))),
            Err(e) => Some(Err(e)),
        }
    }
}

/// Statistics computed across chunked dataset
#[derive(Debug, Clone)]
pub struct ChunkedDatasetStats {
    pub n_samples: usize,
    pub n_features: usize,
    pub mean_x: Array1<Float>,
    pub mean_y: Float,
    pub var_x: Array1<Float>,
    pub var_y: Float,
}

/// Chunked SVM trainer that works with large datasets
pub struct ChunkedSvmTrainer<K: Kernel> {
    kernel: K,
    config: ChunkedProcessingConfig,
    dataset: Option<ChunkedDataset>,
}

impl<K: Kernel> ChunkedSvmTrainer<K> {
    /// Create new chunked SVM trainer
    pub fn new(kernel: K, config: ChunkedProcessingConfig) -> Self {
        Self {
            kernel,
            config,
            dataset: None,
        }
    }

    /// Set dataset
    pub fn set_dataset(&mut self, dataset: ChunkedDataset) {
        self.dataset = Some(dataset);
    }

    /// Train SVM using chunked processing
    pub fn train(&mut self, c: Float, tol: Float, max_iter: usize) -> Result<ChunkedSvmResult> {
        let dataset = self
            .dataset
            .as_mut()
            .ok_or_else(|| SklearsError::InvalidInput("Dataset not set".to_string()))?;

        let n_samples = dataset.n_samples();
        let mut alpha = Array1::zeros(n_samples);
        let mut global_gradient = Array1::zeros(n_samples);

        let mut iteration = 0;
        let mut convergence_history = Vec::new();

        while iteration < max_iter {
            let mut max_violation: Float = 0.0;
            let mut updates_made = 0;

            // Process each chunk
            for chunk_id in 0..dataset.n_chunks() {
                // Get chunk bounds first
                let chunk_start = dataset.chunks[chunk_id].start_idx;
                let chunk_end = dataset.chunks[chunk_id].end_idx;

                let (chunk_x, chunk_y) = dataset.get_chunk(chunk_id)?;

                let chunk_alpha = alpha.slice_mut(s![chunk_start..chunk_end]);
                let chunk_gradient = global_gradient.slice_mut(s![chunk_start..chunk_end]);

                // Simplified SMO-like updates for this chunk - use &self since update_chunk doesn't need &mut self
                let chunk_updates = ChunkedSvmTrainer::<K>::update_chunk_static(
                    &self.kernel,
                    chunk_x,
                    chunk_y,
                    chunk_alpha,
                    chunk_gradient,
                    c,
                    tol,
                )?;

                updates_made += chunk_updates.n_updates;
                max_violation = max_violation.max(chunk_updates.max_violation);
            }

            convergence_history.push(max_violation);

            if max_violation < tol {
                break;
            }

            iteration += 1;
        }

        let n_support_vectors = alpha.iter().filter(|&&a| a > 1e-10).count();

        Ok(ChunkedSvmResult {
            alpha,
            n_iterations: iteration,
            converged: iteration < max_iter,
            convergence_history,
            n_support_vectors,
        })
    }

    /// Update a single chunk
    fn update_chunk(
        &self,
        chunk_x: &Array2<Float>,
        chunk_y: &Array1<Float>,
        chunk_alpha: scirs2_core::ndarray::ArrayViewMut1<Float>,
        chunk_gradient: scirs2_core::ndarray::ArrayViewMut1<Float>,
        c: Float,
        tol: Float,
    ) -> Result<ChunkUpdateResult> {
        Self::update_chunk_static(
            &self.kernel,
            chunk_x,
            chunk_y,
            chunk_alpha,
            chunk_gradient,
            c,
            tol,
        )
    }

    /// Static version of update_chunk to avoid borrowing conflicts
    fn update_chunk_static<K2: Kernel>(
        kernel: &K2,
        chunk_x: &Array2<Float>,
        chunk_y: &Array1<Float>,
        mut chunk_alpha: scirs2_core::ndarray::ArrayViewMut1<Float>,
        mut chunk_gradient: scirs2_core::ndarray::ArrayViewMut1<Float>,
        c: Float,
        tol: Float,
    ) -> Result<ChunkUpdateResult> {
        let n_samples = chunk_x.nrows();
        let mut n_updates = 0;
        let mut max_violation: Float = 0.0;

        // Simple coordinate descent within chunk
        for i in 0..n_samples {
            let old_alpha = chunk_alpha[i];
            let gradient_i = chunk_gradient[i];

            // Compute kernel diagonal element
            let k_ii = kernel.compute(
                chunk_x.row(i).to_owned().view(),
                chunk_x.row(i).to_owned().view(),
            );

            if k_ii <= 0.0 {
                continue;
            }

            // Update alpha
            let mut new_alpha = old_alpha - gradient_i / k_ii;
            new_alpha = new_alpha.max(0.0).min(c);

            let delta_alpha = new_alpha - old_alpha;

            if delta_alpha.abs() < 1e-12 {
                continue;
            }

            chunk_alpha[i] = new_alpha;
            n_updates += 1;

            // Update gradients within chunk
            for j in 0..n_samples {
                let k_ij = kernel.compute(
                    chunk_x.row(i).to_owned().view(),
                    chunk_x.row(j).to_owned().view(),
                );
                chunk_gradient[j] += chunk_y[i] * chunk_y[j] * delta_alpha * k_ij;
            }

            // Compute violation
            let violation = Self::compute_violation_static(new_alpha, gradient_i, chunk_y[i], c);
            max_violation = max_violation.max(violation);
        }

        Ok(ChunkUpdateResult {
            n_updates,
            max_violation,
        })
    }

    /// Compute KKT violation
    fn compute_violation(&self, alpha: Float, gradient: Float, y: Float, c: Float) -> Float {
        Self::compute_violation_static(alpha, gradient, y, c)
    }

    /// Static version of compute_violation
    fn compute_violation_static(alpha: Float, gradient: Float, y: Float, c: Float) -> Float {
        if alpha < 1e-10 {
            (-y * gradient).max(0.0)
        } else if alpha > c - 1e-10 {
            (y * gradient).max(0.0)
        } else {
            (y * gradient).abs()
        }
    }
}

/// Result from updating a chunk
#[derive(Debug)]
struct ChunkUpdateResult {
    n_updates: usize,
    max_violation: Float,
}

/// Result from chunked SVM training
#[derive(Debug, Clone)]
pub struct ChunkedSvmResult {
    pub alpha: Array1<Float>,
    pub n_iterations: usize,
    pub converged: bool,
    pub convergence_history: Vec<Float>,
    pub n_support_vectors: usize,
}

impl Drop for ChunkedDataset {
    fn drop(&mut self) {
        // Clean up temporary files
        for file_path in &self.temp_files {
            if file_path.exists() {
                let _ = std::fs::remove_file(file_path);
            }
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::RbfKernel;
    use scirs2_core::ndarray::array;

    #[test]
    #[ignore]
    fn test_chunked_dataset_creation() {
        let x = Array2::from_shape_vec((100, 2), (0..200).map(|i| i as Float).collect()).unwrap();
        let y = Array1::from_vec(
            (0..100)
                .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
                .collect(),
        );

        let config = ChunkedProcessingConfig {
            max_chunk_size: 30,
            ..Default::default()
        };

        let dataset = ChunkedDataset::from_arrays(&x, &y, config).unwrap();

        assert!(dataset.n_chunks() > 1);
        assert_eq!(dataset.n_samples(), 100);
        assert_eq!(dataset.n_features(), 2);
    }

    #[test]
    #[ignore]
    fn test_chunk_iteration() {
        let x = Array2::from_shape_vec((50, 3), (0..150).map(|i| i as Float).collect()).unwrap();
        let y = Array1::from_vec(
            (0..50)
                .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
                .collect(),
        );

        let config = ChunkedProcessingConfig {
            max_chunk_size: 20,
            ..Default::default()
        };

        let mut dataset = ChunkedDataset::from_arrays(&x, &y, config).unwrap();

        let mut total_samples = 0;
        let mut chunk_iter = dataset.chunk_iter();

        while let Some(chunk_result) = chunk_iter.next() {
            let (chunk_id, chunk_x, chunk_y) = chunk_result.unwrap();
            total_samples += chunk_x.nrows();
            assert_eq!(chunk_x.ncols(), 3);
            assert_eq!(chunk_x.nrows(), chunk_y.len());
        }

        // Due to overlap, total might be > original size
        assert!(total_samples >= 50);
    }

    #[test]
    #[ignore]
    fn test_chunked_dataset_stats() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
        let y = array![1.0, -1.0, 1.0, -1.0];

        let config = ChunkedProcessingConfig {
            max_chunk_size: 2,
            ..Default::default()
        };

        let mut dataset = ChunkedDataset::from_arrays(&x, &y, config).unwrap();
        let stats = dataset.compute_stats().unwrap();

        assert_eq!(stats.n_samples, 4);
        assert_eq!(stats.n_features, 2);
        assert!(stats.mean_x[0] > 0.0);
        assert!(stats.var_x[0] > 0.0);
    }

    #[test]
    #[ignore]
    fn test_chunked_svm_trainer() {
        let x = array![
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [4.0, 5.0],
            [-1.0, -2.0],
            [-2.0, -3.0],
            [-3.0, -4.0],
            [-4.0, -5.0]
        ];
        let y = array![1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0];

        let config = ChunkedProcessingConfig {
            max_chunk_size: 4,
            ..Default::default()
        };

        let dataset = ChunkedDataset::from_arrays(&x, &y, config).unwrap();
        let kernel = RbfKernel::new(1.0);
        let mut trainer = ChunkedSvmTrainer::new(kernel, ChunkedProcessingConfig::default());

        trainer.set_dataset(dataset);
        let result = trainer.train(1.0, 1e-3, 100).unwrap();

        assert!(result.n_support_vectors > 0);
        assert!(result.alpha.sum() > 0.0);
    }
}
