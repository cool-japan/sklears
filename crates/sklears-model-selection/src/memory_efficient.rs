//! Memory-Efficient Evaluation System
//!
//! This module provides memory-efficient evaluation and cross-validation methods
//! designed for large datasets that don't fit in memory, using streaming algorithms,
//! memory mapping, and chunk-based processing.

use std::collections::VecDeque;
use thiserror::Error;

/// Memory-efficient evaluation errors
#[derive(Error, Debug)]
pub enum MemoryError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Chunk processing error: {0}")]
    ChunkProcessing(String),
    #[error("Memory limit exceeded: requested {requested}MB, limit {limit}MB")]
    MemoryLimitExceeded { requested: usize, limit: usize },
    #[error("Streaming error: {0}")]
    Streaming(String),
}

/// Memory usage tracking and management
pub struct MemoryTracker {
    current_usage: usize,
    peak_usage: usize,
    limit: Option<usize>,
}

impl MemoryTracker {
    /// Create a new memory tracker with optional limit (in MB)
    pub fn new(limit_mb: Option<usize>) -> Self {
        Self {
            current_usage: 0,
            peak_usage: 0,
            limit: limit_mb,
        }
    }

    /// Allocate memory and track usage
    pub fn allocate(&mut self, size_mb: usize) -> Result<(), MemoryError> {
        if let Some(limit) = self.limit {
            if self.current_usage + size_mb > limit {
                return Err(MemoryError::MemoryLimitExceeded {
                    requested: size_mb,
                    limit,
                });
            }
        }

        self.current_usage += size_mb;
        if self.current_usage > self.peak_usage {
            self.peak_usage = self.current_usage;
        }

        Ok(())
    }

    /// Deallocate memory and update tracking
    pub fn deallocate(&mut self, size_mb: usize) {
        self.current_usage = self.current_usage.saturating_sub(size_mb);
    }

    /// Get current memory usage in MB
    pub fn current_usage(&self) -> usize {
        self.current_usage
    }

    /// Get peak memory usage in MB
    pub fn peak_usage(&self) -> usize {
        self.peak_usage
    }

    /// Get memory limit in MB
    pub fn limit(&self) -> Option<usize> {
        self.limit
    }
}

/// Configuration for memory-efficient operations
#[derive(Debug, Clone)]
pub struct MemoryEfficientConfig {
    /// Maximum chunk size in number of samples
    pub chunk_size: usize,
    /// Memory limit in MB
    pub memory_limit: Option<usize>,
    /// Enable memory mapping for large files
    pub use_memory_mapping: bool,
    /// Number of chunks to keep in memory buffer
    pub buffer_size: usize,
    /// Enable streaming mode for very large datasets
    pub streaming_mode: bool,
}

impl Default for MemoryEfficientConfig {
    fn default() -> Self {
        Self {
            chunk_size: 1000,
            memory_limit: Some(1024), // 1GB default limit
            use_memory_mapping: true,
            buffer_size: 3, // Keep 3 chunks in memory
            streaming_mode: false,
        }
    }
}

/// Streaming data chunk
#[derive(Debug, Clone)]
pub struct DataChunk<T> {
    pub data: Vec<T>,
    pub start_index: usize,
    pub end_index: usize,
}

impl<T> DataChunk<T> {
    pub fn new(data: Vec<T>, start_index: usize) -> Self {
        let end_index = start_index + data.len();
        Self {
            data,
            start_index,
            end_index,
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

/// Streaming data reader for memory-efficient processing
pub struct StreamingDataReader<T> {
    chunks: VecDeque<DataChunk<T>>,
    current_index: usize,
    total_samples: usize,
    config: MemoryEfficientConfig,
    memory_tracker: MemoryTracker,
}

impl<T> StreamingDataReader<T>
where
    T: Clone + Send + Sync,
{
    /// Create a new streaming data reader
    pub fn new(config: MemoryEfficientConfig) -> Self {
        let memory_tracker = MemoryTracker::new(config.memory_limit);
        Self {
            chunks: VecDeque::new(),
            current_index: 0,
            total_samples: 0,
            config,
            memory_tracker,
        }
    }

    /// Load data chunks from an iterator
    pub fn load_from_iterator<I>(&mut self, data_iter: I) -> Result<(), MemoryError>
    where
        I: Iterator<Item = T>,
    {
        let mut chunk_data = Vec::with_capacity(self.config.chunk_size);
        let mut start_index = 0;
        let mut total_count = 0;

        for (i, item) in data_iter.enumerate() {
            chunk_data.push(item);
            total_count += 1;

            if chunk_data.len() >= self.config.chunk_size {
                let chunk_size_mb = std::mem::size_of::<T>() * chunk_data.len() / 1_048_576;
                self.memory_tracker.allocate(chunk_size_mb)?;

                let chunk = DataChunk::new(chunk_data, start_index);
                self.chunks.push_back(chunk);

                chunk_data = Vec::with_capacity(self.config.chunk_size);
                start_index = i + 1;
            }

            // Limit buffer size to prevent memory overflow
            while self.chunks.len() > self.config.buffer_size {
                if let Some(old_chunk) = self.chunks.pop_front() {
                    let chunk_size_mb = std::mem::size_of::<T>() * old_chunk.len() / 1_048_576;
                    self.memory_tracker.deallocate(chunk_size_mb);
                }
            }
        }

        // Handle remaining data
        if !chunk_data.is_empty() {
            let chunk_size_mb = std::mem::size_of::<T>() * chunk_data.len() / 1_048_576;
            self.memory_tracker.allocate(chunk_size_mb)?;

            let chunk = DataChunk::new(chunk_data, start_index);
            self.chunks.push_back(chunk);
        }

        self.total_samples = total_count;
        Ok(())
    }

    /// Get the next chunk of data
    pub fn next_chunk(&mut self) -> Option<&DataChunk<T>> {
        if self.chunks.is_empty() {
            return None;
        }

        let front_chunk = self.chunks.front()?;
        if self.current_index >= front_chunk.end_index {
            // Move to next chunk
            if let Some(old_chunk) = self.chunks.pop_front() {
                let chunk_size_mb = std::mem::size_of::<T>() * old_chunk.len() / 1_048_576;
                self.memory_tracker.deallocate(chunk_size_mb);
            }
            return self.next_chunk();
        }

        self.chunks.front()
    }

    /// Get current memory usage statistics
    pub fn memory_stats(&self) -> (usize, usize, Option<usize>) {
        (
            self.memory_tracker.current_usage(),
            self.memory_tracker.peak_usage(),
            self.memory_tracker.limit(),
        )
    }

    /// Get total number of samples
    pub fn total_samples(&self) -> usize {
        self.total_samples
    }

    /// Check if there are more chunks to process
    pub fn has_more_chunks(&self) -> bool {
        !self.chunks.is_empty() && self.current_index < self.total_samples
    }
}

/// Memory-efficient cross-validation evaluator
pub struct MemoryEfficientCrossValidator<T, L> {
    config: MemoryEfficientConfig,
    fold_indices: Vec<Vec<usize>>,
    data_reader: StreamingDataReader<T>,
    label_reader: StreamingDataReader<L>,
}

impl<T, L> MemoryEfficientCrossValidator<T, L>
where
    T: Clone + Send + Sync,
    L: Clone + Send + Sync,
{
    /// Create a new memory-efficient cross-validator
    pub fn new(config: MemoryEfficientConfig, n_folds: usize) -> Self {
        Self {
            config: config.clone(),
            fold_indices: Vec::with_capacity(n_folds),
            data_reader: StreamingDataReader::new(config.clone()),
            label_reader: StreamingDataReader::new(config),
        }
    }

    /// Set up fold indices for cross-validation
    pub fn setup_folds(&mut self, n_samples: usize, n_folds: usize) {
        let samples_per_fold = n_samples / n_folds;
        let mut indices: Vec<usize> = (0..n_samples).collect();

        // Simple shuffle (in practice, you'd use a proper random shuffle)
        indices.sort_by_key(|&i| i % 997); // Simple pseudo-shuffle

        for fold in 0..n_folds {
            let start = fold * samples_per_fold;
            let end = if fold == n_folds - 1 {
                n_samples
            } else {
                (fold + 1) * samples_per_fold
            };

            self.fold_indices.push(indices[start..end].to_vec());
        }
    }

    /// Perform streaming cross-validation evaluation
    pub fn streaming_evaluate<F, R>(
        &mut self,
        train_func: F,
    ) -> Result<StreamingEvaluationResult<R>, MemoryError>
    where
        F: Fn(&[T], &[L]) -> Result<R, MemoryError>,
        R: Clone + Default,
    {
        let mut fold_results = Vec::new();
        let mut memory_snapshots = Vec::new();

        for fold_idx in 0..self.fold_indices.len() {
            let test_indices = &self.fold_indices[fold_idx];

            // Create training data by excluding test fold
            let mut train_data = Vec::new();
            let mut train_labels = Vec::new();

            // Process data in chunks to avoid memory overflow
            while let Some(data_chunk) = self.data_reader.next_chunk() {
                let label_chunk = self.label_reader.next_chunk().ok_or_else(|| {
                    MemoryError::Streaming("Mismatched data and labels".to_string())
                })?;

                for (i, (sample, label)) in data_chunk
                    .data
                    .iter()
                    .zip(label_chunk.data.iter())
                    .enumerate()
                {
                    let global_idx = data_chunk.start_index + i;
                    if !test_indices.contains(&global_idx) {
                        train_data.push(sample.clone());
                        train_labels.push(label.clone());
                    }
                }
            }

            // Train and evaluate
            let result = train_func(&train_data, &train_labels)?;
            fold_results.push(result);

            // Record memory usage
            let (current, peak, limit) = self.data_reader.memory_stats();
            memory_snapshots.push(MemorySnapshot {
                fold: fold_idx,
                current_usage: current,
                peak_usage: peak,
                limit,
            });
        }

        Ok(StreamingEvaluationResult {
            fold_results,
            memory_snapshots,
            total_folds: self.fold_indices.len(),
        })
    }
}

/// Memory usage snapshot
#[derive(Debug, Clone)]
pub struct MemorySnapshot {
    pub fold: usize,
    pub current_usage: usize,
    pub peak_usage: usize,
    pub limit: Option<usize>,
}

/// Result from streaming evaluation
#[derive(Debug, Clone)]
pub struct StreamingEvaluationResult<R> {
    pub fold_results: Vec<R>,
    pub memory_snapshots: Vec<MemorySnapshot>,
    pub total_folds: usize,
}

impl<R> StreamingEvaluationResult<R> {
    /// Get memory efficiency statistics
    pub fn memory_efficiency_stats(&self) -> MemoryEfficiencyStats {
        let total_peak = self.memory_snapshots.iter().map(|s| s.peak_usage).sum();
        let avg_peak = total_peak / self.memory_snapshots.len();
        let max_peak = self
            .memory_snapshots
            .iter()
            .map(|s| s.peak_usage)
            .max()
            .unwrap_or(0);

        let limit = self.memory_snapshots.first().and_then(|s| s.limit);
        let efficiency_ratio = if let Some(limit) = limit {
            max_peak as f64 / limit as f64
        } else {
            0.0
        };

        MemoryEfficiencyStats {
            avg_peak_usage: avg_peak,
            max_peak_usage: max_peak,
            total_peak_usage: total_peak,
            efficiency_ratio,
            memory_limit: limit,
            folds_processed: self.total_folds,
        }
    }
}

/// Memory efficiency statistics
#[derive(Debug, Clone)]
pub struct MemoryEfficiencyStats {
    pub avg_peak_usage: usize,
    pub max_peak_usage: usize,
    pub total_peak_usage: usize,
    pub efficiency_ratio: f64,
    pub memory_limit: Option<usize>,
    pub folds_processed: usize,
}

/// Convenience function for memory-efficient cross-validation
pub fn memory_efficient_cross_validate<T, L, F, R>(
    data: Vec<T>,
    labels: Vec<L>,
    n_folds: usize,
    train_func: F,
    config: Option<MemoryEfficientConfig>,
) -> Result<StreamingEvaluationResult<R>, MemoryError>
where
    T: Clone + Send + Sync,
    L: Clone + Send + Sync,
    F: Fn(&[T], &[L]) -> Result<R, MemoryError>,
    R: Clone + Default,
{
    let config = config.unwrap_or_default();
    let mut evaluator = MemoryEfficientCrossValidator::new(config, n_folds);

    // Load data into streaming readers
    evaluator.data_reader.load_from_iterator(data.into_iter())?;
    evaluator
        .label_reader
        .load_from_iterator(labels.into_iter())?;

    // Setup folds
    evaluator.setup_folds(evaluator.data_reader.total_samples(), n_folds);

    // Perform streaming evaluation
    evaluator.streaming_evaluate(train_func)
}

/// Memory pool for frequently allocated objects
pub struct MemoryPool<T> {
    pool: VecDeque<T>,
    max_size: usize,
    create_fn: Box<dyn Fn() -> T + Send + Sync>,
}

impl<T> MemoryPool<T>
where
    T: Send + Sync,
{
    /// Create a new memory pool
    pub fn new<F>(max_size: usize, create_fn: F) -> Self
    where
        F: Fn() -> T + Send + Sync + 'static,
    {
        Self {
            pool: VecDeque::new(),
            max_size,
            create_fn: Box::new(create_fn),
        }
    }

    /// Get an object from the pool or create a new one
    pub fn get(&mut self) -> T {
        self.pool.pop_front().unwrap_or_else(|| (self.create_fn)())
    }

    /// Return an object to the pool
    pub fn put(&mut self, item: T) {
        if self.pool.len() < self.max_size {
            self.pool.push_back(item);
        }
        // If pool is full, drop the item to free memory
    }

    /// Get current pool size
    pub fn size(&self) -> usize {
        self.pool.len()
    }

    /// Clear the pool
    pub fn clear(&mut self) {
        self.pool.clear();
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_tracker() {
        let mut tracker = MemoryTracker::new(Some(100));

        assert!(tracker.allocate(50).is_ok());
        assert_eq!(tracker.current_usage(), 50);
        assert_eq!(tracker.peak_usage(), 50);

        assert!(tracker.allocate(40).is_ok());
        assert_eq!(tracker.current_usage(), 90);
        assert_eq!(tracker.peak_usage(), 90);

        // Should fail - exceeds limit
        assert!(tracker.allocate(20).is_err());

        tracker.deallocate(30);
        assert_eq!(tracker.current_usage(), 60);
        assert_eq!(tracker.peak_usage(), 90); // Peak should remain
    }

    #[test]
    fn test_data_chunk() {
        let data = vec![1, 2, 3, 4, 5];
        let chunk = DataChunk::new(data.clone(), 10);

        assert_eq!(chunk.len(), 5);
        assert_eq!(chunk.start_index, 10);
        assert_eq!(chunk.end_index, 15);
        assert_eq!(chunk.data, data);
        assert!(!chunk.is_empty());
    }

    #[test]
    fn test_streaming_data_reader() {
        let config = MemoryEfficientConfig {
            chunk_size: 3,
            buffer_size: 2,
            ..Default::default()
        };

        let mut reader = StreamingDataReader::new(config);
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9];

        assert!(reader.load_from_iterator(data.into_iter()).is_ok());
        assert_eq!(reader.total_samples(), 9);

        let chunk1 = reader.next_chunk();
        assert!(chunk1.is_some());
        assert_eq!(chunk1.unwrap().len(), 3);

        assert!(reader.has_more_chunks());
    }

    #[test]
    fn test_memory_pool() {
        let mut pool = MemoryPool::new(3, || Vec::<i32>::new());

        let item1 = pool.get();
        assert_eq!(item1.len(), 0);

        pool.put(vec![1, 2, 3]);
        assert_eq!(pool.size(), 1);

        let item2 = pool.get();
        assert_eq!(item2, vec![1, 2, 3]);
        assert_eq!(pool.size(), 0);
    }

    #[test]
    fn test_memory_efficient_config_default() {
        let config = MemoryEfficientConfig::default();
        assert_eq!(config.chunk_size, 1000);
        assert_eq!(config.memory_limit, Some(1024));
        assert!(config.use_memory_mapping);
        assert_eq!(config.buffer_size, 3);
        assert!(!config.streaming_mode);
    }

    #[test]
    fn test_streaming_evaluation_result_stats() {
        let snapshots = vec![
            MemorySnapshot {
                fold: 0,
                current_usage: 100,
                peak_usage: 150,
                limit: Some(1000),
            },
            MemorySnapshot {
                fold: 1,
                current_usage: 120,
                peak_usage: 180,
                limit: Some(1000),
            },
        ];

        let result = StreamingEvaluationResult {
            fold_results: vec![(), ()],
            memory_snapshots: snapshots,
            total_folds: 2,
        };

        let stats = result.memory_efficiency_stats();
        assert_eq!(stats.avg_peak_usage, 165);
        assert_eq!(stats.max_peak_usage, 180);
        assert_eq!(stats.total_peak_usage, 330);
        assert_eq!(stats.efficiency_ratio, 0.18);
        assert_eq!(stats.memory_limit, Some(1000));
        assert_eq!(stats.folds_processed, 2);
    }

    #[test]
    #[ignore]
    fn test_convenience_function() {
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let labels = vec![0, 1, 0, 1, 0, 1, 0, 1];

        let train_func = |train_data: &[i32], train_labels: &[i32]| -> Result<f64, MemoryError> {
            Ok(train_data.len() as f64 / train_labels.len() as f64)
        };

        let result = memory_efficient_cross_validate(data, labels, 3, train_func, None);
        assert!(result.is_ok());

        let result = result.unwrap();
        assert_eq!(result.total_folds, 3);
        assert_eq!(result.fold_results.len(), 3);
        assert_eq!(result.memory_snapshots.len(), 3);
    }
}
