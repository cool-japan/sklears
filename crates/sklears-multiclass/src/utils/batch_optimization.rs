//! Batch optimization utilities for multiclass classification
//!
//! This module provides utilities for optimizing batch prediction operations,
//! including memory-efficient processing, parallel execution, and dynamic
//! batch sizing for improved performance on large datasets.

use scirs2_core::ndarray::{s, Array1, Array2, Axis};
use sklears_core::error::{Result as SklResult, SklearsError};
use std::cmp;

/// Configuration for batch processing optimization
#[derive(Debug, Clone)]
pub struct BatchConfig {
    /// Maximum batch size for processing
    pub max_batch_size: usize,
    /// Target memory usage in bytes (approximate)
    pub target_memory_mb: usize,
    /// Whether to use parallel processing
    pub use_parallel: bool,
    /// Number of threads for parallel processing (None = auto-detect)
    pub n_threads: Option<usize>,
    /// Whether to enable dynamic batch sizing
    pub dynamic_sizing: bool,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 1000,
            target_memory_mb: 512, // 512 MB default
            use_parallel: true,
            n_threads: None,
            dynamic_sizing: true,
        }
    }
}

impl BatchConfig {
    /// Create a new batch configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the maximum batch size
    pub fn with_max_batch_size(mut self, max_batch_size: usize) -> Self {
        self.max_batch_size = max_batch_size;
        self
    }

    /// Set the target memory usage in MB
    pub fn with_target_memory_mb(mut self, target_memory_mb: usize) -> Self {
        self.target_memory_mb = target_memory_mb;
        self
    }

    /// Enable or disable parallel processing
    pub fn with_parallel(mut self, use_parallel: bool) -> Self {
        self.use_parallel = use_parallel;
        self
    }

    /// Set the number of threads for parallel processing
    pub fn with_n_threads(mut self, n_threads: Option<usize>) -> Self {
        self.n_threads = n_threads;
        self
    }

    /// Enable or disable dynamic batch sizing
    pub fn with_dynamic_sizing(mut self, dynamic_sizing: bool) -> Self {
        self.dynamic_sizing = dynamic_sizing;
        self
    }
}

/// Batch processor for optimized multiclass predictions
pub struct BatchProcessor {
    config: BatchConfig,
    optimal_batch_size: Option<usize>,
}

impl BatchProcessor {
    /// Create a new batch processor with the given configuration
    pub fn new(config: BatchConfig) -> Self {
        Self {
            config,
            optimal_batch_size: None,
        }
    }

    /// Create a new batch processor with default configuration
    pub fn default() -> Self {
        Self::new(BatchConfig::default())
    }

    /// Calculate optimal batch size based on data characteristics
    pub fn calculate_optimal_batch_size(&self, n_samples: usize, n_features: usize) -> usize {
        if !self.config.dynamic_sizing {
            return self.config.max_batch_size;
        }

        // Estimate memory usage per sample (in bytes)
        // Assume f64 (8 bytes) for features and additional overhead
        let bytes_per_sample = n_features * 8 + 64; // 64 bytes overhead per sample
        let target_bytes = self.config.target_memory_mb * 1024 * 1024;

        // Calculate batch size that fits in target memory
        let memory_based_batch_size = target_bytes / bytes_per_sample;

        // Use the minimum of memory-based size, max batch size, and total samples
        cmp::min(
            cmp::min(memory_based_batch_size, self.config.max_batch_size),
            n_samples,
        )
        .max(1) // Ensure at least batch size of 1
    }

    /// Update the optimal batch size based on performance feedback
    pub fn update_optimal_batch_size(&mut self, batch_size: usize) {
        self.optimal_batch_size = Some(batch_size);
    }

    /// Get the current optimal batch size
    pub fn get_optimal_batch_size(&self) -> Option<usize> {
        self.optimal_batch_size
    }

    /// Process data in batches with the given prediction function
    ///
    /// # Arguments
    /// * `data` - Input data matrix [n_samples, n_features]
    /// * `predict_fn` - Function that takes a batch and returns predictions
    ///
    /// # Returns
    /// Array of predictions for all samples
    pub fn process_batches<F>(
        &mut self,
        data: &Array2<f64>,
        mut predict_fn: F,
    ) -> SklResult<Array1<i32>>
    where
        F: FnMut(&Array2<f64>) -> SklResult<Array1<i32>>,
    {
        let (n_samples, n_features) = data.dim();

        if n_samples == 0 {
            return Ok(Array1::zeros(0));
        }

        let batch_size = self
            .optimal_batch_size
            .unwrap_or_else(|| self.calculate_optimal_batch_size(n_samples, n_features));

        let mut predictions = Vec::with_capacity(n_samples);

        // Process data in batches
        for start in (0..n_samples).step_by(batch_size) {
            let end = cmp::min(start + batch_size, n_samples);
            let batch = data.slice(s![start..end, ..]).to_owned();

            let batch_predictions = predict_fn(&batch)?;
            predictions.extend(batch_predictions.iter().cloned());
        }

        Ok(Array1::from_vec(predictions))
    }

    /// Process data in batches for probability predictions
    ///
    /// # Arguments
    /// * `data` - Input data matrix [n_samples, n_features]
    /// * `predict_proba_fn` - Function that takes a batch and returns probability predictions
    ///
    /// # Returns
    /// Array of probability predictions for all samples [n_samples, n_classes]
    pub fn process_batches_proba<F>(
        &mut self,
        data: &Array2<f64>,
        mut predict_proba_fn: F,
    ) -> SklResult<Array2<f64>>
    where
        F: FnMut(&Array2<f64>) -> SklResult<Array2<f64>>,
    {
        let (n_samples, n_features) = data.dim();

        if n_samples == 0 {
            return Err(SklearsError::InvalidInput(
                "Cannot process empty data".to_string(),
            ));
        }

        let batch_size = self
            .optimal_batch_size
            .unwrap_or_else(|| self.calculate_optimal_batch_size(n_samples, n_features));

        let mut all_probabilities = Vec::new();
        let mut n_classes = None;

        // Process data in batches
        for start in (0..n_samples).step_by(batch_size) {
            let end = cmp::min(start + batch_size, n_samples);
            let batch = data.slice(s![start..end, ..]).to_owned();

            let batch_probabilities = predict_proba_fn(&batch)?;

            // Determine number of classes from first batch
            if n_classes.is_none() {
                n_classes = Some(batch_probabilities.ncols());
            }

            // Validate consistency
            if batch_probabilities.ncols() != n_classes.unwrap() {
                return Err(SklearsError::InvalidInput(
                    "Inconsistent number of classes across batches".to_string(),
                ));
            }

            all_probabilities.push(batch_probabilities);
        }

        // Concatenate all batch results
        let result = scirs2_core::ndarray::concatenate(
            Axis(0),
            &all_probabilities
                .iter()
                .map(|arr| arr.view())
                .collect::<Vec<_>>(),
        )
        .map_err(|e| {
            SklearsError::InvalidInput(format!("Failed to concatenate batch results: {}", e))
        })?;

        Ok(result)
    }

    /// Get performance statistics for the current configuration
    pub fn get_performance_stats(&self) -> BatchPerformanceStats {
        BatchPerformanceStats {
            optimal_batch_size: self.optimal_batch_size,
            config: self.config.clone(),
        }
    }
}

/// Performance statistics for batch processing
#[derive(Debug, Clone)]
pub struct BatchPerformanceStats {
    /// Current optimal batch size (if determined)
    pub optimal_batch_size: Option<usize>,
    /// Configuration used
    pub config: BatchConfig,
}

impl BatchPerformanceStats {
    /// Get a summary string of the performance statistics
    pub fn summary(&self) -> String {
        format!(
            "BatchPerformanceStats {{ optimal_batch_size: {:?}, max_batch_size: {}, target_memory_mb: {}, use_parallel: {} }}",
            self.optimal_batch_size,
            self.config.max_batch_size,
            self.config.target_memory_mb,
            self.config.use_parallel
        )
    }
}

/// Memory-efficient iterator for processing large datasets in chunks
pub struct BatchIterator<'a> {
    data: &'a Array2<f64>,
    batch_size: usize,
    current_start: usize,
}

impl<'a> BatchIterator<'a> {
    /// Create a new batch iterator
    pub fn new(data: &'a Array2<f64>, batch_size: usize) -> Self {
        Self {
            data,
            batch_size,
            current_start: 0,
        }
    }

    /// Get the total number of batches
    pub fn n_batches(&self) -> usize {
        (self.data.nrows() + self.batch_size - 1) / self.batch_size
    }

    /// Get the remaining number of samples
    pub fn remaining_samples(&self) -> usize {
        if self.current_start >= self.data.nrows() {
            0
        } else {
            self.data.nrows() - self.current_start
        }
    }
}

impl<'a> Iterator for BatchIterator<'a> {
    type Item = Array2<f64>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_start >= self.data.nrows() {
            return None;
        }

        let end = cmp::min(self.current_start + self.batch_size, self.data.nrows());
        let batch = self.data.slice(s![self.current_start..end, ..]).to_owned();
        self.current_start = end;

        Some(batch)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_config_creation() {
        let config = BatchConfig::new()
            .with_max_batch_size(500)
            .with_target_memory_mb(256)
            .with_parallel(false);

        assert_eq!(config.max_batch_size, 500);
        assert_eq!(config.target_memory_mb, 256);
        assert!(!config.use_parallel);
    }

    #[test]
    fn test_batch_processor_optimal_size_calculation() {
        let config = BatchConfig::new().with_dynamic_sizing(true);
        let processor = BatchProcessor::new(config);

        // Test with different data sizes
        let batch_size = processor.calculate_optimal_batch_size(1000, 100);
        assert!(batch_size > 0);
        assert!(batch_size <= 1000);
    }

    #[test]
    fn test_batch_processing() {
        let mut processor = BatchProcessor::default();
        let data = Array2::ones((100, 10));

        // Mock prediction function that returns class 0 for all samples
        let predict_fn =
            |batch: &Array2<f64>| -> SklResult<Array1<i32>> { Ok(Array1::zeros(batch.nrows())) };

        let predictions = processor.process_batches(&data, predict_fn).unwrap();
        assert_eq!(predictions.len(), 100);
        assert!(predictions.iter().all(|&x| x == 0));
    }

    #[test]
    fn test_batch_processing_probabilities() {
        let mut processor = BatchProcessor::default();
        let data = Array2::ones((50, 5));

        // Mock probability prediction function
        let predict_proba_fn = |batch: &Array2<f64>| -> SklResult<Array2<f64>> {
            let n_samples = batch.nrows();
            let probabilities = Array2::from_elem((n_samples, 3), 1.0 / 3.0); // 3 classes
            Ok(probabilities)
        };

        let probabilities = processor
            .process_batches_proba(&data, predict_proba_fn)
            .unwrap();
        assert_eq!(probabilities.dim(), (50, 3));

        // Check that probabilities are approximately 1/3 for each class
        for row in probabilities.outer_iter() {
            for &prob in row.iter() {
                assert!((prob - 1.0 / 3.0).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_batch_iterator() {
        let data = Array2::ones((100, 10));
        let iterator = BatchIterator::new(&data, 30);

        assert_eq!(iterator.n_batches(), 4); // 100 / 30 = 3.33, ceil = 4

        let batches: Vec<_> = iterator.collect();
        assert_eq!(batches.len(), 4);
        assert_eq!(batches[0].nrows(), 30);
        assert_eq!(batches[1].nrows(), 30);
        assert_eq!(batches[2].nrows(), 30);
        assert_eq!(batches[3].nrows(), 10); // Last batch with remaining samples
    }

    #[test]
    fn test_empty_data_handling() {
        let mut processor = BatchProcessor::default();
        let empty_data = Array2::zeros((0, 5));

        let predict_fn = |_: &Array2<f64>| -> SklResult<Array1<i32>> { Ok(Array1::zeros(0)) };

        let predictions = processor.process_batches(&empty_data, predict_fn).unwrap();
        assert_eq!(predictions.len(), 0);
    }

    #[test]
    fn test_performance_stats() {
        let config = BatchConfig::new().with_max_batch_size(100);
        let processor = BatchProcessor::new(config);
        let stats = processor.get_performance_stats();

        assert_eq!(stats.config.max_batch_size, 100);
        assert!(stats.optimal_batch_size.is_none());

        let summary = stats.summary();
        assert!(summary.contains("max_batch_size: 100"));
    }
}
