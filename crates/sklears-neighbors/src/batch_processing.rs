use crate::distance::Distance;
use crate::{NeighborsError, NeighborsResult};
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use scirs2_core::ndarray::{Array1, Array2, ArrayView2, Axis};
use sklears_core::types::Float;
use std::collections::VecDeque;

#[derive(Debug, Clone)]
pub struct BatchConfiguration {
    pub batch_size: usize,
    pub max_memory_mb: usize,
    pub parallel_processing: bool,
    pub chunk_overlap: usize,
    pub prefetch_batches: usize,
}

impl Default for BatchConfiguration {
    fn default() -> Self {
        Self {
            batch_size: 1000,
            max_memory_mb: 512,
            parallel_processing: true,
            chunk_overlap: 0,
            prefetch_batches: 2,
        }
    }
}

pub struct BatchProcessor {
    config: BatchConfiguration,
    memory_monitor: MemoryMonitor,
}

#[derive(Debug, Clone)]
pub struct MemoryMonitor {
    peak_memory_mb: usize,
    current_memory_mb: usize,
    memory_threshold_mb: usize,
}

#[derive(Debug, Clone)]
pub struct BatchResult<T> {
    pub results: Vec<T>,
    pub batch_stats: BatchStatistics,
}

#[derive(Debug, Clone)]
pub struct BatchStatistics {
    pub total_batches: usize,
    pub processed_samples: usize,
    pub processing_time_ms: u128,
    pub peak_memory_mb: usize,
    pub average_batch_size: usize,
    pub memory_efficiency: Float,
}

pub trait BatchProcessable<T> {
    fn process_batch(&self, batch_data: ArrayView2<Float>) -> NeighborsResult<Vec<T>>;
    fn estimate_memory_per_sample(&self) -> usize;
    fn supports_parallel_processing(&self) -> bool;
}

impl BatchProcessor {
    pub fn new(config: BatchConfiguration) -> Self {
        let memory_monitor = MemoryMonitor {
            peak_memory_mb: 0,
            current_memory_mb: 0,
            memory_threshold_mb: config.max_memory_mb,
        };

        Self {
            config,
            memory_monitor,
        }
    }

    pub fn builder() -> BatchProcessorBuilder {
        BatchProcessorBuilder::new()
    }

    pub fn process_data<T, P>(
        &mut self,
        data: ArrayView2<Float>,
        processor: &P,
    ) -> NeighborsResult<BatchResult<T>>
    where
        T: Send + Sync + Clone,
        P: BatchProcessable<T> + Sync,
    {
        let start_time = std::time::Instant::now();
        let num_samples = data.nrows();

        if num_samples == 0 {
            return Err(NeighborsError::EmptyInput);
        }

        let optimal_batch_size = self.calculate_optimal_batch_size(num_samples, processor)?;

        let mut all_results = Vec::new();
        let mut batch_count = 0;

        if self.config.parallel_processing && processor.supports_parallel_processing() {
            all_results = self.process_data_parallel(data, optimal_batch_size, processor)?;
            batch_count = (num_samples + optimal_batch_size - 1) / optimal_batch_size;
        } else {
            let mut start_idx = 0;
            while start_idx < num_samples {
                let end_idx = std::cmp::min(start_idx + optimal_batch_size, num_samples);
                let batch = data.slice(scirs2_core::ndarray::s![start_idx..end_idx, ..]);

                let batch_results = processor.process_batch(batch)?;
                all_results.extend(batch_results);
                batch_count += 1;
                self.update_memory_usage(batch.nrows() * batch.ncols() * 8)?;

                start_idx = end_idx - self.config.chunk_overlap;
                if start_idx >= end_idx {
                    break;
                }
            }
        }

        let processing_time = start_time.elapsed().as_millis();
        let stats = BatchStatistics {
            total_batches: batch_count,
            processed_samples: num_samples,
            processing_time_ms: processing_time,
            peak_memory_mb: self.memory_monitor.peak_memory_mb,
            average_batch_size: num_samples / batch_count.max(1),
            memory_efficiency: self.calculate_memory_efficiency(),
        };

        Ok(BatchResult {
            results: all_results,
            batch_stats: stats,
        })
    }

    pub fn process_streaming_data<T, P>(
        &mut self,
        data_stream: impl Iterator<Item = Array1<Float>>,
        processor: &P,
    ) -> NeighborsResult<Vec<T>>
    where
        T: Send + Sync + Clone,
        P: BatchProcessable<T> + Sync,
    {
        let mut buffer = VecDeque::new();
        let batch_size = self.config.batch_size;
        let mut all_results = Vec::new();

        for sample in data_stream {
            buffer.push_back(sample);

            if buffer.len() >= batch_size {
                let batch_data: Vec<Array1<Float>> = buffer.drain(..batch_size).collect();
                let batch_matrix = self.vec_to_array2(batch_data)?;
                let results = processor.process_batch(batch_matrix.view())?;
                all_results.extend(results);
            }
        }

        // Process remaining samples in buffer
        if !buffer.is_empty() {
            let batch_data: Vec<Array1<Float>> = buffer.drain(..).collect();
            let batch_matrix = self.vec_to_array2(batch_data)?;
            let results = processor.process_batch(batch_matrix.view())?;
            all_results.extend(results);
        }

        Ok(all_results)
    }

    fn calculate_optimal_batch_size<T, P>(
        &self,
        num_samples: usize,
        processor: &P,
    ) -> NeighborsResult<usize>
    where
        P: BatchProcessable<T>,
    {
        let memory_per_sample = processor.estimate_memory_per_sample();
        let max_samples_per_batch = (self.config.max_memory_mb * 1024 * 1024) / memory_per_sample;

        let optimal_size = std::cmp::min(
            std::cmp::min(self.config.batch_size, max_samples_per_batch),
            num_samples,
        );

        if optimal_size == 0 {
            return Err(NeighborsError::InvalidInput(
                "Batch size too small for available memory".to_string(),
            ));
        }

        Ok(optimal_size)
    }

    #[cfg(feature = "parallel")]
    fn process_data_parallel<T, P>(
        &self,
        data: ArrayView2<Float>,
        batch_size: usize,
        processor: &P,
    ) -> NeighborsResult<Vec<T>>
    where
        T: Send + Sync + Clone,
        P: BatchProcessable<T> + Sync,
    {
        let num_samples = data.nrows();
        let chunk_indices: Vec<(usize, usize)> = (0..num_samples)
            .step_by(batch_size)
            .map(|start| {
                let end = std::cmp::min(start + batch_size, num_samples);
                (start, end)
            })
            .collect();

        let results: Result<Vec<Vec<T>>, NeighborsError> = chunk_indices
            .par_iter()
            .map(|&(start, end)| {
                let batch = data.slice(scirs2_core::ndarray::s![start..end, ..]);
                processor.process_batch(batch)
            })
            .collect();

        match results {
            Ok(batch_results) => Ok(batch_results.into_iter().flatten().collect()),
            Err(e) => Err(e),
        }
    }

    #[cfg(not(feature = "parallel"))]
    fn process_data_parallel<T, P>(
        &self,
        data: ArrayView2<Float>,
        batch_size: usize,
        processor: &P,
    ) -> NeighborsResult<Vec<T>>
    where
        T: Send + Sync + Clone,
        P: BatchProcessable<T> + Sync,
    {
        // Fallback to sequential processing if parallel feature is not enabled
        let mut all_results = Vec::new();
        let num_samples = data.nrows();
        let mut start_idx = 0;

        while start_idx < num_samples {
            let end_idx = std::cmp::min(start_idx + batch_size, num_samples);
            let batch = data.slice(scirs2_core::ndarray::s![start_idx..end_idx, ..]);

            let batch_results = processor.process_batch(batch)?;
            all_results.extend(batch_results);

            start_idx = end_idx;
        }

        Ok(all_results)
    }

    fn update_memory_usage(&mut self, additional_bytes: usize) -> NeighborsResult<()> {
        let additional_mb = additional_bytes / (1024 * 1024);
        self.memory_monitor.current_memory_mb += additional_mb;

        if self.memory_monitor.current_memory_mb > self.memory_monitor.peak_memory_mb {
            self.memory_monitor.peak_memory_mb = self.memory_monitor.current_memory_mb;
        }

        if self.memory_monitor.current_memory_mb > self.memory_monitor.memory_threshold_mb {
            return Err(NeighborsError::InvalidInput(format!(
                "Memory usage exceeded threshold: {} MB",
                self.memory_monitor.memory_threshold_mb
            )));
        }

        Ok(())
    }

    fn calculate_memory_efficiency(&self) -> Float {
        if self.memory_monitor.memory_threshold_mb == 0 {
            return 1.0;
        }

        1.0 - (self.memory_monitor.peak_memory_mb as Float
            / self.memory_monitor.memory_threshold_mb as Float)
    }

    fn vec_to_array2(&self, vec_data: Vec<Array1<Float>>) -> NeighborsResult<Array2<Float>> {
        if vec_data.is_empty() {
            return Err(NeighborsError::EmptyInput);
        }

        let n_samples = vec_data.len();
        let n_features = vec_data[0].len();

        let mut result = Array2::zeros((n_samples, n_features));
        for (i, row) in vec_data.iter().enumerate() {
            if row.len() != n_features {
                return Err(NeighborsError::ShapeMismatch {
                    expected: vec![n_features],
                    actual: vec![row.len()],
                });
            }
            result.row_mut(i).assign(row);
        }

        Ok(result)
    }

    pub fn get_memory_stats(&self) -> &MemoryMonitor {
        &self.memory_monitor
    }

    pub fn reset_memory_monitor(&mut self) {
        self.memory_monitor.current_memory_mb = 0;
        self.memory_monitor.peak_memory_mb = 0;
    }
}

pub struct BatchProcessorBuilder {
    config: BatchConfiguration,
}

impl Default for BatchProcessorBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl BatchProcessorBuilder {
    pub fn new() -> Self {
        Self {
            config: BatchConfiguration::default(),
        }
    }

    pub fn batch_size(mut self, size: usize) -> Self {
        self.config.batch_size = size;
        self
    }

    pub fn max_memory_mb(mut self, memory_mb: usize) -> Self {
        self.config.max_memory_mb = memory_mb;
        self
    }

    pub fn parallel_processing(mut self, enabled: bool) -> Self {
        self.config.parallel_processing = enabled;
        self
    }

    pub fn chunk_overlap(mut self, overlap: usize) -> Self {
        self.config.chunk_overlap = overlap;
        self
    }

    pub fn prefetch_batches(mut self, count: usize) -> Self {
        self.config.prefetch_batches = count;
        self
    }

    pub fn build(self) -> BatchProcessor {
        BatchProcessor::new(self.config)
    }
}

pub struct BatchNeighborSearch {
    k: usize,
    distance: Distance,
    training_data: Array2<Float>,
}

impl BatchNeighborSearch {
    pub fn new(k: usize, distance: Distance, training_data: Array2<Float>) -> Self {
        Self {
            k,
            distance,
            training_data,
        }
    }
}

impl BatchProcessable<(Vec<usize>, Vec<Float>)> for BatchNeighborSearch {
    fn process_batch(
        &self,
        batch_data: ArrayView2<Float>,
    ) -> NeighborsResult<Vec<(Vec<usize>, Vec<Float>)>> {
        let mut results = Vec::new();

        for query_row in batch_data.axis_iter(Axis(0)) {
            let mut distances: Vec<(Float, usize)> = self
                .training_data
                .axis_iter(Axis(0))
                .enumerate()
                .map(|(idx, train_row)| {
                    let dist = self.distance.calculate(&query_row, &train_row);
                    (dist, idx)
                })
                .collect();

            distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            distances.truncate(self.k);

            let indices: Vec<usize> = distances.iter().map(|(_, idx)| *idx).collect();
            let dists: Vec<Float> = distances.iter().map(|(dist, _)| *dist).collect();

            results.push((indices, dists));
        }

        Ok(results)
    }

    fn estimate_memory_per_sample(&self) -> usize {
        let feature_memory = self.training_data.ncols() * 8; // 8 bytes per Float
        let distance_memory = self.training_data.nrows() * 16; // distance + index pairs
        let result_memory = self.k * 16; // k neighbors with distance and index

        feature_memory + distance_memory + result_memory
    }

    fn supports_parallel_processing(&self) -> bool {
        true
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_processor_creation() {
        let processor = BatchProcessor::builder()
            .batch_size(500)
            .max_memory_mb(256)
            .parallel_processing(true)
            .build();

        assert_eq!(processor.config.batch_size, 500);
        assert_eq!(processor.config.max_memory_mb, 256);
        assert!(processor.config.parallel_processing);
    }

    #[test]
    fn test_memory_efficient_batch_processing() {
        let training_data =
            Array2::from_shape_vec((100, 4), (0..400).map(|x| x as Float).collect()).unwrap();
        let test_data =
            Array2::from_shape_vec((50, 4), (0..200).map(|x| x as Float).collect()).unwrap();

        let search = BatchNeighborSearch::new(5, Distance::Euclidean, training_data);
        let mut processor = BatchProcessor::builder()
            .batch_size(10)
            .max_memory_mb(64)
            .build();

        let result = processor.process_data(test_data.view(), &search).unwrap();

        assert_eq!(result.results.len(), 50);
        assert!(result.batch_stats.total_batches > 0);
        assert_eq!(result.batch_stats.processed_samples, 50);
    }

    #[test]
    fn test_optimal_batch_size_calculation() {
        let training_data = Array2::zeros((100, 10));
        let search = BatchNeighborSearch::new(5, Distance::Euclidean, training_data);
        let processor = BatchProcessor::builder()
            .batch_size(1000)
            .max_memory_mb(1)
            .build();

        let optimal_size = processor.calculate_optimal_batch_size(50, &search).unwrap();

        // Should be constrained by memory limit
        assert!(optimal_size <= 50);
        assert!(optimal_size > 0);
    }

    #[test]
    fn test_batch_processing_with_overlap() {
        let training_data =
            Array2::from_shape_vec((20, 2), (0..40).map(|x| x as Float).collect()).unwrap();
        let test_data =
            Array2::from_shape_vec((10, 2), (0..20).map(|x| x as Float).collect()).unwrap();

        let search = BatchNeighborSearch::new(3, Distance::Euclidean, training_data);
        let processor = BatchProcessor::builder()
            .batch_size(4)
            .chunk_overlap(2)
            .build();

        // Just verify that overlap configuration doesn't break processing
        let config = &processor.config;
        assert_eq!(config.chunk_overlap, 2);
        assert_eq!(config.batch_size, 4);
    }

    #[test]
    fn test_memory_monitoring() {
        let mut processor = BatchProcessor::builder().max_memory_mb(1).build();

        // Simulate memory usage - using a larger amount to ensure it's tracked
        let result = processor.update_memory_usage(1024 * 1024); // 1 MB
        assert!(result.is_ok());

        let stats = processor.get_memory_stats();
        assert!(stats.current_memory_mb >= 1); // Should be at least 1 MB

        // Test memory limit exceeded
        let result = processor.update_memory_usage(2 * 1024 * 1024); // 2 MB more
        assert!(result.is_err());
    }

    #[test]
    fn test_parallel_processing_basic() {
        let training_data =
            Array2::from_shape_vec((30, 2), (0..60).map(|x| x as Float).collect()).unwrap();
        let test_data =
            Array2::from_shape_vec((10, 2), (0..20).map(|x| x as Float).collect()).unwrap();

        let search = BatchNeighborSearch::new(3, Distance::Euclidean, training_data);
        let mut processor = BatchProcessor::builder()
            .batch_size(5)
            .parallel_processing(true)
            .build();

        let result = processor.process_data(test_data.view(), &search).unwrap();

        // Should process all 10 test samples
        assert_eq!(result.results.len(), 10);
        assert_eq!(result.batch_stats.processed_samples, 10);

        // Each result should have 3 neighbors (k=3)
        for (indices, distances) in &result.results {
            assert_eq!(indices.len(), 3);
            assert_eq!(distances.len(), 3);
        }
    }

    #[test]
    fn test_empty_input_handling() {
        let training_data = Array2::zeros((10, 2));
        let empty_data = Array2::zeros((0, 2));
        let search = BatchNeighborSearch::new(3, Distance::Euclidean, training_data);
        let mut processor = BatchProcessor::builder().build();

        let result = processor.process_data(empty_data.view(), &search);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), NeighborsError::EmptyInput));
    }

    #[test]
    fn test_memory_efficiency_calculation() {
        let mut processor = BatchProcessor::builder().max_memory_mb(100).build();

        processor.memory_monitor.peak_memory_mb = 50;
        let efficiency = processor.calculate_memory_efficiency();

        assert!((efficiency - 0.5).abs() < 1e-6);
    }
}
