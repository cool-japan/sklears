//! Optimized Metric Implementations for High-Performance Computing
//!
//! This module provides performance-optimized implementations of metrics using various
//! acceleration techniques including SIMD vectorization, parallel processing, streaming
//! computation, chunked processing, sparse operations, and approximate algorithms.
//!
//! ## Architecture
//!
//! The optimized metrics are organized into 6 specialized domains:
//!
//! - **SIMD Operations**: Vectorized computations for element-wise operations
//! - **Parallel Processing**: Multi-threaded computation using Rayon
//! - **Streaming Processing**: Incremental computation for very large datasets
//! - **Chunked Processing**: Memory-efficient processing for large arrays
//! - **Sparse Operations**: Specialized algorithms for sparse data structures
//! - **Approximate Methods**: Trade accuracy for significant performance improvements
//!
//! ## Usage
//!
//! The module provides both individual optimized functions and unified selectors that
//! automatically choose the best implementation based on data characteristics and
//! configuration settings.

use crate::MetricsResult;
use scirs2_core::ndarray::Array1;
use scirs2_core::numeric::{Float as FloatTrait, FromPrimitive, Zero};
use std::hash::Hash;

/// Configuration for optimized metric implementations
#[derive(Debug, Clone)]
pub struct OptimizedConfig {
    /// Enable SIMD acceleration when available
    pub use_simd: bool,
    /// Minimum array size to use parallel processing
    pub parallel_threshold: usize,
    /// Chunk size for parallel processing
    pub chunk_size: usize,
    /// Enable streaming processing for very large datasets
    pub use_streaming: bool,
    /// Buffer size for streaming processing
    pub streaming_buffer_size: usize,
    /// Enable sparse optimizations
    pub use_sparse: bool,
    /// Enable approximate methods
    pub use_approximate: bool,
}

impl Default for OptimizedConfig {
    fn default() -> Self {
        Self {
            use_simd: true,
            parallel_threshold: 10000,
            chunk_size: 1024,
            use_streaming: false,
            streaming_buffer_size: 8192,
            use_sparse: false,
            use_approximate: false,
        }
    }
}

// Module declarations
pub mod approximate_methods;
pub mod chunked_processing;
pub mod parallel_processing;
pub mod simd_operations;
// TODO: Migrate to scirs2-sparse (uses sprs types)
// pub mod sparse_operations;
pub mod streaming_processing;

// Re-export SIMD operations
pub use simd_operations::{
    optimized_mean_absolute_error, optimized_mean_squared_error, optimized_r2_score,
};

#[cfg(all(feature = "simd", feature = "disabled-for-stability"))]
pub use simd_operations::{
    simd_cosine_similarity_f64, simd_euclidean_distance_f64, simd_mean_absolute_error_f32,
    simd_mean_absolute_error_f64, simd_mean_squared_error_f64, simd_r2_score_f64,
};

// Re-export parallel processing functions
#[cfg(feature = "parallel")]
pub use parallel_processing::{
    parallel_accuracy, parallel_cosine_similarity, parallel_mean_absolute_error,
    parallel_mean_squared_error, parallel_r2_score,
};

// Re-export streaming processing structures
pub use streaming_processing::{
    ClassificationMetrics, IncrementalMetrics, RegressionMetrics, StreamingConfusionMatrix,
    StreamingMetrics,
};

// Re-export chunked processing
pub use chunked_processing::{ChunkedMetricProcessor, ChunkedRegressionMetrics};

// Re-export sparse operations
// TODO: Migrate to scirs2-sparse (uses sprs types)
// pub use sparse_operations::{SparseClassificationMetrics, SparseConfusionMatrix};

// #[cfg(feature = "sparse")]
// pub use sparse_operations::SparseMetrics;

// Re-export approximate methods
pub use approximate_methods::{
    ApproximateConfig, ApproximateConfusionMatrix, ApproximateHistogram, CountMinSketch,
    QuantileSketch, ReservoirSampler, SamplingMetrics,
};

// Memory-mapped confusion matrix placeholder (feature-gated)
#[cfg(feature = "mmap")]
pub struct MmapConfusionMatrix {
    _phantom: std::marker::PhantomData<()>,
}

/// Comprehensive optimized metrics factory
pub struct OptimizedMetricsFactory {
    config: OptimizedConfig,
}

impl OptimizedMetricsFactory {
    pub fn new(config: OptimizedConfig) -> Self {
        Self { config }
    }

    /// Create streaming metrics processor
    pub fn streaming_metrics<F: FloatTrait + FromPrimitive + Zero>(&self) -> StreamingMetrics<F> {
        StreamingMetrics::new(self.config.clone())
    }

    /// Create incremental metrics processor
    pub fn incremental_metrics<F: FloatTrait + FromPrimitive + Zero + Copy>(
        &self,
    ) -> IncrementalMetrics<F> {
        IncrementalMetrics::new(self.config.clone())
    }

    /// Create chunked metrics processor
    pub fn chunked_processor<F: FloatTrait + FromPrimitive + Send + Sync>(
        &self,
    ) -> ChunkedMetricProcessor<F> {
        ChunkedMetricProcessor::new(self.config.clone())
    }

    // TODO: Migrate to scirs2-sparse (uses sprs types)
    // /// Create sparse confusion matrix
    // pub fn sparse_confusion_matrix<T: PartialEq + Copy + Ord + Hash>(
    //     &self,
    // ) -> SparseConfusionMatrix<T> {
    //     SparseConfusionMatrix::new()
    // }

    /// Create streaming confusion matrix
    pub fn streaming_confusion_matrix<T: PartialEq + Copy + Ord + Hash>(
        &self,
    ) -> StreamingConfusionMatrix<T> {
        StreamingConfusionMatrix::new(self.config.clone())
    }

    /// Create approximate confusion matrix
    pub fn approximate_confusion_matrix<T: PartialEq + Copy + Ord + Hash>(
        &self,
        sketch_width: usize,
        sketch_depth: usize,
        threshold: usize,
    ) -> ApproximateConfusionMatrix<T> {
        ApproximateConfusionMatrix::new(sketch_width, sketch_depth, threshold)
    }

    /// Create sampling metrics with approximate config
    pub fn sampling_metrics<F: FloatTrait + FromPrimitive + Clone>(
        &self,
        approx_config: ApproximateConfig,
    ) -> SamplingMetrics<F> {
        SamplingMetrics::new(approx_config)
    }
}

impl Default for OptimizedMetricsFactory {
    fn default() -> Self {
        Self::new(OptimizedConfig::default())
    }
}

/// Builder for optimized metrics configuration
pub struct OptimizedConfigBuilder {
    config: OptimizedConfig,
}

impl OptimizedConfigBuilder {
    pub fn new() -> Self {
        Self {
            config: OptimizedConfig::default(),
        }
    }

    pub fn parallel_threshold(mut self, threshold: usize) -> Self {
        self.config.parallel_threshold = threshold;
        self
    }

    pub fn chunk_size(mut self, size: usize) -> Self {
        self.config.chunk_size = size;
        self
    }

    pub fn use_simd(mut self, enabled: bool) -> Self {
        self.config.use_simd = enabled;
        self
    }

    pub fn streaming_buffer_size(mut self, size: usize) -> Self {
        self.config.streaming_buffer_size = size;
        self
    }

    pub fn build(self) -> OptimizedConfig {
        self.config
    }
}

impl Default for OptimizedConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience functions for quick access to optimized metrics
pub mod convenience {
    use super::*;

    /// Quick access to optimized MAE with default config
    pub fn mae<
        F: FloatTrait
            + FromPrimitive
            + Send
            + Sync
            + 'static
            + std::iter::Sum<F>
            + for<'a> std::iter::Sum<&'a F>,
    >(
        y_true: &Array1<F>,
        y_pred: &Array1<F>,
    ) -> MetricsResult<F> {
        optimized_mean_absolute_error(y_true, y_pred, None)
    }

    /// Quick access to optimized MSE with default config
    pub fn mse<
        F: FloatTrait
            + FromPrimitive
            + Send
            + Sync
            + 'static
            + std::iter::Sum<F>
            + for<'a> std::iter::Sum<&'a F>,
    >(
        y_true: &Array1<F>,
        y_pred: &Array1<F>,
    ) -> MetricsResult<F> {
        optimized_mean_squared_error(y_true, y_pred, None)
    }

    /// Quick access to optimized R² with default config
    pub fn r2<
        F: FloatTrait
            + FromPrimitive
            + Send
            + Sync
            + 'static
            + std::iter::Sum<F>
            + for<'a> std::iter::Sum<&'a F>,
    >(
        y_true: &Array1<F>,
        y_pred: &Array1<F>,
    ) -> MetricsResult<F> {
        optimized_r2_score(y_true, y_pred, None)
    }

    /// Quick access to chunked MAE processing
    pub fn chunked_mae<F: FloatTrait + FromPrimitive + Send + Sync>(
        y_true: &Array1<F>,
        y_pred: &Array1<F>,
        chunk_size: Option<usize>,
    ) -> MetricsResult<F> {
        let config = OptimizedConfig {
            chunk_size: chunk_size.unwrap_or(1024),
            ..OptimizedConfig::default()
        };
        let processor = ChunkedMetricProcessor::new(config);
        processor.chunked_mean_absolute_error(y_true, y_pred)
    }

    /// Quick access to streaming metrics
    pub fn streaming<F: FloatTrait + FromPrimitive + Zero>() -> StreamingMetrics<F> {
        StreamingMetrics::new(OptimizedConfig::default())
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_optimized_config_builder() {
        let config = OptimizedConfigBuilder::new()
            .parallel_threshold(1000)
            .chunk_size(512)
            .use_simd(false)
            .build();

        assert_eq!(config.parallel_threshold, 1000);
        assert_eq!(config.chunk_size, 512);
        assert!(!config.use_simd);
    }

    #[test]
    fn test_optimized_metrics_factory() {
        let config = OptimizedConfig::default();
        let factory = OptimizedMetricsFactory::new(config);

        let streaming_metrics: StreamingMetrics<f64> = factory.streaming_metrics();
        let _incremental_metrics: IncrementalMetrics<f64> = factory.incremental_metrics();
        let _chunked_processor: ChunkedMetricProcessor<f64> = factory.chunked_processor();
        // let _sparse_matrix: SparseConfusionMatrix<i32> = factory.sparse_confusion_matrix();

        assert_eq!(streaming_metrics.n_samples(), 0);
    }

    #[test]
    fn test_convenience_functions() {
        let y_true = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y_pred = array![1.1, 1.9, 3.1, 3.9, 5.1];

        let mae = convenience::mae(&y_true, &y_pred).unwrap();
        let mse = convenience::mse(&y_true, &y_pred).unwrap();
        let r2 = convenience::r2(&y_true, &y_pred).unwrap();

        assert!(mae > 0.0);
        assert!(mse > 0.0);
        assert!(r2 > 0.5); // Should have good correlation

        let chunked_mae = convenience::chunked_mae(&y_true, &y_pred, Some(2)).unwrap();
        assert_relative_eq!(mae, chunked_mae, epsilon = 1e-10);
    }

    #[test]
    fn test_module_integration() {
        // Test that all modules work together
        let config = OptimizedConfig::default();
        let factory = OptimizedMetricsFactory::new(config);

        // Test streaming + incremental
        let mut streaming: StreamingMetrics<f64> = factory.streaming_metrics();
        let mut incremental: IncrementalMetrics<f64> = factory.incremental_metrics();

        let y_true = array![1.0, 2.0];
        let y_pred = array![1.1, 2.1];

        streaming.update_batch(&y_true, &y_pred).unwrap();
        incremental.update_batch(&y_true, &y_pred).unwrap();

        let streaming_mae = streaming.mean_absolute_error().unwrap();
        let incremental_mae = incremental.mean_absolute_error().unwrap();

        assert_relative_eq!(streaming_mae, incremental_mae, epsilon = 1e-10);
    }
}
