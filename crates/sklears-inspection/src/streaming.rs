//! Streaming explanation computation for large datasets
//!
//! This module provides streaming algorithms for explanation computation
//! that can handle datasets larger than memory by processing them in chunks.

use crate::memory::{CacheConfig, ExplanationCache};
use crate::types::*;
// ✅ SciRS2 Policy Compliant Imports
use scirs2_core::ndarray::{s, Array1, Array2, ArrayView2, Axis};
use sklears_core::prelude::SklearsError;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

/// Configuration for streaming explanation computation
#[derive(Clone, Debug)]
pub struct StreamingConfig {
    /// Chunk size for processing
    pub chunk_size: usize,
    /// Number of chunks to keep in memory
    pub memory_chunks: usize,
    /// Enable online aggregation
    pub online_aggregation: bool,
    /// Minimum chunk size to process
    pub min_chunk_size: usize,
    /// Maximum memory usage in MB
    pub max_memory_mb: usize,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            chunk_size: 1000,
            memory_chunks: 3,
            online_aggregation: true,
            min_chunk_size: 100,
            max_memory_mb: 512,
        }
    }
}

/// Streaming explanation processor
pub struct StreamingExplainer {
    /// Configuration
    config: StreamingConfig,
    /// Cache for repeated computations
    cache: Arc<ExplanationCache>,
    /// Chunk buffer
    chunk_buffer: Arc<Mutex<VecDeque<Array2<Float>>>>,
    /// Accumulated statistics
    stats: Arc<Mutex<StreamingStatistics>>,
}

/// Statistics for streaming computation
#[derive(Clone, Debug, Default)]
pub struct StreamingStatistics {
    /// Number of chunks processed
    pub chunks_processed: usize,
    /// Total samples processed
    pub total_samples: usize,
    /// Current memory usage in bytes
    pub current_memory_usage: usize,
    /// Peak memory usage in bytes
    pub peak_memory_usage: usize,
    /// Processing time per chunk
    pub avg_chunk_time: f64,
}

/// Streaming explanation result
#[derive(Clone, Debug)]
pub struct StreamingExplanationResult {
    /// Aggregated feature importance
    pub feature_importance: Array1<Float>,
    /// Confidence intervals
    pub confidence_intervals: Array2<Float>,
    /// Processing statistics
    pub statistics: StreamingStatistics,
    /// Number of chunks used
    pub chunks_used: usize,
}

/// Online aggregator for streaming results
pub struct OnlineAggregator {
    /// Running sum of feature importance
    running_sum: Array1<Float>,
    /// Running sum of squared values
    running_sum_squared: Array1<Float>,
    /// Number of observations
    count: usize,
    /// Number of features
    n_features: usize,
}

impl StreamingExplainer {
    /// Create a new streaming explainer
    pub fn new(config: StreamingConfig) -> Self {
        let cache_config = CacheConfig {
            max_cache_size_mb: config.max_memory_mb / 4, // Use 1/4 of memory for cache
            ..Default::default()
        };

        Self {
            config,
            cache: Arc::new(ExplanationCache::new(&cache_config)),
            chunk_buffer: Arc::new(Mutex::new(VecDeque::new())),
            stats: Arc::new(Mutex::new(StreamingStatistics::default())),
        }
    }

    /// Process data stream and compute explanations
    pub fn process_stream<F, I>(
        &self,
        data_stream: I,
        model: &F,
    ) -> crate::SklResult<StreamingExplanationResult>
    where
        F: Fn(&ArrayView2<Float>) -> crate::SklResult<Array1<Float>> + Sync + Send,
        I: Iterator<Item = Array2<Float>>,
    {
        let mut aggregator = None;
        let mut chunks_processed = 0;
        let start_time = std::time::Instant::now();

        for chunk in data_stream {
            if chunk.nrows() < self.config.min_chunk_size {
                continue;
            }

            // Initialize aggregator with first chunk
            if aggregator.is_none() {
                aggregator = Some(OnlineAggregator::new(chunk.ncols()));
            }

            // Process chunk
            let chunk_result = self.process_chunk(&chunk.view(), model)?;

            // Update aggregator
            if let Some(ref mut agg) = aggregator {
                agg.update(&chunk_result)?;
            }

            chunks_processed += 1;

            // Update statistics
            {
                let mut stats = self.stats.lock().unwrap();
                stats.chunks_processed = chunks_processed;
                stats.total_samples += chunk.nrows();
                stats.current_memory_usage = self.estimate_memory_usage();
                stats.peak_memory_usage = stats.peak_memory_usage.max(stats.current_memory_usage);
                stats.avg_chunk_time = start_time.elapsed().as_secs_f64() / chunks_processed as f64;
            }

            // Manage memory
            self.manage_memory()?;
        }

        // Finalize results
        let aggregator = aggregator
            .ok_or_else(|| SklearsError::InvalidInput("No valid chunks processed".to_string()))?;

        let (feature_importance, confidence_intervals) = aggregator.finalize();
        let statistics = self.stats.lock().unwrap().clone();

        Ok(StreamingExplanationResult {
            feature_importance,
            confidence_intervals,
            statistics,
            chunks_used: chunks_processed,
        })
    }

    /// Process a single chunk of data
    fn process_chunk<F>(
        &self,
        chunk: &ArrayView2<Float>,
        model: &F,
    ) -> crate::SklResult<Array1<Float>>
    where
        F: Fn(&ArrayView2<Float>) -> crate::SklResult<Array1<Float>>,
    {
        // Use cache-friendly computation
        crate::memory::cache_friendly_permutation_importance(
            chunk,
            &Array1::zeros(chunk.nrows()).view(), // Dummy y for feature importance
            model,
            &self.cache,
            &CacheConfig::default(),
        )
    }

    /// Estimate current memory usage
    fn estimate_memory_usage(&self) -> usize {
        let buffer_size = {
            let buffer = self.chunk_buffer.lock().unwrap();
            buffer
                .iter()
                .map(|chunk| chunk.len() * std::mem::size_of::<Float>())
                .sum::<usize>()
        };

        let cache_size = self.cache.get_statistics().total_size;

        buffer_size + cache_size
    }

    /// Manage memory usage by evicting old chunks
    fn manage_memory(&self) -> crate::SklResult<()> {
        let current_usage = self.estimate_memory_usage();
        let max_usage = self.config.max_memory_mb * 1024 * 1024;

        if current_usage > max_usage {
            // Evict oldest chunks
            let mut buffer = self.chunk_buffer.lock().unwrap();
            while !buffer.is_empty() && self.estimate_memory_usage() > max_usage {
                buffer.pop_front();
            }

            // Clear cache if still over limit
            if self.estimate_memory_usage() > max_usage {
                self.cache.clear_all();
            }
        }

        Ok(())
    }
}

impl OnlineAggregator {
    /// Create a new online aggregator
    pub fn new(n_features: usize) -> Self {
        Self {
            running_sum: Array1::zeros(n_features),
            running_sum_squared: Array1::zeros(n_features),
            count: 0,
            n_features,
        }
    }

    /// Update aggregator with new values
    pub fn update(&mut self, values: &Array1<Float>) -> crate::SklResult<()> {
        if values.len() != self.n_features {
            return Err(SklearsError::InvalidInput(
                "Feature dimension mismatch".to_string(),
            ));
        }

        // Update running sums
        self.running_sum += values;
        self.running_sum_squared += &values.mapv(|x| x * x);
        self.count += 1;

        Ok(())
    }

    /// Finalize aggregation and return mean and confidence intervals
    pub fn finalize(self) -> (Array1<Float>, Array2<Float>) {
        if self.count == 0 {
            return (
                Array1::zeros(self.n_features),
                Array2::zeros((self.n_features, 2)),
            );
        }

        let count_f = self.count as Float;
        let mean = &self.running_sum / count_f;

        // Compute standard deviation
        let variance = (&self.running_sum_squared / count_f) - mean.mapv(|x| x * x);
        let std_dev = variance.mapv(|x| x.sqrt());

        // Compute 95% confidence intervals
        let t_value = 1.96; // For large samples, approximating t-distribution with normal
        let stderr = &std_dev / (count_f.sqrt());
        let margin = &stderr * t_value;

        let mut confidence_intervals = Array2::zeros((self.n_features, 2));
        for i in 0..self.n_features {
            confidence_intervals[(i, 0)] = mean[i] - margin[i]; // Lower bound
            confidence_intervals[(i, 1)] = mean[i] + margin[i]; // Upper bound
        }

        (mean, confidence_intervals)
    }
}

/// Streaming SHAP computation
pub struct StreamingShapExplainer {
    /// Base configuration
    config: StreamingConfig,
    /// Sample buffer for baseline computation
    sample_buffer: Arc<Mutex<VecDeque<Array1<Float>>>>,
    /// Background statistics
    background_stats: Arc<Mutex<BackgroundStatistics>>,
}

/// Background statistics for SHAP computation
#[derive(Clone, Debug, Default)]
pub struct BackgroundStatistics {
    /// Feature means
    pub feature_means: Array1<Float>,
    /// Feature standard deviations
    pub feature_stds: Array1<Float>,
    /// Number of samples seen
    pub samples_seen: usize,
}

impl StreamingShapExplainer {
    /// Create a new streaming SHAP explainer
    pub fn new(config: StreamingConfig) -> Self {
        Self {
            config,
            sample_buffer: Arc::new(Mutex::new(VecDeque::new())),
            background_stats: Arc::new(Mutex::new(BackgroundStatistics::default())),
        }
    }

    /// Compute SHAP values for a stream of data
    pub fn compute_shap_stream<F, I>(
        &self,
        data_stream: I,
        model: &F,
    ) -> crate::SklResult<StreamingExplanationResult>
    where
        F: Fn(&ArrayView2<Float>) -> crate::SklResult<Array1<Float>> + Sync + Send,
        I: Iterator<Item = Array2<Float>>,
    {
        let mut aggregator = None;
        let mut chunks_processed = 0;

        for chunk in data_stream {
            if chunk.nrows() < self.config.min_chunk_size {
                continue;
            }

            // Update background statistics
            self.update_background_stats(&chunk.view())?;

            // Initialize aggregator
            if aggregator.is_none() {
                aggregator = Some(OnlineAggregator::new(chunk.ncols()));
            }

            // Compute SHAP values for chunk
            let shap_values = self.compute_chunk_shap(&chunk.view(), model)?;

            // Aggregate results
            if let Some(ref mut agg) = aggregator {
                let mean_shap = shap_values.mean_axis(Axis(0)).unwrap();
                agg.update(&mean_shap)?;
            }

            chunks_processed += 1;
        }

        // Finalize results
        let aggregator = aggregator
            .ok_or_else(|| SklearsError::InvalidInput("No valid chunks processed".to_string()))?;

        let (feature_importance, confidence_intervals) = aggregator.finalize();

        Ok(StreamingExplanationResult {
            feature_importance,
            confidence_intervals,
            statistics: StreamingStatistics {
                chunks_processed,
                total_samples: chunks_processed * self.config.chunk_size,
                ..Default::default()
            },
            chunks_used: chunks_processed,
        })
    }

    /// Update background statistics with new data
    fn update_background_stats(&self, chunk: &ArrayView2<Float>) -> crate::SklResult<()> {
        let mut stats = self.background_stats.lock().unwrap();

        if stats.samples_seen == 0 {
            // Initialize with first chunk
            stats.feature_means = chunk.mean_axis(Axis(0)).ok_or_else(|| {
                SklearsError::InvalidInput("Cannot compute feature means".to_string())
            })?;
            stats.feature_stds = chunk.std_axis(Axis(0), 0.0);
            stats.samples_seen = chunk.nrows();
        } else {
            // Update statistics incrementally
            let chunk_means = chunk.mean_axis(Axis(0)).ok_or_else(|| {
                SklearsError::InvalidInput("Cannot compute feature means".to_string())
            })?;

            let total_samples = stats.samples_seen + chunk.nrows();
            let weight_old = stats.samples_seen as Float / total_samples as Float;
            let weight_new = chunk.nrows() as Float / total_samples as Float;

            // Update means
            stats.feature_means = &stats.feature_means * weight_old + &chunk_means * weight_new;
            stats.samples_seen = total_samples;
        }

        Ok(())
    }

    /// Compute SHAP values for a single chunk
    fn compute_chunk_shap<F>(
        &self,
        chunk: &ArrayView2<Float>,
        model: &F,
    ) -> crate::SklResult<Array2<Float>>
    where
        F: Fn(&ArrayView2<Float>) -> crate::SklResult<Array1<Float>>,
    {
        let n_samples = chunk.nrows();
        let n_features = chunk.ncols();

        // Get background statistics
        let background_means = {
            let stats = self.background_stats.lock().unwrap();
            stats.feature_means.clone()
        };

        // Compute simplified SHAP values
        let mut shap_values = Array2::zeros((n_samples, n_features));

        for sample_idx in 0..n_samples {
            let sample = chunk.row(sample_idx);

            // Baseline prediction (using background means)
            let baseline_data = background_means.clone().insert_axis(Axis(0));
            let baseline_pred = model(&baseline_data.view())?;
            let baseline_value = baseline_pred[0];

            // Full prediction
            let full_pred = model(&sample.insert_axis(Axis(0)))?;
            let full_value = full_pred[0];

            // Compute marginal contributions
            let total_contribution = full_value - baseline_value;

            // Simple attribution: proportional to deviation from baseline
            let deviations = &sample.to_owned() - &background_means;
            let total_deviation = deviations.mapv(|x| x.abs()).sum();

            if total_deviation > 0.0 {
                for feature_idx in 0..n_features {
                    let feature_contrib = if total_deviation > 0.0 {
                        total_contribution * (deviations[feature_idx].abs() / total_deviation)
                    } else {
                        total_contribution / n_features as Float
                    };

                    shap_values[(sample_idx, feature_idx)] = feature_contrib;
                }
            }
        }

        Ok(shap_values)
    }
}

/// Utility function to create data chunks from large arrays
pub fn create_data_chunks(data: &ArrayView2<Float>, chunk_size: usize) -> Vec<Array2<Float>> {
    let mut chunks = Vec::new();
    let n_samples = data.nrows();

    for start in (0..n_samples).step_by(chunk_size) {
        let end = (start + chunk_size).min(n_samples);
        let chunk = data.slice(s![start..end, ..]).to_owned();
        chunks.push(chunk);
    }

    chunks
}

/// Streaming data iterator for file-based processing
pub struct StreamingDataIterator {
    /// Current position in data
    position: usize,
    /// Data source
    data: Array2<Float>,
    /// Chunk size
    chunk_size: usize,
}

impl StreamingDataIterator {
    /// Create a new streaming data iterator
    pub fn new(data: Array2<Float>, chunk_size: usize) -> Self {
        Self {
            position: 0,
            data,
            chunk_size,
        }
    }
}

impl Iterator for StreamingDataIterator {
    type Item = Array2<Float>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.position >= self.data.nrows() {
            return None;
        }

        let end = (self.position + self.chunk_size).min(self.data.nrows());
        let chunk = self.data.slice(s![self.position..end, ..]).to_owned();
        self.position = end;

        Some(chunk)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    // ✅ SciRS2 Policy Compliant Import
    use scirs2_core::ndarray::array;

    #[test]
    fn test_streaming_config_default() {
        let config = StreamingConfig::default();
        assert_eq!(config.chunk_size, 1000);
        assert_eq!(config.memory_chunks, 3);
        assert!(config.online_aggregation);
    }

    #[test]
    fn test_online_aggregator() {
        let mut aggregator = OnlineAggregator::new(2);

        // Add some values
        aggregator.update(&array![1.0, 2.0]).unwrap();
        aggregator.update(&array![3.0, 4.0]).unwrap();

        let (mean, confidence_intervals) = aggregator.finalize();

        assert_abs_diff_eq!(mean[0], 2.0, epsilon = 1e-6);
        assert_abs_diff_eq!(mean[1], 3.0, epsilon = 1e-6);
        assert_eq!(confidence_intervals.shape(), &[2, 2]);
    }

    #[test]
    fn test_streaming_explainer_creation() {
        let config = StreamingConfig::default();
        let explainer = StreamingExplainer::new(config);

        let stats = explainer.stats.lock().unwrap();
        assert_eq!(stats.chunks_processed, 0);
        assert_eq!(stats.total_samples, 0);
    }

    #[test]
    fn test_create_data_chunks() {
        let data = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
        let chunks = create_data_chunks(&data.view(), 2);

        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].nrows(), 2);
        assert_eq!(chunks[1].nrows(), 2);
    }

    #[test]
    fn test_streaming_data_iterator() {
        let data = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let mut iterator = StreamingDataIterator::new(data, 2);

        let chunk1 = iterator.next().unwrap();
        assert_eq!(chunk1.nrows(), 2);

        let chunk2 = iterator.next().unwrap();
        assert_eq!(chunk2.nrows(), 1);

        assert!(iterator.next().is_none());
    }

    #[test]
    fn test_streaming_shap_explainer() {
        let config = StreamingConfig::default();
        let explainer = StreamingShapExplainer::new(config);

        let stats = explainer.background_stats.lock().unwrap();
        assert_eq!(stats.samples_seen, 0);
    }

    #[test]
    fn test_background_statistics_update() {
        let config = StreamingConfig::default();
        let explainer = StreamingShapExplainer::new(config);

        let chunk = array![[1.0, 2.0], [3.0, 4.0]];
        explainer.update_background_stats(&chunk.view()).unwrap();

        let stats = explainer.background_stats.lock().unwrap();
        assert_eq!(stats.samples_seen, 2);
        assert_abs_diff_eq!(stats.feature_means[0], 2.0, epsilon = 1e-6);
        assert_abs_diff_eq!(stats.feature_means[1], 3.0, epsilon = 1e-6);
    }

    #[test]
    fn test_streaming_statistics_default() {
        let stats = StreamingStatistics::default();
        assert_eq!(stats.chunks_processed, 0);
        assert_eq!(stats.total_samples, 0);
        assert_eq!(stats.current_memory_usage, 0);
    }

    #[test]
    fn test_streaming_explanation_result() {
        let result = StreamingExplanationResult {
            feature_importance: array![0.5, 0.3],
            confidence_intervals: array![[0.4, 0.6], [0.2, 0.4]],
            statistics: StreamingStatistics::default(),
            chunks_used: 3,
        };

        assert_eq!(result.feature_importance.len(), 2);
        assert_eq!(result.confidence_intervals.shape(), &[2, 2]);
        assert_eq!(result.chunks_used, 3);
    }

    #[test]
    fn test_process_chunk_computation() {
        let config = StreamingConfig::default();
        let explainer = StreamingExplainer::new(config);

        let chunk = array![[1.0, 2.0], [3.0, 4.0]];
        let model =
            |_: &ArrayView2<Float>| -> crate::SklResult<Array1<Float>> { Ok(array![0.5, 0.7]) };

        let result = explainer.process_chunk(&chunk.view(), &model).unwrap();
        assert_eq!(result.len(), 2);
    }
}
