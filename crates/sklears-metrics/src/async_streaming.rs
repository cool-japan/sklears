//! Async Streaming Metrics
//!
//! This module provides async/await support for streaming metrics computation,
//! enabling real-time metric evaluation for large datasets and streaming applications.
//!
//! # Features
//!
//! - Async streaming metric computation with backpressure handling
//! - Real-time metric updates for streaming data
//! - Chunked processing for memory-efficient large dataset handling
//! - Concurrent metric computation with controlled parallelism
//! - Stream-based metric aggregation and windowing
//! - Integration with async ecosystems (tokio, async-std)
//!
//! # Examples
//!
//! ```rust
//! use sklears_metrics::async_streaming::*;
//! use tokio_stream::{self as stream, StreamExt};
//! use scirs2_core::ndarray::Array1;
//!
//! #[tokio::main]
//! async fn main() {
//!     // Create streaming data
//!     let data_stream = stream::iter(vec![
//!         (Array1::from_vec(vec![0, 1, 1]), Array1::from_vec(vec![0, 1, 0])),
//!         (Array1::from_vec(vec![0, 1, 0]), Array1::from_vec(vec![0, 0, 1])),
//!     ]);
//!
//!     // Compute streaming metrics
//!     let mut streaming_computer = StreamingMetricsComputer::new()
//!         .with_chunk_size(1000)
//!         .with_metric("accuracy")
//!         .with_metric("f1_score");
//!
//!     let results = streaming_computer
//!         .compute_stream(data_stream)
//!         .await
//!         .unwrap();
//!
//!     println!("Final accuracy: {:.3}", results.get("accuracy").unwrap());
//! }
//! ```

use crate::{
    fluent_api::{MetricResults, ResultMetadata},
    MetricsError, MetricsResult,
};
use futures::stream::{Stream, StreamExt};
use scirs2_core::ndarray::Array1;
use std::collections::HashMap;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};
use tokio::sync::{mpsc, Semaphore};
use tokio::task;

/// Configuration for async streaming metrics
#[derive(Debug, Clone)]
pub struct StreamingConfig {
    /// Chunk size for batch processing
    pub chunk_size: usize,
    /// Maximum number of concurrent metric computations
    pub max_concurrency: usize,
    /// Buffer size for streaming data
    pub buffer_size: usize,
    /// Whether to compute incremental metrics
    pub incremental: bool,
    /// Window size for sliding window metrics
    pub window_size: Option<usize>,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            chunk_size: 1000,
            max_concurrency: 4,
            buffer_size: 100,
            incremental: true,
            window_size: None,
        }
    }
}

/// Streaming metrics computer with async support
#[derive(Debug)]
pub struct StreamingMetricsComputer {
    /// Metrics to compute
    metrics: Vec<String>,
    /// Configuration
    config: StreamingConfig,
    /// Accumulated state for incremental computation
    accumulator: Arc<Mutex<MetricAccumulator>>,
}

/// Accumulator for incremental metric computation
#[derive(Debug, Clone)]
pub struct MetricAccumulator {
    /// Total samples processed
    total_samples: usize,
    /// Running sums for various metrics
    correct_predictions: usize,
    true_positives: usize,
    false_positives: usize,
    false_negatives: usize,
    /// Sample buffer for windowed metrics
    sample_buffer: Vec<(i32, i32)>, // (true, pred) pairs
}

impl MetricAccumulator {
    fn new() -> Self {
        Self {
            total_samples: 0,
            correct_predictions: 0,
            true_positives: 0,
            false_positives: 0,
            false_negatives: 0,
            sample_buffer: Vec::new(),
        }
    }

    fn update(&mut self, y_true: &Array1<i32>, y_pred: &Array1<i32>, window_size: Option<usize>) {
        for (&true_val, &pred_val) in y_true.iter().zip(y_pred.iter()) {
            self.total_samples += 1;

            if true_val == pred_val {
                self.correct_predictions += 1;
            }

            // For binary classification (assuming 1 is positive class)
            if true_val == 1 && pred_val == 1 {
                self.true_positives += 1;
            } else if true_val == 0 && pred_val == 1 {
                self.false_positives += 1;
            } else if true_val == 1 && pred_val == 0 {
                self.false_negatives += 1;
            }

            // Maintain sliding window if specified
            if let Some(window_size) = window_size {
                self.sample_buffer.push((true_val, pred_val));
                if self.sample_buffer.len() > window_size {
                    // Remove oldest sample and update counters
                    let (old_true, old_pred) = self.sample_buffer.remove(0);
                    self.total_samples -= 1;

                    if old_true == old_pred {
                        self.correct_predictions -= 1;
                    }

                    if old_true == 1 && old_pred == 1 {
                        self.true_positives -= 1;
                    } else if old_true == 0 && old_pred == 1 {
                        self.false_positives -= 1;
                    } else if old_true == 1 && old_pred == 0 {
                        self.false_negatives -= 1;
                    }
                }
            }
        }
    }

    fn get_metric(&self, metric_name: &str) -> MetricsResult<f64> {
        match metric_name {
            "accuracy" => {
                if self.total_samples == 0 {
                    return Ok(0.0);
                }
                Ok(self.correct_predictions as f64 / self.total_samples as f64)
            }
            "precision" => {
                let denominator = self.true_positives + self.false_positives;
                if denominator == 0 {
                    return Ok(0.0);
                }
                Ok(self.true_positives as f64 / denominator as f64)
            }
            "recall" => {
                let denominator = self.true_positives + self.false_negatives;
                if denominator == 0 {
                    return Ok(0.0);
                }
                Ok(self.true_positives as f64 / denominator as f64)
            }
            "f1_score" => {
                let precision = self.get_metric("precision")?;
                let recall = self.get_metric("recall")?;
                if precision + recall == 0.0 {
                    return Ok(0.0);
                }
                Ok(2.0 * (precision * recall) / (precision + recall))
            }
            _ => Err(MetricsError::InvalidParameter(format!(
                "Unknown metric: {}",
                metric_name
            ))),
        }
    }
}

impl StreamingMetricsComputer {
    /// Create a new streaming metrics computer
    pub fn new() -> Self {
        Self {
            metrics: Vec::new(),
            config: StreamingConfig::default(),
            accumulator: Arc::new(Mutex::new(MetricAccumulator::new())),
        }
    }

    /// Set chunk size for batch processing
    pub fn with_chunk_size(mut self, chunk_size: usize) -> Self {
        self.config.chunk_size = chunk_size;
        self
    }

    /// Set maximum concurrency
    pub fn with_max_concurrency(mut self, max_concurrency: usize) -> Self {
        self.config.max_concurrency = max_concurrency;
        self
    }

    /// Set buffer size
    pub fn with_buffer_size(mut self, buffer_size: usize) -> Self {
        self.config.buffer_size = buffer_size;
        self
    }

    /// Enable incremental computation
    pub fn with_incremental(mut self, incremental: bool) -> Self {
        self.config.incremental = incremental;
        self
    }

    /// Set window size for sliding window metrics
    pub fn with_window_size(mut self, window_size: usize) -> Self {
        self.config.window_size = Some(window_size);
        self
    }

    /// Add a metric to compute
    pub fn with_metric(mut self, metric_name: &str) -> Self {
        self.metrics.push(metric_name.to_string());
        self
    }

    /// Compute metrics on a stream of (y_true, y_pred) pairs
    pub async fn compute_stream<S>(&mut self, stream: S) -> MetricsResult<MetricResults>
    where
        S: Stream<Item = (Array1<i32>, Array1<i32>)> + Send,
    {
        if self.metrics.is_empty() {
            return Err(MetricsError::InvalidParameter(
                "No metrics specified".to_string(),
            ));
        }

        let semaphore = Arc::new(Semaphore::new(self.config.max_concurrency));
        let accumulator = Arc::clone(&self.accumulator);
        let config = self.config.clone();
        let window_size = config.window_size;

        // Process the stream
        let mut pinned_stream = Box::pin(stream);
        let mut chunk_buffer = Vec::new();

        while let Some((y_true, y_pred)) = pinned_stream.next().await {
            if y_true.len() != y_pred.len() {
                return Err(MetricsError::ShapeMismatch {
                    expected: vec![y_true.len()],
                    actual: vec![y_pred.len()],
                });
            }

            chunk_buffer.push((y_true, y_pred));

            // Process chunk when buffer is full
            if chunk_buffer.len() >= self.config.chunk_size {
                self.process_chunk(
                    chunk_buffer.clone(),
                    Arc::clone(&accumulator),
                    window_size,
                    Arc::clone(&semaphore),
                )
                .await?;
                chunk_buffer.clear();
            }
        }

        // Process remaining items in buffer
        if !chunk_buffer.is_empty() {
            self.process_chunk(
                chunk_buffer,
                Arc::clone(&accumulator),
                window_size,
                Arc::clone(&semaphore),
            )
            .await?;
        }

        // Collect final results
        self.get_current_results().await
    }

    /// Process a chunk of data asynchronously
    async fn process_chunk(
        &self,
        chunk: Vec<(Array1<i32>, Array1<i32>)>,
        accumulator: Arc<Mutex<MetricAccumulator>>,
        window_size: Option<usize>,
        semaphore: Arc<Semaphore>,
    ) -> MetricsResult<()> {
        let _permit = semaphore.acquire().await.map_err(|_| {
            MetricsError::InvalidInput("Failed to acquire semaphore permit".to_string())
        })?;

        // Process chunk in background task
        task::spawn_blocking(move || {
            for (y_true, y_pred) in chunk {
                let mut acc = accumulator.lock().unwrap();
                acc.update(&y_true, &y_pred, window_size);
            }
        })
        .await
        .map_err(|_| MetricsError::InvalidInput("Task execution failed".to_string()))?;

        Ok(())
    }

    /// Get current metric results
    pub async fn get_current_results(&self) -> MetricsResult<MetricResults> {
        let accumulator = self.accumulator.lock().unwrap();
        let mut values = HashMap::new();

        for metric_name in &self.metrics {
            let value = accumulator.get_metric(metric_name)?;
            values.insert(metric_name.clone(), value);
        }

        let metadata = ResultMetadata {
            metrics_computed: self.metrics.clone(),
            averaging_strategy: "Streaming".to_string(),
            has_confidence_intervals: false,
            sample_size: accumulator.total_samples,
            timestamp: "2025-07-04T00:00:00Z".to_string(), // Would use actual timestamp
            config_summary: format!(
                "Streaming computation, chunk_size: {}, window_size: {:?}",
                self.config.chunk_size, self.config.window_size
            ),
        };

        Ok(MetricResults {
            values,
            confidence_intervals: HashMap::new(),
            metadata,
        })
    }

    /// Reset accumulator state
    pub async fn reset(&mut self) {
        let mut accumulator = self.accumulator.lock().unwrap();
        *accumulator = MetricAccumulator::new();
    }
}

impl Default for StreamingMetricsComputer {
    fn default() -> Self {
        Self::new()
    }
}

/// Stream wrapper for real-time metric computation
pub struct MetricStream<S> {
    inner: S,
    computer: StreamingMetricsComputer,
    current_results: Option<MetricResults>,
}

impl<S> MetricStream<S>
where
    S: Stream<Item = (Array1<i32>, Array1<i32>)> + Unpin,
{
    /// Create a new metric stream
    pub fn new(stream: S, metrics: Vec<String>) -> Self {
        let mut computer = StreamingMetricsComputer::new();
        for metric in metrics {
            computer = computer.with_metric(&metric);
        }

        Self {
            inner: stream,
            computer,
            current_results: None,
        }
    }

    /// Get current metric results
    pub async fn current_metrics(&self) -> MetricsResult<Option<MetricResults>> {
        if self.current_results.is_some() {
            Ok(self.current_results.clone())
        } else {
            Ok(None)
        }
    }
}

impl<S> Stream for MetricStream<S>
where
    S: Stream<Item = (Array1<i32>, Array1<i32>)> + Unpin,
{
    type Item = MetricsResult<MetricResults>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        match Pin::new(&mut self.inner).poll_next(cx) {
            Poll::Ready(Some((y_true, y_pred))) => {
                // Update accumulator with new data
                let accumulator = Arc::clone(&self.computer.accumulator);
                {
                    let mut acc = accumulator.lock().unwrap();
                    acc.update(&y_true, &y_pred, self.computer.config.window_size);
                }

                // Get current results
                let mut values = HashMap::new();
                {
                    let acc = accumulator.lock().unwrap();
                    for metric_name in &self.computer.metrics {
                        if let Ok(value) = acc.get_metric(metric_name) {
                            values.insert(metric_name.clone(), value);
                        }
                    }
                }

                let metadata = ResultMetadata {
                    metrics_computed: self.computer.metrics.clone(),
                    averaging_strategy: "Real-time streaming".to_string(),
                    has_confidence_intervals: false,
                    sample_size: {
                        let acc = accumulator.lock().unwrap();
                        acc.total_samples
                    },
                    timestamp: "2025-07-04T00:00:00Z".to_string(),
                    config_summary: "Real-time metric computation".to_string(),
                };

                let results = MetricResults {
                    values,
                    confidence_intervals: HashMap::new(),
                    metadata,
                };

                self.current_results = Some(results.clone());
                Poll::Ready(Some(Ok(results)))
            }
            Poll::Ready(None) => Poll::Ready(None),
            Poll::Pending => Poll::Pending,
        }
    }
}

/// Convenience function for creating a streaming accuracy computer
pub fn streaming_accuracy() -> StreamingMetricsComputer {
    StreamingMetricsComputer::new()
        .with_metric("accuracy")
        .with_incremental(true)
}

/// Convenience function for creating a streaming classification metrics computer
pub fn streaming_classification_metrics() -> StreamingMetricsComputer {
    StreamingMetricsComputer::new()
        .with_metric("accuracy")
        .with_metric("precision")
        .with_metric("recall")
        .with_metric("f1_score")
        .with_incremental(true)
}

/// Async channel-based metric computation
pub struct ChannelMetricsComputer {
    sender: mpsc::Sender<(Array1<i32>, Array1<i32>)>,
    receiver: mpsc::Receiver<MetricsResult<MetricResults>>,
}

impl ChannelMetricsComputer {
    /// Create a new channel-based metrics computer
    pub fn new(metrics: Vec<String>, buffer_size: usize) -> Self {
        let (data_tx, mut data_rx) = mpsc::channel::<(Array1<i32>, Array1<i32>)>(buffer_size);
        let (result_tx, result_rx) = mpsc::channel::<MetricsResult<MetricResults>>(buffer_size);

        // Spawn background task for metric computation
        let metrics_clone = metrics.clone();
        tokio::spawn(async move {
            let mut computer = StreamingMetricsComputer::new();
            for metric in metrics_clone {
                computer = computer.with_metric(&metric);
            }

            while let Some((y_true, y_pred)) = data_rx.recv().await {
                let accumulator = Arc::clone(&computer.accumulator);
                {
                    let mut acc = accumulator.lock().unwrap();
                    acc.update(&y_true, &y_pred, computer.config.window_size);
                }

                if let Ok(results) = computer.get_current_results().await {
                    let _ = result_tx.send(Ok(results)).await;
                }
            }
        });

        Self {
            sender: data_tx,
            receiver: result_rx,
        }
    }

    /// Send data for metric computation
    pub async fn send_data(
        &self,
        y_true: Array1<i32>,
        y_pred: Array1<i32>,
    ) -> Result<(), mpsc::error::SendError<(Array1<i32>, Array1<i32>)>> {
        self.sender.send((y_true, y_pred)).await
    }

    /// Receive computed metrics
    pub async fn recv_results(&mut self) -> Option<MetricsResult<MetricResults>> {
        self.receiver.recv().await
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use futures::stream;

    #[tokio::test]
    async fn test_streaming_metrics_computer() {
        let data = vec![
            (
                Array1::from_vec(vec![0, 1, 1]),
                Array1::from_vec(vec![0, 1, 0]),
            ),
            (
                Array1::from_vec(vec![0, 1, 0]),
                Array1::from_vec(vec![0, 0, 1]),
            ),
        ];

        let data_stream = stream::iter(data);

        let mut computer = StreamingMetricsComputer::new()
            .with_chunk_size(1)
            .with_metric("accuracy")
            .with_metric("f1_score");

        let results = computer.compute_stream(data_stream).await.unwrap();

        assert!(results.contains("accuracy"));
        assert!(results.contains("f1_score"));
        assert_eq!(results.metadata.sample_size, 6);
    }

    #[tokio::test]
    async fn test_windowed_streaming() {
        let data = vec![
            (Array1::from_vec(vec![1]), Array1::from_vec(vec![1])),
            (Array1::from_vec(vec![1]), Array1::from_vec(vec![1])),
            (Array1::from_vec(vec![0]), Array1::from_vec(vec![0])),
            (Array1::from_vec(vec![0]), Array1::from_vec(vec![1])), // Should affect windowed accuracy
        ];

        let data_stream = stream::iter(data);

        let mut computer = StreamingMetricsComputer::new()
            .with_chunk_size(1)
            .with_window_size(3)
            .with_metric("accuracy");

        let results = computer.compute_stream(data_stream).await.unwrap();

        assert!(results.contains("accuracy"));
        assert_eq!(results.metadata.sample_size, 3); // Window size
    }

    #[tokio::test]
    async fn test_metric_stream() {
        let data = vec![
            (Array1::from_vec(vec![0, 1]), Array1::from_vec(vec![0, 1])),
            (Array1::from_vec(vec![1, 0]), Array1::from_vec(vec![1, 1])),
        ];

        let data_stream = stream::iter(data);
        let metrics = vec!["accuracy".to_string(), "precision".to_string()];
        let mut metric_stream = MetricStream::new(data_stream, metrics);

        let mut results_count = 0;
        while let Some(result) = metric_stream.next().await {
            let metrics_result = result.unwrap();
            assert!(metrics_result.contains("accuracy"));
            assert!(metrics_result.contains("precision"));
            results_count += 1;
        }

        assert_eq!(results_count, 2);
    }

    #[tokio::test]
    async fn test_convenience_functions() {
        let data = vec![(Array1::from_vec(vec![0, 1]), Array1::from_vec(vec![0, 1]))];

        let data_stream = stream::iter(data);

        let mut computer = streaming_accuracy();
        let results = computer.compute_stream(data_stream).await.unwrap();

        assert!(results.contains("accuracy"));
        assert_eq!(results.get("accuracy"), Some(1.0));
    }

    #[tokio::test]
    async fn test_channel_metrics_computer() {
        let metrics = vec!["accuracy".to_string()];
        let mut computer = ChannelMetricsComputer::new(metrics, 10);

        // Send data
        let y_true = Array1::from_vec(vec![0, 1, 1]);
        let y_pred = Array1::from_vec(vec![0, 1, 0]);
        computer.send_data(y_true, y_pred).await.unwrap();

        // Receive results
        if let Some(Ok(results)) = computer.recv_results().await {
            assert!(results.contains("accuracy"));
        }
    }

    #[tokio::test]
    async fn test_reset_functionality() {
        let data = vec![(Array1::from_vec(vec![0, 1]), Array1::from_vec(vec![0, 1]))];

        let data_stream = stream::iter(data);

        let mut computer = StreamingMetricsComputer::new().with_metric("accuracy");

        let _ = computer.compute_stream(data_stream).await.unwrap();

        // Reset and check
        computer.reset().await;
        let results = computer.get_current_results().await.unwrap();
        assert_eq!(results.metadata.sample_size, 0);
    }

    #[tokio::test]
    async fn test_error_handling() {
        let data = vec![
            (Array1::from_vec(vec![0, 1]), Array1::from_vec(vec![0])), // Mismatched length
        ];

        let data_stream = stream::iter(data);

        let mut computer = StreamingMetricsComputer::new().with_metric("accuracy");

        let result = computer.compute_stream(data_stream).await;
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            MetricsError::ShapeMismatch { .. }
        ));
    }
}
