//! Streaming pipeline components for real-time data processing
//!
//! This module provides streaming capabilities including windowing strategies,
//! online model updates, incremental processing, and real-time analytics.

use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use sklears_core::{
    error::Result as SklResult,
    prelude::{Predict, SklearsError},
    traits::{Estimator, Fit, Untrained},
    types::Float,
};
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant, SystemTime};

use crate::{PipelinePredictor, PipelineStep};

/// Data point in a stream
#[derive(Debug, Clone)]
pub struct StreamDataPoint {
    /// Feature values
    pub features: Array1<f64>,
    /// Target value (optional)
    pub target: Option<f64>,
    /// Timestamp
    pub timestamp: SystemTime,
    /// Metadata
    pub metadata: HashMap<String, String>,
    /// Data point ID
    pub id: String,
}

impl StreamDataPoint {
    /// Create a new stream data point
    #[must_use]
    pub fn new(features: Array1<f64>, id: String) -> Self {
        Self {
            features,
            target: None,
            timestamp: SystemTime::now(),
            metadata: HashMap::new(),
            id,
        }
    }

    /// Set target value
    #[must_use]
    pub fn with_target(mut self, target: f64) -> Self {
        self.target = Some(target);
        self
    }

    /// Set timestamp
    #[must_use]
    pub fn with_timestamp(mut self, timestamp: SystemTime) -> Self {
        self.timestamp = timestamp;
        self
    }

    /// Set metadata
    #[must_use]
    pub fn with_metadata(mut self, metadata: HashMap<String, String>) -> Self {
        self.metadata = metadata;
        self
    }
}

/// Window of stream data points
#[derive(Debug, Clone)]
pub struct StreamWindow {
    /// Data points in the window
    pub data_points: Vec<StreamDataPoint>,
    /// Window start time
    pub start_time: SystemTime,
    /// Window end time
    pub end_time: SystemTime,
    /// Window metadata
    pub metadata: HashMap<String, String>,
}

impl StreamWindow {
    /// Create a new stream window
    #[must_use]
    pub fn new(start_time: SystemTime, end_time: SystemTime) -> Self {
        Self {
            data_points: Vec::new(),
            start_time,
            end_time,
            metadata: HashMap::new(),
        }
    }

    /// Add a data point to the window
    pub fn add_point(&mut self, point: StreamDataPoint) {
        self.data_points.push(point);
    }

    /// Get features matrix
    pub fn features_matrix(&self) -> SklResult<Array2<f64>> {
        if self.data_points.is_empty() {
            return Err(SklearsError::InvalidInput("Empty window".to_string()));
        }

        let n_samples = self.data_points.len();
        let n_features = self.data_points[0].features.len();

        let mut features = Array2::zeros((n_samples, n_features));
        for (i, point) in self.data_points.iter().enumerate() {
            features.row_mut(i).assign(&point.features);
        }

        Ok(features)
    }

    /// Get targets array
    #[must_use]
    pub fn targets_array(&self) -> Option<Array1<f64>> {
        if self.data_points.iter().all(|p| p.target.is_some()) {
            Some(Array1::from_vec(
                self.data_points.iter().map(|p| p.target.unwrap()).collect(),
            ))
        } else {
            None
        }
    }

    /// Get window size
    #[must_use]
    pub fn size(&self) -> usize {
        self.data_points.len()
    }

    /// Check if window is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data_points.is_empty()
    }
}

/// Windowing strategy for stream processing
pub enum WindowingStrategy {
    /// Fixed time windows
    TumblingTime {
        /// Window duration
        duration: Duration,
    },
    /// Sliding time windows
    SlidingTime {
        /// Window duration
        duration: Duration,
        /// Slide interval
        slide: Duration,
    },
    /// Fixed count windows
    TumblingCount {
        /// Number of elements per window
        count: usize,
    },
    /// Sliding count windows
    SlidingCount {
        /// Window size
        size: usize,
        /// Slide step
        step: usize,
    },
    /// Session windows (gap-based)
    Session {
        /// Maximum gap between elements
        gap: Duration,
    },
    /// Custom windowing
    Custom {
        /// Custom window trigger function
        trigger_fn: Box<dyn Fn(&[StreamDataPoint]) -> bool + Send + Sync>,
    },
}

/// Stream processing configuration
pub struct StreamConfig {
    /// Windowing strategy
    pub windowing: WindowingStrategy,
    /// Buffer size for incoming data
    pub buffer_size: usize,
    /// Processing parallelism
    pub parallelism: usize,
    /// Backpressure threshold
    pub backpressure_threshold: usize,
    /// Latency targets
    pub latency_target: Duration,
    /// Checkpoint interval
    pub checkpoint_interval: Duration,
    /// State management
    pub state_management: StateManagement,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            windowing: WindowingStrategy::TumblingTime {
                duration: Duration::from_secs(60),
            },
            buffer_size: 10000,
            parallelism: 1,
            backpressure_threshold: 8000,
            latency_target: Duration::from_millis(100),
            checkpoint_interval: Duration::from_secs(300),
            state_management: StateManagement::InMemory,
        }
    }
}

/// State management strategy
#[derive(Debug, Clone)]
pub enum StateManagement {
    /// In-memory state (non-persistent)
    InMemory,
    /// Periodic snapshots to disk
    Snapshots {
        /// Snapshot directory
        directory: String,
        /// Snapshot interval
        interval: Duration,
    },
    /// Write-ahead log
    WriteAheadLog {
        /// Log file path
        log_path: String,
    },
    /// External state store
    External {
        /// State store configuration
        config: HashMap<String, String>,
    },
}

/// Online learning update strategy
pub enum UpdateStrategy {
    /// Update on every data point
    Immediate,
    /// Batch updates
    Batch {
        /// Batch size
        batch_size: usize,
    },
    /// Time-based updates
    TimeBased {
        /// Update interval
        interval: Duration,
    },
    /// Adaptive updates based on drift detection
    Adaptive {
        /// Drift detection threshold
        drift_threshold: f64,
        /// Minimum update interval
        min_interval: Duration,
        /// Maximum update interval
        max_interval: Duration,
    },
    /// Custom update trigger
    Custom {
        /// Update trigger function
        trigger_fn: Box<dyn Fn(&StreamWindow, &StreamStats) -> bool + Send + Sync>,
    },
}

/// Stream processing statistics
#[derive(Debug, Clone)]
pub struct StreamStats {
    /// Total processed samples
    pub total_samples: usize,
    /// Current throughput (samples/second)
    pub throughput: f64,
    /// Average latency (milliseconds)
    pub avg_latency: f64,
    /// Current buffer utilization
    pub buffer_utilization: f64,
    /// Model accuracy (if available)
    pub accuracy: Option<f64>,
    /// Data drift metrics
    pub drift_metrics: HashMap<String, f64>,
    /// Error rates
    pub error_rate: f64,
    /// Processing start time
    pub start_time: SystemTime,
    /// Last update time
    pub last_update: SystemTime,
}

impl Default for StreamStats {
    fn default() -> Self {
        let now = SystemTime::now();
        Self {
            total_samples: 0,
            throughput: 0.0,
            avg_latency: 0.0,
            buffer_utilization: 0.0,
            accuracy: None,
            drift_metrics: HashMap::new(),
            error_rate: 0.0,
            start_time: now,
            last_update: now,
        }
    }
}

/// Streaming pipeline processor
pub struct StreamingPipeline<S = Untrained> {
    state: S,
    base_estimator: Option<Box<dyn PipelinePredictor>>,
    config: StreamConfig,
    update_strategy: UpdateStrategy,
    data_buffer: VecDeque<StreamDataPoint>,
    windows: Vec<StreamWindow>,
    statistics: StreamStats,
}

/// Trained state for `StreamingPipeline`
pub struct StreamingPipelineTrained {
    fitted_estimator: Box<dyn PipelinePredictor>,
    config: StreamConfig,
    update_strategy: UpdateStrategy,
    data_buffer: VecDeque<StreamDataPoint>,
    windows: Vec<StreamWindow>,
    statistics: StreamStats,
    model_state: HashMap<String, f64>,
    n_features_in: usize,
    feature_names_in: Option<Vec<String>>,
}

impl StreamingPipeline<Untrained> {
    /// Create a new streaming pipeline
    #[must_use]
    pub fn new(base_estimator: Box<dyn PipelinePredictor>, config: StreamConfig) -> Self {
        Self {
            state: Untrained,
            base_estimator: Some(base_estimator),
            config,
            update_strategy: UpdateStrategy::Batch { batch_size: 100 },
            data_buffer: VecDeque::new(),
            windows: Vec::new(),
            statistics: StreamStats::default(),
        }
    }

    /// Set update strategy
    #[must_use]
    pub fn update_strategy(mut self, strategy: UpdateStrategy) -> Self {
        self.update_strategy = strategy;
        self
    }

    /// Create a tumbling time window pipeline
    #[must_use]
    pub fn tumbling_time(
        base_estimator: Box<dyn PipelinePredictor>,
        window_duration: Duration,
    ) -> Self {
        let config = StreamConfig {
            windowing: WindowingStrategy::TumblingTime {
                duration: window_duration,
            },
            ..StreamConfig::default()
        };
        Self::new(base_estimator, config)
    }

    /// Create a sliding window pipeline
    #[must_use]
    pub fn sliding_window(
        base_estimator: Box<dyn PipelinePredictor>,
        window_size: usize,
        slide_step: usize,
    ) -> Self {
        let config = StreamConfig {
            windowing: WindowingStrategy::SlidingCount {
                size: window_size,
                step: slide_step,
            },
            ..StreamConfig::default()
        };
        Self::new(base_estimator, config)
    }

    /// Create a session window pipeline
    #[must_use]
    pub fn session_window(
        base_estimator: Box<dyn PipelinePredictor>,
        session_gap: Duration,
    ) -> Self {
        let config = StreamConfig {
            windowing: WindowingStrategy::Session { gap: session_gap },
            ..StreamConfig::default()
        };
        Self::new(base_estimator, config)
    }
}

impl Estimator for StreamingPipeline<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, Option<&ArrayView1<'_, Float>>> for StreamingPipeline<Untrained> {
    type Fitted = StreamingPipeline<StreamingPipelineTrained>;

    fn fit(
        self,
        x: &ArrayView2<'_, Float>,
        y: &Option<&ArrayView1<'_, Float>>,
    ) -> SklResult<Self::Fitted> {
        let mut base_estimator = self
            .base_estimator
            .ok_or_else(|| SklearsError::InvalidInput("No base estimator provided".to_string()))?;

        // Initial training on batch data
        if let Some(y_ref) = y {
            base_estimator.fit(x, y_ref)?;
        } else {
            return Err(SklearsError::InvalidInput(
                "No target values provided for initial training".to_string(),
            ));
        }

        // Initialize streaming state
        let mut model_state = HashMap::new();
        model_state.insert("batch_training_samples".to_string(), x.nrows() as f64);

        let mut statistics = self.statistics;
        statistics.total_samples = x.nrows();
        statistics.start_time = SystemTime::now();
        statistics.last_update = SystemTime::now();

        Ok(StreamingPipeline {
            state: StreamingPipelineTrained {
                fitted_estimator: base_estimator,
                config: self.config,
                update_strategy: self.update_strategy,
                data_buffer: self.data_buffer,
                windows: self.windows,
                statistics,
                model_state,
                n_features_in: x.ncols(),
                feature_names_in: None,
            },
            base_estimator: None,
            config: StreamConfig::default(),
            update_strategy: UpdateStrategy::Immediate,
            data_buffer: VecDeque::new(),
            windows: Vec::new(),
            statistics: StreamStats::default(),
        })
    }
}

impl StreamingPipeline<StreamingPipelineTrained> {
    /// Process a single data point from the stream
    pub fn process_point(&mut self, point: StreamDataPoint) -> SklResult<Option<Array1<f64>>> {
        let start_time = Instant::now();

        // Check for backpressure
        if self.state.data_buffer.len() >= self.state.config.backpressure_threshold {
            return Err(SklearsError::InvalidInput(
                "Backpressure threshold exceeded".to_string(),
            ));
        }

        // Add to buffer
        self.state.data_buffer.push_back(point.clone());

        // Update statistics
        self.state.statistics.total_samples += 1;
        self.state.statistics.buffer_utilization =
            self.state.data_buffer.len() as f64 / self.state.config.buffer_size as f64;

        // Create prediction input
        let features_2d =
            Array2::from_shape_vec((1, point.features.len()), point.features.to_vec()).map_err(
                |e| SklearsError::InvalidData {
                    reason: format!("Feature reshaping failed: {e}"),
                },
            )?;

        // Make prediction
        let prediction = self.state.fitted_estimator.predict(&features_2d.view())?;

        // Process windows
        self.process_windows()?;

        // Check for model updates
        self.check_model_update()?;

        // Update latency statistics
        let processing_time = start_time.elapsed().as_millis() as f64;
        self.state.statistics.avg_latency =
            (self.state.statistics.avg_latency * 0.9) + (processing_time * 0.1);

        // Update throughput
        let elapsed = self
            .state
            .statistics
            .start_time
            .elapsed()
            .unwrap_or(Duration::from_secs(1));
        self.state.statistics.throughput =
            self.state.statistics.total_samples as f64 / elapsed.as_secs_f64();

        Ok(Some(prediction))
    }

    /// Process batch of data points
    pub fn process_batch(&mut self, points: Vec<StreamDataPoint>) -> SklResult<Array2<f64>> {
        let mut predictions = Vec::new();

        for point in points {
            if let Some(pred) = self.process_point(point)? {
                predictions.extend(pred.iter().copied());
            }
        }

        if predictions.is_empty() {
            return Ok(Array2::zeros((0, 1)));
        }

        let n_predictions = predictions.len();
        Array2::from_shape_vec((n_predictions, 1), predictions).map_err(|e| {
            SklearsError::InvalidData {
                reason: format!("Batch prediction reshape failed: {e}"),
            }
        })
    }

    /// Process windowing logic
    fn process_windows(&mut self) -> SklResult<()> {
        match &self.state.config.windowing {
            WindowingStrategy::TumblingTime { duration } => {
                self.process_tumbling_time_windows(*duration)
            }
            WindowingStrategy::SlidingTime { duration, slide } => {
                self.process_sliding_time_windows(*duration, *slide)
            }
            WindowingStrategy::TumblingCount { count } => {
                self.process_tumbling_count_windows(*count)
            }
            WindowingStrategy::SlidingCount { size, step } => {
                self.process_sliding_count_windows(*size, *step)
            }
            WindowingStrategy::Session { gap } => self.process_session_windows(*gap),
            WindowingStrategy::Custom { .. } => {
                // Handle custom windowing differently to avoid borrow checker issues
                self.process_custom_windows_safe()
            }
        }
    }

    /// Process tumbling time windows
    fn process_tumbling_time_windows(&mut self, duration: Duration) -> SklResult<()> {
        let now = SystemTime::now();

        // Create new window if needed
        if self.state.windows.is_empty() {
            let window = StreamWindow::new(now, now + duration);
            self.state.windows.push(window);
        }

        // Add points to current window
        while let Some(point) = self.state.data_buffer.pop_front() {
            if let Some(current_window) = self.state.windows.last_mut() {
                if point.timestamp <= current_window.end_time {
                    current_window.add_point(point);
                } else {
                    // Create new window
                    let mut new_window = StreamWindow::new(
                        current_window.end_time,
                        current_window.end_time + duration,
                    );
                    new_window.add_point(point);
                    self.state.windows.push(new_window);
                }
            }
        }

        // Remove completed windows (keep only current)
        self.state.windows.retain(|w| w.end_time > now);

        Ok(())
    }

    /// Process sliding time windows
    fn process_sliding_time_windows(
        &mut self,
        duration: Duration,
        slide: Duration,
    ) -> SklResult<()> {
        // Simplified implementation
        self.process_tumbling_time_windows(duration)
    }

    /// Process tumbling count windows
    fn process_tumbling_count_windows(&mut self, count: usize) -> SklResult<()> {
        let now = SystemTime::now();

        while self.state.data_buffer.len() >= count {
            let mut window = StreamWindow::new(now, now);
            for _ in 0..count {
                if let Some(point) = self.state.data_buffer.pop_front() {
                    window.add_point(point);
                }
            }
            self.state.windows.push(window);
        }

        Ok(())
    }

    /// Process sliding count windows
    fn process_sliding_count_windows(&mut self, size: usize, step: usize) -> SklResult<()> {
        // Simplified implementation - just use tumbling for now
        self.process_tumbling_count_windows(step)
    }

    /// Process session windows
    fn process_session_windows(&mut self, gap: Duration) -> SklResult<()> {
        // Simplified implementation
        let now = SystemTime::now();

        if let Some(mut current_window) = self.state.windows.pop() {
            while let Some(point) = self.state.data_buffer.pop_front() {
                let time_since_last = point
                    .timestamp
                    .duration_since(current_window.end_time)
                    .unwrap_or(Duration::ZERO);

                if time_since_last <= gap {
                    current_window.add_point(point.clone());
                    current_window.end_time = point.timestamp;
                } else {
                    // Start new session
                    self.state.windows.push(current_window);
                    current_window = StreamWindow::new(point.timestamp, point.timestamp);
                    current_window.add_point(point);
                }
            }
            self.state.windows.push(current_window);
        } else if !self.state.data_buffer.is_empty() {
            // Start first session
            if let Some(point) = self.state.data_buffer.pop_front() {
                let mut window = StreamWindow::new(point.timestamp, point.timestamp);
                window.add_point(point);
                self.state.windows.push(window);
            }
        }

        Ok(())
    }

    /// Process custom windows safely (avoiding borrow checker issues)
    fn process_custom_windows_safe(&mut self) -> SklResult<()> {
        // Extract trigger function to avoid borrowing issues
        if let WindowingStrategy::Custom { trigger_fn } = &self.state.config.windowing {
            let buffer_vec: Vec<StreamDataPoint> = self.state.data_buffer.iter().cloned().collect();

            if trigger_fn(&buffer_vec) {
                let now = SystemTime::now();
                let mut window = StreamWindow::new(now, now);

                while let Some(point) = self.state.data_buffer.pop_front() {
                    window.add_point(point);
                }

                if !window.is_empty() {
                    self.state.windows.push(window);
                }
            }
        }

        Ok(())
    }

    /// Process custom windows
    fn process_custom_windows(
        &mut self,
        trigger_fn: &Box<dyn Fn(&[StreamDataPoint]) -> bool + Send + Sync>,
    ) -> SklResult<()> {
        let buffer_vec: Vec<StreamDataPoint> = self.state.data_buffer.iter().cloned().collect();

        if trigger_fn(&buffer_vec) {
            let now = SystemTime::now();
            let mut window = StreamWindow::new(now, now);

            while let Some(point) = self.state.data_buffer.pop_front() {
                window.add_point(point);
            }

            if !window.is_empty() {
                self.state.windows.push(window);
            }
        }

        Ok(())
    }

    /// Check if model should be updated
    fn check_model_update(&mut self) -> SklResult<()> {
        let should_update = match &self.state.update_strategy {
            UpdateStrategy::Immediate => !self.state.data_buffer.is_empty(),
            UpdateStrategy::Batch { batch_size } => self.state.data_buffer.len() >= *batch_size,
            UpdateStrategy::TimeBased { interval } => {
                self.state
                    .statistics
                    .last_update
                    .elapsed()
                    .unwrap_or(Duration::ZERO)
                    >= *interval
            }
            UpdateStrategy::Adaptive {
                drift_threshold,
                min_interval,
                max_interval,
            } => self.check_adaptive_update(*drift_threshold, *min_interval, *max_interval),
            UpdateStrategy::Custom { trigger_fn } => {
                if let Some(window) = self.state.windows.last() {
                    trigger_fn(window, &self.state.statistics)
                } else {
                    false
                }
            }
        };

        if should_update {
            self.update_model()?;
        }

        Ok(())
    }

    /// Check if adaptive update should be triggered
    fn check_adaptive_update(
        &self,
        drift_threshold: f64,
        min_interval: Duration,
        max_interval: Duration,
    ) -> bool {
        let elapsed = self
            .state
            .statistics
            .last_update
            .elapsed()
            .unwrap_or(Duration::ZERO);

        if elapsed < min_interval {
            return false;
        }

        if elapsed >= max_interval {
            return true;
        }

        // Check for drift (simplified)
        let drift_score = self
            .state
            .statistics
            .drift_metrics
            .get("feature_drift")
            .unwrap_or(&0.0);
        *drift_score > drift_threshold
    }

    /// Update the model with recent data
    fn update_model(&mut self) -> SklResult<()> {
        if let Some(window) = self.state.windows.last() {
            if !window.is_empty() {
                let features = window.features_matrix()?;
                let targets = window.targets_array();

                if let Some(targets_array) = targets {
                    // Incremental learning (simplified)
                    self.state
                        .fitted_estimator
                        .fit(&features.view(), &targets_array.view())?;

                    self.state.statistics.last_update = SystemTime::now();
                    self.state
                        .model_state
                        .insert("last_update_samples".to_string(), window.size() as f64);
                }
            }
        }

        Ok(())
    }

    /// Get current statistics
    #[must_use]
    pub fn statistics(&self) -> &StreamStats {
        &self.state.statistics
    }

    /// Get current buffer size
    #[must_use]
    pub fn buffer_size(&self) -> usize {
        self.state.data_buffer.len()
    }

    /// Get number of active windows
    #[must_use]
    pub fn active_windows(&self) -> usize {
        self.state.windows.len()
    }

    /// Checkpoint the current state
    pub fn checkpoint(&self) -> SklResult<HashMap<String, String>> {
        let mut checkpoint = HashMap::new();
        checkpoint.insert(
            "total_samples".to_string(),
            self.state.statistics.total_samples.to_string(),
        );
        checkpoint.insert(
            "buffer_size".to_string(),
            self.state.data_buffer.len().to_string(),
        );
        checkpoint.insert(
            "active_windows".to_string(),
            self.state.windows.len().to_string(),
        );
        checkpoint.insert(
            "throughput".to_string(),
            self.state.statistics.throughput.to_string(),
        );

        Ok(checkpoint)
    }

    /// Clear internal buffers and windows
    pub fn clear_buffers(&mut self) {
        self.state.data_buffer.clear();
        self.state.windows.clear();
    }

    /// Get drift detection metrics
    #[must_use]
    pub fn drift_metrics(&self) -> &HashMap<String, f64> {
        &self.state.statistics.drift_metrics
    }

    /// Detect concept drift (simplified implementation)
    pub fn detect_drift(
        &mut self,
        reference_window: &StreamWindow,
        current_window: &StreamWindow,
    ) -> SklResult<f64> {
        if reference_window.is_empty() || current_window.is_empty() {
            return Ok(0.0);
        }

        let ref_features = reference_window.features_matrix()?;
        let cur_features = current_window.features_matrix()?;

        // Simple drift detection using mean difference
        let ref_mean = ref_features.mean_axis(Axis(0)).unwrap();
        let cur_mean = cur_features.mean_axis(Axis(0)).unwrap();

        let drift_score = (&ref_mean - &cur_mean).mapv(|x| x * x).sum().sqrt();

        // Update drift metrics
        self.state
            .statistics
            .drift_metrics
            .insert("feature_drift".to_string(), drift_score);

        Ok(drift_score)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::MockPredictor;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_stream_data_point() {
        let features = array![1.0, 2.0, 3.0];
        let point =
            StreamDataPoint::new(features.clone(), "test_point".to_string()).with_target(1.0);

        assert_eq!(point.id, "test_point");
        assert_eq!(point.features, features);
        assert_eq!(point.target, Some(1.0));
    }

    #[test]
    fn test_stream_window() {
        let start_time = SystemTime::now();
        let end_time = start_time + Duration::from_secs(60);
        let mut window = StreamWindow::new(start_time, end_time);

        let point1 = StreamDataPoint::new(array![1.0, 2.0], "point1".to_string());
        let point2 = StreamDataPoint::new(array![3.0, 4.0], "point2".to_string());

        window.add_point(point1);
        window.add_point(point2);

        assert_eq!(window.size(), 2);

        let features = window.features_matrix().unwrap();
        assert_eq!(features.nrows(), 2);
        assert_eq!(features.ncols(), 2);
    }

    #[test]
    fn test_streaming_pipeline_creation() {
        let base_estimator = Box::new(MockPredictor::new());
        let pipeline = StreamingPipeline::tumbling_time(base_estimator, Duration::from_secs(60));

        assert!(matches!(
            pipeline.config.windowing,
            WindowingStrategy::TumblingTime { .. }
        ));
    }

    #[test]
    fn test_streaming_pipeline_fit() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![1.0, 0.0];

        let base_estimator = Box::new(MockPredictor::new());
        let pipeline = StreamingPipeline::tumbling_time(base_estimator, Duration::from_secs(60));

        let fitted_pipeline = pipeline.fit(&x.view(), &Some(&y.view())).unwrap();
        assert_eq!(fitted_pipeline.state.n_features_in, 2);
        assert_eq!(fitted_pipeline.state.statistics.total_samples, 2);
    }

    #[test]
    fn test_point_processing() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![1.0, 0.0];

        let base_estimator = Box::new(MockPredictor::new());
        let pipeline = StreamingPipeline::tumbling_time(base_estimator, Duration::from_secs(60));

        let mut fitted_pipeline = pipeline.fit(&x.view(), &Some(&y.view())).unwrap();

        let point = StreamDataPoint::new(array![5.0, 6.0], "test_point".to_string());
        let prediction = fitted_pipeline.process_point(point).unwrap();

        assert!(prediction.is_some());
        assert_eq!(fitted_pipeline.active_windows(), 1);
    }

    #[test]
    fn test_window_strategies() {
        let base_estimator = Box::new(MockPredictor::new());

        // Test tumbling count windows
        let pipeline = StreamingPipeline::new(
            base_estimator,
            StreamConfig {
                windowing: WindowingStrategy::TumblingCount { count: 2 },
                ..StreamConfig::default()
            },
        );

        assert!(matches!(
            pipeline.config.windowing,
            WindowingStrategy::TumblingCount { count: 2 }
        ));
    }

    #[test]
    fn test_update_strategies() {
        let base_estimator = Box::new(MockPredictor::new());
        let pipeline = StreamingPipeline::tumbling_time(base_estimator, Duration::from_secs(60))
            .update_strategy(UpdateStrategy::Batch { batch_size: 10 });

        assert!(matches!(
            pipeline.update_strategy,
            UpdateStrategy::Batch { batch_size: 10 }
        ));
    }
}
