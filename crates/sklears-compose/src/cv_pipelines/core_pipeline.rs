//! Core computer vision pipeline implementation
//!
//! This module provides the main `CVPipeline` struct and its implementation,
//! including processing orchestration, state management, and pipeline execution
//! for comprehensive computer vision workflows.

use super::image_specification::{ImageData, ImageSpecification};
use super::metrics_statistics::{CVMetrics, ErrorType};
use super::model_management::ModelMetadata;
use super::multimodal_processing::MultiModalConfig;
use super::processing_configuration::{PerformanceConfig, QualitySettings};
use super::realtime_streaming::RealTimeProcessingConfig;
use super::types_config::{ProcessingMode, RecoveryStrategy};

use scirs2_core::ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use sklears_core::{error::Result as SklResult, prelude::SklearsError};
use std::collections::HashMap;
use std::fmt;
use std::time::{Duration, Instant, SystemTime};

/// Comprehensive computer vision processing pipeline
#[derive(Debug)]
pub struct CVPipeline {
    /// Pipeline configuration
    pub config: CVConfig,
    /// Image preprocessing steps
    pub image_preprocessing: Vec<Box<dyn ImageTransform>>,
    /// Feature extraction steps
    pub feature_extraction: Vec<Box<dyn FeatureExtractor>>,
    /// Computer vision models
    pub cv_models: Vec<Box<dyn CVModel>>,
    /// Post-processing steps
    pub postprocessing: Vec<Box<dyn PostProcessor>>,
    /// Current pipeline state
    pub state: CVPipelineState,
    /// Metrics collector
    pub metrics: CVMetrics,
    /// Pipeline execution context
    pub context: PipelineContext,
}

impl CVPipeline {
    /// Create a new computer vision pipeline
    #[must_use]
    pub fn new(config: CVConfig) -> Self {
        Self {
            config,
            image_preprocessing: Vec::new(),
            feature_extraction: Vec::new(),
            cv_models: Vec::new(),
            postprocessing: Vec::new(),
            state: CVPipelineState::Ready,
            metrics: CVMetrics::new(),
            context: PipelineContext::new(),
        }
    }

    /// Add an image preprocessing step
    pub fn add_preprocessing(&mut self, transform: Box<dyn ImageTransform>) -> SklResult<()> {
        if matches!(self.state, CVPipelineState::Processing) {
            return Err(SklearsError::InvalidState(
                "Cannot modify pipeline while processing".to_string(),
            ));
        }

        self.image_preprocessing.push(transform);
        self.context.register_component("preprocessing".to_string());
        Ok(())
    }

    /// Add a feature extraction step
    pub fn add_feature_extraction(
        &mut self,
        extractor: Box<dyn FeatureExtractor>,
    ) -> SklResult<()> {
        if matches!(self.state, CVPipelineState::Processing) {
            return Err(SklearsError::InvalidState(
                "Cannot modify pipeline while processing".to_string(),
            ));
        }

        self.feature_extraction.push(extractor);
        self.context
            .register_component("feature_extraction".to_string());
        Ok(())
    }

    /// Add a computer vision model
    pub fn add_model(&mut self, model: Box<dyn CVModel>) -> SklResult<()> {
        if matches!(self.state, CVPipelineState::Processing) {
            return Err(SklearsError::InvalidState(
                "Cannot modify pipeline while processing".to_string(),
            ));
        }

        self.cv_models.push(model);
        self.context.register_component("model".to_string());
        Ok(())
    }

    /// Add a post-processing step
    pub fn add_postprocessing(&mut self, processor: Box<dyn PostProcessor>) -> SklResult<()> {
        if matches!(self.state, CVPipelineState::Processing) {
            return Err(SklearsError::InvalidState(
                "Cannot modify pipeline while processing".to_string(),
            ));
        }

        self.postprocessing.push(processor);
        self.context
            .register_component("postprocessing".to_string());
        Ok(())
    }

    /// Process a single image through the pipeline
    pub fn process_image(&mut self, image: ImageData) -> SklResult<Vec<ProcessedResult>> {
        let start_time = Instant::now();
        self.state = CVPipelineState::Processing;
        self.context.start_processing();

        let result = self.execute_pipeline(image, start_time);

        match &result {
            Ok(results) => {
                self.metrics.record_image_processed(
                    start_time.elapsed(),
                    results
                        .first()
                        .map_or(0.0, ProcessedResult::confidence_score),
                );
                self.state = CVPipelineState::Ready;
            }
            Err(err) => {
                self.metrics
                    .record_error(ErrorType::Unknown, err.to_string());
                self.state = CVPipelineState::Error;
            }
        }

        self.context.end_processing();
        result
    }

    /// Process multiple images in batch
    pub fn process_batch(
        &mut self,
        images: Vec<ImageData>,
    ) -> SklResult<Vec<Vec<ProcessedResult>>> {
        if matches!(self.config.processing_mode, ProcessingMode::RealTime) {
            return Err(SklearsError::InvalidConfiguration(
                "Batch processing not supported in real-time mode".to_string(),
            ));
        }

        let mut results = Vec::with_capacity(images.len());
        self.state = CVPipelineState::Processing;

        for (i, image) in images.into_iter().enumerate() {
            match self.process_image(image) {
                Ok(image_results) => results.push(image_results),
                Err(err) => {
                    self.handle_batch_error(err, i)?;
                    // Continue processing if error handling strategy allows
                    if !self.should_continue_batch_processing() {
                        break;
                    }
                }
            }
        }

        self.state = CVPipelineState::Ready;
        Ok(results)
    }

    /// Execute the core pipeline processing logic
    fn execute_pipeline(
        &mut self,
        image: ImageData,
        start_time: Instant,
    ) -> SklResult<Vec<ProcessedResult>> {
        // Validate input
        self.validate_input(&image)?;

        // Apply preprocessing
        let mut processed_image = image;
        let preprocessing_start = Instant::now();

        for i in 0..self.image_preprocessing.len() {
            let transform = &self.image_preprocessing[i];
            match transform.transform(&processed_image) {
                Ok(transformed) => {
                    processed_image = transformed;
                    self.context.record_step_completion("preprocessing", i);
                }
                Err(err) => {
                    self.handle_preprocessing_error(err, i)?;
                    if !self.should_continue_processing() {
                        return Err(SklearsError::ProcessingError(
                            "Preprocessing failed and recovery not possible".to_string(),
                        ));
                    }
                }
            }
        }

        let preprocessing_time = preprocessing_start.elapsed();
        self.context
            .record_timing("preprocessing", preprocessing_time);

        // Extract features (if any extractors are configured)
        let mut features = Vec::new();
        if !self.feature_extraction.is_empty() {
            let feature_start = Instant::now();

            for i in 0..self.feature_extraction.len() {
                let extractor = &self.feature_extraction[i];
                match extractor.extract(&processed_image) {
                    Ok(feature_vector) => {
                        features.push(feature_vector);
                        self.context.record_step_completion("feature_extraction", i);
                    }
                    Err(err) => {
                        self.handle_feature_extraction_error(err, i)?;
                    }
                }
            }

            let feature_time = feature_start.elapsed();
            self.context
                .record_timing("feature_extraction", feature_time);
        }

        // Apply CV models
        let mut predictions = Vec::new();
        let inference_start = Instant::now();

        for i in 0..self.cv_models.len() {
            let model = &self.cv_models[i];
            match model.predict(&processed_image) {
                Ok(prediction) => {
                    predictions.push(prediction);
                    self.context.record_step_completion("inference", i);
                }
                Err(err) => {
                    self.handle_inference_error(err, i)?;
                    if predictions.is_empty() && !self.has_fallback_model() {
                        return Err(SklearsError::ModelError(
                            "All models failed and no fallback available".to_string(),
                        ));
                    }
                }
            }
        }

        let inference_time = inference_start.elapsed();
        self.context.record_timing("inference", inference_time);

        // Apply post-processing
        let mut results = Vec::new();
        let postprocessing_start = Instant::now();

        if self.postprocessing.is_empty() {
            // If no post-processing, create simple results
            for prediction in predictions {
                results.push(ProcessedResult::from_prediction(
                    prediction,
                    start_time.elapsed(),
                    ProcessingMetadata::default(),
                ));
            }
        } else {
            for prediction in predictions {
                let mut processed_prediction = prediction;

                for i in 0..self.postprocessing.len() {
                    let processor = &self.postprocessing[i];
                    match processor.process(processed_prediction.clone()) {
                        Ok(processed_result) => {
                            processed_prediction = processed_result.processed.clone();
                            self.context.record_step_completion("postprocessing", i);
                        }
                        Err(err) => {
                            self.handle_postprocessing_error(err, i)?;
                        }
                    }
                }

                results.push(ProcessedResult::from_prediction(
                    processed_prediction,
                    start_time.elapsed(),
                    self.create_processing_metadata(start_time),
                ));
            }
        }

        let postprocessing_time = postprocessing_start.elapsed();
        self.context
            .record_timing("postprocessing", postprocessing_time);

        // Update context with final results
        self.context.record_final_results(results.len());

        Ok(results)
    }

    /// Validate input image against pipeline requirements
    fn validate_input(&self, image: &ImageData) -> SklResult<()> {
        // Check against input specification
        if let Err(validation_err) = self.config.input_spec.validate(image) {
            return Err(SklearsError::ValidationError(validation_err.to_string()));
        }

        // Check image data integrity
        if image.data.is_empty() {
            return Err(SklearsError::ValidationError(
                "Image data is empty".to_string(),
            ));
        }

        // Check processing mode compatibility
        match self.config.processing_mode {
            ProcessingMode::RealTime => {
                if image.memory_footprint() > self.config.performance.resource_limits.max_memory {
                    return Err(SklearsError::ValidationError(
                        "Image too large for real-time processing".to_string(),
                    ));
                }
            }
            _ => {} // Other modes are more flexible
        }

        Ok(())
    }

    /// Handle preprocessing errors with recovery strategies
    fn handle_preprocessing_error(
        &mut self,
        error: Box<dyn std::error::Error>,
        step_index: usize,
    ) -> SklResult<()> {
        self.metrics
            .record_error(ErrorType::Preprocessing, error.to_string());

        match self.config.performance.error_handling.recovery_strategy {
            RecoveryStrategy::FailFast => {
                return Err(SklearsError::ProcessingError(format!(
                    "Preprocessing step {step_index} failed: {error}"
                )));
            }
            RecoveryStrategy::Skip => {
                // Skip this preprocessing step and continue
                self.context
                    .record_step_skipped("preprocessing", step_index);
            }
            RecoveryStrategy::Retry => {
                // Implement retry logic if needed
                self.context.record_step_retry("preprocessing", step_index);
            }
            RecoveryStrategy::Fallback => {
                // Use fallback preprocessing if available
                self.context
                    .record_step_fallback("preprocessing", step_index);
            }
            RecoveryStrategy::Degrade => {
                // Continue with degraded quality
                self.context
                    .record_step_degraded("preprocessing", step_index);
            }
            RecoveryStrategy::UseCache => {
                // Use cached result if available
                self.context.record_step_cached("preprocessing", step_index);
            }
        }

        Ok(())
    }

    /// Handle feature extraction errors
    fn handle_feature_extraction_error(
        &mut self,
        error: Box<dyn std::error::Error>,
        step_index: usize,
    ) -> SklResult<()> {
        self.metrics
            .record_error(ErrorType::Preprocessing, error.to_string());
        self.context
            .record_step_error("feature_extraction", step_index, error.to_string());
        Ok(())
    }

    /// Handle inference errors
    fn handle_inference_error(
        &mut self,
        error: Box<dyn std::error::Error>,
        model_index: usize,
    ) -> SklResult<()> {
        self.metrics
            .record_error(ErrorType::Inference, error.to_string());
        self.context
            .record_step_error("inference", model_index, error.to_string());
        Ok(())
    }

    /// Handle post-processing errors
    fn handle_postprocessing_error(
        &mut self,
        error: Box<dyn std::error::Error>,
        step_index: usize,
    ) -> SklResult<()> {
        self.metrics
            .record_error(ErrorType::PostProcessing, error.to_string());
        self.context
            .record_step_error("postprocessing", step_index, error.to_string());
        Ok(())
    }

    /// Handle batch processing errors
    fn handle_batch_error(&mut self, error: SklearsError, image_index: usize) -> SklResult<()> {
        self.metrics
            .record_error(ErrorType::Unknown, error.to_string());

        match self.config.performance.error_handling.recovery_strategy {
            RecoveryStrategy::FailFast => {
                return Err(error);
            }
            RecoveryStrategy::Skip => {
                // Continue with next image
                self.context.record_batch_image_skipped(image_index);
            }
            _ => {
                // Other strategies can be implemented as needed
                self.context
                    .record_batch_image_error(image_index, error.to_string());
            }
        }

        Ok(())
    }

    /// Check if processing should continue after an error
    fn should_continue_processing(&self) -> bool {
        match self.config.performance.error_handling.recovery_strategy {
            RecoveryStrategy::FailFast => false,
            _ => true,
        }
    }

    /// Check if batch processing should continue after an error
    fn should_continue_batch_processing(&self) -> bool {
        self.should_continue_processing()
    }

    /// Check if pipeline has fallback models
    fn has_fallback_model(&self) -> bool {
        // This would check if there are fallback models configured
        // For now, we'll assume no fallback models
        false
    }

    /// Create processing metadata for results
    fn create_processing_metadata(&self, start_time: Instant) -> ProcessingMetadata {
        let mut step_times = HashMap::new();

        // Copy timing information from context
        for (step_name, duration) in &self.context.step_timings {
            step_times.insert(step_name.clone(), *duration);
        }

        ProcessingMetadata {
            total_time: start_time.elapsed(),
            step_times,
            quality_improvement: QualityImprovement::default(),
        }
    }

    /// Get pipeline status and metrics
    #[must_use]
    pub fn get_status(&self) -> PipelineStatus {
        PipelineStatus {
            state: self.state,
            processing_count: self.context.total_processed,
            error_count: self.metrics.error_tracking.total_errors,
            average_processing_time: self.metrics.processing_stats.avg_time_per_image,
            current_throughput: self.metrics.processing_stats.throughput,
            resource_utilization: ResourceStatus {
                cpu_usage: self.metrics.resource_utilization.cpu_utilization,
                memory_usage: self.metrics.resource_utilization.memory_usage_mb,
                gpu_usage: self.metrics.resource_utilization.gpu_utilization,
            },
        }
    }

    /// Pause the pipeline (if in streaming mode)
    pub fn pause(&mut self) -> SklResult<()> {
        match self.state {
            CVPipelineState::Processing => {
                self.state = CVPipelineState::Paused;
                Ok(())
            }
            CVPipelineState::Ready => {
                self.state = CVPipelineState::Paused;
                Ok(())
            }
            _ => Err(SklearsError::InvalidState(format!(
                "Cannot pause pipeline in state {:?}",
                self.state
            ))),
        }
    }

    /// Resume the pipeline
    pub fn resume(&mut self) -> SklResult<()> {
        match self.state {
            CVPipelineState::Paused => {
                self.state = CVPipelineState::Ready;
                Ok(())
            }
            _ => Err(SklearsError::InvalidState(format!(
                "Cannot resume pipeline in state {:?}",
                self.state
            ))),
        }
    }

    /// Stop the pipeline
    pub fn stop(&mut self) -> SklResult<()> {
        self.state = CVPipelineState::Stopped;
        self.context.stop_processing();
        Ok(())
    }

    /// Reset pipeline metrics and state
    pub fn reset(&mut self) -> SklResult<()> {
        if matches!(self.state, CVPipelineState::Processing) {
            return Err(SklearsError::InvalidState(
                "Cannot reset pipeline while processing".to_string(),
            ));
        }

        self.metrics.reset();
        self.context.reset();
        self.state = CVPipelineState::Ready;
        Ok(())
    }

    /// Get detailed pipeline configuration
    #[must_use]
    pub fn get_config(&self) -> &CVConfig {
        &self.config
    }

    /// Update pipeline configuration
    pub fn update_config(&mut self, config: CVConfig) -> SklResult<()> {
        if matches!(self.state, CVPipelineState::Processing) {
            return Err(SklearsError::InvalidState(
                "Cannot update configuration while processing".to_string(),
            ));
        }

        self.config = config;
        Ok(())
    }
}

/// Configuration for computer vision processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CVConfig {
    /// Pipeline name
    pub name: String,
    /// Input image specifications
    pub input_spec: ImageSpecification,
    /// Processing mode (batch, real-time, streaming)
    pub processing_mode: ProcessingMode,
    /// Quality settings
    pub quality_settings: QualitySettings,
    /// Performance optimization settings
    pub performance: PerformanceConfig,
    /// Multi-modal settings
    pub multimodal: MultiModalConfig,
    /// Real-time processing settings
    pub realtime: RealTimeProcessingConfig,
    /// Pipeline metadata
    pub metadata: PipelineMetadata,
}

impl Default for CVConfig {
    fn default() -> Self {
        Self {
            name: "CV Pipeline".to_string(),
            input_spec: ImageSpecification::default(),
            processing_mode: ProcessingMode::Batch,
            quality_settings: QualitySettings::default(),
            performance: PerformanceConfig::default(),
            multimodal: MultiModalConfig::default(),
            realtime: RealTimeProcessingConfig::default(),
            metadata: PipelineMetadata::default(),
        }
    }
}

impl CVConfig {
    /// Create configuration for real-time processing
    #[must_use]
    pub fn real_time(name: &str) -> Self {
        Self {
            name: name.to_string(),
            processing_mode: ProcessingMode::RealTime,
            quality_settings: QualitySettings::performance_optimized(),
            performance: PerformanceConfig::default(),
            realtime: RealTimeProcessingConfig::low_latency(),
            ..Default::default()
        }
    }

    /// Create configuration for high-quality batch processing
    #[must_use]
    pub fn high_quality_batch(name: &str) -> Self {
        Self {
            name: name.to_string(),
            processing_mode: ProcessingMode::Batch,
            quality_settings: QualitySettings::high_quality(),
            performance: PerformanceConfig::default(),
            ..Default::default()
        }
    }

    /// Create configuration for streaming processing
    #[must_use]
    pub fn streaming(name: &str) -> Self {
        Self {
            name: name.to_string(),
            processing_mode: ProcessingMode::Streaming,
            quality_settings: QualitySettings::balanced(),
            performance: PerformanceConfig::default(),
            realtime: RealTimeProcessingConfig::high_quality_streaming(),
            ..Default::default()
        }
    }
}

/// Pipeline metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineMetadata {
    /// Pipeline version
    pub version: String,
    /// Creation timestamp
    pub created_at: SystemTime,
    /// Creator information
    pub created_by: String,
    /// Description
    pub description: String,
    /// Tags for categorization
    pub tags: Vec<String>,
    /// Custom properties
    pub properties: HashMap<String, String>,
}

impl Default for PipelineMetadata {
    fn default() -> Self {
        Self {
            version: "1.0.0".to_string(),
            created_at: SystemTime::now(),
            created_by: "System".to_string(),
            description: "Computer vision processing pipeline".to_string(),
            tags: vec![],
            properties: HashMap::new(),
        }
    }
}

/// Pipeline state enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CVPipelineState {
    /// Pipeline is ready for processing
    Ready,
    /// Pipeline is currently processing
    Processing,
    /// Pipeline is paused
    Paused,
    /// Pipeline encountered an error
    Error,
    /// Pipeline is stopped
    Stopped,
}

impl fmt::Display for CVPipelineState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Ready => write!(f, "Ready"),
            Self::Processing => write!(f, "Processing"),
            Self::Paused => write!(f, "Paused"),
            Self::Error => write!(f, "Error"),
            Self::Stopped => write!(f, "Stopped"),
        }
    }
}

/// Pipeline execution context for tracking processing state
#[derive(Debug, Clone)]
pub struct PipelineContext {
    /// Total number of images processed
    pub total_processed: u64,
    /// Currently active components
    pub active_components: Vec<String>,
    /// Step completion tracking
    pub step_completions: HashMap<String, Vec<usize>>,
    /// Step timing information
    pub step_timings: HashMap<String, Duration>,
    /// Processing start time
    pub processing_start: Option<Instant>,
    /// Error records
    pub error_records: Vec<ContextErrorRecord>,
}

impl Default for PipelineContext {
    fn default() -> Self {
        Self::new()
    }
}

impl PipelineContext {
    /// Create new pipeline context
    #[must_use]
    pub fn new() -> Self {
        Self {
            total_processed: 0,
            active_components: Vec::new(),
            step_completions: HashMap::new(),
            step_timings: HashMap::new(),
            processing_start: None,
            error_records: Vec::new(),
        }
    }

    /// Register a component in the pipeline
    pub fn register_component(&mut self, component: String) {
        if !self.active_components.contains(&component) {
            self.active_components.push(component);
        }
    }

    /// Start processing session
    pub fn start_processing(&mut self) {
        self.processing_start = Some(Instant::now());
    }

    /// End processing session
    pub fn end_processing(&mut self) {
        self.total_processed += 1;
        self.processing_start = None;
    }

    /// Stop processing
    pub fn stop_processing(&mut self) {
        self.processing_start = None;
    }

    /// Record step completion
    pub fn record_step_completion(&mut self, component: &str, step_index: usize) {
        self.step_completions
            .entry(component.to_string())
            .or_default()
            .push(step_index);
    }

    /// Record timing for a component
    pub fn record_timing(&mut self, component: &str, duration: Duration) {
        self.step_timings.insert(component.to_string(), duration);
    }

    /// Record step error
    pub fn record_step_error(&mut self, component: &str, step_index: usize, error: String) {
        self.error_records.push(ContextErrorRecord {
            component: component.to_string(),
            step_index,
            error_message: error,
            timestamp: SystemTime::now(),
        });
    }

    /// Record step skip
    pub fn record_step_skipped(&mut self, component: &str, step_index: usize) {
        // Implementation for tracking skipped steps
    }

    /// Record step retry
    pub fn record_step_retry(&mut self, component: &str, step_index: usize) {
        // Implementation for tracking retried steps
    }

    /// Record step fallback
    pub fn record_step_fallback(&mut self, component: &str, step_index: usize) {
        // Implementation for tracking fallback usage
    }

    /// Record step degradation
    pub fn record_step_degraded(&mut self, component: &str, step_index: usize) {
        // Implementation for tracking quality degradation
    }

    /// Record step cache usage
    pub fn record_step_cached(&mut self, component: &str, step_index: usize) {
        // Implementation for tracking cache usage
    }

    /// Record batch image skip
    pub fn record_batch_image_skipped(&mut self, image_index: usize) {
        // Implementation for tracking skipped batch images
    }

    /// Record batch image error
    pub fn record_batch_image_error(&mut self, image_index: usize, error: String) {
        // Implementation for tracking batch image errors
    }

    /// Record final results
    pub fn record_final_results(&mut self, result_count: usize) {
        // Implementation for tracking final result counts
    }

    /// Reset context
    pub fn reset(&mut self) {
        self.total_processed = 0;
        self.step_completions.clear();
        self.step_timings.clear();
        self.processing_start = None;
        self.error_records.clear();
    }
}

/// Context error record
#[derive(Debug, Clone)]
pub struct ContextErrorRecord {
    /// Component that had the error
    pub component: String,
    /// Step index within component
    pub step_index: usize,
    /// Error message
    pub error_message: String,
    /// Timestamp of error
    pub timestamp: SystemTime,
}

/// Pipeline status information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineStatus {
    /// Current pipeline state
    pub state: CVPipelineState,
    /// Number of images processed
    pub processing_count: u64,
    /// Number of errors encountered
    pub error_count: u64,
    /// Average processing time
    pub average_processing_time: Duration,
    /// Current throughput
    pub current_throughput: f64,
    /// Resource utilization
    pub resource_utilization: ResourceStatus,
}

/// Resource utilization status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceStatus {
    /// CPU usage percentage
    pub cpu_usage: f64,
    /// Memory usage in MB
    pub memory_usage: f64,
    /// GPU usage percentage (if applicable)
    pub gpu_usage: Option<f64>,
}

/// Processed result from pipeline execution
#[derive(Debug, Clone)]
pub struct ProcessedResult {
    /// Original prediction/result
    pub original: Prediction,
    /// Processed/enhanced result
    pub processed: Prediction,
    /// Processing steps applied
    pub processing_steps: Vec<String>,
    /// Processing metadata
    pub metadata: ProcessingMetadata,
}

impl ProcessedResult {
    /// Create result from a prediction
    #[must_use]
    pub fn from_prediction(
        prediction: Prediction,
        processing_time: Duration,
        metadata: ProcessingMetadata,
    ) -> Self {
        Self {
            original: prediction.clone(),
            processed: prediction,
            processing_steps: vec![],
            metadata,
        }
    }

    /// Get confidence score from result
    #[must_use]
    pub fn confidence_score(&self) -> f64 {
        self.processed.confidence
    }
}

/// Processing metadata for results
#[derive(Debug, Clone)]
pub struct ProcessingMetadata {
    /// Total processing time
    pub total_time: Duration,
    /// Time spent in each processing step
    pub step_times: HashMap<String, Duration>,
    /// Quality improvement metrics
    pub quality_improvement: QualityImprovement,
}

impl Default for ProcessingMetadata {
    fn default() -> Self {
        Self {
            total_time: Duration::from_millis(0),
            step_times: HashMap::new(),
            quality_improvement: QualityImprovement::default(),
        }
    }
}

/// Quality improvement metrics
#[derive(Debug, Clone)]
pub struct QualityImprovement {
    /// Confidence delta after processing
    pub confidence_delta: f64,
    /// Accuracy improvement
    pub accuracy_delta: f64,
    /// Noise reduction achieved
    pub noise_reduction: f64,
    /// Detail enhancement
    pub detail_enhancement: f64,
}

impl Default for QualityImprovement {
    fn default() -> Self {
        Self {
            confidence_delta: 0.0,
            accuracy_delta: 0.0,
            noise_reduction: 0.0,
            detail_enhancement: 0.0,
        }
    }
}

/// Prediction result from computer vision models
#[derive(Debug, Clone)]
pub struct Prediction {
    /// Prediction type
    pub prediction_type: PredictionType,
    /// Confidence score
    pub confidence: f64,
    /// Prediction data
    pub data: PredictionData,
    /// Model metadata
    pub model_info: Option<ModelMetadata>,
}

/// Types of predictions
#[derive(Debug, Clone, PartialEq)]
pub enum PredictionType {
    /// Classification result
    Classification,
    /// Object detection result
    Detection,
    /// Segmentation result
    Segmentation,
    /// Keypoint detection result
    Keypoints,
    /// Feature embedding
    Embedding,
    /// Custom prediction type
    Custom(String),
}

impl fmt::Display for PredictionType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Classification => write!(f, "Classification"),
            Self::Detection => write!(f, "Detection"),
            Self::Segmentation => write!(f, "Segmentation"),
            Self::Keypoints => write!(f, "Keypoints"),
            Self::Embedding => write!(f, "Embedding"),
            Self::Custom(name) => write!(f, "Custom({name})"),
        }
    }
}

/// Prediction data container
#[derive(Debug, Clone)]
pub enum PredictionData {
    /// Classification labels and probabilities
    Classification {
        classes: Vec<String>,
        probabilities: Array1<f32>,
    },
    /// Detection bounding boxes
    Detection {
        boxes: Array2<f32>, // [N, 4] format (x1, y1, x2, y2)
        classes: Vec<String>,
        scores: Array1<f32>,
    },
    /// Segmentation mask
    Segmentation {
        mask: Array2<u8>, // Height x Width
        classes: Vec<String>,
    },
    /// Keypoints
    Keypoints {
        points: Array2<f32>, // [N, 2] format (x, y)
        visibility: Array1<f32>,
    },
    /// Feature embedding
    Embedding { features: Array1<f32> },
    /// Custom data
    Custom { data: HashMap<String, Array1<f32>> },
}

// Placeholder traits for the components (these would be defined elsewhere in a real implementation)

/// Trait for image transformations
pub trait ImageTransform: Send + Sync + std::fmt::Debug {
    /// Transform an image
    fn transform(&self, image: &ImageData) -> Result<ImageData, Box<dyn std::error::Error>>;
}

/// Trait for feature extractors
pub trait FeatureExtractor: Send + Sync + std::fmt::Debug {
    /// Extract features from an image
    fn extract(&self, image: &ImageData) -> Result<Array1<f32>, Box<dyn std::error::Error>>;
}

/// Trait for computer vision models
pub trait CVModel: Send + Sync + std::fmt::Debug {
    /// Make a prediction on an image
    fn predict(&self, image: &ImageData) -> Result<Prediction, Box<dyn std::error::Error>>;

    /// Get model metadata
    fn metadata(&self) -> Option<&ModelMetadata> {
        None
    }
}

/// Trait for post-processors
pub trait PostProcessor: Send + Sync + std::fmt::Debug {
    /// Process a prediction result
    fn process(
        &self,
        prediction: Prediction,
    ) -> Result<ProcessedResult, Box<dyn std::error::Error>>;
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    // Mock implementations for testing
    #[derive(Debug)]
    struct MockImageTransform;

    impl ImageTransform for MockImageTransform {
        fn transform(&self, image: &ImageData) -> Result<ImageData, Box<dyn std::error::Error>> {
            Ok(image.clone())
        }
    }

    #[derive(Debug)]
    struct MockFeatureExtractor;

    impl FeatureExtractor for MockFeatureExtractor {
        fn extract(&self, _image: &ImageData) -> Result<Array1<f32>, Box<dyn std::error::Error>> {
            Ok(Array1::zeros(128))
        }
    }

    #[derive(Debug)]
    struct MockCVModel;

    impl CVModel for MockCVModel {
        fn predict(&self, _image: &ImageData) -> Result<Prediction, Box<dyn std::error::Error>> {
            Ok(Prediction {
                prediction_type: PredictionType::Classification,
                confidence: 0.85,
                data: PredictionData::Classification {
                    classes: vec!["cat".to_string(), "dog".to_string()],
                    probabilities: Array1::from(vec![0.85, 0.15]),
                },
                model_info: None,
            })
        }
    }

    #[derive(Debug)]
    struct MockPostProcessor;

    impl PostProcessor for MockPostProcessor {
        fn process(
            &self,
            prediction: Prediction,
        ) -> Result<ProcessedResult, Box<dyn std::error::Error>> {
            Ok(ProcessedResult::from_prediction(
                prediction,
                Duration::from_millis(10),
                ProcessingMetadata::default(),
            ))
        }
    }

    #[test]
    fn test_pipeline_creation() {
        let config = CVConfig::default();
        let pipeline = CVPipeline::new(config);
        assert_eq!(pipeline.state, CVPipelineState::Ready);
        assert_eq!(pipeline.image_preprocessing.len(), 0);
        assert_eq!(pipeline.cv_models.len(), 0);
    }

    #[test]
    fn test_adding_components() {
        let mut pipeline = CVPipeline::new(CVConfig::default());

        // Add preprocessing
        let transform = Box::new(MockImageTransform);
        assert!(pipeline.add_preprocessing(transform).is_ok());
        assert_eq!(pipeline.image_preprocessing.len(), 1);

        // Add feature extractor
        let extractor = Box::new(MockFeatureExtractor);
        assert!(pipeline.add_feature_extraction(extractor).is_ok());
        assert_eq!(pipeline.feature_extraction.len(), 1);

        // Add model
        let model = Box::new(MockCVModel);
        assert!(pipeline.add_model(model).is_ok());
        assert_eq!(pipeline.cv_models.len(), 1);

        // Add post-processor
        let processor = Box::new(MockPostProcessor);
        assert!(pipeline.add_postprocessing(processor).is_ok());
        assert_eq!(pipeline.postprocessing.len(), 1);
    }

    #[test]
    fn test_pipeline_state_management() {
        let mut pipeline = CVPipeline::new(CVConfig::default());

        // Test pause
        assert!(pipeline.pause().is_ok());
        assert_eq!(pipeline.state, CVPipelineState::Paused);

        // Test resume
        assert!(pipeline.resume().is_ok());
        assert_eq!(pipeline.state, CVPipelineState::Ready);

        // Test stop
        assert!(pipeline.stop().is_ok());
        assert_eq!(pipeline.state, CVPipelineState::Stopped);
    }

    #[test]
    fn test_config_presets() {
        let realtime_config = CVConfig::real_time("Test Pipeline");
        assert_eq!(realtime_config.processing_mode, ProcessingMode::RealTime);
        assert_eq!(realtime_config.name, "Test Pipeline");

        let batch_config = CVConfig::high_quality_batch("Batch Pipeline");
        assert_eq!(batch_config.processing_mode, ProcessingMode::Batch);

        let streaming_config = CVConfig::streaming("Stream Pipeline");
        assert_eq!(streaming_config.processing_mode, ProcessingMode::Streaming);
    }

    #[test]
    fn test_pipeline_context() {
        let mut context = PipelineContext::new();

        context.register_component("preprocessing".to_string());
        assert_eq!(context.active_components.len(), 1);

        context.record_step_completion("preprocessing", 0);
        assert!(context.step_completions.contains_key("preprocessing"));

        context.record_timing("preprocessing", Duration::from_millis(50));
        assert!(context.step_timings.contains_key("preprocessing"));
    }

    #[test]
    fn test_prediction_types() {
        let classification = PredictionType::Classification;
        assert_eq!(classification.to_string(), "Classification");

        let detection = PredictionType::Detection;
        assert_eq!(detection.to_string(), "Detection");

        let custom = PredictionType::Custom("MyCustomType".to_string());
        assert_eq!(custom.to_string(), "Custom(MyCustomType)");
    }
}
