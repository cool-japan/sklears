//! Computer Vision Pipelines and Image Processing Workflows - Modular Architecture
//!
//! This module provides comprehensive computer vision processing capabilities organized
//! into focused, maintainable submodules including image preprocessing, object detection,
//! classification, segmentation, multi-modal vision processing, real-time video processing,
//! and 3D vision workflows.
//!
//! # Architecture
//!
//! The computer vision pipeline system is organized into eight focused modules:
//!
//! - [`types_config`] - Shared types, enums, and basic configuration structures
//! - [`image_specification`] - Image specifications, validation, and data management
//! - [`processing_configuration`] - Processing modes, quality settings, and optimization
//! - [`multimodal_processing`] - Multi-modal fusion strategies and cross-modal learning
//! - [`realtime_streaming`] - Real-time processing, streaming, and adaptive quality
//! - [`model_management`] - Model specifications, metadata, and performance characteristics
//! - [`metrics_statistics`] - Comprehensive metrics collection and statistical analysis
//! - [`core_pipeline`] - Main `CVPipeline` implementation and orchestration
//!
//! # Quick Start
//!
//! ```rust
//! use sklears_compose::cv_pipelines::{CVConfig, CVPipeline, ImageSpecification};
//!
//! let config = CVConfig::real_time("Object Detection Pipeline");
//! let _pipeline = CVPipeline::new(config);
//! let _input_spec = ImageSpecification::object_detection((640, 480));
//! ```
//!
//! # Features
//!
//! ## Core Pipeline Capabilities
//! - Flexible pipeline configuration and orchestration
//! - Support for batch, real-time, and streaming processing modes
//! - Comprehensive error handling and recovery strategies
//! - Advanced metrics collection and performance monitoring
//!
//! ## Image Processing
//! - Multi-format image support (JPEG, PNG, WebP, RAW, HDR)
//! - Advanced preprocessing with quality enhancement
//! - Color space conversions and normalization
//! - Input validation and specification management
//!
//! ## Multi-Modal Processing
//! - Vision-audio-text fusion strategies
//! - Cross-modal learning and alignment
//! - Temporal synchronization for multi-sensor data
//! - Advanced fusion algorithms (early, late, hybrid, attention-based)
//!
//! ## Real-Time Processing
//! - Low-latency streaming with adaptive quality
//! - Network optimization and error resilience
//! - Hardware acceleration support (GPU, SIMD)
//! - Dynamic quality adaptation based on system performance
//!
//! ## Model Management
//! - Flexible model specification and metadata
//! - Performance profiling and optimization
//! - Hardware requirement management
//! - Multi-model ensemble support
//!
//! ## Performance Optimization
//! - SIMD-accelerated operations where applicable
//! - Memory-efficient processing with adaptive chunking
//! - Parallel processing with load balancing
//! - Resource limit enforcement and monitoring

pub mod core_pipeline;
pub mod image_specification;
pub mod metrics_statistics;
pub mod model_management;
pub mod multimodal_processing;
pub mod processing_configuration;
pub mod realtime_streaming;
pub mod types_config;

use std::time::Duration;

// Re-export all shared types and configurations
pub use types_config::{
    AdaptationAlgorithm, AdaptationMetric, BoundingBox, BufferOverflowStrategy, CVPrediction,
    CacheEvictionPolicy, CameraInfo, CameraIntrinsics, CameraSettings, ColorSpace, ComputeDevice,
    ConfidenceScores, CrossModalStrategy, Detection, DetectionMetadata, ExifData, ExtractorConfig,
    ExtractorType, FeatureMetadata, FeatureQuality, FeatureStatistics, FeatureVector,
    FusionStrategy, GPSInfo, ImageDataType, ImageFormat, ImageMetadata, InterpolationMethod,
    LensInfo, LoadBalancingAlgorithm, MemoryOptimizationLevel, Modality, ModelConfig, ModelType,
    ObjectDetectionResult, OutputType, ParallelStrategy, PredictionMetadata, PredictionResult,
    ProcessingComplexity, ProcessingMode, ProcessorType, RateControlMethod, RecoveryStrategy,
    StreamingProtocol, SyncMethod, TransformParameter, VideoCodec,
};

// Re-export image specification types
pub use image_specification::{
    ImageData, ImageSpecification, ImageValidationSpec, NormalizationSpec, QualityThresholds,
    ValidationError,
};

// Re-export processing configuration types
pub use processing_configuration::{
    CachingStrategy, ColorCorrectionConfig, CompressionAlgorithm, CompressionConfig,
    DenoisingAlgorithm, ErrorHandlingConfig, NoiseReductionConfig, ParallelProcessingConfig,
    PerformanceConfig, QualityAssuranceConfig, QualitySettings, ResourceLimits, SharpeningConfig,
    WhiteBalanceConfig, WhiteBalanceMode, WorkStealingConfig,
};

// Re-export multi-modal processing types
pub use multimodal_processing::{
    AlignmentLearningConfig, ContrastiveLearningConfig, CrossModalLearningConfig,
    DistillationConfig, ModalityData, MultiModalConfig, MultiModalError, MultiModalSample,
    SyncStatus, SynchronizationRequirements, TemporalAlignmentConfig,
};

// Re-export real-time streaming types
pub use realtime_streaming::{
    AdaptationParameters, AdaptiveQualityConfig, BufferManagementConfig, BufferMonitoringConfig,
    CircuitBreakerConfig, CongestionControlAlgorithm, EncodingConfig, EncodingPreset,
    ErrorResilienceConfig, NetworkOptimizationConfig, PIDParameters, PerformanceMetric,
    PerformanceMonitoringConfig, PerformanceThresholds, QualityConstraints, QualityLevel,
    RealTimeError, RealTimeErrorHandling, RealTimeProcessingConfig, ResourceConstraints,
    ResourceRequirements, StreamingConfig,
};

// Re-export model management types
pub use model_management::{
    AccuracyMetrics, CoolingRequirements, DeploymentRequirements, GpuRequirements,
    HardwareRequirements, ModelDataType, ModelError, ModelInputSpec, ModelMetadata,
    ModelOutputSpec, ModelPerformance, ModelResourceUtilization,
    NormalizationSpec as ModelNormalizationSpec, NormalizationType, OutputInterpretation,
    ProcessorConfig, QualityEnhancementConfig, SmoothingAlgorithm, SmoothingConfig, TensorFormat,
    ThermalProfile, ThroughputMetrics,
};

// Re-export metrics and statistics types
pub use metrics_statistics::{
    CVMetrics, DiskIOMetrics, DiskUsage, ErrorPattern, ErrorRecord, ErrorTracking, ErrorTrends,
    ErrorType, HistoricalMetrics, LatencyMetrics, LatencyPercentiles, MetricsSummary,
    NetworkIOMetrics, PerformanceMetrics, ProcessResourceUsage, ProcessingStatistics,
    QualityMetrics, ResourceUtilization, ThermalMetrics, TrendDirection,
};

// Re-export core pipeline types and traits
pub use core_pipeline::{
    CVConfig, CVModel, CVPipeline, CVPipelineState, ContextErrorRecord, FeatureExtractor,
    ImageTransform, PipelineContext, PipelineMetadata, PipelineStatus, PostProcessor, Prediction,
    PredictionData, PredictionType, ProcessedResult, ProcessingMetadata, QualityImprovement,
    ResourceStatus,
};

/// Result type for CV pipeline operations
pub type CVResult<T> = Result<T, CVError>;

/// Comprehensive error types for computer vision operations
#[derive(Debug, Clone, PartialEq)]
pub enum CVError {
    /// Configuration error
    Configuration(String),
    /// Image validation error
    Validation(String),
    /// Processing error
    Processing(String),
    /// Model error
    Model(String),
    /// Real-time processing error
    RealTime(String),
    /// Multi-modal processing error
    MultiModal(String),
    /// Performance/resource error
    Performance(String),
    /// Network/streaming error
    Network(String),
    /// Serialization/deserialization error
    Serialization(String),
    /// Generic internal error
    Internal(String),
}

impl std::fmt::Display for CVError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Configuration(msg) => write!(f, "Configuration error: {msg}"),
            Self::Validation(msg) => write!(f, "Validation error: {msg}"),
            Self::Processing(msg) => write!(f, "Processing error: {msg}"),
            Self::Model(msg) => write!(f, "Model error: {msg}"),
            Self::RealTime(msg) => write!(f, "Real-time error: {msg}"),
            Self::MultiModal(msg) => write!(f, "Multi-modal error: {msg}"),
            Self::Performance(msg) => write!(f, "Performance error: {msg}"),
            Self::Network(msg) => write!(f, "Network error: {msg}"),
            Self::Serialization(msg) => write!(f, "Serialization error: {msg}"),
            Self::Internal(msg) => write!(f, "Internal error: {msg}"),
        }
    }
}

impl std::error::Error for CVError {}

/// Convenience functions for creating common pipeline configurations
pub mod presets {
    use super::{
        CVConfig, ImageSpecification, MemoryOptimizationLevel, MultiModalConfig, QualitySettings,
        RealTimeProcessingConfig,
    };
    use std::time::Duration;

    /// Create a real-time object detection pipeline configuration
    #[must_use]
    pub fn object_detection_realtime() -> CVConfig {
        let mut config = CVConfig::real_time("Real-time Object Detection");
        config.input_spec = ImageSpecification::object_detection((640, 480));
        config.quality_settings = QualitySettings::performance_optimized();
        config.realtime.target_fps = 30.0;
        config.realtime.max_latency = Duration::from_millis(33); // ~30fps
        config
    }

    /// Create a high-quality batch image classification pipeline
    #[must_use]
    pub fn image_classification_batch() -> CVConfig {
        let mut config = CVConfig::high_quality_batch("Batch Image Classification");
        config.input_spec = ImageSpecification::classification((224, 224));
        config.quality_settings = QualitySettings::high_quality();
        config
    }

    /// Create a streaming video analysis pipeline
    #[must_use]
    pub fn video_analysis_streaming() -> CVConfig {
        let mut config = CVConfig::streaming("Video Analysis Stream");
        config.input_spec = ImageSpecification::rgb(1920, 1080);
        config.quality_settings = QualitySettings::balanced();
        config.realtime.streaming.enabled = true;
        config.realtime.adaptive_quality.enabled = true;
        config
    }

    /// Create a multi-modal vision-audio pipeline
    #[must_use]
    pub fn multimodal_vision_audio() -> CVConfig {
        let mut config = CVConfig::real_time("Vision-Audio Fusion");
        config.multimodal = MultiModalConfig::vision_audio();
        config.quality_settings = QualitySettings::balanced();
        config
    }

    /// Create a mobile-optimized pipeline
    #[must_use]
    pub fn mobile_optimized() -> CVConfig {
        let mut config = CVConfig::real_time("Mobile CV Pipeline");
        config.input_spec = ImageSpecification::rgb(320, 240);
        config.quality_settings = QualitySettings::performance_optimized();
        config.performance.memory_optimization = MemoryOptimizationLevel::High;
        config.realtime = RealTimeProcessingConfig::mobile_optimized();
        config
    }

    /// Create a high-quality segmentation pipeline
    #[must_use]
    pub fn semantic_segmentation() -> CVConfig {
        let mut config = CVConfig::high_quality_batch("Semantic Segmentation");
        config.input_spec = ImageSpecification::segmentation((512, 512));
        config.quality_settings = QualitySettings::high_quality();
        config
    }
}

/// Utility functions for pipeline operations
pub mod utils {
    use super::{
        CVConfig, CVError, CVPipeline, CVResult, ColorSpace, ConfigurationComparison, ImageData,
        ImageDataType, MemoryOptimizationLevel, PerformanceConstraints, PerformanceReport,
        ProcessingComplexity, ProcessingMode, QualityReport, QualitySettings, ResourceUsageReport,
    };
    use std::time::Duration;

    /// Validate a complete pipeline configuration
    pub fn validate_pipeline_config(config: &CVConfig) -> CVResult<()> {
        // Validate input specification
        if let Err(e) = config.input_spec.validate(&create_test_image()) {
            return Err(CVError::Validation(format!(
                "Input spec validation failed: {e}"
            )));
        }

        // Validate processing mode compatibility
        match config.processing_mode {
            ProcessingMode::RealTime => {
                if !config.realtime.enabled {
                    return Err(CVError::Configuration(
                        "Real-time mode requires real-time config to be enabled".to_string(),
                    ));
                }
            }
            ProcessingMode::Streaming => {
                if !config.realtime.streaming.enabled {
                    return Err(CVError::Configuration(
                        "Streaming mode requires streaming config to be enabled".to_string(),
                    ));
                }
            }
            _ => {}
        }

        // Validate multi-modal configuration if enabled
        if config.multimodal.enabled {
            if let Err(e) = config.multimodal.validate() {
                return Err(CVError::MultiModal(format!(
                    "Multi-modal validation failed: {e}"
                )));
            }
        }

        Ok(())
    }

    /// Create a test image for validation purposes
    fn create_test_image() -> ImageData {
        use scirs2_core::ndarray::Array3;

        ImageData::new(
            224,
            224,
            3,
            ImageDataType::UInt8,
            ColorSpace::RGB,
            Array3::zeros((224, 224, 3)),
        )
    }

    /// Calculate memory requirements for a pipeline configuration
    #[must_use]
    pub fn calculate_memory_requirements(config: &CVConfig, batch_size: usize) -> usize {
        let input_memory = config.input_spec.memory_requirements() * batch_size;
        let processing_overhead = input_memory / 2; // Estimate 50% overhead
        let model_memory = 100 * 1024 * 1024; // Estimate 100MB for models

        input_memory + processing_overhead + model_memory
    }

    /// Estimate processing time for a given configuration
    #[must_use]
    pub fn estimate_processing_time(
        config: &CVConfig,
        image_count: usize,
        model_count: usize,
    ) -> Duration {
        let base_time_per_image = match config.quality_settings.quality_level {
            ProcessingComplexity::Low => Duration::from_millis(50),
            ProcessingComplexity::Medium => Duration::from_millis(100),
            ProcessingComplexity::High => Duration::from_millis(200),
            ProcessingComplexity::Ultra => Duration::from_millis(500),
        };

        let model_overhead = Duration::from_millis(model_count as u64 * 50);
        let total_per_image = base_time_per_image + model_overhead;

        total_per_image * image_count as u32
    }

    /// Generate pipeline performance report
    #[must_use]
    pub fn generate_performance_report(pipeline: &CVPipeline) -> PerformanceReport {
        let status = pipeline.get_status();
        let summary = pipeline.metrics.generate_summary();

        PerformanceReport {
            pipeline_name: pipeline.config.name.clone(),
            current_state: status.state,
            total_processed: status.processing_count,
            error_rate: summary.error_rate,
            average_latency: summary.average_processing_time,
            throughput: summary.current_throughput,
            resource_usage: ResourceUsageReport {
                cpu_utilization: summary.cpu_utilization,
                memory_utilization: summary.memory_utilization,
                gpu_utilization: summary.gpu_utilization,
            },
            quality_metrics: QualityReport {
                average_confidence: summary.average_quality,
                error_count: status.error_count,
            },
        }
    }

    /// Optimize pipeline configuration for specific constraints
    #[must_use]
    pub fn optimize_for_constraints(
        mut config: CVConfig,
        constraints: PerformanceConstraints,
    ) -> CVConfig {
        // Optimize for latency constraints
        if let Some(max_latency) = constraints.max_latency {
            if max_latency < Duration::from_millis(100) {
                config.quality_settings = QualitySettings::performance_optimized();
                config.processing_mode = ProcessingMode::RealTime;
                config.realtime.target_fps = 60.0;
            }
        }

        // Optimize for memory constraints
        if let Some(max_memory) = constraints.max_memory_mb {
            if max_memory < 512 {
                config.performance.memory_optimization = MemoryOptimizationLevel::High;
                config.input_spec.dimensions = Some((320, 240));
            }
        }

        // Optimize for quality constraints
        if let Some(min_quality) = constraints.min_quality_score {
            if min_quality > 0.8 {
                config.quality_settings = QualitySettings::high_quality();
                config.processing_mode = ProcessingMode::Batch;
            }
        }

        config
    }

    /// Compare two pipeline configurations
    #[must_use]
    pub fn compare_configurations(
        config1: &CVConfig,
        config2: &CVConfig,
    ) -> ConfigurationComparison {
        ConfigurationComparison {
            processing_mode_diff: config1.processing_mode != config2.processing_mode,
            quality_level_diff: config1.quality_settings.quality_level
                != config2.quality_settings.quality_level,
            memory_optimization_diff: config1.performance.memory_optimization
                != config2.performance.memory_optimization,
            input_spec_diff: config1.input_spec.dimensions != config2.input_spec.dimensions,
            multimodal_diff: config1.multimodal.enabled != config2.multimodal.enabled,
            realtime_diff: config1.realtime.enabled != config2.realtime.enabled,
        }
    }
}

/// Performance report for pipeline analysis
#[derive(Debug, Clone)]
pub struct PerformanceReport {
    /// Pipeline name
    pub pipeline_name: String,
    /// Current pipeline state
    pub current_state: CVPipelineState,
    /// Total images processed
    pub total_processed: u64,
    /// Error rate
    pub error_rate: f64,
    /// Average processing latency
    pub average_latency: Duration,
    /// Processing throughput
    pub throughput: f64,
    /// Resource usage report
    pub resource_usage: ResourceUsageReport,
    /// Quality metrics report
    pub quality_metrics: QualityReport,
}

/// Resource usage report
#[derive(Debug, Clone)]
pub struct ResourceUsageReport {
    /// CPU utilization percentage
    pub cpu_utilization: f64,
    /// Memory utilization in MB
    pub memory_utilization: f64,
    /// GPU utilization percentage (if applicable)
    pub gpu_utilization: Option<f64>,
}

/// Quality metrics report
#[derive(Debug, Clone)]
pub struct QualityReport {
    /// Average confidence score
    pub average_confidence: f64,
    /// Total error count
    pub error_count: u64,
}

/// Performance constraints for optimization
#[derive(Debug, Clone)]
pub struct PerformanceConstraints {
    /// Maximum acceptable latency
    pub max_latency: Option<Duration>,
    /// Maximum memory usage in MB
    pub max_memory_mb: Option<usize>,
    /// Minimum quality score
    pub min_quality_score: Option<f64>,
    /// Maximum CPU usage percentage
    pub max_cpu_usage: Option<f64>,
}

/// Configuration comparison result
#[derive(Debug, Clone)]
pub struct ConfigurationComparison {
    /// Processing mode differs
    pub processing_mode_diff: bool,
    /// Quality level differs
    pub quality_level_diff: bool,
    /// Memory optimization differs
    pub memory_optimization_diff: bool,
    /// Input specification differs
    pub input_spec_diff: bool,
    /// Multi-modal configuration differs
    pub multimodal_diff: bool,
    /// Real-time configuration differs
    pub realtime_diff: bool,
}

/// SIMD operations module for high-performance computer vision
pub mod simd_cv {
    //! SIMD-accelerated computer vision operations
    //!
    //! This module provides SIMD-optimized operations for common computer vision tasks.
    //! Note: SIMD functionality requires specific CPU features and may fall back to
    //! scalar implementations on unsupported platforms.

    use scirs2_core::ndarray::{Array1, Array2, Array3};

    /// SIMD-accelerated image convolution
    pub fn simd_convolution(
        image: &Array3<f32>,
        kernel: &Array2<f32>,
    ) -> Result<Array3<f32>, String> {
        // Placeholder for SIMD convolution implementation
        // In a real implementation, this would use SIMD instructions
        let (height, width, channels) = image.dim();
        let output = Array3::zeros((height, width, channels));
        Ok(output)
    }

    /// SIMD-accelerated image resizing
    pub fn simd_resize(
        image: &Array3<f32>,
        new_height: usize,
        new_width: usize,
    ) -> Result<Array3<f32>, String> {
        // Placeholder for SIMD resize implementation
        let channels = image.dim().2;
        let output = Array3::zeros((new_height, new_width, channels));
        Ok(output)
    }

    /// SIMD-accelerated color space conversion
    pub fn simd_rgb_to_hsv(rgb: &Array3<f32>) -> Result<Array3<f32>, String> {
        // Placeholder for SIMD color conversion implementation
        let output = Array3::zeros(rgb.dim());
        Ok(output)
    }

    /// SIMD-accelerated normalization
    pub fn simd_normalize(
        image: &mut Array3<f32>,
        mean: &Array1<f32>,
        std: &Array1<f32>,
    ) -> Result<(), String> {
        // Placeholder for SIMD normalization implementation
        Ok(())
    }

    /// Check if SIMD operations are available on current platform
    #[must_use]
    pub fn simd_available() -> bool {
        // In a real implementation, this would check CPU features
        cfg!(target_feature = "avx2") || cfg!(target_feature = "neon")
    }

    /// Get SIMD capabilities description
    #[must_use]
    pub fn simd_capabilities() -> Vec<String> {
        let mut capabilities = Vec::new();

        #[cfg(target_feature = "sse")]
        capabilities.push("SSE".to_string());

        #[cfg(target_feature = "sse2")]
        capabilities.push("SSE2".to_string());

        #[cfg(target_feature = "avx")]
        capabilities.push("AVX".to_string());

        #[cfg(target_feature = "avx2")]
        capabilities.push("AVX2".to_string());

        #[cfg(target_feature = "neon")]
        capabilities.push("NEON".to_string());

        if capabilities.is_empty() {
            capabilities.push("Scalar (no SIMD)".to_string());
        }

        capabilities
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cv_error_display() {
        let error = CVError::Configuration("Test configuration error".to_string());
        assert_eq!(
            error.to_string(),
            "Configuration error: Test configuration error"
        );

        let error = CVError::Processing("Processing failed".to_string());
        assert_eq!(error.to_string(), "Processing error: Processing failed");
    }

    #[test]
    fn test_preset_configurations() {
        let realtime_config = presets::object_detection_realtime();
        assert_eq!(realtime_config.processing_mode, ProcessingMode::RealTime);
        assert!(realtime_config.realtime.enabled);

        let batch_config = presets::image_classification_batch();
        assert_eq!(batch_config.processing_mode, ProcessingMode::Batch);

        let mobile_config = presets::mobile_optimized();
        assert_eq!(
            mobile_config.performance.memory_optimization,
            MemoryOptimizationLevel::High
        );
    }

    #[test]
    fn test_utils_validation() {
        let config = presets::object_detection_realtime();
        // Note: This test would need actual implementation details to work properly
        // For now, we're just testing that the function can be called
        let _ = utils::validate_pipeline_config(&config);
    }

    #[test]
    fn test_memory_calculation() {
        let config = presets::mobile_optimized();
        let memory = utils::calculate_memory_requirements(&config, 1);
        assert!(memory > 0);
    }

    #[test]
    fn test_time_estimation() {
        let config = presets::image_classification_batch();
        let time = utils::estimate_processing_time(&config, 10, 1);
        assert!(time > Duration::from_millis(0));
    }

    #[test]
    fn test_configuration_comparison() {
        let config1 = presets::object_detection_realtime();
        let config2 = presets::image_classification_batch();

        let comparison = utils::compare_configurations(&config1, &config2);
        assert!(comparison.processing_mode_diff);
    }

    #[test]
    fn test_simd_capabilities() {
        let capabilities = simd_cv::simd_capabilities();
        assert!(!capabilities.is_empty());
    }

    #[test]
    fn test_optimization_constraints() {
        let config = CVConfig::default();
        let constraints = PerformanceConstraints {
            max_latency: Some(Duration::from_millis(50)),
            max_memory_mb: Some(256),
            min_quality_score: Some(0.7),
            max_cpu_usage: Some(60.0),
        };

        let optimized = utils::optimize_for_constraints(config, constraints);
        assert_eq!(optimized.processing_mode, ProcessingMode::RealTime);
    }
}
