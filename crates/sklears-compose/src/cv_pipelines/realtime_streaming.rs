//! Real-time processing and streaming configuration
//!
//! This module provides comprehensive real-time processing capabilities including
//! streaming protocols, encoding configuration, network optimization, adaptive
//! quality management, and buffer management for computer vision pipelines.

use super::types_config::{
    AdaptationAlgorithm, AdaptationMetric, BufferOverflowStrategy, ProcessingComplexity,
    RateControlMethod, RecoveryStrategy, StreamingProtocol, TransformParameter, VideoCodec,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// Real-time processing configuration for computer vision
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealTimeProcessingConfig {
    /// Enable real-time processing
    pub enabled: bool,
    /// Target frame rate (fps)
    pub target_fps: f64,
    /// Maximum acceptable latency
    pub max_latency: Duration,
    /// Buffer management configuration
    pub buffer_management: BufferManagementConfig,
    /// Adaptive quality configuration
    pub adaptive_quality: AdaptiveQualityConfig,
    /// Streaming configuration
    pub streaming: StreamingConfig,
    /// Performance monitoring
    pub performance_monitoring: PerformanceMonitoringConfig,
    /// Error handling for real-time scenarios
    pub error_handling: RealTimeErrorHandling,
}

impl Default for RealTimeProcessingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            target_fps: 30.0,
            max_latency: Duration::from_millis(100),
            buffer_management: BufferManagementConfig::default(),
            adaptive_quality: AdaptiveQualityConfig::default(),
            streaming: StreamingConfig::default(),
            performance_monitoring: PerformanceMonitoringConfig::default(),
            error_handling: RealTimeErrorHandling::default(),
        }
    }
}

impl RealTimeProcessingConfig {
    /// Create configuration for low-latency applications
    #[must_use]
    pub fn low_latency() -> Self {
        Self {
            enabled: true,
            target_fps: 60.0,
            max_latency: Duration::from_millis(16), // ~1 frame at 60fps
            buffer_management: BufferManagementConfig::minimal(),
            adaptive_quality: AdaptiveQualityConfig::responsive(),
            streaming: StreamingConfig::low_latency(),
            performance_monitoring: PerformanceMonitoringConfig::aggressive(),
            error_handling: RealTimeErrorHandling::fast_recovery(),
        }
    }

    /// Create configuration for high-quality streaming
    #[must_use]
    pub fn high_quality_streaming() -> Self {
        Self {
            enabled: true,
            target_fps: 30.0,
            max_latency: Duration::from_millis(200),
            buffer_management: BufferManagementConfig::quality_focused(),
            adaptive_quality: AdaptiveQualityConfig::quality_preserving(),
            streaming: StreamingConfig::high_quality(),
            performance_monitoring: PerformanceMonitoringConfig::comprehensive(),
            error_handling: RealTimeErrorHandling::robust(),
        }
    }

    /// Create configuration for mobile/constrained environments
    #[must_use]
    pub fn mobile_optimized() -> Self {
        Self {
            enabled: true,
            target_fps: 15.0,
            max_latency: Duration::from_millis(300),
            buffer_management: BufferManagementConfig::memory_efficient(),
            adaptive_quality: AdaptiveQualityConfig::bandwidth_aware(),
            streaming: StreamingConfig::mobile_friendly(),
            performance_monitoring: PerformanceMonitoringConfig::lightweight(),
            error_handling: RealTimeErrorHandling::graceful_degradation(),
        }
    }

    /// Validate configuration for consistency
    pub fn validate(&self) -> Result<(), RealTimeError> {
        if self.enabled {
            if self.target_fps <= 0.0 {
                return Err(RealTimeError::InvalidConfiguration(
                    "Target FPS must be positive".to_string(),
                ));
            }

            if self.max_latency.is_zero() {
                return Err(RealTimeError::InvalidConfiguration(
                    "Maximum latency must be greater than zero".to_string(),
                ));
            }

            // Check buffer configuration consistency
            self.buffer_management.validate()?;
            self.adaptive_quality.validate()?;
            self.streaming.validate()?;
        }

        Ok(())
    }
}

/// Buffer management configuration for real-time processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BufferManagementConfig {
    /// Input buffer size (number of frames)
    pub input_buffer_size: usize,
    /// Output buffer size (number of frames)
    pub output_buffer_size: usize,
    /// Processing buffer size
    pub processing_buffer_size: usize,
    /// Buffer overflow strategy
    pub overflow_strategy: BufferOverflowStrategy,
    /// Enable buffer pre-allocation
    pub pre_allocation: bool,
    /// Enable dynamic buffer resizing
    pub dynamic_resizing: bool,
    /// Buffer monitoring configuration
    pub monitoring: BufferMonitoringConfig,
}

impl Default for BufferManagementConfig {
    fn default() -> Self {
        Self {
            input_buffer_size: 3,
            output_buffer_size: 2,
            processing_buffer_size: 1,
            overflow_strategy: BufferOverflowStrategy::DropOldest,
            pre_allocation: true,
            dynamic_resizing: false,
            monitoring: BufferMonitoringConfig::default(),
        }
    }
}

impl BufferManagementConfig {
    /// Create minimal buffer configuration for lowest latency
    #[must_use]
    pub fn minimal() -> Self {
        Self {
            input_buffer_size: 1,
            output_buffer_size: 1,
            processing_buffer_size: 1,
            overflow_strategy: BufferOverflowStrategy::DropNewest,
            pre_allocation: true,
            dynamic_resizing: false,
            monitoring: BufferMonitoringConfig::lightweight(),
        }
    }

    /// Create quality-focused buffer configuration
    #[must_use]
    pub fn quality_focused() -> Self {
        Self {
            input_buffer_size: 10,
            output_buffer_size: 5,
            processing_buffer_size: 3,
            overflow_strategy: BufferOverflowStrategy::ReduceQuality,
            pre_allocation: true,
            dynamic_resizing: true,
            monitoring: BufferMonitoringConfig::comprehensive(),
        }
    }

    /// Create memory-efficient buffer configuration
    #[must_use]
    pub fn memory_efficient() -> Self {
        Self {
            input_buffer_size: 2,
            output_buffer_size: 1,
            processing_buffer_size: 1,
            overflow_strategy: BufferOverflowStrategy::SkipFrames,
            pre_allocation: false,
            dynamic_resizing: true,
            monitoring: BufferMonitoringConfig::essential(),
        }
    }

    /// Validate buffer configuration
    pub fn validate(&self) -> Result<(), RealTimeError> {
        if self.input_buffer_size == 0 || self.output_buffer_size == 0 {
            return Err(RealTimeError::InvalidConfiguration(
                "Buffer sizes must be greater than zero".to_string(),
            ));
        }

        Ok(())
    }

    /// Calculate total memory usage for buffers
    #[must_use]
    pub fn memory_usage(&self, frame_size_bytes: usize) -> usize {
        (self.input_buffer_size + self.output_buffer_size + self.processing_buffer_size)
            * frame_size_bytes
    }
}

/// Buffer monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BufferMonitoringConfig {
    /// Enable buffer usage monitoring
    pub enabled: bool,
    /// Monitoring interval
    pub interval: Duration,
    /// Alert threshold for buffer usage (0.0-1.0)
    pub usage_threshold: f64,
    /// Track buffer overflow events
    pub track_overflows: bool,
    /// Track buffer underruns
    pub track_underruns: bool,
}

impl Default for BufferMonitoringConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            interval: Duration::from_secs(1),
            usage_threshold: 0.8,
            track_overflows: true,
            track_underruns: true,
        }
    }
}

impl BufferMonitoringConfig {
    /// Create lightweight monitoring configuration
    #[must_use]
    pub fn lightweight() -> Self {
        Self {
            enabled: true,
            interval: Duration::from_secs(5),
            usage_threshold: 0.9,
            track_overflows: true,
            track_underruns: false,
        }
    }

    /// Create comprehensive monitoring configuration
    #[must_use]
    pub fn comprehensive() -> Self {
        Self {
            enabled: true,
            interval: Duration::from_millis(100),
            usage_threshold: 0.7,
            track_overflows: true,
            track_underruns: true,
        }
    }

    /// Create essential monitoring configuration
    #[must_use]
    pub fn essential() -> Self {
        Self {
            enabled: true,
            interval: Duration::from_secs(10),
            usage_threshold: 0.95,
            track_overflows: true,
            track_underruns: false,
        }
    }
}

/// Adaptive quality configuration for real-time processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveQualityConfig {
    /// Enable adaptive quality control
    pub enabled: bool,
    /// Available quality levels
    pub quality_levels: Vec<QualityLevel>,
    /// Quality adaptation algorithm
    pub adaptation_algorithm: AdaptationAlgorithm,
    /// Metrics used for adaptation decisions
    pub adaptation_metrics: Vec<AdaptationMetric>,
    /// Adaptation parameters
    pub adaptation_params: AdaptationParameters,
    /// Quality constraints
    pub constraints: QualityConstraints,
}

impl Default for AdaptiveQualityConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            quality_levels: vec![
                QualityLevel::low(),
                QualityLevel::medium(),
                QualityLevel::high(),
            ],
            adaptation_algorithm: AdaptationAlgorithm::Threshold,
            adaptation_metrics: vec![
                AdaptationMetric::FrameRate,
                AdaptationMetric::Latency,
                AdaptationMetric::CPUUsage,
            ],
            adaptation_params: AdaptationParameters::default(),
            constraints: QualityConstraints::default(),
        }
    }
}

impl AdaptiveQualityConfig {
    /// Create responsive adaptive quality configuration
    #[must_use]
    pub fn responsive() -> Self {
        Self {
            enabled: true,
            quality_levels: vec![QualityLevel::low(), QualityLevel::medium()],
            adaptation_algorithm: AdaptationAlgorithm::PID,
            adaptation_metrics: vec![AdaptationMetric::FrameRate, AdaptationMetric::Latency],
            adaptation_params: AdaptationParameters::responsive(),
            constraints: QualityConstraints::flexible(),
        }
    }

    /// Create quality-preserving adaptive configuration
    #[must_use]
    pub fn quality_preserving() -> Self {
        Self {
            enabled: true,
            quality_levels: vec![
                QualityLevel::medium(),
                QualityLevel::high(),
                QualityLevel::ultra(),
            ],
            adaptation_algorithm: AdaptationAlgorithm::Predictive,
            adaptation_metrics: vec![
                AdaptationMetric::QualityScore,
                AdaptationMetric::Accuracy,
                AdaptationMetric::FrameRate,
            ],
            adaptation_params: AdaptationParameters::conservative(),
            constraints: QualityConstraints::strict(),
        }
    }

    /// Create bandwidth-aware adaptive configuration
    #[must_use]
    pub fn bandwidth_aware() -> Self {
        Self {
            enabled: true,
            quality_levels: vec![QualityLevel::low(), QualityLevel::medium()],
            adaptation_algorithm: AdaptationAlgorithm::Fuzzy,
            adaptation_metrics: vec![
                AdaptationMetric::MemoryUsage,
                AdaptationMetric::CPUUsage,
                AdaptationMetric::FrameRate,
            ],
            adaptation_params: AdaptationParameters::bandwidth_focused(),
            constraints: QualityConstraints::resource_limited(),
        }
    }

    /// Validate adaptive quality configuration
    pub fn validate(&self) -> Result<(), RealTimeError> {
        if self.enabled && self.quality_levels.is_empty() {
            return Err(RealTimeError::InvalidConfiguration(
                "Quality levels cannot be empty when adaptive quality is enabled".to_string(),
            ));
        }

        for level in &self.quality_levels {
            level.validate()?;
        }

        Ok(())
    }
}

/// Quality level configuration for adaptive processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityLevel {
    /// Quality level identifier
    pub id: String,
    /// Image resolution (width, height)
    pub resolution: (usize, usize),
    /// Processing complexity
    pub complexity: ProcessingComplexity,
    /// Expected frame rate
    pub expected_fps: f64,
    /// Quality score (0.0-1.0)
    pub quality_score: f64,
    /// Resource requirements
    pub resource_requirements: ResourceRequirements,
}

impl QualityLevel {
    /// Create low quality level
    #[must_use]
    pub fn low() -> Self {
        Self {
            id: "low".to_string(),
            resolution: (320, 240),
            complexity: ProcessingComplexity::Low,
            expected_fps: 60.0,
            quality_score: 0.3,
            resource_requirements: ResourceRequirements::minimal(),
        }
    }

    /// Create medium quality level
    #[must_use]
    pub fn medium() -> Self {
        Self {
            id: "medium".to_string(),
            resolution: (640, 480),
            complexity: ProcessingComplexity::Medium,
            expected_fps: 30.0,
            quality_score: 0.6,
            resource_requirements: ResourceRequirements::moderate(),
        }
    }

    /// Create high quality level
    #[must_use]
    pub fn high() -> Self {
        Self {
            id: "high".to_string(),
            resolution: (1280, 720),
            complexity: ProcessingComplexity::High,
            expected_fps: 15.0,
            quality_score: 0.8,
            resource_requirements: ResourceRequirements::high(),
        }
    }

    /// Create ultra quality level
    #[must_use]
    pub fn ultra() -> Self {
        Self {
            id: "ultra".to_string(),
            resolution: (1920, 1080),
            complexity: ProcessingComplexity::Ultra,
            expected_fps: 10.0,
            quality_score: 1.0,
            resource_requirements: ResourceRequirements::maximum(),
        }
    }

    /// Validate quality level configuration
    pub fn validate(&self) -> Result<(), RealTimeError> {
        if self.expected_fps <= 0.0 {
            return Err(RealTimeError::InvalidConfiguration(format!(
                "Expected FPS must be positive for quality level '{}'",
                self.id
            )));
        }

        if !(0.0..=1.0).contains(&self.quality_score) {
            return Err(RealTimeError::InvalidConfiguration(format!(
                "Quality score must be between 0.0 and 1.0 for quality level '{}'",
                self.id
            )));
        }

        Ok(())
    }
}

/// Resource requirements for quality levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceRequirements {
    /// CPU usage percentage
    pub cpu_usage: f64,
    /// Memory usage in bytes
    pub memory_usage: usize,
    /// GPU memory usage in bytes (if applicable)
    pub gpu_memory_usage: Option<usize>,
    /// Network bandwidth in bytes/second
    pub bandwidth: usize,
}

impl ResourceRequirements {
    /// Create minimal resource requirements
    #[must_use]
    pub fn minimal() -> Self {
        Self {
            cpu_usage: 20.0,
            memory_usage: 50 * 1024 * 1024,            // 50MB
            gpu_memory_usage: Some(100 * 1024 * 1024), // 100MB
            bandwidth: 1024 * 1024,                    // 1MB/s
        }
    }

    /// Create moderate resource requirements
    #[must_use]
    pub fn moderate() -> Self {
        Self {
            cpu_usage: 50.0,
            memory_usage: 200 * 1024 * 1024,           // 200MB
            gpu_memory_usage: Some(500 * 1024 * 1024), // 500MB
            bandwidth: 5 * 1024 * 1024,                // 5MB/s
        }
    }

    /// Create high resource requirements
    #[must_use]
    pub fn high() -> Self {
        Self {
            cpu_usage: 80.0,
            memory_usage: 500 * 1024 * 1024,            // 500MB
            gpu_memory_usage: Some(1024 * 1024 * 1024), // 1GB
            bandwidth: 15 * 1024 * 1024,                // 15MB/s
        }
    }

    /// Create maximum resource requirements
    #[must_use]
    pub fn maximum() -> Self {
        Self {
            cpu_usage: 95.0,
            memory_usage: 1024 * 1024 * 1024,               // 1GB
            gpu_memory_usage: Some(2 * 1024 * 1024 * 1024), // 2GB
            bandwidth: 50 * 1024 * 1024,                    // 50MB/s
        }
    }
}

/// Adaptation parameters for quality control
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptationParameters {
    /// Adaptation sensitivity (0.0-1.0)
    pub sensitivity: f64,
    /// Adaptation interval
    pub adaptation_interval: Duration,
    /// Hysteresis for preventing oscillation
    pub hysteresis: f64,
    /// Minimum time between adaptations
    pub min_adaptation_interval: Duration,
    /// PID controller parameters (if using PID algorithm)
    pub pid_params: Option<PIDParameters>,
}

impl Default for AdaptationParameters {
    fn default() -> Self {
        Self {
            sensitivity: 0.5,
            adaptation_interval: Duration::from_secs(2),
            hysteresis: 0.1,
            min_adaptation_interval: Duration::from_millis(500),
            pid_params: Some(PIDParameters::default()),
        }
    }
}

impl AdaptationParameters {
    /// Create responsive adaptation parameters
    #[must_use]
    pub fn responsive() -> Self {
        Self {
            sensitivity: 0.8,
            adaptation_interval: Duration::from_millis(500),
            hysteresis: 0.05,
            min_adaptation_interval: Duration::from_millis(100),
            pid_params: Some(PIDParameters::responsive()),
        }
    }

    /// Create conservative adaptation parameters
    #[must_use]
    pub fn conservative() -> Self {
        Self {
            sensitivity: 0.3,
            adaptation_interval: Duration::from_secs(5),
            hysteresis: 0.2,
            min_adaptation_interval: Duration::from_secs(2),
            pid_params: Some(PIDParameters::conservative()),
        }
    }

    /// Create bandwidth-focused adaptation parameters
    #[must_use]
    pub fn bandwidth_focused() -> Self {
        Self {
            sensitivity: 0.6,
            adaptation_interval: Duration::from_secs(1),
            hysteresis: 0.15,
            min_adaptation_interval: Duration::from_millis(250),
            pid_params: Some(PIDParameters::bandwidth_optimized()),
        }
    }
}

/// PID controller parameters for adaptive quality
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PIDParameters {
    /// Proportional gain
    pub kp: f64,
    /// Integral gain
    pub ki: f64,
    /// Derivative gain
    pub kd: f64,
    /// Integral windup limit
    pub integral_limit: f64,
}

impl Default for PIDParameters {
    fn default() -> Self {
        Self {
            kp: 1.0,
            ki: 0.1,
            kd: 0.05,
            integral_limit: 10.0,
        }
    }
}

impl PIDParameters {
    /// Create responsive PID parameters
    #[must_use]
    pub fn responsive() -> Self {
        Self {
            kp: 2.0,
            ki: 0.5,
            kd: 0.1,
            integral_limit: 5.0,
        }
    }

    /// Create conservative PID parameters
    #[must_use]
    pub fn conservative() -> Self {
        Self {
            kp: 0.5,
            ki: 0.02,
            kd: 0.01,
            integral_limit: 20.0,
        }
    }

    /// Create bandwidth-optimized PID parameters
    #[must_use]
    pub fn bandwidth_optimized() -> Self {
        Self {
            kp: 1.5,
            ki: 0.3,
            kd: 0.08,
            integral_limit: 8.0,
        }
    }
}

/// Quality constraints for adaptive processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityConstraints {
    /// Minimum acceptable quality score
    pub min_quality_score: f64,
    /// Maximum acceptable quality degradation per adaptation
    pub max_quality_degradation: f64,
    /// Minimum acceptable frame rate
    pub min_frame_rate: f64,
    /// Maximum acceptable latency
    pub max_latency: Duration,
    /// Maximum resource usage constraints
    pub max_resource_usage: ResourceConstraints,
}

impl Default for QualityConstraints {
    fn default() -> Self {
        Self {
            min_quality_score: 0.3,
            max_quality_degradation: 0.2,
            min_frame_rate: 10.0,
            max_latency: Duration::from_millis(500),
            max_resource_usage: ResourceConstraints::default(),
        }
    }
}

impl QualityConstraints {
    /// Create flexible quality constraints
    #[must_use]
    pub fn flexible() -> Self {
        Self {
            min_quality_score: 0.2,
            max_quality_degradation: 0.3,
            min_frame_rate: 5.0,
            max_latency: Duration::from_secs(1),
            max_resource_usage: ResourceConstraints::relaxed(),
        }
    }

    /// Create strict quality constraints
    #[must_use]
    pub fn strict() -> Self {
        Self {
            min_quality_score: 0.7,
            max_quality_degradation: 0.1,
            min_frame_rate: 20.0,
            max_latency: Duration::from_millis(100),
            max_resource_usage: ResourceConstraints::strict(),
        }
    }

    /// Create resource-limited quality constraints
    #[must_use]
    pub fn resource_limited() -> Self {
        Self {
            min_quality_score: 0.2,
            max_quality_degradation: 0.4,
            min_frame_rate: 5.0,
            max_latency: Duration::from_secs(2),
            max_resource_usage: ResourceConstraints::limited(),
        }
    }
}

/// Resource constraints for adaptive processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceConstraints {
    /// Maximum CPU usage percentage
    pub max_cpu_usage: f64,
    /// Maximum memory usage in bytes
    pub max_memory_usage: usize,
    /// Maximum GPU memory usage in bytes
    pub max_gpu_memory_usage: Option<usize>,
    /// Maximum network bandwidth in bytes/second
    pub max_bandwidth: usize,
}

impl Default for ResourceConstraints {
    fn default() -> Self {
        Self {
            max_cpu_usage: 80.0,
            max_memory_usage: 512 * 1024 * 1024, // 512MB
            max_gpu_memory_usage: Some(1024 * 1024 * 1024), // 1GB
            max_bandwidth: 10 * 1024 * 1024,     // 10MB/s
        }
    }
}

impl ResourceConstraints {
    /// Create relaxed resource constraints
    #[must_use]
    pub fn relaxed() -> Self {
        Self {
            max_cpu_usage: 95.0,
            max_memory_usage: 2 * 1024 * 1024 * 1024, // 2GB
            max_gpu_memory_usage: Some(4 * 1024 * 1024 * 1024), // 4GB
            max_bandwidth: 100 * 1024 * 1024,         // 100MB/s
        }
    }

    /// Create strict resource constraints
    #[must_use]
    pub fn strict() -> Self {
        Self {
            max_cpu_usage: 60.0,
            max_memory_usage: 256 * 1024 * 1024, // 256MB
            max_gpu_memory_usage: Some(512 * 1024 * 1024), // 512MB
            max_bandwidth: 5 * 1024 * 1024,      // 5MB/s
        }
    }

    /// Create limited resource constraints
    #[must_use]
    pub fn limited() -> Self {
        Self {
            max_cpu_usage: 40.0,
            max_memory_usage: 128 * 1024 * 1024, // 128MB
            max_gpu_memory_usage: Some(256 * 1024 * 1024), // 256MB
            max_bandwidth: 2 * 1024 * 1024,      // 2MB/s
        }
    }
}

/// Streaming configuration for real-time video processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingConfig {
    /// Enable streaming
    pub enabled: bool,
    /// Streaming protocol
    pub protocol: StreamingProtocol,
    /// Video encoding configuration
    pub encoding: EncodingConfig,
    /// Network optimization settings
    pub network_optimization: NetworkOptimizationConfig,
    /// Error resilience configuration
    pub error_resilience: ErrorResilienceConfig,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            protocol: StreamingProtocol::WebRTC,
            encoding: EncodingConfig::default(),
            network_optimization: NetworkOptimizationConfig::default(),
            error_resilience: ErrorResilienceConfig::default(),
        }
    }
}

impl StreamingConfig {
    /// Create low-latency streaming configuration
    #[must_use]
    pub fn low_latency() -> Self {
        Self {
            enabled: true,
            protocol: StreamingProtocol::WebRTC,
            encoding: EncodingConfig::low_latency(),
            network_optimization: NetworkOptimizationConfig::latency_optimized(),
            error_resilience: ErrorResilienceConfig::fast_recovery(),
        }
    }

    /// Create high-quality streaming configuration
    #[must_use]
    pub fn high_quality() -> Self {
        Self {
            enabled: true,
            protocol: StreamingProtocol::DASH,
            encoding: EncodingConfig::high_quality(),
            network_optimization: NetworkOptimizationConfig::quality_optimized(),
            error_resilience: ErrorResilienceConfig::robust(),
        }
    }

    /// Create mobile-friendly streaming configuration
    #[must_use]
    pub fn mobile_friendly() -> Self {
        Self {
            enabled: true,
            protocol: StreamingProtocol::HLS,
            encoding: EncodingConfig::mobile_optimized(),
            network_optimization: NetworkOptimizationConfig::bandwidth_efficient(),
            error_resilience: ErrorResilienceConfig::adaptive(),
        }
    }

    /// Validate streaming configuration
    pub fn validate(&self) -> Result<(), RealTimeError> {
        if self.enabled {
            self.encoding.validate()?;
            self.network_optimization.validate()?;
        }
        Ok(())
    }
}

/// Video encoding configuration for streaming
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncodingConfig {
    /// Video codec
    pub codec: VideoCodec,
    /// Target bitrate in bits per second
    pub bitrate: u64,
    /// Key frame interval (in frames)
    pub keyframe_interval: usize,
    /// Rate control method
    pub rate_control: RateControlMethod,
    /// Quality parameter (codec-specific)
    pub quality_parameter: f64,
    /// Encoding preset (speed vs quality trade-off)
    pub preset: EncodingPreset,
    /// Custom encoding parameters
    pub custom_params: HashMap<String, TransformParameter>,
}

impl Default for EncodingConfig {
    fn default() -> Self {
        Self {
            codec: VideoCodec::H264,
            bitrate: 2_000_000, // 2 Mbps
            keyframe_interval: 30,
            rate_control: RateControlMethod::VBR,
            quality_parameter: 23.0, // CRF for H.264
            preset: EncodingPreset::Medium,
            custom_params: HashMap::new(),
        }
    }
}

impl EncodingConfig {
    /// Create low-latency encoding configuration
    #[must_use]
    pub fn low_latency() -> Self {
        Self {
            codec: VideoCodec::H264,
            bitrate: 1_000_000, // 1 Mbps
            keyframe_interval: 15,
            rate_control: RateControlMethod::CBR,
            quality_parameter: 28.0,
            preset: EncodingPreset::UltraFast,
            custom_params: HashMap::new(),
        }
    }

    /// Create high-quality encoding configuration
    #[must_use]
    pub fn high_quality() -> Self {
        Self {
            codec: VideoCodec::H265,
            bitrate: 8_000_000, // 8 Mbps
            keyframe_interval: 60,
            rate_control: RateControlMethod::CRF,
            quality_parameter: 18.0,
            preset: EncodingPreset::Slow,
            custom_params: HashMap::new(),
        }
    }

    /// Create mobile-optimized encoding configuration
    #[must_use]
    pub fn mobile_optimized() -> Self {
        Self {
            codec: VideoCodec::H264,
            bitrate: 500_000, // 500 Kbps
            keyframe_interval: 20,
            rate_control: RateControlMethod::VBR,
            quality_parameter: 30.0,
            preset: EncodingPreset::Fast,
            custom_params: HashMap::new(),
        }
    }

    /// Validate encoding configuration
    pub fn validate(&self) -> Result<(), RealTimeError> {
        if self.bitrate == 0 {
            return Err(RealTimeError::InvalidConfiguration(
                "Bitrate must be greater than zero".to_string(),
            ));
        }

        if self.keyframe_interval == 0 {
            return Err(RealTimeError::InvalidConfiguration(
                "Keyframe interval must be greater than zero".to_string(),
            ));
        }

        Ok(())
    }
}

/// Encoding presets for speed vs quality trade-off
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EncodingPreset {
    /// Fastest encoding, lowest quality
    UltraFast,
    /// Very fast encoding
    VeryFast,
    /// Fast encoding
    Fast,
    /// Balanced speed and quality
    Medium,
    /// Slower encoding, better quality
    Slow,
    /// Very slow encoding, high quality
    VerySlow,
    /// Slowest encoding, highest quality
    Placebo,
}

impl Default for EncodingPreset {
    fn default() -> Self {
        Self::Medium
    }
}

/// Network optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkOptimizationConfig {
    /// Enable adaptive bitrate streaming
    pub adaptive_bitrate: bool,
    /// Enable network condition monitoring
    pub monitor_network: bool,
    /// Enable bandwidth prediction
    pub bandwidth_prediction: bool,
    /// Enable forward error correction
    pub fec_enabled: bool,
    /// Network buffer size in milliseconds
    pub network_buffer_size: Duration,
    /// Congestion control algorithm
    pub congestion_control: CongestionControlAlgorithm,
}

impl Default for NetworkOptimizationConfig {
    fn default() -> Self {
        Self {
            adaptive_bitrate: true,
            monitor_network: true,
            bandwidth_prediction: false,
            fec_enabled: false,
            network_buffer_size: Duration::from_millis(1000),
            congestion_control: CongestionControlAlgorithm::BBR,
        }
    }
}

impl NetworkOptimizationConfig {
    /// Create latency-optimized network configuration
    #[must_use]
    pub fn latency_optimized() -> Self {
        Self {
            adaptive_bitrate: true,
            monitor_network: true,
            bandwidth_prediction: true,
            fec_enabled: false,
            network_buffer_size: Duration::from_millis(100),
            congestion_control: CongestionControlAlgorithm::BBR,
        }
    }

    /// Create quality-optimized network configuration
    #[must_use]
    pub fn quality_optimized() -> Self {
        Self {
            adaptive_bitrate: true,
            monitor_network: true,
            bandwidth_prediction: true,
            fec_enabled: true,
            network_buffer_size: Duration::from_millis(3000),
            congestion_control: CongestionControlAlgorithm::Cubic,
        }
    }

    /// Create bandwidth-efficient network configuration
    #[must_use]
    pub fn bandwidth_efficient() -> Self {
        Self {
            adaptive_bitrate: true,
            monitor_network: true,
            bandwidth_prediction: true,
            fec_enabled: false,
            network_buffer_size: Duration::from_millis(2000),
            congestion_control: CongestionControlAlgorithm::BBR,
        }
    }

    /// Validate network optimization configuration
    pub fn validate(&self) -> Result<(), RealTimeError> {
        if self.network_buffer_size.is_zero() {
            return Err(RealTimeError::InvalidConfiguration(
                "Network buffer size must be greater than zero".to_string(),
            ));
        }
        Ok(())
    }
}

/// Congestion control algorithms for network optimization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CongestionControlAlgorithm {
    /// Bottleneck Bandwidth and Round-trip propagation time
    BBR,
    /// CUBIC TCP
    Cubic,
    /// TCP Reno
    Reno,
    /// TCP `NewReno`
    NewReno,
    /// Custom algorithm
    Custom,
}

impl Default for CongestionControlAlgorithm {
    fn default() -> Self {
        Self::BBR
    }
}

/// Error resilience configuration for streaming
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorResilienceConfig {
    /// Enable automatic retry on failures
    pub auto_retry: bool,
    /// Maximum retry attempts
    pub max_retries: usize,
    /// Retry delay
    pub retry_delay: Duration,
    /// Enable graceful degradation
    pub graceful_degradation: bool,
    /// Recovery strategy
    pub recovery_strategy: RecoveryStrategy,
    /// Error detection sensitivity
    pub error_detection_sensitivity: f64,
}

impl Default for ErrorResilienceConfig {
    fn default() -> Self {
        Self {
            auto_retry: true,
            max_retries: 3,
            retry_delay: Duration::from_millis(500),
            graceful_degradation: true,
            recovery_strategy: RecoveryStrategy::Degrade,
            error_detection_sensitivity: 0.5,
        }
    }
}

impl ErrorResilienceConfig {
    /// Create fast recovery error resilience configuration
    #[must_use]
    pub fn fast_recovery() -> Self {
        Self {
            auto_retry: true,
            max_retries: 2,
            retry_delay: Duration::from_millis(100),
            graceful_degradation: true,
            recovery_strategy: RecoveryStrategy::Skip,
            error_detection_sensitivity: 0.8,
        }
    }

    /// Create robust error resilience configuration
    #[must_use]
    pub fn robust() -> Self {
        Self {
            auto_retry: true,
            max_retries: 5,
            retry_delay: Duration::from_millis(1000),
            graceful_degradation: true,
            recovery_strategy: RecoveryStrategy::Retry,
            error_detection_sensitivity: 0.3,
        }
    }

    /// Create adaptive error resilience configuration
    #[must_use]
    pub fn adaptive() -> Self {
        Self {
            auto_retry: true,
            max_retries: 3,
            retry_delay: Duration::from_millis(250),
            graceful_degradation: true,
            recovery_strategy: RecoveryStrategy::Fallback,
            error_detection_sensitivity: 0.6,
        }
    }
}

/// Performance monitoring configuration for real-time processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMonitoringConfig {
    /// Enable performance monitoring
    pub enabled: bool,
    /// Monitoring interval
    pub monitoring_interval: Duration,
    /// Metrics to track
    pub tracked_metrics: Vec<PerformanceMetric>,
    /// Performance alert thresholds
    pub alert_thresholds: PerformanceThresholds,
    /// History retention period
    pub history_retention: Duration,
}

impl Default for PerformanceMonitoringConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            monitoring_interval: Duration::from_secs(1),
            tracked_metrics: vec![
                PerformanceMetric::FrameRate,
                PerformanceMetric::Latency,
                PerformanceMetric::CPUUsage,
                PerformanceMetric::MemoryUsage,
            ],
            alert_thresholds: PerformanceThresholds::default(),
            history_retention: Duration::from_secs(3600), // 1 hour
        }
    }
}

impl PerformanceMonitoringConfig {
    /// Create aggressive performance monitoring configuration
    #[must_use]
    pub fn aggressive() -> Self {
        Self {
            enabled: true,
            monitoring_interval: Duration::from_millis(100),
            tracked_metrics: vec![
                PerformanceMetric::FrameRate,
                PerformanceMetric::Latency,
                PerformanceMetric::CPUUsage,
                PerformanceMetric::MemoryUsage,
                PerformanceMetric::GPUUsage,
                PerformanceMetric::NetworkBandwidth,
                PerformanceMetric::BufferUtilization,
            ],
            alert_thresholds: PerformanceThresholds::strict(),
            history_retention: Duration::from_secs(7200), // 2 hours
        }
    }

    /// Create comprehensive performance monitoring configuration
    #[must_use]
    pub fn comprehensive() -> Self {
        Self {
            enabled: true,
            monitoring_interval: Duration::from_millis(500),
            tracked_metrics: vec![
                PerformanceMetric::FrameRate,
                PerformanceMetric::Latency,
                PerformanceMetric::CPUUsage,
                PerformanceMetric::MemoryUsage,
                PerformanceMetric::GPUUsage,
                PerformanceMetric::NetworkBandwidth,
                PerformanceMetric::BufferUtilization,
                PerformanceMetric::QualityScore,
                PerformanceMetric::ErrorRate,
            ],
            alert_thresholds: PerformanceThresholds::balanced(),
            history_retention: Duration::from_secs(86400), // 24 hours
        }
    }

    /// Create lightweight performance monitoring configuration
    #[must_use]
    pub fn lightweight() -> Self {
        Self {
            enabled: true,
            monitoring_interval: Duration::from_secs(5),
            tracked_metrics: vec![PerformanceMetric::FrameRate, PerformanceMetric::Latency],
            alert_thresholds: PerformanceThresholds::relaxed(),
            history_retention: Duration::from_secs(1800), // 30 minutes
        }
    }
}

/// Performance metrics for monitoring
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PerformanceMetric {
    /// Frame rate (fps)
    FrameRate,
    /// Processing latency
    Latency,
    /// CPU utilization percentage
    CPUUsage,
    /// Memory utilization
    MemoryUsage,
    /// GPU utilization percentage
    GPUUsage,
    /// Network bandwidth usage
    NetworkBandwidth,
    /// Buffer utilization percentage
    BufferUtilization,
    /// Quality score
    QualityScore,
    /// Error rate
    ErrorRate,
}

/// Performance alert thresholds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceThresholds {
    /// Minimum acceptable frame rate
    pub min_frame_rate: f64,
    /// Maximum acceptable latency
    pub max_latency: Duration,
    /// Maximum CPU usage percentage
    pub max_cpu_usage: f64,
    /// Maximum memory usage percentage
    pub max_memory_usage: f64,
    /// Maximum GPU usage percentage
    pub max_gpu_usage: Option<f64>,
    /// Maximum error rate
    pub max_error_rate: f64,
}

impl Default for PerformanceThresholds {
    fn default() -> Self {
        Self {
            min_frame_rate: 15.0,
            max_latency: Duration::from_millis(200),
            max_cpu_usage: 80.0,
            max_memory_usage: 80.0,
            max_gpu_usage: Some(80.0),
            max_error_rate: 0.05, // 5%
        }
    }
}

impl PerformanceThresholds {
    /// Create strict performance thresholds
    #[must_use]
    pub fn strict() -> Self {
        Self {
            min_frame_rate: 25.0,
            max_latency: Duration::from_millis(50),
            max_cpu_usage: 60.0,
            max_memory_usage: 60.0,
            max_gpu_usage: Some(60.0),
            max_error_rate: 0.01, // 1%
        }
    }

    /// Create balanced performance thresholds
    #[must_use]
    pub fn balanced() -> Self {
        Self::default()
    }

    /// Create relaxed performance thresholds
    #[must_use]
    pub fn relaxed() -> Self {
        Self {
            min_frame_rate: 10.0,
            max_latency: Duration::from_millis(500),
            max_cpu_usage: 95.0,
            max_memory_usage: 90.0,
            max_gpu_usage: Some(90.0),
            max_error_rate: 0.1, // 10%
        }
    }
}

/// Real-time error handling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealTimeErrorHandling {
    /// Recovery strategy for real-time failures
    pub recovery_strategy: RecoveryStrategy,
    /// Maximum recovery time
    pub max_recovery_time: Duration,
    /// Enable fallback processing
    pub enable_fallback: bool,
    /// Fallback quality level
    pub fallback_quality: ProcessingComplexity,
    /// Circuit breaker configuration
    pub circuit_breaker: CircuitBreakerConfig,
}

impl Default for RealTimeErrorHandling {
    fn default() -> Self {
        Self {
            recovery_strategy: RecoveryStrategy::Degrade,
            max_recovery_time: Duration::from_millis(100),
            enable_fallback: true,
            fallback_quality: ProcessingComplexity::Low,
            circuit_breaker: CircuitBreakerConfig::default(),
        }
    }
}

impl RealTimeErrorHandling {
    /// Create fast recovery error handling configuration
    #[must_use]
    pub fn fast_recovery() -> Self {
        Self {
            recovery_strategy: RecoveryStrategy::Skip,
            max_recovery_time: Duration::from_millis(10),
            enable_fallback: true,
            fallback_quality: ProcessingComplexity::Low,
            circuit_breaker: CircuitBreakerConfig::sensitive(),
        }
    }

    /// Create robust error handling configuration
    #[must_use]
    pub fn robust() -> Self {
        Self {
            recovery_strategy: RecoveryStrategy::Retry,
            max_recovery_time: Duration::from_millis(500),
            enable_fallback: true,
            fallback_quality: ProcessingComplexity::Medium,
            circuit_breaker: CircuitBreakerConfig::stable(),
        }
    }

    /// Create graceful degradation error handling configuration
    #[must_use]
    pub fn graceful_degradation() -> Self {
        Self {
            recovery_strategy: RecoveryStrategy::Degrade,
            max_recovery_time: Duration::from_millis(200),
            enable_fallback: true,
            fallback_quality: ProcessingComplexity::Low,
            circuit_breaker: CircuitBreakerConfig::adaptive(),
        }
    }
}

/// Circuit breaker configuration for error handling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitBreakerConfig {
    /// Enable circuit breaker
    pub enabled: bool,
    /// Failure threshold to open circuit
    pub failure_threshold: usize,
    /// Success threshold to close circuit
    pub success_threshold: usize,
    /// Timeout for half-open state
    pub timeout: Duration,
    /// Recovery window
    pub recovery_window: Duration,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            failure_threshold: 5,
            success_threshold: 3,
            timeout: Duration::from_secs(30),
            recovery_window: Duration::from_secs(60),
        }
    }
}

impl CircuitBreakerConfig {
    /// Create sensitive circuit breaker configuration
    #[must_use]
    pub fn sensitive() -> Self {
        Self {
            enabled: true,
            failure_threshold: 2,
            success_threshold: 5,
            timeout: Duration::from_secs(10),
            recovery_window: Duration::from_secs(30),
        }
    }

    /// Create stable circuit breaker configuration
    #[must_use]
    pub fn stable() -> Self {
        Self {
            enabled: true,
            failure_threshold: 10,
            success_threshold: 2,
            timeout: Duration::from_secs(60),
            recovery_window: Duration::from_secs(120),
        }
    }

    /// Create adaptive circuit breaker configuration
    #[must_use]
    pub fn adaptive() -> Self {
        Self {
            enabled: true,
            failure_threshold: 5,
            success_threshold: 3,
            timeout: Duration::from_secs(30),
            recovery_window: Duration::from_secs(90),
        }
    }
}

/// Real-time processing errors
#[derive(Debug, Clone, PartialEq)]
pub enum RealTimeError {
    /// Invalid configuration
    InvalidConfiguration(String),
    /// Buffer overflow
    BufferOverflow,
    /// Buffer underrun
    BufferUnderrun,
    /// Latency exceeded
    LatencyExceeded { actual: Duration, limit: Duration },
    /// Frame rate too low
    FrameRateTooLow { actual: f64, minimum: f64 },
    /// Quality degradation
    QualityDegradation { actual: f64, minimum: f64 },
    /// Resource limit exceeded
    ResourceLimitExceeded(String),
    /// Network error
    NetworkError(String),
    /// Encoding error
    EncodingError(String),
    /// Streaming error
    StreamingError(String),
}

impl std::fmt::Display for RealTimeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidConfiguration(msg) => write!(f, "Invalid configuration: {msg}"),
            Self::BufferOverflow => write!(f, "Buffer overflow occurred"),
            Self::BufferUnderrun => write!(f, "Buffer underrun occurred"),
            Self::LatencyExceeded { actual, limit } => {
                write!(f, "Latency exceeded: {actual:?} > {limit:?}")
            }
            Self::FrameRateTooLow { actual, minimum } => {
                write!(f, "Frame rate too low: {actual} < {minimum}")
            }
            Self::QualityDegradation { actual, minimum } => {
                write!(f, "Quality degradation: {actual} < {minimum}")
            }
            Self::ResourceLimitExceeded(resource) => {
                write!(f, "Resource limit exceeded: {resource}")
            }
            Self::NetworkError(msg) => write!(f, "Network error: {msg}"),
            Self::EncodingError(msg) => write!(f, "Encoding error: {msg}"),
            Self::StreamingError(msg) => write!(f, "Streaming error: {msg}"),
        }
    }
}

impl std::error::Error for RealTimeError {}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_realtime_config_presets() {
        let low_latency = RealTimeProcessingConfig::low_latency();
        assert!(low_latency.enabled);
        assert_eq!(low_latency.target_fps, 60.0);
        assert_eq!(low_latency.max_latency, Duration::from_millis(16));

        let high_quality = RealTimeProcessingConfig::high_quality_streaming();
        assert_eq!(high_quality.target_fps, 30.0);
        assert_eq!(high_quality.max_latency, Duration::from_millis(200));

        let mobile = RealTimeProcessingConfig::mobile_optimized();
        assert_eq!(mobile.target_fps, 15.0);
        assert_eq!(mobile.max_latency, Duration::from_millis(300));
    }

    #[test]
    fn test_buffer_management_config() {
        let minimal = BufferManagementConfig::minimal();
        assert_eq!(minimal.input_buffer_size, 1);
        assert_eq!(
            minimal.overflow_strategy,
            BufferOverflowStrategy::DropNewest
        );

        let quality_focused = BufferManagementConfig::quality_focused();
        assert_eq!(quality_focused.input_buffer_size, 10);
        assert!(quality_focused.dynamic_resizing);

        let memory_efficient = BufferManagementConfig::memory_efficient();
        assert_eq!(memory_efficient.input_buffer_size, 2);
        assert!(!memory_efficient.pre_allocation);
    }

    #[test]
    fn test_adaptive_quality_config() {
        let responsive = AdaptiveQualityConfig::responsive();
        assert_eq!(responsive.adaptation_algorithm, AdaptationAlgorithm::PID);
        assert_eq!(responsive.quality_levels.len(), 2);

        let quality_preserving = AdaptiveQualityConfig::quality_preserving();
        assert_eq!(
            quality_preserving.adaptation_algorithm,
            AdaptationAlgorithm::Predictive
        );
        assert_eq!(quality_preserving.quality_levels.len(), 3);

        let bandwidth_aware = AdaptiveQualityConfig::bandwidth_aware();
        assert_eq!(
            bandwidth_aware.adaptation_algorithm,
            AdaptationAlgorithm::Fuzzy
        );
    }

    #[test]
    fn test_quality_levels() {
        let low = QualityLevel::low();
        assert_eq!(low.complexity, ProcessingComplexity::Low);
        assert_eq!(low.expected_fps, 60.0);

        let high = QualityLevel::high();
        assert_eq!(high.complexity, ProcessingComplexity::High);
        assert_eq!(high.expected_fps, 15.0);

        assert!(low.validate().is_ok());
        assert!(high.validate().is_ok());
    }

    #[test]
    fn test_streaming_config() {
        let low_latency = StreamingConfig::low_latency();
        assert_eq!(low_latency.protocol, StreamingProtocol::WebRTC);
        assert_eq!(low_latency.encoding.rate_control, RateControlMethod::CBR);

        let high_quality = StreamingConfig::high_quality();
        assert_eq!(high_quality.protocol, StreamingProtocol::DASH);
        assert_eq!(high_quality.encoding.codec, VideoCodec::H265);

        let mobile = StreamingConfig::mobile_friendly();
        assert_eq!(mobile.protocol, StreamingProtocol::HLS);
        assert_eq!(mobile.encoding.bitrate, 500_000);
    }

    #[test]
    fn test_encoding_config() {
        let low_latency = EncodingConfig::low_latency();
        assert_eq!(low_latency.preset, EncodingPreset::UltraFast);
        assert_eq!(low_latency.keyframe_interval, 15);

        let high_quality = EncodingConfig::high_quality();
        assert_eq!(high_quality.preset, EncodingPreset::Slow);
        assert_eq!(high_quality.codec, VideoCodec::H265);

        assert!(low_latency.validate().is_ok());
        assert!(high_quality.validate().is_ok());
    }

    #[test]
    fn test_network_optimization_config() {
        let latency_optimized = NetworkOptimizationConfig::latency_optimized();
        assert_eq!(
            latency_optimized.network_buffer_size,
            Duration::from_millis(100)
        );
        assert!(latency_optimized.bandwidth_prediction);

        let quality_optimized = NetworkOptimizationConfig::quality_optimized();
        assert!(quality_optimized.fec_enabled);
        assert_eq!(
            quality_optimized.network_buffer_size,
            Duration::from_millis(3000)
        );

        assert!(latency_optimized.validate().is_ok());
        assert!(quality_optimized.validate().is_ok());
    }

    #[test]
    fn test_error_handling() {
        let fast_recovery = RealTimeErrorHandling::fast_recovery();
        assert_eq!(fast_recovery.recovery_strategy, RecoveryStrategy::Skip);
        assert_eq!(fast_recovery.max_recovery_time, Duration::from_millis(10));

        let robust = RealTimeErrorHandling::robust();
        assert_eq!(robust.recovery_strategy, RecoveryStrategy::Retry);
        assert_eq!(robust.max_recovery_time, Duration::from_millis(500));

        let graceful = RealTimeErrorHandling::graceful_degradation();
        assert_eq!(graceful.recovery_strategy, RecoveryStrategy::Degrade);
    }

    #[test]
    fn test_realtime_error_display() {
        let error = RealTimeError::LatencyExceeded {
            actual: Duration::from_millis(200),
            limit: Duration::from_millis(100),
        };
        let error_str = error.to_string();
        assert!(error_str.contains("Latency exceeded"));
        assert!(error_str.contains("200ms"));
        assert!(error_str.contains("100ms"));

        let error = RealTimeError::FrameRateTooLow {
            actual: 10.0,
            minimum: 20.0,
        };
        let error_str = error.to_string();
        assert!(error_str.contains("Frame rate too low"));
        assert!(error_str.contains("10"));
        assert!(error_str.contains("20"));
    }

    #[test]
    fn test_config_validation() {
        let mut config = RealTimeProcessingConfig::default();
        config.enabled = true;
        assert!(config.validate().is_ok());

        config.target_fps = 0.0;
        assert!(config.validate().is_err());

        config.target_fps = 30.0;
        config.max_latency = Duration::from_secs(0);
        assert!(config.validate().is_err());
    }
}
