//! Shared types and configuration structures for computer vision pipelines
//!
//! This module provides common types, enums, and configuration structures
//! used throughout the computer vision pipeline system including data types,
//! processing modes, and basic configuration options.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Image data types for computer vision processing
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ImageDataType {
    /// 8-bit unsigned integer (0-255)
    UInt8,
    /// 16-bit unsigned integer (0-65535)
    UInt16,
    /// 32-bit floating point
    Float32,
    /// 64-bit floating point
    Float64,
}

impl Default for ImageDataType {
    fn default() -> Self {
        Self::UInt8
    }
}

/// Color spaces for image processing
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ColorSpace {
    /// Red, Green, Blue
    RGB,
    /// Blue, Green, Red
    BGR,
    /// Hue, Saturation, Value
    HSV,
    /// L*a*b* color space
    LAB,
    /// YUV color space
    YUV,
    /// Grayscale (single channel)
    Grayscale,
    /// Cyan, Magenta, Yellow, Key (Black)
    CMYK,
    /// CIE XYZ color space
    XYZ,
}

impl Default for ColorSpace {
    fn default() -> Self {
        Self::RGB
    }
}

/// Supported image formats for input/output
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ImageFormat {
    /// JPEG format
    JPEG,
    /// PNG format
    PNG,
    /// BMP format
    BMP,
    /// TIFF format
    TIFF,
    /// WebP format
    WebP,
    /// Raw sensor data
    RAW,
    /// High Dynamic Range
    HDR,
    /// `OpenEXR` format
    EXR,
    /// Custom format with MIME type
    Custom(String),
}

impl Default for ImageFormat {
    fn default() -> Self {
        Self::PNG
    }
}

/// Processing modes for computer vision pipelines
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ProcessingMode {
    /// Batch processing for multiple images
    Batch,
    /// Real-time processing with low latency
    RealTime,
    /// Streaming processing for continuous input
    Streaming,
    /// Interactive processing with user feedback
    Interactive,
    /// Offline processing for maximum quality
    Offline,
    /// On-demand processing
    OnDemand,
}

impl Default for ProcessingMode {
    fn default() -> Self {
        Self::Batch
    }
}

/// Processing complexity levels for adaptive processing
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum ProcessingComplexity {
    /// Minimal processing for maximum speed
    Low,
    /// Balanced processing for good speed/quality trade-off
    Medium,
    /// High-quality processing with moderate speed
    High,
    /// Maximum quality processing regardless of speed
    Ultra,
}

impl Default for ProcessingComplexity {
    fn default() -> Self {
        Self::Medium
    }
}

/// Memory optimization levels for resource management
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MemoryOptimizationLevel {
    /// Minimal optimization, prioritize speed
    None,
    /// Basic memory management
    Low,
    /// Balanced memory optimization
    Medium,
    /// Aggressive memory optimization
    High,
    /// Maximum memory efficiency
    Extreme,
}

impl Default for MemoryOptimizationLevel {
    fn default() -> Self {
        Self::Medium
    }
}

/// Cache eviction policies for memory management
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CacheEvictionPolicy {
    /// Least Recently Used
    LRU,
    /// Least Frequently Used
    LFU,
    /// First In, First Out
    FIFO,
    /// Least Recently Used with size consideration
    LRUSize,
    /// Random eviction
    Random,
    /// Time-based expiration
    TTL,
}

impl Default for CacheEvictionPolicy {
    fn default() -> Self {
        Self::LRU
    }
}

/// Parallel processing strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ParallelStrategy {
    /// No parallelization
    None,
    /// Thread-based parallelization
    Threading,
    /// Process-based parallelization
    Multiprocessing,
    /// GPU-based parallelization
    GPU,
    /// Hybrid CPU/GPU processing
    Hybrid,
    /// SIMD vectorization
    SIMD,
    /// Distributed processing
    Distributed,
}

impl Default for ParallelStrategy {
    fn default() -> Self {
        Self::Threading
    }
}

/// Load balancing algorithms for parallel processing
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LoadBalancingAlgorithm {
    /// Round-robin scheduling
    RoundRobin,
    /// Weighted round-robin
    WeightedRoundRobin,
    /// Least connections
    LeastConnections,
    /// Least processing time
    LeastTime,
    /// Work stealing
    WorkStealing,
    /// Random assignment
    Random,
    /// Priority-based scheduling
    Priority,
}

impl Default for LoadBalancingAlgorithm {
    fn default() -> Self {
        Self::WorkStealing
    }
}

/// Buffer overflow strategies for real-time processing
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BufferOverflowStrategy {
    /// Drop oldest frames when buffer is full
    DropOldest,
    /// Drop newest frames when buffer is full
    DropNewest,
    /// Block until buffer space is available
    Block,
    /// Reduce processing quality to maintain throughput
    ReduceQuality,
    /// Skip frames intelligently
    SkipFrames,
    /// Increase buffer size dynamically
    DynamicResize,
}

impl Default for BufferOverflowStrategy {
    fn default() -> Self {
        Self::DropOldest
    }
}

/// Modalities for multi-modal processing
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Modality {
    /// Visual/image data
    Visual,
    /// Audio data
    Audio,
    /// Text data
    Text,
    /// Depth sensor data
    Depth,
    /// Infrared sensor data
    Infrared,
    /// `LiDAR` point cloud data
    LiDAR,
    /// Radar data
    Radar,
    /// Thermal imaging
    Thermal,
    /// Hyperspectral imaging
    Hyperspectral,
    /// Time-series sensor data
    TimeSeries,
}

/// Fusion strategies for multi-modal data
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FusionStrategy {
    /// Early fusion at feature level
    EarlyFusion,
    /// Late fusion at decision level
    LateFusion,
    /// Hybrid fusion combining early and late
    HybridFusion,
    /// Attention-based fusion
    AttentionFusion,
    /// Cross-modal transformer fusion
    TransformerFusion,
    /// Graph-based fusion
    GraphFusion,
}

impl Default for FusionStrategy {
    fn default() -> Self {
        Self::LateFusion
    }
}

/// Synchronization methods for temporal alignment
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SyncMethod {
    /// Hardware synchronization
    Hardware,
    /// Software timestamps
    Software,
    /// Network Time Protocol
    NTP,
    /// GPS synchronization
    GPS,
    /// Cross-correlation alignment
    CrossCorrelation,
    /// Manual alignment
    Manual,
}

impl Default for SyncMethod {
    fn default() -> Self {
        Self::Software
    }
}

/// Interpolation methods for data alignment
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum InterpolationMethod {
    /// Nearest neighbor interpolation
    Nearest,
    /// Linear interpolation
    Linear,
    /// Cubic interpolation
    Cubic,
    /// Spline interpolation
    Spline,
    /// Polynomial interpolation
    Polynomial,
    /// Gaussian interpolation
    Gaussian,
}

impl Default for InterpolationMethod {
    fn default() -> Self {
        Self::Linear
    }
}

/// Cross-modal learning strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CrossModalStrategy {
    /// Shared representation learning
    SharedRepresentation,
    /// Canonical correlation analysis
    CCA,
    /// Multi-view learning
    MultiView,
    /// Domain adaptation
    DomainAdaptation,
    /// Contrastive learning
    Contrastive,
    /// Alignment learning
    Alignment,
    /// Knowledge distillation
    Distillation,
}

impl Default for CrossModalStrategy {
    fn default() -> Self {
        Self::SharedRepresentation
    }
}

/// Video streaming protocols
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StreamingProtocol {
    /// Real-Time Messaging Protocol
    RTMP,
    /// Web Real-Time Communication
    WebRTC,
    /// HTTP Live Streaming
    HLS,
    /// Dynamic Adaptive Streaming over HTTP
    DASH,
    /// User Datagram Protocol
    UDP,
    /// Transmission Control Protocol
    TCP,
    /// Quick UDP Internet Connections
    QUIC,
}

impl Default for StreamingProtocol {
    fn default() -> Self {
        Self::WebRTC
    }
}

/// Video codecs for compression
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum VideoCodec {
    /// H.264/AVC
    H264,
    /// H.265/HEVC
    H265,
    /// VP8
    VP8,
    /// VP9
    VP9,
    /// AV1
    AV1,
    /// Motion JPEG
    MJPEG,
    /// Apple `ProRes`
    ProRes,
}

impl Default for VideoCodec {
    fn default() -> Self {
        Self::H264
    }
}

/// Rate control methods for video encoding
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RateControlMethod {
    /// Constant Bitrate
    CBR,
    /// Variable Bitrate
    VBR,
    /// Constant Rate Factor
    CRF,
    /// Constant Quantization Parameter
    CQP,
}

impl Default for RateControlMethod {
    fn default() -> Self {
        Self::VBR
    }
}

/// Quality adaptation algorithms
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AdaptationAlgorithm {
    /// Simple threshold-based adaptation
    Threshold,
    /// PID controller-based adaptation
    PID,
    /// Fuzzy logic adaptation
    Fuzzy,
    /// Reinforcement learning adaptation
    Reinforcement,
    /// Predictive adaptation
    Predictive,
}

impl Default for AdaptationAlgorithm {
    fn default() -> Self {
        Self::Threshold
    }
}

/// Metrics for quality adaptation decisions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AdaptationMetric {
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
    /// Prediction accuracy
    Accuracy,
    /// Overall quality score
    QualityScore,
}

/// Model output types for different CV tasks
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OutputType {
    /// Classification with class probabilities
    Classification,
    /// Regression with continuous values
    Regression,
    /// Semantic or instance segmentation
    Segmentation,
    /// Object detection with bounding boxes
    Detection,
    /// Keypoint detection
    Keypoints,
    /// Feature embeddings
    Embedding,
    /// Raw tensor output
    Raw,
}

/// Types of post-processors for CV results
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ProcessorType {
    /// Non-maximum suppression for object detection
    NonMaximumSuppression,
    /// Noise filtering
    Filtering,
    /// Result smoothing
    Smoothing,
    /// Object tracking across frames
    Tracking,
    /// Model ensemble combination
    Ensemble,
    /// Quality enhancement
    QualityEnhancement,
    /// Custom processing
    Custom,
}

/// Transform parameter types for flexible configuration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TransformParameter {
    /// Boolean parameter
    Bool(bool),
    /// Integer parameter
    Int(i64),
    /// Floating point parameter
    Float(f64),
    /// String parameter
    String(String),
    /// Array of floats
    FloatArray(Vec<f64>),
    /// Array of integers
    IntArray(Vec<i64>),
    /// Nested parameter map
    Map(HashMap<String, TransformParameter>),
}

impl Default for TransformParameter {
    fn default() -> Self {
        Self::Float(0.0)
    }
}

/// Recovery strategies for error handling
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RecoveryStrategy {
    /// Fail immediately on error
    FailFast,
    /// Skip failed items and continue
    Skip,
    /// Retry with backoff
    Retry,
    /// Use fallback processing
    Fallback,
    /// Degrade quality and continue
    Degrade,
    /// Use cached results
    UseCache,
}

impl Default for RecoveryStrategy {
    fn default() -> Self {
        Self::Skip
    }
}

// ========== Missing Types for CV Pipelines ==========

/// Bounding box for object detection
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BoundingBox {
    /// X coordinate of top-left corner
    pub x: f32,
    /// Y coordinate of top-left corner
    pub y: f32,
    /// Width of the bounding box
    pub width: f32,
    /// Height of the bounding box
    pub height: f32,
    /// Confidence score
    pub confidence: f32,
}

/// Computer vision prediction result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CVPrediction {
    /// Prediction class
    pub class: String,
    /// Confidence score
    pub confidence: f32,
    /// Bounding box if applicable
    pub bbox: Option<BoundingBox>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Camera information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CameraInfo {
    /// Camera model
    pub model: String,
    /// Manufacturer
    pub manufacturer: String,
    /// Serial number
    pub serial_number: Option<String>,
    /// Firmware version
    pub firmware_version: Option<String>,
}

/// Camera intrinsic parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CameraIntrinsics {
    /// Focal length X
    pub fx: f64,
    /// Focal length Y
    pub fy: f64,
    /// Principal point X
    pub cx: f64,
    /// Principal point Y
    pub cy: f64,
    /// Distortion coefficients
    pub distortion: Vec<f64>,
}

/// Camera settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CameraSettings {
    /// ISO sensitivity
    pub iso: Option<u32>,
    /// Aperture f-number
    pub aperture: Option<f32>,
    /// Shutter speed in seconds
    pub shutter_speed: Option<f64>,
    /// White balance
    pub white_balance: Option<String>,
    /// Focus mode
    pub focus_mode: Option<String>,
}

/// Compute device specification
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ComputeDevice {
    /// CPU processing
    CPU,
    /// GPU processing
    GPU(u32), // GPU index
    /// TPU processing
    TPU,
    /// Custom accelerator
    Custom(String),
}

/// Confidence scores for predictions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceScores {
    /// Primary confidence score
    pub primary: f32,
    /// Secondary confidence scores
    pub secondary: Vec<f32>,
    /// Threshold used
    pub threshold: f32,
}

/// Object detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Detection {
    /// Object class
    pub class: String,
    /// Bounding box
    pub bbox: BoundingBox,
    /// Confidence score
    pub confidence: f32,
    /// Object ID for tracking
    pub object_id: Option<u32>,
}

/// Detection metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectionMetadata {
    /// Number of detections
    pub count: usize,
    /// Processing time in milliseconds
    pub processing_time_ms: f64,
    /// Model used for detection
    pub model_name: String,
    /// Model version
    pub model_version: String,
}

/// EXIF data from images
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExifData {
    /// Camera make
    pub make: Option<String>,
    /// Camera model
    pub model: Option<String>,
    /// Date and time
    pub datetime: Option<String>,
    /// GPS information
    pub gps_info: Option<GPSInfo>,
    /// Lens information
    pub lens_info: Option<LensInfo>,
    /// Camera settings
    pub camera_settings: Option<CameraSettings>,
}

/// Feature extractor configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractorConfig {
    /// Extractor type
    pub extractor_type: ExtractorType,
    /// Feature dimensions
    pub feature_dims: usize,
    /// Processing parameters
    pub parameters: HashMap<String, String>,
}

/// Feature extractor types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExtractorType {
    /// SIFT features
    SIFT,
    /// SURF features
    SURF,
    /// ORB features
    ORB,
    /// HOG features
    HOG,
    /// Deep learning features
    DeepLearning(String), // Model name
    /// Custom extractor
    Custom(String),
}

/// Feature metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureMetadata {
    /// Number of features extracted
    pub count: usize,
    /// Feature dimensionality
    pub dimensions: usize,
    /// Extractor used
    pub extractor: ExtractorType,
    /// Extraction time
    pub extraction_time_ms: f64,
}

/// Feature quality metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureQuality {
    /// Quality score (0.0 to 1.0)
    pub score: f32,
    /// Distinctiveness measure
    pub distinctiveness: f32,
    /// Repeatability measure
    pub repeatability: f32,
    /// Robustness to noise
    pub robustness: f32,
}

/// Feature statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureStatistics {
    /// Mean feature values
    pub mean: Vec<f32>,
    /// Standard deviation
    pub std_dev: Vec<f32>,
    /// Min values
    pub min: Vec<f32>,
    /// Max values
    pub max: Vec<f32>,
}

/// Feature vector representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureVector {
    /// Feature data
    pub data: Vec<f32>,
    /// Feature metadata
    pub metadata: FeatureMetadata,
    /// Quality metrics
    pub quality: Option<FeatureQuality>,
}

/// GPS information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GPSInfo {
    /// Latitude
    pub latitude: f64,
    /// Longitude
    pub longitude: f64,
    /// Altitude in meters
    pub altitude: Option<f64>,
    /// GPS timestamp
    pub timestamp: Option<String>,
}

/// Image metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageMetadata {
    /// Image width
    pub width: u32,
    /// Image height
    pub height: u32,
    /// Number of channels
    pub channels: u32,
    /// Bit depth
    pub bit_depth: u32,
    /// Color space
    pub color_space: ColorSpace,
    /// EXIF data
    pub exif: Option<ExifData>,
}

/// Lens information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LensInfo {
    /// Lens model
    pub model: String,
    /// Focal length
    pub focal_length: f32,
    /// Maximum aperture
    pub max_aperture: f32,
    /// Minimum aperture
    pub min_aperture: f32,
}

/// Model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Model name
    pub name: String,
    /// Model type
    pub model_type: ModelType,
    /// Model version
    pub version: String,
    /// Input dimensions
    pub input_dims: Vec<usize>,
    /// Output dimensions
    pub output_dims: Vec<usize>,
    /// Model parameters
    pub parameters: HashMap<String, String>,
}

/// Model types for computer vision
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelType {
    /// Classification model
    Classification,
    /// Object detection model
    ObjectDetection,
    /// Segmentation model
    Segmentation,
    /// Feature extraction model
    FeatureExtraction,
    /// Custom model type
    Custom(String),
}

/// Object detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObjectDetectionResult {
    /// List of detections
    pub detections: Vec<Detection>,
    /// Detection metadata
    pub metadata: DetectionMetadata,
    /// Processing statistics
    pub statistics: Option<FeatureStatistics>,
}

/// Prediction metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionMetadata {
    /// Model used for prediction
    pub model_name: String,
    /// Prediction timestamp
    pub timestamp: String,
    /// Processing time
    pub processing_time_ms: f64,
    /// Confidence threshold used
    pub confidence_threshold: f32,
}

/// Prediction result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionResult {
    /// Predictions
    pub predictions: Vec<CVPrediction>,
    /// Metadata
    pub metadata: PredictionMetadata,
    /// Confidence scores
    pub confidence_scores: Option<ConfidenceScores>,
}

impl Default for ComputeDevice {
    fn default() -> Self {
        Self::CPU
    }
}

impl Default for ExtractorType {
    fn default() -> Self {
        Self::SIFT
    }
}

impl Default for ModelType {
    fn default() -> Self {
        Self::Classification
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_values() {
        assert_eq!(ImageDataType::default(), ImageDataType::UInt8);
        assert_eq!(ColorSpace::default(), ColorSpace::RGB);
        assert_eq!(ImageFormat::default(), ImageFormat::PNG);
        assert_eq!(ProcessingMode::default(), ProcessingMode::Batch);
        assert_eq!(
            ProcessingComplexity::default(),
            ProcessingComplexity::Medium
        );
    }

    #[test]
    fn test_processing_complexity_ordering() {
        assert!(ProcessingComplexity::Low < ProcessingComplexity::Medium);
        assert!(ProcessingComplexity::Medium < ProcessingComplexity::High);
        assert!(ProcessingComplexity::High < ProcessingComplexity::Ultra);
    }

    #[test]
    fn test_transform_parameter_variants() {
        let bool_param = TransformParameter::Bool(true);
        let int_param = TransformParameter::Int(42);
        let float_param = TransformParameter::Float(3.14);
        let string_param = TransformParameter::String("test".to_string());

        match bool_param {
            TransformParameter::Bool(val) => assert!(val),
            _ => panic!("Expected Bool variant"),
        }

        match int_param {
            TransformParameter::Int(val) => assert_eq!(val, 42),
            _ => panic!("Expected Int variant"),
        }

        match float_param {
            TransformParameter::Float(val) => assert!((val - 3.14).abs() < f64::EPSILON),
            _ => panic!("Expected Float variant"),
        }

        match string_param {
            TransformParameter::String(val) => assert_eq!(val, "test"),
            _ => panic!("Expected String variant"),
        }
    }

    #[test]
    fn test_serialization() {
        let data_type = ImageDataType::Float32;
        let serialized = serde_json::to_string(&data_type).unwrap();
        let deserialized: ImageDataType = serde_json::from_str(&serialized).unwrap();
        assert_eq!(data_type, deserialized);

        let color_space = ColorSpace::HSV;
        let serialized = serde_json::to_string(&color_space).unwrap();
        let deserialized: ColorSpace = serde_json::from_str(&serialized).unwrap();
        assert_eq!(color_space, deserialized);
    }
}
