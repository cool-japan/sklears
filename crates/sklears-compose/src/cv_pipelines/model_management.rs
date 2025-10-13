//! Model management and configuration
//!
//! This module provides comprehensive model management capabilities including
//! model specifications, metadata, performance characteristics, processor
//! configurations, and quality enhancement for computer vision pipelines.

use super::types_config::{OutputType, ProcessorType, TransformParameter};
use scirs2_core::ndarray::Array1;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::SystemTime;

/// Model input specification for computer vision models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInputSpec {
    /// Expected input image size (width, height)
    pub image_size: (usize, usize),
    /// Number of input channels
    pub channels: usize,
    /// Normalization parameters
    pub normalization: NormalizationSpec,
    /// Required preprocessing steps
    pub preprocessing: Vec<String>,
    /// Input tensor format (NCHW, NHWC, etc.)
    pub tensor_format: TensorFormat,
    /// Data type requirements
    pub data_type: ModelDataType,
    /// Input value range
    pub value_range: (f32, f32),
}

impl Default for ModelInputSpec {
    fn default() -> Self {
        Self {
            image_size: (224, 224),
            channels: 3,
            normalization: NormalizationSpec::imagenet(),
            preprocessing: vec!["resize".to_string(), "normalize".to_string()],
            tensor_format: TensorFormat::NCHW,
            data_type: ModelDataType::Float32,
            value_range: (0.0, 1.0),
        }
    }
}

impl ModelInputSpec {
    /// Create specification for classification models
    #[must_use]
    pub fn classification(image_size: (usize, usize)) -> Self {
        Self {
            image_size,
            channels: 3,
            normalization: NormalizationSpec::imagenet(),
            preprocessing: vec![
                "resize".to_string(),
                "center_crop".to_string(),
                "normalize".to_string(),
            ],
            tensor_format: TensorFormat::NCHW,
            data_type: ModelDataType::Float32,
            value_range: (0.0, 1.0),
        }
    }

    /// Create specification for object detection models
    #[must_use]
    pub fn object_detection(image_size: (usize, usize)) -> Self {
        Self {
            image_size,
            channels: 3,
            normalization: NormalizationSpec::coco(),
            preprocessing: vec![
                "resize".to_string(),
                "letterbox".to_string(),
                "normalize".to_string(),
            ],
            tensor_format: TensorFormat::NCHW,
            data_type: ModelDataType::Float32,
            value_range: (0.0, 1.0),
        }
    }

    /// Create specification for segmentation models
    #[must_use]
    pub fn segmentation(image_size: (usize, usize)) -> Self {
        Self {
            image_size,
            channels: 3,
            normalization: NormalizationSpec::cityscapes(),
            preprocessing: vec!["resize".to_string(), "normalize".to_string()],
            tensor_format: TensorFormat::NCHW,
            data_type: ModelDataType::Float32,
            value_range: (0.0, 1.0),
        }
    }

    /// Validate input specification
    pub fn validate(&self) -> Result<(), ModelError> {
        if self.image_size.0 == 0 || self.image_size.1 == 0 {
            return Err(ModelError::InvalidInputSpec(
                "Image size must be greater than zero".to_string(),
            ));
        }

        if self.channels == 0 {
            return Err(ModelError::InvalidInputSpec(
                "Number of channels must be greater than zero".to_string(),
            ));
        }

        if self.value_range.0 >= self.value_range.1 {
            return Err(ModelError::InvalidInputSpec(
                "Value range minimum must be less than maximum".to_string(),
            ));
        }

        Ok(())
    }

    /// Calculate memory requirements for input
    #[must_use]
    pub fn memory_requirements(&self, batch_size: usize) -> usize {
        let element_size = match self.data_type {
            ModelDataType::Float32 => 4,
            ModelDataType::Float16 => 2,
            ModelDataType::Int8 => 1,
            ModelDataType::UInt8 => 1,
        };

        batch_size * self.channels * self.image_size.0 * self.image_size.1 * element_size
    }
}

/// Normalization specification for model inputs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormalizationSpec {
    /// Mean values per channel
    pub mean: Array1<f32>,
    /// Standard deviation per channel
    pub std: Array1<f32>,
    /// Value range (min, max) for clipping
    pub range: (f32, f32),
    /// Normalization type
    pub norm_type: NormalizationType,
}

impl Default for NormalizationSpec {
    fn default() -> Self {
        Self::imagenet()
    }
}

impl NormalizationSpec {
    /// `ImageNet` normalization parameters
    #[must_use]
    pub fn imagenet() -> Self {
        Self {
            mean: Array1::from(vec![0.485, 0.456, 0.406]),
            std: Array1::from(vec![0.229, 0.224, 0.225]),
            range: (0.0, 1.0),
            norm_type: NormalizationType::StandardScore,
        }
    }

    /// COCO dataset normalization parameters
    #[must_use]
    pub fn coco() -> Self {
        Self {
            mean: Array1::from(vec![0.485, 0.456, 0.406]),
            std: Array1::from(vec![0.229, 0.224, 0.225]),
            range: (0.0, 1.0),
            norm_type: NormalizationType::StandardScore,
        }
    }

    /// Cityscapes dataset normalization parameters
    #[must_use]
    pub fn cityscapes() -> Self {
        Self {
            mean: Array1::from(vec![0.485, 0.456, 0.406]),
            std: Array1::from(vec![0.229, 0.224, 0.225]),
            range: (0.0, 1.0),
            norm_type: NormalizationType::StandardScore,
        }
    }

    /// Custom normalization parameters
    #[must_use]
    pub fn custom(mean: Vec<f32>, std: Vec<f32>, range: (f32, f32)) -> Self {
        Self {
            mean: Array1::from(mean),
            std: Array1::from(std),
            range,
            norm_type: NormalizationType::StandardScore,
        }
    }

    /// Min-max normalization (0-1)
    #[must_use]
    pub fn min_max() -> Self {
        Self {
            mean: Array1::from(vec![0.0, 0.0, 0.0]),
            std: Array1::from(vec![255.0, 255.0, 255.0]),
            range: (0.0, 1.0),
            norm_type: NormalizationType::MinMax,
        }
    }
}

/// Normalization types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NormalizationType {
    /// Standard score normalization (z-score)
    StandardScore,
    /// Min-max normalization
    MinMax,
    /// L2 normalization
    L2,
    /// No normalization
    None,
}

/// Tensor format specifications
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TensorFormat {
    /// Batch, Channels, Height, Width
    NCHW,
    /// Batch, Height, Width, Channels
    NHWC,
    /// Channels, Height, Width (single image)
    CHW,
    /// Height, Width, Channels (single image)
    HWC,
}

impl Default for TensorFormat {
    fn default() -> Self {
        Self::NCHW
    }
}

/// Model data types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelDataType {
    /// 32-bit floating point
    Float32,
    /// 16-bit floating point
    Float16,
    /// 8-bit signed integer
    Int8,
    /// 8-bit unsigned integer
    UInt8,
}

impl Default for ModelDataType {
    fn default() -> Self {
        Self::Float32
    }
}

/// Model output specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelOutputSpec {
    /// Output tensor shapes
    pub output_shapes: Vec<Vec<usize>>,
    /// Output types for each tensor
    pub output_types: Vec<OutputType>,
    /// Post-processing requirements
    pub postprocessing: Vec<String>,
    /// Output interpretation
    pub interpretation: OutputInterpretation,
    /// Confidence thresholds
    pub confidence_thresholds: HashMap<String, f64>,
}

impl Default for ModelOutputSpec {
    fn default() -> Self {
        Self {
            output_shapes: vec![vec![1000]], // ImageNet classes
            output_types: vec![OutputType::Classification],
            postprocessing: vec!["softmax".to_string()],
            interpretation: OutputInterpretation::default(),
            confidence_thresholds: HashMap::new(),
        }
    }
}

impl ModelOutputSpec {
    /// Create specification for classification outputs
    #[must_use]
    pub fn classification(num_classes: usize) -> Self {
        let mut thresholds = HashMap::new();
        thresholds.insert("min_confidence".to_string(), 0.5);

        Self {
            output_shapes: vec![vec![num_classes]],
            output_types: vec![OutputType::Classification],
            postprocessing: vec!["softmax".to_string(), "argmax".to_string()],
            interpretation: OutputInterpretation::classification(),
            confidence_thresholds: thresholds,
        }
    }

    /// Create specification for object detection outputs
    #[must_use]
    pub fn object_detection() -> Self {
        let mut thresholds = HashMap::new();
        thresholds.insert("detection_threshold".to_string(), 0.5);
        thresholds.insert("nms_threshold".to_string(), 0.4);

        Self {
            output_shapes: vec![
                vec![1, 25200, 85], // YOLO-style output
            ],
            output_types: vec![OutputType::Detection],
            postprocessing: vec![
                "decode_boxes".to_string(),
                "nms".to_string(),
                "filter_confidence".to_string(),
            ],
            interpretation: OutputInterpretation::object_detection(),
            confidence_thresholds: thresholds,
        }
    }

    /// Create specification for segmentation outputs
    #[must_use]
    pub fn segmentation(num_classes: usize, height: usize, width: usize) -> Self {
        Self {
            output_shapes: vec![vec![num_classes, height, width]],
            output_types: vec![OutputType::Segmentation],
            postprocessing: vec!["argmax".to_string(), "colormap".to_string()],
            interpretation: OutputInterpretation::segmentation(),
            confidence_thresholds: HashMap::new(),
        }
    }
}

/// Output interpretation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputInterpretation {
    /// Class labels (for classification/detection)
    pub class_labels: Vec<String>,
    /// Label mapping
    pub label_mapping: HashMap<usize, String>,
    /// Output format description
    pub format_description: String,
    /// Units for regression outputs
    pub units: Option<String>,
}

impl Default for OutputInterpretation {
    fn default() -> Self {
        Self {
            class_labels: vec![],
            label_mapping: HashMap::new(),
            format_description: "Raw model output".to_string(),
            units: None,
        }
    }
}

impl OutputInterpretation {
    /// Create interpretation for classification
    #[must_use]
    pub fn classification() -> Self {
        Self {
            class_labels: vec![],
            label_mapping: HashMap::new(),
            format_description: "Class probabilities".to_string(),
            units: Some("probability".to_string()),
        }
    }

    /// Create interpretation for object detection
    #[must_use]
    pub fn object_detection() -> Self {
        Self {
            class_labels: vec![],
            label_mapping: HashMap::new(),
            format_description: "Bounding boxes with class and confidence".to_string(),
            units: Some("normalized_coordinates".to_string()),
        }
    }

    /// Create interpretation for segmentation
    #[must_use]
    pub fn segmentation() -> Self {
        Self {
            class_labels: vec![],
            label_mapping: HashMap::new(),
            format_description: "Per-pixel class predictions".to_string(),
            units: Some("class_index".to_string()),
        }
    }
}

/// Model performance characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPerformance {
    /// Average inference time in milliseconds
    pub inference_time: f64,
    /// Memory usage in bytes
    pub memory_usage: u64,
    /// FLOPs (floating point operations) required
    pub flops: u64,
    /// Model accuracy metrics
    pub accuracy: AccuracyMetrics,
    /// Throughput metrics
    pub throughput: ThroughputMetrics,
    /// Resource utilization
    pub resource_utilization: ModelResourceUtilization,
}

impl Default for ModelPerformance {
    fn default() -> Self {
        Self {
            inference_time: 100.0,           // 100ms
            memory_usage: 100 * 1024 * 1024, // 100MB
            flops: 1_000_000_000,            // 1 GFLOP
            accuracy: AccuracyMetrics::default(),
            throughput: ThroughputMetrics::default(),
            resource_utilization: ModelResourceUtilization::default(),
        }
    }
}

impl ModelPerformance {
    /// Create performance profile for lightweight mobile models
    #[must_use]
    pub fn mobile_optimized() -> Self {
        Self {
            inference_time: 50.0,           // 50ms
            memory_usage: 20 * 1024 * 1024, // 20MB
            flops: 100_000_000,             // 100 MFLOP
            accuracy: AccuracyMetrics::mobile(),
            throughput: ThroughputMetrics::mobile(),
            resource_utilization: ModelResourceUtilization::low(),
        }
    }

    /// Create performance profile for server-side models
    #[must_use]
    pub fn server_optimized() -> Self {
        Self {
            inference_time: 200.0,           // 200ms
            memory_usage: 500 * 1024 * 1024, // 500MB
            flops: 10_000_000_000,           // 10 GFLOP
            accuracy: AccuracyMetrics::high_accuracy(),
            throughput: ThroughputMetrics::server(),
            resource_utilization: ModelResourceUtilization::high(),
        }
    }

    /// Create performance profile for edge devices
    #[must_use]
    pub fn edge_optimized() -> Self {
        Self {
            inference_time: 75.0,           // 75ms
            memory_usage: 50 * 1024 * 1024, // 50MB
            flops: 500_000_000,             // 500 MFLOP
            accuracy: AccuracyMetrics::balanced(),
            throughput: ThroughputMetrics::edge(),
            resource_utilization: ModelResourceUtilization::moderate(),
        }
    }
}

/// Accuracy metrics for model evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccuracyMetrics {
    /// Top-1 accuracy (for classification)
    pub top1_accuracy: Option<f64>,
    /// Top-5 accuracy (for classification)
    pub top5_accuracy: Option<f64>,
    /// Mean Average Precision (for detection/segmentation)
    pub map: Option<f64>,
    /// Intersection over Union (for detection/segmentation)
    pub iou: Option<f64>,
    /// F1 score
    pub f1_score: Option<f64>,
    /// Precision
    pub precision: Option<f64>,
    /// Recall
    pub recall: Option<f64>,
    /// Custom metrics
    pub custom_metrics: HashMap<String, f64>,
}

impl Default for AccuracyMetrics {
    fn default() -> Self {
        Self {
            top1_accuracy: Some(0.75),
            top5_accuracy: Some(0.92),
            map: None,
            iou: None,
            f1_score: Some(0.75),
            precision: Some(0.75),
            recall: Some(0.75),
            custom_metrics: HashMap::new(),
        }
    }
}

impl AccuracyMetrics {
    /// Create metrics profile for mobile models
    #[must_use]
    pub fn mobile() -> Self {
        Self {
            top1_accuracy: Some(0.65),
            top5_accuracy: Some(0.85),
            map: None,
            iou: None,
            f1_score: Some(0.65),
            precision: Some(0.70),
            recall: Some(0.60),
            custom_metrics: HashMap::new(),
        }
    }

    /// Create metrics profile for high-accuracy models
    #[must_use]
    pub fn high_accuracy() -> Self {
        Self {
            top1_accuracy: Some(0.85),
            top5_accuracy: Some(0.97),
            map: Some(0.75),
            iou: Some(0.80),
            f1_score: Some(0.85),
            precision: Some(0.88),
            recall: Some(0.82),
            custom_metrics: HashMap::new(),
        }
    }

    /// Create balanced accuracy metrics
    #[must_use]
    pub fn balanced() -> Self {
        Self::default()
    }
}

/// Throughput metrics for model performance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThroughputMetrics {
    /// Images per second
    pub images_per_second: f64,
    /// Batch processing throughput
    pub batch_throughput: HashMap<usize, f64>, // batch_size -> throughput
    /// Peak throughput
    pub peak_throughput: f64,
    /// Sustained throughput
    pub sustained_throughput: f64,
}

impl Default for ThroughputMetrics {
    fn default() -> Self {
        let mut batch_throughput = HashMap::new();
        batch_throughput.insert(1, 10.0);
        batch_throughput.insert(8, 40.0);
        batch_throughput.insert(16, 60.0);

        Self {
            images_per_second: 10.0,
            batch_throughput,
            peak_throughput: 80.0,
            sustained_throughput: 10.0,
        }
    }
}

impl ThroughputMetrics {
    /// Create throughput metrics for mobile devices
    #[must_use]
    pub fn mobile() -> Self {
        let mut batch_throughput = HashMap::new();
        batch_throughput.insert(1, 20.0);
        batch_throughput.insert(4, 30.0);

        Self {
            images_per_second: 20.0,
            batch_throughput,
            peak_throughput: 35.0,
            sustained_throughput: 18.0,
        }
    }

    /// Create throughput metrics for server deployments
    #[must_use]
    pub fn server() -> Self {
        let mut batch_throughput = HashMap::new();
        batch_throughput.insert(1, 5.0);
        batch_throughput.insert(8, 30.0);
        batch_throughput.insert(16, 50.0);
        batch_throughput.insert(32, 80.0);
        batch_throughput.insert(64, 100.0);

        Self {
            images_per_second: 5.0,
            batch_throughput,
            peak_throughput: 120.0,
            sustained_throughput: 4.0,
        }
    }

    /// Create throughput metrics for edge devices
    #[must_use]
    pub fn edge() -> Self {
        let mut batch_throughput = HashMap::new();
        batch_throughput.insert(1, 13.0);
        batch_throughput.insert(4, 20.0);
        batch_throughput.insert(8, 25.0);

        Self {
            images_per_second: 13.0,
            batch_throughput,
            peak_throughput: 28.0,
            sustained_throughput: 12.0,
        }
    }
}

/// Resource utilization for model execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelResourceUtilization {
    /// CPU utilization percentage
    pub cpu_utilization: f64,
    /// GPU utilization percentage (if applicable)
    pub gpu_utilization: Option<f64>,
    /// Memory utilization percentage
    pub memory_utilization: f64,
    /// Power consumption in watts
    pub power_consumption: Option<f64>,
    /// Thermal characteristics
    pub thermal_profile: ThermalProfile,
}

impl Default for ModelResourceUtilization {
    fn default() -> Self {
        Self {
            cpu_utilization: 50.0,
            gpu_utilization: Some(60.0),
            memory_utilization: 40.0,
            power_consumption: Some(15.0),
            thermal_profile: ThermalProfile::default(),
        }
    }
}

impl ModelResourceUtilization {
    /// Create low resource utilization profile
    #[must_use]
    pub fn low() -> Self {
        Self {
            cpu_utilization: 20.0,
            gpu_utilization: Some(25.0),
            memory_utilization: 15.0,
            power_consumption: Some(5.0),
            thermal_profile: ThermalProfile::cool(),
        }
    }

    /// Create moderate resource utilization profile
    #[must_use]
    pub fn moderate() -> Self {
        Self::default()
    }

    /// Create high resource utilization profile
    #[must_use]
    pub fn high() -> Self {
        Self {
            cpu_utilization: 80.0,
            gpu_utilization: Some(90.0),
            memory_utilization: 70.0,
            power_consumption: Some(50.0),
            thermal_profile: ThermalProfile::hot(),
        }
    }
}

/// Thermal profile for model execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermalProfile {
    /// Operating temperature range (min, max) in Celsius
    pub temperature_range: (f32, f32),
    /// Thermal design power in watts
    pub tdp: Option<f32>,
    /// Cooling requirements
    pub cooling_requirements: CoolingRequirements,
}

impl Default for ThermalProfile {
    fn default() -> Self {
        Self {
            temperature_range: (20.0, 70.0),
            tdp: Some(15.0),
            cooling_requirements: CoolingRequirements::Passive,
        }
    }
}

impl ThermalProfile {
    /// Create cool thermal profile for low-power models
    #[must_use]
    pub fn cool() -> Self {
        Self {
            temperature_range: (15.0, 50.0),
            tdp: Some(5.0),
            cooling_requirements: CoolingRequirements::None,
        }
    }

    /// Create hot thermal profile for high-performance models
    #[must_use]
    pub fn hot() -> Self {
        Self {
            temperature_range: (25.0, 85.0),
            tdp: Some(50.0),
            cooling_requirements: CoolingRequirements::Active,
        }
    }
}

/// Cooling requirements for model execution
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CoolingRequirements {
    /// No cooling required
    None,
    /// Passive cooling (heat sinks)
    Passive,
    /// Active cooling (fans)
    Active,
    /// Liquid cooling
    Liquid,
}

/// Model metadata for management and deployment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    /// Model name
    pub name: String,
    /// Model version
    pub version: String,
    /// Author/organization
    pub author: String,
    /// Model description
    pub description: String,
    /// Training dataset information
    pub training_dataset: Option<String>,
    /// License information
    pub license: Option<String>,
    /// Model file size in bytes
    pub model_size: u64,
    /// Supported platforms/architectures
    pub platforms: Vec<String>,
    /// Creation timestamp
    pub created_at: SystemTime,
    /// Last modified timestamp
    pub modified_at: SystemTime,
    /// Model tags for categorization
    pub tags: Vec<String>,
    /// Deployment requirements
    pub deployment_requirements: DeploymentRequirements,
}

impl Default for ModelMetadata {
    fn default() -> Self {
        Self {
            name: "Unknown Model".to_string(),
            version: "1.0.0".to_string(),
            author: "Unknown".to_string(),
            description: "Computer vision model".to_string(),
            training_dataset: None,
            license: None,
            model_size: 0,
            platforms: vec!["cpu".to_string()],
            created_at: SystemTime::now(),
            modified_at: SystemTime::now(),
            tags: vec![],
            deployment_requirements: DeploymentRequirements::default(),
        }
    }
}

impl ModelMetadata {
    /// Create metadata for a classification model
    #[must_use]
    pub fn classification(name: &str, version: &str) -> Self {
        Self {
            name: name.to_string(),
            version: version.to_string(),
            author: "Unknown".to_string(),
            description: "Image classification model".to_string(),
            training_dataset: Some("ImageNet".to_string()),
            license: Some("MIT".to_string()),
            model_size: 100 * 1024 * 1024, // 100MB
            platforms: vec!["cpu".to_string(), "gpu".to_string()],
            created_at: SystemTime::now(),
            modified_at: SystemTime::now(),
            tags: vec!["classification".to_string(), "vision".to_string()],
            deployment_requirements: DeploymentRequirements::standard(),
        }
    }

    /// Update modification timestamp
    pub fn touch(&mut self) {
        self.modified_at = SystemTime::now();
    }

    /// Add a tag if not already present
    pub fn add_tag(&mut self, tag: String) {
        if !self.tags.contains(&tag) {
            self.tags.push(tag);
        }
    }

    /// Check if model has a specific tag
    #[must_use]
    pub fn has_tag(&self, tag: &str) -> bool {
        self.tags.contains(&tag.to_string())
    }
}

/// Deployment requirements for models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeploymentRequirements {
    /// Minimum runtime version required
    pub min_runtime_version: String,
    /// Required dependencies
    pub dependencies: Vec<String>,
    /// Hardware requirements
    pub hardware_requirements: HardwareRequirements,
    /// Environment variables
    pub environment_variables: HashMap<String, String>,
    /// Configuration files needed
    pub config_files: Vec<String>,
}

impl Default for DeploymentRequirements {
    fn default() -> Self {
        Self {
            min_runtime_version: "1.0.0".to_string(),
            dependencies: vec![],
            hardware_requirements: HardwareRequirements::default(),
            environment_variables: HashMap::new(),
            config_files: vec![],
        }
    }
}

impl DeploymentRequirements {
    /// Create standard deployment requirements
    #[must_use]
    pub fn standard() -> Self {
        Self {
            min_runtime_version: "1.0.0".to_string(),
            dependencies: vec!["opencv".to_string(), "numpy".to_string()],
            hardware_requirements: HardwareRequirements::standard(),
            environment_variables: HashMap::new(),
            config_files: vec!["model_config.json".to_string()],
        }
    }

    /// Create minimal deployment requirements
    #[must_use]
    pub fn minimal() -> Self {
        Self {
            min_runtime_version: "1.0.0".to_string(),
            dependencies: vec![],
            hardware_requirements: HardwareRequirements::minimal(),
            environment_variables: HashMap::new(),
            config_files: vec![],
        }
    }
}

/// Hardware requirements for model deployment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareRequirements {
    /// Minimum RAM in bytes
    pub min_ram: u64,
    /// Minimum storage in bytes
    pub min_storage: u64,
    /// Required CPU features
    pub cpu_features: Vec<String>,
    /// GPU requirements (if applicable)
    pub gpu_requirements: Option<GpuRequirements>,
    /// Architecture requirements
    pub architectures: Vec<String>,
}

impl Default for HardwareRequirements {
    fn default() -> Self {
        Self {
            min_ram: 1024 * 1024 * 1024,    // 1GB
            min_storage: 500 * 1024 * 1024, // 500MB
            cpu_features: vec!["sse4.1".to_string()],
            gpu_requirements: None,
            architectures: vec!["x86_64".to_string(), "arm64".to_string()],
        }
    }
}

impl HardwareRequirements {
    /// Create standard hardware requirements
    #[must_use]
    pub fn standard() -> Self {
        Self::default()
    }

    /// Create minimal hardware requirements
    #[must_use]
    pub fn minimal() -> Self {
        Self {
            min_ram: 256 * 1024 * 1024,     // 256MB
            min_storage: 100 * 1024 * 1024, // 100MB
            cpu_features: vec![],
            gpu_requirements: None,
            architectures: vec!["x86_64".to_string(), "arm64".to_string()],
        }
    }
}

/// GPU requirements for models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuRequirements {
    /// Minimum GPU memory in bytes
    pub min_gpu_memory: u64,
    /// Required compute capability
    pub compute_capability: Option<String>,
    /// Supported GPU vendors
    pub vendors: Vec<String>,
    /// Minimum GPU driver version
    pub min_driver_version: Option<String>,
}

impl Default for GpuRequirements {
    fn default() -> Self {
        Self {
            min_gpu_memory: 2 * 1024 * 1024 * 1024, // 2GB
            compute_capability: Some("6.0".to_string()),
            vendors: vec!["NVIDIA".to_string(), "AMD".to_string()],
            min_driver_version: Some("450.0".to_string()),
        }
    }
}

/// Post-processor configuration for model outputs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessorConfig {
    /// Processor type
    pub processor_type: ProcessorType,
    /// Processing parameters
    pub parameters: HashMap<String, TransformParameter>,
    /// Quality enhancement settings
    pub quality_enhancement: QualityEnhancementConfig,
    /// Processing order/priority
    pub priority: i32,
    /// Enable/disable flag
    pub enabled: bool,
}

impl Default for ProcessorConfig {
    fn default() -> Self {
        Self {
            processor_type: ProcessorType::Filtering,
            parameters: HashMap::new(),
            quality_enhancement: QualityEnhancementConfig::default(),
            priority: 0,
            enabled: true,
        }
    }
}

impl ProcessorConfig {
    /// Create NMS processor configuration
    #[must_use]
    pub fn nms(confidence_threshold: f64, iou_threshold: f64) -> Self {
        let mut parameters = HashMap::new();
        parameters.insert(
            "confidence_threshold".to_string(),
            TransformParameter::Float(confidence_threshold),
        );
        parameters.insert(
            "iou_threshold".to_string(),
            TransformParameter::Float(iou_threshold),
        );

        Self {
            processor_type: ProcessorType::NonMaximumSuppression,
            parameters,
            quality_enhancement: QualityEnhancementConfig::default(),
            priority: 10,
            enabled: true,
        }
    }

    /// Create filtering processor configuration
    #[must_use]
    pub fn filtering(min_confidence: f64) -> Self {
        let mut parameters = HashMap::new();
        parameters.insert(
            "min_confidence".to_string(),
            TransformParameter::Float(min_confidence),
        );

        Self {
            processor_type: ProcessorType::Filtering,
            parameters,
            quality_enhancement: QualityEnhancementConfig::default(),
            priority: 5,
            enabled: true,
        }
    }

    /// Create tracking processor configuration
    #[must_use]
    pub fn tracking() -> Self {
        let mut parameters = HashMap::new();
        parameters.insert("max_disappeared".to_string(), TransformParameter::Int(30));
        parameters.insert("max_distance".to_string(), TransformParameter::Float(50.0));

        Self {
            processor_type: ProcessorType::Tracking,
            parameters,
            quality_enhancement: QualityEnhancementConfig::default(),
            priority: 20,
            enabled: true,
        }
    }
}

/// Quality enhancement configuration for post-processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityEnhancementConfig {
    /// Confidence threshold adjustment
    pub confidence_adjustment: f64,
    /// Enable noise filtering
    pub noise_filtering: bool,
    /// Enable outlier removal
    pub outlier_removal: bool,
    /// Enable temporal consistency (for video)
    pub temporal_consistency: bool,
    /// Smoothing parameters
    pub smoothing: SmoothingConfig,
}

impl Default for QualityEnhancementConfig {
    fn default() -> Self {
        Self {
            confidence_adjustment: 0.0,
            noise_filtering: false,
            outlier_removal: false,
            temporal_consistency: false,
            smoothing: SmoothingConfig::default(),
        }
    }
}

/// Smoothing configuration for quality enhancement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmoothingConfig {
    /// Enable smoothing
    pub enabled: bool,
    /// Smoothing window size
    pub window_size: usize,
    /// Smoothing algorithm
    pub algorithm: SmoothingAlgorithm,
    /// Smoothing strength (0.0-1.0)
    pub strength: f64,
}

impl Default for SmoothingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            window_size: 5,
            algorithm: SmoothingAlgorithm::MovingAverage,
            strength: 0.5,
        }
    }
}

/// Smoothing algorithms for post-processing
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SmoothingAlgorithm {
    /// Moving average smoothing
    MovingAverage,
    /// Exponential smoothing
    Exponential,
    /// Gaussian smoothing
    Gaussian,
    /// Median filtering
    Median,
}

/// Model management errors
#[derive(Debug, Clone, PartialEq)]
pub enum ModelError {
    /// Invalid input specification
    InvalidInputSpec(String),
    /// Invalid output specification
    InvalidOutputSpec(String),
    /// Model loading error
    LoadingError(String),
    /// Inference error
    InferenceError(String),
    /// Post-processing error
    PostProcessingError(String),
    /// Configuration error
    ConfigurationError(String),
    /// Hardware requirement not met
    HardwareRequirementNotMet(String),
    /// Unsupported platform
    UnsupportedPlatform(String),
}

impl std::fmt::Display for ModelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidInputSpec(msg) => write!(f, "Invalid input specification: {msg}"),
            Self::InvalidOutputSpec(msg) => write!(f, "Invalid output specification: {msg}"),
            Self::LoadingError(msg) => write!(f, "Model loading error: {msg}"),
            Self::InferenceError(msg) => write!(f, "Inference error: {msg}"),
            Self::PostProcessingError(msg) => write!(f, "Post-processing error: {msg}"),
            Self::ConfigurationError(msg) => write!(f, "Configuration error: {msg}"),
            Self::HardwareRequirementNotMet(msg) => {
                write!(f, "Hardware requirement not met: {msg}")
            }
            Self::UnsupportedPlatform(platform) => write!(f, "Unsupported platform: {platform}"),
        }
    }
}

impl std::error::Error for ModelError {}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_input_spec() {
        let spec = ModelInputSpec::classification((224, 224));
        assert_eq!(spec.image_size, (224, 224));
        assert_eq!(spec.channels, 3);
        assert!(spec.validate().is_ok());

        let memory = spec.memory_requirements(1);
        assert_eq!(memory, 1 * 3 * 224 * 224 * 4); // Float32 = 4 bytes
    }

    #[test]
    fn test_normalization_spec() {
        let imagenet = NormalizationSpec::imagenet();
        assert_eq!(imagenet.mean.len(), 3);
        assert_eq!(imagenet.std.len(), 3);

        let custom =
            NormalizationSpec::custom(vec![0.5, 0.5, 0.5], vec![0.5, 0.5, 0.5], (0.0, 1.0));
        assert_eq!(custom.mean.len(), 3);
        assert_eq!(custom.norm_type, NormalizationType::StandardScore);
    }

    #[test]
    fn test_model_output_spec() {
        let classification = ModelOutputSpec::classification(1000);
        assert_eq!(classification.output_shapes[0], vec![1000]);
        assert_eq!(classification.output_types[0], OutputType::Classification);

        let detection = ModelOutputSpec::object_detection();
        assert_eq!(detection.output_types[0], OutputType::Detection);
        assert!(detection
            .confidence_thresholds
            .contains_key("detection_threshold"));
    }

    #[test]
    fn test_model_performance() {
        let mobile = ModelPerformance::mobile_optimized();
        assert_eq!(mobile.inference_time, 50.0);
        assert!(mobile.memory_usage < 50 * 1024 * 1024);

        let server = ModelPerformance::server_optimized();
        assert!(server.inference_time > mobile.inference_time);
        assert!(server.memory_usage > mobile.memory_usage);
    }

    #[test]
    fn test_accuracy_metrics() {
        let mobile = AccuracyMetrics::mobile();
        let high_acc = AccuracyMetrics::high_accuracy();

        assert!(high_acc.top1_accuracy.unwrap() > mobile.top1_accuracy.unwrap());
        assert!(high_acc.f1_score.unwrap() > mobile.f1_score.unwrap());
    }

    #[test]
    fn test_model_metadata() {
        let mut metadata = ModelMetadata::classification("ResNet50", "1.0.0");
        assert_eq!(metadata.name, "ResNet50");
        assert_eq!(metadata.version, "1.0.0");

        metadata.add_tag("pretrained".to_string());
        assert!(metadata.has_tag("pretrained"));
        assert!(metadata.has_tag("classification"));

        let before = metadata.modified_at;
        metadata.touch();
        assert!(metadata.modified_at > before);
    }

    #[test]
    fn test_processor_config() {
        let nms = ProcessorConfig::nms(0.5, 0.4);
        assert_eq!(nms.processor_type, ProcessorType::NonMaximumSuppression);
        assert_eq!(nms.priority, 10);
        assert!(nms.enabled);

        let filtering = ProcessorConfig::filtering(0.3);
        assert_eq!(filtering.processor_type, ProcessorType::Filtering);
        assert_eq!(filtering.priority, 5);
    }

    #[test]
    fn test_deployment_requirements() {
        let standard = DeploymentRequirements::standard();
        assert!(!standard.dependencies.is_empty());
        assert!(standard.hardware_requirements.min_ram > 0);

        let minimal = DeploymentRequirements::minimal();
        assert!(minimal.dependencies.is_empty());
        assert!(minimal.hardware_requirements.min_ram < standard.hardware_requirements.min_ram);
    }

    #[test]
    fn test_model_error_display() {
        let error = ModelError::InvalidInputSpec("Image size must be positive".to_string());
        let error_str = error.to_string();
        assert!(error_str.contains("Invalid input specification"));
        assert!(error_str.contains("Image size must be positive"));

        let error = ModelError::UnsupportedPlatform("windows-arm".to_string());
        let error_str = error.to_string();
        assert!(error_str.contains("Unsupported platform"));
        assert!(error_str.contains("windows-arm"));
    }
}
