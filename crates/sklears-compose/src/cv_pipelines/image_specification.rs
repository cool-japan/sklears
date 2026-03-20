//! Image specification and data type management
//!
//! This module provides comprehensive image specification handling including
//! format validation, color space management, normalization parameters,
//! and data type conversions for computer vision pipelines.

use super::types_config::{ColorSpace, ImageDataType, ImageFormat};
use scirs2_core::ndarray::{Array1, Array3};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Comprehensive image specification for input validation and processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageSpecification {
    /// Expected image dimensions (height, width)
    pub dimensions: Option<(usize, usize)>,
    /// Number of channels (1=grayscale, 3=RGB, 4=RGBA)
    pub channels: usize,
    /// Data type (uint8, float32, etc.)
    pub dtype: ImageDataType,
    /// Color space (RGB, BGR, HSV, LAB, etc.)
    pub color_space: ColorSpace,
    /// Bit depth per channel
    pub bit_depth: u8,
    /// Supported input formats
    pub supported_formats: Vec<ImageFormat>,
    /// Validation requirements
    pub validation: ImageValidationSpec,
    /// Preprocessing requirements
    pub preprocessing_requirements: Vec<String>,
}

impl Default for ImageSpecification {
    fn default() -> Self {
        Self {
            dimensions: Some((224, 224)),
            channels: 3,
            dtype: ImageDataType::UInt8,
            color_space: ColorSpace::RGB,
            bit_depth: 8,
            supported_formats: vec![ImageFormat::JPEG, ImageFormat::PNG, ImageFormat::BMP],
            validation: ImageValidationSpec::default(),
            preprocessing_requirements: vec![],
        }
    }
}

impl ImageSpecification {
    /// Create a new image specification with specified dimensions
    #[must_use]
    pub fn new(width: usize, height: usize, channels: usize) -> Self {
        Self {
            dimensions: Some((height, width)),
            channels,
            ..Default::default()
        }
    }

    /// Create specification for grayscale images
    #[must_use]
    pub fn grayscale(width: usize, height: usize) -> Self {
        Self {
            dimensions: Some((height, width)),
            channels: 1,
            color_space: ColorSpace::Grayscale,
            ..Default::default()
        }
    }

    /// Create specification for RGB images
    #[must_use]
    pub fn rgb(width: usize, height: usize) -> Self {
        Self {
            dimensions: Some((height, width)),
            channels: 3,
            color_space: ColorSpace::RGB,
            ..Default::default()
        }
    }

    /// Create specification for RGBA images with alpha channel
    #[must_use]
    pub fn rgba(width: usize, height: usize) -> Self {
        Self {
            dimensions: Some((height, width)),
            channels: 4,
            color_space: ColorSpace::RGB,
            ..Default::default()
        }
    }

    /// Create specification for high dynamic range images
    #[must_use]
    pub fn hdr(width: usize, height: usize) -> Self {
        Self {
            dimensions: Some((height, width)),
            channels: 3,
            dtype: ImageDataType::Float32,
            color_space: ColorSpace::RGB,
            bit_depth: 32,
            supported_formats: vec![ImageFormat::HDR, ImageFormat::EXR],
            ..Default::default()
        }
    }

    /// Validate an image against this specification
    pub fn validate(&self, image_data: &ImageData) -> Result<(), ValidationError> {
        // Check dimensions
        if let Some((expected_h, expected_w)) = self.dimensions {
            if image_data.height != expected_h || image_data.width != expected_w {
                return Err(ValidationError::DimensionMismatch {
                    expected: (expected_h, expected_w),
                    actual: (image_data.height, image_data.width),
                });
            }
        }

        // Check channels
        if image_data.channels != self.channels {
            return Err(ValidationError::ChannelMismatch {
                expected: self.channels,
                actual: image_data.channels,
            });
        }

        // Check data type
        if image_data.dtype != self.dtype {
            return Err(ValidationError::DataTypeMismatch {
                expected: self.dtype,
                actual: image_data.dtype,
            });
        }

        // Check color space
        if image_data.color_space != self.color_space {
            return Err(ValidationError::ColorSpaceMismatch {
                expected: self.color_space,
                actual: image_data.color_space,
            });
        }

        // Apply validation rules
        self.validation.validate(image_data)?;

        Ok(())
    }

    /// Check if a format is supported
    #[must_use]
    pub fn supports_format(&self, format: &ImageFormat) -> bool {
        self.supported_formats.contains(format)
    }

    /// Get memory requirements for an image with this specification
    #[must_use]
    pub fn memory_requirements(&self) -> usize {
        if let Some((height, width)) = self.dimensions {
            let bytes_per_pixel = self.channels * (self.bit_depth as usize / 8);
            height * width * bytes_per_pixel
        } else {
            0 // Unknown dimensions
        }
    }

    /// Create specification for object detection tasks
    #[must_use]
    pub fn object_detection(dimensions: (usize, usize)) -> Self {
        Self {
            dimensions: Some(dimensions),
            channels: 3,
            color_space: ColorSpace::RGB,
            dtype: ImageDataType::UInt8,
            bit_depth: 8,
            supported_formats: vec![ImageFormat::JPEG, ImageFormat::PNG],
            ..Default::default()
        }
    }

    /// Create specification for classification tasks
    #[must_use]
    pub fn classification(dimensions: (usize, usize)) -> Self {
        Self {
            dimensions: Some(dimensions),
            channels: 3,
            color_space: ColorSpace::RGB,
            dtype: ImageDataType::UInt8,
            bit_depth: 8,
            supported_formats: vec![ImageFormat::JPEG, ImageFormat::PNG],
            ..Default::default()
        }
    }

    /// Create specification for segmentation tasks
    #[must_use]
    pub fn segmentation(dimensions: (usize, usize)) -> Self {
        Self {
            dimensions: Some(dimensions),
            channels: 3,
            color_space: ColorSpace::RGB,
            dtype: ImageDataType::UInt8,
            bit_depth: 8,
            supported_formats: vec![ImageFormat::PNG, ImageFormat::TIFF],
            ..Default::default()
        }
    }
}

/// Image validation specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageValidationSpec {
    /// Minimum allowed dimensions
    pub min_dimensions: Option<(usize, usize)>,
    /// Maximum allowed dimensions
    pub max_dimensions: Option<(usize, usize)>,
    /// Allowed aspect ratios (width/height)
    pub allowed_aspect_ratios: Vec<f64>,
    /// Maximum file size in bytes
    pub max_file_size: Option<usize>,
    /// Minimum file size in bytes
    pub min_file_size: Option<usize>,
    /// Required metadata fields
    pub required_metadata: Vec<String>,
    /// Quality thresholds
    pub quality_thresholds: QualityThresholds,
}

impl Default for ImageValidationSpec {
    fn default() -> Self {
        Self {
            min_dimensions: Some((32, 32)),
            max_dimensions: Some((4096, 4096)),
            allowed_aspect_ratios: vec![], // Empty means any aspect ratio is allowed
            max_file_size: Some(10 * 1024 * 1024), // 10MB
            min_file_size: Some(1024),     // 1KB
            required_metadata: vec![],
            quality_thresholds: QualityThresholds::default(),
        }
    }
}

impl ImageValidationSpec {
    /// Validate image data against these specifications
    pub fn validate(&self, image_data: &ImageData) -> Result<(), ValidationError> {
        // Check minimum dimensions
        if let Some((min_h, min_w)) = self.min_dimensions {
            if image_data.height < min_h || image_data.width < min_w {
                return Err(ValidationError::DimensionTooSmall {
                    minimum: (min_h, min_w),
                    actual: (image_data.height, image_data.width),
                });
            }
        }

        // Check maximum dimensions
        if let Some((max_h, max_w)) = self.max_dimensions {
            if image_data.height > max_h || image_data.width > max_w {
                return Err(ValidationError::DimensionTooLarge {
                    maximum: (max_h, max_w),
                    actual: (image_data.height, image_data.width),
                });
            }
        }

        // Check aspect ratio
        if !self.allowed_aspect_ratios.is_empty() {
            let aspect_ratio = image_data.width as f64 / image_data.height as f64;
            let tolerance = 0.01; // 1% tolerance

            let aspect_ratio_valid = self
                .allowed_aspect_ratios
                .iter()
                .any(|&allowed| (aspect_ratio - allowed).abs() < tolerance);

            if !aspect_ratio_valid {
                return Err(ValidationError::InvalidAspectRatio {
                    allowed: self.allowed_aspect_ratios.clone(),
                    actual: aspect_ratio,
                });
            }
        }

        // Check file size if available
        if let Some(file_size) = image_data.file_size {
            if let Some(max_size) = self.max_file_size {
                if file_size > max_size {
                    return Err(ValidationError::FileTooLarge {
                        maximum: max_size,
                        actual: file_size,
                    });
                }
            }

            if let Some(min_size) = self.min_file_size {
                if file_size < min_size {
                    return Err(ValidationError::FileTooSmall {
                        minimum: min_size,
                        actual: file_size,
                    });
                }
            }
        }

        // Validate quality thresholds
        self.quality_thresholds.validate(image_data)?;

        Ok(())
    }
}

/// Quality thresholds for image validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityThresholds {
    /// Minimum brightness level (0.0-1.0)
    pub min_brightness: Option<f64>,
    /// Maximum brightness level (0.0-1.0)
    pub max_brightness: Option<f64>,
    /// Minimum contrast level (0.0-1.0)
    pub min_contrast: Option<f64>,
    /// Maximum blur metric (lower is sharper)
    pub max_blur: Option<f64>,
    /// Minimum signal-to-noise ratio
    pub min_snr: Option<f64>,
}

impl Default for QualityThresholds {
    fn default() -> Self {
        Self {
            min_brightness: Some(0.1),
            max_brightness: Some(0.9),
            min_contrast: Some(0.1),
            max_blur: Some(10.0),
            min_snr: Some(20.0),
        }
    }
}

impl QualityThresholds {
    /// Validate image quality against thresholds
    pub fn validate(&self, image_data: &ImageData) -> Result<(), ValidationError> {
        // Note: In a real implementation, these would calculate actual metrics
        // For now, we'll assume the image data includes quality metrics

        if let Some(brightness) = image_data.quality_metrics.get("brightness") {
            if let Some(min_brightness) = self.min_brightness {
                if *brightness < min_brightness {
                    return Err(ValidationError::QualityTooLow {
                        metric: "brightness".to_string(),
                        threshold: min_brightness,
                        actual: *brightness,
                    });
                }
            }

            if let Some(max_brightness) = self.max_brightness {
                if *brightness > max_brightness {
                    return Err(ValidationError::QualityTooHigh {
                        metric: "brightness".to_string(),
                        threshold: max_brightness,
                        actual: *brightness,
                    });
                }
            }
        }

        if let Some(contrast) = image_data.quality_metrics.get("contrast") {
            if let Some(min_contrast) = self.min_contrast {
                if *contrast < min_contrast {
                    return Err(ValidationError::QualityTooLow {
                        metric: "contrast".to_string(),
                        threshold: min_contrast,
                        actual: *contrast,
                    });
                }
            }
        }

        Ok(())
    }
}

/// Normalization specification for preprocessing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormalizationSpec {
    /// Mean values per channel
    pub mean: Array1<f32>,
    /// Standard deviation per channel
    pub std: Array1<f32>,
    /// Value range (min, max) for clipping
    pub range: (f32, f32),
    /// Whether to apply per-channel normalization
    pub per_channel: bool,
    /// Whether to apply global normalization
    pub global_normalization: bool,
}

impl Default for NormalizationSpec {
    fn default() -> Self {
        // ImageNet normalization values for RGB
        Self {
            mean: Array1::from(vec![0.485, 0.456, 0.406]),
            std: Array1::from(vec![0.229, 0.224, 0.225]),
            range: (0.0, 1.0),
            per_channel: true,
            global_normalization: false,
        }
    }
}

impl NormalizationSpec {
    /// Create normalization spec for grayscale images
    #[must_use]
    pub fn grayscale() -> Self {
        Self {
            mean: Array1::from(vec![0.5]),
            std: Array1::from(vec![0.5]),
            range: (0.0, 1.0),
            per_channel: true,
            global_normalization: false,
        }
    }

    /// Create custom normalization spec
    #[must_use]
    pub fn custom(mean: Vec<f32>, std: Vec<f32>, range: (f32, f32)) -> Self {
        Self {
            mean: Array1::from(mean),
            std: Array1::from(std),
            range,
            per_channel: true,
            global_normalization: false,
        }
    }

    /// Apply normalization to image data
    pub fn normalize(&self, image: &mut Array3<f32>) -> Result<(), ValidationError> {
        let (height, width, channels) = image.dim();

        if self.mean.len() != channels || self.std.len() != channels {
            return Err(ValidationError::NormalizationError(
                "Mean and std dimensions don't match image channels".to_string(),
            ));
        }

        for c in 0..channels {
            let mean_val = self.mean[c];
            let std_val = self.std[c];

            if std_val == 0.0 {
                return Err(ValidationError::NormalizationError(
                    "Standard deviation cannot be zero".to_string(),
                ));
            }

            for h in 0..height {
                for w in 0..width {
                    let pixel_value = (image[[h, w, c]] - mean_val) / std_val;
                    image[[h, w, c]] = pixel_value.clamp(self.range.0, self.range.1);
                }
            }
        }

        Ok(())
    }
}

/// Image data structure for validation and processing
#[derive(Debug, Clone)]
pub struct ImageData {
    /// Image height in pixels
    pub height: usize,
    /// Image width in pixels
    pub width: usize,
    /// Number of channels
    pub channels: usize,
    /// Data type
    pub dtype: ImageDataType,
    /// Color space
    pub color_space: ColorSpace,
    /// Optional file size in bytes
    pub file_size: Option<usize>,
    /// Quality metrics
    pub quality_metrics: HashMap<String, f64>,
    /// Image tensor data
    pub data: Array3<f32>,
    /// Optional metadata
    pub metadata: HashMap<String, String>,
}

impl ImageData {
    /// Create new image data with specified properties
    #[must_use]
    pub fn new(
        height: usize,
        width: usize,
        channels: usize,
        dtype: ImageDataType,
        color_space: ColorSpace,
        data: Array3<f32>,
    ) -> Self {
        Self {
            height,
            width,
            channels,
            dtype,
            color_space,
            file_size: None,
            quality_metrics: HashMap::new(),
            data,
            metadata: HashMap::new(),
        }
    }

    /// Get aspect ratio (width/height)
    #[must_use]
    pub fn aspect_ratio(&self) -> f64 {
        self.width as f64 / self.height as f64
    }

    /// Get total number of pixels
    #[must_use]
    pub fn pixel_count(&self) -> usize {
        self.height * self.width
    }

    /// Calculate memory footprint in bytes
    #[must_use]
    pub fn memory_footprint(&self) -> usize {
        let bytes_per_element = match self.dtype {
            ImageDataType::UInt8 => 1,
            ImageDataType::UInt16 => 2,
            ImageDataType::Float32 => 4,
            ImageDataType::Float64 => 8,
        };
        self.height * self.width * self.channels * bytes_per_element
    }
}

/// Validation errors for image specifications
#[derive(Debug, Clone, PartialEq)]
pub enum ValidationError {
    /// Image dimensions don't match expected values
    DimensionMismatch {
        expected: (usize, usize),
        actual: (usize, usize),
    },
    /// Number of channels doesn't match
    ChannelMismatch { expected: usize, actual: usize },
    /// Data type doesn't match
    DataTypeMismatch {
        expected: ImageDataType,
        actual: ImageDataType,
    },
    /// Color space doesn't match
    ColorSpaceMismatch {
        expected: ColorSpace,
        actual: ColorSpace,
    },
    /// Image dimensions are too small
    DimensionTooSmall {
        minimum: (usize, usize),
        actual: (usize, usize),
    },
    /// Image dimensions are too large
    DimensionTooLarge {
        maximum: (usize, usize),
        actual: (usize, usize),
    },
    /// Invalid aspect ratio
    InvalidAspectRatio { allowed: Vec<f64>, actual: f64 },
    /// File size too large
    FileTooLarge { maximum: usize, actual: usize },
    /// File size too small
    FileTooSmall { minimum: usize, actual: usize },
    /// Quality metric below threshold
    QualityTooLow {
        metric: String,
        threshold: f64,
        actual: f64,
    },
    /// Quality metric above threshold
    QualityTooHigh {
        metric: String,
        threshold: f64,
        actual: f64,
    },
    /// Normalization error
    NormalizationError(String),
}

impl std::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DimensionMismatch { expected, actual } => {
                write!(
                    f,
                    "Dimension mismatch: expected {expected:?}, got {actual:?}"
                )
            }
            Self::ChannelMismatch { expected, actual } => {
                write!(f, "Channel mismatch: expected {expected}, got {actual}")
            }
            Self::DataTypeMismatch { expected, actual } => {
                write!(
                    f,
                    "Data type mismatch: expected {expected:?}, got {actual:?}"
                )
            }
            Self::ColorSpaceMismatch { expected, actual } => {
                write!(
                    f,
                    "Color space mismatch: expected {expected:?}, got {actual:?}"
                )
            }
            Self::DimensionTooSmall { minimum, actual } => {
                write!(
                    f,
                    "Dimensions too small: minimum {minimum:?}, got {actual:?}"
                )
            }
            Self::DimensionTooLarge { maximum, actual } => {
                write!(
                    f,
                    "Dimensions too large: maximum {maximum:?}, got {actual:?}"
                )
            }
            Self::InvalidAspectRatio { allowed, actual } => {
                write!(f, "Invalid aspect ratio: allowed {allowed:?}, got {actual}")
            }
            Self::FileTooLarge { maximum, actual } => {
                write!(
                    f,
                    "File too large: maximum {maximum} bytes, got {actual} bytes"
                )
            }
            Self::FileTooSmall { minimum, actual } => {
                write!(
                    f,
                    "File too small: minimum {minimum} bytes, got {actual} bytes"
                )
            }
            Self::QualityTooLow {
                metric,
                threshold,
                actual,
            } => {
                write!(
                    f,
                    "Quality too low for {metric}: minimum {threshold}, got {actual}"
                )
            }
            Self::QualityTooHigh {
                metric,
                threshold,
                actual,
            } => {
                write!(
                    f,
                    "Quality too high for {metric}: maximum {threshold}, got {actual}"
                )
            }
            Self::NormalizationError(msg) => {
                write!(f, "Normalization error: {msg}")
            }
        }
    }
}

impl std::error::Error for ValidationError {}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array3;

    #[test]
    fn test_image_specification_creation() {
        let spec = ImageSpecification::rgb(640, 480);
        assert_eq!(spec.dimensions, Some((480, 640)));
        assert_eq!(spec.channels, 3);
        assert_eq!(spec.color_space, ColorSpace::RGB);

        let spec = ImageSpecification::grayscale(224, 224);
        assert_eq!(spec.channels, 1);
        assert_eq!(spec.color_space, ColorSpace::Grayscale);
    }

    #[test]
    fn test_image_validation() {
        let spec = ImageSpecification::rgb(224, 224);
        let data = Array3::<f32>::zeros((224, 224, 3));
        let image = ImageData::new(224, 224, 3, ImageDataType::UInt8, ColorSpace::RGB, data);

        assert!(spec.validate(&image).is_ok());

        // Test dimension mismatch
        let data = Array3::<f32>::zeros((256, 256, 3));
        let image = ImageData::new(256, 256, 3, ImageDataType::UInt8, ColorSpace::RGB, data);
        assert!(spec.validate(&image).is_err());
    }

    #[test]
    fn test_normalization_spec() {
        let norm_spec = NormalizationSpec::grayscale();
        assert_eq!(norm_spec.mean.len(), 1);
        assert_eq!(norm_spec.std.len(), 1);

        let norm_spec = NormalizationSpec::custom(
            vec![0.485, 0.456, 0.406],
            vec![0.229, 0.224, 0.225],
            (0.0, 1.0),
        );
        assert_eq!(norm_spec.mean.len(), 3);
        assert_eq!(norm_spec.std.len(), 3);
    }

    #[test]
    fn test_image_data_properties() {
        let data = Array3::<f32>::zeros((480, 640, 3));
        let image = ImageData::new(480, 640, 3, ImageDataType::UInt8, ColorSpace::RGB, data);

        assert_eq!(image.aspect_ratio(), 640.0 / 480.0);
        assert_eq!(image.pixel_count(), 480 * 640);
        assert_eq!(image.memory_footprint(), 480 * 640 * 3 * 1); // UInt8 = 1 byte
    }

    #[test]
    fn test_validation_error_display() {
        let error = ValidationError::DimensionMismatch {
            expected: (224, 224),
            actual: (256, 256),
        };
        let error_str = error.to_string();
        assert!(error_str.contains("Dimension mismatch"));
        assert!(error_str.contains("224"));
        assert!(error_str.contains("256"));
    }
}
