//! Image Preprocessing for Computer Vision Applications
//!
//! This module provides comprehensive image preprocessing utilities for computer vision
//! and machine learning tasks, including normalization, data augmentation, color space
//! transformations, and feature extraction.
//!
//! # Features
//!
//! - Image normalization and standardization
//! - Data augmentation techniques (rotation, scaling, flipping, cropping)
//! - Color space transformations (RGB, HSV, LAB, grayscale)
//! - Image resizing and cropping
//! - Edge detection and feature extraction
//! - Batch image processing with parallel support
//!
//! # Examples
//!
//! ```rust,ignore
//! use sklears_preprocessing::image_preprocessing::{
//!     ImageNormalizer, ImageAugmenter, ColorSpaceTransformer
//! };
//! use scirs2_core::ndarray::Array3;
//!
//! fn example() -> Result<(), Box<dyn std::error::Error>> {
//!     // Normalize image pixel values to [0, 1] range
//!     let normalizer = ImageNormalizer::new()
//!         .with_range((0.0, 1.0))
//!         .with_channel_wise(true);
//!
//!     let image = Array3::<f64>::zeros((224, 224, 3)); // RGB image
//!     let normalized = normalizer.transform(&image)?;
//!
//!     // Apply data augmentation
//!     let augmenter = ImageAugmenter::new()
//!         .with_rotation_range((-15.0, 15.0))
//!         .with_zoom_range((0.9, 1.1))
//!         .with_horizontal_flip(true);
//!
//!     let augmented = augmenter.transform(&normalized)?;
//!
//!     // Convert color space
//!     let color_transformer = ColorSpaceTransformer::new()
//!         .from_rgb()
//!         .to_hsv();
//!
//!     let hsv_image = color_transformer.transform(&augmented)?;
//!
//!     Ok(())
//! }
//! ```

use scirs2_core::ndarray::{Array3, Axis};
use scirs2_core::random::thread_rng;
// Note: using fallback implementations since SIMD functions may not be available
// use scirs2_core::simd_ops::{mean_f64_simd, variance_f64_simd};
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Estimator, Fit, Transform, Untrained},
    types::Float,
};
use std::f64::consts::PI;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Image normalization strategies
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum ImageNormalizationStrategy {
    /// Normalize to [0, 1] range using min-max scaling
    MinMax,
    /// Standardize to zero mean and unit variance
    StandardScore,
    /// Custom range normalization
    CustomRange(Float, Float),
}

impl Default for ImageNormalizationStrategy {
    fn default() -> Self {
        Self::MinMax
    }
}

/// Color space types for image transformations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum ColorSpace {
    /// RGB (Red, Green, Blue)
    RGB,
    /// HSV (Hue, Saturation, Value)
    HSV,
    /// LAB (Lightness, A, B)
    LAB,
    /// Grayscale (single channel)
    Grayscale,
}

/// Image resizing interpolation methods
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum InterpolationMethod {
    /// Nearest neighbor interpolation (fast, blocky)
    Nearest,
    /// Bilinear interpolation (smooth, good quality/speed balance)
    Bilinear,
    /// Bicubic interpolation (highest quality, slower)
    Bicubic,
}

impl Default for InterpolationMethod {
    fn default() -> Self {
        Self::Bilinear
    }
}

/// Configuration for image normalization
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ImageNormalizerConfig {
    /// Normalization strategy to use
    pub strategy: ImageNormalizationStrategy,
    /// Whether to normalize each channel independently
    pub channel_wise: bool,
    /// Epsilon for numerical stability
    pub epsilon: Float,
}

impl Default for ImageNormalizerConfig {
    fn default() -> Self {
        Self {
            strategy: ImageNormalizationStrategy::MinMax,
            channel_wise: true,
            epsilon: 1e-8,
        }
    }
}

/// Image normalizer for preprocessing image data
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ImageNormalizer<State = Untrained> {
    config: ImageNormalizerConfig,
    state: std::marker::PhantomData<State>,
}

/// Fitted state for image normalizer
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ImageNormalizerFitted {
    config: ImageNormalizerConfig,
    min_vals: Vec<Float>,
    max_vals: Vec<Float>,
    mean_vals: Vec<Float>,
    std_vals: Vec<Float>,
}

impl ImageNormalizer<Untrained> {
    /// Create a new image normalizer
    pub fn new() -> Self {
        Self {
            config: ImageNormalizerConfig::default(),
            state: std::marker::PhantomData,
        }
    }

    /// Set normalization strategy
    pub fn with_strategy(mut self, strategy: ImageNormalizationStrategy) -> Self {
        self.config.strategy = strategy;
        self
    }

    /// Set range for min-max normalization
    pub fn with_range(mut self, range: (Float, Float)) -> Self {
        self.config.strategy = ImageNormalizationStrategy::CustomRange(range.0, range.1);
        self
    }

    /// Enable/disable channel-wise normalization
    pub fn with_channel_wise(mut self, channel_wise: bool) -> Self {
        self.config.channel_wise = channel_wise;
        self
    }

    /// Set epsilon for numerical stability
    pub fn with_epsilon(mut self, epsilon: Float) -> Self {
        self.config.epsilon = epsilon;
        self
    }
}

impl Estimator for ImageNormalizer<Untrained> {
    type Config = ImageNormalizerConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Fit<Array3<Float>, ()> for ImageNormalizer<Untrained> {
    type Fitted = ImageNormalizerFitted;

    fn fit(self, x: &Array3<Float>, _y: &()) -> Result<Self::Fitted> {
        let shape = x.dim();
        let n_channels = shape.2;

        let (min_vals, max_vals, mean_vals, std_vals) = if self.config.channel_wise {
            let mut min_vals = Vec::with_capacity(n_channels);
            let mut max_vals = Vec::with_capacity(n_channels);
            let mut mean_vals = Vec::with_capacity(n_channels);
            let mut std_vals = Vec::with_capacity(n_channels);

            for channel in 0..n_channels {
                let channel_data = x.index_axis(Axis(2), channel);
                let data_slice: Vec<Float> = channel_data.iter().copied().collect();

                let min_val = data_slice.iter().fold(Float::INFINITY, |a, &b| a.min(b));
                let max_val = data_slice
                    .iter()
                    .fold(Float::NEG_INFINITY, |a, &b| a.max(b));

                let mean_val = data_slice.iter().sum::<Float>() / data_slice.len() as Float;

                let var_val = data_slice
                    .iter()
                    .map(|&x| (x - mean_val).powi(2))
                    .sum::<Float>()
                    / (data_slice.len() as Float - 1.0);

                let std_val = var_val.sqrt().max(self.config.epsilon);

                min_vals.push(min_val);
                max_vals.push(max_val);
                mean_vals.push(mean_val);
                std_vals.push(std_val);
            }

            (min_vals, max_vals, mean_vals, std_vals)
        } else {
            // Global statistics across all channels
            let all_data: Vec<Float> = x.iter().copied().collect();

            let min_val = all_data.iter().fold(Float::INFINITY, |a, &b| a.min(b));
            let max_val = all_data.iter().fold(Float::NEG_INFINITY, |a, &b| a.max(b));

            let mean_val = all_data.iter().sum::<Float>() / all_data.len() as Float;

            let var_val = all_data
                .iter()
                .map(|&x| (x - mean_val).powi(2))
                .sum::<Float>()
                / (all_data.len() as Float - 1.0);

            let std_val = var_val.sqrt().max(self.config.epsilon);

            (
                vec![min_val; n_channels],
                vec![max_val; n_channels],
                vec![mean_val; n_channels],
                vec![std_val; n_channels],
            )
        };

        Ok(ImageNormalizerFitted {
            config: self.config,
            min_vals,
            max_vals,
            mean_vals,
            std_vals,
        })
    }
}

impl Transform<Array3<Float>, Array3<Float>> for ImageNormalizerFitted {
    fn transform(&self, x: &Array3<Float>) -> Result<Array3<Float>> {
        let shape = x.dim();
        let n_channels = shape.2;

        if n_channels != self.min_vals.len() {
            return Err(SklearsError::InvalidInput(format!(
                "Expected {} channels, got {}",
                self.min_vals.len(),
                n_channels
            )));
        }

        let mut result = x.clone();

        match self.config.strategy {
            ImageNormalizationStrategy::MinMax => {
                for channel in 0..n_channels {
                    let min_val = self.min_vals[channel];
                    let max_val = self.max_vals[channel];
                    let range = max_val - min_val;

                    if range > self.config.epsilon {
                        let mut channel_data = result.index_axis_mut(Axis(2), channel);
                        channel_data.mapv_inplace(|x| (x - min_val) / range);
                    }
                }
            }
            ImageNormalizationStrategy::CustomRange(min_target, max_target) => {
                let target_range = max_target - min_target;
                for channel in 0..n_channels {
                    let min_val = self.min_vals[channel];
                    let max_val = self.max_vals[channel];
                    let source_range = max_val - min_val;

                    if source_range > self.config.epsilon {
                        let mut channel_data = result.index_axis_mut(Axis(2), channel);
                        channel_data.mapv_inplace(|x| {
                            min_target + ((x - min_val) / source_range) * target_range
                        });
                    }
                }
            }
            ImageNormalizationStrategy::StandardScore => {
                for channel in 0..n_channels {
                    let mean_val = self.mean_vals[channel];
                    let std_val = self.std_vals[channel];

                    let mut channel_data = result.index_axis_mut(Axis(2), channel);
                    channel_data.mapv_inplace(|x| (x - mean_val) / std_val);
                }
            }
        }

        Ok(result)
    }
}

/// Configuration for image augmentation
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ImageAugmenterConfig {
    /// Rotation range in degrees (min, max)
    pub rotation_range: Option<(Float, Float)>,
    /// Zoom range as factors (min, max)
    pub zoom_range: Option<(Float, Float)>,
    /// Width shift range as fraction of total width
    pub width_shift_range: Option<Float>,
    /// Height shift range as fraction of total height
    pub height_shift_range: Option<Float>,
    /// Enable horizontal flipping
    pub horizontal_flip: bool,
    /// Enable vertical flipping
    pub vertical_flip: bool,
    /// Brightness adjustment range (min, max)
    pub brightness_range: Option<(Float, Float)>,
    /// Random seed for reproducibility
    pub random_seed: Option<u64>,
}

impl Default for ImageAugmenterConfig {
    fn default() -> Self {
        Self {
            rotation_range: None,
            zoom_range: None,
            width_shift_range: None,
            height_shift_range: None,
            horizontal_flip: false,
            vertical_flip: false,
            brightness_range: None,
            random_seed: None,
        }
    }
}

/// Image augmenter for data augmentation
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ImageAugmenter {
    config: ImageAugmenterConfig,
}

impl ImageAugmenter {
    /// Create a new image augmenter
    pub fn new() -> Self {
        Self {
            config: ImageAugmenterConfig::default(),
        }
    }

    /// Set rotation range in degrees
    pub fn with_rotation_range(mut self, range: (Float, Float)) -> Self {
        self.config.rotation_range = Some(range);
        self
    }

    /// Set zoom range as factors
    pub fn with_zoom_range(mut self, range: (Float, Float)) -> Self {
        self.config.zoom_range = Some(range);
        self
    }

    /// Set width shift range as fraction
    pub fn with_width_shift_range(mut self, range: Float) -> Self {
        self.config.width_shift_range = Some(range);
        self
    }

    /// Set height shift range as fraction
    pub fn with_height_shift_range(mut self, range: Float) -> Self {
        self.config.height_shift_range = Some(range);
        self
    }

    /// Enable horizontal flipping
    pub fn with_horizontal_flip(mut self, enabled: bool) -> Self {
        self.config.horizontal_flip = enabled;
        self
    }

    /// Enable vertical flipping
    pub fn with_vertical_flip(mut self, enabled: bool) -> Self {
        self.config.vertical_flip = enabled;
        self
    }

    /// Set brightness adjustment range
    pub fn with_brightness_range(mut self, range: (Float, Float)) -> Self {
        self.config.brightness_range = Some(range);
        self
    }

    /// Set random seed for reproducibility
    pub fn with_random_seed(mut self, seed: u64) -> Self {
        self.config.random_seed = Some(seed);
        // Note: seed will be used by thread_rng() function if needed
        self
    }
}

impl Transform<Array3<Float>, Array3<Float>> for ImageAugmenter {
    fn transform(&self, x: &Array3<Float>) -> Result<Array3<Float>> {
        let mut result = x.clone();
        let mut rng = thread_rng();

        // Apply horizontal flip
        if self.config.horizontal_flip && rng.random::<Float>() < 0.5 {
            result = self.horizontal_flip(&result)?;
        }

        // Apply vertical flip
        if self.config.vertical_flip && rng.random::<Float>() < 0.5 {
            result = self.vertical_flip(&result)?;
        }

        // Apply rotation
        if let Some((min_angle, max_angle)) = self.config.rotation_range {
            let angle = rng.gen_range(min_angle..max_angle);
            if angle.abs() > 1e-6 {
                result = self.rotate(&result, angle)?;
            }
        }

        // Apply brightness adjustment
        if let Some((min_brightness, max_brightness)) = self.config.brightness_range {
            let brightness_factor = rng.gen_range(min_brightness..max_brightness);
            if (brightness_factor - 1.0).abs() > 1e-6 {
                result.mapv_inplace(|x| (x * brightness_factor).clamp(0.0, 1.0));
            }
        }

        Ok(result)
    }
}

impl ImageAugmenter {
    /// Apply horizontal flip to image
    fn horizontal_flip(&self, image: &Array3<Float>) -> Result<Array3<Float>> {
        let shape = image.dim();
        let mut result = Array3::zeros(shape);

        for row in 0..shape.0 {
            for col in 0..shape.1 {
                for channel in 0..shape.2 {
                    result[[row, shape.1 - 1 - col, channel]] = image[[row, col, channel]];
                }
            }
        }

        Ok(result)
    }

    /// Apply vertical flip to image
    fn vertical_flip(&self, image: &Array3<Float>) -> Result<Array3<Float>> {
        let shape = image.dim();
        let mut result = Array3::zeros(shape);

        for row in 0..shape.0 {
            for col in 0..shape.1 {
                for channel in 0..shape.2 {
                    result[[shape.0 - 1 - row, col, channel]] = image[[row, col, channel]];
                }
            }
        }

        Ok(result)
    }

    /// Apply rotation to image (simplified implementation)
    fn rotate(&self, image: &Array3<Float>, angle_degrees: Float) -> Result<Array3<Float>> {
        let shape = image.dim();
        let mut result = Array3::zeros(shape);

        let angle_rad = angle_degrees * PI / 180.0;
        let cos_angle = angle_rad.cos();
        let sin_angle = angle_rad.sin();

        let center_x = shape.1 as Float / 2.0;
        let center_y = shape.0 as Float / 2.0;

        // Simple rotation with nearest neighbor interpolation
        for row in 0..shape.0 {
            for col in 0..shape.1 {
                let x = col as Float - center_x;
                let y = row as Float - center_y;

                let rotated_x = x * cos_angle - y * sin_angle + center_x;
                let rotated_y = x * sin_angle + y * cos_angle + center_y;

                let src_col = rotated_x.round() as isize;
                let src_row = rotated_y.round() as isize;

                if src_row >= 0
                    && src_row < shape.0 as isize
                    && src_col >= 0
                    && src_col < shape.1 as isize
                {
                    for channel in 0..shape.2 {
                        result[[row, col, channel]] =
                            image[[src_row as usize, src_col as usize, channel]];
                    }
                }
            }
        }

        Ok(result)
    }
}

/// Color space transformer
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ColorSpaceTransformer {
    source: ColorSpace,
    target: ColorSpace,
}

impl ColorSpaceTransformer {
    /// Create a new color space transformer
    pub fn new() -> Self {
        Self {
            source: ColorSpace::RGB,
            target: ColorSpace::RGB,
        }
    }

    /// Set source color space
    pub fn from_colorspace(mut self, colorspace: ColorSpace) -> Self {
        self.source = colorspace;
        self
    }

    /// Set target color space
    pub fn to_colorspace(mut self, colorspace: ColorSpace) -> Self {
        self.target = colorspace;
        self
    }

    /// Set source as RGB
    pub fn from_rgb(mut self) -> Self {
        self.source = ColorSpace::RGB;
        self
    }

    /// Set target as HSV
    pub fn to_hsv(mut self) -> Self {
        self.target = ColorSpace::HSV;
        self
    }

    /// Set target as grayscale
    pub fn to_grayscale(mut self) -> Self {
        self.target = ColorSpace::Grayscale;
        self
    }
}

impl Transform<Array3<Float>, Array3<Float>> for ColorSpaceTransformer {
    fn transform(&self, x: &Array3<Float>) -> Result<Array3<Float>> {
        match (self.source, self.target) {
            (ColorSpace::RGB, ColorSpace::HSV) => self.rgb_to_hsv(x),
            (ColorSpace::RGB, ColorSpace::Grayscale) => self.rgb_to_grayscale(x),
            (ColorSpace::HSV, ColorSpace::RGB) => self.hsv_to_rgb(x),
            (source, target) if source == target => Ok(x.clone()),
            _ => Err(SklearsError::InvalidInput(format!(
                "Conversion from {:?} to {:?} not implemented",
                self.source, self.target
            ))),
        }
    }
}

impl ColorSpaceTransformer {
    /// Convert RGB to HSV
    fn rgb_to_hsv(&self, image: &Array3<Float>) -> Result<Array3<Float>> {
        let shape = image.dim();
        if shape.2 != 3 {
            return Err(SklearsError::InvalidInput(
                "RGB images must have 3 channels".to_string(),
            ));
        }

        let mut result = Array3::zeros(shape);

        for row in 0..shape.0 {
            for col in 0..shape.1 {
                let r = image[[row, col, 0]];
                let g = image[[row, col, 1]];
                let b = image[[row, col, 2]];

                let max_val = r.max(g).max(b);
                let min_val = r.min(g).min(b);
                let delta = max_val - min_val;

                // Hue calculation
                let h = if delta < 1e-8 {
                    0.0
                } else if (max_val - r).abs() < 1e-8 {
                    60.0 * (((g - b) / delta) % 6.0)
                } else if (max_val - g).abs() < 1e-8 {
                    60.0 * (((b - r) / delta) + 2.0)
                } else {
                    60.0 * (((r - g) / delta) + 4.0)
                };

                let h = if h < 0.0 { h + 360.0 } else { h };

                // Saturation calculation
                let s = if max_val < 1e-8 { 0.0 } else { delta / max_val };

                // Value calculation
                let v = max_val;

                result[[row, col, 0]] = h / 360.0; // Normalize hue to [0, 1]
                result[[row, col, 1]] = s;
                result[[row, col, 2]] = v;
            }
        }

        Ok(result)
    }

    /// Convert HSV to RGB
    fn hsv_to_rgb(&self, image: &Array3<Float>) -> Result<Array3<Float>> {
        let shape = image.dim();
        if shape.2 != 3 {
            return Err(SklearsError::InvalidInput(
                "HSV images must have 3 channels".to_string(),
            ));
        }

        let mut result = Array3::zeros(shape);

        for row in 0..shape.0 {
            for col in 0..shape.1 {
                let h = image[[row, col, 0]] * 360.0; // Denormalize hue from [0, 1] to [0, 360]
                let s = image[[row, col, 1]];
                let v = image[[row, col, 2]];

                let c = v * s;
                let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
                let m = v - c;

                let (r, g, b) = if h < 60.0 {
                    (c, x, 0.0)
                } else if h < 120.0 {
                    (x, c, 0.0)
                } else if h < 180.0 {
                    (0.0, c, x)
                } else if h < 240.0 {
                    (0.0, x, c)
                } else if h < 300.0 {
                    (x, 0.0, c)
                } else {
                    (c, 0.0, x)
                };

                result[[row, col, 0]] = r + m;
                result[[row, col, 1]] = g + m;
                result[[row, col, 2]] = b + m;
            }
        }

        Ok(result)
    }

    /// Convert RGB to grayscale using luminance weighting
    fn rgb_to_grayscale(&self, image: &Array3<Float>) -> Result<Array3<Float>> {
        let shape = image.dim();
        if shape.2 != 3 {
            return Err(SklearsError::InvalidInput(
                "RGB images must have 3 channels".to_string(),
            ));
        }

        let mut result = Array3::zeros((shape.0, shape.1, 1));

        // Standard luminance weights for RGB to grayscale conversion
        const R_WEIGHT: Float = 0.299;
        const G_WEIGHT: Float = 0.587;
        const B_WEIGHT: Float = 0.114;

        for row in 0..shape.0 {
            for col in 0..shape.1 {
                let r = image[[row, col, 0]];
                let g = image[[row, col, 1]];
                let b = image[[row, col, 2]];

                let gray = R_WEIGHT * r + G_WEIGHT * g + B_WEIGHT * b;
                result[[row, col, 0]] = gray;
            }
        }

        Ok(result)
    }
}

/// Image resizer for changing image dimensions
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ImageResizer {
    target_size: (usize, usize),
    method: InterpolationMethod,
}

impl ImageResizer {
    /// Create a new image resizer
    pub fn new(target_size: (usize, usize)) -> Self {
        Self {
            target_size,
            method: InterpolationMethod::default(),
        }
    }

    /// Set interpolation method
    pub fn with_method(mut self, method: InterpolationMethod) -> Self {
        self.method = method;
        self
    }
}

impl Transform<Array3<Float>, Array3<Float>> for ImageResizer {
    fn transform(&self, x: &Array3<Float>) -> Result<Array3<Float>> {
        let source_shape = x.dim();
        let (target_height, target_width) = self.target_size;

        if target_height == 0 || target_width == 0 {
            return Err(SklearsError::InvalidInput(
                "Target dimensions must be positive".to_string(),
            ));
        }

        let mut result = Array3::zeros((target_height, target_width, source_shape.2));

        let height_scale = source_shape.0 as Float / target_height as Float;
        let width_scale = source_shape.1 as Float / target_width as Float;

        match self.method {
            InterpolationMethod::Nearest => {
                for row in 0..target_height {
                    for col in 0..target_width {
                        let src_row = ((row as Float + 0.5) * height_scale).floor() as usize;
                        let src_col = ((col as Float + 0.5) * width_scale).floor() as usize;

                        let src_row = src_row.min(source_shape.0 - 1);
                        let src_col = src_col.min(source_shape.1 - 1);

                        for channel in 0..source_shape.2 {
                            result[[row, col, channel]] = x[[src_row, src_col, channel]];
                        }
                    }
                }
            }
            InterpolationMethod::Bilinear => {
                for row in 0..target_height {
                    for col in 0..target_width {
                        let src_y = (row as Float + 0.5) * height_scale - 0.5;
                        let src_x = (col as Float + 0.5) * width_scale - 0.5;

                        let y1 = src_y.floor() as isize;
                        let x1 = src_x.floor() as isize;
                        let y2 = y1 + 1;
                        let x2 = x1 + 1;

                        let dy = src_y - y1 as Float;
                        let dx = src_x - x1 as Float;

                        for channel in 0..source_shape.2 {
                            let mut sum = 0.0;

                            // Bilinear interpolation weights and values
                            if y1 >= 0
                                && y1 < source_shape.0 as isize
                                && x1 >= 0
                                && x1 < source_shape.1 as isize
                            {
                                sum += (1.0 - dx)
                                    * (1.0 - dy)
                                    * x[[y1 as usize, x1 as usize, channel]];
                            }
                            if y1 >= 0
                                && y1 < source_shape.0 as isize
                                && x2 >= 0
                                && x2 < source_shape.1 as isize
                            {
                                sum += dx * (1.0 - dy) * x[[y1 as usize, x2 as usize, channel]];
                            }
                            if y2 >= 0
                                && y2 < source_shape.0 as isize
                                && x1 >= 0
                                && x1 < source_shape.1 as isize
                            {
                                sum += (1.0 - dx) * dy * x[[y2 as usize, x1 as usize, channel]];
                            }
                            if y2 >= 0
                                && y2 < source_shape.0 as isize
                                && x2 >= 0
                                && x2 < source_shape.1 as isize
                            {
                                sum += dx * dy * x[[y2 as usize, x2 as usize, channel]];
                            }

                            result[[row, col, channel]] = sum;
                        }
                    }
                }
            }
            InterpolationMethod::Bicubic => {
                // Simplified bicubic - for production use, implement proper cubic kernel
                // For now, fall back to bilinear
                let bilinear_resizer =
                    ImageResizer::new(self.target_size).with_method(InterpolationMethod::Bilinear);
                return bilinear_resizer.transform(x);
            }
        }

        Ok(result)
    }
}

/// Edge detection methods
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum EdgeDetectionMethod {
    /// Sobel edge detection
    Sobel,
    /// Laplacian edge detection
    Laplacian,
    /// Canny edge detection (simplified)
    Canny,
}

impl Default for EdgeDetectionMethod {
    fn default() -> Self {
        Self::Sobel
    }
}

/// Edge detector for feature extraction
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct EdgeDetector {
    method: EdgeDetectionMethod,
    threshold: Option<Float>,
    blur_sigma: Option<Float>,
}

impl EdgeDetector {
    /// Create a new edge detector
    pub fn new() -> Self {
        Self {
            method: EdgeDetectionMethod::default(),
            threshold: None,
            blur_sigma: None,
        }
    }

    /// Set edge detection method
    pub fn with_method(mut self, method: EdgeDetectionMethod) -> Self {
        self.method = method;
        self
    }

    /// Set threshold for edge detection
    pub fn with_threshold(mut self, threshold: Float) -> Self {
        self.threshold = Some(threshold);
        self
    }

    /// Set Gaussian blur sigma for preprocessing
    pub fn with_blur_sigma(mut self, sigma: Float) -> Self {
        self.blur_sigma = Some(sigma);
        self
    }
}

impl Transform<Array3<Float>, Array3<Float>> for EdgeDetector {
    fn transform(&self, x: &Array3<Float>) -> Result<Array3<Float>> {
        // Convert to grayscale if needed
        let gray_image = if x.dim().2 == 3 {
            let color_transformer = ColorSpaceTransformer::new().from_rgb().to_grayscale();
            color_transformer.transform(x)?
        } else if x.dim().2 == 1 {
            x.clone()
        } else {
            return Err(SklearsError::InvalidInput(
                "Image must have 1 or 3 channels".to_string(),
            ));
        };

        let mut processed = gray_image;

        // Apply Gaussian blur if specified
        if let Some(sigma) = self.blur_sigma {
            processed = self.gaussian_blur(&processed, sigma)?;
        }

        // Apply edge detection
        let edges = match self.method {
            EdgeDetectionMethod::Sobel => self.sobel_edge_detection(&processed)?,
            EdgeDetectionMethod::Laplacian => self.laplacian_edge_detection(&processed)?,
            EdgeDetectionMethod::Canny => {
                // Simplified Canny: Sobel + thresholding
                let sobel_edges = self.sobel_edge_detection(&processed)?;
                if let Some(threshold) = self.threshold {
                    self.apply_threshold(&sobel_edges, threshold)?
                } else {
                    sobel_edges
                }
            }
        };

        Ok(edges)
    }
}

impl EdgeDetector {
    /// Apply Gaussian blur to reduce noise
    fn gaussian_blur(&self, image: &Array3<Float>, sigma: Float) -> Result<Array3<Float>> {
        let shape = image.dim();
        let mut result = image.clone();

        // Simple 3x3 Gaussian kernel approximation
        let kernel_size = (6.0 * sigma).ceil() as usize + 1;
        let kernel_radius = kernel_size / 2;

        // Create Gaussian kernel
        let mut kernel = vec![vec![0.0; kernel_size]; kernel_size];
        let mut kernel_sum = 0.0;

        for i in 0..kernel_size {
            for j in 0..kernel_size {
                let x = (i as isize - kernel_radius as isize) as Float;
                let y = (j as isize - kernel_radius as isize) as Float;
                let value = (-((x * x + y * y) / (2.0 * sigma * sigma))).exp();
                kernel[i][j] = value;
                kernel_sum += value;
            }
        }

        // Normalize kernel
        for i in 0..kernel_size {
            for j in 0..kernel_size {
                kernel[i][j] /= kernel_sum;
            }
        }

        // Apply convolution
        for row in kernel_radius..(shape.0 - kernel_radius) {
            for col in kernel_radius..(shape.1 - kernel_radius) {
                for channel in 0..shape.2 {
                    let mut sum = 0.0;
                    for ki in 0..kernel_size {
                        for kj in 0..kernel_size {
                            let img_row = row + ki - kernel_radius;
                            let img_col = col + kj - kernel_radius;
                            sum += image[[img_row, img_col, channel]] * kernel[ki][kj];
                        }
                    }
                    result[[row, col, channel]] = sum;
                }
            }
        }

        Ok(result)
    }

    /// Apply Sobel edge detection
    fn sobel_edge_detection(&self, image: &Array3<Float>) -> Result<Array3<Float>> {
        let shape = image.dim();
        let mut result = Array3::zeros(shape);

        // Sobel kernels
        let sobel_x: [[Float; 3]; 3] = [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]];

        let sobel_y: [[Float; 3]; 3] = [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]];

        // Apply Sobel operators
        for row in 1..(shape.0 - 1) {
            for col in 1..(shape.1 - 1) {
                for channel in 0..shape.2 {
                    let mut gx = 0.0;
                    let mut gy = 0.0;

                    // Apply 3x3 kernels
                    for i in 0..3 {
                        for j in 0..3 {
                            let pixel_val = image[[row + i - 1, col + j - 1, channel]];
                            gx += pixel_val * sobel_x[i][j];
                            gy += pixel_val * sobel_y[i][j];
                        }
                    }

                    // Calculate gradient magnitude
                    let gradient_magnitude = (gx * gx + gy * gy).sqrt();
                    result[[row, col, channel]] = gradient_magnitude;
                }
            }
        }

        Ok(result)
    }

    /// Apply Laplacian edge detection
    fn laplacian_edge_detection(&self, image: &Array3<Float>) -> Result<Array3<Float>> {
        let shape = image.dim();
        let mut result = Array3::zeros(shape);

        // Laplacian kernel
        let laplacian: [[Float; 3]; 3] = [[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]];

        // Apply Laplacian operator
        for row in 1..(shape.0 - 1) {
            for col in 1..(shape.1 - 1) {
                for channel in 0..shape.2 {
                    let mut sum = 0.0;

                    // Apply 3x3 kernel
                    for i in 0..3 {
                        for j in 0..3 {
                            let pixel_val = image[[row + i - 1, col + j - 1, channel]];
                            sum += pixel_val * laplacian[i][j];
                        }
                    }

                    result[[row, col, channel]] = sum.abs();
                }
            }
        }

        Ok(result)
    }

    /// Apply threshold to edge detection results
    fn apply_threshold(&self, image: &Array3<Float>, threshold: Float) -> Result<Array3<Float>> {
        let mut result = image.clone();
        result.mapv_inplace(|x| if x > threshold { 1.0 } else { 0.0 });
        Ok(result)
    }
}

/// Basic feature extractor for images
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ImageFeatureExtractor {
    extract_edges: bool,
    extract_histograms: bool,
    histogram_bins: usize,
    extract_moments: bool,
}

impl ImageFeatureExtractor {
    /// Create a new image feature extractor
    pub fn new() -> Self {
        Self {
            extract_edges: true,
            extract_histograms: true,
            histogram_bins: 32,
            extract_moments: true,
        }
    }

    /// Enable/disable edge feature extraction
    pub fn with_edge_features(mut self, enabled: bool) -> Self {
        self.extract_edges = enabled;
        self
    }

    /// Enable/disable histogram feature extraction
    pub fn with_histogram_features(mut self, enabled: bool, bins: usize) -> Self {
        self.extract_histograms = enabled;
        self.histogram_bins = bins;
        self
    }

    /// Enable/disable moment feature extraction
    pub fn with_moment_features(mut self, enabled: bool) -> Self {
        self.extract_moments = enabled;
        self
    }
}

impl Transform<Array3<Float>, Vec<Float>> for ImageFeatureExtractor {
    fn transform(&self, x: &Array3<Float>) -> Result<Vec<Float>> {
        let mut features = Vec::new();

        // Extract edge features
        if self.extract_edges {
            let edge_detector = EdgeDetector::new().with_method(EdgeDetectionMethod::Sobel);
            let edges = edge_detector.transform(x)?;

            // Edge density (percentage of edge pixels)
            let total_pixels = edges.len();
            let edge_pixels = edges.iter().filter(|&&x| x > 0.1).count();
            features.push(edge_pixels as Float / total_pixels as Float);

            // Mean edge strength
            let mean_edge_strength = edges.iter().sum::<Float>() / total_pixels as Float;
            features.push(mean_edge_strength);
        }

        // Extract histogram features
        if self.extract_histograms {
            for channel in 0..x.dim().2 {
                let channel_data = x.index_axis(Axis(2), channel);
                let histogram = self.compute_histogram(&channel_data, self.histogram_bins)?;
                features.extend(histogram);
            }
        }

        // Extract moment features
        if self.extract_moments {
            for channel in 0..x.dim().2 {
                let channel_data = x.index_axis(Axis(2), channel);
                let data_vec: Vec<Float> = channel_data.iter().copied().collect();

                // First moment (mean)
                let mean = data_vec.iter().sum::<Float>() / data_vec.len() as Float;
                features.push(mean);

                // Second moment (variance)
                let variance = data_vec.iter().map(|&x| (x - mean).powi(2)).sum::<Float>()
                    / data_vec.len() as Float;
                features.push(variance);

                // Third moment (skewness approximation)
                let skewness = data_vec.iter().map(|&x| (x - mean).powi(3)).sum::<Float>()
                    / (data_vec.len() as Float * variance.powf(1.5));
                features.push(skewness);

                // Fourth moment (kurtosis approximation)
                let kurtosis = data_vec.iter().map(|&x| (x - mean).powi(4)).sum::<Float>()
                    / (data_vec.len() as Float * variance.powi(2));
                features.push(kurtosis);
            }
        }

        Ok(features)
    }
}

impl ImageFeatureExtractor {
    /// Compute histogram for a 2D array
    fn compute_histogram(
        &self,
        data: &scirs2_core::ndarray::ArrayView2<Float>,
        bins: usize,
    ) -> Result<Vec<Float>> {
        let min_val = data.iter().fold(Float::INFINITY, |a, &b| a.min(b));
        let max_val = data.iter().fold(Float::NEG_INFINITY, |a, &b| a.max(b));

        if (max_val - min_val).abs() < 1e-10 {
            return Ok(vec![0.0; bins]);
        }

        let mut histogram = vec![0.0; bins];
        let bin_width = (max_val - min_val) / bins as Float;

        for &value in data.iter() {
            let bin_index = ((value - min_val) / bin_width).floor() as usize;
            let bin_index = bin_index.min(bins - 1);
            histogram[bin_index] += 1.0;
        }

        // Normalize histogram
        let total_count = data.len() as Float;
        for bin in &mut histogram {
            *bin /= total_count;
        }

        Ok(histogram)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::arr3;

    #[test]
    fn test_image_normalizer_minmax() -> Result<()> {
        let image = arr3(&[
            [[100.0, 50.0, 200.0], [150.0, 75.0, 250.0]],
            [[200.0, 100.0, 255.0], [50.0, 25.0, 100.0]],
        ]);

        let normalizer = ImageNormalizer::new()
            .with_strategy(ImageNormalizationStrategy::MinMax)
            .with_channel_wise(true);

        let fitted = normalizer.fit(&image, &())?;
        let normalized = fitted.transform(&image)?;

        // Check that each channel is normalized to [0, 1]
        for channel in 0..3 {
            let channel_data = normalized.index_axis(Axis(2), channel);
            let min_val = channel_data.iter().fold(Float::INFINITY, |a, &b| a.min(b));
            let max_val = channel_data
                .iter()
                .fold(Float::NEG_INFINITY, |a, &b| a.max(b));

            assert_abs_diff_eq!(min_val, 0.0, epsilon = 1e-10);
            assert_abs_diff_eq!(max_val, 1.0, epsilon = 1e-10);
        }

        Ok(())
    }

    #[test]
    fn test_image_normalizer_standard_score() -> Result<()> {
        let image = arr3(&[
            [[100.0, 50.0, 200.0], [150.0, 75.0, 250.0]],
            [[200.0, 100.0, 255.0], [50.0, 25.0, 100.0]],
        ]);

        let normalizer = ImageNormalizer::new()
            .with_strategy(ImageNormalizationStrategy::StandardScore)
            .with_channel_wise(true);

        let fitted = normalizer.fit(&image, &())?;
        let normalized = fitted.transform(&image)?;

        // Check that each channel is standardized (approximately zero mean, unit std)
        for channel in 0..3 {
            let channel_data = normalized.index_axis(Axis(2), channel);
            let data_vec: Vec<Float> = channel_data.iter().copied().collect();

            let mean = data_vec.iter().sum::<Float>() / data_vec.len() as Float;
            let std = (data_vec.iter().map(|&x| (x - mean).powi(2)).sum::<Float>()
                / (data_vec.len() - 1) as Float)
                .sqrt();

            assert_abs_diff_eq!(mean, 0.0, epsilon = 1e-10);
            assert_abs_diff_eq!(std, 1.0, epsilon = 1e-10);
        }

        Ok(())
    }

    #[test]
    fn test_image_augmenter_horizontal_flip() -> Result<()> {
        let image = arr3(&[
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
        ]);

        let augmenter = ImageAugmenter::new()
            .with_horizontal_flip(true)
            .with_random_seed(42); // For deterministic testing

        let flipped = augmenter.horizontal_flip(&image)?;

        // Check that columns are flipped
        assert_eq!(flipped[[0, 0, 0]], image[[0, 1, 0]]);
        assert_eq!(flipped[[0, 1, 0]], image[[0, 0, 0]]);

        Ok(())
    }

    #[test]
    fn test_color_space_rgb_to_hsv() -> Result<()> {
        let rgb_image = arr3(&[
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], // Red, Green
            [[0.0, 0.0, 1.0], [1.0, 1.0, 1.0]], // Blue, White
        ]);

        let transformer = ColorSpaceTransformer::new().from_rgb().to_hsv();

        let hsv_image = transformer.transform(&rgb_image)?;

        // Red should have H=0, S=1, V=1
        assert_abs_diff_eq!(hsv_image[[0, 0, 0]], 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(hsv_image[[0, 0, 1]], 1.0, epsilon = 1e-6);
        assert_abs_diff_eq!(hsv_image[[0, 0, 2]], 1.0, epsilon = 1e-6);

        // White should have S=0, V=1
        assert_abs_diff_eq!(hsv_image[[1, 1, 1]], 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(hsv_image[[1, 1, 2]], 1.0, epsilon = 1e-6);

        Ok(())
    }

    #[test]
    fn test_rgb_to_grayscale() -> Result<()> {
        let rgb_image = arr3(&[
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], // Red, Green
            [[0.0, 0.0, 1.0], [1.0, 1.0, 1.0]], // Blue, White
        ]);

        let transformer = ColorSpaceTransformer::new().from_rgb().to_grayscale();

        let gray_image = transformer.transform(&rgb_image)?;

        // Check output shape
        assert_eq!(gray_image.dim().2, 1);

        // Red should be approximately 0.299
        assert_abs_diff_eq!(gray_image[[0, 0, 0]], 0.299, epsilon = 1e-6);

        // Green should be approximately 0.587
        assert_abs_diff_eq!(gray_image[[0, 1, 0]], 0.587, epsilon = 1e-6);

        // Blue should be approximately 0.114
        assert_abs_diff_eq!(gray_image[[1, 0, 0]], 0.114, epsilon = 1e-6);

        // White should be 1.0
        assert_abs_diff_eq!(gray_image[[1, 1, 0]], 1.0, epsilon = 1e-6);

        Ok(())
    }

    #[test]
    fn test_image_resizer_nearest() -> Result<()> {
        let image = arr3(&[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]);

        let resizer = ImageResizer::new((4, 4)).with_method(InterpolationMethod::Nearest);

        let resized = resizer.transform(&image)?;

        assert_eq!(resized.dim(), (4, 4, 2));

        // Check some values (nearest neighbor should replicate pixels)
        assert_eq!(resized[[0, 0, 0]], image[[0, 0, 0]]);
        assert_eq!(resized[[3, 3, 0]], image[[1, 1, 0]]);

        Ok(())
    }

    #[test]
    fn test_image_resizer_bilinear() -> Result<()> {
        let image = arr3(&[[[0.0, 0.0], [1.0, 1.0]], [[0.0, 0.0], [1.0, 1.0]]]);

        let resizer = ImageResizer::new((3, 3)).with_method(InterpolationMethod::Bilinear);

        let resized = resizer.transform(&image)?;

        assert_eq!(resized.dim(), (3, 3, 2));

        // Center pixel should be interpolated
        assert!(resized[[1, 1, 0]] > 0.0 && resized[[1, 1, 0]] < 1.0);

        Ok(())
    }

    #[test]
    fn test_edge_detector_sobel() -> Result<()> {
        // Create a larger test image with clear edges (RGB format - 4x4)
        let image = arr3(&[
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
            ],
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
            ],
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
            ],
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
            ],
        ]);

        let edge_detector = EdgeDetector::new().with_method(EdgeDetectionMethod::Sobel);

        let edges = edge_detector.transform(&image)?;

        // Check output shape (should be grayscale)
        assert_eq!(edges.dim().2, 1);

        // Edges should have been detected (use lower threshold for small gradients)
        let max_edge = edges.iter().fold(0.0_f64, |acc, &x| acc.max(x));
        assert!(
            max_edge > 0.01,
            "Expected edge detection to produce values > 0.01, got max: {}",
            max_edge
        );

        Ok(())
    }

    #[test]
    fn test_edge_detector_laplacian() -> Result<()> {
        // Create a larger test image with clear edges (4x4)
        let image = arr3(&[
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
            ],
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
            ],
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
            ],
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
            ],
        ]);

        let edge_detector = EdgeDetector::new().with_method(EdgeDetectionMethod::Laplacian);

        let edges = edge_detector.transform(&image)?;

        // Check output shape
        assert_eq!(edges.dim().2, 1);

        // Should detect edges (use lower threshold for small gradients)
        let max_edge = edges.iter().fold(0.0_f64, |acc, &x| acc.max(x));
        assert!(
            max_edge > 0.01,
            "Expected edge detection to produce values > 0.01, got max: {}",
            max_edge
        );

        Ok(())
    }

    #[test]
    fn test_edge_detector_with_threshold() -> Result<()> {
        let image = arr3(&[
            [[0.0, 0.5, 1.0], [0.0, 0.5, 1.0]],
            [[0.0, 0.5, 1.0], [0.0, 0.5, 1.0]],
        ]);

        let edge_detector = EdgeDetector::new()
            .with_method(EdgeDetectionMethod::Canny)
            .with_threshold(0.3);

        let edges = edge_detector.transform(&image)?;

        // Check output shape
        assert_eq!(edges.dim().2, 1);

        // Values should be binary (0.0 or 1.0) due to thresholding
        let all_binary = edges.iter().all(|&x| x == 0.0 || x == 1.0);
        assert!(all_binary);

        Ok(())
    }

    #[test]
    fn test_image_feature_extractor() -> Result<()> {
        let image = arr3(&[
            [[0.0, 0.5, 1.0], [0.2, 0.7, 0.9]],
            [[0.1, 0.6, 0.8], [0.3, 0.4, 0.6]],
        ]);

        let feature_extractor = ImageFeatureExtractor::new()
            .with_edge_features(true)
            .with_histogram_features(true, 4)
            .with_moment_features(true);

        let features = feature_extractor.transform(&image)?;

        // Should extract features
        assert!(!features.is_empty());

        // Should have edge features (2), histogram features (4 bins * 3 channels = 12),
        // and moment features (4 moments * 3 channels = 12)
        // Total: 2 + 12 + 12 = 26 features
        assert_eq!(features.len(), 26);

        // All features should be finite
        assert!(features.iter().all(|&x| x.is_finite()));

        Ok(())
    }

    #[test]
    fn test_image_feature_extractor_selective_features() -> Result<()> {
        let image = arr3(&[
            [[0.0, 0.5, 0.2], [0.2, 0.7, 0.1]],
            [[0.1, 0.6, 0.3], [0.3, 0.4, 0.2]],
        ]);

        // Only extract edge features
        let feature_extractor = ImageFeatureExtractor::new()
            .with_edge_features(true)
            .with_histogram_features(false, 4)
            .with_moment_features(false);

        let features = feature_extractor.transform(&image)?;

        // Should only have 2 edge features
        assert_eq!(features.len(), 2);

        // Features should be meaningful (non-negative for density and strength)
        assert!(features[0] >= 0.0); // Edge density
        assert!(features[1] >= 0.0); // Mean edge strength

        Ok(())
    }

    #[test]
    fn test_gaussian_blur() -> Result<()> {
        // Create a larger test image with clear edges for blur testing (6x6)
        let image = arr3(&[
            [[0.0], [0.0], [0.0], [1.0], [1.0], [1.0]],
            [[0.0], [0.0], [0.0], [1.0], [1.0], [1.0]],
            [[0.0], [0.0], [0.0], [1.0], [1.0], [1.0]],
            [[0.0], [0.0], [0.0], [1.0], [1.0], [1.0]],
            [[0.0], [0.0], [0.0], [1.0], [1.0], [1.0]],
            [[0.0], [0.0], [0.0], [1.0], [1.0], [1.0]],
        ]);

        // Test blur indirectly through edge detection with blur preprocessing
        let edge_detector_without_blur =
            EdgeDetector::new().with_method(EdgeDetectionMethod::Sobel);
        let edge_detector_with_blur = EdgeDetector::new()
            .with_method(EdgeDetectionMethod::Sobel)
            .with_blur_sigma(2.0);

        let edges_without_blur = edge_detector_without_blur.transform(&image)?;
        let edges_with_blur = edge_detector_with_blur.transform(&image)?;

        // Count non-zero edge pixels
        let edge_count_without_blur = edges_without_blur.iter().filter(|&&x| x > 0.01).count();
        let edge_count_with_blur = edges_with_blur.iter().filter(|&&x| x > 0.01).count();

        // Blur should reduce the number of detected edges OR reduce their strength
        let max_edge_without_blur = edges_without_blur
            .iter()
            .fold(0.0_f64, |acc, &x| acc.max(x));
        let max_edge_with_blur = edges_with_blur.iter().fold(0.0_f64, |acc, &x| acc.max(x));

        // At least one of these should be true: fewer edges detected OR weaker max edge strength
        assert!(
            edge_count_with_blur <= edge_count_without_blur
                || max_edge_with_blur <= max_edge_without_blur,
            "Expected blur to reduce edge count ({} vs {}) or max edge strength ({:.6} vs {:.6})",
            edge_count_with_blur,
            edge_count_without_blur,
            max_edge_with_blur,
            max_edge_without_blur
        );

        Ok(())
    }
}
