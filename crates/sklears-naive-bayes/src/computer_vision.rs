//! Computer Vision specific Naive Bayes implementations
//!
//! This module provides specialized Naive Bayes classifiers for computer vision tasks,
//! including image classification, spatial analysis, texture recognition, and more.

// SciRS2 Policy Compliance - Use scirs2-autograd for ndarray types
use scirs2_core::ndarray::{s, Array1, Array2, Array3, Array4, ArrayView2, Zip};
use scirs2_core::numeric::Float;
// SciRS2 Policy Compliance - Use scirs2-core for random functionality
use scirs2_core::random::Rng;
use sklears_core::traits::Estimator;
use std::collections::HashMap;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ComputerVisionError {
    #[error("Invalid image dimensions: {0}")]
    InvalidDimensions(String),
    #[error("Color space conversion error: {0}")]
    ColorSpaceError(String),
    #[error("Texture analysis error: {0}")]
    TextureError(String),
    #[error("Spatial analysis error: {0}")]
    SpatialError(String),
    #[error("Feature pyramid error: {0}")]
    PyramidError(String),
    #[error("Insufficient data for analysis")]
    InsufficientData,
}

type Result<T> = std::result::Result<T, ComputerVisionError>;

/// Image representation and preprocessing utilities
#[derive(Debug, Clone)]
pub struct ImageData {
    /// Image pixels as height x width x channels
    pub pixels: Array3<f64>,
    /// Image metadata
    pub metadata: ImageMetadata,
}

#[derive(Debug, Clone)]
pub struct ImageMetadata {
    pub height: usize,
    pub width: usize,
    pub channels: usize,
    pub color_space: ColorSpace,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ColorSpace {
    /// RGB
    RGB,
    /// Grayscale
    Grayscale,
    /// HSV
    HSV,
    /// LAB
    LAB,
}

/// Configuration for image-based Naive Bayes classifiers
#[derive(Debug, Clone)]
pub struct ImageNBConfig {
    pub use_spatial_info: bool,
    pub histogram_bins: usize,
    pub texture_analysis: bool,
    pub pyramid_levels: usize,
    pub color_space: ColorSpace,
    pub smoothing_alpha: f64,
}

impl Default for ImageNBConfig {
    fn default() -> Self {
        Self {
            use_spatial_info: true,
            histogram_bins: 64,
            texture_analysis: true,
            pyramid_levels: 3,
            color_space: ColorSpace::RGB,
            smoothing_alpha: 1.0,
        }
    }
}

/// Image classification Naive Bayes classifier
#[derive(Debug, Clone)]
pub struct ImageNaiveBayes {
    config: ImageNBConfig,
    classes: Array1<i32>,
    class_log_prior: Array1<f64>,
    color_histograms: Vec<Array2<f64>>, // One histogram per class
    spatial_features: Vec<Array2<f64>>, // Spatial features per class
    texture_features: Vec<Array1<f64>>, // Texture features per class
    pyramid_features: Vec<Vec<Array1<f64>>>, // Pyramid features per class and level
    is_fitted: bool,
}

impl ImageNaiveBayes {
    pub fn new(config: ImageNBConfig) -> Self {
        Self {
            config,
            classes: Array1::zeros(0),
            class_log_prior: Array1::zeros(0),
            color_histograms: Vec::new(),
            spatial_features: Vec::new(),
            texture_features: Vec::new(),
            pyramid_features: Vec::new(),
            is_fitted: false,
        }
    }

    pub fn fit(&mut self, images: &[ImageData], y: &Array1<i32>) -> Result<()> {
        if images.len() != y.len() {
            return Err(ComputerVisionError::InvalidDimensions(
                "Number of images must match number of labels".to_string(),
            ));
        }

        // Extract unique classes
        let unique_classes: Vec<i32> = {
            let mut classes = y.to_vec();
            classes.sort_unstable();
            classes.dedup();
            classes
        };
        self.classes = Array1::from_vec(unique_classes);

        // Compute class priors
        self.class_log_prior = self.compute_class_log_prior(y)?;

        // Initialize feature storage
        let n_classes = self.classes.len();
        self.color_histograms = Vec::with_capacity(n_classes);
        self.spatial_features = Vec::with_capacity(n_classes);
        self.texture_features = Vec::with_capacity(n_classes);
        self.pyramid_features = vec![Vec::new(); n_classes];

        // Process images for each class
        for (class_idx, &class_label) in self.classes.iter().enumerate() {
            let class_images: Vec<&ImageData> = images
                .iter()
                .zip(y.iter())
                .filter(|(_, &label)| label == class_label)
                .map(|(img, _)| img)
                .collect();

            if class_images.is_empty() {
                return Err(ComputerVisionError::InsufficientData);
            }

            // Compute color histograms
            let color_hist = self.compute_class_color_histogram(&class_images)?;
            self.color_histograms.push(color_hist);

            // Compute spatial features if enabled
            if self.config.use_spatial_info {
                let spatial_feat = self.compute_class_spatial_features(&class_images)?;
                self.spatial_features.push(spatial_feat);
            }

            // Compute texture features if enabled
            if self.config.texture_analysis {
                let texture_feat = self.compute_class_texture_features(&class_images)?;
                self.texture_features.push(texture_feat);
            }

            // Compute pyramid features
            if self.config.pyramid_levels > 0 {
                let pyramid_feat = self.compute_class_pyramid_features(&class_images)?;
                self.pyramid_features[class_idx] = pyramid_feat;
            }
        }

        self.is_fitted = true;
        Ok(())
    }

    pub fn predict(&self, images: &[ImageData]) -> Result<Array1<i32>> {
        if !self.is_fitted {
            return Err(ComputerVisionError::InsufficientData);
        }

        let mut predictions = Array1::zeros(images.len());

        for (i, image) in images.iter().enumerate() {
            let log_probs = self.predict_log_proba_single(image)?;
            let best_class_idx = log_probs
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap()
                .0;
            predictions[i] = self.classes[best_class_idx];
        }

        Ok(predictions)
    }

    pub fn predict_proba(&self, images: &[ImageData]) -> Result<Array2<f64>> {
        if !self.is_fitted {
            return Err(ComputerVisionError::InsufficientData);
        }

        let mut probabilities = Array2::zeros((images.len(), self.classes.len()));

        for (i, image) in images.iter().enumerate() {
            let log_probs = self.predict_log_proba_single(image)?;

            // Convert log probabilities to probabilities
            let max_log_prob = log_probs.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let exp_probs: Vec<f64> = log_probs
                .iter()
                .map(|&p| (p - max_log_prob).exp())
                .collect();
            let sum_exp: f64 = exp_probs.iter().sum();

            for (j, &prob) in exp_probs.iter().enumerate() {
                probabilities[[i, j]] = prob / sum_exp;
            }
        }

        Ok(probabilities)
    }

    fn predict_log_proba_single(&self, image: &ImageData) -> Result<Array1<f64>> {
        let mut log_probs = self.class_log_prior.clone();

        // Extract features from the image
        let color_hist = self.extract_color_histogram(image)?;

        // Add color histogram likelihood
        for (class_idx, class_hist) in self.color_histograms.iter().enumerate() {
            let color_log_likelihood =
                self.compute_histogram_log_likelihood(&color_hist, class_hist);
            log_probs[class_idx] += color_log_likelihood;
        }

        // Add spatial features likelihood if enabled
        if self.config.use_spatial_info && !self.spatial_features.is_empty() {
            let spatial_feat = self.extract_spatial_features(image)?;
            for (class_idx, class_spatial) in self.spatial_features.iter().enumerate() {
                let spatial_log_likelihood =
                    self.compute_spatial_log_likelihood(&spatial_feat, class_spatial);
                log_probs[class_idx] += spatial_log_likelihood;
            }
        }

        // Add texture features likelihood if enabled
        if self.config.texture_analysis && !self.texture_features.is_empty() {
            let texture_feat = self.extract_texture_features(image)?;
            for (class_idx, class_texture) in self.texture_features.iter().enumerate() {
                let texture_log_likelihood =
                    self.compute_texture_log_likelihood(&texture_feat, class_texture);
                log_probs[class_idx] += texture_log_likelihood;
            }
        }

        // Add pyramid features likelihood
        if self.config.pyramid_levels > 0 && !self.pyramid_features.is_empty() {
            let pyramid_feat = self.extract_pyramid_features(image)?;
            for (class_idx, class_pyramid) in self.pyramid_features.iter().enumerate() {
                let pyramid_log_likelihood =
                    self.compute_pyramid_log_likelihood(&pyramid_feat, class_pyramid);
                log_probs[class_idx] += pyramid_log_likelihood;
            }
        }

        Ok(log_probs)
    }

    fn compute_class_log_prior(&self, y: &Array1<i32>) -> Result<Array1<f64>> {
        let n_samples = y.len() as f64;
        let mut class_counts = Array1::zeros(self.classes.len());

        for &label in y.iter() {
            for (i, &class) in self.classes.iter().enumerate() {
                if label == class {
                    class_counts[i] += 1.0;
                    break;
                }
            }
        }

        let class_priors = &class_counts / n_samples;
        let class_log_prior = class_priors.mapv(|p: f64| (p + 1e-10).ln());
        Ok(class_log_prior)
    }

    fn compute_class_color_histogram(&self, images: &[&ImageData]) -> Result<Array2<f64>> {
        let n_bins = self.config.histogram_bins;
        let n_channels = images[0].metadata.channels;
        let mut histogram = Array2::zeros((n_channels, n_bins));

        for image in images {
            let img_hist = self.extract_color_histogram(image)?;
            histogram = &histogram + &img_hist;
        }

        // Normalize and apply smoothing
        let total_pixels: f64 = histogram.sum();
        if total_pixels > 0.0 {
            histogram /= total_pixels;
        }
        histogram += self.config.smoothing_alpha / (n_bins as f64);

        // Renormalize after smoothing
        let new_total: f64 = histogram.sum();
        if new_total > 0.0 {
            histogram /= new_total;
        }

        Ok(histogram)
    }

    fn extract_color_histogram(&self, image: &ImageData) -> Result<Array2<f64>> {
        let n_bins = self.config.histogram_bins;
        let n_channels = image.metadata.channels;
        let mut histogram = Array2::zeros((n_channels, n_bins));

        for channel in 0..n_channels {
            let channel_data = image.pixels.slice(s![.., .., channel]);
            for &pixel_value in channel_data.iter() {
                let bin =
                    ((pixel_value.clamp(0.0, 1.0) * (n_bins - 1) as f64) as usize).min(n_bins - 1);
                histogram[[channel, bin]] += 1.0;
            }
        }

        // Normalize
        let total_pixels = histogram.sum();
        if total_pixels > 0.0 {
            histogram /= total_pixels;
        }

        Ok(histogram)
    }

    fn compute_histogram_log_likelihood(
        &self,
        observed: &Array2<f64>,
        learned: &Array2<f64>,
    ) -> f64 {
        let mut log_likelihood = 0.0;

        Zip::from(observed).and(learned).for_each(|&obs, &learned| {
            if obs > 0.0 && learned > 0.0 {
                log_likelihood += obs * learned.ln();
            }
        });

        log_likelihood
    }

    fn compute_class_spatial_features(&self, images: &[&ImageData]) -> Result<Array2<f64>> {
        // Simple spatial features: mean and std per spatial region
        let mut spatial_features = Vec::new();

        for image in images {
            let spatial_feat = self.extract_spatial_features(image)?;
            spatial_features.push(spatial_feat);
        }

        if spatial_features.is_empty() {
            return Err(ComputerVisionError::InsufficientData);
        }

        // Average spatial features across all images in the class
        let feature_size = spatial_features[0].len();
        let mut mean_features = Array2::zeros((2, feature_size)); // mean and std

        for i in 0..feature_size {
            let values: Vec<f64> = spatial_features.iter().map(|f| f[i]).collect();
            let mean = values.iter().sum::<f64>() / values.len() as f64;
            let variance =
                values.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
            let std = variance.sqrt();

            mean_features[[0, i]] = mean;
            mean_features[[1, i]] = std + 1e-10; // Add small epsilon for numerical stability
        }

        Ok(mean_features)
    }

    fn extract_spatial_features(&self, image: &ImageData) -> Result<Array1<f64>> {
        // Extract spatial features from image regions (4x4 grid)
        let height = image.metadata.height;
        let width = image.metadata.width;
        let channels = image.metadata.channels;

        let grid_h = 4;
        let grid_w = 4;
        let block_h = height / grid_h;
        let block_w = width / grid_w;

        let mut features = Vec::new();

        for grid_i in 0..grid_h {
            for grid_j in 0..grid_w {
                let start_h = grid_i * block_h;
                let end_h = ((grid_i + 1) * block_h).min(height);
                let start_w = grid_j * block_w;
                let end_w = ((grid_j + 1) * block_w).min(width);

                for channel in 0..channels {
                    let region = image
                        .pixels
                        .slice(s![start_h..end_h, start_w..end_w, channel]);
                    let mean = region.mean().unwrap_or(0.0);
                    let variance = region.mapv(|x| (x - mean).powi(2)).mean().unwrap_or(0.0);

                    features.push(mean);
                    features.push(variance.sqrt());
                }
            }
        }

        Ok(Array1::from_vec(features))
    }

    fn compute_spatial_log_likelihood(&self, observed: &Array1<f64>, learned: &Array2<f64>) -> f64 {
        let mut log_likelihood = 0.0;

        for (i, &obs_val) in observed.iter().enumerate() {
            let mean = learned[[0, i]];
            let std = learned[[1, i]];

            // Gaussian likelihood
            let gaussian_log_prob = -0.5 * ((obs_val - mean) / std).powi(2)
                - std.ln()
                - 0.5 * (2.0 * std::f64::consts::PI).ln();
            log_likelihood += gaussian_log_prob;
        }

        log_likelihood
    }

    fn compute_class_texture_features(&self, images: &[&ImageData]) -> Result<Array1<f64>> {
        let mut texture_features = Vec::new();

        for image in images {
            let texture_feat = self.extract_texture_features(image)?;
            texture_features.push(texture_feat);
        }

        if texture_features.is_empty() {
            return Err(ComputerVisionError::InsufficientData);
        }

        // Average texture features
        let feature_size = texture_features[0].len();
        let mut mean_features = Array1::zeros(feature_size);

        for i in 0..feature_size {
            let mean =
                texture_features.iter().map(|f| f[i]).sum::<f64>() / texture_features.len() as f64;
            mean_features[i] = mean;
        }

        Ok(mean_features)
    }

    fn extract_texture_features(&self, image: &ImageData) -> Result<Array1<f64>> {
        // Simple texture features based on gradients and local patterns
        let mut features = Vec::new();

        for channel in 0..image.metadata.channels {
            let channel_data = image.pixels.slice(s![.., .., channel]);

            // Compute gradients
            let (grad_x, grad_y) = self.compute_gradients(&channel_data);

            // Gradient statistics
            let grad_magnitude = Zip::from(&grad_x)
                .and(&grad_y)
                .map_collect(|&gx, &gy| (gx * gx + gy * gy).sqrt());

            features.push(grad_magnitude.mean().unwrap_or(0.0));
            features.push(grad_magnitude.std(0.0));

            // Local Binary Pattern-like features
            let lbp_features = self.compute_simple_lbp(&channel_data);
            features.extend(lbp_features);
        }

        Ok(Array1::from_vec(features))
    }

    fn compute_gradients(&self, image: &ArrayView2<f64>) -> (Array2<f64>, Array2<f64>) {
        let (height, width) = image.dim();
        let mut grad_x = Array2::zeros((height, width));
        let mut grad_y = Array2::zeros((height, width));

        // Sobel operators
        for i in 1..height - 1 {
            for j in 1..width - 1 {
                grad_x[[i, j]] = -image[[i - 1, j - 1]] + image[[i - 1, j + 1]]
                    - 2.0 * image[[i, j - 1]]
                    + 2.0 * image[[i, j + 1]]
                    - image[[i + 1, j - 1]]
                    + image[[i + 1, j + 1]];

                grad_y[[i, j]] =
                    -image[[i - 1, j - 1]] - 2.0 * image[[i - 1, j]] - image[[i - 1, j + 1]]
                        + image[[i + 1, j - 1]]
                        + 2.0 * image[[i + 1, j]]
                        + image[[i + 1, j + 1]];
            }
        }

        (grad_x, grad_y)
    }

    fn compute_simple_lbp(&self, image: &ArrayView2<f64>) -> Vec<f64> {
        let (height, width) = image.dim();
        let mut lbp_histogram = vec![0; 256];

        for i in 1..height - 1 {
            for j in 1..width - 1 {
                let center = image[[i, j]];
                let mut lbp_code = 0u8;

                // 8-connectivity LBP
                let neighbors = [
                    image[[i - 1, j - 1]],
                    image[[i - 1, j]],
                    image[[i - 1, j + 1]],
                    image[[i, j + 1]],
                    image[[i + 1, j + 1]],
                    image[[i + 1, j]],
                    image[[i + 1, j - 1]],
                    image[[i, j - 1]],
                ];

                for (k, &neighbor) in neighbors.iter().enumerate() {
                    if neighbor >= center {
                        lbp_code |= 1 << k;
                    }
                }

                lbp_histogram[lbp_code as usize] += 1;
            }
        }

        // Normalize histogram
        let total: i32 = lbp_histogram.iter().sum();
        if total > 0 {
            lbp_histogram
                .iter()
                .map(|&count| count as f64 / total as f64)
                .collect()
        } else {
            vec![0.0; 256]
        }
    }

    fn compute_texture_log_likelihood(&self, observed: &Array1<f64>, learned: &Array1<f64>) -> f64 {
        let mut log_likelihood = 0.0;

        for (&obs, &learned_val) in observed.iter().zip(learned.iter()) {
            if learned_val > 0.0 {
                // Simple Gaussian assumption for texture features
                let diff = obs - learned_val;
                log_likelihood += -0.5 * diff * diff / (0.1 * 0.1) - (0.1_f64).ln();
            }
        }

        log_likelihood
    }

    fn compute_class_pyramid_features(&self, images: &[&ImageData]) -> Result<Vec<Array1<f64>>> {
        let mut pyramid_features = Vec::with_capacity(self.config.pyramid_levels);

        for level in 0..self.config.pyramid_levels {
            let mut level_features = Vec::new();

            for image in images {
                let level_feat = self.extract_pyramid_features_at_level(image, level)?;
                level_features.push(level_feat);
            }

            if level_features.is_empty() {
                return Err(ComputerVisionError::InsufficientData);
            }

            // Average features at this level
            let feature_size = level_features[0].len();
            let mut mean_features = Array1::zeros(feature_size);

            for i in 0..feature_size {
                let mean =
                    level_features.iter().map(|f| f[i]).sum::<f64>() / level_features.len() as f64;
                mean_features[i] = mean;
            }

            pyramid_features.push(mean_features);
        }

        Ok(pyramid_features)
    }

    fn extract_pyramid_features(&self, image: &ImageData) -> Result<Vec<Array1<f64>>> {
        let mut pyramid_features = Vec::with_capacity(self.config.pyramid_levels);

        for level in 0..self.config.pyramid_levels {
            let level_feat = self.extract_pyramid_features_at_level(image, level)?;
            pyramid_features.push(level_feat);
        }

        Ok(pyramid_features)
    }

    fn extract_pyramid_features_at_level(
        &self,
        image: &ImageData,
        level: usize,
    ) -> Result<Array1<f64>> {
        // Create downsampled version of the image
        let scale_factor = 2_usize.pow(level as u32);
        let new_height = (image.metadata.height / scale_factor).max(1);
        let new_width = (image.metadata.width / scale_factor).max(1);

        let downsampled = self.downsample_image(image, new_height, new_width)?;

        // Extract basic statistics from downsampled image
        let mut features = Vec::new();

        for channel in 0..image.metadata.channels {
            let channel_data = downsampled.slice(s![.., .., channel]);
            features.push(channel_data.mean().unwrap_or(0.0));
            features.push(channel_data.std(0.0));
            features.push(
                channel_data
                    .iter()
                    .fold(f64::NEG_INFINITY, |a, &b| a.max(b)),
            );
            features.push(channel_data.iter().fold(f64::INFINITY, |a, &b| a.min(b)));
        }

        Ok(Array1::from_vec(features))
    }

    fn downsample_image(
        &self,
        image: &ImageData,
        new_height: usize,
        new_width: usize,
    ) -> Result<Array3<f64>> {
        let old_height = image.metadata.height;
        let old_width = image.metadata.width;
        let channels = image.metadata.channels;

        let mut downsampled = Array3::zeros((new_height, new_width, channels));

        let height_ratio = old_height as f64 / new_height as f64;
        let width_ratio = old_width as f64 / new_width as f64;

        for i in 0..new_height {
            for j in 0..new_width {
                for c in 0..channels {
                    let old_i = ((i as f64 + 0.5) * height_ratio - 0.5).round() as usize;
                    let old_j = ((j as f64 + 0.5) * width_ratio - 0.5).round() as usize;

                    let old_i = old_i.min(old_height - 1);
                    let old_j = old_j.min(old_width - 1);

                    downsampled[[i, j, c]] = image.pixels[[old_i, old_j, c]];
                }
            }
        }

        Ok(downsampled)
    }

    fn compute_pyramid_log_likelihood(
        &self,
        observed: &Vec<Array1<f64>>,
        learned: &Vec<Array1<f64>>,
    ) -> f64 {
        let mut log_likelihood = 0.0;

        for (obs_level, learned_level) in observed.iter().zip(learned.iter()) {
            for (&obs_val, &learned_val) in obs_level.iter().zip(learned_level.iter()) {
                // Simple Gaussian assumption
                let diff = obs_val - learned_val;
                log_likelihood += -0.5 * diff * diff / (0.1 * 0.1) - (0.1_f64).ln();
            }
        }

        log_likelihood
    }
}

/// Spatial Naive Bayes for analyzing spatial relationships in images
#[derive(Debug, Clone)]
pub struct SpatialNaiveBayes {
    config: SpatialNBConfig,
    classes: Array1<i32>,
    class_log_prior: Array1<f64>,
    spatial_models: Vec<SpatialModel>,
    is_fitted: bool,
}

#[derive(Debug, Clone)]
pub struct SpatialNBConfig {
    pub grid_size: (usize, usize),
    pub neighborhood_radius: usize,
    pub use_spatial_correlation: bool,
    pub smoothing_alpha: f64,
}

impl Default for SpatialNBConfig {
    fn default() -> Self {
        Self {
            grid_size: (8, 8),
            neighborhood_radius: 1,
            use_spatial_correlation: true,
            smoothing_alpha: 1.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SpatialModel {
    grid_probabilities: Array2<f64>,
    spatial_transitions: Array4<f64>, // [from_grid_i, from_grid_j, to_grid_i, to_grid_j]
    neighborhood_stats: HashMap<(usize, usize), NeighborhoodStats>,
}

#[derive(Debug, Clone)]
pub struct NeighborhoodStats {
    mean: f64,
    variance: f64,
    count: usize,
}

impl SpatialNaiveBayes {
    pub fn new(config: SpatialNBConfig) -> Self {
        Self {
            config,
            classes: Array1::zeros(0),
            class_log_prior: Array1::zeros(0),
            spatial_models: Vec::new(),
            is_fitted: false,
        }
    }

    pub fn fit(&mut self, images: &[ImageData], y: &Array1<i32>) -> Result<()> {
        if images.len() != y.len() {
            return Err(ComputerVisionError::InvalidDimensions(
                "Number of images must match number of labels".to_string(),
            ));
        }

        // Extract unique classes
        let unique_classes: Vec<i32> = {
            let mut classes = y.to_vec();
            classes.sort_unstable();
            classes.dedup();
            classes
        };
        self.classes = Array1::from_vec(unique_classes);

        // Compute class priors
        let n_samples = y.len() as f64;
        let mut class_counts = Array1::zeros(self.classes.len());

        for &label in y.iter() {
            for (i, &class) in self.classes.iter().enumerate() {
                if label == class {
                    class_counts[i] += 1.0;
                    break;
                }
            }
        }

        let class_priors = &class_counts / n_samples;
        self.class_log_prior = class_priors.mapv(|p: f64| (p + 1e-10).ln());

        // Build spatial models for each class
        for (class_idx, &class_label) in self.classes.iter().enumerate() {
            let class_images: Vec<&ImageData> = images
                .iter()
                .zip(y.iter())
                .filter(|(_, &label)| label == class_label)
                .map(|(img, _)| img)
                .collect();

            if class_images.is_empty() {
                return Err(ComputerVisionError::InsufficientData);
            }

            let spatial_model = self.build_spatial_model(&class_images)?;
            self.spatial_models.push(spatial_model);
        }

        self.is_fitted = true;
        Ok(())
    }

    pub fn predict(&self, images: &[ImageData]) -> Result<Array1<i32>> {
        if !self.is_fitted {
            return Err(ComputerVisionError::InsufficientData);
        }

        let mut predictions = Array1::zeros(images.len());

        for (i, image) in images.iter().enumerate() {
            let log_probs = self.predict_log_proba_single(image)?;
            let best_class_idx = log_probs
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap()
                .0;
            predictions[i] = self.classes[best_class_idx];
        }

        Ok(predictions)
    }

    fn build_spatial_model(&self, images: &[&ImageData]) -> Result<SpatialModel> {
        let (grid_h, grid_w) = self.config.grid_size;
        let mut grid_probabilities = Array2::zeros((grid_h, grid_w));
        let spatial_transitions = Array4::zeros((grid_h, grid_w, grid_h, grid_w));
        let mut neighborhood_stats = HashMap::new();

        // Process each image to build spatial statistics
        for image in images {
            let spatial_features = self.extract_spatial_grid_features(image)?;

            // Update grid probabilities
            for ((i, j), &value) in spatial_features.indexed_iter() {
                grid_probabilities[[i, j]] += value;

                // Update neighborhood statistics
                for di in -(self.config.neighborhood_radius as isize)
                    ..=(self.config.neighborhood_radius as isize)
                {
                    for dj in -(self.config.neighborhood_radius as isize)
                        ..=(self.config.neighborhood_radius as isize)
                    {
                        if di == 0 && dj == 0 {
                            continue;
                        }

                        let ni = (i as isize + di) as usize;
                        let nj = (j as isize + dj) as usize;

                        if ni < grid_h && nj < grid_w {
                            let neighbor_value = spatial_features[[ni, nj]];

                            let key = (i * grid_w + j, ni * grid_w + nj);
                            let stats =
                                neighborhood_stats.entry(key).or_insert(NeighborhoodStats {
                                    mean: 0.0,
                                    variance: 0.0,
                                    count: 0,
                                });

                            // Online mean and variance update
                            stats.count += 1;
                            let delta = neighbor_value - stats.mean;
                            stats.mean += delta / stats.count as f64;
                            let delta2 = neighbor_value - stats.mean;
                            stats.variance += delta * delta2;
                        }
                    }
                }
            }
        }

        // Normalize grid probabilities
        let total = grid_probabilities.sum();
        if total > 0.0 {
            grid_probabilities /= total;
        }
        grid_probabilities += self.config.smoothing_alpha / (grid_h * grid_w) as f64;

        // Finalize neighborhood statistics
        for (_, stats) in neighborhood_stats.iter_mut() {
            if stats.count > 1 {
                stats.variance /= (stats.count - 1) as f64;
            }
        }

        Ok(SpatialModel {
            grid_probabilities,
            spatial_transitions,
            neighborhood_stats,
        })
    }

    fn extract_spatial_grid_features(&self, image: &ImageData) -> Result<Array2<f64>> {
        let (grid_h, grid_w) = self.config.grid_size;
        let height = image.metadata.height;
        let width = image.metadata.width;

        let block_h = height / grid_h;
        let block_w = width / grid_w;

        let mut features = Array2::zeros((grid_h, grid_w));

        for i in 0..grid_h {
            for j in 0..grid_w {
                let start_h = i * block_h;
                let end_h = ((i + 1) * block_h).min(height);
                let start_w = j * block_w;
                let end_w = ((j + 1) * block_w).min(width);

                // Compute average intensity for this grid cell
                let mut sum = 0.0;
                let mut count = 0;

                for ch in 0..image.metadata.channels {
                    let region = image.pixels.slice(s![start_h..end_h, start_w..end_w, ch]);
                    sum += region.sum();
                    count += region.len();
                }

                features[[i, j]] = if count > 0 { sum / count as f64 } else { 0.0 };
            }
        }

        Ok(features)
    }

    fn predict_log_proba_single(&self, image: &ImageData) -> Result<Array1<f64>> {
        let mut log_probs = self.class_log_prior.clone();
        let spatial_features = self.extract_spatial_grid_features(image)?;

        for (class_idx, spatial_model) in self.spatial_models.iter().enumerate() {
            let spatial_log_likelihood =
                self.compute_spatial_log_likelihood(&spatial_features, spatial_model);
            log_probs[class_idx] += spatial_log_likelihood;
        }

        Ok(log_probs)
    }

    fn compute_spatial_log_likelihood(&self, features: &Array2<f64>, model: &SpatialModel) -> f64 {
        let mut log_likelihood = 0.0;

        // Grid-based likelihood
        for ((i, j), &value) in features.indexed_iter() {
            let prob = model.grid_probabilities[[i, j]];
            if prob > 0.0 {
                log_likelihood += value * prob.ln();
            }
        }

        // Spatial correlation likelihood if enabled
        if self.config.use_spatial_correlation {
            let (grid_h, grid_w) = features.dim();

            for i in 0..grid_h {
                for j in 0..grid_w {
                    let current_value = features[[i, j]];

                    for di in -(self.config.neighborhood_radius as isize)
                        ..=(self.config.neighborhood_radius as isize)
                    {
                        for dj in -(self.config.neighborhood_radius as isize)
                            ..=(self.config.neighborhood_radius as isize)
                        {
                            if di == 0 && dj == 0 {
                                continue;
                            }

                            let ni = (i as isize + di) as usize;
                            let nj = (j as isize + dj) as usize;

                            if ni < grid_h && nj < grid_w {
                                let neighbor_value = features[[ni, nj]];
                                let key = (i * grid_w + j, ni * grid_w + nj);

                                if let Some(stats) = model.neighborhood_stats.get(&key) {
                                    if stats.count > 0 {
                                        let std = (stats.variance + 1e-10).sqrt();
                                        let gaussian_log_prob = -0.5
                                            * ((neighbor_value - stats.mean) / std).powi(2)
                                            - std.ln();
                                        log_likelihood += gaussian_log_prob;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        log_likelihood
    }
}

/// Utility functions for image processing and computer vision
pub mod utils {
    use super::*;

    /// Convert RGB image to grayscale
    pub fn rgb_to_grayscale(image: &ImageData) -> Result<ImageData> {
        if image.metadata.channels != 3 {
            return Err(ComputerVisionError::ColorSpaceError(
                "Image must have 3 channels for RGB conversion".to_string(),
            ));
        }

        let height = image.metadata.height;
        let width = image.metadata.width;
        let mut gray_pixels = Array3::zeros((height, width, 1));

        for i in 0..height {
            for j in 0..width {
                let r = image.pixels[[i, j, 0]];
                let g = image.pixels[[i, j, 1]];
                let b = image.pixels[[i, j, 2]];

                // Standard grayscale conversion
                gray_pixels[[i, j, 0]] = 0.299 * r + 0.587 * g + 0.114 * b;
            }
        }

        Ok(ImageData {
            pixels: gray_pixels,
            metadata: ImageMetadata {
                height,
                width,
                channels: 1,
                color_space: ColorSpace::Grayscale,
            },
        })
    }

    /// Convert RGB to HSV color space
    pub fn rgb_to_hsv(image: &ImageData) -> Result<ImageData> {
        if image.metadata.channels != 3 {
            return Err(ComputerVisionError::ColorSpaceError(
                "Image must have 3 channels for RGB to HSV conversion".to_string(),
            ));
        }

        let height = image.metadata.height;
        let width = image.metadata.width;
        let mut hsv_pixels = Array3::zeros((height, width, 3));

        for i in 0..height {
            for j in 0..width {
                let r = image.pixels[[i, j, 0]];
                let g = image.pixels[[i, j, 1]];
                let b = image.pixels[[i, j, 2]];

                let max_val = r.max(g).max(b);
                let min_val = r.min(g).min(b);
                let delta = max_val - min_val;

                // Hue calculation
                let h = if delta == 0.0 {
                    0.0
                } else if max_val == r {
                    60.0 * (((g - b) / delta) % 6.0)
                } else if max_val == g {
                    60.0 * ((b - r) / delta + 2.0)
                } else {
                    60.0 * ((r - g) / delta + 4.0)
                };

                // Saturation calculation
                let s = if max_val == 0.0 { 0.0 } else { delta / max_val };

                // Value calculation
                let v = max_val;

                hsv_pixels[[i, j, 0]] = h / 360.0; // Normalize to [0, 1]
                hsv_pixels[[i, j, 1]] = s;
                hsv_pixels[[i, j, 2]] = v;
            }
        }

        Ok(ImageData {
            pixels: hsv_pixels,
            metadata: ImageMetadata {
                height,
                width,
                channels: 3,
                color_space: ColorSpace::HSV,
            },
        })
    }

    /// Create a simple test image for demonstration
    pub fn create_test_image(height: usize, width: usize, pattern: TestPattern) -> ImageData {
        let mut pixels = Array3::zeros((height, width, 3));

        match pattern {
            TestPattern::Gradient => {
                for i in 0..height {
                    for j in 0..width {
                        let intensity = (i + j) as f64 / (height + width) as f64;
                        pixels[[i, j, 0]] = intensity;
                        pixels[[i, j, 1]] = intensity;
                        pixels[[i, j, 2]] = intensity;
                    }
                }
            }
            TestPattern::Checkerboard => {
                let block_size = 10;
                for i in 0..height {
                    for j in 0..width {
                        let block_i = i / block_size;
                        let block_j = j / block_size;
                        let intensity = if (block_i + block_j) % 2 == 0 {
                            1.0
                        } else {
                            0.0
                        };
                        pixels[[i, j, 0]] = intensity;
                        pixels[[i, j, 1]] = intensity;
                        pixels[[i, j, 2]] = intensity;
                    }
                }
            }
            TestPattern::Random => {
                let mut rng = scirs2_core::random::thread_rng();
                for i in 0..height {
                    for j in 0..width {
                        for c in 0..3 {
                            pixels[[i, j, c]] = rng.gen::<f64>();
                        }
                    }
                }
            }
        }

        ImageData {
            pixels,
            metadata: ImageMetadata {
                height,
                width,
                channels: 3,
                color_space: ColorSpace::RGB,
            },
        }
    }

    #[derive(Debug, Clone)]
    pub enum TestPattern {
        /// Gradient
        Gradient,
        /// Checkerboard
        Checkerboard,
        /// Random
        Random,
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {

    use super::utils::*;
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_image_naive_bayes_creation() {
        let config = ImageNBConfig::default();
        let nb = ImageNaiveBayes::new(config);
        assert!(!nb.is_fitted);
    }

    #[test]
    fn test_image_naive_bayes_fit_predict() {
        let mut nb = ImageNaiveBayes::new(ImageNBConfig::default());

        // Create test images
        let image1 = create_test_image(32, 32, TestPattern::Gradient);
        let image2 = create_test_image(32, 32, TestPattern::Checkerboard);
        let image3 = create_test_image(32, 32, TestPattern::Random);

        let images = vec![image1.clone(), image2.clone(), image3.clone()];
        let labels = Array1::from_vec(vec![0, 1, 0]);

        nb.fit(&images, &labels).unwrap();
        assert!(nb.is_fitted);

        let predictions = nb.predict(&images).unwrap();
        assert_eq!(predictions.len(), 3);

        let probabilities = nb.predict_proba(&images).unwrap();
        assert_eq!(probabilities.dim(), (3, 2));

        // Check probabilities sum to 1
        for i in 0..3 {
            let row_sum: f64 = probabilities.row(i).sum();
            assert_abs_diff_eq!(row_sum, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_spatial_naive_bayes_creation() {
        let config = SpatialNBConfig::default();
        let nb = SpatialNaiveBayes::new(config);
        assert!(!nb.is_fitted);
    }

    #[test]
    fn test_spatial_naive_bayes_fit_predict() {
        let mut nb = SpatialNaiveBayes::new(SpatialNBConfig::default());

        let image1 = create_test_image(16, 16, TestPattern::Gradient);
        let image2 = create_test_image(16, 16, TestPattern::Checkerboard);

        let images = vec![image1.clone(), image2.clone()];
        let labels = Array1::from_vec(vec![0, 1]);

        nb.fit(&images, &labels).unwrap();
        assert!(nb.is_fitted);

        let predictions = nb.predict(&images).unwrap();
        assert_eq!(predictions.len(), 2);
    }

    #[test]
    fn test_color_space_conversions() {
        let rgb_image = create_test_image(10, 10, TestPattern::Gradient);

        let gray_image = rgb_to_grayscale(&rgb_image).unwrap();
        assert_eq!(gray_image.metadata.channels, 1);
        assert_eq!(gray_image.metadata.color_space, ColorSpace::Grayscale);

        let hsv_image = rgb_to_hsv(&rgb_image).unwrap();
        assert_eq!(hsv_image.metadata.channels, 3);
        assert_eq!(hsv_image.metadata.color_space, ColorSpace::HSV);
    }

    #[test]
    fn test_extract_color_histogram() {
        let config = ImageNBConfig::default();
        let nb = ImageNaiveBayes::new(config);
        let image = create_test_image(10, 10, TestPattern::Random);

        let histogram = nb.extract_color_histogram(&image).unwrap();
        assert_eq!(histogram.dim(), (3, 64)); // 3 channels, 64 bins

        // Check histogram is normalized
        let total: f64 = histogram.sum();
        assert_abs_diff_eq!(total, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_extract_spatial_features() {
        let nb = ImageNaiveBayes::new(ImageNBConfig::default());
        let image = create_test_image(16, 16, TestPattern::Gradient);

        let features = nb.extract_spatial_features(&image).unwrap();
        assert!(features.len() > 0);
    }

    #[test]
    fn test_extract_texture_features() {
        let nb = ImageNaiveBayes::new(ImageNBConfig::default());
        let image = create_test_image(16, 16, TestPattern::Checkerboard);

        let features = nb.extract_texture_features(&image).unwrap();
        assert!(features.len() > 0);
    }

    #[test]
    fn test_pyramid_features() {
        let mut config = ImageNBConfig::default();
        config.pyramid_levels = 2;
        let nb = ImageNaiveBayes::new(config);
        let image = create_test_image(32, 32, TestPattern::Gradient);

        let pyramid_features = nb.extract_pyramid_features(&image).unwrap();
        assert_eq!(pyramid_features.len(), 2);
    }

    #[test]
    fn test_image_downsampling() {
        let nb = ImageNaiveBayes::new(ImageNBConfig::default());
        let image = create_test_image(16, 16, TestPattern::Gradient);

        let downsampled = nb.downsample_image(&image, 8, 8).unwrap();
        assert_eq!(downsampled.dim(), (8, 8, 3));
    }

    #[test]
    fn test_gradient_computation() {
        let nb = ImageNaiveBayes::new(ImageNBConfig::default());
        let image = create_test_image(10, 10, TestPattern::Gradient);
        let channel_data = image.pixels.slice(s![.., .., 0]);

        let (grad_x, grad_y) = nb.compute_gradients(&channel_data);
        assert_eq!(grad_x.dim(), (10, 10));
        assert_eq!(grad_y.dim(), (10, 10));
    }

    #[test]
    fn test_simple_lbp() {
        let nb = ImageNaiveBayes::new(ImageNBConfig::default());
        let image = create_test_image(10, 10, TestPattern::Checkerboard);
        let channel_data = image.pixels.slice(s![.., .., 0]);

        let lbp_features = nb.compute_simple_lbp(&channel_data);
        assert_eq!(lbp_features.len(), 256);

        // Check normalization
        let total: f64 = lbp_features.iter().sum();
        assert_abs_diff_eq!(total, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_spatial_grid_features() {
        let config = SpatialNBConfig::default();
        let nb = SpatialNaiveBayes::new(config.clone());
        let image = create_test_image(16, 16, TestPattern::Gradient);

        let features = nb.extract_spatial_grid_features(&image).unwrap();
        assert_eq!(features.dim(), config.grid_size);
    }

    #[test]
    fn test_error_handling() {
        let mut nb = ImageNaiveBayes::new(ImageNBConfig::default());
        let image = create_test_image(10, 10, TestPattern::Random);
        let empty_labels = Array1::from_vec(vec![]);

        // Test mismatched dimensions
        let result = nb.fit(&vec![image], &empty_labels);
        assert!(result.is_err());

        // Test prediction before fitting
        let prediction_result = nb.predict(&vec![]);
        assert!(prediction_result.is_err());
    }
}
