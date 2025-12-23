//! Computer Vision Interpretability Methods
//!
//! This module provides specialized interpretability methods for computer vision models,
//! including image-specific LIME, Grad-CAM visualizations, saliency maps, object detection
//! explanations, and segmentation explanations.

use crate::{types::Float, SklResult, SklearsError};
// ✅ SciRS2 Policy Compliant Import
use scirs2_core::ndarray::{Array1, Array2, Array3};
use scirs2_core::random::Rng;
use std::collections::HashMap;

/// Configuration for computer vision interpretability methods
#[derive(Debug, Clone)]
pub struct ComputerVisionConfig {
    /// Number of superpixels for LIME segmentation
    pub n_superpixels: usize,
    /// Number of perturbations for LIME
    pub n_perturbations: usize,
    /// Smoothing parameter for Grad-CAM
    pub gradcam_smoothing: Float,
    /// Resolution for saliency maps
    pub saliency_resolution: (usize, usize),
    /// Threshold for object detection
    pub detection_threshold: Float,
    /// Minimum segment size for segmentation
    pub min_segment_size: usize,
    /// Noise level for perturbations
    pub noise_level: Float,
}

impl Default for ComputerVisionConfig {
    fn default() -> Self {
        Self {
            n_superpixels: 100,
            n_perturbations: 1000,
            gradcam_smoothing: 1.0,
            saliency_resolution: (224, 224),
            detection_threshold: 0.5,
            min_segment_size: 50,
            noise_level: 0.1,
        }
    }
}

/// Image with metadata for CV explanations
#[derive(Debug, Clone)]
pub struct Image {
    /// Image data (height, width, channels)
    pub data: Array3<Float>,
    /// Image width
    pub width: usize,
    /// Image height
    pub height: usize,
    /// Number of channels
    pub channels: usize,
    /// Image format (e.g., "RGB", "BGR", "Grayscale")
    pub format: String,
}

/// Superpixel segment
#[derive(Debug, Clone)]
pub struct Superpixel {
    /// Segment identifier
    pub id: usize,
    /// Pixel coordinates in the segment
    pub pixels: Vec<(usize, usize)>,
    /// Mean color of the segment
    pub mean_color: Array1<Float>,
    /// Segment centroid
    pub centroid: (Float, Float),
    /// Segment area (number of pixels)
    pub area: usize,
}

/// Image LIME explanation result
#[derive(Debug, Clone)]
pub struct ImageLimeResult {
    /// Original image
    pub image: Image,
    /// Superpixel segmentation
    pub superpixels: Vec<Superpixel>,
    /// Importance scores for each superpixel
    pub importance_scores: Array1<Float>,
    /// Explanation mask (same size as image)
    pub explanation_mask: Array2<Float>,
    /// Positive and negative contributions
    pub positive_mask: Array2<Float>,
    /// negative_mask
    pub negative_mask: Array2<Float>,
}

/// Grad-CAM explanation result
#[derive(Debug, Clone)]
pub struct GradCAMResult {
    /// Original image
    pub image: Image,
    /// Heatmap showing important regions
    pub heatmap: Array2<Float>,
    /// Guided Grad-CAM result
    pub guided_gradcam: Option<Array3<Float>>,
    /// Class activation map
    pub class_activation: Array2<Float>,
    /// Target class index
    pub target_class: usize,
    /// Activation statistics
    pub activation_stats: GradCAMStats,
}

/// Grad-CAM statistics
#[derive(Debug, Clone)]
pub struct GradCAMStats {
    /// Maximum activation value
    pub max_activation: Float,
    /// Mean activation value
    pub mean_activation: Float,
    /// Standard deviation of activations
    pub std_activation: Float,
    /// Percentage of image with high activation
    pub high_activation_percentage: Float,
}

/// Saliency map result
#[derive(Debug, Clone)]
pub struct SaliencyMapResult {
    /// Original image
    pub image: Image,
    /// Saliency map
    pub saliency_map: Array2<Float>,
    /// Integrated gradients
    pub integrated_gradients: Option<Array3<Float>>,
    /// Smooth gradients
    pub smooth_gradients: Option<Array3<Float>>,
    /// Method used for saliency computation
    pub method: SaliencyMethod,
}

/// Saliency computation methods
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SaliencyMethod {
    /// Vanilla gradients
    Vanilla,
    /// Integrated gradients
    IntegratedGradients,
    /// SmoothGrad
    SmoothGrad,
    /// Guided backpropagation
    GuidedBackprop,
}

/// Object detection explanation
#[derive(Debug, Clone)]
pub struct ObjectDetectionExplanation {
    /// Original image
    pub image: Image,
    /// Detected objects with explanations
    pub detections: Vec<DetectedObject>,
    /// Global attention map
    pub attention_map: Array2<Float>,
    /// Feature importance for detection
    pub feature_importance: Array2<Float>,
}

/// Detected object with explanation
#[derive(Debug, Clone)]
pub struct DetectedObject {
    /// Bounding box (x, y, width, height)
    pub bbox: (Float, Float, Float, Float),
    /// Object class
    pub class: String,
    /// Confidence score
    pub confidence: Float,
    /// Local explanation for this detection
    pub local_explanation: Array2<Float>,
    /// Important features for this object
    pub key_features: Vec<KeyFeature>,
}

/// Key feature for object detection
#[derive(Debug, Clone)]
pub struct KeyFeature {
    /// Feature name/description
    pub name: String,
    /// Feature location (x, y)
    pub location: (Float, Float),
    /// Feature importance score
    pub importance: Float,
    /// Feature size
    pub size: (Float, Float),
}

/// Segmentation explanation
#[derive(Debug, Clone)]
pub struct SegmentationExplanation {
    /// Original image
    pub image: Image,
    /// Segmentation mask
    pub segmentation_mask: Array2<usize>,
    /// Per-pixel explanations
    pub pixel_explanations: Array2<Float>,
    /// Segment-level explanations
    pub segments: Vec<SegmentExplanation>,
    /// Boundary explanations
    pub boundary_importance: Array2<Float>,
}

/// Explanation for a single segment
#[derive(Debug, Clone)]
pub struct SegmentExplanation {
    /// Segment identifier
    pub segment_id: usize,
    /// Segment class/label
    pub class: String,
    /// Segment confidence
    pub confidence: Float,
    /// Pixels in this segment
    pub pixels: Vec<(usize, usize)>,
    /// Segment-level importance
    pub importance: Float,
    /// Key visual features
    pub key_features: Vec<String>,
}

/// Image-specific LIME explanation
pub fn explain_image_with_lime<F>(
    image: &Image,
    model_fn: F,
    config: &ComputerVisionConfig,
) -> SklResult<ImageLimeResult>
where
    F: Fn(&Array3<Float>) -> SklResult<Array1<Float>>,
{
    // Generate superpixel segmentation
    let superpixels = generate_superpixels(image, config)?;

    // Generate perturbations by masking different superpixels
    let mut perturbations = Vec::new();
    let mut labels = Vec::new();

    // Original prediction
    let original_pred = model_fn(&image.data)?;

    // Generate perturbations
    for _ in 0..config.n_perturbations {
        let (perturbed_image, active_superpixels) =
            generate_image_perturbation(image, &superpixels, 0.5)?;
        let pred = model_fn(&perturbed_image)?;

        perturbations.push(active_superpixels);
        labels.push(pred);
    }

    // Compute importance scores
    let importance_scores = compute_lime_weights_image(&perturbations, &labels, &original_pred)?;

    // Generate explanation masks
    let (explanation_mask, positive_mask, negative_mask) =
        generate_explanation_masks(image, &superpixels, &importance_scores)?;

    Ok(ImageLimeResult {
        image: image.clone(),
        superpixels,
        importance_scores,
        explanation_mask,
        positive_mask,
        negative_mask,
    })
}

/// Generate Grad-CAM visualization
pub fn generate_gradcam<F>(
    image: &Image,
    model_fn: F,
    target_class: usize,
    config: &ComputerVisionConfig,
) -> SklResult<GradCAMResult>
where
    F: Fn(&Array3<Float>) -> SklResult<(Array1<Float>, Array3<Float>)>, // Returns (predictions, feature_maps)
{
    // Get model predictions and feature maps
    let (predictions, feature_maps) = model_fn(&image.data)?;

    if target_class >= predictions.len() {
        return Err(SklearsError::InvalidInput(
            "Target class index out of range".to_string(),
        ));
    }

    // Compute gradients (simplified - in practice would use automatic differentiation)
    let gradients = compute_gradients(&predictions, &feature_maps, target_class)?;

    // Generate class activation map
    let class_activation = compute_class_activation_map(&feature_maps, &gradients)?;

    // Apply smoothing
    let heatmap = apply_smoothing(&class_activation, config.gradcam_smoothing)?;

    // Compute statistics
    let activation_stats = compute_gradcam_stats(&heatmap)?;

    Ok(GradCAMResult {
        image: image.clone(),
        heatmap,
        guided_gradcam: None,
        class_activation,
        target_class,
        activation_stats,
    })
}

/// Generate saliency map
pub fn generate_saliency_map<F>(
    image: &Image,
    model_fn: F,
    method: SaliencyMethod,
    config: &ComputerVisionConfig,
) -> SklResult<SaliencyMapResult>
where
    F: Fn(&Array3<Float>) -> SklResult<Array1<Float>>,
{
    let saliency_map = match method {
        SaliencyMethod::Vanilla => compute_vanilla_gradients(image, &model_fn)?,
        SaliencyMethod::IntegratedGradients => {
            compute_integrated_gradients(image, &model_fn, config)?
        }
        SaliencyMethod::SmoothGrad => compute_smooth_gradients(image, &model_fn, config)?,
        SaliencyMethod::GuidedBackprop => compute_guided_backprop(image, &model_fn)?,
    };

    Ok(SaliencyMapResult {
        image: image.clone(),
        saliency_map,
        integrated_gradients: None,
        smooth_gradients: None,
        method,
    })
}

/// Explain object detection
pub fn explain_object_detection<F>(
    image: &Image,
    model_fn: F,
    config: &ComputerVisionConfig,
) -> SklResult<ObjectDetectionExplanation>
where
    F: Fn(&Array3<Float>) -> SklResult<Vec<(Float, Float, Float, Float, String, Float)>>, // Returns (x, y, w, h, class, confidence)
{
    // Get detections
    let detections_raw = model_fn(&image.data)?;

    // Filter detections by confidence threshold
    let filtered_detections: Vec<_> = detections_raw
        .into_iter()
        .filter(|(_, _, _, _, _, conf)| *conf >= config.detection_threshold)
        .collect();

    // Generate explanations for each detection
    let mut detections = Vec::new();
    for (x, y, w, h, class, confidence) in filtered_detections {
        let bbox = (x, y, w, h);
        let local_explanation = generate_detection_explanation(image, &bbox, config)?;
        let key_features = extract_key_features(image, &bbox, &local_explanation)?;

        detections.push(DetectedObject {
            bbox,
            class,
            confidence,
            local_explanation,
            key_features,
        });
    }

    // Generate global attention map
    let attention_map = generate_global_attention_map(image, &detections)?;

    // Compute feature importance
    let feature_importance = compute_detection_feature_importance(image, &detections)?;

    Ok(ObjectDetectionExplanation {
        image: image.clone(),
        detections,
        attention_map,
        feature_importance,
    })
}

/// Explain image segmentation
pub fn explain_segmentation<F>(
    image: &Image,
    model_fn: F,
    config: &ComputerVisionConfig,
) -> SklResult<SegmentationExplanation>
where
    F: Fn(&Array3<Float>) -> SklResult<(Array2<usize>, Array1<Float>)>, // Returns (segmentation, confidence)
{
    // Get segmentation
    let (segmentation_mask, confidence) = model_fn(&image.data)?;

    // Generate per-pixel explanations
    let pixel_explanations = generate_pixel_explanations(image, &segmentation_mask, config)?;

    // Extract segments
    let segments = extract_segments(&segmentation_mask, &pixel_explanations, &confidence, config)?;

    // Compute boundary importance
    let boundary_importance = compute_boundary_importance(&segmentation_mask, &pixel_explanations)?;

    Ok(SegmentationExplanation {
        image: image.clone(),
        segmentation_mask,
        pixel_explanations,
        segments,
        boundary_importance,
    })
}

// Helper functions

/// Generate superpixels using simple grid-based segmentation
fn generate_superpixels(
    image: &Image,
    config: &ComputerVisionConfig,
) -> SklResult<Vec<Superpixel>> {
    let mut superpixels = Vec::new();

    // Simple grid-based superpixels (in practice, would use SLIC or similar)
    let grid_size = (config.n_superpixels as Float).sqrt() as usize;
    let step_x = image.width / grid_size;
    let step_y = image.height / grid_size;

    let mut id = 0;
    for row in 0..grid_size {
        for col in 0..grid_size {
            let start_x = col * step_x;
            let end_x = ((col + 1) * step_x).min(image.width);
            let start_y = row * step_y;
            let end_y = ((row + 1) * step_y).min(image.height);

            let mut pixels = Vec::new();
            let mut color_sum = Array1::zeros(image.channels);

            for y in start_y..end_y {
                for x in start_x..end_x {
                    pixels.push((x, y));
                    if y < image.data.shape()[0] && x < image.data.shape()[1] {
                        for c in 0..image.channels {
                            if c < image.data.shape()[2] {
                                color_sum[c] += image.data[[y, x, c]];
                            }
                        }
                    }
                }
            }

            let mean_color = if !pixels.is_empty() {
                color_sum / pixels.len() as Float
            } else {
                Array1::zeros(image.channels)
            };

            let centroid = (
                (start_x + end_x) as Float / 2.0,
                (start_y + end_y) as Float / 2.0,
            );

            superpixels.push(Superpixel {
                id,
                pixels,
                mean_color,
                centroid,
                area: (end_x - start_x) * (end_y - start_y),
            });

            id += 1;
        }
    }

    Ok(superpixels)
}

/// Generate image perturbation by masking superpixels
fn generate_image_perturbation(
    image: &Image,
    superpixels: &[Superpixel],
    mask_probability: Float,
) -> SklResult<(Array3<Float>, Vec<bool>)> {
    let mut perturbed = image.data.clone();
    let mut active_superpixels = vec![false; superpixels.len()];

    for (i, superpixel) in superpixels.iter().enumerate() {
        let is_active = scirs2_core::random::thread_rng().random::<Float>() > mask_probability;
        active_superpixels[i] = is_active;

        if !is_active {
            // Mask this superpixel (set to mean color or zero)
            for &(x, y) in &superpixel.pixels {
                if y < perturbed.shape()[0] && x < perturbed.shape()[1] {
                    for c in 0..perturbed.shape()[2] {
                        perturbed[[y, x, c]] = 0.0; // or superpixel.mean_color[c]
                    }
                }
            }
        }
    }

    Ok((perturbed, active_superpixels))
}

/// Compute LIME weights for image explanation
fn compute_lime_weights_image(
    perturbations: &[Vec<bool>],
    predictions: &[Array1<Float>],
    original_pred: &Array1<Float>,
) -> SklResult<Array1<Float>> {
    if perturbations.is_empty() || predictions.is_empty() {
        return Err(SklearsError::InvalidInput(
            "No perturbations provided".to_string(),
        ));
    }

    let n_superpixels = perturbations[0].len();
    let mut weights = Array1::zeros(n_superpixels);

    // Simple correlation-based weights
    for j in 0..n_superpixels {
        let mut correlation = 0.0;
        let mut count = 0;

        for (i, pred) in predictions.iter().enumerate() {
            if i < perturbations.len() {
                let feature_active = perturbations[i][j] as usize as Float;
                let target_change = (pred[0] - original_pred[0]).abs();
                correlation += feature_active * target_change;
                count += 1;
            }
        }

        if count > 0 {
            weights[j] = correlation / count as Float;
        }
    }

    Ok(weights)
}

/// Generate explanation masks from superpixels and importance scores
fn generate_explanation_masks(
    image: &Image,
    superpixels: &[Superpixel],
    importance_scores: &Array1<Float>,
) -> SklResult<(Array2<Float>, Array2<Float>, Array2<Float>)> {
    let mut explanation_mask = Array2::zeros((image.height, image.width));
    let mut positive_mask = Array2::zeros((image.height, image.width));
    let mut negative_mask = Array2::zeros((image.height, image.width));

    for (i, superpixel) in superpixels.iter().enumerate() {
        let importance = importance_scores[i];

        for &(x, y) in &superpixel.pixels {
            if y < explanation_mask.shape()[0] && x < explanation_mask.shape()[1] {
                explanation_mask[[y, x]] = importance;

                if importance > 0.0 {
                    positive_mask[[y, x]] = importance;
                } else {
                    negative_mask[[y, x]] = importance.abs();
                }
            }
        }
    }

    Ok((explanation_mask, positive_mask, negative_mask))
}

/// Compute gradients (simplified implementation)
fn compute_gradients(
    predictions: &Array1<Float>,
    feature_maps: &Array3<Float>,
    target_class: usize,
) -> SklResult<Array3<Float>> {
    // Simplified gradient computation
    // In practice, this would use automatic differentiation
    let mut gradients = Array3::zeros(feature_maps.raw_dim());

    if target_class < predictions.len() {
        let target_score = predictions[target_class];

        // Simple approximation: gradients proportional to feature maps and target score
        for ((i, j, k), &feature_val) in feature_maps.indexed_iter() {
            gradients[[i, j, k]] = feature_val * target_score;
        }
    }

    Ok(gradients)
}

/// Compute class activation map
fn compute_class_activation_map(
    feature_maps: &Array3<Float>,
    gradients: &Array3<Float>,
) -> SklResult<Array2<Float>> {
    if feature_maps.shape() != gradients.shape() {
        return Err(SklearsError::InvalidInput(
            "Feature maps and gradients shape mismatch".to_string(),
        ));
    }

    let (height, width, _) = feature_maps.dim();
    let mut activation_map = Array2::zeros((height, width));

    // Global average pooling of gradients for weights
    let mut weights = Array1::zeros(feature_maps.shape()[2]);
    for k in 0..feature_maps.shape()[2] {
        let mut sum = 0.0;
        let mut count = 0;

        for i in 0..height {
            for j in 0..width {
                sum += gradients[[i, j, k]];
                count += 1;
            }
        }

        weights[k] = if count > 0 { sum / count as Float } else { 0.0 };
    }

    // Weighted combination of feature maps
    for i in 0..height {
        for j in 0..width {
            let mut activation: Float = 0.0;
            for k in 0..feature_maps.shape()[2] {
                activation += weights[k] * feature_maps[[i, j, k]];
            }
            activation_map[[i, j]] = activation.max(0.0); // ReLU
        }
    }

    Ok(activation_map)
}

/// Apply smoothing to heatmap
fn apply_smoothing(heatmap: &Array2<Float>, smoothing: Float) -> SklResult<Array2<Float>> {
    // Simple Gaussian-like smoothing
    let mut smoothed = heatmap.clone();

    if smoothing > 0.0 {
        let (height, width) = heatmap.dim();

        for i in 1..(height - 1) {
            for j in 1..(width - 1) {
                let mut sum = 0.0;
                let mut count = 0;

                // 3x3 kernel
                for di in -1i32..=1 {
                    for dj in -1i32..=1 {
                        let ni = (i as i32 + di) as usize;
                        let nj = (j as i32 + dj) as usize;

                        if ni < height && nj < width {
                            sum += heatmap[[ni, nj]];
                            count += 1;
                        }
                    }
                }

                if count > 0 {
                    smoothed[[i, j]] = sum / count as Float;
                }
            }
        }
    }

    Ok(smoothed)
}

/// Compute Grad-CAM statistics
fn compute_gradcam_stats(heatmap: &Array2<Float>) -> SklResult<GradCAMStats> {
    let values: Vec<Float> = heatmap.iter().cloned().collect();

    if values.is_empty() {
        return Ok(GradCAMStats {
            max_activation: 0.0,
            mean_activation: 0.0,
            std_activation: 0.0,
            high_activation_percentage: 0.0,
        });
    }

    let max_activation = values.iter().cloned().fold(Float::NEG_INFINITY, Float::max);
    let mean_activation = values.iter().sum::<Float>() / values.len() as Float;

    let variance = values
        .iter()
        .map(|&x| (x - mean_activation).powi(2))
        .sum::<Float>()
        / values.len() as Float;
    let std_activation = variance.sqrt();

    let threshold = mean_activation + std_activation;
    let high_count = values.iter().filter(|&&x| x > threshold).count();
    let high_activation_percentage = high_count as Float / values.len() as Float;

    Ok(GradCAMStats {
        max_activation,
        mean_activation,
        std_activation,
        high_activation_percentage,
    })
}

/// Compute vanilla gradients
fn compute_vanilla_gradients<F>(image: &Image, model_fn: F) -> SklResult<Array2<Float>>
where
    F: Fn(&Array3<Float>) -> SklResult<Array1<Float>>,
{
    // Simplified gradient computation
    let (height, width) = (image.height, image.width);
    let mut gradients = Array2::zeros((height, width));

    let original_pred = model_fn(&image.data)?;
    let epsilon = 1e-4;

    // Finite difference approximation
    for i in 0..height {
        for j in 0..width {
            let mut perturbed = image.data.clone();

            // Perturb pixel
            for c in 0..image.channels {
                if c < perturbed.shape()[2] {
                    perturbed[[i, j, c]] += epsilon;
                }
            }

            let perturbed_pred = model_fn(&perturbed)?;
            let gradient = (perturbed_pred[0] - original_pred[0]) / epsilon;
            gradients[[i, j]] = gradient;
        }
    }

    Ok(gradients)
}

/// Compute integrated gradients
fn compute_integrated_gradients<F>(
    image: &Image,
    model_fn: F,
    config: &ComputerVisionConfig,
) -> SklResult<Array2<Float>>
where
    F: Fn(&Array3<Float>) -> SklResult<Array1<Float>>,
{
    let n_steps = 50;
    let baseline = Array3::<Float>::zeros(image.data.raw_dim());

    let mut integrated_gradients = Array2::zeros((image.height, image.width));

    for step in 0..n_steps {
        let alpha = step as Float / n_steps as Float;

        // Interpolate between baseline and input
        let interpolated = &baseline + alpha * (&image.data - &baseline);

        // Compute gradients at this point
        let gradients = compute_vanilla_gradients(
            &Image {
                data: interpolated,
                width: image.width,
                height: image.height,
                channels: image.channels,
                format: image.format.clone(),
            },
            &model_fn,
        )?;

        integrated_gradients = integrated_gradients + gradients;
    }

    // Average and multiply by (input - baseline)
    integrated_gradients /= n_steps as Float;

    Ok(integrated_gradients)
}

/// Compute smooth gradients
fn compute_smooth_gradients<F>(
    image: &Image,
    model_fn: F,
    config: &ComputerVisionConfig,
) -> SklResult<Array2<Float>>
where
    F: Fn(&Array3<Float>) -> SklResult<Array1<Float>>,
{
    let n_samples = 50;
    let mut smooth_gradients = Array2::zeros((image.height, image.width));

    for _ in 0..n_samples {
        // Add noise to image
        let mut noisy_image = image.data.clone();
        for elem in noisy_image.iter_mut() {
            *elem +=
                (scirs2_core::random::thread_rng().random::<Float>() - 0.5) * config.noise_level;
        }

        // Compute gradients
        let gradients = compute_vanilla_gradients(
            &Image {
                data: noisy_image,
                width: image.width,
                height: image.height,
                channels: image.channels,
                format: image.format.clone(),
            },
            &model_fn,
        )?;

        smooth_gradients = smooth_gradients + gradients;
    }

    smooth_gradients /= n_samples as Float;
    Ok(smooth_gradients)
}

/// Compute guided backpropagation
fn compute_guided_backprop<F>(image: &Image, model_fn: F) -> SklResult<Array2<Float>>
where
    F: Fn(&Array3<Float>) -> SklResult<Array1<Float>>,
{
    // Simplified - in practice would need access to intermediate activations
    compute_vanilla_gradients(image, model_fn)
}

/// Generate detection explanation for a bounding box
fn generate_detection_explanation(
    image: &Image,
    bbox: &(Float, Float, Float, Float),
    config: &ComputerVisionConfig,
) -> SklResult<Array2<Float>> {
    let (x, y, w, h) = *bbox;
    let mut explanation = Array2::zeros((image.height, image.width));

    // Simple box-based explanation
    let x_start = (x as usize).min(image.width);
    let y_start = (y as usize).min(image.height);
    let x_end = ((x + w) as usize).min(image.width);
    let y_end = ((y + h) as usize).min(image.height);

    for i in y_start..y_end {
        for j in x_start..x_end {
            explanation[[i, j]] = 1.0;
        }
    }

    Ok(explanation)
}

/// Extract key features from bounding box region
fn extract_key_features(
    image: &Image,
    bbox: &(Float, Float, Float, Float),
    explanation: &Array2<Float>,
) -> SklResult<Vec<KeyFeature>> {
    let mut features = Vec::new();

    let (x, y, w, h) = *bbox;

    // Simple feature: center of bounding box
    features.push(KeyFeature {
        name: "Center".to_string(),
        location: (x + w / 2.0, y + h / 2.0),
        importance: 1.0,
        size: (w, h),
    });

    // Add corners as features
    features.push(KeyFeature {
        name: "Top-left corner".to_string(),
        location: (x, y),
        importance: 0.8,
        size: (w * 0.1, h * 0.1),
    });

    features.push(KeyFeature {
        name: "Bottom-right corner".to_string(),
        location: (x + w, y + h),
        importance: 0.8,
        size: (w * 0.1, h * 0.1),
    });

    Ok(features)
}

/// Generate global attention map from all detections
fn generate_global_attention_map(
    image: &Image,
    detections: &[DetectedObject],
) -> SklResult<Array2<Float>> {
    let mut attention_map = Array2::zeros((image.height, image.width));

    for detection in detections {
        // Add local explanation weighted by confidence
        for i in 0..image.height {
            for j in 0..image.width {
                if i < detection.local_explanation.shape()[0]
                    && j < detection.local_explanation.shape()[1]
                {
                    attention_map[[i, j]] +=
                        detection.local_explanation[[i, j]] * detection.confidence;
                }
            }
        }
    }

    Ok(attention_map)
}

/// Compute feature importance for object detection
fn compute_detection_feature_importance(
    image: &Image,
    detections: &[DetectedObject],
) -> SklResult<Array2<Float>> {
    let mut feature_importance = Array2::zeros((image.height, image.width));

    // Compute importance based on overlap with detection regions
    for detection in detections {
        let (x, y, w, h) = detection.bbox;
        let x_start = (x as usize).min(image.width);
        let y_start = (y as usize).min(image.height);
        let x_end = ((x + w) as usize).min(image.width);
        let y_end = ((y + h) as usize).min(image.height);

        for i in y_start..y_end {
            for j in x_start..x_end {
                feature_importance[[i, j]] += detection.confidence;
            }
        }
    }

    Ok(feature_importance)
}

/// Generate per-pixel explanations for segmentation
fn generate_pixel_explanations(
    image: &Image,
    segmentation: &Array2<usize>,
    config: &ComputerVisionConfig,
) -> SklResult<Array2<Float>> {
    let mut explanations = Array2::zeros((image.height, image.width));

    // Simple explanation based on segment membership
    for i in 0..image.height {
        for j in 0..image.width {
            if i < segmentation.shape()[0] && j < segmentation.shape()[1] {
                let segment_id = segmentation[[i, j]];
                explanations[[i, j]] = (segment_id as Float + 1.0) / 256.0; // Normalize
            }
        }
    }

    Ok(explanations)
}

/// Extract segments from segmentation mask
fn extract_segments(
    segmentation: &Array2<usize>,
    pixel_explanations: &Array2<Float>,
    confidence: &Array1<Float>,
    config: &ComputerVisionConfig,
) -> SklResult<Vec<SegmentExplanation>> {
    let mut segments = Vec::new();
    let mut segment_pixels: HashMap<usize, Vec<(usize, usize)>> = HashMap::new();

    // Group pixels by segment
    for ((i, j), &segment_id) in segmentation.indexed_iter() {
        segment_pixels.entry(segment_id).or_default().push((i, j));
    }

    // Create segment explanations
    for (segment_id, pixels) in segment_pixels {
        if pixels.len() >= config.min_segment_size {
            let importance = pixels
                .iter()
                .map(|&(i, j)| {
                    if i < pixel_explanations.shape()[0] && j < pixel_explanations.shape()[1] {
                        pixel_explanations[[i, j]]
                    } else {
                        0.0
                    }
                })
                .sum::<Float>()
                / pixels.len() as Float;

            let segment_confidence = if segment_id < confidence.len() {
                confidence[segment_id]
            } else {
                0.5
            };

            segments.push(SegmentExplanation {
                segment_id,
                class: format!("class_{}", segment_id),
                confidence: segment_confidence,
                pixels,
                importance,
                key_features: vec!["color".to_string(), "texture".to_string()],
            });
        }
    }

    Ok(segments)
}

/// Compute boundary importance for segmentation
fn compute_boundary_importance(
    segmentation: &Array2<usize>,
    pixel_explanations: &Array2<Float>,
) -> SklResult<Array2<Float>> {
    let (height, width) = segmentation.dim();
    let mut boundary_importance = Array2::zeros((height, width));

    // Detect boundaries using simple edge detection
    for i in 1..(height - 1) {
        for j in 1..(width - 1) {
            let center_segment = segmentation[[i, j]];
            let mut is_boundary = false;

            // Check 8-connected neighbors
            for di in -1i32..=1 {
                for dj in -1i32..=1 {
                    if di == 0 && dj == 0 {
                        continue;
                    }

                    let ni = (i as i32 + di) as usize;
                    let nj = (j as i32 + dj) as usize;

                    if ni < height && nj < width && segmentation[[ni, nj]] != center_segment {
                        is_boundary = true;
                        break;
                    }
                }
                if is_boundary {
                    break;
                }
            }

            if is_boundary {
                boundary_importance[[i, j]] = pixel_explanations[[i, j]];
            }
        }
    }

    Ok(boundary_importance)
}

#[cfg(test)]
mod tests {
    use super::*;
    // ✅ SciRS2 Policy Compliant Import
    use scirs2_core::ndarray::array;

    fn create_test_image() -> Image {
        Image {
            data: Array3::zeros((4, 4, 3)),
            width: 4,
            height: 4,
            channels: 3,
            format: "RGB".to_string(),
        }
    }

    #[test]
    fn test_computer_vision_config_default() {
        let config = ComputerVisionConfig::default();

        assert_eq!(config.n_superpixels, 100);
        assert_eq!(config.n_perturbations, 1000);
        assert_eq!(config.gradcam_smoothing, 1.0);
        assert_eq!(config.saliency_resolution, (224, 224));
        assert_eq!(config.detection_threshold, 0.5);
        assert_eq!(config.min_segment_size, 50);
        assert_eq!(config.noise_level, 0.1);
    }

    #[test]
    fn test_superpixel_generation() {
        let image = create_test_image();
        let config = ComputerVisionConfig {
            n_superpixels: 4,
            ..Default::default()
        };

        let superpixels = generate_superpixels(&image, &config).unwrap();

        assert_eq!(superpixels.len(), 4);
        for superpixel in &superpixels {
            assert!(!superpixel.pixels.is_empty());
            assert_eq!(superpixel.mean_color.len(), 3);
        }
    }

    #[test]
    fn test_image_perturbation() {
        let image = create_test_image();
        let config = ComputerVisionConfig {
            n_superpixels: 4,
            ..Default::default()
        };
        let superpixels = generate_superpixels(&image, &config).unwrap();

        let (perturbed, active) = generate_image_perturbation(&image, &superpixels, 0.5).unwrap();

        assert_eq!(perturbed.shape(), image.data.shape());
        assert_eq!(active.len(), superpixels.len());
    }

    #[test]
    fn test_gradcam_stats() {
        let heatmap = array![[0.1, 0.8], [0.3, 0.9]];
        let stats = compute_gradcam_stats(&heatmap).unwrap();

        assert_eq!(stats.max_activation, 0.9);
        assert!(stats.mean_activation > 0.0);
        assert!(stats.std_activation > 0.0);
        assert!(stats.high_activation_percentage >= 0.0);
        assert!(stats.high_activation_percentage <= 1.0);
    }

    #[test]
    fn test_smoothing() {
        let heatmap = array![[1.0, 0.0], [0.0, 1.0]];
        let smoothed = apply_smoothing(&heatmap, 1.0).unwrap();

        assert_eq!(smoothed.shape(), heatmap.shape());
        // Smoothed values should be different from original (except edges)
    }

    #[test]
    fn test_class_activation_map() {
        let feature_maps = Array3::ones((2, 2, 3));
        let gradients = Array3::ones((2, 2, 3));

        let activation_map = compute_class_activation_map(&feature_maps, &gradients).unwrap();

        assert_eq!(activation_map.shape(), &[2, 2]);
        // All values should be positive due to ReLU
        for &val in activation_map.iter() {
            assert!(val >= 0.0);
        }
    }

    #[test]
    fn test_explanation_mask_generation() {
        let image = create_test_image();
        let config = ComputerVisionConfig {
            n_superpixels: 4,
            ..Default::default()
        };
        let superpixels = generate_superpixels(&image, &config).unwrap();
        let importance_scores = Array1::from_vec(vec![0.5, -0.3, 0.8, -0.1]);

        let (explanation_mask, positive_mask, negative_mask) =
            generate_explanation_masks(&image, &superpixels, &importance_scores).unwrap();

        assert_eq!(explanation_mask.shape(), &[4, 4]);
        assert_eq!(positive_mask.shape(), &[4, 4]);
        assert_eq!(negative_mask.shape(), &[4, 4]);

        // Check that positive and negative masks are properly separated
        for i in 0..4 {
            for j in 0..4 {
                if positive_mask[[i, j]] > 0.0 {
                    assert_eq!(negative_mask[[i, j]], 0.0);
                }
                if negative_mask[[i, j]] > 0.0 {
                    assert_eq!(positive_mask[[i, j]], 0.0);
                }
            }
        }
    }

    #[test]
    fn test_key_feature_extraction() {
        let image = create_test_image();
        let bbox = (1.0, 1.0, 2.0, 2.0);
        let explanation = Array2::ones((4, 4));

        let features = extract_key_features(&image, &bbox, &explanation).unwrap();

        assert!(!features.is_empty());
        // Should have at least center feature
        assert!(features.iter().any(|f| f.name == "Center"));
    }

    #[test]
    fn test_boundary_importance_computation() {
        let segmentation = array![[0, 0, 1, 1], [0, 0, 1, 1], [2, 2, 3, 3], [2, 2, 3, 3]];
        let pixel_explanations = Array2::ones((4, 4));

        let boundary_importance =
            compute_boundary_importance(&segmentation, &pixel_explanations).unwrap();

        assert_eq!(boundary_importance.shape(), &[4, 4]);
        // Boundary pixels should have non-zero importance
        assert!(boundary_importance.sum() > 0.0);
    }

    #[test]
    fn test_image_lime_explanation() {
        let image = create_test_image();
        let config = ComputerVisionConfig {
            n_superpixels: 4,
            n_perturbations: 10,
            ..Default::default()
        };

        // Mock model function
        let model_fn = |image_data: &Array3<Float>| -> SklResult<Array1<Float>> {
            Ok(array![image_data.sum()])
        };

        let result = explain_image_with_lime(&image, model_fn, &config).unwrap();

        assert_eq!(result.superpixels.len(), 4);
        assert_eq!(result.importance_scores.len(), 4);
        assert_eq!(result.explanation_mask.shape(), &[4, 4]);
        assert_eq!(result.positive_mask.shape(), &[4, 4]);
        assert_eq!(result.negative_mask.shape(), &[4, 4]);
    }

    #[test]
    fn test_saliency_method_variants() {
        use SaliencyMethod::*;

        let methods = vec![Vanilla, IntegratedGradients, SmoothGrad, GuidedBackprop];

        // Test that each variant exists and can be compared
        assert_eq!(methods.len(), 4);

        for method in methods {
            match method {
                Vanilla => assert_eq!(method, Vanilla),
                IntegratedGradients => assert_eq!(method, IntegratedGradients),
                SmoothGrad => assert_eq!(method, SmoothGrad),
                GuidedBackprop => assert_eq!(method, GuidedBackprop),
            }
        }

        // Test inequality
        assert_ne!(Vanilla, IntegratedGradients);
        assert_ne!(SmoothGrad, GuidedBackprop);
    }
}
