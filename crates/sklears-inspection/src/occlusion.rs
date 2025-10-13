//! Occlusion and Masking Methods
//!
//! This module provides occlusion-based model inspection methods including
//! feature occlusion importance, integrated gradients, saliency maps, and
//! relevance propagation techniques.

// ✅ SciRS2 Policy Compliant Import
use scirs2_core::ndarray::{Array1, Array2, Array3, ArrayView2, Axis};
use scirs2_core::random::{Rng, SeedableRng};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    types::Float,
};

/// Configuration for occlusion-based analysis
#[derive(Debug, Clone)]
pub struct OcclusionConfig {
    /// Method for occlusion analysis
    pub method: OcclusionMethod,
    /// Occlusion strategy (how to occlude features)
    pub occlusion_strategy: OcclusionStrategy,
    /// Size of occlusion window (for patch-based occlusion)
    pub window_size: Option<(usize, usize)>,
    /// Stride for sliding window occlusion
    pub stride: Option<(usize, usize)>,
    /// Number of steps for integrated gradients
    pub n_steps: usize,
    /// Baseline values for integrated gradients
    pub baseline: Option<Array1<Float>>,
    /// Random state for reproducibility
    pub random_state: Option<u64>,
}

impl Default for OcclusionConfig {
    fn default() -> Self {
        Self {
            method: OcclusionMethod::FeatureOcclusion,
            occlusion_strategy: OcclusionStrategy::Zero,
            window_size: None,
            stride: None,
            n_steps: 50,
            baseline: None,
            random_state: Some(42),
        }
    }
}

/// Occlusion analysis methods
#[derive(Debug, Clone, Copy)]
pub enum OcclusionMethod {
    /// Simple feature occlusion importance
    FeatureOcclusion,
    /// Integrated gradients method
    IntegratedGradients,
    /// Saliency map generation
    SaliencyMaps,
    /// Guided backpropagation (approximated)
    GuidedBackpropagation,
    /// Layer-wise relevance propagation
    LayerWiseRelevancePropagation,
}

/// Strategies for occluding features
#[derive(Debug, Clone, Copy)]
pub enum OcclusionStrategy {
    /// Replace with zeros
    Zero,
    /// Replace with feature mean
    Mean,
    /// Replace with random noise
    Noise,
    /// Replace with baseline values
    Baseline,
    /// Remove features entirely (set to NaN)
    Remove,
}

/// Result of occlusion analysis
#[derive(Debug, Clone)]
pub struct OcclusionResult {
    /// Importance scores for each feature
    pub importance_scores: Array1<Float>,
    /// Saliency map (if applicable)
    pub saliency_map: Option<Array2<Float>>,
    /// Integrated gradients (if applicable)
    pub integrated_gradients: Option<Array2<Float>>,
    /// Attribution map for visualization
    pub attribution_map: Option<Array2<Float>>,
    /// Per-step attributions (for integrated gradients)
    pub step_attributions: Option<Array3<Float>>,
    /// Feature names
    pub feature_names: Option<Vec<String>>,
}

///
/// let model_fn = |x: &scirs2_core::ndarray::ArrayView2<f64>| -> Vec<f64> {
///     x.rows().into_iter()
///         .map(|row| row[0] * 2.0 + row[1] * 0.5)
///         .collect()
/// };
///
/// let X = array![[0.5, 0.7], [0.3, 0.9]];
///
/// let config = OcclusionConfig {
///     method: OcclusionMethod::FeatureOcclusion,
///     ..Default::default()
/// };
///
/// let result = analyze_occlusion(&model_fn, &X.view(), None, &config).unwrap();
/// assert_eq!(result.importance_scores.len(), 2);
/// ```
pub fn analyze_occlusion<F>(
    model_fn: &F,
    X: &ArrayView2<Float>,
    target_class: Option<usize>,
    config: &OcclusionConfig,
) -> SklResult<OcclusionResult>
where
    F: Fn(&ArrayView2<Float>) -> Vec<Float>,
{
    match config.method {
        OcclusionMethod::FeatureOcclusion => {
            feature_occlusion_analysis(model_fn, X, target_class, config)
        }
        OcclusionMethod::IntegratedGradients => {
            integrated_gradients_analysis(model_fn, X, target_class, config)
        }
        OcclusionMethod::SaliencyMaps => saliency_map_analysis(model_fn, X, target_class, config),
        OcclusionMethod::GuidedBackpropagation => {
            guided_backpropagation_analysis(model_fn, X, target_class, config)
        }
        OcclusionMethod::LayerWiseRelevancePropagation => {
            lrp_analysis(model_fn, X, target_class, config)
        }
    }
}

/// Feature occlusion-based importance analysis
fn feature_occlusion_analysis<F>(
    model_fn: &F,
    X: &ArrayView2<Float>,
    _target_class: Option<usize>,
    config: &OcclusionConfig,
) -> SklResult<OcclusionResult>
where
    F: Fn(&ArrayView2<Float>) -> Vec<Float>,
{
    let (n_samples, n_features) = X.dim();
    let mut rng = match config.random_state {
        Some(seed) => scirs2_core::random::rngs::StdRng::seed_from_u64(seed),
        None => scirs2_core::random::rngs::StdRng::from_rng(&mut scirs2_core::random::thread_rng()),
    };

    // Get baseline predictions
    let baseline_predictions = model_fn(X);

    // Calculate feature means for mean occlusion strategy
    let feature_means = X.mean_axis(Axis(0)).unwrap();

    let mut importance_scores = Array1::zeros(n_features);

    // For each feature
    for feature_idx in 0..n_features {
        let mut total_importance = 0.0;

        // For each sample
        for sample_idx in 0..n_samples {
            let mut X_occluded = X.to_owned();

            // Apply occlusion strategy
            match config.occlusion_strategy {
                OcclusionStrategy::Zero => {
                    X_occluded[[sample_idx, feature_idx]] = 0.0;
                }
                OcclusionStrategy::Mean => {
                    X_occluded[[sample_idx, feature_idx]] = feature_means[feature_idx];
                }
                OcclusionStrategy::Noise => {
                    let noise_std = feature_means[feature_idx] * 0.1; // 10% of mean as noise
                    X_occluded[[sample_idx, feature_idx]] = rng.gen_range(-noise_std..=noise_std);
                }
                OcclusionStrategy::Baseline => {
                    if let Some(ref baseline) = config.baseline {
                        X_occluded[[sample_idx, feature_idx]] = baseline[feature_idx];
                    } else {
                        X_occluded[[sample_idx, feature_idx]] = 0.0;
                    }
                }
                OcclusionStrategy::Remove => {
                    X_occluded[[sample_idx, feature_idx]] = Float::NAN;
                }
            }

            let occluded_predictions = model_fn(&X_occluded.view());
            let importance =
                (baseline_predictions[sample_idx] - occluded_predictions[sample_idx]).abs();
            total_importance += importance;
        }

        importance_scores[feature_idx] = total_importance / n_samples as Float;
    }

    Ok(OcclusionResult {
        importance_scores,
        saliency_map: None,
        integrated_gradients: None,
        attribution_map: None,
        step_attributions: None,
        feature_names: None,
    })
}

/// Integrated gradients analysis
fn integrated_gradients_analysis<F>(
    model_fn: &F,
    X: &ArrayView2<Float>,
    _target_class: Option<usize>,
    config: &OcclusionConfig,
) -> SklResult<OcclusionResult>
where
    F: Fn(&ArrayView2<Float>) -> Vec<Float>,
{
    let (n_samples, n_features) = X.dim();
    let n_steps = config.n_steps;

    // Create baseline (zeros if not provided)
    let baseline = match &config.baseline {
        Some(b) => b.clone(),
        None => Array1::zeros(n_features),
    };

    let mut integrated_gradients = Array2::zeros((n_samples, n_features));

    // For each sample
    for sample_idx in 0..n_samples {
        let x_sample = X.row(sample_idx);
        let mut gradients_sum = Array1::zeros(n_features);

        // Compute gradients along the path from baseline to input
        for step in 0..n_steps {
            let alpha = step as Float / (n_steps - 1) as Float;

            // Interpolate between baseline and input
            let mut x_interpolated = Array1::zeros(n_features);
            for j in 0..n_features {
                x_interpolated[j] = baseline[j] + alpha * (x_sample[j] - baseline[j]);
            }

            // Compute numerical gradients at this point
            let gradients = compute_numerical_gradients(model_fn, &x_interpolated, 0.01);
            gradients_sum += &gradients;
        }

        // Scale by (x - baseline) and normalize by number of steps
        for j in 0..n_features {
            integrated_gradients[[sample_idx, j]] =
                (x_sample[j] - baseline[j]) * gradients_sum[j] / n_steps as Float;
        }
    }

    // Compute importance scores as mean absolute integrated gradients
    let importance_scores = integrated_gradients.map_axis(Axis(0), |col| {
        col.iter().map(|&x| x.abs()).sum::<Float>() / n_samples as Float
    });

    Ok(OcclusionResult {
        importance_scores,
        saliency_map: None,
        integrated_gradients: Some(integrated_gradients),
        attribution_map: None,
        step_attributions: None,
        feature_names: None,
    })
}

/// Saliency map analysis
fn saliency_map_analysis<F>(
    model_fn: &F,
    X: &ArrayView2<Float>,
    _target_class: Option<usize>,
    config: &OcclusionConfig,
) -> SklResult<OcclusionResult>
where
    F: Fn(&ArrayView2<Float>) -> Vec<Float>,
{
    let (n_samples, n_features) = X.dim();
    let mut saliency_map = Array2::zeros((n_samples, n_features));

    // For each sample, compute gradients (saliency)
    for sample_idx in 0..n_samples {
        let x_sample = X.row(sample_idx).to_owned();
        let gradients = compute_numerical_gradients(model_fn, &x_sample, 0.001);

        for j in 0..n_features {
            saliency_map[[sample_idx, j]] = gradients[j].abs();
        }
    }

    // Compute importance scores as mean saliency
    let importance_scores = saliency_map.mean_axis(Axis(0)).unwrap();

    Ok(OcclusionResult {
        importance_scores,
        saliency_map: Some(saliency_map.clone()),
        integrated_gradients: None,
        attribution_map: Some(saliency_map),
        step_attributions: None,
        feature_names: None,
    })
}

/// Guided backpropagation analysis (approximated using modified gradients)
fn guided_backpropagation_analysis<F>(
    model_fn: &F,
    X: &ArrayView2<Float>,
    _target_class: Option<usize>,
    _config: &OcclusionConfig,
) -> SklResult<OcclusionResult>
where
    F: Fn(&ArrayView2<Float>) -> Vec<Float>,
{
    let (n_samples, n_features) = X.dim();
    let mut guided_gradients = Array2::zeros((n_samples, n_features));

    // For each sample, compute guided gradients
    for sample_idx in 0..n_samples {
        let x_sample = X.row(sample_idx).to_owned();
        let gradients = compute_numerical_gradients(model_fn, &x_sample, 0.001);

        // Apply guided backpropagation rule: keep only positive gradients
        // (This is a simplified approximation)
        for j in 0..n_features {
            guided_gradients[[sample_idx, j]] = gradients[j].max(0.0);
        }
    }

    // Compute importance scores
    let importance_scores = guided_gradients.mean_axis(Axis(0)).unwrap();

    Ok(OcclusionResult {
        importance_scores,
        saliency_map: None,
        integrated_gradients: None,
        attribution_map: Some(guided_gradients),
        step_attributions: None,
        feature_names: None,
    })
}

/// Layer-wise relevance propagation analysis (simplified version)
fn lrp_analysis<F>(
    model_fn: &F,
    X: &ArrayView2<Float>,
    _target_class: Option<usize>,
    _config: &OcclusionConfig,
) -> SklResult<OcclusionResult>
where
    F: Fn(&ArrayView2<Float>) -> Vec<Float>,
{
    let (n_samples, n_features) = X.dim();
    let mut relevance_scores = Array2::zeros((n_samples, n_features));

    // Simplified LRP: decompose output back to inputs
    // This is a basic approximation - real LRP requires model internals
    for sample_idx in 0..n_samples {
        let x_sample = X.row(sample_idx).to_owned();
        let base_prediction = {
            let x_mat = x_sample.clone().insert_axis(Axis(0));
            model_fn(&x_mat.view())[0]
        };

        // Use gradient-based approximation for relevance
        let gradients = compute_numerical_gradients(model_fn, &x_sample, 0.001);

        // LRP-like relevance: gradient * input
        for j in 0..n_features {
            relevance_scores[[sample_idx, j]] = gradients[j] * x_sample[j];
        }

        // Normalize to sum to prediction (conservation property)
        let total_relevance: Float = relevance_scores.row(sample_idx).sum();
        if total_relevance.abs() > 1e-10 {
            let normalization_factor = base_prediction / total_relevance;
            for j in 0..n_features {
                relevance_scores[[sample_idx, j]] *= normalization_factor;
            }
        }
    }

    // Compute importance scores
    let importance_scores = relevance_scores.map_axis(Axis(0), |col| {
        col.iter().map(|&x| x.abs()).sum::<Float>() / n_samples as Float
    });

    Ok(OcclusionResult {
        importance_scores,
        saliency_map: None,
        integrated_gradients: None,
        attribution_map: Some(relevance_scores),
        step_attributions: None,
        feature_names: None,
    })
}

// Helper functions

/// Compute numerical gradients using finite differences
fn compute_numerical_gradients<F>(
    model_fn: &F,
    x: &Array1<Float>,
    step_size: Float,
) -> Array1<Float>
where
    F: Fn(&ArrayView2<Float>) -> Vec<Float>,
{
    let n_features = x.len();
    let mut gradients = Array1::zeros(n_features);

    for i in 0..n_features {
        // Forward step
        let mut x_plus = x.clone();
        x_plus[i] += step_size;
        let x_plus_mat = x_plus.insert_axis(Axis(0));
        let pred_plus = model_fn(&x_plus_mat.view())[0];

        // Backward step
        let mut x_minus = x.clone();
        x_minus[i] -= step_size;
        let x_minus_mat = x_minus.insert_axis(Axis(0));
        let pred_minus = model_fn(&x_minus_mat.view())[0];

        // Central difference
        gradients[i] = (pred_plus - pred_minus) / (2.0 * step_size);
    }

    gradients
}

/// Create occlusion mask for patch-based analysis
pub fn create_patch_mask(
    feature_shape: (usize, usize),
    patch_center: (usize, usize),
    patch_size: (usize, usize),
) -> Array2<bool> {
    let (height, width) = feature_shape;
    let (center_h, center_w) = patch_center;
    let (patch_h, patch_w) = patch_size;

    let mut mask = Array2::from_elem((height, width), false);

    let start_h = center_h.saturating_sub(patch_h / 2);
    let end_h = (center_h + patch_h / 2 + 1).min(height);
    let start_w = center_w.saturating_sub(patch_w / 2);
    let end_w = (center_w + patch_w / 2 + 1).min(width);

    for i in start_h..end_h {
        for j in start_w..end_w {
            mask[[i, j]] = true;
        }
    }

    mask
}

/// Sliding window occlusion analysis for image-like data
pub fn sliding_window_occlusion<F>(
    model_fn: &F,
    X: &ArrayView2<Float>,
    feature_shape: (usize, usize),
    window_size: (usize, usize),
    stride: (usize, usize),
    occlusion_value: Float,
) -> SklResult<Array2<Float>>
where
    F: Fn(&ArrayView2<Float>) -> Vec<Float>,
{
    let (height, width) = feature_shape;
    let (window_h, window_w) = window_size;
    let (stride_h, stride_w) = stride;

    if X.ncols() != height * width {
        return Err(SklearsError::InvalidInput(
            "Input features must match specified feature shape".to_string(),
        ));
    }

    // Get baseline prediction
    let baseline_pred = model_fn(X)[0];

    let mut occlusion_map = Array2::zeros((
        (height - window_h) / stride_h + 1,
        (width - window_w) / stride_w + 1,
    ));

    // Slide window across feature space
    for (map_i, window_start_h) in (0..=height - window_h).step_by(stride_h).enumerate() {
        for (map_j, window_start_w) in (0..=width - window_w).step_by(stride_w).enumerate() {
            // Create occluded version
            let mut X_occluded = X.to_owned();

            for h in window_start_h..(window_start_h + window_h) {
                for w in window_start_w..(window_start_w + window_w) {
                    let feature_idx = h * width + w;
                    X_occluded[[0, feature_idx]] = occlusion_value;
                }
            }

            let occluded_pred = model_fn(&X_occluded.view())[0];
            occlusion_map[[map_i, map_j]] = baseline_pred - occluded_pred;
        }
    }

    Ok(occlusion_map)
}

#[cfg(test)]
mod tests {
    use super::*;
    // ✅ SciRS2 Policy Compliant Import
    use scirs2_core::ndarray::array;

    fn simple_model(x: &ArrayView2<Float>) -> Vec<Float> {
        x.rows()
            .into_iter()
            .map(|row| row[0] * 2.0 + row[1] * 0.5 + row.get(2).unwrap_or(&0.0) * 0.1)
            .collect()
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_feature_occlusion_analysis() {
        let X = array![[0.5, 0.7, 0.3]];
        let config = OcclusionConfig {
            method: OcclusionMethod::FeatureOcclusion,
            occlusion_strategy: OcclusionStrategy::Zero,
            ..Default::default()
        };

        let result = analyze_occlusion(&simple_model, &X.view(), None, &config).unwrap();
        assert_eq!(result.importance_scores.len(), 3);

        // Feature 0 should have highest importance (coefficient 2.0)
        assert!(result.importance_scores[0] > result.importance_scores[1]);
        assert!(result.importance_scores[1] > result.importance_scores[2]);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_integrated_gradients_analysis() {
        let X = array![[0.5, 0.7]];
        let config = OcclusionConfig {
            method: OcclusionMethod::IntegratedGradients,
            n_steps: 10,
            baseline: Some(array![0.0, 0.0]),
            ..Default::default()
        };

        let result = analyze_occlusion(&simple_model, &X.view(), None, &config).unwrap();
        assert_eq!(result.importance_scores.len(), 2);
        assert!(result.integrated_gradients.is_some());

        let ig = result.integrated_gradients.unwrap();
        assert_eq!(ig.dim(), (1, 2));
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_saliency_map_analysis() {
        let X = array![[0.5, 0.7, 0.3]];
        let config = OcclusionConfig {
            method: OcclusionMethod::SaliencyMaps,
            ..Default::default()
        };

        let result = analyze_occlusion(&simple_model, &X.view(), None, &config).unwrap();
        assert_eq!(result.importance_scores.len(), 3);
        assert!(result.saliency_map.is_some());
        assert!(result.attribution_map.is_some());
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_guided_backpropagation_analysis() {
        let X = array![[0.5, 0.7]];
        let config = OcclusionConfig {
            method: OcclusionMethod::GuidedBackpropagation,
            ..Default::default()
        };

        let result = analyze_occlusion(&simple_model, &X.view(), None, &config).unwrap();
        assert_eq!(result.importance_scores.len(), 2);
        assert!(result.attribution_map.is_some());
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_lrp_analysis() {
        let X = array![[0.5, 0.7]];
        let config = OcclusionConfig {
            method: OcclusionMethod::LayerWiseRelevancePropagation,
            ..Default::default()
        };

        let result = analyze_occlusion(&simple_model, &X.view(), None, &config).unwrap();
        assert_eq!(result.importance_scores.len(), 2);
        assert!(result.attribution_map.is_some());
    }

    #[test]
    fn test_patch_mask_creation() {
        let mask = create_patch_mask((10, 10), (5, 5), (3, 3));
        assert_eq!(mask.dim(), (10, 10));

        // Check that center area is masked
        assert!(mask[[5, 5]]);
        assert!(mask[[4, 4]]);
        assert!(mask[[6, 6]]);

        // Check that corners are not masked
        assert!(!mask[[0, 0]]);
        assert!(!mask[[9, 9]]);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_sliding_window_occlusion() {
        let X = array![[0.5, 0.7, 0.3, 0.8]]; // 2x2 feature map
        let feature_shape = (2, 2);
        let window_size = (1, 1);
        let stride = (1, 1);

        let model_2d = |x: &ArrayView2<Float>| -> Vec<Float> { vec![x.row(0).sum()] };

        let result = sliding_window_occlusion(
            &model_2d,
            &X.view(),
            feature_shape,
            window_size,
            stride,
            0.0,
        )
        .unwrap();

        assert_eq!(result.dim(), (2, 2));
    }

    #[test]
    fn test_numerical_gradients() {
        let x = array![0.5, 0.7];
        let model_1d = |x_mat: &ArrayView2<Float>| -> Vec<Float> {
            let row = x_mat.row(0);
            vec![row[0] * 2.0 + row[1] * 0.5]
        };

        let gradients = compute_numerical_gradients(&model_1d, &x, 0.001);

        // Should approximate [2.0, 0.5]
        assert!((gradients[0] - 2.0).abs() < 0.1);
        assert!((gradients[1] - 0.5).abs() < 0.1);
    }
}
