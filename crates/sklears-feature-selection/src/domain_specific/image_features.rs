//! Image feature selection module
//!
//! This module provides specialized feature selection algorithms for image data,
//! including spatial correlation analysis, frequency domain features, and texture analysis.

use crate::base::SelectorMixin;
use scirs2_core::ndarray::{s, Array1, Array2, ArrayView1, Axis};
use sklears_core::{
    error::{validate, Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Trained, Transform, Untrained},
    types::Float,
};
use std::marker::PhantomData;

/// Image feature selection using spatial correlation and frequency analysis
///
/// This selector analyzes image features represented as flattened pixel matrices
/// or extracted feature vectors and applies image-specific selection criteria:
/// - Spatial correlation analysis for neighboring pixels
/// - Frequency domain analysis using variance as a proxy
/// - Texture analysis using local variance measurements
/// - Combined scoring with target correlation
///
/// # Input Format
///
/// The input matrix `X` should be structured as:
/// - Rows: Images
/// - Columns: Features (pixels, extracted features, etc.)
/// - Values: Pixel intensities, feature values, or derived measurements
///
/// # Examples
///
/// ```rust,ignore
/// use sklears_feature_selection::domain_specific::image_features::ImageFeatureSelector;
/// use sklears_core::traits::{Fit, Transform};
/// use scirs2_core::ndarray::{Array1, Array2};
///
/// let selector = ImageFeatureSelector::new()
///     .include_spatial(true)     // Enable spatial correlation analysis
///     .include_frequency(true)   // Enable frequency domain analysis
///     .include_texture(true)     // Enable texture analysis
///     .spatial_threshold(0.15)   // Threshold for spatial features
///     .k(Some(100));             // Select top 100 features
///
/// let x = Array2::zeros((50, 784)); // 50 images, 784 pixels (28x28)
/// let y = Array1::zeros(50);         // Image labels
///
/// let fitted_selector = selector.fit(&x, &y)?;
/// let transformed_x = fitted_selector.transform(&x)?;
/// ```
#[derive(Debug, Clone)]
pub struct ImageFeatureSelector<State = Untrained> {
    /// Whether to include spatial correlation features
    include_spatial: bool,
    /// Whether to include frequency domain features
    include_frequency: bool,
    /// Whether to include texture features
    include_texture: bool,
    /// Threshold for spatial correlation
    spatial_threshold: f64,
    /// Number of top features to select
    k: Option<usize>,
    state: PhantomData<State>,
    // Trained state
    spatial_scores_: Option<Array1<Float>>,
    frequency_scores_: Option<Array1<Float>>,
    texture_scores_: Option<Array1<Float>>,
    selected_features_: Option<Vec<usize>>,
}

impl ImageFeatureSelector<Untrained> {
    /// Create a new image feature selector with default parameters
    ///
    /// Default configuration:
    /// - `include_spatial`: true
    /// - `include_frequency`: true
    /// - `include_texture`: true
    /// - `spatial_threshold`: 0.1
    /// - `k`: None (use threshold-based selection)
    pub fn new() -> Self {
        Self {
            include_spatial: true,
            include_frequency: true,
            include_texture: true,
            spatial_threshold: 0.1,
            k: None,
            state: PhantomData,
            spatial_scores_: None,
            frequency_scores_: None,
            texture_scores_: None,
            selected_features_: None,
        }
    }

    /// Enable or disable spatial correlation analysis
    ///
    /// When enabled, the selector computes correlations between features
    /// and the target variable, emphasizing spatial relationships.
    pub fn include_spatial(mut self, include_spatial: bool) -> Self {
        self.include_spatial = include_spatial;
        self
    }

    /// Enable or disable frequency domain analysis
    ///
    /// When enabled, the selector analyzes frequency content using
    /// variance as a proxy for spectral energy, combined with target correlation.
    pub fn include_frequency(mut self, include_frequency: bool) -> Self {
        self.include_frequency = include_frequency;
        self
    }

    /// Enable or disable texture analysis
    ///
    /// When enabled, the selector computes local variance measurements
    /// to identify texture-rich regions that correlate with the target.
    pub fn include_texture(mut self, include_texture: bool) -> Self {
        self.include_texture = include_texture;
        self
    }

    /// Set the threshold for spatial correlation selection
    ///
    /// Features with combined scores below this threshold will be filtered out
    /// (when not using k-based selection).
    pub fn spatial_threshold(mut self, threshold: f64) -> Self {
        self.spatial_threshold = threshold;
        self
    }

    /// Set the number of top features to select
    ///
    /// When set to `Some(k)`, selects the top k features by combined score.
    /// When set to `None`, uses threshold-based selection.
    pub fn k(mut self, k: Option<usize>) -> Self {
        self.k = k;
        self
    }
}

impl Default for ImageFeatureSelector<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for ImageFeatureSelector<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<Array2<Float>, Array1<Float>> for ImageFeatureSelector<Untrained> {
    type Fitted = ImageFeatureSelector<Trained>;

    fn fit(self, x: &Array2<Float>, y: &Array1<Float>) -> SklResult<Self::Fitted> {
        validate::check_consistent_length(x, y)?;

        let (_, n_features) = x.dim();

        // Compute spatial correlation scores
        let spatial_scores = if self.include_spatial {
            Some(compute_spatial_correlation_scores(x, y))
        } else {
            None
        };

        // Compute frequency domain scores
        let frequency_scores = if self.include_frequency {
            Some(compute_frequency_domain_scores(x, y))
        } else {
            None
        };

        // Compute texture scores
        let texture_scores = if self.include_texture {
            Some(compute_texture_scores(x, y))
        } else {
            None
        };

        // Combine scores and select features
        let mut combined_scores = Array1::zeros(n_features);
        let mut weight_sum = 0.0;

        if let Some(ref spatial) = spatial_scores {
            for i in 0..n_features {
                combined_scores[i] += 0.4 * spatial[i];
            }
            weight_sum += 0.4;
        }

        if let Some(ref frequency) = frequency_scores {
            for i in 0..n_features {
                combined_scores[i] += 0.3 * frequency[i];
            }
            weight_sum += 0.3;
        }

        if let Some(ref texture) = texture_scores {
            for i in 0..n_features {
                combined_scores[i] += 0.3 * texture[i];
            }
            weight_sum += 0.3;
        }

        if weight_sum > 0.0 {
            combined_scores /= weight_sum;
        }

        // Select features based on combined scores
        let selected_features = self.select_features_from_combined_scores(&combined_scores);

        Ok(ImageFeatureSelector {
            include_spatial: self.include_spatial,
            include_frequency: self.include_frequency,
            include_texture: self.include_texture,
            spatial_threshold: self.spatial_threshold,
            k: self.k,
            state: PhantomData,
            spatial_scores_: spatial_scores,
            frequency_scores_: frequency_scores,
            texture_scores_: texture_scores,
            selected_features_: Some(selected_features),
        })
    }
}

impl ImageFeatureSelector<Untrained> {
    fn select_features_from_combined_scores(&self, scores: &Array1<Float>) -> Vec<usize> {
        let mut feature_indices: Vec<(usize, Float)> = scores
            .indexed_iter()
            .map(|(i, &score)| (i, score))
            .collect();

        feature_indices.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let selected: Vec<usize> = if let Some(k) = self.k {
            feature_indices
                .iter()
                .take(k.min(feature_indices.len()))
                .map(|(i, _)| *i)
                .collect()
        } else {
            feature_indices
                .iter()
                .filter(|(_, score)| *score >= self.spatial_threshold)
                .map(|(i, _)| *i)
                .collect()
        };

        let mut selected_sorted = selected;
        selected_sorted.sort();

        if selected_sorted.is_empty() {
            if let Some(&(best_idx, _)) = feature_indices.first() {
                selected_sorted.push(best_idx);
                selected_sorted.sort();
            }
        }
        selected_sorted
    }
}

impl Transform<Array2<Float>> for ImageFeatureSelector<Trained> {
    fn transform(&self, x: &Array2<Float>) -> SklResult<Array2<Float>> {
        let selected_features = self.selected_features_.as_ref().unwrap();
        if selected_features.is_empty() {
            return Err(SklearsError::InvalidInput(
                "No features were selected".to_string(),
            ));
        }

        let selected_indices: Vec<usize> = selected_features.to_vec();
        Ok(x.select(Axis(1), &selected_indices))
    }
}

impl SelectorMixin for ImageFeatureSelector<Trained> {
    fn get_support(&self) -> SklResult<Array1<bool>> {
        let selected_features = self.selected_features_.as_ref().unwrap();
        let n_features = if let Some(ref scores) = self.spatial_scores_ {
            scores.len()
        } else if let Some(ref scores) = self.frequency_scores_ {
            scores.len()
        } else if let Some(ref scores) = self.texture_scores_ {
            scores.len()
        } else {
            selected_features.iter().max().unwrap_or(&0) + 1
        };

        let mut support = Array1::from_elem(n_features, false);
        for &idx in selected_features {
            if idx < n_features {
                support[idx] = true;
            }
        }
        Ok(support)
    }

    fn transform_features(&self, indices: &[usize]) -> SklResult<Vec<usize>> {
        let selected_features = self.selected_features_.as_ref().unwrap();
        Ok(indices
            .iter()
            .filter_map(|&idx| selected_features.iter().position(|&f| f == idx))
            .collect())
    }
}

impl ImageFeatureSelector<Trained> {
    /// Get the spatial correlation scores (if spatial analysis was enabled)
    ///
    /// Returns `None` if spatial analysis was not enabled during fitting.
    pub fn spatial_scores(&self) -> Option<&Array1<Float>> {
        self.spatial_scores_.as_ref()
    }

    /// Get the frequency domain scores (if frequency analysis was enabled)
    ///
    /// Returns `None` if frequency analysis was not enabled during fitting.
    pub fn frequency_scores(&self) -> Option<&Array1<Float>> {
        self.frequency_scores_.as_ref()
    }

    /// Get the texture scores (if texture analysis was enabled)
    ///
    /// Returns `None` if texture analysis was not enabled during fitting.
    pub fn texture_scores(&self) -> Option<&Array1<Float>> {
        self.texture_scores_.as_ref()
    }

    /// Get the indices of selected features
    pub fn selected_features(&self) -> &[usize] {
        self.selected_features_.as_ref().unwrap()
    }

    /// Get the number of selected features
    pub fn n_features_selected(&self) -> usize {
        self.selected_features_.as_ref().unwrap().len()
    }

    /// Get a summary of feature scores across all analysis types
    ///
    /// Returns a vector of tuples containing (feature_index, spatial_score, frequency_score, texture_score)
    /// for all selected features. Scores are `None` if the corresponding analysis was not enabled.
    pub fn feature_summary(&self) -> Vec<(usize, Option<Float>, Option<Float>, Option<Float>)> {
        let indices = self.selected_features();
        let mut summary = Vec::with_capacity(indices.len());

        for &idx in indices {
            let spatial_score = self.spatial_scores_.as_ref().map(|scores| scores[idx]);
            let frequency_score = self.frequency_scores_.as_ref().map(|scores| scores[idx]);
            let texture_score = self.texture_scores_.as_ref().map(|scores| scores[idx]);

            summary.push((idx, spatial_score, frequency_score, texture_score));
        }

        summary
    }
}

// ================================================================================================
// Helper Functions
// ================================================================================================

/// Compute spatial correlation scores between features and target
///
/// This function calculates the Pearson correlation coefficient between
/// each feature and the target variable. Higher absolute correlations
/// indicate stronger spatial relationships with the prediction target.
fn compute_spatial_correlation_scores(x: &Array2<Float>, y: &Array1<Float>) -> Array1<Float> {
    let (_, n_features) = x.dim();
    let mut scores = Array1::zeros(n_features);

    for j in 0..n_features {
        let feature = x.column(j);
        // Compute correlation with target
        let corr = compute_pearson_correlation(&feature, y);
        scores[j] = corr.abs();
    }

    scores
}

/// Compute frequency domain scores for image features
///
/// This function uses variance as a proxy for frequency content,
/// combined with correlation to the target variable. Higher variance
/// indicates more spectral energy, which may be important for classification.
fn compute_frequency_domain_scores(x: &Array2<Float>, y: &Array1<Float>) -> Array1<Float> {
    let (_, n_features) = x.dim();
    let mut scores = Array1::zeros(n_features);

    // Simplified frequency domain analysis
    for j in 0..n_features {
        let feature = x.column(j);
        // Compute variance as a proxy for frequency content
        let variance = feature.var(0.0);
        // Combine with correlation to target
        let corr = compute_pearson_correlation(&feature, y);
        scores[j] = variance * corr.abs();
    }

    scores
}

/// Compute texture scores using local variance analysis
///
/// This function analyzes texture by computing local variance measurements
/// within small windows. High texture regions often contain important
/// discriminative information for image classification tasks.
fn compute_texture_scores(x: &Array2<Float>, y: &Array1<Float>) -> Array1<Float> {
    let (_, n_features) = x.dim();
    let mut scores = Array1::zeros(n_features);

    // Simplified texture analysis using local variance
    for j in 0..n_features {
        let feature = x.column(j);
        let local_variance = compute_local_variance(&feature);
        let corr = compute_pearson_correlation(&feature, y);
        scores[j] = local_variance * corr.abs();
    }

    scores
}

/// Compute Pearson correlation coefficient between two variables
///
/// Returns the linear correlation coefficient between x and y,
/// ranging from -1 (perfect negative correlation) to +1 (perfect positive correlation).
fn compute_pearson_correlation(x: &ArrayView1<Float>, y: &Array1<Float>) -> Float {
    let n = x.len().min(y.len());
    if n < 2 {
        return 0.0;
    }

    let x_mean = x.iter().take(n).sum::<Float>() / n as Float;
    let y_mean = y.iter().take(n).sum::<Float>() / n as Float;

    let mut numerator = 0.0;
    let mut x_var = 0.0;
    let mut y_var = 0.0;

    for i in 0..n {
        let x_i = x[i] - x_mean;
        let y_i = y[i] - y_mean;
        numerator += x_i * y_i;
        x_var += x_i * x_i;
        y_var += y_i * y_i;
    }

    let denominator = (x_var * y_var).sqrt();
    if denominator.abs() < 1e-10 {
        0.0
    } else {
        numerator / denominator
    }
}

/// Compute local variance for texture analysis
///
/// Uses a sliding window approach to compute variance within small neighborhoods.
/// This helps identify regions with high texture content that may be important
/// for image classification or object detection tasks.
fn compute_local_variance(feature: &ArrayView1<Float>) -> Float {
    let n = feature.len();
    if n < 3 {
        return 0.0;
    }

    let mut local_var = 0.0;
    let _window_size = 3; // Simple 3-point window

    for i in 1..(n - 1) {
        let window = &feature.slice(s![i - 1..i + 2]);
        let var = window.var(0.0);
        local_var += var;
    }

    local_var / (n - 2) as Float
}

/// Create a new image feature selector
pub fn create_image_feature_selector() -> ImageFeatureSelector<Untrained> {
    ImageFeatureSelector::new()
}

/// Create an image feature selector optimized for low-resolution images
///
/// Suitable for small images (e.g., 28x28, 32x32) where spatial relationships
/// are more important than high-frequency details.
pub fn create_low_resolution_selector() -> ImageFeatureSelector<Untrained> {
    ImageFeatureSelector::new()
        .include_spatial(true)
        .include_frequency(false)
        .include_texture(true)
        .spatial_threshold(0.05)
}

/// Create an image feature selector optimized for high-resolution images
///
/// Suitable for large images where frequency domain and texture analysis
/// can capture fine-grained details that are important for classification.
pub fn create_high_resolution_selector() -> ImageFeatureSelector<Untrained> {
    ImageFeatureSelector::new()
        .include_spatial(true)
        .include_frequency(true)
        .include_texture(true)
        .spatial_threshold(0.1)
        .k(Some(500))
}

/// Create an image feature selector focused on texture analysis
///
/// Suitable for applications where texture is the primary discriminative
/// feature, such as material classification or medical image analysis.
pub fn create_texture_focused_selector() -> ImageFeatureSelector<Untrained> {
    ImageFeatureSelector::new()
        .include_spatial(false)
        .include_frequency(false)
        .include_texture(true)
        .spatial_threshold(0.2)
}

/// Create an image feature selector focused on spatial relationships
///
/// Suitable for applications where spatial structure is most important,
/// such as object detection or shape classification.
pub fn create_spatial_focused_selector() -> ImageFeatureSelector<Untrained> {
    ImageFeatureSelector::new()
        .include_spatial(true)
        .include_frequency(false)
        .include_texture(false)
        .spatial_threshold(0.15)
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{array, Array2};

    #[test]
    fn test_pearson_correlation_computation() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0];
        let corr = compute_pearson_correlation(&x.view(), &y);

        // Perfect positive correlation
        assert!((corr - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_local_variance_computation() {
        let feature = array![1.0, 1.0, 1.0, 5.0, 5.0, 5.0]; // Two constant regions
        let local_var = compute_local_variance(&feature.view());

        // Should detect variance at the boundary
        assert!(local_var > 0.0);
    }

    #[test]
    fn test_spatial_correlation_scores() {
        let x = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 2.0, 4.0, 3.0, 6.0]).unwrap();
        let y = array![1.0, 2.0, 3.0];

        let scores = compute_spatial_correlation_scores(&x, &y);
        assert_eq!(scores.len(), 2);

        // Both features should have perfect correlation with target
        assert!((scores[0] - 1.0).abs() < 1e-10);
        assert!((scores[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_image_feature_selector_basic() {
        let selector = ImageFeatureSelector::new()
            .include_spatial(true)
            .include_frequency(false)
            .include_texture(false)
            .k(Some(1));

        let x = Array2::from_shape_vec(
            (4, 3),
            vec![1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 3.0, 6.0, 9.0, 4.0, 8.0, 12.0],
        )
        .unwrap();
        let y = array![1.0, 2.0, 3.0, 4.0];

        let fitted = selector.fit(&x, &y).unwrap();
        assert_eq!(fitted.n_features_selected(), 1);

        let transformed = fitted.transform(&x).unwrap();
        assert_eq!(transformed.ncols(), 1);
    }

    #[test]
    fn test_feature_selection_with_threshold() {
        let selector = ImageFeatureSelector::new()
            .include_spatial(true)
            .spatial_threshold(0.8); // High threshold

        let x = Array2::from_shape_vec(
            (3, 3),
            vec![
                1.0, 0.0, 1.0, // Strong correlation
                2.0, 1.0, 2.0, // Weak correlation
                3.0, 0.0, 3.0, // Strong correlation
            ],
        )
        .unwrap();
        let y = array![1.0, 2.0, 3.0];

        let fitted = selector.fit(&x, &y).unwrap();

        // Should select features with high correlation (0 and 2)
        assert!(fitted.n_features_selected() >= 1);
        assert!(fitted.n_features_selected() <= 2);
    }
}
