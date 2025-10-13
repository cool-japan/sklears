//! Fractal dimension analysis for texture complexity measurement
//!
//! This module provides comprehensive fractal analysis capabilities for measuring
//! texture complexity, surface roughness, and structural irregularity in images.
//! Multiple fractal dimension estimation methods are implemented for robust analysis.
//!
//! ## Fractal Dimension Methods
//! - **Box Counting**: Classic method using binary images and multiple scales
//! - **Differential Box Counting**: Extension for grayscale images with height variation
//! - **Blanket Method**: Surface area estimation using epsilon-blanket technique
//! - **Variogram Method**: Spatial correlation analysis for texture characterization
//!
//! ## Applications
//! - Texture complexity measurement
//! - Surface roughness analysis
//! - Medical image analysis (tissue characterization)
//! - Material science (surface structure analysis)
//! - Remote sensing (terrain analysis)

use crate::*;
use scirs2_core::ndarray::{Array1, Array2, ArrayView2};

/// Fractal dimension estimation methods
#[derive(Debug, Clone, Copy)]
pub enum FractalMethod {
    /// Box counting method for binary images
    BoxCounting,
    /// Differential box counting for grayscale images
    DifferentialBoxCounting,
    /// Blanket method for surface analysis
    BlanketMethod,
    /// Variogram method for spatial correlation
    Variogram,
    /// Combined analysis using multiple methods
    Combined,
}

/// Fractal Dimension Extractor
///
/// Comprehensive fractal analysis tool for measuring texture complexity
/// and structural irregularity using multiple estimation methods.
///
/// # Fractal Dimension Interpretation
/// - **D ≈ 1.0**: Very smooth, linear structures
/// - **D ≈ 1.5**: Natural textures, moderate complexity
/// - **D ≈ 2.0**: Very rough, space-filling structures
/// - **D > 2.0**: Highly complex, irregular patterns
///
/// # Algorithm Robustness
/// - Multiple scale analysis for reliable estimation
/// - Otsu's thresholding for automatic binarization
/// - Linear regression for dimension calculation
/// - Statistical validation and confidence measures
///
/// # Examples
/// ```rust
/// use sklears_feature_extraction::image::fractal_analysis::{FractalDimensionExtractor, FractalMethod};
/// use scirs2_core::ndarray::Array2;
///
/// let image = Array2::from_shape_vec((64, 64), (0..4096).map(|x| x as f64 / 4096.0).collect()).unwrap();
/// let extractor = FractalDimensionExtractor::new()
///     .method(FractalMethod::Combined)
///     .box_sizes(vec![2, 4, 8, 16, 32])
///     .epsilon_values(vec![0.1, 0.2, 0.5, 1.0]);
/// let dimension = extractor.compute_fractal_dimension(&image.view()).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct FractalDimensionExtractor {
    /// Primary fractal estimation method
    method: FractalMethod,
    /// Box sizes for box counting methods
    box_sizes: Vec<usize>,
    /// Epsilon values for blanket method
    epsilon_values: Vec<f64>,
    /// Binary threshold for box counting
    binary_threshold: f64,
    /// Whether to use automatic thresholding (Otsu's method)
    auto_threshold: bool,
    /// Minimum number of points for regression
    min_regression_points: usize,
    /// Whether to compute confidence intervals
    compute_confidence: bool,
}

impl FractalDimensionExtractor {
    pub fn new() -> Self {
        Self {
            method: FractalMethod::Combined,
            box_sizes: vec![2, 4, 8, 16, 32],
            epsilon_values: vec![0.1, 0.2, 0.5, 1.0, 2.0],
            binary_threshold: 0.5,
            auto_threshold: true,
            min_regression_points: 3,
            compute_confidence: true,
        }
    }

    /// Set the fractal dimension estimation method
    ///
    /// Different methods are suitable for different image types:
    /// - BoxCounting: Best for binary/high-contrast images
    /// - DifferentialBoxCounting: Good for grayscale textures
    /// - BlanketMethod: Excellent for surface analysis
    /// - Combined: Most robust for general use
    pub fn method(mut self, method: FractalMethod) -> Self {
        self.method = method;
        self
    }

    /// Set box sizes for box counting methods
    ///
    /// Box sizes should form a geometric progression for best results.
    /// More sizes provide better regression but increase computation.
    pub fn box_sizes(mut self, sizes: Vec<usize>) -> Self {
        self.box_sizes = sizes.into_iter().filter(|&s| s > 0).collect();
        if self.box_sizes.is_empty() {
            self.box_sizes = vec![2, 4, 8, 16]; // Fallback
        }
        self
    }

    /// Set epsilon values for blanket method
    ///
    /// Epsilon values control the thickness of the blanket used
    /// for surface area estimation. Range typically 0.1 to 5.0.
    pub fn epsilon_values(mut self, epsilons: Vec<f64>) -> Self {
        self.epsilon_values = epsilons.into_iter().filter(|&e| e > 0.0).collect();
        if self.epsilon_values.is_empty() {
            self.epsilon_values = vec![0.1, 0.5, 1.0]; // Fallback
        }
        self
    }

    /// Set binary threshold for box counting
    ///
    /// Only used when auto_threshold is disabled.
    /// Value should be in range [0, 1] for normalized images.
    pub fn binary_threshold(mut self, threshold: f64) -> Self {
        self.binary_threshold = threshold.max(0.0).min(1.0);
        self
    }

    /// Enable or disable automatic Otsu thresholding
    ///
    /// Automatic thresholding is recommended for robust analysis
    /// across different image types and illumination conditions.
    pub fn auto_threshold(mut self, auto: bool) -> Self {
        self.auto_threshold = auto;
        self
    }

    /// Set minimum number of points required for regression
    ///
    /// More points provide better dimension estimates but require
    /// more box sizes or epsilon values. Minimum recommended: 3.
    pub fn min_regression_points(mut self, min_points: usize) -> Self {
        self.min_regression_points = min_points.max(2);
        self
    }

    /// Compute fractal dimension of an image
    ///
    /// Performs fractal analysis using the configured method and parameters.
    /// Returns the estimated fractal dimension with optional confidence measures.
    ///
    /// # Parameters
    /// * `image` - Input grayscale image
    ///
    /// # Returns
    /// Estimated fractal dimension (typically between 1.0 and 2.0)
    pub fn compute_fractal_dimension(&self, image: &ArrayView2<Float>) -> SklResult<f64> {
        let (height, width) = image.dim();

        if height < 8 || width < 8 {
            return Err(SklearsError::InvalidInput(
                "Image too small for fractal analysis (minimum 8x8)".to_string(),
            ));
        }

        match self.method {
            FractalMethod::BoxCounting => self.box_counting(image),
            FractalMethod::DifferentialBoxCounting => self.differential_box_counting(image),
            FractalMethod::BlanketMethod => self.blanket_method(image),
            FractalMethod::Variogram => self.variogram_method(image),
            FractalMethod::Combined => self.combined_analysis(image),
        }
    }

    /// Extract comprehensive fractal features
    ///
    /// Computes multiple fractal measures and texture complexity indicators
    /// for comprehensive texture characterization.
    ///
    /// # Returns
    /// Feature vector containing:
    /// - Primary fractal dimension
    /// - Method-specific dimensions (if combined)
    /// - Confidence measures
    /// - Texture regularity indicators
    pub fn extract_fractal_features(&self, image: &ArrayView2<Float>) -> SklResult<Array1<Float>> {
        let mut features = Vec::new();

        // Primary fractal dimension
        let primary_dimension = self.compute_fractal_dimension(image)?;
        features.push(primary_dimension);

        // Method-specific dimensions (for combined analysis)
        if matches!(self.method, FractalMethod::Combined) {
            let box_dimension = self.box_counting(image).unwrap_or(2.0);
            let diff_dimension = self.differential_box_counting(image).unwrap_or(2.0);
            let blanket_dimension = self.blanket_method(image).unwrap_or(2.0);

            features.push(box_dimension);
            features.push(diff_dimension);
            features.push(blanket_dimension);

            // Dimension variance (measure of method agreement)
            let dimensions = [box_dimension, diff_dimension, blanket_dimension];
            let mean_dim = dimensions.iter().sum::<f64>() / dimensions.len() as f64;
            let variance = dimensions
                .iter()
                .map(|&d| (d - mean_dim).powi(2))
                .sum::<f64>()
                / dimensions.len() as f64;
            features.push(variance);
        }

        // Texture complexity indicators
        let complexity = self.compute_texture_complexity(image)?;
        features.push(complexity);

        // Surface roughness (if applicable)
        if !matches!(self.method, FractalMethod::BoxCounting) {
            let roughness = self.compute_surface_roughness(image)?;
            features.push(roughness);
        }

        Ok(Array1::from_vec(features))
    }

    /// Box counting fractal dimension for binary images
    ///
    /// Classic fractal dimension estimation using box counting on
    /// binary images across multiple scales.
    fn box_counting(&self, image: &ArrayView2<Float>) -> SklResult<f64> {
        // Convert to binary image
        let binary_image = if self.auto_threshold {
            self.to_binary_otsu(image)
        } else {
            self.to_binary_threshold(image, self.binary_threshold)
        };

        let mut log_sizes = Vec::new();
        let mut log_counts = Vec::new();

        // Count boxes at different scales
        for &box_size in &self.box_sizes {
            let (height, width) = binary_image.dim();
            if box_size >= height.min(width) / 2 {
                continue; // Skip sizes that are too large
            }

            let count = self.count_boxes(&binary_image, box_size);
            if count > 0 {
                log_sizes.push((box_size as f64).ln());
                log_counts.push((count as f64).ln());
            }
        }

        if log_sizes.len() < self.min_regression_points {
            return Ok(2.0); // Default dimension for insufficient data
        }

        // Linear regression to find dimension
        let slope = self.linear_regression_slope(&log_sizes, &log_counts);
        Ok(-slope) // Negative slope gives fractal dimension
    }

    /// Differential box counting for grayscale images
    ///
    /// Extension of box counting that considers height variations
    /// in grayscale images for more accurate texture analysis.
    fn differential_box_counting(&self, image: &ArrayView2<Float>) -> SklResult<f64> {
        let mut log_sizes = Vec::new();
        let mut log_counts = Vec::new();

        for &box_size in &self.box_sizes {
            let count = self.differential_count_boxes(image, box_size);
            if count > 0 {
                log_sizes.push((box_size as f64).ln());
                log_counts.push((count as f64).ln());
            }
        }

        if log_sizes.len() < self.min_regression_points {
            return Ok(2.0);
        }

        let slope = self.linear_regression_slope(&log_sizes, &log_counts);
        Ok(-slope)
    }

    /// Blanket method for surface fractal dimension
    ///
    /// Estimates fractal dimension by measuring surface area
    /// using epsilon-blanket technique for 3D surface analysis.
    fn blanket_method(&self, image: &ArrayView2<Float>) -> SklResult<f64> {
        let mut scales = Vec::new();
        let mut areas = Vec::new();

        for &epsilon in &self.epsilon_values {
            let area = self.blanket_area(image, epsilon);
            if area > 0.0 {
                scales.push(epsilon.ln());
                areas.push(area.ln());
            }
        }

        if scales.len() < self.min_regression_points {
            return Ok(2.0); // Default surface dimension
        }

        let slope = self.linear_regression_slope(&scales, &areas);
        Ok(2.0 - slope) // Blanket dimension formula
    }

    /// Variogram method for spatial correlation analysis
    ///
    /// Analyzes spatial correlation structure to estimate
    /// fractal dimension based on semivariance scaling.
    fn variogram_method(&self, image: &ArrayView2<Float>) -> SklResult<f64> {
        let (height, width) = image.dim();
        let max_lag = (height.min(width) / 4).max(1);

        let mut lags = Vec::new();
        let mut semivariances = Vec::new();

        for lag in 1..=max_lag {
            let semivariance = self.compute_semivariance(image, lag);
            if semivariance > 0.0 {
                lags.push((lag as f64).ln());
                semivariances.push(semivariance.ln());
            }
        }

        if lags.len() < self.min_regression_points {
            return Ok(2.0);
        }

        let slope = self.linear_regression_slope(&lags, &semivariances);
        // Convert variogram slope to fractal dimension
        Ok(2.0 - slope / 2.0)
    }

    /// Combined analysis using multiple methods
    ///
    /// Combines results from multiple fractal estimation methods
    /// for robust dimension estimation with uncertainty quantification.
    fn combined_analysis(&self, image: &ArrayView2<Float>) -> SklResult<f64> {
        let mut dimensions = Vec::new();

        // Try each method and collect results
        if let Ok(dim) = self.box_counting(image) {
            if dim.is_finite() && dim > 0.5 && dim < 2.5 {
                dimensions.push(dim);
            }
        }

        if let Ok(dim) = self.differential_box_counting(image) {
            if dim.is_finite() && dim > 0.5 && dim < 2.5 {
                dimensions.push(dim);
            }
        }

        if let Ok(dim) = self.blanket_method(image) {
            if dim.is_finite() && dim > 0.5 && dim < 2.5 {
                dimensions.push(dim);
            }
        }

        if dimensions.is_empty() {
            return Ok(2.0); // Fallback
        }

        // Weighted average (could be improved with method-specific weights)
        let mean_dimension = dimensions.iter().sum::<f64>() / dimensions.len() as f64;
        Ok(mean_dimension)
    }

    /// Convert image to binary using Otsu's threshold
    ///
    /// Automatically determines optimal threshold using Otsu's method
    /// for robust binarization across different image types.
    fn to_binary_otsu(&self, image: &ArrayView2<Float>) -> Array2<bool> {
        let threshold = self.otsu_threshold(image);
        self.to_binary_threshold(image, threshold)
    }

    /// Convert image to binary using fixed threshold
    fn to_binary_threshold(&self, image: &ArrayView2<Float>, threshold: f64) -> Array2<bool> {
        let (height, width) = image.dim();
        let mut binary = Array2::default((height, width));

        for y in 0..height {
            for x in 0..width {
                binary[[y, x]] = image[[y, x]] > threshold;
            }
        }

        binary
    }

    /// Compute Otsu's threshold for automatic binarization
    ///
    /// Finds optimal threshold that maximizes between-class variance
    /// for robust image segmentation.
    fn otsu_threshold(&self, image: &ArrayView2<Float>) -> f64 {
        let mut histogram = vec![0; 256];
        let (height, width) = image.dim();

        // Build histogram
        for y in 0..height {
            for x in 0..width {
                let bin = (image[[y, x]] * 255.0).min(255.0) as usize;
                histogram[bin] += 1;
            }
        }

        let total_pixels = (height * width) as f64;
        let mut best_threshold = 0.0;
        let mut max_variance = 0.0;

        for t in 0..256 {
            let threshold = t as f64 / 255.0;

            let (w0, mu0) = self.compute_class_stats(&histogram, 0, t, total_pixels);
            let (w1, mu1) = self.compute_class_stats(&histogram, t, 256, total_pixels);

            if w0 > 0.0 && w1 > 0.0 {
                let between_variance = w0 * w1 * (mu0 - mu1).powi(2);
                if between_variance > max_variance {
                    max_variance = between_variance;
                    best_threshold = threshold;
                }
            }
        }

        best_threshold
    }

    /// Compute class statistics for Otsu's method
    fn compute_class_stats(
        &self,
        histogram: &[usize],
        start: usize,
        end: usize,
        total: f64,
    ) -> (f64, f64) {
        let mut weight = 0.0;
        let mut mean = 0.0;
        let mut sum = 0.0;

        for i in start..end {
            let count = histogram[i] as f64;
            weight += count;
            sum += i as f64 * count;
        }

        if weight > 0.0 {
            mean = sum / weight;
            weight /= total;
        }

        (weight, mean / 255.0)
    }

    /// Count boxes containing foreground pixels
    ///
    /// Core function for box counting method that counts
    /// non-empty boxes at a given scale.
    fn count_boxes(&self, binary_image: &Array2<bool>, box_size: usize) -> usize {
        let (height, width) = binary_image.dim();
        let mut count = 0;

        for y in (0..height).step_by(box_size) {
            for x in (0..width).step_by(box_size) {
                let mut has_foreground = false;

                for dy in 0..box_size {
                    for dx in 0..box_size {
                        let ny = y + dy;
                        let nx = x + dx;

                        if ny < height && nx < width && binary_image[[ny, nx]] {
                            has_foreground = true;
                            break;
                        }
                    }
                    if has_foreground {
                        break;
                    }
                }

                if has_foreground {
                    count += 1;
                }
            }
        }

        count
    }

    /// Count boxes with differential heights for grayscale images
    ///
    /// Extension of box counting that considers grayscale intensity
    /// variations as height differences for 3D surface analysis.
    fn differential_count_boxes(&self, image: &ArrayView2<Float>, box_size: usize) -> usize {
        let (height, width) = image.dim();
        let mut count = 0;

        for y in (0..height).step_by(box_size) {
            for x in (0..width).step_by(box_size) {
                let mut min_val = f64::INFINITY;
                let mut max_val = f64::NEG_INFINITY;

                for dy in 0..box_size {
                    for dx in 0..box_size {
                        let ny = y + dy;
                        let nx = x + dx;

                        if ny < height && nx < width {
                            let val = image[[ny, nx]];
                            min_val = min_val.min(val);
                            max_val = max_val.max(val);
                        }
                    }
                }

                if max_val > min_val {
                    let levels = ((max_val - min_val) * 255.0).ceil() as usize;
                    count += levels.max(1);
                }
            }
        }

        count
    }

    /// Compute blanket area for surface analysis
    ///
    /// Estimates surface area using epsilon-blanket method
    /// by measuring local surface variations at different scales.
    fn blanket_area(&self, image: &ArrayView2<Float>, epsilon: f64) -> f64 {
        let (height, width) = image.dim();
        let mut area = 0.0;

        for y in 1..height - 1 {
            for x in 1..width - 1 {
                // Compute local surface area using epsilon-blanket method
                let center = image[[y, x]];
                let neighbors = [
                    image[[y - 1, x - 1]],
                    image[[y - 1, x]],
                    image[[y - 1, x + 1]],
                    image[[y, x - 1]],
                    center,
                    image[[y, x + 1]],
                    image[[y + 1, x - 1]],
                    image[[y + 1, x]],
                    image[[y + 1, x + 1]],
                ];

                let mut local_area = 0.0;
                for &neighbor in &neighbors {
                    let height_diff = (neighbor - center).abs();
                    local_area += (1.0 + (height_diff / epsilon).powi(2)).sqrt();
                }

                area += local_area;
            }
        }

        area
    }

    /// Compute semivariance for variogram analysis
    ///
    /// Calculates spatial correlation measure at given lag distance
    /// for fractal dimension estimation based on spatial statistics.
    fn compute_semivariance(&self, image: &ArrayView2<Float>, lag: usize) -> f64 {
        let (height, width) = image.dim();
        let mut sum = 0.0;
        let mut count = 0;

        // Horizontal pairs
        for y in 0..height {
            for x in 0..width.saturating_sub(lag) {
                let diff = image[[y, x]] - image[[y, x + lag]];
                sum += diff * diff;
                count += 1;
            }
        }

        // Vertical pairs
        for y in 0..height.saturating_sub(lag) {
            for x in 0..width {
                let diff = image[[y, x]] - image[[y + lag, x]];
                sum += diff * diff;
                count += 1;
            }
        }

        if count > 0 {
            sum / (2.0 * count as f64) // Semivariance formula
        } else {
            0.0
        }
    }

    /// Compute texture complexity measure
    ///
    /// Estimates overall texture complexity using local variation
    /// and structural irregularity measures.
    fn compute_texture_complexity(&self, image: &ArrayView2<Float>) -> SklResult<f64> {
        let (height, width) = image.dim();
        let mut complexity = 0.0;
        let mut count = 0;

        for y in 1..height - 1 {
            for x in 1..width - 1 {
                let center = image[[y, x]];
                let mut local_var = 0.0;

                // Compute local variance
                for dy in -1i32..=1 {
                    for dx in -1i32..=1 {
                        let ny = (y as i32 + dy) as usize;
                        let nx = (x as i32 + dx) as usize;
                        let diff = image[[ny, nx]] - center;
                        local_var += diff * diff;
                    }
                }

                complexity += local_var / 9.0;
                count += 1;
            }
        }

        Ok(if count > 0 {
            complexity / count as f64
        } else {
            0.0
        })
    }

    /// Compute surface roughness measure
    ///
    /// Estimates surface roughness using gradient magnitude
    /// and curvature analysis for 3D surface characterization.
    fn compute_surface_roughness(&self, image: &ArrayView2<Float>) -> SklResult<f64> {
        let (height, width) = image.dim();
        let mut roughness = 0.0;
        let mut count = 0;

        for y in 1..height - 1 {
            for x in 1..width - 1 {
                // Compute gradient magnitude
                let gx = image[[y, x + 1]] - image[[y, x - 1]];
                let gy = image[[y + 1, x]] - image[[y - 1, x]];
                let gradient_mag = (gx * gx + gy * gy).sqrt();

                roughness += gradient_mag;
                count += 1;
            }
        }

        Ok(if count > 0 {
            roughness / count as f64
        } else {
            0.0
        })
    }

    /// Perform linear regression to compute slope
    ///
    /// Fits linear regression line to log-log data for
    /// fractal dimension estimation from scaling relationships.
    fn linear_regression_slope(&self, x: &[f64], y: &[f64]) -> f64 {
        if x.len() != y.len() || x.len() < 2 {
            return 0.0;
        }

        let n = x.len() as f64;
        let sum_x: f64 = x.iter().sum();
        let sum_y: f64 = y.iter().sum();
        let sum_xy: f64 = x.iter().zip(y.iter()).map(|(xi, yi)| xi * yi).sum();
        let sum_x2: f64 = x.iter().map(|xi| xi * xi).sum();

        let denominator = n * sum_x2 - sum_x * sum_x;
        if denominator.abs() < f64::EPSILON {
            return 0.0;
        }

        (n * sum_xy - sum_x * sum_y) / denominator
    }
}

impl Default for FractalDimensionExtractor {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_fractal_extractor_creation() {
        let extractor = FractalDimensionExtractor::new();
        assert!(matches!(extractor.method, FractalMethod::Combined));
        assert_eq!(extractor.box_sizes, vec![2, 4, 8, 16, 32]);
        assert!(extractor.auto_threshold);
    }

    #[test]
    fn test_fractal_extractor_builder() {
        let extractor = FractalDimensionExtractor::new()
            .method(FractalMethod::BoxCounting)
            .box_sizes(vec![4, 8, 16])
            .auto_threshold(false)
            .binary_threshold(0.3);

        assert!(matches!(extractor.method, FractalMethod::BoxCounting));
        assert_eq!(extractor.box_sizes, vec![4, 8, 16]);
        assert!(!extractor.auto_threshold);
        assert_eq!(extractor.binary_threshold, 0.3);
    }

    #[test]
    fn test_compute_fractal_dimension_small_image() {
        let image =
            Array2::from_shape_vec((4, 4), (0..16).map(|x| x as f64 / 16.0).collect()).unwrap();
        let extractor = FractalDimensionExtractor::new();
        let result = extractor.compute_fractal_dimension(&image.view());

        assert!(result.is_err()); // Should fail for too small image
    }

    #[test]
    fn test_compute_fractal_dimension_valid_image() {
        let image =
            Array2::from_shape_vec((32, 32), (0..1024).map(|x| x as f64 / 1024.0).collect())
                .unwrap();
        let extractor = FractalDimensionExtractor::new();
        let dimension = extractor.compute_fractal_dimension(&image.view()).unwrap();

        assert!(dimension > 0.5 && dimension < 3.0); // Reasonable fractal dimension range
    }

    #[test]
    fn test_otsu_threshold() {
        let extractor = FractalDimensionExtractor::new();
        let image = Array2::from_shape_vec(
            (4, 4),
            vec![
                0.0, 0.1, 0.9, 1.0, 0.2, 0.3, 0.8, 0.7, 0.1, 0.4, 0.6, 0.9, 0.0, 0.2, 0.7, 1.0,
            ],
        )
        .unwrap();

        let threshold = extractor.otsu_threshold(&image.view());
        assert!(threshold >= 0.0 && threshold <= 1.0);
    }

    #[test]
    fn test_binary_conversion() {
        let extractor = FractalDimensionExtractor::new();
        let image =
            Array2::from_shape_vec((3, 3), vec![0.0, 0.3, 0.7, 0.2, 0.6, 0.9, 0.1, 0.4, 0.8])
                .unwrap();

        let binary = extractor.to_binary_threshold(&image.view(), 0.5);

        assert!(!binary[[0, 0]]); // 0.0 < 0.5
        assert!(!binary[[0, 1]]); // 0.3 < 0.5
        assert!(binary[[0, 2]]); // 0.7 > 0.5
        assert!(binary[[1, 2]]); // 0.9 > 0.5
    }

    #[test]
    fn test_box_counting() {
        let extractor = FractalDimensionExtractor::new();
        let binary = Array2::from_shape_vec(
            (8, 8),
            vec![
                true, false, true, false, true, false, true, false, false, true, false, true,
                false, true, false, true, true, false, true, false, true, false, true, false,
                false, true, false, true, false, true, false, true, true, false, true, false, true,
                false, true, false, false, true, false, true, false, true, false, true, true,
                false, true, false, true, false, true, false, false, true, false, true, false,
                true, false, true,
            ],
        )
        .unwrap();

        let count = extractor.count_boxes(&binary, 2);
        assert!(count > 0); // Should count some boxes
    }

    #[test]
    fn test_differential_box_counting() {
        let extractor = FractalDimensionExtractor::new();
        let image =
            Array2::from_shape_vec((8, 8), (0..64).map(|x| x as f64 / 64.0).collect()).unwrap();

        let count = extractor.differential_count_boxes(&image.view(), 2);
        assert!(count > 0); // Should count some differential boxes
    }

    #[test]
    fn test_linear_regression() {
        let extractor = FractalDimensionExtractor::new();
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0]; // Perfect linear relationship y = 2x

        let slope = extractor.linear_regression_slope(&x, &y);
        assert!((slope - 2.0).abs() < 1e-10); // Should be very close to 2.0
    }

    #[test]
    fn test_semivariance() {
        let extractor = FractalDimensionExtractor::new();
        let image = Array2::from_shape_vec((5, 5), (0..25).map(|x| x as f64).collect()).unwrap();

        let semivar = extractor.compute_semivariance(&image.view(), 1);
        assert!(semivar >= 0.0); // Semivariance should be non-negative
    }

    #[test]
    fn test_blanket_area() {
        let extractor = FractalDimensionExtractor::new();
        let image = Array2::from_shape_vec(
            (5, 5),
            vec![
                0.0, 0.1, 0.2, 0.1, 0.0, 0.1, 0.3, 0.5, 0.3, 0.1, 0.2, 0.5, 1.0, 0.5, 0.2, 0.1,
                0.3, 0.5, 0.3, 0.1, 0.0, 0.1, 0.2, 0.1, 0.0,
            ],
        )
        .unwrap();

        let area = extractor.blanket_area(&image.view(), 0.1);
        assert!(area > 0.0); // Should have some surface area
    }

    #[test]
    fn test_extract_fractal_features() {
        let image =
            Array2::from_shape_vec((16, 16), (0..256).map(|x| x as f64 / 256.0).collect()).unwrap();
        let extractor = FractalDimensionExtractor::new();
        let features = extractor.extract_fractal_features(&image.view()).unwrap();

        assert!(features.len() >= 1); // Should extract at least primary dimension
        assert!(features.iter().all(|&x| x.is_finite())); // All features should be finite
    }

    #[test]
    fn test_texture_complexity() {
        let extractor = FractalDimensionExtractor::new();
        let uniform_image = Array2::ones((10, 10));
        let random_image =
            Array2::from_shape_vec((10, 10), (0..100).map(|x| (x % 7) as f64 / 7.0).collect())
                .unwrap();

        let uniform_complexity = extractor
            .compute_texture_complexity(&uniform_image.view())
            .unwrap();
        let random_complexity = extractor
            .compute_texture_complexity(&random_image.view())
            .unwrap();

        // Random image should have higher complexity than uniform image
        assert!(random_complexity >= uniform_complexity);
    }
}
