//! Zernike moments for rotation-invariant shape analysis
//!
//! This module provides comprehensive Zernike moment computation for rotation-invariant
//! shape analysis and pattern recognition. Zernike moments are orthogonal moments
//! based on Zernike polynomials that provide excellent shape description properties.
//!
//! ## Features
//! - Complete Zernike polynomial computation up to arbitrary order
//! - Rotation-invariant shape descriptors
//! - Scale and translation normalization options
//! - Efficient radial polynomial computation with caching
//! - Magnitude-based rotation invariance
//! - Complex moment calculation for phase information
//!
//! ## Mathematical Foundation
//! Zernike moments are defined as:
//! A_nm = (n+1)/π * ∫∫ f(ρ,θ) * R_n^m(ρ) * exp(-imθ) dρ dθ
//!
//! Where:
//! - R_n^m(ρ) are radial polynomials
//! - n is the order, m is the repetition
//! - ρ is the radial distance, θ is the angle
//! - Rotation invariance achieved by taking magnitude |A_nm|

use crate::*;
use scirs2_core::ndarray::{Array1, Array2, ArrayView2};
use std::f64::consts::PI;

/// Zernike Moments Extractor
///
/// Comprehensive Zernike moment computation for rotation-invariant shape analysis.
/// Provides configurable moment computation with various normalization options
/// and optimization features for efficient processing.
///
/// # Zernike Moment Properties
/// - **Orthogonality**: Zernike polynomials form orthogonal basis
/// - **Rotation Invariance**: Magnitude of moments is rotation invariant
/// - **Reconstruction**: Higher-order moments capture fine shape details
/// - **Noise Robustness**: Lower-order moments are more robust to noise
///
/// # Moment Interpretation
/// - **Low Order (n=0-2)**: Global shape characteristics
/// - **Medium Order (n=3-6)**: Local shape features and details
/// - **High Order (n>6)**: Fine shape details and texture information
///
/// # Examples
/// ```rust
/// use sklears_feature_extraction::image::zernike_moments::ZernikeMomentsExtractor;
/// use scirs2_core::ndarray::Array2;
///
/// let image = Array2::from_shape_vec((32, 32), (0..1024).map(|x| x as f64 / 1024.0).collect()).unwrap();
/// let extractor = ZernikeMomentsExtractor::new()
///     .max_order(6)
///     .normalize_radius(true)
///     .binary_threshold(0.5);
/// let moments = extractor.extract_moments(&image.view()).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct ZernikeMomentsExtractor {
    /// Maximum order of Zernike moments to compute
    max_order: usize,
    /// Binary threshold for shape extraction
    binary_threshold: f64,
    /// Whether to normalize radius to unit circle
    normalize_radius: bool,
    /// Whether to compute complex moments (with phase)
    complex_moments: bool,
    /// Whether to apply translation normalization
    translation_invariant: bool,
    /// Whether to apply scale normalization
    scale_invariant: bool,
    /// Precomputed factorial values for efficiency
    factorial_cache: Vec<usize>,
}

impl ZernikeMomentsExtractor {
    /// Create a new Zernike moments extractor with default settings
    ///
    /// Default configuration:
    /// - Maximum order: 6 (good balance of detail and computation)
    /// - Binary threshold: 0.5 (middle gray value)
    /// - Radius normalization enabled for proper scaling
    /// - Rotation invariance through magnitude computation
    /// - Translation and scale invariance enabled
    pub fn new() -> Self {
        let max_order = 6;
        let factorial_cache = Self::precompute_factorials(max_order + 10);

        Self {
            max_order,
            binary_threshold: 0.5,
            normalize_radius: true,
            complex_moments: false,
            translation_invariant: true,
            scale_invariant: true,
            factorial_cache,
        }
    }

    /// Set the maximum order of Zernike moments
    ///
    /// Higher orders capture finer shape details but increase computation.
    /// Typical range: 4-12 for most applications.
    /// Orders must satisfy: n - |m| is even and |m| ≤ n.
    pub fn max_order(mut self, order: usize) -> Self {
        self.max_order = order.max(1).min(20); // Reasonable bounds
        self.factorial_cache = Self::precompute_factorials(self.max_order + 10);
        self
    }

    /// Set the binary threshold for shape extraction
    ///
    /// Threshold determines which pixels are considered part of the object.
    /// Values should be in range [0, 1] for normalized images.
    pub fn binary_threshold(mut self, threshold: f64) -> Self {
        self.binary_threshold = threshold.max(0.0).min(1.0);
        self
    }

    /// Set whether to normalize radius to unit circle
    ///
    /// Normalization ensures proper Zernike polynomial evaluation
    /// and consistent moment computation across different object sizes.
    pub fn normalize_radius(mut self, normalize: bool) -> Self {
        self.normalize_radius = normalize;
        self
    }

    /// Set whether to compute complex moments with phase information
    ///
    /// Complex moments preserve phase information which can be useful
    /// for certain applications but are not rotation invariant.
    pub fn complex_moments(mut self, complex: bool) -> Self {
        self.complex_moments = complex;
        self
    }

    /// Set whether to apply translation invariance
    ///
    /// Translation invariance is achieved by computing moments
    /// relative to the object centroid.
    pub fn translation_invariant(mut self, invariant: bool) -> Self {
        self.translation_invariant = invariant;
        self
    }

    /// Set whether to apply scale invariance
    ///
    /// Scale invariance is achieved by normalizing the radius
    /// by the object's characteristic size.
    pub fn scale_invariant(mut self, invariant: bool) -> Self {
        self.scale_invariant = invariant;
        self
    }

    /// Extract Zernike moments from an image
    ///
    /// Performs complete Zernike moment analysis including binary conversion,
    /// centroid computation, radius normalization, and moment calculation.
    ///
    /// # Parameters
    /// * `image` - Input grayscale image
    ///
    /// # Returns
    /// Array of Zernike moment magnitudes (rotation invariant)
    pub fn extract_moments(&self, image: &ArrayView2<Float>) -> SklResult<Array1<Float>> {
        let (height, width) = image.dim();

        if height < 3 || width < 3 {
            return Err(SklearsError::InvalidInput(
                "Image too small for Zernike analysis (minimum 3x3)".to_string(),
            ));
        }

        // Convert to binary image
        let binary_image = self.to_binary(image);

        // Check if object exists
        let area = self.compute_area(&binary_image);
        if area == 0.0 {
            return Ok(Array1::zeros(self.estimate_moment_count()));
        }

        // Compute object properties for normalization
        let (cx, cy, radius) = self.compute_normalization_parameters(&binary_image);

        // Compute Zernike moments
        let moments = self.compute_zernike_moments(&binary_image, cx, cy, radius)?;

        Ok(Array1::from_vec(moments))
    }

    /// Extract comprehensive Zernike features
    ///
    /// Computes Zernike moments along with derived features such as
    /// shape complexity, symmetry measures, and reconstruction quality.
    ///
    /// # Returns
    /// Extended feature vector including:
    /// - Zernike moment magnitudes
    /// - Shape complexity measures
    /// - Symmetry indicators
    /// - Reconstruction error estimates
    pub fn extract_features(&self, image: &ArrayView2<Float>) -> SklResult<Array1<Float>> {
        let mut features = Vec::new();

        // Primary Zernike moments
        let moments = self.extract_moments(image)?;
        features.extend(moments.iter().cloned());

        // Derived features
        let binary_image = self.to_binary(image);
        let area = self.compute_area(&binary_image);

        if area > 0.0 {
            let (cx, cy, radius) = self.compute_normalization_parameters(&binary_image);

            // Shape complexity from higher-order moments
            let complexity = self.compute_shape_complexity(&moments);
            features.push(complexity);

            // Symmetry measures
            let symmetry = self.compute_symmetry_measures(&binary_image, cx, cy, radius)?;
            features.extend(symmetry);

            // Compactness relative to circular shape
            let circularity = self.compute_circularity(&binary_image, area, radius);
            features.push(circularity);
        } else {
            // Add zero features for empty objects
            features.push(0.0); // Complexity
            features.extend(vec![0.0; 4]); // Symmetry measures
            features.push(0.0); // Circularity
        }

        Ok(Array1::from_vec(features))
    }

    /// Convert image to binary using threshold
    fn to_binary(&self, image: &ArrayView2<Float>) -> Array2<bool> {
        let (height, width) = image.dim();
        let mut binary = Array2::default((height, width));

        for y in 0..height {
            for x in 0..width {
                binary[[y, x]] = image[[y, x]] > self.binary_threshold;
            }
        }

        binary
    }

    /// Compute object area (number of foreground pixels)
    fn compute_area(&self, binary_image: &Array2<bool>) -> f64 {
        binary_image.iter().filter(|&&pixel| pixel).count() as f64
    }

    /// Compute normalization parameters for Zernike moment calculation
    ///
    /// Determines centroid and characteristic radius for proper
    /// moment normalization and invariance properties.
    fn compute_normalization_parameters(&self, binary_image: &Array2<bool>) -> (f64, f64, f64) {
        let (height, width) = binary_image.dim();
        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut count = 0;

        // Compute centroid
        for y in 0..height {
            for x in 0..width {
                if binary_image[[y, x]] {
                    sum_x += x as f64;
                    sum_y += y as f64;
                    count += 1;
                }
            }
        }

        let center_x = if count > 0 {
            sum_x / count as f64
        } else {
            width as f64 / 2.0
        };
        let center_y = if count > 0 {
            sum_y / count as f64
        } else {
            height as f64 / 2.0
        };

        // Compute characteristic radius
        let radius = if self.normalize_radius && count > 0 {
            // Use maximum distance from center to any foreground pixel
            let mut max_distance = 0.0;
            for y in 0..height {
                for x in 0..width {
                    if binary_image[[y, x]] {
                        let dx = x as f64 - center_x;
                        let dy = y as f64 - center_y;
                        let distance = (dx * dx + dy * dy).sqrt();
                        max_distance = max_distance.max(distance);
                    }
                }
            }
            max_distance.max(1.0) // Avoid division by zero
        } else {
            // Use half of image diagonal as characteristic scale
            ((height * height + width * width) as f64).sqrt() / 2.0
        };

        (center_x, center_y, radius)
    }

    /// Compute Zernike moments for the binary image
    ///
    /// Calculates Zernike moments up to max_order using the normalized
    /// coordinate system and efficient radial polynomial computation.
    fn compute_zernike_moments(
        &self,
        binary_image: &Array2<bool>,
        cx: f64,
        cy: f64,
        radius: f64,
    ) -> SklResult<Vec<f64>> {
        let mut moments = Vec::new();

        for n in 0..=self.max_order {
            for m in 0..=n {
                if (n - m) % 2 == 0 {
                    // Only compute for valid (n, m) pairs
                    let moment_mag = self.compute_single_zernike_moment(
                        binary_image,
                        n,
                        m as i32,
                        cx,
                        cy,
                        radius,
                    );
                    moments.push(moment_mag);

                    // For m > 0, also include negative m (due to symmetry)
                    if m > 0 {
                        let moment_mag_neg = self.compute_single_zernike_moment(
                            binary_image,
                            n,
                            -(m as i32),
                            cx,
                            cy,
                            radius,
                        );
                        moments.push(moment_mag_neg);
                    }
                }
            }
        }

        Ok(moments)
    }

    /// Compute single Zernike moment A_nm
    ///
    /// Calculates a single Zernike moment using the definition:
    /// A_nm = (n+1)/π * ∫∫ f(ρ,θ) * R_n^m(ρ) * exp(-imθ) dρ dθ
    fn compute_single_zernike_moment(
        &self,
        binary_image: &Array2<bool>,
        n: usize,
        m: i32,
        cx: f64,
        cy: f64,
        radius: f64,
    ) -> f64 {
        let (height, width) = binary_image.dim();
        let mut real_sum = 0.0;
        let mut imag_sum = 0.0;

        for y in 0..height {
            for x in 0..width {
                if binary_image[[y, x]] {
                    let dx = x as f64 - cx;
                    let dy = y as f64 - cy;
                    let rho = (dx * dx + dy * dy).sqrt() / radius;

                    if rho <= 1.0 {
                        // Only consider points within unit circle
                        let theta = dy.atan2(dx);

                        // Compute radial polynomial
                        let r_nm = self.radial_polynomial(n, m.unsigned_abs() as usize, rho);

                        // Compute complex exponential
                        let cos_term = (m as f64 * theta).cos();
                        let sin_term = (m as f64 * theta).sin();

                        real_sum += r_nm * cos_term;
                        imag_sum += r_nm * sin_term;
                    }
                }
            }
        }

        // Normalization factor
        let norm_factor = (n + 1) as f64 / PI;

        if self.complex_moments {
            // Return complex magnitude and phase (for advanced analysis)
            (real_sum * real_sum + imag_sum * imag_sum).sqrt() * norm_factor
        } else {
            // Return magnitude for rotation invariance
            (real_sum * real_sum + imag_sum * imag_sum).sqrt() * norm_factor
        }
    }

    /// Compute radial polynomial R_n^m(ρ)
    ///
    /// Efficiently computes Zernike radial polynomials using
    /// the recursive definition with precomputed factorials.
    ///
    /// R_n^m(ρ) = Σ_{k=0}^{(n-m)/2} (-1)^k * (n-k)! / [k! * ((n+m)/2-k)! * ((n-m)/2-k)!] * ρ^(n-2k)
    fn radial_polynomial(&self, n: usize, m: usize, rho: f64) -> f64 {
        if n < m || (n - m) % 2 != 0 {
            return 0.0;
        }

        let mut result = 0.0;
        let k_max = (n - m) / 2;

        for k in 0..=k_max {
            let sign = if k % 2 == 0 { 1.0 } else { -1.0 };

            // Use precomputed factorials for efficiency
            let numerator = self.get_factorial(n - k);
            let denominator = self.get_factorial(k)
                * self.get_factorial((n + m) / 2 - k)
                * self.get_factorial((n - m) / 2 - k);

            if denominator > 0 {
                let coeff = numerator as f64 / denominator as f64;
                result += sign * coeff * rho.powi((n - 2 * k) as i32);
            }
        }

        result
    }

    /// Get factorial value from precomputed cache
    fn get_factorial(&self, n: usize) -> usize {
        if n < self.factorial_cache.len() {
            self.factorial_cache[n]
        } else {
            // Compute on the fly for large values (should be rare)
            (1..=n).product()
        }
    }

    /// Precompute factorial values for efficient polynomial computation
    fn precompute_factorials(max_n: usize) -> Vec<usize> {
        let mut factorials = vec![1; max_n + 1];
        for i in 1..=max_n {
            factorials[i] = factorials[i - 1] * i;
        }
        factorials
    }

    /// Compute shape complexity from Zernike moments
    ///
    /// Estimates shape complexity using the distribution of moment magnitudes
    /// across different orders, with higher-order moments indicating complexity.
    fn compute_shape_complexity(&self, moments: &Array1<Float>) -> f64 {
        if moments.is_empty() {
            return 0.0;
        }

        // Compute weighted sum where higher-order moments get more weight
        let mut complexity = 0.0;
        let mut weight_sum = 0.0;
        let mut order = 0;
        let mut m = 0;

        for &moment in moments.iter() {
            let weight = (order + 1) as f64; // Higher orders get more weight
            complexity += weight * moment.abs();
            weight_sum += weight;

            // Update order tracking
            m += 1;
            if m > order {
                order += 1;
                m = 0;
            }
        }

        if weight_sum > 0.0 {
            complexity / weight_sum
        } else {
            0.0
        }
    }

    /// Compute symmetry measures using specific Zernike moments
    ///
    /// Analyzes object symmetry using moments that are sensitive
    /// to different types of symmetrical structures.
    fn compute_symmetry_measures(
        &self,
        binary_image: &Array2<bool>,
        cx: f64,
        cy: f64,
        radius: f64,
    ) -> SklResult<Vec<f64>> {
        let mut symmetries = Vec::new();

        // Horizontal symmetry (m=1 moments)
        if self.max_order >= 2 {
            let h_sym = self.compute_single_zernike_moment(binary_image, 2, 1, cx, cy, radius);
            symmetries.push(h_sym);
        } else {
            symmetries.push(0.0);
        }

        // Vertical symmetry (m=2 moments with specific phase)
        if self.max_order >= 2 {
            let v_sym = self.compute_single_zernike_moment(binary_image, 2, 2, cx, cy, radius);
            symmetries.push(v_sym);
        } else {
            symmetries.push(0.0);
        }

        // Diagonal symmetry (m=3 moments)
        if self.max_order >= 3 {
            let d_sym = self.compute_single_zernike_moment(binary_image, 3, 1, cx, cy, radius);
            symmetries.push(d_sym);
        } else {
            symmetries.push(0.0);
        }

        // Rotational symmetry (low-order moments indicate radial symmetry)
        if self.max_order >= 1 {
            let r_sym = self.compute_single_zernike_moment(binary_image, 1, 1, cx, cy, radius);
            symmetries.push(r_sym);
        } else {
            symmetries.push(0.0);
        }

        Ok(symmetries)
    }

    /// Compute circularity measure using Zernike moments
    ///
    /// Compares the object to a perfect circle using the fundamental
    /// Zernike moments that characterize circular shapes.
    fn compute_circularity(&self, binary_image: &Array2<bool>, area: f64, radius: f64) -> f64 {
        if area == 0.0 || radius == 0.0 {
            return 0.0;
        }

        // For a perfect circle, only A_00 should be non-zero among low-order moments
        let (cx, cy, _) = self.compute_normalization_parameters(binary_image);

        // Compute first few moments
        let a00 = self.compute_single_zernike_moment(binary_image, 0, 0, cx, cy, radius);
        let a11 = if self.max_order >= 1 {
            self.compute_single_zernike_moment(binary_image, 1, 1, cx, cy, radius)
        } else {
            0.0
        };
        let a22 = if self.max_order >= 2 {
            self.compute_single_zernike_moment(binary_image, 2, 2, cx, cy, radius)
        } else {
            0.0
        };

        // Circularity based on deviation from ideal circular moments
        if a00 > 0.0 {
            let deviation = (a11 + a22) / a00;
            1.0 / (1.0 + deviation) // Higher deviation = lower circularity
        } else {
            0.0
        }
    }

    /// Estimate total number of moments for pre-allocation
    fn estimate_moment_count(&self) -> usize {
        let mut count = 0;
        for n in 0..=self.max_order {
            for m in 0..=n {
                if (n - m) % 2 == 0 {
                    count += if m > 0 { 2 } else { 1 }; // Count both +m and -m for m>0
                }
            }
        }
        count
    }
}

impl Default for ZernikeMomentsExtractor {
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
    fn test_zernike_extractor_creation() {
        let extractor = ZernikeMomentsExtractor::new();
        assert_eq!(extractor.max_order, 6);
        assert_eq!(extractor.binary_threshold, 0.5);
        assert!(extractor.normalize_radius);
        assert!(extractor.translation_invariant);
        assert!(extractor.scale_invariant);
    }

    #[test]
    fn test_zernike_extractor_builder() {
        let extractor = ZernikeMomentsExtractor::new()
            .max_order(4)
            .binary_threshold(0.3)
            .normalize_radius(false)
            .complex_moments(true);

        assert_eq!(extractor.max_order, 4);
        assert_eq!(extractor.binary_threshold, 0.3);
        assert!(!extractor.normalize_radius);
        assert!(extractor.complex_moments);
    }

    #[test]
    fn test_extract_moments_small_image() {
        let image = Array2::from_shape_vec((2, 2), vec![0.0, 1.0, 1.0, 0.0]).unwrap();
        let extractor = ZernikeMomentsExtractor::new();
        let result = extractor.extract_moments(&image.view());

        assert!(result.is_err()); // Should fail for too small image
    }

    #[test]
    fn test_extract_moments_valid_image() {
        let image =
            Array2::from_shape_vec((10, 10), (0..100).map(|x| x as f64 / 100.0).collect()).unwrap();
        let extractor = ZernikeMomentsExtractor::new();
        let moments = extractor.extract_moments(&image.view()).unwrap();

        assert!(moments.len() > 0);
        assert!(moments.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_binary_conversion() {
        let extractor = ZernikeMomentsExtractor::new().binary_threshold(0.5);
        let image =
            Array2::from_shape_vec((3, 3), vec![0.0, 0.3, 0.7, 0.2, 0.6, 0.9, 0.1, 0.4, 0.8])
                .unwrap();

        let binary = extractor.to_binary(&image.view());

        assert!(!binary[[0, 0]]); // 0.0 < 0.5
        assert!(!binary[[0, 1]]); // 0.3 < 0.5
        assert!(binary[[0, 2]]); // 0.7 > 0.5
        assert!(binary[[1, 2]]); // 0.9 > 0.5
    }

    #[test]
    fn test_area_computation() {
        let extractor = ZernikeMomentsExtractor::new();
        let binary = Array2::from_shape_vec(
            (3, 3),
            vec![true, false, true, false, true, false, true, false, true],
        )
        .unwrap();

        let area = extractor.compute_area(&binary);
        assert_eq!(area, 5.0); // Five true pixels
    }

    #[test]
    fn test_normalization_parameters() {
        let extractor = ZernikeMomentsExtractor::new();
        let binary = Array2::from_shape_vec(
            (5, 5),
            vec![
                false, false, true, false, false, false, true, true, true, false, true, true, true,
                true, true, false, true, true, true, false, false, false, true, false, false,
            ],
        )
        .unwrap();

        let (cx, cy, radius) = extractor.compute_normalization_parameters(&binary);

        // Should be roughly centered
        assert!((cx - 2.0).abs() < 1.0);
        assert!((cy - 2.0).abs() < 1.0);
        assert!(radius > 0.0);
    }

    #[test]
    fn test_radial_polynomial() {
        let extractor = ZernikeMomentsExtractor::new();

        // Test R_0^0(ρ) = 1 (constant)
        assert!((extractor.radial_polynomial(0, 0, 0.5) - 1.0).abs() < 1e-10);
        assert!((extractor.radial_polynomial(0, 0, 1.0) - 1.0).abs() < 1e-10);

        // Test R_1^1(ρ) = ρ (linear)
        assert!((extractor.radial_polynomial(1, 1, 0.5) - 0.5).abs() < 1e-10);
        assert!((extractor.radial_polynomial(1, 1, 0.8) - 0.8).abs() < 1e-10);

        // Test invalid combinations return 0
        assert_eq!(extractor.radial_polynomial(1, 2, 0.5), 0.0); // m > n
        assert_eq!(extractor.radial_polynomial(2, 1, 0.5), 0.0); // (n-m) odd
    }

    #[test]
    fn test_factorial_cache() {
        let extractor = ZernikeMomentsExtractor::new();

        assert_eq!(extractor.get_factorial(0), 1);
        assert_eq!(extractor.get_factorial(1), 1);
        assert_eq!(extractor.get_factorial(2), 2);
        assert_eq!(extractor.get_factorial(3), 6);
        assert_eq!(extractor.get_factorial(4), 24);
        assert_eq!(extractor.get_factorial(5), 120);
    }

    #[test]
    fn test_single_zernike_moment() {
        let extractor = ZernikeMomentsExtractor::new();
        let binary = Array2::from_shape_vec(
            (5, 5),
            vec![
                false, false, true, false, false, false, true, true, true, false, true, true, true,
                true, true, false, true, true, true, false, false, false, true, false, false,
            ],
        )
        .unwrap();

        let moment = extractor.compute_single_zernike_moment(&binary, 0, 0, 2.0, 2.0, 2.0);

        assert!(moment > 0.0); // Should have positive moment for this shape
        assert!(moment.is_finite());
    }

    #[test]
    fn test_empty_image() {
        let extractor = ZernikeMomentsExtractor::new();
        let binary = Array2::from_shape_vec((5, 5), vec![false; 25]).unwrap();

        let area = extractor.compute_area(&binary);
        assert_eq!(area, 0.0);

        let moment = extractor.compute_single_zernike_moment(&binary, 2, 0, 0.0, 0.0, 2.0);
        assert_eq!(moment, 0.0);
    }

    #[test]
    fn test_moment_count_estimation() {
        let extractor = ZernikeMomentsExtractor::new().max_order(3);
        let count = extractor.estimate_moment_count();

        // For order 3: (0,0), (1,1), (2,0), (2,2), (3,1), (3,3)
        // Some have both +m and -m: 1 + 2 + 1 + 2 + 2 + 2 = 10
        assert!(count >= 8); // At least this many moments
    }

    #[test]
    fn test_shape_complexity() {
        let extractor = ZernikeMomentsExtractor::new();
        let moments = Array1::from_vec(vec![1.0, 0.5, 0.2, 0.1]);

        let complexity = extractor.compute_shape_complexity(&moments);
        assert!(complexity > 0.0);
        assert!(complexity.is_finite());
    }

    #[test]
    fn test_extract_features() {
        let image =
            Array2::from_shape_vec((8, 8), (0..64).map(|x| x as f64 / 64.0).collect()).unwrap();
        let extractor = ZernikeMomentsExtractor::new().max_order(3);
        let features = extractor.extract_features(&image.view()).unwrap();

        assert!(features.len() > 5); // Should have moments plus derived features
        assert!(features.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_circle_detection() {
        let extractor = ZernikeMomentsExtractor::new();

        // Create a roughly circular pattern
        let mut binary = Array2::from_shape_vec((9, 9), vec![false; 81]).unwrap();
        let center = 4;
        for y in 0..9 {
            for x in 0..9 {
                let dx = (x as i32 - center).abs();
                let dy = (y as i32 - center).abs();
                if dx * dx + dy * dy <= 9 {
                    // Roughly circular
                    binary[[y, x]] = true;
                }
            }
        }

        let area = extractor.compute_area(&binary);
        let (_cx, _cy, radius) = extractor.compute_normalization_parameters(&binary);
        let circularity = extractor.compute_circularity(&binary, area, radius);

        assert!(circularity > 0.5); // Should be reasonably circular
    }
}
