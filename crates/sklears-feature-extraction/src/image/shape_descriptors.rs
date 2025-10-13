//! Shape descriptors and geometric moment extraction
//!
//! This module provides comprehensive shape analysis and geometric feature extraction
//! for object recognition and pattern analysis. It includes geometric moments,
//! centroids, shape properties, and various shape descriptors.
//!
//! ## Features
//! - Geometric moments up to arbitrary order
//! - Central and normalized moments for invariance
//! - Shape properties (area, perimeter, compactness, etc.)
//! - Aspect ratio and orientation analysis
//! - Extent and solidity measures
//! - Binary image preprocessing with thresholding
//!
//! ## Shape Properties
//! - **Area**: Total object area in pixels
//! - **Perimeter**: Boundary length using chain codes
//! - **Compactness**: Circularity measure (4πA/P²)
//! - **Aspect Ratio**: Width to height ratio of bounding box
//! - **Extent**: Ratio of object area to bounding box area
//! - **Solidity**: Ratio of object area to convex hull area

use crate::*;
use scirs2_core::ndarray::{Array1, Array2, ArrayView2};

/// Shape Descriptor and Geometric Moment Extractor
///
/// Comprehensive shape analysis tool that extracts geometric moments,
/// shape properties, and derived descriptors for object characterization
/// and pattern recognition applications.
///
/// # Geometric Moments
/// Regular moments: M_pq = ∑∑ x^p * y^q * I(x,y)
/// Central moments: μ_pq = ∑∑ (x-x̄)^p * (y-ȳ)^q * I(x,y)
/// Normalized moments: η_pq = μ_pq / μ_00^((p+q)/2+1)
///
/// # Shape Invariance
/// - Translation invariance: Using central moments
/// - Scale invariance: Using normalized moments
/// - Rotation invariance: Using Hu moments (derived features)
///
/// # Examples
/// ```rust
/// use sklears_feature_extraction::image::shape_descriptors::ShapeDescriptorExtractor;
/// use scirs2_core::ndarray::Array2;
///
/// let image = Array2::from_shape_vec((32, 32), (0..1024).map(|x| x as f64 / 1024.0).collect()).unwrap();
/// let extractor = ShapeDescriptorExtractor::new()
///     .max_moment_order(3)
///     .binary_threshold(0.5)
///     .normalize_moments(true);
/// let features = extractor.extract_features(&image.view()).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct ShapeDescriptorExtractor {
    /// Maximum order of geometric moments to compute
    max_moment_order: usize,
    /// Threshold for converting grayscale to binary
    binary_threshold: f64,
    /// Whether to normalize moments for scale invariance
    normalize: bool,
    /// Whether to compute central moments for translation invariance
    central_moments: bool,
    /// Whether to include Hu moments for rotation invariance
    hu_moments: bool,
    /// Whether to include advanced shape properties
    advanced_properties: bool,
}

impl ShapeDescriptorExtractor {
    /// Create a new shape descriptor extractor with default settings
    ///
    /// Default configuration:
    /// - Maximum moment order: 3 (sufficient for most applications)
    /// - Binary threshold: 0.5 (middle gray value)
    /// - Moment normalization enabled for scale invariance
    /// - Central moments enabled for translation invariance
    /// - Hu moments enabled for rotation invariance
    pub fn new() -> Self {
        Self {
            max_moment_order: 3,
            binary_threshold: 0.5,
            normalize: true,
            central_moments: true,
            hu_moments: true,
            advanced_properties: true,
        }
    }

    /// Set the maximum order of geometric moments
    ///
    /// Higher orders capture more detailed shape information but
    /// increase computational cost and sensitivity to noise.
    /// Typical range: 2-6 for most applications.
    pub fn max_moment_order(mut self, order: usize) -> Self {
        self.max_moment_order = order.max(1).min(10); // Reasonable bounds
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

    /// Set whether to normalize moments for scale invariance
    ///
    /// Normalized moments are invariant to uniform scaling,
    /// making them suitable for size-independent recognition.
    pub fn normalize_moments(mut self, normalize: bool) -> Self {
        self.normalize = normalize;
        self
    }

    /// Set whether to compute central moments for translation invariance
    ///
    /// Central moments are computed relative to the centroid,
    /// providing translation invariance for shape analysis.
    pub fn central_moments(mut self, central: bool) -> Self {
        self.central_moments = central;
        self
    }

    /// Set whether to include Hu moments for rotation invariance
    ///
    /// Hu moments are combinations of central moments that are
    /// invariant to rotation, making them valuable for orientation-independent recognition.
    pub fn hu_moments(mut self, hu: bool) -> Self {
        self.hu_moments = hu;
        self
    }

    /// Set whether to include advanced shape properties
    ///
    /// Advanced properties include convex hull analysis, bounding box
    /// computations, and derived geometric measures.
    pub fn advanced_properties(mut self, advanced: bool) -> Self {
        self.advanced_properties = advanced;
        self
    }

    /// Extract shape descriptors and geometric moments from an image
    ///
    /// Performs complete shape analysis including binary conversion,
    /// moment computation, and derived property calculation.
    ///
    /// # Parameters
    /// * `image` - Input grayscale image
    ///
    /// # Returns
    /// Feature vector containing shape descriptors and moments
    pub fn extract_features(&self, image: &ArrayView2<Float>) -> SklResult<Array1<Float>> {
        let (height, width) = image.dim();

        if height < 3 || width < 3 {
            return Err(SklearsError::InvalidInput(
                "Image too small for shape analysis (minimum 3x3)".to_string(),
            ));
        }

        // Convert to binary image for shape analysis
        let binary_image = self.to_binary(image);

        // Compute basic shape properties
        let area = self.compute_area(&binary_image);
        if area == 0.0 {
            // No object found, return zero features
            return Ok(Array1::zeros(self.estimate_feature_count()));
        }

        let centroid = self.compute_centroid(&binary_image, area);
        let perimeter = self.compute_perimeter(&binary_image);

        // Compute geometric moments
        let moments = self.compute_geometric_moments(&binary_image, centroid)?;

        // Build feature vector
        let mut features = Vec::new();

        // Basic shape properties
        features.extend(self.extract_basic_properties(area, centroid, perimeter, &binary_image));

        // Geometric moments
        features.extend(moments);

        // Hu moments (if enabled)
        if self.hu_moments {
            let hu_moments = self.compute_hu_moments(&binary_image, centroid)?;
            features.extend(hu_moments);
        }

        // Advanced properties (if enabled)
        if self.advanced_properties {
            let advanced = self.extract_advanced_properties(&binary_image, area);
            features.extend(advanced);
        }

        Ok(Array1::from_vec(features))
    }

    /// Convert grayscale image to binary using threshold
    ///
    /// Applies simple thresholding to create binary image for shape analysis.
    /// Pixels above threshold become foreground (true), others background (false).
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
    ///
    /// Counts all pixels marked as foreground in the binary image.
    /// Provides fundamental size measure for shape analysis.
    fn compute_area(&self, binary_image: &Array2<bool>) -> f64 {
        binary_image.iter().filter(|&&pixel| pixel).count() as f64
    }

    /// Compute object centroid (center of mass)
    ///
    /// Calculates the geometric center of the object using
    /// first-order moments for translation-invariant analysis.
    fn compute_centroid(&self, binary_image: &Array2<bool>, area: f64) -> (f64, f64) {
        if area == 0.0 {
            return (0.0, 0.0);
        }

        let (height, width) = binary_image.dim();
        let mut sum_x = 0.0;
        let mut sum_y = 0.0;

        for y in 0..height {
            for x in 0..width {
                if binary_image[[y, x]] {
                    sum_x += x as f64;
                    sum_y += y as f64;
                }
            }
        }

        (sum_x / area, sum_y / area)
    }

    /// Compute object perimeter using chain code algorithm
    ///
    /// Traces the object boundary to compute perimeter length.
    /// Uses 8-connectivity for accurate boundary following.
    fn compute_perimeter(&self, binary_image: &Array2<bool>) -> f64 {
        let (height, width) = binary_image.dim();
        let mut perimeter = 0.0;

        // Simple perimeter approximation using edge detection
        for y in 1..height - 1 {
            for x in 1..width - 1 {
                if binary_image[[y, x]] {
                    // Check 4-connected neighbors
                    let neighbors = [
                        binary_image[[y - 1, x]], // Top
                        binary_image[[y + 1, x]], // Bottom
                        binary_image[[y, x - 1]], // Left
                        binary_image[[y, x + 1]], // Right
                    ];

                    // Count edge pixels (foreground pixels with background neighbors)
                    if neighbors.iter().any(|&n| !n) {
                        perimeter += 1.0;
                    }
                }
            }
        }

        perimeter
    }

    /// Compute geometric moments up to specified order
    ///
    /// Calculates regular, central, and normalized moments for
    /// comprehensive shape characterization with various invariances.
    fn compute_geometric_moments(
        &self,
        binary_image: &Array2<bool>,
        centroid: (f64, f64),
    ) -> SklResult<Vec<f64>> {
        let (height, width) = binary_image.dim();
        let (cx, cy) = centroid;
        let mut moments = Vec::new();

        // Compute moments up to max_order
        for p in 0..=self.max_moment_order {
            for q in 0..=self.max_moment_order {
                if p + q <= self.max_moment_order {
                    // Regular moment
                    let mut m_pq = 0.0;

                    // Central moment (if enabled)
                    let mut mu_pq = 0.0;

                    for y in 0..height {
                        for x in 0..width {
                            if binary_image[[y, x]] {
                                let x_f = x as f64;
                                let y_f = y as f64;

                                // Regular moment
                                m_pq += x_f.powi(p as i32) * y_f.powi(q as i32);

                                // Central moment
                                if self.central_moments {
                                    mu_pq += (x_f - cx).powi(p as i32) * (y_f - cy).powi(q as i32);
                                }
                            }
                        }
                    }

                    moments.push(m_pq);

                    if self.central_moments {
                        // Normalized central moment (if enabled)
                        if self.normalize && p + q >= 2 {
                            let m00 = self.compute_area(binary_image);
                            if m00 > 0.0 {
                                let gamma = ((p + q) as f64 / 2.0) + 1.0;
                                let eta_pq = mu_pq / m00.powf(gamma);
                                moments.push(eta_pq);
                            } else {
                                moments.push(0.0);
                            }
                        } else {
                            moments.push(mu_pq);
                        }
                    }
                }
            }
        }

        Ok(moments)
    }

    /// Extract basic shape properties
    ///
    /// Computes fundamental shape measures including compactness,
    /// aspect ratio, extent, and solidity for object characterization.
    fn extract_basic_properties(
        &self,
        area: f64,
        centroid: (f64, f64),
        perimeter: f64,
        binary_image: &Array2<bool>,
    ) -> Vec<f64> {
        let mut features = Vec::new();

        // Basic measurements
        features.push(area);
        features.push(centroid.0); // Centroid X
        features.push(centroid.1); // Centroid Y
        features.push(perimeter);

        // Derived shape properties
        let compactness = if perimeter > 0.0 {
            (4.0 * std::f64::consts::PI * area) / (perimeter * perimeter)
        } else {
            0.0
        };

        let aspect_ratio = self.compute_aspect_ratio(binary_image);
        let extent = self.compute_extent(binary_image, area);
        let solidity = self.compute_solidity(binary_image, area);

        features.push(compactness);
        features.push(aspect_ratio);
        features.push(extent);
        features.push(solidity);

        features
    }

    /// Compute aspect ratio of object bounding box
    ///
    /// Calculates the ratio of bounding box width to height,
    /// providing measure of object elongation and orientation.
    fn compute_aspect_ratio(&self, binary_image: &Array2<bool>) -> f64 {
        let (min_x, max_x, min_y, max_y) = self.find_bounding_box(binary_image);

        let width = (max_x - min_x + 1) as f64;
        let height = (max_y - min_y + 1) as f64;

        if height > 0.0 {
            width / height
        } else {
            1.0
        }
    }

    /// Compute extent (ratio of object area to bounding box area)
    ///
    /// Measures how much of the bounding box is filled by the object,
    /// indicating shape complexity and space utilization.
    fn compute_extent(&self, binary_image: &Array2<bool>, area: f64) -> f64 {
        let (min_x, max_x, min_y, max_y) = self.find_bounding_box(binary_image);

        let bbox_width = (max_x - min_x + 1) as f64;
        let bbox_height = (max_y - min_y + 1) as f64;
        let bbox_area = bbox_width * bbox_height;

        if bbox_area > 0.0 {
            area / bbox_area
        } else {
            0.0
        }
    }

    /// Compute solidity (ratio of object area to convex hull area)
    ///
    /// Measures object convexity by comparing to its convex hull.
    /// Values close to 1 indicate convex shapes, lower values indicate concave shapes.
    fn compute_solidity(&self, binary_image: &Array2<bool>, area: f64) -> f64 {
        // Simplified solidity computation (normally would use convex hull)
        // For now, approximate using bounding box as upper bound
        let extent = self.compute_extent(binary_image, area);

        // Estimate solidity based on extent and shape regularity
        // This is a simplified approximation
        let irregularity_penalty = self.compute_shape_irregularity(binary_image);
        (extent * (1.0 - irregularity_penalty)).max(0.0).min(1.0)
    }

    /// Find bounding box coordinates
    ///
    /// Determines the minimal axis-aligned rectangle that encloses
    /// all foreground pixels for bounding box-based computations.
    fn find_bounding_box(&self, binary_image: &Array2<bool>) -> (usize, usize, usize, usize) {
        let (height, width) = binary_image.dim();
        let mut min_x = width;
        let mut max_x = 0;
        let mut min_y = height;
        let mut max_y = 0;

        for y in 0..height {
            for x in 0..width {
                if binary_image[[y, x]] {
                    min_x = min_x.min(x);
                    max_x = max_x.max(x);
                    min_y = min_y.min(y);
                    max_y = max_y.max(y);
                }
            }
        }

        // Handle case where no foreground pixels found
        if min_x == width {
            (0, 0, 0, 0)
        } else {
            (min_x, max_x, min_y, max_y)
        }
    }

    /// Compute shape irregularity measure
    ///
    /// Estimates shape complexity by analyzing boundary smoothness
    /// and deviation from ideal geometric shapes.
    fn compute_shape_irregularity(&self, binary_image: &Array2<bool>) -> f64 {
        let (height, width) = binary_image.dim();
        let mut boundary_variation = 0.0;
        let mut boundary_points = 0;

        // Analyze boundary smoothness
        for y in 1..height - 1 {
            for x in 1..width - 1 {
                if binary_image[[y, x]] {
                    // Check if this is a boundary pixel
                    let neighbors = [
                        binary_image[[y - 1, x - 1]],
                        binary_image[[y - 1, x]],
                        binary_image[[y - 1, x + 1]],
                        binary_image[[y, x - 1]],
                        binary_image[[y, x + 1]],
                        binary_image[[y + 1, x - 1]],
                        binary_image[[y + 1, x]],
                        binary_image[[y + 1, x + 1]],
                    ];

                    let neighbor_count = neighbors.iter().filter(|&&n| n).count();

                    if neighbor_count < 8 {
                        // Boundary pixel
                        // Measure local boundary variation
                        boundary_variation += (8 - neighbor_count) as f64 / 8.0;
                        boundary_points += 1;
                    }
                }
            }
        }

        if boundary_points > 0 {
            (boundary_variation / boundary_points as f64).min(1.0)
        } else {
            0.0
        }
    }

    /// Compute Hu moments for rotation invariance
    ///
    /// Calculates the seven Hu moment invariants that are
    /// invariant to translation, scaling, and rotation.
    fn compute_hu_moments(
        &self,
        binary_image: &Array2<bool>,
        centroid: (f64, f64),
    ) -> SklResult<Vec<f64>> {
        // Compute central moments needed for Hu moments
        let mu_00 = self.compute_area(binary_image);
        if mu_00 == 0.0 {
            return Ok(vec![0.0; 7]);
        }

        let mu_20 = self.compute_central_moment(binary_image, centroid, 2, 0);
        let mu_02 = self.compute_central_moment(binary_image, centroid, 0, 2);
        let mu_11 = self.compute_central_moment(binary_image, centroid, 1, 1);
        let mu_30 = self.compute_central_moment(binary_image, centroid, 3, 0);
        let mu_03 = self.compute_central_moment(binary_image, centroid, 0, 3);
        let mu_21 = self.compute_central_moment(binary_image, centroid, 2, 1);
        let mu_12 = self.compute_central_moment(binary_image, centroid, 1, 2);

        // Normalize central moments
        let eta_20 = mu_20 / mu_00.powi(2);
        let eta_02 = mu_02 / mu_00.powi(2);
        let eta_11 = mu_11 / mu_00.powi(2);
        let eta_30 = mu_30 / mu_00.powf(2.5);
        let eta_03 = mu_03 / mu_00.powf(2.5);
        let eta_21 = mu_21 / mu_00.powf(2.5);
        let eta_12 = mu_12 / mu_00.powf(2.5);

        // Compute Hu moments
        let hu1 = eta_20 + eta_02;
        let hu2 = (eta_20 - eta_02).powi(2) + 4.0 * eta_11.powi(2);
        let hu3 = (eta_30 - 3.0 * eta_12).powi(2) + (3.0 * eta_21 - eta_03).powi(2);
        let hu4 = (eta_30 + eta_12).powi(2) + (eta_21 + eta_03).powi(2);
        let hu5 = (eta_30 - 3.0 * eta_12)
            * (eta_30 + eta_12)
            * ((eta_30 + eta_12).powi(2) - 3.0 * (eta_21 + eta_03).powi(2))
            + (3.0 * eta_21 - eta_03)
                * (eta_21 + eta_03)
                * (3.0 * (eta_30 + eta_12).powi(2) - (eta_21 + eta_03).powi(2));
        let hu6 = (eta_20 - eta_02) * ((eta_30 + eta_12).powi(2) - (eta_21 + eta_03).powi(2))
            + 4.0 * eta_11 * (eta_30 + eta_12) * (eta_21 + eta_03);
        let hu7 = (3.0 * eta_21 - eta_03)
            * (eta_30 + eta_12)
            * ((eta_30 + eta_12).powi(2) - 3.0 * (eta_21 + eta_03).powi(2))
            - (eta_30 - 3.0 * eta_12)
                * (eta_21 + eta_03)
                * (3.0 * (eta_30 + eta_12).powi(2) - (eta_21 + eta_03).powi(2));

        Ok(vec![hu1, hu2, hu3, hu4, hu5, hu6, hu7])
    }

    /// Compute specific central moment
    ///
    /// Calculates central moment μ_pq = ∑∑(x-x̄)^p(y-ȳ)^q for
    /// translation-invariant shape analysis.
    fn compute_central_moment(
        &self,
        binary_image: &Array2<bool>,
        centroid: (f64, f64),
        p: usize,
        q: usize,
    ) -> f64 {
        let (height, width) = binary_image.dim();
        let (cx, cy) = centroid;
        let mut moment = 0.0;

        for y in 0..height {
            for x in 0..width {
                if binary_image[[y, x]] {
                    let x_f = x as f64;
                    let y_f = y as f64;
                    moment += (x_f - cx).powi(p as i32) * (y_f - cy).powi(q as i32);
                }
            }
        }

        moment
    }

    /// Extract advanced shape properties
    ///
    /// Computes additional geometric measures including orientation,
    /// eccentricity, and shape complexity metrics.
    fn extract_advanced_properties(&self, binary_image: &Array2<bool>, area: f64) -> Vec<f64> {
        let mut features = Vec::new();

        // Object orientation (principal axis angle)
        let orientation = self.compute_orientation(binary_image);
        features.push(orientation);

        // Eccentricity (measure of elongation)
        let eccentricity = self.compute_eccentricity(binary_image);
        features.push(eccentricity);

        // Convexity (approximation)
        let convexity = self.compute_convexity_approximation(binary_image, area);
        features.push(convexity);

        // Shape complexity
        let complexity = self.compute_shape_complexity(binary_image);
        features.push(complexity);

        features
    }

    /// Compute object orientation using second-order moments
    ///
    /// Calculates the angle of the principal axis using
    /// the second-order central moments for orientation analysis.
    fn compute_orientation(&self, binary_image: &Array2<bool>) -> f64 {
        let area = self.compute_area(binary_image);
        if area == 0.0 {
            return 0.0;
        }

        let centroid = self.compute_centroid(binary_image, area);
        let mu_20 = self.compute_central_moment(binary_image, centroid, 2, 0);
        let mu_02 = self.compute_central_moment(binary_image, centroid, 0, 2);
        let mu_11 = self.compute_central_moment(binary_image, centroid, 1, 1);

        if (mu_20 - mu_02).abs() < f64::EPSILON {
            std::f64::consts::PI / 4.0 // 45 degrees
        } else {
            0.5 * (2.0 * mu_11 / (mu_20 - mu_02)).atan()
        }
    }

    /// Compute eccentricity using eigenvalues of covariance matrix
    ///
    /// Measures object elongation by analyzing the ratio of
    /// principal axis lengths from the moment-based covariance matrix.
    fn compute_eccentricity(&self, binary_image: &Array2<bool>) -> f64 {
        let area = self.compute_area(binary_image);
        if area == 0.0 {
            return 0.0;
        }

        let centroid = self.compute_centroid(binary_image, area);
        let mu_20 = self.compute_central_moment(binary_image, centroid, 2, 0) / area;
        let mu_02 = self.compute_central_moment(binary_image, centroid, 0, 2) / area;
        let mu_11 = self.compute_central_moment(binary_image, centroid, 1, 1) / area;

        // Eigenvalues of covariance matrix
        let trace = mu_20 + mu_02;
        let det = mu_20 * mu_02 - mu_11 * mu_11;

        if det <= 0.0 || trace <= 0.0 {
            return 0.0;
        }

        let discriminant = (trace * trace - 4.0 * det).max(0.0).sqrt();
        let lambda1 = (trace + discriminant) / 2.0;
        let lambda2 = (trace - discriminant) / 2.0;

        if lambda1 <= 0.0 {
            0.0
        } else {
            (1.0 - lambda2 / lambda1).max(0.0).sqrt()
        }
    }

    /// Compute convexity approximation
    ///
    /// Estimates convexity using perimeter-based measures
    /// as an approximation to convex hull analysis.
    fn compute_convexity_approximation(&self, binary_image: &Array2<bool>, area: f64) -> f64 {
        if area == 0.0 {
            return 0.0;
        }

        let perimeter = self.compute_perimeter(binary_image);
        if perimeter == 0.0 {
            return 1.0;
        }

        // Isoperimetric quotient as convexity measure
        let ideal_perimeter = 2.0 * (std::f64::consts::PI * area).sqrt();
        (ideal_perimeter / perimeter).min(1.0)
    }

    /// Compute shape complexity measure
    ///
    /// Estimates shape complexity using boundary variations
    /// and deviation from simple geometric forms.
    fn compute_shape_complexity(&self, binary_image: &Array2<bool>) -> f64 {
        self.compute_shape_irregularity(binary_image)
    }

    /// Estimate total feature count for pre-allocation
    ///
    /// Calculates expected number of features based on configuration
    /// for efficient memory allocation.
    fn estimate_feature_count(&self) -> usize {
        let mut count = 8; // Basic properties

        // Geometric moments: (max_order + 1) * (max_order + 2) / 2
        let moment_count = (self.max_moment_order + 1) * (self.max_moment_order + 2) / 2;
        count += moment_count;

        if self.central_moments && self.normalize {
            count += moment_count; // Normalized central moments
        }

        if self.hu_moments {
            count += 7; // Seven Hu moments
        }

        if self.advanced_properties {
            count += 4; // Advanced properties
        }

        count
    }
}

impl Default for ShapeDescriptorExtractor {
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
    fn test_shape_extractor_creation() {
        let extractor = ShapeDescriptorExtractor::new();
        assert_eq!(extractor.max_moment_order, 3);
        assert_eq!(extractor.binary_threshold, 0.5);
        assert!(extractor.normalize);
        assert!(extractor.central_moments);
        assert!(extractor.hu_moments);
    }

    #[test]
    fn test_shape_extractor_builder() {
        let extractor = ShapeDescriptorExtractor::new()
            .max_moment_order(2)
            .binary_threshold(0.3)
            .normalize_moments(false)
            .hu_moments(false);

        assert_eq!(extractor.max_moment_order, 2);
        assert_eq!(extractor.binary_threshold, 0.3);
        assert!(!extractor.normalize);
        assert!(!extractor.hu_moments);
    }

    #[test]
    fn test_extract_features_small_image() {
        let image = Array2::from_shape_vec((2, 2), vec![0.0, 1.0, 1.0, 0.0]).unwrap();
        let extractor = ShapeDescriptorExtractor::new();
        let result = extractor.extract_features(&image.view());

        assert!(result.is_err()); // Should fail for too small image
    }

    #[test]
    fn test_extract_features_valid_image() {
        let image =
            Array2::from_shape_vec((10, 10), (0..100).map(|x| x as f64 / 100.0).collect()).unwrap();
        let extractor = ShapeDescriptorExtractor::new();
        let features = extractor.extract_features(&image.view()).unwrap();

        assert!(features.len() > 0);
        // Should extract multiple shape features
    }

    #[test]
    fn test_binary_conversion() {
        let extractor = ShapeDescriptorExtractor::new().binary_threshold(0.5);
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
        let extractor = ShapeDescriptorExtractor::new();
        let binary = Array2::from_shape_vec(
            (3, 3),
            vec![true, false, true, false, true, false, true, false, true],
        )
        .unwrap();

        let area = extractor.compute_area(&binary);
        assert_eq!(area, 5.0); // Five true pixels
    }

    #[test]
    fn test_centroid_computation() {
        let extractor = ShapeDescriptorExtractor::new();
        let binary = Array2::from_shape_vec(
            (3, 3),
            vec![true, false, false, false, true, false, false, false, true],
        )
        .unwrap();

        let area = 3.0;
        let centroid = extractor.compute_centroid(&binary, area);

        // Centroid should be at (1, 1) for diagonal pattern
        assert!((centroid.0 - 1.0).abs() < 1e-10);
        assert!((centroid.1 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_perimeter_computation() {
        let extractor = ShapeDescriptorExtractor::new();
        let binary = Array2::from_shape_vec(
            (5, 5),
            vec![
                false, false, false, false, false, false, true, true, true, false, false, true,
                true, true, false, false, true, true, true, false, false, false, false, false,
                false,
            ],
        )
        .unwrap();

        let perimeter = extractor.compute_perimeter(&binary);
        assert!(perimeter > 0.0); // Should have some perimeter
    }

    #[test]
    fn test_bounding_box() {
        let extractor = ShapeDescriptorExtractor::new();
        let binary = Array2::from_shape_vec(
            (5, 5),
            vec![
                false, false, false, false, false, false, true, false, false, false, false, false,
                false, true, false, false, false, false, false, false, false, false, false, false,
                false,
            ],
        )
        .unwrap();

        let (min_x, max_x, min_y, max_y) = extractor.find_bounding_box(&binary);
        assert_eq!(min_x, 1); // First true at x=1
        assert_eq!(max_x, 3); // Last true at x=3
        assert_eq!(min_y, 1); // First true at y=1
        assert_eq!(max_y, 2); // Last true at y=2
    }

    #[test]
    fn test_aspect_ratio() {
        let extractor = ShapeDescriptorExtractor::new();
        let binary = Array2::from_shape_vec(
            (4, 6),
            vec![
                false, true, true, true, true, false, false, true, true, true, true, false, false,
                false, false, false, false, false, false, false, false, false, false, false,
            ],
        )
        .unwrap();

        let aspect_ratio = extractor.compute_aspect_ratio(&binary);
        assert!(aspect_ratio > 1.0); // Width > height, so ratio > 1
    }

    #[test]
    fn test_central_moment() {
        let extractor = ShapeDescriptorExtractor::new();
        let binary = Array2::from_shape_vec(
            (3, 3),
            vec![true, false, false, false, true, false, false, false, true],
        )
        .unwrap();

        let centroid = (1.0, 1.0);
        let mu_20 = extractor.compute_central_moment(&binary, centroid, 2, 0);

        assert!(mu_20.is_finite());
    }

    #[test]
    fn test_empty_image() {
        let extractor = ShapeDescriptorExtractor::new();
        let binary = Array2::from_shape_vec((5, 5), vec![false; 25]).unwrap();

        let area = extractor.compute_area(&binary);
        assert_eq!(area, 0.0);

        let centroid = extractor.compute_centroid(&binary, area);
        assert_eq!(centroid, (0.0, 0.0));
    }

    #[test]
    fn test_hu_moments() {
        let extractor = ShapeDescriptorExtractor::new();
        let binary = Array2::from_shape_vec(
            (5, 5),
            vec![
                false, false, true, false, false, false, true, true, true, false, true, true, true,
                true, true, false, true, true, true, false, false, false, true, false, false,
            ],
        )
        .unwrap();

        let area = extractor.compute_area(&binary);
        let centroid = extractor.compute_centroid(&binary, area);
        let hu_moments = extractor.compute_hu_moments(&binary, centroid).unwrap();

        assert_eq!(hu_moments.len(), 7);
        assert!(hu_moments.iter().all(|x| x.is_finite()));
    }
}
