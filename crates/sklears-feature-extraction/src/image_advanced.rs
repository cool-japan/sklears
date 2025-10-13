//! Advanced image feature extraction utilities
//!
//! This module contains additional advanced image processing and feature extraction
//! capabilities including contour analysis and morphological features.

use crate::*;
use scirs2_core::ndarray::{Array2, ArrayView2};

/// Contour Analysis Extractor
///
/// Extract shape features from contours in binary images, including contour properties,
/// hierarchy analysis, and geometric measurements.
///
/// # Parameters
///
/// * `min_contour_area` - Minimum area threshold for contour detection
/// * `max_contours` - Maximum number of contours to analyze
/// * `chain_approximation` - Contour approximation method
/// * `hierarchy_analysis` - Whether to analyze contour hierarchy
///
/// # Features Extracted
///
/// * Contour count and total area
/// * Perimeter and arc length statistics
/// * Convexity measures and defects
/// * Orientation and bounding box properties
/// * Solidity and extent ratios
/// * Hierarchy depth and parent-child relationships
///
/// # Examples
///
/// ```
/// use sklears_feature_extraction::image_advanced::ContourAnalysisExtractor;
/// use scirs2_core::ndarray::Array2;
///
/// let image = Array2::from_shape_vec((16, 16),
///     (0..256).map(|x| if x % 8 < 4 { 1.0 } else { 0.0 }).collect()).unwrap();
///
/// let contour_extractor = ContourAnalysisExtractor::new()
///     .min_contour_area(10.0)
///     .max_contours(Some(5));
///
/// let features = contour_extractor.extract_features(&image.view()).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct ContourAnalysisExtractor {
    min_contour_area: f64,
    max_contours: Option<usize>,
    chain_approximation: ChainApproximation,
    hierarchy_analysis: bool,
    binary_threshold: f64,
}

#[derive(Debug, Clone, Copy)]
pub enum ChainApproximation {
    None, // Store all contour points
    /// Simple
    Simple, // Compress horizontal, vertical, and diagonal segments
    /// TC89L1
    TC89L1, // Apply Teh-Chin chain approximation algorithm
    /// TC89KCOS
    TC89KCOS, // Apply Teh-Chin algorithm with K-cosine curvature
}

impl ContourAnalysisExtractor {
    /// Create a new ContourAnalysisExtractor
    pub fn new() -> Self {
        Self {
            min_contour_area: 10.0,
            max_contours: None,
            chain_approximation: ChainApproximation::Simple,
            hierarchy_analysis: true,
            binary_threshold: 0.5,
        }
    }

    /// Set minimum contour area threshold
    pub fn min_contour_area(mut self, min_area: f64) -> Self {
        self.min_contour_area = min_area;
        self
    }

    /// Set maximum number of contours to analyze
    pub fn max_contours(mut self, max_contours: Option<usize>) -> Self {
        self.max_contours = max_contours;
        self
    }

    /// Set chain approximation method
    pub fn chain_approximation(mut self, method: ChainApproximation) -> Self {
        self.chain_approximation = method;
        self
    }

    /// Set whether to perform hierarchy analysis
    pub fn hierarchy_analysis(mut self, enable: bool) -> Self {
        self.hierarchy_analysis = enable;
        self
    }

    /// Set binary threshold for image binarization
    pub fn binary_threshold(mut self, threshold: f64) -> Self {
        self.binary_threshold = threshold;
        self
    }

    /// Extract contour features from image
    pub fn extract_features(&self, image: &ArrayView2<f64>) -> SklResult<Vec<f64>> {
        let binary_image = self.binarize_image(image);
        let contours = self.find_contours(&binary_image)?;
        let filtered_contours = self.filter_contours(contours);

        let mut features = Vec::new();

        // Basic contour statistics
        features.push(filtered_contours.len() as f64); // Number of contours

        let total_area: f64 = filtered_contours.iter().map(|c| c.area).sum();
        features.push(total_area); // Total contour area

        let total_perimeter: f64 = filtered_contours.iter().map(|c| c.perimeter).sum();
        features.push(total_perimeter); // Total perimeter

        // Statistics of individual contours
        if !filtered_contours.is_empty() {
            let areas: Vec<f64> = filtered_contours.iter().map(|c| c.area).collect();
            let perimeters: Vec<f64> = filtered_contours.iter().map(|c| c.perimeter).collect();
            let solidities: Vec<f64> = filtered_contours.iter().map(|c| c.solidity).collect();
            let extents: Vec<f64> = filtered_contours.iter().map(|c| c.extent).collect();
            let convexities: Vec<f64> = filtered_contours.iter().map(|c| c.convexity).collect();

            // Area statistics
            features.push(self.mean(&areas)); // Mean area
            features.push(self.std_dev(&areas)); // Std dev of areas
            features.push(
                *areas
                    .iter()
                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                    .unwrap(),
            ); // Max area
            features.push(
                *areas
                    .iter()
                    .min_by(|a, b| a.partial_cmp(b).unwrap())
                    .unwrap(),
            ); // Min area

            // Perimeter statistics
            features.push(self.mean(&perimeters)); // Mean perimeter
            features.push(self.std_dev(&perimeters)); // Std dev of perimeters

            // Shape characteristics
            features.push(self.mean(&solidities)); // Mean solidity
            features.push(self.mean(&extents)); // Mean extent
            features.push(self.mean(&convexities)); // Mean convexity

            // Aspect ratio statistics
            let aspect_ratios: Vec<f64> =
                filtered_contours.iter().map(|c| c.aspect_ratio).collect();
            features.push(self.mean(&aspect_ratios)); // Mean aspect ratio
            features.push(self.std_dev(&aspect_ratios)); // Std dev of aspect ratios

            // Orientation statistics
            let orientations: Vec<f64> = filtered_contours.iter().map(|c| c.orientation).collect();
            features.push(self.mean(&orientations)); // Mean orientation
            features.push(self.std_dev(&orientations)); // Std dev of orientations
        } else {
            // No contours found - add zeros
            for _ in 0..11 {
                features.push(0.0);
            }
        }

        // Hierarchy analysis if enabled
        if self.hierarchy_analysis {
            let hierarchy_features = self.analyze_hierarchy(&filtered_contours);
            features.extend(hierarchy_features);
        }

        Ok(features)
    }

    fn binarize_image(&self, image: &ArrayView2<f64>) -> Array2<bool> {
        let (height, width) = image.dim();
        let mut binary = Array2::default((height, width));

        for y in 0..height {
            for x in 0..width {
                binary[[y, x]] = image[[y, x]] > self.binary_threshold;
            }
        }

        binary
    }

    fn find_contours(&self, binary_image: &Array2<bool>) -> SklResult<Vec<Contour>> {
        let (height, width) = binary_image.dim();
        let mut contours = Vec::new();
        let mut visited = Array2::<bool>::default((height, width));

        // Simple contour detection using border following
        for y in 1..height - 1 {
            for x in 1..width - 1 {
                if binary_image[[y, x]] && !visited[[y, x]] {
                    if let Some(contour) = self.trace_contour(binary_image, &mut visited, x, y) {
                        contours.push(contour);
                    }
                }
            }
        }

        Ok(contours)
    }

    fn trace_contour(
        &self,
        binary_image: &Array2<bool>,
        visited: &mut Array2<bool>,
        start_x: usize,
        start_y: usize,
    ) -> Option<Contour> {
        let mut points = Vec::new();
        let mut x = start_x;
        let mut y = start_y;

        // 8-connectivity directions
        let directions = [
            (-1, -1),
            (0, -1),
            (1, -1),
            (-1, 0),
            (1, 0),
            (-1, 1),
            (0, 1),
            (1, 1),
        ];

        let mut direction_index = 0;
        let start_point = (x, y);

        loop {
            visited[[y, x]] = true;
            points.push((x as f64, y as f64));

            // Find next contour point
            let mut found_next = false;
            for i in 0..8 {
                let dir_idx = (direction_index + i) % 8;
                let (dx, dy) = directions[dir_idx];
                let next_x = (x as i32 + dx) as usize;
                let next_y = (y as i32 + dy) as usize;

                if next_x < binary_image.ncols()
                    && next_y < binary_image.nrows()
                    && binary_image[[next_y, next_x]]
                    && !visited[[next_y, next_x]]
                {
                    x = next_x;
                    y = next_y;
                    direction_index = dir_idx;
                    found_next = true;
                    break;
                }
            }

            if !found_next || (x, y) == start_point {
                break;
            }

            if points.len() > 10000 {
                // Prevent infinite loops
                break;
            }
        }

        if points.len() < 3 {
            return None;
        }

        Some(self.create_contour_from_points(points))
    }

    fn create_contour_from_points(&self, points: Vec<(f64, f64)>) -> Contour {
        let area = self.compute_contour_area(&points);
        let perimeter = self.compute_contour_perimeter(&points);
        let convex_hull = self.compute_convex_hull(&points);
        let convex_area = self.compute_contour_area(&convex_hull);
        let bounding_box = self.compute_bounding_box(&points);

        let solidity = if convex_area > 0.0 {
            area / convex_area
        } else {
            0.0
        };
        let extent = if bounding_box.2 * bounding_box.3 > 0.0 {
            area / (bounding_box.2 * bounding_box.3)
        } else {
            0.0
        };
        let convexity = if perimeter > 0.0 {
            self.compute_contour_perimeter(&convex_hull) / perimeter
        } else {
            0.0
        };
        let aspect_ratio = if bounding_box.3 > 0.0 {
            bounding_box.2 / bounding_box.3
        } else {
            1.0
        };
        let orientation = self.compute_orientation(&points);

        /// Contour
        Contour {
            points,
            area,
            perimeter,
            convex_hull,
            solidity,
            extent,
            convexity,
            aspect_ratio,
            orientation,
            bounding_box,
        }
    }

    fn compute_contour_area(&self, points: &[(f64, f64)]) -> f64 {
        if points.len() < 3 {
            return 0.0;
        }

        let mut area = 0.0;
        let n = points.len();

        for i in 0..n {
            let j = (i + 1) % n;
            area += points[i].0 * points[j].1;
            area -= points[j].0 * points[i].1;
        }

        (area / 2.0).abs()
    }

    fn compute_contour_perimeter(&self, points: &[(f64, f64)]) -> f64 {
        if points.len() < 2 {
            return 0.0;
        }

        let mut perimeter = 0.0;
        for i in 0..points.len() {
            let j = (i + 1) % points.len();
            let dx = points[j].0 - points[i].0;
            let dy = points[j].1 - points[i].1;
            perimeter += (dx * dx + dy * dy).sqrt();
        }

        perimeter
    }

    fn compute_convex_hull(&self, points: &[(f64, f64)]) -> Vec<(f64, f64)> {
        if points.len() < 3 {
            return points.to_vec();
        }

        // Graham scan algorithm for convex hull
        let mut sorted_points = points.to_vec();

        // Find the bottom-most point (and left-most in case of tie)
        let mut min_idx = 0;
        for i in 1..sorted_points.len() {
            if sorted_points[i].1 < sorted_points[min_idx].1
                || (sorted_points[i].1 == sorted_points[min_idx].1
                    && sorted_points[i].0 < sorted_points[min_idx].0)
            {
                min_idx = i;
            }
        }
        sorted_points.swap(0, min_idx);

        let p0 = sorted_points[0];

        // Sort points by polar angle with respect to p0
        sorted_points[1..].sort_by(|a, b| {
            let cross = self.cross_product(p0, *a, *b);
            if cross == 0.0 {
                // Collinear points - sort by distance
                let dist_a = (a.0 - p0.0).powi(2) + (a.1 - p0.1).powi(2);
                let dist_b = (b.0 - p0.0).powi(2) + (b.1 - p0.1).powi(2);
                dist_a.partial_cmp(&dist_b).unwrap()
            } else {
                cross.partial_cmp(&0.0).unwrap().reverse()
            }
        });

        // Build convex hull
        let mut hull = Vec::new();
        for point in sorted_points {
            while hull.len() > 1
                && self.cross_product(hull[hull.len() - 2], hull[hull.len() - 1], point) <= 0.0
            {
                hull.pop();
            }
            hull.push(point);
        }

        hull
    }

    fn cross_product(&self, o: (f64, f64), a: (f64, f64), b: (f64, f64)) -> f64 {
        (a.0 - o.0) * (b.1 - o.1) - (a.1 - o.1) * (b.0 - o.0)
    }

    fn compute_bounding_box(&self, points: &[(f64, f64)]) -> (f64, f64, f64, f64) {
        if points.is_empty() {
            return (0.0, 0.0, 0.0, 0.0);
        }

        let mut min_x = points[0].0;
        let mut max_x = points[0].0;
        let mut min_y = points[0].1;
        let mut max_y = points[0].1;

        for &(x, y) in points.iter().skip(1) {
            min_x = min_x.min(x);
            max_x = max_x.max(x);
            min_y = min_y.min(y);
            max_y = max_y.max(y);
        }

        (min_x, min_y, max_x - min_x, max_y - min_y)
    }

    fn compute_orientation(&self, points: &[(f64, f64)]) -> f64 {
        if points.len() < 2 {
            return 0.0;
        }

        // Compute the orientation of the major axis using moments
        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let n = points.len() as f64;

        for &(x, y) in points {
            sum_x += x;
            sum_y += y;
        }

        let mean_x = sum_x / n;
        let mean_y = sum_y / n;

        let mut m11 = 0.0;
        let mut m20 = 0.0;
        let mut m02 = 0.0;

        for &(x, y) in points {
            let dx = x - mean_x;
            let dy = y - mean_y;
            m11 += dx * dy;
            m20 += dx * dx;
            m02 += dy * dy;
        }

        if m20 == m02 {
            0.0
        } else {
            0.5 * (2.0 * m11).atan2(m20 - m02)
        }
    }

    fn filter_contours(&self, mut contours: Vec<Contour>) -> Vec<Contour> {
        // Filter by area
        contours.retain(|c| c.area >= self.min_contour_area);

        // Sort by area (largest first)
        contours.sort_by(|a, b| b.area.partial_cmp(&a.area).unwrap());

        // Limit number of contours
        if let Some(max_count) = self.max_contours {
            contours.truncate(max_count);
        }

        contours
    }

    fn analyze_hierarchy(&self, contours: &[Contour]) -> Vec<f64> {
        let mut hierarchy_features = Vec::new();

        // Simple hierarchy analysis - check containment
        let mut hierarchy_levels = Vec::new();

        for (i, contour_i) in contours.iter().enumerate() {
            let mut level = 0;
            for (j, contour_j) in contours.iter().enumerate() {
                if i != j && self.is_contour_inside(contour_i, contour_j) {
                    level += 1;
                }
            }
            hierarchy_levels.push(level as f64);
        }

        if !hierarchy_levels.is_empty() {
            hierarchy_features.push(
                *hierarchy_levels
                    .iter()
                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                    .unwrap(),
            ); // Max depth
            hierarchy_features.push(self.mean(&hierarchy_levels)); // Mean depth
            hierarchy_features.push(self.std_dev(&hierarchy_levels)); // Std dev of depths
        } else {
            hierarchy_features.extend(vec![0.0, 0.0, 0.0]);
        }

        hierarchy_features
    }

    fn is_contour_inside(&self, inner: &Contour, outer: &Contour) -> bool {
        // Simple point-in-polygon test for centroid
        if inner.area >= outer.area {
            return false;
        }

        let centroid = self.compute_centroid(&inner.points);
        self.point_in_polygon(centroid, &outer.points)
    }

    fn compute_centroid(&self, points: &[(f64, f64)]) -> (f64, f64) {
        let n = points.len() as f64;
        let sum_x: f64 = points.iter().map(|p| p.0).sum();
        let sum_y: f64 = points.iter().map(|p| p.1).sum();
        (sum_x / n, sum_y / n)
    }

    fn point_in_polygon(&self, point: (f64, f64), polygon: &[(f64, f64)]) -> bool {
        let mut inside = false;
        let n = polygon.len();

        let mut j = n - 1;
        for i in 0..n {
            if ((polygon[i].1 > point.1) != (polygon[j].1 > point.1))
                && (point.0
                    < (polygon[j].0 - polygon[i].0) * (point.1 - polygon[i].1)
                        / (polygon[j].1 - polygon[i].1)
                        + polygon[i].0)
            {
                inside = !inside;
            }
            j = i;
        }

        inside
    }

    fn mean(&self, values: &[f64]) -> f64 {
        if values.is_empty() {
            0.0
        } else {
            values.iter().sum::<f64>() / values.len() as f64
        }
    }

    fn std_dev(&self, values: &[f64]) -> f64 {
        if values.len() < 2 {
            return 0.0;
        }

        let mean = self.mean(values);
        let variance =
            values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (values.len() - 1) as f64;

        variance.sqrt()
    }
}

impl Default for ContourAnalysisExtractor {
    fn default() -> Self {
        Self::new()
    }
}

/// Contour structure containing contour properties
#[derive(Debug, Clone)]
pub struct Contour {
    /// points
    pub points: Vec<(f64, f64)>,
    /// area
    pub area: f64,
    /// perimeter
    pub perimeter: f64,
    /// convex_hull
    pub convex_hull: Vec<(f64, f64)>,
    /// solidity
    pub solidity: f64, // area / convex_hull_area
    /// extent
    pub extent: f64, // area / bounding_box_area
    /// convexity
    pub convexity: f64, // convex_hull_perimeter / perimeter
    /// aspect_ratio
    pub aspect_ratio: f64, // width / height of bounding box
    /// orientation
    pub orientation: f64, // angle of major axis
    /// bounding_box
    pub bounding_box: (f64, f64, f64, f64), // (min_x, min_y, width, height)
}

/// Morphological Features Extractor
///
/// Extract morphological features from binary images using mathematical morphology
/// operations like erosion, dilation, opening, closing, and morphological gradients.
///
/// # Parameters
///
/// * `kernel_size` - Size of the morphological kernel
/// * `kernel_shape` - Shape of the morphological kernel (rectangular, elliptical, cross)
/// * `operations` - Set of morphological operations to apply
/// * `multi_scale` - Whether to apply operations at multiple scales
///
/// # Features Extracted
///
/// * Erosion and dilation response statistics
/// * Opening and closing response statistics  
/// * Morphological gradient magnitude and direction
/// * Granulometry features (size distribution)
/// * Structural element fit measures
/// * Pattern spectrum analysis
///
/// # Examples
///
/// ```
/// use sklears_feature_extraction::image_advanced::{MorphologicalFeaturesExtractor, KernelShape};
/// use scirs2_core::ndarray::Array2;
///
/// let image = Array2::from_shape_vec((16, 16),
///     (0..256).map(|x| if x % 8 < 4 { 1.0 } else { 0.0 }).collect()).unwrap();
///
/// let morph_extractor = MorphologicalFeaturesExtractor::new()
///     .kernel_size(3)
///     .kernel_shape(KernelShape::Elliptical)
///     .multi_scale(true);
///
/// let features = morph_extractor.extract_features(&image.view()).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct MorphologicalFeaturesExtractor {
    kernel_size: usize,
    kernel_shape: KernelShape,
    operations: Vec<MorphOperation>,
    multi_scale: bool,
    binary_threshold: f64,
    scale_range: (usize, usize),
}

#[derive(Debug, Clone, Copy)]
pub enum KernelShape {
    /// Rectangular
    Rectangular,
    /// Elliptical
    Elliptical,
    /// Cross
    Cross,
    /// Diamond
    Diamond,
}

#[derive(Debug, Clone, Copy)]
pub enum MorphOperation {
    /// Erosion
    Erosion,
    /// Dilation
    Dilation,
    /// Opening
    Opening,
    /// Closing
    Closing,
    /// Gradient
    Gradient,
    /// TopHat
    TopHat,
    /// BlackHat
    BlackHat,
}

impl MorphologicalFeaturesExtractor {
    /// Create a new MorphologicalFeaturesExtractor
    pub fn new() -> Self {
        Self {
            kernel_size: 3,
            kernel_shape: KernelShape::Elliptical,
            operations: vec![
                MorphOperation::Erosion,
                MorphOperation::Dilation,
                MorphOperation::Opening,
                MorphOperation::Closing,
                MorphOperation::Gradient,
            ],
            multi_scale: true,
            binary_threshold: 0.5,
            scale_range: (3, 15),
        }
    }

    /// Set morphological kernel size
    pub fn kernel_size(mut self, size: usize) -> Self {
        self.kernel_size = size;
        self
    }

    /// Set morphological kernel shape
    pub fn kernel_shape(mut self, shape: KernelShape) -> Self {
        self.kernel_shape = shape;
        self
    }

    /// Set morphological operations to apply
    pub fn operations(mut self, operations: Vec<MorphOperation>) -> Self {
        self.operations = operations;
        self
    }

    /// Set whether to use multi-scale analysis
    pub fn multi_scale(mut self, enable: bool) -> Self {
        self.multi_scale = enable;
        self
    }

    /// Set binary threshold
    pub fn binary_threshold(mut self, threshold: f64) -> Self {
        self.binary_threshold = threshold;
        self
    }

    /// Set scale range for multi-scale analysis
    pub fn scale_range(mut self, range: (usize, usize)) -> Self {
        self.scale_range = range;
        self
    }

    /// Extract morphological features from image
    pub fn extract_features(&self, image: &ArrayView2<f64>) -> SklResult<Vec<f64>> {
        let binary_image = self.binarize_image(image);
        let mut features = Vec::new();

        if self.multi_scale {
            // Multi-scale analysis
            for size in (self.scale_range.0..=self.scale_range.1).step_by(2) {
                let scale_features = self.extract_features_at_scale(&binary_image, size)?;
                features.extend(scale_features);
            }

            // Granulometry analysis
            let granulometry_features = self.compute_granulometry(&binary_image)?;
            features.extend(granulometry_features);
        } else {
            // Single-scale analysis
            let single_scale_features =
                self.extract_features_at_scale(&binary_image, self.kernel_size)?;
            features.extend(single_scale_features);
        }

        Ok(features)
    }

    fn binarize_image(&self, image: &ArrayView2<f64>) -> Array2<bool> {
        let (height, width) = image.dim();
        let mut binary = Array2::default((height, width));

        for y in 0..height {
            for x in 0..width {
                binary[[y, x]] = image[[y, x]] > self.binary_threshold;
            }
        }

        binary
    }

    fn extract_features_at_scale(
        &self,
        binary_image: &Array2<bool>,
        kernel_size: usize,
    ) -> SklResult<Vec<f64>> {
        let kernel = self.create_kernel(kernel_size);
        let mut features = Vec::new();

        for &operation in &self.operations {
            let result = self.apply_morphological_operation(binary_image, &kernel, operation)?;
            let operation_features = self.compute_operation_statistics(&result);
            features.extend(operation_features);
        }

        Ok(features)
    }

    fn create_kernel(&self, size: usize) -> Array2<bool> {
        let mut kernel = Array2::default((size, size));
        let center = size / 2;

        match self.kernel_shape {
            KernelShape::Rectangular => {
                kernel.fill(true);
            }
            KernelShape::Elliptical => {
                let radius_sq = (center as f64).powi(2);
                for y in 0..size {
                    for x in 0..size {
                        let dx = x as f64 - center as f64;
                        let dy = y as f64 - center as f64;
                        kernel[[y, x]] = (dx * dx + dy * dy) <= radius_sq;
                    }
                }
            }
            KernelShape::Cross => {
                for i in 0..size {
                    kernel[[center, i]] = true;
                    kernel[[i, center]] = true;
                }
            }
            KernelShape::Diamond => {
                for y in 0..size {
                    for x in 0..size {
                        let dx = (x as i32 - center as i32).abs();
                        let dy = (y as i32 - center as i32).abs();
                        kernel[[y, x]] = dx + dy <= center as i32;
                    }
                }
            }
        }

        kernel
    }

    fn apply_morphological_operation(
        &self,
        image: &Array2<bool>,
        kernel: &Array2<bool>,
        operation: MorphOperation,
    ) -> SklResult<Array2<bool>> {
        match operation {
            MorphOperation::Erosion => self.erosion(image, kernel),
            MorphOperation::Dilation => self.dilation(image, kernel),
            MorphOperation::Opening => {
                let eroded = self.erosion(image, kernel)?;
                self.dilation(&eroded, kernel)
            }
            MorphOperation::Closing => {
                let dilated = self.dilation(image, kernel)?;
                self.erosion(&dilated, kernel)
            }
            MorphOperation::Gradient => {
                let dilated = self.dilation(image, kernel)?;
                let eroded = self.erosion(image, kernel)?;
                self.subtract(&dilated, &eroded)
            }
            MorphOperation::TopHat => {
                let opened = {
                    let eroded = self.erosion(image, kernel)?;
                    self.dilation(&eroded, kernel)?
                };
                self.subtract(image, &opened)
            }
            MorphOperation::BlackHat => {
                let closed = {
                    let dilated = self.dilation(image, kernel)?;
                    self.erosion(&dilated, kernel)?
                };
                self.subtract(&closed, image)
            }
        }
    }

    fn erosion(&self, image: &Array2<bool>, kernel: &Array2<bool>) -> SklResult<Array2<bool>> {
        let (img_height, img_width) = image.dim();
        let (ker_height, ker_width) = kernel.dim();
        let mut result = Array2::default((img_height, img_width));

        let ker_center_y = ker_height / 2;
        let ker_center_x = ker_width / 2;

        for y in 0..img_height {
            for x in 0..img_width {
                let mut min_val = true;

                for ky in 0..ker_height {
                    for kx in 0..ker_width {
                        if kernel[[ky, kx]] {
                            let img_y = y as i32 + ky as i32 - ker_center_y as i32;
                            let img_x = x as i32 + kx as i32 - ker_center_x as i32;

                            if img_y >= 0
                                && img_y < img_height as i32
                                && img_x >= 0
                                && img_x < img_width as i32
                            {
                                if !image[[img_y as usize, img_x as usize]] {
                                    min_val = false;
                                    break;
                                }
                            } else {
                                min_val = false;
                                break;
                            }
                        }
                    }
                    if !min_val {
                        break;
                    }
                }

                result[[y, x]] = min_val;
            }
        }

        Ok(result)
    }

    fn dilation(&self, image: &Array2<bool>, kernel: &Array2<bool>) -> SklResult<Array2<bool>> {
        let (img_height, img_width) = image.dim();
        let (ker_height, ker_width) = kernel.dim();
        let mut result = Array2::default((img_height, img_width));

        let ker_center_y = ker_height / 2;
        let ker_center_x = ker_width / 2;

        for y in 0..img_height {
            for x in 0..img_width {
                let mut max_val = false;

                for ky in 0..ker_height {
                    for kx in 0..ker_width {
                        if kernel[[ky, kx]] {
                            let img_y = y as i32 + ky as i32 - ker_center_y as i32;
                            let img_x = x as i32 + kx as i32 - ker_center_x as i32;

                            if img_y >= 0
                                && img_y < img_height as i32
                                && img_x >= 0
                                && img_x < img_width as i32
                                && image[[img_y as usize, img_x as usize]]
                            {
                                max_val = true;
                                break;
                            }
                        }
                    }
                    if max_val {
                        break;
                    }
                }

                result[[y, x]] = max_val;
            }
        }

        Ok(result)
    }

    fn subtract(&self, image1: &Array2<bool>, image2: &Array2<bool>) -> SklResult<Array2<bool>> {
        let (height, width) = image1.dim();
        let mut result = Array2::default((height, width));

        for y in 0..height {
            for x in 0..width {
                result[[y, x]] = image1[[y, x]] && !image2[[y, x]];
            }
        }

        Ok(result)
    }

    fn compute_operation_statistics(&self, image: &Array2<bool>) -> Vec<f64> {
        let mut features = Vec::new();

        // Count true pixels
        let true_count = image.iter().filter(|&&x| x).count() as f64;
        let total_pixels = (image.nrows() * image.ncols()) as f64;

        features.push(true_count); // Number of foreground pixels
        features.push(true_count / total_pixels); // Foreground density

        // Compute connected components statistics
        let components = self.count_connected_components(image);
        features.push(components as f64); // Number of connected components

        features
    }

    fn count_connected_components(&self, image: &Array2<bool>) -> usize {
        let (height, width) = image.dim();
        let mut visited = Array2::<bool>::default((height, width));
        let mut count = 0;

        for y in 0..height {
            for x in 0..width {
                if image[[y, x]] && !visited[[y, x]] {
                    self.flood_fill(&mut visited, image, x, y);
                    count += 1;
                }
            }
        }

        count
    }

    fn flood_fill(&self, visited: &mut Array2<bool>, image: &Array2<bool>, x: usize, y: usize) {
        let (height, width) = image.dim();
        let mut stack = vec![(x, y)];

        while let Some((cx, cy)) = stack.pop() {
            if visited[[cy, cx]] || !image[[cy, cx]] {
                continue;
            }

            visited[[cy, cx]] = true;

            // Check 4-connected neighbors
            for (dx, dy) in [(-1, 0), (1, 0), (0, -1), (0, 1)].iter() {
                let nx = cx as i32 + dx;
                let ny = cy as i32 + dy;

                if nx >= 0 && nx < width as i32 && ny >= 0 && ny < height as i32 {
                    let nx = nx as usize;
                    let ny = ny as usize;

                    if !visited[[ny, nx]] && image[[ny, nx]] {
                        stack.push((nx, ny));
                    }
                }
            }
        }
    }

    fn compute_granulometry(&self, image: &Array2<bool>) -> SklResult<Vec<f64>> {
        let mut granulometry = Vec::new();
        let original_area = image.iter().filter(|&&x| x).count() as f64;

        // Apply opening with increasing kernel sizes
        for size in (self.scale_range.0..=self.scale_range.1).step_by(2) {
            let kernel = self.create_kernel(size);
            let opened = {
                let eroded = self.erosion(image, &kernel)?;
                self.dilation(&eroded, &kernel)?
            };

            let remaining_area = opened.iter().filter(|&&x| x).count() as f64;
            let area_ratio = if original_area > 0.0 {
                remaining_area / original_area
            } else {
                0.0
            };

            granulometry.push(area_ratio);
        }

        // Compute granulometry statistics
        let mut features = Vec::new();
        if !granulometry.is_empty() {
            features.push(self.mean(&granulometry)); // Mean area retention
            features.push(self.std_dev(&granulometry)); // Std dev of area retention

            // Compute pattern spectrum (derivatives)
            let mut pattern_spectrum = Vec::new();
            for i in 1..granulometry.len() {
                pattern_spectrum.push(granulometry[i - 1] - granulometry[i]);
            }

            if !pattern_spectrum.is_empty() {
                features.push(self.mean(&pattern_spectrum)); // Mean pattern spectrum
                features.push(self.std_dev(&pattern_spectrum)); // Std dev pattern spectrum

                // Find dominant scale (max in pattern spectrum)
                let max_idx = pattern_spectrum
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                features.push((max_idx + 1) as f64); // Dominant scale index
            } else {
                features.extend(vec![0.0, 0.0, 0.0]);
            }
        } else {
            features.extend(vec![0.0, 0.0, 0.0, 0.0, 0.0]);
        }

        Ok(features)
    }

    fn mean(&self, values: &[f64]) -> f64 {
        if values.is_empty() {
            0.0
        } else {
            values.iter().sum::<f64>() / values.len() as f64
        }
    }

    fn std_dev(&self, values: &[f64]) -> f64 {
        if values.len() < 2 {
            return 0.0;
        }

        let mean = self.mean(values);
        let variance =
            values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (values.len() - 1) as f64;

        variance.sqrt()
    }
}

impl Default for MorphologicalFeaturesExtractor {
    fn default() -> Self {
        Self::new()
    }
}
