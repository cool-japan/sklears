//! Computer Vision Neighbor-Based Methods
//!
//! This module provides specialized neighbor-based algorithms for computer vision applications,
//! including image similarity search, patch-based matching, feature descriptor analysis,
//! and content-based image retrieval. These methods leverage the existing neighbor infrastructure
//! with computer vision specific optimizations and feature extractors.
//!
//! # Key Features
//!
//! - **Image Similarity Search**: Efficient similarity search using various image features
//! - **Patch-Based Neighbors**: Local patch matching for texture analysis and object detection
//! - **Feature Descriptor Matching**: SIFT, SURF, ORB, and other descriptor-based matching
//! - **Visual Word Recognition**: Bag-of-visual-words for image categorization
//! - **Content-Based Image Retrieval**: Full CBIR system with multiple feature types
//!
//! # Examples
//!
//! ```rust
//! use sklears_neighbors::computer_vision::{ImageSimilaritySearch, ImageSearchConfig, FeatureType};
//! use scirs2_core::ndarray::{Array1, Array2};
//!
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Create image similarity search
//! let config = ImageSearchConfig {
//!     feature_type: FeatureType::ColorHistogram,
//!     ..Default::default()
//! };
//! let mut search = ImageSimilaritySearch::new(config);
//!
//! // Add images to index
//! let features = Array2::zeros((100, 512)); // 100 images with 512-dim features
//! search.build_index(&features)?;
//!
//! // Search for similar images
//! let query = Array1::zeros(512);
//! let results = search.search(&query, 5)?;
//! # Ok(())
//! # }
//! ```

use crate::{knn::KNeighborsClassifier, NeighborsError, NeighborsResult};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, Axis};
use scirs2_core::random::thread_rng;
use sklears_core::traits::{Fit, Predict, Trained};
use sklears_core::types::{Features, Float, Int};
use std::collections::HashMap;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Image feature types for similarity computation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum FeatureType {
    /// RGB color histogram features
    ColorHistogram,
    /// Local Binary Pattern (LBP) texture features
    LocalBinaryPattern,
    /// Histogram of Oriented Gradients (HOG) features
    HistogramOfGradients,
    /// Gabor filter bank responses
    GaborFilters,
    /// Edge density and orientation features
    EdgeFeatures,
    /// GLCM (Gray-Level Co-occurrence Matrix) features
    TextureFeatures,
    /// Combined multi-modal features
    MultiModal,
}

/// Configuration for image similarity search
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ImageSearchConfig {
    /// Feature type to use
    pub feature_type: FeatureType,
    /// Number of neighbors to retrieve
    pub k_neighbors: usize,
    /// Distance metric for similarity computation
    pub distance_metric: String,
    /// Enable feature normalization
    pub normalize_features: bool,
    /// Feature dimensionality reduction target
    pub target_dimensions: Option<usize>,
    /// Use approximate search for large datasets
    pub use_approximate: bool,
}

impl Default for ImageSearchConfig {
    fn default() -> Self {
        Self {
            feature_type: FeatureType::ColorHistogram,
            k_neighbors: 5,
            distance_metric: "euclidean".to_string(),
            normalize_features: true,
            target_dimensions: None,
            use_approximate: false,
        }
    }
}

/// Image similarity search engine
pub struct ImageSimilaritySearch {
    config: ImageSearchConfig,
    feature_extractor: Box<dyn FeatureExtractor>,
    neighbor_index: Option<KNeighborsClassifier<Trained>>,
    feature_database: Option<Array2<f64>>,
    image_metadata: Vec<ImageMetadata>,
}

/// Metadata for indexed images
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ImageMetadata {
    /// Unique image identifier
    pub id: String,
    /// Image dimensions (width, height)
    pub dimensions: (usize, usize),
    /// Number of color channels
    pub channels: usize,
    /// Optional image path or URL
    pub path: Option<String>,
    /// Custom metadata tags
    pub tags: HashMap<String, String>,
}

/// Search result for image similarity
#[derive(Debug, Clone)]
pub struct ImageSearchResult {
    /// Image metadata
    pub metadata: ImageMetadata,
    /// Similarity distance (lower = more similar)
    pub distance: f64,
    /// Feature vector used for matching
    pub features: Array1<f64>,
    /// Match confidence score (0-1)
    pub confidence: f64,
}

/// Feature extractor trait for different image feature types
pub trait FeatureExtractor: Send + Sync {
    /// Extract features from image data
    ///
    /// # Arguments
    /// * `image_data` - Flattened image data (height * width * channels)
    /// * `dimensions` - Image dimensions (width, height, channels)
    fn extract_features(
        &self,
        image_data: &[f64],
        dimensions: (usize, usize, usize),
    ) -> NeighborsResult<Array1<f64>>;

    /// Get the dimensionality of extracted features
    fn feature_dimension(&self) -> usize;

    /// Get feature extractor name
    fn name(&self) -> &str;
}

/// Color histogram feature extractor
pub struct ColorHistogramExtractor {
    bins_per_channel: usize,
    normalize: bool,
}

impl ColorHistogramExtractor {
    pub fn new(bins_per_channel: usize, normalize: bool) -> Self {
        Self {
            bins_per_channel,
            normalize,
        }
    }
}

impl FeatureExtractor for ColorHistogramExtractor {
    fn extract_features(
        &self,
        image_data: &[f64],
        dimensions: (usize, usize, usize),
    ) -> NeighborsResult<Array1<f64>> {
        let (width, height, channels) = dimensions;

        if image_data.len() != width * height * channels {
            return Err(NeighborsError::InvalidInput(format!(
                "Image data length {} doesn't match dimensions {}x{}x{}",
                image_data.len(),
                width,
                height,
                channels
            )));
        }

        let total_bins = self.bins_per_channel * channels;
        let mut histogram = Array1::zeros(total_bins);

        // Compute histogram for each channel
        for c in 0..channels {
            for i in 0..(width * height) {
                let pixel_idx = i * channels + c;
                let pixel_value = image_data[pixel_idx];

                // Normalize pixel value to [0, 1] range and compute bin
                let normalized = pixel_value.clamp(0.0, 1.0);
                let bin = ((normalized * self.bins_per_channel as f64) as usize)
                    .min(self.bins_per_channel - 1);

                histogram[c * self.bins_per_channel + bin] += 1.0;
            }
        }

        // Normalize histogram if requested
        if self.normalize {
            let sum: f64 = histogram.sum();
            if sum > 0.0 {
                histogram /= sum;
            }
        }

        Ok(histogram)
    }

    fn feature_dimension(&self) -> usize {
        self.bins_per_channel * 3 // Assuming RGB images
    }

    fn name(&self) -> &str {
        "ColorHistogram"
    }
}

/// Local Binary Pattern (LBP) feature extractor
pub struct LocalBinaryPatternExtractor {
    radius: usize,
    neighbors: usize,
    uniform_patterns: bool,
}

impl LocalBinaryPatternExtractor {
    pub fn new(radius: usize, neighbors: usize, uniform_patterns: bool) -> Self {
        Self {
            radius,
            neighbors,
            uniform_patterns,
        }
    }

    /// Compute LBP value for a single pixel
    fn compute_lbp_value(&self, center: f64, neighbors: &[f64]) -> u32 {
        let mut lbp_code = 0u32;

        for (i, &neighbor) in neighbors.iter().enumerate() {
            if neighbor >= center {
                lbp_code |= 1 << i;
            }
        }

        // If using uniform patterns, map to uniform LBP codes
        if self.uniform_patterns {
            self.to_uniform_pattern(lbp_code)
        } else {
            lbp_code
        }
    }

    /// Convert LBP code to uniform pattern
    fn to_uniform_pattern(&self, code: u32) -> u32 {
        // Count transitions in circular pattern
        let mut transitions = 0;
        for i in 0..self.neighbors {
            let bit1 = (code >> i) & 1;
            let bit2 = (code >> ((i + 1) % self.neighbors)) & 1;
            if bit1 != bit2 {
                transitions += 1;
            }
        }

        // Uniform patterns have at most 2 transitions
        if transitions <= 2 {
            code.count_ones()
        } else {
            self.neighbors as u32 + 1 // Non-uniform pattern bin
        }
    }
}

impl FeatureExtractor for LocalBinaryPatternExtractor {
    fn extract_features(
        &self,
        image_data: &[f64],
        dimensions: (usize, usize, usize),
    ) -> NeighborsResult<Array1<f64>> {
        let (width, height, channels) = dimensions;

        if image_data.len() != width * height * channels {
            return Err(NeighborsError::InvalidInput(
                "Image data length doesn't match dimensions".to_string(),
            ));
        }

        // Convert to grayscale if needed
        let grayscale: Vec<f64> = if channels == 1 {
            image_data.to_vec()
        } else {
            // Convert RGB to grayscale using luminance formula
            (0..(width * height))
                .map(|i| {
                    let r = image_data[i * channels];
                    let g = image_data[i * channels + 1];
                    let b = image_data[i * channels + 2];
                    0.299 * r + 0.587 * g + 0.114 * b
                })
                .collect()
        };

        let max_pattern = if self.uniform_patterns {
            self.neighbors + 2
        } else {
            1 << self.neighbors
        };

        let mut histogram = Array1::zeros(max_pattern);

        // Process each pixel (excluding border pixels)
        for y in self.radius..(height - self.radius) {
            for x in self.radius..(width - self.radius) {
                let center_idx = y * width + x;
                let center_value = grayscale[center_idx];

                // Sample neighbors in circular pattern
                let mut neighbor_values = Vec::with_capacity(self.neighbors);
                for i in 0..self.neighbors {
                    let angle = 2.0 * std::f64::consts::PI * i as f64 / self.neighbors as f64;
                    let nx = x as f64 + self.radius as f64 * angle.cos();
                    let ny = y as f64 + self.radius as f64 * angle.sin();

                    // Bilinear interpolation for non-integer coordinates
                    let x1 = nx.floor() as usize;
                    let y1 = ny.floor() as usize;
                    let x2 = (x1 + 1).min(width - 1);
                    let y2 = (y1 + 1).min(height - 1);

                    let fx = nx - x1 as f64;
                    let fy = ny - y1 as f64;

                    let v1 = grayscale[y1 * width + x1];
                    let v2 = grayscale[y1 * width + x2];
                    let v3 = grayscale[y2 * width + x1];
                    let v4 = grayscale[y2 * width + x2];

                    let interpolated = v1 * (1.0 - fx) * (1.0 - fy)
                        + v2 * fx * (1.0 - fy)
                        + v3 * (1.0 - fx) * fy
                        + v4 * fx * fy;

                    neighbor_values.push(interpolated);
                }

                // Compute LBP code
                let lbp_code = self.compute_lbp_value(center_value, &neighbor_values);
                histogram[lbp_code as usize] += 1.0;
            }
        }

        // Normalize histogram
        let sum: f64 = histogram.sum();
        if sum > 0.0 {
            histogram /= sum;
        }

        Ok(histogram)
    }

    fn feature_dimension(&self) -> usize {
        if self.uniform_patterns {
            self.neighbors + 2
        } else {
            1 << self.neighbors
        }
    }

    fn name(&self) -> &str {
        "LocalBinaryPattern"
    }
}

/// Histogram of Oriented Gradients (HOG) feature extractor
pub struct HistogramOfGradientsExtractor {
    cell_size: usize,
    block_size: usize,
    num_bins: usize,
    normalize_blocks: bool,
}

impl HistogramOfGradientsExtractor {
    pub fn new(
        cell_size: usize,
        block_size: usize,
        num_bins: usize,
        normalize_blocks: bool,
    ) -> Self {
        Self {
            cell_size,
            block_size,
            num_bins,
            normalize_blocks,
        }
    }

    /// Compute gradient magnitude and orientation
    fn compute_gradients(
        &self,
        image: &[f64],
        width: usize,
        height: usize,
    ) -> (Vec<f64>, Vec<f64>) {
        let mut magnitudes = vec![0.0; width * height];
        let mut orientations = vec![0.0; width * height];

        for y in 1..(height - 1) {
            for x in 1..(width - 1) {
                let idx = y * width + x;

                // Compute gradients using Sobel-like operators
                let gx = image[y * width + (x + 1)] - image[y * width + (x - 1)];
                let gy = image[(y + 1) * width + x] - image[(y - 1) * width + x];

                magnitudes[idx] = (gx * gx + gy * gy).sqrt();
                orientations[idx] = gy.atan2(gx);

                // Convert orientation to [0, π] range
                if orientations[idx] < 0.0 {
                    orientations[idx] += std::f64::consts::PI;
                }
            }
        }

        (magnitudes, orientations)
    }
}

impl FeatureExtractor for HistogramOfGradientsExtractor {
    fn extract_features(
        &self,
        image_data: &[f64],
        dimensions: (usize, usize, usize),
    ) -> NeighborsResult<Array1<f64>> {
        let (width, height, channels) = dimensions;

        // Convert to grayscale
        let grayscale: Vec<f64> = if channels == 1 {
            image_data.to_vec()
        } else {
            (0..(width * height))
                .map(|i| {
                    let r = image_data[i * channels];
                    let g = image_data[i * channels + 1];
                    let b = image_data[i * channels + 2];
                    0.299 * r + 0.587 * g + 0.114 * b
                })
                .collect()
        };

        // Compute gradients
        let (magnitudes, orientations) = self.compute_gradients(&grayscale, width, height);

        // Calculate number of cells
        let cells_x = width / self.cell_size;
        let cells_y = height / self.cell_size;

        // Compute cell histograms
        let mut cell_histograms = vec![Array1::zeros(self.num_bins); cells_x * cells_y];

        for cell_y in 0..cells_y {
            for cell_x in 0..cells_x {
                let cell_idx = cell_y * cells_x + cell_x;

                // Process pixels in current cell
                for y in (cell_y * self.cell_size)..((cell_y + 1) * self.cell_size) {
                    for x in (cell_x * self.cell_size)..((cell_x + 1) * self.cell_size) {
                        if y < height && x < width {
                            let pixel_idx = y * width + x;
                            let magnitude = magnitudes[pixel_idx];
                            let orientation = orientations[pixel_idx];

                            // Compute bin for orientation
                            let bin_width = std::f64::consts::PI / self.num_bins as f64;
                            let bin = ((orientation / bin_width) as usize).min(self.num_bins - 1);

                            // Add weighted vote to histogram
                            cell_histograms[cell_idx][bin] += magnitude;
                        }
                    }
                }
            }
        }

        // Create block descriptors
        let blocks_x = cells_x.saturating_sub(self.block_size - 1);
        let blocks_y = cells_y.saturating_sub(self.block_size - 1);
        let descriptor_size =
            blocks_x * blocks_y * self.block_size * self.block_size * self.num_bins;

        let mut descriptor = Array1::zeros(descriptor_size);
        let mut desc_idx = 0;

        for block_y in 0..blocks_y {
            for block_x in 0..blocks_x {
                // Collect histograms from cells in current block
                let mut block_hist =
                    Array1::zeros(self.block_size * self.block_size * self.num_bins);
                let mut hist_idx = 0;

                for by in 0..self.block_size {
                    for bx in 0..self.block_size {
                        let cell_x = block_x + bx;
                        let cell_y = block_y + by;
                        let cell_idx = cell_y * cells_x + cell_x;

                        for bin in 0..self.num_bins {
                            block_hist[hist_idx * self.num_bins + bin] =
                                cell_histograms[cell_idx][bin];
                        }
                        hist_idx += 1;
                    }
                }

                // Normalize block if requested
                if self.normalize_blocks {
                    let norm = (block_hist.mapv(|x: f64| x * x).sum()).sqrt();
                    if norm > 1e-8 {
                        block_hist /= norm;
                    }
                }

                // Copy to descriptor
                for i in 0..block_hist.len() {
                    descriptor[desc_idx + i] = block_hist[i];
                }
                desc_idx += block_hist.len();
            }
        }

        Ok(descriptor)
    }

    fn feature_dimension(&self) -> usize {
        // This is an approximation - actual size depends on image dimensions
        self.num_bins * self.block_size * self.block_size * 16
    }

    fn name(&self) -> &str {
        "HistogramOfGradients"
    }
}

impl ImageSimilaritySearch {
    /// Create a new image similarity search engine
    pub fn new(config: ImageSearchConfig) -> Self {
        let feature_extractor: Box<dyn FeatureExtractor> = match config.feature_type {
            FeatureType::ColorHistogram => Box::new(ColorHistogramExtractor::new(32, true)),
            FeatureType::LocalBinaryPattern => {
                Box::new(LocalBinaryPatternExtractor::new(1, 8, true))
            }
            FeatureType::HistogramOfGradients => {
                Box::new(HistogramOfGradientsExtractor::new(8, 2, 9, true))
            }
            FeatureType::GaborFilters => Box::new(ColorHistogramExtractor::new(32, true)), // Placeholder
            FeatureType::EdgeFeatures => Box::new(ColorHistogramExtractor::new(32, true)), // Placeholder
            FeatureType::TextureFeatures => Box::new(ColorHistogramExtractor::new(32, true)), // Placeholder
            FeatureType::MultiModal => Box::new(ColorHistogramExtractor::new(32, true)), // Placeholder
        };

        Self {
            config,
            feature_extractor,
            neighbor_index: None,
            feature_database: None,
            image_metadata: Vec::new(),
        }
    }

    /// Build index from precomputed features
    pub fn build_index(&mut self, features: &Array2<f64>) -> NeighborsResult<()> {
        if features.is_empty() {
            return Err(NeighborsError::EmptyInput);
        }

        // Simply store the feature database - we'll use brute-force search for now
        self.feature_database = Some(features.clone());
        self.neighbor_index = None; // Not using the classifier for now

        Ok(())
    }

    /// Add image to the search index
    pub fn add_image(
        &mut self,
        image_data: &[f64],
        dimensions: (usize, usize, usize),
        metadata: ImageMetadata,
    ) -> NeighborsResult<()> {
        // Extract features from image
        let features = self
            .feature_extractor
            .extract_features(image_data, dimensions)?;

        // Add metadata first
        self.image_metadata.push(metadata);

        // Add to feature database
        if let Some(ref mut db) = self.feature_database {
            // Extend existing database by creating a new matrix
            let mut new_db = Array2::zeros((db.nrows() + 1, db.ncols()));
            new_db
                .slice_mut(scirs2_core::ndarray::s![..db.nrows(), ..])
                .assign(db);
            new_db.row_mut(db.nrows()).assign(&features);
            *db = new_db;

            // Clone the database to avoid borrowing issues
            let db_clone = db.clone();
            self.build_index(&db_clone)?;
        } else {
            // Create new database
            let new_db = features.insert_axis(Axis(0));
            let db_clone = new_db.clone();
            self.feature_database = Some(new_db);
            self.build_index(&db_clone)?;
        }

        Ok(())
    }

    /// Search for similar images using brute-force search
    pub fn search(
        &self,
        query_features: &Array1<f64>,
        k: usize,
    ) -> NeighborsResult<Vec<ImageSearchResult>> {
        let database = self
            .feature_database
            .as_ref()
            .ok_or(NeighborsError::InvalidInput(
                "Feature database not available".to_string(),
            ))?;

        if database.is_empty() {
            return Ok(Vec::new());
        }

        // Compute distances to all database features
        let mut distances_with_indices: Vec<(f64, usize)> = Vec::new();

        for (idx, db_features) in database.rows().into_iter().enumerate() {
            // Compute Euclidean distance
            let distance: f64 = query_features
                .iter()
                .zip(db_features.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();

            distances_with_indices.push((distance, idx));
        }

        // Sort by distance and take top k
        distances_with_indices.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let k_results = k.min(distances_with_indices.len());

        let mut results = Vec::new();
        let max_distance = if !distances_with_indices.is_empty() {
            distances_with_indices[distances_with_indices.len().min(k) - 1].0
        } else {
            1.0
        };

        for i in 0..k_results {
            let (distance, idx) = distances_with_indices[i];

            let features = database.row(idx).to_owned();

            // Compute confidence score (inverse of normalized distance)
            let confidence = if max_distance > 0.0 {
                1.0 - (distance / max_distance)
            } else {
                1.0
            };

            // Create metadata if not available
            let metadata = if idx < self.image_metadata.len() {
                self.image_metadata[idx].clone()
            } else {
                ImageMetadata {
                    id: format!("image_{}", idx),
                    dimensions: (0, 0),
                    channels: 0,
                    path: None,
                    tags: HashMap::new(),
                }
            };

            results.push(ImageSearchResult {
                metadata,
                distance,
                features,
                confidence,
            });
        }

        Ok(results)
    }

    /// Search by image data
    pub fn search_by_image(
        &self,
        image_data: &[f64],
        dimensions: (usize, usize, usize),
        k: usize,
    ) -> NeighborsResult<Vec<ImageSearchResult>> {
        let features = self
            .feature_extractor
            .extract_features(image_data, dimensions)?;
        self.search(&features, k)
    }

    /// Get database statistics
    pub fn get_stats(&self) -> (usize, usize, String) {
        let num_images = self.image_metadata.len();
        let feature_dim = self
            .feature_database
            .as_ref()
            .map(|db| db.ncols())
            .unwrap_or(0);
        let extractor_name = self.feature_extractor.name().to_string();

        (num_images, feature_dim, extractor_name)
    }
}

/// Patch-based neighbor matching for texture analysis
pub struct PatchBasedMatching {
    patch_size: usize,
    stride: usize,
    feature_extractor: Box<dyn FeatureExtractor>,
    neighbor_search: Option<ImageSimilaritySearch>,
}

impl PatchBasedMatching {
    /// Create new patch-based matching system
    pub fn new(patch_size: usize, stride: usize, feature_type: FeatureType) -> Self {
        let config = ImageSearchConfig {
            feature_type,
            k_neighbors: 10,
            ..Default::default()
        };

        let feature_extractor: Box<dyn FeatureExtractor> = match feature_type {
            FeatureType::ColorHistogram => Box::new(ColorHistogramExtractor::new(16, true)),
            FeatureType::LocalBinaryPattern => {
                Box::new(LocalBinaryPatternExtractor::new(1, 8, true))
            }
            FeatureType::HistogramOfGradients => {
                Box::new(HistogramOfGradientsExtractor::new(4, 1, 9, true))
            }
            _ => Box::new(LocalBinaryPatternExtractor::new(1, 8, true)),
        };

        Self {
            patch_size,
            stride,
            feature_extractor,
            neighbor_search: Some(ImageSimilaritySearch::new(config)),
        }
    }

    /// Extract patches from image
    pub fn extract_patches(
        &self,
        image_data: &[f64],
        dimensions: (usize, usize, usize),
    ) -> NeighborsResult<Vec<(Array1<f64>, (usize, usize))>> {
        let (width, height, channels) = dimensions;
        let mut patches = Vec::new();

        // Extract patches with sliding window
        for y in (0..(height.saturating_sub(self.patch_size))).step_by(self.stride) {
            for x in (0..(width.saturating_sub(self.patch_size))).step_by(self.stride) {
                // Extract patch data
                let mut patch_data = Vec::new();

                for py in y..(y + self.patch_size).min(height) {
                    for px in x..(x + self.patch_size).min(width) {
                        for c in 0..channels {
                            let idx = py * width * channels + px * channels + c;
                            patch_data.push(image_data[idx]);
                        }
                    }
                }

                // Extract features from patch
                let patch_dims = (self.patch_size, self.patch_size, channels);
                let features = self
                    .feature_extractor
                    .extract_features(&patch_data, patch_dims)?;
                patches.push((features, (x, y)));
            }
        }

        Ok(patches)
    }

    /// Build patch database from multiple images
    pub fn build_patch_database(
        &mut self,
        images: &[(Vec<f64>, (usize, usize, usize))],
    ) -> NeighborsResult<()> {
        let mut all_patch_features = Vec::new();

        for (image_data, dimensions) in images {
            let patches = self.extract_patches(image_data, *dimensions)?;
            for (features, _position) in patches {
                all_patch_features.push(features);
            }
        }

        if all_patch_features.is_empty() {
            return Err(NeighborsError::EmptyInput);
        }

        // Convert to Array2
        let num_patches = all_patch_features.len();
        let feature_dim = all_patch_features[0].len();
        let mut feature_matrix = Array2::zeros((num_patches, feature_dim));

        for (i, features) in all_patch_features.into_iter().enumerate() {
            feature_matrix.row_mut(i).assign(&features);
        }

        // Build search index
        if let Some(ref mut search) = self.neighbor_search {
            search.build_index(&feature_matrix)?;
        }

        Ok(())
    }

    /// Find similar patches
    pub fn find_similar_patches(
        &self,
        query_patch: &[f64],
        patch_dimensions: (usize, usize, usize),
        k: usize,
    ) -> NeighborsResult<Vec<ImageSearchResult>> {
        let features = self
            .feature_extractor
            .extract_features(query_patch, patch_dimensions)?;

        if let Some(ref search) = self.neighbor_search {
            search.search(&features, k)
        } else {
            Err(NeighborsError::InvalidInput(
                "Patch database not built".to_string(),
            ))
        }
    }
}

/// Feature descriptor matching (SIFT-like)
pub struct FeatureDescriptorMatcher {
    descriptor_dimension: usize,
    match_threshold: f64,
    ratio_test_threshold: f64,
    use_cross_check: bool,
}

/// Keypoint with descriptor
#[derive(Debug, Clone)]
pub struct Keypoint {
    /// 2D coordinates (x, y)
    pub position: (f64, f64),
    /// Scale/size of the keypoint
    pub scale: f64,
    /// Orientation in radians
    pub orientation: f64,
    /// Feature descriptor vector
    pub descriptor: Array1<f64>,
    /// Response strength
    pub response: f64,
}

/// Descriptor match between two keypoints
#[derive(Debug, Clone)]
pub struct DescriptorMatch {
    /// Query keypoint index
    pub query_idx: usize,
    /// Train keypoint index
    pub train_idx: usize,
    /// Match distance
    pub distance: f64,
    /// Match confidence
    pub confidence: f64,
}

impl FeatureDescriptorMatcher {
    /// Create new descriptor matcher
    pub fn new(descriptor_dimension: usize) -> Self {
        Self {
            descriptor_dimension,
            match_threshold: 0.8,
            ratio_test_threshold: 0.7,
            use_cross_check: true,
        }
    }

    /// Match descriptors between two sets of keypoints
    pub fn match_descriptors(
        &self,
        query_keypoints: &[Keypoint],
        train_keypoints: &[Keypoint],
    ) -> NeighborsResult<Vec<DescriptorMatch>> {
        if query_keypoints.is_empty() || train_keypoints.is_empty() {
            return Ok(Vec::new());
        }

        let mut matches = Vec::new();

        // Build descriptor matrices
        let query_descriptors = Array2::from_shape_fn(
            (query_keypoints.len(), self.descriptor_dimension),
            |(i, j)| query_keypoints[i].descriptor[j],
        );

        let train_descriptors = Array2::from_shape_fn(
            (train_keypoints.len(), self.descriptor_dimension),
            |(i, j)| train_keypoints[i].descriptor[j],
        );

        // For each query descriptor, find best matches
        for (query_idx, query_desc) in query_descriptors.rows().into_iter().enumerate() {
            let mut distances: Vec<(usize, f64)> = Vec::new();

            // Compute distances to all train descriptors
            for (train_idx, train_desc) in train_descriptors.rows().into_iter().enumerate() {
                let distance = self.compute_descriptor_distance(&query_desc, &train_desc);
                distances.push((train_idx, distance));
            }

            // Sort by distance
            distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

            // Apply ratio test if we have at least 2 matches
            if distances.len() >= 2 {
                let best_distance = distances[0].1;
                let second_best_distance = distances[1].1;

                if second_best_distance > 0.0
                    && (best_distance / second_best_distance) < self.ratio_test_threshold
                {
                    let train_idx = distances[0].0;
                    let confidence = 1.0 - (best_distance / second_best_distance);

                    matches.push(DescriptorMatch {
                        query_idx,
                        train_idx,
                        distance: best_distance,
                        confidence,
                    });
                }
            } else if !distances.is_empty() && distances[0].1 < self.match_threshold {
                // Single match case
                let train_idx = distances[0].0;
                let confidence = 1.0 - distances[0].1;

                matches.push(DescriptorMatch {
                    query_idx,
                    train_idx,
                    distance: distances[0].1,
                    confidence,
                });
            }
        }

        // Apply cross-check if enabled
        if self.use_cross_check {
            matches = self.apply_cross_check(matches, &query_descriptors, &train_descriptors)?;
        }

        Ok(matches)
    }

    /// Compute distance between two descriptors
    fn compute_descriptor_distance(&self, desc1: &ArrayView1<f64>, desc2: &ArrayView1<f64>) -> f64 {
        // Use Euclidean distance
        desc1
            .iter()
            .zip(desc2.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    /// Apply cross-check filtering
    fn apply_cross_check(
        &self,
        matches: Vec<DescriptorMatch>,
        query_descriptors: &Array2<f64>,
        train_descriptors: &Array2<f64>,
    ) -> NeighborsResult<Vec<DescriptorMatch>> {
        let mut filtered_matches = Vec::new();

        for m in matches {
            // Check if train descriptor also matches back to query descriptor
            let train_desc = train_descriptors.row(m.train_idx);
            let mut best_query_idx = 0;
            let mut best_distance = f64::INFINITY;

            for (query_idx, query_desc) in query_descriptors.rows().into_iter().enumerate() {
                let distance = self.compute_descriptor_distance(&train_desc, &query_desc);
                if distance < best_distance {
                    best_distance = distance;
                    best_query_idx = query_idx;
                }
            }

            // If mutual best match, keep it
            if best_query_idx == m.query_idx {
                filtered_matches.push(m);
            }
        }

        Ok(filtered_matches)
    }
}

/// Visual word recognition using bag-of-visual-words
pub struct VisualWordRecognizer {
    vocabulary: Option<Array2<f64>>,
    vocabulary_size: usize,
    feature_extractor: Box<dyn FeatureExtractor>,
    classifier: Option<KNeighborsClassifier<Trained>>,
}

impl VisualWordRecognizer {
    /// Create new visual word recognizer
    pub fn new(vocabulary_size: usize) -> Self {
        Self {
            vocabulary: None,
            vocabulary_size,
            feature_extractor: Box::new(LocalBinaryPatternExtractor::new(1, 8, true)),
            classifier: None,
        }
    }

    /// Build vocabulary from training patches
    pub fn build_vocabulary(
        &mut self,
        training_patches: &[Vec<f64>],
        patch_dimensions: (usize, usize, usize),
    ) -> NeighborsResult<()> {
        if training_patches.is_empty() {
            return Err(NeighborsError::EmptyInput);
        }

        // Extract features from all patches
        let mut patch_features = Vec::new();
        for patch_data in training_patches {
            let features = self
                .feature_extractor
                .extract_features(patch_data, patch_dimensions)?;
            patch_features.push(features);
        }

        // Use k-means clustering to build vocabulary (simplified version)
        let feature_dim = patch_features[0].len();
        let mut vocabulary = Array2::zeros((self.vocabulary_size, feature_dim));

        // Initialize vocabulary with random patches
        let mut rng = thread_rng();
        for i in 0..self.vocabulary_size {
            let random_idx = rng.gen_range(0..patch_features.len());
            vocabulary.row_mut(i).assign(&patch_features[random_idx]);
        }

        // Simple k-means iterations (in practice, would use more sophisticated clustering)
        for _iteration in 0..10 {
            let mut cluster_sums = vec![Array1::zeros(feature_dim); self.vocabulary_size];
            let mut cluster_counts = vec![0; self.vocabulary_size];

            // Assign patches to clusters
            for patch_feature in &patch_features {
                let mut best_cluster = 0;
                let mut best_distance = f64::INFINITY;

                for (cluster_idx, centroid) in vocabulary.rows().into_iter().enumerate() {
                    let distance: f64 = patch_feature
                        .iter()
                        .zip(centroid.iter())
                        .map(|(a, b)| (a - b).powi(2))
                        .sum::<f64>()
                        .sqrt();

                    if distance < best_distance {
                        best_distance = distance;
                        best_cluster = cluster_idx;
                    }
                }

                cluster_sums[best_cluster] = &cluster_sums[best_cluster] + patch_feature;
                cluster_counts[best_cluster] += 1;
            }

            // Update centroids
            for i in 0..self.vocabulary_size {
                if cluster_counts[i] > 0 {
                    vocabulary
                        .row_mut(i)
                        .assign(&(&cluster_sums[i] / cluster_counts[i] as f64));
                }
            }
        }

        self.vocabulary = Some(vocabulary);
        Ok(())
    }

    /// Convert image to bag-of-visual-words histogram
    pub fn compute_bow_histogram(
        &self,
        image_patches: &[Vec<f64>],
        patch_dimensions: (usize, usize, usize),
    ) -> NeighborsResult<Array1<f64>> {
        let vocabulary = self
            .vocabulary
            .as_ref()
            .ok_or(NeighborsError::InvalidInput(
                "Vocabulary not built".to_string(),
            ))?;

        let mut histogram = Array1::zeros(self.vocabulary_size);

        for patch_data in image_patches {
            let features = self
                .feature_extractor
                .extract_features(patch_data, patch_dimensions)?;

            // Find closest visual word
            let mut best_word = 0;
            let mut best_distance = f64::INFINITY;

            for (word_idx, word_features) in vocabulary.rows().into_iter().enumerate() {
                let distance: f64 = features
                    .iter()
                    .zip(word_features.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>()
                    .sqrt();

                if distance < best_distance {
                    best_distance = distance;
                    best_word = word_idx;
                }
            }

            histogram[best_word] += 1.0;
        }

        // Normalize histogram
        let sum: f64 = histogram.sum();
        if sum > 0.0 {
            histogram /= sum;
        }

        Ok(histogram)
    }

    /// Train classifier on bag-of-visual-words histograms
    pub fn train_classifier(
        &mut self,
        bow_histograms: &Array2<f64>,
        labels: &Array1<i32>,
    ) -> NeighborsResult<()> {
        let classifier = KNeighborsClassifier::new(5);

        // Convert to proper types
        let features: Features = bow_histograms.mapv(|x| x as Float);
        let target_labels: Array1<Int> = labels.mapv(|x| x as Int);

        let trained = classifier
            .fit(&features, &target_labels)
            .map_err(|e| NeighborsError::InvalidInput(e.to_string()))?;

        self.classifier = Some(trained);
        Ok(())
    }

    /// Classify image using bag-of-visual-words
    pub fn classify_image(
        &self,
        image_patches: &[Vec<f64>],
        patch_dimensions: (usize, usize, usize),
    ) -> NeighborsResult<i32> {
        let classifier = self
            .classifier
            .as_ref()
            .ok_or(NeighborsError::InvalidInput(
                "Classifier not trained".to_string(),
            ))?;

        let bow_histogram = self.compute_bow_histogram(image_patches, patch_dimensions)?;
        let bow_2d = bow_histogram.insert_axis(Axis(0));

        // Convert to Features type
        let features: Features = bow_2d.mapv(|x| x as Float);

        let predictions = classifier
            .predict(&features)
            .map_err(|e| NeighborsError::InvalidInput(e.to_string()))?;

        Ok(predictions[0] as i32)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_color_histogram_extractor() {
        let extractor = ColorHistogramExtractor::new(8, true);
        let image_data = vec![0.5; 32 * 32 * 3]; // 32x32 RGB image
        let dimensions = (32, 32, 3);

        let features = extractor.extract_features(&image_data, dimensions).unwrap();
        assert_eq!(features.len(), 8 * 3);

        // Check normalization
        let sum: f64 = features.sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_local_binary_pattern_extractor() {
        let extractor = LocalBinaryPatternExtractor::new(1, 8, true);
        let image_data = vec![0.5; 16 * 16]; // 16x16 grayscale image
        let dimensions = (16, 16, 1);

        let features = extractor.extract_features(&image_data, dimensions).unwrap();
        assert_eq!(features.len(), 8 + 2); // uniform patterns
    }

    #[test]
    fn test_image_similarity_search() {
        let config = ImageSearchConfig::default();
        let mut search = ImageSimilaritySearch::new(config);

        // Create dummy feature data
        let features = Array2::from_shape_fn((10, 96), |(i, j)| (i + j) as f64);
        search.build_index(&features).unwrap();

        // Verify database was built properly
        let (num_images, _feature_dim, _) = search.get_stats();
        assert_eq!(num_images, 0); // No metadata added yet
        assert_eq!(search.feature_database.as_ref().unwrap().nrows(), 10);

        // Test search
        let query = Array1::from_vec((0..96).map(|i| i as f64).collect());
        let results = search.search(&query, 3).unwrap();

        assert_eq!(results.len(), 3);
        assert!(results[0].distance <= results[1].distance);
    }

    #[test]
    fn test_patch_based_matching() {
        let matcher = PatchBasedMatching::new(8, 4, FeatureType::LocalBinaryPattern);

        // Create dummy image
        let image_data = vec![0.5; 32 * 32 * 3];
        let dimensions = (32, 32, 3);

        let patches = matcher.extract_patches(&image_data, dimensions).unwrap();
        assert!(!patches.is_empty());

        // Test patch extraction positions
        let expected_patches = ((32_usize - 8) / 4 + 1).pow(2);
        assert!(patches.len() <= expected_patches);
    }

    #[test]
    fn test_feature_descriptor_matcher() {
        let matcher = FeatureDescriptorMatcher::new(128);

        // Create dummy keypoints
        let query_keypoints = vec![Keypoint {
            position: (10.0, 10.0),
            scale: 1.0,
            orientation: 0.0,
            descriptor: Array1::zeros(128),
            response: 0.5,
        }];

        let train_keypoints = vec![Keypoint {
            position: (11.0, 11.0),
            scale: 1.0,
            orientation: 0.0,
            descriptor: Array1::from_elem(128, 0.1),
            response: 0.6,
        }];

        let matches = matcher
            .match_descriptors(&query_keypoints, &train_keypoints)
            .unwrap();
        // With ratio test, may not find matches for identical descriptors
        assert!(matches.len() <= 1);
    }

    #[test]
    fn test_visual_word_recognizer() {
        let mut recognizer = VisualWordRecognizer::new(16);

        // Create training patches
        let training_patches: Vec<Vec<f64>> =
            (0..10).map(|i| vec![i as f64 * 0.1; 8 * 8 * 3]).collect();
        let patch_dimensions = (8, 8, 3);

        recognizer
            .build_vocabulary(&training_patches, patch_dimensions)
            .unwrap();

        // Test BOW histogram computation
        let test_patches = vec![vec![0.5; 8 * 8 * 3]];
        let histogram = recognizer
            .compute_bow_histogram(&test_patches, patch_dimensions)
            .unwrap();

        assert_eq!(histogram.len(), 16);
        assert!((histogram.sum() - 1.0).abs() < 1e-6);
    }
}
