//! Image feature extraction module
//!
//! This module provides comprehensive image feature extraction capabilities for computer vision
//! and machine learning applications. It includes implementations of various feature extraction
//! algorithms, from traditional methods like SIFT and SURF to advanced techniques like fractal
//! analysis and Zernike moments.
//!
//! # Architecture
//!
//! The image feature extraction module is organized into focused sub-modules:
//!
//! - **simd_accelerated**: SIMD-optimized operations for high-performance computing
//! - **patch_extraction**: 2D image patch extraction and reconstruction
//! - **sift_features**: Scale-Invariant Feature Transform (SIFT) implementation
//! - **surf_features**: Speeded-Up Robust Features (SURF) implementation
//! - **wavelet_features**: Wavelet-based texture analysis and feature extraction
//! - **shape_descriptors**: Geometric moments and shape property extraction
//! - **fractal_analysis**: Fractal dimension computation using multiple methods
//! - **zernike_moments**: Zernike moments for rotation-invariant shape analysis
//! - **image_utils**: Utility functions and fallback implementations
//!
//! # Features
//!
//! ## Traditional Computer Vision
//! - SIFT (Scale-Invariant Feature Transform) keypoint detection and description
//! - SURF (Speeded-Up Robust Features) with integral image optimization
//! - Image patch extraction with configurable sampling strategies
//!
//! ## Advanced Mathematical Analysis
//! - Wavelet-based texture analysis (Haar, Daubechies, Biorthogonal, Coiflets)
//! - Fractal dimension estimation (Box Counting, Differential Box Counting, Blanket Method)
//! - Zernike moments for rotation-invariant shape descriptors
//! - Geometric moments and Hu moments for shape analysis
//!
//! ## High-Performance Computing
//! - SIMD-accelerated operations with fallback implementations
//! - Parallel processing support for large-scale feature extraction
//! - Memory-efficient algorithms for processing large images
//!
//! # Examples
//!
//! ## Basic Patch Extraction
//!
//! ```rust
//! use sklears_feature_extraction::image::extract_patches_2d;
//! use scirs2_core::ndarray::Array2;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Create a sample image
//! let image = Array2::from_shape_vec((10, 10), (0..100).map(|x| x as f64).collect())?;
//!
//! // Extract 3x3 patches
//! let patches = extract_patches_2d(&image.view(), (3, 3), Some(5), None)?;
//! println!("Extracted {} patches", patches.shape()[0]);
//! # Ok(())
//! # }
//! ```
//!
//! ## SIFT Feature Extraction
//!
//! ```rust
//! use sklears_feature_extraction::image::SIFTExtractor;
//! use scirs2_core::ndarray::Array2;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Create SIFT extractor
//! let sift = SIFTExtractor::new()
//!     .n_octaves(4)
//!     .n_scales_per_octave(3)
//!     .sigma(1.6);
//!
//! // Process an image
//! let image = Array2::from_shape_vec((64, 64), (0..4096).map(|x| (x as f64).sin()).collect())?;
//! let keypoints = sift.detect_keypoints(&image.view())?;
//! println!("Detected {} SIFT keypoints", keypoints.len());
//! # Ok(())
//! # }
//! ```
//!
//! ## Wavelet Feature Analysis
//!
//! ```rust
//! use sklears_feature_extraction::image::{WaveletFeatureExtractor, WaveletType};
//! use scirs2_core::ndarray::Array2;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Create wavelet feature extractor
//! let wavelet = WaveletFeatureExtractor::new()
//!     .wavelet_type(WaveletType::Daubechies(4))
//!     .decomposition_levels(3)
//!     .enable_all_features()
//!     .normalize_features(true);
//!
//! // Extract features from an image
//! let image = Array2::from_shape_vec((32, 32), (0..1024).map(|x| x as f64).collect())?;
//! let features = wavelet.extract_features(&image.view())?;
//! println!("Extracted {} wavelet features", features.len());
//! # Ok(())
//! # }
//! ```
//!
//! ## Fractal Dimension Analysis
//!
//! ```rust
//! use sklears_feature_extraction::image::{FractalDimensionExtractor, FractalMethod};
//! use scirs2_core::ndarray::Array2;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Create fractal dimension extractor
//! let fractal = FractalDimensionExtractor::new()
//!     .method(FractalMethod::BoxCounting)
//!     .box_sizes(vec![2, 4, 8, 16, 32])
//!     .min_regression_points(4);
//!
//! // Compute fractal dimension
//! let image = Array2::from_shape_vec((64, 64), (0..4096).map(|x| x as f64).collect())?;
//! let dimension = fractal.compute_fractal_dimension(&image.view())?;
//! println!("Fractal dimension: {:.3}", dimension);
//! # Ok(())
//! # }
//! ```

// Sub-module declarations
pub mod fractal_analysis;
pub mod image_utils;
pub mod patch_extraction;
pub mod shape_descriptors;
pub mod sift_features;
pub mod simd_accelerated;
pub mod surf_features;
pub mod wavelet_features;
pub mod zernike_moments;

// Import core types and dependencies
use crate::*;
use scirs2_core::ndarray::{Array2, Array3, ArrayView2, ArrayView3};

// Re-export main structures and types
pub use fractal_analysis::{FractalDimensionExtractor, FractalMethod};
pub use image_utils::{
    compute_entropy_fallback, compute_image_statistics, compute_kurtosis_fallback,
    compute_skewness_fallback, extract_patches_fallback, normalize_image, rgb_to_grayscale,
    simd_image,
};
pub use patch_extraction::{PatchExtractor, SamplingStrategy};
pub use shape_descriptors::ShapeDescriptorExtractor;
pub use sift_features::{SIFTExtractor, SIFTKeypoint};
pub use simd_accelerated::*;
pub use surf_features::{SURFExtractor, SURFKeypoint};
pub use wavelet_features::{FeatureConfig, WaveletFeatureExtractor, WaveletType};
pub use zernike_moments::ZernikeMomentsExtractor;

/// Extract patches from a 2D image
///
/// Extract patches from a 2D image. This is useful for texture analysis,
/// local feature extraction, and preprocessing for machine learning algorithms.
///
/// # Arguments
///
/// * `image` - Input 2D image
/// * `patch_size` - Size of the patches to extract (height, width)
/// * `max_patches` - Maximum number of patches to extract
/// * `random_state` - Random seed for patch sampling
///
/// # Returns
///
/// 3D array of patches with shape (n_patches, patch_height, patch_width)
///
/// # Examples
///
/// ```rust
/// use sklears_feature_extraction::image::extract_patches_2d;
/// use scirs2_core::ndarray::Array2;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let image = Array2::from_shape_vec((10, 10), (0..100).map(|x| x as f64).collect())?;
/// let patches = extract_patches_2d(&image.view(), (3, 3), Some(5), None)?;
/// assert_eq!(patches.shape()[1..], [3, 3]); // Patch dimensions
/// # Ok(())
/// # }
/// ```
pub fn extract_patches_2d(
    image: &ArrayView2<Float>,
    patch_size: (usize, usize),
    max_patches: Option<usize>,
    random_state: Option<u64>,
) -> SklResult<Array3<Float>> {
    let extractor = PatchExtractor::new()
        .patch_size(patch_size)
        .max_patches(max_patches)
        .random_state(random_state);

    extractor.extract(image)
}

/// Reconstruct image from patches
///
/// Reconstruct a 2D image from extracted patches by averaging overlapping regions.
/// This function assumes patches were extracted in a regular grid pattern.
///
/// # Arguments
///
/// * `patches` - Array of patches with shape (n_patches, patch_height, patch_width)
/// * `image_size` - Size of the original image (height, width)
///
/// # Returns
///
/// Reconstructed 2D image
///
/// # Examples
///
/// ```rust
/// use sklears_feature_extraction::image::{extract_patches_2d, reconstruct_from_patches_2d};
/// use scirs2_core::ndarray::Array2;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let image = Array2::from_shape_vec((6, 6), (0..36).map(|x| x as f64).collect())?;
/// let patches = extract_patches_2d(&image.view(), (3, 3), None, None)?;
/// let reconstructed = reconstruct_from_patches_2d(&patches.view(), (6, 6))?;
/// # Ok(())
/// # }
/// ```
pub fn reconstruct_from_patches_2d(
    patches: &ArrayView3<Float>,
    image_size: (usize, usize),
) -> SklResult<Array2<Float>> {
    // Use the standalone function instead of a method
    crate::image::patch_extraction::reconstruct_from_patches_2d(patches, image_size)
}

/// Comprehensive image feature extraction pipeline
///
/// This structure provides a unified interface for extracting multiple types of features
/// from images using different algorithms.
///
/// # Examples
///
/// ```rust
/// use sklears_feature_extraction::image::ImageFeaturePipeline;
/// use scirs2_core::ndarray::Array2;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let pipeline = ImageFeaturePipeline::builder()
///     .enable_sift(true)
///     .enable_wavelet(true)
///     .enable_shape_descriptors(true)
///     .build();
///
/// let image = Array2::from_shape_vec((64, 64), (0..4096).map(|x| x as f64).collect())?;
/// let features = pipeline.extract_all_features(&image.view())?;
/// println!("Extracted {} total features", features.len());
/// # Ok(())
/// # }
/// ```
pub struct ImageFeaturePipeline {
    /// sift_extractor
    pub sift_extractor: Option<SIFTExtractor>,
    /// surf_extractor
    pub surf_extractor: Option<SURFExtractor>,
    /// wavelet_extractor
    pub wavelet_extractor: Option<WaveletFeatureExtractor>,
    /// shape_extractor
    pub shape_extractor: Option<ShapeDescriptorExtractor>,
    /// fractal_extractor
    pub fractal_extractor: Option<FractalDimensionExtractor>,
    /// zernike_extractor
    pub zernike_extractor: Option<ZernikeMomentsExtractor>,
}

impl ImageFeaturePipeline {
    /// Create a new builder for the image feature pipeline
    pub fn builder() -> ImageFeaturePipelineBuilder {
        ImageFeaturePipelineBuilder::default()
    }

    /// Extract all enabled features from an image
    ///
    /// # Arguments
    /// * `image` - Input image
    ///
    /// # Returns
    /// Concatenated feature vector from all enabled extractors
    pub fn extract_all_features(&self, image: &ArrayView2<Float>) -> SklResult<Array1<Float>> {
        let mut all_features = Vec::new();

        // Extract SIFT features
        if let Some(ref extractor) = self.sift_extractor {
            let keypoints = extractor.detect_keypoints(image)?;
            if !keypoints.is_empty() {
                let descriptors = extractor.extract_descriptors(image, &keypoints)?;
                for row in descriptors.rows() {
                    all_features.extend_from_slice(row.as_slice().unwrap());
                }
            }
        }

        // Extract SURF features
        if let Some(ref extractor) = self.surf_extractor {
            let keypoints = extractor.detect_keypoints(image)?;
            if !keypoints.is_empty() {
                let descriptors = extractor.extract_descriptors(image, &keypoints)?;
                for row in descriptors.rows() {
                    all_features.extend_from_slice(row.as_slice().unwrap());
                }
            }
        }

        // Extract wavelet features
        if let Some(ref extractor) = self.wavelet_extractor {
            let features = extractor.extract_features(image)?;
            all_features.extend_from_slice(features.as_slice().unwrap());
        }

        // Extract shape descriptors
        if let Some(ref extractor) = self.shape_extractor {
            let features = extractor.extract_features(image)?;
            all_features.extend_from_slice(features.as_slice().unwrap());
        }

        // Extract fractal dimension
        if let Some(ref extractor) = self.fractal_extractor {
            let dimension = extractor.compute_fractal_dimension(image)?;
            all_features.push(dimension);
        }

        // Extract Zernike moments
        if let Some(ref extractor) = self.zernike_extractor {
            let moments = extractor.extract_features(image)?;
            all_features.extend_from_slice(moments.as_slice().unwrap());
        }

        if all_features.is_empty() {
            return Err(SklearsError::InvalidInput(
                "No feature extractors enabled in pipeline".to_string(),
            ));
        }

        Ok(Array1::from_vec(all_features))
    }
}

/// Builder for ImageFeaturePipeline
#[derive(Default)]
pub struct ImageFeaturePipelineBuilder {
    enable_sift: bool,
    enable_surf: bool,
    enable_wavelet: bool,
    enable_shape_descriptors: bool,
    enable_fractal: bool,
    enable_zernike: bool,
}

impl ImageFeaturePipelineBuilder {
    /// Enable SIFT feature extraction
    pub fn enable_sift(mut self, enable: bool) -> Self {
        self.enable_sift = enable;
        self
    }

    /// Enable SURF feature extraction
    pub fn enable_surf(mut self, enable: bool) -> Self {
        self.enable_surf = enable;
        self
    }

    /// Enable wavelet feature extraction
    pub fn enable_wavelet(mut self, enable: bool) -> Self {
        self.enable_wavelet = enable;
        self
    }

    /// Enable shape descriptor extraction
    pub fn enable_shape_descriptors(mut self, enable: bool) -> Self {
        self.enable_shape_descriptors = enable;
        self
    }

    /// Enable fractal dimension extraction
    pub fn enable_fractal(mut self, enable: bool) -> Self {
        self.enable_fractal = enable;
        self
    }

    /// Enable Zernike moments extraction
    pub fn enable_zernike(mut self, enable: bool) -> Self {
        self.enable_zernike = enable;
        self
    }

    /// Build the image feature pipeline
    pub fn build(self) -> ImageFeaturePipeline {
        ImageFeaturePipeline {
            sift_extractor: if self.enable_sift {
                Some(SIFTExtractor::default())
            } else {
                None
            },
            surf_extractor: if self.enable_surf {
                Some(SURFExtractor::default())
            } else {
                None
            },
            wavelet_extractor: if self.enable_wavelet {
                Some(WaveletFeatureExtractor::default())
            } else {
                None
            },
            shape_extractor: if self.enable_shape_descriptors {
                Some(ShapeDescriptorExtractor::default())
            } else {
                None
            },
            fractal_extractor: if self.enable_fractal {
                Some(FractalDimensionExtractor::default())
            } else {
                None
            },
            zernike_extractor: if self.enable_zernike {
                Some(ZernikeMomentsExtractor::default())
            } else {
                None
            },
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_extract_patches_2d() {
        let image = Array2::from_shape_vec((10, 10), (0..100).map(|x| x as f64).collect()).unwrap();
        let patches = extract_patches_2d(&image.view(), (3, 3), Some(5), None).unwrap();

        assert_eq!(patches.shape()[1], 3);
        assert_eq!(patches.shape()[2], 3);
        assert!(patches.shape()[0] <= 5);
    }

    #[test]
    fn test_reconstruct_from_patches_2d() {
        let image = Array2::from_shape_vec((6, 6), (0..36).map(|x| x as f64).collect()).unwrap();
        let patches = extract_patches_2d(&image.view(), (3, 3), None, None).unwrap();
        let reconstructed = reconstruct_from_patches_2d(&patches.view(), (6, 6)).unwrap();

        assert_eq!(reconstructed.shape(), &[6, 6]);
    }

    #[test]
    fn test_image_feature_pipeline() {
        let pipeline = ImageFeaturePipeline::builder()
            .enable_shape_descriptors(true)
            .enable_fractal(true)
            .build();

        let image =
            Array2::from_shape_vec((32, 32), (0..1024).map(|x| x as f64).collect()).unwrap();
        let features = pipeline.extract_all_features(&image.view()).unwrap();

        assert!(features.len() > 0);
    }

    #[test]
    fn test_empty_pipeline() {
        let pipeline = ImageFeaturePipeline::builder().build();
        let image = Array2::from_shape_vec((10, 10), (0..100).map(|x| x as f64).collect()).unwrap();

        let result = pipeline.extract_all_features(&image.view());
        assert!(result.is_err());
    }
}
